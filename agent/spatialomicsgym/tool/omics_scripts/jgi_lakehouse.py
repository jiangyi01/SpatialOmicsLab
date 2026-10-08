"""JGI Lakehouse Dremio REST API client for querying GOLD/IMG/Mycocosm/Phytozome databases.

Adapted from omics-skills (https://github.com/fmschulz/omics-skills).
Requires LBNL network access and DREMIO_PAT environment variable.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

DEFAULT_JOB_TIMEOUT = 300
# (connect, read) timeout for every HTTP call — without it a stalled socket hangs the agent forever
# (the wait_for_job job-timeout only fires BETWEEN polls, not during a blocked request).
_HTTP_TIMEOUT = (10, 60)


def _get_config() -> tuple[str, str]:
    host = os.getenv("DREMIO_HOST", "lakehouse-1.jgi.lbl.gov")
    port = os.getenv("DREMIO_PORT", "9047")
    return host, port


#: Every request carries ``Authorization: Bearer <DREMIO_PAT>``, a long-lived personal token. The URL
#: was always ``http://``, so the token crossed the network in cleartext and ``verify=False`` guarded
#: nothing (hunt 2026-09-30, uT6-literature-21). HTTPS with certificate checks is the default now;
#: plaintext needs the operator to choose it twice, by scheme and by an explicit allowance.
_SCHEME_ENV = "DREMIO_SCHEME"
_PLAINTEXT_ENV = "DREMIO_ALLOW_PLAINTEXT_TOKEN"


def _get_token() -> str:
    token = os.getenv("DREMIO_PAT")
    if not token:
        raise ValueError("DREMIO_PAT environment variable not set")
    return token


def _get_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Content-Type": "application/json",
    }


def _base_url() -> str:
    host, port = _get_config()
    scheme = (os.getenv(_SCHEME_ENV) or "https").strip().lower()
    if scheme not in ("https", "http"):
        raise ValueError(f"{_SCHEME_ENV} must be 'https' or 'http', not {scheme!r}")
    if scheme == "http" and os.getenv(_PLAINTEXT_ENV, "").strip().lower() not in ("1", "true", "yes", "on"):
        raise ValueError(
            f"{_SCHEME_ENV}=http would send DREMIO_PAT unencrypted to {host}:{port}. Use https, or set "
            f"{_PLAINTEXT_ENV}=1 to accept sending the token in cleartext."
        )
    return f"{scheme}://{host}:{port}/api/v3"


def _send(send: Any, url: str, **kwargs: Any) -> requests.Response:
    """One Dremio call through ``send`` (``requests.get`` or ``requests.post``), with the token attached.

    Dremio serves plain HTTP on 9047 unless TLS is enabled, so against such a server the https
    default fails in the handshake -- and the bare ``SSLError`` ("wrong version number") named
    neither setting the operator can change (hunt 2026-09-30, uT6-literature-21 review).
    """
    try:
        return send(url, headers=_get_headers(), timeout=_HTTP_TIMEOUT, **kwargs)
    except requests.exceptions.SSLError as exc:
        host, port = _get_config()
        raise requests.exceptions.SSLError(
            f"TLS to {host}:{port} failed: {exc}. If this Dremio serves plain HTTP (its default on 9047), "
            f"set {_SCHEME_ENV}=http and {_PLAINTEXT_ENV}=1 to accept sending DREMIO_PAT in cleartext; if it "
            "serves HTTPS under a private CA, point REQUESTS_CA_BUNDLE at that CA."
        ) from exc


def execute_sql(sql: str, context: list[str] | None = None) -> dict[str, Any]:
    """Submit a SQL query to the Dremio REST API."""
    url = f"{_base_url()}/sql"
    payload: dict[str, Any] = {"sql": sql}
    if context:
        payload["context"] = context
    response = _send(requests.post, url, json=payload)
    response.raise_for_status()
    return response.json()


def wait_for_job(job_id: str, timeout: int = DEFAULT_JOB_TIMEOUT, poll_interval: float = 1.0) -> dict[str, Any]:
    """Poll job status until completion or timeout."""
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
        url = f"{_base_url()}/job/{job_id}"
        response = _send(requests.get, url)
        response.raise_for_status()
        status = response.json()
        job_state = status.get("jobState")
        if job_state == "COMPLETED":
            return status
        elif job_state in ("FAILED", "CANCELED", "CANCELLED"):
            error_msg = status.get("errorMessage", "Unknown error")
            raise RuntimeError(f"Job {job_state}: {error_msg}")
        time.sleep(poll_interval)


def get_job_results(job_id: str, offset: int = 0, limit: int = 100) -> dict[str, Any]:
    """Get results from a completed Dremio job."""
    url = f"{_base_url()}/job/{job_id}/results"
    params = {"offset": offset, "limit": min(limit, 500)}
    response = _send(requests.get, url, params=params)
    response.raise_for_status()
    return response.json()


def query(
    sql: str, context: list[str] | None = None, limit: int = 100, timeout: int = DEFAULT_JOB_TIMEOUT
) -> list[dict[str, Any]]:
    """Execute SQL and return results with job polling."""
    job_info = execute_sql(sql, context)
    job_id = job_info.get("id")
    if not job_id:
        raise ValueError(f"No job ID in response: {job_info}")
    wait_for_job(job_id, timeout=timeout)
    # get_job_results caps a single page at 500 rows; paginate so a caller asking for limit>500 isn't
    # silently truncated (show_schemas passes limit=2000).
    rows: list[dict[str, Any]] = []
    offset = 0
    while offset < limit:
        batch = get_job_results(job_id, offset=offset, limit=min(500, limit - offset))
        got = batch.get("rows", [])
        if not got:
            break
        rows.extend(got)
        offset += len(got)
    return rows[:limit]


def query_all(
    sql: str, context: list[str] | None = None, timeout: int = DEFAULT_JOB_TIMEOUT, batch_size: int = 500
) -> list[dict[str, Any]]:
    """Execute SQL and return ALL results with auto-pagination."""
    job_info = execute_sql(sql, context)
    job_id = job_info.get("id")
    if not job_id:
        raise ValueError(f"No job ID in response: {job_info}")
    status = wait_for_job(job_id, timeout=timeout)
    total_rows = status.get("rowCount", 0)
    all_rows: list[dict[str, Any]] = []
    offset = 0
    while offset < total_rows:
        results = get_job_results(job_id, offset=offset, limit=batch_size)
        rows = results.get("rows", [])
        if not rows:
            break
        all_rows.extend(rows)
        offset += len(rows)
    return all_rows


def list_catalogs() -> list[dict[str, Any]]:
    """List available Dremio catalogs/sources."""
    url = f"{_base_url()}/catalog"
    response = _send(requests.get, url)
    response.raise_for_status()
    return response.json().get("data", [])


def show_schemas(limit: int = 2000) -> list[str]:
    """List all available Dremio schemas."""
    results = query("SHOW SCHEMAS", limit=limit)
    return [row.get("SCHEMA_NAME") for row in results if row.get("SCHEMA_NAME")]
