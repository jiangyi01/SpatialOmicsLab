"""
Tier-2 LLM validation target — a real ``get_llm().invoke`` in the base env.

Run *inside* a provisioned env so it exercises the actual runtime stack:

    conda run -n <basic_env> python -m sog_install._llm_ping

Prints a single JSON line ``{"ok": bool, "model": str, "source": str,
"error": str}`` to stdout and exits 0 on success / 1 on failure. All heavy
imports are deferred into :func:`main` so merely importing this module in the
thin agent env (e.g. during test collection) never pulls in langchain.
"""

from __future__ import annotations

import json
import os
import sys


def _load_install_env() -> None:
    """Load the install's own ``.env`` so the keys the agent uses are visible here, never overriding.

    (hunt 2026-09-30, u35b-setup-state-17) A bare ``load_dotenv()`` walks up from THIS file: on a
    checkout that is always the repo ``.env``, even when the run isolated its config with
    ``SOG_SETUP_DOTENV`` or asked for none with ``SOG_SKIP_DOTENV``; off a checkout it never reaches the
    instance-root ``.env`` the wizard wrote and can pick up an unrelated ``~/.env``. Same file and same
    gate as ``agent/stcoscientist.py``'s import-time load: ``chat_cli.install_env_file()``."""
    if os.environ.get("SOG_SKIP_DOTENV"):
        return
    try:
        from dotenv import load_dotenv

        from spatialomicsgym.chat_cli import install_env_file

        path = install_env_file()
        if path is not None:
            load_dotenv(str(path), override=False)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    result = {"ok": False, "model": "", "source": "", "error": ""}
    try:
        _load_install_env()

        from spatialomicsgym.config import SpatialOmicsGymConfig
        from spatialomicsgym.llm import get_llm

        config = SpatialOmicsGymConfig()
        result["model"] = config.llm
        result["source"] = config.source or "(auto)"

        llm = get_llm(config=config)
        resp = llm.invoke("Reply with the single word: ok")
        text = getattr(resp, "content", str(resp))
        result["ok"] = bool(text)
        result["reply"] = (text or "")[:80] if isinstance(text, str) else str(text)[:80]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
