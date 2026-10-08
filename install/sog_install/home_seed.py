"""
Seed the writable instance root from the wheel's read-only platform payload.

On a pip-only install the platform trees (``tools/``, the canonical MCP config, the spec
corpus, the ``tools_user`` helper twins) ship frozen inside ``spatialomicsgym/_platform/``. Everything that *writes* -- the wizard's state and backups,
``apply_to_canonical``, ``.env``, user-created tools -- routes through
``constants.repo_root()``, which off a checkout answers the writable instance root
(SOG_HOME, default ``~/.spatialomicsgym``). This module makes that root real: it copies the
payload there laid out like the packaged copy (``tools/``, ``tools_user/``, ``MCP_server/`` at the
top, the recipes under ``setup/``), and every path helper joins through
``platform_root.platform_dir`` / ``recipes_dir``, which answer for that layout as well as a
checkout's ``agent/`` + ``install/recipes/``.

Policy (the whole of it):

* **Checkouts are never touched** -- ``running_from_checkout()`` short-circuits first, so
  in-repo behavior is byte-identical and a developer's ``~/.spatialomicsgym`` is never
  created as a side effect of running tests or the wizard.
* **Only the payload is ever written.** Files outside :data:`platform_root.PLATFORM_PAYLOAD`
  (user tools, ``.env``, ``.sog_setup/`` state, backups) are invisible to the seeder.
* **The canonical config is seeded only if absent.** Once ``sog-setup``'s finalize has
  rewritten the home's ``MCP_server/mcp_config.yaml`` for this machine it is verified wiring; a version
  refresh must not roll it back to the build-frozen blanket copy.
* **Idempotent via the manifest.** The seeded root carries its own
  ``PLATFORM_MANIFEST.json``; a matching version whose listed files are all present is a
  no-op, anything else refreshes the platform files in place.
* **A refresh also retires.** Files our own prior manifest seeded that the new payload no
  longer carries are deleted (never the canonical config) -- otherwise an upgrade that
  removes a worker leaves the old code running in every seeded home forever. Only paths
  from our own manifest are ever deleted; nothing else on the machine is reachable.
* **The packaged manifest is the statement of record.** Files are selected from its
  ``files`` list, not by re-globbing the packaged directory, so a stray file that ended up
  beside the payload is never promoted into a seeded home; an unreadable packaged manifest
  refuses the whole payload (``bad-source``).
* **Externally-managed roots are refused.** A root that already carries the platform marker
  but no seed manifest (an operator's own clone reached via ``SOG_PLATFORM_ROOT`` or cwd)
  was not built by us and is reported, not written to. A seed manifest that exists but does
  not parse is the opposite case -- our own torn write -- and is reclaimed and reseeded.

Stdlib + :mod:`spatialomicsgym.platform_root` only, like every setup module. All writes are
``.partial.<pid>`` + ``os.replace`` -- pid-unique so two concurrent runs (a stuck wizard plus
a retry in another terminal) never share a partial inode and every replaced file is complete.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING, Any

from spatialomicsgym import platform_root

if TYPE_CHECKING:
    from pathlib import Path

#: The one payload file with keep-if-present semantics (finalize owns it once it exists). A packaged
#: (flat) path, like every path the packaged manifest lists.
CANONICAL_CONFIG_REL = "MCP_server/mcp_config.yaml"

#: Version stamped when a root is claimed before its first successful seed. Deliberately NOT
#: the empty string an unreadable manifest degrades to: with "" for both, a corrupt packaged
#: manifest plus a crash mid-seed would compare equal and report a half-copied tree as
#: ``current`` forever.
SEEDING_SENTINEL = "__seeding__"


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _clean_rel_paths(value: Any) -> list[str]:
    """Manifest ``files`` entries that are safe to join under a root: relative, no ``..``."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for rel in value:
        if not isinstance(rel, str) or not rel or rel.startswith(("/", "\\")):
            continue
        if any(part in ("", "..") for part in rel.replace("\\", "/").split("/")):
            continue
        out.append(rel)
    return out


def _partial_name(target: Path) -> Path:
    # pid-unique: concurrent seeders must never truncate/replace each other's in-flight
    # partial, so every os.replace below installs a complete file (last writer wins whole).
    return target.with_name(f"{target.name}.partial.{os.getpid()}")


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_name(target)
    try:
        shutil.copyfile(source, partial)
        os.replace(partial, target)
    except BaseException:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_manifest(path: Path, doc: dict[str, Any]) -> None:
    partial = _partial_name(path)
    try:
        partial.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(partial, path)
    except BaseException:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def ensure_seeded() -> dict[str, Any]:
    """Bring the instance root up to this install's platform payload. Returns a report dict.

    ``status`` is one of: ``checkout`` (in-repo run, nothing done), ``no-source`` (no
    ``_platform`` payload in this install -- also nothing to run, and the resolvers'
    ``describe_search`` says so), ``bad-source`` (the payload is present but its manifest is
    unreadable -- a torn install; refused, reinstall the package), ``external`` (the root
    carries the marker but not our manifest -- refused), ``current`` (manifest version
    matches and every listed file is present -- no-op), ``seeded`` / ``refreshed`` (files
    written; ``written``/``kept``/``removed`` count the copies made, the canonical config
    left alone, and the retired files deleted). Never raises for the caller to babysit: the
    wizard wraps this in a WARN-only hook regardless.
    """
    if platform_root.running_from_checkout():
        return {"status": "checkout"}
    source = platform_root.packaged_platform_dir()
    if source is None:
        return {"status": "no-source"}

    manifest = _read_manifest(source / platform_root.PLATFORM_MANIFEST_NAME)
    if manifest is None:
        # ``packaged_platform_dir`` trusts on the manifest's presence; one that exists but
        # does not parse means the payload itself cannot be trusted. Refuse to seed from it
        # rather than guessing at its contents (and never let its degraded "" version
        # compare equal to a claimed root's sentinel).
        return {"status": "bad-source", "source": str(source)}
    version = str(manifest.get("version", ""))
    root = platform_root.instance_root()
    seed_manifest_path = root / platform_root.PLATFORM_MANIFEST_NAME
    prior = _read_manifest(seed_manifest_path)

    if prior is None:
        try:
            manifest_file_exists = seed_manifest_path.exists()
        except OSError:
            manifest_file_exists = False
        if manifest_file_exists:
            # A manifest file that exists but does not parse is OUR torn write -- the
            # external case below is a marked root with no manifest at all. Reclaim and
            # reseed instead of refusing the root forever.
            prior = {"version": SEEDING_SENTINEL, "files": []}
        elif platform_root.is_platform_root(root):
            return {"status": "external", "root": str(root)}

    prior_files = _clean_rel_paths((prior or {}).get("files"))
    if (
        prior is not None
        and str(prior.get("version", "")) == version
        and platform_root.is_platform_root(root)
        and prior_files
        and all((root / rel).is_file() for rel in prior_files)
    ):
        return {"status": "current", "root": str(root), "version": version}

    written = 0
    kept = 0
    removed = 0
    root.mkdir(parents=True, exist_ok=True)
    if prior is None:
        # Claim the root BEFORE copying: a crash mid-seed must read as "ours, unfinished"
        # (picked up and completed by the next run's version mismatch) rather than as an
        # externally-managed platform root -- which it would look like forever once the
        # canonical config and ``tools/`` had landed without a manifest.
        _write_manifest(seed_manifest_path, {"version": SEEDING_SENTINEL, "files": []})
    # The packaged manifest's ``files`` list is what the build actually shipped; re-globbing
    # the packaged directory would also sweep in any stray file that ended up beside the
    # payload (a stale build_lib, site-packages litter). The glob is only the fallback for a
    # manifest with no usable list.
    files = [rel for rel in _clean_rel_paths(manifest.get("files")) if (source / rel).is_file()]
    if not files:
        files = platform_root.payload_files(source)
    for rel in files:
        target = root / rel
        if rel == CANONICAL_CONFIG_REL and target.is_file():
            kept += 1
            continue
        _atomic_copy(source / rel, target)
        written += 1

    # Retire what the new payload dropped: files OUR prior manifest seeded that no current
    # payload file replaces. Without this an upgrade that removes a worker leaves the old
    # code in every seeded home forever (and the finalize-era config may still route to it).
    # The canonical config is never deleted -- finalize owns it once it exists.
    current_files = set(files)
    for rel in prior_files:
        if rel in current_files or rel == CANONICAL_CONFIG_REL:
            continue
        try:
            (root / rel).unlink()
            removed += 1
        except OSError:  # already gone, or undeletable -- either way not worth failing over
            pass

    _write_manifest(seed_manifest_path, {"version": version, "files": files})
    return {
        "status": "refreshed" if prior is not None else "seeded",
        "root": str(root),
        "version": version,
        "written": written,
        "kept": kept,
        "removed": removed,
    }
