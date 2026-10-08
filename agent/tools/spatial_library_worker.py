#!/usr/bin/env python3
"""
Spatial dataset library search worker.

Searches the registry.json index for matching datasets.
Returns JSON with matching datasets and their h5ad paths.

Usage:
    python tools/spatial_library_worker.py --organ Brain
    python tools/spatial_library_worker.py --disease cancer --organism human
    python tools/spatial_library_worker.py --keyword glioblastoma --max-results 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_spatial_library_registry import SPECIES_MAP, h5ad_spatial_facts

# Default to the in-repo catalog snapshot (self-contained); override with SOG_SPATIAL_LIBRARY_REGISTRY.
# Falls back to the legacy external location if the in-repo snapshot is absent.
#
# Read the override the same way _local_root reads its sibling below: strip it, and treat a blank
# value as "not set". os.environ.get returns "" rather than the default when a variable is exported
# empty, so passing the default straight to it meant one `export SOG_SPATIAL_LIBRARY_REGISTRY=` --
# or an empty line in a .env -- silently replaced the whole catalog with nothing. The Python
# re-export of this search (spatialomicsgym/tool/transcriptomics_skills.py) already strips.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_INREPO_REGISTRY = os.path.join(_REPO_ROOT, "spatialomicsgym", "data", "spatial_library", "registry.json")
_REGISTRY_OVERRIDE = os.environ.get("SOG_SPATIAL_LIBRARY_REGISTRY", "").strip()
REGISTRY_PATH = _REGISTRY_OVERRIDE or (
    _INREPO_REGISTRY if os.path.exists(_INREPO_REGISTRY) else "/workspace/data/spatial_library/registry.json"
)

_registry_cache = None

#: Every spelling of an organism, from either the common name or the binomial (the builder's
#: SPECIES_MAP). The snapshot records "Homo sapiens" in both organism and species, so the first
#: example the tool gives, organism="human", matched none of its 63 records (hunt 2026-09-30,
#: u30-uncovered-mcp-2).
_ORGANISM_SPELLINGS: dict[str, tuple[str, ...]] = {}
for _common, _binomial in SPECIES_MAP.items():
    _spellings = (_common, _binomial.lower(), f"{_binomial[0]}. {_binomial.split()[-1]}".lower())
    for _s in _spellings:
        _ORGANISM_SPELLINGS[_s] = _spellings

#: Disease words that name a class rather than a label. The registry labels its tumours by type
#: ("Glioblastoma", "Invasive Ductal Carcinoma"), so disease="cancer" -- the know_how's own example --
#: found none of the four brain tumours and 3 of the 9 breast ones (hunt 2026-09-30,
#: u30-uncovered-mcp-3). Such a word matches a diseased record whose label, description or
#: sample_id names any tumour type below.
_CANCER_WORDS = frozenset(
    {"cancer", "cancers", "tumor", "tumors", "tumour", "tumours", "malignant", "malignancy", "neoplasm"}
)
_CANCER_LABELS = (
    "cancer",
    "carcinoma",
    "glioblastoma",
    "glioma",
    "melanoma",
    "sarcoma",
    "lymphoma",
    "leukemia",
    "leukaemia",
    "myeloma",
    "blastoma",
    "tumor",
    "tumour",
    "neoplasm",
    "metasta",
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[\s,;/]+", (text or "").lower()) if t]


def _record_text(r: dict, keys: tuple[str, ...]) -> str:
    """The record's fields as one lower-case string; sample_ids also read with ``_`` as a space."""
    parts = [str(r.get(k) or "") for k in keys]
    if "sample_id" in keys:
        parts.append(str(r.get("sample_id") or "").replace("_", " "))
    if "organism" in keys:
        parts.extend(_ORGANISM_SPELLINGS.get(str(r.get("organism") or "").lower(), ()))
    return " ".join(parts).lower()


def _organism_matches(r: dict, organism: str) -> bool:
    wanted = _ORGANISM_SPELLINGS.get(organism.lower().strip(), (organism.lower().strip(),))
    held = f"{r.get('organism', '')} {r.get('species', '')}".lower()
    return any(w in held for w in wanted)


def _disease_matches(r: dict, disease: str) -> bool:
    """Every word of ``disease`` is in the label, description or sample_id; a class word such as
    "cancer" matches a diseased record that names a tumour type."""
    text = _record_text(r, ("disease", "description", "sample_id"))
    for token in _tokens(disease):
        if token in _CANCER_WORDS:
            if r.get("is_healthy") is not False or not any(label in text for label in _CANCER_LABELS):
                return False
        elif token not in text:
            return False
    return True


def _local_root() -> str:
    """Where this machine keeps its copy of the dataset library, if anywhere.

    The shipped registry is a *catalog*: every ``spatial_dir`` is an absolute path on the
    machine that built it. Anyone else who downloads the library puts it somewhere else, so
    they point at it once with SOG_SPATIAL_LIBRARY_ROOT instead of rewriting 63 entries.
    """
    return os.environ.get("SOG_SPATIAL_LIBRARY_ROOT", "").strip()


def _relocate(path: str, root: str) -> str:
    """Find a catalogued path under the local library root, by matching trailing components.

    Tries the most specific suffix first, so ``.../human/Brain/Sample`` prefers
    ``<root>/human/Brain/Sample`` over a bare ``<root>/Sample`` that happens to exist.
    Returns the original path when nothing matches — a wrong guess would be worse than an
    honest "not here".
    """
    if not path or not root or os.path.exists(path):
        return path
    parts = [p for p in path.split(os.sep) if p]
    for k in range(len(parts), 0, -1):
        candidate = os.path.join(root, *parts[-k:])
        if os.path.exists(candidate):
            return candidate
    return path


def _find_h5ad(spatial_dir: str) -> str | None:
    """Locate the .h5ad inside a present Space Ranger directory (the catalog records none)."""
    if not spatial_dir or not os.path.isdir(spatial_dir):
        return None
    import glob

    found = sorted(glob.glob(os.path.join(spatial_dir, "*.h5ad"))) or sorted(
        glob.glob(os.path.join(spatial_dir, "**", "*.h5ad"), recursive=True)
    )
    return found[0] if found else None


def load_registry(registry_path: str = REGISTRY_PATH) -> list[dict]:
    """Load and cache the registry."""
    global _registry_cache
    if _registry_cache is None:
        if not os.path.exists(registry_path):
            return []
        try:
            with open(registry_path) as f:
                _registry_cache = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: Failed to load registry {registry_path}: {e}", file=sys.stderr)
            return []
    return _registry_cache


def search(
    organism: str = "",
    organ: str = "",
    disease: str = "",
    technology: str = "",
    keyword: str = "",
    is_healthy: bool | None = None,
    max_results: int = 5,
    registry_path: str = REGISTRY_PATH,
) -> dict:
    """Search the spatial dataset library."""
    registry = load_registry(registry_path)

    if not registry:
        return {
            "status": "ok",
            "n_results": 0,
            "n_total": 0,
            "results": [],
            "message": "Registry is empty or not found.",
        }

    results = list(registry)

    # Exact filters (case-insensitive)
    if organism:
        results = [r for r in results if _organism_matches(r, organism)]

    if organ:
        organ_lower = organ.lower()
        results = [
            r
            for r in results
            if organ_lower in r.get("organ", "").lower() or organ_lower in r.get("anatomical_entity", "").lower()
        ]

    if technology:
        tech_lower = technology.lower()
        results = [r for r in results if tech_lower in r.get("technology", "").lower()]

    if is_healthy is not None:
        results = [r for r in results if r.get("is_healthy") == is_healthy]

    # Fuzzy filters (word match)
    if disease:
        results = [r for r in results if _disease_matches(r, disease)]

    if keyword:
        # Every word must be somewhere in the record, not the whole phrase in one place: as one
        # contiguous substring, "human glioblastoma" found 1 of the 4 glioblastomas (hunt 2026-09-30,
        # u30-uncovered-mcp-8). Records whose organ or disease carries more of the words rank first.
        words = _tokens(keyword)
        fields = ("sample_id", "organ", "disease", "technology", "description", "organism", "anatomical_entity")
        scored = []
        for r in results:
            searchable = _record_text(r, fields)
            if all(w in searchable for w in words):
                focus = f"{r.get('organ', '')} {r.get('disease', '')}".lower()
                scored.append((sum(w in focus for w in words), r))
        scored.sort(key=lambda x: -x[0])
        results = [r for _, r in scored]

    # Limit results
    total_matches = len(results)
    results = results[:max_results]

    # Clean output (remove internal fields)
    root = _local_root()
    clean_results = []
    for r in results:
        # The registry is a catalog of another machine's paths. Say plainly whether each hit
        # is readable *here*: without that, an absent dataset looks identical to a present one
        # and the agent reads the resulting open() failure as "the tool layer is broken".
        spatial_dir = _relocate(r.get("spatial_dir") or "", root) or None
        h5ad_path = _relocate(r.get("h5ad_path") or "", root) or None
        if not h5ad_path:
            h5ad_path = _find_h5ad(spatial_dir or "")
        available = bool((h5ad_path and os.path.exists(h5ad_path)) or (spatial_dir and os.path.isdir(spatial_dir)))
        # Read off the file when the catalog does not say: the snapshot carries no n_spots/n_genes,
        # so both were null on every hit, and nothing said a hit had no positions at all
        # (u30-uncovered-mcp-4, -9).
        facts = h5ad_spatial_facts(h5ad_path) if h5ad_path and os.path.isfile(h5ad_path) else {}
        has_coords = r.get("has_spatial_coordinates")
        if has_coords is None:
            has_coords = facts.get("has_spatial_coordinates")
        clean_results.append(
            {
                # The one field of fourteen that was read by subscript. The registry is hand-writable
                # through SOG_SPATIAL_LIBRARY_REGISTRY, so an entry without this key raised KeyError
                # out of main() -- and nothing in this file catches, so the MCP server got a bare
                # traceback on stderr and *nothing* on stdout: a search that answers becomes a tool
                # that broke. Report the absence as null, the way the Python re-export already does
                # and the way `is_healthy` two lines down already does. It is disclosed below rather
                # than filled in: the hit may still be perfectly usable through spatial_dir, but
                # inventing a plausible identifier would let the agent quote back a sample name that
                # is not in the registry.
                "sample_id": r.get("sample_id"),
                "organism": r.get("organism", ""),
                "species": r.get("species", ""),
                "organ": r.get("organ", ""),
                "disease": r.get("disease", ""),
                "is_healthy": r.get("is_healthy"),
                "technology": r.get("technology", ""),
                "preservation": r.get("preservation", ""),
                "n_spots": r.get("n_spots") if r.get("n_spots") is not None else facts.get("n_spots"),
                "n_genes": r.get("n_genes") if r.get("n_genes") is not None else facts.get("n_genes"),
                "h5ad_path": h5ad_path,
                "spatial_dir": spatial_dir,
                "available": available,
                # None: not known here (the file is absent or unreadable).
                "has_spatial_coordinates": has_coords,
                "available_for_spatial": bool(available and has_coords is not False),
                "description": r.get("description", ""),
            }
        )

    n_available = sum(1 for r in clean_results if r["available"])
    out = {
        "status": "ok",
        "n_results": len(clean_results),
        "n_available": n_available,
        "n_total": total_matches,
        "query": {
            "organism": organism,
            "organ": organ,
            "disease": disease,
            "technology": technology,
            "keyword": keyword,
            "is_healthy": is_healthy,
        },
        "results": clean_results,
    }
    # Both advisories share one channel, so a hit that is neither present nor named says both.
    notes = []
    if clean_results and n_available == 0:
        notes.append(
            f"{len(clean_results)} dataset(s) match, but none are present on this machine — the "
            "registry is a catalog of dataset metadata, not the data itself. If you have "
            "downloaded the library, set SOG_SPATIAL_LIBRARY_ROOT to its directory; otherwise ask "
            "the user for a data path instead of trying to open the paths above."
        )
    no_coords = [r["sample_id"] or r["spatial_dir"] for r in clean_results if r["has_spatial_coordinates"] is False]
    if no_coords:
        notes.append(
            f"{', '.join(map(str, no_coords))} carry no spatial coordinates (no obsm['spatial'] and no position "
            "columns): they are expression atlases, not spatial datasets, and spatial tools cannot run on them "
            "(available_for_spatial is false). Pick another dataset for a spatial analysis."
        )
    if not clean_results and (organism or disease or organ):
        # Name what the catalog does hold, so "0 results" is not read as "the library has no such data".
        held = {
            "organism": sorted({str(r.get("organism")) for r in registry if r.get("organism")}),
            "organ": sorted({str(r.get("organ")) for r in registry if r.get("organ")}),
            "disease": sorted({str(r.get("disease")) for r in registry if r.get("disease")}),
        }
        asked = [k for k, v in (("organism", organism), ("organ", organ), ("disease", disease)) if v]
        notes.append(
            "No record matches. The library holds "
            + "; ".join(f"{k}: {', '.join(held[k])}" for k in asked)
            + ". Loosen or respell the filter rather than concluding the library has no such data."
        )
    n_unnamed = sum(1 for r in clean_results if not r["sample_id"])
    if n_unnamed:
        notes.append(
            f"{n_unnamed} of the returned record(s) carry no sample_id — the registry entry is "
            "missing that field, so there is no identifier to quote back for them. Refer to those "
            "by their spatial_dir instead of naming a sample."
        )
    if notes:
        out["message"] = " ".join(notes)
    return out


def main():
    parser = argparse.ArgumentParser(description="Search spatial dataset library")
    parser.add_argument("--organism", default="", help="Filter by organism")
    parser.add_argument("--organ", default="", help="Filter by organ")
    parser.add_argument("--disease", default="", help="Filter by disease")
    parser.add_argument("--technology", default="", help="Filter by technology")
    parser.add_argument("--keyword", default="", help="Free-text keyword search")
    parser.add_argument(
        "--is-healthy", default=None, type=lambda x: x.lower() == "true", help="Filter healthy/diseased"
    )
    parser.add_argument("--max-results", type=int, default=5, help="Max results")
    parser.add_argument("--registry-path", default=REGISTRY_PATH, help="Path to registry.json")
    args = parser.parse_args()

    result = search(
        organism=args.organism,
        organ=args.organ,
        disease=args.disease,
        technology=args.technology,
        keyword=args.keyword,
        is_healthy=args.is_healthy,
        max_results=args.max_results,
        registry_path=args.registry_path,
    )
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
