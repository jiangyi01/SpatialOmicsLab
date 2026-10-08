#!/usr/bin/env python3
"""
Build the spatial dataset library registry.

Scans /workspace/data/spatial_library/ recursively for datasets.
Each dataset must have a *.h5ad file. Optionally has sample_metadata.json.
Outputs registry.json in the library root.

Usage:
    python tools/build_spatial_library_registry.py
    python tools/build_spatial_library_registry.py --library-path /custom/path
"""

import argparse
import json
import os
import sys

LIBRARY_PATH_DEFAULT = "/workspace/data/spatial_library"

SPECIES_MAP = {
    "human": "Homo sapiens",
    "mouse": "Mus musculus",
    "rat": "Rattus norvegicus",
    "zebrafish": "Danio rerio",
    "drosophila": "Drosophila melanogaster",
}


#: obs column pairs that hold spot positions when obsm carries none.
_OBS_COORDINATE_PAIRS = (("array_row", "array_col"), ("x", "y"), ("x_centroid", "y_centroid"), ("imagerow", "imagecol"))


def h5ad_spatial_facts(h5ad_path: str) -> dict:
    """``n_spots``, ``n_genes`` and ``has_spatial_coordinates`` of an h5ad, read from its HDF5 keys alone.

    The registry recorded neither a sample's size nor whether it has positions at all, so the search
    handed out a 193,108-cell single-cell lung atlas (obsm X_pca/X_umap only) as a dataset ready for
    any spatial tool, and n_spots/n_genes were null on every hit (hunt 2026-09-30,
    u30-uncovered-mcp-4, u30-uncovered-mcp-9). A fact the file does not answer stays None.
    """
    facts: dict = {"n_spots": None, "n_genes": None, "has_spatial_coordinates": None}
    try:
        import h5py

        def keys(node) -> set:
            if isinstance(node, h5py.Group):
                return {str(k).lower() for k in node.keys()}
            names = getattr(getattr(node, "dtype", None), "names", None)  # pre-0.7 structured obs
            return {str(k).lower() for k in names or ()}

        with h5py.File(h5ad_path, "r") as f:
            X = f.get("X")
            shape = X.attrs.get("shape") if isinstance(X, h5py.Group) else getattr(X, "shape", None)
            if shape is not None and len(shape) == 2:
                facts["n_spots"], facts["n_genes"] = int(shape[0]), int(shape[1])
            obs_keys = keys(f.get("obs"))
            facts["has_spatial_coordinates"] = any("spatial" in k for k in keys(f.get("obsm"))) or any(
                a in obs_keys and b in obs_keys for a, b in _OBS_COORDINATE_PAIRS
            )
    except Exception as exc:
        # Unknown, not false -- but said, so a builder run shows which file could not be read and why
        # (hunt 2026-09-30, rp-u30 registry minor).
        print(
            f"  [warn] {h5ad_path}: size and coordinates not read ({type(exc).__name__}: {exc}); recorded as unknown",
            file=sys.stderr,
        )
    return facts


def find_h5ad(sample_dir: str) -> str | None:
    """Find the primary h5ad file in a sample directory."""
    h5ad_files = [f for f in os.listdir(sample_dir) if f.endswith(".h5ad")]
    if not h5ad_files:
        return None
    # Prefer files that don't have suffixes like _repaired, _prepared
    primary = [f for f in h5ad_files if "_repaired" not in f and "_prepared" not in f]
    return os.path.join(sample_dir, primary[0] if primary else h5ad_files[0])


def extract_from_metadata(meta: dict, sample_dir: str, species_key: str) -> dict:
    """Extract registry fields from sample_metadata.json."""
    disease_status = meta.get("disease_status", {})
    if isinstance(disease_status, str):
        disease_status = {"label": disease_status, "is_healthy": disease_status.lower() == "healthy"}

    metrics = {}
    src = meta.get("source_metadata_original", {})
    if isinstance(src, dict):
        mi = src.get("metrics_info", {})
        if isinstance(mi, dict):
            metrics = mi.get("metrics_summary", {})

    return {
        "sample_id": meta.get("sample_id", os.path.basename(sample_dir)),
        "organism": meta.get("organism", SPECIES_MAP.get(species_key, species_key)),
        "species": species_key,
        "organ": meta.get("organ", meta.get("tissue_category", "")),
        "anatomical_entity": meta.get("anatomical_entity", ""),
        "disease": disease_status.get("label", "unknown") if isinstance(disease_status, dict) else str(disease_status),
        "is_healthy": disease_status.get("is_healthy") if isinstance(disease_status, dict) else None,
        "technology": meta.get("technology", ""),
        "preservation": meta.get("preservation_method", "unknown"),
        "modality": meta.get("modality", "gene_expression"),
        "n_spots": metrics.get("number_of_spots_under_tissue"),
        "n_genes": metrics.get("total_genes_detected"),
        "median_genes_per_spot": metrics.get("median_genes_per_spot"),
        "median_umi_per_spot": metrics.get("median_umi_counts_per_spot"),
    }


def extract_from_h5ad(h5ad_path: str, species_key: str, organ: str) -> dict:
    """Extract minimal registry fields from h5ad when no metadata JSON exists."""
    try:
        import anndata as ad

        adata = ad.read_h5ad(h5ad_path, backed="r")
        # The registry is built by walking a whole library directory, so this runs once per sample.
        # With the close inside the ``try`` body, one sample that raised on ``n_obs`` held its HDF5
        # lock for the rest of the walk. The inner ``finally`` releases it and then lets the same
        # exception reach the same handler, so a sample that cannot be read still reports None/None.
        try:
            n_spots = adata.n_obs
            n_genes = adata.n_vars
        finally:
            adata.file.close()
    except Exception:
        n_spots = None
        n_genes = None

    sample_id = os.path.splitext(os.path.basename(h5ad_path))[0]
    return {
        "sample_id": sample_id,
        "organism": SPECIES_MAP.get(species_key, species_key),
        "species": species_key,
        "organ": organ,
        "anatomical_entity": organ,
        "disease": "unknown",
        "is_healthy": None,
        "technology": "",
        "preservation": "unknown",
        "modality": "gene_expression",
        "n_spots": n_spots,
        "n_genes": n_genes,
        "median_genes_per_spot": None,
        "median_umi_per_spot": None,
    }


def build_description(entry: dict) -> str:
    """Build a human-readable description from registry fields."""
    parts = [
        entry.get("organism") or entry.get("species", "unknown"),
        entry.get("organ", ""),
        entry.get("disease", ""),
        entry.get("technology", ""),
    ]
    parts = [p for p in parts if p and p != "unknown"]
    desc = ", ".join(parts)
    if entry.get("n_spots") and entry.get("n_genes"):
        desc += f", {entry['n_spots']} spots x {entry['n_genes']} genes"
    return desc


def scan_library(library_path: str) -> list[dict]:
    """Scan the library directory and build registry entries."""
    registry = []

    for species_dir in sorted(os.listdir(library_path)):
        species_path = os.path.join(library_path, species_dir)
        if not os.path.isdir(species_path) or species_dir.startswith("."):
            continue

        species_key = species_dir.lower()

        for organ_dir in sorted(os.listdir(species_path)):
            organ_path = os.path.join(species_path, organ_dir)
            if not os.path.isdir(organ_path) or organ_dir.startswith("."):
                continue

            for sample_dir_name in sorted(os.listdir(organ_path)):
                sample_dir = os.path.join(organ_path, sample_dir_name)
                if not os.path.isdir(sample_dir) or sample_dir_name.startswith("."):
                    continue

                h5ad_path = find_h5ad(sample_dir)
                if not h5ad_path:
                    continue

                meta_path = os.path.join(sample_dir, "sample_metadata.json")
                if os.path.exists(meta_path):
                    try:
                        meta = json.load(open(meta_path))
                        entry = extract_from_metadata(meta, sample_dir, species_key)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"  WARN: Bad metadata in {meta_path}: {e}", file=sys.stderr)
                        entry = extract_from_h5ad(h5ad_path, species_key, organ_dir)
                else:
                    entry = extract_from_h5ad(h5ad_path, species_key, organ_dir)

                # Whether the sample has positions at all, and its size when the metadata gave none:
                # read off the file (u30-uncovered-mcp-4, -9).
                facts = h5ad_spatial_facts(h5ad_path)
                entry["has_spatial_coordinates"] = facts["has_spatial_coordinates"]
                for key in ("n_spots", "n_genes"):
                    if entry.get(key) is None:
                        entry[key] = facts[key]

                # Set paths to actual local locations (NOT from metadata)
                entry["h5ad_path"] = os.path.abspath(h5ad_path)
                spatial_dir = os.path.join(sample_dir, "spatial")
                entry["spatial_dir"] = os.path.abspath(spatial_dir) if os.path.isdir(spatial_dir) else None
                entry["description"] = build_description(entry)

                registry.append(entry)

    return registry


def main():
    parser = argparse.ArgumentParser(description="Build spatial dataset library registry")
    parser.add_argument("--library-path", default=LIBRARY_PATH_DEFAULT, help="Path to spatial library root")
    args = parser.parse_args()

    library_path = args.library_path
    if not os.path.isdir(library_path):
        print(f"ERROR: Library path does not exist: {library_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {library_path}")
    registry = scan_library(library_path)

    # Write registry
    registry_path = os.path.join(library_path, "registry.json")
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2, default=str)

    # Print summary
    species = {r["species"] for r in registry}
    organs = {r["organ"] for r in registry}
    diseases = {r["disease"] for r in registry}
    healthy = sum(1 for r in registry if r.get("is_healthy"))
    diseased = sum(1 for r in registry if r.get("is_healthy") is False)
    no_coords = sorted(r["sample_id"] for r in registry if r.get("has_spatial_coordinates") is False)

    print(f"\nRegistry built: {registry_path}")
    print(f"  Total datasets: {len(registry)}")
    print(f"  Species: {sorted(species)}")
    print(f"  Organs ({len(organs)}): {sorted(organs)}")
    print(f"  Healthy: {healthy}, Diseased: {diseased}, Unknown: {len(registry) - healthy - diseased}")
    print(f"  Diseases: {sorted(diseases)}")
    if no_coords:
        print(f"  No spatial coordinates ({len(no_coords)}; not usable by spatial tools): {no_coords}")


if __name__ == "__main__":
    main()
