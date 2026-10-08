"""What a format conversion wrote, recorded inside the file it wrote.

``run_spatial_pipeline``, ``repair_spatial_h5ad``, ``auto_convert`` and the ``convert_*`` family turn
the user's data into an h5ad an MCP tool can read. The file they write is the *input*, re-encoded:
every obs column in it arrived with the data or is a QC count computed from it. Nothing downstream
could tell, because the file carried no record of what made it -- an MCP worker leaves
``sog_run_provenance.json`` beside its outputs, and these utilities left nothing. So post-analysis
read the copy in ``outputs/`` (``repaired_merfish.h5ad``; ``converted_merfish.h5ad`` in r3) as a
tool result: all three E-03 ``qc_01`` trials were typed ``spatial_clustering``, published the
input's own ``cell_type_annot`` as "13 spatial domains" with a domain map, and were reviewed ``ok``
-- on the portal, the user's labels presented as a clustering this run produced.

The record lives in ``uns[STAMP_KEY]``, in the file, rather than in a sidecar. A sidecar is per
directory, and a conversion usually writes into ``outputs/`` itself -- the directory the model's own
tables and every later tool share -- so one record there would either describe files it did not
write or overwrite a worker's provenance for files it did. The uns entry travels with the file
through a copy, a move and a rename.

It also travels into every file a tool derives from it, because a tool that reads an h5ad and writes
one back keeps ``uns``. So the record carries the *layout* of the file as the conversion left it --
every top-level slot and the names inside it -- and a file only counts as the conversion's own output
while its layout is still that one. A clustering tool's result has a new obs column, a new uns entry
or a graph in ``obsp``; any of them makes it a descendant, which is a tool result and is analysed as
one. What a descendant keeps is :attr:`ConversionRecord.obs_columns`: the names of the columns that
were already there when the input was converted -- every label the input arrived with, beside the QC
counts the conversion computed. Names only: a tool that writes over one of those columns leaves its
name on the list, so what is said of such a column is "unless a tool wrote over it".

Stdlib only at import. ``h5py`` is imported by the two functions that open a file, and both never
raise: a conversion must not fail because its record could not be written, and post-analysis must
not fail because a file could not be asked about one.
"""

from __future__ import annotations

import json
from typing import NamedTuple

__all__ = ["STAMP_KEY", "ConversionRecord", "read_conversion_record", "stamp_conversion"]

#: The ``uns`` entry holding the record. Not a name any tool or the benchmark reads, and it contains
#: none of the substrings the code that walks ``uns`` by name matches on -- as of 2026-09-25,
#: ``cell_type`` / ``abundance`` (``output_standardizer``), ``image`` (``tuning/adaptive``) and
#: ``nhood`` / ``enrich`` (``viz/pipelines``).
STAMP_KEY = "sog_conversion"


class ConversionRecord(NamedTuple):
    """What :func:`read_conversion_record` found.

    ``unchanged`` is the question discovery asks: is this file still exactly what the conversion
    wrote? ``obs_columns`` answers the one review asks of a descendant too: which labels came with
    the input? ``source`` is empty when the writer was not told what it converted from.
    """

    producer: str
    source: str
    obs_columns: tuple[str, ...]
    unchanged: bool


def stamp_conversion(path, *, producer: str, source=None) -> bool:
    """Record in ``path``'s ``uns`` that ``producer`` wrote it from ``source``. Never raises.

    Called after the file is written and closed, so the layout recorded is the layout on disk -- the
    same function computes it here and in :func:`read_conversion_record`, so the two can only
    disagree when the file changed in between. Stamping again replaces the record: the pipeline
    re-stamps after it embeds images, which changes the layout the converter recorded.
    """
    try:
        import h5py

        write_elem = _write_elem()
        if write_elem is None:
            return False
        with h5py.File(str(path), "r+") as handle:
            if "uns" not in handle:
                write_elem(handle, "uns", {})
            uns = handle["uns"]
            if STAMP_KEY in uns:
                del uns[STAMP_KEY]
            contents = {"layout": _layout(handle), "obs_columns": _obs_columns(handle)}
            record = {
                "producer": str(producer),
                "source": "" if source is None else str(source),
                "contents": json.dumps(contents, sort_keys=True),
            }
            try:
                write_elem(uns, STAMP_KEY, record)
            except Exception:
                # Half a record is worse than none: ``anndata.read_h5ad`` would trip over a group
                # with no encoding attributes, and the file is the user's data.
                if STAMP_KEY in uns:
                    del uns[STAMP_KEY]
                raise
        return True
    except Exception:
        return False


def read_conversion_record(path) -> ConversionRecord | None:
    """The record ``path`` carries, or ``None`` when it carries none (or is not an h5ad). Never raises.

    The record is returned for a descendant as well as for the conversion's own output, with
    ``unchanged`` telling the two apart; see the module docstring for why a descendant keeps one.
    """
    if not str(path).lower().endswith(".h5ad"):
        return None
    try:
        import h5py

        with h5py.File(str(path), "r") as handle:
            uns = handle.get("uns")
            if not isinstance(uns, h5py.Group) or STAMP_KEY not in uns:
                return None
            stamp = uns[STAMP_KEY]
            if not isinstance(stamp, h5py.Group):
                return None
            contents = json.loads(_text(stamp["contents"][()]))
            unchanged = contents.get("layout") == json.loads(json.dumps(_layout(handle), sort_keys=True))
            return ConversionRecord(
                producer=_text(stamp["producer"][()]) if "producer" in stamp else "",
                source=_text(stamp["source"][()]) if "source" in stamp else "",
                obs_columns=tuple(str(c) for c in contents.get("obs_columns") or ()),
                unchanged=bool(unchanged),
            )
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------


def _write_elem():
    """anndata's element writer, under whichever name this anndata publishes it."""
    try:
        from anndata.io import write_elem

        return write_elem
    except Exception:
        pass
    try:
        from anndata.experimental import write_elem

        return write_elem
    except Exception:
        return None


def _layout(handle) -> dict[str, dict[str, list]]:
    """Every top-level slot of an open h5ad: the names in a group, the shape of a dataset.

    Names, not values: reading ``X`` to fingerprint it would cost a full pass over the counts, and a
    tool that changed the file changed its names -- a label column, a graph, an embedding, a uns
    entry recording the parameters. ``X``'s shape rides along in its group's ``shape`` attribute when
    it is sparse, so a subset of the cells or genes is a different layout as well.

    What names cannot see is a tool that changed values only -- ``X`` normalised, denoised or imputed
    and written to a new path with no new name anywhere, or a column overwritten in place. Such a
    file reads as the copy and is passed over, though it is a result. Before the record it was
    analysed, and when the input carried a label column, as a clustering of that column -- the
    defect this record exists to stop. The same holds for a clustering written entirely under names
    the input already had: the input's own ``leiden`` column, say, when its ``uns['leiden']`` came
    with it too. Accepted rather than read by value, because a fingerprint of ``X`` is a full pass
    over the counts at every stamp and every question. The recorded E-01/E-03 trees (2026-09-25)
    keep 11 h5ad files, at most one per trial, so none of those is such a descendant -- which
    covers only what the harness kept. It replaced 32 more h5ad outputs, in 24 of the 96 trials,
    with ``.skipped-oversize`` placeholders holding a byte count, and it removes ``data/`` when a
    trial ends, so what the other files were cannot be measured from the record.
    """
    import h5py

    layout: dict[str, dict[str, list]] = {}
    for name in sorted(handle.keys()):
        member = handle[name]
        if isinstance(member, h5py.Group):
            keys = sorted(k for k in member.keys() if not (name == "uns" and k == STAMP_KEY))
            entry: dict[str, list] = {"keys": keys}
            shape = member.attrs.get("shape")
            if shape is not None:
                entry["shape"] = [int(v) for v in shape]
        else:
            entry = {"shape": [int(v) for v in member.shape]}
        layout[name] = entry
    # A file written with nothing in ``uns`` may have no ``uns`` group until the record creates one.
    layout.setdefault("uns", {"keys": []})
    return layout


def _obs_columns(handle) -> list[str]:
    """The obs column names, in order, without the index or a legacy categories group."""
    obs = handle.get("obs")
    if obs is None or not hasattr(obs, "attrs"):
        return []
    order = obs.attrs.get("column-order")
    if order is not None:
        if isinstance(order, (str, bytes)):
            return [_text(order)]
        return [_text(c) for c in (order.tolist() if hasattr(order, "tolist") else order)]
    index = _text(obs.attrs.get("_index", "_index"))
    return sorted(k for k in obs.keys() if k not in (index, "__categories"))


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
