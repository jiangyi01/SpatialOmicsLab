"""Finding what to analyse: the files, the results directory, and the spatial coordinates.

Two pieces of knowledge already exist elsewhere in the repo and are borrowed rather than
re-derived, because a second copy is a copy that drifts:

* which directories hold *staged inputs* rather than results
  (``output_inspector.STAGED_INPUT_DIRS``) -- Celloscope's ``celloscope_data/matB.csv`` is genes x
  cell types with real cell-type column names, so it reads as a perfectly plausible proportions
  matrix and has been picked as one;
* the registry's per-tool output specs, consulted through :mod:`.detect`.

There is deliberately *no* resolver for "where does a tool write by default" here. The contract
(``docs/design/post_analysis_contract.md``, architect's correction) rules that out: a run currently
scatters across two roots, ``tools/base_mcp.py`` is not importable from an installed wheel, and a
third by-path loader for it would be the wrong answer. The results directory is either given by the
caller or derived from the source directory the caller already named -- never guessed.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from .manifest import is_our_manifest
from .tables import is_tabular

if TYPE_CHECKING:
    from spatialomicsgym.tool.conversion_record import ConversionRecord

logger = logging.getLogger(__name__)

#: The directory the current post-analysis run was pointed at, set by the engine for the duration
#: of one run. Staging markers describe scratch directories a worker creates *inside* a run's tree;
#: the caller-designated root (and its ancestors on an absolute path) is not evidence, so the
#: predicate only scans components below it. A ContextVar rather than a parameter because the
#: question is asked from a dozen places -- detection, coordinate search, the scan inventory, every
#: task runner -- and all of them mean "within the run the engine is currently analysing".
_ACTIVE_SCAN_ROOT: ContextVar[Path | None] = ContextVar("_ACTIVE_SCAN_ROOT", default=None)


@contextmanager
def staged_scan_root(root: Path | str):
    """Scope within which :func:`is_staged_input` judges paths relative to ``root``."""
    token = _ACTIVE_SCAN_ROOT.set(Path(root))
    try:
        yield
    finally:
        _ACTIVE_SCAN_ROOT.reset(token)


#: Files that are output *about* the run, never input to this analysis.
_IGNORED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".pdf", ".svg", ".log", ".lock", ".tmp"})

#: Column-name pairs that mean "spatial coordinates", most explicit first.
#:
#: Each pair is ``(x_column, y_column)`` -- horizontal first -- because that is how
#: :func:`_coordinates_from_frame` consumes it. Four of these are natively written (row, col) and
#: are therefore listed here the other way round: a *row* index is the vertical position and a
#: *column* index the horizontal one. Listing them in written order, as this did, put the vertical
#: coordinate on the x axis and reflected every tissue map across its main diagonal -- on Visium's
#: 78 x 128 capture grid, a landscape section rendered portrait and stopped matching the H&E image
#: beside it. 10x's own ``obsm['spatial']`` is ``[pxl_col_in_fullres, pxl_row_in_fullres]``, which
#: is why :func:`coordinates_from_anndata` reading columns 0 and 1 as x and y is the convention to
#: agree with rather than a second valid one.
_COORDINATE_COLUMN_PAIRS: tuple[tuple[str, str], ...] = (
    ("x", "y"),
    ("coor_x", "coor_y"),
    ("x_coord", "y_coord"),
    ("imagecol", "imagerow"),
    ("pxl_col_in_fullres", "pxl_row_in_fullres"),
    ("array_col", "array_row"),
    ("col", "row"),
    ("spatial_1", "spatial_2"),
    ("x_pixel", "y_pixel"),
)

#: Below this overlap with the prediction index, a coordinate table is about some other object.
_MIN_INDEX_OVERLAP = 0.5

#: The sidecar every MCP worker leaves beside its outputs -- ``tools/worker_utils.PROVENANCE_FILENAME``,
#: spelled out because ``tools/`` is not importable from an installed wheel. A test pins the two.
WORKER_PROVENANCE = "sog_run_provenance.json"

#: MCP workers whose every output file is the input re-encoded, by the ``tool`` their sidecar names.
#: Found by reading every ``WorkerOutput(...)`` under ``tools/`` (2026-09-25): ``data_converter`` --
#: h5ad to and from a CSV bundle or a Seurat RDS -- is the only one; every other worker writes an
#: analysis. The in-process converters need no name here, because they record it in the file itself.
#: A test fails when an MCP server offers a ``convert_*`` function whose worker is not listed.
#:
#: The limit of reading it this way: there is one sidecar per folder, so a later worker that writes into
#: the converter's folder -- a figure, or a second ``data_converter`` call -- replaces the record, and
#: the converted file then reads as a result again. The in-process converters' stamp, carried in the
#: file itself, does not have this hole; a converter moved to that stamp would close it.
CONVERSION_WORKERS = frozenset({"data_converter"})


def is_staged_input(path: Path) -> bool:
    """Whether ``path`` is a staged input / derived artifact rather than a tool result.

    Inside an engine run (see :func:`staged_scan_root`) only the components below the run's own
    source directory are scanned, so the name the caller chose for that directory -- however much
    it resembles a staging marker, ``scanpy_on_converted`` being the recorded live case -- cannot
    disqualify the results inside it.

    A file a conversion utility wrote is one too, wherever it sits and whatever it is called: it is
    the input re-encoded (:func:`is_conversion_output`). The path rule cannot see that --
    ``outputs/repaired_merfish.h5ad`` and ``data_converter``'s ``converted.h5ad`` have no staging
    component -- so it is asked of the file and of the record beside it.
    """
    try:
        from spatialomicsgym.benchmarking.output_inspector import _is_staged_input_path

        staged = bool(_is_staged_input_path(path, root=_ACTIVE_SCAN_ROOT.get()))
    except Exception:
        staged = False
    return staged or is_conversion_output(path)


def is_conversion_output(path: Path) -> bool:
    """Whether ``path`` is still exactly the file a conversion utility wrote, by that utility's record.

    Two records answer it, one per kind of converter:

    * ``run_spatial_pipeline`` and the in-process converters stamp the h5ad they write
      (:mod:`spatialomicsgym.tool.conversion_record`); a tool result derived from such a file keeps
      the stamp but not the layout, and is not one.
    * the MCP ``data_converter`` runs in a worker env where that module does not import, and
      records what it wrote the way every worker does: :data:`WORKER_PROVENANCE` beside its first
      output, naming the tool and its ``output_files`` (:func:`_listed_by_a_converter`). Its CSV
      bundle is the input re-encoded as much as its h5ad is, so a CSV can be one too.

    Asked from every picker, so the answer is cached on the size and mtime of the file and of the
    sidecar beside it -- never on the path alone.
    """
    return conversion_evidence(path) is not None


def conversion_evidence(path: Path) -> str | None:
    """Which record says ``path`` is a conversion utility's own output, as a phrase; ``None`` if none does.

    For a check that fails on it and has to say what it read: ``its own uns['sog_conversion']
    record`` or ``data_converter's sog_run_provenance.json beside it``.
    """
    try:
        own = Path(path).stat()
    except (OSError, ValueError, TypeError):
        return None
    try:
        beside = (Path(path).parent / WORKER_PROVENANCE).stat()
        sidecar = (beside.st_size, beside.st_mtime_ns)
    except (OSError, ValueError, TypeError):
        sidecar = None
    return _conversion_evidence_at(str(path), own.st_size, own.st_mtime_ns, sidecar)


def conversion_record(path: Path) -> ConversionRecord | None:
    """The conversion record ``path`` carries -- its own or inherited -- or ``None``. Never raises."""
    try:
        from spatialomicsgym.tool.conversion_record import read_conversion_record

        return read_conversion_record(path)
    except Exception:
        return None


@lru_cache(maxsize=512)
def _conversion_evidence_at(path: str, size: int, mtime_ns: int, sidecar: tuple[int, int] | None) -> str | None:
    if path.lower().endswith(".h5ad"):
        record = conversion_record(Path(path))
        if record is not None and record.unchanged:
            return "its own uns['sog_conversion'] record"
    # The sidecar is written after the files it lists (``WorkerOutput.to_dict``), so a listed file
    # newer than it was written again since -- the model put its own labels into the converted h5ad
    # in place, say -- and is a result. The same reading ``unchanged`` gives the stamped route, by
    # the only measure the record allows: it holds paths, not sizes or layouts.
    if sidecar is not None and mtime_ns <= sidecar[1]:
        tool = _listed_by_a_converter(Path(path))
        if tool:
            return f"{tool}'s {WORKER_PROVENANCE} beside it"
    return None


def _listed_by_a_converter(path: Path) -> str | None:
    """The converter whose sidecar beside ``path`` lists it among the files it wrote, or ``None``.

    Only a tool in :data:`CONVERSION_WORKERS` counts: every worker leaves the same sidecar, and a
    clustering tool's listing its own h5ad is what makes that file a result. A listed path is
    matched on its name and its folder's name rather than resolved, because the sidecar describes the
    files beside it and a workspace that was copied or moved -- the harness keeps each trial's tree
    under another root -- still holds them under both names.
    """
    try:
        record = json.loads((path.parent / WORKER_PROVENANCE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    tool = record.get("tool") if isinstance(record, dict) else None
    files = record.get("output_files") if isinstance(record, dict) else None
    if not isinstance(tool, str) or tool not in CONVERSION_WORKERS or not isinstance(files, dict):
        return None
    for value in files.values():
        for listed in value if isinstance(value, list) else [value]:
            # ``add_output_file(key, None)`` records a null, and a sidecar is a file anyone can edit.
            if not isinstance(listed, str) or not listed:
                continue
            written = Path(listed)
            if written.name == path.name and written.parent.name in ("", path.parent.name):
                return tool
    return None


#: Kept as a module-local name because this is where the question is asked from; the rule itself
#: lives beside the writer (:func:`manifest.is_our_manifest`) so the benchmark inspector, which asks
#: the same question one layer down, cannot drift from it.
_is_post_analysis_manifest = is_our_manifest


def collect_files(
    source: Path,
    exclude: Path | None = None,
    *,
    skipped: list[Path] | None = None,
    not_analysed: list[Path] | None = None,
) -> list[Path]:
    """Every non-empty candidate file under ``source``, sorted, excluding our own results dir.

    Excluding ``exclude`` is what makes a second run of the same directory idempotent: the default
    results dir lives *inside* the tool's output dir, so without this the engine would scan its own
    figures and tables back in.

    ``exclude`` names the directory *this* run writes to, which is not the same thing as "our own
    output", and the gap between the two is reachable in one step: run once with the default, then
    again with ``results_dir=`` pointing somewhere else, and the first run's ``post_analysis/`` is
    still sitting in the source with nothing excluding it. On the recorded stPlus fixture that
    second run scanned 8 files instead of 3 and published ``status: ok`` with
    ``top_cell_type: "n_spots"`` -- a column of the previous run's own findings table, read back as
    a cell type. So every directory holding a manifest we wrote is skipped as well, identified by
    content, which covers the mirror case too (a previous run pointed at a custom directory inside
    the source, and this run using the default).

    Pass ``skipped`` to be told which files were dropped for being ours. Skipping silently makes one
    directory look empty that plainly is not -- the previous report itself -- and the caller cannot
    otherwise tell that case from a genuinely empty one, so it would report "no readable output
    files" about a directory holding ten of them.

    Pass ``not_analysed`` for the same reason one clause down: the files dropped for their suffix or
    for being empty are still files the tool wrote, and the return value is the *candidate* list, not
    the directory. Every caller that publishes a count was reading ``len`` of the candidates and
    calling it "Files written by the tool"; on the recorded stlearn run that is 2 against 202, and
    the 200 excluded ``.jpeg`` tiles are the run's deliverable. Across the 92 recorded runs the two
    numbers differ on 22, and not one of the 394 excluded files is a ``.log``, a ``.lock`` or empty
    -- they are images and PDFs, which is to say output.
    """
    source = Path(source)
    if source.is_file():
        return [source]
    if not source.is_dir():
        return []
    entries = [p for p in sorted(source.rglob("*")) if p.is_file()]
    ours = [p.parent for p in entries if p.name == "manifest.json" and _is_post_analysis_manifest(p)]
    found: list[Path] = []
    for path in entries:
        if exclude is not None and _is_within(path, exclude):
            continue
        if any(_is_within(path, d) for d in ours):
            if skipped is not None:
                skipped.append(path)
            continue
        # Below this line a file is *in* the directory and merely unreadable to this engine, so it
        # goes to ``not_analysed`` rather than nowhere. Above it the file is somebody else's -- the
        # excluded results dir, or a previous report -- and must stay out of both lists, which is
        # why these two tests come last: counting "every file" from the top would make the second
        # run of a directory credit the tool with the first run's figures, and grow the number on
        # every run after that.
        if path.suffix.lower() in _IGNORED_SUFFIXES:
            if not_analysed is not None:
                not_analysed.append(path)
            continue
        try:
            empty = path.stat().st_size == 0
        except OSError:
            empty = True  # unreadable is not analysable either, and it is still a file that is there
        if empty:
            if not_analysed is not None:
                not_analysed.append(path)
            continue
        found.append(path)
    return found


def scan_rows(files: list[Path], root: Path) -> list[dict[str, object]]:
    """The file inventory every run publishes as ``tables/output_scan.csv``."""
    rows = []
    for path in files:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        try:
            name = str(path.relative_to(root))
        except ValueError:
            name = path.name
        rows.append(
            {
                "file": name,
                "size_bytes": size,
                "suffix": path.suffix.lower().lstrip("."),
                "staged_input": is_staged_input(path),
            }
        )
    return rows


def is_under_benchmark_results(path: Path) -> bool:
    """``benchmarks/results/`` holds recorded experiment outputs and is read-only, always.

    How the path is *spelled*. Prefer :func:`leads_under_benchmark_results` anywhere a write can
    follow; this one is for the case where there is nothing on disk to resolve through yet.

    Deliberately **not** anchored to this checkout's root, and deliberately over-broad. A second
    clone (the ``REPRODUCE.md`` flow makes one) grows its own ``benchmarks/results/`` that is just
    as unregenerable, and ``benchmarks/results/`` is not in git, so there is no root to anchor to
    that would still cover it. The trade is accepted in one direction only: a user directory that
    happens to be spelled ``.../benchmarks/results/...`` gets its post-analysis redirected to a temp
    dir with a note (``engine._resolve_results_dir``) and is skipped by auto-discovery -- annoying,
    and undone with ``--results-dir``. The other direction loses 669 GB that cannot be rebuilt. Nine
    test modules assert the guard fires under ``tmp_path``, i.e. on this exact over-breadth.
    """
    parts = [p.lower() for p in Path(path).parts]
    return any(parts[i] == "benchmarks" and parts[i + 1] == "results" for i in range(len(parts) - 1))


def leads_under_benchmark_results(path: Path) -> bool:
    """Whether ``path`` *leads* under ``benchmarks/results/`` -- following symlinks to find out.

    :func:`is_under_benchmark_results` reads ``Path.parts``, which is how a path is spelled and not
    where it goes. A symlink steps over it. That is not hypothetical in this repo: both
    ``benchmarks/manuscript/per_cell_results/multi_llm`` and ``.../agent_legacy_default`` point
    straight into ``benchmarks/results/``, so a caller naming a path through either one -- or simply
    analysing a source reached that way, since the default results dir is derived from the source --
    wrote into 669 GB of recorded experiment output that is not in git and cannot be regenerated.

    Every layer that can write must use *this* one. It lives here rather than in a caller because it
    was duplicated once already: L1 resolved and L2 did not, and L2 rewrote a recorded manifest.

    Both spellings are checked. ``resolve()`` is what catches the link; the literal check is what
    still holds when there is nothing on disk yet to resolve *through* (the results dir is usually
    the part that does not exist), and is also the answer if resolution itself fails.
    """
    if is_under_benchmark_results(path):
        return True
    try:
        return is_under_benchmark_results(Path(path).resolve())
    except (OSError, RuntimeError):  # symlink loop, permission denied -- the literal check stands
        return False


def _is_within(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except (ValueError, OSError):
        return False


# ---------------------------------------------------------------------------------------
# anndata / coordinates
# ---------------------------------------------------------------------------------------


@contextmanager
def open_anndata(path: Path, backed: bool = False):
    """Yield an AnnData for ``path`` (or ``None``), and hand the file back on the way out.

    ``backed="r"`` keeps an open HDF5 handle, and HDF5 takes a lock: a reader that returns without
    closing makes the next writer to that path fail with ``errno 11``. Letting the local fall out of
    scope is not enough either -- ``adata.raw`` holds a reference back to its parent, so the pair is
    a refcount cycle that only the cyclic collector breaks. Hence the explicit ``finally``, which is
    also what ``test/test_h5ad_readers_release_the_file.py`` scans for.
    """
    adata = None
    try:
        import anndata as ad

        adata = ad.read_h5ad(str(path), backed="r" if backed else None)
    except Exception:
        adata = None
    try:
        yield adata
    finally:
        try:
            if adata is not None and getattr(adata, "isbacked", False):
                adata.file.close()
        except Exception:
            pass


def read_anndata(path: Path):
    """Read a *fully loaded* h5ad, or ``None``. Backed reads must use :func:`open_anndata`."""
    with open_anndata(path, backed=False) as adata:
        return adata


def coordinates_from_anndata(adata):
    """``(n_obs, 2)`` DataFrame of x/y from ``obsm['spatial']`` or an obs column pair, else ``None``."""
    import pandas as pd

    for key in ("spatial", "X_spatial", "spatial_coords"):
        matrix = adata.obsm.get(key) if hasattr(adata.obsm, "get") else None
        if matrix is None:
            continue
        try:
            values = matrix.to_numpy() if hasattr(matrix, "to_numpy") else matrix
            if values.shape[1] >= 2:
                return pd.DataFrame({"x": values[:, 0], "y": values[:, 1]}, index=[str(i) for i in adata.obs_names])
        except Exception:
            continue
    return _coordinates_from_frame(adata.obs, index=[str(i) for i in adata.obs_names])


def _coordinates_from_h5ad(path: Path):
    """Coordinates out of one h5ad, read *backed*: ``obs`` and ``obsm`` only, never ``X``.

    A full read pulls the whole expression matrix through RAM to hand back a two-column array --
    measured on clean processes with a 12,000 x 4,000 section, 338 MB peak RSS against 152 MB for
    the backed peek, the difference being ``X`` -- and this is paid once per candidate h5ad until
    one is accepted. A 200k-spot x 20k-gene section makes that tens of GB. :func:`detect._from_anndata`
    already peeks the same slots the same way.

    The second pass is not decoration. :func:`open_anndata` reports *every* failure as ``None``, so
    a file anndata will only read whole would otherwise be indistinguishable from one holding no
    coordinates, and its tissue map would go missing from the report with nothing said.
    """
    for backed in (True, False):
        with open_anndata(path, backed=backed) as adata:
            if adata is not None:
                return coordinates_from_anndata(adata)
    return None


def _coordinates_from_frame(frame, index=None):
    import pandas as pd

    lowered = {str(c).strip().lower(): c for c in frame.columns}
    for x_name, y_name in _COORDINATE_COLUMN_PAIRS:
        if x_name in lowered and y_name in lowered:
            try:
                out = pd.DataFrame(
                    {
                        "x": pd.to_numeric(frame[lowered[x_name]], errors="coerce").to_numpy(),
                        "y": pd.to_numeric(frame[lowered[y_name]], errors="coerce").to_numpy(),
                    },
                    index=[str(i) for i in (index if index is not None else frame.index)],
                )
            except Exception:
                continue
            if out.notna().all(axis=1).sum() >= 2:
                return out.dropna()
    return None


def _rejected_coordinates_note(path: Path, n_coordinates: int, matched: int, n_wanted: int) -> str:
    """Why the best coordinate table in the directory was not the one for this result."""
    if matched:
        measured = (
            f"only {matched} of the {n_wanted} prediction rows are among them ({matched / n_wanted:.0%}, "
            f"under the {_MIN_INDEX_OVERLAP:.0%} it takes to be a table about the same spots)"
        )
    else:
        measured = f"not one of the {n_wanted} prediction rows is among them"
    return (
        f"{path.name} holds coordinates for {n_coordinates} spots and {measured}, so the spatial "
        "figures were skipped. Whatever this result is indexed by, those are not the spots in that "
        "file -- either the two describe different objects, or the IDs are written in another form."
    )


def find_coordinates(files: list[Path], index=None, *, notes: list[str] | None = None) -> tuple[object, Path | None]:
    """``(coordinates, source_path)`` for the spots in ``index``, or ``(None, None)``.

    ``index`` is the prediction's own spot IDs. A coordinate table that overlaps it by less than
    half is about some other object -- a single-cell reference, a different section -- and is
    rejected rather than plotted against mismatched IDs.

    ``(None, None)`` means two different things and this is the only place that can tell them
    apart, so -- exactly as :func:`align_to_coordinates` does one step downstream -- it says which
    one through ``notes`` instead of leaving each caller to guess. Both callers guessed the same
    way: "No spatial coordinates found next to the result ... Keep the spatial h5ad (or a
    coordinates CSV) in the output directory to get them." Measured over the 92 recorded runs, 18
    emit that sentence and on four of them a coordinate table is already in the directory --
    istar's ``locs-raw.tsv``, bayestme's ``bayestme_deconvolved.h5ad``, iris_spatial's
    ``iris_proportions.csv`` and moscot's ``adata_mapping_spatial.h5ad``, 200 spots each -- so the
    advice asks for a file that is there.

    The overlap it measured on all four is *zero*, and that number is the answer to a question the
    run has already answered without evidence: istar publishes ``n_spots: 50`` under the label
    "Spots deconvolved" over 50 rows that are its gene panel, and bayestme 500 the same way. The
    experiment's spots are named in the table that got discarded and not one of the rows is among
    them. Rejecting it silently spends the one measurement that shows the claim is wrong.

    Reported, not diagnosed: zero overlap establishes that these are not the spots in that file,
    and cannot distinguish rows that are genes from spot IDs written in another form.
    """
    wanted = None if index is None else {str(i) for i in index}
    # The rejected candidate that came closest, so the note names the most favourable reading in a
    # directory holding several and does not imply the others were acceptable.
    best: tuple[int, int, Path] | None = None

    def _accept(frame, path):
        nonlocal best
        if frame is None or frame.empty:
            return None
        if wanted is None:
            return frame
        hits = frame.index.astype(str).isin(wanted)
        matched = int(hits.sum())
        if matched / max(len(wanted), 1) < _MIN_INDEX_OVERLAP:
            if best is None or matched > best[0]:
                best = (matched, len(frame), path)
            return None
        return frame.loc[hits]

    for path in [f for f in files if f.suffix == ".h5ad" and not is_staged_input(f)]:
        accepted = _accept(_coordinates_from_h5ad(path), path)
        if accepted is not None:
            return accepted, path

    from .tables import read_table

    for path in [f for f in files if is_tabular(f) and not is_staged_input(f)]:
        frame = read_table(path)
        if frame is None:
            continue
        accepted = _accept(_coordinates_from_frame(frame), path)
        if accepted is not None:
            return accepted, path

    # Staged inputs, last. The staging veto exists so a staged copy of the *input* is never
    # mistaken for the *prediction*; coordinates are the opposite case -- they describe the
    # input's spots, and the staging directory is often the only place they exist. Live round 2:
    # RCTD fed from ``csv_spatial/`` skipped every tissue map while the tool's own
    # ``--spatial-coords-csv`` sat one directory below the result. Non-staged candidates keep
    # absolute priority above, and the index-overlap gate still judges what staging no longer
    # vetoes. Tabulars are sniffed at four rows first so a staged counts matrix of hundreds of
    # MB is rejected on its header line, not read whole.
    for path in [f for f in files if f.suffix == ".h5ad" and is_staged_input(f)]:
        accepted = _accept(_coordinates_from_h5ad(path), path)
        if accepted is not None:
            return accepted, path
    for path in [f for f in files if is_tabular(f) and is_staged_input(f)]:
        sniff = read_table(path, nrows=4)
        if sniff is None or _coordinates_from_frame(sniff) is None:
            continue
        frame = read_table(path)
        if frame is None:
            continue
        accepted = _accept(_coordinates_from_frame(frame), path)
        if accepted is not None:
            return accepted, path

    if best is not None and wanted:
        matched, n_coordinates, path = best
        _refuse(_rejected_coordinates_note(path, n_coordinates, matched, len(wanted)), notes)
    return None, None


def _one_row_per_spot(frame, what: str, notes: list[str] | None = None):
    """``frame`` re-indexed by string spot ID, first row kept where an ID repeats.

    ``.loc[labels]`` expands a repeated label into every row carrying it, so a duplicate on either
    side leaves the two frames built from different label multisets -- and the caller consumes them
    as parallel arrays. Measured: a prediction indexed ``[BC0, BC1, BC1, BC2]`` against coordinates
    ``[BC0, BC1, BC2]`` returned 6 values beside 4 positions; the mirror case returned 3 beside 4
    and :func:`plots.spatial_scatter` raised ``IndexError``, which ``step()`` records as a warning
    -- so the tissue map was simply *absent* from the report rather than visibly wrong.
    """
    frame = frame.copy()
    frame.index = frame.index.astype(str)
    repeated = frame.index.duplicated()
    if repeated.any():
        message = (
            f"{int(repeated.sum())} of {len(frame)} {what} rows repeat a spot ID; keeping the first "
            "row of each so the values and the coordinates stay row-for-row aligned"
        )
        logger.warning("%s", message)
        if notes is not None:
            notes.append(message)
        frame = frame[~repeated]
    return frame


def _refuse(message: str, notes: list[str] | None) -> None:
    """Record why :func:`align_to_coordinates` is about to return ``(None, None)``.

    Both refusal branches go through here so neither can grow a return without a reason. The
    callers turn ``notes`` into ``ctx.warn``, which is what puts the sentence in the manifest, the
    review and the report -- a figure that is absent for a reason the reader can see is a result;
    one that is absent silently is indistinguishable from one that was never attempted.
    """
    logger.warning("%s", message)
    if notes is not None:
        notes.append(message)


def _index_is_a_row_counter(frame) -> bool:
    """True when the index is pandas' row numbering rather than identifiers the file carried.

    :func:`tables.read_table` leaves a ``RangeIndex`` in place exactly when it found no identifier
    column, and nothing else here produces one -- ``_set_identifier_index`` moves a real integer
    ``spot`` column onto the index and what comes back is a plain ``Index``. So this is what
    separates "the tool named its spots 0..n-1" from "we numbered the rows".
    """
    import pandas as pd

    return isinstance(frame.index, pd.RangeIndex)


def align_to_coordinates(frame, coordinates, *, notes: list[str] | None = None):
    """Restrict ``frame`` and ``coordinates`` to their shared spots, in ``frame`` order.

    The two come back the same length and in the same order, so the caller can hand them to a plot
    as parallel arrays -- which is what every caller does. :func:`_one_row_per_spot` is what makes
    that true when a spot ID repeats on either side.

    ``(None, None)`` means the two cannot be matched, and this function is the only place that
    knows why, so it always says so through ``notes`` rather than leaving each caller to guess. The
    callers used to add a sentence of their own here; both were unreachable -- ``find_coordinates``
    restricts the coordinates to the prediction's own IDs, so the intersection can never be empty
    by the time it hands them over -- and once the row-counter refusal below made the branch live,
    "share no spot IDs" would have been a second, vaguer statement of a case it does not describe.

    Pass ``notes`` to collect what it had to repair. A repeated spot ID means the tool wrote a
    malformed prediction, which is a finding about the *result* -- so the task runners hand it to
    ``ctx.warn`` and it reaches the manifest, the review and the report. Without that it went only
    to the module logger, where nothing downstream of the run ever reads it.

    Restricting is the other thing worth saying. Every figure downstream covers the shared spots
    while the tables beside them cover all of them, and both are captioned with the count they were
    computed over -- so when the two sets differ the report holds two numbers for the same tissue
    and no way to reconcile them. Measured on the recorded ``novosparc`` run, 300 prediction rows
    against 200 coordinates: the composition bar chart says "across 300 spots" and the tissue-map
    grid, captioned "Top 9 columns by mean proportion", ranks over the 200 and draws nine columns
    with no overlap at all with the bar chart's top nine.
    """
    # A prediction with no ID column has no spot IDs to share, only row numbers the reader supplied,
    # and intersecting those with a coordinate index that is also small integers matches on
    # arithmetic. novosparc's coupling is the recorded case: 300 cells x 200 locations written with
    # a bare `0,1,...,199` header and no index column, against an h5ad whose 200 locations are named
    # `0..199`. 200 of the 300 row numbers collided, `find_coordinates` cleared its 0.5 overlap floor
    # at 0.67, and the run published two tissue maps drawing cell 5's coupling row at location 5's
    # coordinates -- spatial structure nothing in the data supports. Same-length is the case that
    # must survive: when a tool writes one row per spot in the coordinate file's order, row i is
    # spot i, and that is a convention rather than a coincidence (cytospace, 200 against 200).
    # Different lengths admit no such reading, so the figures are refused instead of invented.
    if _index_is_a_row_counter(frame) and len(frame) != len(coordinates):
        _refuse(
            f"The prediction carries no spot identifiers: the file had no ID column, so its "
            f"{len(frame)} rows are numbered 0-{len(frame) - 1} by the reader. Pairing them with "
            f"the {len(coordinates)} coordinates would match each row to the spot whose name is its "
            f"row number, and {len(frame)} rows cannot be {len(coordinates)} spots -- so the "
            "spatial figures were skipped rather than drawn on a coincidence of small integers.",
            notes,
        )
        return None, None

    frame = _one_row_per_spot(frame, "prediction", notes)
    coordinates = _one_row_per_spot(coordinates, "coordinate", notes)
    # Hoisted out of the comprehension it used to sit in, where the whole coordinate index was
    # converted and hashed once per spot: 0.02 s for 500 spots but 4.69 s for 8,000, i.e. ~50
    # minutes for a 200k-spot section, inside a step() that has no timeout. Kept as a list rather
    # than Index.intersection because the row order is the contract -- see the docstring.
    known = set(coordinates.index)
    shared = [i for i in frame.index if i in known]
    if not shared:
        _refuse(
            f"The {len(frame)} prediction rows and the {len(coordinates)} coordinates share no spot "
            "ID at all, so there is no way to place a prediction on the tissue and the spatial "
            "figures were skipped. The two files most likely describe different objects.",
            notes,
        )
        return None, None
    dropped = len(frame) - len(shared)
    if dropped:
        message = (
            f"{dropped} of {len(frame)} prediction rows have no matching coordinate and are left "
            f"out of the spatial figures, which cover the {len(shared)} rows that have one. Tables "
            "and counts elsewhere in this report are over all of them."
        )
        logger.warning("%s", message)
        if notes is not None:
            notes.append(message)
    return frame.loc[shared], coordinates.loc[shared]
