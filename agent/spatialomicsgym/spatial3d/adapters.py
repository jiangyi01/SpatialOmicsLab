"""Where each shipped aligner actually left its answer, and what it takes to read it.

Sixteen alignment functions are named here and no two of them agree about anything. PASTE's own
stacking overwrites ``obsm['spatial']`` on a copy; the worker puts the input back and writes the
answer beside it, to ``obsm['spatial_aligned']`` (``obsm['spatial_3d_aligned']`` when a z spacing
was declared), and files written before it did hold nothing but the overwritten key. GPSA writes
``obsm['spatial_aligned']`` in the input's units, and wrote it in a jointly min-max normalised
[0, 1] box before it said so in ``uns``. SPACEL writes the same key, as a ``DataFrame``,
median-centred per slice. moscot writes ``obsm['moscot_spatial_warp']`` for every section of one
merged object, onto a reference section it names in its params. ST-GEARS's answer is in
``obsm['spatial_elas']``, not in the ``obsm['st_gears_xyz']`` its own worker adds. SLAT writes a
table of matched cell indices and STACKer a warped NIfTI, neither of which is a coordinate. A
Phase-2 caller that wanted "the aligned coordinates" therefore had sixteen different problems, and
the ones that look easiest -- the tools that do write an obsm key -- are where the quiet errors
live, because the key is there and the numbers are in a frame nobody declared.

This module is the table that resolves that, function by function, against the worker source.

**A refusal is a result here, not a failure.** Four of the sixteen cannot produce a contract frame
at all, and they are listed with the same weight as the ones that can, each carrying the specific
reason and what would have to exist to lift it. That is why :class:`~.contract.Refusal` is
returned rather than raised: a caller walking the table wants twelve frames and four stated
reasons, not an exception that ends the walk at the first image-space tool.

**Some answers here are fitted by this module, and they say so.** SLAT produces correspondences,
not coordinates, so :func:`read_aligned` fits the transform itself and stamps ``derived_by`` on
the result; so does a GPSA file from before the worker inverted its [0, 1] box, whose units are
restored here. Coordinates a tool wrote in its own frame are read as written and carry no
``derived_by`` -- STalign's included: its CSV holds (y, x) point coordinates with no identifiers,
and the reader only swaps the two columns and joins the rows to the named slice by position,
which fits nothing. Nobody downstream should be able to read fitted coordinates as SLAT's own
opinion about where a cell belongs, nor GPSA's or STalign's own answer as something this module
computed.

**``seed_exposed`` is recorded, never omitted.** Six of the twelve portals -- moscot, STalign, SLAT,
SPIRAL, STACKer and SPACEL -- expose no seed through their MCP parameters (and paste2's
``paste2_estimate_overlap`` none either, though its alignment does). Two of those fix one
internally where the caller cannot reach it (SPACEL's ``Scube.align`` at 42, SPIRAL at 0). Leaving
the field out would let a reproducibility claim be made on their behalf by silence.

**No z is invented.** Nothing in this module supplies a spacing. ``z=None`` produces frames whose
z is genuinely absent and whose ``z_source`` is ``unknown``; a number appears only because the
caller passed one, which is why it is recorded as ``asked_user``.

Nothing heavy is imported at module scope: the table is read by ``postanalysis.detect`` and by
the portals, which must stay importable where anndata is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import contract

#: Stamped on anything this module computed rather than read out of a tool's own output.
DERIVED_BY = "spatialomicsgym.spatial3d.adapters"

#: The table can be read and its rows can be adapted, derived, refused or pending.
STATUS_ADAPTED = "adapted"  # the tool wrote coordinates and they are read as they are
STATUS_DERIVED = "derived"  # the tool wrote something else and the coordinates are fitted here
STATUS_REFUSED = "refused"  # no coordinate frame can honestly be produced from this output
STATUS_PENDING = "pending"  # a peer is adding the portal; the row exists so its absence is visible

STATUSES = (STATUS_ADAPTED, STATUS_DERIVED, STATUS_REFUSED, STATUS_PENDING)

#: Every obsm key one of these tools writes an *alignment answer* into. ``postanalysis.detect``
#: reads this to recognise an aligner's output on disk.
#:
#: Plain ``spatial`` is deliberately absent. It is the key every spatial file already has, so
#: including it would mark every dataset in the repository as aligned -- and PASTE, the one tool
#: whose own code overwrites it, has its worker restore the input there and write the answer to
#: ``spatial_aligned`` / ``spatial_3d_aligned`` instead. The contract's ``spatial_3d_aligned`` is
#: recognised by ``postanalysis.detect`` through its alignment-name rule, not through this list.
ALIGNED_OBSM_KEYS = (
    "moscot_spatial_warp",
    "spatial_aligned",
    "spatial_elas",
    "spatial_elas_reuse",
    "spatial_rigid",
    "st_gears_xyz",
)

#: Coordinate keys these tools write that are *not* answers, kept separate so a detector does not
#: read one as a frame. SPACEL writes ``spatial_pair`` -- the input coordinates with the per-slice
#: median subtracted -- on its way to the aligned key (``SPACEL/Scube/alignment.py:179-181``).
COORDINATE_SIDE_EFFECT_KEYS = ("spatial_pair",)

#: ST-GEARS writes three keys as it goes and the last one present is the answer. ``_reuse`` exists
#: only when binning ran, which is the default, and is the one at the original spot resolution.
ST_GEARS_KEYS = ("spatial_elas_reuse", "spatial_elas", "spatial_rigid")

#: Fitting a similarity from correspondences needs enough of them for the fit to mean anything.
#: Three is the algebraic minimum for a 2D similarity; below it the result is arbitrary rather
#: than merely poor, so it is refused instead of reported with a large residual.
MIN_CORRESPONDENCES = 3


@dataclass(frozen=True)
class AlignerAdapter:
    """One shipped alignment function, and everything needed to read what it wrote.

    ``writes_key`` is where the answer is, in the tool's own words -- an obsm key, or a sentence
    for the tools whose output is not an AnnData. ``role`` is the contract role the resulting
    frame takes; it is empty for a row that cannot produce one.
    """

    function: str
    server: str
    writes_key: str
    role: str
    original_recoverable: bool
    z_source: str
    seed_exposed: bool
    notes: str
    status: str = STATUS_ADAPTED
    #: Which entry of the worker's ``output_files`` carries the answer. A trailing ``<i>`` means a
    #: numbered family of keys, one per slice.
    output_key: str = ""
    #: Where ``obsm['spatial']`` for these same rows can be found: the output file itself, or the
    #: recorded input paths, or nowhere.
    original_from: str = ""
    #: The obsm key the adapted frame is written under. Everything that aligns lands in the
    #: contract's ``aligned`` key; a registration into somebody else's space gets its own name.
    frame_key: str = contract.ALIGNED_FRAME
    #: Whether :func:`read_aligned` cannot proceed without ``input_slices``.
    needs_input_slices: bool = False
    #: file:line in this repository or in the tool's own package, so every claim above is checkable.
    evidence: str = ""
    refusal_reason: str = ""
    refusal_remedy: str = ""

    def refusal(self, detail: str = "") -> contract.Refusal:
        """The named no this row stands for, optionally narrowed by what the caller actually had."""
        reason = self.refusal_reason or "this function produces no coordinate frame"
        if detail:
            reason = f"{reason} ({detail})"
        return contract.Refusal(what=self.function, reason=reason, remedy=self.refusal_remedy)

    def row(self) -> dict[str, Any]:
        """The flat record a table or a report prints."""
        return {
            "function": self.function,
            "server": self.server,
            "status": self.status,
            "writes_key": self.writes_key,
            "role": self.role,
            "original_recoverable": self.original_recoverable,
            "z_source": self.z_source,
            "seed_exposed": self.seed_exposed,
        }


@dataclass(frozen=True)
class AlignedPart:
    """One output object's worth of aligned coordinates.

    A tool that writes one file per slice produces one part per slice; a tool that writes a single
    merged object produces one part carrying ``slice_labels``. Both shapes are kept as they were
    found rather than normalised into each other, because splitting a merged object needs the
    slice column, and which column that is comes from the run's parameters rather than from the
    file.
    """

    label: str
    path: str
    key: str
    xy: Any
    z: Any | None = None
    obs_names: tuple[str, ...] = ()
    original_xy: Any | None = None
    slice_labels: tuple[str, ...] = ()

    @property
    def n_obs(self) -> int:
        return int(len(self.xy))

    def xyz(self) -> Any:
        """(n, 3) coordinates, or a :class:`~.contract.ContractError` if the z is not known."""
        import numpy as np

        if self.z is None:
            raise contract.ContractError(
                f"{self.label}: no z is known for this part, so it is a 2D frame. Pass z= to "
                f"read_aligned, or ask the user for the section spacing -- this package has no default."
            )
        return np.column_stack([np.asarray(self.xy, dtype="float64"), np.asarray(self.z, dtype="float64")])


@dataclass(frozen=True)
class AlignedFrames:
    """What an aligner produced, in the contract's terms, with how it was obtained attached."""

    function: str
    adapter: AlignerAdapter
    parts: tuple[AlignedPart, ...]
    z_source: str = "unknown"
    derived_by: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def frame_key(self) -> str:
        return self.adapter.frame_key

    @property
    def has_z(self) -> bool:
        return all(p.z is not None for p in self.parts) and bool(self.parts)

    def frame(self, **overrides: Any) -> contract.Frame:
        """The :class:`~.contract.Frame` provenance record for writing these coordinates.

        Units are left ``unknown`` unless the caller names them: this module reads numbers out of
        files that declare no units, and guessing here would put a fabricated declaration into the
        provenance block that ``contract.validate`` exists to check.

        A registration into somebody else's space takes ``registered_by`` rather than
        ``source_frame``, for the reason the contract gives about the Allen CCF: an external frame
        that claims a source frame in this stack reads as though Phase 2 produced it.
        """
        role = self.adapter.role or "aligned"
        fields: dict[str, Any] = {
            "key": self.adapter.frame_key,
            "role": role,
            "z_source": self.z_source,
            "xy_from": contract.ORIGINAL_KEY,
            "source_frame": contract.RAW_FRAME if role == "aligned" else "",
            "registered_by": self.adapter.server if role == "external" else "",
            "aligner": self.adapter.server,
            "aligner_function": self.function,
            "seed_exposed": self.adapter.seed_exposed,
            "created_by": self.derived_by or self.adapter.server,
            "params": dict(self.params),
        }
        fields.update(overrides)
        return contract.Frame(**fields)

    def summary(self) -> str:
        """One line, naming the key read, the parts, and anything fitted rather than read."""
        keys = sorted({p.key for p in self.parts})
        head = f"{self.function}: {len(self.parts)} part(s) from {', '.join(keys)}, z_source={self.z_source}"
        if self.derived_by:
            head += f" (fitted by {self.derived_by})"
        return head


# ---------------------------------------------------------------------------------------------
# The table. Every claim below was read out of the worker named in ``evidence``.
# ---------------------------------------------------------------------------------------------

_ROWS: tuple[AlignerAdapter, ...] = (
    AlignerAdapter(
        function="paste_pairwise_align",
        server="paste",
        writes_key="spatial_aligned / spatial_3d_aligned (in paste_pairwise_aligned_slice_<i>.h5ad)",
        role="aligned",
        original_recoverable=True,
        original_from="same_file",
        z_source="asked_user",
        seed_exposed=True,
        status=STATUS_ADAPTED,
        output_key="aligned_slice_<i>",
        evidence="tools/paste_worker.py::_preserve_and_label, ::_load_slices; paste/visualization.py:62-66",
        notes=(
            "stack_slices_pairwise copies each slice and replaces obsm['spatial'] with the aligned "
            "coordinates. The worker puts the input back into obsm['spatial'] and writes the aligned "
            "values beside it: obsm['spatial_3d_aligned'] when a z spacing was declared (third column "
            "= slice index x z_spacing, the caller's number), obsm['spatial_aligned'] when it was not. "
            "Both are read from the same file. A file written before the worker restored the input "
            "carries neither key; its obsm['spatial'] is PASTE's overwrite, and the adapter recovers "
            "the originals from the recorded input paths and refuses without them. The Procrustes "
            "step also recentres every slice on its own transport mass (paste/visualization.py:174), "
            "so the reference slice moves too and a before/after offset is not zero by construction. "
            "Since 2026-09-30 the worker leaves out spots with obs['in_tissue'] == 0 as it reads each "
            "slice (params.in_tissue_dropped_per_slice), so an output file holds the in-tissue spots of "
            "its input, not all of them. random_seed is accepted and ignored in this mode "
            "(params.ignored): pairwise_align is deterministic."
        ),
    ),
    AlignerAdapter(
        function="paste_center_align",
        server="paste",
        writes_key="spatial_aligned / spatial_3d_aligned (in paste_center_aligned_slice_<i>.h5ad)",
        role="aligned",
        original_recoverable=True,
        original_from="same_file",
        z_source="asked_user",
        seed_exposed=True,
        status=STATUS_ADAPTED,
        output_key="aligned_slice_<i>",
        evidence="tools/paste_worker.py::_preserve_and_label, ::_load_slices; paste/visualization.py:62-66",
        notes=(
            "Same restore-and-label as the pairwise mode, around a consensus centre rather than "
            "pairwise: the input is back in obsm['spatial'] and the answer is in "
            "obsm['spatial_aligned'] or obsm['spatial_3d_aligned']. paste_center_slice.h5ad is that "
            "consensus object: it has its own rows and is not one of the sections, so it is not read "
            "as a part. center_align is additionally run after filter_for_common_genes, which mutates "
            "the loaded slices but not the files on disk. As in the pairwise mode, spots with "
            "obs['in_tissue'] == 0 are left out as each slice is read, so an output file holds the "
            "in-tissue spots of its input."
        ),
    ),
    AlignerAdapter(
        function="moscot_run",
        server="moscot",
        writes_key="moscot_spatial_warp",
        role="aligned",
        original_recoverable=True,
        original_from="same_file",
        z_source="unknown",
        seed_exposed=False,
        status=STATUS_ADAPTED,
        output_key="adata_aligned_h5ad",
        evidence=(
            "tools/moscot_worker.py run_alignment (ap.align) and main (params.batch_key, "
            "params.reference_batch_used); moscot/problems/space/_mixins.py:54-56,138-161"
        ),
        notes=(
            "obsm['spatial'] survives, because align() writes a new key. Every alignment run calls "
            "ap.align() and writes adata_alignment_aligned.h5ad: onto reference_batch when it was "
            "named, otherwise onto the first level of obs[batch_key] in order of appearance, and the "
            "section held fixed is params.reference_batch_used. In the default 'warp' mode the "
            "reference keeps its own obsm['spatial'] and every other section is a barycentric "
            "projection onto it, so the result is non-rigid; uns['moscot_spatial_warp']"
            "['alignment_metadata'] is written only in 'affine' mode, so there is no transform to "
            "record here. All sections are rows of one merged object, told apart by "
            "obs[params.batch_key]; the worker records batch_key since 2026-09-30, and an output whose "
            "params do not name it is refused rather than read as one plane. Workers before "
            "2026-09-21 skipped align() unless a reference_batch was named, and then wrote no "
            "coordinates at all."
        ),
        refusal_reason=(
            "moscot_run's output_files carry no adata_aligned_h5ad: an alignment run always writes "
            "adata_alignment_aligned.h5ad (ap.align() runs onto reference_batch, or onto the first "
            "level of batch_key when none was named), so its absence means the run was not "
            "problem_type='alignment' or stopped before align()"
        ),
        refusal_remedy=(
            "Pass the output_files, or the whole result, of a moscot_run(problem_type='alignment') run "
            "that finished; its answer is obsm['moscot_spatial_warp'] of adata_alignment_aligned.h5ad."
        ),
    ),
    AlignerAdapter(
        function="st_gears_reconstruct_3d",
        server="st_gears",
        writes_key="spatial_elas_reuse / spatial_elas / spatial_rigid",
        role="aligned",
        original_recoverable=True,
        original_from="same_file",
        z_source="unknown",
        seed_exposed=True,
        status=STATUS_ADAPTED,
        output_key="aligned_h5ad",
        evidence="tools/st_gears_worker.py add_ordinal_z / build_frame; st_gears/recons.py:150,519; st_gears/granularity_adjusting.py:89-103",
        notes=(
            "The aligned coordinates are ST-GEARS's own three-column keys, and their third column is "
            "column 2 of the obsm['spatial'] ST-GEARS was handed, copied back unchanged (recons.py:150). "
            "ST-GEARS indexes that column unconditionally, so for a two-column input (every converter's "
            "shape) the worker hands it a working copy whose z is the slice ordinal and writes the "
            "two-column obsm['spatial'] back; params.z_column_added says which happened. That z is an "
            "ordinal, not a distance, so z_source stays 'unknown' unless a z is passed. With binning on "
            "(the default), spots outside the binned grid's convex hull are NaN in spatial_elas_reuse "
            "(obs['st_gears_xy_source'] == 'none', counted in summary.n_spots_without_aligned_xy); "
            "with allow_nearest_fallback=True the worker fills them in its own obsm['spatial_3d_aligned'] "
            "/ obsm['st_gears_xyz'] (marked 'nearest') and leaves ST-GEARS's key as ST-GEARS wrote it. "
            "Only the sections in start_idx..end_idx are in the output. A failed anndata.concat is an "
            "error, not a file without coordinates."
        ),
    ),
    AlignerAdapter(
        function="spatrio_align_multiomics",
        server="spatrio",
        writes_key="none -- spatrio_aligned.csv is a (spot, cell, value) mapping table",
        role="",
        original_recoverable=False,
        z_source="unknown",
        seed_exposed=True,
        status=STATUS_REFUSED,
        output_key="alignment_csv",
        evidence="tools/spatrio_worker.py:435-484; spatrio/spatrio.py:180-190",
        notes=(
            "spatrio.ot_alignment returns a long-form DataFrame of spot-cell transport weights, not "
            "an AnnData and not coordinates; the worker publishes that table as spatrio_aligned.csv under "
            "output_files['alignment_csv'] ('aligned_h5ad' is a deprecated alias of the same CSV). Coordinates "
            "come from spatrio.assign_coord, which the worker does not call. It is also not a "
            "serial-section aligner: it maps single cells onto the spots of one slice."
        ),
        refusal_reason=(
            "SpaTrio wrote a spot-to-cell transport table, not coordinates -- ot_alignment returns a "
            "(spot, cell, value) DataFrame and the worker never calls assign_coord"
        ),
        refusal_remedy=(
            "Placing those cells needs spatrio.assign_coord(adata1, adata2, out_data), which the "
            "worker would have to call and write; and even then the result is cells within one "
            "section, not two sections in one frame."
        ),
    ),
    AlignerAdapter(
        function="gpsa_align_slices",
        server="gpsa",
        writes_key="spatial_aligned",
        role="aligned",
        original_recoverable=True,
        original_from="same_file",
        z_source="unknown",
        seed_exposed=True,
        status=STATUS_ADAPTED,
        output_key="slice1_aligned_h5ad, slice2_aligned_h5ad",
        evidence=(
            "tools/gpsa_worker.py::run_alignment (_from_unit_box; obsm['spatial_aligned'], "
            "uns['gpsa_alignment']); gpsa/models/vgpsa.py:262-273"
        ),
        notes=(
            "obsm['spatial'] survives in both output files. Since 2026-09-30 the worker leaves out "
            "spots with obs['in_tissue'] == 0 before training (params.in_tissue_dropped_per_slice), so "
            "each output file holds the in-tissue spots of its input. The worker inverts its joint [0, 1] "
            "training frame itself and writes obsm['spatial_aligned'] in the input's units, marked by "
            "uns['gpsa_alignment']['coordinate_frame'] == 'input' (the [0, 1] values are kept under "
            "obsm['spatial_aligned_normalized']); such files are read as written and carry no "
            "derived_by, and with n_spatial_dims=3 the third column is kept as z. Files written before "
            "that carry no marker and hold the [0, 1] box; for those the adapter restores the units by "
            "recomputing the min and range from the two surviving obsm['spatial'] arrays, which is "
            "exact, stamps derived_by and says so in its notes. Slice 1 is the fixed view: upstream "
            "VariationalGPSA returns its observed coordinates as its G_means, so it does not move."
        ),
    ),
    AlignerAdapter(
        function="stalign_align_points",
        server="stalign",
        writes_key="stalign_aligned_points.csv, columns (y, x)",
        role="aligned",
        original_recoverable=True,
        original_from="input_slices",
        z_source="unknown",
        seed_exposed=False,
        status=STATUS_ADAPTED,
        output_key="aligned_points_csv",
        needs_input_slices=True,
        evidence=(
            "tools/stalign_worker.py::run_align_points (stalign_aligned_points.csv, columns y, x); "
            "MCP_server/mcp_config.yaml stalign_align_points"
        ),
        notes=(
            "STalign consumes and produces (y, x)-ordered CSV point clouds and never touches an h5ad, "
            "so the coordinates arrive with the axes swapped relative to obsm['spatial'] and with no "
            "identifier of any kind. The adapter swaps them back to (x, y) and joins positionally to "
            "the slice the caller names, refusing unless the row counts agree -- there is no key to "
            "join on, so an unequal count is a silent mis-assignment waiting to happen. The row order "
            "is the source CSV's, which is the caller's responsibility to have preserved. Nothing is "
            "fitted: the coordinates are STalign's own, so the frames carry no derived_by."
        ),
    ),
    AlignerAdapter(
        function="stalign_align_to_image",
        server="stalign",
        writes_key="stalign_aligned_points.csv, columns (y, x), in image pixel space",
        role="external",
        original_recoverable=True,
        original_from="input_slices",
        z_source="unknown",
        seed_exposed=False,
        status=STATUS_ADAPTED,
        output_key="aligned_points_csv",
        frame_key=contract.EXTERNAL_PREFIX + "image",
        needs_input_slices=True,
        evidence="tools/stalign_worker.py::run_align_to_image (stalign_aligned_points.csv, columns y, x)",
        notes=(
            "Same file shape and the same positional join as align_points, but the target frame is the "
            "pixel grid of the supplied tissue image, not another section. It takes the contract's "
            "'external' role for that reason: folding it into 'aligned' would let Phase 2 report a "
            "registration to a photograph as a section-to-section alignment. As with align_points, "
            "nothing is fitted here and the frames carry no derived_by."
        ),
    ),
    AlignerAdapter(
        function="slat_align_slices",
        server="slat",
        writes_key="slat_matching.csv -- matched cell indices, no coordinates",
        role="aligned",
        original_recoverable=True,
        original_from="input_slices",
        z_source="unknown",
        seed_exposed=False,
        status=STATUS_DERIVED,
        output_key="matching_csv",
        needs_input_slices=True,
        evidence="tools/slat_worker.py:_match; scSLAT/model/utils.py:235-259,284",
        notes=(
            "scSLAT matches cells; it never moves one. The adapter fits a 2D similarity from the "
            "matched pairs with geometry._umeyama and stamps derived_by, so nobody reads the result "
            "as SLAT's own coordinates. The matching has one row per spot of the smaller slice "
            "(slice 2 when the sizes are equal), and that slice's column is the enumeration "
            "0..len-1; the worker labels each column by its slice. Workers before 2026-09-29 wrote "
            "the enumeration under slice1_idx whatever the sizes, so whenever n1 >= n2 their two "
            "columns held each other's contents. The adapter recognises that layout from the file "
            "itself (n1 >= n2 with the enumeration under slice1_idx) and swaps it back, so a CSV "
            "from either worker reads correctly."
        ),
    ),
    AlignerAdapter(
        function="spiral_integrate",
        server="spiral",
        writes_key="spatial (written on a newly built merged object; not an alignment)",
        role="",
        original_recoverable=False,
        z_source="unknown",
        seed_exposed=False,
        status=STATUS_REFUSED,
        output_key="integrated_h5ad",
        evidence="tools/spiral_worker.py::_integrate (merged object, obsm['spatial'] from the coord CSVs); ::write_spiral_csvs",
        notes=(
            "The integrate path corrects expression, not geometry. The merged object it builds carries "
            "obsm['spatial'], which looks like a common frame and is not one: it is each slice's own "
            "input coordinates concatenated, each still in its own frame, read back from the per-slice "
            "CSVs the worker wrote. Reading it as an aligned frame would report a completely "
            "unregistered stack as aligned, which is the failure this package exists to prevent. Its "
            "obs_names are also prefixed s<i>_ and no longer match the input files."
        ),
        refusal_reason=(
            "spiral_integrate performs expression integration, not coordinate alignment -- the "
            "obsm['spatial'] on its merged object is the per-slice input coordinates concatenated, "
            "each still in its own frame"
        ),
        refusal_remedy="Use spiral_align, which runs the same integration and then fits coordinates with GW optimal transport.",
    ),
    AlignerAdapter(
        function="spiral_align",
        server="spiral",
        writes_key="spatial_aligned",
        role="aligned",
        original_recoverable=False,
        original_from="input_slices",
        z_source="unknown",
        seed_exposed=False,
        status=STATUS_ADAPTED,
        output_key="aligned_h5ad",
        evidence="tools/spiral_worker.py::run_alignment (obsm['spatial_aligned'], obs['spiral_aligned'])",
        notes=(
            "Written on the merged object, with obs['batch'] naming the slice. Only clusters shared by "
            "both slices are moved. Since 2026-09-29 the worker writes every other slice_1 spot as NaN "
            "and marks it False in obs['spiral_aligned'], and the reader leaves those rows out; an "
            "output written before then holds their input coordinates instead, so its frame is a "
            "mixture. The merged object's obs_names carry an s0_/s1_ prefix, so they do not join to "
            "the input files by name. Its obsm['spatial'] is the concatenated per-slice originals, not "
            "this object's own original, which is why original_recoverable is False. The mapping is "
            "the worker's own per-cluster fused Gromov-Wasserstein, not SPIRAL's CoordAlignment. "
            "SPIRAL's seed is fixed at 0 inside the worker (spiral_worker.py::_integrate) and is not reachable "
            "from the portal."
        ),
    ),
    AlignerAdapter(
        function="stacker_register",
        server="stacker",
        writes_key="stacker_warped.nii.gz and a warp field -- image space, not coordinates",
        role="",
        original_recoverable=False,
        z_source="unknown",
        seed_exposed=False,
        status=STATUS_REFUSED,
        output_key="warped_image",
        evidence="tools/stacker_worker.py:153-481,544-549",
        notes=(
            "STACKer registers one image to another and writes a warped NIfTI plus the transform "
            "(an ANTs .mat or .nii.gz field, or a VoxelMorph displacement .npy). Every one of those is "
            "in voxel indices of the fixed image. Nothing in the worker's output says how a voxel "
            "index relates to a spot coordinate, and the two are not the same thing even for the same "
            "slide, so there is no arithmetic that lands a cell in the warped frame."
        ),
        refusal_reason=(
            "STACKer's answer is a voxel warp of an image, and mapping it onto cells needs an "
            "image-to-coordinate affine that the worker never writes"
        ),
        refusal_remedy=(
            "It would take three things the run does not record: the pixel size and origin of both the "
            "moving and the fixed image, the scalefactor tying obsm['spatial'] to that pixel grid "
            "(Visium's tissue_hires_scalef, or its equivalent), and the axis order the NIfTI was "
            "written in. With those a caller could resample the displacement field at each spot; "
            "without them any mapping is a guess at the scale."
        ),
    ),
    AlignerAdapter(
        function="run_spacel_scube",
        server="spacel",
        writes_key="spatial_aligned",
        role="aligned",
        original_recoverable=True,
        original_from="same_file",
        z_source="unknown",
        seed_exposed=False,
        status=STATUS_ADAPTED,
        output_key="aligned_h5ads",
        evidence=(
            "tools/spacel_worker.py::run_scube (aligned_h5ads, obsm['spatial_aligned']); "
            "SPACEL/Scube/alignment.py:179-181,214-248"
        ),
        notes=(
            "obsm['spatial'] survives, and Scube writes obsm['spatial_aligned'] as a pandas DataFrame "
            "with columns X/Y -- read it through numpy, not by column name. Two facts the tool's own "
            "description obscures: the aligned coordinates are median-centred per slice, so the frame "
            "does not share an origin with obsm['spatial'], and despite '3D alignment' in its "
            "description the output has as many columns as the input -- two, for an ordinary Visium "
            "slide, with no z anywhere. Scube.align does take a seed, fixed at its default of 42, and "
            "the portal exposes no parameter for it. It also leaves obsm['spatial_pair'] behind, which "
            "is the centred input rather than the answer."
        ),
    ),
    AlignerAdapter(
        function="paste2_partial_align",
        server="paste2",
        writes_key="spatial_3d_aligned / spatial_aligned",
        role="aligned",
        original_recoverable=True,
        original_from="same_file",
        z_source="asked_user",
        seed_exposed=True,
        status=STATUS_ADAPTED,
        output_key="aligned_slice_<i>",
        evidence="tools/paste2_worker.py::_preserve_and_label",
        notes=(
            "Written against the worker rather than inferred. PASTE2's own stacking step is PASTE's "
            "-- partial_stack_slices_pairwise replaces obsm['spatial'] in the copies it returns -- "
            "so the worker snapshots the input and restores it, and the aligned values go to "
            "obsm['spatial_3d_aligned'] when a z spacing was declared and obsm['spatial_aligned'] "
            "when it was not. Verified on a real run: obsm['spatial'] byte-identical, two columns, "
            "and the 3D frame carrying z=0 and z=50. The overlap fraction actually used is in "
            "overlap_fractions.csv and in params, per pair, because it is a claim about the tissue "
            "and not only a setting."
        ),
    ),
    AlignerAdapter(
        function="paste2_estimate_overlap",
        server="paste2",
        writes_key="",
        role="",
        original_recoverable=True,
        z_source="unknown",
        seed_exposed=False,
        status=STATUS_REFUSED,
        evidence="tools/paste2_worker.py::run_estimate",
        notes="A measurement, not an alignment: it writes overlap_fractions.csv and touches no coordinate.",
        refusal_reason=(
            "paste2_estimate_overlap measures how much each adjacent pair overlaps and aligns "
            "nothing, so there is no aligned frame to read back"
        ),
        refusal_remedy="Call paste2_partial_align to actually align, using the fraction this reported.",
    ),
    AlignerAdapter(
        function="cast_align_slices",
        server="cast",
        writes_key="spatial_3d_aligned / spatial_aligned",
        role="aligned",
        original_recoverable=True,
        original_from="same_file",
        z_source="asked_user",
        seed_exposed=True,
        status=STATUS_ADAPTED,
        output_key="aligned_slice_<i>",
        evidence="tools/cast_worker.py::run_align",
        notes=(
            "CAST_STACK returns coords_final, which replaces the input frame; the worker snapshots "
            "the input and restores it, putting the registered coordinates in a key of their own. "
            "Verified on a real CPU run. The registration is onto ONE reference section, named in "
            "params as `reference`, and every other section is warped onto it -- so the reference "
            "is part of the result, not a detail. ffd_iterations=0 makes the output affine-only."
        ),
    ),
)

#: function name -> its row. The table is the interface; the tuple above is its order.
ADAPTERS: dict[str, AlignerAdapter] = {a.function: a for a in _ROWS}

#: Functions whose names or descriptions look like alignment but which do not register serial
#: sections to each other. Each is excluded by name and with its reason, so the completeness check
#: stays data-driven without flagging tools that were never in scope.
NOT_SLICE_ALIGNMENT: dict[str, str] = {
    "novosparc_reconstruct_spatial": (
        "reconstructs coordinates for dissociated cells de novo; there are no sections to align"
    ),
    "run_spacel_splane": "identifies spatial domains within one slide; it writes no coordinates",
    # The 3D-diagnosis portals. They are alignment-family tools -- they carry task type
    # spatial_alignment and their whole purpose is to decide whether an alignment is needed -- but
    # they are on the asking side of the question, not the answering side. A diagnosis measures a
    # stack and writes a report; it moves no coordinate, so there is no aligned frame for an
    # adapter to locate. Without these two names here the completeness check flags this package's
    # own tools for not adapting themselves.
    "diagnose_3d_stack": (
        "measures a stack and writes a report; it moves no coordinate, so there is no aligned frame to read back"
    ),
    "list_aligner_adapters": "prints this very table; it is not an aligner",
    "inspect_3d_coordinates": "reads an object's existing coordinate keys; it writes nothing",
    "explain_3d_contract": "returns the contract's own rules; it takes no data at all",
    # A plot of an alignment is not an alignment. The name rule above matches on the word, which is
    # the same mistake `resolve_category("plot_deconvolution") == "deconvolution"` made on the viz
    # portal a day earlier: a drawing function named after an analysis read as one. This tool DRAWS
    # a before and an after that some aligner produced; the frames it reads are already located by
    # the row belonging to whichever tool wrote them.
    "plot_alignment_qc": (
        "draws an alignment somebody else performed; it writes a figure, not a coordinate, so there "
        "is no aligned frame of its own for an adapter to locate"
    ),
}


def table() -> list[dict[str, Any]]:
    """Every row, in table order, as flat records."""
    return [a.row() for a in _ROWS]


def describe_table() -> str:
    """The table as text, for a report or a terminal."""
    rows = table()
    cols = ["function", "server", "status", "role", "z_source", "original_recoverable", "seed_exposed", "writes_key"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    out = [" ".join(c.ljust(widths[c]) for c in cols), " ".join("-" * widths[c] for c in cols)]
    out += [" ".join(str(r[c]).ljust(widths[c]) for c in cols) for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# Reading what a run left behind
# ---------------------------------------------------------------------------------------------


def _unwrap(output_files: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a caller's argument into (output_files, params).

    A worker's whole result dict is accepted as well as its ``output_files`` block, because that is
    what a portal has in hand and re-deriving the input paths from it is exactly what the PASTE row
    needs.
    """
    if not isinstance(output_files, dict):
        return {}, {}
    inner = output_files.get("output_files")
    if isinstance(inner, dict):
        params = output_files.get("params")
        return dict(inner), dict(params) if isinstance(params, dict) else {}
    return dict(output_files), {}


def _numbered(files: dict[str, Any], prefix: str) -> list[str]:
    """Paths under ``prefix0``, ``prefix1``, ... in numeric order, not in string order."""
    found: list[tuple[int, str]] = []
    for key, value in files.items():
        if not isinstance(key, str) or not key.startswith(prefix) or not isinstance(value, str):
            continue
        tail = key[len(prefix) :]
        if tail.isdigit():
            found.append((int(tail), value))
    return [p for _, p in sorted(found)]


def _listed(files: dict[str, Any], key: str) -> list[str]:
    """The path or list of paths under one key; an empty list when it is absent or None."""
    value = files.get(key)
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v]
    return []


def _open(path: str) -> Any:
    """Open an h5ad for its obs and obsm only.

    Not backed, deliberately. A backed handle returned from here outlives this function, and every
    caller reads two or three obsm keys and then drops it -- so the HDF5 lock on the aligner's own
    output is held until the object is collected, and whatever rewrites that path next fails with
    errno 11. ``test/test_h5ad_readers_release_the_file.py`` is the check that caught it.

    The frames are (n_obs, 2 or 3) and the obs table is small, so the saving a backed read would
    have bought is the count matrix, which nothing here touches.
    """
    import anndata

    return anndata.read_h5ad(path)


def _obsm(adata: Any, key: str) -> Any:
    """``obsm[key]`` as a float64 array, or None. SPACEL stores a DataFrame under its key."""
    import numpy as np

    obsm = getattr(adata, "obsm", None)
    if obsm is None or key not in obsm:
        return None
    try:
        arr = np.asarray(getattr(obsm[key], "values", obsm[key]), dtype="float64")
    except Exception:
        return None
    return arr if arr.ndim == 2 and arr.shape[1] >= 2 else None


def _names(adata: Any) -> tuple[str, ...]:
    return tuple(str(n) for n in adata.obs_names)


def _labels(adata: Any, column: str) -> tuple[str, ...]:
    if not column:
        return ()
    obs = getattr(adata, "obs", None)
    if obs is None or column not in obs:
        return ()
    return tuple(str(v) for v in obs[column])


def _z_plan(z: Any, n_planes: int) -> tuple[list[float] | None, str, str]:
    """Turn the caller's ``z`` into one value per plane, and say where that value came from.

    A scalar is a spacing between adjacent planes, applied in the order they were read -- which is
    the order the caller passed the slices to the tool. With a single plane a spacing has nothing to
    span, so the scalar is that plane's own z. A sequence is taken literally, one value per plane.
    ``None`` stays None all the way through: this package has no default spacing and does not
    acquire one here.
    """
    import numbers

    if z is None:
        return None, "unknown", ""
    # A string was iterated character by character -- '50' became z = [5, 0], labelled as the
    # caller's -- and a numpy integer spacing was refused as neither a number nor a sequence
    # (hunt 2026-09-30, u21-3d-17).
    if isinstance(z, (str, bytes)):
        return (
            None,
            "unknown",
            f"z was given as text ({z!r}); pass a number (the spacing) or one number per plane, so no z was assigned",
        )
    if isinstance(z, numbers.Real) and not isinstance(z, bool) and type(z).__name__ not in ("bool_", "bool"):
        value = float(z)
        if n_planes == 1:
            return [value], "asked_user", f"z set to {value:g} for the single plane read, as the caller supplied it"
        return (
            [i * value for i in range(n_planes)],
            "asked_user",
            f"z assigned as {value:g} per step in the order the planes were read; the caller supplied the spacing",
        )
    try:
        values = [float(v) for v in z]
    except (TypeError, ValueError):
        return None, "unknown", "z was neither a number nor a sequence of numbers, so no z was assigned"
    if len(values) != n_planes:
        return None, "unknown", f"z had {len(values)} values for {n_planes} planes, so no z was assigned"
    return values, "asked_user", "z taken per plane from the sequence the caller supplied"


def read_aligned(
    function: str,
    output_files: Any,
    *,
    input_slices: list[str] | None = None,
    z: Any = None,
) -> AlignedFrames | contract.Refusal:
    """The aligned coordinates a run produced, in the contract's terms -- or why there are none.

    ``output_files`` is the worker's ``output_files`` mapping, or its whole result dict.
    ``input_slices`` is the list of files the run was given, needed by the four functions whose
    output does not carry its own originals or its own identifiers. ``z`` is a spacing or one
    value per part; it is never invented, and without it the frames are two-dimensional and say so.
    """
    adapter = ADAPTERS.get(function)
    if adapter is None:
        return contract.Refusal(
            what=function,
            reason="no adapter row for this function",
            remedy=f"Known functions: {', '.join(sorted(ADAPTERS))}.",
        )
    if adapter.status in (STATUS_REFUSED, STATUS_PENDING):
        return adapter.refusal()

    files, params = _unwrap(output_files)
    if not files:
        return contract.Refusal(
            what=function,
            reason="no output_files were given, so there is nothing to read",
            remedy=f"Pass the worker's output_files mapping; this row's answer is under {adapter.output_key!r}.",
        )
    slices = list(input_slices) if input_slices else [str(p) for p in (params.get("input_slices") or [])]
    if adapter.needs_input_slices and not slices:
        return contract.Refusal(
            what=function,
            reason=f"this output cannot be interpreted without the input slices ({adapter.original_from})",
            remedy="Pass input_slices=[...] in the order the run received them.",
        )
    return _READERS[function](adapter, files, slices, z, params)


def _frames(
    adapter: AlignerAdapter,
    parts: list[AlignedPart],
    z: Any,
    params: dict[str, Any],
    notes: list[str],
    derived_by: str = "",
    file_z_source: str = "unknown",
) -> AlignedFrames | contract.Refusal:
    """Attach the caller's z to the parts that were read, and assemble the result.

    ``file_z_source`` is where a z the file already carried came from, for the readers that can
    say (PASTE's third column is the spacing the caller declared at run time); it applies only
    when the caller passes no z of their own.

    A plane is a section, which is not the same thing as a part: one file per slice gives a plane
    per part, but a merged object is one part holding several. Assigning per part in that second
    case would put every section of an atlas at the same z and call the result a 3D frame, so the
    merged case is planned over the distinct slice labels instead, in the order they first appear
    -- never sorted, because four Zhuang sections share a z exactly and sorting invents an order
    that is not in the tissue.
    """
    import numpy as np

    if not parts:
        return contract.Refusal(
            what=adapter.function,
            reason="no coordinates were found in the files this row points at",
            remedy=f"Expected {adapter.writes_key} under output_files[{adapter.output_key!r}].",
        )
    merged = len(parts) == 1 and bool(parts[0].slice_labels)
    order: list[str] = []
    if merged:
        for label in parts[0].slice_labels:
            if label not in order:
                order.append(label)
    values, z_source, note = _z_plan(z, len(order) if merged else len(parts))
    if note:
        notes.append(note)
    if values is not None:
        if any(p.z is not None for p in parts):
            notes.append("the z the file already carried was replaced by the one the caller supplied")
        if merged:
            by_label = dict(zip(order, values, strict=True))
            per_row = np.array([by_label[label] for label in parts[0].slice_labels], dtype="float64")
            parts = [_with_z(parts[0], per_row)]
        else:
            parts = [_with_z(p, np.full(len(p.xy), values[i], dtype="float64")) for i, p in enumerate(parts)]
    elif any(p.z is not None for p in parts):
        # A part that arrived with its own z keeps it; the source is whatever the file already had,
        # which this module cannot characterise unless its reader said where it came from.
        z_source = file_z_source
    return AlignedFrames(
        function=adapter.function,
        adapter=adapter,
        parts=tuple(parts),
        z_source=z_source,
        derived_by=derived_by,
        params=dict(params),
        notes=tuple(notes),
    )


def _with_z(part: AlignedPart, z: Any) -> AlignedPart:
    """The same part carrying a different z. Parts are frozen, so this is the only way to set one."""
    return AlignedPart(
        label=part.label,
        path=part.path,
        key=part.key,
        xy=part.xy,
        z=z,
        obs_names=part.obs_names,
        original_xy=part.original_xy,
        slice_labels=part.slice_labels,
    )


#: Where the PASTE, PASTE2 and CAST workers leave the aligned coordinates (``_preserve_and_label`` /
#: ``run_align``): three columns when a z spacing was declared, two when it was not.
#: ``obsm['spatial']`` of the same file is the input, which each worker restores after the tool's
#: own stacking overwrites it.
PASTE_ALIGNED_KEYS = ("spatial_3d_aligned", "spatial_aligned")


def _read_paste(
    adapter: AlignerAdapter, files: dict[str, Any], slices: list[str], z: Any, params: dict[str, Any]
) -> AlignedFrames | contract.Refusal:
    """PASTE's (and PASTE2's and CAST's) aligned slices, read from the key the worker writes them to.

    The input is back in ``obsm['spatial']`` of each output file and the answer sits beside it, so
    both come from one file. A PASTE file from before the worker restored the input carries neither
    aligned key: its ``obsm['spatial']`` is PASTE's overwrite, and only for such a file are the
    originals read from the recorded input paths.
    """
    out_paths = _numbered(files, "aligned_slice_")
    if not out_paths:
        return contract.Refusal(
            what=adapter.function,
            reason="no aligned_slice_<i> entries in output_files",
            remedy="The worker writes one h5ad per slice under aligned_slice_<i>; pass the mapping it emitted.",
        )
    if slices and len(out_paths) != len(slices):
        return contract.Refusal(
            what=adapter.function,
            reason=f"{len(out_paths)} aligned files against {len(slices)} input slices",
            remedy="The originals are matched to the outputs by position, so the two lists must agree.",
        )
    parts: list[AlignedPart] = []
    legacy: list[int] = []
    keys_read: list[str] = []
    for i, out_path in enumerate(out_paths):
        in_path = slices[i] if slices else ""
        aligned = _open(out_path)
        key = next((k for k in PASTE_ALIGNED_KEYS if _obsm(aligned, k) is not None), "")
        zcol = None
        if key:
            arr = _obsm(aligned, key)
            if key == "spatial_3d_aligned" and arr.shape[1] >= 3:
                zcol = arr[:, 2]
            original = _obsm(aligned, contract.ORIGINAL_KEY)
            if original is None:
                return contract.Refusal(
                    what=adapter.function,
                    reason=f"{out_path} has obsm[{key!r}] but no obsm['spatial'], so its original is not in the file",
                )
        else:
            arr = _obsm(aligned, contract.ORIGINAL_KEY)
            if arr is None:
                return contract.Refusal(
                    what=adapter.function,
                    reason=(
                        f"{out_path} has none of obsm['spatial_3d_aligned'], obsm['spatial_aligned'] "
                        "or obsm['spatial'] to read the aligned coordinates from"
                    ),
                )
            if not in_path:
                return contract.Refusal(
                    what=adapter.function,
                    reason=(
                        f"{out_path} carries no obsm['spatial_aligned'] or obsm['spatial_3d_aligned']: it "
                        "was written before the worker restored the input, so its obsm['spatial'] is "
                        "the aligner's own overwrite and the originals exist only in the input files"
                    ),
                    remedy="Pass input_slices=[...] in the order the run received them, or the worker's whole result.",
                )
            original = _obsm(_open(in_path), contract.ORIGINAL_KEY)
            if original is None:
                return contract.Refusal(
                    what=adapter.function,
                    reason=f"input slice {in_path} has no obsm['spatial'], so the original cannot be recovered",
                )
            key = contract.ORIGINAL_KEY
            legacy.append(i)
        if len(original) != len(arr):
            return contract.Refusal(
                what=adapter.function,
                reason=f"{out_path} holds {len(arr)} aligned rows against {len(original)} original rows",
                remedy="The aligner preserves row order and count, so a mismatch means these are not the same run.",
            )
        if key not in keys_read:
            keys_read.append(key)
        parts.append(
            AlignedPart(
                label=(_stem(in_path) if in_path else _stem(out_path)) or f"slice_{i}",
                path=out_path,
                key=key,
                xy=arr[:, :2],
                z=zcol,
                obs_names=_names(aligned),
                original_xy=original[:, :2],
            )
        )
    notes: list[str] = []
    if len(legacy) < len(parts):
        read = " / ".join(f"obsm[{k!r}]" for k in keys_read if k != contract.ORIGINAL_KEY)
        notes.append(
            f"the aligned coordinates were read from {read} and the originals from obsm['spatial'] of the "
            "same file, which the worker restores after the tool's own stacking overwrites it"
        )
    if legacy:
        notes.append(
            f"slice(s) {legacy} were written before the worker restored obsm['spatial']: their aligned "
            "coordinates are the aligner's own overwrite of that key, and their originals were read from the "
            "recorded input paths"
        )
    has_file_z = any(p.z is not None for p in parts)
    if has_file_z and z is None:
        notes.append(
            "z is the third column of obsm['spatial_3d_aligned']: the slice index times the z_spacing "
            "the caller declared when the run was made"
            + (f" (params.z_spacing={params['z_spacing']})" if params.get("z_spacing") else "")
        )
    return _frames(adapter, parts, z, params, notes, file_z_source="asked_user" if has_file_z else "unknown")


def _gpsa_frame(adata: Any) -> str:
    """What frame a GPSA output says its ``obsm['spatial_aligned']`` is in; '' when it does not say.

    Files written before the worker inverted its [0, 1] training frame carry no record, and are in
    that box.
    """
    try:
        return str(adata.uns["gpsa_alignment"]["coordinate_frame"])
    except Exception:
        return ""


def _read_gpsa(
    adapter: AlignerAdapter, files: dict[str, Any], slices: list[str], z: Any, params: dict[str, Any]
) -> AlignedFrames | contract.Refusal:
    import numpy as np

    paths = _listed(files, "slice1_aligned_h5ad") + _listed(files, "slice2_aligned_h5ad")
    if len(paths) != 2:
        return contract.Refusal(
            what=adapter.function,
            reason="GPSA writes exactly two aligned slices and both are needed",
            remedy="Pass slice1_aligned_h5ad and slice2_aligned_h5ad from the worker's output_files.",
        )
    read = [_open(p) for p in paths]
    aligned = [_obsm(a, "spatial_aligned") for a in read]
    original = [_obsm(a, contract.ORIGINAL_KEY) for a in read]
    if any(a is None for a in aligned) or any(o is None for o in original):
        return contract.Refusal(
            what=adapter.function,
            reason="one of the GPSA outputs is missing obsm['spatial_aligned'] or obsm['spatial']",
        )
    notes: list[str] = []
    # Only the legacy [0, 1] restore below computes anything here; every other branch reads what
    # GPSA wrote, and stamping this module on it would claim a fit that never happened.
    derived_by = ""
    zs: list[Any] = [None, None]
    if all(_gpsa_frame(a) == "input" for a in read):
        xy = [a[:, :2] for a in aligned]
        notes.append(
            "GPSA wrote obsm['spatial_aligned'] in the input's units "
            "(uns['gpsa_alignment']['coordinate_frame'] == 'input'); read as written"
        )
        n_cols = min(a.shape[1] for a in aligned)
        if n_cols >= 3:
            zs = [a[:, 2] for a in aligned]
            notes.append(
                f"obsm['spatial_aligned'] has {n_cols} columns (n_spatial_dims={n_cols}): GPSA aligned in "
                "three dimensions, and its third column was kept as z -- GPSA's own aligned value in the "
                "input's units, warped from the input's third column, not a spacing anybody declared"
                + ("; columns past the third were not read" if n_cols > 3 else "")
            )
    elif aligned[0].shape[1] != 2:
        notes.append(
            f"the aligned array has {aligned[0].shape[1]} columns (n_spatial_dims != 2), so the "
            f"[0, 1] normalisation was not inverted and these numbers are not in the input's units"
        )
        xy = [a[:, :2] for a in aligned]
    else:
        pooled = np.vstack([o[:, :2] for o in original])
        lo = pooled.min(axis=0)
        span = pooled.max(axis=0) - lo
        span[span == 0] = 1.0
        xy = [a[:, :2] * span + lo for a in aligned]
        derived_by = DERIVED_BY
        notes.append(
            "GPSA's aligned coordinates are jointly min-max normalised to [0, 1]; they were "
            f"restored to the input's units with the same min ({lo[0]:.6g}, {lo[1]:.6g}) and range "
            f"({span[0]:.6g}, {span[1]:.6g}) recomputed from the two surviving obsm['spatial'] arrays"
        )
    parts = [
        AlignedPart(
            label=_stem(paths[i]) or f"slice_{i + 1}",
            path=paths[i],
            key="spatial_aligned",
            xy=xy[i],
            z=zs[i],
            obs_names=_names(read[i]),
            original_xy=original[i][:, :2],
        )
        for i in range(2)
    ]
    return _frames(adapter, parts, z, params, notes, derived_by=derived_by)


def _read_single_obsm(
    adapter: AlignerAdapter,
    files: dict[str, Any],
    keys: tuple[str, ...],
    output_key: str,
    slice_column: str,
    z: Any,
    params: dict[str, Any],
    notes: list[str],
) -> AlignedFrames | contract.Refusal:
    """One merged output file, one obsm key out of ``keys`` -- the first that is actually there."""
    paths = _listed(files, output_key)
    if not paths:
        return contract.Refusal(
            what=adapter.function,
            reason=f"output_files has no {output_key!r} entry",
            remedy=f"That is where this row's answer ({adapter.writes_key}) lives.",
        )
    adata = _open(paths[0])
    key = next((k for k in keys if _obsm(adata, k) is not None), "")
    if not key:
        return contract.Refusal(
            what=adapter.function,
            reason=f"{paths[0]} carries none of {', '.join(keys)}",
            remedy=adapter.refusal_remedy or "Check the worker's log: the alignment step may not have run.",
        )
    arr = _obsm(adata, key)
    original = _obsm(adata, contract.ORIGINAL_KEY)
    zcol = arr[:, 2] if arr.shape[1] >= 3 else None
    if zcol is not None:
        notes.append(f"the third column of obsm[{key!r}] was kept as z; this module cannot say how it was obtained")
    part = AlignedPart(
        label=_stem(paths[0]),
        path=paths[0],
        key=key,
        xy=arr[:, :2],
        z=zcol,
        obs_names=_names(adata),
        original_xy=None if original is None else original[:, :2],
        slice_labels=_labels(adata, slice_column),
    )
    return _frames(adapter, [part], z, params, notes)


def _read_moscot(
    adapter: AlignerAdapter, files: dict[str, Any], slices: list[str], z: Any, params: dict[str, Any]
) -> AlignedFrames | contract.Refusal:
    """moscot's merged object, split into its sections by the column the run was given.

    Every section is a row block of one AnnData, so the section column is what turns it into planes.
    It comes from ``params.batch_key``, never from a guess at the file's columns: without it the
    object would be read as one plane, and a caller's z spacing would put every section at the same
    height.
    """
    paths = _listed(files, "adata_aligned_h5ad")
    if not paths:
        return adapter.refusal()
    batch_key = str(params.get("batch_key") or "")
    if not batch_key:
        return contract.Refusal(
            what=adapter.function,
            reason=(
                f"{paths[0]} holds every section in one object and the run's params name no batch_key "
                "(moscot workers before 2026-09-30 did not record it), so the sections cannot be told "
                "apart; read as one part, every section would share one z"
            ),
            remedy=(
                "Pass the worker's whole result, whose params carry batch_key, or add "
                "params={'batch_key': <the obs column the run was given>} beside its output_files."
            ),
        )
    notes = ["mode='warp' is moscot's default, so this is a barycentric projection onto the reference, not an affine"]
    reference = params.get("reference_batch_used")
    if reference not in (None, ""):
        notes.append(
            f"section {str(reference)!r} of obs[{batch_key!r}] is the reference (params.reference_batch_used): "
            "its rows keep their own obsm['spatial'], and every other section is projected onto it"
        )
    frames = _read_single_obsm(
        adapter,
        files,
        ("moscot_spatial_warp",),
        "adata_aligned_h5ad",
        batch_key,
        z,
        params,
        notes,
    )
    if isinstance(frames, AlignedFrames) and frames.parts and not frames.parts[0].slice_labels:
        return contract.Refusal(
            what=adapter.function,
            reason=f"params.batch_key is {batch_key!r}, and {paths[0]} has no obs[{batch_key!r}] to split its sections by",
            remedy="The params and the output file are from different runs; pass the result the file came from.",
        )
    return frames


def _read_st_gears(
    adapter: AlignerAdapter, files: dict[str, Any], slices: list[str], z: Any, params: dict[str, Any]
) -> AlignedFrames | contract.Refusal:
    notes = [
        "obsm['st_gears_xyz'] was not read: it is the worker's own copy of spatial_3d_aligned (ST-GEARS's "
        "aligned xy with the slice ordinal as z, plus the nearest-bin fill when allow_nearest_fallback ran); "
        "ST-GEARS's own key is read instead",
        "rows ST-GEARS could not place are NaN in spatial_elas_reuse (obs['st_gears_xy_source'] == 'none', "
        "counted in summary.n_spots_without_aligned_xy) and are left out of this frame",
    ]
    frames = _read_single_obsm(
        adapter,
        files,
        ST_GEARS_KEYS,
        "aligned_h5ad",
        str(params.get("slice_key") or ""),
        z,
        params,
        notes,
    )
    # Kept in the frame, they reached write_frame and then the before/after geometry, which raised
    # 'data must be finite'; the SPIRAL reader already left the same rows out (hunt 2026-09-30, u21-3d-21).
    return _without_unplaced_rows(frames, "ST-GEARS could not place")


def _without_unplaced_rows(frames: AlignedFrames | contract.Refusal, who: str) -> AlignedFrames | contract.Refusal:
    """A one-part frame without the rows whose aligned xy is not finite, and a note counting them."""
    import dataclasses

    import numpy as np

    if not isinstance(frames, AlignedFrames) or len(frames.parts) != 1:
        return frames
    part = frames.parts[0]
    keep = np.isfinite(np.asarray(part.xy, dtype="float64")).all(axis=1)
    if keep.all():
        return frames

    def _rows(values: Any) -> Any:
        if values is None or np.ndim(values) == 0 or len(values) != len(keep):
            return values
        picked = np.asarray(values, dtype=object if isinstance(values, tuple) else None)[keep]
        return tuple(picked.tolist()) if isinstance(values, tuple) else picked

    kept = dataclasses.replace(
        part,
        xy=np.asarray(part.xy)[keep],
        z=_rows(part.z),
        obs_names=_rows(part.obs_names),
        original_xy=_rows(part.original_xy),
        slice_labels=_rows(part.slice_labels),
    )
    dropped = f"{int((~keep).sum())} of {len(keep)} spots {who} (NaN) were left out of this frame"
    return dataclasses.replace(frames, parts=(kept,), notes=(*frames.notes, dropped))


def _read_spiral_align(
    adapter: AlignerAdapter, files: dict[str, Any], slices: list[str], z: Any, params: dict[str, Any]
) -> AlignedFrames | contract.Refusal:
    notes = [
        "only spots in clusters shared by both slices are placed; the worker writes the rest as NaN "
        "(obs['spiral_aligned'] is False) and they are left out of this frame. An output written "
        "before 2026-09-29 holds their input coordinates there instead, so its frame is a mixture",
        "obs_names carry an s0_/s1_ prefix and do not join to the input files by name",
    ]
    frames = _read_single_obsm(adapter, files, ("spatial_aligned",), "aligned_h5ad", "batch", z, params, notes)
    return _without_unplaced_rows(frames, "the worker did not place")


def _read_spacel(
    adapter: AlignerAdapter, files: dict[str, Any], slices: list[str], z: Any, params: dict[str, Any]
) -> AlignedFrames | contract.Refusal:
    paths = _listed(files, "aligned_h5ads")
    if not paths:
        return contract.Refusal(
            what=adapter.function,
            reason="output_files has no 'aligned_h5ads' entry",
            remedy="SPACEL writes one h5ad per slice under that one key, as a list.",
        )
    parts: list[AlignedPart] = []
    for i, path in enumerate(paths):
        adata = _open(path)
        arr = _obsm(adata, "spatial_aligned")
        if arr is None:
            return contract.Refusal(
                what=adapter.function,
                reason=f"{path} has no obsm['spatial_aligned']",
                remedy="Scube.align writes it in place; its absence means the align step did not complete.",
            )
        original = _obsm(adata, contract.ORIGINAL_KEY)
        parts.append(
            AlignedPart(
                label=_stem(path) or f"slice_{i}",
                path=path,
                key="spatial_aligned",
                xy=arr[:, :2],
                z=arr[:, 2] if arr.shape[1] >= 3 else None,
                obs_names=_names(adata),
                original_xy=None if original is None else original[:, :2],
            )
        )
    notes = [
        "Scube median-centres each slice before aligning, so this frame does not share an origin with obsm['spatial']"
    ]
    return _frames(adapter, parts, z, params, notes)


def _read_stalign(
    adapter: AlignerAdapter, files: dict[str, Any], slices: list[str], z: Any, params: dict[str, Any]
) -> AlignedFrames | contract.Refusal:
    import numpy as np
    import pandas as pd

    paths = _listed(files, "aligned_points_csv")
    if not paths:
        return contract.Refusal(
            what=adapter.function,
            reason="output_files has no 'aligned_points_csv' entry",
        )
    if len(slices) != 1:
        return contract.Refusal(
            what=adapter.function,
            reason=f"one slice must be named for the {len(slices)} given, because the CSV has no identifiers",
            remedy="Pass input_slices=[the h5ad whose rows produced the source CSV], in that row order.",
        )
    frame = pd.read_csv(paths[0])
    cols = [str(c).lower() for c in frame.columns]
    if "y" in cols and "x" in cols:
        yx = frame.iloc[:, [cols.index("y"), cols.index("x")]].to_numpy(dtype="float64")
    else:
        yx = frame.iloc[:, :2].to_numpy(dtype="float64")
    xy = np.column_stack([yx[:, 1], yx[:, 0]])

    adata = _open(slices[0])
    original = _obsm(adata, contract.ORIGINAL_KEY)
    if len(xy) != adata.n_obs:
        return contract.Refusal(
            what=adapter.function,
            reason=f"the aligned CSV has {len(xy)} rows and {slices[0]} has {adata.n_obs}",
            remedy=(
                "STalign writes no identifier, so rows can only be matched by position and an "
                "unequal count would assign coordinates to the wrong cells."
            ),
        )
    part = AlignedPart(
        label=_stem(slices[0]),
        path=paths[0],
        key="stalign_aligned_points.csv",
        xy=xy,
        obs_names=_names(adata),
        original_xy=None if original is None else original[:, :2],
    )
    notes = [
        "the CSV's (y, x) columns were swapped to (x, y) to match obsm['spatial']",
        "rows were joined to the slice by position, which is the only join available",
    ]
    # Read as written: a column swap and a positional join compute nothing, so this module is not
    # stamped on STalign's own coordinates (it is on SLAT's, which are fitted here).
    return _frames(adapter, [part], z, params, notes)


def _read_slat(
    adapter: AlignerAdapter, files: dict[str, Any], slices: list[str], z: Any, params: dict[str, Any]
) -> AlignedFrames | contract.Refusal:
    import numpy as np
    import pandas as pd

    from .geometry import _umeyama

    paths = _listed(files, "matching_csv")
    if not paths:
        return contract.Refusal(what=adapter.function, reason="output_files has no 'matching_csv' entry")
    if len(slices) != 2:
        return contract.Refusal(
            what=adapter.function,
            reason=f"{len(slices)} input slices given; SLAT matches exactly two and both are needed",
            remedy="Pass input_slices=[h5ad_path_1, h5ad_path_2], in the order the run received them.",
        )
    match = pd.read_csv(paths[0])
    if "slice1_idx" not in match.columns or "slice2_idx" not in match.columns:
        return contract.Refusal(
            what=adapter.function,
            reason=f"{paths[0]} has columns {list(match.columns)}, not slice1_idx/slice2_idx",
            remedy="Workers before 2026-09-29 dumped the raw matching when its shape was unexpected; that form cannot be read.",
        )
    a1, a2 = _open(slices[0]), _open(slices[1])
    c1, c2 = _obsm(a1, contract.ORIGINAL_KEY), _obsm(a2, contract.ORIGINAL_KEY)
    if c1 is None or c2 is None:
        return contract.Refusal(what=adapter.function, reason="one of the input slices has no obsm['spatial']")
    n1, n2 = int(a1.n_obs), int(a2.n_obs)
    if len(match) != min(n1, n2):
        return contract.Refusal(
            what=adapter.function,
            reason=f"the matching has {len(match)} rows, and scSLAT always writes min(n1, n2) = {min(n1, n2)}",
            remedy="A different count means these files are not the ones the run was given.",
        )

    left = match["slice1_idx"].to_numpy(dtype=int)
    right = match["slice2_idx"].to_numpy(dtype=int)
    # One row per spot of the smaller slice (slice 2 when the sizes are equal); that slice's column
    # is the enumeration 0..len-1 and the other holds its best match in the larger slice. The
    # worker labels each column by its slice. Workers before 2026-09-29 wrote the enumeration under
    # slice1_idx whatever the sizes, so whenever n1 >= n2 their columns held each other's contents.
    # The layout is read from the file, not from the sizes alone: swapping on n1 >= n2 would turn a
    # correctly labelled file into a wrong one.
    enum = np.arange(len(match))
    swapped = n1 >= n2 and np.array_equal(left, enum) and not np.array_equal(right, enum)
    idx1, idx2 = (right, left) if swapped else (left, right)
    notes = []
    if swapped:
        notes.append(
            f"the matching CSV's column labels were swapped back: n1={n1} >= n2={n2} and its "
            "'slice1_idx' column is the enumeration of slice 2, the layout the slat worker wrote "
            "before 2026-09-29"
        )
    keep = (idx1 >= 0) & (idx1 < n1) & (idx2 >= 0) & (idx2 < n2)
    if int(keep.sum()) < MIN_CORRESPONDENCES:
        return contract.Refusal(
            what=adapter.function,
            reason=f"only {int(keep.sum())} usable correspondences, below the {MIN_CORRESPONDENCES} a similarity needs",
        )
    src = c2[idx2[keep], :2]
    dst = c1[idx1[keep], :2]
    rot, scale, shift = _umeyama(src, dst)
    moved = (scale * (c2[:, :2] @ rot.T)) + shift
    resid = float(np.median(np.linalg.norm((scale * (src @ rot.T)) + shift - dst, axis=1)))
    notes.append(
        f"SLAT wrote no coordinates; a 2D similarity was fitted here from {int(keep.sum())} matched "
        f"pairs (scale {scale:.4g}, median residual {resid:.4g} in the slices' own units) and applied "
        f"to every row of slice 2, including the unmatched ones"
    )
    parts = [
        AlignedPart(
            label=_stem(slices[0]),
            path=slices[0],
            key="fitted from slat_matching.csv",
            xy=c1[:, :2],
            obs_names=_names(a1),
            original_xy=c1[:, :2],
        ),
        AlignedPart(
            label=_stem(slices[1]),
            path=slices[1],
            key="fitted from slat_matching.csv",
            xy=moved,
            obs_names=_names(a2),
            original_xy=c2[:, :2],
        ),
    ]
    return _frames(adapter, parts, z, params, notes, derived_by=DERIVED_BY)


def _stem(path: str) -> str:
    """The file's name without its extension -- a slice label that a reader recognises."""
    from pathlib import Path

    name = Path(str(path)).name
    for suffix in (".h5ad", ".csv", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


_READERS = {
    "paste_pairwise_align": _read_paste,
    "paste_center_align": _read_paste,
    # PASTE2 and CAST leave the same layout PASTE's worker does -- aligned_slice_<i>, the input
    # restored in obsm['spatial'], the answer beside it -- and their rows were adapted with no
    # reader behind them, so read_aligned raised KeyError on both.
    "paste2_partial_align": _read_paste,
    "cast_align_slices": _read_paste,
    "moscot_run": _read_moscot,
    "st_gears_reconstruct_3d": _read_st_gears,
    "gpsa_align_slices": _read_gpsa,
    "stalign_align_points": _read_stalign,
    "stalign_align_to_image": _read_stalign,
    "slat_align_slices": _read_slat,
    "spiral_align": _read_spiral_align,
    "run_spacel_scube": _read_spacel,
}


# ---------------------------------------------------------------------------------------------
# Completeness against the config, rather than against a list in a test
# ---------------------------------------------------------------------------------------------

#: A function whose name contains one of these is treated as an alignment candidate. Names are
#: checked rather than descriptions because a description mentioning alignment is common and a
#: name claiming it is not: the pattern catches eleven of the thirteen shipped functions, and the
#: two it misses (moscot_run, spiral_integrate) are caught by the server rule below.
_ALIGNMENT_WORDS = ("align", "register", "registration", "reconstruct", "warp", "stack", "scube")


def _enabled_tools(config_path: Any = None) -> list[tuple[str, str]]:
    """(server, function) for every tool of every enabled server in the config."""
    import yaml

    if config_path is None:
        from spatialomicsgym.mcp_config_path import find_mcp_config

        config_path = find_mcp_config()
    if config_path is None:
        return []
    with open(str(config_path), encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    servers = config.get("mcp_servers")
    if not isinstance(servers, dict):
        return []
    out: list[tuple[str, str]] = []
    for server, cfg in servers.items():
        if not isinstance(cfg, dict) or not cfg.get("enabled", True):
            continue
        for tool in cfg.get("tools") or []:
            name = (tool or {}).get("spatialomicsgym_name")
            if name:
                out.append((str(server), str(name)))
    return out


def alignment_functions(config_path: Any = None) -> dict[str, str]:
    """Every enabled function in the config that this table should have a row for.

    Two rules, both data-driven. A function whose *name* claims alignment is a candidate; and so is
    every function of a server that already has a row here, because a portal that aligns is where
    the next aligner will be added and its name need not say so -- ``moscot_run`` and
    ``spiral_integrate`` are both already in that shape. :data:`NOT_SLICE_ALIGNMENT` then removes
    the ones that are not serial-section alignment, by name and with a reason each.
    """
    servers_with_rows = {a.server for a in _ROWS if a.status != STATUS_PENDING}
    found: dict[str, str] = {}
    for server, name in _enabled_tools(config_path):
        if name in NOT_SLICE_ALIGNMENT:
            continue
        lowered = name.lower()
        if server in servers_with_rows or any(word in lowered for word in _ALIGNMENT_WORDS):
            found[name] = server
    return found


def adapters_missing(config_path: Any = None) -> list[str]:
    """Every enabled alignment function in the config with no row in this table.

    Sorted, and empty when the table is complete. A completeness test asserts on this rather than
    on a hardcoded list, so a portal added tomorrow shows up as a failing name rather than as a
    frame nobody can read six months later.
    """
    return sorted(name for name in alignment_functions(config_path) if name not in ADAPTERS)


def adapters_not_in_config(config_path: Any = None) -> list[str]:
    """Rows whose function is not an enabled tool in the config.

    Expected to be exactly the pending ones. Anything else means a portal was disabled or renamed
    under a row that still claims to describe it.
    """
    enabled = {name for _, name in _enabled_tools(config_path)}
    return sorted(a.function for a in _ROWS if a.status != STATUS_PENDING and a.function not in enabled)
