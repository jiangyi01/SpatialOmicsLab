"""Loader for know-how documents."""

import glob
import os
from pathlib import Path

#: Where the merged external skill packs live: one directory below the corpus, so the non-recursive
#: ``*.md`` glob in ``_load_documents`` never sees them and the tier-1 pin (``KNOW_HOW_HASH``) does
#: not move when a pack does. Read by :meth:`KnowHowLoader.load_packs` and by nothing else.
PACKS_SUBDIR = "packs"

#: The most characters of a pack's Short Description :meth:`KnowHowLoader.get_pack_summaries` hands
#: the second retrieval pass, per document. The upstream descriptions run to 1,010 characters; over
#: the whole set that was a 22 KB candidate list on every turn with the packs on, and the first 240
#: characters are where a description says what the document is for. Cut at a word boundary with an
#: ellipsis, the same shape as ``sog_portal/skills_api._clip``.
PACK_SUMMARY_CHARS = 240


def clip_summary(text: object, limit: int = PACK_SUMMARY_CHARS) -> str:
    """Collapse whitespace and cut at ``limit``, snapping back to a word boundary, ellipsis appended."""
    out = " ".join(("" if text is None else str(text)).split())
    if len(out) <= limit:
        return out
    cut = out[: limit - 1]
    space = cut.rfind(" ")
    if space >= limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:.") + "…"


#: Documents that stay on disk and out of the prompt a scored run is measured against.
#:
#: Both describe something that CANNOT happen during a scored run, which is why this is scope and
#: not caution. ``autonomous_iteration_loop`` is about a research loop ``research_allowed`` refuses
#: under benchmarking. ``visual_analysis`` is about choosing and reading figures, and the two
#: visualization portals declare ``benchmark_visible: false``, so a scored run has no figure to
#: choose and no plotting tool to call.
#:
#: ``spatial_3d_reconstruction`` is gated for a different reason than the two above, and the
#: difference matters: the alignment portals it describes are NOT ``benchmark_visible: false``, so
#: the scope argument used for ``visual_analysis`` does not transfer. Its spine is a checkpoint at
#: which the run stops and asks the user which interpretation of their tissue to act on -- and a
#: scored run has no user. ``benchmarking_enabled`` means one non-interactive turn graded against a
#: fixture, so a document whose central instruction cannot execute on that path does not belong in
#: the prompt that path is measured against. Two supporting facts: the empirical leaderboard covers
#: only spatial_clustering, deconvolution and svg_detection, so no scored instance exercises
#: alignment at all; and ~22 KB added to a ~453 KB scored prompt perturbs every instance of the
#: three families that ARE scored, for no scored benefit.
#:
#: Adding a name here moves ``KNOW_HOW_HASH`` (defined over the on-disk glob) and must NOT move the
#: scored corpus digest. Those two pins exist to tell exactly that pair apart, and the corpus lens
#: fails loudly if the second one moves.
_NOT_IN_A_SCORED_PROMPT: frozenset[str] = frozenset(
    {"autonomous_iteration_loop.md", "visual_analysis.md", "spatial_3d_reconstruction.md"}
)


class KnowHowLoader:
    """Load and manage know-how documents for the agent.

    Two tiers, two dicts, and the constructor fills only the first:

    * ``documents`` -- tier 1, the curated corpus in ``know_how/*.md``. This is what every scored
      run is measured against, so ``get_document_summaries``, ``get_all_documents`` and
      ``get_document_by_id`` read it and nothing else.
    * ``pack_documents`` -- tier 2, the merged external packs under ``know_how/packs/``. Filled by
      :meth:`load_packs` when a caller decides to, never by ``__init__``, and read only through the
      ``*_pack_*`` accessors. A loader that never had ``load_packs`` called behaves byte-for-byte
      as it did before the packs existed.
    """

    def __init__(self, know_how_dir: str | None = None):
        """Initialize the know-how loader.

        Args:
            know_how_dir: Directory containing know-how documents.
                         If None, uses the default know_how directory.

        """
        if know_how_dir is None:
            # Default to the know_how directory in the package
            current_dir = Path(__file__).parent
            know_how_dir = str(current_dir)

        self.know_how_dir = know_how_dir
        self.documents = {}
        # Tier 2 -- see the class docstring. Empty until load_packs() is called.
        self.pack_documents = {}
        self._packs_loaded = False
        self._load_documents()

    def _load_documents(self):
        """Load all markdown documents from the know-how directory."""
        pattern = os.path.join(self.know_how_dir, "*.md")
        # SORTED, because this list becomes ~427 KB of the assembled system prompt and `glob` does
        # not sort -- it returns whatever `os.scandir` hands back, which is directory order, which
        # depends on the order the files were created on THAT filesystem. Measured on this
        # checkout: the glob order is not the sorted order, so the same task on a different box
        # built a different prompt and neither was wrong. `loader.py:218` already sorts its sibling
        # glob and `transcriptomics_skills.py:68` carries the comment "``glob`` does not sort";
        # this is the one that reaches every scored run. It also makes `KNOW_HOW_HASH` a real pin:
        # it covers the content of these files and, until now, not the order they were laid out in.
        md_files = sorted(glob.glob(pattern))

        # Conditional know-how gating — memory_user_mcp_tools.md is loaded
        # ONLY when default_config.memory_enabled is True (zero-knowledge-
        # when-off discipline; see plan §14 C1/C2).
        memory_enabled = False
        benchmarking = True
        try:
            from spatialomicsgym.config import default_config

            memory_enabled = bool(getattr(default_config, "memory_enabled", False))
            # Read here and not per-file: the iteration playbook below is excluded from a SCORED
            # run, and an unreadable configuration answers "yes, benchmarking" so the exclusion
            # holds. Same failure direction as ``know_how.enrolment._benchmarking_now`` and for
            # the same reason -- the safe answer is "behave exactly as before", never "quietly
            # enrol more".
            benchmarking = bool(getattr(default_config, "benchmarking_enabled", False))
        except Exception:
            memory_enabled = False
            benchmarking = True

        for filepath in md_files:
            filename = os.path.basename(filepath)
            filename_without_ext = os.path.splitext(filename)[0]

            # Skip README, QUICK_START, and other meta documentation (all caps filenames)
            if filename.upper() in ["README.MD", "QUICK_START.MD"] or filename_without_ext.isupper():
                continue

            # Skip the memory know-how when memory is disabled. The creation playbooks still mention
            # memory, but every hook sits behind ``memory_enabled`` and never runs when it is off
            # (the earlier claim that the agent "must not even know the concept exists" was untrue;
            # hunt 2026-09-30, u13k1-knowhow-29).
            if filename == "memory_user_mcp_tools.md" and not memory_enabled:
                continue

            # Skip the autonomous-iteration playbook during a scored run.
            #
            # ``enrolment`` forces mode ``all`` whenever benchmarking is on, so every document in
            # this directory is in the prompt a benchmark is measured against. Adding one there
            # changes the eval prompt, which is the one thing the agent-performance red line
            # forbids -- and this playbook is about a loop a scored run is not allowed to start
            # anyway (``research.loop.research_allowed`` refuses under benchmarking for the same
            # reason: a scored run must not grow extra unrequested turns).
            #
            # So the gate is not caution, it is the honest scope: the document describes something
            # that cannot happen during a scored run, and off that path the prompt is byte-
            # identical to what it was before this file existed.
            if filename in _NOT_IN_A_SCORED_PROMPT and benchmarking:
                continue

            content = self._read_document(filepath)
            if content is None:
                continue

            doc_id = os.path.splitext(filename)[0]
            self.documents[doc_id] = self._build_document(doc_id, filename, filepath, content)

    def _read_document(self, filepath: str) -> str | None:
        """Read one document: utf-8, latin-1 on a decode failure, ``None`` when it cannot be opened.

        Encoding-lenient plus a per-file guard, because a single non-UTF-8 or unreadable ``.md``
        must not crash the whole corpus load -- which would take the constructor, and therefore
        agent start-up, down with it. Shared by both tiers so they cannot read a file differently.
        """
        try:
            try:
                with open(filepath, encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(filepath, encoding="latin-1") as f:
                    return f.read()
        except OSError as read_err:
            print(f"[know-how] skipping unreadable {os.path.basename(filepath)}: {read_err}")
            return None

    def _build_document(self, doc_id: str, filename: str, filepath: str, content: str) -> dict:
        """The document record both tiers store: title, description, both bodies, metadata."""
        title, description, metadata = self._extract_metadata(content, filename)

        # Use short_description from metadata if available, otherwise fall back to extracted description
        if "short_description" in metadata and metadata["short_description"]:
            description = metadata["short_description"]

        return {
            "id": doc_id,
            "name": title,
            "description": description,
            "content": content,
            "content_without_metadata": self._strip_metadata(content),
            "filepath": filepath,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------ tier 2: the merged packs

    def load_packs(self) -> int:
        """Load tier 2 -- the merged external packs under ``packs/<pack>/<skill>.md``.

        Exactly one level below ``packs/``: a ``glob`` over ``packs/*/*.md``, never an ``rglob``,
        so a deeper file is no more a document than ``resource/`` is for tier 1. The same meta-file
        skip as tier 1 (``README.md``, ``QUICK_START.md``, any all-caps stem). Ids are namespaced
        by pack -- ``kdense/scvi-tools`` -- so two packs may document the same library.

        Never called by the constructor, and it fills ``pack_documents`` only: ``documents`` and
        the tier-1 accessors do not change by one byte when this runs, which is what lets a scored
        run keep the prompt it was measured under while a portal turn gets more. Idempotent: a
        second call is a no-op and returns the same count.

        A no-op under ``benchmarking_enabled``: it empties ``pack_documents`` and returns 0, whoever
        calls it and whatever it held. The per-turn gate (``enrolment.packs_enabled``) already keeps a
        filled loader out of a scored prompt; this closes the other direction, so no wire-up path --
        the constructor's, a catalogue page's, :meth:`reload` -- can fill tier 2 on a scored run.
        Same fail-safe direction as that gate: an unreadable configuration answers "benchmarking".
        """
        from spatialomicsgym.know_how import enrolment

        if enrolment._benchmarking_now():
            self.pack_documents = {}
            self._packs_loaded = False
            return 0
        if self._packs_loaded:
            return len(self.pack_documents)
        pattern = os.path.join(self.know_how_dir, PACKS_SUBDIR, "*", "*.md")
        for filepath in sorted(glob.glob(pattern)):
            filename = os.path.basename(filepath)
            stem = os.path.splitext(filename)[0]
            if filename.upper() in ["README.MD", "QUICK_START.MD"] or stem.isupper():
                continue
            content = self._read_document(filepath)
            if content is None:
                continue
            pack = os.path.basename(os.path.dirname(filepath))
            doc_id = f"{pack}/{stem}"
            doc = self._build_document(doc_id, filename, filepath, content)
            doc["tier"] = 2
            doc["pack"] = pack
            self.pack_documents[doc_id] = doc
        self._packs_loaded = True
        return len(self.pack_documents)

    def get_pack_summaries(self) -> list[dict]:
        """Tier-2 candidates for the retriever's second pass: ``id``, ``name``, ``description``, ``pack``.

        ``description`` is the Short Description clipped to :data:`PACK_SUMMARY_CHARS` by
        :func:`clip_summary`; the document itself (``get_pack_document_by_id``) keeps the full line.
        """
        return [
            {"id": doc["id"], "name": doc["name"], "description": clip_summary(doc["description"]), "pack": doc["pack"]}
            for doc in self.pack_documents.values()
        ]

    def get_pack_document_by_id(self, doc_id: str) -> dict | None:
        """A tier-2 document by its namespaced id, or ``None``. Never falls through to tier 1."""
        return self.pack_documents.get(doc_id)

    def remove_pack_document(self, doc_id: str) -> bool:
        """Drop one tier-2 document (the commercial-mode filter uses this). ``True`` if it was there."""
        return self.pack_documents.pop(doc_id, None) is not None

    def _extract_metadata(self, content: str, filename: str) -> tuple[str, str, dict]:
        """Extract title, description, and metadata from markdown content.

        Args:
            content: Markdown content
            filename: Filename (used as fallback for title)

        Returns:
            Tuple of (title, description, metadata_dict)

        """
        lines = content.split("\n")

        # Extract title (first h1)
        title = None
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        if title is None:
            # Fallback to filename
            title = filename.replace("_", " ").replace(".md", "").title()

        # Extract metadata section
        metadata = {}
        in_metadata = False
        current_field = None

        for _i, line in enumerate(lines):
            if line.startswith("## Metadata"):
                in_metadata = True
                continue
            elif in_metadata:
                if line.startswith("##") and "Metadata" not in line:
                    # End of metadata section
                    break
                elif line.startswith("**") and "**:" in line:
                    # New metadata field (e.g., **Authors**:)
                    field_match = line.split("**")[1]
                    current_field = field_match.lower().replace(" ", "_")
                    # Get the value after the colon if it exists on the same line
                    colon_idx = line.find("**:")
                    if colon_idx != -1:
                        value_part = line[colon_idx + 3 :].strip()
                        if value_part:
                            metadata[current_field] = value_part
                        else:
                            metadata[current_field] = ""
                elif current_field and line.strip() and not line.startswith("---"):
                    # Continuation of current field
                    if current_field not in metadata:
                        metadata[current_field] = ""
                    if line.startswith("- "):
                        # List item
                        if metadata[current_field]:
                            metadata[current_field] += ", " + line[2:].strip()
                        else:
                            metadata[current_field] = line[2:].strip()
                    elif not line.startswith("```"):
                        # Regular text
                        if metadata[current_field]:
                            metadata[current_field] += " " + line.strip()
                        else:
                            metadata[current_field] = line.strip()

        # Extract description (content under ## Overview or first paragraph)
        description = ""
        in_overview = False
        overview_lines = []

        for _i, line in enumerate(lines):
            if line.startswith("## Overview"):
                in_overview = True
                continue
            elif in_overview:
                if line.startswith("##"):
                    # End of overview section
                    break
                elif line.strip():
                    overview_lines.append(line.strip())

        if overview_lines:
            description = " ".join(overview_lines)
        else:
            # Fallback: get first non-empty paragraph after title
            found_title = False
            for line in lines:
                if line.startswith("# "):
                    found_title = True
                    continue
                if found_title and line.strip() and not line.startswith("#"):
                    description = line.strip()
                    break

        # Limit description length
        if len(description) > 200:
            description = description[:197] + "..."

        return title, description, metadata

    def _strip_metadata(self, content: str) -> str:
        """Strip the metadata section from document content.

        Args:
            content: Full document content with metadata

        Returns:
            Content without metadata section

        """
        lines = content.split("\n")
        result_lines = []
        in_metadata = False
        skip_until_separator = False
        found_first_h1 = False

        for line in lines:
            # Track first H1 (title)
            if line.startswith("# ") and not found_first_h1:
                result_lines.append(line)
                found_first_h1 = True
                continue

            # Detect metadata section start
            if line.startswith("## Metadata"):
                in_metadata = True
                continue

            # Skip separator lines before and after metadata
            if line.strip() == "---":
                if not in_metadata:
                    # This might be the separator before metadata
                    skip_until_separator = True
                    continue
                else:
                    # This is the separator after metadata
                    in_metadata = False
                    skip_until_separator = False
                    continue

            # Skip lines in metadata section
            if in_metadata or skip_until_separator:
                # Check if we hit another H2 (end of metadata)
                if line.startswith("##") and "Metadata" not in line:
                    in_metadata = False
                    skip_until_separator = False
                    result_lines.append(line)
                continue

            # Keep all other lines
            result_lines.append(line)

        # Join and clean up extra blank lines
        result = "\n".join(result_lines)

        # Remove excessive blank lines (more than 2 consecutive)
        while "\n\n\n\n" in result:
            result = result.replace("\n\n\n\n", "\n\n\n")

        return result.strip()

    def get_all_documents(self) -> list[dict]:
        """Get all know-how documents as a list.

        Returns:
            List of document dictionaries with keys: id, name, description, content

        """
        return list(self.documents.values())

    def get_document_by_id(self, doc_id: str) -> dict | None:
        """Get a specific know-how document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document dictionary or None if not found

        """
        return self.documents.get(doc_id)

    def get_document_summaries(self) -> list[dict]:
        """Get summaries of all documents (without full content).

        Returns:
            List of document summaries with keys: id, name, description

        """
        return [
            {"id": doc["id"], "name": doc["name"], "description": doc["description"]} for doc in self.documents.values()
        ]

    def add_custom_document(self, doc_id: str, name: str, description: str, content: str, metadata: dict | None = None):
        """Add a custom know-how document programmatically.

        Args:
            doc_id: Unique identifier for the document
            name: Document title
            description: Brief description
            content: Full document content
            metadata: Optional metadata dictionary (authors, license, etc.)

        """
        if metadata is None:
            metadata = {}

        self.documents[doc_id] = {
            "id": doc_id,
            "name": name,
            "description": description,
            "content": content,
            # Mirror _load_documents: the prompt-assembly consumers (agent/execution.py) read
            # doc["content_without_metadata"] by subscript, so a programmatically-added doc that
            # omitted this key raised KeyError and took agent start-up down with it.
            "content_without_metadata": self._strip_metadata(content),
            "filepath": None,
            "metadata": metadata,
        }

    def get_document_metadata(self, doc_id: str) -> dict | None:
        """Get metadata for a specific document.

        Args:
            doc_id: Document identifier

        Returns:
            Metadata dictionary or None if not found

        """
        doc = self.documents.get(doc_id)
        return doc.get("metadata", {}) if doc else None

    def print_document_info(self, doc_id: str):
        """Print formatted information about a document including metadata.

        Args:
            doc_id: Document identifier

        """
        doc = self.documents.get(doc_id)
        if not doc:
            print(f"Document '{doc_id}' not found")
            return

        print("=" * 70)
        print(f"📚 {doc['name']}")
        print("=" * 70)
        print(f"\nDescription: {doc['description']}")

        metadata = doc.get("metadata", {})
        if metadata:
            print("\n" + "-" * 70)
            print("METADATA")
            print("-" * 70)

            # Display key metadata fields
            if "authors" in metadata:
                print(f"Authors: {metadata['authors']}")
            if "affiliations" in metadata:
                print(f"Affiliations: {metadata['affiliations']}")
            if "version" in metadata:
                print(f"Version: {metadata['version']}")
            if "last_updated" in metadata:
                print(f"Last Updated: {metadata['last_updated']}")
            if "license" in metadata:
                print(f"License: {metadata['license']}")
            if "commercial_use" in metadata:
                print(f"Commercial Use: {metadata['commercial_use']}")
            if "status" in metadata:
                print(f"Status: {metadata['status']}")

        print("=" * 70)

    def remove_document(self, doc_id: str):
        """Remove a know-how document.

        Args:
            doc_id: Document identifier

        """
        if doc_id in self.documents:
            del self.documents[doc_id]

    def reload(self):
        """Reload tier 1 from disk; tier 2 is emptied, and refilled only if it had been loaded AND
        ``enrolment.packs_enabled()`` is true right now (False under benchmarking, False by default).

        A loader whose packs were loaded before benchmarking came on is emptied by a reload, never
        refilled: the class docstring's promise that a scored run behaves as if the packs did not
        exist has to hold through this public method too, not only through the constructor.
        """
        self.documents = {}
        self._load_documents()
        had_packs = self._packs_loaded
        self.pack_documents = {}
        self._packs_loaded = False
        if had_packs:
            from spatialomicsgym.know_how import enrolment

            if enrolment.packs_enabled():
                self.load_packs()
