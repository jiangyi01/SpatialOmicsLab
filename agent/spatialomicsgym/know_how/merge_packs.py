"""Merge the admitted external skill packs into ``know_how/packs/`` as tier-2 know-how documents.

Two external skill repositories were audited (``docs/audit/wave_a/intake-sciagent.md`` and
``intake-kdense.md``; decisions D-045/D-046). Neither is merged as a repository. A manifest
(``packs/MANIFEST.yaml``) names the skills the intakes admit that the later audit and the live
measurement kept (52 on 2026-09-20), and this script turns each one -- deterministically, from a
clone pinned at the manifest's commit -- into a markdown document in the loader's own header
convention, under ``packs/<pack>/<skill>.md``.

Why a script and not a copy. Every one of these files was written for a different runtime: it
tells the reader to ``pip install`` into whatever environment is current, points at helper scripts
we do not vendor, and carries a YAML frontmatter our loader does not parse. Left as copied, one
``uv pip install`` reaching a live worker environment corrupts it (intake-kdense §7.4), and one
``---`` in a body makes ``KnowHowLoader._strip_metadata`` drop everything up to the next heading.
The rules below are applied in a fixed order and the result is refused -- not written -- when any
of them fails to hold at the end. A refusal names the file and the rule.

The rules, in order:

1. the clone's ``HEAD`` must equal the manifest commit, or nothing is rendered;
2. frontmatter: ``name`` and ``description`` are read (``license`` if present); everything else is
   dropped -- the emitted file has no frontmatter at all;
3. body = everything after the frontmatter, with its first H1 removed (it becomes the title);
4. named upstream sections are dropped (K-Dense ends every skill with an instruction to cite the
   pack in the user's manuscript; that is an instruction to the agent, not knowledge, and it goes);
5. the SciAgent "Check before installing" blockquote is removed;
6. every fenced block containing an install command -- ``pip``/``conda``/``uv pip`` and their kin,
   and since 2026-09-20 also the environment-making shapes ``uv run|venv|sync|init``,
   ``docker pull|run|build``, ``python -m pip|venv`` and ``source …activate``; since 2026-09-30
   ``cargo|go|gem|vcpkg|pipx|spack install``, ``gget setup``, ``hf auth login``, the R installers,
   a Nextflow container/Conda ``-profile``, and a ``curl``/``wget`` that writes a file or pipes into
   a shell or an unpacker (an API call that prints its answer is not an install) -- is replaced,
   whole, by one pointer: the script pointer when every install-shaped command in it only ran an
   upstream script under ``uv run``, the provisioning pointer otherwise; the one-line lead-in that
   introduced the block ("Install arboreto:") goes with it; inline install spans are rewritten; a
   file where any install line -- or install instruction in prose ("Install `x`", "then install
   y", "create an isolated environment") -- survives is refused;
7. every body ``---`` is removed (the ``_strip_metadata`` hazard);
8. the listed ``references/`` files, and only those, are folded in as ``## Reference: <H1>`` with
   their headings demoted; mentions of a folded file point at its section;
9. paths into upstream ``scripts/``, ``references/``, ``assets/`` and ``skills/<name>/``, and dotted
   ``scripts.`` imports, are marked as not vendored; in a fence the marked command's continuation
   lines go with it, and a fence left with nothing runnable becomes one pointer;
10. the manifest's ``replace:`` / ``excise:`` entries are applied, and a listed substring that is
    not found refuses the file (a silent no-op would let an excision rot). An edit that must
    remove an excluded identifier cannot quote it -- the manifest sits under ``packs/`` and is
    scanned -- so those are ``line_edits:``, addressed by line number and the sha256 of the exact
    line, applied to the raw source before step 2 and refused when the line has moved; a
    ``short_description:`` entry replaces the Short Description (the line the second retrieval
    pass ranks on) and is recorded in Modifications;
11. post-checks: size, fence pairing, no excluded identifier, summary length and uniqueness, no
    personal path, no upstream path, no surviving install line, no frontmatter, no dotenv file
    named, no API-key variable the manifest does not declare (``may_name_api_keys:``), no link to
    a URL shortener or download aggregator, and (since 2026-09-30) no reference to a skill the
    manifest refused or measured out -- or, given the clones, one that was never merged.

The header's Wrapped Tool License is the manifest's ``wrapped_tool_licence:`` when it has one; the
upstream frontmatter ``license`` is shown as evidence but never labelled the tool's (both upstreams
use that field for the skill text in some files and for the tool in others).

Nothing here downloads anything. The clones are given on the command line and verified by SHA.

Run::

    python -m spatialomicsgym.know_how.merge_packs --src <dir holding sciagent/ and kdense/>
    python -m spatialomicsgym.know_how.merge_packs --check [--src <dir>]

``--check`` never writes. Without ``--src`` it verifies the files on disk against the manifest and
the post-checks and prints the tier-2 digest; with ``--src`` it also re-renders every document and
reports any drift between the render and the file on disk.

The constants below are imported by the corpus lens
(``test/test_a_merged_pack_document_keeps_its_source_and_loses_its_installer.py``) so the script
and the test cannot disagree about what an excluded identifier is. For install lines the lens
deliberately does NOT rely on that import alone: it carries its own forbidden list, written from
the audit that found the gap, so a hole in these regexes is no longer a hole in the lens.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

KNOW_HOW_DIR = Path(__file__).resolve().parent
PACKS_DIR = KNOW_HOW_DIR / "packs"
MANIFEST_PATH = PACKS_DIR / "MANIFEST.yaml"
#: The repository root -- ``<repo>/agent/spatialomicsgym/know_how`` -> ``<repo>`` -- where the manifest's
#: ``licence_copy`` paths (``THIRD_PARTY_LICENSES/...``) are anchored.
REPO_ROOT = KNOW_HOW_DIR.parents[2]

#: The largest tier-1 document is 56,369 characters; nothing in tier 2 may be bigger than this.
MAX_PACK_DOC_BYTES = 64 * 1024

#: The floor ``test/test_a_know_how_summary_is_not_the_same_line_in_every_file.py`` applies.
MIN_SUMMARY_CHARS = 40

#: The eighteen identifiers both intakes excluded under RL-1 (``UNCERTAIN = EXCLUDE``). Thirteen
#: from intake-sciagent §2.2 rows E1-E9 and §6.2 row E10; five from intake-kdense §3.2 rows B-1 to
#: B-6. Matched on word boundaries, case-insensitively, over every emitted document -- and the
#: manifest under ``packs/`` deliberately refers to these by pointer, never by string, so that this
#: tuple is the only place in the package the strings exist.
EXCLUDED_IDENTIFIERS: tuple[str, ...] = (
    "ddinter",
    "scbdd",
    "cellchat",
    "jinworks",
    "fastp",
    "opengene",
    "hrbmu",
    "biogpt",
    "graphormer",
    "grover",
    "attentivefp",
    "torchdrug",
    "d4data",
    "pacsomatic",
    "qwen",
    "deepseek",
    "seedream",
    "bytedance",
)
EXCLUDED_IDENTIFIER_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, EXCLUDED_IDENTIFIERS)) + r")\b", re.IGNORECASE)

#: The environment-making commands that are not "<tool> install" but do the same thing: ``uv run
#: --with pkg==ver`` resolves and installs into an ephemeral environment, ``uv venv`` / ``python -m
#: venv`` / ``source .venv/bin/activate`` stand one up, ``docker pull|run|build`` fetches an image.
#: Added 2026-09-20 after an audit found all of these passing the post-check AND the corpus lens,
#: which imported these regexes and so was blind the same way; the lens now carries its own list.
_ENVIRONMENT_COMMAND = (
    r"uv\s+(?:run|venv|sync|init)"
    r"|docker(?:\s+compose)?\s+(?:pull|run|build)"
    r"|python3?\s+-m\s+(?:pip|venv)"
    r"|source\s+\S*activate"
)

#: Installers and provisioning steps that are not ``<package manager> install`` and that the first
#: two widenings missed (hunt 2026-09-30, u13k2-packs-2): ``cargo|go|gem|vcpkg|pipx|spack install``
#: build or fetch into the live toolchain; ``gget setup`` pip-installs module dependencies into the
#: running interpreter; ``hf auth login`` writes a token to disk; the R installers do what pip does.
_OTHER_INSTALLER = (
    r"(?:cargo|go|gem|vcpkg|pipx|spack)\s+install"
    r"|gget(?:\s+|\.)setup"
    r"|hf\s+auth\s+login|huggingface-cli\s+login"
    r"|BiocManager::install|install\.packages|(?:remotes|devtools)::install_\w+"
)

#: A line that IS an install command: the first token, after an optional prompt character and an
#: optional ``sudo``, is one of the package or environment tools. ``curl``/``wget`` used to be on
#: this list, so every REST example -- the IDC ``curl -s $B/version`` calls, the Folklore JSON-RPC
#: call -- was replaced by the installation pointer; a download is now DOWNLOAD_TO_FILE_RE's
#: question, asked of what the command writes (hunt 2026-09-30, u13k2-packs-3).
INSTALL_COMMAND_RE = re.compile(
    r"^\s*(?:[$%!>]\s*)?(?:sudo\s+)?"
    r"(?:pip3?|uv\s+(?:pip|add|tool|run|venv|sync|init)|conda|mamba|micromamba|pixi|npm|npx|brew|apt(?:-get)?"
    r"|" + _ENVIRONMENT_COMMAND + r"|" + _OTHER_INSTALLER + r")\b"
)

#: An install *phrase* anywhere on a line -- the shape an inline span or a code comment carries.
INSTALL_PHRASE_RE = re.compile(
    r"\b(?:pip3?|uv\s+pip|uv\s+add|uv\s+tool|conda|mamba|micromamba|pixi|npm|npx|brew|apt(?:-get)?)"
    r"\s+(?:install|create|add|env\s+create|i)\b"
    r"|\b(?:" + _ENVIRONMENT_COMMAND + r"|" + _OTHER_INSTALLER + r")\b"
)

#: ``curl ... | sh`` and its relatives: the six pipe-to-shell installers intake-kdense §5.2 counted.
PIPE_TO_SHELL_RE = re.compile(r"\b(?:curl|wget)\b[^\n]*\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b")

#: A ``curl``/``wget`` that downloads: writes a file (``-o FILE``, ``-O``, ``--output``, a ``>``
#: redirect; ``wget`` by default, unless sent to stdout with ``-O -``) or pipes into an unpacker.
#: An API call that prints its answer is not an install (hunt 2026-09-30, u13k2-packs-3).
_NOT_A_FILE = r"(?!/dev/null\b|/dev/stdout\b|-(?:\s|$))"
DOWNLOAD_TO_FILE_RE = re.compile(
    r"\bcurl\b[^\n]*?(?:\s-(?!-)[A-Za-z]*[oO](?:\s*" + _NOT_A_FILE + r"\S)"
    r"|\s--(?:output|output-dir)[\s=]+" + _NOT_A_FILE + r"\S|\s--remote-name(?:-all)?\b|\s--create-dirs\b)"
    r"|\b(?:curl|wget)\b[^\n]*?(?:\s>{1,2}\s*"
    + _NOT_A_FILE
    + r"[^\s&|]|\|\s*(?:sudo\s+)?(?:tar|unzip|gunzip|bsdtar|funzip)\b)"
    r"|\bwget\b(?![^\n]*?(?:\s-[A-Za-z]*O\s*(?:-(?:\s|$)|/dev/stdout\b)|\s--output-document[=\s]+(?:-(?:\s|$)|/dev/stdout\b)))"
)

#: A Nextflow run that provisions: a container or Conda profile pulls images or builds one
#: environment per process (hunt 2026-09-30, u13k2-packs-2).
NEXTFLOW_PROVISIONING_RE = re.compile(
    r"(?<![\w-])-profile[\s=]+\S*?\b(?:docker|conda|mamba|singularity|apptainer|podman|charliecloud|shifter|wave)\b"
)

#: A tool the corpus lens names outright, whatever the line shape: ``npx`` fetches and runs a
#: package. (``wget`` was here too; it is now DOWNLOAD_TO_FILE_RE's, which counts it whenever it
#: writes a file -- its default. The lens keeps its own outright ban.)
DOWNLOADER_RE = re.compile(r"\bnpx\b")

#: An install *instruction in prose*: the imperative a sentence, a list item or a code comment gives
#: -- "Install `idc-index`", "then install histolab", "(install separately)", "use the install
#: command it prints", "Keep Biopython updated", "create an isolated Python environment". None of
#: these is a command, so none of them replaces a fence; each refuses the file at the post-check
#: until the manifest rewrites the sentence (hunt 2026-09-30, u13k2-packs-2). Headings are exempt:
#: "## Installation" names a section, it does not instruct.
PROSE_INSTALL_RE = re.compile(
    r"(?<![\w`'\"/.-])Install\s+(?=[\w`(\[\"'*])"
    r"|(?i:\b(?:then|also|first|and|or|to|please)\s+install\b(?!ed|ing|ation))"
    r"|(?i:\binstall\s+(?:it\s+|them\s+)?separately\b)"
    r"|(?i:\binstall\s+command\b)"
    r"|(?i:\bkeep\s+[\w.-]+\s+(?:updated|up[- ]to[- ]date)\b)"
    r"|(?i:\b(?:create|activate|set\s+up)\s+(?:or\s+activate\s+)?(?:an?\s+|the\s+)?"
    r"(?:isolated\s+|clean\s+|new\s+|fresh\s+|separate\s+|dedicated\s+)?(?:python\s+|conda\s+|virtual\s+|tool\s+)?"
    r"(?:environment|venv|virtualenv)\b)"
    r"|(?i:\b(?:in|use|prefer)\s+(?:an?\s+)?(?:(?:clean|isolated|fresh|separate|dedicated|project|virtual|python)\s+){1,2}"
    r"environments?\b)"
)

#: A dotenv file named in a body. A retrievable document must never tell a model that holds a live
#: REPL to open the file this checkout loads its credentials from -- not even "narrowly", not even
#: to forbid it: the word itself is the pointer. (``database-lookup`` did exactly this and was
#: removed from the admitted set on 2026-09-20; two other files said it in passing and are
#: reworded by the manifest.) ``os.environ`` does not match: the ``\b`` needs a word boundary
#: after ``env``.
DOTENV_RE = re.compile(r"\.env\b")

#: An API-key environment variable named in a body. Allowed only where the manifest declares it
#: for that skill (``may_name_api_keys:``) -- the library's own configuration surface, read through
#: ``os.environ`` (``NCBI_API_KEY``, ``ESM_API_KEY``, ``ALPHAGENOME_API_KEY``). A name the manifest
#: does not declare refuses the file, so a table of other services' keys cannot arrive silently.
API_KEY_NAME_RE = re.compile(r"\b[A-Z][A-Z0-9_]*_API_KEY\b")

#: Link hosts a document may not point at: URL shorteners (the destination is not reviewable) and
#: download aggregators (a rehosted installer). The manifest strips the two upstream carried.
DENIED_LINK_HOST_RE = re.compile(
    r"https?://(?:[A-Za-z0-9-]+\.)*(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|is\.gd|buff\.ly|cutt\.ly"
    r"|software\.informer\.com|softonic\.com|softpedia\.com|filehippo\.com)\b",
    re.IGNORECASE,
)

#: A path that belongs to one person's machine (the ``primekg`` skill leaked one; D-045).
PERSONAL_PATH_RE = re.compile(r"(?:[A-Za-z]:\\Users\\|/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/)")

#: A path into an upstream directory we do not vendor. The strict form is what the post-check and
#: the corpus lens apply; the lookbehind keeps prose rewriting off the middle of a URL, where a
#: rewrite would mangle the link and a survivor is reported instead.
#:
#: Also an upstream *skill* directory (``skills/<name>/...``, ``cd skills/x/scripts``) and a dotted
#: import of the upstream ``scripts`` package (``from scripts.query_primekg import ...``): this
#: repository has a ``skills/`` directory of its own, so ``skills/data-visualization/...`` looked
#: openable and the agent spent steps on FileNotFoundError (hunt 2026-09-30, u13k2-packs-5). The
#: lookbehind keeps both off a URL, where ``.../blob/<sha>/skills/x/SKILL.md`` is the Source line.
UPSTREAM_PATH_ANY_RE = re.compile(
    r"(?:references|scripts|assets)/[A-Za-z0-9_./*-]*"
    r"|(?<![\w/.-])skills/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./*-]*)?"
    r"|(?<![\w/.-])scripts\.[A-Za-z_]\w*"
)
UPSTREAM_PATH_RE = re.compile(
    r"(?<![\w/])(?:\./)?(?:references|scripts|assets)/(?:[A-Za-z0-9_./*-]*[A-Za-z0-9_/*-])?"
    r"|(?<![\w/.-])skills/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./*-]*[A-Za-z0-9_/*-])?"
    r"|(?<![\w/.-])scripts\.[A-Za-z_]\w*"
)

#: What replaces an install fence. A blockquote, so the loader's metadata scanner cannot mistake it
#: for a field, and one line, so a document that had five install fences does not gain five
#: paragraphs.
#: What stands where an install block was. It used to say every named package "is provisioned by
#: the platform's environment recipes", which is true of almost none of them in the interpreter the
#: REPL runs: the model was told not to install, then met ModuleNotFoundError, and stalled or
#: invented results (hunt 2026-09-30, u13k2-packs-1 / u13-prompt-7). The packs never reach a scored
#: prompt, so this moves KNOW_HOW_PACKS_HASH and no epoch.
PROVISIONING_POINTER = (
    "> Installation is not done from this document. A package this text names is available only if "
    "it imports in the environment you are running in; if the import fails, say the library is not "
    "available here and continue without it. Do not install anything into a live analysis environment."
)
INLINE_INSTALL_POINTER = (
    "(not installed from this document: if the import fails, the library is not available here -- say so, "
    "do not install it)"
)
SCRIPT_POINTER = "(upstream helper script, not vendored)"
REFERENCE_POINTER = "(upstream reference file, not included)"
SKILL_POINTER = "(upstream skill, not merged here)"

#: What stands where a fence ran -- or, once its paths were marked, could only have run -- an
#: upstream helper script. The install pointer used to stand there, so a validator run or a
#: library-search CLI read as an installation step (hunt 2026-09-30, u13k2-packs-3 / -12).
SCRIPT_FENCE_POINTER = (
    f"> {SCRIPT_POINTER} The command that stood here runs a helper script this platform does not ship, "
    "so it cannot be run from this document; use the library's own API below, if it imports in this environment."
)
#: The same for a fence left holding nothing but pointers into upstream reference files.
REFERENCE_FENCE_POINTER = (
    f"> {REFERENCE_POINTER} The commands that stood here searched upstream reference files that are not "
    "included; nothing in them can be run from this document."
)

#: The default commercial-use sentence. Worded around the three strings
#: ``filter_know_how_for_commercial_mode`` keys on -- the *text* is commercially usable; the
#: software it describes is a separate question answered by the wrapped licence line.
DEFAULT_COMMERCIAL_USE = (
    "This text may be used commercially under its licence (see License above); the software it "
    "describes is governed by the Wrapped Tool License, not by this document."
)

_FENCE_OPEN_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


class MergeRefusal(Exception):
    """A document failed a rule. The message names the document and the rule."""


# --------------------------------------------------------------------------- small text helpers


def _scan_fences(lines: list[str]) -> tuple[list[tuple[int, int]], bool]:
    """``(blocks, open_at_end)``: each fenced block as inclusive ``(start, end)`` line indices, and
    whether the document ends inside a fence that never closed.

    A block opened with N backticks (or tildes) closes only on a marker of the same character with
    at least N of them and nothing else on the line -- so a four-backtick fence may contain a
    three-backtick one, which is how some upstream files show a fence inside a fence.
    """
    blocks: list[tuple[int, int]] = []
    open_char: str | None = None
    open_len = 0
    start = 0
    for i, line in enumerate(lines):
        match = _FENCE_OPEN_RE.match(line)
        if open_char is None:
            if match:
                open_char, open_len, start = match.group(2)[0], len(match.group(2)), i
            continue
        if match and match.group(2)[0] == open_char and len(match.group(2)) >= open_len and not match.group(3).strip():
            blocks.append((start, i))
            open_char = None
    return blocks, open_char is not None


def fence_blocks(lines: list[str]) -> list[tuple[int, int]]:
    return _scan_fences(lines)[0]


def fence_mask(lines: list[str]) -> list[bool]:
    """``True`` for every line inside a fenced block, the fence markers included."""
    mask = [False] * len(lines)
    blocks, open_at_end = _scan_fences(lines)
    for start, end in blocks:
        for i in range(start, end + 1):
            mask[i] = True
    if open_at_end:
        # an unclosed fence runs to the end; mark it so nothing treats its lines as prose
        last_close = blocks[-1][1] if blocks else -1
        opener = next(i for i in range(len(lines) - 1, last_close, -1) if _FENCE_OPEN_RE.match(lines[i]))
        for i in range(opener, len(lines)):
            mask[i] = True
    return mask


def fences_pair_up(text: str) -> bool:
    """Every fence that opens closes before the document ends."""
    return not _scan_fences(text.split("\n"))[1]


def split_frontmatter(raw: str) -> tuple[dict, str]:
    """``(frontmatter, body)`` -- the body starts after the closing ``---``."""
    lines = raw.split("\n")
    if not lines or lines[0].strip() != "---":
        raise MergeRefusal("no frontmatter (the file does not start with ---)")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            try:
                front = yaml.safe_load("\n".join(lines[1:i])) or {}
            except yaml.YAMLError as exc:
                raise MergeRefusal(f"frontmatter is not valid YAML ({exc})") from exc
            if not isinstance(front, dict):
                raise MergeRefusal("frontmatter is not a mapping")
            return front, "\n".join(lines[i + 1 :])
    raise MergeRefusal("frontmatter never closes")


def install_lines(text: str) -> list[tuple[int, str]]:
    """Every line that still carries an install command, an install phrase, a pipe-to-shell, a
    download, a provisioning profile, or an install instruction in prose. Shared with the corpus
    lens so both answer the question the same way.

    The prose rule skips markdown headings outside fences ("## Installation" names a section); a
    ``#`` line inside a fence is a code comment and is read like any other sentence."""
    hits: list[tuple[int, str]] = []
    lines = text.split("\n")
    mask = fence_mask(lines)
    for number, line in enumerate(lines, 1):
        heading = not mask[number - 1] and _HEADING_RE.match(line)
        if _is_install_line(line) or (not heading and PROSE_INSTALL_RE.search(line)):
            hits.append((number, line))
    return hits


def _is_install_line(line: str) -> bool:
    """A command-shaped install: what makes the whole fence around it the provisioning pointer."""
    return bool(
        INSTALL_COMMAND_RE.search(line)
        or INSTALL_PHRASE_RE.search(line)
        or PIPE_TO_SHELL_RE.search(line)
        or DOWNLOAD_TO_FILE_RE.search(line)
        or DOWNLOADER_RE.search(line)
        or NEXTFLOW_PROVISIONING_RE.search(line)
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def document_digests(documents: dict[str, str]) -> dict[str, str]:
    """``id -> sha256 of the text``, the per-document record the corpus lens keeps."""
    return {doc_id: _sha256(text.encode("utf-8")) for doc_id, text in documents.items()}


def pack_digest(digests: dict[str, str]) -> str:
    """One digest over ``id -> per-document digest``, in id order -- byte-for-byte the
    ``corpus_digest`` construction of the tier-1 corpus lens, so the two pins are read the same way."""
    running = hashlib.sha256()
    for doc_id in sorted(digests):
        running.update(doc_id.encode("utf-8"))
        running.update(b"\0")
        running.update(digests[doc_id].encode("ascii"))
        running.update(b"\n")
    return running.hexdigest()


def on_disk_documents(packs_dir: Path = PACKS_DIR) -> dict[str, str]:
    """``<pack>/<stem> -> text`` over ``packs/*/*.md`` -- exactly the loader's glob, one level deep."""
    out: dict[str, str] = {}
    for path in sorted(packs_dir.glob("*/*.md")):
        if path.name.upper() in ("README.MD", "QUICK_START.MD") or path.stem.isupper():
            continue
        out[f"{path.parent.name}/{path.stem}"] = path.read_text(encoding="utf-8")
    return out


# --------------------------------------------------------------------------- the transformation


@dataclass
class Rendered:
    dest: str
    text: str
    title: str
    summary: str
    modifications: dict[str, int] = field(default_factory=dict)


def _strip_first_h1(body: str) -> tuple[str, str | None]:
    """Remove the first H1 outside a fence; return the body and the H1's text."""
    lines = body.split("\n")
    mask = fence_mask(lines)
    for i, line in enumerate(lines):
        if mask[i]:
            continue
        match = _HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            del lines[i]
            return "\n".join(lines), match.group(2).strip()
    return body, None


def _drop_sections(body: str, titles: tuple[str, ...]) -> tuple[str, int]:
    """Remove each H2 whose text matches one of ``titles``, up to the next H1/H2 or the end."""
    if not titles:
        return body, 0
    lines = body.split("\n")
    mask = fence_mask(lines)
    out: list[str] = []
    dropped = 0
    skipping = False
    for i, line in enumerate(lines):
        heading = None if mask[i] else _HEADING_RE.match(line)
        if heading and len(heading.group(1)) <= 2:
            if heading.group(2).strip() in titles and len(heading.group(1)) == 2:
                skipping = True
                dropped += 1
                continue
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out), dropped


def _drop_blockquote(body: str, marker: str) -> tuple[str, int]:
    """Remove every contiguous ``>`` block (outside fences) that contains ``marker``."""
    lines = body.split("\n")
    mask = fence_mask(lines)
    out: list[str] = []
    dropped = 0
    i = 0
    while i < len(lines):
        if not mask[i] and lines[i].lstrip().startswith(">"):
            j = i
            while j < len(lines) and not mask[j] and lines[j].lstrip().startswith(">"):
                j += 1
            block = lines[i:j]
            if any(marker in line for line in block):
                dropped += 1
            else:
                out.extend(block)
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), dropped


_SCRIPT_FILE_RE = re.compile(r"[\w$/{}.\"'-]*\.(?:py|sh|R|pl)\b")
_UV_RUN_RE = re.compile(r"\buv\s+run\b")
_POINTERS = (PROVISIONING_POINTER, SCRIPT_FENCE_POINTER, REFERENCE_FENCE_POINTER)


def _logical_commands(lines: list[str]) -> list[str]:
    """A fence body as shell commands: a line ending in ``\\`` continues onto the next one."""
    commands: list[str] = []
    current = ""
    for line in lines:
        current = f"{current} {line.strip()}" if current else line.strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        if current:
            commands.append(current)
        current = ""
    if current:
        commands.append(current)
    return commands


def _runs_a_script(command: str) -> bool:
    """A command whose only install shape is ``uv run`` and which runs a script file: it is a
    script run (``uv run python scripts/validate_dxapp.py``), not a provisioning step."""
    return bool(_SCRIPT_FILE_RE.search(command)) and not _is_install_line(_UV_RUN_RE.sub("", command))


#: A lead-in that only pointed at a script or reference that is not here ("For unfamiliar files,
#: start with the bundled read-only inspector:").
_SCRIPT_LEAD_IN_RE = re.compile(
    r"\b(?:scripts?|helpers?|bundled|validator|inspector|CLIs?|skill (?:root|directory)|reference files?)\b", re.I
)


def _is_lead_in(out: list[str], index: int, pointer: str = PROVISIONING_POINTER) -> bool:
    """``out[index]`` is a one-line paragraph ending in a colon that only introduced the fence now
    replaced by ``pointer``: an install instruction ("Install arboreto:"), a short label ("For
    spatial workflows:"), or -- for a script or reference fence -- a sentence about that script.
    A longer sentence that says something of its own ("... the old loaders are deprecated:") stays."""
    line = out[index]
    text = line.strip().strip("*_").rstrip()
    if not text.endswith(":") or line.startswith((" ", "\t")) or _HEADING_RE.match(line):
        return False
    says_more = (
        len(text.split()) > 8
        and not PROSE_INSTALL_RE.search(text)
        and not (pointer != PROVISIONING_POINTER and _SCRIPT_LEAD_IN_RE.search(text))
    )
    if says_more:
        return False
    if re.match(r"^(?:[-*+>|]|\d+[.)])\s", line.lstrip()):
        return False  # a list item, quote or table row is part of a larger structure; leave it
    previous = out[index - 1] if index > 0 else ""
    return not previous.strip() or bool(_HEADING_RE.match(previous))


def _replace_install_fences(body: str) -> tuple[str, int]:
    """Any fenced block with an install line inside it becomes one pointer, whole.

    The pointer says what stood there: the provisioning pointer for an install, the script pointer
    when every install-shaped command in the block only ran an upstream script under ``uv run``
    (hunt 2026-09-30, u13k2-packs-3). The one-line lead-in that introduced the block ("Install
    arboreto:") goes with it -- left behind, it is an install instruction directly above the
    sentence saying installation is not done here (hunt 2026-09-30, u13k2-packs-2).
    """
    lines = body.split("\n")
    out: list[str] = []
    replaced = 0
    cursor = 0
    for start, end in fence_blocks(lines):
        out.extend(lines[cursor:start])
        block = lines[start : end + 1]
        if any(_is_install_line(line) for line in block[1:-1]):
            triggers = [c for c in _logical_commands(block[1:-1]) if _is_install_line(c)]
            pointer = (
                SCRIPT_FENCE_POINTER if triggers and all(_runs_a_script(c) for c in triggers) else PROVISIONING_POINTER
            )
            _drop_lead_in(out, pointer)
            out.append(pointer)
            replaced += 1
        else:
            out.extend(block)
        cursor = end + 1
    out.extend(lines[cursor:])
    return "\n".join(_collapse_pointers(out)), replaced


def _drop_lead_in(out: list[str], pointer: str = PROVISIONING_POINTER) -> None:
    """Remove the lead-in sentence (and the blank lines after it) at the end of ``out``, if any."""
    index = len(out) - 1
    while index >= 0 and not out[index].strip():
        index -= 1
    if index >= 0 and _is_lead_in(out, index, pointer):
        del out[index:]


def _collapse_pointers(lines: list[str]) -> list[str]:
    """Two replaced fences in a row (nextflow: a launcher fence, then an nf-core fence) would leave
    two identical pointers a blank line apart; one says everything the two would."""
    collapsed: list[str] = []
    for line in lines:
        if line in _POINTERS:
            previous = next((prior for prior in reversed(collapsed) if prior.strip()), "")
            if previous == line:
                while collapsed and not collapsed[-1].strip():
                    collapsed.pop()
                continue
        collapsed.append(line)
    return collapsed


def _rewrite_inline_installs(body: str) -> tuple[str, int]:
    """A backtick span carrying an install phrase, outside a fence, becomes the inline pointer."""
    lines = body.split("\n")
    mask = fence_mask(lines)
    count = 0
    for i, line in enumerate(lines):
        if mask[i] or "`" not in line:
            continue

        def _sub(match: re.Match[str]) -> str:
            nonlocal count
            span = match.group(0)
            if INSTALL_PHRASE_RE.search(span) or PIPE_TO_SHELL_RE.search(span):
                count += 1
                return INLINE_INSTALL_POINTER
            return span

        lines[i] = re.sub(r"`[^`\n]+`", _sub, line)
    return "\n".join(lines), count


def _remove_rules(body: str) -> tuple[str, int]:
    lines = body.split("\n")
    kept = [line for line in lines if line.strip() != "---"]
    return "\n".join(kept), len(lines) - len(kept)


def _demote_headings(body: str) -> str:
    lines = body.split("\n")
    mask = fence_mask(lines)
    for i, line in enumerate(lines):
        if not mask[i] and _HEADING_RE.match(line):
            lines[i] = "#" + line
    return "\n".join(lines)


_UPSTREAM_COMMAND_RE = re.compile(r"(?:\bpython3?|/bin/python)\s+(?!-m\b)([A-Za-z0-9_./\-]+\.py)\b")


def _rewrite_upstream_commands(body: str) -> tuple[str, int]:
    """Mark ``python <script>.py`` for a script that is not vendored.

    The path rule above catches ``scripts/x.py``; a bare ``python score_variants.py`` names the
    same file without its directory, and a reader told to run it finds nothing (the lens
    ``test_docs_commands_resolve`` checks exactly that over every document the loader can reach).
    No pack vendors a script, so every such command becomes the pointer, in fences and prose alike.
    """
    count = 0

    def _mark(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"python {SCRIPT_POINTER.rstrip(')')}: {match.group(1)})"

    return _UPSTREAM_COMMAND_RE.sub(_mark, body), count


def _rewrite_upstream_paths(body: str) -> tuple[str, int]:
    """Mark every path into ``scripts/``, ``references/``, ``assets/`` or an upstream ``skills/<name>/``
    directory -- and every dotted ``scripts.`` import -- as not vendored.

    Inside a fence the whole line becomes a comment, because a pointer in the middle of a command
    is not a command -- and the lines that continued that command (a trailing ``\\``) go with it:
    left behind, ``--quant-dir quant/ ...`` ran as a command of its own (hunt 2026-09-30,
    u13k2-packs-12). Outside a fence: a markdown link keeps its text, a backtick span becomes the
    pointer, a bare path becomes the pointer.
    """
    lines = body.split("\n")
    mask = fence_mask(lines)
    count = 0

    def _pointer(path: str) -> str:
        path = path.lstrip("./")
        if path.startswith(("scripts/", "scripts.")) or "/scripts/" in path or path.endswith("/scripts"):
            return SCRIPT_POINTER
        if path.startswith("skills/") and not re.search(r"/(?:references|assets)/", path):
            return SKILL_POINTER
        return REFERENCE_POINTER

    continued = False
    for i, line in enumerate(lines):
        if mask[i]:
            if continued and not _FENCE_OPEN_RE.match(line):
                continued = line.rstrip().endswith("\\")
                lines[i] = None  # a continuation of a command that is now a comment
                continue
            continued = False
            if _FENCE_OPEN_RE.match(line) or not UPSTREAM_PATH_ANY_RE.search(line):
                continue
            continued = line.rstrip().endswith("\\")
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{indent}# {_pointer(UPSTREAM_PATH_ANY_RE.search(line).group(0))}"
            count += 1
            continue
        continued = False
        if not UPSTREAM_PATH_RE.search(line):
            continue

        def _link(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return f"{match.group(1)} {_pointer(match.group(2))}"

        line = re.sub(r"\[([^\]]*)\]\(((?:\./)?(?:references|scripts|assets|skills)/[^)]*)\)", _link, line)

        def _span(match: re.Match[str]) -> str:
            nonlocal count
            span = match.group(0)
            found = UPSTREAM_PATH_RE.search(span)
            if found:
                count += 1
                return _pointer(found.group(0))
            return span

        line = re.sub(r"`[^`\n]+`", _span, line)

        def _bare(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return _pointer(match.group(0))

        line = UPSTREAM_PATH_RE.sub(_bare, line)
        lines[i] = line
    return "\n".join(line for line in lines if line is not None), count


#: The pointer texts a fence line can carry once the path rules have run (the script pointer also
#: in its command form, ``python (upstream helper script, not vendored: x.py)``).
_LINE_POINTERS = (SCRIPT_POINTER.rstrip(")"), REFERENCE_POINTER, SKILL_POINTER)
_DEAD_LINE_RE = re.compile(r"^\s*(?:#.*|.*(?:" + "|".join(map(re.escape, _LINE_POINTERS)) + r").*)?$")


def _collapse_dead_fences(body: str) -> tuple[str, int]:
    """A fence left with nothing runnable becomes one pointer, and its lead-in goes with it.

    After the path rules, a fence that ran upstream scripts holds only comments and marked commands
    ("# (upstream helper script, not vendored)", ``python (upstream helper script, not vendored:
    x.py) --list``): nothing in it can be run, and a reader who runs it anyway gets ``command not
    found`` (hunt 2026-09-30, u13k2-packs-12). Only a fence that carries at least one pointer is
    touched -- a fence of plain comments is someone's prose, not a casualty of these rules.
    """
    lines = body.split("\n")
    out: list[str] = []
    collapsed = 0
    cursor = 0
    for start, end in fence_blocks(lines):
        out.extend(lines[cursor:start])
        inner = lines[start + 1 : end]
        has_pointer = any(marker in line for line in inner for marker in _LINE_POINTERS)
        if has_pointer and all(_DEAD_LINE_RE.match(line) for line in inner):
            only_references = all(REFERENCE_POINTER in line for line in inner if any(m in line for m in _LINE_POINTERS))
            pointer = REFERENCE_FENCE_POINTER if only_references else SCRIPT_FENCE_POINTER
            _drop_lead_in(out, pointer)
            out.append(pointer)
            collapsed += 1
        else:
            out.extend(lines[start : end + 1])
        cursor = end + 1
    out.extend(lines[cursor:])
    return "\n".join(_collapse_pointers(out)), collapsed


def _point_at_folded(body: str, ref_rel: str, heading: str) -> str:
    """Mentions of a folded reference file point at its section instead of at a missing file."""
    target = f'(see the section "Reference: {heading}" below)'
    name = ref_rel.split("/")[-1]
    tail = "references/" + name
    body = re.sub(r"`[^`\n]*" + re.escape(tail) + r"[^`\n]*`", target, body)
    body = re.sub(r"\[([^\]]*)\]\((?:\./)?" + re.escape(tail) + r"\)", lambda m: f"{m.group(1)} {target}", body)
    return body.replace(tail, target)


def _clean_reference(raw: str) -> tuple[str, str]:
    """A reference file, made safe the same way a body is: no H1, no rules, no install fences."""
    if raw.lstrip().startswith("---"):
        front, body = split_frontmatter(raw)
        heading = str(front.get("name") or "")
    else:
        body, heading = raw, ""
    body, h1 = _strip_first_h1(body)
    heading = h1 or heading or "Reference"
    body, _ = _replace_install_fences(body)
    body, _ = _rewrite_inline_installs(body)
    body, _ = _remove_rules(body)
    return _demote_headings(body).strip("\n"), heading


def _apply_edits(text: str, replacements: list[dict], excisions: list[str]) -> tuple[str, int, int]:
    """Exact-substring edits; a listed substring that is absent is a refusal, never a no-op."""
    n_replace = 0
    for entry in replacements:
        old, new = str(entry["old"]), str(entry.get("new", ""))
        if old not in text:
            raise MergeRefusal(f"replace: substring not found: {old[:60]!r}")
        n_replace += text.count(old)
        text = text.replace(old, new)
    n_excise = 0
    for old in excisions:
        old = str(old)
        if old not in text:
            raise MergeRefusal(f"excise: substring not found: {old[:60]!r}")
        n_excise += text.count(old)
        if "\n" in old:
            text = text.replace(old, "")
            continue
        kept: list[str] = []
        for line in text.split("\n"):
            if old in line:
                line = line.replace(old, "")
                if not line.strip():
                    continue  # the excision emptied the line; an empty line would read as a paragraph break
            kept.append(line)
        text = "\n".join(kept)
    return text, n_replace, n_excise


def _apply_line_edits(raw: str, edits: list[dict], dest: str) -> tuple[str, int]:
    """Replace or delete whole source lines addressed by number and by the sha256 of their text.

    This is how an edit removes an excluded identifier without the manifest -- which lives under
    ``packs/`` and is scanned for those identifiers -- ever quoting it. A line that is not where the
    manifest says, or not the text it says, refuses the file rather than editing something else.
    """
    if not edits:
        return raw, 0
    lines = raw.split("\n")
    for entry in sorted(edits, key=lambda e: int(e["line"]), reverse=True):
        number = int(entry["line"])
        if not 0 < number <= len(lines):
            raise MergeRefusal(f"{dest}: line_edits: line {number} is outside the file")
        actual = _sha256(lines[number - 1].encode("utf-8"))
        if actual != str(entry["sha256"]):
            raise MergeRefusal(f"{dest}: line_edits: line {number} is not the line the manifest recorded (moved?)")
        replacement = entry.get("new")
        if replacement is None:
            del lines[number - 1]
        else:
            lines[number - 1] = str(replacement)
    return "\n".join(lines), len(edits)


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def render_skill(
    pack_key: str, pack: dict, skill: dict, clone: Path, absent_skills: dict[str, bool] | None = None
) -> Rendered:
    """One skill, start to finish. Raises :class:`MergeRefusal` rather than emitting a bad file.

    ``absent_skills`` are the skill names a document may not send the reader to (see
    :func:`absent_skill_names`); the post-check refuses a document that still does."""
    dest = str(skill["dest"])
    source = str(skill["source"])
    src_path = clone / source
    if not src_path.is_file():
        raise MergeRefusal(f"{dest}: source missing from the clone: {source}")
    if EXCLUDED_IDENTIFIER_RE.search(dest) or EXCLUDED_IDENTIFIER_RE.search(source):
        raise MergeRefusal(f"{dest}: the skill's own name is an excluded identifier")

    raw = src_path.read_text(encoding="utf-8")
    raw, n_line_edits = _apply_line_edits(raw, list(skill.get("line_edits") or []), dest)
    try:
        front, body = split_frontmatter(raw)
    except MergeRefusal as exc:
        raise MergeRefusal(f"{dest}: {exc}") from exc
    for key in ("name", "description"):
        if not isinstance(front.get(key), str) or not front[key].strip():
            raise MergeRefusal(f"{dest}: frontmatter lacks a usable {key!r}")
    mods: dict[str, int] = {}
    # The manifest may replace the Short Description outright. This is the one line the second
    # retrieval pass ranks on, and the live measurement (D-056 addendum) showed an upstream
    # description written to *sell* a library attaching it to tasks the platform's own tools
    # already cover; a scoped sentence is the remedy that keeps the document.
    override = skill.get("short_description")
    if override is not None:
        if not isinstance(override, str) or len(_one_line(override)) < MIN_SUMMARY_CHARS:
            raise MergeRefusal(f"{dest}: short_description override is not a string of >= {MIN_SUMMARY_CHARS} chars")
        summary = _one_line(override)
        mods["short_description_overridden"] = 1
    else:
        summary = _one_line(front["description"])
    wrapped_licence = _wrapped_licence_line(skill.get("wrapped_tool_licence"), front.get("license"))
    allowed_api_keys = frozenset(str(name) for name in (skill.get("may_name_api_keys") or []))
    body, h1 = _strip_first_h1(body)
    title = h1 or str(front["name"])
    body, mods["sections_dropped"] = _drop_sections(body, tuple(pack.get("drop_sections") or ()))
    body, mods["blockquotes_removed"] = _drop_blockquote(body, "Check before installing")
    body, mods["install_fences_replaced"] = _replace_install_fences(body)
    body, mods["inline_installs_rewritten"] = _rewrite_inline_installs(body)
    body, mods["rules_removed"] = _remove_rules(body)

    folded: list[str] = []
    for ref_rel in skill.get("references") or []:
        ref_path = clone / str(ref_rel)
        if not ref_path.is_file():
            raise MergeRefusal(f"{dest}: reference missing from the clone: {ref_rel}")
        ref_body, heading = _clean_reference(ref_path.read_text(encoding="utf-8"))
        body = _point_at_folded(body, str(ref_rel), heading)
        body = body.rstrip("\n") + f"\n\n## Reference: {heading}\n\n" + ref_body + "\n"
        folded.append(heading)
    mods["references_folded"] = len(folded)

    body, mods["upstream_paths_marked"] = _rewrite_upstream_paths(body)
    body, commands_marked = _rewrite_upstream_commands(body)
    mods["upstream_paths_marked"] += commands_marked
    body, mods["dead_fences_collapsed"] = _collapse_dead_fences(body)

    # Edits apply to the summary as well as the body: an excluded name can sit in the frontmatter
    # description, which is exactly the line the retriever reads.
    replacements = list(skill.get("replace") or [])
    excisions = list(skill.get("excise") or [])
    combined = summary + "\n\x00\n" + body
    combined, mods["replacements"], mods["excisions"] = _apply_edits(combined, replacements, excisions)
    mods["line_edits"] = n_line_edits
    summary, body = combined.split("\n\x00\n", 1)
    summary = _one_line(summary)

    body = body.strip("\n")
    first = next((line for line in body.split("\n") if line.strip()), "")
    if not first.startswith("#"):
        # The loader's metadata scanner treats every non-heading line after ``## Metadata`` as a
        # continuation of the last field until it meets a heading. A body that opens with prose
        # therefore needs a heading of its own, or the prose is read as part of ``Modifications``.
        body = "## Overview\n\n" + body
        mods["overview_heading_added"] = 1

    commercial = _one_line(skill.get("commercial_use") or pack.get("commercial_use") or DEFAULT_COMMERCIAL_USE)
    source_url = f"{pack['upstream'].rstrip('/')}/blob/{pack['commit']}/{source}"
    text = (
        f"# {title}\n\n"
        "## Metadata\n\n"
        f"**Short Description**: {summary}\n"
        f"**Source**: {source_url}\n"
        f"**License**: {_one_line(pack['licence_line'])}\n"
        f"**Wrapped Tool License**: {_one_line(wrapped_licence)}\n"
        f"**Commercial Use**: {commercial}\n"
        "**Tier**: 2\n"
        f"**Modifications**: {_modifications_sentence(mods, pack.get('drop_sections') or (), folded)}\n\n"
        "---\n\n"
        f"{body}\n"
    )
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    _post_check(dest, text, summary, allowed_api_keys, absent_skills)
    return Rendered(dest=dest, text=text, title=title, summary=summary, modifications=mods)


def _wrapped_licence_line(override: object, frontmatter_licence: object) -> str:
    """The Wrapped Tool License value, saying where it came from.

    It used to be ``override or frontmatter license`` followed by "(as stated in the upstream
    frontmatter)" -- for every document, overrides included. But both upstreams use the frontmatter
    ``license`` field for the skill *text* in some files and for the wrapped tool in others, so
    kdense/pathml read "MIT" for a GPL-2.0 library its own body names, and omero-integration "MIT"
    for GPL-2.0-or-later omero-py (hunt 2026-09-30, u13k2-packs-15 / u13k2-packs-extra-13). The
    frontmatter value is still shown -- it is evidence -- but never as the tool's licence.
    """
    if override:
        return f"{_one_line(str(override))} (recorded in packs/MANIFEST.yaml)"
    if frontmatter_licence:
        return (
            f'not stated upstream for the tool; the upstream frontmatter `license` field reads "'
            f'{_one_line(str(frontmatter_licence))}", and upstream uses that field for the skill text in some '
            "files and for the tool in others, so it is not taken as the tool's licence"
        )
    return "not stated upstream"


def _modifications_sentence(mods: dict[str, int], dropped: tuple | list, folded: list[str]) -> str:
    parts = [
        "re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; "
        "upstream frontmatter reduced to this header; first H1 replaced by the title above"
    ]
    if mods.get("sections_dropped"):
        parts.append(f"{mods['sections_dropped']} upstream section(s) dropped ({', '.join(dropped)})")
    if mods.get("blockquotes_removed"):
        parts.append(f"{mods['blockquotes_removed']} 'check before installing' blockquote(s) removed")
    if mods.get("install_fences_replaced"):
        parts.append(f"{mods['install_fences_replaced']} install fence(s) replaced by a provisioning pointer")
    if mods.get("inline_installs_rewritten"):
        parts.append(f"{mods['inline_installs_rewritten']} inline install span(s) rewritten")
    if mods.get("rules_removed"):
        parts.append(f"{mods['rules_removed']} horizontal rule(s) removed")
    if folded:
        parts.append(f"{len(folded)} upstream reference file(s) folded in ({', '.join(folded)})")
    if mods.get("upstream_paths_marked"):
        parts.append(f"{mods['upstream_paths_marked']} upstream script/reference path(s) marked as not vendored")
    if mods.get("dead_fences_collapsed"):
        parts.append(f"{mods['dead_fences_collapsed']} fence(s) left with nothing runnable replaced by a pointer")
    if mods.get("replacements") or mods.get("excisions"):
        parts.append(
            f"{mods.get('replacements', 0)} manifest replacement(s) and {mods.get('excisions', 0)} excision(s) applied"
        )
    if mods.get("line_edits"):
        parts.append(f"{mods['line_edits']} source line(s) replaced or removed by the manifest")
    if mods.get("short_description_overridden"):
        parts.append("the Short Description is the manifest's, not the upstream frontmatter's")
    if mods.get("overview_heading_added"):
        parts.append("an Overview heading added above the opening prose")
    return "; ".join(parts) + "."


def skill_reference_pattern(name: str, refused: bool = True) -> re.Pattern[str]:
    """Where a document sends the reader to the skill ``name``.

    A hyphenated name the manifest *refused* (``database-lookup``) is a skill name and nothing else,
    so it counts wherever it stands as a word. Any other name -- a plain one, which is usually also
    a library (``scanpy``), or one that was simply never merged (``scikit-learn`` is a skill upstream
    and a library everywhere) -- counts only next to the word "skill": "the scanpy skill",
    "**omics-plotting** SKILL". Naming the library is fine; routing to the skill is not.
    """
    escaped = re.escape(name)
    if refused and "-" in name:
        return re.compile(rf"(?<![\w/.-]){escaped}(?![\w-])", re.IGNORECASE)
    return re.compile(
        rf"(?<![\w/.-])[`*]*{escaped}[`*]*\s+skills?\b|\bskills?\s+[`*]*{escaped}[`*]*(?![\w-])", re.IGNORECASE
    )


def _post_check(
    dest: str,
    text: str,
    summary: str,
    allowed_api_keys: frozenset[str] = frozenset(),
    absent_skills: dict[str, bool] | None = None,
) -> None:
    if text.startswith("---"):
        raise MergeRefusal(f"{dest}: the emitted file starts with frontmatter")
    if len(text.encode("utf-8")) > MAX_PACK_DOC_BYTES:
        raise MergeRefusal(f"{dest}: {len(text.encode('utf-8'))} bytes exceeds MAX_PACK_DOC_BYTES={MAX_PACK_DOC_BYTES}")
    if not fences_pair_up(text):
        raise MergeRefusal(f"{dest}: a code fence never closes")
    hit = EXCLUDED_IDENTIFIER_RE.search(text)
    if hit:
        raise MergeRefusal(f"{dest}: excluded identifier {hit.group(0)!r} survives")
    if len(summary) < MIN_SUMMARY_CHARS:
        raise MergeRefusal(f"{dest}: short description is {len(summary)} chars, floor {MIN_SUMMARY_CHARS}")
    if PERSONAL_PATH_RE.search(text):
        raise MergeRefusal(f"{dest}: a personal path survives: {PERSONAL_PATH_RE.search(text).group(0)!r}")
    if UPSTREAM_PATH_ANY_RE.search(text):
        raise MergeRefusal(f"{dest}: an upstream path survives: {UPSTREAM_PATH_ANY_RE.search(text).group(0)!r}")
    survivors = install_lines(text.split("\n---\n", 1)[1] if "\n---\n" in text else text)
    if survivors:
        number, line = survivors[0]
        raise MergeRefusal(f"{dest}: an install line survives at body line {number}: {line.strip()[:80]!r}")
    body = text.split("\n---\n", 1)[1] if "\n---\n" in text else ""
    if any(line.strip() == "---" for line in body.split("\n")):
        raise MergeRefusal(f"{dest}: a horizontal rule survives in the body")
    if re.search(r"^(?:allowed-tools|compatibility|skill-author):", text, re.MULTILINE):
        raise MergeRefusal(f"{dest}: an upstream frontmatter field survives")
    hit = DOTENV_RE.search(body)
    if hit:
        line = body.count("\n", 0, hit.start()) + 1
        raise MergeRefusal(f"{dest}: a dotenv file is named at body line {line} -- a document may not point at .env")
    undeclared = sorted(set(API_KEY_NAME_RE.findall(body)) - allowed_api_keys)
    if undeclared:
        raise MergeRefusal(f"{dest}: names API-key variable(s) the manifest does not declare: {undeclared}")
    hit = DENIED_LINK_HOST_RE.search(body)
    if hit:
        raise MergeRefusal(f"{dest}: links a URL shortener or download aggregator: {hit.group(0)!r}")
    # A document that sends the reader to a skill the manifest refused or measured out (or one that
    # was never merged) steers the agent toward a capability that is deliberately absent, and a model
    # that knows the upstream repo may try to fetch it (hunt 2026-09-30, u13k2-packs-14). The Short
    # Description -- the line the second retrieval pass ranks on, where upstream does its routing
    # ("statistical-analysis for test guidance") -- is held to the strict form for every hyphenated name.
    for name, refused in sorted((absent_skills or {}).items()):
        hit = skill_reference_pattern(name, refused or "-" in name).search(summary)
        hit = hit or skill_reference_pattern(name, refused).search(body)
        if hit:
            raise MergeRefusal(f"{dest}: sends the reader to the absent skill {name!r}: {hit.group(0)!r}")


# --------------------------------------------------------------------------- the manifest


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("packs"), dict):
        raise MergeRefusal(f"{path}: not a manifest")
    excluded = manifest.get("excluded") or {}
    declared = int((excluded.get("identifiers") or {}).get("count") or 0)
    if declared != len(EXCLUDED_IDENTIFIERS):
        raise MergeRefusal(
            f"{path}: excluded.identifiers.count is {declared}; merge_packs.EXCLUDED_IDENTIFIERS has "
            f"{len(EXCLUDED_IDENTIFIERS)} -- the two records disagree"
        )
    seen: set[str] = set()
    for key, pack in manifest["packs"].items():
        for needed in (
            "upstream",
            "commit",
            "licence_file",
            "licence_sha256",
            "licence_copy",
            "licence_line",
            "skills",
        ):
            if needed not in pack:
                raise MergeRefusal(f"{path}: pack {key!r} lacks {needed!r}")
        for skill in pack["skills"]:
            dest = str(skill.get("dest") or "")
            if not re.fullmatch(rf"{re.escape(key)}/[A-Za-z0-9][A-Za-z0-9_.-]*\.md", dest) or Path(dest).stem.isupper():
                raise MergeRefusal(f"{path}: skill dest {dest!r} is not '{key}/<stem>.md'")
            if dest in seen:
                raise MergeRefusal(f"{path}: duplicate dest {dest!r}")
            seen.add(dest)
    return manifest


def manifest_dests(manifest: dict) -> set[str]:
    return {str(skill["dest"]) for pack in manifest["packs"].values() for skill in pack["skills"]}


def manifest_api_key_names(manifest: dict) -> dict[str, frozenset[str]]:
    """``dest -> the API-key variable names that skill's text may carry`` (``may_name_api_keys:``)."""
    return {
        str(skill["dest"]): frozenset(str(name) for name in (skill.get("may_name_api_keys") or []))
        for pack in manifest["packs"].values()
        for skill in pack["skills"]
    }


#: The keys under ``excluded.refused.<pack>`` that name whole skills (a list of names, or a
#: name -> reason mapping). ``never_copied`` names paths, not skills, and is not one of them.
_REFUSED_SKILL_KEYS = (
    "struck_in_verification",
    "not_recommended_yet",
    "measured_out",
    "refused_in_audit",
    "names_that_carry_no_identifier",
)


def absent_skill_names(manifest: dict, src: Path | None = None) -> dict[str, bool]:
    """``name -> refused``: every skill a document may not send the reader to.

    ``True`` for the skills the manifest refused or measured out (``excluded.refused``); with the
    clones (``src``), ``False`` for every other upstream skill directory that was never merged. An
    admitted skill is never absent, whatever a refusal list says.
    """
    admitted = {Path(dest).stem for dest in manifest_dests(manifest)}
    names: dict[str, bool] = {}
    if src is not None:
        for key, pack in manifest["packs"].items():
            for skill_md in sorted((src / str(pack.get("clone_dir") or key) / "skills").rglob("SKILL.md")):
                names[skill_md.parent.name] = False
    for refusals in ((manifest.get("excluded") or {}).get("refused") or {}).values():
        for key in _REFUSED_SKILL_KEYS:
            entry = (refusals or {}).get(key)
            for name in entry.keys() if isinstance(entry, dict) else entry or ():
                names[str(name)] = True
    return {name: refused for name, refused in names.items() if name not in admitted}


def _clone_head(clone: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MergeRefusal(f"{clone}: cannot read HEAD ({exc})") from exc


def render_pack(key: str, pack: dict, clone: Path, absent_skills: dict[str, bool] | None = None) -> list[Rendered]:
    head = _clone_head(clone)
    if head != str(pack["commit"]):
        raise MergeRefusal(f"pack {key!r}: clone HEAD {head} is not the manifest commit {pack['commit']}")
    licence = clone / str(pack["licence_file"])
    if not licence.is_file():
        raise MergeRefusal(f"pack {key!r}: licence file {pack['licence_file']} missing from the clone")
    if _sha256(licence.read_bytes()) != str(pack["licence_sha256"]):
        raise MergeRefusal(f"pack {key!r}: licence file sha256 is not the manifest's")
    rendered = [render_skill(key, pack, skill, clone, absent_skills) for skill in pack["skills"]]
    summaries: dict[str, str] = {}
    for doc in rendered:
        if doc.summary in summaries:
            raise MergeRefusal(f"{doc.dest}: short description duplicates {summaries[doc.summary]}")
        summaries[doc.summary] = doc.dest
    return sorted(rendered, key=lambda d: d.dest)


def check_dormant_excisions(manifest: dict, src: Path) -> list[str]:
    """The dormant one-liner excisions: report whether each line is still where the intake found it.

    Verified by the line's sha256, so this record can say which line it means without carrying
    the string it exists to keep out of the tree.
    """
    notes: list[str] = []
    for entry in (manifest.get("excluded") or {}).get("line_excisions") or []:
        pack = manifest["packs"].get(entry["pack"]) or {}
        clone = src / str(pack.get("clone_dir") or entry["pack"])
        path = clone / str(entry["source"])
        if not path.is_file():
            notes.append(f"dormant {entry['pack']}:{entry['source']}: file absent from the clone")
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
        number = int(entry["line"])
        actual = _sha256(lines[number - 1].encode("utf-8")) if 0 < number <= len(lines) else ""
        state = "still present, dormant" if actual == entry["line_sha256"] else "MOVED -- re-verify before any widening"
        notes.append(f"dormant {entry['pack']}:{entry['source']}:{number}: {state}")
    return notes


# --------------------------------------------------------------------------- disk verification


def verify_on_disk(manifest: dict, packs_dir: Path = PACKS_DIR, repo_root: Path = REPO_ROOT) -> list[str]:
    """Everything ``--check`` can say without a clone. Returns the problems; empty means clean."""
    problems: list[str] = []
    docs = on_disk_documents(packs_dir)
    dests = manifest_dests(manifest)
    on_disk = {f"{d}.md" for d in docs}
    if dests != on_disk:
        problems.append(
            f"manifest/disk mismatch: only in manifest={sorted(dests - on_disk)} only on disk={sorted(on_disk - dests)}"
        )
    summaries: dict[str, str] = {}
    allowed_keys = manifest_api_key_names(manifest)
    absent = absent_skill_names(manifest)  # no clone here: the manifest's refusals only
    for doc_id, text in docs.items():
        match = re.search(r"^\*\*Short Description\*\*: (.*)$", text, re.MULTILINE)
        summary = match.group(1).strip() if match else ""
        try:
            _post_check(f"{doc_id}.md", text, summary, allowed_keys.get(f"{doc_id}.md", frozenset()), absent)
        except MergeRefusal as exc:
            problems.append(str(exc))
        pack = doc_id.split("/", 1)[0]
        if (pack, summary) in summaries:
            problems.append(f"{doc_id}: short description duplicates {summaries[(pack, summary)]}")
        summaries[(pack, summary)] = doc_id
        commit = str((manifest["packs"].get(pack) or {}).get("commit") or "")
        if commit and f"/blob/{commit}/" not in text:
            problems.append(f"{doc_id}: Source line does not carry the pack's pinned commit")
    for key, pack in manifest["packs"].items():
        copy = repo_root / str(pack["licence_copy"])
        if not copy.is_file():
            problems.append(f"pack {key!r}: licence copy {pack['licence_copy']} is missing")
        elif _sha256(copy.read_bytes()) != str(pack["licence_sha256"]):
            problems.append(f"pack {key!r}: licence copy {pack['licence_copy']} does not match the manifest sha256")
    return problems


# --------------------------------------------------------------------------- entry point


def _write_atomic(path: Path, data: bytes) -> None:
    """Write through a ``.partial`` sibling and ``os.replace``: a reader -- the loader, the corpus lens
    -- never sees half a document, and an identical file is left untouched."""
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    os.replace(partial, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, help="directory holding one clone per pack (named by clone_dir)")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--check", action="store_true", help="verify and report; write nothing")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except MergeRefusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    packs_dir = args.manifest.parent
    repo_root = REPO_ROOT
    rendered: dict[str, Rendered] = {}
    if args.src is not None:
        absent = absent_skill_names(manifest, args.src)
        try:
            for key, pack in manifest["packs"].items():
                clone = args.src / str(pack.get("clone_dir") or key)
                for doc in render_pack(key, pack, clone, absent):
                    rendered[doc.dest] = doc
        except MergeRefusal as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        for note in check_dormant_excisions(manifest, args.src):
            print(note)

    if args.check:
        problems = verify_on_disk(manifest, packs_dir, repo_root)
        if rendered:
            disk = on_disk_documents(packs_dir)
            for dest, doc in rendered.items():
                doc_id = dest[: -len(".md")]
                if disk.get(doc_id) != doc.text:
                    problems.append(f"{dest}: the render differs from the file on disk")
        for problem in problems:
            print(f"DRIFT: {problem}")
        print(f"tier-2 documents on disk: {len(on_disk_documents(packs_dir))}")
        print(f"KNOW_HOW_PACKS_HASH {pack_digest(document_digests(on_disk_documents(packs_dir)))}")
        return 1 if problems else 0

    if args.src is None:
        parser.error("--src is required to write (use --check to verify what is on disk)")

    # write: licence copies first, then the documents, then remove generated files the manifest no
    # longer names -- the manifest is the source of truth for this directory
    for key, pack in manifest["packs"].items():
        clone = args.src / str(pack.get("clone_dir") or key)
        _write_atomic(repo_root / str(pack["licence_copy"]), (clone / str(pack["licence_file"])).read_bytes())
    for dest, doc in rendered.items():
        _write_atomic(packs_dir / dest, doc.text.encode("utf-8"))
        counts = ", ".join(f"{k}={v}" for k, v in doc.modifications.items() if v)
        print(f"wrote {dest} ({len(doc.text.encode('utf-8'))} bytes; {counts})")
    for stale in sorted(packs_dir.glob("*/*.md")):
        rel = f"{stale.parent.name}/{stale.name}"
        if rel not in rendered:
            stale.unlink()
            print(f"removed {rel}: not in the manifest")
    problems = verify_on_disk(manifest, packs_dir, repo_root)
    for problem in problems:
        print(f"DRIFT: {problem}")
    print(f"KNOW_HOW_PACKS_HASH {pack_digest(document_digests(on_disk_documents(packs_dir)))}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
