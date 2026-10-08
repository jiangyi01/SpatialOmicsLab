"""Citations a research report is allowed to print, and the ones it must confess to instead.

A model asked to ground a finding in the literature will produce something DOI-shaped whether or
not the paper exists. So a reference earns the References section by being looked up and found --
``validated is True`` and nothing else. Everything else is printed too, in a section that says so,
with the identifier visible: dropping an unverifiable claim silently would hide that it was made,
which is the failure this whole layer exists to prevent.

Three verdicts, never two:

``True``
    The registry returned metadata for this identifier. The reference is real, and the title and
    authors below are the registry's, not the model's.
``False``
    The registry answered, and answered that it has no such record. This is a refutation.
``None``
    Nobody answered -- the network was down, the service rate-limited us, the lookup raised. That
    is not evidence about the paper, and reporting it as ``False`` would invent a refutation out
    of an outage.

The lookups are injected. The defaults are the two dependency-free functions in
:mod:`spatialomicsgym.tool.omics_skills`; every test passes its own, so nothing here needs a
network to be exercised.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "ARXIV_DOI_PREFIX",
    "ARXIV_RE",
    "DOI_RE",
    "Citation",
    "CitationLedger",
    "harvest",
    "normalise_doi",
]

#: arXiv mints its DOIs under this prefix, and registers them with **DataCite**. Crossref -- which
#: is what `validate_doi` asks -- has never heard of them, so a Crossref lookup of a perfectly good
#: preprint returns "not found". Reporting that as `False` would refute a paper that exists. These
#: are routed to arXiv's own API instead, which is the registry that actually minted them.
ARXIV_DOI_PREFIX = "10.48550/arxiv."

#: Deliberately not anchored on a trailing boundary: a DOI may legally contain almost anything, so
#: the trailing junk is stripped afterwards by `normalise_doi` rather than guessed at by the regex.
DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>{}\\^`\[\]]+", re.IGNORECASE)

#: An arXiv id is only recognised when something says it is one. A bare `2301.00001` is also a
#: price, a version number and a date, and harvesting those would fill the ledger with noise that
#: then fails to verify and is reported to the user as an unverified claim they never made.
ARXIV_RE = re.compile(
    r"(?:arxiv\s*[:/]\s*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(v\d+)?",
    re.IGNORECASE,
)

#: Punctuation that ends an English sentence but never ends a DOI.
_TRAILING = ".,;:)]}>\"'"

_CLAIM_CHARS = 240


def normalise_doi(value: str) -> str:
    """A DOI with its URL wrapper, trailing sentence punctuation and case noise removed."""
    text = str(value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
            break
    # Balance first, then strip: `(see 10.1/x)` must lose its `)`, but `10.1/x(2)` must keep it.
    while text and text[-1] in _TRAILING:
        if text[-1] == ")" and text.count("(") > text.count(")") - 1 and "(" in text:
            break
        text = text[:-1]
    return text.strip()


@dataclass(frozen=True)
class Citation:
    """One identifier the model produced, and what a registry said about it."""

    key: str
    kind: str  # "doi" | "arxiv"
    claim: str = ""
    validated: bool | None = None
    title: str = ""
    authors: tuple[str, ...] = ()
    venue: str = ""
    year: str = ""
    url: str = ""
    formatted: str = ""
    note: str = ""

    @property
    def label(self) -> str:
        """What to print for this citation, which is never nothing.

        An unverified citation still shows its identifier: the reader's next move is to paste it
        into a search bar, and a reference that hides its own identifier cannot be checked by the
        one person with the most reason to check it.
        """
        if self.formatted:
            return self.formatted
        bits = [b for b in (self.title, self.venue, self.year) if b]
        return (", ".join(bits) + f" ({self.display_id})") if bits else self.display_id

    @property
    def display_id(self) -> str:
        return f"arXiv:{self.key}" if self.kind == "arxiv" else f"https://doi.org/{self.key}"


def harvest(text: str, *, claim_chars: int = _CLAIM_CHARS) -> list[Citation]:
    """Every identifier in `text`, in the order it appears, each carrying its own sentence.

    The sentence is what makes an unverified reference actionable: "this DOI does not resolve" is a
    fact about a string, while "this DOI does not resolve, and it was the support for *cell type X
    is enriched in the tumour margin*" is a fact about the report.
    """
    body = str(text or "")
    seen: set[tuple[str, str]] = set()
    out: list[Citation] = []

    def claim_for(start: int, end: int) -> str:
        left = max(body.rfind(".", 0, start), body.rfind("\n", 0, start)) + 1
        right = min((i for i in (body.find(".", end), body.find("\n", end)) if i != -1), default=len(body))
        return " ".join(body[left : right + 1].split())[:claim_chars].strip()

    for match in ARXIV_RE.finditer(body):
        key = match.group(1)
        if ("arxiv", key) in seen:
            continue
        seen.add(("arxiv", key))
        out.append(Citation(key=key, kind="arxiv", claim=claim_for(*match.span())))

    for match in DOI_RE.finditer(body):
        found = match.group(0)
        raw = normalise_doi(found)
        if not raw:
            continue
        # Where the identifier really ends, which is not where the *match* ends. The character class
        # above accepts `)` and `.` -- it has to, a DOI may legally contain both -- so a DOI that
        # closes a sentence matches through `).` and the match end lands on the space of the *next*
        # sentence. Handing that to `claim_for` made the claim run to the sentence after the one
        # that cited anything, which in a report is a citation credited with a statement it was
        # never offered as support for. Measured on "...at the margin (10.1038/...). We then
        # clustered." -- the claim included the clustering.
        end = match.end() - (len(found) - len(raw))
        lowered = raw.lower()
        if lowered.startswith(ARXIV_DOI_PREFIX):
            kind, key = "arxiv", raw[len(ARXIV_DOI_PREFIX) :]
        else:
            kind, key = "doi", raw
        if not key or (kind, key) in seen:
            continue
        seen.add((kind, key))
        out.append(Citation(key=key, kind=kind, claim=claim_for(match.start(), end)))

    return out


#: The only two answers from ``validate_doi`` that are about the identifier rather than about the
#: lookup: Crossref's 404, and a string that cannot be a DOI at all. Its other errors -- "request
#: failed: ...", "HTTP 429", "HTTP 503", "unexpected Crossref response" -- are the registry not
#: answering. An allow-list rather than a deny-list, so an error text nobody has seen yet lands on
#: "could not be checked" and never on a refutation.
_DOI_REFUSALS = ("DOI not found in Crossref", "invalid DOI format")


def _loads(payload: object) -> dict:
    """The tool functions return JSON strings; a test may hand back the dict directly."""
    if isinstance(payload, dict):
        return payload
    try:
        value = json.loads(str(payload))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


@dataclass
class CitationLedger:
    """Every identifier a run produced, verified once and remembered."""

    validate_doi: Callable[[str], object] | None = None
    fetch_arxiv: Callable[[str], object] | None = None
    entries: dict[tuple[str, str], Citation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.validate_doi is None or self.fetch_arxiv is None:
            from spatialomicsgym.tool import omics_skills

            self.validate_doi = self.validate_doi or omics_skills.validate_doi
            self.fetch_arxiv = self.fetch_arxiv or omics_skills.fetch_arxiv_by_ids

    # ------------------------------------------------------------------ gathering

    def add(self, text: str) -> list[Citation]:
        """Harvest `text` and record anything new. Returns only what was new."""
        fresh = []
        for citation in harvest(text):
            slot = (citation.kind, citation.key)
            if slot in self.entries:
                continue
            self.entries[slot] = citation
            fresh.append(citation)
        return fresh

    # ------------------------------------------------------------------ verifying

    def verify_all(self) -> None:
        """Look up everything not yet looked up. Safe to call after every round."""
        for slot, citation in list(self.entries.items()):
            if citation.validated is None and not citation.note:
                self.entries[slot] = self.verify(citation)

    def verify(self, citation: Citation) -> Citation:
        checker = self._verify_arxiv if citation.kind == "arxiv" else self._verify_doi
        try:
            return checker(citation)
        except Exception as exc:  # an outage is not a refutation
            return replace(citation, validated=None, note=f"could not be checked ({type(exc).__name__})")

    def _verify_doi(self, citation: Citation) -> Citation:
        data = _loads(self.validate_doi(citation.key))
        if not data:
            return replace(citation, validated=None, note="the DOI registry did not answer")
        if not data.get("valid"):
            # Every miss used to be "no record of this DOI in Crossref", which turned a dropped
            # connection or a 429 into a report telling the reader the paper does not exist (hunt
            # 2026-09-30, u19-pa-tasks-research-3). A refutation now needs the lookup to say so:
            # ``not_found``, or -- from a lookup that folds outages into ``valid: false`` -- one of
            # the two refusal texts. An explicit ``valid: null`` and any other error are unchecked.
            # A payload with no error at all is still the registry's plain "no".
            error = str(data.get("error") or "").strip()
            refused = data.get("not_found") is True or (
                data.get("valid") is not None and (not error or error in _DOI_REFUSALS)
            )
            if not refused:
                agency = str(data.get("registration_agency") or "").strip()
                if agency.casefold() == "crossref":
                    # doi.org names Crossref itself: the DOI exists and Crossref's works index does not
                    # hold it yet. "registered with Crossref, not Crossref" contradicted itself (review
                    # of uT6-literature-2's repair, 2026-10-01).
                    return replace(
                        citation,
                        validated=None,
                        note="registered with Crossref, but Crossref's works index has no record of it yet",
                    )
                if agency:
                    # The registry did answer: the DOI exists, with another agency, and Crossref holds
                    # no record to read. "did not answer" said the opposite (hunt 2026-09-30, u19
                    # review of u19-pa-tasks-research-3).
                    return replace(
                        citation,
                        validated=None,
                        note=f"registered with {agency[:60]}, not Crossref, so its record could not be read here",
                    )
                why = f" ({error[:120]})" if error else ""
                return replace(citation, validated=None, note=f"the DOI registry did not answer{why}")
            return replace(citation, validated=False, note="no record of this DOI in Crossref")
        return replace(
            citation,
            validated=True,
            title=str(data.get("title") or ""),
            venue=str(data.get("journal") or ""),
            year=str(data.get("year") or ""),
            url=f"https://doi.org/{citation.key}",
            formatted=str(data.get("citation_apa") or ""),
        )

    def _verify_arxiv(self, citation: Citation) -> Citation:
        data = _loads(self.fetch_arxiv(citation.key))
        results = data.get("results") if isinstance(data.get("results"), list) else []
        if not data or not data.get("success", bool(results)):
            return replace(citation, validated=None, note="arXiv did not answer")
        if not results:
            return replace(citation, validated=False, note="arXiv has no paper with this id")
        paper = results[0] if isinstance(results[0], dict) else {}
        authors = tuple(str(a) for a in (paper.get("authors") or []) if a)
        return replace(
            citation,
            validated=True,
            title=" ".join(str(paper.get("title") or "").split()),
            authors=authors,
            venue=str(paper.get("journal_ref") or "arXiv"),
            year=str(paper.get("published") or "")[:4],
            url=str(paper.get("abs_url") or f"https://arxiv.org/abs/{citation.key}"),
        )

    # ------------------------------------------------------------------ reporting

    def verified(self) -> list[Citation]:
        return [c for c in self.entries.values() if c.validated is True]

    def unverified(self) -> list[Citation]:
        """Refuted *and* unchecked, together, because both are things the reader must not trust.

        They are one section rather than two on purpose: the reader's action is identical -- look
        it up yourself -- and splitting "we know it is wrong" from "we could not tell" invites the
        second pile to be read as the first.
        """
        return [c for c in self.entries.values() if c.validated is not True]

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.entries),
            "verified": len(self.verified()),
            "refuted": sum(1 for c in self.entries.values() if c.validated is False),
            "unchecked": sum(1 for c in self.entries.values() if c.validated is None),
        }
