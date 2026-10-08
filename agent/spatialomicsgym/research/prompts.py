"""The words a research run says to the agent, and the honest account of what it cannot say.

A research run is several ordinary agent turns in a row. Only the *first* prompt is written here;
from round two the instruction is :func:`spatialomicsgym.postanalysis.next_step.build_followup_prompt`,
which already names the tool output, the findings, the caveats and where to write figures. This
module adds the two things that function has no reason to know about -- that the run is an
investigation with a question behind it, and that every claim it makes has to be citable -- and it
adds them as text appended to that prompt rather than as an edit to it.

Three rules the prompts here obey, each of them paid for by a real failure:

* **ASCII only.** A Unicode bullet in a prompt has already produced a SyntaxError retry loop in the
  execute stage, so every string that reaches a model goes through :func:`ascii_only`. Two private
  one-line copies of this already exist (``next_step._ascii``, ``conversation._ascii``); this one is
  public because the research loop in the sibling module needs it too, and a fourth private copy is
  how a rule stops being one rule.
* **Short, action-first.** The same reason the rest of the prompt guidance is: a long preamble makes
  models skim the parameters.
* **Name what is missing before round one, not in the conclusion.** :func:`unavailable_grounding`
  answers "which literature calls would raise on this box", so the run can say up front that it
  cannot read full texts rather than discovering it four rounds in.
"""

from __future__ import annotations

import unicodedata

#: Grounding calls a research round is told about, paired with what a round would use them for.
#: Every one lives in :mod:`spatialomicsgym.tool.omics_skills`, which imports nothing outside the
#: standard library -- the reason the grounding story runs through that module and not through
#: :mod:`spatialomicsgym.tool.literature`, six of whose eight functions cannot be called here.
#: The two JGI lakehouse functions are deliberately absent: they query a metagenomics warehouse
#: that has nothing to say about a spatial transcriptomics slide.
_GROUNDING: tuple[tuple[str, str], ...] = (
    ("search_arxiv_advanced", "search arXiv by terms, author, category or date"),
    # bioRxiv only: the client queries api.biorxiv.org/details/biorxiv and takes no server, so a
    # medRxiv preprint can never be found (hunt 2026-09-30, uT3-atlases-16, uT6-literature-16).
    ("search_biorxiv", "search bioRxiv preprints (medRxiv is not covered)"),
    ("search_crossref_by_title", "find the DOI of a paper you can name"),
    ("fetch_arxiv_by_ids", "fetch title, authors and abstract for arXiv ids you already have"),
    ("validate_doi", "check a DOI resolves, and read back its title, journal and year"),
    ("format_citation_from_doi", "render a citation from a DOI"),
    ("assess_scientific_impact", "citation count and venue for a DOI"),
)

#: Public functions of :mod:`spatialomicsgym.tool.literature` whose absence changes what a research
#: run may claim, paired with the sentence the run says instead. A function missing from this table
#: is one nothing here promised, so its absence needs no disclosure.
_DISCLOSED: dict[str, str] = {
    "extract_pdf_content": "read the full text of a PDF",
    "extract_url_content": "read the body of a web page",
    "query_scholar": "count citations through Google Scholar",
    "search_google": "search the open web",
}

#: Appended to every round prompt. The ledger harvests DOIs and arXiv ids out of the answer text and
#: checks each one against the registry that issued it, so a claim whose identifier never appears in
#: the prose cannot be verified however true it is -- and one that appears but does not resolve is
#: shown to the user as unverified rather than quietly dropped.
#:
#: An instruction, not a guarantee, and the difference is written down here because it used to be
#: promised elsewhere as if it were one. Nothing enforces this rule: no round is retried for citing
#: nothing, no stop reason mentions citations, and the runs measured on this box -- including a
#: 668-second one -- finished with ``citations: []``. The report has always been honest about that
#: ("This run cited no literature, so there is nothing to verify"); the composer's wording now is
#: too, and says a run may cite none rather than promising "a cited report".
#:
#: What would close the gap is not a retry loop that nags the model. The literature calls already
#: RETURN the identifiers -- ``search_arxiv_advanced`` returns arXiv ids, ``search_crossref_by_title``
#: returns DOIs -- and the ledger would take them as they stand. What is missing is a route: the
#: loop reads a round's ``final`` text and nothing else, so a tool observation carrying a real DOI
#: is never offered to :meth:`CitationLedger.add`. Threading it is a change to what the loop reads,
#: not a change to this string, and it would have to keep "the model said it" separate from "a tool
#: fetched it" -- a citation the prose never made is a reference credited with a claim nobody made.
CITATION_RULE = (
    "CITING: when you state something you did not measure in this run, name its source inline as a "
    "DOI (10.xxxx/...) or an arXiv id (arXiv:2401.12345). Look it up first with one of the search "
    "calls above rather than writing an identifier from memory: every identifier you write is "
    "checked against the registry that issued it, and one that does not resolve is shown to the "
    "user as an unverified claim."
)


def ascii_only(text: object) -> str:
    """``text`` with every non-ASCII character replaced, never raising on odd input."""
    return str(text).encode("ascii", "replace").decode("ascii")


#: Bullet characters the scaffolding rule is about, mapped to the ASCII the rest of it uses.
_BULLETS = str.maketrans({"\u2022": "-", "\u25e6": "-", "\u25aa": "-", "\u2023": "-", "\u2043": "-"})


def the_users_words(text: object) -> str:
    """The person's own words, as they wrote them: only control characters and bullets go.

    :func:`ascii_only` is for this module's scaffolding, where a Unicode bullet once reached an
    ``<execute>`` block. Applied to the question and the dataset title it erased them: a Chinese
    question reached the agent as "Investigate this question: ???????" and a title that no longer
    matched the attached dataset -- while the report showed the original question over an
    unrelated analysis (hunt 2026-09-30, u06-history-15). The chat path never folded them.
    """
    kept = "".join(ch for ch in str(text or "") if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cs"))
    return kept.translate(_BULLETS)


def grounding_tools() -> list[tuple[str, str]]:
    """The subset of :data:`_GROUNDING` this installation can actually call.

    Derived, not asserted: a function renamed or removed in ``omics_skills`` drops out of the prompt
    instead of becoming an instruction to call something that no longer exists. An import failure of
    the whole module answers "none of them", which is the truth a research run then discloses.
    """
    try:
        from spatialomicsgym.tool import omics_skills
    except Exception:  # an uninstallable module offers no tools; the caller discloses that
        return []
    return [(name, use) for name, use in _GROUNDING if callable(getattr(omics_skills, name, None))]


def unavailable_grounding() -> list[dict[str, str]]:
    """What this box cannot do for a research run, and the ``pip install`` that would restore it.

    Each entry is ``{"function", "cannot", "install"}``. The answer comes from
    :func:`spatialomicsgym.tool.literature.requirement_for`, which asks about the *distribution* a
    function defers into its body -- ``hasattr`` says yes about every one of these and is how the
    same question was got wrong before.
    """
    try:
        from spatialomicsgym.tool import literature
    except Exception:
        # The module itself is unreachable, so every disclosed capability is missing and no
        # `pip install` we could name would be the fix. Say so rather than inventing one.
        return [{"function": fn, "cannot": what, "install": ""} for fn, what in sorted(_DISCLOSED.items())]
    missing: list[dict[str, str]] = []
    for function, what in sorted(_DISCLOSED.items()):
        try:
            requirement = literature.requirement_for(function)
            distribution = literature.distribution_for(function)
        except Exception:  # a module that cannot answer is not evidence against its own tool
            continue
        if requirement:
            # requirement_for can name a precondition ("a Claude model as the agent's LLM ...") no install
            # satisfies; only distribution_for names a package (hunt 2026-09-30, uT6-literature-5 review).
            install = f"pip install {distribution}" if distribution else ""
            missing.append({"function": function, "cannot": what, "install": install})
    return missing


def _tool_lines() -> list[str]:
    tools = grounding_tools()
    if not tools:
        return ["No literature search is available in this install, so cite nothing you cannot measure here."]
    return ["Literature calls available to you:"] + [f"  - {name}: {use}" for name, use in tools]


#: What a multi-round run is asked to fix before it starts running.
#:
#: Adapted from ``uditgoenka/autoresearch`` (MIT, commit 050e30dc), after Karpathy's autoresearch;
#: the full playbook is ``know_how/autonomous_iteration_loop.md``. The import is the discipline,
#: not the text -- and the reason it is here rather than only in the know-how corpus is that the
#: corpus is loaded into the SYSTEM prompt, which a scored run also sees, while these strings are
#: built only by a research run, which cannot start under benchmarking at all.
#:
#: Four rounds without a metric is four analyses, and the run reports the last one. A number and a
#: direction are what make round 2 comparable to round 1; the predicate is what makes "done" mean
#: something other than "the budget ran out".
#:
#: Asked only when ``rounds > 1``. With a single round there is no loop, and asking for a metric
#: buys nothing but a paragraph the model writes instead of the analysis.
_ITERATION_SETUP: tuple[str, ...] = (
    "Because this run has more than one round, fix three things in this round and state them:",
    "  METRIC: one number this analysis produces, that you can print from a file it wrote.",
    "  DIRECTION: higher_is_better or lower_is_better. Say it once; do not re-derive it later.",
    "  DONE WHEN: the condition that ends the run, written now and not recomputed each round.",
    "If the question has no measurable answer, say so in those words and do not invent a proxy.",
)

#: What every later round is told about the run it is continuing.
_ITERATION_ROUND: tuple[str, ...] = (
    "Change ONE thing this round. Two changes and nothing can say which one moved the number.",
    "Report the metric for this round, and whether the post-analysis guard passed.",
    "A round whose guard failed does not count as an improvement however good the number is.",
)

#: The stop vocabulary. The point is that each word says what the NUMBERS did, which is exactly
#: what a run reporting "the steps were exhausted" after one round of four does not.
_ITERATION_CLOSE: tuple[str, ...] = (
    "  - the metric: where it started, where it ended, and which round produced the end;",
    "  - which rounds you kept and which you discarded, and what the kept ones had in common;",
    "  - why the run stopped, in one of these words: converged (the DONE WHEN condition was met),",
    "    plateau (the metric stopped net-improving), ceiling (the rounds ran out), blocked",
    "    (something made further rounds impossible -- say what).",
)


def opening_prompt(question: str, *, dataset: str = "", rounds: int = 1) -> str:
    """Round one: what is being investigated, on what, and how many rounds it gets.

    The round budget is stated because it changes what a good first round looks like: with one round
    left the right move is to answer, and with four it is to establish the ground truth the later
    rounds build on. A model told nothing assumes one.
    """
    question = the_users_words(question).strip()
    dataset = the_users_words(dataset).strip()
    lines = [
        f"RESEARCH ROUND 1 OF {max(1, int(rounds))}.",
        f"Investigate this question: {question}",
    ]
    if dataset:
        lines.append(f"The data to investigate is the dataset named {dataset}, which is already attached.")
    lines += [
        "Start by establishing what the data actually is -- modality, size, what is measured -- and",
        "run the one analysis that most directly addresses the question. Save every figure and table",
        "you produce; later rounds build on them.",
        "Do not summarise the whole field. One question, one analysis, one honest result.",
    ]
    if int(rounds) > 1:
        lines += ["", *_ITERATION_SETUP]
    lines.append("")
    lines += _tool_lines()
    missing = unavailable_grounding()
    if missing:
        lines.append(
            "You cannot "
            + ", ".join(m["cannot"] for m in missing)
            + " in this install, so cite abstracts and metadata, and never claim to have read a full text."
        )
    lines += ["", CITATION_RULE]
    return "\n".join(lines)


def continuing_prompt(followup: str, *, round_number: int, rounds: int, progress: str = "") -> str:
    """Round two and later: ``build_followup_prompt``'s text, placed in the arc of the run.

    The follow-on prompt is used verbatim and is not edited: it is the same string the ordinary
    post-analysis path builds, and keeping it byte-identical is what stops this loop from becoming a
    second, drifting copy of that one. Everything this function adds sits around it.

    ``progress`` is what the run has established so far -- one line per round, the ledger the loop
    keeps. Optional and empty by default, so every existing caller is byte-identical; when it is
    given, the round is told what is already tried and what the incumbent is, which is the
    difference between a loop that improves and four analyses in a row. A round that is not told
    what the previous rounds did will re-try the setting that was already discarded.
    """
    number = max(2, int(round_number))
    total = max(number, int(rounds))
    head = f"RESEARCH ROUND {number} OF {total}."
    body = [ascii_only(followup).strip()]
    carried = ascii_only(progress).strip()
    if carried:
        body += ["", "WHAT THIS RUN HAS ESTABLISHED SO FAR:", carried]
    body += ["", *_ITERATION_ROUND]
    tail = [CITATION_RULE]
    if number >= total:
        tail.insert(0, "This is the last round, so end with the conclusion rather than another proposal.")
    return "\n".join([head, *body, "", *tail])


def closing_prompt(question: str, *, rounds_run: int) -> str:
    """The synthesis turn: one written answer to the question the run started with.

    Separate from the last analysis round on purpose. A round that both analyses and concludes
    produces a conclusion about the thing it just did; this one is handed the whole run.
    """
    return "\n".join(
        [
            "RESEARCH CONCLUSION.",
            f"You have run {max(1, int(rounds_run))} rounds of analysis on this question:",
            the_users_words(question).strip(),
            "",
            "Write the answer now, for a biologist who did not watch the run:",
            "  - what the data showed, naming the figures and tables you produced;",
            "  - how confident you are, and what would change the answer;",
            "  - what you could not determine, stated plainly rather than hedged.",
            *(_ITERATION_CLOSE if int(rounds_run) > 1 else ()),
            "Do not run further analyses. Do not repeat the reasoning trace.",
            "",
            CITATION_RULE,
        ]
    )
