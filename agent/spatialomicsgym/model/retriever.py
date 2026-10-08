import re

from langchain_core.messages import HumanMessage, SystemMessage

from spatialomicsgym.provider_backoff import invoke_with_backoff


def _record_call(usage, node: str, response) -> None:
    """Note one paid model call on the turn's ledger, if the caller supplied one.

    Defensive on purpose: this module is constructed directly by several tests and by
    ``tools_user`` scripts that pass no ledger at all, and telemetry must never be the reason a
    retrieval fails. A ledger that raises, or an object that only looks like one, costs the count
    and nothing else.
    """
    if usage is None:
        return
    try:
        usage.record(node, response)
    except Exception:
        pass


def _parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated index list, dropping only the tokens that aren't integers.

    The LLM occasionally emits a stray label inside an index list (e.g. ``[0, 3, foo, 5]``). The
    previous code wrapped the whole comprehension in ``contextlib.suppress(ValueError)``, so ONE
    bad token raised and was swallowed — silently discarding the ENTIRE category (every valid
    index with it). Here a bad token loses only itself; ``[0, 3, foo, 5]`` -> ``[0, 3, 5]``.
    """
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out


#: Send the instructions + resource list as a stable system message and the query as the user turn,
#: so the ~46k-token list is a byte-identical prompt prefix on every turn. A provider that caches
#: prefixes on its own serves it from cache, as both Azure deployments this box uses do (46,208 of
#: 46,381 input tokens on azure-gpt-5.4; gpt-6-astra below). Anthropic caches only up to a
#: ``cache_control`` breakpoint, which no request here set when checked on 2026-09-25, so on Claude
#: this is the precondition and not the saving (see ``llm._rerole_observation_turns``). ``None``
#: means "read the configuration" (``SOG_RETRIEVAL_QUERY_LAST``, default OFF); tests set it directly.
#:
#: OFF by default, on measurement (2026-09-25, 20 fixed prompts x 3 arms, gpt-6-astra): the split is
#: served 99.9% from cache on every call after the first, but it changes WHAT is selected beyond the
#: retriever's own noise -- tool-selection Jaccard 0.71 against 0.93 between two unchanged runs, and
#: 70.5 tools named per turn instead of 90.8. That is a change to the scored prompt, so it waits for
#: its own arm (HARNESS_LOG.md §6.3) instead of riding in as a cache optimisation.
QUERY_AFTER_RESOURCES: bool | None = None


def _query_last() -> bool:
    if QUERY_AFTER_RESOURCES is not None:
        return bool(QUERY_AFTER_RESOURCES)
    try:
        from spatialomicsgym.config import default_config

        return bool(getattr(default_config, "retrieval_query_last", False))
    except Exception:
        return False


#: Markdown decoration a chat model wraps its answer in. The prompt asks for a plain
#: ``TOOLS: [0, 3, 5]`` and models comply -- in bold, in backticks, under a bullet -- which puts
#: characters in the two places the parse depended on being empty: between the label and the colon
#: (``**TOOLS**:``) and between the colon and the list (``TOOLS: **[0, 3, 5]**``). Every one of
#: those replies used to select ZERO tools, ZERO data lake items and ZERO libraries, and nothing
#: downstream notices: the empty lists go straight into the system prompt and the turn runs with no
#: tools advertised. The fail-safe written to prevent that stranding could not fire, because it
#: asked its question the same too-strict way and so could not tell a lost parse from a deliberate
#: ``TOOLS: []``. Both regexes below share this class for that reason.
#:
#: Only whitespace and emphasis marks pass: a WORD between the colon and the bracket means the model
#: wrote prose rather than a list, which must stay unparseable -- reading across it would let one
#: category's header adopt the next category's indices.
_DECOR = r"[\s*`_]*"


def _wrote_a_nonempty_list(text: str) -> bool:
    """Did the model put anything at all inside a category's brackets?

    This separates ``TOOLS: []`` -- a deliberate "nothing here is relevant" -- from ``TOOLS:
    [tool_0]`` and ``TOOLS: [900]``, where the model did make a selection and we then failed to turn
    it into resources. All three end with zero resources selected; only the first one should.

    It reuses the extractor's own header-and-bracket regex on purpose: a guard that asks a different
    question than the parser it guards cannot see the parser's misses.
    """
    for category in ("TOOLS", "DATA_LAKE", "LIBRARIES", "KNOW[-_]HOW"):
        match = re.search(rf"{category}{_DECOR}:{_DECOR}\[(.*?)\]", text, re.IGNORECASE | re.DOTALL)
        if match and match.group(1).strip():
            return True
    return False


#: Header -> the key it fills in the selection dict.
_CATEGORY_KEYS: dict[str, str] = {
    "TOOLS": "tools",
    "DATA_LAKE": "data_lake",
    "LIBRARIES": "libraries",
    "KNOW[-_]HOW": "know_how",
}


def _tried_to_select(text: str, category: str) -> bool:
    """Did the model put a digit in brackets under THIS category's header?

    The extractor is deliberately strict: a WORD between the colon and the bracket means prose, and
    reading across it would let one category's header adopt the next category's indices. That
    strictness is right for the parser and wrong for the fail-safe, which only has to answer "did
    the model try?" -- and asked it with the parser's own regex, so every reply shape the parser
    missed was invisible to the guard as well. Measured, with the real retriever and a scripted
    LLM: ``TOOLS: the relevant ones are [0, 1]`` selected nothing, degraded to nothing, and printed
    no warning at all.

    So this probe is looser than the parser in exactly one way -- it allows prose and newlines
    between the header and the bracket -- and no looser. The tempered ``(?!...)`` stops the span at
    the next category header, which is what keeps it per-category: without it a ``TOOLS:`` with
    nothing under it would match ``DATA_LAKE``'s list and wrongly restore every tool.

    A bracket with no digit (``TOOLS: []``) is deliberately NOT a try. That is the model saying
    "nothing here is relevant", and it must keep meaning that.
    """
    other = "|".join(_CATEGORY_KEYS)
    pattern = rf"{category}{_DECOR}:(?:(?!(?:{other}){_DECOR}:).){{0,200}}?\[[^\]]*\d"
    return bool(re.search(pattern, text, re.IGNORECASE | re.DOTALL))


def _degrade_to_baseline(resources: dict) -> dict:
    """What a failed retrieval hands back: every tool, dataset and library -- and the know-how the
    agent would have started the turn with anyway.

    Both fail-safes below used to return the caller's dict untouched, which is right for tools,
    data and libraries (stranding a turn with none of those is the disaster they exist to prevent)
    and wrong for know-how, because `update_system_prompt_with_selected_resources` resolves each
    selected summary back to its FULL document. So "retrieval didn't parse" quietly became "load
    the entire 394 KB corpus", announced by one print -- demand-driven enrolment reverting to
    everything at the exact moment nobody is watching.

    The floor is the *baseline*: whatever `know_how.enrolment` would have put in the initial prompt
    under the mode in force. Under the default mode that is every document, so this returns exactly
    what it returned before; under `demand` it is the short list. Tying the two together means there
    is one answer to "what does this installation always want loaded", not two that can disagree.

    A copy, not the caller's own dict: the docstring below notes the returned dict need not carry
    all three keys, and that stays true of a copy while mutation of the caller's dict does not.
    """
    out = dict(resources or {})
    candidates = out.get("know_how")
    if not candidates:
        return out
    try:
        from spatialomicsgym.know_how.enrolment import ON_DEMAND_DOCUMENTS, enrolment_mode

        if enrolment_mode() == "all":
            return out
        keep = set(ON_DEMAND_DOCUMENTS)
    except Exception:
        return out  # unreadable enrolment config degrades the way it always did
    out["know_how"] = [c for c in candidates if isinstance(c, dict) and str(c.get("id") or "") in keep]
    return out


class ToolRetriever:
    """Retrieve tools from the tool registry."""

    def __init__(self):
        pass

    def prompt_based_retrieval(self, query: str, resources: dict, llm=None, usage=None) -> dict:
        """Use a prompt-based approach to retrieve the most relevant resources for a query.

        Args:
            query: The user's query
            resources: A dictionary with keys 'tools', 'data_lake', 'libraries', and 'know_how',
                      each containing a list of available resources
            llm: Optional LLM instance to use for retrieval (if None, will create a new one)
            usage: Optional ``agent.usage.TurnUsage`` to record this call on. It is a PAID call --
                one per turn, over a prompt that carries every tool description -- and it was
                missing from the ledger, so every turn under-reported itself by one call (two when
                a tier-2 pack pass runs). ``summary()``'s ``measured_calls`` / ``complete`` fields
                exist for exactly this: a call recorded without token counts still moves ``calls``
                and turns ``complete`` False, which is the honest answer, where omitting it looked
                like a complete count of a smaller number.

        Returns:
            A dictionary with the same keys, but containing only the most relevant resources

        """
        # Build prompt sections for available resources
        prompt_sections = []
        # The query is LAST. Every byte above it is the same for every turn of every run -- the tool
        # list alone is ~40k tokens -- and a provider's prompt cache covers only a byte-identical
        # PREFIX, so a query at the top made the whole list uncacheable on every turn. Measured on
        # azure-gpt-5.4: a repeated 11.7k-token prefix is served 98% from cache.
        query_last = _query_last()
        head = "" if query_last else f"\nUSER QUERY: {query}\n"
        prompt_sections.append(f"""
You are an expert biomedical research assistant. Your task is to select the relevant resources to help answer a user's query{" (it is given at the end, after the resources)" if query_last else ""}.
{head}
Below are the available resources. For each category, select items that are directly or indirectly relevant to answering the query.
Be generous in your selection - include resources that might be useful for the task, even if they're not explicitly mentioned in the query.
It's better to include slightly more resources than to miss potentially useful ones.

AVAILABLE TOOLS:
{self._format_resources_for_prompt(resources.get("tools", []))}

AVAILABLE DATA LAKE ITEMS:
{self._format_resources_for_prompt(resources.get("data_lake", []))}

AVAILABLE SOFTWARE LIBRARIES:
{self._format_resources_for_prompt(resources.get("libraries", []))}""")

        # Add know-how section if available
        if "know_how" in resources and resources["know_how"]:
            prompt_sections.append(f"""
AVAILABLE KNOW-HOW DOCUMENTS (Best Practices & Protocols):
{self._format_resources_for_prompt(resources.get("know_how", []))}""")

        # Build response format based on available categories
        response_format = """
For each category, respond with ONLY the indices of the relevant items in the following format:
TOOLS: [list of indices]
DATA_LAKE: [list of indices]
LIBRARIES: [list of indices]"""

        if "know_how" in resources and resources["know_how"]:
            response_format += "\nKNOW_HOW: [list of indices]"

        response_format += """

For example:
TOOLS: [0, 3, 5, 7, 9]
DATA_LAKE: [1, 2, 4]
LIBRARIES: [0, 2, 4, 5, 8]"""

        if "know_how" in resources and resources["know_how"]:
            response_format += "\nKNOW_HOW: [0, 1]"

        response_format += """

If a category has no relevant items, use an empty list, e.g., DATA_LAKE: []

IMPORTANT GUIDELINES:
1. Be generous but not excessive - aim to include all potentially relevant resources
2. ALWAYS prioritize database tools for general queries - include as many database tools as possible
3. Include all literature search tools
4. For wet lab sequence type of queries, ALWAYS include molecular biology tools
5. For data lake items, include datasets that could provide useful information
6. For libraries, include those that provide functions needed for analysis
7. For know-how documents, include those that provide relevant protocols, best practices, or troubleshooting guidance
8. Don't exclude resources just because they're not explicitly mentioned in the query
9. When in doubt about a database tool or molecular biology tool, include it rather than exclude it
"""

        prompt = "\n".join(prompt_sections) + response_format
        query_turn = f"USER QUERY: {query}"
        if query_last:
            prompt += f"\n{query_turn}\n"  # the plain-callable path below still sends one string

        # Use the provided LLM or create a new one
        if llm is None:
            from spatialomicsgym.llm import get_llm

            llm = get_llm()

        # Invoke the LLM
        try:
            if hasattr(llm, "invoke") and query_last:
                # The instructions and the ~46k-token resource list as a SYSTEM message, the query as
                # the user turn. Measured on both Azure Responses deployments this box uses: sent as
                # one user message the list is never cached; split like this, 46,208 of 46,381
                # tokens (azure-gpt-5.4) are served from cache on every call after the first.
                stable = prompt[: -len(query_turn) - 2]
                response = invoke_with_backoff(llm, [SystemMessage(content=stable), HumanMessage(content=query_turn)])
                _record_call(usage, "retriever", response)
                response_content = response.content
            elif hasattr(llm, "invoke"):
                # For LangChain-style LLMs
                response = invoke_with_backoff(llm, [HumanMessage(content=prompt)])
                _record_call(usage, "retriever", response)
                response_content = response.content
            else:
                # For other LLM interfaces
                response_content = str(llm(prompt))
                _record_call(usage, "retriever", None)
        except Exception as e:
            print(f"Warning: LLM retrieval failed ({e}), returning all resources")
            return _degrade_to_baseline(resources)

        # Parse the response to extract the selected indices
        selected_indices = self._parse_llm_response(response_content)

        # Resolve the indices into resources. Guard each index with 0 <= i < len: an integer index list
        # from the LLM may include a negative token (e.g. -1), which _parse_int_list correctly keeps as a
        # valid integer. Without the lower bound, i=-1 passes `i < len` and silently wraps to the LAST
        # resource (a wrong selection) — or raises KeyError on a resource key that is absent entirely.
        # dict.fromkeys dedups the selected indices while preserving first-occurrence order, so an LLM
        # reply that repeats an index (e.g. TOOLS: [3, 3, 5]) doesn't inject the same resource twice.
        selected_resources = {
            "tools": [
                resources["tools"][i]
                for i in dict.fromkeys(selected_indices.get("tools", []))
                if 0 <= i < len(resources.get("tools", []))
            ],
            "data_lake": [
                resources["data_lake"][i]
                for i in dict.fromkeys(selected_indices.get("data_lake", []))
                if 0 <= i < len(resources.get("data_lake", []))
            ],
            "libraries": [
                resources["libraries"][i]
                for i in dict.fromkeys(selected_indices.get("libraries", []))
                if 0 <= i < len(resources.get("libraries", []))
            ],
        }

        # Add know-how if present
        if "know_how" in resources and resources["know_how"]:
            selected_resources["know_how"] = [
                resources["know_how"][i]
                for i in dict.fromkeys(selected_indices.get("know_how", []))
                if 0 <= i < len(resources.get("know_how", []))
            ]

        # Fail-safe on a response that came back fine and still left the agent with nothing. The
        # exception path above returns ALL resources; a successful response that selected nothing
        # usable should degrade the SAME way rather than stranding the turn with no tools, no datasets
        # and no libraries — nothing downstream notices, the empty lists go straight into the system
        # prompt and "Tools: 0 selected" is printed on the way out.
        #
        # The question is asked about the RESOLVED resources, not the parsed indices, because there are
        # two ways to arrive at nothing and both strand the turn:
        #   * nothing parsed   — "TOOLS: [tool_0]" names resources instead of numbering them, so every
        #     token fails int() and the category comes back empty;
        #   * nothing resolved — "TOOLS: [900]" parses perfectly and is then dropped index-by-index by
        #     the 0 <= i < len bound above.
        # A guard placed before resolution saw only the first, and only when the index list had also
        # lost its brackets; the second never reached it, because the parsed lists were not empty.
        #
        # Both must stay distinguishable from a DELIBERATE empty selection, which survives untouched:
        # "TOOLS: []" and "TOOLS: none" are the model answering "nothing here is relevant". A header
        # with an empty bracket, or with no bracket at all, is that answer; a header with anything
        # inside its bracket, or a bare digit after its colon, is a selection we lost.
        #
        # `_offered_anything` keeps the degrade honest at the degenerate end: a turn given no candidate
        # resources cannot be stranded by us, and "return all resources" there would hand the caller
        # back its own dict — which, unlike the selection we build, need not carry all three keys.
        _offered_anything = any(resources.get(k) for k in ("tools", "data_lake", "libraries", "know_how"))
        _all_text = response_content if isinstance(response_content, str) else str(response_content)
        if _offered_anything and not any(selected_resources.values()):
            _text = _all_text
            has_header = re.search(r"(TOOLS|DATA_LAKE|LIBRARIES|KNOW[-_]HOW)\s*:", _text, re.IGNORECASE)
            # A digit right after a category header means the model DID try to select resources but in a
            # format we couldn't parse (e.g. no brackets: "TOOLS: 0, 3, 5"). _DECOR appears here in
            # exactly the places it appears in the extractor: a guard that asks a stricter question than
            # the parser it guards cannot see the parser's misses, which is how "**TOOLS:** [0, 3, 5]"
            # used to come back as a silent zero-resource selection.
            tried_but_unparsed = (
                re.search(
                    rf"(TOOLS|DATA_LAKE|LIBRARIES|KNOW[-_]HOW){_DECOR}:{_DECOR}\[?{_DECOR}\d", _text, re.IGNORECASE
                )
                or _wrote_a_nonempty_list(_text)
                # The shape both of the above miss: prose, or a newline, between the header and the
                # bracket. See :func:`_tried_to_select`.
                or any(_tried_to_select(_text, category) for category in _CATEGORY_KEYS)
            )
            if not has_header or tried_but_unparsed:
                print(
                    "Warning: LLM retrieval selected no usable resources (unparseable response, or "
                    "indices that resolve to nothing), returning all resources"
                )
                return _degrade_to_baseline(resources)

        # Per-category rescue, for the hole the all-or-nothing test above cannot reach.
        #
        # ``not any(selected_resources.values())`` requires EVERY category to be empty, so losing
        # only the tools -- while data_lake, libraries and know_how parse fine -- never triggers
        # the degrade. Measured with the real retriever: a reply whose TOOLS list sat on the line
        # after its header selected 0 tools, 1 dataset, 1 library and 1 know-how doc, printed no
        # warning, and the turn ran against a system prompt advertising no tools at all. The model
        # then improvises with whatever is still in the REPL namespace, or answers conceptually,
        # and ``Tools: 0 selected`` is the only trace.
        #
        # Restores one category, and only when the model demonstrably tried to fill it: a header,
        # a digit in brackets under that header, nothing resolved, and something to restore. An
        # explicit empty list still means "nothing here is relevant".
        for category, key in _CATEGORY_KEYS.items():
            if selected_resources.get(key) or not resources.get(key):
                continue
            if not _tried_to_select(_all_text, category):
                continue
            print(f"Warning: LLM retrieval named {key} but none resolved; restoring all {key} for this turn")
            selected_resources[key] = _degrade_to_baseline(resources).get(key, resources.get(key))

        return selected_resources

    def retrieve_packs(self, query: str, packs: list, llm=None, budget: int = 3, usage=None) -> list:
        """Tier 2: pick at most ``budget`` merged pack documents for a query -- a SEPARATE pass.

        Separate on purpose. ``prompt_based_retrieval`` above is the pass every scored run is
        measured under; its prompt, its candidates and ``_CATEGORY_KEYS`` (pinned to four) are not
        touched, which is what makes tier 1's selection identical with the packs on or off. This
        method runs afterwards, over the packs alone, with its own prompt and its own reply line.

        The failure direction is the opposite of tier 1's: there, losing the parse restores every
        tool so a turn is never stranded; here, ANY failure -- no LLM, an exception, no ``PACKS:``
        line, nothing parseable -- returns ``[]``. Tier 2 degrades to nothing, never to everything.
        """
        try:
            budget = int(budget)
        except (TypeError, ValueError):
            return []
        candidates = list(packs or [])
        if budget <= 0 or not candidates:
            return []

        prompt = f"""
You are an expert biomedical research assistant. The platform's own tools, datasets, libraries and
know-how playbooks have ALREADY been selected for this query in an earlier step; that selection is
final and is not your concern here. Your only job is to decide whether any of the EXTERNAL reference
documents below adds library or database knowledge that the task genuinely needs.

USER QUERY: {query}

EXTERNAL REFERENCE DOCUMENTS (tier 2):
{self._format_resources_for_prompt(candidates)}

Rules:
1. Choose AT MOST {budget}.
2. Prefer none over a weak match: a document earns its place only if the task will actually use the
   library or database it documents. Do not pick a document because it is merely related.
3. Pick a document ONLY if the task names a library, database or method that the platform's own tools
   do not provide; for a standard spatial-transcriptomics task (spatial domains, spatially variable
   genes, deconvolution, cell-cell communication, segmentation, batch integration, tool creation)
   answer PACKS: [].
4. Respond with ONLY the indices, in this exact format:
PACKS: [list of indices]

For example:
PACKS: [2]

If nothing qualifies:
PACKS: []
"""
        try:
            if llm is None:
                from spatialomicsgym.llm import get_llm

                llm = get_llm()
            if hasattr(llm, "invoke"):
                response = invoke_with_backoff(llm, [HumanMessage(content=prompt)])
                _record_call(usage, "retriever_packs", response)
                response_content = response.content
            else:
                response_content = str(llm(prompt))
                _record_call(usage, "retriever_packs", None)
        except Exception as e:
            print(f"Warning: tier-2 pack retrieval failed ({e}); no pack is added this turn")
            return []

        text = self._response_text(response_content)
        match = re.search(rf"PACKS{_DECOR}:{_DECOR}\[(.*?)\]", text, re.IGNORECASE | re.DOTALL)
        if not match:
            print("Warning: tier-2 pack retrieval returned no PACKS line; no pack is added this turn")
            return []
        picked: list = []
        for i in dict.fromkeys(_parse_int_list(match.group(1))):
            if 0 <= i < len(candidates):
                picked.append(candidates[i])
            if len(picked) >= budget:
                break
        return picked

    @staticmethod
    def _response_text(response) -> str:
        """A reply as one string: a plain string, or a Responses API-style list of content blocks."""
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            parts = []
            for item in response:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(p for p in parts if p)
        return str(response)

    def _format_resources_for_prompt(self, resources: list) -> str:
        """Format resources for inclusion in the prompt."""
        formatted = []
        for i, resource in enumerate(resources or []):  # a present-but-None category must not crash the prompt build
            if isinstance(resource, dict):
                # Handle dictionary format (from tool registry or data lake/libraries with descriptions)
                name = resource.get("name", f"Resource {i}")
                description = resource.get("description", "")
                formatted.append(f"{i}. {name}: {description}")
            elif isinstance(resource, str):
                # Handle string format (simple strings)
                formatted.append(f"{i}. {resource}")
            else:
                # Try to extract name and description from tool objects
                name = getattr(resource, "name", str(resource))
                desc = getattr(resource, "description", "")
                formatted.append(f"{i}. {name}: {desc}")

        return "\n".join(formatted) if formatted else "None available"

    def _parse_llm_response(self, response) -> dict:
        """Parse the LLM response to extract the selected indices.

        Accepts either a plain string or a Responses API-style list of content blocks.
        """
        # Normalize response to string if it's a list of content blocks (Responses API)
        if isinstance(response, list):
            parts = []
            for item in response:
                # LangChain Responses API returns list of dicts like {"type": "text", "text": "..."}
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        parts.append(str(item.get("text", "")))
                    # If it's a tool_call or other block, ignore for this simple parsing
                elif isinstance(item, str):
                    parts.append(item)
            response = "\n".join([p for p in parts if p])
        elif not isinstance(response, str):
            response = str(response)
        selected_indices = {"tools": [], "data_lake": [], "libraries": [], "know_how": []}

        # Extract indices for each category. re.DOTALL so a list that the model wrapped across lines
        # ("TOOLS: [0, 3,\n5, 7]") still matches — without it the `.` stopped at the newline, the regex
        # failed, and (headers being present) the fail-safe did not fire, stranding the agent with 0 tools.
        # _DECOR absorbs the markdown a chat model puts around its answer ("**TOOLS:** [0, 3]"), which
        # otherwise separated the header from its list and selected nothing at all.
        _flags = re.IGNORECASE | re.DOTALL
        tools_match = re.search(rf"TOOLS{_DECOR}:{_DECOR}\[(.*?)\]", response, _flags)
        if tools_match and tools_match.group(1).strip():
            selected_indices["tools"] = _parse_int_list(tools_match.group(1))

        data_lake_match = re.search(rf"DATA_LAKE{_DECOR}:{_DECOR}\[(.*?)\]", response, _flags)
        if data_lake_match and data_lake_match.group(1).strip():
            selected_indices["data_lake"] = _parse_int_list(data_lake_match.group(1))

        libraries_match = re.search(rf"LIBRARIES{_DECOR}:{_DECOR}\[(.*?)\]", response, _flags)
        if libraries_match and libraries_match.group(1).strip():
            selected_indices["libraries"] = _parse_int_list(libraries_match.group(1))

        # Extract know-how indices
        know_how_match = re.search(rf"KNOW[-_]HOW{_DECOR}:{_DECOR}\[(.*?)\]", response, _flags)
        if know_how_match and know_how_match.group(1).strip():
            selected_indices["know_how"] = _parse_int_list(know_how_match.group(1))

        return selected_indices
