import os
import re
import sys
from typing import TYPE_CHECKING, Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from spatialomicsgym.provider_names import (
    CANONICAL_SOURCES,
    OVERRIDABLE_ENV_SOURCES,
    canonical_source,
    source_from_model_prefix,
)

if TYPE_CHECKING:
    from spatialomicsgym.config import SpatialOmicsGymConfig

SourceType = Literal["OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom"]
ALLOWED_SOURCES: set[str] = set(SourceType.__args__)


class _ResponsesNoStopMixin:
    """Strip ``stop``/``temperature`` from the Responses-API request payload.

    gpt-5* models served via the OpenAI/Azure **Responses API** reject ``stop``
    outright and accept only the default ``temperature`` (a non-default value
    returns HTTP 400). This mixin drops both fields whenever the outgoing call
    will use the Responses API. It is shared by the OpenAI and Azure chat
    subclasses so the two code paths can't drift apart.

    Removing ``stop`` puts nothing in its place, so the provider writes on past the first
    ``</execute>`` -- an invented observation, more cells, sometimes an answer -- until it decides to
    end, and the loop throws all of it away (``cut_at_stop_sequence``). One E-06 call (cumulus r2)
    wrote 57,928 characters past its stop and took about 285 s; the trial's other calls took 5-40 s.
    ``generate()`` therefore STREAMS a reply from a client that says :attr:`reads_to_first_stop` and
    stops reading at the first stop sequence (``spatialomicsgym.responses_stream``).
    """

    #: ``generate()`` streams this client's reply and stops reading at its first stop sequence,
    #: because the provider will not take ``stop``. Read with ``is True``, so a stand-in that answers
    #: every attribute (a ``MagicMock``) is not streamed by accident.
    reads_to_first_stop = True

    def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[override]
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        try:
            if hasattr(self, "_use_responses_api") and self._use_responses_api(payload):
                payload.pop("stop", None)
                payload.pop("temperature", None)
        except Exception:
            # Be conservative: on any uncertainty still drop both fields.
            payload.pop("stop", None)
            payload.pop("temperature", None)
        return payload


class _ReasoningNoTempMixin:
    """Strip ``stop`` + ``temperature`` from EVERY Chat Completions request payload.

    OpenAI o-series (o1/o3/o4) — used directly OR via an Azure deployment — run on the Chat Completions
    API (NOT the Responses API, so :class:`_ResponsesNoStopMixin`'s use-responses gate never fires) yet
    still reject ``stop`` and any non-default ``temperature`` (HTTP 400: "only the default (1) value is
    supported"). This unconditionally drops both. Shared by the OpenAI and Azure reasoning subclasses so
    the two paths can't drift apart (mirrors how :class:`_ResponsesNoStopMixin` is shared)."""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[override]
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload.pop("stop", None)
        payload.pop("temperature", None)
        return payload


# Azure Responses API version handling ---------------------------------------------------
# The Responses API (required by gpt-5*) is only enabled for api-version 2025-03-01-preview
# or later; older previews 400 with "Azure OpenAI Responses API is enabled only for
# api-version 2025-03-01-preview and later". We floor the resolved version when Responses
# is active. Date-prefixed preview strings ("2025-04-01-preview") order correctly under `<`.
RESPONSES_MIN_API_VERSION = "2025-03-01-preview"


def _api_version_from_url(url: str) -> str | None:
    """Extract the ``api-version`` query parameter from an Azure endpoint URL, if present.

    Users commonly set a full endpoint like
    ``https://x.openai.azure.com/openai/responses?api-version=2025-04-01-preview``. The
    Azure branch strips the path to recover the bare resource host, which would otherwise
    silently discard the deliberately-chosen api-version — so recover it here."""
    try:
        from urllib.parse import parse_qs, urlsplit

        vals = parse_qs(urlsplit(url).query).get("api-version")
        return vals[0] if vals else None
    except Exception:
        return None


def _azure_resource_endpoint(raw: str) -> str:
    """The resource address `AzureChatOpenAI` wants, from whatever the operator set.

    ``OPENAI_ENDPOINT`` is commonly the full URL --
    ``https://<resource>.openai.azure.com/openai/responses?api-version=...`` -- and the client wants
    the address without the API path on it.

    This was written as ``raw.split("/openai", 1)[0]``, which reads the STRING and not the URL. The
    first ``/openai`` in ``https://openai-prod.openai.azure.com`` is the one formed by the second
    slash of ``https://`` and the resource name, so that endpoint became ``https:/`` and every
    request DNS-failed -- on Azure, which is this install's shipped default provider. ``openai-prod``,
    ``openai-eastus`` and plain ``openai`` are ordinary Microsoft resource names.

    So the cut is made in the PATH, never in the authority: split the URL first, drop the
    ``/openai...`` suffix from the path only, and put it back together. A reverse proxy's own prefix
    (``https://gw.example.com/azure/openai/...``) is part of the address and survives; a value that
    is not a URL at all is handed back unchanged apart from a trailing slash, because a wrong
    endpoint that is still the operator's is debuggable and ``https:/`` is not.
    """
    from urllib.parse import urlsplit

    raw = (raw or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw.rstrip("/")
    path = parts.path
    cut = path.find("/openai")
    if cut >= 0:
        path = path[:cut]
    path = path.rstrip("/")
    if not parts.scheme or not parts.netloc:
        # No authority to protect: `urlsplit` put the whole value in `path`, so the cut above has
        # already done the only safe thing and there is nothing to reassemble.
        return path or raw.rstrip("/")
    return f"{parts.scheme}://{parts.netloc}{path}"


def _resolve_azure_api_version(raw_endpoint: str, *, explicit: str | None, use_responses: bool) -> str:
    """Resolve the Azure ``api-version``.

    Precedence: an explicit ``OPENAI_API_VERSION`` > the version embedded in the endpoint
    URL's ``?api-version=`` query (the user set it deliberately, and the host-stripping in
    the Azure branch would otherwise discard it) > a conservative default. When the Responses
    API is active the result is floored to :data:`RESPONSES_MIN_API_VERSION`, so a user whose
    endpoint/env predates it (or omits it entirely) still reaches gpt-5 instead of a 400."""
    version = explicit or _api_version_from_url(raw_endpoint) or "2024-12-01-preview"
    if use_responses and version < RESPONSES_MIN_API_VERSION:
        version = RESPONSES_MIN_API_VERSION
    return version


# Anthropic newest-family gate -----------------------------------------------------------
# Anthropic's newest models share ONE behavioural boundary, verified live against the API. They
#   (a) deprecate the `temperature` sampling parameter — sending ANY value returns HTTP 400
#       "temperature is deprecated for this model.", AND
#   (b) reject assistant-message *prefill* — a conversation whose final message has role
#       "assistant" returns HTTP 400 "This model does not support assistant message prefill.
#       The conversation must end with a user message."
# The two boundaries are NOT identical (corrected 2026-09-30; the earlier note said they were, and
# claude-opus-4-6 / claude-sonnet-4-6 then 400'd on every ReAct turn after the first observation --
# hunt u16-llm-config-1). Per Anthropic's model-migration reference:
#   * prefill is rejected from 4.6 on: claude-opus-4-6, claude-sonnet-4-6, claude-opus-4-7/4-8, and
#     the Claude 5 family (opus/sonnet/fable 5, opus 5.5, fable 5.1, mythos 5.1);
#   * temperature is rejected from 4.7 on: claude-opus-4-7/4-8 and the Claude 5 family (mythos too).
#     claude-opus-4-6 and claude-sonnet-4-6 still accept temperature.
# claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5 and every claude-3-* accept both. Each
# behaviour layers its own deploy-box env hatch on top -- meet a newer/other model before this gate
# is updated without a code change -- and must target only its own family: omitting temperature or
# re-roling turns everywhere would silently alter models that currently work (and are benchmarked).
_CLAUDE_NEWEST_FAMILY_PREFIXES = ("claude-opus-4-7", "claude-opus-4-8")
# Anchor the Claude 5 generation on a `claude-<family>-5` head so dated ids
# (claude-sonnet-5-20260514) match too, without catching the older 4.x line —
# claude-sonnet-4-5 / claude-haiku-4-5 still accept both temperature and prefill.
_CLAUDE_NEWEST_FAMILY_RE = re.compile(r"^claude-(?:opus|sonnet|haiku|fable|mythos)-5(?:$|[.\-])")
# Prefill is rejected one generation earlier than temperature: the 4.6 pair too.
_CLAUDE_NO_PREFILL_EXTRA_PREFIXES = ("claude-opus-4-6", "claude-sonnet-4-6")


def _is_claude_newest_family(model: str) -> bool:
    """Whether ``model`` is a newest-generation Claude, which rejects ``temperature`` (and prefill)."""
    m = (model or "").strip().lower()
    return m.startswith(_CLAUDE_NEWEST_FAMILY_PREFIXES) or bool(_CLAUDE_NEWEST_FAMILY_RE.match(m))


def _is_claude_no_prefill_family(model: str) -> bool:
    """Whether ``model`` rejects an assistant-terminated conversation: the newest family plus the 4.6 pair."""
    m = (model or "").strip().lower()
    return _is_claude_newest_family(m) or m.startswith(_CLAUDE_NO_PREFILL_EXTRA_PREFIXES)


# OpenAI generations from gpt-5 onward REQUIRE the Responses API and reject both ``stop`` and any
# non-default ``temperature``. gpt-5 was the first; gpt-6 does the same, measured on this box
# against the Azure deployment ``gpt-6-astra`` on 2026-09-21:
#
#     temperature=0.7 -> 400 "Unsupported parameter: 'temperature' is not supported with this model."
#     stop=["x"]      -> 400 "Unknown parameter: 'stop'. Did you mean 'store'?"
#     temperature=1   -> 200          (plain call -> 200)
#
# A literal ``startswith("gpt-5")`` gate misses a gpt-6 deployment TWICE, and both misses are
# fatal rather than cosmetic: the client would pick Chat Completions instead of the Responses API,
# and :class:`SpatialOmicsGymConfig`'s 0.7 temperature would reach the wire and 400 on EVERY turn.
# The failure is silent until the first real call, which is exactly the shape
# ``_wants_completion_tokens`` exists to stop the wizard signing off green.
#
# Anchored on the generation digit so ``gpt-5``, ``gpt-5.5``, ``gpt-6-astra`` and a future
# ``gpt-7`` all match, while ``gpt-4``/``gpt-4o``/``gpt-35-turbo`` -- which still accept both --
# do not. Deliberately single-digit: ``gpt-35-turbo`` is Azure's name for GPT-3.5, and a
# two-or-more-digit alternative would swallow it. A ``gpt-10`` needs one character here, and this
# comment is where whoever meets it should look.
_OPENAI_RESPONSES_FAMILY_RE = re.compile(r"^gpt-[5-9](?:$|[.\-])")


def _is_openai_responses_family(model: str) -> bool:
    """Whether ``model`` needs the Responses API and rejects ``stop`` / non-default ``temperature``.

    Takes the BARE deployment name: the Azure branch strips its ``azure-`` prefix before asking.
    """
    return bool(_OPENAI_RESPONSES_FAMILY_RE.match((model or "").strip().lower()))


# OpenAI o-series reasoning models (o1/o3/o4-*) run on the Chat Completions API but — like gpt-5* on the
# Responses API — reject ``stop`` and any non-default ``temperature`` (HTTP 400: "only the default (1)
# value is supported for 'temperature'"). gpt-5* is handled separately (it forces the Responses API);
# this matches ONLY the o-series so the Chat-Completions branch can strip both fields. Mirrors the
# o-series arm of setup-side ``llm_chat._REASONING_MODEL_RE``: a separator-anchored ``o1``/``o3``/``o4``
# so a real reasoning id (``o3-mini``, ``my-o4-deploy``) matches while ``gpt-4o``/``amigo1`` do NOT.
_OPENAI_REASONING_RE = re.compile(r"(?i)(?:^|[-_/])o[134](?:$|[-_/])")


def _is_openai_reasoning_model(model: str) -> bool:
    """Whether ``model`` is an OpenAI o-series reasoning model (rejects ``stop`` + non-default temperature)."""
    return bool(_OPENAI_REASONING_RE.search(model or ""))


def _env_flag(name: str) -> bool:
    """Whether env var ``name`` holds a truthy string (``true``/``1``/``yes``/``on``).

    The documented ``SOG_*`` boolean set — ``config._env_bool``: strip, lower, true/1/yes/on.
    Kept inline rather than importing that parser so this module's flags stay readable next to
    the model rules they gate, but the *spelling* must not diverge: ``on`` was missing here, so
    ``SOG_ANTHROPIC_NO_TEMPERATURE=on`` (a spelling the setup wizard and the web settings panel
    both accept) read as off and the deploy-box escape hatch for a model whose temperature or
    prefill rule changed under us silently never opened.
    """
    return os.getenv(name, "").strip().lower() in ("true", "1", "yes", "on")


def accepts_an_explicit_temperature(model: str) -> bool:
    """Whether a caller may PIN ``temperature`` on ``model`` without the provider refusing the call.

    Three families reject any non-default value with an HTTP 400 -- the newest Claude generation,
    the OpenAI Responses family (gpt-5*, gpt-6*) and the o-series reasoning models -- and each is
    already recognised above for the factory's own use. This is the same question asked from the
    outside, so a caller that wants determinism (the tool retriever under ``benchmarking_enabled``,
    which otherwise picks the scored prompt's contents at temperature 0.7 with no seed) can ask for
    it where it is available and leave the model alone where it is not.

    Conservative by construction: it takes the model id alone, so if ANY of the three patterns
    matches, the answer is no. A false no costs determinism on one auxiliary call; a false yes
    costs the turn.
    """
    name = (model or "").strip()
    if not name:
        return False
    bare = name[len("azure-") :] if name.lower().startswith("azure-") else name
    # A Bedrock id carries its provider and region in front (``us.anthropic.claude-opus-4-8-v1:0``),
    # and the deploy-box hatch the factory honours is honoured here too: this said yes for both while
    # get_llm had dropped the temperature, and the bound temperature=0 400'd every scored retrieval
    # (u16-llm-config-10).
    if "anthropic." in bare:
        bare = bare.split("anthropic.")[-1]
    if bare.lower().startswith("claude") and _anthropic_omit_temperature(bare):
        return False
    return not (_is_claude_newest_family(bare) or _is_openai_responses_family(bare) or _is_openai_reasoning_model(bare))


def _anthropic_omit_temperature(model: str) -> bool:
    """Whether ``temperature`` must be omitted when building ``ChatAnthropic`` for ``model``.

    Newest-family models 400 on any temperature; older models keep their configured value.
    ChatAnthropic strips any None-valued field from its request payload, so passing
    ``temperature=None`` is a clean omit. ``SOG_ANTHROPIC_FORCE_TEMPERATURE`` /
    ``SOG_ANTHROPIC_NO_TEMPERATURE`` are deploy-box escape hatches (force-keep wins over
    force-drop) for reaching a model before this gate is updated.
    """
    if _env_flag("SOG_ANTHROPIC_FORCE_TEMPERATURE"):
        return False
    if _env_flag("SOG_ANTHROPIC_NO_TEMPERATURE"):
        return True
    return _is_claude_newest_family(model)


def _bedrock_temperature(model: str, temperature: float | None) -> float | None:
    """``temperature`` for ``ChatBedrock``, gated exactly like the direct-Anthropic path.

    Anthropic models are served on Bedrock too (auto-detected from the ``anthropic.`` /
    ``us.anthropic.`` id prefix) and enforce the SAME breaking change: the newest family (opus-4-7/-4-8,
    Claude 5) 400s on any ``temperature``. Strip the Bedrock provider/region prefix to the bare
    ``claude-…`` id and reuse ``_anthropic_omit_temperature`` → ``None`` (omit) for the newest family,
    the configured value otherwise. Anthropic Bedrock ids honor the same ``SOG_ANTHROPIC_*_TEMPERATURE``
    hatches as the direct path. A non-Anthropic Bedrock model (``amazon.titan-…``, ``meta.llama*``)
    ALWAYS keeps its configured value — those hatches are Anthropic-scoped and never strip a
    non-Anthropic provider's temperature. (Assistant-prefill on the newest family is
    also rejected on Bedrock, but ``ChatBedrock`` lacks ChatAnthropic's ``_get_request_payload``
    chokepoint, so that re-role is not wired here — use the direct Anthropic path, or
    ``SOG_ANTHROPIC_NO_PREFILL``, for a newest-family model on Bedrock.)"""
    bare = model.split("anthropic.")[-1] if model else model
    # The SOG_ANTHROPIC_* hatches (and the newest-family gate) are Anthropic-scoped by name; only apply
    # them to an Anthropic Bedrock id (an ``anthropic.``/``us.anthropic.``/``eu.anthropic.`` prefix, or a
    # bare ``claude-*`` id). A non-Anthropic provider (amazon.titan-*, meta.llama*, mistral.*, cohere.*,
    # ai21.*) has no newest-Claude temperature-400 problem, so it must keep its configured value even
    # when a deploy box sets SOG_ANTHROPIC_NO_TEMPERATURE for its Claude models — otherwise that
    # Anthropic-scoped knob silently strips Titan/Llama temperature too (contradicting the contract above).
    is_anthropic_bedrock = bool(model) and ("anthropic." in model or bare.startswith("claude-"))
    if not is_anthropic_bedrock:
        return temperature
    return None if _anthropic_omit_temperature(bare) else temperature


# Anthropic assistant-message prefill rejection ------------------------------------------
# The STCoscientist ReAct loop appends each tool observation to the transcript as an *assistant*
# message (spatialomicsgym/agent/execution.py), so the conversation handed to the model on the
# next turn ends with an assistant turn — a "prefill". Newest-family models reject that outright
# (HTTP 400, see the gate above). The fix re-roles the environment-observation assistant turns to
# "user" -- every one of them, not only the trailing run that makes the prefill -- so the
# conversation is user-terminated and a turn's role never changes between steps (each request is
# then a prefix of the next; see _rerole_observation_turns). This is semantically correct: an
# observation is tool/environment feedback — the analogue of a native ``tool_result``, which
# Anthropic carries as a user-role block — and the model keys off the ``<observation>`` tags, not
# the role, so behaviour is preserved. The model's own ``<execute>``/``<solution>`` turns stay
# "assistant". Env hatches mirror the temperature ones.
_OBSERVATION_TAG_RE = re.compile(r"<\s*observation\s*>", re.IGNORECASE)


def _anthropic_no_prefill(model: str) -> bool:
    """Whether the outgoing Anthropic payload must be made user-terminated for ``model``.

    ``SOG_ANTHROPIC_FORCE_PREFILL`` (allow prefill / no re-role) / ``SOG_ANTHROPIC_NO_PREFILL``
    (force the re-role) are deploy-box escape hatches; force-allow wins over force-fix.
    """
    if _env_flag("SOG_ANTHROPIC_FORCE_PREFILL"):
        return False
    if _env_flag("SOG_ANTHROPIC_NO_PREFILL"):
        return True
    return _is_claude_no_prefill_family(model)


def _payload_message_is_observation(message: dict) -> bool:
    """Whether an Anthropic-format payload message carries an environment ``<observation>`` tag.

    Content may be a plain string or a list of content blocks (dicts with a ``text`` field); the
    tag match is case-insensitive, matching what the ReAct loop emits.
    """
    content = message.get("content")
    if isinstance(content, str):
        return bool(_OBSERVATION_TAG_RE.search(content))
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and _OBSERVATION_TAG_RE.search(text):
                    return True
    return False


def _payload_message_is_only_observation(message: dict) -> bool:
    """Whether a payload message is WHOLLY an environment observation -- what ``execute()`` appends
    (``<observation>...</observation>``), as opposed to a model turn that mentions the tag."""
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict) and isinstance(b.get("text"), str))
    if not isinstance(content, str):
        return False
    text = content.strip()
    return bool(re.match(r"<\s*observation\s*>", text, re.IGNORECASE)) and bool(
        re.search(r"</\s*observation\s*>\Z", text, re.IGNORECASE)
    )


def _rerole_observation_turns(messages: list) -> list:
    """Re-role observation *assistant* turns to ``user`` so the payload is user-terminated AND stable.

    Every turn that is wholly an observation is re-roled, not only the trailing run. Re-roling only
    the tail flipped an observation's role between steps -- ``user`` while it was last, ``assistant``
    again once the model answered it -- so the request at step N was never a prefix of step N+1's,
    and the model reread its own history with a role changed under it. Re-roling them all keeps each
    turn's role fixed for the life of the turn and the conversation strictly alternating, which is
    also the shape a native ``tool_result`` has.

    That prefix is not a cache hit, and on 2026-09-25 it saved nothing on Claude. Anthropic caches
    only up to a ``cache_control`` breakpoint -- top-level, or on a content block -- and a request
    carrying neither is not cached at all. None of these requests carried one when checked that day:
    ``get_llm`` sets none, langchain_anthropic 0.3.22 adds one only when a caller passes it -- as an
    ``invoke`` keyword or on a content block -- and no caller did.
    ``test_no_request_carries_a_cache_breakpoint`` pins only the first of those; a caller passing
    ``cache_control=`` to ``invoke`` would not red it. A stable prefix is what a breakpoint would
    need in order to hit, if one is added. This docstring first said the change stopped a cache
    miss, and the commit that made it (68efa6d) that the old payloads "cost more": both described a
    cache that was never on.

    The trailing run still uses the looser "carries an ``<observation>`` tag" test it always has,
    and the model's own ``<execute>``/``<solution>`` turns stay ``assistant``. If the list is still
    assistant-terminated afterwards (a prefill the ReAct loop does not produce), a minimal user turn
    is appended as a safety net so a prefill-rejecting model can never 400. Mutates in place and
    returns ``messages``.
    """
    for k, message in enumerate(messages):
        if (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and _payload_message_is_only_observation(message)
        ):
            messages[k] = {**message, "role": "user"}
    i = len(messages) - 1
    while (
        i >= 0
        and isinstance(messages[i], dict)
        and messages[i].get("role") == "assistant"
        and _payload_message_is_observation(messages[i])
    ):
        messages[i] = {**messages[i], "role": "user"}
        i -= 1
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
        messages.append({"role": "user", "content": "Please continue."})
    return messages


class _AnthropicNoPrefillMixin:
    """Re-role ``<observation>`` assistant turns → user in the Anthropic request payload.

    Newest-family Claude models reject a conversation that ends with an assistant message; the
    STCoscientist ReAct loop produces exactly that (the tool observation is appended as an
    assistant turn). This mixin fixes the single Anthropic payload chokepoint — all four
    generation methods (``_generate``/``_stream``/``_agenerate``/``_astream``) route through
    ``_get_request_payload``, exactly as ``_ResponsesNoStopMixin`` relies on — so every code path
    is covered by one override. Attached only when :func:`_anthropic_no_prefill` is True.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[override]
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if isinstance(messages, list) and messages:
            _rerole_observation_turns(messages)
        return payload


def _load_anthropic_key_from_profile() -> bool:
    """Best-effort: populate ``ANTHROPIC_API_KEY`` from ``~/.bash_profile`` when it's absent from
    the process env (some deploy boxes keep the key only in the login profile).

    Robustness matters here because the naive form silently corrupts the key:

    * The profile's **stdout** is redirected to ``/dev/null`` *before* the value is emitted. A
      profile that prints a banner/MOTD during ``source`` would otherwise be captured ahead of the
      key, yielding a multi-line ``"Welcome to devbox\\nsk-ant-..."`` value → every request 401s
      with no obvious cause.
    * The value is emitted with ``printf %s`` after a ``;`` (not ``&&``): a profile whose last
      command exits non-zero must not prevent the key from being read, and ``${VAR:-}`` keeps a
      ``set -u`` profile from erroring.
    * Human-readable notices go to **stderr**, never stdout, so they can't pollute a
      ``--json`` / machine-readable consumer of the CLI.

    The env is still mutated (preserving key inheritance to MCP child processes). Returns True iff
    the key was newly loaded.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import subprocess

        result = subprocess.run(
            ["bash", "-c", 'source ~/.bash_profile >/dev/null 2>&1; printf %s "${ANTHROPIC_API_KEY:-}"'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        key = result.stdout.strip()
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
            print("✓ Loaded ANTHROPIC_API_KEY from ~/.bash_profile", file=sys.stderr)
            return True
    except Exception as e:
        print(f"Note: Could not load ANTHROPIC_API_KEY from bash_profile: {e}", file=sys.stderr)
    return False


# Content-block types that carry model-authored prose. Everything else in a Responses-API content
# list (``reasoning`` summaries, ``tool_use`` calls, ``refusal`` objects) is structure, not answer
# text, and must not be concatenated into it.
_TEXT_BLOCK_TYPES = frozenset({"text", "output_text", "redacted_text"})


def content_to_text(content) -> str:
    """Flatten an ``AIMessage.content`` into plain text.

    The OpenAI/Azure **Responses API** (``OPENAI_USE_RESPONSES_API=1``, required by the gpt-5
    family) returns a list of content blocks rather than a string::

        [{"type": "text", "text": "ping", "annotations": []}]

    Every consumer that did ``content.strip()`` / ``json.loads(content)`` / ``"".join(contents)``
    therefore raised ``AttributeError``/``TypeError`` — or, worse, silently produced garbage — the
    moment the agent was pointed at an Azure gpt-5 deployment. This is the single normalizer all of
    them share, so those implementations cannot drift apart again.

    Always returns a ``str``: ``None`` and an empty list become ``""``, and an unexpected type is
    stringified rather than raised, because every caller is on a response path where a hard failure
    would discard work the model already paid for.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        join = ItemJoin()
        for block in content:
            try:
                if isinstance(block, str):
                    # Some providers emit a bare string alongside dict blocks; dropping it would
                    # silently lose part of the answer.
                    parts.append(join.piece(block))
                elif isinstance(block, dict):
                    btype = block.get("type")
                    # A block with no "type" but a "text" is still text (older/partial payloads).
                    if btype in _TEXT_BLOCK_TYPES or (btype is None and "text" in block):
                        part = block.get("text") or block.get("content") or ""
                        if isinstance(part, str):
                            parts.append(join.piece(part, block.get("id")))
            except Exception:
                # Be conservative; skip malformed blocks rather than losing the whole message.
                continue
        return "".join(parts)
    return str(content)


class ItemJoin:
    """How the text of one reply is put together from its parts: the rule :func:`content_to_text`
    and the streamed reader (``spatialomicsgym.responses_stream``) share, so a reply read as a
    stream and the same reply read whole, with its item ids, come out byte-identical.

    Two OUTPUT ITEMS are two messages: the Responses API can return a plan item and then an action
    item, and joining them with "" fused the first's last sentence into the second's first
    ("...tags.Most recent observation"), 15 turns over the E-01/E-02 trials. So a part of a
    different item than the text before it starts a new paragraph, and parts of ONE item run
    together exactly as written. A part that names no item joins as it always did -- nothing says
    where an item ends.

    Where the item ids come from matters, and is not everywhere: langchain_openai's ``v0`` output
    (what ``get_llm`` builds the Responses clients with) strips ``id`` from every text block of a
    reply read WHOLE, so a list from ``.invoke`` carries no item to split on and joins with "" --
    measured on the library's own converter. Its STREAM names each message item as it opens it,
    which is the path ``generate()`` reads a Responses reply by.

    An empty part is skipped outright -- no break, no item -- because a stream sends no text for an
    empty item at all, and the two readings must agree.
    """

    __slots__ = ("_item", "_started")

    def __init__(self) -> None:
        self._started = False
        self._item: object = None

    def piece(self, part: str, item: object = None) -> str:
        """What ``part`` of ``item`` adds to the reply: the part, after a paragraph break when it
        begins an item other than the one before it."""
        if not part:
            return ""
        brk = self._started and item is not None and self._item is not None and item != self._item
        self._started = True
        if item is not None:
            self._item = item
        return "\n\n" + part if brk else part


def message_to_text(message) -> str:
    """``content_to_text`` applied to a message object (``None``-safe)."""
    return content_to_text(getattr(message, "content", None))


# Providers whose env default a foreign model-name prefix may override — the rule, and why those
# names and not the others, are documented on the definition in ``provider_names``. It lives there
# so the setup health probe can weigh the same names this resolver does: that probe cannot import
# THIS module, because langchain_core (imported above) is one of the packages it exists to check
# for. Re-exported under the private name this module's callers and tests already reach for.
_OVERRIDABLE_ENV_SOURCES: frozenset[str] = OVERRIDABLE_ENV_SOURCES

_BEDROCK_MODEL_PREFIXES = (
    "anthropic.claude-",
    "amazon.titan-",
    "meta.llama",
    "mistral.",
    "cohere.",
    "ai21.",
    "us.",
    "eu.",
    "apac.",
)

_OLLAMA_NAME_HINTS = (
    "llama",
    "mistral",
    "qwen",
    "gemma",
    "phi",
    "dolphin",
    "orca",
    "vicuna",
    "deepseek",
)


# The prefix table lives in ``provider_names`` (stdlib only) so ``sog_install.base_env`` can weigh a
# model name exactly as this resolver does without importing langchain_core -- see the note on
# ``_OVERRIDABLE_ENV_SOURCES`` above. Kept under the private name this module and its tests use.
_source_from_model_prefix = source_from_model_prefix


def _env_source() -> str | None:
    """The provider the *environment* defaults to, under either name this project ships.

    ``SOG_SOURCE`` and ``LLM_SOURCE`` are aliases: ``.env.example`` writes ``SOG_SOURCE`` under the
    heading "SOG_SOURCE selects the provider", ``sog-setup`` writes both (sog_install/credentials.py:64),
    and the CLI has always read ``SOG_SOURCE or LLM_SOURCE``. This module previously read only
    ``LLM_SOURCE``, so every guard built around the env lane -- above all "a model name that proves
    its provider outranks a stale default" -- was blind to the variable users actually set.

    ``SOG_SOURCE`` first, matching ``chat_cli._detect_source``, so the two agree when both are set.
    """
    return os.getenv("SOG_SOURCE") or os.getenv("LLM_SOURCE")


def _is_azure_endpoint(endpoint: str | None) -> bool:
    """True when ``OPENAI_ENDPOINT`` names an Azure OpenAI resource.

    ``OPENAI_ENDPOINT`` is Azure-only throughout this project — ``.env.example``, ``REPRODUCE.md``
    and the setup wizard's ``azure`` provider all write a ``*.openai.azure.com`` URL into it — so
    the host is a reliable signal. Matching on the host (not a substring) keeps a look-alike
    domain from qualifying.
    """
    if not endpoint or not isinstance(endpoint, str):
        return False
    from urllib.parse import urlsplit

    host = (urlsplit(endpoint if "//" in endpoint else f"//{endpoint}").hostname or "").lower()
    # Azure AI Foundry resources hand out cognitiveservices / services.ai hosts for the same service
    # (hunt 2026-09-30, u16-llm-config-2): only matching *.openai.azure.com left their keys unguarded.
    return host.endswith((".openai.azure.com", ".cognitiveservices.azure.com", ".services.ai.azure.com"))


def _guard_azure_credentials_against_openai(model: str | None, resolved_source: str) -> None:
    """Refuse to post an Azure key to ``api.openai.com``.

    A bare Azure *deployment* id is usually named after the model it serves (``gpt-5``), and that
    ``gpt-`` prefix resolves to OpenAI — deliberately, so a stale env provider cannot hijack it.
    But the OpenAI branch never reads ``OPENAI_ENDPOINT``, so the Azure resource is dropped and
    ``ChatOpenAI`` sends the Azure key to ``api.openai.com``. The credential reaches a vendor the
    user never configured, and the 401 that comes back points at platform.openai.com — the one
    place the key does not live.

    Deliberately narrow, so nothing that works today changes: it fires only when an Azure endpoint
    is configured AND the key is not an ``sk-`` OpenAI key. A real OpenAI key beside a leftover
    Azure endpoint keeps routing to OpenAI, and ``source="OpenAI"`` / ``SOG_SOURCE=OpenAI`` remain
    explicit escapes (the caller-supplied ``source`` never reaches here at all). The env escape is
    read case-insensitively, like :func:`resolve_source` reads it: spelled ``openai`` it did not
    open, so the guard refused a call the user had explicitly authorised.

    The remedies it prints name ``SOG_SOURCE``, not the older ``LLM_SOURCE`` this message used to
    name. ``_env_source`` reads ``SOG_SOURCE`` first, and a ``SOG_SOURCE`` line is what both
    ``.env.example`` and the setup wizard write — so for the default install the old advice was
    outranked by a variable the message never mentioned, and following it reprinted this error
    unchanged. ``LLM_SOURCE`` still works for anyone who has only that one; it is just no longer
    the instruction, because it is not the one that always wins.
    """
    problem = azure_key_to_openai_problem(model, resolved_source)
    if problem:
        raise ValueError(problem)


def azure_key_to_openai_problem(model: str | None, resolved_source: str | None) -> str | None:
    """The refusal :func:`_guard_azure_credentials_against_openai` raises, as text, or ``None``.

    The front doors resolve the provider themselves and hand ``get_llm`` the answer as an explicit
    ``source=`` -- the lane the guard deliberately leaves open for a caller who typed it. So a
    *derived* ``OpenAI`` from ``stcoscientist -m gpt-5`` / the portal's model switch skipped the
    guard, and the switch's reachability ping sent the Azure key to api.openai.com (hunt
    2026-09-30, uL6-parity-1). Their key pre-flight asks this instead, before any client exists.
    """
    if resolved_source != "OpenAI" or canonical_source(_env_source()) == "OpenAI":
        return None
    if not _is_azure_endpoint(os.getenv("OPENAI_ENDPOINT")):
        return None
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key or key.startswith("sk-"):
        return None
    name = model or "gpt-5"
    return (
        f"OPENAI_ENDPOINT points at an Azure OpenAI resource, but the model name {name!r} routes to "
        "public OpenAI, which would send your Azure key to api.openai.com. Pick one:\n"
        f"  - name the deployment as Azure:  llm='azure-{name}'  (or SOG_LLM=azure-{name})\n"
        "  - or set SOG_SOURCE=AzureOpenAI in your .env  (SOG_SOURCE is what sog-setup and\n"
        "    .env.example write, and it overrides the older LLM_SOURCE alias)\n"
        "If you really do mean public OpenAI, set SOG_SOURCE=OpenAI and use an sk-... key."
    )


def resolve_source(model: str | None, source: str | None = None, base_url: str | None = None) -> str:
    """Decide which provider to build a client for.

    Precedence: explicit ``source=`` > an unambiguous provider prefix in the model name >
    ``SOG_SOURCE``/``LLM_SOURCE`` > ``base_url`` > name-based fallback.

    The model prefix outranks the env default because the env var is process-wide (this repo's
    ``.env`` ships one, and so does every box ``sog-setup`` touches) while the model name is the
    caller's per-call choice. Consulting the env first meant ``STCoscientist(llm="azure-gpt-5.4-mini")``
    silently built a ChatAnthropic client and 400'd. Only a *provably wrong* env value is overridden
    — see :data:`_OVERRIDABLE_ENV_SOURCES`. Both env aliases are read (:func:`_env_source`); reading
    only ``LLM_SOURCE`` left the variable ``.env.example`` actually documents outside every guard.
    Either alias is matched ignoring case and surrounding whitespace, the way the CLI and the setup
    probe already match it — a spelling this project accepts must not name a different vendor here.
    """
    if source is not None:
        # Spelled any way the env lanes accept: ``source="custom"`` was returned verbatim and then
        # refused as invalid, while ``SOG_SOURCE=custom`` worked (hunt 2026-09-30, uL4-honesty-14).
        return canonical_source(source) or source

    prefix_source = _source_from_model_prefix(model)
    # Read the env value the way every other reader of it does -- ignoring case and surrounding
    # whitespace. This one compared by exact case, and dropping the value is not "fall back to the
    # same answer": it hands the decision to the model name, which then wins unopposed. So a
    # lowercase ``SOG_SOURCE=ollama`` beside a ``gpt-`` model name did not fail, it quietly built an
    # OpenAI client -- a local-only deployment posting its prompts to a public API. Same shape for
    # ``bedrock``/``azureopenai`` beside a ``claude-`` name: out of the user's account, and if
    # ANTHROPIC_API_KEY happens to be set (``.env.example`` ships the line) the call even succeeds.
    # A value that names no provider at all is still dropped -- ``canonical_source`` returns None.
    env_source = canonical_source(_env_source())

    if prefix_source is not None and (env_source is None or env_source in _OVERRIDABLE_ENV_SOURCES):
        # The name names a specific provider. Honor it unless the env var picked a free-form
        # provider whose deployment could legitimately carry that same name.
        return prefix_source

    if env_source is not None:
        return env_source
    if prefix_source is not None:
        return prefix_source

    model = model or ""
    if model.startswith("gpt-oss"):
        return "Ollama"
    if "groq" in model.lower():
        return "Groq"
    if base_url is not None:
        return "Custom"
    if model.startswith(_BEDROCK_MODEL_PREFIXES):
        # Bedrock provider-ids are detected BEFORE the Ollama substring sniff below. A real
        # Bedrock id — ``meta.llama3-70b-instruct-v1:0`` / ``mistral.mistral-large-2402-v1:0`` —
        # CONTAINS the substring ``llama``/``mistral``, so an Ollama-first order captured it
        # first and routed every source-less Bedrock Llama/Mistral call to ``localhost:11434``
        # (a confusing connection-refused instead of AWS). A local Ollama ``llama3``/``mistral``
        # has no Bedrock provider prefix, so it still falls through to Ollama. ``meta.llama``
        # (not a dead ``meta.llama-``) matches the real ``meta.llama2``/``meta.llama3`` ids.
        # ``us.``/``eu.``/``apac.`` are AWS cross-region inference-profile prefixes
        # (e.g. ``eu.anthropic.claude-3-5-sonnet-...``); all three regions must be listed or a
        # source-less EU/APAC profile id raises "Unable to determine model source".
        return "Bedrock"
    if "/" in model or any(name in model.lower() for name in _OLLAMA_NAME_HINTS):
        return "Ollama"
    raise ValueError("Unable to determine model source. Please specify 'source' parameter.")


def _config_source_override(config: Optional["SpatialOmicsGymConfig"]) -> str | None:
    """A provider the caller genuinely set on ``config``, or ``None`` when it merely echoes the env.

    ``config.source`` is usually nothing but ``SOG_SOURCE`` copied out of .env (config.py:206).
    Forwarding it as an explicit source would hand an *environment default* to
    :func:`resolve_source` through the *explicit* lane, where it is returned verbatim and the
    model name is never even consulted. A stock .env.example install would then build a
    ChatAnthropic client for ``llm="gpt-4o"`` — the very failure the resolver's env handling
    exists to prevent. When it merely echoes the env we leave it to :func:`resolve_source`, which
    weighs it against the model name; a source the caller genuinely set on the config object
    differs from the env value and is still honoured verbatim.
    """
    if config is None:
        return None
    return config.source if config.source != _env_source() else None


def effective_source(
    model: str | None,
    source: str | None = None,
    base_url: str | None = None,
    config: Optional["SpatialOmicsGymConfig"] = None,
) -> str:
    """Which provider :func:`get_llm` will build a client for, decided without building one.

    A caller that has to *name* the provider before the client exists — the agent's startup
    banner does, since it prints before the first request — has only one honest answer: the one
    ``get_llm`` is about to reach. Asking here rather than re-deriving it is what stops a banner
    from naming a vendor the run never posts to.

    Deliberately does not run :func:`_guard_azure_credentials_against_openai`: that guard exists
    to refuse a request, and refusing has to happen where the request is made, not where a label
    is printed.
    """
    if source is None:
        source = _config_source_override(config)
    return resolve_source(model, source=source, base_url=base_url)


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
    source: SourceType | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    config: Optional["SpatialOmicsGymConfig"] = None,
    *,
    request_timeout: float | None = None,
    max_retries: int | None = None,
) -> BaseChatModel:
    """
    Get a language model instance based on the specified model name and source.
    This function supports models from OpenAI, Azure OpenAI, Anthropic, Ollama, Gemini, Bedrock, and custom model serving.
    Args:
        model (str): The model name to use
        temperature (float): Temperature setting for generation
        stop_sequences (list): Sequences that will stop generation
        source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", or "Custom"
                      If None, will attempt to auto-detect from model name
        base_url (str): The base URL for custom model serving (e.g., "http://localhost:8000/v1"), default is None
        api_key (str): The API key for the custom llm
        config (SpatialOmicsGymConfig): Optional configuration object. If provided, unspecified parameters will use config values
        request_timeout, max_retries: per-call overrides of the configured ``SOG_LLM_REQUEST_TIMEOUT`` /
            ``SOG_LLM_MAX_RETRIES`` -- for a short probe, such as the portal's reachability ping
    """
    # Use config values for any unspecified parameters
    if config is not None:
        if model is None:
            model = config.llm
        if temperature is None:
            temperature = config.temperature
        if source is None:
            # Only a source the caller genuinely set on the config object — never ``SOG_SOURCE``
            # echoed back at us. See ``_config_source_override`` for why that distinction is the
            # whole ballgame; the ``resolve_source`` call below is guarded by ``if source is None``
            # and returns an explicit source verbatim, model name unread.
            source = _config_source_override(config)
        if base_url is None:
            base_url = config.base_url
        if api_key is None:
            api_key = config.api_key or "EMPTY"

    # Per-request timeout + retries — bound a stalled provider so a single ReAct step can't hang the
    # whole run forever (a hang is unrecoverable; the stream driver only catches exceptions). getattr
    # keeps this working if an older config lacks the fields; `timeout` + `max_retries` are the accepted
    # (aliased) kwargs for ChatOpenAI/AzureChatOpenAI/ChatAnthropic, and models that don't recognize
    # them ignore them (extra="ignore").
    # Without a config, the operator's knobs still apply: they are read from default_config, which
    # is where SOG_LLM_REQUEST_TIMEOUT / SOG_LLM_MAX_RETRIES land. A config-less call used a hard
    # 600 s x 3, so the portal's model-switch ping could hold a request ~30 minutes against an
    # endpoint that accepts and never answers (u16-llm-config-15).
    knobs = config
    if knobs is None:
        try:
            from spatialomicsgym.config import default_config as knobs
        except Exception:
            knobs = None
    _llm_timeout = getattr(knobs, "llm_request_timeout_seconds", None) if knobs is not None else None
    if request_timeout is not None:
        _llm_timeout = request_timeout
    if _llm_timeout is None:
        _llm_timeout = 600.0
    elif _llm_timeout == 0:
        # Explicit 0 = NO per-request timeout (unlimited). Escape hatch (SOG_LLM_REQUEST_TIMEOUT=0) for a
        # legitimately-long high-effort reasoning turn that a fixed cap would otherwise cut + retry.
        _llm_timeout = None
    elif _llm_timeout < 0:
        _llm_timeout = 600.0  # a malformed negative value -> the default, never a negative timeout
    _llm_max_retries = getattr(knobs, "llm_max_retries", None) if knobs is not None else None
    if max_retries is not None:
        _llm_max_retries = max_retries
    if _llm_max_retries is None or _llm_max_retries < 0:
        _llm_max_retries = 2

    # Use defaults if still not specified
    if model is None:
        # Mirror config.py's default; the previous literal (claude-3-5-sonnet-20241022) was retired
        # and now 404s at invoke. Reachable only when both model AND config are None (the agent always
        # passes config, so config.llm resolves first), but a dead default is still a latent 404.
        model = "azure-gpt-6-astra"
    if temperature is None:
        temperature = 0.7
    if api_key is None:
        api_key = "EMPTY"
    # Auto-detect source from model name if not specified
    if source is None:
        source = resolve_source(model, source=None, base_url=base_url)
        _guard_azure_credentials_against_openai(model, source)
    else:
        source = canonical_source(source) or source  # as resolve_source reads an explicit one

    # Create appropriate model based on source
    if source == "OpenAI":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai"
            )
        # Newer OpenAI models (e.g., gpt-5-*) require the Responses API and may reject
        # legacy Chat Completions parameters like `stop`. Force Responses API when
        # using gpt-5 models to avoid 400 errors such as: "Unsupported parameter: 'stop'".
        use_responses = _is_openai_responses_family(model)

        if use_responses:
            # gpt-5* reject `stop` and non-default `temperature`; the shared mixin
            # strips both from the Responses-API payload (see _ResponsesNoStopMixin).
            class _ChatOpenAIResponsesNoStop(_ResponsesNoStopMixin, ChatOpenAI):
                pass

            return _ChatOpenAIResponsesNoStop(
                model=model,
                temperature=1,  # default for gpt-5; mixin removes it from the payload
                stop_sequences=stop_sequences,
                use_responses_api=True,
                output_version="v0",
                timeout=_llm_timeout,
                max_retries=_llm_max_retries,
            )
        elif _is_openai_reasoning_model(model):
            # o-series (o1/o3/o4-*) use the Chat Completions API (NOT Responses), but still reject
            # `stop` and any non-default `temperature`. `_ResponsesNoStopMixin` only strips when the
            # Responses API is active, so it never fires here — `_ReasoningNoTempMixin` strips both from
            # every Chat-Completions payload, else the config-default `temperature` (0.7) → HTTP 400.
            class _ChatOpenAIReasoningNoTemp(_ReasoningNoTempMixin, ChatOpenAI):
                pass

            return _ChatOpenAIReasoningNoTemp(model=model, timeout=_llm_timeout, max_retries=_llm_max_retries)
        else:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                stop_sequences=stop_sequences,
                timeout=_llm_timeout,
                max_retries=_llm_max_retries,
            )

    elif source == "AzureOpenAI":
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Azure OpenAI models. Install with: pip install langchain-openai"
            )
        # Only as a prefix, and in any case: 'Azure-gpt-5.5' routes here (the provider sniff
        # lowercases) but kept its prefix, so the deployment 404'd and the Responses routing was lost;
        # and a deployment named 'team-azure-gpt4' lost the middle of its name (u16-llm-config-12).
        model = model[len("azure-") :] if model.lower().startswith("azure-") else model
        # Strip path/query from endpoint — AzureChatOpenAI wants the resource host only,
        # but some envs include `/openai/responses?api-version=...` in OPENAI_ENDPOINT.
        raw_endpoint = os.getenv("OPENAI_ENDPOINT") or ""
        azure_endpoint = _azure_resource_endpoint(raw_endpoint)
        # GPT-5+ Azure deployments require the Responses API and reject `stop` +
        # non-default `temperature`. gpt-5 therefore always uses the Responses API
        # (it requires it); the env var still lets a legacy GPT-4 deployment opt in.
        # A GPT-4 deployment without the flag is completely unaffected.
        is_gpt5 = _is_openai_responses_family(model)
        # Through _env_flag, not a second inline comparison: setup's `_wants_completion_tokens`
        # reads this same variable with the wider, stripped `_TRUTHY` to validate the deployment,
        # so a narrower/unstripped copy here means the wizard signs a deployment off green and the
        # agent's first real turn 400s.
        use_responses = is_gpt5 or _env_flag("OPENAI_USE_RESPONSES_API")
        # Resolve the api-version (env > endpoint `?api-version=` > default, floored for
        # Responses). Without this the deliberately-set version is dropped when the URL is
        # stripped to the host, falling back to a default that predates the Responses API.
        API_VERSION = _resolve_azure_api_version(
            raw_endpoint, explicit=os.getenv("OPENAI_API_VERSION"), use_responses=use_responses
        )
        # gpt-5 accepts only the default temperature; force it so the config default
        # (0.7) can't trigger a 400. The mixin additionally strips it from the payload.
        azure_temperature = 1 if is_gpt5 else temperature
        kwargs = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "azure_endpoint": azure_endpoint,
            "azure_deployment": model,
            "openai_api_version": API_VERSION,
            "temperature": azure_temperature,
            # The ReAct stop sequences, as the OpenAI branch passes them for the same model: a
            # Chat-Completions deployment kept writing past </execute> (u16-llm-config-9). The
            # Responses and reasoning mixins below strip `stop` where a deployment rejects it.
            "stop_sequences": stop_sequences,
            "use_responses_api": use_responses,
            "timeout": _llm_timeout,
            "max_retries": _llm_max_retries,
        }
        if use_responses:
            # Flatten Responses API content blocks → plain string (mirrors OpenAI branch).
            kwargs["output_version"] = "v0"

            class _AzureChatOpenAIResponsesNoStop(_ResponsesNoStopMixin, AzureChatOpenAI):
                pass

            return _AzureChatOpenAIResponsesNoStop(**kwargs)
        if _is_openai_reasoning_model(model):
            # An o-series (o1/o3/o4) Azure DEPLOYMENT uses Chat Completions (not Responses) but still
            # rejects `stop` + non-default `temperature` — the exact gap the OpenAI branch closes via
            # _ReasoningNoTempMixin. Without it the config-default temperature (0.7) + stop → HTTP 400 on
            # the first call (the Azure branch previously special-cased only gpt-5). `azure-` is already
            # stripped above, so an `azure-o3-mini` / `o1` deployment id is recognized here.
            class _AzureChatOpenAIReasoningNoTemp(_ReasoningNoTempMixin, AzureChatOpenAI):
                pass

            return _AzureChatOpenAIReasoningNoTemp(**kwargs)
        return AzureChatOpenAI(**kwargs)

    elif source == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-anthropic package is required for Anthropic models. Install with: pip install langchain-anthropic"
            )

        # Ensure ANTHROPIC_API_KEY is loaded from the login profile if it's only there. The helper
        # hardens the capture (profile stdout can't corrupt the key; notices go to stderr).
        _load_anthropic_key_from_profile()

        # Newest Anthropic models (claude-opus-4-7/-4-8, the Claude 5 family) reject `temperature`
        # outright — "temperature is deprecated for this model." (HTTP 400). Omit it for those
        # (ChatAnthropic drops the None field from the payload); every other model keeps its value.
        anthropic_temperature = None if _anthropic_omit_temperature(model) else temperature
        anthropic_kwargs = {
            "model": model,
            "temperature": anthropic_temperature,
            "max_tokens": 8192,
            "stop_sequences": stop_sequences,
            "timeout": _llm_timeout,
            "max_retries": _llm_max_retries,
        }
        # The same newest models also reject assistant-message prefill. The STCoscientist ReAct
        # loop ends each turn with an <observation> assistant message, so re-role the observation
        # turns -- every one, not only the trailing run, so no turn's role changes between steps --
        # to `user` at the payload chokepoint (see _AnthropicNoPrefillMixin).
        if _anthropic_no_prefill(model):

            class _ChatAnthropicNoPrefill(_AnthropicNoPrefillMixin, ChatAnthropic):
                pass

            return _ChatAnthropicNoPrefill(**anthropic_kwargs)
        return ChatAnthropic(**anthropic_kwargs)

    elif source == "Gemini":
        # If you want to use ChatGoogleGenerativeAI, you need to pass the stop sequences upon invoking the model.
        # return ChatGoogleGenerativeAI(
        #     model=model,
        #     temperature=temperature,
        #     google_api_key=api_key,
        # )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Gemini models. Install with: pip install langchain-openai"
            )
        gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not gemini_key:
            # api_key=None makes the openai SDK fall back to OPENAI_API_KEY -- on an Azure install the
            # Azure key -- and send it to Google (hunt 2026-09-30, u16-llm-config-3).
            raise ValueError("GEMINI_API_KEY is not set; a Gemini model needs its own key.")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            stop_sequences=stop_sequences,
            timeout=_llm_timeout,
            max_retries=_llm_max_retries,
        )

    elif source == "Groq":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Groq models. Install with: pip install langchain-openai"
            )
        groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if not groq_key:
            # Same fallback as Gemini's: OPENAI_API_KEY would be sent to api.groq.com.
            raise ValueError("GROQ_API_KEY is not set; a Groq model needs its own key.")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            stop_sequences=stop_sequences,
            timeout=_llm_timeout,
            max_retries=_llm_max_retries,
        )

    elif source == "Ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-ollama package is required for Ollama models. Install with: pip install langchain-ollama"
            )
        return ChatOllama(
            model=model,
            temperature=temperature,
            # ChatOllama has no top-level timeout/max_retries field (they'd be silently dropped by
            # extra="ignore"); its real per-request bound is the underlying httpx client's timeout.
            client_kwargs={"timeout": _llm_timeout},
        )

    elif source == "Bedrock":
        try:
            from langchain_aws import ChatBedrock
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-aws package is required for Bedrock models. Install with: pip install langchain-aws"
            )
        _bedrock_kwargs = {
            "model": model,
            "temperature": _bedrock_temperature(model, temperature),
            "stop_sequences": stop_sequences,
            "region_name": os.getenv("AWS_REGION", "us-east-1"),
        }
        try:
            # Bedrock's real per-request bound AND retry policy live in the botocore Config -- a
            # top-level timeout=/max_retries= are silently dropped by ChatBedrock. read_timeout=None
            # means unlimited (None-safe); botocore ships with langchain_aws so this import is available
            # whenever Bedrock actually is.
            from botocore.config import Config as _BotoConfig

            _bedrock_kwargs["config"] = _BotoConfig(
                # botocore's retries.max_attempts IS the retry count (it adds 1 internally for the
                # initial request to form total_max_attempts), so this maps _llm_max_retries directly.
                read_timeout=_llm_timeout,
                retries={"max_attempts": _llm_max_retries},
            )
        except Exception:
            pass
        return ChatBedrock(**_bedrock_kwargs)

    elif source == "Custom":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for custom models. Install with: pip install langchain-openai"
            )
        # Custom LLM serving such as SGLang. Must expose an openai compatible API.
        assert base_url is not None, "base_url must be provided for customly served LLMs"
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
            base_url=base_url,
            api_key=api_key,
            timeout=_llm_timeout,
            max_retries=_llm_max_retries,
        )
        return llm

    else:
        # Built from the one list, so a valid provider (``Custom``) cannot be missing from it.
        valid = ", ".join(repr(name) for name in CANONICAL_SOURCES)
        raise ValueError(f"Invalid source: {source}. Valid options are {valid} (any capitalisation).")
