"""
Minimal stdlib ``urllib`` chat client — the Stage-B guide's brain-stem.

The guide can't ``import get_llm`` (langchain lives in the env being built), so
this is a tiny, dependency-free chat client that POSTs to the provider's
chat/completions endpoint using the key already collected in Stage A. It
supports the OpenAI-shaped providers (OpenAI/Azure/Gemini/Groq/Custom),
Anthropic Messages, and Ollama; **Bedrock is unsupported** (SigV4 signing is too
heavy for stdlib) and raises :class:`LLMUnsupported` so the guide degrades to a
plain menu.

Only two calls are used by the guide: :meth:`ChatClient.chat` (free text) and
:meth:`ChatClient.chat_json` (a small JSON object, with parse-and-retry).
"""

from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from . import constants, credentials

# gpt-5-or-later / o-series reasoning models reject `max_tokens` and require
# `max_completion_tokens` (Azure + OpenAI). A deployment token at the start of the id or after a
# -/_// separator — so "gpt-5.5", "gpt-6-astra", "o3-mini", "my-o4-deploy" match, but "gpt-4o",
# "gpt-35-turbo" and "amigo1" do not.
#
# The generation range is [5-9] rather than a literal 5 for the reason llm.py's twin gate states:
# this is the copy the wizard's Stage-B guide reads (the portal's /api/config/model probe goes through
# llm.get_llm), so a narrower spelling here signs a deployment off GREEN and the agent's first real
# turn 400s. Measured against the Azure deployment `gpt-6-astra` on 2026-09-21:
#
#     max_tokens            -> 400 "Unsupported parameter: 'max_tokens' is not supported with
#                                   this model. Use 'max_completion_tokens' instead."
#     max_completion_tokens -> 200
#
# Kept as its own regex rather than importing `llm._is_openai_responses_family`: this module is
# part of the stdlib-only setup path and must not pull in langchain. The two are twins on the
# gpt- arm and the comment in each names the other.
_REASONING_MODEL_RE = re.compile(r"(?i)(?:^|[-_/])(?:gpt-?[5-9]|o[134])")
_TRUTHY = {"1", "true", "yes", "y", "on"}
# The two spellings of the token-limit parameter, swapped for each other on a provider 400.
_TOKEN_PARAMS = ("max_tokens", "max_completion_tokens")


class LLMChatError(RuntimeError):
    """A network / HTTP / parse failure talking to the provider."""


class LLMUnsupported(LLMChatError):
    """This provider can't be driven from the stdlib client (e.g. Bedrock)."""


@dataclass
class ChatClient:
    source: str  # ALLOWED_SOURCES value
    model: str
    field_values: dict  # env-var → value collected in Stage A
    timeout: int = constants.LLM_PING_TIMEOUT_SEC

    def __post_init__(self) -> None:
        # Belt-and-braces: if a stale/invalid model id slips through onboarding (a hand-edited .env,
        # an older vault entry, or a direct construction), repair it here so the guide never sends a
        # model that 404s. Onboarding already warns the user about the substitution when it makes it;
        # this is the silent last line of defense. ``credentials`` is a stdlib-only leaf → no cycle.
        repaired, _ = credentials.sanitize_model(self.model, source=self.source)
        if repaired is not None:
            self.model = repaired

    # -- public API -----------------------------------------------------------
    def chat(self, messages: list[dict], *, system: str | None = None, max_tokens: int = 1024) -> str:
        """Return the assistant's text for a message list. Raises on failure.

        If the provider rejects the token-limit parameter we sent (a gpt-5/o-series deployment
        wants ``max_completion_tokens`` where we sent ``max_tokens``, or the inverse for an older
        model), transparently swap to the spelling the error names and retry once. Azure
        deployment names are user-chosen and needn't reveal the model, so the provider's own 400
        is the authoritative signal — the preemptive heuristic in ``_build_request`` only saves a
        round-trip in the common case."""
        url, headers, body = self._build_request(messages, system, max_tokens)
        try:
            raw = self._post(url, headers, body)
        except LLMChatError as exc:
            # The token-limit-parameter swap only makes sense for OpenAI/Azure (the gpt-5/o-series
            # ``max_completion_tokens`` case). For Anthropic/Ollama/Groq/Custom the body's
            # ``max_tokens`` is already correct, so never let a coincidental error string trigger a
            # pointless — or wrong-provider — retry.
            if self.source not in ("OpenAI", "AzureOpenAI"):
                raise
            swapped = _swap_token_param(body)
            if swapped is None or not _is_token_param_error(str(exc)):
                raise
            raw = self._post(url, headers, swapped)  # provider dictates the token param; honor it
        return self._extract_text(raw)

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        retries: int = 2,
        max_tokens: int = 1024,
    ) -> dict:
        """Ask for a single JSON object and parse it, retrying on malformed output."""
        sys_prompt = system + "\n\nReply with ONE JSON object and nothing else — no prose, no code fences."
        messages = [{"role": "user", "content": user}]
        last_err = ""
        for _ in range(retries + 1):
            try:
                text = self.chat(messages, system=sys_prompt, max_tokens=max_tokens)
            except LLMUnsupported:
                raise
            except LLMChatError as exc:
                last_err = str(exc)
                continue
            parsed = _extract_json(text)
            if parsed is not None:
                return parsed
            last_err = f"non-JSON reply: {text[:120]!r}"
            # nudge the model on the next attempt
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": text},
                {"role": "user", "content": "That was not valid JSON. Reply with ONLY the JSON object."},
            ]
        raise LLMChatError(f"could not obtain JSON after {retries + 1} tries: {last_err}")

    # -- request building (per provider) -------------------------------------
    def _build_request(self, messages, system, max_tokens):
        fv = self.field_values
        src = self.source

        if src == "Bedrock":
            raise LLMUnsupported("Bedrock is not supported by the stdlib guide client")

        if src == "Anthropic":
            headers = {
                "x-api-key": fv.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            }
            if system:
                body["system"] = system
            return "https://api.anthropic.com/v1/messages", headers, body

        if src == "Ollama":
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            return (
                "http://localhost:11434/api/chat",
                {"content-type": "application/json"},
                {"model": self.model, "messages": msgs, "stream": False},
            )

        # OpenAI-shaped: OpenAI / Azure / Gemini / Groq / Custom
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        token_param = "max_completion_tokens" if _wants_completion_tokens(src, self.model, fv) else "max_tokens"
        body = {"model": self.model, "messages": msgs, token_param: max_tokens}
        headers = {"content-type": "application/json"}

        if src == "OpenAI":
            headers["Authorization"] = f"Bearer {fv.get('OPENAI_API_KEY', '')}"
            return "https://api.openai.com/v1/chat/completions", headers, body
        if src == "Gemini":
            headers["Authorization"] = f"Bearer {fv.get('GEMINI_API_KEY', '')}"
            return (
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers,
                body,
            )
        if src == "Groq":
            headers["Authorization"] = f"Bearer {fv.get('GROQ_API_KEY', '')}"
            return "https://api.groq.com/openai/v1/chat/completions", headers, body
        if src == "Custom":
            base = fv.get("SOG_CUSTOM_BASE_URL", "").rstrip("/")
            if fv.get("SOG_CUSTOM_API_KEY"):
                headers["Authorization"] = f"Bearer {fv['SOG_CUSTOM_API_KEY']}"
            return f"{base}/chat/completions", headers, body
        if src == "AzureOpenAI":
            raw_endpoint = fv.get("OPENAI_ENDPOINT") or ""
            endpoint = credentials.azure_resource_endpoint(raw_endpoint)
            # The deployment and api-version llm.py would call (hunt 2026-09-30, u35b-setup-state-5). An
            # `azure-` prefix only picks the provider -- llm.py strips it, case-insensitively, before using
            # the name as the deployment -- so the documented `SOG_LLM=azure-gpt-5.5` 404'd here and the
            # guide fell back to plain menus. The version follows llm._resolve_azure_api_version: the
            # explicit OPENAI_API_VERSION, else the `?api-version=` the user put in the endpoint, else the
            # default (no Responses floor: this client speaks chat/completions).
            model = self.model or ""
            deployment = model[len("azure-") :] if model.lower().startswith("azure-") else model
            version = fv.get("OPENAI_API_VERSION") or _api_version_from_url(raw_endpoint) or "2024-12-01-preview"
            headers["api-key"] = fv.get("OPENAI_API_KEY", "")
            url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}"
            return url, headers, body

        raise LLMUnsupported(f"unknown source {src!r}")

    # -- transport / parsing --------------------------------------------------
    def _post(self, url: str, headers: dict, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        try:
            # ``Request(...)`` must be INSIDE the try: it raises ``ValueError('unknown url type')`` at
            # CONSTRUCTION for a schemeless endpoint (``api.host/v1`` with no ``https://`` — the common
            # custom/Azure typo), so building it outside would let that escape unwrapped (C-F2).
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200] if hasattr(exc, "read") else ""
            raise LLMChatError(f"HTTP {exc.code} from {self.source}: {detail}") from exc
        except json.JSONDecodeError as exc:
            raise LLMChatError(f"non-JSON HTTP body from {self.source}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, http.client.HTTPException) as exc:
            # A schemeless / malformed endpoint (``api.host/v1`` with no ``https://``) makes ``urlopen``
            # raise ``ValueError('unknown url type')``; a non-numeric port raises ``http.client.InvalidURL``
            # (an ``HTTPException``, NOT an ``OSError``). Neither is caught by the URLError/OSError arm, so
            # without this a typo'd custom/Azure endpoint escapes as a bare exception past the guide's
            # graceful ``LLMChatError`` fallback (C-F2) and aborts the wizard. ``JSONDecodeError`` (a
            # ``ValueError`` subclass) is handled by the arm ABOVE, so a bad body still reads "non-JSON".
            raise LLMChatError(f"could not reach {self.source}: {getattr(exc, 'reason', exc)}") from exc

    def _extract_text(self, raw: dict) -> str:
        try:
            if self.source == "Anthropic":
                parts = raw.get("content", [])
                text: object = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
            elif self.source == "Ollama":
                text = raw.get("message", {}).get("content", "")
            else:  # OpenAI-shaped
                text = raw["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            # AttributeError guards the Ollama ``{"message": null}`` case: ``raw.get("message", {})``
            # returns None (the key exists), so ``.get("content", "")`` would raise and otherwise
            # escape as a raw traceback that kills the wizard instead of a friendly LLMChatError.
            raise LLMChatError(f"unexpected response shape from {self.source}: {exc}") from exc
        # Guarantee a str (G1). Some OpenAI-compatible backends return ``content`` as a LIST of
        # structured parts (``[{"type":"text","text":...}, ...]``) rather than a bare string; a few
        # return a number. Left as-is, that non-str flows into ``_extract_json``'s ``text.strip()``
        # and crashes the wizard with a raw ``AttributeError`` — past ``_ask_json``'s graceful
        # ``LLMChatError`` net. Flatten a parts-list into its text, and coerce anything else to str.
        if isinstance(text, list):
            text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
        return text if isinstance(text, str) else str(text)


def _api_version_from_url(url: str) -> str | None:
    """The ``api-version`` query parameter of an Azure endpoint URL, if present (mirrors
    ``llm._api_version_from_url``; this module does not import langchain-bearing ``llm``)."""
    try:
        vals = parse_qs(urlsplit(url).query).get("api-version")
    except ValueError:
        return None
    return vals[0] if vals else None


def _wants_completion_tokens(source: str, model: str, field_values: dict) -> bool:
    """Preemptively pick ``max_completion_tokens`` over ``max_tokens`` for a request.

    Only OpenAI/Azure host the gpt-5/o-series reasoning models that require it; other
    OpenAI-shaped backends (Groq, Gemini-compat, local/Custom) may not accept the newer spelling
    at all, so they keep ``max_tokens``. The explicit ``OPENAI_USE_RESPONSES_API`` knob is
    honored as a user signal that the deployment is gpt-5+, since an Azure deployment name can
    be arbitrary. Anything this misses is caught by the 400-driven swap in ``chat``."""
    if source not in ("OpenAI", "AzureOpenAI"):
        return False
    if str(field_values.get("OPENAI_USE_RESPONSES_API", "")).strip().lower() in _TRUTHY:
        return True
    return bool(_REASONING_MODEL_RE.search(model or ""))


def _swap_token_param(body: dict) -> dict | None:
    """Return a copy of an OpenAI-shaped request body with the token-limit parameter swapped to
    the other spelling (``max_tokens`` ↔ ``max_completion_tokens``), preserving every other key.
    ``None`` when neither spelling is present (nothing to swap)."""
    for i, name in enumerate(_TOKEN_PARAMS):
        if name in body:
            other = _TOKEN_PARAMS[1 - i]
            swapped = {k: v for k, v in body.items() if k not in _TOKEN_PARAMS}
            swapped[other] = body[name]
            return swapped
    return None


def _is_token_param_error(detail: str) -> bool:
    """True when a provider 400 says the token-limit parameter we sent is unsupported and names
    the other spelling — the reasoning-model "use max_completion_tokens" case (and its inverse).
    Kept strict so an unrelated 400/auth/network error is never mistaken for it."""
    d = detail.lower()
    names_param = "max_tokens" in d or "max_completion_tokens" in d
    return names_param and ("unsupported parameter" in d or "not supported" in d or "is not supported" in d)


def _extract_json(text: str) -> dict | None:
    """Best-effort parse of a single JSON object out of an LLM reply."""
    if not text:
        return None
    s = text.strip()
    # strip ```json fences
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.lstrip("`")
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
    s = s.strip()
    # direct parse
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # fallback: scan for balanced ``{...}`` spans and try each until one parses. String-aware, so a
    # brace *inside a JSON string value* (e.g. ``{"note": "use a }"}``) doesn't miscount the depth and
    # close the object early; and a span that fails to parse doesn't abort the scan — we keep looking
    # for the next balanced object rather than giving up on the first malformed candidate.
    depth = 0
    in_str = False
    esc = False
    span_start = -1
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                span_start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and span_start != -1:
                try:
                    obj = json.loads(s[span_start : i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass  # not valid here — keep scanning for the next balanced span
                span_start = -1
    return None
