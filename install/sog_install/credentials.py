"""
Credential & configuration catalog — the single source of truth for what the
wizard collects and persists.

Grounded directly in the runtime contract (``spatialomicsgym/llm.py`` provider
construction, ``spatialomicsgym/config.py`` ``SOG_*`` env reads), NOT guessed.
Both Stage A (:mod:`onboarding`) and the Stage-B guide read from here, so a new
provider or service key is added in exactly one place.

Four tables:

* ``PROVIDER_SPECS``  — LLM providers: required/optional fields, default model,
  where-to-get-a-key pointer, and a Tier-1 REST ping descriptor.
* ``SERVICE_KEYS``    — optional per-tool credentials, each gating ONE feature;
  collected in-context by the guide only when a selected tool needs one.
* ``KNOBS``           — first-run configuration knobs (offered as advanced/defaults).
* ``STALE_NAMES``     — legacy env names the current code no longer reads
  (migrate-or-ignore; never re-prompt).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

# --------------------------------------------------------------------------- #
# Field / spec dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EnvField:
    """One environment variable the user supplies for a provider."""

    env_var: str
    label: str
    secret: bool = True  # read without echo + mask on display
    placeholder: str = ""  # shown as a hint, never a real value
    help: str = ""


@dataclass(frozen=True)
class PingSpec:
    """How to Tier-1 validate a provider over plain ``urllib`` (no SDK).

    ``auth`` selects the header/query scheme; ``url`` may contain ``{endpoint}``
    or ``{base_url}`` placeholders resolved from the user's input. ``supported``
    is False for providers we cannot cheaply ping from stdlib (Bedrock's SigV4),
    which the guide degrades to presence-only + menu fallback.
    """

    supported: bool
    method: str = "GET"
    url: str = ""
    auth: str = "none"  # x-api-key | bearer | api-key-header | query-key | none | aws-presence
    extra_headers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ProviderSpec:
    """An LLM provider the wizard can configure.

    ``source`` MUST equal one of ``spatialomicsgym.llm.ALLOWED_SOURCES`` — the
    wizard writes it to BOTH ``LLM_SOURCE`` and ``SOG_SOURCE`` so auto-detection
    is bypassed and the user's choice is authoritative.
    """

    key: str  # short menu id, e.g. "anthropic"
    label: str  # human name, e.g. "Anthropic"
    source: str  # ALLOWED_SOURCES value written to *_SOURCE
    default_model: str  # suggested SOG_LLM (user may override)
    where_to_get_key: str  # one-line pointer printed in the chooser
    required: tuple[EnvField, ...] = ()
    optional: tuple[EnvField, ...] = ()
    ping: PingSpec = field(default_factory=lambda: PingSpec(supported=False))
    notes: str = ""

    def all_fields(self) -> tuple[EnvField, ...]:
        return self.required + self.optional

    def secret_vars(self) -> set[str]:
        return {f.env_var for f in self.all_fields() if f.secret}


@dataclass(frozen=True)
class ServiceKey:
    """An optional per-tool credential, collected only when a selected tool needs it."""

    env_var: str
    label: str
    gates: str  # which tool/feature this unlocks (human text)
    where_to_get: str
    secret: bool = True


@dataclass(frozen=True)
class Knob:
    """A first-run configuration knob (offered with a sensible default)."""

    env_var: str
    label: str
    default: str
    kind: str  # path | bool | int | str
    help: str = ""


@dataclass(frozen=True)
class StaleName:
    """A legacy env name the current code ignores; migrate its value forward."""

    old: str
    new: str | None  # None → drop it (no modern equivalent)


# --------------------------------------------------------------------------- #
# PROVIDER_SPECS — LLM providers (default: Azure OpenAI; repo default model is
# azure-gpt-6-astra so OPENAI_API_KEY + OPENAI_ENDPOINT are the out-of-box keys).
# --------------------------------------------------------------------------- #
_ANTHROPIC = ProviderSpec(
    key="anthropic",
    label="Anthropic",
    source="Anthropic",
    default_model="claude-opus-4-8",
    where_to_get_key="https://console.anthropic.com/settings/keys",
    required=(EnvField("ANTHROPIC_API_KEY", "Anthropic API key", secret=True, placeholder="sk-ant-..."),),
    ping=PingSpec(
        supported=True,
        url="https://api.anthropic.com/v1/models",
        auth="x-api-key",
        extra_headers=(("anthropic-version", "2023-06-01"),),
    ),
    notes="claude-* model names route here automatically.",
)

_OPENAI = ProviderSpec(
    key="openai",
    label="OpenAI",
    source="OpenAI",
    default_model="gpt-4o",
    where_to_get_key="https://platform.openai.com/api-keys",
    required=(EnvField("OPENAI_API_KEY", "OpenAI API key", secret=True, placeholder="sk-..."),),
    ping=PingSpec(supported=True, url="https://api.openai.com/v1/models", auth="bearer"),
    notes="gpt-5* models are routed through the Responses API automatically.",
)

_AZURE = ProviderSpec(
    key="azure",
    label="Azure OpenAI",
    source="AzureOpenAI",
    default_model="gpt-6-astra",  # the Azure *deployment* name (matches the repo default azure-gpt-6-astra)
    where_to_get_key="Azure Portal → your OpenAI resource → Keys and Endpoint",
    required=(
        EnvField("OPENAI_API_KEY", "Azure OpenAI key", secret=True),
        EnvField(
            "OPENAI_ENDPOINT",
            "Azure resource endpoint",
            secret=False,
            placeholder="https://<resource>.openai.azure.com",
        ),
    ),
    optional=(
        EnvField("OPENAI_API_VERSION", "API version", secret=False, placeholder="2024-12-01-preview"),
        EnvField(
            "OPENAI_USE_RESPONSES_API",
            "Use Responses API (gpt-5+)",
            secret=False,
            placeholder="false",
            help="Set true only for gpt-5+ deployments that reject max_tokens.",
        ),
    ),
    ping=PingSpec(
        supported=True,
        # Data-plane "list models" GET — the canonical route a valid api-key can reach on
        # the resource host. (The old `/openai/deployments` *list* route is control-plane
        # only — AAD-authenticated on management.azure.com — so it 404s on the data-plane
        # host for every modern api-version. The real client hits the per-deployment
        # `/openai/deployments/{deployment}/chat/completions` route, never this one.)
        # api-version appended at ping time from OPENAI_API_VERSION (or its default).
        url="{endpoint}/openai/models",
        auth="api-key-header",
    ),
    notes="Default provider — the model name is your Azure *deployment* id. Endpoint host only — path/query is stripped.",
)

_GEMINI = ProviderSpec(
    key="gemini",
    label="Gemini",
    source="Gemini",
    default_model="gemini-2.0-flash",
    where_to_get_key="https://aistudio.google.com/app/apikey",
    required=(EnvField("GEMINI_API_KEY", "Gemini API key", secret=True),),
    ping=PingSpec(
        supported=True,
        url="https://generativelanguage.googleapis.com/v1beta/openai/models",
        auth="bearer",
    ),
    notes="Served via the Google OpenAI-compatible endpoint.",
)

_GROQ = ProviderSpec(
    key="groq",
    label="Groq",
    source="Groq",
    default_model="llama-3.3-70b-versatile",
    where_to_get_key="https://console.groq.com/keys",
    required=(EnvField("GROQ_API_KEY", "Groq API key", secret=True, placeholder="gsk_..."),),
    ping=PingSpec(supported=True, url="https://api.groq.com/openai/v1/models", auth="bearer"),
)

_BEDROCK = ProviderSpec(
    key="bedrock",
    label="AWS Bedrock",
    source="Bedrock",
    default_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    where_to_get_key="AWS Console → Bedrock → model access (uses your AWS credentials)",
    required=(EnvField("AWS_REGION", "AWS region", secret=False, placeholder="us-east-1"),),
    optional=(
        EnvField("AWS_BEARER_TOKEN_BEDROCK", "Bedrock bearer token", secret=True),
        EnvField("AWS_ACCESS_KEY_ID", "AWS access key id", secret=True),
        EnvField("AWS_SECRET_ACCESS_KEY", "AWS secret access key", secret=True),
    ),
    ping=PingSpec(supported=False, auth="aws-presence"),
    notes="Requires a region plus either a bearer token or an access-key pair. "
    "Stdlib cannot cheaply SigV4-sign, so Tier-1 is presence-only and the guide "
    "falls back to menu-with-advice.",
)

_CUSTOM = ProviderSpec(
    key="custom",
    label="Custom / local (OpenAI-compatible)",
    source="Custom",
    default_model="default",
    where_to_get_key="Your own serving endpoint (vLLM / SGLang / TGI, OpenAI-compatible)",
    required=(
        EnvField(
            "SOG_CUSTOM_BASE_URL",
            "Base URL",
            secret=False,
            placeholder="http://localhost:8000/v1",
        ),
    ),
    optional=(EnvField("SOG_CUSTOM_API_KEY", "API key (if the server needs one)", secret=True),),
    ping=PingSpec(supported=True, url="{base_url}/models", auth="bearer"),
    notes="Any server exposing the OpenAI /v1 API. Key may be omitted (defaults to 'EMPTY').",
)

_OLLAMA = ProviderSpec(
    key="ollama",
    label="Ollama (local)",
    source="Ollama",
    default_model="llama3.1",
    where_to_get_key="No key needed — run `ollama serve` locally",
    ping=PingSpec(supported=True, url="http://localhost:11434/api/tags", auth="none"),
    notes="Talks to a local Ollama daemon on :11434. Pull the model first with `ollama pull`.",
)

# Ordered for the chooser: Anthropic first (default).
PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    _ANTHROPIC,
    _OPENAI,
    _AZURE,
    _GEMINI,
    _GROQ,
    _BEDROCK,
    _CUSTOM,
    _OLLAMA,
)

# The provider of the shipped default model (config.py: azure-gpt-6-astra).
DEFAULT_PROVIDER_KEY = "azure"

_PROVIDERS_BY_KEY = {p.key: p for p in PROVIDER_SPECS}
_PROVIDERS_BY_SOURCE = {p.source: p for p in PROVIDER_SPECS}


def get_provider(key_or_source: str) -> ProviderSpec:
    """Look a provider up by menu key (``anthropic``) or source (``Anthropic``)."""
    if key_or_source in _PROVIDERS_BY_KEY:
        return _PROVIDERS_BY_KEY[key_or_source]
    if key_or_source in _PROVIDERS_BY_SOURCE:
        return _PROVIDERS_BY_SOURCE[key_or_source]
    raise KeyError(f"unknown provider {key_or_source!r}")


def default_provider() -> ProviderSpec:
    return _PROVIDERS_BY_KEY[DEFAULT_PROVIDER_KEY]


# Model ids that older sog-setup versions shipped as a default and therefore WROTE into a user's
# ``.env`` (``SOG_LLM``) and key vault, but which are invalid at the provider — they pass the
# key-only Tier-1 validation (Anthropic's ``/v1/models`` probe checks the *key*, not the model) and
# then return HTTP 404 on the first real chat call. Each maps to the provider *source* whose current
# ``default_model`` should replace it; resolving through the ProviderSpec (rather than hardcoding a
# replacement string here) keeps this migration in lockstep with the shipped default — bump
# ``_ANTHROPIC.default_model`` and the repair follows automatically.
_KNOWN_BAD_MODELS: dict[str, str] = {
    # A "4.5" label fused with Sonnet-4.0's 20250514 date — never a real model id. It was the
    # sog-setup default before 2026-07-10, so any ``.env``/vault written by an older clone carries it
    # and 404s the Stage-B guide (and, later, the agent) even after ``git pull`` fixes the default.
    "claude-sonnet-4-5-20250514": "Anthropic",
}


def sanitize_model(model: str | None, *, source: str | None = None) -> tuple[str | None, str | None]:
    """Repair a model id that an older sog-setup wrote to ``.env``/the key vault but that is invalid
    at the provider (passes the key-only Tier-1 ping, then 404s on the first real call).

    Returns ``(model, note)``. When a substitution happens, ``model`` is the corrected id and
    ``note`` is a plain-language message to surface to the user; otherwise ``model`` is returned
    unchanged and ``note`` is ``None``. Never raises — a lookup miss is simply a no-op.

    ``source`` (the config's provider source, when known) gates the migration to the matching
    provider, so a Custom/Ollama endpoint that legitimately names a colliding string is left alone.
    """
    if not model or not isinstance(model, str):
        # Honor the "Never raises" contract even for a truthy non-string model. ``answers._validate``
        # now rejects a non-string ``llm.model`` on the ``--answers`` path, but this helper is also
        # reached by the llm_chat Stage-B guide from a possibly hand-corrupted vault/.env that never
        # passes through that validator — a bare ``.strip()`` there would AttributeError. Return as-is.
        return model, None
    bad = model.strip()
    repl_source = _KNOWN_BAD_MODELS.get(bad)
    if repl_source is None:
        return model, None
    if source is not None and source != repl_source:
        return model, None
    try:
        repl = get_provider(repl_source).default_model
    except KeyError:  # pragma: no cover — repl_source is always a shipped source
        return model, None
    note = (
        f"the saved model id {bad!r} is not a valid model — it passes the key check but returns "
        f"HTTP 404 on the first real call, so I'm using {repl!r} instead. Set SOG_LLM in your .env "
        f"to pick a different model."
    )
    return repl, note


def azure_deployment_from_endpoint(endpoint: str) -> str | None:
    """Recover the Azure *deployment* name from a full inference endpoint like
    ``…/openai/deployments/<name>/chat/completions`` — users routinely paste the whole URL,
    and it names the exact deployment the agent must call (there is no universal Azure
    default, unlike every other provider). Returns ``None`` for a bare resource host, where
    the name genuinely isn't knowable and the caller must ask or fall back.

    Lives here (with the Azure spec, at the bottom of the setup dep graph) so BOTH the
    interactive prompt (``onboarding``) and the reuse/persist path (``llm_setup``) recover the
    deployment identically — otherwise re-running the wizard silently re-persists ``gpt-4o``."""
    m = re.search(r"/deployments/([^/?]+)", endpoint or "")
    return m.group(1) if m else None


def azure_resource_endpoint(endpoint: str) -> str:
    """The resource address, from whatever the operator pasted.

    Users routinely paste the full inference URL, and every caller that builds a request needs the
    address without the API path on it.

    This was written in two places as ``endpoint.split("/openai", 1)[0]``, which reads the STRING
    and not the URL: the first ``/openai`` in ``https://openai-prod.openai.azure.com`` is the one
    formed by the second slash of ``https://`` and the resource name, so that endpoint became
    ``https:/`` and both the wizard's ping and the agent's first turn DNS-failed. ``openai-prod``,
    ``openai-eastus`` and plain ``openai`` are ordinary Microsoft resource names.

    Lives here, beside :func:`azure_deployment_from_endpoint` and for the same stated reason: both
    the ping (``llm_setup``) and the live chat (``llm_chat``) must recover it identically, and the
    way two copies drift is exactly what this bug was. It mirrors ``llm._azure_resource_endpoint``
    by hand because the setup package does not import the agent package -- the wizard's
    non-interference invariant -- so the rule is stated twice on purpose and tested in both places.
    """
    raw = (endpoint or "").strip()
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
        # No authority to protect: `urlsplit` put the whole value in `path`.
        return path or raw.rstrip("/")
    return f"{parts.scheme}://{parts.netloc}{path}"


# --------------------------------------------------------------------------- #
# SERVICE_KEYS — optional, tool-gating. Each gates exactly one feature and is
# collected in-context by the guide only when a *selected* tool declares it.
# --------------------------------------------------------------------------- #
SERVICE_KEYS: tuple[ServiceKey, ...] = (
    ServiceKey(
        "PROTOCOLS_IO_ACCESS_TOKEN",
        "protocols.io access token",
        gates="Retrieving full protocols.io method text",
        where_to_get="https://www.protocols.io/developers",
    ),
    ServiceKey(
        "SYNAPSE_AUTH_TOKEN",
        "Synapse auth token",
        gates="Downloading controlled data from Synapse (Sage Bionetworks)",
        where_to_get="https://www.synapse.org → Account → Personal Access Tokens",
    ),
    ServiceKey(
        "UCD_TOKEN",
        "UCDeconvolve token",
        gates="UCDeconvolve cloud deconvolution",
        where_to_get="https://ucdeconvolve.org (register for an API token)",
    ),
    ServiceKey(
        "DREMIO_PAT",
        "Dremio personal access token",
        gates="Querying a Dremio data lakehouse",
        where_to_get="Your Dremio instance → Account Settings → Personal Access Tokens",
    ),
    ServiceKey(
        "NCBI_EMAIL",
        "NCBI Entrez email",
        gates="Higher-rate NCBI Entrez requests (identifies you; not secret)",
        where_to_get="Your own email address",
        secret=False,
    ),
    ServiceKey(
        "ALTMETRIC_API_KEY",
        "Altmetric API key",
        gates="Altmetric attention scores for publications",
        where_to_get="https://www.altmetric.com/products/altmetric-api/",
    ),
    ServiceKey(
        "OPENALEX_MAILTO",
        "OpenAlex polite-pool email",
        gates="Faster OpenAlex literature queries (polite pool; not secret)",
        where_to_get="Your own email address",
        secret=False,
    ),
)

SERVICE_KEY_BY_VAR = {s.env_var: s for s in SERVICE_KEYS}
_SERVICE_KEYS_BY_VAR = SERVICE_KEY_BY_VAR  # legacy internal alias


def service_keys_in(text: str) -> list[ServiceKey]:
    """Return the service keys whose env var literally appears in ``text``.

    Used to detect which credentials a *selected* tool actually needs by
    scanning its worker source / description — so the guide asks for
    ``UCD_TOKEN`` when UCDeconvolve is picked but never for RCTD. Non-brittle:
    the mapping follows the code, not a hand-maintained table.
    """
    if not text:
        return []
    return [s for s in SERVICE_KEYS if s.env_var in text]


# --------------------------------------------------------------------------- #
# KNOBS — first-run configuration (offered as advanced; each has a safe default).
# --------------------------------------------------------------------------- #
KNOBS: tuple[Knob, ...] = (
    Knob("SOG_DATA_PATH", "Data directory", "./data", "path", "Where datasets are read/written."),
    Knob("SOG_LLM", "Model name", "", "str", "Override the provider's default model."),
    Knob("SOG_TIMEOUT_SECONDS", "Per-step timeout (s)", "600", "int", "Max seconds per agent execution step."),
    Knob(
        "SOG_TOOL_CREATION_ENABLED",
        "Allow creating tools from GitHub",
        "false",
        "bool",
        "Lets the agent build new MCP tools from GitHub repos.",
    ),
    Knob(
        "SOG_SELF_REVIEW_ENABLED",
        "Auto-repair failed builds/tests",
        "false",
        "bool",
        "Classifies and remediates failures before rolling back.",
    ),
    Knob(
        "SOG_MEMORY_ENABLED",
        "Remember tool-creation attempts",
        "false",
        "bool",
        "Advisory memory of prior successes/failures.",
    ),
)


# --------------------------------------------------------------------------- #
# STALE_NAMES — legacy env vars the current code no longer reads. Migrate their
# value to the modern name (with a heads-up) and never re-prompt for them.
# --------------------------------------------------------------------------- #
STALE_NAMES: tuple[StaleName, ...] = (
    StaleName("BIOMNI_TEMPERATURE", "SOG_TEMPERATURE"),
    StaleName("CUSTOM_MODEL_BASE_URL", "SOG_CUSTOM_BASE_URL"),
    StaleName("CUSTOM_MODEL_API_KEY", "SOG_CUSTOM_API_KEY"),
    StaleName("BIOMNI_DATA_PATH", "SOG_DATA_PATH"),
    StaleName("BIOMNI_TIMEOUT_SECONDS", "SOG_TIMEOUT_SECONDS"),
)

STALE_BY_OLD = {s.old: s for s in STALE_NAMES}


# --------------------------------------------------------------------------- #
# NEVER_ASK — capabilities that are NOT credential-gated. The guide must never
# prompt for these (they work against public/open endpoints).
# --------------------------------------------------------------------------- #
NEVER_ASK: tuple[str, ...] = (
    "Public S3 dataset downloads (open bucket, no key)",
    "Public HuggingFace model/dataset downloads (no token for public repos)",
    "NCBI E-utilities / BLAST (open; NCBI_EMAIL is optional courtesy only)",
)
