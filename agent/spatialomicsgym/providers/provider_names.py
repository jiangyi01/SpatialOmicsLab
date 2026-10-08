"""Provider-name rules the LLM layer and the setup probe both have to answer the same way.

Two layers decide which vendor a run belongs to, and neither can import the other's answer.
:mod:`spatialomicsgym.llm` builds the client, but it imports ``langchain_core`` at module scope --
one of the very packages the setup health probe exists to check *for* -- so importing the resolver
from the probe would fail on exactly the broken environment the probe was written to detect.
:mod:`sog_install.base_env` is stdlib-only for the same reason.

So the shared rules live here, in a module with no third-party imports and no dependency on the
rest of the package. Deliberately absent: anything that reads the environment. Each caller has its
own precedence over *which* variables to consult and when, and those differ for good reasons --
what must not differ is what a model name proves, which environment defaults such a name may
override, and how a provider's name is spelled.
"""

from __future__ import annotations

# Mirrors ``llm.SourceType``, which cannot be imported here (langchain at module scope) and cannot
# be built from this tuple either (``Literal`` takes literals). ``test_the_canonical_source_names
# _stay_in_step_with_the_type`` pins the two equal, so the mirror cannot silently fall behind.
CANONICAL_SOURCES: tuple[str, ...] = (
    "OpenAI",
    "AzureOpenAI",
    "Anthropic",
    "Ollama",
    "Gemini",
    "Bedrock",
    "Groq",
    "Custom",
)

# Providers with a fixed, published model catalogue: a foreign prefix PROVES such an env default
# wrong, so the model name may override it. The others (AzureOpenAI/Bedrock/Ollama/Custom/Groq)
# carry free-form names the user picks — an Azure deployment may legitimately be called "gpt-4o" or
# "claude-sonnet" — so a foreign-looking name proves nothing and the env var still wins.
OVERRIDABLE_ENV_SOURCES: frozenset[str] = frozenset({"Anthropic", "OpenAI", "Gemini"})


def canonical_source(value: str | None) -> str | None:
    """The canonically-spelled provider ``value`` names, or ``None`` if it names none.

    Whitespace-tolerant and case-insensitive, because the value normally arrives from a hand-edited
    ``.env`` line. Returning ``None`` rather than the input for an unrecognised name is what lets a
    caller distinguish "no provider configured" from "a provider I could not place", which are
    different situations: the first is ordinary, the second is a typo worth not acting on.
    """
    if not value or not isinstance(value, str):
        return None
    folded = value.strip().lower()
    return next((name for name in CANONICAL_SOURCES if name.lower() == folded), None)


def source_from_model_prefix(model: str | None) -> str | None:
    """Return the provider a model name *proves*, or ``None`` when the name proves nothing.

    A name beginning ``claude-``/``gpt-``/``gemini-``/``azure-`` names its provider outright.
    Ambiguous names (a bare ``llama3``, a fine-tune, a free-form Azure deployment id) return
    ``None`` so the caller's ``SOG_SOURCE``/``base_url``/fallback chain keeps deciding.

    Matched case-insensitively, as ``chat_cli._detect_source`` has always done (it lowercases at
    chat_cli.py:213). While this table was case-sensitive, ``STCoscientist(llm="GPT-4o")`` raised
    "Unable to determine model source" for a name ``stcoscientist -m GPT-4o`` resolved without
    complaint -- and the same for ``Claude-``/``Gemini-``/``Azure-``. Providers accept these names
    case-insensitively; only our name table did not.
    """
    if not model or not isinstance(model, str):
        return None
    model = model.lower()
    if model.startswith("claude-"):
        return "Anthropic"
    if model.startswith("gpt-oss"):
        # A local Ollama build — NOT OpenAI. Must be checked before the bare `gpt-` rule, but it is
        # an Ollama (free-form) name, so it never overrides an explicit env source.
        return None
    if model.startswith(("gpt-", "chatgpt-")):
        return "OpenAI"
    if model[:1] == "o" and model[1:2].isdigit() and model[2:3] in ("", "-"):
        # OpenAI's o-series reasoning models (o1, o3, o3-mini, o4-mini) carry no ``gpt`` anywhere in
        # the name, so this function used to prove nothing about them and ``resolve_source`` fell
        # through to "Unable to determine model source": ``STCoscientist(llm="o3-mini")`` raised
        # while ``stcoscientist -m o3-mini`` ran, off an identical name table in chat_cli. Same "o"
        # + digit + boundary test the CLI uses (chat_cli.py:223), so ``ollama``/``o1lama`` stay
        # unclaimed. An Azure deployment named ``o3-mini`` is unaffected: AzureOpenAI is not in
        # ``OVERRIDABLE_ENV_SOURCES``, so its env default still outranks this.
        return "OpenAI"
    if model.startswith("azure-"):
        return "AzureOpenAI"
    if model.startswith("gemini-"):
        return "Gemini"
    return None
