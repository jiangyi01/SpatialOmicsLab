"""
``stcoscientist`` / ``sog-chat`` — an interactive terminal chat with
the ST-Coscientist spatial-omics agent.

Two ways to talk to the agent, chosen automatically:

* **Interactive REPL** — no question given and attached to a terminal::

      $ stcoscientist

  A live, multi-turn session (the agent keeps context between questions, like a
  chat). Type a spatial-omics question in plain English; type ``/exit`` to quit.
  Up/Down recall previous prompts, Tab completes ``/commands``, and a line ending
  with a backslash continues on the next line (multi-line prompts).

* **One-shot** — a quoted question, or a question piped on stdin::

      $ stcoscientist "Which tool detects spatial domains in Visium data?"
      $ echo "Summarise my h5ad at ./data/visium.h5ad" | stcoscientist
      $ stcoscientist -q "..."          # -q prints ONLY the answer (pipe-friendly)
      $ stcoscientist --json "..."      # emit {"answer": "..."} for scripts

Discovery, no key needed::

      $ stcoscientist --list-models     # common model names per provider
      $ stcoscientist --version

Design notes
------------
* The heavy agent (:class:`spatialomicsgym.agent.STCoscientist` — it pulls in
  LangGraph and the whole tool registry, and its ``default_config`` singleton
  reads the ``SOG_*`` environment at *import* time) is imported **lazily**: only
  after argument parsing, ``.env`` loading, and the API-key pre-flight all pass.
  So ``--help`` / ``--version`` / ``--list-models`` / a missing-key message stay
  fast and never trip an import-time failure on a half-provisioned deploy box.
* ``.env`` is loaded with ``python-dotenv`` (a core dependency), ``override=False``
  — mirroring the agent module's own behavior, so a key already exported in the
  shell wins over the file. We never print a key value.
* The agent streams its own reasoning to stdout (``go()`` calls ``pretty_print``,
  which prints each step). The REPL/one-shot add a clean framed final answer on
  top; ``--quiet`` suppresses the streamed steps and prints only the final answer.
* All CLI chrome is ASCII-only so it can't wedge a ``LANG=C`` / non-UTF-8 terminal
  on another device. ``readline`` (history/editing/completion) is best-effort and
  degrades to a plain ``input()`` when it isn't available (e.g. a bare Windows box).

This module intentionally does **not** modify the agent — it is a thin, additive
front door. Run ``python -m spatialomicsgym.chat_cli`` for the same behavior.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import time

from spatialomicsgym import platform_root, redaction
from spatialomicsgym.provider_names import OVERRIDABLE_ENV_SOURCES, canonical_source, source_from_model_prefix

_PROG = "stcoscientist"

# Mask anything key-shaped in an exception before it's printed — a provider/proxy 401 can echo an
# Authorization header or a full key. This used to be an inlined copy carrying a comment claiming it
# mirrored the webui's pattern; it did not — the JWT/bearer shape was missing here and present there,
# on the front door whose ``--json`` path writes to stdout (a machine-capture / CI-log surface).
# There is now one list, and every surface imports it.
_SECRET_RE = redaction.SECRET_RE


def _redact_err(exc: object) -> str:
    """``str(exc)`` with any credential masked, so an error message can't leak one to stdout/stderr.

    Same two-pass policy as the browser front door: the setup registry (exact, shape-independent)
    then the key-shape match.
    """
    return redaction.redact(exc)


# The model used when neither --model nor SOG_LLM / SOG_LLM_MODEL is set. Mirrors
# ``SpatialOmicsGymConfig.llm`` in spatialomicsgym/config.py — keep them in step.
_DEFAULT_MODEL = "azure-gpt-6-astra"

# provider -> the env var(s) that must hold a real key for that source. Mirrors
# ``get_llm()`` in ``spatialomicsgym/llm.py``. Ollama (local server), Bedrock
# (AWS credentials, checked softly) and Custom (needs a base_url, not a single
# key var) intentionally have no hard key requirement here.
_PROVIDER_KEY_ENV: dict[str, tuple[str, ...]] = {
    "OpenAI": ("OPENAI_API_KEY",),
    "AzureOpenAI": ("OPENAI_API_KEY",),
    "Anthropic": ("ANTHROPIC_API_KEY",),
    "Gemini": ("GEMINI_API_KEY",),
    "Groq": ("GROQ_API_KEY",),
    "Bedrock": (),
    "Ollama": (),
    "Custom": (),
}

# Common, real model names by provider — a discovery aid for ``--list-models``.
# These are EXAMPLES, not an exhaustive or guaranteed-available list: any model
# your chosen provider serves works. The trailing note in each entry is a hint.
_MODEL_PRESETS: dict[str, tuple[str, ...]] = {
    "Anthropic": ("claude-opus-4-8", "claude-sonnet-5", "claude-sonnet-4-5", "claude-haiku-4-5"),
    "OpenAI": ("gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-4.1"),
    "Gemini": ("gemini-2.0-flash", "gemini-1.5-pro"),
    # Groq ids are llama/gemma-like — indistinguishable from a local Ollama model by name,
    # so these carry a --source hint (copy them as `stcoscientist --source Groq -m ...`).
    "Groq": ("llama-3.3-70b-versatile  (pass --source Groq)", "gemma2-9b-it  (pass --source Groq)"),
    "AzureOpenAI": ("azure-gpt-6-astra", "azure-<your-deployment>  (deployment name lives in the endpoint URL)"),
    "Bedrock": ("anthropic.claude-3-5-sonnet-20241022-v2:0", "meta.llama3-70b-instruct-v1:0"),
    # No Qwen/DeepSeek recommendation (RL-1, D-028): --list-models advertised qwen2.5 as a pick
    # (u17-cli-report-20). Whether those names still ROUTE to Ollama is a separate open decision.
    "Ollama": ("llama3", "gpt-oss-20b  (local server, no API key)"),
    "Custom": ("<any model your --base-url endpoint serves>",),
}

# Slash commands offered in the REPL (also the Tab-completion vocabulary).
_REPL_COMMANDS = (
    "/help",
    "/model",
    "/source",
    "/config",
    "/tools",
    "/mcp",
    "/history",
    "/save",
    "/export",
    "/retry",
    "/reset",
    "/clear",
    "/version",
    "/exit",
    "/quit",
)

_BAR = "=" * 64
_RULE = "-" * 64


# --------------------------------------------------------------------------- #
# Small, dependency-light helpers
# --------------------------------------------------------------------------- #
def _version() -> str:
    try:
        from spatialomicsgym.version import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _install_root() -> str:
    """Directory of the ``spatialomicsgym`` package that is ACTUALLY running.

    A console script resolves through whichever python owns its shebang, so a stale
    checkout on that interpreter's sys.path runs silently in place of the one the user
    thinks they installed. Printing the resolved package root wherever the version is
    shown makes that mismatch visible instead of silent."""
    try:
        import spatialomicsgym

        return os.path.dirname(os.path.abspath(spatialomicsgym.__file__))
    except Exception:
        return "unknown"


def _repo_root() -> str:
    """Best-effort repository root: the checkout this package runs from (``layout.repo_root()``, the
    directory whose ``pyproject.toml`` names this project), else the current working directory. Used
    only to resolve the default MCP config path robustly when the CLI is launched from a
    subdirectory. The old walk also stopped at the first directory holding ``MCP_server/``, which
    since the re-layout is ``agent/`` -- one level short of ``install/recipes/``."""
    try:
        from spatialomicsgym import layout

        root = layout.repo_root()
    except Exception:
        root = None
    return str(root) if root is not None else os.getcwd()


def _looks_like_placeholder(value: str | None) -> bool:
    """True when ``value`` is empty / a template stub (``sk-ant-...``, ``<your-key>``,
    ``EMPTY``). Reuses the vetted setup helper when importable; falls back to a
    self-contained check so the CLI never hard-depends on the setup package."""
    try:
        from sog_install.llm_setup import value_looks_like_placeholder

        if value_looks_like_placeholder(value or ""):
            return True
    except Exception:
        pass
    v = (value or "").strip().strip("\"'").strip()
    if not v or v.upper() == "EMPTY":
        return True
    # Fallback only (the setup helper above is the rule). Same whole-word discipline: a random key
    # body can contain "xxxx" or "your" and is still a key.
    low = v.lower()
    if any(tok in low for tok in ("<", "...")):
        return True
    words = [w for w in re.split(r"[^a-z0-9]+", low) if w]
    return any(w in ("your", "changeme", "placeholder") or set(w) == {"x"} for w in words if len(w) >= 3)


def _detect_source(
    model: str, base_url: str | None, explicit: str | None, *, ignore_env_source: bool = False
) -> str | None:
    """Resolve the LLM provider the way ``get_llm()`` does, without importing the heavy LLM factory.
    Precedence, mirroring :func:`spatialomicsgym.llm.resolve_source`: explicit ``--source`` > a model
    name that *proves* a provider (only over an overridable env one) > ``SOG_SOURCE`` / ``LLM_SOURCE``
    env > the free-form model-name sniffs below > ``Custom`` when a base_url is set.

    ``ignore_env_source`` turns on that proven-prefix step; without it the env provider is consulted
    first, whatever the name proves. Every launch of the terminal front door passes it -- ``-m``, a
    bare launch reading ``SOG_LLM`` (hunt 2026-09-30, u17-cli-report-10), and ``/model`` -- as does
    the web model-switch, so none of them inherits a stale provider. It never mutates the
    process-global ``os.environ``, which would race a concurrent agent build reading
    ``LLM_SOURCE`` on another thread."""
    # Accept any capitalisation of a known provider name (``--source anthropic`` -> ``Anthropic``,
    # ``SOG_SOURCE=openai`` -> ``OpenAI``). An unknown explicit value is kept verbatim so the key
    # pre-flight can name it in its friendly "Unknown provider" hint.
    canon = {k.lower(): k for k in _PROVIDER_KEY_ENV}
    if explicit:
        return canon.get(explicit.lower(), explicit)

    # A named model displaces the configured provider only when the name PROVES a different one.
    # OVERRIDABLE_ENV_SOURCES documents which providers that holds for: OpenAI/Anthropic/Gemini
    # publish fixed catalogues, whereas an Azure deployment or an Ollama build carries whatever name
    # its owner chose -- so a foreign-looking name proves nothing about it and the env var decides.
    # Dropping the env for EVERY name instead let the free-form sniffs below answer where
    # resolve_source would have kept the env, and that answer is final: it is handed on as an
    # explicit ``source=`` kwarg, which resolve_source returns verbatim. An Azure user whose
    # deployment is called "gpt-4o" -- the commonest Azure convention -- got OpenAI.
    env_source = canonical_source(os.getenv("SOG_SOURCE") or os.getenv("LLM_SOURCE"))
    if ignore_env_source:
        proven = source_from_model_prefix(model)
        if proven is not None and (env_source is None or env_source in OVERRIDABLE_ENV_SOURCES):
            return proven
    if env_source is not None:
        return env_source

    m = (model or "").lower()
    if m.startswith("claude-"):
        return "Anthropic"
    if m.startswith("gpt-oss"):
        return "Ollama"
    if m.startswith(("gpt-", "chatgpt-")):
        return "OpenAI"
    # OpenAI's o-series reasoning models (o1, o3, o3-mini, o4-mini, ...) carry no "gpt" prefix and
    # would otherwise fall through to the Ollama catch-all / "unknown". Match an "o" + digit token
    # with a boundary so "ollama" / a hypothetical "o1lama" don't get mis-routed.
    if m[:1] == "o" and m[1:2].isdigit() and m[2:3] in ("", "-"):
        return "OpenAI"
    if m.startswith("azure-"):
        return "AzureOpenAI"
    if m.startswith("gemini-"):
        return "Gemini"
    if "groq" in m:
        return "Groq"
    if m.startswith(
        # ``meta.llama`` (no trailing hyphen): real ids are ``meta.llama3-*`` / ``meta.llama2-*``
        # with a version digit — the old ``meta.llama-`` never matched and fell through to the
        # Ollama catch-all below. ``eu.``/``apac.`` are the non-US cross-region profile prefixes.
        ("anthropic.claude-", "amazon.titan-", "meta.llama", "mistral.", "cohere.", "ai21.", "us.", "eu.", "apac.")
    ):
        return "Bedrock"
    if base_url:
        return "Custom"
    if "/" in m or any(n in m for n in ("llama", "mistral", "qwen", "gemma", "phi", "deepseek", "vicuna")):
        return "Ollama"
    return None  # unknown -> let the key pre-flight give a friendly hint


def _key_preflight(
    source: str | None, model: str, base_url: str | None, *, source_typed: bool = False
) -> tuple[bool, str]:
    """Return ``(ok, message)``. Never prints or returns a key value — only whether
    a usable one is present, plus a friendly hint pointing at ``.env`` / ``sog-setup``
    when it isn't. We deliberately do not ask the user to paste a key anywhere.

    ``source_typed`` says the user named ``source`` (``--source``); only then is a public-OpenAI
    route for an Azure key let through, the escape ``get_llm`` gives an explicit ``source=``."""
    if source is None:
        return (
            False,
            f"Could not tell which provider '{model}' belongs to.\n"
            "  Set the provider explicitly with --source (OpenAI/Anthropic/Gemini/Groq/AzureOpenAI/Bedrock/Ollama/Custom),\n"
            "  or pick a model whose name makes it obvious (e.g. claude-*, gpt-*, gemini-*).",
        )
    if source not in _PROVIDER_KEY_ENV:
        return (
            False,
            f"Unknown provider {source!r}. Use one of: {'/'.join(_PROVIDER_KEY_ENV)}\n"
            "  (check the spelling and capitalisation, e.g. 'Anthropic', 'OpenAI', 'Ollama').",
        )
    if source == "Ollama":
        return True, ""  # local server, no API key
    if source == "Custom":
        if not base_url:
            return (
                False,
                "A Custom LLM needs an endpoint. Pass --base-url http://host:port/v1 "
                "(or set SOG_CUSTOM_BASE_URL), plus --api-key if the server requires one.",
            )
        return True, ""
    if source == "Bedrock":
        # AWS credentials come from the standard chain (env / ~/.aws / IAM role);
        # we can't reliably probe them here, so warn softly rather than block.
        if not (os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE") or os.getenv("AWS_ROLE_ARN")):
            return True, "note: no obvious AWS credentials in the environment; Bedrock will use the default AWS chain."
        return True, ""

    if source == "Anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        # get_llm reads a key kept only in ~/.bash_profile (some deploy boxes); this check did not,
        # so the CLI and every portal turn refused what STCoscientist(llm="claude-...") accepted in
        # the same shell (hunt 2026-09-30, uL6-parity-14). Asked once per process, as get_llm asks.
        _load_profile_key_once()
    key_vars = _PROVIDER_KEY_ENV.get(source, ())
    present = [v for v in key_vars if not _looks_like_placeholder(os.getenv(v))]
    if key_vars and not present:
        want = " or ".join(key_vars)
        hint = (
            f"No usable {source} API key found (looked for {want}).\n"
            f"  Fix it without pasting the key here:\n"
            f"    * run the setup wizard:   sog-setup\n"
            f"    * or add it to .env:      echo '{key_vars[0]}=<your-real-key>' >> .env\n"
            f"    * or export it:           export {key_vars[0]}=<your-real-key>\n"
        )
        return False, hint

    if source == "OpenAI" and not source_typed and os.getenv("OPENAI_ENDPOINT"):
        # A provider derived from the model name reaches get_llm as an explicit source=, the lane its
        # Azure-key guard leaves open -- so `-m gpt-5` on an Azure box (and the portal's switch ping,
        # which shares this pre-flight) posted the Azure key to api.openai.com. Ask the guard here,
        # where the source is still known to be derived (hunt 2026-09-30, uL6-parity-1).
        try:
            from spatialomicsgym.llm import azure_key_to_openai_problem
        except Exception as exc:
            # Closed, not open: a guard that cannot be asked has not said yes (hunt 2026-09-30,
            # uL6-parity-1). get_llm cannot import either, so nothing that would have worked is lost.
            return False, (
                f"OPENAI_ENDPOINT is set, and the check that keeps an Azure key away from api.openai.com "
                f"could not run ({type(exc).__name__}: {exc}); refusing the public-OpenAI route. "
                "Name the provider with --source to choose it yourself."
            )
        problem = azure_key_to_openai_problem(model, source)
        if problem:
            return False, problem

    if source == "AzureOpenAI" and _looks_like_placeholder(os.getenv("OPENAI_ENDPOINT")):
        return True, "note: Azure needs OPENAI_ENDPOINT set to your resource URL; the call will fail without it."
    return True, ""


_PROFILE_KEY_ASKED: list[bool] = []


def _load_profile_key_once() -> None:
    """``llm._load_anthropic_key_from_profile`` at most once per process (it sources a shell profile)."""
    if _PROFILE_KEY_ASKED:
        return
    _PROFILE_KEY_ASKED.append(True)
    try:
        from spatialomicsgym.llm import _load_anthropic_key_from_profile

        _load_anthropic_key_from_profile()
    except Exception:
        pass


def _resolve_mcp_config(arg: str | None) -> str | None:
    """Resolve the MCP config to wire. ``arg`` is the value of ``--mcp``: an explicit
    path, or the sentinel ``"__default__"`` meaning "find the one this box was set up
    with". Returns an existing *file* path or ``None`` (caller warns).

    The ``__default__`` search mirrors the wizard's own precedence
    (``sog_install.conncheck._discover_config``) so the CLI **and** the web UI
    wire the exact config the setup wizard resolved for *this* machine and that the
    ``sog-setup conncheck`` diagnostic reports on:

        ``SOG_MCP_CONFIG`` (the pointer the wizard records in ``.env``)
        > the setup-generated ``install/recipes/mcp_config.setup.yaml``
        > the canonical ``agent/MCP_server/mcp_config.yaml``

    Each location is joined through ``platform_root.recipes_dir`` / ``platform_dir``, so a root laid
    out like a seeded home (``setup/``, ``MCP_server/`` at the top) is read in its own layout.

    Consulting only cwd-relative well-known paths (the old behaviour) ignored that
    recorded pointer, so a box whose setup config lives elsewhere — or a CLI launched
    from another directory — silently wired a stale/foreign config in which every tool
    points at an interpreter that does not exist on this host.

    Off a checkout (pip-only install) three more rungs follow, because ``_repo_root()``
    finds no checkout above ``site-packages`` and degrades to the cwd, leaving the well-known
    candidates above covering only the working directory: the writable instance root's
    setup config, its canonical copy, and finally the wheel's read-only ``_platform``
    canonical. The instance root is ``platform_root.instance_root()`` -- the SAME resolver
    the wizard writes through -- so a relocated root (``SOG_PLATFORM_ROOT``) is read from
    exactly where setup wrote it, not from a ``~/.spatialomicsgym`` setup never touched.
    Checkout runs never reach these rungs, so a stray seeded home cannot hijack a checkout
    session."""
    if not arg:
        return None
    if arg != "__default__":
        # Require a real *file*, not merely an existing path: os.path.exists is True for a
        # directory, so `--mcp somedir/` (a fat-finger) used to resolve to the directory and
        # then die downstream with a cryptic IsADirectoryError inside add_mcp's open(). Mirror
        # the __default__ branch's os.path.isfile so a non-file resolves to None and the caller
        # emits the friendly _mcp_resolve_miss_message ("is a directory" / "check the path").
        # `~` expanded first: `/mcp ~/my_setup.yaml` was checked as a literal './~/...' and reported
        # as "does not exist" beside a file that does (u17-cli-report-12).
        arg = os.path.expanduser(arg)
        return arg if os.path.isfile(arg) else None
    # Default: the recorded pointer first, then the setup-generated config, then the
    # canonical repo config. Try a few well-known locations for the latter two so the
    # CLI works from any cwd.
    root = _repo_root()
    candidates: list[str] = []
    pointer = os.environ.get("SOG_MCP_CONFIG")
    if pointer:
        candidates.append(os.path.expanduser(pointer.strip()))
    cwd = os.getcwd()
    candidates += [
        str(platform_root.recipes_dir(cwd) / "mcp_config.setup.yaml"),
        str(platform_root.recipes_dir(root) / "mcp_config.setup.yaml"),
        str(platform_root.platform_dir(cwd) / "MCP_server" / "mcp_config.yaml"),
        str(platform_root.platform_dir(root) / "MCP_server" / "mcp_config.yaml"),
    ]
    if not platform_root.running_from_checkout():
        instance = platform_root.instance_root()
        candidates += [
            str(platform_root.recipes_dir(instance) / "mcp_config.setup.yaml"),
            str(platform_root.platform_dir(instance) / "MCP_server" / "mcp_config.yaml"),
        ]
        packaged = platform_root.packaged_canonical_config()
        if packaged is not None:
            candidates.append(str(packaged))
    resolved: str | None = None
    for cand in candidates:
        try:
            if os.path.isfile(cand):
                resolved = cand
                break
        except OSError:
            continue
    # Skipping a dead pointer is deliberate (a session with partly-right tools beats no session),
    # but doing it in silence is not: the substitute is eventually the canonical repo config, whose
    # interpreters are somebody else's. Disclose here rather than at each caller so no front door —
    # launch, /mcp, or sog-web's build_handle — can be added later and forget to. stderr keeps the
    # --json stdout contract intact.
    note = _stale_mcp_pointer_note(resolved)
    if note:
        print(f"[mcp] {note}", file=sys.stderr)
    return resolved


def _stale_mcp_pointer_note(resolved: str | None) -> str | None:
    """Thin lazy shim onto ``sog_install.constants.stale_mcp_pointer_note`` (the shared sentence).

    Imported inside the function per the module Design notes. ``ImportError`` alone is swallowed —
    a trimmed install without the setup package must not fail to launch over a warning string —
    and nothing else is, so a real fault in the builder still surfaces instead of restoring the
    very silence this exists to remove."""
    try:
        from sog_install.constants import stale_mcp_pointer_note
    except ImportError:
        return None
    return stale_mcp_pointer_note(resolved)


def _mcp_resolve_miss_message(arg: str) -> str:
    """The situation-specific line when ``--mcp`` was asked for but resolved to nothing.

    ``arg`` is the raw ``--mcp`` value (post-demux): the sentinel ``"__default__"`` for a
    bare ``--mcp`` whose discovery came up empty, or an explicit path the user typed that
    doesn't resolve to a config file. Collapsing these into one generic "no config found"
    hides the fix — a bare-flag miss means *run sog-setup*; a named path that is a directory
    means *point at the .yaml itself*; any other named-path miss means *check your path*."""
    if arg == "__default__":
        pointer = os.environ.get("SOG_MCP_CONFIG")
        looked = "SOG_MCP_CONFIG, " if pointer else ""
        return (
            f"[mcp] no analysis-tool config found for this box (looked for {looked}"
            "install/recipes/mcp_config.setup.yaml, agent/MCP_server/mcp_config.yaml) - run sog-setup to "
            "enable tools; continuing without them."
        )
    if os.path.isdir(arg):
        # Exists but is a directory: the resolver (os.path.isfile) already declined it. Name the
        # real fix instead of the generic "does not exist", which would read as untrue — the path
        # is right, it is just a folder, so point --mcp at the .yaml file inside it.
        return (
            f"[mcp] the path you passed (--mcp {arg}) is a directory, not a config file - "
            "point --mcp at the .yaml file itself (continuing without tools)."
        )
    return f"[mcp] the config you passed (--mcp {arg}) does not exist - check the path (continuing without tools)."


def _looks_like_config_path(value: str) -> bool:
    """Heuristic: does ``value`` look like an MCP *config path* (vs a free-text question)?

    Used to disambiguate the ``--mcp`` optional value. A YAML extension, any path
    separator, or a ``~`` home prefix marks it as a (possibly mistyped) config path we
    honour as-is; anything else after a bare ``--mcp`` is almost certainly the user's
    question that argparse greedily swallowed."""
    s = value.strip()
    if not s:
        return False
    if s.lower().endswith((".yaml", ".yml")):
        return True
    # A value with INTERNAL whitespace is a free-text question, not a bare config path (R24-C2).
    # This must sit AFTER the .yaml check (so a legit spacey path like "my configs/mcp.yaml" still
    # resolves) and works because _demux already honours any value that exists on disk via
    # os.path.exists FIRST. Without this, a slash-bearing phrase the user typed after a bare --mcp
    # ("compare PI3K/AKT and mTOR tools") trips the os.sep branch below and gets swallowed as a
    # (missing) config path instead of being asked.
    if any(ch.isspace() for ch in s):
        return False
    if os.sep in s or (os.altsep and os.altsep in s):
        return True
    return s.startswith("~")


def _demux_mcp_and_question(args) -> None:
    """Repair the ``--mcp <question>`` collision in place (argparse ``nargs='?'``).

    ``--mcp`` takes an optional value so ``--mcp path.yaml`` works, but that lets a bare
    ``--mcp`` written *before* the positional question greedily consume it:
    ``stcoscientist --mcp "detect spatial domains"`` parses to ``question=[]``,
    ``mcp="detect spatial domains"`` — the question is silently swallowed, then treated
    as a missing config path, so the user's actual request is never asked.

    When ``--mcp``'s value is clearly NOT a config path (doesn't exist and isn't
    path-shaped) and no positional question was supplied, reinterpret it as the question
    and enable the tools with the default (install-aware) config. Every genuine form —
    ``--mcp`` alone, ``--mcp real.yaml``, ``--mcp typo.yaml``, and an explicit
    question — is left untouched."""
    val = args.mcp
    if not val or val == "__default__":
        return  # no value, or the bare-flag sentinel — nothing to disambiguate
    if args.question:
        return  # a real positional question was given ⇒ --mcp's value is a config arg
    if os.path.exists(val) or _looks_like_config_path(val):
        return  # a genuine (or mistyped) config path — honour it / let the caller warn
    # It's the swallowed question. Move it back and wire tools with the default config.
    args.question = [val]
    args.mcp = "__default__"


def _print_models() -> None:
    """Print common model names per provider (a discovery aid; needs no key)."""
    print("Common model names by provider (examples - any model your provider serves works):")
    print()
    for prov, models in _MODEL_PRESETS.items():
        env = _PROVIDER_KEY_ENV.get(prov, ())
        keyhint = f"[key: {' or '.join(env)}]" if env else "[no API key required]"
        print(f"  {prov}  {keyhint}")
        for m in models:
            mark = "   <- default" if m.split()[0] == _DEFAULT_MODEL else ""
            print(f"      {m}{mark}")
        print()
    print(f"default model: {_DEFAULT_MODEL}")
    print(f"pick one with:  {_PROG} -m <model>       (or set SOG_LLM=<model> in your environment/.env)")
    print(f"force a provider with:  {_PROG} --source <Provider>")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=_PROG,
        description="Chat with the ST-Coscientist spatial-omics agent from your terminal.",
        epilog=(
            "examples:\n"
            f"  {_PROG}                              start an interactive session\n"
            f'  {_PROG} "detect spatial domains"     ask one question and exit\n'
            f'  {_PROG} -q "..." > answer.txt        one-shot, answer only (pipe-friendly)\n'
            f'  {_PROG} --json "..."                 one-shot, JSON output for scripts\n'
            f"  {_PROG} --list-models                show model names you can pass to -m\n"
            f"  {_PROG} --mcp                        enable the analysis tools (needs sog-setup envs)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("question", nargs="*", help="a question to ask (omit for an interactive session)")
    p.add_argument("-m", "--model", metavar="NAME", help="LLM model (default: $SOG_LLM or the built-in default)")
    p.add_argument(
        "--source",
        metavar="PROVIDER",
        # Built from the table --source is validated against, and comma-separated on purpose:
        # a slash-joined list is one unbreakable "word" to textwrap, so at argparse's default
        # 80 columns it split as ".../Groq/Azure" + "OpenAI/Bedrock/..." — which reads as the
        # real product name "Azure OpenAI" and is then rejected as an unknown provider.
        help="force the provider: " + ", ".join(_PROVIDER_KEY_ENV),
    )
    p.add_argument(
        "-p",
        "--path",
        metavar="DIR",
        help="data directory for the agent (default: $SOG_PATH, else $SOG_DATA_PATH, else ./data)",
    )
    p.add_argument("--base-url", metavar="URL", help="base URL for a custom/self-hosted OpenAI-compatible endpoint")
    p.add_argument(
        "--api-key", metavar="KEY", help="API key for a custom endpoint (providers use their env var instead)"
    )
    p.add_argument("--temperature", type=float, metavar="T", help="sampling temperature")
    p.add_argument("--timeout", type=int, metavar="SEC", help="per-step code-execution timeout in seconds")
    p.add_argument("--commercial", action="store_true", help="commercial mode (exclude non-commercial datasets)")
    p.add_argument("--no-tool-retriever", action="store_true", help="skip AI tool/dataset pre-selection (faster start)")
    p.add_argument(
        "--mcp",
        nargs="?",
        const="__default__",
        default=None,
        metavar="CONFIG",
        # The order _resolve_mcp_config("__default__") reads; sog-web already printed it (hunt 2026-09-30,
        # uL6-parity-22, uL5-drift-17).
        help="wire the MCP analysis tools; optional path to a config (default: $SOG_MCP_CONFIG, else "
        "install/recipes/mcp_config.setup.yaml, else agent/MCP_server/mcp_config.yaml)",
    )
    p.add_argument(
        "--env-file",
        metavar="FILE",
        default=".env",
        help="extra dotenv file, read after the install's own .env and named when read (default: ./.env)",
    )
    p.add_argument(
        "-q", "--quiet", action="store_true", help="one-shot: print only the final answer (no streamed steps)"
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json",
        help='one-shot: print the answer as JSON {"answer": "..."} (implies quiet)',
    )
    p.add_argument("--list-models", action="store_true", help="list common model names per provider and exit")
    p.add_argument("--no-banner", action="store_true", help="suppress the REPL banner")
    p.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s (SpatialOmicsLab {_version()}) [package: {_install_root()}]",
    )
    return p


# --------------------------------------------------------------------------- #
# Agent construction (lazy — heavy import happens here, never at module load)
# --------------------------------------------------------------------------- #
def _make_agent_builder(args, model: str, source: str | None, mcp_path: str | None):
    """Return a zero-arg callable that constructs (and MCP-wires) a fresh agent.
    Used both for the first build and for the REPL ``/reset`` / ``/model`` commands."""

    def _build():
        from spatialomicsgym.agent import STCoscientist  # heavy: imported lazily

        # `temperature` is NOT a STCoscientist constructor argument — the agent sources it
        # from the global `default_config` singleton that `get_llm` reads. Apply the CLI
        # override there (after the lazy import so the singleton exists). Passing it as a
        # kwarg raised `TypeError: unexpected keyword argument 'temperature'` and crashed
        # every run that set --temperature.
        if getattr(args, "temperature", None) is not None:
            try:
                from spatialomicsgym.config import default_config

                default_config.temperature = args.temperature
            except Exception:
                os.environ["SOG_TEMPERATURE"] = repr(args.temperature)

        # An interactive session is a conversation: a follow-up ("redo it with 12 domains") must not
        # have to restate the input path. Without this the banner's "Context is kept across turns" is
        # false — a live probe had the agent search /tmp for output it had written two minutes before.
        # /reset rebuilds the agent, which is what clears the recap.
        kwargs: dict = {"llm": model, "conversation_memory": True}
        if source:
            kwargs["source"] = source
        if args.path:
            kwargs["path"] = args.path
        if args.base_url:
            kwargs["base_url"] = args.base_url
        if args.api_key:
            kwargs["api_key"] = args.api_key
        if args.timeout is not None:
            kwargs["timeout_seconds"] = args.timeout
        if args.commercial:
            kwargs["commercial_mode"] = True
        if args.no_tool_retriever:
            kwargs["use_tool_retriever"] = False
        agent = STCoscientist(**kwargs)
        if mcp_path:
            # Kept on the agent: the banner and /reset read it. A cancelled or failed wire was
            # dropped here and the status line said "on (145 tools ready)" regardless
            # (u17-cli-report-4).
            agent._sog_mcp_wired = _wire_mcp(agent, mcp_path)
        return agent

    return _build


def _server_will_wire(meta: object) -> bool:
    """True iff ``add_mcp`` would actually register a server — matching *both* of its gates,
    not just the first: the entry must be a mapping, not disabled by a *falsy* ``enabled``
    (mcp_integration.py:164), **and** carry a non-empty *list* ``command``
    (mcp_integration.py:168-171 skips an otherwise-enabled server whose command is missing or
    not a list). Counting on ``enabled`` alone overstates the linkage for a hand-edited or
    partially-written config, so the banner would promise tools the agent never wired."""
    if not isinstance(meta, dict):
        # add_mcp calls ``server_meta.get(...)``; a non-mapping value would raise there and
        # abort wiring entirely — so it wires nothing, and must not be counted as a live tool.
        return False
    if not meta.get("enabled", True):
        # Byte-exact mirror of add_mcp's own gate ``if not server_meta.get("enabled", True)``
        # (mcp_integration.py:164): it skips on *falsiness*, not identity. So ``enabled: 0`` /
        # ``null`` / ``""`` / ``[]`` (all falsy, all common in a hand-edited config) mean the
        # server is NOT wired — an earlier ``is False`` identity check counted those as linked,
        # overstating the banner. Absent key ⇒ default True ⇒ wired, exactly as add_mcp does.
        return False
    cmd = meta.get("command")
    return isinstance(cmd, list) and len(cmd) > 0


def _registered_mcp(agent) -> tuple[int, int] | None:
    """``(functions, servers)`` that ``add_mcp`` actually registered on ``agent``; None if it can't say.

    The config counts below answer "what does this file list"; this answers "what can the agent call".
    ``add_mcp`` skips a server whose interpreter or worker script is missing, and a config-side count
    reported those as linked and "will fail when called" when they were never wired at all
    (u17-cli-report-3). Read off ``_custom_tools``, which is where ``add_mcp`` files each wrapper.
    """
    tools = getattr(agent, "_custom_tools", None)
    if not isinstance(tools, dict):
        return None
    modules = [
        str(meta.get("module") or "")
        for meta in tools.values()
        if isinstance(meta, dict) and str(meta.get("module") or "").startswith("mcp_servers.")
    ]
    return len(modules), len(set(modules))


def _interpreter_present(command: object) -> bool:
    """Whether a server's launch interpreter resolves on *this* box — a byte-exact mirror of
    ``add_mcp``'s own gate (``_resolve_portal_interp`` + ``_interp_runnable``,
    mcp_integration.py), so this count never promises more (or fewer) tools than the agent wires.

    * An **absolute** interpreter path is stat-checked: a stale setup config — or one copied from
      another machine — pins a since-deleted **base** env's python, so the server is ``enabled``
      yet its portal can never start (``add_mcp`` registers the wrapper from the inline tool list,
      so the failure only surfaces when the agent finally calls the tool). The per-tool worker
      interpreter is a different variable entirely (``env[{PREFIX}_PYTHON]``), repaired on its own
      by ``base_mcp._resolve_worker_python``, and says nothing about whether the server wires.
    * A bare ``python``/``python3`` (the canonical repo config's style) is **always present**:
      ``add_mcp`` pins it to the agent's own interpreter (``sys.executable``), so a healthy config
      is never penalised by a false negative even on a box where the env isn't PATH-active.
    * Any *other* bare/relative name (a hand-edited or foreign ``Rscript`` etc.) must resolve via
      ``shutil.which`` — exactly what ``_interp_runnable`` requires — so a name that can't launch
      here is honestly excluded rather than optimistically counted."""
    if not (isinstance(command, list) and command):
        return False
    exe = command[0]
    if not isinstance(exe, str) or not exe:
        return False
    if os.path.isabs(exe):
        return os.path.exists(exe)
    if os.path.basename(exe) in ("python", "python3"):
        return True
    return shutil.which(exe) is not None


def _count_enabled_servers(config_path: str | None) -> int | None:
    """How many analysis-tool servers a config would actually wire — the ones ``add_mcp``
    both keeps (``enabled != false``) *and* can launch (a valid ``command``).

    Returns ``0`` for a config that reads fine but wires nothing (mirrors ``add_mcp``'s own
    ``if not mcp_servers: return`` — an absent, null, or empty ``mcp_servers`` section), and
    ``None`` only when the config can't be read or interpreted at all, so callers can tell an
    honest, actionable "0 tools linked (run sog-setup)" apart from a count-free "unknown"."""
    return _count_servers(config_path, runnable_only=False)


def _count_runnable_servers(config_path: str | None) -> int | None:
    """Of the wired servers, how many can actually launch here — their interpreter exists on
    this box. A registered-but-unrunnable tool fails only when the agent calls it, so surfacing
    this gap up front is the difference between an honest linkage and a promise the box can't
    keep. Returns None when the config can't be read (callers stay count-free)."""
    return _count_servers(config_path, runnable_only=True)


def _count_listed_servers(config_path: str | None) -> int | None:
    """Every server the config *names*, wired or not — the denominator for an honest wiring line.

    A setup-generated config disables whatever its run did not install or could no longer find,
    and the bare "N tools wired" cannot be told apart from a complete install — on one live box a
    leftover setup config wired 6 of 88 while 121 healthy tool envs sat unused, and nothing said
    so. Counting what is *listed* lets the wiring message disclose the gap. ``None`` mirrors the
    other counters (unreadable config → count-free)."""
    return _count_servers(config_path, runnable_only=False, listed_only=True)


def _count_tool_functions(
    config_path: str | None, *, runnable_only: bool = False, servers: dict | None = None
) -> int | None:
    """How many individual analysis **functions** the wired servers publish -- the number a
    biologist means by "how many tools do I have".

    ``_count_enabled_servers`` counts *portals*: one ``seurat`` entry, one ``squidpy`` entry. But
    ``add_mcp`` registers one callable per entry of each server's ``tools:`` list
    (mcp_integration.py, ``for tool_meta in tools_config``), so those two servers alone publish
    fourteen callables. On the shipped config that is 114 functions behind 88 servers, and a UI
    that says "88 tools" undercounts the toolbox by a quarter -- the same defect
    ``validate_project.py`` already had and fixed for its ``total_mcp_tools`` line.

    ``None`` (unknown) whenever the file alone cannot answer: an uninterpretable config (same
    ladder as every other counter), or a wired server that declares no ``tools:`` list at all --
    ``add_mcp`` then *discovers* that server's functions by launching it, so no honest number
    exists until it runs. Callers fall back to the server count rather than print a guess."""
    # ``servers`` lets a caller count an inventory it has already resolved, rather than a path.
    # The portal needs this: ``add_mcp`` merges ``mcp_config_user.yaml`` over the shipped config,
    # so the shipped file is a SUBSET of what gets wired, and a chip counting the file while the
    # catalogue counts the merge is two numbers for one question -- which is exactly what a live
    # box showed, "121" on the chip over a catalogue holding 122. Same spine either way: the
    # selection rule and the tri-state below are shared, so this cannot become a third counter.
    if servers is not None:
        if any(not isinstance(meta, dict) for meta in servers.values()):
            return None
        metas = [
            meta
            for meta in servers.values()
            if _server_will_wire(meta) and (not runnable_only or _interpreter_present(meta.get("command")))
        ]
    else:
        metas = _wired_server_metas(config_path, runnable_only=runnable_only)
    if metas is None:
        return None
    total = 0
    for meta in metas:
        declared = _declared_tool_functions(meta)
        if declared is None:
            return None
        total += declared
    return total


def _declared_tool_functions(meta: dict) -> int | None:
    """How many functions one server block declares, or ``None`` when the block does not say.

    Mirrors ``add_mcp``'s registration loop rather than a bare ``len()``: a non-mapping entry and
    an entry with no ``spatialomicsgym_name``/``name`` are both skipped there with a warning, so
    neither becomes a callable and neither may be counted as one. An absent/empty/non-list
    ``tools`` is the auto-discovery case -- unknowable statically, hence ``None``."""
    tools = meta.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    return sum(
        1 for entry in tools if isinstance(entry, dict) and (entry.get("spatialomicsgym_name") or entry.get("name"))
    )


def _count_servers(config_path: str | None, *, runnable_only: bool, listed_only: bool = False) -> int | None:
    metas = _wired_server_metas(config_path, runnable_only=runnable_only, listed_only=listed_only)
    return None if metas is None else len(metas)


_CONFIG_DOC_CACHE: dict[str, tuple[int, int, int, object]] = {}


def _read_config_doc(config_path: str) -> object:
    """The parsed MCP config, shared between the counters that all want the same file.

    Two measurements, both on this checkout's 330 KB ``mcp_config.yaml``, are the whole reason this
    exists. **One** ``yaml.safe_load`` of it costs 0.375s in the pure-Python parser -- and
    ``/api/config`` asks four counters for a number (``_count_enabled_servers``,
    ``_count_runnable_servers``, and ``_count_tool_functions`` twice), so the settings panel was
    paying for the same parse four times over: 1.506s measured end to end, against 0.001s for
    ``/api/providers`` and ``/api/keys``. That was the slowest thing behind the gear button.

    So: parse with libyaml when the wheel carries it (0.04s, a 9x cut, byte-identical output -- it
    is the same SafeLoader in C), and hold the document against ``(mtime_ns, size, inode)`` so the
    other three counters read it for free. The key is the *identity of the bytes*, not the path, so a
    config an operator edits under a running server -- or one ``sog-setup`` rewrites -- reparses on
    the next call rather than serving a stale count until restart. The inode is in the key beside
    the size and the timestamp because this repo writes configs by ``.partial`` + ``os.replace``,
    and a restored backup can arrive with its original mtime preserved; the inode is what changes
    in that case. Invalidation is the reason this is not ``functools.lru_cache``.

    **The document is shared, so callers must treat it as read-only.** Every reader here is a
    counter or a view that only ``.get``s; the modules that *write* a config (``sog_install.bundle``,
    ``sog_install.mcp_resolver``, ``agent.mcp_config_merger``) keep their own parse and must, because a
    writer mutating this would corrupt every later reader's answer.

    A write that lands between the ``stat`` and the read stores the new document under the old
    stamp, which the next call sees as a miss and reparses -- so the window costs one extra parse
    and can never wedge a stale answer in place.

    Raises whatever the read or parse raises -- the callers already degrade to a count-free
    ``None`` on any exception, and swallowing it here would turn "unreadable" into "empty".
    """
    import yaml

    st = os.stat(config_path)
    key = os.path.realpath(config_path)
    hit = _CONFIG_DOC_CACHE.get(key)
    if hit is not None and hit[:3] == (st.st_mtime_ns, st.st_size, st.st_ino):
        return hit[3]
    with open(config_path, encoding="utf-8") as fh:
        raw = fh.read()
    loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader
    doc = yaml.load(raw, Loader=loader)  # SafeLoader -- the C one when present, never the full loader
    # Bounded: one live config, not a growing map of every path a test ever passed.
    if len(_CONFIG_DOC_CACHE) > 8:
        _CONFIG_DOC_CACHE.clear()
    _CONFIG_DOC_CACHE[key] = (st.st_mtime_ns, st.st_size, st.st_ino, doc)
    return doc


def _wired_server_metas(
    config_path: str | None, *, runnable_only: bool, listed_only: bool = False
) -> list[dict] | None:
    """The server blocks a config would actually wire -- the one spine under every tool counter.

    Three-valued by design, and the tri-state is the whole point: a list (possibly empty) is a
    determinate answer, where ``[]`` means "reads fine, wires nothing"; ``None`` means the config
    cannot be interpreted at all, so callers stay count-free instead of printing a number they
    cannot stand behind."""
    if not config_path:
        return None
    try:
        # pyyaml is a core dep; guard anyway so display never crashes wiring
        cfg = _read_config_doc(config_path)
    except Exception:
        return None  # genuinely unreadable/unparseable → unknown, degrade to count-free
    if cfg is None:
        # Empty / whitespace / comment-only file: add_mcp does ``yaml.safe_load(...) or {}``
        # (mcp_integration.py:151), so a None document becomes ``{}`` → "No MCP servers found" →
        # wires *zero* cleanly (no crash). That is a determinate 0 (→ honest "0 tools linked, run
        # sog-setup"), not an unknown. A truncated/partial write lands here too, and "0, run
        # sog-setup" is still the right, more helpful message than a bare "on".
        return []
    if not isinstance(cfg, dict):
        # Non-None, non-mapping top level (a bare scalar or list document): the merger treats it as
        # empty (mcp_config_merger: ``loaded if isinstance(loaded, dict) else {}``), so add_mcp wires
        # nothing and raises nothing. What this file means is unknowable → None; _wire_mcp decides
        # on what the agent actually registered, not on this.
        return None
    servers = cfg.get("mcp_servers")
    if not servers:
        # add_mcp does ``cfg.get("mcp_servers", {})`` then ``if not mcp_servers: return`` — so an
        # absent, null, or empty ``mcp_servers`` all wire *zero* servers (with a "No MCP servers
        # found" warning), never crash. That is a determinate 0, not an unknown: mirror it so the
        # banner says the honest "0 tools linked (run sog-setup)" instead of a misleading "wired".
        return []
    if not isinstance(servers, dict):
        # Present but a truthy non-mapping (e.g. a populated list): the merger treats it as empty
        # ("non-mapping 'mcp_servers'; treating as empty"), so add_mcp wires nothing and raises
        # nothing. A count is meaningless here → None.
        return None
    if any(not isinstance(meta, dict) for meta in servers.values()):
        # A dict whose values are not all mappings: add_mcp skips each non-mapping value with a
        # warning and wires the rest (u14-mcp-wiring-9). Report unknown (None) rather than a count
        # that reads as a definite "N tools linked"; _wire_mcp counts what the agent registered.
        # (These three comments said add_mcp crashed and the wire "failed loudly"; it has not since
        # u14-mcp-wiring-9 and the merger's coercion, so the wire reported success with nothing
        # registered -- hunt 2026-09-30, u17-cli-report-extra-25.)
        return None
    return [
        meta
        for meta in servers.values()
        if listed_only or (_server_will_wire(meta) and (not runnable_only or _interpreter_present(meta.get("command"))))
    ]


def _tools_status(state) -> str:
    """One honest phrase describing the analysis-tool linkage for banners/status lines."""
    if not state.get("mcp_on"):
        return "off (use /mcp or --mcp to enable)"
    path = state.get("mcp_path")
    n = _count_enabled_servers(path)
    registered = _registered_mcp(state.get("agent"))
    if registered is not None and n:
        functions, servers = registered
        if functions == 0:
            return "on, but 0 tools were wired (re-run sog-setup to repair the tool envs)"
        if servers < n:
            return (
                f"on ({functions} tools across {servers} servers; {n - servers} listed server"
                f"{'s' if n - servers != 1 else ''} not wired here - re-run sog-setup to repair)"
            )
        return (
            f"on ({functions} tools ready across {servers} servers)"
            if functions > servers
            else f"on ({functions} tools)"
        )
    if n == 0:
        # Wiring ran but this config lists nothing this box can launch — say so plainly and
        # point at the fix, rather than an ambiguous "on (0 tools)" that reads like a bug.
        return "on, but 0 tools linked (run sog-setup to enable tools)"
    if n is None:
        return "on"
    runnable = _count_runnable_servers(path)
    if runnable is not None and runnable < n:
        # Linked but not runnable: the config pins interpreters missing on this box (a stale or
        # copied setup config). Honest split so the banner never promises tools that will fail.
        return f"on ({n} linked, {runnable} runnable here - re-run sog-setup to repair)"
    functions = _count_tool_functions(path)
    if functions is not None and functions > n:
        # Multi-tool portals: the shipped config wires 88 servers publishing 114 callables, and
        # "88 tools" quietly hides 26 of them. Name both numbers -- the count that answers "what
        # can it do" and the count that answers "what has to start up".
        return f"on ({functions} tools ready across {n} servers)"
    return f"on ({n} tool{'s' if n != 1 else ''})"


def _config_is_canonical(config_path: str | None) -> bool:
    """True iff ``config_path`` is a *shipped* ``MCP_server/mcp_config.yaml`` — the repo's copy
    (``agent/MCP_server/mcp_config.yaml``),
    or on a pip-only install the wheel's read-only ``_platform`` copy. Both are the *blanket*
    config whose servers are all ``enabled: true`` regardless of which tool envs this machine
    actually has. Reached only as the last fallback (no ``SOG_MCP_CONFIG`` pointer, no
    setup-generated config), it advertises tools that will fail at call time on a box that
    wasn't provisioned for them, so wiring it warrants an honest 'unverified here' note.

    The seeded SOG_HOME canonical deliberately does NOT match: once ``sog-setup``'s resolver has
    rewritten it for this machine it is verified wiring, not the build-frozen shipped copy."""
    if not config_path:
        return False
    try:
        real = os.path.realpath(config_path)
        canon = os.path.realpath(str(platform_root.platform_dir(_repo_root()) / "MCP_server" / "mcp_config.yaml"))
        if real == canon:
            return True
        packaged = platform_root.packaged_canonical_config()
        return packaged is not None and real == os.path.realpath(str(packaged))
    except Exception:
        return False


def _wire_mcp(agent, config_path: str) -> bool:
    """Attach MCP tools; tolerate failure (missing envs on a fresh box, etc.)."""
    try:
        import nest_asyncio  # noqa: F401  (add_mcp relies on it being importable)
    except Exception:
        print("[mcp] note: nest_asyncio not available; skipping tool wiring.", file=sys.stderr)
        return False
    try:
        agent.add_mcp(config_path)
    except KeyboardInterrupt:
        # add_mcp spawns MCP-server subprocesses and blocks on async tool discovery — genuinely
        # slow. Cancel just the wiring — matching /model and /reset — rather than let ^C escape
        # the slash-command dispatch (outside input()'s try/except) up to main() and kill the
        # session. Also guards the /model re-wire path (_cmd_switch_model calls _wire_mcp).
        print("\n[mcp] wiring cancelled.", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"[mcp] could not wire tools ({_redact_err(exc)}); continuing without them.", file=sys.stderr)
        return False
    # STCoscientist.add_mcp catches its own failure, prints a warning and returns normally, so a
    # normal return is not a wire: an unreadable config came back here as "[mcp] analysis tools
    # wired" and "tools: on" with nothing registered (u17-cli-report-2).
    failed = getattr(agent, "_mcp_wiring_error", None)
    if failed:
        print(f"[mcp] could not wire tools ({failed}); continuing without them.", file=sys.stderr)
        return False
    n = _count_enabled_servers(config_path)
    registered = _registered_mcp(agent)
    if registered is not None:
        # The agent can say what it registered, so that alone decides. The config-side count was
        # consulted first and only when it was a positive number: a config whose top level or
        # `mcp_servers` is not a mapping, or whose servers are not mappings, wires nothing without
        # raising (the merger coerces it, add_mcp skips it), _mcp_wiring_error stays None, and this
        # printed "[mcp] analysis tools wired" and left tools "on" with nothing callable (hunt
        # 2026-09-30, u17-cli-report-extra-25).
        functions, servers = registered
        if servers == 0:
            if n:
                why = (
                    f"none of its {n} listed server{'s' if n != 1 else ''} could be wired here - an interpreter "
                    "or worker script is missing on this box (stale or copied setup config); re-run sog-setup "
                    "to rebuild/repair those envs"
                )
            elif n == 0:
                why = "it lists no runnable tools for this box (all servers disabled or command-less); run sog-setup"
            else:
                why = "its top level or its mcp_servers is not a mapping of server blocks, so no server could be read"
            print(
                f"[mcp] no analysis tools were wired from {config_path} ({why}); continuing without them.",
                file=sys.stderr,
            )
            return False
        if n and servers < n:
            print(
                f"[mcp] warning: {n - servers} of {n} listed server{'s' if n != 1 else ''} were not wired "
                "(an interpreter or worker script is missing on this box - stale or copied setup config); "
                "re-run sog-setup to rebuild/repair those envs.",
                file=sys.stderr,
            )
        print(
            f"[mcp] {functions} analysis tool{'s' if functions != 1 else ''} across {servers} "
            f"server{'s' if servers != 1 else ''} wired from {config_path}",
            file=sys.stderr,
        )
        _canonical_config_note(config_path)
        return True
    if n == 0:
        # add_mcp succeeded but there is nothing runnable here (every server disabled or
        # command-less). Report it as an empty-but-successful wire so the caller can still say
        # "on, 0 tools" honestly instead of a misleading tool count.
        print(
            "[mcp] this config lists no runnable tools for this box (all servers disabled or "
            "command-less) - continuing without analysis tools; run sog-setup to enable them.",
            file=sys.stderr,
        )
    elif n is not None:
        # Count what the agent can *call*, not just what it has to *start*: one seurat server
        # publishes eight callables, so the server count alone under-reports the toolbox.
        functions = _count_tool_functions(config_path)
        wired = (
            f"{functions} analysis tools across {n} servers"
            if functions is not None and functions > n
            else f"{n} analysis tool{'s' if n != 1 else ''}"
        )
        print(f"[mcp] {wired} wired from {config_path}", file=sys.stderr)
        runnable = _count_runnable_servers(config_path)
        if runnable is not None and runnable < n:
            missing = n - runnable
            print(
                f"[mcp] warning: {missing} of {n} linked tool{'s' if n != 1 else ''} point at an "
                "interpreter missing on this box (stale or copied setup config) - they will fail when "
                "called; re-run sog-setup to rebuild/repair those envs.",
                file=sys.stderr,
            )
        listed = _count_listed_servers(config_path)
        if listed is not None and listed > n and not _config_is_canonical(config_path):
            # A setup-generated config narrows the toolbox to what its run installed — honest, but
            # only if said out loud: "6 wired" alone reads the same on a deliberate 6-tool install
            # and on a stale leftover shadowing 88 canonical tools (a live box hit exactly that).
            off = listed - n
            print(
                f"[mcp] note: this config lists {listed} servers but enables {n} - the other "
                f"{off} were left off by the sog-setup run that generated it; re-run sog-setup "
                "if a tool you need is missing.",
                file=sys.stderr,
            )
    else:
        print(f"[mcp] analysis tools wired from {config_path}", file=sys.stderr)
    _canonical_config_note(config_path)
    return True


def _canonical_config_note(config_path: str) -> None:
    """The blanket repo default: honest about the gap between "listed" and "runnable here"."""
    if _config_is_canonical(config_path):
        print(
            "[mcp] note: this is the repo's default tool list, not verified against this "
            "machine's installed envs - run sog-setup to confirm which tools can actually run "
            "here (or pass --mcp <your setup config>).",
            file=sys.stderr,
        )


# --------------------------------------------------------------------------- #
# One-shot
# --------------------------------------------------------------------------- #
# Final-answer display cleanup lives in ``spatialomicsgym/answer.py`` so the CLI, the web UI and
# library users (``from spatialomicsgym import clean_answer``) all share one implementation, and one
# set of regexes. Re-exported here under the original private names: ``sog_portal/server.py`` imports
# ``_clean_final_answer`` and ``_strip_agent_tags`` *through this module*, and the display-helper
# regression tests reach the other three the same way. F401 is expected -- these are deliberate
# re-exports for those importers, not names this module itself calls.
from spatialomicsgym.answer import (  # noqa: F401
    _LOOP_HEADER_RE,
    _extract_solution,
    _strip_agent_tags,
    _strip_leading_classification,
)
from spatialomicsgym.answer import clean_answer as _clean_final_answer


def _answer_text(answer) -> str:
    # No isinstance(str) guard. ``go()`` returns the raw ``message.content``, which is a str only on
    # a plain turn -- any turn carrying tool use (the normal shape after an MCP analysis step) comes
    # back as a LIST of content blocks. The old guard sent exactly those to ``str()``, printing a
    # Python repr with the answer buried inside it and the <solution> tags still attached, which is
    # the one case ``clean_answer(Any)`` was extended to handle. It flattens via ``answer_to_text``.
    #
    # Masked for the same reason ``sog_portal/server.py:865`` masks the same string, and the mask goes
    # here rather than in the cleaner: an answer is downstream of the stdout of code the agent
    # wrote and ran and of the stderr of the workers it launched, which is where a credential
    # surfaces. This is the single reader behind five surfaces -- the printed answer, ``--quiet``'s
    # stdout contract, ``--json``'s ``answer`` field, ``/history``, and the ``/export`` markdown
    # transcript, which is a file the user is invited to keep and send on. ``_run_one_shot`` shows
    # the gap without reference to the browser: its ``error`` key is built with ``_redact_err`` and
    # its ``answer`` key, four lines away, was not.
    #
    # Not inside ``clean_answer``: that is also the published corpus's cleaner
    # (``huggingface_data/build_dataset.py:245``), which redacts separately and deliberately. Clean
    # first, then mask, in that order and byte-for-byte the web's -- the cleaner drops whole
    # regions rather than truncating, so it cannot sever a token out of the matcher's reach.
    return redaction.redact(_clean_final_answer(answer))


#: Figures named under a CLI answer. The strip is a pointer to the report, not the report.
_MAX_CLI_FIGURES = 4


def _present(target) -> bool:
    """Is this contained path a file we could actually open? Never raises.

    ``safe_subpath`` has already said the path belongs to the run; this is the second half of the
    question, and the half the terminal card was skipping. An unreadable path counts as "not there",
    the way it does everywhere else in the report layer: describing a result must not cost it.
    """
    try:
        return target is not None and target.is_file()
    except OSError:
        return False


def _run_card_lines(agent, since: float) -> list[str]:
    """ASCII summary of the analysis run this turn produced, or ``[]`` -- never raises.

    The web chat attaches the same run to its answer bubble and renders the figures inline. A
    terminal cannot render a PNG, so the honest equivalent is to say what was produced, warn about
    it, and hand over one clickable ``file://`` link to ``report.html`` -- the page that already
    shows every figure of the run. Naming a directory and stopping there is the behaviour this
    replaces: it is the difference between the agent producing a result and the user seeing one.

    Discovery, identity and the freshness rule come from ``report.discover``, the same source the
    portal and the web chat use, so the three front doors cannot disagree about which runs exist.
    """
    try:
        from spatialomicsgym.paths import results_search_roots
        from spatialomicsgym.report import manifest as _manifest
        from spatialomicsgym.report import redact_text
        from spatialomicsgym.report.discover import MAX_DECLARED_TO_STAT, resolve_run, runs_since

        roots = results_search_roots(getattr(agent, "path", None))
        fresh = runs_since(roots, since, limit=1)
        if not fresh:
            return []
        run = fresh[0]
        # ``resolve_run``, not ``contain``: ``dir`` is relative to the run's OWN root, and a legal
        # relative path is legal under every root, so ``contain`` would hand back whichever root
        # comes first. The ``root_key`` the run list carries is what disambiguates them.
        found = resolve_run(run.get("dir"), roots, run.get("root_key"))
        if found is None:
            return []
        run_root, run_dir = found
        m = _manifest.load(run_dir)
    except Exception:
        # Describing the result must never cost the result: the answer is already printed.
        return []

    # Every string below is agent-influenced, so it goes through the same redactor the web card
    # uses. One implementation, two front doors: a token that must not be echoed on a shared page
    # must not be echoed in a pasted terminal log either.
    review = m.get("review") or {}
    verdict = review.get("verdict") if isinstance(review, dict) else None
    head = " | ".join(
        redact_text(part)
        for part in (m.get("tool_name") or run.get("name") or "", m.get("task_type") or "", m.get("status") or "")
        if part
    )
    lines = [_RULE, f"  results: {head}" if head else "  results", f"  {run_dir}"]
    if verdict:
        lines.append(f"  review: {redact_text(verdict)}")

    # Contain, then ``is_file`` -- the same two steps the web card takes before it offers a figure,
    # and the same two ``discover._n_present`` counts with. This line names files and nothing else:
    # a terminal cannot draw a PNG, so the name is the whole of what the CLI reader gets, and a
    # declaration whose plotting step raised after the manifest was written -- or whose ``figures/``
    # directory did not survive a copy -- sent them to open a file that is not there.
    #
    # The shortfall is stated rather than quietly filtered away. That a run *meant* to draw three
    # plots is what makes "none of them are here" worth reading, which is why ``discover`` publishes
    # ``n_figures`` beside ``n_figures_present`` instead of just the survivors.
    declared = m.get("figures") or []
    # Bounded for the reason ``_n_present`` is: ``MAX_MANIFEST_BYTES`` caps the manifest's bytes and
    # not its entry count, and this runs inside the turn. Past the cap presence is simply not
    # checked -- "we did not finish counting" is reported as silence, never as absence.
    checked = len(declared) <= MAX_DECLARED_TO_STAT
    if checked:
        figures = [f for f in declared if _present(_manifest.safe_subpath(run_dir, f.get("path")))]
        missing = len(declared) - len(figures)
    else:
        # Nothing checked, so nothing named. Filtering on ``safe_subpath`` alone -- which is what
        # this branch used to do -- is containment without existence, the very thing the checked
        # branch above exists to stop, restored for exactly the runs with the most declarations.
        # It also bought nothing: ``safe_subpath`` resolves the root once per entry and is ~95% of
        # the price of the checked branch, so walking 100,000 declarations to print four names cost
        # what verifying them would have. The count is still reported below; the names are not.
        figures, missing = [], 0
    if figures:
        shown = ", ".join(redact_text(str(f.get("path"))) for f in figures[:_MAX_CLI_FIGURES])
        more = f" (+{len(figures) - _MAX_CLI_FIGURES} more)" if len(figures) > _MAX_CLI_FIGURES else ""
        lines.append(f"  figures: {shown}{more}")
    elif declared and not checked:
        # Saying nothing here would read as "this run drew no figures", which is the opposite of
        # what a manifest declaring tens of thousands of them says.
        lines.append(f"  figures: {len(declared)} declared, too many to verify - not checked on disk")
    if missing:
        lines.append(f"  ! {missing} of {len(declared)} declared figure(s) are not in the run directory")
    tables = m.get("tables") or []
    if tables:
        lines.append(f"  tables:  {len(tables)}")
    # What the tool drew for itself, which the manifest never mentions. Counted, not listed: the
    # terminal cannot show a PNG, and the point is that a run reported as ``failed`` with zero
    # figures may still have plots worth opening -- so the reader is told they exist and where.
    try:
        from spatialomicsgym.report.discover import tool_figure_dir, tool_figures

        _shown, n_tool, capped = tool_figures(run_dir, run_root)
        counted = f"{n_tool}+" if capped else str(n_tool)
        source = tool_figure_dir(run_dir, run_root)
    except Exception:
        n_tool, capped, counted, source = 0, False, "0", None
    if n_tool and source is not None:
        lines.append(f"  the tool also wrote {counted} image(s) of its own in {source}")
    elif capped and source is not None:
        # Zero *found* is not zero *there*: the scan can stop on the entry budget before reaching
        # any image, and printing nothing here reads as "the tool drew none" -- the opposite of the
        # thing this block exists to tell the reader of a failed run.
        lines.append(f"  the tool's own output directory was too large to scan: {source}")
    # Warnings are why this block exists at all. A signal-free or transposed result is a
    # plausible-looking file that the answer text will happily describe as a success.
    for warning in (m.get("warnings") or [])[:3]:
        lines.append(f"  ! {redact_text(warning)}")

    report = run_dir / "report.html"
    # Encoded and quoted: a path with a space or '#' was cut at the space by the terminal's linkifier
    # and split into two arguments when pasted, and a bare `python` may not be this interpreter
    # (u17-cli-report-23).
    if report.is_file():
        try:
            link = report.resolve().as_uri()
        except (OSError, ValueError):
            link = f"file://{report}"
        lines.append(f"  open: {link}")
    else:
        lines.append(f"  open: {shlex.quote(sys.executable)} -m spatialomicsgym.report {shlex.quote(str(run_dir))}")
    return lines


def _print_run_card(agent, since: float) -> None:
    for line in _run_card_lines(agent, since):
        print(line)


def _print_answer(answer) -> None:
    print("\n" + _RULE)
    print(_answer_text(answer))
    print()


@contextlib.contextmanager
def _stdout_diverted_to_stderr():
    """Hold file descriptor 1 clear for the duration, forwarding what it catches to stderr.

    ``contextlib.redirect_stdout`` is not enough here, and a live ``--json`` run proved it:
    the output began with ``TIMEOUT: Code execution timed out`` and an ``Ai Message`` banner
    before the JSON object, so no caller could parse it.

    Two paths write past a rebound ``sys.stdout``:

    * ``agent/mcp_integration.py`` restores the real stream around every MCP subprocess
      launch (asyncio needs a stream with a working ``fileno()``). That save/restore is
      correct in itself, but ``sys.stdout`` is *process-global* while the need is
      *per-thread* — and the agent runs generated code in a worker thread
      (``utils/execution.py``). While that thread sits inside a long tool call with the real
      stdout installed, everything the main thread prints goes to the terminal.
    * MCP worker subprocesses inherit fd 1 outright and never consult ``sys.stdout``.

    Both ultimately write through the descriptor, so the descriptor is where we intercept.
    A temporary file rather than a pipe: agent chatter is unbounded and a full pipe with
    no reader would deadlock the run.

    Captured output is forwarded to stderr rather than dropped — a twenty-minute run still
    has to be able to tell the user why.

    Both layers are needed, and neither subsumes the other. The Python-level redirect
    catches ordinary ``print()`` calls — including under a harness like pytest's ``capsys``,
    whose ``sys.stdout`` has no connection to fd 1 at all. The descriptor-level redirect
    catches what escapes it. Dropping either one lets chatter through.
    """
    pybuf = io.StringIO()
    try:
        saved_fd = os.dup(1)
    except OSError:
        # No usable fd 1 (closed, or an exotic harness). The Python-level redirect alone is
        # weaker, but better than failing on a setup that otherwise works.
        with contextlib.redirect_stdout(pybuf):
            yield
        _forward_to_stderr(pybuf.getvalue())
        return

    spool = tempfile.TemporaryFile()
    try:
        for stream in (sys.stdout, sys.__stdout__):
            with contextlib.suppress(Exception):
                stream.flush()
        os.dup2(spool.fileno(), 1)
        try:
            with contextlib.redirect_stdout(pybuf):
                yield
        finally:
            # Flush before restoring, or buffered chatter lands on the real stdout after
            # the JSON object and breaks the very contract we are defending.
            for stream in (sys.stdout, sys.__stdout__):
                with contextlib.suppress(Exception):
                    stream.flush()
            os.dup2(saved_fd, 1)
    finally:
        with contextlib.suppress(Exception):
            os.close(saved_fd)
        with contextlib.suppress(Exception):
            spool.seek(0)
            shutil.copyfileobj(spool, sys.stderr.buffer)
            sys.stderr.flush()
        spool.close()
        _forward_to_stderr(pybuf.getvalue())


def _forward_to_stderr(text: str) -> None:
    if text:
        with contextlib.suppress(Exception):
            sys.stderr.write(text)
            sys.stderr.flush()


def _emit_stdout(text: str, stream=None) -> None:
    """Write one line to the caller's stdout, not to whatever the agent left behind.

    ``stream`` is the ``sys.stdout`` captured on entry, *before* the agent got a chance to
    rebind it (``mcp_integration`` sets it to ``sys.__stdout__``). Holding that reference is
    what makes the payload land where the caller is actually reading — including an
    in-process harness whose stdout is a Python-level object, which a raw ``os.write(1, ...)``
    would sail straight past.
    """
    stream = stream if stream is not None else sys.stdout
    try:
        stream.write(text + "\n")
        stream.flush()
    except Exception:
        # Closed or broken stream — last resort, straight at the descriptor.
        with contextlib.suppress(OSError):
            os.write(1, (text + "\n").encode("utf-8", "replace"))


#: A turn the agent could not finish. Distinct from 1 (``go()`` raised), 2 (nothing to run),
#: 3 (provider pre-flight) and 130 (interrupted), because it is the one non-zero status where
#: stdout still carries real output: the answer is there, it is just cut short.
_EXIT_DEGRADED = 4


def _degrade_note(agent) -> str:
    """The note for a turn the agent stopped early, or ``""`` if it finished.

    ``getattr``, not attribute access: this function is also handed third-party and older agent
    objects (anything with a ``go()``), and the attribute is only guaranteed on ours.

    Masked here, at the CLI's only reader of the note, exactly as ``sog_portal/server.py:857`` masks it
    at the web's. Two reasons beyond the obvious one that the note quotes a provider's error body:

    * the note's own scrubber (``stcoscientist._KEYISH_RE``) is a narrower list than
      :data:`spatialomicsgym.redaction.SECRET_RE` -- it knows ``sk-``, ``ghp_`` and ``Bearer`` and
      not the Groq, Google, AWS, Slack, JWT or hex-digest shapes -- so on this path the shared list
      is what closes the difference;
    * :func:`_is_whole_answer` compares this note against :func:`_answer_text`'s output, and that
      one now masks. Both sides must go through the same masking or a note carrying a shape only
      the shared list knows would stop matching the answer it *is*, and a turn that produced
      nothing would be reported as a turn that produced an answer plus a warning.
    """
    return redaction.redact(str(getattr(agent, "last_turn_degraded", "") or "").strip())


def _is_whole_answer(text: str, note: str) -> bool:
    """Did the turn stop before producing anything, leaving the note as the entire answer?

    ``go()`` returns the note itself when no message was ever emitted, and the partial message
    alone when one was, so removing the note is what separates the two shapes. This compares the
    note the agent recorded against the string it returned -- the same object, not a phrase match.
    """
    return not text.replace(note, "").strip()


def _run_one_shot(build_agent, question: str, quiet: bool, json_out: bool = False) -> int:
    # Grab the caller's stdout before the agent can rebind it; every payload below is
    # written to this reference rather than to whatever ``sys.stdout`` ends up being.
    out = sys.stdout
    if json_out:
        # Structured output for scripts: suppress ALL agent chatter, emit one JSON
        # object on stdout. Errors also come back as JSON so a caller can parse them.
        try:
            with _stdout_diverted_to_stderr():
                agent = build_agent()
                _log, answer = agent.go(question)
        except KeyboardInterrupt:
            _emit_stdout(json.dumps({"error": "interrupted"}), out)
            return 130
        except Exception as exc:
            _emit_stdout(json.dumps({"error": _redact_err(exc)}), out)
            return 1
        text = _answer_text(answer)
        note = _degrade_note(agent)
        if note and _is_whole_answer(text, note):
            # There is no answer -- publishing the note under ``answer`` would be a false
            # statement about what the object holds, which is how this went unnoticed live.
            _emit_stdout(json.dumps({"error": note}), out)
            return _EXIT_DEGRADED
        if note:
            _emit_stdout(json.dumps({"answer": text, "degraded": note}), out)
            return _EXIT_DEGRADED
        _emit_stdout(json.dumps({"answer": text}), out)
        return 0

    if quiet:
        # Suppress everything the agent prints (config block + streamed steps) so
        # stdout carries only the final answer — pipe-friendly. Build + run inside
        # the redirect; restore stdout before printing the answer.
        try:
            with _stdout_diverted_to_stderr():
                agent = build_agent()
                _log, answer = agent.go(question)
        except KeyboardInterrupt:
            print("interrupted.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"error: {_redact_err(exc)}", file=sys.stderr)
            return 1
        text = _answer_text(answer)
        _emit_stdout(text, out)
        note = _degrade_note(agent)
        if note:
            # ``-q``'s stdout is a contract with a pipe, so the warning goes beside it. Skip the
            # echo when the note IS the answer -- it just went to stdout a line ago.
            if not _is_whole_answer(text, note):
                print(note.strip(), file=sys.stderr)
            return _EXIT_DEGRADED
        return 0

    try:
        agent = build_agent()
        started_at = time.time()
        _log, answer = agent.go(question)
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nerror: {_redact_err(exc)}", file=sys.stderr)
        return 1
    _print_answer(answer)
    # Only this branch. ``--quiet`` and ``--json`` are contracts with a *pipe*: their stdout is the
    # answer text and one JSON object respectively, and a results block would corrupt both.
    _print_run_card(agent, started_at)
    note = _degrade_note(agent)
    if note:
        # Shape 2 drops the note from the returned answer, so a reader of this terminal would
        # otherwise see a truncated answer presented exactly like a complete one.
        if not _is_whole_answer(_answer_text(answer), note):
            print(note.strip(), file=sys.stderr)
        return _EXIT_DEGRADED
    return 0


# --------------------------------------------------------------------------- #
# REPL — readline (history / editing / completion) + rich slash commands
# --------------------------------------------------------------------------- #
def _save_history(readline_mod, path: str) -> None:
    try:
        readline_mod.write_history_file(path)
    except Exception:
        pass


def _make_completer(readline_mod):
    def _complete(text, state):
        if not text.startswith("/"):
            return None
        opts = [c + " " for c in _REPL_COMMANDS if c.startswith(text)]
        return opts[state] if state < len(opts) else None

    return _complete


def _init_readline() -> None:
    """Best-effort line editing, persistent history, and Tab-completion of slash
    commands. Silently no-ops if ``readline`` is unavailable (a bare Windows console
    lacks it) — the REPL still works, just without these conveniences. The history
    file is ``$SOG_CLI_HISTORY`` or ``~/.stcoscientist_history``."""
    try:
        import atexit
        import readline
    except Exception:
        return
    hist = os.environ.get("SOG_CLI_HISTORY") or os.path.expanduser("~/.stcoscientist_history")
    try:
        readline.read_history_file(hist)
    except Exception:
        pass  # first run / unreadable — start empty
    try:
        readline.set_history_length(1000)
        atexit.register(_save_history, readline, hist)
    except Exception:
        pass
    try:
        readline.set_completer(_make_completer(readline))
        # The default word-break set contains ``/``, so readline would hand the completer only
        # the text *after* the slash (``mod`` for ``/mod``) — which the ``startswith("/")``
        # completer then rejects, making Tab-completion silently dead. Restrict the break set to
        # whitespace so the whole ``/command`` token reaches the completer.
        readline.set_completer_delims(" \t\n")
        # macOS ships a libedit-backed readline that ignores GNU "tab: complete" syntax and
        # needs the editline bind form, or Tab-completion is silently dead there.
        if "libedit" in (getattr(readline, "__doc__", "") or ""):
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
    except Exception:
        pass


def _discover_tool_names(agent) -> list[str]:
    """Best-effort list of the agent's tool names. Returns ``[]`` when the agent
    exposes nothing enumerable (the CLI must not assume the agent's internals).

    ``_custom_tools`` first: it is where ``add_mcp`` files every wrapper, keyed by name. None of the
    other names exists on STCoscientist, so with 145 tools wired /tools said the agent exposes no
    names (u17-cli-report-19); the others stay for agents that do spell it that way."""
    for attr in ("_custom_tools", "mcp_tools", "tools", "custom_tools", "tool_registry"):
        val = getattr(agent, attr, None)
        if val is None:
            continue
        try:
            if isinstance(val, dict):
                return sorted(str(k) for k in val)
            if isinstance(val, (list, tuple, set)):
                out = []
                for t in val:
                    nm = getattr(t, "name", None) or getattr(t, "__name__", None) or str(t)
                    out.append(str(nm))
                return sorted(out)
        except Exception:
            continue
    return []


_REPL_HELP = r"""commands:
  /help                 show this help
  /model [NAME]         show the active model, or switch to NAME
  /source               show the active provider
  /config               show the full active configuration
  /tools                list the wired analysis tools (if the agent exposes them)
  /mcp  [CONFIG]        wire the MCP analysis tools now (optional config path)
  /history              show this session's questions and answers
  /save   [FILE]        save the conversation as a PDF (default: stcoscientist_conversation.pdf)
  /export [FILE]        save a markdown transcript (default: stcoscientist_transcript.md)
  /retry                ask the last question again
  /reset, /clear        start a fresh conversation (clears context + history)
  /version              show the version
  /exit,  /quit         leave
tips: end a line with a backslash to continue it on the next line; Up/Down recall prompts; Tab completes /commands.
Anything else is sent to the agent as a question."""


def _read_multiline(first: str) -> str | None:
    """Given the first typed line, gather backslash-continued lines into one prompt.
    Returns the joined text, or ``None`` if the continuation was Ctrl-C cancelled."""
    if not first.endswith("\\"):
        return first
    parts = [first[:-1]]
    while True:
        try:
            cont = input("...> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n(^C - input cancelled)")
            return None
        except UnicodeDecodeError:
            # The main prompt guards this (a LANG=C / non-UTF-8 terminal can feed an
            # undecodable byte); the continuation prompt must too, or a mid-multiline bad
            # byte crashes the whole REPL with a raw traceback. Cancel this continuation.
            print("(couldn't decode that input - is your terminal set to UTF-8?)", file=sys.stderr)
            return None
        if cont.endswith("\\"):
            parts.append(cont[:-1])
        else:
            parts.append(cont)
            break
    return "\n".join(parts)


def _repl(
    build_agent,
    model: str,
    source: str | None,
    mcp_on: bool,
    show_banner: bool,
    *,
    args=None,
    switch=None,
    mcp_path: str | None = None,
) -> int:
    """Interactive multi-turn session. ``switch`` (optional) is a callable
    ``new_model -> (build_callable, new_source) | None`` enabling ``/model NAME``;
    ``args`` (optional) is the parsed namespace, used only by ``/config``.

    ``mcp_path`` is the config the launch build wired (if any). We track the
    *runtime-effective* path in ``state['mcp_path']`` so a later ``/mcp`` and every
    ``/model`` switch keep the tool linkage — and the reported status — honest."""
    launch_mcp_path = mcp_path  # immutable: what --mcp wired; /reset reverts to this
    try:
        agent = build_agent()
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        # Masked like every other CLI error path: a construction error can quote a credentialed
        # Custom base URL or a key (hunt 2026-09-30, uL6-parity-21).
        print(f"error: could not start the agent: {_redact_err(exc)}", file=sys.stderr)
        return 1

    # readline conveniences only when we're truly interactive (keeps tests and piped
    # runs from touching a history file in the user's home directory).
    if _stdin_isatty():
        _init_readline()

    state = {
        "agent": agent,
        "model": model,
        "source": source,
        # What the launch build actually wired, when it says (u17-cli-report-4): a cancelled or
        # failed wire left "tools: on" in the banner.
        "mcp_on": bool(getattr(agent, "_sog_mcp_wired", mcp_on)),
        "mcp_path": launch_mcp_path,  # runtime-effective config; updated by /mcp, honoured on switch
        "history": [],
        "last_attempt": None,
        # The builder of the model this session is ON: /model replaces it, /reset calls it. /reset
        # used to call the launch builder, so after `/model gpt-4o` a reset quietly went back to
        # the launch model while /model and /config still said gpt-4o (u17-cli-report-5).
        "build": build_agent,
        "build_wires_mcp": True,  # the launch builder wires --mcp itself; a /model builder does not
    }

    if show_banner:
        tools = _tools_status(state)
        print(_BAR)
        print("  SpatialOmicsLab  -  ST-Coscientist  (interactive session)")
        print(f"  model: {model}    provider: {source or 'auto'}    tools: {tools}")
        print(f"  package: {_install_root()}")
        print(_BAR)
        print("  Ask a spatial-omics question in plain English. Context is kept")
        print("  across turns. Type /help for commands, /exit to quit.")
        print(_BAR)

    def _ask(question: str) -> None:
        """Run one agent turn; record it on success; survive interrupts/errors."""
        state["last_attempt"] = question  # /retry re-runs the last *attempt*, even if it errored
        started_at = time.time()  # before the turn: the card shows what THIS turn produced
        try:
            _log, answer = state["agent"].go(question)
        except KeyboardInterrupt:
            print("\n(^C - turn cancelled)")
            return
        except Exception as exc:
            print(f"\n[error] {_redact_err(exc)}", file=sys.stderr)
            return
        # A turn cut short (a provider error after part of an answer) returns the partial answer,
        # and the REPL printed it and filed it as finished; only one-shot said so (u17-cli-report-6).
        note = _degrade_note(state["agent"])
        cut_short = bool(note) and not _is_whole_answer(_answer_text(answer), note)
        # The note goes into the record too, as a third element: /history and /export listed a
        # cut-short answer as if it were complete -- the warning reached stderr once and nothing
        # kept it (hunt 2026-09-30, u17-cli-report-extra-26). "" for a turn that finished.
        state["history"].append((question, answer, note if cut_short else ""))
        # The run card walks up to MAX_DIRS directories; a ^C there was outside the turn's try and
        # ended the whole session with exit 130 (u17-cli-report-8). The turn is already recorded.
        try:
            _print_answer(answer)
            if cut_short:
                print(note.strip(), file=sys.stderr)
            _print_run_card(state["agent"], started_at)
        except KeyboardInterrupt:
            print("\n(^C - skipped the results summary; the answer is kept)")

    while True:
        try:
            raw = input("\nsog> ")
        except EOFError:
            print("\nbye.")
            return 0
        except KeyboardInterrupt:
            print("\n(^C - type /exit to quit)")
            continue
        except UnicodeDecodeError:
            # Output streams are hardened (_harden_streams); the input side can still meet an
            # undecodable byte on a LANG=C / non-UTF-8 terminal. Skip that line rather than
            # crash the whole REPL with a traceback — the module claims non-UTF-8 resilience.
            print("(couldn't decode that input - is your terminal set to UTF-8?)", file=sys.stderr)
            continue

        # Multi-line continuation only for plain prompts, never for /commands.
        if raw.endswith("\\") and not raw.strip().startswith("/"):
            joined = _read_multiline(raw)
            if joined is None:
                continue
            line = joined.strip()
        else:
            line = raw.strip()

        if not line:
            continue

        # Users habituated to python / psql / mysql type a bare `exit` or `quit` (no slash).
        # Honour it instead of spending an agent turn (and API cost) on the literal word.
        if line.lower() in ("exit", "quit"):
            print("bye.  (tip: /exit also works)")
            return 0

        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            cmd = cmd.lower()
            rest = rest.strip()

            if cmd in ("/exit", "/quit", "/q"):
                print("bye.")
                return 0
            if cmd in ("/help", "/h", "/?"):
                print(_REPL_HELP)
                continue
            if cmd == "/version":
                print(f"{_PROG} (SpatialOmicsLab {_version()})")
                print(f"  package: {_install_root()}")
                continue
            if cmd == "/model":
                if rest:
                    _cmd_switch_model(state, rest, switch)
                else:
                    print(
                        f"model: {state['model']}   provider: {state['source'] or 'auto'}   "
                        f"tools: {_tools_status(state)}"
                    )
                continue
            if cmd == "/source":
                if rest:
                    # /source is display-only — the provider is fixed at launch and follows the
                    # model. A user typing `/source OpenAI` (by analogy with /model NAME) expects a
                    # switch; tell them how to actually change it rather than silently ignoring rest.
                    print(
                        f"[source] the provider is fixed for this session (currently "
                        f"{state['source'] or 'auto'}). To change it, relaunch with --source, or "
                        f"use /model with a name that implies the provider (e.g. claude-*, gpt-*)."
                    )
                else:
                    print(f"provider: {state['source'] or 'auto'}   model: {state['model']}")
                continue
            if cmd == "/config":
                _cmd_config(state, args)
                continue
            if cmd == "/tools":
                _cmd_tools(state)
                continue
            if cmd == "/history":
                _cmd_history(state["history"])
                continue
            if cmd == "/save":
                _cmd_save(state["agent"], rest)
                continue
            if cmd == "/export":
                _cmd_export(state["history"], rest)
                continue
            if cmd == "/retry":
                # Prefer the last *attempted* prompt (which may have errored) over the last
                # recorded success — retrying after a failure is exactly when /retry is used.
                last_q = state.get("last_attempt") or (state["history"][-1][0] if state["history"] else None)
                if not last_q:
                    print("(nothing to retry yet - ask a question first)")
                    continue
                print(f"[retry] {last_q}")
                _ask(last_q)
                continue
            if cmd == "/mcp":
                arg = rest or "__default__"
                path = _resolve_mcp_config(arg)
                if not path:
                    # Same situation-aware voice as launch-time --mcp (run sog-setup vs. check
                    # the path) so a miss reads identically whichever entry point hit it.
                    print(_mcp_resolve_miss_message(arg), file=sys.stderr)
                elif _wire_mcp(state["agent"], path):
                    # Remember the runtime-effective config so a later /model switch (which
                    # rebuilds the agent) re-attaches exactly these tools, not the launch set.
                    state["mcp_on"] = True
                    state["mcp_path"] = path
                continue
            if cmd in ("/reset", "/clear"):
                try:
                    fresh = state["build"]()
                    # Back to the LAUNCH tool config either way: the launch builder wires it, and a
                    # /model builder builds unwired, so it is attached here. The runtime-effective
                    # path (a prior /mcp) is reverted so the reported status matches the agent.
                    wired = bool(launch_mcp_path) and bool(getattr(fresh, "_sog_mcp_wired", True))
                    if launch_mcp_path and not state["build_wires_mcp"]:
                        wired = _wire_mcp(fresh, launch_mcp_path)
                    state["agent"] = fresh
                    state["history"].clear()
                    state["last_attempt"] = None  # else /retry re-runs a pre-reset question
                    state["mcp_path"] = launch_mcp_path
                    state["mcp_on"] = wired
                    print(f"[reset] fresh conversation started; tools: {_tools_status(state)}.")
                except KeyboardInterrupt:
                    # A rebuild (heavy import + construct) can be slow; ^C should cancel just
                    # the reset — matching how a normal turn handles ^C — not kill the CLI.
                    print("\n[reset] cancelled.")
                except Exception as exc:
                    print(f"[reset] failed: {_redact_err(exc)}", file=sys.stderr)
                continue
            print(f"unknown command {cmd!r}; type /help.")
            continue

        # A normal question -> one agent turn (multi-turn memory via fixed thread id).
        _ask(line)


def _cmd_switch_model(state, new_model: str, switch) -> None:
    if switch is None:
        print("model switching isn't available in this session; relaunch with -m NAME.")
        return
    res = switch(new_model)
    if not res:
        print(f"[model] cannot switch to {new_model!r} (see the note above).", file=sys.stderr)
        return
    new_build, new_source = res
    try:
        state["agent"] = new_build()  # built UNWIRED (see _switch): we re-attach tools below
    except KeyboardInterrupt:
        # A rebuild (heavy import + construct + MCP wiring) can be slow; ^C should cancel just the
        # switch — matching /reset (1063) and a normal turn (_ask) — not propagate out of the REPL
        # dispatch (which sits OUTSIDE the input() try/except) up to main() and kill the whole
        # session. state["agent"]/model/source are still the old, consistent values here (they are
        # only reassigned below, after this build succeeds), so the session is fully preserved.
        print("\n[model] switch cancelled.")
        return
    except Exception as exc:
        print(f"[model] failed to switch: {_redact_err(exc)}", file=sys.stderr)
        return
    # Re-attach the runtime-effective analysis tools onto the fresh agent so a model switch
    # never silently drops them (nor leaves the status claiming tools that aren't wired).
    if state.get("mcp_path"):
        state["mcp_on"] = _wire_mcp(state["agent"], state["mcp_path"])
    else:
        state["mcp_on"] = False
    state["model"], state["source"] = new_model, new_source
    state["build"], state["build_wires_mcp"] = new_build, False
    state["history"].clear()
    state["last_attempt"] = None  # a switched session shouldn't /retry the old model's question
    print(
        f"[model] switched to {new_model} (provider: {new_source or 'auto'}); "
        f"context cleared; tools: {_tools_status(state)}."
    )


def _effective(state, config, name: str):
    """A setting as this session holds it: off the built agent, else off the config it builds from.

    A CLI flag is only one of the two lanes. ``_make_agent_builder`` passes a kwarg **only when
    the flag is present**, and ``STCoscientist.__init__`` then fills every unspecified parameter
    from ``default_config`` — which ``config.__post_init__`` populated from the ``SOG_*``
    environment a ``.env`` or ``sog-setup`` wrote. The agent is therefore where both lanes have
    already been resolved into one answer.
    """
    value = getattr(state.get("agent"), name, None)
    return getattr(config, name) if value is None else value


def _reported_temperature(state, config) -> str:
    """The temperature this session's requests actually carry.

    Neither lane is the honest source here: ``get_llm`` overrides both for whole model families
    — newest-Claude and the OpenAI o-series have ``temperature`` omitted from the request
    entirely, and gpt-5 has it pinned to 1 whatever was configured. The built client already
    holds the outcome of every one of those rules, so read it there rather than keeping a second
    copy of the table that can drift from the first. ``None`` means nothing is sent at all, which
    is the one case "(model default)" was always right about.
    """
    client = getattr(state.get("agent"), "llm", None)
    if client is None:  # nothing built yet — report what the next build will start from
        return str(config.temperature)
    temperature = getattr(client, "temperature", None)
    return "(model default)" if temperature is None else str(temperature)


def _cmd_config(state, args) -> None:
    print("active configuration:")
    print(f"  model         : {state['model']}")
    print(f"  provider      : {state['source'] or 'auto'}")
    print(f"  tools (mcp)   : {_tools_status(state)}")
    if state.get("mcp_path"):
        print(f"  tools config  : {state['mcp_path']}")
    if args is not None:
        from spatialomicsgym.config import default_config  # lazy: see the module Design notes

        # Every line names the value this session is configured with, whichever lane supplied it.
        # Reading ``args`` alone made the whole env lane invisible — with a .env and no flags, all
        # five of these were wrong, and two of them are acted on: ``commercial`` contradicted the
        # constructor's own "Commercial mode" banner, and ``tool-retriever`` claimed a pre-selection
        # step that was not running. ``data path`` stays the user-facing base directory (the agent's
        # own ``.path`` is the derived ``<path>/spatialomicsgym_data`` lake), and reading it off the
        # config rather than ``os.getenv`` picks up the ``SOG_DATA_PATH`` alias too.
        print(f"  data path     : {args.path or default_config.path}")
        print(f"  temperature   : {_reported_temperature(state, default_config)}")
        print(f"  step timeout  : {_effective(state, default_config, 'timeout_seconds')} s")
        print(f"  commercial    : {_effective(state, default_config, 'commercial_mode')}")
        print(f"  tool-retriever: {_effective(state, default_config, 'use_tool_retriever')}")
    print(f"  turns so far  : {len(state['history'])}")


def _cmd_tools(state) -> None:
    names = _discover_tool_names(state["agent"])
    if names:
        print(f"tools available to the agent ({len(names)}):")
        for n in names[:60]:
            print(f"  {n}")
        if len(names) > 60:
            print(f"  ... and {len(names) - 60} more")
        return
    where = "on" if state["mcp_on"] else "off (use /mcp, or relaunch with --mcp)"
    print(f"analysis tools: {where}")
    print("(the agent doesn't expose per-tool names here; MCP servers launch on demand at run time)")


def _history_turn(entry) -> tuple[str, object, str]:
    """``(question, answer, cut_short_note)`` from a history record; a 2-tuple has no note."""
    return entry[0], entry[1], (entry[2] if len(entry) > 2 else "") or ""


def _cmd_history(history) -> None:
    if not history:
        print("(no turns yet)")
        return
    for i, entry in enumerate(history, 1):
        q, a, cut = _history_turn(entry)
        one_line = _answer_text(a).replace("\n", " ")
        print(f"{i}. Q: {q}")
        print(f"   A: {one_line[:100]}{'...' if len(one_line) > 100 else ''}")
        if cut:
            print(f"   (cut short: {' '.join(cut.split())})")


def _pdf_output_path(target: str) -> str:
    """Mirror ``STCoscientist.save_conversation_history``'s naming so the CLI echoes the
    file that's actually written: an existing ``.pdf`` is kept, a ``.md`` is swapped for
    ``.pdf`` (not doubled to ``.md.pdf``), anything else gets ``.pdf`` appended."""
    if target.endswith(".pdf"):
        return target
    base = target[:-3] if target.endswith(".md") else target
    return base + ".pdf"


def _cmd_save(agent, rest: str) -> None:
    target = os.path.expanduser(rest) if rest else "stcoscientist_conversation.pdf"
    out = _pdf_output_path(target)
    # A file left by an earlier session is not this save (u17-cli-report-18): success is claimed
    # only when the file is new or changed.
    before = _file_stamp(out)
    try:
        agent.save_conversation_history(target)
    except KeyboardInterrupt:
        # The PDF render runs under a 60s timeout (save_conversation_history) and is the slowest
        # thing a user can trigger in the REPL, so it's the likeliest ^C target. Cancel just the
        # save — matching /model (_cmd_switch_model) and /reset — rather than let the interrupt
        # escape the slash-command dispatch (which sits OUTSIDE input()'s try/except) up to main()
        # and kill the whole session.
        print("\n[save] cancelled.")
        return
    except Exception as exc:
        print(f"[save] could not write PDF ({_redact_err(exc)}).", file=sys.stderr)
        return
    # save_conversation_history returns nothing and can fail *silently* (a reportlab
    # hiccup, or it writes somewhere unexpected). Only claim success if the file actually
    # landed on disk, so the CLI never says "wrote X" when X isn't there.
    after = _file_stamp(out)
    if after is not None and after != before:
        print(f"[save] wrote {out}")
    else:
        # The PDF goes through weasyprint, markdown2pdf or pandoc (+ a LaTeX engine); reportlab,
        # which this hint used to name, is not one of them.
        print(
            f"[save] no PDF was written to {out} (the conversion needs weasyprint, markdown2pdf, or "
            "pandoc with a LaTeX engine; use /export for a dependency-free markdown transcript).",
            file=sys.stderr,
        )


def _file_stamp(path: str) -> tuple[int, int] | None:
    """``(mtime_ns, size)`` of ``path``, or None when it is not there."""
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    return st.st_mtime_ns, st.st_size


def _cmd_export(history, rest: str) -> None:
    """Write a plain-markdown transcript of the session. Robust by design — no
    reportlab / heavy deps (unlike ``/save``'s PDF), so it works on any deploy box."""
    if not history:
        print("[export] nothing to export yet (ask a question first).")
        return
    target = os.path.expanduser(rest) if rest else "stcoscientist_transcript.md"
    lines = ["# ST-Coscientist session transcript", ""]
    for i, entry in enumerate(history, 1):
        q, a, cut = _history_turn(entry)
        lines += [f"## {i}. {q}", "", _answer_text(a), ""]
        if cut:
            lines += [f"(cut short: {cut})", ""]
    try:
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"[export] wrote {target} ({len(history)} turn(s)).")
    except Exception as exc:  # OSError, or ValueError on an embedded-NUL path, etc.
        print(f"[export] could not write {target} ({_redact_err(exc)}).", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _stdin_isatty() -> bool:
    """``sys.stdin.isatty()`` that never raises. On a console-less host (Windows
    ``pythonw``, some service managers) ``sys.stdin`` can be ``None``; treat that as
    "not a terminal" rather than crashing with an ``AttributeError``."""
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _harden_streams() -> None:
    """Best-effort: let stdout/stderr tolerate characters the terminal can't encode
    instead of crashing. The agent's answers carry biomedical text (µm, °C, Greek
    letters, en-dashes) we cannot pre-sanitise, so on a ``LANG=C`` /
    ``PYTHONIOENCODING=ascii`` box we degrade the odd unrepresentable character to a
    backslash escape rather than raise ``UnicodeEncodeError``. Mirrors the setup
    package's progress/PromptIO stream hardening; a no-op when a stream can't
    reconfigure (e.g. it is an ``io.StringIO`` under test)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except Exception:
            pass


def install_env_file() -> os.PathLike[str] | None:
    """The install's own ``.env`` as a ``Path``, or ``None`` when there is no such file. Never raises.

    On a checkout it is the one sog-setup writes (``constants.dotenv_path()``: the repo's own, or
    ``SOG_SETUP_DOTENV``); off a checkout the writable instance root's, the same resolver the wizard
    writes keys through. Never the working directory: the front doors read a cwd ``.env`` only as
    their ``--env-file`` default, and the Python API not at all. One answer for :func:`_load_env_files`
    and for ``agent/stcoscientist.py``'s import-time load, which read ``./.env`` from wherever the
    importing process stood (hunt 2026-09-30, u16-llm-config-21).
    """
    try:
        if platform_root.running_from_checkout():
            # On a checkout the install's .env is the one sog-setup writes (`constants.dotenv_path`,
            # the repo's own). Only ./.env was read, so `stcoscientist` from a project directory
            # exited "no key -- run sog-setup" after sog-setup had written one (u17-cli-report-13).
            # Still not the seeded home, which a checkout never consults.
            from sog_install import constants

            path = constants.dotenv_path()
        else:
            path = platform_root.instance_root() / ".env"
        return path if path.is_file() else None
    except Exception:
        return None


def _load_env_files(env_file: str | None) -> None:
    """Load dotenv files early, never overriding exported shell variables, never raising.

    The install's own ``.env`` (:func:`install_env_file`) loads FIRST, then ``env_file`` (the
    ``--env-file`` flag, default ``./.env``), so precedence is shell > install ``.env`` > that
    extra file (``override=False`` means first writer wins). The order used to be the other way
    round, and a ``.env`` in whatever directory the user stood in -- a downloaded dataset folder --
    outranked the install's: its ``OPENAI_ENDPOINT`` took the real key from the install ``.env`` to
    its own host, and its ``SOG_MCP_CONFIG`` chose the commands ``--mcp`` spawns (hunt 2026-09-30,
    uL6-parity-3, the MED-12 threat the agent import was fixed for). An extra file still fills keys
    the install leaves unset, and is named on stderr with its absolute path whenever it is read, so
    it can never act silently; the install ``.env`` itself (a checkout's ``./.env``) is not named.

    Off a checkout the install ``.env`` is the writable instance root's --
    ``platform_root.instance_root()``, the same resolver the wizard writes keys and the
    ``SOG_MCP_CONFIG`` pointer through. Checkout runs never consult it, so a stray seeded home
    cannot inject variables into a checkout session. Shared by both front doors (terminal +
    browser) like ``_harden_streams``, so they agree on where keys come from. Never echo a value.

    ``SOG_SKIP_DOTENV`` (any non-empty value, as the agent import reads it) skips the install
    ``.env`` and the ``./.env`` default; an ``env_file`` named explicitly is still read.
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        _register_env_secrets()
        return  # dotenv is a core dep; if it's somehow missing, env vars still work

    # ``SOG_SKIP_DOTENV`` turns off the IMPLICIT loads -- the install's .env and the ``./.env``
    # default -- as it does for the agent import and the setup ping. This loader never read it, so
    # ``python -m spatialomicsgym.report`` and ``stcoscientist`` called in-process (a test, from the
    # repository root) put the developer's provider keys into ``os.environ`` for the rest of the
    # process (hunt 2026-09-30, u39-test-infra-5). A file the caller NAMED is a request, not a default,
    # and is still read.
    skip_implicit = bool(os.environ.get("SOG_SKIP_DOTENV"))
    # Each in its own try, so a broken install .env cannot stop a named file, nor the reverse.
    install_env = None
    try:
        install_env = None if skip_implicit else install_env_file()
        if install_env is not None:
            load_dotenv(str(install_env), override=False)
    except Exception:
        pass
    if env_file:
        env_file = os.path.expanduser(env_file)
    if skip_implicit and env_file == ".env":
        env_file = None
    if env_file and os.path.isfile(env_file):
        try:
            same = install_env is not None and os.path.samefile(env_file, install_env)
        except OSError:
            same = False
        if not same:
            try:
                load_dotenv(env_file, override=False)
                print(
                    f"[env] also read {os.path.abspath(env_file)} -- for keys the install's own .env "
                    "and the shell leave unset",
                    file=sys.stderr,
                )
            except Exception:
                pass
    elif env_file and env_file != ".env":
        # Named on the command line and not there. Said once, because the pre-flight that follows
        # tells the user to "add it to .env" -- the file they just named (u17-cli-report-11).
        print(f"warning: --env-file {env_file} is not a file; nothing was loaded from it.", file=sys.stderr)
    from spatialomicsgym import mirror_legacy_env

    mirror_legacy_env()  # a BIOMNI_* key that lives only in a .env (u16-llm-config-22)
    _register_env_secrets()


def _register_env_secrets() -> None:
    """Enter every secret-named value in the environment into the redaction registry. Never raises.

    MED-4 (still open until 2026-09-30, hunt u17-cli-report-14): only the setup paths registered
    secrets, so a key loaded here -- above all one with no vendor prefix, like AWS_SECRET_ACCESS_KEY,
    which no shape pattern can recognise -- reached the --json answer, the /export transcript and the
    portal's step frames unmasked when the model printed os.environ. Name-gated like the setup
    registry (session_log._looks_secretish); register_secret ignores values too short to be keys.
    """
    try:
        from sog_install import session_log

        for name, value in os.environ.items():
            if value and session_log._looks_secretish(name):
                session_log.register_secret(value)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``stcoscientist`` / ``sog-chat`` console scripts and for
    ``python -m spatialomicsgym.chat_cli``. Harden the output streams first, then run;
    any stray Ctrl-C that slips past the inner handlers becomes a clean exit 130 so
    both invocation styles behave identically."""
    _harden_streams()
    try:
        return _run(argv)
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130


def _numeric_flag_problem(args) -> str | None:
    """Why ``--timeout`` / ``--temperature`` cannot be used, or None. The ranges ``config`` applies.

    The knob bounds reached the env and the web settings but not these flags: ``--timeout
    99999999999999999`` overflowed ``thread.join`` on every code step, ``--timeout 0`` timed every
    step out at once, and ``--temperature nan`` was sent to the provider (u17-cli-report-21).
    """
    import math

    from spatialomicsgym.config import NUMERIC_RANGES

    for flag, key in (("timeout", "timeout"), ("temperature", "temperature")):
        value = getattr(args, flag, None)
        if value is None:
            continue
        low, high = NUMERIC_RANGES[key]
        if not math.isfinite(float(value)) or not (low <= value <= high):
            return f"--{flag} {value!r} is outside {low:g}..{high:g}"
    return None


def _run(argv: list[str] | None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Discovery flags need neither .env nor a key — handle before anything else.
    if args.list_models:
        _print_models()
        return 0

    # `~` in a path flag is expanded once, here. `--path=~/data` reached the agent verbatim (bash
    # does not expand `~` after `=`) and made a literal ./~/data tree in the cwd (u17-cli-report-12).
    for name in ("path", "env_file"):
        value = getattr(args, name, None)
        if isinstance(value, str) and value.startswith("~"):
            setattr(args, name, os.path.expanduser(value))

    # Load .env early (see _load_env_files: shell > the install's own .env, install_env_file -- the
    # one the agent module loads at import -- > --env-file, default ./.env). Never echo a value.
    _load_env_files(args.env_file)

    # After the .env load, not before: the check imports `config`, whose default_config reads the
    # environment once, at import -- imported first, every SOG_* in .env would be ignored.
    problem = _numeric_flag_problem(args)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2

    # The data root made absolute ONCE, here, as sog-web's AgentHandle does. Cells run in this
    # process and may chdir, and every /reset or /model rebuild re-resolved a relative --path (or
    # the ./data default) against the new cwd -- an empty data lake (hunt 2026-09-30, uL6-parity-24).
    if not args.path:
        from spatialomicsgym.config import default_config

        args.path = default_config.path
    if args.path:
        args.path = os.path.abspath(os.path.expanduser(args.path))

    # Resolve model + provider without importing the heavy agent.
    model = args.model or os.getenv("SOG_LLM") or os.getenv("SOG_LLM_MODEL") or _DEFAULT_MODEL
    base_url = args.base_url or os.getenv("SOG_CUSTOM_BASE_URL")
    # A model name that PROVES its provider beats a leftover overridable SOG_SOURCE/LLM_SOURCE, on
    # a bare launch too. _detect_source consults the env ahead of the model prefix unless told
    # otherwise, so a stale LLM_SOURCE=Anthropic (sog-setup writes one, this repo's own .env ships
    # one, and switching models never clears it) made `stcoscientist -m gpt-4o` resolve Anthropic --
    # and that answer is not a guess we later revise: _make_agent_builder passes it through as
    # kwargs["source"], which get_llm honours verbatim. Only `-m` was fixed, so a bare launch with
    # SOG_LLM=gpt-4o under SOG_SOURCE=Anthropic still built ChatAnthropic('gpt-4o') while
    # STCoscientist() and resolve_source answered OpenAI (hunt 2026-09-30, u17-cli-report-10).
    # The free-form providers are not touched: OVERRIDABLE_ENV_SOURCES leaves AzureOpenAI/Bedrock/
    # Ollama/Custom/Groq in charge, so an Azure deployment called "gpt-4o" stays Azure, and a name
    # that proves nothing still defers to the env var.
    source = _detect_source(model, base_url, args.source, ignore_env_source=True)

    ok, message = _key_preflight(source, model, base_url, source_typed=bool(args.source))
    if message:
        print(message, file=sys.stderr)
    if not ok:
        if args.json:
            # --json promises machine-readable output even on failure, but the pre-flight runs
            # before mode dispatch — so a caller doing json.loads(stdout) on the most common
            # first-run failure (no key / unknown provider) would otherwise get empty stdout.
            # Emit a one-line JSON error on stdout too (the human hint still went to stderr).
            print(json.dumps({"error": message.splitlines()[0] if message else "provider pre-flight failed"}))
        return 3

    _demux_mcp_and_question(args)  # repair `--mcp "<question>"` before resolving either
    mcp_path = _resolve_mcp_config(args.mcp)
    if args.mcp and not mcp_path:
        # Two very different situations wear the same "no config" outcome — name each so the
        # fix is obvious: a bare `--mcp` that discovered nothing (this box was never set up)
        # vs. an explicit `--mcp <path>` the user named that doesn't exist (almost always a typo).
        print(_mcp_resolve_miss_message(args.mcp), file=sys.stderr)

    build_agent = _make_agent_builder(args, model, source, mcp_path)

    # Decide mode. A question on the command line, or piped stdin, => one-shot.
    question = " ".join(args.question).strip() if args.question else ""
    if not question and not _stdin_isatty():
        try:
            piped = sys.stdin.read().strip()
        except KeyboardInterrupt:
            # Keep the --json contract uniform: every --json exit emits parseable JSON on stdout
            # (mirrors the one-shot KI path at _run_one_shot), so a caller's json.loads(stdout)
            # never meets an empty buffer. The human hint stays on stderr.
            if args.json:
                print(json.dumps({"error": "interrupted"}))
            print("\ninterrupted.", file=sys.stderr)
            return 130
        except Exception:
            piped = ""  # no readable stdin (console-less host) -> treat as none
        if piped:
            question = piped

    if question:
        return _run_one_shot(build_agent, question, quiet=args.quiet, json_out=args.json)

    if args.json:
        # --json is a machine-readable ONE-SHOT contract: the interactive REPL can neither be
        # driven by a non-interactive caller nor stream a single JSON answer, so with no question
        # there is nothing to emit. Refuse on a TTY as well as a pipe — a PTY caller (`docker run
        # -it`, `ssh -t`, pexpect) would otherwise fall through to _repl below, print the human
        # banner to stdout, and BLOCK on input(); json.loads(stdout) would then hang + never parse.
        # Emit a parseable one-line error on stdout (the human hint goes to stderr), mirroring the
        # pre-flight --json branch. Subsumes the old no-TTY-only guard so both cases are covered.
        print(f'no question given. Try: {_PROG} --json "your question"', file=sys.stderr)
        print(json.dumps({"error": "no question given"}))
        return 2

    if not _stdin_isatty():
        # No question, no TTY, nothing piped -> nothing to do.
        print(f'no question given and no interactive terminal. Try: {_PROG} "your question"', file=sys.stderr)
        return 2

    # /model NAME switching: keep the launch --source when one was given (Groq/Custom model
    # names are llama-like and can't be re-detected reliably), otherwise re-detect from the
    # new model name. Re-run the key pre-flight and hand back a fresh builder (or None if the
    # resolved provider has no usable key). Warn on an apparent provider mismatch so a
    # `/model gpt-4o` after `--source Anthropic` doesn't silently POST to the wrong endpoint.
    def _switch(new_model: str):
        # `/model gpt-4o` always names a model explicitly, so the env provider never outranks it
        # (ignore_env_source=True, unconditionally -- the same rule sog-web's settings panel uses).
        # Without it a switch inherited the *previous* provider's stale SOG_SOURCE/LLM_SOURCE and
        # kept posting the new model to the old endpoint.
        new_source = _detect_source(new_model, base_url, args.source, ignore_env_source=True)
        if args.source:
            implied = _detect_source(new_model, base_url, None, ignore_env_source=True)
            if implied and implied != args.source:
                print(
                    f"[model] note: '{new_model}' looks like a {implied} model, but this session is "
                    f"pinned to {args.source} (from --source). Relaunch with --source {implied} to use it there.",
                    file=sys.stderr,
                )
        ok2, msg2 = _key_preflight(new_source, new_model, base_url, source_typed=bool(args.source))
        if msg2:
            print(msg2, file=sys.stderr)
        if not ok2:
            return None
        # Build UNWIRED (mcp_path=None): _cmd_switch_model re-attaches the runtime-effective
        # config (state['mcp_path']) exactly once, so a runtime /mcp survives the switch and
        # we never double-wire (which would spawn duplicate server subprocesses).
        return _make_agent_builder(args, new_model, new_source, None), new_source

    return _repl(
        build_agent,
        model,
        source,
        mcp_on=bool(mcp_path),
        show_banner=not args.no_banner,
        args=args,
        switch=_switch,
        mcp_path=mcp_path,
    )


if __name__ == "__main__":
    # main() hardens streams and maps a stray Ctrl-C to 130 itself, so both this
    # ``-m`` path and the installed console scripts behave identically.
    raise SystemExit(main())
