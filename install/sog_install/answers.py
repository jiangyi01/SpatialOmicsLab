"""
Scripted decision source — the reproducible, offline path.

Loads an ``answers.yaml`` scenario, resolves ``${env:VAR}`` references from the
environment (so secrets never live in the file), validates the schema, and
answers every Stage-B decision from it with zero prompts and zero LLM calls.
This is what ``--answers`` drives, and what a scripted scenario matrix runs on.

Because it implements the same :class:`~sog_install.decisions.DecisionSource`
contract as the interactive guide, the wizard/engines can't tell the two apart.

Schema (all keys optional unless noted)::

    llm:                       # consumed by Stage-A onboarding, not the guide
      provider: anthropic
      fields: {ANTHROPIC_API_KEY: "${env:ANTHROPIC_API_KEY}"}
      model: claude-opus-4-8
      validate: true
      reuse_existing: false
      on_validation_fail: retry|switch|continue_unvalidated
    categories: [deconvolution]        # >=1 category OR a non-empty `servers`
    servers: [spacexr, tangram]        # optional explicit subset (any installable tool key;
                                       #   the interactive picker selects these per-tool)
    base_env: {mode: new|reuse, name: wtdemo, install_editable: true}   # required
    provision: {on_tool_fail: continue|abort}
    tests: {tier1: true, tier2: false, categories: [deconvolution], eval: false}
    cleanup: {delete_artifacts: false}
    service_keys: {UCD_TOKEN: "${env:UCD_TOKEN}"}
    knobs: {SOG_DATA_PATH: ./data}
    confirm: true                      # scripted auto-confirm gate
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from . import credentials
from .constants import _validate_env_name, base_env_protection_reason
from .decisions import (
    BaseEnvDecision,
    BaseEnvMode,
    CategorySelection,
    CleanupDecision,
    DecisionSource,
    GuideContext,
    OnToolFail,
    OnValidationFail,
    Proposal,
    ProvisionDecision,
    TestDecision,
    ToolPlan,
    expand_category_servers,
)

_ENV_RE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


def _present_bool(d: dict, key: str, default: bool) -> bool:
    """Read a scripted boolean field, treating a PRESENT-but-NULL value (``key:`` left blank → ``None``)
    as the DEFAULT — not ``bool(None) == False``. A naive ``bool(d.get(key, default))`` silently flips an
    on-by-default field OFF when the user leaves it blank (the 2-arg default only fires on an ABSENT key).
    This is the exact present-null class already fixed for ``validate`` / ``on_validation_fail`` in
    onboarding (``True if v is None else bool(v)``); apply it to the default-``True`` scripted fields too."""
    return _as_bool(d.get(key), key, default)


class AnswersError(ValueError):
    """Malformed or incomplete ``answers.yaml``."""


_TRUE_WORDS = frozenset({"true", "yes", "y", "on", "1"})
_FALSE_WORDS = frozenset({"false", "no", "n", "off", "0"})


def _as_bool(value: Any, key: str, default: bool) -> bool:
    """A scripted boolean, read strictly. ``None`` (and a blank string) is the default.

    ``bool(value)`` read a quoted or ``${env:}``-templated ``"false"``/``"no"`` as True -- the template
    always yields a string -- so ``delete_artifacts: "${env:CLEANUP}"`` with ``CLEANUP=false`` deleted the
    artifacts and ``tier2: "false"`` ran a paid agent run per category (hunt 2026-09-30, u35-setup-ux-11).
    The usual spellings (true/yes/y/on/1, false/no/n/off/0) are read as words; anything else is an
    :class:`AnswersError` naming the field, never a guess."""
    if value is None or isinstance(value, bool):
        return default if value is None else value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if not word:
            return default
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise AnswersError(f"`{key}` must be true or false, got {value!r}")


#: Every scripted on/off field, as ``(section, key)``. Checked at load so a bad spelling is exit 2 up
#: front, not a crash (or a wrong guess) in the phase that reads it.
_BOOL_FIELDS = (
    ("llm", "validate"),
    ("llm", "reuse_existing"),
    ("base_env", "install_editable"),
    ("tests", "tier1"),
    ("tests", "tier2"),
    ("tests", "eval"),
    ("cleanup", "delete_artifacts"),
)


def _resolve_env(obj: Any, missing: list[str]) -> Any:
    """Recursively replace ``${env:VAR}`` in every string value."""
    if isinstance(obj, str):

        def sub(m: re.Match) -> str:
            var = m.group(1)
            if var not in os.environ:
                missing.append(var)
                return ""
            return os.environ[var]

        return _ENV_RE.sub(sub, obj)
    if isinstance(obj, dict):
        return {k: _resolve_env(v, missing) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(v, missing) for v in obj]
    return obj


def load_answers(path: str | Path, *, skip_install: bool = False) -> dict:
    """Load + env-resolve + validate an answers scenario. Raises :class:`AnswersError`.

    ``skip_install`` is the CLI's ``--skip-install``: it relaxes the tool-selection and ``base_env``
    requirements exactly as the file's own ``skip_install: true`` does. Validation ran before the flag
    was consulted, so an llm-only answers file was refused even with the flag (hunt 2026-09-30,
    u35-setup-ux-15)."""
    p = Path(path)
    if not p.exists():
        raise AnswersError(f"answers file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise AnswersError(f"could not parse {p}: {exc}") from exc
    except (UnicodeDecodeError, OSError) as exc:
        # read_text also raises: UnicodeDecodeError (a latin-1/utf-16/stray-byte file), or OSError
        # (the path is a directory, perms, or it vanished after the exists() check). These are NOT
        # yaml.YAMLError, so without this arm they'd escape past the friendly exit-2 --answers path
        # to the cli catch-all (exit 1 + a misleading "re-run to resume"). Honour the docstring's
        # "Raises AnswersError" contract for EVERY answers-file failure mode.
        raise AnswersError(f"could not read {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise AnswersError(f"{p}: top level must be a mapping")

    missing: list[str] = []
    data = _resolve_env(data, missing)
    if missing:
        # A referenced env var wasn't set — surface it rather than silently blanking.
        raise AnswersError(f"unset environment variables referenced by {p}: {sorted(set(missing))}")

    _validate(data, p, skip_install=skip_install)
    return data


def _validate(data: dict, p: Path, *, skip_install: bool = False) -> None:
    # Shape checks first. YAML makes it easy to hand a scalar where a list/mapping is meant, and the
    # scripted path reads these fields with ``.get(...)``/iteration downstream — a wrong type there
    # blows up as an AttributeError/char-by-char garbage deep in the run and surfaces as a generic
    # exit-1 traceback. Catch it here with a friendly, actionable message (exit 2) instead.
    #
    # ``categories``/``servers`` must be lists: a bare string (``categories: deconvolution``) is
    # truthy, so ``data.get(...) or []`` would keep the STRING and the membership check in
    # ``_validate_tool_picks`` would iterate it letter-by-letter ("unknown categories ['d','e',…]").
    for key in ("categories", "servers"):
        val = data.get(key)
        if val is None:
            continue
        if not isinstance(val, list):
            raise AnswersError(f"{p}: `{key}` must be a list of names, got {type(val).__name__}")
        # Element check: a list ELEMENT that isn't a string — a trailing-colon YAML slip turns
        # ``- spatial_clustering:`` into ``{'spatial_clustering': None}``, a nested ``[a, b]`` into a
        # list — is unhashable, so the ``x not in <set>`` membership test in ``_validate_tool_picks``
        # blows up as a raw ``TypeError`` → generic exit 1, defeating this validator's whole purpose
        # (it exists to turn a hand-edit slip into a friendly exit-2 message). Flag it here instead.
        bad = [x for x in val if not isinstance(x, str)]
        if bad:
            raise AnswersError(f"{p}: `{key}` must be a list of NAMES (strings); bad entries: {bad!r}")
    # These sections are consumed later as mappings (``ScriptedSource.choose_*`` → ``.get(...)``,
    # ``llm_setup.write_dotenv`` → ``knobs.items()``). A scalar (``tests: true``) would raise
    # ``True.get(...)`` at that point; validate the shape now.
    for section in ("llm", "tests", "provision", "cleanup", "service_keys", "knobs"):
        val = data.get(section)
        if val is not None and not isinstance(val, dict):
            raise AnswersError(f"{p}: `{section}` must be a mapping of key: value, got {type(val).__name__}")
    # ``skip_install: true`` short-circuits the wizard to the chat, so it must be a real bool — a
    # stray string is truthy and would silently skip a full install the user meant to run.
    si = data.get("skip_install")
    if si is not None and not isinstance(si, bool):
        raise AnswersError(f"{p}: `skip_install` must be true or false, got {type(si).__name__}")

    # Nested ``llm.fields`` (an ENV_VAR: value map) and ``llm.model`` (a model id string) are consumed
    # unguarded downstream: ``onboarding`` does ``dict(llm.get("fields", {}))`` and
    # ``credentials.sanitize_model`` does ``model.strip()``. ``llm`` is validated as a mapping just
    # above, but a scalar/None ``fields`` (``dict(None)`` → TypeError) or a non-string ``model``
    # (``5.strip()`` → AttributeError) still escapes as a generic exit 1 on the flagship ``--answers``
    # path. Guard the SAME char-split/TypeError-class trap the section checks give, → friendly exit 2.
    llm = data.get("llm")
    if isinstance(llm, dict):
        fields = llm.get("fields")
        if fields is not None and not isinstance(fields, dict):
            raise AnswersError(f"{p}: `llm.fields` must be a mapping of ENV_VAR: value, got {type(fields).__name__}")
        model = llm.get("model")
        if model is not None and not isinstance(model, str):
            raise AnswersError(f"{p}: `llm.model` must be a string (a model id), got {type(model).__name__}")

    # Nested ``tests.categories`` is consumed as ``list(t.get("categories") or …)`` in
    # ``ScriptedSource.choose_tests`` — the SAME char-split / TypeError trap as the top-level
    # ``categories`` above (D-F2): a bare string (``tests: {categories: deconvolution}``) would be
    # split letter-by-letter and silently test NOTHING (exit 0); a scalar would ``list(5)``-TypeError
    # to a generic exit 1. ``tests`` is already validated as a mapping just above, so guard the field.
    tests = data.get("tests")
    if isinstance(tests, dict):
        tcats = tests.get("categories")
        if tcats is not None:
            if not isinstance(tcats, list):
                raise AnswersError(f"{p}: `tests.categories` must be a list of names, got {type(tcats).__name__}")
            bad = [x for x in tcats if not isinstance(x, str)]
            if bad:
                raise AnswersError(f"{p}: `tests.categories` must be a list of NAMES (strings); bad entries: {bad!r}")

    # ``knobs`` values become ``.env`` right-hand sides (``assemble_owned_keys`` → ``write_dotenv`` →
    # ``_format_kv``). A YAML *scalar* is fine: ``SOG_DATA_PATH: ./data`` (str),
    # ``SOG_TIMEOUT_SECONDS: 600`` (int) and ``SOG_SELF_REVIEW_ENABLED: true`` (bool) all coerce
    # cleanly to a string RHS (done at consumption in ``assemble_owned_keys``). All three are real
    # entries of ``credentials.KNOBS``: an example nothing reads teaches a knob that does nothing,
    # and nothing downstream would object -- the key names are never validated, only the values.
    # A *container* value (``FOO: [a, b]`` / ``FOO: {x: 1}``) has no sensible
    # ``.env`` spelling AND, being non-str, hits ``_format_kv``'s ``"$" in value`` membership test as a
    # raw ``TypeError`` → generic exit 1 (C-F1). Reject containers here with a friendly, actionable
    # message — the same fail-fast the ``categories``/``tests.categories`` element checks give above.
    knobs = data.get("knobs")
    if isinstance(knobs, dict):
        bad_knobs = sorted(k for k, v in knobs.items() if isinstance(v, (list, dict, tuple)))
        if bad_knobs:
            raise AnswersError(
                f"{p}: `knobs` values must be scalars (string/number/bool); "
                f"these have list/mapping values: {bad_knobs!r}"
            )

    # `service_keys` values become `.env` right-hand sides too (ScriptedSource.collect_service_key →
    # write_dotenv). Guard them exactly like `knobs`: a container value has no sensible `.env` spelling,
    # and collect_service_key does `str(val)` — so a YAML indentation slip (`UCD_TOKEN:\n  - tok` →
    # `{'UCD_TOKEN': ['tok']}`) would otherwise be str()-ed into the garbage token "['tok']", written to
    # `.env`, and reported as "saved 1 service key(s)" (the placeholder-gate misses it — no `<`/`>`/`...`),
    # only to 401 confusingly at runtime. Reject containers here with the same friendly exit-2 every other
    # `--answers` hand-edit slip already gets, rather than silently persisting a broken credential.
    svc_keys = data.get("service_keys")
    if isinstance(svc_keys, dict):
        bad_svc = sorted(k for k, v in svc_keys.items() if isinstance(v, (list, dict, tuple)))
        if bad_svc:
            raise AnswersError(
                f"{p}: `service_keys` values must be scalars (string/number); "
                f"these have list/mapping values: {bad_svc!r}"
            )

    for section, key in _BOOL_FIELDS:
        block = data.get(section)
        if isinstance(block, dict):
            try:
                _as_bool(block.get(key), f"{section}.{key}", False)
            except AnswersError as exc:
                raise AnswersError(f"{p}: {exc}") from exc

    skip_install = skip_install or bool(data.get("skip_install"))
    cats = data.get("categories") or []
    servers = data.get("servers") or []
    if not skip_install and not cats and not servers:
        raise AnswersError(f"{p}: provide at least one of `categories` or `servers` (or set `skip_install: true`)")
    _validate_tool_picks(cats, servers, p)

    base = data.get("base_env")
    # A skip-install run may omit base_env — it's only consumed if a base env must actually be built
    # (ScriptedSource.choose_base_env), which the skip path does only on a fresh machine. Validate it
    # whenever it's provided, or whenever a full install is scripted (where it stays required).
    if base is not None or not skip_install:
        if not isinstance(base, dict) or not base.get("name"):
            raise AnswersError(f"{p}: `base_env.name` is required")
        # The name flows verbatim into ``conda create -n <name>`` (ScriptedSource.choose_base_env →
        # base_env.create_base). Validate it's a legal conda env name HERE — the same N15/#100 rule the
        # interactive guide applies — so a space/slash/leading-dot fails friendly (exit 2) now, not as a
        # cryptic ``CondaError`` exit-3 deep in the build. Shared validator lives in ``constants``.
        name_reason = _validate_env_name(str(base["name"]))
        if name_reason is not None:
            raise AnswersError(f"{p}: invalid `base_env.name` {base['name']!r} — {name_reason}")
        # M1: reject a reserved env name up front on the CI path — otherwise ``--answers`` + ``--yes``
        # would auto-confirm an install INTO a protected env (base_env.establish refuses it too, but a
        # clean exit-2 here beats a mid-run phase-fail).
        protected_reason = base_env_protection_reason(str(base["name"]))
        if protected_reason is not None:
            raise AnswersError(f"{p}: {protected_reason}")
        try:
            BaseEnvMode.coerce(base.get("mode", "new"))
        except ValueError as exc:
            raise AnswersError(f"{p}: {exc}") from exc
    if data.get("provision"):
        try:
            OnToolFail.coerce(data["provision"].get("on_tool_fail", "continue"))
        except ValueError as exc:
            raise AnswersError(f"{p}: {exc}") from exc

    # llm block (Stage-A). A typo'd provider would otherwise KeyError deep inside
    # onboarding (_choose_provider → credentials.get_provider); catch it here with a
    # clear, actionable list instead. Likewise validate on_validation_fail early.
    llm = data.get("llm")
    if isinstance(llm, dict):
        prov = llm.get("provider") or llm.get("source")
        if prov:
            try:
                credentials.get_provider(str(prov))
            except KeyError as exc:
                valid = ", ".join(sorted(pspec.key for pspec in credentials.PROVIDER_SPECS))
                raise AnswersError(f"{p}: unknown llm.provider {prov!r}; valid providers: {valid}") from exc
        if llm.get("on_validation_fail") is not None:
            try:
                OnValidationFail.coerce(llm["on_validation_fail"])
            except ValueError as exc:
                raise AnswersError(f"{p}: {exc}") from exc


def _validate_tool_picks(cats: list, servers: list, p: Path) -> None:
    """Reject unknown ``categories``/``servers`` entries with an actionable message.

    On the scripted ``--answers``/CI path a hand-written typo would otherwise vanish
    silently: :func:`~sog_install.decisions.expand_category_servers` maps an
    unknown category to ``[]`` and trusts an explicit ``servers`` list verbatim, so the
    run builds nothing (or the wrong subset) yet reports success. The interactive picker
    can only emit real keys, so this guards the scripted path only — mirroring the
    ``llm.provider`` check: fail loudly here with the valid list.

    Best-effort: if the installable universe can't be loaded (an odd checkout missing the
    MCP config or the tool specs), skip rather than invent a *new* failure mode.
    """
    # Lazy import (matches cli's per-subcommand import style) — keeps `answers` light and
    # avoids any import-time coupling to the spec/category loaders on the fast paths.
    from . import categories as _categories
    from . import specs as _specs

    try:
        known_cats = {c.name for c in _categories.load_categories()}
    except Exception:  # pragma: no cover - can't load the universe ⇒ don't block the run
        known_cats = set()
    try:
        known_servers = set(_specs.load_all_specs())
    except Exception:  # pragma: no cover
        known_servers = set()

    if known_cats:
        bad_cats = [c for c in cats if c not in known_cats]
        if bad_cats:
            valid = ", ".join(sorted(known_cats))
            noun = "category" if len(bad_cats) == 1 else "categories"
            raise AnswersError(f"{p}: unknown {noun} {bad_cats!r}; valid categories: {valid}")

    if known_servers:
        bad = [s for s in servers if s not in known_servers]
        if bad:
            # Common naming trap: RCTD's installable tool key is `spacexr`, not `rctd`.
            hint = " (RCTD's tool key is `spacexr`)" if any(str(s).lower() == "rctd" for s in bad) else ""
            noun = "tool key" if len(bad) == 1 else "tool keys"
            raise AnswersError(
                f"{p}: unknown {noun} in `servers`: {bad!r}{hint}; "
                "check the spelling, or run `sog-setup` interactively to see the tool menu"
            )


class ScriptedSource(DecisionSource):
    """A :class:`DecisionSource` backed entirely by a parsed answers dict."""

    def __init__(self, answers: dict) -> None:
        self.answers = answers

    # -- Stage-A helper (not part of the DecisionSource ABC) ------------------
    @property
    def llm(self) -> dict | None:
        return self.answers.get("llm")

    # -- categories -----------------------------------------------------------
    def choose_categories(self, ctx: GuideContext) -> CategorySelection:
        cats = list(self.answers.get("categories") or [])
        explicit = list(self.answers.get("servers") or [])
        servers = expand_category_servers(cats, explicit, ctx.categories)
        return CategorySelection(categories=cats, servers=servers, rationale="scripted answers")

    # -- base env -------------------------------------------------------------
    def choose_base_env(self, ctx: GuideContext) -> BaseEnvDecision:
        # A skip-install run may omit `base_env` (answers._validate allows it) yet still reach here
        # when a base env must be built on a fresh machine — default to a NEW "sog" env instead of a
        # KeyError. A full run always carries base_env (validated), so the default never applies there.
        base = self.answers.get("base_env") or {}
        return BaseEnvDecision(
            mode=BaseEnvMode.coerce(base.get("mode", "new")),
            # str(): YAML reads `name: 2024` as an int, which _validate checks as "2024" but which then
            # reached the env-name guards raw and crashed them (`.strip()` on an int → exit 1) (hunt
            # 2026-09-30, u35-setup-ux-12).
            basic_env_name=str(base.get("name") or "sog").strip(),
            # present-null → default True (a blank `install_editable:` must not silently skip `pip install -e .`,
            # which in NEW mode then fails the base-env phase with a confusing "core packages missing").
            install_editable=_present_bool(base, "install_editable", True),
            source_env_yaml=base.get("source_env_yaml"),
            rationale="scripted answers",
        )

    # -- provisioning ---------------------------------------------------------
    def plan_provision(self, ctx: GuideContext) -> ProvisionDecision:
        candidates: list[ToolPlan] = ctx.extra.get("candidate_tools", [])
        on_fail = OnToolFail.coerce((self.answers.get("provision") or {}).get("on_tool_fail", "continue"))
        return ProvisionDecision(tools=list(candidates), on_tool_fail=on_fail)

    # -- contextual service keys ---------------------------------------------
    def collect_service_key(self, env_var: str, ctx: GuideContext) -> str | None:
        val = (self.answers.get("service_keys") or {}).get(env_var)
        return str(val) if val else None

    # -- tests ----------------------------------------------------------------
    def choose_tests(self, ctx: GuideContext) -> TestDecision:
        t = self.answers.get("tests") or {}
        tcats = t.get("categories")
        run_tier2 = _as_bool(t.get("tier2"), "tests.tier2", False)
        if tcats is not None and not tcats:
            # An explicit [] reached testing.run_tests, whose `categories_to_test or <every category>`
            # turned it back into ALL categories -- a real agent run per category, the opposite of what
            # the file says. Tier-2 is the per-category test, so no categories means no Tier-2 run
            # (hunt 2026-09-30, u35-setup-ux-13).
            run_tier2 = False
        if tcats is None:
            # Key ABSENT → default to the categories chosen for the build. But an EXPLICIT empty list
            # (`tests: {categories: []}`) means "run the general tiers, skip every category-specific
            # portal test" and must be honoured verbatim. The old `t.get("categories") or (…selected)`
            # collapsed `[]` (falsy) back to the whole build set, silently over-testing against the
            # answer file's stated intent — `is None` distinguishes absent from deliberately-empty (D-E).
            tcats = ctx.selected.categories if ctx.selected else []
        return TestDecision(
            # present-null → default True: a blank `tests.tier1:` must not silently skip Tier-1 mini-data
            # validation and still exit 0 green (CI could not tell a validated build from an unvalidated one).
            run_tier1=_present_bool(t, "tier1", True),
            run_tier2=run_tier2,
            categories_to_test=list(tcats),
            tier2_eval=_as_bool(t.get("eval"), "tests.eval", False),
        )

    # -- cleanup --------------------------------------------------------------
    def choose_cleanup(self, ctx: GuideContext) -> CleanupDecision:
        c = self.answers.get("cleanup") or {}
        return CleanupDecision(delete_artifacts=_as_bool(c.get("delete_artifacts"), "cleanup.delete_artifacts", False))

    # -- confirm gate ---------------------------------------------------------
    def confirm(self, proposal: Proposal, ctx: GuideContext) -> bool:
        val = self.answers.get("confirm")
        if val is None:
            # absent OR present-but-null → default auto-confirm. A blank top-level `confirm:` must not
            # decline every propose→execute gate (which would build nothing and exit 1 for a non-obvious reason).
            return True
        if isinstance(val, str):
            return val.strip().lower() in ("true", "yes", "auto", "y", "1")
        return bool(val)
