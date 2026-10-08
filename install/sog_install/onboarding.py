"""
Stage-A onboarding — the front door.

The first thing a user sees: an emoji banner + roadmap checklist, then a
friendly LLM-connection walkthrough (choose a provider → enter key/endpoint/model
→ live Tier-1 validation with actionable retry → safe ``.env`` write), closing
with the auto-handoff line to the Stage-B guide.

Runs before any conda env exists, so it is stdlib-only and talks to providers
over :mod:`sog_install.llm_setup` (urllib), never langchain.

Two drive modes:

* **interactive** — real prompts via :class:`~sog_install.prompts.PromptIO`.
* **scripted** — an ``answers`` dict (the ``llm:`` block of ``answers.yaml``)
  satisfies every step with zero prompts (the ``--answers``/CI path).

Returns an :class:`~sog_install.decisions.LLMChoice` carrying the
verified source/model plus the field values *in memory* (for the guide's chat
client). No secret is ever persisted to setup state.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from . import constants, credentials, key_library, llm_setup
from .decisions import LLMChoice, OnValidationFail
from .session_log import register_secret

#: Onboarding Step 1, when the vault holds keys AND ``.env`` holds a working config: keep that config.
CURRENT_ID = "__current__"

if TYPE_CHECKING:
    from .prompts import PromptIO
    from .session_log import SessionLog


def _provider_menu() -> list[tuple[str, str, str]]:
    opts = []
    for p in credentials.PROVIDER_SPECS:
        need = ", ".join(f.env_var for f in p.required) or "no key"
        opts.append((p.key, p.label, f"needs: {need}"))
    return opts


def _choose_provider(io: PromptIO, answers: dict | None) -> credentials.ProviderSpec:
    if answers is not None:
        key = answers.get("provider") or answers.get("source") or credentials.DEFAULT_PROVIDER_KEY
        try:
            return credentials.get_provider(key)
        except KeyError as exc:
            # The flagship --answers path pre-validates the provider in answers.load_answers, but a
            # resumed-state dict or any other caller could still carry a typo here — surface the same
            # friendly, actionable message the CLI already maps to exit 2, not a raw KeyError + exit 1.
            from .answers import AnswersError

            valid = ", ".join(sorted(p.key for p in credentials.PROVIDER_SPECS))
            raise AnswersError(f"unknown llm provider {key!r}; valid providers: {valid}") from exc
    chosen = io.select("  Provider?", _provider_menu(), default=credentials.DEFAULT_PROVIDER_KEY)
    return credentials.get_provider(chosen)


def _collect_fields(io: PromptIO, provider: credentials.ProviderSpec, answers: dict | None) -> dict[str, str]:
    """Gather the provider's required + (offered) optional field values."""
    if answers is not None:
        # `.get("fields", {})` returns None when `fields:` is present-but-empty (the default only
        # applies to an ABSENT key), and `dict(None)` TypeErrors — treat empty/None as "no fields".
        # A non-mapping non-None `fields` (scalar/list) is rejected up front by answers._validate.
        given = dict(answers.get("fields") or {})
        # Only keep vars this provider knows about. Stripped and registered for masking as the
        # interactive ask_secret does: a CI secret with a trailing newline went into the ping header
        # raw, and http.client's refusal quoted the whole key into the console and the run log
        # (hunt 2026-09-30, u35-setup-ux-1). A control character INSIDE a value cannot be a key.
        values: dict[str, str] = {}
        for f in provider.all_fields():
            v = given.get(f.env_var)
            if v in (None, ""):
                continue
            text = str(v).strip()
            if not text:
                continue
            if f.secret:
                register_secret(text)
            if any(ord(c) < 32 or ord(c) == 127 for c in text):
                from .answers import AnswersError

                raise AnswersError(
                    f"llm.fields.{f.env_var} contains a line break or other control character inside the "
                    "value -- fix the value the answers file reads"
                )
            values[f.env_var] = text
        return values

    values: dict[str, str] = {}
    if provider.where_to_get_key:
        io.note(f"Get a key at {provider.where_to_get_key}")
    for f in provider.required:
        values[f.env_var] = io.ask_secret(f"  Paste {f.env_var}") if f.secret else io.ask_text(f"  {f.label}")
    for f in provider.optional:
        prompt = f"  {f.label} ({f.env_var})" + (f" [{f.placeholder}]" if f.placeholder else "")
        if io.ask_yesno(f"  Set optional {f.env_var}?", default=False):
            values[f.env_var] = (
                io.ask_secret(prompt) if f.secret else io.ask_text(prompt, default=f.placeholder or None)
            )
    return values


def _validate(io: PromptIO, provider: credentials.ProviderSpec, values: dict[str, str]) -> llm_setup.PingResult:
    io.say("  🔄 Validating...")
    result = llm_setup.tier1_ping(provider, values)
    if result.ok and _unconfirmed(result):
        io.warn(f"endpoint reachable, but the key was not confirmed ({result.detail})")
    elif result.ok:
        io.ok(result.detail if result.presence_only else f"key works ({result.detail})")
    else:
        io.err(f"validation failed: {result.detail}")
    return result


def _unconfirmed(result: llm_setup.PingResult) -> bool:
    """True for a ping that let the config through without proving the key: the Azure/Custom models
    route answered 404, so ``auth_ok`` stayed ``None``. Printing "key works" for that, and saving the
    key to the vault as validated, claimed a check that never happened (hunt 2026-09-30,
    uL4-honesty-12). Bedrock's presence-only check says what it is and is unchanged."""
    return result.ok and result.auth_ok is None and not result.presence_only


def _validated(result: llm_setup.PingResult) -> bool:
    """The ping proved the config works -- not merely that nothing refused it."""
    return result.ok and not _unconfirmed(result)


def _answer_flag(answers: dict | None, key: str, default: bool) -> bool:
    """A scripted ``llm:`` boolean, read strictly: ``bool("false")`` is True, so a quoted or
    ``${env:}``-templated ``validate: "false"`` still pinged and ``reuse_existing: "no"`` still reused
    (hunt 2026-09-30, u35-setup-ux-11). Absent and present-null both mean ``default`` (R24-A1)."""
    if answers is None:
        return default
    from .answers import _as_bool  # lazy, like AnswersError: answers pulls in yaml

    return _as_bool(answers.get(key), f"llm.{key}", default)


def _persist(
    io: PromptIO,
    provider: credentials.ProviderSpec,
    values: dict[str, str],
    model: str | None,
    knobs: dict[str, str] | None,
) -> None:
    owned = llm_setup.assemble_owned_keys(provider, values, model=model, knobs=knobs)
    report, backup = llm_setup.write_dotenv(owned)
    # Reflect into this process so later in-process get_llm / the guide client see it.
    os.environ.update(owned)
    where = f" (backed up to {backup.name})" if backup else ""
    io.ok(f"Saved to {constants.DOTENV_REL}{where}.")
    if report.migrated:
        io.note("migrated legacy names: " + ", ".join(f"{o}→{n}" for o, n in report.migrated))
    if report.deduped:
        io.note("removed duplicate keys: " + ", ".join(report.deduped))
    # Azure-only: a host-only endpoint (no deployment in the URL) with no SOG_LLM falls back to the
    # `gpt-4o` *placeholder* deployment. That passes key validation but 404s (DeploymentNotFound) on
    # the agent's first real call, so surface it now — at setup time — with an actionable fix.
    if llm_setup.azure_deployment_is_placeholder(provider, values, model):
        ph = provider.default_model
        io.warn(
            f"No Azure deployment name was found (your endpoint URL has no '/deployments/<name>/' "
            f"path and SOG_LLM is unset), so the placeholder '{ph}' will be used. If your Azure "
            f"resource has no deployment named exactly '{ph}', the agent returns HTTP 404 "
            f"(DeploymentNotFound) on its first call. Set SOG_LLM in {constants.DOTENV_REL} to your "
            f"real deployment name (shown under 'Deployments' in your Azure resource)."
        )


def _reusable_existing() -> tuple[llm_setup.ExistingLLM, credentials.ProviderSpec] | None:
    """The ``.env`` LLM config onboarding may offer to keep, or ``None``.

    A .env freshly copied from .env.example still holds template stubs (e.g.
    ``ANTHROPIC_API_KEY=sk-ant-...``). detect_existing_llm marks that "complete" on presence alone, so
    without this screen we'd offer to KEEP the placeholder (default-yes) and the agent would fail its
    first real call. Every required field is screened, not only the secret ones: the shipped
    ``OPENAI_ENDPOINT=https://<resource>.openai.azure.com`` beside a real key was offered as "a working
    Azure OpenAI config" (hunt 2026-09-30, u35-setup-ux-5)."""
    existing = llm_setup.detect_existing_llm()
    if existing is None or not existing.looks_complete or not existing.provider_key:
        return None
    provider = credentials.get_provider(existing.provider_key)
    reuse_vals = llm_setup.read_dotenv_values()
    if any(llm_setup.value_looks_like_placeholder(reuse_vals.get(f.env_var, "")) for f in provider.required):
        return None
    return existing, provider


def _current_model_for(provider: credentials.ProviderSpec) -> str | None:
    """The model ``.env`` names for ``provider``, or ``None`` (another provider, or no SOG_LLM)."""
    existing = llm_setup.detect_existing_llm()
    if existing is None or existing.provider_key != provider.key or existing.model_is_default:
        return None
    return existing.model


def _describe_model(existing: llm_setup.ExistingLLM) -> str:
    if existing.model_is_default:
        return f"none set; the default {existing.model} would be used"
    return str(existing.model)


def _maybe_reuse_existing(
    io: PromptIO, answers: dict | None, *, knobs: dict | None = None, persist: bool = True, ask: bool = True
) -> LLMChoice | None:
    """Offer to keep an already-working config (never silently repoint).

    ``knobs`` (the answers file's top-level ``knobs:`` block) is still applied on the reuse path so
    a scripted run's documented knobs (e.g. ``SOG_DATA_PATH``) reach ``.env`` even when the LLM key
    itself is reused rather than freshly entered (Round-6 C).

    ``ask=False`` skips the interactive "Reuse it?" question for a caller that already asked it (the
    saved-key menu's "Keep the current config" row)."""
    found = _reusable_existing()
    if found is None:
        return None
    existing, provider = found

    if answers is not None:
        if not _answer_flag(answers, "reuse_existing", False):
            return None
        # Reuse keeps .env's config; a provider or model the file also pins is not applied. Say so
        # rather than drop it without a word (hunt 2026-09-30, skeptic note on u35-setup-ux-2/-4).
        pinned = answers.get("provider") or answers.get("source")
        try:
            pinned_spec = credentials.get_provider(str(pinned)) if pinned else None
        except KeyError:
            pinned_spec = None
        if pinned_spec is not None and pinned_spec.key != provider.key:
            io.warn(
                f"llm.provider {pinned!r} is not applied: reuse_existing keeps the {provider.label} config "
                f"already in {constants.DOTENV_REL}"
            )
        elif answers.get("model") and answers.get("model") != existing.model:
            io.warn(
                f"llm.model {answers.get('model')!r} is not applied: reuse_existing keeps the model "
                f"{_describe_model(existing)}"
            )
    elif ask:
        io.say(f"  Found a working {provider.label} config (model {_describe_model(existing)}).")
        if not io.ask_yesno("  Reuse it?", default=True):
            return None

    values = {
        k: v for k, v in llm_setup.read_dotenv_values().items() if k in {f.env_var for f in provider.all_fields()}
    }
    # Migrate a model id an older sog-setup wrote to .env that is invalid at the provider — it passed
    # the key-only ping when saved but 404s the guide/agent on the first real call. Using the repaired
    # id for the re-persist AND the returned choice durably rewrites SOG_LLM on this very run, so the
    # stale id is gone next time rather than resurrected again.
    # Only a model .env actually names goes to _persist: the provider default handed over as if
    # configured made azure_deployment_is_placeholder() False and silenced its DeploymentNotFound
    # warning (hunt 2026-09-30, u35-setup-ux-14).
    configured = None if existing.model_is_default else existing.model
    model, bad_model_note = credentials.sanitize_model(configured, source=provider.source)
    if bad_model_note:
        io.warn(bad_model_note)
    # Absent AND present-null `validate:` both mean "default: do validate"; only an explicit false
    # skips it (R24-A1) -- and a quoted "false" is false (u35-setup-ux-11).
    should_validate = _answer_flag(answers, "validate", True)
    result = _validate(io, provider, values) if should_validate else None
    validated = result is not None and _validated(result)
    on_fail = OnValidationFail.RETRY
    if result is not None and not result.ok:
        # A failed ping used to be followed by "Saved to .env" and "LLM connected" regardless, and a
        # scripted reuse_existing run went on to build every env and exit 0 on a dead key. Decide
        # exactly as the fresh path does, BEFORE anything is saved (hunt 2026-09-30, u35-setup-ux-5).
        if answers is not None:
            on_fail = OnValidationFail.coerce(answers.get("on_validation_fail") or "retry")
            if on_fail != OnValidationFail.CONTINUE_UNVALIDATED:
                from .answers import AnswersError

                raise AnswersError(
                    f"scripted LLM validation of the reused {constants.DOTENV_REL} config failed and "
                    f"on_validation_fail='{on_fail.value}' cannot proceed non-interactively. Set "
                    "on_validation_fail: continue_unvalidated to build the tool envs anyway, or fix the key."
                )
        else:
            on_fail = _ask_validation_branch(io)
            if on_fail != OnValidationFail.CONTINUE_UNVALIDATED:
                return None  # re-enter or switch: the collect-a-new-key flow takes over
        io.warn("Continuing with an unvalidated key — the agent may fail later.")
    if persist:
        # Re-persist owned keys idempotently (ensures SOG_SOURCE/LLM_SOURCE present).
        _persist(io, provider, values, model, knobs)
        if validated:
            _remember_key(io, provider, values, model)
    else:
        io.note(f"[dry-run] would reuse the existing {provider.label} config (nothing written)")
    return LLMChoice(
        source=provider.source,
        model=model or provider.default_model,
        field_values=values,
        validated=validated,
        on_validation_fail=on_fail,
    )


def configure_key(
    io: PromptIO,
    provider: credentials.ProviderSpec,
    *,
    values: dict[str, str] | None = None,
    model: str | None = None,
    knobs: dict[str, str] | None = None,
    validate: bool = True,
    remember: bool = True,
    persist: bool = True,
) -> bool:
    """Collect (or accept) a provider's key fields and sync them into ``.env`` the canonical
    "ST-Coscientist way" — ``assemble_owned_keys`` + ``write_dotenv`` (backup + merge-not-clobber
    + secret masking) + ``os.environ.update``.

    Shared by first-run onboarding and the post-install run phase (:mod:`demo`), which connects a
    key at launch time. ``values`` supplies the fields directly (the "reuse the setup key" path, no
    re-prompt); when ``None`` they are collected interactively via :func:`_collect_fields`
    (``io.ask_secret`` — masked, never echoed). Returns ``True`` when a non-empty key was written,
    ``False`` when nothing usable was gathered (caller keeps its key-absent path).

    When ``remember`` and ``validate`` and the ping succeeds, the working key is added to the saved
    key library (:mod:`key_library`) so a later run can offer it in the pick-list. Callers reusing a
    known key (the demo gate's "reuse setup key" / library-pick paths) pass ``remember=False``.

    ``knobs`` (a scripted run's top-level ``knobs:`` block) is folded into the written env alongside
    the key, so documented knobs (e.g. ``SOG_DATA_PATH``) reach ``.env`` on the library-pick path too;
    ``None`` (the interactive demo callers) writes no extra vars.

    ``persist=False`` (interactive ``--dry-run``) collects/validates the key but writes nothing —
    no ``.env``, no backup, no vault entry, no ``os.environ`` mutation. Returns ``True`` when a
    usable key was gathered all the same, so the caller's in-process choice still reflects it."""
    # Defensive migration at the single persist chokepoint. Every caller — onboarding's reuse/library
    # paths AND the post-install demo run phase (which can pass a stale vault ``entry.model``) — funnels
    # through here before ``_persist`` writes ``SOG_LLM``. Repairing a known-bad id here guarantees no
    # path can re-poison ``.env`` with a model that 404s, even a future caller that skips the
    # onboarding-layer guards. The onboarding paths pre-sanitize (and warn), so this is a silent no-op
    # there; the demo path relies on it. See :func:`credentials.sanitize_model`.
    model, _bad_model_note = credentials.sanitize_model(model, source=provider.source)
    if _bad_model_note:
        io.warn(_bad_model_note)
    vals = dict(values) if values is not None else _collect_fields(io, provider, None)
    # A key-requiring provider with nothing usable → keep the caller's key-absent path. But a KEYLESS
    # provider (Ollama: required == []) legitimately gathers {} — selecting it IS the config — so the
    # empty-check must not drop it, else _persist (which writes SOG_SOURCE/LLM_SOURCE/SOG_LLM) never
    # runs and the Ollama pick silently vanishes. Gate the reject on the provider actually needing a
    # field. (SETUP-3c)
    if provider.required and not any((v or "").strip() for v in vals.values()):
        return False
    ok = False
    if validate:
        try:
            # best-effort; a failed ping is non-fatal (mirrors onboarding). An unconfirmed one (a 404 on
            # the models route) is not a validated key either (hunt 2026-09-30, uL4-honesty-12).
            ok = _validated(_validate(io, provider, vals))
        except Exception:  # a network hiccup must never block writing a key the user gave us
            ok = False
    if not persist:  # dry-run: a usable key was gathered, but write nothing to disk
        return True
    _persist(io, provider, vals, model, knobs)
    if remember and validate and ok:
        _remember_key(io, provider, vals, model)
    return True


def _remember_key(
    io: PromptIO,
    provider: credentials.ProviderSpec,
    values: dict[str, str],
    model: str | None,
) -> None:
    """Add a just-validated key to the saved-key library, labeled by its LLM.

    Best-effort: a vault read/write hiccup must never block onboarding or a launch, so every failure
    is swallowed. The model is resolved the same way :func:`_persist` writes it
    (``assemble_owned_keys`` → ``SOG_LLM``), so an Azure entry stores its *deployment* name rather
    than the ``gpt-4o`` default."""
    try:
        resolved = llm_setup.assemble_owned_keys(provider, values, model=model).get("SOG_LLM") or (
            model or provider.default_model
        )
        key_library.remember(
            io,
            source=provider.source,
            model=resolved,
            field_values=values,
            validated=True,
            now=key_library._now(),
        )
    except Exception:  # vault is best-effort; never let it break the key write / launch
        pass


def _select_from_library(
    io: PromptIO, lib: key_library.KeyLibrary, answers: dict | None, *, knobs: dict | None = None, persist: bool = True
) -> LLMChoice | None:
    """Offer the saved-key pick-list at onboarding Step 1.

    Returns an :class:`LLMChoice` for a reused saved key, or ``None`` to fall through to the
    collect-a-new-key flow (the user picked "Enter a new key", the library is unusable, or a
    scripted run didn't name a ``saved_key``). Reuse writes ``.env`` via :func:`configure_key`
    without re-prompting or re-validating — the key already proved it works when it was saved.

    When ``.env`` already holds a working config it is the FIRST row and the default ("Keep the
    current config"). Without it the default was the most recently used vault entry, so a
    non-interactive run, or an Enter, overwrote a hand-edited or portal-changed ``.env`` with a stale
    vault key -- the silent repoint ``_maybe_reuse_existing`` exists to prevent (hunt 2026-09-30,
    u35-setup-ux-3).

    ``persist=False`` (dry-run) reuses the picked key *in-process* only — no ``.env`` write, no
    ``last_used`` touch on the vault entry."""
    options = key_library.menu_options(lib)
    if not options:
        return None
    current = _reusable_existing() if answers is None else None
    head = []
    if current is not None:
        existing, provider = current
        head = [
            (
                CURRENT_ID,
                f"Keep the current {constants.DOTENV_REL} config",
                f"{provider.label} · model {_describe_model(existing)}",
            )
        ]
    menu = head + options + [(key_library.NEW_ID, "Enter a new key", "")]
    if answers is not None:
        choice = str(answers.get("saved_key") or key_library.NEW_ID)
    else:
        choice = io.select("  Pick a saved key or add a new one:", menu, default=menu[0][0])
    if choice == CURRENT_ID:
        return _maybe_reuse_existing(io, None, knobs=knobs, persist=persist, ask=False)
    if choice == key_library.NEW_ID:
        return None
    entry = key_library.get(lib, choice)
    if entry is None:
        return None
    provider = key_library._provider_for(entry.source)
    if provider is None:
        return None
    # A vault entry saved by an older sog-setup can carry an invalid model id (valid key, bad model);
    # migrate it so the reused config writes the repaired id to .env and the guide gets a good model.
    model, bad_model_note = credentials.sanitize_model(entry.model, source=entry.source)
    if bad_model_note:
        io.warn(bad_model_note)
    if not configure_key(
        io,
        provider,
        values=entry.field_values,
        model=model,
        knobs=knobs,
        validate=False,
        remember=False,
        persist=persist,
    ):
        return None
    if persist:
        key_library.touch(entry.id, key_library._now())
    io.ok(f"{'[dry-run] would reuse' if not persist else 'Reusing'} {key_library.label_for(entry)}")
    return LLMChoice(
        source=entry.source,
        model=model or provider.default_model,
        field_values=dict(entry.field_values),
        validated=entry.validated,
    )


def run_onboarding(
    io: PromptIO,
    log: SessionLog | None = None,
    *,
    answers: dict | None = None,
    knobs: dict | None = None,
    roadmap_done: tuple[int, ...] = (0,),
    persist: bool = True,
) -> LLMChoice:
    """Execute Stage A and return the verified :class:`LLMChoice`.

    ``answers`` is the ``llm:`` block for scripted runs; when ``None`` the flow
    is interactive.

    ``knobs`` is the answers file's **top-level** ``knobs:`` block (e.g. ``SOG_DATA_PATH``),
    passed in separately because it is a sibling of ``llm:`` — reading it out of ``answers``
    (the ``llm`` sub-block) silently dropped it, so documented knobs never reached ``.env``
    (Round-6 C). ``None`` (interactive, or no ``knobs:`` in the file) writes no extra vars.

    ``persist=False`` (interactive ``--dry-run``) keeps the collect/validate UX so the user
    sees what *would* happen, but writes nothing to disk — no ``.env``, no backup, no saved-key
    vault entry, no ``os.environ`` mutation. The returned :class:`LLMChoice` still carries the
    field values in-process so the guide client works for the plan preview.
    """
    io.banner(
        f"{constants.PRODUCT_NAME} — SETUP GUIDE",
        "I'll get you from a fresh clone to a working agent. Here's the plan:",
    )
    io.roadmap(done=roadmap_done)
    io.section("Step 1 — Connect your LLM")

    # 0) reuse a saved/working config if present (and desired). When the saved-key library holds
    #    anything, its multi-key pick-list (headed by the current .env config, when there is one)
    #    supersedes the single-.env "Reuse it?" yes/no; an empty library falls back to the unchanged
    #    first-run reuse offer, so behavior is identical until the first key is actually saved.
    lib = key_library.load()
    if answers is not None and _answer_flag(answers, "reuse_existing", False):
        # A scripted `reuse_existing: true` is consulted whatever the vault holds. It used to be read
        # only on the empty-vault branch, and its own first run fills the vault, so the identical file
        # failed (exit 2, "missing required field") on its second run (hunt 2026-09-30, u35-setup-ux-4).
        reused = _maybe_reuse_existing(io, answers, knobs=knobs, persist=persist)
        if reused is not None:
            _handoff(io)
            return reused
    if lib.entries:
        picked = _select_from_library(io, lib, answers, knobs=knobs, persist=persist)
        if picked is not None:
            _handoff(io)
            return picked
        # "Enter a new key" (or an unusable pick) → fall through to the collect-new loop below.
    elif answers is None:
        reused = _maybe_reuse_existing(io, answers, knobs=knobs, persist=persist)
        if reused is not None:
            _handoff(io)
            return reused

    llm_answers = answers or None
    on_fail_default = (
        # `... or "retry"` (not `.get(k, "retry")`): a present-but-null `on_validation_fail:` returns
        # None (the default only applies when the key is ABSENT), and `OnValidationFail.coerce(None)`
        # deliberately raises ValueError (decisions.py). That raw ValueError is NOT wrapped in
        # AnswersError, so it unwinds to cli's generic handler as exit 1 ("internal crash") instead of
        # the exit-2 "malformed answers file". Collapse absent AND present-null to the retry default,
        # matching the answers validator's `on_validation_fail is not None` intent. (R20)
        OnValidationFail.coerce((llm_answers or {}).get("on_validation_fail") or "retry")
        if llm_answers
        else OnValidationFail.RETRY
    )

    while True:
        provider = _choose_provider(io, llm_answers)
        values = _collect_fields(io, provider, llm_answers)
        model = (llm_answers or {}).get("model") or None
        # Interactive entry never asks a non-Azure model, so a user who only rotated the key had the
        # provider default written over their SOG_LLM (claude-haiku -> claude-opus, no word said).
        # Same provider as .env: keep its model, and say so (hunt 2026-09-30, u35-setup-ux-7).
        kept = _current_model_for(provider) if model is None and llm_answers is None else None
        if kept and provider.key != "azure":
            model = kept
        # A scripted answers.model naming a known-invalid id is repaired here too (interactive
        # non-Azure entry leaves model None → default_model, which is already valid).
        model, bad_model_note = credentials.sanitize_model(model, source=provider.source)
        if bad_model_note:
            io.warn(bad_model_note)
        if kept and model and provider.key != "azure":
            io.note(f"  Keeping your current model {model} (SOG_LLM in {constants.DOTENV_REL}).")

        # Azure's "model" is the *deployment* name baked into the request URL — there is no
        # universal default, so a silent fall-back to `default_model` (gpt-4o) points the agent
        # at a deployment that may not exist (→ an inference-time 404 that reads as "still
        # broken"). Prompt for it interactively, pre-filled from a full inference endpoint when
        # the user pasted one. Scripted runs supply it via `answers.model` (unchanged).
        if provider.key == "azure" and model is None and llm_answers is None:
            guessed = credentials.azure_deployment_from_endpoint(values.get("OPENAI_ENDPOINT", ""))
            if guessed is None and not kept:
                io.note(
                    f"  Your endpoint URL has no '/deployments/<name>/' path, so I can't detect the "
                    f"deployment. '{provider.default_model}' is only a placeholder — enter the exact "
                    f"deployment name from your Azure resource, or the first call will 404."
                )
            model = (
                io.ask_text(
                    "  Azure deployment name (exactly as named in your Azure resource)",
                    default=guessed or kept or provider.default_model,
                )
                or None
            )

        # A present-but-null `validate:` in an --answers file must not read as false (R24-A1), and a
        # quoted "false" must not read as true (u35-setup-ux-11). Absent and present-null both mean
        # "default: validate".
        should_validate = _answer_flag(llm_answers, "validate", True)
        if should_validate:
            result = _validate(io, provider, values)
            validated = _validated(result)
        else:
            # Explicitly skipped (scripted/offline): proceed, but do NOT claim it was validated.
            result = llm_setup.PingResult(ok=True, detail="(skipped)")
            validated = False

        if result.ok:
            if persist:
                _persist(io, provider, values, model, knobs)
                if validated:
                    _remember_key(io, provider, values, model)
            else:
                io.note(f"[dry-run] would save {provider.label} key to {constants.DOTENV_REL} (nothing written)")
            _handoff(io)
            return LLMChoice(
                source=provider.source, model=model or provider.default_model, field_values=values, validated=validated
            )

        # validation failed — decide what to do next
        if llm_answers is not None:
            action = on_fail_default
        else:
            action = _ask_validation_branch(io)

        if action == OnValidationFail.CONTINUE_UNVALIDATED:
            io.warn("Continuing with an unvalidated key — the agent may fail later.")
            if persist:
                _persist(io, provider, values, model, knobs)
            else:
                io.note(f"[dry-run] would save {provider.label} key to {constants.DOTENV_REL} (nothing written)")
            _handoff(io)
            return LLMChoice(
                source=provider.source,
                model=model or provider.default_model,
                field_values=values,
                validated=False,
                on_validation_fail=action,
            )

        # Scripted runs can't re-prompt, so neither RETRY (re-enter the *same* fields)
        # nor SWITCH (re-select the *same* pinned provider) can make progress — either
        # would loop forever. Both are a terminal give-up with a clear error; only
        # CONTINUE_UNVALIDATED (handled above) can proceed non-interactively.
        if llm_answers is not None:
            # Genuinely terminal: retry/switch both need an interactive prompt we don't have in a scripted
            # run. Returning a choice here (as this did) marked onboarding DONE and let the run build every
            # tool env and exit 0 with a dead/unvalidated key AND without persisting it — dishonest green on
            # the flagship --answers CI path, and the "aborting" message was simply false. Raise the same
            # AnswersError _choose_provider uses for an unrecoverable scripted problem; cli maps it → exit 2.
            from .answers import AnswersError

            raise AnswersError(
                f"scripted LLM validation failed and on_validation_fail='{action.value}' cannot proceed "
                "non-interactively (retry/switch both need an interactive prompt). Set "
                "on_validation_fail: continue_unvalidated to build the tool envs anyway, or fix the key."
            )

        if action == OnValidationFail.SWITCH:
            continue  # interactive: re-choose a different provider
        # RETRY (interactive): same provider, re-enter the fields — loop


def _ask_validation_branch(io: PromptIO) -> OnValidationFail:
    choice = io.select(
        "  What next?",
        [
            ("retry", "Re-enter the key", "same provider"),
            ("switch", "Switch provider", "pick a different one"),
            ("continue_unvalidated", "Continue anyway", "save the key without validating"),
        ],
        default="retry",
    )
    return OnValidationFail.coerce(choice)


def _handoff(io: PromptIO) -> None:
    io.handoff("1. LLM connected — handing you to the SpatialOmicsLab guide")
