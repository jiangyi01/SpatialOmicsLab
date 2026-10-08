"""
Interactive decision source — the Stage-B "hands-on guide".

This is the experience the onboarding stage hands off to the instant the LLM key
validates: a conversational orchestrator that turns each raw menu into a
recommendation. It implements the same
:class:`~sog_install.decisions.DecisionSource` contract as
:class:`~sog_install.answers.ScriptedSource`, so the wizard driver and
engines can't tell a human-guided run from a scripted one.

How each decision is made:

* **LLM recommends** — one bounded :meth:`ChatClient.chat_json` call feeds the
  provider the context (preflight, category descriptions, current env, the
  user's stated goals) and gets back a small JSON object carrying advice +
  the concrete choice. The advice (``say``/``rationale``) is printed; the choice
  becomes a ``decisions.py`` object.
* **Graceful degradation** — any :class:`LLMChatError` (network/quota, a
  malformed reply that survived :meth:`chat_json`'s own parse-retry, or a
  ``Bedrock`` source the stdlib client can't drive) drops that one decision to a
  deterministic **menu-with-advice** over the same context. Setup never stalls.

Two things are deliberately **never** delegated to the LLM:

* :meth:`collect_service_key` — secrets are read straight from the user via
  :class:`PromptIO`; a tool credential is never sent to the provider.
* :meth:`confirm` — the propose→confirm→execute gate is a plain yes/no the LLM
  cannot bypass. The driver executes a side effect only when this returns True.

Per-decision the guide is single-shot (recommend → return); the user's freeform
intent is captured once upstream into ``ctx.stated_goals`` and any "let me change
that" happens at the confirm gate or by the driver re-invoking. Stdlib only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import credentials
from .constants import PROTECTED_ENVS, _validate_env_name, base_env_protection_reason
from .decisions import (
    BaseEnvDecision,
    BaseEnvMode,
    CategorySelection,
    CleanupDecision,
    DecisionSource,
    GuideContext,
    OnToolFail,
    Proposal,
    ProvisionDecision,
    TestDecision,
    ToolPlan,
    categories_for_servers,
    expand_category_servers,
)
from .llm_chat import ChatClient, LLMChatError, LLMUnsupported

if TYPE_CHECKING:
    from .prompts import PromptIO
    from .session_log import SessionLog


def _model_not_found_hint(detail: str, source: str | None = None) -> str | None:
    """When a guide LLM call fails specifically because the configured model doesn't exist (rather
    than a network / auth / quota problem), return an actionable one-liner naming the fix; otherwise
    ``None``.

    This generalizes the stale-model repair: the one *known*-bad default is auto-migrated
    (:func:`credentials.sanitize_model`), but a user who hand-typed some *other* invalid id into
    ``SOG_LLM`` still lands here, and a bare "HTTP 404" is not self-explanatory. The example model is
    the user's own provider default (falling back to the menu default) so the suggestion is valid for
    their key."""
    d = detail.lower()
    if not ("model_not_found" in d or "not_found_error" in d or ("404" in d and "model" in d)):
        return None
    try:
        example = (
            credentials.get_provider(source).default_model if source else credentials.default_provider().default_model
        )
    except KeyError:
        example = credentials.default_provider().default_model
    return (
        "→ That looks like an invalid model name rather than a connection problem. Set SOG_LLM in "
        f"your .env to a model your key can access (e.g. {example}) and re-run to restore the "
        "AI-assisted guide."
    )


def _auth_quota_hint(detail: str) -> str | None:
    """When a guide LLM call fails on authentication (HTTP 401/403) or rate/quota (HTTP 429), return
    an actionable one-liner; otherwise ``None``.

    The guide phase is often the FIRST time a key that onboarding could only *presence*-check (a
    Custom / Azure / Ollama endpoint we can't deep-validate offline) is actually exercised against the
    provider — so a bad or throttled key surfaces here, not at onboarding. This names that cause
    specifically, distinct from a wrong *model* id (:func:`_model_not_found_hint`) or a transient
    network blip (which get no hint — nothing for the user to fix). Matches the provider error strings
    ``llm_chat`` raises (``"HTTP 401 from OpenAI: ..."``) plus the common provider error-body tokens."""
    d = detail.lower()
    if any(t in d for t in ("http 401", "http 403", "unauthorized", "invalid_api_key", "authentication")):
        return (
            "→ Your API key looks invalid or unauthorized. Check the key (and its permissions) in your "
            ".env, then re-run to restore the AI-assisted guide — the menus below still work."
        )
    if any(t in d for t in ("http 429", "rate limit", "rate_limit", "quota", "insufficient_quota")):
        return (
            "→ The provider is rate-limiting or you're out of quota. The menus below still work; re-run "
            "later to restore the AI-assisted guide."
        )
    return None


def _as_str(value: object) -> str:
    """Coerce an LLM-supplied field that *should* be a string into one, never raising.

    ``chat_json`` guarantees the reply is a ``dict``, but NOT that each field has the type the prompt
    asked for — a model can return ``"say": {...}`` or ``"name": 3``. A bare ``.strip()`` on a dict
    then raises ``AttributeError`` and aborts the whole wizard *past* its menu-fallback safety net
    (the ``choose_*`` methods are called unwrapped by the driver), discarding envs already built. A
    ``dict``/``list``/``None`` is treated as 'no usable value' (empty) so the caller falls back to its
    default; a number/bool is stringified."""
    if value is None or isinstance(value, (dict, list)):
        return ""
    return value if isinstance(value, str) else str(value)


def _as_str_list(value: object) -> list[str]:
    """Coerce an LLM-supplied field that *should* be a list-of-strings into one, never raising.

    Tolerates a bare string (wrapped to a one-item list), a number/bool/None/dict (→ empty), and a
    list with non-string members (each coerced via :func:`_as_str`, blanks dropped). A mistyped
    ``categories``/``servers`` field thus degrades to 'nothing usable' — the caller then falls back to
    its deterministic default — instead of a ``TypeError`` from ``list(3)`` aborting the wizard."""
    if isinstance(value, str):
        value = [value]
    elif not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        s = _as_str(item).strip()
        if s:
            out.append(s)
    return out


# Conda env-name validation (N15 / #100): the canonical rule now lives in ``constants``
# (``_validate_env_name`` / ``_ENV_NAME_RE``) — a stdlib leaf — so the interactive guide here, the
# scripted ``--answers`` path (``answers._validate``), and any future entry point share ONE
# definition. Re-exported at the top of this module for ``_ask_new_env_name`` + the #100 guard.


# --------------------------------------------------------------------------- #
# System preambles (one per decision; kept short — tokens are bounded)
# --------------------------------------------------------------------------- #
_ROLE = (
    "You are the setup guide for SpatialOmicsLab, an agent platform for spatial "
    "transcriptomics. You help a user who just cloned the repo stand it up. Be "
    "concise and practical; recommend the smallest choice that serves their goal."
)

_SYS_CATEGORIES = (
    _ROLE + " Choose analysis-tool categories. Reply JSON: "
    '{"say": str, "categories": [category-name,...], '
    '"servers": [tool-key,...] or null, "rationale": str}. '
    "Put every category you recommend in `categories`. Use `servers` ONLY to "
    "narrow to a lighter starter subset of those categories' tools; null means "
    "all tools in the chosen categories."
)

_SYS_BASE_ENV = (
    _ROLE + " Decide the base conda env. Reply JSON: "
    '{"say": str, "mode": "new" or "reuse", "name": str, '
    '"install_editable": bool, "rationale": str}. '
    "Prefer `new` (a fresh named env) unless the user clearly wants to reuse the "
    "current env. `name` is a short lowercase identifier; every tool env is named "
    "<name>_<tool>."
)

_SYS_TESTS = (
    _ROLE + " Decide how to test the freshly built envs. Reply JSON: "
    '{"say": str, "tier1": bool, "tier2": bool, '
    '"categories": [category-name,...], "eval": bool}. '
    "tier1 = fast worker check on a mini dataset (recommended true). tier2 = a "
    "real agent Q&A per category (needs the LLM key; slower)."
)

_SYS_CLEANUP = (
    _ROLE + " The run finished. Decide whether to delete the temporary test files this "
    "run created (the built conda envs are never touched). Reply JSON: "
    '{"say": str, "delete_artifacts": bool}.'
)


class InteractiveGuide(DecisionSource):
    """LLM-guided :class:`DecisionSource` with a deterministic menu fallback."""

    def __init__(
        self,
        io: PromptIO,
        chat_client: ChatClient | None = None,
        log: SessionLog | None = None,
    ) -> None:
        self.io = io
        self.chat = chat_client
        self.log = log
        # Print the "why I dropped to menus" explanation at most once per run: each of the ~4
        # decisions degrades independently on a bad key/model, and four identical scary lines read
        # as four separate failures (G2).
        self._fallback_noted = False
        # Sticky circuit-breaker (G2): once one decision has hard-failed, stop calling the model for
        # the rest of the run — the usual cause fails every decision, so retrying just adds latency.
        self._degraded = False

    # -- LLM plumbing ---------------------------------------------------------
    def _ask_json(self, system: str, user: str) -> dict | None:
        """One bounded JSON exchange, or ``None`` to signal 'fall back to menu'."""
        # Once any decision has degraded, don't call the model again (G2). The dominant causes —
        # no/bad key, an unreachable or unsupported endpoint — fail *every* decision identically, so
        # re-asking the remaining ~3 times only piles on the per-call timeout (a dead endpoint hangs
        # each time) and duplicate ``guide_fallback`` log events. ``chat_json`` has already spent its
        # own internal parse-retries before raising, so the model has had its chances on this run.
        if self.chat is None or self._degraded:
            return None
        try:
            return self.chat.chat_json(system, user)
        except LLMChatError as exc:
            # Covers network/quota, a reply that failed parse-retry, and Bedrock / an unknown source
            # (LLMUnsupported). Trip the breaker and degrade this decision to a menu, explaining why —
            # once per run, in cause-specific words, with an actionable hint when we can name the fix.
            self._degraded = True
            self._note_fallback(exc)
            if self.log is not None:
                self.log.event("guide_fallback", detail=str(exc))
            return None

    def _note_fallback(self, exc: LLMChatError) -> None:
        """Explain — ONCE per run — why the AI guide dropped to menus, worded to the actual cause,
        plus a model / auth / quota hint when we can name the fix (G2/G3).

        Only the first fallback speaks; the menus that follow are self-evidently menus, so repeating
        the notice per decision would just read as a cascade of failures. Never raises: a display
        hiccup must not turn a graceful degrade into a crash."""
        if self._fallback_noted:
            return
        self._fallback_noted = True
        detail = str(exc)
        low = detail.lower()
        source = getattr(self.chat, "source", None)
        # ``chat_json`` wraps every non-LLMUnsupported failure as "could not obtain JSON after N
        # tries: <real cause>", so branch on the real cause carried inside — a network drop, a bad
        # key, and a genuinely-malformed reply each deserve different words (and a network blip
        # deserves no "fix your key" hint). All three share "using quick menus" so the notice reads
        # consistently no matter the cause.
        if isinstance(exc, LLMUnsupported):
            # Bedrock / an unknown source the stdlib client can't drive — not transient, so don't
            # imply "try again": the menus are the path forward for this provider.
            self.io.note(f"(the AI-assisted guide can't drive {source or 'this provider'} — using quick menus instead)")
        elif "could not reach" in low:
            self.io.note("(couldn't reach the model — using quick menus instead; check your connection)")
        elif "non-json reply" in low:
            self.io.note("(the model didn't return usable JSON — using quick menus instead)")
        else:
            self.io.note(f"(the AI guide couldn't use the model — using quick menus instead: {detail})")
        hint = _model_not_found_hint(detail, source) or _auth_quota_hint(detail)
        if hint:
            self.io.note(hint)

    def _present(self, rec: dict) -> None:
        say = _as_str(rec.get("say")).strip()
        if say:
            self.io.say(f"🧬 {say}")

    # -- categories -----------------------------------------------------------
    def choose_categories(self, ctx: GuideContext) -> CategorySelection:
        """LLM pre-fills, the human always confirms.

        The model's goal-based suggestion (when a key is configured) becomes the
        *pre-selection* of the grouped per-tool picker. The picker opens as an
        *accordion*: those recommended tools seed a live basket (shown ``★``-marked
        and ``✓``-checked) and their category is opened first, so the user accepts
        with a single Enter — or toggles tools by number, opens other categories by
        their letter, and finishes with ``done``. With no model / a failed reply, the
        deterministic recommendation is used instead, so the basket still starts with
        a sensible small set pre-checked — even with no key.
        """
        suggested: list[str] = []
        rec = self._ask_json(_SYS_CATEGORIES, self._ctx_categories(ctx))
        if rec is not None:
            # Coerce both fields (G1): a model can return ``categories`` as a bare string or a number
            # and ``servers`` as a non-list. Left raw, ``list(3)`` / iterating a non-list raises and
            # aborts the wizard *past* its menu safety net. A mistyped field → empty → the goal-first
            # deterministic recommendation below still seeds a sensible basket. ``servers`` empty means
            # "all tools in the chosen categories" (``expand_category_servers``'s ``None`` branch).
            cats = _as_str_list(rec.get("categories"))
            suggested = expand_category_servers(cats, _as_str_list(rec.get("servers")) or None, ctx.categories)
            self._present(rec)
        # The LLM's suggestion wins when present and mappable; otherwise fall back to the
        # deterministic, goal-first recommendation the wizard computed (never empty), so the
        # picker always opens with a sensible small set pre-marked — even with no key.
        preselected = suggested or list(ctx.extra.get("recommended") or [])
        return self._pick_tools(ctx, preselected=preselected)

    def _pick_tools(self, ctx: GuideContext, *, preselected: list[str] | None = None) -> CategorySelection:
        """The grouped, all-tools multi-select. Falls back to the category-level
        menu only when no per-tool grouping is available (a bare launcher env)."""
        if not ctx.tool_groups:
            return self._menu_categories(ctx)
        # Advisory only: if preflight saw no usable GPU, tell the user the deep-learning
        # tools will fall back to CPU — it never changes what is offered or pre-selected.
        if (ctx.preflight or {}).get("gpu", {}).get("level") == "warn":
            self.io.note("No GPU detected — GPU tools still install and run, just slower on CPU.")
        chosen = self.io.grouped_multiselect(
            "  Which tools should I install?",
            ctx.tool_groups,
            minimum=1,
            preselected=preselected,
            collapsible=True,
        )
        cats = categories_for_servers(chosen, ctx.categories)
        return CategorySelection(categories=cats, servers=chosen, rationale="menu:tools")

    def _menu_categories(self, ctx: GuideContext) -> CategorySelection:
        options = [(c["name"], c["name"], c.get("description", "")) for c in ctx.categories]
        if not options:
            # G3: this IS the safety-net path (reached when the LLM degraded *and* there are no
            # tool groups). Raising ``LLMChatError`` here escapes past ``choose_categories`` — which
            # does NOT wrap the menu — and aborts the whole wizard, exactly what the fallback exists
            # to prevent. With no category metadata there is simply nothing to offer, so degrade to an
            # empty selection: the base env still installs (a launcher-only setup), and the user can
            # add analysis tools later. Never crash the run over a missing catalog.
            self.io.note("(no analysis-tool catalog found — setting up the base environment only)")
            return CategorySelection(categories=[], servers=[], rationale="menu:empty")
        chosen = self.io.multiselect("  Which analysis categories?", options, minimum=1)
        servers = expand_category_servers(chosen, None, ctx.categories)
        return CategorySelection(categories=chosen, servers=servers, rationale="menu")

    # -- base env -------------------------------------------------------------
    def choose_base_env(self, ctx: GuideContext) -> BaseEnvDecision:
        rec = self._ask_json(_SYS_BASE_ENV, self._ctx_base_env(ctx))
        if rec is None or not rec.get("name"):
            return self._menu_base_env(ctx)
        try:
            mode = BaseEnvMode.coerce(rec.get("mode", "new"))
        except ValueError:
            return self._menu_base_env(ctx)
        name = str(rec["name"]).strip()
        if mode is BaseEnvMode.REUSE:
            # Reuse must target the active env, not a name the model invented — else the install
            # / `conda run -n <name>` lands on an env that doesn't exist. No active env → fall to
            # the menu (which won't even offer reuse).
            if not ctx.current_env:
                return self._menu_base_env(ctx)
            name = ctx.current_env
            # M1: reusing means INSTALLING into the active env. If that env is reserved (e.g. the
            # user launched sog-setup from `base` or `spatialomicsgym_env`), route to the menu, which
            # now offers only "create fresh" — never silently mutate a protected env.
            if name in PROTECTED_ENVS:
                self.io.note(f"  the active env {name!r} is reserved — let's create a fresh one instead")
                return self._menu_base_env(ctx)
        elif _validate_env_name(name) is not None:
            # NEW mode: the model can invent a name conda rejects (spaces, slashes, a leading dot),
            # which would otherwise surface much later as a raw `conda create` CondaError exit-3.
            # The typed menu path already validates + re-asks with a plain reason — route through it
            # rather than trusting the suggestion (mirrors the bad-mode / reuse-without-active-env
            # fallbacks above; #100, same guard as _ask_new_env_name's N15).
            self.io.note("  the suggested env name isn't a valid conda name — let's pick one together")
            return self._menu_base_env(ctx)
        elif base_env_protection_reason(name) is not None:
            # M1: NEW mode targeting a reserved env (the model invented `base`/`spatialomicsgym_env`/…).
            # Creating into it would mutate a protected env — route to the menu for a safe fresh name.
            self.io.note(f"  {name!r} is a reserved env — let's pick a fresh name instead")
            return self._menu_base_env(ctx)
        self._present(rec)
        return BaseEnvDecision(
            mode=mode,
            basic_env_name=name,
            install_editable=bool(rec.get("install_editable", True)),
            rationale=rec.get("rationale", "guide"),
        )

    def _menu_base_env(self, ctx: GuideContext) -> BaseEnvDecision:
        cur = ctx.current_env
        options = [("new", "Create a fresh env", "recommended — leaves your setup untouched")]
        if cur and cur not in PROTECTED_ENVS:
            # "Reuse" is only meaningful when there IS an active env to reuse — and never for a
            # reserved env (M1): reusing installs into the active env, which must never modify a
            # protected env like `base` or `spatialomicsgym_env`. Offer only "create fresh" there.
            options.append(("reuse", "Reuse the current env", f"install missing core pkgs into {cur}"))
        mode = self.io.select("  Base environment?", options, default="new")
        if mode == "reuse":
            # Reuse targets the *active* conda env itself — never a typed name. Prompting for a
            # name here (and using it) is what made a run pick a non-existent env, so the
            # downstream `conda run -n <name>` / install failed. `cur` is truthy here (reuse is
            # only offered when it is).
            name = cur
            self.io.note(f"reusing the active env {name!r}")
        else:
            name = self._ask_new_env_name()
        editable = self.io.ask_yesno(
            "  Install the SpatialOmicsLab agent into this environment so you can run it?", default=True
        )
        return BaseEnvDecision(
            mode=BaseEnvMode.coerce(mode),
            basic_env_name=name,
            install_editable=editable,
            rationale="menu",
        )

    def _ask_new_env_name(self) -> str:
        """Prompt for a NEW base-env name, re-asking on an invalid conda name (N15).

        A name with a space/slash/leading-dot would otherwise be accepted and fail much later as a
        raw ``CondaError`` exit-3; validate it here and re-ask with a plain reason. A run that can't
        be re-asked (non-interactive, nothing scripted) falls back to the safe default rather than
        spinning — setup never stalls."""
        while True:
            name = self.io.ask_text("  Name for the new env", default="sog")
            # Reject both a syntactically-bad conda name (N15) and a reserved env name (M1 — creating
            # into it would mutate a protected env); re-ask with a plain reason either way.
            reason = _validate_env_name(name) or base_env_protection_reason(name)
            if reason is None:
                return name
            self.io.warn(f"  {reason}")
            if self.io.non_interactive:
                self.io.note("  using the default env name 'sog' instead")
                return "sog"

    # -- provisioning ---------------------------------------------------------
    def plan_provision(self, ctx: GuideContext) -> ProvisionDecision:
        """Build EVERY tool the user picked — no goal-based narrowing.

        The candidates are exactly the servers the user just confirmed in the
        grouped picker (``state.selected_servers`` → ``_candidate_tools``), so an
        explicit pick is an explicit instruction: install all of them, even ones
        outside the stated goal. We never let the model prune the list here — that
        is what silently dropped 3 of 4 picked tools. The failure policy stays
        ``continue`` so one bad build never sinks the others; the per-tool confirm
        gate still lets the user decline an individual build.
        """
        candidates: list[ToolPlan] = ctx.extra.get("candidate_tools", [])
        if candidates:
            n = len(candidates)
            self.io.say(f"🧬 Building all {n} selected tool{'s' if n != 1 else ''}; I'll keep going if one fails.")
        return self._menu_provision(candidates)

    def _menu_provision(self, candidates: list[ToolPlan]) -> ProvisionDecision:
        # The plans are pre-built; we simply build them all, continuing past any
        # single failure. Per-plan confirmation still happens at the gate.
        return ProvisionDecision(tools=list(candidates), on_tool_fail=OnToolFail.CONTINUE)

    # -- contextual service keys (NEVER via the LLM) --------------------------
    def collect_service_key(self, env_var: str, ctx: GuideContext) -> str | None:
        spec = credentials.SERVICE_KEY_BY_VAR.get(env_var)
        label = spec.label if spec else env_var
        where = f" — get one at {spec.where_to_get}" if spec and spec.where_to_get else ""
        self.io.note(f"One selected tool needs {label} ({env_var}){where}.")
        val = self.io.ask_secret(f"  Paste {env_var} (enter to skip this one tool)", allow_empty=True)
        return val or None

    # -- tests ----------------------------------------------------------------
    def choose_tests(self, ctx: GuideContext) -> TestDecision:
        default_cats = ctx.selected.categories if ctx.selected else []
        rec = self._ask_json(_SYS_TESTS, self._ctx_tests(ctx))
        if rec is None:
            return self._menu_tests(ctx, default_cats)
        self._present(rec)
        return TestDecision(
            run_tier1=bool(rec.get("tier1", True)),
            run_tier2=bool(rec.get("tier2", False)),
            # G1: coerce ``categories`` (a model may return a bare string / number); a mistyped value
            # → empty → fall back to the categories the user actually selected, never a TypeError.
            categories_to_test=_as_str_list(rec.get("categories")) or default_cats,
            tier2_eval=bool(rec.get("eval", False)),
        )

    def _menu_tests(self, ctx: GuideContext, default_cats: list[str]) -> TestDecision:
        tier1 = self.io.ask_yesno("  Run fast worker tests on the mini dataset?", default=True)
        tier2 = self.io.ask_yesno("  Also run a real agent Q&A per category? (needs the key)", default=False)
        return TestDecision(run_tier1=tier1, run_tier2=tier2, categories_to_test=default_cats)

    # -- cleanup --------------------------------------------------------------
    def choose_cleanup(self, ctx: GuideContext) -> CleanupDecision:
        rec = self._ask_json(_SYS_CLEANUP, self._ctx_cleanup(ctx))
        if rec is None:
            delete = self.io.ask_yesno("  Delete the temporary test files this run created?", default=False)
            return CleanupDecision(delete_artifacts=delete)
        self._present(rec)
        return CleanupDecision(delete_artifacts=bool(rec.get("delete_artifacts", False)))

    # -- the confirm gate (deterministic; LLM cannot bypass) ------------------
    def confirm(self, proposal: Proposal, ctx: GuideContext) -> bool:
        size = f" (~{proposal.est_gb:.1f} GB)" if proposal.est_gb else ""
        rev = "" if proposal.reversible else "  ⚠️  not easily reversible"
        self.io.say(f"  Plan → {proposal.action}: {proposal.detail}{size}{rev}")
        return self.io.ask_yesno("  Proceed?", default=True)

    # -- ad-hoc Q&A -----------------------------------------------------------
    def ask(self, question: str, ctx: GuideContext) -> str:
        if self.chat is None:
            return ""
        try:
            return self.chat.chat(
                [{"role": "user", "content": question}],
                system=_ROLE + " Answer the user's question briefly and concretely.",
            )
        except LLMChatError:
            return ""

    # -- context serializers --------------------------------------------------
    def _ctx_categories(self, ctx: GuideContext) -> str:
        lines = ["Available categories:"]
        for c in ctx.categories:
            # `.get("tools") or []`, not `.get("tools", [])`: a present-but-null `tools:` yields None
            # (the 2-arg default only fires on an ABSENT key), and `", ".join(None)` raises TypeError —
            # which would escape this method's menu-fallback safety net and abort the wizard. Matches the
            # convention decisions.py already documents for these same category dicts. (R27 A-F1)
            tools = ", ".join(c.get("tools") or []) or "(no tools listed)"
            lines.append(f"- {c.get('name')}: {c.get('description', '')} [tools: {tools}]")
        lines.append(f'\nUser goal: "{ctx.stated_goals or "(not stated)"}"')
        # Thread the machine's own limits so the model recommends a set that actually
        # fits this box (advisory; silent when preflight hasn't run yet).
        pf = ctx.preflight or {}
        gpu, disk = pf.get("gpu", {}), pf.get("disk", {})
        if gpu:
            lines.append(f"Machine GPU: {gpu.get('detail', 'unknown')} ({gpu.get('level', '?')}).")
        if disk:
            lines.append(f"Machine disk: {disk.get('detail', 'unknown')} ({disk.get('level', '?')}).")
        lines.append("Prefer a small, buildable starter set that fits this machine; the user can add more later.")
        return "\n".join(lines)

    def _ctx_base_env(self, ctx: GuideContext) -> str:
        sel = ", ".join(ctx.selected.categories) if ctx.selected else "(none yet)"
        return (
            f'User goal: "{ctx.stated_goals or "(not stated)"}"\n'
            f"Current conda env: {ctx.current_env or '(unknown)'}\n"
            f"Selected categories: {sel}\n"
            "Recommend a base env mode and a short name."
        )

    def _ctx_tests(self, ctx: GuideContext) -> str:
        sel = ", ".join(ctx.selected.categories) if ctx.selected else "(none)"
        return f"Selected categories: {sel}\nRecommend a testing depth."

    def _ctx_cleanup(self, ctx: GuideContext) -> str:
        return "Setup finished. Recommend whether to keep the test artifacts."
