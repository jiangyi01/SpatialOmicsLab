"""Pipeline integration - connects tuning to benchmark runner and STCoscientist agent.

Provides:
- build_tool_command(): Construct CLI commands for direct tool execution
- explain_tool_command(): The same, with the reason when no command can be built
- inject_tuned_params(): Override MCP tool parameters with tuned values
- registry_fixed_params(): The values a benchmark dataset fixes for every trial of a tool
- TunedBenchmarkRunner: Wrapper around benchmark_runner with tuning support
"""

from __future__ import annotations

import ast
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from spatialomicsgym import platform_root
from spatialomicsgym.mcp_config_path import describe_search, find_mcp_config
from spatialomicsgym.tuning.core import TuningMode

logger = logging.getLogger(__name__)

# The agent part (``<repo>/agent``), which holds ``tools/``, ``tools_user/`` and ``benchmarks/``.
PROJECT_ROOT = Path(__file__).parent.parent.parent

# The vocabulary `get_tunable_tools` types a tool with. Two dialects, because it reads two things.
#
# The prose lists are matched against a tool's `description`, which is English, so every entry has
# to be spelled the way English spells it. Four entries used to be snake_case -- `spatial_domain`,
# `spatially_variable`, `variable_gene`, `cell_type_mapping` -- and matched nothing at all across
# all 114 registered descriptions, while their prose spellings hit 12, 9, 9 and 19. That dead
# quarter of the vocabulary is most of why the majority of tools typed as `unknown`.
_CLUSTERING_PROSE = ("clustering", "domain", "segmentation", "spatial domain")
_SVG_PROSE = ("svg", "spatially variable", "variable gene", "hotspot", "autocorrelation")
# `cell-type annotation` rather than a bare `annotation`: the bare word was carrying two genuine
# label-transfer tools and three false ones, where it named an input file list, an input column,
# and a config key. Qualifying it keeps the two and drops the three.
_DECONV_PROSE = (
    "deconvolution",
    "deconvolve",
    "cell type mapping",
    "cell-type annotation",
    "cell type annotation",
    "proportion",
)

# The same vocabulary matched against whole underscore-separated segments of the tool's own name.
# A name segment is the tool declaring what it does; a description is prose that also names
# prerequisites ("1) run <pkg>_deconvolution first"), input columns ("an annotation/cluster
# column") and, in one case, a fraction of spectral energy called a "proportion". Where the two
# disagree the name is right: of the 30 registered tools with a task word in their name, 27
# already agreed with their prose and all three disagreements were the prose.
#
# Segments, not substrings -- a substring rule reads `cluster` out of tangram's prose and types a
# mapping tool as clustering. `cluster` earns a place here that it cannot have in _CLUSTERING_PROSE
# for exactly that reason.
_CLUSTERING_NAME_WORDS = frozenset({"clustering", "cluster", "domain", "domains", "segmentation"})
_SVG_NAME_WORDS = frozenset({"svg", "hotspot"})
_DECONV_NAME_WORDS = frozenset({"deconvolution", "deconvolve", "deconv"})


def load_mcp_config(
    config_path: str | None = None,
) -> dict[str, Any]:
    """Load and parse the MCP config file.

    With no argument, resolves through :func:`spatialomicsgym.mcp_config_path.find_mcp_config`
    rather than ``PROJECT_ROOT``: ``MCP_server/`` is not in the wheel, so on a non-editable
    install the package-relative path does not exist and this used to raise a bare
    ``FileNotFoundError`` naming a site-packages directory the user never chose.
    """
    if config_path is None:
        resolved = find_mcp_config()
        if resolved is None:
            raise FileNotFoundError(describe_search())
        config_path = str(resolved)

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_tool_server_info(tool_name: str, mcp_config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Find the MCP server and tool config for a given tool name."""
    if mcp_config is None:
        mcp_config = load_mcp_config()

    for server_key, server_cfg in mcp_config.get("mcp_servers", {}).items():
        if not isinstance(server_cfg, dict):
            continue
        for tool in server_cfg.get("tools", []):
            if tool.get("spatialomicsgym_name") == tool_name:
                return {
                    "server_key": server_key,
                    "server_command": server_cfg.get("command", []),
                    "server_env": server_cfg.get("env", {}),
                    "tool_config": tool,
                    "parameters": tool.get("parameters", {}),
                }
    return None


_FLAG_LITERAL_RE = re.compile(r"""["'](--[A-Za-z0-9][A-Za-z0-9_-]*)["']""")

# The ``--task`` each multi-task portal hardcodes into its own argv. No config block declares a
# ``task`` parameter for these tools -- the caller never chooses it, the portal does -- so gating
# the injection on one (as this used to) never fired, and graphst_worker.py and prost_worker.py
# declare ``--task`` required: every trial of four search-space tools exited 2 in argparse
# (hunt 2026-09-30, u32-tuning-1). Spelled as the portal spells it; prost's worker also takes
# ``index_svg``/``pnn`` as aliases, but the portal is what an ordinary call sends.
_TOOL_TASK_FLAGS: dict[str, str] = {
    "graphst_spatial_clustering": "clustering",
    "graphst_deconvolution": "deconvolution",
    "prost_index_svg": "index",
    "prost_pnn_domains": "domains",
    "stlearn_spatial_clustering": "clustering",
    "spagft_identify_svg": "svg",
}


@lru_cache(maxsize=256)
def _portal_worker_defaults(server_script: str) -> tuple[str, str, str] | None:
    """``(env_prefix, default_interpreter, default_worker)`` from the portal's ``get_worker_paths`` call.

    A portal resolves its launch as ``{PREFIX}_PYTHON``/``{PREFIX}_WORKER`` from its environment, and
    otherwise the defaults written into that call. When this was found (2026-09-30) 47 of the 95
    canonical server blocks declared neither, and a generated or hand-kept config still can; for
    such a block the defaults *are* the launch -- and the tuner, which read only the env block, ran
    the worker under a bare ``python`` (the agent env, no torch) and never found a ``.R`` worker at
    all (hunt 2026-09-30, u32-tuning-3). Read, not imported: nothing under ``spatialomicsgym/``
    imports ``tools/``. ``None`` when the call is absent or not all literals.
    """
    try:
        tree = ast.parse(Path(server_script).read_text(encoding="utf-8"), filename=server_script)
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")) != "get_worker_paths":
            continue
        names = ("env_prefix", "default_python", "default_worker")
        given = dict(zip(names, node.args, strict=False))
        given.update({k.arg: k.value for k in node.keywords if k.arg in names})
        values = [given.get(n) for n in names]
        if all(isinstance(v, ast.Constant) and isinstance(v.value, str) and v.value for v in values):
            return values[0].value, values[1].value, values[2].value
    return None


# Calls that hand a path on unchanged apart from normalising it -- the only transformations under
# which "the portal passes parameter P as flag F" still means "pass the tuner's value as F".
_PASS_THROUGH_CALLS = frozenset({"str", "Path", "PurePath", "expanduser", "abspath", "realpath", "fspath", "resolve"})


def _passed_through(expr: ast.AST, params: frozenset[str], local_of: dict[str, str | None]) -> str | None:
    """The portal parameter *expr* hands on unchanged (bar path normalisation), or ``None``."""
    if isinstance(expr, ast.Name):
        return expr.id if expr.id in params else local_of.get(expr.id)
    if not isinstance(expr, ast.Call) or expr.keywords:
        return None
    func = expr.func
    if (func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")) not in _PASS_THROUGH_CALLS:
        return None
    if len(expr.args) == 1:  # str(x), Path(x), os.path.expanduser(x)
        return _passed_through(expr.args[0], params, local_of)
    if not expr.args and isinstance(func, ast.Attribute):  # Path(x).expanduser()
        return _passed_through(func.value, params, local_of)
    return None


@lru_cache(maxsize=256)
def _portal_flag_map(server_script: str, function_name: str) -> dict[str, str]:
    """``{portal parameter: worker flag}`` for the parameters the portal passes through unchanged.

    The config names a portal's parameters; the portal translates some of them before the worker
    sees them -- ``spatial_h5ad_path`` becomes ``--spatial-h5ad``, ``sc_h5ad_path`` ``--sc-h5ad``.
    Spelling the config name mechanically finds no such flag, so the tuner refused the dataset of
    cell2location, destvi and sedr outright, and declared them untunable while the search spaces
    called cell2location a high-value target (hunt 2026-09-30, u32-tuning-14). Only a value that
    reaches the argv unchanged counts: a portal that stages its input (seurat converts an h5ad into
    a counts directory) or splits it (spiral needs two slices) is still, honestly, not drivable.
    """
    try:
        tree = ast.parse(Path(server_script).read_text(encoding="utf-8"), filename=server_script)
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return {}
    func = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
        ),
        None,
    )
    if func is None:
        return {}
    params = frozenset(a.arg for a in [*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs])
    # Locals assigned once from a passed-through parameter (``sc_path = str(Path(sc_h5ad_path)...)``).
    # A name assigned twice is ambiguous and is left out rather than guessed at.
    local_of: dict[str, str | None] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            local_of[name] = None if name in local_of else _passed_through(node.value, params, {})
    flags: dict[str, str] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.List):
            continue
        for flag, value in zip(node.elts, node.elts[1:], strict=False):
            if isinstance(flag, ast.Constant) and isinstance(flag.value, str) and flag.value.startswith("--"):
                param = _passed_through(value, params, local_of)
                if param is not None:
                    flags.setdefault(param, flag.value)
    return flags


@lru_cache(maxsize=256)
def _value_taking_flags(worker_path: str) -> frozenset[str]:
    """The declared flags that read a value off the argv, rather than being switches.

    ``_cli_spelling`` spelled every boolean as a switch, so ``--using-dec type=str`` (sedr) got a
    bare ``--using-dec`` for True -- argparse "expected one argument", exit 2 -- and nothing for False,
    which left the worker on its own default of True while the trial recorded False
    (hunt 2026-09-30, u32-tuning-21). A flag is a switch only when its ``action`` says so. Empty for
    a worker that is not Python: its flags are not readable this way, and the switch spelling stays.
    """
    try:
        tree = ast.parse(Path(worker_path).read_text(encoding="utf-8"), filename=worker_path)
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return frozenset()
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        action = next((k.value for k in node.keywords if k.arg == "action"), None)
        nargs = next((k.value for k in node.keywords if k.arg == "nargs"), None)
        if action is not None and not (
            isinstance(action, ast.Constant) and action.value in ("store", "append", "extend")
        ):
            continue  # a switch, or an action class whose arity cannot be read: keep the switch spelling
        if isinstance(nargs, ast.Constant) and nargs.value == 0:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flags.add(arg.value)
    return frozenset(flags)


@lru_cache(maxsize=256)
def _declared_worker_flags(worker_path: str) -> frozenset[str] | None:
    """Every option string the worker recognises, or ``None`` if they cannot be read.

    ``None`` callers must treat as "cannot tell" and fall back to the historical mapping -- reading
    it as "declares no flags" would refuse every command for that worker.
    """
    try:
        source = Path(worker_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    try:
        tree = ast.parse(source, filename=worker_path)
    except (SyntaxError, ValueError):
        # Not Python: seurat_worker.R walks commandArgs() by hand and stops with "Unknown argument"
        # on anything else. Every flag it accepts is still a quoted literal, so scan for those.
        # Over-collecting one from a help string just restores the old permissive behaviour for that
        # flag, while under-collecting would refuse a runnable tool -- so this stays deliberately
        # broad. Verified against seurat_worker.R: all 35 dispatch keys found, `--data-path` (which
        # it rejects) correctly absent.
        return frozenset(_FLAG_LITERAL_RE.findall(source)) or None
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"):
                    flags.add(arg.value)
    return frozenset(flags) or None


def _cli_value(value: Any) -> str:
    """The single argv token that carries *value* to a worker.

    A list is spelled comma-separated. That is the convention every worker in ``tools/`` that reads a
    list from its CLI implements -- ``spacel_worker.py``, ``stage_worker.py``, ``st_gears_worker.py``
    and ``data_converter_worker.py`` all split on a comma, and ``build_tool_command``'s own note on
    the multi-slice input names says so in prose. Plain ``str()`` would hand them a Python repr whose
    brackets survive the split. (``st_gears`` used to fail ``float("[0.8")`` and substitute its own
    hardcoded list; since 2026-09-29 it reads a bracketed repr and refuses any other unparseable list.)

    A dict has no such convention here -- no reachable tool declares one -- so it is left to ``str()``
    rather than guessing at an encoding no worker has been shown to parse.
    """
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _cli_spelling(
    param: str,
    value: Any,
    declared: frozenset[str],
    value_flags: frozenset[str] = frozenset(),
) -> tuple[list[str], str | None]:
    """How to spell one parameter for a worker declaring *declared*.

    Returns ``(args, unreachable_reason)``. ``([], None)`` is a real answer: it means the requested
    value is what the worker already does when the flag is absent.

    A config parameter name is the *portal's*; the worker's CLI is a separate namespace, so each
    spelling has to be checked against what the worker declares. Both separator styles are tried
    because workers are inconsistent (``run_miso`` declares ``--n_clusters``, most declare
    ``--n-clusters``). *value_flags* are the declared flags that take a value: a boolean bound for
    one of those is spelled ``--flag True|False``, never as a switch (u32-tuning-21).
    """
    stems = dict.fromkeys((param.replace("_", "-"), param))  # ordered, deduped
    positive = next((f"--{s}" for s in stems if f"--{s}" in declared), None)
    negative = next((f"--no-{s}" for s in stems if f"--no-{s}" in declared), None)

    if isinstance(value, bool):
        if positive and positive in value_flags:
            return [positive, str(value)], None
        if value:
            if positive:
                return [positive], None
            if negative:
                # The worker offers only the disable form, so requesting the enabled behaviour is
                # spelled by saying nothing -- `action="store_false"` defaults to True.
                return [], None
            return [], "the worker declares no flag for this boolean"
        if negative:
            return [negative], None
        if positive:
            return [], None  # a store_true flag: omitting it *is* False
        return [], "the worker declares no flag for this boolean"

    if positive:
        return [positive, _cli_value(value)], None
    return [], "the worker declares no such flag"


def _worker_in_this_checkout(worker_path: str | None) -> str | None:
    """Adopt this clone's copy of a worker script whose configured path is absent.

    Every launcher in the repo reads the same ``*_WORKER`` values out of the same ``env`` blocks,
    and all 127 of them in the two shipped configs (41 canonical, 86 generated) are absolute paths
    naming one particular checkout. The MCP path repairs that -- ``base_mcp.resolve_worker_script``
    for the worker, ``agent.mcp_integration._rebase_script_args`` for the server script -- and this
    one did not, so a config that travels leaves the portal working and every tuning trial refused.
    ``sog-setup`` rewrites the canonical from the live resolution, so a freshly provisioned clone
    was never the case at issue; ``--keep-agent-config``, a checkout moved after setup, and a config
    copied from another box all are.

    Same policy as both twins, so a correctly-pinned config builds a byte-identical command: only an
    **absolute** ``.py``/``.R`` path that is **missing** is reconsidered, in favour of the
    identically-named file in this checkout's ``tools/`` or the sibling ``tools_user/`` the agent
    writes its own tools into -- ``tools/`` first, so a user-created tool cannot shadow a shipped
    worker. A path with no local counterpart is returned unchanged, so the caller's warning still
    names what was actually configured.

    Neither twin is imported: nothing under ``spatialomicsgym/`` imports ``tools/`` at all, and
    ``mcp_integration`` pulls in ``spatialomicsgym.agent``, hence the whole agent stack, to reach a
    private helper. The checkout root is read from ``__file__`` on each call rather than off the
    module-level ``PROJECT_ROOT``, because the question is where this module is now.
    """
    if not worker_path or not isinstance(worker_path, str):
        return worker_path
    if not worker_path.endswith((".py", ".R")) or not Path(worker_path).is_absolute():
        return worker_path
    if Path(worker_path).exists():
        return worker_path
    tools_dir = Path(__file__).resolve().parents[2] / "tools"
    basename = Path(worker_path).name
    directories = [tools_dir, tools_dir.parent / "tools_user"]
    # On a pip-only install the checkout-derived pair lands in site-packages and is not there;
    # the platform rungs (seeded SOG_HOME, the wheel's ``_platform`` copy) then serve the same
    # mirrored layout. On a checkout ``platform_root.tools_dir()`` IS ``tools_dir``, so this
    # appends nothing and the search is byte-identical to before.
    platform_tools = platform_root.tools_dir()
    if platform_tools is not None and platform_tools not in directories:
        directories += [platform_tools, platform_tools.parent / "tools_user"]
    for directory in directories:
        candidate = directory / basename
        if candidate.exists():
            return str(candidate)
    return worker_path


def _interpreter_on_this_box(interpreter: str) -> str:
    """Adopt this box's copy of a conda interpreter whose configured path is absent.

    The twin of :func:`_worker_in_this_checkout`, for the other half of the launch: that one repairs
    *what* is run, this one repairs *what runs it*. Every portal already does this
    (``base_mcp._resolve_worker_python``, called at ``base_mcp.py`` 514 and 566); the tuner read the
    same ``*_PYTHON``/``*_RSCRIPT`` values out of the same ``env`` blocks and put them straight into
    ``cmd``, where ``executor._run_tool`` swallows the resulting ``FileNotFoundError`` into a plain
    "trial failed" with no diagnosis. So a tool the agent can dispatch through MCP was untunable.

    Same three descending-authority steps as the portal, so a config both launchers can already use
    builds a byte-identical command: the branded/legacy env **alias** under the pinned root; the same
    env name under the root **this process** is running from; then both at once. The interpreter tail
    (``bin/python`` vs ``bin/Rscript``) is carried through untouched -- it names the language, so a
    python is never a substitute for an Rscript. An interpreter that exists, one that is relative
    (including the bare ``python`` default), one with no ``envs/`` segment, and one that resolves
    nowhere are all returned unchanged, so the operator is still told what they configured.

    Measured across the two shipped configs: 127 interpreters declared, 81 absent on this box, and
    exactly one (svca, whose ``spatialomicsgym_e1`` env is present here under its legacy name) that
    the portal repairs and this did not. The other 80 resolve nowhere under either launcher and must
    keep building a command -- declining them here would change what a failed trial reports.

    ``base_mcp`` is not imported: nothing under ``spatialomicsgym/`` imports ``tools/`` at all. The
    walk itself lives in :func:`sog_install.constants.interpreter_on_this_box` -- the
    stdlib-only module every package-side launcher can reach -- so the tuner and the agent-side R
    converter share one implementation rather than two hand-mirrored ones. That leaves exactly two
    copies in the repo, the package's and the portals', pinned against each other by
    ``test_cross_device_stability.py::TestTheTunerLaunchesTheInterpreterThisBoxHas``.
    """
    from sog_install.constants import interpreter_on_this_box

    return interpreter_on_this_box(interpreter)


def build_tool_command(
    tool_name: str,
    params: dict[str, Any],
    dataset_path: str | None = None,
    output_dir: str | None = None,
    sc_reference_path: str | None = None,
    mcp_config: dict[str, Any] | None = None,
) -> list[str] | None:
    """Build a CLI command to run a tool's worker script directly.

    This bypasses the MCP server and agent for efficient tuning evaluation. ``None`` when no
    runnable command exists; :func:`explain_tool_command` says why.
    """
    return explain_tool_command(tool_name, params, dataset_path, output_dir, sc_reference_path, mcp_config)[0]


def explain_tool_command(
    tool_name: str,
    params: dict[str, Any],
    dataset_path: str | None = None,
    output_dir: str | None = None,
    sc_reference_path: str | None = None,
    mcp_config: dict[str, Any] | None = None,
) -> tuple[list[str] | None, str]:
    """:func:`build_tool_command`, plus the reason when it builds nothing.

    The refusal used to reach only the log, so a trial that never launched was recorded as "Tool
    execution failed" and a tuning run whose every trial was refused reported "All trial
    configurations failed" with no word of what was missing (hunt 2026-09-30, u32-tuning-10/14).
    The reason is ``""`` whenever a command is returned.
    """
    info = get_tool_server_info(tool_name, mcp_config)
    if info is None:
        logger.warning("Tool %s not found in MCP config", tool_name)
        return None, f"{tool_name} is not in the MCP config"

    env = info.get("server_env", {})

    # Determine worker script path from environment variables
    worker_path = None
    worker_python = None
    worker_rscript = None

    # Common pattern: TOOL_WORKER and TOOL_PYTHON env vars
    for key, val in env.items():
        if key.endswith("_WORKER"):
            worker_path = val
        elif key.endswith("_PYTHON"):
            worker_python = val
        elif key.endswith("_RSCRIPT"):
            worker_rscript = val

    # The override was written by whichever box ran setup, and it is absolute on every shipped
    # config. Both MCP-side launchers repair one naming a checkout that is not this one.
    worker_path = _worker_in_this_checkout(worker_path)

    server_cmd = info.get("server_command", [])
    server_script = _worker_in_this_checkout(server_cmd[-1]) if len(server_cmd) >= 2 else None

    # What the env block leaves unsaid, the portal's own `get_worker_paths(...)` defaults say -- that
    # is what the portal launches, so it is what a trial must launch (u32-tuning-3). Filled in only
    # where the env block is silent, so a config that pins both builds the same command as before.
    portal_defaults = _portal_worker_defaults(server_script) if server_script else None
    portal_interpreter = None
    if portal_defaults is not None:
        _prefix, portal_interpreter, portal_worker = portal_defaults
        if worker_path is None:
            candidate = _worker_in_this_checkout(portal_worker)
            if Path(candidate).exists():
                worker_path = candidate

    if worker_path is None and server_script:
        # Fallback: infer the worker from the server script's name -- derived from an equally
        # machine-specific path. A `.R` worker is tried too: 22 servers ship one.
        for suffix in ("_worker.py", "_worker.R"):
            worker_candidate = _worker_in_this_checkout(server_script.replace("_mcp_server.py", suffix))
            if Path(worker_candidate).exists():
                worker_path = worker_candidate
                if worker_python is None and server_cmd[0] != "python":
                    worker_python = server_cmd[0]
                break

    if worker_path is None or not Path(worker_path).exists():
        logger.warning("Worker script not found for %s", tool_name)
        return None, f"no worker script for {tool_name} was found (configured: {worker_path})"

    # Which interpreter runs it is a property of what the worker is written in, not of which env var
    # happened to be read: 22 servers ship a `.R` worker, and reading only `*_PYTHON` fell them all
    # through to the `"python"` default, which dies on line 1 of an R file.
    #
    # Read `*_PYTHON` first even here, because the spelling is not where the interpreter lives --
    # the wiring is. `get_worker_paths` (tools/base_mcp.py) resolves a portal's interpreter from
    # `{PREFIX}_PYTHON` and no other key, so an R tool has nowhere else to put its override: 21 of
    # the 22 specs marked `worker_kind: rscript` declare `{PREFIX}_PYTHON` holding an *Rscript*, and
    # that is the key `sog-setup` repoints at the env it built. seurat is the lone exception, since
    # its portal reads `SEURAT_RSCRIPT` directly -- so `*_RSCRIPT` stays as the fallback. Keying off
    # the spelling refused 19 of those 22 outright, and on precast and spacet -- which also carry an
    # un-rewired `*_RSCRIPT` inherited from the canonical config -- it launched the build box's
    # interpreter, which is not installed on the deployed one.
    #
    # Guessing an interpreter we were not given is still worse than saying we have none: refuse, and
    # name what is missing. The portal's `get_worker_paths` default is not a guess -- it is what the
    # portal runs when its env block is silent -- so it comes next; only a server with neither keeps
    # the historical `python` default (u32-tuning-3).
    if Path(worker_path).suffix.lower() == ".r":
        interpreter = worker_python or worker_rscript or portal_interpreter
        if not interpreter:
            logger.warning(
                "Tool %s has an R worker (%s), but its server block declares no interpreter under "
                "*_PYTHON or *_RSCRIPT, so there is nothing to run it with. Use the MCP path for "
                "this tool.",
                tool_name,
                Path(worker_path).name,
            )
            return None, f"{tool_name} has an R worker and no interpreter is declared for it"
    else:
        interpreter = worker_python or portal_interpreter or "python"
    cmd = [_interpreter_on_this_box(interpreter), worker_path]

    # Map parameters to CLI arguments
    tool_params = info.get("parameters", {})
    all_params = {}
    # Parameters that carry the trial itself. An optional knob the worker cannot receive is worth
    # dropping with a warning; one of these is not -- running a tool with no input, or with its
    # results going somewhere the caller will not look, is worse than not running it.
    essential: set[str] = set()

    # Start with tool defaults
    for pname, pconfig in tool_params.items():
        if isinstance(pconfig, dict) and "default" in pconfig:
            all_params[pname] = pconfig["default"]

    # Override with provided params
    all_params.update(params)

    # Map dataset_path to the parameter this tool actually declares. Every name below is a real
    # convention in mcp_config.yaml; there is no default, because this path builds a CLI call and a
    # name the worker does not have becomes `--st-h5ad <path>`, which argparse rejects outright.
    # `counts_h5` is excluded on purpose: it is a 10x `.h5`, not an `.h5ad`, and each of the five
    # tools declaring it also declares an h5ad name, which is preferred anyway.
    if dataset_path:
        input_param_names = [
            "st_h5ad",
            "h5ad_path",
            "spatial_h5ad_path",
            "spatial_h5ad",
            "data_path",
            "adata_path",
            "counts_h5ad",
            "counts_h5ad_path",
            "input_path",
            # Multi-slice tools. spacel_scube's spatial_h5ad_paths is a comma-separated str, so one
            # path is a valid value; h5ad_paths / slice_h5ads are lists and the worker CLI parses
            # the same comma-separated form.
            "spatial_h5ad_paths",
            "h5ad_paths",
            "slice_h5ads",
        ]
        mapped = False
        for pname in input_param_names:
            if pname in tool_params:
                all_params[pname] = dataset_path
                essential.add(pname)
                mapped = True
                break
        if not mapped:
            # The tool reads something other than an .h5ad slide -- a CSV pair (card, spacexr_rctd,
            # mistyr), a GEM file (spotgf), a prepared directory (istar), a TOML config (xfuse). No
            # flag here can carry the dataset, so say so instead of emitting one that cannot parse.
            logger.warning(
                "Tool %s declares no .h5ad input parameter, so the dataset cannot be passed on its "
                "CLI; it needs a conversion step first. Declared parameters: %s",
                tool_name,
                sorted(tool_params),
            )
            return None, f"{tool_name} declares no .h5ad input parameter; it needs a conversion step first"

    if output_dir:
        output_param_names = ["output_dir", "results_dir", "output_path", "save_path"]
        mapped = False
        for pname in output_param_names:
            if pname in tool_params:
                all_params[pname] = output_dir
                essential.add(pname)
                mapped = True
                break
        if not mapped:
            all_params["output_dir"] = output_dir
            essential.add("output_dir")

    if sc_reference_path:
        ref_param_names = [
            "sc_h5ad",
            "sc_h5ad_path",
            "scrna_h5ad",
            "sc_reference",
            "reference_h5ad",
        ]
        for pname in ref_param_names:
            if pname in tool_params:
                all_params[pname] = sc_reference_path
                essential.add(pname)
                break

    # Inject --task for multi-task workers: whenever the tool is one, not only when its config
    # declares a `task` parameter -- none does (u32-tuning-1).
    if tool_name in _TOOL_TASK_FLAGS and "task" not in all_params:
        all_params["task"] = _TOOL_TASK_FLAGS[tool_name]

    # Build CLI args. The config names are the portal's; the worker's argparse is a separate
    # namespace, and no worker in tools/ uses parse_known_args -- so a flag it does not declare makes
    # argparse print usage and exit 2, discarding the whole trial. Emit only what the worker declares.
    declared = _declared_worker_flags(worker_path)
    value_flags = _value_taking_flags(worker_path)
    portal_flags = _portal_flag_map(server_script, tool_name) if server_script else {}
    unreachable: dict[str, str] = {}
    for pname, pvalue in all_params.items():
        if pvalue is None or pvalue == "":
            continue
        if declared is None:
            # Worker flags are unreadable (an R script); keep the historical mapping rather than
            # refuse every command for it.
            arg_name = f"--{pname.replace('_', '-')}"
            if isinstance(pvalue, bool):
                if pvalue:
                    cmd.append(arg_name)
            else:
                cmd.extend([arg_name, _cli_value(pvalue)])
            continue
        args, why = _cli_spelling(pname, pvalue, declared, value_flags)
        if why and pname in essential and portal_flags.get(pname) in declared:
            # The portal renames this path on its way to the worker; pass it the way the portal
            # does. Essential paths only: a renamed knob may also be re-encoded, a path is not.
            args, why = [portal_flags[pname], _cli_value(pvalue)], None
        if why:
            unreachable[pname] = why
            continue
        cmd.extend(args)

    blocking = sorted(unreachable.keys() & essential)
    if blocking:
        output_names = ("output_dir", "results_dir", "output_path", "save_path")
        missing = "input" if any(b not in output_names for b in blocking) else "output directory"
        logger.warning(
            "Tool %s cannot be driven from the CLI: %s has no flag in %s, so the command would carry "
            "no %s. Its worker reads %s. Use the MCP path for this tool.",
            tool_name,
            ", ".join(blocking),
            Path(worker_path).name,
            missing,
            sorted(declared),
        )
        return None, (
            f"{tool_name} cannot be driven from the CLI: {', '.join(blocking)} has no flag in "
            f"{Path(worker_path).name}, so the command would carry no {missing}"
        )
    if unreachable:
        logger.warning(
            "Tool %s: dropped %s -- %s declares no matching flag, so these knobs are unreachable "
            "from the CLI and the worker's own defaults apply.",
            tool_name,
            ", ".join(sorted(unreachable)),
            Path(worker_path).name,
        )

    return cmd, ""


def inject_tuned_params(
    tool_name: str,
    original_params: dict[str, Any],
    tuning_dir: str | None = None,
) -> dict[str, Any]:
    """Override tool parameters with tuned values if available.

    Used by the benchmark runner to apply tuned configs.
    """
    from spatialomicsgym.tuning.persistence import load_best_config

    best = load_best_config(tool_name, tuning_dir)
    if best is None:
        return original_params

    tuned_params = best.get("params", {})
    merged = original_params.copy()
    merged.update(tuned_params)

    logger.info(
        "Injected %d tuned params for %s (score=%.4f, mode=%s)",
        len(tuned_params),
        tool_name,
        best.get("score", 0.0),
        best.get("mode", "unknown"),
    )
    return merged


def get_tunable_tools(mcp_config_path: str | None = None) -> list[dict[str, str]]:
    """List all MCP tools that support tuning.

    Returns list of dicts with tool_name, server_key, task_type.
    """
    config = load_mcp_config(mcp_config_path)
    tools = []

    for server_key, server_cfg in config.get("mcp_servers", {}).items():
        for tool in server_cfg.get("tools", []):
            tool_name = tool.get("spatialomicsgym_name", "")
            description = tool.get("description", "").lower()

            # Task type inference from tool names first, then descriptions. The task type is not
            # decoration: mode_router picks a search space from it and executor hands it to the
            # scorer, so a tool typed wrongly is tuned against the wrong objective. Ask the tool
            # what it is before asking prose that may be describing a prerequisite or an input.
            # `unknown` is the safe answer -- mode_router routes it to DEFAULT_FALLBACK and says
            # why -- so the fallthrough below leaves a tool untyped rather than guessing.
            name_words = set(tool_name.lower().split("_"))
            if name_words & _CLUSTERING_NAME_WORDS:
                task_type = "spatial_clustering"
            elif name_words & _SVG_NAME_WORDS:
                task_type = "svg_detection"
            elif name_words & _DECONV_NAME_WORDS:
                task_type = "deconvolution"
            elif any(kw in description for kw in _CLUSTERING_PROSE):
                task_type = "spatial_clustering"
            elif any(kw in description for kw in _SVG_PROSE):
                task_type = "svg_detection"
            elif any(kw in description for kw in _DECONV_PROSE):
                task_type = "deconvolution"
            else:
                task_type = "unknown"

            # Count tunable params (non-path, non-device params with defaults)
            params = tool.get("parameters", {})
            tunable_count = 0
            for pname, pcfg in params.items():
                if isinstance(pcfg, dict) and "default" in pcfg:
                    # Skip path/device/key params
                    if not any(kw in pname.lower() for kw in ["path", "dir", "device", "key", "description"]):
                        tunable_count += 1

            if tunable_count > 0:
                tools.append(
                    {
                        "tool_name": tool_name,
                        "server_key": server_key,
                        "task_type": task_type,
                        "tunable_params": tunable_count,
                    }
                )

    return tools


# What a benchmark dataset fixes for every trial, by the parameter names tools declare for it. A
# search ranks configurations under whatever the untuned knobs are set to, so these have to be the
# values the scored run will use -- the portal defaults are not: STAGATE trials on DLPFC ran
# `--n-clusters 6` against the registry's 7, and tangram/graphst trials passed `cell_type` to a
# reference whose column is `CellType` or `subclass_label`, which those workers refuse
# (hunt 2026-09-30, u32-tuning-5). Exact names only: cellcharter's n_clusters_min/max is a search
# range, not a count, and is left to its own defaults.
_CLUSTER_COUNT_PARAMS = ("n_clusters", "n_domains", "K", "target_n_clusters")
_REFERENCE_LABEL_PARAMS = ("annotation_key", "celltype_key", "cell_type_key", "labels_key")


def _same_file(a: str, b: str) -> bool:
    """Whether two paths name one file; an unresolvable path is no evidence that they do."""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def registry_fixed_params(
    tool_name: str,
    task_type: str,
    metadata: dict[str, Any] | None,
    mcp_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The values a benchmark dataset's registry entry fixes for *tool_name*, keyed as the tool names them.

    Pass the result as ``fixed_params`` to :func:`spatialomicsgym.tuning.tune` or
    :meth:`TunedBenchmarkRunner.get_params_for_tool`. Only parameters the tool declares are returned,
    so nothing reaches a worker under a name it does not have.
    """
    info = get_tool_server_info(tool_name, mcp_config)
    if info is None or not isinstance(metadata, dict):
        return {}
    declared = info.get("parameters") or {}
    fixed: dict[str, Any] = {}
    n_clusters = metadata.get("n_clusters")
    if task_type == "spatial_clustering" and isinstance(n_clusters, int) and not isinstance(n_clusters, bool):
        fixed.update({name: n_clusters for name in _CLUSTER_COUNT_PARAMS if name in declared})
    label_key = metadata.get("sc_reference_celltype_key")
    if task_type == "deconvolution" and isinstance(label_key, str) and label_key:
        fixed.update({name: label_key for name in _REFERENCE_LABEL_PARAMS if name in declared})
    return fixed


class TunedBenchmarkRunner:
    """Wrapper that adds tuning support to benchmark runs.

    Usage:
        runner = TunedBenchmarkRunner(mode=TuningMode.BENCHMARK_LIGHT)
        result = runner.run_tool(tool_name, dataset_entry, task_type, config)
    """

    def __init__(
        self,
        mode: TuningMode = TuningMode.BENCHMARK_LIGHT,
        tuning_dir: str | None = None,
        use_cached: bool = True,
    ):
        self.mode = mode
        self.tuning_dir = tuning_dir
        self.use_cached = use_cached

    def get_params_for_tool(
        self,
        tool_name: str,
        task_type: str,
        dataset_path: str | None = None,
        sc_reference_path: str | None = None,
        svg_ground_truth_path: str | None = None,
        ground_truth_key: str | None = None,
        fixed_params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Get parameters for a tool, using cached tuning results or running tuning.

        Returns (params_dict, mode_used_description). The mapping holds only values a search
        selected -- here, or for this dataset in an earlier run -- and is empty when none did. The
        benchmark states every value it receives as "a hyperparameter search selected these values
        for this tool on this dataset", so plain defaults, a search whose every trial failed, and a
        score measured on another dataset are not handed over as tuned (hunt 2026-09-30,
        u32-tuning-7); the description still says what happened.

        *svg_ground_truth_path*, *ground_truth_key* and *fixed_params* reach :func:`tune`. Without
        them an svg search cannot score a trial at all, a clustering search is scored against
        whichever column the evaluator falls back to, and every trial runs at the portal's
        n_clusters and reference label column rather than the dataset's (u32-tuning-4/5/8).
        """
        # Check for cached tuned config
        if self.use_cached:
            from spatialomicsgym.tuning.persistence import load_best_config

            # A hit needs parameters in it, which is what the other cache readers that guard at all
            # ask (agent/stcoscientist.py and mode_router.py -- inject_tuned_params below does not,
            # it relies on load_best_config). Asking only whether the file parsed made an entry with
            # no ``params`` a KeyError out of a benchmark run, and an entry with ``params: {}`` a
            # reported "cached:" hit that handed the tool nothing and skipped the tuning this runner
            # was configured to do. Note this is a truthiness test and cannot express "is a mapping":
            # a params key holding a list or a string is rejected upstream, in load_best_config.
            cached = load_best_config(tool_name, self.tuning_dir)
            if cached and cached.get("params"):
                # The cache is keyed on the tool alone. An entry recorded for a different dataset is
                # not a result for this one, and this runner was asked to tune this one.
                recorded = cached.get("dataset")
                if not (dataset_path and recorded and not _same_file(str(recorded), dataset_path)):
                    return cached["params"], f"cached:{cached.get('mode', 'unknown')}"
                logger.info(
                    "Not reusing the cached config for %s: it was measured on %s, not %s",
                    tool_name,
                    Path(str(recorded)).name,
                    Path(dataset_path).name,
                )

        # Run tuning if we have a dataset
        if dataset_path and self.mode != TuningMode.DEFAULT_FALLBACK:
            from spatialomicsgym.tuning import tune

            result = tune(
                tool_name=tool_name,
                task_type=task_type,
                dataset_path=dataset_path,
                mode=self.mode,
                sc_reference_path=sc_reference_path,
                svg_ground_truth_path=svg_ground_truth_path,
                output_dir=self.tuning_dir,
                ground_truth_key=ground_truth_key,
                fixed_params=fixed_params,
            )
            if result.mode_used == TuningMode.DEFAULT_FALLBACK:
                # No trial succeeded, or none could run: the "best" is the untouched baseline.
                return {}, f"default_fallback ({result.mode_reason})"
            return result.best_params, result.mode_used.value

        # No search runs in default_fallback, and with no dataset there is nothing to search on.
        return {}, "default_fallback"
