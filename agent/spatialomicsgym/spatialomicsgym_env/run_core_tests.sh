#!/usr/bin/env bash
# Run the SpatialOmicsLab agent-core test suite in the minimal spatialomicsgym_env.
# Usage: bash agent/spatialomicsgym/spatialomicsgym_env/run_core_tests.sh [extra pytest args]
set -euo pipefail

# Locate the spatialomicsgym_env interpreter. `conda env create -f ...env.yml` puts the env under
# whatever conda base the user has (~/miniconda3, ~/mambaforge, ...); this used to hardcode
# /opt/conda, which is only where it lands on the machine this was written on. Everyone else got
# "No such file or directory" from the command the project documents as *the* way to run the tests.
# Ordered cheapest-first; each branch is a full `if` because `[ x ] && PY=...` returning nonzero
# would trip `set -e`.
PY="${SOG_TEST_PYTHON:-}"
if [ -z "$PY" ] && [ -n "${CONDA_EXE:-}" ]; then
    _cand="${CONDA_EXE%/bin/conda}/envs/spatialomicsgym_env/bin/python"
    if [ -x "$_cand" ]; then PY="$_cand"; fi
fi
if [ -z "$PY" ] && command -v conda >/dev/null 2>&1; then
    _base="$(conda info --base 2>/dev/null || true)"
    if [ -n "$_base" ] && [ -x "$_base/envs/spatialomicsgym_env/bin/python" ]; then
        PY="$_base/envs/spatialomicsgym_env/bin/python"
    fi
fi
if [ -z "$PY" ] && [ -x "/opt/conda/envs/spatialomicsgym_env/bin/python" ]; then
    PY="/opt/conda/envs/spatialomicsgym_env/bin/python"
fi
if [ -z "$PY" ]; then
    echo "error: could not find the 'spatialomicsgym_env' python." >&2
    echo "  create it:      conda env create -f agent/spatialomicsgym/spatialomicsgym_env/spatialomicsgym_env.yml" >&2
    echo "  or point at it: SOG_TEST_PYTHON=/path/to/env/bin/python bash \"\$0\"" >&2
    exit 1
fi

# This script sits at <repo>/agent/spatialomicsgym/spatialomicsgym_env/, three levels below the
# repository root -- the directory that holds pyproject.toml (whose pytest config this run reads)
# and test/.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

# test/ is untracked: a clone or an unpacked release does not carry it. Say so instead of letting
# pytest report "no tests ran" -- a gate with nothing to run has not passed.
if [ ! -d "$ROOT/test" ]; then
    echo "error: test/ is not present in this checkout -- the test suite ships only with the development tree." >&2
    exit 1
fi

echo ">>> core unit/portal suite (spatialomicsgym_env)"
# The WHOLE suite, not an allowlist. This used to name 12 files explicitly, which meant a new test
# file was covered by nothing until someone remembered to add it here -- 95 test files, including
# every regression test added over months, were silently never run by any gate.
#
# The allowlist existed only because running the whole tree used to hang: the hand-run debug
# drivers (now under test/manual/) are named `test_*.py` and launched per-tool workers at import
# time, and pytest imports everything it collects. That is fixed at the source -- pyproject.toml's
# `[tool.pytest.ini_options]` ignores that directory and test/installation/ (the setup wizard's
# artifacts), and test/test_repo_hygiene.py fails if a module doing import-time work reappears in
# the collected tree -- so the gate can just run `test` and new tests are covered by default.
"$PY" -m pytest test "$@"
echo ">>> static portal smoke (every configured MCP server, levels 1-2; a per-tool env this box never built SKIPs as not provisioned)"
# test/ has no __init__.py (`test` is a standard-library package name), so the harness is the
# top-level package `smoke`: put test/ on the path and stay at the repository root.
PYTHONPATH="$ROOT/test${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m smoke.run_smoke_test --levels 1,2 --workers 4
