# Shell-side half of the agent-env adoption shim.
#
# Scripts that select the heavy agent env run BEFORE that env is on the path, so they cannot
# import spatialomicsgym to read constants.LEGACY_ENV_ALIASES. This file is therefore the one
# place the shell layer may name the pre-rebrand env: test/test_setup/test_no_legacy_env_name.py
# sanctions it, and test/test_setup/test_agent_env_shim_matches_constants.py fails if the list
# below ever drifts from LEGACY_ENV_ALIASES.
#
# Usage -- the caller resolves REPO_ROOT first (that part is a genuine bootstrap and stays
# inlined), then:
#
#     . "$REPO_ROOT/install/sog_install/agent_env.sh"
#     sog_resolve_agent_env || exit 1     # sets CONDA_BASE and AGENT_ENV
#
# Resolution only. Callers decide whether to `conda activate "$AGENT_ENV"` -- not every driver
# does (see benchmarks/run_merfish_benchmark.sh, which has always used whatever python is on
# PATH and still needs CONDA_BASE/AGENT_ENV for its library path).

# Newest first, so a box carrying both envs prefers the branded one.
SOG_AGENT_ENV_CANDIDATES="spatialomicsgym_e1 biomni_e1"

sog_resolve_agent_env() {
    CONDA_BASE="$(conda info --base)" || {
        echo "[ERROR] conda not available on PATH" >&2
        return 1
    }
    AGENT_ENV=""
    for _cand in $SOG_AGENT_ENV_CANDIDATES; do
        if [ -d "$CONDA_BASE/envs/$_cand" ]; then
            AGENT_ENV="$_cand"
            break
        fi
    done
    if [ -z "$AGENT_ENV" ]; then
        echo "[ERROR] agent env not found (looked for $SOG_AGENT_ENV_CANDIDATES under $CONDA_BASE/envs)" >&2
        return 1
    fi
    return 0
}
