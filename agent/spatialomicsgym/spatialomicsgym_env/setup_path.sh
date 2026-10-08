#!/bin/bash
# Added by spatialomicsgym setup -- source this to put the vendored CLI tools on PATH.
#
# install_cli_tools.sh generates $TOOLS_DIR/setup_path.sh with $TOOLS_DIR already expanded, so the
# copy it writes is correct on the box that ran it. Nothing else writes one: setup.sh calls that
# installer and stopped writing its own (hunt 2026-09-30, u38a-packaging-5, repair).
# This tracked copy is different: it ships in every wheel (MANIFEST.in's recursive-include picks up
# *.sh) and DETAILS.md tells the reader to source it, so it reaches machines the installer never ran
# on. It therefore has to resolve the tools tree the way the installers pick it -- from SOG_TOOLS_DIR,
# else from its own location -- rather than naming one machine's directory.

_sog_setup_path_self="${BASH_SOURCE[0]:-$0}"
_sog_setup_path_dir="$(cd "$(dirname "$_sog_setup_path_self")" 2>/dev/null && pwd)"

if [ -n "${SOG_TOOLS_DIR:-}" ]; then
    _sog_tools_bin="$SOG_TOOLS_DIR/bin"
else
    _sog_tools_bin="$_sog_setup_path_dir/spatialomicsgym_tools/bin"
fi

if [ -d "$_sog_tools_bin" ]; then
    # Remove any old paths first to avoid duplicates
    PATH=$(echo $PATH | tr ':' '\n' | grep -v "spatialomicsgym_tools/bin" | tr '\n' ':' | sed 's/:$//')
    export PATH="$_sog_tools_bin:$PATH"
else
    # Nothing to add. Say so and leave PATH untouched: stripping the entry a working install put there
    # and replacing it with a directory that does not exist is worse than doing nothing.
    echo "spatialomicsgym: no CLI tools found at $_sog_tools_bin" >&2
    echo "spatialomicsgym: run agent/spatialomicsgym/spatialomicsgym_env/install_cli_tools.sh, or point SOG_TOOLS_DIR at an existing install." >&2
fi

unset _sog_setup_path_self _sog_setup_path_dir _sog_tools_bin
