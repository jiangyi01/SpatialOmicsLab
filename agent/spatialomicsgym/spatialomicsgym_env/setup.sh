#!/bin/bash

# BioAgentOS - SpatialOmicsLab Environment Setup Script
# This script sets up a comprehensive bioinformatics environment with various tools and packages

# Set up colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Every recipe this script names -- environment.yml, bio_env.yml, r_packages.yml, install_r_packages.R,
# install_cli_tools.sh -- sits beside it, and the documented command runs it from the repository root.
# Resolved against the caller's directory, every step failed there, and the script still printed
# "Setup Completed!" and exited 0 (hunt 2026-09-30, u38a-packaging-4, uL4-honesty-15).
# The caller's directory is kept first: a tools directory the caller gives is theirs to resolve.
CALLER_DIR="$PWD"
cd "$(dirname "${BASH_SOURCE[0]}")" || { echo "Error: cannot enter the directory that holds setup.sh" >&2; exit 1; }

# A tools directory as the caller meant it: `~` is their home, and a relative path is relative to where
# they typed the command. Read after the cd above, `mytools` became a directory beside the recipes,
# inside the package tree, where MANIFEST.in's *.sh include would ship the setup_path.sh generated there
# (hunt 2026-09-30, u38a-packaging-4, repair).
resolve_tools_dir() {
    local dir=$1
    case "$dir" in
        "~") dir="$HOME" ;;
        "~/"*) dir="$HOME/${dir:2}" ;;
    esac
    case "$dir" in
        /*) printf '%s\n' "$dir" ;;
        *) printf '%s\n' "$CALLER_DIR/$dir" ;;
    esac
}

# Default tools directory: SOG_TOOLS_DIR when the caller set it (install_cli_tools.sh reads the same
# variable), otherwise beside the recipes, where MANIFEST.in prunes it from every sdist.
if [ -n "${SOG_TOOLS_DIR:-}" ]; then
    DEFAULT_TOOLS_DIR="$(resolve_tools_dir "$SOG_TOOLS_DIR")"
else
    DEFAULT_TOOLS_DIR="$(pwd)/spatialomicsgym_tools"
fi
TOOLS_DIR=""

# Steps that failed, counted whether or not the run continued past them. The final banner and the
# exit status both come from this.
SETUP_ERRORS=0

echo -e "${YELLOW}=== SpatialOmicsLab Environment Setup ===${NC}"
echo -e "${BLUE}This script will set up a comprehensive bioinformatics environment with various tools and packages.${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null && ! command -v micromamba &> /dev/null; then
    echo -e "${RED}Error: Conda is not installed or not in PATH.${NC}"
    echo "Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# redirect to micromamba if needed
if ! command -v conda &> /dev/null && command -v micromamba &> /dev/null; then
    conda() {
        micromamba "$@"
    }
    export -f conda
fi

# Function to handle errors
handle_error() {
    local exit_code=$1
    local error_message=$2
    local optional=${3:-false}

    if [ $exit_code -ne 0 ]; then
        SETUP_ERRORS=$((SETUP_ERRORS + 1))
        echo -e "${RED}Error: $error_message${NC}"
        if [ "$optional" = true ]; then
            # This branch used to `return 0`, so the caller's `if [ $? -eq 0 ]` printed "Successfully
            # installed" right after the step failed. Continue, but hand back the step's own status
            # (hunt 2026-09-30, u38a-packaging-4).
            echo -e "${YELLOW}Continuing with setup as this component is optional.${NC}"
        else
            if [ -z "$NON_INTERACTIVE" ]; then
                read -p "Continue with setup? (y/n) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo -e "${RED}Setup aborted.${NC}"
                    exit 1
                fi
            else
                echo -e "${YELLOW}Non-interactive mode: continuing despite error.${NC}"
            fi
        fi
    fi
    return $exit_code
}

# Function to install a specific environment file
install_env_file() {
    local env_file=$1
    local description=$2
    local optional=${3:-false}

    echo -e "\n${BLUE}=== Installing $description ===${NC}"

    if [ "$optional" = true ]; then
        if [ -z "$NON_INTERACTIVE" ]; then
            read -p "Do you want to install $description? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}Skipping $description installation.${NC}"
                return 0
            fi
        else
            echo -e "${YELLOW}Non-interactive mode: automatically installing $description.${NC}"
        fi
    fi

    echo -e "${YELLOW}Installing $description from $env_file...${NC}"
    conda env update -f $env_file
    handle_error $? "Failed to install $description." $optional

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully installed $description!${NC}"
    fi
}

# Function to install CLI tools
install_cli_tools() {
    echo -e "\n${BLUE}=== Installing Command-Line Bioinformatics Tools ===${NC}"

    # Ask user for the directory to install CLI tools
    if [ -z "$NON_INTERACTIVE" ]; then
        echo -e "${YELLOW}Where would you like to install the command-line tools?${NC}"
        echo -e "${BLUE}Default: $DEFAULT_TOOLS_DIR${NC}"
        read -p "Enter directory path (or press Enter for default): " user_tools_dir
    else
        user_tools_dir=""
        echo -e "${YELLOW}Non-interactive mode: using default directory $DEFAULT_TOOLS_DIR for CLI tools.${NC}"
    fi

    if [ -z "$user_tools_dir" ]; then
        TOOLS_DIR="$DEFAULT_TOOLS_DIR"
    else
        TOOLS_DIR="$(resolve_tools_dir "$user_tools_dir")"
    fi

    # Export the tools directory for the CLI tools installer
    export SOG_TOOLS_DIR="$TOOLS_DIR"

    echo -e "${YELLOW}Installing command-line tools (PLINK 2.0, IQ-TREE, BWA, etc.) to $TOOLS_DIR...${NC}"

    # Set environment variable to skip prompts in the CLI tools installer
    export SOG_AUTO_INSTALL=1

    # Run the CLI tools installer
    bash install_cli_tools.sh
    handle_error $? "Failed to install CLI tools." true

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully installed command-line tools!${NC}"

        # install_cli_tools.sh writes $TOOLS_DIR/setup_path.sh with the path expanded. This used to
        # write ./setup_path.sh as well -- which, run from here, is the TRACKED portable copy -- with
        # this box's directory frozen into it (hunt 2026-09-30, u38a-packaging-5).
        echo -e "${YELLOW}You can add the tools to your PATH by running:${NC}"
        echo -e "${GREEN}source $TOOLS_DIR/setup_path.sh${NC}"

        # Also add to the current session
        # Remove any old paths first to avoid duplicates
        PATH=$(echo $PATH | tr ':' '\n' | grep -v "spatialomicsgym_tools/bin" | tr '\n' ':' | sed 's/:$//')
        export PATH="$TOOLS_DIR/bin:$PATH"
    fi

    # Unset the environment variables
    unset SOG_AUTO_INSTALL
    unset SOG_TOOLS_DIR
}

# Main installation process
main() {
    # Step 1: Create base conda environment
    echo -e "\n${YELLOW}Step 1: Creating base environment from environment.yml...${NC}"
    conda env create -n spatialomicsgym_e1 -f environment.yml
    handle_error $? "Failed to create base conda environment."

    # Step 2: Activate the environment
    echo -e "\n${YELLOW}Step 2: Activating conda environment...${NC}"
    if command -v micromamba &> /dev/null; then
        eval "$("$MAMBA_EXE" shell hook --shell bash)"
        micromamba activate spatialomicsgym_e1
    else
        eval "$(conda shell.bash hook)"
        conda activate spatialomicsgym_e1
    fi
    handle_error $? "Failed to activate spatialomicsgym_e1 environment."

    # Step 3: Install core bioinformatics tools (including QIIME2)
    echo -e "\n${YELLOW}Step 3: Installing core bioinformatics tools (including QIIME2)...${NC}"
    install_env_file "bio_env.yml" "core bioinformatics tools"

    # Step 4: Install R packages
    echo -e "\n${YELLOW}Step 4: Installing R packages...${NC}"
    install_env_file "r_packages.yml" "core R packages"

    # Step 5: Install additional R packages through R's package manager
    echo -e "\n${YELLOW}Step 5: Installing additional R packages through R's package manager...${NC}"
    Rscript install_r_packages.R
    handle_error $? "Failed to install additional R packages." true

    # Step 6: Install CLI tools
    echo -e "\n${YELLOW}Step 6: Installing command-line bioinformatics tools...${NC}"
    install_cli_tools

    # Setup finished. bio_analysis_example.py and a BioAgentOS directory were named here; neither
    # exists in this repository (hunt 2026-09-30, uL4-honesty-15).
    if [ "$SETUP_ERRORS" -eq 0 ]; then
        echo -e "\n${GREEN}=== SpatialOmicsLab Environment Setup Completed! ===${NC}"
    else
        echo -e "\n${RED}=== SpatialOmicsLab Environment Setup finished with $SETUP_ERRORS error(s) -- see the messages above ===${NC}"
    fi
    echo -e "To activate this environment in the future, run: ${YELLOW}conda activate spatialomicsgym_e1${NC}"

    # Display CLI tools setup instructions
    if [ -n "$TOOLS_DIR" ]; then
        echo -e "\n${BLUE}=== Command-Line Tools Setup ===${NC}"
        echo -e "The command-line tools are installed in: ${YELLOW}$TOOLS_DIR${NC}"
        echo -e "To add these tools to your PATH, run: ${YELLOW}source $TOOLS_DIR/setup_path.sh${NC}"
        echo -e "You can also add this line to your shell profile for permanent access:"
        echo -e "${GREEN}export PATH=\"$TOOLS_DIR/bin:\$PATH\"${NC}"

        # Test if tools are accessible
        echo -e "\n${BLUE}=== Testing CLI Tools ===${NC}"
        # No GCTA probe: cli_tools_config.json no longer installs GCTA, so it always reported it
        # missing (hunt 2026-09-30, u38a-packaging-4).
        if command -v plink2 &> /dev/null; then
            echo -e "${GREEN}PLINK2 is accessible in the current PATH${NC}"
            echo -e "PLINK2 location: $(which plink2)"
        else
            echo -e "${RED}PLINK2 is not accessible in the current PATH${NC}"
            echo -e "Please run: ${YELLOW}source $TOOLS_DIR/setup_path.sh${NC} to update your PATH"
        fi

        if command -v iqtree2 &> /dev/null; then
            echo -e "${GREEN}IQ-TREE is accessible in the current PATH${NC}"
            echo -e "IQ-TREE location: $(which iqtree2)"
        else
            echo -e "${RED}IQ-TREE is not accessible in the current PATH${NC}"
            echo -e "Please run: ${YELLOW}source $TOOLS_DIR/setup_path.sh${NC} to update your PATH"
        fi
    fi

    PATH=$(echo $PATH | tr ':' '\n' | grep -v "spatialomicsgym_tools/bin" | tr '\n' ':' | sed 's/:$//')
    export PATH="$(pwd)/spatialomicsgym_tools/bin:$PATH"
}

# Run the main installation process
main

# Non-zero when any step failed, including the optional ones the run continued past.
if [ "$SETUP_ERRORS" -ne 0 ]; then
    exit 1
fi
