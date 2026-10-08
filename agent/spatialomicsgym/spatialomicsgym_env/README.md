# SpatialOmicsLab Environment Setup

This directory contains scripts and configuration files to set up a comprehensive bioinformatics environment with various tools and packages.

1. Clone the repository:
   ```bash
   git clone https://github.com/jiangyi01/SpatialOmicsLab.git
   cd SpatialOmicsLab/agent/spatialomicsgym/spatialomicsgym_env
   ```

2. Setting up the environment:
- (a) If you want to use or try out the basic agent without the full E1 or install your own softwares, run the following script:

```bash
conda env create -f environment.yml
```

- (b) If you want to use the full environment E1, run the setup script (this script takes > 10 hours to setup, and requires a disk of at least 30 GB quota). Follow the prompts to install the desired components.

```bash
bash setup.sh
```

If you already installed the base version, and just wants to add the additional packages in the new release, you can simply do:

```bash
bash new_software_v008.sh
```

Note: we have only tested this setup.sh script with Ubuntu 22.04, 64 bit.

- (c) If you want to use a reduced conda environment without R or CLI tools, run the following script:

```bash
conda env create -f fixed_env.yml
```

This contains most of the packages from environment.yml and bio_env.yml, and requires a disk of at elast 13GB quota.

- (d) **Python 3.10 Environment for Copy Number Analysis**: If you specifically need to use the `analyze_copy_number_purity_ploidy_and_focal_events` function, we provide a Python 3.10 environment option. This function has specific dependency requirements that are best met with Python 3.10. To set up this environment:

```bash
conda env create -f bio_env_py310.yml
```

This environment is optimized for copy number variation analysis and includes the necessary packages for purity, ploidy, and focal event detection.

3. Lastly, to activate the spatialomicsgym environment:
```bash
conda activate spatialomicsgym_e1
```

For the Python 3.10 environment specifically:
```bash
conda activate bio_env_py310
```

### 📦 Langchain Package Support

The SpatialOmicsLab environment comes with a minimal set of langchain packages by default:
- `langchain-openai` - for OpenAI model support
- `langchain-anthropic` - for Anthropic model support
- `langchain-ollama` - for Ollama model support

If you need support for other external models or services, you'll need to install additional langchain packages manually. For example:

```bash
# For AWS Bedrock support
pip install langchain-aws

# For Google Gemini support
pip install langchain-google-genai

```

## General-purpose fallback env (`spatialomicsgym_env_general.yml`)

The agent-core recipe above plus an analysis stack (squidpy, omnipath, decoupler 1.9.2, gseapy,
statsmodels, scikit-image, umap-learn, seaborn). This is where ST-Coscientist may redo a step
in-process when a tool's own worker environment fails to import on a machine -- the model is told so
in the observation and the answer must disclose the substitute (see `agent/spatialomicsgym/tool/general_env.py`
and `agent/spatialomicsgym/agent/env_fallback.py`). It is never used under benchmarking.

```bash
conda env create -n spatialomicsgym_env_general -f agent/spatialomicsgym/spatialomicsgym_env/spatialomicsgym_env_general.yml
/opt/conda/envs/spatialomicsgym_env_general/bin/pip install -e . --no-deps
```

It is not a tool env: no portal points at it, `sog-setup` never provisions, repairs or removes it
(it is on `PROTECTED_ENVS`), and `SOG_GENERAL_PYTHON=/path/to/python` names its interpreter when it
lives somewhere `sog_install.constants.general_python()` would not look. Its pip list up to the analysis
block is the core recipe's, byte for byte -- keep it so. decoupler is held at 1.9.2 deliberately
(2.x depends on marsilea; see `CHINA_EXCLUSION_REPO.md` section 18).
