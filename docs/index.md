# SpatialOmicsLab

**An integrated research environment for AI co-scientists in spatial transcriptomics.**

Spatial transcriptomics has hundreds of analysis methods, and most come with their own software stack: PyTorch for
one, R/Bioconductor for the next, a pinned CUDA build for a third. SpatialOmicsLab is a research platform built
around an LLM agent, **ST-Coscientist**, that takes this work off your hands. You describe an analysis in ordinary
language ("find the spatial domains in this Visium slide", "deconvolve these spots against my single-cell
reference"), and the agent plans it, runs it and reports back.

*SpatialOmicsLab* is the project, *ST-Coscientist* is the agent, and `spatialomicsgym` is the Python package.

```{figure} ../figures/Framework.png
:alt: The SpatialOmicsLab framework
:figclass: sog-framework
:width: 100%

**(A)** The user poses a question in plain English and provides a dataset, such as 10x Visium spatial transcriptomics
with a single-cell reference. **(B)** The SpatialOmicsLab environment enriches the question with a prompt enhancer
(question management, parameter checking, data conversion, hyper-parameter tuning, benchmarking, post-analysis) and a
skills library (tool integration procedures, reference and spatial dataset libraries, memory). **(C)** ST-Coscientist
retrieves the relevant tools, assembles the system prompt and runs a ReAct loop over a unified registry of default and
user-created MCP tools. **(D)** Each analysis produces conversation logs, h5ad/csv results, visualization figures and a
web report.
```

## What it brings

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} A ReAct agent for spatial omics
:link: architecture/index
:link-type: doc

ST-Coscientist retrieves the tools, datasets and curated know-how relevant to your task, writes and executes code,
checks the output and decides the next step.
:::

:::{grid-item-card} A large tool platform
:link: tools/index
:link-type: doc

{{ n_tools_total }} tools: {{ n_mcp_functions }} analysis functions served by {{ n_servers }} Model Context Protocol
(MCP) servers, plus {{ n_in_process }} in-process research tools (databases, expression atlases, pharmacology,
literature).
:::

:::{grid-item-card} Conflict-free environments
:link: sog-setup
:link-type: doc

Each method that needs one runs in its own conda environment, so methods with incompatible dependencies coexist on
one machine. A guided installer, `sog-setup`, builds and tests them.
:::

:::{grid-item-card} Two ways in
:link: usage/terminal-chat
:link-type: doc

A terminal chat (`stcoscientist`) and a Python API. The agent works with Anthropic, OpenAI, Azure OpenAI, Gemini,
Groq, AWS Bedrock, Ollama and any OpenAI-compatible endpoint.
:::

::::

## Start here

1. [Install](getting-started/installation.md) the package and run the guided installer.
2. [Ask a first question](getting-started/quickstart.md) from the terminal, then run a first analysis from Python.
3. Browse the [analysis tools](tools/index.md) the agent can call, and the [configuration](configuration.md) it reads.

```{code-block} console
:caption: The short version

$ git clone https://github.com/jiangyi01/SpatialOmicsLab.git && cd SpatialOmicsLab
$ conda env create -f agent/spatialomicsgym/spatialomicsgym_env/spatialomicsgym_env.yml
$ conda activate spatialomicsgym_env
$ pip install -e .
$ sog-setup
$ stcoscientist --mcp
```

## Citation

If you use SpatialOmicsLab in your research, please cite:

> Jiang, Y., Zhan, X., Quan, P., Wang, R., Wu, F., Mi, J., Yao, J., Yao, B., Xiao, G., Shi, W., & Xie, Y.
> *SpatialOmicsLab: an integrated research environment for AI co-scientists in spatial transcriptomics.*

## License

SpatialOmicsLab is released under the **GNU Affero General Public License v3.0 only**
([LICENSE](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/LICENSE)). If you run a modified version as a
network service, you must make its source available to its users. Third-party components keep their own licenses
([THIRD_PARTY_LICENSES/](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/THIRD_PARTY_LICENSES/README.md)), and
some integrated datasets are non-commercial
([license_info.md](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/license_info.md)); review them before
commercial use.

## Contact

Suggestions, ideas, or trouble using it: open a
[GitHub issue](https://github.com/jiangyi01/SpatialOmicsLab/issues).

```{toctree}
:hidden:
:caption: Getting started
:maxdepth: 1

getting-started/installation
getting-started/quickstart
getting-started/verify
```

```{toctree}
:hidden:
:caption: User guide
:maxdepth: 1

usage/terminal-chat
usage/python-api
usage/results
configuration
sog-setup
```

```{toctree}
:hidden:
:caption: Analysis tools
:maxdepth: 1

tools/index
```

```{toctree}
:hidden:
:caption: Architecture
:maxdepth: 1

architecture/index
architecture/agent-package
architecture/mcp-tools
```

```{toctree}
:hidden:
:caption: Tutorials
:maxdepth: 1

tutorials/index
```

```{toctree}
:hidden:
:caption: Reference
:maxdepth: 1

api/index
```

```{toctree}
:hidden:
:caption: Project
:maxdepth: 1

Contributing <https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/CONTRIBUTION.md>
Security <https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/SECURITY.md>
GitHub <https://github.com/jiangyi01/SpatialOmicsLab>
```
