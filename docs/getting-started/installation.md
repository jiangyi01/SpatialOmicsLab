# Installation

SpatialOmicsLab installs in two layers. The **agent environment** holds the `spatialomicsgym` package and everything
the agent itself needs; it is small and takes minutes. The **tool environments** are one conda environment per
analysis method, built by the guided installer `sog-setup`; they are large, and you choose which ones to build.

## Requirements

1. Linux or macOS.
2. Python ≥ 3.11 — <https://www.python.org>.
3. conda, mamba or micromamba — <https://docs.conda.io>. Needed for the per-tool environments; chat *without*
   analysis tools works without it.
4. An API key for one LLM provider (Anthropic, OpenAI, Azure OpenAI, Gemini, Groq, AWS Bedrock, or an
   OpenAI-compatible endpoint), or a local [Ollama](https://ollama.com) server.

GPUs are optional. Some methods are faster with one and a few are built for it (the [tool catalogue](../tools/index.md)
says which); the others run on the CPU.

## Install the agent

1. Clone the repository:

   ```bash
   git clone https://github.com/jiangyi01/SpatialOmicsLab.git
   cd SpatialOmicsLab
   ```

2. Create the agent environment and install the package into it, in editable mode:

   ```bash
   conda env create -f agent/spatialomicsgym/spatialomicsgym_env/spatialomicsgym_env.yml
   conda activate spatialomicsgym_env
   pip install -e .
   ```

   This gives you three commands: `sog-setup` (the installer), `stcoscientist` (the terminal chat) and `sog-chat`
   (an alias of `stcoscientist`).

3. Run the guided installer:

   ```bash
   sog-setup
   ```

   It configures your LLM provider (and checks the key with a live ping), lets you choose skill categories, then
   builds and tests one conda environment per tool. It is resumable: re-running it continues where it stopped.
   When it finishes, it offers to start the terminal chat. The [sog-setup](../sog-setup.md) page describes every
   phase, flag and subcommand, including the scripted, prompt-free run for servers and CI.

```{tip}
In a hurry? `sog-setup --skip-install` configures the provider and goes straight to the chat without building the
tool environments (a small agent-core environment is still built if none exists). You can ask questions right away;
come back and run `sog-setup` again to add the analysis tools.
```

## Configure the provider by hand

`sog-setup` writes the `.env` file for you. To do it yourself instead, copy the template and fill in the provider you
use:

```bash
cp .env.example .env
```

Only three things are required: `SOG_SOURCE` (the provider), `SOG_LLM` (the model or deployment name), and the key
block for that provider. For example, for Anthropic:

```ini
SOG_SOURCE=Anthropic
SOG_LLM=claude-opus-4-8
ANTHROPIC_API_KEY=<your key>
```

Every setting, and the key block of every provider, is listed on the [configuration](../configuration.md) page.

## Install without a clone

There is no PyPI package. To install the package straight from GitHub into an existing environment:

```bash
pip install "spatialomicsgym @ git+https://github.com/jiangyi01/SpatialOmicsLab.git"
```

A wheel install carries the agent, the MCP portals and the tool recipes, but not the repository's working tree, so
its runtime state (setup progress, the generated MCP config, saved keys) lives under `$SOG_HOME`, which defaults to
`~/.spatialomicsgym`. `sog-setup` seeds that directory on its first run. In a clone, the same state lives in
`.sog_setup/` at the repository root.

## Optional extras

| Extra | Installs | For |
|---|---|---|
| `pip install -e ".[tuning]"` | Optuna | hyperparameter tuning (`SOG_TUNING_ENABLED`) |
| `pip install -e ".[dev]"` | pytest, ruff | contributing; see [Contributing](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/CONTRIBUTION.md) |

The agent environment deliberately does **not** contain the heavy scientific stacks (torch, squidpy, R, ...). Those
belong to the per-tool environments that `sog-setup` builds, which is what keeps methods with incompatible
dependencies from colliding.

## Upgrade

In a clone, pull and reinstall:

```bash
git pull
pip install -e .
sog-setup            # resumes; rebuilds only what changed
```

`sog-setup doctor` reports the health of every tool environment after an upgrade (see [Verify the install](verify.md)).

## Move an install to another machine

`sog-setup pack` writes a one-file bundle of a working install, including tools you created; `sog-setup unpack BUNDLE`
restores it on the destination. Keys never travel in the bundle. Details are in [sog-setup: pack and unpack](../sog-setup.md#pack-and-unpack).

## Next

[Quick start](quickstart.md): a first question from the terminal, then a first analysis from Python.
