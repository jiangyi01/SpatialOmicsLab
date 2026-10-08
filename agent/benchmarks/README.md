# agent/benchmarks/: the benchmark harness and evaluator

**What this is.** The top-level Python package `benchmarks`: the evaluator that scores tool outputs against ground truth, the SVG gene reader, the agent-driven smoke-test driver, and an older strategy/runner framework that only tests use. It is not `agent/spatialomicsgym/benchmarking/`, which is a separate package inside the agent.
**How runtime finds it.** As the import name `benchmarks`: the editable install maps it, pytest puts `agent/` on `sys.path`, and the tuning objectives add `agent/` themselves if the import fails. At run time only the tuning objectives import it.
**What must not change.** The import name, its place directly under `agent/`, and the path segments `benchmarks/results` and `benchmarks/`, which other code matches by name.

## What is live, what is dormant

| Module | Status | Reached from |
|---|---|---|
| `evaluation/evaluator.py`, `evaluation/celltype_aggregation/*.yaml` | live: scores clustering, deconvolution and SVG outputs | `agent/spatialomicsgym/tuning/objectives.py:69-74` (benchmark-mode tuning trials, `agent/spatialomicsgym/tuning/executor.py:429`) |
| `workflows/benchmark_runner.py` | half live: `_extract_svg_genes` (`:251`) is imported at run time; the CLI and `run_benchmark()` need the dataset registry (below) | `agent/spatialomicsgym/tuning/objectives.py:99` |
| `turn_outcome.py` | live: what a finished turn reports (`degrade_note`, `tool_was_invoked`) | the runner and the smoke-test driver |
| `config/benchmark_config.py` | `repository_root()` is live; `BenchmarkConfig` belongs to the framework | the runner, the smoke-test driver, `workflows/prompt_generator.py` |
| `workflows/spatialomicsgym_smoke_test.py`, `prompt_generator.py`, `result_recorder.py` | runnable by hand; nothing in the repository launches them | `python -m benchmarks.workflows.spatialomicsgym_smoke_test` (it runs the agent on every enabled tool, so it needs a configured LLM provider) |
| `reporting/result_schema.py`, `BenchmarkResult` in `strategies/base_strategy.py` | used by the smoke-test driver | the smoke-test driver |
| `data/data_registry.py` | used by `run_benchmark()` and the framework | the runner |
| `strategies/`, `runners/`, `reporting/report_generator.py`, `workflows/full_benchmark_workflow.py`, `config/default_config.yaml`, `config/full_benchmark_config.yaml` | dormant framework: only tests reach it, but `__init__.py` imports part of it on every `import benchmarks.<anything>` | tests |
| `strategies/integration_strategy.py`, `workflows/smoke_test_workflow.py`, `config/smoke_test_config.yaml`, `config/clustering_benchmark_config.yaml`, `config/multi_llm/*.yaml` | orphaned: nothing in code or tests uses them (`strategies/__init__.py` still imports `IntegrationStrategy`) | none |

`__init__.py` imports `config`, `data`, `reporting`, `runners` and `strategies` eagerly, and
`reporting/report_generator.py:10` imports `spatialomicsgym`, so importing the evaluator always imports the agent
package. `workflows/__init__.py` imports the smoke-test driver, so importing the runner for `_extract_svg_genes`
loads the driver too (without running it). In the other direction `spatialomicsgym` imports `benchmarks` only
inside functions, which keeps the cycle between the two packages from running at import time.

## The dataset registry is not in the repository

`benchmark_data/` (with `registry.yaml`), `manuscript/` and the archived drivers left the repository in commit
`18118d62` ("slim the repository to what the project needs to run"). What still needs them:
- `workflows/benchmark_runner.py`: the CLI and `run_benchmark()` read `benchmark_data/registry.yaml`
  (`workflows/benchmark_runner.py:1904-1905`) and cannot run without it.
- `evaluation/evaluator.py`: the deconvolution scorer looks the dataset up in the registry
  (`evaluation/evaluator.py:1672,1676-1691`) for its aggregation table and reference labels; without it the entry
  is reported as `registry unreadable` and dataset-specific scoring is off.
- the Slide-seqV2 entries of the dataset map in `workflows/prompt_generator.py`, which point into `benchmark_data/`.

The run-time paths (tuning scoring, the SVG gene reader) and the smoke-test driver do not need it.
[data/README.md](data/README.md) describes the registry layout the evaluator reads.

## How runtime finds it

| Mechanism | Site |
|---|---|
| Editable install maps `benchmarks` to this directory by absolute path; moving it needs `pip install -e .` again in every env | `pyproject.toml:189-190` |
| pytest | `pyproject.toml:235` (`pythonpath = ["agent"]`) |
| The tuning fallback inserts `agent/` (three levels up from its own file) when the first import fails | `agent/spatialomicsgym/tuning/objectives.py:69-74` |
| The runner and the smoke-test driver find `agent/` three levels up from their own files | `workflows/benchmark_runner.py:22`, `workflows/spatialomicsgym_smoke_test.py:40`, `workflows/prompt_generator.py:263` |
| The evaluator puts `agent/tools/` on `sys.path` at import, for `eval_metrics` | `evaluation/evaluator.py:16-18` |
| `repository_root()` maps `agent/` to the repository root, where `data/` and `test/test_data/` live | `config/benchmark_config.py:13-24` |

## What must not change

- **The import name `benchmarks`** and its place directly under `agent/`: the tuning objectives, the tests and the
  anchors above depend on both.
- **`benchmarks/results`.** The post-analysis write guard refuses any path where `benchmarks` is directly followed
  by `results` (`agent/spatialomicsgym/postanalysis/sources.py:318-334`); the tuning cache defaults to
  `agent/benchmarks/results/tuning` (`agent/spatialomicsgym/tuning/persistence.py:51`); post-analysis discovery
  skips any directory named `benchmarks` (`agent/spatialomicsgym/postanalysis/autorun.py:92,313`). `results/` is
  ignored by git and holds recorded outputs; never delete `results/tuning/`.
- **`benchmarks/` in a dataset path** routes tuning into benchmark mode (`agent/spatialomicsgym/tuning/mode_router.py:36`).
- **Do not move code across the pin boundary.** This directory is outside `AGENT_SRC_HASH`, which folds only the `.py` files
  under `agent/spatialomicsgym/` and `agent/tools/`. Moving a module from here into either of those, or from
  `agent/spatialomicsgym/benchmarking/` into here, changes the pin.

## Packaging

A wheel ships the Python modules here as the top-level package `benchmarks`, without `manuscript/`, `results/` and
`benchmark_data/` (`pyproject.toml:199-208`). The YAML files here are not package data, so a wheel install has no
`evaluation/celltype_aggregation/*.yaml` and deconvolution aggregation is off there. `install/sog_install/bundle.py:113`
leaves this directory out of the bundle's code scan.

## Keeping this page true

`test/test_the_docs_survive_first_contact.py` skips everything under `agent/benchmarks/`, so a dotted
`spatialomicsgym...` name written here is not checked; `test/test_docs_commands_resolve.py` still checks every
command on this page.
