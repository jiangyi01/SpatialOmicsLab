# Benchmark Workflows

Pre-defined benchmark workflows that combine strategies, runners, and reporting.

## Available Workflows

### Smoke Test (`smoke_test_workflow.py`)
Quick validation that each tool starts, processes data, and returns valid JSON.
- Timeout: 120s per tool
- Metrics: Basic (ARI, NMI only)
- Purpose: CI/CD gate, pre-merge validation

### Full Benchmark (`full_benchmark_workflow.py`)
Comprehensive evaluation of all tools against all compatible datasets.
- Timeout: Configurable (default 600s)
- Metrics: Full suite (ARI, NMI, Jaccard, RMSE, Pearson, Spearman)
- Optional regression detection against saved baselines
- Purpose: Release validation, performance tracking

## Adding New Workflows

1. Create a new file in this directory.
2. Compose existing strategies and runners.
3. Use `ReportGenerator` for output.
4. Follow the pattern: `configure() -> run() -> run_and_report()`.
