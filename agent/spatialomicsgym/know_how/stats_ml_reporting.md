# Bio Stats ML Reporting (from omics-skills)

Source: https://github.com/fmschulz/omics-skills

Aggregate results, train ML models, and produce reports with validated references.

## Overview
Aggregating results into tables, fitting and reporting a model without overstating it, and writing up findings whose references were actually verified, with the quality gates a report must pass.

## Instructions

1. Join outputs in DuckDB and build feature tables.
2. Train baseline models and evaluate with cross-validation.
3. Generate reports and validate references.

## Quick Reference

| Task | Action |
|------|--------|
| Run workflow | Follow the steps in this skill and capture outputs. |
| Validate inputs | Confirm required inputs and reference data exist. |
| Review outputs | Inspect reports and QC gates before proceeding. |
| Tool docs | `MCP_server/mcp_config.yaml` is the list of tools this installation has. |
| References | Verify every citation; an unverified DOI belongs in an unverified section. |

## Input Requirements

Prerequisites:
- Tools available in the active environment (conda/system), as listed in `MCP_server/mcp_config.yaml`.
- Results tables and metadata are available.
Inputs:
- results/*.parquet or results/*.tsv
- metadata.tsv

## Output

- results/bio-stats-ml-reporting/models/
- results/bio-stats-ml-reporting/metrics.tsv
- results/bio-stats-ml-reporting/report.md
- results/bio-stats-ml-reporting/logs/

## Quality Gates

- [ ] Model performance sanity checks pass.
- [ ] Reference validation passes.
- [ ] On failure: retry with alternative parameters; if still failing, record in report and exit non-zero.
- [ ] Verify input tables are readable and schema-consistent.

## Examples

### Example 1: Expected input layout

```text
results/*.parquet or results/*.tsv
metadata.tsv
```

## Troubleshooting

**Issue**: Missing inputs or reference databases
**Solution**: Verify paths and permissions before running the workflow.

**Issue**: Low-quality results or failed QC gates
**Solution**: Review reports, adjust parameters, and re-run the affected step.
