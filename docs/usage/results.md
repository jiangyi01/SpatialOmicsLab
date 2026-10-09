# Results and reports

Each analysis leaves four kinds of output on disk: the conversation log, the result files a method wrote (h5ad/csv),
the figures, and an HTML report. This page says where each one lands and what it contains.

## Where output goes

A run writes to more than one root, and the split is deliberate.

| Root | Who writes there | What |
|---|---|---|
| `<data path>/outputs/` | Code the agent writes itself in its REPL (`run_python_repl`). The agent's data path is `SOG_PATH` / `SOG_DATA_PATH`, default `./data`. | Ad-hoc tables, figures and intermediate files the model saves. |
| the `output_dir` you (or the agent) pass to a tool | The MCP tool's worker, in its own conda environment. | The method's own output: a result `.h5ad`, csv tables, figures. |
| `$SOG_WORK_DIR`, else a writable `/workspace/work`, else `./work` | Every MCP tool whose call omitted `output_dir`. | The same, under a per-tool directory. |

Every tool's parameter table on the [tool pages](../tools/index.md) names the files it writes and the `output_dir`
convention it follows (most take `./work/<something>` under the run's working directory). The agent prints the real
paths in the terminal; a report or an exported transcript shows them relative to the root they are under, so a file
that travels off the machine does not carry a home directory with it.

## What a tool writes

A worker replies to its portal with a standard JSON document:

```text
status          "ok" or "error"
tool, task      which function ran
data            the input's dimensions (n_spots, n_genes, ...) and what was actually used
output_files    every file written, by role ("annotated_h5ad", "clusters_csv", ...)
params          the parameters that ran, including the defaults the worker filled in and, under
                params.ignored, any it did not use
summary         the results and metrics (n_clusters, cluster sizes, top genes, ...)
analysis        a human-readable interpretation
error, traceback   only on error
```

The portal passes this back to the agent as the tool's observation, which is why the agent can say *which*
resolution a clustering ran with, or that spots outside the tissue were left out. The same record is what the
post-analysis step reads.

Result files follow the method: most Python tools write a new `.h5ad` with the result in `obs` (a domain label, a
cell-type abundance column per type) or `obsm`, plus a csv of the same table; R tools write csv, and many also an
`.rds`. The [data conversion](../tools/data_conversion.md) tools translate between h5ad, csv and Seurat RDS when the
next step needs another format.

## Post-analysis and the HTML report

After a tool runs, the post-analysis step (`SOG_POST_ANALYSIS_ENABLED`, on by default) scans the result, draws the
figures appropriate to the task type (spatial domain maps, abundance maps, SVG rankings, alignment overlays, ...),
reviews them and proposes a next step. It writes a `manifest.json` next to the results and renders a self-contained
`report.html` from it: one page with the figures embedded, the summary tables and the parameters, which opens in any
browser with nothing else installed.

To render reports by hand, for a run copied off a cluster or one made before the report generator existed:

```bash
python -m spatialomicsgym.report <results_dir>        # one run
python -m spatialomicsgym.report --all <results_root> # every run under a root
python -m spatialomicsgym.report                      # every run under this install's output roots
```

It needs only the agent environment (stdlib, no plotting, no agent). Exit status 0 means every run found got its
report; 1 that some did not, or no run was found; 2 that none was written.

## The conversation

The terminal chat keeps the session's questions and answers (`/history`) and saves them as a PDF with the captured
figures (`/save`) or as a Markdown transcript (`/export`). From Python, `agent.log` holds the step log of the last
turn and `agent.save_conversation_history("file.pdf")` writes the PDF. Both include the code the agent ran and what
each tool returned, which is the record to keep with a result.

## Reproducing a run

Two things identify what produced a result:

- `python -m spatialomicsgym.source_pin --json` prints `AGENT_SRC_HASH` and `KNOW_HOW_HASH`, the fingerprints of the
  agent source and of the know-how corpus that went into every prompt.
- The `params` block of each tool's reply records the parameters that actually ran, including the defaults the
  worker filled in.

Benchmark and scored runs (`SOG_BENCHMARKING_ENABLED`) make these mandatory and add output inspection before any
metric is computed; see `agent/benchmarks/README.md` in the repository.
