# Spatial domains in a Visium slide

The most common first analysis: partition a 10x Visium slide into spatial domains, look at them, and ask what
distinguishes each one. This tutorial uses the public human lymph node slide from 10x Genomics, which scanpy can
download, and runs the agent from the terminal and from Python.

**You need:** a finished installation with the `spatial_clustering` category built (at least `graphst`; `stagate` for
the comparison at the end), and the agent environment active.

## 1. Prepare the data

The spatial clustering tools take an AnnData `.h5ad` with **raw counts** in `X` and spot coordinates in
`obsm['spatial']`, which is exactly what Space Ranger output read with scanpy gives you. Download the lymph node
slide and save it:

```python
import scanpy as sc

adata = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node")   # downloads from 10x Genomics
adata.var_names_make_unique()
adata.write_h5ad("data/V1_Human_Lymph_Node.h5ad")
```

Keep the counts raw: the tools normalise internally, and a matrix that was already scaled (negative values) is
refused with an error that says so. If your own data lives in `adata.raw`, the tools accept `use_raw_counts=true`;
if it is a Seurat object or a csv, the [data conversion](../tools/data_conversion.md) tools translate it, and you can
simply ask the agent to do that first.

```{tip}
The built-in spatial dataset library knows many public slides. Ask *"Which datasets do you have for human lymph
node?"* and the agent searches it (`search_spatial_datasets`) before you download anything.
```

## 2. Ask

From the terminal, with the tools wired:

```console
$ stcoscientist --mcp
> Identify spatial domains in data/V1_Human_Lymph_Node.h5ad with GraphST, using 7 domains,
  and write the results to work/lymph_node_domains. Then summarize each domain's marker genes.
```

Or from Python:

```python
from spatialomicsgym import clean_answer
from spatialomicsgym.agent import STCoscientist

agent = STCoscientist(path="./data")
agent.add_mcp()

log, answer = agent.go(
    "Identify spatial domains in data/V1_Human_Lymph_Node.h5ad with GraphST, using 7 domains, "
    "and write the results to work/lymph_node_domains. Then summarize each domain's marker genes."
)
print(clean_answer(answer))
agent.save_conversation_history("work/lymph_node_domains/conversation.pdf")
```

Three things in the request do real work. Naming the file lets the pre-flight diagnosis check the format before any
tool runs. Naming the method pins it (leave it out and the agent chooses from the category, preferring the methods
its leaderboard ranks highest for the data type). Giving the number of domains matters for methods that take it as a
parameter: GraphST's `n_clusters` is required, and for the others a known count steers the resolution search.

## 3. What happens

Watch the steps as they stream. A typical run:

1. The agent diagnoses the file (spots, genes, coordinates present, counts raw) and retrieves the tools and know-how
   for spatial clustering.
2. It calls `graphst_spatial_clustering(st_h5ad=..., output_dir="work/lymph_node_domains", n_clusters=7, ...)`.
   The portal runs the worker in the `GraphST` environment; the reply reports the spots used, the clustering backend
   (`mclust` by default) and the files written: an annotated `.h5ad` with the domain in `obs`, and a csv of labels.
3. Post-analysis draws the domain map over the tissue, reviews it, and writes `manifest.json` and `report.html` next
   to the results.
4. For the marker genes, the agent runs a differential expression step in its own REPL (scanpy's
   `rank_genes_groups` over the domain labels) or calls the visualisation server to draw a marker dot plot, then
   writes the summary.

The final answer names the output files and gives the per-domain summary. If the turn ended early (a timeout, a tool
error it could not recover from), the answer says so and the terminal exits with status 4; see
[Terminal chat](../usage/terminal-chat.md#exit-status-and-automation).

## 4. Read the results

Open `work/lymph_node_domains/report.html` in a browser: the domain map, the summary tables and the parameters that
ran, on one self-contained page. The files themselves:

```python
import scanpy as sc

adata = sc.read_h5ad("work/lymph_node_domains/graphst_clustering_output.h5ad")
adata.obs["domain"].value_counts()
sc.pl.spatial(adata, color="domain", spot_size=150)
```

The reply's `params` block (also in the conversation log) records every parameter that ran, including the defaults
the worker filled in, which is what to keep with the figure.

## 5. Refine

The session remembers the turn, so a follow-up need not restate the path:

```text
> Redo it with 10 domains.
> Which domains express germinal centre markers (BCL6, AICDA, MKI67)?
> Plot the expression of CR2 and CXCL13 over the tissue next to the domain map.
```

The last request goes to the visualisation server (`plot_spatial_expression`, `compose_figure`), which writes
publication-quality figures to the output directory.

## 6. Compare methods

Ask for a second method on the same slide and let the agent compare:

```text
> Run STAGATE on the same slide with 7 domains into work/lymph_node_stagate, and compare its
  domains with the GraphST result (ARI, and which domains correspond).
```

The agent runs `stagate_spatial_domains`, then computes the agreement in its REPL. For a slide with annotated ground
truth (the DLPFC reference, for instance), name the annotation column: GraphST takes it as `label_key` and reports
the ARI itself, and the agent scores the other methods against it in its REPL.

## If something goes wrong

- *"refused: X has negative values"*: the matrix is scaled. Point the tool at raw counts (`use_raw_counts=true` if they
  are in `adata.raw`, or re-export from Space Ranger output).
- *The agent has no `graphst_*` tool*: the tools were not wired (`--mcp` / `add_mcp()`), or the `GraphST` environment
  was not built. `sog-setup doctor` says which; `sog-setup --only graphst` builds it.
- *`mclust` fails*: GraphST's default backend needs R (`R_HOME`); ask for `cluster_tool='leiden'` instead, or set
  `GRAPHST_R_HOME`.
- *A step timed out*: raise `--timeout` / `SOG_TIMEOUT_SECONDS` for large slides; GraphST builds dense
  spot × spot matrices.
