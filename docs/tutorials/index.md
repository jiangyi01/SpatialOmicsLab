# Tutorials

Worked examples, each a complete analysis from a question to a report. They assume a finished
[installation](../getting-started/installation.md) with the relevant tool environments built.

| Tutorial | What it shows | Tools |
|---|---|---|
| [Spatial domains in a Visium slide](visium-spatial-domains.md) | Preparing an `.h5ad`, asking for spatial domains, reading the result and the report, refining the request, comparing methods. | GraphST, STAGATE, the visualisation server |

More tutorials are planned: deconvolution against a single-cell reference (cell2location, RCTD), spatially variable
genes (SpatialDE, SPARK), aligning serial sections into a 3D stack (PASTE, STalign), and letting the agent build a
tool from a GitHub repository. Contributions are welcome: a tutorial is a Markdown page under `docs/tutorials/`, and
the [contribution guide](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/CONTRIBUTION.md) explains how
to submit one.

```{toctree}
:hidden:
:maxdepth: 1

visium-spatial-domains
```
