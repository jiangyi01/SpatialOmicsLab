# Skills Module

Standalone module for managing domain-specific analysis skills in SpatialOmicsLab.

## Architecture

```
skills/
├── base.py                  # BaseSkill abstract class
├── registry.py              # SkillRegistry - central skill management
├── spatial_clustering/      # 14 tools: scanpy, graphst, stagate, deepst, ...
├── deconvolution/           # 31 tools: cell2location, tangram, rctd, ...
├── svg_detection/           # 10 tools: hotspot, somde, spatialde, ...
├── cell_segmentation/       # 4 tools: cellpose, deepcell, bidcell, clustermap
├── spatial_alignment/       # 18 tools: paste, stalign, gpsa, spiral, 3D diagnosis, ...
├── spatial_communication/   # 7 tools: commot, ncem, deeplinc, mistyr, ...
├── spatial/                 # 57 tools: squidpy, seurat, QC, imputation, pathway enrichment, visualization, ...
├── data_conversion/         # 4 tools: h5ad/csv/seurat converters
└── omics/                   # 7 tools: arxiv, biorxiv, doi, crossref, jgi
```

Total: 152 tools across 9 skill domains. That is the count of *MCP-backed* tools -- see
"What belongs in this catalog" below for the much larger set of tools that deliberately are not here.
The per-domain counts and this total are checked against `SkillRegistry.create_default()` by
`test/test_the_skills_readme_counts_the_catalog_it_describes.py` (they had drifted to 11 / 54 / 142
after the 3D rows landed; hunt 2026-09-30, u29b-skills-config-10).

## Design Principles

1. **Isolation**: Skills module does not import from `spatialomicsgym/` core.
2. **Registry-based**: All skills register through `SkillRegistry` for discoverability.
3. **Extensible**: Add new skill domains by creating a new subdirectory with `skill.py` and `mcp_tools.py`.
4. **Knowledge-driven**: Each skill encapsulates domain knowledge about which tools to use and when.

## Adding a New Skill Domain

1. Create a new directory under `skills/` (e.g., `skills/proteomics/`).
2. Create `__init__.py`, `skill.py` (subclass `BaseSkill`), and `mcp_tools.py`.
3. Register in `skills/registry.py` `create_default()` or programmatically via `SkillRegistry.register()`.

## What belongs in this catalog

`mcp_tools.py` is a catalog of **MCP-backed** tools, and only those. It exists because an MCP tool is
invisible to the agent's primary tool index: `read_module2api()` globs `agent/spatialomicsgym/tool/tool_description/*.py`,
an MCP tool has no file there, so without an entry here the retriever can never rank it. Two regression
tests pin that gap from real incidents --
`test/test_a_registered_tool_no_skill_names_is_one_the_model_never_sees.py` (a tool that was callable
but unrankable) and `test/test_every_recommended_tool_is_in_the_catalog_the_retriever_sees.py`
(a tool the know-how recommended by name that retrieval could not reach).

**A static two-file tool needs no entry here, and should not have one.** It is auto-discovered by the
`tool_description/*.py` glob, so it is already a first-class retrieval candidate carrying its own full
description; `backend/sog_portal/skills_api.py` enumerates it on `/skills` from the same glob. Adding
it here would not make it more discoverable -- it is already discoverable -- and it would make everything
else less so, because `STCoscientist` renders this whole catalog into one string that
`agent/execution.py` injects as the `skills_catalog` *candidate description*, scored on every query.
Measured at the ToolUniverse vendoring: the catalog is 8,569 characters, and filing the 107 vendored
static tools here would have taken it to roughly 16,900 -- double the per-query cost, under a caption
that hardcodes the word "MCP" and eight analysis task names none of them belong to.

So the 107 tools vendored from ToolUniverse (`gene_identifiers`, `ontology`, `scholarly_literature`,
`expression_atlases`, `spatial_atlases`, `model_organisms` -- see `VENDORING.md`) are static two-file
tools and are **not** listed here. They reach the model through `module2api` -> `ToolRegistry` like the
other 269 static tools.

**The merged skill packs are not here either, and must never be.** The 56 documents rendered from
`SciAgent-Skills` and `scientific-agent-skills` (`VENDORING.md` §12) are **know-how tier 2**: they live
under `agent/spatialomicsgym/know_how/packs/`, reach a turn only through the retriever's *second* pass
(`ToolRetriever.retrieve_packs`, after this catalog and the tier-1 playbooks have already been
selected), never on a benchmark run, and are off by default until the retrieval measurement says
otherwise. None of them is an `mcp_tools.py` entry — there is no MCP callable behind any of them — and
none of them appears in the `skills_catalog` string the retriever scores on every query. Filing one here
would put ~400 characters of description into every turn's tier-1 ranking for prose that has no tool
to route to, which is exactly the per-query cost this section exists to refuse.
