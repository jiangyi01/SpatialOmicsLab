# Phylogenetics

## Metadata

**Short Description**: Build and analyze phylogenetic trees using MAFFT (multiple alignment), IQ-TREE 2 (maximum likelihood), and FastTree (fast NJ/ML). Visualize with ETE 4 or FigTree. For evolutionary analysis, microbial genomics, viral phylodynamics, protein family analysis, and molecular clock studies.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/phylogenetics/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: not stated upstream for the tool; the upstream frontmatter `license` field reads "Unknown", and upstream uses that field for the skill text in some files and for the tool in others, so it is not taken as the tool's licence
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 install fence(s) replaced by a provisioning pointer; 8 manifest replacement(s) and 0 excision(s) applied.

---

## Overview

Phylogenetic analysis reconstructs the evolutionary history of biological sequences (genes, proteins, genomes) by inferring the branching pattern of descent. This skill covers the standard pipeline:

1. **MAFFT** — Multiple sequence alignment
2. **IQ-TREE 2** — Maximum likelihood tree inference with model selection
3. **FastTree** — Fast approximate maximum likelihood (for large datasets)
4. **ETE 4** (`ete4` 4.4.0) — Python library for tree manipulation and visualization. The code below is ETE 4; ETE 3's `ete3` API differs (`format=` became `parser=`, `is_leaf()` the `is_leaf` property)

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

## When to Use This Skill

Use phylogenetics when:

- **Evolutionary relationships**: Which organism/gene is most closely related to my sequence?
- **Viral phylodynamics**: Trace outbreak spread and estimate transmission dates
- **Protein family analysis**: Infer evolutionary relationships within a gene family
- **Horizontal gene transfer detection**: Identify genes with discordant species/gene trees
- **Ancestral sequence reconstruction**: Infer ancestral protein sequences
- **Molecular clock analysis**: Estimate divergence dates using temporal sampling
- **GWAS companion**: Place variants in evolutionary context (e.g., SARS-CoV-2 variants)
- **Microbiology**: Species phylogeny from 16S rRNA or core genome phylogeny

## Standard Workflow

### 1. Multiple Sequence Alignment with MAFFT

```python
import subprocess
import os

def run_mafft(input_fasta: str, output_fasta: str, method: str = "auto",
               n_threads: int = 4) -> str:
    """
    Align sequences with MAFFT.

    Args:
        input_fasta: Path to unaligned FASTA file
        output_fasta: Path for aligned output
        method: 'auto' (auto-select), 'einsi' (accurate), 'linsi' (accurate, slow),
                'fftnsi' (medium), 'fftns' (fast), 'retree2' (fast)
        n_threads: Number of CPU threads

    Returns:
        Path to aligned FASTA file
    """
    methods = {
        "auto": ["mafft", "--auto"],
        "einsi": ["mafft", "--genafpair", "--maxiterate", "1000"],
        "linsi": ["mafft", "--localpair", "--maxiterate", "1000"],
        "fftnsi": ["mafft", "--fftnsi"],
        "fftns": ["mafft", "--fftns"],
        "retree2": ["mafft", "--retree", "2"],
    }

    cmd = methods.get(method, methods["auto"])
    cmd += ["--thread", str(n_threads), "--inputorder", input_fasta]

    with open(output_fasta, 'w') as out:
        result = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"MAFFT failed:\n{result.stderr}")

    # Count aligned sequences
    with open(output_fasta) as f:
        n_seqs = sum(1 for line in f if line.startswith('>'))
    print(f"MAFFT: aligned {n_seqs} sequences → {output_fasta}")

    return output_fasta

# MAFFT method selection guide:
# Few sequences (<200), accurate: linsi or einsi
# Many sequences (<1000), moderate: fftnsi
# Large datasets (>1000): fftns or auto
# Ultra-fast (>10000): mafft --retree 1
```

### 2. Trim Alignment (Optional but Recommended)

```python
def trim_alignment_trimal(aligned_fasta: str, output_fasta: str,
                            method: str = "automated1") -> str:
    """
    Trim poorly aligned columns with TrimAl.

    Methods:
    - 'automated1': Automatic heuristic (recommended)
    - 'gappyout': Remove gappy columns
    - 'strict': Strict gap threshold
    """
    cmd = ["trimal", f"-{method}", "-in", aligned_fasta, "-out", output_fasta, "-fasta"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"TrimAl warning: {result.stderr}")
        # Fall back to using the untrimmed alignment
        import shutil
        shutil.copy(aligned_fasta, output_fasta)
    return output_fasta
```

### 3. IQ-TREE 2 — Maximum Likelihood Tree

```python
def run_iqtree(aligned_fasta: str, output_prefix: str,
                model: str = "TEST", bootstrap: int = 1000,
                n_threads: int = 4, extra_args: list = None) -> dict:
    """
    Build a maximum likelihood tree with IQ-TREE 2.

    Args:
        aligned_fasta: Aligned FASTA file
        output_prefix: Prefix for output files
        model: 'TEST' for automatic model selection, or specify (e.g., 'GTR+G' for DNA,
               'LG+G4' for proteins, 'JTT+G' for proteins)
        bootstrap: Number of ultrafast bootstrap replicates (1000 recommended)
        n_threads: Number of threads ('AUTO' to auto-detect)
        extra_args: Additional IQ-TREE arguments

    Returns:
        Dict with paths to output files
    """
    cmd = [
        "iqtree2",
        "-s", aligned_fasta,
        "--prefix", output_prefix,
        "-m", model,
        "-B", str(bootstrap),   # Ultrafast bootstrap
        "-T", str(n_threads),
        "--redo"                # Overwrite existing results
    ]

    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"IQ-TREE failed:\n{result.stderr}")

    # Print model selection result
    log_file = f"{output_prefix}.log"
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                if "Best-fit model" in line:
                    print(f"IQ-TREE: {line.strip()}")

    output_files = {
        "tree": f"{output_prefix}.treefile",
        "log": f"{output_prefix}.log",
        "iqtree": f"{output_prefix}.iqtree",  # Full report
        "model": f"{output_prefix}.model.gz",
    }

    print(f"IQ-TREE: Tree saved to {output_files['tree']}")
    return output_files

# IQ-TREE model selection guide:
# DNA:     TEST → GTR+G, HKY+G, TrN+G
# Protein: TEST → LG+G4, WAG+G, JTT+G, Q.pfam+G
# Codon:   TEST → MG+F3X4

# For temporal (molecular clock) analysis, add:
# extra_args = ["--date", "dates.txt", "--clock-test", "--date-CI", "95"]
```

### 4. FastTree — Fast Approximate ML

For large datasets (>1000 sequences) where IQ-TREE is too slow:

```python
def run_fasttree(aligned_fasta: str, output_tree: str,
                  sequence_type: str = "nt", model: str = "gtr",
                  n_threads: int = 4) -> str:
    """
    Build a fast approximate ML tree with FastTree.

    Args:
        sequence_type: 'nt' for nucleotide or 'aa' for amino acid
        model: For nt: 'gtr' (recommended) or 'jc'; for aa: 'lg', 'wag', 'jtt'
    """
    if sequence_type == "nt":
        cmd = ["FastTree", "-nt", "-gtr"]
    else:
        cmd = ["FastTree", f"-{model}"]

    cmd += [aligned_fasta]

    with open(output_tree, 'w') as out:
        result = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FastTree failed:\n{result.stderr}")

    print(f"FastTree: Tree saved to {output_tree}")
    return output_tree
```

### 5. Tree Analysis and Visualization with ETE 4

```python
from ete4 import Tree


def load_tree(tree_file: str) -> Tree:
    """Load a Newick tree file. ETE 4 reads an open file object (or Newick text), not a path."""
    with open(tree_file) as handle:
        t = Tree(handle)
    print(f"Tree: {len(t)} leaves, {sum(1 for _ in t.traverse())} nodes")
    return t

def basic_tree_stats(t: Tree) -> dict:
    """Basic tree statistics over ALL leaves -- exact, no leaf subsample."""
    nodes = list(t.traverse("postorder"))  # children before their parent
    n = sum(1 for node in nodes if node.is_leaf)
    below, depth, diameter = {}, {}, 0.0
    for node in nodes:
        key = id(node)
        if node.is_leaf:
            below[key], depth[key] = 1, 0.0
            continue
        below[key] = sum(below[id(c)] for c in node.children)
        reach = sorted(((c.dist or 0.0) + depth[id(c)] for c in node.children), reverse=True)
        depth[key] = reach[0]
        diameter = max(diameter, sum(reach[:2]))  # longest leaf-to-leaf path through this node
    # every branch lies on the path of each leaf pair it separates
    crossed = sum((node.dist or 0.0) * below[id(node)] * (n - below[id(node)]) for node in nodes if not node.is_root)
    n_pairs = n * (n - 1) // 2

    stats = {
        "n_leaves": n,
        "n_internal_nodes": len(nodes) - n,
        "total_branch_length": sum(node.dist or 0.0 for node in nodes),
        "max_leaf_distance": diameter,
        "mean_leaf_distance": crossed / n_pairs if n_pairs else 0.0,
    }
    return stats

def find_mrca(t: Tree, leaf_names: list) -> Tree:
    """Find the most recent common ancestor of a set of leaves."""
    return t.common_ancestor(leaf_names)

def visualize_tree(t: Tree, output_file: str = "tree.png",
                    show_branch_support: bool = True,
                    color_groups: dict = None,
                    width: int = 800) -> None:
    """
    Render phylogenetic tree to image with ETE 4's optional Qt treeview (ete4[treeview]).

    Args:
        t: ETE 4 Tree object
        color_groups: Dict mapping leaf_name → color (for coloring taxa)
        show_branch_support: Show bootstrap values
    """
    from ete4.treeview import NodeStyle, TreeStyle

    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.show_branch_support = show_branch_support
    ts.mode = "r"  # 'r' = rectangular, 'c' = circular

    if color_groups:
        for node in t.leaves():
            if node.name in color_groups:
                nstyle = NodeStyle()
                nstyle["fgcolor"] = color_groups[node.name]
                nstyle["size"] = 8
                node.set_style(nstyle)

    t.render(output_file, tree_style=ts, w=width, units="px")
    print(f"Tree saved to: {output_file}")

def midpoint_root(t: Tree) -> Tree:
    """Root tree at midpoint (use when outgroup unknown)."""
    t.set_midpoint_outgroup()
    return t

def prune_tree(t: Tree, keep_leaves: list) -> Tree:
    """Prune tree to keep only specified leaves."""
    t.prune(keep_leaves, preserve_branch_length=True)
    return t
```

### 6. Complete Analysis Script

```python
import subprocess, os
from ete4 import Tree

def full_phylogenetic_analysis(
    input_fasta: str,
    output_dir: str = "phylo_results",
    sequence_type: str = "nt",
    n_threads: int = 4,
    bootstrap: int = 1000,
    use_fasttree: bool = False
) -> dict:
    """
    Complete phylogenetic pipeline: align → trim → tree → visualize.

    Args:
        input_fasta: Unaligned FASTA
        sequence_type: 'nt' (nucleotide) or 'aa' (amino acid/protein)
        use_fasttree: Use FastTree instead of IQ-TREE (faster for large datasets)
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, "phylo")

    print("=" * 50)
    print("Step 1: Multiple Sequence Alignment (MAFFT)")
    aligned = run_mafft(input_fasta, f"{prefix}_aligned.fasta",
                         method="auto", n_threads=n_threads)

    print("\nStep 2: Tree Inference")
    if use_fasttree:
        tree_file = run_fasttree(
            aligned, f"{prefix}.tree",
            sequence_type=sequence_type,
            model="gtr" if sequence_type == "nt" else "lg"
        )
    else:
        model = "TEST" if sequence_type == "nt" else "TEST"
        iqtree_files = run_iqtree(
            aligned, prefix,
            model=model,
            bootstrap=bootstrap,
            n_threads=n_threads
        )
        tree_file = iqtree_files["tree"]

    print("\nStep 3: Tree Analysis")
    t = load_tree(tree_file)
    t = midpoint_root(t)

    stats = basic_tree_stats(t)
    print(f"Tree statistics: {stats}")

    print("\nStep 4: Visualization")
    visualize_tree(t, f"{prefix}_tree.png", show_branch_support=True)

    # Save rooted tree
    rooted_tree_file = f"{prefix}_rooted.nwk"
    t.write(outfile=rooted_tree_file, parser=1)

    results = {
        "aligned_fasta": aligned,
        "tree_file": tree_file,
        "rooted_tree": rooted_tree_file,
        "visualization": f"{prefix}_tree.png",
        "stats": stats
    }

    print("\n" + "=" * 50)
    print("Phylogenetic analysis complete!")
    print(f"Results in: {output_dir}/")
    return results
```

## IQ-TREE Model Guide

### DNA Models

| Model | Description | Use case |
|-------|-------------|---------|
| `GTR+G4` | General Time Reversible + Gamma | Most flexible DNA model |
| `HKY+G4` | Hasegawa-Kishino-Yano + Gamma | Two-rate model (common) |
| `TrN+G4` | Tamura-Nei | Unequal transitions |
| `JC` | Jukes-Cantor | Simplest; all rates equal |

### Protein Models

| Model | Description | Use case |
|-------|-------------|---------|
| `LG+G4` | Le-Gascuel + Gamma | Best average protein model |
| `WAG+G4` | Whelan-Goldman | Widely used |
| `JTT+G4` | Jones-Taylor-Thornton | Classical model |
| `Q.pfam+G4` | pfam-trained | For Pfam-like protein families |
| `Q.bird+G4` | Bird-specific | Vertebrate proteins |

**Tip:** Use `-m TEST` to let IQ-TREE automatically select the best model.

## Best Practices

- **Alignment quality first**: Poor alignment → unreliable trees; check alignment manually
- **Use `linsi` for small (<200 seq), `fftns` or `auto` for large alignments**
- **Model selection**: Always use `-m TEST` for IQ-TREE unless you have a specific reason
- **Bootstrap**: Use ≥1000 ultrafast bootstraps (`-B 1000`) for branch support
- **Root the tree**: Unrooted trees can be misleading; use outgroup or midpoint rooting
- **FastTree for >5000 sequences**: IQ-TREE becomes slow; FastTree is 10–100× faster
- **Trim long alignments**: TrimAl removes unreliable columns; improves tree accuracy
- **Check for recombination** in viral/bacterial sequences before building trees (`RDP4`, `GARD`)

## Additional Resources

- **MAFFT**: https://mafft.cbrc.jp/alignment/software/
- **IQ-TREE 2**: http://www.iqtree.org/ | Tutorial: https://www.iqtree.org/workshop/molevol2022
- **FastTree**: http://www.microbesonline.org/fasttree/
- **ETE 4**: https://etetoolkit.github.io/ete/ (etetoolkit.org/docs/latest documents ETE 3)
- **FigTree** (GUI visualization): https://tree.bio.ed.ac.uk/software/figtree/
- **iTOL** (web visualization): https://itol.embl.de/
- **MUSCLE** (alternative aligner): https://www.drive5.com/muscle/
- **TrimAl** (alignment trimming): https://vicfero.github.io/trimal/
