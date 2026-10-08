import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

from spatialomicsgym.llm import get_llm
from spatialomicsgym.utils.http_client import HttpError, request_text

# esm, gget, gseapy, torch and pybiomart are imported inside the tools that use them. Imported here,
# they made all 19 tools in this module unimportable in the agent env, which has none of them, while
# the prompt tells the model to import the tools from this module -- tools that need only scanpy or
# pybedtools included (hunt 2026-09-30, uT4-genomics-3).

#: Ensembl REST is reached through utils.http_client (timeout, retries, egress policy) rather than a
#: bare requests.get with no timeout (hunt 2026-09-30, uT4-genomics-22).
_ENSEMBL_HOSTS = ("rest.ensembl.org",)

#: Times annotate_celltype_scRNA asks the model for one cluster before recording it as unassigned.
_ANNOTATION_ATTEMPTS = 3


def _replace_into(path, write):
    """Write through ``path + '.partial'`` and move it into place, so a crash leaves no half file."""
    partial = f"{path}.partial"
    write(partial)
    os.replace(partial, path)


def _default_output_dir(subdir=""):
    """The run's output directory (``$SOG_WORK_DIR`` when set), created on use.

    Tools here wrote fixed names into the input's ``data_dir`` -- often the shared dataset library,
    against the prompt's rule never to write next to the inputs -- or into the process CWD, which the
    portal's REPL shares between every chat of one account (hunt 2026-09-30, uT4-genomics-14/-29).
    """
    from spatialomicsgym.paths import tool_output_root

    path = tool_output_root(subdir)
    os.makedirs(path, exist_ok=True)
    return path


#: ``(abspath, st_mtime_ns, st_size)`` -> frozen cell-ontology name set. One entry is
#: plenty (every call in a session reads the same data-lake parquet); the key moves if the
#: file is replaced, so a refreshed data lake re-reads. Mirrors
#: ``prompt_builder._SPATIAL_DIAGNOSIS_MEMO``.
_CZI_CELLTYPE_MEMO: dict[tuple[str, int, int], frozenset[str]] = {}
_CZI_CELLTYPE_MEMO_CAP = 2


def _czi_celltype_names(parquet_path):
    """Frozen set of stripped cell-ontology names from the CZI census parquet.

    Memoized per (path, mtime, size) so a session that annotates many clusters stops
    re-reading the multi-hundred-MB parquet on every call; a miss reads only the one
    column it needs. A failed stat falls through to a direct, uncached read.

    The set is the primary form: names like "CD4-positive, alpha-beta T cell" embed
    commas, so the prompt-facing joined string cannot be split back into names.
    """
    key = None
    try:
        st = os.stat(parquet_path)
        key = (os.path.abspath(str(parquet_path)), st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and key in _CZI_CELLTYPE_MEMO:
        return _CZI_CELLTYPE_MEMO[key]
    df = pd.read_parquet(parquet_path, columns=["cell_type"])
    names = frozenset(cell_type.strip() for cell_types in df["cell_type"] for cell_type in str(cell_types).split(";"))
    if key is not None:
        _CZI_CELLTYPE_MEMO[key] = names
        while len(_CZI_CELLTYPE_MEMO) > _CZI_CELLTYPE_MEMO_CAP:
            del _CZI_CELLTYPE_MEMO[next(iter(_CZI_CELLTYPE_MEMO))]
    return names


def _czi_celltype_catalogue(parquet_path):
    """Sorted, comma-joined rendering of ``_czi_celltype_names`` for prompt text."""
    return ", ".join(sorted(_czi_celltype_names(parquet_path)))


def unsupervised_celltype_transfer_between_scRNA_datasets(
    path_to_annotated_h5ad: str,
    path_to_not_annotated_h5ad: str,
    ref_labels_key: str,
    query_batch_key: str = None,
    ref_batch_key: str = None,
    CELLTYPIST=False,
    KNN_BBKNN=False,
    KNN_HARMONY=False,
    KNN_SCANORAMA=False,
    KNN_SCVI=False,
    ONCLASS=False,
    Random_Forest=False,
    SCANVI_POPV=True,
    Support_Vector=False,
    XGboost=False,
    n_jobs: int = 1,
    output_folder: str = None,
    n_samples_per_label: int = 10,
):
    """Transfer cell type labels from an annotated reference to a query dataset with popV.

    ``output_folder`` holds the trained models and ``popv_output/predictions.csv``; default
    ``popv/`` under the run's output directory (``$SOG_WORK_DIR`` when set).
    """
    import os

    import scanpy as sc

    # "./tmp/" was the CWD the portal's REPL shares between an account's chats
    # (hunt 2026-09-30, uT4-genomics-29).
    if output_folder is None:
        output_folder = _default_output_dir("popv")
    output_folder = os.path.abspath(output_folder)

    os.environ["PYTHONUTF8"] = "1"
    import popv

    steps = []
    steps.append("Starting unsupervised cell type transfer using popV")

    steps.append(f"Loading reference annotated dataset from: {path_to_annotated_h5ad}")
    ref_adata = sc.read_h5ad(path_to_annotated_h5ad)
    steps.append(f"Reference annotated dataset loaded: {ref_adata.n_obs} cells, {ref_adata.n_vars} genes")

    steps.append(f"Loading not annotated query dataset from: {path_to_not_annotated_h5ad}")
    query_adata = sc.read_h5ad(path_to_not_annotated_h5ad)
    steps.append(f"query dataset loaded: {query_adata.n_obs} cells, {query_adata.n_vars} genes")

    # this is required for scvi/scanvi based classifier
    ref_adata.layers["counts"] = ref_adata.X.copy()
    query_adata.layers["counts"] = query_adata.X.copy()

    popv.settings.n_jobs = n_jobs
    output_folder = output_folder
    os.makedirs(output_folder, exist_ok=True)
    steps.append(f"Created output folder: {output_folder}")

    steps.append("Processing query dataset for annotation with popV")
    adata = popv.preprocessing.Process_Query(
        query_adata,
        ref_adata,
        ref_labels_key=ref_labels_key,
        query_batch_key=query_batch_key,
        ref_batch_key=ref_batch_key,
        save_path_trained_models=output_folder,
        cl_obo_folder=False,
        prediction_mode="retrain",
    ).adata

    # passing arugments this way decreases chance of LLM generation and parsing errors
    flags = {
        "CELLTYPIST": CELLTYPIST,
        "KNN_BBKNN": KNN_BBKNN,
        "KNN_HARMONY": KNN_HARMONY,
        "KNN_SCANORAMA": KNN_SCANORAMA,
        "KNN_SCVI": KNN_SCVI,
        "ONCLASS": ONCLASS,
        "Random_Forest": Random_Forest,
        "SCANVI_POPV": SCANVI_POPV,
        "Support_Vector": Support_Vector,
        "XGboost": XGboost,
    }
    selected_methods = [name for name, val in flags.items() if val]
    if selected_methods:
        steps.append(f"Selected annotation methods: {', '.join(selected_methods)}")
    else:
        steps.append("No annotation methods selected")

    steps.append("Starting annotation with SCANVI_POPV method")
    popv.annotation.annotate_data(
        adata,
        methods=selected_methods,
        save_path=f"{output_folder}/popv_output",
    )

    steps.append(f"Annotation completed. Results saved to: {output_folder}/popv_output/predictions.csv")

    return "\n".join(steps)


def interspecies_gene_conversion(
    gene_list: list[str],
    source_species: str,
    target_species: str,
) -> str:
    """Convert ENSEMBL gene IDs between different species using BioMart homology mapping.

    This function converts a list of ENSEMBL gene IDs from one species to their
    homologous counterparts in another species using the Ensembl BioMart database.
    The conversion is based on one-to-one ortholog mappings between species.

    Parameters
    ----------
    gene_list : list[str]
        List of ENSEMBL gene IDs to convert (e.g., ['ENSG00000007372', 'ENSG00000181449'])
    source_species : str
        Source species name. Supported species: human, mouse, rat, zebrafish, fly,
        drosophila, worm, yeast, chicken, pig, cow, dog, macaque
    target_species : str
        Target species name. Same supported species as source_species.

    Returns
    -------
    str
        Multi-line string containing the steps performed and file paths where
        results were saved. Returns empty string if source and target species are the same.

    Raises
    ------
    ValueError
        If source_species or target_species is not in the supported species list.

    Notes
    -----
    - The function saves conversion results to a CSV file named
      '{source_species}_to_{target_species}_gene_conversion.csv' in the run's output directory
      (``$SOG_WORK_DIR`` when set) and reports its absolute path
    - '{target_species}_genes' holds the one-to-one ortholog; a gene with only one-to-many or
      many-to-many homologues has NaN there, and every homologue is listed in
      'all_{target_species}_homologs' with its type in '{target_species}_orthology_type'
    - Genes that cannot be converted will have NaN values in the output
    - If the filtered BioMart query fails, the whole homology table is fetched once and filtered
    - Species synonyms are supported (e.g., 'drosophila' maps to 'fly')

    Examples
    --------
    >>> gene_ids = ["ENSG00000007372", "ENSG00000181449"]
    >>> result = interspecies_gene_conversion(gene_ids, "human", "mouse")
    >>> print(result)
    Gene conversion results saved to: human_to_mouse_gene_conversion.csv
    """

    steps = []

    species_mapping = {
        "human": "hsapiens_gene_ensembl",
        "mouse": "mmusculus_gene_ensembl",
        "rat": "rnorvegicus_gene_ensembl",
        "zebrafish": "drerio_gene_ensembl",
        "fly": "dmelanogaster_gene_ensembl",
        "drosophila": "dmelanogaster_gene_ensembl",
        "worm": "celegans_gene_ensembl",
        "yeast": "scerevisiae_gene_ensembl",
        "chicken": "ggallus_gene_ensembl",
        "pig": "sscrofa_gene_ensembl",
        "cow": "btaurus_gene_ensembl",
        "dog": "cfamiliaris_gene_ensembl",
        "macaque": "mmulatta_gene_ensembl",
    }

    homolog_mapping = {
        "human": "hsapiens_homolog_ensembl_gene",
        "mouse": "mmusculus_homolog_ensembl_gene",
        "rat": "rnorvegicus_homolog_ensembl_gene",
        "zebrafish": "drerio_homolog_ensembl_gene",
        "fly": "dmelanogaster_homolog_ensembl_gene",
        "drosophila": "dmelanogaster_homolog_ensembl_gene",
        "worm": "celegans_homolog_ensembl_gene",
        "yeast": "scerevisiae_homolog_ensembl_gene",
        "chicken": "ggallus_homolog_ensembl_gene",
        "pig": "sscrofa_homolog_ensembl_gene",
        "cow": "btaurus_homolog_ensembl_gene",
        "dog": "cfamiliaris_homolog_ensembl_gene",
        "macaque": "mmulatta_homolog_ensembl_gene",
    }

    if source_species not in species_mapping:
        raise ValueError(
            f"Source species '{source_species}' not supported. Available species: {list(species_mapping.keys())}"
        )

    if target_species not in homolog_mapping:
        raise ValueError(
            f"Target species '{target_species}' not supported. Available species: {list(homolog_mapping.keys())}"
        )

    original_length = len(gene_list)

    if source_species == target_species:
        return "\n".join(steps)

    from pybiomart import Dataset

    source_dataset_name = species_mapping[source_species]
    homolog_attribute = homolog_mapping[target_species]
    # e.g. mmusculus_homolog_ensembl_gene -> mmusculus_homolog_orthology_type
    orthology_attribute = homolog_attribute.replace("_homolog_ensembl_gene", "_homolog_orthology_type")
    attributes = ["ensembl_gene_id", homolog_attribute, orthology_attribute]

    # use_attr_names=True: pybiomart names columns by display name ("Gene stable ID") by default, so
    # both fallbacks' all_mapping["ensembl_gene_id"] raised KeyError, cost two whole-genome downloads
    # and ended in an all-NaN CSV reported as results. The orthology type is requested so the
    # promised one-to-one mapping is real: dict(zip(...)) kept whichever homolog row came last
    # (pax6a or pax6b for human PAX6). https, not http -- with port 443, since pybiomart builds
    # host:port/path and defaults the port to 80 (hunt 2026-09-30, uT4-genomics-15).
    def _query(filters=None):
        dataset = Dataset(name=source_dataset_name, host="https://www.ensembl.org", port=443)
        return dataset.query(attributes=attributes, filters=filters, use_attr_names=True)

    try:
        mapping = _query({"link_ensembl_gene_id": gene_list})
    except Exception as e:
        steps.append(f"Filtered BioMart query failed ({e}); fetching the whole homology table instead.")
        try:
            all_mapping = _query()
        except Exception as e2:
            steps.append(f"Error: gene conversion from {source_species} to {target_species} failed: {e2}")
            return "\n".join(steps)
        mapping = all_mapping[all_mapping["ensembl_gene_id"].isin(gene_list)]

    mapping = mapping[mapping[homolog_attribute].notna() & (mapping[homolog_attribute] != "")]
    if mapping.empty:
        steps.append(
            f"Error: BioMart returned no {target_species} homologue for any of the {original_length} IDs. "
            f"They must be Ensembl gene IDs of {source_species} (e.g. ENSG... for human), not symbols."
        )
        return "\n".join(steps)

    homologs_by_gene = {}
    for gene, homolog, kind in mapping[attributes].itertuples(index=False):
        homologs_by_gene.setdefault(gene, []).append((homolog, kind))

    converted_genes, orthology_types, all_homologs = [], [], []
    for gene in gene_list:
        rows = homologs_by_gene.get(gene, [])
        one_to_one = [h for h, kind in rows if kind == "ortholog_one2one"]
        converted_genes.append(one_to_one[0] if one_to_one else np.nan)
        orthology_types.append(";".join(sorted({str(kind) for _, kind in rows})) if rows else np.nan)
        all_homologs.append(";".join(str(h) for h, _ in rows) if rows else np.nan)

    assert len(converted_genes) == original_length, "Input and output gene list shapes must be the same"

    conversion_df = pd.DataFrame(
        {
            f"{source_species}_genes": gene_list,
            f"{target_species}_genes": converted_genes,
            f"{target_species}_orthology_type": orthology_types,
            f"all_{target_species}_homologs": all_homologs,
        }
    )
    # Into the run's output directory and reported absolute; it went to the CWD under a fixed name
    # (hunt 2026-09-30, uT4-genomics-29).
    filename = os.path.abspath(
        os.path.join(_default_output_dir(), f"{source_species}_to_{target_species}_gene_conversion.csv")
    )
    _replace_into(filename, lambda p: conversion_df.to_csv(p, index=False))
    n_one = sum(isinstance(g, str) for g in converted_genes)
    n_other = sum(isinstance(h, str) for h in all_homologs) - n_one
    steps.append(
        f"{n_one} of {original_length} genes have a one-to-one {target_species} ortholog "
        f"(column {target_species}_genes); {n_other} more have only one-to-many or many-to-many homologues, "
        f"listed in all_{target_species}_homologs."
    )
    steps.append(f"Gene conversion results saved to: {filename}")

    return "\n".join(steps)


def _fetch_isoform_sequences(ensembl_gene_id: str) -> list[str]:
    """
    this is not registered as a tool
    Fetch all protein isoform FASTA sequences for a given Ensembl gene ID.
    Returns a list of protein sequences (strings) for each isoform, or an empty list if not found.
    For ENSG00000012048, removes the sequence with header containing 'ENSP00000420781.2'.

    Args:
        ensembl_gene_id: Ensembl gene identifier (e.g., "ENSG00000012048")

    Returns:
        List of protein sequences as strings

    Raises:
        HttpError: Ensembl did not answer, or answered with a non-2xx status. This returned [] for
            any non-OK response -- a 429 rate limit included -- and the caller then logged "No
            sequences ... found for gene X", which reads as a fact about the gene; the bare
            requests.get also had no timeout (hunt 2026-09-30, uT4-genomics-22).
    """
    url = f"https://rest.ensembl.org/sequence/id/{ensembl_gene_id}?type=protein;multiple_sequences=1"
    headers = {"Content-Type": "text/x-fasta"}
    text = request_text(url, allowed_hosts=_ENSEMBL_HOSTS, headers=headers, timeout=30)
    if not text.startswith(">"):
        return []
    sequences = []
    fasta_entries = text.strip().split(">")
    for entry in fasta_entries:
        if not entry.strip():
            continue
        lines = entry.strip().splitlines()
        lines[0]
        sequence = "".join(lines[1:])
        sequences.append(sequence)
    return sequences


def generate_gene_embeddings_with_ESM_models(
    ensembl_gene_ids: list[str],
    model_name: str = "esm2_t6_8M_UR50D",
    layer: int = 6,
    save_path: str | None = None,
    batch_size: int = 1,
    max_sequence_length: int = 1024,
) -> str:
    """
    Generate average protein embeddings for a list of Ensembl gene IDs.
    Memory-friendly implementation with rolling averages and small batch processing.
    Embeddings are automatically saved to disk with a default filename if no save_path is provided.

    Args:
        ensembl_gene_ids: List of Ensembl gene IDs (e.g., ["ENSG00000012048", ...])
        model_name: ESM model name to use (default: "esm2_t6_8M_UR50D")
        layer: Which layer to extract embeddings from (default: 6)
        save_path: Optional path to save embeddings as PyTorch dict. If None, uses default filename
                  format: "esm_embeddings_{model_name}_{gene_list}.pt" in the run's output directory
                  (default: None)
        batch_size: Number of sequences to process at once (default: 1)
        max_sequence_length: Maximum sequence length to process (default: 1024)

    Returns:
        String containing the steps performed during the embedding generation process
    """
    import esm
    import torch

    steps = []
    steps.append(f"Loading ESM model: {model_name}")
    # model loading take a while, once loaded for smaller models generation is relatively fast
    # running bilion model ESM models require FSDP or GPUs with 80 GB of memory
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    steps.append(f"Using device: {device}")

    steps.append("Fetching protein isoform sequences...")
    ensembl_gene_isoforms_map: dict[str, list[str]] = {}
    for ensembl_id in tqdm(ensembl_gene_ids, desc="Fetching sequences"):
        try:
            isoform_sequences = _fetch_isoform_sequences(ensembl_id)
        except HttpError as exc:
            # A failed request, not a gene without short isoforms (uT4-genomics-22).
            steps.append(f"Error: could not fetch protein sequences for {ensembl_id} from Ensembl: {exc.detail}")
            continue
        # Filter sequences by length to avoid memory issues
        filtered_sequences = [seq for seq in isoform_sequences if len(seq) <= max_sequence_length]
        if filtered_sequences:
            ensembl_gene_isoforms_map[ensembl_id] = filtered_sequences
        else:
            steps.append(f"  No sequences under {max_sequence_length} residues found for gene {ensembl_id}")

    # Prepare all data for processing
    all_data: list[tuple[str, str]] = []
    for gene, isoform_seqs in ensembl_gene_isoforms_map.items():
        for i, seq in enumerate(isoform_seqs):
            all_data.append((f"{gene}_isoform_{i}", seq))

    steps.append(f"Processing {len(all_data)} sequences in batches of {batch_size}")

    # Rolling average tracking
    gene_embeddings: dict[str, list[torch.Tensor]] = {}
    gene_avg_embedding: dict[str, np.ndarray] = {}

    # Process in small batches to avoid memory issues
    for i in tqdm(range(0, len(all_data), batch_size), desc="Processing ESM embeddings"):
        batch_data = all_data[i : i + batch_size]

        try:
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            batch_lens: torch.Tensor = (batch_tokens != alphabet.padding_idx).sum(1)

            steps.append(f"  Running ESM model on batch {i // batch_size + 1}...")
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[layer], return_contacts=False)
            token_representations: torch.Tensor = results["representations"][layer]

            # Process each sequence in the batch
            for j, tokens_len in enumerate(batch_lens):
                # Extract gene name from label (remove _isoform_X suffix)
                gene_name = batch_labels[j].split("_isoform_")[0]

                # Get sequence embedding (excluding BOS/EOS tokens)
                sequence_embedding = token_representations[j, 1 : tokens_len - 1].mean(0).detach().cpu()

                # Store embedding for this gene
                if gene_name not in gene_embeddings:
                    gene_embeddings[gene_name] = []
                gene_embeddings[gene_name].append(sequence_embedding)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                steps.append(f"  Memory error on batch {i // batch_size + 1}, reducing batch size...")
                # Clear cache and try with smaller batch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                # Process this batch one by one
                for single_item in batch_data:
                    try:
                        single_batch_labels, single_batch_strs, single_batch_tokens = batch_converter([single_item])
                        single_batch_tokens = single_batch_tokens.to(device)
                        single_batch_lens: torch.Tensor = (single_batch_tokens != alphabet.padding_idx).sum(1)

                        with torch.no_grad():
                            single_results = model(single_batch_tokens, repr_layers=[layer], return_contacts=False)
                        single_token_representations: torch.Tensor = single_results["representations"][layer]

                        gene_name = single_item[0].split("_isoform_")[0]
                        sequence_embedding = (
                            single_token_representations[0, 1 : single_batch_lens[0] - 1].mean(0).detach().cpu()
                        )

                        if gene_name not in gene_embeddings:
                            gene_embeddings[gene_name] = []
                        gene_embeddings[gene_name].append(sequence_embedding)

                    except Exception as single_e:
                        steps.append(f"    Failed to process {single_item[0]}: {str(single_e)}")
            else:
                steps.append(f"  Error processing batch {i // batch_size + 1}: {str(e)}")

    # Compute final averages for each gene
    steps.append("Computing final gene embeddings...")
    for gene, embeddings in gene_embeddings.items():
        if embeddings:
            # Stack all embeddings for this gene and compute mean
            stacked_embeddings = torch.stack(embeddings, dim=0)
            avg_embedding = stacked_embeddings.mean(dim=0).numpy()
            gene_avg_embedding[gene] = avg_embedding
            steps.append(f"  Gene {gene}: averaged {len(embeddings)} isoform embeddings")
        else:
            steps.append(f"  Gene {gene}: no valid embeddings found")

    # Use default save path if none provided
    if save_path is None:
        gene_list_str = "_".join(ensembl_gene_ids[:3])  # Use first 3 genes for filename
        if len(ensembl_gene_ids) > 3:
            gene_list_str += f"_and_{len(ensembl_gene_ids) - 3}_more"
        # In the run's output directory, not the CWD the portal shares across an account's chats
        # (hunt 2026-09-30, uT4-genomics-29).
        save_path = os.path.join(_default_output_dir(), f"esm_embeddings_{model_name}_{gene_list_str}.pt")

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    torch_embeddings = {gene_id: torch.from_numpy(embedding) for gene_id, embedding in gene_avg_embedding.items()}

    torch.save(torch_embeddings, save_path)
    steps.append(f"Embeddings saved to: {os.path.abspath(save_path)}")
    steps.append(f"File size: {os.path.getsize(save_path) / (1024 * 1024):.2f} MB")

    return "\n".join(steps)


def annotate_celltype_scRNA(
    adata_filename,
    data_dir,
    data_info,
    data_lake_path,
    cluster="leiden",
    llm=None,
    composition=None,
    output_dir=None,
):
    """Annotate cell types based on gene markers and transferred labels using LLM.
    After clustering, annotate the clusters of ``adata.obs[cluster]`` using their differentially
    expressed genes and optionally incorporate transferred labels from reference datasets.

    Parameters
    ----------
    - adata_filename (str): Name of the AnnData file containing scRNA-seq data
    - data_dir (str): Directory containing the data files
    - data_info (str): Information about the scRNA-seq data (e.g., "homo sapiens, brain tissue, normal")
    - data_lake_path (str): Path to the data lake
    - cluster (str): obs column holding the clustering to annotate (default: "leiden")
    - llm (str, optional): Model name for cell type prediction. Default: the agent's configured model.
    - composition (pd.DataFrame, optional): Transferred cell type composition for each cluster
    - output_dir (str, optional): Where the annotated AnnData is written. Default: the run's output
      directory (``$SOG_WORK_DIR`` when set) -- never next to the input.
    Returns:
    - str: Steps performed and file paths where results were saved

    """

    def _cluster_info(cluster_id, marker_genes, composition_df=None):
        """Format cluster information for LLM prompt."""
        if composition_df is None:
            return f"The enriched genes in this cluster are: {', '.join(marker_genes)}."

        info = [
            f"The enriched genes in this cluster are: {', '.join(marker_genes)}.",
            f"For a starting point, the transferred reference cell type composition {cluster_id} is:",
        ]

        cluster_comp = []
        for celltype, proportion in composition_df.loc[cluster_id].items():
            if proportion > 0:
                cluster_comp.append(f"{celltype}:{proportion:.2f}")

        return "\n".join(info) + " " + "; ".join(cluster_comp) + "\n"

    from langchain_core.prompts import PromptTemplate

    # from langchain.chains import LLMChain

    steps = []
    steps.append(f"Loading AnnData from {data_dir}/{adata_filename}")
    adata = sc.read_h5ad(f"{data_dir}/{adata_filename}")

    steps.append(f"Identifying marker genes for clusters defined by {cluster} clustering.")
    # Markers come from the clustering being annotated, keyed by cluster name. groupby was hard-coded
    # to "leiden" (KeyError without that column, or another clustering's markers on these labels),
    # and markers were matched to clusters by position, which assumed labels '0'..'k-1'
    # (hunt 2026-09-30, uT4-genomics-11).
    adata.obs[cluster] = adata.obs[cluster].astype(str).astype("category")
    sc.tl.rank_genes_groups(adata, groupby=cluster, method="wilcoxon", use_raw=False)
    genes = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(20)
    scores = pd.DataFrame(adata.uns["rank_genes_groups"]["scores"]).head(20)

    markers = {}
    for name in genes.columns:
        gene_names = genes[name].tolist()
        gene_scores = scores[name].tolist()
        markers[str(name)] = list(np.array(gene_names)[np.array(gene_scores) > 0])

    czi_celltype = _czi_celltype_catalogue(data_lake_path + "/czi_census_datasets_v4.parquet")

    prompt_template = f"""
Please think carefully, and identify the cell type in {data_info} based on the gene markers.
Optionally refer to the transferred cell type information but do not trust it when the percentage is lower than 0.5.

{{cluster_info}}

The cell type names should come from cell ontology: {czi_celltype}.
Only provide the cell type name, confidence score (0-1), and detailed reason.
Output format: "name; score; reason".
No numbers before name or spaces before number.
"""
    # Some can be a mixture of multiple cell types.

    # The agent's configured model unless the caller names one. The default was the literal
    # claude-3-5-sonnet-20241022, which llm.py records as retired (it 404s at invoke) and which needs
    # Anthropic credentials the shipped deployment does not have (hunt 2026-09-30, uT4-genomics-12).
    from spatialomicsgym.config import default_config

    llm = get_llm(llm, config=default_config)
    prompt = PromptTemplate(input_variables=["cluster_info"], template=prompt_template)
    chain = prompt | llm

    steps.append("Annotating cell types of each cluster based on gene markers and transferred labels.")
    czi_celltype_set = _czi_celltype_names(data_lake_path + "/czi_census_datasets_v4.parquet")
    cluster_annotations = {}
    annotation_reasons = []

    print(f"Annotate each cluster of {cluster}")
    unassigned = []
    for _idx in markers:
        cluster_info = _cluster_info(_idx, markers[_idx], composition)

        # Bounded: `while True` re-asked forever (a billed call each time, the prompt growing a line
        # per round) when the model kept answering a name outside the ontology (uT4-genomics-11).
        for _attempt in range(_ANNOTATION_ATTEMPTS):
            response = chain.invoke({"cluster_info": cluster_info})

            # Handle different response types
            if hasattr(response, "content"):  # For AIMessage
                response = response.content
            elif isinstance(response, dict) and "text" in response:
                response = response["text"]
            elif isinstance(response, str):
                response = response
            else:
                response = str(response)

            try:
                predicted_celltype, confidence, reason = [x.strip() for x in response.split(";", 2)]
                if predicted_celltype in czi_celltype_set or predicted_celltype.lower() in czi_celltype_set:
                    cluster_annotations[_idx] = predicted_celltype
                    annotation_reasons.append((predicted_celltype, reason))
                    break
                else:
                    cluster_info += "\nAssigned cell type name must be in cell ontology!"
            except ValueError:
                cluster_info += "\nPlease follow the format: name; score; reason"
        else:
            unassigned.append(_idx)
            cluster_annotations[_idx] = "unassigned"
            annotation_reasons.append(
                (
                    "unassigned",
                    f"cluster {_idx}: no cell-ontology name after {_ANNOTATION_ATTEMPTS} attempts; last answer: {response}",
                )
            )
        print(f"Cluster {_idx}: {response}")
    if unassigned:
        steps.append(
            f"Clusters {', '.join(unassigned)} are 'unassigned': the model gave no cell-ontology name in "
            f"{_ANNOTATION_ATTEMPTS} attempts; its last answers are in 'cell_type_reason'."
        )

    # create reason dictionary
    reason_dict = {}
    for celltype, reason in annotation_reasons:
        if celltype not in reason_dict:
            reason_dict[celltype] = []
        reason_dict[celltype].append(reason)

    reason_dict = {k: "\n".join(v) for k, v in reason_dict.items()}

    adata.obs["cell_type"] = adata.obs[cluster].astype(str).map(cluster_annotations)
    adata.obs["cell_type_reason"] = adata.obs["cell_type"].map(reason_dict).astype(str)

    # Into output_dir under the input's stem; it was a fixed {data_dir}/annotated.h5ad, next to the
    # input -- often the shared dataset library (hunt 2026-09-30, uT4-genomics-14).
    if output_dir is None:
        output_dir = _default_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(adata_filename))[0]
    annotated_path = os.path.abspath(os.path.join(output_dir, f"{stem}_annotated.h5ad"))
    steps.append(f"Saving annotated adata to {annotated_path}, the annotations are in the 'cell_type' column.")
    _replace_into(annotated_path, lambda p: adata.write_h5ad(p, compression="gzip"))

    return "\n".join(steps)


def annotate_celltype_with_panhumanpy(
    adata_path,
    feature_names_col=None,
    refine=True,
    umap=True,
    output_dir=None,
):
    """Perform hierarchical cell type annotation using panhumanpy and Azimuth Neural Network.

    This function implements the panhumanpy workflow for cell type annotation using the
    Azimuth Neural Network, providing hierarchical cell type labels with confidence scores.

    Parameters
    ----------
    adata_path : str
        Path to the AnnData file containing scRNA-seq data
    feature_names_col : str, optional
        Column name in adata.var containing gene names (default: None, uses index)
    refine : bool, optional
        Whether to perform label refinement for consistent granularity (default: True)
    umap : bool, optional
        Whether to generate ANN embeddings and UMAP (default: True)
    output_dir : str, optional
        Directory to save results (default: ``panhumanpy/`` under the run's output directory,
        ``$SOG_WORK_DIR`` when set)

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    Notes
    -----
    Performance is not ensured for diseased and/or non-human cells.
    """
    import json
    import shutil
    import subprocess
    import tempfile

    # "./output" was the CWD the portal's REPL shares between an account's chats
    # (hunt 2026-09-30, uT4-genomics-29).
    if output_dir is None:
        output_dir = _default_output_dir("panhumanpy")
    output_dir = os.path.abspath(output_dir)

    def conda_env_exists(env_name):
        try:
            result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=True)
            return any(env_name in line.split() for line in result.stdout.splitlines())
        except Exception:
            return False

    def create_panhumanpy_env(env_name):
        # Create env and install panhumanpy
        subprocess.run(["conda", "create", "-y", "-n", env_name, "python=3.10"], check=True)
        # Install panhumanpy in the new env
        subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install", "git+https://github.com/satijalab/panhumanpy.git"],
            check=True,
        )

    PANHUMANPY_ENV = "panhumanpy_env"

    # 1. Check/create panhumanpy_env
    if not conda_env_exists(PANHUMANPY_ENV):
        create_panhumanpy_env(PANHUMANPY_ENV)

    # 2. Write a temp script to run in the panhumanpy_env
    temp_dir = tempfile.mkdtemp()
    script_path = os.path.join(temp_dir, "run_panhumanpy.py")
    result_path = os.path.join(temp_dir, "result.json")
    with open(script_path, "w") as f:
        f.write(
            f"""
import os
import sys
import json
import numpy as np
import scanpy as sc
import pandas as pd
try:
    import panhumanpy as ph
except ImportError as e:
    with open(r'{result_path}', 'w') as out:
        out.write(json.dumps({{"error": str(e)}}))
    sys.exit(1)

adata_path = r'''{adata_path}'''
feature_names_col = {repr(feature_names_col)}
refine = {refine}
umap = {umap}
output_dir = r'''{output_dir}'''
log = []
try:
    os.makedirs(output_dir, exist_ok=True)
    log.append("# Performing cell type annotation with Panhuman Azimuth")
    log.append(f"Loading object from: {{adata_path}}")
    adata = sc.read_h5ad(adata_path)
    log.append(f"✓ Successfully loaded object with {{adata.n_obs}} cells and {{adata.n_vars}} genes")
    if feature_names_col is None:
        log.append("Using gene names from adata.var.index")
    else:
        log.append(f"Using gene names from column: {{feature_names_col}}")
        if feature_names_col not in adata.var.columns:
            log.append(f"⚠ Warning: Column '{{feature_names_col}}' not found in adata.var")
            log.append(f"Available columns: {{list(adata.var.columns)}}")
            log.append("Falling back to index")
            feature_names_col = None
    if feature_names_col is None:
        azimuth = ph.AzimuthNN(adata)
    else:
        azimuth = ph.AzimuthNN(adata, feature_names_col=feature_names_col)
    cell_metadata = azimuth.cells_meta
    log.append("✓ Successfully annotated all cells")
    # The UMAP is azimuth_umap()'s return value; `umap` is the boolean flag, which was what got saved
    # as ann_umap.npy. An embedding or UMAP failure is logged and the finished annotations are still
    # saved -- it exited before '## Saving results' (hunt 2026-09-30, uT4-genomics-17).
    embeddings = None
    umap_coords = None
    if umap:
        log.append("## Generating ANN embeddings")
        try:
            embeddings = azimuth.azimuth_embed()
        except Exception as e:
            log.append(f"✗ Error generating embeddings: {{str(e)}}")
        if embeddings is not None:
            log.append("## Calculating UMAP")
            try:
                umap_coords = azimuth.azimuth_umap()
                log.append("✓ Generated UMAP of ANN embeddings")
            except Exception as e:
                log.append(f"✗ Error generating UMAP: {{str(e)}}")
    else:
        log.append("## Skipping embeddings and UMAP generation")
    if refine:
        log.append("## Performing label refinement")
        try:
            azimuth.azimuth_refine()
            cell_metadata = azimuth.cells_meta
            refined_columns = [col for col in cell_metadata.columns if col.startswith("azimuth_")]
            log.append(f"✓ Applied label refinement, results are in columns: {{refined_columns}}")
        except Exception as e:
            log.append(f"✗ Error during label refinement: {{str(e)}}")
    log.append("## Saving results")
    try:
        metadata_file = f"{output_dir}/annotated_cell_metadata.csv"
        cell_metadata.to_csv(metadata_file)
        log.append(f"✓ Saved cell metadata to: {{metadata_file}}")
        if embeddings is not None:
            embeddings_file = f"{output_dir}/ann_embeddings.npy"
            np.save(embeddings_file, embeddings)
            log.append(f"✓ Saved embeddings to: {{embeddings_file}}")
        if umap_coords is not None:
            umap_file = f"{output_dir}/ann_umap.npy"
            np.save(umap_file, np.asarray(umap_coords))
            log.append(f"✓ Saved UMAP to: {{umap_file}}")
        if not umap:
            log.append("Skipped saving embeddings and UMAP (umap=False)")
        elif umap_coords is None:
            log.append("Error: no UMAP was saved (see the errors above); the annotations are saved")
        annotated_save_path = f"{output_dir}/annotated_obj.h5ad"
        azimuth.pack_adata(save_path=annotated_save_path)
        log.append(f"✓ Saved annotated object to: {{annotated_save_path}}")
    except Exception as e:
        log.append(f"✗ Error saving results: {{str(e)}}")
        with open(r'{result_path}', 'w') as out:
            out.write(json.dumps({{"log": log}}))
        sys.exit(0)
    log.append(f"- All results saved to: {output_dir}")
    with open(r'{result_path}', 'w') as out:
        out.write(json.dumps({{"log": log}}))
except Exception as e:
    with open(r'{result_path}', 'w') as out:
        out.write(json.dumps({{"error": str(e)}}))
    sys.exit(1)
"""
        )

    # 3. Run the script in the panhumanpy_env
    try:
        run_cmd = ["conda", "run", "-n", PANHUMANPY_ENV, "python", script_path]
        subprocess.run(run_cmd, check=True)
    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir)
        return f"Error running panhumanpy in conda env: {e}"

    # 4. Read the result
    try:
        with open(result_path) as f:
            result = json.load(f)
        if "log" in result:
            log = result["log"]
        elif "error" in result:
            log = [f"Error: {result['error']}"]
        else:
            log = ["Unknown error running panhumanpy script."]
    except Exception as e:
        log = [f"Error reading result: {e}"]

    # 5. Clean up temp files
    shutil.rmtree(temp_dir)

    return "\n".join(log)


# === Integration ===


def _derived_h5ad_path(output_dir, adata_filename, suffix, subdir=""):
    """``<output_dir>/<input stem>_<suffix>.h5ad``, absolute; ``output_dir`` defaults to the run's.

    The integration and mapping tools wrote fixed names (scvi_emb_data.h5ad, harmony_emb_data.h5ad,
    adata_with_mapped_celltypes.h5ad) into the input's ``data_dir`` -- often the shared dataset
    library, against the prompt's rule never to write next to the inputs -- so a second call
    overwrote the first and a read-only library failed after the expensive step
    (hunt 2026-09-30, uT4-genomics-14).
    """
    if output_dir is None:
        output_dir = _default_output_dir(subdir)
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(adata_filename))[0]
    return os.path.abspath(os.path.join(output_dir, f"{stem}_{suffix}.h5ad"))


def create_scvi_embeddings_scRNA(adata_filename, batch_key, label_key, data_dir, output_dir=None):
    # Import scvi-tools correctly - the package name is still 'scvi' when installed
    try:
        import scvi
    except ImportError:
        return "Please install scvi-tools: pip install scvi-tools"

    steps = []
    steps.append(f"Loading AnnData from {data_dir}/{adata_filename}")
    adata = sc.read_h5ad(f"{data_dir}/{adata_filename}")

    steps.append(f"Setting up AnnData for scVI with batch key: {batch_key}")
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)

    steps.append("Creating and training scVI model")
    model = scvi.model.SCVI(adata)
    model.train()

    steps.append("Generating latent representation using scVI")
    adata.obsm["X_scVI"] = model.get_latent_representation()

    steps.append(f"Creating scANVI model with label key: {label_key}")
    lvae = scvi.model.SCANVI.from_scvi_model(
        model,
        adata=adata,
        labels_key=label_key,
        unlabeled_category="Unknown",
    )
    steps.append("Training scANVI model")
    lvae.train()

    steps.append("Generating latent representation using scANVI")
    adata.obsm["X_scANVI"] = lvae.get_latent_representation(adata)

    output_filename = _derived_h5ad_path(output_dir, adata_filename, "scvi_emb")
    _replace_into(output_filename, adata.write_h5ad)
    steps.append(f"Saving AnnData with scVI and scANVI embeddings to {output_filename}")

    return "\n".join(steps)


def create_harmony_embeddings_scRNA(adata_filename, batch_key, data_dir, output_dir=None):
    # https://pypi.org/project/harmony-pytorch/
    from harmony import harmonize

    steps = []
    steps.append(f"Loading AnnData from {data_dir}/{adata_filename}")
    adata = sc.read_h5ad(f"{data_dir}/{adata_filename}")

    steps.append(f"Running Harmony integration with batch key: {batch_key}")
    adata.obsm["X_harmony"] = harmonize(adata.obsm["X_pca"], adata.obs, batch_key=batch_key)

    output_filename = _derived_h5ad_path(output_dir, adata_filename, "harmony_emb")
    steps.append(f"Saving the Harmony embeddings to {output_filename}.")
    _replace_into(output_filename, adata.write_h5ad)

    return "\n".join(steps)


def get_uce_embeddings_scRNA(
    adata_filename,
    data_dir,
    DATA_ROOT=os.environ.get("SOG_SINGLECELL_DATA_PATH", "./data/singlecell/"),
    custom_args=None,
    output_dir=None,
):
    """Compute UCE cell embeddings for an AnnData file (optional; needs a local UCE checkout).

    UCE is not bundled -- no shipped conda environment includes it, so this tool only works
    after the UCE repository has been cloned to ``{DATA_ROOT}/UCE`` (DATA_ROOT defaults to
    ``$SOG_SINGLECELL_DATA_PATH`` or ``./data/singlecell/``). The resulting embeddings can be
    mapped to the IMA reference via ``map_to_ima_interpret_scRNA``.
    The custom_args is a list of strings that will be passed as command line arguments to the UCE script,
    like ["--adata_path", adata_file, "--dir", output_dir]. The default value is None.
    output_dir is where UCE writes its working files and ``<name>_uce_adata.h5ad`` (embeddings in
    ``obsm['X_uce']``); default ``uce/`` under the run's output directory (``$SOG_WORK_DIR`` when set).
    """
    import sys

    steps = []

    # eval_single_anndata lives in the UCE repo root, so the checkout must be importable
    # before the import is attempted.
    uce_dir = f"{DATA_ROOT}/UCE"
    if uce_dir not in sys.path:
        sys.path.append(uce_dir)
    steps.append(f"Added {uce_dir} to sys.path")

    try:
        from eval_single_anndata import main, parse_args_uce

        steps.append("Successfully imported UCE main function")
    except Exception:
        steps.append(
            f"UCE is not available: could not import eval_single_anndata from {uce_dir}. "
            "No shipped conda environment includes UCE -- clone it first "
            f"(git clone https://github.com/snap-stanford/UCE.git {uce_dir}) "
            "or set SOG_SINGLECELL_DATA_PATH to a directory whose UCE/ subfolder holds the checkout."
        )
        return "\n".join(steps)

    from accelerate import Accelerator

    # UCE writes {--dir}{name}_uce_adata.h5ad, name being the input's file name less ".h5ad". The
    # path checked and reported here was {data_dir}/{name.split('.')[0]}_uce_adata.h5ad -- not where
    # UCE writes -- so the skip-if-exists check never fired and the success line named a file that
    # was not there. The embeddings are in obsm['X_uce'], which map_to_ima_interpret_scRNA reads; both
    # messages said obs (hunt 2026-09-30, uT4-genomics-19).
    # UCE's --dir was {data_dir}/uce/, next to the input -- often the shared dataset library
    # (uT4-genomics-14).
    if output_dir is None:
        output_dir = _default_output_dir("uce")
    uce_out_dir = os.path.join(os.path.abspath(output_dir), "")
    os.makedirs(uce_out_dir, exist_ok=True)
    _base_name = os.path.basename(adata_filename).replace(".h5ad", "")
    adata_file_proc = os.path.abspath(f"{uce_out_dir}{_base_name}_uce_adata.h5ad")
    if os.path.exists(adata_file_proc):
        steps.append(
            f"{adata_file_proc} already exists, skipping. The UCE embeddings are stored as adata.obsm['X_uce']"
        )
        return "\n".join(steps)

    # Prepare and parse arguments. A copy, so the caller's list is not extended on every call.
    custom_args = list(custom_args or [])
    custom_args.extend(["--adata_path", f"{data_dir}/{adata_filename}", "--dir", uce_out_dir])
    parsed_args = parse_args_uce(custom_args)
    steps.append(f"Parsed arguments: {vars(parsed_args)}")

    # Initialize Accelerator
    accelerator = Accelerator(project_dir=parsed_args.dir)
    steps.append("Initialized Accelerator")

    # Run UCE main function
    main(parsed_args, accelerator)
    steps.append("UCE main function completed successfully.")
    if not os.path.exists(adata_file_proc):
        steps.append(f"Error: UCE finished but {adata_file_proc} is not there; see {uce_out_dir} for what it wrote.")
        return "\n".join(steps)
    steps.append(f"{adata_file_proc} is saved, the UCE embeddings are stored as adata.obsm['X_uce']")

    return "\n".join(steps)


def map_to_ima_interpret_scRNA(adata_filename, data_dir, custom_args=None, reference_h5ad=None, output_dir=None):
    """Map cell embeddings from the input dataset to the Integrated Megascale Atlas reference dataset using UCE embeddings.

    ``custom_args`` may carry ``n_neighbors`` (default 3) and ``metric`` (default "euclidean").
    ``reference_h5ad`` is the IMA reference AnnData (UCE embeddings in ``X``, labels in
    ``obs['coarse_cell_type_yanay']``); default ``{data_dir}/uce_10000_per_dataset_33l_8ep_coarse_ct.h5ad``.
    Nothing ships it. The mapped AnnData goes to ``output_dir`` (default: the run's output
    directory, ``$SOG_WORK_DIR`` when set).
    """
    from sklearn.neighbors import NearestNeighbors

    # custom_args defaults to None and was read with .get(), an AttributeError after both h5ads had
    # loaded; the reference path was hard-coded and undocumented (hunt 2026-09-30, uT4-genomics-18).
    custom_args = custom_args or {}
    if reference_h5ad is None:
        reference_h5ad = os.path.join(data_dir, "uce_10000_per_dataset_33l_8ep_coarse_ct.h5ad")
    if not os.path.isfile(reference_h5ad):
        return (
            f"Error: the IMA reference {reference_h5ad} does not exist. No shipped dataset provides it; "
            "pass reference_h5ad= the UCE-embedded Integrated Megascale Atlas h5ad."
        )

    steps = []
    steps.append(f"Loading AnnData from {data_dir}/{adata_filename}")
    adata = sc.read_h5ad(f"{data_dir}/{adata_filename}")

    if "X_uce" not in adata.obsm:
        raise ValueError("Error: adata.obsm['X_uce'] not found. Please run get_uce_embeddings() first.")

    steps.append("adata.obsm['X_uce'] found. Proceeding with cell type mapping.")

    IMA_adata = sc.read_h5ad(reference_h5ad)
    steps.append(f"Loaded Integrated Megascale Atlas (IMA) reference dataset from {reference_h5ad}")

    if adata.obsm["X_uce"].shape[1] != IMA_adata.X.shape[1]:
        raise ValueError("UCE embedding dimensions do not match between datasets.")

    # Create a NearestNeighbors object
    n_neighbors = custom_args.get("n_neighbors", 3)
    metric = custom_args.get("metric", "euclidean")
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(IMA_adata.X)

    # Find the nearest neighbors for each cell in adata
    distances, indices = nn.kneighbors(adata.obsm["X_uce"])

    # Extract cell types of the nearest neighbors
    mapped_cell_types = IMA_adata.obs["coarse_cell_type_yanay"].iloc[indices.flatten()].values

    # from umap import UMAP
    if n_neighbors > 1:
        # Implement majority voting

        def custom_mode(x):
            unique, counts = np.unique(x, return_counts=True)
            return unique[np.argmax(counts)]

        mapped_cell_types = np.apply_along_axis(custom_mode, 1, mapped_cell_types.reshape(-1, n_neighbors))

    # Add mapped cell types and confidence scores to adata
    adata.obs["mapped_cell_type"] = mapped_cell_types
    adata.obs["mapping_confidence"] = 1 / (1 + distances.mean(axis=1))

    steps.append("Mapped cell types based on nearest neighbors in UCE space")
    steps.append("Mapped cell types and confidence scores added to adata.obs")

    # Generate summary statistics
    mapping_summary = adata.obs["mapped_cell_type"].value_counts().to_dict()
    steps.append(f"Mapping summary: {mapping_summary}")

    # Save the updated adata object, under output_dir and the input's stem (uT4-genomics-14)
    output_filename = _derived_h5ad_path(output_dir, adata_filename, "mapped_celltypes")
    _replace_into(output_filename, lambda p: adata.write_h5ad(p, compression="gzip"))
    steps.append(f"Updated adata saved to {output_filename}")

    return "\n".join(steps)


def get_rna_seq_archs4(gene_name: str, K: int = 10) -> str:
    """Given a gene name, this function returns the steps it performs and the max K transcripts-per-million (TPM)
    per tissue from the RNA-seq expression.

    Parameters
    ----------
    - gene_name (str): The gene name for which RNA-seq data is being fetched.
    - K (int): The number of tissues to return. Default is 10.

    Returns
    -------
    - str: The steps performed and the result.

    """
    import gget

    steps_log = f"Starting RNA-seq data fetch for gene: {gene_name} with K: {K}\n"

    try:
        # Fetch RNA-seq data using gget
        steps_log += "Fetching RNA-seq data using gget.archs4...\n"
        data = gget.archs4(gene_name, which="tissue")

        if data.empty:
            steps_log += f"No RNA-seq data found for the gene {gene_name}.\n"
            return steps_log

        # Create a readable output string
        steps_log += f"RNA-seq expression data for {gene_name} fetched successfully. Formatting the top {K} tissues:\n"
        readable_output = ""
        for index, row in data.iterrows():
            if index < K:
                tissue = row["id"]
                median_tpm = row["median"]
                readable_output += f"\nTissue: {tissue}\n  - Median TPM: {median_tpm}\n"
            else:
                break

        steps_log += readable_output
        return steps_log

    except Exception as e:
        # "Error: " at line start is what the ReAct loop reads as a failed step; "An error occurred:"
        # read as a successful one (hunt 2026-09-30, uT4-genomics-23).
        return f"Error: ARCHS4 fetch for {gene_name} failed: {e}"


def get_gene_set_enrichment_analysis_supported_database_list() -> list:
    import gseapy

    return gseapy.get_library_name()


def gene_set_enrichment_analysis(
    genes: list,
    top_k: int = 10,
    database: str = "ontology",
    background_list: list = None,
    plot: bool = False,
) -> str:
    """Perform enrichment analysis for a list of genes, with optional background gene set and plotting functionality.

    Parameters
    ----------
    - genes (list): List of gene symbols to analyze.
    - top_k (int): Number of top pathways to return. Default is 10.
    - database (str): User-friendly name of the database to use for enrichment analysis.
        Popular options include:
        - 'pathway'      (KEGG_2021_Human)
        - 'transcription'   (ChEA_2016)
        - 'ontology'     (GO_Biological_Process_2021)
        - 'diseases_drugs'  (GWAS_Catalog_2019)
        - 'celltypes'     (PanglaoDB_Augmented_2021)
        - 'kinase_interactions' (KEA_2015)
        You can use get_gene_set_enrichment_analysis_supported_database_list tool to get the list of supported databases.

    - background_list (list, optional): List of background genes to use for enrichment analysis.
    - plot (bool, optional): If True, generates a bar plot of the top K enrichment results.

    Returns
    -------
    - str: The steps performed and the top K enrichment results.

    """
    import gget

    steps_log = (
        f"Starting enrichment analysis for genes: {', '.join(genes)} using {database} database and top_k: {top_k}\n"
    )

    if background_list:
        steps_log += f"Using background list with {len(background_list)} genes.\n"

    try:
        # Perform enrichment analysis with or without background list
        steps_log += f"Performing enrichment analysis using gget.enrichr with the {database} database...\n"
        df = gget.enrichr(genes, database=database, background_list=background_list, plot=plot)

        # Limit to top K results
        steps_log += f"Filtering the top {top_k} enrichment results...\n"
        df = df.head(top_k)

        # Format the result
        output_str = ""
        for _idx, row in df.iterrows():
            output_str += (
                f"Rank: {row['rank']}\n"
                f"Path Name: {row['path_name']}\n"
                f"P-value: {row['p_val']:.2e}\n"
                f"Z-score: {row['z_score']:.6f}\n"
                f"Combined Score: {row['combined_score']:.6f}\n"
                f"Overlapping Genes: {', '.join(row['overlapping_genes'])}\n"
                f"Adjusted P-value: {row['adj_p_val']:.2e}\n"
                f"Database: {row['database']}\n"
                "----------------------------------------\n"
            )

        steps_log += output_str

        return steps_log

    except Exception as e:
        # "Error: " at line start, which the ReAct loop reads as a failed step (uT4-genomics-23).
        return f"Error: enrichment analysis failed: {e}"


def analyze_chromatin_interactions(hic_file_path, regulatory_elements_bed, output_dir=None):
    """Analyze chromatin interactions from Hi-C data to identify enhancer-promoter interactions and TADs.

    Parameters
    ----------
    hic_file_path : str
        Path to the Hi-C data file in cooler format (.cool, or an .mcool URI such as
        ``file.mcool::resolutions/10000``). Juicer .hic is not readable here; convert it first
        (e.g. with hic2cool).
    regulatory_elements_bed : str
        Path to BED file containing genomic coordinates of regulatory elements
        (enhancers, promoters, CTCF sites, etc.)
    output_dir : str
        Directory to save output files (default: ``chromatin_interactions/`` under the run's output
        directory, ``$SOG_WORK_DIR`` when set)

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import cooler
    import numpy as np
    import pandas as pd
    from scipy import stats

    # "./output" was the CWD the portal's REPL shares between an account's chats
    # (hunt 2026-09-30, uT4-genomics-29).
    if output_dir is None:
        output_dir = _default_output_dir("chromatin_interactions")
    output_dir = os.path.abspath(output_dir)
    # The description promised ".cool or .hic", but cooler cannot open a Juicer .hic file
    # (hunt 2026-09-30, uT4-genomics-31).
    if str(hic_file_path).lower().split("::")[0].endswith(".hic"):
        return (
            f"Error: {hic_file_path} is a Juicer .hic file, which this tool cannot read. Convert it to "
            ".cool/.mcool first (e.g. hic2cool convert in.hic out.mcool) and pass that."
        )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize research log
    log = []
    log.append("# Chromatin Interaction Analysis\n")

    # Step 1: Load Hi-C data
    log.append("## Step 1: Loading Hi-C data")
    try:
        c = cooler.Cooler(hic_file_path)
        log.append(f"Successfully loaded Hi-C data from {hic_file_path}")
        log.append(f"Resolution: {c.binsize} bp")
        log.append(f"Chromosomes: {c.chromnames}")

        # Check if balancing weights are available
        has_weights = False
        try:
            # Try different ways to check for weights
            if (
                hasattr(c.bins(), "columns")
                and "weight" in c.bins().columns
                or hasattr(c.bins(), "keys")
                and "weight" in c.bins()
                or hasattr(c, "bins")
                and callable(c.bins)
                and "bins/weight" in c.bins()
            ):
                has_weights = True
        except Exception as e:
            log.append(f"Warning: Could not check for balancing weights: {str(e)}")

        if not has_weights:
            log.append("Warning: No balancing weights found in the cooler file. Using unbalanced data.")

        # Function to get matrix with proper balancing option
        def get_matrix(chrom, balance_if_possible=True):
            try:
                # Try to get balanced matrix if requested and available
                if balance_if_possible and has_weights:
                    return c.matrix(balance=True).fetch(chrom)
                else:
                    return c.matrix(balance=False).fetch(chrom)
            except Exception as e:
                log.append(f"Warning: Error getting matrix with balance={balance_if_possible}: {str(e)}")
                # If balanced fails, try unbalanced as fallback
                if balance_if_possible:
                    log.append("Falling back to unbalanced matrix")
                    return c.matrix(balance=False).fetch(chrom)
                else:
                    # If even unbalanced fails, raise the exception
                    raise e

    except Exception as e:
        # "Error: " at line start, which the ReAct loop reads as a failed step (uT4-genomics-23).
        log.append(f"Error: could not load Hi-C data: {str(e)}")
        return "\n".join(log)

    # Step 2: Load regulatory elements
    log.append("\n## Step 2: Loading regulatory elements")
    try:
        reg_elements = pd.read_csv(
            regulatory_elements_bed,
            sep="\t",
            names=["chrom", "start", "end", "name", "score", "strand"],
        )
        log.append(f"Loaded {len(reg_elements)} regulatory elements")

        # Separate enhancers and promoters based on name field (assuming naming convention)
        enhancers = reg_elements[reg_elements["name"].str.contains("enhancer", case=False)]
        promoters = reg_elements[reg_elements["name"].str.contains("promoter", case=False)]
        log.append(f"Identified {len(enhancers)} enhancers and {len(promoters)} promoters")
    except Exception as e:
        log.append(f"Error: could not load regulatory elements: {str(e)}")
        return "\n".join(log)

    # Step 3: Normalize Hi-C matrix using iterative correction
    log.append("\n## Step 3: Normalizing Hi-C contact matrix")
    try:
        # Cooler already implements iterative correction (IC)
        log.append("Applying iterative correction to normalize for technical biases")
    except Exception as e:
        log.append(f"Error during normalization: {str(e)}")

    # Step 4: Identify significant interactions between regulatory elements
    log.append("\n## Step 4: Identifying significant interactions")

    # Initialize results dataframe for enhancer-promoter interactions. Each step records whether its
    # file was written, and the summary names only files that exist (uT4-genomics-8).
    interactions = []
    interactions_written = tads_written = matrix_written = False
    tad_results = []

    try:
        # Process chromosome by chromosome
        for chrom in c.chromnames:
            log.append(f"Processing chromosome {chrom}...")

            # Get matrix for this chromosome using the helper function
            try:
                matrix = get_matrix(chrom, balance_if_possible=True)
            except Exception as e:
                log.append(f"Error getting matrix for chromosome {chrom}: {str(e)}")
                log.append(f"Skipping chromosome {chrom}")
                continue

            # Filter enhancers and promoters for this chromosome
            chrom_enhancers = enhancers[enhancers["chrom"] == chrom]
            chrom_promoters = promoters[promoters["chrom"] == chrom]

            if len(chrom_enhancers) == 0 or len(chrom_promoters) == 0:
                log.append(f"No enhancers or promoters found on chromosome {chrom}, skipping")
                continue

            # For each enhancer, find interactions with promoters
            for _, enh in chrom_enhancers.iterrows():
                enh_bin = int(enh["start"] // c.binsize)

                for _, prom in chrom_promoters.iterrows():
                    prom_bin = int(prom["start"] // c.binsize)

                    # Skip if bins are out of range
                    if enh_bin >= matrix.shape[0] or prom_bin >= matrix.shape[0]:
                        continue

                    # Get interaction strength
                    interaction_strength = matrix[enh_bin, prom_bin]

                    # Calculate expected interaction based on genomic distance
                    distance = abs(enh_bin - prom_bin)
                    if distance == 0:
                        continue

                    # Get average interaction at this distance
                    diag_values = np.diagonal(matrix, offset=distance)
                    expected = np.nanmean(diag_values) if len(diag_values) > 0 else 0

                    # Calculate significance (fold enrichment)
                    fold_enrichment = interaction_strength / expected if expected > 0 else 0

                    # Store significant interactions (fold enrichment > 2)
                    if fold_enrichment > 2:
                        interactions.append(
                            {
                                "chrom": chrom,
                                "enhancer_start": enh["start"],
                                "enhancer_end": enh["end"],
                                "enhancer_name": enh["name"],
                                "promoter_start": prom["start"],
                                "promoter_end": prom["end"],
                                "promoter_name": prom["name"],
                                "interaction_strength": interaction_strength,
                                "fold_enrichment": fold_enrichment,
                            }
                        )

        # Define interactions file path (moved before the empty check)
        interactions_file = os.path.join(output_dir, "enhancer_promoter_interactions.tsv")

        # Save interactions to file if any were found
        if interactions:
            interactions_df = pd.DataFrame(interactions)
            interactions_df.to_csv(interactions_file, sep="\t", index=False)
            log.append(f"Identified {len(interactions)} significant enhancer-promoter interactions")
            log.append(f"Saved interactions to {interactions_file}")
        else:
            # Create an empty file with headers if no interactions were found
            pd.DataFrame(
                columns=[
                    "chrom",
                    "enhancer_start",
                    "enhancer_end",
                    "enhancer_name",
                    "promoter_start",
                    "promoter_end",
                    "promoter_name",
                    "interaction_strength",
                    "fold_enrichment",
                ]
            ).to_csv(interactions_file, sep="\t", index=False)
            log.append("No significant enhancer-promoter interactions were identified")
            log.append(f"Created empty interactions file at {interactions_file}")
        interactions_written = True
    except Exception as e:
        log.append(f"Error: identifying enhancer-promoter interactions failed: {str(e)}")
        # Ensure the interactions_file variable is defined even in case of error
        interactions_file = os.path.join(output_dir, "enhancer_promoter_interactions.tsv")

    # Step 5: Call TADs using simple insulation score method
    log.append("\n## Step 5: Identifying Topologically Associated Domains (TADs)")

    # Define tads_file early to ensure it's always available
    tads_file = os.path.join(output_dir, "topological_domains.bed")

    try:
        # We'll use a simple approach to identify TADs based on insulation scores
        # For a real analysis, specialized tools like tadbit or HiCExplorer would be better

        window_size = 5  # Window size for insulation score calculation

        for chrom in c.chromnames:
            if chrom not in c.chromsizes:
                continue

            # Get matrix with balancing if possible
            try:
                matrix = get_matrix(chrom, balance_if_possible=True)
            except Exception as e:
                log.append(f"Error getting matrix for chromosome {chrom}: {str(e)}")
                log.append(f"Skipping chromosome {chrom}")
                continue

            # A chromosome with no more bins than the two insulation windows (chrM at 10 kb is 2 bins)
            # has no insulation score. np.zeros of a negative size raised, and that one chromosome's
            # exception skipped the save for every chromosome -- while the summary still reported the
            # TADs and named a file that was never written. Short chromosomes are skipped, and any other
            # per-chromosome failure costs only that chromosome (hunt 2026-09-30, uT4-genomics-8).
            if matrix.shape[0] <= 2 * window_size:
                log.append(
                    f"Skipping chromosome {chrom} for TADs: {matrix.shape[0]} bins, fewer than the "
                    f"{2 * window_size + 1} the insulation window needs"
                )
                continue
            try:
                # Calculate insulation score
                insulation_score = np.zeros(matrix.shape[0] - 2 * window_size)
                for i in range(window_size, matrix.shape[0] - window_size):
                    # Calculate sum of interactions crossing bin i
                    cross_interactions = np.sum(matrix[i - window_size : i, i : i + window_size])
                    insulation_score[i - window_size] = cross_interactions

                # Normalize insulation score
                if len(insulation_score) > 0:
                    insulation_score = stats.zscore(insulation_score, nan_policy="omit")

                    # Find local minima as TAD boundaries
                    boundaries = []
                    for i in range(1, len(insulation_score) - 1):
                        if (
                            insulation_score[i] < insulation_score[i - 1]
                            and insulation_score[i] < insulation_score[i + 1]
                            and insulation_score[i] < -1
                        ):  # Threshold for boundary strength
                            boundaries.append(i + window_size)

                    # Define TADs as regions between boundaries
                    if len(boundaries) > 1:
                        for i in range(len(boundaries) - 1):
                            start_bin = boundaries[i]
                            end_bin = boundaries[i + 1]

                            # Convert bin positions to genomic coordinates
                            start_pos = start_bin * c.binsize
                            end_pos = end_bin * c.binsize

                            tad_results.append(
                                {
                                    "chrom": chrom,
                                    "start": start_pos,
                                    "end": end_pos,
                                    "size_kb": (end_pos - start_pos) / 1000,
                                }
                            )
            except Exception as e:
                log.append(f"Warning: TAD calling failed on chromosome {chrom}: {str(e)}; its TADs are left out")
                continue

        # Save TADs to file if any were found
        tads_df = pd.DataFrame(tad_results)
        if not tads_df.empty:
            tads_df.to_csv(tads_file, sep="\t", index=False)
            log.append(f"Identified {len(tad_results)} topologically associated domains")
            log.append(f"Average TAD size: {tads_df['size_kb'].mean():.2f} kb")
            log.append(f"Saved TADs to {tads_file}")
        else:
            # Create an empty file with headers if no TADs were found
            pd.DataFrame(columns=["chrom", "start", "end", "size_kb"]).to_csv(tads_file, sep="\t", index=False)
            log.append("No topologically associated domains were identified")
            log.append(f"Created empty TADs file at {tads_file}")
        tads_written = True
    except Exception as e:
        log.append(f"Error: TAD calling failed: {str(e)}")

    # Step 6: Generate chromatin interaction map for visualization
    log.append("\n## Step 6: Generating chromatin interaction maps")

    # Define matrix_file early to ensure it's always available
    matrix_file = os.path.join(output_dir, "contact_matrix.npy")

    try:
        # Export normalized matrix for first chromosome as example
        if len(c.chromnames) > 0:
            chrom = c.chromnames[0]
            matrix_file = os.path.join(output_dir, f"{chrom}_contact_matrix.npy")

            # Get matrix with balancing if possible
            try:
                matrix = get_matrix(chrom, balance_if_possible=True)
                np.save(matrix_file, matrix)
                matrix_written = True
                log.append(f"Saved contact matrix for {chrom} to {matrix_file}")
            except Exception as e:
                # A 10x10 zeros array was saved here and listed as the contact matrix
                # (uT4-genomics-8).
                log.append(f"Error: could not export the contact matrix for chromosome {chrom}: {str(e)}")
        else:
            log.append("No chromosomes available for matrix export")
    except Exception as e:
        log.append(f"Error: generating interaction maps failed: {str(e)}")

    # Step 7: Summary
    log.append("\n## Summary")
    log.append(f"- Processed Hi-C data at {c.binsize} bp resolution")
    log.append(f"- Analyzed {len(reg_elements)} regulatory elements")

    log.append(
        f"- Identified {len(interactions)} significant enhancer-promoter interactions"
        if interactions_written
        else "- Enhancer-promoter interactions: not identified (see the error above)"
    )
    log.append(
        f"- Called {len(tad_results)} topologically associated domains"
        if tads_written
        else "- TADs: not called (see the error above)"
    )
    log.append("\nOutput files:")
    if interactions_written:
        log.append(f"- Enhancer-promoter interactions: {interactions_file}")
    if tads_written:
        log.append(f"- Topological domains: {tads_file}")
    if matrix_written:
        log.append(f"- Contact matrix: {matrix_file}")
    if not (interactions_written or tads_written or matrix_written):
        log.append("- none")

    return "\n".join(log)


def analyze_comparative_genomics_and_haplotypes(
    sample_fasta_files, reference_genome_path, output_dir=None, max_mismatch_fraction=0.1
):
    """Compare genome samples with a reference and report their shared and sample-specific variants.

    Each sample sequence is compared base by base with the reference sequence of the same FASTA ID.
    No alignment is made, so the comparison is valid only for samples already collinear with the
    reference (same coordinates, no indels): after an indel every base reads as a mismatch. A
    sequence whose length differs from the reference's, or that differs from it at more than
    ``max_mismatch_fraction`` of the compared bases, is therefore refused with an Error rather than
    compared. A variant is a single-base difference; positions where either base is N are skipped.
    Haplotype structure is not inferred -- there is no phasing and no divergence-time estimate --
    beyond listing samples whose variant sets are identical.

    Parameters
    ----------
    sample_fasta_files : list of str
        Paths to FASTA files containing whole-genome sequences to be analyzed
    reference_genome_path : str
        Path to the reference genome FASTA file
    output_dir : str, optional
        Directory to store output files (default: ``comparative_genomics/`` under the run's output
        directory, ``$SOG_WORK_DIR`` when set)
    max_mismatch_fraction : float, optional
        Largest fraction of compared bases at which a sample sequence may differ from the reference
        and still be taken as collinear with it (default: 0.1). Above it the call returns an Error:
        that is the signature of an indel or of a different assembly, not of single-base variants.

    Returns
    -------
    str
        A research log summarizing the analysis steps and results

    """
    import time

    from Bio import SeqIO

    # None or a non-numeric string raised a TypeError at the first comparison, partway through the
    # samples, instead of an Error line; NaN compared False and admitted every sequence
    # (hunt 2026-09-30, uT4-genomics-2, reviewer follow-up).
    try:
        max_mismatch_fraction = float(max_mismatch_fraction)
    except (TypeError, ValueError):
        return f"Error: max_mismatch_fraction must be a number from 0 to 1 (got {max_mismatch_fraction!r})."
    if not 0.0 <= max_mismatch_fraction <= 1.0:
        return f"Error: max_mismatch_fraction must be a number from 0 to 1 (got {max_mismatch_fraction!r})."

    # This tool fabricated most of its result: with BWA on PATH every sample's "variants" were the
    # one placeholder string "Simulated variant detection from SAM file"; haplotype groups were
    # assigned by sample order (index mod 3); and each group got a random number of distinguishing
    # variants and a random divergence time -- reported as findings with no word that they were
    # simulated. What it reports now is only what the comparison measures (hunt 2026-09-30,
    # uT4-genomics-2). Outputs go to output_dir, not "./output" in the shared CWD (uT4-genomics-29).
    if output_dir is None:
        output_dir = _default_output_dir("comparative_genomics")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log = ["# Comparative Genomics Analysis\n"]
    log.append(f"Analysis started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.append(f"Reference genome: {reference_genome_path}")
    log.append(f"Number of samples: {len(sample_fasta_files)}")
    log.append(f"Sample files: {', '.join(os.path.basename(f) for f in sample_fasta_files)}\n")

    log.append("## Step 1: Loading the reference")
    reference_seqs = {record.id: str(record.seq).upper() for record in SeqIO.parse(reference_genome_path, "fasta")}
    if not reference_seqs:
        log.append(f"Error: {reference_genome_path} holds no FASTA records.")
        return "\n".join(log)
    log.append(f"Loaded {len(reference_seqs)} reference sequence(s).\n")

    log.append("## Step 2: Base-by-base comparison with the reference")
    log.append(
        "No alignment is made: each sample sequence is compared with the reference sequence of the same "
        "ID, so the result holds only for sequences collinear with the reference (no indels). Positions "
        "where either base is N are skipped."
    )
    variants_by_sample = {}
    not_compared = []
    # A sample with one indel against the reference turned every downstream base into a "variant",
    # and a length mismatch was logged and then compared anyway, so a frame shift was reported as
    # thousands of findings. A sequence that is not collinear is now refused, and the call returns
    # an Error naming each one (hunt 2026-09-30, uT4-genomics-2).
    not_collinear = []
    for sample_path in sample_fasta_files:
        sample_name = os.path.basename(sample_path).split(".")[0]
        sample_seqs = {record.id: str(record.seq).upper() for record in SeqIO.parse(sample_path, "fasta")}
        shared_ids = sorted(set(reference_seqs) & set(sample_seqs))
        if not shared_ids:
            not_compared.append(sample_name)
            log.append(f"- {sample_name}: no sequence ID in common with the reference, so it was not compared")
            continue
        variants = set()
        for seq_id in shared_ids:
            ref_seq, sample_seq = reference_seqs[seq_id], sample_seqs[seq_id]
            if len(ref_seq) != len(sample_seq):
                not_collinear.append(
                    f"{sample_name}/{seq_id}: length {len(sample_seq)} vs reference {len(ref_seq)} (an indel "
                    "or a different assembly)"
                )
                continue
            ref_codes = np.frombuffer(ref_seq.encode("ascii", "replace"), dtype=np.uint8)
            sample_codes = np.frombuffer(sample_seq.encode("ascii", "replace"), dtype=np.uint8)
            called = (ref_codes != ord("N")) & (sample_codes != ord("N"))
            differs = (ref_codes != sample_codes) & called
            fraction = float(differs.sum()) / max(int(called.sum()), 1)
            if fraction > max_mismatch_fraction:
                not_collinear.append(
                    f"{sample_name}/{seq_id}: differs from the reference at {fraction:.1%} of "
                    f"{int(called.sum())} compared bases (max_mismatch_fraction={max_mismatch_fraction})"
                )
                continue
            for pos in map(int, np.flatnonzero(differs)):
                variants.add((seq_id, pos + 1, ref_seq[pos], sample_seq[pos]))
        variants_by_sample[sample_name] = variants
        log.append(f"- {sample_name}: {len(variants)} single-base differences over {len(shared_ids)} sequence(s)")

    if not_collinear:
        log.append(
            "Error: these sequences are not collinear with the reference, so a base-by-base comparison "
            "would report every base after an indel as a variant: "
            + "; ".join(not_collinear)
            + ". Align the samples to the reference and call variants with an aligner and variant caller "
            "first; raise max_mismatch_fraction only for samples known to be collinear and that divergent."
        )
        return "\n".join(log)

    if not variants_by_sample:
        log.append("Error: no sample shares a sequence ID with the reference, so nothing was compared.")
        return "\n".join(log)

    log.append("\n## Step 3: Shared and sample-specific variants")
    compared = list(variants_by_sample)
    occurrences = {}
    for sample in compared:
        for variant in variants_by_sample[sample]:
            occurrences[variant] = occurrences.get(variant, 0) + 1
    shared_by_all = sorted(v for v, k in occurrences.items() if k == len(compared))
    unique_counts = {s: sum(1 for v in variants_by_sample[s] if occurrences[v] == 1) for s in compared}

    variants_file = os.path.join(output_dir, "variants.tsv")
    table = pd.DataFrame(
        [
            {"sample": s, "seq_id": v[0], "position": v[1], "ref": v[2], "alt": v[3], "n_samples": occurrences[v]}
            for s in compared
            for v in sorted(variants_by_sample[s])
        ],
        columns=["sample", "seq_id", "position", "ref", "alt", "n_samples"],
    )
    _replace_into(variants_file, lambda p: table.to_csv(p, sep="\t", index=False))

    comparison_results = os.path.join(output_dir, "comparative_analysis.txt")
    summary_lines = ["Comparative Analysis of Genomic Regions", "=" * 38, ""]
    for sample in compared:
        summary_lines.append(
            f"{sample}: {len(variants_by_sample[sample])} variants, {unique_counts[sample]} found in no other sample"
        )
    if len(compared) > 1:
        summary_lines.append(f"\nVariants shared by all {len(compared)} compared samples: {len(shared_by_all)}")

    # Samples whose variant sets are identical are the one grouping a base-by-base comparison
    # supports; anything finer would need phasing.
    groups = {}
    for sample in compared:
        groups.setdefault(frozenset(variants_by_sample[sample]), []).append(sample)
    identical = [members for members in groups.values() if len(members) > 1]
    summary_lines.append("\nSamples with identical variant sets:")
    summary_lines.extend([f"  {', '.join(m)}" for m in identical] or ["  none"])
    _replace_into(comparison_results, lambda p: Path(p).write_text("\n".join(summary_lines) + "\n", encoding="utf-8"))

    for line in summary_lines[3:]:
        log.append(line)
    log.append(f"\nVariant table saved to {variants_file}")
    log.append(f"Comparative analysis saved to {comparison_results}")

    log.append("\n## Step 4: Haplotype structure")
    log.append(
        "Not inferred: this tool does no phasing and estimates no divergence times. The only grouping "
        "it reports is the samples with identical variant sets (above)."
    )

    log.append("\n## Summary of Findings")
    log.append(f"- Compared {len(compared)} of {len(sample_fasta_files)} samples with the reference, base by base")
    if not_compared:
        log.append(f"- Not compared (no sequence ID in common with the reference): {', '.join(not_compared)}")
    log.append(
        f"- {len(occurrences)} distinct single-base variants; {len(shared_by_all)} shared by every compared sample"
    )
    log.append(f"- All results saved to {output_dir}")

    return "\n".join(log)


def perform_chipseq_peak_calling_with_macs2(
    chip_seq_file,
    control_file,
    output_name="macs2_output",
    genome_size="hs",
    q_value=0.05,
):
    """Perform ChIP-seq peak calling using MACS2 to identify genomic regions with significant binding.

    Parameters
    ----------
    chip_seq_file : str
        Path to the ChIP-seq read data file (BAM, BED, or other supported format)
    control_file : str
        Path to the control/input data file (BAM, BED, or other supported format)
    output_name : str, optional
        Prefix for output files, with or without a directory part (default: "macs2_output"). A bare
        prefix is written under the run's output directory (``$SOG_WORK_DIR`` when set).
    genome_size : str, optional
        Effective genome size shorthand: 'hs' for human, 'mm' for mouse, etc. (default: "hs")
    q_value : float, optional
        q-value (minimum FDR) cutoff for peak calling (default: 0.05)

    Returns
    -------
    str
        A research log summarizing the peak calling process and results

    """
    import datetime
    import subprocess

    # Create log with timestamp
    log = f"ChIP-seq Peak Calling with MACS2 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += "=" * 80 + "\n\n"

    # Validate input files
    if not os.path.exists(chip_seq_file):
        log += f"Error: ChIP-seq file does not exist: {chip_seq_file}\n"
        return log

    if not os.path.exists(control_file):
        log += f"Error: Control file does not exist: {control_file}\n"
        return log

    # MACS2 writes os.path.join(--outdir, -n + suffix). -n carried the whole output_name and --outdir
    # its directory, so 'results/run1' targeted results/results/run1_* and died after the peak calling
    # was done. -n is now the bare prefix (hunt 2026-09-30, uT4-genomics-7). A bare prefix goes under
    # the run's output directory, not the CWD the portal shares across an account's chats
    # (uT4-genomics-29).
    if not os.path.dirname(output_name):
        output_name = os.path.join(_default_output_dir(), output_name)
    output_name = os.path.abspath(output_name)
    output_dir = os.path.dirname(output_name)
    output_prefix = os.path.basename(output_name)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            log += f"Created output directory: {output_dir}\n"
        except Exception as e:
            log += f"Error creating output directory: {str(e)}\n"

    # Log input parameters
    log += "Input Parameters:\n"
    log += f"- ChIP-seq data: {chip_seq_file}\n"
    log += f"- Control data: {control_file}\n"
    log += f"- Output prefix: {output_name}\n"
    log += f"- Genome size: {genome_size}\n"
    log += f"- q-value cutoff: {q_value}\n\n"

    # Check if MACS2 is installed
    try:
        check_process = subprocess.run(
            ["macs2", "--version"],
            check=False,
            capture_output=True,
            timeout=5,
        )
        log += f"MACS2 is available: {check_process.stdout.decode().strip() if check_process.stdout else check_process.stderr.decode().strip()}\n\n"
    except (subprocess.SubprocessError, FileNotFoundError):
        log += "Error: MACS2 is not installed or not available in the system PATH.\n"
        log += "Please install MACS2 using: pip install macs2\n"
        return log
    except Exception as e:
        log += f"Error checking for MACS2: {str(e)}\n"

    # Construct MACS2 command
    macs2_cmd = [
        "macs2",
        "callpeak",
        "-t",
        chip_seq_file,
        "-c",
        control_file,
        "-n",
        output_prefix,
        "-g",
        genome_size,
        "-q",
        str(q_value),
    ]

    # Add output directory parameter if it exists
    if output_dir:
        macs2_cmd.extend(["--outdir", output_dir])

    log += "Executing MACS2 with the following command:\n"
    log += " ".join(macs2_cmd) + "\n\n"

    # Define expected output files with full paths, from the same (outdir, prefix) pair MACS2 gets
    expected_files = [
        os.path.join(output_dir, f"{output_prefix}_peaks.xls"),
        os.path.join(output_dir, f"{output_prefix}_peaks.narrowPeak"),
        os.path.join(output_dir, f"{output_prefix}_summits.bed"),
    ]

    try:
        # Run MACS2
        log += "Running MACS2 peak calling...\n"
        process = subprocess.run(macs2_cmd, capture_output=True, text=True, check=True, timeout=300)

        # Log stdout and stderr (truncate if too long)
        stdout_text = process.stdout
        if len(stdout_text) > 2000:
            stdout_text = stdout_text[:1000] + "\n...[output truncated]...\n" + stdout_text[-1000:]

        stderr_text = process.stderr
        if stderr_text and len(stderr_text) > 2000:
            stderr_text = stderr_text[:1000] + "\n...[output truncated]...\n" + stderr_text[-1000:]

        log += "MACS2 Standard Output:\n"
        log += stdout_text + "\n"

        if stderr_text:
            log += "MACS2 Standard Error:\n"
            log += stderr_text + "\n"

        # Check for output files
        log += "Output Files Generated:\n"
        peak_count = 0

        for file in expected_files:
            if os.path.exists(file):
                file_size = os.path.getsize(file)
                log += f"- {file} ({file_size} bytes)\n"

                # Count peaks in narrowPeak file
                if file.endswith("narrowPeak") and os.path.exists(file):
                    try:
                        with open(file) as f:
                            peak_count = sum(1 for _ in f)
                        log += f"\nTotal peaks identified: {peak_count}\n"
                    except Exception as e:
                        log += f"Error reading peak file: {str(e)}\n"
            else:
                log += f"- {file} (Not found)\n"

        log += "\nPeak calling completed successfully.\n"

        # Add more detailed explanation about the output files
        log += "\nOutput Files Explanation:\n"
        log += "1. *_peaks.narrowPeak: BED6+4 format file containing peak locations with statistical significance\n"
        log += "   - Each line represents a peak with chromosome, start, end, and statistical significance\n"
        log += "   - Additional columns include summit position, -log10(pvalue), -log10(qvalue), and fold enrichment\n"
        log += "2. *_summits.bed: BED format file containing the summit locations (highest point) of each peak\n"
        log += "3. *_peaks.xls: Tab-delimited file with detailed information about each peak\n"

        # Add basic statistics
        log += "\nBasic Statistics:\n"
        log += f"- Total peaks called: {peak_count}\n"

    except subprocess.CalledProcessError as e:
        # "Error: " at line start, which the ReAct loop reads as a failed step (uT4-genomics-23).
        log += "Error: MACS2 execution failed:\n"
        if e.stderr:
            log += e.stderr + "\n"
        log += "Peak calling failed.\n"
    except subprocess.TimeoutExpired:
        log += "Error: MACS2 execution timed out after 300 seconds.\n"
        log += "This may happen with very large input files or insufficient computational resources.\n"
    except Exception as e:
        log += f"Error: peak calling failed unexpectedly: {str(e)}\n"

    return log


def find_enriched_motifs_with_homer(
    peak_file,
    genome="hg38",
    background_file=None,
    motif_length="8,10,12",
    output_dir=None,
    num_motifs=10,
    threads=4,
):
    """Find DNA sequence motifs enriched in genomic regions using the HOMER motif discovery software.

    Parameters
    ----------
    peak_file : str
        Path to peak file in BED format containing genomic regions to analyze for motif enrichment
    genome : str, optional
        Reference genome for sequence extraction (default: "hg38")
    background_file : str, optional
        Path to BED file with background regions for comparison. If None, HOMER will generate random
        background sequences automatically (default: None)
    motif_length : str, optional
        Comma-separated list of motif lengths to discover (default: "8,10,12")
    output_dir : str, optional
        Directory to save output files (default: ``homer_motifs/`` under the run's output directory,
        ``$SOG_WORK_DIR`` when set)
    num_motifs : int, optional
        Number of motifs to find (default: 10)
    threads : int, optional
        Number of CPU threads to use (default: 4)

    Returns
    -------
    str
        A research log summarizing the motif discovery process and results

    """
    import datetime
    import re
    import subprocess

    # "./homer_motifs" was the CWD the portal's REPL shares between an account's chats
    # (hunt 2026-09-30, uT4-genomics-29).
    if output_dir is None:
        output_dir = _default_output_dir("homer_motifs")
    output_dir = os.path.abspath(output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize research log
    log = f"Motif Discovery with HOMER - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += "=" * 80 + "\n\n"

    # Log input parameters
    log += "Input Parameters:\n"
    log += f"- Peak file: {peak_file}\n"
    log += f"- Reference genome: {genome}\n"
    log += f"- Background: {'User-defined' if background_file else 'HOMER-generated'}\n"
    log += f"- Motif lengths: {motif_length}\n"
    log += f"- Number of motifs: {num_motifs}\n"
    log += f"- Threads: {threads}\n\n"

    # Construct HOMER findMotifsGenome.pl command
    homer_cmd = [
        "findMotifsGenome.pl",
        peak_file,
        genome,
        output_dir,
        "-size",
        "200",  # Default search region size (200bp centered on peak)
        "-len",
        motif_length,
        "-S",
        str(num_motifs),
        "-p",
        str(threads),
    ]

    # Add background file if provided
    if background_file:
        homer_cmd.extend(["-bg", background_file])

    log += "Executing HOMER with the following command:\n"
    log += " ".join(homer_cmd) + "\n\n"

    try:
        # Run HOMER
        log += "Running HOMER motif discovery...\n"
        process = subprocess.run(homer_cmd, capture_output=True, text=True, check=True)

        # Log stdout and stderr (truncate if too long)
        stdout_text = process.stdout
        if len(stdout_text) > 2000:
            stdout_text = stdout_text[:1000] + "\n...[output truncated]...\n" + stdout_text[-1000:]

        log += "HOMER Standard Output (truncated if long):\n"
        log += stdout_text + "\n"

        # Check and parse results
        log += "\n## Motif Discovery Results\n\n"

        # Look for HTML results file
        html_results = os.path.join(output_dir, "homerResults.html")
        if os.path.exists(html_results):
            log += f"HOMER results saved to {html_results}\n\n"

        # Parse motif files. findMotifsGenome.pl writes the de novo motifs as homerResults/motifN.motif
        # (beside motifN.similar.K.motif and motifN.RV.motif); the root holds homerResults.html and
        # knownResults*. The glob looked for motif*.motif in the root, matched nothing and reported
        # 0 motifs for every run, and a string sort would put motif10 before motif2
        # (hunt 2026-09-30, uT4-genomics-10).
        homer_results_dir = os.path.join(output_dir, "homerResults")
        numbered = []
        if os.path.isdir(homer_results_dir):
            for name in os.listdir(homer_results_dir):
                match = re.fullmatch(r"motif(\d+)\.motif", name)
                if match:
                    numbered.append((int(match.group(1)), os.path.join(homer_results_dir, name)))
        motif_files = [path for _, path in sorted(numbered)]

        if motif_files:
            log += f"Found {len(motif_files)} enriched motifs:\n\n"

            # Parse each motif file for information
            for motif_file in motif_files[:5]:  # Show info for top 5 motifs only
                with open(motif_file) as f:
                    header = f.readline().strip()
                    # Extract motif name, p-value, etc. from header
                    # Example header: >ATGCGCTA	MA0139.1(CTCF)/Jaspar	1e-10	-1.234e+02	0	0.00%
                    parts = header.split("\t")
                    if len(parts) >= 3:
                        motif_id = os.path.basename(motif_file)
                        motif_consensus = parts[0].replace(">", "")

                        # Try to extract transcription factor info if available
                        tf_match = ""
                        if len(parts) > 1:
                            tf_match = parts[1]

                        # Extract p-value if available
                        p_value = ""
                        if len(parts) > 2:
                            p_value = parts[2]

                        log += f"Motif: {motif_id}\n"
                        log += f"- Consensus: {motif_consensus}\n"
                        log += f"- Matching TF: {tf_match}\n"
                        log += f"- P-value: {p_value}\n\n"

            if len(motif_files) > 5:
                log += f"... and {len(motif_files) - 5} more motifs ...\n\n"
        else:
            log += "No significant motifs were found.\n\n"

        # Check for known motif enrichment results
        known_results = os.path.join(output_dir, "knownResults.html")
        if os.path.exists(known_results):
            log += "Known motif enrichment results are available in knownResults.html\n"

            # Try to parse known motifs results
            known_motifs_txt = os.path.join(output_dir, "knownResults.txt")
            if os.path.exists(known_motifs_txt):
                with open(known_motifs_txt) as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Header + at least one motif
                        log += "\nTop known motifs enriched in the dataset:\n"
                        # Show top 3 known motifs if available
                        for line in lines[1:4]:  # Skip header, show top 3
                            parts = line.strip().split("\t")
                            if len(parts) >= 4:
                                log += f"- {parts[0]} (p-value: {parts[2]})\n"

        # Summarize results
        log += "\n## Summary\n"
        log += f"- Analyzed genomic regions from {peak_file}\n"
        log += f"- Discovered {len(motif_files)} enriched motifs\n"
        log += f"- All results saved to {output_dir}\n"

    except subprocess.CalledProcessError as e:
        # "Error: " at line start, which the ReAct loop reads as a failed step (uT4-genomics-23).
        log += "Error: HOMER execution failed:\n"
        log += (e.stderr or "") + "\n"
        log += "Motif discovery failed.\n"
    except Exception as e:
        log += f"Error: motif discovery failed unexpectedly: {str(e)}\n"

    return log


def analyze_genomic_region_overlap(region_sets, output_prefix="overlap_analysis"):
    """Analyze overlaps between two or more sets of genomic regions.

    Parameters
    ----------
    region_sets : list
        List of genomic region sets. Each item can be either:
        - A string path to a BED file
        - A list of tuples/lists with format (chrom, start, end) or (chrom, start, end, name)
    output_prefix : str, optional
        Prefix for output files (default: "overlap_analysis"). A bare prefix is written under the
        run's output directory (``$SOG_WORK_DIR`` when set).

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    from datetime import datetime

    # A str that was not an existing file fell through to the coordinate branch and was iterated
    # character by character: nothing was written, the log said "Created Region_Set_2 from 13
    # provided coordinates", and every comparison with it reported 0 overlaps -- a typo became a
    # confident negative result. Every set is checked before anything runs (hunt 2026-09-30,
    # uT4-genomics-9).
    for i, regions in enumerate(region_sets):
        if isinstance(regions, (str, os.PathLike)):
            if not os.path.isfile(regions):
                return f"Error: BED file not found for region set {i + 1}: {regions}"
        elif not isinstance(regions, (list, tuple)):
            return (
                f"Error: region set {i + 1} is a {type(regions).__name__}; pass a BED file path or a list of "
                "(chrom, start, end[, name]) tuples."
            )
        else:
            bad = [r for r in regions if not isinstance(r, (list, tuple)) or len(r) < 3]
            if bad:
                return (
                    f"Error: region set {i + 1} has {len(bad)} entries that are not (chrom, start, end[, name]) "
                    f"tuples, e.g. {bad[0]!r}; they would have been dropped without a word."
                )

    import pandas as pd
    import pybedtools

    # A bare prefix lands in the run's output directory, not the CWD the portal shares across an
    # account's chats, and the log names absolute paths (uT4-genomics-29).
    if not os.path.dirname(str(output_prefix)):
        output_prefix = os.path.join(_default_output_dir(), str(output_prefix))
    output_prefix = os.path.abspath(output_prefix)
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    # Start research log
    log = "# Genomic Region Overlap Analysis\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Create BedTool objects from inputs
    bedtools = []
    set_names = []

    log += "## Input Processing\n"
    for i, regions in enumerate(region_sets):
        set_name = f"Region_Set_{i + 1}"
        set_names.append(set_name)

        if isinstance(regions, (str, os.PathLike)):
            # Input is a file path
            bedtool = pybedtools.BedTool(str(regions))
            log += f"- Loaded {set_name} from file: {regions}\n"
        else:
            # Input is a list of coordinates
            temp_bed_file = f"{output_prefix}_{set_name}.bed"
            with open(temp_bed_file, "w") as f:
                for region in regions:
                    if len(region) >= 3:
                        chrom, start, end = region[0], region[1], region[2]
                        name = region[3] if len(region) > 3 else f"feature_{i}"
                        f.write(f"{chrom}\t{start}\t{end}\t{name}\n")
            bedtool = pybedtools.BedTool(temp_bed_file)
            log += f"- Created {set_name} from {len(regions)} provided coordinates\n"

        bedtools.append(bedtool)

    log += "\n## Analysis Results\n"

    # Calculate basic statistics for each set
    stats = []
    for i, bt in enumerate(bedtools):
        # Calculate total base pairs from start and end coordinates
        total_bp = sum(int(feature.end) - int(feature.start) for feature in bt)
        stats.append({"Set": set_names[i], "Regions": len(bt), "Total_BP": total_bp})

    # Create a results dataframe to store pairwise overlaps
    results = []

    # Perform pairwise overlaps
    for i in range(len(bedtools)):
        for j in range(i + 1, len(bedtools)):
            # Find overlaps using a simpler approach - just get the overlap regions
            try:
                # Use intersect with -u option to get unique overlapping features
                # and -wa option to get the original features from the first set
                overlapping_regions_a = bedtools[i].intersect(bedtools[j], u=True)

                # Get overlapping features from the second set
                overlapping_regions_b = bedtools[j].intersect(bedtools[i], u=True)

                # Calculate statistics
                overlap_regions_count_a = len(overlapping_regions_a)
                overlap_regions_count_b = len(overlapping_regions_b)

                # Now find the actual overlap coordinates using intersect with -wo
                # -wo outputs the original features plus the overlap length
                overlap_with_bases = bedtools[i].intersect(bedtools[j], wo=True)

                if len(overlap_with_bases) > 0:
                    # Extract overlap base pairs from the last column of -wo output
                    overlap_bp = 0
                    unique_overlaps = set()

                    for line in overlap_with_bases:
                        # The last field in -wo output is the overlap size in bases
                        # Convert all fields to strings first to handle different data types
                        fields = [str(field) for field in line]

                        # The actual overlap width is usually the last column with -wo
                        try:
                            # Get overlap width from last field
                            overlap_width = int(fields[-1])
                            overlap_bp += overlap_width

                            # Create a unique identifier for this overlap
                            # Using the coordinates from both sets
                            chrom_a = fields[0]
                            start_a = int(fields[1])
                            end_a = int(fields[2])
                            chrom_b = fields[len(fields) // 2]  # Middle field is typically the start of second feature
                            identifier = f"{chrom_a}:{start_a}-{end_a}_{chrom_b}"
                            unique_overlaps.add(identifier)
                        except (ValueError, IndexError) as e:
                            # If parsing fails, try an alternative approach
                            log += f"  Warning: Couldn't parse overlap width, falling back to alternative method: {str(e)}\n"

                            # Alternative: manually calculate overlap from coordinates
                            try:
                                midpoint = len(fields) // 2
                                start_a = int(fields[1])
                                end_a = int(fields[2])
                                start_b = int(fields[midpoint + 1])
                                end_b = int(fields[midpoint + 2])

                                overlap_start = max(start_a, start_b)
                                overlap_end = min(end_a, end_b)

                                if overlap_end > overlap_start:
                                    overlap_bp += overlap_end - overlap_start
                            except (ValueError, IndexError):
                                # If even this fails, log the issue and continue
                                log += f"  Warning: Couldn't calculate overlap from: {fields}\n"

                    # Number of unique overlapping regions
                    overlap_regions = len(unique_overlaps)

                    # Calculate percentages
                    pct_of_set1 = (overlap_bp / stats[i]["Total_BP"]) * 100 if stats[i]["Total_BP"] > 0 else 0
                    pct_of_set2 = (overlap_bp / stats[j]["Total_BP"]) * 100 if stats[j]["Total_BP"] > 0 else 0

                    results.append(
                        {
                            "Set1": set_names[i],
                            "Set2": set_names[j],
                            "Overlap_Regions": overlap_regions,
                            "Overlap_BP": overlap_bp,
                            "Pct_of_Set1": pct_of_set1,
                            "Pct_of_Set2": pct_of_set2,
                        }
                    )

                    # Save detailed overlaps to file
                    overlap_file = f"{output_prefix}_{set_names[i]}_{set_names[j]}_overlaps.bed"
                    overlap_with_bases.saveas(overlap_file)

                    log += f"- Between {set_names[i]} and {set_names[j]}:\n"
                    log += f"  * Overlapping regions from set 1: {overlap_regions_count_a}\n"
                    log += f"  * Overlapping regions from set 2: {overlap_regions_count_b}\n"
                    log += f"  * Unique overlaps: {overlap_regions}\n"
                    log += f"  * Total overlap size: {overlap_bp} bp\n"
                    log += f"  * Percentage of {set_names[i]}: {pct_of_set1:.2f}%\n"
                    log += f"  * Percentage of {set_names[j]}: {pct_of_set2:.2f}%\n"
                    log += f"  * Detailed overlaps saved to: {overlap_file}\n\n"
                else:
                    # No overlaps found
                    results.append(
                        {
                            "Set1": set_names[i],
                            "Set2": set_names[j],
                            "Overlap_Regions": 0,
                            "Overlap_BP": 0,
                            "Pct_of_Set1": 0,
                            "Pct_of_Set2": 0,
                        }
                    )
                    log += f"- No overlaps found between {set_names[i]} and {set_names[j]}\n\n"
            except Exception as e:
                # Handle any errors during the intersection process
                # "Error: " at line start, which the ReAct loop reads as a failed step (uT4-genomics-23).
                log += f"Error: overlap analysis between {set_names[i]} and {set_names[j]} failed: {str(e)}\n"
                results.append(
                    {
                        "Set1": set_names[i],
                        "Set2": set_names[j],
                        "Overlap_Regions": 0,
                        "Overlap_BP": 0,
                        "Pct_of_Set1": 0,
                        "Pct_of_Set2": 0,
                        "Error": str(e),
                    }
                )

    # Save summary statistics to file
    summary_file = f"{output_prefix}_summary.tsv"
    summary_df = pd.DataFrame(results)
    _replace_into(summary_file, lambda p: summary_df.to_csv(p, sep="\t", index=False))

    log += "## Summary\n"
    log += f"- Total sets analyzed: {len(bedtools)}\n"
    log += f"- Pairwise comparisons: {len(results)}\n"
    log += f"- Summary statistics saved to: {summary_file}\n"

    return log


def generate_embeddings_with_state(
    adata_filename,
    data_dir,
    model_folder,
    output_filename=None,
    checkpoint=None,
    embed_key="X_state",
    protein_embeddings=None,
    batch_size=500,
    output_dir=None,
):
    """
    Generate State embeddings for single-cell RNA-seq data using the SE-600M model.

    This function downloads the SE-600M model from Hugging Face (when it is not already in
    ``model_folder``) and generates embeddings for the input AnnData object. It installs nothing:
    git-lfs and uv must already be on PATH, and ``uv run state`` must resolve the arc-state package.

    Parameters:
    -----------
    adata_filename : str
        Name of the input AnnData file (.h5ad format)
    data_dir : str
        Directory containing the input data file
    model_folder : str
        Directory where the SE-600M model will be downloaded and stored
    output_filename : str, optional
        Name of the output file. If None, will use input filename with '_state_embeddings' suffix
    checkpoint : str, optional
        Path to the specific model checkpoint. If None, uses the latest checkpoint in model_folder
    embed_key : str, default="X_state"
        Name of key to store embeddings in the output AnnData object
    protein_embeddings : str, optional
        Path to protein embeddings override (.pt). If omitted, auto-detects in model folder
    batch_size : int, default=500
        Batch size for embedding forward pass. Increase to use more VRAM and speed up embedding
    output_dir : str, optional
        Directory for the output file. Default: ``state/`` under the run's output directory
        (``$SOG_WORK_DIR`` when set) -- never next to the input.

    Returns:
    --------
    str
        Research log summarizing the steps performed and results

    Notes:
    ------
    - Requires 10GB+ GPU memory for model loading
    - Downloads ~25GB model files from Hugging Face
    - Needs git-lfs and uv on PATH and the arc-state package reachable by ``uv run state``; it raises
      when git-lfs is missing and installs none of them (the description said it installed all three)
    - Validates input h5ad file format (CSR matrix and gene_name column)
    - Provides real-time streaming output for git clone and embedding generation
    - Includes automatic retry with reduced batch size on failure
    """
    import os
    import subprocess

    # Initialize steps list for logging
    steps = []

    # Set up paths
    input_path = os.path.join(data_dir, adata_filename)
    if output_filename is None:
        base_name = os.path.splitext(adata_filename)[0]
        output_filename = f"{base_name}_state_embeddings.h5ad"

    # Into output_dir, not {data_dir}/output next to the input -- often the shared dataset library
    # (hunt 2026-09-30, uT4-genomics-14).
    if output_dir is None:
        output_dir = _default_output_dir("state")
    output_path = os.path.abspath(os.path.join(output_dir, output_filename))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    steps.append(f"Created output directory: {os.path.dirname(output_path)}")

    # Validate h5ad file format
    steps.append("Validating h5ad file format...")
    try:
        import scanpy as sc

        adata = sc.read_h5ad(input_path)

        import warnings

        if not hasattr(adata.X, "indices"):
            warning_msg = (
                "Input data is not in CSR (Compressed Sparse Row) matrix format. Proceeding, but this may cause issues."
            )
            warnings.warn(warning_msg, stacklevel=2)
            steps.append(f"Warning: {warning_msg}")

        if "gene_name" not in adata.var.columns:
            warning_msg = (
                "'gene_name' column is not present in the var dataframe. Proceeding, but this may cause issues."
            )
            warnings.warn(warning_msg, stacklevel=2)
            steps.append(f"Warning: {warning_msg}")

        steps.append(f"  - Matrix format: {type(adata.X).__name__}")
        steps.append(f"  - Number of cells: {adata.n_obs}")
        steps.append(f"  - Number of genes: {adata.n_vars}")
        steps.append(f"  - Gene names available: {'gene_name' in adata.var.columns}")

    except Exception as e:
        error_msg = f"h5ad file validation failed: {e}"
        steps.append(error_msg)
        raise ValueError(error_msg) from e

    # Check if git-lfs is installed
    steps.append("Checking if git-lfs is installed...")
    try:
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        steps.append("git-lfs is available")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        error_msg = (
            "git-lfs is not installed or not found in PATH. "
            "Please install git-lfs before running this function.\n"
            "Run the following command with root privileges:\n"
            "sudo apt-get install git-lfs\n"
            "git lfs install"
        )
        steps.append(error_msg)
        raise RuntimeError(error_msg) from e

    steps.append("Downloading SE-600M model...")

    # Check if model weights actually exist (not just the directory)
    model_exists = False
    if os.path.exists(model_folder):
        # Check for SE-600M specific checkpoint files
        checkpoint_indicators = ["se600m_epoch16.ckpt", "se600m_epoch4.ckpt", "model.safetensors", "config.yaml"]
        model_exists = any(os.path.exists(os.path.join(model_folder, indicator)) for indicator in checkpoint_indicators)

    if not model_exists:
        os.makedirs(model_folder, exist_ok=True)
        steps.append(f"Created model directory: {model_folder}")
        try:
            steps.append(f"Cloning SE-600M model to {model_folder}")
            steps.append("This may take a while and final model weights are approx 25GB!...")
            process = subprocess.Popen(
                ["git", "clone", "https://huggingface.co/arcinstitute/SE-600M", model_folder],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            for line in iter(process.stdout.readline, ""):
                line = line.rstrip()
                print(line)
                steps.append(line)

            process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, ["git", "clone", "https://huggingface.co/arcinstitute/SE-600M", model_folder]
                )
            steps.append("SE-600M model downloaded successfully")
        except subprocess.CalledProcessError as e:
            error_msg = f"Error downloading model: {e}"
            steps.append(error_msg)
            raise
    else:
        steps.append(f"Model already downloaded: {model_folder}")

    steps.append("Generating embeddings...")

    # torch is only for the GPU report below; the embedding runs in `uv run state`'s own environment.
    # It was a module-level import, which made every tool in this module unimportable where torch is
    # absent (hunt 2026-09-30, uT4-genomics-3).
    try:
        import torch
    except ImportError:
        torch = None
        steps.append("PyTorch is not importable in this interpreter, so no GPU report; `uv run state` uses its own.")

    if torch is None:
        pass
    elif torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        current_device = torch.cuda.current_device()
        steps.append(f"✓ GPU detected: {device_name}")
        steps.append(f"✓ GPU memory: {gpu_memory:.1f} GB")
        steps.append(f"✓ Current CUDA device: {current_device}")
        if gpu_memory < 10:
            warning_msg = f"⚠️  WARNING: GPU has only {gpu_memory:.1f} GB memory. SE600 model requires 10GB+ GPU memory."
            steps.append(warning_msg)
    else:
        steps.append("⚠️  WARNING: No GPU detected! This will run on CPU and be very slow.")
        steps.append("⚠️  SE600 model requires 10GB+ GPU memory for optimal performance.")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    steps.append(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

    if torch is not None:
        steps.append(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch is not None and torch.cuda.is_available():
        steps.append(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        steps.append(f"PyTorch current device: {torch.cuda.current_device()}")

    steps.append("embedding data can take a while...")
    max_retries = 3
    retries = 0
    last_exception = None

    while retries <= max_retries:
        try:
            cmd = [
                "uv",
                "run",
                "state",
                "emb",
                "transform",
                "--model-folder",
                model_folder,
                "--input",
                input_path,
                "--output",
                output_path,
            ]

            if checkpoint:
                cmd.extend(["--checkpoint", checkpoint])
            if embed_key != "X_state":
                cmd.extend(["--embed-key", embed_key])
            if protein_embeddings:
                cmd.extend(["--protein-embeddings", protein_embeddings])
            if batch_size:
                cmd.extend(["--batch-size", str(batch_size)])

            steps.append(f"Running command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
            )

            for line in iter(process.stdout.readline, ""):
                line = line.rstrip()
                print(line)  # Print to console first
                steps.append(line)  # Then add to steps list

            process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

            steps.append("Embeddings generated successfully!")
            steps.append("Output Format and Location:")
            steps.append("  Output format: .h5ad (AnnData format)")
            steps.append("  Embeddings stored in: X_state layer (by default)")
            steps.append("  Location: Specified by --output parameter")
            steps.append(f"  In your code: Saved to {output_path}")
            steps.append("")
            return "\n".join(steps)

        except subprocess.CalledProcessError as e:
            error_msg = f"Error generating embeddings (attempt {retries + 1}): {e}"
            steps.append(error_msg)
            last_exception = e
            if batch_size and retries < max_retries:
                batch_size = max(1, batch_size // 2)
                steps.append(f"Retrying with reduced batch size: {batch_size}")
                retries += 1
            else:
                break

    steps.append("Failed to generate embeddings after multiple attempts.")
    if last_exception:
        steps.append(f"Last exception: {last_exception}")
        raise last_exception

    return "\n".join(steps)


def generate_transcriptformer_embeddings(
    adata_filename: str,
    data_dir: str,
    output_filename: str = None,
    model_type: str = "tf-sapiens",
    checkpoint_path: str = None,
    batch_size: int = 8,
    precision: str = "16-mixed",
    clip_counts: int = 30,
    embedding_layer_index: int = -1,
    num_gpus: int = 1,
    n_data_workers: int = 0,
    gene_col_name: str = "ensembl_id",
    pretrained_embedding: str = None,
    filter_to_vocabs: bool = True,
    use_raw: str = "None",
    emb_type: str = "cell",
    remove_duplicate_genes: bool = False,
    oom_dataloader: bool = False,
    output_dir: str = None,
) -> str:
    """
    Generate Transcriptformer embeddings for single-cell RNA-seq data.

    Parameters:
    -----------
    adata_filename : str
        Name of the input AnnData file (.h5ad format)
    data_dir : str
        Directory containing the input data file
    output_filename : str, optional
        Name of the output embeddings file. If None, will use input filename with '_transcriptformer_embeddings' suffix
    model_type : str, optional
        Type of transcriptformer model to download and use. Options: "tf-sapiens", "tf-exemplar", "tf-metazoa" (default: "tf-sapiens")
    checkpoint_path : str, optional
        Path to the transcriptformer checkpoint directory. If None, will use "./checkpoints/{model_type}" (default: None)
    batch_size : int, optional
        Batch size for inference (default: 8)
    precision : str, optional
        Precision for inference. Options: "16-mixed", "32" (default: "16-mixed")
    clip_counts : int, optional
        Maximum count value to clip to (default: 30)
    embedding_layer_index : int, optional
        Which layer to extract embeddings from (default: -1, last layer)
    num_gpus : int, optional
        Number of GPUs to use (default: 1)
    n_data_workers : int, optional
        Number of data loading workers (default: 0)
    gene_col_name : str, optional
        Column name in AnnData.var containing gene identifiers (default: "ensembl_id")
    pretrained_embedding : str, optional
        Path to pretrained embeddings for out-of-distribution species (default: None)
    filter_to_vocabs : bool, optional
        Whether to filter genes to only those in the vocabulary (default: True)
    use_raw : str, optional
        Whether to use raw counts from AnnData.raw.X (True), adata.X (False), or auto-detect (None/auto) (default: "None")
    emb_type : str, optional
        Type of embeddings to extract: 'cell' for mean-pooled cell embeddings or 'cge' for contextual gene embeddings (default: "cell")
    remove_duplicate_genes : bool, optional
        Remove duplicate genes if found instead of raising an error (default: False)
    oom_dataloader : bool, optional
        Use map-style out-of-memory DataLoader (DistributedSampler-friendly) (default: False)
    output_dir : str, optional
        Directory for the embeddings file and the prepared input copy. Default: ``transcriptformer/``
        under the run's output directory (``$SOG_WORK_DIR`` when set) -- never next to the input.
    Returns:
    --------
    str
        Path to the generated embeddings file
    """
    import os
    import subprocess

    import pandas as pd
    import scanpy as sc
    import torch

    if checkpoint_path is None:
        checkpoint_path = f"./checkpoints/{model_type}"

    input_path = os.path.join(data_dir, adata_filename)
    if output_filename is None:
        base_name = os.path.splitext(adata_filename)[0]
        output_filename = f"{base_name}_transcriptformer_embeddings.h5ad"

    # Into output_dir. The output went to {data_dir}/output and a full-size prepared copy to
    # {data_dir}/temp_transcriptformer_input.h5ad -- next to the input, often the shared dataset
    # library -- and the copy was left there whenever inference raised (hunt 2026-09-30,
    # uT4-genomics-14). The copy now has a per-call name and is removed in a finally.
    import uuid

    if output_dir is None:
        output_dir = _default_output_dir("transcriptformer")
    output_path = os.path.abspath(os.path.join(output_dir, output_filename))
    temp_data_path = os.path.join(os.path.dirname(output_path), f".transcriptformer_input_{uuid.uuid4().hex[:8]}.h5ad")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        current_device = torch.cuda.current_device()
        print(f"✓ GPU detected: {device_name}")
        print(f"✓ GPU memory: {gpu_memory:.1f} GB")
        print(f"✓ Current CUDA device: {current_device}")
    else:
        print("⚠️  WARNING: No GPU detected! This will run on CPU and be very slow.")
        print("⚠️  Transcriptformer requires GPU for optimal performance.")

    import os

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        print(f"PyTorch current device: {torch.cuda.current_device()}")

    try:
        os.makedirs(checkpoint_path, exist_ok=True)

        def is_model_downloaded(checkpoint_path, model_type):
            """Check if model is already downloaded by looking for key files"""
            required_files = [
                "config.yaml",
                "pytorch_model.bin",
            ]

            for file_name in required_files:
                file_path = os.path.join(checkpoint_path, file_name)
                if not os.path.exists(file_path):
                    return False

            return True

        if is_model_downloaded(checkpoint_path, model_type):
            print(f"✓ Model checkpoints for {model_type} already exist at: {checkpoint_path}")
            print("✓ Skipping download...")
        else:
            print(f"Downloading transcriptformer model checkpoints for {model_type}...")
            print("This may take a while and model checkpoints are several GB!...")
            print(f"Model will be saved to: {checkpoint_path}")

            try:
                process = subprocess.Popen(
                    ["transcriptformer", "download", model_type],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                for line in iter(process.stdout.readline, ""):
                    print(line.rstrip())

                process.wait()

                if process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        process.returncode, ["transcriptformer", "download", model_type]
                    )
                print(f"✓ Model checkpoints for {model_type} downloaded successfully")
            except subprocess.CalledProcessError as e:
                print(f"Error downloading model checkpoints: {e}")
                raise

        print("Loading and preparing AnnData object...")
        try:
            adata = sc.read_h5ad(input_path)
            print(f"✓ Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes")

            def check_ensembl_patterns(gene_index):
                patterns = {
                    "human_ensembl": r"^ENSG\d{11}(\.\d+)?$",
                    "mouse_ensembl": r"^ENSMUSG\d{11}(\.\d+)?$",
                    "zebrafish_ensembl": r"^ENSDARG\d{11}(\.\d+)?$",
                    "fly_ensembl": r"^FBgn\d{7}$",
                    "worm_ensembl": r"^WBGene\d{8}$",
                    "yeast_ensembl": r"^Y[A-P][LR]\d{3}[WC]?$",
                    "generic_ensembl": r"^ENS[A-Z]{3}G\d{11}(\.\d+)?$",
                }

                results = {}
                for pattern_name, pattern in patterns.items():
                    matches = gene_index.str.match(pattern, na=False)
                    count = matches.sum()
                    results[pattern_name] = {
                        "count": count,
                        "percentage": (count / len(gene_index)) * 100,
                        "matches": gene_index[matches].tolist()[:5] if count > 0 else [],
                    }

                return results

            print("Analyzing gene index for Ensembl ID patterns...")
            ensembl_analysis = check_ensembl_patterns(adata.var.index)

            for pattern_name, data in ensembl_analysis.items():
                if data["count"] > 0:
                    print(f"✓ {pattern_name}: {data['count']} genes ({data['percentage']:.1f}%)")
                    if data["matches"]:
                        print(f"  Examples: {data['matches']}")

            best_pattern = max(ensembl_analysis.items(), key=lambda x: x[1]["count"])
            if best_pattern[1]["count"] > 0:
                print(f"✓ Best match: {best_pattern[0]} with {best_pattern[1]['count']} genes")
            else:
                print("❌ ERROR: No clear Ensembl ID patterns detected in gene index")
                print(f"  Gene index examples: {adata.var.index[:5].tolist()}")
                print("  This dataset cannot be processed by transcriptformer.")
                print("  Please ensure your data contains valid Ensembl gene IDs.")
                raise ValueError(
                    "No Ensembl ID patterns found in gene index. Cannot proceed with transcriptformer inference."
                )

            if "ensembl_id" not in adata.var.columns:
                if "gene_ids" in adata.var.columns:
                    print("✓ Found 'gene_ids' column, renaming to 'ensembl_id'")
                    adata.var["ensembl_id"] = adata.var["gene_ids"]
                elif best_pattern[1]["count"] > 0:
                    adata.var["ensembl_id"] = adata.var.index
                    print(f"✓ Using gene index as ensembl_id (detected {best_pattern[0]} pattern)")
                else:
                    print("❌ ERROR: No 'ensembl_id' column found and gene index doesn't match Ensembl patterns")
                    print("  This dataset cannot be processed by transcriptformer.")
                    print("  Please ensure your data contains valid Ensembl gene IDs.")
                    raise ValueError("No valid Ensembl gene IDs found. Cannot proceed with transcriptformer inference.")

            if adata.raw is None:
                print("⚠️  Warning: Transcriptformer expects raw (unnormalized) count data in adata.X.")
                if not pd.api.types.is_integer_dtype(adata.X.dtype):
                    print("⚠️  Warning: adata.X does not appear to be unnormalized counts (integer type not detected).")
                adata.raw = adata
                print("✓ Set adata.raw to current adata")

            if remove_duplicate_genes:
                print("Pre-processing: Removing duplicate genes to prevent dimension mismatch...")
                adata.var["ensembl_id_clean"] = adata.var["ensembl_id"].str.split(".").str[0]

                duplicate_mask = adata.var["ensembl_id_clean"].duplicated(keep="first")
                n_duplicates = duplicate_mask.sum()

                if n_duplicates > 0:
                    print(f"Found {n_duplicates} duplicate genes, removing them...")
                    adata = adata[:, ~duplicate_mask].copy()
                    print(f"✓ Removed {n_duplicates} duplicate genes. Remaining: {adata.n_vars} genes")
                else:
                    print("✓ No duplicate genes found")

                adata.var["ensembl_id"] = adata.var["ensembl_id_clean"]
                adata.var.drop("ensembl_id_clean", axis=1, inplace=True)

            adata.write(temp_data_path)
            print(f"✓ Saved prepared AnnData to {temp_data_path}")

        except Exception as e:
            print(f"Error preparing AnnData object: {e}")
            raise

        print("Running transcriptformer inference...")
        print("This may take a while depending on data size and GPU...")
        try:
            cmd = [
                "transcriptformer",
                "inference",
                "--checkpoint-path",
                checkpoint_path,
                "--data-file",
                temp_data_path,
                "--output-path",
                os.path.dirname(output_path),
                "--output-filename",
                os.path.basename(output_path),
                "--batch-size",
                str(batch_size),
                "--gene-col-name",
                gene_col_name,
                "--precision",
                precision,
                "--clip-counts",
                str(clip_counts),
                "--model-type",
                "transcriptformer",
                "--use-raw",
                use_raw,
                "--emb-type",
                emb_type,
                "--num-gpus",
                str(num_gpus),
                "--n-data-workers",
                str(n_data_workers),
            ]

            # Check if assay column exists in obs, if not create it
            if "assay" not in adata.obs.columns:
                print("✓ Creating 'assay' column in obs metadata with 'unknown' values")
                adata.obs["assay"] = "unknown"
                # Save the updated adata with assay column
                adata.write(temp_data_path)
                print(f"✓ Updated AnnData saved to {temp_data_path}")

            if pretrained_embedding is not None:
                cmd.extend(["--pretrained-embedding", pretrained_embedding])

            if filter_to_vocabs:
                cmd.append("--filter-to-vocabs")

            if remove_duplicate_genes:
                cmd.append("--remove-duplicate-genes")

            if oom_dataloader:
                cmd.append("--oom-dataloader")

            # The layer was accepted and advertised but never sent, so an intermediate-layer request
            # silently returned last-layer embeddings (hunt 2026-09-30, uT4-genomics-20). Sent only
            # when it differs from the CLI's own default of -1.
            if int(embedding_layer_index) != -1:
                cmd.extend(["--embedding-layer-index", str(int(embedding_layer_index))])

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
            )

            for line in iter(process.stdout.readline, ""):
                print(line.rstrip())

            process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

            print("✓ Transcriptformer inference completed successfully")
            print(f"✓ Output saved to: {output_path}")

        except subprocess.CalledProcessError as e:
            print(f"Error running transcriptformer inference: {e}")
            raise

        print("✓ Transcriptformer embeddings generated successfully!")
        print("\nOutput Format and Location:")
        print("  Output format: .h5ad (AnnData format)")

        return output_path

    except Exception as e:
        print(f"Fatal error in transcriptformer embedding generation: {e}")
        raise
    finally:
        try:
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
                print("✓ Cleaned up temporary files")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up temporary files: {e}")
