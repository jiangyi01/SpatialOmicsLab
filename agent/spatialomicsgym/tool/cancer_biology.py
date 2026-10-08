def analyze_ddr_network_in_cancer(expression_data_path, mutation_data_path, output_dir="./results"):
    """Analyze DNA Damage Response (DDR) network alterations and dependencies in cancer samples.

    This function reconstructs the DDR network from genomic data, identifies disruptions
    in the network, and analyzes dependencies between DDR pathway components in cancer.

    Parameters
    ----------
    expression_data_path : str
        Path to gene expression data file (CSV format with genes as rows, samples as columns)
    mutation_data_path : str
        Path to mutation data file (CSV format with genes as rows, samples as columns,
        values indicating mutation status)
    output_dir : str, optional
        Directory to save output files (default: "./results")

    Returns
    -------
    str
        Research log summarizing the DDR network analysis, findings about disruptions,
        and potential therapeutic vulnerabilities

    """
    import os
    from datetime import datetime

    import gseapy as gp
    import networkx as nx
    import pandas as pd
    from scipy.stats import pearsonr

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    research_log = f"DDR Network Analysis Research Log - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    research_log += "=" * 80 + "\n\n"

    # Step 1: Load and preprocess genomic data
    research_log += "STEP 1: Loading and preprocessing genomic data\n"

    try:
        expr_data = pd.read_csv(expression_data_path, index_col=0)
        mut_data = pd.read_csv(mutation_data_path, index_col=0)

        research_log += f"- Loaded expression data: {expr_data.shape[0]} genes × {expr_data.shape[1]} samples\n"
        research_log += f"- Loaded mutation data: {mut_data.shape[0]} genes × {mut_data.shape[1]} samples\n"

        # Define core DDR genes (based on literature)
        ddr_genes = [
            # DNA damage sensors
            "ATM",
            "ATR",
            "PRKDC",
            "RAD50",
            "MRE11",
            "NBN",
            # Signal transducers
            "CHEK1",
            "CHEK2",
            "TP53",
            "BRCA1",
            "BRCA2",
            "MDC1",
            "H2AX",
            # Effectors - Homologous Recombination
            "RAD51",
            "RAD52",
            "PALB2",
            "RAD54L",
            # Effectors - Non-Homologous End Joining
            "XRCC4",
            "LIG4",
            "XRCC5",
            "XRCC6",
            # Effectors - Base Excision Repair
            "PARP1",
            "APEX1",
            "OGG1",
            "XRCC1",
            # Effectors - Nucleotide Excision Repair
            "XPA",
            "XPC",
            "ERCC1",
            "ERCC2",
            "ERCC3",
            "ERCC4",
            "ERCC5",
            # Effectors - Mismatch Repair
            "MLH1",
            "MSH2",
            "MSH6",
            "PMS2",
        ]

        # Filter expression and mutation data for DDR genes
        ddr_expr = expr_data.loc[expr_data.index.isin(ddr_genes)]
        ddr_mut = mut_data.loc[mut_data.index.isin(ddr_genes)]

        research_log += f"- Filtered data for {len(ddr_genes)} DDR pathway genes\n"
        research_log += f"- Found {ddr_expr.shape[0]} DDR genes in expression data\n"
        research_log += f"- Found {ddr_mut.shape[0]} DDR genes in mutation data\n\n"

    except Exception as e:
        research_log += f"Error in data loading: {str(e)}\n\n"
        return research_log

    # Step 2: Network reconstruction
    research_log += "STEP 2: Reconstructing DDR gene network\n"

    # Create correlation-based network
    G = nx.Graph()

    # Add nodes (genes)
    for gene in ddr_expr.index:
        # Add node with mutation frequency information
        mutation_freq = ddr_mut.loc[gene].mean() if gene in ddr_mut.index else 0

        G.add_node(gene, mutation_freq=mutation_freq)

    # Add edges based on gene expression correlation
    for i, gene1 in enumerate(ddr_expr.index):
        for gene2 in ddr_expr.index[i + 1 :]:
            corr, p_value = pearsonr(ddr_expr.loc[gene1], ddr_expr.loc[gene2])
            if abs(corr) > 0.4 and p_value < 0.05:  # Significant correlation threshold
                G.add_edge(gene1, gene2, weight=abs(corr), correlation=corr)

    research_log += f"- Constructed network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges\n"

    # Save network for visualization
    network_file = os.path.join(output_dir, "ddr_network.graphml")
    nx.write_graphml(G, network_file)
    research_log += f"- Network saved to: {network_file}\n\n"

    # Step 3: Network analysis
    research_log += "STEP 3: Analyzing DDR network properties\n"

    # Calculate network centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Identify hub genes (high degree centrality)
    hub_genes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

    research_log += "- Top 5 hub genes (highest connectivity):\n"
    for gene, centrality in hub_genes:
        research_log += f"  * {gene}: {centrality:.4f}\n"

    # Identify bottleneck genes (high betweenness centrality)
    bottleneck_genes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

    research_log += "- Top 5 bottleneck genes (critical for information flow):\n"
    for gene, centrality in bottleneck_genes:
        research_log += f"  * {gene}: {centrality:.4f}\n"

    # Identify frequently mutated DDR genes
    mutation_freq = {node: G.nodes[node]["mutation_freq"] for node in G.nodes()}
    frequently_mutated = sorted(mutation_freq.items(), key=lambda x: x[1], reverse=True)[:5]

    research_log += "- Top 5 frequently mutated DDR genes:\n"
    for gene, freq in frequently_mutated:
        research_log += f"  * {gene}: {freq:.4f}\n\n"

    # Step 4: Community detection to identify DDR sub-pathways
    research_log += "STEP 4: Identifying DDR sub-pathways through community detection\n"

    # Use Louvain method for community detection
    try:
        from community import best_partition

        partition = best_partition(G)
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)

        research_log += f"- Identified {len(communities)} DDR sub-pathways\n"

        for i, (_, genes) in enumerate(communities.items()):
            research_log += f"  * Sub-pathway {i + 1}: {', '.join(genes)}\n"

    except ImportError:
        research_log += "- Community detection skipped (python-louvain package not installed)\n"

    # Step 5: Identify disrupted DDR pathways using GSEA
    research_log += "\nSTEP 5: Pathway enrichment analysis of DDR genes\n"

    try:
        # Create gene list for GSEA
        gene_list = ddr_expr.mean(axis=1).sort_values(ascending=False)

        # Run GSEA with GO Biological Process
        enrichr_results = gp.enrichr(
            gene_list=gene_list.index.tolist(),
            gene_sets=["GO_Biological_Process_2021"],
            outdir=os.path.join(output_dir, "enrichr_output"),
            cutoff=0.5,
        )

        # Filter for DNA repair related terms
        ddr_terms = [
            term
            for term in enrichr_results.results["Term"]
            if any(x in term.lower() for x in ["dna repair", "damage", "recombination", "checkpoint"])
        ]

        if ddr_terms:
            research_log += "- Enriched DDR-related pathways:\n"
            for term in ddr_terms[:5]:  # Top 5 DDR-related terms
                research_log += f"  * {term}\n"
        else:
            research_log += "- No significant DDR pathway enrichment found\n"

        # Save full enrichment results
        enrichment_file = os.path.join(output_dir, "ddr_pathway_enrichment.csv")
        enrichr_results.results.to_csv(enrichment_file)
        research_log += f"- Full enrichment results saved to: {enrichment_file}\n\n"

    except Exception as e:
        research_log += f"- Pathway enrichment analysis error: {str(e)}\n\n"

    # Step 6: Summary of findings
    research_log += "SUMMARY OF FINDINGS\n"
    research_log += "=" * 80 + "\n"

    # Key hub genes (potential therapeutic targets)
    research_log += "1. Key DDR hub genes that may serve as potential therapeutic targets:\n"
    for gene, _ in hub_genes[:3]:
        research_log += f"   - {gene}\n"

    # Frequently mutated genes
    research_log += "\n2. Frequently mutated DDR genes in the samples:\n"
    for gene, freq in frequently_mutated[:3]:
        research_log += f"   - {gene} (mutation frequency: {freq:.2%})\n"

    # Network structure insights
    research_log += "\n3. DDR network structure insights:\n"
    research_log += f"   - Network density: {nx.density(G):.4f}\n"

    try:
        avg_clustering = nx.average_clustering(G)
        research_log += f"   - Average clustering coefficient: {avg_clustering:.4f}\n"
    except Exception:
        pass

    research_log += "\n4. Potential DDR dependencies and synthetic lethality targets:\n"
    # Identify potential synthetic lethality pairs (genes with high correlation and one is frequently mutated)
    synthetic_lethality_candidates = []
    for gene1, gene2, data in G.edges(data=True):
        if abs(data["correlation"]) > 0.6:  # Strong correlation
            mut_freq1 = G.nodes[gene1]["mutation_freq"]
            mut_freq2 = G.nodes[gene2]["mutation_freq"]
            if (mut_freq1 > 0.1 and mut_freq2 < 0.05) or (mut_freq2 > 0.1 and mut_freq1 < 0.05):
                if mut_freq1 > mut_freq2:
                    synthetic_lethality_candidates.append((gene1, gene2, data["correlation"], mut_freq1))
                else:
                    synthetic_lethality_candidates.append((gene2, gene1, data["correlation"], mut_freq2))

    if synthetic_lethality_candidates:
        for mutated, target, _, _ in sorted(synthetic_lethality_candidates, key=lambda x: x[3], reverse=True)[:3]:
            research_log += f"   - {mutated} (frequently mutated) and {target} (potential dependency)\n"
    else:
        research_log += "   - No strong synthetic lethality candidates identified\n"

    return research_log


def analyze_cell_senescence_and_apoptosis(fcs_file_path):
    """Analyze flow cytometry data to quantify senescent and apoptotic cell populations.

    Parameters
    ----------
    fcs_file_path : str
        Path to the FCS file containing flow cytometry data with measurements for
        senescence-associated β-galactosidase (SA-β-Gal) and Annexin V/7-AAD staining

    Returns
    -------
    str
        A research log summarizing the analysis steps and results, including percentages
        of senescent and apoptotic cell populations

    """
    import os

    import numpy as np

    log = "# Flow Cytometry Analysis of Cell Senescence and Apoptosis\n\n"

    # FlowCytometryTools 0.5.1 fails at import on Python >= 3.10 (collections.MutableMapping) and on
    # numpy 2 even where it is installed, and the AttributeError came out as a bare traceback
    # (hunt 2026-09-30, uT8-imaging-9). The failure is reported, naming the package.
    try:
        from FlowCytometryTools import FCMeasurement
    except Exception as exc:
        return log + (
            "Error: this tool reads FCS files with the 'FlowCytometryTools' package, which cannot be imported "
            f"here ({type(exc).__name__}: {exc}). Nothing was analysed.\n"
        )

    try:
        # Step 1: Load the FCS file
        log += "## Step 1: Loading Flow Cytometry Data\n"
        log += f"- Loading FCS file from: {fcs_file_path}\n"

        sample = FCMeasurement(ID="Sample", datafile=fcs_file_path)
        log += f"- Successfully loaded data with {len(sample)} events\n"
        log += f"- Available channels: {', '.join(sample.channel_names)}\n\n"

        # Step 2: Apply compensation if needed (assuming pre-compensated data)
        log += "## Step 2: Data Preprocessing\n"
        log += "- Checking for outliers and debris based on forward/side scatter\n"

        # Basic filtering to remove debris based on FSC and SSC
        # Assuming FSC-A and SSC-A are the channel names (adjust if different)
        fsc_channel = "FSC-A" if "FSC-A" in sample.channel_names else sample.channel_names[0]
        ssc_channel = "SSC-A" if "SSC-A" in sample.channel_names else sample.channel_names[1]

        # Filter out debris (low FSC and SSC)
        sample_filtered = sample.gate(f"{fsc_channel} > 10000 and {ssc_channel} > 5000")

        log += f"- Filtered out debris: {len(sample)} → {len(sample_filtered)} events ({len(sample_filtered) / len(sample) * 100:.1f}%)\n\n"

        # Step 3: Identify senescent cells (SA-β-Gal positive)
        log += "## Step 3: Identifying Senescent Cells (SA-β-Gal+)\n"

        # Find SA-β-Gal channel (adjust based on actual channel name)
        sa_bgal_channel = None
        for channel in sample_filtered.channel_names:
            if "GAL" in channel.upper() or "FITC" in channel.upper():
                sa_bgal_channel = channel
                break

        if not sa_bgal_channel:
            log += "- WARNING: Could not identify SA-β-Gal channel. Using first fluorescence channel as placeholder.\n"
            # Use the first fluorescence channel as a fallback
            for channel in sample_filtered.channel_names:
                if any(x in channel.upper() for x in ["FL", "BL", "FITC", "PE", "APC"]):
                    sa_bgal_channel = channel
                    break

        log += f"- Using {sa_bgal_channel} as SA-β-Gal activity indicator\n"

        # Determine threshold for SA-β-Gal positivity (using a simple percentile approach)
        # In practice, this would be based on controls or known thresholds
        sa_bgal_threshold = np.percentile(sample_filtered.data[sa_bgal_channel], 80)
        senescent_cells = sample_filtered.gate(f"{sa_bgal_channel} > {sa_bgal_threshold}")

        senescent_percentage = (len(senescent_cells) / len(sample_filtered)) * 100
        log += f"- Applied threshold at {sa_bgal_threshold:.1f} fluorescence intensity\n"
        log += f"- Identified {len(senescent_cells)} senescent cells ({senescent_percentage:.2f}%)\n\n"

        # Step 4: Identify apoptotic cells (Annexin V+/7-AAD+)
        log += "## Step 4: Identifying Apoptotic Cells (Annexin V/7-AAD)\n"

        # Find Annexin V and 7-AAD channels (adjust based on actual channel names)
        annexin_channel = None
        aad_channel = None

        for channel in sample_filtered.channel_names:
            if "ANNEXIN" in channel.upper() or "PE" in channel.upper():
                annexin_channel = channel
            if "7AAD" in channel.upper() or "AAD" in channel.upper() or "PerCP" in channel.upper():
                aad_channel = channel

        if not annexin_channel or not aad_channel:
            log += "- WARNING: Could not identify Annexin V and/or 7-AAD channels. Using placeholder channels.\n"
            # Use fallback channels
            fluorescence_channels = [
                ch
                for ch in sample_filtered.channel_names
                if any(x in ch.upper() for x in ["FL", "BL", "FITC", "PE", "APC", "PerCP"])
            ]
            if len(fluorescence_channels) >= 2:
                if not annexin_channel:
                    annexin_channel = fluorescence_channels[0]
                if not aad_channel:
                    aad_channel = fluorescence_channels[1]

        log += f"- Using {annexin_channel} as Annexin V indicator\n"
        log += f"- Using {aad_channel} as 7-AAD indicator\n"

        # Determine thresholds for Annexin V and 7-AAD (using simple percentile approach)
        annexin_threshold = np.percentile(sample_filtered.data[annexin_channel], 90)
        aad_threshold = np.percentile(sample_filtered.data[aad_channel], 90)

        # Early apoptotic: Annexin V+ / 7-AAD-
        early_apoptotic = sample_filtered.gate(
            f"{annexin_channel} > {annexin_threshold} and {aad_channel} < {aad_threshold}"
        )
        early_apoptotic_percentage = (len(early_apoptotic) / len(sample_filtered)) * 100

        # Late apoptotic/necrotic: Annexin V+ / 7-AAD+
        late_apoptotic = sample_filtered.gate(
            f"{annexin_channel} > {annexin_threshold} and {aad_channel} > {aad_threshold}"
        )
        late_apoptotic_percentage = (len(late_apoptotic) / len(sample_filtered)) * 100

        # Total apoptotic (early + late)
        total_apoptotic_percentage = early_apoptotic_percentage + late_apoptotic_percentage

        log += f"- Early apoptotic cells (Annexin V+/7-AAD-): {early_apoptotic_percentage:.2f}%\n"
        log += f"- Late apoptotic cells (Annexin V+/7-AAD+): {late_apoptotic_percentage:.2f}%\n"
        log += f"- Total apoptotic cells: {total_apoptotic_percentage:.2f}%\n\n"

        # Step 5: Summary of results
        log += "## Summary of Results\n"
        log += f"- Total events analyzed: {len(sample_filtered)}\n"
        log += f"- Senescent cells (SA-β-Gal+): {senescent_percentage:.2f}%\n"
        log += f"- Early apoptotic cells (Annexin V+/7-AAD-): {early_apoptotic_percentage:.2f}%\n"
        log += f"- Late apoptotic cells (Annexin V+/7-AAD+): {late_apoptotic_percentage:.2f}%\n"
        log += f"- Total apoptotic cells: {total_apoptotic_percentage:.2f}%\n"

        # Save results to CSV file
        results_file = os.path.splitext(os.path.basename(fcs_file_path))[0] + "_results.csv"
        with open(results_file, "w") as f:
            f.write("Cell Population,Percentage\n")
            f.write(f"Senescent cells,{senescent_percentage:.2f}\n")
            f.write(f"Early apoptotic cells,{early_apoptotic_percentage:.2f}\n")
            f.write(f"Late apoptotic cells,{late_apoptotic_percentage:.2f}\n")
            f.write(f"Total apoptotic cells,{total_apoptotic_percentage:.2f}\n")

        log += f"\nResults saved to: {results_file}\n"

    except Exception as e:
        log += "\n## ERROR: An error occurred during analysis\n"
        log += f"- Error message: {str(e)}\n"

    return log


def detect_and_annotate_somatic_mutations(
    tumor_bam, normal_bam, reference_genome, output_prefix, snpeff_database="GRCh38.105"
):
    """Detects and annotates somatic mutations in tumor samples compared to matched normal samples.

    This function uses GATK Mutect2 for variant calling, GATK FilterMutectCalls for filtering,
    and SnpEff for functional annotation of somatic mutations.

    Parameters
    ----------
    tumor_bam : str
        Path to the tumor sample BAM file
    normal_bam : str
        Path to the matched normal sample BAM file
    reference_genome : str
        Path to the reference genome FASTA file
    output_prefix : str
        Prefix for output files
    snpeff_database : str, optional
        SnpEff database to use for annotation (default: "GRCh38.105")

    Returns
    -------
    str
        A research log summarizing the steps performed and results obtained

    """
    import datetime
    import os
    import subprocess

    # Initialize research log
    log = "# Somatic Mutation Analysis Log\n"
    log += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    log += "## Input Files\n"
    log += f"- Tumor BAM: {tumor_bam}\n"
    log += f"- Normal BAM: {normal_bam}\n"
    log += f"- Reference Genome: {reference_genome}\n\n"

    # Step 1: Run Mutect2 for somatic variant calling
    log += "## Step 1: Somatic Variant Calling with Mutect2\n"
    raw_vcf = f"{output_prefix}.unfiltered.vcf"

    # -normal is the normal sample's read-group SM, not its file name: a BAM called
    # patient1_normal.sorted.bam with SM "NORMAL" made Mutect2 refuse, and capture_output plus
    # str(e) dropped GATK's own message saying so (hunt 2026-09-30, uT8-imaging-13). GATK reads the
    # name from the header itself (GetSampleName), and every failure below carries GATK's stderr.
    sample_file = f"{output_prefix}.normal_sample_name.txt"
    sample_cmd = ["gatk", "GetSampleName", "-I", normal_bam, "-O", sample_file]
    log += f"Reading the normal sample name: {' '.join(sample_cmd)}\n"
    try:
        subprocess.run(sample_cmd, check=True, capture_output=True)
        with open(sample_file) as handle:
            normal_sample = handle.read().strip()
    except (subprocess.CalledProcessError, OSError) as e:
        log += f"Error reading the normal sample name (read-group SM) from {normal_bam}: {e}{_stderr_tail(e)}\n"
        return log
    if not normal_sample:
        log += f"Error: {normal_bam} has no read-group sample name (SM); Mutect2's -normal needs one.\n"
        return log
    log += f"Normal sample (read-group SM): {normal_sample}\n"

    mutect2_cmd = [
        "gatk",
        "Mutect2",
        "-R",
        reference_genome,
        "-I",
        tumor_bam,
        "-I",
        normal_bam,
        "-normal",
        normal_sample,
        "-O",
        raw_vcf,
    ]

    log += f"Running command: {' '.join(mutect2_cmd)}\n"
    try:
        subprocess.run(mutect2_cmd, check=True, capture_output=True)
        log += f"Successfully generated raw VCF: {raw_vcf}\n\n"
    except (subprocess.CalledProcessError, OSError) as e:
        log += f"Error running Mutect2: {str(e)}{_stderr_tail(e)}\n"
        return log

    # Step 2: Filter somatic calls
    log += "## Step 2: Filtering Somatic Variants\n"
    filtered_vcf = f"{output_prefix}.filtered.vcf"

    filter_cmd = [
        "gatk",
        "FilterMutectCalls",
        "-R",
        reference_genome,
        "-V",
        raw_vcf,
        "-O",
        filtered_vcf,
    ]

    log += f"Running command: {' '.join(filter_cmd)}\n"
    try:
        subprocess.run(filter_cmd, check=True, capture_output=True)
        log += f"Successfully filtered variants: {filtered_vcf}\n\n"
    except (subprocess.CalledProcessError, OSError) as e:
        log += f"Error filtering variants: {str(e)}{_stderr_tail(e)}\n"
        return log

    # Step 3: Annotate variants with SnpEff
    log += "## Step 3: Functional Annotation with SnpEff\n"
    annotated_vcf = f"{output_prefix}.annotated.vcf"

    snpeff_cmd = ["snpEff", "-v", snpeff_database, filtered_vcf]

    log += f"Running command: {' '.join(snpeff_cmd)} > {annotated_vcf}\n"
    # stdout written to the file directly (no shell, so paths with spaces or quotes are safe),
    # through a .partial so a failed run leaves no half-written VCF under the final name.
    partial = f"{annotated_vcf}.partial"
    try:
        with open(partial, "wb") as out:
            subprocess.run(snpeff_cmd, stdout=out, stderr=subprocess.PIPE, check=True)
        os.replace(partial, annotated_vcf)
        log += f"Successfully annotated variants: {annotated_vcf}\n\n"
    except (subprocess.CalledProcessError, OSError) as e:
        if os.path.exists(partial):
            os.remove(partial)
        log += f"Error annotating variants: {str(e)}{_stderr_tail(e)}\n"
        return log

    # Step 4: Generate summary statistics
    log += "## Step 4: Generating Summary Statistics\n"
    summary_file = f"{output_prefix}_mutation_summary.txt"

    # FilterMutectCalls marks calls in the FILTER column rather than removing them, so every record
    # was counted as a somatic variant; and the type counts grepped "SNP"/"INS"/"DEL" anywhere in the
    # line, words neither Mutect2 nor SnpEff writes as a type (hunt 2026-09-30, uT8-imaging-13). Only
    # PASS records count, typed from REF and ALT.
    counts = _count_pass_variants(annotated_vcf)
    total_variants = counts["PASS"]
    log += f"Records in the VCF: {counts['records']} ({counts['records'] - total_variants} failed a filter)\n"
    log += f"Total somatic variants detected (FILTER = PASS): {total_variants}\n"

    # Count by variant type
    log += "Variant types:\n"
    for variant_type in ["SNP", "MNP", "INS", "DEL", "MULTIALLELIC"]:
        log += f"- {variant_type}: {counts[variant_type]}\n"

    # Count high impact variants
    log += f"High impact variants (SnpEff ANN impact HIGH, PASS only): {counts['HIGH']}\n\n"

    # Step 5: Save summary to file
    with open(summary_file, "w") as f:
        f.write("Somatic Mutation Analysis Summary\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total somatic variants: {total_variants}\n")
        f.write(f"Output VCF file: {annotated_vcf}\n")

    log += "## Results\n"
    log += "Analysis complete. Results saved to:\n"
    log += f"- Annotated VCF: {annotated_vcf}\n"
    log += f"- Summary file: {summary_file}\n\n"

    return log


def detect_and_characterize_structural_variations(
    bam_file_path,
    reference_genome_path,
    output_dir,
    cosmic_db_path=None,
    clinvar_db_path=None,
):
    """Detects and characterizes structural variations (SVs) in genomic sequencing data.

    This function uses LUMPY for SV detection followed by annotation with COSMIC and/or ClinVar
    databases to identify and characterize various types of structural variations including
    deletions, inversions, translocations, and duplications.

    Parameters
    ----------
    bam_file_path : str
        Path to the aligned sequencing data in BAM format
    reference_genome_path : str
        Path to the reference genome in FASTA format
    output_dir : str
        Directory where results will be saved
    cosmic_db_path : str, optional
        Path to the COSMIC database for cancer annotation
    clinvar_db_path : str, optional
        Path to the ClinVar database for clinical annotation

    Returns
    -------
    str
        A research log summarizing the steps performed and results obtained

    """
    import datetime
    import os
    import subprocess

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize research log
    log = []
    log.append("## Structural Variation Detection and Characterization")
    log.append(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append(f"Input BAM file: {bam_file_path}")
    log.append(f"Reference genome: {reference_genome_path}")
    log.append(f"Output directory: {output_dir}")
    log.append("\n")

    # Step 1: Extract discordant read-pairs and split-reads for LUMPY
    log.append("### Step 1: Extracting discordant read-pairs and split-reads")

    discordant_bam = os.path.join(output_dir, "discordant.bam")
    split_bam = os.path.join(output_dir, "split.bam")

    try:
        # Extract discordant read pairs
        subprocess.run(
            [
                "samtools",
                "view",
                "-b",
                "-F",
                "1294",
                "-o",
                discordant_bam,
                bam_file_path,
            ],
            check=True,
        )

        # Extract split reads: samtools view -h | extractSplitReads_BwaMem -i stdin | samtools view -Sb.
        # A list with shell=True runs only its first item -- bare ``samtools``, which prints usage and
        # fails -- and hands the rest to sh as positional parameters, so this step could never succeed
        # (hunt 2026-09-30, uT8-imaging-10). The pipeline is chained as processes, no shell.
        _run_pipeline(
            [
                ["samtools", "view", "-h", bam_file_path],
                ["extractSplitReads_BwaMem", "-i", "stdin"],
                ["samtools", "view", "-Sb", "-o", split_bam, "-"],
            ]
        )

        log.append("Successfully extracted discordant read-pairs and split-reads")
    except subprocess.CalledProcessError as e:
        log.append(f"Error during read extraction: {e}{_stderr_tail(e)}")
        return "\n".join(log)
    except OSError as e:
        log.append(f"Error during read extraction: {e} (samtools and LUMPY's extractSplitReads_BwaMem must be on PATH)")
        return "\n".join(log)

    # Step 2: Run LUMPY for SV detection
    log.append("\n### Step 2: Running LUMPY for structural variation detection")

    vcf_output = os.path.join(output_dir, "structural_variants.vcf")

    try:
        subprocess.run(
            [
                "lumpyexpress",
                "-B",
                bam_file_path,
                "-S",
                split_bam,
                "-D",
                discordant_bam,
                "-o",
                vcf_output,
            ],
            check=True,
        )

        log.append("LUMPY analysis completed successfully")
        log.append(f"Raw SV calls saved to: {vcf_output}")
    except subprocess.CalledProcessError as e:
        log.append(f"Error during LUMPY execution: {e}")
        return "\n".join(log)

    # Step 3: Filter SVs by quality and size
    log.append("\n### Step 3: Filtering structural variants by quality and size")

    filtered_vcf = os.path.join(output_dir, "filtered_structural_variants.vcf")

    try:
        # Filter SVs with quality score >= 100 and size >= 100bp. LUMPY writes a NEGATIVE SVLEN for a
        # deletion and none for a breakend, so "SVLEN>=100" dropped every DEL and every BND (hunt
        # 2026-09-30, uT8-imaging-11): size is |SVLEN|, and a BND has no size to filter on.
        subprocess.run(
            [
                "bcftools",
                "filter",
                "-i",
                'QUAL>=100 && (INFO/SVTYPE=="BND" || abs(INFO/SVLEN)>=100)',
                "-o",
                filtered_vcf,
                vcf_output,
            ],
            check=True,
        )

        # Count SVs by type
        sv_counts = {}
        sv_types = ["DEL", "DUP", "INV", "BND", "INS"]

        for sv_type in sv_types:
            result = subprocess.run(
                ["grep", "-c", f"SVTYPE={sv_type}", filtered_vcf],
                check=False,
                capture_output=True,
                text=True,
            )

            count = 0
            if result.returncode == 0:
                count = int(result.stdout.strip())
            sv_counts[sv_type] = count

        log.append("SV filtering completed")
        log.append(f"Filtered SVs saved to: {filtered_vcf}")
        log.append("\nSV counts by type:")
        for sv_type, count in sv_counts.items():
            log.append(f"- {sv_type}: {count}")
    except subprocess.CalledProcessError as e:
        log.append(f"Error during SV filtering: {e}")
        return "\n".join(log)

    # Step 4: Annotate SVs with COSMIC and ClinVar (if databases provided)
    log.append("\n### Step 4: Annotating structural variants")

    annotated_vcf = os.path.join(output_dir, "annotated_structural_variants.vcf")

    try:
        # No annotator is wired in: the file below is the filtered VCF unchanged. "Using COSMIC
        # database" was logged for a database nothing read (hunt 2026-09-30, uT8-imaging-11).
        for label, db_path in (("COSMIC", cosmic_db_path), ("ClinVar", clinvar_db_path)):
            if db_path:
                log.append(f"NOT USED: {label} database {db_path} -- this tool has no SV annotator to read it")
        log.append("No annotation was performed: the 'annotated' VCF is a copy of the filtered VCF.")
        log.append("Annotate it with AnnotSV or VEP to add COSMIC/ClinVar information.")

        # Instead of running the command, we'll create a simple annotated file
        with open(filtered_vcf) as infile, open(annotated_vcf, "w") as outfile:
            for line in infile:
                if line.startswith("#"):
                    outfile.write(line)
                else:
                    outfile.write(line)
                    # In a real implementation, annotation would be added here

        log.append(f"Unannotated copy saved to: {annotated_vcf}")
    except Exception as e:
        log.append(f"Error during SV annotation: {e}")
        return "\n".join(log)

    # Step 5: Generate summary report
    log.append("\n### Step 5: Generating summary report")

    summary_file = os.path.join(output_dir, "sv_summary_report.tsv")

    try:
        # Convert VCF to a tabular format for easier analysis
        subprocess.run(
            [
                "bcftools",
                "query",
                "-f",
                "%CHROM\t%POS\t%INFO/SVTYPE\t%INFO/SVLEN\t%QUAL\n",
                "-o",
                summary_file,
                annotated_vcf,
            ],
            check=True,
        )

        log.append(f"Summary report generated: {summary_file}")
    except subprocess.CalledProcessError as e:
        log.append(f"Error generating summary report: {e}")
        return "\n".join(log)

    # Final summary
    log.append("\n## Summary")
    total_svs = sum(sv_counts.values())
    log.append(f"Total SVs detected: {total_svs}")
    log.append("SV types breakdown:")
    for sv_type, count in sv_counts.items():
        # No SV passing the filter divided by zero here, after all the work, and lost the log
        # (hunt 2026-09-30, uT8-imaging-11).
        share = f" ({count / total_svs * 100:.1f}%)" if total_svs else ""
        log.append(f"- {sv_type}: {count}{share}")

    log.append("\nFiles generated:")
    log.append(f"- Raw SV calls: {vcf_output}")
    log.append(f"- Filtered SVs: {filtered_vcf}")
    log.append(f"- Annotated SVs: {annotated_vcf}")
    log.append(f"- Summary report: {summary_file}")

    log.append(f"\nAnalysis completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(log)


def perform_gene_expression_nmf_analysis(
    expression_data_path,
    n_components=10,
    normalize=True,
    output_dir="nmf_results",
    random_state=42,
):
    """Performs Non-negative Matrix Factorization (NMF) on gene expression data to extract
    metagenes and their associated sample weights for tumor subtype identification.

    Parameters
    ----------
    expression_data_path : str
        Path to a CSV or TSV file containing gene expression data with genes as rows and samples as columns.
        Values should be non-negative (e.g., normalized counts or expression values).
    n_components : int, default=10
        Number of metagenes (components) to extract.
    normalize : bool, default=True
        Whether to normalize the expression data before applying NMF.
    output_dir : str, default="nmf_results"
        Directory to save the output files.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    str
        Research log summarizing the analysis steps and results.

    """
    import os
    from datetime import datetime

    import numpy as np
    import pandas as pd
    from sklearn.decomposition import NMF

    # Start research log
    log = []
    log.append(f"NMF GENE EXPRESSION ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append(f"Parameters: n_components={n_components}, normalize={normalize}")

    # Load expression data from file
    try:
        if expression_data_path.endswith(".csv"):
            expression_data = pd.read_csv(expression_data_path, index_col=0)
        elif expression_data_path.endswith((".tsv", ".txt")):
            expression_data = pd.read_csv(expression_data_path, sep="\t", index_col=0)
        else:
            log.append("ERROR: Unsupported file format. Please provide a CSV or TSV file.")
            return "\n".join(log)

        log.append(f"Successfully loaded expression data from {expression_data_path}")
    except Exception as e:
        log.append(f"ERROR: Failed to load expression data from {expression_data_path}. Error: {str(e)}")
        return "\n".join(log)

    # Extract gene and sample information
    genes = expression_data.index.tolist()
    samples = expression_data.columns.tolist()
    X = expression_data.values
    log.append(f"Input data: {len(genes)} genes × {len(samples)} samples")

    # Check for non-negative values
    if np.any(X < 0):
        log.append("WARNING: Negative values found in expression data. Converting to absolute values.")
        X = np.abs(X)

    # Normalize data if requested
    if normalize:
        log.append("Normalizing expression data...")
        X = X / np.sum(X, axis=0, keepdims=True) * 1000  # TPM-like normalization

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log.append(f"Created output directory: {output_dir}")

    # Apply NMF
    log.append(f"Applying NMF to extract {n_components} metagenes...")
    model = NMF(
        n_components=n_components,
        init="random",
        random_state=random_state,
        max_iter=1000,
    )

    try:
        # W: metagenes (genes × components)
        # H: sample weights (components × samples)
        W = model.fit_transform(X)
        H = model.components_

        log.append(f"NMF completed successfully in {model.n_iter_} iterations")
        # reconstruction_err_ is ||X - WH|| (Frobenius; lower is better, unbounded), and was printed as
        # "Explained variance" -- read as a fraction when choosing n_components (hunt 2026-09-30,
        # uT8-imaging-26). Both are reported, each under its own name.
        log.append(f"Reconstruction error (Frobenius norm ||X - WH||): {model.reconstruction_err_:.4f}")
        centred_ss = float(np.sum((X - X.mean()) ** 2))
        if centred_ss > 0:
            log.append(
                f"Explained variance (1 - ||X - WH||^2 / ||X - mean(X)||^2): {1 - model.reconstruction_err_**2 / centred_ss:.4f}"
            )

        # Save metagenes (W matrix)
        metagenes_file = os.path.join(output_dir, "metagenes.csv")
        pd.DataFrame(W, index=genes, columns=[f"Metagene_{i + 1}" for i in range(n_components)]).to_csv(metagenes_file)
        log.append(f"Saved metagenes to {metagenes_file}")

        # Save sample weights (H matrix)
        weights_file = os.path.join(output_dir, "sample_weights.csv")
        pd.DataFrame(H, index=[f"Metagene_{i + 1}" for i in range(n_components)], columns=samples).to_csv(weights_file)
        log.append(f"Saved sample weights to {weights_file}")

        top_genes_file = os.path.join(output_dir, "top_genes_per_metagene.csv")
        top_genes = {}
        for i in range(n_components):
            # Get indices of top 20 genes for this metagene
            top_indices = np.argsort(W[:, i])[::-1][:20]
            top_genes[f"Metagene_{i + 1}"] = [genes[idx] for idx in top_indices]

        pd.DataFrame(top_genes).to_csv(top_genes_file)
        log.append(f"Saved top genes per metagene to {top_genes_file}")

    except Exception as e:
        log.append(f"ERROR: NMF failed with error: {str(e)}")
        return "\n".join(log)

    log.append("\nSUMMARY:")
    log.append(f"- Successfully extracted {n_components} metagenes from {X.shape[0]} genes across {X.shape[1]} samples")
    log.append(f"- Results saved to: {output_dir}")
    log.append("- Files generated: metagenes.csv, sample_weights.csv, top_genes_per_metagene.csv")
    log.append("\nNEXT STEPS:")
    log.append("- Analyze the metagenes to identify biological pathways")
    log.append("- Cluster samples based on metagene weights to identify potential tumor subtypes")
    log.append("- Correlate subtypes with clinical outcomes if available")

    return "\n".join(log)


def analyze_copy_number_purity_ploidy_and_focal_events(
    tumor_bam,
    reference_genome,
    normal_bam=None,
    output_dir="cn_analysis_results",
    targets_bed=None,
    antitargets_bed=None,
    gene_bed=None,
    focal_genes=None,
    log2_amp_threshold=1.0,
    log2_del_threshold=-1.0,
):
    """CNVkit-based copy number workflow: CNV segmentation, purity/ploidy & HRD approximation, focal events.

    This function orchestrates a CNVkit-based copy number analysis workflow for tumor samples.
    It performs CNV segmentation, derives approximate purity & ploidy metrics, calculates a simplified
    HRD-style summary, and detects focal amplifications/deletions in key oncogenes / tumor suppressors.

    Parameters
    ----------
    tumor_bam : str
        Path to tumor BAM (indexed)
    reference_genome : str
        Path to reference FASTA (with index files present)
    normal_bam : str, optional
        Matched normal BAM (recommended for allele-specific & purity inference)
    output_dir : str, default "cn_analysis_results"
        Directory for all outputs
    targets_bed : str, optional
        BED of target regions (e.g. exome / panel) for CNVkit / gCNV
    antitargets_bed : str, optional
        BED of antitarget regions for CNVkit (if panel / exome)
    gene_bed : str, optional
        BED file with gene coordinates (chrom, start, end, gene). Used for focal event annotation
    focal_genes : list[str], optional
        List of genes to highlight for focal amplification/deletion; defaults to ["MYC","ERBB2","CDKN2A"]
    log2_amp_threshold : float, default 1.0
        Log2 ratio threshold to call focal amplification (~ >2x copy relative to baseline)
    log2_del_threshold : float, default -1.0
        Log2 ratio threshold to call focal deep deletion (~ homozygous loss)

    Returns
    -------
    str
        Research log summarizing steps, tool executions, and findings.
    """
    import datetime
    import math
    import os
    import shutil
    import statistics
    import subprocess

    import pandas as pd

    log = []
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.append(f"CNVKIT COPY NUMBER ANALYSIS WORKFLOW - {ts}")
    log.append("=" * 80)
    log.append(f"Tumor BAM: {tumor_bam}")
    if normal_bam:
        log.append(f"Normal BAM: {normal_bam}")
    log.append(f"Reference genome: {reference_genome}")
    log.append(f"Output directory: {output_dir}\n")

    os.makedirs(output_dir, exist_ok=True)
    sample_name = os.path.splitext(os.path.basename(tumor_bam))[0]

    if focal_genes is None:
        focal_genes = ["MYC", "ERBB2", "CDKN2A"]
    # Single informative log line (was duplicated previously)
    log.append(f"Focal genes for analysis: {', '.join(focal_genes)}")

    # ----------------------------------------------------------------------------------
    # STEP 1: CNV SEGMENTATION with CNVkit
    # ----------------------------------------------------------------------------------
    log.append("STEP 1: CNV Segmentation with CNVkit")
    cnvkit_path = shutil.which("cnvkit.py") or shutil.which("cnvkit")
    # The env cnvkit.py was found in. Detection could find it in bio_env_py310 while every command
    # then ran in ``conda run -n spatialomicsgym_e1``, where it is not, and the failure was logged as
    # a WARNING (hunt 2026-09-30, uT8-imaging-14). The commands now run where it was found.
    cnvkit_env = None

    if not cnvkit_path:
        for env_name in ("spatialomicsgym_e1", "bio_env_py310"):
            try:
                result = subprocess.run(
                    ["conda", "run", "-n", env_name, "which", "cnvkit.py"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
            if result.returncode == 0 and result.stdout.strip():
                cnvkit_path = "cnvkit.py"  # Will use via conda run
                cnvkit_env = env_name
                log.append(f"- CNVkit found in {env_name} environment")
                break

    if not cnvkit_path:
        log.append("- CNVkit not detected in current PATH, spatialomicsgym_e1, or bio_env_py310 environment")
        log.append("- Install manually (e.g., pip install cnvkit==0.9.11) or run setup.sh")
        log.append("- Workflow will exit without CNVkit")
        return "\n".join(log)

    cnvkit_cns = None
    log.append(f"- CNVkit detected at: {cnvkit_path}")

    # Determine environment prefix for CNVkit commands
    env_prefix = ["conda", "run", "-n", cnvkit_env] if cnvkit_env else []

    # CNVkit batch command
    batch_cmd = env_prefix + [cnvkit_path, "batch", tumor_bam, "-d", output_dir, "-f", reference_genome]
    if normal_bam:
        batch_cmd.extend(["-n", normal_bam])
    if targets_bed:
        batch_cmd.extend(["-t", targets_bed])
    if antitargets_bed:
        batch_cmd.extend(["-a", antitargets_bed])
    batch_cmd.extend(["--scatter", "--diagram"])

    log.append(f"- Running CNVkit batch command: {' '.join(batch_cmd)}")
    try:
        subprocess.run(batch_cmd, check=True, capture_output=True)
        log.append("- CNVkit batch completed successfully")
    except subprocess.CalledProcessError as e:
        log.append(f"- WARNING: CNVkit batch failed: {e}. Proceeding with limited downstream analyses")

    # Check for CNVkit output and run call command
    cnvkit_cns = os.path.join(output_dir, f"{sample_name}.cns")
    if os.path.exists(cnvkit_cns):
        call_cmd = env_prefix + [
            cnvkit_path,
            "call",
            cnvkit_cns,
            "-o",
            os.path.join(output_dir, f"{sample_name}.call.cns"),
            "-m",
            "clonal",
        ]
        log.append(f"- Calling absolute CN: {' '.join(call_cmd)}")
        try:
            subprocess.run(call_cmd, check=True, capture_output=True)
            cnvkit_cns = os.path.join(output_dir, f"{sample_name}.call.cns")
            log.append("- CNVkit call succeeded")
        except subprocess.CalledProcessError as e:
            log.append(f"- WARNING: CNVkit call failed: {e}")
    else:
        log.append("- WARNING: Expected CNVkit .cns file not found")

    # ----------------------------------------------------------------------------------
    # STEP 2: Purity & ploidy approximation (based on CNVkit outputs)
    # ----------------------------------------------------------------------------------
    log.append("\nSTEP 2: Purity & Ploidy Approximation")
    purity_est = None
    ploidy_est = None
    seg_source = None
    seg_df = None
    try:
        if cnvkit_cns and os.path.exists(cnvkit_cns):
            seg_source = cnvkit_cns
            seg_df = pd.read_csv(cnvkit_cns, sep="\t")
            log.append(f"- Using CNVkit segments from: {seg_source}")

            # Harmonize columns
            cols = [c.lower() for c in seg_df.columns]
            seg_df.columns = cols

            # Obtain log2 column
            log2_col = None
            for candidate in ["log2", "log2ratio", "cnlr", "logr"]:
                if candidate in cols:
                    log2_col = candidate
                    break
            for candidate in ["end", "chromend", "stop"]:
                if candidate in cols:
                    end_col = candidate
                    break
            for candidate in ["start", "chromstart", "begin"]:
                if candidate in cols:
                    start_col = candidate
                    break

            if log2_col and start_col and end_col:
                seg_df["seg_length"] = seg_df[end_col] - seg_df[start_col]
                # Weighted average absolute copy ratio -> approximate ploidy baseline
                # Convert log2 ratio to absolute copy ratio (assuming diploid baseline = 2)
                seg_df["abs_cn"] = 2 * (2 ** seg_df[log2_col])
                weighted_mean_cn = (seg_df["abs_cn"] * seg_df["seg_length"]).sum() / seg_df["seg_length"].sum()
                ploidy_est = weighted_mean_cn
                log.append(f"- Estimated ploidy (weighted mean CN): {ploidy_est:.2f}")
                # Purity is not estimated. The heuristic here, 1 - median|log2| / 1.5, fell as the
                # copy-number signal rose -- the opposite of its own rationale -- so a flattened,
                # normal-contaminated profile read as ~100% pure (hunt 2026-09-30, uT8-imaging-15).
                # Total-copy-number log2 ratios cannot separate purity from how much of the genome
                # is altered; that takes allele frequencies.
                median_abs_log2 = statistics.median([abs(x) for x in seg_df[log2_col] if not math.isnan(x)])
                log.append(
                    "- Purity: not estimated -- segment log2 ratios alone cannot separate tumour purity from "
                    "the altered fraction of the genome; use PureCN, FACETS or ABSOLUTE (allele frequencies), "
                    f"or give a known purity to `cnvkit.py call --purity`. Median |log2| = {median_abs_log2:.3f}"
                )
            else:
                log.append("- WARNING: Could not identify necessary columns for purity/ploidy estimation")
        else:
            log.append("- No CNVkit segmentation available for purity/ploidy estimation")
    except Exception as e:
        log.append(f"- ERROR: Purity/ploidy estimation failed: {e}")

    # ----------------------------------------------------------------------------------
    # STEP 3: HRD-like summary (simplified proxy)
    # ----------------------------------------------------------------------------------
    log.append("\nSTEP 3: HRD Approximation (Simplified)")
    hrd_metrics = {}
    try:
        if seg_df is not None and "seg_length" in seg_df:
            # Count large-scale transitions (LST proxy: adjacent segments >10 Mb with log2 change >0.2)
            # Adjacent means neighbours on one chromosome, in position order. The sort was by the
            # chromosome name alone -- lexicographic (chr1, chr10, chr2) and unstable, so segments were
            # scrambled within a chromosome -- and a transition across two chromosomes was counted
            # (hunt 2026-09-30, uT8-imaging-15).
            chrom_col = (
                [c for c in seg_df.columns if c.startswith("chrom") or c == "chrom" or c == "chromosome"][0]
                if any(c.startswith("chrom") for c in seg_df.columns)
                else seg_df.columns[0]
            )
            seg_df_sorted = seg_df.assign(
                _chrom_key=seg_df[chrom_col].map(_natural_chromosome_key), _start_key=seg_df[start_col]
            ).sort_values(by=["_chrom_key", "_start_key"], kind="stable")
            lst_count = 0
            prev = None
            for _, row in seg_df_sorted.iterrows():
                if prev is not None and row[chrom_col] == prev[chrom_col]:
                    if row["seg_length"] >= 10_000_000 and prev["seg_length"] >= 10_000_000:
                        if log2_col and abs(row[log2_col] - prev[log2_col]) > 0.2:
                            lst_count += 1
                prev = row
            hrd_metrics["LST_like"] = lst_count
            # HRD-LOH proxy: segments with log2 < -0.3 and length >15 Mb
            if log2_col:
                hrd_loh = ((seg_df[log2_col] < -0.3) & (seg_df["seg_length"] > 15_000_000)).sum()
                hrd_metrics["HRD_LOH_like"] = int(hrd_loh)
            # Telomeric AI proxy omitted (needs allele-specific data)
            hrd_score = sum(hrd_metrics.values())
            hrd_metrics["Simplified_HRD_score"] = hrd_score
            log.append(f"- LST-like events: {hrd_metrics['LST_like']}")
            log.append(f"- HRD-LOH-like events: {hrd_metrics['HRD_LOH_like']}")
            log.append(f"- Simplified composite HRD score: {hrd_metrics['Simplified_HRD_score']}")
            log.append("- NOTE: For clinical/research use, apply scarHRD or HRDetect with allele-specific data")
        else:
            log.append("- Segmentation not available; HRD approximation skipped")
    except Exception as e:
        log.append(f"- ERROR: HRD approximation failed: {e}")

    # ----------------------------------------------------------------------------------
    # STEP 4: Focal amplifications/deletions in key genes
    # ----------------------------------------------------------------------------------
    log.append("\nSTEP 4: Focal Amplifications / Deletions in Key Genes")
    focal_events = []
    try:
        if seg_df is not None and gene_bed and os.path.exists(gene_bed):
            genes_df = pd.read_csv(gene_bed, sep="\t", header=None, names=["chrom", "start", "end", "gene"])

            # Normalize chromosome naming
            def norm_chr(x):
                return x.replace("chr", "") if isinstance(x, str) else x

            seg_df["chr_norm"] = (
                seg_df[[c for c in seg_df.columns if c in ["chrom", "chromosome", "chr"]][0]].astype(str).map(norm_chr)
            )
            genes_df["chr_norm"] = genes_df["chrom"].astype(str).map(norm_chr)
            for gene in focal_genes:
                region = genes_df[genes_df["gene"] == gene]
                if region.empty:
                    continue
                r = region.iloc[0]
                # Ensure we work on a copy to avoid SettingWithCopyWarning
                overlaps = seg_df[
                    (seg_df["chr_norm"] == r["chr_norm"])
                    & (seg_df[start_col] <= r["end"])
                    & (seg_df[end_col] >= r["start"])
                ].copy()
                if not overlaps.empty and log2_col:
                    # Choose segment with largest overlap length
                    r_end = r["end"]
                    r_start = r["start"]
                    overlaps["ov_len"] = overlaps.apply(
                        lambda row, r_end=r_end, r_start=r_start: (
                            min(row[end_col], r_end) - max(row[start_col], r_start)
                        ),
                        axis=1,
                    )
                    best = overlaps.sort_values("ov_len", ascending=False).iloc[0]
                    l2 = best[log2_col]
                    status = None
                    if l2 >= log2_amp_threshold:
                        status = "AMPLIFICATION"
                    elif l2 <= log2_del_threshold:
                        status = "DELETION"
                    if status:
                        focal_events.append((gene, status, l2))
            if focal_events:
                for gene, status, l2 in focal_events:
                    log.append(f"- {gene}: {status} (log2={l2:.2f})")
            else:
                log.append("- No focal events meeting thresholds in target genes")
        else:
            if not gene_bed:
                log.append("- Skipped: gene_bed not provided")
            else:
                log.append("- Skipped: gene_bed file not found or segmentation unavailable")
    except Exception as e:
        log.append(f"- ERROR: Focal event detection failed: {e}")

    # ----------------------------------------------------------------------------------
    # STEP 5: Summaries & file outputs
    # ----------------------------------------------------------------------------------
    log.append("\nSTEP 5: Summary & Output Files")
    summary = {
        "purity_estimate": purity_est,
        "ploidy_estimate": ploidy_est,
    }
    summary.update(hrd_metrics)
    summary_file = os.path.join(output_dir, f"{sample_name}_cn_summary.tsv")
    try:
        pd.DataFrame([summary]).to_csv(summary_file, sep="\t", index=False)
        log.append(f"- Saved summary metrics: {summary_file}")
    except Exception as e:
        log.append(f"- WARNING: Could not save summary file: {e}")

    if focal_events:
        focal_file = os.path.join(output_dir, f"{sample_name}_focal_events.tsv")
        try:
            pd.DataFrame(focal_events, columns=["Gene", "Event", "Log2"]).to_csv(focal_file, sep="\t", index=False)
            log.append(f"- Saved focal events: {focal_file}")
        except Exception as e:
            log.append(f"- WARNING: Could not save focal events file: {e}")

    log.append("\nNEXT STEPS:")
    log.append("- For improved purity/ploidy estimation, consider ABSOLUTE, FACETS, PureCN, or Sequenza")
    log.append("- For validated HRD scoring, use scarHRD or HRDetect with allele-specific data")
    log.append("- Integrate CN events with expression data to assess driver gene activation/dosage")
    log.append("- Visualize CNV profiles using CNVkit's built-in plotting functions")

    return "\n".join(log)


def _run_pipeline(commands):
    """Run ``commands`` as a shell-free pipeline (each one's stdout into the next one's stdin).

    Raises CalledProcessError for the first stage that exits non-zero, with the tail of its stderr
    attached. Each stage's stderr goes to its own temporary file: a pipe read one stage at a time
    can fill up and stall the whole pipeline.
    """
    import subprocess
    import tempfile

    procs, logs = [], []
    previous = None
    try:
        for index, cmd in enumerate(commands):
            last = index == len(commands) - 1
            err = tempfile.TemporaryFile()
            logs.append(err)
            proc = subprocess.Popen(
                cmd,
                stdin=previous.stdout if previous is not None else None,
                stdout=None if last else subprocess.PIPE,
                stderr=err,
            )
            if previous is not None:
                previous.stdout.close()  # so the upstream stage sees SIGPIPE if this one exits early
            procs.append(proc)
            previous = proc
        codes = [proc.wait() for proc in procs]
        for cmd, code, err in zip(commands, codes, logs, strict=True):
            if code != 0:
                err.seek(0)
                raise subprocess.CalledProcessError(code, cmd, stderr=err.read()[-2000:])
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for err in logs:
            err.close()


def _stderr_tail(error) -> str:
    """The last 2,000 characters of a failed command's stderr, as a log suffix ('' when there is none)."""
    err = getattr(error, "stderr", None)
    if not err:
        return ""
    if isinstance(err, bytes):
        err = err.decode("utf-8", "replace")
    return f"\n  stderr: {err.strip()[-2000:]}"


def _count_pass_variants(vcf_path) -> dict:
    """Counts over a VCF's records: all, PASS, PASS by type (from REF/ALT), and PASS with a HIGH impact."""
    counts = dict.fromkeys(["records", "PASS", "SNP", "MNP", "INS", "DEL", "MULTIALLELIC", "HIGH"], 0)
    with open(vcf_path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            counts["records"] += 1
            if len(fields) < 8 or fields[6] != "PASS":
                continue
            counts["PASS"] += 1
            ref, alts = fields[3], fields[4].split(",")
            if len(alts) > 1:
                kind = "MULTIALLELIC"
            elif len(ref) == len(alts[0]):
                kind = "SNP" if len(ref) == 1 else "MNP"
            else:
                kind = "INS" if len(alts[0]) > len(ref) else "DEL"
            counts[kind] += 1
            if "|HIGH|" in fields[7]:
                counts["HIGH"] += 1
    return counts


def _natural_chromosome_key(name):
    """Sort key putting chromosomes in genome order: 1..22, then X, Y, M/MT, then anything else by name."""
    text = str(name)
    bare = text[3:] if text.lower().startswith("chr") else text
    if bare.isdigit():
        return (0, int(bare), "")
    order = {"x": 1, "y": 2, "m": 3, "mt": 3}
    return (1, order.get(bare.lower(), 4), bare)
