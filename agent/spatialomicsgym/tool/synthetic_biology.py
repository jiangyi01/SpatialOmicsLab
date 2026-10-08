def _default_output_dir() -> str:
    """Where this installation writes tool output when the caller named no directory.

    Several functions here built a fixed, relative filename and wrote it next to whatever directory
    the process happened to start in -- the repo checkout, for sog-web -- so two chats overwrote
    each other's results (hunt 2026-09-30, uT5-molbio-22). ``spatialomicsgym.paths`` owns the
    branch set (``$SOG_WORK_DIR``, then /workspace/work, then ./work); this only falls back to its
    last branch if the package cannot be imported at all.
    """
    try:
        from spatialomicsgym.paths import tool_output_root

        return tool_output_root()
    except Exception:
        import os

        return os.environ.get("SOG_WORK_DIR", "").strip() or "./work"


def _run_stem(label: str = "") -> str:
    """``<label>_<timestamp>`` -- a stem no concurrent call collides on."""
    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cleaned = _safe_stem(label)
    return f"{cleaned}_{stamp}" if cleaned else stamp


def _safe_stem(label: str) -> str:
    """``label`` reduced to a filename component: no separators, no traversal, no surprises."""
    import os
    import re

    stem = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(str(label))).strip("._")
    return stem[:80]


def _output_path(output_dir: str, name: str) -> str:
    """``name`` inside ``output_dir``, refusing anything that would land outside it."""
    import os

    target = os.path.realpath(os.path.join(output_dir, name))
    root = os.path.realpath(output_dir)
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(f"refusing to write {name!r} outside {output_dir}")
    return target


def engineer_bacterial_genome_for_therapeutic_delivery(bacterial_genome_file, genetic_parts, output_dir=None):
    """Engineer a bacterial genome by integrating therapeutic genetic parts for therapeutic delivery.

    Parameters
    ----------
    bacterial_genome_file : str
        Path to the file containing the bacterial genome sequence in FASTA format
    output_dir : str, optional
        Directory for the engineered FASTA and the plasmid map. Defaults to this installation's
        tool output root ($SOG_WORK_DIR, else /workspace/work, else ./work). The two filenames are
        derived from the genome's FASTA header, reduced to a safe filename component.
    genetic_parts : dict
        Dictionary containing genetic parts to be integrated::

        {
            'promoters': list of dict with 'name', 'sequence', and 'position',
            'genes': list of dict with 'name', 'sequence', and 'position',
            'terminators': list of dict with 'name', 'sequence', and 'position',
            'cargo': dict with 'name' and 'sequence' of the therapeutic cargo
        }

    Returns
    -------
    str
        Research log summarizing the engineering process, including the filename of the
        engineered genome and plasmid map

    """
    import copy
    import datetime
    import os

    from Bio import SeqIO
    from Bio.Graphics import GenomeDiagram
    from Bio.Seq import Seq
    from Bio.SeqFeature import FeatureLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord
    from reportlab.lib import colors

    # The caller's own dict stays the caller's. Both loops below used to rewrite the positions
    # inside it, so re-running the same call -- the normal reaction to an error in a REPL whose
    # namespace persists -- inserted the parts 10 bp further along each time (hunt 2026-09-30,
    # uT5-molbio-19).
    genetic_parts = copy.deepcopy(genetic_parts)

    output_dir = output_dir or _default_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    # Initialize research log
    log = ["# Bacterial Genome Engineering for Therapeutic Delivery\n"]
    log.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Step 1: Load the bacterial genome
    log.append("## Step 1: Loading Bacterial Genome")
    try:
        genome_record = SeqIO.read(bacterial_genome_file, "fasta")
        genome_seq = genome_record.seq
        log.append(f"Successfully loaded genome: {genome_record.id}")
        log.append(f"Genome length: {len(genome_seq)} bp\n")
    except Exception as e:
        log.append(f"Error loading genome: {str(e)}")
        return "\n".join(log)

    # Step 2: Design and integrate genetic parts
    log.append("## Step 2: Integrating Genetic Parts")

    # Create a new genome sequence for modifications
    # In newer BioPython versions, Seq objects are mutable by default
    # or we can convert to a string and modify it, then convert back to Seq
    engineered_seq = str(genome_seq)

    # Track features for visualization
    features = []

    # Every part's 'position' is a coordinate in the ORIGINAL genome, so an insertion has to be
    # offset by the parts already inserted upstream of it -- and only by those. The old code did it
    # twice: it rewrote each downstream part's own 'position' AND accumulated a running total, then
    # added the total to the already-rewritten value, so a promoter of 10 bp at 100 pushed a gene
    # designed for 200 to 220 and a terminator designed for 300 to 360 (hunt 2026-09-30,
    # uT5-molbio-17). One mechanism, computed from what was actually inserted.
    inserted = []  # (original position, inserted length)

    def _offset(original_position):
        return sum(length for pos, length in inserted if pos <= original_position)

    # Add promoters
    if "promoters" in genetic_parts:
        log.append("### Adding Promoters:")
        for promoter in genetic_parts["promoters"]:
            position = promoter["position"] + _offset(promoter["position"])

            # Insert the sequence at the position
            engineered_seq = engineered_seq[:position] + promoter["sequence"] + engineered_seq[position:]
            inserted.append((promoter["position"], len(promoter["sequence"])))

            # Add feature for visualization
            feature = SeqFeature(
                FeatureLocation(position, position + len(promoter["sequence"])),
                type="promoter",
                qualifiers={"label": promoter["name"]},
            )
            features.append(feature)

            log.append(f"  - Added promoter {promoter['name']} at position {position}")

    # Add genes
    if "genes" in genetic_parts:
        log.append("### Adding Genes:")
        for gene in genetic_parts["genes"]:
            position = gene["position"] + _offset(gene["position"])

            # Insert the sequence at the position
            engineered_seq = engineered_seq[:position] + gene["sequence"] + engineered_seq[position:]
            inserted.append((gene["position"], len(gene["sequence"])))

            # Add feature for visualization
            feature = SeqFeature(
                FeatureLocation(position, position + len(gene["sequence"])),
                type="gene",
                qualifiers={"label": gene["name"]},
            )
            features.append(feature)

            log.append(f"  - Added gene {gene['name']} at position {position}")

    # Add terminators
    if "terminators" in genetic_parts:
        log.append("### Adding Terminators:")
        for terminator in genetic_parts["terminators"]:
            position = terminator["position"] + _offset(terminator["position"])

            # Insert the sequence at the position
            engineered_seq = engineered_seq[:position] + terminator["sequence"] + engineered_seq[position:]
            inserted.append((terminator["position"], len(terminator["sequence"])))

            # Add feature for visualization
            feature = SeqFeature(
                FeatureLocation(position, position + len(terminator["sequence"])),
                type="terminator",
                qualifiers={"label": terminator["name"]},
            )
            features.append(feature)

            log.append(f"  - Added terminator {terminator['name']} at position {position}")

    # Add therapeutic cargo
    if "cargo" in genetic_parts:
        cargo = genetic_parts["cargo"]
        # Find a suitable position (after the last added element)
        position = max([f.location.end for f in features]) if features else len(engineered_seq) // 2

        # Insert the sequence at the position
        engineered_seq = engineered_seq[:position] + cargo["sequence"] + engineered_seq[position:]

        # Add feature for visualization
        feature = SeqFeature(
            FeatureLocation(position, position + len(cargo["sequence"])),
            type="therapeutic_cargo",
            qualifiers={"label": cargo["name"]},
        )
        features.append(feature)

        log.append("### Added Therapeutic Cargo:")
        log.append(f"  - Added {cargo['name']} at position {position}")
        log.append(f"  - Cargo length: {len(cargo['sequence'])} bp\n")

    # Step 3: Create the engineered genome record
    log.append("## Step 3: Finalizing Engineered Genome")
    engineered_genome = SeqRecord(
        seq=Seq(engineered_seq),
        id=f"{genome_record.id}_engineered",
        name=f"{genome_record.id}_engineered",
        description=f"Engineered {genome_record.id} for therapeutic delivery",
    )

    # Add all features to the engineered genome
    engineered_genome.features = features

    # Save the engineered genome to a file. The stem is the header reduced to a filename
    # component, plus a run stamp: it used to be interpolated raw and written relative to the
    # process CWD, so a header of '>../../tmp/x' chose the write path and a plain one dropped both
    # files beside the server, where the next run of the same genome replaced them (hunt
    # 2026-09-30, uT5-molbio-18 / uT5-molbio-22).
    stem = _run_stem(_safe_stem(genome_record.id) or "genome")
    output_genome_file = _output_path(output_dir, f"{stem}_engineered.fasta")
    SeqIO.write(engineered_genome, output_genome_file, "fasta")
    log.append(f"Engineered genome saved to: {output_genome_file}")
    log.append(f"Total genome length: {len(engineered_genome.seq)} bp\n")

    # Step 4: Generate a plasmid map
    log.append("## Step 4: Generating Plasmid Map")

    # Create the diagram
    gd_diagram = GenomeDiagram.Diagram("Engineered Bacterial Genome")
    gd_track = gd_diagram.new_track(1, name="Engineered Features")
    gd_feature_set = gd_track.new_set()

    # Add features with different colors
    for feature in features:
        if feature.type == "promoter":
            color = colors.green
        elif feature.type == "gene":
            color = colors.blue
        elif feature.type == "terminator":
            color = colors.red
        elif feature.type == "therapeutic_cargo":
            color = colors.purple
        else:
            color = colors.grey

        gd_feature_set.add_feature(feature, color=color, label=True)

    # Draw and save the plasmid map
    plasmid_map_file = _output_path(output_dir, f"{stem}_engineered_map.pdf")
    gd_diagram.draw(format="circular", circular=True, pagesize=(1000, 1000))
    gd_diagram.write(plasmid_map_file, "PDF")

    log.append(f"Plasmid map generated and saved to: {plasmid_map_file}\n")

    # Step 5: Final summary
    log.append("## Summary of Engineered Bacterial Strain")
    log.append(f"- Base strain: {genome_record.id}")
    log.append("- Added genetic parts:")

    if "promoters" in genetic_parts:
        log.append(f"  - Promoters: {len(genetic_parts['promoters'])}")
    if "genes" in genetic_parts:
        log.append(f"  - Genes: {len(genetic_parts['genes'])}")
    if "terminators" in genetic_parts:
        log.append(f"  - Terminators: {len(genetic_parts['terminators'])}")
    if "cargo" in genetic_parts:
        log.append(f"  - Therapeutic cargo: {genetic_parts['cargo']['name']}")

    log.append("\nThe engineered bacterial strain is ready for transformation and selection.")

    return "\n".join(log)


def analyze_bacterial_growth_rate(time_points, od_measurements, strain_name="Unknown strain", output_dir="./"):
    """Analyze bacterial growth data and extract growth parameters from OD600 measurements.

    Parameters
    ----------
    time_points : list or numpy.ndarray
        Time points at which OD600 measurements were taken (in hours)
    od_measurements : list or numpy.ndarray
        Optical density (OD600) measurements corresponding to each time point
    strain_name : str, optional
        Name of the bacterial strain being analyzed, default is "Unknown strain"
    output_dir : str, optional
        Directory where to save the output files, default is current directory

    Returns
    -------
    str
        A research log summarizing the analysis steps, fitted parameters, and results

    """
    import os
    from datetime import datetime

    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert inputs to numpy arrays if they aren't already
    time_points = np.array(time_points, dtype=float)
    od_measurements = np.array(od_measurements, dtype=float)

    # Checked here, not left to curve_fit. curve_fit raises ValueError for a shape mismatch and for
    # NaN, and only RuntimeError for non-convergence, so the two likeliest bad inputs -- a truncated
    # plate-reader export and NaNs in the series -- escaped as a raw numpy broadcast error instead
    # of reaching the error branch written for them below (hunt 2026-09-30, uT5-molbio-28).
    if time_points.shape != od_measurements.shape:
        return f"""## Bacterial Growth Rate Analysis: {strain_name}

### Error in Analysis
{len(time_points)} time points were given with {len(od_measurements)} OD measurements; they must be
paired one to one.

### Suggestions
- Check the export for a truncated column
- Pass the full series for both: do not subsample either to make the lengths match
"""

    # Define the Gompertz growth model function
    def gompertz_model(t, lag, mu_max, A):
        """Gompertz growth model.

        Parameters
        ----------
        t: time points
        lag: lag time (hours)
        mu_max: maximum growth rate (per hour)
        A: carrying capacity (maximum OD)

        """
        return A * np.exp(-np.exp(mu_max * np.exp(1) * (lag - t) / A + 1))

    # Initial parameter guesses
    p0 = [
        np.mean(time_points) / 3,  # lag time guess: 1/3 of the mean time
        0.5,  # growth rate guess: 0.5 per hour
        max(od_measurements) * 1.1,  # carrying capacity: slightly above max OD
    ]

    # Fit the model to the data
    try:
        if not np.all(np.isfinite(time_points)) or not np.all(np.isfinite(od_measurements)):
            raise ValueError("time_points and od_measurements must be finite (no NaN or inf)")
        popt, pcov = curve_fit(
            gompertz_model,
            time_points,
            od_measurements,
            p0=p0,
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
        )
        lag_time, mu_max, carrying_capacity = popt

        # In the modified Gompertz form fitted above the carrying capacity sits inside the
        # exponent, so mu_max is the maximum SLOPE of OD against time (OD/h), not a specific growth
        # rate: ln(2)/mu_max has units of hours x OD and is not a doubling time. Two cultures with
        # identical kinetics but a 10x different final OD were reported with 10x different
        # "doubling times" (hunt 2026-09-30, uT5-molbio-29). The relative rate at the inflection
        # point of this model is mu_max * e / A, which is what a doubling time divides into.
        specific_growth_rate = (
            mu_max * np.e / carrying_capacity if carrying_capacity > 0 and mu_max > 0 else float("nan")
        )
        doubling_time = np.log(2) / specific_growth_rate if specific_growth_rate > 0 else float("nan")

        # Generate fitted curve for plotting
        t_smooth = np.linspace(min(time_points), max(time_points), 100)
        od_fit = gompertz_model(t_smooth, *popt)

        # Create growth curve plot
        plt.figure(figsize=(10, 6))
        plt.scatter(time_points, od_measurements, label="Observed OD600")
        plt.plot(t_smooth, od_fit, "r-", label="Fitted Gompertz model")
        plt.xlabel("Time (hours)")
        plt.ylabel("Optical Density (OD600)")
        plt.title(f"Bacterial Growth Curve: {strain_name}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(output_dir, f"{strain_name.replace(' ', '_')}_{timestamp}_growth_curve.png")
        plt.savefig(plot_filename)
        plt.close()

        # Create research log
        log = f"""## Bacterial Growth Rate Analysis: {strain_name}

### Analysis Summary
- **Date and Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Sample:** {strain_name}
- **Data Points:** {len(time_points)} measurements over {max(time_points) - min(time_points):.2f} hours

### Growth Parameters (Gompertz Model)
- **Lag Phase Duration:** {lag_time:.2f} hours
- **Maximum Growth Rate (μ_max):** {mu_max:.4f} OD600 units per hour (the model's maximum slope)
- **Specific Growth Rate at Inflection (μ_max·e/A):** {specific_growth_rate:.4f} per hour
- **Doubling Time:** {doubling_time:.2f} hours (ln2 divided by the specific growth rate above)
- **Carrying Capacity (max OD):** {carrying_capacity:.4f} OD600 units

### Methods
1. OD600 measurements were taken at various time points
2. Data was fitted to the Gompertz growth model
3. Growth parameters were extracted from the fitted model
4. Growth curve was plotted and saved as: {plot_filename}

### Note
The Gompertz model provides a good representation of bacterial growth phases including lag, exponential, and stationary phases.
"""

        return log

    except (RuntimeError, ValueError) as e:
        return f"""## Bacterial Growth Rate Analysis: {strain_name}

### Error in Analysis
Failed to fit the growth model to the provided data: {str(e)}

### Suggestions
- Check if the data shows a typical growth pattern
- Ensure sufficient data points are provided
- Try different initial parameter guesses or a different growth model
"""


def analyze_barcode_sequencing_data(
    input_file,
    barcode_pattern=None,
    flanking_seq_5prime=None,
    flanking_seq_3prime=None,
    min_count=5,
    output_dir="./results",
    barcode_length=None,
):
    """Analyze sequencing data to extract, quantify and determine lineage relationships of barcodes.

    Parameters
    ----------
    input_file : str
        Path to the input sequencing file in FASTQ or FASTA format
    barcode_pattern : str, optional
        Regular expression pattern to identify barcodes. If None, will use flanking sequences
    flanking_seq_5prime : str, optional
        5' flanking sequence of the barcode region
    flanking_seq_3prime : str, optional
        3' flanking sequence of the barcode region
    min_count : int, default=5
        Minimum count threshold for considering a barcode
    output_dir : str, default="./results"
        Directory to save output files
    barcode_length : int, optional
        Exact barcode length in nucleotides, used only in flanking-sequence mode. Give it when the
        library has one: it anchors the match so a barcode whose 3' end happens to start the 3'
        flank is not silently shortened. Without it the extracted lengths are reported in the log
        so any truncation is at least visible.

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os
    import re
    from collections import Counter

    import numpy as np
    from Bio import SeqIO
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize research log
    log = "# Barcode Sequencing Analysis Research Log\n\n"
    log += f"## Input Data\n- File: {input_file}\n\n"

    # Step 1: Read sequences and extract barcodes
    log += "## Step 1: Extracting barcodes from sequencing data\n"

    # Determine file format based on extension
    file_format = "fastq" if input_file.endswith((".fastq", ".fq")) else "fasta"

    barcodes = []
    total_reads = 0

    # Compiled once, before the read loop, and with the flanks escaped. The flanks are documented
    # as sequences, not as patterns, and were interpolated raw: a flank quoted from a vector map
    # with a stray bracket escaped as a bare re.error that named no argument (hunt 2026-09-30,
    # uT5-molbio-24). The barcode group is anchored to barcode_length when the caller states one,
    # because a non-greedy (.*?) stops at the earliest run matching the flank and counts the
    # truncated string as a barcode of its own (hunt 2026-09-30, uT5-molbio-25).
    compiled_pattern = None
    if barcode_pattern:
        try:
            compiled_pattern = re.compile(barcode_pattern)
        except re.error as exc:
            raise ValueError(f"barcode_pattern is not a valid regular expression: {exc}") from None
    elif flanking_seq_5prime and flanking_seq_3prime:
        # Escaped, a flank that is not DNA can no longer crash -- but it matches nothing, which
        # reads as "no barcodes in this data". Say which argument is not a sequence, and where a
        # pattern belongs (hunt 2026-09-30, uT5-molbio-24).
        for name, flank in (
            ("flanking_seq_5prime", flanking_seq_5prime),
            ("flanking_seq_3prime", flanking_seq_3prime),
        ):
            stray = sorted({c for c in str(flank) if c not in "ACGTUNacgtun"})
            if stray:
                raise ValueError(
                    f"{name}={flank!r} is matched literally and is not a DNA sequence: {stray} "
                    f"are not bases. Pass a plain ACGT/N flank, or give the whole match as "
                    f"barcode_pattern, which is read as a regular expression."
                )
        middle = f"([ACGTNacgtn]{{{int(barcode_length)}}})" if barcode_length else "(.*?)"
        compiled_pattern = re.compile(f"{re.escape(flanking_seq_5prime)}{middle}{re.escape(flanking_seq_3prime)}")

    for record in SeqIO.parse(input_file, file_format):
        total_reads += 1
        seq_str = str(record.seq)

        # Extract barcode using pattern or flanking sequences
        if compiled_pattern is None:
            continue
        match = compiled_pattern.search(seq_str)
        if match:
            barcodes.append(match.group(0) if barcode_pattern else match.group(1))

    log += f"- Total reads processed: {total_reads}\n"
    log += f"- Barcodes extracted: {len(barcodes)}\n"
    if barcodes:
        lengths = Counter(len(bc) for bc in barcodes)
        log += "- Extracted barcode lengths: " + ", ".join(f"{n} nt x{c}" for n, c in sorted(lengths.items())) + "\n"
        if len(lengths) > 1 and not barcode_length:
            log += (
                "- NOTE: the extracted barcodes are not all the same length. In flanking-sequence "
                "mode the match stops at the first occurrence of the 3' flank, so a barcode whose "
                "3' end starts that flank is reported short. Pass barcode_length to anchor it.\n"
            )
    log += "\n"

    if len(barcodes) == 0:
        log += "ERROR: No barcodes found. Check your barcode pattern or flanking sequences.\n"
        return log

    # Step 2: Quantify barcode abundances
    log += "## Step 2: Quantifying barcode abundances\n"
    barcode_counts = Counter(barcodes)

    # Filter low-abundance barcodes
    filtered_barcodes = {bc: count for bc, count in barcode_counts.items() if count >= min_count}

    log += f"- Unique barcodes: {len(barcode_counts)}\n"
    log += f"- Barcodes with count ≥ {min_count}: {len(filtered_barcodes)}\n"

    # Save barcode counts to file
    count_file = os.path.join(output_dir, "barcode_counts.tsv")
    with open(count_file, "w") as f:
        f.write("Barcode\tCount\tFrequency\n")
        for bc, count in sorted(filtered_barcodes.items(), key=lambda x: x[1], reverse=True):
            freq = count / total_reads
            f.write(f"{bc}\t{count}\t{freq:.6f}\n")

    log += f"- Barcode abundances saved to: {count_file}\n\n"

    # Step 3: Analyze barcode relationships (lineage analysis)
    log += "## Step 3: Analyzing barcode relationships and lineages\n"

    # Only proceed if we have enough barcodes
    if len(filtered_barcodes) < 2:
        log += "- Not enough barcodes for lineage analysis after filtering\n\n"
    else:
        # Convert barcodes to numerical representation for distance calculation
        barcode_list = list(filtered_barcodes.keys())

        # Create a simple distance matrix based on Hamming distance
        def hamming_distance(s1, s2):
            # If sequences have different lengths, pad the shorter one
            if len(s1) != len(s2):
                max_len = max(len(s1), len(s2))
                s1 = s1.ljust(max_len)
                s2 = s2.ljust(max_len)
            return sum(c1 != c2 for c1, c2 in zip(s1, s2, strict=False))

        # Create distance matrix
        n = len(barcode_list)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = hamming_distance(barcode_list[i], barcode_list[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # Perform hierarchical clustering
        try:
            # squareform, not pdist: dist_matrix is already the full Hamming matrix, and pdist
            # computed the EUCLIDEAN distance between its rows, so the cutoff of 3 thresholded a
            # distance-of-distances and two barcodes exactly 3 mismatches apart (distance 4.24
            # after pdist) were split into separate lineages. The error was one-sided -- it always
            # over-split (hunt 2026-09-30, uT5-molbio-23).
            condensed_dist = squareform(dist_matrix, checks=False)
            Z = linkage(condensed_dist, method="average")
            max_dist = 3  # Max Hamming distance to consider barcodes in same lineage
            clusters = fcluster(Z, max_dist, criterion="distance")

            # Count clusters
            cluster_counts = Counter(clusters)

            log += f"- Identified {len(cluster_counts)} potential lineages\n"
            log += f"- Largest lineage contains {max(cluster_counts.values())} barcodes\n"

            # Save lineage information
            lineage_file = os.path.join(output_dir, "barcode_lineages.tsv")
            with open(lineage_file, "w") as f:
                f.write("Barcode\tCount\tLineage\n")
                for i, bc in enumerate(barcode_list):
                    f.write(f"{bc}\t{filtered_barcodes[bc]}\t{clusters[i]}\n")

            log += f"- Lineage assignments saved to: {lineage_file}\n\n"
        except Exception as e:
            log += f"- Error in lineage analysis: {str(e)}\n\n"

    # Step 4: Summary
    log += "## Summary\n"
    log += f"- Processed {total_reads} sequencing reads\n"
    log += f"- Identified {len(barcode_counts)} unique barcodes\n"
    log += f"- {len(filtered_barcodes)} barcodes passed abundance threshold (≥ {min_count})\n"

    return log


def analyze_bifurcation_diagram(time_series_data, parameter_values, system_name="Dynamical System", output_dir="./"):
    """Performs bifurcation analysis on a dynamical system and generates a bifurcation diagram.

    Parameters
    ----------
    time_series_data : numpy.ndarray
        A 2D array where each row represents a time series for a specific parameter value.
        Shape should be (n_parameter_values, n_time_points).
    parameter_values : numpy.ndarray
        1D array of parameter values corresponding to each time series.
        Shape should be (n_parameter_values,).
    system_name : str, optional
        Name of the dynamical system being analyzed, used for plot titles.
    output_dir : str, optional
        Directory to save the output files.

    Returns
    -------
    str
        Research log summarizing the bifurcation analysis process and results.

    """
    import os

    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize research log
    research_log = f"# Bifurcation Analysis for {system_name}\n\n"
    research_log += "## Analysis Steps\n\n"

    # Step 1: Verify input data
    research_log += "### Step 1: Data Verification\n"
    if len(parameter_values) != time_series_data.shape[0]:
        research_log += "Error: Number of parameter values does not match number of time series.\n"
        return research_log

    research_log += f"- Analyzed {len(parameter_values)} parameter values\n"
    research_log += f"- Each time series contains {time_series_data.shape[1]} data points\n\n"

    # Step 2: Calculate indicators for different dynamical regimes
    research_log += "### Step 2: Calculating Dynamical Indicators\n"

    # Initialize arrays to store results
    local_maxima_counts = np.zeros(len(parameter_values))
    attractor_levels = np.zeros(len(parameter_values))
    relative_step = np.zeros(len(parameter_values))
    attractor_points = []

    # How many distinct values an orbit visits, in bins of 1% of that series' OWN range. The regime
    # label used to come from log(mean|diff|), which carries the units of the state variable: the
    # same period-2 orbit was reported 'period-15' at amplitude 0.05 and 'chaotic' at amplitude 5,
    # and the periodicity label counted peaks in the window rather than distinct attractor values
    # (hunt 2026-09-30, uT5-molbio-27). Counting levels relative to the range is scale-free, and it
    # also gives the 3-peak case a label instead of leaving a period-3 window as 'unknown'.
    attractor_bins = 100

    def _levels(values, low, span):
        if span <= 0:
            return 1
        index = np.clip(
            ((np.asarray(values, dtype=float) - low) / (span / attractor_bins)).astype(int), 0, attractor_bins - 1
        )
        return int(len(np.unique(index)))

    for i, series in enumerate(time_series_data):
        # Use the last 70% of the time series to avoid transients
        steady_state = series[int(0.7 * len(series)) :]

        # Find local maxima (peaks) to identify periodicity
        peaks, _ = find_peaks(steady_state)
        local_maxima_counts[i] = len(peaks)

        span = float(np.ptp(steady_state))
        low = float(np.min(steady_state))
        step = float(np.mean(np.abs(np.diff(steady_state)))) if len(steady_state) > 1 else 0.0
        # Mean step as a fraction of the range: near 1 for a discrete map, whose consecutive
        # samples jump across the whole attractor, and small for a finely sampled trajectory.
        relative_step[i] = step / span if span > 0 else 0.0

        # Collect points for the bifurcation diagram. A discrete orbit has no meaningful peaks, so
        # every steady-state sample is an attractor point; a smooth trajectory is sampled at its
        # local maxima, as before.
        if relative_step[i] > 0.3:
            points = steady_state
        elif len(peaks) > 0:
            points = steady_state[peaks]
        else:
            # If no peaks, use the last few points
            points = steady_state[-5:]
        attractor_points.append((parameter_values[i], points))
        attractor_levels[i] = _levels(points, low, span)

    research_log += "- Counted the distinct attractor levels each orbit visits (bins of 1% of its own range)\n"
    research_log += "- Identified local maxima for periodicity analysis\n"
    research_log += "- Sampled attractor points for bifurcation diagram\n\n"

    # Step 3: Classify dynamical regimes
    research_log += "### Step 3: Classifying Dynamical Regimes\n"

    # Initialize arrays for regime classification
    regimes = np.full(len(parameter_values), "unknown", dtype=object)

    # Classify on the attractor, not on the signal's amplitude. An orbit that settles on a
    # handful of levels is periodic at that order; one whose levels fill a tenth of the range or
    # more is a band, which is what chaos looks like here.
    for i in range(len(parameter_values)):
        levels = int(attractor_levels[i])
        if relative_step[i] <= 0.3 and local_maxima_counts[i] == 0:
            regimes[i] = "stable"
        elif levels <= 1:
            regimes[i] = "period-1"
        elif levels <= 10:
            regimes[i] = f"period-{levels}"
        else:
            regimes[i] = "chaotic"

    # Count regime types
    unique_regimes = np.unique(regimes)
    for regime in unique_regimes:
        count = np.sum(regimes == regime)
        research_log += f"- Identified {count} parameter values in '{regime}' regime\n"

    research_log += "\n"

    # Step 4: Create bifurcation diagram
    research_log += "### Step 4: Creating Bifurcation Diagram\n"

    plt.figure(figsize=(12, 8))

    # Plot bifurcation points
    for param_val, points in attractor_points:
        y_values = points.flatten()  # Flatten in case points is multi-dimensional
        plt.plot([param_val] * len(y_values), y_values, "k.", markersize=0.5)

    # Highlight different regimes with colored background
    regime_changes = np.where(regimes[:-1] != regimes[1:])[0]
    regime_boundaries = (
        [parameter_values[0]] + [parameter_values[i + 1] for i in regime_changes] + [parameter_values[-1]]
    )

    colors = {
        "stable": "lightblue",
        "period-1": "lightgreen",
        "period-2": "lightyellow",
        "period-3": "lightsalmon",
        "period-4": "lightpink",
        "chaotic": "salmon",
    }

    for i in range(len(regime_boundaries) - 1):
        start_idx = list(parameter_values).index(regime_boundaries[i])
        regime = regimes[start_idx]
        if regime in colors:
            plt.axvspan(
                regime_boundaries[i],
                regime_boundaries[i + 1],
                alpha=0.3,
                color=colors.get(regime, "lightgray"),
            )

    plt.xlabel("Parameter Value")
    plt.ylabel("State Variable")
    plt.title(f"Bifurcation Diagram for {system_name}")

    # Add legend for regimes
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors.get(regime, "lightgray"), alpha=0.3, label=regime)
        for regime in unique_regimes
        if regime in colors
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    # Save the figure
    output_file = os.path.join(output_dir, f"{system_name.replace(' ', '_')}_bifurcation_diagram.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    research_log += "- Created bifurcation diagram showing different dynamical regimes\n"
    research_log += f"- Diagram saved as: {output_file}\n\n"

    # Step 5: Identify regime transitions
    research_log += "### Step 5: Identifying Regime Transitions\n"

    if len(regime_changes) > 0:
        for i in regime_changes:
            research_log += f"- Transition from '{regimes[i]}' to '{regimes[i + 1]}' at parameter value ≈ {parameter_values[i + 1]:.4f}\n"
    else:
        research_log += "- No regime transitions detected in the parameter range\n"

    research_log += "\n### Summary\n"
    research_log += f"Bifurcation analysis completed for {system_name}. "
    research_log += f"The system exhibits {len(unique_regimes)} different dynamical regimes "
    research_log += f"across the parameter range [{parameter_values[0]:.4f}, {parameter_values[-1]:.4f}].\n"

    return research_log


def create_biochemical_network_sbml_model(reaction_network, kinetic_parameters, output_file="biochemical_model.xml"):
    """Generate a mathematical model of a biochemical network in SBML format.

    Parameters
    ----------
    reaction_network : list of dict
        List of dictionaries, each representing a reaction with keys:
        - 'id': Reaction identifier
        - 'name': Reaction name
        - 'reactants': Dict of reactant species IDs and their stoichiometry
        - 'products': Dict of product species IDs and their stoichiometry
        - 'reversible': Boolean indicating if reaction is reversible

    kinetic_parameters : dict
        Dictionary mapping reaction IDs to their kinetic law parameters.
        Each entry should contain:
        - 'law_type': Type of kinetic law (e.g., 'mass_action', 'michaelis_menten')
        - 'parameters': Dict of parameter names and values

    output_file : str, optional
        File path to save the SBML model (default: "biochemical_model.xml")

    Returns
    -------
    str
        Research log summarizing the model creation process

    """
    import os

    import libsbml

    log = "# Biochemical Network SBML Model Creation Log\n\n"

    # Step 1: Create an SBML document
    log += "## Step 1: Creating SBML document\n"
    sbml_ns = libsbml.SBMLNamespaces(3, 2)  # SBML Level 3 Version 2
    document = libsbml.SBMLDocument(sbml_ns)
    model = document.createModel()
    model.setId("biochemical_network_model")
    model.setName("Biochemical Network Model")
    log += "- Created SBML Level 3 Version 2 document\n"
    log += "- Created model with ID 'biochemical_network_model'\n\n"

    # Step 2: Create default compartment
    log += "## Step 2: Creating default compartment\n"
    compartment = model.createCompartment()
    compartment.setId("default")
    compartment.setConstant(True)
    compartment.setSize(1.0)
    compartment.setSpatialDimensions(3)
    log += "- Created default compartment\n\n"

    # Step 3: Create species
    log += "## Step 3: Creating species\n"
    species_set = set()

    # Collect all unique species from reactions
    for reaction in reaction_network:
        for species_id in list(reaction["reactants"].keys()) + list(reaction["products"].keys()):
            species_set.add(species_id)

    # Create species in the model
    for species_id in species_set:
        species = model.createSpecies()
        species.setId(species_id)
        species.setCompartment("default")
        species.setInitialConcentration(0.0)  # Default initial concentration
        species.setHasOnlySubstanceUnits(False)
        species.setBoundaryCondition(False)
        species.setConstant(False)
        log += f"- Created species '{species_id}'\n"
    log += "\n"

    # Step 4: Create reactions
    log += "## Step 4: Creating reactions and kinetic laws\n"
    for reaction_data in reaction_network:
        reaction = model.createReaction()
        reaction.setId(reaction_data["id"])
        reaction.setName(reaction_data["name"])
        reaction.setReversible(reaction_data["reversible"])
        # No setFast: the attribute was removed in SBML Level 3 Version 2, which is what this
        # document is, and libsbml answered -2 (not valid for this level/version) to a call whose
        # return value was never read (hunt 2026-09-30, uT5-molbio-26, side note).

        # Add reactants
        for reactant_id, stoichiometry in reaction_data["reactants"].items():
            species_ref = reaction.createReactant()
            species_ref.setSpecies(reactant_id)
            species_ref.setStoichiometry(stoichiometry)
            species_ref.setConstant(True)

        # Add products
        for product_id, stoichiometry in reaction_data["products"].items():
            species_ref = reaction.createProduct()
            species_ref.setSpecies(product_id)
            species_ref.setStoichiometry(stoichiometry)
            species_ref.setConstant(True)

        # Create kinetic law
        if reaction_data["id"] in kinetic_parameters:
            kinetic_data = kinetic_parameters[reaction_data["id"]]
            kinetic_law = reaction.createKineticLaw()

            # Add parameters to kinetic law
            for param_name, param_value in kinetic_data["parameters"].items():
                parameter = kinetic_law.createParameter()
                parameter.setId(param_name)
                parameter.setValue(param_value)

            # Set the appropriate formula based on the law type
            if kinetic_data["law_type"] == "mass_action":
                # Simple mass action kinetics
                reactants_formula = " * ".join([f"{reactant_id}" for reactant_id in reaction_data["reactants"]])
                formula = f"k * {reactants_formula}" if reactants_formula else "k"
                kinetic_law.setMath(libsbml.parseL3Formula(formula))
            elif kinetic_data["law_type"] == "michaelis_menten":
                # Michaelis-Menten kinetics (assuming single substrate)
                substrate = list(reaction_data["reactants"].keys())[0] if reaction_data["reactants"] else "S"
                formula = f"Vmax * {substrate} / (Km + {substrate})"
                kinetic_law.setMath(libsbml.parseL3Formula(formula))
            # Custom formula provided in the parameters
            elif "formula" in kinetic_data:
                kinetic_law.setMath(libsbml.parseL3Formula(kinetic_data["formula"]))

        log += f"- Created reaction '{reaction_data['id']}' ({reaction_data['name']})\n"
        if reaction_data["id"] in kinetic_parameters:
            log += f"  - Added kinetic law of type '{kinetic_parameters[reaction_data['id']]['law_type']}'\n"
    log += "\n"

    # Step 5: Validate the model
    log += "## Step 5: Validating the SBML model\n"
    document.setConsistencyChecks(libsbml.LIBSBML_CAT_UNITS_CONSISTENCY, False)
    consistent = document.checkConsistency()
    if consistent == 0:
        log += "- Model passed consistency check\n"
    else:
        log += "- Model has consistency issues:\n"
        for i in range(document.getNumErrors()):
            error = document.getError(i)
            log += f"  - {error.getMessage()}\n"
    log += "\n"

    # Step 6: Write model to file. writeSBMLToFile returns a bool and never raises, so an
    # unwritable path -- a directory that does not exist, a read-only mount -- used to end the log
    # with "Model saved to ..." and leave a later step to fail on a file that was never created
    # (hunt 2026-09-30, uT5-molbio-26).
    log += "## Step 6: Saving the SBML model\n"
    written = libsbml.writeSBMLToFile(document, output_file)
    if not written:
        log += f"- Error: could not write the model to '{os.path.abspath(output_file)}'\n"
        log += "- Check that the directory exists and is writable, then call again\n"
        return log
    log += f"- Model saved to '{os.path.abspath(output_file)}'\n\n"

    # Step 7: Summary
    log += "## Summary\n"
    log += f"- Created SBML model with {len(species_set)} species and {len(reaction_network)} reactions\n"
    log += f"- Model saved as '{output_file}'\n"

    return log


def optimize_codons_for_heterologous_expression(target_sequence, host_codon_usage, output_dir=None):
    """Analyzes and optimizes a DNA/RNA sequence for improved expression in a heterologous host organism.

    Parameters
    ----------
    target_sequence : str
        The DNA or RNA sequence of the target gene to be optimized.
        Should contain complete codons (length divisible by 3).

    host_codon_usage : dict
        Dictionary mapping codons to their usage frequency in the host organism.
        Format: {'AUG': 0.8, 'GCC': 0.6, ...} or {'ATG': 0.8, 'GCC': 0.6, ...}

    output_dir : str, optional
        Directory for the two sequence files. Defaults to this installation's tool output root
        ($SOG_WORK_DIR, else /workspace/work, else ./work). The filenames carry a run stamp, so
        two calls do not overwrite each other.

    Returns
    -------
    str
        A research log summarizing the optimization process and results.

    """
    import os

    from Bio.Data import CodonTable

    # Initialize research log
    log = "# Codon Optimization Analysis Research Log\n\n"

    # Check if sequence is DNA or RNA and standardize to DNA
    is_rna = "U" in target_sequence
    working_seq = target_sequence.replace("U", "T") if is_rna else target_sequence

    # The table gets the same normalisation as the sequence. Only the sequence used to be converted,
    # so a host table written in RNA -- the form this function's own docstring offers first -- missed
    # on every lookup, every frequency read as 0, and the run was reported as an optimisation that
    # had changed nothing (hunt 2026-09-30, uT5-molbio-21).
    host_codon_usage = {
        str(codon).upper().replace("U", "T"): value for codon, value in (host_codon_usage or {}).items()
    }

    log += "## Input Sequence Analysis\n"
    log += f"- Sequence type: {'RNA' if is_rna else 'DNA'}\n"
    log += f"- Sequence length: {len(working_seq)} nucleotides\n"

    # Verify sequence length is divisible by 3
    if len(working_seq) % 3 != 0:
        log += (
            f"- WARNING: Sequence length ({len(working_seq)}) is not divisible by 3. Optimization may be incomplete.\n"
        )

    # Extract codons from the sequence
    original_codons = [working_seq[i : i + 3] for i in range(0, len(working_seq), 3)]
    log += f"- Number of codons: {len(original_codons)}\n\n"

    # Get standard genetic code
    standard_table = CodonTable.standard_dna_table

    # Create codon to amino acid mapping
    codon_to_aa = {}
    for codon, amino_acid in standard_table.forward_table.items():
        codon_to_aa[codon] = amino_acid

    # Add stop codons
    for stop_codon in standard_table.stop_codons:
        codon_to_aa[stop_codon] = "*"  # * represents stop codon

    # Create amino acid to codon mapping with host frequencies
    aa_to_codons = {}
    for codon, aa in codon_to_aa.items():
        if aa not in aa_to_codons:
            aa_to_codons[aa] = []
        # Use host frequency or default to 0 if not provided
        frequency = host_codon_usage.get(codon, 0)
        aa_to_codons[aa].append((codon, frequency))

    # Sort codons by frequency for each amino acid
    for aa in aa_to_codons:
        aa_to_codons[aa] = sorted(aa_to_codons[aa], key=lambda x: x[1], reverse=True)

    # Optimize codons
    optimized_codons = []
    untouched_amino_acids = set()
    for codon in original_codons:
        try:
            aa = codon_to_aa[codon]
            # Select the highest frequency codon for this amino acid -- but only when the host table
            # actually prefers one. A zero-frequency tie used to be broken by whatever the standard
            # table happened to list first, which rewrote codons for amino acids the host table says
            # nothing about (hunt 2026-09-30, uT5-molbio-21).
            best_codon, best_frequency = aa_to_codons[aa][0] if aa_to_codons[aa] else (codon, 0)
            if best_frequency > 0:
                optimized_codons.append(best_codon)
            else:
                optimized_codons.append(codon)
                untouched_amino_acids.add(aa)
        except KeyError:
            # If codon not in standard table, keep original
            optimized_codons.append(codon)
            log += f"- WARNING: Non-standard codon detected: {codon}. Keeping original.\n"

    # Combine optimized codons into sequence
    optimized_sequence = "".join(optimized_codons)

    # Convert back to RNA if input was RNA
    if is_rna:
        optimized_sequence = optimized_sequence.replace("T", "U")

    # Calculate optimization statistics
    codon_changes = sum(1 for i, j in zip(original_codons, optimized_codons, strict=False) if i != j)
    percent_changed = (codon_changes / len(original_codons)) * 100 if original_codons else 0

    log += "## Optimization Results\n"
    log += f"- Codons modified: {codon_changes} out of {len(original_codons)} ({percent_changed:.2f}%)\n"
    if untouched_amino_acids:
        log += (
            "- The host table gave no non-zero frequency for "
            + ", ".join(sorted(untouched_amino_acids))
            + ": those codons were left as they were. Check the table's keys if that is unexpected.\n"
        )

    # Save sequences to files. Absolute, so the paths this log reports back are ones the caller can
    # actually find -- and under the caller's output directory with a run stamp, because two fixed
    # relative names meant one chat silently replaced another chat's result (hunt 2026-09-30,
    # uT5-molbio-22).
    output_dir = output_dir or _default_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    stem = _run_stem("codon_optimization")
    original_file = _output_path(output_dir, f"{stem}_original_sequence.txt")
    optimized_file = _output_path(output_dir, f"{stem}_optimized_sequence.txt")

    with open(original_file, "w") as f:
        f.write(target_sequence)

    with open(optimized_file, "w") as f:
        f.write(optimized_sequence)

    log += f"- Original sequence saved to: {original_file}\n"
    log += f"- Optimized sequence saved to: {optimized_file}\n\n"

    # Add summary
    log += "## Summary\n"
    log += "The target gene sequence has been optimized for expression in the provided host organism. "
    log += f"The optimization process replaced {codon_changes} codons ({percent_changed:.2f}%) "
    log += "with synonymous codons that have higher usage frequency in the host.\n"

    return log


def simulate_gene_circuit_with_growth_feedback(
    circuit_topology,
    kinetic_params,
    growth_params,
    simulation_time=100,
    time_points=1000,
    output_dir=None,
):
    """Simulate gene regulatory circuit dynamics with growth feedback.

    Parameters
    ----------
    circuit_topology : numpy.ndarray
        Adjacency matrix representing the gene circuit topology.
        Positive values indicate activation, negative values indicate repression.
        Shape should be (n_genes, n_genes) where n_genes is the number of genes in the circuit.

    kinetic_params : dict
        Dictionary containing kinetic parameters:
        - 'basal_rates': list of basal expression rates for each gene
        - 'degradation_rates': list of degradation rates for each gene
        - 'hill_coefficients': list of Hill coefficients for regulatory interactions
        - 'threshold_constants': list of threshold constants for regulatory interactions

    growth_params : dict
        Dictionary containing growth-related parameters:
        - 'max_growth_rate': maximum cell growth rate
        - 'growth_inhibition': how gene expression affects growth
        - 'gene_growth_weights': weights for how each gene affects growth

    simulation_time : float, optional
        Total simulation time (default: 100)

    time_points : int, optional
        Number of time points to sample (default: 1000)

    output_dir : str, optional
        Directory for the results files. Defaults to a gene_circuit_results folder under this
        installation's tool output root ($SOG_WORK_DIR, else /workspace/work, else ./work), rather
        than under whatever directory the server happens to have started in.

    Returns
    -------
    str
        Research log summarizing the simulation and results with file names of saved data

    """
    import datetime
    import json
    import os

    import numpy as np
    from scipy.integrate import solve_ivp

    # Extract parameters
    n_genes = circuit_topology.shape[0]
    basal_rates = kinetic_params["basal_rates"]
    degradation_rates = kinetic_params["degradation_rates"]
    hill_coefficients = kinetic_params["hill_coefficients"]
    threshold_constants = kinetic_params["threshold_constants"]

    max_growth_rate = growth_params["max_growth_rate"]
    growth_inhibition = growth_params["growth_inhibition"]
    gene_growth_weights = growth_params["gene_growth_weights"]

    # Define the ODE system
    def gene_circuit_odes(t, state):
        # Extract state variables
        gene_expressions = state[:n_genes]
        cell_mass = state[n_genes]

        # Initialize derivatives
        dxdt = np.zeros(n_genes + 1)

        # Cell growth dynamics, computed FIRST: the dilution term below reads this derivative, and
        # it was assigned after the gene loop, so dxdt[n_genes] was still 0 when every gene read it
        # and the growth-mediated dilution this tool exists for was identically zero -- final
        # expression was the same to six figures for max_growth_rate 0.0, 0.5 and 2.0 (hunt
        # 2026-09-30, uT5-molbio-20). Nothing here depends on dxdt, only on the state.
        growth_burden = sum(gene_growth_weights[i] * gene_expressions[i] for i in range(n_genes))
        dxdt[n_genes] = max_growth_rate * cell_mass / (1 + growth_inhibition * growth_burden)
        specific_growth_rate = dxdt[n_genes] / cell_mass if cell_mass > 0 else 0.0

        # Gene expression dynamics
        for i in range(n_genes):
            # Basal expression
            production = basal_rates[i]

            # Regulatory influences
            for j in range(n_genes):
                if circuit_topology[i, j] != 0:
                    # Calculate regulatory effect
                    regulation = gene_expressions[j] ** hill_coefficients[j] / (
                        threshold_constants[j] ** hill_coefficients[j] + gene_expressions[j] ** hill_coefficients[j]
                    )

                    if circuit_topology[i, j] > 0:  # Activation
                        production *= 1 + circuit_topology[i, j] * regulation
                    else:  # Repression
                        production *= 1 + circuit_topology[i, j] * (1 - regulation)

            # Dilution due to growth and degradation
            dilution = degradation_rates[i] + specific_growth_rate

            # Final rate equation for gene i
            dxdt[i] = production - dilution * gene_expressions[i]

        return dxdt

    # Initial conditions (starting with low gene expression and unit cell mass)
    initial_state = np.zeros(n_genes + 1)
    initial_state[:n_genes] = 0.1  # Low initial gene expression
    initial_state[n_genes] = 1.0  # Initial cell mass

    # Solve the ODE system
    t_span = (0, simulation_time)
    t_eval = np.linspace(0, simulation_time, time_points)

    solution = solve_ivp(
        gene_circuit_odes,
        t_span,
        initial_state,
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    # Extract results
    time_series = solution.t
    gene_expression_time_series = solution.y[:n_genes, :]
    cell_growth_time_series = solution.y[n_genes, :]

    # Save results to files. The directory used to be the bare relative "gene_circuit_results",
    # which the REPL worker resolved against the server's own CWD (hunt 2026-09-30, uT5-molbio-22).
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    results_dir = output_dir or os.path.join(_default_output_dir(), "gene_circuit_results")

    os.makedirs(results_dir, exist_ok=True)

    # Save time series data
    data_filename = _output_path(results_dir, f"gene_circuit_simulation_{timestamp}.npz")
    np.savez(
        data_filename,
        time=time_series,
        gene_expression=gene_expression_time_series,
        cell_growth=cell_growth_time_series,
    )

    # Save parameters for reference
    params_filename = _output_path(results_dir, f"simulation_params_{timestamp}.json")
    params_dict = {
        "circuit_topology": circuit_topology.tolist(),
        "kinetic_params": kinetic_params,
        "growth_params": growth_params,
        "simulation_time": simulation_time,
        "time_points": time_points,
    }

    with open(params_filename, "w") as f:
        json.dump(params_dict, f, indent=2)

    # Create research log
    log = f"""Gene Regulatory Circuit Simulation with Growth Feedback - {timestamp}

SIMULATION SUMMARY:
- Simulated a gene regulatory circuit with {n_genes} genes
- Incorporated growth feedback mechanisms
- Total simulation time: {simulation_time} time units
- Collected {time_points} data points

PARAMETERS:
- Basal expression rates: {basal_rates}
- Degradation rates: {degradation_rates}
- Max growth rate: {max_growth_rate}
- Growth inhibition factor: {growth_inhibition}

RESULTS:
- Final gene expression levels: {gene_expression_time_series[:, -1].tolist()}
- Final cell mass: {cell_growth_time_series[-1]:.4f}
- Growth rate at end of simulation: {max_growth_rate / (1 + growth_inhibition * sum(gene_growth_weights[i] * gene_expression_time_series[i, -1] for i in range(n_genes))):.4f}

FILES:
- Time series data saved to: {data_filename}
- Simulation parameters saved to: {params_filename}
"""

    return log


def identify_fas_functional_domains(sequence, sequence_type="protein", output_file="fas_domains_report.txt"):
    """Identifies functional domains within a Fatty Acid Synthase (FAS) sequence and predicts their roles.

    Parameters
    ----------
    sequence : str
        The nucleotide or protein sequence of a FAS gene
    sequence_type : str
        Type of sequence provided - "protein" or "nucleotide" (default: "protein")
    output_file : str
        Name of the output file to save the detailed domain report (default: "fas_domains_report.txt")

    Returns
    -------
    str
        A research log summarizing the steps taken and results of the domain analysis

    """
    import json
    import os
    import time

    import requests
    from Bio.Seq import Seq

    research_log = "# Fatty Acid Synthase (FAS) Domain Analysis\n\n"
    research_log += "## Input Sequence Analysis\n"
    research_log += f"- Sequence type: {sequence_type}\n"
    research_log += (
        f"- Sequence length: {len(sequence)} {'nucleotides' if sequence_type == 'nucleotide' else 'amino acids'}\n\n"
    )

    # Convert nucleotide to protein if needed
    if sequence_type == "nucleotide":
        research_log += "## Sequence Translation\n"
        research_log += "- Converting nucleotide sequence to protein\n"
        try:
            protein_seq = str(Seq(sequence).translate())
            research_log += f"- Translated protein length: {len(protein_seq)} amino acids\n\n"
        except Exception as e:
            research_log += f"- Error in translation: {str(e)}\n"
            return research_log
    else:
        protein_seq = sequence

    # FAS domain information dictionary
    fas_domains = {
        "ketoacyl-synt": {
            "name": "Ketoacyl Synthase (KS)",
            "function": "Catalyzes the condensation of malonyl-ACP with the growing fatty acid chain",
        },
        "Ketoacyl-synt_C": {
            "name": "Ketoacyl Synthase C-terminal (KS-C)",
            "function": "C-terminal domain of KS, involved in substrate binding",
        },
        "Acyl_transf_1": {
            "name": "Acyltransferase (AT)",
            "function": "Transfers acyl groups from acyl-CoA to ACP",
        },
        "ketoacyl-red": {
            "name": "Ketoacyl Reductase (KR)",
            "function": "Reduces the beta-ketoacyl group to a beta-hydroxyacyl group",
        },
        "Thioesterase": {
            "name": "Thioesterase (TE)",
            "function": "Releases the fatty acid from ACP by hydrolysis",
        },
        "PS-DH": {
            "name": "Dehydratase (DH)",
            "function": "Removes water from beta-hydroxyacyl-ACP to form trans-2-enoyl-ACP",
        },
        "Enoyl_red": {
            "name": "Enoyl Reductase (ER)",
            "function": "Reduces the double bond in enoyl-ACP to a saturated acyl-ACP",
        },
        "ACP": {
            "name": "Acyl Carrier Protein (ACP)",
            "function": "Carries the growing fatty acid chain between enzyme domains",
        },
        "PP-binding": {
            "name": "Phosphopantetheine Binding Domain",
            "function": "Attachment site for the phosphopantetheine prosthetic group in ACP",
        },
    }

    # HMMER web API for domain identification
    research_log += "## Domain Identification\n"
    research_log += "- Using HMMER web API to search against Pfam database\n"

    url = "https://www.ebi.ac.uk/Tools/hmmer/search/hmmscan"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    data = {"hmmdb": "pfam", "seq": protein_seq}

    # Both calls carry a timeout. `requests` has no default, so a stalled EBI connection held the
    # REPL cell until the socket died -- inside the ReAct loop, a turn that never returns an
    # observation (hunt 2026-09-30, uT5-molbio-30). Raise HMMER_TIMEOUT_SECONDS /
    # HMMER_POLL_SECONDS if the queue is slow; neither is worked around by truncating the sequence.
    request_timeout = float(os.environ.get("HMMER_TIMEOUT_SECONDS", "60"))
    poll_budget = float(os.environ.get("HMMER_POLL_SECONDS", "120"))

    try:
        response = requests.post(url, data=json.dumps(data), headers=headers, timeout=request_timeout)
        if response.status_code != 200:
            research_log += f"- Error: HMMER API returned status code {response.status_code}\n"
            return research_log

        result_url = response.headers.get("Location", "")
        if not result_url:
            research_log += "- Error: No result URL returned from HMMER API\n"
            return research_log

        # Poll until the job is done, within a budget. A single sleep(2) reported
        # "Could not retrieve results" for a perfectly good sequence whose job was merely still
        # queued, which read the same as an outage and the same as "no domains".
        deadline = time.monotonic() + poll_budget
        result_response = None
        while True:
            time.sleep(2)
            result_response = requests.get(f"{result_url}.json", timeout=request_timeout)
            if result_response.status_code == 200:
                break
            if time.monotonic() >= deadline:
                research_log += (
                    f"- Error: the HMMER job at {result_url} was still not ready after "
                    f"{poll_budget:.0f}s (last status {result_response.status_code}). This is a "
                    "queue or outage, not a result: raise HMMER_POLL_SECONDS and try again.\n"
                )
                return research_log

        results = result_response.json()

        # Process results
        domains_found = []
        if "results" in results and "hits" in results["results"]:
            hits = results["results"]["hits"]
            for hit in hits:
                if "domains" in hit:
                    for domain in hit["domains"]:
                        domain_name = hit.get("name", "").split(".")[0]
                        domain_desc = hit.get("desc", "Unknown")
                        domain_start = domain.get("ali_from", 0)
                        domain_end = domain.get("ali_to", 0)

                        domains_found.append(
                            {
                                "name": domain_name,
                                "description": domain_desc,
                                "start": domain_start,
                                "end": domain_end,
                                "score": domain.get("score", 0),
                            }
                        )

        # Check for FAS domains
        research_log += f"- Found {len(domains_found)} domains in total\n\n"
        research_log += "## FAS Functional Domains Identified\n"

        fas_domains_found = []
        for domain in domains_found:
            for fas_key, fas_info in fas_domains.items():
                if fas_key.lower() in domain["name"].lower():
                    fas_domains_found.append(
                        {
                            "name": fas_info["name"],
                            "pfam_id": domain["name"],
                            "start": domain["start"],
                            "end": domain["end"],
                            "function": fas_info["function"],
                        }
                    )

        if not fas_domains_found:
            research_log += "- No specific FAS domains identified in the sequence\n"
        else:
            research_log += f"- Found {len(fas_domains_found)} FAS-related domains\n"
            for i, domain in enumerate(fas_domains_found, 1):
                research_log += f"\n### {i}. {domain['name']} (Positions {domain['start']}-{domain['end']})\n"
                research_log += f"- Pfam ID: {domain['pfam_id']}\n"
                research_log += f"- Function: {domain['function']}\n"

        # Save detailed report to file
        with open(output_file, "w") as f:
            f.write("# Fatty Acid Synthase (FAS) Domain Analysis Detailed Report\n\n")
            f.write("## Input Sequence\n")
            f.write(f"- Type: {sequence_type}\n")
            f.write(f"- Length: {len(sequence)} {'nucleotides' if sequence_type == 'nucleotide' else 'amino acids'}\n")
            if sequence_type == "nucleotide":
                f.write(f"- Translated protein length: {len(protein_seq)} amino acids\n")

            f.write("\n## All Domains Found\n")
            for i, domain in enumerate(domains_found, 1):
                f.write(f"\n### {i}. {domain['name']} (Positions {domain['start']}-{domain['end']})\n")
                f.write(f"- Description: {domain['description']}\n")
                f.write(f"- Score: {domain['score']}\n")

            f.write("\n## FAS Functional Domains\n")
            if not fas_domains_found:
                f.write("- No specific FAS domains identified in the sequence\n")
            else:
                for i, domain in enumerate(fas_domains_found, 1):
                    f.write(f"\n### {i}. {domain['name']} (Positions {domain['start']}-{domain['end']})\n")
                    f.write(f"- Pfam ID: {domain['pfam_id']}\n")
                    f.write(f"- Function: {domain['function']}\n")

        research_log += "\n## Summary\n"
        research_log += f"- Detailed domain analysis saved to: {output_file}\n"
        if fas_domains_found:
            research_log += "- The sequence contains multiple domains characteristic of Fatty Acid Synthase\n"
            research_log += "- The identified domains suggest this sequence is involved in fatty acid biosynthesis\n"
        else:
            research_log += "- The sequence does not contain typical FAS domains\n"
            research_log += "- This may not be a Fatty Acid Synthase gene or may be a partial sequence\n"

    except Exception as e:
        research_log += f"- Error in domain identification: {str(e)}\n"

    return research_log
