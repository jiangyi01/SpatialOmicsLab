def _safe_filename_part(text):
    """``text`` made safe to embed in a single output filename.

    Strain, tissue and protein names arrive as free text, and a '/' in one ('E. coli K-12/MG1655') turned
    the output name into a path through a directory that does not exist -- so the save failed after the
    analysis had already run (hunt 2026-09-30, uT7-immunology-28).
    """
    import re

    return re.sub(r"[^\w.-]+", "_", str(text)).strip(".") or "unnamed"


def analyze_atac_seq_differential_accessibility(
    treatment_bam,
    control_bam,
    output_dir="./atac_results",
    genome_size="hs",
    q_value=0.05,
    name_prefix="atac",
):
    """Perform ATAC-seq peak calling and differential accessibility analysis using MACS2.

    Parameters
    ----------
    treatment_bam : str
        Path to the treatment condition BAM file with aligned ATAC-seq reads
    control_bam : str
        Path to the control condition BAM file with aligned ATAC-seq reads
    output_dir : str
        Directory to save output files (default: "./atac_results")
    genome_size : str
        Genome size parameter for MACS2 (default: "hs" for human)
    q_value : float
        q-value cutoff for peak detection (default: 0.05)
    name_prefix : str
        Prefix for output file names (default: "atac")

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import datetime
    import glob
    import os
    import re
    import subprocess

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize research log
    log = f"ATAC-seq Analysis Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += "=" * 80 + "\n\n"

    # This pipeline could never finish (hunt 2026-09-30, uT7-immunology-7): -n carried output_dir as well as
    # --outdir, so MACS2 wrote under output_dir/output_dir/; without -B callpeak writes no bedGraphs for
    # bdgdiff to read; and bdgdiff names its output <prefix>_c<cutoff>_cond1.bed, not <prefix>_cond1.bed.
    # -n is now the bare name, -B is passed, and every path is built from output_dir + that name.
    def run_macs2(cmd):
        """``None`` on success, else the error text for the log."""
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            return "macs2 is not installed or not on PATH"
        except subprocess.CalledProcessError as e:
            return e.stderr
        return None

    def filtered_tag_count(peaks_xls):
        """Reads after duplicate filtering, from the header callpeak writes into <name>_peaks.xls."""
        with open(peaks_xls) as f:
            header = f.read(20000)
        for pattern in (
            r"# (?:tag|fragment)s after filtering in treatment: (\d+)",
            r"# total (?:tag|fragment)s in treatment: (\d+)",
        ):
            found = re.search(pattern, header)
            if found:
                return int(found.group(1))
        return None

    names = {}
    peak_counts = {}
    depths = {}
    for step, (label, bam) in enumerate((("treatment", treatment_bam), ("control", control_bam)), start=1):
        # Step 1/2: Run MACS2 for peak calling on each sample
        log += f"Step {step}: Peak calling on {label} sample\n"
        name = f"{name_prefix}_{label}"
        cmd = [
            "macs2",
            "callpeak",
            "-t",
            bam,
            "-f",
            "BAM",
            "-g",
            genome_size,
            "-n",
            name,
            "--outdir",
            output_dir,
            "-B",
            "--nomodel",
            "--shift",
            "-100",
            "--extsize",
            "200",
            "-q",
            str(q_value),
        ]
        log += f"Command: {' '.join(cmd)}\n"
        error = run_macs2(cmd)
        if error is not None:
            log += f"Error in {label} peak calling: {error}\n\n"
            return log
        log += f"{label.capitalize()} peak calling completed successfully.\n"

        peaks_file = os.path.join(output_dir, f"{name}_peaks.narrowPeak")
        try:
            # Count peaks
            with open(peaks_file) as f:
                peak_counts[label] = sum(1 for _ in f)
            depths[label] = filtered_tag_count(os.path.join(output_dir, f"{name}_peaks.xls"))
        except OSError as e:
            log += f"Error: macs2 callpeak reported success but its output is missing: {e}\n\n"
            return log
        if not depths[label]:
            log += f"Error: could not read the filtered read count for the {label} sample from {name}_peaks.xls.\n\n"
            return log
        names[label] = name
        log += f"Identified {peak_counts[label]} peaks in {label} sample ({depths[label]} reads after filtering).\n\n"

    # Step 3: Perform differential accessibility analysis using MACS2 bdgdiff
    log += "Step 3: Differential accessibility analysis\n"
    diff_prefix = f"{name_prefix}_differential"
    diff_cmd = [
        "macs2",
        "bdgdiff",
        "--t1",
        os.path.join(output_dir, f"{names['treatment']}_treat_pileup.bdg"),
        "--c1",
        os.path.join(output_dir, f"{names['treatment']}_control_lambda.bdg"),
        "--t2",
        os.path.join(output_dir, f"{names['control']}_treat_pileup.bdg"),
        "--c2",
        os.path.join(output_dir, f"{names['control']}_control_lambda.bdg"),
        # Depths in millions of filtered reads, so bdgdiff scales the deeper sample down to the other
        "--d1",
        f"{depths['treatment'] / 1e6:g}",
        "--d2",
        f"{depths['control'] / 1e6:g}",
        "--outdir",
        output_dir,
        "--o-prefix",
        diff_prefix,
    ]

    log += f"Command: {' '.join(diff_cmd)}\n"
    error = run_macs2(diff_cmd)
    if error is not None:
        log += f"Error in differential analysis: {error}\n\n"
        return log
    log += "Differential accessibility analysis completed successfully.\n"

    # Count differential regions (bdgdiff appends the logLR cutoff: <prefix>_c3.0_cond1.bed)
    region_files = {}
    for condition in ("cond1", "cond2"):
        matches = sorted(
            glob.glob(os.path.join(glob.escape(output_dir), f"{glob.escape(diff_prefix)}_c*_{condition}.bed"))
        )
        if len(matches) != 1:
            log += f"Error: expected one {diff_prefix}_c*_{condition}.bed from bdgdiff in {output_dir}, "
            log += f"found {matches}\n\n"
            return log
        region_files[condition] = matches[0]
    with open(region_files["cond1"]) as f:
        treatment_enriched_count = sum(1 for line in f if not line.startswith(("track", "#")))
    with open(region_files["cond2"]) as f:
        control_enriched_count = sum(1 for line in f if not line.startswith(("track", "#")))

    log += f"Found {treatment_enriched_count} regions with higher accessibility in treatment.\n"
    log += f"Found {control_enriched_count} regions with higher accessibility in control.\n\n"

    # Step 4: Summary of results
    log += "Step 4: Analysis Summary\n"
    log += f"Total peaks in treatment: {peak_counts['treatment']}\n"
    log += f"Total peaks in control: {peak_counts['control']}\n"
    log += f"Differentially accessible regions: {treatment_enriched_count + control_enriched_count}\n"
    log += f"  - Enriched in treatment: {treatment_enriched_count}\n"
    log += f"  - Enriched in control: {control_enriched_count}\n\n"

    # Output file locations
    log += "Output Files:\n"
    log += f"  - Treatment peaks: {os.path.join(output_dir, names['treatment'] + '_peaks.narrowPeak')}\n"
    log += f"  - Control peaks: {os.path.join(output_dir, names['control'] + '_peaks.narrowPeak')}\n"
    log += f"  - Treatment-enriched regions: {region_files['cond1']}\n"
    log += f"  - Control-enriched regions: {region_files['cond2']}\n"

    return log


def analyze_bacterial_growth_curve(time_points, od_values, strain_name, output_dir="."):
    """Analyzes bacterial growth curve data to determine growth parameters.

    Parameters
    ----------
    time_points : list or numpy.ndarray
        Time points of measurements in hours
    od_values : list or numpy.ndarray
        Optical density measurements corresponding to each time point
    strain_name : str
        Name of the bacterial strain being analyzed
    output_dir : str, optional
        Directory where output files will be saved (default: current directory)

    Returns
    -------
    str
        A research log summarizing the analysis steps and results

    """
    import os
    from math import log

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert inputs to numpy arrays if they aren't already
    time_points = np.array(time_points)
    od_values = np.array(od_values)

    # Create a DataFrame for easier data handling
    pd.DataFrame({"Time (h)": time_points, "OD": od_values})

    # Define the logistic growth model function
    def logistic_growth(t, k, n0, r):
        """Logistic growth model: N(t) = K / (1 + ((K - N0) / N0) * exp(-r*t)).

        Parameters
        ----------
        t: time
        k: carrying capacity (maximum population size)
        n0: initial population
        r: growth rate

        """
        return k / (1 + ((k - n0) / n0) * np.exp(-r * t))

    # Initial parameter estimates
    p0 = [max(od_values), od_values[0], 0.5]

    # Fit the model to the data
    try:
        popt, pcov = curve_fit(logistic_growth, time_points, od_values, p0=p0)
        k_fit, n0_fit, r_fit = popt

        # Calculate doubling time (in hours)
        doubling_time = log(2) / r_fit

        # Calculate lag phase (approximation)
        lag_phase = (np.log((k_fit / n0_fit) - 1) - np.log((k_fit / (0.05 * k_fit)) - 1)) / r_fit
        lag_phase = max(0, lag_phase)  # Ensure non-negative

        # Generate fitted curve for plotting
        time_fine = np.linspace(min(time_points), max(time_points), 100)
        od_fitted = logistic_growth(time_fine, *popt)

        # Create the growth curve plot
        plt.figure(figsize=(10, 6))
        plt.scatter(time_points, od_values, label="Observed OD")
        plt.plot(time_fine, od_fitted, "r-", label="Fitted curve")
        plt.xlabel("Time (hours)")
        plt.ylabel("Optical Density (OD)")
        plt.title(f"Growth Curve for {strain_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save the plot
        plot_filename = os.path.join(output_dir, f"{_safe_filename_part(strain_name)}_growth_curve.png")
        plt.savefig(plot_filename)
        plt.close()

        # Create a research log
        log_text = f"""
Bacterial Growth Curve Analysis Log
==================================
Strain: {strain_name}

Data Summary:
------------
- Number of time points: {len(time_points)}
- Time range: {min(time_points):.1f} to {max(time_points):.1f} hours
- Initial OD: {od_values[0]:.4f}
- Final OD: {od_values[-1]:.4f}

Growth Model:
------------
- Model type: Logistic growth
- Fitted parameters:
  * Maximum population (carrying capacity, K): {k_fit:.4f} OD units
  * Initial population (N0): {n0_fit:.4f} OD units
  * Growth rate (r): {r_fit:.4f} per hour

Growth Metrics:
--------------
- Doubling time: {doubling_time:.2f} hours
- Maximum growth rate: {r_fit:.4f} per hour
- Lag phase duration: {lag_phase:.2f} hours
- Maximum OD (carrying capacity): {k_fit:.4f}

Output Files:
-----------
- Growth curve plot: {plot_filename}

Analysis completed successfully.
"""
        return log_text

    except RuntimeError:
        return f"Error: Could not fit growth model to data for {strain_name}. Please check your input data."


def isolate_purify_immune_cells(
    tissue_type,
    target_cell_type,
    enzyme_type="collagenase",
    macs_antibody=None,
    digestion_time_min=45,
    output_dir=".",
):
    """Simulates the isolation and purification of immune cells from tissue samples.

    Parameters
    ----------
    tissue_type : str
        The type of tissue sample (e.g., 'adipose', 'kidney', 'liver', 'lung', 'spleen')
    target_cell_type : str
        The immune cell population to isolate (e.g., 'macrophages', 'leukocytes', 'T cells')
    enzyme_type : str, optional
        The enzyme used for tissue digestion (default: 'collagenase')
    macs_antibody : str, optional
        Specific antibody for magnetic-assisted cell sorting (default: None, will be set based on target cell type)
    digestion_time_min : int, optional
        Digestion time in minutes (default: 45)
    output_dir : str, optional
        Directory to save the simulated cell-count CSV (default: current directory)

    Returns
    -------
    str
        A research log describing the cell isolation and purification process

    """
    import os
    from datetime import datetime

    import numpy as np
    import pandas as pd

    # Initialize research log
    log = []
    log.append(f"CELL ISOLATION AND PURIFICATION LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append(f"Tissue type: {tissue_type}")
    log.append(f"Target cell population: {target_cell_type}")
    log.append("-" * 50)

    # Set default MACS antibody if not provided
    if macs_antibody is None:
        if target_cell_type.lower() == "macrophages":
            macs_antibody = "CD11b"
        elif target_cell_type.lower() == "t cells":
            macs_antibody = "CD3"
        elif target_cell_type.lower() == "b cells":
            macs_antibody = "CD19"
        else:
            macs_antibody = "CD45"  # General leukocyte marker

    # Step 1: Tissue preparation
    log.append("1. TISSUE PREPARATION")
    log.append(f"   - {tissue_type.capitalize()} tissue was collected and placed in cold PBS")
    log.append("   - Tissue was minced into small pieces (1-2 mm) using sterile scissors")

    # Step 2: Enzymatic digestion
    log.append("\n2. ENZYMATIC DIGESTION")
    log.append(
        f"   - Tissue fragments were incubated in {enzyme_type} solution at 37°C for {digestion_time_min} minutes"
    )
    log.append("   - Gentle agitation was applied every 15 minutes to enhance digestion")

    # Simulate cell count after digestion
    initial_cell_count = np.random.randint(1e6, 1e7)
    log.append(f"   - Cell count after digestion: {initial_cell_count:,} cells")

    # Step 3: Filtration/Cell straining
    log.append("\n3. FILTRATION/CELL STRAINING")
    log.append("   - Cell suspension was filtered through a 70 μm cell strainer")
    log.append("   - Additional washing with PBS was performed to maximize cell recovery")

    # Simulate cell count after filtration
    post_filtration_count = int(initial_cell_count * np.random.uniform(0.7, 0.9))
    log.append(f"   - Cell count after filtration: {post_filtration_count:,} cells")

    # Step 4: Density gradient centrifugation
    log.append("\n4. DENSITY GRADIENT CENTRIFUGATION")
    log.append("   - Cell suspension was carefully layered over Ficoll-Paque medium")
    log.append("   - Centrifugation was performed at 400 × g for 30 minutes at room temperature")
    log.append("   - The interface layer containing mononuclear cells was collected")

    # Simulate cell count after density gradient
    post_gradient_count = int(post_filtration_count * np.random.uniform(0.3, 0.6))
    log.append(f"   - Cell count after density gradient: {post_gradient_count:,} cells")

    # Step 5: Magnetic-assisted cell sorting (MACS)
    log.append("\n5. MAGNETIC-ASSISTED CELL SORTING (MACS)")
    log.append(f"   - Cells were labeled with anti-{macs_antibody} magnetic microbeads")
    log.append("   - Labeled cell suspension was passed through a MACS column in a magnetic field")
    log.append(f"   - {target_cell_type.capitalize()} were collected based on their binding to the magnetic beads")

    # Simulate final purified cell count
    final_cell_count = int(post_gradient_count * np.random.uniform(0.1, 0.3))
    purity = np.random.uniform(0.85, 0.98)
    log.append(f"   - Final cell count: {final_cell_count:,} {target_cell_type}")
    log.append(f"   - Estimated purity: {purity:.1%}")

    # Step 6: Quality assessment
    log.append("\n6. QUALITY ASSESSMENT")
    log.append("   - Cell viability was assessed using trypan blue exclusion")
    viability = np.random.uniform(0.85, 0.98)
    log.append(f"   - Cell viability: {viability:.1%}")
    log.append("   - Purity was confirmed by flow cytometry analysis")

    # Create a summary table and save as CSV
    cell_data = {
        "Process Step": [
            "Initial",
            "Post-Filtration",
            "Post-Gradient",
            "Final Purified",
        ],
        "Cell Count": [
            initial_cell_count,
            post_filtration_count,
            post_gradient_count,
            final_cell_count,
        ],
        "Viability (%)": [
            np.random.uniform(0.7, 0.9) * 100,
            np.random.uniform(0.75, 0.92) * 100,
            np.random.uniform(0.8, 0.95) * 100,
            viability * 100,
        ],
    }

    df = pd.DataFrame(cell_data)
    # Under output_dir and reported absolute: the bare relative name landed in the REPL's working directory,
    # outside the chat's outputs (hunt 2026-09-30, uT7-immunology-21), and a '/' in a free-text tissue or
    # cell type made it a path into a directory that does not exist (uT7-immunology-28).
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.abspath(
        os.path.join(
            output_dir,
            f"{_safe_filename_part(tissue_type)}_{_safe_filename_part(target_cell_type)}_isolation_data.csv",
        )
    )
    df.to_csv(csv_filename, index=False)

    log.append(f"\nCell count data saved to: {csv_filename}")
    log.append("\nPURIFICATION COMPLETE")

    return "\n".join(log)


def estimate_cell_cycle_phase_durations(flow_cytometry_data, initial_estimates):
    """
    Estimate cell cycle phase durations using dual-nucleoside pulse labeling data and mathematical modeling.

    Parameters
    ----------
    flow_cytometry_data : dict
        Dictionary containing experimental data from flow cytometry with EdU and BrdU labeling.
        Expected format::

            {
                'time_points': list of time points (hours),
                'edu_positive': list of percentages of EdU+ cells at each time point,
                'brdu_positive': list of percentages of BrdU+ cells at each time point,
                'double_positive': list of percentages of EdU+BrdU+ cells at each time point
            }

    initial_estimates : dict
        Initial estimates for cell cycle phase durations and death rates.
        Expected format::

            {
                'g1_duration': float (hours),
                's_duration': float (hours),
                'g2m_duration': float (hours),
                'death_rate': float (fraction per hour)
            }

    Returns
    -------
    str
        Research log summarizing the cell cycle phase duration estimation process and results.
    """

    import time

    import numpy as np
    from scipy import optimize

    # Start research log
    log = "# Cell Cycle Phase Duration Estimation Research Log\n\n"
    log += f"Analysis started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Log input data summary
    log += "## Input Data Summary\n"
    log += f"- Number of time points: {len(flow_cytometry_data['time_points'])}\n"
    log += (
        f"- Time range: {min(flow_cytometry_data['time_points'])} to {max(flow_cytometry_data['time_points'])} hours\n"
    )
    log += f"- Initial estimates: G1={initial_estimates['g1_duration']}h, S={initial_estimates['s_duration']}h, "
    log += f"G2/M={initial_estimates['g2m_duration']}h, Death rate={initial_estimates['death_rate']}\n\n"

    # Define the objective function for optimization
    def objective_function(params):
        g1_duration, s_duration, g2m_duration, death_rate = params

        # Simple cell cycle model simulation
        simulated_results = simulate_cell_population(
            flow_cytometry_data["time_points"],
            g1_duration,
            s_duration,
            g2m_duration,
            death_rate,
        )

        # Calculate error between simulated and experimental data
        edu_error = np.sum(
            (np.array(simulated_results["edu_positive"]) - np.array(flow_cytometry_data["edu_positive"])) ** 2
        )
        brdu_error = np.sum(
            (np.array(simulated_results["brdu_positive"]) - np.array(flow_cytometry_data["brdu_positive"])) ** 2
        )
        double_error = np.sum(
            (np.array(simulated_results["double_positive"]) - np.array(flow_cytometry_data["double_positive"])) ** 2
        )

        # Total error
        total_error = edu_error + brdu_error + double_error
        return total_error

    # Function to simulate cell population dynamics
    def simulate_cell_population(time_points, g1_duration, s_duration, g2m_duration, death_rate):
        # Simple simulation of cell populations based on ODE model
        # This is a simplified version - a real implementation would use differential equations

        # Initialize results
        edu_positive = []
        brdu_positive = []
        double_positive = []

        total_cycle_time = g1_duration + s_duration + g2m_duration

        for t in time_points:
            # Simplified model for demonstration
            # In a real implementation, this would involve solving ODEs

            # Calculate fractions of cells in each phase
            s_fraction = s_duration / total_cycle_time

            # Simple model for EdU and BrdU incorporation
            edu_pos = s_fraction * np.exp(-death_rate * t)
            brdu_pos = s_fraction * (1 - np.exp(-t / s_duration))
            double_pos = s_fraction * np.exp(-death_rate * t) * (1 - np.exp(-t / s_duration))

            # Adjust values to be realistic
            edu_pos = min(edu_pos, 1.0) * 100  # Convert to percentage
            brdu_pos = min(brdu_pos, 1.0) * 100
            double_pos = min(double_pos, 1.0) * 100

            edu_positive.append(edu_pos)
            brdu_positive.append(brdu_pos)
            double_positive.append(double_pos)

        return {
            "edu_positive": edu_positive,
            "brdu_positive": brdu_positive,
            "double_positive": double_positive,
        }

    # Set up initial parameter values and bounds for optimization
    initial_params = [
        initial_estimates["g1_duration"],
        initial_estimates["s_duration"],
        initial_estimates["g2m_duration"],
        initial_estimates["death_rate"],
    ]

    # Parameter bounds (all positive values)
    bounds = [(0.1, 50.0), (0.1, 30.0), (0.1, 20.0), (0.0, 1.0)]

    log += "## Optimization Process\n"
    log += "Starting parameter optimization using SciPy's L-BFGS-B algorithm...\n\n"

    # Run optimization
    optimization_start = time.time()
    result = optimize.minimize(objective_function, initial_params, method="L-BFGS-B", bounds=bounds)
    optimization_time = time.time() - optimization_start

    # Extract optimized parameters
    optimized_g1, optimized_s, optimized_g2m, optimized_death = result.x

    # Calculate final error
    final_error = result.fun

    # Log optimization results
    log += "## Optimization Results\n"
    log += f"- Optimization completed in {optimization_time:.2f} seconds\n"
    log += f"- Optimization success: {result.success}\n"
    log += f"- Final error value: {final_error:.4f}\n\n"

    # The model sees G1 and G2/M only through their sum (total_cycle_time); every split of that sum fits
    # equally well, so the separate G1 and G2/M "estimates" this used to print were whatever the optimizer's
    # starting point left them at. Only S, G1 + G2/M and the total are reported (hunt 2026-09-30,
    # uT7-immunology-18).
    optimized_g1_g2m = optimized_g1 + optimized_g2m
    initial_g1_g2m = initial_estimates["g1_duration"] + initial_estimates["g2m_duration"]

    log += "## Estimated Cell Cycle Phase Durations\n"
    log += f"- S phase: {optimized_s:.2f} hours\n"
    log += f"- G1 + G2/M phases (combined): {optimized_g1_g2m:.2f} hours\n"
    log += f"- Total cell cycle time: {optimized_g1_g2m + optimized_s:.2f} hours\n"
    log += f"- Cell death rate: {optimized_death:.4f} per hour\n"
    log += (
        "- G1 and G2/M are not separately identifiable: the labelling model depends on them only through "
        "their sum, so this data cannot split the combined value between them.\n\n"
    )

    # Compare with initial estimates
    log += "## Comparison with Initial Estimates\n"
    log += f"- S phase: {optimized_s:.2f}h (initial: {initial_estimates['s_duration']}h, "
    log += f"change: {(optimized_s - initial_estimates['s_duration']) / initial_estimates['s_duration'] * 100:.1f}%)\n"

    log += f"- G1 + G2/M phases: {optimized_g1_g2m:.2f}h (initial: {initial_g1_g2m}h, "
    log += f"change: {(optimized_g1_g2m - initial_g1_g2m) / initial_g1_g2m * 100:.1f}%)\n"

    log += f"- Death rate: {optimized_death:.4f} (initial: {initial_estimates['death_rate']}, "
    log += f"change: {(optimized_death - initial_estimates['death_rate']) / (initial_estimates['death_rate'] if initial_estimates['death_rate'] > 0 else 1) * 100:.1f}%)\n\n"

    # Final summary
    log += "## Conclusion\n"
    log += "The mathematical modeling and parameter optimization has estimated the cell cycle phase durations "
    log += "based on the provided dual-nucleoside pulse labeling data. "
    log += "These estimates provide insight into the proliferation dynamics of the studied cell population.\n\n"

    log += f"Analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"

    return log


def track_immune_cells_under_flow(
    image_sequence_path,
    output_dir="./output",
    pixel_size_um=1.0,
    time_interval_sec=1.0,
    flow_direction="right",
):
    """Track immune cells under flow conditions and classify their behaviors.

    Args:
        image_sequence_path (str): Path to image sequence directory or video file.
        output_dir (str): Directory to save output files.
        pixel_size_um (float): Pixel size in micrometers.
        time_interval_sec (float): Time interval between frames in seconds.
        flow_direction (str): Direction of flow ('right', 'left', 'up', 'down').

    Returns:
        str: Log of the analysis process.

    """
    import os

    import cv2
    import numpy as np
    import pandas as pd
    import trackpy as tp

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize log
    log = "# Immune Cell Tracking Under Flow Analysis\n\n"
    log += "## Parameters\n"
    log += f"- Pixel size: {pixel_size_um} μm\n"
    log += f"- Time interval: {time_interval_sec} sec\n"
    log += f"- Flow direction: {flow_direction}\n\n"

    # Load image sequence. A frame that cannot be read is an error, not a blank or copied stand-in: those
    # stand-ins fed dummy features and then simulated tracks into a log that ended "Analysis completed
    # successfully" (hunt 2026-09-30, uT7-immunology-2).
    log += "## Data Loading\n"
    if os.path.isdir(image_sequence_path):
        image_files = sorted(
            [f for f in os.listdir(image_sequence_path) if f.endswith((".png", ".jpg", ".tif", ".tiff"))]
        )
        if not image_files:
            return log + f"Error: no .png/.jpg/.tif/.tiff images in {image_sequence_path}\n"
        log += f"- Loaded {len(image_files)} images from directory\n"

        frames = []
        for img_file in image_files:
            img = cv2.imread(os.path.join(image_sequence_path, img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return log + f"Error: could not read {os.path.join(image_sequence_path, img_file)} as an image\n"
            frames.append(img)
    else:
        # Assume it's a video file
        cap = cv2.VideoCapture(image_sequence_path)
        if not cap.isOpened():
            return log + f"Error: could not open {image_sequence_path} as a video or an image directory\n"
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        log += f"- Loaded {len(frames)} frames from video\n"
        if not frames:
            return log + f"Error: no frame could be read from {image_sequence_path}\n"

    first_img = frames[0]
    log += f"- Image dimensions: {first_img.shape[1]}x{first_img.shape[0]} pixels\n\n"

    # Cell segmentation and feature extraction
    log += "## Cell Segmentation\n"
    features_by_frame = []

    for i, frame in enumerate(frames):
        # Enhance contrast
        frame_eq = cv2.equalizeHist(frame)

        # Apply Gaussian blur to reduce noise
        frame_blur = cv2.GaussianBlur(frame_eq, (5, 5), 0)

        # Adaptive thresholding to identify cells
        thresh = cv2.adaptiveThreshold(
            frame_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )

        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Identify cells using connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Filter out small objects (noise) and large objects (not cells)
        min_size = 20  # Minimum cell area in pixels
        max_size = 500  # Maximum cell area in pixels

        # Extract features for each cell
        frame_features = []
        for j in range(1, num_labels):  # Skip label 0 (background)
            area = stats[j, cv2.CC_STAT_AREA]
            if min_size <= area <= max_size:
                x = centroids[j, 0]
                y = centroids[j, 1]
                width = stats[j, cv2.CC_STAT_WIDTH]
                height = stats[j, cv2.CC_STAT_HEIGHT]

                # Calculate cell roundness (approximation)
                roundness = min(width, height) / max(width, height) if max(width, height) > 0 else 0

                frame_features.append(
                    {
                        "frame": i,
                        "y": y,
                        "x": x,
                        "area": area,
                        "width": width,
                        "height": height,
                        "roundness": roundness,
                    }
                )

        features_by_frame.append(pd.DataFrame(frame_features))

    # Combine all features
    all_features = pd.concat(features_by_frame, ignore_index=True)
    log += f"- Detected {len(all_features)} cell instances across all frames\n"
    log += f"- Average {len(all_features) / len(frames):.1f} cells per frame\n\n"
    if len(all_features) == 0:
        return (
            log + "Error: no cells (connected components of 20-500 px) were detected in any frame; nothing to track\n"
        )

    # Cell tracking
    log += "## Cell Tracking\n"

    # Use trackpy to link cell positions across frames
    search_range = 20  # Maximum distance a cell can move between frames
    memory = 3  # Allow linking across gaps of up to 3 frames
    min_track_length = 5  # Minimum number of frames a cell must be tracked

    # trackpy keys time on the 'frame' column. Renaming it to 't' made tp.link raise KeyError on every
    # run, and the except swapped in round-robin "simulated" tracks (uT7-immunology-2).
    linked = tp.link(all_features, search_range, memory=memory)
    log += f"- Identified {linked['particle'].nunique()} cell trajectories\n"

    # Filter tracks that are too short. filter_stubs indexes its result by 'frame' while keeping the
    # column, so sorting a track by 'frame' below raised "ambiguous" for every track (uT7-immunology-2).
    tracks = tp.filter_stubs(linked, min_track_length).reset_index(drop=True)
    track_ids = tracks["particle"].unique()
    log += f"- Filtered to {len(track_ids)} trajectories of length ≥ {min_track_length} frames\n\n"
    if len(track_ids) == 0:
        return log + (
            f"Error: no cell was tracked for {min_track_length} or more frames (search range {search_range} px), "
            "so no behaviour can be classified\n"
        )

    # Analyze cell behaviors
    log += "## Cell Behavior Classification\n"

    # Define behavior thresholds
    speed_threshold = 5.0 * pixel_size_um / time_interval_sec  # μm/s
    arrest_time_threshold = 5  # frames

    # Calculate displacement and speed for each track
    behaviors = []

    for track_id in track_ids:
        try:
            track_data = tracks[tracks["particle"] == track_id].sort_values("frame")

            # Skip tracks that are too short
            if len(track_data) < 5:  # Using fixed value instead of min_track_length
                continue

            # Calculate displacements between consecutive frames
            track_data["dx"] = track_data["x"].diff()
            track_data["dy"] = track_data["y"].diff()
            track_data["displacement"] = np.sqrt(track_data["dx"] ** 2 + track_data["dy"] ** 2)
            track_data["speed"] = track_data["displacement"] * pixel_size_um / time_interval_sec

            # Calculate direction relative to flow
            if flow_direction == "right":
                track_data["flow_alignment"] = track_data["dx"] / (track_data["displacement"] + 1e-6)
            elif flow_direction == "left":
                track_data["flow_alignment"] = -track_data["dx"] / (track_data["displacement"] + 1e-6)
            elif flow_direction == "down":
                track_data["flow_alignment"] = track_data["dy"] / (track_data["displacement"] + 1e-6)
            else:  # up
                track_data["flow_alignment"] = -track_data["dy"] / (track_data["displacement"] + 1e-6)

            # Classify behaviors for each time point
            track_data["behavior"] = "unknown"

            # Tethering/Rolling: Moving with flow direction, moderate speed
            rolling_mask = (track_data["flow_alignment"] > 0.7) & (track_data["speed"] < speed_threshold)
            track_data.loc[rolling_mask, "behavior"] = "rolling"

            # Arrest: Very low speed for multiple frames
            arrest_mask = track_data["speed"] < (speed_threshold * 0.2)

            # Find continuous arrest periods
            arrest_periods = []
            current_period = []

            for i, is_arrest in enumerate(arrest_mask):
                if is_arrest:
                    current_period.append(i)
                elif current_period:
                    if len(current_period) >= arrest_time_threshold:
                        arrest_periods.append(current_period)
                    current_period = []

            if current_period and len(current_period) >= arrest_time_threshold:
                arrest_periods.append(current_period)

            # Mark arrest periods
            for period in arrest_periods:
                track_data.loc[track_data.index[period], "behavior"] = "arrest"

            # Crawling: After arrest, moving slowly with changing directions
            for period in arrest_periods:
                if period[-1] + 1 < len(track_data):
                    crawl_start = period[-1] + 1
                    for i in range(crawl_start, len(track_data)):
                        if track_data.iloc[i]["speed"] < speed_threshold * 0.5:
                            track_data.loc[track_data.index[i], "behavior"] = "crawling"
                        else:
                            break

            # Diapedesis: Detected by significant change in morphology (roundness decreases)
            if "roundness" in track_data.columns:
                diapedesis_mask = (track_data["behavior"] == "crawling") & (track_data["roundness"] < 0.5)
                track_data.loc[diapedesis_mask, "behavior"] = "diapedesis"

            # Add to behaviors list
            behaviors.append(track_data)
        except Exception as e:
            log += f"- Warning: Error processing track {track_id}: {str(e)}\n"
            continue

    # Check if we have any valid behaviors
    if behaviors:
        # Combine all behaviors
        all_behaviors = pd.concat(behaviors, ignore_index=True)

        # Count behaviors
        behavior_counts = all_behaviors["behavior"].value_counts()
        log += "- Behavior classification results:\n"
        for behavior, count in behavior_counts.items():
            log += f"  - {behavior}: {count} instances ({count / len(all_behaviors) * 100:.1f}%)\n"

        # Save results
        trajectories_file = os.path.join(output_dir, "cell_trajectories.csv")
        all_behaviors.to_csv(trajectories_file, index=False)
        log += "\n## Results\n"
        log += f"- Saved cell trajectories with behavior classifications to: {trajectories_file}\n"

        # Summary statistics
        avg_track_length = all_behaviors.groupby("particle").size().mean()
        avg_speed = all_behaviors["speed"].mean()

        log += "\n## Summary Statistics\n"
        log += f"- Average track length: {avg_track_length:.1f} frames\n"
        log += f"- Average cell speed: {avg_speed:.2f} μm/s\n"
    else:
        return log + "Error: every trajectory failed behaviour classification (see the warnings above)\n"

    log += "- Analysis completed successfully\n"

    return log


def analyze_cfse_cell_proliferation(fcs_file_path, cfse_channel="FL1-A", lymphocyte_gate=None):
    """Analyze CFSE-labeled cell samples to quantify cell division and proliferation.

    This function processes flow cytometry data from CFSE-labeled cells to calculate
    the cell division index and percentage of proliferating cells. It performs gating,
    identifies cell populations based on CFSE intensity, and quantifies proliferation metrics.

    Parameters
    ----------
    fcs_file_path : str
        Path to the FCS file containing flow cytometry data from CFSE-labeled cells
    cfse_channel : str, optional
        Name of the channel containing CFSE fluorescence data (default: 'FL1-A')
    lymphocyte_gate : tuple or None, optional
        Tuple of (min_fsc, max_fsc, min_ssc, max_ssc) for lymphocyte gating
        If None, automatic gating will be attempted (default: None)

    Returns
    -------
    str
        Research log summarizing the analysis steps and results, including cell division
        index and percentage of proliferating cells

    """
    import os

    import numpy as np

    research_log = []
    research_log.append("# CFSE-based Cell Proliferation Assay Analysis Log")
    research_log.append(f"## Data Source: {os.path.basename(fcs_file_path)}")

    # Every exit below without metrics is an error that says why. This tool used to fall back to a mock
    # histogram when FlowCytometryTools would not import, and to "reasonable" 1.2 / 65% when the analysis
    # raised -- both printed under "## Results:" for any input, including a path that does not exist
    # (hunt 2026-09-30, uT7-immunology-1).
    if not os.path.isfile(fcs_file_path):
        return f"Error: FCS file not found: {fcs_file_path}. No proliferation metrics were computed."

    try:
        # First attempt to patch collections.MutableMapping for compatibility
        import collections
        import collections.abc

        if not hasattr(collections, "MutableMapping"):
            collections.MutableMapping = collections.abc.MutableMapping

        from FlowCytometryTools import FCMeasurement
    except Exception as e:
        return (
            f"Error: FlowCytometryTools could not be imported ({type(e).__name__}: {e}). This tool reads the FCS "
            "file with it, so no proliferation metrics were computed; install FlowCytometryTools in this "
            "environment, or read the FCS file with another library."
        )

    # Load FCS data
    research_log.append("\n## Step 1: Loading Flow Cytometry Data")

    try:
        sample = FCMeasurement(ID="CFSE_Sample", datafile=fcs_file_path)
        data = sample.data
        research_log.append(f"Successfully loaded data with {len(data)} events")

        # Apply lymphocyte gate if provided. Gated with boolean masks over the event table:
        # FCMeasurement.gate() takes a gate object, and the f-string gates passed here raised TypeError on
        # every call (hunt 2026-09-30, uT7-immunology-3).
        research_log.append("\n## Step 2: Gating Lymphocyte Population")
        fsc, ssc = data["FSC-A"], data["SSC-A"]
        if lymphocyte_gate:
            min_fsc, max_fsc, min_ssc, max_ssc = lymphocyte_gate
            gated_data = data[(fsc > min_fsc) & (fsc < max_fsc) & (ssc > min_ssc) & (ssc < max_ssc)]
            research_log.append(
                f"Applied manual lymphocyte gate: FSC-A ({min_fsc}-{max_fsc}), SSC-A ({min_ssc}-{max_ssc})"
            )
            research_log.append(f"Gated population contains {len(gated_data)} events")
        else:
            # Simple automatic gating based on FSC and SSC
            fsc_median = np.median(fsc)
            ssc_median = np.median(ssc)
            gated_data = data[
                (fsc > fsc_median * 0.5)
                & (fsc < fsc_median * 1.8)
                & (ssc > ssc_median * 0.5)
                & (ssc < ssc_median * 1.8)
            ]
            research_log.append("Applied automatic lymphocyte gate based on median FSC-A and SSC-A values")
            research_log.append(f"Gated population contains {len(gated_data)} events")

        # Extract CFSE data
        research_log.append("\n## Step 3: Analyzing CFSE Intensity Distribution")
        if cfse_channel not in gated_data.columns:
            available_channels = ", ".join(map(str, data.columns))
            return f"Error: CFSE channel '{cfse_channel}' not found. Available channels: {available_channels}"
        cfse_data = gated_data[cfse_channel].to_numpy(dtype=float)
        research_log.append(f"Extracted CFSE intensity data from channel {cfse_channel}")
        if len(cfse_data) == 0:
            research_log.append("Error: the lymphocyte gate kept 0 events. No proliferation metrics were computed.")
            return "\n".join(research_log)

        # Identify generations based on CFSE intensity
        # CFSE intensity halves with each cell division
        research_log.append("\n## Step 4: Identifying Cell Generations")

        # Log transform CFSE data for better separation of peaks
        log_cfse = np.log10(np.clip(cfse_data, 0, None) + 1)  # +1 to avoid log(0)

        # Find the generation peaks on a lightly smoothed histogram: raw bin-to-bin noise made every local
        # maximum a "generation". Bin centres, not left edges, are the peak positions.
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks

        hist, bin_edges = np.histogram(log_cfse, bins=100)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed = gaussian_filter1d(hist.astype(float), sigma=1.5)
        peak_indices, _ = find_peaks(smoothed, prominence=0.05 * smoothed.max())

        if len(peak_indices) == 0:
            research_log.append(
                "Error: no distinct peak found in the CFSE intensity distribution, so the undivided "
                "population cannot be located. No proliferation metrics were computed."
            )
            return "\n".join(research_log)

        # Sort peaks by intensity (highest CFSE = undivided cells)
        sorted_peaks = np.sort(bin_centres[peak_indices])[::-1]  # Descending order

        # Everything below rests on the brightest DETECTED peak being the undivided generation. A peak the
        # finder misses -- an undivided population under the 5 % prominence cut, or broad neighbours merged
        # into one -- renumbers every generation and moves the division index with no error (a 1 % undivided
        # peak gave 2.54 for a true 3.53). The index is not re-derived; the assumption is stated and the two
        # signatures of a missed peak are reported next to it (hunt 2026-09-30, uT7-immunology-1).
        one_division = np.log10(2)  # CFSE halves per division: adjacent generations sit log10(2) apart
        caveats = [
            f"Assumption: the brightest detected peak (CFSE {10 ** sorted_peaks[0]:.2f}) is the undivided "
            "generation (Gen 0). Peaks are found on a smoothed histogram and must stand out by 5% of its "
            "tallest bin, so a smaller undivided population is not detected; if one exists, every generation "
            "here is numbered one too low and the division index is understated."
        ]
        # A brighter generation the finder missed leaves its events more than half a division above the
        # brightest peak. One generation alone puts under 0.3x its own upper half there at CV <= 25 %
        # (sd <= 0.11 log10, measured on synthetic generations); a missed 1 % undivided peak puts 0.5x and up.
        # Past that width one broad generation crosses the line by itself (an all-undivided sample at sd
        # 0.13-0.15 gave 0.44-0.60x), so the warning says so rather than blaming a missed peak outright
        # (hunt 2026-09-30, uT7-immunology-1, second review).
        above = int(np.sum(log_cfse > sorted_peaks[0] + one_division / 2))
        upper_half = int(np.sum((log_cfse >= sorted_peaks[0]) & (log_cfse <= sorted_peaks[0] + one_division / 2)))
        excess = above / max(upper_half, 1)
        if excess > 0.4:
            caveats.append(
                f"Warning: {above} events ({above / len(log_cfse) * 100:.1f}% of gated) lie more than half a "
                f"division above the brightest detected peak -- {excess:.2f}x the events in that peak's own "
                "upper half, where one generation of sd up to 0.11 log10 (CV 26%) puts under 0.3x. A brighter "
                "generation (a small undivided population, or broad peaks merged into one) was probably not "
                "detected; if so, every generation is numbered one too low and the division index is "
                "understated. A single broader peak does this too -- an all-undivided sample at sd 0.13-0.15 "
                "log10 (CV 31-36%) reached 0.44-0.60x -- so check this peak's width before trusting either."
            )
        # Adjacent detected peaks should be one division apart. Measured spacings on resolved synthetic
        # generations stay within 0.75-1.25x; ~2x is a generation the finder did not separate, ~0.5x one
        # generation split into two peaks.
        for hi, lo in zip(sorted_peaks[:-1], sorted_peaks[1:], strict=True):
            ratio = (hi - lo) / one_division
            if not 2 / 3 <= ratio <= 3 / 2:
                caveats.append(
                    f"Warning: adjacent peaks at CFSE {10**hi:.2f} and {10**lo:.2f} are {hi - lo:.3f} log10 "
                    f"units apart, {ratio:.2f}x the log10(2) = {one_division:.3f} of one division. Near 2x, a "
                    "generation between them was not detected (merged or too small); near 0.5x, one "
                    "generation was split into two peaks. The generation counts and the division index "
                    "count each detected peak as one division."
                )

        if len(sorted_peaks) == 1:
            # Only one peak - assume it's undivided cells
            undivided_peak = sorted_peaks[0]
            research_log.append(f"Detected single peak at CFSE intensity {10**undivided_peak:.2f}")

            # Estimate threshold for proliferating cells (half the undivided intensity, one division)
            proliferation_threshold = 10 ** (undivided_peak - 0.3)
            proliferating_cells = np.sum(cfse_data < proliferation_threshold)
            total_cells = len(cfse_data)
            percent_proliferating = (proliferating_cells / total_cells) * 100

            # Simplified division index calculation
            division_index = percent_proliferating / 100 * 1  # Assume average of 1 division

            research_log.append("Single peak detected, using threshold-based estimation for proliferation")
            # "Lower bounds" held only for a narrow peak: the cut also takes in the undivided peak's own lower
            # tail, which on an all-undivided sample read as 0.5-2.4 % proliferating at sd 0.13-0.15 (hunt
            # 2026-09-30, uT7-immunology-1, second review). The numbers are unchanged; the claim is qualified.
            caveats.append(
                "With one peak the generations cannot be counted. Proliferating cells are those a full "
                "division (log10 2) or more below the peak, which leaves out the brighter half of the first "
                "divided generation; the division index is that fraction with every proliferating cell "
                "counted as ONE division. They are lower bounds only while the peak is narrow: the cut also "
                "takes in the undivided peak's own lower tail, which on an all-undivided sample read as "
                "0.2-0.3% proliferating at sd 0.11 log10 (CV 26%), 0.5-1.1% at 0.13 and 1.1-2.4% at 0.15 "
                "(CV 36%), so a percentage of that order can be no proliferation at all."
            )
        else:
            # Multiple peaks - can identify generations
            undivided_peak = sorted_peaks[0]
            research_log.append("Detected multiple peaks in CFSE intensity distribution")
            research_log.append(
                f"Undivided cell peak (assumed: the brightest detected peak) at CFSE intensity {10**undivided_peak:.2f}"
            )

            # Generation g spans from the midpoint below its peak to the midpoint above it. Generation 0
            # used to start AT its peak, so the lower half of the undivided population was counted nowhere.
            generation_boundaries = [
                10 ** ((hi + lo) / 2) for hi, lo in zip(sorted_peaks[:-1], sorted_peaks[1:], strict=True)
            ]

            # Count cells in each generation
            generation_counts = [int(np.sum(cfse_data >= generation_boundaries[0]))]  # Gen 0
            for upper, lower in zip(generation_boundaries[:-1], generation_boundaries[1:], strict=True):
                generation_counts.append(int(np.sum((cfse_data < upper) & (cfse_data >= lower))))
            generation_counts.append(int(np.sum(cfse_data < generation_boundaries[-1])))  # Most divided

            # Calculate division index and percent proliferating
            total_cells = sum(generation_counts)
            division_index = sum(i * count for i, count in enumerate(generation_counts)) / total_cells
            percent_proliferating = sum(generation_counts[1:]) / total_cells * 100

            research_log.append(f"Identified {len(generation_counts)} cell generations")
            research_log.append(
                f"Generation distribution: {', '.join([f'Gen {i}: {count} cells' for i, count in enumerate(generation_counts)])}"
            )
    except Exception as e:
        research_log.append(f"Error during analysis: {type(e).__name__}: {e}")
        research_log.append("No proliferation metrics were computed.")
        return "\n".join(research_log)

    # Report results
    research_log.append("\n## Results:")
    research_log.append(f"Cell Division Index: {division_index:.2f}")
    research_log.append(f"Percentage of Proliferating Cells: {percent_proliferating:.2f}%")
    research_log.append("\n## Caveats:")
    research_log.extend(f"- {caveat}" for caveat in caveats)

    return "\n".join(research_log)


def analyze_cytokine_production_in_cd4_tcells(fcs_files_dict, output_dir="./results"):
    """Analyze cytokine production (IFN-γ, IL-17) in CD4+ T cells after antigen stimulation.

    Parameters
    ----------
    fcs_files_dict : dict
        Dictionary mapping stimulation conditions to FCS file paths.
        Expected keys: 'unstimulated', 'Mtb300', 'CMV', 'SEB'
        Example: {'unstimulated': 'path/to/unstim.fcs', 'Mtb300': 'path/to/mtb.fcs'}

    output_dir : str, optional
        Directory to save the results file (default: './results')

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os
    import re

    import pandas as pd
    from FlowCytometryTools import FCMeasurement

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log = "# Cytokine Production Analysis by Flow Cytometry\n\n"
    log += "## Data Loading and Preprocessing\n"

    results = {}

    # Check required stimulation conditions
    required_conditions = ["unstimulated", "Mtb300", "CMV", "SEB"]
    missing_conditions = [cond for cond in required_conditions if cond not in fcs_files_dict]
    if missing_conditions:
        log += f"WARNING: Missing data for conditions: {', '.join(missing_conditions)}\n"

    # Process each stimulation condition
    for condition, fcs_file in fcs_files_dict.items():
        log += f"\nProcessing {condition} condition from file: {os.path.basename(fcs_file)}\n"

        # Load FCS file
        sample = FCMeasurement(ID=condition, datafile=fcs_file)

        # Compensation is not applied. FCMeasurement has no compensate(); the call always raised, and the
        # except reported "No compensation matrix found" even for files that carry one (hunt 2026-09-30,
        # uT7-immunology-3).
        meta = getattr(sample, "meta", None) or {}
        if any(str(key).lstrip("$").upper() in ("SPILLOVER", "SPILL", "COMP") for key in meta):
            log += (
                "- The FCS file carries a spillover matrix, but this tool does not apply it: data are uncompensated\n"
            )
        else:
            log += "- No spillover matrix in the FCS file; data are uncompensated\n"

        # Transform data (typically logicle transformation for cytometry data)
        channels = sample.channel_names
        cytokine_channels = [ch for ch in channels if "IFN" in ch or "IL-17" in ch]
        # CD4 as a whole marker name: a substring match took 'CD45-BV510' or 'CD45RA' when it came first.
        cd4_channel = next((ch for ch in channels if re.search(r"(?<![A-Za-z0-9])CD4(?![0-9A-Za-z])", ch)), None)

        if not cd4_channel:
            log += "ERROR: CD4 channel not found in data\n"
            continue

        if not cytokine_channels:
            log += "ERROR: Cytokine channels (IFN-γ, IL-17) not found in data\n"
            continue

        log += f"- Identified channels: CD4={cd4_channel}, Cytokines={', '.join(cytokine_channels)}\n"

        # Apply gates to identify CD4+ T cells -- as boolean masks over the event table. FCMeasurement.gate()
        # takes a gate object, and the f-strings passed to it raised TypeError for every condition.
        try:
            data = sample.data
            # Create a threshold gate for CD4+ cells (adjust threshold as needed)
            cd4_positive = data[data[cd4_channel] > 1000]  # Threshold value should be adjusted based on data
            log += f"- Applied CD4+ gating: {len(cd4_positive)} cells (from {len(data)} total)\n"
            if len(cd4_positive) == 0:
                log += "ERROR: no CD4+ events, so cytokine frequencies within CD4+ T cells are undefined\n"
                continue

            # Extract cytokine data for CD4+ cells
            cytokine_data = {}
            for cytokine_channel in cytokine_channels:
                # Determine cytokine name from channel
                if "IFN" in cytokine_channel:
                    cytokine_name = "IFN-γ"
                elif "IL-17" in cytokine_channel:
                    cytokine_name = "IL-17"
                else:
                    continue

                # Gate for cytokine positive cells (threshold to be adjusted based on data)
                cytokine_positive = cd4_positive[cd4_positive[cytokine_channel] > 500]  # Threshold should be adjusted

                # Calculate frequency of cytokine-producing cells within CD4+ population
                frequency = len(cytokine_positive) / len(cd4_positive) * 100

                log += f"- {cytokine_name}+ frequency: {frequency:.2f}% of CD4+ T cells\n"
                cytokine_data[cytokine_name] = frequency

            results[condition] = cytokine_data

        except Exception as e:
            log += f"ERROR during analysis: {str(e)}\n"

    # Summarize results across conditions
    if results:
        log += "\n## Results Summary\n"

        # Create DataFrame for results
        df_results = pd.DataFrame()

        for condition in results:
            for cytokine, frequency in results[condition].items():
                df_results.loc[condition, cytokine] = frequency

        # Calculate background (unstimulated)
        if "unstimulated" in results:
            log += "\n### Background-subtracted frequencies\n"
            background = results["unstimulated"]

            for condition in [c for c in results if c != "unstimulated"]:
                for cytokine in results[condition]:
                    if cytokine in background:
                        background_subtracted = results[condition][cytokine] - background[cytokine]
                        background_subtracted = max(0, background_subtracted)  # Ensure non-negative
                        df_results.loc[condition, f"{cytokine} (background-subtracted)"] = background_subtracted

        # Save results to CSV
        results_file = os.path.join(output_dir, "cytokine_frequencies.csv")
        df_results.to_csv(results_file)
        log += f"\nResults saved to: {results_file}\n"

        # Add results table to log
        log += "\n### Frequencies of cytokine-producing CD4+ T cells (%)\n"
        log += df_results.to_string()

        # Interpret results
        log += "\n\n## Interpretation\n"

        # Compare responses to different stimuli
        if "Mtb300" in results and "CMV" in results and "SEB" in results:
            log += "\nComparison of responses to different stimuli:\n"

            for cytokine in ["IFN-γ", "IL-17"]:
                if cytokine in results["Mtb300"] and cytokine in results["CMV"] and cytokine in results["SEB"]:
                    mtb_response = results["Mtb300"][cytokine]
                    cmv_response = results["CMV"][cytokine]
                    seb_response = results["SEB"][cytokine]

                    log += f"- {cytokine}: Mtb300 ({mtb_response:.2f}%), CMV ({cmv_response:.2f}%), SEB ({seb_response:.2f}%)\n"
    else:
        log += "\nNo valid results were generated for any condition.\n"

    return log


def analyze_ebv_antibody_titers(raw_od_data, standard_curve_data, sample_metadata, output_dir="./"):
    """Analyze ELISA data to quantify EBV antibody titers in plasma/serum samples.

    Parameters
    ----------
    raw_od_data : dict
        Dictionary containing optical density (OD) readings for each sample.
        Format: {sample_id: {'VCA_IgG': float, 'VCA_IgM': float, 'EA_IgG': float, 'EA_IgM': float, 'EBNA1_IgG': float, 'EBNA1_IgM': float}}
    standard_curve_data : dict
        Dictionary containing standard curve data for each antibody type.
        Format: {antibody_type: [(concentration, OD), ...]}
    sample_metadata : dict
        Dictionary containing metadata for each sample.
        Format: {sample_id: {'group': str, 'collection_date': str}}
    output_dir : str, optional
        Directory to save output files. Default is current directory.

    Returns
    -------
    str
        Research log summarizing the analysis process and results.

    """
    import os
    from datetime import datetime

    import numpy as np
    import pandas as pd

    # Initialize log
    log = [
        "## EBV Antibody Titer Quantification Analysis",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Number of samples: {len(raw_od_data)}",
        "",
    ]

    # Create dataframe for results
    results_df = pd.DataFrame(
        columns=[
            "Sample_ID",
            "Group",
            "Collection_Date",
            "VCA_IgG",
            "VCA_IgM",
            "EA_IgG",
            "EA_IgM",
            "EBNA1_IgG",
            "EBNA1_IgM",
        ]
    )

    log.append("### 1. Data Preprocessing")
    log.append("- Checking for missing values and outliers in OD readings")

    # Check for missing values
    missing_values = False
    for sample_id, readings in raw_od_data.items():
        for antibody_type in [
            "VCA_IgG",
            "VCA_IgM",
            "EA_IgG",
            "EA_IgM",
            "EBNA1_IgG",
            "EBNA1_IgM",
        ]:
            if antibody_type not in readings:
                missing_values = True
                log.append(f"  - Warning: Missing {antibody_type} reading for sample {sample_id}")

    if not missing_values:
        log.append("  - No missing values detected")

    log.append("")
    log.append("### 2. Standard Curve Fitting")
    log.append("- Fitting standard curves for antibody quantification")

    # Fit standard curves (simple linear regression)
    standard_curves = {}
    for antibody_type, curve_data in standard_curve_data.items():
        concentrations, ods = zip(*curve_data, strict=False)
        # Simple linear regression for standard curve
        slope, intercept = np.polyfit(ods, concentrations, 1)
        standard_curves[antibody_type] = (slope, intercept)
        log.append(f"  - {antibody_type}: Fitted curve with slope={slope:.4f}, intercept={intercept:.4f}")

    log.append("")
    log.append("### 3. Antibody Titer Quantification")
    log.append("- Calculating antibody concentrations using standard curves")

    # Calculate antibody titers for each sample
    for sample_id, readings in raw_od_data.items():
        sample_data = {"Sample_ID": sample_id}

        # Add metadata
        if sample_id in sample_metadata:
            sample_data["Group"] = sample_metadata[sample_id].get("group", "Unknown")
            sample_data["Collection_Date"] = sample_metadata[sample_id].get("collection_date", "Unknown")
        else:
            sample_data["Group"] = "Unknown"
            sample_data["Collection_Date"] = "Unknown"

        # Calculate antibody titers
        for antibody_type in [
            "VCA_IgG",
            "VCA_IgM",
            "EA_IgG",
            "EA_IgM",
            "EBNA1_IgG",
            "EBNA1_IgM",
        ]:
            if antibody_type in readings:
                od = readings[antibody_type]
                # Get the appropriate standard curve: the one keyed by this antibody type, as the documented
                # {antibody_type: [...]} contract has it, else the isotype's. Only 'IgG'/'IgM' keys used to
                # work, and per-antibody keys raised KeyError: 'IgG' (hunt 2026-09-30, uT7-immunology-9).
                curve_type = "IgG" if antibody_type.endswith("IgG") else "IgM"
                curve = standard_curves.get(antibody_type, standard_curves.get(curve_type))
                if curve is None:
                    return (
                        f"Error: no standard curve for {antibody_type}. standard_curve_data has "
                        f"{sorted(standard_curves)}; key each curve by antibody type (e.g. '{antibody_type}') "
                        f"or by isotype ('{curve_type}')."
                    )

                slope, intercept = curve
                # Calculate concentration
                concentration = slope * od + intercept
                sample_data[antibody_type] = concentration
            else:
                sample_data[antibody_type] = np.nan

        # Add to results dataframe
        results_df = pd.concat([results_df, pd.DataFrame([sample_data])], ignore_index=True)

    log.append("")
    log.append("### 4. Results Summary")

    # Calculate summary statistics
    summary_stats = results_df.groupby("Group")[
        ["VCA_IgG", "VCA_IgM", "EA_IgG", "EA_IgM", "EBNA1_IgG", "EBNA1_IgM"]
    ].agg(["mean", "std"])

    # Log summary statistics
    for group in summary_stats.index:
        log.append(f"#### Group: {group}")
        for antibody in [
            "VCA_IgG",
            "VCA_IgM",
            "EA_IgG",
            "EA_IgM",
            "EBNA1_IgG",
            "EBNA1_IgM",
        ]:
            mean = summary_stats.loc[group, (antibody, "mean")]
            std = summary_stats.loc[group, (antibody, "std")]
            log.append(f"- {antibody}: {mean:.2f} ± {std:.2f} U/mL")
        log.append("")

    # Save results to CSV
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "ebv_antibody_titers_results.csv")
    results_df.to_csv(results_file, index=False)
    log.append(f"Full results saved to: {results_file}")

    return "\n".join(log)


def analyze_cns_lesion_histology(image_path, output_dir="./output", stain_type="H&E"):
    """Analyzes histological images of CNS lesions to quantify immune cell infiltration,
    demyelination, and tissue damage.

    Parameters
    ----------
    image_path : str
        Path to the microscopy image file of brain or spinal cord tissue section
    output_dir : str, optional
        Directory to save output files (default: "./output")
    stain_type : str, optional
        Type of histological stain used (default: "H&E", other options: "LFB", "IHC")

    Returns
    -------
    str
        Research log summarizing the analysis steps, findings, and saved file paths

    """
    import os
    from datetime import datetime

    import numpy as np

    # Every path that cannot analyse the image is an error that says why. A missing file, a missing
    # scikit-image or any exception used to fall through to a "simulated analysis" -- 1250 cells, damage 7.5,
    # "HIGH cellular infiltration detected" -- reported as the slide's pathology (hunt 2026-09-30,
    # uT7-immunology-4).
    if not os.path.isfile(image_path):
        return f"Error: image file not found: {image_path}. No histology metrics were computed."
    if stain_type not in ("H&E", "LFB", "IHC"):
        return (
            f"Error: stain_type must be 'H&E', 'LFB' or 'IHC' (got {stain_type!r}). No histology metrics were computed."
        )

    # Attempt to load required libraries
    try:
        from skimage import (
            color,
            exposure,
            filters,
            io,
            measure,
            morphology,
            segmentation,
            util,
        )
        from skimage.feature import graycomatrix, graycoprops
    except ImportError as e:
        return (
            f"Error: scikit-image is required for this analysis but could not be imported ({e}). "
            "No histology metrics were computed."
        )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize log
    log = []
    log.append(f"CNS Lesion Histological Analysis ({stain_type})")
    log.append(f"Image: {os.path.basename(image_path)}")
    log.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append("")
    log.append("Using scikit-image for histology analysis")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = os.path.abspath(f"{output_dir}/cns_lesion_metrics_{timestamp}.txt")
    result_filename = os.path.abspath(f"{output_dir}/cns_lesion_analysis_{timestamp}.png")

    # Default values
    cell_count = 0
    damage_score = 0

    try:
        # Load and preprocess the image
        log.append("Loading and preprocessing image...")
        original_image = io.imread(image_path)
        # PNG exports often carry an alpha channel, which rgb2gray rejects
        if original_image.ndim == 3 and original_image.shape[2] == 4:
            original_image = color.rgba2rgb(original_image)

        # Convert to grayscale if RGB
        gray_image = color.rgb2gray(original_image) if len(original_image.shape) > 2 else original_image

        # Enhance contrast
        enhanced_image = exposure.equalize_adapthist(gray_image)

        # Image segmentation based on stain type
        if stain_type == "H&E":
            log.append("Performing H&E stain analysis...")
            # For H&E staining, segment based on intensity thresholds
            # Identify cell nuclei (typically dark in H&E)
            thresh_val = filters.threshold_otsu(enhanced_image)
            nuclei_mask = enhanced_image < thresh_val
            nuclei_mask = morphology.remove_small_objects(nuclei_mask, min_size=30)

            # Count nuclei (cells)
            labeled_nuclei = measure.label(nuclei_mask)
            cell_props = measure.regionprops(labeled_nuclei)
            cell_count = len(cell_props)

            # Calculate texture features for damage assessment
            glcm = graycomatrix(
                (enhanced_image * 255).astype(np.uint8),
                distances=[1],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=256,
                symmetric=True,
                normed=True,
            )

            # Extract texture features
            contrast = graycoprops(glcm, "contrast").mean()
            homogeneity = graycoprops(glcm, "homogeneity").mean()
            energy = graycoprops(glcm, "energy").mean()

            # Higher contrast and lower homogeneity often indicate tissue damage
            damage_score = contrast / (homogeneity * energy)

            log.append(f"Detected {cell_count} cells")
            log.append(f"Texture analysis completed: contrast={contrast:.3f}, homogeneity={homogeneity:.3f}")
            log.append(f"Calculated damage score: {damage_score:.2f}")

        elif stain_type == "LFB":
            log.append("Performing LFB (myelin) stain analysis...")
            # For Luxol Fast Blue staining (myelin)
            # Blue intensity correlates with myelin content
            if len(original_image.shape) > 2:
                # Extract blue channel for RGB images
                blue_channel = original_image[:, :, 2] if original_image.shape[2] >= 3 else gray_image

                # Threshold to identify myelin
                thresh_val = filters.threshold_otsu(blue_channel)
                myelin_mask = blue_channel > thresh_val
                myelin_mask = morphology.remove_small_objects(myelin_mask, min_size=100)
            else:
                # Grayscale LFB handling
                thresh_val = filters.threshold_otsu(enhanced_image)
                myelin_mask = enhanced_image > thresh_val

            # Calculate myelin content as percentage of tissue area
            myelin_percent = np.sum(myelin_mask) / myelin_mask.size * 100

            # Demyelination score (inverse of myelin content)
            demyelination_score = 100 - myelin_percent

            # Cell count is less relevant for LFB stain
            cell_count = 0
            damage_score = demyelination_score

            log.append(f"Myelin content: {myelin_percent:.2f}%")
            log.append(f"Demyelination score: {demyelination_score:.2f}")

        else:  # IHC
            log.append("Performing immunohistochemistry analysis...")
            # For immunohistochemistry (specific immune cell markers)
            # Positive staining appears brown/dark
            thresh_val = filters.threshold_otsu(enhanced_image)
            positive_stain_mask = enhanced_image < thresh_val
            positive_stain_mask = morphology.remove_small_objects(positive_stain_mask, min_size=30)

            # Calculate percentage of positive staining
            positive_percent = np.sum(positive_stain_mask) / positive_stain_mask.size * 100

            # Count individual positive cells
            labeled_cells = measure.label(positive_stain_mask)
            cell_props = measure.regionprops(labeled_cells)
            cell_count = len(cell_props)

            # Infiltration score based on positive staining percentage
            infiltration_score = positive_percent
            damage_score = infiltration_score

            log.append(f"Detected {cell_count} positive cells")
            log.append(f"Positive staining: {positive_percent:.2f}%")
            log.append(f"Infiltration score: {infiltration_score:.2f}")
    except Exception as e:
        log.append(f"Error during image analysis: {type(e).__name__}: {e}")
        log.append("No histology metrics were computed.")
        return "\n".join(log)

    # Save segmentation result
    try:
        # Create a visualization of the segmentation
        if stain_type == "H&E":
            boundaries = segmentation.find_boundaries(labeled_nuclei)
        elif stain_type == "LFB":
            boundaries = segmentation.find_boundaries(myelin_mask)
        else:
            boundaries = segmentation.find_boundaries(labeled_cells)

        # Create overlay for visualization, on a [0, 1] float copy: multiplying the uint8 original by 255
        # wrapped around, so a background of 200 was saved as 56 (hunt 2026-09-30, uT7-immunology-24).
        if len(original_image.shape) > 2:
            overlay = util.img_as_float(original_image)[:, :, :3].copy()
        else:
            gray_float = util.img_as_float(gray_image)
            overlay = np.stack([gray_float, gray_float, gray_float], axis=-1)
        if np.any(boundaries):
            overlay[boundaries, 0] = 1.0  # Red channel
            overlay[boundaries, 1:3] = 0.0  # Green and Blue channels

        # Save the overlay image
        io.imsave(result_filename, util.img_as_ubyte(np.clip(overlay, 0, 1)))
        log.append(f"Segmentation result saved to: {result_filename}")
    except Exception as e:
        log.append(f"Warning: Could not save segmentation result: {str(e)}")

    # Create metrics file
    try:
        with open(metrics_filename, "w") as f:
            f.write("CNS Lesion Analysis Results\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Stain type: {stain_type}\n")
            f.write(f"Analysis timestamp: {timestamp}\n\n")
            f.write(f"Cell count: {cell_count}\n")
            f.write(f"Damage/infiltration score: {damage_score:.2f}\n")

            if stain_type == "H&E":
                f.write("Texture metrics:\n")
                f.write(f"  - Contrast: {contrast:.4f}\n")
                f.write(f"  - Homogeneity: {homogeneity:.4f}\n")
                f.write(f"  - Energy: {energy:.4f}\n")
            elif stain_type == "LFB":
                f.write(f"Myelin content: {myelin_percent:.2f}%\n")
                f.write(f"Demyelination score: {demyelination_score:.2f}\n")
            else:
                f.write(f"Positive staining: {positive_percent:.2f}%\n")
                f.write(f"Infiltration score: {infiltration_score:.2f}\n")

        log.append(f"Metrics saved to: {metrics_filename}")
    except Exception as e:
        log.append(f"Warning: Could not save metrics file: {str(e)}")

    # Add interpretation based on the results
    log.append("\nINTERPRETATION:")
    if stain_type == "H&E":
        if cell_count > 1000:
            log.append("- HIGH cellular infiltration detected, indicating significant inflammation")
        elif cell_count > 500:
            log.append("- MODERATE cellular infiltration detected")
        else:
            log.append("- LOW cellular infiltration detected")

        if damage_score > 10:
            log.append("- Tissue texture analysis suggests SEVERE tissue damage/disorganization")
        elif damage_score > 5:
            log.append("- Tissue texture analysis suggests MODERATE tissue damage/disorganization")
        else:
            log.append("- Tissue texture analysis suggests MINIMAL tissue damage/disorganization")

    elif stain_type == "LFB":
        if demyelination_score > 70:
            log.append("- SEVERE demyelination detected (>70% myelin loss)")
        elif demyelination_score > 40:
            log.append("- MODERATE demyelination detected (40-70% myelin loss)")
        else:
            log.append("- MILD demyelination detected (<40% myelin loss)")

    else:  # IHC
        if infiltration_score > 30:
            log.append("- HIGH level of immune marker positivity, indicating SEVERE inflammation/infiltration")
        elif infiltration_score > 15:
            log.append("- MODERATE level of immune marker positivity")
        else:
            log.append("- LOW level of immune marker positivity")

    return "\n".join(log)


def _ihc_auto_stain_mode(img, gray_img):
    """``stain_mode="auto"`` for ``analyze_immunohistochemistry_image``: ``(mode, reason)``.

    The median rule only: the median pixel is taken as background, so the stain is the minority of the
    image. That premise is stated in ``reason`` every time, because it is the one thing this rule cannot
    check -- a slide more than half stained is read the wrong way round.

    Three review rounds of a colour rule on top of the median (hunt 2026-09-30, uT7-immunology-10) each
    fixed the previous round's repros and broke another plausible slide: an illumination cast read as
    stain colour, a near-black chromogen read as having no colour, block-mean saturation undoing dense
    fine-grained stain. The image does not reliably say which side is the stain, so the default became
    the tool's own subject -- chromogenic IHC, ``stain_mode="brightfield"`` -- and "auto" says what it
    assumed rather than guessing better. ``img`` is accepted and ignored so callers keep one signature.
    """
    import numpy as np

    lo, hi = float(np.min(gray_img)), float(np.max(gray_img))
    if hi <= lo:
        return "brightfield", "the image is one flat grey level, so there is no stain to tell from background"
    background_level = (float(np.median(gray_img)) - lo) / (hi - lo)
    mode = "brightfield" if background_level > 0.5 else "fluorescence"
    return mode, (
        f"the median pixel, taken as background, is {'bright' if background_level > 0.5 else 'dark'}; this "
        "assumes the stain covers less than half of the image -- for a slide more than half stained, set "
        "stain_mode"
    )


def analyze_immunohistochemistry_image(
    image_path, protein_name="Unknown", output_dir="./ihc_results/", stain_mode="brightfield"
):
    """Analyzes immunohistochemistry images to quantify protein expression and spatial distribution.

    Parameters
    ----------
    image_path : str
        Path to the microscopy image of tissue section stained with antibodies
    protein_name : str, optional
        Name of the protein being analyzed (default: "Unknown")
    output_dir : str, optional
        Directory to save output files (default: "./ihc_results/")
    stain_mode : str, optional
        "brightfield" for a chromogenic stain (e.g. DAB) that is darker than the background (the
        default: this is immunohistochemistry), "fluorescence" for a stain brighter than a dark
        background (immunofluorescence), or "auto" to take the median pixel as background, which
        assumes the stain covers less than half of the image; the log says which was used
        (default: "brightfield")

    Returns
    -------
    str
        Research log summarizing the analysis steps, results, and saved file locations

    """
    import os
    from datetime import datetime

    import numpy as np
    from skimage import filters, io, measure, morphology
    from skimage.color import rgb2gray, rgba2rgb
    from skimage.util import img_as_float

    if stain_mode not in ("auto", "brightfield", "fluorescence"):
        return f"Error: stain_mode must be 'auto', 'brightfield' or 'fluorescence' (got {stain_mode!r})"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{_safe_filename_part(protein_name)}_{timestamp}"

    # Load the image
    try:
        img = io.imread(image_path)
        log = f"Loaded image from {image_path}\n"
    except Exception as e:
        return f"Error loading image: {str(e)}"

    # Convert to grayscale if RGB (PNG exports often carry an alpha channel, which rgb2gray rejects)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = rgba2rgb(img)
    colour_img = None
    if len(img.shape) == 3 and img.shape[2] >= 3:
        colour_img = img
        gray_img = rgb2gray(img)
        log += "Converted RGB image to grayscale for analysis\n"
    else:
        gray_img = img
        log += "Image already in grayscale format\n"

    # Segment the STAIN, whichever way round it is. A chromogenic (brightfield) stain is darker than the
    # background, and `> otsu` on the raw grey level selected the unstained background and dropped the
    # stained cells (hunt 2026-09-30, uT7-immunology-10). Brightfield images are analysed on 1 - grey level,
    # so the stain is bright and "intensity" grows with staining in both modes.
    if stain_mode == "auto":
        stain_mode, reason = _ihc_auto_stain_mode(colour_img, gray_img)
        log += f"Stain mode: {stain_mode} (auto: {reason})\n"
    else:
        log += f"Stain mode: {stain_mode}\n"
    signal_img = 1.0 - img_as_float(gray_img) if stain_mode == "brightfield" else gray_img

    # Segment stained cells/tissue using thresholding -- on the signal itself. The 2-98 percentile stretch
    # this used to threshold collapses to a constant when under 2% of the slide is stained (both
    # percentiles fall on background), which reported a sparsely stained slide as 0 regions (found while
    # fixing uT7-immunology-10).
    threshold_value = filters.threshold_otsu(signal_img)
    binary_mask = signal_img > threshold_value

    # Clean up the mask with morphological operations
    binary_mask = morphology.remove_small_objects(binary_mask, min_size=50)
    binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=50)
    log += "Segmented stained regions using Otsu thresholding and morphological cleanup\n"

    # Label connected regions
    labeled_mask, num_features = measure.label(binary_mask, return_num=True)
    log += f"Identified {num_features} distinct stained regions\n"

    # Quantify protein expression by measuring intensity in segmented regions
    region_props = measure.regionprops(labeled_mask, intensity_image=signal_img)

    # Calculate intensity metrics
    mean_intensities = [prop.mean_intensity for prop in region_props]
    total_intensity = sum(prop.mean_intensity * prop.area for prop in region_props)
    avg_intensity = np.mean(mean_intensities) if mean_intensities else 0

    log += "Protein expression quantification:\n"
    log += f"- Total intensity: {total_intensity:.2f}\n"
    log += f"- Average intensity: {avg_intensity:.2f}\n"
    log += f"- Number of regions analyzed: {len(region_props)}\n"

    # Analyze spatial distribution
    if region_props:
        # Calculate centroid coordinates for each region
        centroids = [prop.centroid for prop in region_props]

        # Calculate distances between centroids to assess clustering
        from scipy.spatial import distance

        if len(centroids) > 1:
            distances = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    distances.append(distance.euclidean(centroids[i], centroids[j]))

            avg_distance = np.mean(distances)
            log += "Spatial distribution analysis:\n"
            log += f"- Average distance between regions: {avg_distance:.2f} pixels\n"
            log += f"- Minimum distance between regions: {min(distances):.2f} pixels\n"
            log += f"- Maximum distance between regions: {max(distances):.2f} pixels\n"
        else:
            log += "Spatial distribution analysis: Only one region detected\n"

    # Save results
    segmentation_file = os.path.join(output_dir, f"{base_filename}_segmentation.png")
    io.imsave(segmentation_file, labeled_mask)

    # Create a CSV with region properties
    import csv

    csv_file = os.path.join(output_dir, f"{base_filename}_region_data.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Region ID", "Area", "Mean Intensity", "Centroid Y", "Centroid X"])
        for i, prop in enumerate(region_props):
            writer.writerow(
                [
                    i + 1,
                    prop.area,
                    prop.mean_intensity,
                    prop.centroid[0],
                    prop.centroid[1],
                ]
            )

    log += "\nResults saved:\n"
    log += f"- Segmentation image: {segmentation_file}\n"
    log += f"- Region data: {csv_file}\n"

    return log
