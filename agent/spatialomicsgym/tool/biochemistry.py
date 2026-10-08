def analyze_circular_dichroism_spectra(
    sample_name,
    sample_type,
    wavelength_data,
    cd_signal_data,
    temperature_data=None,
    thermal_cd_data=None,
    output_dir="./",
):
    """Analyzes circular dichroism (CD) spectroscopy data to determine secondary structure and thermal stability.

    Parameters
    ----------
    sample_name : str
        Name of the biomolecule sample (e.g., "Znf706", "G-quadruplex")
    sample_type : str
        Type of biomolecule ("protein" or "nucleic_acid")
    wavelength_data : list or numpy.ndarray
        Wavelength values in nm for CD spectrum
    cd_signal_data : list or numpy.ndarray
        CD signal intensity values (typically in mdeg or Δε)
    temperature_data : list or numpy.ndarray, optional
        Temperature values (°C) for thermal denaturation experiment
    thermal_cd_data : list or numpy.ndarray, optional
        CD signal values at specific wavelength across different temperatures
    output_dir : str, optional
        Directory to save result files, defaults to current directory

    Returns
    -------
    str
        Research log summarizing the CD analysis steps and results

    """
    import os
    import re
    from datetime import datetime

    import numpy as np

    # Three crashes, two after the files were written: `structure` existed only for 'protein' and
    # 'nucleic_acid' (so 'DNA' raised UnboundLocalError), the conclusion read `tm` whenever
    # temperature_data was given although it is computed only with thermal_cd_data too, and a '/' in
    # sample_name built a path into a missing directory (hunt 2026-09-30, uT2-pharmacology-12).
    kind = str(sample_type).strip().lower().replace(" ", "_").replace("-", "_")
    if kind in ("protein", "peptide", "polypeptide"):
        kind = "protein"
    elif kind in ("nucleic_acid", "dna", "rna", "oligonucleotide", "g_quadruplex"):
        kind = "nucleic_acid"
    file_stem = re.sub(r"[^\w.-]+", "_", str(sample_name)).strip("._") or "sample"
    structure = f"unclassified (unsupported sample_type {sample_type!r}; use 'protein' or 'nucleic_acid')"
    tm = None

    # Initialize research log
    log = f"# Circular Dichroism Analysis Report for {sample_name}\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    log += "## Sample Information\n"
    log += f"- Sample Name: {sample_name}\n"
    log += f"- Sample Type: {sample_type}\n\n"

    # Convert inputs to numpy arrays if they aren't already
    wavelength_data = np.array(wavelength_data)
    cd_signal_data = np.array(cd_signal_data)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Analyze CD spectrum for secondary structure
    log += "## Secondary Structure Analysis\n"

    # Different analysis approaches based on sample type
    if kind == "protein":
        # Analyze protein secondary structure based on characteristic spectral features
        alpha_helix_signal = np.sum((wavelength_data >= 190) & (wavelength_data <= 195) & (cd_signal_data > 0))
        beta_sheet_signal = np.sum((wavelength_data >= 215) & (wavelength_data <= 220) & (cd_signal_data < 0))
        random_coil_signal = np.sum((wavelength_data >= 195) & (wavelength_data <= 200) & (cd_signal_data < 0))

        # Simple classification based on signal patterns
        if alpha_helix_signal > beta_sheet_signal and alpha_helix_signal > random_coil_signal:
            structure = "predominantly alpha-helical"
        elif beta_sheet_signal > alpha_helix_signal and beta_sheet_signal > random_coil_signal:
            structure = "predominantly beta-sheet"
        else:
            structure = "mixed or predominantly random coil"

        log += f"- The CD spectrum indicates {structure} structure for {sample_name}.\n"
        log += "- Key spectral features:\n"
        log += "  - 190-195 nm region: associated with alpha-helical content\n"
        log += "  - 215-220 nm region: associated with beta-sheet content\n\n"

    elif kind == "nucleic_acid":
        # Analyze nucleic acid structure (e.g., G-quadruplex has characteristic positive peak ~295 nm)
        g_quadruplex_signal = np.sum((wavelength_data >= 290) & (wavelength_data <= 300) & (cd_signal_data > 0))
        b_form_signal = np.sum((wavelength_data >= 270) & (wavelength_data <= 280) & (cd_signal_data > 0))

        if g_quadruplex_signal > 0:
            structure = "G-quadruplex characteristics"
        elif b_form_signal > 0:
            structure = "B-form characteristics"
        else:
            structure = "non-standard structure"

        log += f"- The CD spectrum indicates {structure} for {sample_name}.\n"
        log += "- Key spectral features:\n"
        log += "  - 290-300 nm positive peak: characteristic of G-quadruplex structures\n"
        log += "  - 270-280 nm positive peak: characteristic of B-form DNA\n\n"
    else:
        log += f"- Secondary structure not classified: {structure}.\n\n"

    # Save spectral data results
    spectral_file = os.path.join(output_dir, f"{file_stem}_cd_spectrum_analysis.txt")
    with open(spectral_file, "w") as f:
        f.write("Wavelength (nm)\tCD Signal\n")
        for wl, signal in zip(wavelength_data, cd_signal_data, strict=False):
            f.write(f"{wl:.1f}\t{signal:.4f}\n")

    log += f"- Detailed spectral data saved to: {spectral_file}\n\n"

    # 2. Thermal stability analysis (if temperature data provided)
    if temperature_data is not None and thermal_cd_data is not None:
        temperature_data = np.array(temperature_data)
        thermal_cd_data = np.array(thermal_cd_data)

        log += "## Thermal Stability Analysis\n"

        # Simple Tm estimation (melting temperature) - find temperature at 50% unfolding
        # Normalize thermal data to 0-1 range for unfolding fraction
        min_signal = np.min(thermal_cd_data)
        max_signal = np.max(thermal_cd_data)
        unfolded_fraction = (thermal_cd_data - min_signal) / (max_signal - min_signal)

        # Find the temperature closest to 50% unfolding
        tm_idx = np.argmin(np.abs(unfolded_fraction - 0.5))
        tm = temperature_data[tm_idx]

        log += f"- Estimated melting temperature (Tm): {tm:.1f}°C\n"

        # Cooperativity assessment (crude estimate based on transition steepness)
        t_range = temperature_data[-1] - temperature_data[0]
        transition_width = (
            t_range / len(temperature_data) * np.sum((unfolded_fraction > 0.2) & (unfolded_fraction < 0.8))
        )

        if transition_width < 0.2 * t_range:
            cooperativity = "highly cooperative (sharp transition)"
        elif transition_width < 0.4 * t_range:
            cooperativity = "moderately cooperative"
        else:
            cooperativity = "non-cooperative (broad transition)"

        log += f"- Thermal transition: {cooperativity}\n"

        # Save thermal denaturation data
        thermal_file = os.path.join(output_dir, f"{file_stem}_thermal_denaturation.txt")
        with open(thermal_file, "w") as f:
            f.write("Temperature (°C)\tCD Signal\tUnfolded Fraction\n")
            for temp, signal, unfold in zip(temperature_data, thermal_cd_data, unfolded_fraction, strict=False):
                f.write(f"{temp:.1f}\t{signal:.4f}\t{unfold:.4f}\n")

        log += f"- Thermal denaturation data saved to: {thermal_file}\n\n"
    elif temperature_data is not None or thermal_cd_data is not None:
        log += "## Thermal Stability Analysis\n"
        log += "- Not analysed: temperature_data and thermal_cd_data are both needed.\n\n"

    # 3. Summary and conclusions
    log += "## Conclusions\n"
    if kind == "protein":
        log += f"- {sample_name} shows {structure} according to CD spectroscopy.\n"
    elif kind == "nucleic_acid":
        log += f"- {sample_name} exhibits {structure} according to CD spectroscopy.\n"
    else:
        log += f"- Secondary structure of {sample_name}: {structure}.\n"

    if tm is not None:
        log += f"- The molecule has a melting temperature of {tm:.1f}°C with {cooperativity}.\n"

    return log


def analyze_rna_secondary_structure_features(dot_bracket_structure, sequence=None):
    """Calculate numeric values for various structural features of an RNA secondary structure.

    Parameters
    ----------
    dot_bracket_structure : str
        RNA secondary structure in dot-bracket notation (e.g., "(((...)))").
        Parentheses represent base pairs, dots represent unpaired bases.
    sequence : str, optional
        The RNA sequence corresponding to the structure. If provided,
        sequence-dependent energy calculations will be performed.

    Returns
    -------
    str
        A research log summarizing the calculated structural features and analysis steps.

    """
    # Initialize research log
    log = "# RNA Secondary Structure Feature Analysis\n\n"

    # Validate input
    if not all(c in "().[]{}" for c in dot_bracket_structure):
        return "Error: Invalid dot-bracket notation. Use only '()', '[]', '{}', and '.'"

    log += f"Input structure (length: {len(dot_bracket_structure)}): {dot_bracket_structure}\n"
    if sequence:
        log += f"Input sequence (length: {len(sequence)}): {sequence}\n"
        if len(sequence) != len(dot_bracket_structure):
            return "Error: Sequence and structure lengths do not match."

    # Extract base pairs, one stack per bracket type: [] and {} mark pairs that cross the () pairs
    # (pseudoknots). One shared stack rejected '((..[[..))..]]' as mismatched (hunt 2026-09-30,
    # uT2-pharmacology-13).
    pairs = []
    opening_of = {")": "(", "]": "[", "}": "{"}
    stacks = {"(": [], "[": [], "{": []}

    for i, char in enumerate(dot_bracket_structure):
        if char in stacks:
            stacks[char].append(i)
        elif char in opening_of:
            stack = stacks[opening_of[char]]
            if not stack:
                return "Error: Unbalanced structure. More closing than opening brackets."
            pairs.append((stack.pop(), i))

    if any(stacks.values()):
        return "Error: Unbalanced structure. More opening than closing brackets."

    # Sort pairs by position
    pairs.sort()

    # Identify stems (consecutive base pairs)
    stems = []
    current_stem = []

    for i, (start, end) in enumerate(pairs):
        if (i == 0 or start != pairs[i - 1][0] + 1 or end != pairs[i - 1][1] - 1) and current_stem:
            stems.append(current_stem)
            current_stem = []
        current_stem.append((start, end))

    if current_stem:
        stems.append(current_stem)

    # Calculate stem lengths
    stem_lengths = [len(stem) for stem in stems]

    # Calculate hairpin loop sizes: the unpaired bases closed by a stem's innermost pair. This used to
    # measure from that pair's closing base to the next stem or the sequence end -- a linker or the 3'
    # tail, not the loop (hunt 2026-09-30, uT2-pharmacology-13).
    paired_positions = {k for pair in pairs for k in pair}
    loops = []
    for stem in stems:
        inner_start, inner_end = stem[-1]
        if not any(k in paired_positions for k in range(inner_start + 1, inner_end)):
            loops.append(inner_end - inner_start - 1)

    # Calculate base pair statistics
    total_paired_bases = len(pairs) * 2
    total_unpaired_bases = len(dot_bracket_structure) - total_paired_bases

    # Calculate simplified free energy if sequence is provided
    stem_energies = []
    if sequence and len(stems) > 0:
        # Simplified energy parameters for nearest-neighbor model
        # Values are approximate and simplified for illustration
        energy_params = {
            "AU": -0.9,
            "UA": -0.9,
            "GC": -2.1,
            "CG": -2.1,
            "GU": -0.5,
            "UG": -0.5,
        }

        for stem in stems:
            stem_energy = 0
            for start, end in stem:
                if start < len(sequence) and end < len(sequence):
                    pair = sequence[start] + sequence[end]
                    stem_energy += energy_params.get(pair, 0)
            stem_energies.append(stem_energy)

    # Prepare results
    log += "\n## Structural Features\n\n"
    log += f"Total base pairs: {len(pairs)}\n"
    log += f"Number of stems: {len(stems)}\n"
    log += f"Longest stem length: {max(stem_lengths) if stem_lengths else 0}\n"
    log += f"Average stem length: {sum(stem_lengths) / len(stem_lengths) if stem_lengths else 0:.2f}\n"
    log += f"Paired bases: {total_paired_bases} ({total_paired_bases / len(dot_bracket_structure) * 100:.1f}%)\n"
    log += f"Unpaired bases: {total_unpaired_bases} ({total_unpaired_bases / len(dot_bracket_structure) * 100:.1f}%)\n"

    log += f"Number of hairpin loops: {len(loops)}\n"
    if loops:
        log += f"Average hairpin loop size: {sum(loops) / len(loops):.2f}\n"
        log += f"Largest hairpin loop size: {max(loops)}\n"
    log += (
        f"Unpaired bases outside hairpin loops (internal loops, bulges, junctions, linkers, tails): "
        f"{total_unpaired_bases - sum(loops)}\n"
    )

    if sequence and stem_energies:
        log += "\n## Energy Calculations\n\n"
        log += f"Total estimated free energy: {sum(stem_energies):.2f} kcal/mol\n"

        if len(stems) >= 2:
            log += f"Upstream stem free energy: {stem_energies[0]:.2f} kcal/mol\n"
            log += f"Downstream stem free energy: {stem_energies[-1]:.2f} kcal/mol\n"

        # If the first stem is the "zipper" stem
        if stem_lengths and stem_lengths[0] >= 3:
            log += f"Zipper stem free energy: {stem_energies[0]:.2f} kcal/mol\n"

    log += "\n## Stem Details\n\n"
    for i, stem in enumerate(stems):
        log += f"Stem {i + 1}: {len(stem)} base pairs\n"
        log += f"  Positions: {stem[0][0]}-{stem[0][1]} to {stem[-1][0]}-{stem[-1][1]}\n"
        if sequence and i < len(stem_energies):
            log += f"  Estimated stability: {stem_energies[i]:.2f} kcal/mol\n"

    return log


def _fit_michaelis_menten(substrate, rates, km_guess):
    """Fit v = Vmax·S/(Km + S); return (vmax, km, vmax_err, km_err) in the units of the inputs.

    curve_fit's stopping tolerances are absolute, so velocities on a 1e-6 scale (M/s, a common unit) stopped
    it near its starting guess: noiseless data with Km 8 µM came back as 8.887 µM (R² 0.998) from the assay
    tool and 21 µM from the protease tool. The fit now runs on velocities scaled to max |v| = 1 and substrate
    scaled to max S = 1, and a fit that never left its starting guess raises (hunt 2026-09-30,
    uT2-pharmacology-14).
    """
    import numpy as np
    from scipy.optimize import curve_fit

    s = np.asarray(substrate, dtype=float)
    v = np.asarray(rates, dtype=float)
    v_scale = float(np.max(np.abs(v))) if v.size else 0.0
    s_scale = float(np.max(s)) if s.size else 0.0
    if not (np.isfinite(v_scale) and float(np.max(v)) > 0):
        raise ValueError("no positive, finite velocity: there is no saturation curve to fit")
    if not (np.isfinite(s_scale) and s_scale > 0):
        raise ValueError("no positive substrate concentration")

    def michaelis_menten(x, vmax, km):
        return vmax * x / (km + x)

    p0 = [float(np.max(v)) / v_scale, float(km_guess) / s_scale]
    popt, pcov = curve_fit(
        michaelis_menten,
        s / s_scale,
        v / v_scale,
        p0=p0,
        bounds=([0, 0], [np.inf, np.inf]),
        x_scale="jac",
        maxfev=10000,
    )
    if np.allclose(popt, p0, rtol=1e-6, atol=1e-12):
        raise RuntimeError("the optimiser never moved from its starting guess, so nothing was fitted")
    vmax_err, km_err = np.sqrt(np.abs(np.diag(pcov)))
    return popt[0] * v_scale, popt[1] * s_scale, vmax_err * v_scale, km_err * s_scale


def analyze_protease_kinetics(
    time_points,
    fluorescence_data,
    substrate_concentrations,
    enzyme_concentration,
    output_prefix="protease_kinetics",
    output_dir="./",
    fluorescence_per_uM_product=None,
):
    """Analyze protease kinetics data from fluorogenic peptide cleavage assays.

    This function processes time-course fluorescence data from protease-mediated peptide
    cleavage assays, fits the data to Michaelis-Menten kinetics, and determines Vmax and KM,
    and -- given a product fluorescence calibration -- kcat and catalytic efficiency.

    Parameters
    ----------
    time_points : numpy.ndarray
        Array of time points (in seconds) at which measurements were taken

    fluorescence_data : numpy.ndarray
        2D array of fluorescence measurements where each row corresponds to a different
        substrate concentration and each column corresponds to a time point

    substrate_concentrations : numpy.ndarray
        Array of substrate concentrations (in μM) corresponding to each row in fluorescence_data

    enzyme_concentration : float
        Concentration of the protease enzyme (in μM)

    output_prefix : str, optional
        Prefix for output files (default: "protease_kinetics")

    output_dir : str, optional
        Directory to save output files (default: "./"; with the default prefix and directory, a later call's
        files get a _2, _3, ... suffix instead of overwriting an earlier result)

    fluorescence_per_uM_product : float, optional
        Fluorescence units per μM of cleaved product (the slope of a product standard curve). Needed to
        report kcat in s^-1; without it the tool reports Vmax/[E] in a.u. s^-1 μM^-1 and no kcat.

    Returns
    -------
    str
        A research log summarizing the analysis steps and results

    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create full output paths. With the default prefix and directory a second call overwrote the files
    # the first log names (hunt 2026-09-30, uT2-pharmacology-23).
    output_names = [f"{output_prefix}_mm_plot.png", f"{output_prefix}_results.txt"]
    if output_prefix == "protease_kinetics" and output_dir == "./":
        from spatialomicsgym.tool.pharmacology import _fresh_output_paths

        plot_filename, results_filename = _fresh_output_paths(output_dir, output_names)
    else:
        plot_filename, results_filename = [os.path.join(output_dir, n) for n in output_names]

    # Step 1: Calculate initial velocities for each substrate concentration
    initial_velocities = np.zeros(len(substrate_concentrations))

    for i, fluorescence_curve in enumerate(fluorescence_data):
        # Linear fit to the initial portion of the curve (first 20% of time points or at least 5 points)
        num_points = max(5, int(len(time_points) * 0.2))
        slope, _ = np.polyfit(time_points[:num_points], fluorescence_curve[:num_points], 1)
        initial_velocities[i] = slope

    # The slopes are fluorescence per second, so Vmax/[E] is in a.u. s^-1 μM^-1. It used to be labelled
    # kcat in s^-1 (and kcat/KM in μM^-1 s^-1) with no calibration at all, so a 10x detector gain gave a
    # 10x 'kcat'. kcat is reported only from slopes converted to μM/s with a product standard curve
    # (hunt 2026-09-30, uT2-pharmacology-28).
    calibrated = fluorescence_per_uM_product is not None
    if calibrated:
        gain = float(fluorescence_per_uM_product)
        if not gain > 0:
            return "Error: fluorescence_per_uM_product (fluorescence units per μM of product) must be > 0."
        initial_velocities = initial_velocities / gain
    v_unit = "μM/s" if calibrated else "a.u./s"

    # Step 2: Define Michaelis-Menten equation for curve fitting
    def michaelis_menten(s, vmax, km):
        return vmax * s / (km + s)

    # Step 3: Fit the data to the Michaelis-Menten equation, on rescaled axes so a slow enzyme's
    # μM/s rates fit as well as a.u./s ones (hunt 2026-09-30, uT2-pharmacology-14)
    try:
        vmax, km, vmax_std, km_std = _fit_michaelis_menten(
            substrate_concentrations, initial_velocities, np.mean(substrate_concentrations)
        )

        # Step 4: Calculate kcat and catalytic efficiency
        kcat = vmax / enzyme_concentration
        kcat_std = vmax_std / enzyme_concentration
        catalytic_efficiency = kcat / km
        catalytic_efficiency_std = catalytic_efficiency * np.sqrt((kcat_std / kcat) ** 2 + (km_std / km) ** 2)

        # Step 5: Create a plot and save it
        plt.figure(figsize=(10, 6))
        plt.scatter(
            substrate_concentrations,
            initial_velocities,
            color="blue",
            label="Experimental data",
        )

        # Generate smooth curve for the fitted model
        s_curve = np.linspace(0, max(substrate_concentrations) * 1.2, 100)
        v_curve = michaelis_menten(s_curve, vmax, km)
        plt.plot(s_curve, v_curve, "r-", label="Michaelis-Menten fit")

        plt.xlabel("Substrate Concentration (μM)")
        plt.ylabel(f"Initial Velocity ({v_unit})")
        plt.title("Michaelis-Menten Kinetics")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(plot_filename)
        plt.close()

        result_lines = [
            f"Vmax: {vmax:.4f} ± {vmax_std:.4f} {v_unit}",
            f"KM: {km:.4f} ± {km_std:.4f} μM",
        ]
        if calibrated:
            result_lines += [
                f"kcat: {kcat:.4f} ± {kcat_std:.4f} s^-1",
                f"Catalytic efficiency (kcat/KM): {catalytic_efficiency:.4f} ± {catalytic_efficiency_std:.4f} μM^-1 s^-1",
            ]
        else:
            result_lines += [
                f"Vmax/[E]: {kcat:.4f} ± {kcat_std:.4f} a.u. s^-1 μM^-1 -- not a kcat: fluorescence was not "
                "calibrated to product (pass fluorescence_per_uM_product, the slope of a product standard "
                "curve, for kcat in s^-1 and kcat/KM)",
            ]

        # Step 6: Save numerical results to a file
        with open(results_filename, "w") as f:
            f.write("Protease Kinetics Analysis Results\n")
            f.write("==================================\n\n")
            f.write("\n".join(result_lines) + "\n")

        # Step 7: Create research log
        research_log = f"""
Protease Kinetics Analysis Research Log
======================================

Analysis Steps:
1. Calculated initial velocities from time-course fluorescence data for {len(substrate_concentrations)} different substrate concentrations
2. Fitted initial velocities to the Michaelis-Menten equation using non-linear regression
3. Determined kinetic parameters and their uncertainties

Results:
{chr(10).join("- " + line for line in result_lines)}

Files Generated:
1. {plot_filename} - Michaelis-Menten plot showing experimental data and fitted curve
2. {results_filename} - Text file containing detailed results

Analysis completed successfully.
"""

        return research_log

    except Exception as e:
        return f"Error during analysis: {str(e)}"


def analyze_enzyme_kinetics_assay(
    enzyme_name,
    substrate_concentrations,
    enzyme_concentration,
    modulators=None,
    time_points=None,
    output_dir="./",
    velocities=None,
    modulator_activities=None,
    time_course_activity=None,
):
    """Fit measured in vitro enzyme kinetics data: Michaelis-Menten parameters and modulator IC50s.

    The tool fits the measurements it is given; it does not simulate an assay. Without measured
    ``velocities`` it returns an error naming what is missing.

    Parameters
    ----------
    enzyme_name : str
        Name of the purified enzyme being tested
    substrate_concentrations : list or numpy.ndarray
        List of substrate concentrations in μM for kinetic analysis
    enzyme_concentration : float
        Concentration of the enzyme in nM
    modulators : dict, optional
        Dictionary of modulators where keys are modulator names and values are lists of
        concentrations in μM. Default is None (no modulators).
    time_points : list or numpy.ndarray, optional
        Time points in minutes of a measured time course (paired with ``time_course_activity``).
    output_dir : str, optional
        Directory to save output files. Default is current directory.
    velocities : list or numpy.ndarray
        Measured initial velocity at each substrate concentration (same order and length as
        ``substrate_concentrations``). Required.
    modulator_activities : dict, optional
        ``{modulator name: [activity as % of the uninhibited control at each concentration]}``,
        same order and length as that modulator's concentrations in ``modulators``.
    time_course_activity : list or numpy.ndarray, optional
        Measured activity at each of ``time_points``, used to report the linear range.

    Returns
    -------
    str
        Research log summarizing the fitted parameters, or an error naming the missing data

    """
    import csv
    import os

    import numpy as np
    from scipy.optimize import curve_fit

    # This tool used to take no measurements at all: it simulated activity from a hard-coded
    # Vmax=120/Km=25, drew a random IC50 per modulator (after reseeding the caller's global RNG)
    # and reported the fit of that noise as the enzyme's kinetics. It now fits only what it is
    # given and refuses without it (hunt 2026-09-30, uT2-pharmacology-1).
    if velocities is None:
        return (
            "Error: analyze_enzyme_kinetics_assay fits measured data and does not simulate an assay. "
            "Pass `velocities`: the measured initial velocity at each of the substrate_concentrations "
            "(same order and length). For modulators also pass `modulator_activities` "
            "({name: [% of control activity at each concentration]}), and for a time course pass "
            "`time_points` with `time_course_activity`."
        )

    try:
        substrate = np.asarray(substrate_concentrations, dtype=float).ravel()
        rates = np.asarray(velocities, dtype=float).ravel()
    except (TypeError, ValueError) as e:
        return f"Error: substrate_concentrations and velocities must be numeric: {e}"
    if substrate.size != rates.size:
        return (
            f"Error: {rates.size} velocities for {substrate.size} substrate concentrations; "
            "pass one measured velocity per concentration."
        )
    finite = np.isfinite(substrate) & np.isfinite(rates)
    if int(finite.sum()) < 3:
        return "Error: at least 3 finite (substrate concentration, velocity) pairs are needed to fit Km and Vmax."
    substrate, rates = substrate[finite], rates[finite]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Michaelis-Menten equation for curve fitting
    def michaelis_menten(s, vmax, km):
        return vmax * s / (km + s)

    log = f"## In Vitro Enzyme Kinetics Analysis: {enzyme_name}\n\n"
    log += f"Enzyme concentration: {enzyme_concentration} nM\n"
    log += "All values below are fitted to the measurements supplied by the caller.\n"
    summary = []

    # 1. Time course (only when one was measured)
    if time_course_activity is not None:
        log += "\n### Time-Course Linear Range\n\n"
        if time_points is None:
            log += "Not analysed: time_course_activity was given without its time_points.\n"
        else:
            t = np.asarray(time_points, dtype=float).ravel()
            a = np.asarray(time_course_activity, dtype=float).ravel()
            if t.size != a.size or t.size < 2:
                log += f"Not analysed: {a.size} activities for {t.size} time points (need equal lengths, >= 2).\n"
            else:
                order = np.argsort(t)
                t, a = t[order], a[order]
                rise = a - a[0]
                span = np.nanmax(rise)
                if not np.isfinite(span) or span <= 0:
                    log += "Not analysed: the measured activity never rises above its first value.\n"
                else:
                    # The linear range ends at the first point reaching 30% of the observed rise.
                    cutoff = int(np.flatnonzero(rise >= 0.3 * span)[0])
                    linear_end = t[cutoff]
                    log += f"Linear range (to 30% of the observed rise): {t[0]:g}-{linear_end:g} minutes\n"
                    summary.append(f"- Linear range of the measured time course: {t[0]:g}-{linear_end:g} minutes")
                time_course_file = os.path.abspath(os.path.join(output_dir, f"{enzyme_name}_time_course.csv"))
                with open(time_course_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Time (min)", "Activity (as measured)"])
                    writer.writerows(zip(t.tolist(), a.tolist(), strict=True))
                log += f"Time-course data saved to: {time_course_file}\n"
    elif time_points is not None:
        log += "\nNote: time_points were given without time_course_activity, so no time course was analysed.\n"

    # 2. Substrate kinetics (Michaelis-Menten fit of the measured velocities)
    log += "\n### Substrate Kinetics Analysis\n\n"
    try:
        positive = substrate[substrate > 0]
        vmax, km, vmax_err, km_err = _fit_michaelis_menten(
            substrate, rates, float(np.median(positive)) if positive.size else 1.0
        )
        fitted = michaelis_menten(substrate, vmax, km)
        ss_tot = float(np.sum((rates - rates.mean()) ** 2))
        r_squared = 1 - float(np.sum((rates - fitted) ** 2)) / ss_tot if ss_tot > 0 else float("nan")

        kinetics_file = os.path.abspath(os.path.join(output_dir, f"{enzyme_name}_substrate_kinetics.csv"))
        with open(kinetics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Substrate (μM)", "Measured velocity", "Fitted velocity"])
            writer.writerows(zip(substrate.tolist(), rates.tolist(), fitted.tolist(), strict=True))

        log += "Michaelis-Menten parameters (fitted to the measured velocities):\n"
        log += f"- Vmax: {vmax:.4g} ± {vmax_err:.2g} (velocity units as supplied)\n"
        log += f"- Km: {km:.4g} ± {km_err:.2g} μM\n"
        log += f"- R-squared: {r_squared:.4f}\n"
        if km > substrate.max():
            log += "- Caution: Km exceeds the highest substrate concentration tested, so it is extrapolated.\n"
        log += f"Substrate kinetics data saved to: {kinetics_file}\n"
        summary.append(f"- Michaelis-Menten fit: Vmax {vmax:.4g} (as supplied), Km {km:.4g} μM, R² {r_squared:.3f}")
    except Exception as e:
        log += f"Error: could not fit the measured velocities to the Michaelis-Menten model: {e}\n"
        summary.append("- Michaelis-Menten fit failed (see above)")

    # 3. Modulator effects (only those whose activities were measured)
    if modulators:
        log += "\n### Modulator Effects Analysis\n\n"
        measured = modulator_activities or {}

        def dose_response(x, ic50, hill):
            return 100 / (1 + (x / ic50) ** hill)

        for modulator_name, concentrations in modulators.items():
            log += f"#### Modulator: {modulator_name}\n\n"
            if modulator_name not in measured:
                log += (
                    f"Not analysed: no measured activities for {modulator_name}; pass "
                    f"modulator_activities={{'{modulator_name}': [% of control at each concentration]}}.\n\n"
                )
                summary.append(f"- {modulator_name}: not analysed (no measured activities)")
                continue
            conc = np.asarray(concentrations, dtype=float).ravel()
            act = np.asarray(measured[modulator_name], dtype=float).ravel()
            if conc.size != act.size:
                log += f"Not analysed: {act.size} activities for {conc.size} concentrations.\n\n"
                summary.append(f"- {modulator_name}: not analysed (length mismatch)")
                continue

            modulator_file = os.path.abspath(
                os.path.join(output_dir, f"{enzyme_name}_{modulator_name}_dose_response.csv")
            )
            with open(modulator_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"{modulator_name} (μM)", "Activity (% of control, as measured)"])
                writer.writerows(zip(conc.tolist(), act.tolist(), strict=True))

            keep = (conc > 0) & np.isfinite(conc) & np.isfinite(act)
            if int(keep.sum()) < 3:
                log += f"Insufficient non-zero data points (need 3) to calculate IC50 for {modulator_name}.\n"
                summary.append(f"- {modulator_name}: IC50 not calculated (fewer than 3 non-zero doses)")
            else:
                try:
                    lo, hi = float(conc[keep].min()), float(conc[keep].max())
                    params, _ = curve_fit(
                        dose_response,
                        conc[keep],
                        act[keep],
                        p0=[float(np.sqrt(lo * hi)), 1.0],
                        bounds=([lo / 1000, 0.1], [hi * 1000, 10]),
                        maxfev=10000,
                    )
                    calc_ic50, calc_hill = params
                    log += f"Dose-response fit for {modulator_name} (to the measured activities):\n"
                    log += f"- IC50: {calc_ic50:.4g} μM\n"
                    log += f"- Hill coefficient: {calc_hill:.2f}\n"
                    if not lo <= calc_ic50 <= hi:
                        log += "- Caution: the IC50 lies outside the tested concentration range (extrapolated).\n"
                    summary.append(f"- {modulator_name}: IC50 {calc_ic50:.4g} μM (Hill {calc_hill:.2f})")
                except Exception as e:
                    log += f"Error: could not fit a dose-response curve for {modulator_name}: {e}\n"
                    summary.append(f"- {modulator_name}: dose-response fit failed")
            log += f"Dose-response data for {modulator_name} saved to: {modulator_file}\n\n"

    # 4. Summary of what was actually fitted
    log += "\n### Summary\n\n"
    log += f"Fitted the supplied kinetics measurements for {enzyme_name}.\n"
    log += "\n".join(summary) + "\n"

    return log


def analyze_itc_binding_thermodynamics(
    itc_data_path=None,
    itc_data=None,
    temperature=298.15,
    protein_concentration=None,
    ligand_concentration=None,
    volume_unit="uL",
    cell_volume_ml=1.4,
    heat_unit="ucal",
):
    """Analyzes isothermal titration calorimetry (ITC) data to determine binding affinity and thermodynamic parameters.

    Parameters
    ----------
    itc_data_path : str, optional
        Path to CSV or TSV file containing ITC thermogram data with columns for injection number,
        injection volume, and heat released/absorbed. Expected columns: 'injection', 'volume', 'heat'
    itc_data : numpy.ndarray, optional
        Raw ITC thermogram data as a numpy array.
        Expected shape: (n_injections, 3) with columns for injection number, injection volume, and heat
        This parameter is provided for backward compatibility and will be deprecated.
    temperature : float, optional
        Temperature in Kelvin at which the experiment was conducted. Default is 298.15 K (25°C).
    protein_concentration : float, optional
        Initial concentration of protein in the cell in molar (M). Required for accurate fitting.
    ligand_concentration : float, optional
        Concentration of ligand in the syringe in molar (M). Required: no fit is attempted without both
        concentrations.
    volume_unit : str, optional
        Unit of the injection 'volume' column: 'uL' (default, the ITC convention), 'mL' or 'L'.
    cell_volume_ml : float, optional
        Active cell volume in mL (default 1.4, a VP-ITC; about 0.2 for an ITC200/PEAQ-ITC).
    heat_unit : str, optional
        Unit of the 'heat' column: heat per injection in 'ucal' (default), 'mcal', 'cal', 'uJ', 'mJ' or
        'J', or heat normalised per mole of injectant in 'kcal/mol' or 'kJ/mol'. Heats are taken as
        already corrected for the heat of dilution.

    Returns
    -------
    str
        A research log summarizing the analysis steps and results, including binding affinity (Kd),
        binding enthalpy (ΔH), binding entropy (ΔS), and Gibbs free energy (ΔG).

    """
    import datetime

    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    log = []
    log.append(f"# ITC Binding Affinity Analysis - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.append("\n## Data Preprocessing")

    # Check if we have data to process
    if itc_data_path is None and itc_data is None:
        log.append("Error: No data provided. Please provide either itc_data_path or itc_data.")
        return "\n".join(log)

    # Load data from file if path is provided
    if itc_data_path is not None:
        try:
            if itc_data_path.endswith(".csv"):
                loaded_data = pd.read_csv(itc_data_path)
            elif itc_data_path.endswith((".tsv", ".txt")):
                loaded_data = pd.read_csv(itc_data_path, sep="\t")
            else:
                log.append("Error: Unsupported file format. Please provide a CSV or TSV file.")
                return "\n".join(log)

            log.append(f"- Loaded data from file: {itc_data_path}")
            log.append(f"- Input data: DataFrame with {len(loaded_data)} injections")

            if all(col in loaded_data.columns for col in ["injection", "volume", "heat"]):
                data = loaded_data[["injection", "volume", "heat"]].values
            else:
                log.append("- Error: DataFrame must contain 'injection', 'volume', and 'heat' columns")
                return "\n".join(log)
        except Exception as e:
            log.append(f"- Error loading data from file: {str(e)}")
            return "\n".join(log)
    # Use provided numpy array data
    elif itc_data is not None:
        if isinstance(itc_data, pd.DataFrame):
            log.append("Warning: Passing DataFrame directly is deprecated. Please use itc_data_path instead.")
            log.append(f"- Input data: DataFrame with {len(itc_data)} injections")
            if all(col in itc_data.columns for col in ["injection", "volume", "heat"]):
                data = itc_data[["injection", "volume", "heat"]].values
            else:
                log.append("- Error: DataFrame must contain 'injection', 'volume', and 'heat' columns")
                return "\n".join(log)
        else:
            log.append(f"- Input data: Array with {len(itc_data)} injections")
            data = np.array(itc_data)

    # The model below is the standard single-set-of-identical-sites (Wiseman) isotherm with the
    # MicroCal displaced-volume correction. The previous one divided a cumulative injection volume of
    # undeclared units by a 1.4 mL cell, so conventional microlitre injections made every protein
    # concentration negative; it substituted 1 M / 10 M when no concentrations were given and still
    # printed an absolute Kd; and it logged 'Model fitting successful' and 'thermodynamic parameters
    # have been estimated' whatever the R-squared, even after the fit raised (hunt 2026-09-30,
    # uT2-pharmacology-14).
    if protein_concentration is None or ligand_concentration is None:
        log.append(
            "- Error: protein_concentration (cell) and ligand_concentration (syringe), both in molar, are "
            "required to fit Kd, n and ΔH; no fit was attempted."
        )
        return "\n".join(log)

    volume_scale = {"ul": 1e-6, "µl": 1e-6, "μl": 1e-6, "ml": 1e-3, "l": 1.0}.get(str(volume_unit).strip().lower())
    per_injection = {"ucal": 1e-6, "µcal": 1e-6, "μcal": 1e-6, "mcal": 1e-3, "cal": 1.0}
    per_injection.update({"uj": 1e-6 / 4.184, "µj": 1e-6 / 4.184, "mj": 1e-3 / 4.184, "j": 1 / 4.184})
    normalised = {"kcal/mol": 1e3, "kj/mol": 1e3 / 4.184}
    unit_key = str(heat_unit).strip().lower()
    if volume_scale is None or (unit_key not in per_injection and unit_key not in normalised):
        log.append(
            f"- Error: unsupported volume_unit {volume_unit!r} or heat_unit {heat_unit!r}; use volume_unit "
            "'uL', 'mL' or 'L', and heat_unit 'ucal', 'mcal', 'cal', 'uJ', 'mJ', 'J', 'kcal/mol' or 'kJ/mol'."
        )
        return "\n".join(log)

    # Extract data
    data = np.asarray(data, dtype=float)
    injections = data[:, 0]
    dv = data[:, 1] * volume_scale  # L
    v0 = float(cell_volume_ml) * 1e-3  # L
    m0 = float(protein_concentration)
    x0 = float(ligand_concentration)
    if unit_key in per_injection:
        heats = data[:, 2] * per_injection[unit_key]  # cal per injection
    else:
        heats = data[:, 2] * normalised[unit_key] * x0 * dv  # cal/mol injectant -> cal per injection

    log.append(f"- Processed {len(injections)} injections")
    log.append(f"- Cell volume {cell_volume_ml} mL; injection volumes read as {volume_unit}; heats read as {heat_unit}")
    cumulative = np.cumsum(dv)
    if np.any(dv <= 0) or cumulative[-1] >= v0:
        log.append(
            f"- Error: the injection volumes total {cumulative[-1] * 1e6:.1f} µL against a "
            f"{cell_volume_ml} mL cell; check volume_unit and cell_volume_ml."
        )
        return "\n".join(log)

    log.append("\n## Model Fitting")
    log.append("- One set of identical sites (Wiseman isotherm) with the displaced-volume correction")

    def one_site_model(_x, log10_kd, dH, n):
        """Heat of each injection (cal) for dissociation constant 10**log10_kd (M), ΔH (cal/mol), n."""
        ka = 10.0 ** (-log10_kd)
        half = cumulative / (2 * v0)
        mt = m0 * (1 - half) / (1 + half)
        xt = x0 * (cumulative / v0) * (1 - half)
        r = xt / (n * mt)
        b = 1 + r + 1 / (n * ka * mt)
        q = n * mt * dH * v0 / 2 * (b - np.sqrt(np.maximum(b * b - 4 * r, 0.0)))
        q_prev = np.concatenate(([0.0], q[:-1]))
        return q + (dv / v0) * (q + q_prev) / 2 - q_prev

    # Initial guesses: a tight binder takes up all of the first injection, so ΔQ1 ≈ ΔH·X0·dV1.
    dh0 = heats[0] / (x0 * dv[0]) if heats[0] != 0 else -5000.0

    # The heats are about 1e-6 cal per injection, and curve_fit's absolute gradient tolerance stopped it at
    # its starting guess on ITC200-scale titrations: Kd 0.2 µM and 5 µM both came back as the p0 of 1 µM
    # with R² 0.94 / 0.92 and 'Model fitting successful'. The fit now runs on heats scaled to max |q| = 1,
    # with ΔH in units of its first-injection estimate, from several starting Kd, and a fit that never left
    # its start, or whose n ran to a bound, is a failure. The R² floor stays 0.9: converged fits of noisy
    # c ~ 1 titrations score 0.90-0.98 with Kd within ~50%, so a higher floor refuses real data
    # (hunt 2026-09-30, uT2-pharmacology-14).
    heat_scale = float(np.max(np.abs(heats))) if heats.size else 0.0
    if not np.isfinite(heat_scale) or heat_scale == 0:
        log.append("- Error: every injection heat is zero or not finite; there is no isotherm to fit.")
        return "\n".join(log)
    dh_scale = min(abs(dh0), 1e6) if np.isfinite(dh0) and dh0 != 0 else 5000.0
    dh_sign = -1.0 if dh0 < 0 else 1.0
    y = heats / heat_scale

    def scaled_model(x, log10_kd, h, n):
        return one_site_model(x, log10_kd, h * dh_scale, n) / heat_scale

    bounds = ([-12.0, -1e7 / dh_scale, 0.01], [0.0, 1e7 / dh_scale, 20.0])
    min_r_squared = 0.9
    best, errors = None, []
    for start_log10_kd in (-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0):
        p0 = [start_log10_kd, dh_sign, 1.0]
        try:
            popt, pcov = curve_fit(scaled_model, injections, y, p0=p0, bounds=bounds, x_scale="jac", maxfev=20000)
        except Exception as e:
            errors.append(str(e))
            continue
        sse = float(np.sum((y - scaled_model(injections, *popt)) ** 2))
        if np.isfinite(sse) and (best is None or sse < best[0]):
            best = (sse, popt, pcov, p0)
    if best is None:
        log.append(f"- Fit failed: {errors[-1] if errors else 'no starting point gave a finite fit'}")
        log.append("\n## Conclusion")
        log.append("- No thermodynamic parameters were estimated: the one-site model could not be fitted.")
        return "\n".join(log)

    _, popt, pcov, p0 = best
    log10_kd, dH, n = popt[0], popt[1] * dh_scale, popt[2]
    perr = np.sqrt(np.abs(np.diag(pcov)))
    residuals = heats - one_site_model(injections, log10_kd, dH, n)
    ss_tot = np.sum((heats - np.mean(heats)) ** 2)
    r_squared = 1 - np.sum(residuals**2) / ss_tot if ss_tot > 0 else float("nan")

    never_moved = np.allclose(popt, p0, rtol=1e-6, atol=1e-9)
    kd_at_bound = np.isclose(log10_kd, bounds[0][0]) or np.isclose(log10_kd, bounds[1][0])
    n_at_bound = np.isclose(n, bounds[0][2]) or np.isclose(n, bounds[1][2])
    if never_moved or not np.isfinite(r_squared) or r_squared < min_r_squared or kd_at_bound or n_at_bound:
        if never_moved:
            reason = "the optimiser never moved from its starting guess, so nothing was fitted"
        elif not (np.isfinite(r_squared) and r_squared >= min_r_squared):
            reason = f"R-squared {r_squared:.3f} < {min_r_squared}"
        else:
            reason = "Kd ran to its bound" if kd_at_bound else "n ran to its bound"
        log.append(f"- Fit rejected ({reason}): the one-site model does not describe these data")
        log.append("\n## Conclusion")
        log.append(
            "- No thermodynamic parameters are reported. Check the concentrations, volume_unit, heat_unit, "
            "cell_volume_ml and that the heat of dilution was subtracted, or use a multi-site model."
        )
        return "\n".join(log)

    Kd = 10.0**log10_kd
    Kd_err = Kd * np.log(10) * perr[0]
    dH_err, n_err = perr[1] * dh_scale, perr[2]

    # Calculate other thermodynamic parameters
    R = 1.9872  # Gas constant in cal/(mol·K)
    dG = R * temperature * np.log(Kd)  # Gibbs free energy (1 M standard state)
    dS = (dH - dG) / temperature  # Entropy
    c_value = n * m0 / Kd

    log.append(f"- Model fitting successful (R-squared: {r_squared:.4f})")
    log.append("\n## Results")
    log.append(f"- Binding Stoichiometry (n): {n:.2f} ± {n_err:.2f}")
    log.append(f"- Dissociation Constant (Kd): {Kd * 1e6:.4g} ± {Kd_err * 1e6:.2g} μM")
    log.append(f"- Association Constant (Ka): {1 / Kd:.4g} M^-1")
    log.append(f"- Binding Enthalpy (ΔH): {dH:.0f} ± {dH_err:.0f} cal/mol")
    log.append(f"- Binding Entropy (ΔS): {dS:.2f} cal/(mol·K)")
    log.append(f"- Gibbs Free Energy (ΔG): {dG:.0f} cal/mol")
    log.append(f"- R-squared: {r_squared:.4f}")
    log.append(f"- Wiseman c-value (n·[P]/Kd): {c_value:.3g}")
    if not 1 <= c_value <= 1000:
        log.append("- Caution: c is outside 1-1000, so Kd (and n) are poorly determined by this titration.")

    log.append("\n## Conclusion")
    log.append("- The thermodynamic parameters above were estimated with a one-site binding model.")
    log.append(
        "- For more complex binding scenarios, consider using multi-site binding models or specialized ITC analysis software."
    )

    return "\n".join(log)


def analyze_protein_conservation(protein_sequences, output_dir="./", muscle_timeout_s=600):
    """Perform multiple sequence alignment and phylogenetic analysis to identify conserved protein regions.

    Parameters
    ----------
    protein_sequences : list of str
        List of protein sequences in FASTA format from multiple organisms.
    output_dir : str, optional
        Directory to save output files (default: "./"; with the default, a later call's files get a _2,
        _3, ... suffix instead of overwriting an earlier result)
    muscle_timeout_s : int, optional
        Seconds to allow the MUSCLE alignment (default 600)

    Returns
    -------
    str
        Research log summarizing the analysis steps and results, including filenames of saved outputs.

    """
    import os
    import shutil
    import subprocess

    from Bio import AlignIO, Phylo, SeqIO
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

    # Create a research log
    log = []
    log.append("# Protein Sequence Alignment and Conservation Analysis")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # With the default ./ a second call overwrote the alignment, tree and scores the first log names, and
    # wrote its input over any input.fasta already in the working directory (hunt 2026-09-30,
    # uT2-pharmacology-23); an output_dir the caller names is used as given.
    output_names = ["input.fasta", "aligned.fasta", "tree.newick", "conservation.txt"]
    if output_dir == "./":
        from spatialomicsgym.tool.pharmacology import _fresh_output_paths

        input_file, aligned_file, tree_file, conservation_file = _fresh_output_paths(output_dir, output_names)
    else:
        input_file, aligned_file, tree_file, conservation_file = [os.path.join(output_dir, n) for n in output_names]

    # Step 1: Save input sequences to a temporary file
    log.append("\n## Step 1: Preparing Input Sequences")

    # Check if input is already in FASTA format or needs conversion
    if isinstance(protein_sequences, list):
        if all(">" in seq for seq in protein_sequences):
            # Already in FASTA format
            with open(input_file, "w") as f:
                f.write("\n".join(protein_sequences))
        else:
            # Convert to FASTA format
            with open(input_file, "w") as f:
                for i, seq in enumerate(protein_sequences):
                    f.write(f">Sequence_{i + 1}\n{seq}\n")
    else:
        # Assume it's a single string in FASTA format
        with open(input_file, "w") as f:
            f.write(protein_sequences)

    log.append(f"Input sequences saved to {input_file}")
    log.append(f"Number of sequences: {len(list(SeqIO.parse(input_file, 'fasta')))}")

    # Step 2: Perform multiple sequence alignment using MUSCLE
    log.append("\n## Step 2: Multiple Sequence Alignment")

    # Bio.Align.Applications (MuscleCommandline) is gone from Biopython 1.86, the version the full env
    # pins, so every call died on the import; on older Biopython a MUSCLE failure fell back to padding
    # the sequences with '-', and conservation and the tree were computed over unaligned columns under
    # 'The analysis successfully completed'. MUSCLE now runs as a subprocess (v5 syntax, then v3), and
    # with no aligner the tool refuses unless the input is already an alignment (hunt 2026-09-30,
    # uT2-pharmacology-15).
    records = list(SeqIO.parse(input_file, "fasta"))
    muscle = shutil.which("muscle")
    if os.path.exists(aligned_file):
        os.remove(aligned_file)  # never read a previous call's alignment as this one's
    if muscle:
        failures = []
        for cmd in (
            [muscle, "-align", input_file, "-output", aligned_file],  # MUSCLE v5
            [muscle, "-in", input_file, "-out", aligned_file],  # MUSCLE v3
        ):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=muscle_timeout_s)
            except subprocess.TimeoutExpired:
                log.append(f"Error: MUSCLE did not finish within muscle_timeout_s={muscle_timeout_s} s; raise it.")
                return "\n".join(log)
            if proc.returncode == 0 and os.path.exists(aligned_file) and os.path.getsize(aligned_file) > 0:
                break
            failures.append((proc.stderr or proc.stdout or "").strip()[-300:])
        else:
            log.append(f"Error: MUSCLE ({muscle}) failed to align the sequences: {' | '.join(failures)}")
            return "\n".join(log)
        log.append(f"Multiple sequence alignment completed using MUSCLE ({muscle})")
    elif len({len(r.seq) for r in records}) == 1 and any("-" in str(r.seq) for r in records):
        SeqIO.write(records, aligned_file, "fasta")
        log.append(
            "MUSCLE is not installed; the input is already aligned (equal lengths, '-' gaps) and is used as given"
        )
    else:
        log.append(
            "Error: MUSCLE ('muscle') is not on PATH, so no alignment was made, and conservation or a tree "
            "over unaligned sequences would be meaningless. Install MUSCLE in this environment, or pass "
            "sequences that are already aligned (equal length, with '-' gaps)."
        )
        return "\n".join(log)

    log.append(f"Alignment saved to {aligned_file}")
    alignment = AlignIO.read(aligned_file, "fasta")
    log.append(f"Alignment length: {alignment.get_alignment_length()} positions")

    # Step 3: Generate a phylogenetic tree
    log.append("\n## Step 3: Phylogenetic Analysis")

    # Calculate distance matrix
    calculator = DistanceCalculator("identity")
    dm = calculator.get_distance(alignment)

    # Construct the phylogenetic tree using neighbor-joining method
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)

    # Save the tree
    Phylo.write(tree, tree_file, "newick")
    log.append("Phylogenetic tree constructed using neighbor-joining method")
    log.append(f"Tree saved to {tree_file}")

    # Step 4: Analyze conserved regions
    log.append("\n## Step 4: Conservation Analysis")

    # Simple conservation analysis
    alignment_length = alignment.get_alignment_length()
    conserved_positions = []

    with open(conservation_file, "w") as f:
        f.write("Position\tConservation_Score\tConsensus\n")

        for i in range(alignment_length):
            # Get all amino acids at this position
            column = alignment[:, i]
            set(column)

            # Calculate a simple conservation score (percentage of most common AA)
            most_common_aa = max(column, key=column.count)
            conservation_score = column.count(most_common_aa) / len(column)

            f.write(f"{i + 1}\t{conservation_score:.2f}\t{most_common_aa}\n")

            # Consider positions with >80% conservation as conserved
            if conservation_score > 0.8:
                conserved_positions.append(i + 1)

    log.append(f"Conservation analysis completed and saved to {conservation_file}")
    log.append(f"Identified {len(conserved_positions)} highly conserved positions (>80% conservation)")

    if conserved_positions:
        log.append(f"Conserved positions: {', '.join(map(str, conserved_positions[:10]))}")
        if len(conserved_positions) > 10:
            log.append(f"... and {len(conserved_positions) - 10} more")

    # Final summary
    log.append("\n## Summary")
    log.append("The analysis successfully completed with the following outputs:")
    log.append(f"1. Multiple sequence alignment: {aligned_file}")
    log.append(f"2. Phylogenetic tree: {tree_file}")
    log.append(f"3. Conservation analysis: {conservation_file}")

    return "\n".join(log)
