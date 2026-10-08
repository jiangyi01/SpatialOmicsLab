description = [
    {
        "description": "Analyzes circular dichroism (CD) spectroscopy data to "
        "determine secondary structure and thermal stability.",
        "name": "analyze_circular_dichroism_spectra",
        "optional_parameters": [
            {
                "default": None,
                "description": "Temperature values (°C) for thermal denaturation experiment",
                "name": "temperature_data",
                "type": "list or numpy.ndarray",
            },
            {
                "default": None,
                "description": "CD signal values at specific wavelength across different temperatures",
                "name": "thermal_cd_data",
                "type": "list or numpy.ndarray",
            },
            {
                "default": "./",
                "description": "Directory to save result files",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": 'Name of the biomolecule sample (e.g., "Znf706", "G-quadruplex")',
                "name": "sample_name",
                "type": "str",
            },
            {
                "default": None,
                "description": 'Type of biomolecule ("protein" or "nucleic_acid")',
                "name": "sample_type",
                "type": "str",
            },
            {
                "default": None,
                "description": "Wavelength values in nm for CD spectrum",
                "name": "wavelength_data",
                "type": "list or numpy.ndarray",
            },
            {
                "default": None,
                "description": "CD signal intensity values (typically in mdeg or Δε)",
                "name": "cd_signal_data",
                "type": "list or numpy.ndarray",
            },
        ],
    },
    {
        "description": "Calculate numeric values for various structural features of an RNA secondary structure.",
        "name": "analyze_rna_secondary_structure_features",
        "optional_parameters": [
            {
                "default": None,
                "description": "The RNA sequence corresponding to "
                "the structure. If provided, "
                "sequence-dependent energy "
                "calculations will be performed.",
                "name": "sequence",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "RNA secondary structure in "
                "dot-bracket notation (e.g., "
                '"(((...)))"). Parentheses represent '
                "base pairs, dots represent unpaired "
                "bases.",
                "name": "dot_bracket_structure",
                "type": "str",
            }
        ],
    },
    {
        "description": "Analyze protease kinetics data from fluorogenic peptide "
        "cleavage assays, fit the data to Michaelis-Menten kinetics, "
        "and determine Vmax and KM; kcat and kcat/KM need a product "
        "fluorescence calibration (fluorescence_per_uM_product).",
        "name": "analyze_protease_kinetics",
        "optional_parameters": [
            {
                "default": "protease_kinetics",
                "description": "Prefix for output files",
                "name": "output_prefix",
                "type": "str",
            },
            {
                "default": "./",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": None,
                "description": "Fluorescence units per μM of cleaved product (slope of a product standard "
                "curve); needed to report kcat in s^-1",
                "name": "fluorescence_per_uM_product",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Array of time points (in seconds) at which measurements were taken",
                "name": "time_points",
                "type": "numpy.ndarray",
            },
            {
                "default": None,
                "description": "2D array of fluorescence "
                "measurements where each row "
                "corresponds to a different "
                "substrate concentration and each "
                "column corresponds to a time point",
                "name": "fluorescence_data",
                "type": "numpy.ndarray",
            },
            {
                "default": None,
                "description": "Array of substrate concentrations "
                "(in μM) corresponding to each row "
                "in fluorescence_data",
                "name": "substrate_concentrations",
                "type": "numpy.ndarray",
            },
            {
                "default": None,
                "description": "Concentration of the protease enzyme (in μM)",
                "name": "enzyme_concentration",
                "type": "float",
            },
        ],
    },
    {
        "description": "Fits measured in vitro enzyme kinetics data: Michaelis-Menten Vmax and Km from "
        "measured initial velocities, and modulator IC50s from measured dose-response "
        "activities. It fits the measurements given and does not simulate an assay.",
        "name": "analyze_enzyme_kinetics_assay",
        "optional_parameters": [
            {
                "default": None,
                "description": "Dictionary of modulators where keys "
                "are modulator names and values are "
                "lists of concentrations in μM",
                "name": "modulators",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Measured activity of each modulator as % of the "
                "uninhibited control, {name: [value at each concentration]}, "
                "in the same order as its concentrations in modulators",
                "name": "modulator_activities",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Time points in minutes of a measured time course (with time_course_activity)",
                "name": "time_points",
                "type": "list or numpy.ndarray",
            },
            {
                "default": None,
                "description": "Measured activity at each of time_points, used to report the linear range",
                "name": "time_course_activity",
                "type": "list or numpy.ndarray",
            },
            {
                "default": "./",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Name of the purified enzyme being tested",
                "name": "enzyme_name",
                "type": "str",
            },
            {
                "default": None,
                "description": "List of substrate concentrations in μM for kinetic analysis",
                "name": "substrate_concentrations",
                "type": "list or numpy.ndarray",
            },
            {
                "default": None,
                "description": "Concentration of the enzyme in nM",
                "name": "enzyme_concentration",
                "type": "float",
            },
            {
                "default": None,
                "description": "Measured initial velocity at each substrate concentration, "
                "in the same order as substrate_concentrations",
                "name": "velocities",
                "type": "list or numpy.ndarray",
            },
        ],
    },
    {
        "description": "Analyzes isothermal titration calorimetry (ITC) data to "
        "determine binding affinity and thermodynamic parameters.",
        "name": "analyze_itc_binding_thermodynamics",
        "optional_parameters": [
            {
                "default": None,
                "description": "Path to CSV or TSV file containing "
                "ITC thermogram data with columns "
                "for injection number, injection "
                "volume, and heat released/absorbed.",
                "name": "itc_data_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Raw ITC thermogram data as a numpy "
                "array with shape (n_injections, 3) "
                "containing injection number, "
                "injection volume, and heat.",
                "name": "itc_data",
                "type": "numpy.ndarray",
            },
            {
                "default": 298.15,
                "description": "Temperature in Kelvin at which the experiment was conducted.",
                "name": "temperature",
                "type": "float",
            },
            {
                "default": None,
                "description": "Initial concentration of protein in the cell in molar (M); needed for any fit.",
                "name": "protein_concentration",
                "type": "float",
            },
            {
                "default": None,
                "description": "Concentration of ligand in the syringe in molar (M); needed for any fit.",
                "name": "ligand_concentration",
                "type": "float",
            },
            {
                "default": "uL",
                "description": "Unit of the injection volume column: 'uL', 'mL' or 'L'.",
                "name": "volume_unit",
                "type": "str",
            },
            {
                "default": 1.4,
                "description": "Active cell volume in mL (1.4 for a VP-ITC, about 0.2 for an ITC200/PEAQ-ITC).",
                "name": "cell_volume_ml",
                "type": "float",
            },
            {
                "default": "ucal",
                "description": "Unit of the heat column: per-injection 'ucal', 'mcal', 'cal', 'uJ', 'mJ' or 'J', "
                "or per mole of injectant 'kcal/mol' or 'kJ/mol'.",
                "name": "heat_unit",
                "type": "str",
            },
        ],
        "required_parameters": [],
    },
    {
        "description": "Perform multiple sequence alignment and phylogenetic "
        "analysis to identify conserved protein regions.",
        "name": "analyze_protein_conservation",
        "optional_parameters": [
            {
                "default": "./",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": 600,
                "description": "Seconds to allow the MUSCLE alignment",
                "name": "muscle_timeout_s",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "List of protein sequences in FASTA format from multiple organisms.",
                "name": "protein_sequences",
                "type": "list of str",
            }
        ],
    },
]
