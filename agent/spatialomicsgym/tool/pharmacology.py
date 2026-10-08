import os
import pickle
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from difflib import get_close_matches

import numpy as np
import pandas as pd


def _fresh_output_paths(directory, names):
    """Absolute paths in ``directory`` for a set of fixed-name outputs that overwrite no earlier call's files.

    Tools that take no output location, or are called with their default one, wrote fixed names into the
    working directory, so a second call replaced the files an earlier log still pointed at (hunt
    2026-09-30, uT2-pharmacology-23). The first call keeps the documented names; a later one gets one
    shared suffix (``_2``, ``_3``, ...) on all of them, so the files of one run stay together.
    """
    base = os.path.abspath(directory)
    n = 1
    while True:
        paths = []
        for name in names:
            stem, ext = os.path.splitext(name)
            paths.append(os.path.join(base, name if n == 1 else f"{stem}_{n}{ext}"))
        if not any(os.path.exists(path) for path in paths):
            return paths
        n += 1


def _fresh_output_path(name):
    """Absolute path for one fixed-name output that never overwrites an earlier call's result."""
    return _fresh_output_paths(os.path.dirname(name) or os.curdir, [os.path.basename(name)])[0]


def run_diffdock_with_smiles(
    pdb_path, smiles_string, local_output_dir, gpu_device=0, use_gpu=True, docker_timeout_s=3600
):
    # Every docker call ran with no timeout, and the inference `docker run` had no --rm, so each call left
    # a stopped container behind (hunt 2026-09-30, uT2-pharmacology-20). Without docker the tool now says
    # so up front (uT2-pharmacology-22).
    if shutil.which("docker") is None:
        return (
            "Error: run_diffdock_with_smiles runs DiffDock in a Docker container, and `docker` is not on PATH "
            "in this environment."
        )
    container = f"sog-diffdock-{uuid.uuid4().hex[:12]}"
    try:
        summary = []

        # Check if PDB file exists
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"The PDB file '{pdb_path}' does not exist.")
        summary.append(f"PDB file '{pdb_path}' found.")

        # Ensure the output directory exists
        if not os.path.exists(local_output_dir):
            os.makedirs(local_output_dir)
        summary.append(f"Output directory '{local_output_dir}' is ready.")

        # Pull the pre-built container from Docker Hub
        summary.append("Pulling DiffDock container from Docker Hub...")
        subprocess.run(["docker", "pull", "rbgcsail/diffdock"], check=True, timeout=docker_timeout_s)
        summary.append("DiffDock container pulled successfully.")

        # Check for GPU availability (if using GPU)
        if use_gpu:
            summary.append("Checking for GPU availability...")
            gpu_check = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--gpus",
                    "all",
                    "nvidia/cuda:11.7.1-devel-ubuntu22.04",
                    "nvidia-smi",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=docker_timeout_s,
            )
            summary.append(f"GPU Status: {gpu_check.stdout.strip()}")

        # Prepare the GPU flag
        gpu_flag = ["--gpus", f"device={gpu_device}"] if use_gpu else []

        # Docker run command
        summary.append("Running DiffDock inference...")
        run_command = (
            ["docker", "run", "--rm", "--name", container]
            + gpu_flag
            + [
                # Mount the local directory to /home/appuser/output inside the container
                "-v",
                f"{os.path.abspath(pdb_path)}:/home/appuser/input/protein.pdb",  # PDB file mount
                "-v",
                f"{os.path.abspath(local_output_dir)}:/home/appuser/output",  # Output directory mount
                "--entrypoint",
                "/bin/bash",
                "rbgcsail/diffdock",
                "-c",
                # Command to run inference using micromamba environment
                f"micromamba run -n diffdock python -m inference --config default_inference_args.yaml "
                f"--protein_path /home/appuser/input/protein.pdb --ligand '{smiles_string}' --out_dir /home/appuser/output",
            ]
        )

        # Execute the Docker command
        try:
            result = subprocess.run(run_command, check=False, capture_output=True, text=True, timeout=docker_timeout_s)
        except subprocess.TimeoutExpired:
            # Killing the docker client leaves the container running; remove it by name.
            subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True, timeout=120)
            raise

        # Check for errors
        if result.returncode != 0:
            summary.append(f"Error during inference: {result.stderr.strip()}")
            return "\n".join(summary)
        else:
            summary.append("DiffDock inference completed successfully.")
            summary.append(f"Results stored in '{local_output_dir}'.")

        return "\n".join(summary)

    except FileNotFoundError as e:
        return f"File error: {e}"
    except subprocess.TimeoutExpired as e:
        return (
            f"Error: `{' '.join(map(str, e.cmd[:3]))} ...` did not finish within docker_timeout_s={docker_timeout_s} s; "
            "raise docker_timeout_s to wait longer."
        )
    except subprocess.CalledProcessError as e:
        return f"Command execution error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"


def docking_autodock_vina(smiles_list, receptor_pdb_file, box_center, box_size, ncpu=1):
    # A missing PyTDC used to surface as a bare ModuleNotFoundError traceback (hunt 2026-09-30,
    # uT2-pharmacology-22); say what is missing, as the DeepPurpose tools do.
    try:
        from tdc import Oracle
    except ImportError:
        return (
            "ERROR: PyTDC (module 'tdc', with its pyscreener docking oracle) is not installed in this environment, "
            "so AutoDock Vina docking cannot run here."
        )

    log = []

    # Log the start of the process
    log.append("Step 1: Initializing the Oracle")
    log.append(f"Receptor PDB File: {receptor_pdb_file}")
    log.append(f"Box Center: {box_center}")
    log.append(f"Box Size: {box_size}")

    # Initialize the Oracle object
    oracle = Oracle(
        name="pyscreener",
        receptor_pdb_file=receptor_pdb_file,
        box_center=box_center,
        box_size=box_size,
        ncpu=ncpu,
    )
    log.append("Oracle initialized successfully.")

    # Log the list of SMILES strings
    log.append(f"\nStep 2: Processing SMILES strings: {smiles_list}")

    # Get the docking scores
    docking_scores = oracle(smiles_list)
    log.append(f"Docking scores calculated: {docking_scores}")

    # Create a dictionary mapping SMILES to their docking scores
    results_dict = dict(zip(smiles_list, docking_scores, strict=False))

    # Log the result mapping
    log.append("\nStep 3: Mapping SMILES to docking scores:")
    log.append(f"Results: {results_dict}")

    # Convert the log to a string and return it
    research_log = "\n".join(log)
    return research_log


def run_autosite(pdb_file, output_dir, spacing=1.0):
    # Both programs come from the ADFR suite, which no environment here installs (hunt 2026-09-30,
    # uT2-pharmacology-22).
    missing = [tool for tool in ("prepare_receptor", "autosite") if shutil.which(tool) is None]
    if missing:
        return f"Error: run_autosite needs the ADFR suite's {' and '.join(missing)} on PATH, and this environment has none."

    # Prepare the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert the PDB file to PDBQT format (assuming prepare_receptor4.py is accessible). The name was
    # pdb_file.replace('.pdb', '.pdbqt'): for a .ent or .cif input that IS the input, so prepare_receptor
    # overwrote the user's structure, and a '.pdb' in a directory name was rewritten too. It now goes in
    # output_dir under the input's stem (hunt 2026-09-30, uT2-pharmacology-24).
    pdbqt_file = os.path.join(output_dir, os.path.splitext(os.path.basename(pdb_file))[0] + ".pdbqt")
    subprocess.run(["prepare_receptor", "-r", pdb_file, "-o", pdbqt_file], check=True)

    # Run AutoSite
    autosite_cmd = [
        "autosite",
        "-r",
        pdbqt_file,
        "--spacing",
        str(spacing),
        "-o",
        output_dir,
    ]
    subprocess.run(autosite_cmd, check=True)

    # Parse the results to find the box center and size
    box_center, box_size = None, None
    log_path = os.path.join(output_dir, "_AutoSiteSummary.log")
    with open(log_path) as log_file:
        log_content = log_file.read()

        # Extract box center and size from the log (assuming standard output format)
        box_center_match = re.search(r"Box center:\s*\(([^)]+)\)", log_content)
        box_size_match = re.search(r"Box size:\s*\(([^)]+)\)", log_content)

        if box_center_match:
            box_center = box_center_match.group(1)
        if box_size_match:
            box_size = box_size_match.group(1)

    # Create a research log string
    research_log = f"AutoSite run for {pdb_file} with spacing {spacing}\n"
    research_log += f"Output directory: {output_dir}\n"
    if box_center and box_size:
        research_log += f"Box Center: {box_center}\nBox Size: {box_size}"
    else:
        research_log += "Box Center and Size information not found in log."

    return research_log


# Function to get TxGNN predictions and return a summarized string output
def retrieve_topk_repurposing_drugs_from_disease_txgnn(disease_name, data_lake_path, k=5):
    """This function computes TxGNN model predictions for drug repurposing. It takes in the paths to the data,
    the disease name, and returns a summary of the top K predicted drugs with their sigmoid-transformed scores.

    Args:
    - disease_name (str): The name of the disease for which the drug predictions are to be retrieved.
    - data_lake_path (str): The path to the data lake containing the TxGNN predictions.
    - k (int, optional): The number of top drug predictions to return. Defaults to 5.

    Returns:
    - str: A summary of the steps and the top K drug predictions with their scores.

    """

    # Sigmoid function to convert raw prediction scores
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Step 1: Load the mappings and prediction data from the provided paths
    name_mapping_path = os.path.join(data_lake_path, "txgnn_name_mapping.pkl")
    result_path = os.path.join(data_lake_path, "txgnn_prediction.pkl")

    # No setup step provisions these files (the Biomni data-lake download they came from was removed),
    # so name both and where they were looked for instead of raising on the first one (hunt 2026-09-30,
    # uT2-pharmacology-21).
    missing = [p for p in (name_mapping_path, result_path) if not os.path.isfile(p)]
    if missing:
        return (
            "Error: TxGNN prediction files not found: "
            + ", ".join(os.path.abspath(p) for p in missing)
            + ". This tool needs txgnn_name_mapping.pkl and txgnn_prediction.pkl in data_lake_path; "
            "no SpatialOmicsLab setup step downloads them."
        )

    with open(name_mapping_path, "rb") as f:
        mapping = pickle.load(f)

    with open(result_path, "rb") as f:
        result = pickle.load(f)

    # Step 2: Fuzzy match the disease name to find the closest match
    possible_diseases = result.keys()
    matched_disease = get_close_matches(disease_name, possible_diseases, n=1, cutoff=0.6)

    if not matched_disease:
        return f"Error: No matching disease found for '{disease_name}'. Please try a different name."

    matched_disease = matched_disease[0]

    # Step 3: Retrieve the prediction scores for the matched disease
    disease_predictions = result[matched_disease]

    # Step 4: Apply the sigmoid function to the raw prediction scores
    sigmoid_predictions = {drug_id: sigmoid(score) for drug_id, score in disease_predictions.items()}

    # Step 5: Sort the drugs by prediction score in descending order
    top_k_drugs = sorted(sigmoid_predictions.items(), key=lambda x: x[1], reverse=True)[:k]

    # Step 6: Map drug IDs to their names and format the results
    top_k_drug_names = [(mapping["id2name_drug"].get(drug_id, "Unknown Drug"), score) for drug_id, score in top_k_drugs]

    # Step 7: Create a human and LLM-friendly summary string
    summary = f"TxGNN Drug Repurposing Predictions for '{matched_disease}':\n"
    summary += f"Top {k} predicted drugs and their corresponding prediction scores (post-sigmoid transformation):\n"

    for i, (drug_name, score) in enumerate(top_k_drug_names, 1):
        summary += f"{i}. {drug_name} - Prediction Score: {score:.4f}\n"

    summary += "\nProcess Summary:\n"
    summary += f"- Fuzzy matching was used to match the input disease name to '{matched_disease}'.\n"
    summary += "- Sigmoid function was applied to raw prediction scores to convert them into probabilities.\n"
    summary += f"- The top {k} drugs were selected based on their prediction scores.\n"

    return summary


# ADMET prediction function with research log format
def predict_admet_properties(smiles_list, ADMET_model_type="MPNN"):
    try:
        from DeepPurpose import CompoundPred, utils
    except ImportError:
        return (
            "ERROR: DeepPurpose is not installed in this environment, so ADMET properties cannot be "
            "predicted. Install it where the agent runs (e.g. 'pip install DeepPurpose') and retry."
        )

    # Define available model types
    available_model_types = ["MPNN", "CNN", "Morgan"]

    # Check if the provided model type is valid
    if ADMET_model_type not in available_model_types:
        return f"Error: Invalid ADMET model type '{ADMET_model_type}'. Available options are: {', '.join(available_model_types)}."

    # Load pretrained ADMET models only once
    model_ADMETs = {}
    tasks = [
        "AqSolDB",
        "Caco2",
        "HIA",
        "Pgp_inhibitor",
        "Bioavailability",
        "BBB_MolNet",
        "PPBR",
        "CYP2C19",
        "CYP2D6",
        "CYP3A4",
        "CYP1A2",
        "CYP2C9",
        "ClinTox",
        "Lipo_AZ",
        "Half_life_eDrug3D",
        "Clearance_eDrug3D",
    ]

    for task in tasks:
        model_ADMETs[task + "_" + ADMET_model_type + "_model"] = CompoundPred.model_pretrained(
            model=task + "_" + ADMET_model_type + "_model"
        )

    # Helper function for ADMET prediction
    def ADMET_pred(drug, task, unit):
        model = model_ADMETs[task + "_" + ADMET_model_type + "_model"]
        X_pred = utils.data_process(
            X_drug=[drug],
            y=[0],
            drug_encoding=ADMET_model_type,
            split_method="no_split",
        )
        y_pred = model.predict(X_pred)[0]

        if unit == "%":
            y_pred = y_pred * 100

        return f"{y_pred:.2f} " + unit

    # Initialize research log string
    research_log = "Research Log for ADMET Predictions:\n"
    research_log += "-------------------------------------\n"

    # Process each SMILES string in the list
    for smiles in smiles_list:
        research_log += f"\nCompound SMILES: {smiles}\n"
        research_log += "Predicted ADMET properties:\n"

        # Physiochemical properties
        solubility = ADMET_pred(smiles, "AqSolDB", "log mol/L")
        lipophilicity = ADMET_pred(smiles, "Lipo_AZ", "(log-ratio)")
        research_log += f"- Solubility: {solubility}\n"
        research_log += f"- Lipophilicity: {lipophilicity}\n"

        # Absorption
        caco2 = ADMET_pred(smiles, "Caco2", "cm/s")
        hia = ADMET_pred(smiles, "HIA", "%")
        pgp = ADMET_pred(smiles, "Pgp_inhibitor", "%")
        bioavail = ADMET_pred(smiles, "Bioavailability", "%")
        research_log += f"- Absorption (Caco-2 permeability): {caco2}\n"
        research_log += f"- Absorption (HIA): {hia}\n"
        research_log += f"- Absorption (Pgp Inhibitor): {pgp}\n"
        research_log += f"- Absorption (Bioavailability): {bioavail}\n"

        # Distribution
        bbb = ADMET_pred(smiles, "BBB_MolNet", "%")
        ppbr = ADMET_pred(smiles, "PPBR", "%")
        research_log += f"- Distribution (BBB permeation): {bbb}\n"
        research_log += f"- Distribution (PPBR): {ppbr}\n"

        # Metabolism
        cyp2c19 = ADMET_pred(smiles, "CYP2C19", "%")
        cyp2d6 = ADMET_pred(smiles, "CYP2D6", "%")
        cyp3a4 = ADMET_pred(smiles, "CYP3A4", "%")
        cyp1a2 = ADMET_pred(smiles, "CYP1A2", "%")
        cyp2c9 = ADMET_pred(smiles, "CYP2C9", "%")
        research_log += f"- Metabolism (CYP2C19): {cyp2c19}\n"
        research_log += f"- Metabolism (CYP2D6): {cyp2d6}\n"
        research_log += f"- Metabolism (CYP3A4): {cyp3a4}\n"
        research_log += f"- Metabolism (CYP1A2): {cyp1a2}\n"
        research_log += f"- Metabolism (CYP2C9): {cyp2c9}\n"

        # Excretion
        half_life = ADMET_pred(smiles, "Half_life_eDrug3D", "h")
        clearance = ADMET_pred(smiles, "Clearance_eDrug3D", "mL/min/kg")
        research_log += f"- Excretion (Half-life): {half_life}\n"
        research_log += f"- Excretion (Clearance): {clearance}\n"

        # Clinical Toxicity
        clinical_toxicity = ADMET_pred(smiles, "ClinTox", "%")
        research_log += f"- Clinical Toxicity: {clinical_toxicity}\n"

        research_log += "-------------------------------------\n"

    return research_log


# Binding Affinity prediction function with model_type validation
def predict_binding_affinity_protein_1d_sequence(smiles_list, amino_acid_sequence, affinity_model_type="MPNN-CNN"):
    try:
        from DeepPurpose import DTI, utils
    except ImportError:
        return (
            "ERROR: DeepPurpose is not installed in this environment, so binding affinity cannot be "
            "predicted. Install it where the agent runs (e.g. 'pip install DeepPurpose') and retry."
        )

    # Define available model types for Binding Affinity
    available_affinity_model_types = [
        "CNN-CNN",
        "MPNN-CNN",
        "Morgan-CNN",
        "Morgan-AAC",
        "Daylight-AAC",
    ]

    # Check if the provided affinity model type is valid
    if affinity_model_type not in available_affinity_model_types:
        return f"Error: Invalid affinity model type '{affinity_model_type}'. Available options are: {', '.join(available_affinity_model_types)}."

    # Load the pre-trained affinity model
    model_DTI = DTI.model_pretrained(model=affinity_model_type.replace("-", "_") + "_BindingDB")

    # Initialize research log string
    research_log = "Research Log for Binding Affinity Predictions:\n"
    research_log += "-------------------------------------\n"

    # Process each SMILES string in the list
    for smiles in smiles_list:
        research_log += f"\nCompound SMILES: {smiles}\n"
        research_log += f"Amino Acid Sequence: {amino_acid_sequence}\n"

        # Predict binding affinity
        X_pred = utils.data_process(
            X_drug=[smiles],
            X_target=[amino_acid_sequence],
            y=[0],
            drug_encoding=affinity_model_type.split("-")[0],
            target_encoding=affinity_model_type.split("-")[1],
            split_method="no_split",
        )
        y_pred = model_DTI.predict(X_pred)[0]
        y_pred_nM = 10 ** (-y_pred) / 1e-9

        research_log += f"Predicted Binding Affinity: {y_pred_nM:.2f} nM\n"
        research_log += "-------------------------------------\n"

    return research_log


def analyze_accelerated_stability_of_pharmaceutical_formulations(formulations, storage_conditions, time_points):
    """Project the chemical stability of pharmaceutical formulations under accelerated storage conditions.

    This is a first-order kinetic PROJECTION from each formulation's own degradation rate, scaled to
    each storage temperature with the Q10 rule of thumb -- not a stability study result.

    Parameters
    ----------
    formulations : list of dict
        List of formulation dictionaries, each containing:
        - 'name': str, name of the formulation
        - 'degradation_rate_per_day': float, first-order degradation rate constant at 25 °C (1/day),
          from the formulation's own stability data (required)
        - 'q10': float, optional, rate increase per +10 °C (default 2, the rule of thumb)
        - 'active_ingredient', 'concentration', 'excipients': optional, recorded only

    storage_conditions : list of dict
        List of storage condition dictionaries, each containing:
        - 'temperature': float, temperature in °C
        - 'humidity': float, relative humidity in percentage (optional; recorded, not modelled)
        - 'description': str, description of storage condition (e.g., "Room Temperature", "Accelerated")

    time_points : list of int
        List of time points in days to evaluate stability

    Returns
    -------
    str
        Research log summarizing the projection, or an error naming the missing rates

    """
    # Every formulation used to get the same fixed k=0.001/day (scaled only by temperature and an
    # invented humidity factor), plus made-up physical-stability and particle-size curves; nothing read
    # the formulation, so all formulations got identical numbers and the strict '>' named the first one
    # 'Most stable' under a 'KEY FINDINGS' heading. The projection now uses each formulation's own rate
    # and says it is a projection (hunt 2026-09-30, uT2-pharmacology-11).
    missing = []
    for formulation in formulations:
        try:
            rate = float(formulation["degradation_rate_per_day"])
            if rate < 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            missing.append(str(formulation.get("name", "<unnamed>")))
    if missing:
        return (
            "Error: this tool projects stability from each formulation's own first-order degradation rate "
            f"and does not invent one. Add 'degradation_rate_per_day' (k at 25 °C, 1/day, >= 0) to: {', '.join(missing)}. "
            "Optionally add 'q10' (default 2)."
        )

    # Create output directory if it doesn't exist
    output_dir = os.path.abspath("stability_test_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize results storage
    all_results = []

    # Process each formulation under each storage condition
    for formulation in formulations:
        k25 = float(formulation["degradation_rate_per_day"])
        q10 = float(formulation.get("q10", 2.0))
        for condition in storage_conditions:
            # Initialize stability parameters
            results = []

            # Temperature scaling of the formulation's own rate (Q10 rule of thumb)
            temp_c = condition["temperature"]
            k = k25 * q10 ** ((temp_c - 25) / 10)

            # Calculate stability parameters at each time point
            initial_content = 100.0  # Starting at 100%

            for time in time_points:
                # Chemical stability (% of initial content), first-order degradation
                chemical_stability = initial_content * np.exp(-k * time)

                results.append(
                    {
                        "Formulation": formulation["name"],
                        "Storage_Condition": condition["description"],
                        "Temperature_C": temp_c,
                        "Humidity_RH": condition.get("humidity", "N/A"),
                        "Time_Days": time,
                        "Rate_Constant_per_Day": round(k, 6),
                        "Projected_Chemical_Stability_Percent": round(chemical_stability, 2),
                    }
                )

            all_results.extend(results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results to CSV
    csv_filename = os.path.join(output_dir, f"stability_results_{timestamp}.csv")
    results_df.to_csv(csv_filename, index=False)

    # Generate research log
    log = "Accelerated Stability PROJECTION for Pharmaceutical Formulations (first-order kinetics; not measured data)\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    log += "1. PROJECTION PARAMETERS\n"
    log += f"   - Number of formulations: {len(formulations)}\n"
    log += f"   - Formulations: {', '.join([f['name'] for f in formulations])}\n"
    log += (
        "   - Storage conditions: "
        + ", ".join(
            [
                f"{c['description']} ({c['temperature']}°C" + (f"/{c['humidity']}% RH" if "humidity" in c else "") + ")"
                for c in storage_conditions
            ]
        )
        + "\n"
    )
    log += f"   - Time points projected (days): {', '.join(map(str, time_points))}\n\n"

    log += "2. METHOD\n"
    log += "   - Content projected as 100% x exp(-k t), with k = each formulation's degradation_rate_per_day at 25 °C\n"
    log += "   - k scaled to each storage temperature by the Q10 rule of thumb (k x Q10^((T - 25)/10))\n"
    log += "   - Humidity, physical stability and particle size are not modelled\n\n"

    log += "3. PROJECTED CONTENT AT THE LAST TIME POINT\n"

    # Summarize stability at final time point for each formulation/condition
    final_time = max(time_points)
    final_results = results_df[results_df["Time_Days"] == final_time]

    for formulation in formulations:
        form_results = final_results[final_results["Formulation"] == formulation["name"]]
        log += f"   {formulation['name']}:\n"

        for _, row in form_results.iterrows():
            condition = row["Storage_Condition"]
            chem_stab = row["Projected_Chemical_Stability_Percent"]

            stability_assessment = "Stable"
            if chem_stab < 90:
                stability_assessment = "Potentially unstable"
            if chem_stab < 85:
                stability_assessment = "Unstable"

            log += f"     - {condition}: projected content {chem_stab}% - {stability_assessment}\n"
        log += "\n"

    log += "4. CONCLUSION (from the projection)\n"

    # Identify most stable formulation; a tie is reported as a tie, not as the first one listed
    averages = {
        formulation["name"]: final_results[final_results["Formulation"] == formulation["name"]][
            "Projected_Chemical_Stability_Percent"
        ].mean()
        for formulation in formulations
    }
    best_stability = max(averages.values())
    best = [name for name, value in averages.items() if np.isclose(value, best_stability)]
    if len(best) == len(averages) and len(best) > 1:
        log += f"   - No difference between formulations (avg. projected content: {best_stability:.2f}%)\n"
    else:
        log += f"   - Most stable formulation: {', '.join(best)} (avg. projected content: {best_stability:.2f}%)\n"
    log += f"   - Detailed results saved to: {csv_filename}\n"

    return log


def run_3d_chondrogenic_aggregate_assay(
    chondrocyte_cells, test_compounds, culture_duration_days=21, measurement_intervals=7
):
    """Generates a detailed protocol for performing a 3D chondrogenic aggregate culture assay to evaluate compounds' effects on chondrogenesis.

    Parameters
    ----------
    chondrocyte_cells : dict
        Dictionary with cell information including 'source', 'passage_number', and 'cell_density'
    test_compounds : list of dict
        List of compounds to test, each with 'name', 'concentration', and 'vehicle' keys
    culture_duration_days : int
        Total duration of the culture period in days (default: 21)
    measurement_intervals : int
        Interval in days between measurements (default: 7)

    Returns
    -------
    str
        Detailed protocol document for the 3D chondrogenic aggregate culture assay

    """
    from datetime import datetime

    # Create experiment ID
    experiment_id = f"CHOND3D_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create time points for measurements
    timepoints = list(range(0, culture_duration_days + 1, measurement_intervals))
    if timepoints[-1] != culture_duration_days:
        timepoints.append(culture_duration_days)

    # Generate the protocol document
    protocol = f"# 3D Chondrogenic Aggregate Culture Assay Protocol - {experiment_id}\n\n"

    protocol += "## 1. Materials and Reagents\n\n"
    protocol += "- Chondrocyte cells\n"
    protocol += "- Chondrogenic differentiation medium\n"
    protocol += "- Transforming growth factor-β3 (TGF-β3)\n"
    protocol += "- Dexamethasone\n"
    protocol += "- Ascorbate-2-phosphate\n"
    protocol += "- 96-well V-bottom plates\n"
    protocol += "- Gaussia luciferase reporter assay kit\n"
    protocol += "- Luminometer\n"
    protocol += "- Test compounds with respective vehicles\n"
    protocol += "- Centrifuge\n"
    protocol += "- CO2 incubator\n"
    protocol += "- Sterile pipettes and tips\n\n"

    protocol += "## 2. Experimental Information\n\n"
    protocol += "### Cell Information:\n"
    protocol += f"- Cell source: {chondrocyte_cells['source']}\n"
    protocol += f"- Passage number: {chondrocyte_cells['passage_number']}\n"
    protocol += f"- Cell density: {chondrocyte_cells['cell_density']} cells/mL\n\n"

    protocol += "### Experimental Design:\n"
    protocol += f"- Culture duration: {culture_duration_days} days\n"
    protocol += f"- Measurement timepoints: {', '.join(map(str, timepoints))} days\n\n"

    protocol += "### Test Compounds:\n"
    for i, compound in enumerate(test_compounds):
        protocol += f"- Compound {i + 1}: {compound['name']} at {compound['concentration']} in {compound['vehicle']}\n"
    protocol += "- Control: Vehicle only\n\n"

    protocol += "## 3. Detailed Procedure\n\n"
    protocol += "### Day 0: Setup\n\n"
    protocol += "1. Prepare chondrogenic differentiation medium containing:\n"
    protocol += "   - High-glucose DMEM\n"
    protocol += "   - 10 ng/mL TGF-β3\n"
    protocol += "   - 100 nM Dexamethasone\n"
    protocol += "   - 50 μg/mL Ascorbate-2-phosphate\n"
    protocol += "   - 1% ITS+ premix (insulin, transferrin, selenium)\n"
    protocol += "   - 1 mM Sodium pyruvate\n"
    protocol += "   - 100 U/mL Penicillin/Streptomycin\n\n"

    protocol += "2. Harvest and count chondrocyte cells\n\n"

    protocol += "3. Prepare cell suspension at the specified density:\n"
    protocol += f"   - {chondrocyte_cells['cell_density']} cells/mL\n\n"

    protocol += "4. Form 3D cell aggregates:\n"
    protocol += "   - Aliquot 2.5×10^5 cells per well in 96-well V-bottom plates\n"
    protocol += "   - Centrifuge plates at 500g for 5 minutes to pellet cells\n\n"

    protocol += "5. Add test compounds to respective wells:\n"
    for _, compound in enumerate(test_compounds):
        protocol += f"   - Add {compound['name']} at {compound['concentration']} in {compound['vehicle']}\n"
    protocol += "   - Add vehicle only to control wells\n\n"

    protocol += "6. Incubate the plates at 37°C, 5% CO2\n\n"

    protocol += "### Day 1 to Day " + str(culture_duration_days) + ":\n\n"
    protocol += "1. Change medium every 2-3 days:\n"
    protocol += "   - Carefully remove 50% of the medium without disturbing the aggregates\n"
    protocol += "   - Replace with fresh medium containing test compounds at the same concentrations\n\n"

    protocol += f"2. At days {', '.join(map(str, timepoints))}, collect samples for analysis:\n"
    protocol += (
        "   - Take medium samples for Gaussia luciferase activity measurement (if using COL2A1-GLuc reporter cells)\n"
    )
    protocol += "   - Fix aggregates in 4% paraformaldehyde for histological analysis\n\n"

    return protocol


def grade_adverse_events_using_vcog_ctcae(clinical_data_file):
    """Grade and monitor adverse events in animal studies using the VCOG-CTCAE standard.

    Parameters
    ----------
    clinical_data_file : str
        Path to a CSV file containing clinical evaluation data with columns:
        subject_id, time_point, symptom, severity, measurement (optional)

    Returns
    -------
    str
        A research log summarizing the adverse event grading process and results.
        The graded events are saved to 'vcog_ctcae_graded_events.csv' (a numbered sibling such as
        'vcog_ctcae_graded_events_2.csv' when that file already exists; the log names the path).

    """
    import json
    from datetime import datetime

    import pandas as pd

    # Initialize the research log
    log = "# Adverse Event Grading using VCOG-CTCAE v1.1\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Step 1: Load the clinical data
    log += "## Step 1: Loading clinical evaluation data\n"
    try:
        data = pd.read_csv(clinical_data_file)
        log += f"Successfully loaded data from {clinical_data_file}\n"
        log += f"Total records: {len(data)}\n"
        log += f"Columns found: {', '.join(data.columns)}\n\n"
    except Exception as e:
        log += f"Error loading data: {str(e)}\n"
        return log

    # Step 2: Define VCOG-CTCAE grading criteria
    log += "## Step 2: Applying VCOG-CTCAE grading criteria\n"

    # Comprehensive VCOG-CTCAE grading criteria based on VCOG-CTCAE v1.1
    vcog_criteria = {
        # Hematologic
        "neutropenia": {
            "description": "Neutrophil count decrease",
            "unit": "cells/µL",
            "grades": {
                0: {"criteria": "≥ 1500", "range": [1500, float("inf")]},
                1: {"criteria": "1000 - <1500", "range": [1000, 1500]},
                2: {"criteria": "500 - <1000", "range": [500, 1000]},
                3: {"criteria": "100 - <500", "range": [100, 500]},
                4: {"criteria": "<100", "range": [0, 100]},
                5: {"criteria": "Death due to neutropenic sepsis", "range": None},
            },
        },
        "anemia": {
            "description": "Hemoglobin decrease",
            "unit": "g/dL",
            "grades": {
                0: {"criteria": "Within reference range", "range": [10, float("inf")]},
                1: {"criteria": "Mild; clinical signs not present", "range": [8, 10]},
                2: {"criteria": "Moderate; clinical signs present", "range": [6.5, 8]},
                3: {"criteria": "Severe; transfusion indicated", "range": [5, 6.5]},
                4: {"criteria": "Life-threatening", "range": [0, 5]},
                5: {"criteria": "Death", "range": None},
            },
        },
        "thrombocytopenia": {
            "description": "Platelet count decrease",
            "unit": "cells/µL",
            "grades": {
                0: {"criteria": "≥ 100,000", "range": [100000, float("inf")]},
                1: {"criteria": "50,000 - <100,000", "range": [50000, 100000]},
                2: {"criteria": "25,000 - <50,000", "range": [25000, 50000]},
                3: {"criteria": "10,000 - <25,000", "range": [10000, 25000]},
                4: {"criteria": "<10,000 or spontaneous bleeding", "range": [0, 10000]},
                5: {"criteria": "Death", "range": None},
            },
        },
        # Gastrointestinal
        # Every "range" is half-open, [min, max), which is how apply_vcog_grade tests it. The two count
        # criteria were written as closed integer ranges ([1, 2], [3, 5]), so 0, 2 and 5 vomiting
        # episodes and 3 or 6 stools matched no grade and fell back to 'Grade 1: Default grade'
        # (hunt 2026-09-30, uT2-pharmacology-9).
        "vomiting": {
            "description": "Vomiting frequency",
            "unit": "episodes per 24h period",
            "grades": {
                0: {"criteria": "None", "range": [0, 1]},
                1: {
                    "criteria": "1-2 episodes in 24h; medical intervention not indicated",
                    "range": [1, 3],
                },
                2: {"criteria": "3-5 episodes in 24h; ≤3 days", "range": [3, 6]},
                3: {
                    "criteria": ">5 episodes in 24h; >3 days; hospitalization indicated",
                    "range": [6, float("inf")],
                },
                4: {"criteria": "Life-threatening consequences", "range": None},
                5: {"criteria": "Death", "range": None},
            },
        },
        "diarrhea": {
            "description": "Diarrhea frequency",
            "unit": "episodes per 24h period",
            "grades": {
                0: {"criteria": "None", "range": [0, 1]},
                1: {
                    "criteria": "Increase of <4 stools per day over baseline",
                    "range": [1, 4],
                },
                2: {
                    "criteria": "Increase of 4-6 stools per day over baseline",
                    "range": [4, 7],
                },
                3: {
                    "criteria": "Increase of ≥7 stools per day; hospitalization indicated",
                    "range": [7, float("inf")],
                },
                4: {"criteria": "Life-threatening consequences", "range": None},
                5: {"criteria": "Death", "range": None},
            },
        },
        "anorexia": {
            "description": "Appetite/food intake decrease",
            "unit": "percent of normal intake",
            "grades": {
                0: {"criteria": "Normal", "range": [100, float("inf")]},
                1: {"criteria": "Decreased appetite, but eating", "range": [75, 100]},
                2: {"criteria": "Decreased intake <3 days", "range": [50, 75]},
                3: {"criteria": "Decreased intake ≥3 days", "range": [25, 50]},
                4: {
                    "criteria": "Life-threatening consequences; urgent intervention indicated",
                    "range": [0, 25],
                },
                5: {"criteria": "Death", "range": None},
            },
        },
        # Hepatic
        "alt_increase": {
            "description": "Alanine aminotransferase increased",
            "unit": "x ULN (upper limit of normal)",
            "grades": {
                0: {"criteria": "≤ ULN", "range": [0, 1]},
                1: {"criteria": ">ULN - 2.5xULN", "range": [1, 2.5]},
                2: {"criteria": ">2.5 - 5.0xULN", "range": [2.5, 5]},
                3: {"criteria": ">5.0 - 20.0xULN", "range": [5, 20]},
                4: {"criteria": ">20.0xULN", "range": [20, float("inf")]},
                5: {"criteria": "Death", "range": None},
            },
        },
        # Renal
        "creatinine_increase": {
            "description": "Creatinine increased",
            "unit": "x ULN",
            "grades": {
                0: {"criteria": "≤ ULN", "range": [0, 1]},
                1: {"criteria": ">ULN - 1.5xULN", "range": [1, 1.5]},
                2: {"criteria": ">1.5 - 3.0xULN", "range": [1.5, 3]},
                3: {"criteria": ">3.0 - 6.0xULN", "range": [3, 6]},
                4: {"criteria": ">6.0xULN", "range": [6, float("inf")]},
                5: {"criteria": "Death", "range": None},
            },
        },
        # Constitutional
        "fever": {
            "description": "Fever",
            "unit": "°C",
            "grades": {
                0: {"criteria": "None", "range": [0, 39]},
                1: {"criteria": "39.0 - 39.5°C", "range": [39, 39.5]},
                2: {"criteria": ">39.5 - 40.0°C", "range": [39.5, 40]},
                3: {"criteria": ">40.0 - 41.0°C", "range": [40, 41]},
                4: {"criteria": ">41.0°C for >24 hrs", "range": [41, float("inf")]},
                5: {"criteria": "Death", "range": None},
            },
        },
        "weight_loss": {
            "description": "Weight loss",
            "unit": "percent of baseline weight",
            "grades": {
                0: {"criteria": "<5%", "range": [0, 5]},
                1: {"criteria": "5% - <10%", "range": [5, 10]},
                2: {"criteria": "10% - <20%", "range": [10, 20]},
                3: {"criteria": "≥20%", "range": [20, float("inf")]},
                4: {"criteria": "Life-threatening", "range": None},
                5: {"criteria": "Death", "range": None},
            },
        },
        # Dermatologic
        "alopecia": {
            "description": "Hair loss",
            "unit": None,
            "grades": {
                0: {"criteria": "None", "range": None},
                1: {"criteria": "Hair loss at injection/treatment site", "range": None},
                2: {"criteria": "Moderate alopecia", "range": None},
                3: {"criteria": "Complete alopecia", "range": None},
                4: {"criteria": "Not applicable", "range": None},
                5: {"criteria": "Not applicable", "range": None},
            },
        },
        # Neurologic
        "neuropathy": {
            "description": "Peripheral neuropathy",
            "unit": None,
            "grades": {
                0: {"criteria": "None", "range": None},
                1: {
                    "criteria": "Asymptomatic; clinically detectable on examination",
                    "range": None,
                },
                2: {
                    "criteria": "Mild symptoms; limiting instrumental ADL",
                    "range": None,
                },
                3: {
                    "criteria": "Severe symptoms; limiting self-care ADL",
                    "range": None,
                },
                4: {"criteria": "Life-threatening consequences", "range": None},
                5: {"criteria": "Death", "range": None},
            },
        },
    }

    def apply_vcog_grade(symptom, severity, measurement=None):
        """Apply VCOG-CTCAE grading criteria to an adverse event.

        Parameters
        ----------
        symptom : str
            The type of adverse event
        severity : str
            The severity description
        measurement : float or None
            Quantitative measurement related to the symptom, if available

        Returns
        -------
        int
            The VCOG-CTCAE grade (0-5)
        str
            Description of the grading rationale

        """
        # Standard severity-based grading if no specific criteria exist
        grade_map = {
            "none": 0,
            "mild": 1,
            "moderate": 2,
            "severe": 3,
            "life-threatening": 4,
            "death": 5,
        }

        # A blank or 'n/a' cell is read as NaN, and NaN.lower() crashed the whole grading run
        # (hunt 2026-09-30, uT2-pharmacology-9).
        symptom_lower = str(symptom).strip().lower() if pd.notna(symptom) else ""
        severity = str(severity).strip() if pd.notna(severity) else ""

        # Check if the symptom has specific VCOG-CTCAE criteria
        if symptom_lower in vcog_criteria:
            criteria = vcog_criteria[symptom_lower]

            # If measurement is provided and criteria has numeric ranges
            if measurement is not None:
                try:
                    measurement_value = float(measurement)

                    # Find the appropriate grade based on the measurement ranges
                    for grade, grade_info in criteria["grades"].items():
                        if grade_info["range"] is not None:
                            min_val, max_val = grade_info["range"]
                            if min_val <= measurement_value < max_val:
                                return (
                                    grade,
                                    f"Grade {grade}: {criteria['description']} - {criteria['grades'][grade]['criteria']}",
                                )
                except (ValueError, TypeError):
                    # If measurement can't be converted to float, fall back to severity-based grading
                    pass

            # If there's a reported severity with no valid measurement
            if severity.lower() in grade_map:
                # Check if the grade exists in the criteria
                severity_grade = grade_map[severity.lower()]
                if severity_grade in criteria["grades"]:
                    return (
                        severity_grade,
                        f"Grade {severity_grade}: {criteria['description']} - {criteria['grades'][severity_grade]['criteria']}",
                    )

        # Default to using the severity mapping if no specific criteria match
        if severity.lower() in grade_map:
            return grade_map[severity.lower()], f"Grade {grade_map[severity.lower()]}: Based on reported severity"

        # Default grade if no specific criteria match
        return 1, "Grade 1: Default grade (specific criteria not found)"

    # Step 3: Apply grading to each record
    log += "Applying VCOG-CTCAE v1.1 grading criteria to each adverse event...\n"

    # Create new columns for the grade and rationale
    grading_results = data.apply(
        lambda row: apply_vcog_grade(
            row["symptom"],
            row["severity"],
            row["measurement"] if "measurement" in data.columns else None,
        ),
        axis=1,
    )

    # Split the returned tuples into separate columns
    data["vcog_grade"] = [result[0] for result in grading_results]
    data["grading_rationale"] = [result[1] for result in grading_results]

    # Step 4: Analyze patterns across time points (if available)
    if "time_point" in data.columns:
        log += "\n## Step 3: Analyzing adverse event patterns across time points\n"

        # Group by subject and symptom to track progression
        progression_analysis = data.pivot_table(
            index=["subject_id", "symptom"],
            columns="time_point",
            values="vcog_grade",
            aggfunc="max",
        ).reset_index()

        # Calculate if grade is increasing, decreasing, or stable for each subject-symptom pair
        trend_counts = {"increasing": 0, "decreasing": 0, "stable": 0, "fluctuating": 0}

        numeric_columns = [col for col in progression_analysis.columns if col not in ["subject_id", "symptom"]]

        if len(numeric_columns) >= 2:
            # Sort columns to ensure chronological order
            numeric_columns.sort()

            for _, row in progression_analysis.iterrows():
                values = [row[col] for col in numeric_columns if not pd.isna(row[col])]
                if len(values) >= 2:
                    if all(values[i] < values[i + 1] for i in range(len(values) - 1)):
                        trend_counts["increasing"] += 1
                    elif all(values[i] > values[i + 1] for i in range(len(values) - 1)):
                        trend_counts["decreasing"] += 1
                    elif all(values[i] == values[i + 1] for i in range(len(values) - 1)):
                        trend_counts["stable"] += 1
                    else:
                        trend_counts["fluctuating"] += 1

            log += "Adverse event progression patterns:\n"
            for trend, count in trend_counts.items():
                log += f"- {trend.capitalize()}: {count} subject-symptom pairs\n"

        # Save progression analysis
        # Absolute, so the path this log reports back is one the caller can actually find.
        progression_file = _fresh_output_path("vcog_ctcae_progression_analysis.csv")
        progression_analysis.to_csv(progression_file)
        log += f"\nDetailed progression analysis saved to: {progression_file}\n"

    # Step 5: Summarize the grading results
    log += "\n## Step 4: Summarizing adverse event grades\n"

    # Count events by grade
    grade_counts = data["vcog_grade"].value_counts().sort_index()
    log += "Grade distribution:\n"
    for grade, count in grade_counts.items():
        log += f"- Grade {grade}: {count} events\n"

    # Summarize by symptom type
    symptom_summary = data.groupby("symptom")["vcog_grade"].agg(["max", "mean", "count"])
    log += "\nSymptom severity summary:\n"
    for symptom, stats in symptom_summary.iterrows():
        log += f"- {symptom}: max grade = {stats['max']}, avg grade = {stats['mean']:.2f}, count = {stats['count']}\n"

    # Summarize by subject
    subject_summary = data.groupby("subject_id")["vcog_grade"].agg(["max", "mean", "count"])
    log += f"\nSubjects with adverse events: {len(subject_summary)}\n"
    log += f"Subjects with Grade 3+ events: {len(subject_summary[subject_summary['max'] >= 3])}\n"

    # Create a summary of most severe events
    most_severe = data.sort_values("vcog_grade", ascending=False).head(10)
    log += "\nTop 10 most severe adverse events:\n"
    for i, (_, event) in enumerate(most_severe.iterrows(), 1):
        log += f"{i}. Subject {event['subject_id']}: {event['symptom']} (Grade {event['vcog_grade']})\n"

    # Step 6: Save detailed results to file. Absolute, so the paths this log reports back are ones
    # the caller can actually find.
    output_file = _fresh_output_path("vcog_ctcae_graded_events.csv")
    data.to_csv(output_file, index=False)
    log += "\n## Step 5: Results saved\n"
    log += f"Detailed graded events saved to: {output_file}\n"

    # Save the VCOG criteria as a reference
    criteria_file = _fresh_output_path("vcog_ctcae_criteria_reference.json")
    with open(criteria_file, "w") as f:
        json.dump(vcog_criteria, f, indent=2)
    log += f"VCOG-CTCAE criteria reference saved to: {criteria_file}\n"

    return log


def analyze_radiolabeled_antibody_biodistribution(time_points, tissue_data):
    """Analyze biodistribution and pharmacokinetic profile of radiolabeled antibodies.

    Parameters
    ----------
    time_points : list or numpy.ndarray
        Time points (hours) at which measurements were taken
    tissue_data : dict
        Dictionary where keys are tissue names and values are lists/arrays of %IA/g
        measurements corresponding to time_points. Must include 'tumor' as one of the keys.

    Returns
    -------
    str
        Research log summarizing the biodistribution analysis, pharmacokinetic parameters,
        and tumor-to-normal tissue ratios

    """
    import json
    import os

    import numpy as np
    from scipy.optimize import curve_fit

    # Validate inputs
    if "tumor" not in tissue_data:
        return "Error: Tumor data must be provided in tissue_data dictionary"

    # Define bi-exponential model for pharmacokinetic analysis
    # C(t) = A*exp(-alpha*t) + B*exp(-beta*t)
    def bi_exp_model(t, A, alpha, B, beta):
        return A * np.exp(-alpha * t) + B * np.exp(-beta * t)

    # Initialize results dictionary
    results = {
        "tissues_analyzed": list(tissue_data.keys()),
        "pk_parameters": {},
        "tumor_to_normal_ratios": {},
        "auc_values": {},
    }

    # Analyze each tissue. Every curve used to be fitted with a bi-exponential DECAY and any fit was
    # reported, so a tumour uptake curve that rises and then falls got an 'elimination half-life' of
    # 10^13 hours. A curve with an uptake phase is reported by its observed peak, and a decay fit only
    # when it describes the data (hunt 2026-09-30, uT2-pharmacology skeptic note 1).
    # A peak within 10% of the first measurement is read as noise on a decay, not as an uptake phase: blood
    # [40, 40.5, 15, 8, 3] was refused a fit for its 1% early bump (hunt 2026-09-30, uT2-pharmacology
    # skeptic note 1, review); the R-squared and bound checks below still reject a decay that does not
    # describe the curve.
    t_obs = np.asarray(time_points, dtype=float)
    for tissue, measurements in tissue_data.items():
        y_obs = np.asarray(measurements, dtype=float)
        peak = int(np.argmax(y_obs)) if y_obs.size else 0
        first = int(np.argmin(t_obs)) if t_obs.size else 0
        if y_obs.size and t_obs[peak] > t_obs[first] and y_obs[peak] > 1.1 * max(y_obs[first], 0.0):
            results["pk_parameters"][tissue] = (
                f"Not fitted: uptake phase (observed peak {y_obs[peak]:g} %IA/g at {t_obs[peak]:g} h), "
                "which a bi-exponential decay cannot describe"
            )
            continue
        try:
            # Fit bi-exponential model
            lower, upper = [0, 0, 0, 0], [100, 5, 100, 1]
            params, _ = curve_fit(
                bi_exp_model,
                time_points,
                measurements,
                p0=[50, 0.1, 50, 0.01],  # Initial parameter guess
                bounds=(lower, upper),  # Parameter bounds
            )

            A, alpha, B, beta = params
            fitted = bi_exp_model(t_obs, *params)
            ss_tot = float(np.sum((y_obs - y_obs.mean()) ** 2))
            r_squared = 1 - float(np.sum((y_obs - fitted) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
            if not r_squared >= 0.9:
                results["pk_parameters"][tissue] = (
                    f"Fit rejected: the bi-exponential decay does not describe these data (R-squared {r_squared:.3f} < 0.9)"
                )
                continue
            # A parameter pinned at a bound leaves a phase undetermined: blood [40, 38, 20, 12, 10] fitted with
            # beta at its bound of 0 and was reported with an 'elimination half-life' of 5e18 hours at
            # R² > 0.9 (hunt 2026-09-30, uT2-pharmacology skeptic note 1, review).
            pinned = [
                name
                for name, value, lo, hi in zip(("A", "alpha", "B", "beta"), params, lower, upper, strict=True)
                if np.isclose(value, lo, rtol=1e-6, atol=1e-9) or np.isclose(value, hi, rtol=1e-6, atol=1e-9)
            ]
            if pinned:
                results["pk_parameters"][tissue] = (
                    f"Fit rejected: {', '.join(pinned)} ran to its bound, so these data do not determine both "
                    "phases' half-lives"
                )
                continue

            # Calculate pharmacokinetic parameters
            # Distribution half-life (fast component)
            t_half_dist = np.log(2) / alpha

            # Elimination half-life (slow component)
            t_half_elim = np.log(2) / beta

            # Area under the curve (AUC)
            auc = A / alpha + B / beta

            # Mean residence time (MRT)
            mrt = (A / (alpha**2) + B / (beta**2)) / auc

            # Clearance (for blood/plasma only - conceptual)
            clearance = 1 / auc if tissue.lower() in ["blood", "plasma"] else None

            # Store results
            results["pk_parameters"][tissue] = {
                "A": float(A),
                "alpha": float(alpha),
                "B": float(B),
                "beta": float(beta),
                "r_squared": float(r_squared),
                "distribution_half_life_h": float(t_half_dist),
                "elimination_half_life_h": float(t_half_elim),
                "mean_residence_time_h": float(mrt),
            }

            if clearance:
                results["pk_parameters"][tissue]["clearance"] = float(clearance)

            # Calculate AUC
            results["auc_values"][tissue] = float(auc)

        except Exception as e:
            results["pk_parameters"][tissue] = f"Fitting failed: {str(e)}"

    # Calculate tumor-to-normal tissue ratios at each time point
    for tissue in tissue_data:
        if tissue != "tumor":
            ratios = [
                t / n if n > 0 else float("inf")
                for t, n in zip(tissue_data["tumor"], tissue_data[tissue], strict=False)
            ]
            results["tumor_to_normal_ratios"][tissue] = {
                "values": [float(r) for r in ratios],
                "max_ratio": float(max(ratios)),
                "max_ratio_time_point": float(time_points[np.argmax(ratios)]),
            }

    # Save results to JSON file
    filename = _fresh_output_path("biodistribution_pk_results.json")
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    # Generate research log
    log = "# Biodistribution and Pharmacokinetic Analysis of Radiolabeled Antibody\n\n"
    log += "## Analysis Summary\n"
    log += f"- Analyzed biodistribution data across {len(tissue_data)} tissues\n"
    log += f"- Time points analyzed: {time_points} hours\n"
    log += "- Performed bi-exponential pharmacokinetic modeling\n\n"

    log += "## Key Pharmacokinetic Parameters\n"
    for tissue, params in results["pk_parameters"].items():
        if isinstance(params, dict):
            log += f"\n### {tissue.capitalize()}\n"
            log += f"- Distribution half-life: {params['distribution_half_life_h']:.2f} hours\n"
            log += f"- Elimination half-life: {params['elimination_half_life_h']:.2f} hours\n"
            log += f"- Mean residence time: {params['mean_residence_time_h']:.2f} hours\n"
            if "clearance" in params:
                log += f"- Clearance: {params['clearance']:.4f} units\n"
            log += f"- Fit R-squared: {params['r_squared']:.4f}\n"
        else:
            log += f"\n### {tissue.capitalize()}\n- {params}\n"

    log += "\n## Tumor-to-Normal Tissue Ratios\n"
    for tissue, ratio_data in results["tumor_to_normal_ratios"].items():
        log += f"- {tissue.capitalize()}: Max ratio {ratio_data['max_ratio']:.2f} at {ratio_data['max_ratio_time_point']:.1f} hours\n"

    log += "\n## Detailed Results\n"
    log += f"Complete analysis results saved to: {os.path.abspath(filename)}\n"

    return log


def estimate_alpha_particle_radiotherapy_dosimetry(
    biodistribution_data, radiation_parameters, output_file="dosimetry_results.csv"
):
    """Estimate radiation absorbed doses to tumor and normal organs for alpha-particle radiotherapeutics.

    This function implements the Medical Internal Radiation Dose (MIRD) schema to calculate
    absorbed doses based on biodistribution data from healthy mice and radiation transport parameters.

    Parameters
    ----------
    biodistribution_data : dict
        Dictionary containing organ/tissue names as keys and a list of time-activity measurements as values.
        Each measurement should be a tuple of (time_hours, percent_injected_activity).
        Must include entries for all relevant organs including 'tumor'.

    radiation_parameters : dict
        Dictionary containing radiation parameters for the alpha-emitting radionuclide:
        - 'radionuclide': str - Name of the radionuclide (e.g., 'Ac-225')
        - 'half_life_hours': float - Physical half-life in hours
        - 'energy_per_decay_MeV': float - Energy released per decay in MeV
        - 'radiation_weighting_factor': float - Radiation weighting factor for alpha particles
        - 'S_factors': dict - S-factors (Gy/Bq-s) for each source-target organ pair, keyed as
          {('source', 'target'): S}, {'source': {'target': S}} or {'source->target': S}

    output_file : str, optional
        Filename to save the dosimetry results (default: "dosimetry_results.csv"; with the default, a
        later call writes dosimetry_results_2.csv, ... instead of overwriting an earlier result)

    Returns
    -------
    str
        Research log summarizing the dosimetry estimation process and results

    """
    import csv
    from datetime import datetime

    import numpy as np
    from scipy.integrate import trapezoid

    # Initialize research log
    log = f"Alpha-Particle Radiotherapy Dosimetry Estimation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += f"Radionuclide: {radiation_parameters['radionuclide']}\n"
    log += f"Half-life: {radiation_parameters['half_life_hours']} hours\n\n"

    # Step 1: Calculate time-integrated activity for each organ
    log += "Step 1: Calculating time-integrated activity for each organ\n"
    time_integrated_activity = {}

    for organ, measurements in biodistribution_data.items():
        times = [m[0] for m in measurements]
        activities = [m[1] for m in measurements]

        # Apply physical decay correction
        decay_constant = np.log(2) / radiation_parameters["half_life_hours"]
        decay_corrected_activities = [a * np.exp(-decay_constant * t) for a, t in zip(activities, times, strict=False)]

        # Calculate time-integrated activity using trapezoidal integration
        cumulated_activity = trapezoid(decay_corrected_activities, times)
        time_integrated_activity[organ] = cumulated_activity

        log += f"  - {organ}: {cumulated_activity:.4f} %IA-h\n"

    # Step 2: Calculate absorbed dose using MIRD schema
    log += "\nStep 2: Calculating absorbed doses using MIRD schema\n"

    # %IA x h -> Bq x s per MBq injected: 0.01 (fraction of IA) x 1e6 (Bq per MBq) x 3600 (s per h).
    # It used to stop at the 0.01, so with S in Gy/(Bq.s) every dose was 3.6e9-fold too small; the
    # radiation weighting factor was then folded in, which makes an equivalent dose (Sv), and the
    # result was still labelled Gy/MBq. Absorbed dose (Gy/MBq) and equivalent dose (Sv/MBq) are now
    # reported separately (hunt 2026-09-30, uT2-pharmacology-10).
    conversion_factor = 0.01 * 1e6 * 3600

    # S_factors may be keyed by (source, target) tuples, nested {source: {target: S}} dicts (the shape
    # JSON gives), or 'source->target' strings. Only tuple keys were read, so the other two shapes made
    # every organ 0.0000 Gy/MBq and that was saved as the result.
    s_factors = {}
    for key, value in (radiation_parameters.get("S_factors") or {}).items():
        if isinstance(value, dict):
            for target, s_value in value.items():
                s_factors[(str(key), str(target))] = float(s_value)
        elif isinstance(key, tuple) and len(key) == 2:
            s_factors[(str(key[0]), str(key[1]))] = float(value)
        elif isinstance(key, str) and "->" in key:
            source, target = (part.strip() for part in key.split("->", 1))
            s_factors[(source, target)] = float(value)

    matched_pairs = [
        pair for pair in s_factors if pair[0] in time_integrated_activity and pair[1] in biodistribution_data
    ]
    if not matched_pairs:
        return (
            log + "\nError: no S-factor matched any (source, target) organ pair of biodistribution_data. "
            "Give S_factors in Gy/(Bq.s) as {('liver', 'liver'): S, ...}, {'liver': {'liver': S}} or "
            "{'liver->liver': S}, with organ names exactly as in biodistribution_data "
            f"({', '.join(map(str, biodistribution_data))}).\n"
        )

    # Calculate absorbed dose for each target organ
    absorbed_doses = {}
    equivalent_doses = {}
    w_r = radiation_parameters["radiation_weighting_factor"]

    for target_organ in biodistribution_data:
        absorbed_dose = 0
        contributing = 0

        # Sum contributions from all source organs
        for source_organ, cumulated_activity in time_integrated_activity.items():
            if (source_organ, target_organ) in s_factors:
                s_value = s_factors[(source_organ, target_organ)]
                organ_contribution = cumulated_activity * conversion_factor * s_value
                absorbed_dose += organ_contribution
                contributing += 1

        if not contributing:
            log += f"  - {target_organ}: not computed (no S-factor with this organ as the target)\n"
            continue

        # Store as Gy/MBq; the radiation weighting factor gives the equivalent dose in Sv/MBq
        absorbed_doses[target_organ] = absorbed_dose
        equivalent_doses[target_organ] = absorbed_dose * w_r
        log += f"  - {target_organ}: {absorbed_dose:.4g} Gy/MBq absorbed, {absorbed_dose * w_r:.4g} Sv/MBq equivalent (w_R={w_r})\n"

    # Step 3: Calculate therapeutic index (tumor-to-normal tissue dose ratios)
    log += "\nStep 3: Calculating therapeutic indices (tumor-to-normal tissue ratios)\n"

    tumor_dose = absorbed_doses.get("tumor", 0)
    if tumor_dose > 0:
        for organ, dose in absorbed_doses.items():
            if organ != "tumor" and dose > 0:
                therapeutic_index = tumor_dose / dose
                log += f"  - Tumor-to-{organ} ratio: {therapeutic_index:.2f}\n"

    # Save results to CSV file. With the default name a second call overwrote the file the first log
    # names (hunt 2026-09-30, uT2-pharmacology-23).
    if output_file == "dosimetry_results.csv":
        output_file = _fresh_output_path(output_file)
    output_file = os.path.abspath(output_file)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Organ", "Absorbed Dose (Gy/MBq)", "Equivalent Dose (Sv/MBq)"])
        for organ, dose in absorbed_doses.items():
            writer.writerow([organ, f"{dose:.6g}", f"{equivalent_doses[organ]:.6g}"])

    log += f"\nDosimetry results saved to {output_file}\n"

    return log


def perform_mwas_cyp2c19_metabolizer_status(
    methylation_data_path,
    metabolizer_status_path,
    covariates_path=None,
    pvalue_threshold=0.05,
    output_file="significant_cpg_sites.csv",
):
    """Perform a Methylome-wide Association Study (MWAS) to identify CpG sites significantly associated with CYP2C19 metabolizer status.

    Parameters
    ----------
    methylation_data_path : str
        Path to CSV or TSV file containing DNA methylation beta values.
        Rows should be samples, columns should be CpG sites.
    metabolizer_status_path : str
        Path to CSV or TSV file containing CYP2C19 metabolizer status for each sample.
        Should have a sample ID column and a status column (e.g., poor, intermediate, normal, rapid, ultrarapid).
    covariates_path : str, optional
        Path to CSV or TSV file containing covariates to adjust for in the regression model
        (e.g., age, sex, smoking status).
    pvalue_threshold : float, optional
        P-value threshold for significance after multiple testing correction. Default is 0.05.
    output_file : str, optional
        Filename to save significant CpG sites. Default is "significant_cpg_sites.csv"; with the default,
        a later call writes significant_cpg_sites_2.csv, ... instead of overwriting an earlier result.

    Returns
    -------
    str
        A research log summarizing the MWAS analysis and results.

    """
    import time

    import pandas as pd
    from scipy.stats import linregress
    from statsmodels.formula.api import ols

    start_time = time.time()
    log = ["## Methylome-wide Association Study (MWAS) of CYP2C19 Metabolizer Status"]

    # Load data from files
    log.append("\n### Loading Data")
    try:
        # Load methylation data
        if methylation_data_path.endswith(".csv"):
            methylation_data = pd.read_csv(methylation_data_path, index_col=0)
        elif methylation_data_path.endswith((".tsv", ".txt")):
            methylation_data = pd.read_csv(methylation_data_path, sep="\t", index_col=0)
        else:
            log.append("Error: Unsupported file format for methylation data. Please provide a CSV or TSV file.")
            return "\n".join(log)
        log.append(f"- Successfully loaded methylation data from {methylation_data_path}")

        # Load metabolizer status data
        if metabolizer_status_path.endswith(".csv"):
            metabolizer_status_df = pd.read_csv(metabolizer_status_path, index_col=0)
        elif metabolizer_status_path.endswith((".tsv", ".txt")):
            metabolizer_status_df = pd.read_csv(metabolizer_status_path, sep="\t", index_col=0)
        else:
            log.append("Error: Unsupported file format for metabolizer status. Please provide a CSV or TSV file.")
            return "\n".join(log)
        log.append(f"- Successfully loaded metabolizer status data from {metabolizer_status_path}")

        # Convert DataFrame to Series if necessary
        if metabolizer_status_df.shape[1] == 1:
            metabolizer_status = metabolizer_status_df.iloc[:, 0]
        else:
            log.append("Error: Metabolizer status file should contain a single column with status values.")
            return "\n".join(log)

        # Load covariates if provided
        covariates = None
        if covariates_path is not None:
            if covariates_path.endswith(".csv"):
                covariates = pd.read_csv(covariates_path, index_col=0)
            elif covariates_path.endswith((".tsv", ".txt")):
                covariates = pd.read_csv(covariates_path, sep="\t", index_col=0)
            else:
                log.append("Error: Unsupported file format for covariates. Please provide a CSV or TSV file.")
                return "\n".join(log)
            log.append(f"- Successfully loaded covariates data from {covariates_path}")
    except Exception as e:
        log.append(f"Error loading data: {str(e)}")
        return "\n".join(log)

    # Step 1: Data preprocessing
    log.append("\n### Data Preprocessing")
    log.append(f"- Methylation data shape: {methylation_data.shape} (samples × CpG sites)")
    log.append(f"- Number of samples with metabolizer status: {len(metabolizer_status)}")

    # Ensure sample IDs match between methylation data and metabolizer status
    common_samples = methylation_data.index.intersection(metabolizer_status.index)
    methylation_data = methylation_data.loc[common_samples]
    metabolizer_status = metabolizer_status.loc[common_samples]

    log.append(f"- Number of samples after matching: {len(common_samples)}")

    # Check for covariates
    if covariates is not None:
        log.append(f"- Covariates provided: {', '.join(covariates.columns)}")
        covariates = covariates.loc[common_samples]

    # Step 2: Perform regression for each CpG site
    log.append("\n### Association Analysis")
    log.append(f"- Total CpG sites to analyze: {methylation_data.shape[1]}")

    results = []
    cpg_sites = methylation_data.columns

    # Convert metabolizer status to numeric if it's categorical
    if metabolizer_status.dtype == "object":
        # Create a mapping dictionary for metabolizer status
        # Assuming order: poor < intermediate < normal < rapid < ultrarapid
        status_order = {
            "poor": 1,
            "intermediate": 2,
            "normal": 3,
            "rapid": 4,
            "ultrarapid": 5,
        }
        # CPIC's abbreviations and spellings. The map used to be exact and lower-case only, so 'Poor',
        # 'PM' or 'Normal metabolizer' became NaN, every p-value NaN, and the log said 'No significant
        # CpG sites found' (hunt 2026-09-30, uT2-pharmacology-8).
        aliases = {"pm": "poor", "im": "intermediate", "nm": "normal", "em": "normal", "extensive": "normal"}
        aliases.update({"rm": "rapid", "um": "ultrarapid", "ultra rapid": "ultrarapid", "ultra-rapid": "ultrarapid"})

        def _status_key(value):
            if pd.isna(value):
                return value
            key = re.sub(r"\s+", " ", str(value).strip().lower())
            key = re.sub(r"\s*metaboli[sz]er$", "", key)
            return status_order.get(aliases.get(key, key))

        metabolizer_status_numeric = metabolizer_status.map(_status_key)
        unmapped = sorted(
            {str(v) for v, n in zip(metabolizer_status, metabolizer_status_numeric, strict=False) if pd.isna(n)}
        )
        if unmapped:
            log.append(
                f"Error: metabolizer status labels {unmapped} are not recognised. Use poor, intermediate, "
                "normal, rapid or ultrarapid (or PM/IM/NM/RM/UM), or numeric codes 1-5."
            )
            return "\n".join(log)
        log.append("- Converted metabolizer status to numeric values")
    else:
        metabolizer_status_numeric = metabolizer_status
        if metabolizer_status_numeric.isna().any():
            log.append(f"Error: {int(metabolizer_status_numeric.isna().sum())} samples have no metabolizer status.")
            return "\n".join(log)

    # Perform regression for each CpG site
    for cpg in cpg_sites:
        methylation_values = methylation_data[cpg]

        # Basic model without covariates
        if covariates is None:
            model = linregress(metabolizer_status_numeric, methylation_values)
            pvalue = model.pvalue
            coefficient = model.slope
        else:
            # Create DataFrame for regression with covariates
            data_for_regression = pd.DataFrame(
                {
                    "methylation": methylation_values,
                    "metabolizer": metabolizer_status_numeric,
                }
            )

            # Add covariates
            for col in covariates.columns:
                data_for_regression[col] = covariates[col]

            # Formula for regression with covariates
            formula = "methylation ~ metabolizer + " + " + ".join(covariates.columns)
            model = ols(formula, data=data_for_regression).fit()

            pvalue = model.pvalues["metabolizer"]
            coefficient = model.params["metabolizer"]

        results.append({"CpG_site": cpg, "coefficient": coefficient, "pvalue": pvalue})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Step 3: Multiple testing correction
    log.append("\n### Multiple Testing Correction")
    log.append("- Applying Bonferroni correction")

    # Bonferroni correction
    results_df["adjusted_pvalue"] = results_df["pvalue"] * len(results_df)
    results_df["adjusted_pvalue"] = results_df["adjusted_pvalue"].clip(upper=1.0)  # Ensure p-values don't exceed 1

    # Step 4: Identify significant CpG sites
    significant_sites = results_df[results_df["adjusted_pvalue"] < pvalue_threshold]
    significant_sites = significant_sites.sort_values("adjusted_pvalue")

    log.append("\n### Results")
    log.append(f"- Number of significant CpG sites (adjusted p < {pvalue_threshold}): {len(significant_sites)}")

    if len(significant_sites) > 0:
        # Save significant sites to file. With the default name a second call overwrote the file the
        # first log names (hunt 2026-09-30, uT2-pharmacology-23).
        if output_file == "significant_cpg_sites.csv":
            output_file = _fresh_output_path(output_file)
        significant_sites.to_csv(output_file, index=False)
        log.append("- Top 5 significant CpG sites:")

        for _, row in significant_sites.head(5).iterrows():
            log.append(
                f"  * {row['CpG_site']}: coefficient = {row['coefficient']:.4f}, adj. p-value = {row['adjusted_pvalue']:.6f}"
            )

        log.append(f"- Full results saved to: {output_file}")
    else:
        log.append("- No significant CpG sites found after multiple testing correction")

    # Execution time
    execution_time = time.time() - start_time
    log.append("\n### Summary")
    log.append(f"- Analysis completed in {execution_time:.2f} seconds")

    return "\n".join(log)


def calculate_physicochemical_properties(smiles_string):
    """Calculate key physicochemical properties of a drug candidate molecule.

    Parameters
    ----------
    smiles_string : str
        The molecular structure in SMILES format

    Returns
    -------
    str
        A research log summarizing the calculated physicochemical properties and
        indicating where the detailed results are saved

    """
    import csv

    # RDKit is in none of the agent's interpreters on this box; say so rather than raise
    # ModuleNotFoundError (hunt 2026-09-30, uT2-pharmacology-22).
    try:
        from rdkit import Chem
        from rdkit.Chem import QED, Crippen, Descriptors, Lipinski
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:
        return (
            "ERROR: RDKit is not installed in this environment, so physicochemical properties cannot be "
            "calculated here."
        )

    # Create RDKit molecule from SMILES
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return "ERROR: Invalid SMILES string provided."
    except Exception as e:
        return f"ERROR: Failed to process SMILES string: {str(e)}"

    # Calculate basic properties
    properties = {
        "SMILES": smiles_string,
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "cLogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Ring Count": Descriptors.RingCount(mol),
    }

    # Ionisable groups by functional-group pattern. 'Drug-likeness Score' used to be Crippen.MolMR --
    # molar refractivity -- 'Estimated logD7.4' was cLogP copied over (wrong for any ionisable drug),
    # 'Estimated Acidic Groups' counted every O bonded to a trigonal carbon (ketones, esters and ethers
    # included) and 'Estimated Basic Groups' every N with under four neighbours (amides and nitriles
    # included). Drug-likeness is now QED, logD is not estimated, and the groups are counted by SMARTS
    # (hunt 2026-09-30, uT2-pharmacology-30).
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    # One match per acidic/basic centre (the C, S or P atom), however many OH or N it carries.
    acid_smarts = (
        "[CX3;$(C(=O)[OX2H1])]",  # carboxylic acid
        "[SX4;$(S(=O)(=O)[OX2H1])]",  # sulfonic acid
        "[PX4;$(P(=O)[OX2H1])]",  # phosphoric / phosphonic acid
        "[c;$(c1nnn[nH]1),$(c1nn[nH]n1)]",  # tetrazole (1H or 2H)
    )
    base_smarts = (
        "[NX3;H2,H1,H0;!$(N-[!#6;!#1]);!$(N-C=[O,S,N]);!$(N-a);!$(N-C#N)]",  # aliphatic amine
        "[CX3;!a;$(C(=[NX2;!a])[NX3;!a])]",  # amidine / guanidine carbon
    )

    def count_groups(patterns):
        return sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(p), uniquify=True)) for p in patterns)

    properties["Estimated Acidic Groups"] = count_groups(acid_smarts)
    properties["Estimated Basic Groups"] = count_groups(base_smarts)

    # Drug-likeness: quantitative estimate of drug-likeness (Bickerton et al. 2012), 0 to 1
    properties["Drug-likeness (QED)"] = round(QED.qed(mol), 3)
    properties["Molar Refractivity"] = round(Crippen.MolMR(mol), 2)

    # Save results to CSV
    csv_filename = _fresh_output_path("physicochemical_properties.csv")
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Property", "Value"])
        for prop, value in properties.items():
            writer.writerow([prop, value])

    # Generate research log
    log = f"""Physicochemical Property Calculation Research Log:

Analyzed compound with SMILES: {smiles_string}

Key properties:
- Molecular Weight: {properties["Molecular Weight"]} g/mol
- cLogP: {properties["cLogP"]}
- Topological Polar Surface Area: {properties["TPSA"]} Å²
- H-Bond Donors: {properties["H-Bond Donors"]}
- H-Bond Acceptors: {properties["H-Bond Acceptors"]}
- Rotatable Bonds: {properties["Rotatable Bonds"]}
- Drug-likeness (QED, 0-1): {properties["Drug-likeness (QED)"]}
- Molar Refractivity: {properties["Molar Refractivity"]}
- logD (at pH 7.4): not estimated -- it needs pKa prediction; cLogP above is the neutral form's logP
- Estimated Acidic Groups (carboxylic, sulfonic, phosphonic acids, tetrazoles): {properties["Estimated Acidic Groups"]}
- Estimated Basic Groups (aliphatic amines, amidines, guanidines): {properties["Estimated Basic Groups"]}

Complete results saved to: {csv_filename}
"""

    return log


def analyze_xenograft_tumor_growth_inhibition(
    data_path,
    time_column,
    volume_column,
    group_column,
    subject_column,
    output_dir="./results",
    control_group=None,
):
    """Analyze tumor growth inhibition in xenograft models across different treatment groups.

    Parameters
    ----------
    data_path : str
        Path to CSV or TSV file containing tumor volume measurements. The file should have columns for
        time, volume, treatment group, and subject ID
    time_column : str
        Name of the column containing time points (e.g., 'Day', 'Time')
    volume_column : str
        Name of the column containing tumor volume measurements
    group_column : str
        Name of the column containing treatment group labels
    subject_column : str
        Name of the column containing subject/mouse identifiers
    output_dir : str, optional
        Directory to save output files (default: "./results"; with the default, a later call's files get a
        _2, _3, ... suffix instead of overwriting an earlier result)
    control_group : str, optional
        Label of the control/vehicle group. Default: the one group whose label names a vehicle,
        control, PBS, saline, placebo or untreated arm; the tool asks for it when that is ambiguous.

    Returns
    -------
    str
        Research log summarizing the analysis steps, findings, and generated file paths

    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # With the default ./results a second call overwrote the files the first log names (hunt
    # 2026-09-30, uT2-pharmacology-23); an output_dir the caller names is used as given.
    xenograft_names = [
        "tumor_volume_statistics.csv",
        "repeated_measures_anova.csv",
        "tukey_posthoc_results.txt",
        "tumor_growth_curves.png",
    ]
    if output_dir == "./results":
        stats_file, anova_file, tukey_file, plot_file = _fresh_output_paths(output_dir, xenograft_names)
    else:
        stats_file, anova_file, tukey_file, plot_file = [os.path.join(output_dir, n) for n in xenograft_names]

    # Initialize research log
    log = "# Xenograft Tumor Growth Inhibition Analysis\n\n"

    # Load data from file
    log += "## 1. Data Loading and Summary\n\n"
    try:
        if data_path.endswith(".csv"):
            data_df = pd.read_csv(data_path)
        elif data_path.endswith((".tsv", ".txt")):
            data_df = pd.read_csv(data_path, sep="\t")
        else:
            log += "Error: Unsupported file format. Please provide a CSV or TSV file.\n"
            return log
        log += f"Successfully loaded tumor growth data from {data_path}\n"
    except Exception as e:
        log += f"Error loading data: {str(e)}\n"
        return log

    # Validate required columns
    required_columns = [time_column, volume_column, group_column, subject_column]
    missing_columns = [col for col in required_columns if col not in data_df.columns]
    if missing_columns:
        log += f"Error: Missing required columns: {', '.join(missing_columns)}\n"
        return log

    # Get unique groups and time points
    groups = data_df[group_column].unique()

    # The control used to be whichever group appeared first in the file, so an alphabetically sorted
    # file measured TGI against a drug arm and could name the vehicle 'the most effective treatment'
    # (hunt 2026-09-30, uT2-pharmacology-25).
    if control_group is not None:
        matches = [g for g in groups if str(g) == str(control_group)]
        if not matches:
            log += f"Error: control_group {control_group!r} is not one of the groups: {', '.join(map(str, groups))}\n"
            return log
    else:
        pattern = re.compile(r"\b(vehicle|control|ctrl|pbs|saline|placebo|untreated)\b", re.IGNORECASE)
        matches = [g for g in groups if pattern.search(str(g).replace("_", " "))]
        if len(matches) != 1:
            log += (
                "Error: cannot tell which group is the control "
                f"({'several look like one: ' + ', '.join(map(str, matches)) if matches else 'none is labelled as one'}). "
                f"Pass control_group= one of: {', '.join(map(str, groups))}\n"
            )
            return log
    control_group = matches[0]
    time_points = sorted(data_df[time_column].unique())
    n_groups = len(groups)
    log += f"- Number of treatment groups: {n_groups} ({', '.join(map(str, groups))})\n"
    log += f"- Number of time points: {len(time_points)}\n"
    log += f"- Number of subjects: {data_df[subject_column].nunique()}\n"
    log += f"- Total number of measurements: {len(data_df)}\n\n"

    # 2. Calculate group statistics at each time point
    log += "## 2. Tumor Growth Analysis\n\n"

    # Group statistics
    stats_df = (
        data_df.groupby([group_column, time_column])[volume_column]
        .agg(mean="mean", sem=lambda x: stats.sem(x), count="count")
        .reset_index()
    )

    # Save group statistics
    stats_df.to_csv(stats_file, index=False)
    log += f"Group statistics saved to: {stats_file}\n\n"

    # 3. Calculate tumor growth rates
    log += "## 3. Tumor Growth Rate Analysis\n\n"

    growth_rates = {}
    for group in groups:
        group_data = data_df[data_df[group_column] == group]

        # Calculate growth rate for each subject
        subject_growth_rates = []
        for subject in group_data[subject_column].unique():
            subject_data = group_data[group_data[subject_column] == subject]

            if len(subject_data) >= 2:
                # Simple linear regression for growth rate
                x = subject_data[time_column].values
                y = subject_data[volume_column].values
                slope, _, _, _, _ = stats.linregress(x, y)
                subject_growth_rates.append(slope)

        growth_rates[group] = subject_growth_rates
        mean_rate = np.mean(subject_growth_rates)
        sem_rate = stats.sem(subject_growth_rates)

        log += (
            f"- {group}: Mean growth rate = {mean_rate:.2f} ± {sem_rate:.2f} mm³/day (n={len(subject_growth_rates)})\n"
        )

    # 4. Calculate Tumor Growth Inhibition (TGI)
    log += "\n## 4. Tumor Growth Inhibition (TGI)\n\n"

    log += f"Control group: {control_group}\n\n"

    # Calculate TGI for the final time point
    final_time = max(time_points)
    final_data = data_df[data_df[time_column] == final_time]

    control_final_mean = final_data[final_data[group_column] == control_group][volume_column].mean()

    tgi_results = {}
    for group in groups:
        if group == control_group:
            continue

        group_final_mean = final_data[final_data[group_column] == group][volume_column].mean()
        tgi = ((control_final_mean - group_final_mean) / control_final_mean) * 100
        tgi_results[group] = tgi

        log += f"- {group}: TGI = {tgi:.1f}% (relative to {control_group})\n"

    # 5. Statistical Analysis
    log += "\n## 5. Statistical Analysis\n\n"

    # Repeated measures ANOVA
    log += "### Repeated Measures ANOVA\n\n"

    try:
        # Prepare data for repeated measures ANOVA
        formula = f"{volume_column} ~ C({group_column}) * C({time_column}) + C({subject_column})"
        model = ols(formula, data=data_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Save ANOVA results
        anova_table.to_csv(anova_file)

        log += f"ANOVA results saved to: {anova_file}\n\n"

        # Extract p-values
        group_effect_p = anova_table.loc[f"C({group_column})", "PR(>F)"]
        time_effect_p = anova_table.loc[f"C({time_column})", "PR(>F)"]
        interaction_p = anova_table.loc[f"C({group_column}):C({time_column})", "PR(>F)"]

        log += f"- Treatment effect: p = {group_effect_p:.4f}\n"
        log += f"- Time effect: p = {time_effect_p:.4f}\n"
        log += f"- Treatment × Time interaction: p = {interaction_p:.4f}\n\n"

        # Post-hoc analysis at final time point
        log += "### Post-hoc Analysis (Final Time Point)\n\n"

        # Perform Tukey's HSD test
        tukey = pairwise_tukeyhsd(endog=final_data[volume_column], groups=final_data[group_column], alpha=0.05)

        # Save Tukey results
        with open(tukey_file, "w") as f:
            f.write(str(tukey.summary()))

        log += f"Tukey's HSD results saved to: {tukey_file}\n\n"

        # Summarize significant comparisons
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

        sig_pairs = tukey_df[tukey_df["p-adj"] < 0.05]
        if len(sig_pairs) > 0:
            log += "Significant pairwise comparisons:\n"
            for _, row in sig_pairs.iterrows():
                log += f"- {row['group1']} vs {row['group2']}: p = {row['p-adj']:.4f}\n"
        else:
            log += "No significant pairwise comparisons found.\n"

    except Exception as e:
        log += f"Error in statistical analysis: {str(e)}\n"

    # 6. Generate tumor growth curves
    log += "\n## 6. Tumor Growth Visualization\n\n"

    plt.figure(figsize=(10, 6))

    for group in groups:
        group_stats = stats_df[stats_df[group_column] == group]
        plt.errorbar(
            group_stats[time_column],
            group_stats["mean"],
            yerr=group_stats["sem"],
            label=group,
            capsize=3,
            marker="o",
        )

    plt.xlabel(f"{time_column} (days)")
    plt.ylabel(f"Tumor Volume ({volume_column})")
    plt.title("Xenograft Tumor Growth Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save the plot
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    log += f"Tumor growth curve plot saved to: {plot_file}\n"

    # 7. Conclusion
    log += "\n## 7. Conclusion\n\n"

    # Summarize most effective treatment
    if tgi_results:
        best_treatment = max(tgi_results.items(), key=lambda x: x[1])
        if best_treatment[1] > 0:
            log += f"The most effective treatment was {best_treatment[0]} with a tumor growth inhibition of {best_treatment[1]:.1f}%.\n"
        else:
            log += f"No treatment reduced the final tumor volume relative to {control_group}.\n"

    # Statistical significance summary
    try:
        if group_effect_p < 0.05:
            log += "Statistical analysis confirmed significant differences between treatment groups.\n"
        else:
            log += "No statistically significant differences were found between treatment groups.\n"
    except Exception:
        pass

    return log


def analyze_pixel_distribution(image_path: str) -> dict:
    """Analyze western blot or DNA electrophoresis images and return pixel distribution statistics.

    Parameters
    ----------
    image_path : str
        Path to the input grayscale image. Automatically appends .png if no suffix is provided.

    Returns
    -------
    dict
        Summary dictionary containing image shape, intensity statistics, percentiles,
        histogram values, and brightness distribution for predefined buckets.

    """
    import cv2

    _DEFAULT_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    _DEFAULT_BRIGHTNESS_BUCKETS: tuple[tuple[int, int], ...] = (
        (0, 20),
        (20, 50),
        (50, 80),
        (80, 110),
        (110, 140),
        (140, 170),
        (170, 200),
        (200, 256),
    )

    # The description and docstring promise '.png is appended automatically'; the code never did, so a
    # model that followed them got FileNotFoundError (hunt 2026-09-30, uT2-pharmacology-19).
    if not os.path.splitext(str(image_path))[1] and not os.path.exists(str(image_path)):
        image_path = f"{image_path}.png"

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    percentiles = np.percentile(image, _DEFAULT_PERCENTILES).tolist()
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total_pixels = int(image.size)

    brightness_lines = []
    for low, high in _DEFAULT_BRIGHTNESS_BUCKETS:
        count = int(histogram[low:high].sum())
        ratio = round((count / total_pixels) * 100, 2) if total_pixels > 0 else 0.0
        brightness_lines.append(f"Range [{low:>3}, {high:>3}): {count:>8} px ({ratio:5.2f}%)")

    min_intensity = int(image.min())
    max_intensity = int(image.max())
    mean_intensity = round(float(image.mean()), 2)
    std_intensity = round(float(image.std()), 2)

    return {
        "shape": f"({image.shape[0]}, {image.shape[1]})",
        "intensity_stats": {
            "min": min_intensity,
            "max": max_intensity,
            "mean": mean_intensity,
            "std_dev": std_intensity,
        },
        "percentiles_label": "percentiles (1, 5, 10, 25, 50, 75, 90, 95, 99):",
        "percentiles_values": ", ".join(f"{float(p):.1f}" for p in percentiles),
        "pixel_brightness_distribution": brightness_lines,
    }


def find_roi_from_image(
    image_path: str,
    lower_threshold: int,
    upper_threshold: int,
    number_of_bands: int,
    debug: bool = True,
    output_dir: str | None = None,
) -> tuple[str, list]:
    """Find the ROIs of the bands from the image which is determined by analyze_pixel_distribution function.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    lower_threshold : int
        Pixel intensities lower than this value are used to make the binary image.
    upper_threshold : int
        Pixel intensities greater than or equal to this value are used to make the binary image.
    number_of_bands : int
        The actual number of bands in the image.
    debug : bool, optional
        If True, draw green contours (hulls) and blue keypoint boxes for debugging.
        Default is True.
    output_dir : str, optional
        Directory for the mask and annotated images. Default: the current working directory (never
        the input image's own directory, which may be an upload or a shared library).

    Returns
    -------
    tuple[str, list]
        A tuple containing:
        - str: Absolute path to the saved annotated image
        - list: List of ROI coordinates in (x, y, width, height) format.
        The ROI list can be converted to target_bands for analyze_western_blot:
        annotated_path, rois = find_roi_from_image(...)
        target_bands = [{"name": f"band_{i}", "roi": list(roi)} for i, roi in enumerate(rois)]

    Raises
    ------
    ValueError
        If threshold values are outside the valid range or inconsistent.
    FileNotFoundError
        If the source image cannot be loaded.

    """
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import cv2

    ROI = tuple[int, int, int, int]

    def load_grayscale_image(path: str) -> cv2.Mat:
        """Load a grayscale image from disk."""
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to load image at '{path}'")
        return image

    def build_blob_detector(
        min_threshold: int = 0,
        max_threshold: int = 200,
        min_area: int = 120,
        min_convexity: float = 0.7,
        min_inertia: float = 0.001,
        max_inertia: float = 0.4,
    ) -> cv2.SimpleBlobDetector:
        """Configure and return a SimpleBlobDetector instance."""
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold
        params.filterByArea = True
        params.minArea = min_area
        params.filterByConvexity = True
        params.minConvexity = min_convexity
        params.filterByInertia = True
        params.minInertiaRatio = min_inertia
        params.maxInertiaRatio = max_inertia
        return cv2.SimpleBlobDetector_create(params)

    def detect_blobs(image: cv2.Mat, detector: cv2.SimpleBlobDetector) -> list[cv2.KeyPoint]:
        """Detect blob keypoints in the provided image."""
        keypoints = detector.detect(image)
        print(f"Detected {len(keypoints)} keypoints.")
        for index, keypoint in enumerate(keypoints):
            print(f"[{index}] position={keypoint.pt}, size={keypoint.size}")
        return keypoints

    def find_band_contours(
        binary_mask: cv2.Mat,
        min_area: int = 100,
        use_morphology: bool = True,
    ) -> list[cv2.Mat]:
        """Find band contours from binary mask using morphological operations."""
        processed_mask = binary_mask.copy()

        if use_morphology:
            # 가로(Horizontal) 방향으로 떨어진 덩어리를 잇기 위해 가로가 긴 커널 사용
            # (50, 1)의 50은 두 덩어리 사이의 픽셀 거리보다 커야 합니다.
            kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))

            # OPEN(끊기) 대신 CLOSE(잇기)를 사용하여 빈 공간을 메움
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel_connect, iterations=1)

        # Find ALL contours (not just external) to detect separate bands
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Found {len(contours)} total contours")

        # Filter contours by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                filtered_contours.append(contour)

        print(f"Filtered to {len(filtered_contours)} contours with area >= {min_area}")

        return filtered_contours

    def analyze_roi_pixel_distribution(
        image: cv2.Mat,
        roi: ROI,
    ) -> dict:
        """Analyze pixel distribution of an ROI to distinguish between text and bands.

        Parameters
        ----------
        image : cv2.Mat
            Original grayscale image
        roi : ROI
            ROI coordinates (x, y, width, height)

        Returns
        -------
        dict
            Dictionary containing edge_strength, std_dev, and gradient_magnitude

        """
        x, y, w, h = roi

        # Extract ROI region from original image
        roi_region = image[y : y + h, x : x + w]

        if roi_region.size == 0:
            return {"edge_strength": 0.0, "std_dev": 0.0, "gradient_magnitude": 0.0}

        # Calculate standard deviation of pixel intensities
        std_dev = float(np.std(roi_region))

        # Calculate edge strength using Laplacian
        laplacian = cv2.Laplacian(roi_region, cv2.CV_64F)
        edge_strength = float(np.mean(np.abs(laplacian)))

        # Calculate gradient magnitude using Sobel
        sobelx = cv2.Sobel(roi_region, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi_region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))

        return {
            "edge_strength": edge_strength,
            "std_dev": std_dev,
            "gradient_magnitude": gradient_magnitude,
        }

    def filter_rois_by_pixel_distribution(
        image: cv2.Mat,
        rois: list[ROI],
        hulls: list[cv2.Mat],
        max_edge_strength: float = 10.0,
        max_gradient_magnitude: float = 70.0,
        max_std_dev: float = 50.0,
    ) -> tuple[list[ROI], list[cv2.Mat]]:
        """Filter ROIs to remove text-like regions and keep band-like regions.

        Text regions have very high edge strength, sharp gradients, and high std_dev.
        Band regions have low edge strength, moderate gradients, and moderate std_dev.

        Parameters
        ----------
        image : cv2.Mat
            Original grayscale image
        rois : List[ROI]
            List of ROI coordinates
        hulls : List[cv2.Mat]
            List of corresponding convex hulls
        max_edge_strength : float, optional
            Maximum edge strength for band-like regions (text typically >20)
        max_gradient_magnitude : float, optional
            Maximum gradient magnitude for band-like regions (text typically >100)
        max_std_dev : float, optional
            Maximum standard deviation for band-like regions (text typically >80)

        Returns
        -------
        tuple
            Filtered ROIs and hulls that are band-like

        """
        filtered_rois: list[ROI] = []
        filtered_hulls: list[cv2.Mat] = []

        print("\n=== ROI Pixel Distribution Analysis ===")

        for idx, (roi, hull) in enumerate(zip(rois, hulls, strict=True)):
            analysis = analyze_roi_pixel_distribution(image, roi)

            edge_strength = analysis["edge_strength"]
            gradient_magnitude = analysis["gradient_magnitude"]
            std_dev = analysis["std_dev"]

            # Determine if this ROI is band-like or text-like
            # Text has very high edge strength (>20) and very high gradient (>100)
            # Bands have low edge strength (<10) and moderate gradient (<50)
            # Also check std_dev to filter out high-contrast text regions
            is_band = (
                edge_strength <= max_edge_strength
                and gradient_magnitude <= max_gradient_magnitude
                and std_dev <= max_std_dev
            )

            status = "✓ BAND" if is_band else "✗ TEXT"
            print(f"ROI {idx}: edge={edge_strength:.2f}, grad={gradient_magnitude:.2f}, std={std_dev:.2f} -> {status}")

            if is_band:
                filtered_rois.append(roi)
                filtered_hulls.append(hull)

        print(f"\nFiltered: {len(filtered_rois)}/{len(rois)} ROIs kept as bands")
        print("=" * 40 + "\n")

        return filtered_rois, filtered_hulls

    def compute_rois(
        image: cv2.Mat,
        binary_mask: cv2.Mat,
        keypoints: Iterable[cv2.KeyPoint],
        padding: tuple[int, int] = (5, 5),
        min_contour_area: int = 100,
        filter_by_distribution: bool = True,
    ) -> tuple[list[ROI], list[cv2.Mat]]:
        """Compute global ROIs for each keypoint by matching them to band contours."""
        # Find band contours from binary mask
        band_contours = find_band_contours(binary_mask, min_area=min_contour_area)

        if not band_contours:
            print("No band contours found!")
            return [], []

        auto_rois: list[ROI] = []
        global_hulls: list[cv2.Mat] = []
        matched_contours = set()  # Track which contours have been matched
        used_contours = set()  # Track which contours have already been used to avoid duplicates

        for keypoint in keypoints:
            cx, cy = int(keypoint.pt[0]), int(keypoint.pt[1])
            keypoint_center = (float(cx), float(cy))

            # Find the contour that contains this keypoint
            matched_contour = None
            matched_idx = None
            for idx, contour in enumerate(band_contours):
                # Skip if this contour has already been used
                if idx in used_contours:
                    continue
                # Use pointPolygonTest to check if keypoint center is inside contour
                # Returns positive if inside, negative if outside, zero if on edge
                distance = cv2.pointPolygonTest(contour, keypoint_center, False)
                if distance >= 0:  # Inside or on edge
                    matched_contour = contour
                    matched_idx = idx
                    matched_contours.add(idx)
                    print(f"Keypoint at ({cx}, {cy}) matched to contour {idx}")
                    break

            if matched_contour is None:
                print(f"Warning: Keypoint at ({cx}, {cy}) not matched to any contour")
                continue

            # Mark this contour as used to avoid duplicate ROIs
            used_contours.add(matched_idx)

            # Compute convex hull from the matched contour
            hull = cv2.convexHull(matched_contour)
            if hull is None or len(hull) < 3:
                continue

            # Get bounding rectangle from hull with padding
            pad_x, pad_y = padding
            rx, ry, rw, rh = cv2.boundingRect(hull)

            # Skip if bounding rect is too large (likely covering entire image or invalid)
            max_roi_area_ratio = 0.5  # Maximum 50% of image area
            roi_area = rw * rh
            image_area = image.shape[0] * image.shape[1]
            if roi_area > image_area * max_roi_area_ratio:
                print(f"Warning: ROI too large ({rw}x{rh}), skipping. This may indicate a detection error.")
                continue

            global_x = max(0, rx - pad_x)
            global_y = max(0, ry - pad_y)
            global_w = min(image.shape[1] - global_x, rw + 2 * pad_x)
            global_h = min(image.shape[0] - global_y, rh + 2 * pad_y)

            if global_w <= 0 or global_h <= 0:
                continue

            roi = (global_x, global_y, global_w, global_h)
            auto_rois.append(roi)
            global_hulls.append(hull)

        print(f"Matched {len(matched_contours)} contours to keypoints")

        # Filter ROIs by pixel distribution to remove text-like regions
        if filter_by_distribution and auto_rois:
            auto_rois, global_hulls = filter_rois_by_pixel_distribution(image, auto_rois, global_hulls)

        return auto_rois, global_hulls

    def annotate_keypoints(
        image: cv2.Mat,
        keypoints: Iterable[cv2.KeyPoint],
        rois: Iterable[ROI],
        hulls: Iterable[cv2.Mat] | None = None,
        debug: bool = False,
    ) -> cv2.Mat:
        """Draw ROIs, convex hulls, and index labels on the image.

        Parameters
        ----------
        image : cv2.Mat
            Input grayscale image
        keypoints : Iterable[cv2.KeyPoint]
            Detected keypoints
        rois : Iterable[ROI]
            ROI coordinates to draw
        hulls : Iterable[cv2.Mat] | None, optional
            Convex hulls to draw (only if debug=True)
        debug : bool, optional
            If True, draw green contours (hulls) and blue keypoint boxes

        Returns
        -------
        cv2.Mat
            Annotated image

        """
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw convex hulls in green if debug mode is enabled
        if debug and hulls is not None:
            for hull in hulls:
                cv2.drawContours(output, [hull], -1, (0, 255, 0), 2)  # Green color in BGR

        # Draw ROIs in red (always drawn)
        for roi in rois:
            x, y, w, h = roi
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw keypoints in blue and index labels (only if debug mode is enabled)
        if debug:
            for index, keypoint in enumerate(keypoints):
                x, y = keypoint.pt
                size = keypoint.size

                # Draw blue rectangle around keypoint
                # Use size as half-width and half-height for the rectangle
                half_size = int(size / 2)
                pt1 = (int(x) - half_size, int(y) - half_size)
                pt2 = (int(x) + half_size, int(y) + half_size)
                cv2.rectangle(output, pt1, pt2, (255, 0, 0), 2)  # Blue color in BGR

                # Draw index label
                cv2.putText(
                    output,
                    str(index),
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        return output

    def show_rois(rois: Sequence[ROI]) -> None:
        """Print ROI information to stdout."""
        print(f"Detected ROI count: {len(rois)}")
        for index, roi in enumerate(rois):
            print(f"ROI {index}: {roi}")

    if not 0 <= lower_threshold <= 255:
        raise ValueError("lower_threshold must be within [0, 255].")
    if not 0 <= upper_threshold <= 255:
        raise ValueError("upper_threshold must be within [0, 255].")
    if lower_threshold > upper_threshold:
        raise ValueError("lower_threshold cannot be greater than upper_threshold.")

    original_image = load_grayscale_image(image_path)
    mask = cv2.inRange(original_image, lower_threshold, upper_threshold)
    mask = cv2.bitwise_not(mask)
    detector = build_blob_detector()

    # Detect blobs in the mask image
    keypoints = detect_blobs(mask, detector)
    rois, hulls = compute_rois(original_image, mask, keypoints)
    show_rois(rois)

    # Draw ROIs, convex hulls, and keypoints on the mask image
    # The images used to go beside the INPUT (an upload directory or the shared library), and
    # cv2.imwrite's False was ignored, so the returned 'saved annotated image' path could name a file
    # that was never written (hunt 2026-09-30, uT2-pharmacology-18).
    out_dir = Path(output_dir) if output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_image(path: Path, image) -> None:
        if not cv2.imwrite(str(path), image):
            raise OSError(f"could not write {path}; pass a writable output_dir")

    annotated_mask = annotate_keypoints(mask, keypoints, rois, hulls, debug=debug)
    mask_path = out_dir / f"{Path(image_path).stem}_mask.png"
    write_image(mask_path, annotated_mask)

    # Draw ROIs, convex hulls, and keypoints on the original image
    annotated_image = annotate_keypoints(original_image, keypoints, rois, hulls, debug=debug)
    annotated_image_path = out_dir / f"{Path(image_path).stem}_annotated.png"
    write_image(annotated_image_path, annotated_image)

    if len(rois) != number_of_bands:
        print(f"Warning: Detected {len(rois)} ROIs, but expected {number_of_bands} ROIs.")
        print(
            "Please check the image and try to adjust the thresholds. Or you can manually infer the ROIs from the annotated image."
        )

    return str(annotated_image_path.resolve()), rois


def analyze_western_blot(
    blot_image_path,
    target_bands,
    loading_control_band,
    antibody_info,
    output_dir="./results",
    invert=None,
):
    """Performs densitometric analysis of Western blot images to quantify relative protein expression.

    Parameters
    ----------
    blot_image_path : str
        Path to the Western blot image file
    target_bands : list of dict
        List of dictionaries containing information about target protein bands.
        Each dict should have 'name' and 'roi' (region of interest as [x, y, width, height]).
        To generate this from find_roi_from_image output:
        annotated_path, rois = find_roi_from_image(...)
        target_bands = [{"name": f"band_{i}", "roi": list(roi)} for i, roi in enumerate(rois)]
        Or manually specify names: target_bands = [{"name": "protein_name", "roi": [x, y, w, h]}, ...]
    loading_control_band : dict
        Dictionary with 'name' and 'roi' for the loading control protein (e.g., β-actin, GAPDH)
    antibody_info : dict
        Dictionary containing information about antibodies used
        Should have 'primary' and 'secondary' keys with antibody details
    output_dir : str, optional
        Directory to save output files, defaults to './results' (with the default, a later call writes
        western_blot_results_2.csv, ... instead of overwriting an earlier result)
    invert : bool or None, optional
        True for dark bands on a light background (the usual export), False for light bands on a dark
        background, None (default) to detect it from the image

    Returns
    -------
    str
        Research log summarizing the Western blot analysis process and results

    """
    import os

    import numpy as np
    from skimage import io

    # Band intensity was np.sum of the raw ROI pixels: no inversion, so on the usual dark-band export a
    # stronger band gave a SMALLER number and relative expression came out inverted; no background
    # subtraction, so the background inside each box dominated; and colour images were averaged over
    # every channel, alpha included. Signal is now the background-subtracted band density, in the
    # polarity of the blot, from a luminance image at the file's own bit depth (hunt 2026-09-30,
    # uT2-pharmacology-17).

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the Western blot image
    image = np.asarray(io.imread(blot_image_path))
    if image.ndim == 3:  # colour: luminance from the colour channels only (never alpha)
        if image.shape[-1] >= 3:
            image = image[..., :3].astype(float) @ np.array([0.2125, 0.7154, 0.0721])
        else:  # grey + alpha
            image = image[..., 0].astype(float)
    else:
        image = image.astype(float)

    def roi_pixels(roi):
        x, y, w, h = (int(v) for v in roi)
        return image[y : y + h, x : x + w]

    def local_background(roi):
        """Median of a ring of pixels around the ROI (the whole image's median if there is none)."""
        x, y, w, h = (int(v) for v in roi)
        margin = max(2, int(round(0.25 * min(w, h))))
        y0, y1 = max(0, y - margin), min(image.shape[0], y + h + margin)
        x0, x1 = max(0, x - margin), min(image.shape[1], x + w + margin)
        ring = np.ones((y1 - y0, x1 - x0), dtype=bool)
        ring[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = False
        values = image[y0:y1, x0:x1][ring]
        return float(np.median(values)) if values.size else float(np.median(image))

    all_bands = [loading_control_band, *target_bands]
    band_values = np.concatenate([roi_pixels(b["roi"]).ravel() for b in all_bands])
    if band_values.size == 0:
        return "Error: every ROI is empty or outside the image; check the [x, y, width, height] values."
    if invert is None:
        invert = bool(np.median(band_values) < np.median(image))
        polarity = "detected from the image"
    else:
        polarity = "as given"
    sign = -1.0 if invert else 1.0

    def band_signal(roi):
        return float(np.sum(sign * (roi_pixels(roi) - local_background(roi))))

    # Initialize results dictionary
    results = {
        "loading_control": {"name": loading_control_band["name"], "intensity": 0},
        "targets": [],
    }

    # Analyze loading control band
    lc_intensity = band_signal(loading_control_band["roi"])
    results["loading_control"]["intensity"] = lc_intensity
    if lc_intensity <= 0:
        return (
            f"Error: the loading control ({loading_control_band['name']}) has no signal above its local "
            "background in this polarity; check its ROI, or set invert= explicitly."
        )

    # Analyze target protein bands
    for band in target_bands:
        band_intensity = band_signal(band["roi"])

        # Calculate relative expression (normalized to loading control)
        relative_expression = band_intensity / lc_intensity

        results["targets"].append(
            {
                "name": band["name"],
                "intensity": band_intensity,
                "relative_expression": relative_expression,
            }
        )

    # Generate results table and save to CSV
    # With the default ./results a second call overwrote the file the first log names (hunt 2026-09-30,
    # uT2-pharmacology-23); an output_dir the caller names is used as given.
    if output_dir == "./results":
        results_file = _fresh_output_path(os.path.join(output_dir, "western_blot_results.csv"))
    else:
        results_file = os.path.abspath(os.path.join(output_dir, "western_blot_results.csv"))
    with open(results_file, "w") as f:
        f.write("Protein,Background-subtracted Intensity,Relative Expression\n")
        f.write(f"{results['loading_control']['name']},{results['loading_control']['intensity']:.6g},1.0\n")
        for target in results["targets"]:
            f.write(f"{target['name']},{target['intensity']:.6g},{target['relative_expression']:.4f}\n")

    # Generate research log
    log = "## Western Blot Analysis\n\n"
    log += f"Analyzed Western blot image: {os.path.basename(blot_image_path)}\n\n"
    log += "### Antibodies Used\n"
    log += f"Primary antibody: {antibody_info['primary']}\n"
    log += f"Secondary antibody: {antibody_info['secondary']}\n\n"
    log += "### Analysis Steps\n"
    log += "1. Loaded Western blot image and converted to grayscale (luminance; alpha ignored)\n"
    log += (
        f"2. Bands read as {'dark on a light' if invert else 'light on a dark'} background ({polarity}); "
        "each band's signal is its pixels' summed difference from the median of a ring around it\n"
    )
    log += f"3. Quantified loading control ({loading_control_band['name']}) and target band intensities\n"
    log += "4. Calculated relative expression by normalizing to loading control\n\n"
    log += "### Results\n"
    log += f"Loading control ({loading_control_band['name']}): {results['loading_control']['intensity']:.6g} intensity units\n\n"
    log += "Target proteins:\n"
    for target in results["targets"]:
        log += f"- {target['name']}: {target['intensity']:.6g} intensity units, "
        log += f"{target['relative_expression']:.4f} relative expression\n"
    log += f"\nDetailed results saved to: {results_file}\n"

    return log


# DDInter Drug-Drug Interaction Analysis Functions


#: The eight DDInter 2.0 per-ATC-class downloads the loader reads (a subset is enough).
_DDINTER_CSV_FILES = (
    "ddinter_alimentary_tract_metabolism.csv",
    "ddinter_antineoplastic.csv",
    "ddinter_antiparasitic.csv",
    "ddinter_blood_organs.csv",
    "ddinter_dermatological.csv",
    "ddinter_hormonal.csv",
    "ddinter_respiratory.csv",
    "ddinter_various.csv",
)

#: The ATC categories an interaction record can carry -- the only "type" DDInter 2.0 records.
_DDINTER_CATEGORIES = tuple(name[len("ddinter_") : -len(".csv")] for name in _DDINTER_CSV_FILES)


def _ddinter_csvs_in(directory):
    """The DDInter CSVs present in ``directory`` (absolute paths, in the canonical order)."""
    if not directory or not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, name) for name in _DDINTER_CSV_FILES if os.path.isfile(os.path.join(directory, name))
    ]


def _resolve_ddinter_data_lake(data_lake_path):
    """Where the DDInter CSVs are read from, and every directory that was checked.

    ``data_lake_path=None`` used to mean the package's ``tool/schema_db`` directory, which never holds
    the CSVs, so every default call answered "No DDInter CSV files found" without saying where it had
    looked. The default now checks the agent's data lake (``<SOG_PATH>/spatialomicsgym_data/data_lake``,
    as ``STCoscientist`` lays it out), the pre-rename Biomni layout beside it where these CSVs were
    staged, and that legacy package directory -- and the error names all three (hunt 2026-09-30,
    uT2-pharmacology-21).
    """
    if data_lake_path:
        candidates = [os.path.abspath(os.path.expanduser(str(data_lake_path)))]
    else:
        root = os.path.abspath(
            os.path.expanduser(os.environ.get("SOG_PATH") or os.environ.get("SOG_DATA_PATH") or "./data")
        )
        candidates = [
            os.path.join(root, "spatialomicsgym_data", "data_lake"),
            os.path.join(root, "biomni_data", "data_lake"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema_db"),
        ]
    for directory in candidates:
        if _ddinter_csvs_in(directory):
            return directory, candidates
    return None, candidates


def _ddinter_cache_dir(csv_paths):
    """A per-user cache directory for the derived pickles, keyed by the exact source CSVs.

    The pickles used to be written into the package's ``tool/schema_db/`` with plain ``open(..., "wb")``:
    read-only under the portal's dropped uid (so every call re-parsed ~222k rows and then failed), and a
    run killed mid-write left a truncated pickle the all-exist check accepted forever. The cache now
    lives under ``$XDG_CACHE_HOME`` (which the portal points at each account's own home), and its key
    changes whenever a source CSV does (hunt 2026-09-30, uT2-pharmacology-27).
    """
    import hashlib

    digest = hashlib.sha256()
    for path in csv_paths:
        st = os.stat(path)
        digest.update(f"{os.path.abspath(path)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "spatialomicsgym", "ddinter", digest.hexdigest()[:16])


def _write_pickle_atomically(obj, path):
    """Pickle ``obj`` to ``path`` through a sibling temp file, so a reader never sees a partial one."""
    import tempfile

    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path) + ".", suffix=".partial")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _load_ddinter_data(data_lake_path):
    """
    Load the DDInter datasets, deriving (and caching) the lookup tables from the CSVs if needed.

    Parameters
    ----------
    data_lake_path : str or None
        Directory containing the DDInter CSVs (``ddinter_*.csv``); None checks the default locations
        listed by ``_resolve_ddinter_data_lake``.

    Returns
    -------
    tuple
        (drug_info, interaction_matrix, name_mapping) dictionaries

    Raises
    ------
    FileNotFoundError
        When no DDInter CSV is found; the message names every directory checked and the file names.
    """
    directory, checked = _resolve_ddinter_data_lake(data_lake_path)
    if directory is None:
        raise FileNotFoundError(
            "No DDInter CSV files found. Checked: "
            + ", ".join(checked)
            + ". Expected one or more of: "
            + ", ".join(_DDINTER_CSV_FILES)
            + " (DDInter 2.0 downloads). Pass data_lake_path= the directory that holds them."
        )
    csv_paths = _ddinter_csvs_in(directory)
    cache_dir = _ddinter_cache_dir(csv_paths)
    names = ("ddinter_drugs.pkl", "ddinter_interactions.pkl", "ddinter_name_mapping.pkl")
    pkl_files = [os.path.join(cache_dir, name) for name in names]

    if all(os.path.exists(f) for f in pkl_files):
        try:
            loaded = []
            for path in pkl_files:
                with open(path, "rb") as f:
                    loaded.append(pickle.load(f))
            return tuple(loaded)
        except Exception:
            pass  # a damaged cache entry is rebuilt from the CSVs below, never trusted

    return _process_ddinter_data_inline(directory, cache_dir)


def _process_ddinter_data_inline(data_lake_path, output_dir):
    """
    Process DDInter CSV files into standardized lookup tables, caching them as pickles.

    This function processes raw DDInter 2.0 CSV files and creates standardized
    data structures for use in SpatialOmicsLab drug-drug interaction analysis.

    Parameters
    ----------
    data_lake_path : str
        Path to data lake directory containing raw DDInter CSV files
    output_dir : str
        Cache directory for the processed pickle files. Each is written atomically; a directory that
        cannot be written only costs the cache, never the result.

    Returns
    -------
    tuple
        (drug_info, interaction_matrix, name_mapping) dictionaries
    """
    import pandas as pd

    # Load and combine all CSV files
    dataframes = []
    for file_path in _ddinter_csvs_in(data_lake_path):
        df = pd.read_csv(file_path)
        # Add source category
        category = os.path.basename(file_path).replace("ddinter_", "").replace(".csv", "")
        df["category"] = category
        dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError(
            f"No DDInter CSV files found in {os.path.abspath(data_lake_path)} "
            f"(expected one or more of: {', '.join(_DDINTER_CSV_FILES)})"
        )

    # Process data
    drug_info = _build_drug_registry_inline(dataframes)
    interaction_matrix = _create_interaction_matrix_inline(dataframes)
    name_mapping = _create_name_mapping_inline(drug_info)
    stats = _generate_ddinter_statistics_inline(drug_info, interaction_matrix)

    try:
        os.makedirs(output_dir, exist_ok=True)
        _write_pickle_atomically(drug_info, os.path.join(output_dir, "ddinter_drugs.pkl"))
        _write_pickle_atomically(interaction_matrix, os.path.join(output_dir, "ddinter_interactions.pkl"))
        _write_pickle_atomically(name_mapping, os.path.join(output_dir, "ddinter_name_mapping.pkl"))
        _write_pickle_atomically(stats, os.path.join(output_dir, "ddinter_statistics.pkl"))
    except OSError:
        pass  # caching is an optimisation; the tables below are already complete in memory

    return drug_info, interaction_matrix, name_mapping


def _standardize_drug_name_processing(drug_name):
    """Standardize drug names for consistent matching during processing."""
    import pandas as pd

    if pd.isna(drug_name):
        return ""

    # Convert to lowercase and strip whitespace
    standardized = str(drug_name).strip().lower()

    # Remove common suffixes and prefixes
    standardized = standardized.replace(" hydrochloride", "")
    standardized = standardized.replace(" sulfate", "")
    standardized = standardized.replace(" sodium", "")
    standardized = standardized.replace(" potassium", "")
    standardized = standardized.replace(" calcium", "")
    standardized = standardized.replace(" magnesium", "")

    return standardized


def _build_drug_registry_inline(dataframes):
    """Build comprehensive drug registry from all interactions."""

    drug_registry = {}

    for df in dataframes:
        for _, row in df.iterrows():
            drug_a_id = row["DDInterID_A"]
            drug_a_name = row["Drug_A"]
            drug_b_id = row["DDInterID_B"]
            drug_b_name = row["Drug_B"]

            # Add Drug A
            if drug_a_id not in drug_registry:
                drug_registry[drug_a_id] = {
                    "name": drug_a_name,
                    "standardized_name": _standardize_drug_name_processing(drug_a_name),
                    "categories": set(),
                    "interactions": set(),
                }
            drug_registry[drug_a_id]["categories"].add(row["category"])

            # Add Drug B
            if drug_b_id not in drug_registry:
                drug_registry[drug_b_id] = {
                    "name": drug_b_name,
                    "standardized_name": _standardize_drug_name_processing(drug_b_name),
                    "categories": set(),
                    "interactions": set(),
                }
            drug_registry[drug_b_id]["categories"].add(row["category"])

            # Record interactions
            drug_registry[drug_a_id]["interactions"].add(drug_b_id)
            drug_registry[drug_b_id]["interactions"].add(drug_a_id)

    # Convert sets to lists for pickle serialization
    for drug_id in drug_registry:
        drug_registry[drug_id]["categories"] = list(drug_registry[drug_id]["categories"])
        drug_registry[drug_id]["interactions"] = list(drug_registry[drug_id]["interactions"])

    return drug_registry


def _create_interaction_matrix_inline(dataframes):
    """Create interaction matrix for fast lookups using standardized drug names."""
    from collections import defaultdict

    import pandas as pd

    combined_df = pd.concat(dataframes, ignore_index=True)
    interaction_matrix = defaultdict(lambda: defaultdict(list))

    # Create bidirectional interaction matrix using standardized names
    for _, row in combined_df.iterrows():
        drug_a_std = _standardize_drug_name_processing(row["Drug_A"])
        drug_b_std = _standardize_drug_name_processing(row["Drug_B"])
        level = row["Level"]
        category = row["category"]

        interaction_data = {
            "level": level,
            "category": category,
            "drug_a_id": row["DDInterID_A"],
            "drug_b_id": row["DDInterID_B"],
            "drug_a_name": row["Drug_A"],
            "drug_b_name": row["Drug_B"],
        }

        # Add both directions using standardized names as keys
        interaction_matrix[drug_a_std][drug_b_std].append(interaction_data)
        interaction_matrix[drug_b_std][drug_a_std].append(interaction_data)

    # Convert to regular dict for pickle
    interaction_matrix = dict(interaction_matrix)
    for drug in interaction_matrix:
        interaction_matrix[drug] = dict(interaction_matrix[drug])

    return interaction_matrix


def _create_name_mapping_inline(drug_info):
    """Create drug name to ID mapping for fuzzy matching."""
    name_mapping = {}

    for drug_id, drug_data in drug_info.items():
        original_name = drug_data["name"]
        standardized_name = drug_data["standardized_name"]

        # Map both original and standardized names
        name_mapping[original_name.lower()] = drug_id
        name_mapping[standardized_name] = drug_id

    return name_mapping


def _generate_ddinter_statistics_inline(drug_info, interaction_matrix):
    """Generate statistics about the processed data."""
    from collections import defaultdict

    stats = {
        "total_drugs": len(drug_info),
        "total_interactions": 0,
        "interaction_levels": defaultdict(int),
        "drug_categories": defaultdict(int),
        "most_connected_drugs": [],
    }

    # Count interactions and levels
    for drug_a in interaction_matrix:
        for drug_b in interaction_matrix[drug_a]:
            interactions = interaction_matrix[drug_a][drug_b]
            stats["total_interactions"] += len(interactions)

            for interaction in interactions:
                stats["interaction_levels"][interaction["level"]] += 1

    # Count drug categories
    for drug_data in drug_info.values():
        for category in drug_data["categories"]:
            stats["drug_categories"][category] += 1

    # Find most connected drugs
    connection_counts = []
    for drug_id, drug_data in drug_info.items():
        connection_counts.append(
            {"drug_id": drug_id, "name": drug_data["name"], "connections": len(drug_data["interactions"])}
        )

    connection_counts.sort(key=lambda x: x["connections"], reverse=True)
    stats["most_connected_drugs"] = connection_counts[:10]

    return stats


def _standardize_drug_name(drug_name, name_mapping):
    """
    Standardize drug names using fuzzy matching against DDInter database.

    Parameters
    ----------
    drug_name : str
        Original drug name
    name_mapping : dict
        Drug name to ID mapping dictionary

    Returns
    -------
    str or None
        Standardized drug name or None if not found
    """
    from difflib import get_close_matches

    # name_mapping holds both the original and the salt-stripped spelling, but interaction_matrix is
    # keyed by the stripped one only. Returning the matched spelling as-is meant 'Magnesium sulfate'
    # never found a single interaction, and a Major pair was reported 'Safe 100/100'; the matrix key
    # is what every caller looks up (hunt 2026-09-30, uT2-pharmacology-2).
    query = str(drug_name).strip().lower()

    # Direct match
    if query in name_mapping:
        return _standardize_drug_name_processing(query)

    # Fuzzy match
    matches = get_close_matches(query, name_mapping.keys(), n=1, cutoff=0.8)
    if matches:
        return _standardize_drug_name_processing(matches[0])

    return None


def _format_interaction_result(interaction_data, drug_name_a, drug_name_b, include_mechanisms=True):
    """
    Format interaction results for research log.

    Parameters
    ----------
    interaction_data : list
        List of interaction data dictionaries
    drug_name_a : str
        First drug name
    drug_name_b : str
        Second drug name
    include_mechanisms : bool
        Whether to include detailed mechanism information

    Returns
    -------
    str
        Formatted interaction description
    """
    if not interaction_data:
        return f"No interactions found between {drug_name_a} and {drug_name_b}"

    result = f"Interaction between {drug_name_a} and {drug_name_b}:\n"

    for i, interaction in enumerate(interaction_data, 1):
        level = interaction.get("level", "Unknown")
        category = interaction.get("category", "Unknown")

        result += f"  {i}. Severity: {level}\n"
        result += f"     Category: {category.replace('_', ' ').title()}\n"

        if include_mechanisms:
            if level in ("Major", "Moderate", "Minor"):
                result += f"     Clinical significance: {level} interaction requiring appropriate monitoring\n"
            else:
                result += "     Clinical significance: severity not graded by DDInter -- an interaction is recorded\n"

    return result


def query_drug_interactions(drug_names, interaction_types=None, severity_levels=None, data_lake_path=None):
    """
    Query drug-drug interactions from DDInter database.

    Parameters
    ----------
    drug_names : list of str
        List of drug names to query for interactions
    interaction_types : list of str, optional
        Filter by interaction types (e.g., ['synergistic', 'antagonistic'])
    severity_levels : list of str, optional
        Filter by severity levels (e.g., ['Major', 'Moderate', 'Minor'])
    data_lake_path : str, optional
        Path to data lake directory containing DDInter data

    Returns
    -------
    str
        Research log with detailed interaction analysis
    """
    from datetime import datetime

    # Initialize research log
    log = "DDInter Drug-Drug Interaction Query\n"
    log += "=" * 40 + "\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    log += "Query Parameters:\n"
    log += f"- Target drugs: {', '.join(drug_names)}\n"
    log += f"- Severity filter: {severity_levels if severity_levels else 'All levels'}\n"
    log += f"- Interaction types: {interaction_types if interaction_types else 'All types'}\n\n"

    # DDInter 2.0 records a severity level and the ATC category of each record -- nothing that says
    # 'synergistic' or 'antagonistic'. Filtering on the documented example values silently emptied
    # every result; a value that names no category is now refused, naming the ones that exist
    # (hunt 2026-09-30, uT2-pharmacology-2).
    wanted_categories = None
    if interaction_types:
        wanted_categories = {str(t).strip().lower().replace(" ", "_") for t in interaction_types}
        unmatched = sorted(wanted_categories - set(_DDINTER_CATEGORIES))
        if unmatched:
            log += (
                f"Error: interaction_types {unmatched} match no DDInter category. DDInter records only a "
                "severity level and the ATC category of each interaction record; valid values are: "
                + ", ".join(_DDINTER_CATEGORIES)
                + "\n"
            )
            return log
    wanted_levels = {str(level).strip().capitalize() for level in severity_levels} if severity_levels else None

    try:
        # Load DDInter data
        drug_info, interaction_matrix, name_mapping = _load_ddinter_data(data_lake_path)
        log += f"Successfully loaded DDInter database with {len(drug_info)} drugs\n\n"

        # Standardize drug names
        standardized_names = []
        missing_drugs = []

        for drug_name in drug_names:
            standardized = _standardize_drug_name(drug_name, name_mapping)
            if standardized:
                standardized_names.append(standardized)
            else:
                missing_drugs.append(drug_name)

        if missing_drugs:
            log += "Warning: The following drugs were not found in DDInter database:\n"
            for drug in missing_drugs:
                log += f"- {drug}\n"
            log += "\n"

        if not standardized_names:
            log += "Error: No valid drugs found in DDInter database\n"
            return log

        # Query interactions
        interactions_found = []

        for i, drug_a in enumerate(standardized_names):
            for j, drug_b in enumerate(standardized_names):
                if i >= j:  # Avoid duplicate pairs
                    continue

                if drug_a in interaction_matrix and drug_b in interaction_matrix[drug_a]:
                    interactions = interaction_matrix[drug_a][drug_b]

                    # Apply filters
                    filtered_interactions = interactions

                    if wanted_levels:
                        filtered_interactions = [
                            int_data for int_data in filtered_interactions if int_data.get("level") in wanted_levels
                        ]

                    if wanted_categories:
                        filtered_interactions = [
                            int_data
                            for int_data in filtered_interactions
                            if int_data.get("category") in wanted_categories
                        ]

                    if filtered_interactions:
                        interactions_found.append(
                            {"drug_a": drug_a, "drug_b": drug_b, "interactions": filtered_interactions}
                        )

        # Format results
        log += "Interaction Analysis Results:\n"
        log += f"Found {len(interactions_found)} drug pairs with interactions\n\n"

        if interactions_found:
            for pair in interactions_found:
                log += _format_interaction_result(
                    pair["interactions"], pair["drug_a"].title(), pair["drug_b"].title(), include_mechanisms=True
                )
                log += "\n"
        else:
            log += "No interactions found between the specified drugs with the given filters\n"

        # Summary statistics
        total_interactions = sum(len(pair["interactions"]) for pair in interactions_found)
        log += "Summary:\n"
        log += f"- Total drug pairs analyzed: {len(standardized_names) * (len(standardized_names) - 1) // 2}\n"
        log += f"- Drug pairs with interactions: {len(interactions_found)}\n"
        log += f"- Total interactions found: {total_interactions}\n"

        if interactions_found:
            severity_counts = {}
            for pair in interactions_found:
                for interaction in pair["interactions"]:
                    level = interaction.get("level", "Unknown")
                    severity_counts[level] = severity_counts.get(level, 0) + 1

            log += f"- Severity distribution: {dict(severity_counts)}\n"

    except FileNotFoundError as e:
        log += f"Error during interaction query: {str(e)}\n"
    except Exception as e:
        log += f"Error during interaction query: {str(e)}\n"

    return log


def check_drug_combination_safety(drug_list, include_mechanisms=True, include_management=True, data_lake_path=None):
    """
    Analyze safety of a drug combination for potential interactions.

    Parameters
    ----------
    drug_list : list of str
        List of drugs to analyze for combination safety
    include_mechanisms : bool, default True
        Include interaction mechanism descriptions
    include_management : bool, default True
        Include management recommendations
    data_lake_path : str, optional
        Path to data lake directory containing DDInter data

    Returns
    -------
    str
        Research log with safety analysis and recommendations
    """
    from datetime import datetime

    # Initialize research log
    log = "Drug Combination Safety Analysis\n"
    log += "=" * 35 + "\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    log += "Safety Analysis Parameters:\n"
    log += f"- Drug combination: {', '.join(drug_list)}\n"
    log += f"- Include mechanisms: {include_mechanisms}\n"
    log += f"- Include management: {include_management}\n\n"

    try:
        # Load DDInter data
        drug_info, interaction_matrix, name_mapping = _load_ddinter_data(data_lake_path)
        log += "Successfully loaded DDInter database\n\n"

        # Standardize drug names
        standardized_drugs = []
        missing_drugs = []

        for drug in drug_list:
            standardized = _standardize_drug_name(drug, name_mapping)
            if standardized:
                standardized_drugs.append(standardized)
            else:
                missing_drugs.append(drug)

        if missing_drugs:
            log += "Warning: The following drugs were not found in DDInter database:\n"
            for drug in missing_drugs:
                log += f"- {drug}\n"
            log += "\n"

        if len(standardized_drugs) < 2:
            log += "Error: At least 2 valid drugs required for combination analysis\n"
            return log

        # Analyze all pairwise interactions
        interactions_found = []
        major_interactions = 0
        moderate_interactions = 0
        minor_interactions = 0
        # 47,182 of DDInter's 222,383 rows carry Level 'Unknown'. Counting only the three graded levels
        # made a pair whose interactions are all ungraded 'Safe 100/100 -- No significant interactions
        # detected' directly above its own interaction listing; an ungraded interaction is now its own,
        # never-safe tier (hunt 2026-09-30, uT2-pharmacology-2).
        unknown_interactions = 0

        for i, drug_a in enumerate(standardized_drugs):
            for j, drug_b in enumerate(standardized_drugs):
                if i >= j:  # Avoid duplicate pairs
                    continue

                if drug_a in interaction_matrix and drug_b in interaction_matrix[drug_a]:
                    interactions = interaction_matrix[drug_a][drug_b]

                    for interaction in interactions:
                        level = interaction.get("level", "Unknown")
                        if level == "Major":
                            major_interactions += 1
                        elif level == "Moderate":
                            moderate_interactions += 1
                        elif level == "Minor":
                            minor_interactions += 1
                        else:
                            unknown_interactions += 1

                    interactions_found.append({"drug_a": drug_a, "drug_b": drug_b, "interactions": interactions})

        # Overall safety assessment
        log += "Overall Safety Assessment:\n"

        safety_score = 100
        safety_level = "Safe"

        if major_interactions > 0:
            safety_score -= major_interactions * 30
            safety_level = "High Risk"
        elif moderate_interactions > 2:
            safety_score -= moderate_interactions * 15
            safety_level = "Moderate Risk"
        elif moderate_interactions > 0:
            safety_score -= moderate_interactions * 10
            safety_level = "Low to Moderate Risk"
        elif minor_interactions > 0:
            safety_score -= minor_interactions * 5
            safety_level = "Low Risk"

        safety_score = max(0, safety_score)
        if unknown_interactions and safety_level in ("Safe", "Low Risk"):
            safety_level = "Interaction of unknown severity -- not assessed as safe"

        log += f"- Safety Level: {safety_level}\n"
        if unknown_interactions:
            log += f"- Safety Score: not assigned ({unknown_interactions} interaction(s) of unknown severity)\n"
        else:
            log += f"- Safety Score: {safety_score}/100\n"
        log += f"- Major interactions: {major_interactions}\n"
        log += f"- Moderate interactions: {moderate_interactions}\n"
        log += f"- Minor interactions: {minor_interactions}\n"
        log += f"- Interactions of unknown severity: {unknown_interactions}\n\n"

        # Detailed interaction analysis
        if interactions_found:
            log += "Detailed Interaction Analysis:\n"
            log += "-" * 30 + "\n"

            for pair in interactions_found:
                log += _format_interaction_result(
                    pair["interactions"],
                    pair["drug_a"].title(),
                    pair["drug_b"].title(),
                    include_mechanisms=include_mechanisms,
                )
                log += "\n"

        # Clinical recommendations
        log += "Clinical Recommendations:\n"
        log += "-" * 25 + "\n"

        if major_interactions > 0:
            log += "- CONTRAINDICATED: This combination contains major interactions\n"
            log += "- Consider alternative medications or consult specialist\n"
            log += "- If combination is necessary, intensive monitoring required\n"
        elif moderate_interactions > 2:
            log += "- CAUTION: Multiple moderate interactions detected\n"
            log += "- Monitor patient closely for adverse effects\n"
            log += "- Consider dose adjustments or alternative agents\n"
        elif moderate_interactions > 0:
            log += "- MONITOR: Moderate interactions present\n"
            log += "- Regular patient monitoring recommended\n"
            log += "- Be aware of potential side effects\n"
        elif unknown_interactions > 0:
            log += "- UNKNOWN SEVERITY: DDInter records an interaction but does not grade it\n"
            log += "- This is not evidence that the combination is safe\n"
            log += "- Check the severity in another source (e.g. the product labels) before combining\n"
        elif minor_interactions > 0:
            log += "- AWARENESS: Minor interactions detected\n"
            log += "- Standard monitoring sufficient\n"
            log += "- Educate patient about potential minor effects\n"
        else:
            log += "- SAFE: No significant interactions detected\n"
            log += "- Standard clinical monitoring appropriate\n"

        if include_management:
            log += "\nGeneral Management Strategies:\n"
            log += "- Separate administration times when possible\n"
            log += "- Monitor for signs of toxicity or reduced efficacy\n"
            log += "- Consider therapeutic drug monitoring if available\n"
            log += "- Educate patient about potential interaction symptoms\n"

    except Exception as e:
        log += f"Error during safety analysis: {str(e)}\n"

    return log


def analyze_interaction_mechanisms(drug_pair, detailed_analysis=True, data_lake_path=None):
    """
    Analyze interaction mechanisms between two specific drugs.

    Parameters
    ----------
    drug_pair : tuple of str
        Pair of drug names to analyze (drug1, drug2)
    detailed_analysis : bool, default True
        Include detailed mechanistic information
    data_lake_path : str, optional
        Path to data lake directory containing DDInter data

    Returns
    -------
    str
        Research log with mechanism analysis
    """
    from datetime import datetime

    # Initialize research log
    log = "Drug Interaction Mechanism Analysis\n"
    log += "=" * 37 + "\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    drug_a, drug_b = drug_pair
    log += "Mechanism Analysis Parameters:\n"
    log += f"- Drug A: {drug_a}\n"
    log += f"- Drug B: {drug_b}\n"
    log += f"- Detailed analysis: {detailed_analysis}\n\n"

    try:
        # Load DDInter data
        drug_info, interaction_matrix, name_mapping = _load_ddinter_data(data_lake_path)
        log += "Successfully loaded DDInter database\n\n"

        # Standardize drug names
        std_drug_a = _standardize_drug_name(drug_a, name_mapping)
        std_drug_b = _standardize_drug_name(drug_b, name_mapping)

        if not std_drug_a:
            log += f"Error: Drug '{drug_a}' not found in DDInter database\n"
            return log
        if not std_drug_b:
            log += f"Error: Drug '{drug_b}' not found in DDInter database\n"
            return log

        # Query interactions
        interactions = []
        if std_drug_a in interaction_matrix and std_drug_b in interaction_matrix[std_drug_a]:
            interactions = interaction_matrix[std_drug_a][std_drug_b]

        if not interactions:
            log += f"No interactions found between {drug_a} and {drug_b}\n"
            return log

        # Get drug information
        drug_a_id = name_mapping[std_drug_a]
        drug_b_id = name_mapping[std_drug_b]
        drug_a_info = drug_info.get(drug_a_id, {})
        drug_b_info = drug_info.get(drug_b_id, {})

        log += "Drug Profile Analysis:\n"
        log += "-" * 20 + "\n"
        log += f"{drug_a.title()}:\n"
        log += f"- Categories: {', '.join(drug_a_info.get('categories', ['Unknown']))}\n"
        log += f"- Total known interactions: {len(drug_a_info.get('interactions', []))}\n\n"

        log += f"{drug_b.title()}:\n"
        log += f"- Categories: {', '.join(drug_b_info.get('categories', ['Unknown']))}\n"
        log += f"- Total known interactions: {len(drug_b_info.get('interactions', []))}\n\n"

        # Analyze interaction mechanisms
        log += "Interaction Mechanism Analysis:\n"
        log += "-" * 30 + "\n"

        for i, interaction in enumerate(interactions, 1):
            level = interaction.get("level", "Unknown")
            category = interaction.get("category", "Unknown")

            log += f"Interaction {i}:\n"
            log += f"- Severity: {level}\n"
            log += f"- Category: {category.replace('_', ' ').title()}\n"

            if detailed_analysis:
                # DDInter records only a severity level and an ATC category per record; the lines below
                # are fixed guidance keyed on those two fields, so they are labelled as such rather than
                # presented as this pair's mechanism (hunt 2026-09-30, uT2-pharmacology-2).
                log += "- General guidance for this severity level (not pair-specific; DDInter records no mechanism):\n"
                if level == "Major":
                    log += "- Clinical Impact: High risk interaction requiring immediate attention\n"
                    log += "- Mechanism: Likely involves significant pharmacokinetic or pharmacodynamic effects\n"
                    log += "- Management: Avoid combination or use with extreme caution\n"
                elif level == "Moderate":
                    log += "- Clinical Impact: Moderate risk requiring monitoring\n"
                    log += "- Mechanism: May involve enzyme induction/inhibition or receptor competition\n"
                    log += "- Management: Monitor closely, consider dose adjustment\n"
                elif level == "Minor":
                    log += "- Clinical Impact: Low risk, usually manageable\n"
                    log += "- Mechanism: Minor pharmacokinetic or pharmacodynamic effects\n"
                    log += "- Management: Standard monitoring sufficient\n"
                else:
                    log += "- Clinical Impact: Not graded by DDInter -- an interaction is recorded, severity unknown\n"
                    log += "- Management: Check the severity in another source before combining\n"

                # Category-specific mechanism insights
                category_mechanisms = {
                    "alimentary_tract_metabolism": "Gastrointestinal absorption or metabolic interactions",
                    "antineoplastic": "Bone marrow suppression or tumor resistance mechanisms",
                    "blood_organs": "Hematological effects or coagulation pathway interactions",
                    "hormonal": "Endocrine system interactions or hormone receptor effects",
                    "respiratory": "Pulmonary function or bronchodilation interactions",
                    "dermatological": "Skin absorption or topical application interactions",
                    "antiparasitic": "Antimicrobial resistance or metabolic pathway interactions",
                    "various": "Multiple potential interaction pathways",
                }

                mechanism = category_mechanisms.get(category, "Unknown mechanism")
                log += f"- ATC-class context (general, not pair-specific): {mechanism}\n"

            log += "\n"

        # Summary and recommendations
        log += "Summary and Recommendations:\n"
        log += "-" * 28 + "\n"

        severity_counts = {}
        for interaction in interactions:
            level = interaction.get("level", "Unknown")
            severity_counts[level] = severity_counts.get(level, 0) + 1

        log += f"- Total interactions analyzed: {len(interactions)}\n"
        log += f"- Severity distribution: {dict(severity_counts)}\n"

        # Overall recommendation
        if any(int_data.get("level") == "Major" for int_data in interactions):
            log += "- Overall recommendation: AVOID - Major interaction detected\n"
            log += "- Consider alternative medications\n"
        elif any(int_data.get("level") == "Moderate" for int_data in interactions):
            log += "- Overall recommendation: MONITOR - Moderate interaction present\n"
            log += "- Close patient monitoring required\n"
        elif any(int_data.get("level") not in ("Major", "Moderate", "Minor") for int_data in interactions):
            log += "- Overall recommendation: UNKNOWN SEVERITY - DDInter records an interaction it does not grade\n"
            log += "- Not evidence of safety; check the severity in another source before combining\n"
        else:
            log += "- Overall recommendation: AWARENESS - Minor interactions only\n"
            log += "- Standard monitoring appropriate\n"

        if detailed_analysis:
            log += "\nGeneral Considerations (not pair-specific; DDInter records no mechanism):\n"
            log += f"- Monitor for additive effects in the {category.replace('_', ' ')} system\n"
            log += "- Consider potential for altered drug metabolism\n"
            log += "- Be aware of possible changes in drug efficacy or toxicity\n"
            log += "- Timing of administration may be important\n"

    except Exception as e:
        log += f"Error during mechanism analysis: {str(e)}\n"

    return log


def find_alternative_drugs_ddinter(target_drug, contraindicated_drugs, therapeutic_class=None, data_lake_path=None):
    """
    Find alternative drugs that don't interact with contraindicated drugs.

    Parameters
    ----------
    target_drug : str
        Drug to find alternatives for
    contraindicated_drugs : list of str
        List of drugs to avoid interactions with
    therapeutic_class : str, optional
        Limit search to specific therapeutic class
    data_lake_path : str, optional
        Path to data lake directory containing DDInter data

    Returns
    -------
    str
        Research log with alternative drug recommendations
    """
    from datetime import datetime

    # Initialize research log
    log = "Alternative Drug Finder (DDInter)\n"
    log += "=" * 32 + "\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    log += "Alternative Drug Search Parameters:\n"
    log += f"- Target drug: {target_drug}\n"
    log += f"- Contraindicated drugs: {', '.join(contraindicated_drugs)}\n"
    log += f"- Therapeutic class filter: {therapeutic_class if therapeutic_class else 'All classes'}\n\n"

    try:
        # Load DDInter data
        drug_info, interaction_matrix, name_mapping = _load_ddinter_data(data_lake_path)
        log += f"Successfully loaded DDInter database with {len(drug_info)} drugs\n\n"

        # Standardize target drug name
        std_target = _standardize_drug_name(target_drug, name_mapping)
        if not std_target:
            log += f"Error: Target drug '{target_drug}' not found in DDInter database\n"
            return log

        # Standardize contraindicated drug names
        std_contraindicated = []
        missing_contraindicated = []

        for drug in contraindicated_drugs:
            std_drug = _standardize_drug_name(drug, name_mapping)
            if std_drug:
                std_contraindicated.append(std_drug)
            else:
                missing_contraindicated.append(drug)

        if missing_contraindicated:
            log += "Warning: The following contraindicated drugs were not found:\n"
            for drug in missing_contraindicated:
                log += f"- {drug}\n"
            log += "\n"

        # Get target drug information
        target_id = name_mapping[std_target]
        target_info = drug_info.get(target_id, {})
        target_categories = target_info.get("categories", [])

        log += "Target Drug Profile:\n"
        log += f"- Drug: {target_drug}\n"
        log += f"- Categories: {', '.join(target_categories)}\n"
        log += f"- Total interactions: {len(target_info.get('interactions', []))}\n\n"

        # Find alternative drugs
        alternatives = []

        for drug_id, drug_data in drug_info.items():
            drug_name = drug_data["name"]
            drug_categories = drug_data.get("categories", [])

            # Skip the target drug itself, and the drugs to be avoided: having no interaction with
            # itself, a contraindicated drug used to rank as the top 'alternative' (hunt 2026-09-30,
            # uT2-pharmacology-2).
            if drug_id == target_id or drug_data["standardized_name"] in std_contraindicated:
                continue

            # Apply therapeutic class filter
            if therapeutic_class:
                if not any(therapeutic_class.lower() in cat.lower() for cat in drug_categories):
                    continue
            else:
                # Look for drugs in similar categories as target
                if not any(cat in target_categories for cat in drug_categories):
                    continue

            # Check if this drug interacts with any contraindicated drugs
            has_contraindicated_interactions = False
            interaction_count = 0
            major_interactions = 0
            level_counts = {}

            std_drug_name = drug_data["standardized_name"]

            for contraindicated in std_contraindicated:
                if std_drug_name in interaction_matrix and contraindicated in interaction_matrix[std_drug_name]:
                    interactions = interaction_matrix[std_drug_name][contraindicated]
                    interaction_count += len(interactions)
                    for interaction in interactions:
                        level = interaction.get("level", "Unknown")
                        level_counts[level] = level_counts.get(level, 0) + 1

                    # Check for major interactions
                    for interaction in interactions:
                        if interaction.get("level") == "Major":
                            major_interactions += 1
                            has_contraindicated_interactions = True
                            break

                    if has_contraindicated_interactions:
                        break

            # Add to alternatives if no major contraindicated interactions
            if not has_contraindicated_interactions:
                alternatives.append(
                    {
                        "name": drug_name,
                        "categories": drug_categories,
                        "interaction_count": interaction_count,
                        "level_counts": level_counts,
                        "total_interactions": len(drug_data.get("interactions", [])),
                    }
                )

        # Sort alternatives by interaction count (fewer is better)
        alternatives.sort(key=lambda x: x["interaction_count"])

        # Present results
        log += "Alternative Drug Analysis:\n"
        log += "-" * 25 + "\n"

        if alternatives:
            log += f"Found {len(alternatives)} potential alternatives:\n\n"

            # Show top 10 alternatives
            top_alternatives = alternatives[:10]

            for i, alt in enumerate(top_alternatives, 1):
                log += f"{i}. {alt['name']}\n"
                log += f"   - Categories: {', '.join(alt['categories'])}\n"
                log += f"   - Interactions with contraindicated drugs: {alt['interaction_count']}\n"
                log += f"   - Total known interactions: {alt['total_interactions']}\n"

                # Risk assessment
                if alt["interaction_count"] == 0:
                    risk = "No known interactions"
                elif alt["interaction_count"] <= 2:
                    risk = "Low interaction risk"
                elif alt["interaction_count"] <= 5:
                    risk = "Moderate interaction risk"
                else:
                    risk = "Higher interaction risk"

                log += f"   - Risk assessment: {risk}\n\n"

            if len(alternatives) > 10:
                log += f"... and {len(alternatives) - 10} additional alternatives\n\n"
        else:
            log += "No suitable alternatives found in the DDInter database\n"
            log += "Consider:\n"
            log += "- Expanding therapeutic class search criteria\n"
            log += "- Consulting additional drug databases\n"
            log += "- Seeking specialist pharmacological advice\n\n"

        # Recommendations
        log += "Clinical Recommendations:\n"
        log += "-" * 22 + "\n"

        if alternatives:
            best_alternative = alternatives[0]
            log += f"- Primary recommendation: {best_alternative['name']}\n"
            log += "- Rationale: Lowest interaction risk with contraindicated drugs\n"

            if best_alternative["interaction_count"] == 0:
                log += "- Safety profile: No known interactions with specified drugs\n"
            else:
                # It counted every non-Major interaction (Moderate and ungraded included) and called them
                # all 'minor' (hunt 2026-09-30, uT2-pharmacology-2).
                levels = ", ".join(f"{k}: {v}" for k, v in sorted(best_alternative["level_counts"].items()))
                log += (
                    f"- Safety profile: {best_alternative['interaction_count']} interaction(s) with the "
                    f"specified drugs, none Major (by DDInter severity -- {levels})\n"
                )

            log += "- Next steps: Verify therapeutic equivalence and dosing\n"
            log += "- Monitoring: Standard clinical monitoring recommended\n"
        else:
            log += "- No direct alternatives identified\n"
            log += "- Consider non-pharmacological approaches\n"
            log += "- Consult clinical pharmacist or specialist\n"
            log += "- Review patient's complete medication profile\n"

        log += "\nImportant Notes:\n"
        log += "- This analysis is based on DDInter 2.0 data only\n"
        log += "- Always verify therapeutic equivalence before substitution\n"
        log += "- Consider patient-specific factors (allergies, comorbidities)\n"
        log += "- Monitor patient response after any medication changes\n"

    except Exception as e:
        log += f"Error during alternative drug search: {str(e)}\n"

    return log


# OpenFDA Integration Functions


#: openFDA's own ceiling on ``limit`` for one request; a larger value is answered with HTTP 400.
_FDA_MAX_LIMIT = 1000

#: The adverse-event outcome filters this module accepts, and the openFDA field each one reads.
_FDA_OUTCOME_FIELDS = {
    "life_threatening": "seriousnesslifethreatening",
    "hospitalization": "seriousnesshospitalization",
    "death": "seriousnessdeath",
}


def _fda_phrase(value) -> str:
    """One double-quoted openFDA search term.

    Unquoted, a multi-word name such as 'acetylsalicylic acid' went out as the field term
    'acetylsalicylic' plus a bare 'acid' searched across every field (hunt 2026-09-30,
    uT2-pharmacology-7).
    """
    return '"' + str(value).replace('"', " ").strip() + '"'


def _fda_date(value) -> str:
    """``YYYY-MM-DD`` (or ``YYYYMMDD``) as the ``YYYYMMDD`` openFDA ranges take; ValueError otherwise."""
    digits = re.sub(r"[-/.]", "", str(value).strip())
    datetime.strptime(digits, "%Y%m%d")  # rejects anything that is not a real date
    return digits


def _normalize_recall_classes(classification) -> list[str]:
    """``['class i', 'II', '3']`` -> ``['Class I', 'Class II', 'Class III']``; ValueError on anything else."""
    canonical = {
        "i": "Class I",
        "1": "Class I",
        "ii": "Class II",
        "2": "Class II",
        "iii": "Class III",
        "3": "Class III",
    }
    out = []
    for item in classification:
        key = re.sub(r"^class\s*", "", str(item).strip().lower())
        if key not in canonical:
            raise ValueError(f"unknown recall classification {item!r}; use 'Class I', 'Class II' or 'Class III'")
        out.append(canonical[key])
    return out


class OpenFDAClient:
    """
    Client for interacting with the FDA's OpenFDA API.

    Provides comprehensive drug safety monitoring, adverse event analysis,
    and regulatory intelligence capabilities through the OpenFDA API.
    """

    BASE_URL = "https://api.fda.gov"

    def __init__(self):
        import time

        import requests

        self.requests = requests
        self.time = time
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SpatialOmicsLab-Agent/1.0 (https://spatialomicsgym.stanford.edu)"})
        self.retry_attempts = 3
        self.timeout = 30
        self.rate_limit_delay = 0.2  # 5 requests/second
        self.last_request_time = 0

    def _handle_rate_limiting(self):
        """Implement rate limiting to respect FDA API limits."""
        current_time = self.time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            self.time.sleep(self.rate_limit_delay - time_since_last)

        self.last_request_time = self.time.time()

    def _validate_response(self, response_data: dict) -> dict:
        """Validate FDA API response structure and handle variations."""
        if not isinstance(response_data, dict):
            raise ValueError("Invalid FDA API response format")

        # Check for error responses
        if "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown FDA API error")
            raise Exception(f"FDA API Error: {error_msg}")

        # Validate expected fields exist
        if "meta" not in response_data and "results" not in response_data:
            # Some endpoints return data directly without meta
            return {"results": [response_data], "meta": {"results": {"total": 1}}}

        return response_data

    def _handle_api_variations(self, endpoint: str, params: dict) -> dict:
        """Handle known FDA API endpoint variations and parameter mappings."""
        endpoint_param_mappings = {
            "drug/event": {
                "drug_name": "patient.drug.openfda.brand_name.exact",
                "generic_name": "patient.drug.openfda.generic_name.exact",
            },
            "drug/label": {"drug_name": "openfda.brand_name.exact", "generic_name": "openfda.generic_name.exact"},
            "drug/enforcement": {"drug_name": "openfda.brand_name.exact", "generic_name": "openfda.generic_name.exact"},
        }

        # Transform parameters based on endpoint
        if endpoint in endpoint_param_mappings:
            new_params = {}
            for key, value in params.items():
                if key in endpoint_param_mappings[endpoint]:
                    new_params[endpoint_param_mappings[endpoint][key]] = value
                else:
                    new_params[key] = value
            return new_params

        return params

    def _build_fda_search_params(self, endpoint: str, params: dict) -> dict:
        """Build FDA API search parameters from input parameters.

        Every filter the public functions accept goes into the openFDA ``search`` itself. They used to
        stop here at the drug name: date_range and classification were echoed in the output as applied
        filters but never sent, and Class II/III recalls were listed under 'Classification filter:
        Class I' (hunt 2026-09-30, uT2-pharmacology-4). Labels and recalls search the generic as well
        as the brand name, so an INN finds branded products (uT2-pharmacology-7).
        """
        fda_params = {}
        terms = []

        # Handle drug name searches
        if "drug_name" in params:
            name = _fda_phrase(params["drug_name"])
            if endpoint == "drug/event":
                # For adverse events, search in medicinalproduct field
                terms.append(f"patient.drug.medicinalproduct:{name}")
            else:
                # For drug labels and enforcement/recalls, either name
                terms.append(f"(openfda.generic_name:{name} OR openfda.brand_name:{name})")

        date_field = {"drug/event": "receivedate", "drug/enforcement": "recall_initiation_date"}.get(endpoint)
        if params.get("date_range") and date_field:
            start, end = params["date_range"]
            terms.append(f"{date_field}:[{_fda_date(start)} TO {_fda_date(end)}]")

        if params.get("classification"):
            classes = _normalize_recall_classes(params["classification"])
            terms.append("(" + " OR ".join(f"classification:{_fda_phrase(c)}" for c in classes) + ")")

        severity = {str(v).strip().lower() for v in params.get("severity_filter") or []}
        unknown = severity - {"serious", "non_serious"}
        if unknown:
            raise ValueError(f"unknown severity_filter {sorted(unknown)}; use 'serious' and/or 'non_serious'")
        if severity == {"serious"}:
            terms.append("serious:1")
        elif severity == {"non_serious"}:
            terms.append("serious:2")

        outcomes = [str(v).strip().lower() for v in params.get("outcome_filter") or []]
        unknown = sorted(set(outcomes) - set(_FDA_OUTCOME_FIELDS))
        if unknown:
            raise ValueError(f"unknown outcome_filter {unknown}; use any of {sorted(_FDA_OUTCOME_FIELDS)}")
        if outcomes:
            terms.append("(" + " OR ".join(f"{_FDA_OUTCOME_FIELDS[o]}:1" for o in dict.fromkeys(outcomes)) + ")")

        if terms:
            fda_params["search"] = " AND ".join(terms)

        # Handle other parameters
        for key in ("limit", "skip"):
            if key in params:
                fda_params[key] = params[key]

        return fda_params

    def _make_request(self, endpoint: str, params: dict) -> dict:
        """Make API request with retry logic and error handling."""
        self._handle_rate_limiting()

        # Build FDA API search parameters
        fda_params = self._build_fda_search_params(endpoint, params)

        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(f"{self.BASE_URL}/{endpoint}.json", params=fda_params, timeout=self.timeout)

                if response.status_code == 404:
                    return {
                        "results": [],
                        "meta": {"results": {"total": 0}},
                        "message": "No results found for the specified query",
                    }

                response.raise_for_status()

                # Validate and normalize response
                data = self._validate_response(response.json())

                return data

            except self.requests.exceptions.Timeout:
                if attempt == self.retry_attempts - 1:
                    raise Exception("FDA API request timed out after multiple attempts") from None
                self.time.sleep(2**attempt)  # Exponential backoff

            except self.requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limiting - wait and retry
                    if attempt < self.retry_attempts - 1:
                        self.time.sleep(5 * (attempt + 1))
                    continue
                else:
                    detail = ""
                    try:
                        detail = ": " + str(e.response.json()["error"]["message"])
                    except Exception:
                        pass
                    raise Exception(f"FDA API HTTP Error {e.response.status_code}{detail}") from e

            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise Exception(f"FDA API request failed: {str(e)}") from e
                self.time.sleep(2**attempt)

        # Every attempt was rate-limited. This returned {}, which every caller read as 'no reports
        # found' (hunt 2026-09-30, uT2-pharmacology-3).
        raise Exception(f"FDA API rate limit (HTTP 429) persisted after {self.retry_attempts} attempts")

    def query_adverse_events(
        self,
        drug_name: str,
        limit: int = 100,
        date_range: tuple[str, str] | None = None,
        severity_filter: list[str] | None = None,
        outcome_filter: list[str] | None = None,
    ) -> dict:
        """Query adverse events. A failed request comes back with an ``error`` key -- check it."""
        endpoint = "drug/event"
        params = {
            "drug_name": drug_name,
            "limit": limit,
            "date_range": date_range,
            "severity_filter": severity_filter,
            "outcome_filter": outcome_filter,
        }

        try:
            data = self._make_request(endpoint, params)

            # Add FDA disclaimer to results
            data["disclaimer"] = (
                "FDA Disclaimer: These data do not establish causation. "
                "Reports are voluntary and subject to reporting bias. "
                "Data should not be used for regulatory decision-making."
            )

            return data

        except Exception as e:
            return {
                "results": [],
                "meta": {"results": {"total": 0}},
                "error": str(e),
                "disclaimer": (
                    "FDA Disclaimer: These data do not establish causation. "
                    "Reports are voluntary and subject to reporting bias."
                ),
            }

    def query_drug_labels(self, drug_name: str, sections: list[str] | None = None) -> dict:
        """Query FDA drug label information."""
        endpoint = "drug/label"
        params = {"drug_name": drug_name, "limit": 50}

        return self._make_request(endpoint, params)

    def query_drug_recalls(
        self, drug_name: str, classification: list[str] | None = None, date_range: tuple[str, str] | None = None
    ) -> dict:
        """Query FDA drug recall and enforcement information."""
        endpoint = "drug/enforcement"
        params = {"drug_name": drug_name, "limit": 100, "classification": classification, "date_range": date_range}

        return self._make_request(endpoint, params)


# Helper Functions for OpenFDA Data Processing


def _standardize_drug_name_fda(drug_name: str) -> str:
    """Standardize drug names for FDA API queries."""
    # Handle None/empty values
    if not drug_name:
        return ""

    # Remove common suffixes
    suffixes = ["sodium", "hydrochloride", "sulfate", "phosphate", "acetate", "citrate"]

    # Clean and standardize
    name = drug_name.strip().lower()

    for suffix in suffixes:
        if name.endswith(f" {suffix}"):
            name = name[: -len(f" {suffix}")]

    return name


def _apply_fda_filters(response_data: dict, filters: dict) -> dict:
    """Apply post-query filtering to FDA responses."""
    if not response_data.get("results"):
        return response_data

    filtered_results = []

    for result in response_data["results"]:
        include = True

        # Apply severity filter
        if filters.get("severity_filter") or filters.get("severity"):
            severity_list = filters.get("severity_filter", filters.get("severity", []))
            if "serious" in severity_list:
                # For serious filter, only include if serious == '1'
                if result.get("serious") != "1":
                    include = False
            elif "non_serious" in severity_list:
                # For non-serious filter, only include if serious != '1'
                if result.get("serious") == "1":
                    include = False

        # Apply outcome filter
        if (filters.get("outcome_filter") or filters.get("outcome")) and include:
            outcome_list = filters.get("outcome_filter", filters.get("outcome", []))
            if "life_threatening" in outcome_list:
                # Check if the result has life threatening outcome
                if result.get("seriousnesslifethreatening") != "1":
                    include = False
            elif "hospitalization" in outcome_list:
                # Check if the result has hospitalization outcome
                if result.get("seriousnesshospitalization") != "1":
                    include = False
            elif "death" in outcome_list:
                # Check if the result has death outcome
                if result.get("seriousnessdeath") != "1":
                    include = False

        # Apply classification filter (for recalls)
        if filters.get("classification") and include:
            classification_list = filters.get("classification", [])
            result_class = result.get("classification", "")
            if result_class not in classification_list:
                include = False

        if include:
            filtered_results.append(result)

    response_data["results"] = filtered_results
    response_data["meta"]["results"]["total"] = len(filtered_results)

    return response_data


def _extract_fda_safety_signals(response_list: list[dict]) -> dict:
    """Extract safety signals from adverse event data."""
    drug_signals = {}
    reaction_patterns = {}
    temporal_patterns = {}

    for response in response_list:
        if not response.get("results"):
            continue

        for result in response["results"]:
            # Extract drug information
            drugs = result.get("patient", {}).get("drug", [])
            for drug in drugs:
                # Use the existing standardization function
                drug_name = _standardize_drug_name_fda(drug.get("medicinalproduct", ""))
                if drug_name:
                    if drug_name not in drug_signals:
                        drug_signals[drug_name] = {"total_reports": 0, "serious_reports": 0, "common_reactions": []}

                    drug_signals[drug_name]["total_reports"] += 1
                    if result.get("serious") == "1":
                        drug_signals[drug_name]["serious_reports"] += 1

            # Extract reaction patterns
            reactions = result.get("patient", {}).get("reaction", [])
            for reaction in reactions:
                reaction_name = reaction.get("reactionmeddrapt", "")
                if reaction_name:
                    if reaction_name not in reaction_patterns:
                        reaction_patterns[reaction_name] = {
                            "count": 0,
                            "severity_counts": {"serious": 0, "non_serious": 0},
                        }

                    reaction_patterns[reaction_name]["count"] += 1

                    # Count severity
                    if result.get("serious") == "1":
                        reaction_patterns[reaction_name]["severity_counts"]["serious"] += 1
                    else:
                        reaction_patterns[reaction_name]["severity_counts"]["non_serious"] += 1

            # Extract temporal patterns
            receipt_date = result.get("receiptdate")
            if receipt_date and len(receipt_date) >= 6:
                year_month = receipt_date[:6]  # YYYYMM
                if year_month not in temporal_patterns:
                    temporal_patterns[year_month] = {"count": 0, "serious_count": 0}

                temporal_patterns[year_month]["count"] += 1
                if result.get("serious") == "1":
                    temporal_patterns[year_month]["serious_count"] += 1

    # Build common reactions for each drug based on actual data
    for drug_name in drug_signals:
        # Find reactions that occurred with this specific drug
        drug_reactions = {}

        for response in response_list:
            if not response.get("results"):
                continue

            for result in response["results"]:
                drugs = result.get("patient", {}).get("drug", [])
                has_this_drug = any(
                    _standardize_drug_name_fda(drug.get("medicinalproduct", "")) == drug_name for drug in drugs
                )

                if has_this_drug:
                    reactions = result.get("patient", {}).get("reaction", [])
                    for reaction in reactions:
                        reaction_name = reaction.get("reactionmeddrapt", "")
                        if reaction_name:
                            if reaction_name not in drug_reactions:
                                drug_reactions[reaction_name] = 0
                            drug_reactions[reaction_name] += 1

        # Get top 3 reactions for this drug
        top_reactions = sorted(drug_reactions.items(), key=lambda x: x[1], reverse=True)[:3]
        drug_signals[drug_name]["common_reactions"] = [r[0] for r in top_reactions]

    return {
        "drug_signals": drug_signals,
        "reaction_patterns": reaction_patterns,
        "temporal_patterns": temporal_patterns,
    }


def _generate_fda_statistics(response_data: dict) -> dict:
    """Generate summary statistics from FDA responses.

    ``total_reports`` is openFDA's count of every matching report (``meta.results.total``); the other
    counts are over the ``sample_size`` reports actually returned. Outcomes are the report-level
    ``seriousness*`` flags: the ``patient.patient*`` fields this read do not exist in openFDA, so
    hospitalisation and life-threatening counts were always 0, and 'Total Reports' was the page size
    (hunt 2026-09-30, uT2-pharmacology-5).
    """
    stats = {
        "total_reports": 0,
        "sample_size": 0,
        "serious_reports": 0,
        "death_reports": 0,
        "life_threatening_reports": 0,
        "hospitalization_reports": 0,
        "top_reactions": [],
        "temporal_pattern": {},
    }

    if not response_data.get("results"):
        return stats

    reaction_counts = {}

    for result in response_data["results"]:
        stats["sample_size"] += 1

        # Count serious reports
        if result.get("serious") == "1":
            stats["serious_reports"] += 1

        # Count specific outcomes
        outcomes = result.get("patient", {}).get("reaction", [])
        for outcome in outcomes:
            outcome_name = outcome.get("reactionmeddrapt", "Unknown")
            reaction_counts[outcome_name] = reaction_counts.get(outcome_name, 0) + 1

        # Count deaths and other serious outcomes
        if result.get("seriousnessdeath") == "1":
            stats["death_reports"] += 1

        if result.get("seriousnesslifethreatening") == "1":
            stats["life_threatening_reports"] += 1

        if result.get("seriousnesshospitalization") == "1":
            stats["hospitalization_reports"] += 1

    total = ((response_data.get("meta") or {}).get("results") or {}).get("total")
    stats["total_reports"] = (
        int(total) if isinstance(total, (int, float)) and total >= stats["sample_size"] else stats["sample_size"]
    )

    # Top reactions
    sorted_reactions = sorted(reaction_counts.items(), key=lambda x: x[1], reverse=True)
    stats["top_reactions"] = sorted_reactions[:10]

    # Report distribution (over the sample)
    n = stats["sample_size"]
    stats["report_distribution"] = {
        "serious_percentage": (stats["serious_reports"] / n * 100) if n > 0 else 0,
        "non_serious_percentage": ((n - stats["serious_reports"]) / n * 100) if n > 0 else 0,
        "death_percentage": (stats["death_reports"] / n * 100) if n > 0 else 0,
    }

    return stats


def _format_adverse_event_summary(response_data: dict, drug_name: str, include_details: bool = True) -> str:
    """Format adverse event data into readable summary.

    A failed request is reported as a failure. It used to read only ``results`` and so turned a network
    error, an HTTP 4xx/5xx or exhausted 429 retries into 'No adverse events found for X in the FDA
    database' (hunt 2026-09-30, uT2-pharmacology-3).
    """
    if response_data.get("error"):
        return f"Error: openFDA request failed for {drug_name}: {response_data['error']}"
    if not response_data.get("results"):
        return f"No adverse events found for {drug_name} in the FDA database."

    stats = _generate_fda_statistics(response_data)
    n = stats["sample_size"]

    summary = "Adverse Event Summary\n"
    summary += "=" * 21 + "\n"
    summary += f"Drug: {drug_name}\n"
    summary += f"Total Reports: {stats['total_reports']:,} (all openFDA reports matching the query)\n\n"

    if n > 0:
        if n < stats["total_reports"]:
            summary += f"Summary Statistics (over the {n:,} reports returned, a sample of the total):\n"
        else:
            summary += "Summary Statistics:\n"
        summary += f"- Serious Reports: {stats['serious_reports']:,} ({stats['serious_reports'] / n * 100:.1f}%)\n"

        if stats["death_reports"] > 0:
            summary += f"- Deaths: {stats['death_reports']:,} ({stats['death_reports'] / n * 100:.1f}%)\n"

        if stats["life_threatening_reports"] > 0:
            summary += f"- Life-threatening: {stats['life_threatening_reports']:,} ({stats['life_threatening_reports'] / n * 100:.1f}%)\n"

        if stats["hospitalization_reports"] > 0:
            summary += f"- Hospitalizations: {stats['hospitalization_reports']:,} ({stats['hospitalization_reports'] / n * 100:.1f}%)\n"

        if stats["top_reactions"]:
            summary += "\nCommon Reactions:\n"
            for i, (reaction, count) in enumerate(stats["top_reactions"][:5], 1):
                summary += f"{i}. {reaction} ({count:,} reports)\n"

    # Add FDA disclaimer
    summary += "\n" + response_data.get("disclaimer", "")

    return summary


def _format_drug_label_summary(response_data: dict, drug_name: str, sections: list[str] | None = None) -> str:
    """Format drug label information into readable summary."""
    if not response_data.get("results"):
        return f"No drug label information found for {drug_name} in the FDA database."

    result = response_data["results"][0]  # Use first result

    summary = "OpenFDA Drug Label Information\n"
    summary += "=" * 29 + "\n"
    summary += f"Drug: {drug_name}\n"

    # Extract key information
    if "effective_time" in result:
        summary += f"Effective Date: {result['effective_time']}\n"

    if "openfda" in result:
        openfda = result["openfda"]
        if "brand_name" in openfda:
            summary += f"Brand Name: {', '.join(openfda['brand_name'])}\n"
        if "generic_name" in openfda:
            summary += f"Generic Name: {', '.join(openfda['generic_name'])}\n"
        if "manufacturer_name" in openfda:
            summary += f"Manufacturer: {', '.join(openfda['manufacturer_name'])}\n"

    summary += "\n"

    # Display specific sections
    section_mapping = {
        "indications_and_usage": "Indications and Usage",
        "contraindications": "Contraindications",
        "warnings": "Warnings",
        "dosage_and_administration": "Dosage and Administration",
        "adverse_reactions": "Adverse Reactions",
        "clinical_pharmacology": "Clinical Pharmacology",
    }

    sections_to_show = sections if sections else section_mapping.keys()

    for section_key in sections_to_show:
        if section_key in result:
            section_title = section_mapping.get(section_key, section_key.title())
            summary += f"{section_title}:\n"

            content = result[section_key]
            if isinstance(content, list):
                content = " ".join(content)

            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."

            summary += f"{content}\n\n"

    return summary


def _format_recall_summary(response_data: dict, drug_name: str, include_details: bool = True) -> str:
    """Format recall information into structured output."""
    if not response_data.get("results"):
        return f"No drug recalls found for {drug_name} in the FDA database."

    summary = "OpenFDA Drug Recall Information\n"
    summary += "=" * 31 + "\n"
    summary += f"Drug: {drug_name}\n"
    summary += f"Total recalls found: {len(response_data['results'])}\n\n"

    if include_details:
        summary += "Recall Details:\n"

        for i, recall in enumerate(response_data["results"][:5], 1):  # Show top 5
            summary += f"{i}. Recall Number: {recall.get('recall_number', 'N/A')}\n"
            summary += f"   - Product: {recall.get('product_description', 'N/A')}\n"
            summary += f"   - Classification: {recall.get('classification', 'N/A')}\n"
            summary += f"   - Reason: {recall.get('reason_for_recall', 'N/A')}\n"
            summary += f"   - Date: {recall.get('recall_initiation_date', 'N/A')}\n"
            summary += f"   - Status: {recall.get('status', 'N/A')}\n"
            summary += f"   - Distribution: {recall.get('distribution_pattern', 'N/A')}\n\n"

        if len(response_data["results"]) > 5:
            summary += f"... and {len(response_data['results']) - 5} additional recalls\n"

    return summary


def _format_safety_signal_summary(
    signals_data: dict,
    drug_list: list[str],
    comparison_period: tuple[str, str] | None = None,
    signal_threshold: float = 2.0,
    per_drug_responses: dict | None = None,
) -> str:
    """Format safety signal analysis results.

    Per-drug lines come from that drug's own query when ``per_drug_responses`` is given. They used to be
    looked up by the caller's spelling in a table keyed by the lower-cased, salt-stripped
    ``medicinalproduct``, so 'Aspirin' printed 'No data found' directly above a cross-drug section built
    from its own reports (hunt 2026-09-30, uT2-pharmacology-6). No trend or disproportionality analysis
    is run, so none is described: the 'Trend Analysis ... seasonal variations and reporting delays'
    lines described work that never happened (uT2-pharmacology-4).
    """
    summary = "OpenFDA Safety Signal Analysis\n"
    summary += "=" * 29 + "\n"
    summary += f"Drugs analyzed: {drug_list}\n"

    # Add comparison period and threshold info
    if comparison_period:
        summary += f"Reports restricted to those received {comparison_period[0]} to {comparison_period[1]}\n"
    summary += (
        "Counts below are of the reports returned per drug; no disproportionality statistic (e.g. PRR) is computed"
    )
    if signal_threshold != 2.0:
        summary += f", so signal_threshold={signal_threshold} was not applied"
    summary += ".\n\n"

    if not signals_data:
        summary += "No safety signals detected.\n"
        return summary

    summary += "Signal Detection Results:\n"

    # Handle the actual data structure from _extract_fda_safety_signals
    drug_signals = signals_data.get("drug_signals", {})
    reaction_patterns = signals_data.get("reaction_patterns", {})

    # Display drug-specific signals
    for i, drug_name in enumerate(drug_list, 1):
        summary += f"{i}. {drug_name.title()}\n"
        response = (per_drug_responses or {}).get(drug_name)
        if response is not None and response.get("error"):
            summary += f"   - Error: openFDA request failed: {response['error']}\n\n"
            continue
        if response is not None:
            stats = _generate_fda_statistics(response)
            if stats["sample_size"]:
                summary += (
                    f"   - Total reports: {stats['total_reports']:,} (sample analysed: {stats['sample_size']:,})\n"
                )
                summary += f"   - Serious reports (in sample): {stats['serious_reports']:,}\n"
                if stats["top_reactions"]:
                    summary += f"   - Common reactions: {', '.join(r for r, _ in stats['top_reactions'][:3])}\n"
                summary += "\n"
                continue
        else:
            drug_data = drug_signals.get(_standardize_drug_name_fda(drug_name), {})
            if drug_data:
                summary += f"   - Total reports: {drug_data['total_reports']:,}\n"
                summary += f"   - Serious reports: {drug_data['serious_reports']:,}\n"
                if drug_data.get("common_reactions"):
                    summary += f"   - Common reactions: {', '.join(drug_data['common_reactions'])}\n"
                summary += "\n"
                continue
        summary += "   - No data found\n\n"

    # Display cross-drug reaction patterns
    if reaction_patterns:
        summary += "Cross-drug Analysis:\n"
        sorted_reactions = sorted(reaction_patterns.items(), key=lambda x: x[1]["count"], reverse=True)

        for reaction, data in sorted_reactions[:5]:  # Show top 5 reactions
            summary += f"- {reaction}: {data['count']:,} reports\n"
            if data["severity_counts"]["serious"] > 0:
                summary += f"  * Serious: {data['severity_counts']['serious']:,}\n"

    return summary


# Main OpenFDA Integration Functions


def query_fda_adverse_events(
    drug_name: str,
    date_range: tuple[str, str] | None = None,
    severity_filter: list[str] | None = None,
    outcome_filter: list[str] | None = None,
    limit: int = 100,
) -> str:
    """
    Query FDA adverse event reports for specific drugs.

    Args:
        drug_name: Name of the drug to query
        date_range: Optional date range as (start_date, end_date) in YYYY-MM-DD format
        severity_filter: Optional filter by severity levels ["serious", "non_serious"]
        outcome_filter: Optional filter by outcomes ["life_threatening", "hospitalization", "death"]
        limit: Maximum number of results to return (1-1000, openFDA's per-request ceiling)

    Returns:
        Formatted string with adverse event analysis
    """
    try:
        # Validate input
        if not drug_name or not drug_name.strip():
            return "Error: Drug name cannot be empty"
        if not isinstance(limit, int) or not 1 <= limit <= _FDA_MAX_LIMIT:
            return f"Error: limit must be an integer from 1 to {_FDA_MAX_LIMIT} (openFDA's per-request maximum)"

        client = OpenFDAClient()

        # Standardize drug name
        standardized_name = _standardize_drug_name_fda(drug_name)
        if not standardized_name:
            return f"Error: Unable to standardize drug name '{drug_name}'"

        # The filters are part of the openFDA query (hunt 2026-09-30, uT2-pharmacology-4), so the
        # matching total openFDA reports is the filtered total.
        try:
            client._build_fda_search_params(
                "drug/event",
                {"date_range": date_range, "severity_filter": severity_filter, "outcome_filter": outcome_filter},
            )
        except (TypeError, ValueError) as e:
            return f"Error: invalid filter for openFDA: {e}"
        response = client.query_adverse_events(
            standardized_name,
            limit=limit,
            date_range=date_range,
            severity_filter=severity_filter,
            outcome_filter=outcome_filter,
        )

        # Format results with main function title
        formatted_result = _format_adverse_event_summary(response, drug_name, include_details=True)
        if formatted_result.startswith("Error"):
            return formatted_result

        # Replace title for main function
        if formatted_result.startswith("Adverse Event Summary"):
            formatted_result = formatted_result.replace(
                "Adverse Event Summary\n" + "=" * 21, "OpenFDA Adverse Event Query Results\n" + "=" * 35, 1
            )

        # Add filter and date range info if specified
        lines = formatted_result.split("\n")
        insert_index = -1

        # Find insertion point (after drug name)
        for i, line in enumerate(lines):
            if line.startswith("Drug: "):
                insert_index = i + 1
                break

        if insert_index >= 0:
            # Add date range info
            if date_range:
                lines.insert(insert_index, f"Date range (receivedate): {date_range[0]} to {date_range[1]}")
                insert_index += 1

            # Add severity filter info
            if severity_filter:
                lines.insert(insert_index, f"Severity filter: {severity_filter}")
                insert_index += 1

            # Add outcome filter info
            if outcome_filter:
                lines.insert(insert_index, f"Outcome filter: {outcome_filter}")
                insert_index += 1

        formatted_result = "\n".join(lines)

        return formatted_result

    except Exception as e:
        return f"Error querying FDA adverse events for {drug_name}: {str(e)}"


def get_fda_drug_label_info(drug_name: str, sections: list[str] | None = None) -> str:
    """
    Retrieve FDA drug label information.

    Args:
        drug_name: Name of the drug to query
        sections: Optional list of specific sections to retrieve
                 ["indications_and_usage", "contraindications", "warnings", "dosage_and_administration"]

    Returns:
        Formatted string with drug label information
    """
    try:
        # Validate input
        if not drug_name or not drug_name.strip():
            return "Error: Drug name cannot be empty"

        client = OpenFDAClient()

        # Standardize drug name
        standardized_name = _standardize_drug_name_fda(drug_name)
        if not standardized_name:
            return f"Error: Unable to standardize drug name '{drug_name}'"

        # Query drug labels
        response = client.query_drug_labels(standardized_name, sections=sections)

        # Check if we got results
        if not response.get("results"):
            return f"No label information found for drug: {drug_name}"

        # Format results
        return _format_drug_label_summary(response, drug_name, sections=sections)

    except Exception as e:
        return f"Error retrieving FDA drug label for {drug_name}: {str(e)}"


def check_fda_drug_recalls(
    drug_name: str, classification: list[str] | None = None, date_range: tuple[str, str] | None = None
) -> str:
    """
    Check for FDA drug recalls and enforcement actions.

    Args:
        drug_name: Name of the drug to check
        classification: Optional filter by recall class ["Class I", "Class II", "Class III"]
        date_range: Optional date range for recalls as (start_date, end_date), YYYY-MM-DD,
            matched against the recall initiation date

    Returns:
        Formatted string with recall information
    """
    try:
        # Validate input
        if not drug_name or not drug_name.strip():
            return "Error: Drug name cannot be empty"

        client = OpenFDAClient()

        # Standardize drug name
        standardized_name = _standardize_drug_name_fda(drug_name)
        if not standardized_name:
            return f"Error: Unable to standardize drug name '{drug_name}'"

        try:
            client._build_fda_search_params(
                "drug/enforcement", {"classification": classification, "date_range": date_range}
            )
        except (TypeError, ValueError) as e:
            return f"Error: invalid filter for openFDA: {e}"

        # Query drug recalls -- the classification and date range are part of the query now
        # (hunt 2026-09-30, uT2-pharmacology-4).
        response = client.query_drug_recalls(standardized_name, classification=classification, date_range=date_range)

        # Format results with filter information
        formatted_result = _format_recall_summary(response, drug_name, include_details=True)

        # Add filter information to the output
        if classification:
            formatted_result = formatted_result.replace(
                f"Drug: {drug_name}\n",
                f"Drug: {drug_name}\nClassification filter: {', '.join(_normalize_recall_classes(classification))}\n",
            )

        if date_range:
            formatted_result = formatted_result.replace(
                f"Drug: {drug_name}\n",
                f"Drug: {drug_name}\nDate range (recall initiation): {date_range[0]} to {date_range[1]}\n",
            )

        return formatted_result

    except Exception as e:
        return f"Error checking FDA drug recalls for {drug_name}: {str(e)}"


def analyze_fda_safety_signals(
    drug_list: list[str], comparison_period: tuple[str, str] | None = None, signal_threshold: float = 2.0
) -> str:
    """
    Analyze safety signals across multiple drugs.

    Args:
        drug_list: List of drug names to analyze
        comparison_period: Optional (start_date, end_date), YYYY-MM-DD; only reports received in this
            period are analysed
        signal_threshold: Accepted for compatibility; no disproportionality statistic is computed, so
            it is not applied (the output says so)

    Returns:
        Formatted string with safety signal analysis
    """
    try:
        # Validate input parameters
        if not drug_list:
            return "Error: At least one drug must be provided for analysis"

        if len(drug_list) < 2:
            return "Error: At least 2 drugs required for comparative safety signal analysis"

        # Validate drug names
        valid_drugs = [drug.strip() for drug in drug_list if drug and drug.strip()]
        if not valid_drugs:
            return "Error: No valid drug names provided"

        client = OpenFDAClient()
        if comparison_period:
            try:
                client._build_fda_search_params("drug/event", {"date_range": comparison_period})
            except (TypeError, ValueError) as e:
                return f"Error: invalid comparison_period for openFDA: {e}"

        # Collect data for all drugs. A failed request is kept and reported per drug: it used to be
        # dropped, so a network error read as 'No adverse event data found' (hunt 2026-09-30,
        # uT2-pharmacology-3).
        all_responses = []
        per_drug = {}

        for drug in valid_drugs:
            standardized_name = _standardize_drug_name_fda(drug)
            if standardized_name:  # Only query if standardization worked
                response = client.query_adverse_events(standardized_name, limit=200, date_range=comparison_period)
                per_drug[drug] = response

                if response.get("results"):
                    all_responses.append(response)

        errors = {drug: r["error"] for drug, r in per_drug.items() if r.get("error")}
        if errors and len(errors) == len(per_drug):
            return "Error: openFDA request failed for every drug: " + "; ".join(f"{d}: {e}" for d, e in errors.items())

        # Check if we got any data
        if not all_responses:
            if errors:
                return "Error: no reports retrieved; openFDA request failed for: " + "; ".join(
                    f"{d}: {e}" for d, e in errors.items()
                )
            return "No adverse event reports found in openFDA for any of the provided drugs"

        # Extract safety signals
        signals = _extract_fda_safety_signals(all_responses)

        # Format results with comparison period and threshold info
        return _format_safety_signal_summary(
            signals, valid_drugs, comparison_period, signal_threshold, per_drug_responses=per_drug
        )

    except Exception as e:
        return f"Error analyzing FDA safety signals: {str(e)}"
