def _claim_outputs(*names):
    """Absolute paths for ``names`` that no earlier result occupies, created empty so no other run takes them.

    These tools wrote fixed names (``ras_simulation_results.csv`` and friends) into the working
    directory, which every chat of an account shares, so a second run replaced the first run's
    results under a log that still pointed at them (hunt 2026-09-30, uT6-literature-36). A name that
    is already taken gets one shared time-and-random suffix instead; the log reports what was used.
    """
    import os
    import time
    import uuid

    stamp = ""
    while True:
        paths = []
        for name in names:
            root, ext = os.path.splitext(os.path.abspath(name))
            paths.append(f"{root}{stamp}{ext}")
        claimed = []
        try:
            for path in paths:
                os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
                claimed.append(path)
        except FileExistsError:
            for path in claimed:
                os.unlink(path)
            stamp = f"_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
            continue
        return paths


def _write_atomically(path, write):
    """Run ``write(tmp)`` and move the result onto ``path``, so a reader never sees half a file."""
    import os

    partial = f"{path}.partial"
    try:
        write(partial)
        os.replace(partial, path)
    except BaseException:
        for leftover in (partial, path):
            if os.path.exists(leftover):
                os.unlink(leftover)
        raise


#: ChatNT's pipeline is code from its Hugging Face repo (``trust_remote_code=True``), run inside this
#: process. Unpinned, that is whatever the repo holds on the day of the call, so the call runs only
#: once an operator names the commit they reviewed (hunt 2026-09-30, uT6-literature-38).
_CHATNT_REVISION_ENV = "SOG_CHATNT_REVISION"


def query_chatnt(question, sequence, device=-1):
    """
    Call ChatNT to answer a question about a DNA sequence.

    The model's pipeline is remote code, so it is pinned: set ``SOG_CHATNT_REVISION`` to the commit
    of ``InstaDeepAI/ChatNT`` you have reviewed. Without it the call refuses rather than run
    whatever the repository holds today.

    Parameters:
    -----------
    question : str
        Question to ask about the DNA sequence
    sequence : str
        DNA sequence to analyze
    device : int, optional
        Device to use for the ChatNT model. Default is -1 (CPU).

    Returns:
    --------
    str
        Answer to the question
    """
    import os

    revision = (os.environ.get(_CHATNT_REVISION_ENV) or "").strip()
    if not revision:
        raise RuntimeError(
            "query_chatnt runs the Python code stored in the InstaDeepAI/ChatNT Hugging Face repository "
            "(trust_remote_code=True) inside this process, so it needs a pinned, reviewed revision. Set "
            f"{_CHATNT_REVISION_ENV} to that commit hash to allow the call."
        )

    from transformers import pipeline

    pipe = pipeline(model="InstaDeepAI/ChatNT", trust_remote_code=True, revision=revision, device=device)

    # Define custom inputs (note that the number of <DNA> token in the english sequence must be equal to len(dna_sequences))
    english_sequence = f"{question} <DNA> ?"
    dna_sequences = [sequence]

    # Generate sequence
    generated_english_sequence = pipe(inputs={"english_sequence": english_sequence, "dna_sequences": dna_sequences})

    # The contract is a string; the pipeline's raw output was handed back as-is.
    if isinstance(generated_english_sequence, str):
        return generated_english_sequence
    return str(generated_english_sequence)


def perform_flux_balance_analysis(model_file, constraints=None, objective_reaction=None, output_file="fba_results.csv"):
    """
    Perform Flux Balance Analysis (FBA) on a genome-scale metabolic network model.

    FBA is a computational technique that predicts metabolic flux distributions
    by formulating and solving a linear optimization problem.

    Parameters:
    -----------
    model_file : str
        Path to the metabolic model file (SBML or JSON format)
    constraints : dict, optional
        Dictionary of reaction constraints where keys are reaction IDs and
        values are tuples of (lower_bound, upper_bound)
    objective_reaction : str, optional
        Reaction ID to use as the objective function (e.g., biomass reaction)
        If None, uses the model's default objective
    output_file : str, optional
        File name to save the flux distribution results

    Returns:
    --------
    str
        Research log summarizing the FBA process and results
    """

    import cobra
    import pandas as pd

    # Initialize research log
    log = "# Flux Balance Analysis (FBA) Research Log\n\n"

    # Step 1: Load the metabolic network model
    log += "## Step 1: Loading metabolic model\n"
    try:
        if model_file.endswith(".xml") or model_file.endswith(".sbml"):
            model = cobra.io.read_sbml_model(model_file)
        elif model_file.endswith(".json"):
            model = cobra.io.load_json_model(model_file)
        else:
            model = cobra.io.load_model(model_file)
        log += f"- Successfully loaded model from {model_file}\n"
        log += f"- Model contains {len(model.reactions)} reactions and {len(model.metabolites)} metabolites\n\n"
    except Exception as e:
        log += f"- Error loading model: {str(e)}\n"
        return log

    # Step 2: Set constraints
    log += "## Step 2: Setting constraints\n"
    if constraints:
        log += "- Applied the following constraints:\n"
        for reaction_id, (lb, ub) in constraints.items():
            try:
                reaction = model.reactions.get_by_id(reaction_id)
                reaction.bounds = (lb, ub)
                log += f"  * {reaction_id}: lower_bound={lb}, upper_bound={ub}\n"
            except Exception as e:
                log += f"  * Error setting constraint for {reaction_id}: {str(e)}\n"
    else:
        log += "- No additional constraints specified, using model defaults\n"
    log += "\n"

    # Step 3: Set objective function
    log += "## Step 3: Setting objective function\n"
    if objective_reaction:
        try:
            model.objective = objective_reaction
            log += f"- Set objective function to maximize {objective_reaction}\n\n"
        except Exception as e:
            log += f"- Error setting objective function: {str(e)}\n"
            log += "- Using model's default objective function\n\n"
    else:
        log += f"- Using model's default objective function: {model.objective.expression}\n\n"

    # Step 4: Solve the FBA problem
    log += "## Step 4: Solving FBA optimization problem\n"
    try:
        solution = model.optimize()
        log += f"- Optimization status: {solution.status}\n"
        log += f"- Objective value: {solution.objective_value:.6f}\n\n"
    except Exception as e:
        log += f"- Error during optimization: {str(e)}\n"
        return log

    # Step 5: Save and report results
    log += "## Step 5: Analyzing flux distribution\n"

    # Create a dataframe with the flux distribution
    flux_distribution = pd.DataFrame(
        {
            "reaction_id": [r.id for r in model.reactions],
            "reaction_name": [r.name for r in model.reactions],
            "flux": [solution.fluxes[r.id] for r in model.reactions],
            "lower_bound": [r.lower_bound for r in model.reactions],
            "upper_bound": [r.upper_bound for r in model.reactions],
        }
    )

    # Save to file
    (output_file,) = _claim_outputs(output_file)
    _write_atomically(output_file, lambda path: flux_distribution.to_csv(path, index=False))
    log += f"- Flux distribution saved to {output_file}\n"

    # Report active reactions (non-zero flux)
    active_reactions = flux_distribution[abs(flux_distribution["flux"]) > 1e-6]
    log += f"- Number of active reactions (flux > 1e-6): {len(active_reactions)}\n"

    # Report top reactions by absolute flux
    top_reactions = flux_distribution.iloc[abs(flux_distribution["flux"]).argsort()[::-1]].head(10)
    log += "- Top 10 reactions by absolute flux magnitude:\n"
    for _, row in top_reactions.iterrows():
        log += f"  * {row['reaction_id']} ({row['reaction_name']}): {row['flux']:.6f}\n"

    log += f"\nFBA analysis complete. Full results available in {output_file}\n"

    return log


def model_protein_dimerization_network(monomer_concentrations, dimerization_affinities, network_topology):
    """
    Model protein dimerization networks to find equilibrium concentrations of dimers.

    Parameters:
    -----------
    monomer_concentrations : dict
        Dictionary mapping monomer names to their initial concentrations (in arbitrary units)
    dimerization_affinities : dict
        Dictionary mapping dimer names (as 'A-B' strings) to their association constants (Ka)
    network_topology : list of tuples
        List of (monomer1, monomer2) pairs that can form dimers

    Returns:
    --------
    str
        Research log summarizing the modeling process and results
    """
    import time

    import numpy as np
    from scipy.integrate import solve_ivp

    # Start timing
    start_time = time.time()

    # Extract unique monomers and create mapping to indices
    all_monomers = list(monomer_concentrations.keys())
    monomer_to_idx = {monomer: i for i, monomer in enumerate(all_monomers)}

    # Create mapping from dimer indices to names
    dimer_pairs = []
    dimer_names = []
    dimer_affinities = []

    for m1, m2 in network_topology:
        dimer_name = f"{m1}-{m2}"
        if dimer_name in dimerization_affinities:
            affinity = dimerization_affinities[dimer_name]
        else:
            # Try reverse order
            dimer_name = f"{m2}-{m1}"
            if dimer_name in dimerization_affinities:
                affinity = dimerization_affinities[dimer_name]
            else:
                raise ValueError(f"No affinity found for dimer pair {m1}-{m2}")

        dimer_pairs.append((monomer_to_idx[m1], monomer_to_idx[m2]))
        dimer_names.append(dimer_name)
        dimer_affinities.append(affinity)

    # Initial conditions (monomer concentrations followed by dimer concentrations)
    num_monomers = len(all_monomers)
    num_dimers = len(dimer_pairs)
    y0 = np.zeros(num_monomers + num_dimers)

    for monomer, conc in monomer_concentrations.items():
        y0[monomer_to_idx[monomer]] = conc

    # Define ODE system
    def dimerization_odes(t, y):
        dydt = np.zeros_like(y)

        # Extract current concentrations
        monomer_concs = y[:num_monomers]
        dimer_concs = y[num_monomers:]

        # Calculate rate of change for each species
        for i, ((m1_idx, m2_idx), affinity) in enumerate(zip(dimer_pairs, dimer_affinities, strict=False)):
            # Formation rate: kon * [A] * [B]
            # Dissociation rate: koff * [AB]
            # At equilibrium: kon/koff = Ka (affinity)
            # For simplicity, set kon = Ka and koff = 1

            kon = affinity
            koff = 1.0

            # Dimer formation/dissociation
            dimer_idx = num_monomers + i

            # Rate of change
            formation_rate = kon * monomer_concs[m1_idx] * monomer_concs[m2_idx]
            dissociation_rate = koff * dimer_concs[i]
            net_rate = formation_rate - dissociation_rate

            # Update rates for monomers and dimers
            dydt[m1_idx] -= net_rate
            dydt[m2_idx] -= net_rate
            dydt[dimer_idx] += net_rate

        return dydt

    # Solve ODE system to equilibrium
    # Using a long enough time to reach equilibrium
    t_span = (0, 1000)  # Arbitrary long time to reach equilibrium

    sol = solve_ivp(
        dimerization_odes,
        t_span,
        y0,
        method="BDF",  # Good for stiff problems
        rtol=1e-6,
        atol=1e-9,
    )

    # Extract final concentrations (equilibrium)
    final_monomer_concs = {monomer: sol.y[idx][-1] for monomer, idx in monomer_to_idx.items()}
    final_dimer_concs = {dimer_name: sol.y[num_monomers + i][-1] for i, dimer_name in enumerate(dimer_names)}

    # Calculate time taken
    elapsed_time = time.time() - start_time

    # Generate research log
    log = "Protein Dimerization Network Modeling\n"
    log += "=====================================\n\n"
    log += "Network summary:\n"
    log += f"- Number of monomers: {num_monomers}\n"
    log += f"- Number of possible dimers: {num_dimers}\n\n"

    log += "Initial conditions:\n"
    for monomer, conc in monomer_concentrations.items():
        log += f"- {monomer}: {conc:.4f}\n"
    log += "\n"

    log += "Dimerization affinities:\n"
    for dimer, affinity in zip(dimer_names, dimer_affinities, strict=False):
        log += f"- {dimer}: {affinity:.4e}\n"
    log += "\n"

    log += "Equilibrium concentrations:\n"
    log += "Monomers:\n"
    for monomer, conc in final_monomer_concs.items():
        log += f"- {monomer}: {conc:.4f}\n"

    log += "\nDimers:\n"
    for dimer, conc in final_dimer_concs.items():
        log += f"- {dimer}: {conc:.4f}\n"
    log += "\n"

    log += f"Simulation completed in {elapsed_time:.2f} seconds.\n"

    return log


def simulate_metabolic_network_perturbation(
    model_file, initial_concentrations, perturbation_params, simulation_time=100, time_points=1000
):
    """
    Construct and simulate kinetic models of metabolic networks and analyze their responses to perturbations.

    Parameters:
    -----------
    model_file : str
        Path to the COBRA model file (SBML format)
    initial_concentrations : dict
        Dictionary mapping metabolite IDs to their initial concentrations
    perturbation_params : dict
        Dictionary with the following keys:
        - 'time': float, time at which perturbation is applied
        - 'metabolite': str, ID of the metabolite to perturb
        - 'factor': float, multiplication factor for the metabolite concentration
    simulation_time : float, optional
        Total simulation time (default: 100)
    time_points : int, optional
        Number of time points to simulate (default: 1000)

    Returns:
    --------
    str
        Research log summarizing the steps taken and results obtained
    """

    import cobra
    import numpy as np
    import pandas as pd
    from scipy.integrate import solve_ivp
    from scipy.sparse import csr_matrix

    # The perturbation is checked before anything is integrated. A time of 0 made the "before" index
    # wrap to the last sample and a time past simulation_time raised IndexError after the result
    # files were already written (hunt 2026-09-30, uT6-literature-4).
    missing = [key for key in ("time", "metabolite", "factor") if key not in (perturbation_params or {})]
    if missing:
        return f"Error: perturbation_params needs the keys 'time', 'metabolite' and 'factor'; missing {missing}."
    perturb_time = float(perturbation_params["time"])
    perturb_factor = float(perturbation_params["factor"])
    perturb_metabolite = perturbation_params["metabolite"]
    if not 0 < perturb_time < simulation_time:
        return (
            f"Error: perturbation time {perturb_time} must lie strictly between 0 and simulation_time "
            f"({simulation_time}). To perturb at time 0, scale the metabolite in initial_concentrations instead."
        )

    # Step 1: Load the metabolic model
    try:
        model = cobra.io.read_sbml_model(model_file)
        log = f"Loaded metabolic model from {model_file} with {len(model.reactions)} reactions and {len(model.metabolites)} metabolites.\n\n"
    except Exception as e:
        return f"Error loading model: {str(e)}"

    # Step 2: Set up metabolite list and initial concentrations
    metabolites = list(model.metabolites)
    metabolite_ids = [m.id for m in metabolites]
    if perturb_metabolite not in metabolite_ids:
        return (
            f"Error: metabolite {perturb_metabolite!r} is not in the model; "
            f"its metabolite ids look like {metabolite_ids[:5]}."
        )
    perturb_idx = metabolite_ids.index(perturb_metabolite)

    # Set default initial concentrations for metabolites not specified
    conc_array = np.ones(len(metabolites))
    for i, m_id in enumerate(metabolite_ids):
        if m_id in initial_concentrations:
            conc_array[i] = initial_concentrations[m_id]

    log += "Initial concentrations set up for all metabolites.\n\n"

    # Step 3: Define kinetic model using simple mass-action kinetics
    # The stoichiometry is read once into a sparse matrix, and each reaction's rate is the product of
    # its substrate concentrations raised to their coefficients (1.0 for a reaction with no
    # substrate, as before). The right-hand side used to loop over every metabolite and every
    # reaction with a list scan per pair, about 1 s per evaluation at 1000 x 2000, so a genome-scale
    # model never finished (hunt 2026-09-30, uT6-literature-34).
    reactions = list(model.reactions)
    reaction_ids = [r.id for r in reactions]
    position = {m_id: i for i, m_id in enumerate(metabolite_ids)}
    rows, cols, coeffs = [], [], []
    sub_reaction, sub_metabolite, sub_power = [], [], []
    for j, reaction in enumerate(reactions):
        for metabolite, coeff in reaction.metabolites.items():
            i = position[metabolite.id]
            rows.append(i)
            cols.append(j)
            coeffs.append(float(coeff))
            if coeff < 0:  # Substrate
                sub_reaction.append(j)
                sub_metabolite.append(i)
                sub_power.append(abs(float(coeff)))
    stoichiometry = csr_matrix((coeffs, (rows, cols)), shape=(len(metabolite_ids), len(reactions)))
    sub_reaction = np.asarray(sub_reaction, dtype=int)
    sub_metabolite = np.asarray(sub_metabolite, dtype=int)
    sub_power = np.asarray(sub_power, dtype=float)

    def reaction_rates(concentrations):
        rates = np.ones(len(reactions))
        np.multiply.at(rates, sub_reaction, concentrations[sub_metabolite] ** sub_power)
        return rates

    # Step 4: Define ODE system
    def ode_system(t, concentrations):
        return stoichiometry @ reaction_rates(concentrations)

    log += "Kinetic model defined using mass-action kinetics for all reactions.\n\n"

    # Step 5: Simulate the system
    # In two pieces, with the perturbation applied to the state between them. It used to be applied
    # by mutating `y` inside the right-hand side, which LSODA never takes into its own state, so the
    # pulse never reached the trajectory and every run reported no response (hunt 2026-09-30,
    # uT6-literature-4).
    t_eval = np.linspace(0, simulation_time, time_points)
    before = t_eval[t_eval < perturb_time]
    after = t_eval[t_eval > perturb_time]

    log += f"Starting simulation for {simulation_time} time units with perturbation of {perturb_metabolite} "
    log += f"by factor {perturb_factor} at time {perturb_time}.\n\n"

    try:
        first = solve_ivp(
            ode_system, (0, perturb_time), conc_array, t_eval=np.append(before, perturb_time), method="LSODA"
        )
        if not first.success:
            return f"Error during simulation before the perturbation: {first.message}"
        pre_state = first.y[:, -1].copy()
        post_state = pre_state.copy()
        post_state[perturb_idx] *= perturb_factor
        second = solve_ivp(
            ode_system,
            (perturb_time, simulation_time),
            post_state,
            t_eval=np.insert(after, 0, perturb_time),
            method="LSODA",
        )
        if not second.success:
            return f"Error during simulation after the perturbation: {second.message}"
    except Exception as e:
        return f"Error during simulation: {str(e)}"

    # Both states at the perturbation time are kept, so the step is visible in the saved trajectory.
    times = np.concatenate([first.t, second.t])
    states = np.concatenate([first.y, second.y], axis=1)
    log += (
        f"Simulation completed successfully with {len(times)} time points "
        f"(the perturbation time {perturb_time} appears twice: just before and just after the step).\n\n"
    )

    # Step 6: Calculate fluxes at each time point
    fluxes = np.array([reaction_rates(states[:, k]) for k in range(states.shape[1])])

    log += "Calculated reaction fluxes for all time points.\n\n"

    # Step 7: Save results to files
    conc_df = pd.DataFrame(states.T, columns=metabolite_ids)
    conc_df["time"] = times
    # Absolute, so the paths this log reports back are ones the caller can actually find.
    conc_file, flux_file = _claim_outputs("metabolite_concentrations.csv", "reaction_fluxes.csv")
    _write_atomically(conc_file, lambda path: conc_df.to_csv(path, index=False))

    flux_df = pd.DataFrame(fluxes, columns=reaction_ids)
    flux_df["time"] = times
    _write_atomically(flux_file, lambda path: flux_df.to_csv(path, index=False))

    log += f"Results saved to {conc_file} and {flux_file}.\n\n"

    # Step 8: Analyze perturbation response
    # From the state just before the step to the first sample after it (the post-step state itself
    # when no sample falls after it).
    pre_perturb = pre_state
    post_perturb = second.y[:, 1] if second.y.shape[1] > 1 else second.y[:, 0]

    # Find metabolites with significant changes
    significant_changes = []
    for i, m_id in enumerate(metabolite_ids):
        rel_change = abs(post_perturb[i] - pre_perturb[i]) / pre_perturb[i] if pre_perturb[i] > 0 else 0
        if rel_change > 0.05:  # 5% change threshold
            significant_changes.append((m_id, rel_change))

    log += "Perturbation Analysis Results:\n"
    log += f"Perturbation of {perturb_metabolite} at time {perturb_time} by factor {perturb_factor}.\n"
    log += (
        f"{perturb_metabolite}: {pre_state[perturb_idx]:.6g} just before the perturbation, "
        f"{post_state[perturb_idx]:.6g} just after.\n"
    )
    log += f"Number of metabolites with significant immediate response: {len(significant_changes)}.\n"

    if significant_changes:
        log += "Top 5 most affected metabolites (by relative concentration change):\n"
        for m_id, change in sorted(significant_changes, key=lambda x: x[1], reverse=True)[:5]:
            log += f"  - {m_id}: {change * 100:.2f}% change\n"

    log += "\nSimulation and perturbation analysis completed successfully."

    return log


def simulate_protein_signaling_network(
    network_structure, reaction_params, species_params, simulation_time=100, time_points=1000
):
    """
    Simulate protein signaling network dynamics using ODE-based logic modeling with normalized Hill functions.

    Parameters:
    -----------
    network_structure : dict
        Dictionary defining the network topology. Each key is a target protein and its value is a list of tuples
        (regulator, regulation_type) where regulation_type is 1 for activation and -1 for inhibition.

    reaction_params : dict
        Dictionary of reaction parameters. Keys are tuples (regulator, target) and values are dictionaries
        with keys 'W' (weight), 'n' (Hill coefficient), and 'EC50' (half-maximal effective concentration).

    species_params : dict
        Dictionary of species parameters. Keys are protein names and values are dictionaries
        with keys 'tau' (time constant), 'y0' (initial concentration), and 'ymax' (maximum concentration).

    simulation_time : float, optional
        Total simulation time in arbitrary time units. Default is 100.

    time_points : int, optional
        Number of time points for the simulation. Default is 1000.

    Returns:
    --------
    str
        Research log summarizing the simulation process and results.
    """
    import csv

    import numpy as np
    from scipy.integrate import solve_ivp

    # Extract all unique protein species
    all_proteins = set(network_structure.keys())
    for _target, regulators in network_structure.items():
        for regulator, _ in regulators:
            all_proteins.add(regulator)

    # Create ordered list of proteins for indexing
    proteins = sorted(all_proteins)
    protein_indices = {protein: i for i, protein in enumerate(proteins)}

    # Define the normalized Hill function
    def hill_function(x, n, ec50):
        return x**n / (x**n + ec50**n)

    # Define the ODE system
    def ode_system(t, y):
        dydt = np.zeros_like(y)

        for target, regulators in network_structure.items():
            target_idx = protein_indices[target]
            target_params = species_params[target]

            # Calculate regulation term for each regulator
            regulation_terms = []
            for regulator, reg_type in regulators:
                regulator_idx = protein_indices[regulator]
                params = reaction_params.get((regulator, target), {})

                if not params:
                    continue

                x = y[regulator_idx]
                n = params["n"]
                ec50 = params["EC50"]
                weight = params["W"]

                # Apply Hill function based on regulation type
                if reg_type == 1:  # Activation
                    term = weight * hill_function(x, n, ec50)
                else:  # Inhibition
                    term = weight * (1 - hill_function(x, n, ec50))

                regulation_terms.append(term)

            # Combine regulation terms (if any)
            if regulation_terms:
                # Simple summation of regulation terms
                regulation = sum(regulation_terms) / len(regulation_terms)

                # Ensure regulation stays within [0, 1]
                regulation = max(0, min(1, regulation))

                # Calculate rate of change
                tau = target_params["tau"]
                ymax = target_params["ymax"]
                dydt[target_idx] = (1 / tau) * (regulation * ymax - y[target_idx])

        return dydt

    # Set up initial conditions
    y0 = np.zeros(len(proteins))
    for protein, params in species_params.items():
        if protein in protein_indices:
            y0[protein_indices[protein]] = params["y0"]

    # Set up time points
    t_span = (0, simulation_time)
    t_eval = np.linspace(0, simulation_time, time_points)

    # Solve the ODE system
    solution = solve_ivp(ode_system, t_span, y0, method="LSODA", t_eval=t_eval, rtol=1e-6, atol=1e-9)

    # Save results to CSV
    # Absolute, so the path this log reports back is one the caller can actually find.
    (output_file,) = _claim_outputs("protein_signaling_simulation_results.csv")

    def _write(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header
            header = ["Time"] + proteins
            writer.writerow(header)

            # Write data
            for i in range(len(solution.t)):
                row = [solution.t[i]] + list(solution.y[:, i])
                writer.writerow(row)

    _write_atomically(output_file, _write)

    # Generate research log
    log = "ODE-based Logic Modeling of Protein Signaling Networks\n"
    log += "=============================================\n\n"
    log += f"Network structure: {len(network_structure)} target proteins, {len(proteins)} total proteins\n"
    log += f"Simulation time: {simulation_time} time units\n"
    log += f"Number of time points: {time_points}\n\n"

    log += "Simulation steps:\n"
    log += "1. Initialized protein concentrations based on provided initial values\n"
    log += "2. Set up ODE system using normalized Hill functions for protein interactions\n"
    log += "3. Solved the ODE system using LSODA integration method\n"
    log += "4. Saved time-series data for all protein concentrations\n\n"

    log += f"Results saved to: {output_file}\n\n"

    # Add summary statistics
    log += "Summary of final protein concentrations:\n"
    for i, protein in enumerate(proteins):
        final_conc = solution.y[i, -1]
        log += f"- {protein}: {final_conc:.4f}\n"

    return log


def compare_protein_structures(pdb_file1, pdb_file2, chain_id1="A", chain_id2="A", output_prefix="protein_comparison"):
    """
    Compares two protein structures to identify structural differences and conformational changes.

    Parameters:
    -----------
    pdb_file1 : str
        Path to the first PDB file
    pdb_file2 : str
        Path to the second PDB file
    chain_id1 : str, optional
        Chain ID to analyze in the first structure (default: 'A')
    chain_id2 : str, optional
        Chain ID to analyze in the second structure (default: 'A')
    output_prefix : str, optional
        Prefix for output files (default: "protein_comparison")

    Returns:
    --------
    str
        A research log summarizing the structural comparison analysis
    """
    import warnings

    import numpy as np
    from Bio.PDB import PDBIO, PDBParser, Select, Superimposer
    from Bio.PDB.PDBExceptions import PDBConstructionWarning

    # Suppress PDB parsing warnings
    warnings.filterwarnings("ignore", category=PDBConstructionWarning)

    research_log = []
    research_log.append("# Protein Structure Comparison Analysis\n")
    research_log.append(f"Comparing structures from {pdb_file1} and {pdb_file2}\n")

    # Initialize parser
    parser = PDBParser()

    # Parse structures
    research_log.append("## Loading PDB structures")
    structure1 = parser.get_structure("structure1", pdb_file1)
    structure2 = parser.get_structure("structure2", pdb_file2)

    # Get specified chains
    try:
        chain1 = structure1[0][chain_id1]
        chain2 = structure2[0][chain_id2]
        research_log.append(f"Successfully loaded chain {chain_id1} from {pdb_file1}")
        research_log.append(f"Successfully loaded chain {chain_id2} from {pdb_file2}")
    except KeyError as e:
        return f"Error: Chain not found: {str(e)}"

    # Get CA atoms for alignment
    ca_atoms1 = []
    ca_atoms2 = []

    # Create residue mappings based on residue ID
    # Keyed by (number, insertion code) over amino-acid residues only. Keyed by number alone, 100, 100A
    # and 100B collapsed into one entry, and a HETATM calcium ion -- whose atom is also named "CA" --
    # was paired and superposed as if it were an alpha carbon (hunt 2026-09-30, uT6-literature-37).
    # A HETATM residue with a backbone whose CA is a carbon (selenomethionine in a SeMet crystal) is an
    # amino acid and stays in; an ATOM-only filter dropped it from the fit and the report.
    def _amino_acid(residue):
        if residue.id[0] == " ":
            return residue.has_id("CA")
        if residue.id[0] == "W" or not all(residue.has_id(name) for name in ("N", "CA", "C")):
            return False
        return str(getattr(residue["CA"], "element", "")).strip().upper() == "C"

    residues1 = {(r.id[1], r.id[2]): r for r in chain1 if _amino_acid(r)}
    residues2 = {(r.id[1], r.id[2]): r for r in chain2 if _amino_acid(r)}

    def _label(key):
        number, icode = key
        return f"{number}{str(icode).strip()}"

    # Find common residues
    common_residues = sorted(set(residues1.keys()) & set(residues2.keys()))

    if len(common_residues) == 0:
        return "Error: No common residues with CA atoms found between the structures"

    research_log.append(f"Found {len(common_residues)} common residues for alignment\n")

    # Get paired atoms for alignment
    for res_id in common_residues:
        ca_atoms1.append(residues1[res_id]["CA"])
        ca_atoms2.append(residues2[res_id]["CA"])

    # Perform structural alignment
    research_log.append("## Structural Alignment")
    super_imposer = Superimposer()
    super_imposer.set_atoms(ca_atoms1, ca_atoms2)
    super_imposer.apply(structure2.get_atoms())

    # Calculate RMSD
    rmsd = super_imposer.rms
    research_log.append(f"Overall RMSD: {rmsd:.4f} Å\n")

    # Calculate per-residue distance after alignment
    research_log.append("## Conformational Differences Analysis")

    # Find regions with significant differences
    distance_data = []
    significant_changes = []
    threshold = 2.0  # Angstroms threshold for significant difference

    for res_id in common_residues:
        res1 = residues1[res_id]
        res2 = residues2[res_id]

        # Calculate distance between CA atoms
        ca1 = res1["CA"]
        ca2 = res2["CA"]
        distance = np.linalg.norm(ca1.coord - ca2.coord)

        distance_data.append((res_id, distance))

        if distance > threshold:
            significant_changes.append((res_id, res1.get_resname(), distance))

    # Save alignment as PDB files
    aligned_file1, aligned_file2, distance_file = _claim_outputs(
        f"{output_prefix}_ref.pdb", f"{output_prefix}_aligned.pdb", f"{output_prefix}_residue_distances.csv"
    )

    io = PDBIO()
    io.set_structure(structure1)
    _write_atomically(aligned_file1, lambda path: io.save(path, select=Select()))

    io.set_structure(structure2)
    _write_atomically(aligned_file2, lambda path: io.save(path, select=Select()))

    research_log.append(f"Aligned structures saved as {aligned_file1} and {aligned_file2}")

    # Report on significant differences
    if significant_changes:
        research_log.append(
            f"\nIdentified {len(significant_changes)} residues with significant conformational changes:"
        )
        for res_id, res_name, distance in significant_changes:
            research_log.append(f"  - Residue {res_name}{_label(res_id)}: {distance:.2f} Å displacement")
    else:
        research_log.append("\nNo significant conformational changes detected (threshold: 2.0 Å)")

    # Save per-residue distance data
    def _write_distances(path):
        with open(path, "w") as f:
            f.write("Residue_ID,Distance(Å)\n")
            for res_id, distance in distance_data:
                f.write(f"{_label(res_id)},{distance:.4f}\n")

    _write_atomically(distance_file, _write_distances)

    research_log.append(f"\nPer-residue distance data saved to {distance_file}")

    # Identify regions with continuous changes
    regions = []
    current_region = []

    # Two residues are consecutive when nothing lies between them in EITHER chain's own residue order
    # and their numbers step by at most one (by none for an insertion code: 100 -> 100A -> 101). The
    # earlier test, "next entry in the list both structures share", skipped every residue missing from
    # one structure, so the displaced stretches either side of an unresolved loop were reported as one
    # region spanning the loop (hunt 2026-09-30, uT6-literature-37 review).
    position1 = {key: index for index, key in enumerate(residues1)}
    position2 = {key: index for index, key in enumerate(residues2)}

    def _follows(previous, key):
        return (
            position1[key] == position1[previous] + 1
            and position2[key] == position2[previous] + 1
            and 0 <= key[0] - previous[0] <= 1
        )

    for res_id, _, distance in significant_changes:
        if not current_region or _follows(current_region[-1][0], res_id):
            current_region.append((res_id, distance))
        else:
            if len(current_region) >= 3:  # Consider regions with at least 3 consecutive residues
                regions.append(current_region)
            current_region = [(res_id, distance)]

    if current_region and len(current_region) >= 3:
        regions.append(current_region)

    if regions:
        research_log.append("\n## Continuous Regions with Conformational Changes")
        for i, region in enumerate(regions, 1):
            start_res = _label(region[0][0])
            end_res = _label(region[-1][0])
            avg_dist = sum(r[1] for r in region) / len(region)
            research_log.append(f"Region {i}: Residues {start_res}-{end_res} (Average displacement: {avg_dist:.2f} Å)")

    research_log.append("\n## Summary")
    research_log.append(f"- Compared structures from {pdb_file1} and {pdb_file2}")
    research_log.append(f"- Overall RMSD: {rmsd:.4f} Å")
    research_log.append(f"- {len(significant_changes)} residues with significant conformational changes")
    research_log.append(f"- {len(regions)} continuous regions of conformational change")
    research_log.append(f"- Files generated: {aligned_file1}, {aligned_file2}, {distance_file}")

    return "\n".join(research_log)


def simulate_renin_angiotensin_system_dynamics(
    initial_concentrations, rate_constants, feedback_params, simulation_time=48, time_points=100
):
    """
    Simulate the time-dependent concentrations of renin-angiotensin system (RAS) components.

    Parameters:
    -----------
    initial_concentrations : dict
        Initial concentrations of RAS components with keys:
        'renin', 'angiotensinogen', 'angiotensin_I', 'angiotensin_II',
        'ACE2_angiotensin_II', 'angiotensin_1_7'

    rate_constants : dict
        Kinetic rate constants with keys:
        'k_ren' (renin production), 'k_agt' (angiotensinogen production),
        'k_ace' (ACE conversion rate), 'k_ace2' (ACE2 conversion rate),
        'k_at1r' (AT1R binding rate), 'k_mas' (Mas receptor binding rate)

    feedback_params : dict
        Parameters controlling feedback mechanisms with keys:
        'fb_ang_II' (angiotensin II feedback), 'fb_ace2' (ACE2 feedback)

    simulation_time : float, optional
        Total simulation time in hours (default: 48)

    time_points : int, optional
        Number of time points to evaluate (default: 100)

    Returns:
    --------
    str
        Research log summarizing the simulation steps and results
    """

    import numpy as np
    import pandas as pd
    from scipy.integrate import solve_ivp

    # Extract initial concentrations
    y0 = [
        initial_concentrations["renin"],
        initial_concentrations["angiotensinogen"],
        initial_concentrations["angiotensin_I"],
        initial_concentrations["angiotensin_II"],
        initial_concentrations["ACE2_angiotensin_II"],
        initial_concentrations["angiotensin_1_7"],
    ]

    # Define the system of ODEs
    def ras_ode_system(t, y):
        renin, agt, ang_I, ang_II, ace2_ang_II, ang_1_7 = y

        # Production rates with feedback
        renin_production = rate_constants["k_ren"] * (1 / (1 + feedback_params["fb_ang_II"] * ang_II))
        agt_production = rate_constants["k_agt"]

        # Conversion rates
        ang_I_formation = renin * agt
        ang_II_formation = rate_constants["k_ace"] * ang_I
        # ACE2 feedback, in the same form as the renin feedback above: Ang II suppresses ACE2
        # activity, 1/(1 + fb_ace2 * [Ang II]); fb_ace2 = 0 is no feedback. fb_ace2 was required and
        # printed under "Feedback parameters" but never entered the equations, so every value gave
        # the same trajectories (hunt 2026-09-30, uT6-literature-35).
        ace2_binding = rate_constants["k_ace2"] * ang_II / (1 + feedback_params["fb_ace2"] * ang_II)
        ang_1_7_formation = ace2_ang_II

        # Clearance/degradation (simplified as proportional to concentration)
        renin_clearance = 0.1 * renin
        agt_clearance = 0.05 * agt
        ang_I_clearance = 0.2 * ang_I
        ang_II_clearance = 0.3 * ang_II + rate_constants["k_at1r"] * ang_II
        ace2_ang_II_clearance = 0.15 * ace2_ang_II
        ang_1_7_clearance = 0.25 * ang_1_7 + rate_constants["k_mas"] * ang_1_7

        # ODEs
        drenin_dt = renin_production - renin_clearance
        dagt_dt = agt_production - agt_clearance - ang_I_formation
        dang_I_dt = ang_I_formation - ang_I_clearance - ang_II_formation
        dang_II_dt = ang_II_formation - ang_II_clearance - ace2_binding
        dace2_ang_II_dt = ace2_binding - ace2_ang_II_clearance - ang_1_7_formation
        dang_1_7_dt = ang_1_7_formation - ang_1_7_clearance

        return [drenin_dt, dagt_dt, dang_I_dt, dang_II_dt, dace2_ang_II_dt, dang_1_7_dt]

    # Time points for simulation
    t_span = (0, simulation_time)
    t_eval = np.linspace(0, simulation_time, time_points)

    # Solve the ODE system
    solution = solve_ivp(ras_ode_system, t_span, y0, method="RK45", t_eval=t_eval, rtol=1e-6)

    # Create DataFrame with results
    component_names = [
        "Renin",
        "Angiotensinogen",
        "Angiotensin I",
        "Angiotensin II",
        "ACE2-Angiotensin II",
        "Angiotensin 1-7",
    ]
    results_df = pd.DataFrame(solution.y.T, columns=component_names)
    results_df.insert(0, "Time (hours)", solution.t)

    # Save results to CSV
    # Absolute, so the path this log reports back is one the caller can actually find.
    (results_file,) = _claim_outputs("ras_simulation_results.csv")
    _write_atomically(results_file, lambda path: results_df.to_csv(path, index=False))

    # Create research log
    log = f"""RAS ODE Modeling Simulation Log:

1. Initialized RAS component concentrations:
   - Renin: {initial_concentrations["renin"]}
   - Angiotensinogen: {initial_concentrations["angiotensinogen"]}
   - Angiotensin I: {initial_concentrations["angiotensin_I"]}
   - Angiotensin II: {initial_concentrations["angiotensin_II"]}
   - ACE2-Angiotensin II complex: {initial_concentrations["ACE2_angiotensin_II"]}
   - Angiotensin 1-7: {initial_concentrations["angiotensin_1_7"]}

2. Applied rate constants:
   - Renin production (k_ren): {rate_constants["k_ren"]}
   - Angiotensinogen production (k_agt): {rate_constants["k_agt"]}
   - ACE conversion rate (k_ace): {rate_constants["k_ace"]}
   - ACE2 conversion rate (k_ace2): {rate_constants["k_ace2"]}
   - AT1R binding rate (k_at1r): {rate_constants["k_at1r"]}
   - Mas receptor binding rate (k_mas): {rate_constants["k_mas"]}

3. Feedback parameters:
   - Angiotensin II feedback (fb_ang_II): {feedback_params["fb_ang_II"]} (renin production x 1/(1 + fb_ang_II*[Ang II]))
   - ACE2 feedback (fb_ace2): {feedback_params["fb_ace2"]} (ACE2 conversion x 1/(1 + fb_ace2*[Ang II]))

4. Simulation parameters:
   - Total simulation time: {simulation_time} hours
   - Number of time points: {time_points}

5. Solved system of ODEs using SciPy's solve_ivp with RK45 method.

6. Final concentrations at {simulation_time} hours:
   - Renin: {results_df["Renin"].iloc[-1]:.4f}
   - Angiotensinogen: {results_df["Angiotensinogen"].iloc[-1]:.4f}
   - Angiotensin I: {results_df["Angiotensin I"].iloc[-1]:.4f}
   - Angiotensin II: {results_df["Angiotensin II"].iloc[-1]:.4f}
   - ACE2-Angiotensin II complex: {results_df["ACE2-Angiotensin II"].iloc[-1]:.4f}
   - Angiotensin 1-7: {results_df["Angiotensin 1-7"].iloc[-1]:.4f}

7. Results saved to file: {results_file}

8. Key observations:
   - The model captures the conversion of Angiotensinogen to Angiotensin I via Renin
   - Angiotensin I is converted to Angiotensin II via ACE
   - Angiotensin II binds with ACE2 to form the ACE2-Angiotensin II complex
   - The complex is converted to Angiotensin 1-7
   - Feedback mechanisms regulate the production rates
"""

    return log
