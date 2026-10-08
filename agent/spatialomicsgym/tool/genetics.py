def liftover_coordinates(
    chromosome: str,
    position: int,
    input_format: str,
    output_format: str,
    data_path: str,
) -> str:
    """Perform liftover of genomic coordinates between hg19 and hg38 formats with detailed intermediate steps.

    Args:
        chromosome (str): Chromosome number (e.g., '1', 'X').
        position (int): Genomic position.
        input_format (str): Input genome build ('hg19' or 'hg38').
        output_format (str): Output genome build ('hg19' or 'hg38').
        data_path (str): Data-lake root holding ``liftover/hg19ToHg38.over.chain.gz`` and
            ``liftover/hg38ToHg19.over.chain.gz``, or the directory holding those chain files itself.
            Only the chain for the requested direction is read.

    Returns:
        str: A detailed string explaining the steps and the final result or any error encountered.

    """
    from pyliftover import LiftOver

    steps = []

    try:
        steps.append(
            f"Starting liftover process for chromosome {chromosome}, position {position} from {input_format} to {output_format}."
        )

        # Choose the chain for the requested direction, then load only that one. Both chains were
        # loaded from data_path + "/liftover/" whatever the direction, so a caller who passed the
        # folder that holds the chains -- what the description asked for -- got an exception, and
        # one missing chain broke the other direction too (hunt 2026-09-30, uT4-genomics-31).
        if input_format == "hg19" and output_format == "hg38":
            chain_name = "hg19ToHg38.over.chain.gz"
            steps.append("Selected liftover chain: hg19 to hg38.")
        elif input_format == "hg38" and output_format == "hg19":
            chain_name = "hg38ToHg19.over.chain.gz"
            steps.append("Selected liftover chain: hg38 to hg19.")
        else:
            steps.append("Error: Unsupported format conversion.")
            return "\n".join(
                steps
                + ["Error: Unsupported format conversion. Supported formats are 'hg19' to 'hg38' or 'hg38' to 'hg19'."]
            )

        candidates = [os.path.join(data_path, "liftover", chain_name), os.path.join(data_path, chain_name)]
        chain_path = next((c for c in candidates if os.path.isfile(c)), None)
        if chain_path is None:
            steps.append(f"Error: liftover chain {chain_name} not found; looked for {' and '.join(candidates)}.")
            return "\n".join(steps)
        steps.append(f"Loading liftover chain file {chain_path}...")
        lo = LiftOver(chain_path)
        steps.append("Liftover chain file loaded successfully.")

        # Perform the liftover conversion
        steps.append(f"Performing liftover for chr{chromosome}, position {position}...")
        lifted_coordinates = lo.convert_coordinate(f"chr{chromosome}", position)

        if lifted_coordinates:
            result = (
                f"Successfully lifted coordinates from {input_format} to {output_format}.\n"
                f"Original: chr{chromosome}, position {position}.\n"
                f"Lifted: chromosome {lifted_coordinates[0][0]}, position {lifted_coordinates[0][1]}, strand {lifted_coordinates[0][2]}."
            )
            steps.append(result)
            return "\n".join(steps)
        else:
            steps.append("Error: Liftover failed. No coordinates found for the given input.")
            return "\n".join(steps + ["Error: Liftover failed. No coordinates found."])

    except Exception as e:
        # "Error: " at line start is what the ReAct loop reads as a failed step; "Exception
        # encountered:" read as a successful one (hunt 2026-09-30, uT4-genomics-23).
        steps.append(f"Error: liftover failed: {str(e)}")
        return "\n".join(steps)


import math
import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

# torch is imported inside bayesian_finemapping_with_deep_vi, the one tool that uses it. At module
# level it made every tool here unimportable in the agent env, which has no torch, while the prompt
# tells the model to import them from this module (hunt 2026-09-30, uT4-genomics-3).


def _under_output_root(name):
    """Absolute path for an output file name or prefix.

    A bare name (no directory part) went to the process CWD, which the portal's REPL shares between
    every chat of one account, and was reported back bare -- so a second chat overwrote the first
    one's file and neither could say where it was. A bare name now goes under the run's output
    directory (``$SOG_WORK_DIR`` when set); a name with a directory part is used as given
    (hunt 2026-09-30, uT4-genomics-29).
    """
    name = str(name)
    if os.path.dirname(name):
        return os.path.abspath(name)
    from spatialomicsgym.paths import tool_output_root

    root = tool_output_root()
    os.makedirs(root, exist_ok=True)
    return os.path.abspath(os.path.join(root, name))


def _replace_into(path, write):
    """Write through ``path + '.partial'`` and move it into place, so a crash leaves no half file."""
    partial = f"{path}.partial"
    write(partial)
    os.replace(partial, path)


def _finemapping_neg_elbo(z, ld, pips, prior_pi, log, eps=1e-8):
    """Negative ELBO of the fine-mapping model under q(s) = prod_i Bernoulli(pips_i).

    The data term is the expectation of 0.5 * ||z - R (s * z)||^2 in closed form -- the mean term
    plus each variant's Bernoulli variance -- so it is differentiable in ``pips``; the prior term is
    the KL from Bernoulli(prior_pi). For identity LD the optimum is
    logit(pip_i) = logit(prior_pi) + z_i^2 / 2. Uses only operators torch tensors and numpy arrays
    share, so the training loop and a test can call the same code.
    """
    residual = z - ld @ (pips * z)
    variance = pips * (1 - pips) * z**2 * (ld**2).sum(0)
    data = (residual**2).sum() + variance.sum()
    kl = (
        pips * (log(pips + eps) - math.log(prior_pi)) + (1 - pips) * (log(1 - pips + eps) - math.log(1 - prior_pi))
    ).sum()
    return 0.5 * data + kl


#: Prior variance W of the causal variant's standardised effect (its expected z) in the single-effect
#: Bayes factor below: 50, the ``prior_variance`` default of CRAN susieR's ``susie_rss`` (0.12) for
#: z-scores without a sample size. susie_rss uses it as the start of an estimate; here it is fixed.
#: With equal priors W only sets how sharply the shares follow z: alpha_i is proportional to
#: exp(z_i^2 W / (2 (1 + W))).
_SER_PRIOR_VARIANCE = 50.0


def _single_effect_log_bf(z, prior_variance=_SER_PRIOR_VARIANCE):
    """Each variant's log Bayes factor for carrying the one causal effect, from its marginal z-score.

    Under the summary-statistics model z ~ N(R b, R) with one effect b = b_i e_i, b_i ~ N(0, W), the
    evidence for variant i reduces to its own z (R_ii = 1): BF_i = N(z_i; 0, 1 + W) / N(z_i; 0, 1),
    so log BF_i = -log(1 + W) / 2 + z_i^2 W / (2 (1 + W)) -- SuSiE-RSS's single-effect regression
    with L = 1. It does not depend on R. Raises ValueError on a non-finite z.
    """
    z = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError(f"{int((~np.isfinite(z)).sum())} variant(s) have a non-finite z-score")
    w = float(prior_variance)
    return -0.5 * math.log1p(w) + 0.5 * z**2 * w / (1.0 + w)


def _single_effect_alpha(log_weights):
    """Each variant's share of one causal signal: alpha_i proportional to exp(log_weights_i).

    ``log_weights`` is log prior_i + log BF_i up to a constant; with the tool's equal priors that is
    the log Bayes factor. Raises ValueError on a non-finite value.
    """
    x = np.asarray(log_weights, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{int((~np.isfinite(x)).sum())} variant(s) have a non-finite log Bayes factor")
    if not x.size:
        return x
    weights = np.exp(x - x.max())
    return weights / weights.sum()


def _credible_set(results_df, threshold, prior_variance=_SER_PRIOR_VARIANCE):
    """The smallest set of top variants whose shares of one causal signal reach ``threshold``.

    The shares are the single-effect posterior alpha_i = prior_i * BF_i / sum_j prior_j * BF_j, with
    equal priors and BF_i from the variant's marginal ``z_score`` (``_single_effect_log_bf``), and
    the variant whose share carries the cumulative sum across the threshold belongs to the set.

    The original rule kept rows with cumulative raw PIP <= threshold, which leaves that variant out
    and, with every PIP near 0, kept every variant. Its first repair rescaled the PIPs to sum to 1,
    so the nulls' prior-level PIPs outweighed any hit (a 25-of-30 95% set for one hit at p = 1e-20).
    The second took alpha as the softmax of the deep-VI model's fitted logits, which is prior x BF
    only at the closed-form optimum with identity LD: mean-field VI gives an LD block's whole signal
    to one member, often a tag, and strong hits' float32 logits stop where the sigmoid saturates, so
    the shares came from the optimiser and the seed (hunt 2026-09-30, uT4-genomics-1).
    """
    if "z_score" not in results_df.columns:
        raise ValueError("the credible set is built from a 'z_score' column, which is missing")
    ranked = results_df.copy()
    ranked["log_bf"] = _single_effect_log_bf(ranked["z_score"].to_numpy(dtype=float), prior_variance)
    ranked["alpha"] = _single_effect_alpha(ranked["log_bf"].to_numpy())
    ranked = ranked.sort_values("alpha", ascending=False, kind="mergesort")
    ranked["cumulative_alpha"] = ranked["alpha"].cumsum()
    n_in = int((ranked["cumulative_alpha"] < threshold - 1e-12).sum()) + 1
    return ranked.iloc[: min(n_in, len(ranked))]


def _credible_set_purity(ld_matrix, positions):
    """Smallest absolute LD correlation between two members of a credible set (1.0 for one member).

    SuSiE's purity: one causal signal's set is made of variants in LD with one another, so a low
    value says the set spans signals the single-effect model cannot tell apart.
    """
    positions = list(positions)
    if len(positions) < 2:
        return 1.0
    block = np.abs(np.asarray(ld_matrix, dtype=float)[np.ix_(positions, positions)])
    return float(block.min())


#: JASPAR is reached through utils.http_client (timeout, retries, egress policy), not a bare
#: requests.get that could hold a ReAct step open on a stalled socket (hunt 2026-09-30, uT4-genomics-22).
_JASPAR_HOSTS = ("jaspar.genereg.net",)

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _tfbs_site(sequence, position, length):
    """``(start, strand, site)`` for one hit from Biopython's ``PSSM.search``.

    ``search`` reports a minus-strand hit at forward start ``p`` as ``p - len(sequence)``. The site
    is read in the motif's own orientation, so a minus-strand site is the reverse complement of the
    forward bases. ``start`` is the 0-based forward-strand start either way.
    """
    if position >= 0:
        return position, "+", sequence[position : position + length]
    start = len(sequence) + position
    return start, "-", sequence[start : start + length].translate(_COMPLEMENT)[::-1]


def _reml_mm(y, kernels, init, max_iter=500, tol=1e-6):
    """REML variance components for y ~ N(1*mu, sum_k sigma2_k * K_k) by the MM update.

    sigma2_k <- sigma2_k * sqrt(y'P K_k P y / tr(P K_k)) (Zhou et al. 2019, J Comput Graph Stat 28:350),
    monotone in the REML likelihood; its fixed point is the REML score equation y'PKPy = tr(PK).
    The update this replaced, y'PKPy / tr(PK) with no sigma2_k factor, is dimensionless: it gave the
    same variances whatever the phenotype's units (hunt 2026-09-30, uT4-genomics-16).

    Returns ``(sigma2 list, iterations, converged)``. ``kernels`` ending in the identity with one
    other kernel (the additive model) take an O(n)-per-iteration path through that kernel's
    eigendecomposition; anything else inverts V each iteration.
    """
    from scipy import linalg

    n = len(y)
    ones = np.ones((n, 1))
    floor = 1e-10 * max(float(np.var(y)), 1e-300)
    sigma2 = [max(float(s), floor) for s in init]

    if len(kernels) == 2 and np.array_equal(kernels[1], np.eye(n)):
        lam, U = linalg.eigh(kernels[0])
        lam = np.clip(lam, 0.0, None)
        y_r, x_r = U.T @ y, U.T @ np.ones(n)
        for it in range(1, max_iter + 1):
            d = sigma2[0] * lam + sigma2[1]
            Py = y_r / d - (x_r / d) * float(x_r @ (y_r / d)) / float(x_r @ (x_r / d))
            xx = float(x_r @ (x_r / d))
            tr_pk = float(np.sum(lam / d) - np.sum(lam * x_r**2 / d**2) / xx)
            tr_p = float(np.sum(1.0 / d) - np.sum(x_r**2 / d**2) / xx)
            new = [
                max(sigma2[0] * float(np.sqrt(max(float(np.sum(lam * Py**2)), 0.0) / tr_pk)), floor),
                max(sigma2[1] * float(np.sqrt(max(float(np.sum(Py**2)), 0.0) / tr_p)), floor),
            ]
            change = max(abs(a - b) / max(b, floor) for a, b in zip(new, sigma2, strict=True))
            sigma2 = new
            if change < tol:
                return sigma2, it, True
        return sigma2, max_iter, False

    for it in range(1, max_iter + 1):
        V = sum(s * K for s, K in zip(sigma2, kernels, strict=True))
        V_inv = linalg.inv(V)
        V_inv_1 = V_inv @ ones
        P = V_inv - (V_inv_1 @ V_inv_1.T) / float(ones.T @ V_inv_1)
        Py = P @ y
        new = [
            # trace(P K) as sum(P * K): the kernels are symmetric, and this is O(n^2) not O(n^3).
            max(s * float(np.sqrt(max(Py @ K @ Py, 0.0) / float(np.sum(P * K)))), floor)
            for s, K in zip(sigma2, kernels, strict=True)
        ]
        change = max(abs(a - b) / max(b, floor) for a, b in zip(new, sigma2, strict=True))
        sigma2 = new
        if change < tol:
            return sigma2, it, True
    return sigma2, max_iter, False


def _report_every(n_iterations):
    """Iterations between progress lines: n // 5 was a modulo by zero for n < 5 (uT4-genomics-30)."""
    return max(1, int(n_iterations) // 5)


def bayesian_finemapping_with_deep_vi(
    gwas_summary_path,
    ld_matrix,
    n_iterations=5000,
    learning_rate=0.01,
    hidden_dim=64,
    credible_threshold=0.95,
    output_dir=None,
):
    """Performs Bayesian fine-mapping from GWAS summary statistics using deep variational inference.

    This function implements a deep neural network-based variational inference approach to compute
    posterior inclusion probabilities (PIPs) for putative causal variants from GWAS summary
    statistics and linkage disequilibrium (LD) information, and a credible set for one causal signal.

    The PIPs are the deep-VI model's mean-field posterior. Under strong LD it tends to give an LD
    block's whole signal to one member, which need not be the causal variant, and a strong hit's
    PIP is 1.0 in float32. The credible set therefore does not use them: it is the single-effect
    (SuSiE with L = 1) posterior, alpha_i proportional to prior_i x BF_i, with equal priors and each
    Bayes factor from the variant's marginal z-score under a N(0, 50) prior on the causal z. For one
    effect that Bayes factor does not depend on the LD matrix. One set is reported, for one causal
    signal: a region with several independent signals is not split into one set per signal, and the
    log reports the set's purity (smallest |r| between members) so a set that spans unlinked
    variants is visible.

    Parameters
    ----------
    gwas_summary_path : str
        Path to CSV or TSV file containing GWAS summary statistics. Expected columns:
        - 'variant_id': Identifier for each variant
        - 'effect_size': Effect size (beta) for each variant
        - 'pvalue': P-value for each variant
        - 'se': Standard error for each variant (optional)

    ld_matrix : numpy.ndarray
        Linkage disequilibrium matrix with pairwise correlations between variants.

    n_iterations : int, optional
        Number of training iterations for the variational inference algorithm.
        Default is 5000.

    learning_rate : float, optional
        Learning rate for the optimization algorithm. Default is 0.01.

    hidden_dim : int, optional
        Hidden dimension size for the neural network. Default is 64.

    credible_threshold : float, optional
        Threshold for defining the credible set (e.g., 0.95 for a 95% credible set).
        Default is 0.95. The set is the smallest group of top variants whose shares of one
        causal signal reach the threshold; a variant's share (``alpha`` in the results) is its
        single-effect posterior, prior x Bayes factor from its marginal z-score, normalised.

    output_dir : str, optional
        Directory for the result CSVs and the plot. Default: the run's output directory
        (``$SOG_WORK_DIR`` when set).

    Returns
    -------
    str
        A detailed research log of the fine-mapping analysis including:
        - Number of variants analyzed
        - Top variants ranked by single-effect share, with the deep-VI PIPs beside them
        - Credible set variants and the set's purity
        - Visualizations of the posterior distributions
        The results CSV holds ``pip`` and ``log_posterior_odds`` (the deep-VI model) and ``log_bf``
        and ``alpha`` (the single-effect model); the credible-set CSV adds ``cumulative_alpha``.

    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from torch import nn, optim

    # Initialize the research log
    log = []
    log.append(
        f"# Bayesian Fine-mapping Analysis with Deep Variational Inference - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    if int(n_iterations) < 1:
        return f"Error: n_iterations must be at least 1 (got {n_iterations})."
    log.append("\n## Data Preprocessing")

    # Load data from file
    try:
        if gwas_summary_path.endswith(".csv"):
            gwas_summary = pd.read_csv(gwas_summary_path)
        elif gwas_summary_path.endswith((".tsv", ".txt")):
            gwas_summary = pd.read_csv(gwas_summary_path, sep="\t")
        else:
            log.append("Error: Unsupported file format. Please provide a CSV or TSV file.")
            return "\n".join(log)
        log.append(f"Successfully loaded GWAS summary data from {gwas_summary_path}")
    except Exception as e:
        log.append(f"Error loading GWAS summary data: {str(e)}")
        return "\n".join(log)

    # Check input data
    if gwas_summary is None:
        log.append("Error: Failed to load GWAS summary data.")
        return "\n".join(log)

    if ld_matrix is None:
        log.append("Error: LD matrix is required for fine-mapping analysis.")
        return "\n".join(log)

    n_variants = len(gwas_summary)
    log.append(f"Analyzing {n_variants} genetic variants")

    # Check if LD matrix dimensions match the number of variants
    if ld_matrix.shape[0] != n_variants or ld_matrix.shape[1] != n_variants:
        log.append(f"Error: LD matrix dimensions ({ld_matrix.shape}) do not match number of variants ({n_variants})")
        return "\n".join(log)

    # Prepare data for analysis
    log.append("\nPreparing data for analysis...")

    # Compute Z-scores if not already present
    if "z_score" not in gwas_summary.columns:
        log.append("Computing Z-scores from effect sizes and standard errors...")
        if "se" in gwas_summary.columns:
            gwas_summary["z_score"] = gwas_summary["effect_size"] / gwas_summary["se"]
        else:
            # Approximate Z-scores from p-values
            log.append("Standard errors not available, approximating Z-scores from p-values...")
            # Convert p-values to Z-scores (two-sided test)
            from scipy.stats import norm

            # isf(p/2), not ppf(1 - p/2): 1 - p/2 rounds to 1.0 below p ~ 1e-16 and ppf(1.0) is inf,
            # so the strongest GWAS hits became inf and turned every PIP into NaN; and sign(),
            # not abs()/x, which is 0/0 for a zero effect (hunt 2026-09-30, uT4-genomics-1).
            gwas_summary["z_score"] = np.sign(gwas_summary["effect_size"]) * norm.isf(gwas_summary["pvalue"] / 2)

    # The credible set's Bayes factors are read from these z-scores. A p-value of 0 or a zero standard
    # error gives an infinite one, which the training turns into NaN PIPs (hunt 2026-09-30, uT4-genomics-1).
    try:
        z_values = gwas_summary["z_score"].to_numpy(dtype=float)
    except (TypeError, ValueError) as e:
        log.append(f"Error: the z-scores are not numeric: {e}")
        return "\n".join(log)
    bad_z = ~np.isfinite(z_values)
    if bad_z.any():
        examples = ", ".join(map(str, gwas_summary["variant_id"][bad_z][:5]))
        log.append(
            f"Error: {int(bad_z.sum())} variant(s) have a missing or infinite z-score ({examples}); a p-value of 0 "
            "or a zero or missing standard error gives one. Supply a finite 'z_score' column, or 'se' with "
            "'effect_size'."
        )
        return "\n".join(log)

    # Convert data to tensors
    z_scores = torch.FloatTensor(z_values)
    ld_tensor = torch.FloatTensor(ld_matrix)

    log.append(f"Processed {len(z_scores)} z-scores from GWAS summary")
    log.append("LD matrix shape: " + str(ld_matrix.shape))

    # Define the variational inference model
    class VariationalFineMapping(nn.Module):
        def __init__(self, n_variants, hidden_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_variants, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            # Output log alpha parameters for the Bernoulli variables
            self.log_alpha = nn.Linear(hidden_dim, n_variants)

        def logits(self, x):
            # Log posterior odds of inclusion, one per variant.
            return self.log_alpha(self.encoder(x))

        def forward(self, x):
            # Apply sigmoid to get inclusion probabilities
            return torch.sigmoid(self.logits(x))

        def elbo_loss(self, z_scores, ld_matrix, pips, n_samples=10):
            # The data term was averaged over torch.bernoulli samples, which carry no gradient, so
            # only the 0.01*sum(pips) prior trained and every PIP went to 0 whatever the GWAS said.
            # It is now the exact expectation under q(s), and the prior is the Bernoulli KL an ELBO
            # needs (hunt 2026-09-30, uT4-genomics-1). n_samples is kept for callers; it is unused.
            return _finemapping_neg_elbo(z_scores, ld_matrix, pips, prior_pi, torch.log)

    # Initialize model, optimizer and training
    log.append("\n## Initializing deep variational inference model")
    # One causal variant expected a priori (the usual fine-mapping default).
    prior_pi = 1.0 / max(n_variants, 2)
    log.append(f"Prior inclusion probability per variant: {prior_pi:.4g}")
    model = VariationalFineMapping(n_variants, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    log.append("\n## Training variational inference model")
    losses = []
    report_every = _report_every(n_iterations)

    for i in range(n_iterations):
        optimizer.zero_grad()
        pips = model(z_scores)
        loss = model.elbo_loss(z_scores, ld_tensor, pips)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % report_every == 0:
            log.append(f"  Iteration {i + 1}/{n_iterations}, Loss: {loss.item():.4f}")

    # Get final posterior inclusion probabilities, as logits: a strong hit's PIP is exactly 1.0 in
    # float32. And .tolist(), not .numpy(): in the full env (torch 2.2 built for NumPy 1, beside NumPy
    # 2.2) .numpy() raises "Numpy is not available" (hunt 2026-09-30, uT4-genomics-1).
    from scipy.special import expit

    with torch.no_grad():
        final_log_odds = np.asarray(model.logits(z_scores).tolist(), dtype=float)

    # Create DataFrame with results
    results_df = gwas_summary.copy()
    results_df["pip"] = expit(final_log_odds)
    results_df["log_posterior_odds"] = final_log_odds

    # Generate credible sets. The set is the single-effect posterior from the z-scores, not the deep-VI
    # fit: under LD the mean-field fit put a block's whole signal on one member, often a tag, and
    # saturated float32 logits split two hits by seed (hunt 2026-09-30, uT4-genomics-1).
    log.append("\n## Generating credible sets")
    log.append(
        "Credible set for one causal signal, from the single-effect model (SuSiE with L = 1), not from the "
        "deep-VI PIPs: each variant's share is alpha_i = prior_i x BF_i / sum_j prior_j x BF_j, with equal "
        f"priors and BF_i from the variant's marginal z-score under a N(0, {_SER_PRIOR_VARIANCE:g}) prior on "
        "the causal z. For one causal effect BF_i depends on z_i alone, so neither the LD matrix nor the "
        "deep-VI fit enters the set. One set is reported: a region with more than one independent signal "
        "is not split into one set per signal."
    )
    try:
        credible_set = _credible_set(results_df, credible_threshold)
    except ValueError as e:
        log.append(f"Error: no credible set can be formed: {e}")
        return "\n".join(log)
    results_df["log_bf"] = _single_effect_log_bf(z_values)
    results_df["alpha"] = _single_effect_alpha(results_df["log_bf"].to_numpy())
    results_df = results_df.sort_values("alpha", ascending=False, kind="mergesort")
    purity = _credible_set_purity(ld_matrix, gwas_summary.index.get_indexer(credible_set.index))

    log.append(f"Identified {len(credible_set)} variants in the {credible_threshold * 100}% credible set")
    log.append(f"Purity of the set (smallest |r| between two members): {purity:.3f}")
    if purity < 0.5:
        log.append(
            "Warning: the set's members are not in LD with one another (purity below 0.5), which one causal "
            "signal does not produce. The region may carry more than one signal, and this single-effect set "
            "cannot tell them apart."
        )

    # Save results to files, in output_dir (default: the run's output directory) under a run id. They
    # went to the process CWD under a per-second timestamp and were reported as bare names
    # (hunt 2026-09-30, uT4-genomics-29).
    if output_dir is None:
        from spatialomicsgym.paths import tool_output_root

        output_dir = tool_output_root()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    results_file = os.path.abspath(os.path.join(output_dir, f"finemapping_results_{timestamp}.csv"))
    credible_set_file = os.path.abspath(os.path.join(output_dir, f"credible_set_{timestamp}.csv"))

    _replace_into(results_file, lambda p: results_df.to_csv(p, index=False))
    _replace_into(credible_set_file, lambda p: credible_set.to_csv(p, index=False))

    log.append(f"\nFull results saved to: {results_file}")
    log.append(f"Credible set saved to: {credible_set_file}")

    # Summary of top variants
    log.append("\n## Top variants by share of the signal (alpha), with the deep-VI PIP")
    for i, (_, row) in enumerate(results_df.head(10).iterrows()):
        log.append(
            f"  {i + 1}. Variant: {row['variant_id']}, alpha: {row['alpha']:.4f}, deep-VI PIP: {row['pip']:.4f}, "
            f"P-value: {row['pvalue']:.2e}"
        )

    log.append("\n## Variants in the credible set")
    for i, (_, row) in enumerate(credible_set.iterrows()):
        log.append(
            f"  {i + 1}. Variant: {row['variant_id']}, alpha: {row['alpha']:.4f}, "
            f"cumulative: {row['cumulative_alpha']:.4f}, deep-VI PIP: {row['pip']:.4f}"
        )

    # Create a simple visualization of the deep-VI PIPs, sorted by PIP as the axis says (the table is
    # sorted by alpha).
    try:
        by_pip = results_df.sort_values("pip", ascending=False, kind="mergesort")
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(by_pip[:50])), by_pip["pip"][:50])
        plt.xlabel("Variant index (sorted by PIP)")
        plt.ylabel("Posterior Inclusion Probability")
        plt.title("Top 50 variants by PIP")
        plot_file = os.path.abspath(os.path.join(output_dir, f"pip_plot_{timestamp}.png"))
        _replace_into(plot_file, lambda p: plt.savefig(p, format="png"))
        plt.close()
        log.append(f"\nPlot of PIPs saved to: {plot_file}")
    except Exception as e:
        log.append(f"\nCould not create visualization: {str(e)}")

    log.append("\n## Analysis complete")

    return "\n".join(log)


def analyze_cas9_mutation_outcomes(
    reference_sequences,
    edited_sequences,
    cell_line_info=None,
    output_prefix="cas9_mutation_analysis",
):
    """Analyzes and categorizes mutations induced by Cas9 at target sites.

    Parameters
    ----------
    reference_sequences : dict
        Dictionary mapping sequence IDs to reference DNA sequences (strings)
    edited_sequences : dict of dict
        Nested dictionary: {sequence_id: {read_id: sequence}}
        Contains the edited/mutated sequences for each reference
    cell_line_info : dict, optional
        Dictionary mapping sequence IDs to cell line information (e.g., wildtype, knockout gene)
    output_prefix : str, optional
        Prefix for output files. A bare prefix (no directory part) is written under the run's
        output directory (``$SOG_WORK_DIR`` when set).

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    from collections import defaultdict

    from Bio import pairwise2

    # Initialize results storage
    results = []
    mutation_counts = defaultdict(lambda: defaultdict(int))

    # Define mutation categories
    categories = {
        "no_mutation": "No mutation detected",
        "short_deletion": "Short deletion (1-10 bp)",
        "medium_deletion": "Medium deletion (11-30 bp)",
        "long_deletion": "Long deletion (>30 bp)",
        "single_insertion": "Single base insertion",
        "longer_insertion": "Longer insertion (>1 bp)",
        "indel": "Insertion and deletion",
    }

    log = "# Cas9-Induced Mutation Outcome Analysis\n\n"
    log += "## Analysis Steps:\n\n"
    log += "1. Loading and processing sequence data\n"
    log += f"2. Analyzing {len(reference_sequences)} target sites\n"

    # Process each reference sequence and its edited versions
    for seq_id, ref_seq in reference_sequences.items():
        cell_line = cell_line_info.get(seq_id, "Unknown") if cell_line_info else "Unknown"
        log += f"\n### Processing target site: {seq_id} (Cell line: {cell_line})\n"

        site_results = []
        site_mutation_counts = defaultdict(int)
        total_reads = len(edited_sequences.get(seq_id, {}))

        if total_reads == 0:
            log += f"No edited sequences found for {seq_id}\n"
            continue

        log += f"Analyzing {total_reads} sequence reads...\n"

        # Process each edited sequence for this reference
        for read_id, edited_seq in edited_sequences.get(seq_id, {}).items():
            # Perform sequence alignment
            alignments = pairwise2.align.globalms(ref_seq, edited_seq, 2, -1, -2, -0.5)

            if not alignments:
                log += f"Warning: Could not align read {read_id}\n"
                continue

            best_alignment = alignments[0]
            ref_aligned, edited_aligned, score, start, end = best_alignment

            # Analyze mutations
            deletions = []
            insertions = []
            del_count = 0
            ins_count = 0

            i, j = 0, 0
            while i < len(ref_aligned) and j < len(edited_aligned):
                if ref_aligned[i] == "-":  # Insertion in edited sequence
                    ins_start = j
                    while i < len(ref_aligned) and ref_aligned[i] == "-":
                        i += 1
                        j += 1
                    insertions.append((ins_start, j - ins_start))
                    ins_count += j - ins_start
                elif edited_aligned[j] == "-":  # Deletion in edited sequence
                    del_start = i
                    while j < len(edited_aligned) and edited_aligned[j] == "-":
                        i += 1
                        j += 1
                    deletions.append((del_start, i - del_start))
                    del_count += i - del_start
                else:
                    i += 1
                    j += 1

            # Categorize mutation
            mutation_type = "no_mutation"
            if del_count > 0 and ins_count > 0:
                mutation_type = "indel"
            elif del_count > 0:
                if del_count <= 10:
                    mutation_type = "short_deletion"
                elif del_count <= 30:
                    mutation_type = "medium_deletion"
                else:
                    mutation_type = "long_deletion"
            elif ins_count > 0:
                mutation_type = "single_insertion" if ins_count == 1 else "longer_insertion"

            # Add to results
            site_results.append(
                {
                    "sequence_id": seq_id,
                    "read_id": read_id,
                    "cell_line": cell_line,
                    "mutation_type": mutation_type,
                    "deletion_count": del_count,
                    "insertion_count": ins_count,
                }
            )

            site_mutation_counts[mutation_type] += 1
            mutation_counts[cell_line][mutation_type] += 1

        # Calculate percentages for this site
        log += "\nMutation distribution for this target site:\n"
        for mut_type, count in site_mutation_counts.items():
            percentage = (count / total_reads) * 100
            log += f"- {categories[mut_type]}: {count} reads ({percentage:.1f}%)\n"

        # Add site results to overall results
        results.extend(site_results)

    # Create results dataframe and save to CSV. A bare prefix lands in the run's output directory and
    # the log names absolute paths (hunt 2026-09-30, uT4-genomics-29).
    output_prefix = _under_output_root(output_prefix)
    results_df = pd.DataFrame(results)
    output_file = f"{output_prefix}_detailed_results.csv"
    _replace_into(output_file, lambda p: results_df.to_csv(p, index=False))

    # Create summary dataframe
    summary_data = []
    for cell_line, mut_counts in mutation_counts.items():
        total = sum(mut_counts.values())
        for mut_type, count in mut_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            summary_data.append(
                {
                    "cell_line": cell_line,
                    "mutation_type": mut_type,
                    "count": count,
                    "percentage": percentage,
                }
            )

    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{output_prefix}_summary.csv"
    _replace_into(summary_file, lambda p: summary_df.to_csv(p, index=False))

    # Add summary to log
    log += "\n## Overall Results Summary\n\n"
    log += f"Total sequences analyzed: {len(results)}\n"
    log += f"Detailed results saved to: {output_file}\n"
    log += f"Summary results saved to: {summary_file}\n\n"

    if cell_line_info:
        log += "### Mutation Distribution by Cell Line\n\n"
        for cell_line, mut_counts in mutation_counts.items():
            total = sum(mut_counts.values())
            if total == 0:
                continue

            log += f"#### {cell_line}\n"
            for mut_type, count in sorted(mut_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                log += f"- {categories[mut_type]}: {count} ({percentage:.1f}%)\n"
            log += "\n"

    return log


def analyze_crispr_genome_editing(original_sequence, edited_sequence, guide_rna, repair_template=None):
    """Analyzes CRISPR-Cas9 genome editing results by comparing original and edited sequences.

    Parameters
    ----------
    original_sequence : str
        The original DNA sequence before CRISPR-Cas9 editing
    edited_sequence : str
        The DNA sequence after CRISPR-Cas9 editing
    guide_rna : str
        The CRISPR guide RNA (crRNA) sequence used for targeting
    repair_template : str, optional
        The homology-directed repair template sequence, if used

    Returns
    -------
    str
        A research log summarizing the CRISPR-Cas9 editing analysis, including identified
        mutations and characterization of the edited loci

    """
    import datetime

    from Bio import pairwise2
    from Bio.Seq import Seq

    log = []
    log.append(f"CRISPR-Cas9 Genome Editing Analysis - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append("=" * 70)

    # Step 1: Find the target site in the original sequence
    log.append("\n1. Identifying target site in original sequence")
    target_site = original_sequence.find(guide_rna)
    if target_site == -1:
        # Try with the reverse complement
        guide_rna_seq = Seq(guide_rna)
        rev_comp_guide = str(guide_rna_seq.reverse_complement())
        target_site = original_sequence.find(rev_comp_guide)
        if target_site != -1:
            log.append(f"   - Target site found at position {target_site} (using reverse complement of guide RNA)")
            guide_rna = rev_comp_guide
        else:
            log.append("   - Warning: Guide RNA sequence not found in original sequence")
            target_site = None
    else:
        log.append(f"   - Target site found at position {target_site}")

    # Step 2: Align sequences to identify mutations
    log.append("\n2. Aligning original and edited sequences to identify mutations")
    alignments = pairwise2.align.globalms(original_sequence, edited_sequence, 2, -1, -2, -0.5)
    best_alignment = alignments[0]

    # Extract aligned sequences
    aligned_orig = best_alignment[0]
    aligned_edit = best_alignment[1]

    # Find mutations. Each edit keeps its numeric position beside its text: on-target was decided by
    # looking for the window's numbers as substrings of the text, so with the guide at 10-30 an edit
    # at position 125 matched "12" and read as on-target (hunt 2026-09-30, uT4-genomics-6).
    mutations = []
    indels = []
    edit_positions = []

    for i in range(len(aligned_orig)):
        if aligned_orig[i] != aligned_edit[i]:
            orig_base = aligned_orig[i]
            edit_base = aligned_edit[i]

            # Skip gaps in counting the actual position
            actual_pos = len(aligned_orig[:i].replace("-", ""))

            if orig_base == "-":  # Insertion
                text = f"Insertion of {edit_base} at position {actual_pos}"
                indels.append(text)
            elif edit_base == "-":  # Deletion
                text = f"Deletion of {orig_base} at position {actual_pos}"
                indels.append(text)
            else:  # Substitution
                text = f"{orig_base}→{edit_base} at position {actual_pos}"
                mutations.append(text)
            edit_positions.append((actual_pos, text))

    # Log mutations
    if mutations:
        log.append("   - Substitutions detected:")
        for mutation in mutations:
            log.append(f"     * {mutation}")
    else:
        log.append("   - No substitutions detected")

    if indels:
        log.append("   - Insertions/Deletions detected:")
        for indel in indels:
            log.append(f"     * {indel}")
    else:
        log.append("   - No insertions or deletions detected")

    # Step 3: Check if editing occurred near the target site
    if target_site is not None:
        log.append("\n3. Analyzing mutations relative to target site")
        target_end = target_site + len(guide_rna)
        # Include some buffer
        on_target_edits = [text for pos, text in edit_positions if target_site - 3 <= pos < target_end + 3]

        if on_target_edits:
            log.append("   - On-target edits detected near guide RNA binding site:")
            for edit in on_target_edits:
                log.append(f"     * {edit}")
        else:
            log.append("   - No edits detected near guide RNA binding site")

    # Step 4: Check for homology-directed repair if template was provided
    if repair_template:
        log.append("\n4. Checking for homology-directed repair template incorporation")
        # Look for unique sequence markers from the repair template
        template_len = len(repair_template)
        marker_size = min(10, template_len // 3)
        marker = repair_template[template_len // 2 - marker_size // 2 : template_len // 2 + marker_size // 2]

        if marker in edited_sequence and marker not in original_sequence:
            log.append(f"   - Repair template marker '{marker}' found in edited sequence")
            log.append("   - Homology-directed repair likely successful")
        else:
            log.append("   - No clear evidence of repair template incorporation")
            log.append("   - Editing likely resulted from non-homologous end joining (NHEJ)")

    # Step 5: Overall assessment
    log.append("\n5. Overall assessment")
    if mutations or indels:
        log.append("   - CRISPR-Cas9 editing appears successful")
        if target_site is not None and any(
            target_site <= pos < target_site + len(guide_rna) for pos, _text in edit_positions
        ):
            log.append("   - Edits occurred at the intended target site")
        else:
            log.append("   - Edits may have occurred outside the intended target site")
    else:
        log.append("   - No significant editing detected, CRISPR-Cas9 may not have been effective")

    return "\n".join(log)


def simulate_demographic_history(
    num_samples=10,
    sequence_length=100000,
    recombination_rate=1e-8,
    mutation_rate=1e-8,
    demographic_model="constant",
    demographic_params=None,
    coalescent_model="kingman",
    beta_coalescent_param=None,
    random_seed=None,
    output_file="simulated_sequences.vcf",
):
    """Simulate DNA sequences with specified demographic and coalescent histories using msprime.

    Parameters
    ----------
    num_samples : int
        Number of diploid individuals to sample; the VCF holds one column per individual and
        2 * num_samples sampled sequences in all
    sequence_length : int
        Length of the simulated sequence in base pairs
    recombination_rate : float
        Per-base recombination rate
    mutation_rate : float
        Per-base mutation rate
    demographic_model : str
        Type of demographic model to simulate. Options:
        - "constant": Constant population size
        - "bottleneck": Population bottleneck
        - "expansion": Population expansion
        - "contraction": Population contraction
        - "sawtooth": Sawtooth pattern of population size changes
    demographic_params : dict
        Parameters specific to the chosen demographic model. Supported formats::

            - For "constant": {"N": population size}
            - For "bottleneck": {
                "N_initial": initial pop size,
                "N_bottleneck": bottleneck pop size,
                "T_bottleneck": time of bottleneck (generations ago),
                "T_recovery": time of recovery (generations ago)
            }
            - For "expansion": {"N_initial": initial pop size, "N_final": final pop size, "T_expansion": time of expansion (generations ago)}
            - For "contraction": {"N_initial": initial pop size, "N_final": final pop size, "T_contraction": time of contraction (generations ago)}
            - For "sawtooth": {"N_values": list of population sizes, "times": list of times for changes}
    coalescent_model : str
        Type of coalescent model to use. Options:
        - "kingman": Standard Kingman coalescent
        - "beta": Beta-coalescent model
    beta_coalescent_param : float
        Parameter for beta-coalescent model (required if coalescent_model="beta")
    random_seed : int
        Seed for random number generator (for reproducibility)
    output_file : str
        Filename to save the simulated sequences (VCF format). A bare file name is written under
        the run's output directory (``$SOG_WORK_DIR`` when set).

    Returns
    -------
    str
        Research log summarizing the simulation parameters and results

    """
    import time

    import msprime

    start_time = time.time()
    log = []
    log.append("Demographic History Simulation using msprime")
    log.append("=============================================")
    log.append("Parameters:")
    log.append(f"  - Number of samples: {num_samples} diploid individuals ({2 * num_samples} sequences)")
    log.append(f"  - Sequence length: {sequence_length} bp")
    log.append(f"  - Recombination rate: {recombination_rate}")
    log.append(f"  - Mutation rate: {mutation_rate}")
    log.append(f"  - Demographic model: {demographic_model}")
    log.append(f"  - Coalescent model: {coalescent_model}")

    # Set up demographic model
    if demographic_params is None:
        demographic_params = {"N": 10000}  # Default to constant population of 10000

    # Every branch builds its own Demography. An unconditional add_population("pop0") here, followed
    # by the constant branch adding "pop0" again, made msprime raise "Duplicate population name" on
    # the zero-argument call (hunt 2026-09-30, uT4-genomics-4).
    if demographic_model == "constant":
        N = demographic_params.get("N", 10000)
        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=N)
        log.append(f"  - Constant population size: N = {N}")

    elif demographic_model == "bottleneck":
        N_initial = demographic_params.get("N_initial", 10000)
        N_bottleneck = demographic_params.get("N_bottleneck", 1000)
        T_bottleneck = demographic_params.get("T_bottleneck", 1000)
        T_recovery = demographic_params.get("T_recovery", 500)

        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=N_initial)
        demography.add_population_parameters_change(time=T_recovery, initial_size=N_bottleneck)
        demography.add_population_parameters_change(time=T_bottleneck, initial_size=N_initial)

        log.append("  - Bottleneck model:")
        log.append(f"    * Initial population size: {N_initial}")
        log.append(f"    * Bottleneck population size: {N_bottleneck}")
        log.append(f"    * Bottleneck time (generations ago): {T_bottleneck}")
        log.append(f"    * Recovery time (generations ago): {T_recovery}")

    elif demographic_model == "expansion":
        N_initial = demographic_params.get("N_initial", 1000)
        N_final = demographic_params.get("N_final", 10000)
        T_expansion = demographic_params.get("T_expansion", 1000)

        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=N_final)
        demography.add_population_parameters_change(time=T_expansion, initial_size=N_initial)

        log.append("  - Expansion model:")
        log.append(f"    * Initial population size: {N_initial}")
        log.append(f"    * Final population size: {N_final}")
        log.append(f"    * Expansion time (generations ago): {T_expansion}")

    elif demographic_model == "contraction":
        N_initial = demographic_params.get("N_initial", 10000)
        N_final = demographic_params.get("N_final", 1000)
        T_contraction = demographic_params.get("T_contraction", 1000)

        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=N_final)
        demography.add_population_parameters_change(time=T_contraction, initial_size=N_initial)

        log.append("  - Contraction model:")
        log.append(f"    * Initial population size: {N_initial}")
        log.append(f"    * Final population size: {N_final}")
        log.append(f"    * Contraction time (generations ago): {T_contraction}")

    elif demographic_model == "sawtooth":
        N_values = demographic_params.get("N_values", [10000, 5000, 15000, 7500])
        times = demographic_params.get("times", [500, 1000, 1500])

        if len(N_values) != len(times) + 1:
            raise ValueError("For sawtooth model, N_values should have one more element than times")

        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=N_values[0])

        # `t`, not `time`: the loop variable shadowed the time module, so time.time() below raised
        # after the VCF was already written (hunt 2026-09-30, uT4-genomics-4).
        for i, t in enumerate(times):
            demography.add_population_parameters_change(time=t, initial_size=N_values[i + 1])

        log.append("  - Sawtooth model:")
        log.append(f"    * Population sizes: {N_values}")
        log.append(f"    * Change times (generations ago): {times}")

    else:
        raise ValueError(f"Unknown demographic model: {demographic_model}")

    # Set up coalescent model
    model = None
    if coalescent_model == "kingman":
        model = msprime.StandardCoalescent()
        log.append("  - Using standard Kingman coalescent")
    elif coalescent_model == "beta":
        if beta_coalescent_param is None:
            beta_coalescent_param = 1.5
        model = msprime.BetaCoalescent(alpha=beta_coalescent_param)
        log.append(f"  - Using Beta-coalescent with alpha = {beta_coalescent_param}")
    else:
        raise ValueError(f"Unknown coalescent model: {coalescent_model}")

    # Run simulation
    log.append("\nRunning simulation...")

    ts = msprime.sim_ancestry(
        samples=num_samples,
        recombination_rate=recombination_rate,
        sequence_length=sequence_length,
        demography=demography,
        model=model,
        random_seed=random_seed,
    )

    # Add mutations
    mts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=random_seed)

    # Save to VCF. A bare name lands in the run's output directory, and the file appears whole or not
    # at all (hunt 2026-09-30, uT4-genomics-29).
    output_file = _under_output_root(output_file)

    def _write_vcf(path):
        with open(path, "w") as vcf_file:
            mts.write_vcf(vcf_file)

    _replace_into(output_file, _write_vcf)

    # Calculate some basic statistics
    diversity = mts.diversity()
    num_sites = mts.num_sites
    num_trees = mts.num_trees

    # Log results
    end_time = time.time()
    runtime = end_time - start_time

    log.append(f"Simulation completed in {runtime:.2f} seconds")
    log.append("\nResults:")
    log.append(f"  - Number of segregating sites: {num_sites}")
    log.append(f"  - Number of trees in ARG: {num_trees}")
    log.append(f"  - Nucleotide diversity (π): {diversity:.6f}")
    log.append(f"  - Output saved to: {os.path.abspath(output_file)}")

    return "\n".join(log)


def identify_transcription_factor_binding_sites(sequence, tf_name, threshold=0.8, output_file=None):
    """Identifies binding sites for a specific transcription factor in a genomic sequence.

    Parameters
    ----------
    sequence : str
        The genomic DNA sequence to analyze
    tf_name : str
        Name of the transcription factor to search for (e.g., 'Hsf1', 'GATA1')
    threshold : float, optional
        Minimum relative score for reporting binding sites, (score - min) / (max - min) of the
        motif's PSSM (0.0-1.0, default: 0.8)
    output_file : str, optional
        Path to save the results (default: None, results only in log)

    Returns
    -------
    str
        Research log detailing the binding site identification process and results

    """
    import datetime
    import io

    from Bio import motifs
    from Bio.Seq import Seq

    from spatialomicsgym.utils.http_client import request_json, request_text

    log = f"# Transcription Factor Binding Site Analysis: {tf_name}\n"
    log += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Step 1: Get the PWM for the transcription factor from JASPAR database
    log += "## Step 1: Retrieving transcription factor PWM\n"

    try:
        # Search for the TF in JASPAR database
        tf_data = request_json(
            "https://jaspar.genereg.net/api/v1/matrix/",
            allowed_hosts=_JASPAR_HOSTS,
            params={"name": tf_name},
            timeout=30,
        )

        if not tf_data["results"] or len(tf_data["results"]) == 0:
            log += f"No PWM found for {tf_name} in JASPAR database.\n"
            return log

        # Get the first match's ID
        matrix_id = tf_data["results"][0]["matrix_id"]
        log += f"Found PWM with ID: {matrix_id}\n"

        # Retrieve the PWM
        pwm_url = f"https://jaspar.genereg.net/api/v1/matrix/{matrix_id}.pfm"
        pwm_text = request_text(pwm_url, allowed_hosts=_JASPAR_HOSTS, timeout=30)

        # Parse the PWM
        handle = io.StringIO(pwm_text)
        motif = motifs.read(handle, "jaspar")
        log += f"Successfully retrieved PWM for {tf_name}\n"

        # Calculate position-specific scoring matrix (PSSM). Pseudocounts keep a zero count in the
        # PFM from becoming a -inf log-odds, which made pssm.min -inf and every relative score NaN
        # (hunt 2026-09-30, uT4-genomics-5).
        motif.pseudocounts = 0.5
        pssm = motif.pssm
        # pssm.length is the motif length; len(pssm) is the alphabet size (4), which cut every
        # reported site to 4 nt (uT4-genomics-5).
        motif_length = pssm.length

        # Step 2: Scan the sequence for binding sites
        log += "\n## Step 2: Scanning sequence for binding sites\n"
        log += f"Sequence length: {len(sequence)} bp\n"

        # Find binding sites
        binding_sites = []
        max_score = pssm.max
        min_score = pssm.min
        # threshold is a relative score (0-1); pssm.search takes an absolute log-odds cutoff, so it is
        # converted here. Passed straight through, 0.8 was a near-floor bound for long motifs
        # (uT4-genomics-5).
        absolute_threshold = min_score + threshold * (max_score - min_score)
        log += f"Using relative score threshold: {threshold} (log-odds >= {absolute_threshold:.2f})\n"
        log += "Positions are 0-based starts on the forward strand.\n\n"

        for position, score in pssm.search(Seq(sequence), threshold=absolute_threshold):
            relative_score = (score - min_score) / (max_score - min_score)
            start, strand, site_seq = _tfbs_site(sequence, int(position), motif_length)

            binding_sites.append(
                {
                    "position": start,
                    "strand": strand,
                    "score": score,
                    "relative_score": relative_score,
                    "sequence": site_seq,
                }
            )

        # Step 3: Summarize results
        log += f"## Step 3: Results - Found {len(binding_sites)} potential binding sites\n\n"

        if binding_sites:
            # Sort by position
            binding_sites.sort(key=lambda x: x["position"])

            # Create a table of results
            log += "| Position | Strand | Sequence | Score | Relative Score |\n"
            log += "|----------|--------|----------|-------|---------------|\n"

            for site in binding_sites:
                log += f"| {site['position']} | {site['strand']} | {site['sequence']} | {site['score']:.2f} | {site['relative_score']:.2f} |\n"
        else:
            log += "No binding sites found meeting the threshold criteria.\n"

        # Save results to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(f"# {tf_name} binding sites in sequence\n")
                f.write("Position\tStrand\tSequence\tScore\tRelative Score\n")

                for site in binding_sites:
                    f.write(
                        f"{site['position']}\t{site['strand']}\t{site['sequence']}\t{site['score']:.2f}\t{site['relative_score']:.2f}\n"
                    )

            log += f"\nResults saved to file: {output_file}\n"

    except Exception as e:
        # "Error: " at line start, which the ReAct loop reads as a failed step (hunt 2026-09-30,
        # uT4-genomics-23). An HttpError already carries the prefix.
        message = str(e)
        log += "\n" + (
            message if message.startswith("Error: ") else f"Error: TF binding site analysis failed: {message}"
        )
        log += "\n"

    log += "\n## Analysis complete\n"
    return log


def fit_genomic_prediction_model(
    genotypes,
    phenotypes,
    fixed_effects=None,
    model_type="additive",
    output_file="genomic_prediction_results.csv",
):
    """Fit a linear mixed model for genomic prediction using genotype and phenotype data.

    Parameters
    ----------
    genotypes : numpy.ndarray
        Matrix of genotype data, with individuals in rows and markers in columns.
        Values are typically coded as 0, 1, 2 for additive models or with specific
        encoding for dominance effects.
    phenotypes : numpy.ndarray
        Vector or matrix of phenotype data, with individuals in rows and traits in columns.
    fixed_effects : numpy.ndarray, optional
        Matrix of fixed effects (e.g., environment, management), with individuals in rows
        and effects in columns.
    model_type : str, optional
        Type of genetic model to fit: "additive" or "additive_dominance".
    output_file : str, optional
        File name to save the results. A bare file name is written under the run's output
        directory (``$SOG_WORK_DIR`` when set).

    Returns
    -------
    str
        Research log summarizing the genomic prediction analysis, including model parameters,
        REML variance components, breeding values, and the in-sample fit (the correlation of the
        observed phenotype with its BLUP; not a cross-validated accuracy).

    """
    import pandas as pd
    from scipy import linalg

    # Initialize research log
    log = "# Multi-trait Genomic Prediction Analysis\n\n"
    log += f"Model type: {model_type}\n"

    # Basic validation
    n_individuals, n_markers = genotypes.shape
    n_pheno, n_traits = (phenotypes.shape[0], 1) if phenotypes.ndim == 1 else phenotypes.shape

    if n_individuals != n_pheno:
        raise ValueError(f"Number of individuals in genotypes ({n_individuals}) and phenotypes ({n_pheno}) don't match")

    log += f"Number of individuals: {n_individuals}\n"
    log += f"Number of markers: {n_markers}\n"
    log += f"Number of traits: {n_traits}\n\n"

    # Ensure phenotypes is 2D
    if phenotypes.ndim == 1:
        phenotypes = phenotypes.reshape(-1, 1)

    # Center genotypes (common preprocessing step)
    genotypes_centered = genotypes - np.mean(genotypes, axis=0)

    # Create genomic relationship matrix (G)
    if model_type == "additive":
        # Additive genomic relationship matrix
        G = np.dot(genotypes_centered, genotypes_centered.T) / n_markers
        log += "Constructed additive genomic relationship matrix (G)\n\n"
    elif model_type == "additive_dominance":
        # For additive-dominance model, we need both A (additive) and D (dominance) matrices
        # Assuming genotypes are coded as {0,1,2} for {aa,Aa,AA}
        # Create dominance matrix - simple implementation assuming standard coding
        dom_genotypes = np.zeros_like(genotypes)
        # Code heterozygotes (1) as 1, homozygotes (0,2) as 0 for dominance effects
        dom_genotypes[genotypes == 1] = 1
        dom_genotypes_centered = dom_genotypes - np.mean(dom_genotypes, axis=0)

        # Additive and dominance matrices
        G_a = np.dot(genotypes_centered, genotypes_centered.T) / n_markers
        G_d = np.dot(dom_genotypes_centered, dom_genotypes_centered.T) / n_markers
        log += "Constructed additive (G_a) and dominance (G_d) genomic relationship matrices\n\n"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Initialize results storage
    trait_results = []

    # Fit model for each trait
    for trait_idx in range(n_traits):
        trait_phenotypes = phenotypes[:, trait_idx]
        log += f"## Trait {trait_idx + 1} Analysis\n\n"

        # Handle fixed effects if provided
        if fixed_effects is not None:
            # Simple fixed effects adjustment - more complex models would use proper mixed model fitting
            X = fixed_effects
            # Fit fixed effects model
            beta = np.linalg.lstsq(X, trait_phenotypes, rcond=None)[0]
            # Adjust phenotypes for fixed effects
            y_adj = trait_phenotypes - X @ beta
            log += f"Applied adjustment for {X.shape[1]} fixed effects\n"
        else:
            y_adj = trait_phenotypes
            log += "No fixed effects provided\n"

        # Fit mixed model
        if model_type == "additive":
            # REML variance components, iterated to convergence (it was 5 iterations of an update
            # that ignored the phenotype's scale -- hunt 2026-09-30, uT4-genomics-16).
            (var_g, var_e), n_iter, converged = _reml_mm(
                y_adj, [G, np.eye(n_individuals)], [np.var(y_adj) * 0.5, np.var(y_adj) * 0.5]
            )
            log += (
                f"REML (MM) converged after {n_iter} iterations\n"
                if converged
                else f"Warning: REML (MM) did not converge in {n_iter} iterations; the variances are the last iterate\n"
            )

            # Calculate heritability
            heritability = var_g / (var_g + var_e)

            # BLUP solutions for breeding values
            V = var_g * G + var_e * np.eye(n_individuals)
            V_inv = linalg.inv(V)
            breeding_values = var_g * G @ V_inv @ y_adj

            # Predicted phenotypes
            predicted_phenotypes = breeding_values

            # In-sample fit: the correlation of the observed phenotype with its own BLUP. It was
            # labelled "Prediction accuracy", which reads as held-out accuracy (uT4-genomics-16).
            accuracy = np.corrcoef(trait_phenotypes, predicted_phenotypes)[0, 1]

            # Log results
            log += f"Estimated additive genetic variance: {var_g:.4f}\n"
            log += f"Estimated residual variance: {var_e:.4f}\n"
            log += f"Estimated heritability: {heritability:.4f}\n"
            log += f"In-sample fit (correlation of observed with BLUP, not cross-validated): {accuracy:.4f}\n\n"

            # Store results
            trait_result = {
                "trait": trait_idx + 1,
                "var_g": var_g,
                "var_e": var_e,
                "heritability": heritability,
                "accuracy": accuracy,
                "breeding_values": breeding_values,
                "predicted_phenotypes": predicted_phenotypes,
            }
            trait_results.append(trait_result)

        elif model_type == "additive_dominance":
            # Similar approach but with both additive and dominance effects
            # Initial variance component estimates
            var_a_init = np.var(y_adj) * 0.4  # additive variance
            var_d_init = np.var(y_adj) * 0.1  # dominance variance
            var_e_init = np.var(y_adj) * 0.5  # residual variance

            # REML variance components, iterated to convergence (uT4-genomics-16)
            (var_a, var_d, var_e), n_iter, converged = _reml_mm(
                y_adj, [G_a, G_d, np.eye(n_individuals)], [var_a_init, var_d_init, var_e_init]
            )
            log += (
                f"REML (MM) converged after {n_iter} iterations\n"
                if converged
                else f"Warning: REML (MM) did not converge in {n_iter} iterations; the variances are the last iterate\n"
            )

            # Calculate heritabilities
            narrow_heritability = var_a / (var_a + var_d + var_e)
            broad_heritability = (var_a + var_d) / (var_a + var_d + var_e)

            # BLUP solutions for breeding values and dominance deviations
            V = var_a * G_a + var_d * G_d + var_e * np.eye(n_individuals)
            V_inv = linalg.inv(V)
            breeding_values = var_a * G_a @ V_inv @ y_adj
            dominance_deviations = var_d * G_d @ V_inv @ y_adj

            # Predicted phenotypes
            predicted_phenotypes = breeding_values + dominance_deviations

            # Calculate accuracy
            accuracy = np.corrcoef(trait_phenotypes, predicted_phenotypes)[0, 1]

            # Log results
            log += f"Estimated additive genetic variance: {var_a:.4f}\n"
            log += f"Estimated dominance genetic variance: {var_d:.4f}\n"
            log += f"Estimated residual variance: {var_e:.4f}\n"
            log += f"Estimated narrow-sense heritability: {narrow_heritability:.4f}\n"
            log += f"Estimated broad-sense heritability: {broad_heritability:.4f}\n"
            log += f"In-sample fit (correlation of observed with BLUP, not cross-validated): {accuracy:.4f}\n\n"

            # Store results
            trait_result = {
                "trait": trait_idx + 1,
                "var_a": var_a,
                "var_d": var_d,
                "var_e": var_e,
                "narrow_heritability": narrow_heritability,
                "broad_heritability": broad_heritability,
                "accuracy": accuracy,
                "breeding_values": breeding_values,
                "dominance_deviations": dominance_deviations,
                "predicted_phenotypes": predicted_phenotypes,
            }
            trait_results.append(trait_result)

    # Save results to file
    results_df = pd.DataFrame()

    for i, trait_result in enumerate(trait_results):
        # Create individual-level results
        ind_data = {
            "individual": np.arange(1, n_individuals + 1),
            f"trait_{i + 1}_observed": phenotypes[:, i],
            f"trait_{i + 1}_predicted": trait_result["predicted_phenotypes"],
            f"trait_{i + 1}_breeding_value": trait_result["breeding_values"],
        }

        if model_type == "additive_dominance":
            ind_data[f"trait_{i + 1}_dominance_deviation"] = trait_result["dominance_deviations"]

        # Create or append to dataframe
        if i == 0:
            results_df = pd.DataFrame(ind_data)
        else:
            for key, value in ind_data.items():
                if key != "individual":  # Skip duplicating individual column
                    results_df[key] = value

    # Save to CSV. A bare name lands in the run's output directory and the log names the absolute path
    # (hunt 2026-09-30, uT4-genomics-29).
    output_file = _under_output_root(output_file)
    _replace_into(output_file, lambda p: results_df.to_csv(p, index=False))
    log += f"Results saved to {output_file}\n"

    return log


def perform_pcr_and_gel_electrophoresis(
    genomic_dna,
    forward_primer=None,
    reverse_primer=None,
    target_region=None,
    annealing_temp=58,
    extension_time=30,
    cycles=35,
    gel_percentage=2.0,
    output_prefix="pcr_result",
):
    """Performs PCR amplification of a target transgene and visualizes results using agarose gel electrophoresis.

    Parameters
    ----------
    genomic_dna : str
        Path to file containing genomic DNA sequence in FASTA format or the sequence itself
    forward_primer : str, optional
        Forward primer sequence. If not provided, will be designed based on target_region
    reverse_primer : str, optional
        Reverse primer sequence. If not provided, will be designed based on target_region
    target_region : tuple, optional
        Tuple of (start, end) positions for the target region in the genomic DNA
    annealing_temp : float, default=58
        Annealing temperature for PCR in °C
    extension_time : int, default=30
        Extension time in seconds
    cycles : int, default=35
        Number of PCR cycles
    gel_percentage : float, default=2.0
        Percentage of agarose gel
    output_prefix : str, default="pcr_result"
        Prefix for output files. A bare prefix (no directory part) is written under the run's
        output directory (``$SOG_WORK_DIR`` when set).

    Returns
    -------
    str
        Research log summarizing the PCR and gel electrophoresis procedures and results

    """
    import datetime

    import matplotlib.pyplot as plt
    import numpy as np
    from Bio import SeqIO
    from Bio.Seq import Seq

    log = f"PCR AMPLIFICATION AND GEL ELECTROPHORESIS LOG - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    log += "=" * 80 + "\n\n"

    # Step 1: Load the genomic DNA
    log += "STEP 1: PREPARING GENOMIC DNA\n"
    if os.path.isfile(genomic_dna):
        try:
            record = SeqIO.read(genomic_dna, "fasta")
            dna_sequence = str(record.seq)
            log += f"- Loaded genomic DNA from file: {genomic_dna}\n"
            log += f"- Sequence length: {len(dna_sequence)} bp\n"
        except Exception as e:
            log += f"- Error loading DNA file: {str(e)}\n"
            return log
    else:
        dna_sequence = genomic_dna
        log += "- Using provided DNA sequence\n"
        log += f"- Sequence length: {len(dna_sequence)} bp\n"

    log += "\n"

    # Step 2: Design or validate primers
    log += "STEP 2: PCR PRIMER PREPARATION\n"

    if forward_primer is None or reverse_primer is None:
        if target_region is None:
            log += "- Error: Either primers or target region must be provided\n"
            return log

        # Simple primer design based on target region
        start, end = target_region

        if forward_primer is None:
            # Take 20bp from the start of the target region for forward primer
            forward_primer = dna_sequence[start : start + 20]
            log += "- Designed forward primer based on target region\n"

        if reverse_primer is None:
            # Take 20bp from the end of the target region for reverse primer (reverse complement)
            reverse_seq = Seq(dna_sequence[end - 20 : end])
            reverse_primer = str(reverse_seq.reverse_complement())
            log += "- Designed reverse primer based on target region\n"

    log += f"- Forward primer: 5'-{forward_primer}-3' ({len(forward_primer)} bp)\n"
    log += f"- Reverse primer: 5'-{reverse_primer}-3' ({len(reverse_primer)} bp)\n"

    # Step 3: PCR Setup and Amplification
    log += "\nSTEP 3: PCR AMPLIFICATION\n"
    log += "- PCR reaction setup:\n"
    log += "  * Template DNA: Genomic DNA\n"
    log += f"  * Forward primer: 5'-{forward_primer}-3'\n"
    log += f"  * Reverse primer: 5'-{reverse_primer}-3'\n"
    log += f"  * Annealing temperature: {annealing_temp}°C\n"
    log += f"  * Extension time: {extension_time} seconds\n"
    log += f"  * Number of cycles: {cycles}\n"

    # Simulate PCR by finding binding sites and determining amplicon size
    amplicon_size = None
    amplicon_sequence = None

    # Find forward primer binding site
    fwd_pos = dna_sequence.find(forward_primer)
    if fwd_pos == -1:
        log += "- Warning: Forward primer binding site not found in sequence\n"

    # Find reverse primer binding site (need to search for reverse complement)
    rev_primer_seq = Seq(reverse_primer)
    rev_primer_rc = str(rev_primer_seq.reverse_complement())
    rev_pos = dna_sequence.find(rev_primer_rc)
    if rev_pos == -1:
        log += "- Warning: Reverse primer binding site not found in sequence\n"

    # If we found both binding sites, calculate amplicon size
    if fwd_pos != -1 and rev_pos != -1:
        if fwd_pos < rev_pos:
            amplicon_size = rev_pos + len(reverse_primer) - fwd_pos
            amplicon_sequence = dna_sequence[fwd_pos : rev_pos + len(reverse_primer)]
            log += "- PCR amplification successful\n"
            log += f"- Amplicon size: {amplicon_size} bp\n"
        else:
            log += "- Error: Primer binding sites are in incorrect orientation\n"
    # Simulate based on target region if provided
    elif target_region is not None:
        start, end = target_region
        amplicon_size = end - start + len(forward_primer) + len(reverse_primer)
        log += "- PCR amplification simulated based on target region\n"
        log += f"- Expected amplicon size: {amplicon_size} bp\n"
    else:
        log += "- PCR amplification failed: could not determine amplicon size\n"
        return log

    # Step 4: Gel Electrophoresis
    log += "\nSTEP 4: AGAROSE GEL ELECTROPHORESIS\n"
    log += f"- Prepared {gel_percentage}% agarose gel\n"
    log += "- Loaded PCR product alongside DNA ladder\n"
    log += "- Ran electrophoresis at 100V for 45 minutes\n"

    # Create a simulated gel image
    fig, ax = plt.subplots(figsize=(6, 8))

    # Draw gel lanes
    ax.add_patch(plt.Rectangle((0, 0), 6, 10, color="lightgray", alpha=0.5))

    # DNA Ladder (100bp increments)
    ladder_sizes = [100, 200, 300, 500, 700, 1000, 1500, 2000]
    ladder_positions = [10 - (np.log(size) / np.log(2000) * 8) for size in ladder_sizes]

    # Plot ladder
    for pos, size in zip(ladder_positions, ladder_sizes, strict=False):
        ax.add_patch(plt.Rectangle((0.5, pos - 0.1), 1, 0.2, color="black", alpha=0.8))
        ax.text(0.2, pos, f"{size}bp", fontsize=8, ha="right", va="center")

    # Plot sample band
    if amplicon_size:
        sample_position = 10 - (np.log(amplicon_size) / np.log(2000) * 8)
        ax.add_patch(plt.Rectangle((3.5, sample_position - 0.15), 1, 0.3, color="black", alpha=0.8))
        ax.text(
            4.5,
            sample_position,
            f"{amplicon_size}bp",
            fontsize=8,
            ha="left",
            va="center",
        )

    # Set up the plot
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 10)
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(["Ladder", "Sample"])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"{gel_percentage}% Agarose Gel")

    # Save the gel image. A bare prefix lands in the run's output directory and the log names absolute
    # paths (hunt 2026-09-30, uT4-genomics-29).
    output_prefix = _under_output_root(output_prefix)
    gel_image_path = f"{output_prefix}_gel.png"
    _replace_into(gel_image_path, lambda p: plt.savefig(p, format="png", dpi=300, bbox_inches="tight"))
    plt.close()

    log += f"- Gel image saved as: {gel_image_path}\n"

    # Results interpretation
    log += "\nRESULTS INTERPRETATION:\n"
    if amplicon_size:
        log += f"- Detected band at approximately {amplicon_size} bp\n"

        # Save amplicon sequence if available
        if amplicon_sequence:
            seq_file = f"{output_prefix}_amplicon.fasta"

            def _write_fasta(path):
                with open(path, "w") as f:
                    f.write(f">PCR_Amplicon_{amplicon_size}bp\n")
                    f.write(amplicon_sequence)

            _replace_into(seq_file, _write_fasta)
            log += f"- Amplicon sequence saved as: {seq_file}\n"
    else:
        log += "- No bands detected\n"

    return log


def analyze_protein_phylogeny(
    fasta_sequences,
    output_dir="./",
    alignment_method="clustalw",
    tree_method="iqtree",
):
    """Perform phylogenetic analysis on a set of protein sequences.

    This function takes protein sequences in FASTA format, performs multiple sequence alignment,
    constructs a phylogenetic tree, and visualizes the evolutionary relationships.

    Parameters
    ----------
    fasta_sequences : str
        Path to a FASTA file containing protein sequences or a string with FASTA-formatted sequences
    output_dir : str, optional
        Directory to save output files (default: current directory)
    alignment_method : str, optional
        Method for sequence alignment: "clustalw", "muscle", or "pre-aligned" (default: "clustalw")
    tree_method : str, optional
        Method for tree construction: "iqtree", which falls back to neighbor-joining when IQ-TREE
        fails (default: "iqtree"). The signature default was "fasttree", which no branch handled, so
        a default call aligned and then stopped (hunt 2026-09-30, uT4-genomics-13).

    Returns
    -------
    str
        Research log summarizing the phylogenetic analysis process

    """
    import datetime
    import subprocess
    import tempfile

    from Bio import AlignIO, Phylo, SeqIO

    # Bio.Align.Applications (ClustalwCommandline, MuscleCommandline) is gone from Biopython 1.86, the
    # version the full env pins, and importing it killed every call. The aligners are run directly
    # (hunt 2026-09-30, uT4-genomics-13).

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize log
    log = []
    log.append(f"Phylogenetic Analysis - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append("=" * 50)

    # Check if input is a file path or string content
    if os.path.isfile(fasta_sequences):
        input_file = fasta_sequences
        log.append(f"Using sequences from file: {input_file}")
    else:
        # Create temporary file with the string content
        temp_fasta = tempfile.NamedTemporaryFile(delete=False, suffix=".fasta", dir=output_dir)
        temp_fasta.write(fasta_sequences.encode())
        temp_fasta.close()
        input_file = temp_fasta.name
        log.append(f"Created temporary FASTA file from provided sequences: {input_file}")

    # Count sequences
    try:
        sequences = list(SeqIO.parse(input_file, "fasta"))
        log.append(f"Loaded {len(sequences)} protein sequences")
    except Exception as e:
        log.append(f"Warning: Could not parse sequences as FASTA: {str(e)}")
        # This might be pre-aligned data already
        sequences = []

    # Create filenames for outputs
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    alignment_file = os.path.join(output_dir, f"{base_name}_aligned.aln")
    tree_file = os.path.join(output_dir, f"{base_name}_tree.nwk")
    tree_image = os.path.join(output_dir, f"{base_name}_phylogeny.png")

    # Perform multiple sequence alignment
    log.append("\nStep 1: Multiple Sequence Alignment")

    # Special case for pre-aligned input
    if alignment_method.lower() == "pre-aligned":
        log.append("Using pre-aligned sequences")
        try:
            # If the input is already an alignment file, copy it to the alignment_file path
            with open(input_file) as src, open(alignment_file, "w") as dst:
                dst.write(src.read())
            log.append(f"Copied pre-aligned file to: {alignment_file}")
        except Exception as e:
            log.append(f"Error processing pre-aligned sequences: {str(e)}")
            return "\n".join(log)
    elif alignment_method.lower() == "clustalw":
        log.append("Using ClustalW for alignment")
        try:
            subprocess.run(
                ["clustalw", f"-infile={input_file}", f"-outfile={alignment_file}"],
                check=True,
                capture_output=True,
                text=True,
            )
            log.append("Alignment completed successfully")
        except (subprocess.CalledProcessError, OSError) as e:
            log.append(f"ClustalW alignment failed: {str(e)}")
            # Try alternative approach using MUSCLE if ClustalW fails
            alignment_method = "muscle"

    if alignment_method.lower() == "muscle":
        log.append("Using MUSCLE for alignment")
        try:
            subprocess.run(
                ["muscle", "-in", input_file, "-out", alignment_file],
                check=True,
                capture_output=True,
                text=True,
            )
            log.append("Alignment completed successfully")
        except (subprocess.CalledProcessError, OSError) as e:
            # No aligner, no alignment. The fallback here wrote one pairwise2 alignment per sequence
            # under a "CLUSTAL W" header -- rows of different lengths, not a multiple alignment -- and
            # the tree built from it was reported as the result (uT4-genomics-13).
            log.append(
                f"Error: MUSCLE alignment failed ({str(e)}), so no multiple sequence alignment was made. "
                "Install clustalw or muscle, or pass an aligned file with alignment_method='pre-aligned'."
            )
            return "\n".join(log)

    # Verify alignment file exists
    if not os.path.exists(alignment_file):
        log.append(f"Error: Alignment file {alignment_file} does not exist")
        return "\n".join(log)

    # Build phylogenetic tree
    log.append("\nStep 2: Phylogenetic Tree Construction")

    if tree_method.lower() == "iqtree":
        log.append("Using IQ-TREE for phylogenetic tree construction")
        try:
            # An argv list, not a shell string: a path with a space or a shell metacharacter broke it.
            subprocess.run(
                [
                    "iqtree",
                    "-s",
                    alignment_file,
                    "-m",
                    "LG",
                    "-bb",
                    "1000",
                    "-pre",
                    os.path.join(output_dir, base_name),
                ],
                check=True,
                capture_output=True,
            )
            # IQ-TREE creates files with .treefile extension
            iqtree_file = os.path.join(output_dir, f"{base_name}.treefile")
            if os.path.exists(iqtree_file):
                # Rename to our standard name
                os.rename(iqtree_file, tree_file)
            log.append("Tree construction completed successfully")
        except Exception as e:
            log.append(f"Error during IQ-TREE execution: {str(e)}")
            log.append("Falling back to neighbor-joining method")

            try:
                from Bio.Phylo.TreeConstruction import (
                    DistanceCalculator,
                    DistanceTreeConstructor,
                )

                # Try to read the alignment in different formats
                alignment = None
                for format in ["clustal", "fasta"]:
                    try:
                        alignment = AlignIO.read(alignment_file, format)
                        break
                    except Exception:
                        continue

                if alignment is None:
                    # A fixed four-leaf tree was written here and reported as the result
                    # (hunt 2026-09-30, uT4-genomics-13).
                    log.append(
                        "Error: could not parse the alignment file in any supported format, so no tree was built."
                    )
                    return "\n".join(log)
                else:
                    # Calculate the distance matrix
                    calculator = DistanceCalculator("identity")
                    dm = calculator.get_distance(alignment)

                    # Construct the tree
                    constructor = DistanceTreeConstructor()
                    tree = constructor.nj(dm)

                    # Write the tree to file
                    Phylo.write(tree, tree_file, "newick")
                    log.append("Created tree using neighbor-joining method")
            except Exception as e:
                # The placeholder tree again, reported as the result (uT4-genomics-13).
                log.append(f"Error: neighbor-joining fallback failed ({str(e)}), so no tree was built.")
                return "\n".join(log)
    else:
        log.append(f"Error: unsupported tree method {tree_method!r}; use 'iqtree' (falls back to neighbor-joining).")
        return "\n".join(log)

    # Verify tree file exists
    if not os.path.exists(tree_file):
        log.append(f"Error: Tree file {tree_file} does not exist")
        return "\n".join(log)

    # Visualize the tree
    log.append("\nStep 3: Phylogenetic Tree Visualization")
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt

        tree = Phylo.read(tree_file, "newick")
        fig = plt.figure(figsize=(10, len(sequences) * 0.3 if sequences else 5))
        axes = fig.add_subplot(1, 1, 1)
        Phylo.draw(tree, axes=axes, do_show=False)
        plt.savefig(tree_image, dpi=300, bbox_inches="tight")
        plt.close()
        log.append(f"Tree visualization saved to: {tree_image}")
    except Exception as e:
        log.append(f"Error during tree visualization: {str(e)}")

    # Summary
    log.append("\nSummary:")
    log.append(f"- Input sequences: {len(sequences) if sequences else 'pre-aligned data'}")
    log.append(f"- Alignment file: {alignment_file}")
    log.append(f"- Phylogenetic tree file: {tree_file}")
    log.append(f"- Tree visualization: {tree_image}")

    return "\n".join(log)
