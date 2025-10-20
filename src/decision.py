"""
Decision-theoretic value mapping and Expected Value of Information.
Implements conditional risk for threshold-based policies and EVI approximation.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, List
import warnings


def conditional_risk(mu: float, sigma: float,
                    tau: float, L_FP: float, L_FN: float, L_TP: float,
                    L_TN: float = 0.0) -> float:
    """
    Compute Bayes-optimal conditional risk for threshold decision.

    Given posterior N(μ, σ²), decision rule:
        - Act (maintain) if P(x > τ | data) > p_T
        - Don't act otherwise

    where p_T = L_FP / (L_FP + L_FN - L_TP)

    Args:
        mu: Posterior mean
        sigma: Posterior std deviation
        tau: Decision threshold (e.g., IRI limit)
        L_FP: False positive cost
        L_FN: False negative cost
        L_TP: True positive cost
        L_TN: True negative cost (default 0)

    Returns:
        risk: Expected loss under Bayes-optimal action
    """
    if sigma <= 0:
        # Degenerate case: certain knowledge
        if mu > tau:
            return L_TP  # Maintain (correct)
        else:
            return L_TN  # Don't maintain (correct)

    # Posterior failure probability
    p_f = 1.0 - norm.cdf((tau - mu) / sigma)

    # Probability threshold for action
    denom = L_FP + L_FN - L_TP
    if abs(denom) < 1e-10:
        warnings.warn("Near-singular decision cost matrix")
        p_T = 0.5
    else:
        p_T = L_FP / denom

    # Conditional risk for each action
    risk_no_action = p_f * L_FN + (1 - p_f) * L_TN
    risk_action = p_f * L_TP + (1 - p_f) * L_FP

    # Bayes-optimal risk
    return min(risk_no_action, risk_action)


def expected_loss(mu_post: np.ndarray,
                 sigma_post: np.ndarray,
                 decision_config,
                 test_indices: np.ndarray = None) -> float:
    """
    Compute expected economic loss averaged over test set.

    Args:
        mu_post: Posterior means (n,)
        sigma_post: Posterior std deviations (n,)
        decision_config: DecisionConfig object
        test_indices: Subset to evaluate (if None, use all)

    Returns:
        expected_loss: Mean conditional risk over test set (GBP)
    """
    if test_indices is None:
        test_indices = np.arange(len(mu_post))

    risks = np.array([
        conditional_risk(
            mu_post[i], sigma_post[i],
            decision_config.tau_iri,
            decision_config.L_FP_gbp,
            decision_config.L_FN_gbp,
            decision_config.L_TP_gbp,
            decision_config.L_TN_gbp
        )
        for i in test_indices
    ])

    return risks.mean()


def evi_monte_carlo(Q_pr, mu_pr, H, R_diag, decision_config,
                   n_samples: int = 500,
                   rng: np.random.Generator = None) -> float:
    """
    Approximate Expected Value of Information via Monte Carlo.

    Samples from prior, generates observations, computes posterior,
    evaluates Bayes action loss, and averages.

    Args:
        Q_pr: Prior precision
        mu_pr: Prior mean
        H: Observation matrix for selected sensors
        R_diag: Noise variances
        decision_config: Decision parameters
        n_samples: Number of MC samples
        rng: Random generator

    Returns:
        evi: Expected reduction in decision loss (GBP)
    """
    from spatial_field import sample_gmrf
    from inference import compute_posterior, compute_posterior_variance_diagonal

    if rng is None:
        rng = np.random.default_rng()

    n = Q_pr.shape[0]
    risks = []

    for _ in range(n_samples):
        # Sample true state
        x_true = sample_gmrf(Q_pr, mu_pr, rng)

        # Generate observation
        y_clean = H @ x_true
        noise = rng.normal(0, np.sqrt(R_diag))
        y = y_clean + noise

        # Compute posterior
        mu_post, factor = compute_posterior(Q_pr, mu_pr, H, R_diag, y)

        # Get posterior variances (sample subset for speed)
        sample_idx = rng.choice(n, size=min(100, n), replace=False)
        var_post_sample = compute_posterior_variance_diagonal(factor, sample_idx)
        sigma_post_sample = np.sqrt(var_post_sample)

        # Compute expected loss on sample
        loss = expected_loss(
            mu_post[sample_idx],
            sigma_post_sample,
            decision_config,
            test_indices=np.arange(len(sample_idx))
        )
        risks.append(loss)

    # Prior risk (no information)
    # Assume constant prior uncertainty σ_pr everywhere
    sigma_pr = 1.0 / np.sqrt(np.diag(Q_pr.toarray()[:10, :10]).mean())  # Rough estimate
    prior_risk = conditional_risk(
        mu_pr[0], sigma_pr,
        decision_config.tau_iri,
        decision_config.L_FP_gbp,
        decision_config.L_FN_gbp,
        decision_config.L_TP_gbp
    )

    posterior_risk = np.mean(risks)
    evi = prior_risk - posterior_risk

    return evi


def evi_unscented(Q_pr, mu_pr, H, R_diag, decision_config,
                 alpha: float = 1.0, beta: float = 2.0,
                 kappa: float = 0.0) -> float:
    """
    Approximate EVI using Unscented Transform in measurement space.

    Generates sigma points from predictive distribution p(y | S),
    propagates through posterior computation and decision function.

    Args:
        Q_pr: Prior precision
        mu_pr: Prior mean
        H: Observation matrix (m × n)
        R_diag: Noise variances (m,)
        decision_config: Decision parameters
        alpha, beta, kappa: UT parameters

    Returns:
        evi: Expected reduction in decision loss (GBP)
    """
    from inference import compute_posterior, compute_posterior_variance_diagonal
    import scipy.sparse as sp

    n = Q_pr.shape[0]
    m = len(R_diag)

    # Predictive distribution: y ~ N(H μ_pr, H Σ_pr H^T + R)
    # Mean
    y_mean = H @ mu_pr

    # Covariance (approximate via sparse solves)
    # For small m, we can do this directly
    if m <= 100:
        # Solve Q_pr X = H^T to get Σ_pr H^T
        from inference import SparseFactor
        factor_pr = SparseFactor(Q_pr)

        # Solve for each column of H^T
        H_dense = H.toarray() if sp.issparse(H) else H
        Sigma_pr_HT = np.zeros((n, m))
        for i in range(m):
            Sigma_pr_HT[:, i] = factor_pr.solve(H_dense[i, :])

        y_cov = H_dense @ Sigma_pr_HT + np.diag(R_diag)
    else:
        # For large m, use approximation
        warnings.warn("Large m in UT, using diagonal approximation")
        # Approximate as diagonal
        y_cov = np.diag(R_diag) + 0.1 * np.eye(m)

    # Compute UT weights
    lambda_param = alpha**2 * (m + kappa) - m
    weights_m = np.full(2*m + 1, 1.0 / (2 * (m + lambda_param)))
    weights_m[0] = lambda_param / (m + lambda_param)
    weights_c = weights_m.copy()
    weights_c[0] += (1 - alpha**2 + beta)

    # Generate sigma points
    try:
        L = np.linalg.cholesky(y_cov)
    except np.linalg.LinAlgError:
        # Add nugget for stability
        L = np.linalg.cholesky(y_cov + 1e-6 * np.eye(m))

    scale = np.sqrt(m + lambda_param)
    sigma_points = [y_mean]  # Center point

    for i in range(m):
        sigma_points.append(y_mean + scale * L[:, i])
        sigma_points.append(y_mean - scale * L[:, i])

    # Propagate through decision function
    risks = []
    for y_sigma in sigma_points:
        # Compute posterior
        mu_post, factor = compute_posterior(Q_pr, mu_pr, H, R_diag, y_sigma)

        # Sample a few test points
        test_idx = np.linspace(0, n-1, min(50, n), dtype=int)
        var_post = compute_posterior_variance_diagonal(factor, test_idx)
        sigma_post = np.sqrt(var_post)

        # Expected loss
        loss = expected_loss(
            mu_post[test_idx],
            sigma_post,
            decision_config,
            test_indices=np.arange(len(test_idx))
        )
        risks.append(loss)

    # Weighted average
    posterior_risk = np.dot(weights_m, risks)

    # Prior risk (rough estimate)
    # Use diagonal elements of Q_pr^{-1} to estimate prior variance
    from inference import SparseFactor
    factor_pr = SparseFactor(Q_pr)
    sample_idx = np.linspace(0, n-1, min(10, n), dtype=int)
    var_pr_sample = np.array([
        factor_pr.solve(np.eye(n)[i, :])[i] for i in sample_idx
    ])
    sigma_pr = np.sqrt(var_pr_sample.mean())

    prior_risk = conditional_risk(
        mu_pr[0], sigma_pr,
        decision_config.tau_iri,
        decision_config.L_FP_gbp,
        decision_config.L_FN_gbp,
        decision_config.L_TP_gbp
    )

    evi = prior_risk - posterior_risk

    return evi

    # Compute UT weights
    lambda_param = alpha**2 * (m + kappa) - m
    weights_m = np.full(2*m + 1, 1.0 / (2 * (m + lambda_param)))
    weights_m[0] = lambda_param / (m + lambda_param)
    weights_c = weights_m.copy()
    weights_c[0] += (1 - alpha**2 + beta)

    # Generate sigma points
    try:
        L = np.linalg.cholesky(y_cov)
    except np.linalg.LinAlgError:
        # Add nugget for stability
        L = np.linalg.cholesky(y_cov + 1e-6 * np.eye(m))

    scale = np.sqrt(m + lambda_param)
    sigma_points = [y_mean]  # Center point

    for i in range(m):
        sigma_points.append(y_mean + scale * L[:, i])
        sigma_points.append(y_mean - scale * L[:, i])

    # Propagate through decision function
    risks = []
    for y_sigma in sigma_points:
        # Compute posterior
        mu_post, factor = compute_posterior(Q_pr, mu_pr, H, R_diag, y_sigma)

        # Sample a few test points
        test_idx = np.linspace(0, n-1, min(50, n), dtype=int)
        var_post = compute_posterior_variance_diagonal(factor, test_idx)
        sigma_post = np.sqrt(var_post)

        # Expected loss
        loss = expected_loss(
            mu_post[test_idx],
            sigma_post,
            decision_config,
            test_indices=np.arange(len(test_idx))
        )
        risks.append(loss)

    # Weighted average
    posterior_risk = np.dot(weights_m, risks)

    # Prior risk
    sigma_pr = 1.0 / np.sqrt(np.diag(Q_pr.toarray()[:10, :10]).mean())
    prior_risk = conditional_risk(
        mu_pr[0], sigma_pr,
        decision_config.tau_iri,
        decision_config.L_FP_gbp,
        decision_config.L_FN_gbp,
        decision_config.L_TP_gbp
    )

    evi = prior_risk - posterior_risk

    return evi


if __name__ == "__main__":
    from config import load_config
    from geometry import build_grid2d_geometry
    from spatial_field import build_prior
    from sensors import generate_sensor_pool
    from sensors import assemble_H_R

    cfg = load_config()
    rng = cfg.get_rng()

    # Setup
    geom = build_grid2d_geometry(20, 20, h=cfg.geometry.h)
    Q_pr, mu_pr = build_prior(geom, cfg.prior)

    # Generate sensors
    sensors = generate_sensor_pool(geom, cfg.sensors, rng)
    selected = rng.choice(sensors, size=10, replace=False)
    H, R = assemble_H_R(selected, geom.n)

    print("Testing EVI computation...")

    # Monte Carlo (small sample for speed)
    print("  Monte Carlo (n=50)...")
    evi_mc = evi_monte_carlo(Q_pr, mu_pr, H, R, cfg.decision, n_samples=50, rng=rng)
    print(f"    EVI = £{evi_mc:.2f}")

    # Unscented Transform
    print("  Unscented Transform...")
    evi_ut = evi_unscented(Q_pr, mu_pr, H, R, cfg.decision)
    print(f"    EVI = £{evi_ut:.2f}")

    print(f"  Difference: £{abs(evi_mc - evi_ut):.2f}")