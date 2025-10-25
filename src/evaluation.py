"""
Performance evaluation, spatial cross-validation, and diagnostic metrics.
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist


def spatial_block_cv(coords: np.ndarray,
                     k_folds: int,
                     buffer_width: float = 0.0,
                     block_strategy: str = "kmeans",
                     rng: np.random.Generator = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Spatial block cross-validation with optional buffer zones.

    Args:
        coords: (n, d) spatial coordinates
        k_folds: Number of folds
        buffer_width: Buffer distance (points within buffer excluded from both sets)
        block_strategy: "kmeans" | "grid"
        rng: Random generator

    Returns:
        folds: List of (train_idx, test_idx) tuples
    """
    from geometry import get_spatial_blocks

    n = len(coords)

    # Partition into spatial blocks
    block_labels = get_spatial_blocks(coords, k_folds, strategy=block_strategy, rng=rng)

    folds = []
    for fold_id in range(k_folds):
        # Test set: current block
        test_mask = (block_labels == fold_id)
        test_idx = np.where(test_mask)[0]

        # Train set: all other blocks
        train_mask = ~test_mask

        # Apply buffer if requested
        if buffer_width > 0:
            # Find points within buffer distance of test set
            test_coords = coords[test_idx]
            distances = cdist(coords, test_coords, metric='euclidean')
            min_dist_to_test = distances.min(axis=1)

            # Exclude buffer points from train
            buffer_mask = min_dist_to_test <= buffer_width
            train_mask = train_mask & ~buffer_mask

        train_idx = np.where(train_mask)[0]

        folds.append((train_idx, test_idx))

        print(f"  Fold {fold_id+1}: train={len(train_idx)}, test={len(test_idx)}")

    return folds


def compute_metrics(mu_post: np.ndarray,
                    sigma_post: np.ndarray,
                    x_true: np.ndarray,
                    test_idx: np.ndarray,
                    decision_config) -> Dict[str, float]:
    """
    Compute performance metrics on test set.

    Args:
        mu_post: Posterior means (n,)
        sigma_post: Posterior std deviations (n,)
        x_true: True state (n,)
        test_idx: Test set indices
        decision_config: Decision configuration

    Returns:
        metrics: Dictionary of metric values
    """
    from decision import expected_loss

    # Extract test values
    mu_test = mu_post[test_idx]
    sigma_test = sigma_post[test_idx]
    x_test = x_true[test_idx]

    # Reconstruction metrics
    errors = mu_test - x_test
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    r2 = 1.0 - np.sum(errors ** 2) / np.sum((x_test - x_test.mean()) ** 2)

    # Decision-aware loss
    exp_loss = expected_loss(mu_test, sigma_test, decision_config)

    # 🔥 Z-scores for calibration (prevent division by zero)
    z_scores = errors / np.maximum(sigma_test, 1e-12)

    # Calibration: coverage
    coverage_90 = np.mean(np.abs(z_scores) <= 1.645)  # 90% CI

    # MSSE (Mean Squared Standardized Error)
    standardized_errors = errors / np.maximum(sigma_test, 1e-12)
    msse = np.mean(standardized_errors ** 2)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'expected_loss_gbp': exp_loss,
        'coverage_90': coverage_90,
        'msse': msse,
        'n_test': len(test_idx),
        'z_scores': z_scores.astype(float)  # 🔥 新增：用于校准图
    }


def morans_i(residuals: np.ndarray,
            adjacency: np.ndarray,
            n_permutations: int = 999,
            rng: np.random.Generator = None) -> Tuple[float, float]:
    """
    Compute Moran's I spatial autocorrelation statistic with permutation test.

    Args:
        residuals: Residual values (n,)
        adjacency: Binary adjacency matrix (n, n) or sparse
        n_permutations: Number of permutations for p-value
        rng: Random generator

    Returns:
        I: Moran's I statistic
        p_value: Permutation p-value
    """
    import scipy.sparse as sp

    if rng is None:
        rng = np.random.default_rng()

    n = len(residuals)

    # Convert to dense if sparse
    if sp.issparse(adjacency):
        W = adjacency.toarray()
    else:
        W = adjacency

    # Row-standardize weights
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums[:, None]

    # Center residuals
    r = residuals - residuals.mean()

    # Compute Moran's I
    W_sum = W.sum()
    numerator = n * np.sum(W * np.outer(r, r))
    denominator = W_sum * np.sum(r**2)

    I_obs = numerator / denominator if denominator > 0 else 0.0

    # Permutation test
    I_perm = np.zeros(n_permutations)
    for perm_idx in range(n_permutations):
        # Randomly permute residuals
        r_perm = rng.permutation(r)

        # Compute I for permutation
        numerator_perm = n * np.sum(W * np.outer(r_perm, r_perm))
        I_perm[perm_idx] = numerator_perm / denominator

    # p-value: proportion of permuted I >= observed I
    p_value = (np.sum(np.abs(I_perm) >= np.abs(I_obs)) + 1) / (n_permutations + 1)

    return I_obs, p_value


def spatial_bootstrap(block_ids: np.ndarray,
                     metric_values: np.ndarray,
                     n_bootstrap: int = 1000,
                     confidence_level: float = 0.95,
                     rng: np.random.Generator = None) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval using spatial block resampling.

    Args:
        block_ids: Block assignment for each observation (n,)
        metric_values: Metric value for each observation (n,)
        n_bootstrap: Number of bootstrap samples
        confidence_level: CI level
        rng: Random generator

    Returns:
        ci: Dictionary with 'mean', 'lower', 'upper'
    """
    if rng is None:
        rng = np.random.default_rng()

    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)

    bootstrap_means = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Resample blocks with replacement
        sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)

        # Collect observations from sampled blocks
        sampled_values = []
        for block in sampled_blocks:
            block_values = metric_values[block_ids == block]
            sampled_values.extend(block_values)

        # Compute statistic
        bootstrap_means[b] = np.mean(sampled_values)

    # Compute percentiles
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return {
        'mean': np.mean(metric_values),
        'lower': lower,
        'upper': upper,
        'std': np.std(bootstrap_means)
    }


def run_cv_experiment(geom, Q_pr, mu_pr, x_true, sensors,
                     selection_method, k, cv_config,
                     decision_config, rng) -> Dict:
    """
    Run complete CV experiment for one method and budget.

    Args:
        geom: Geometry
        Q_pr, mu_pr: Prior parameters
        x_true: True state
        sensors: Candidate sensor pool
        selection_method: Selection function
        k: Budget
        cv_config: CV configuration dict or object
        decision_config: Decision configuration
        rng: Random generator

    Returns:
        results: Dictionary with fold results and aggregates
    """
    from inference import compute_posterior, compute_posterior_variance_diagonal
    from sensors import get_observation

    # Handle cv_config as dict or object
    if hasattr(cv_config, '__dict__'):
        cv_dict = cv_config.__dict__
    else:
        cv_dict = cv_config

    # Generate CV folds
    # Rough correlation length estimate
    corr_length = np.sqrt(8.0) / 0.08  # Default if not available
    if hasattr(cv_config, 'buffer_width_multiplier'):
        buffer_width = cv_dict.get('buffer_width_multiplier', 1.5) * corr_length
    else:
        buffer_width = cv_dict.get('buffer_width_multiplier', 1.5) * corr_length

    folds = spatial_block_cv(
        geom.coords,
        cv_dict.get('k_folds', 5),
        buffer_width,
        cv_dict.get('block_strategy', 'kmeans'),
        rng
    )

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n  Fold {fold_idx+1}/{len(folds)}")

        # Select sensors (on full domain or train only - depends on application)
        # Here we select on full domain for simplicity
        selection_result = selection_method(sensors, k, Q_pr)
        selected_sensors = [sensors[i] for i in selection_result.selected_ids]

        # Generate observations
        y, H, R = get_observation(x_true, selected_sensors, rng)

        # Compute posterior
        mu_post, factor = compute_posterior(Q_pr, mu_pr, H, R, y)

        # Get posterior variances on test set
        var_post_test = compute_posterior_variance_diagonal(factor, test_idx)
        sigma_post_test = np.sqrt(var_post_test)

        # Expand to full arrays (for metrics function)
        sigma_post = np.zeros(len(mu_post))
        sigma_post[test_idx] = sigma_post_test

        # Compute metrics
        metrics = compute_metrics(
            mu_post, sigma_post, x_true, test_idx, decision_config
        )

        # Spatial diagnostic: Moran's I on residuals
        residuals = mu_post - x_true
        I_stat, I_pval = morans_i(
            residuals[test_idx],
            geom.adjacency[test_idx][:, test_idx],
            n_permutations=cv_dict.get('morans_permutations', 999),
            rng=rng
        )
        metrics['morans_i'] = I_stat
        metrics['morans_pval'] = I_pval

        fold_result_dict = {
            'success': True,
            'metrics': metrics,
            'selection_result': selection_result,
            'residuals': residuals[test_idx],  # 🔥 新增：保存残差
        }

        fold_results.append(fold_result_dict)

        fold_results.append(metrics)

        print(f"    RMSE={metrics['rmse']:.3f}, Loss=£{metrics['expected_loss_gbp']:.0f}, "
              f"I={I_stat:.3f} (p={I_pval:.3f})")

    # Aggregate across folds
    aggregated = {}
    for key in fold_results[0].keys():
        values = np.array([fr[key] for fr in fold_results])
        aggregated[key] = {
            'mean': values.mean(),
            'std': values.std(),
            'values': values
        }

    return {
        'fold_results': fold_results,
        'aggregated': aggregated,
        'selection_result': selection_result
    }


if __name__ == "__main__":
    from config import load_config
    from geometry import build_grid2d_geometry
    from spatial_field import build_prior, sample_gmrf
    from sensors import generate_sensor_pool
    from selection import greedy_mi

    cfg = load_config()
    rng = cfg.get_rng()

    # Setup
    geom = build_grid2d_geometry(20, 20, h=cfg.geometry.h)
    Q_pr, mu_pr = build_prior(geom, cfg.prior)
    x_true = sample_gmrf(Q_pr, mu_pr, rng)
    sensors = generate_sensor_pool(geom, cfg.sensors, rng)

    print("Testing spatial CV...")

    # Generate folds
    folds = spatial_block_cv(
        geom.coords,
        k_folds=3,
        buffer_width=15.0,
        rng=rng
    )

    # Test Moran's I
    residuals = rng.normal(0, 0.5, geom.n)
    I, p = morans_i(residuals, geom.adjacency, n_permutations=99, rng=rng)
    print(f"\nMoran's I: {I:.3f} (p={p:.3f})")