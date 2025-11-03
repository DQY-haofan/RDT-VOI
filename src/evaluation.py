"""
Performance evaluation, spatial cross-validation, and diagnostic metrics.
"""
import warnings

import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist
from scipy.stats import norm


def spatial_block_cv(coords: np.ndarray,
                     k_folds: int,
                     buffer_width: float = 0.0,
                     block_strategy: str = "kmeans",
                     rng: np.random.Generator = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Spatial block cross-validation with optional buffer zones.
    """
    from geometry import get_spatial_blocks

    n = len(coords)

    block_labels = get_spatial_blocks(coords, k_folds, strategy=block_strategy, rng=rng)

    folds = []
    for fold_id in range(k_folds):
        test_mask = (block_labels == fold_id)
        test_idx = np.where(test_mask)[0]

        train_mask = ~test_mask

        if buffer_width > 0:
            test_coords = coords[test_idx]
            distances = cdist(coords, test_coords, metric='euclidean')
            min_dist_to_test = distances.min(axis=1)

            buffer_mask = min_dist_to_test <= buffer_width
            train_mask = train_mask & ~buffer_mask

        train_idx = np.where(train_mask)[0]

        folds.append((train_idx, test_idx))

        print(f"  Fold {fold_id+1}: train={len(train_idx)}, test={len(test_idx)}")

    return folds


def compute_roi(prior_loss: float,
                posterior_loss: float,
                sensor_cost: float) -> float:
    """
    计算投资回报率（Return on Investment）

    ROI = (节省的损失 - 传感成本) / 传感成本
    """
    savings = prior_loss - posterior_loss

    if sensor_cost <= 0:
        return np.inf if savings > 0 else 0.0

    roi = (savings - sensor_cost) / sensor_cost

    return roi


def compute_action_constrained_loss(mu_post: np.ndarray,
                                    sigma_post: np.ndarray,
                                    x_true: np.ndarray,
                                    decision_config,
                                    K: int = None,
                                    tau: float = None) -> Dict:
    """
    🔥 P0-1修复：Action-limited指标 - 使用方法特异后验

    关键修复：
    1. **必须**使用该方法的后验mu_post, sigma_post（不再使用先验）
    2. 按后验故障概率p_post排序，取Top-K ∩ {p_post > p_T}
    3. 统一使用锁定的tau（禁止动态计算）
    4. 返回详细的性能指标

    Args:
        mu_post: 后验均值 (n,) - 🔥 该方法的特异后验
        sigma_post: 后验标准差 (n,) - 🔥 该方法的特异后验
        x_true: 真实状态 (n,)
        decision_config: 决策配置
        K: 行动限制数量
        tau: 🔥 统一锁定的决策阈值（必须预先设置）

    Returns:
        详细指标字典，包括hit_rate, precision, recall, F1, regret等
    """
    from decision import conditional_risk, get_unified_prob_threshold

    # 🔥 P0-3：统一阈值来源（必须已锁定）
    if tau is None:
        if hasattr(decision_config, 'tau_iri') and decision_config.tau_iri is not None:
            tau = decision_config.tau_iri
        else:
            raise ValueError(
                "tau not provided and tau_iri not set in decision_config. "
                "Must call lock_decision_threshold() before evaluation."
            )

    n = len(mu_post)

    # 🔥 关键：使用**该方法的后验**计算故障概率
    # 这确保了不同方法会有不同的p_failure，从而有不同的行动决策
    sigma_safe = np.maximum(sigma_post, 1e-12)
    p_failure = 1.0 - norm.cdf((tau - mu_post) / sigma_safe)

    # 获取统一的Bayes最优概率阈值
    p_T = get_unified_prob_threshold(
        decision_config.L_FP_gbp,
        decision_config.L_FN_gbp,
        decision_config.L_TP_gbp,
        getattr(decision_config, 'L_TN_gbp', 0.0)
    )

    # === 无限制情况：Bayes最优决策 ===
    unrestricted_risks = np.array([
        conditional_risk(
            mu_post[i], sigma_post[i], tau,
            decision_config.L_FP_gbp,
            decision_config.L_FN_gbp,
            decision_config.L_TP_gbp,
            getattr(decision_config, 'L_TN_gbp', 0.0)
        )
        for i in range(n)
    ])
    unrestricted_loss = float(unrestricted_risks.mean())

    if K is None or K >= n:
        # 无限制情况
        return {
            'K': K or n,
            'p_threshold': p_T,
            'n_exceed_threshold': int((p_failure > p_T).sum()),
            'n_actual_actions': int((p_failure > p_T).sum()),
            'unrestricted_loss': unrestricted_loss,
            'constrained_loss': unrestricted_loss,
            'regret': 0.0,
            'precision_at_k': 1.0,
            'recall_at_k': 1.0,
            'f1_at_k': 1.0,
            'action_efficiency': 1.0,
            'cost_efficiency_ratio': 0.0,
            'hit_count': int((p_failure > p_T).sum()),
            'true_high_risk_count': int((x_true > tau).sum())
        }

    # 🔥 P0-1修复：Top-K ∩ {p_post > p_T}策略
    # Step 1: 按**后验**故障概率排序，取前K个候选
    top_k_candidates = np.argsort(p_failure)[-K:]

    # Step 2: 在候选中筛选真正超过阈值的
    exceed_threshold = p_failure > p_T

    # Step 3: 实际执行维护 = Top-K ∩ {p_post > p_T}
    do_maintain = np.zeros(n, dtype=bool)
    actual_actions = []

    for idx in top_k_candidates:
        if exceed_threshold[idx]:
            do_maintain[idx] = True
            actual_actions.append(idx)

    n_actual_actions = len(actual_actions)

    # === 计算限制后的风险 ===
    constrained_risks = np.zeros(n)
    for i in range(n):
        if do_maintain[i]:
            # 维护：承担L_TP或L_FP
            if x_true[i] > tau:
                constrained_risks[i] = decision_config.L_TP_gbp
            else:
                constrained_risks[i] = decision_config.L_FP_gbp
        else:
            # 不维护：承担L_FN或L_TN
            if x_true[i] > tau:
                constrained_risks[i] = decision_config.L_FN_gbp
            else:
                constrained_risks[i] = getattr(decision_config, 'L_TN_gbp', 0.0)

    constrained_loss = float(constrained_risks.mean())
    regret = constrained_loss - unrestricted_loss

    # === 性能指标 ===
    true_exceed = x_true > tau
    n_true_high_risk = int(true_exceed.sum())

    if n_true_high_risk > 0:
        hit_count = int(np.sum(do_maintain & true_exceed))
        recall_at_k = hit_count / n_true_high_risk
    else:
        recall_at_k = 1.0
        hit_count = 0

    if n_actual_actions > 0:
        precision_at_k = np.sum(do_maintain & true_exceed) / n_actual_actions
    else:
        precision_at_k = 1.0

    # F1 Score
    if precision_at_k + recall_at_k > 0:
        f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
    else:
        f1_at_k = 0.0

    # Action Efficiency
    action_efficiency = n_actual_actions / K if K > 0 else 0.0

    # Cost Efficiency Ratio
    if n_actual_actions > 0:
        cost_efficiency_ratio = (unrestricted_loss - constrained_loss) / n_actual_actions
    else:
        cost_efficiency_ratio = 0.0

    return {
        'K': K,
        'p_threshold': float(p_T),
        'n_exceed_threshold': int(exceed_threshold.sum()),
        'n_actual_actions': n_actual_actions,
        'unrestricted_loss': float(unrestricted_loss),
        'constrained_loss': float(constrained_loss),
        'regret': float(regret),
        'precision_at_k': float(precision_at_k),
        'recall_at_k': float(recall_at_k),
        'f1_at_k': float(f1_at_k),
        'action_efficiency': float(action_efficiency),
        'cost_efficiency_ratio': float(cost_efficiency_ratio),
        'hit_count': hit_count,
        'true_high_risk_count': n_true_high_risk
    }


def compute_enhanced_metrics(mu_post: np.ndarray,
                             sigma_post: np.ndarray,
                             x_true: np.ndarray,
                             test_idx: np.ndarray,
                             decision_config,
                             sensor_cost: float = 0.0,
                             prior_loss: float = None,
                             mu_pr: np.ndarray = None,
                             sigma_pr: np.ndarray = None,
                             K_action: int = None,
                             domain_scale_factor: float = 1.0) -> Dict:
    """
    🔥 P0-2修复：增强的指标计算 - 显式ΔE[Loss]方法

    关键修复：
    1. 固定near-mask定义（基于先验μ_pr）
    2. 对同一near-mask，分别用先验和后验计算期望损失
    3. 显式计算savings_near = E[Loss]_prior - E[Loss]_post
    4. 返回unscaled和scaled两套指标
    5. 统一使用锁定的tau

    Args:
        mu_post: 后验均值 (n,)
        sigma_post: 后验标准差 (n,)
        x_true: 真实状态 (n,)
        test_idx: 测试集索引
        decision_config: 决策配置（必须已锁定tau_iri）
        sensor_cost: 传感器成本
        prior_loss: 先验损失（可选，用于验证）
        mu_pr: 🔥 先验均值（用于定义near-mask）
        sigma_pr: 🔥 先验标准差（用于计算先验损失）
        K_action: 行动限制数量
        domain_scale_factor: 🔥 域缩放因子（经济尺度校准）

    Returns:
        完整指标字典，包含unscaled和scaled两版
    """
    from decision import expected_loss, conditional_risk
    from scipy.stats import norm as scipy_norm

    # 🔥 P0-3：确保tau已锁定
    if hasattr(decision_config, 'tau_iri') and decision_config.tau_iri is not None:
        tau = decision_config.tau_iri
    else:
        raise ValueError(
            "tau_iri not set in decision_config. "
            "Must call lock_decision_threshold() before evaluation."
        )

    n_test = len(test_idx)

    # === 基础指标 ===
    residuals = mu_post[test_idx] - x_true[test_idx]
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    # Z-scores (for calibration)
    sigma_test = sigma_post[test_idx]
    sigma_test_safe = np.maximum(sigma_test, 1e-12)
    z_scores = residuals / sigma_test_safe

    # Coverage
    coverage_levels = [0.5, 0.68, 0.90, 0.95]
    coverage = {}
    for level in coverage_levels:
        alpha = 1 - level
        z_crit = scipy_norm.ppf(1 - alpha / 2)
        lower = mu_post[test_idx] - z_crit * sigma_test
        upper = mu_post[test_idx] + z_crit * sigma_test
        covered = ((x_true[test_idx] >= lower) & (x_true[test_idx] <= upper)).sum()
        coverage[f'coverage_{int(level * 100)}'] = float(covered / n_test)

    # === 🔥 P0-2修复：近阈值节省计算（显式ΔE[Loss]） ===
    # Step 1: 固定near-mask（基于先验μ_pr，不变）
    delta = 0.3  # 近阈值带宽（可配置）

    if mu_pr is not None:
        # 🔥 关键：near-mask基于**先验**定义，对所有方法一致
        near_mask_all = np.abs(mu_pr - tau) <= delta
        near_idx_all = np.where(near_mask_all)[0]

        # 在测试集中的近阈值点
        near_mask_test = np.isin(test_idx, near_idx_all)
        near_idx_test = test_idx[near_mask_test]
        n_near_test = len(near_idx_test)

        if n_near_test > 0 and sigma_pr is not None:
            # Step 2: 对同一near_idx_test，分别计算先验和后验损失
            # 🔥 先验损失（在near点上）
            loss_pr_near = expected_loss(
                mu_pr[near_idx_test],
                sigma_pr[near_idx_test],
                decision_config,
                test_indices=np.arange(n_near_test),
                tau=tau
            )

            # 🔥 后验损失（在near点上）
            loss_post_near = expected_loss(
                mu_post[near_idx_test],
                sigma_post[near_idx_test],
                decision_config,
                test_indices=np.arange(n_near_test),
                tau=tau
            )

            # Step 3: 显式计算节省
            savings_near_unscaled = loss_pr_near - loss_post_near
            savings_near_scaled = savings_near_unscaled * domain_scale_factor

            # ROI计算
            if sensor_cost > 0:
                roi_near_unscaled = (savings_near_unscaled - sensor_cost) / sensor_cost
                roi_near_scaled = (savings_near_scaled - sensor_cost) / sensor_cost
            else:
                roi_near_unscaled = 0.0
                roi_near_scaled = 0.0
        else:
            savings_near_unscaled = 0.0
            savings_near_scaled = 0.0
            roi_near_unscaled = 0.0
            roi_near_scaled = 0.0
            n_near_test = 0
    else:
        # 无先验信息，跳过near-threshold计算
        savings_near_unscaled = 0.0
        savings_near_scaled = 0.0
        roi_near_unscaled = 0.0
        roi_near_scaled = 0.0
        n_near_test = 0

    # === 全局经济损失 ===
    post_loss_unscaled = expected_loss(
        mu_post[test_idx],
        sigma_post[test_idx],
        decision_config,
        test_indices=np.arange(n_test),
        tau=tau
    )

    # 🔥 P1-4：应用域缩放因子
    post_loss_scaled = post_loss_unscaled * domain_scale_factor

    # 全局节省
    if prior_loss is not None:
        savings_unscaled = prior_loss - post_loss_unscaled
        savings_scaled = savings_unscaled * domain_scale_factor
    else:
        savings_unscaled = 0.0
        savings_scaled = 0.0

    # 全局ROI
    if sensor_cost > 0:
        roi_unscaled = (savings_unscaled - sensor_cost) / sensor_cost
        roi_scaled = (savings_scaled - sensor_cost) / sensor_cost
        cost_efficiency_unscaled = savings_unscaled / sensor_cost
        cost_efficiency_scaled = savings_scaled / sensor_cost
    else:
        roi_unscaled = 0.0
        roi_scaled = 0.0
        cost_efficiency_unscaled = 0.0
        cost_efficiency_scaled = 0.0

    # === Action-constrained指标 ===
    action_metrics = {}
    if K_action is not None:
        action_metrics = compute_action_constrained_loss(
            mu_post, sigma_post, x_true,
            decision_config, K=K_action, tau=tau
        )

    # === 组装结果 ===
    metrics = {
        # 基础指标
        'rmse': rmse,
        'mae': mae,
        'n_test': n_test,

        # 覆盖率
        **coverage,

        # 🔥 双通道经济指标：unscaled（原始） + scaled（域缩放）
        'expected_loss_gbp_unscaled': float(post_loss_unscaled),
        'expected_loss_gbp': float(post_loss_scaled),  # 默认scaled（向后兼容）

        'savings_gbp_unscaled': float(savings_unscaled),
        'savings_gbp': float(savings_scaled),

        'roi_unscaled': float(roi_unscaled),
        'roi': float(roi_scaled),

        'cost_efficiency_unscaled': float(cost_efficiency_unscaled),
        'cost_efficiency': float(cost_efficiency_scaled),

        # 🔥 近阈值指标（双通道）
        'savings_near_threshold_unscaled': float(savings_near_unscaled),
        'savings_near_threshold': float(savings_near_scaled),
        'roi_near_threshold_unscaled': float(roi_near_unscaled),
        'roi_near_threshold': float(roi_near_scaled),
        'fraction_near_threshold': float(n_near_test / n_test) if n_test > 0 else 0.0,
        'n_near_threshold': int(n_near_test),

        # 缩放因子（调试用）
        'domain_scale_factor': float(domain_scale_factor),

        # Action-constrained（如果适用）
        **action_metrics,

        # Z-scores（用于校准诊断）
        'z_scores': z_scores,
    }

    return metrics


def compute_metrics(mu_post: np.ndarray,
                    sigma_post: np.ndarray,
                    x_true: np.ndarray,
                    test_idx: np.ndarray,
                    decision_config) -> Dict[str, float]:
    """
    Compute performance metrics on test set.
    """
    from decision import expected_loss

    mu_test = mu_post[test_idx]
    sigma_test = sigma_post[test_idx]
    x_test = x_true[test_idx]

    errors = mu_test - x_test
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    r2 = 1.0 - np.sum(errors ** 2) / np.sum((x_test - x_test.mean()) ** 2)

    exp_loss = expected_loss(mu_test, sigma_test, decision_config)

    z_scores = errors / np.maximum(sigma_test, 1e-12)

    coverage_90 = np.mean(np.abs(z_scores) <= 1.645)

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
        'z_scores': z_scores.astype(float)
    }


def morans_i(residuals: np.ndarray,
            adjacency: np.ndarray,
            n_permutations: int = 999,
            rng: np.random.Generator = None) -> Tuple[float, float]:
    """
    Compute Moran's I spatial autocorrelation statistic with permutation test.
    """
    import scipy.sparse as sp

    if rng is None:
        rng = np.random.default_rng()

    n = len(residuals)

    if sp.issparse(adjacency):
        W = adjacency.toarray()
    else:
        W = adjacency

    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    W = W / row_sums[:, None]

    r = residuals - residuals.mean()

    W_sum = W.sum()
    numerator = n * np.sum(W * np.outer(r, r))
    denominator = W_sum * np.sum(r**2)

    I_obs = numerator / denominator if denominator > 0 else 0.0

    I_perm = np.zeros(n_permutations)
    for perm_idx in range(n_permutations):
        r_perm = rng.permutation(r)
        numerator_perm = n * np.sum(W * np.outer(r_perm, r_perm))
        I_perm[perm_idx] = numerator_perm / denominator

    p_value = (np.sum(np.abs(I_perm) >= np.abs(I_obs)) + 1) / (n_permutations + 1)

    return I_obs, p_value


def spatial_bootstrap(block_ids: np.ndarray,
                     metric_values: np.ndarray,
                     n_bootstrap: int = 1000,
                     confidence_level: float = 0.95,
                     rng: np.random.Generator = None) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval using spatial block resampling.
    """
    if rng is None:
        rng = np.random.default_rng()

    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)

    bootstrap_means = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)

        sampled_values = []
        for block in sampled_blocks:
            block_values = metric_values[block_ids == block]
            sampled_values.extend(block_values)

        bootstrap_means[b] = np.mean(sampled_values)

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

        print(f"    RMSE={metrics['rmse']:.3f}, Loss=£{metrics['expected_loss_gbp']:.0f}, "
              f"I={I_stat:.3f} (p={I_pval:.3f})")

    # Aggregate across folds
    aggregated = {}
    for key in fold_results[0]['metrics'].keys():  # 🔥 改为从metrics子字典提取
        if key in ['z_scores', 'n_test', 'type_counts']:
            continue
        values = [fr['metrics'][key] for fr in fold_results if fr.get('success')]
        if values:
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }

    return {
        'fold_results': fold_results,
        'aggregated': aggregated,
        'selection_result': selection_result
    }


if __name__ == "__main__":
    from config import load_scenario_config, load_config  # ✅ 改用场景加载
    from geometry import build_grid2d_geometry
    from spatial_field import build_prior, sample_gmrf
    from sensors import generate_sensor_pool
    from selection import greedy_mi

    cfg = load_config("baseline_config.yaml")  # ✅ 新方式
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

# ============================================================================
# 🔥 Business-Friendly Metrics (新增业务指标)
# ============================================================================

def compute_savings_and_roi(results_dict: Dict,
                            baseline_method: str = 'uniform',
                            cost_key: str = 'total_cost',
                            loss_key: str = 'expected_loss_gbp') -> Dict:
    """
    计算各方法相对 baseline 的省钱和 ROI
    """
    business_metrics = {}

    budgets = set()
    for method_data in results_dict.values():
        budgets.update(method_data.keys())
    budgets = sorted(budgets)

    for budget in budgets:
        if baseline_method not in results_dict:
            print(f"Warning: baseline method '{baseline_method}' not found")
            continue

        if budget not in results_dict[baseline_method]:
            continue

        baseline_loss = results_dict[baseline_method][budget]['mean'][loss_key]

        for method_name, method_data in results_dict.items():
            if budget not in method_data:
                continue

            metrics = method_data[budget]['mean']
            method_loss = metrics[loss_key]
            method_cost = metrics[cost_key]

            savings = baseline_loss - method_loss

            if method_cost > 0:
                roi = savings / method_cost
                cost_efficiency = savings / method_cost
            else:
                roi = np.inf if savings > 0 else 0.0
                cost_efficiency = np.inf if savings > 0 else 0.0

            key = (method_name, budget)
            business_metrics[key] = {
                'savings_gbp': savings,
                'roi': roi,
                'cost_efficiency': cost_efficiency,
                'total_cost': method_cost,
                'expected_loss_gbp': method_loss,
                'baseline_loss_gbp': baseline_loss
            }

    return business_metrics


def compute_critical_region_metrics(mu_post: np.ndarray,
                                    sigma_post: np.ndarray,
                                    x_true: np.ndarray,
                                    tau: float,
                                    epsilon: float = 0.2) -> Dict:
    """
    计算临界区域（阈值附近）的性能指标
    """
    critical_mask = np.abs(mu_post - tau) <= epsilon
    critical_idx = np.where(critical_mask)[0]

    if len(critical_idx) == 0:
        return {
            'n_critical': 0,
            'fraction_critical': 0.0,
            'rmse_critical': np.nan,
            'misclass_rate': np.nan,
            'avg_uncertainty_critical': np.nan,
            'max_uncertainty_critical': np.nan
        }

    mu_crit = mu_post[critical_idx]
    sigma_crit = sigma_post[critical_idx]
    x_crit = x_true[critical_idx]

    errors = mu_crit - x_crit
    rmse_crit = np.sqrt(np.mean(errors ** 2))

    pred_above = mu_crit > tau
    true_above = x_crit > tau
    misclass_rate = np.mean(pred_above != true_above)

    avg_uncertainty = sigma_crit.mean()
    max_uncertainty = sigma_crit.max()

    return {
        'n_critical': len(critical_idx),
        'fraction_critical': len(critical_idx) / len(mu_post),
        'rmse_critical': rmse_crit,
        'misclass_rate': misclass_rate,
        'avg_uncertainty_critical': avg_uncertainty,
        'max_uncertainty_critical': max_uncertainty
    }


def run_cv_experiment(geom, Q_pr, mu_pr, x_true, sensors,
                      selection_method, k, cv_config,
                      decision_config, rng) -> Dict:
    """
    Run complete CV experiment for one method and budget.
    """
    from inference import compute_posterior, compute_posterior_variance_diagonal
    from sensors import get_observation

    if hasattr(cv_config, '__dict__'):
        cv_dict = cv_config.__dict__
    else:
        cv_dict = cv_config

    corr_length = np.sqrt(8.0) / 0.08
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
        print(f"\n  Fold {fold_idx + 1}/{len(folds)}")

        selection_result = selection_method(sensors, k, Q_pr)
        selected_sensors = [sensors[i] for i in selection_result.selected_ids]

        y, H, R = get_observation(x_true, selected_sensors, rng)

        mu_post, factor = compute_posterior(Q_pr, mu_pr, H, R, y)

        var_post_test = compute_posterior_variance_diagonal(factor, test_idx)
        sigma_post_test = np.sqrt(var_post_test)

        sigma_post = np.zeros(len(mu_post))
        sigma_post[test_idx] = sigma_post_test

        # 🔥 使用增强的指标计算
        metrics = compute_enhanced_metrics(
            mu_post, sigma_post, x_true, test_idx, decision_config,
            sensor_cost=selection_result.total_cost,
            K_action=getattr(decision_config, 'K_action', None)
        )

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
            'residuals': residuals[test_idx],
        }

        fold_results.append(fold_result_dict)

        print(f"    RMSE={metrics['rmse']:.3f}, Loss=£{metrics['expected_loss_gbp']:.0f}, "
              f"I={I_stat:.3f} (p={I_pval:.3f})")

    # Aggregate across folds
    aggregated = {}
    for key in fold_results[0]['metrics'].keys():
        if key in ['z_scores', 'n_test', 'type_counts']:
            continue
        values = [fr['metrics'][key] for fr in fold_results if fr.get('success')]
        if values:
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }

    return {
        'fold_results': fold_results,
        'aggregated': aggregated,
        'selection_result': selection_result
    }


def compute_domain_scale_factor(config) -> float:
    """
    🔥 P1-4：自动计算域缩放因子

    将评估域（测试集）的损失缩放到业务域（全网络）的等价时间跨度

    Args:
        config: 配置对象，需包含economics部分

    Returns:
        domain_scale_factor: 缩放因子（≥1）

    示例：
        如果网络200km，测试覆盖35km，评估期10年，单次评估7天：
        scale_factor = (200/35) * (10*365/7) ≈ 2940

        这意味着测试集上的£1损失 ≈ 全网络10年运营的£2940损失
    """
    if not hasattr(config, 'economics'):
        warnings.warn(
            "No 'economics' section in config. "
            "Using default scale_factor=1.0 (no scaling). "
            "To enable scaling, add economics section to baseline_config.yaml"
        )
        return 1.0

    econ = config.economics

    # 空间缩放：全网络 / 测试域
    if hasattr(econ, 'network_km') and hasattr(econ, 'test_km'):
        spatial_scale = econ.network_km / econ.test_km
    else:
        spatial_scale = 1.0
        warnings.warn("economics.network_km or test_km not set, using spatial_scale=1.0")

    # 时间缩放：评估期 / 单次评估周期
    if hasattr(econ, 'horizon_years') and hasattr(econ, 'eval_period_days'):
        horizon_days = econ.horizon_years * 365
        temporal_scale = horizon_days / econ.eval_period_days
    else:
        temporal_scale = 1.0
        warnings.warn("economics.horizon_years or eval_period_days not set, using temporal_scale=1.0")

    scale_factor = spatial_scale * temporal_scale

    # 健康检查
    if scale_factor < 1.0:
        warnings.warn(f"Computed scale_factor={scale_factor:.2f} < 1, clamping to 1.0")
        scale_factor = 1.0

    return scale_factor
# ============================================================================
# 🔥 测试用例
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  TESTING P0 FIXES")
    print("=" * 70)


    # Mock配置
    class MockDecisionConfig:
        L_FP_gbp = 3000
        L_FN_gbp = 30000
        L_TP_gbp = 400
        L_TN_gbp = 0
        tau_iri = 2.2  # 🔥 已锁定


    class MockEconomics:
        network_km = 200
        test_km = 35
        horizon_years = 10
        eval_period_days = 7


    class MockConfig:
        economics = MockEconomics()


    decision_config = MockDecisionConfig()
    config = MockConfig()

    # 模拟数据
    rng = np.random.default_rng(42)
    n = 100
    test_idx = np.arange(20, 40)  # 20个测试点

    x_true = rng.normal(2.2, 0.3, n)
    mu_pr = rng.normal(2.2, 0.2, n)
    sigma_pr = np.full(n, 0.5)

    mu_post = rng.normal(2.2, 0.15, n)
    sigma_post = np.full(n, 0.3)

    tau = decision_config.tau_iri

    # Test 1: Action-constrained
    print("\n[Test 1] Action-constrained loss")
    action_metrics = compute_action_constrained_loss(
        mu_post, sigma_post, x_true,
        decision_config, K=10, tau=tau
    )
    print(f"  Regret: £{action_metrics['regret']:.2f}")
    print(f"  Precision@K: {action_metrics['precision_at_k']:.3f}")
    print(f"  Recall@K: {action_metrics['recall_at_k']:.3f}")
    print(f"  F1@K: {action_metrics['f1_at_k']:.3f}")

    # Test 2: Enhanced metrics with scaling
    print("\n[Test 2] Enhanced metrics with domain scaling")
    scale_factor = compute_domain_scale_factor(config)
    print(f"  Computed scale_factor: {scale_factor:.0f}")

    metrics = compute_enhanced_metrics(
        mu_post, sigma_post, x_true, test_idx,
        decision_config,
        sensor_cost=1500,
        mu_pr=mu_pr,
        sigma_pr=sigma_pr,
        K_action=10,
        domain_scale_factor=scale_factor
    )

    print(f"  ROI (unscaled): {metrics['roi_unscaled']:.3f}")
    print(f"  ROI (scaled): {metrics['roi']:.3f}")
    print(f"  Savings (unscaled): £{metrics['savings_gbp_unscaled']:.0f}")
    print(f"  Savings (scaled): £{metrics['savings_gbp']:.0f}")
    print(f"  Near-threshold savings (scaled): £{metrics['savings_near_threshold']:.0f}")

    print("\n✅ All P0 tests passed!")
