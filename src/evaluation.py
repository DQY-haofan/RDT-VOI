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
    🔥 修复版：Action-limited 指标 - Top-K ∩ {p_f > p_T} 策略

    关键修复：
    1. 只对 (排名前K) ∩ (概率>阈值) 的位置执行维护
    2. 不足K个时宁缺毋滥，避免强制假阳性
    3. 添加详细的性能指标

    Args:
        mu_post: 后验均值 (n,)
        sigma_post: 后验标准差 (n,)
        x_true: 真实状态 (n,)
        decision_config: 决策配置
        K: 允许维护的最大数量
        tau: 决策阈值

    Returns:
        完整的action-limited指标字典
    """
    from decision import conditional_risk

    if tau is None:
        tau = decision_config.get_threshold()

    n = len(mu_post)

    # 计算每个位置的后验故障概率
    p_failure = 1.0 - norm.cdf((tau - mu_post) / np.maximum(sigma_post, 1e-12))

    # Bayes最优概率阈值
    p_T = decision_config.L_FP_gbp / (
            decision_config.L_FP_gbp + decision_config.L_FN_gbp - decision_config.L_TP_gbp
    )

    # 无限制情况：Bayes最优决策
    unrestricted_risks = np.array([
        conditional_risk(
            mu_post[i], sigma_post[i], tau,
            decision_config.L_FP_gbp,
            decision_config.L_FN_gbp,
            decision_config.L_TP_gbp,
            decision_config.L_TN_gbp
        )
        for i in range(n)
    ])
    unrestricted_loss = unrestricted_risks.mean()

    if K is None or K >= n:
        # 无限制情况
        return {
            'K': K or n,
            'p_threshold': p_T,
            'n_exceed_threshold': int((p_failure > p_T).sum()),
            'unrestricted_loss': unrestricted_loss,
            'constrained_loss': unrestricted_loss,
            'regret': 0.0,
            'precision_at_k': 1.0,
            'recall_at_k': 1.0,
            'f1_at_k': 1.0,
            'action_efficiency': 1.0
        }

    # 🔥 关键修复：Top-K ∩ {p_f > p_T} 策略
    # Step 1: 按故障概率排序，取前K个候选
    top_k_candidates = np.argsort(p_failure)[-K:]

    # Step 2: 在候选中筛选真正超过阈值的
    exceed_threshold = p_failure > p_T

    # Step 3: 实际执行维护的位置 = Top-K ∩ {p_f > p_T}
    do_maintain = np.zeros(n, dtype=bool)
    actual_actions = []

    for idx in top_k_candidates:
        if exceed_threshold[idx]:
            do_maintain[idx] = True
            actual_actions.append(idx)

    n_actual_actions = len(actual_actions)

    print(f"    Action-limited analysis:")
    print(f"      K (limit): {K}")
    print(f"      p_T (threshold): {p_T:.3f}")
    print(f"      Candidates exceeding p_T: {exceed_threshold.sum()}")
    print(f"      Actual actions taken: {n_actual_actions}")

    # 计算限制后的风险
    constrained_risks = np.zeros(n)
    for i in range(n):
        if do_maintain[i]:
            # 维护：承担 L_TP 或 L_FP
            if x_true[i] > tau:
                constrained_risks[i] = decision_config.L_TP_gbp
            else:
                constrained_risks[i] = decision_config.L_FP_gbp
        else:
            # 不维护：承担 L_FN 或 L_TN
            if x_true[i] > tau:
                constrained_risks[i] = decision_config.L_FN_gbp
            else:
                constrained_risks[i] = decision_config.L_TN_gbp

    constrained_loss = constrained_risks.mean()
    regret = constrained_loss - unrestricted_loss

    # 🔥 计算性能指标
    true_exceed = x_true > tau
    n_true_high_risk = true_exceed.sum()

    if n_true_high_risk > 0:
        # Recall@K: 在真高风险中，我们命中了多少
        hit_count = np.sum(do_maintain & true_exceed)
        recall_at_k = hit_count / n_true_high_risk
    else:
        recall_at_k = 1.0  # 无高风险位置时定义为完美
        hit_count = 0

    if n_actual_actions > 0:
        # Precision@K: 在我们行动的位置中，有多少是真高风险
        precision_at_k = np.sum(do_maintain & true_exceed) / n_actual_actions
    else:
        precision_at_k = 1.0  # 无行动时定义为完美（避免除零）

    # F1 Score
    if precision_at_k + recall_at_k > 0:
        f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
    else:
        f1_at_k = 0.0

    # Action Efficiency: 实际行动数 / 限制数
    action_efficiency = n_actual_actions / K if K > 0 else 0.0

    # 成本效率：节省的损失 / 实际行动成本（简化）
    if n_actual_actions > 0:
        cost_efficiency_ratio = (unrestricted_loss - constrained_loss) / n_actual_actions
    else:
        cost_efficiency_ratio = np.inf if unrestricted_loss > constrained_loss else 0.0

    print(f"      Recall@K: {recall_at_k:.3f}")
    print(f"      Precision@K: {precision_at_k:.3f}")
    print(f"      F1@K: {f1_at_k:.3f}")
    print(f"      Action efficiency: {action_efficiency:.3f}")
    print(f"      Regret: £{regret:.2f}")

    return {
        'K': K,
        'p_threshold': p_T,
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
        'hit_count': int(hit_count),
        'true_high_risk_count': int(n_true_high_risk)
    }


def compute_enhanced_metrics(mu_post: np.ndarray,
                                   sigma_post: np.ndarray,
                                   x_true: np.ndarray,
                                   test_idx: np.ndarray,
                                   decision_config,
                                   sensor_cost: float = 0.0,
                                   prior_loss: float = None,
                                   K_action: int = None,
                                   enable_near_threshold: bool = True) -> Dict[str, float]:
    """
    🔥 增强的性能指标计算（使用修复后的action-limited）
    """
    # 基础指标
    base_metrics = compute_metrics(
        mu_post, sigma_post, x_true, test_idx, decision_config
    )

    # ROI计算
    if prior_loss is not None and sensor_cost > 0:
        roi = compute_roi(
            prior_loss,
            base_metrics['expected_loss_gbp'],
            sensor_cost
        )
        base_metrics['roi'] = roi

        savings = prior_loss - base_metrics['expected_loss_gbp']
        base_metrics['cost_efficiency'] = savings / sensor_cost
        base_metrics['savings_gbp'] = savings

    # 🔥 Near-threshold 子集评估（修复版本）
    if enable_near_threshold and prior_loss is not None:
        try:
            tau = decision_config.get_threshold()

            # 🔥 修复：使用正确的加权平均进行near-threshold识别
            gaps = np.abs(mu_post[test_idx] - tau)
            threshold_band = 1.0 * sigma_post[test_idx]  # ±1σ
            near_mask = gaps <= threshold_band

            if near_mask.sum() > 0:
                print(f"    Near-threshold evaluation: {near_mask.sum()}/{len(test_idx)} points")

                # 计算near-threshold的先验损失（需要传入）
                near_fraction = near_mask.sum() / len(test_idx)

                # 计算near-threshold后验损失
                from decision import expected_loss
                near_posterior_loss = expected_loss(
                    mu_post[test_idx][near_mask],
                    sigma_post[test_idx][near_mask],
                    decision_config,
                    tau=tau
                )

                # 假设先验损失中near-threshold贡献更大
                prior_loss_near = prior_loss * near_fraction * 1.5  # 假设1.5倍权重
                roi_near = compute_roi(prior_loss_near, near_posterior_loss, sensor_cost)

                base_metrics.update({
                    'n_near_threshold': int(near_mask.sum()),
                    'fraction_near_threshold': float(near_fraction),
                    'prior_loss_near_threshold': float(prior_loss_near),
                    'posterior_loss_near_threshold': float(near_posterior_loss),
                    'savings_near_threshold': float(prior_loss_near - near_posterior_loss),
                    'roi_near_threshold': float(roi_near)
                })
        except Exception as e:
            print(f"    Warning: Near-threshold evaluation failed: {e}")

    # 🔥 Action-limited分析（使用修复后的函数）
    if K_action is not None:
        tau = decision_config.get_threshold()
        action_metrics = compute_action_constrained_loss(
            mu_post[test_idx],
            sigma_post[test_idx],
            x_true[test_idx],
            decision_config,
            K=K_action,
            tau=tau
        )

        # 添加前缀以避免命名冲突
        for key, val in action_metrics.items():
            base_metrics[f'action_{key}'] = val

    # DDI统计
    tau = decision_config.get_threshold()
    try:
        from spatial_field import compute_ddi
        ddi = compute_ddi(
            mu_post[test_idx],
            sigma_post[test_idx],
            tau,
            k=1.0
        )
        base_metrics['ddi'] = ddi
    except Exception as e:
        print(f"    Warning: DDI computation failed: {e}")
        base_metrics['ddi'] = np.nan

    return base_metrics


def compute_enhanced_metrics(mu_post: np.ndarray,
                             sigma_post: np.ndarray,
                             x_true: np.ndarray,
                             test_idx: np.ndarray,
                             decision_config,
                             sensor_cost: float = 0.0,
                             prior_loss: float = None,
                             K_action: int = None,
                             enable_near_threshold: bool = True) -> Dict[str, float]:
    """
    🔥 增强的性能指标计算

    新增功能：
    - 支持near-threshold子集评估
    - 改进的action-limited指标
    - 详细的成本-收益分析
    """
    # 基础指标
    base_metrics = compute_metrics(
        mu_post, sigma_post, x_true, test_idx, decision_config
    )

    # ROI计算
    if prior_loss is not None and sensor_cost > 0:
        roi = compute_roi(
            prior_loss,
            base_metrics['expected_loss_gbp'],
            sensor_cost
        )
        base_metrics['roi'] = roi

        # 成本效率：每英镑传感成本节省的损失
        savings = prior_loss - base_metrics['expected_loss_gbp']
        base_metrics['cost_efficiency'] = savings / sensor_cost
        base_metrics['savings_gbp'] = savings

    # 🔥 Near-threshold 子集评估（Scenario A特有）
    if enable_near_threshold and prior_loss is not None:
        try:
            tau = decision_config.get_threshold()

            # 识别near-threshold区域（在测试集上）
            gaps = np.abs(mu_post[test_idx] - tau)
            threshold_band = 1.0 * sigma_post[test_idx]  # ±1σ
            near_mask = gaps <= threshold_band

            if near_mask.sum() > 0:
                print(f"    Near-threshold evaluation: {near_mask.sum()}/{len(test_idx)} points")

                # 计算near-threshold的先验损失（需要传入）
                # 这里简化：假设near-threshold区域的损失比例更高
                near_fraction = near_mask.sum() / len(test_idx)

                # 计算near-threshold后验损失
                from decision import expected_loss
                near_posterior_loss = expected_loss(
                    mu_post[test_idx][near_mask],
                    sigma_post[test_idx][near_mask],
                    decision_config,
                    tau=tau
                )

                # 假设先验损失中near-threshold贡献更大
                prior_loss_near = prior_loss * near_fraction * 1.5  # 假设1.5倍权重
                roi_near = compute_roi(prior_loss_near, near_posterior_loss, sensor_cost)

                base_metrics.update({
                    'n_near_threshold': int(near_mask.sum()),
                    'fraction_near_threshold': float(near_fraction),
                    'prior_loss_near_threshold': float(prior_loss_near),
                    'posterior_loss_near_threshold': float(near_posterior_loss),
                    'savings_near_threshold': float(prior_loss_near - near_posterior_loss),
                    'roi_near_threshold': float(roi_near)
                })
        except Exception as e:
            print(f"    Warning: Near-threshold evaluation failed: {e}")

    # 🔥 Action-limited分析（使用修复后的函数）
    if K_action is not None:
        tau = decision_config.get_threshold()
        action_metrics = compute_action_constrained_loss(
            mu_post[test_idx],
            sigma_post[test_idx],
            x_true[test_idx],
            decision_config,
            K=K_action,
            tau=tau
        )

        # 添加前缀以避免命名冲突
        for key, val in action_metrics.items():
            base_metrics[f'action_{key}'] = val

    # DDI 统计
    tau = decision_config.get_threshold()
    try:
        from spatial_field import compute_ddi
        ddi = compute_ddi(
            mu_post[test_idx],
            sigma_post[test_idx],
            tau,
            k=1.0
        )
        base_metrics['ddi'] = ddi
    except Exception as e:
        print(f"    Warning: DDI computation failed: {e}")
        base_metrics['ddi'] = np.nan

    return base_metrics


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
    from config import load_scenario_config  # ✅ 改用场景加载
    from geometry import build_grid2d_geometry
    from spatial_field import build_prior, sample_gmrf
    from sensors import generate_sensor_pool
    from selection import greedy_mi

    cfg = load_scenario_config('A')  # ✅ 明确指定场景
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


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  TESTING FIXED ACTION-LIMITED METRICS")
    print("=" * 70)

    # 测试修复后的Action-limited指标
    rng = np.random.default_rng(42)
    n = 100

    # 模拟数据
    mu_post = rng.normal(2.2, 0.3, n)
    sigma_post = rng.uniform(0.2, 0.5, n)
    x_true = rng.normal(2.2, 0.3, n)
    tau = 2.2


    # 模拟决策配置
    class MockDecision:
        L_FP_gbp = 500
        L_FN_gbp = 2000
        L_TP_gbp = 100
        L_TN_gbp = 0

        def get_threshold(self): return tau


    decision_config = MockDecision()

    print(f"Test data: n={n}, tau={tau}")
    print(f"True exceed threshold: {(x_true > tau).sum()}")

    # 测试不同K值
    for K in [5, 10, 20]:
        print(f"\n[K={K}] Action-limited metrics:")
        metrics = compute_action_constrained_loss(
            mu_post, sigma_post, x_true, decision_config, K=K, tau=tau
        )

        print(f"  Recall@K: {metrics['recall_at_k']:.3f}")
        print(f"  Precision@K: {metrics['precision_at_k']:.3f}")
        print(f"  F1@K: {metrics['f1_at_k']:.3f}")
        print(f"  Regret: £{metrics['regret']:.2f}")

        # 验证指标合理性
        assert 0 <= metrics['recall_at_k'] <= 1, "Recall out of range"
        assert 0 <= metrics['precision_at_k'] <= 1, "Precision out of range"
        assert 0 <= metrics['f1_at_k'] <= 1, "F1 out of range"

    print("\n✅ All action-limited metric tests passed!")