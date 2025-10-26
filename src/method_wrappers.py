"""
Method wrappers for easy integration into main evaluation pipeline.
Provides unified interface for all selection methods.

🔥 修复版本 - 2025-01-25
修复问题：
1. 统一 costs 参数处理（添加 dtype=float）
2. 修复 greedy_evi 导入名称
3. 添加 maxmin 支持
"""

import numpy as np
from typing import List, Callable
from sensors import Sensor
import scipy.sparse as sp


def get_selection_method(method_name: str, config, geom,
                         x_true: np.ndarray = None,
                         test_idx: np.ndarray = None) -> Callable:
    """
    Get a selection method function with unified signature.
    """
    method_lower = method_name.lower().replace('-', '_').replace(' ', '_')

    # =====================================================================
    # 1. Greedy MI
    # =====================================================================
    if method_lower in ['greedy_mi', 'mi']:
        from selection import greedy_mi

        def wrapper(sensors, k, Q_pr, mu_pr=None):
            # ✅ 修复：确保 costs 长度与 sensors 一致
            n_sensors = len(sensors)
            costs = np.array([s.cost for s in sensors], dtype=float)

            # 验证长度
            assert len(costs) == n_sensors, f"Cost length mismatch: {len(costs)} vs {n_sensors}"

            batch_size = 64
            lazy = True

            if hasattr(config.selection, 'greedy_mi'):
                mi_cfg = config.selection.greedy_mi
                batch_size = mi_cfg.get('batch_size', 64)

            return greedy_mi(
                sensors=sensors,
                k=k,
                Q_pr=Q_pr,
                costs=costs,
                lazy=lazy,
                batch_size=batch_size
            )

        return wrapper

    # =====================================================================
    # 2. Greedy A-optimal
    # =====================================================================
    elif method_lower in ['greedy_aopt', 'greedy-aopt', 'greedy_a', 'aopt', 'a']:
        from selection import greedy_aopt

        def wrapper(sensors, k, Q_pr, mu_pr=None):
            n_sensors = len(sensors)
            costs = np.array([s.cost for s in sensors], dtype=float)
            assert len(costs) == n_sensors

            n_probes = 16
            use_cost = True

            if hasattr(config.selection, 'greedy_aopt'):
                aopt_cfg = config.selection.greedy_aopt
                n_probes = aopt_cfg.get('n_probes', 16)
                use_cost = aopt_cfg.get('use_cost', True)

            return greedy_aopt(
                sensors=sensors,
                k=k,
                Q_pr=Q_pr,
                costs=costs,
                n_probes=n_probes,
                use_cost=use_cost
            )

        return wrapper

    # =====================================================================
    # 3. Greedy EVI (决策感知)
    # =====================================================================
    elif method_lower in ['greedy_evi', 'greedy-evi', 'evi', 'myopic_evi']:
        from selection import greedy_evi_myopic_fast

        if x_true is None:
            raise ValueError("EVI method requires x_true")
        if test_idx is None:
            # 🔥 使用分层抽样替代均匀随机
            test_idx = _stratified_test_sampling(geom, Q_pr, config, n_test=min(300, geom.n))

        def wrapper(sensors, k, Q_pr, mu_pr):
            n_sensors = len(sensors)
            costs = np.array([s.cost for s in sensors], dtype=float)
            assert len(costs) == n_sensors

            rng = config.get_rng()

            # 🔥 优化后的默认参数
            n_y_samples = 0
            use_cost = True
            mi_prescreen = True
            keep_fraction = 0.5  # ✅ 从 0.25 放宽到 0.5

            if hasattr(config.selection, 'greedy_evi'):
                evi_cfg = config.selection.greedy_evi
                n_y_samples = evi_cfg.get('n_y_samples', 0)
                use_cost = evi_cfg.get('use_cost', True)
                mi_prescreen = evi_cfg.get('mi_prescreen', True)
                keep_fraction = evi_cfg.get('keep_fraction', 0.5)  # ✅ 新默认值

            return greedy_evi_myopic_fast(
                sensors=sensors,
                k=k,
                Q_pr=Q_pr,
                mu_pr=mu_pr,
                decision_config=config.decision,
                test_idx=test_idx,
                costs=costs,
                n_y_samples=n_y_samples,
                use_cost=use_cost,
                mi_prescreen=mi_prescreen,
                keep_fraction=keep_fraction,
                rng=rng,
                verbose=False
            )

        return wrapper

    # =====================================================================
    # 4. Maxmin k-center
    # =====================================================================
    elif method_lower in ['maxmin', 'k-center', 'kcenter', 'max-min']:
        from selection import maxmin_k_center

        def wrapper(sensors, k, Q_pr, mu_pr=None):
            n_sensors = len(sensors)
            costs = np.array([s.cost for s in sensors], dtype=float)
            assert len(costs) == n_sensors

            use_cost = True
            if hasattr(config.selection, 'maxmin'):
                maxmin_cfg = config.selection.maxmin
                use_cost = maxmin_cfg.get('use_cost', True)

            return maxmin_k_center(
                sensors=sensors,
                k=k,
                coords=geom.coords,
                costs=costs,
                use_cost=use_cost
            )

        return wrapper

    # =====================================================================
    # 5. Uniform
    # =====================================================================
    elif method_lower in ['uniform', 'uniform_random']:
        from selection import SelectionResult

        def wrapper(sensors, k, Q_pr, mu_pr=None):
            rng = config.get_rng()
            n_sensors = len(sensors)

            if k > n_sensors:
                k = n_sensors

            selected_ids = rng.choice(n_sensors, size=k, replace=False).tolist()
            total_cost = sum(sensors[i].cost for i in selected_ids)

            return SelectionResult(
                selected_ids=selected_ids,
                objective_values=[0.0] * k,
                marginal_gains=[0.0] * k,
                total_cost=total_cost,
                method_name="Uniform"
            )

        return wrapper

    # =====================================================================
    # 6. Random
    # =====================================================================
    elif method_lower == 'random':
        from selection import SelectionResult

        def wrapper(sensors, k, Q_pr, mu_pr=None):
            rng = config.get_rng()
            n_sensors = len(sensors)
            costs = np.array([s.cost for s in sensors], dtype=float)
            assert len(costs) == n_sensors

            weights = 1.0 / (costs + 1.0)
            weights = weights / weights.sum()

            if k > n_sensors:
                k = n_sensors

            selected_ids = rng.choice(n_sensors, size=k, replace=False, p=weights).tolist()
            total_cost = sum(sensors[i].cost for i in selected_ids)

            return SelectionResult(
                selected_ids=selected_ids,
                objective_values=[0.0] * k,
                marginal_gains=[0.0] * k,
                total_cost=total_cost,
                method_name="Random"
            )

        return wrapper

    else:
        raise ValueError(f"Unknown method: {method_name}")


def get_available_methods(config) -> List[str]:
    """Get list of methods configured for evaluation."""
    if hasattr(config.selection, 'methods'):
        return config.selection.methods
    else:
        return ['greedy_mi', 'greedy_aopt', 'uniform', 'random']


def should_use_evi(method_name: str, budget: int, fold_idx: int,
                   config) -> bool:
    """
    决定是否运行 EVI 的跳过策略

    确保：
    1. 非EVI方法总是返回True
    2. must_budgets的预算运行所有折
    3. 其他预算至少保留第1折 + 每N折运行一次

    Args:
        method_name: 方法名称
        budget: 当前预算
        fold_idx: 当前fold索引（从0开始）
        config: 配置对象

    Returns:
        bool: 是否应该运行该fold
    """
    # 确保只对 EVI 方法应用限制
    method_lower = method_name.lower()
    if method_lower not in ['greedy_evi', 'evi', 'greedy-evi', 'myopic_evi']:
        return True

    # 检查 EVI 配置约束
    if hasattr(config.selection, 'greedy_evi'):
        evi_cfg = config.selection.greedy_evi

        # must_budgets - 这些预算必须运行所有折
        must_budgets = set(evi_cfg.get('must_budgets', [10, 30]))
        if budget in must_budgets:
            return True  # 必须运行的预算，不跳过任何fold

        # 检查 budget 约束
        if 'budgets_subset' in evi_cfg:
            budgets_subset = evi_cfg.get('budgets_subset', [])
            if budgets_subset and budget not in budgets_subset:
                return False  # 不在子集中，跳过

        # fold 约束 - 至少保留第1折
        if fold_idx == 0:
            return True  # 第1折总是运行（用于基准测试）

        # 每N折运行一次
        every_n = evi_cfg.get('every_n_folds', 2)
        if every_n and every_n > 1:
            if (fold_idx % every_n) != 0:
                return False

        # max_folds约束（可选）
        max_folds = evi_cfg.get('max_folds')
        if max_folds is not None and fold_idx >= max_folds:
            return False

    return True  # 默认运行


def _stratified_test_sampling(geom, Q_pr, config, n_test: int = 300) -> np.ndarray:
    """
    🔥 按先验方差分层抽样测试集

    高方差区域 → 更多测试点（这些是决策不确定性高的区域）
    低方差区域 → 较少测试点

    Args:
        geom: 几何对象
        Q_pr: 先验精度矩阵
        config: 配置对象
        n_test: 测试集大小

    Returns:
        test_idx: 分层采样的测试索引
    """
    from inference import SparseFactor, compute_posterior_variance_diagonal

    n = geom.n
    rng = config.get_rng()

    # 快速估计先验方差（Hutchinson 近似）
    n_probes = min(16, n // 10)
    if n_probes < 4:
        # 太小的问题，直接均匀采样
        return rng.choice(n, size=min(n_test, n), replace=False)

    try:
        factor = SparseFactor(Q_pr)

        # 采样一些点估计方差分布
        sample_idx = rng.choice(n, size=min(n, 500), replace=False)
        sample_vars = compute_posterior_variance_diagonal(factor, sample_idx)

        # 根据方差分位数分层
        quantiles = np.quantile(sample_vars, [0, 0.33, 0.67, 1.0])
        strata = np.digitize(sample_vars, quantiles[1:-1])  # 0, 1, 2 三层

        # 每层的采样数量（高方差层多采）
        strata_weights = np.array([0.2, 0.3, 0.5])  # 低、中、高方差的权重
        strata_counts = (strata_weights * n_test).astype(int)
        strata_counts[-1] = n_test - strata_counts[:-1].sum()  # 确保总和为 n_test

        # 分层采样
        test_idx_list = []
        for stratum_id in range(3):
            stratum_mask = (strata == stratum_id)
            stratum_indices = sample_idx[stratum_mask]

            if len(stratum_indices) > 0:
                n_sample = min(strata_counts[stratum_id], len(stratum_indices))
                sampled = rng.choice(stratum_indices, size=n_sample, replace=False)
                test_idx_list.extend(sampled)

        test_idx = np.array(test_idx_list)

        print(f"  ✓ Stratified test sampling: {len(test_idx)} points")
        print(f"    Strata sizes: {[np.sum(strata == i) for i in range(3)]}")

        return test_idx

    except Exception as e:
        print(f"  Warning: Stratified sampling failed ({e}), using uniform")
        return rng.choice(n, size=min(n_test, n), replace=False)