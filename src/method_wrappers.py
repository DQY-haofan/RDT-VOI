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
            rng = np.random.default_rng(config.experiment.seed)
            n_test = min(200, geom.n)
            test_idx = rng.choice(geom.n, size=n_test, replace=False)

        def wrapper(sensors, k, Q_pr, mu_pr):
            n_sensors = len(sensors)
            costs = np.array([s.cost for s in sensors], dtype=float)
            assert len(costs) == n_sensors

            rng = config.get_rng()

            n_y_samples = 0
            use_cost = True
            mi_prescreen = True
            keep_fraction = 0.25

            if hasattr(config.selection, 'greedy_evi'):
                evi_cfg = config.selection.greedy_evi
                n_y_samples = evi_cfg.get('n_y_samples', 0)
                use_cost = evi_cfg.get('use_cost', True)
                mi_prescreen = evi_cfg.get('mi_prescreen', True)
                keep_fraction = evi_cfg.get('keep_fraction', 0.25)

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