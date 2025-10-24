"""
Method wrappers for easy integration into main evaluation pipeline.
Provides unified interface for all selection methods.
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
    # 统一方法名
    method_lower = method_name.lower().replace('-', '_').replace(' ', '_')

    if method_lower in ['greedy_mi', 'mi']:
        from selection import greedy_mi

        def wrapper(sensors, k, Q_pr, mu_pr=None):
            costs = np.array([s.cost for s in sensors])

            # 🔥 修复：只传入 greedy_mi 实际支持的参数
            # greedy_mi 签名：(sensors, k, Q_pr, costs=None, lazy=True, batch_size=64)

            # 从配置获取参数（如果存在）
            batch_size = 64  # 默认值
            lazy = True  # 默认值

            if hasattr(config.selection, 'greedy_mi'):
                mi_cfg = config.selection.greedy_mi
                batch_size = mi_cfg.get('batch_size', 64)
                # lazy 参数目前配置文件没有，使用默认值

            return greedy_mi(
                sensors=sensors,
                k=k,
                Q_pr=Q_pr,
                costs=costs,
                lazy=lazy,
                batch_size=batch_size
            )

        return wrapper

    elif method_lower in ['greedy_aopt', 'greedy-aopt', 'greedy_a', 'aopt', 'a']:
        from selection import greedy_aopt

        def wrapper(sensors, k, Q_pr, mu_pr=None):
            costs = np.array([s.cost for s in sensors])

            # 从配置获取参数
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

    elif method_lower in ['greedy_evi', 'greedy-evi', 'evi', 'myopic_evi']:
        from selection import greedy_evi_myopic

        if x_true is None:
            raise ValueError("EVI method requires x_true")
        if test_idx is None:
            # 使用随机子集
            rng = np.random.default_rng(config.experiment.seed)
            n_test = min(200, geom.n)
            test_idx = rng.choice(geom.n, size=n_test, replace=False)

        def wrapper(sensors, k, Q_pr, mu_pr):
            costs = np.array([s.cost for s in sensors])
            rng = config.get_rng()

            # 从配置获取参数
            n_y_samples = 10
            use_cost = True

            if hasattr(config.selection, 'greedy_evi'):
                evi_cfg = config.selection.greedy_evi
                n_y_samples = evi_cfg.get('n_y_samples', 10)
                use_cost = evi_cfg.get('use_cost', True)

            return greedy_evi_myopic(
                sensors=sensors,
                k=k,
                Q_pr=Q_pr,
                mu_pr=mu_pr,
                decision_config=config.decision,
                test_idx=test_idx,
                costs=costs,
                n_y_samples=n_y_samples,
                use_cost=use_cost,
                rng=rng,
                verbose=False  # 在批量运行时关闭详细输出
            )

        return wrapper

    elif method_lower in ['maxmin', 'k-center', 'kcenter', 'max-min']:
        from selection import maxmin_k_center

        def wrapper(sensors, k, Q_pr, mu_pr=None):
            costs = np.array([s.cost for s in sensors])

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

    elif method_lower == 'random':
        from selection import SelectionResult

        def wrapper(sensors, k, Q_pr, mu_pr=None):
            rng = config.get_rng()
            n_sensors = len(sensors)
            costs = np.array([s.cost for s in sensors])

            # 逆成本加权
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
    🔥 修复版本：更合理的 EVI 跳过策略

    Determine if EVI should be run for this (method, budget, fold) combination.

    Since EVI is expensive, we can run it on a subset of configurations.

    ⚠️  WARNING: This function should ONLY apply constraints to EVI methods!
                For other methods, always return True (run all configurations).

    Args:
        method_name: Name of the method (should be 'greedy_evi' or similar)
        budget: Budget value
        fold_idx: Fold index (0-based)
        config: Configuration object

    Returns:
        True if should run, False if should skip

    修改：
    1. 对非 EVI 方法总是返回 True
    2. 添加 must_budgets 配置 - 这些预算必须运行所有折
    3. 改进 max_folds 逻辑 - 至少保留第 1 折
    """
    # ✅ 确保只对 EVI 方法应用限制
    method_lower = method_name.lower()
    if method_lower not in ['greedy_evi', 'evi', 'greedy-evi', 'myopic_evi']:
        # 非 EVI 方法总是运行
        return True

    # 检查 EVI 配置约束
    if hasattr(config.selection, 'greedy_evi'):
        evi_cfg = config.selection.greedy_evi

        # 🔥 新增：must_budgets - 这些预算必须运行所有折
        if 'must_budgets' in evi_cfg and evi_cfg['must_budgets']:
            if budget in evi_cfg['must_budgets']:
                return True  # 必须运行的预算，不跳过任何折

        # 检查 budget 约束
        if 'budgets_subset' in evi_cfg:
            budgets_subset = evi_cfg['budgets_subset']
            if budgets_subset and budget not in budgets_subset:
                # 如果定义了 budgets_subset 且当前 budget 不在内，跳过
                return False

        # 🔥 改进：fold 约束 - 至少保留第 1 折
        if 'max_folds' in evi_cfg:
            max_folds = evi_cfg['max_folds']
            if max_folds is not None:
                if fold_idx == 0:
                    # 第 1 折总是运行（用于基准测试）
                    return True
                elif fold_idx >= max_folds:
                    # 超过最大折数，跳过
                    return False

        # 🔥 新增：every_n_folds - 每 N 折运行一次
        if 'every_n_folds' in evi_cfg:
            every_n = evi_cfg['every_n_folds']
            if every_n and every_n > 1:
                # 第 0 折 + 每 N 折
                if fold_idx > 0 and (fold_idx % every_n) != 0:
                    return False

    # 默认运行
    return True
