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

    确保：
    1. 非EVI方法总是返回True
    2. must_budgets的预算运行所有折
    3. 其他预算至少保留第1折 + 每N折运行一次
    """
    # ✅ 确保只对 EVI 方法应用限制
    method_lower = method_name.lower()
    if method_lower not in ['greedy_evi', 'evi', 'greedy-evi', 'myopic_evi']:
        return True

    # 检查 EVI 配置约束
    if hasattr(config.selection, 'greedy_evi'):
        evi_cfg = config.selection.greedy_evi

        # 🔥 must_budgets - 这些预算必须运行所有折
        must_budgets = set(evi_cfg.get('must_budgets', [5, 20, 40]))
        if budget in must_budgets:
            return True  # 必须运行的预算，不跳过任何折

        # 检查 budget 约束
        if 'budgets_subset' in evi_cfg:
            budgets_subset = evi_cfg.get('budgets_subset', [])
            if budgets_subset and budget not in budgets_subset:
                return False  # 不在子集中，跳过

        # 🔥 改进：fold 约束 - 至少保留第1折
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
