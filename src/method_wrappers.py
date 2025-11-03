"""
Method wrappers for parallel processing (Pickle-safe version)
使用顶层类替代嵌套函数以支持多进程序列化
"""

import numpy as np
from typing import List, Callable
from sensors import Sensor
import scipy.sparse as sp


# ============================================================================
# 🔥 修复：使用顶层类替代嵌套函数（支持 pickle）
# ============================================================================

class GreedyAoptWrapper:
    """Greedy A-optimal wrapper (pickle-safe)"""

    def __init__(self, config):
        self.config = config
        self.n_probes = 16
        self.use_cost = True

        if hasattr(config.selection, 'greedy_aopt'):
            aopt_cfg = config.selection.greedy_aopt
            self.n_probes = aopt_cfg.get('n_probes', 16)
            self.use_cost = aopt_cfg.get('use_cost', True)

    def __call__(self, sensors, k, Q_pr, mu_pr=None):
        from selection import greedy_aopt

        n_sensors = len(sensors)
        costs = np.array([s.cost for s in sensors], dtype=float)
        assert len(costs) == n_sensors

        # 🔥 P0-3修复：传递rng
        rng = self.config.get_rng()

        return greedy_aopt(
            sensors=sensors,
            k=k,
            Q_pr=Q_pr,
            costs=costs,
            n_probes=self.n_probes,
            use_cost=self.use_cost,
            rng=rng  # 🔥 新增参数
        )

class GreedyMIWrapper:
    """Greedy MI selection wrapper (pickle-safe)"""

    def __init__(self, config):
        self.config = config
        self.batch_size = 64
        self.lazy = True
        self.use_cost = True
        self.keep_fraction = None

        if hasattr(config.selection, 'greedy_mi'):
            mi_cfg = config.selection.greedy_mi
            self.batch_size = mi_cfg.get('batch_size', 64)
            self.use_cost = mi_cfg.get('use_cost', True)

            keep_frac_raw = mi_cfg.get('keep_fraction')
            if keep_frac_raw is not None:
                self.keep_fraction = keep_frac_raw

    def __call__(self, sensors, k, Q_pr, mu_pr=None):
        from selection import greedy_mi

        n_sensors = len(sensors)
        costs = np.array([s.cost for s in sensors], dtype=float)
        assert len(costs) == n_sensors

        return greedy_mi(
            sensors=sensors,
            k=k,
            Q_pr=Q_pr,
            costs=costs,
            lazy=self.lazy,
            batch_size=self.batch_size,
            use_cost=self.use_cost,
            keep_fraction=self.keep_fraction
        )


class GreedyEVIWrapper:
    """Greedy EVI wrapper (pickle-safe)"""

    def __init__(self, config, geom, x_true, test_idx):
        self.config = config
        self.geom = geom
        self.x_true = x_true
        self.test_idx = test_idx

        self.n_y_samples = 0
        self.use_cost = True
        self.mi_prescreen = True
        self.keep_fraction = None

        if hasattr(config.selection, 'greedy_evi'):
            evi_cfg = config.selection.greedy_evi
            self.n_y_samples = evi_cfg.get('n_y_samples', 0)
            self.use_cost = evi_cfg.get('use_cost', True)
            self.mi_prescreen = evi_cfg.get('mi_prescreen', True)

            keep_frac_raw = evi_cfg.get('keep_fraction')
            if keep_frac_raw is not None:
                self.keep_fraction = keep_frac_raw

    def __call__(self, sensors, k, Q_pr, mu_pr):
        from selection import greedy_evi_myopic_fast

        n_sensors = len(sensors)
        costs = np.array([s.cost for s in sensors], dtype=float)
        assert len(costs) == n_sensors

        # ✅ 已经使用config.get_rng()
        rng = self.config.get_rng()

        return greedy_evi_myopic_fast(
            sensors=sensors,
            k=k,
            Q_pr=Q_pr,
            mu_pr=mu_pr,
            decision_config=self.config.decision,
            test_idx=self.test_idx,
            costs=costs,
            n_y_samples=self.n_y_samples,
            use_cost=self.use_cost,
            mi_prescreen=self.mi_prescreen,
            keep_fraction=self.keep_fraction,
            rng=rng,
            verbose=False
        )


class MaxminWrapper:
    """Maxmin k-center wrapper (pickle-safe)"""

    def __init__(self, config, geom):
        self.config = config
        self.coords = geom.coords
        self.use_cost = True

        if hasattr(config.selection, 'maxmin'):
            maxmin_cfg = config.selection.maxmin
            self.use_cost = maxmin_cfg.get('use_cost', True)

    def __call__(self, sensors, k, Q_pr, mu_pr=None):
        from selection import maxmin_k_center

        n_sensors = len(sensors)
        costs = np.array([s.cost for s in sensors], dtype=float)
        assert len(costs) == n_sensors

        return maxmin_k_center(
            sensors=sensors,
            k=k,
            coords=self.coords,
            costs=costs,
            use_cost=self.use_cost
        )


class UniformWrapper:
    """Uniform random wrapper (pickle-safe)"""

    def __init__(self, config):
        self.config = config

    def __call__(self, sensors, k, Q_pr, mu_pr=None):
        from selection import SelectionResult

        # ✅ 已经使用config.get_rng()
        rng = self.config.get_rng()
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


class RandomWrapper:
    """Random (cost-weighted) wrapper (pickle-safe)"""

    def __init__(self, config):
        self.config = config

    def __call__(self, sensors, k, Q_pr, mu_pr=None):
        from selection import SelectionResult

        # ✅ 已经使用config.get_rng()
        rng = self.config.get_rng()
        n_sensors = len(sensors)
        costs = np.array([s.cost for s in sensors], dtype=float)

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


# ============================================================================
# 🔥 修复后的工厂函数
# ============================================================================

def get_selection_method(method_name: str, config, geom,
                         x_true: np.ndarray = None,
                         test_idx: np.ndarray = None) -> Callable:
    """Get a selection method wrapper (pickle-safe version)"""
    method_lower = method_name.lower().replace('-', '_').replace(' ', '_')

    if method_lower in ['greedy_mi', 'mi']:
        return GreedyMIWrapper(config)

    elif method_lower in ['greedy_aopt', 'greedy-aopt', 'greedy_a', 'aopt', 'a']:
        return GreedyAoptWrapper(config)

    elif method_lower in ['greedy_evi', 'greedy-evi', 'evi', 'myopic_evi']:
        if x_true is None:
            raise ValueError("EVI method requires x_true")
        if test_idx is None:
            test_idx = _stratified_test_sampling(geom, compute_Q_pr(geom, config),
                                                 config, n_test=min(300, geom.n))
        return GreedyEVIWrapper(config, geom, x_true, test_idx)

    elif method_lower in ['maxmin', 'k-center', 'kcenter', 'max-min']:
        return MaxminWrapper(config, geom)

    elif method_lower in ['uniform', 'uniform_random']:
        return UniformWrapper(config)

    elif method_lower == 'random':
        return RandomWrapper(config)

    else:
        raise ValueError(f"Unknown method: {method_name}")


def get_available_methods(config) -> List[str]:
    """Get list of methods configured for evaluation."""
    if hasattr(config.selection, 'methods'):
        return config.selection.methods
    else:
        return ['greedy_mi', 'greedy_aopt', 'uniform', 'random']



def should_use_evi(method_name: str, budget: int, fold_idx: int,
                   config, strict_mode: bool = True) -> bool:
    """
    ✅ 修复版：决定是否运行EVI的跳过策略

    关键改进：
    - 默认strict_mode=True：所有fold/budget都运行（确保评测公平）
    - 只在显式设置strict_mode=False时才应用跳过逻辑

    Args:
        method_name: 方法名称
        budget: 预算
        fold_idx: fold索引
        config: 配置对象
        strict_mode: 🔥 严格模式（默认True，禁用跳过）

    Returns:
        是否运行该fold/budget组合
    """
    method_lower = method_name.lower()
    if method_lower not in ['greedy_evi', 'evi', 'greedy-evi', 'myopic_evi']:
        return True

    # 🔥 关键修复：默认运行所有fold/budget
    if strict_mode:
        return True

    # 以下是旧的跳过逻辑（仅在strict_mode=False时启用）
    if hasattr(config.selection, 'greedy_evi'):
        evi_cfg = config.selection.greedy_evi

        # must_budgets - 这些预算必须运行所有折
        must_budgets = set(evi_cfg.get('must_budgets', []))
        if budget in must_budgets:
            return True

        # 检查budget约束
        if 'budgets_subset' in evi_cfg:
            budgets_subset = evi_cfg.get('budgets_subset', [])
            if budgets_subset and budget not in budgets_subset:
                return False

        # fold约束 - 至少保留第1折
        if fold_idx == 0:
            return True

        # 每N折运行一次
        every_n = evi_cfg.get('every_n_folds', 1)  # 默认1=运行所有
        if every_n and every_n > 1:
            if (fold_idx % every_n) != 0:
                return False

        # max_folds约束（可选）
        max_folds = evi_cfg.get('max_folds')
        if max_folds is not None and fold_idx >= max_folds:
            return False

    return True


def _stratified_test_sampling(geom, Q_pr, config, n_test: int = 300) -> np.ndarray:
    """
    分层测试集采样

    ✅ 已经正确使用config.get_rng()
    """
    from inference import SparseFactor, compute_posterior_variance_diagonal

    n = geom.n
    rng = config.get_rng()  # ✅ 正确使用

    n_probes = min(16, n // 10)
    if n_probes < 4:
        return rng.choice(n, size=min(n_test, n), replace=False)

    try:
        factor = SparseFactor(Q_pr)
        sample_idx = rng.choice(n, size=min(n, 500), replace=False)
        sample_vars = compute_posterior_variance_diagonal(factor, sample_idx)

        quantiles = np.quantile(sample_vars, [0, 0.33, 0.67, 1.0])
        strata = np.digitize(sample_vars, quantiles[1:-1])

        strata_weights = np.array([0.2, 0.3, 0.5])
        strata_counts = (strata_weights * n_test).astype(int)
        strata_counts[-1] = n_test - strata_counts[:-1].sum()

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

        return test_idx

    except Exception as e:
        print(f"  Warning: Stratified sampling failed ({e}), using uniform")
        return rng.choice(n, size=min(n_test, n), replace=False)


def compute_Q_pr(geom, config):
    """快速构建先验精度矩阵（用于测试集采样）"""
    from spatial_field import build_prior
    Q_pr, _ = build_prior(geom, config.prior)
    return Q_pr

