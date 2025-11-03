"""
传感器选择算法集合 - 完整版本

包括：
1. greedy_mi - 贪心互信息
2. greedy_aopt - 贪心A-optimal（迹最小化）
3. greedy_evi_myopic_fast - 快速决策感知EVI（核心创新）
4. maxmin_k_center - 最大最小覆盖
5. uniform_selection - 均匀随机选择（向后兼容）
6. random_selection - 逆成本加权随机选择（向后兼容）

🔥 修复版本 - 2025-01-25
"""
import heapq

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple
from dataclasses import dataclass
from sensors import Sensor, assemble_H_R
from inference import SparseFactor


@dataclass
class HeapItem:
    """堆元素（Python heapq是最小堆，所以存负值）"""
    neg_score: float  # -（gain/cost）
    timestamp: int  # 版本号，用于判断过期
    candidate_id: int

    def __lt__(self, other):
        return self.neg_score < other.neg_score


class LazyGreedySelector:
    """
    Lazy-Greedy选择器基类

    核心思想：
    1. 维护最大堆（存每个候选的estimated score）
    2. 每次弹出堆顶时验证是否过期
    3. 如果过期（当前因子下重算后score下降）→ 更新堆
    4. 如果仍是最大 → 选中并做rank-1更新
    """

    def __init__(self,
                 sensors: List[Sensor],
                 Q_pr,
                 costs: np.ndarray,
                 use_cost: bool = True,
                 prescreen_fraction: float = None,
                 verbose: bool = False):
        """
        Args:
            sensors: 候选传感器列表
            Q_pr: 先验精度矩阵
            costs: 成本数组
            use_cost: 是否使用成本归一化
            prescreen_fraction: MI预筛比例（None=动态计算）
            verbose: 是否打印详细信息
        """
        self.sensors = sensors
        self.Q_pr = Q_pr
        self.costs = costs
        self.use_cost = use_cost
        self.verbose = verbose

        self.n = Q_pr.shape[0]
        self.C = len(sensors)

        # 预处理传感器矩阵
        self.H_rows = []
        self.R_list = []
        for s in sensors:
            h = np.zeros(self.n)
            h[s.idxs] = s.weights
            self.H_rows.append(h)
            self.R_list.append(s.noise_var)

        self.H_rows = np.array(self.H_rows)
        self.R_list = np.array(self.R_list)

        # 因子分解
        self.factor = SparseFactor(Q_pr)

        # 预筛参数
        if prescreen_fraction is None:
            # 🔥 专家建议：B = max(3k, ρC)，这里先算ρ
            # 运行时会根据k动态调整
            self.prescreen_fraction = 0.3
        else:
            self.prescreen_fraction = prescreen_fraction

    def compute_marginal_gain(self, idx: int) -> float:
        """
        计算候选idx的边际增益（子类实现）

        Returns:
            gain: 未归一化的边际增益
        """
        raise NotImplementedError

    def update_factor(self, idx: int):
        """
        选中idx后更新因子（rank-1更新）
        """
        h = self.H_rows[idx]
        r = self.R_list[idx]
        self.factor.rank1_update(h, weight=1.0 / r)

    def prescreen_by_mi(self, k: int) -> np.ndarray:
        """
        🔥 MI预筛：保留 B = max(3k, ρC) 个候选
        """
        n_keep_budget = 3 * k  # 至少3k个（保证理论保证）
        n_keep_pool = int(np.ceil(self.prescreen_fraction * self.C))
        n_keep = max(n_keep_budget, n_keep_pool, k + 10)
        n_keep = min(n_keep, self.C)  # 不超过总数

        if self.verbose:
            print(f"    MI prescreen: computing initial gains for {self.C} candidates...")

        # 批量计算初始MI
        Z = self.factor.solve_multi(self.H_rows.T)
        quad = np.sum(self.H_rows * Z.T, axis=1)
        mi_values = 0.5 * np.log1p(quad / self.R_list)

        # 取top-n_keep
        top_indices = np.argpartition(mi_values, -n_keep)[-n_keep:]

        if self.verbose:
            print(f"    ✓ Kept {n_keep}/{self.C} candidates "
                  f"({100 * n_keep / self.C:.0f}%)")

        return top_indices

    def lazy_greedy_select(self, k: int,
                           use_prescreen: bool = True) -> Tuple[List[int], List[float]]:
        """
        🔥 Lazy-Greedy核心算法

        Returns:
            selected_ids: 选中的候选索引
            marginal_gains: 对应的边际增益
        """
        # === 预筛阶段 ===
        if use_prescreen and self.C > 100:
            alive_candidates = self.prescreen_by_mi(k)
        else:
            alive_candidates = np.arange(self.C)

        alive_set = set(alive_candidates)

        # === 初始化堆 ===
        heap = []
        timestamp = 0  # 全局版本号
        candidate_timestamps = {}  # 记录每个候选的最新版本

        if self.verbose:
            print(f"    Initializing heap with {len(alive_candidates)} candidates...")

        for idx in alive_candidates:
            gain = self.compute_marginal_gain(idx)
            score = gain / self.costs[idx] if self.use_cost else gain

            item = HeapItem(
                neg_score=-score,  # 最小堆，存负值
                timestamp=timestamp,
                candidate_id=idx
            )
            heapq.heappush(heap, item)
            candidate_timestamps[idx] = timestamp

        # === Lazy-Greedy主循环 ===
        selected = []
        marginal_gains = []
        total_cost = 0.0

        recomputes = 0  # 统计重计算次数

        for step in range(k):
            if len(heap) == 0:
                if self.verbose:
                    print(f"    Heap empty at step {step + 1}, stopping.")
                break

            # === 弹出堆顶 ===
            best_item = heapq.heappop(heap)
            idx = best_item.candidate_id
            old_timestamp = best_item.timestamp

            # === 验证是否过期 ===
            # 过期条件：timestamp < 当前全局timestamp（说明因子已更新）
            is_stale = (old_timestamp < timestamp)

            if is_stale:
                # 重新计算当前因子下的边际增益
                gain_new = self.compute_marginal_gain(idx)
                score_new = gain_new / self.costs[idx] if self.use_cost else gain_new

                # 更新堆（带新版本号）
                new_item = HeapItem(
                    neg_score=-score_new,
                    timestamp=timestamp,
                    candidate_id=idx
                )
                heapq.heappush(heap, new_item)
                candidate_timestamps[idx] = timestamp

                recomputes += 1

                # 继续下一轮（不选中，继续验证堆顶）
                continue

            # === 如果没过期，说明这是真正的最大增益 → 选中 ===
            gain_actual = self.compute_marginal_gain(idx)
            score_actual = gain_actual / self.costs[idx] if self.use_cost else gain_actual

            if score_actual <= 0:
                if self.verbose:
                    print(f"    Step {step + 1}: no positive gain, stopping.")
                break

            # 选中
            selected.append(int(idx))
            marginal_gains.append(float(gain_actual))
            total_cost += float(self.costs[idx])

            if self.verbose and (step + 1) % max(1, k // 10) == 0:
                print(f"    Step {step + 1}/{k}: selected #{idx}, "
                      f"gain={gain_actual:.4f}, cost={self.costs[idx]:.0f}")

            # === Rank-1更新因子 ===
            self.update_factor(idx)

            # 从活跃集移除
            alive_set.discard(idx)

            # 🔥 关键：递增全局版本号
            # 这会让堆中所有元素"过期"，下次弹出时会重新验证
            timestamp += 1

        if self.verbose:
            avg_recomputes = recomputes / max(1, len(selected))
            print(f"    ✓ Lazy-Greedy stats: {len(selected)} selected, "
                  f"{recomputes} recomputes (avg {avg_recomputes:.1f} per selection)")

        return selected, marginal_gains


class LazyGreedyMI(LazyGreedySelector):
    """Lazy-Greedy for Mutual Information"""

    def compute_marginal_gain(self, idx: int) -> float:
        """MI边际增益：0.5 * log(1 + h^T Σ h / r)"""
        h = self.H_rows[idx]
        r = self.R_list[idx]

        z = self.factor.solve(h)
        quad = np.dot(h, z)
        mi = 0.5 * np.log1p(quad / r)

        return mi


class LazyGreedyEVI(LazyGreedySelector):
    """Lazy-Greedy for EVI (带测试集评估)"""

    def __init__(self,
                 sensors: List[Sensor],
                 Q_pr,
                 mu_pr: np.ndarray,
                 costs: np.ndarray,
                 decision_config,
                 test_idx: np.ndarray,
                 tau_fixed: float,
                 use_cost: bool = True,
                 prescreen_fraction: float = None,
                 verbose: bool = False):
        """
        EVI版本需要额外参数：
        - mu_pr: 先验均值
        - decision_config: 决策配置
        - test_idx: 测试点索引
        - tau_fixed: 锁定的决策阈值
        """
        super().__init__(sensors, Q_pr, costs, use_cost,
                         prescreen_fraction, verbose)

        self.mu_pr = mu_pr
        self.decision_config = decision_config
        self.test_idx = test_idx
        self.tau_fixed = tau_fixed

        # 预计算测试集的先验对角方差
        from inference import compute_posterior_variance_diagonal
        var_test = compute_posterior_variance_diagonal(self.factor, test_idx)
        self.diag_test = np.maximum(var_test, 1e-12)
        self.sigma_test = np.sqrt(self.diag_test)

        # 预计算Z矩阵（用于快速rank-1更新）
        if self.verbose:
            print(f"    Precomputing Z matrix for {len(test_idx)} test points...")
        self.Z = self.factor.solve_multi(self.H_rows.T)  # (n, C)

        # 先验风险（固定）
        from decision import expected_loss
        mu_test = mu_pr[test_idx]
        self.prior_risk = expected_loss(
            mu_test, self.sigma_test, decision_config,
            test_indices=np.arange(len(test_idx)),
            tau=tau_fixed
        )

    def compute_marginal_gain(self, idx: int) -> float:
        """
        EVI边际增益：prior_risk - posterior_risk
        """
        from decision import expected_loss

        # 当前因子下的测试点方差
        z_test = self.Z[self.test_idx, idx]
        h = self.H_rows[idx]
        r = self.R_list[idx]

        # Sherman-Morrison对角方差更新
        quad = np.dot(h, self.Z[:, idx])
        denom = r + quad
        denom = max(denom, 1e-12)

        diag_post = self.diag_test - (z_test ** 2) / denom
        diag_post = np.maximum(diag_post, 1e-12)
        sigma_post = np.sqrt(diag_post)

        # 后验风险
        mu_test = self.mu_pr[self.test_idx]
        post_risk = expected_loss(
            mu_test, sigma_post, self.decision_config,
            test_indices=np.arange(len(self.test_idx)),
            tau=self.tau_fixed
        )

        # EVI = 风险减少
        evi_gain = self.prior_risk - post_risk
        evi_gain = max(evi_gain, 0.0)  # 防守式钳位

        return evi_gain

    def update_factor(self, idx: int):
        """
        EVI版本需要同时更新：
        1. 因子（rank-1更新）
        2. Z矩阵（Sherman-Morrison）
        3. 测试集对角方差
        4. 先验风险
        """
        h = self.H_rows[idx]
        r = self.R_list[idx]
        z_star = self.Z[:, idx]

        # 计算分母
        quad = np.dot(h, z_star)
        denom = r + quad
        denom = max(denom, 1e-12)

        # 更新测试集方差
        z_test = z_star[self.test_idx]
        self.diag_test = self.diag_test - (z_test ** 2) / denom
        self.diag_test = np.maximum(self.diag_test, 1e-12)
        self.sigma_test = np.sqrt(self.diag_test)

        # 更新先验风险（用于下一轮）
        from decision import expected_loss
        mu_test = self.mu_pr[self.test_idx]
        self.prior_risk = expected_loss(
            mu_test, self.sigma_test, self.decision_config,
            test_indices=np.arange(len(self.test_idx)),
            tau=self.tau_fixed
        )

        # 更新Z矩阵（Sherman-Morrison）
        c = h @ self.Z
        self.Z -= np.outer(z_star, c) / denom

        # 更新因子
        self.factor.rank1_update(h, weight=1.0 / r)



@dataclass
class SelectionResult:
    """传感器选择结果"""
    selected_ids: List[int]
    objective_values: List[float]
    marginal_gains: List[float]
    total_cost: float
    method_name: str


# =====================================================================
# 1. Greedy MI（互信息）
# =====================================================================

def greedy_mi(sensors, k: int, Q_pr, costs: np.ndarray = None,
              lazy: bool = True,
              batch_size: int = 1,
              use_cost: bool = True,
              keep_fraction: float = None,
              verbose: bool = False) -> 'SelectionResult':  # 🔥 添加这个参数
    """
    🔥 Lazy-Greedy MI（带堆优化）

    向后兼容接口，与原greedy_mi签名一致
    """
    from selection import SelectionResult

    C = len(sensors)

    if costs is None:
        costs = np.ones(C, dtype=float)
    else:
        costs = np.asarray(costs, dtype=float)

    # 使用Lazy-Greedy选择器
    selector = LazyGreedyMI(
        sensors=sensors,
        Q_pr=Q_pr,
        costs=costs,
        use_cost=use_cost,
        prescreen_fraction=keep_fraction,  # ✅ 使用参数名
        verbose=verbose  # ✅ 现在有这个参数了
    )

    selected, marginal_gains = selector.lazy_greedy_select(k, use_prescreen=True)

    # 计算累积目标值
    objective_values = []
    cumsum = 0.0
    for mg in marginal_gains:
        cumsum += mg
        objective_values.append(cumsum)

    total_cost = sum(costs[i] for i in selected)

    return SelectionResult(
        selected_ids=selected,
        objective_values=objective_values,
        marginal_gains=marginal_gains,
        total_cost=total_cost,
        method_name="Greedy-MI"
    )

# =====================================================================
# 2. Greedy A-optimal（迹最小化）
# =====================================================================

def greedy_aopt(sensors, k: int, Q_pr, costs: np.ndarray = None,
                n_probes: int = 16, use_cost: bool = True,
                rng: np.random.Generator = None) -> 'SelectionResult':  # 🔥 新增参数
    """
    Greedy A-optimal design (trace minimization)

    🔥 P0-3修复：添加rng参数支持外部随机数生成器

    Args:
        sensors: 候选传感器列表
        k: 选择数量
        Q_pr: 先验精度矩阵
        costs: 成本数组
        n_probes: 迹估计的探针数量
        use_cost: 是否使用成本归一化
        rng: 随机数生成器（🔥 新增，用于探针采样）
    """
    from inference import SparseFactor

    n = Q_pr.shape[0]
    C = len(sensors)

    if costs is None:
        costs = np.ones(C, dtype=float)
    else:
        costs = np.asarray(costs, dtype=float)
        if len(costs) != C:
            raise ValueError(f"Cost array length {len(costs)} doesn't match sensor count {C}")

    # 🔥 P0-3修复：使用传入的rng
    if rng is None:
        import warnings
        warnings.warn(
            "No RNG provided to greedy_aopt, creating new one. "
            "Pass rng from config.get_rng() for reproducibility.",
            UserWarning, stacklevel=2
        )
        rng = np.random.default_rng()  # 🔥 移除硬编码的42

    selected = []
    marginal_gains = []
    objective_values = []
    total_cost = 0.0

    H_rows = []
    R_list = []
    for s in sensors:
        h = np.zeros(n)
        h[s.idxs] = s.weights
        H_rows.append(h)
        R_list.append(s.noise_var)

    H_rows = np.array(H_rows)
    R_list = np.array(R_list)

    factor = SparseFactor(Q_pr)

    # 🔥 使用传入的rng生成探针
    probes = rng.standard_normal((n, n_probes))
    Z_probes = factor.solve_multi(probes)
    trace_current = np.mean(np.sum(probes * Z_probes, axis=0))

    alive = np.ones(C, dtype=bool)

    for step in range(k):
        best_idx = -1
        best_gain = -np.inf
        best_reduction = 0.0

        candidates = np.where(alive)[0]

        for idx in candidates:
            h = H_rows[idx]
            r = R_list[idx]

            z = factor.solve(h)
            quad = np.dot(h, z)
            zz = np.dot(z, z)

            denom = r + quad
            if denom > 1e-12:
                reduction = zz / denom
                gain = reduction / costs[idx] if use_cost else reduction

                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_reduction = reduction

        if best_idx < 0 or best_gain <= 0:
            break

        selected.append(int(best_idx))
        marginal_gains.append(float(best_reduction))
        total_cost += float(costs[best_idx])
        trace_current -= best_reduction
        objective_values.append(float(trace_current))

        h_star = H_rows[best_idx]
        r_star = R_list[best_idx]
        factor.rank1_update(h_star, weight=1.0 / r_star)

        alive[best_idx] = False

    from selection import SelectionResult
    return SelectionResult(
        selected_ids=selected,
        objective_values=objective_values,
        marginal_gains=marginal_gains,
        total_cost=total_cost,
        method_name="Greedy-Aopt"
    )

# =====================================================================
# 3. 🔥 Greedy EVI Myopic Fast（核心创新）
# =====================================================================

def greedy_evi_myopic_fast(
        sensors,
        k: int,
        Q_pr,
        mu_pr: np.ndarray,
        decision_config,
        test_idx: np.ndarray,
        costs: np.ndarray = None,
        n_y_samples: int = 0,              # 🔥 添加这个
        use_cost: bool = True,
        mi_prescreen: bool = True,
        keep_fraction: float = None,
        rng: np.random.Generator = None,
        verbose: bool = False
) -> 'SelectionResult':
    """
    🔥 Lazy-Greedy EVI（带堆优化）

    向后兼容接口
    """
    from selection import SelectionResult

    C = len(sensors)

    if costs is None:
        costs = np.ones(C, dtype=float)
    else:
        costs = np.asarray(costs, dtype=float)

    # 🔥 确保tau已锁定
    if hasattr(decision_config, 'tau_iri') and decision_config.tau_iri is not None:
        tau_fixed = decision_config.tau_iri
    else:
        raise ValueError(
            "tau_iri not set in decision_config. "
            "Call config.lock_decision_threshold(mu_pr) before EVI selection."
        )

    # 使用Lazy-Greedy EVI选择器
    selector = LazyGreedyEVI(
        sensors=sensors,
        Q_pr=Q_pr,
        mu_pr=mu_pr,
        costs=costs,
        decision_config=decision_config,
        test_idx=test_idx,
        tau_fixed=tau_fixed,
        use_cost=use_cost,
        prescreen_fraction=keep_fraction,  # ✅ 使用参数名
        verbose=verbose  # ✅ 参数已存在
    )

    selected, marginal_gains = selector.lazy_greedy_select(k, use_prescreen=True)

    # 计算累积目标值
    objective_values = []
    cumsum = 0.0
    for mg in marginal_gains:
        cumsum += mg
        objective_values.append(cumsum)

    total_cost = sum(costs[i] for i in selected)

    return SelectionResult(
        selected_ids=selected,
        objective_values=objective_values,
        marginal_gains=marginal_gains,
        total_cost=total_cost,
        method_name="Greedy-EVI"
    )

# =====================================================================
# 4. Maxmin k-center
# =====================================================================

def maxmin_k_center(sensors, k: int, coords: np.ndarray,
                    costs: np.ndarray = None, use_cost: bool = True) -> 'SelectionResult':
    """Maxmin k-center (spatial coverage)"""
    import numpy as np
    from scipy.spatial.distance import cdist

    C = len(sensors)

    if costs is None:
        costs = np.ones(C, dtype=float)
    else:
        costs = np.asarray(costs, dtype=float)
        if len(costs) != C:
            raise ValueError(f"Cost array length {len(costs)} doesn't match sensor count {C}")

    sensor_coords = np.array([coords[s.idxs[0]] for s in sensors])
    dist_matrix = cdist(coords, sensor_coords)

    selected = []
    total_cost = 0.0

    avg_dist = dist_matrix.mean(axis=0)
    score = avg_dist / costs if use_cost else avg_dist
    first = int(np.argmax(score))
    selected.append(first)
    total_cost += float(costs[first])

    min_dist = dist_matrix[:, first].copy()

    for step in range(1, k):
        best_idx = -1
        best_score = -np.inf

        for idx in range(C):
            if idx in selected:
                continue

            new_min_dist = np.minimum(min_dist, dist_matrix[:, idx])
            maxmin_dist = new_min_dist.min()

            if use_cost:
                score = maxmin_dist / costs[idx]
            else:
                score = maxmin_dist

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break

        selected.append(int(best_idx))
        total_cost += float(costs[best_idx])
        min_dist = np.minimum(min_dist, dist_matrix[:, best_idx])

    return SelectionResult(
        selected_ids=selected,
        objective_values=[0.0] * len(selected),
        marginal_gains=[0.0] * len(selected),
        total_cost=total_cost,
        method_name="Maxmin"
    )


# =====================================================================
# 5. Uniform Selection（向后兼容）
# =====================================================================

def uniform_selection(sensors: List[Sensor], k: int, Q_pr: sp.spmatrix = None,
                     mu_pr: np.ndarray = None, rng: np.random.Generator = None) -> SelectionResult:
    """均匀随机选择（向后兼容）"""
    if rng is None:
        rng = np.random.default_rng()

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


# =====================================================================
# 6. Random Selection（向后兼容）
# =====================================================================

def random_selection(sensors: List[Sensor], k: int, Q_pr: sp.spmatrix = None,
                    mu_pr: np.ndarray = None, rng: np.random.Generator = None) -> SelectionResult:
    """随机选择（逆成本加权）"""
    if rng is None:
        rng = np.random.default_rng()

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