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

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple
from dataclasses import dataclass
from sensors import Sensor, assemble_H_R


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
              lazy: bool = True, batch_size: int = 1) -> 'SelectionResult':
    """
    Greedy mutual information maximization (批量优化版本)

    Args:
        sensors: 候选传感器列表
        k: 要选择的传感器数量
        Q_pr: 先验精度矩阵
        costs: 传感器成本数组（长度必须等于len(sensors)）
        lazy: 是否使用lazy evaluation
        batch_size: 批量大小
    """
    import numpy as np
    import scipy.sparse as sp
    from inference import SparseFactor

    n = Q_pr.shape[0]
    C = len(sensors)

    # ✅ 确保costs维度正确
    if costs is None:
        costs = np.ones(C, dtype=float)
    else:
        costs = np.asarray(costs, dtype=float)
        if len(costs) != C:
            raise ValueError(f"Cost array length {len(costs)} doesn't match sensor count {C}")

    # 初始化
    selected = []
    marginal_gains = []
    objective_values = []
    total_cost = 0.0

    # 预计算所有候选的h向量（稠密）
    H_rows = []
    R_list = []
    for s in sensors:
        # 构造h向量（n维）
        h = np.zeros(n)
        h[s.idxs] = s.weights
        H_rows.append(h)
        R_list.append(s.noise_var)

    H_rows = np.array(H_rows)  # (C, n)
    R_list = np.array(R_list)  # (C,)

    # 初始因子
    factor = SparseFactor(Q_pr)

    # 批量计算初始MI（如果需要预筛选）
    if lazy and C > 100:
        # 一次性求解 Z = Σ H^T
        Z = factor.solve_multi(H_rows.T)  # (n, C)

        # ✅ 关键修复：计算 h^T Σ h = sum over n (H[c,n] * Z[n,c])
        # H_rows: (C, n)
        # Z.T: (C, n)
        # quad[c] = sum_n H_rows[c,n] * Z.T[c,n] = sum_n H_rows[c,n] * Z[n,c]
        quad = np.sum(H_rows * Z.T, axis=1)  # (C,) ✅ 正确！

        mi_values = 0.5 * np.log1p(quad / R_list)
    else:
        mi_values = None

    # Greedy循环
    alive = np.ones(C, dtype=bool)

    for step in range(k):
        best_idx = -1
        best_gain = -np.inf
        best_mi = 0.0

        # 候选评估
        candidates = np.where(alive)[0]

        for idx in candidates:
            h = H_rows[idx]  # (n,)
            r = R_list[idx]

            # 计算边际MI
            z = factor.solve(h)  # (n,)
            quad = np.dot(h, z)
            mi = 0.5 * np.log1p(quad / r)

            # 成本归一化得分
            gain = mi / costs[idx]

            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_mi = mi

        if best_idx < 0 or best_gain <= 0:
            break

        # 记录选择
        selected.append(int(best_idx))
        marginal_gains.append(float(best_mi))
        total_cost += float(costs[best_idx])
        objective_values.append(
            objective_values[-1] + best_mi if objective_values else best_mi
        )

        # 更新：rank-1增量
        h_star = H_rows[best_idx]
        r_star = R_list[best_idx]

        # Rank-1 update: Q_new = Q + (1/r) * h h^T
        factor.rank1_update(h_star, weight=1.0 / r_star)

        # 标记已选
        alive[best_idx] = False

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
                n_probes: int = 16, use_cost: bool = True) -> 'SelectionResult':
    """
    Greedy A-optimal design (trace minimization)

    使用 Hutchinson++ 估计 trace(Σ)
    """
    import numpy as np
    from inference import SparseFactor

    n = Q_pr.shape[0]
    C = len(sensors)

    # ✅ 修复costs维度
    if costs is None:
        costs = np.ones(C, dtype=float)
    else:
        costs = np.asarray(costs, dtype=float)
        if len(costs) != C:
            raise ValueError(f"Cost array length {len(costs)} doesn't match sensor count {C}")

    # 初始化
    selected = []
    marginal_gains = []
    objective_values = []
    total_cost = 0.0

    # 预计算h向量
    H_rows = []
    R_list = []
    for s in sensors:
        h = np.zeros(n)
        h[s.idxs] = s.weights
        H_rows.append(h)
        R_list.append(s.noise_var)

    H_rows = np.array(H_rows)  # (C, n)
    R_list = np.array(R_list)

    # 初始因子和trace估计
    factor = SparseFactor(Q_pr)

    # Hutchinson++ probes
    rng = np.random.default_rng(42)
    probes = rng.standard_normal((n, n_probes))
    Z_probes = factor.solve_multi(probes)  # (n, n_probes)
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

            # 计算trace reduction
            z = factor.solve(h)
            quad = np.dot(h, z)

            # Sherman-Morrison: trace(Σ') = trace(Σ) - quad/(r + quad)
            denom = r + quad
            if denom > 1e-12:
                reduction = quad / denom
                gain = reduction / costs[idx] if use_cost else reduction

                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_reduction = reduction

        if best_idx < 0 or best_gain <= 0:
            break

        # 记录
        selected.append(int(best_idx))
        marginal_gains.append(float(best_reduction))
        total_cost += float(costs[best_idx])
        trace_current -= best_reduction
        objective_values.append(float(trace_current))

        # Rank-1 update
        h_star = H_rows[best_idx]
        r_star = R_list[best_idx]
        factor.rank1_update(h_star, weight=1.0 / r_star)

        alive[best_idx] = False

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
        n_y_samples: int = 0,
        use_cost: bool = True,
        mi_prescreen: bool = True,
        keep_fraction: float = 0.25,
        rng: np.random.Generator = None,
        verbose: bool = False
) -> 'SelectionResult':
    """
    🔥 快速版 Myopic EVI (决策感知) - 使用 MI 预筛 + rank-1 更新

    关键优化：
    1. 不为每个候选做因子化；每步仅一次稀疏解 + 纯向量代数
    2. 后验协方差与 y 无关，用 rank-1 闭式更新
    3. 只在 test_idx 上计算风险
    4. MI预筛选减少候选数量
    """
    import numpy as np
    import scipy.sparse as sp
    from inference import SparseFactor
    from decision import expected_loss

    if rng is None:
        rng = np.random.default_rng()

    n = Q_pr.shape[0]
    C = len(sensors)

    if costs is None:
        costs = np.array([s.cost for s in sensors], dtype=float)

    if verbose:
        print(f"\n  🚀 Fast Greedy-EVI: n={n}, candidates={C}, budget={k}, test={len(test_idx)}")

    # ---------------------------
    # 1. 先验因子与测试集先验方差
    # ---------------------------
    F = SparseFactor(Q_pr)

    # 用单位基的子集一次多RHS解出 diag(Σ)_test
    I_sub = np.zeros((n, len(test_idx)), dtype=float)
    I_sub[test_idx, np.arange(len(test_idx))] = 1.0
    Z_sub = F.solve_multi(I_sub)
    diag_test = np.einsum('ij,ij->j', Z_sub[test_idx, :], I_sub[test_idx, :])
    diag_test = np.maximum(diag_test, 1e-12)
    sigma_test = np.sqrt(diag_test)
    mu_test = mu_pr[test_idx].copy()

    # 计算先验风险
    prior_risk = expected_loss(
        mu_test,
        sigma_test,
        decision_config,
        test_indices=np.arange(len(test_idx))
    )

    if verbose:
        print(f"    Prior risk: £{prior_risk:.2f}")
        print(f"    Prior σ on test: mean={sigma_test.mean():.4f}, std={sigma_test.std():.4f}")

    # ---------------------------
    # 2. 预取每个候选的稀疏 h（行向量）
    # ---------------------------
    idx_list = [s.idxs for s in sensors]
    w_list = [s.weights for s in sensors]
    r_list = np.array([s.noise_var for s in sensors], dtype=float)

    # 把所有候选合成稠密 H（行=候选，列=n）
    H_dense = np.stack(
        [np.bincount(idxs, weights=w, minlength=n) for idxs, w in zip(idx_list, w_list)],
        axis=0
    )  # C × n

    # ---------------------------
    # 3. MI 预筛：一次多 RHS 得到 Z=Σ H^T
    # ---------------------------
    keep_mask = np.ones(C, dtype=bool)
    original_indices = np.arange(C)  # 保存原始索引映射

    if mi_prescreen and C > 50:
        if verbose:
            print(f"    🔍 MI prescreening over {C} candidates ...")

        Z = F.solve_multi(H_dense.T)  # n × C

        # 每个候选的 quad = h^T Σ h = h^T z_h
        quad = np.einsum('ij,ij->j', H_dense, Z.T)  # (C,)
        mi = 0.5 * np.log1p(quad / r_list)

        n_keep = max(20, int(C * keep_fraction))
        keep_idx = np.argpartition(mi, -n_keep)[-n_keep:]
        keep_mask[:] = False
        keep_mask[keep_idx] = True

        # 更新所有数据结构
        H_dense = H_dense[keep_mask, :]
        r_list = r_list[keep_mask]
        costs = costs[keep_mask]
        idx_list = [idx_list[i] for i in range(C) if keep_mask[i]]
        w_list = [w_list[i] for i in range(C) if keep_mask[i]]
        original_indices = original_indices[keep_mask]
        Z = Z[:, keep_mask]  # n × C_new
        C = H_dense.shape[0]

        if verbose:
            print(f"    ✓ kept {C} ({100 * C / len(keep_mask):.0f}%), "
                  f"MI∈[{mi[keep_mask].min():.3f},{mi[keep_mask].max():.3f}] nats")
    else:
        Z = F.solve_multi(H_dense.T)  # n × C

    # ---------------------------
    # 4. Greedy 循环：每步 1 次解 + 向量代数
    # ---------------------------
    selected = []
    mg = []
    obj = []
    tot_cost = 0.0
    alive = np.ones(C, dtype=bool)  # 当前未被选中的候选

    for step in range(k):
        if verbose:
            print(f"    Step {step + 1}/{k}")

        # 对所有仍在池内的候选，计算"加它后的 posterior 风险"
        # posterior diag on test: diag' = diag - (z_h_test^2)/(r + h^T z_h)
        zt = Z[test_idx, :]  # m_test × C
        num = np.sum(zt * zt, axis=0)  # (C,)
        denom = r_list + np.einsum('ij,ij->j', H_dense, Z.T)  # (C,)
        denom = np.maximum(denom, 1e-12)

        # 逐候选得到 Σ' 的 test 对角
        diag_post_all = diag_test[:, None] - (zt * zt) / denom[None, :]  # m_test × C
        diag_post_all = np.maximum(diag_post_all, 1e-12)

        # 计算 posterior 风险
        sigma_post_all = np.sqrt(diag_post_all)

        # 向量化计算风险
        post_risk = np.empty(C)
        for j in range(C):
            if not alive[j]:
                post_risk[j] = np.inf
                continue
            post_risk[j] = expected_loss(
                mu_test,
                sigma_post_all[:, j],
                decision_config,
                test_indices=np.arange(len(test_idx))
            )

        # 计算 EVI 增益
        evi_gain = prior_risk - post_risk  # (C,)
        score = evi_gain / costs if use_cost else evi_gain
        best = int(np.argmax(score))

        if not np.isfinite(score[best]) or score[best] <= 0:
            if verbose:
                print("      ⚠️  no positive EVI gain; stopping.")
            break

        # 记录（使用原始索引）
        selected.append(int(original_indices[best]))
        mg.append(float(evi_gain[best]))
        tot_cost += float(costs[best])
        obj.append(obj[-1] + mg[-1] if obj else mg[-1])

        if verbose:
            print(f"      pick #{step + 1}: cand={original_indices[best]}, "
                  f"ΔEVI=£{mg[-1]:.2f}, cost=£{costs[best]:.0f}")

        # ---- Rank-1 更新 Z 与 diag_test ----
        z_star = Z[:, best]
        h_star = H_dense[best, :]
        den = denom[best]

        # 更新 test 对角
        diag_test = diag_test - (z_star[test_idx] ** 2) / den
        diag_test = np.maximum(diag_test, 1e-12)
        sigma_test = np.sqrt(diag_test)
        prior_risk = expected_loss(
            mu_test,
            sigma_test,
            decision_config,
            test_indices=np.arange(len(test_idx))
        )

        # 更新 Z：Z' = Z - z_* (h_*^T Z)/den
        c = h_star @ Z  # (C,) = (n,) @ (n×C)
        Z -= np.outer(z_star, c) / den  # (n×1) (1×C) / scalar

        # 标记该候选失效
        alive[best] = False

    return SelectionResult(
        selected_ids=selected,
        objective_values=obj,
        marginal_gains=mg,
        total_cost=tot_cost,
        method_name="Greedy-EVI-fast"
    )
# =====================================================================
# 4. Maxmin k-center
# =====================================================================

def maxmin_k_center(sensors, k: int, coords: np.ndarray,
                    costs: np.ndarray = None, use_cost: bool = True) -> 'SelectionResult':
    """
    Maxmin k-center (spatial coverage)

    选择使最小覆盖距离最大化的传感器
    """
    import numpy as np
    from scipy.spatial.distance import cdist

    C = len(sensors)

    # ✅ 修复costs维度
    if costs is None:
        costs = np.ones(C, dtype=float)
    else:
        costs = np.asarray(costs, dtype=float)
        if len(costs) != C:
            raise ValueError(f"Cost array length {len(costs)} doesn't match sensor count {C}")

    # 获取传感器位置（使用第一个idxs作为代表）
    sensor_coords = np.array([coords[s.idxs[0]] for s in sensors])

    # 计算所有点到传感器的距离矩阵
    dist_matrix = cdist(coords, sensor_coords)  # (n, C)

    selected = []
    total_cost = 0.0

    # 初始化：选择离所有点平均距离最远的传感器
    avg_dist = dist_matrix.mean(axis=0)
    if use_cost:
        score = avg_dist / costs
    else:
        score = avg_dist
    first = int(np.argmax(score))
    selected.append(first)
    total_cost += float(costs[first])

    # 跟踪每个点到已选传感器的最小距离
    min_dist = dist_matrix[:, first].copy()

    for step in range(1, k):
        # 对每个未选传感器，计算如果选它，最小距离会如何变化
        best_idx = -1
        best_score = -np.inf

        for idx in range(C):
            if idx in selected:
                continue

            # 计算新的最小距离
            new_min_dist = np.minimum(min_dist, dist_matrix[:, idx])

            # 评分：最小距离的最小值（maxmin准则）
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
        objective_values=[0.0] * len(selected),  # 不记录目标值
        marginal_gains=[0.0] * len(selected),
        total_cost=total_cost,
        method_name="Maxmin"
    )


# =====================================================================
# 5. Uniform Selection（向后兼容）
# =====================================================================

def uniform_selection(sensors: List[Sensor], k: int, Q_pr: sp.spmatrix = None,
                     mu_pr: np.ndarray = None, rng: np.random.Generator = None) -> SelectionResult:
    """
    均匀随机选择（不考虑信息量）

    向后兼容函数：供 main.py 直接导入使用

    Args:
        sensors: 候选传感器列表
        k: 预算
        Q_pr: 先验精度矩阵（未使用）
        mu_pr: 先验均值（未使用）
        rng: 随机数生成器

    Returns:
        SelectionResult
    """
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
    """
    随机选择（逆成本加权）

    向后兼容函数：供 main.py 直接导入使用

    Args:
        sensors: 候选传感器列表
        k: 预算
        Q_pr: 先验精度矩阵（未使用）
        mu_pr: 先验均值（未使用）
        rng: 随机数生成器

    Returns:
        SelectionResult
    """
    if rng is None:
        rng = np.random.default_rng()

    n_sensors = len(sensors)
    costs = np.array([s.cost for s in sensors], dtype=float)

    # 逆成本加权（便宜的传感器更可能被选中）
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