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
              lazy: bool = True, batch_size: int = 1,
              use_cost: bool = True,
              keep_fraction: float = None) -> 'SelectionResult':
    """
    Greedy mutual information maximization (批量优化版本)
    """
    import numpy as np
    import scipy.sparse as sp
    from inference import SparseFactor

    n = Q_pr.shape[0]
    C = len(sensors)

    # 🔥 动态计算keep_fraction(专家建议)
    if keep_fraction is None:
        n_keep_budget = 4 * k
        n_keep_pool = int(np.ceil(0.25 * C))
        n_keep = max(n_keep_budget, n_keep_pool, k + 10)
        actual_keep_fraction = n_keep / C
    else:
        actual_keep_fraction = keep_fraction
        n_keep = int(C * actual_keep_fraction)

    if costs is None:
        costs = np.ones(C, dtype=float)
    else:
        costs = np.asarray(costs, dtype=float)
        if len(costs) != C:
            raise ValueError(f"Cost array length {len(costs)} doesn't match sensor count {C}")

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

    if lazy and C > 100:
        Z = factor.solve_multi(H_rows.T)
        quad = np.sum(H_rows * Z.T, axis=1)
        mi_values = 0.5 * np.log1p(quad / R_list)
    else:
        mi_values = None

    if mi_values is not None and C > 100:
        top_indices = np.argpartition(mi_values, -n_keep)[-n_keep:]
        alive = np.zeros(C, dtype=bool)
        alive[top_indices] = True
        print(f"  MI prescreen: kept {n_keep}/{C} candidates ({100*actual_keep_fraction:.0f}%)")
    else:
        alive = np.ones(C, dtype=bool)

    for step in range(k):
        best_idx = -1
        best_gain = -np.inf
        best_mi = 0.0

        candidates = np.where(alive)[0]

        for idx in candidates:
            h = H_rows[idx]
            r = R_list[idx]

            z = factor.solve(h)
            quad = np.dot(h, z)
            mi = 0.5 * np.log1p(quad / r)

            if use_cost:
                gain = mi / costs[idx]
            else:
                gain = mi

            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_mi = mi

        if best_idx < 0 or best_gain <= 0:
            break

        selected.append(int(best_idx))
        marginal_gains.append(float(best_mi))
        total_cost += float(costs[best_idx])
        objective_values.append(
            objective_values[-1] + best_mi if objective_values else best_mi
        )

        h_star = H_rows[best_idx]
        r_star = R_list[best_idx]
        factor.rank1_update(h_star, weight=1.0 / r_star)

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
    """
    import numpy as np
    from inference import SparseFactor

    n = Q_pr.shape[0]
    C = len(sensors)

    if costs is None:
        costs = np.ones(C, dtype=float)
    else:
        costs = np.asarray(costs, dtype=float)
        if len(costs) != C:
            raise ValueError(f"Cost array length {len(costs)} doesn't match sensor count {C}")

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

    rng = np.random.default_rng(42)
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
        keep_fraction: float = None,
        rng: np.random.Generator = None,
        verbose: bool = False
) -> 'SelectionResult':
    """
    🔥 修复版 Myopic EVI - 解决阈值不一致和负ROI问题

    关键修复：
    1. 锁定阈值tau_fixed，确保先验/后验风险计算一致
    2. 防守式校验，避免负EVI
    3. 记录tau_iri到decision_config以便追溯
    """
    import numpy as np
    import scipy.sparse as sp
    from inference import SparseFactor
    from decision import expected_loss

    if rng is None:
        rng = np.random.default_rng()

    n = Q_pr.shape[0]
    C = len(sensors)

    # 🔥 动态计算keep_fraction
    if keep_fraction is None:
        n_keep_budget = 4 * k
        n_keep_pool = int(np.ceil(0.25 * C))
        n_keep = max(n_keep_budget, n_keep_pool, k + 10)
        actual_keep_fraction = n_keep / C
    else:
        actual_keep_fraction = keep_fraction
        n_keep = int(C * actual_keep_fraction)

    if costs is None:
        costs = np.array([s.cost for s in sensors], dtype=float)
    else:
        costs = np.asarray(costs, dtype=float)
        if len(costs) != C:
            raise ValueError(f"Cost length {len(costs)} != sensor count {C}")

    if verbose:
        print(f"\n  🚀 Fixed Greedy-EVI: n={n}, candidates={C}, budget={k}, test={len(test_idx)}")

    # ---------------------------
    # 🔥 关键修复1：锁定阈值
    # ---------------------------
    F = SparseFactor(Q_pr)

    # 在计算prior_risk之前固定tau
    if decision_config.tau_iri is not None:
        tau_fixed = float(decision_config.tau_iri)
        if verbose:
            print(f"    Using pre-cached tau: {tau_fixed:.3f}")
    else:
        tau_fixed = float(np.quantile(mu_pr, decision_config.tau_quantile))
        decision_config.tau_iri = tau_fixed  # 🔥 记录，保证可追溯
        if verbose:
            print(f"    Computed and cached tau: {tau_fixed:.3f}")

    # 用单位基的子集一次多RHS解出 diag(Σ)_test
    I_sub = np.zeros((n, len(test_idx)), dtype=float)
    I_sub[test_idx, np.arange(len(test_idx))] = 1.0
    Z_sub = F.solve_multi(I_sub)
    diag_test = np.einsum('ij,ij->j', Z_sub[test_idx, :], I_sub[test_idx, :])
    diag_test = np.maximum(diag_test, 1e-12)
    sigma_test = np.sqrt(diag_test)
    mu_test = mu_pr[test_idx].copy()

    # 🔥 关键修复2：使用固定tau计算先验风险
    prior_risk = expected_loss(
        mu_test,
        sigma_test,
        decision_config,
        test_indices=np.arange(len(test_idx)),
        tau=tau_fixed  # 显式传入固定阈值
    )

    if verbose:
        print(f"    Prior risk (fixed tau): £{prior_risk:.2f}")

    # 预取每个候选的稀疏 h（行向量）
    idx_list = [s.idxs for s in sensors]
    w_list = [s.weights for s in sensors]
    r_list = np.array([s.noise_var for s in sensors], dtype=float)

    H_dense = np.stack(
        [np.bincount(idxs, weights=w, minlength=n) for idxs, w in zip(idx_list, w_list)],
        axis=0
    )

    # MI 预筛
    keep_mask = np.ones(C, dtype=bool)
    original_indices = np.arange(C)

    if mi_prescreen and C > 50:
        if verbose:
            print(f"    🔍 MI prescreening over {C} candidates ...")

        Z = F.solve_multi(H_dense.T)
        quad = np.sum(H_dense * Z.T, axis=1)
        mi = 0.5 * np.log1p(quad / r_list)

        keep_idx = np.argpartition(mi, -n_keep)[-n_keep:]
        keep_mask[:] = False
        keep_mask[keep_idx] = True

        H_dense = H_dense[keep_mask, :]
        r_list = r_list[keep_mask]
        costs = costs[keep_mask]
        idx_list = [idx_list[i] for i in range(C) if keep_mask[i]]
        w_list = [w_list[i] for i in range(C) if keep_mask[i]]
        original_indices = original_indices[keep_mask]
        Z = Z[:, keep_mask]
        C = H_dense.shape[0]

        if verbose:
            print(f"    ✓ kept {C} ({100 * C / len(keep_mask):.0f}%)")
    else:
        Z = F.solve_multi(H_dense.T)

    # Greedy 循环
    selected = []
    mg = []
    obj = []
    tot_cost = 0.0
    alive = np.ones(C, dtype=bool)

    for step in range(k):
        if verbose:
            print(f"    Step {step + 1}/{k}")

        zt = Z[test_idx, :]
        num = np.sum(zt * zt, axis=0)
        denom = r_list + np.sum(H_dense * Z.T, axis=1)
        denom = np.maximum(denom, 1e-12)

        diag_post_all = diag_test[:, None] - (zt * zt) / denom[None, :]
        diag_post_all = np.maximum(diag_post_all, 1e-12)

        sigma_post_all = np.sqrt(diag_post_all)

        # 🔥 关键修复3：所有后验风险计算都使用相同的固定tau
        post_risk = np.empty(C)
        for j in range(C):
            if not alive[j]:
                post_risk[j] = np.inf
                continue
            post_risk[j] = expected_loss(
                mu_test,
                sigma_post_all[:, j],
                decision_config,
                test_indices=np.arange(len(test_idx)),
                tau=tau_fixed  # 🔥 显式传入相同的固定阈值
            )

        # 🔥 关键修复4：防守式校验EVI增益
        evi_gain = prior_risk - post_risk

        # 检查并修复负增益（数值误差容忍）
        if np.any(evi_gain < -1e-9):
            if verbose:
                print(f"      ⚠️  Detected negative EVI gains, applying defensive clipping")
            evi_gain = np.maximum(evi_gain, 0.0)

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

        # Rank-1 更新
        z_star = Z[:, best]
        h_star = H_dense[best, :]
        den = denom[best]

        diag_test = diag_test - (z_star[test_idx] ** 2) / den
        diag_test = np.maximum(diag_test, 1e-12)
        sigma_test = np.sqrt(diag_test)

        # 🔥 关键修复5：先验风险更新也使用固定tau
        prior_risk = expected_loss(
            mu_test,
            sigma_test,
            decision_config,
            test_indices=np.arange(len(test_idx)),
            tau=tau_fixed  # 🔥 保持一致
        )

        c = h_star @ Z
        Z -= np.outer(z_star, c) / den
        alive[best] = False

    return SelectionResult(
        selected_ids=selected,
        objective_values=obj,
        marginal_gains=mg,
        total_cost=tot_cost,
        method_name="Greedy-EVI-fixed"
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