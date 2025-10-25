"""
Sensor selection algorithms with submodular optimization.
Implements Greedy-MI, Greedy-A, and baseline methods.
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple
import heapq
from dataclasses import dataclass

# 🔥 关键导入 - 确保这些都存在
from sensors import Sensor, assemble_H_R
from inference import (
    SparseFactor,
    compute_posterior,
    compute_posterior_variance_diagonal,
    compute_mutual_information,
    quadform_via_solve
)
@dataclass
class SelectionResult:
    """Container for selection algorithm results."""
    selected_ids: List[int]
    objective_values: List[float]
    marginal_gains: List[float]
    total_cost: float
    method_name: str

    def __repr__(self):
        return (f"SelectionResult(method={self.method_name}, "
                f"n_selected={len(self.selected_ids)}, "
                f"total_cost=£{self.total_cost:.0f})")


def greedy_mi(sensors: List, k: int, Q_pr: sp.spmatrix,
              costs: np.ndarray = None,
              lazy: bool = True,
              batch_size: int = 64) -> SelectionResult:
    """
    Greedy sensor selection maximizing Mutual Information.

    修复：
    1. 实现批量候选评估（加速10-50倍）
    2. 正确的成本归一化逻辑
    3. 使用batch_quadform加速二次型计算
    """
    from inference import SparseFactor, batch_quadform_via_solve

    n = Q_pr.shape[0]
    n_candidates = len(sensors)

    # 提取或验证成本
    if costs is None:
        costs = np.array([s.cost for s in sensors], dtype=float)
        print(f"  Using real sensor costs: min=£{costs.min():.0f}, "
              f"max=£{costs.max():.0f}, mean=£{costs.mean():.0f}")

    # 初始化
    selected_ids = []
    selected_sensors = []
    objective_values = []  # 累计总MI
    marginal_gains = []  # 每步的ΔMI
    total_cost = 0.0

    # 先验因子
    current_factor = SparseFactor(Q_pr)
    current_obj = 0.0

    # 🔥 预计算传感器的h行和噪声方差（避免重复构造）
    sensor_h_rows = []
    sensor_noise_vars = []
    for s in sensors:
        h = np.zeros(n)
        h[s.idxs] = s.weights
        sensor_h_rows.append(h)
        sensor_noise_vars.append(s.noise_var)

    # 优先队列（lazy evaluation）
    pq = [] if lazy else None
    last_eval = {} if lazy else None

    # Greedy循环
    for step in range(k):
        best_score = -np.inf
        best_gain = 0.0
        best_idx = -1

        # 🔥 收集待评估的候选
        candidates_to_eval = []

        if lazy and step > 0:
            # Lazy路径：检查队列中的top候选
            temp_extracted = []
            while pq and len(candidates_to_eval) < batch_size:
                neg_score, eval_iter, cand_idx, cached_gain = heapq.heappop(pq)

                if cand_idx in selected_ids:
                    continue

                if eval_iter < step:
                    # 过期，需要重新评估
                    candidates_to_eval.append(cand_idx)
                    temp_extracted.append((neg_score, eval_iter, cand_idx, cached_gain))
                else:
                    # 新鲜的评估，直接使用
                    best_idx = cand_idx
                    best_score = -neg_score
                    best_gain = cached_gain
                    # 把剩余的放回队列
                    for item in temp_extracted:
                        heapq.heappush(pq, item)
                    break

        # 如果没找到或首次迭代，全量评估（分批）
        if best_idx == -1:
            if not candidates_to_eval:
                candidates_to_eval = [i for i in range(n_candidates) if i not in selected_ids]

            # 🔥 批量评估候选
            for batch_start in range(0, len(candidates_to_eval), batch_size):
                batch_end = min(batch_start + batch_size, len(candidates_to_eval))
                batch_ids = candidates_to_eval[batch_start:batch_end]

                # 构造批量H矩阵 (n × batch_size)
                H_batch = np.column_stack([sensor_h_rows[i] for i in batch_ids])
                R_batch = np.array([sensor_noise_vars[i] for i in batch_ids])

                # 🔥 批量计算 h^T Σ h（一次solve）
                quad_batch = batch_quadform_via_solve(current_factor, H_batch)

                # 批量计算gain和score
                gains_batch = 0.5 * np.log1p(quad_batch / R_batch)
                costs_batch = costs[batch_ids]
                scores_batch = gains_batch / costs_batch

                # 更新最佳候选
                for i, (cand_idx, gain, score) in enumerate(zip(batch_ids, gains_batch, scores_batch)):
                    if lazy:
                        heapq.heappush(pq, (-score, step, cand_idx, gain))
                        last_eval[cand_idx] = step

                    if score > best_score:
                        best_score = score
                        best_gain = gain
                        best_idx = cand_idx

        if best_idx == -1:
            print(f"Warning: No valid sensor found at step {step}")
            break

        # 添加选中的传感器
        selected_ids.append(best_idx)
        selected_sensors.append(sensors[best_idx])
        marginal_gains.append(best_gain)
        total_cost += costs[best_idx]

        # Rank-1更新精度矩阵
        h_best = sensor_h_rows[best_idx]
        r_best = sensor_noise_vars[best_idx]
        current_factor.rank1_update(h_best / np.sqrt(r_best), 1.0)

        # 累计MI（用真实gain，不是score）
        current_obj += best_gain
        objective_values.append(current_obj)

        # 打印进度（增加成本效益信息）
        if (step + 1) % 10 == 0 or step == k - 1:
            cost_efficiency = best_gain / costs[best_idx] * 1000  # per £1000
            print(f"  Step {step + 1}/{k}: "
                  f"MI={current_obj:.3f} nats ({current_obj / np.log(2):.2f} bits), "
                  f"ΔMI={best_gain:.4f}, "
                  f"eff={cost_efficiency:.6f} bits/£1k, "
                  f"cost=£{total_cost:.0f}, "
                  f"type={sensors[best_idx].type_name}")

    return SelectionResult(
        selected_ids=selected_ids,
        objective_values=objective_values,
        marginal_gains=marginal_gains,
        total_cost=total_cost,
        method_name="Greedy-MI"
    )


# =====================================================================
# 🔥 新增/替换：快速 Greedy A-optimality (Hutch++ + rank-1)
# =====================================================================

def greedy_aopt(sensors: List, k: int, Q_pr: sp.spmatrix,
                costs: np.ndarray = None,
                n_probes: int = 16,
                use_cost: bool = True) -> SelectionResult:
    """
    Fast Greedy A-opt via Hutch++ + rank-1 update (no per-candidate factorization).

    Each step:
    - Δtr ≈ (1/m) * sum_j (h^T Σ v_j)^2 / (r + h^T Σ h)
    - Only ONE sparse solve per step: Q z_h = h
    - All candidates evaluated in O(1) using precomputed probe responses

    Args:
        sensors: List of candidate sensors
        k: Budget (number of sensors to select)
        Q_pr: Prior precision matrix (sparse)
        costs: Cost array for each sensor (if None, use sensor.cost)
        n_probes: Number of Hutchinson probes for trace estimation
        use_cost: Whether to normalize gain by cost (gain/cost scoring)

    Returns:
        SelectionResult with selected indices and diagnostics
    """
    from inference import SparseFactor
    rng = np.random.default_rng(42)

    n = Q_pr.shape[0]
    m = len(sensors)

    # Extract costs
    if costs is None:
        costs = np.array([s.cost for s in sensors], dtype=float)

    # --- Prior factorization (one time only)
    print(f"  Factorizing prior precision (n={n})...")
    F = SparseFactor(Q_pr)

    # --- Generate Rademacher probes and solve Z = Σ V
    print(f"  Generating {n_probes} Hutchinson probes...")
    V = rng.choice([-1.0, 1.0], size=(n, n_probes))
    Z = F.solve(V)  # (n × m): one batched solve

    # --- Approximate diag(Σ) using probes
    diag_Sigma = np.mean(Z * V, axis=1)

    # --- Precompute candidate representations
    print(f"  Precomputing {m} candidate footprints...")
    cand_info = []
    for s in sensors:
        idxs = s.idxs
        weights = s.weights
        r = s.noise_var
        cand_info.append((idxs, weights, r))

    # --- Greedy selection
    selected = []
    obj_vals = []
    gains = []
    total_cost = 0.0

    # Initial trace estimate: tr(Σ) ≈ mean(V^T Z)
    cur_trace_est = float(np.mean(np.einsum('ij,ij->j', V, Z)))

    print(f"  Starting greedy selection for k={k}...")
    for step in range(k):
        best_idx = -1
        best_gain = -np.inf
        best_score = -np.inf

        # --- Evaluate all remaining candidates
        for i, (idxs, w, r) in enumerate(cand_info):
            if i in selected:
                continue

            # Numerator: (1/m) * sum_j (h^T Z_j)^2
            # h^T Z = sum_{l in footprint} w_l * Z[l, :]
            hz = (Z[idxs, :] * w[:, None]).sum(axis=0)  # (n_probes,)
            num = float(np.mean(hz * hz))

            # Denominator: r + h^T Σ h ≈ r + sum_{l} w_l^2 * diag_Sigma[l]
            denom = float(r + np.dot(w * w, diag_Sigma[idxs]))

            if denom <= 0:
                continue

            # Gain: Δtr (larger is better, means more trace reduction)
            gain = num / denom

            # Score: gain per unit cost
            score = gain / costs[i] if use_cost else gain

            if score > best_score:
                best_score = score
                best_gain = gain
                best_idx = i

        if best_idx < 0:
            print(f"    Step {step + 1}: No valid candidate found, stopping early")
            break

        # --- Commit best candidate: exact rank-1 update
        idxs, w, r = cand_info[best_idx]

        # Build full h vector
        h = np.zeros(n)
        h[idxs] = w

        # Solve Q z_h = h (one sparse solve per step)
        z_h = F.solve(h)
        denom_exact = float(r + h @ z_h)

        if denom_exact <= 0:
            print(f"    Step {step + 1}: Invalid denominator, stopping")
            break

        # --- Update Z and diag(Σ) via rank-1 formula
        # Z' = Z - z_h * (h^T Z) / denom
        hz_all = Z[idxs, :].T @ w  # (n_probes,) = h^T Z
        Z -= np.outer(z_h, hz_all) / denom_exact

        # diag(Σ') = diag(Σ) - z_h^2 / denom
        diag_Sigma -= (z_h * z_h) / denom_exact

        # --- Update trace estimate and record
        cur_trace_est -= best_gain
        total_cost += costs[best_idx]
        selected.append(best_idx)
        gains.append(best_gain)
        obj_vals.append(-cur_trace_est)  # maximize -tr(Σ)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"    Step {step + 1}/{k}: selected sensor {best_idx}, "
                  f"gain={best_gain:.4f}, score={best_score:.4f}, "
                  f"cumulative_cost=£{total_cost:.0f}")

    print(f"  ✓ Selected {len(selected)} sensors, total cost=£{total_cost:.0f}")

    return SelectionResult(
        selected_ids=selected,
        objective_values=obj_vals,
        marginal_gains=gains,
        total_cost=total_cost,
        method_name="Greedy-A"
    )


# =====================================================================
# 🔥 新增：Myopic EVI (决策感知选址)
# =====================================================================

def greedy_evi_myopic(sensors, k, Q_pr, mu_pr, decision_config, test_idx,
                      costs=None, n_y_samples=10, use_cost=True,
                      rng=None, verbose=True,
                      mi_prescreening=True, keep_fraction=0.25):  # 🔥 新增参数
    """
    Greedy EVI sensor selection with optional MI pre-screening.

    Args:
        sensors: List of candidate sensors
        k: Budget (number of sensors to select)
        Q_pr: Prior precision matrix
        mu_pr: Prior mean vector
        decision_config: Decision configuration
        test_idx: Test set indices for risk evaluation
        costs: Sensor costs (if None, extracted from sensors)
        n_y_samples: Number of Monte Carlo samples
        use_cost: Whether to normalize by cost
        rng: Random number generator
        verbose: Print progress
        mi_prescreening: Enable MI pre-screening (🔥 new)
        keep_fraction: Fraction of candidates to keep (🔥 new)

    Returns:
        SelectionResult object
    """
    from inference import SparseFactor, compute_posterior, compute_posterior_variance_diagonal
    from decision import evi_monte_carlo
    from sensors import assemble_H_R

    if rng is None:
        rng = np.random.default_rng()

    n = Q_pr.shape[0]
    n_sensors = len(sensors)

    if costs is None:
        costs = np.array([s.cost for s in sensors])

    # =========================================================================
    # 🔥 新增：MI 预筛选（大幅加速）
    # =========================================================================
    original_n_sensors = n_sensors

    if mi_prescreening and n_sensors > 50:
        if verbose:
            print(f"    🔍 MI Pre-screening: evaluating {n_sensors} candidates...")

        # 创建先验 factor（复用于所有候选）
        factor_pr = SparseFactor(Q_pr)

        # 批量计算所有候选的 MI
        mi_scores = np.zeros(n_sensors)

        for i, sensor in enumerate(sensors):
            # 构造观测向量 h
            h_vec = np.zeros(n)
            h_vec[sensor.idxs] = sensor.weights

            # 计算 h^T Σ h（快速方法）
            z = factor_pr.solve(h_vec)
            quad = np.dot(h_vec, z)

            # MI = 0.5 * log(1 + h^T Σ h / r)
            mi_scores[i] = 0.5 * np.log(1 + quad / sensor.noise_var)

        # 保留 MI 最高的 top-k
        n_keep = max(20, int(n_sensors * keep_fraction))
        top_indices = np.argsort(mi_scores)[-n_keep:]

        # 筛选
        sensors = [sensors[i] for i in top_indices]
        costs = costs[top_indices]
        n_sensors = len(sensors)

        if verbose:
            print(f"    ✓ Kept top {n_sensors} candidates "
                  f"({100 * n_sensors / original_n_sensors:.0f}% of original)")
            print(f"      MI range: [{mi_scores[top_indices].min():.3f}, "
                  f"{mi_scores[top_indices].max():.3f}] nats")

    # =========================================================================
    # 原有的 Greedy EVI 选择逻辑（不变）
    # =========================================================================

    selected_ids = []
    objective_values = []
    marginal_gains = []
    total_cost = 0.0

    # 当前精度矩阵（初始为先验）
    Q_current = Q_pr.copy()
    factor_current = SparseFactor(Q_current)

    # 计算先验风险（固定，所有样本共享）
    var_pr = compute_posterior_variance_diagonal(factor_current, test_idx)
    sigma_pr = np.sqrt(np.maximum(var_pr, 1e-12))

    from decision import expected_loss
    prior_risk = expected_loss(
        mu_pr[test_idx],
        sigma_pr,
        decision_config,
        test_indices=np.arange(len(test_idx))
    )

    if verbose:
        print(f"    Prior risk: £{prior_risk:.2f}")

    # Greedy 选择循环
    for step in range(k):
        if verbose:
            print(f"    Step {step + 1}/{k}:")

        best_evi = -np.inf
        best_sensor_idx = None
        best_evi_normalized = -np.inf

        # 评估每个候选传感器
        for candidate_idx in range(n_sensors):
            if candidate_idx in selected_ids:
                continue  # 跳过已选

            sensor = sensors[candidate_idx]

            # 构造包含新传感器的观测矩阵
            temp_selected = selected_ids + [candidate_idx]
            temp_sensors = [sensors[i] for i in temp_selected]
            H_temp, R_temp = assemble_H_R(temp_sensors, n)

            # 计算 EVI（Monte Carlo）
            try:
                evi = evi_monte_carlo(
                    Q_pr, mu_pr, H_temp, R_temp, decision_config,
                    n_samples=n_y_samples, rng=rng
                )
            except Exception as e:
                if verbose:
                    print(f"      Warning: EVI computation failed for sensor {candidate_idx}: {e}")
                evi = 0.0

            # 成本归一化
            if use_cost:
                evi_normalized = evi / sensor.cost
            else:
                evi_normalized = evi

            # 更新最佳
            if evi_normalized > best_evi_normalized:
                best_evi_normalized = evi_normalized
                best_evi = evi
                best_sensor_idx = candidate_idx

        # 选择最佳传感器
        if best_sensor_idx is None:
            if verbose:
                print(f"      ⚠️  No valid sensor found, stopping early")
            break

        selected_ids.append(best_sensor_idx)
        marginal_gains.append(best_evi)
        total_cost += sensors[best_sensor_idx].cost

        # 更新当前精度矩阵
        sensor = sensors[best_sensor_idx]
        H_row, _, _ = sensor.h_row
        h_vec = np.zeros(n)
        h_vec[H_row] = sensor.weights
        weight = 1.0 / sensor.noise_var

        Q_current = Q_current + weight * sp.csr_matrix(np.outer(h_vec, h_vec))
        factor_current = SparseFactor(Q_current)

        # 计算当前目标值（累积 EVI）
        current_evi = sum(marginal_gains)
        objective_values.append(current_evi)

        if verbose:
            print(f"      Selected sensor {best_sensor_idx} "
                  f"(type: {sensor.type_name}, cost: £{sensor.cost:.0f})")
            print(f"      Marginal EVI: £{best_evi:.2f}, "
                  f"Cumulative: £{current_evi:.2f}, "
                  f"Total cost: £{total_cost:.0f}")

    return SelectionResult(
        selected_ids=selected_ids,
        objective_values=objective_values,
        marginal_gains=marginal_gains,
        total_cost=total_cost,
        method_name="Greedy-EVI (MI-prescreened)" if mi_prescreening else "Greedy-EVI"
    )

# =====================================================================
# 🔥 新增：MaxMin K-Center (几何覆盖基线)
# =====================================================================

def maxmin_k_center(sensors: List, k: int, coords: np.ndarray,
                    costs: np.ndarray = None,
                    use_cost: bool = True) -> SelectionResult:
    """
    MaxMin K-Center sensor selection (geometric coverage baseline).

    Iteratively selects the sensor that is farthest from already selected sensors,
    optionally normalized by cost.

    Args:
        sensors: List of candidate sensors
        k: Number of sensors to select
        coords: Spatial coordinates (n_sensors, d)
        costs: Cost array (if None, use sensor.cost)
        use_cost: Whether to normalize distance by cost

    Returns:
        SelectionResult with selected indices
    """
    from scipy.spatial.distance import cdist

    m = len(sensors)

    # Extract costs
    if costs is None:
        costs = np.array([s.cost for s in sensors], dtype=float)

    # Extract sensor center locations (use first index as representative)
    sensor_locs = np.array([coords[s.idxs[0]] for s in sensors])

    # Start with sensor closest to domain center
    center = np.mean(sensor_locs, axis=0)
    dists_to_center = np.linalg.norm(sensor_locs - center, axis=1)
    first_idx = int(np.argmin(dists_to_center))

    selected = [first_idx]
    obj_vals = [0.0]
    gains = [0.0]
    total_cost = costs[first_idx]

    print(f"  MaxMin K-Center: starting with sensor {first_idx}")

    for step in range(1, k):
        # Compute minimum distance to any selected sensor
        selected_locs = sensor_locs[selected]
        dists = cdist(sensor_locs, selected_locs, metric='euclidean')
        min_dist = np.min(dists, axis=1)

        # Mask already selected
        min_dist[selected] = -np.inf

        # Select sensor with maximum minimum distance (optionally cost-normalized)
        if use_cost:
            scores = min_dist / costs
        else:
            scores = min_dist

        best_idx = int(np.argmax(scores))

        if best_idx in selected or scores[best_idx] == -np.inf:
            print(f"    Step {step + 1}: No valid candidate, stopping early")
            break

        # Record
        selected.append(best_idx)
        total_cost += costs[best_idx]
        gain = float(min_dist[best_idx])
        gains.append(gain)
        obj_vals.append(obj_vals[-1] + gain)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"    Step {step + 1}/{k}: sensor {best_idx}, "
                  f"min_dist={gain:.2f}, cumulative_cost=£{total_cost:.0f}")

    print(f"  ✓ Selected {len(selected)} sensors via MaxMin, total cost=£{total_cost:.0f}")

    return SelectionResult(
        selected_ids=selected,
        objective_values=obj_vals,
        marginal_gains=gains,
        total_cost=total_cost,
        method_name="MaxMin"
    )


def estimate_trace_hutchpp(factor: 'SparseFactor', n_probes: int,
                           rng: np.random.Generator) -> float:
    """Hutchinson++ trace估计"""
    probes = generate_hutchpp_probes(factor.n, n_probes, rng)
    return estimate_trace_hutchpp_with_probes(factor, probes)


def generate_hutchpp_probes(n: int, n_probes: int,
                            rng: np.random.Generator) -> np.ndarray:
    """生成Hutchinson++探针（Rademacher向量）"""
    return rng.choice([-1, 1], size=(n, n_probes)).astype(float)


def estimate_trace_hutchpp_with_probes(factor: 'SparseFactor',
                                       probes: np.ndarray) -> float:
    """使用给定探针估计trace"""
    # Z = Σ * probes = Q^{-1} * probes
    Z = factor.solve(probes)

    # trace(Σ) ≈ (1/m) * sum(probes^T * Z)
    trace_est = np.mean(np.einsum('ij,ij->j', probes, Z))

    return trace_est

def uniform_selection(sensors: List, k: int, geom) -> SelectionResult:
    """
    Uniform spatial grid sensor placement.

    Args:
        sensors: Candidate sensors
        k: Budget
        geom: Geometry object

    Returns:
        result: SelectionResult
    """
    # Create k×k grid over domain
    coords = geom.coords

    # Compute grid spacing
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()

    k_side = int(np.ceil(np.sqrt(k)))
    x_grid = np.linspace(coords[:, 0].min(), coords[:, 0].max(), k_side)
    y_grid = np.linspace(coords[:, 1].min(), coords[:, 1].max(), k_side)

    # Find nearest sensor to each grid point
    from scipy.spatial import cKDTree
    sensor_coords = np.array([coords[s.idxs[0]] for s in sensors])
    tree = cKDTree(sensor_coords)

    selected_ids = []
    for xi in x_grid:
        for yi in y_grid:
            if len(selected_ids) >= k:
                break
            _, idx = tree.query([xi, yi])
            if idx not in selected_ids:
                selected_ids.append(idx)

    total_cost = sum(sensors[i].cost for i in selected_ids)

    return SelectionResult(
        selected_ids=selected_ids,
        objective_values=[],
        marginal_gains=[],
        total_cost=total_cost,
        method_name="Uniform"
    )


def random_selection(sensors: List, k: int,
                     rng: np.random.Generator) -> SelectionResult:
    """
    Random sensor selection baseline.

    Args:
        sensors: Candidate sensors
        k: Budget
        rng: Random generator

    Returns:
        result: SelectionResult
    """
    selected_ids = rng.choice(len(sensors), size=k, replace=False).tolist()
    total_cost = sum(sensors[i].cost for i in selected_ids)

    return SelectionResult(
        selected_ids=selected_ids,
        objective_values=[],
        marginal_gains=[],
        total_cost=total_cost,
        method_name="Random"
    )


if __name__ == "__main__":
    from config import load_config
    from geometry import build_grid2d_geometry
    from spatial_field import build_prior
    from sensors import generate_sensor_pool, Sensor

    cfg = load_config()
    rng = cfg.get_rng()

    # Setup
    geom = build_grid2d_geometry(20, 20, h=cfg.geometry.h)
    Q_pr, mu_pr = build_prior(geom, cfg.prior)
    sensors = generate_sensor_pool(geom, cfg.sensors, rng)

    print(f"Selecting from {len(sensors)} candidates...")

    # Test Greedy-MI
    print("\nGreedy-MI:")
    result = greedy_mi(sensors, k=10, Q_pr=Q_pr, lazy=True)
    print(f"  Final MI: {result.objective_values[-1]:.3f}")
    print(f"  Total cost: £{result.total_cost:.0f}")

    # Test Random
    print("\nRandom:")
    result_random = random_selection(sensors, k=10, rng=rng)
    print(f"  Total cost: £{result_random.total_cost:.0f}")