"""
Sensor selection algorithms with submodular optimization.
Implements Greedy-MI, Greedy-A, and baseline methods.
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple
import heapq
from dataclasses import dataclass


@dataclass
class SelectionResult:
    """Result container for sensor selection."""
    selected_ids: List[int]
    objective_values: List[float]  # Value after each addition
    marginal_gains: List[float]  # Marginal gain at each step
    total_cost: float
    method_name: str


def greedy_mi(sensors: List, k: int, Q_pr: sp.spmatrix,
              costs: np.ndarray = None,
              lazy: bool = True) -> SelectionResult:
    """
    Greedy sensor selection maximizing Mutual Information.

    修复：正确实现成本约束下的选择逻辑
    - score = gain/cost 用于 argmax（选哪个）
    - gain 用于累计 objective_values（总MI曲线）
    """
    from inference import SparseFactor, quadform_via_solve
    from sensors import assemble_H_R

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

    # 优先队列：(-score, iteration, sensor_id, gain_cache)
    # score = gain/cost（有成本）或 gain（无成本）
    pq = [] if lazy else None
    last_eval = {} if lazy else None

    # 预计算传感器的h行和噪声方差
    sensor_h_rows = []
    sensor_noise_vars = []
    for s in sensors:
        h = np.zeros(n)
        h[s.idxs] = s.weights
        sensor_h_rows.append(h)
        sensor_noise_vars.append(s.noise_var)

    # Greedy循环
    for step in range(k):
        best_score = -np.inf  # 用于选择（argmax score）
        best_gain = 0.0  # 用于累计MI曲线
        best_idx = -1

        # Lazy evaluation路径
        if lazy and step > 0:
            while pq:
                neg_score, eval_iter, cand_idx, cached_gain = heapq.heappop(pq)

                if cand_idx in selected_ids:
                    continue

                # 检查是否新鲜
                if eval_iter < step:
                    # 过期，重新评估
                    h = sensor_h_rows[cand_idx]
                    r = sensor_noise_vars[cand_idx]
                    q = quadform_via_solve(current_factor, h)
                    gain = 0.5 * np.log1p(q / r)
                    score = gain / costs[cand_idx]  # 成本归一化

                    # 重新入队，标记为新鲜
                    heapq.heappush(pq, (-score, step, cand_idx, gain))
                    last_eval[cand_idx] = step
                else:
                    # 新鲜的评估，直接使用
                    best_idx = cand_idx
                    best_score = -neg_score
                    best_gain = cached_gain
                    break

        # 如果没找到或首次迭代，全量评估
        if best_idx == -1:
            for cand_idx in range(n_candidates):
                if cand_idx in selected_ids:
                    continue

                h = sensor_h_rows[cand_idx]
                r = sensor_noise_vars[cand_idx]

                # 计算边际MI增益
                q = quadform_via_solve(current_factor, h)
                gain = 0.5 * np.log1p(q / r)

                # 计算score（用于选择）
                score = gain / costs[cand_idx]

                if lazy:
                    # 入队
                    heapq.heappush(pq, (-score, step, cand_idx, gain))
                    last_eval[cand_idx] = step

                # 更新最佳候选
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
            print(f"  Step {step + 1}/{k}: "
                  f"MI={current_obj:.3f} nats ({current_obj / np.log(2):.2f} bits), "
                  f"ΔMI={best_gain:.4f}, "
                  f"score={best_score:.6f}, "
                  f"cost=£{total_cost:.0f}, "
                  f"type={sensors[best_idx].type_name}")

    return SelectionResult(
        selected_ids=selected_ids,
        objective_values=objective_values,
        marginal_gains=marginal_gains,
        total_cost=total_cost,
        method_name="Greedy-MI"
    )


def greedy_aopt(sensors: List, k: int, Q_pr: sp.spmatrix,
                costs: np.ndarray = None) -> SelectionResult:
    """
    Greedy sensor selection minimizing A-optimality (trace of Σ_post).

    Note: This is a heuristic without submodularity guarantees.

    Args:
        sensors: List of Sensor objects
        k: Budget
        Q_pr: Prior precision
        costs: Optional costs

    Returns:
        result: SelectionResult
    """
    from inference import SparseFactor, compute_posterior_variance_diagonal
    from sensors import assemble_H_R

    n = Q_pr.shape[0]
    if costs is None:
        costs = np.ones(len(sensors))

    selected_ids = []
    selected_sensors = []
    objective_values = []
    marginal_gains = []
    total_cost = 0.0

    # Sample points for trace estimation
    trace_sample_idx = np.linspace(0, n - 1, min(100, n), dtype=int)

    # Initial trace (prior)
    factor_pr = SparseFactor(Q_pr)
    var_pr = compute_posterior_variance_diagonal(factor_pr, trace_sample_idx)
    current_trace = var_pr.sum() * (n / len(trace_sample_idx))  # Scale to full trace

    for step in range(k):
        best_gain = -np.inf
        best_idx = -1

        # Evaluate all candidates
        for cand_idx, sensor in enumerate(sensors):
            if cand_idx in selected_ids:
                continue

            # Build H with candidate added
            test_sensors = selected_sensors + [sensor]
            H_test, R_test = assemble_H_R(test_sensors, n)

            # Compute posterior variance on sample
            from inference import compute_posterior
            mu_post, factor_post = compute_posterior(
                Q_pr, np.zeros(n), H_test, R_test, np.zeros(len(test_sensors))
            )
            var_post = compute_posterior_variance_diagonal(factor_post, trace_sample_idx)
            new_trace = var_post.sum() * (n / len(trace_sample_idx))

            # Gain is reduction in trace
            gain = current_trace - new_trace
            ratio = gain / costs[cand_idx]

            if gain > best_gain:
                best_gain = gain
                best_idx = cand_idx

        if best_idx == -1:
            break

        # Add sensor
        selected_ids.append(best_idx)
        selected_sensors.append(sensors[best_idx])
        marginal_gains.append(best_gain)
        total_cost += costs[best_idx]

        # Update trace
        current_trace -= best_gain
        objective_values.append(-current_trace)  # Negative for minimization

        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{k}: Trace={current_trace:.1f}, Δ={best_gain:.1f}")

    return SelectionResult(
        selected_ids=selected_ids,
        objective_values=objective_values,
        marginal_gains=marginal_gains,
        total_cost=total_cost,
        method_name="Greedy-A"
    )


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
    from sensors import generate_sensor_pool

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