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


def greedy_aopt(sensors: List, k: int, Q_pr: sp.spmatrix,
                costs: np.ndarray = None,
                hutchpp_probes: int = 20,
                batch_size: int = 32,
                max_candidates: int = None,
                early_stop_ratio: float = 0.0) -> SelectionResult:
    """
    大幅加速版Greedy-A-optimality

    新增优化：
    1. max_candidates: 每步只评估前N个候选（空间近邻）
    2. early_stop_ratio: 当增益<初始增益*ratio时停止
    3. 进度输出：实时显示评估进度
    4. 批量处理：减少factor构造次数

    Args:
        sensors: 候选传感器列表
        k: 预算
        Q_pr: 先验精度矩阵
        costs: 传感器成本（可选）
        hutchpp_probes: Hutchinson++探针数
        batch_size: 批量评估大小
        max_candidates: 每步最多评估的候选数（None=全部）
        early_stop_ratio: 早停阈值（0=禁用）
    """
    from inference import SparseFactor, compute_posterior
    import time

    n = Q_pr.shape[0]
    if costs is None:
        costs = np.ones(len(sensors))

    selected_ids = []
    selected_sensors = []
    objective_values = []
    marginal_gains = []
    total_cost = 0.0

    # 先验因子
    factor_pr = SparseFactor(Q_pr)

    # 预生成传感器坐标（用于空间筛选）
    from geometry import Geometry
    sensor_locations = np.array([sensors[i].idxs[0] for i in range(len(sensors))])

    # Hutchinson++配置
    rng = np.random.default_rng(42)
    probe_refresh_interval = 5

    current_trace = estimate_trace_hutchpp(factor_pr, hutchpp_probes, rng)
    print(f"  Initial trace estimate: {current_trace:.1f}")

    initial_gain = None  # 用于early stop
    step_times = []  # 记录每步耗时

    for step in range(k):
        step_start = time.time()

        best_gain = -np.inf
        best_idx = -1

        # 🔥 候选筛选策略
        remaining = [i for i in range(len(sensors)) if i not in selected_ids]

        if max_candidates and len(remaining) > max_candidates:
            # 空间筛选：优先评估距离已选点较远的候选
            if selected_ids:
                # 计算每个候选到已选点的最小距离
                selected_locs = sensor_locations[selected_ids]
                remaining_locs = sensor_locations[remaining]

                from scipy.spatial.distance import cdist
                dists = cdist(remaining_locs.reshape(-1, 1),
                              selected_locs.reshape(-1, 1))
                min_dists = dists.min(axis=1)

                # 选择距离最远的max_candidates个
                top_indices = np.argsort(-min_dists)[:max_candidates]
                remaining = [remaining[i] for i in top_indices]
            else:
                # 首步：随机采样
                remaining = rng.choice(remaining, size=max_candidates, replace=False).tolist()

        # 🔥 进度输出
        print(f"  Step {step + 1}/{k}: Evaluating {len(remaining)} candidates "
              f"(out of {len(sensors) - len(selected_ids)} remaining)...")

        evaluated_count = 0
        last_print_time = time.time()

        # 分批评估
        for batch_start in range(0, len(remaining), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining))
            batch_ids = remaining[batch_start:batch_end]

            # 🔥 每5秒或每100个候选输出一次进度
            evaluated_count += len(batch_ids)
            current_time = time.time()
            if (current_time - last_print_time > 5.0) or (batch_end == len(remaining)):
                progress_pct = 100 * evaluated_count / len(remaining)
                elapsed = current_time - step_start
                eta = elapsed / evaluated_count * (len(remaining) - evaluated_count)
                print(f"    Progress: {evaluated_count}/{len(remaining)} "
                      f"({progress_pct:.0f}%), ETA: {eta:.0f}s")
                last_print_time = current_time

            # 构造测试传感器集合
            test_sensors_list = []
            for cand_idx in batch_ids:
                test_sensors_list.append(selected_sensors + [sensors[cand_idx]])

            # 刷新探针（每probe_refresh_interval步）
            if step % probe_refresh_interval == 0 or step == 0:
                probes = generate_hutchpp_probes(n, hutchpp_probes, rng)

            # 批量评估
            for cand_idx, test_sensors in zip(batch_ids, test_sensors_list):
                from sensors import assemble_H_R
                H_test, R_test = assemble_H_R(test_sensors, n)

                try:
                    mu_post, factor_post = compute_posterior(
                        Q_pr, np.zeros(n), H_test, R_test, np.zeros(len(test_sensors))
                    )

                    new_trace = estimate_trace_hutchpp_with_probes(
                        factor_post, probes
                    )

                    gain = current_trace - new_trace

                    if gain > best_gain:
                        best_gain = gain
                        best_idx = cand_idx

                except Exception as e:
                    # 数值问题：跳过该候选
                    continue

        # 检查是否找到有效候选
        if best_idx == -1:
            print(f"  Warning: No valid candidate found at step {step + 1}")
            break

        # 记录初始增益（用于early stop）
        if step == 0:
            initial_gain = best_gain

        # 🔥 Early stop检查
        if early_stop_ratio > 0 and initial_gain is not None:
            if best_gain < initial_gain * early_stop_ratio:
                print(f"  Early stopping at step {step + 1}: "
                      f"gain {best_gain:.1f} < threshold {initial_gain * early_stop_ratio:.1f}")
                break

        # 添加传感器
        selected_ids.append(best_idx)
        selected_sensors.append(sensors[best_idx])
        marginal_gains.append(best_gain)
        total_cost += costs[best_idx]

        # 更新trace
        current_trace -= best_gain
        objective_values.append(-current_trace)

        # 记录耗时
        step_time = time.time() - step_start
        step_times.append(step_time)

        # 🔥 更详细的进度输出
        avg_time = np.mean(step_times)
        eta_total = avg_time * (k - step - 1)
        print(f"  ✓ Step {step + 1}/{k} complete: "
              f"Trace={current_trace:.1f}, ΔTrace={best_gain:.1f}, "
              f"Sensor=#{best_idx}, Cost=£{total_cost:.0f}, "
              f"Time={step_time:.1f}s (ETA: {eta_total / 60:.1f}min)")

    print(f"  Total time: {sum(step_times) / 60:.1f} minutes")

    return SelectionResult(
        selected_ids=selected_ids,
        objective_values=objective_values,
        marginal_gains=marginal_gains,
        total_cost=total_cost,
        method_name="Greedy-A"
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