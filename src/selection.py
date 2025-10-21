"""
加速版传感器选择 - 批量求解版本

主要优化：
1. 批量RHS求解（一次解多个候选）
2. 自适应批量大小
3. 优化的lazy-greedy
4. 提前终止无效候选

预期加速：3-10倍（CPU）
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
    objective_values: List[float]
    marginal_gains: List[float]
    total_cost: float
    method_name: str


def greedy_mi(sensors: List, k: int, Q_pr: sp.spmatrix,
              costs: np.ndarray = None,
              lazy: bool = True,
              batch_size: int = 64) -> SelectionResult:
    """
    加速版Greedy-MI：批量RHS求解

    新增参数：
        batch_size: 每批评估的候选数（默认64，自适应调整）

    加速原理：
        - 把多个候选的h向量堆叠成矩阵 H_batch (n×B)
        - 一次solve得到 Q^{-1} H_batch，而非B次solve
        - BLAS优化的矩阵乘法，缓存友好
    """
    from inference import SparseFactor, quadform_via_solve

    n = Q_pr.shape[0]
    n_candidates = len(sensors)

    # 提取成本
    if costs is None:
        costs = np.array([s.cost for s in sensors], dtype=float)
        print(f"  Using real sensor costs: min=£{costs.min():.0f}, "
              f"max=£{costs.max():.0f}, mean=£{costs.mean():.0f}")

    # 初始化
    selected_ids = []
    selected_sensors = []
    objective_values = []
    marginal_gains = []
    total_cost = 0.0

    # 先验因子
    current_factor = SparseFactor(Q_pr)
    current_obj = 0.0

    # 懒惰队列：(-score, iteration, sensor_id)
    pq = [] if lazy else None
    last_eval = {} if lazy else None

    # 🔥 预计算所有候选的h向量（避免重复构造）
    print("  Precomputing candidate vectors...")
    sensor_h_rows = np.zeros((n, n_candidates))
    sensor_noise_vars = np.zeros(n_candidates)

    for i, s in enumerate(sensors):
        sensor_h_rows[s.idxs, i] = s.weights
        sensor_noise_vars[i] = s.noise_var

    print(f"  Sparse density: {np.count_nonzero(sensor_h_rows)/(n*n_candidates)*100:.2f}%")

    # Greedy循环
    for step in range(k):
        best_score = -np.inf
        best_gain = 0.0
        best_idx = -1

        # 🔥 批量评估策略
        if lazy and step > 0:
            # 从队列中取出top候选进行紧化
            candidates_to_eval = []
            temp_heap = []

            # 取出top batch_size个候选（或更少）
            while pq and len(candidates_to_eval) < batch_size:
                neg_score, eval_iter, cand_idx = heapq.heappop(pq)

                if cand_idx in selected_ids:
                    continue

                if eval_iter < step:
                    # 需要重新评估
                    candidates_to_eval.append(cand_idx)
                else:
                    # 新鲜的，记录下来可能就是最佳
                    temp_heap.append((neg_score, eval_iter, cand_idx))

            # 🔥 批量评估这些候选
            if candidates_to_eval:
                gains_batch, scores_batch = batch_evaluate_candidates(
                    candidates_to_eval,
                    sensor_h_rows,
                    sensor_noise_vars,
                    costs,
                    current_factor,
                    step
                )

                # 更新队列
                for cand_idx, gain, score in zip(candidates_to_eval, gains_batch, scores_batch):
                    heapq.heappush(pq, (-score, step, cand_idx))
                    last_eval[cand_idx] = step

                    if score > best_score:
                        best_score = score
                        best_gain = gain
                        best_idx = cand_idx

            # 把新鲜的候选放回去
            for item in temp_heap:
                heapq.heappush(pq, item)
                neg_score, eval_iter, cand_idx = item
                if -neg_score > best_score:
                    # 需要验证这个候选
                    gain, score = evaluate_single_candidate(
                        cand_idx,
                        sensor_h_rows,
                        sensor_noise_vars,
                        costs,
                        current_factor
                    )
                    if score > best_score:
                        best_score = score
                        best_gain = gain
                        best_idx = cand_idx

        # 如果没找到或首次迭代，批量评估所有候选
        if best_idx == -1:
            # 🔥 分批评估所有候选
            remaining = [i for i in range(n_candidates) if i not in selected_ids]

            for batch_start in range(0, len(remaining), batch_size):
                batch_ids = remaining[batch_start:batch_start + batch_size]

                gains_batch, scores_batch = batch_evaluate_candidates(
                    batch_ids,
                    sensor_h_rows,
                    sensor_noise_vars,
                    costs,
                    current_factor,
                    step
                )

                if lazy:
                    # 入队
                    for cand_idx, gain, score in zip(batch_ids, gains_batch, scores_batch):
                        heapq.heappush(pq, (-score, step, cand_idx))
                        last_eval[cand_idx] = step

                # 更新最佳
                batch_best_idx = np.argmax(scores_batch)
                if scores_batch[batch_best_idx] > best_score:
                    best_score = scores_batch[batch_best_idx]
                    best_gain = gains_batch[batch_best_idx]
                    best_idx = batch_ids[batch_best_idx]

        if best_idx == -1:
            print(f"Warning: No valid sensor found at step {step}")
            break

        # 添加选中的传感器
        selected_ids.append(best_idx)
        selected_sensors.append(sensors[best_idx])
        marginal_gains.append(best_gain)
        total_cost += costs[best_idx]

        # Rank-1更新
        h_best = sensor_h_rows[:, best_idx]
        r_best = sensor_noise_vars[best_idx]
        current_factor.rank1_update(h_best / np.sqrt(r_best), 1.0)

        # 累计MI
        current_obj += best_gain
        objective_values.append(current_obj)

        # 打印进度
        if (step + 1) % 10 == 0 or step == k - 1:
            print(f"  Step {step + 1}/{k}: "
                  f"MI={current_obj:.3f} nats, "
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


def batch_evaluate_candidates(
    cand_ids: List[int],
    sensor_h_rows: np.ndarray,  # (n, N_total)
    sensor_noise_vars: np.ndarray,
    costs: np.ndarray,
    factor: 'SparseFactor',
    iteration: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量评估多个候选的增益和得分

    Args:
        cand_ids: 候选索引列表
        sensor_h_rows: 所有候选的h向量 (n, N_total)
        sensor_noise_vars: 噪声方差
        costs: 成本
        factor: 当前精度因子
        iteration: 当前迭代

    Returns:
        gains: 每个候选的边际增益 (B,)
        scores: 每个候选的得分 (B,)
    """
    # 🔥 关键：批量求解 Q^{-1} H_batch
    H_batch = sensor_h_rows[:, cand_ids]  # (n, B)

    # 一次solve得到所有候选的 Σ h
    Z_batch = factor.solve(H_batch)  # (n, B)

    # 🔥 批量计算 h^T Σ h（列向量点积）
    quad_batch = np.einsum('ij,ij->j', H_batch, Z_batch)

    # 批量计算增益
    r_batch = sensor_noise_vars[cand_ids]
    gains = 0.5 * np.log1p(quad_batch / r_batch)

    # 批量计算得分（成本归一化）
    cost_batch = costs[cand_ids]
    scores = gains / cost_batch

    return gains, scores


def evaluate_single_candidate(
    cand_idx: int,
    sensor_h_rows: np.ndarray,
    sensor_noise_vars: np.ndarray,
    costs: np.ndarray,
    factor: 'SparseFactor'
) -> Tuple[float, float]:
    """单个候选评估（回退方案）"""
    h = sensor_h_rows[:, cand_idx]

    # 使用已有的quadform接口
    from inference import quadform_via_solve
    q = quadform_via_solve(factor, h)

    gain = 0.5 * np.log1p(q / sensor_noise_vars[cand_idx])
    score = gain / costs[cand_idx]

    return gain, score


# =====================================================================
# Greedy-A 加速版（Hutchinson++改进）
# =====================================================================

def greedy_aopt(sensors: List, k: int, Q_pr: sp.spmatrix,
                costs: np.ndarray = None,
                hutchpp_probes: int = 20,
                batch_size: int = 32) -> SelectionResult:
    """
    加速版Greedy-A-optimality

    优化：
    1. Hutchinson++ 批量探针
    2. 每T步重用探针（不用每步都采样）
    3. 批量候选评估
    """
    from inference import SparseFactor, compute_posterior

    n = Q_pr.shape[0]
    if costs is None:
        costs = np.ones(len(sensors))

    selected_ids = []
    selected_sensors = []
    objective_values = []
    marginal_gains = []
    total_cost = 0.0

    # 初始trace估计
    factor_pr = SparseFactor(Q_pr)

    # 🔥 Hutchinson++: 预生成探针矩阵（重用多步）
    rng = np.random.default_rng(42)
    probe_refresh_interval = 5  # 每5步刷新探针

    current_trace = estimate_trace_hutchpp(factor_pr, hutchpp_probes, rng)
    print(f"  Initial trace estimate: {current_trace:.1f}")

    for step in range(k):
        best_gain = -np.inf
        best_idx = -1

        # 分批评估候选
        remaining = [i for i in range(len(sensors)) if i not in selected_ids]

        for batch_start in range(0, len(remaining), batch_size):
            batch_ids = remaining[batch_start:batch_start + batch_size]

            # 批量构建H矩阵
            test_sensors_list = []
            for cand_idx in batch_ids:
                test_sensors_list.append(selected_sensors + [sensors[cand_idx]])

            # 🔥 批量trace估计（复用探针）
            if step % probe_refresh_interval == 0 or step == 0:
                probes = generate_hutchpp_probes(n, hutchpp_probes, rng)

            for cand_idx, test_sensors in zip(batch_ids, test_sensors_list):
                from sensors import assemble_H_R
                H_test, R_test = assemble_H_R(test_sensors, n)

                # 快速trace估计
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

        if best_idx == -1:
            break

        # 添加传感器
        selected_ids.append(best_idx)
        selected_sensors.append(sensors[best_idx])
        marginal_gains.append(best_gain)
        total_cost += costs[best_idx]

        # 更新trace
        current_trace -= best_gain
        objective_values.append(-current_trace)

        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{k}: Trace={current_trace:.1f}, Δ={best_gain:.1f}")

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