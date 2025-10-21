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

    Uses lazy evaluation and rank-1 Cholesky updates for efficiency.

    Args:
        sensors: List of Sensor objects
        k: Budget (number of sensors)
        Q_pr: Prior precision matrix
        costs: Optional cost vector (if None, extract from sensors)
        lazy: Use lazy-greedy optimization

    Returns:
        result: SelectionResult with selected sensors and diagnostics
    """
    from inference import SparseFactor, quadform_via_solve
    from sensors import assemble_H_R

    n = Q_pr.shape[0]
    n_candidates = len(sensors)

    # ✅ 修复：如果没有提供成本，从传感器对象中提取真实成本
    if costs is None:
        costs = np.array([s.cost for s in sensors], dtype=float)
        print(f"  Using real sensor costs: min=£{costs.min():.0f}, max=£{costs.max():.0f}, mean=£{costs.mean():.0f}")

    # Initialize
    selected_ids = []
    selected_sensors = []
    objective_values = []
    marginal_gains = []
    total_cost = 0.0

    # Start with prior
    current_factor = SparseFactor(Q_pr)
    current_obj = 0.0  # MI starts at 0

    # Priority queue for lazy evaluation: (-gain/cost, iteration, sensor_id)
    if lazy:
        pq = []
        last_eval = {}  # Track when each sensor was last evaluated

    # Precompute sensor H rows
    sensor_h_rows = []
    sensor_noise_vars = []
    for s in sensors:
        h = np.zeros(n)
        h[s.idxs] = s.weights
        sensor_h_rows.append(h)
        sensor_noise_vars.append(s.noise_var)

    # Greedy loop
    for step in range(k):
        best_gain = -np.inf
        best_idx = -1

        if lazy and step > 0:
            # Lazy evaluation
            while pq:
                neg_ratio, eval_iter, cand_idx = heapq.heappop(pq)

                if cand_idx in selected_ids:
                    continue

                # Check if stale
                if eval_iter < step:
                    # Re-evaluate
                    h = sensor_h_rows[cand_idx]
                    r = sensor_noise_vars[cand_idx]
                    q = quadform_via_solve(current_factor, h)
                    gain = 0.5 * np.log(1 + q / r)
                    ratio = gain / costs[cand_idx]

                    # Re-insert with updated value
                    heapq.heappush(pq, (-ratio, step, cand_idx))
                    last_eval[cand_idx] = step
                else:
                    # Fresh evaluation, use it
                    best_idx = cand_idx
                    best_gain = -neg_ratio * costs[cand_idx]
                    break

        if best_idx == -1:
            # Evaluate all candidates
            for cand_idx in range(n_candidates):
                if cand_idx in selected_ids:
                    continue

                h = sensor_h_rows[cand_idx]
                r = sensor_noise_vars[cand_idx]

                # Compute marginal MI gain
                q = quadform_via_solve(current_factor, h)
                gain = 0.5 * np.log(1 + q / r)

                ratio = gain / costs[cand_idx]

                if lazy:
                    heapq.heappush(pq, (-ratio, step, cand_idx))
                    last_eval[cand_idx] = step

                if ratio * costs[cand_idx] > best_gain:
                    best_gain = gain
                    best_idx = cand_idx

        if best_idx == -1:
            print(f"Warning: No valid sensor found at step {step}")
            break

        # Add selected sensor
        selected_ids.append(best_idx)
        selected_sensors.append(sensors[best_idx])
        marginal_gains.append(best_gain)
        total_cost += costs[best_idx]

        # Update factor with rank-1 update
        h_best = sensor_h_rows[best_idx]
        r_best = sensor_noise_vars[best_idx]
        current_factor.rank1_update(h_best / np.sqrt(r_best), 1.0)

        current_obj += best_gain
        objective_values.append(current_obj)

        if (step + 1) % 10 == 0 or step == k - 1:
            print(f"  Step {step + 1}/{k}: MI={current_obj:.3f} nats ({current_obj/np.log(2):.2f} bits), "
                  f"ΔMI={best_gain:.3f}, cost=£{total_cost:.0f}, sensor_type={sensors[best_idx].type_name}")

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