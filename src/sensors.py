"""
Sensor modeling and observation matrix assembly.
Supports point sensors, averaging footprints, and convolutional operators.
"""

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Sensor:
    """
    Sensor specification.

    Attributes:
        id: Unique sensor identifier
        idxs: State indices in sensor footprint
        weights: Weights for each index (sum to 1 for averaging)
        noise_var: Observation noise variance
        cost: Deployment cost (GBP)
        type_name: Sensor type identifier
    """
    id: int
    idxs: np.ndarray  # (k,) array of state indices
    weights: np.ndarray  # (k,) array of weights
    noise_var: float
    cost: float
    type_name: str = "generic"

    @property
    def h_row(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (indices, values, shape) for sparse H row construction."""
        return self.idxs, self.weights, None

    def __repr__(self):
        footprint_size = len(self.idxs)
        return (f"Sensor(id={self.id}, type={self.type_name}, "
                f"footprint={footprint_size}, noise_std={np.sqrt(self.noise_var):.3f}, "
                f"cost=£{self.cost:.0f})")


def get_footprint_indices(geom, center_idx: int,
                          footprint_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get state indices and weights for a sensor footprint.

    Args:
        geom: Geometry object
        center_idx: Center location index
        footprint_type: "point" | "avg3x3" | "avg5x5" | "avg7x7"

    Returns:
        idxs: Array of state indices in footprint
        weights: Corresponding weights (sum to 1)
    """
    if footprint_type == "point":
        return np.array([center_idx]), np.array([1.0])

    elif footprint_type.startswith("avg"):
        # Extract window size
        size = int(footprint_type.replace("avg", "").replace("x", "")[0])
        radius = size // 2

        if geom.mode == "grid2d":
            # 2D grid footprint
            nx = int(np.sqrt(geom.n))
            ny = nx

            # Convert center to (i, j)
            center_i = center_idx // ny
            center_j = center_idx % ny

            idxs = []
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni, nj = center_i + di, center_j + dj
                    if 0 <= ni < nx and 0 <= nj < ny:
                        idxs.append(ni * ny + nj)

            idxs = np.array(idxs)
            weights = np.ones(len(idxs)) / len(idxs)  # Uniform average
            return idxs, weights

        elif geom.mode == "polyline1d":
            # 1D polyline footprint
            idxs = []
            for di in range(-radius, radius + 1):
                ni = center_idx + di
                if 0 <= ni < geom.n:
                    idxs.append(ni)

            idxs = np.array(idxs)
            weights = np.ones(len(idxs)) / len(idxs)
            return idxs, weights

        else:
            # Graph: use k-hop neighborhood
            # Simple BFS implementation
            visited = {center_idx}
            frontier = {center_idx}
            for _ in range(radius):
                new_frontier = set()
                for node in frontier:
                    neighbors = geom.get_neighbors(node)
                    for nb in neighbors:
                        if nb not in visited:
                            new_frontier.add(nb)
                            visited.add(nb)
                frontier = new_frontier

            idxs = np.array(sorted(visited))
            weights = np.ones(len(idxs)) / len(idxs)
            return idxs, weights

    else:
        raise ValueError(f"Unknown footprint type: {footprint_type}")


def generate_sensor_pool(geom, sensors_config,
                         rng: np.random.Generator) -> List[Sensor]:
    """
    Generate candidate sensor pool.

    Args:
        geom: Geometry object
        sensors_config: SensorsConfig from config
        rng: Random number generator

    Returns:
        sensors: List of Sensor objects
    """
    n_total = geom.n
    pool_size = int(n_total * sensors_config.pool_fraction)

    # Select candidate locations
    if sensors_config.pool_strategy == "grid_subsample":
        # Uniform subsample
        step = max(1, n_total // pool_size)
        candidate_locs = np.arange(0, n_total, step)[:pool_size]

    elif sensors_config.pool_strategy == "random":
        candidate_locs = rng.choice(n_total, size=pool_size, replace=False)

    elif sensors_config.pool_strategy == "importance":
        # Could weight by predicted variance, centrality, etc.
        # For now, fallback to random
        candidate_locs = rng.choice(n_total, size=pool_size, replace=False)

    else:
        raise ValueError(f"Unknown pool strategy: {sensors_config.pool_strategy}")

    # Assign sensor types according to mix
    n_types = len(sensors_config.types)
    type_counts = (np.array(sensors_config.type_mix) * pool_size).astype(int)
    # Adjust for rounding
    type_counts[-1] = pool_size - type_counts[:-1].sum()

    type_assignments = []
    for type_idx, count in enumerate(type_counts):
        type_assignments.extend([type_idx] * count)
    rng.shuffle(type_assignments)

    # Create sensors
    sensors = []
    for sensor_id, (loc, type_idx) in enumerate(zip(candidate_locs, type_assignments)):
        stype = sensors_config.types[type_idx]

        # Get footprint
        idxs, weights = get_footprint_indices(geom, loc, stype.footprint)

        sensor = Sensor(
            id=sensor_id,
            idxs=idxs,
            weights=weights,
            noise_var=stype.noise_std ** 2,
            cost=stype.cost_gbp,
            type_name=stype.name
        )
        sensors.append(sensor)

    return sensors


def assemble_H_R(sensors: List[Sensor], n: int) -> Tuple[sp.spmatrix, np.ndarray]:
    """
    Assemble sparse observation matrix H and noise variance vector R.

    Args:
        sensors: List of m sensors
        n: State dimension

    Returns:
        H: Sparse observation matrix (m × n)
        R_diag: Diagonal of noise covariance (m,)
    """
    m = len(sensors)

    row_idx = []
    col_idx = []
    data = []
    R_diag = np.zeros(m)

    for i, sensor in enumerate(sensors):
        # Add sensor's row to H
        for j, (state_idx, weight) in enumerate(zip(sensor.idxs, sensor.weights)):
            row_idx.append(i)
            col_idx.append(state_idx)
            data.append(weight)

        # Store noise variance
        R_diag[i] = sensor.noise_var

    H = sp.coo_matrix((data, (row_idx, col_idx)), shape=(m, n))

    return H.tocsr(), R_diag


def get_observation(x_true: np.ndarray,
                    sensors: List[Sensor],
                    rng: np.random.Generator = None) -> Tuple[np.ndarray, sp.spmatrix, np.ndarray]:
    """
    Generate noisy observations from true state.

    Args:
        x_true: True state vector (n,)
        sensors: List of sensors
        rng: Random number generator

    Returns:
        y: Observation vector (m,)
        H: Observation matrix (m × n)
        R_diag: Noise variances (m,)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(x_true)
    H, R_diag = assemble_H_R(sensors, n)

    # Noiseless observation
    y_clean = H @ x_true

    # Add noise
    noise = rng.normal(0, np.sqrt(R_diag))
    y = y_clean + noise

    return y, H, R_diag


def compute_sensor_coverage(sensors: List[Sensor], n: int) -> np.ndarray:
    """
    Compute how many sensors cover each state location.

    Args:
        sensors: List of sensors
        n: State dimension

    Returns:
        coverage: (n,) array of coverage counts
    """
    coverage = np.zeros(n, dtype=int)
    for sensor in sensors:
        coverage[sensor.idxs] += 1
    return coverage


if __name__ == "__main__":
    from config import load_config
    from geometry import build_grid2d_geometry

    cfg = load_config()
    rng = cfg.get_rng()

    # Build geometry
    geom = build_grid2d_geometry(20, 20, h=cfg.geometry.h)

    # Generate sensor pool
    sensors = generate_sensor_pool(geom, cfg.sensors, rng)
    print(f"Generated {len(sensors)} candidate sensors")

    # Show type distribution
    type_counts = {}
    for s in sensors:
        type_counts[s.type_name] = type_counts.get(s.type_name, 0) + 1
    print("Type distribution:", type_counts)

    # Test observation
    x_true = rng.normal(2.0, 0.5, size=geom.n)
    y, H, R = get_observation(x_true, sensors[:10], rng)
    print(f"Observation: m={len(y)}, H shape={H.shape}, nnz={H.nnz}")

    # Coverage
    coverage = compute_sensor_coverage(sensors, geom.n)
    print(f"Coverage: mean={coverage.mean():.2f}, max={coverage.max()}")