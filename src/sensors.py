"""
Sensor modeling and observation matrix assembly.
Supports point sensors, averaging footprints, and convolutional operators.
"""

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
from typing import List, Tuple, Dict


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
                         rng: np.random.Generator,
                         cost_zones: List[Dict] = None) -> List[Sensor]:
    """
    生成传感器池（支持异质化和传统模式）

    🔥 增强版：支持基于位置的成本和噪声分层

    Args:
        geom: 几何对象
        sensors_config: 传感器配置
        rng: 随机数生成器
        cost_zones: 成本区域定义（可选），例如：
            [{'center': (x, y), 'radius': r, 'cost_multiplier': 2.0,
              'noise_multiplier': 0.5, 'allowed_types': ['A', 'B']}]

    Returns:
        sensors: 传感器列表
    """
    if rng is None:
        rng = np.random.default_rng()

    n_total = geom.n
    pool_size = int(n_total * sensors_config.pool_fraction)

    # 选择候选位置
    if sensors_config.pool_strategy == "grid_subsample":
        step = max(1, n_total // pool_size)
        candidate_locs = np.arange(0, n_total, step)[:pool_size]
    elif sensors_config.pool_strategy == "random":
        candidate_locs = rng.choice(n_total, size=pool_size, replace=False)
    elif sensors_config.pool_strategy == "importance":
        # 未来可以基于预测方差加权
        candidate_locs = rng.choice(n_total, size=pool_size, replace=False)
    else:
        raise ValueError(f"Unknown pool strategy: {sensors_config.pool_strategy}")

    # 🔥 检查是否需要异质化（从 config 或显式 cost_zones）
    use_heterogeneous = (
                                hasattr(sensors_config, 'use_heterogeneous') and
                                sensors_config.use_heterogeneous
                        ) or (cost_zones is not None)

    if use_heterogeneous:
        # ===== 异质化模式 =====
        if cost_zones is None:
            # 从配置中获取或使用默认
            if hasattr(sensors_config, 'cost_zones') and sensors_config.cost_zones:
                cost_zones = sensors_config.cost_zones
            else:
                # 创建默认区域
                cost_zones = create_cost_zones_example(geom)

        # 构建类型映射
        type_map = {st.name: st for st in sensors_config.types}

        sensors = []

        for sensor_id, loc in enumerate(candidate_locs):
            loc_coords = geom.coords[loc]

            # 确定该位置的区域属性
            zone_props = _get_zone_properties(loc_coords, cost_zones)

            # 从允许的类型中选择
            allowed_types = zone_props.get('allowed_types',
                                           [st.name for st in sensors_config.types])

            # 筛选可用类型
            available_types = [st for st in sensors_config.types
                               if st.name in allowed_types]

            if not available_types:
                continue

            # 加权选择类型
            type_weights = np.array([sensors_config.type_mix[sensors_config.types.index(st)]
                                     for st in available_types])
            type_weights = type_weights / type_weights.sum()

            stype = rng.choice(available_types, p=type_weights)

            # 应用区域调整
            cost_mult = zone_props.get('cost_multiplier', 1.0)
            noise_mult = zone_props.get('noise_multiplier', 1.0)

            adjusted_cost = stype.cost_gbp * cost_mult
            adjusted_noise_std = stype.noise_std * noise_mult

            # 获取足迹
            idxs, weights = get_footprint_indices(geom, loc, stype.footprint)

            sensor = Sensor(
                id=sensor_id,
                idxs=idxs,
                weights=weights,
                noise_var=adjusted_noise_std ** 2,
                cost=adjusted_cost,
                type_name=stype.name
            )
            sensors.append(sensor)

        # 统计分布
        print(f"  Generated {len(sensors)} heterogeneous sensors:")
        type_counts = {}
        cost_stats = []
        noise_stats = []

        for s in sensors:
            type_counts[s.type_name] = type_counts.get(s.type_name, 0) + 1
            cost_stats.append(s.cost)
            noise_stats.append(np.sqrt(s.noise_var))

        print("    Type distribution:")
        for tname, count in type_counts.items():
            print(f"      {tname}: {count} ({count / len(sensors) * 100:.1f}%)")

        print(f"    Cost range: £{np.min(cost_stats):.0f} - £{np.max(cost_stats):.0f}")
        print(f"    Noise std range: {np.min(noise_stats):.3f} - {np.max(noise_stats):.3f}")

    else:
        # ===== 传统均匀模式 =====
        n_types = len(sensors_config.types)
        type_counts_target = (np.array(sensors_config.type_mix) * pool_size).astype(int)
        type_counts_target[-1] = pool_size - type_counts_target[:-1].sum()

        type_assignments = []
        for type_idx, count in enumerate(type_counts_target):
            type_assignments.extend([type_idx] * count)
        rng.shuffle(type_assignments)

        sensors = []
        for sensor_id, (loc, type_idx) in enumerate(zip(candidate_locs, type_assignments)):
            stype = sensors_config.types[type_idx]

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

        print(f"  Generated {len(sensors)} uniform sensors")

    return sensors


def _get_zone_properties(coords: np.ndarray,
                         cost_zones: List[Dict] = None) -> Dict:
    """
    获取给定坐标点的区域属性

    Args:
        coords: (x, y) 坐标
        cost_zones: 区域定义列表

    Returns:
        属性字典 {'cost_multiplier': 1.5, 'noise_multiplier': 0.8,
                 'allowed_types': [...]}
    """
    if cost_zones is None or len(cost_zones) == 0:
        return {}

    x, y = coords[0], coords[1]

    # 检查所有区域，取最近的一个
    min_dist = np.inf
    best_zone = {}

    for zone in cost_zones:
        center = np.array(zone['center_m'])
        radius = zone['radius_m']

        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        if dist <= radius and dist < min_dist:
            min_dist = dist
            best_zone = zone

    return best_zone


def create_cost_zones_example(geom) -> List[Dict]:
    """
    创建示例成本区域配置

    典型场景：
    - 高风险区（桥面）：昂贵高精度传感器
    - 中等区域：中等成本
    - 远程区域：传感器运输成本高

    Returns:
        区域配置列表
    """
    if geom.mode != "grid2d":
        return []

    nx = int(np.sqrt(geom.n))
    ny = nx

    center_x = nx * geom.h / 2
    center_y = ny * geom.h / 2

    zones = [
        # 高风险区（中心）：只允许高精度，成本 1.5倍
        {
            'center_m': [center_x, center_y],
            'radius_m': nx * geom.h * 0.2,
            'cost_multiplier': 1.5,
            'noise_multiplier': 0.7,
            'allowed_types': ['inertial_profiler', 'photogrammetry']
        },

        # 远程区域（左下角）：运输成本高，噪声大
        {
            'center_m': [center_x * 0.3, center_y * 0.3],
            'radius_m': nx * geom.h * 0.15,
            'cost_multiplier': 2.0,
            'noise_multiplier': 1.3,
            'allowed_types': ['smartphone']  # 只有便宜但噪声大的
        },
    ]

    return zones


def generate_heterogeneous_sensor_pool(geom, sensors_config,
                                       cost_zones: List[Dict] = None,
                                       rng: np.random.Generator = None) -> List[Sensor]:
    """
    生成异质化传感器池（带空间分区的成本/噪声调整）

    这是 generate_sensor_pool() 的增强版，支持：
    - 不同区域使用不同传感器类型
    - 区域化成本倍增
    - 区域化噪声调整

    Args:
        geom: 几何对象
        sensors_config: 传感器配置
        cost_zones: 成本区域定义列表，格式：
            [{'center_m': [x, y], 'radius_m': r,
              'cost_multiplier': 1.5, 'noise_multiplier': 0.8,
              'allowed_types': ['type1', 'type2']}]
        rng: 随机数生成器

    Returns:
        sensors: 异质化传感器列表
    """
    if rng is None:
        rng = np.random.default_rng()

    n_total = geom.n
    pool_size = int(n_total * sensors_config.pool_fraction)

    # 选择候选位置
    if sensors_config.pool_strategy == "grid_subsample":
        step = max(1, n_total // pool_size)
        candidate_locs = np.arange(0, n_total, step)[:pool_size]
    elif sensors_config.pool_strategy == "random":
        candidate_locs = rng.choice(n_total, size=pool_size, replace=False)
    else:
        candidate_locs = rng.choice(n_total, size=pool_size, replace=False)

    # 如果没有提供cost_zones，使用默认行为
    if cost_zones is None:
        print("  No cost zones provided, using uniform sensor generation")
        return generate_sensor_pool(geom, sensors_config, rng)

    # 构建类型映射
    type_map = {st.name: st for st in sensors_config.types}

    sensors = []

    for sensor_id, loc in enumerate(candidate_locs):
        loc_coords = geom.coords[loc]

        # 确定该位置的区域属性
        zone_props = _get_zone_properties(loc_coords, cost_zones)

        # 从允许的类型中选择
        allowed_types = zone_props.get('allowed_types',
                                       [st.name for st in sensors_config.types])

        # 筛选可用类型
        available_types = [st for st in sensors_config.types
                           if st.name in allowed_types]

        if not available_types:
            # 如果区域限制导致没有可用类型，使用所有类型
            available_types = sensors_config.types

        # 加权选择类型
        type_weights = np.array([sensors_config.type_mix[sensors_config.types.index(st)]
                                 for st in available_types])
        type_weights = type_weights / type_weights.sum()

        stype = rng.choice(available_types, p=type_weights)

        # 应用区域调整
        cost_mult = zone_props.get('cost_multiplier', 1.0)
        noise_mult = zone_props.get('noise_multiplier', 1.0)

        adjusted_cost = stype.cost_gbp * cost_mult
        adjusted_noise_std = stype.noise_std * noise_mult

        # 获取足迹
        idxs, weights = get_footprint_indices(geom, loc, stype.footprint)

        sensor = Sensor(
            id=sensor_id,
            idxs=idxs,
            weights=weights,
            noise_var=adjusted_noise_std ** 2,
            cost=adjusted_cost,
            type_name=stype.name
        )
        sensors.append(sensor)

    # 统计分布
    print(f"  Generated {len(sensors)} heterogeneous sensors:")
    type_counts = {}
    cost_stats = []
    noise_stats = []

    for s in sensors:
        type_counts[s.type_name] = type_counts.get(s.type_name, 0) + 1
        cost_stats.append(s.cost)
        noise_stats.append(np.sqrt(s.noise_var))

    print("    Type distribution:")
    for tname, count in type_counts.items():
        print(f"      {tname}: {count} ({count / len(sensors) * 100:.1f}%)")

    print(f"    Cost range: £{np.min(cost_stats):.0f} - £{np.max(cost_stats):.0f}")
    print(f"    Noise std range: {np.min(noise_stats):.3f} - {np.max(noise_stats):.3f}")

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