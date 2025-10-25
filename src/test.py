"""
快速测试脚本 - 最小复现
"""

import numpy as np
from config import load_config
from geometry import build_grid2d_geometry
from spatial_field import build_prior, sample_gmrf
from sensors import generate_sensor_pool
from selection import greedy_mi

print("快速测试...")

# 设置
cfg = load_config()
rng = cfg.get_rng()

# 构建
geom = build_grid2d_geometry(40, 40, h=5.0)
Q_pr, mu_pr = build_prior(geom, cfg.prior)
x_true = sample_gmrf(Q_pr, mu_pr, rng)
sensors = generate_sensor_pool(geom, cfg.sensors, rng)

print(f"n={geom.n}, C={len(sensors)}")
print(f"Q_pr: {Q_pr.shape}")
print(f"mu_pr: {mu_pr.shape}")
print(f"x_true: {x_true.shape}")

# 测试调用
print("\n测试 greedy_mi...")
try:
    costs = np.array([s.cost for s in sensors], dtype=float)
    print(f"costs: {costs.shape}")

    result = greedy_mi(sensors, k=3, Q_pr=Q_pr, costs=costs)
    print(f"✓ 成功! 选中: {result.selected_ids}")

except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()