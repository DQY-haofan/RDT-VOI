"""
诊断脚本：在 selection.py 中插入打印语句追踪执行
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from config import load_scenario_config
from geometry import build_grid2d_geometry
from spatial_field import build_prior, sample_gmrf
from sensors import generate_sensor_pool

print("=" * 70)
print("  诊断：追踪 selection.py 函数执行")
print("=" * 70)

# 构建环境
cfg = load_scenario_config('A')
geom = build_grid2d_geometry(10, 10, h=5.0)
Q_pr, mu_pr = build_prior(geom, cfg.prior)
rng = cfg.get_rng()
x_true = sample_gmrf(Q_pr, mu_pr, rng)
sensors = generate_sensor_pool(geom, cfg.sensors, rng)

print(f"\n环境: n={geom.n}, sensors={len(sensors)}")

# 直接导入并修改 greedy_mi
print("\n[1] 测试 greedy_mi with keep_fraction=None...")

from selection import greedy_mi

# 插入调试版本
def greedy_mi_debug(sensors, k, Q_pr, costs=None, keep_fraction=None, **kwargs):
    """带调试输出的版本"""
    C = len(sensors)

    print(f"\n  === greedy_mi_debug 开始 ===")
    print(f"  输入参数:")
    print(f"    C (候选数): {C}")
    print(f"    k (预算): {k}")
    print(f"    keep_fraction: {keep_fraction} (type: {type(keep_fraction)})")

    # 🔥 关键：检查这里是否正确处理
    if keep_fraction is None:
        print(f"  keep_fraction 为 None，开始计算...")
        n_keep_budget = 4 * k
        n_keep_pool = int(np.ceil(0.25 * C))
        n_keep = max(n_keep_budget, n_keep_pool, k + 10)
        actual_keep_fraction = n_keep / C
        print(f"    计算结果:")
        print(f"      n_keep_budget (4k): {n_keep_budget}")
        print(f"      n_keep_pool (0.25C): {n_keep_pool}")
        print(f"      n_keep (max): {n_keep}")
        print(f"      actual_keep_fraction: {actual_keep_fraction:.2%}")
    else:
        print(f"  使用传入的 keep_fraction: {keep_fraction}")
        actual_keep_fraction = keep_fraction
        n_keep = int(C * actual_keep_fraction)
        print(f"    n_keep: {n_keep}")

    print(f"\n  准备调用原始 greedy_mi...")
    print(f"    传递 keep_fraction: {actual_keep_fraction}")

    # 调用原始函数
    return greedy_mi(sensors, k, Q_pr, costs=costs,
                     keep_fraction=actual_keep_fraction, **kwargs)

# 测试
try:
    costs = np.array([s.cost for s in sensors])
    result = greedy_mi_debug(
        sensors=sensors,
        k=3,
        Q_pr=Q_pr,
        costs=costs,
        keep_fraction=None,  # 显式传递 None
        use_cost=True
    )
    print(f"\n  ✓ greedy_mi 执行成功")
    print(f"    选择: {result.selected_ids}")

except Exception as e:
    print(f"\n  ✗ greedy_mi 执行失败")
    print(f"    错误类型: {type(e).__name__}")
    print(f"    错误信息: {e}")
    print(f"\n  完整追踪:")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("  诊断完成")
print("=" * 70)