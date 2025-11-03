"""
先验不确定性参数调整指南

问题：Prior variance CV=0.52% 过低
警告：Prior uncertainty very uniform! MI advantage will be weak.

原因：先验不确定性在空间上过于均匀，导致基于信息的方法（如MI、EVI）
      相对于简单方法没有优势。

解决方案：增加先验不确定性的空间异质性
"""

import numpy as np
import yaml
from pathlib import Path

# ============================================================================
# 问题诊断
# ============================================================================

print("=" * 80)
print("  先验不确定性异质性调整指南")
print("=" * 80)

print("""
问题分析：
-----------
当前状态：
  ✓ Final σ²=0.2500 (error: 0.0%)  ← 目标方差正确
  ✗ Prior variance CV=0.52%        ← 方差变异系数太小
  ⚠️  Prior uncertainty very uniform! ← 不确定性过于均匀

CV (Coefficient of Variation) = std(σ²) / mean(σ²)
  • CV < 5%: 非常均匀（问题）
  • CV = 10-30%: 中等异质性（可接受）
  • CV > 30%: 高度异质性（理想）

为什么这是问题？
-----------
当先验不确定性均匀时：
  • 所有位置的"信息价值"相似
  • MI/EVI 无法识别"高价值"传感器位置
  • 性能接近随机选择或均匀分布

解决方案：
-----------
增加空间异质性有两种方法：
  1. 增加/增强 hotspots（高不确定性区域）
  2. 增大 beta_base/beta_hot 比值（降低hotspots的精度）
""")

# ============================================================================
# 方案1：调整 Hotspots
# ============================================================================

print("\n" + "=" * 80)
print("  方案1：调整 Hotspots 配置")
print("=" * 80)

print("""
Hotspots 是先验中不确定性较高的区域。
通过增加hotspots数量、调整位置和强度，可以创建空间异质性。

当前配置（baseline_config.yaml）：
""")

CURRENT_HOTSPOTS = """
prior:
  # ... 其他参数 ...
  beta_base: 1.0e-03  # 基础精度（高 = 低不确定性）
  beta_hot: 1.0e-05   # Hotspot精度（低 = 高不确定性）
  hotspots:
    - center: [25.0, 75.0]
      radius: 25.0
    - center: [75.0, 25.0]
      radius: 25.0
    - center: [50.0, 50.0]
      radius: 25.0
"""

print(CURRENT_HOTSPOTS)

print("""
问题：
  • beta_base/beta_hot 比值 = 1e-3/1e-5 = 100（较小）
  • 只有3个hotspots
  • Hotspots半径较大，可能重叠

建议调整（选项A - 增强现有hotspots）：
""")

IMPROVED_HOTSPOTS_A = """
prior:
  nu: 1.0
  kappa: 0.2
  sigma2: 0.25
  alpha: 2
  beta: 1.0e-05
  mu_prior_mean: 2.2
  mu_prior_std: 0.3
  beta_base: 1.0e-02  # ← 提高10倍（降低基础不确定性）
  beta_hot: 1.0e-05   # ← 保持不变（保持hotspots高不确定性）
  hotspots:
    - center: [25.0, 75.0]
      radius: 20.0       # ← 减小半径（避免过度重叠）
    - center: [75.0, 25.0]
      radius: 20.0
    - center: [50.0, 50.0]
      radius: 20.0
    - center: [25.0, 25.0]  # ← 新增hotspot
      radius: 15.0
    - center: [75.0, 75.0]  # ← 新增hotspot
      radius: 15.0
"""

print(IMPROVED_HOTSPOTS_A)

print("""
效果：
  • beta_base/beta_hot 比值 = 1e-2/1e-5 = 1000（提高10倍）
  • 5个hotspots（增加2个）
  • 预期 CV ≈ 15-25%（显著改善）

建议调整（选项B - 激进增强）：
""")

IMPROVED_HOTSPOTS_B = """
prior:
  nu: 1.0
  kappa: 0.2
  sigma2: 0.25
  alpha: 2
  beta: 1.0e-05
  mu_prior_mean: 2.2
  mu_prior_std: 0.3
  beta_base: 5.0e-02  # ← 提高50倍
  beta_hot: 1.0e-05   # ← 保持不变
  hotspots:
    - center: [20.0, 80.0]
      radius: 18.0
    - center: [80.0, 20.0]
      radius: 18.0
    - center: [50.0, 50.0]
      radius: 15.0
    - center: [20.0, 20.0]
      radius: 12.0
    - center: [80.0, 80.0]
      radius: 12.0
    - center: [35.0, 65.0]  # ← 额外的小hotspot
      radius: 10.0
    - center: [65.0, 35.0]
      radius: 10.0
"""

print(IMPROVED_HOTSPOTS_B)

print("""
效果：
  • beta_base/beta_hot 比值 = 5e-2/1e-5 = 5000（提高50倍）
  • 7个hotspots（多样化的大小和位置）
  • 预期 CV ≈ 30-50%（理想的异质性）
""")

# ============================================================================
# 方案2：理解参数的作用
# ============================================================================

print("\n" + "=" * 80)
print("  参数作用详解")
print("=" * 80)

print("""
Beta参数（精度参数）：
-----------------------
beta_base: 基础区域的精度（nugget effect）
  • 值越大 → 精度越高 → 不确定性越低
  • 典型值：1e-5 到 1e-1

beta_hot: Hotspot区域的精度
  • 值越小 → 精度越低 → 不确定性越高
  • 典型值：1e-6 到 1e-4

关键比值：beta_base / beta_hot
  • 比值越大 → hotspots 相对不确定性越高
  • 比值 = 100: 轻微异质性
  • 比值 = 1000: 中等异质性（推荐）
  • 比值 = 5000+: 强异质性

Hotspot配置：
-----------------------
center: [x, y]
  • 坐标范围：[0, 100]（对于100x100的域）
  • 建议：避免边界，分散布局

radius: 半径（米）
  • 影响hotspot覆盖范围
  • 典型值：10-25米
  • 注意：过大会导致重叠

数量建议：
  • 小网格（15x15）：3-5个hotspots
  • 中等网格（20x20）：5-7个hotspots
  • 大网格（25x25）：7-10个hotspots
""")

# ============================================================================
# 方案3：自动生成配置
# ============================================================================

print("\n" + "=" * 80)
print("  自动生成 Hotspot 配置")
print("=" * 80)

def generate_hotspot_config(grid_size=20, n_hotspots=5, beta_ratio=1000):
    """
    自动生成优化的hotspot配置

    Args:
        grid_size: 网格大小（nx=ny）
        n_hotspots: hotspot数量
        beta_ratio: beta_base/beta_hot 比值
    """
    domain_size = grid_size * 5  # 假设spacing=5m

    # 计算beta值
    beta_hot = 1.0e-05
    beta_base = beta_hot * beta_ratio

    # 生成hotspot位置（使用黄金分割避免聚集）
    rng = np.random.default_rng(42)

    # 在网格上分布hotspots
    hotspots = []

    # 策略：在(0.2, 0.8)范围内均匀分布
    for i in range(n_hotspots):
        # 使用泊松盘采样避免过近
        attempts = 0
        while attempts < 100:
            x = rng.uniform(0.2 * domain_size, 0.8 * domain_size)
            y = rng.uniform(0.2 * domain_size, 0.8 * domain_size)

            # 检查是否与现有hotspots过近
            too_close = False
            min_dist = 20  # 最小距离
            for h in hotspots:
                dist = np.sqrt((x - h['center'][0])**2 + (y - h['center'][1])**2)
                if dist < min_dist:
                    too_close = True
                    break

            if not too_close:
                # 半径随机化（大的和小的混合）
                if i < n_hotspots // 2:
                    radius = rng.uniform(15, 20)
                else:
                    radius = rng.uniform(10, 15)

                hotspots.append({
                    'center': [float(x), float(y)],
                    'radius': float(radius)
                })
                break

            attempts += 1

    # 生成YAML配置
    config = {
        'prior': {
            'nu': 1.0,
            'kappa': 0.2,
            'sigma2': 0.25,
            'alpha': 2,
            'beta': beta_hot,
            'mu_prior_mean': 2.2,
            'mu_prior_std': 0.3,
            'beta_base': float(beta_base),
            'beta_hot': float(beta_hot),
            'hotspots': hotspots
        }
    }

    return config

# 生成示例配置
print("""
示例：自动生成的配置
""")

config_example = generate_hotspot_config(grid_size=20, n_hotspots=5, beta_ratio=1000)

print(yaml.dump(config_example, default_flow_style=False, sort_keys=False))

print(f"""
生成的配置特点：
  • beta_base/beta_hot = 1000
  • 5个hotspots，位置随机但分散
  • 半径混合（大hotspots和小hotspots）
  • 预期 CV ≈ 20-30%
""")

# ============================================================================
# 方案4：验证调整效果
# ============================================================================

print("\n" + "=" * 80)
print("  验证调整效果")
print("=" * 80)

print("""
如何验证调整是否有效：

1. 运行实验并查看输出：
   python main.py --quick-test

2. 检查先验构建输出：
   ✓ Final σ²=0.2500 (error: 0.0%)
   ✓ Prior variance CV=XX.X%  ← 期望 > 10%

3. 如果仍然显示 "Prior uncertainty very uniform"：
   • 继续增大 beta_base/beta_hot 比值
   • 增加hotspots数量
   • 减小hotspots半径（避免过度重叠）

4. 查看方法性能差异：
   • MI/EVI应该显著优于uniform/random
   • 如果差异不明显，继续增强异质性

预期结果（修复后）：
   Prior variance CV=20.5%  ← 显著改善
   ✓ Spatial heterogeneity sufficient for information-based methods
""")

# ============================================================================
# 快速修复步骤
# ============================================================================

print("\n" + "=" * 80)
print("  快速修复步骤（5分钟）")
print("=" * 80)

print("""
Step 1: 打开 baseline_config.yaml

Step 2: 找到 prior 部分，修改以下参数：

  prior:
    # ... 保持其他参数不变 ...
    beta_base: 1.0e-02  # ← 从 1.0e-03 改为 1.0e-02
    beta_hot: 1.0e-05   # ← 保持不变
    hotspots:
      - center: [25.0, 75.0]
        radius: 20.0      # ← 可选：从 25.0 改为 20.0
      - center: [75.0, 25.0]
        radius: 20.0
      - center: [50.0, 50.0]
        radius: 20.0
      - center: [25.0, 25.0]  # ← 新增
        radius: 15.0
      - center: [75.0, 75.0]  # ← 新增
        radius: 15.0

Step 3: 保存并测试

  python main.py --quick-test

Step 4: 检查输出

  应该看到：
    Prior variance CV=15-25%  （而不是 0.52%）
    不再有 "Prior uncertainty very uniform" 警告

Step 5: 如果仍然太低

  进一步增大 beta_base：
    beta_base: 5.0e-02  # 增加到 5e-2

  或者使用自动生成的配置（见上面的示例）
""")

# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 80)
print("  总结")
print("=" * 80)

print("""
关键要点：
-----------
1. CV < 5% = 先验过于均匀，MI/EVI 无优势
2. CV = 10-30% = 合理的异质性，推荐
3. CV > 30% = 高度异质性，信息方法优势明显

调整策略：
-----------
• 温和调整：beta_base x10，增加2个hotspots → CV ≈ 15-20%
• 激进调整：beta_base x50，增加4个hotspots → CV ≈ 30-50%
• 自动生成：使用上面的 generate_hotspot_config()

验证：
-----------
• 运行 python main.py --quick-test
• 检查 Prior variance CV
• 确认 MI/EVI 性能 > uniform/random

推荐的快速修复：
-----------
将 beta_base 从 1.0e-03 改为 1.0e-02
增加2个hotspots（共5个）
预期CV提升到15-25%
""")

print("=" * 80)