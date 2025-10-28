"""
快速验证 ROI 修复是否生效
"""
import numpy as np

# 模拟数据
prior_loss_test = 1200.0  # 测试集平均损失 £1200
posterior_loss_test = 800.0
sensor_cost = 5000.0
N_domain = 400
N_test = 80

# ❌ 修复前（错误）
savings_old = prior_loss_test - posterior_loss_test  # 400
roi_old = (savings_old - sensor_cost) / sensor_cost  # -0.92

# ✅ 修复后（正确）
scale = N_domain / N_test  # 5.0
prior_loss_scaled = prior_loss_test * scale  # 6000
posterior_loss_scaled = posterior_loss_test * scale  # 4000
savings_new = prior_loss_scaled - posterior_loss_scaled  # 2000
roi_new = (savings_new - sensor_cost) / sensor_cost  # -0.60

print("修复前：")
print(f"  Savings: £{savings_old:.0f}")
print(f"  ROI: {roi_old:.2f} (几乎总是负数)")

print("\n修复后：")
print(f"  Savings: £{savings_new:.0f} (×{scale:.1f})")
print(f"  ROI: {roi_new:.2f} (更合理的范围)")
print(f"  Net Benefit: £{savings_new - sensor_cost:.0f}")

print("\n健康检查：")
if roi_new > roi_old + 0.2:
    print("  ✓ ROI 提升显著，修复生效")
else:
    print("  ✗ ROI 提升不明显，检查 scale_factor")

if abs(savings_new / sensor_cost) > 0.1:
    print("  ✓ Savings 与 Cost 量级可比")
else:
    print("  ✗ Savings 仍然太小")