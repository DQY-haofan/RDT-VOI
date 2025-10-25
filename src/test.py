"""
验证配置是否能体现算法差异
"""
from config import load_config
from geometry import build_grid2d_geometry
from spatial_field import build_prior
import numpy as np

cfg = load_config("config.yaml")  # 使用优化配置
geom = build_grid2d_geometry(40, 40, h=5.0)
Q_pr, mu_pr = build_prior(geom, cfg.prior)

from inference import SparseFactor, compute_posterior_variance_diagonal

print("=" * 70)
print("  配置验证")
print("=" * 70)

# 1. 检查先验异质性
print("\n[1] 先验异质性检查")
F = SparseFactor(Q_pr)
n_samples = 100
test_idx = np.linspace(0, geom.n-1, n_samples, dtype=int)
vars = compute_posterior_variance_diagonal(F, test_idx)
cv = vars.std() / vars.mean()
print(f"  CV = {cv:.2%}")
if cv < 0.15:
    print(f"  ⚠️  CV 太小！建议 CV > 20%")
elif cv > 0.20:
    print(f"  ✓ CV 良好！足以区分方法")

# 2. 检查决策阈值位置
print("\n[2] 决策阈值检查")
tau = cfg.decision.tau_iri
mu_mean = mu_pr.mean()
mu_std = mu_pr.std()
z_score = (tau - mu_mean) / mu_std
percentile = 100 * (1 + np.erf(z_score / np.sqrt(2))) / 2
print(f"  tau = {tau:.2f}")
print(f"  μ_prior = {mu_mean:.2f} ± {mu_std:.2f}")
print(f"  tau 位于 {percentile:.1f} 百分位")
if 40 <= percentile <= 60:
    print(f"  ✓ 阈值位置理想（接近中位数）")
else:
    print(f"  ⚠️  阈值偏离中位数，决策不够敏感")

# 3. 检查损失函数不对称性
print("\n[3] 损失函数不对称性")
ratio_fn_fp = cfg.decision.L_FN_gbp / cfg.decision.L_FP_gbp
ratio_fn_tp = cfg.decision.L_FN_gbp / cfg.decision.L_TP_gbp
print(f"  FN:FP = {ratio_fn_fp:.1f}:1")
print(f"  FN:TP = {ratio_fn_tp:.1f}:1")
if ratio_fn_fp >= 15:
    print(f"  ✓ 不对称性足够强（≥15:1）")
else:
    print(f"  ⚠️  建议增大 L_FN 到 150k+")

# 4. 检查预算范围
print("\n[4] 预算范围检查")
budgets = cfg.selection.budgets
pool_size = int(geom.n * cfg.sensors.pool_fraction)
print(f"  候选池大小: {pool_size}")
print(f"  预算范围: {budgets}")
print(f"  最大预算占比: {max(budgets)/pool_size:.1%}")
if min(budgets) <= 5:
    print(f"  ✓ 包含小预算（差距最明显）")
if max(budgets) / pool_size <= 0.15:
    print(f"  ✓ 最大预算合理（不过饱和）")

print("\n" + "=" * 70)