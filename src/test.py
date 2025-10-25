"""
验证配置是否能体现算法差异
"""
from config import load_config
from geometry import build_grid2d_geometry
from spatial_field import build_prior
import numpy as np
from scipy.special import erf  # ✅ 修复：从 scipy 导入
from scipy.stats import norm   # 或者用 norm.cdf

def validate_config(config_path="config.yaml"):
    """验证配置参数"""

    cfg = load_config(config_path)
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

    print(f"  先验方差 CV = {cv:.2%}")
    print(f"  方差范围: [{vars.min():.4f}, {vars.max():.4f}]")
    print(f"  方差均值: {vars.mean():.4f}")

    if cv < 0.15:
        print(f"  ⚠️  CV={cv:.2%} 太小！建议 CV > 20%")
        print(f"      -> 增大 beta_base 或减小 beta_hot")
    elif cv > 0.20:
        print(f"  ✓ CV={cv:.2%} 良好！足以区分方法")
    else:
        print(f"  ⚡ CV={cv:.2%} 勉强可用，建议调整到 >25%")

    # 2. 检查决策阈值位置
    print("\n[2] 决策阈值检查")
    tau = cfg.decision.tau_iri
    mu_mean = mu_pr.mean()
    mu_std = mu_pr.std()

    # 使用 scipy.stats.norm
    percentile = norm.cdf(tau, loc=mu_mean, scale=mu_std) * 100

    print(f"  决策阈值 tau = {tau:.2f}")
    print(f"  先验均值 μ = {mu_mean:.2f}")
    print(f"  先验标准差 σ = {mu_std:.2f}")
    print(f"  tau 位于先验分布的 {percentile:.1f}th 百分位")

    if 40 <= percentile <= 60:
        print(f"  ✓ 阈值位置理想（接近中位数，决策最敏感）")
    elif 30 <= percentile <= 70:
        print(f"  ⚡ 阈值位置可用，但建议更接近 50%")
    else:
        print(f"  ⚠️  阈值偏离中位数！决策不够敏感")
        if percentile < 40:
            print(f"      -> 建议增大 tau_iri 到 {mu_mean:.2f}")
        else:
            print(f"      -> 建议减小 tau_iri 到 {mu_mean:.2f}")

    # 3. 检查损失函数不对称性
    print("\n[3] 损失函数不对称性")
    L_FN = cfg.decision.L_FN_gbp
    L_FP = cfg.decision.L_FP_gbp
    L_TP = cfg.decision.L_TP_gbp
    L_TN = cfg.decision.L_TN_gbp

    ratio_fn_fp = L_FN / L_FP
    ratio_fn_tp = L_FN / L_TP

    print(f"  L_FN = £{L_FN:,}")
    print(f"  L_FP = £{L_FP:,}")
    print(f"  L_TP = £{L_TP:,}")
    print(f"  L_TN = £{L_TN:,}")
    print(f"  ")
    print(f"  成本比例:")
    print(f"    FN:FP = {ratio_fn_fp:.1f}:1  (漏检 vs 误报)")
    print(f"    FN:TP = {ratio_fn_tp:.1f}:1  (漏检 vs 正确维护)")

    # 计算 Bayes 阈值
    p_threshold = L_FP / (L_FP + L_FN - L_TP)
    print(f"  ")
    print(f"  Bayes 最优概率阈值: p* = {p_threshold:.3f}")

    if ratio_fn_fp >= 15:
        print(f"  ✓ 不对称性强（FN:FP ≥ 15:1），EVI 优势明显")
    elif ratio_fn_fp >= 10:
        print(f"  ⚡ 不对称性中等（10:1 ≤ FN:FP < 15:1）")
        print(f"      -> 建议增大 L_FN 到 {L_FP * 15:,.0f}+ 以强化 EVI")
    else:
        print(f"  ⚠️  不对称性太弱（FN:FP < 10:1）")
        print(f"      -> 强烈建议增大 L_FN 到 {L_FP * 20:,.0f}")

    # 4. 检查预算范围
    print("\n[4] 预算范围检查")
    budgets = cfg.selection.budgets
    pool_size = int(geom.n * cfg.sensors.pool_fraction)

    print(f"  状态空间大小: n = {geom.n}")
    print(f"  候选池大小: C = {pool_size}")
    print(f"  预算点: {budgets}")
    print(f"  预算范围: [{min(budgets)}, {max(budgets)}]")
    print(f"  最小预算占比: {min(budgets)/pool_size:.1%}")
    print(f"  最大预算占比: {max(budgets)/pool_size:.1%}")

    issues = []
    if min(budgets) <= 5:
        print(f"  ✓ 包含小预算（k≤5），差距最明显")
    else:
        issues.append(f"建议添加小预算点（3, 5）")

    if max(budgets) / pool_size <= 0.15:
        print(f"  ✓ 最大预算合理（≤15% 池大小）")
    else:
        issues.append(f"最大预算可能过大，建议 ≤ {int(pool_size*0.15)}")

    # 检查预算点密度
    budget_gaps = np.diff(budgets)
    max_gap = budget_gaps.max()
    if max_gap <= 10:
        print(f"  ✓ 预算点密集（最大间隔 {max_gap}）")
    else:
        issues.append(f"预算点较稀疏，建议在大间隔处添加点")

    if issues:
        print(f"  ")
        print(f"  改进建议:")
        for issue in issues:
            print(f"    • {issue}")

    # 5. 检查传感器类型多样性
    print("\n[5] 传感器池多样性检查")
    sensor_types = cfg.sensors.types
    type_mix = cfg.sensors.type_mix

    print(f"  传感器类型数: {len(sensor_types)}")
    for i, (stype, mix) in enumerate(zip(sensor_types, type_mix)):
        expected_count = int(pool_size * mix)
        print(f"    {stype.name}: {mix*100:.0f}% ≈ {expected_count} 个")
        print(f"      noise_std={stype.noise_std:.3f}, cost=£{stype.cost_gbp}")

    # 计算成本范围
    costs = [s.cost_gbp for s in sensor_types]
    cost_ratio = max(costs) / min(costs)
    print(f"  ")
    print(f"  成本范围: £{min(costs)} - £{max(costs)} (比例 {cost_ratio:.1f}:1)")

    if cost_ratio >= 2:
        print(f"  ✓ 成本多样性足够")
    else:
        print(f"  ⚠️  成本差距较小，考虑增加高精度传感器成本")

    # 6. EVI 配置检查
    print("\n[6] EVI 配置检查")
    if hasattr(cfg.selection, 'greedy_evi'):
        evi_cfg = cfg.selection.greedy_evi
        print(f"  n_y_samples: {evi_cfg.get('n_y_samples', 0)}")
        print(f"  mi_prescreen: {evi_cfg.get('mi_prescreen', False)}")
        print(f"  keep_fraction: {evi_cfg.get('keep_fraction', 0.25)}")
        print(f"  must_budgets: {evi_cfg.get('must_budgets', [])}")

        if evi_cfg.get('mi_prescreen', False):
            print(f"  ✓ MI 预筛选已启用（推荐）")
        else:
            print(f"  ⚠️  建议启用 mi_prescreen=true 以加速")

        if evi_cfg.get('n_y_samples', 0) == 0:
            print(f"  ✓ 使用协方差模式（最快）")
        else:
            print(f"  ⚡ 使用 MC 采样模式（较慢但更精确）")

    # 7. 总结和建议
    print("\n" + "=" * 70)
    print("  总体评估")
    print("=" * 70)

    score = 0
    max_score = 5

    if cv >= 0.20:
        score += 1
        print(f"  ✓ 先验异质性: 良好")
    else:
        print(f"  ⚠️  先验异质性: 偏低")

    if 40 <= percentile <= 60:
        score += 1
        print(f"  ✓ 决策阈值: 理想")
    else:
        print(f"  ⚠️  决策阈值: 偏离中位数")

    if ratio_fn_fp >= 15:
        score += 1
        print(f"  ✓ 损失不对称性: 强")
    else:
        print(f"  ⚠️  损失不对称性: 偏弱")

    if min(budgets) <= 5 and max(budgets) / pool_size <= 0.15:
        score += 1
        print(f"  ✓ 预算范围: 合理")
    else:
        print(f"  ⚠️  预算范围: 需优化")

    if hasattr(cfg.selection, 'greedy_evi') and cfg.selection.greedy_evi.get('mi_prescreen', False):
        score += 1
        print(f"  ✓ EVI 优化: 已启用")
    else:
        print(f"  ⚠️  EVI 优化: 未充分配置")

    print(f"  ")
    print(f"  配置评分: {score}/{max_score}")

    if score >= 4:
        print(f"  ")
        print(f"  🎉 配置优秀！应该能清晰展示 EVI 优势")
    elif score >= 3:
        print(f"  ")
        print(f"  ⚡ 配置可用，但有改进空间（见上述建议）")
    else:
        print(f"  ")
        print(f"  ⚠️  配置需要较大改进以体现算法差异")

    print("=" * 70)

    return score


if __name__ == "__main__":
    import sys

    # 可以传入配置文件路径
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    print(f"验证配置文件: {config_path}\n")
    score = validate_config(config_path)

    sys.exit(0 if score >= 3 else 1)