"""
RDT-VoI 配置诊断工具 - 预测方法差异能力

使用方法：
    python diagnose_config.py
    python diagnose_config.py --config my_config.yaml
    python diagnose_config.py --quick  # 快速模式
    python diagnose_config.py --save-plots  # 保存诊断图

功能：
    ✅ 先验空间异质性检查
    ✅ DDI 目标达成验证
    ✅ 方法差异预测
    ✅ 传感器池质量评估
    ✅ 生成诊断报告和建议
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from geometry import build_grid2d_geometry
from spatial_field import build_prior, build_prior_with_ddi, compute_ddi
from sensors import generate_sensor_pool
from inference import SparseFactor, compute_posterior_variance_diagonal


# ============================================================================
# 诊断结果数据结构
# ============================================================================

@dataclass
class DiagnosticResult:
    """诊断结果容器"""
    # 先验质量
    prior_variance_cv: float
    prior_mean_range: Tuple[float, float]
    prior_std_range: Tuple[float, float]
    spatial_correlation_length: float

    # DDI 指标
    target_ddi: float
    actual_ddi: float
    ddi_error: float
    near_threshold_pixels: int

    # 传感器池
    n_sensors: int
    sensor_type_diversity: int
    cost_range: Tuple[float, float]
    noise_range: Tuple[float, float]
    pool_coverage: float

    # 方法差异预测
    predicted_mi_evi_gap: float
    method_differentiation_score: float
    roi_feasibility: str

    # 整体评估
    overall_grade: str
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]

    def __str__(self):
        """格式化输出"""
        lines = [
            "\n" + "=" * 70,
            "  🔬 RDT-VoI CONFIGURATION DIAGNOSTIC REPORT",
            "=" * 70,
            "",
            "📊 PRIOR QUALITY",
            f"  Spatial heterogeneity (CV):  {self.prior_variance_cv:.2%}",
            self._grade_cv(self.prior_variance_cv),
            f"  Mean range:                  [{self.prior_mean_range[0]:.2f}, {self.prior_mean_range[1]:.2f}]",
            f"  Std range:                   [{self.prior_std_range[0]:.3f}, {self.prior_std_range[1]:.3f}]",
            f"  Correlation length:          {self.spatial_correlation_length:.1f}m",
            "",
            "🎯 DECISION DIFFICULTY INDEX (DDI)",
            f"  Target DDI:                  {self.target_ddi:.1%}",
            f"  Actual DDI:                  {self.actual_ddi:.1%}",
            f"  Error:                       {self.ddi_error:.1%}",
            self._grade_ddi(self.ddi_error),
            f"  Near-threshold pixels:       {self.near_threshold_pixels}",
            "",
            "🎛️ SENSOR POOL",
            f"  Candidates:                  {self.n_sensors}",
            f"  Type diversity:              {self.sensor_type_diversity} types",
            f"  Cost range:                  £{self.cost_range[0]:.0f} - £{self.cost_range[1]:.0f}",
            f"  Noise range:                 {self.noise_range[0]:.3f} - {self.noise_range[1]:.3f}",
            f"  Domain coverage:             {self.pool_coverage:.1%}",
            self._grade_sensor_pool(),
            "",
            "📈 METHOD DIFFERENTIATION PREDICTION",
            f"  Predicted MI-EVI gap:        {self.predicted_mi_evi_gap:.1%}",
            self._grade_method_gap(self.predicted_mi_evi_gap),
            f"  Differentiation score:       {self.method_differentiation_score:.2f}/10",
            f"  ROI feasibility:             {self.roi_feasibility}",
            "",
            f"🎓 OVERALL GRADE: {self.overall_grade}",
            "",
        ]

        if self.critical_issues:
            lines.extend([
                "❌ CRITICAL ISSUES:",
                *[f"  • {issue}" for issue in self.critical_issues],
                ""
            ])

        if self.warnings:
            lines.extend([
                "⚠️  WARNINGS:",
                *[f"  • {warn}" for warn in self.warnings],
                ""
            ])

        if self.recommendations:
            lines.extend([
                "💡 RECOMMENDATIONS:",
                *[f"  • {rec}" for rec in self.recommendations],
                ""
            ])

        lines.append("=" * 70)
        return "\n".join(lines)

    def _grade_cv(self, cv: float) -> str:
        if cv >= 0.20:
            return "  ✅ EXCELLENT - Strong spatial heterogeneity"
        elif cv >= 0.10:
            return "  ✅ GOOD - Adequate heterogeneity"
        elif cv >= 0.05:
            return "  ⚠️  FAIR - Weak heterogeneity, methods may overlap"
        else:
            return "  ❌ POOR - Insufficient heterogeneity!"

    def _grade_ddi(self, error: float) -> str:
        if error <= 0.05:
            return "  ✅ EXCELLENT - DDI target achieved"
        elif error <= 0.10:
            return "  ✅ GOOD - Close to target"
        elif error <= 0.15:
            return "  ⚠️  FAIR - Noticeable deviation"
        else:
            return "  ❌ POOR - DDI target missed!"

    def _grade_sensor_pool(self) -> str:
        if self.sensor_type_diversity >= 5 and self.pool_coverage >= 0.15:
            return "  ✅ GOOD - Diverse and adequate coverage"
        elif self.sensor_type_diversity >= 3 and self.pool_coverage >= 0.10:
            return "  ⚠️  FAIR - Limited diversity or coverage"
        else:
            return "  ❌ POOR - Insufficient pool quality"

    def _grade_method_gap(self, gap: float) -> str:
        if gap >= 0.10:
            return "  ✅ EXCELLENT - Methods will differentiate clearly"
        elif gap >= 0.05:
            return "  ✅ GOOD - Noticeable method differences"
        elif gap >= 0.02:
            return "  ⚠️  FAIR - Small differences, may need large budgets"
        else:
            return "  ❌ POOR - Methods likely indistinguishable!"


# ============================================================================
# 核心诊断函数
# ============================================================================

def diagnose_prior_quality(geom, Q_pr, mu_pr, config) -> Dict:
    """诊断先验质量"""
    print("\n[1/5] Diagnosing prior quality...")

    n = geom.n
    factor = SparseFactor(Q_pr)

    # 采样计算方差
    sample_size = min(200, n)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    sample_vars = compute_posterior_variance_diagonal(factor, sample_idx)
    sample_stds = np.sqrt(np.maximum(sample_vars, 1e-12))

    # 方差异质性（关键指标）
    variance_cv = sample_stds.std() / sample_stds.mean()

    # 均值范围
    mean_range = (mu_pr.min(), mu_pr.max())
    std_range = (sample_stds.min(), sample_stds.max())

    # 空间相关长度
    corr_length = np.sqrt(8 * config.prior.nu) / config.prior.kappa

    print(f"  Prior variance CV: {variance_cv:.2%}")
    print(f"  Mean range: [{mean_range[0]:.2f}, {mean_range[1]:.2f}]")
    print(f"  Std range: [{std_range[0]:.3f}, {std_range[1]:.3f}]")

    return {
        'variance_cv': variance_cv,
        'mean_range': mean_range,
        'std_range': std_range,
        'correlation_length': corr_length,
        'sample_stds': sample_stds,
        'sample_idx': sample_idx
    }


def diagnose_ddi(mu_pr, sigma_pr, tau, target_ddi, sample_idx=None) -> Dict:
    """诊断 DDI 达成情况"""
    print("\n[2/5] Diagnosing DDI...")

    if sample_idx is not None:
        mu = mu_pr[sample_idx]
        sigma = sigma_pr
    else:
        mu = mu_pr
        # 估算全域 sigma（简化）
        sigma = np.full_like(mu, sigma_pr.mean())

    # 计算实际 DDI
    gaps = np.abs(mu - tau)
    d = gaps / np.maximum(sigma, 1e-12)

    # 使用 target_ddi 的分位数作为 epsilon
    if target_ddi > 0 and target_ddi < 1:
        epsilon = np.quantile(d, target_ddi)
        epsilon = np.clip(epsilon, 0.1, 5.0)
    else:
        epsilon = 1.0

    near_threshold = d <= epsilon
    actual_ddi = near_threshold.mean()

    ddi_error = abs(actual_ddi - target_ddi)

    print(f"  Target DDI: {target_ddi:.1%}")
    print(f"  Actual DDI: {actual_ddi:.1%}")
    print(f"  Error: {ddi_error:.1%}")
    print(f"  Epsilon used: {epsilon:.3f}σ")

    return {
        'target_ddi': target_ddi,
        'actual_ddi': actual_ddi,
        'ddi_error': ddi_error,
        'near_threshold_count': int(near_threshold.sum()),
        'epsilon': epsilon
    }


def diagnose_sensor_pool(sensors, geom) -> Dict:
    """诊断传感器池质量"""
    print("\n[3/5] Diagnosing sensor pool...")

    n_sensors = len(sensors)

    # 类型多样性
    types = set(s.type_name for s in sensors)
    n_types = len(types)

    # 成本和噪声范围
    costs = [s.cost for s in sensors]
    noises = [np.sqrt(s.noise_var) for s in sensors]

    cost_range = (min(costs), max(costs))
    noise_range = (min(noises), max(noises))

    # 覆盖率
    pool_coverage = n_sensors / geom.n

    print(f"  Sensor count: {n_sensors}")
    print(f"  Type diversity: {n_types}")
    print(f"  Cost range: £{cost_range[0]:.0f} - £{cost_range[1]:.0f}")
    print(f"  Noise range: {noise_range[0]:.3f} - {noise_range[1]:.3f}")
    print(f"  Coverage: {pool_coverage:.1%}")

    return {
        'n_sensors': n_sensors,
        'n_types': n_types,
        'cost_range': cost_range,
        'noise_range': noise_range,
        'pool_coverage': pool_coverage,
        'type_counts': {t: sum(1 for s in sensors if s.type_name == t) for t in types}
    }


def predict_method_differentiation(prior_cv: float, ddi: float,
                                   pool_coverage: float, n_types: int) -> Dict:
    """预测方法差异能力"""
    print("\n[4/5] Predicting method differentiation...")

    # 基于经验规则的预测模型

    # 因子1: 空间异质性 (最重要)
    if prior_cv >= 0.20:
        cv_score = 1.0
    elif prior_cv >= 0.10:
        cv_score = 0.7
    elif prior_cv >= 0.05:
        cv_score = 0.4
    else:
        cv_score = 0.1

    # 因子2: DDI (中等重要)
    if 0.20 <= ddi <= 0.35:
        ddi_score = 1.0
    elif 0.15 <= ddi <= 0.40:
        ddi_score = 0.7
    else:
        ddi_score = 0.4

    # 因子3: 传感器多样性 (次要)
    if n_types >= 5 and pool_coverage >= 0.15:
        pool_score = 1.0
    elif n_types >= 3 and pool_coverage >= 0.10:
        pool_score = 0.7
    else:
        pool_score = 0.4

    # 综合评分 (加权平均)
    weights = [0.5, 0.3, 0.2]  # CV 最重要
    differentiation_score = (
                                    weights[0] * cv_score +
                                    weights[1] * ddi_score +
                                    weights[2] * pool_score
                            ) * 10

    # 预测 MI-EVI 性能差距
    # 经验公式: gap ≈ 0.5 * cv_score * ddi_score
    predicted_gap = 0.5 * cv_score * ddi_score

    # ROI 可行性
    if predicted_gap >= 0.10 and prior_cv >= 0.10:
        roi_feasibility = "✅ HIGH - Positive ROI likely at k=5-10"
    elif predicted_gap >= 0.05 and prior_cv >= 0.05:
        roi_feasibility = "⚠️  MEDIUM - May need larger budgets (k>15)"
    else:
        roi_feasibility = "❌ LOW - ROI unlikely without parameter tuning"

    print(f"  Predicted MI-EVI gap: {predicted_gap:.1%}")
    print(f"  Differentiation score: {differentiation_score:.1f}/10")
    print(f"  ROI feasibility: {roi_feasibility}")

    return {
        'predicted_gap': predicted_gap,
        'differentiation_score': differentiation_score,
        'roi_feasibility': roi_feasibility,
        'cv_score': cv_score,
        'ddi_score': ddi_score,
        'pool_score': pool_score
    }


def generate_recommendations(results: Dict) -> Tuple[List[str], List[str], List[str]]:
    """生成问题和建议"""
    critical = []
    warnings = []
    recommendations = []

    # 检查关键问题
    if results['prior']['variance_cv'] < 0.05:
        critical.append(
            "Prior variance CV < 5% - Methods will NOT differentiate!"
        )
        recommendations.append(
            f"URGENT: Increase beta_base to {results['prior']['std_range'][1] * 0.3:.2e} "
            f"and decrease beta_hot to {results['prior']['std_range'][0] * 0.001:.2e}"
        )

    if results['ddi']['ddi_error'] > 0.15:
        critical.append(
            f"DDI error {results['ddi']['ddi_error']:.1%} > 15% - Target missed!"
        )
        if results['ddi']['actual_ddi'] < results['ddi']['target_ddi']:
            recommendations.append(
                "Add more/larger hotspots to increase near-threshold regions"
            )
        else:
            recommendations.append(
                "Reduce target_ddi or increase prior heterogeneity"
            )

    if results['prediction']['predicted_gap'] < 0.05:
        critical.append(
            "Predicted method gap < 5% - EVI advantage unclear!"
        )

    # 检查警告
    if results['prior']['variance_cv'] < 0.10:
        warnings.append(
            "Prior CV < 10% - Consider strengthening spatial heterogeneity"
        )
        recommendations.append(
            "Increase hotspot radius by 50% or add 2 more hotspots"
        )

    if results['pool']['n_types'] < 4:
        warnings.append(
            "Limited sensor type diversity may reduce cost-benefit analysis clarity"
        )

    if results['pool']['pool_coverage'] < 0.15:
        warnings.append(
            "Low pool coverage - increase pool_fraction to 0.20-0.25"
        )

    # 生成正向建议
    if not critical:
        if results['prior']['variance_cv'] >= 0.15:
            recommendations.append(
                "✅ Prior heterogeneity good - no changes needed"
            )

        if 0.20 <= results['ddi']['actual_ddi'] <= 0.35:
            recommendations.append(
                "✅ DDI in optimal range - maintain current settings"
            )

        if results['prediction']['differentiation_score'] >= 7.0:
            recommendations.append(
                "✅ Strong method differentiation predicted - proceed with experiments"
            )

    return critical, warnings, recommendations


def assign_overall_grade(differentiation_score: float, critical_count: int) -> str:
    """分配总体评级"""
    if critical_count > 0:
        return "❌ F (FAIL) - Critical issues must be fixed"
    elif differentiation_score >= 8.0:
        return "✅ A (EXCELLENT) - Ready for publication"
    elif differentiation_score >= 7.0:
        return "✅ B (GOOD) - Solid experimental setup"
    elif differentiation_score >= 5.0:
        return "⚠️  C (FAIR) - May show weak results"
    else:
        return "❌ D (POOR) - Unlikely to demonstrate advantages"


# ============================================================================
# 可视化诊断
# ============================================================================

def plot_diagnostics(geom, mu_pr, sigma_pr, tau, sensors,
                     results: Dict, output_path: Path = None):
    """生成诊断可视化"""
    print("\n[5/5] Generating diagnostic plots...")

    if geom.mode != "grid2d":
        print("  Visualization only supports grid2d")
        return

    n = geom.n
    nx = int(np.sqrt(n))
    ny = nx

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 先验均值
    ax = axes[0, 0]
    mu_map = mu_pr.reshape(nx, ny)
    im1 = ax.imshow(mu_map, cmap='RdYlGn_r', origin='lower')
    ax.contour(mu_map, levels=[tau], colors='black', linewidths=3)
    ax.set_title(f'Prior Mean (τ={tau:.2f})', fontweight='bold')
    plt.colorbar(im1, ax=ax, label='Mean')

    # 2. 先验标准差
    ax = axes[0, 1]
    sample_idx = results['prior']['sample_idx']
    sigma_map = np.zeros(n)
    sigma_map[sample_idx] = results['prior']['sample_stds']
    sigma_map = sigma_map.reshape(nx, ny)
    im2 = ax.imshow(sigma_map, cmap='viridis', origin='lower')
    ax.set_title(f'Prior Std (CV={results["prior"]["variance_cv"]:.2%})',
                 fontweight='bold')
    plt.colorbar(im2, ax=ax, label='Std σ')

    # 3. DDI 热力图
    ax = axes[0, 2]
    gaps = np.abs(mu_pr - tau)
    sigma_full = np.full(n, results['prior']['sample_stds'].mean())
    d = gaps / np.maximum(sigma_full, 1e-12)
    epsilon = results['ddi']['epsilon']
    difficulty = np.where(d <= epsilon, 1.0, np.exp(-0.5 * ((d - epsilon) / epsilon) ** 2))
    difficulty_map = difficulty.reshape(nx, ny)
    im3 = ax.imshow(difficulty_map, cmap='hot', origin='lower', vmin=0, vmax=1)
    ax.set_title(f'Decision Difficulty (DDI={results["ddi"]["actual_ddi"]:.1%})',
                 fontweight='bold')
    plt.colorbar(im3, ax=ax, label='Difficulty')

    # 4. 传感器位置和类型
    ax = axes[1, 0]
    sensor_coords = np.array([geom.coords[s.idxs[0]] for s in sensors])
    type_colors = {
        'smartphone': 'green',
        'basic_point': 'blue',
        'laser_profiler': 'red',
        'photogrammetry': 'orange',
        'vehicle_avg': 'purple',
        'inertial_profiler': 'cyan'
    }
    for sensor in sensors:
        coord = geom.coords[sensor.idxs[0]]
        color = type_colors.get(sensor.type_name, 'gray')
        ax.scatter(coord[0], coord[1], c=color, s=20, alpha=0.6)
    ax.set_title(f'Sensor Pool (n={len(sensors)})', fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # 5. 成本-噪声分布
    ax = axes[1, 1]
    costs = [s.cost for s in sensors]
    noises = [np.sqrt(s.noise_var) for s in sensors]
    types = [s.type_name for s in sensors]
    for t in set(types):
        mask = [s.type_name == t for s in sensors]
        ax.scatter(
            [c for c, m in zip(costs, mask) if m],
            [n for n, m in zip(noises, mask) if m],
            label=t, s=50, alpha=0.6
        )
    ax.set_xlabel('Cost (£)', fontweight='bold')
    ax.set_ylabel('Noise Std', fontweight='bold')
    ax.set_title('Cost-Noise Trade-off', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. 预测得分
    ax = axes[1, 2]
    categories = ['Spatial\nHeterogeneity', 'DDI\nControl', 'Sensor\nPool']
    scores = [
        results['prediction']['cv_score'] * 10,
        results['prediction']['ddi_score'] * 10,
        results['prediction']['pool_score'] * 10
    ]
    colors_bar = ['green' if s >= 7 else 'orange' if s >= 5 else 'red' for s in scores]
    bars = ax.bar(categories, scores, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.axhline(y=7, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (7+)')
    ax.axhline(y=5, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Fair (5+)')
    ax.set_ylim(0, 11)
    ax.set_ylabel('Score (0-10)', fontweight='bold')
    ax.set_title('Component Scores', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 在每个柱子上显示数值
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('RDT-VoI Configuration Diagnostic Dashboard',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved diagnostic plot: {output_path}")
    else:
        plt.savefig('diagnostic_report.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved diagnostic plot: diagnostic_report.png")

    plt.close()


# ============================================================================
# 主诊断流程
# ============================================================================

def run_full_diagnosis(config_path: str = None, quick: bool = False,
                       save_plots: bool = False) -> DiagnosticResult:
    """运行完整诊断"""
    print("\n" + "=" * 70)
    print("  🔬 RDT-VoI CONFIGURATION DIAGNOSTIC")
    print("=" * 70)

    # 加载配置
    if config_path is None:
        config_path = "baseline_config.yaml"

    print(f"\nLoading configuration: {config_path}")
    try:
        cfg = load_config(config_path)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

    rng = cfg.get_rng()

    # 构建几何
    print(f"\nBuilding domain: {cfg.geometry.nx}×{cfg.geometry.ny}")
    geom = build_grid2d_geometry(cfg.geometry.nx, cfg.geometry.ny, cfg.geometry.h)

    # 构建先验
    print("Building prior...")

    # 如果有 DDI 目标，使用 DDI 控制版本
    if hasattr(cfg.decision, 'target_ddi') and cfg.decision.target_ddi > 0:
        Q_temp, mu_temp = build_prior(geom, cfg.prior)
        tau = cfg.decision.get_threshold(mu_temp)
        Q_pr, mu_pr = build_prior_with_ddi(
            geom, cfg.prior, tau=tau,
            target_ddi=cfg.decision.target_ddi
        )
    else:
        Q_pr, mu_pr = build_prior(geom, cfg.prior)
        tau = cfg.decision.get_threshold(mu_pr)

    # 生成传感器池
    print("Generating sensor pool...")
    sensors = generate_sensor_pool(geom, cfg.sensors, rng)

    # 运行诊断
    results = {}

    # 1. 先验质量
    results['prior'] = diagnose_prior_quality(geom, Q_pr, mu_pr, cfg)

    # 2. DDI
    target_ddi = getattr(cfg.decision, 'target_ddi', 0.25)
    results['ddi'] = diagnose_ddi(
        mu_pr,
        results['prior']['sample_stds'],
        tau,
        target_ddi,
        results['prior']['sample_idx']
    )

    # 3. 传感器池
    results['pool'] = diagnose_sensor_pool(sensors, geom)

    # 4. 方法差异预测
    results['prediction'] = predict_method_differentiation(
        results['prior']['variance_cv'],
        results['ddi']['actual_ddi'],
        results['pool']['pool_coverage'],
        results['pool']['n_types']
    )

    # 5. 生成建议
    critical, warnings_list, recommendations = generate_recommendations(results)

    # 6. 总体评级
    overall_grade = assign_overall_grade(
        results['prediction']['differentiation_score'],
        len(critical)
    )

    # 创建诊断结果对象
    diagnostic = DiagnosticResult(
        prior_variance_cv=results['prior']['variance_cv'],
        prior_mean_range=results['prior']['mean_range'],
        prior_std_range=results['prior']['std_range'],
        spatial_correlation_length=results['prior']['correlation_length'],
        target_ddi=results['ddi']['target_ddi'],
        actual_ddi=results['ddi']['actual_ddi'],
        ddi_error=results['ddi']['ddi_error'],
        near_threshold_pixels=results['ddi']['near_threshold_count'],
        n_sensors=results['pool']['n_sensors'],
        sensor_type_diversity=results['pool']['n_types'],
        cost_range=results['pool']['cost_range'],
        noise_range=results['pool']['noise_range'],
        pool_coverage=results['pool']['pool_coverage'],
        predicted_mi_evi_gap=results['prediction']['predicted_gap'],
        method_differentiation_score=results['prediction']['differentiation_score'],
        roi_feasibility=results['prediction']['roi_feasibility'],
        overall_grade=overall_grade,
        critical_issues=critical,
        warnings=warnings_list,
        recommendations=recommendations
    )

    # 生成可视化
    if save_plots or not quick:
        plot_diagnostics(geom, mu_pr, results['prior']['sample_stds'],
                         tau, sensors, results)

    return diagnostic


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Diagnose RDT-VoI configuration for method differentiation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diagnose_config.py
  python diagnose_config.py --config my_config.yaml
  python diagnose_config.py --quick
  python diagnose_config.py --save-plots
        """
    )

    parser.add_argument(
        '--config', '-c', type=str, default=None,
        help='Configuration file path (default: baseline_config.yaml)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode (skip visualization)'
    )
    parser.add_argument(
        '--save-plots', action='store_true',
        help='Save diagnostic plots'
    )

    args = parser.parse_args()

    # 运行诊断
    try:
        result = run_full_diagnosis(
            config_path=args.config,
            quick=args.quick,
            save_plots=args.save_plots
        )

        # 打印报告
        print(result)

        # 退出码
        if result.overall_grade.startswith('❌'):
            sys.exit(1)  # 失败
        elif result.overall_grade.startswith('⚠️'):
            sys.exit(2)  # 警告
        else:
            sys.exit(0)  # 成功

    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()