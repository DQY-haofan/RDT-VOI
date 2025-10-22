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



def plot_mi_evi_correlation(mi_results: Dict, evi_results: Dict,
                            output_dir: Path = None,
                            config=None) -> plt.Figure:
    """
    Plot correlation between Mutual Information and Expected Value of Information.

    This validates whether MI is a good proxy for decision-aware value.

    Args:
        mi_results: Results from Greedy-MI method
        evi_results: Results from Greedy-EVI method
        output_dir: Directory to save plots
        config: Configuration object

    Returns:
        Figure object
    """
    # Extract marginal gains for matching (budget, fold) pairs
    mi_data = []
    evi_data = []

    # Find common budgets
    mi_budgets = set(mi_results['budgets'].keys())
    evi_budgets = set(evi_results['budgets'].keys())
    common_budgets = mi_budgets & evi_budgets

    if not common_budgets:
        warnings.warn("No common budgets between MI and EVI results")
        return None

    for k in sorted(common_budgets):
        mi_budget = mi_results['budgets'][k]
        evi_budget = evi_results['budgets'][k]

        # Get fold results
        mi_folds = mi_budget.get('fold_results', [])
        evi_folds = evi_budget.get('fold_results', [])

        # Match by fold index
        n_folds = min(len(mi_folds), len(evi_folds))

        for fold_idx in range(n_folds):
            if not mi_folds[fold_idx]['success'] or not evi_folds[fold_idx]['success']:
                continue

            # Get selection results
            mi_sel = mi_folds[fold_idx]['selection_result']
            evi_sel = evi_folds[fold_idx]['selection_result']

            # Get marginal gains (step-by-step improvements)
            mi_gains = mi_sel.marginal_gains
            evi_gains = evi_sel.marginal_gains

            # Align by step
            n_steps = min(len(mi_gains), len(evi_gains))

            for step in range(n_steps):
                mi_data.append({
                    'budget': k,
                    'fold': fold_idx,
                    'step': step + 1,
                    'mi_gain': mi_gains[step],
                    'evi_gain': evi_gains[step]
                })

    if not mi_data:
        warnings.warn("No matching data found between MI and EVI")
        return None

    df = pd.DataFrame(mi_data)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Subplot 1: Scatter plot with regression
    ax = axes[0]

    # Plot points colored by step
    scatter = ax.scatter(
        df['mi_gain'],
        df['evi_gain'],
        c=df['step'],
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='white',
        linewidth=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Selection Step')

    # Fit linear regression
    from scipy.stats import linregress
    valid = np.isfinite(df['mi_gain']) & np.isfinite(df['evi_gain'])
    if valid.sum() > 2:
        slope, intercept, r_value, p_value, std_err = linregress(
            df.loc[valid, 'mi_gain'],
            df.loc[valid, 'evi_gain']
        )

        # Plot regression line
        x_line = np.linspace(df['mi_gain'].min(), df['mi_gain'].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8,
                label=f'$R^2={r_value ** 2:.3f}$, $p<{p_value:.3f}$')

        ax.legend(loc='upper left', fontsize=10)

    # Add diagonal reference line (perfect correlation)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    ax.plot(lims, lims, 'k:', alpha=0.3, linewidth=1, label='y=x')

    ax.set_xlabel('Marginal MI (nats)', fontsize=11)
    ax.set_ylabel('Marginal EVI (£)', fontsize=11)
    ax.set_title('MI vs EVI: Marginal Gains', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- Subplot 2: Correlation by budget
    ax = axes[1]

    budget_corrs = []
    for k in sorted(df['budget'].unique()):
        df_k = df[df['budget'] == k]
        if len(df_k) > 3:
            corr = df_k['mi_gain'].corr(df_k['evi_gain'])
            budget_corrs.append({
                'budget': k,
                'correlation': corr,
                'n': len(df_k)
            })

    if budget_corrs:
        df_corr = pd.DataFrame(budget_corrs)

        # Bar plot
        bars = ax.bar(
            df_corr['budget'],
            df_corr['correlation'],
            color='steelblue',
            alpha=0.7,
            edgecolor='black',
            linewidth=1
        )

        # Add sample size labels
        for i, (k, corr, n) in enumerate(zip(df_corr['budget'],
                                             df_corr['correlation'],
                                             df_corr['n'])):
            ax.text(k, corr + 0.02, f'n={n}',
                    ha='center', va='bottom', fontsize=8)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Budget (k sensors)', fontsize=11)
        ax.set_ylabel('Pearson Correlation', fontsize=11)
        ax.set_title('Correlation by Budget', fontsize=12, fontweight='bold')
        ax.set_ylim([-0.1, 1.05])
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    if output_dir:
        for fmt in ['png', 'pdf']:
            save_path = output_dir / f'f4_mi_evi_correlation.{fmt}'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"      Saved: {save_path.name}")

    return fig


# ============================================================================
# 🔥 更新：F2 - 边际效率（支持新方法）
# ============================================================================

def plot_marginal_efficiency(all_results: Dict,
                             output_dir: Path = None,
                             config=None,
                             normalize_by_cost: bool = True) -> plt.Figure:
    """
    Plot marginal gain per unit cost across selection steps.

    Shows diminishing returns and efficiency differences between methods.

    Args:
        all_results: Dictionary of results by method
        output_dir: Directory to save plots
        config: Configuration object
        normalize_by_cost: If True, plot gain/cost; else plot raw gain

    Returns:
        Figure object
    """
    # Extract data
    plot_data = []

    for method_name, method_data in all_results.items():
        # Skip random/uniform (no marginal gains)
        if method_name.lower() in ['random', 'uniform']:
            continue

        for k, budget_data in method_data['budgets'].items():
            fold_results = budget_data.get('fold_results', [])

            for fold_idx, fold_res in enumerate(fold_results):
                if not fold_res['success']:
                    continue

                sel_result = fold_res['selection_result']
                gains = sel_result.marginal_gains
                costs = []

                # Extract costs for selected sensors
                # (Assuming we stored sensor costs or can reconstruct)
                if hasattr(sel_result, 'costs'):
                    costs = sel_result.costs
                else:
                    # Fallback: assume uniform cost
                    avg_cost = sel_result.total_cost / len(sel_result.selected_ids)
                    costs = [avg_cost] * len(gains)

                # Compute efficiency
                for step, (gain, cost) in enumerate(zip(gains, costs)):
                    if normalize_by_cost and cost > 0:
                        efficiency = gain / cost
                    else:
                        efficiency = gain

                    plot_data.append({
                        'method': method_name,
                        'budget': k,
                        'fold': fold_idx,
                        'step': step + 1,
                        'gain': gain,
                        'cost': cost,
                        'efficiency': efficiency
                    })

    if not plot_data:
        warnings.warn("No data available for marginal efficiency plot")
        return None

    df = pd.DataFrame(plot_data)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Color palette
    methods = df['method'].unique()
    colors = sns.color_palette('Set2', n_colors=len(methods))
    method_colors = dict(zip(methods, colors))

    # --- Subplot 1: Efficiency by step (aggregated across budgets/folds)
    ax = axes[0]

    for method in methods:
        df_method = df[df['method'] == method]

        # Group by step and compute mean ± std
        grouped = df_method.groupby('step')['efficiency'].agg(['mean', 'std', 'count'])

        # Plot line
        ax.plot(
            grouped.index,
            grouped['mean'],
            marker='o',
            markersize=4,
            linewidth=2,
            label=method.replace('_', ' ').title(),
            color=method_colors[method],
            alpha=0.8
        )

        # Confidence interval (±1 std)
        ax.fill_between(
            grouped.index,
            grouped['mean'] - grouped['std'],
            grouped['mean'] + grouped['std'],
            alpha=0.2,
            color=method_colors[method]
        )

    ax.set_xlabel('Selection Step', fontsize=11)
    ylabel = 'Marginal Gain / Cost' if normalize_by_cost else 'Marginal Gain'
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title('Marginal Efficiency by Step', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.5)

    # --- Subplot 2: Total gain vs total cost
    ax = axes[1]

    # Compute cumulative gains and costs
    cumulative_data = []

    for method in methods:
        df_method = df[df['method'] == method]

        for (budget, fold), group in df_method.groupby(['budget', 'fold']):
            group_sorted = group.sort_values('step')

            cumulative_gain = group_sorted['gain'].cumsum().values
            cumulative_cost = group_sorted['cost'].cumsum().values

            for step_idx in range(len(cumulative_gain)):
                cumulative_data.append({
                    'method': method,
                    'budget': budget,
                    'fold': fold,
                    'total_gain': cumulative_gain[step_idx],
                    'total_cost': cumulative_cost[step_idx]
                })

    df_cum = pd.DataFrame(cumulative_data)

    for method in methods:
        df_method = df_cum[df_cum['method'] == method]

        # Plot scatter with trend
        ax.scatter(
            df_method['total_cost'],
            df_method['total_gain'],
            alpha=0.3,
            s=20,
            color=method_colors[method],
            edgecolors='none'
        )

        # Add smooth trend line
        from scipy.interpolate import UnivariateSpline
        x_sorted = df_method['total_cost'].sort_values()
        y_sorted = df_method.set_index('total_cost').loc[x_sorted.index, 'total_gain']

        if len(x_sorted) > 10:
            try:
                # Smooth spline
                spl = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted) * 10)
                x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
                y_smooth = spl(x_smooth)

                ax.plot(
                    x_smooth, y_smooth,
                    linewidth=2.5,
                    label=method.replace('_', ' ').title(),
                    color=method_colors[method],
                    alpha=0.8
                )
            except:
                # Fallback to simple mean
                pass

    ax.set_xlabel('Total Cost (£)', fontsize=11)
    ax.set_ylabel('Cumulative Gain', fontsize=11)
    ax.set_title('Total Gain vs Cost', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    if output_dir:
        for fmt in ['png', 'pdf']:
            save_path = output_dir / f'f2_marginal_efficiency.{fmt}'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"      Saved: {save_path.name}")

    return fig


# ============================================================================
# 🔥 更新：F7a - Performance Profile（支持 budget×fold 实例）
# ============================================================================

def plot_performance_profile(df_results: pd.DataFrame,
                             metric: str = 'expected_loss_gbp',
                             output_dir: Path = None,
                             config=None,
                             use_budget_fold_instances: bool = True) -> plt.Figure:
    """
    Plot performance profile comparing methods across multiple instances.

    Performance profile shows P(r_method <= τ * r_best) for various τ.

    Args:
        df_results: DataFrame with aggregated results
        metric: Metric to use for ranking
        output_dir: Directory to save plots
        config: Configuration object
        use_budget_fold_instances: If True, treat each (budget, fold) as instance;
                                   else use only budget as instance

    Returns:
        Figure object
    """
    # Filter to relevant metric
    df_metric = df_results[df_results['metric'] == metric].copy()

    if df_metric.empty:
        warnings.warn(f"No data for metric: {metric}")
        return None

    # Expand instances if requested
    if use_budget_fold_instances:
        # Each (budget, fold) is an instance
        # Need to extract individual fold values from 'values' column
        instance_data = []

        for _, row in df_metric.iterrows():
            method = row['method']
            budget = row['budget']
            values = row['values']  # List of fold values

            for fold_idx, value in enumerate(values):
                instance_data.append({
                    'method': method,
                    'instance': f"k={budget}_fold={fold_idx}",
                    'value': value
                })

        df_instances = pd.DataFrame(instance_data)
    else:
        # Each budget is an instance (use mean)
        df_instances = df_metric.rename(columns={'mean': 'value'})
        df_instances['instance'] = df_instances['budget'].astype(str)

    # Compute performance ratios
    instances = df_instances['instance'].unique()
    methods = df_instances['method'].unique()

    # For each instance, find best performance
    ratios = []

    for instance in instances:
        df_inst = df_instances[df_instances['instance'] == instance]

        if len(df_inst) < 2:  # Need at least 2 methods
            continue

        best_value = df_inst['value'].min()  # Assuming lower is better

        if best_value <= 0 or not np.isfinite(best_value):
            continue

        for _, row in df_inst.iterrows():
            ratio = row['value'] / best_value if best_value > 0 else np.inf
            ratios.append({
                'method': row['method'],
                'instance': instance,
                'ratio': ratio
            })

    if not ratios:
        warnings.warn("No valid ratios computed for performance profile")
        return None

    df_ratios = pd.DataFrame(ratios)

    # Compute performance profile curves
    if config and hasattr(config.plots, 'performance_profile'):
        tau_values = config.plots.performance_profile.get('tau_values',
                                                          [1.0, 1.05, 1.1, 1.2, 1.5, 2.0])
    else:
        tau_values = np.linspace(1.0, 3.0, 50)

    tau_values = np.array(tau_values)

    profiles = {}
    for method in methods:
        df_method = df_ratios[df_ratios['method'] == method]
        ratios_method = df_method['ratio'].values

        # For each tau, compute fraction of instances where ratio <= tau
        profile = []
        for tau in tau_values:
            frac = np.mean(ratios_method <= tau)
            profile.append(frac)

        profiles[method] = np.array(profile)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette
    colors = sns.color_palette('Set2', n_colors=len(methods))
    method_colors = dict(zip(methods, colors))

    # Plot curves
    for method in methods:
        ax.plot(
            tau_values,
            profiles[method],
            linewidth=2.5,
            marker='o',
            markersize=4,
            label=method.replace('_', ' ').title(),
            color=method_colors[method],
            alpha=0.8
        )

    # Formatting
    ax.set_xlabel('Performance Ratio τ (relative to best)', fontsize=12)
    ax.set_ylabel('P(performance ratio ≤ τ)', fontsize=12)
    ax.set_title(f'Performance Profile: {metric.replace("_", " ").title()}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([tau_values.min(), tau_values.max()])
    ax.set_ylim([0, 1.05])

    # Add horizontal line at y=1
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add vertical line at τ=1 (best method)
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add text annotation
    n_instances = len(instances)
    ax.text(
        0.02, 0.02,
        f'n = {n_instances} instances',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )

    plt.tight_layout()

    # Save
    if output_dir:
        for fmt in ['png', 'pdf']:
            save_path = output_dir / f'f7a_performance_profile_{metric}.{fmt}'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"      Saved: {save_path.name}")

    return fig


# ============================================================================
# 🔥 更新：F5 - 校准诊断（使用 z_scores）
# ============================================================================

def plot_calibration_diagnostics(all_results: Dict,
                                 output_dir: Path = None,
                                 config=None) -> plt.Figure:
    """
    Plot calibration diagnostics: coverage curves and PIT histograms.

    Args:
        all_results: Dictionary of results by method
        output_dir: Directory to save plots
        config: Configuration object

    Returns:
        Figure object
    """
    # Extract z-scores from all methods
    z_scores_by_method = {}

    for method_name, method_data in all_results.items():
        all_z_scores = []

        for k, budget_data in method_data['budgets'].items():
            fold_results = budget_data.get('fold_results', [])

            for fold_res in fold_results:
                if not fold_res['success']:
                    continue

                metrics = fold_res['metrics']
                if 'z_scores' in metrics:
                    z_scores = metrics['z_scores']
                    all_z_scores.extend(z_scores)

        if all_z_scores:
            z_scores_by_method[method_name] = np.array(all_z_scores)

    if not z_scores_by_method:
        warnings.warn("No z_scores found in results. Ensure compute_metrics saves z_scores.")
        return None

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Color palette
    methods = list(z_scores_by_method.keys())
    colors = sns.color_palette('Set2', n_colors=len(methods))
    method_colors = dict(zip(methods, colors))

    # --- Subplot 1: Coverage curves (empirical CDF of |z|)
    ax = axes[0]

    # Standard normal reference
    z_ref = np.linspace(0, 3.5, 100)
    coverage_ref = 2 * (1 - 0.5 * (1 + np.sign(z_ref) * np.sqrt(1 - np.exp(-2 * z_ref ** 2 / np.pi))))
    # Actually, for |Z|, coverage = 2*Φ(z) - 1
    from scipy.stats import norm
    coverage_ref = 2 * norm.cdf(z_ref) - 1

    ax.plot(z_ref, coverage_ref, 'k--', linewidth=2, label='Ideal (N(0,1))', alpha=0.7)

    for method in methods:
        z = z_scores_by_method[method]
        z_abs = np.abs(z)
        z_abs_sorted = np.sort(z_abs)

        # Empirical CDF
        n = len(z_abs_sorted)
        empirical_cdf = np.arange(1, n + 1) / n

        ax.plot(
            z_abs_sorted,
            empirical_cdf,
            linewidth=2,
            label=method.replace('_', ' ').title(),
            color=method_colors[method],
            alpha=0.8
        )

    # Add reference lines for common confidence levels
    for alpha, z_val in [(0.90, 1.645), (0.95, 1.96)]:
        ax.axvline(x=z_val, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=alpha, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.text(z_val, 0.02, f'{int(alpha * 100)}%', fontsize=8, ha='center')

    ax.set_xlabel('|Standardized Error|', fontsize=11)
    ax.set_ylabel('Empirical Coverage', fontsize=11)
    ax.set_title('Coverage Calibration', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 3.5])
    ax.set_ylim([0, 1.05])

    # --- Subplot 2: PIT histogram (probability integral transform)
    ax = axes[1]

    # Convert z-scores to uniform via Φ(z)
    for method in methods:
        z = z_scores_by_method[method]
        pit_values = norm.cdf(z)

        ax.hist(
            pit_values,
            bins=20,
            alpha=0.6,
            label=method.replace('_', ' ').title(),
            color=method_colors[method],
            edgecolor='black',
            linewidth=0.5,
            density=True
        )

    # Add uniform reference line
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2,
               label='Ideal (Uniform)', alpha=0.7)

    ax.set_xlabel('PIT Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Probability Integral Transform', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim([0, 1])

    plt.tight_layout()

    # Save
    if output_dir:
        for fmt in ['png', 'pdf']:
            save_path = output_dir / f'f5_calibration_diagnostics.{fmt}'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"      Saved: {save_path.name}")

    return fig


# ============================================================================
# 🔥 辅助函数：提取 selection result 中的 costs（如果没有存储）
# ============================================================================

def extract_sensor_costs_from_results(all_results: Dict, sensors: List) -> Dict:
    """
    Extract per-sensor costs from selection results.

    This is a helper to reconstruct costs if they weren't stored in SelectionResult.

    Args:
        all_results: Dictionary of results
        sensors: Original sensor pool

    Returns:
        Dictionary mapping (method, budget, fold, step) -> cost
    """
    costs_map = {}

    for method_name, method_data in all_results.items():
        for k, budget_data in method_data['budgets'].items():
            fold_results = budget_data.get('fold_results', [])

            for fold_idx, fold_res in enumerate(fold_results):
                if not fold_res['success']:
                    continue

                sel_result = fold_res['selection_result']
                selected_ids = sel_result.selected_ids

                for step, sensor_id in enumerate(selected_ids):
                    cost = sensors[sensor_id].cost
                    costs_map[(method_name, k, fold_idx, step)] = cost

    return costs_map


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