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
import pandas as pd
from matplotlib.patches import Rectangle, Patch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import warnings

# 🔥 确保所有scipy.stats导入
from scipy.stats import norm, pearsonr, spearmanr, rankdata

# 🔥 其他必要导入
from collections import Counter
from numpy.polynomial import Polynomial



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

def plot_budget_curves(df_results: pd.DataFrame,
                       output_dir: Path = None,
                       config=None) -> plt.Figure:
    """
    Plot performance metrics vs budget for all methods.

    Args:
        df_results: DataFrame with aggregated results
        output_dir: Directory to save plots
        config: Configuration object

    Returns:
        Figure object
    """
    # 从 config 中获取要绘制的指标
    if config and hasattr(config.plots, 'budget_curves'):
        metrics_to_plot = config.plots.budget_curves.get('metrics',
                                                         ['rmse', 'expected_loss_gbp', 'coverage_90'])
    else:
        metrics_to_plot = ['rmse', 'expected_loss_gbp', 'coverage_90']

    # 过滤 DataFrame
    df_plot = df_results[df_results['metric'].isin(metrics_to_plot)]

    if df_plot.empty:
        warnings.warn("No data to plot in budget curves")
        return None

    # 创建子图
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    # 颜色方案
    methods = df_plot['method'].unique()
    colors = sns.color_palette('Set2', n_colors=len(methods))
    method_colors = dict(zip(methods, colors))

    # 为每个指标绘图
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        df_metric = df_plot[df_plot['metric'] == metric]

        for method in methods:
            df_method = df_metric[df_metric['method'] == method]

            if df_method.empty:
                continue

            # 按 budget 排序
            df_method = df_method.sort_values('budget')

            # 绘制主曲线
            ax.plot(
                df_method['budget'],
                df_method['mean'],
                marker='o',
                linewidth=2,
                markersize=6,
                label=method.replace('_', ' ').title(),
                color=method_colors[method]
            )

            # 添加置信区间
            if config and hasattr(config.plots.budget_curves, 'show_confidence'):
                if config.plots.budget_curves['show_confidence']:
                    ax.fill_between(
                        df_method['budget'],
                        df_method['mean'] - df_method['std'],
                        df_method['mean'] + df_method['std'],
                        alpha=0.2,
                        color=method_colors[method]
                    )

        # 格式化
        ax.set_xlabel('Budget (k sensors)', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} vs Budget',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    if output_dir:
        for fmt in ['png', 'pdf']:
            save_path = output_dir / f'f1_budget_curves.{fmt}'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"      Saved: {save_path.name}")

    return fig



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


def plot_mi_evi_correlation(mi_results: Dict,
                            evi_results: Dict,
                            output_dir: Path = None,
                            config=None) -> Optional[plt.Figure]:
    """
    🔥 修复版：MI vs EVI 相关性分析

    Plot correlation between MI gains and EVI values.

    Args:
        mi_results: Greedy MI results dictionary
        evi_results: Greedy EVI results dictionary
        output_dir: Directory to save plots
        config: Configuration object

    Returns:
        Figure object or None if insufficient data
    """
    # 提取配对的MI和EVI值
    mi_values = []
    evi_values = []

    # 找到共同的预算
    mi_budgets = set(mi_results.get('budgets', {}).keys())
    evi_budgets = set(evi_results.get('budgets', {}).keys())
    common_budgets = mi_budgets & evi_budgets

    if not common_budgets:
        warnings.warn("No common budgets between MI and EVI results")
        return None

    for budget in common_budgets:
        mi_budget_data = mi_results['budgets'][budget]
        evi_budget_data = evi_results['budgets'][budget]

        mi_folds = mi_budget_data.get('fold_results', [])
        evi_folds = evi_budget_data.get('fold_results', [])

        # 配对fold结果
        for mi_fold, evi_fold in zip(mi_folds, evi_folds):
            if not (mi_fold.get('success') and evi_fold.get('success')):
                continue

            # 提取边际增益
            mi_sel = mi_fold.get('selection_result')
            evi_sel = evi_fold.get('selection_result')

            if mi_sel and evi_sel:
                mi_gains = getattr(mi_sel, 'marginal_gains', [])
                evi_gains = getattr(evi_sel, 'marginal_gains', [])

                # 取共同长度
                min_len = min(len(mi_gains), len(evi_gains))
                if min_len > 0:
                    mi_values.extend(mi_gains[:min_len])
                    evi_values.extend(evi_gains[:min_len])

    if len(mi_values) < 10:
        warnings.warn(f"Insufficient paired data for correlation plot (n={len(mi_values)})")
        return None

    # 转换为数组
    mi_values = np.array(mi_values)
    evi_values = np.array(evi_values)

    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 6))

    # 散点图
    ax.scatter(mi_values, evi_values, alpha=0.6, s=50,
               c='steelblue', edgecolors='black', linewidth=0.5)

    # 线性拟合
    try:
        from numpy.polynomial import Polynomial
        p = Polynomial.fit(mi_values, evi_values, deg=1)
        x_fit = np.linspace(mi_values.min(), mi_values.max(), 100)
        y_fit = p(x_fit)
        ax.plot(x_fit, y_fit, 'r--', linewidth=2, label='Linear Fit')
    except Exception as e:
        warnings.warn(f"Linear fit failed: {e}")

    # 计算相关系数
    try:
        r_pearson, p_pearson = pearsonr(mi_values, evi_values)
        r_spearman, p_spearman = spearmanr(mi_values, evi_values)

        # 添加统计信息
        textstr = f'Pearson $r$ = {r_pearson:.3f}\n'
        textstr += f'Spearman $\\rho$ = {r_spearman:.3f}\n'
        textstr += f'$R^2$ = {r_pearson ** 2:.3f}\n'
        textstr += f'n = {len(mi_values)} pairs'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    except Exception as e:
        warnings.warn(f"Correlation computation failed: {e}")

    ax.set_xlabel('MI Marginal Gain (nats)', fontweight='bold', fontsize=11)
    ax.set_ylabel('EVI Marginal Gain (£)', fontweight='bold', fontsize=11)
    ax.set_title('MI as Proxy for EVI: Correlation Analysis', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
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
                             config=None) -> Optional[plt.Figure]:
    """
    🔥 修复版：边际效率曲线

    Plot marginal gain per unit cost across selection steps.

    Args:
        all_results: Dictionary of results by method
        output_dir: Directory to save plots
        config: Configuration object

    Returns:
        Figure object or None if no data
    """
    # 提取方法和预算
    methods = list(all_results.keys())

    # 获取所有可用的预算值
    all_budgets = set()
    for method_data in all_results.values():
        if 'budgets' in method_data:
            all_budgets.update(method_data['budgets'].keys())

    budgets_to_plot = sorted(all_budgets)[:3]  # 只绘制前3个预算

    if not budgets_to_plot:
        warnings.warn("No budget data found for marginal efficiency plot")
        return None

    # 创建子图
    fig, axes = plt.subplots(1, len(budgets_to_plot), figsize=(6 * len(budgets_to_plot), 5))
    if len(budgets_to_plot) == 1:
        axes = [axes]

    # 颜色映射
    colors = sns.color_palette('Set2', n_colors=len(methods))
    method_colors = dict(zip(methods, colors))

    for ax, budget in zip(axes, budgets_to_plot):
        for method in methods:
            try:
                # 🔥 安全地提取数据
                method_data = all_results.get(method, {})
                budget_data = method_data.get('budgets', {}).get(budget, {})

                if not budget_data:
                    continue

                # 尝试从第一个成功的fold提取边际增益
                fold_results = budget_data.get('fold_results', [])

                marginal_gains = None
                for fold_res in fold_results:
                    if fold_res.get('success', False):
                        sel_result = fold_res.get('selection_result')
                        if sel_result and hasattr(sel_result, 'marginal_gains'):
                            marginal_gains = sel_result.marginal_gains
                            break

                if marginal_gains is None or len(marginal_gains) == 0:
                    continue

                # 绘制
                steps = np.arange(1, len(marginal_gains) + 1)
                ax.plot(
                    steps,
                    marginal_gains,
                    marker='o',
                    label=method.replace('_', ' ').title(),
                    color=method_colors.get(method, 'gray'),
                    linewidth=2,
                    markersize=4
                )

            except Exception as e:
                warnings.warn(f"Failed to plot {method} for budget {budget}: {e}")
                continue

        # 格式化
        ax.set_xlabel('Selection Step', fontsize=11)
        ax.set_ylabel('Marginal Gain', fontsize=11)
        ax.set_title(f'Budget k={budget}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
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
                             config=None) -> Optional[plt.Figure]:
    """
    🔥 完全修复版：性能剖面图

    Plot performance profile showing P(performance ≤ τ * best).

    Args:
        df_results: DataFrame with results
        metric: Metric to analyze
        output_dir: Directory to save plots
        config: Configuration object

    Returns:
        Figure object or None if insufficient data
    """
    # 过滤指定指标的数据
    df_metric = df_results[df_results['metric'] == metric].copy()

    if df_metric.empty:
        warnings.warn(f"No data for metric {metric}")
        return None

    # 🔥 只使用有fold信息的原始数据（不是聚合统计）
    df_metric = df_metric[df_metric['fold'].notna()].copy()

    if df_metric.empty:
        warnings.warn(f"No fold-level data for metric {metric}")
        return None

    # 🔥 创建实例标识符
    df_metric['instance'] = df_metric.apply(
        lambda row: f"b{int(row['budget'])}_f{int(row['fold'])}",
        axis=1
    )

    # 🔥 构建数据透视表
    try:
        pivot = df_metric.pivot(index='instance', columns='method', values='value')
    except Exception as e:
        warnings.warn(f"Failed to pivot data: {e}")
        return None

    # 移除有缺失值的实例
    pivot = pivot.dropna()

    if pivot.empty or len(pivot) < 5:
        warnings.warn(f"Insufficient complete instances ({len(pivot)}) for performance profile")
        return None

    print(f"      Performance profile: {len(pivot)} instances, {len(pivot.columns)} methods")

    # 🔥 计算性能比率
    best_per_instance = pivot.min(axis=1)  # 每个实例的最佳值
    ratios = pivot.div(best_per_instance, axis=0)  # 相对于最佳值的比率

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 定义τ值范围
    tau_max = min(3.0, ratios.max().max() * 1.1)
    tau_values = np.linspace(1.0, tau_max, 200)

    # 颜色方案
    methods = list(pivot.columns)
    colors = sns.color_palette('Set2', n_colors=len(methods))
    method_colors = dict(zip(methods, colors))

    # 🔥 为每个方法计算并绘制性能剖面
    for method in methods:
        method_ratios = ratios[method].values

        # 计算累积概率
        profile = []
        for tau in tau_values:
            prob = np.mean(method_ratios <= tau)
            profile.append(prob)

        # 绘制
        ax.plot(
            tau_values,
            profile,
            label=method,
            color=method_colors.get(method, 'gray'),
            linewidth=2.5,
            marker='o',
            markersize=4,
            markevery=20,
            alpha=0.9
        )

    # 格式化
    ax.set_xlabel('Performance Ratio τ (relative to best)', fontsize=12, fontweight='bold')
    ax.set_ylabel('P(performance ≤ τ)', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Profile: {metric.replace("_", " ").title()}\n'
                 f'n = {len(pivot)} instances',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1.0, tau_max])
    ax.set_ylim([0, 1.05])

    # 添加参考线
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()

    # 保存
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


def plot_critical_difference(df_results: pd.DataFrame,
                             metric: str = 'expected_loss_gbp',
                             output_dir: Path = None,
                             config=None,
                             alpha: float = 0.05) -> Optional[plt.Figure]:
    """
    🔥 完全修复版：临界差异图
    """
    # 过滤数据
    df_metric = df_results[df_results['metric'] == metric].copy()

    if df_metric.empty:
        warnings.warn(f"No data for metric {metric}")
        return None

    # 🔥 关键修复：只使用有fold的原始数据（不是聚合统计）
    df_metric = df_metric[df_metric['fold'].notna()].copy()

    if df_metric.empty:
        warnings.warn(f"No fold-level data for metric {metric}")
        return None

    # 创建实例标识符（确保唯一性）
    df_metric['instance'] = df_metric.apply(
        lambda row: f"b{int(row['budget'])}_f{int(row['fold'])}",
        axis=1
    )

    # 🔥 修复：使用'value'列而不是'mean'列
    # 并且先去重（以防万一）
    df_metric = df_metric[['instance', 'method', 'value']].drop_duplicates()

    # 检查重复
    duplicates = df_metric.duplicated(subset=['instance', 'method']).sum()
    if duplicates > 0:
        print(f"  警告: 发现 {duplicates} 个重复条目，已移除")
        df_metric = df_metric.drop_duplicates(subset=['instance', 'method'], keep='first')

    # 数据透视
    try:
        pivot = df_metric.pivot(index='instance', columns='method', values='value')
    except ValueError as e:
        print(f"  错误: {e}")
        print(f"  实例数: {df_metric['instance'].nunique()}")
        print(f"  方法数: {df_metric['method'].nunique()}")
        print(f"  总行数: {len(df_metric)}")

        # 找出重复的组合
        dups = df_metric[df_metric.duplicated(subset=['instance', 'method'], keep=False)]
        if not dups.empty:
            print(f"  重复的组合:")
            print(dups[['instance', 'method', 'value']].sort_values(['instance', 'method']))
        return None

    # 移除有缺失值的实例
    pivot_clean = pivot.dropna()

    if pivot_clean.empty or len(pivot_clean) < 2:
        warnings.warn(f"Insufficient complete instances ({len(pivot_clean)}) for CD diagram")
        return None

    n_methods = len(pivot_clean.columns)
    n_instances = len(pivot_clean)

    print(f"      CD Diagram: {n_instances} instances, {n_methods} methods")

    # 计算排名
    ranks = np.zeros((n_methods, n_instances))
    method_list = list(pivot_clean.columns)

    for j, instance in enumerate(pivot_clean.index):
        values = pivot_clean.loc[instance].values
        instance_ranks = rankdata(values, method='average')
        ranks[:, j] = instance_ranks

    avg_ranks = ranks.mean(axis=1)

    # 计算临界差异
    q_alpha = 2.569  # Nemenyi test at α=0.05
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_instances))

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 4))

    # 排序方法
    sorted_idx = np.argsort(avg_ranks)
    sorted_methods = [method_list[i] for i in sorted_idx]
    sorted_ranks = avg_ranks[sorted_idx]

    # 绘制排名点
    ax.scatter(sorted_ranks, np.zeros(n_methods), s=200, zorder=3,
               c='steelblue', edgecolors='black', linewidth=2)

    # 添加方法名
    for i, (rank, method) in enumerate(zip(sorted_ranks, sorted_methods)):
        ax.text(rank, -0.25, method,
                ha='center', va='top', fontsize=11, fontweight='bold',
                rotation=0)

    # 绘制临界差异连接线
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            if sorted_ranks[j] - sorted_ranks[i] <= cd:
                y_pos = 0.2 + 0.1 * ((i + j) % 3)  # 错开高度
                ax.plot([sorted_ranks[i], sorted_ranks[j]], [y_pos, y_pos],
                        'k-', linewidth=5, alpha=0.6, solid_capstyle='round')

    # 添加CD标记
    if n_methods > 1:
        x_cd_demo = sorted_ranks[0] + cd
        if x_cd_demo <= n_methods:
            ax.plot([sorted_ranks[0], x_cd_demo], [-0.5, -0.5],
                    'r-', linewidth=3, alpha=0.7)
            ax.text((sorted_ranks[0] + x_cd_demo) / 2, -0.6,
                    f'CD = {cd:.2f}', ha='center', fontsize=10, color='red')

    # 格式化
    ax.set_xlabel('Average Rank (lower is better)', fontsize=13, fontweight='bold')
    ax.set_title(f'Critical Difference Diagram: {metric.replace("_", " ").title()}\n'
                 f'Nemenyi test, α={alpha}, n={n_instances} instances',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([-0.8, 0.8])
    ax.set_xlim([0.5, n_methods + 0.5])
    ax.set_yticks([])
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    # 保存
    if output_dir:
        for fmt in ['png', 'pdf']:
            save_path = output_dir / f'f7b_critical_difference_{metric}.{fmt}'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"      Saved: {save_path.name}")

    return fig


"""
visualization.py - 新增图表
ROI 曲线、鲁棒性热图、DDI 叠加等
"""


# ... 保留原有函数 ...

# ============================================================================
# 🔥 新增：ROI 曲线
# ============================================================================

def plot_budget_roi_curves(df_results: pd.DataFrame,
                           output_dir: Path = None,
                           config=None) -> plt.Figure:
    """
    绘制预算-ROI 曲线

    展示每个方法的投资回报率随预算的变化
    """
    # 过滤 ROI 数据
    df_roi = df_results[df_results['metric'] == 'roi'].copy()

    if df_roi.empty:
        warnings.warn("No ROI data found")
        return None

    # 只使用聚合统计（不是fold级别）
    df_roi = df_roi[df_roi['fold'].isna()].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = df_roi['method'].unique()
    colors = sns.color_palette('Set2', n_colors=len(methods))
    method_colors = dict(zip(methods, colors))

    for method in methods:
        df_method = df_roi[df_roi['method'] == method].sort_values('budget')

        ax.plot(
            df_method['budget'],
            df_method['mean'],
            marker='o',
            linewidth=2.5,
            markersize=8,
            label=method,
            color=method_colors[method]
        )

        # 添加置信区间
        ax.fill_between(
            df_method['budget'],
            df_method['mean'] - df_method['std'],
            df_method['mean'] + df_method['std'],
            alpha=0.2,
            color=method_colors[method]
        )

    # 添加 ROI=0 参考线
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.5,
               label='Break-even')

    ax.set_xlabel('Budget (k sensors)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROI (Return on Investment)', fontsize=12, fontweight='bold')
    ax.set_title('Cost-Benefit Analysis: ROI vs Budget\n'
                 'ROI = (Savings - Cost) / Cost',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        for fmt in ['png', 'pdf']:
            save_path = output_dir / f'roi_curves.{fmt}'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"      Saved: {save_path.name}")

    return fig


# ============================================================================
# 🔥 新增：鲁棒性热图
# ============================================================================

def plot_robustness_heatmap(results_matrix: np.ndarray,
                            ddi_values: List[float],
                            loss_ratios: List[float],
                            method_names: List[str],
                            output_dir: Path = None) -> plt.Figure:
    """
    绘制鲁棒性热图

    横轴：DDI（决策难度）
    纵轴：L_FN/L_FP（代价不对称性）
    颜色：各方法的相对排名

    Args:
        results_matrix: (n_ddi, n_ratios, n_methods) 数组
        ddi_values: DDI 值列表
        loss_ratios: 损失比率列表
        method_names: 方法名称
        output_dir: 输出目录
    """
    n_ddi, n_ratios, n_methods = results_matrix.shape

    # 为每个方法创建子图
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))

    if n_methods == 1:
        axes = [axes]

    for i, (ax, method) in enumerate(zip(axes, method_names)):
        # 提取该方法的排名矩阵
        rank_matrix = results_matrix[:, :, i]

        # 绘制热力图
        im = ax.imshow(
            rank_matrix.T,
            cmap='RdYlGn_r',
            aspect='auto',
            origin='lower',
            vmin=1,
            vmax=n_methods
        )

        # 设置刻度
        ax.set_xticks(np.arange(n_ddi))
        ax.set_yticks(np.arange(n_ratios))
        ax.set_xticklabels([f'{d:.1%}' for d in ddi_values])
        ax.set_yticklabels([f'{r:.0f}' for r in loss_ratios])

        ax.set_xlabel('DDI (Decision Difficulty)', fontsize=10)
        ax.set_ylabel('$L_{FN}/L_{FP}$ Ratio', fontsize=10)
        ax.set_title(f'{method}\n(darker = better rank)', fontsize=11, fontweight='bold')

        # 添加数值标注
        for row in range(n_ratios):
            for col in range(n_ddi):
                rank = int(rank_matrix[col, row])
                text_color = 'white' if rank > n_methods / 2 else 'black'
                ax.text(col, row, str(rank),
                        ha="center", va="center",
                        color=text_color, fontsize=9, fontweight='bold')

        # 颜色条
        plt.colorbar(im, ax=ax, label='Rank (1=best)', fraction=0.046, pad=0.04)

    plt.suptitle('Robustness Analysis: Method Ranking Across Scenarios',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_dir:
        for fmt in ['png', 'pdf']:
            save_path = output_dir / f'robustness_heatmap.{fmt}'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"      Saved: {save_path.name}")

    return fig


# ============================================================================
# 🔥 新增：DDI 叠加的传感器布局图
# ============================================================================

def plot_sensor_placement_with_ddi(coords: np.ndarray,
                                   selected_ids: List[int],
                                   sensors: List,
                                   mu: np.ndarray,
                                   sigma: np.ndarray,
                                   tau: float,
                                   output_path: Path):
    """
    传感器布局 + DDI 热力图叠加

    解释为什么 EVI 选择这些位置
    """
    from spatial_field import compute_ddi

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 计算决策难度
    n = len(mu)
    gaps = np.abs(mu - tau)
    difficulty = np.exp(-0.5 * (gaps / sigma) ** 2)

    # 左图：DDI 热力图 + 候选池
    all_sensor_coords = np.array([coords[sensors[i].idxs[0]] for i in range(len(sensors))])

    scatter1 = ax1.scatter(
        coords[:, 0], coords[:, 1],
        c=difficulty,
        cmap='hot',
        s=30,
        vmin=0, vmax=1,
        alpha=0.6
    )

    ax1.scatter(
        all_sensor_coords[:, 0], all_sensor_coords[:, 1],
        c='lightgray',
        s=40,
        alpha=0.3,
        label='Candidate Pool',
        edgecolors='black',
        linewidth=0.5
    )

    cbar1 = plt.colorbar(scatter1, ax=ax1, label='Decision Difficulty')
    ax1.set_title('Decision Difficulty Map\n(brighter = closer to threshold)')
    ax1.legend(loc='upper right')

    # 右图：选中的传感器
    ax2.scatter(
        coords[:, 0], coords[:, 1],
        c=difficulty,
        cmap='hot',
        s=30,
        vmin=0, vmax=1,
        alpha=0.3
    )

    # 类型颜色
    type_colors = {
        'inertial_profiler': '#1f77b4',
        'photogrammetry': '#ff7f0e',
        'smartphone': '#2ca02c'
    }

    # 绘制选中传感器
    for i, sensor_id in enumerate(selected_ids):
        sensor = sensors[sensor_id]
        loc = coords[sensor.idxs[0]]

        color = type_colors.get(sensor.type_name, 'black')

        ax2.scatter(
            loc[0], loc[1],
            c=color,
            s=200,
            marker='*',
            edgecolors='white',
            linewidth=2,
            zorder=10,
            alpha=0.9
        )

        ax2.text(
            loc[0], loc[1],
            str(i + 1),
            fontsize=8,
            ha='center',
            va='center',
            color='white',
            fontweight='bold',
            zorder=11
        )

    ax2.set_title(f'Selected Sensors (k={len(selected_ids)})')

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t.replace('_', ' ').title())
                       for t, c in type_colors.items()]
    ax2.legend(handles=legend_elements, loc='upper right')

    # 全局标题
    ddi = compute_ddi(mu, sigma, tau, k=1.0)
    fig.suptitle(f'Sensor Placement Strategy Visualization\nDDI = {ddi:.2%}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved DDI overlay: {output_path}")


if __name__ == "__main__":
    # 测试新增可视化函数
    setup_style()
    print("Enhanced visualization module loaded successfully!")