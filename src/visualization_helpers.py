"""
可视化辅助函数 - 生成专家建议中缺失的三张图

包括：
1. Critical Difference 图（使用fold-level数据）
2. Type Composition 堆叠图
3. Spatial Diagnostics 图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import scipy.sparse as sp


# ============================================================================
# 1. Critical Difference 图（使用fold-level数据）
# ============================================================================

def aggregate_fold_level_results(all_results) -> pd.DataFrame:
    """
    聚合fold-level结果，为Critical Difference图准备数据

    Returns:
        DataFrame with columns: method, budget, fold, expected_loss_gbp, rmse, ...
    """
    rows = []

    for method_name, method_data in all_results.items():
        budgets_data = method_data.get('budgets', {})

        for budget_key, budget_data in budgets_data.items():
            budget = int(budget_key)
            fold_results = budget_data.get('fold_results', [])

            for fold_idx, fold_res in enumerate(fold_results):
                if not fold_res or not fold_res.get('success', False):
                    continue

                metrics = fold_res.get('metrics', {})

                row = {
                    'method': method_name,
                    'budget': budget,
                    'fold': fold_idx,
                    'expected_loss_gbp': metrics.get('expected_loss_gbp', np.nan),
                    'rmse': metrics.get('rmse', np.nan),
                    'mae': metrics.get('mae', np.nan),
                    'r2': metrics.get('r2', np.nan),
                    'coverage_90': metrics.get('coverage_90', np.nan),
                    'msse': metrics.get('msse', np.nan),
                }

                rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Aggregated {len(df)} fold-level results")
    return df


def plot_critical_difference_from_folds(df: pd.DataFrame,
                                        metric: str,
                                        output_path: Path,
                                        alpha: float = 0.05):
    """
    绘制Critical Difference图（使用fold-level数据）

    Args:
        df: fold-level results DataFrame
        metric: 'expected_loss_gbp' or other metric
        output_path: 输出路径
        alpha: 显著性水平
    """
    try:
        from scipy.stats import friedmanchisquare, rankdata
        from itertools import combinations
    except ImportError:
        print("  Warning: scipy not available for CD diagram")
        return

    # 过滤有效数据
    sub = df[df[metric].notna()].copy()

    if len(sub) < 10:
        print(f"  Skipping CD diagram: insufficient data (n={len(sub)})")
        return

    # 创建实例ID（budget + fold组合）
    sub['instance'] = sub['budget'].astype(str) + '_f' + sub['fold'].astype(int).astype(str)

    # Pivot：行=实例，列=方法
    pivot = sub.pivot_table(index='instance', columns='method', values=metric, aggfunc='mean')

    methods = pivot.columns.tolist()
    n_methods = len(methods)
    n_instances = len(pivot)

    print(f"  CD diagram: {n_methods} methods × {n_instances} instances")

    if n_methods < 2 or n_instances < 5:
        print(f"  Skipping CD diagram: too few methods or instances")
        return

    # 计算排名（每个实例内排名，值越小越好）
    ranks = pivot.rank(axis=1, method='average')
    avg_ranks = ranks.mean(axis=0).sort_values()

    # Friedman检验
    try:
        stat, p_value = friedmanchisquare(*[pivot[m].dropna() for m in methods])
        print(f"  Friedman test: χ²={stat:.2f}, p={p_value:.4f}")
    except:
        p_value = np.nan

    # 计算Critical Difference
    # CD = q_α * sqrt(k(k+1) / (6N))
    # 使用Nemenyi近似：q_0.05 ≈ 2.576 (k=3), 2.850 (k=5)
    q_values = {2: 1.960, 3: 2.576, 4: 2.772, 5: 2.850, 6: 2.936, 7: 3.036, 8: 3.098}
    q_alpha = q_values.get(n_methods, 2.850)  # 默认值

    cd_value = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_instances))

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 4))

    y_pos = np.arange(n_methods)
    ax.barh(y_pos, avg_ranks.values, color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(avg_ranks.index)
    ax.set_xlabel('Average Rank (lower is better)')
    ax.set_title(f'Critical Difference Diagram - {metric}\nCD={cd_value:.2f} at α={alpha}')
    ax.axvline(avg_ranks.values[0] + cd_value, color='red', linestyle='--',
               linewidth=2, label=f'CD={cd_value:.2f}')

    # 添加横线连接不显著差异的方法
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            rank_diff = abs(avg_ranks.iloc[j] - avg_ranks.iloc[i])
            if rank_diff < cd_value:
                # 不显著差异，画线
                ax.plot([avg_ranks.iloc[i], avg_ranks.iloc[j]],
                        [i, j], 'k-', linewidth=1, alpha=0.3)

    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved Critical Difference diagram: {output_path}")


# ============================================================================
# 2. Type Composition 堆叠图
# ============================================================================

def summarize_type_composition(all_results, sensors) -> pd.DataFrame:
    """
    统计每个方法和预算下选择的传感器类型分布

    Args:
        all_results: 结果字典
        sensors: 传感器列表

    Returns:
        DataFrame with columns: method, budget, type, count
    """
    rows = []

    for method_name, method_data in all_results.items():
        budgets_data = method_data.get('budgets', {})

        for budget_key, budget_data in budgets_data.items():
            budget = int(budget_key)
            fold_results = budget_data.get('fold_results', [])

            # 统计所有fold的类型（取平均）
            type_counts = {}
            valid_folds = 0

            for fold_res in fold_results:
                if not fold_res or not fold_res.get('success', False):
                    continue

                sel_result = fold_res.get('selection_result')
                if sel_result is None:
                    continue

                valid_folds += 1
                selected_ids = sel_result.selected_ids

                for sid in selected_ids:
                    sensor_type = sensors[sid].type_name
                    type_counts[sensor_type] = type_counts.get(sensor_type, 0) + 1

            # 平均每fold的类型数量
            if valid_folds > 0:
                for sensor_type, count in type_counts.items():
                    avg_count = count / valid_folds
                    rows.append({
                        'method': method_name,
                        'budget': budget,
                        'type': sensor_type,
                        'count': avg_count
                    })

    df = pd.DataFrame(rows)
    print(f"  Summarized type composition: {len(df)} records")
    return df


def plot_type_composition(df: pd.DataFrame, output_path: Path,
                          selected_budgets: List[int] = None):
    """
    绘制传感器类型堆叠柱状图

    Args:
        df: type composition DataFrame
        output_path: 输出路径
        selected_budgets: 要显示的预算列表（None=全部）
    """
    if len(df) == 0:
        print("  No type composition data to plot")
        return

    # 过滤预算
    if selected_budgets:
        df = df[df['budget'].isin(selected_budgets)]

    if len(df) == 0:
        return

    # Pivot：行=method-budget，列=type
    df['method_budget'] = df['method'] + '\n(k=' + df['budget'].astype(str) + ')'
    pivot = df.pivot_table(index='method_budget', columns='type',
                           values='count', aggfunc='sum', fill_value=0)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))

    pivot.plot(kind='bar', stacked=True, ax=ax,
               colormap='tab10', width=0.8)

    ax.set_xlabel('Method and Budget')
    ax.set_ylabel('Average Sensor Count')
    ax.set_title('Sensor Type Composition by Method and Budget')
    ax.legend(title='Sensor Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved type composition plot: {output_path}")


# ============================================================================
# 3. Spatial Diagnostics 图
# ============================================================================

def plot_spatial_diagnostics(geom, Q_pr, selected_sensors,
                             residuals, output_path: Path):
    """
    绘制空间诊断图：先验方差热图 + 选址 + Moran's I

    Args:
        geom: 几何对象
        Q_pr: 先验精度矩阵
        selected_sensors: 选择的传感器列表（某个方法和预算）
        residuals: 残差向量 (n,)
        output_path: 输出路径
    """
    from inference import SparseFactor, compute_posterior_variance_diagonal
    from evaluation import morans_i

    n = geom.n

    # 1. 估计先验方差（Hutchinson近似）
    print("  Computing prior variance (Hutchinson)...")
    factor = SparseFactor(Q_pr)

    # 采样少量probe估计对角方差
    n_probes = 8
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((n, n_probes))
    V = factor.solve(Z)
    diag_var_pr = (Z * V).mean(axis=1)

    # 2. 计算Moran's I
    I_stat, I_pval = morans_i(residuals, geom.adjacency,
                              n_permutations=99, rng=rng)

    # 3. 绘图
    if geom.mode == "grid2d":
        nx = int(np.sqrt(n))
        ny = nx

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：先验方差热图 + 选址
        var_map = diag_var_pr.reshape(nx, ny)
        im1 = axes[0].imshow(var_map, cmap='YlOrRd', origin='lower')
        axes[0].set_title(f'Prior Variance (CV={diag_var_pr.std() / diag_var_pr.mean():.2%})')
        axes[0].set_xlabel('x (grid)')
        axes[0].set_ylabel('y (grid)')
        plt.colorbar(im1, ax=axes[0], label='Variance')

        # 标记选择的传感器
        if selected_sensors:
            for sensor in selected_sensors:
                center_idx = sensor.idxs[0]  # 取中心点
                i, j = center_idx // ny, center_idx % ny
                axes[0].plot(j, i, 'b*', markersize=10,
                             markeredgecolor='white', markeredgewidth=0.5)

        # 右图：残差热图 + Moran's I
        resid_map = residuals.reshape(nx, ny)
        im2 = axes[1].imshow(resid_map, cmap='RdBu_r', origin='lower',
                             vmin=-np.abs(residuals).max(),
                             vmax=np.abs(residuals).max())
        axes[1].set_title(f"Residuals (Moran's I={I_stat:.3f}, p={I_pval:.3f})")
        axes[1].set_xlabel('x (grid)')
        axes[1].set_ylabel('y (grid)')
        plt.colorbar(im2, ax=axes[1], label='Residual')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved spatial diagnostics: {output_path}")
    else:
        print(f"  Spatial diagnostics plot not implemented for mode={geom.mode}")


# ============================================================================
# 主入口：生成所有专家建议的图
# ============================================================================

def generate_expert_plots(all_results, sensors, geom, Q_pr,
                          output_dir: Path, config):
    """
    生成专家建议中缺失的三张图

    Args:
        all_results: 完整结果字典
        sensors: 传感器列表
        geom: 几何对象
        Q_pr: 先验精度矩阵
        output_dir: 输出目录
        config: 配置对象
    """
    print("\n" + "=" * 70)
    print("Generating Expert-Recommended Plots")
    print("=" * 70)

    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    # 1. Critical Difference 图（fold-level）
    print("\n1. Critical Difference Diagram...")
    df_folds = aggregate_fold_level_results(all_results)

    if len(df_folds) > 0:
        plot_critical_difference_from_folds(
            df_folds,
            metric='expected_loss_gbp',
            output_path=curves_dir / "f7b_critical_difference.png",
            alpha=0.05
        )

    # 2. Type Composition 堆叠图
    print("\n2. Sensor Type Composition...")
    df_types = summarize_type_composition(all_results, sensors)

    if len(df_types) > 0:
        selected_budgets = [10, 30, 50] if 50 in df_types['budget'].unique() else None
        plot_type_composition(
            df_types,
            output_path=curves_dir / "f3_type_composition.png",
            selected_budgets=selected_budgets
        )

    # 3. Spatial Diagnostics（取一个代表性case）
    # 找一个成功的fold结果
    for method_name, method_data in all_results.items():
        budgets_data = method_data.get('budgets', {})

        for budget_key, budget_data in budgets_data.items():
            fold_results = budget_data.get('fold_results', [])

            for fold_idx, fold_res in enumerate(fold_results):
                if not fold_res or not fold_res.get('success', False):
                    continue

                # 找到一个有效的case
                sel_result = fold_res.get('selection_result')

                # 🔥 关键修复：检查是否有完整的残差
                if 'mu_post' not in fold_res or 'x_true' not in fold_res:
                    continue

                mu_post = fold_res['mu_post']
                x_true = fold_res['x_true']

                # 🔥 计算完整域的残差（而不是测试集）
                if len(mu_post) == geom.n and len(x_true) == geom.n:
                    residuals = mu_post - x_true
                else:
                    # 如果只有测试集数据，跳过
                    print(f"    Skipping: incomplete posterior data")
                    continue

                if sel_result:
                    selected = [sensors[i] for i in sel_result.selected_ids]

                    try:
                        plot_spatial_diagnostics(
                            geom, Q_pr, selected, residuals,
                            output_path=curves_dir / f"f6_spatial_diagnostics_{method_name}_k{budget_key}.png"
                        )

                        # 只画一个就够了
                        print("\n" + "=" * 70)
                        return
                    except Exception as e:
                        print(f"    Failed to plot spatial diagnostics: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

    print("  Warning: No valid fold results found for spatial diagnostics")
    print("\n" + "=" * 70)


# ============================================================================
# 🔥 Effect Size Analysis (新增效应量分析)
# ============================================================================

def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算 Cliff's Delta 效应量

    Cliff's Delta 是一种非参数效应量测度，范围 [-1, 1]：
    - δ = 0: 两组无差异
    - δ = -1: y 中所有值都优于 x（对于越小越好的指标）
    - δ = +1: x 中所有值都优于 y

    解释（Cohen's 风格）：
    - |δ| < 0.147: negligible
    - 0.147 ≤ |δ| < 0.330: small
    - 0.330 ≤ |δ| < 0.474: medium
    - |δ| ≥ 0.474: large

    Args:
        x: 方法 A 的结果数组
        y: 方法 B 的结果数组

    Returns:
        delta: Cliff's Delta 值
    """
    n_x, n_y = len(x), len(y)

    if n_x == 0 or n_y == 0:
        return np.nan

    # 计算所有配对的优劣关系
    dominance = 0
    for xi in x:
        for yi in y:
            if xi < yi:  # x 更好（假设越小越好）
                dominance += 1
            elif xi > yi:  # y 更好
                dominance -= 1

    delta = dominance / (n_x * n_y)
    return delta


def interpret_cliffs_delta(delta: float) -> str:
    """解释 Cliff's Delta 的效应大小"""
    abs_delta = abs(delta)

    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.330:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


def compute_win_rate(x: np.ndarray, y: np.ndarray,
                     lower_is_better: bool = True) -> float:
    """
    计算配对胜率（方法 x 优于方法 y 的比例）

    Args:
        x, y: 两个方法的结果数组
        lower_is_better: 是否越小越好

    Returns:
        win_rate: x 胜出的比例 [0, 1]
    """
    if len(x) != len(y):
        raise ValueError("Arrays must have same length for pairwise comparison")

    if len(x) == 0:
        return np.nan

    if lower_is_better:
        wins = np.sum(x < y)
    else:
        wins = np.sum(x > y)

    return wins / len(x)


def compute_pairwise_statistics(df_folds: pd.DataFrame,
                                metric: str = 'expected_loss_gbp',
                                lower_is_better: bool = True) -> tuple:
    """
    计算所有方法间的配对统计量

    Args:
        df_folds: fold-level DataFrame with columns [method, budget, fold, metric, ...]
        metric: 要比较的指标
        lower_is_better: 指标是否越小越好

    Returns:
        delta_matrix: Cliff's Delta 矩阵 (n_methods, n_methods)
        winrate_matrix: 胜率矩阵 (n_methods, n_methods)
        summary_df: 汇总 DataFrame
    """
    methods = sorted(df_folds['method'].unique())
    n_methods = len(methods)

    delta_matrix = np.zeros((n_methods, n_methods))
    winrate_matrix = np.zeros((n_methods, n_methods))

    # 计算配对统计量
    for i, method_a in enumerate(methods):
        for j, method_b in enumerate(methods):
            if i == j:
                delta_matrix[i, j] = 0
                winrate_matrix[i, j] = 0.5
                continue

            # 获取两个方法在所有 (budget, fold) 实例上的结果
            vals_a = df_folds[df_folds['method'] == method_a][metric].values
            vals_b = df_folds[df_folds['method'] == method_b][metric].values

            # 确保长度相同（配对比较）
            min_len = min(len(vals_a), len(vals_b))
            vals_a = vals_a[:min_len]
            vals_b = vals_b[:min_len]

            # Cliff's Delta（对于越小越好的指标，负值表示 a 更好）
            if lower_is_better:
                delta = compute_cliffs_delta(vals_b, vals_a)  # 反转，让负值表示 a 更好
            else:
                delta = compute_cliffs_delta(vals_a, vals_b)

            delta_matrix[i, j] = delta

            # 胜率（a 优于 b 的比例）
            winrate = compute_win_rate(vals_a, vals_b, lower_is_better)
            winrate_matrix[i, j] = winrate

    # 创建汇总 DataFrame
    summary_rows = []
    for i, method_a in enumerate(methods):
        for j, method_b in enumerate(methods):
            if i != j:
                summary_rows.append({
                    'Method A': method_a,
                    'Method B': method_b,
                    'Cliff_Delta': delta_matrix[i, j],
                    'Effect_Size': interpret_cliffs_delta(delta_matrix[i, j]),
                    'Win_Rate': winrate_matrix[i, j],
                    'A_Better': winrate_matrix[i, j] > 0.5
                })

    summary_df = pd.DataFrame(summary_rows)

    return delta_matrix, winrate_matrix, summary_df


def plot_effect_size_heatmaps(delta_matrix: np.ndarray,
                              winrate_matrix: np.ndarray,
                              methods: list,
                              output_path: Path,
                              metric_name: str = 'Expected Loss'):
    """
    绘制效应量双热力图（Cliff's Delta + 胜率）

    Args:
        delta_matrix: Cliff's Delta 矩阵
        winrate_matrix: 胜率矩阵
        methods: 方法名称列表
        output_path: 输出路径
        metric_name: 指标名称（用于标题）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    n_methods = len(methods)

    # === 左图：Cliff's Delta ===
    im1 = ax1.imshow(delta_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(n_methods))
    ax1.set_yticks(range(n_methods))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_yticklabels(methods)
    ax1.set_title(f"Cliff's Delta Effect Size\n{metric_name}\n"
                  "(negative = row method better)", fontsize=12)
    ax1.set_xlabel('Method B (compared to)', fontsize=10)
    ax1.set_ylabel('Method A (baseline)', fontsize=10)

    # 添加数值和效应大小标注
    for i in range(n_methods):
        for j in range(n_methods):
            delta_val = delta_matrix[i, j]
            effect = interpret_cliffs_delta(delta_val)

            # 文本颜色（深色背景用白色，浅色背景用黑色）
            text_color = 'white' if abs(delta_val) > 0.5 else 'black'

            # 主数值
            ax1.text(j, i, f'{delta_val:.2f}',
                    ha="center", va="center", color=text_color,
                    fontsize=9, fontweight='bold')

            # 效应大小注释（对角线除外）
            if i != j and abs(delta_val) >= 0.147:  # 只标注非negligible的
                ax1.text(j, i + 0.3, f'({effect[0]})',
                        ha="center", va="center", color=text_color,
                        fontsize=7, style='italic')

    # 颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Cliff's Delta\n(negative = A better)", fontsize=10)

    # === 右图：胜率 ===
    im2 = ax2.imshow(winrate_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(range(n_methods))
    ax2.set_yticks(range(n_methods))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_yticklabels(methods)
    ax2.set_title(f"Pairwise Win Rate\n{metric_name}\n"
                  "(row method wins over column method)", fontsize=12)
    ax2.set_xlabel('Method B (opponent)', fontsize=10)
    ax2.set_ylabel('Method A (player)', fontsize=10)

    # 添加数值标注
    for i in range(n_methods):
        for j in range(n_methods):
            winrate_val = winrate_matrix[i, j]

            # 文本颜色
            if winrate_val > 0.75:
                text_color = 'white'
            elif winrate_val < 0.25:
                text_color = 'white'
            else:
                text_color = 'black'

            # 主数值
            ax2.text(j, i, f'{winrate_val:.0%}',
                    ha="center", va="center", color=text_color,
                    fontsize=10, fontweight='bold')

            # 显著标记（胜率 > 60% 或 < 40%）
            if i != j:
                if winrate_val > 0.6:
                    marker = '↑'  # A 显著更好
                elif winrate_val < 0.4:
                    marker = '↓'  # B 显著更好
                else:
                    marker = '≈'  # 接近

                ax2.text(j, i + 0.35, marker,
                        ha="center", va="center", color=text_color,
                        fontsize=12)

    # 颜色条
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Win Rate\n(A wins over B)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved effect size heatmaps: {output_path}")


def generate_pairwise_comparison_table(summary_df: pd.DataFrame,
                                       output_path: Path,
                                       top_k: int = 20):
    """
    生成配对比较表（显示最大效应量的比较）

    Args:
        summary_df: 配对统计汇总 DataFrame
        output_path: 输出路径
        top_k: 显示前 k 对比较
    """
    # 按 Cliff's Delta 绝对值排序
    summary_df['Abs_Delta'] = summary_df['Cliff_Delta'].abs()
    top_comparisons = summary_df.nlargest(top_k, 'Abs_Delta')

    # 格式化显示
    display_df = top_comparisons[[
        'Method A', 'Method B', 'Cliff_Delta', 'Effect_Size', 'Win_Rate', 'A_Better'
    ]].copy()

    display_df['Cliff_Delta'] = display_df['Cliff_Delta'].map('{:.3f}'.format)
    display_df['Win_Rate'] = display_df['Win_Rate'].map('{:.1%}'.format)

    # 保存为 CSV
    display_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved pairwise comparison table: {output_path}")

    return display_df



if __name__ == "__main__":
    print("Visualization helpers loaded successfully")