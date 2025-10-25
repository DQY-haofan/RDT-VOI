"""
Main experimental script for RDT-VoI simulation.
Runs complete benchmarking suite with spatial CV.
"""

from pathlib import Path
from datetime import datetime
import json
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from visualization_helpers import generate_expert_plots

from matplotlib import pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from typing import Dict, List
import pickle
import time
from config import load_config
from geometry import build_grid2d_geometry
from spatial_field import build_prior, sample_gmrf
from sensors import generate_sensor_pool
from inference import compute_posterior, compute_posterior_variance_diagonal
from sensors import get_observation

from selection import greedy_mi, greedy_aopt, uniform_selection, random_selection
from evaluation import spatial_block_cv, compute_metrics, morans_i
# 🔥 Import new method wrappers
from method_wrappers import (
    get_selection_method,
    get_available_methods,
    should_use_evi
)

# 🔥 Import visualization
from visualization import (
    plot_budget_curves,
    plot_performance_profile,
    plot_critical_difference,
    plot_marginal_mi,
    plot_mi_evi_correlation,
    plot_marginal_efficiency,
    plot_calibration_diagnostics,
    plot_sensor_placement_map, setup_style
)

# 🔥 添加自定义JSON encoder
class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON encoder"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# ✅ 在顶层定义方法映射函数（可以被 pickle）

def run_single_fold_worker(fold_data: Dict) -> Dict:
    """
    Worker function for a single CV fold.
    Separates selection from evaluation for parallel execution.

    Args:
        fold_data: Dictionary containing:
            - train_idx, test_idx: CV split
            - selection_method: Function to call
            - k: Budget
            - Q_pr, mu_pr: Prior
            - x_true: True state
            - sensors: Sensor pool
            - decision_config: Decision model
            - rng_seed: Random seed for this fold

    Returns:
        Dictionary with metrics and selection results
    """
    # Unpack data
    train_idx = fold_data['train_idx']
    test_idx = fold_data['test_idx']
    selection_method = fold_data['selection_method']
    k = fold_data['k']
    Q_pr = fold_data['Q_pr']
    mu_pr = fold_data['mu_pr']
    x_true = fold_data['x_true']
    sensors = fold_data['sensors']
    decision_config = fold_data['decision_config']
    geom = fold_data['geom']
    rng = np.random.default_rng(fold_data['rng_seed'])

    try:
        # 🔥 Run selection (pass mu_pr for methods that need it)
        t_start = time.time()
        selection_result = selection_method(sensors, k, Q_pr, mu_pr)
        selection_time = time.time() - t_start

        # Get selected sensors
        selected_sensors = [sensors[i] for i in selection_result.selected_ids]

        # Generate observations
        y, H, R = get_observation(x_true, selected_sensors, rng)

        # Compute posterior
        t_start = time.time()
        mu_post, factor = compute_posterior(Q_pr, mu_pr, H, R, y)
        inference_time = time.time() - t_start

        # Get posterior uncertainties on test set
        var_post_test = compute_posterior_variance_diagonal(factor, test_idx)
        sigma_post_test = np.sqrt(np.maximum(var_post_test, 1e-12))

        # Expand to full arrays for metrics
        sigma_post = np.zeros(len(mu_post))
        sigma_post[test_idx] = sigma_post_test

        # Compute metrics
        metrics = compute_metrics(
            mu_post, sigma_post, x_true, test_idx, decision_config
        )

        # Spatial diagnostics
        residuals = mu_post - x_true
        if geom.adjacency is not None:
            I_stat, I_pval = morans_i(
                residuals[test_idx],
                geom.adjacency[test_idx][:, test_idx],
                n_permutations=999,
                rng=rng
            )
            metrics['morans_i'] = I_stat
            metrics['morans_pval'] = I_pval

        # Add timing
        metrics['selection_time_sec'] = selection_time
        metrics['inference_time_sec'] = inference_time

        # Add selection diagnostics
        metrics['n_selected'] = len(selection_result.selected_ids)
        metrics['total_cost'] = selection_result.total_cost

        return {
            'success': True,
            'metrics': metrics,
            'selection_result': selection_result,
            'mu_post': mu_post,
            'sigma_post': sigma_post
        }

    except Exception as e:
        warnings.warn(f"Fold evaluation failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'metrics': {}
        }


def run_method_evaluation(method_name: str, cfg, geom, Q_pr, mu_pr,
                          x_true, sensors, test_idx_global=None) -> Dict:
    """
    🔥 修复版：运行完整的方法评估

    Run complete evaluation for one method across all budgets and CV folds.
    """
    print(f"\n{'=' * 70}")
    print(f"  Method: {method_name.upper()}")
    print(f"{'=' * 70}")

    rng = cfg.get_rng()

    # 创建选择方法wrapper
    try:
        selection_method = get_selection_method(
            method_name=method_name,
            config=cfg,
            geom=geom,
            x_true=x_true,
            test_idx=test_idx_global
        )
    except Exception as e:
        print(f"  ✗ Failed to create method wrapper: {e}")
        raise

    # 生成CV folds
    buffer_width = cfg.cv.buffer_width_multiplier * cfg.prior.correlation_length
    folds = spatial_block_cv(
        geom.coords,
        cfg.cv.k_folds,
        buffer_width,
        cfg.cv.block_strategy,
        rng
    )

    # 打印fold信息
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}: train={len(train_idx)}, test={len(test_idx)}")

    results = {
        'budgets': {},
        'method_name': method_name,
        'n_folds': len(folds)
    }

    # 遍历budgets
    for k in cfg.selection.budgets:
        print(f"\n  Budget k={k}")
        print(f"  {'-' * 50}")

        budget_results = {
            'fold_results': [],
            'fold_metrics': []
        }

        # 遍历folds
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            # 🔥 关键修复：只对 EVI 方法检查是否跳过
            is_evi_method = method_name.lower() in ['greedy_evi', 'evi', 'greedy-evi', 'myopic_evi']

            if is_evi_method:
                # EVI 方法可能需要跳过某些配置以节省时间
                if not should_use_evi(method_name, k, fold_idx, cfg):
                    print(f"    Fold {fold_idx + 1}/{len(folds)}: SKIPPED (EVI subset)")
                    budget_results['fold_results'].append({
                        'success': False,
                        'skipped': True,
                        'reason': 'EVI budget/fold subset'
                    })
                    continue

            print(f"    Fold {fold_idx + 1}/{len(folds)}: train={len(train_idx)}, test={len(test_idx)}")

            # 准备fold数据
            fold_data = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'selection_method': selection_method,
                'k': k,
                'Q_pr': Q_pr,
                'mu_pr': mu_pr,
                'x_true': x_true,
                'sensors': sensors,
                'decision_config': cfg.decision,
                'geom': geom,
                'rng_seed': rng.integers(0, 2 ** 31)
            }

            # 运行fold
            try:
                fold_result = run_single_fold_worker(fold_data)
                budget_results['fold_results'].append(fold_result)

                if fold_result['success']:
                    metrics = fold_result['metrics']
                    budget_results['fold_metrics'].append(metrics)

                    # 打印关键指标
                    print(f"        RMSE={metrics['rmse']:.3f}, "
                          f"Loss=£{metrics['expected_loss_gbp']:.0f}, "
                          f"Coverage={metrics['coverage_90'] * 100:.2f}%")
                else:
                    print(f"        ✗ FAILED: {fold_result.get('error', 'unknown error')}")

            except Exception as e:
                print(f"        ✗ Exception: {str(e)}")
                import traceback
                traceback.print_exc()

                budget_results['fold_results'].append({
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })

        # 计算budget级别的统计
        if budget_results['fold_metrics']:
            n_folds = len(budget_results['fold_metrics'])

            # 聚合指标
            aggregated = {}
            for key in budget_results['fold_metrics'][0].keys():
                # 跳过非数值指标
                if key in ['z_scores', 'n_test', 'n_selected']:
                    continue

                values = [m[key] for m in budget_results['fold_metrics'] if key in m]
                if values and all(isinstance(v, (int, float)) for v in values):
                    aggregated[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'values': values
                    }

            budget_results['aggregated'] = aggregated

            print(f"\n    Summary (n={n_folds} folds):")
            for metric in ['expected_loss_gbp', 'rmse']:
                if metric in aggregated:
                    stats = aggregated[metric]

                    # 🔥 修复：条件格式化
                    if 'loss' in metric:
                        mean_str = f"{stats['mean']:.0f}"
                        std_str = f"{stats['std']:.0f}"
                    else:
                        mean_str = f"{stats['mean']:.3f}"
                        std_str = f"{stats['std']:.3f}"

                    print(f"      {metric.replace('_', ' ').title()}: "
                          f"{mean_str} ± {std_str}")
        else:
            print(f"\n    ⚠️  No successful folds for budget k={k}")

        results['budgets'][k] = budget_results

    return results


def aggregate_results_for_visualization(all_results: Dict) -> pd.DataFrame:
    """
    🔥 强化修复版：将结果转换为DataFrame供可视化使用

    处理所有方法的数据，包括可能的结构差异
    """
    rows = []

    print("    开始聚合结果...")

    for method_name, method_data in all_results.items():
        print(f"      处理方法: {method_name}")

        # 🔥 统一方法显示名称
        method_display = method_name.replace('_', ' ').title()

        # 🔥 检查数据结构
        if not isinstance(method_data, dict):
            print(f"        ⚠️  跳过：数据类型错误 ({type(method_data)})")
            continue

        budgets_data = method_data.get('budgets', {})

        if not budgets_data:
            print(f"        ⚠️  跳过：无budgets数据")
            continue

        print(f"        找到 {len(budgets_data)} 个budgets")

        for budget, budget_data in budgets_data.items():
            # 🔥 安全地获取fold_results
            if isinstance(budget_data, dict):
                fold_results = budget_data.get('fold_results', [])
            else:
                print(f"          Budget {budget}: 数据类型错误 ({type(budget_data)})")
                continue

            if not fold_results:
                print(f"          Budget {budget}: 无fold结果")
                continue

            # 收集所有fold的指标
            valid_folds = 0

            for fold_idx, fold_res in enumerate(fold_results):
                # 🔥 检查fold_res结构
                if not isinstance(fold_res, dict):
                    continue

                if not fold_res.get('success', False):
                    continue

                metrics = fold_res.get('metrics', {})

                if not metrics or not isinstance(metrics, dict):
                    continue

                valid_folds += 1

                # 🔥 为每个指标创建行
                for metric_name, metric_value in metrics.items():
                    # 跳过非标量指标
                    if metric_name in ['z_scores'] or metric_name.startswith('_'):
                        continue

                    # 🔥 确保值是标量
                    if isinstance(metric_value, (list, np.ndarray)):
                        continue

                    if metric_value is None or (isinstance(metric_value, float) and np.isnan(metric_value)):
                        continue

                    rows.append({
                        'method': method_display,
                        'budget': int(budget),
                        'fold': fold_idx + 1,
                        'metric': metric_name,
                        'value': float(metric_value)
                    })

            print(f"          Budget {budget}: {valid_folds} 个有效folds")

    if not rows:
        warnings.warn("没有有效的结果可以聚合")
        return pd.DataFrame()

    # 创建DataFrame
    df = pd.DataFrame(rows)

    # 🔥 计算统计量
    stats_rows = []
    for (method, budget, metric), group in df.groupby(['method', 'budget', 'metric']):
        values = group['value'].values
        stats_rows.append({
            'method': method,
            'budget': budget,
            'fold': None,  # 聚合统计
            'metric': metric,
            'value': np.mean(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'n_folds': len(values)
        })

    # 合并原始数据和统计数据
    df_stats = pd.DataFrame(stats_rows)
    df_combined = pd.concat([df, df_stats], ignore_index=True)

    print(f"    ✓ 聚合完成: {len(df)} 行原始数据 + {len(df_stats)} 行统计数据")
    print(f"    ✓ 方法: {sorted(df['method'].unique())}")
    print(f"    ✓ 指标: {len(df['metric'].unique())} 个")

    return df_combined

def serialize_sparse_matrix(mat):
    """将稀疏矩阵序列化为字典"""
    mat_csr = mat.tocsr()
    return {
        'data': mat_csr.data,
        'indices': mat_csr.indices,
        'indptr': mat_csr.indptr,
        'shape': mat_csr.shape
    }


def serialize_geometry(geom):
    """将 Geometry 对象序列化"""
    return {
        'mode': geom.mode,
        'n': geom.n,
        'coords': geom.coords,
        'adj_data': geom.adjacency.data,
        'adj_indices': geom.adjacency.indices,
        'adj_indptr': geom.adjacency.indptr,
        'lap_data': geom.laplacian.data,
        'lap_indices': geom.laplacian.indices,
        'lap_indptr': geom.laplacian.indptr,
        'h': geom.h
    }


def serialize_sensors(sensors):
    """将传感器列表序列化"""
    return [
        {
            'id': s.id,
            'idxs': s.idxs,
            'weights': s.weights,
            'noise_var': s.noise_var,
            'cost': s.cost,
            'type_name': s.type_name
        }
        for s in sensors
    ]


def run_cv_experiment_parallel(geom, Q_pr, mu_pr, x_true, sensors,
                               method_name, k, cv_config,
                               decision_config, seed, n_workers=5):
    """
    并行运行CV实验

    Args:
        geom: 几何对象
        Q_pr, mu_pr: 先验参数
        x_true: 真实状态
        sensors: 候选传感器池
        method_name: 方法名称
        k: 预算
        cv_config: CV配置
        decision_config: 决策配置
        seed: 随机种子
        n_workers: 并行worker数量

    Returns:
        results: CV结果字典
    """
    from evaluation import spatial_block_cv
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # 生成CV折
    buffer_width = 15.0  # 或从 cv_config 获取
    folds = spatial_block_cv(
        geom.coords,
        k_folds=cv_config.get('k_folds', 3),
        buffer_width=buffer_width,
        block_strategy=cv_config.get('block_strategy', 'kmeans'),
        rng=np.random.default_rng(seed)
    )

    print(f"\n  Running {len(folds)} folds in parallel with {n_workers} workers...")

    # 提交所有任务
    futures = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            # 🔥 关键修复：传递所有参数
            future = executor.submit(
                run_single_fold_worker,
                method_name,  # 参数1
                geom,  # 参数2
                Q_pr,  # 参数3
                mu_pr,  # 参数4
                x_true,  # 参数5
                sensors,  # 参数6
                k,  # 参数7
                train_idx,  # 参数8
                test_idx,  # 参数9
                decision_config,  # 参数10
                fold_idx,  # 参数11
                seed  # 参数12
            )
            futures.append(future)

        # 收集结果
        fold_results = []
        for future in as_completed(futures):
            try:
                fold_idx, metrics = future.result()
                fold_results.append(metrics)
            except Exception as e:
                print(f"\n✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                raise

    # 按fold_idx排序
    # fold_results.sort(key=lambda x: x.get('fold_idx', 0))

    # 聚合结果
    aggregated = {}
    for key in fold_results[0].keys():
        # 跳过非数值字段
        if key in ['selection_result', 'mu_post', 'x_true', 'test_idx', 'fold_idx']:
            continue

        values = np.array([fr[key] for fr in fold_results])
        aggregated[key] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'values': values.tolist()
        }

    return {
        'fold_results': fold_results,
        'aggregated': aggregated
    }

def create_output_dir(cfg) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.experiment.output_dir) / f"{cfg.experiment.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "curves").mkdir(exist_ok=True)
    (output_dir / "diagnostics").mkdir(exist_ok=True)

    # Save config
    cfg.save_to(output_dir)

    # Save environment
    import subprocess
    try:
        env_info = subprocess.check_output(['pip', 'freeze']).decode()
        with open(output_dir / "environment.txt", 'w', encoding='utf-8') as f:
            f.write(env_info)
    except:
        pass

    return output_dir


def run_milestone_m1(cfg, output_dir):
    """
    Milestone M1: Single-fold validation on small grid.
    Verify monotonic decrease and diminishing returns.
    """
    print("\n" + "="*60)
    print("MILESTONE M1: Small-scale validation")
    print("="*60)

    rng = cfg.get_rng()

    # Override config for M1
    nx = ny = int(np.sqrt(cfg.acceptance.m1_grid_size))
    geom = build_grid2d_geometry(nx, ny, h=cfg.geometry.h)

    print(f"Grid: {nx}×{ny} = {geom.n} locations")

    # Build prior and sample true state
    Q_pr, mu_pr = build_prior(geom, cfg.prior)
    x_true = sample_gmrf(Q_pr, mu_pr, rng)

    # Generate sensors
    sensors = generate_sensor_pool(geom, cfg.sensors, rng)
    print(f"Sensor pool: {len(sensors)} candidates")

    # 只运行一次最大 budget
    max_budget = max(cfg.acceptance.m1_budgets)
    print(f"\nRunning Greedy-MI with budget k={max_budget}")
    result = greedy_mi(sensors, max_budget, Q_pr, lazy=True)

    # Save marginal VoI plot
    plot_marginal_mi(
        result,
        output_dir / "curves" / f"m1_marginal_voi_full.png"
    )

    # Check monotonicity
    is_monotonic = all(
        result.objective_values[i] <= result.objective_values[i+1]
        for i in range(len(result.objective_values)-1)
    )

    # Check diminishing returns
    is_diminishing = all(
        result.marginal_gains[i] >= result.marginal_gains[i+1] - 1e-6
        for i in range(len(result.marginal_gains)-1)
    )

    print("\n" + "-"*60)
    print(f"M1 RESULTS:")
    print(f"  Total steps: {len(result.objective_values)}")
    print(f"  Final MI: {result.objective_values[-1]:.3f} nats")
    print(f"  Monotonic increase: {'✓ PASS' if is_monotonic else '✗ FAIL'}")
    print(f"  Diminishing returns: {'✓ PASS' if is_diminishing else '✗ FAIL'}")
    print(f"\n  First 5 marginal gains: {result.marginal_gains[:5]}")
    print(f"  Last 5 marginal gains: {result.marginal_gains[-5:]}")
    print("-"*60)

    # Save results
    m1_summary = {
        'monotonic': is_monotonic,
        'diminishing': is_diminishing,
        'final_mi': float(result.objective_values[-1]),
        'marginal_gains': [float(x) for x in result.marginal_gains],
        'max_budget': max_budget
    }

    with open(output_dir / "m1_summary.json", 'w', encoding='utf-8') as f:
        json.dump(m1_summary, f, indent=2, cls=NumpyEncoder)

    return m1_summary


def run_milestone_m2(cfg, output_dir):
    """
    Milestone M2: Baseline comparison with spatial CV.
    """
    print("\n" + "="*60)
    print("MILESTONE M2: Baseline comparison")
    print("="*60)

    rng = cfg.get_rng()

    # Build full geometry
    geom = build_grid2d_geometry(cfg.geometry.nx, cfg.geometry.ny, h=cfg.geometry.h)
    print(f"Grid: {cfg.geometry.nx}×{cfg.geometry.ny} = {geom.n} locations")

    # Build prior and sample
    Q_pr, mu_pr = build_prior(geom, cfg.prior)
    x_true = sample_gmrf(Q_pr, mu_pr, rng)

    # Generate sensors
    sensors = generate_sensor_pool(geom, cfg.sensors, rng)
    print(f"Sensor pool: {len(sensors)} candidates")

    # 方法名称列表
    method_names = ['Greedy-MI', 'Greedy-A', 'Uniform', 'Random']

    # Run experiments for each method and budget
    all_results = {}

    for method_name in method_names:
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        all_results[method_name] = {}

        for k in cfg.selection.budgets:
            print(f"\nBudget k={k}")

            # 使用并行版本
            cv_results = run_cv_experiment_parallel(
                geom=geom,
                Q_pr=Q_pr,
                mu_pr=mu_pr,
                x_true=x_true,
                sensors=sensors,
                method_name=method_name,
                k=k,
                cv_config=cfg.cv.__dict__,  # 或直接传cfg.cv
                decision_config=cfg.decision,
                seed=cfg.experiment.seed,
                n_workers=5
            )

            all_results[method_name][k] = cv_results

    # Save results
    with open(output_dir / "m2_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)

    # Generate plots
    print("\nGenerating plots...")

    # Budget curves for each metric
    for metric in ['expected_loss_gbp', 'rmse', 'mae']:
        plot_budget_curves(
            all_results, metric,
            output_dir / "curves" / f"m2_budget_{metric}.png",
            show_ci=True
        )

    # Performance profile
    perf_data = {}
    for method in method_names:
        perf_data[method] = {
            f"k{k}": all_results[method][k]['aggregated']['expected_loss_gbp']['mean']
            for k in cfg.selection.budgets
        }

    plot_performance_profile(
        perf_data, 'expected_loss_gbp',
        output_dir / "curves" / "m2_performance_profile.png"
    )

    # Critical difference diagram
    plot_critical_difference(
        perf_data, 'expected_loss_gbp',
        output_dir / "curves" / "m2_critical_difference.png"
    )

    # Check acceptance criterion
    greedy_mi_loss = all_results['Greedy-MI'][cfg.selection.budgets[-1]]['aggregated']['expected_loss_gbp']['mean']
    random_loss = all_results['Random'][cfg.selection.budgets[-1]]['aggregated']['expected_loss_gbp']['mean']

    improvement = (random_loss - greedy_mi_loss) / random_loss
    passes_m2 = improvement >= cfg.acceptance.m2_min_improvement_vs_random

    print("\n" + "-"*60)
    print(f"M2 RESULTS:")
    print(f"  Greedy-MI loss: £{greedy_mi_loss:.0f}")
    print(f"  Random loss: £{random_loss:.0f}")
    print(f"  Improvement: {improvement*100:.1f}%")
    print(f"  Threshold: {cfg.acceptance.m2_min_improvement_vs_random*100:.1f}%")
    print(f"  Status: {'✓ PASS' if passes_m2 else '✗ FAIL'}")
    print("-"*60)

    m2_summary = {
        'passes': passes_m2,
        'improvement': improvement,
        'greedy_mi_loss': greedy_mi_loss,
        'random_loss': random_loss
    }


    with open(output_dir / "m2_summary.json", 'w', encoding='utf-8') as f:
        json.dump(m2_summary, f, indent=2, cls=NumpyEncoder)

    try:
        generate_all_expert_plots(all_results, cfg, geom, sensors, output_dir)
    except Exception as e:
        print(f"⚠️  Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()

    return m2_summary, all_results


def generate_all_expert_plots(all_results, cfg, geom, sensors, output_dir):
    """
    生成专家建议的所有图像（改进的错误处理版本）
    """
    from visualization import (
        setup_style,
        plot_budget_curves,
        plot_marginal_efficiency,
        plot_type_composition,
        plot_mi_voi_correlation,
        plot_calibration_diagnostics,
        plot_spatial_diagnostics_enhanced,
        plot_performance_profile,
        plot_critical_difference,
        plot_sensor_placement_map
    )
    import traceback

    print("\n" + "=" * 60)
    print("GENERATING EXPERT-RECOMMENDED VISUALIZATIONS")
    print("=" * 60)

    setup_style()

    # 创建输出子目录
    curves_dir = output_dir / "curves"
    maps_dir = output_dir / "maps"
    calibration_dir = output_dir / "calibration"
    comparison_dir = output_dir / "comparison"

    for d in [curves_dir, maps_dir, calibration_dir, comparison_dir]:
        d.mkdir(exist_ok=True)

    # ========== F1: 预算-损失前沿 ==========
    print("\n📈 F1: Budget-Loss Frontier...")
    try:
        results_by_method = {}
        for method_name in all_results.keys():
            results_by_method[method_name] = {}
            for budget in all_results[method_name].keys():
                results_by_method[method_name][budget] = all_results[method_name][budget]

        plot_budget_curves(
            results_by_method,
            metric='expected_loss_gbp',
            output_path=curves_dir / "f1_budget_loss_frontier.png",
            show_ci=True
        )
        print("  ✅ F1 saved")
    except Exception as e:
        print(f"  ❌ F1 failed: {e}")
        traceback.print_exc()

    # ========== F2 + F3 + F10: 需要 selection_result ==========
    print("\n💰 F2, 🎨 F3, 🗺️ F10: Processing selection results...")

    for method_name in ['Greedy-MI', 'Greedy-A']:
        if method_name not in all_results:
            print(f"  ⏭️ Skipping {method_name} (not in results)")
            continue

        budgets = sorted(all_results[method_name].keys())

        for budget in budgets:
            result_data = all_results[method_name][budget]

            # 🔥 调试：检查数据结构
            print(f"\n  Debug {method_name} k={budget}:")
            print(f"    Keys in result_data: {list(result_data.keys())}")

            if 'fold_results' not in result_data:
                print(f"    ❌ No fold_results")
                continue

            fold_results = result_data['fold_results']
            print(f"    Number of folds: {len(fold_results)}")

            if len(fold_results) == 0:
                print(f"    ❌ Empty fold_results")
                continue

            # 检查第一个fold
            fold_0 = fold_results[0]
            print(f"    Keys in fold_0: {list(fold_0.keys())}")

            if 'selection_result' not in fold_0:
                print(f"    ❌ No selection_result in fold")
                continue

            selection_result = fold_0['selection_result']
            print(f"    ✅ Found selection_result with {len(selection_result.selected_ids)} sensors")

            # ========== F2: 单位成本效率 ==========
            try:
                print(f"\n  💰 Generating F2 for {method_name} k={budget}...")
                plot_marginal_efficiency(
                    selection_result,
                    sensors,
                    output_path=curves_dir / f"f2_efficiency_{method_name.lower().replace('-', '_')}_k{budget}.png"
                )
                print(f"    ✅ F2 saved")
            except Exception as e:
                print(f"    ❌ F2 failed: {e}")
                traceback.print_exc()

            # ========== F3: 类型堆叠图 ==========
            try:
                print(f"\n  🎨 Generating F3 for {method_name} k={budget}...")
                plot_type_composition(
                    selection_result,
                    sensors,
                    output_path=curves_dir / f"f3_type_composition_{method_name.lower().replace('-', '_')}_k{budget}.png"
                )
                print(f"    ✅ F3 saved")
            except Exception as e:
                print(f"    ❌ F3 failed: {e}")
                traceback.print_exc()

            # ========== F10: 选址地图 ==========
            try:
                print(f"\n  🗺️ Generating F10 for {method_name} k={budget}...")
                plot_sensor_placement_map(
                    geom.coords,
                    selection_result.selected_ids,
                    sensors,
                    output_path=maps_dir / f"f10_placement_{method_name.lower().replace('-', '_')}_k{budget}.png"
                )
                print(f"    ✅ F10 saved")
            except Exception as e:
                print(f"    ❌ F10 failed: {e}")
                traceback.print_exc()

    # ========== F4: MI vs VoI 相关性 ==========
    print("\n🔗 F4: MI vs VoI Correlation...")
    print("  ⏭️ F4 skipped (requires EVI computation)")

    # ========== F5: 校准诊断 ==========
    print("\n📊 F5: Calibration Diagnostics...")
    try:
        for method_name in ['Greedy-MI']:
            if method_name not in all_results:
                continue

            all_fold_results = []
            for budget in all_results[method_name].keys():
                result_data = all_results[method_name][budget]
                if 'fold_results' in result_data:
                    all_fold_results.extend(result_data['fold_results'])

            if all_fold_results:
                print(f"  Processing {len(all_fold_results)} folds for {method_name}...")
                plot_calibration_diagnostics(
                    all_fold_results,
                    output_path=calibration_dir / f"f5_calibration_{method_name.lower().replace('-', '_')}.png"
                )
                print(f"  ✅ F5 saved")
            else:
                print(f"  ❌ No fold results for {method_name}")
    except Exception as e:
        print(f"  ❌ F5 failed: {e}")
        traceback.print_exc()

    # ========== F6: 空间诊断 ==========
    print("\n🗺️ F6: Spatial Diagnostics...")
    try:
        for method_name in ['Greedy-MI', 'Random']:
            if method_name not in all_results:
                continue

            budgets = list(all_results[method_name].keys())
            mid_budget = budgets[len(budgets) // 2]
            result_data = all_results[method_name][mid_budget]

            if 'fold_results' in result_data and len(result_data['fold_results']) > 0:
                fold_0 = result_data['fold_results'][0]

                # 🔥 检查是否有需要的数据
                if all(k in fold_0 for k in ['mu_post', 'x_true', 'test_idx']):
                    print(f"  Generating F6 for {method_name}...")
                    plot_spatial_diagnostics_enhanced(
                        fold_0['mu_post'],
                        fold_0['x_true'],
                        geom.coords,
                        fold_0['test_idx'],
                        method_name,
                        output_path=maps_dir / f"f6_spatial_{method_name.lower().replace('-', '_')}.png"
                    )
                    print(f"  ✅ F6 for {method_name} saved")
                else:
                    print(f"  ⏭️ F6 for {method_name} skipped (missing data)")
    except Exception as e:
        print(f"  ❌ F6 failed: {e}")
        traceback.print_exc()

    # ========== F7: 性能剖面 + CD图 ==========
    print("\n📉 F7: Performance Profile & Critical Difference...")
    try:
        perf_data = {}
        for method_name in all_results.keys():
            perf_data[method_name] = {}
            for budget in all_results[method_name].keys():
                instance_key = f"k={budget}"
                agg = all_results[method_name][budget]['aggregated']
                perf_data[method_name][instance_key] = agg['expected_loss_gbp']['mean']

        plot_performance_profile(
            perf_data,
            metric='expected_loss_gbp',
            output_path=comparison_dir / "f7a_performance_profile.png",
            tau_max=3.0
        )

        plot_critical_difference(
            perf_data,
            metric='expected_loss_gbp',
            output_path=comparison_dir / "f7b_critical_difference.png",
            alpha=0.05
        )
        print("  ✅ F7 saved")
    except Exception as e:
        print(f"  ❌ F7 failed: {e}")
        traceback.print_exc()

    # ========== F8 & F9 ==========
    print("\n🔬 F8: Ablation Study...")
    print("  ⏭️ F8 skipped (requires separate experiments)")

    print("\n⏱️ F9: Complexity Analysis...")
    print("  ⏭️ F9 skipped (requires timing instrumentation)")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\n📂 All plots saved to: {output_dir}")


def run_full_experiment(cfg):
    """Run complete experimental pipeline."""
    print("\n" + "="*70)
    print(" RDT-VoI SIMULATION: FULL EXPERIMENTAL PIPELINE")
    print("="*70)
    print(f"\nExperiment: {cfg.experiment.name}")
    print(f"Random seed: {cfg.experiment.seed}")
    print(f"Output directory: {cfg.experiment.output_dir}")

    # Create output directory
    output_dir = create_output_dir(cfg)
    print(f"\nOutput directory: {output_dir}")

    # Setup visualization
    setup_style(cfg.plots.style)

    # Run milestones
    try:
        # M1: Small-scale validation
        m1_summary = run_milestone_m1(cfg, output_dir)

        # M2: Full comparison
        m2_summary, all_results = run_milestone_m2(cfg, output_dir)

        # Final summary
        print("\n" + "="*70)
        print(" EXPERIMENT COMPLETE")
        print("="*70)
        print(f"\nAll results saved to: {output_dir}")
        print("\nMilestone Status:")
        print(f"  M1 (Monotonicity): {'✓ PASS' if m1_summary['monotonic'] else '✗ FAIL'}")
        print(f"  M1 (Diminishing):  {'✓ PASS' if m1_summary['diminishing'] else '✗ FAIL'}")
        print(f"  M2 (Improvement):  {'✓ PASS' if m2_summary['passes'] else '✗ FAIL'}")

        # Write final summary
        final_summary = {
            'experiment_name': cfg.experiment.name,
            'timestamp': datetime.now().isoformat(),
            'milestones': {
                'M1': m1_summary,
                'M2': m2_summary
            },
            'output_directory': str(output_dir)
        }

        with open(output_dir / "final_summary.json", 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, cls=NumpyEncoder)

        print(f"\n✓ Summary saved: {output_dir / 'final_summary.json'}")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

        # Save error log
        with open(output_dir / "error.log", 'w', encoding='utf-8') as f:
            traceback.print_exc(file=f)

        raise


def main():
    """Main evaluation pipeline."""

    print("=" * 70)
    print("  RDT-VoI EVALUATION FRAMEWORK")
    print("=" * 70)

    # ========================================================================
    # 1. SETUP
    # ========================================================================
    print("\n[1] Loading configuration...")
    cfg = load_config()
    rng = cfg.get_rng()

    # Create output directory
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"    Output directory: {output_dir}")

    # Save config copy
    cfg.save_to(output_dir)

    # ========================================================================
    # 2. BUILD SPATIAL DOMAIN
    # ========================================================================
    print("\n[2] Building spatial domain...")
    geom = build_grid2d_geometry(cfg.geometry.nx, cfg.geometry.ny, cfg.geometry.h)
    print(f"    Domain: {geom.n} locations, {geom.adjacency.nnz} edges")

    # ========================================================================
    # 3. CONSTRUCT PRIOR
    # ========================================================================
    print("\n[3] Constructing GMRF prior...")
    Q_pr, mu_pr = build_prior(geom, cfg.prior)
    print(f"    Precision sparsity: {Q_pr.nnz / geom.n ** 2 * 100:.2f}%")
    print(f"    Correlation length: {cfg.prior.correlation_length:.1f} m")

    # ========================================================================
    # 4. SAMPLE TRUE STATE
    # ========================================================================
    print("\n[4] Sampling true deterioration state...")
    x_true = sample_gmrf(Q_pr, mu_pr, rng)
    print(f"    State range: [{x_true.min():.2f}, {x_true.max():.2f}]")
    print(f"    Mean: {x_true.mean():.2f}, Std: {x_true.std():.2f}")

    # Save true state
    np.save(output_dir / 'x_true.npy', x_true)

    # ========================================================================
    # 5. GENERATE SENSOR POOL
    # ========================================================================
    print("\n[5] Generating sensor pool...")
    sensors = generate_sensor_pool(geom, cfg.sensors, rng)
    print(f"    Pool size: {len(sensors)} candidates")

    # Type distribution
    type_counts = {}
    for s in sensors:
        type_counts[s.type_name] = type_counts.get(s.type_name, 0) + 1
    print("    Type distribution:")
    for tname, count in type_counts.items():
        print(f"      {tname}: {count} ({count / len(sensors) * 100:.1f}%)")

    # ========================================================================
    # 6. PREPARE TEST SET (for EVI and diagnostics)
    # ========================================================================
    print("\n[6] Preparing global test set for EVI...")
    n_test = min(200, geom.n)
    test_idx_global = rng.choice(geom.n, size=n_test, replace=False)
    print(f"    Test set size: {n_test}")

    # ========================================================================
    # 7. RUN METHOD EVALUATIONS
    # ========================================================================
    print("\n[7] Running method evaluations...")

    methods = get_available_methods(cfg)
    print(f"    Methods to evaluate: {', '.join(methods)}")

    all_results = {}

    for method_name in methods:
        t_method_start = time.time()

        try:
            results = run_method_evaluation(
                method_name=method_name,
                cfg=cfg,  # 🔥 确保参数名与函数定义匹配
                geom=geom,
                Q_pr=Q_pr,
                mu_pr=mu_pr,
                x_true=x_true,
                sensors=sensors,
                test_idx_global=test_idx_global
            )
            all_results[method_name] = results

            t_method_elapsed = time.time() - t_method_start
            print(f"\n  ✓ {method_name} completed in {t_method_elapsed:.1f}s")

        except Exception as e:
            print(f"\n  ✗ {method_name} FAILED: {str(e)}")
            warnings.warn(f"Method {method_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # ========================================================================
    # 8. SAVE RESULTS
    # ========================================================================
    print("\n[8] Saving results...")

    # Save raw results
    with open(output_dir / 'results_raw.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"    Saved: results_raw.pkl")

    # 🔥 Convert to DataFrame with error handling
    print("    Converting results to DataFrame...")
    try:
        df_results = aggregate_results_for_visualization(all_results)

        if df_results.empty:
            print("    ⚠️  Warning: No results to aggregate (all methods may have failed)")
        else:
            df_results.to_csv(output_dir / 'results_aggregated.csv', index=False)
            print(f"    Saved: results_aggregated.csv ({len(df_results)} rows)")
    except Exception as e:
        print(f"    ✗ Failed to create aggregated DataFrame: {e}")
        df_results = pd.DataFrame()  # Empty DataFrame

    # ========================================================================
    # 9. VISUALIZATION
    # ========================================================================
    print("\n[9] Generating visualizations...")

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # 🔥 只有在有数据时才尝试可视化
    if df_results.empty:
        print("    ⚠️  Skipping visualization: no data available")
    else:
        try:
            # F1: Budget-performance curves
            print("    [F1] Budget-performance curves...")
            try:
                fig = plot_budget_curves(
                    df_results,
                    output_dir=plots_dir,
                    config=cfg
                )
                if fig:
                    plt.close(fig)
            except Exception as e:
                print(f"      Skipped: {e}")

            # F2: Marginal efficiency
            print("    [F2] Marginal efficiency...")
            try:
                fig = plot_marginal_efficiency(
                    all_results,
                    output_dir=plots_dir,
                    config=cfg
                )
                if fig:
                    plt.close(fig)
            except Exception as e:
                print(f"      Skipped: {e}")

            # F7a: Performance profile
            print("    [F7a] Performance profile...")
            try:
                fig = plot_performance_profile(
                    df_results,
                    metric='expected_loss_gbp',
                    output_dir=plots_dir,
                    config=cfg
                )
                if fig:
                    plt.close(fig)
            except Exception as e:
                print(f"      Skipped: {e}")

            # F7b: Critical difference diagram
            print("    [F7b] Critical difference diagram...")
            try:
                fig = plot_critical_difference(
                    df_results,
                    metric='expected_loss_gbp',
                    output_dir=plots_dir,
                    config=cfg
                )
                if fig:
                    plt.close(fig)
            except Exception as e:
                print(f"      Skipped: {e}")

            # F4: MI-EVI correlation (if both methods ran)
            if 'greedy_mi' in all_results and 'greedy_evi' in all_results:
                print("    [F4] MI-EVI correlation...")
                try:
                    fig = plot_mi_evi_correlation(
                        all_results['greedy_mi'],
                        all_results['greedy_evi'],
                        output_dir=plots_dir,
                        config=cfg
                    )
                    if fig:
                        plt.close(fig)
                except Exception as e:
                    print(f"      Skipped: {e}")

            # F5: Calibration diagnostics
            print("    [F5] Calibration diagnostics...")
            try:
                fig = plot_calibration_diagnostics(
                    all_results,
                    output_dir=plots_dir,
                    config=cfg
                )
                if fig:
                    plt.close(fig)
            except Exception as e:
                print(f"      Skipped: {e}")

            print(f"    ✓ Visualization complete")

        except Exception as e:
            print(f"    ✗ Visualization phase failed: {str(e)}")
            import traceback
            traceback.print_exc()

    generate_expert_plots(
        all_results=all_results,
        sensors=sensors,
        geom=geom,
        Q_pr=Q_pr,
        output_dir=output_dir,
        config=cfg
    )
    # ========================================================================
    # 10. SUMMARY REPORT
    # ========================================================================
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)

    print("\nSummary:")
    print(f"  Methods attempted: {len(methods)}")
    print(f"  Methods completed: {len(all_results)}")
    print(f"  Budgets: {cfg.selection.budgets}")
    print(f"  CV folds: {cfg.cv.k_folds}")
    print(f"  Total domain size: {geom.n}")
    print(f"  Candidate pool: {len(sensors)}")

    print(f"\nResults saved to: {output_dir}")
    print(f"  - results_raw.pkl")
    if not df_results.empty:
        print(f"  - results_aggregated.csv")
        print(f"  - plots/ directory")

    # Print best method per metric (if data available)
    if not df_results.empty and 30 in cfg.selection.budgets:
        print("\nBest methods (at k=30):")
        for metric in ['rmse', 'expected_loss_gbp', 'coverage_90']:
            try:
                df_k30 = df_results[
                    (df_results['budget'] == 30) &
                    (df_results['metric'] == metric)
                    ]
                if not df_k30.empty:
                    if metric == 'coverage_90':
                        # Closer to 0.90 is better
                        df_k30 = df_k30.copy()
                        df_k30['score'] = -np.abs(df_k30['mean'] - 0.90)
                        best_row = df_k30.loc[df_k30['score'].idxmax()]
                    else:
                        # Lower is better
                        best_row = df_k30.loc[df_k30['mean'].idxmin()]
                    print(f"  {metric:20s}: {best_row['method']:15s} "
                          f"({best_row['mean']:.3f} ± {best_row['std']:.3f})")
            except Exception:
                pass

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()