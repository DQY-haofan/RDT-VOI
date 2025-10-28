"""
Main experimental script for RDT-VoI simulation (CLI version with parallel support)
支持命令行参数指定配置文件和输出目录，集成并行处理

# 场景A（串行）
python main.py --scenario A

# 场景B（并行，5 workers）
python main.py -s B --parallel --workers 5

# 快速测试（并行）
python main.py -s A --quick-test --parallel

# 自定义配置
python main.py --config my_config.yaml --parallel

"""

from pathlib import Path
from datetime import datetime
import json
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, load_scenario_config
from geometry import build_grid2d_geometry
from spatial_field import build_prior, sample_gmrf, build_prior_with_ddi
from sensors import generate_sensor_pool
from inference import compute_posterior, compute_posterior_variance_diagonal
from sensors import get_observation

from method_wrappers import get_selection_method, get_available_methods
from evaluation import spatial_block_cv, compute_metrics, morans_i

from visualization import (
    setup_style,
    generate_all_visualizations_v2
)


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


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='RDT-VoI Simulation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用场景A配置
  python main.py --scenario A
  
  # 使用场景B配置，指定输出目录
  python main.py --scenario B --output results/scenario_B
  
  # 快速测试运行
  python main.py -s A --quick-test
  
  # 自定义配置文件
  python main.py --config path/to/custom.yaml
  
  # 并行处理（5个worker）
  python main.py -s A --parallel --workers 5
        """
    )

    # 主要参数
    parser.add_argument(
        '-s', '--scenario',
        type=str,
        choices=['A', 'B', 'a', 'b'],
        default=None,
        help='场景类型 (A=高风险, B=算力/鲁棒性) - 优先级高于 --config'
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='自定义配置文件路径（仅在未指定 --scenario 时使用）'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出根目录 (默认: 从配置文件读取)'
    )

    # 可选覆盖参数
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子 (覆盖配置文件)'
    )

    parser.add_argument(
        '--budgets',
        type=int,
        nargs='+',
        default=None,
        help='预算列表 (覆盖配置文件), 例如: --budgets 5 10 20'
    )

    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=None,
        help='要运行的方法 (覆盖配置文件), 例如: --methods greedy_mi greedy_evi'
    )

    # 🔥 并行处理选项
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='启用并行处理（显著加速CV折叠）'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='并行worker数量（默认：CPU核心数-1）'
    )

    # 测试和调试选项
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='快速测试模式 (小网格, 少预算, 少fold)'
    )

    parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='跳过可视化生成 (仅运行实验)'
    )

    parser.add_argument(
        '--viz-only',
        action='store_true',
        help='仅生成可视化 (加载已有结果)'
    )

    parser.add_argument(
        '--results-file',
        type=str,
        default=None,
        help='已有结果文件路径 (用于 --viz-only)'
    )

    # 详细程度
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='详细输出模式'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='静默模式（最少输出）'
    )

    return parser.parse_args()


def detect_scenario_from_config(cfg) -> str:
    """从配置自动检测场景类型"""
    exp_name = cfg.experiment.name.lower()
    if 'highstakes' in exp_name or 'high_stakes' in exp_name or '_a_' in exp_name:
        return 'A'
    elif 'proxy' in exp_name or 'compute' in exp_name or '_b_' in exp_name:
        return 'B'

    ddi = getattr(cfg.decision, 'target_ddi', 0.0)
    fn_fp_ratio = cfg.decision.L_FN_gbp / cfg.decision.L_FP_gbp if cfg.decision.L_FP_gbp > 0 else 1.0

    if ddi >= 0.2 or fn_fp_ratio > 10:
        return 'A'
    elif ddi < 0.15 or fn_fp_ratio < 5:
        return 'B'

    print("  ⚠️  Cannot determine scenario, defaulting to A (high-stakes)")
    return 'A'


def create_output_dir_from_config(cfg, config_path: str, custom_output: str = None) -> Path:
    """根据配置文件名创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 从配置文件名提取场景标识
    config_name = Path(config_path).stem

    # 移除 "config_" 前缀（如果有）
    if config_name.startswith('config_'):
        scenario_name = config_name[7:]
    else:
        scenario_name = config_name

    # 确定输出根目录
    if custom_output:
        base_dir = Path(custom_output)
    else:
        base_dir = Path(cfg.experiment.output_dir)

    # 创建层级目录结构
    output_dir = base_dir / f"scenario_{scenario_name}" / f"exp_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    (output_dir / "curves").mkdir(exist_ok=True)
    (output_dir / "diagnostics").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    # 保存配置副本
    cfg.save_to(output_dir)

    # 保存环境信息
    import subprocess
    try:
        env_info = subprocess.check_output(['pip', 'freeze']).decode()
        with open(output_dir / "environment.txt", 'w', encoding='utf-8') as f:
            f.write(env_info)
    except:
        pass

    # 保存运行命令
    with open(output_dir / "run_command.txt", 'w', encoding='utf-8') as f:
        f.write(' '.join(sys.argv))

    return output_dir


def apply_quick_test_overrides(cfg):
    """应用快速测试模式的覆盖"""
    print("\n  🚀 Quick test mode enabled:")

    # 小网格
    cfg.geometry.nx = 10
    cfg.geometry.ny = 10
    print(f"    → Grid: {cfg.geometry.nx}×{cfg.geometry.ny}")

    # 少预算
    cfg.selection.budgets = [3, 5]
    print(f"    → Budgets: {cfg.selection.budgets}")

    # 少fold
    cfg.cv.k_folds = 2
    print(f"    → CV folds: {cfg.cv.k_folds}")

    # 减少采样
    cfg.evi.monte_carlo_samples = 8
    print(f"    → MC samples: {cfg.evi.monte_carlo_samples}")

    return cfg


# ============================================================================
# 🔥 并行处理函数（从旧版移植）
# ============================================================================

def run_single_fold_worker(fold_data: dict) -> dict:
    """
    单fold评估worker（完整增强版）

    包含：
    - 先验/后验损失对比
    - ROI 计算
    - DDI 统计
    - 行动受限评估
    - 完整诊断信息
    """
    import time
    import warnings
    import numpy as np
    from inference import compute_posterior, compute_posterior_variance_diagonal, SparseFactor
    from sensors import get_observation
    from evaluation import compute_metrics, morans_i
    from decision import expected_loss

    # 解包数据
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
        # 1. 计算先验损失（用于 ROI）
        t_prior_start = time.time()
        tau = decision_config.get_threshold(mu_pr)
        factor_pr = SparseFactor(Q_pr)
        var_pr_test = compute_posterior_variance_diagonal(factor_pr, test_idx)
        sigma_pr_test = np.sqrt(np.maximum(var_pr_test, 1e-12))
        prior_loss = expected_loss(
            mu_pr[test_idx], sigma_pr_test, decision_config,
            test_indices=np.arange(len(test_idx)), tau=tau
        )
        prior_time = time.time() - t_prior_start

        # 2. 传感器选择
        t_sel_start = time.time()
        selection_result = selection_method(sensors, k, Q_pr, mu_pr)
        selection_time = time.time() - t_sel_start

        selected_sensors = [sensors[i] for i in selection_result.selected_ids]
        sensor_cost = selection_result.total_cost

        # 3. 生成观测
        y, H, R = get_observation(x_true, selected_sensors, rng)

        # 4. 计算后验
        t_inf_start = time.time()
        mu_post, factor_post = compute_posterior(Q_pr, mu_pr, H, R, y)
        inference_time = time.time() - t_inf_start

        var_post_test = compute_posterior_variance_diagonal(factor_post, test_idx)
        sigma_post_test = np.sqrt(np.maximum(var_post_test, 1e-12))
        sigma_post = np.zeros(len(mu_post))
        sigma_post[test_idx] = sigma_post_test

        # 5. 基础指标
        metrics = compute_metrics(mu_post, sigma_post, x_true, test_idx, decision_config)
        posterior_loss = metrics['expected_loss_gbp']

        # 6. ROI 和成本效率
        savings = prior_loss - posterior_loss
        if sensor_cost > 0:
            roi = (savings - sensor_cost) / sensor_cost
            cost_efficiency = savings / sensor_cost
        else:
            roi = np.inf if savings > 0 else 0.0
            cost_efficiency = np.inf if savings > 0 else 0.0

        metrics['roi'] = float(roi)
        metrics['cost_efficiency'] = float(cost_efficiency)
        metrics['prior_loss_gbp'] = float(prior_loss)
        metrics['posterior_loss_gbp'] = float(posterior_loss)
        metrics['savings_gbp'] = float(savings)

        # 7. DDI（决策难度指数）统计
        try:
            from spatial_field import compute_ddi
            ddi_test = compute_ddi(mu_post[test_idx], sigma_post_test, tau, k=1.0)
            metrics['ddi_test'] = float(ddi_test)

            sample_size = min(200, len(mu_pr))
            sample_idx = rng.choice(len(mu_pr), size=sample_size, replace=False)
            var_pr_sample = compute_posterior_variance_diagonal(factor_pr, sample_idx)
            sigma_pr_sample = np.sqrt(np.maximum(var_pr_sample, 1e-12))
            ddi_prior = compute_ddi(mu_pr[sample_idx], sigma_pr_sample, tau, k=1.0)
            metrics['ddi_prior'] = float(ddi_prior)
        except Exception as e:
            warnings.warn(f"DDI computation failed: {e}")
            metrics['ddi_test'] = np.nan
            metrics['ddi_prior'] = np.nan

        # 8. 行动受限评估（如果配置了 K_action）
        if hasattr(decision_config, 'K_action') and decision_config.K_action is not None:
            try:
                from scipy.stats import norm
                K_action = decision_config.K_action
                p_failure = 1.0 - norm.cdf((tau - mu_post[test_idx]) / np.maximum(sigma_post_test, 1e-12))

                if K_action < len(test_idx):
                    top_k_local = np.argsort(p_failure)[-K_action:]
                else:
                    top_k_local = np.arange(len(test_idx))

                constrained_risks = np.zeros(len(test_idx))
                for i in range(len(test_idx)):
                    global_idx = test_idx[i]
                    if i in top_k_local:
                        if x_true[global_idx] > tau:
                            constrained_risks[i] = decision_config.L_TP_gbp
                        else:
                            constrained_risks[i] = decision_config.L_FP_gbp
                    else:
                        if x_true[global_idx] > tau:
                            constrained_risks[i] = decision_config.L_FN_gbp
                        else:
                            constrained_risks[i] = decision_config.L_TN_gbp

                constrained_loss = constrained_risks.mean()
                true_exceed = x_true[test_idx] > tau
                if true_exceed.sum() > 0:
                    hit_rate = np.sum(np.isin(top_k_local, np.where(true_exceed)[0])) / true_exceed.sum()
                else:
                    hit_rate = 1.0

                metrics['action_K'] = int(K_action)
                metrics['action_constrained_loss'] = float(constrained_loss)
                metrics['action_regret'] = float(constrained_loss - posterior_loss)
                metrics['action_hit_rate'] = float(hit_rate)
                metrics['action_n_true_exceed'] = int(true_exceed.sum())
                metrics['action_n_maintained'] = int(len(top_k_local))
            except Exception as e:
                warnings.warn(f"Action-constrained evaluation failed: {e}")

        # 9. 空间诊断（Moran's I）
        residuals = mu_post - x_true
        if geom.adjacency is not None:
            try:
                I_stat, I_pval = morans_i(
                    residuals[test_idx],
                    geom.adjacency[test_idx][:, test_idx],
                    n_permutations=999, rng=rng
                )
                metrics['morans_i'] = float(I_stat)
                metrics['morans_pval'] = float(I_pval)
            except Exception as e:
                warnings.warn(f"Moran's I computation failed: {e}")
                metrics['morans_i'] = np.nan
                metrics['morans_pval'] = np.nan

        # 10. 时间统计
        metrics['prior_computation_time_sec'] = float(prior_time)
        metrics['selection_time_sec'] = float(selection_time)
        metrics['inference_time_sec'] = float(inference_time)
        metrics['total_time_sec'] = float(prior_time + selection_time + inference_time)

        # 11. 传感器选择诊断
        metrics['n_selected'] = len(selection_result.selected_ids)
        metrics['total_cost'] = float(sensor_cost)

        type_counts = {}
        for sid in selection_result.selected_ids:
            stype = sensors[sid].type_name
            type_counts[stype] = type_counts.get(stype, 0) + 1
        metrics['type_counts'] = {k: int(v) for k, v in type_counts.items()}

        selected_costs = [sensors[i].cost for i in selection_result.selected_ids]
        metrics['cost_mean'] = float(np.mean(selected_costs))
        metrics['cost_std'] = float(np.std(selected_costs))
        metrics['cost_min'] = float(np.min(selected_costs))
        metrics['cost_max'] = float(np.max(selected_costs))

        selected_noise_vars = [sensors[i].noise_var for i in selection_result.selected_ids]
        metrics['noise_mean'] = float(np.mean(selected_noise_vars))
        metrics['noise_std'] = float(np.std(selected_noise_vars))

        # 12. 近阈值区域统计
        try:
            selected_locs = [sensors[i].idxs[0] for i in selection_result.selected_ids]
            selected_gaps = np.abs(mu_pr[selected_locs] - tau)
            var_selected = compute_posterior_variance_diagonal(factor_pr, np.array(selected_locs))
            sigma_selected = np.sqrt(np.maximum(var_selected, 1e-12))
            near_threshold = selected_gaps <= sigma_selected
            metrics['frac_sensors_near_threshold'] = float(near_threshold.mean())
        except Exception as e:
            warnings.warn(f"Near-threshold statistics failed: {e}")
            metrics['frac_sensors_near_threshold'] = np.nan

        return {
            'success': True,
            'metrics': metrics,
            'selection_result': selection_result,
            'mu_post': mu_post,
            'sigma_post': sigma_post,
            'residuals': residuals[test_idx],
            'test_idx': test_idx,
            'tau': tau,
            'prior_loss': prior_loss,
            'posterior_loss': posterior_loss,
            'savings': savings,
            'roi': roi
        }

    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        warnings.warn(f"Fold evaluation failed: {error_msg}")

        return {
            'success': False,
            'error': error_msg,
            'traceback': error_trace,
            'metrics': {},
            'selection_result': None,
            'mu_post': None,
            'sigma_post': None,
            'residuals': None,
            'test_idx': test_idx,
            'tau': None,
            'prior_loss': None,
            'posterior_loss': None,
            'savings': None,
            'roi': None
        }


def run_method_evaluation(method_name: str, cfg, geom, Q_pr, mu_pr,
                          x_true, sensors, test_idx_global=None,
                          use_parallel=False, n_workers=None, verbose=True) -> dict:
    """
    运行方法评估（支持并行处理）

    Args:
        use_parallel: 是否使用并行处理
        n_workers: 并行worker数量（None=自动检测）
        verbose: 是否输出详细信息
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Method: {method_name.upper()}")
        if use_parallel:
            print(f"  Mode: PARALLEL ({n_workers or 'auto'} workers)")
        else:
            print(f"  Mode: SEQUENTIAL")
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
        if verbose:
            print(f"  ✗ Failed to create method wrapper: {e}")
        raise

    # 生成CV folds
    buffer_width = cfg.cv.buffer_width_multiplier * cfg.prior.correlation_length
    folds = spatial_block_cv(
        geom.coords, cfg.cv.k_folds, buffer_width,
        cfg.cv.block_strategy, rng
    )

    if verbose:
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            print(f"  Fold {fold_idx + 1}: train={len(train_idx)}, test={len(test_idx)}")

    results = {
        'budgets': {},
        'method_name': method_name,
        'n_folds': len(folds)
    }

    # 遍历budgets
    for k in cfg.selection.budgets:
        if verbose:
            print(f"\n  Budget k={k}")
            print(f"  {'-' * 50}")

        budget_results = {
            'fold_results': [],
            'fold_metrics': []
        }

        # 🔥 准备所有fold的数据
        fold_data_list = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            # 检查是否需要跳过（仅EVI方法）
            is_evi_method = method_name.lower() in ['greedy_evi', 'evi', 'greedy-evi', 'myopic_evi']
            if is_evi_method:
                from method_wrappers import should_use_evi
                if not should_use_evi(method_name, k, fold_idx, cfg):
                    if verbose:
                        print(f"    Fold {fold_idx + 1}/{len(folds)}: SKIPPED (EVI subset)")
                    budget_results['fold_results'].append({
                        'success': False,
                        'skipped': True,
                        'reason': 'EVI budget/fold subset'
                    })
                    continue

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
            fold_data_list.append((fold_idx, fold_data))

        # 🔥 并行或串行执行
        if use_parallel and len(fold_data_list) > 1:
            # 并行模式
            if n_workers is None:
                n_workers = max(1, mp.cpu_count() - 1)

            if verbose:
                print(f"    Running {len(fold_data_list)} folds in parallel "
                      f"with {n_workers} workers...")

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # 提交所有任务
                future_to_fold = {
                    executor.submit(run_single_fold_worker, fold_data): fold_idx
                    for fold_idx, fold_data in fold_data_list
                }

                # 收集结果
                for future in as_completed(future_to_fold):
                    fold_idx = future_to_fold[future]
                    try:
                        fold_result = future.result()
                        budget_results['fold_results'].append(fold_result)

                        if fold_result['success']:
                            metrics = fold_result['metrics']
                            budget_results['fold_metrics'].append(metrics)
                            if verbose:
                                print(f"    Fold {fold_idx + 1}: "
                                      f"RMSE={metrics['rmse']:.3f}, "
                                      f"Loss=£{metrics['expected_loss_gbp']:.0f}")
                        else:
                            if verbose:
                                print(f"    Fold {fold_idx + 1}: "
                                      f"✗ {fold_result.get('error', 'unknown')}")
                    except Exception as e:
                        if verbose:
                            print(f"    Fold {fold_idx + 1}: ✗ Exception: {e}")
                        budget_results['fold_results'].append({
                            'success': False,
                            'error': str(e)
                        })
        else:
            # 串行模式
            for fold_idx, fold_data in fold_data_list:
                if verbose:
                    print(f"    Fold {fold_idx + 1}/{len(folds)}: "
                          f"train={len(fold_data['train_idx'])}, "
                          f"test={len(fold_data['test_idx'])}")

                try:
                    fold_result = run_single_fold_worker(fold_data)
                    budget_results['fold_results'].append(fold_result)

                    if fold_result['success']:
                        metrics = fold_result['metrics']
                        budget_results['fold_metrics'].append(metrics)
                        if verbose:
                            print(f"        RMSE={metrics['rmse']:.3f}, "
                                  f"Loss=£{metrics['expected_loss_gbp']:.0f}, "
                                  f"Coverage={metrics['coverage_90'] * 100:.2f}%")
                    else:
                        if verbose:
                            print(f"        ✗ FAILED: "
                                  f"{fold_result.get('error', 'unknown error')}")
                except Exception as e:
                    if verbose:
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
            aggregated = {}

            for key in budget_results['fold_metrics'][0].keys():
                if key in ['z_scores', 'n_test', 'n_selected', 'type_counts']:
                    continue

                values = [m[key] for m in budget_results['fold_metrics'] if key in m]
                if values and all(isinstance(v, (int, float)) for v in values):
                    aggregated[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'values': values
                    }

            budget_results['aggregated'] = aggregated

            if verbose:
                print(f"\n    Summary (n={n_folds} folds):")
                for metric in ['expected_loss_gbp', 'rmse']:
                    if metric in aggregated:
                        stats = aggregated[metric]
                        if 'loss' in metric:
                            mean_str = f"{stats['mean']:.0f}"
                            std_str = f"{stats['std']:.0f}"
                        else:
                            mean_str = f"{stats['mean']:.3f}"
                            std_str = f"{stats['std']:.3f}"
                        print(f"      {metric.replace('_', ' ').title()}: "
                              f"{mean_str} ± {std_str}")
        else:
            if verbose:
                print(f"\n    ⚠️  No successful folds for budget k={k}")

        results['budgets'][k] = budget_results

    return results


def aggregate_results_for_visualization(all_results: dict) -> pd.DataFrame:
    """将结果转换为DataFrame供可视化使用"""
    rows = []
    print("    开始聚合结果...")

    for method_name, method_data in all_results.items():
        print(f"      处理方法: {method_name}")
        if not isinstance(method_data, dict):
            continue

        budgets_data = method_data.get('budgets', {})
        if not budgets_data:
            continue

        for budget, budget_data in budgets_data.items():
            if not isinstance(budget_data, dict):
                continue

            fold_results = budget_data.get('fold_results', [])
            if not fold_results:
                continue

            for fold_idx, fold_res in enumerate(fold_results):
                if not isinstance(fold_res, dict):
                    continue
                if not fold_res.get('success', False):
                    continue

                metrics = fold_res.get('metrics', {})
                if not metrics or not isinstance(metrics, dict):
                    continue

                for metric_name, metric_value in metrics.items():
                    if metric_name in ['z_scores', 'type_counts']:
                        continue
                    if metric_name.startswith('_'):
                        continue
                    if isinstance(metric_value, (list, np.ndarray, dict)):
                        continue
                    if metric_value is None:
                        continue

                    try:
                        scalar_value = float(metric_value)
                    except (ValueError, TypeError):
                        continue

                    if np.isnan(scalar_value):
                        continue

                    rows.append({
                        'method': method_name,
                        'budget': int(budget),
                        'fold': fold_idx + 1,
                        'metric': metric_name,
                        'value': scalar_value
                    })

    if not rows:
        warnings.warn("没有有效的结果可以聚合")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 计算统计量
    stats_rows = []
    for (method, budget, metric), group in df.groupby(['method', 'budget', 'metric']):
        values = group['value'].values
        stats_rows.append({
            'method': method,
            'budget': budget,
            'fold': None,
            'metric': metric,
            'value': np.mean(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'n_folds': len(values)
        })

    df_stats = pd.DataFrame(stats_rows)
    df_combined = pd.concat([df, df_stats], ignore_index=True)

    print(f"    ✓ 聚合完成: {len(df)} 行原始数据 + {len(df_stats)} 行统计数据")
    return df_combined


def main():
    """主评估流程 - CLI版本"""

    # 解析命令行参数
    args = parse_arguments()

    # 设置输出详细程度
    verbose = not args.quiet

    if verbose:
        print("=" * 70)
        print("  RDT-VoI EVALUATION FRAMEWORK (CLI VERSION)")
        print("=" * 70)

    # 1. 加载配置
    if verbose:
        print(f"\n[1] Loading configuration...")

    # 🔥 场景模式：优先级最高
    if args.scenario:
        scenario_upper = args.scenario.upper()
        cfg = load_scenario_config(scenario_upper)
        config_path = f"config_{scenario_upper}_{'highstakes' if scenario_upper == 'A' else 'proxy'}.yaml"
        if verbose:
            print(f"    ✓ Loaded scenario {scenario_upper} config")
    elif args.config:
        cfg = load_config(args.config)
        config_path = args.config
        if verbose:
            print(f"    ✓ Loaded custom config: {args.config}")
    else:
        print("ERROR: Must specify either --scenario or --config")
        print("Examples:")
        print("  python main.py --scenario A")
        print("  python main.py --config path/to/config.yaml")
        sys.exit(1)

    # 应用命令行覆盖
    if args.seed:
        cfg.experiment.seed = args.seed
        if verbose:
            print(f"    → Override seed: {args.seed}")

    if args.budgets:
        cfg.selection.budgets = args.budgets
        if verbose:
            print(f"    → Override budgets: {args.budgets}")

    if args.methods:
        cfg.selection.methods = args.methods
        if verbose:
            print(f"    → Override methods: {args.methods}")

    # 快速测试模式
    if args.quick_test:
        cfg = apply_quick_test_overrides(cfg)

    rng = cfg.get_rng()

    # 检测场景类型
    scenario = detect_scenario_from_config(cfg)
    if verbose:
        print(f"\n    🎯 Detected Scenario: {scenario}")
        if scenario == 'A':
            print("       → High-stakes / Near-threshold decision focus")
        else:
            print("       → Compute efficiency / Robustness focus")

    # 🔥 并行处理设置
    use_parallel = args.parallel
    n_workers = args.workers
    if use_parallel and verbose:
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        print(f"\n    ⚡ Parallel processing enabled: {n_workers} workers")

    # 2. 创建输出目录
    output_dir = create_output_dir_from_config(cfg, config_path, args.output)
    if verbose:
        print(f"\n[2] Output directory: {output_dir}")

    # 仅可视化模式
    if args.viz_only:
        if not args.results_file:
            print("  ✗ Error: --viz-only requires --results-file")
            sys.exit(1)

        print(f"\n[VIZ-ONLY] Loading results from: {args.results_file}")
        with open(args.results_file, 'rb') as f:
            all_results = pickle.load(f)

        df_results = aggregate_results_for_visualization(all_results)
        print("  ⚠️  Visualization-only mode: skipping full pipeline")
        return

    # 3-8. 完整实验流程
    if verbose:
        print("\n[3] Building spatial domain...")
    geom = build_grid2d_geometry(cfg.geometry.nx, cfg.geometry.ny, cfg.geometry.h)

    if verbose:
        print("\n[4] Constructing GMRF prior with DDI control...")

    if hasattr(cfg.decision, 'target_ddi') and cfg.decision.target_ddi > 0:
        Q_temp, mu_temp = build_prior(geom, cfg.prior)
        tau = cfg.decision.get_threshold(mu_temp)
        if verbose:
            print(f"    Target DDI: {cfg.decision.target_ddi:.2%}")
            print(f"    Decision threshold: τ = {tau:.3f}")
        Q_pr, mu_pr = build_prior_with_ddi(
            geom, cfg.prior, tau=tau, target_ddi=cfg.decision.target_ddi
        )
    else:
        Q_pr, mu_pr = build_prior(geom, cfg.prior)

    if cfg.plots.ddi_overlay.get('enable', True) and verbose:
        from spatial_field import plot_ddi_heatmap, compute_ddi
        from inference import SparseFactor, compute_posterior_variance_diagonal
        factor = SparseFactor(Q_pr)
        sample_idx = rng.choice(geom.n, size=min(100, geom.n), replace=False)
        sample_vars = compute_posterior_variance_diagonal(factor, sample_idx)
        sigma_est = np.sqrt(np.mean(sample_vars)) * np.ones(geom.n)
        tau = cfg.decision.get_threshold(mu_pr)
        plot_ddi_heatmap(geom, mu_pr, sigma_est, tau,
                        output_path=output_dir / 'ddi_heatmap_prior.png')

    if verbose:
        print(f"    Precision sparsity: {Q_pr.nnz / geom.n ** 2 * 100:.2f}%")
        print(f"    Correlation length: {cfg.prior.correlation_length:.1f} m")

    if verbose:
        print("\n[5] Sampling true deterioration state...")
    x_true = sample_gmrf(Q_pr, mu_pr, rng)
    if verbose:
        print(f"    State range: [{x_true.min():.2f}, {x_true.max():.2f}]")
        print(f"    Mean: {x_true.mean():.2f}, Std: {x_true.std():.2f}")
    np.save(output_dir / 'x_true.npy', x_true)

    if verbose:
        print("\n[6] Generating sensor pool...")
    if cfg.sensors.use_heterogeneous:
        from sensors import generate_heterogeneous_sensor_pool, create_cost_zones_example
        if cfg.sensors.cost_zones:
            cost_zones = cfg.sensors.cost_zones
        else:
            cost_zones = create_cost_zones_example(geom)
        sensors = generate_heterogeneous_sensor_pool(
            geom, cfg.sensors, cost_zones=cost_zones, rng=rng
        )
    else:
        sensors = generate_sensor_pool(geom, cfg.sensors, rng)

    if verbose:
        print("\n[7] Preparing global test set for EVI...")
    n_test = min(200, geom.n)
    test_idx_global = rng.choice(geom.n, size=n_test, replace=False)
    if verbose:
        print(f"    Test set size: {n_test}")

    if verbose:
        print("\n[8] Running method evaluations...")
    methods = get_available_methods(cfg)
    if verbose:
        print(f"    Methods to evaluate: {', '.join(methods)}")

    all_results = {}
    for method_name in methods:
        t_method_start = datetime.now()
        try:
            results = run_method_evaluation(
                method_name=method_name,
                cfg=cfg,
                geom=geom,
                Q_pr=Q_pr,
                mu_pr=mu_pr,
                x_true=x_true,
                sensors=sensors,
                test_idx_global=test_idx_global,
                use_parallel=use_parallel,  # 🔥 传递并行参数
                n_workers=n_workers,
                verbose=verbose
            )
            all_results[method_name] = results
            t_method_elapsed = (datetime.now() - t_method_start).total_seconds()
            if verbose:
                print(f"\n  ✓ {method_name} completed in {t_method_elapsed:.1f}s")
        except Exception as e:
            if verbose:
                print(f"\n  ✗ {method_name} FAILED: {str(e)}")
            warnings.warn(f"Method {method_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 9. 保存结果
    if verbose:
        print("\n[9] Saving results...")
    with open(output_dir / 'results_raw.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    if verbose:
        print(f"    Saved: results_raw.pkl")

    if verbose:
        print("    Converting results to DataFrame...")
    try:
        df_results = aggregate_results_for_visualization(all_results)
        if df_results.empty:
            if verbose:
                print("    ⚠️  Warning: No results to aggregate")
        else:
            df_results.to_csv(output_dir / 'results_aggregated.csv', index=False)
            if verbose:
                print(f"    Saved: results_aggregated.csv ({len(df_results)} rows)")
    except Exception as e:
        if verbose:
            print(f"    ✗ Failed to create aggregated DataFrame: {e}")
        df_results = pd.DataFrame()

    # 10. 可视化
    if not args.skip_viz and not df_results.empty:
        if verbose:
            print("\n[10] Generating visualizations...")
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        try:
            generate_all_visualizations_v2(
                all_results=all_results,
                df_results=df_results,
                geom=geom,
                sensors=sensors,
                Q_pr=Q_pr,
                mu_pr=mu_pr,
                output_dir=output_dir,
                config=cfg,
                scenario=scenario
            )
            if verbose:
                print(f"    ✓ Visualization complete")
        except Exception as e:
            if verbose:
                print(f"    ✗ Visualization phase failed: {str(e)}")
            import traceback
            traceback.print_exc()
    elif args.skip_viz:
        if verbose:
            print("\n[10] Skipping visualization (--skip-viz)")

    # 总结
    if verbose:
        print("\n" + "=" * 70)
        print(f"  EVALUATION COMPLETE - SCENARIO {scenario}")
        print("=" * 70)
        print("\nSummary:")
        print(f"  Scenario: {scenario}")
        print(f"  Config: {config_path}")
        print(f"  Methods completed: {len(all_results)}/{len(methods)}")
        print(f"  Budgets: {cfg.selection.budgets}")
        print(f"  CV folds: {cfg.cv.k_folds}")
        if use_parallel:
            print(f"  Parallel workers: {n_workers}")
        print(f"\nResults saved to: {output_dir}")
        print("=" * 70)


if __name__ == "__main__":
    main()