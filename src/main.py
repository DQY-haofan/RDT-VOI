"""
主实验脚本 - 完整版（支持参数扫描和灵活配置）

🔥 主要改进：
1. 统一配置文件 + 参数覆盖
2. 支持参数扫描（单参数或多参数组合）
3. 保持向后兼容
4. 包含所有必要的核心函数

使用示例：
# 基础使用
python main.py                                    # 使用默认配置
python main.py --preset high_stakes               # 使用高风险预设
python main.py --preset low_stakes                # 使用低风险预设

# 单参数调整
python main.py --ddi 0.30 --fn-cost 120000        # 快速调整关键参数
python main.py --grid-size 25 --budgets 5,10,15   # 调整实验规模

# 参数扫描
python main.py --scan ddi=0.1,0.2,0.3             # DDI扫描
python main.py --scan fn_cost=30000,60000,120000   # 成本扫描
python main.py --scan ddi=0.2,0.3 fn_cost=60000,120000  # 组合扫描

# 控制选项
python main.py --parallel --workers 6             # 并行处理
python main.py --quick-test                       # 快速测试
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
import itertools
import scipy.sparse as sp
# 🔥 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


from config import load_config, generate_parameter_combinations, parse_scan_parameter
from geometry import build_grid2d_geometry
from spatial_field import build_prior, sample_gmrf, build_prior_with_ddi
from sensors import generate_sensor_pool
from inference import compute_posterior, compute_posterior_variance_diagonal, SparseFactor
from sensors import get_observation

from method_wrappers import get_selection_method, get_available_methods
from evaluation import spatial_block_cv, compute_metrics, morans_i

from visualization import (
    setup_style,
    generate_all_visualizations_v2,
    aggregate_results_for_visualization
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


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_arguments():
    """
    🔥 增强的命令行参数解析 - 支持参数扫描
    """
    parser = argparse.ArgumentParser(
        description='RDT-VoI 参数化实验框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：

# 基础使用
  python main.py                               # 使用默认配置
  python main.py --preset high_stakes          # 高风险场景
  python main.py --preset low_stakes           # 低风险场景

# 单参数调整  
  python main.py --ddi 0.30                    # 调整DDI
  python main.py --fn-cost 120000              # 调整误检成本
  python main.py --grid-size 25                # 调整网格大小
  python main.py --budgets 5,10,15,20          # 调整预算列表

# 参数扫描
  python main.py --scan ddi=0.1,0.2,0.3                    # DDI扫描
  python main.py --scan fn_cost=30000,60000,120000         # 成本扫描  
  python main.py --scan ddi=0.2,0.3 fn_cost=60000,120000  # 组合扫描

# 高级选项
  python main.py --parallel --workers 6        # 并行处理
  python main.py --quick-test                  # 快速测试
  python main.py --dry-run                     # 预览参数组合
        """
    )

    # 核心配置选项
    config_group = parser.add_argument_group('配置选项')
    config_group.add_argument(
        '--config', '-c', type=str, default='baseline_config.yaml',
        help='配置文件路径 (默认: baseline_config.yaml)'
    )
    config_group.add_argument(
        '--preset', '-p', type=str, choices=['high_stakes', 'low_stakes'],
        help='预设场景 (high_stakes=高风险, low_stakes=低风险)'
    )

    # 🔥 关键参数快速调整
    param_group = parser.add_argument_group('关键参数调整')
    param_group.add_argument(
        '--ddi', type=float,
        help='决策难度指数 (0.0-1.0, 典型值: 0.10-0.30)'
    )
    param_group.add_argument(
        '--fn-cost', '--fn_cost', type=float,
        help='误检成本 (£, 典型值: 30000-120000)'
    )
    param_group.add_argument(
        '--fp-cost', '--fp_cost', type=float,
        help='误报成本 (£, 典型值: 5000-30000)'
    )
    param_group.add_argument(
        '--tau-quantile', '--tau_quantile', type=float,
        help='阈值分位数 (0.0-1.0, 典型值: 0.65-0.88)'
    )
    param_group.add_argument(
        '--action-limit', '--K_action', type=int,
        help='行动限制 (整数, null=无限制)'
    )

    # 实验规模调整
    scale_group = parser.add_argument_group('实验规模')
    scale_group.add_argument(
        '--grid-size', '--grid_size', type=int,
        help='网格大小 (nx=ny, 典型值: 15-25)'
    )
    scale_group.add_argument(
        '--budgets', type=str,
        help='预算列表 (逗号分隔, 如: 5,10,15,20)'
    )
    scale_group.add_argument(
        '--methods', type=str,
        help='方法列表 (逗号分隔, 如: greedy_mi,greedy_evi,uniform)'
    )
    scale_group.add_argument(
        '--folds', '--k_folds', type=int,
        help='交叉验证折数 (典型值: 3-10)'
    )

    # 🔥 参数扫描功能
    scan_group = parser.add_argument_group('参数扫描')
    scan_group.add_argument(
        '--scan', type=str, nargs='+',
        help='参数扫描 (格式: param=val1,val2,val3). 例如: --scan ddi=0.1,0.2,0.3 fn_cost=30000,60000'
    )
    scan_group.add_argument(
        '--scan-presets', type=str, nargs='+',
        help='预设扫描 (如: --scan-presets high_stakes low_stakes)'
    )
    scan_group.add_argument(
        '--dry-run', action='store_true',
        help='仅显示参数组合，不执行实验'
    )

    # 执行控制
    exec_group = parser.add_argument_group('执行控制')
    exec_group.add_argument(
        '--parallel', action='store_true',
        help='启用并行处理'
    )
    exec_group.add_argument(
        '--workers', type=int, default=None,
        help='并行worker数量 (默认: CPU核心数-1)'
    )
    exec_group.add_argument(
        '--output', '-o', type=str, default=None,
        help='输出目录 (默认: 从配置读取)'
    )

    # 调试和测试
    debug_group = parser.add_argument_group('调试和测试')
    debug_group.add_argument(
        '--quick-test', action='store_true',
        help='快速测试模式 (小网格，少预算，少fold)'
    )
    debug_group.add_argument(
        '--skip-viz', action='store_true',
        help='跳过可视化生成'
    )
    debug_group.add_argument(
        '--seed', type=int, default=None,
        help='随机种子覆盖'
    )
    debug_group.add_argument(
        '-v', '--verbose', action='store_true',
        help='详细输出'
    )
    debug_group.add_argument(
        '-q', '--quiet', action='store_true',
        help='安静模式'
    )

    return parser.parse_args()


# ============================================================================
# 配置处理函数
# ============================================================================

def apply_cli_overrides(cfg, args):
    """
    🔥 应用命令行参数覆盖到配置
    """
    overrides = {}

    # 收集所有非空的CLI参数
    cli_mappings = {
        'ddi': 'target_ddi',
        'fn_cost': 'L_FN_gbp',
        'fp_cost': 'L_FP_gbp',
        'tau_quantile': 'tau_quantile',
        'action_limit': 'K_action',
        'grid_size': 'grid_size',
        'budgets': 'budgets',
        'methods': 'methods',
        'folds': 'k_folds',
        'seed': 'seed'
    }

    for cli_arg, config_key in cli_mappings.items():
        value = getattr(args, cli_arg, None)
        if value is not None:
            overrides[config_key] = value

    # 应用覆盖
    if overrides:
        if not args.quiet:
            print(f"\n📝 Applying CLI overrides: {overrides}")
        cfg = cfg.apply_parameter_overrides(overrides, verbose=not args.quiet)

    return cfg


def parse_scan_parameters(scan_args):
    """
    🔥 解析扫描参数

    Args:
        scan_args: ['ddi=0.1,0.2,0.3', 'fn_cost=30000,60000']

    Returns:
        {'ddi': [0.1, 0.2, 0.3], 'fn_cost': [30000, 60000]}
    """
    scan_params = {}

    for scan_spec in scan_args:
        if '=' not in scan_spec:
            raise ValueError(f"Invalid scan format: {scan_spec}. Use param=val1,val2,val3")

        param_name, values_str = scan_spec.split('=', 1)
        param_name = param_name.strip()

        # 解析值列表
        values = parse_scan_parameter(values_str)
        if not values:
            raise ValueError(f"No values found for parameter: {param_name}")

        scan_params[param_name] = values

    return scan_params


def create_experiment_configs(base_cfg, args):
    """
    🔥 创建实验配置列表（支持参数扫描）
    """
    configs = []

    # 情况1: 预设扫描
    if args.scan_presets:
        print(f"\n🔍 Preset scanning: {args.scan_presets}")
        for preset_name in args.scan_presets:
            try:
                preset_cfg = base_cfg.apply_preset(preset_name, verbose=not args.quiet)
                preset_cfg.experiment.name = f"{base_cfg.experiment.name}_{preset_name}"
                configs.append(preset_cfg)
            except Exception as e:
                print(f"❌ Failed to apply preset {preset_name}: {e}")
                continue

    # 情况2: 参数扫描
    elif args.scan:
        print(f"\n🔍 Parameter scanning: {args.scan}")
        scan_params = parse_scan_parameters(args.scan)
        combinations = generate_parameter_combinations(scan_params)

        print(f"📊 Generated {len(combinations)} parameter combinations")

        for i, combo in enumerate(combinations):
            combo_cfg = base_cfg.apply_parameter_overrides(combo, verbose=False)

            # 生成描述性名称
            combo_desc = "_".join(f"{k}{v}" for k, v in list(combo.items())[:3])  # 限制长度
            combo_cfg.experiment.name = f"{base_cfg.experiment.name}_scan_{combo_desc}"

            if not args.quiet:
                print(f"  {i+1}: {combo}")

            configs.append(combo_cfg)

    # 情况3: 单一配置
    else:
        configs.append(base_cfg)

    return configs


def detect_scenario_from_config(cfg) -> str:
    """从配置自动检测场景类型"""
    exp_name = cfg.experiment.name.lower()
    if 'high' in exp_name or 'stakes' in exp_name:
        return 'A'
    elif 'low' in exp_name or 'proxy' in exp_name:
        return 'B'

    ddi = getattr(cfg.decision, 'target_ddi', 0.0)
    fn_fp_ratio = cfg.decision.L_FN_gbp / cfg.decision.L_FP_gbp if cfg.decision.L_FP_gbp > 0 else 1.0

    if ddi >= 0.2 or fn_fp_ratio > 10:
        return 'A'
    elif ddi < 0.15 or fn_fp_ratio < 5:
        return 'B'

    return 'A'  # 默认


def create_output_dir_from_config(cfg, config_path: str, custom_output: str = None) -> Path:
    """根据配置文件名创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 从配置文件名提取场景标识
    config_name = Path(config_path).stem
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
    output_dir = base_dir / f"exp_{cfg.experiment.name}" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    (output_dir / "plots").mkdir(exist_ok=True)

    # 保存配置副本
    cfg.save_to(output_dir)

    # 保存运行命令
    with open(output_dir / "run_command.txt", 'w', encoding='utf-8') as f:
        f.write(' '.join(sys.argv))

    return output_dir


def apply_quick_test_overrides(cfg):
    """应用快速测试模式的覆盖"""
    print(f"\n🚀 Quick test mode enabled:")

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
# 🔥 核心函数1: run_single_fold_worker (完整版)
# ============================================================================

def run_single_fold_worker(fold_data: dict) -> dict:
    """
    ✅ 完全修复版：避免geom对象序列化，直接使用原始数据
    """
    import time
    import warnings
    import numpy as np
    import scipy.sparse as sp
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

    # 🔥 关键修复：直接使用标量，不构建geom对象
    n_domain = fold_data['n_domain']
    coords = fold_data['coords']
    adjacency_test_data = fold_data.get('adjacency_test')

    rng = np.random.default_rng(fold_data['rng_seed'])

    # 从config读取是否启用domain scaling
    enable_scaling = fold_data.get('enable_domain_scaling', True)

    # 检测场景类型
    scenario = fold_data.get('scenario', 'A')

    morans_permutations = fold_data.get('morans_permutations', 999)

    try:
        # ====================================================================
        # 1. 计算先验损失（用于ROI）
        # ====================================================================
        t_prior_start = time.time()
        tau = decision_config.get_threshold(mu_pr)
        factor_pr = SparseFactor(Q_pr)
        var_pr_test = compute_posterior_variance_diagonal(factor_pr, test_idx)
        sigma_pr_test = np.sqrt(np.maximum(var_pr_test, 1e-12))

        prior_loss_test = expected_loss(
            mu_pr[test_idx], sigma_pr_test, decision_config,
            test_indices=np.arange(len(test_idx)), tau=tau
        )

        # 🔥 Domain Scaling
        if enable_scaling:
            N_test = len(test_idx)
            scale_factor = n_domain / N_test
            prior_loss_scaled = prior_loss_test * scale_factor
        else:
            prior_loss_scaled = prior_loss_test
            scale_factor = 1.0

        prior_time = time.time() - t_prior_start

        # ====================================================================
        # 2. 传感器选择
        # ====================================================================
        t_sel_start = time.time()
        selection_result = selection_method(sensors, k, Q_pr, mu_pr)
        selection_time = time.time() - t_sel_start

        selected_sensors = [sensors[i] for i in selection_result.selected_ids]
        sensor_cost = selection_result.total_cost

        # ====================================================================
        # 3. 生成观测 + 计算后验
        # ====================================================================
        y, H, R = get_observation(x_true, selected_sensors, rng)

        t_inf_start = time.time()
        mu_post, factor_post = compute_posterior(Q_pr, mu_pr, H, R, y)
        inference_time = time.time() - t_inf_start

        var_post_test = compute_posterior_variance_diagonal(factor_post, test_idx)
        sigma_post_test = np.sqrt(np.maximum(var_post_test, 1e-12))
        sigma_post = np.zeros(len(mu_post))
        sigma_post[test_idx] = sigma_post_test

        # ====================================================================
        # 4. 基础指标
        # ====================================================================
        metrics = compute_metrics(mu_post, sigma_post, x_true, test_idx, decision_config)
        posterior_loss_test = metrics['expected_loss_gbp']

        if enable_scaling:
            posterior_loss_scaled = posterior_loss_test * scale_factor
        else:
            posterior_loss_scaled = posterior_loss_test

        # ROI计算
        savings_scaled = prior_loss_scaled - posterior_loss_scaled

        if sensor_cost > 0:
            roi = (savings_scaled - sensor_cost) / sensor_cost
            cost_efficiency = savings_scaled / sensor_cost
        else:
            roi = np.inf if savings_scaled > 0 else 0.0
            cost_efficiency = np.inf if savings_scaled > 0 else 0.0

        # ====================================================================
        # 5. Scenario A 特有：Near-threshold 子集评估
        # ====================================================================
        near_threshold_metrics = {}
        if scenario == 'A':
            try:
                gaps_prior = np.abs(mu_pr[test_idx] - tau)
                near_mask = gaps_prior <= 1.0 * sigma_pr_test

                if near_mask.sum() > 0:
                    prior_loss_near = expected_loss(
                        mu_pr[test_idx][near_mask],
                        sigma_pr_test[near_mask],
                        decision_config,
                        test_indices=np.arange(near_mask.sum()),
                        tau=tau
                    )

                    posterior_loss_near = expected_loss(
                        mu_post[test_idx][near_mask],
                        sigma_post_test[near_mask],
                        decision_config,
                        test_indices=np.arange(near_mask.sum()),
                        tau=tau
                    )

                    if enable_scaling:
                        prior_loss_near *= scale_factor
                        posterior_loss_near *= scale_factor

                    savings_near = prior_loss_near - posterior_loss_near
                    roi_near = (savings_near - sensor_cost) / sensor_cost if sensor_cost > 0 else 0.0

                    near_threshold_metrics = {
                        'n_near_threshold': int(near_mask.sum()),
                        'fraction_near_threshold': float(near_mask.sum() / len(test_idx)),
                        'prior_loss_near_threshold': float(prior_loss_near),
                        'posterior_loss_near_threshold': float(posterior_loss_near),
                        'savings_near_threshold': float(savings_near),
                        'roi_near_threshold': float(roi_near)
                    }
            except Exception as e:
                warnings.warn(f"Near-threshold evaluation failed: {e}")

        # 记录完整指标
        metrics.update({
            'roi': float(roi),
            'cost_efficiency': float(cost_efficiency),
            'prior_loss_gbp': float(prior_loss_scaled),
            'posterior_loss_gbp': float(posterior_loss_scaled),
            'savings_gbp': float(savings_scaled),
            'total_cost': float(sensor_cost),
            'prior_loss_test_only': float(prior_loss_test),
            'domain_scale_factor': float(scale_factor),
            **near_threshold_metrics
        })

        # ====================================================================
        # 6. DDI统计
        # ====================================================================
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

        # ====================================================================
        # 7. Scenario A 特有：行动受限评估
        # ====================================================================
        if scenario == 'A' and hasattr(decision_config, 'K_action') and decision_config.K_action is not None:
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
                metrics['action_regret'] = float(constrained_loss - posterior_loss_test)
                metrics['action_hit_rate'] = float(hit_rate)
            except Exception as e:
                warnings.warn(f"Action-constrained evaluation failed: {e}")

        # ====================================================================
        # 8. Moran's I
        # ====================================================================
        residuals = mu_post - x_true
        if adjacency_test_data is not None:
            try:
                adj_test = sp.coo_matrix(
                    (adjacency_test_data['data'],
                     (adjacency_test_data['row'], adjacency_test_data['col'])),
                    shape=adjacency_test_data['shape']
                ).tocsr()

                I_stat, I_pval = morans_i(
                    residuals[test_idx],
                    adj_test,
                    n_permutations=morans_permutations,
                    rng=rng
                )
                metrics['morans_i'] = float(I_stat)
                metrics['morans_pval'] = float(I_pval)
            except Exception as e:
                warnings.warn(f"Moran's I computation failed: {e}")

        # ====================================================================
        # 9. 时间统计
        # ====================================================================
        metrics['prior_computation_time_sec'] = float(prior_time)
        metrics['selection_time_sec'] = float(selection_time)
        metrics['inference_time_sec'] = float(inference_time)
        metrics['total_time_sec'] = float(prior_time + selection_time + inference_time)

        # ====================================================================
        # 10. 传感器诊断
        # ====================================================================
        metrics['n_selected'] = len(selection_result.selected_ids)

        type_counts = {}
        for sid in selection_result.selected_ids:
            stype = sensors[sid].type_name
            type_counts[stype] = type_counts.get(stype, 0) + 1
        metrics['type_counts'] = {k: int(v) for k, v in type_counts.items()}

        selected_costs = [sensors[i].cost for i in selection_result.selected_ids]
        metrics['cost_mean'] = float(np.mean(selected_costs))
        metrics['cost_std'] = float(np.std(selected_costs))

        if 'coverage_90' in metrics:
            metrics['coverage_90'] = float(np.clip(metrics['coverage_90'], 0.0, 1.0))

        return {
            'success': True,
            'metrics': metrics,
            'selection_result': selection_result,
            'mu_post': mu_post,
            'sigma_post': sigma_post,
            'residuals': mu_post[test_idx] - x_true[test_idx],
            'test_idx': test_idx,
            'tau': tau,
            'prior_loss': prior_loss_scaled,
            'posterior_loss': posterior_loss_scaled,
            'savings': savings_scaled,
            'roi': roi
        }

    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


# ============================================================================
# 🔥 核心函数2: run_method_evaluation (完整版)
# ============================================================================

def run_method_evaluation(method_name: str, cfg, geom, Q_pr, mu_pr,
                          x_true, sensors, test_idx_global=None,
                          use_parallel=False, n_workers=None, verbose=True) -> dict:
    """
    运行方法评估（支持并行处理）
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

    # 检测场景类型
    if hasattr(cfg.decision, 'target_ddi'):
        if cfg.decision.target_ddi >= 0.20:
            scenario = 'A'
        else:
            scenario = 'B'
    else:
        scenario = 'A'

    enable_scaling = getattr(cfg.metrics, 'scale_savings_to_domain', True) if hasattr(cfg, 'metrics') else True

    # 🔥 关键修复：提取必要的标量和可序列化数据
    n_domain = geom.n
    coords = geom.coords

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
            # 🔥 关键：提取test_idx对应的adjacency子矩阵
            try:
                adj_test_submatrix = geom.adjacency[test_idx][:, test_idx]
                adj_test_coo = adj_test_submatrix.tocoo()
                adjacency_data = {
                    'data': adj_test_coo.data,
                    'row': adj_test_coo.row,
                    'col': adj_test_coo.col,
                    'shape': adj_test_coo.shape
                }
            except Exception as e:
                if verbose:
                    print(f"    Warning: Failed to extract adjacency submatrix: {e}")
                adjacency_data = None

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
                'n_domain': n_domain,
                'coords': coords,
                'adjacency_test': adjacency_data,
                'rng_seed': rng.integers(0, 2 ** 31),
                'enable_domain_scaling': enable_scaling,
                'scenario': scenario,
                'morans_permutations': cfg.cv.morans_permutations if hasattr(cfg.cv, 'morans_permutations') else 999,
                'verbose': verbose
            }
            fold_data_list.append((fold_idx, fold_data))

        # 并行或串行执行
        if use_parallel and len(fold_data_list) > 1:
            if n_workers is None:
                n_workers = max(1, mp.cpu_count() - 1)

            if verbose:
                print(f"    Running {len(fold_data_list)} folds in parallel "
                      f"with {n_workers} workers...")

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_fold = {
                    executor.submit(run_single_fold_worker, fold_data): fold_idx
                    for fold_idx, fold_data in fold_data_list
                }

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


# ============================================================================
# 🔥 核心函数3: run_single_experiment
# ============================================================================

def run_single_experiment(cfg, args, exp_index=None, total_experiments=None):
    """
    🔥 运行单个实验配置（完整修复版）

    修复要点：
    1. 先应用DDI控制（如果需要），然后锁定阈值
    2. 基于最终的先验分布锁定阈值，而不是初始先验
    3. 完善的异常检测和诊断
    """
    exp_prefix = f"[{exp_index + 1}/{total_experiments}] " if exp_index is not None else ""

    if not args.quiet:
        print(f"\n{'=' * 70}")
        print(f"  {exp_prefix}EXPERIMENT: {cfg.experiment.name}")
        print(f"{'=' * 70}")

    t_start = datetime.now()

    # 创建输出目录
    output_dir = create_output_dir_from_config(
        cfg,
        args.config or "baseline_config.yaml",
        args.output
    )
    if not args.quiet:
        print(f"\n📁 Output: {output_dir}")

    # 快速测试模式调整
    if args.quick_test:
        cfg = apply_quick_test_overrides(cfg)

    rng = cfg.get_rng()

    # ========================================================================
    # 🔥 【核心修复】构建域和先验的正确顺序
    # ========================================================================

    if not args.quiet:
        print(f"\n🌐 Building domain: {cfg.geometry.nx}×{cfg.geometry.ny}")
    geom = build_grid2d_geometry(cfg.geometry.nx, cfg.geometry.ny, cfg.geometry.h)

    if not args.quiet:
        print(f"🔧 Building prior...")

    # 步骤1：判断是否需要DDI控制
    use_ddi = (hasattr(cfg.decision, 'target_ddi') and
               cfg.decision.target_ddi is not None and
               cfg.decision.target_ddi > 0)

    if use_ddi:
        # ====================================================================
        # 🔥 情况A：需要DDI控制
        # 顺序：构建初始先验 → DDI控制 → 锁定阈值
        # ====================================================================

        # 1. 构建初始先验
        Q_temp, mu_temp = build_prior(geom, cfg.prior)

        # 2. 计算临时阈值（用于DDI控制，不锁定）
        if hasattr(cfg.decision, 'tau_quantile') and cfg.decision.tau_quantile is not None:
            tau_temp = float(np.quantile(mu_temp, cfg.decision.tau_quantile))
            if not args.quiet:
                print(f"  📊 Initial prior for DDI control:")
                print(f"     mean={mu_temp.mean():.3f}, std={mu_temp.std():.3f}")
                print(f"     Using tau_quantile={cfg.decision.tau_quantile:.2f} "
                      f"→ τ_temp={tau_temp:.3f}")
        else:
            # 回退到成本映射
            p_T = cfg.decision.prob_threshold
            tau_temp = float(np.quantile(mu_temp, p_T))
            if not args.quiet:
                print(f"  📊 Using cost-based threshold: p_T={p_T:.3f} "
                      f"→ τ_temp={tau_temp:.3f}")

        # 3. 应用DDI控制
        if not args.quiet:
            print(f"  🎯 Applying DDI control (target={cfg.decision.target_ddi:.1%})...")

        try:
            Q_pr, mu_pr = build_prior_with_ddi(
                geom, cfg.prior, tau=tau_temp, target_ddi=cfg.decision.target_ddi
            )

            if not args.quiet:
                print(f"  ✓ DDI control applied")
                print(f"     Final prior: mean={mu_pr.mean():.3f}, std={mu_pr.std():.3f}")

        except Exception as e:
            if not args.quiet:
                print(f"  ⚠️  DDI control failed: {e}")
                print(f"  Falling back to standard prior without DDI control")
            Q_pr, mu_pr = Q_temp, mu_temp
            use_ddi = False  # 标记DDI控制失败

        # 4. 🔥 关键：基于DDI控制后的最终先验锁定阈值
        if not args.quiet:
            print(f"  🔒 Locking threshold based on DDI-adjusted prior...")
        cfg.lock_decision_threshold(mu_pr, verbose=not args.quiet)

    else:
        # ====================================================================
        # 🔥 情况B：不需要DDI控制
        # 顺序：构建先验 → 锁定阈值
        # ====================================================================

        Q_pr, mu_pr = build_prior(geom, cfg.prior)
        cfg.lock_decision_threshold(mu_pr, verbose=not args.quiet)

    # ========================================================================
    # 健康检查和诊断
    # ========================================================================

    tau = cfg.decision.tau_iri

    # 计算先验统计信息
    mu_stats = {
        'min': mu_pr.min(),
        'max': mu_pr.max(),
        'mean': mu_pr.mean(),
        'median': np.median(mu_pr),
        'std': mu_pr.std(),
        'q10': np.quantile(mu_pr, 0.1),
        'q50': np.quantile(mu_pr, 0.5),
        'q90': np.quantile(mu_pr, 0.9),
    }

    # 检查1：阈值是否在合理范围
    threshold_issues = []

    if tau < 0:
        threshold_issues.append(f"Threshold is negative (τ={tau:.3f})")
    elif tau > 5:
        threshold_issues.append(f"Threshold exceeds typical IRI range (τ={tau:.3f} > 5)")

    # 检查2：阈值是否与先验分布匹配
    if tau < mu_stats['q10']:
        threshold_issues.append(f"Threshold below 10th percentile ({tau:.3f} < {mu_stats['q10']:.3f})")
    elif tau > mu_stats['max']:
        threshold_issues.append(f"Threshold exceeds maximum value ({tau:.3f} > {mu_stats['max']:.3f})")

    # 检查3：先验分布是否合理
    if mu_stats['mean'] < -2 or mu_stats['mean'] > 5:
        threshold_issues.append(f"Prior mean unusual ({mu_stats['mean']:.3f})")

    if mu_stats['median'] < -1 or mu_stats['median'] > 4:
        threshold_issues.append(f"Prior median unusual ({mu_stats['median']:.3f})")

    # 如果有问题，显示详细诊断
    if threshold_issues:
        print(f"\n  ⚠️  THRESHOLD DIAGNOSTICS")
        print(f"  {'=' * 68}")
        print(f"  Locked threshold: τ = {tau:.3f}")
        print(f"\n  Issues detected:")
        for issue in threshold_issues:
            print(f"    • {issue}")

        print(f"\n  📊 Prior distribution:")
        print(f"    Range: [{mu_stats['min']:.3f}, {mu_stats['max']:.3f}]")
        print(f"    Mean: {mu_stats['mean']:.3f}, Median: {mu_stats['median']:.3f}, Std: {mu_stats['std']:.3f}")
        print(f"    Quantiles: p10={mu_stats['q10']:.3f}, p50={mu_stats['q50']:.3f}, p90={mu_stats['q90']:.3f}")

        if use_ddi:
            print(f"\n  ℹ️  DDI control was applied (target={cfg.decision.target_ddi:.1%})")
            print(f"  Recommendations:")
            print(f"    1. Lower target_ddi (try 0.10-0.20 instead of {cfg.decision.target_ddi:.2f})")
            print(f"    2. Disable DDI control (set target_ddi: null)")
            print(f"    3. Adjust tau_quantile (try 0.75-0.80 instead of current value)")
            print(f"    4. Modify prior.mu_prior_mean to center distribution better")
        else:
            print(f"\n  ℹ️  No DDI control")
            print(f"  Recommendations:")
            print(f"    1. Check prior.mu_prior_mean in config (affects distribution center)")
            print(f"    2. Adjust tau_quantile (try lower values like 0.75)")
            print(f"    3. Verify prior variance settings")

        print(f"  {'=' * 68}")

        # 严重问题时可以选择终止
        if tau < -5 or tau > 10:
            print(f"\n  ❌ CRITICAL: Threshold extremely unusual, aborting experiment")
            print(f"  Please fix configuration before proceeding")
            sys.exit(1)

    if not args.quiet:
        print(f"✅ Prior setup complete (τ={tau:.3f})")

    # ========================================================================
    # 计算完整的先验标准差（用于后续评估）
    # ========================================================================

    from inference import SparseFactor, compute_posterior_variance_diagonal

    factor_pr = SparseFactor(Q_pr)
    var_pr = compute_posterior_variance_diagonal(factor_pr, indices=None)
    sigma_pr = np.sqrt(np.maximum(var_pr, 1e-12))

    # 可选：获取域缩放因子
    if hasattr(cfg, 'economics') and cfg.economics is not None:
        scale_factor = cfg.get_domain_scale_factor(verbose=not args.quiet)
    else:
        scale_factor = 1.0

    # ========================================================================
    # 生成真实状态和传感器
    # ========================================================================

    x_true = sample_gmrf(Q_pr, mu_pr, rng)
    np.save(output_dir / 'x_true.npy', x_true)

    sensors = generate_sensor_pool(geom, cfg.sensors, rng)

    if not args.quiet:
        print(f"  Generated {len(sensors)} heterogeneous sensors:")

        # 传感器类型统计
        type_counts = {}
        for s in sensors:
            type_counts[s.type_name] = type_counts.get(s.type_name, 0) + 1

        print(f"    Type distribution:")
        for stype, count in sorted(type_counts.items()):
            print(f"      {stype}: {count} ({count / len(sensors) * 100:.1f}%)")

        costs = [s.cost for s in sensors]
        noises = [s.noise_var ** 0.5 for s in sensors]
        print(f"    Cost range: £{min(costs):.0f} - £{max(costs):.0f}")
        print(f"    Noise std range: {min(noises):.3f} - {max(noises):.3f}")

    # 全局测试集
    n_test = min(200, geom.n)
    test_idx_global = rng.choice(geom.n, size=n_test, replace=False)

    # ========================================================================
    # 运行方法评估
    # ========================================================================

    if not args.quiet:
        print(f"\n🚀 Running methods: {', '.join(cfg.selection.methods)}")

    all_results = {}
    methods = get_available_methods(cfg)

    for method_name in methods:
        method_start = datetime.now()
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
                use_parallel=args.parallel,
                n_workers=args.workers,
                verbose=not args.quiet
            )
            all_results[method_name] = results

            method_elapsed = (datetime.now() - method_start).total_seconds()
            if not args.quiet:
                print(f"✅ {method_name} completed in {method_elapsed:.1f}s")
        except Exception as e:
            if not args.quiet:
                print(f"❌ {method_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # ========================================================================
    # 保存结果
    # ========================================================================

    import pickle
    with open(output_dir / 'results_raw.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # 转换为DataFrame
    try:
        df_results = aggregate_results_for_visualization(all_results)
        if not df_results.empty:
            df_results.to_csv(output_dir / 'results_aggregated.csv', index=False)
            if not args.quiet:
                print(f"💾 Saved {len(df_results)} result rows")
    except Exception as e:
        if not args.quiet:
            print(f"⚠️ DataFrame conversion failed: {e}")
        df_results = pd.DataFrame()

    # ========================================================================
    # 可视化
    # ========================================================================

    if not args.skip_viz and not df_results.empty:
        if not args.quiet:
            print(f"\n📊 Generating visualizations...")
        try:
            scenario = detect_scenario_from_config(cfg)
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
            if not args.quiet:
                print(f"✅ Visualization complete")
        except Exception as e:
            if not args.quiet:
                print(f"❌ Visualization failed: {str(e)}")

    # ========================================================================
    # 实验总结
    # ========================================================================

    total_elapsed = (datetime.now() - t_start).total_seconds()
    if not args.quiet:
        print(f"\n{exp_prefix}✅ Experiment completed in {total_elapsed:.1f}s")
        print(f"📁 Results saved to: {output_dir}")

    return {
        'config': cfg,
        'output_dir': output_dir,
        'results': all_results,
        'elapsed_time': total_elapsed,
        'success': len(all_results) > 0,
        'threshold': tau,
        'prior_stats': mu_stats,
        'domain_scale_factor': scale_factor
    }

# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    🔥 增强的主函数 - 支持参数扫描和灵活配置
    """
    args = parse_arguments()

    verbose = not args.quiet

    if verbose:
        print("=" * 70)
        print("  RDT-VoI 参数化实验框架")
        print("=" * 70)

    # 1. 加载基础配置
    try:
        base_cfg = load_config(args.config)
        if verbose:
            print(f"✅ Loaded config: {args.config}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

    # 2. 应用预设
    if args.preset:
        try:
            base_cfg = base_cfg.apply_preset(args.preset, verbose=verbose)
            if verbose:
                print(f"✅ Applied preset: {args.preset}")
        except Exception as e:
            print(f"❌ Failed to apply preset {args.preset}: {e}")
            sys.exit(1)

    # 3. 应用命令行覆盖
    base_cfg = apply_cli_overrides(base_cfg, args)

    # 4. 创建实验配置列表
    try:
        configs = create_experiment_configs(base_cfg, args)
        if verbose:
            print(f"📋 Created {len(configs)} experiment configuration(s)")
    except Exception as e:
        print(f"❌ Failed to create experiment configs: {e}")
        sys.exit(1)

    # 5. Dry run模式
    if args.dry_run:
        print(f"\n🔍 DRY RUN - Parameter combinations:")
        for i, cfg in enumerate(configs):
            print(f"\n  Experiment {i+1}: {cfg.experiment.name}")
            print(f"    DDI: {getattr(cfg.decision, 'target_ddi', 'N/A')}")
            print(f"    L_FN: £{cfg.decision.L_FN_gbp:,.0f}")
            print(f"    L_FP: £{cfg.decision.L_FP_gbp:,.0f}")
            print(f"    Budgets: {cfg.selection.budgets}")
            print(f"    Methods: {cfg.selection.methods}")
        print(f"\n✅ Dry run complete. Use without --dry-run to execute.")
        return

    # 6. 执行实验
    successful_experiments = []
    failed_experiments = []

    total_start = datetime.now()

    for i, cfg in enumerate(configs):
        try:
            result = run_single_experiment(cfg, args, exp_index=i, total_experiments=len(configs))
            if result['success']:
                successful_experiments.append(result)
            else:
                failed_experiments.append(result)
        except Exception as e:
            if verbose:
                print(f"❌ Experiment {i+1} failed: {str(e)}")
            failed_experiments.append({
                'config': cfg,
                'error': str(e),
                'success': False
            })
            import traceback
            traceback.print_exc()

    # 7. 总结报告
    total_elapsed = (datetime.now() - total_start).total_seconds()

    if verbose:
        print(f"\n" + "=" * 70)
        print(f"  EXPERIMENT SUMMARY")
        print(f"=" * 70)
        print(f"✅ Successful: {len(successful_experiments)}")
        print(f"❌ Failed: {len(failed_experiments)}")
        print(f"⏱️  Total time: {total_elapsed:.1f}s")

        if successful_experiments:
            print(f"\n📁 Output directories:")
            for result in successful_experiments:
                print(f"  - {result['output_dir']}")

    # 退出码
    if failed_experiments and not successful_experiments:
        sys.exit(1)
    elif failed_experiments:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()