"""
Main experimental script for RDT-VoI simulation.
Runs complete benchmarking suite with spatial CV.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from geometry import build_grid2d_geometry
from spatial_field import build_prior, sample_gmrf
from sensors import generate_sensor_pool
from selection import greedy_mi, greedy_aopt, uniform_selection, random_selection
from evaluation import run_cv_experiment, spatial_bootstrap, spatial_block_cv
from visualization import (setup_style, plot_budget_curves, plot_marginal_mi,
                          plot_residual_map, plot_performance_profile,
                          plot_critical_difference)


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
def get_selection_method(method_name: str, geom, rng_seed: int = None):
    """
    获取选择方法的工厂函数（硬编码优化参数版本）

    Args:
        method_name: 方法名称 ("Greedy-MI" | "Greedy-A" | "Uniform" | "Random")
        geom: 几何对象
        rng_seed: 随机种子

    Returns:
        selection_function: (sensors, k, Q_pr) -> SelectionResult
    """
    import numpy as np
    from selection import greedy_mi, greedy_aopt, uniform_selection, random_selection

    if method_name == "Greedy-MI":
        def mi_wrapper(sensors, k, Q_pr):
            costs = np.array([s.cost for s in sensors])
            return greedy_mi(
                sensors=sensors,
                k=k,
                Q_pr=Q_pr,
                costs=costs,
                lazy=True,
                batch_size=64
            )

        return mi_wrapper

    elif method_name == "Greedy-A":
        def aopt_wrapper(sensors, k, Q_pr):
            costs = np.array([s.cost for s in sensors])
            return greedy_aopt(
                sensors=sensors,
                k=k,
                Q_pr=Q_pr,
                costs=costs,
                hutchpp_probes=3,  # 🔥 加速：从20减到3
                batch_size=8,  # 🔥 加速：从32减到8
                max_candidates=50,  # 🔥 加速：只评估50个候选
                early_stop_ratio=0.3  # 🔥 加速：早停阈值
            )

        return aopt_wrapper

    elif method_name == "Uniform":
        return lambda s, k, Q: uniform_selection(s, k, geom)

    elif method_name == "Random":
        rng = np.random.default_rng(rng_seed)
        return lambda s, k, Q: random_selection(s, k, rng)

    else:
        raise ValueError(f"Unknown method: {method_name}")

def run_single_fold_worker(args):
    """
    单个 fold 的工作函数（顶层函数，可以被 pickle）
    """
    (fold_idx, train_idx, test_idx, geom_dict, Q_pr_data, mu_pr, x_true,
     sensors_data, method_name, k, cv_dict, decision_config_dict, seed) = args

    # 在子进程中重建对象
    import scipy.sparse as sp
    from geometry import Geometry
    from config import DecisionConfig
    from inference import compute_posterior, compute_posterior_variance_diagonal
    from sensors import Sensor, get_observation
    from evaluation import compute_metrics, morans_i

    # 重建 Geometry 对象
    geom = Geometry(
        mode=geom_dict['mode'],
        n=geom_dict['n'],
        coords=geom_dict['coords'],
        adjacency=sp.csr_matrix(
            (geom_dict['adj_data'], geom_dict['adj_indices'], geom_dict['adj_indptr']),
            shape=(geom_dict['n'], geom_dict['n'])
        ),
        laplacian=sp.csr_matrix(
            (geom_dict['lap_data'], geom_dict['lap_indices'], geom_dict['lap_indptr']),
            shape=(geom_dict['n'], geom_dict['n'])
        ),
        h=geom_dict.get('h')
    )

    # 重建 Q_pr
    Q_pr = sp.csr_matrix(
        (Q_pr_data['data'], Q_pr_data['indices'], Q_pr_data['indptr']),
        shape=(Q_pr_data['shape'][0], Q_pr_data['shape'][1])
    )

    # 重建传感器
    sensors = []
    for s_data in sensors_data:
        sensors.append(Sensor(
            id=s_data['id'],
            idxs=s_data['idxs'],
            weights=s_data['weights'],
            noise_var=s_data['noise_var'],
            cost=s_data['cost'],
            type_name=s_data['type_name']
        ))

    # 重建 decision_config
    decision_config = DecisionConfig(**decision_config_dict)

    # 创建 RNG
    rng = np.random.default_rng(seed + fold_idx)

    # 获取选择方法
    method_func = get_selection_method(method_name, geom, seed + fold_idx)

    print(f"    Fold {fold_idx+1}: Starting...")

    # 选择传感器
    selection_result = method_func(sensors, k, Q_pr)
    selected_sensors = [sensors[i] for i in selection_result.selected_ids]

    # 生成观测
    y, H, R = get_observation(x_true, selected_sensors, rng)

    # 计算后验
    mu_post, factor = compute_posterior(Q_pr, mu_pr, H, R, y)

    # 后验方差
    var_post_test = compute_posterior_variance_diagonal(factor, test_idx)
    sigma_post_test = np.sqrt(var_post_test)

    sigma_post = np.zeros(len(mu_post))
    sigma_post[test_idx] = sigma_post_test

    # 计算指标
    metrics = compute_metrics(
        mu_post, sigma_post, x_true, test_idx, decision_config
    )

    # Moran's I
    residuals = mu_post - x_true
    test_adjacency = geom.adjacency[test_idx][:, test_idx]
    I_stat, I_pval = morans_i(
        residuals[test_idx],
        test_adjacency,
        n_permutations=cv_dict.get('morans_permutations', 999),
        rng=rng
    )
    metrics['morans_i'] = I_stat
    metrics['morans_pval'] = I_pval

    print(f"    Fold {fold_idx+1}: RMSE={metrics['rmse']:.3f}, Loss=£{metrics['expected_loss_gbp']:.0f}")

    return fold_idx, metrics


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
                               decision_config, seed, n_workers=None):
    """
    并行版本的 CV 实验

    Args:
        method_name: 方法名称字符串（如 "Greedy-MI"）
    """
    if hasattr(cv_config, '__dict__'):
        cv_dict = cv_config.__dict__
    else:
        cv_dict = cv_config

    # 生成 CV folds
    corr_length = np.sqrt(8.0) / 0.08
    buffer_width = cv_dict.get('buffer_width_multiplier', 1.5) * corr_length

    folds = spatial_block_cv(
        geom.coords,
        cv_dict.get('k_folds', 5),
        buffer_width,
        cv_dict.get('block_strategy', 'kmeans'),
        np.random.default_rng(seed)
    )

    # 序列化数据
    geom_dict = serialize_geometry(geom)
    Q_pr_data = serialize_sparse_matrix(Q_pr)
    sensors_data = serialize_sensors(sensors)
    decision_dict = decision_config.__dict__

    # 准备并行任务
    tasks = [
        (fold_idx, train_idx, test_idx, geom_dict, Q_pr_data, mu_pr, x_true,
         sensors_data, method_name, k, cv_dict, decision_dict, seed)
        for fold_idx, (train_idx, test_idx) in enumerate(folds)
    ]

    # 并行执行
    if n_workers is None:
        n_workers = min(len(folds), max(1, mp.cpu_count() - 1))

    print(f"\n  Running {len(folds)} folds in parallel with {n_workers} workers...")

    fold_results = [None] * len(folds)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_fold_worker, task): task[0]
                   for task in tasks}

        for future in as_completed(futures):
            fold_idx, metrics = future.result()
            fold_results[fold_idx] = metrics

    # 聚合结果
    aggregated = {}
    for key in fold_results[0].keys():
        values = np.array([fr[key] for fr in fold_results])
        aggregated[key] = {
            'mean': values.mean(),
            'std': values.std(),
            'values': values
        }

    # 返回一个简化的 selection_result（因为我们不需要它的详细信息）
    from selection import SelectionResult
    selection_result = SelectionResult(
        selected_ids=[],
        objective_values=[],
        marginal_gains=[],
        total_cost=0.0,
        method_name=method_name
    )

    return {
        'fold_results': fold_results,
        'aggregated': aggregated,
        'selection_result': selection_result
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
                geom, Q_pr, mu_pr, x_true, sensors,
                method_name, k,
                cfg.cv, cfg.decision, cfg.experiment.seed,
                n_workers=5  # 使用 5 个worker（对应 5 个 fold）
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
    生成专家建议的所有图像

    Args:
        all_results: 所有方法的实验结果
        cfg: 配置对象
        geom: 几何对象
        sensors: 传感器列表
        output_dir: 输出目录
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
        # 重组数据为 plot_budget_curves 需要的格式
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

    # ========== F2: 单位成本效率 ==========
    print("\n💰 F2: Marginal Efficiency Curves...")
    try:
        for method_name in ['Greedy-MI', 'Greedy-A']:
            if method_name not in all_results:
                continue

            # 获取最大预算的结果（包含完整的选择历史）
            max_budget = max(all_results[method_name].keys())
            result_data = all_results[method_name][max_budget]

            # 从第一个fold获取selection_result
            if 'fold_results' in result_data and len(result_data['fold_results']) > 0:
                fold_0 = result_data['fold_results'][0]
                if 'selection_result' in fold_0:
                    selection_result = fold_0['selection_result']

                    plot_marginal_efficiency(
                        selection_result,
                        sensors,
                        output_path=curves_dir / f"f2_efficiency_{method_name.lower().replace('-', '_')}.png"
                    )
        print("  ✅ F2 saved")
    except Exception as e:
        print(f"  ❌ F2 failed: {e}")

    # ========== F3: 类型堆叠图 ==========
    print("\n🎨 F3: Type Composition...")
    try:
        for method_name in ['Greedy-MI']:
            if method_name not in all_results:
                continue

            max_budget = max(all_results[method_name].keys())
            result_data = all_results[method_name][max_budget]

            if 'fold_results' in result_data and len(result_data['fold_results']) > 0:
                fold_0 = result_data['fold_results'][0]
                if 'selection_result' in fold_0:
                    selection_result = fold_0['selection_result']

                    plot_type_composition(
                        selection_result,
                        sensors,
                        output_path=curves_dir / f"f3_type_composition_{method_name.lower().replace('-', '_')}.png"
                    )
        print("  ✅ F3 saved")
    except Exception as e:
        print(f"  ❌ F3 failed: {e}")

    # ========== F4: MI vs VoI 相关性 ==========
    print("\n🔗 F4: MI vs VoI Correlation...")
    try:
        # 需要收集MI和EVI数据
        # 这需要在实验过程中保存EVI值
        # 如果没有EVI数据，跳过
        print("  ⏭️  F4 skipped (requires EVI computation)")
    except Exception as e:
        print(f"  ❌ F4 failed: {e}")

    # ========== F5: 校准诊断 ==========
    print("\n📊 F5: Calibration Diagnostics...")
    try:
        for method_name in ['Greedy-MI']:
            if method_name not in all_results:
                continue

            # 收集所有预算的fold结果
            all_fold_results = []
            for budget in all_results[method_name].keys():
                result_data = all_results[method_name][budget]
                if 'fold_results' in result_data:
                    all_fold_results.extend(result_data['fold_results'])

            if all_fold_results:
                plot_calibration_diagnostics(
                    all_fold_results,
                    output_path=calibration_dir / f"f5_calibration_{method_name.lower().replace('-', '_')}.png"
                )
        print("  ✅ F5 saved")
    except Exception as e:
        print(f"  ❌ F5 failed: {e}")

    # ========== F6: 空间诊断 ==========
    print("\n🗺️  F6: Spatial Diagnostics...")
    try:
        for method_name in ['Greedy-MI', 'Random']:
            if method_name not in all_results:
                continue

            # 使用中等预算的结果
            budgets = list(all_results[method_name].keys())
            mid_budget = budgets[len(budgets) // 2]
            result_data = all_results[method_name][mid_budget]

            if 'fold_results' in result_data and len(result_data['fold_results']) > 0:
                fold_0 = result_data['fold_results'][0]

                # 需要从fold中提取 mu_post, x_true, test_idx
                # 这些数据需要在实验时保存
                print(f"  ⏭️  F6 for {method_name} skipped (needs mu_post/x_true from folds)")
        print("  ⚠️  F6 partially implemented")
    except Exception as e:
        print(f"  ❌ F6 failed: {e}")

    # ========== F7: 性能剖面 + CD图 ==========
    print("\n📉 F7: Performance Profile & Critical Difference...")
    try:
        # 重组数据为性能剖面格式
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

    # ========== F8: 消融实验 ==========
    print("\n🔬 F8: Ablation Study...")
    print("  ⏭️  F8 skipped (requires separate experiments)")

    # ========== F9: 复杂度分析 ==========
    print("\n⏱️  F9: Complexity Analysis...")
    print("  ⏭️  F9 skipped (requires timing instrumentation)")

    # ========== F10: 选址地图 ==========
    print("\n🗺️  F10: Sensor Placement Maps...")
    try:
        for method_name in ['Greedy-MI']:
            if method_name not in all_results:
                continue

            for budget in all_results[method_name].keys():
                result_data = all_results[method_name][budget]

                if 'fold_results' in result_data and len(result_data['fold_results']) > 0:
                    fold_0 = result_data['fold_results'][0]
                    if 'selection_result' in fold_0:
                        selection_result = fold_0['selection_result']

                        plot_sensor_placement_map(
                            geom.coords,
                            selection_result.selected_ids,
                            sensors,
                            output_path=maps_dir / f"f10_placement_{method_name.lower().replace('-', '_')}_k{budget}.png"
                        )
        print("  ✅ F10 saved")
    except Exception as e:
        print(f"  ❌ F10 failed: {e}")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\n📂 All plots saved to: {output_dir}")
    print(f"   - Curves: {curves_dir}")
    print(f"   - Maps: {maps_dir}")
    print(f"   - Calibration: {calibration_dir}")
    print(f"   - Comparison: {comparison_dir}")

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
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='RDT-VoI Simulation')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--milestone', choices=['m1', 'm2', 'full'],
                       default='full', help='Which milestone to run')
    args = parser.parse_args()

    # Load configuration
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"\n⌫ Error: {e}")
        print("\nPlease create config.yaml in project root.")
        print("You can copy from the provided template.")
        return 1

    # Run experiment
    try:
        run_full_experiment(cfg)
        return 0
    except Exception as e:
        print(f"\n⌫ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()