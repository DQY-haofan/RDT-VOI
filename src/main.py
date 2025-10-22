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


def run_single_fold_worker(method_name, geom, Q_pr, mu_pr, x_true,
                           sensors, k, train_idx, test_idx,
                           decision_config, fold_idx, seed):
    """
    单个fold的worker函数

    Args:
        method_name: 方法名称
        geom: 几何对象
        Q_pr, mu_pr: 先验参数
        x_true: 真实状态
        sensors: 候选传感器列表
        k: 预算
        train_idx: 训练集索引
        test_idx: 测试集索引
        decision_config: 决策配置
        fold_idx: Fold索引
        seed: 随机种子

    Returns:
        (fold_idx, metrics): Fold索引和指标字典
    """
    print(f"    Fold {fold_idx + 1}: Starting...")

    # 获取方法
    method_func = get_selection_method(method_name, geom, seed + fold_idx)

    # 选择传感器
    selection_result = method_func(sensors, k, Q_pr)
    selected_sensors = [sensors[i] for i in selection_result.selected_ids]

    # 生成观测
    from sensors import get_observation
    import numpy as np
    rng = np.random.default_rng(seed + fold_idx)
    y, H, R = get_observation(x_true, selected_sensors, rng)

    # 计算后验
    from inference import compute_posterior, compute_posterior_variance_diagonal
    mu_post, factor = compute_posterior(Q_pr, mu_pr, H, R, y)

    # 计算后验方差（仅在测试集上）
    var_post_test = compute_posterior_variance_diagonal(factor, test_idx)
    sigma_post_test = np.sqrt(np.maximum(var_post_test, 1e-12))

    # 扩展到全数组（用于compute_metrics）
    sigma_post = np.zeros(len(mu_post))
    sigma_post[test_idx] = sigma_post_test

    # 计算指标
    from evaluation import compute_metrics, morans_i
    metrics = compute_metrics(
        mu_post, sigma_post, x_true, test_idx, decision_config
    )

    # 计算Moran's I
    residuals = mu_post - x_true
    try:
        I_stat, I_pval = morans_i(
            residuals[test_idx],
            geom.adjacency[test_idx][:, test_idx],
            n_permutations=499,
            rng=rng
        )
        metrics['morans_i'] = float(I_stat)
        metrics['morans_pval'] = float(I_pval)
    except:
        metrics['morans_i'] = 0.0
        metrics['morans_pval'] = 1.0

    # 🔥 保存额外数据用于可视化
    metrics['selection_result'] = selection_result
    metrics['mu_post'] = mu_post
    metrics['x_true'] = x_true
    metrics['test_idx'] = test_idx
    metrics['fold_idx'] = fold_idx

    print(f"    Fold {fold_idx + 1}: RMSE={metrics['rmse']:.3f}, "
          f"Loss=£{metrics['expected_loss_gbp']:.0f}")

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