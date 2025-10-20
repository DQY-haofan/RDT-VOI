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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from geometry import build_grid2d_geometry
from spatial_field import build_prior, sample_gmrf
from sensors import generate_sensor_pool
from selection import greedy_mi, greedy_aopt, uniform_selection, random_selection
from evaluation import run_cv_experiment, spatial_bootstrap
from visualization import (setup_style, plot_budget_curves, plot_marginal_voi,
                          plot_residual_map, plot_performance_profile,
                          plot_critical_difference)


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
        with open(output_dir / "environment.txt", 'w') as f:
            f.write(env_info)
    except:
        pass

    return output_dir


def run_milestone_m1(cfg, output_dir):
    """
    Milestone M1: Single-fold validation on small grid.
    Verify monotonic decrease and diminishing returns.
    """
    print("\n" + "=" * 60)
    print("MILESTONE M1: Small-scale validation")
    print("=" * 60)

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

    # ✅ 修复：只运行一次最大 budget，检查完整序列
    max_budget = max(cfg.acceptance.m1_budgets)
    print(f"\nRunning Greedy-MI with budget k={max_budget}")
    result = greedy_mi(sensors, max_budget, Q_pr, lazy=True)

    # 保存边际 VoI 图
    plot_marginal_voi(
        result,
        output_dir / "curves" / f"m1_marginal_voi_full.png"
    )

    # ✅ 检查单调性：objective_values 应该单调递增
    is_monotonic = all(
        result.objective_values[i] <= result.objective_values[i + 1]
        for i in range(len(result.objective_values) - 1)
    )

    # ✅ 检查递减性：marginal_gains 应该单调递减（或接近）
    # 允许小的数值误差
    is_diminishing = all(
        result.marginal_gains[i] >= result.marginal_gains[i + 1] - 1e-6
        for i in range(len(result.marginal_gains) - 1)
    )

    print("\n" + "-" * 60)
    print(f"M1 RESULTS:")
    print(f"  Total steps: {len(result.objective_values)}")
    print(f"  Final MI: {result.objective_values[-1]:.3f} nats")
    print(f"  Monotonic increase: {'✓ PASS' if is_monotonic else '✗ FAIL'}")
    print(f"  Diminishing returns: {'✓ PASS' if is_diminishing else '✗ FAIL'}")

    # 显示前几步的边际收益
    print(f"\n  First 5 marginal gains: {result.marginal_gains[:5]}")
    print(f"  Last 5 marginal gains: {result.marginal_gains[-5:]}")
    print("-" * 60)

    # Save results
    m1_summary = {
        'monotonic': is_monotonic,
        'diminishing': is_diminishing,
        'final_mi': result.objective_values[-1],
        'marginal_gains': result.marginal_gains,
        'max_budget': max_budget
    }

    with open(output_dir / "m1_summary.json", 'w', encoding='utf-8') as f:
        json.dump(m1_summary, f, indent=2)

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

    # Define methods
    methods = {
        'Greedy-MI': lambda s, k, Q: greedy_mi(s, k, Q, lazy=True),
        'Greedy-A': lambda s, k, Q: greedy_aopt(s, k, Q),
        'Uniform': lambda s, k, Q: uniform_selection(s, k, geom),
        'Random': lambda s, k, Q: random_selection(s, k, rng)
    }

    # Run experiments for each method and budget
    all_results = {}

    for method_name, method_func in methods.items():
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        all_results[method_name] = {}

        for k in cfg.selection.budgets:
            print(f"\nBudget k={k}")

            # Run CV experiment
            cv_results = run_cv_experiment(
                geom, Q_pr, mu_pr, x_true, sensors,
                method_func, k,
                cfg.cv.__dict__, cfg.decision, rng
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
    for method in methods.keys():
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

    with open(output_dir / "m2_summary.json", 'w') as f:
        json.dump(m2_summary, f, indent=2)

    return m2_summary, all_results


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

        with open(output_dir / "final_summary.json", 'w') as f:
            json.dump(final_summary, f, indent=2)

        print(f"\n✓ Summary saved: {output_dir / 'final_summary.json'}")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

        # Save error log
        with open(output_dir / "error.log", 'w') as f:
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
        print(f"\n❌ Error: {e}")
        print("\nPlease create config.yaml in project root.")
        print("You can copy from the provided template.")
        return 1

    # Run experiment
    try:
        run_full_experiment(cfg)
        return 0
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()