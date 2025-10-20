"""
Visualization functions for experimental results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
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


def plot_budget_curves(results_by_method: Dict,
                       metric: str,
                       output_path: Path,
                       show_ci: bool = True):
    """
    Plot metric vs budget curves for multiple methods.

    Args:
        results_by_method: {method_name: {budget: result_dict}}
        metric: Metric to plot ('expected_loss_gbp', 'rmse', 'mae')
        output_path: Output file path
        show_ci: Show confidence intervals
    """
    fig, ax = plt.subplots(figsize=(8, 5))

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

        ax.plot(budgets, means, marker='o', label=method_name, linewidth=2)

        if show_ci and lowers:
            ax.fill_between(budgets, lowers, uppers, alpha=0.2)

    ax.set_xlabel('Sensor Budget $k$')

    metric_labels = {
        'expected_loss_gbp': 'Expected Economic Loss (£)',
        'rmse': 'RMSE (m/km)',
        'mae': 'MAE (m/km)',
        'r2': '$R^2$'
    }
    ax.set_ylabel(metric_labels.get(metric, metric))

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_marginal_voi(selection_result, output_path: Path):
    """
    Plot marginal Value of Information curve.

    Args:
        selection_result: SelectionResult object
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    steps = np.arange(1, len(selection_result.marginal_gains) + 1)
    gains = selection_result.marginal_gains

    ax.plot(steps, gains, marker='o', color='steelblue', linewidth=2)
    ax.fill_between(steps, 0, gains, alpha=0.3, color='steelblue')

    ax.set_xlabel('Sensor Addition Step')
    ax.set_ylabel('Marginal MI Gain (nats)')
    ax.set_title(f'Diminishing Returns: {selection_result.method_name}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_residual_map(mu_post: np.ndarray,
                      x_true: np.ndarray,
                      coords: np.ndarray,
                      output_path: Path,
                      title: str = "Prediction Residuals"):
    """
    Plot spatial map of prediction residuals.

    Args:
        mu_post: Posterior means
        x_true: True state
        coords: Spatial coordinates (n, 2)
        output_path: Output file path
        title: Plot title
    """
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

    print(f"  Saved: {output_path}")


def plot_performance_profile(results_by_method: Dict,
                             metric: str,
                             output_path: Path,
                             tau_max: float = 3.0):
    """
    Plot Dolan-Moré performance profile.

    Args:
        results_by_method: {method: {instance: metric_value}}
        metric: Metric name
        output_path: Output file path
        tau_max: Maximum τ value
    """
    # Compute performance ratios
    methods = list(results_by_method.keys())
    instances = list(next(iter(results_by_method.values())).keys())

    # Build performance matrix
    perf_matrix = np.zeros((len(methods), len(instances)))
    for i, method in enumerate(methods):
        for j, instance in enumerate(instances):
            perf_matrix[i, j] = results_by_method[method][instance]

    # For minimization metrics, lower is better
    # Find best performance per instance
    best_perf = perf_matrix.min(axis=0)

    # Compute ratios
    ratios = perf_matrix / best_perf[None, :]

    # Plot profiles
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

    print(f"  Saved: {output_path}")


def plot_critical_difference(results_by_method: Dict,
                             metric: str,
                             output_path: Path,
                             alpha: float = 0.05):
    """
    Plot critical difference diagram with Nemenyi test.

    Args:
        results_by_method: {method: {instance: metric_value}}
        metric: Metric name
        output_path: Output file path
        alpha: Significance level
    """
    from scipy.stats import rankdata

    methods = list(results_by_method.keys())
    instances = list(next(iter(results_by_method.values())).keys())
    n_methods = len(methods)
    n_instances = len(instances)

    # Build rank matrix
    ranks = np.zeros((n_methods, n_instances))
    for j, instance in enumerate(instances):
        values = [results_by_method[m][instance] for m in methods]
        # Lower is better for loss/error metrics
        instance_ranks = rankdata(values, method='average')
        ranks[:, j] = instance_ranks

    # Average ranks
    avg_ranks = ranks.mean(axis=1)

    # Nemenyi critical difference
    q_alpha = 2.569  # For alpha=0.05, k=4 methods (lookup from table)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_instances))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 2))

    # Sort methods by rank
    sorted_idx = np.argsort(avg_ranks)
    sorted_methods = [methods[i] for i in sorted_idx]
    sorted_ranks = avg_ranks[sorted_idx]

    # Plot ranks
    ax.scatter(sorted_ranks, np.zeros(n_methods), s=100, zorder=3)

    # Add method labels
    for i, (rank, method) in enumerate(zip(sorted_ranks, sorted_methods)):
        ax.text(rank, -0.15, method, ha='center', va='top', fontsize=9)

    # Add CD bar
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            if sorted_ranks[j] - sorted_ranks[i] <= cd:
                # Connect with horizontal line (not significantly different)
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

    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    # Test visualization functions
    setup_style()

    # Mock data
    methods = ['Greedy-MI', 'Greedy-A', 'Uniform', 'Random']
    budgets = [10, 20, 40]

    results = {}
    for method in methods:
        results[method] = {}
        for k in budgets:
            results[method][k] = {
                'aggregated': {
                    'expected_loss_gbp': {
                        'mean': 10000 / k * (1.2 if method == 'Random' else 1.0),
                        'ci_lower': 9000 / k,
                        'ci_upper': 11000 / k
                    }
                }
            }

    plot_budget_curves(results, 'expected_loss_gbp', Path('test_budget.png'))
    print("Test plots generated")