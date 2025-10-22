"""
Method wrappers for easy integration into main evaluation pipeline.
Provides unified interface for all selection methods.
"""

import numpy as np
from typing import List, Callable
from sensors import Sensor
import scipy.sparse as sp


def get_selection_method(method_name: str, config, geom,  # 🔥 参数名必须是 config
                        x_true: np.ndarray = None,
                        test_idx: np.ndarray = None) -> Callable:
    """
    Get a selection method function with unified signature.

    Args:
        method_name: One of ['greedy_mi', 'greedy_aopt', 'greedy_evi',
                             'maxmin', 'uniform', 'random']
        config: Configuration object (注意参数名是 config 不是 cfg)
        geom: Geometry object
        x_true: True state (needed for some methods)
        test_idx: Test indices (needed for EVI)

    Returns:
        Function with signature: f(sensors, k, Q_pr, mu_pr=None) -> SelectionResult
    """

    if method_name.lower() in ['greedy_mi', 'greedy-mi', 'mi']:
        from selection import greedy_mi

        def wrapper(sensors, k, Q_pr, mu_pr=None):
            costs = np.array([s.cost for s in sensors])
            # Use config parameters if available
            if hasattr(config.selection, 'greedy_mi'):
                mi_cfg = config.selection.greedy_mi
                return greedy_mi(
                    sensors=sensors,
                    k=k,
                    Q_pr=Q_pr,
                    costs=costs,
                    batch_size=mi_cfg.get('batch_size', 1),
                    adaptive_pruning=mi_cfg.get('adaptive_pruning', False),
                    prune_threshold=mi_cfg.get('prune_threshold', 0.1),
                    max_candidates=mi_cfg.get('max_candidates', None),
                    verbose=False
                )
            else:
                return greedy_mi(sensors, k, Q_pr, costs=costs)
        return wrapper

    elif method_name.lower() in ['greedy_aopt', 'greedy-aopt', 'greedy_a', 'aopt', 'a']:
        from selection import greedy_aopt

        def wrapper(sensors, k, Q_pr, mu_pr=None):
            costs = np.array([s.cost for s in sensors])
            # Use fast A-opt with Hutch++
            return greedy_aopt(
                sensors=sensors,
                k=k,
                Q_pr=Q_pr,
                costs=costs,
                n_probes=16,
                use_cost=True
            )
        return wrapper

    elif method_name.lower() in ['greedy_evi', 'greedy-evi', 'evi', 'myopic_evi']:
        from selection import greedy_evi_myopic

        if x_true is None:
            raise ValueError("EVI method requires x_true")
        if test_idx is None:
            # Use random subset
            rng = np.random.default_rng(config.experiment.seed)
            n_test = min(200, geom.n)
            test_idx = rng.choice(geom.n, size=n_test, replace=False)

        def wrapper(sensors, k, Q_pr, mu_pr):
            costs = np.array([s.cost for s in sensors])
            rng = config.get_rng()
            return greedy_evi_myopic(
                sensors=sensors,
                k=k,
                Q_pr=Q_pr,
                mu_pr=mu_pr,
                decision_config=config.decision,
                test_idx=test_idx,
                costs=costs,
                n_y_samples=25,
                use_cost=True,
                rng=rng
            )
        return wrapper

    elif method_name.lower() in ['maxmin', 'k-center', 'kcenter', 'max-min']:
        from selection import maxmin_k_center

        def wrapper(sensors, k, Q_pr, mu_pr=None):
            costs = np.array([s.cost for s in sensors])
            return maxmin_k_center(
                sensors=sensors,
                k=k,
                coords=geom.coords,
                costs=costs,
                use_cost=True
            )
        return wrapper

    elif method_name.lower() in ['uniform', 'uniform_random']:
        # Uniform random selection
        def wrapper(sensors, k, Q_pr, mu_pr=None):
            from selection import SelectionResult
            rng = config.get_rng()

            n_sensors = len(sensors)
            if k > n_sensors:
                k = n_sensors

            selected_ids = rng.choice(n_sensors, size=k, replace=False).tolist()
            total_cost = sum(sensors[i].cost for i in selected_ids)

            return SelectionResult(
                selected_ids=selected_ids,
                objective_values=[0.0] * k,
                marginal_gains=[0.0] * k,
                total_cost=total_cost,
                method_name="Uniform"
            )
        return wrapper

    elif method_name.lower() == 'random':
        # Cost-weighted random selection
        def wrapper(sensors, k, Q_pr, mu_pr=None):
            from selection import SelectionResult
            rng = config.get_rng()

            n_sensors = len(sensors)
            costs = np.array([s.cost for s in sensors])

            # Inverse cost weighting
            weights = 1.0 / (costs + 1.0)
            weights = weights / weights.sum()

            if k > n_sensors:
                k = n_sensors

            selected_ids = rng.choice(n_sensors, size=k, replace=False, p=weights).tolist()
            total_cost = sum(sensors[i].cost for i in selected_ids)

            return SelectionResult(
                selected_ids=selected_ids,
                objective_values=[0.0] * k,
                marginal_gains=[0.0] * k,
                total_cost=total_cost,
                method_name="Random"
            )
        return wrapper

    else:
        raise ValueError(f"Unknown method: {method_name}")


def get_available_methods(config) -> List[str]:
    """Get list of methods configured for evaluation."""
    if hasattr(config.selection, 'methods'):
        return config.selection.methods
    else:
        return ['greedy_mi', 'greedy_aopt', 'uniform', 'random']


def should_use_evi(method_name: str, budget: int, fold_idx: int,
                   config) -> bool:
    """
    Determine if EVI should be run for this (method, budget, fold) combination.

    Since EVI is expensive, we can run it on a subset of configurations.
    """
    if method_name.lower() not in ['greedy_evi', 'evi']:
        return True  # Always run non-EVI methods

    # Check if EVI is configured with constraints
    if hasattr(config.selection, 'greedy_evi'):
        evi_cfg = config.selection.greedy_evi

        # Check budget constraint
        if 'budgets_subset' in evi_cfg:
            budgets_subset = evi_cfg['budgets_subset']
            if budgets_subset and budget not in budgets_subset:
                return False

        # Check fold constraint
        if 'max_folds' in evi_cfg:
            max_folds = evi_cfg['max_folds']
            if max_folds is not None and fold_idx >= max_folds:
                return False

    return True