"""
Method wrappers for easy integration into main evaluation pipeline.
Provides unified interface for all selection methods.
"""

import numpy as np
from typing import List, Callable
from sensors import Sensor
from selection import (
    greedy_mi, greedy_aopt, greedy_evi_myopic, maxmin_k_center,
    uniform_random_selection, SelectionResult
)
import scipy.sparse as sp


def get_selection_method(method_name: str, config, geom,
                         x_true: np.ndarray = None,
                         test_idx: np.ndarray = None) -> Callable:
    """
    Get a selection method function with unified signature.

    Args:
        method_name: One of ['greedy_mi', 'greedy_aopt', 'greedy_evi',
                             'maxmin', 'uniform', 'random']
        config: Configuration object
        geom: Geometry object
        x_true: True state (needed for some methods)
        test_idx: Test indices (needed for EVI)

    Returns:
        Function with signature: f(sensors, k, Q_pr, **kwargs) -> SelectionResult
    """

    if method_name.lower() in ['greedy_mi', 'greedy-mi', 'mi']:
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
        def wrapper(sensors, k, Q_pr, mu_pr=None):
            costs = np.array([s.cost for s in sensors])
            # 🔥 Use fast A-opt with Hutch++
            return greedy_aopt(
                sensors=sensors,
                k=k,
                Q_pr=Q_pr,
                costs=costs,
                n_probes=16,  # Good balance of speed/accuracy
                use_cost=True
            )

        return wrapper

    elif method_name.lower() in ['greedy_evi', 'greedy-evi', 'evi', 'myopic_evi']:
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
                n_y_samples=25,  # Fast approximation
                use_cost=True,
                rng=rng
            )

        return wrapper

    elif method_name.lower() in ['maxmin', 'k-center', 'kcenter', 'max-min']:
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
        def wrapper(sensors, k, Q_pr, mu_pr=None):
            rng = config.get_rng()
            return uniform_random_selection(
                sensors=sensors,
                k=k,
                strategy='uniform',
                rng=rng
            )

        return wrapper

    elif method_name.lower() == 'random':
        def wrapper(sensors, k, Q_pr, mu_pr=None):
            rng = config.get_rng()
            return uniform_random_selection(
                sensors=sensors,
                k=k,
                strategy='random',
                rng=rng
            )

        return wrapper

    else:
        raise ValueError(f"Unknown method: {method_name}")


def get_available_methods(config) -> List[str]:
    """Get list of methods configured for evaluation."""
    if hasattr(config.selection, 'methods'):
        return config.selection.methods
    else:
        # Default methods
        return ['greedy_mi', 'greedy_aopt', 'uniform', 'random']


def should_use_evi(method_name: str, budget: int, fold_idx: int,
                   config) -> bool:
    """
    Determine if EVI should be run for this (method, budget, fold) combination.

    Since EVI is expensive, we can run it on a subset of configurations:
    - Only on representative budgets (e.g., k=10, k=40)
    - Only on first 2-3 folds
    """
    if method_name.lower() not in ['greedy_evi', 'evi']:
        return True  # Always run non-EVI methods

    # Check if EVI is configured with constraints
    if hasattr(config.selection, 'greedy_evi'):
        evi_cfg = config.selection.greedy_evi

        # Check budget constraint
        if 'budgets_subset' in evi_cfg:
            if budget not in evi_cfg['budgets_subset']:
                return False

        # Check fold constraint
        if 'max_folds' in evi_cfg:
            if fold_idx >= evi_cfg['max_folds']:
                return False

    return True