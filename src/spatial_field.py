"""
GMRF spatial field prior construction and sampling.
Implements SPDE-Matérn approach with sparse precision matrices.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple
from scipy.special import gamma


def matern_tau_from_params(nu: float, kappa: float, sigma2: float,
                           d: int = 2, alpha: int = 2) -> float:
    """
    Compute SPDE noise scale τ from Matérn parameters.

    Uses relation: σ² = Γ(ν) / [Γ(α) (4π)^{d/2} κ^{2ν} τ²]
    where α = ν + d/2

    Args:
        nu: Matérn smoothness
        kappa: Inverse correlation range
        sigma2: Marginal variance
        d: Spatial dimension
        alpha: SPDE operator power

    Returns:
        tau: SPDE noise scale
    """
    numerator = gamma(nu)
    denominator = gamma(alpha) * (4 * np.pi) ** (d / 2) * kappa ** (2 * nu) * sigma2
    tau_squared = numerator / denominator
    return np.sqrt(tau_squared)


def build_grid_precision_spde(nx: int, ny: int, h: float,
                              kappa: float, beta: float = 1e-6) -> sp.spmatrix:
    """
    Build GMRF precision matrix for 2D grid using SPDE discretization.

    Implements Q = (κ² - Δ) with 5-point stencil:

        [-1/h²]
    [-1/h²][κ² + 4/h²][-1/h²]
        [-1/h²]

    Plus nugget β*I for strict positive-definiteness.

    Args:
        nx, ny: Grid dimensions
        h: Grid spacing
        kappa: SPDE parameter κ
        beta: Nugget term for SPD

    Returns:
        Q: Sparse precision matrix (n×n)
    """
    n = nx * ny

    def idx(i, j):
        return i * ny + j

    # Center coefficient: κ² + 4/h²
    center_coef = kappa ** 2 + 4.0 / h ** 2
    # Neighbor coefficient: -1/h²
    neigh_coef = -1.0 / h ** 2

    row_idx = []
    col_idx = []
    data = []

    for i in range(nx):
        for j in range(ny):
            current = idx(i, j)

            # Diagonal (center)
            row_idx.append(current)
            col_idx.append(current)
            data.append(center_coef + beta)  # Add nugget

            # Right neighbor
            if i < nx - 1:
                row_idx.append(current)
                col_idx.append(idx(i + 1, j))
                data.append(neigh_coef)

            # Left neighbor
            if i > 0:
                row_idx.append(current)
                col_idx.append(idx(i - 1, j))
                data.append(neigh_coef)

            # Top neighbor
            if j < ny - 1:
                row_idx.append(current)
                col_idx.append(idx(i, j + 1))
                data.append(neigh_coef)

            # Bottom neighbor
            if j > 0:
                row_idx.append(current)
                col_idx.append(idx(i, j - 1))
                data.append(neigh_coef)

    Q = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n, n))
    return Q.tocsr()


def build_graph_precision(L: sp.spmatrix, alpha: float, beta: float) -> sp.spmatrix:
    """
    Build GMRF precision from graph Laplacian.

    Q = α*L + β*I

    Args:
        L: Graph Laplacian matrix
        alpha: Smoothness weight
        beta: Nugget for SPD

    Returns:
        Q: Precision matrix
    """
    n = L.shape[0]
    Q = alpha * L + beta * sp.eye(n)
    return Q.tocsr()


def sample_gmrf(Q: sp.spmatrix,
                mu: np.ndarray = None,
                rng: np.random.Generator = None) -> np.ndarray:
    """
    Sample from GMRF with precision Q and mean μ.

    Uses sparse Cholesky: Q = L L^T, solve L^T x = z where z ~ N(0, I)

    Args:
        Q: Sparse precision matrix (must be SPD)
        mu: Mean vector (if None, use zero mean)
        rng: Random number generator

    Returns:
        x: Sample from N(μ, Q^{-1})
    """
    n = Q.shape[0]
    if mu is None:
        mu = np.zeros(n)
    if rng is None:
        rng = np.random.default_rng()

    # Sample standard normal
    z = rng.standard_normal(n)

    try:
        # Attempt sparse Cholesky
        from sksparse.cholmod import cholesky
        factor = cholesky(Q)
        # Solve L^T x_centered = z
        x_centered = factor.solve_Lt(z, use_LDLt_decomposition=False)
    except ImportError:
        # Fallback to splu (slower but always available)
        lu = spla.splu(Q)
        x_centered = lu.solve(z)

    return mu + x_centered


def build_prior(geom, prior_config) -> Tuple[sp.spmatrix, np.ndarray]:
    """
    Build GMRF prior precision and mean from geometry and config.

    Args:
        geom: Geometry object
        prior_config: PriorConfig object

    Returns:
        Q_pr: Prior precision matrix
        mu_pr: Prior mean vector
    """
    n = geom.n

    if geom.mode == "grid2d":
        Q_pr = build_grid_precision_spde(
            nx=int(np.sqrt(n)),
            ny=int(np.sqrt(n)),
            h=geom.h,
            kappa=prior_config.kappa,
            beta=prior_config.beta
        )
    elif geom.mode in ["polyline1d", "graph"]:
        Q_pr = build_graph_precision(
            L=geom.laplacian,
            alpha=prior_config.alpha,
            beta=prior_config.beta
        )
    else:
        raise ValueError(f"Unknown geometry mode: {geom.mode}")

    # Construct mean field
    if prior_config.mu_prior_std > 0:
        # Sample a smooth mean field
        Q_mean = build_graph_precision(
            geom.laplacian,
            alpha=0.1,  # Smoother than prior
            beta=prior_config.beta
        )
        rng_mean = np.random.default_rng(42)  # Fixed seed for mean
        mu_pr = prior_config.mu_prior_mean + \
                prior_config.mu_prior_std * sample_gmrf(Q_mean, rng=rng_mean)
    else:
        # Constant mean
        mu_pr = np.full(n, prior_config.mu_prior_mean)

    return Q_pr, mu_pr


def validate_prior(Q: sp.spmatrix, mu: np.ndarray,
                   rng: np.random.Generator = None,
                   n_samples: int = 5) -> dict:
    """
    Validate prior by checking samples and computing statistics.

    Args:
        Q: Precision matrix
        mu: Mean vector
        rng: Random generator
        n_samples: Number of test samples

    Returns:
        stats: Dictionary of validation statistics
    """
    if rng is None:
        rng = np.random.default_rng()

    # Check SPD
    min_eig = spla.eigsh(Q, k=1, which='SA', return_eigenvectors=False)[0]

    # Sample and compute empirical statistics
    samples = [sample_gmrf(Q, mu, rng) for _ in range(n_samples)]
    samples = np.array(samples)

    emp_mean = samples.mean(axis=0)
    emp_std = samples.std(axis=0)

    stats = {
        'n': Q.shape[0],
        'nnz': Q.nnz,
        'sparsity': Q.nnz / Q.shape[0] ** 2,
        'min_eigenvalue': min_eig,
        'is_spd': min_eig > 0,
        'mean_deviation': np.abs(emp_mean - mu).max(),
        'empirical_std_range': (emp_std.min(), emp_std.max())
    }

    return stats


if __name__ == "__main__":
    # Test SPDE precision construction
    from config import load_config
    from geometry import build_grid2d_geometry

    cfg = load_config()
    geom = build_grid2d_geometry(20, 20, h=cfg.geometry.h)

    Q_pr, mu_pr = build_prior(geom, cfg.prior)

    print("Prior construction:")
    print(f"  n = {Q_pr.shape[0]}")
    print(f"  nnz = {Q_pr.nnz} ({Q_pr.nnz / Q_pr.shape[0] ** 2 * 100:.2f}%)")

    # Validate
    rng = cfg.get_rng()
    stats = validate_prior(Q_pr, mu_pr, rng)
    print(f"  Min eigenvalue: {stats['min_eigenvalue']:.6f}")
    print(f"  Is SPD: {stats['is_spd']}")

    # Sample true state
    x_true = sample_gmrf(Q_pr, mu_pr, rng)
    print(f"  Sample range: [{x_true.min():.3f}, {x_true.max():.3f}]")