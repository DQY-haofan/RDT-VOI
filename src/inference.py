"""
Bayesian posterior computation with sparse solvers.
Implements exact inference for linear-Gaussian GMRF models.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, Optional
import warnings


class SparseFactor:
    """
    Wrapper for sparse Cholesky factor of precision matrix.
    Provides efficient solve, logdet, and rank-1 update operations.
    """

    def __init__(self, Q: sp.spmatrix, method: str = "cholmod"):
        """
        Factorize sparse SPD matrix Q.

        Args:
            Q: Sparse precision matrix (must be SPD)
            method: "cholmod" | "splu" | "pcg"
        """
        self.n = Q.shape[0]
        self.method = method
        self.Q = Q.tocsc()  # CSC for factorization

        if method == "cholmod":
            try:
                from sksparse.cholmod import cholesky
                self.factor = cholesky(self.Q)
                self._has_cholmod = True
            except ImportError:
                warnings.warn("cholmod not available, falling back to splu")
                self.factor = spla.splu(self.Q)
                self._has_cholmod = False

        elif method == "splu":
            self.factor = spla.splu(self.Q)
            self._has_cholmod = False

        elif method == "pcg":
            # For PCG, store Q and use incomplete Cholesky preconditioner
            from scipy.sparse.linalg import LinearOperator
            self._has_cholmod = False
            self.factor = None
            # Try incomplete Cholesky for preconditioner
            try:
                from scipy.sparse.linalg import spilu
                self.ilu = spilu(self.Q)
            except:
                self.ilu = None

        else:
            raise ValueError(f"Unknown factorization method: {method}")

    def solve(self, b: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """
        Solve Q * x = b.

        Args:
            b: Right-hand side (n,) or (n, k)
            tol: Tolerance for iterative solvers

        Returns:
            x: Solution vector(s)
        """
        if self.method == "pcg":
            if b.ndim == 1:
                if self.ilu is not None:
                    M = spla.LinearOperator(
                        (self.n, self.n),
                        matvec=self.ilu.solve
                    )
                else:
                    M = None

                x, info = spla.cg(self.Q, b, tol=tol, M=M, maxiter=self.n)
                if info != 0:
                    warnings.warn(f"PCG did not converge (info={info})")
                return x
            else:
                # Solve column by column
                return np.column_stack([self.solve(b[:, i], tol) for i in range(b.shape[1])])

        else:
            # For both cholmod and splu
            if self._has_cholmod:
                return self.factor.solve_A(b)
            else:
                return self.factor.solve(b)

    def logdet(self) -> float:
        """
        Compute log determinant of Q.

        For Cholesky Q = L L^T: log|Q| = 2 * sum(log(diag(L)))
        """
        if self._has_cholmod:
            return self.factor.logdet()

        elif self.method == "splu":
            # From LU factorization
            L_diag = self.factor.L.diagonal()
            U_diag = self.factor.U.diagonal()
            return np.sum(np.log(np.abs(L_diag))) + np.sum(np.log(np.abs(U_diag)))

        else:
            # Expensive: compute via solve
            warnings.warn("logdet via PCG is expensive, consider using cholmod/splu")
            # Use Hutchinson estimator
            rng = np.random.default_rng(42)
            n_probes = 20
            trace_inv = 0.0
            for _ in range(n_probes):
                z = rng.standard_normal(self.n)
                trace_inv += np.dot(z, self.solve(z))
            trace_inv /= n_probes
            # log|Q| ≈ -log(trace(Q^{-1})) (rough approximation)
            return -np.log(trace_inv) * self.n

    def rank1_update(self, h: np.ndarray, weight: float = 1.0):
        """
        Update factor to reflect Q_new = Q + weight * h h^T.

        Uses cholupdate if available, otherwise requires refactorization.

        Args:
            h: Update vector (n,)
            weight: Scaling factor
        """
        if self._has_cholmod and weight > 0:
            # Update factor in-place
            self.factor.update_inplace(h, weight)
        else:
            # Refactorize
            self.Q = self.Q + weight * sp.csr_matrix(np.outer(h, h))
            self.__init__(self.Q, self.method)


def compute_posterior(Q_pr: sp.spmatrix,
                      mu_pr: np.ndarray,
                      H: sp.spmatrix,
                      R_diag: np.ndarray,
                      y: np.ndarray,
                      method: str = "cholmod",
                      tol: float = 1e-8) -> Tuple[np.ndarray, SparseFactor]:
    """
    Compute Gaussian posterior for linear observation model.
    ...
    """
    n, m = Q_pr.shape[0], len(y)

    # Compute Q_post = Q_pr + H^T R^{-1} H
    R_inv = 1.0 / R_diag
    H_weighted = sp.diags(R_inv) @ H  # R^{-1} H
    Q_post = Q_pr + H.T @ H_weighted

    # Right-hand side: H^T R^{-1} y + Q_pr μ_pr
    rhs = H.T @ (R_inv * y) + Q_pr @ mu_pr

    # Factorize and solve
    factor = SparseFactor(Q_post, method=method)
    mu_post = factor.solve(rhs, tol=tol)

    # Validate solution - 添加零检查
    residual = Q_post @ mu_post - rhs
    rhs_norm = np.linalg.norm(rhs)

    if rhs_norm > 1e-14:  # ✅ 添加这个检查
        rel_residual = np.linalg.norm(residual) / rhs_norm
        if rel_residual > tol:
            warnings.warn(f"High relative residual: {rel_residual:.2e} > {tol:.2e}")
    else:
        # rhs 接近零，检查绝对残差
        abs_residual = np.linalg.norm(residual)
        if abs_residual > tol:
            warnings.warn(f"High absolute residual: {abs_residual:.2e} > {tol:.2e}")

    return mu_post, factor


def compute_posterior_variance_diagonal(factor: SparseFactor,
                                       indices: np.ndarray = None) -> np.ndarray:
    """
    Compute diagonal elements of posterior covariance Σ_post = Q_post^{-1}.

    Uses "solve unit vector" method: solve Q z_i = e_i, then σ²_i = z_i[i]

    Args:
        factor: SparseFactor for Q_post
        indices: Subset of indices to compute (if None, compute all)

    Returns:
        var_diag: Posterior variances (len(indices),) or (n,)
    """
    n = factor.n
    if indices is None:
        indices = np.arange(n)

    var_diag = np.zeros(len(indices))

    for i, idx in enumerate(indices):
        e_i = np.zeros(n)
        e_i[idx] = 1.0
        z_i = factor.solve(e_i)
        var_diag[i] = z_i[idx]

    return var_diag


def quadform_via_solve(factor: SparseFactor, h: np.ndarray) -> float:
    """
    Compute quadratic form h^T Σ_post h = h^T Q_post^{-1} h.

    Solves Q z = h, then computes h^T z.

    Args:
        factor: SparseFactor for Q_post
        h: Vector (n,)

    Returns:
        q: Scalar h^T Σ_post h
    """
    z = factor.solve(h)
    return np.dot(h, z)


def compute_mutual_information(Q_pr: sp.spmatrix,
                               H: sp.spmatrix,
                               R_diag: np.ndarray) -> float:
    """
    Compute mutual information I(x; y) for Gaussian model.

    I(x; y) = 0.5 * log |I + Σ_pr H^T R^{-1} H|
            = 0.5 * (log|Q_post| - log|Q_pr|)

    Args:
        Q_pr: Prior precision
        H: Observation matrix
        R_diag: Observation noise variances

    Returns:
        mi: Mutual information (nats)
    """
    # Compute Q_post
    R_inv = 1.0 / R_diag
    H_weighted = sp.diags(R_inv) @ H
    Q_post = Q_pr + H.T @ H_weighted

    # Factorize
    factor_pr = SparseFactor(Q_pr)
    factor_post = SparseFactor(Q_post)

    # MI = 0.5 * (log|Q_post| - log|Q_pr|)
    mi = 0.5 * (factor_post.logdet() - factor_pr.logdet())

    return mi


if __name__ == "__main__":
    from config import load_config
    from geometry import build_grid2d_geometry
    from spatial_field import build_prior, sample_gmrf
    from sensors import generate_sensor_pool, get_observation

    cfg = load_config()
    rng = cfg.get_rng()

    # Setup
    geom = build_grid2d_geometry(20, 20, h=cfg.geometry.h)
    Q_pr, mu_pr = build_prior(geom, cfg.prior)
    x_true = sample_gmrf(Q_pr, mu_pr, rng)

    # Generate sensors and observe
    sensors = generate_sensor_pool(geom, cfg.sensors, rng)
    selected = rng.choice(sensors, size=20, replace=False)
    y, H, R = get_observation(x_true, selected, rng)

    # Compute posterior
    print("Computing posterior...")
    mu_post, factor = compute_posterior(Q_pr, mu_pr, H, R, y)

    print(f"  Prior mean range: [{mu_pr.min():.3f}, {mu_pr.max():.3f}]")
    print(f"  Posterior mean range: [{mu_post.min():.3f}, {mu_post.max():.3f}]")
    print(f"  True state range: [{x_true.min():.3f}, {x_true.max():.3f}]")

    # Compute MI
    mi = compute_mutual_information(Q_pr, H, R)
    print(f"  Mutual Information: {mi:.3f} nats ({mi/np.log(2):.3f} bits)")

    # Check posterior variance at few points
    test_idx = [0, geom.n//2, geom.n-1]
    var_post = compute_posterior_variance_diagonal(factor, test_idx)
    print(f"  Posterior std at test points: {np.sqrt(var_post)}")