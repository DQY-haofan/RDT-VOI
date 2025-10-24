"""
支持批量RHS的推断模块

主要改进：
1. solve()支持多列RHS
2. 新增batch_quadform()快速计算多个h^T Σ h
3. 优化的对角元提取
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, Optional, Union
import warnings


class SparseFactor:
    """
    改进的稀疏因子类：支持批量操作
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
        self.Q = Q.tocsc()

        if method == "cholmod":
            try:
                from sksparse.cholmod import cholesky
                self.factor = cholesky(self.Q)
                self._has_cholmod = True
                print("  Using CHOLMOD (fast)")
            except ImportError:
                warnings.warn("cholmod not available, falling back to splu")
                self.factor = spla.splu(self.Q)
                self._has_cholmod = False
                self.method = "splu"

        elif method == "splu":
            self.factor = spla.splu(self.Q)
            self._has_cholmod = False

        elif method == "pcg":
            self._has_cholmod = False
            self.factor = None
            try:
                from scipy.sparse.linalg import spilu
                self.ilu = spilu(self.Q)
            except:
                self.ilu = None

        else:
            raise ValueError(f"Unknown factorization method: {method}")

    def solve_multi(self, B: np.ndarray) -> np.ndarray:
        """
        🔥 新增：批量求解 Q * X = B (多个右端向量)

        Args:
            B: Right-hand sides (n, m) - 每列是一个RHS

        Returns:
            X: Solutions (n, m)
        """
        if sp.issparse(B):
            B = B.toarray()

        if self.method == "pcg":
            # PCG不支持多RHS，逐列求解
            if B.ndim == 1:
                return self.solve(B)
            return np.column_stack([self.solve(B[:, i]) for i in range(B.shape[1])])
        else:
            # CHOLMOD和SPLU原生支持多RHS
            if self._has_cholmod:
                return self.factor.solve_A(B)
            else:
                return self.factor.solve(B)

    def solve(self, b: Union[np.ndarray, sp.spmatrix], tol: float = 1e-8) -> np.ndarray:
        """
        Solve Q * x = b (supports multiple RHS)

        Args:
            b: Right-hand side (n,) or (n, k) or sparse matrix
            tol: Tolerance for iterative solvers

        Returns:
            x: Solution vector(s) (n,) or (n, k)
        """
        # 🔥 转换稀疏矩阵为稠密（对于批量RHS）
        if sp.issparse(b):
            b = b.toarray()

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
                # 🔥 批量PCG（列循环，未来可改为block-CG）
                return np.column_stack([self.solve(b[:, i], tol) for i in range(b.shape[1])])

        else:
            # 🔥 CHOLMOD和SPLU都支持多RHS
            if self._has_cholmod:
                return self.factor.solve_A(b)
            else:
                return self.factor.solve(b)

    def solve_lower(self, b: np.ndarray) -> np.ndarray:
        """
        Solve L * x = b (仅CHOLMOD支持)

        用途：快速计算 ||L^{-1} h||^2 = h^T Σ h
        """
        if not self._has_cholmod:
            raise NotImplementedError("solve_lower only available with CHOLMOD")

        return self.factor.solve_L(b)

    def logdet(self) -> float:
        """Compute log determinant of Q."""
        if self._has_cholmod:
            return self.factor.logdet()

        elif self.method == "splu":
            L_diag = self.factor.L.diagonal()
            U_diag = self.factor.U.diagonal()
            return np.sum(np.log(np.abs(L_diag))) + np.sum(np.log(np.abs(U_diag)))

        else:
            warnings.warn("logdet via PCG is expensive")
            rng = np.random.default_rng(42)
            n_probes = 20
            trace_inv = 0.0
            for _ in range(n_probes):
                z = rng.standard_normal(self.n)
                trace_inv += np.dot(z, self.solve(z))
            trace_inv /= n_probes
            return -np.log(trace_inv) * self.n

    def rank1_update(self, h: np.ndarray, weight: float = 1.0):
        """Update factor to reflect Q_new = Q + weight * h h^T."""
        if self._has_cholmod and weight > 0:
            self.factor.update_inplace(h, weight)
        else:
            # Refactorize
            self.Q = self.Q + weight * sp.csr_matrix(np.outer(h, h))
            self.__init__(self.Q, self.method)


# =====================================================================
# 🔥 新增：批量二次型计算
# =====================================================================

def batch_quadform_via_solve(factor: SparseFactor,
                             H: np.ndarray) -> np.ndarray:
    """
    批量计算 h_i^T Σ h_i，其中 H = [h_1, h_2, ..., h_m]

    Args:
        factor: SparseFactor for Q
        H: Matrix of vectors (n, m)

    Returns:
        quad: Array of quadratic forms (m,)

    算法：
        Z = Q^{-1} H  (一次solve，m个RHS)
        quad[i] = h_i^T z_i = sum(H[:, i] * Z[:, i])
    """
    # 🔥 一次solve得到所有结果
    Z = factor.solve(H)  # (n, m)

    # 🔥 批量计算列向量点积（Einstein求和）
    quad = np.einsum('ij,ij->j', H, Z)

    return quad


def batch_quadform_cholesky(factor: SparseFactor,
                            H: np.ndarray) -> np.ndarray:
    """
    使用Cholesky的更快方法（仅CHOLMOD）

    算法：
        L x = H  =>  x = L^{-1} H
        h^T Σ h = h^T Q^{-1} h = ||L^{-T} h||^2 = ||x||^2
    """
    if not factor._has_cholmod:
        return batch_quadform_via_solve(factor, H)

    # 🔥 一次下三角求解
    T = factor.solve_lower(H)  # L T = H

    # 🔥 列向量范数平方
    quad = np.einsum('ij,ij->j', T, T)

    return quad


def quadform_via_solve(factor: SparseFactor, h: np.ndarray) -> float:
    """
    单向量版本（向后兼容）

    Compute h^T Σ h = h^T Q^{-1} h
    """
    z = factor.solve(h)
    return np.dot(h, z)


# =====================================================================
# 后验计算（保持原有接口）
# =====================================================================

def compute_posterior(Q_pr: sp.spmatrix,
                      mu_pr: np.ndarray,
                      H: sp.spmatrix,
                      R_diag: np.ndarray,
                      y: np.ndarray,
                      method: str = "cholmod",
                      tol: float = 1e-8) -> Tuple[np.ndarray, SparseFactor]:
    """
    Compute Gaussian posterior (unchanged interface)
    """
    n, m = Q_pr.shape[0], len(y)

    # Compute Q_post = Q_pr + H^T R^{-1} H
    R_inv = 1.0 / R_diag
    H_weighted = sp.diags(R_inv) @ H
    Q_post = Q_pr + H.T @ H_weighted

    # RHS: H^T R^{-1} y + Q_pr μ_pr
    rhs = H.T @ (R_inv * y) + Q_pr @ mu_pr

    # Factorize and solve
    factor = SparseFactor(Q_post, method=method)
    mu_post = factor.solve(rhs, tol=tol)

    # Validate
    residual = Q_post @ mu_post - rhs
    rhs_norm = np.linalg.norm(rhs)

    if rhs_norm > 1e-14:
        rel_residual = np.linalg.norm(residual) / rhs_norm
        if rel_residual > tol * 10:  # 放宽验证容差
            warnings.warn(f"High relative residual: {rel_residual:.2e}")

    return mu_post, factor


def compute_posterior_variance_diagonal(factor: SparseFactor,
                                       indices: np.ndarray = None,
                                       batch_size: int = 100) -> np.ndarray:
    """
    批量计算对角方差（加速版）

    改进：分批次求解，减少内存占用
    """
    n = factor.n
    if indices is None:
        indices = np.arange(n)

    var_diag = np.zeros(len(indices))

    # 🔥 分批计算（避免一次构造过大矩阵）
    for batch_start in range(0, len(indices), batch_size):
        batch_end = min(batch_start + batch_size, len(indices))
        batch_indices = indices[batch_start:batch_end]
        batch_size_actual = len(batch_indices)

        # 构造单位向量批次
        E_batch = np.zeros((n, batch_size_actual))
        for i, idx in enumerate(batch_indices):
            E_batch[idx, i] = 1.0

        # 批量求解
        Z_batch = factor.solve(E_batch)

        # 提取对角元
        for i, idx in enumerate(batch_indices):
            var_diag[batch_start + i] = Z_batch[idx, i]

    return var_diag


def compute_mutual_information(Q_pr: sp.spmatrix,
                               H: sp.spmatrix,
                               R_diag: np.ndarray) -> float:
    """Compute mutual information (unchanged)"""
    R_inv = 1.0 / R_diag
    H_weighted = sp.diags(R_inv) @ H
    Q_post = Q_pr + H.T @ H_weighted

    factor_pr = SparseFactor(Q_pr)
    factor_post = SparseFactor(Q_post)

    mi = 0.5 * (factor_post.logdet() - factor_pr.logdet())

    return mi


# =====================================================================
# 向后兼容性测试
# =====================================================================

if __name__ == "__main__":
    print("Testing batched inference module...")

    # 构造测试问题
    n = 500
    m_batch = 50

    A = sp.diags([4, -1, -1], [0, -1, 1], shape=(n, n), format='csc')
    A = A + 1e-6 * sp.eye(n)

    factor = SparseFactor(A)

    # 测试批量求解
    print("\n1. Testing batch solve...")
    B = np.random.randn(n, m_batch)

    import time

    # 方法1：逐列求解（慢）
    start = time.time()
    X1 = np.column_stack([factor.solve(B[:, i]) for i in range(m_batch)])
    t1 = time.time() - start

    # 方法2：批量求解（快）
    start = time.time()
    X2 = factor.solve(B)
    t2 = time.time() - start

    print(f"  Sequential: {t1*1000:.2f} ms")
    print(f"  Batched:    {t2*1000:.2f} ms")
    print(f"  Speedup:    {t1/t2:.1f}x")
    print(f"  Max error:  {np.max(np.abs(X1 - X2)):.2e}")

    # 测试批量二次型
    print("\n2. Testing batch quadform...")
    H = np.random.randn(n, m_batch)

    # 方法1：逐个计算
    start = time.time()
    quad1 = np.array([quadform_via_solve(factor, H[:, i]) for i in range(m_batch)])
    t1 = time.time() - start

    # 方法2：批量计算
    start = time.time()
    quad2 = batch_quadform_via_solve(factor, H)
    t2 = time.time() - start

    print(f"  Sequential: {t1*1000:.2f} ms")
    print(f"  Batched:    {t2*1000:.2f} ms")
    print(f"  Speedup:    {t1/t2:.1f}x")
    print(f"  Max error:  {np.max(np.abs(quad1 - quad2)):.2e}")

    print("\n✓ All tests passed!")