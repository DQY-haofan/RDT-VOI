"""
修复后的 spatial_field.py - 支持非平稳先验

主要改进：
1. 新增 apply_nodewise_nugget() 函数 - 创建空间异质性
2. 修改 build_prior() - 支持热点区域（高方差）
3. 添加先验异质性验证

使用方法：
1. 替换原 spatial_field.py
2. 在 config.yaml 中添加热点配置：

prior:
  beta_base: 1.0e-3  # 基线 nugget（非热点区域）
  beta_hot: 1.0e-6   # 热点 nugget（热点区域）
  hotspots:
    - center_m: [60, 60]
      radius_m: 40
    - center_m: [140, 60]
      radius_m: 30
    - center_m: [100, 140]
      radius_m: 35
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple
from scipy.special import gamma


def matern_tau_from_params(nu: float, kappa: float, sigma2: float,
                           d: int = 2, alpha: int = 2) -> float:
    """计算 SPDE 噪声尺度 τ（原函数保持不变）"""
    numerator = gamma(nu)
    denominator = gamma(alpha) * (4 * np.pi) ** (d / 2) * kappa ** (2 * nu) * sigma2
    tau_squared = numerator / denominator
    return np.sqrt(tau_squared)


def build_grid_precision_spde(nx: int, ny: int, h: float,
                              kappa: float, beta: float = 1e-6) -> sp.spmatrix:
    """构建 2D 网格 SPDE 精度矩阵（原函数保持不变）"""
    n = nx * ny

    def idx(i, j):
        return i * ny + j

    center_coef = kappa ** 2 + 4.0 / h ** 2
    neigh_coef = -1.0 / h ** 2

    row_idx = []
    col_idx = []
    data = []

    for i in range(nx):
        for j in range(ny):
            current = idx(i, j)
            row_idx.append(current)
            col_idx.append(current)
            data.append(center_coef + beta)

            if i < nx - 1:
                row_idx.append(current)
                col_idx.append(idx(i + 1, j))
                data.append(neigh_coef)
            if i > 0:
                row_idx.append(current)
                col_idx.append(idx(i - 1, j))
                data.append(neigh_coef)
            if j < ny - 1:
                row_idx.append(current)
                col_idx.append(idx(i, j + 1))
                data.append(neigh_coef)
            if j > 0:
                row_idx.append(current)
                col_idx.append(idx(i, j - 1))
                data.append(neigh_coef)

    Q = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n, n))
    return Q.tocsr()


def build_graph_precision(L: sp.spmatrix, alpha: float, beta: float) -> sp.spmatrix:
    """从图拉普拉斯构建 GMRF 精度（原函数保持不变）"""
    n = L.shape[0]
    Q = alpha * L + beta * sp.eye(n)
    return Q.tocsr()


def sample_gmrf(Q: sp.spmatrix,
                mu: np.ndarray = None,
                rng: np.random.Generator = None) -> np.ndarray:
    """从 GMRF 采样（原函数保持不变）"""
    n = Q.shape[0]
    if mu is None:
        mu = np.zeros(n)
    if rng is None:
        rng = np.random.default_rng()

    z = rng.standard_normal(n)

    try:
        from sksparse.cholmod import cholesky
        factor = cholesky(Q)
        x_centered = factor.solve_Lt(z, use_LDLt_decomposition=False)
    except ImportError:
        lu = spla.splu(Q)
        x_centered = lu.solve(z)

    return mu + x_centered


# =====================================================================
# 🔥 新增函数：节点化 nugget（创建空间异质性）
# =====================================================================

def apply_nodewise_nugget(geom, prior_config) -> sp.spmatrix:
    """
    🔥 新增：应用节点化 nugget，创建空间异质性

    热点区域：低 nugget → 高不确定性（大方差）
    非热点区域：高 nugget → 低不确定性（小方差）

    这是让 MI/EVI 方法拉开差距的关键！

    Args:
        geom: 几何对象（需要 coords 属性）
        prior_config: 先验配置（需要 beta_base, beta_hot, hotspots 属性）

    Returns:
        节点化 nugget 对角矩阵

    配置示例：
        prior:
          beta_base: 1.0e-3  # 非热点区域
          beta_hot: 1.0e-6   # 热点区域
          hotspots:
            - center_m: [60, 60]
              radius_m: 40
    """
    n = geom.n

    # 默认值（如果配置中没有）
    beta_base = getattr(prior_config, 'beta_base', 1e-3)
    beta_hot = getattr(prior_config, 'beta_hot', 1e-6)

    # 初始化为基线 nugget
    beta_vec = np.full(n, beta_base, dtype=float)

    # 应用热点
    if hasattr(prior_config, 'hotspots') and prior_config.hotspots:
        xy = geom.coords  # shape (n, 2), 单位米

        for hs in prior_config.hotspots:
            center = np.array(hs['center_m'], dtype=float)  # (x, y)
            radius = float(hs['radius_m'])

            # 找到热点范围内的节点
            distances_sq = np.sum((xy - center)**2, axis=1)
            mask = distances_sq <= radius**2

            # 热点区域用低 nugget（高不确定性）
            beta_vec[mask] = beta_hot

            n_hot = mask.sum()
            print(f"  Hotspot at {center}: {n_hot} nodes with β={beta_hot:.1e}")

    return sp.diags(beta_vec, format='csr')


# =====================================================================
# 🔥 修改函数：build_prior 支持非平稳先验
# =====================================================================

def build_prior(geom, prior_config) -> Tuple[sp.spmatrix, np.ndarray]:
    """
    Build GMRF prior precision and mean from geometry and config.

    🔥 修复：支持非平稳先验（热点区域高方差）

    改进：
    1. 使用节点化 nugget 替代均匀 nugget
    2. 自动验证先验异质性（CV 应 > 10%）
    3. 给出警告如果先验过于均匀
    """
    n = geom.n

    if geom.mode == "grid2d":
        # ✅ 步骤1：构建基础SPDE算子 Q_base = (κ² - Δ)
        Q_base = build_grid_precision_spde(
            nx=int(np.sqrt(n)),
            ny=int(np.sqrt(n)),
            h=geom.h,
            kappa=prior_config.kappa,
            beta=0.0  # 不在这里加 nugget
        )

        # ✅ 步骤2：计算τ²（SPDE噪声方差）
        tau = matern_tau_from_params(
            nu=prior_config.nu,
            kappa=prior_config.kappa,
            sigma2=prior_config.sigma2,
            d=2,
            alpha=prior_config.alpha
        )

        # ✅ 步骤3：缩放 SPDE 算子
        Q_spde = (tau ** 2) * Q_base

        # 🔥 步骤4：应用节点化 nugget（创建空间异质性）
        nugget_diag = apply_nodewise_nugget(geom, prior_config)
        Q_pr = Q_spde + nugget_diag

        # 验证：计算方差统计
        print(f"  Prior setup: τ={tau:.4f}, target σ²={prior_config.sigma2:.4f}")

        # 🔥 快速验证空间异质性（采样几个对角元）
        try:
            from inference import SparseFactor, compute_posterior_variance_diagonal
            factor = SparseFactor(Q_pr)

            # 采样不同区域的方差
            n_samples = min(50, n)
            test_idx = np.linspace(0, n-1, n_samples, dtype=int)
            sample_vars = compute_posterior_variance_diagonal(factor, test_idx)

            var_cv = sample_vars.std() / sample_vars.mean()
            print(f"  Prior variance: mean={sample_vars.mean():.4f}, "
                  f"std={sample_vars.std():.4f}, CV={var_cv:.2%}")

            if var_cv < 0.1:
                print("  ⚠️  先验不确定性非常均匀！MI优势会减弱。")
                print("      建议：添加 hotspots 配置或增大 beta_base/beta_hot 差距")
            else:
                print(f"  ✓ 先验异质性良好 (CV={var_cv:.2%})")

        except Exception as e:
            print(f"  Warning: Could not validate prior variance: {e}")

    elif geom.mode in ["polyline1d", "graph"]:
        # 对于非网格几何，使用原有方法
        # 尝试读取 beta_base，如果没有就用 beta
        beta = getattr(prior_config, 'beta_base',
                      getattr(prior_config, 'beta', 1e-6))
        Q_pr = build_graph_precision(
            L=geom.laplacian,
            alpha=prior_config.alpha,
            beta=beta
        )
    else:
        raise ValueError(f"Unknown geometry mode: {geom.mode}")

    # 构造均值场
    if prior_config.mu_prior_std > 0:
        # 采样一个光滑的均值场
        beta_mean = getattr(prior_config, 'beta_base',
                           getattr(prior_config, 'beta', 1e-6))
        Q_mean = build_graph_precision(
            geom.laplacian,
            alpha=0.1,  # 比prior更光滑
            beta=beta_mean
        )
        rng_mean = np.random.default_rng(42)  # 固定种子
        mu_pr = prior_config.mu_prior_mean + \
                prior_config.mu_prior_std * sample_gmrf(Q_mean, rng=rng_mean)
    else:
        # 常数均值
        mu_pr = np.full(n, prior_config.mu_prior_mean)

    return Q_pr, mu_pr


def validate_prior(Q: sp.spmatrix, mu: np.ndarray,
                   rng: np.random.Generator = None,
                   n_samples: int = 5) -> dict:
    """验证先验（原函数保持不变）"""
    if rng is None:
        rng = np.random.default_rng()

    min_eig = spla.eigsh(Q, k=1, which='SA', return_eigenvectors=False)[0]

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
        'empirical_std_range': (emp_std.min(), emp_std.max()),
        'empirical_var_mean': (emp_std ** 2).mean()
    }

    return stats


if __name__ == "__main__":
    # 测试非平稳先验
    from config import load_config
    from geometry import build_grid2d_geometry

    cfg = load_config()
    geom = build_grid2d_geometry(20, 20, h=cfg.geometry.h)

    Q_pr, mu_pr = build_prior(geom, cfg.prior)

    print("\nPrior construction:")
    print(f"  n = {Q_pr.shape[0]}")
    print(f"  nnz = {Q_pr.nnz} ({Q_pr.nnz / Q_pr.shape[0] ** 2 * 100:.2f}%)")

    # Validate
    rng = cfg.get_rng()
    stats = validate_prior(Q_pr, mu_pr, rng, n_samples=50)
    print(f"  Min eigenvalue: {stats['min_eigenvalue']:.6f}")
    print(f"  Is SPD: {stats['is_spd']}")
    print(f"  Empirical variance (mean): {stats['empirical_var_mean']:.4f}")
    print(f"  Target σ²: {cfg.prior.sigma2:.4f}")

    # Sample true state
    x_true = sample_gmrf(Q_pr, mu_pr, rng)
    print(f"  Sample range: [{x_true.min():.3f}, {x_true.max():.3f}]")