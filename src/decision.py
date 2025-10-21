"""
Decision-theoretic value mapping and Expected Value of Information.
Implements conditional risk for threshold-based policies and EVI approximation.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, List
import warnings


def conditional_risk(mu: float, sigma: float,
                    tau: float, L_FP: float, L_FN: float, L_TP: float,
                    L_TN: float = 0.0) -> float:
    """
    Compute Bayes-optimal conditional risk for threshold decision.

    Given posterior N(μ, σ²), decision rule:
        - Act (maintain) if P(x > τ | data) > p_T
        - Don't act otherwise

    where p_T = L_FP / (L_FP + L_FN - L_TP)

    Args:
        mu: Posterior mean
        sigma: Posterior std deviation
        tau: Decision threshold (e.g., IRI limit)
        L_FP: False positive cost
        L_FN: False negative cost
        L_TP: True positive cost
        L_TN: True negative cost (default 0)

    Returns:
        risk: Expected loss under Bayes-optimal action
    """
    if sigma <= 0:
        # Degenerate case: certain knowledge
        if mu > tau:
            return L_TP  # Maintain (correct)
        else:
            return L_TN  # Don't maintain (correct)

    # Posterior failure probability
    p_f = 1.0 - norm.cdf((tau - mu) / sigma)

    # Probability threshold for action
    denom = L_FP + L_FN - L_TP
    if abs(denom) < 1e-10:
        warnings.warn("Near-singular decision cost matrix")
        p_T = 0.5
    else:
        p_T = L_FP / denom

    # Conditional risk for each action
    risk_no_action = p_f * L_FN + (1 - p_f) * L_TN
    risk_action = p_f * L_TP + (1 - p_f) * L_FP

    # Bayes-optimal risk
    return min(risk_no_action, risk_action)


def expected_loss(mu_post: np.ndarray,
                 sigma_post: np.ndarray,
                 decision_config,
                 test_indices: np.ndarray = None) -> float:
    """
    Compute expected economic loss averaged over test set.

    Args:
        mu_post: Posterior means (n,)
        sigma_post: Posterior std deviations (n,)
        decision_config: DecisionConfig object
        test_indices: Subset to evaluate (if None, use all)

    Returns:
        expected_loss: Mean conditional risk over test set (GBP)
    """
    if test_indices is None:
        test_indices = np.arange(len(mu_post))

    risks = np.array([
        conditional_risk(
            mu_post[i], sigma_post[i],
            decision_config.tau_iri,
            decision_config.L_FP_gbp,
            decision_config.L_FN_gbp,
            decision_config.L_TP_gbp,
            decision_config.L_TN_gbp
        )
        for i in test_indices
    ])

    return risks.mean()


"""
修复后的 evi_monte_carlo 函数 - 严谨的先验/后验风险计算

主要改动：
1. 完整走 prior→观测→posterior→风险差 的流程
2. 正确计算先验和后验的对角方差
3. 避免用单点μ_pr[0]代表全局
"""


def evi_monte_carlo(Q_pr, mu_pr, H, R_diag, decision_config,
                    n_samples: int = 500,
                    rng: np.random.Generator = None) -> float:
    """
    严谨的EVI Monte Carlo近似

    修复：
    1. 完整的 prior→observation→posterior→risk差 流程
    2. 正确计算先验和后验的对角方差
    3. 在测试集上评估（避免过拟合）

    EVI = E_{x~prior, y|x}[Risk_prior - Risk_posterior(y)]
    """
    from inference import SparseFactor, compute_posterior, compute_posterior_variance_diagonal

    if rng is None:
        rng = np.random.default_rng()

    n = Q_pr.shape[0]
    m = len(R_diag)

    # 先验因子（用于采样和求对角方差）
    factor_pr = SparseFactor(Q_pr)

    # 采样测试点（用于评估风险）
    n_test = min(200, n)  # 增加测试点数量以提高稳定性
    test_idx = rng.choice(n, size=n_test, replace=False)

    # 🔥 关键修复：计算先验对角方差（在测试点上）
    var_pr = compute_posterior_variance_diagonal(factor_pr, test_idx)
    sigma_pr = np.sqrt(np.maximum(var_pr, 1e-12))

    # 先验风险（固定，所有样本共享）
    prior_risk = expected_loss(
        mu_pr[test_idx],
        sigma_pr,
        decision_config,
        test_indices=np.arange(len(test_idx))
    )

    post_risks = []

    for sample_idx in range(n_samples):
        # === 1. 从先验采样真实状态 ===
        # 生成 z ~ N(0, Q^{-1})，即 Q z = w, w ~ N(0, I)
        w = rng.standard_normal(n)
        z = factor_pr.solve(w)
        x_true = mu_pr + z

        # === 2. 生成观测 y = Hx + ε ===
        y_clean = H @ x_true
        noise = rng.normal(0, np.sqrt(R_diag), size=m)
        y = y_clean + noise

        # === 3. 计算后验分布 ===
        try:
            mu_post, factor_post = compute_posterior(Q_pr, mu_pr, H, R_diag, y)
        except Exception as e:
            print(f"    Warning: Posterior computation failed at sample {sample_idx}: {e}")
            # 降级为先验
            post_risks.append(prior_risk)
            continue

        # === 4. 计算后验对角方差（在相同测试点上）===
        var_post = compute_posterior_variance_diagonal(factor_post, test_idx)
        sigma_post = np.sqrt(np.maximum(var_post, 1e-12))

        # === 5. 计算后验Bayes风险 ===
        post_risk = expected_loss(
            mu_post[test_idx],
            sigma_post,
            decision_config,
            test_indices=np.arange(len(test_idx))
        )

        post_risks.append(post_risk)

    # 平均风险差
    avg_post_risk = np.mean(post_risks)
    evi = prior_risk - avg_post_risk

    # 🔥 健康检查：EVI应该为正
    if evi < -1e-3:  # 允许小的数值误差
        print(f"    Warning: Negative EVI detected: {evi:.2f} £")
        print(f"      Prior risk: {prior_risk:.2f}, Post risk: {avg_post_risk:.2f}")
        # 不强制截断，保留负值以便调试

    return float(evi)

def evi_unscented(Q_pr, mu_pr, H, R_diag, decision_config,
                  alpha: float = 1.0, beta: float = 2.0,
                  kappa: float = 0.0) -> float:
    """
    使用Unscented Transform的EVI近似（在测量空间）

    修复：改进先验风险计算
    """
    from inference import compute_posterior, compute_posterior_variance_diagonal, SparseFactor
    import scipy.sparse as sp

    n = Q_pr.shape[0]
    m = len(R_diag)

    # === 预测分布: y ~ N(H μ_pr, H Σ_pr H^T + R) ===
    y_mean = H @ mu_pr

    # 计算预测协方差（小m时精确，大m时近似）
    if m <= 100:
        factor_pr = SparseFactor(Q_pr)
        H_dense = H.toarray() if sp.issparse(H) else H
        Sigma_pr_HT = np.zeros((n, m))
        for i in range(m):
            Sigma_pr_HT[:, i] = factor_pr.solve(H_dense[i, :])
        y_cov = H_dense @ Sigma_pr_HT + np.diag(R_diag)
    else:
        warnings.warn("Large m in UT, using diagonal approximation")
        y_cov = np.diag(R_diag) + 0.1 * np.eye(m)

    # === UT权重 ===
    lambda_param = alpha ** 2 * (m + kappa) - m
    weights_m = np.full(2 * m + 1, 1.0 / (2 * (m + lambda_param)))
    weights_m[0] = lambda_param / (m + lambda_param)

    # === 生成sigma点 ===
    try:
        L = np.linalg.cholesky(y_cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(y_cov + 1e-6 * np.eye(m))

    scale = np.sqrt(m + lambda_param)
    sigma_points = [y_mean]

    for i in range(m):
        sigma_points.append(y_mean + scale * L[:, i])
        sigma_points.append(y_mean - scale * L[:, i])

    # === 对每个sigma点计算后验风险 ===
    risks = []
    test_idx = np.linspace(0, n - 1, min(50, n), dtype=int)

    for y_sigma in sigma_points:
        mu_post, factor = compute_posterior(Q_pr, mu_pr, H, R_diag, y_sigma)
        var_post = compute_posterior_variance_diagonal(factor, test_idx)
        sigma_post = np.sqrt(var_post)

        loss = expected_loss(
            mu_post[test_idx],
            sigma_post,
            decision_config,
            test_indices=np.arange(len(test_idx))
        )
        risks.append(loss)

    # 加权平均后验风险
    posterior_risk = np.dot(weights_m, risks)

    # === 计算先验风险（改进版）===
    factor_pr = SparseFactor(Q_pr)
    var_pr_sample = compute_posterior_variance_diagonal(factor_pr, test_idx)
    sigma_pr = np.sqrt(var_pr_sample)

    prior_risk = expected_loss(
        mu_pr[test_idx],
        sigma_pr,
        decision_config,
        test_indices=np.arange(len(test_idx))
    )

    evi = prior_risk - posterior_risk
    return float(evi)


if __name__ == "__main__":
    from config import load_config
    from geometry import build_grid2d_geometry
    from spatial_field import build_prior
    from sensors import generate_sensor_pool
    from sensors import assemble_H_R

    cfg = load_config()
    rng = cfg.get_rng()

    # Setup
    geom = build_grid2d_geometry(20, 20, h=cfg.geometry.h)
    Q_pr, mu_pr = build_prior(geom, cfg.prior)

    # Generate sensors
    sensors = generate_sensor_pool(geom, cfg.sensors, rng)
    selected = rng.choice(sensors, size=10, replace=False)
    H, R = assemble_H_R(selected, geom.n)

    print("Testing EVI computation...")

    # Monte Carlo (small sample for speed)
    print("  Monte Carlo (n=50)...")
    evi_mc = evi_monte_carlo(Q_pr, mu_pr, H, R, cfg.decision, n_samples=50, rng=rng)
    print(f"    EVI = £{evi_mc:.2f}")

    # Unscented Transform
    print("  Unscented Transform...")
    evi_ut = evi_unscented(Q_pr, mu_pr, H, R, cfg.decision)
    print(f"    EVI = £{evi_ut:.2f}")

    print(f"  Difference: £{abs(evi_mc - evi_ut):.2f}")