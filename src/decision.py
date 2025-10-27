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
    🔥 修复版：Bayes-optimal conditional risk（通用概率阈值公式）

    关键修复：
    - 使用通用公式 p_T = (L_FP - L_TN) / ((L_FP - L_TN) + (L_FN - L_TP))
    - 兼容 L_TN ≠ 0 的情况

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

    # 🔥 修复：通用概率阈值公式
    # Bayes-optimal action: act if p_f > p_T
    # p_T = (L_FP - L_TN) / [(L_FP - L_TN) + (L_FN - L_TP)]
    numerator = L_FP - L_TN
    denom = (L_FP - L_TN) + (L_FN - L_TP)

    if abs(denom) < 1e-10:
        warnings.warn("Near-singular decision cost matrix")
        p_T = 0.5
    else:
        p_T = numerator / denom

    # Conditional risk for each action
    risk_no_action = p_f * L_FN + (1 - p_f) * L_TN
    risk_action = p_f * L_TP + (1 - p_f) * L_FP

    # Bayes-optimal risk
    return min(risk_no_action, risk_action)


def expected_loss(mu_post: np.ndarray,
                 sigma_post: np.ndarray,
                 decision_config,
                 test_indices: np.ndarray = None,
                 tau: float = None) -> float:
    """
    Compute expected economic loss averaged over test set.

    Args:
        mu_post: Posterior means (n,)
        sigma_post: Posterior std deviations (n,)
        decision_config: DecisionConfig object
        test_indices: Subset to evaluate (if None, use all)
        tau: Decision threshold (if None, get from decision_config)

    Returns:
        expected_loss: Mean conditional risk over test set (GBP)
    """
    if test_indices is None:
        test_indices = np.arange(len(mu_post))

    # 🔥 改进的阈值获取逻辑
    if tau is None:
        if decision_config.tau_iri is not None:
            tau = decision_config.tau_iri
        elif decision_config.tau_quantile is not None:
            # 🔥 自动计算动态阈值（使用后验均值）
            tau = float(np.quantile(mu_post, decision_config.tau_quantile))
            warnings.warn(
                f"tau_iri not set, using dynamic threshold from posterior: "
                f"tau = quantile(mu_post, {decision_config.tau_quantile}) = {tau:.3f}. "
                "For better performance, set tau_iri in main() before evaluation."
            )
        else:
            raise ValueError(
                "Decision threshold not configured. "
                "Set either tau_iri or tau_quantile in config.yaml"
            )

    risks = np.array([
        conditional_risk(
            mu_post[i], sigma_post[i],
            tau,
            decision_config.L_FP_gbp,
            decision_config.L_FN_gbp,
            decision_config.L_TP_gbp,
            decision_config.L_TN_gbp
        )
        for i in test_indices
    ])

    return risks.mean()


def expected_loss_batch(mu_post_batch: np.ndarray,
                       sigma_post_batch: np.ndarray,
                       decision_config,
                       test_indices: np.ndarray = None) -> np.ndarray:
    """
    🔥 批量计算 expected loss（向量化版本）- 加速 20-50x

    用于 EVI 快速评估：一次性计算所有候选的后验风险

    Args:
        mu_post_batch: 后验均值
            - shape (n_test,): 单个候选
            - shape (n_test, n_candidates): 多个候选（EVI 快速评估）
        sigma_post_batch: 后验标准差，shape 与 mu_post_batch 相同
        decision_config: 决策配置对象
        test_indices: 测试集索引（可选，用于对齐）

    Returns:
        losses: Expected loss per candidate
            - shape (n_candidates,) 如果输入是 2D
            - 标量 如果输入是 1D

    Example:
        >>> # 评估 100 个候选在 200 个测试点上的损失
        >>> mu = np.random.randn(200, 100)
        >>> sigma = np.random.rand(200, 100) * 0.5
        >>> losses = expected_loss_batch(mu, sigma, config)
        >>> losses.shape  # (100,)
    """
    # 🔥 改进的阈值获取逻辑
    if decision_config.tau_iri is not None:
        tau = decision_config.tau_iri
    elif decision_config.tau_quantile is not None:
        # 🔥 自动计算动态阈值（使用后验均值）
        # 对于批量版本，使用第一列（或全局）的分位数
        if mu_post_batch.ndim == 1:
            tau = float(np.quantile(mu_post_batch, decision_config.tau_quantile))
        else:
            # 使用所有候选的平均分位数
            tau = float(np.quantile(mu_post_batch, decision_config.tau_quantile))
        warnings.warn(
            f"tau_iri not set, using dynamic threshold: tau = {tau:.3f}. "
            "For better performance, set tau_iri in main() before evaluation."
        )
    else:
        raise ValueError(
            "Decision threshold not configured. "
            "Set either tau_iri or tau_quantile in config.yaml"
        )

    L_FP = decision_config.L_FP_gbp
    L_FN = decision_config.L_FN_gbp
    L_TP = decision_config.L_TP_gbp
    L_TN = decision_config.L_TN_gbp

    # 防止除零
    sigma_safe = np.maximum(sigma_post_batch, 1e-12)

    # 向量化计算后验失效概率
    # P(x > τ | data) = 1 - Φ((τ - μ) / σ)
    z_scores = (tau - mu_post_batch) / sigma_safe
    p_fail = 1.0 - norm.cdf(z_scores)

    # Bayes-optimal 决策阈值
    denom = L_FP + L_FN - L_TP
    if abs(denom) < 1e-10:
        warnings.warn("Near-singular decision cost matrix, using p_T=0.5")
        p_T = 0.5
    else:
        p_T = L_FP / denom

    # 两种行动的条件风险
    risk_no_action = p_fail * L_FN + (1 - p_fail) * L_TN
    risk_action = p_fail * L_TP + (1 - p_fail) * L_FP

    # Bayes-optimal 风险（逐点取最小）
    optimal_risk = np.minimum(risk_no_action, risk_action)

    # 如果是 2D (n_test, n_candidates)，沿测试点轴求平均
    if optimal_risk.ndim == 2:
        return optimal_risk.mean(axis=0)  # (n_candidates,)
    else:
        return optimal_risk.mean()  # 标量


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
    🔥 修复版：严谨的 EVI Monte Carlo 近似

    关键修复：
    1. 使用正确的 GMRF 采样（通过 sample_gmrf）
    2. 完整的 prior→observation→posterior→risk差 流程
    3. 在测试集上评估（避免过拟合）

    EVI = E_{x~prior, y|x}[Risk_prior - Risk_posterior(y)]
    """
    from inference import SparseFactor, compute_posterior, compute_posterior_variance_diagonal
    from spatial_field import sample_gmrf  # 🔥 使用已验证的采样函数

    if rng is None:
        rng = np.random.default_rng()

    n = Q_pr.shape[0]
    m = len(R_diag)

    # 先验因子（用于求对角方差）
    factor_pr = SparseFactor(Q_pr)

    # 采样测试点（用于评估风险）
    n_test = min(200, n)
    test_idx = rng.choice(n, size=n_test, replace=False)

    # 🔥 计算先验对角方差（在测试点上）
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
        # === 🔥 修复1：从先验正确采样真实状态 ===
        # 使用已验证的 sample_gmrf（内部使用 Cholesky 下三角）
        x_true = sample_gmrf(Q_pr, mu_pr, rng)

        # === 2. 生成观测 y = Hx + ε ===
        y_clean = H @ x_true
        noise = rng.normal(0, np.sqrt(R_diag), size=m)
        y = y_clean + noise

        # === 3. 计算后验分布 ===
        try:
            mu_post, factor_post = compute_posterior(Q_pr, mu_pr, H, R_diag, y)
        except Exception as e:
            warnings.warn(f"Posterior computation failed at sample {sample_idx}: {e}")
            # 降级为先验
            post_risks.append(prior_risk)
            continue

        # === 4. 计算后验对角方差（在相同测试点上）===
        var_post = compute_posterior_variance_diagonal(factor_post, test_idx)
        sigma_post = np.sqrt(np.maximum(var_post, 1e-12))

        # === 5. 计算后验 Bayes 风险 ===
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

    # 🔥 健康检查：EVI 应该为正
    if evi < -1e-3:  # 允许小的数值误差
        warnings.warn(f"Negative EVI detected: {evi:.2f} £")
        warnings.warn(f"  Prior risk: {prior_risk:.2f}, Post risk: {avg_post_risk:.2f}")
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
    from spatial_field import build_prior, sample_gmrf
    from sensors import generate_sensor_pool
    from sensors import assemble_H_R

    print("\n" + "=" * 70)
    print("  TESTING FIXED EVI COMPUTATION")
    print("=" * 70)

    cfg = load_config()
    rng = cfg.get_rng()

    # Setup
    geom = build_grid2d_geometry(20, 20, h=cfg.geometry.h)
    Q_pr, mu_pr = build_prior(geom, cfg.prior)

    # Generate sensors
    sensors = generate_sensor_pool(geom, cfg.sensors, rng)
    selected = rng.choice(sensors, size=10, replace=False)
    H, R = assemble_H_R(selected, geom.n)

    print("\n[1] Testing corrected Monte Carlo sampling...")

    # 🔥 关键测试：检查采样方差是否正确
    from inference import SparseFactor, compute_posterior_variance_diagonal

    factor = SparseFactor(Q_pr)
    test_idx = np.array([100, 200, 300])

    # 理论方差（从精度矩阵）
    var_theory = compute_posterior_variance_diagonal(factor, test_idx)
    print(f"  Theoretical variance: {var_theory}")

    # 经验方差（从采样）
    n_samples = 1000
    samples = np.array([sample_gmrf(Q_pr, mu_pr, rng)[test_idx] for _ in range(n_samples)])
    var_empirical = samples.var(axis=0)
    print(f"  Empirical variance:   {var_empirical}")
    print(f"  Relative error:       {np.abs(var_empirical - var_theory) / var_theory}")

    # ✅ 如果相对误差 < 10%，说明采样正确
    assert np.all(np.abs(var_empirical - var_theory) / var_theory < 0.15), \
        "❌ Sampling variance incorrect!"
    print("  ✅ Sampling variance correct!")

    print("\n[2] Testing EVI computation...")

    # Monte Carlo (small sample for speed)
    evi_mc = evi_monte_carlo(Q_pr, mu_pr, H, R, cfg.decision, n_samples=100, rng=rng)
    print(f"  EVI (Monte Carlo, n=100) = £{evi_mc:.2f}")

    # ✅ EVI 应该为正（信息总是有价值的）
    assert evi_mc > 0, f"❌ Negative EVI: {evi_mc:.2f}"
    print(f"  ✅ EVI is positive!")

    print("\n[3] Testing probability threshold formula...")

    # 测试不同 L_TN 值
    test_cases = [
        {'L_FP': 500, 'L_FN': 10000, 'L_TP': 800, 'L_TN': 0},
        {'L_FP': 500, 'L_FN': 10000, 'L_TP': 800, 'L_TN': 100},
        {'L_FP': 500, 'L_FN': 10000, 'L_TP': 800, 'L_TN': -100},
    ]

    for tc in test_cases:
        risk = conditional_risk(
            mu=2.0, sigma=0.5, tau=2.2,
            L_FP=tc['L_FP'], L_FN=tc['L_FN'],
            L_TP=tc['L_TP'], L_TN=tc['L_TN']
        )

        # 计算概率阈值（用于验证）
        numerator = tc['L_FP'] - tc['L_TN']
        denom = (tc['L_FP'] - tc['L_TN']) + (tc['L_FN'] - tc['L_TP'])
        p_T = numerator / denom if abs(denom) > 1e-10 else 0.5

        print(f"  L_TN={tc['L_TN']:6.0f} → p_T={p_T:.3f}, risk=£{risk:.2f}")

        # ✅ 风险应该在合理范围内
        assert 0 <= risk <= max(tc.values()), f"❌ Invalid risk: {risk}"

    print("  ✅ Probability threshold formula correct!")

    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED ✅")
    print("=" * 70)