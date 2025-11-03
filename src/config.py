"""
Configuration management for RDT-VoI simulation (Enhanced version)
✅ 修改版：单一基准配置 + 参数扫描支持 + 向后兼容

主要改进：
1. 使用单一 baseline_config.yaml 作为默认配置
2. 支持运行时参数覆盖和扫描
3. 保持向后兼容（load_scenario_config 等函数名不变）
4. 新增 apply_parameter_overrides() 功能
"""
import warnings

import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import sys
import copy


@dataclass
class ExperimentConfig:
    """Experiment metadata and control."""
    name: str
    seed: int
    output_dir: Path


@dataclass
class NumericsConfig:
    """Numerical precision and solver settings."""
    linear_solver_tol: float
    cholesky_nugget: float
    pcg_max_iter: int
    logdet_method: str


@dataclass
class GeometryConfig:
    """Spatial domain configuration."""
    mode: str
    nx: int = None
    ny: int = None
    h: float = None
    adjacency_file: Path = None

    @property
    def n_total(self) -> int:
        if self.mode == "grid2d":
            return self.nx * self.ny
        else:
            raise NotImplementedError(f"n_total for mode={self.mode}")


@dataclass
class PriorConfig:
    """GMRF prior hyperparameters"""
    nu: float
    kappa: float
    sigma2: float
    alpha: int
    beta: float
    mu_prior_mean: float
    mu_prior_std: float
    beta_base: float = None
    beta_hot: float = None
    hotspots: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.beta_base is None:
            self.beta_base = self.beta * 100
        if self.beta_hot is None:
            self.beta_hot = self.beta
        if self.hotspots is None:
            self.hotspots = []

    @property
    def correlation_length(self) -> float:
        return np.sqrt(8 * self.nu) / self.kappa


@dataclass
class SensorType:
    """Sensor type specification."""
    name: str
    noise_std: float
    cost_gbp: float
    footprint: str


@dataclass
class SensorsConfig:
    """Sensor pool configuration."""
    types: List[SensorType]
    pool_strategy: str
    pool_fraction: float
    type_mix: List[float]
    use_heterogeneous: bool = False
    cost_zones: List[Dict] = None


@dataclass
class DecisionConfig:
    """Decision-theoretic parameters."""
    L_FP_gbp: float  # False positive cost
    L_FN_gbp: float  # False negative cost
    L_TP_gbp: float  # True positive cost
    L_TN_gbp: float = 0.0  # True negative cost
    tau_quantile: float = 0.75  # Quantile for threshold
    tau_iri: Optional[float] = None  # 🔥 锁定的决策阈值
    target_ddi: Optional[float] = None  # 🔥 目标DDI
    K_action: Optional[int] = None  # 🔥 行动限制

    @property
    def prob_threshold(self) -> float:
        """Compute Bayes-optimal probability threshold."""
        numerator = self.L_FP_gbp - self.L_TN_gbp
        denominator = (self.L_FP_gbp - self.L_TN_gbp) + (self.L_FN_gbp - self.L_TP_gbp)

        if abs(denominator) < 1e-10:
            import warnings
            warnings.warn("Near-singular decision cost matrix, using p_T=0.5")
            return 0.5

        p_T = numerator / denominator
        return np.clip(p_T, 0.0, 1.0)

    def get_threshold(self, mu: np.ndarray) -> float:
        """从成本映射计算阈值"""
        p_T = self.prob_threshold
        tau = float(np.quantile(mu, p_T))
        return tau


@dataclass
class EconomicsConfig:
    """
    🔥 P1-4：经济尺度配置

    用于将评估域（测试集）的损失缩放到业务域（全网络）的等价时间跨度

    Attributes:
        network_km: 业务网络总长度（公里）
        test_km: 单次CV fold测试域覆盖长度（公里）
        horizon_years: 决策评估期（年）
        eval_period_days: 单次评估对应的时间周期（天）

    示例：
        如果网络200km，测试覆盖35km，评估期10年，单次评估7天：
        scale_factor = (200/35) * (10*365/7) ≈ 2940
    """
    network_km: float = 200.0  # 全网络长度
    test_km: float = 35.0  # 测试域长度
    horizon_years: float = 10.0  # 评估期（年）
    eval_period_days: float = 7.0  # 单次评估周期（天）

    @property
    def spatial_scale(self) -> float:
        """空间缩放因子"""
        return self.network_km / self.test_km

    @property
    def temporal_scale(self) -> float:
        """时间缩放因子"""
        horizon_days = self.horizon_years * 365
        return horizon_days / self.eval_period_days

    @property
    def domain_scale_factor(self) -> float:
        """综合缩放因子"""
        return self.spatial_scale * self.temporal_scale


@dataclass
class SelectionConfig:
    """Sensor selection algorithm settings."""
    methods: List[str]
    budgets: List[int]
    greedy_mi: Dict[str, Any]
    budget_type: str = "count"
    greedy_aopt: Dict[str, Any] = None
    greedy_evi: Dict[str, Any] = None
    maxmin: Dict[str, Any] = None

    def __post_init__(self):
        if self.greedy_aopt is None:
            self.greedy_aopt = {'n_probes': 16, 'use_cost': True}
        if self.greedy_evi is None:
            self.greedy_evi = {
                'n_y_samples': 25,
                'use_cost': True,
                'budgets_subset': [],
                'max_folds': None
            }
        if self.maxmin is None:
            self.maxmin = {'use_cost': True}


@dataclass
class EVIConfig:
    """Expected Value of Information computation."""
    compute_for: List[str]
    method: str
    monte_carlo_samples: int
    unscented_alpha: float = 1.0
    unscented_beta: float = 2.0
    unscented_kappa: float = 0.0


@dataclass
class CVConfig:
    """Cross-validation settings."""
    scheme: str
    k_folds: int
    buffer_width_multiplier: float
    block_strategy: str
    ensure_connected: bool
    morans_permutations: int


@dataclass
class UQConfig:
    """Uncertainty quantification settings."""
    bootstrap_method: str
    bootstrap_samples: int
    confidence_level: float
    coverage_percentile: int
    compute_crps: bool


@dataclass
class DiagnosticsConfig:
    """Diagnostic metrics configuration."""
    morans_i: Dict[str, Any]
    calibration: Dict[str, bool]


@dataclass
class PlotsConfig:
    """Visualization settings."""
    save_formats: List[str]
    dpi: int
    style: str
    budget_curves: Dict[str, Any]
    performance_profile: Dict[str, float]
    critical_difference: Dict[str, Any]
    business_metrics: Dict[str, Any] = None
    effect_size: Dict[str, Any] = None
    critical_region: Dict[str, Any] = None
    expert_plots: Dict[str, Any] = None
    roi_curves: Dict[str, Any] = None
    robustness_heatmap: Dict[str, Any] = None
    ddi_overlay: Dict[str, Any] = None

    def __post_init__(self):
        if self.expert_plots is None:
            self.expert_plots = {
                'enable_all': False,
                'marginal_efficiency': {'enable': False},
                'type_composition': {'enable': False},
                'mi_voi_correlation': {'enable': False},
                'calibration_plots': {'enable': False},
                'spatial_diagnostics': {'enable': False},
                'ablation_study': {'enable': False},
                'sensor_placement_map': {'enable': False}
            }
        if self.business_metrics is None:
            self.business_metrics = {'enable': False}
        if self.effect_size is None:
            self.effect_size = {'enable': False}
        if self.critical_region is None:
            self.critical_region = {'enable': False}
        if self.roi_curves is None:
            self.roi_curves = {'enable': True}
        if self.robustness_heatmap is None:
            self.robustness_heatmap = {'enable': False}
        if self.ddi_overlay is None:
            self.ddi_overlay = {'enable': True}


@dataclass
class AcceptanceConfig:
    """Milestone acceptance criteria."""
    m1_grid_size: int
    m1_budgets: List[int]
    m1_check_monotonic: bool
    m1_check_diminishing: bool
    m2_min_improvement_vs_random: float
    m2_confidence_level: float
    m3_small_instance_n: int
    m3_small_instance_k: int
    m3_max_suboptimality: float
    m4_morans_alpha: float
    m4_coverage_tolerance: float
    m4_msse_tolerance: float


@dataclass
class MetricsConfig:
    """Metrics computation settings."""
    scale_savings_to_domain: bool = True
    coverage_clip: Tuple[float, float] = (0.0, 1.0)


class Config:
    """Master configuration container (Enhanced with parameter override support)."""

    def __init__(self, config_path: str):
        """必须明确指定配置文件路径"""
        self.config_path = self._find_config(config_path)
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw = yaml.safe_load(f)

        # Parse nested configurations
        self.experiment = ExperimentConfig(**self._raw['experiment'])
        self.experiment.output_dir = Path(self.experiment.output_dir)

        self.numerics = NumericsConfig(**self._raw['numerics'])
        self.geometry = GeometryConfig(**self._raw['geometry'])
        self.prior = PriorConfig(**self._raw['prior'])

        sensor_types = [SensorType(**st) for st in self._raw['sensors']['types']]
        self.sensors = SensorsConfig(
            types=sensor_types,
            pool_strategy=self._raw['sensors']['pool_strategy'],
            pool_fraction=self._raw['sensors']['pool_fraction'],
            type_mix=self._raw['sensors']['type_mix'],
            use_heterogeneous=self._raw['sensors'].get('use_heterogeneous', False),
            cost_zones=self._raw['sensors'].get('cost_zones', None)
        )

        self.decision = DecisionConfig(**self._raw['decision'])
        self.selection = SelectionConfig(**self._raw['selection'])
        self.evi = EVIConfig(**self._raw['evi'])
        self.cv = CVConfig(**self._raw['cv'])
        self.uq = UQConfig(**self._raw['uq'])
        self.diagnostics = DiagnosticsConfig(**self._raw['diagnostics'])
        self.plots = PlotsConfig(**self._raw['plots'])
        self.acceptance = AcceptanceConfig(**self._raw['acceptance'])

        # metrics配置解析
        if 'metrics' in self._raw:
            self.metrics = MetricsConfig(**self._raw['metrics'])
        else:
            self.metrics = MetricsConfig()

        # 🔥 新增：参数扫描预设
        # 🔥 新增：参数扫描预设
        self.parameter_scan_presets = self._raw.get('parameter_scan_presets', {})

        # 🔥 【必须添加】economics 解析
        if 'economics' in self._raw:
            self.economics = EconomicsConfig(**self._raw['economics'])
        else:
            self.economics = None
        # 🔥 【添加结束】

        self.validate()  # ← 这行必须在最后

    def get_rng(self) -> np.random.Generator:
        """Get seeded random number generator."""
        return np.random.default_rng(self.experiment.seed)

        # 🔥 新增方法1: 锁定决策阈值

    def lock_decision_threshold(self, mu_prior: np.ndarray = None, verbose: bool = True):
        """
        🔥 P0-3：锁定决策阈值（全局统一）

        必须在任何评估/选择算法运行前调用一次。
        之后所有函数使用统一的tau_iri，避免动态计算导致的不一致。

        Args:
            mu_prior: 先验均值（可选，用于从分位数计算tau）
            verbose: 是否打印锁定信息

        使用方法：
            在main.py中，构建先验后立即调用：

            ```python
            Q_pr, mu_pr = build_prior(geom, config.prior)
            config.lock_decision_threshold(mu_pr)  # 🔥 关键步骤
            ```

        注意：
            - 如果tau_iri已设置，不会重复计算
            - 如果使用tau_quantile，会自动计算并缓存tau_iri
            - 锁定后，tau_quantile将被禁用（避免不一致）
        """
        # 如果已锁定，跳过
        if hasattr(self.decision, 'tau_iri') and self.decision.tau_iri is not None:
            if verbose:
                print(f"  ℹ️  Decision threshold already locked: τ = {self.decision.tau_iri:.3f}")
            return

        # 从分位数计算阈值
        if hasattr(self.decision, 'tau_quantile') and self.decision.tau_quantile is not None:
            if mu_prior is None:
                raise ValueError(
                    "mu_prior is required when tau_quantile is set. "
                    "Call lock_decision_threshold(mu_pr) after building prior."
                )

            tau = float(np.quantile(mu_prior, self.decision.tau_quantile))
            self.decision.tau_iri = tau

            if verbose:
                print(f"  🔒 Decision threshold locked from quantile {self.decision.tau_quantile:.2f}")
                print(f"     τ_IRI = {tau:.3f}")

            # 🔥 关键：禁用tau_quantile，避免后续函数误用
            self.decision.tau_quantile = None

        else:
            raise ValueError(
                "Cannot lock threshold: neither tau_iri nor tau_quantile is set in config. "
                "Set one of them in baseline_config.yaml."
            )

    def get_domain_scale_factor(self, verbose: bool = False) -> float:
        """
        🔥 P1-4：获取域缩放因子

        将评估域（测试集）的损失缩放到业务域（全网络）的等价时间跨度

        Returns:
            domain_scale_factor: 缩放因子（≥1）

        使用方法：
            在evaluation.py中计算指标时使用：

            ```python
            scale_factor = config.get_domain_scale_factor()
            metrics = compute_enhanced_metrics(
                ...,
                domain_scale_factor=scale_factor
            )
            ```
        """
        if not hasattr(self, 'economics'):
            if verbose:
                warnings.warn(
                    "No 'economics' section in config. "
                    "Using default scale_factor=1.0 (no scaling). "
                    "Add economics section to baseline_config.yaml to enable scaling."
                )
            return 1.0

        # 使用EconomicsConfig的属性
        scale_factor = self.economics.domain_scale_factor

        if verbose:
            print(f"  📊 Domain scaling:")
            print(f"     Spatial: {self.economics.spatial_scale:.1f}x "
                  f"({self.economics.network_km}km / {self.economics.test_km}km)")
            print(f"     Temporal: {self.economics.temporal_scale:.1f}x "
                  f"({self.economics.horizon_years}y / {self.economics.eval_period_days}d)")
            print(f"     Combined: {scale_factor:.0f}x")

        # 健康检查
        if scale_factor < 1.0:
            warnings.warn(f"Computed scale_factor={scale_factor:.2f} < 1, clamping to 1.0")
            scale_factor = 1.0

        return scale_factor

    def validate_economics_config(self) -> bool:
        """
        验证economics配置的合理性

        Returns:
            True if valid, False otherwise
        """
        if not hasattr(self, 'economics'):
            return False

        econ = self.economics

        # 检查必需字段
        required_fields = ['network_km', 'test_km', 'horizon_years', 'eval_period_days']
        for field in required_fields:
            if not hasattr(econ, field):
                warnings.warn(f"Economics config missing field: {field}")
                return False

        # 合理性检查
        if econ.network_km <= econ.test_km:
            warnings.warn(
                f"network_km ({econ.network_km}) should be > test_km ({econ.test_km})"
            )
            return False

        if econ.horizon_years <= 0 or econ.eval_period_days <= 0:
            warnings.warn("horizon_years and eval_period_days must be positive")
            return False

        if econ.eval_period_days > econ.horizon_years * 365:
            warnings.warn("eval_period_days should not exceed horizon in days")
            return False

        return True

    def print_config_summary(self, include_economics: bool = True):
        """
        🔥 增强的配置摘要打印

        Args:
            include_economics: 是否包含经济尺度信息
        """
        print("\n" + "=" * 70)
        print("  CONFIGURATION SUMMARY")
        print("=" * 70)

        print(f"\n[Experiment]")
        print(f"  Name: {self.experiment.name}")
        print(f"  Seed: {self.experiment.seed}")

        print(f"\n[Geometry]")
        print(f"  Mode: {self.geometry.mode}")
        if self.geometry.mode == "grid2d":
            print(f"  Grid: {self.geometry.nx}×{self.geometry.ny} = {self.geometry.n_total} cells")
            print(f"  Spacing: {self.geometry.h}m")

        print(f"\n[Prior]")
        print(f"  Correlation length: {self.prior.correlation_length:.1f}m")
        print(f"  Target variance: σ² = {self.prior.sigma2:.3f}")
        print(f"  Spatial smoothing: α = {self.prior.alpha:.2e}")
        print(f"  Nugget: β_base = {self.prior.beta_base:.2e}, β_hot = {self.prior.beta_hot:.2e}")
        if self.prior.hotspots:
            print(f"  Hotspots: {len(self.prior.hotspots)} regions")

        print(f"\n[Sensors]")
        print(f"  Types: {len(self.sensors.types)}")
        print(f"  Pool strategy: {self.sensors.pool_strategy}")
        print(f"  Pool fraction: {self.sensors.pool_fraction:.1%}")

        print(f"\n[Decision]")
        print(f"  L_FP: £{self.decision.L_FP_gbp:,.0f}")
        print(f"  L_FN: £{self.decision.L_FN_gbp:,.0f}")
        print(f"  L_TP: £{self.decision.L_TP_gbp:,.0f}")
        print(f"  FN/FP ratio: {self.decision.L_FN_gbp / self.decision.L_FP_gbp:.1f}:1")
        print(f"  Prob threshold: p_T = {self.decision.prob_threshold:.3f}")

        if hasattr(self.decision, 'tau_iri') and self.decision.tau_iri is not None:
            print(f"  🔒 Threshold locked: τ = {self.decision.tau_iri:.3f}")

        if hasattr(self.decision, 'target_ddi') and self.decision.target_ddi is not None:
            print(f"  Target DDI: {self.decision.target_ddi:.1%}")

        if hasattr(self.decision, 'K_action') and self.decision.K_action is not None:
            print(f"  Action limit: K = {self.decision.K_action}")

        # 🔥 Economics信息
        if include_economics and hasattr(self, 'economics'):
            print(f"\n[Economics]")
            scale_factor = self.get_domain_scale_factor(verbose=False)
            print(f"  Network span: {self.economics.network_km}km")
            print(f"  Test span: {self.economics.test_km}km")
            print(f"  Evaluation horizon: {self.economics.horizon_years}y")
            print(f"  Eval period: {self.economics.eval_period_days}d")
            print(f"  → Domain scale factor: {scale_factor:.0f}x")

        print(f"\n[Selection]")
        print(f"  Methods: {', '.join(self.selection.methods)}")
        print(f"  Budgets: {self.selection.budgets}")

        print(f"\n[Cross-Validation]")
        print(f"  Scheme: {self.cv.scheme}")
        print(f"  Folds: {self.cv.k_folds}")

        print("=" * 70 + "\n")

    # ============================================================================
    # 🔥 Config类构造函数的增强（添加economics解析）
    # ============================================================================

    def _parse_config_with_economics(self, cfg_dict: dict):
        """
        🔥 增强的配置解析，添加economics支持

        在Config.__init__()中调用此函数来解析economics部分

        使用方法：
            在Config.__init__()的最后添加：

            ```python
            # 解析economics（如果存在）
            if 'economics' in cfg_dict:
                self.economics = EconomicsConfig(**cfg_dict['economics'])
            else:
                self.economics = None  # 可选
            ```
        """
        if 'economics' in cfg_dict:
            econ_dict = cfg_dict['economics']
            self.economics = EconomicsConfig(**econ_dict)
        else:
            # 使用默认值（可选，或设为None）
            self.economics = None


    def verify_threshold_locked(self) -> bool:
        """
        验证阈值是否已锁定

        Returns:
            True if threshold is locked, False otherwise
        """
        if not hasattr(self, 'decision'):
            return False

        return (hasattr(self.decision, 'tau_iri') and
                self.decision.tau_iri is not None and
                np.isfinite(self.decision.tau_iri))

        # 🔥 新增方法3: 获取已锁定的阈值

    def get_locked_threshold(self) -> float:
        """
        获取已锁定的阈值

        Returns:
            tau: 锁定的阈值

        Raises:
            RuntimeError: 如果阈值未锁定
        """
        if not self.verify_threshold_locked():
            raise RuntimeError(
                "Threshold not locked! Call lock_decision_threshold() first."
            )
        return self.decision.tau_iri

    def _parse_config_with_economics(self, cfg_dict: dict):
        """
        🔥 增强的配置解析，添加economics支持

        在Config.__init__()中调用此函数来解析economics部分

        使用方法：
            在Config.__init__()的最后添加：

            ```python
            # 解析economics（如果存在）
            if 'economics' in cfg_dict:
                self.economics = EconomicsConfig(**cfg_dict['economics'])
            else:
                self.economics = None  # 可选
            ```
        """
        if 'economics' in cfg_dict:
            econ_dict = cfg_dict['economics']
            self.economics = EconomicsConfig(**econ_dict)
        else:
            # 使用默认值（可选，或设为None）
            self.economics = None

    def _find_config(self, config_name: str) -> Path:
        """搜索配置文件"""
        if Path(config_name).exists():
            return Path(config_name)

        current_file = Path(__file__).resolve()
        search_paths = [
            current_file.parent / config_name,
            current_file.parent.parent / config_name,
            # 🔥 添加对基准配置的搜索
            current_file.parent / "baseline_config.yaml",
            current_file.parent.parent / "baseline_config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                print(f"Found config at: {path}")
                return path

        raise FileNotFoundError(
            f"Could not find '{config_name}'. Searched:\n" +
            "\n".join(f"  - {p}" for p in search_paths) +
            "\n\nUsage:\n"
            "  python main.py                    # Uses baseline_config.yaml\n"
            "  python main.py --config custom.yaml  # Uses custom config\n"
            "  python main.py --preset high_stakes  # Applies preset overrides"
        )

    def validate(self):
        """Validate configuration consistency."""
        assert all('_gbp' in k for k in vars(self.decision) if k.startswith('L_'))

        p_T = self.decision.prob_threshold
        assert 0 < p_T < 1, f"Invalid prob_threshold={p_T}"

        max_budget = max(self.selection.budgets)
        pool_size = int(self.geometry.n_total * self.sensors.pool_fraction)
        assert max_budget <= pool_size, f"Budget {max_budget} exceeds pool {pool_size}"

        assert abs(sum(self.sensors.type_mix) - 1.0) < 1e-6

        print(f"✓ Configuration validated: {self.experiment.name}")
        print(f"  Domain: {self.geometry.mode}, n={self.geometry.n_total}")
        print(f"  Correlation length: {self.prior.correlation_length:.2f} m")
        print(f"  CV: {self.cv.k_folds}-fold {self.cv.scheme}")
        print(f"  Methods: {', '.join(self.selection.methods)}")

    def get_rng(self) -> np.random.Generator:
        """Get seeded random number generator."""
        return np.random.default_rng(self.experiment.seed)

    def save_to(self, output_dir: Path):
        """Save a copy of config to output directory."""
        output_path = output_dir / "config.yaml"
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._raw, f, default_flow_style=False)
        print(f"  Config saved to {output_path}")

    # 🔥 新增：参数覆盖功能
    def apply_parameter_overrides(self, overrides: Dict[str, Any],
                                 verbose: bool = True) -> 'Config':
        """
        应用参数覆盖，返回新的 Config 实例

        Args:
            overrides: 参数覆盖字典，如 {'target_ddi': 0.30, 'L_FN_gbp': 120000}
            verbose: 是否打印覆盖信息

        Returns:
            新的 Config 实例
        """
        # 深拷贝原始配置
        new_raw = copy.deepcopy(self._raw)

        if verbose and overrides:
            print(f"\n  🔧 Applying parameter overrides:")

        for key, value in overrides.items():
            if self._apply_single_override(new_raw, key, value, verbose):
                if verbose:
                    print(f"    ✓ {key} = {value}")
            else:
                if verbose:
                    print(f"    ✗ Unknown parameter: {key}")

        # 创建新的 Config 实例
        temp_path = Path("temp_config.yaml")
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_raw, f, default_flow_style=False)

            new_config = Config(str(temp_path))
            # 更新实验名称以反映参数覆盖
            if overrides:
                override_str = "_".join(f"{k}{v}" for k, v in list(overrides.items())[:3])
                new_config.experiment.name = f"{self.experiment.name}_{override_str}"

            return new_config
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _apply_single_override(self, config_dict: Dict, key: str, value: Any,
                              verbose: bool = True) -> bool:
        """
        应用单个参数覆盖到配置字典

        Returns:
            bool: 是否成功应用覆盖
        """
        # 定义参数映射：命令行参数名 -> 配置路径
        param_mappings = {
            # Decision parameters
            'target_ddi': ['decision', 'target_ddi'],
            'ddi': ['decision', 'target_ddi'],
            'L_FN_gbp': ['decision', 'L_FN_gbp'],
            'fn_cost': ['decision', 'L_FN_gbp'],
            'L_FP_gbp': ['decision', 'L_FP_gbp'],
            'fp_cost': ['decision', 'L_FP_gbp'],
            'tau_quantile': ['decision', 'tau_quantile'],
            'K_action': ['decision', 'K_action'],
            'action_limit': ['decision', 'K_action'],

            # EVI parameters
            'monte_carlo_samples': ['evi', 'monte_carlo_samples'],
            'mc_samples': ['evi', 'monte_carlo_samples'],
            'n_y_samples': ['selection', 'greedy_evi', 'n_y_samples'],

            # CV parameters
            'k_folds': ['cv', 'k_folds'],
            'folds': ['cv', 'k_folds'],

            # Geometry parameters
            'nx': ['geometry', 'nx'],
            'ny': ['geometry', 'ny'],
            'grid_size': ['geometry', 'nx'],  # 同时设置 nx 和 ny

            # Pool parameters
            'pool_fraction': ['sensors', 'pool_fraction'],
            'pool_size': ['sensors', 'pool_fraction'],

            # Budget parameters
            'budgets': ['selection', 'budgets'],

            # Method selection
            'methods': ['selection', 'methods'],

            # Seed
            'seed': ['experiment', 'seed'],
        }

        if key not in param_mappings:
            return False

        path = param_mappings[key]

        # 特殊处理：grid_size 同时设置 nx 和 ny
        if key == 'grid_size':
            self._set_nested_value(config_dict, ['geometry', 'nx'], value)
            self._set_nested_value(config_dict, ['geometry', 'ny'], value)
            return True

        # 特殊处理：action_limit 为 None 的情况
        if key in ['K_action', 'action_limit'] and value in ['null', 'none', 'None']:
            value = None

        # 特殊处理：budgets 和 methods 列表
        if key in ['budgets', 'methods'] and isinstance(value, str):
            if ',' in value:
                value = [item.strip() for item in value.split(',')]
                # 对于 budgets，转换为整数
                if key == 'budgets':
                    value = [int(x) for x in value]

        self._set_nested_value(config_dict, path, value)
        return True

    def _set_nested_value(self, config_dict: Dict, path: List[str], value: Any):
        """在嵌套字典中设置值"""
        current = config_dict
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def apply_preset(self, preset_name: str, verbose: bool = True) -> 'Config':
        """
        应用预设配置

        Args:
            preset_name: 预设名称（如 'high_stakes', 'low_stakes'）
            verbose: 是否打印应用信息

        Returns:
            新的 Config 实例
        """
        if preset_name not in self.parameter_scan_presets:
            available_presets = list(self.parameter_scan_presets.keys())
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available presets: {available_presets}"
            )

        preset_params = self.parameter_scan_presets[preset_name]
        if verbose:
            print(f"\n  🎯 Applying preset: {preset_name}")

        return self.apply_parameter_overrides(preset_params, verbose)


# ============================================================================
# 🔥 新增：参数扫描功能
# ============================================================================

def generate_parameter_combinations(scan_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    生成参数扫描的所有组合

    Args:
        scan_params: 扫描参数字典，如 {'target_ddi': [0.1, 0.2, 0.3], 'L_FN_gbp': [30000, 60000]}

    Returns:
        参数组合列表
    """
    from itertools import product

    if not scan_params:
        return [{}]

    keys = list(scan_params.keys())
    values = list(scan_params.values())

    combinations = []
    for value_combo in product(*values):
        combo_dict = dict(zip(keys, value_combo))
        combinations.append(combo_dict)

    return combinations


def parse_scan_parameter(param_string: str) -> List[Union[int, float, str]]:
    """
    解析扫描参数字符串

    Args:
        param_string: 如 "0.1,0.2,0.3" 或 "30000,60000,120000"

    Returns:
        解析后的值列表
    """
    if not param_string or param_string.strip() == '':
        return []

    values = []
    for item in param_string.split(','):
        item = item.strip()

        # 尝试转换为数字
        try:
            if '.' in item:
                values.append(float(item))
            else:
                values.append(int(item))
        except ValueError:
            # 保持为字符串
            values.append(item)

    return values


# ============================================================================
# 🔥 向后兼容的函数（保持原有函数名）
# ============================================================================

def load_config(path: str = None) -> Config:
    """
    加载配置文件

    🔥 修改：默认使用 baseline_config.yaml
    """
    if path is None:
        path = "baseline_config.yaml"

    return Config(path)


def load_scenario_config(scenario: str = None) -> Config:
    """
    🔥 向后兼容函数：模拟原有的场景加载行为

    现在通过预设方式实现场景切换
    """
    base_config = load_config("baseline_config.yaml")

    if scenario is None:
        print("  ℹ️  No scenario specified, using baseline configuration")
        return base_config

    scenario_upper = scenario.upper()

    if scenario_upper == 'A':
        # 高风险场景
        print("  🎯 Loading Scenario A (High-stakes) via preset")
        return base_config.apply_preset('high_stakes', verbose=False)

    elif scenario_upper == 'B':
        # 低风险场景
        print("  🎯 Loading Scenario B (Low-stakes) via preset")
        return base_config.apply_preset('low_stakes', verbose=False)

    else:
        print(f"  ⚠️  Unknown scenario '{scenario}', using baseline")
        return base_config


def detect_scenario_from_config(cfg) -> str:
    """
    🔥 向后兼容函数：从配置推断场景类型
    """
    ddi = getattr(cfg.decision, 'target_ddi', 0.20)
    fn_fp_ratio = cfg.decision.L_FN_gbp / cfg.decision.L_FP_gbp if cfg.decision.L_FP_gbp > 0 else 1.0

    if ddi >= 0.25 or fn_fp_ratio > 8:
        return 'A'  # 高风险
    elif ddi <= 0.15 or fn_fp_ratio < 3:
        return 'B'  # 低风险
    else:
        return 'M'  # 中等风险（新类型）


# ============================================================================
# 测试和示例用法
# ============================================================================


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  TESTING CONFIG ENHANCEMENTS")
    print("=" * 70)

    # Test 1: EconomicsConfig
    print("\n[Test 1] Economics configuration")
    econ = EconomicsConfig(
        network_km=200,
        test_km=35,
        horizon_years=10,
        eval_period_days=7
    )

    print(f"  Spatial scale: {econ.spatial_scale:.1f}x")
    print(f"  Temporal scale: {econ.temporal_scale:.1f}x")
    print(f"  Domain scale factor: {econ.domain_scale_factor:.0f}x")

    # Test 2: 使用真实的 Config 对象（而不是 Mock）
    print("\n[Test 2] Threshold locking with real Config")
    try:
        # 加载真实配置
        cfg = load_config("baseline_config.yaml")

        # 模拟先验
        mu_pr = np.random.normal(2.2, 0.3, 100)

        # 锁定阈值
        cfg.lock_decision_threshold(mu_pr, verbose=True)

        # 验证
        assert cfg.decision.tau_iri is not None, "tau_iri should be set"
        print(f"  ✓ Threshold locked: τ = {cfg.decision.tau_iri:.3f}")

        # Test 3: Domain scale factor
        print("\n[Test 3] Domain scale factor")
        scale = cfg.get_domain_scale_factor(verbose=True)
        assert scale > 1, "Scale factor should be > 1"
        print(f"  ✓ Scale factor: {scale:.0f}")

        # Test 4: Economics validation
        print("\n[Test 4] Economics validation")
        is_valid = cfg.validate_economics_config()
        print(f"  Economics config valid: {is_valid}")

        print("\n✅ All config enhancement tests passed!")

    except FileNotFoundError:
        print("  ⚠️  Could not find baseline_config.yaml, skipping real config test")
        print("  Run this test from the project root directory")