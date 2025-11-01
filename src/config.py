"""
Configuration management for RDT-VoI simulation (Enhanced version)
支持多配置文件和场景检测
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
import sys


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
    """Decision model parameters."""
    L_TP_gbp: float
    L_FP_gbp: float
    L_FN_gbp: float
    L_TN_gbp: float
    tau_iri: float = None
    tau_quantile: float = None
    K_action: int = None
    target_ddi: float = 0.0

    def __post_init__(self):
        if self.tau_iri is None and self.tau_quantile is None:
            raise ValueError("Must specify either tau_iri or tau_quantile")
        if self.tau_iri is not None and self.tau_quantile is not None:
            print(f"  Warning: Both tau_iri and tau_quantile specified. "
                  f"Using tau_quantile={self.tau_quantile}")
        if self.tau_quantile is not None:
            if not (0 < self.tau_quantile < 1):
                raise ValueError(f"tau_quantile must be in (0, 1), got {self.tau_quantile}")

    def get_threshold(self, mu_prior=None):
        """获取决策阈值"""
        if self.tau_quantile is not None:
            if mu_prior is None:
                raise ValueError("tau_quantile mode requires mu_prior")
            tau = float(np.quantile(mu_prior, self.tau_quantile))
            print(f"  Dynamic threshold: τ = quantile(μ_prior, {self.tau_quantile}) = {tau:.3f}")
            return tau
        else:
            return self.tau_iri

    @property
    def prob_threshold(self) -> float:
        """Bayes-optimal probability threshold."""
        return self.L_FP_gbp / (self.L_FP_gbp + self.L_FN_gbp - self.L_TP_gbp)


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


# 🔥 在这里添加新的 MetricsConfig 类：
@dataclass
class MetricsConfig:
    """Metrics computation settings."""
    scale_savings_to_domain: bool = True
    coverage_clip: Tuple[float, float] = (0.0, 1.0)

class Config:
    """Master configuration container (Enhanced)."""

    def __init__(self, config_path: str):
        """必须明确指定配置文件路径"""
        self.config_path = self._find_config(config_path)
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw = yaml.safe_load(f)

        # Parse nested configurations (保持不变)
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

        # 🔥 新增：metrics配置解析
        if 'metrics' in self._raw:
            self.metrics = MetricsConfig(**self._raw['metrics'])
        else:
            self.metrics = MetricsConfig()  # 使用默认值

        self.validate()

    def _find_config(self, config_name: str) -> Path:
        """搜索配置文件"""
        if Path(config_name).exists():
            return Path(config_name)

        current_file = Path(__file__).resolve()
        search_paths = [
            current_file.parent / config_name,
            current_file.parent.parent / config_name,
        ]

        for path in search_paths:
            if path.exists():
                print(f"Found config at: {path}")
                return path

        raise FileNotFoundError(
            f"Could not find '{config_name}'. Searched:\n" +
            "\n".join(f"  - {p}" for p in search_paths) +
            "\n\nAvailable scenarios:\n"
            "  python main.py --scenario A  (config_A_highstakes.yaml)\n"
            "  python main.py --scenario B  (config_B_proxy.yaml)"
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


def load_config(path: str = None) -> Config:
    """
    加载并验证配置

    ✅ 修改：如果不提供路径，优先查找场景配置

    Args:
        path: 配置文件路径，如果为 None 则尝试自动检测

    Examples:
        >>> cfg = load_config("config_A_highstakes.yaml")
        >>> cfg = load_config()  # 自动查找
    """
    if path is None:
        raise ValueError(
            "Must specify config path or use load_scenario_config(scenario)!\n"
            "Examples:\n"
            "  cfg = load_scenario_config('A')  # High-stakes scenario\n"
            "  cfg = load_scenario_config('B')  # Compute/Robustness scenario\n"
            "  cfg = load_config('custom_config.yaml')  # Custom config"
        )

    return Config(path)


def load_scenario_config(scenario: str = 'A') -> Config:
    """
    根据场景类型加载配置（推荐方式）

    Args:
        scenario: 'A' (高风险) 或 'B' (算力/鲁棒性)

    Returns:
        Config object
    """
    if scenario.upper() == 'A':
        config_file = "config_A_highstakes.yaml"
    elif scenario.upper() == 'B':
        config_file = "config_B_proxy.yaml"
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Use 'A' or 'B'")

    return Config(config_file)


if __name__ == "__main__":
    # 测试配置加载
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        cfg = load_scenario_config(scenario)
    else:
        print("Usage: python config.py [A|B]")
        print("Testing with scenario A...")
        cfg = load_scenario_config('A')

    print(f"\nDecision threshold p_T = {cfg.decision.prob_threshold:.3f}")
    print(f"Buffer width = {cfg.cv.buffer_width_multiplier * cfg.prior.correlation_length:.1f} m")