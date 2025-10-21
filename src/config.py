"""
Configuration management for RDT-VoI simulation.
Loads, validates, and provides typed access to config.yaml.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np


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
    mode: str  # "grid2d" | "polyline1d" | "graph"
    nx: int = None
    ny: int = None
    h: float = None
    adjacency_file: Path = None

    @property
    def n_total(self) -> int:
        """Total number of state locations."""
        if self.mode == "grid2d":
            return self.nx * self.ny
        else:
            raise NotImplementedError(f"n_total for mode={self.mode}")


@dataclass
class PriorConfig:
    """GMRF prior hyperparameters."""
    nu: float
    kappa: float
    sigma2: float
    alpha: int
    beta: float
    mu_prior_mean: float
    mu_prior_std: float

    @property
    def correlation_length(self) -> float:
        """Approximate correlation length: sqrt(8*nu)/kappa."""
        return np.sqrt(8 * self.nu) / self.kappa


@dataclass
class SensorType:
    """Sensor type specification."""
    name: str
    noise_std: float
    cost_gbp: float
    footprint: str  # "point" | "avg3x3" | "avg5x5"


@dataclass
class SensorsConfig:
    """Sensor pool configuration."""
    types: List[SensorType]
    pool_strategy: str
    pool_fraction: float
    type_mix: List[float]


@dataclass
class DecisionConfig:
    """Decision model parameters."""
    tau_iri: float
    L_TP_gbp: float
    L_FP_gbp: float
    L_FN_gbp: float
    L_TN_gbp: float

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
    greedy_aopt: Dict[str, Any]


@dataclass
class EVIConfig:
    """Expected Value of Information computation."""
    compute_for: List[str]
    method: str
    monte_carlo_samples: int
    unscented_alpha: float
    unscented_beta: float
    unscented_kappa: float


@dataclass
class CVConfig:
    """Cross-validation settings."""
    scheme: str
    k_folds: int
    buffer_width_multiplier: float
    block_strategy: str
    ensure_connected: bool


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
    expert_plots: Dict[str, Any] = None  # 🔥 新增字段（可选）

    def __post_init__(self):
        """处理可选的expert_plots字段"""
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


class Config:
    """Master configuration container."""

    def __init__(self, config_path: str = "config.yaml"):
        # Smart path resolution: search up from current file
        self.config_path = self._find_config(config_path)
        # ✅ 修复：添加 encoding='utf-8'
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw = yaml.safe_load(f)

        # Parse nested configurations
        self.experiment = ExperimentConfig(**self._raw['experiment'])
        self.experiment.output_dir = Path(self.experiment.output_dir)

        self.numerics = NumericsConfig(**self._raw['numerics'])
        self.geometry = GeometryConfig(**self._raw['geometry'])
        self.prior = PriorConfig(**self._raw['prior'])

        # Parse sensor types
        sensor_types = [SensorType(**st) for st in self._raw['sensors']['types']]
        self.sensors = SensorsConfig(
            types=sensor_types,
            pool_strategy=self._raw['sensors']['pool_strategy'],
            pool_fraction=self._raw['sensors']['pool_fraction'],
            type_mix=self._raw['sensors']['type_mix']
        )

        self.decision = DecisionConfig(**self._raw['decision'])
        self.selection = SelectionConfig(**self._raw['selection'])
        self.evi = EVIConfig(**self._raw['evi'])
        self.cv = CVConfig(**self._raw['cv'])
        self.uq = UQConfig(**self._raw['uq'])
        self.diagnostics = DiagnosticsConfig(**self._raw['diagnostics'])
        self.plots = PlotsConfig(**self._raw['plots'])
        self.acceptance = AcceptanceConfig(**self._raw['acceptance'])

        self.validate()

    def _find_config(self, config_name: str) -> Path:
        """Search for config file starting from script location up to project root."""
        # Try current directory first
        if Path(config_name).exists():
            return Path(config_name)

        # Try relative to this file's location
        current_file = Path(__file__).resolve()

        # Search upwards from src/ directory
        search_paths = [
            current_file.parent / config_name,  # src/config.yaml
            current_file.parent.parent / config_name,  # project_root/config.yaml
        ]

        for path in search_paths:
            if path.exists():
                print(f"Found config at: {path}")
                return path

        # If not found, raise helpful error
        raise FileNotFoundError(
            f"Could not find '{config_name}'. Searched:\n" +
            "\n".join(f"  - {p}" for p in search_paths) +
            "\n\nPlease ensure config.yaml exists in project root."
        )

    def validate(self):
        """Validate configuration consistency."""
        # Check units are SI + GBP (implicit in naming conventions)
        assert all('_gbp' in k for k in vars(self.decision) if k.startswith('L_'))

        # Check probability threshold is valid
        p_T = self.decision.prob_threshold
        assert 0 < p_T < 1, f"Invalid prob_threshold={p_T}"

        # Check budget feasibility
        max_budget = max(self.selection.budgets)
        pool_size = int(self.geometry.n_total * self.sensors.pool_fraction)
        assert max_budget <= pool_size, f"Budget {max_budget} exceeds pool {pool_size}"

        # Check sensor type mix sums to 1
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
        # ✅ 修复：添加 encoding='utf-8'
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._raw, f, default_flow_style=False)
        print(f"  Config saved to {output_path}")


def load_config(path: str = "config.yaml") -> Config:
    """Load and validate configuration."""
    return Config(path)


if __name__ == "__main__":
    # Test configuration loading
    cfg = load_config()
    print(f"\nDecision threshold p_T = {cfg.decision.prob_threshold:.3f}")
    print(f"Buffer width = {cfg.cv.buffer_width_multiplier * cfg.prior.correlation_length:.1f} m")