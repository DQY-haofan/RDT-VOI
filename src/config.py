"""
Configuration management for RDT-VoI simulation (Enhanced version)
✅ 修改版：单一基准配置 + 参数扫描支持 + 向后兼容

主要改进：
1. 使用单一 baseline_config.yaml 作为默认配置
2. 支持运行时参数覆盖和扫描
3. 保持向后兼容（load_scenario_config 等函数名不变）
4. 新增 apply_parameter_overrides() 功能
"""

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
        self.parameter_scan_presets = self._raw.get('parameter_scan_presets', {})

        self.validate()

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
    print("🔧 Testing enhanced configuration system...")

    # 测试基准配置加载
    print("\n[1] Loading baseline config...")
    cfg = load_config()
    print(f"  Loaded: {cfg.experiment.name}")
    print(f"  DDI: {cfg.decision.target_ddi}")
    print(f"  L_FN/L_FP ratio: {cfg.decision.L_FN_gbp / cfg.decision.L_FP_gbp:.1f}")

    # 测试参数覆盖
    print("\n[2] Testing parameter overrides...")
    overrides = {
        'target_ddi': 0.35,
        'L_FN_gbp': 150000,
        'grid_size': 25
    }
    cfg_modified = cfg.apply_parameter_overrides(overrides)
    print(f"  Modified DDI: {cfg_modified.decision.target_ddi}")
    print(f"  Modified L_FN: {cfg_modified.decision.L_FN_gbp}")
    print(f"  Modified grid: {cfg_modified.geometry.nx}x{cfg_modified.geometry.ny}")

    # 测试预设应用
    print("\n[3] Testing preset application...")
    cfg_high_stakes = cfg.apply_preset('high_stakes')
    print(f"  High-stakes DDI: {cfg_high_stakes.decision.target_ddi}")
    print(f"  High-stakes L_FN: {cfg_high_stakes.decision.L_FN_gbp}")

    # 测试参数扫描组合生成
    print("\n[4] Testing parameter scan combinations...")
    scan_params = {
        'target_ddi': [0.1, 0.2, 0.3],
        'L_FN_gbp': [30000, 60000]
    }
    combinations = generate_parameter_combinations(scan_params)
    print(f"  Generated {len(combinations)} combinations:")
    for i, combo in enumerate(combinations):
        print(f"    {i+1}: {combo}")

    # 测试向后兼容
    print("\n[5] Testing backward compatibility...")
    cfg_scenario_a = load_scenario_config('A')
    cfg_scenario_b = load_scenario_config('B')
    print(f"  Scenario A DDI: {cfg_scenario_a.decision.target_ddi}")
    print(f"  Scenario B DDI: {cfg_scenario_b.decision.target_ddi}")

    print("\n✅ All tests passed!")