#!/usr/bin/env python3
"""
ROI诊断脚本 - 深度分析算法选择策略差异

功能：
1. 对比不同算法的传感器选择模式
2. 分解ROI计算的各个组成部分
3. 分析成本-收益权衡
4. 识别导致负ROI的具体原因
5. 给出针对性的调参建议

用法：
    python roi_diagnostic.py --config baseline_config.yaml --budget 5
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import yaml
import argparse

# 假设项目模块可导入
try:
    from geometry import build_grid2d_geometry
    from spatial_field import build_prior, sample_field
    from sensors import build_sensor_pool
    from selection import greedy_mi, greedy_aopt, greedy_evi_myopic_fast, maxmin_k_center, SelectionResult
    from decision import compute_expected_loss, compute_bayes_decisions
    from inference import SparseFactor, compute_posterior
    from config import Config
except ImportError:
    print("警告：无法导入项目模块，请确保脚本在项目目录运行")


@dataclass
class AlgorithmDiagnostics:
    """单个算法的诊断结果"""
    method_name: str
    selected_ids: List[int]
    total_cost: float
    sensor_types: List[str]
    sensor_costs: List[float]
    sensor_noises: List[float]

    # 空间分布
    selected_coords: np.ndarray
    spatial_coverage: float  # 空间覆盖率

    # 决策质量
    n_TP: int
    n_FP: int
    n_TN: int
    n_FN: int

    # 成本分解
    cost_TP: float
    cost_FP: float
    cost_TN: float
    cost_FN: float
    total_loss: float

    # ROI组成
    baseline_loss: float
    savings: float
    roi: float

    # 信息指标
    posterior_variance_mean: float
    posterior_variance_std: float
    mi_total: float  # 总互信息


class ROIDiagnostic:
    """ROI诊断工具"""

    def __init__(self, config_path: str):
        """初始化诊断工具"""
        self.config = self._load_config(config_path)
        self.output_dir = Path("diagnostics_output")
        self.output_dir.mkdir(exist_ok=True)

    def _load_config(self, config_path: str):
        """加载配置文件"""
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        return Config(cfg_dict)

    def run_diagnostic(self, budget: int, methods: List[str] = None):
        """运行完整诊断"""
        print("=" * 80)
        print(f"🔍 ROI诊断 - 预算 k={budget}")
        print("=" * 80)

        # 1. 设置实验
        print("\n[1/6] 设置实验环境...")
        geom, Q_pr, mu_pr, x_true, sensors = self._setup_experiment()

        # 2. 运行所有算法
        print("\n[2/6] 运行算法并收集选择...")
        if methods is None:
            methods = ['greedy_mi', 'greedy_aopt', 'greedy_evi', 'maxmin']

        results = {}
        for method in methods:
            print(f"  运行 {method}...")
            try:
                results[method] = self._run_method(
                    method, sensors, budget, Q_pr, mu_pr, x_true, geom
                )
            except Exception as e:
                print(f"    ⚠️ {method} 失败: {e}")

        # 3. 分析选择模式
        print("\n[3/6] 分析选择模式...")
        diagnostics = {}
        for method, result in results.items():
            diagnostics[method] = self._analyze_selection(
                result, sensors, geom, Q_pr, mu_pr, x_true
            )

        # 4. 生成报告
        print("\n[4/6] 生成诊断报告...")
        self._print_summary_report(diagnostics, budget)

        # 5. 可视化
        print("\n[5/6] 生成可视化...")
        self._create_visualizations(diagnostics, geom, x_true, budget)

        # 6. 调参建议
        print("\n[6/6] 生成调参建议...")
        self._generate_tuning_advice(diagnostics, sensors)

        print(f"\n✅ 诊断完成！结果保存在: {self.output_dir}")

        return diagnostics

    def _setup_experiment(self):
        """设置单次实验"""
        # 构建几何
        geom = build_grid2d_geometry(
            self.config.geometry.nx,
            self.config.geometry.ny,
            self.config.geometry.h
        )

        # 构建先验
        Q_pr, mu_pr = build_prior(geom, self.config.prior)

        # 采样真实场
        rng = self.config.get_rng()
        x_true = sample_field(Q_pr, mu_pr, rng)

        # 构建传感器池
        sensors = build_sensor_pool(geom, self.config.sensors, rng)

        print(f"  ✓ 网格: {geom.n} 点")
        print(f"  ✓ 传感器候选: {len(sensors)} 个")
        print(f"  ✓ 传感器类型: {len(set(s.sensor_type for s in sensors))} 种")

        return geom, Q_pr, mu_pr, x_true, sensors

    def _run_method(self, method_name: str, sensors, k: int,
                    Q_pr, mu_pr, x_true, geom) -> SelectionResult:
        """运行单个算法"""
        costs = np.array([s.cost for s in sensors])

        if method_name == 'greedy_mi':
            return greedy_mi(
                sensors=sensors,
                k=k,
                Q_pr=Q_pr,
                costs=costs,
                lazy=True,
                batch_size=64,
                use_cost=True,
                keep_fraction=self.config.selection.greedy_mi.get('keep_fraction', 0.20)
            )

        elif method_name == 'greedy_aopt':
            return greedy_aopt(
                sensors=sensors,
                k=k,
                Q_pr=Q_pr,
                costs=costs,
                n_probes=self.config.selection.greedy_aopt.get('n_probes', 8),
                use_cost=True
            )

        elif method_name == 'greedy_evi':
            test_idx = np.arange(min(300, geom.n))
            return greedy_evi_myopic_fast(
                sensors=sensors,
                k=k,
                Q_pr=Q_pr,
                mu_pr=mu_pr,
                decision_config=self.config.decision,
                test_idx=test_idx,
                costs=costs,
                n_y_samples=self.config.selection.greedy_evi.get('n_y_samples', 16),
                use_cost=True,
                mi_prescreen=True,
                keep_fraction=None,
                rng=self.config.get_rng(),
                verbose=False
            )

        elif method_name == 'maxmin':
            return maxmin_k_center(
                sensors=sensors,
                k=k,
                coords=geom.coords,
                costs=costs,
                use_cost=True
            )

        else:
            raise ValueError(f"Unknown method: {method_name}")

    def _analyze_selection(self, result: SelectionResult, sensors, geom,
                           Q_pr, mu_pr, x_true) -> AlgorithmDiagnostics:
        """深度分析单个算法的选择"""
        selected_ids = result.selected_ids
        selected_sensors = [sensors[i] for i in selected_ids]

        # 基本信息
        sensor_types = [s.sensor_type for s in selected_sensors]
        sensor_costs = [s.cost for s in selected_sensors]
        sensor_noises = [s.noise_std for s in selected_sensors]
        total_cost = sum(sensor_costs)

        # 空间分布
        selected_coords = np.array([geom.coords[s.location_idx] for s in selected_sensors])
        spatial_coverage = self._compute_spatial_coverage(selected_coords, geom)

        # 模拟观测并计算后验
        y_obs, H_obs, R_diag_obs = self._simulate_observations(
            selected_sensors, x_true
        )

        mu_post, factor_post = compute_posterior(
            Q_pr, mu_pr, H_obs, R_diag_obs, y_obs
        )

        # 计算后验方差
        var_post = self._compute_posterior_variances(factor_post, geom.n)

        # 计算决策质量
        tau_iri = self._get_tau_iri(x_true)
        decisions_prior = (mu_pr >= tau_iri).astype(int)
        decisions_post = (mu_post >= tau_iri).astype(int)
        truth = (x_true >= tau_iri).astype(int)

        # 混淆矩阵
        n_TP = np.sum((decisions_post == 1) & (truth == 1))
        n_FP = np.sum((decisions_post == 1) & (truth == 0))
        n_TN = np.sum((decisions_post == 0) & (truth == 0))
        n_FN = np.sum((decisions_post == 0) & (truth == 1))

        # 成本分解
        L_TP = self.config.decision.L_TP_gbp
        L_FP = self.config.decision.L_FP_gbp
        L_TN = self.config.decision.L_TN_gbp
        L_FN = self.config.decision.L_FN_gbp

        cost_TP = n_TP * L_TP
        cost_FP = n_FP * L_FP
        cost_TN = n_TN * L_TN
        cost_FN = n_FN * L_FN
        total_loss = cost_TP + cost_FP + cost_TN + cost_FN

        # 基线损失（先验决策）
        n_TP_prior = np.sum((decisions_prior == 1) & (truth == 1))
        n_FP_prior = np.sum((decisions_prior == 1) & (truth == 0))
        n_TN_prior = np.sum((decisions_prior == 0) & (truth == 0))
        n_FN_prior = np.sum((decisions_prior == 0) & (truth == 1))
        baseline_loss = (n_TP_prior * L_TP + n_FP_prior * L_FP +
                         n_TN_prior * L_TN + n_FN_prior * L_FN)

        # ROI计算
        savings = baseline_loss - total_loss - total_cost
        roi = savings / total_cost if total_cost > 0 else 0.0

        # 互信息（近似）
        mi_total = 0.5 * np.sum(np.log(np.maximum(var_post, 1e-10)))

        return AlgorithmDiagnostics(
            method_name=result.method_name,
            selected_ids=selected_ids,
            total_cost=total_cost,
            sensor_types=sensor_types,
            sensor_costs=sensor_costs,
            sensor_noises=sensor_noises,
            selected_coords=selected_coords,
            spatial_coverage=spatial_coverage,
            n_TP=n_TP, n_FP=n_FP, n_TN=n_TN, n_FN=n_FN,
            cost_TP=cost_TP, cost_FP=cost_FP, cost_TN=cost_TN, cost_FN=cost_FN,
            total_loss=total_loss,
            baseline_loss=baseline_loss,
            savings=savings,
            roi=roi,
            posterior_variance_mean=np.mean(var_post),
            posterior_variance_std=np.std(var_post),
            mi_total=mi_total
        )

    def _get_tau_iri(self, x_true: np.ndarray) -> float:
        """计算决策阈值"""
        tau_quantile = self.config.decision.tau_quantile
        return np.quantile(x_true, tau_quantile)

    def _simulate_observations(self, sensors, x_true):
        """模拟传感器观测"""
        m = len(sensors)
        n = len(x_true)

        y_obs = np.zeros(m)
        R_diag = np.zeros(m)
        rows, cols, data = [], [], []

        for i, sensor in enumerate(sensors):
            # 获取观测值
            y_obs[i] = sensor.observe(x_true)
            R_diag[i] = sensor.noise_std ** 2

            # 构建H矩阵
            for j in sensor.footprint_indices:
                rows.append(i)
                cols.append(j)
                data.append(sensor.footprint_weights[sensor.footprint_indices.index(j)])

        H = sp.csr_matrix((data, (rows, cols)), shape=(m, n))

        return y_obs, H, R_diag

    def _compute_posterior_variances(self, factor: SparseFactor, n: int) -> np.ndarray:
        """计算后验方差（对角元）"""
        # 快速采样方法
        sample_size = min(n, 100)
        sample_idx = np.random.choice(n, size=sample_size, replace=False)

        var_sample = np.zeros(sample_size)
        for i, idx in enumerate(sample_idx):
            e_i = np.zeros(n)
            e_i[idx] = 1.0
            z = factor.solve(e_i)
            var_sample[i] = z[idx]

        # 插值到全域
        var_full = np.full(n, np.mean(var_sample))
        var_full[sample_idx] = var_sample

        return var_full

    def _compute_spatial_coverage(self, coords: np.ndarray, geom) -> float:
        """计算空间覆盖率"""
        if len(coords) == 0:
            return 0.0

        # 简化：计算最小包围圆占总面积的比例
        center = coords.mean(axis=0)
        max_dist = np.max(np.linalg.norm(coords - center, axis=1))

        total_area = (geom.coords[:, 0].max() - geom.coords[:, 0].min()) * \
                     (geom.coords[:, 1].max() - geom.coords[:, 1].min())
        coverage_area = np.pi * max_dist ** 2

        return min(coverage_area / total_area, 1.0)

    def _print_summary_report(self, diagnostics: Dict[str, AlgorithmDiagnostics], budget: int):
        """打印摘要报告"""
        print("\n" + "=" * 80)
        print(f"📊 诊断报告摘要 (k={budget})")
        print("=" * 80)

        # 创建对比表
        data = []
        for method, diag in diagnostics.items():
            data.append({
                '算法': method,
                '总成本(£)': f"{diag.total_cost:.0f}",
                'ROI': f"{diag.roi:.3f}",
                '节省(£)': f"{diag.savings:.0f}",
                '总损失(£)': f"{diag.total_loss:.0f}",
                'TP': diag.n_TP,
                'FP': diag.n_FP,
                'FN': diag.n_FN,
                '覆盖率': f"{diag.spatial_coverage:.2f}"
            })

        df = pd.DataFrame(data)
        print("\n" + df.to_string(index=False))

        # 详细分析每个算法
        print("\n" + "-" * 80)
        print("📋 详细分析")
        print("-" * 80)

        for method, diag in diagnostics.items():
            print(f"\n【{method}】")
            print(f"  传感器选择:")
            print(f"    - 类型分布: {pd.Series(diag.sensor_types).value_counts().to_dict()}")
            print(f"    - 平均成本: £{np.mean(diag.sensor_costs):.1f}")
            print(f"    - 平均噪声: {np.mean(diag.sensor_noises):.3f}")

            print(f"  决策质量:")
            total_decisions = diag.n_TP + diag.n_FP + diag.n_TN + diag.n_FN
            print(f"    - 准确率: {(diag.n_TP + diag.n_TN) / total_decisions:.3f}")
            print(f"    - 精确率: {diag.n_TP / max(diag.n_TP + diag.n_FP, 1):.3f}")
            print(f"    - 召回率: {diag.n_TP / max(diag.n_TP + diag.n_FN, 1):.3f}")

            print(f"  成本分解:")
            print(f"    - 基线损失: £{diag.baseline_loss:.0f}")
            print(f"    - 后验损失: £{diag.total_loss:.0f}")
            print(f"    - 传感器成本: £{diag.total_cost:.0f}")
            print(f"    - 净节省: £{diag.savings:.0f}")
            print(f"    - ROI: {diag.roi:.3f}")

            # 🔥 关键：识别ROI为负的原因
            if diag.roi < 0:
                print(f"  ⚠️ ROI为负的原因:")
                loss_reduction = diag.baseline_loss - diag.total_loss
                if loss_reduction < diag.total_cost:
                    print(f"    - 损失减少(£{loss_reduction:.0f}) < 传感器成本(£{diag.total_cost:.0f})")
                    print(f"    - 差额: £{diag.total_cost - loss_reduction:.0f}")
                if diag.n_FP > 0:
                    print(f"    - 误报成本过高: {diag.n_FP} × £{self.config.decision.L_FP_gbp} = £{diag.cost_FP:.0f}")
                if diag.n_FN > 0:
                    print(f"    - 漏报成本过高: {diag.n_FN} × £{self.config.decision.L_FN_gbp} = £{diag.cost_FN:.0f}")

        # 保存报告
        report_path = self.output_dir / f"summary_report_k{budget}.txt"
        with open(report_path, 'w') as f:
            f.write(df.to_string(index=False))

        print(f"\n报告保存至: {report_path}")

    def _create_visualizations(self, diagnostics: Dict[str, AlgorithmDiagnostics],
                               geom, x_true, budget: int):
        """生成可视化"""
        n_methods = len(diagnostics)

        # 图1: ROI分解对比
        fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig1.suptitle(f'ROI诊断分析 (k={budget})', fontsize=16, fontweight='bold')

        methods = list(diagnostics.keys())

        # 1.1 ROI对比
        ax = axes[0, 0]
        rois = [diagnostics[m].roi for m in methods]
        colors = ['green' if r > 0 else 'red' for r in rois]
        ax.barh(methods, rois, color=colors, alpha=0.7)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('ROI', fontweight='bold')
        ax.set_title('ROI对比', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # 1.2 成本分解
        ax = axes[0, 1]
        costs_data = {
            '传感器成本': [diagnostics[m].total_cost for m in methods],
            '决策损失': [diagnostics[m].total_loss for m in methods],
            '基线损失': [diagnostics[m].baseline_loss for m in methods]
        }
        x_pos = np.arange(len(methods))
        width = 0.25
        for i, (label, values) in enumerate(costs_data.items()):
            ax.bar(x_pos + i * width, values, width, label=label, alpha=0.7)
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('成本 (£)', fontweight='bold')
        ax.set_title('成本结构对比', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 1.3 决策质量
        ax = axes[1, 0]
        decision_metrics = ['准确率', '精确率', '召回率']
        for method in methods:
            diag = diagnostics[method]
            total = diag.n_TP + diag.n_FP + diag.n_TN + diag.n_FN
            accuracy = (diag.n_TP + diag.n_TN) / total
            precision = diag.n_TP / max(diag.n_TP + diag.n_FP, 1)
            recall = diag.n_TP / max(diag.n_TP + diag.n_FN, 1)
            ax.plot(decision_metrics, [accuracy, precision, recall],
                    marker='o', label=method, linewidth=2)
        ax.set_ylim([0, 1.05])
        ax.set_ylabel('得分', fontweight='bold')
        ax.set_title('决策质量对比', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # 1.4 传感器类型分布
        ax = axes[1, 1]
        type_counts = {}
        for method in methods:
            diag = diagnostics[method]
            type_counts[method] = pd.Series(diag.sensor_types).value_counts().to_dict()

        # 堆叠条形图
        all_types = set()
        for counts in type_counts.values():
            all_types.update(counts.keys())
        all_types = sorted(all_types)

        bottom = np.zeros(len(methods))
        for sensor_type in all_types:
            values = [type_counts[m].get(sensor_type, 0) for m in methods]
            ax.bar(methods, values, bottom=bottom, label=sensor_type, alpha=0.7)
            bottom += values

        ax.set_ylabel('传感器数量', fontweight='bold')
        ax.set_title('传感器类型分布', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        fig1_path = self.output_dir / f'roi_breakdown_k{budget}.png'
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ ROI分解图: {fig1_path}")
        plt.close()

        # 图2: 空间分布
        fig2, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(16, 10))
        axes = axes.flatten() if n_methods > 1 else [axes]
        fig2.suptitle(f'传感器空间分布 (k={budget})', fontsize=16, fontweight='bold')

        for idx, method in enumerate(methods):
            ax = axes[idx]
            diag = diagnostics[method]

            # 绘制真实场
            x_grid = x_true.reshape(geom.coords[:, 0].max() // 5 + 1,
                                    geom.coords[:, 1].max() // 5 + 1)
            im = ax.contourf(x_grid, levels=15, cmap='RdYlGn_r', alpha=0.6)

            # 绘制传感器
            coords = diag.selected_coords
            costs = diag.sensor_costs
            scatter = ax.scatter(coords[:, 0], coords[:, 1],
                                 s=np.array(costs) * 2, c=costs,
                                 cmap='viridis', edgecolors='black',
                                 linewidths=2, alpha=0.9)

            ax.set_title(f'{method}\nROI={diag.roi:.3f}, Cost=£{diag.total_cost:.0f}',
                         fontweight='bold')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            plt.colorbar(scatter, ax=ax, label='传感器成本(£)')

        # 隐藏多余子图
        for idx in range(len(methods), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        fig2_path = self.output_dir / f'spatial_distribution_k{budget}.png'
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 空间分布图: {fig2_path}")
        plt.close()

        # 图3: 混淆矩阵
        fig3, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))
        axes = [axes] if n_methods == 1 else axes
        fig3.suptitle(f'决策混淆矩阵 (k={budget})', fontsize=16, fontweight='bold')

        for idx, method in enumerate(methods):
            diag = diagnostics[method]
            confusion = np.array([
                [diag.n_TP, diag.n_FP],
                [diag.n_FN, diag.n_TN]
            ])

            ax = axes[idx]
            sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                        ax=ax, cbar=True, square=True,
                        xticklabels=['预测维护', '预测不维护'],
                        yticklabels=['实际维护', '实际不维护'])
            ax.set_title(f'{method}\nROI={diag.roi:.3f}', fontweight='bold')

        plt.tight_layout()
        fig3_path = self.output_dir / f'confusion_matrices_k{budget}.png'
        plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 混淆矩阵: {fig3_path}")
        plt.close()

    def _generate_tuning_advice(self, diagnostics: Dict[str, AlgorithmDiagnostics],
                                sensors):
        """生成调参建议"""
        print("\n" + "=" * 80)
        print("💡 调参建议")
        print("=" * 80)

        # 分析最佳和最差算法
        sorted_methods = sorted(diagnostics.items(), key=lambda x: x[1].roi, reverse=True)
        best_method, best_diag = sorted_methods[0]
        worst_method, worst_diag = sorted_methods[-1]

        print(f"\n🏆 最佳算法: {best_method} (ROI={best_diag.roi:.3f})")
        print(f"💀 最差算法: {worst_method} (ROI={worst_diag.roi:.3f})")
        print(f"📊 性能差距: {best_diag.roi - worst_diag.roi:.3f}")

        # 通用建议
        print("\n【通用调参建议】")

        # 1. 检查决策成本
        avg_roi = np.mean([d.roi for d in diagnostics.values()])
        if avg_roi < 0:
            print("\n⚠️ 所有算法ROI都为负，主要问题可能在决策成本设置：")

            L_FP = self.config.decision.L_FP_gbp
            L_FN = self.config.decision.L_FN_gbp
            ratio = L_FN / L_FP

            print(f"  当前设置: L_FN/L_FP = {ratio:.1f}:1")

            if ratio > 15:
                print(f"  ❌ 不对称性过高！建议降低到10:1")
                print(f"  建议: L_FP={L_FP // 2:.0f}, L_FN={L_FN // 5:.0f}")

            # 检查传感器成本
            avg_sensor_cost = np.mean([s.cost for s in sensors])
            avg_loss_reduction = np.mean([d.baseline_loss - d.total_loss for d in diagnostics.values()])

            if avg_sensor_cost > avg_loss_reduction:
                print(f"\n  ❌ 传感器成本(£{avg_sensor_cost:.0f}) > 损失减少(£{avg_loss_reduction:.0f})")
                print(f"  建议1: 降低所有传感器成本50%")
                print(f"  建议2: 增加pool_fraction到0.6以获得更好的选择")

        # 2. 分析为什么A-opt表现最好
        if best_method == 'greedy_aopt':
            print("\n🔍 A-optimal表现最佳的原因分析：")

            # 比较传感器选择
            aopt_types = pd.Series(best_diag.sensor_types).value_counts()
            print(f"  A-opt偏好: {aopt_types.to_dict()}")

            aopt_avg_cost = np.mean(best_diag.sensor_costs)
            aopt_avg_noise = np.mean(best_diag.sensor_noises)

            print(f"  平均成本: £{aopt_avg_cost:.1f}")
            print(f"  平均噪声: {aopt_avg_noise:.3f}")

            # 与其他算法对比
            for method, diag in diagnostics.items():
                if method != best_method:
                    cost_diff = np.mean(diag.sensor_costs) - aopt_avg_cost
                    noise_diff = np.mean(diag.sensor_noises) - aopt_avg_noise
                    print(f"\n  vs {method}:")
                    print(f"    成本差异: £{cost_diff:+.1f} ({'更便宜' if cost_diff < 0 else '更贵'})")
                    print(f"    噪声差异: {noise_diff:+.3f} ({'更低噪声' if noise_diff < 0 else '更高噪声'})")

                    if cost_diff > 0 and diag.roi < best_diag.roi:
                        print(f"    💡 {method}选择了更贵的传感器但ROI更低")
                        print(f"       建议: 调整{method}的成本权重参数")

        # 3. 针对性建议
        print("\n【算法特定建议】")

        for method, diag in diagnostics.items():
            if diag.roi < 0:
                print(f"\n📉 {method} (ROI={diag.roi:.3f}):")

                # 成本效率分析
                cost_efficiency = diag.savings / diag.total_cost if diag.total_cost > 0 else 0
                print(f"  成本效率: {cost_efficiency:.3f}")

                if method == 'greedy_mi':
                    current_keep = self.config.selection.greedy_mi.get('keep_fraction', 0.20)
                    print(f"  当前keep_fraction={current_keep}")
                    if current_keep < 0.4:
                        print(f"  ❌ 预筛选过严！建议增加到0.4-0.5")

                elif method == 'greedy_evi':
                    n_samples = self.config.selection.greedy_evi.get('n_y_samples', 16)
                    print(f"  当前n_y_samples={n_samples}")
                    if diag.n_FP > diag.n_FN * 2:
                        print(f"  ❌ 误报过多！可能需要更多样本")
                        print(f"  建议: n_y_samples={n_samples * 2}")

                elif method == 'greedy_aopt':
                    n_probes = self.config.selection.greedy_aopt.get('n_probes', 8)
                    print(f"  当前n_probes={n_probes}")
                    if n_probes < 16:
                        print(f"  建议: 增加n_probes到16以提高方差估计精度")

        # 保存建议到文件
        advice_path = self.output_dir / "tuning_advice.txt"
        with open(advice_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("自动调参建议\n")
            f.write("=" * 80 + "\n\n")

            f.write("【关键发现】\n")
            f.write(f"- 最佳算法: {best_method} (ROI={best_diag.roi:.3f})\n")
            f.write(f"- 平均ROI: {avg_roi:.3f}\n\n")

            if avg_roi < 0:
                f.write("【紧急修复】\n")
                f.write("所有算法ROI为负，建议立即修改：\n")
                f.write(f"1. 降低L_FN: {self.config.decision.L_FN_gbp} → {self.config.decision.L_FN_gbp // 5}\n")
                f.write(f"2. 降低L_FP: {self.config.decision.L_FP_gbp} → {self.config.decision.L_FP_gbp // 2}\n")
                f.write("3. 降低所有传感器成本50%\n")
                f.write("4. 增加pool_fraction到0.6\n\n")

            f.write("【算法特定建议】\n")
            for method, diag in diagnostics.items():
                if diag.roi < best_diag.roi - 0.1:
                    f.write(f"\n{method}:\n")
                    if method == 'greedy_mi':
                        f.write(f"  - keep_fraction: 提升到0.4-0.5\n")
                    elif method == 'greedy_evi':
                        f.write(f"  - n_y_samples: 提升到32\n")
                    elif method == 'greedy_aopt':
                        f.write(f"  - n_probes: 提升到16\n")

        print(f"\n建议已保存至: {advice_path}")

        # 生成修复后的配置文件
        self._generate_fixed_config(diagnostics, best_diag)

    def _generate_fixed_config(self, diagnostics, best_diag):
        """生成修复后的配置文件"""
        avg_roi = np.mean([d.roi for d in diagnostics.values()])

        if avg_roi < 0:
            print("\n📝 生成修复配置文件...")

            # 读取原配置
            fixed_config = yaml.safe_load(open(self.config._config_path))

            # 应用修复
            L_FN_old = fixed_config['decision']['L_FN_gbp']
            L_FP_old = fixed_config['decision']['L_FP_gbp']

            fixed_config['decision']['L_FN_gbp'] = L_FN_old // 5
            fixed_config['decision']['L_FP_gbp'] = L_FP_old // 2
            fixed_config['decision']['target_ddi'] = 0.20

            fixed_config['sensors']['pool_fraction'] = 0.60

            # 降低传感器成本
            for sensor_type in fixed_config['sensors']['types']:
                sensor_type['cost_gbp'] = sensor_type['cost_gbp'] * 0.5

            # 调整算法参数
            if 'greedy_mi' in fixed_config['selection']:
                fixed_config['selection']['greedy_mi']['keep_fraction'] = 0.40
            if 'greedy_aopt' in fixed_config['selection']:
                fixed_config['selection']['greedy_aopt']['n_probes'] = 16
            if 'greedy_evi' in fixed_config['selection']:
                fixed_config['selection']['greedy_evi']['n_y_samples'] = 24

            # 保存
            fixed_path = self.output_dir / "auto_fixed_config.yaml"
            with open(fixed_path, 'w') as f:
                yaml.dump(fixed_config, f, default_flow_style=False, allow_unicode=True)

            print(f"  ✓ 修复配置: {fixed_path}")
            print(f"\n运行命令测试修复效果:")
            print(f"  python main.py --config {fixed_path} --budgets 5,10,15")


def main():
    parser = argparse.ArgumentParser(description='ROI诊断工具')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--budget', type=int, default=5,
                        help='传感器预算 (默认: 5)')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['greedy_mi', 'greedy_aopt', 'greedy_evi', 'maxmin'],
                        help='要诊断的算法')

    args = parser.parse_args()

    # 运行诊断
    diagnostic = ROIDiagnostic(args.config)
    diagnostic.run_diagnostic(args.budget, args.methods)


if __name__ == '__main__':
    main()