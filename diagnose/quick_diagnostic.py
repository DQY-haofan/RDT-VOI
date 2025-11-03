#!/usr/bin/env python3
"""
轻量级ROI诊断工具 - 快速分析算法选择差异

无需运行完整实验，直接分析配置文件找出问题

用法：
    python quick_diagnostic.py --config baseline_config.yaml
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple


class QuickDiagnostic:
    """快速配置诊断"""

    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.output_dir = Path("quick_diagnostics")
        self.output_dir.mkdir(exist_ok=True)

    def run_all_checks(self):
        """运行所有诊断检查"""
        print("=" * 80)
        print("🔍 快速配置诊断")
        print("=" * 80)

        issues = []

        # 检查1: 决策成本合理性
        print("\n[检查1/5] 决策成本设置...")
        issue1 = self.check_decision_costs()
        if issue1:
            issues.extend(issue1)

        # 检查2: 传感器成本结构
        print("\n[检查2/5] 传感器成本结构...")
        issue2 = self.check_sensor_costs()
        if issue2:
            issues.extend(issue2)

        # 检查3: 候选池密度
        print("\n[检查3/5] 候选池配置...")
        issue3 = self.check_pool_configuration()
        if issue3:
            issues.extend(issue3)

        # 检查4: 算法参数
        print("\n[检查4/5] 算法参数设置...")
        issue4 = self.check_algorithm_parameters()
        if issue4:
            issues.extend(issue4)

        # 检查5: 先验配置
        print("\n[检查5/5] 先验分布配置...")
        issue5 = self.check_prior_configuration()
        if issue5:
            issues.extend(issue5)

        # 生成报告
        self.generate_report(issues)
        self.create_comparison_plots()

        return issues

    def check_decision_costs(self) -> List[str]:
        """检查决策成本设置"""
        issues = []

        decision = self.config['decision']
        L_FP = decision['L_FP_gbp']
        L_FN = decision['L_FN_gbp']
        L_TP = decision['L_TP_gbp']

        ratio = L_FN / L_FP

        print(f"  当前设置:")
        print(f"    L_FP (误报): £{L_FP:,.0f}")
        print(f"    L_FN (漏报): £{L_FN:,.0f}")
        print(f"    L_TP (维护): £{L_TP:,.0f}")
        print(f"    不对称比: {ratio:.1f}:1")

        # 计算维护概率
        p_T = L_FP / (L_FP + L_FN - L_TP) if (L_FP + L_FN - L_TP) > 0 else 0
        print(f"    隐含维护概率: {p_T * 100:.1f}%")

        # 检查问题
        if ratio > 15:
            issues.append(f"❌ 不对称性过高 ({ratio:.1f}:1)，建议降到10:1以下")
            print(f"  ⚠️ 不对称性过高！")
            print(f"     建议: L_FP={L_FP // 2}, L_FN={L_FN // 5}")

        if p_T < 0.08:
            issues.append(f"❌ 维护概率过低 ({p_T * 100:.1f}%)，传感器收益机会太少")
            print(f"  ⚠️ 维护概率过低！目标应在8-12%")

        if L_FN > 100000:
            issues.append(f"❌ 漏报成本过高 (£{L_FN:,})，可能主导ROI计算")
            print(f"  ⚠️ 漏报成本绝对值过高")

        # 检查target_ddi
        target_ddi = decision.get('target_ddi', 0.25)
        print(f"    目标DDI: {target_ddi * 100:.0f}%")
        if target_ddi <= 0.25:
            issues.append(f"⚠️ target_ddi={target_ddi}可能触发代码下限，建议≥0.28")

        if not issues:
            print(f"  ✓ 决策成本配置合理")

        return issues

    def check_sensor_costs(self) -> List[str]:
        """检查传感器成本结构"""
        issues = []

        sensors = self.config['sensors']['types']

        costs = [s['cost_gbp'] for s in sensors]
        noises = [s['noise_std'] for s in sensors]
        names = [s['name'] for s in sensors]

        print(f"  传感器类型数: {len(sensors)}")
        print(f"  成本范围: £{min(costs)} - £{max(costs)}")
        print(f"  成本比: {max(costs) / min(costs):.1f}:1")

        # 计算成本-性能比
        snrs = [1 / (n ** 2) for n in noises]
        efficiencies = [snr / cost for snr, cost in zip(snrs, costs)]

        print(f"\n  详细分析:")
        for name, cost, noise, eff in zip(names, costs, noises, efficiencies):
            print(f"    {name:20s}: £{cost:>5.0f}, 噪声={noise:.3f}, 效率={eff:.4f}")

        # 检查问题
        cost_ratio = max(costs) / min(costs)
        if cost_ratio > 30:
            issues.append(f"❌ 成本范围过大 ({cost_ratio:.0f}:1)，边际效益严重递减")
            print(f"\n  ⚠️ 成本梯度过陡！")
            print(f"     建议: 压缩到10:1以内")

        if max(costs) > 800:
            issues.append(f"❌ 最高成本传感器过贵 (£{max(costs)})，难以回本")
            print(f"  ⚠️ 高端传感器过贵")

        # 检查性价比分布
        eff_range = max(efficiencies) / min(efficiencies)
        if eff_range > 20:
            issues.append(f"⚠️ 效率差异过大，某些传感器可能永远不被选择")
            print(f"  ⚠️ 效率差异: {eff_range:.1f}:1")

        if not issues:
            print(f"  ✓ 传感器成本配置合理")

        return issues

    def check_pool_configuration(self) -> List[str]:
        """检查候选池配置"""
        issues = []

        geometry = self.config['geometry']
        sensors_cfg = self.config['sensors']

        nx, ny = geometry['nx'], geometry['ny']
        total_points = nx * ny

        pool_fraction = sensors_cfg.get('pool_fraction', 1.0)
        n_candidates = int(total_points * pool_fraction)

        print(f"  网格大小: {nx} × {ny} = {total_points} 点")
        print(f"  候选池比例: {pool_fraction * 100:.0f}%")
        print(f"  候选点数: {n_candidates}")

        # 检查预算
        budgets = self.config['selection'].get('budgets', [5, 10, 15])
        max_budget = max(budgets)
        selection_pressure = max_budget / n_candidates

        print(f"  最大预算: {max_budget}")
        print(f"  选择压力: {selection_pressure * 100:.1f}%")

        # 检查问题
        if pool_fraction < 0.4:
            issues.append(f"❌ 候选池过小 ({pool_fraction * 100:.0f}%)，限制算法优化空间")
            print(f"  ⚠️ 候选池过稀疏！")
            print(f"     建议: pool_fraction ≥ 0.5")

        if selection_pressure > 0.3:
            issues.append(f"⚠️ 选择压力过高 ({selection_pressure * 100:.0f}%)，接近饱和")

        if n_candidates < 100:
            issues.append(f"❌ 候选点太少 ({n_candidates})，算法差异难以体现")

        if not issues:
            print(f"  ✓ 候选池配置合理")

        return issues

    def check_algorithm_parameters(self) -> List[str]:
        """检查算法参数"""
        issues = []

        selection = self.config.get('selection', {})

        # 检查Greedy-MI
        if 'greedy_mi' in selection:
            mi_cfg = selection['greedy_mi']
            keep_frac = mi_cfg.get('keep_fraction', 0.20)

            print(f"  Greedy-MI:")
            print(f"    keep_fraction: {keep_frac}")

            if keep_frac < 0.3:
                issues.append(f"⚠️ Greedy-MI keep_fraction过小 ({keep_frac})，可能错过最优解")
                print(f"    ⚠️ 预筛选过严，建议≥0.4")

        # 检查Greedy-Aopt
        if 'greedy_aopt' in selection:
            aopt_cfg = selection['greedy_aopt']
            n_probes = aopt_cfg.get('n_probes', 8)

            print(f"  Greedy-Aopt:")
            print(f"    n_probes: {n_probes}")

            if n_probes < 16:
                issues.append(f"⚠️ Greedy-Aopt n_probes较少 ({n_probes})，可能低估方差")
                print(f"    ⚠️ 探针数偏少，建议≥16")

        # 检查Greedy-EVI
        if 'greedy_evi' in selection:
            evi_cfg = selection['greedy_evi']
            n_samples = evi_cfg.get('n_y_samples', 16)
            keep_frac_evi = evi_cfg.get('keep_fraction')

            print(f"  Greedy-EVI:")
            print(f"    n_y_samples: {n_samples}")
            print(f"    keep_fraction: {keep_frac_evi}")

            if n_samples < 16:
                issues.append(f"⚠️ Greedy-EVI样本数较少 ({n_samples})，可能不稳定")

            if keep_frac_evi is not None and keep_frac_evi < 0.3:
                issues.append(f"⚠️ Greedy-EVI预筛选过严")

        if not issues:
            print(f"  ✓ 算法参数配置合理")

        return issues

    def check_prior_configuration(self) -> List[str]:
        """检查先验配置"""
        issues = []

        prior = self.config.get('prior', {})

        alpha = prior.get('alpha', 1e-3)
        beta = prior.get('beta', 1e-3)
        beta_base = prior.get('beta_base', 1e-3)
        beta_hot = prior.get('beta_hot', 1e-4)

        print(f"  Alpha: {alpha:.2e}")
        print(f"  Beta: {beta:.2e}")
        print(f"  Beta_base: {beta_base:.2e}")
        print(f"  Beta_hot: {beta_hot:.2e}")

        # 检查冲突
        if abs(beta - beta_base) > 1e-6 and beta_base > beta * 10:
            issues.append(f"❌ beta_base与beta数值冲突 ({beta_base} vs {beta})")
            print(f"  ⚠️ 参数冲突！beta_base应与beta一致")

        # 检查异质性
        heterogeneity = beta_base / beta_hot if beta_hot > 0 else 1
        print(f"  方差异质性: {heterogeneity:.1f}:1")

        if heterogeneity < 5:
            issues.append(f"⚠️ 空间异质性不足 ({heterogeneity:.0f}:1)，建议≥10:1")
            print(f"  ⚠️ 空间差异太小")

        if not issues:
            print(f"  ✓ 先验配置合理")

        return issues

    def generate_report(self, issues: List[str]):
        """生成诊断报告"""
        print("\n" + "=" * 80)
        print("📊 诊断总结")
        print("=" * 80)

        if not issues:
            print("\n✅ 未发现严重问题，配置基本合理")
            return

        print(f"\n发现 {len(issues)} 个问题:\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")

        # 生成修复建议
        print("\n" + "=" * 80)
        print("💡 修复建议（按优先级排序）")
        print("=" * 80)

        priority_fixes = []

        # 优先级1: 决策成本
        if any('不对称性过高' in i or '维护概率' in i for i in issues):
            priority_fixes.append({
                'priority': 1,
                'category': '决策成本',
                'action': [
                    f"L_FN_gbp: {self.config['decision']['L_FN_gbp']} → {self.config['decision']['L_FN_gbp'] // 5}",
                    f"L_FP_gbp: {self.config['decision']['L_FP_gbp']} → {self.config['decision']['L_FP_gbp'] // 2}",
                    "target_ddi: 0.30"
                ],
                'expected_impact': '预计ROI提升 +0.4~0.6'
            })

        # 优先级2: 传感器成本
        if any('成本范围过大' in i or '过贵' in i for i in issues):
            priority_fixes.append({
                'priority': 2,
                'category': '传感器成本',
                'action': [
                    "所有传感器成本 × 0.5",
                    "或：重新设计为线性梯度 (£25, £80, £200)"
                ],
                'expected_impact': '预计ROI提升 +0.3~0.5'
            })

        # 优先级3: 候选池
        if any('候选池' in i for i in issues):
            priority_fixes.append({
                'priority': 3,
                'category': '候选池密度',
                'action': [
                    f"pool_fraction: {self.config['sensors']['pool_fraction']} → 0.60"
                ],
                'expected_impact': '预计ROI提升 +0.2~0.3'
            })

        # 优先级4: 算法参数
        if any('keep_fraction' in i or 'n_probes' in i for i in issues):
            priority_fixes.append({
                'priority': 4,
                'category': '算法参数',
                'action': [
                    "greedy_mi.keep_fraction: 0.40",
                    "greedy_aopt.n_probes: 16",
                    "greedy_evi.n_y_samples: 24"
                ],
                'expected_impact': '预计算法差异提升 +5~10%'
            })

        for fix in priority_fixes:
            print(f"\n🔥 优先级 {fix['priority']}: {fix['category']}")
            for action in fix['action']:
                print(f"   - {action}")
            print(f"   预期效果: {fix['expected_impact']}")

        # 保存报告
        report_path = self.output_dir / "diagnostic_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("配置诊断报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"发现问题数: {len(issues)}\n\n")
            for i, issue in enumerate(issues, 1):
                f.write(f"{i}. {issue}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("修复建议\n")
            f.write("=" * 80 + "\n\n")

            for fix in priority_fixes:
                f.write(f"优先级 {fix['priority']}: {fix['category']}\n")
                for action in fix['action']:
                    f.write(f"  - {action}\n")
                f.write(f"  {fix['expected_impact']}\n\n")

        print(f"\n报告已保存: {report_path}")

    def create_comparison_plots(self):
        """创建对比可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('配置参数分析', fontsize=16, fontweight='bold')

        # 图1: 决策成本
        ax = axes[0, 0]
        decision = self.config['decision']
        costs = [
            decision['L_FP_gbp'],
            decision['L_FN_gbp'],
            decision['L_TP_gbp']
        ]
        labels = ['误报\n(FP)', '漏报\n(FN)', '维护\n(TP)']
        colors = ['orange', 'red', 'blue']

        ax.bar(labels, costs, color=colors, alpha=0.7)
        ax.set_ylabel('成本 (£)', fontweight='bold')
        ax.set_title('决策损失函数', fontweight='bold')
        ax.set_yscale('log')
        for i, (label, cost) in enumerate(zip(labels, costs)):
            ax.text(i, cost, f'£{cost:,.0f}', ha='center', va='bottom', fontweight='bold')

        # 图2: 传感器成本分布
        ax = axes[0, 1]
        sensors = self.config['sensors']['types']
        names = [s['name'] for s in sensors]
        costs = [s['cost_gbp'] for s in sensors]
        noises = [s['noise_std'] for s in sensors]

        scatter = ax.scatter(noises, costs, s=200, alpha=0.7, c=range(len(names)), cmap='viridis')
        for name, noise, cost in zip(names, noises, costs):
            ax.annotate(name, (noise, cost), fontsize=8, ha='right')

        ax.set_xlabel('噪声标准差', fontweight='bold')
        ax.set_ylabel('成本 (£)', fontweight='bold')
        ax.set_title('传感器成本 vs 性能', fontweight='bold')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)

        # 图3: 候选池配置
        ax = axes[1, 0]
        geometry = self.config['geometry']
        total_points = geometry['nx'] * geometry['ny']
        pool_fraction = self.config['sensors'].get('pool_fraction', 1.0)
        n_candidates = int(total_points * pool_fraction)

        data = [n_candidates, total_points - n_candidates]
        labels = [f'候选点\n({n_candidates})', f'非候选\n({total_points - n_candidates})']
        colors = ['green', 'lightgray']

        ax.pie(data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'候选池覆盖 ({pool_fraction * 100:.0f}%)', fontweight='bold')

        # 图4: 算法参数对比
        ax = axes[1, 1]
        selection = self.config.get('selection', {})

        param_data = []
        if 'greedy_mi' in selection:
            param_data.append(('MI\nkeep_frac', selection['greedy_mi'].get('keep_fraction', 0.2)))
        if 'greedy_aopt' in selection:
            param_data.append(('Aopt\nn_probes', selection['greedy_aopt'].get('n_probes', 8) / 20))  # 归一化
        if 'greedy_evi' in selection:
            param_data.append(('EVI\nn_samples', selection['greedy_evi'].get('n_y_samples', 16) / 50))  # 归一化

        if param_data:
            labels, values = zip(*param_data)
            ax.bar(labels, values, alpha=0.7, color=['blue', 'green', 'red'][:len(param_data)])
            ax.set_ylabel('参数值 (归一化)', fontweight='bold')
            ax.set_title('算法参数设置', fontweight='bold')
            ax.set_ylim([0, 1])
            ax.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='建议下限')
            ax.legend()

        plt.tight_layout()
        plot_path = self.output_dir / 'config_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n可视化已保存: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='快速配置诊断工具')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')

    args = parser.parse_args()

    diagnostic = QuickDiagnostic(args.config)
    issues = diagnostic.run_all_checks()

    if issues:
        print("\n" + "=" * 80)
        print("⚠️ 建议使用修复后的配置重新运行实验")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✅ 配置检查通过！")
        print("=" * 80)


if __name__ == '__main__':
    main()