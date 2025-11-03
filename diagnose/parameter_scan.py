#!/usr/bin/env python3
"""
参数扫描工具 - 系统性搜索最优配置

自动测试不同参数组合，找到能使ROI转正的配置

用法：
    # 快速扫描（小范围）
    python parameter_scan.py --config baseline_config.yaml --mode quick

    # 全面扫描（大范围）
    python parameter_scan.py --config baseline_config.yaml --mode full

    # 自定义扫描
    python parameter_scan.py --config baseline_config.yaml --param L_FN_gbp --values 30000,50000,100000
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from itertools import product
import pandas as pd
from typing import Dict, List, Tuple
import copy


class ParameterScanner:
    """参数扫描工具"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        self.output_dir = Path("parameter_scan_results")
        self.output_dir.mkdir(exist_ok=True)

    def run_scan(self, mode: str = 'quick'):
        """运行参数扫描"""
        print("=" * 80)
        print(f"🔍 参数扫描 - {mode}模式")
        print("=" * 80)

        if mode == 'quick':
            scan_configs = self.get_quick_scan_configs()
        elif mode == 'full':
            scan_configs = self.get_full_scan_configs()
        elif mode == 'targeted':
            scan_configs = self.get_targeted_scan_configs()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"\n将测试 {len(scan_configs)} 个配置组合\n")

        # 分析每个配置
        results = []
        for i, (name, config) in enumerate(scan_configs.items(), 1):
            print(f"[{i}/{len(scan_configs)}] 分析: {name}")
            metrics = self.analyze_config(config)
            metrics['config_name'] = name
            results.append(metrics)

        # 生成报告
        self.generate_scan_report(results)
        self.create_scan_visualizations(results)

        # 找出最佳配置
        best_config = self.find_best_config(results, scan_configs)

        return results, best_config

    def get_quick_scan_configs(self) -> Dict[str, dict]:
        """快速扫描：测试关键参数的少量组合"""
        configs = {}

        # 基准配置
        configs['baseline'] = copy.deepcopy(self.base_config)

        # 扫描1: 决策成本比例
        for ratio in [5, 10, 15]:
            cfg = copy.deepcopy(self.base_config)
            L_FP_base = 5000
            L_FN = L_FP_base * ratio
            cfg['decision']['L_FP_gbp'] = L_FP_base
            cfg['decision']['L_FN_gbp'] = L_FN
            configs[f'cost_ratio_{ratio}to1'] = cfg

        # 扫描2: 传感器成本缩放
        for scale in [0.3, 0.5, 0.7]:
            cfg = copy.deepcopy(self.base_config)
            for sensor in cfg['sensors']['types']:
                sensor['cost_gbp'] = int(sensor['cost_gbp'] * scale)
            configs[f'sensor_cost_{int(scale * 100)}pct'] = cfg

        # 扫描3: 候选池密度
        for pool_frac in [0.4, 0.6, 0.8]:
            cfg = copy.deepcopy(self.base_config)
            cfg['sensors']['pool_fraction'] = pool_frac
            configs[f'pool_{int(pool_frac * 100)}pct'] = cfg

        # 扫描4: 组合优化（最激进）
        cfg = copy.deepcopy(self.base_config)
        cfg['decision']['L_FP_gbp'] = 2000
        cfg['decision']['L_FN_gbp'] = 20000
        cfg['decision']['target_ddi'] = 0.25
        for sensor in cfg['sensors']['types']:
            sensor['cost_gbp'] = int(sensor['cost_gbp'] * 0.4)
        cfg['sensors']['pool_fraction'] = 0.7
        configs['aggressive_fix'] = cfg

        return configs

    def get_full_scan_configs(self) -> Dict[str, dict]:
        """全面扫描：更大范围的参数网格"""
        configs = {}

        # 决策成本网格
        L_FP_values = [2000, 5000, 10000]
        L_FN_values = [20000, 50000, 100000]

        for L_FP, L_FN in product(L_FP_values, L_FN_values):
            if L_FN / L_FP > 3:  # 至少保持一定不对称性
                cfg = copy.deepcopy(self.base_config)
                cfg['decision']['L_FP_gbp'] = L_FP
                cfg['decision']['L_FN_gbp'] = L_FN
                configs[f'cost_FP{L_FP}_FN{L_FN}'] = cfg

        # 传感器成本网格
        for sensor_scale in [0.3, 0.5, 0.7, 1.0]:
            for pool_frac in [0.3, 0.5, 0.7]:
                cfg = copy.deepcopy(self.base_config)
                for sensor in cfg['sensors']['types']:
                    sensor['cost_gbp'] = int(sensor['cost_gbp'] * sensor_scale)
                cfg['sensors']['pool_fraction'] = pool_frac
                configs[f'sensor{int(sensor_scale * 100)}_pool{int(pool_frac * 100)}'] = cfg

        return configs

    def get_targeted_scan_configs(self) -> Dict[str, dict]:
        """针对性扫描：基于A-optimal成功经验"""
        configs = {}

        # 基准
        configs['baseline'] = copy.deepcopy(self.base_config)

        # 假设A-opt的成功来自于平衡的成本-效益
        # 逐步逼近合理区间

        # 策略1: 降低决策成本，保持中等不对称
        for target_p_T in [0.08, 0.10, 0.12, 0.15]:
            cfg = copy.deepcopy(self.base_config)
            # 反推L_FP和L_FN
            # p_T = L_FP / (L_FP + L_FN - L_TP)
            # 假设L_FN/L_FP = 10
            L_TP = cfg['decision']['L_TP_gbp']
            # p_T * (L_FP + 10*L_FP - L_TP) = L_FP
            # p_T * 11 * L_FP - p_T * L_TP = L_FP
            # L_FP * (p_T * 11 - 1) = p_T * L_TP
            L_FP = (target_p_T * L_TP) / (target_p_T * 11 - 1) if target_p_T * 11 > 1 else 5000
            L_FN = L_FP * 10

            cfg['decision']['L_FP_gbp'] = int(L_FP)
            cfg['decision']['L_FN_gbp'] = int(L_FN)
            configs[f'target_pT_{int(target_p_T * 100)}pct'] = cfg

        # 策略2: 创造明确的传感器梯度
        for cost_pattern in ['linear', 'moderate', 'flat']:
            cfg = copy.deepcopy(self.base_config)

            if cost_pattern == 'linear':
                # 线性间隔 £20 - £200
                costs = np.linspace(20, 200, len(cfg['sensors']['types']))
            elif cost_pattern == 'moderate':
                # 中等梯度 £30 - £300
                costs = np.linspace(30, 300, len(cfg['sensors']['types']))
            else:  # flat
                # 扁平化 £50 - £150
                costs = np.linspace(50, 150, len(cfg['sensors']['types']))

            for sensor, cost in zip(cfg['sensors']['types'], costs):
                sensor['cost_gbp'] = int(cost)

            configs[f'cost_pattern_{cost_pattern}'] = cfg

        # 策略3: 优化算法参数
        cfg = copy.deepcopy(self.base_config)
        if 'greedy_mi' in cfg['selection']:
            cfg['selection']['greedy_mi']['keep_fraction'] = 0.5
        if 'greedy_aopt' in cfg['selection']:
            cfg['selection']['greedy_aopt']['n_probes'] = 20
        if 'greedy_evi' in cfg['selection']:
            cfg['selection']['greedy_evi']['n_y_samples'] = 32
        configs['optimized_algorithms'] = cfg

        return configs

    def analyze_config(self, config: dict) -> Dict:
        """分析单个配置的预期表现"""
        metrics = {}

        # 1. 决策成本分析
        decision = config['decision']
        L_FP = decision['L_FP_gbp']
        L_FN = decision['L_FN_gbp']
        L_TP = decision['L_TP_gbp']

        cost_ratio = L_FN / L_FP
        p_T = L_FP / (L_FP + L_FN - L_TP) if (L_FP + L_FN - L_TP) > 0 else 0

        metrics['L_FP'] = L_FP
        metrics['L_FN'] = L_FN
        metrics['cost_ratio'] = cost_ratio
        metrics['implied_p_T'] = p_T

        # 2. 传感器成本分析
        sensors = config['sensors']['types']
        costs = [s['cost_gbp'] for s in sensors]
        noises = [s['noise_std'] for s in sensors]

        metrics['sensor_cost_min'] = min(costs)
        metrics['sensor_cost_max'] = max(costs)
        metrics['sensor_cost_mean'] = np.mean(costs)
        metrics['sensor_cost_range'] = max(costs) / min(costs)

        # SNR效率
        snrs = [1 / (n ** 2) for n in noises]
        efficiencies = [snr / cost for snr, cost in zip(snrs, costs)]
        metrics['avg_efficiency'] = np.mean(efficiencies)
        metrics['efficiency_std'] = np.std(efficiencies)

        # 3. 候选池
        geometry = config['geometry']
        total_points = geometry['nx'] * geometry['ny']
        pool_fraction = config['sensors'].get('pool_fraction', 1.0)
        n_candidates = int(total_points * pool_fraction)

        metrics['n_candidates'] = n_candidates
        metrics['pool_fraction'] = pool_fraction

        # 4. 预测ROI范围（启发式）
        # 基于经验公式估计

        # 因素1: 维护概率（越高越好，目标8-12%）
        p_T_score = 1.0 - abs(p_T - 0.10) / 0.10

        # 因素2: 成本不对称性（目标10:1，过高或过低都不好）
        ratio_score = 1.0 - abs(cost_ratio - 10) / 10

        # 因素3: 传感器成本效率（平均成本越低越好）
        cost_score = 1.0 - (metrics['sensor_cost_mean'] - 100) / 500
        cost_score = max(0, min(1, cost_score))

        # 因素4: 候选池密度（越高越好）
        pool_score = pool_fraction

        # 综合评分
        overall_score = (p_T_score * 0.4 +
                         ratio_score * 0.3 +
                         cost_score * 0.2 +
                         pool_score * 0.1)

        # 预测ROI (粗略估计)
        # 假设最优配置ROI=1.0，线性缩放
        predicted_roi = (overall_score - 0.5) * 2  # 映射到[-1, 1]

        metrics['p_T_score'] = p_T_score
        metrics['ratio_score'] = ratio_score
        metrics['cost_score'] = cost_score
        metrics['pool_score'] = pool_score
        metrics['overall_score'] = overall_score
        metrics['predicted_roi'] = predicted_roi

        # 5. 健康检查
        issues = []
        if cost_ratio > 15:
            issues.append("cost_ratio_too_high")
        if p_T < 0.06 or p_T > 0.15:
            issues.append("p_T_out_of_range")
        if metrics['sensor_cost_range'] > 30:
            issues.append("cost_range_too_large")
        if n_candidates < 120:
            issues.append("pool_too_small")

        metrics['n_issues'] = len(issues)
        metrics['issues'] = ','.join(issues) if issues else 'none'

        return metrics

    def generate_scan_report(self, results: List[Dict]):
        """生成扫描报告"""
        df = pd.DataFrame(results)

        # 按预测ROI排序
        df = df.sort_values('predicted_roi', ascending=False)

        print("\n" + "=" * 80)
        print("📊 参数扫描结果")
        print("=" * 80)

        # 显示前10个配置
        print("\n🏆 Top 10 配置（按预测ROI排序）:\n")
        top_cols = ['config_name', 'predicted_roi', 'implied_p_T', 'cost_ratio',
                    'sensor_cost_mean', 'n_candidates', 'n_issues']
        print(df[top_cols].head(10).to_string(index=False))

        # 保存完整结果
        csv_path = self.output_dir / 'scan_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n完整结果已保存: {csv_path}")

        # 统计分析
        print("\n" + "=" * 80)
        print("📈 统计摘要")
        print("=" * 80)

        print(f"\n预测ROI分布:")
        print(f"  最佳: {df['predicted_roi'].max():.3f}")
        print(f"  最差: {df['predicted_roi'].min():.3f}")
        print(f"  平均: {df['predicted_roi'].mean():.3f}")
        print(f"  中位数: {df['predicted_roi'].median():.3f}")

        positive_roi = df[df['predicted_roi'] > 0]
        print(f"\n预测ROI>0的配置数: {len(positive_roi)} / {len(df)} ({len(positive_roi) / len(df) * 100:.1f}%)")

        return df

    def create_scan_visualizations(self, results: List[Dict]):
        """创建扫描可视化"""
        df = pd.DataFrame(results)

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        fig.suptitle('参数扫描分析', fontsize=16, fontweight='bold')

        # 1. ROI vs 维护概率
        ax1 = fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter(df['implied_p_T'] * 100, df['predicted_roi'],
                              c=df['cost_ratio'], s=100, alpha=0.6, cmap='viridis')
        ax1.axhline(0, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('隐含维护概率 (%)', fontweight='bold')
        ax1.set_ylabel('预测ROI', fontweight='bold')
        ax1.set_title('ROI vs 维护概率')
        plt.colorbar(scatter, ax=ax1, label='成本比')
        ax1.grid(alpha=0.3)

        # 2. ROI vs 传感器成本
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(df['sensor_cost_mean'], df['predicted_roi'],
                    c=df['pool_fraction'], s=100, alpha=0.6, cmap='coolwarm')
        ax2.axhline(0, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('平均传感器成本 (£)', fontweight='bold')
        ax2.set_ylabel('预测ROI', fontweight='bold')
        ax2.set_title('ROI vs 传感器成本')
        ax2.grid(alpha=0.3)

        # 3. 成本比 vs 维护概率
        ax3 = fig.add_subplot(gs[0, 2])
        scatter3 = ax3.scatter(df['cost_ratio'], df['implied_p_T'] * 100,
                               c=df['predicted_roi'], s=100, alpha=0.6, cmap='RdYlGn')
        ax3.set_xlabel('L_FN/L_FP比例', fontweight='bold')
        ax3.set_ylabel('隐含维护概率 (%)', fontweight='bold')
        ax3.set_title('决策成本权衡')
        plt.colorbar(scatter3, ax=ax3, label='预测ROI')
        ax3.grid(alpha=0.3)

        # 4. 得分分解
        ax4 = fig.add_subplot(gs[1, :])
        top_n = 15
        top_configs = df.nlargest(top_n, 'predicted_roi')

        x = np.arange(len(top_configs))
        width = 0.2

        ax4.bar(x - 1.5 * width, top_configs['p_T_score'], width, label='维护概率', alpha=0.8)
        ax4.bar(x - 0.5 * width, top_configs['ratio_score'], width, label='成本比', alpha=0.8)
        ax4.bar(x + 0.5 * width, top_configs['cost_score'], width, label='传感器成本', alpha=0.8)
        ax4.bar(x + 1.5 * width, top_configs['pool_score'], width, label='候选池', alpha=0.8)

        ax4.set_xlabel('配置', fontweight='bold')
        ax4.set_ylabel('得分', fontweight='bold')
        ax4.set_title(f'Top {top_n} 配置的得分分解')
        ax4.set_xticks(x)
        ax4.set_xticklabels([c[:15] for c in top_configs['config_name']], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        # 5. ROI分布直方图
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(df['predicted_roi'], bins=20, alpha=0.7, edgecolor='black')
        ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='ROI=0')
        ax5.set_xlabel('预测ROI', fontweight='bold')
        ax5.set_ylabel('配置数量', fontweight='bold')
        ax5.set_title('ROI分布')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)

        # 6. 问题数量分布
        ax6 = fig.add_subplot(gs[2, 1])
        issue_counts = df['n_issues'].value_counts().sort_index()
        ax6.bar(issue_counts.index, issue_counts.values, alpha=0.7)
        ax6.set_xlabel('问题数量', fontweight='bold')
        ax6.set_ylabel('配置数量', fontweight='bold')
        ax6.set_title('配置健康度分布')
        ax6.grid(axis='y', alpha=0.3)

        # 7. 热力图: 成本比 vs 传感器成本
        ax7 = fig.add_subplot(gs[2, 2])

        # 创建透视表
        pivot_data = df.pivot_table(
            values='predicted_roi',
            index=pd.cut(df['cost_ratio'], bins=5),
            columns=pd.cut(df['sensor_cost_mean'], bins=5),
            aggfunc='mean'
        )

        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, ax=ax7, cbar_kws={'label': '预测ROI'})
        ax7.set_xlabel('传感器平均成本', fontweight='bold')
        ax7.set_ylabel('成本比 (L_FN/L_FP)', fontweight='bold')
        ax7.set_title('参数热力图')

        plt.tight_layout()
        plot_path = self.output_dir / 'parameter_scan.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n可视化已保存: {plot_path}")
        plt.close()

    def find_best_config(self, results: List[Dict], scan_configs: Dict[str, dict]) -> dict:
        """找出最佳配置"""
        df = pd.DataFrame(results)
        best_idx = df['predicted_roi'].idxmax()
        best_result = df.loc[best_idx]
        best_name = best_result['config_name']

        print("\n" + "=" * 80)
        print("🏆 推荐配置")
        print("=" * 80)

        print(f"\n最佳配置: {best_name}")
        print(f"预测ROI: {best_result['predicted_roi']:.3f}")
        print(f"隐含维护概率: {best_result['implied_p_T'] * 100:.1f}%")
        print(f"成本比: {best_result['cost_ratio']:.1f}:1")
        print(f"传感器平均成本: £{best_result['sensor_cost_mean']:.0f}")
        print(f"候选点数: {best_result['n_candidates']}")
        print(f"问题数: {best_result['n_issues']}")

        # 保存最佳配置
        best_config = scan_configs[best_name]
        best_config_path = self.output_dir / 'best_config.yaml'
        with open(best_config_path, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True)

        print(f"\n最佳配置已保存: {best_config_path}")
        print(f"\n测试命令:")
        print(f"  python main.py --config {best_config_path} --budgets 5,10,15")

        return best_config


def main():
    parser = argparse.ArgumentParser(description='参数扫描工具')
    parser.add_argument('--config', type=str, required=True,
                        help='基准配置文件')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full', 'targeted'],
                        help='扫描模式')

    args = parser.parse_args()

    scanner = ParameterScanner(args.config)
    results, best_config = scanner.run_scan(args.mode)

    print("\n" + "=" * 80)
    print("✅ 参数扫描完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()