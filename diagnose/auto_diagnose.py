#!/usr/bin/env python3
"""
一键ROI诊断 - 自动化诊断流程

自动运行所有诊断工具，生成完整报告

用法：
    python auto_diagnose.py --config baseline_config.yaml

选项：
    --quick-only: 仅运行快速检查
    --with-scan: 包含参数扫描
    --full: 运行所有工具（包括完整算法诊断）
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time


def run_command(cmd: list, description: str) -> bool:
    """运行命令并显示进度"""
    print(f"\n{'=' * 80}")
    print(f"▶️  {description}")
    print(f"{'=' * 80}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} - 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - 失败")
        print(f"   错误: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description} - 跳过 ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(description='一键ROI诊断')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--quick-only', action='store_true',
                        help='仅运行快速检查')
    parser.add_argument('--with-scan', action='store_true',
                        help='包含参数扫描')
    parser.add_argument('--full', action='store_true',
                        help='运行所有工具')

    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    print("=" * 80)
    print("🔍 ROI一键诊断")
    print("=" * 80)
    print(f"配置文件: {config_path}")
    print(f"模式: {'完整' if args.full else '快速' if args.quick_only else '标准'}")

    start_time = time.time()

    # 步骤1: 快速配置检查（必须）
    success1 = run_command(
        ['python', 'quick_diagnostic.py', '--config', config_path],
        "步骤1: 快速配置检查"
    )

    if not success1:
        print("\n⚠️  快速检查失败，但继续执行...")

    # 步骤2: 参数扫描（可选）
    if args.with_scan or args.full:
        mode = 'full' if args.full else 'quick'
        success2 = run_command(
            ['python', 'parameter_scan.py', '--config', config_path, '--mode', mode],
            f"步骤2: 参数扫描 ({mode}模式)"
        )

        if success2:
            # 检查是否生成了最佳配置
            best_config = Path('parameter_scan_results/best_config.yaml')
            if best_config.exists():
                print(f"\n💾 最佳配置已生成: {best_config}")
                print(f"   测试命令: python main.py --config {best_config} --budgets 5")

    # 步骤3: 完整算法诊断（仅full模式）
    if args.full:
        success3 = run_command(
            ['python', 'roi_diagnostic.py', '--config', config_path, '--budget', '5'],
            "步骤3: 完整算法诊断 (k=5)"
        )

        if not success3:
            print("\n⚠️  完整诊断需要项目模块，已跳过")

    # 生成总结报告
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("📊 诊断总结")
    print("=" * 80)

    print(f"\n⏱️  总用时: {elapsed:.1f} 秒")

    # 列出生成的文件
    print("\n📁 生成的文件:\n")

    outputs = []

    # 快速诊断输出
    quick_dir = Path('quick_diagnostics')
    if quick_dir.exists():
        outputs.append(("快速诊断", [
            quick_dir / 'diagnostic_report.txt',
            quick_dir / 'config_analysis.png',
            quick_dir / 'tuning_advice.txt'
        ]))

    # 参数扫描输出
    if args.with_scan or args.full:
        scan_dir = Path('parameter_scan_results')
        if scan_dir.exists():
            outputs.append(("参数扫描", [
                scan_dir / 'scan_results.csv',
                scan_dir / 'parameter_scan.png',
                scan_dir / 'best_config.yaml'
            ]))

    # 完整诊断输出
    if args.full:
        diag_dir = Path('diagnostics_output')
        if diag_dir.exists():
            outputs.append(("完整诊断", [
                diag_dir / 'summary_report_k5.txt',
                diag_dir / 'roi_breakdown_k5.png',
                diag_dir / 'spatial_distribution_k5.png'
            ]))

    for category, files in outputs:
        print(f"  {category}:")
        for file in files:
            if file.exists():
                print(f"    ✅ {file}")
            else:
                print(f"    ⚠️  {file} (未生成)")

    # 给出建议
    print("\n" + "=" * 80)
    print("💡 下一步建议")
    print("=" * 80)

    print("\n1. 查看快速诊断报告:")
    print("   cat quick_diagnostics/diagnostic_report.txt")

    if args.with_scan or args.full:
        print("\n2. 查看参数扫描结果:")
        print("   cat parameter_scan_results/scan_results.csv")

        best_config = Path('parameter_scan_results/best_config.yaml')
        if best_config.exists():
            print("\n3. 测试最佳配置:")
            print(f"   python main.py --config {best_config} --budgets 5,10,15")

    if args.full:
        print("\n4. 查看算法对比:")
        print("   cat diagnostics_output/summary_report_k5.txt")

    print("\n5. 查看完整使用指南:")
    print("   cat DIAGNOSTIC_GUIDE.md")

    print("\n" + "=" * 80)
    print("✅ 诊断完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()