"""
实验2: 数据增强消融实验
负责人: B1
功能: 对比三种数据增强策略: 无增强 -> 基础增强 -> 基础+MixUp
使用方法:
    python experiments/02_augmentation_ablation.py
    python experiments/02_augmentation_ablation.py --strategies none basic  # 只运行指定策略
"""

import os
import sys

# 切换到project目录，确保相对路径正确
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)
sys.path.append(project_dir)

import argparse
import subprocess
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def run_experiment(config_path: str, strategy_name: str, output_suffix: str = None) -> dict:
    """
    运行单个实验

    Args:
        config_path: 配置文件路径
        strategy_name: 策略名称
        output_suffix: 输出目录后缀

    Returns:
        result: 包含训练信息的字典
    """
    print(f"\n{'='*70}")
    print(f"数据增强策略: {strategy_name}")
    print(f"配置文件: {config_path}")
    print(f"{'='*70}\n")

    start_time = time.time()

    # 运行训练
    cmd = f"python main.py --config {config_path}"
    result = subprocess.run(cmd, shell=True)

    end_time = time.time()
    training_time = end_time - start_time

    return {
        'strategy': strategy_name,
        'config': config_path,
        'training_time': training_time,
        'success': result.returncode == 0
    }


def main(args):
    print(f"\n{'='*70}")
    print("实验2: 数据增强消融实验")
    print(f"{'='*70}\n")

    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 增强策略配置
    # 按PDF要求: 无增强 -> 基础增强 -> 基础+MixUp
    strategy_configs = {
        'none': 'configs/exp2_no_aug.yaml',           # 无增强
        'basic': 'configs/exp2_augmentation.yaml',    # 基础增强
        'basic_mixup': 'configs/exp2_mixup.yaml'      # 基础+MixUp
    }

    strategy_names = {
        'none': '无增强 (仅Resize+Normalize)',
        'basic': '基础增强 (RandomCrop+Flip+ColorJitter)',
        'basic_mixup': '基础增强 + MixUp'
    }

    # 筛选要运行的策略
    if args.strategies:
        strategy_configs = {k: v for k, v in strategy_configs.items() if k in args.strategies}

    if len(strategy_configs) == 0:
        print("没有选中任何策略!")
        return

    print("将运行以下增强策略:")
    for key in strategy_configs:
        print(f"  - {strategy_names[key]}")
    print()

    # 运行实验
    results = []
    for strategy_key, config_path in strategy_configs.items():
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}")
            continue

        result = run_experiment(config_path, strategy_names[strategy_key])
        result['strategy_key'] = strategy_key
        results.append(result)

        print(f"\n{strategy_names[strategy_key]} 训练完成!")
        print(f"耗时: {result['training_time']/60:.2f} 分钟")
        print(f"状态: {'成功' if result['success'] else '失败'}")

    # 生成汇总报告
    output_dir = 'outputs/exp2_ablation'
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'ablation_summary.csv'), index=False)

    # 生成报告
    report_path = os.path.join(output_dir, 'ablation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("实验2: 数据增强消融实验报告\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("实验设计:\n")
        f.write("-" * 70 + "\n")
        f.write("研究问题: 数据增强对模型性能的影响\n")
        f.write("控制变量: 模型(ResNet-18), 损失函数(CE), 其他超参数\n")
        f.write("自变量: 数据增强策略\n")
        f.write("\n")

        f.write("增强策略说明:\n")
        f.write("1. 无增强: 仅Resize(256)->CenterCrop(224)->Normalize\n")
        f.write("2. 基础增强: RandomResizedCrop + RandomHorizontalFlip + ColorJitter\n")
        f.write("3. 基础+MixUp: 基础增强 + MixUp(alpha=1.0)\n")
        f.write("\n")

        f.write("训练结果:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'策略':<35} {'训练时间(分钟)':<15} {'状态':<10}\n")
        f.write("-" * 70 + "\n")

        for r in results:
            status = '成功' if r['success'] else '失败'
            f.write(f"{r['strategy']:<35} {r['training_time']/60:<15.2f} {status:<10}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("\n预期结果:\n")
        f.write("无增强 < 基础增强 < 基础+MixUp (准确率递增)\n")
        f.write("\n后续步骤:\n")
        f.write("1. 使用evaluate.py评估各策略的测试集准确率\n")
        f.write("2. 绘制准确率对比图\n")
        f.write("3. 分析MixUp对过拟合的抑制效果\n")

    print(f"\n{'='*70}")
    print("实验2完成!")
    print(f"报告保存至: {report_path}")
    print(f"{'='*70}\n")

    # 打印评估命令
    print("\n后续步骤 - 评估各策略:")
    exp_names = {
        'none': 'exp2_no_aug',
        'basic': 'exp2_augmentation',
        'basic_mixup': 'exp2_mixup'
    }

    for strategy_key in strategy_configs:
        exp_name = exp_names.get(strategy_key)
        if exp_name:
            checkpoint = f"outputs/{exp_name}/models/best_model.pth"
            config = strategy_configs[strategy_key]
            print(f"  python evaluate.py --checkpoint {checkpoint} --config {config}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='实验2: 数据增强消融')

    parser.add_argument(
        '--strategies',
        nargs='+',
        default=None,
        choices=['none', 'basic', 'basic_mixup'],
        help='要运行的增强策略 (默认运行全部)'
    )

    args = parser.parse_args()
    main(args)
