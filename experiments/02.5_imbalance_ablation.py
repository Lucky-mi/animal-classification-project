"""
实验2.5: 类别不平衡处理消融实验 (重点实验)
负责人: B1
功能: 对比4种损失函数处理类别不平衡的效果
       CE -> Weighted CE -> Focal Loss -> Weighted Focal Loss
使用方法:
    python experiments/02.5_imbalance_ablation.py
    python experiments/02.5_imbalance_ablation.py --losses ce focal  # 只运行指定损失函数
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
from datetime import datetime


def run_experiment(config_path: str, loss_name: str) -> dict:
    """
    运行单个实验

    Args:
        config_path: 配置文件路径
        loss_name: 损失函数名称（用于显示）

    Returns:
        result: 包含训练信息的字典
    """
    print(f"\n{'='*70}")
    print(f"损失函数: {loss_name}")
    print(f"配置文件: {config_path}")
    print(f"{'='*70}\n")

    start_time = time.time()

    # 运行训练
    cmd = f"python main.py --config {config_path}"
    result = subprocess.run(cmd, shell=True)

    end_time = time.time()
    training_time = end_time - start_time

    return {
        'loss_name': loss_name,
        'config': config_path,
        'training_time': training_time,
        'success': result.returncode == 0
    }


def main(args):
    print(f"\n{'='*70}")
    print("实验2.5: 类别不平衡处理消融实验")
    print(f"{'='*70}\n")

    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 损失函数配置 - 每个损失函数使用独立的配置文件，保存到独立路径
    loss_configs = {
        'ce': ('configs/exp2.5_ce.yaml', 'CrossEntropy (基线)'),
        'weighted_ce': ('configs/exp2.5_weighted_ce.yaml', 'Weighted CrossEntropy'),
        'focal': ('configs/exp2.5_focal.yaml', 'Focal Loss (gamma=2)'),
        'weighted_focal': ('configs/exp2.5_weighted_focal.yaml', 'Weighted Focal Loss (最佳方案)')
    }

    # 筛选要运行的损失函数
    if args.losses:
        loss_configs = {k: v for k, v in loss_configs.items() if k in args.losses}

    if len(loss_configs) == 0:
        print("没有选中任何损失函数!")
        return

    print("将运行以下损失函数:")
    for key, (config, name) in loss_configs.items():
        print(f"  - {name}")
        print(f"    配置: {config}")
    print()

    # 运行实验
    results = []
    for loss_key, (config_path, loss_name) in loss_configs.items():
        result = run_experiment(config_path, loss_name)
        result['loss_key'] = loss_key
        results.append(result)

        print(f"\n{loss_name} 训练完成!")
        print(f"耗时: {result['training_time']/60:.2f} 分钟")
        print(f"状态: {'成功' if result['success'] else '失败'}")

    # 生成汇总报告
    output_dir = 'outputs/exp2.5_comparison'
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'imbalance_summary.csv'), index=False)

    # 生成报告
    report_path = os.path.join(output_dir, 'imbalance_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("实验2.5: 类别不平衡处理消融实验报告\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("实验设计:\n")
        f.write("-" * 70 + "\n")
        f.write("研究问题: 不同损失函数对类别不平衡问题的处理效果\n")
        f.write("控制变量: 模型(ResNet-18), 数据增强(basic), 其他超参数\n")
        f.write("自变量: 损失函数类型\n")
        f.write("\n")

        f.write("损失函数说明:\n")
        f.write("1. CrossEntropy: 标准交叉熵，不考虑类别不平衡\n")
        f.write("2. Weighted CE: 根据类别样本数反比加权\n")
        f.write("3. Focal Loss: 降低易分类样本权重 (gamma=2)\n")
        f.write("4. Weighted Focal: 结合类别权重和Focal机制\n")
        f.write("\n")

        f.write("训练结果:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'损失函数':<35} {'训练时间(分钟)':<15} {'状态':<10}\n")
        f.write("-" * 70 + "\n")

        for r in results:
            status = '成功' if r['success'] else '失败'
            f.write(f"{r['loss_name']:<35} {r['training_time']/60:<15.2f} {status:<10}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("\n模型保存路径:\n")
        f.write("-" * 70 + "\n")
        for loss_key, (config, name) in loss_configs.items():
            exp_name = os.path.splitext(os.path.basename(config))[0]
            f.write(f"{name}:\n")
            f.write(f"  outputs/{exp_name}/models/best_model.pth\n")

        f.write("\n后续步骤:\n")
        f.write("-" * 70 + "\n")
        f.write("评估各模型:\n")
        for loss_key, (config, name) in loss_configs.items():
            exp_name = os.path.splitext(os.path.basename(config))[0]
            f.write(f"python evaluate.py --checkpoint outputs/{exp_name}/models/best_model.pth --config {config} --split test --analyze_errors\n")

    print(f"\n{'='*70}")
    print("实验2.5完成!")
    print(f"报告保存至: {report_path}")
    print(f"{'='*70}\n")

    # 打印评估命令
    print("\n后续步骤 - 评估各模型:")
    print("-" * 70)
    for loss_key, (config, name) in loss_configs.items():
        exp_name = os.path.splitext(os.path.basename(config))[0]
        checkpoint = f"outputs/{exp_name}/models/best_model.pth"
        print(f"# {name}")
        print(f"python evaluate.py --checkpoint {checkpoint} --config {config} --split test --analyze_errors")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='实验2.5: 类别不平衡处理')

    parser.add_argument(
        '--losses',
        nargs='+',
        default=None,
        choices=['ce', 'weighted_ce', 'focal', 'weighted_focal'],
        help='要运行的损失函数列表 (默认运行全部)'
    )

    args = parser.parse_args()
    main(args)
