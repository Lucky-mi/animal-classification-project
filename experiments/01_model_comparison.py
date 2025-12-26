"""
实验1: 模型对比实验
负责人: B2
功能: 自动运行ResNet-18、MobileNetV2、DeiT-Tiny三个模型的训练和评估
使用方法:
    python experiments/01_model_comparison.py
    python experiments/01_model_comparison.py --models resnet18 mobilenet_v2  # 只运行指定模型
"""

import os
import sys
sys.path.append('..')

import argparse
import subprocess
import time
import torch
import pandas as pd
from datetime import datetime


def run_experiment(config_path: str, model_name: str) -> dict:
    """
    运行单个实验

    Args:
        config_path: 配置文件路径
        model_name: 模型名称

    Returns:
        result: 包含训练时间等信息的字典
    """
    print(f"\n{'='*70}")
    print(f"开始训练: {model_name}")
    print(f"配置文件: {config_path}")
    print(f"{'='*70}\n")

    start_time = time.time()

    # 运行训练
    cmd = f"python ../main.py --config {config_path}"
    result = subprocess.run(cmd, shell=True)

    end_time = time.time()
    training_time = end_time - start_time

    return {
        'model': model_name,
        'config': config_path,
        'training_time': training_time,
        'success': result.returncode == 0
    }


def main(args):
    print(f"\n{'='*70}")
    print("实验1: 模型对比实验")
    print(f"{'='*70}\n")

    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 模型配置
    model_configs = {
        'resnet18': '../configs/exp1_baseline.yaml',
        'mobilenet_v2': '../configs/exp1_mobilenet.yaml',
        'deit_tiny': '../configs/exp1_deit.yaml'
    }

    # 筛选要运行的模型
    if args.models:
        model_configs = {k: v for k, v in model_configs.items() if k in args.models}

    if len(model_configs) == 0:
        print("没有选中任何模型!")
        return

    print(f"将运行以下模型: {list(model_configs.keys())}\n")

    # 运行实验
    results = []
    for model_name, config_path in model_configs.items():
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}")
            continue

        result = run_experiment(config_path, model_name)
        results.append(result)

        print(f"\n{model_name} 训练完成!")
        print(f"耗时: {result['training_time']/60:.2f} 分钟")
        print(f"状态: {'成功' if result['success'] else '失败'}")

    # 生成汇总报告
    output_dir = '../outputs/exp1_comparison'
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'training_summary.csv'), index=False)

    # 生成报告
    report_path = os.path.join(output_dir, 'experiment_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("实验1: 模型对比实验报告\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"设备: {device}\n")
        if device == 'cuda':
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write("\n")

        f.write("训练结果:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'模型':<20} {'训练时间(分钟)':<15} {'状态':<10}\n")
        f.write("-" * 70 + "\n")

        for r in results:
            status = '成功' if r['success'] else '失败'
            f.write(f"{r['model']:<20} {r['training_time']/60:<15.2f} {status:<10}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("\n后续步骤:\n")
        f.write("1. 运行 evaluate.py 评估各模型在测试集上的表现\n")
        f.write("2. 运行 04_efficiency.py 对比模型效率\n")
        f.write("3. 运行 03_gradcam.py 生成Grad-CAM可视化\n")

    print(f"\n{'='*70}")
    print("实验1完成!")
    print(f"报告保存至: {report_path}")
    print(f"{'='*70}\n")

    # 打印后续步骤
    print("\n后续步骤:")
    print("1. 评估模型:")
    for model_name, config_path in model_configs.items():
        exp_name = os.path.splitext(os.path.basename(config_path))[0]
        checkpoint = f"../outputs/{exp_name}/models/best_model.pth"
        print(f"   python ../evaluate.py --checkpoint {checkpoint} --config {config_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='实验1: 模型对比')

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        choices=['resnet18', 'mobilenet_v2', 'deit_tiny'],
        help='要运行的模型列表 (默认运行全部)'
    )

    args = parser.parse_args()
    main(args)
