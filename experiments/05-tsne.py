"""
实验5: t-SNE特征可视化
负责人: B2
功能: 使用t-SNE降维可视化模型学到的特征
使用方法:
    python experiments/05_tsne.py
"""

import os
import sys

# 切换到project目录，确保相对路径正确
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)
sys.path.append(project_dir)

import torch
import yaml

from data import get_dataloaders, get_val_transform
from models import get_model
from utils.visualizer import visualize_tsne


def main():
    print(f"\n{'='*70}")
    print("🎨 实验5: t-SNE特征可视化")
    print(f"{'='*70}\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 类别名称
    class_names = [
        'butterfly', 'cat', 'chicken', 'cow', 'dog',
        'elephant', 'horse', 'sheep', 'spider', 'squirrel'
    ]

    # 要可视化的模型列表
    model_configs = [
        ('resnet18', '../outputs/exp1_baseline/models/best_model.pth', 'ResNet-18 (Baseline)'),
        ('mobilenet_v2', '../outputs/exp1_mobilenet/models/best_model.pth', 'MobileNetV2'),
        ('resnet18', '../outputs/exp2.5_imbalance/models/best_model.pth', 'ResNet-18 (Weighted Focal)'),
    ]

    # 加载测试数据（只加载一次）
    print("📦 加载测试数据...")
    val_transform = get_val_transform()

    # 检测是否有CUDA
    use_pin_memory = torch.cuda.is_available()
    num_workers = 4 if torch.cuda.is_available() else 0

    _, _, test_loader, _ = get_dataloaders(
        root_dir='../Animals-10',
        csv_file='train_test_val_split.csv',
        train_transform=val_transform,
        val_transform=val_transform,
        batch_size=32,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    print("✅ 数据加载完成\n")

    # 创建输出目录
    output_dir = '../outputs/tsne_visualization'
    os.makedirs(output_dir, exist_ok=True)

    # 为每个模型生成t-SNE
    for model_name, checkpoint_path, display_name in model_configs:
        print(f"\n{'-'*70}")
        print(f"📥 处理模型: {display_name}")
        print(f"{'-'*70}")

        if not os.path.exists(checkpoint_path):
            print(f"⚠️ 跳过 - Checkpoint不存在: {checkpoint_path}")
            continue

        # 加载模型
        model = get_model(
            model_name=model_name,
            num_classes=10,
            pretrained=False,
            dropout=0.3,
            device=device
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✅ 模型加载成功")

        # 生成文件名（去除特殊字符）
        safe_name = display_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        save_path = os.path.join(output_dir, f'tsne_{safe_name}.png')

        # 生成t-SNE可视化
        print(f"🔄 生成t-SNE可视化...")
        visualize_tsne(
            model=model,
            data_loader=test_loader,
            class_names=class_names,
            save_path=save_path,
            n_components=2,
            perplexity=30,
            device=device
        )
        print(f"💾 保存至: {save_path}")

        # 清理显存
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("✅ 所有t-SNE可视化完成！")
    print(f"📂 结果保存在: {output_dir}")
    print(f"{'='*70}\n")

    print("\n📊 t-SNE分析建议:")
    print("1. 对比不同模型的聚类效果")
    print("2. 检查是否有类别混淆（如cat和dog聚在一起）")
    print("3. 观察Weighted Focal Loss是否改善了少数类的聚类")
    print("4. 如果某个类别分散，说明类内差异大")
    print()


if __name__ == "__main__":
    main()