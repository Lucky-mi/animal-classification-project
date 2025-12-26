"""
实验5: t-SNE特征可视化
负责人: B2
功能: 使用t-SNE降维可视化模型学到的特征
使用方法:
    python experiments/05_tsne.py
"""

import os
import sys
sys.path.append('..')

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
    
    # 加载ResNet-18模型
    model_name = 'resnet18'
    checkpoint_path = '../outputs/exp1_baseline/models/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        print("请先运行实验1训练模型")
        return
    
    print(f"📥 加载模型: {model_name}")
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
    print(f"✅ 模型加载成功\n")
    
    # 加载测试数据
    print("📦 加载测试数据...")
    val_transform = get_val_transform()
    
    _, _, test_loader, _ = get_dataloaders(
        root_dir='../Animals-10',
        csv_file='train_test_val_split.csv',
        train_transform=val_transform,
        val_transform=val_transform,
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建输出目录
    output_dir = '../outputs/tsne_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成t-SNE可视化
    save_path = os.path.join(output_dir, f'tsne_{model_name}.png')
    
    visualize_tsne(
        model=model,
        data_loader=test_loader,
        class_names=class_names,
        save_path=save_path,
        n_components=2,
        perplexity=30,
        device=device
    )
    
    print(f"\n{'='*70}")
    print("✅ t-SNE可视化完成！")
    print(f"📂 结果保存在: {output_dir}")
    print(f"{'='*70}\n")
    
    print("\n📊 t-SNE分析建议:")
    print("1. 观察不同类别的聚类效果")
    print("2. 检查是否有类别混淆（如cat和dog聚在一起）")
    print("3. 如果某个类别分散，说明类内差异大，需要:")
    print("   - 检查数据质量")
    print("   - 增加数据增强")
    print("   - 使用更强的特征提取器")
    print()


if __name__ == "__main__":
    main()