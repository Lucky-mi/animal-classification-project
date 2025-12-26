"""
实验3: Grad-CAM可视化
负责人: B3
功能: 对比不同模型的Grad-CAM热力图，分析模型关注区域
使用方法:
    python experiments/03_gradcam.py
"""

import os
import sys
sys.path.append('..')

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from data import get_val_transform
from models import get_model
from utils.visualizer import GradCAMVisualizer, get_target_layer, plot_gradcam_comparison


def main():
    print(f"\n{'='*70}")
    print("🔥 实验3: Grad-CAM可视化")
    print(f"{'='*70}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 类别名称
    class_names = [
        'butterfly', 'cat', 'chicken', 'cow', 'dog',
        'elephant', 'horse', 'sheep', 'spider', 'squirrel'
    ]
    
    # 加载模型
    models_dict = {}
    model_configs = [
        ('resnet18', '../outputs/exp1_baseline/models/best_model.pth'),
        ('mobilenet_v2', '../outputs/exp1_mobilenet/models/best_model.pth'),
    ]
    
    for model_name, checkpoint_path in model_configs:
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ 跳过{model_name}: checkpoint不存在")
            continue
        
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
        
        # 获取目标层
        try:
            target_layer = get_target_layer(model, model_name)
            models_dict[model_name] = (model, target_layer)
            print(f"✅ {model_name} 加载成功\n")
        except NotImplementedError as e:
            print(f"⚠️ {model_name} 不支持Grad-CAM: {e}\n")
    
    if len(models_dict) == 0:
        print("❌ 没有可用的模型，请先训练模型")
        return
    
    # 准备测试图像
    # 这里我们从测试集中随机选择几张图像
    import pandas as pd
    
    csv_path = '../Animals-10/train_test_val_split.csv'
    df = pd.read_csv(csv_path)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    # 为每个类别选择一张图像
    output_dir = '../outputs/gradcam_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    transform = get_val_transform()
    
    print(f"🖼️  为每个类别生成Grad-CAM可视化...\n")
    
    for class_name in class_names[:5]:  # 选择前5个类别作为示例
        # 找到该类别的一张图像
        class_samples = test_df[test_df['label'] == class_name]
        
        if len(class_samples) == 0:
            print(f"⚠️ 跳过{class_name}: 测试集中没有样本")
            continue
        
        # 随机选择一张
        sample = class_samples.sample(1).iloc[0]
        img_path = os.path.join('../Animals-10', sample['filename'])
        
        if not os.path.exists(img_path):
            print(f"⚠️ 图像不存在: {img_path}")
            continue
        
        # 加载图像
        original_img = Image.open(img_path).convert('RGB')
        original_img_np = np.array(original_img)
        
        # 预处理
        img_tensor = transform(original_img).unsqueeze(0)
        
        # 生成Grad-CAM对比图
        save_path = os.path.join(output_dir, f'gradcam_{class_name}.png')
        
        try:
            plot_gradcam_comparison(
                models_dict=models_dict,
                image=img_tensor,
                original_image=original_img_np,
                class_names=class_names,
                save_path=save_path
            )
            print(f"✅ {class_name} Grad-CAM已生成")
        
        except Exception as e:
            print(f"❌ {class_name} Grad-CAM生成失败: {e}")
    
    print(f"\n{'='*70}")
    print(f"✅ Grad-CAM可视化完成！")
    print(f"📂 结果保存在: {output_dir}")
    print(f"{'='*70}\n")
    
    # 生成对比分析
    print("\n📊 Grad-CAM分析建议:")
    print("1. 观察ResNet和MobileNet关注的区域是否一致")
    print("2. 检查模型是否关注了正确的特征（如猫的耳朵、狗的鼻子）")
    print("3. 如果模型关注背景而非主体，考虑:")
    print("   - 增加数据增强（背景多样化）")
    print("   - 使用注意力机制")
    print("   - 收集更多高质量样本")
    print()


if __name__ == "__main__":
    main()