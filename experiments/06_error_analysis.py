"""
实验6: 错误样本深度分析 (重点实验)
负责人: B3
功能: 深入分析被错分的样本，找出规律
使用方法:
    python experiments/06_error_analysis.py --checkpoint ../outputs/exp2.5_imbalance/models/best_model.pth
"""

import os
import sys
sys.path.append('..')

import argparse
import torch
import yaml

from data import get_dataloaders, get_val_transform
from models import get_model
from utils import Evaluator, ErrorAnalyzer


def main(args):
    print(f"\n{'='*70}")
    print("🔍 实验6: 错误样本深度分析")
    print(f"{'='*70}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 类别名称
    class_names = [
        'butterfly', 'cat', 'chicken', 'cow', 'dog',
        'elephant', 'horse', 'sheep', 'spider', 'squirrel'
    ]
    
    # 加载模型
    print(f"📥 加载模型...")
    model = get_model(
        model_name='resnet18',
        num_classes=10,
        pretrained=False,
        dropout=0.3,
        device=device
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ 模型加载成功\n")
    
    # 加载测试数据
    print("📦 加载测试数据...")
    val_transform = get_val_transform()
    
    _, _, test_loader, class_to_idx = get_dataloaders(
        root_dir='../Animals-10',
        csv_file='train_test_val_split.csv',
        train_transform=val_transform,
        val_transform=val_transform,
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    
    # 评估模型
    print("\n📊 评估模型...")
    evaluator = Evaluator(
        model=model,
        device=device,
        class_names=class_names
    )
    
    results = evaluator.evaluate(test_loader, return_predictions=True)
    evaluator.print_results(results)
    
    # 创建输出目录
    output_dir = '../outputs/error_analysis_deep'
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取错误样本
    print(f"\n{'='*70}")
    print("🔎 收集错误样本...")
    print(f"{'='*70}\n")
    
    misclassified = evaluator.get_misclassified_samples(test_loader)
    
    total_errors = sum(len(samples) for samples in misclassified.values())
    print(f"总错误样本数: {total_errors}")
    print(f"错误类型数: {len(misclassified)}\n")
    
    # 创建分析器
    analyzer = ErrorAnalyzer(
        class_names=class_names,
        save_dir=output_dir
    )
    
    # 1. 保存错误样本
    print("💾 保存错误样本...")
    from data.dataset import AnimalDataset
    
    test_dataset = AnimalDataset(
        root_dir='../Animals-10',
        csv_file='train_test_val_split.csv',
        split='test',
        transform=val_transform,
        class_to_idx=class_to_idx
    )
    
    analyzer.save_misclassified_samples(
        misclassified=misclassified,
        dataset=test_dataset,
        max_samples_per_pair=20  # 每对最多保存20个样本
    )
    
    # 2. 可视化错误样本
    print("\n🖼️  可视化错误样本...")
    analyzer.visualize_error_samples(
        misclassified=misclassified,
        top_k_pairs=5,
        samples_per_pair=8
    )
    
    # 3. 生成分析报告
    print("\n📄 生成分析报告...")
    analyzer.generate_report(
        conf_matrix=results['confusion_matrix'],
        misclassified=misclassified
    )
    
    # 4. 深度分析：针对top-3混淆对
    print(f"\n{'='*70}")
    print("🧠 深度分析Top-3混淆对")
    print(f"{'='*70}\n")
    
    top_confusion = analyzer.analyze_confusion_pairs(
        conf_matrix=results['confusion_matrix'],
        top_k=3
    )
    
    for rank, (true_cls, pred_cls, count) in enumerate(top_confusion, 1):
        print(f"\n{rank}. {true_cls} → {pred_cls} ({count}个样本)")
        print("-" * 70)
        
        # 找到对应的错误样本
        true_idx = class_names.index(true_cls)
        pred_idx = class_names.index(pred_cls)
        key = (true_idx, pred_idx)
        
        if key in misclassified:
            samples = misclassified[key]
            
            # 计算平均置信度
            confidences = [s['confidence'] for s in samples]
            avg_conf = sum(confidences) / len(confidences)
            
            print(f"平均置信度: {avg_conf:.2%}")
            print(f"最高置信度: {max(confidences):.2%}")
            print(f"最低置信度: {min(confidences):.2%}")
            
            print(f"\n💡 可能的原因:")
            
            # 基于类别特点给出分析
            if true_cls == 'cat' and pred_cls == 'dog':
                print("   - 猫和狗的毛发纹理相似")
                print("   - 可能是侧面或远景照片，缺少关键特征（如耳朵形状）")
                print("   - 建议: 使用Grad-CAM检查模型是否关注了耳朵、鼻子等区分特征")
            
            elif true_cls == 'butterfly' and pred_cls == 'spider':
                print("   - 都有多条腿和复杂的身体结构")
                print("   - 可能是微距照片，背景干扰")
                print("   - 建议: 增加数据增强，特别是背景变化")
            
            else:
                print(f"   - {true_cls}和{pred_cls}可能存在视觉相似性")
                print(f"   - 建议查看保存的错误样本图像进行人工检查")
    
    print(f"\n{'='*70}")
    print("✅ 错误样本分析完成！")
    print(f"📂 结果保存在: {output_dir}")
    print(f"{'='*70}\n")
    
    print("\n📝 后续工作建议:")
    print("1. 查看保存的错误样本图像，人工检查是否有标注错误")
    print("2. 使用Grad-CAM可视化错误样本，看模型关注哪里")
    print("3. 针对高频混淆对，考虑:")
    print("   - 收集更多区分性样本")
    print("   - 设计特定的数据增强策略")
    print("   - 调整类别权重")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='错误样本深度分析')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='../outputs/exp1_baseline/models/best_model.pth',
        help='模型checkpoint路径'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint不存在: {args.checkpoint}")
        print("请先训练模型")
        sys.exit(1)
    
    main(args)