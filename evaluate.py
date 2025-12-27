"""
模型评估脚本
功能: 评估训练好的模型，生成完整的评估报告
使用方法:
    python evaluate.py --checkpoint ../outputs/exp1_baseline/models/best_model.pth --config configs/exp1_baseline.yaml
"""

import os
import yaml
import argparse
import torch
import numpy as np

from data import get_dataloaders, get_val_transform
from models import get_model
from utils import Evaluator, plot_confusion_matrix, ErrorAnalyzer


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """主评估流程"""
    
    print(f"\n{'='*70}")
    print(f"📊 模型评估")
    print(f"{'='*70}\n")
    
    # 加载配置
    config = load_config(args.config)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用设备: {device}\n")
    
    # 加载数据
    val_transform = get_val_transform(image_size=config['data']['image_size'])
    
    _, val_loader, test_loader, class_to_idx = get_dataloaders(
        root_dir=config['data']['root_dir'],
        csv_file=config['data']['csv_file'],
        train_transform=val_transform,  # 评估时不需要训练增强
        val_transform=val_transform,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    class_names = config['data']['classes']
    
    # 加载模型
    print(f"🔧 加载模型: {config['model']['name']}")
    model = get_model(
        model_name=config['model']['name'],
        num_classes=config['data']['num_classes'],
        pretrained=False,  # 评估时不需要预训练权重
        dropout=config['model']['dropout'],
        device=device
    )
    
    # 加载checkpoint
    print(f"📥 加载checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   训练epoch: {checkpoint['epoch']}")
    print(f"   最佳验证准确率: {checkpoint['best_val_acc']:.2f}%\n")
    
    # 创建评估器
    evaluator = Evaluator(
        model=model,
        device=device,
        class_names=class_names
    )
    
    # 选择评估集
    if args.split == 'val':
        eval_loader = val_loader
        print("📊 评估验证集...")
    else:
        eval_loader = test_loader
        print("📊 评估测试集...")
    
    # 评估
    results = evaluator.evaluate(eval_loader, return_predictions=True)
    
    # 打印结果
    evaluator.print_results(results)
    
 # 创建输出目录
    # 优先从checkpoint路径推断exp_name: ../outputs/exp_name/models/best_model.pth
    checkpoint_abs = os.path.abspath(args.checkpoint)
    checkpoint_dir = os.path.dirname(os.path.dirname(checkpoint_abs))
    exp_name = os.path.basename(checkpoint_dir)

      # 如果推断失败，尝试从配置文件读取或使用配置文件名
    if exp_name in ['outputs', '..', '', 'project']:
        if 'exp_name' in config:
            exp_name = config['exp_name']
        else:
            exp_name = os.path.splitext(os.path.basename(args.config))[0]

    output_dir = os.path.join('../outputs', exp_name, 'evaluation')
    print(f"📁 输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存混淆矩阵
    conf_matrix_path = os.path.join(output_dir, f'confusion_matrix_{args.split}.png')
    plot_confusion_matrix(
        conf_matrix=results['confusion_matrix'],
        class_names=class_names,
        save_path=conf_matrix_path,
        normalize=False
    )
    
    # 保存归一化混淆矩阵
    conf_matrix_norm_path = os.path.join(output_dir, f'confusion_matrix_{args.split}_normalized.png')
    plot_confusion_matrix(
        conf_matrix=results['confusion_matrix'],
        class_names=class_names,
        save_path=conf_matrix_norm_path,
        normalize=True
    )
    
    # 错误样本分析
    if args.analyze_errors:
        print(f"\n{'='*70}")
        print("🔍 错误样本分析")
        print(f"{'='*70}\n")
        
        analyzer = ErrorAnalyzer(
            class_names=class_names,
            save_dir=os.path.join(output_dir, 'error_analysis')
        )
        
        # 获取错误样本
        misclassified = evaluator.get_misclassified_samples(eval_loader)
        
        # 保存错误样本
        from data.dataset import AnimalDataset
        eval_dataset = AnimalDataset(
            root_dir=config['data']['root_dir'],
            csv_file=config['data']['csv_file'],
            split=args.split,
            transform=val_transform,
            class_to_idx=class_to_idx
        )
        
        analyzer.save_misclassified_samples(
            misclassified=misclassified,
            dataset=eval_dataset,
            max_samples_per_pair=10
        )
        
        # 可视化错误样本
        analyzer.visualize_error_samples(
            misclassified=misclassified,
            top_k_pairs=5,
            samples_per_pair=6
        )
        
        # 生成分析报告
        analyzer.generate_report(
            conf_matrix=results['confusion_matrix'],
            misclassified=misclassified
        )
    
    # 保存评估结果
    results_path = os.path.join(output_dir, f'results_{args.split}.npz')
    np.savez(
        results_path,
        accuracy=results['accuracy'],
        macro_f1=results['macro_f1'],
        confusion_matrix=results['confusion_matrix'],
        per_class_precision=results['per_class_metrics']['precision'],
        per_class_recall=results['per_class_metrics']['recall'],
        per_class_f1=results['per_class_metrics']['f1'],
        per_class_accuracy=results['per_class_metrics']['accuracy'],
        predictions=results['predictions'],
        labels=results['labels']
    )
    
    print(f"\n💾 评估结果保存至: {output_dir}")
    print(f"\n{'='*70}")
    print("✅ 评估完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型评估')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型checkpoint路径'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='评估哪个数据集'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='批次大小'
    )
    
    parser.add_argument(
        '--analyze_errors',
        action='store_true',
        help='是否进行错误样本分析'
    )
    
    args = parser.parse_args()
    
    main(args)