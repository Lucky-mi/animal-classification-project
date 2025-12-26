"""
主训练脚本
功能: 统一的训练入口，支持不同的实验配置
使用方法:
    python main.py --config configs/exp1_baseline.yaml
    python main.py --config configs/exp2.5_imbalance.yaml --model resnet18
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random

# 导入自定义模块
from data import get_dataloaders, get_train_transform, get_val_transform, get_loss_function
from models import get_model
from utils import Trainer


def set_seed(seed: int = 42):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 随机种子设置为: {seed}")


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """主训练流程"""
    
    # 加载配置
    print(f"\n{'='*70}")
    print(f"📁 加载配置文件: {args.config}")
    print(f"{'='*70}\n")
    
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    if args.model:
        config['model']['name'] = args.model
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.loss:
        config['loss']['type'] = args.loss
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设置设备
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用设备: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # 创建输出目录
    output_root = config['output']['root_dir']
    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    exp_output_dir = os.path.join(output_root, exp_name)
    
    model_save_dir = os.path.join(exp_output_dir, 'models')
    log_dir = os.path.join(exp_output_dir, 'logs')
    
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"📂 输出目录: {exp_output_dir}\n")
    
    # 准备数据增强
    use_augmentation = config.get('use_augmentation', 'basic')
    train_transform = get_train_transform(
        image_size=config['data']['image_size'],
        use_augmentation=use_augmentation
    )
    val_transform = get_val_transform(
        image_size=config['data']['image_size']
    )
    
    # 加载数据
    print(f"{'='*70}")
    print("📦 加载数据集")
    print(f"{'='*70}\n")

    # 只有在CUDA可用时才使用pin_memory
    use_pin_memory = config['training']['pin_memory'] and torch.cuda.is_available()
    # Windows下num_workers>0可能有问题，CPU模式下建议设为0
    num_workers = config['training']['num_workers'] if torch.cuda.is_available() else 0

    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        root_dir=config['data']['root_dir'],
        csv_file=config['data']['csv_file'],
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config['training']['batch_size'],
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    # 获取类别统计（用于类别不平衡处理）
    from data.dataset import AnimalDataset
    temp_dataset = AnimalDataset(
        root_dir=config['data']['root_dir'],
        csv_file=config['data']['csv_file'],
        split='train',
        transform=None,
        class_to_idx=class_to_idx
    )
    class_counts = temp_dataset.get_class_distribution()
    
    # 创建模型
    print(f"\n{'='*70}")
    print("🤖 创建模型")
    print(f"{'='*70}\n")
    
    model = get_model(
        model_name=config['model']['name'],
        num_classes=config['data']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout'],
        device=device
    )
    
    # 创建损失函数
    print(f"\n{'='*70}")
    print("📊 创建损失函数")
    print(f"{'='*70}\n")
    
    criterion = get_loss_function(
        loss_type=config['loss']['type'],
        class_counts=class_counts if 'weighted' in config['loss']['type'] or 'focal' in config['loss']['type'] else None,
        num_classes=config['data']['num_classes'],
        device=device
    )
    
    # 创建优化器
    optimizer_config = config['training']['optimizer']
    
    if optimizer_config['type'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_config['type'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=0.9,
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        raise ValueError(f"未知的优化器: {optimizer_config['type']}")
    
    print(f"\n✅ 优化器: {optimizer_config['type']}")
    print(f"   学习率: {optimizer_config['lr']}")
    print(f"   权重衰减: {optimizer_config['weight_decay']}")
    
    # 创建学习率调度器
    scheduler_config = config['training']['scheduler']
    
    if scheduler_config['type'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config['eta_min']
        )
    else:
        scheduler = None
    
    if scheduler:
        print(f"✅ 学习率调度器: {scheduler_config['type']}\n")
    
    # 创建训练器
    use_mixup = config.get('use_mixup', False)
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=model_save_dir,
        use_tensorboard=config['logging']['use_tensorboard'],
        log_dir=log_dir,
        use_mixup=use_mixup,
        mixup_alpha=config.get('mixup_alpha', 1.0) if use_mixup else 1.0
    )
    
    # 开始训练
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        save_best_only=config['output']['save_best_only'],
        save_frequency=config['output']['save_frequency']
    )
    
    # 保存配置文件副本
    config_save_path = os.path.join(exp_output_dir, 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    print(f"\n💾 配置文件已保存: {config_save_path}")
    print(f"\n{'='*70}")
    print("🎉 训练完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Animal-10 图像分类训练')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径 (例: configs/exp1_baseline.yaml)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='模型名称 (覆盖配置文件)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='批次大小 (覆盖配置文件)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练轮数 (覆盖配置文件)'
    )
    
    parser.add_argument(
        '--loss',
        type=str,
        default=None,
        help='损失函数类型 (覆盖配置文件)'
    )
    
    args = parser.parse_args()
    
    main(args)