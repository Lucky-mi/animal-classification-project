"""
数据增强模块
负责人: B1
功能: 定义训练和验证的数据增强策略，包括MixUp
"""

import torch
import numpy as np
from torchvision import transforms
from typing import Tuple


def get_train_transform(image_size: int = 224, use_augmentation: str = 'basic') -> transforms.Compose:
    """
    获取训练集数据增强
    
    Args:
        image_size: 图像大小
        use_augmentation: 'none', 'basic', 'mixup'
    
    Returns:
        transform: 数据增强组合
    """
    
    if use_augmentation == 'none':
        # 无增强（用于消融实验）
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    elif use_augmentation == 'basic':
        # 基础增强
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    elif use_augmentation == 'advanced':
        # 高级增强（用于对比实验）
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                 saturation=0.3, hue=0.15),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    else:
        raise ValueError(f"未知的增强类型: {use_augmentation}")


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """
    获取验证/测试集数据增强（无随机操作）
    
    Args:
        image_size: 图像大小
    
    Returns:
        transform: 数据增强组合
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


class MixUpAugmentation:
    """
    MixUp数据增强
    论文: mixup: Beyond Empirical Risk Minimization (ICLR 2018)
    
    使用方法:
        mixup = MixUpAugmentation(alpha=1.0)
        images, labels_a, labels_b, lam = mixup(images, labels)
        
        # 在训练循环中:
        outputs = model(images)
        loss = lam * criterion(outputs, labels_a) + (1-lam) * criterion(outputs, labels_b)
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Beta分布的参数，控制混合强度
                  alpha=1.0: 均匀混合
                  alpha>1.0: 更倾向于原始样本
                  alpha<1.0: 更激进的混合
        """
        self.alpha = alpha
    
    def __call__(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        对一个batch应用MixUp
        
        Args:
            images: [batch_size, C, H, W]
            labels: [batch_size]
        
        Returns:
            mixed_images: 混合后的图像
            labels_a: 第一个标签
            labels_b: 第二个标签
            lam: 混合系数
        """
        batch_size = images.size(0)
        
        # 从Beta分布采样混合系数
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # 随机打乱索引
        index = torch.randperm(batch_size).to(images.device)
        
        # 混合图像: x_mix = λ*x_i + (1-λ)*x_j
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # 返回两个标签
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_images, labels_a, labels_b, lam


if __name__ == "__main__":
    # 测试数据增强
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # 创建测试图像
    test_img = Image.new('RGB', (500, 500), color='red')
    
    # 测试不同增强
    transforms_dict = {
        'none': get_train_transform(use_augmentation='none'),
        'basic': get_train_transform(use_augmentation='basic'),
        'advanced': get_train_transform(use_augmentation='advanced'),
        'val': get_val_transform()
    }
    
    print("✅ 数据增强测试:")
    for name, transform in transforms_dict.items():
        img_tensor = transform(test_img)
        print(f"{name}: {img_tensor.shape}, 范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    
    # 测试MixUp
    print("\n✅ MixUp测试:")
    mixup = MixUpAugmentation(alpha=1.0)
    batch_images = torch.randn(4, 3, 224, 224)
    batch_labels = torch.tensor([0, 1, 2, 3])
    
    mixed_images, labels_a, labels_b, lam = mixup(batch_images, batch_labels)
    print(f"混合图像形状: {mixed_images.shape}")
    print(f"标签A: {labels_a}, 标签B: {labels_b}, λ: {lam:.3f}")