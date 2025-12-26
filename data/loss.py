"""
损失函数模块
负责人: B1
功能: 实现多种损失函数用于处理类别不平衡问题
- CrossEntropyLoss (基线)
- WeightedCrossEntropyLoss (加权交叉熵)
- FocalLoss (Focal Loss)
- WeightedFocalLoss (加权Focal Loss) ⭐重点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss实现
    论文: Focal Loss for Dense Object Detection (ICCV 2017)
    
    公式: FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    
    Args:
        alpha: 类别平衡因子 (None 或 Tensor[num_classes])
        gamma: 聚焦参数，通常取2.0
        reduction: 'none', 'mean', 'sum'
    """
    
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes] 模型原始输出 (logits)
            targets: [batch_size] 真实标签
        
        Returns:
            loss: 标量损失值
        """
        # 计算交叉熵（不做reduction）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算预测概率 p_t
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)
        
        # 计算focal权重: (1-p_t)^γ
        focal_weight = (1 - pt) ** self.gamma
        
        # 计算focal loss
        focal_loss = focal_weight * ce_loss
        
        # 如果提供了alpha，应用类别平衡
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # 获取每个样本对应类别的alpha值
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    加权Focal Loss（结合类别权重和Focal机制）
    这是实验2.5的最佳方案 ⭐
    
    Args:
        class_weights: 类别权重 Tensor[num_classes]
        gamma: Focal Loss的γ参数
        reduction: 'none', 'mean', 'sum'
    """
    
    def __init__(
        self,
        class_weights: torch.Tensor,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes]
            targets: [batch_size]
        
        Returns:
            loss: 标量损失值
        """
        # 使用FocalLoss实现
        focal_loss = FocalLoss(alpha=self.class_weights, gamma=self.gamma, reduction=self.reduction)
        return focal_loss(inputs, targets)


def calculate_class_weights(class_counts: dict, method: str = 'inverse') -> torch.Tensor:
    """
    计算类别权重
    
    Args:
        class_counts: 类别样本数 {'cat': 2000, 'dog': 1500, ...}
        method: 'inverse' 或 'effective'
    
    Returns:
        weights: Tensor[num_classes]
    
    示例:
        class_counts = {'cat': 2000, 'dog': 1500, 'bird': 500}
        weights = calculate_class_weights(class_counts, method='inverse')
        # weights ≈ [0.25, 0.33, 1.0] (归一化后)
    """
    # 按类别名称排序（确保顺序一致）
    sorted_classes = sorted(class_counts.keys())
    counts = torch.tensor([class_counts[c] for c in sorted_classes], dtype=torch.float32)
    
    if method == 'inverse':
        # 反比例: w_i = 1 / n_i
        weights = 1.0 / counts
        
    elif method == 'effective':
        # 有效样本数方法 (Class-Balanced Loss, CVPR 2019)
        # w_i = (1 - β) / (1 - β^n_i)
        beta = 0.9999
        weights = (1 - beta) / (1 - torch.pow(beta, counts))
    
    else:
        raise ValueError(f"未知的方法: {method}")
    
    # 归一化权重（使平均权重为1）
    weights = weights / weights.mean()
    
    print(f"\n📊 类别权重 (method={method}):")
    for cls, weight in zip(sorted_classes, weights):
        print(f"  {cls}: {weight:.3f}")
    
    return weights


def get_loss_function(
    loss_type: str,
    class_counts: Optional[dict] = None,
    num_classes: int = 10,
    device: str = 'cuda'
) -> nn.Module:
    """
    获取损失函数
    
    Args:
        loss_type: 损失函数类型
            - 'ce': 标准交叉熵
            - 'weighted_ce': 加权交叉熵
            - 'focal': Focal Loss (γ=2)
            - 'weighted_focal': 加权Focal Loss ⭐推荐
        class_counts: 类别样本数统计
        num_classes: 类别数量
        device: 设备
    
    Returns:
        criterion: 损失函数
    """
    
    if loss_type == 'ce':
        print("✅ 使用标准交叉熵损失")
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'weighted_ce':
        if class_counts is None:
            raise ValueError("weighted_ce需要提供class_counts")
        
        weights = calculate_class_weights(class_counts, method='inverse')
        weights = weights.to(device)
        
        print("✅ 使用加权交叉熵损失")
        return nn.CrossEntropyLoss(weight=weights)
    
    elif loss_type == 'focal':
        print("✅ 使用Focal Loss (γ=2.0)")
        return FocalLoss(alpha=None, gamma=2.0, reduction='mean')
    
    elif loss_type == 'weighted_focal':
        if class_counts is None:
            raise ValueError("weighted_focal需要提供class_counts")
        
        weights = calculate_class_weights(class_counts, method='inverse')
        weights = weights.to(device)
        
        print("✅ 使用加权Focal Loss (γ=2.0) ⭐最佳方案")
        return WeightedFocalLoss(class_weights=weights, gamma=2.0, reduction='mean')
    
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")


if __name__ == "__main__":
    # 测试损失函数
    print("=" * 50)
    print("测试损失函数模块")
    print("=" * 50)
    
    # 模拟数据
    batch_size = 8
    num_classes = 10
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 模拟类别不平衡
    class_counts = {
        'butterfly': 2000,
        'cat': 1800,
        'chicken': 1500,
        'cow': 2200,
        'dog': 1900,
        'elephant': 800,   # 少数类
        'horse': 1700,
        'sheep': 1600,
        'spider': 900,     # 少数类
        'squirrel': 1000   # 少数类
    }
    
    # 测试所有损失函数
    loss_types = ['ce', 'weighted_ce', 'focal', 'weighted_focal']
    
    for loss_type in loss_types:
        print(f"\n{'='*50}")
        criterion = get_loss_function(
            loss_type=loss_type,
            class_counts=class_counts if 'weighted' in loss_type else None,
            device='cpu'
        )
        
        loss = criterion(logits, targets)
        print(f"损失值: {loss.item():.4f}")
    
    print("\n" + "=" * 50)
    print("✅ 所有损失函数测试通过！")
    print("=" * 50)