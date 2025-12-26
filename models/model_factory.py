"""
模型工厂
负责人: B2
功能: 统一接口获取不同模型（ResNet-18, MobileNetV2, DeiT-Tiny）
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from torchinfo import summary
from typing import Tuple


def get_model(
    model_name: str,
    num_classes: int = 10,
    pretrained: bool = True,
    dropout: float = 0.3,
    device: str = 'cuda'
) -> nn.Module:
    """
    获取模型
    
    Args:
        model_name: 模型名称
            - 'resnet18': ResNet-18 (经典CNN)
            - 'mobilenet_v2': MobileNetV2 (轻量级CNN)
            - 'deit_tiny': DeiT-Tiny (轻量级Transformer)
        num_classes: 分类数量
        pretrained: 是否使用ImageNet预训练权重
        dropout: Dropout比例
        device: 设备
    
    Returns:
        model: 模型实例
    """
    
    print(f"\n{'='*60}")
    print(f"🔧 创建模型: {model_name}")
    print(f"   预训练: {pretrained}")
    print(f"   类别数: {num_classes}")
    print(f"   Dropout: {dropout}")
    print(f"{'='*60}\n")
    
    if model_name == 'resnet18':
        # ResNet-18 (经典基线)
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(weights=None)
        
        # 修改最后的全连接层
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    
    elif model_name == 'mobilenet_v2':
        # MobileNetV2 (轻量级)
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            model = models.mobilenet_v2(weights=weights)
        else:
            model = models.mobilenet_v2(weights=None)
        
        # 修改分类器
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    
    elif model_name == 'deit_tiny':
        # DeiT-Tiny (Data-efficient Image Transformer)
        # 使用timm库
        if pretrained:
            model = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
        else:
            model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
        
        # DeiT已经有内置的dropout，可以调整
        if hasattr(model, 'head_drop'):
            model.head_drop = nn.Dropout(dropout)
    
    else:
        raise ValueError(f"未知的模型名称: {model_name}")
    
    model = model.to(device)
    
    # 打印模型信息
    print_model_info(model, model_name, device)
    
    return model


def print_model_info(model: nn.Module, model_name: str, device: str):
    """打印模型信息"""
    try:
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 {model_name} 模型统计:")
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   参数量(MB): {total_params * 4 / 1024 / 1024:.2f}")
        
        # 使用torchinfo打印详细信息
        summary(model, input_size=(1, 3, 224, 224), device=device, verbose=0)
        
    except Exception as e:
        print(f"⚠️ 无法打印模型详细信息: {e}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数量
    
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_flops(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> float:
    """
    计算模型FLOPs (需要安装thop库)
    
    Args:
        model: 模型
        input_size: 输入尺寸 (batch, channels, height, width)
    
    Returns:
        flops: FLOPs数量
    """
    try:
        from thop import profile
        
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        return flops
    
    except ImportError:
        print("⚠️ 需要安装thop库来计算FLOPs: pip install thop")
        return 0.0


if __name__ == "__main__":
    # 测试所有模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_names = ['resnet18', 'mobilenet_v2', 'deit_tiny']
    
    for name in model_names:
        print(f"\n{'='*80}")
        model = get_model(
            model_name=name,
            num_classes=10,
            pretrained=True,
            dropout=0.3,
            device=device
        )
        
        # 测试前向传播
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        output = model(dummy_input)
        print(f"\n✅ 输入: {dummy_input.shape} -> 输出: {output.shape}")
        
        del model
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print("✅ 所有模型测试通过！")