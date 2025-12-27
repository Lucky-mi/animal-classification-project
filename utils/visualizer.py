"""
可视化模块
负责人: B2 (Grad-CAM, t-SNE) + B3 (混淆矩阵, 训练曲线)
功能: Grad-CAM热力图、t-SNE降维可视化、混淆矩阵、训练曲线
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import List, Optional, Tuple
import cv2
from sklearn.manifold import TSNE
from tqdm import tqdm
import os

# 配置中文字体支持
def setup_chinese_font():
    """设置matplotlib中文字体"""
    import platform
    system = platform.system()

    if system == 'Windows':
        # Windows系统使用微软雅黑或黑体
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun']
    elif system == 'Darwin':
        # macOS
        font_list = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
    else:
        # Linux
        font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']

    for font in font_list:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            break
        except:
            continue

# 初始化时设置字体
setup_chinese_font()


class GradCAMVisualizer:
    """
    Grad-CAM可视化器
    论文: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (ICCV 2017)
    
    负责人: B2
    功能: 生成类激活图，展示模型关注的区域
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: 要可视化的模型
            target_layer: 目标层（通常是最后一个卷积层）
            device: 设备
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        
        self.gradients = None
        self.activations = None
        
        # 注册hook
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向hook"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        生成类激活图
        
        Args:
            image: 输入图像 [1, 3, H, W]
            target_class: 目标类别（None则使用预测类别）
        
        Returns:
            cam: 类激活图 [H, W]
        """
        self.model.eval()
        
        # 前向传播
        image = image.to(self.device)
        output = self.model(image)
        
        # 如果没有指定目标类别，使用预测类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # 计算权重：全局平均池化梯度
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        
        # 加权求和
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # ReLU
        cam = torch.relu(cam)
        
        # 归一化到[0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(
        self,
        image: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        可视化Grad-CAM
        
        Args:
            image: 归一化后的输入图像 [1, 3, H, W]
            original_image: 原始图像 [H, W, 3], RGB, [0, 255]
            target_class: 目标类别
            alpha: 热力图透明度
        
        Returns:
            visualization: 叠加后的图像 [H, W, 3]
        """
        # 生成CAM
        cam = self.generate_cam(image, target_class)
        
        # 调整CAM大小到原图尺寸
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # 转换为热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 叠加
        visualization = heatmap * alpha + original_image * (1 - alpha)
        visualization = np.uint8(visualization)
        
        return visualization


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    获取目标层（最后一个卷积层）
    
    Args:
        model: 模型
        model_name: 模型名称
    
    Returns:
        target_layer: 目标层
    """
    if model_name == 'resnet18':
        return model.layer4[-1]
    
    elif model_name == 'mobilenet_v2':
        return model.features[-1]
    
    elif model_name == 'deit_tiny':
        # Transformer没有卷积层，Grad-CAM不适用
        # 可以使用Attention Rollout等方法
        raise NotImplementedError("DeiT-Tiny不支持标准Grad-CAM，请使用Attention可视化")
    
    else:
        raise ValueError(f"未知模型: {model_name}")


def plot_gradcam_comparison(
    models_dict: dict,
    image: torch.Tensor,
    original_image: np.ndarray,
    class_names: List[str],
    save_path: str
):
    """
    对比不同模型的Grad-CAM
    
    Args:
        models_dict: {'model_name': (model, target_layer)}
        image: 归一化图像 [1, 3, H, W]
        original_image: 原始图像 [H, W, 3]
        class_names: 类别名称
        save_path: 保存路径
    """
    num_models = len(models_dict)
    fig, axes = plt.subplots(1, num_models + 1, figsize=(5 * (num_models + 1), 5))
    
    # 显示原图
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # 为每个模型生成Grad-CAM
    for idx, (model_name, (model, target_layer)) in enumerate(models_dict.items(), 1):
        try:
            visualizer = GradCAMVisualizer(model, target_layer, device=image.device)
            vis = visualizer.visualize(image, original_image)
            
            axes[idx].imshow(vis)
            axes[idx].set_title(f'{model_name}', fontsize=14)
            axes[idx].axis('off')
        
        except Exception as e:
            print(f"⚠️ {model_name} Grad-CAM生成失败: {e}")
            axes[idx].text(0.5, 0.5, f'Error: {model_name}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Grad-CAM对比图保存至: {save_path}")
    plt.close()


def visualize_tsne(
    model: nn.Module,
    data_loader,
    class_names: List[str],
    save_path: str,
    n_components: int = 2,
    perplexity: int = 30,
    device: str = 'cuda'
):
    """
    t-SNE特征可视化
    
    负责人: B2
    功能: 将高维特征降维到2D，可视化不同类别的聚类效果
    
    Args:
        model: 模型
        data_loader: 数据加载器
        class_names: 类别名称
        save_path: 保存路径
        n_components: 降维维度（2或3）
        perplexity: t-SNE的困惑度参数
        device: 设备
    """
    model.eval()
    
    features_list = []
    labels_list = []
    
    print("🔍 提取特征用于t-SNE...")
    
    # 提取特征（从最后一层卷积或倒数第二层全连接）
    def hook_fn(module, input, output):
        features_list.append(output.detach().cpu())
    
    # 注册hook到倒数第二层
    if hasattr(model, 'fc'):
        # ResNet-18
        handle = model.avgpool.register_forward_hook(hook_fn)
    elif hasattr(model, 'classifier'):
        # MobileNetV2
        handle = model.features.register_forward_hook(hook_fn)
    else:
        raise NotImplementedError("不支持的模型结构")
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Extracting features'):
            images = images.to(device)
            _ = model(images)
            labels_list.append(labels)
    
    handle.remove()
    
    # 合并特征
    features = torch.cat(features_list, dim=0)
    
    # 如果是4D特征图，进行全局平均池化
    if len(features.shape) == 4:
        features = features.mean(dim=[2, 3])
    
    features = features.numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    print(f"特征形状: {features.shape}")
    
    # t-SNE降维
    print(f"🧮 执行t-SNE降维 (perplexity={perplexity})...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, verbose=1)
    features_2d = tsne.fit_transform(features)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    
    # 添加图例
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, loc='best', fontsize=10)
    
    plt.title('t-SNE Visualization of Features', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 t-SNE可视化保存至: {save_path}")
    plt.close()


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalize: bool = False,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    绘制混淆矩阵
    
    负责人: B3
    
    Args:
        conf_matrix: 混淆矩阵 [num_classes, num_classes]
        class_names: 类别名称
        save_path: 保存路径
        normalize: 是否归一化
        figsize: 图像大小
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 混淆矩阵保存至: {save_path}")
    plt.close()


def plot_training_curves(
    history: dict,
    save_path: str,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    绘制训练曲线
    
    负责人: B3
    
    Args:
        history: 训练历史，包含train_loss, train_acc, val_loss, val_acc, lr
        save_path: 保存路径
        figsize: 图像大小
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curves', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy Curves', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate曲线
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14)
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 训练曲线保存至: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("可视化模块测试需要完整的模型和数据")
    print("请在实验脚本中使用")