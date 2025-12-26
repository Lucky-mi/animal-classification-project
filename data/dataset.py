"""
Animal-10 数据集加载器
负责人: B1
功能: 从CSV文件读取数据，加载图像，应用数据增强
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional, Tuple, Dict
import torch


class AnimalDataset(Dataset):
    """
    Animal-10 数据集类
    
    Args:
        root_dir: 数据集根目录
        csv_file: CSV文件路径
        split: 'train', 'val', 或 'test'
        transform: 数据增强函数
        class_to_idx: 类别到索引的映射
    """
    
    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 读取CSV文件
        csv_path = os.path.join(root_dir, csv_file)
        self.data = pd.read_csv(csv_path)
        
        # 筛选对应的split
        self.data = self.data[self.data['split'] == split].reset_index(drop=True)
        
        # 创建类别映射
        if class_to_idx is None:
            unique_labels = sorted(self.data['label'].unique())
            self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())
        
        print(f"[{split.upper()}] 加载了 {len(self.data)} 张图像")
        print(f"类别映射: {self.class_to_idx}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取一个样本
        
        Returns:
            image: 增强后的图像张量
            label: 类别索引
        """
        # 获取图像路径和标签
        row = self.data.iloc[idx]
        img_name = row['path']  # CSV列名是'path'而非'filename'
        label_name = row['label']
        
        # 构建完整路径
        img_path = os.path.join(self.root_dir, img_name)
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 加载图像失败: {img_path}, 错误: {e}")
            # 返回一个黑色图像作为备用
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 应用数据增强
        if self.transform:
            image = self.transform(image)
        
        # 获取标签索引
        label = self.class_to_idx[label_name]
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        return self.data['label'].value_counts().to_dict()


def get_dataloaders(
    root_dir: str,
    csv_file: str,
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    创建训练、验证、测试的DataLoader
    
    Args:
        root_dir: 数据集根目录
        csv_file: CSV文件名
        train_transform: 训练集数据增强
        val_transform: 验证/测试集数据增强
        batch_size: 批次大小
        num_workers: 数据加载线程数
        pin_memory: 是否使用pin_memory加速
    
    Returns:
        train_loader, val_loader, test_loader, class_to_idx
    """
    
    # 先创建训练集以获取class_to_idx
    train_dataset = AnimalDataset(
        root_dir=root_dir,
        csv_file=csv_file,
        split='train',
        transform=train_transform
    )
    
    class_to_idx = train_dataset.class_to_idx
    
    # 创建验证集和测试集
    val_dataset = AnimalDataset(
        root_dir=root_dir,
        csv_file=csv_file,
        split='val',
        transform=val_transform,
        class_to_idx=class_to_idx
    )
    
    test_dataset = AnimalDataset(
        root_dir=root_dir,
        csv_file=csv_file,
        split='test',
        transform=val_transform,
        class_to_idx=class_to_idx
    )
    
    # 打印类别分布
    print("\n📊 类别分布:")
    print(f"训练集: {train_dataset.get_class_distribution()}")
    print(f"验证集: {val_dataset.get_class_distribution()}")
    print(f"测试集: {test_dataset.get_class_distribution()}")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader, class_to_idx


if __name__ == "__main__":
    # 测试代码
    from augmentation import get_train_transform, get_val_transform
    
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        root_dir="../../Animals-10",
        csv_file="train_test_val_split.csv",
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=8,
        num_workers=0
    )
    
    print(f"\n✅ DataLoader测试成功！")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 测试一个batch
    images, labels = next(iter(train_loader))
    print(f"\n批次形状: {images.shape}, 标签形状: {labels.shape}")