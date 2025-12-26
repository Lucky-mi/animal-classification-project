"""
训练器模块
负责人: B3
功能: 封装训练循环，支持MixUp、学习率调度、checkpoint保存
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Dict, Tuple
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    训练器类
    
    功能:
    - 训练循环
    - 验证循环
    - 学习率调度
    - Checkpoint保存
    - TensorBoard日志
    - MixUp支持
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        device: str = 'cuda',
        save_dir: str = '../outputs/models',
        use_tensorboard: bool = True,
        log_dir: str = '../outputs/logs',
        use_mixup: bool = False,
        mixup_alpha: float = 1.0
    ):
        """
        Args:
            model: 模型
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 训练设备
            save_dir: 模型保存目录
            use_tensorboard: 是否使用TensorBoard
            log_dir: TensorBoard日志目录
            use_mixup: 是否使用MixUp
            mixup_alpha: MixUp的alpha参数
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.use_mixup = use_mixup
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # TensorBoard
        self.writer = None
        if use_tensorboard:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            print(f"📊 TensorBoard日志: {log_dir}")
        
        # MixUp
        if use_mixup:
            from data.augmentation import MixUpAugmentation
            self.mixup = MixUpAugmentation(alpha=mixup_alpha)
            print(f"✅ 启用MixUp (alpha={mixup_alpha})")
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.current_epoch = 0
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch数
        
        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # MixUp
            if self.use_mixup:
                images, labels_a, labels_b, lam = self.mixup(images, labels)
                
                # 前向传播
                outputs = self.model(images)
                
                # MixUp损失
                loss = lam * self.criterion(outputs, labels_a) + \
                       (1 - lam) * self.criterion(outputs, labels_b)
                
                # 准确率（使用原始标签）
                _, predicted = outputs.max(1)
                correct += (lam * predicted.eq(labels_a).sum().item() + \
                           (1 - lam) * predicted.eq(labels_b).sum().item())
            
            else:
                # 标准训练
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # 准确率
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
            
            total += labels.size(0)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            
            # 更新进度条
            avg_loss = running_loss / (batch_idx + 1)
            avg_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = running_loss / len(train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        验证
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                running_loss += loss.item()
        
        avg_loss = running_loss / len(val_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_best_only: bool = True,
        save_frequency: int = 5
    ):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_best_only: 是否只保存最佳模型
            save_frequency: 保存频率（每N个epoch）
        """
        print(f"\n{'='*60}")
        print(f"🚀 开始训练")
        print(f"   总epoch数: {num_epochs}")
        print(f"   训练批次: {len(train_loader)}")
        print(f"   验证批次: {len(val_loader)}")
        print(f"   设备: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            # 打印结果
            print(f"\n📊 Epoch {epoch}/{num_epochs} 结果:")
            print(f"   训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"   验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"   学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth', is_best=True)
                print(f"   ⭐ 新的最佳验证准确率: {val_acc:.2f}%")
            
            # 定期保存
            if not save_best_only and epoch % save_frequency == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth')
            
            print()
        
        print(f"{'='*60}")
        print(f"✅ 训练完成！")
        print(f"   最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        保存检查点
        
        Args:
            filename: 文件名
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"   💾 保存最佳模型: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        加载检查点
        
        Args:
            filepath: 检查点文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        print(f"✅ 加载检查点: {filepath}")
        print(f"   Epoch: {self.current_epoch}, 最佳验证准确率: {self.best_val_acc:.2f}%")


if __name__ == "__main__":
    # 测试训练器
    print("训练器模块可以独立运行，但需要完整的数据和模型")
    print("请在main.py中使用")