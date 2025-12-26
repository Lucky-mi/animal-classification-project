"""
错误样本分析模块
负责人: B3
功能: 深度分析被错分的样本，这是实验6的核心 ⭐
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
from collections import defaultdict
from PIL import Image


class ErrorAnalyzer:
    """
    错误样本分析器
    
    功能:
    1. 统计混淆对（哪两个类别最容易混淆）
    2. 保存被错分的样本
    3. 分析错误模式
    4. 生成错误分析报告
    """
    
    def __init__(
        self,
        class_names: List[str],
        save_dir: str = '../outputs/error_analysis'
    ):
        """
        Args:
            class_names: 类别名称列表
            save_dir: 保存目录
        """
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def analyze_confusion_pairs(
        self,
        conf_matrix: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, str, int]]:
        """
        分析最容易混淆的类别对
        
        Args:
            conf_matrix: 混淆矩阵 [num_classes, num_classes]
            top_k: 返回前K个混淆对
        
        Returns:
            confusion_pairs: [(真实类别, 预测类别, 错误数量)]
        """
        num_classes = len(self.class_names)
        confusion_pairs = []
        
        # 遍历非对角线元素
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:  # 跳过对角线
                    count = conf_matrix[i, j]
                    if count > 0:
                        confusion_pairs.append((
                            self.class_names[i],
                            self.class_names[j],
                            int(count)
                        ))
        
        # 按错误数量排序
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return confusion_pairs[:top_k]
    
    def save_misclassified_samples(
        self,
        misclassified: Dict,
        dataset,
        max_samples_per_pair: int = 10
    ):
        """
        保存被错分的样本
        
        Args:
            misclassified: {(true_label, pred_label): [sample_info]}
            dataset: 数据集对象
            max_samples_per_pair: 每对最多保存多少样本
        """
        print(f"\n💾 保存错误样本到: {self.save_dir}")
        
        for (true_idx, pred_idx), samples in misclassified.items():
            true_name = self.class_names[true_idx]
            pred_name = self.class_names[pred_idx]
            
            # 创建子目录
            pair_dir = os.path.join(self.save_dir, f'{true_name}_as_{pred_name}')
            os.makedirs(pair_dir, exist_ok=True)
            
            # 保存样本
            num_saved = min(len(samples), max_samples_per_pair)
            
            for i, sample in enumerate(samples[:num_saved]):
                img_tensor = sample['image']
                confidence = sample['confidence']
                
                # 反归一化
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1)
                
                # 转换为PIL图像
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                
                # 保存
                filename = f'sample_{i+1}_conf_{confidence:.3f}.jpg'
                img_pil.save(os.path.join(pair_dir, filename))
            
            print(f"   {true_name} → {pred_name}: 保存了 {num_saved}/{len(samples)} 个样本")
    
    def visualize_error_samples(
        self,
        misclassified: Dict,
        top_k_pairs: int = 3,
        samples_per_pair: int = 6
    ):
        """
        可视化错误样本
        
        Args:
            misclassified: {(true_label, pred_label): [sample_info]}
            top_k_pairs: 展示前K个混淆对
            samples_per_pair: 每对展示多少样本
        """
        # 按错误数量排序
        sorted_pairs = sorted(
            misclassified.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:top_k_pairs]
        
        for pair_idx, ((true_idx, pred_idx), samples) in enumerate(sorted_pairs):
            true_name = self.class_names[true_idx]
            pred_name = self.class_names[pred_idx]
            
            num_samples = min(len(samples), samples_per_pair)
            
            fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
            if num_samples == 1:
                axes = [axes]
            
            fig.suptitle(
                f'错误样本: {true_name} 被预测为 {pred_name} (共{len(samples)}个)',
                fontsize=14
            )
            
            for i in range(num_samples):
                sample = samples[i]
                img_tensor = sample['image']
                confidence = sample['confidence']
                
                # 反归一化
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1)
                
                img_np = img_tensor.permute(1, 2, 0).numpy()
                
                axes[i].imshow(img_np)
                axes[i].set_title(f'置信度: {confidence:.2%}', fontsize=10)
                axes[i].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f'error_pair_{pair_idx+1}_{true_name}_as_{pred_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 错误样本可视化: {save_path}")
            plt.close()
    
    def generate_report(
        self,
        conf_matrix: np.ndarray,
        misclassified: Dict,
        save_path: str = None
    ):
        """
        生成错误分析报告
        
        Args:
            conf_matrix: 混淆矩阵
            misclassified: 错误样本字典
            save_path: 报告保存路径
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'error_analysis_report.txt')
        
        # 分析混淆对
        top_confusion = self.analyze_confusion_pairs(conf_matrix, top_k=10)
        
        # 统计总错误数
        total_errors = sum(len(samples) for samples in misclassified.values())
        
        # 生成报告
        report = []
        report.append("=" * 70)
        report.append("错误样本分析报告")
        report.append("=" * 70)
        report.append("")
        
        report.append(f"📊 总体统计:")
        report.append(f"   总错误样本数: {total_errors}")
        report.append(f"   错误类型数: {len(misclassified)}")
        report.append("")
        
        report.append(f"🔝 Top-10 最常见的混淆:")
        report.append(f"{'排名':<6} {'真实类别':<12} {'预测为':<12} {'错误数量':<10}")
        report.append("-" * 70)
        
        for rank, (true_cls, pred_cls, count) in enumerate(top_confusion, 1):
            report.append(f"{rank:<6} {true_cls:<12} {pred_cls:<12} {count:<10}")
        
        report.append("")
        report.append("=" * 70)
        report.append("")
        
        report.append("💡 分析建议:")
        report.append("")
        
        # 自动生成分析建议
        if len(top_confusion) > 0:
            top1_true, top1_pred, top1_count = top_confusion[0]
            report.append(f"1. 最严重的混淆: {top1_true} ↔ {top1_pred} ({top1_count}个样本)")
            report.append(f"   建议: 检查这两个类别的视觉相似性，考虑:")
            report.append(f"   - 使用Grad-CAM查看模型关注的区域")
            report.append(f"   - 增加针对性的数据增强")
            report.append(f"   - 收集更多区分性样本")
            report.append("")
        
        # 检查是否有某个类别特别容易被误分
        per_class_errors = defaultdict(int)
        for (true_idx, pred_idx), samples in misclassified.items():
            per_class_errors[true_idx] += len(samples)
        
        if per_class_errors:
            worst_class_idx = max(per_class_errors, key=per_class_errors.get)
            worst_class_name = self.class_names[worst_class_idx]
            worst_count = per_class_errors[worst_class_idx]
            
            report.append(f"2. 最难识别的类别: {worst_class_name} (被误分{worst_count}次)")
            report.append(f"   建议: 这个类别可能是:")
            report.append(f"   - 样本质量较差（模糊、遮挡、光照不足）")
            report.append(f"   - 类内差异大（不同视角、姿态）")
            report.append(f"   - 与其他类别视觉相似")
            report.append("")
        
        report.append("=" * 70)
        
        # 保存报告
        report_text = "\n".join(report)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📄 错误分析报告保存至: {save_path}")
        print("\n" + report_text)


if __name__ == "__main__":
    print("错误分析模块测试需要完整的评估结果")
    print("请在实验脚本中使用")