"""
评估器模块
负责人: B3
功能: 模型评估，计算多种指标（Accuracy, Macro F1, Per-class Accuracy）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm


class Evaluator:
    """
    评估器类
    
    功能:
    - 计算整体准确率
    - 计算Macro F1
    - 计算每个类别的精确率/召回率/F1
    - 生成混淆矩阵
    - 收集错误样本
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        class_names: List[str] = None
    ):
        """
        Args:
            model: 要评估的模型
            device: 设备
            class_names: 类别名称列表
        """
        self.model = model
        self.device = device
        self.class_names = class_names
    
    def evaluate(
        self,
        data_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict:
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            return_predictions: 是否返回预测结果
        
        Returns:
            results: 评估结果字典
        """
        self.model.eval()
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc='Evaluating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # 计算指标
        results = self.calculate_metrics(all_labels, all_preds, all_probs)
        
        # 如果需要返回预测结果
        if return_predictions:
            results['predictions'] = all_preds
            results['probabilities'] = all_probs
            results['labels'] = all_labels
        
        return results
    
    def calculate_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray
    ) -> Dict:
        """
        计算评估指标
        
        Args:
            labels: 真实标签 [N]
            preds: 预测标签 [N]
            probs: 预测概率 [N, num_classes]
        
        Returns:
            metrics: 指标字典
        """
        # 整体准确率
        accuracy = accuracy_score(labels, preds) * 100
        
        # 每个类别的精确率、召回率、F1
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        
        # Macro平均
        macro_precision = precision.mean() * 100
        macro_recall = recall.mean() * 100
        macro_f1 = f1.mean() * 100
        
        # 混淆矩阵
        conf_matrix = confusion_matrix(labels, preds)
        
        # 每个类别的准确率
        per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1) * 100
        
        # 组织结果
        results = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'confusion_matrix': conf_matrix,
            'per_class_metrics': {
                'precision': precision * 100,
                'recall': recall * 100,
                'f1': f1 * 100,
                'accuracy': per_class_acc,
                'support': support
            }
        }
        
        return results
    
    def print_results(self, results: Dict):
        """
        打印评估结果
        
        Args:
            results: 评估结果字典
        """
        print(f"\n{'='*70}")
        print("📊 评估结果")
        print(f"{'='*70}\n")
        
        # 整体指标
        print("🎯 整体指标:")
        print(f"   准确率 (Accuracy): {results['accuracy']:.2f}%")
        print(f"   Macro 精确率: {results['macro_precision']:.2f}%")
        print(f"   Macro 召回率: {results['macro_recall']:.2f}%")
        print(f"   Macro F1: {results['macro_f1']:.2f}%")
        
        # 每个类别的指标
        print(f"\n📋 各类别指标:")
        print(f"{'类别':<12} {'精确率':>8} {'召回率':>8} {'F1':>8} {'准确率':>8} {'样本数':>8}")
        print("-" * 70)
        
        per_class = results['per_class_metrics']
        
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<12} "
                  f"{per_class['precision'][i]:>7.2f}% "
                  f"{per_class['recall'][i]:>7.2f}% "
                  f"{per_class['f1'][i]:>7.2f}% "
                  f"{per_class['accuracy'][i]:>7.2f}% "
                  f"{per_class['support'][i]:>8}")
        
        print(f"{'='*70}\n")
    
    def get_misclassified_samples(
        self,
        data_loader: DataLoader
    ) -> Dict:
        """
        获取所有被错分的样本
        
        Returns:
            misclassified: 字典，键为(true_label, pred_label)，值为样本索引列表
        """
        self.model.eval()
        
        misclassified = {}
        sample_idx = 0
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc='Finding errors'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                # 找到错误样本
                mask = predicted != labels
                
                for i in range(len(labels)):
                    if mask[i]:
                        true_label = labels[i].item()
                        pred_label = predicted[i].item()
                        key = (true_label, pred_label)
                        
                        if key not in misclassified:
                            misclassified[key] = []
                        
                        misclassified[key].append({
                            'index': sample_idx + i,
                            'image': images[i].cpu(),
                            'true_label': true_label,
                            'pred_label': pred_label,
                            'confidence': torch.softmax(outputs[i], dim=0)[pred_label].item()
                        })
                
                sample_idx += len(labels)
        
        return misclassified


if __name__ == "__main__":
    print("评估器模块测试需要完整的模型和数据")
    print("请在实验脚本中使用")