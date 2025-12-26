# Animal-10 图像分类项目

**课程**: 模式识别大作业  
**任务**: 基于深度学习的动物图像分类  
**数据集**: Animal-10 (10类动物，26,179张图像)

---

## 项目概述

本项目对比CNN（ResNet-18, MobileNetV2）与轻量级Transformer（DeiT-Tiny）在Animal-10数据集上的图像分类性能，通过消融实验探究数据增强对不同架构模型的影响差异，并通过Grad-CAM可视化分析CNN模型的关注区域。

### 核心亮点

- 实验2.5: 类别不平衡处理（加权Focal Loss）
- 实验6: 错误样本深度分析
- 完整工作流: 数据加载到模型训练到评估可视化

---

## 项目结构

project目录下包含：
- data目录: 数据处理模块（B1负责）
- models目录: 模型模块（B2负责）
- utils目录: 工具模块（B2+B3协作）
- configs目录: 实验配置文件
- experiments目录: 实验脚本
- main.py: 统一训练入口
- evaluate.py: 统一评估入口

---

## 快速开始

### 1. 环境配置

创建虚拟环境（Windows）：

cd big_work/project
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt

创建虚拟环境（Linux/Mac）：

cd big_work/project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 2. 验证安装

python -c "import torch; print('PyTorch安装成功')"

### 3. 运行实验

实验1 - ResNet-18基线：

python main.py --config configs/exp1_baseline.yaml

实验2.5 - 类别不平衡处理（重点）：

python main.py --config configs/exp2.5_imbalance.yaml --loss weighted_focal

实验6 - 错误样本分析（重点）：

python experiments/06_error_analysis.py --checkpoint ../outputs/exp1_baseline/models/best_model.pth

### 4. 评估模型

python evaluate.py --checkpoint ../outputs/exp1_baseline/models/best_model.pth --config configs/exp1_baseline.yaml --split test --analyze_errors

---

## 实验结果预期

### 实验1: 模型对比

ResNet-18: 参数量约11M, FLOPs约1.8G, 预期准确率92-95%
MobileNetV2: 参数量约3.5M, FLOPs约0.3G, 预期准确率88-92%
DeiT-Tiny: 参数量约5.7M, FLOPs约1.3G, 预期准确率85-90%

### 实验2.5: 类别不平衡处理

CrossEntropy: 约92%（基线）
Weighted CE: 约93%
Focal Loss: 约93.5%
Weighted Focal: 约94%（最佳方案）

---

## 分工说明

### 子组A: PPT汇报组（3人）

A1 (主讲): 独立完成4分钟PPT讲解
A2 (答辩-方法): 回答背景/方法类问题
A3 (答辩-结果): 回答结果/分析类问题

### 子组B: 代码汇报组（3人）

B1 (数据工程师): 数据模块汇报约100秒
B2 (模型工程师): 模型模块汇报约120秒
B3 (训练工程师): 训练评估模块汇报约80秒

---

## 常见问题

Q: 训练时显存不足怎么办？
A: 减小batch size，例如：python main.py --config configs/exp1_baseline.yaml --batch_size 16

Q: 如何查看TensorBoard日志？
A: tensorboard --logdir ../outputs/exp1_baseline/logs

Q: 实验时间太长怎么办？
A: 减少epochs或只运行关键实验（1, 2.5, 6）

---

## 参考文献

1. He K, et al. Deep Residual Learning for Image Recognition. CVPR 2016.
2. Sandler M, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.
3. Touvron H, et al. Training data-efficient image transformers. ICML 2021.
4. Lin T, et al. Focal Loss for Dense Object Detection. ICCV 2017.
5. Selvaraju R, et al. Grad-CAM: Visual Explanations from Deep Networks. ICCV 2017.

---

最后更新: 2025-12-26