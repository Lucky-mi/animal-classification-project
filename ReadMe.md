# Animal-10 图像分类项目

**课程**: 模式识别大作业
**任务**: 基于深度学习的动物图像分类
**数据集**: Animal-10 (10类动物，26,179张图像)

---

## 项目概述

本项目对比CNN（ResNet-18, MobileNetV2）与轻量级Transformer（DeiT-Tiny）在Animal-10数据集上的图像分类性能，通过消融实验探究数据增强对不同架构模型的影响差异，并通过Grad-CAM可视化分析CNN模型的关注区域。

### 核心实验

| 实验 | 内容 | 脚本 |
|------|------|------|
| 实验1 | 模型对比 (ResNet-18 vs MobileNetV2 vs DeiT-Tiny) | `experiments/01_model_comparison.py` |
| 实验2 | 数据增强消融 (无增强 → 基础增强 → 基础+MixUp) | `experiments/02_augmentation_ablation.py` |
| 实验2.5 | 类别不平衡处理 (CE → Weighted CE → Focal → Weighted Focal) | `main.py --config configs/exp2.5_imbalance.yaml` |
| 实验3 | Grad-CAM可视化 (仅CNN) | `experiments/03_gradcam.py` |
| 实验4 | 模型效率对比 (参数量/FLOPs/推理速度) | `experiments/04_efficiency.py` |
| 实验5 | t-SNE特征可视化 | `experiments/05-tsne.py` |
| 实验6 | 错误样本深度分析 | `experiments/06_error_analysis.py` |

---

## 项目结构

```
project/
├── data/                    # 数据处理模块 (B1负责)
│   ├── __init__.py
│   ├── dataset.py          # AnimalDataset类，CSV数据加载
│   ├── augmentation.py     # 数据增强 (none/basic/advanced + MixUp)
│   └── loss.py             # 损失函数 (CE/Weighted CE/Focal/Weighted Focal)
│
├── models/                  # 模型模块 (B2负责)
│   ├── __init__.py
│   └── model_factory.py    # get_model() 统一接口
│
├── utils/                   # 工具模块 (B2+B3协作)
│   ├── __init__.py
│   ├── trainer.py          # Trainer类 (B3)
│   ├── evaluator.py        # Evaluator类 (B3)
│   ├── visualizer.py       # Grad-CAM, t-SNE, 混淆矩阵 (B2)
│   └── analyzer.py         # ErrorAnalyzer类 (B3)
│
├── configs/                 # 实验配置文件
│   ├── base.yaml           # 基础配置
│   ├── exp1_baseline.yaml  # ResNet-18基线
│   ├── exp1_mobilenet.yaml # MobileNetV2
│   ├── exp1_deit.yaml      # DeiT-Tiny
│   ├── exp2_no_aug.yaml    # 无增强
│   ├── exp2_augmentation.yaml  # 基础增强
│   ├── exp2_mixup.yaml     # 基础+MixUp
│   └── exp2.5_imbalance.yaml   # 类别不平衡处理
│
├── experiments/             # 实验脚本
│   ├── 01_model_comparison.py      # 实验1: 模型对比
│   ├── 02_augmentation_ablation.py # 实验2: 数据增强消融
│   ├── 03_gradcam.py               # 实验3: Grad-CAM
│   ├── 04_efficiency.py            # 实验4: 效率对比
│   ├── 05-tsne.py                  # 实验5: t-SNE
│   └── 06_error_analysis.py        # 实验6: 错误分析
│
├── main.py                  # 统一训练入口
├── evaluate.py              # 统一评估入口
├── requirements.txt         # 依赖包
└── Readme.md               # 本文件
```

---

## 快速开始

### 1. 环境配置

**创建虚拟环境:**

```bash
cd big_work/project
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

**安装PyTorch (CUDA版本):**

```bash
# 先安装PyTorch CUDA版本 (根据你的CUDA版本选择)
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 验证CUDA是否可用
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"
```

**安装其他依赖:**

```bash
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import timm; print('timm:', timm.__version__)"
```

### 3. 运行实验

**实验1 - 模型对比 (自动运行3个模型):**

```bash
cd experiments
python 01_model_comparison.py
```

**实验1 - 单独运行某个模型:**

```bash
python main.py --config configs/exp1_baseline.yaml      # ResNet-18
python main.py --config configs/exp1_mobilenet.yaml     # MobileNetV2
python main.py --config configs/exp1_deit.yaml          # DeiT-Tiny
```

**实验2 - 数据增强消融:**

```bash
cd experiments
python 02_augmentation_ablation.py
```

**实验2.5 - 类别不平衡处理:**

```bash
# 运行4次实验，每次修改--loss参数
python main.py --config configs/exp2.5_imbalance.yaml --loss ce
python main.py --config configs/exp2.5_imbalance.yaml --loss weighted_ce
python main.py --config configs/exp2.5_imbalance.yaml --loss focal
python main.py --config configs/exp2.5_imbalance.yaml --loss weighted_focal
```

**实验3-6 - 可视化与分析 (需先完成实验1):**

```bash
cd experiments
python 03_gradcam.py           # Grad-CAM可视化
python 04_efficiency.py        # 模型效率对比
python 05-tsne.py              # t-SNE特征可视化
python 06_error_analysis.py    # 错误样本分析
```

### 4. 评估模型

```bash
python evaluate.py --checkpoint ../outputs/exp1_baseline/models/best_model.pth \
                   --config configs/exp1_baseline.yaml \
                   --split test \
                   --analyze_errors
```

### 5. 查看TensorBoard日志

```bash
tensorboard --logdir ../outputs/exp1_baseline/logs
```

---

## 训练参数

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 224×224 |
| 预训练 | ImageNet |
| 优化器 | AdamW |
| 学习率 | 0.001 |
| 权重衰减 | 0.0001 |
| Batch Size | 32 |
| Epochs | 30 |
| 学习率调度 | CosineAnnealingLR |
| Focal Loss γ | 2.0 |

---

## 实验结果预期

### 实验1: 模型对比

| 模型 | 参数量 | FLOPs | 预期准确率 |
|------|--------|-------|-----------|
| ResNet-18 | ~11M | ~1.8G | 92-95% |
| MobileNetV2 | ~3.5M | ~0.3G | 88-92% |
| DeiT-Tiny | ~5.7M | ~1.3G | 85-90% |

### 实验2.5: 类别不平衡处理

| 损失函数 | 预期准确率 |
|----------|-----------|
| CrossEntropy | ~92% (基线) |
| Weighted CE | ~93% |
| Focal Loss | ~93.5% |
| Weighted Focal | ~94% (最佳) |

---

## 分工说明

### 子组A: PPT汇报组 (3人)

| 角色 | 职责 |
|------|------|
| A1 (主讲) | 独立完成4分钟PPT讲解 |
| A2 (答辩-方法) | 回答背景/方法类问题 |
| A3 (答辩-结果) | 回答结果/分析类问题 |

### 子组B: 代码汇报组 (3人)

| 角色 | 职责 | 时长 |
|------|------|------|
| B1 (数据工程师) | 数据模块 (data/) | ~100秒 |
| B2 (模型工程师) | 模型模块 (models/) + 可视化 | ~120秒 |
| B3 (训练工程师) | 训练评估模块 (utils/trainer, evaluator) | ~80秒 |

---

## 常见问题

**Q: 显示 "CUDA不可用" 但我有NVIDIA显卡?**

A: 需要安装CUDA版本的PyTorch:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Q: 训练时显存不足 (OOM)?**

A: 减小batch size:
```bash
python main.py --config configs/exp1_baseline.yaml --batch_size 16
```

**Q: Windows下DataLoader报错?**

A: 代码已自动处理，CPU模式下会设置 `num_workers=0`

**Q: 实验时间太长?**

A:
1. 确保使用GPU训练
2. 减少epochs: `--epochs 10`
3. 只运行关键实验 (1, 2.5, 6)

---

## 参考文献

1. He K, et al. Deep Residual Learning for Image Recognition. CVPR 2016.
2. Sandler M, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.
3. Touvron H, et al. Training data-efficient image transformers. ICML 2021.
4. Lin T, et al. Focal Loss for Dense Object Detection. ICCV 2017.
5. Selvaraju R, et al. Grad-CAM: Visual Explanations from Deep Networks. ICCV 2017.

---

最后更新: 2025-12-26
