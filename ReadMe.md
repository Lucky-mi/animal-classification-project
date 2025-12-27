# Animal-10 图像分类项目

**课程**: 模式识别大作业
**任务**: 基于深度学习的动物图像分类
**数据集**: Animal-10 (10类动物，26,179张图像)

---

## 项目概述

本项目对比CNN（ResNet-18, MobileNetV2）与轻量级Transformer（DeiT-Tiny）在Animal-10数据集上的图像分类性能，通过消融实验探究数据增强和损失函数对模型性能的影响，并通过Grad-CAM和t-SNE进行可视化分析。

### 核心发现

| 发现 | 详情 |
|------|------|
| 最佳模型 | MobileNetV2 (94.29% 准确率，2.24M 参数) |
| CNN vs Transformer | CNN显著优于Transformer (差距约12%) |
| 类别不平衡处理 | Focal Loss最有效 (+0.27%) |
| 数据增强 | 基础增强+MixUp有效提升泛化能力 |

---

## 实验结果

### 模型性能对比

| 模型 | 验证准确率 | 测试准确率 | 参数量 | FLOPs | 推理时间 |
|------|-----------|-----------|--------|-------|---------|
| ResNet-18 | 94.22% | 93.41% | 11.18M | 1.82G | 5.74ms |
| **MobileNetV2** | **94.95%** | **94.29%** | **2.24M** | **0.33G** | 8.93ms |
| DeiT-Tiny | 83.44% | 81.92% | 5.53M | 1.07G | 10.80ms |

### 数据增强消融

| 增强策略 | 测试准确率 | 相对基线 |
|----------|-----------|----------|
| 无增强 | 92.73% | -0.68% |
| 基础增强 | 93.41% | 基线 |
| 基础+MixUp | 93.56% | +0.15% |

### 损失函数对比

| 损失函数 | 验证准确率 | 测试准确率 |
|----------|-----------|-----------|
| CrossEntropy | 94.22% | 93.41% |
| Weighted CE | 94.03% | 93.26% |
| **Focal Loss** | **94.61%** | **93.68%** |
| Weighted Focal | 94.30% | 93.49% |

### 各类别准确率 (ResNet-18)

| 类别 | 准确率 | 类别 | 准确率 |
|------|--------|------|--------|
| spider | **98.76%** | horse | 93.92% |
| dog | 95.28% | elephant | 93.84% |
| chicken | 94.86% | squirrel | 92.51% |
| butterfly | 89.15% | cat | 88.69% |
| cow | 88.83% | sheep | **85.71%** |

---

## 项目结构

```
project/
├── data/                    # 数据处理模块
│   ├── dataset.py          # AnimalDataset类
│   ├── augmentation.py     # 数据增强策略
│   └── loss.py             # 损失函数 (CE/Focal/Weighted)
│
├── models/                  # 模型模块
│   └── model_factory.py    # 统一模型接口
│
├── utils/                   # 工具模块
│   ├── trainer.py          # 训练器
│   ├── evaluator.py        # 评估器
│   ├── visualizer.py       # Grad-CAM, t-SNE, 混淆矩阵
│   └── analyzer.py         # 错误分析器
│
├── configs/                 # 实验配置文件 (13个)
│   ├── base.yaml           # 基础配置
│   ├── exp1_*.yaml         # 模型对比实验
│   ├── exp2_*.yaml         # 数据增强消融
│   └── exp2.5_*.yaml       # 损失函数对比
│
├── experiments/             # 实验脚本 (10个)
│   ├── 01_model_comparison.py
│   ├── 02_augmentation_ablation.py
│   ├── 02.5_imbalance_ablation.py
│   ├── 03_gradcam.py
│   ├── 04_efficiency.py
│   ├── 05-tsne.py
│   ├── 06_error_analysis.py
│   ├── 07_generate_ppt_figures.py
│   ├── 07_summary_report.py
│   └── 08_generate_extra_figures.py
│
├── main.py                  # 统一训练入口
├── evaluate.py              # 统一评估入口
└── requirements.txt         # 依赖包
```

### 输出目录结构

```
outputs/
├── exp1_baseline/          # ResNet-18实验
├── exp1_mobilenet/         # MobileNetV2实验
├── exp1_deit/              # DeiT-Tiny实验
├── exp2_no_aug/            # 无增强实验
├── exp2_augmentation/      # 基础增强实验
├── exp2_mixup/             # MixUp实验
├── exp2.5_*/               # 损失函数对比 (4个)
├── gradcam_visualization/  # Grad-CAM热力图
├── tsne_visualization/     # t-SNE特征图
├── efficiency_analysis/    # 效率分析报告
├── ppt_figures/            # PPT图表 (8张)
├── extra_figures/          # 额外分析图表 (6张)
└── summary_report/         # 汇总报告
```

---

## 快速开始

### 1. 环境配置

```bash
cd project
python -m venv venv
venv\Scripts\activate  # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. 运行实验

```bash
# 模型对比实验
python main.py --config configs/exp1_baseline.yaml      # ResNet-18
python main.py --config configs/exp1_mobilenet.yaml     # MobileNetV2
python main.py --config configs/exp1_deit.yaml          # DeiT-Tiny

# 数据增强消融
python main.py --config configs/exp2_no_aug.yaml
python main.py --config configs/exp2_augmentation.yaml
python main.py --config configs/exp2_mixup.yaml

# 损失函数对比
python main.py --config configs/exp2.5_ce.yaml
python main.py --config configs/exp2.5_focal.yaml
python main.py --config configs/exp2.5_weighted_focal.yaml
```

### 3. 评估与可视化

```bash
# 评估模型
python evaluate.py --checkpoint ../outputs/exp1_baseline/models/best_model.pth \
                   --config configs/exp1_baseline.yaml --split test --analyze_errors

# 可视化分析
cd experiments
python 03_gradcam.py           # Grad-CAM
python 04_efficiency.py        # 效率分析
python 05-tsne.py              # t-SNE
python 07_generate_ppt_figures.py  # 生成PPT图表
python 08_generate_extra_figures.py # 生成额外图表
```

### 4. 查看TensorBoard

```bash
tensorboard --logdir ../outputs/exp1_baseline/logs
```

---

## 训练配置

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 224x224 |
| 预训练 | ImageNet |
| 优化器 | AdamW (lr=0.001, wd=0.0001) |
| Batch Size | 32 |
| Epochs | 30 |
| 学习率调度 | CosineAnnealingLR |
| Focal Loss gamma | 2.0 |
| MixUp alpha | 1.0 |

---

## 数据集信息

### Animal-10 数据集

| 属性 | 值 |
|------|-----|
| 总样本数 | 26,179张 |
| 训练集 | 20,938张 (80%) |
| 验证集 | 2,614张 (10%) |
| 测试集 | 2,627张 (10%) |
| 类别数 | 10类 |
| 类别不平衡比 | 3.37:1 |

### 类别分布 (训练集)

| 类别 | 样本数 | 类别 | 样本数 |
|------|--------|------|--------|
| dog | 3,890 | cow | 1,492 |
| spider | 3,856 | squirrel | 1,489 |
| chicken | 2,478 | sheep | 1,456 |
| horse | 2,098 | cat | 1,334 |
| butterfly | 1,689 | elephant | 1,156 |

---

## 可视化图表

### PPT图表 (`outputs/ppt_figures/`)

| 图表 | 文件名 | 内容 |
|------|--------|------|
| Fig.1 | `fig1_model_accuracy_comparison.png` | 三模型准确率对比 |
| Fig.2 | `fig2_model_efficiency.png` | 效率分析散点图 |
| Fig.3 | `fig3_class_accuracy_radar.png` | 类别准确率雷达图 |
| Fig.4 | `fig4_class_distribution.png` | 数据集分布图 |
| Fig.5 | `fig5_training_curves.png` | 训练曲线对比 |
| Fig.6 | `fig6_error_analysis.png` | 错误分析对比 |
| Fig.7 | `fig7_summary_table.png` | 综合对比表格 |
| Fig.8 | `fig8_key_findings.png` | 关键发现信息图 |

### 额外图表 (`outputs/extra_figures/`)

| 图表 | 内容 |
|------|------|
| `fig_augmentation_ablation.png` | 数据增强消融对比 |
| `fig_loss_function_comparison.png` | 损失函数详细对比 |
| `fig_class_accuracy_heatmap.png` | 类别准确率热力图 |
| `fig_training_progress.png` | 训练过程详图 |
| `fig_model_complexity.png` | 模型复杂度对比 |
| `fig_experiment_summary.png` | 实验总结图 |

---

## 常见问题

**Q: CUDA不可用?**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Q: 显存不足 (OOM)?**
```bash
python main.py --config configs/exp1_baseline.yaml --batch_size 16
```

**Q: 加快训练速度?**
```bash
python main.py --config configs/exp1_baseline.yaml --epochs 15
```

---

## 参考文献

1. He K, et al. Deep Residual Learning for Image Recognition. CVPR 2016.
2. Sandler M, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.
3. Touvron H, et al. Training data-efficient image transformers. ICML 2021.
4. Lin T, et al. Focal Loss for Dense Object Detection. ICCV 2017.
5. Selvaraju R, et al. Grad-CAM: Visual Explanations from Deep Networks. ICCV 2017.
6. Zhang H, et al. mixup: Beyond Empirical Risk Minimization. ICLR 2018.

---

**最后更新**: 2025-12-27
