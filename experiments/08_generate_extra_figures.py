"""
额外可视化图表生成脚本
生成数据增强消融和损失函数对比的详细图表

运行方式: python experiments/08_generate_extra_figures.py
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd

# 设置中文字体和学术风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# 学术配色方案
COLORS = {
    'no_aug': '#95a5a6',       # 灰色
    'basic_aug': '#3498db',    # 蓝色
    'mixup': '#27ae60',        # 绿色
    'ce': '#3498db',           # 蓝色
    'weighted_ce': '#9b59b6',  # 紫色
    'focal': '#e74c3c',        # 红色
    'weighted_focal': '#f39c12', # 橙色
}


def create_output_dir():
    """创建输出目录"""
    output_dir = Path("../outputs/extra_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def fig_augmentation_ablation(output_dir):
    """数据增强消融实验对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 数据
    strategies = ['No Augmentation', 'Basic Augmentation', 'Basic + MixUp']
    val_acc = [93.15, 94.22, 94.38]
    test_acc = [92.73, 93.41, 93.56]

    x = np.arange(len(strategies))
    width = 0.35

    # 左图：验证/测试准确率对比
    bars1 = axes[0].bar(x - width/2, val_acc, width, label='Validation',
                        color=[COLORS['no_aug'], COLORS['basic_aug'], COLORS['mixup']], alpha=0.8)
    bars2 = axes[0].bar(x + width/2, test_acc, width, label='Test',
                        color=[COLORS['no_aug'], COLORS['basic_aug'], COLORS['mixup']], alpha=0.5, hatch='///')

    # 添加数值标签
    for bar, val in zip(bars1, val_acc):
        axes[0].annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, test_acc):
        axes[0].annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Data Augmentation Ablation Study', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['No Aug', 'Basic Aug', 'MixUp'], fontsize=11)
    axes[0].legend(loc='lower right')
    axes[0].set_ylim(90, 96)
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.5)

    # 右图：相对基线提升
    baseline = 93.41  # Basic Aug test acc
    improvements = [test_acc[0] - baseline, 0, test_acc[2] - baseline]
    colors_imp = ['#e74c3c' if x < 0 else '#27ae60' for x in improvements]

    bars = axes[1].bar(strategies, improvements, color=colors_imp, alpha=0.8, edgecolor='white', linewidth=2)

    for bar, val in zip(bars, improvements):
        y_pos = bar.get_height() + 0.03 if val >= 0 else bar.get_height() - 0.08
        axes[1].annotate(f'{val:+.2f}%', xy=(bar.get_x() + bar.get_width()/2, y_pos),
                        ha='center', fontsize=11, fontweight='bold')

    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_ylabel('Relative to Baseline (%)', fontsize=12)
    axes[1].set_title('Improvement over Basic Augmentation', fontsize=14, fontweight='bold')
    axes[1].set_xticklabels(['No Aug', 'Basic Aug\n(Baseline)', 'MixUp'], fontsize=10)
    axes[1].set_ylim(-1, 0.5)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_augmentation_ablation.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig_augmentation_ablation.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Data Augmentation Ablation figure generated")


def fig_loss_function_comparison(output_dir):
    """损失函数对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 数据
    losses = ['CrossEntropy', 'Weighted CE', 'Focal Loss', 'Weighted Focal']
    val_acc = [94.22, 94.03, 94.61, 94.30]
    test_acc = [93.41, 93.26, 93.68, 93.49]
    colors = [COLORS['ce'], COLORS['weighted_ce'], COLORS['focal'], COLORS['weighted_focal']]

    # 图1：验证/测试准确率
    x = np.arange(len(losses))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, val_acc, width, label='Validation', color=colors, alpha=0.8)
    bars2 = axes[0].bar(x + width/2, test_acc, width, label='Test', color=colors, alpha=0.5, hatch='///')

    for bar, val in zip(bars1, val_acc):
        axes[0].annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, test_acc):
        axes[0].annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Loss Function Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['CE', 'W-CE', 'Focal', 'W-Focal'], fontsize=10)
    axes[0].legend(loc='lower right')
    axes[0].set_ylim(92, 96)
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.5)

    # 标注最佳
    best_idx = np.argmax(val_acc)
    axes[0].annotate('Best', xy=(best_idx - width/2, val_acc[best_idx] + 0.15),
                    fontsize=10, ha='center', color='green', fontweight='bold')

    # 图2：少数类准确率对比
    minority_classes = ['elephant', 'cat', 'sheep']
    ce_acc = [93.84, 88.69, 85.71]
    wf_acc = [94.52, 89.88, 87.91]

    x2 = np.arange(len(minority_classes))
    bars1 = axes[1].bar(x2 - width/2, ce_acc, width, label='CrossEntropy', color=COLORS['ce'], alpha=0.8)
    bars2 = axes[1].bar(x2 + width/2, wf_acc, width, label='Weighted Focal', color=COLORS['weighted_focal'], alpha=0.8)

    # 添加提升标注
    for i, (ce, wf) in enumerate(zip(ce_acc, wf_acc)):
        improvement = wf - ce
        axes[1].annotate(f'+{improvement:.1f}%', xy=(i + width/2, wf + 0.5),
                        fontsize=10, ha='center', color='green', fontweight='bold')

    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Minority Class Improvement', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(minority_classes, fontsize=11)
    axes[1].legend(loc='lower right')
    axes[1].set_ylim(82, 98)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.5)

    # 图3：相对基线提升
    baseline_test = 93.41
    improvements = [0, test_acc[1] - baseline_test, test_acc[2] - baseline_test, test_acc[3] - baseline_test]
    colors_imp = ['#3498db'] + ['#e74c3c' if x < 0 else '#27ae60' for x in improvements[1:]]

    bars = axes[2].bar(losses, improvements, color=colors_imp, alpha=0.8, edgecolor='white', linewidth=2)

    for bar, val in zip(bars, improvements):
        y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.06
        axes[2].annotate(f'{val:+.2f}%', xy=(bar.get_x() + bar.get_width()/2, y_pos),
                        ha='center', fontsize=10, fontweight='bold')

    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[2].set_ylabel('Relative to CE Baseline (%)', fontsize=12)
    axes[2].set_title('Test Accuracy Improvement', fontsize=14, fontweight='bold')
    axes[2].set_xticklabels(['CE\n(Baseline)', 'W-CE', 'Focal', 'W-Focal'], fontsize=9)
    axes[2].set_ylim(-0.25, 0.4)
    axes[2].yaxis.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_loss_function_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig_loss_function_comparison.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Loss Function Comparison figure generated")


def fig_class_accuracy_heatmap(output_dir):
    """各类别各模型/策略准确率热力图"""
    fig, ax = plt.subplots(figsize=(12, 8))

    classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
               'elephant', 'horse', 'sheep', 'spider', 'squirrel']

    # 数据矩阵
    data = np.array([
        [89.15, 92.92, 91.04, 74.53],   # butterfly
        [88.69, 92.26, 89.88, 63.69],   # cat
        [94.86, 96.78, 95.18, 84.89],   # chicken
        [88.83, 90.96, 90.43, 78.19],   # cow
        [95.28, 92.81, 92.61, 85.83],   # dog
        [93.84, 92.47, 94.52, 95.21],   # elephant
        [93.92, 95.82, 93.92, 74.52],   # horse
        [85.71, 89.01, 87.91, 72.53],   # sheep
        [98.76, 98.76, 97.93, 91.93],   # spider
        [92.51, 93.58, 94.65, 85.03],   # squirrel
    ])

    models = ['ResNet-18', 'MobileNetV2', 'ResNet+W-Focal', 'DeiT-Tiny']

    # 绘制热力图
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=60, vmax=100)

    # 设置标签
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(models, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)

    # 添加数值标注
    for i in range(len(classes)):
        for j in range(len(models)):
            text_color = 'white' if data[i, j] < 80 else 'black'
            ax.text(j, i, f'{data[i, j]:.1f}%', ha='center', va='center',
                   fontsize=10, color=text_color, fontweight='bold')

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va='bottom', fontsize=12)

    ax.set_title('Per-Class Accuracy Comparison Across Models', fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_class_accuracy_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig_class_accuracy_heatmap.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Class Accuracy Heatmap figure generated")


def fig_training_progress(output_dir):
    """训练过程详细对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = np.arange(1, 31)

    # ResNet-18 数据
    resnet_train_loss = [0.6705, 0.4450, 0.3557, 0.2983, 0.2728, 0.2244, 0.1979, 0.1781, 0.1550, 0.1376,
                         0.1088, 0.1020, 0.0806, 0.0662, 0.0628, 0.0475, 0.0431, 0.0261, 0.0230, 0.0226,
                         0.0162, 0.0127, 0.0097, 0.0077, 0.0055, 0.0041, 0.0041, 0.0042, 0.0038, 0.0034]
    resnet_val_loss = [0.5672, 0.5947, 0.4756, 0.4623, 0.3596, 0.4319, 0.3694, 0.3638, 0.3294, 0.2772,
                       0.3782, 0.3542, 0.3858, 0.3697, 0.3433, 0.3329, 0.3112, 0.3676, 0.3503, 0.3065,
                       0.2828, 0.3382, 0.3186, 0.3087, 0.2879, 0.2965, 0.2818, 0.2827, 0.2773, 0.2852]
    resnet_val_acc = [81.18, 80.26, 84.32, 85.35, 88.60, 86.73, 88.91, 89.59, 89.94, 91.66,
                      88.52, 90.44, 89.90, 90.02, 91.28, 91.70, 92.43, 91.81, 92.46, 93.04,
                      93.53, 92.85, 92.92, 93.80, 93.73, 93.99, 94.19, 94.22, 94.07, 93.96]

    # MobileNetV2 数据
    mobilenet_val_acc = [84.32, 87.11, 89.33, 88.14, 88.45, 88.79, 90.13, 90.97, 90.82, 90.47,
                         90.21, 90.90, 91.12, 91.43, 91.20, 92.88, 93.23, 93.31, 92.46, 93.46,
                         93.46, 94.19, 93.38, 94.38, 94.19, 94.45, 94.72, 94.76, 94.95, 94.76]

    # DeiT 数据
    deit_val_acc = [47.51, 50.92, 57.96, 63.58, 64.96, 66.07, 67.50, 68.17, 70.01, 72.69,
                   73.57, 74.50, 75.78, 76.50, 77.28, 78.00, 79.80, 79.99, 80.64, 81.00,
                   81.30, 81.50, 81.71, 81.98, 82.10, 82.25, 82.71, 83.44, 83.44, 82.79]

    # 学习率曲线
    initial_lr = 0.001
    lr_schedule = [initial_lr * (1 + np.cos(np.pi * i / 30)) / 2 for i in range(30)]

    # 图1: 训练损失
    axes[0, 0].plot(epochs, resnet_train_loss, 'o-', linewidth=2, markersize=4, label='ResNet-18', color='#2E86AB')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Training Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Curve (ResNet-18)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].set_ylim(0, 0.8)

    # 图2: 验证损失
    axes[0, 1].plot(epochs, resnet_val_loss, 'o-', linewidth=2, markersize=4, label='ResNet-18', color='#2E86AB')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Validation Loss', fontsize=12)
    axes[0, 1].set_title('Validation Loss Curve (ResNet-18)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].set_ylim(0.2, 0.7)

    # 图3: 验证准确率对比
    axes[1, 0].plot(epochs, resnet_val_acc, '-', linewidth=2, label='ResNet-18', color='#2E86AB')
    axes[1, 0].plot(epochs, mobilenet_val_acc, '-', linewidth=2, label='MobileNetV2', color='#A23B72')
    axes[1, 0].plot(epochs, deit_val_acc, '-', linewidth=2, label='DeiT-Tiny', color='#F18F01')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1, 0].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 0].set_ylim(45, 100)

    # 添加最佳点标注
    axes[1, 0].axhline(y=94.95, color='#A23B72', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=94.22, color='#2E86AB', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=83.44, color='#F18F01', linestyle='--', alpha=0.5)

    # 图4: 学习率曲线
    axes[1, 1].plot(epochs, lr_schedule, 'g-', linewidth=2)
    axes[1, 1].fill_between(epochs, 0, lr_schedule, alpha=0.2, color='green')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Cosine Annealing Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].set_ylim(0, 0.0012)

    # 添加说明
    axes[1, 1].annotate('Initial LR = 0.001', xy=(1, 0.001), xytext=(5, 0.0011),
                        fontsize=10, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))
    axes[1, 1].annotate('Final LR ≈ 0', xy=(30, 0), xytext=(25, 0.0002),
                        fontsize=10, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_training_progress.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig_training_progress.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Training Progress figure generated")


def fig_model_complexity(output_dir):
    """模型复杂度对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = ['ResNet-18', 'MobileNetV2', 'DeiT-Tiny']
    params = [11.18, 2.24, 5.53]
    flops = [1.82, 0.33, 1.07]
    inference = [5.74, 8.93, 10.80]
    test_acc = [93.41, 94.29, 81.92]
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # 图1: 参数量
    bars = axes[0].bar(models, params, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    axes[0].set_ylabel('Parameters (M)', fontsize=12)
    axes[0].set_title('Model Parameters', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, params):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val}M',
                    ha='center', fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 14)

    # 图2: FLOPs
    bars = axes[1].bar(models, flops, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    axes[1].set_ylabel('FLOPs (G)', fontsize=12)
    axes[1].set_title('Computational Cost (FLOPs)', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, flops):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val}G',
                    ha='center', fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, 2.2)

    # 图3: 效率-准确率散点图
    sizes = [p * 30 for p in params]
    scatter = axes[2].scatter(inference, test_acc, s=sizes, c=colors, alpha=0.7, edgecolors='white', linewidth=2)

    for i, model in enumerate(models):
        axes[2].annotate(f'{model}\n({params[i]}M)', xy=(inference[i], test_acc[i]),
                        xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    axes[2].set_xlabel('Inference Time (ms)', fontsize=12)
    axes[2].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[2].set_title('Efficiency vs Accuracy', fontsize=14, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].set_xlim(4, 12)
    axes[2].set_ylim(78, 96)

    # 添加帕累托前沿
    axes[2].plot([5.74, 8.93], [93.41, 94.29], 'g--', linewidth=2, alpha=0.5, label='Pareto Frontier')
    axes[2].legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_model_complexity.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig_model_complexity.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Model Complexity figure generated")


def fig_experiment_summary(output_dir):
    """实验总结图"""
    fig = plt.figure(figsize=(16, 12))

    # 创建GridSpec布局
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 模型性能对比 (占据左上2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    models = ['ResNet-18', 'MobileNetV2', 'DeiT-Tiny']
    test_acc = [93.41, 94.29, 81.92]
    params = [11.18, 2.24, 5.53]
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, test_acc, width, label='Test Accuracy (%)', color=colors, alpha=0.8)

    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, params, width, label='Parameters (M)', color=colors, alpha=0.4, hatch='///')

    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, color='black')
    ax1_twin.set_ylabel('Parameters (M)', fontsize=12, color='gray')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylim(75, 100)
    ax1_twin.set_ylim(0, 15)

    # 添加数值标签
    for bar, val in zip(bars1, test_acc):
        ax1.annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=12, fontweight='bold')

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    # 2. 数据增强效果 (右上)
    ax2 = fig.add_subplot(gs[0, 2])
    strategies = ['No Aug', 'Basic', 'MixUp']
    aug_acc = [92.73, 93.41, 93.56]
    colors_aug = ['#95a5a6', '#3498db', '#27ae60']

    bars = ax2.bar(strategies, aug_acc, color=colors_aug, alpha=0.8, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Test Acc (%)', fontsize=10)
    ax2.set_title('Data Augmentation', fontsize=12, fontweight='bold')
    ax2.set_ylim(91, 95)
    for bar, val in zip(bars, aug_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}%',
                ha='center', fontsize=9, fontweight='bold')

    # 3. 损失函数效果 (右中)
    ax3 = fig.add_subplot(gs[1, 2])
    losses = ['CE', 'W-CE', 'Focal', 'W-Focal']
    loss_acc = [93.41, 93.26, 93.68, 93.49]
    colors_loss = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12']

    bars = ax3.bar(losses, loss_acc, color=colors_loss, alpha=0.8, edgecolor='white', linewidth=2)
    ax3.set_ylabel('Test Acc (%)', fontsize=10)
    ax3.set_title('Loss Functions', fontsize=12, fontweight='bold')
    ax3.set_ylim(92.5, 94.5)
    for bar, val in zip(bars, loss_acc):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.1f}%',
                ha='center', fontsize=8, fontweight='bold')

    # 4. 关键发现文字框 (下方)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    findings = """
    Key Findings:

    1. Best Model: MobileNetV2 achieves 94.29% test accuracy with only 2.24M parameters (80% fewer than ResNet-18)

    2. CNN vs Transformer: CNN models significantly outperform DeiT-Tiny (12% gap) on this medium-scale dataset

    3. Class Imbalance: Focal Loss improves overall accuracy by 0.27%, with 2.2% improvement on minority classes (sheep)

    4. Data Augmentation: Basic augmentation provides 0.68% improvement; MixUp adds additional 0.15%

    5. Transfer Learning: Pretrained models improve accuracy by ~8-12% compared to training from scratch
    """

    ax4.text(0.5, 0.5, findings, transform=ax4.transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='orange'),
            family='monospace')

    plt.savefig(output_dir / 'fig_experiment_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig_experiment_summary.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Experiment Summary figure generated")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Generating Extra Visualization Figures")
    print("="*60 + "\n")

    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}\n")

    # 生成所有图表
    fig_augmentation_ablation(output_dir)
    fig_loss_function_comparison(output_dir)
    fig_class_accuracy_heatmap(output_dir)
    fig_training_progress(output_dir)
    fig_model_complexity(output_dir)
    fig_experiment_summary(output_dir)

    print("\n" + "="*60)
    print("All extra figures generated successfully!")
    print(f"Check: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
