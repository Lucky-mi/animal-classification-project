"""
从TensorBoard导出的CSV生成exp2.5对比图
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 切换到project目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)

# CSV文件路径
val_csv = "../outputs/csv.csv"  # 验证准确率
train_csv = "../outputs/csv (1).csv"  # 训练准确率

# 读取数据
val_df = pd.read_csv(val_csv)
train_df = pd.read_csv(train_csv)

# 损失函数名称
loss_names = ['CrossEntropy', 'Weighted CE', 'Focal Loss', 'Weighted Focal']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 每个实验30个epoch
epochs = 30

# 提取各损失函数的数据
val_data = {}
train_data = {}

for i, name in enumerate(loss_names):
    start_idx = i * epochs
    end_idx = (i + 1) * epochs
    val_data[name] = val_df['Value'].iloc[start_idx:end_idx].values
    train_data[name] = train_df['Value'].iloc[start_idx:end_idx].values

# 创建输出目录
output_dir = "../outputs/exp2.5_comparison"
os.makedirs(output_dir, exist_ok=True)

# 图1: 验证准确率对比
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(1, epochs + 1)

for i, name in enumerate(loss_names):
    ax.plot(x, val_data[name], label=name, color=colors[i], linewidth=2)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('验证准确率 (%)', fontsize=12)
ax.set_title('实验2.5: 不同损失函数的验证准确率对比', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, epochs)
ax.set_ylim(70, 100)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'val_accuracy_comparison.png'), dpi=150, bbox_inches='tight')
print(f"保存: {output_dir}/val_accuracy_comparison.png")

# 图2: 训练准确率对比
fig, ax = plt.subplots(figsize=(10, 6))

for i, name in enumerate(loss_names):
    ax.plot(x, train_data[name], label=name, color=colors[i], linewidth=2)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('训练准确率 (%)', fontsize=12)
ax.set_title('实验2.5: 不同损失函数的训练准确率对比', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, epochs)
ax.set_ylim(70, 100)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'train_accuracy_comparison.png'), dpi=150, bbox_inches='tight')
print(f"保存: {output_dir}/train_accuracy_comparison.png")

# 图3: 训练和验证准确率 - 2x2子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, name in enumerate(loss_names):
    ax = axes[i]
    ax.plot(x, train_data[name], label='训练', color=colors[i], linewidth=2)
    ax.plot(x, val_data[name], label='验证', color=colors[i], linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('准确率 (%)', fontsize=10)
    ax.set_title(name, fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, epochs)
    ax.set_ylim(70, 100)

plt.suptitle('实验2.5: 各损失函数的训练-验证准确率曲线', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'train_val_comparison.png'), dpi=150, bbox_inches='tight')
print(f"保存: {output_dir}/train_val_comparison.png")

# 图4: 最终准确率柱状图
fig, ax = plt.subplots(figsize=(10, 6))

final_val = [val_data[name][-1] for name in loss_names]
final_train = [train_data[name][-1] for name in loss_names]

x_pos = np.arange(len(loss_names))
width = 0.35

bars1 = ax.bar(x_pos - width/2, final_train, width, label='训练准确率', color='#4CAF50', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, final_val, width, label='验证准确率', color='#2196F3', alpha=0.8)

ax.set_xlabel('损失函数', fontsize=12)
ax.set_ylabel('准确率 (%)', fontsize=12)
ax.set_title('实验2.5: 最终准确率对比', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(loss_names, fontsize=10)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(90, 100)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'final_accuracy_bar.png'), dpi=150, bbox_inches='tight')
print(f"保存: {output_dir}/final_accuracy_bar.png")

# 打印汇总
print("\n" + "="*60)
print("实验2.5 最终结果汇总")
print("="*60)
print(f"{'损失函数':<20} {'训练准确率':<15} {'验证准确率':<15}")
print("-"*60)
for name in loss_names:
    print(f"{name:<20} {train_data[name][-1]:<15.2f} {val_data[name][-1]:<15.2f}")
print("="*60)

# 找出最佳
best_idx = np.argmax([val_data[name][-1] for name in loss_names])
print(f"\n验证集最佳: {loss_names[best_idx]} ({val_data[loss_names[best_idx]][-1]:.2f}%)")

plt.close('all')
print(f"\n所有图表已保存到: {output_dir}/")
