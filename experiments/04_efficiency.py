"""
实验4: 模型效率对比
负责人: B2
功能: 对比不同模型的参数量、FLOPs、推理速度
使用方法:
    python experiments/04_efficiency.py
"""

import os
import sys
sys.path.append('..')

import torch
import time
import numpy as np
from thop import profile
import matplotlib.pyplot as plt

from models import get_model, count_parameters


def measure_inference_time(model, device, num_runs=100):
    """
    测量模型推理时间
    
    Args:
        model: 模型
        device: 设备
        num_runs: 运行次数
    
    Returns:
        avg_time: 平均推理时间(ms)
    """
    model.eval()
    
    # 预热
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 测量
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.time()
            times.append((end - start) * 1000)  # 转换为毫秒
    
    return np.mean(times), np.std(times)


def main():
    print(f"\n{'='*70}")
    print("⚡ 实验4: 模型效率对比")
    print(f"{'='*70}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_names = ['resnet18', 'mobilenet_v2', 'deit_tiny']
    
    results = {
        'model': [],
        'params': [],
        'params_mb': [],
        'flops': [],
        'flops_g': [],
        'inference_time_mean': [],
        'inference_time_std': []
    }
    
    for model_name in model_names:
        print(f"\n📊 分析模型: {model_name}")
        print("-" * 70)
        
        # 创建模型
        model = get_model(
            model_name=model_name,
            num_classes=10,
            pretrained=False,
            dropout=0.3,
            device=device
        )
        
        # 1. 参数量
        total_params, trainable_params = count_parameters(model)
        params_mb = total_params * 4 / (1024 ** 2)  # 假设float32
        
        print(f"参数量: {total_params:,} ({params_mb:.2f} MB)")
        
        # 2. FLOPs
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9
            print(f"FLOPs: {flops_g:.2f} G")
        except Exception as e:
            print(f"⚠️ FLOPs计算失败: {e}")
            flops = 0
            flops_g = 0
        
        # 3. 推理速度
        print(f"测量推理速度 (100次运行)...", end=' ')
        mean_time, std_time = measure_inference_time(model, device, num_runs=100)
        print(f"完成")
        print(f"推理时间: {mean_time:.2f} ± {std_time:.2f} ms")
        
        # 保存结果
        results['model'].append(model_name)
        results['params'].append(total_params)
        results['params_mb'].append(params_mb)
        results['flops'].append(flops)
        results['flops_g'].append(flops_g)
        results['inference_time_mean'].append(mean_time)
        results['inference_time_std'].append(std_time)
        
        # 清理显存
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # 生成对比图
    output_dir = '../outputs/efficiency_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 参数量对比
    axes[0].bar(results['model'], results['params_mb'], color='skyblue')
    axes[0].set_ylabel('Parameters (MB)', fontsize=12)
    axes[0].set_title('Model Size Comparison', fontsize=14)
    axes[0].set_xticklabels(results['model'], rotation=45)
    
    # FLOPs对比
    axes[1].bar(results['model'], results['flops_g'], color='lightcoral')
    axes[1].set_ylabel('FLOPs (G)', fontsize=12)
    axes[1].set_title('Computational Cost Comparison', fontsize=14)
    axes[1].set_xticklabels(results['model'], rotation=45)
    
    # 推理速度对比
    axes[2].bar(results['model'], results['inference_time_mean'], 
                yerr=results['inference_time_std'], color='lightgreen', capsize=5)
    axes[2].set_ylabel('Inference Time (ms)', fontsize=12)
    axes[2].set_title('Inference Speed Comparison', fontsize=14)
    axes[2].set_xticklabels(results['model'], rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'model_efficiency_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 效率对比图保存至: {save_path}")
    plt.close()
    
    # 生成报告
    report_path = os.path.join(output_dir, 'efficiency_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("模型效率对比报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'模型':<15} {'参数量(M)':<12} {'FLOPs(G)':<12} {'推理时间(ms)':<15}\n")
        f.write("-" * 70 + "\n")
        
        for i, model_name in enumerate(results['model']):
            f.write(f"{model_name:<15} "
                   f"{results['params'][i]/1e6:<12.2f} "
                   f"{results['flops_g'][i]:<12.2f} "
                   f"{results['inference_time_mean'][i]:<15.2f}\n")
        
        f.write("\n" + "=" * 70 + "\n\n")
        
        f.write("💡 分析建议:\n\n")
        
        # 找到最快的模型
        fastest_idx = np.argmin(results['inference_time_mean'])
        fastest_model = results['model'][fastest_idx]
        
        f.write(f"1. 速度最快: {fastest_model}\n")
        f.write(f"   推理时间: {results['inference_time_mean'][fastest_idx]:.2f} ms\n")
        f.write(f"   适用场景: 移动端部署、实时应用\n\n")
        
        # 找到参数量最小的模型
        lightest_idx = np.argmin(results['params'])
        lightest_model = results['model'][lightest_idx]
        
        f.write(f"2. 最轻量: {lightest_model}\n")
        f.write(f"   参数量: {results['params_mb'][lightest_idx]:.2f} MB\n")
        f.write(f"   适用场景: 存储受限设备\n\n")
        
        f.write(f"3. 性能权衡:\n")
        f.write(f"   - 如果追求准确率，选择ResNet-18\n")
        f.write(f"   - 如果追求速度，选择MobileNetV2\n")
        f.write(f"   - 如果希望平衡，根据实际需求决定\n")
    
    print(f"💾 效率报告保存至: {report_path}")
    
    print(f"\n{'='*70}")
    print("✅ 效率分析完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # 检查是否安装了thop
    try:
        import thop
    except ImportError:
        print("❌ 请先安装thop库: pip install thop")
        sys.exit(1)
    
    main()