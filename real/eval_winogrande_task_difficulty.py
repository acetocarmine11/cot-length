#!/usr/bin/env python3
"""
Winogrande任务难度评估脚本：分析optimal length和准确率的相关性
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy import stats
import pandas as pd

least_samples = 3


def calculate_cot_length(sample: str) -> float:
    """
    计算COT长度
    方法：将\\n\\n替换成\\n，然后按\\n分割，最后除以5
    """
    # 将\\n\\n替换成\\n
    processed_sample = sample.replace('\n\n', '\n')
    
    # 按\\n分割
    lines = processed_sample.split('\n')
    
    cot_length = len(lines) // 5
    
    return cot_length


def check_correctness(sample: str, option1: str, option2: str, ground_truth: str) -> bool:
    """
    检查sample是否正确
    逻辑：匹配option1和option2，最后一个出现的那个作为答案，再与正确答案比较
    如果都没有就算错
    """
    # 查找option1和option2在sample中的位置
    option1_positions = []
    option2_positions = []
    
    # 使用正则表达式查找所有匹配项
    option1_pattern = re.compile(r'\b' + re.escape(option1) + r'\b', re.IGNORECASE)
    option2_pattern = re.compile(r'\b' + re.escape(option2) + r'\b', re.IGNORECASE)
    
    # 查找所有匹配位置
    for match in option1_pattern.finditer(sample):
        option1_positions.append(match.start())
    
    for match in option2_pattern.finditer(sample):
        option2_positions.append(match.start())
    
    # 如果没有找到任何一个选项，返回False
    if not option1_positions and not option2_positions:
        return False
    
    # 找到最后一个出现的选项
    last_position = -1
    last_option = None
    
    if option1_positions:
        max_pos1 = max(option1_positions)
        if max_pos1 > last_position:
            last_position = max_pos1
            last_option = option1
    
    if option2_positions:
        max_pos2 = max(option2_positions)
        if max_pos2 > last_position:
            last_position = max_pos2
            last_option = option2
    
    # 确定正确答案
    if ground_truth == "1":
        correct_option = option1
    elif ground_truth == "2":
        correct_option = option2
    else:
        return False
    
    # 比较最后出现的选项与正确答案
    return last_option == correct_option


def find_optimal_length(samples: List[str], option1: str, option2: str, ground_truth: str) -> Tuple[float, float]:
    """
    找到最优长度和对应的准确率
    
    Returns:
        (optimal_length, optimal_accuracy)
    """
    # 按长度分组统计
    length_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for sample in samples:
        cot_length = calculate_cot_length(sample)
        is_correct = check_correctness(sample, option1, option2, ground_truth)
        
        # 四舍五入到小数点后1位进行分组
        length_key = round(cot_length, 1)
        length_stats[length_key]['total'] += 1
        if is_correct:
            length_stats[length_key]['correct'] += 1
    
    # 找到准确率最高的长度组
    optimal_length = None
    optimal_accuracy = 0.0
    
    for length, stats in length_stats.items():
        if stats['total'] >= least_samples:  # 至少需要2个样本
            accuracy = stats['correct'] / stats['total']
            if accuracy > optimal_accuracy:
                optimal_accuracy = accuracy
                optimal_length = length
    
    return optimal_length, optimal_accuracy


def calculate_question_accuracy(samples: List[str], option1: str, option2: str, ground_truth: str) -> float:
    """
    计算问题的准确率
    """
    correct_count = 0
    total_count = 0
    
    for sample in samples:
        is_correct = check_correctness(sample, option1, option2, ground_truth)
        if is_correct:
            correct_count += 1
        total_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0


def process_model_data(model_dir: Path) -> Dict[str, Any]:
    """
    处理单个模型的数据，分析任务难度
    """
    results = {
        'model_name': model_dir.name,
        'questions': [],
        'filtered_questions': [],
        'optimal_lengths': [],
        'accuracies': []
    }
    
    # 获取所有JSON文件
    json_files = list(model_dir.glob('*.json'))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            problem_id = json_file.stem
            item = data.get('item', {})
            sentence = item.get('sentence', '')
            option1 = item.get('option1', '')
            option2 = item.get('option2', '')
            ground_truth = data.get('ground_truth', '')
            samples = data.get('samples', [])
            
            if not samples:
                continue
            
            # 计算问题准确率
            question_accuracy = calculate_question_accuracy(samples, option1, option2, ground_truth)
            
            # 过滤全对或全错的问题
            if question_accuracy == 0.0 or question_accuracy == 1.0:
                results['filtered_questions'].append({
                    'problem_id': problem_id,
                    'sentence': sentence,
                    'option1': option1,
                    'option2': option2,
                    'ground_truth': ground_truth,
                    'accuracy': question_accuracy,
                    'sample_count': len(samples)
                })
                continue
            
            # 找到最优长度
            optimal_length, optimal_accuracy = find_optimal_length(samples, option1, option2, ground_truth)
            
            question_result = {
                'problem_id': problem_id,
                'sentence': sentence,
                'option1': option1,
                'option2': option2,
                'ground_truth': ground_truth,
                'accuracy': question_accuracy,
                'optimal_length': optimal_length,
                'optimal_accuracy': optimal_accuracy,
                'sample_count': len(samples)
            }
            
            results['questions'].append(question_result)
            
            if optimal_length is not None:
                results['optimal_lengths'].append(optimal_length)
                results['accuracies'].append(question_accuracy)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return results


def analyze_correlation(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析optimal length和准确率的相关性
    """
    if len(results['optimal_lengths']) < 2:
        return {'error': '样本数量不足'}
    
    # 计算相关系数
    correlation, p_value = stats.pearsonr(results['optimal_lengths'], results['accuracies'])
    
    # 计算Spearman相关系数
    spearman_corr, spearman_p = stats.spearmanr(results['optimal_lengths'], results['accuracies'])
    
    return {
        'pearson_correlation': correlation,
        'pearson_p_value': p_value,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'sample_count': len(results['optimal_lengths'])
    }


def plot_correlation(results: Dict[str, Any], output_dir: Path):
    """
    绘制optimal length和准确率的相关性图
    """
    if len(results['optimal_lengths']) < 2:
        print("样本数量不足，无法绘制相关性图")
        return
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 主散点图
    scatter = ax1.scatter(results['optimal_lengths'], results['accuracies'], 
                          alpha=0.6, s=60, c=results['accuracies'], cmap='viridis')
    ax1.set_xlabel('Optimal Length')
    ax1.set_ylabel('Question Accuracy')
    ax1.set_title(f'{results["model_name"]} - Optimal Length vs Accuracy')
    ax1.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(results['optimal_lengths'], results['accuracies'], 1)
    p = np.poly1d(z)
    ax1.plot(results['optimal_lengths'], p(results['optimal_lengths']), "r--", alpha=0.8, linewidth=2)
    
    # 添加相关系数文本
    if 'correlation_analysis' in results and 'error' not in results['correlation_analysis']:
        corr = results['correlation_analysis']
        text = f'Pearson r = {corr["pearson_correlation"]:.3f}\np = {corr["pearson_p_value"]:.3f}'
        ax1.text(0.05, 0.95, text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Accuracy')
    
    # 最优长度分布直方图
    ax2.hist(results['optimal_lengths'], bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
    ax2.set_xlabel('Optimal Length')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{results["model_name"]} - Optimal Length Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 准确率分布直方图
    ax3.hist(results['accuracies'], bins=20, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax3.set_xlabel('Question Accuracy')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'{results["model_name"]} - Accuracy Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 箱线图
    # 将数据按长度分组
    length_groups = defaultdict(list)
    for length, acc in zip(results['optimal_lengths'], results['accuracies']):
        length_groups[round(length, 1)].append(acc)
    
    if length_groups:
        group_labels = sorted(length_groups.keys())
        group_data = [length_groups[label] for label in group_labels]
        
        bp = ax4.boxplot(group_data, labels=[f'{label:.1f}' for label in group_labels])
        ax4.set_xlabel('Optimal Length')
        ax4.set_ylabel('Accuracy')
        ax4.set_title(f'{results["model_name"]} - Accuracy by Length Group')
        ax4.grid(True, alpha=0.3)
        
        # 设置x轴标签旋转
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图片
    plot_file = output_dir / f"{results['model_name']}_correlation.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"相关性图已保存到: {plot_file}")
    
    # 创建额外的详细散点图
    create_detailed_scatter_plots(results, output_dir)


def create_detailed_scatter_plots(results: Dict[str, Any], output_dir: Path):
    """
    创建更详细的散点图
    """
    if len(results['optimal_lengths']) < 2:
        return
    
    # 创建大尺寸的详细散点图
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 使用不同颜色和大小来表示准确率
    scatter = ax.scatter(results['optimal_lengths'], results['accuracies'], 
                        s=[100 + acc*200 for acc in results['accuracies']],  # 大小基于准确率
                        c=results['accuracies'], 
                        cmap='plasma', 
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5)
    
    ax.set_xlabel('Optimal Length', fontsize=12)
    ax.set_ylabel('Question Accuracy', fontsize=12)
    ax.set_title(f'{results["model_name"]} - Detailed Scatter Plot\nOptimal Length vs Question Accuracy', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(results['optimal_lengths'], results['accuracies'], 1)
    p = np.poly1d(z)
    ax.plot(results['optimal_lengths'], p(results['optimal_lengths']), 
            "r--", alpha=0.8, linewidth=3, label=f'Trend line (slope={z[0]:.3f})')
    
    # 添加相关系数信息
    if 'correlation_analysis' in results and 'error' not in results['correlation_analysis']:
        corr = results['correlation_analysis']
        text = f'Pearson r = {corr["pearson_correlation"]:.3f}\np = {corr["pearson_p_value"]:.3f}\nSpearman ρ = {corr["spearman_correlation"]:.3f}'
        ax.text(0.02, 0.98, text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
                verticalalignment='top', fontsize=11, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Accuracy', fontsize=12)
    
    # 添加图例
    ax.legend()
    
    # 设置坐标轴范围
    ax.set_xlim(min(results['optimal_lengths']) - 0.5, max(results['optimal_lengths']) + 0.5)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # 保存详细散点图
    detailed_plot_file = output_dir / f"{results['model_name']}_detailed_scatter.png"
    plt.savefig(detailed_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"详细散点图已保存到: {detailed_plot_file}")
    
    # 创建按准确率分组的散点图
    create_grouped_scatter_plots(results, output_dir)


def create_grouped_scatter_plots(results: Dict[str, Any], output_dir: Path):
    """
    创建按准确率分组的散点图
    """
    if len(results['optimal_lengths']) < 2:
        return
    
    # 按准确率分组
    low_acc = []  # 准确率 < 0.3
    mid_acc = []  # 准确率 0.3-0.7
    high_acc = []  # 准确率 > 0.7
    
    for length, acc in zip(results['optimal_lengths'], results['accuracies']):
        if acc < 0.3:
            low_acc.append((length, acc))
        elif acc < 0.7:
            mid_acc.append((length, acc))
        else:
            high_acc.append((length, acc))
    
    # 创建分组散点图
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = ['red', 'orange', 'green']
    labels = ['Low Acc (<0.3)', 'Mid Acc (0.3-0.7)', 'High Acc (>0.7)']
    groups = [low_acc, mid_acc, high_acc]
    
    for i, (group, color, label) in enumerate(zip(groups, colors, labels)):
        if group:
            lengths, accs = zip(*group)
            ax.scatter(lengths, accs, c=color, s=80, alpha=0.7, 
                      label=f'{label} (n={len(group)})', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Optimal Length', fontsize=12)
    ax.set_ylabel('Question Accuracy', fontsize=12)
    ax.set_title(f'{results["model_name"]} - Scatter Plot by Accuracy Groups', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 添加趋势线
    z = np.polyfit(results['optimal_lengths'], results['accuracies'], 1)
    p = np.poly1d(z)
    ax.plot(results['optimal_lengths'], p(results['optimal_lengths']), 
            "k--", alpha=0.8, linewidth=2, label='Overall trend')
    
    plt.tight_layout()
    
    # 保存分组散点图
    grouped_plot_file = output_dir / f"{results['model_name']}_grouped_scatter.png"
    plt.savefig(grouped_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"分组散点图已保存到: {grouped_plot_file}")


def print_analysis_results(results: Dict[str, Any], correlation_results: Dict[str, Any]):
    """
    打印分析结果
    """
    print(f"\n{results['model_name']} - Winogrande任务难度分析结果:")
    print("=" * 60)
    
    print(f"总问题数: {len(results['questions']) + len(results['filtered_questions'])}")
    print(f"保留问题数: {len(results['questions'])}")
    print(f"过滤问题数: {len(results['filtered_questions'])}")
    print(f"有最优长度的问题数: {len(results['optimal_lengths'])}")
    
    if 'error' not in correlation_results:
        print(f"\n相关性分析:")
        print(f"Pearson相关系数: {correlation_results['pearson_correlation']:.4f}")
        print(f"Pearson p值: {correlation_results['pearson_p_value']:.4f}")
        print(f"Spearman相关系数: {correlation_results['spearman_correlation']:.4f}")
        print(f"Spearman p值: {correlation_results['spearman_p_value']:.4f}")
        
        # 判断显著性
        alpha = 0.05
        if correlation_results['pearson_p_value'] < alpha:
            print(f"Pearson相关性显著 (p < {alpha})")
        else:
            print(f"Pearson相关性不显著 (p >= {alpha})")
        
        if correlation_results['spearman_p_value'] < alpha:
            print(f"Spearman相关性显著 (p < {alpha})")
        else:
            print(f"Spearman相关性不显著 (p >= {alpha})")
    
    # 打印一些统计信息
    if results['optimal_lengths']:
        print(f"\n统计信息:")
        print(f"平均最优长度: {np.mean(results['optimal_lengths']):.3f}")
        print(f"最优长度标准差: {np.std(results['optimal_lengths']):.3f}")
        print(f"平均准确率: {np.mean(results['accuracies']):.3f}")
        print(f"准确率标准差: {np.std(results['accuracies']):.3f}")


def main():
    parser = argparse.ArgumentParser(description='分析Winogrande任务难度和最优长度相关性')
    parser.add_argument('--data_dir', type=str, default='outputs/winogrande_xs',
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='task_difficulty_results',
                       help='输出目录路径')
    parser.add_argument('--model', type=str, default=None,
                       help='指定模型名称，如果不指定则处理所有模型')
    parser.add_argument('--plot', action='store_true',
                       help='是否生成相关性图')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有模型目录
    if args.model:
        model_dirs = [data_dir / args.model]
    else:
        model_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    all_results = {}
    
    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"模型目录不存在: {model_dir}")
            continue
            
        print(f"处理模型: {model_dir.name}")
        results = process_model_data(model_dir)
        
        # 分析相关性
        correlation_results = analyze_correlation(results)
        
        # 合并结果
        results['correlation_analysis'] = correlation_results
        all_results[model_dir.name] = results
        
        # 保存单个模型的结果
        model_output_file = output_dir / f"{model_dir.name}_winogrande_task_difficulty.json"
        with open(model_output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印分析结果
        print_analysis_results(results, correlation_results)
        
        # 生成相关性图
        if args.plot:
            plot_correlation(results, output_dir)
        
        print()
    
    # 保存所有结果
    all_results_file = output_dir / "all_models_winogrande_task_difficulty.json"
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 生成汇总报告
    summary_file = output_dir / "winogrande_task_difficulty_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Winogrande任务难度分析汇总报告\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"模型: {model_name}\n")
            f.write(f"  保留问题数: {len(results['questions'])}\n")
            f.write(f"  过滤问题数: {len(results['filtered_questions'])}\n")
            f.write(f"  有最优长度的问题数: {len(results['optimal_lengths'])}\n")
            
            if 'error' not in results['correlation_analysis']:
                corr = results['correlation_analysis']
                f.write(f"  Pearson相关系数: {corr['pearson_correlation']:.4f} (p={corr['pearson_p_value']:.4f})\n")
                f.write(f"  Spearman相关系数: {corr['spearman_correlation']:.4f} (p={corr['spearman_p_value']:.4f})\n")
            f.write("\n")
    
    print(f"分析完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main() 