#!/usr/bin/env python3
"""
评估脚本：计算每个模型的COT长度和正确性
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


def calculate_cot_length(sample: str) -> float:
    """
    计算COT长度
    方法：将\\n\\n替换成\\n，然后按\\n分割，最后除以5
    """
    # 将\\n\\n替换成\\n
    processed_sample = sample.replace('\n\n', '\n')
    
    # 按\\n分割
    lines = processed_sample.split('\n')
    
    # 计算长度并除以5
    cot_length = len(lines) // 5
    

    
    return cot_length


def check_correctness(sample: str, answer: str) -> bool:
    """
    检查sample是否正确
    通过查找\\boxed{}中的内容并与answer比较
    """
    # 查找\\boxed{}中的内容
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, sample)
    
    if not boxed_matches:
        return False
    
    # 获取最后一个\\boxed{}中的内容
    sample_answer = boxed_matches[-1].strip()
    
    # 清理answer，移除多余的空白字符
    clean_answer = answer.strip()
    
    # 比较答案
    return sample_answer == clean_answer


def process_model_data(model_dir: Path, filter_all_same: bool = True) -> Dict[str, Any]:
    """
    处理单个模型的数据
    
    Args:
        model_dir: 模型目录路径
        filter_all_same: 是否过滤所有sample全对或全错的question
    """
    results = {
        'model_name': model_dir.name,
        'problems': [],
        'filtered_problems': [],
        'total_correct': 0,
        'total_samples': 0,
        'avg_cot_length': 0.0,
        'length_accuracy_stats': defaultdict(lambda: {'correct': 0, 'total': 0})
    }
    
    # 获取所有JSON文件
    json_files = list(model_dir.glob('*.json'))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            problem_id = json_file.stem
            problem = data.get('problem', '')
            answer = data.get('answer', '')
            samples = data.get('samples', [])
            
            problem_result = {
                'problem_id': problem_id,
                'problem': problem,
                'answer': answer,
                'samples': []
            }
            
            problem_correct = 0
            problem_cot_lengths = []
            
            for i, sample in enumerate(samples):
                # 计算COT长度
                cot_length = calculate_cot_length(sample)
                
                # 检查正确性
                is_correct = check_correctness(sample, answer)
                
                sample_result = {
                    'sample_id': i,
                    'cot_length': cot_length,
                    'is_correct': is_correct,
                    'sample': sample
                }
                
                problem_result['samples'].append(sample_result)
                problem_cot_lengths.append(cot_length)
                
                if is_correct:
                    problem_correct += 1
            
            # 计算问题级别的统计
            problem_result['correct_count'] = problem_correct
            problem_result['total_samples'] = len(samples)
            problem_result['accuracy'] = problem_correct / len(samples) if samples else 0.0
            problem_result['avg_cot_length'] = sum(problem_cot_lengths) / len(problem_cot_lengths) if problem_cot_lengths else 0.0
            
            # 检查是否需要过滤（所有sample全对或全错）
            should_filter = False
            if filter_all_same and samples:
                if problem_correct == 0 or problem_correct == len(samples):
                    should_filter = True
                    results['filtered_problems'].append(problem_result)
                    # print(f"过滤问题 {problem_id}: 准确率 = {problem_result['accuracy']:.3f} ({problem_correct}/{len(samples)})")
            
            if not should_filter:
                results['problems'].append(problem_result)
                
                # 只对未过滤的问题统计长度准确率
                for sample_result in problem_result['samples']:
                    cot_length = sample_result['cot_length']
                    is_correct = sample_result['is_correct']
                    
                    # 统计每个长度的准确率
                    length_key = round(cot_length, 1)  # 四舍五入到小数点后1位
                    results['length_accuracy_stats'][length_key]['total'] += 1
                    if is_correct:
                        results['length_accuracy_stats'][length_key]['correct'] += 1
                
                results['total_correct'] += problem_correct
                results['total_samples'] += len(samples)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # 计算模型级别的统计
    if results['total_samples'] > 0:
        results['overall_accuracy'] = results['total_correct'] / results['total_samples']
        all_cot_lengths = []
        for problem in results['problems']:
            all_cot_lengths.extend([s['cot_length'] for s in problem['samples']])
        results['avg_cot_length'] = sum(all_cot_lengths) / len(all_cot_lengths) if all_cot_lengths else 0.0
    
    # 添加过滤统计
    results['total_problems'] = len(results['problems']) + len(results['filtered_problems'])
    results['filtered_count'] = len(results['filtered_problems'])
    results['kept_count'] = len(results['problems'])
    
    return results


def plot_length_accuracy(results: Dict[str, Any], output_dir: Path):
    """
    绘制COT长度与准确率的柱状图
    """
    length_stats = results['length_accuracy_stats']
    
    if not length_stats:
        print("没有找到长度统计数据")
        return
    
    # 准备数据
    lengths = []
    accuracies = []
    counts = []
    
    for length, stats in sorted(length_stats.items()):
        if stats['total'] > 10:  # 只显示样本数>10的长度
            lengths.append(length)
            accuracy = stats['correct'] / stats['total']
            accuracies.append(accuracy)
            counts.append(stats['total'])
    
    if not lengths:
        print("没有足够样本数的长度区间")
        return
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 准确率柱状图
    bars1 = ax1.bar(lengths, accuracies, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('COT长度')
    ax1.set_ylabel('准确率')
    ax1.set_title(f'{results["model_name"]} - COT长度与准确率关系')
    ax1.grid(True, alpha=0.3)
    
    # 在柱子上添加数值标签
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 样本数量柱状图
    bars2 = ax2.bar(lengths, counts, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_xlabel('COT长度')
    ax2.set_ylabel('样本数量')
    ax2.set_title(f'{results["model_name"]} - COT长度分布')
    ax2.grid(True, alpha=0.3)
    
    # 在柱子上添加数值标签
    for bar, count in zip(bars2, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图片
    plot_file = output_dir / f"{results['model_name']}_length_accuracy.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"柱状图已保存到: {plot_file}")


def print_length_accuracy_stats(results: Dict[str, Any]):
    """
    打印每个长度的准确率统计
    """
    print(f"\n{results['model_name']} - COT长度准确率统计:")
    print("-" * 60)
    print(f"{'长度':<8} {'正确数':<8} {'总数':<8} {'准确率':<10}")
    print("-" * 60)
    
    length_stats = results['length_accuracy_stats']
    for length in sorted(length_stats.keys()):
        stats = length_stats[length]
        if stats['total'] >= 10:  # 只显示样本数>=10的
            accuracy = stats['correct'] / stats['total']
            print(f"{length:<8.1f} {stats['correct']:<8} {stats['total']:<8} {accuracy:<10.4f}")


def print_filter_stats(results: Dict[str, Any]):
    """
    打印过滤统计信息
    """
    print(f"\n{results['model_name']} - 过滤统计:")
    print("-" * 40)
    print(f"总问题数: {results['total_problems']}")
    print(f"保留问题数: {results['kept_count']}")
    print(f"过滤问题数: {results['filtered_count']}")
    print(f"过滤比例: {results['filtered_count']/results['total_problems']:.3f}")
    print(f"保留比例: {results['kept_count']/results['total_problems']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='评估COT长度和正确性')
    parser.add_argument('--data_dir', type=str, default='outputs/math500.jsonl',
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                       help='输出目录路径')
    parser.add_argument('--model', type=str, default=None,
                       help='指定模型名称，如果不指定则处理所有模型')
    parser.add_argument('--plot', action='store_true',
                       help='是否生成柱状图')
    parser.add_argument('--no-filter', action='store_true',
                       help='不过滤全对或全错的问题')
    
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
        results = process_model_data(model_dir, filter_all_same=not args.no_filter)
        all_results[model_dir.name] = results
        
        # 保存单个模型的结果
        model_output_file = output_dir / f"{model_dir.name}_results.json"
        with open(model_output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印模型级别的统计
        print(f"模型: {model_dir.name}")
        print(f"  总样本数: {results['total_samples']}")
        print(f"  正确样本数: {results['total_correct']}")
        print(f"  总体准确率: {results.get('overall_accuracy', 0):.4f}")
        print(f"  平均COT长度: {results.get('avg_cot_length', 0):.4f}")
        
        # 打印过滤统计
        print_filter_stats(results)
        
        # 打印长度准确率统计
        print_length_accuracy_stats(results)
        
        # 生成柱状图
        if args.plot:
            plot_length_accuracy(results, output_dir)
        
        print()
    
    # 保存所有结果
    all_results_file = output_dir / "all_models_results.json"
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 生成汇总报告
    summary_file = output_dir / "summary_report.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("模型评估汇总报告\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"模型: {model_name}\n")
            f.write(f"  总样本数: {results['total_samples']}\n")
            f.write(f"  正确样本数: {results['total_correct']}\n")
            f.write(f"  总体准确率: {results.get('overall_accuracy', 0):.4f}\n")
            f.write(f"  平均COT长度: {results.get('avg_cot_length', 0):.4f}\n")
            f.write("\n")
    
    print(f"评估完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main() 