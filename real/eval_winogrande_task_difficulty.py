#!/usr/bin/env python3
"""
Task difficulty evaluation script: Analyze correlation between optimal length and accuracy
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
import numpy as np
from collections import defaultdict
from scipy import stats

least_samples = 3


def calculate_cot_length(sample: str) -> float:
    """
    Calculate COT length by replacing \\n\\n with \\n, splitting by \\n, then dividing by 5
    """
    # Replace \\n\\n with \\n
    processed_sample = sample.replace('\n\n', '\n')
    
    # Split by \\n
    lines = processed_sample.split('\n')
    
    cot_length = len(lines) // 5
    
    return cot_length


def check_correctness(sample: str, option1: str, option2: str, ground_truth: str) -> bool:
    """
    Check if sample is correct
    Logic: match option1 and option2, use the last appearing one as answer, compare with correct answer
    Return False if neither option is found
    """
    # Find positions of option1 and option2 in sample
    option1_positions = []
    option2_positions = []
    
    # Use regex to find all matches
    option1_pattern = re.compile(r'\b' + re.escape(option1) + r'\b', re.IGNORECASE)
    option2_pattern = re.compile(r'\b' + re.escape(option2) + r'\b', re.IGNORECASE)
    
    # Find all match positions
    for match in option1_pattern.finditer(sample):
        option1_positions.append(match.start())
    
    for match in option2_pattern.finditer(sample):
        option2_positions.append(match.start())
    
    # Return False if neither option is found
    if not option1_positions and not option2_positions:
        return False
    
    # Find the last appearing option
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
    
    # Determine correct answer
    if ground_truth == "1":
        correct_option = option1
    elif ground_truth == "2":
        correct_option = option2
    else:
        return False
    
    # Compare last appearing option with correct answer
    return last_option == correct_option


def find_optimal_length(samples: List[str], option1: str, option2: str, ground_truth: str) -> Tuple[float, float]:
    """
    Find optimal length and corresponding accuracy
    
    Returns:
        (optimal_length, optimal_accuracy)
    """
    # Group statistics by length
    length_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for sample in samples:
        cot_length = calculate_cot_length(sample)
        is_correct = check_correctness(sample, option1, option2, ground_truth)
        
        # Round to 1 decimal place for grouping
        length_key = round(cot_length, 1)
        length_stats[length_key]['total'] += 1
        if is_correct:
            length_stats[length_key]['correct'] += 1
    
    # Find length group with highest accuracy
    optimal_length = None
    optimal_accuracy = 0.0
    
    for length, stats in length_stats.items():
        if stats['total'] >= least_samples:  # Need at least 3 samples
            accuracy = stats['correct'] / stats['total']
            if accuracy > optimal_accuracy:
                optimal_accuracy = accuracy
                optimal_length = length
    
    return optimal_length, optimal_accuracy


def calculate_question_accuracy(samples: List[str], option1: str, option2: str, ground_truth: str) -> float:
    """
    Calculate question accuracy
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
    Process single model data and analyze task difficulty
    """
    results = {
        'model_name': model_dir.name,
        'questions': [],
        'filtered_questions': [],
        'optimal_lengths': [],
        'accuracies': []
    }
    
    # Get all JSON files
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
            
            # Calculate question accuracy
            question_accuracy = calculate_question_accuracy(samples, option1, option2, ground_truth)
            
            # Filter questions with all correct or all incorrect
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
            
            # Find optimal length
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
    Analyze correlation between optimal length and accuracy
    """
    if len(results['optimal_lengths']) < 2:
        return {'error': 'Insufficient samples'}
    
    # Calculate correlation coefficient
    correlation, p_value = stats.pearsonr(results['optimal_lengths'], results['accuracies'])
    
    return {
        'pearson_correlation': correlation,
        'pearson_p_value': p_value,
        'sample_count': len(results['optimal_lengths'])
    }




def print_analysis_results(results: Dict[str, Any], correlation_results: Dict[str, Any]):
    """
    Print analysis results
    """
    print(f"\n{results['model_name']} - Task Difficulty Analysis Results:")
    print("=" * 60)
    
    print(f"Total questions: {len(results['questions']) + len(results['filtered_questions'])}")
    print(f"Kept questions: {len(results['questions'])}")
    print(f"Filtered questions: {len(results['filtered_questions'])}")
    print(f"Questions with optimal length: {len(results['optimal_lengths'])}")
    
    if 'error' not in correlation_results:
        print(f"\nCorrelation Analysis:")
        print(f"Pearson correlation: {correlation_results['pearson_correlation']:.4f}")
        print(f"Pearson p-value: {correlation_results['pearson_p_value']:.4f}")
        
        # Check significance
        alpha = 0.05
        if correlation_results['pearson_p_value'] < alpha:
            print(f"Pearson correlation is significant (p < {alpha})")
        else:
            print(f"Pearson correlation is not significant (p >= {alpha})")
    
    # Print statistical information
    if results['optimal_lengths']:
        print(f"\nStatistical Information:")
        print(f"Average optimal length: {np.mean(results['optimal_lengths']):.3f}")
        print(f"Optimal length std dev: {np.std(results['optimal_lengths']):.3f}")
        print(f"Average accuracy: {np.mean(results['accuracies']):.3f}")
        print(f"Accuracy std dev: {np.std(results['accuracies']):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze task difficulty and optimal length correlation')
    parser.add_argument('--data_dir', type=str, default='outputs/winogrande_xs',
                       help='Data directory path')
    parser.add_argument('--output_dir', type=str, default='task_difficulty_results',
                       help='Output directory path')
    parser.add_argument('--model', type=str, default=None,
                       help='Specify model name, process all models if not specified')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get all model directories
    if args.model:
        model_dirs = [data_dir / args.model]
    else:
        model_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    all_results = {}
    
    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"Model directory does not exist: {model_dir}")
            continue
            
        print(f"Processing model: {model_dir.name}")
        results = process_model_data(model_dir)
        
        # Analyze correlation
        correlation_results = analyze_correlation(results)
        
        # Merge results
        results['correlation_analysis'] = correlation_results
        all_results[model_dir.name] = results
        
        # Save individual model results
        model_output_file = output_dir / f"{model_dir.name}_winogrande_task_difficulty.json"
        with open(model_output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print analysis results
        print_analysis_results(results, correlation_results)
        

        print()
    
    # Save all results
    all_results_file = output_dir / "all_models_winogrande_task_difficulty.json"
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Generate summary report
    summary_file = output_dir / "winogrande_task_difficulty_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Task Difficulty Analysis Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"Model: {model_name}\n")
            f.write(f"  Kept questions: {len(results['questions'])}\n")
            f.write(f"  Filtered questions: {len(results['filtered_questions'])}\n")
            f.write(f"  Questions with optimal length: {len(results['optimal_lengths'])}\n")
            
            if 'error' not in results['correlation_analysis']:
                corr = results['correlation_analysis']
                f.write(f"  Pearson correlation: {corr['pearson_correlation']:.4f} (p={corr['pearson_p_value']:.4f})\n")
            f.write("\n")
    
    print(f"Analysis complete! Results saved in: {output_dir}")


if __name__ == "__main__":
    main() 