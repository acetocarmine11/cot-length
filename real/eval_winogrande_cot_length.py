#!/usr/bin/env python3
"""
Evaluation script: Calculate COT length and correctness for each model
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from collections import defaultdict


def calculate_cot_length(sample: str) -> float:
    """
    Calculate COT length by replacing \\n\\n with \\n, splitting by \\n, then dividing by 5
    """
    # Replace \\n\\n with \\n
    processed_sample = sample.replace('\n\n', '\n')
    
    # Split by \\n
    lines = processed_sample.split('\n')
    
    # Calculate length and divide by 5
    cot_length = len(lines) // 5
    

    
    return cot_length


def check_correctness(sample: str, answer: str) -> bool:
    """
    Check if sample is correct by finding content in \\boxed{} and comparing with answer
    """
    # Find content in \\boxed{}
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, sample)
    
    if not boxed_matches:
        return False
    
    # Get content from last \\boxed{}
    sample_answer = boxed_matches[-1].strip()
    
    # Clean answer by removing extra whitespace
    clean_answer = answer.strip()
    
    # Compare answers
    return sample_answer == clean_answer


def process_model_data(model_dir: Path, filter_all_same: bool = True) -> Dict[str, Any]:
    """
    Process data for a single model
    
    Args:
        model_dir: Model directory path
        filter_all_same: Whether to filter questions where all samples are correct or incorrect
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
    
    # Get all JSON files
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
                # Calculate COT length
                cot_length = calculate_cot_length(sample)
                
                # Check correctness
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
            
            # Calculate problem-level statistics
            problem_result['correct_count'] = problem_correct
            problem_result['total_samples'] = len(samples)
            problem_result['accuracy'] = problem_correct / len(samples) if samples else 0.0
            problem_result['avg_cot_length'] = sum(problem_cot_lengths) / len(problem_cot_lengths) if problem_cot_lengths else 0.0
            
            # Check if filtering is needed (all samples correct or incorrect)
            should_filter = False
            if filter_all_same and samples:
                if problem_correct == 0 or problem_correct == len(samples):
                    should_filter = True
                    results['filtered_problems'].append(problem_result)
            
            if not should_filter:
                results['problems'].append(problem_result)
                
                # Calculate length accuracy stats only for unfiltered problems
                for sample_result in problem_result['samples']:
                    cot_length = sample_result['cot_length']
                    is_correct = sample_result['is_correct']
                    
                    # Statistics for each length
                    length_key = round(cot_length, 1)  # Round to 1 decimal place
                    results['length_accuracy_stats'][length_key]['total'] += 1
                    if is_correct:
                        results['length_accuracy_stats'][length_key]['correct'] += 1
                
                results['total_correct'] += problem_correct
                results['total_samples'] += len(samples)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Calculate model-level statistics
    if results['total_samples'] > 0:
        results['overall_accuracy'] = results['total_correct'] / results['total_samples']
        all_cot_lengths = []
        for problem in results['problems']:
            all_cot_lengths.extend([s['cot_length'] for s in problem['samples']])
        results['avg_cot_length'] = sum(all_cot_lengths) / len(all_cot_lengths) if all_cot_lengths else 0.0
    
    # Add filtering statistics
    results['total_problems'] = len(results['problems']) + len(results['filtered_problems'])
    results['filtered_count'] = len(results['filtered_problems'])
    results['kept_count'] = len(results['problems'])
    
    return results




def print_length_accuracy_stats(results: Dict[str, Any]):
    """
    Print accuracy statistics for each length
    """
    print(f"\n{results['model_name']} - COT Length Accuracy Stats:")
    print("-" * 60)
    print(f"{'Length':<8} {'Correct':<8} {'Total':<8} {'Accuracy':<10}")
    print("-" * 60)
    
    length_stats = results['length_accuracy_stats']
    for length in sorted(length_stats.keys()):
        stats = length_stats[length]
        if stats['total'] >= 10:  # Only show lengths with sample count >= 10
            accuracy = stats['correct'] / stats['total']
            print(f"{length:<8.1f} {stats['correct']:<8} {stats['total']:<8} {accuracy:<10.4f}")


def print_filter_stats(results: Dict[str, Any]):
    """
    Print filtering statistics
    """
    print(f"\n{results['model_name']} - Filter Stats:")
    print("-" * 40)
    print(f"Total problems: {results['total_problems']}")
    print(f"Kept problems: {results['kept_count']}")
    print(f"Filtered problems: {results['filtered_count']}")
    print(f"Filter ratio: {results['filtered_count']/results['total_problems']:.3f}")
    print(f"Keep ratio: {results['kept_count']/results['total_problems']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate COT length and correctness')
    parser.add_argument('--data_dir', type=str, default='outputs/math500.jsonl',
                       help='Data directory path')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                       help='Output directory path')
    parser.add_argument('--model', type=str, default=None,
                       help='Specify model name, process all models if not specified')
    parser.add_argument('--no-filter', action='store_true',
                       help='Do not filter problems with all correct or incorrect answers')
    
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
        results = process_model_data(model_dir, filter_all_same=not args.no_filter)
        all_results[model_dir.name] = results
        
        # Save individual model results
        model_output_file = output_dir / f"{model_dir.name}_results.json"
        with open(model_output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print model-level statistics
        print(f"Model: {model_dir.name}")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Correct samples: {results['total_correct']}")
        print(f"  Overall accuracy: {results.get('overall_accuracy', 0):.4f}")
        print(f"  Average COT length: {results.get('avg_cot_length', 0):.4f}")
        
        # Print filter statistics
        print_filter_stats(results)
        
        # Print length accuracy statistics
        print_length_accuracy_stats(results)
        

        print()
    
    # Save all results
    all_results_file = output_dir / "all_models_results.json"
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Generate summary report
    summary_file = output_dir / "summary_report.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Model Evaluation Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"Model: {model_name}\n")
            f.write(f"  Total samples: {results['total_samples']}\n")
            f.write(f"  Correct samples: {results['total_correct']}\n")
            f.write(f"  Overall accuracy: {results.get('overall_accuracy', 0):.4f}\n")
            f.write(f"  Average COT length: {results.get('avg_cot_length', 0):.4f}\n")
            f.write("\n")
    
    print(f"Evaluation complete! Results saved in: {output_dir}")


if __name__ == "__main__":
    main() 