# When More is Less: Understanding Chain-of-Thought Length in LLMs

Official code repository for the paper "When More is Less: Understanding Chain-of-Thought Length in LLMs". 

## Overview

This repository contains two main components:

1. **Synthetic Experiments**: Training small transformer models on arithmetic and dynamic programming tasks to study optimal CoT length in controlled settings
2. **Real-world Analysis**: Analyzing CoT length patterns in state-of-the-art LLMs on MATH500 and WinoGrande datasets（codes can adapt to more datasets）

The synthetic training component is inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).

## Project Structure

```
cot-length/
├── synthetic/                 # Synthetic experiments
│   ├── dataset/              # Dataset generation
│   │   ├── gen_arith_data.py    # Arithmetic dataset generator
│   │   ├── gen_dp_data.py       # DP dataset generator
│   │   └── gen_*_test_data.py   # Test data generators
│   ├── model/                # Model architectures
│   │   ├── vanilla_gpt2.py      # Standard GPT-2 implementation
│   │   └── looped_gpt2.py       # Looped transformer variant
│   ├── scripts/              # Training/evaluation scripts
│   │   ├── run_train.sh         # Training script
│   │   └── run_eval.sh          # Evaluation script
│   ├── train.py              # Main training script
│   ├── eval.py               # Main evaluation script
│   └── tokenizor.py          # Tokenization utilities
├── real/                     # Real-world analysis
│   ├── run_math500.py           # MATH500 sample generation
│   ├── run_winogrande.py        # WinoGrande sample generation
│   ├── eval_*_cot_length.py     # CoT length analysis
│   └── eval_*_task_difficulty.py # Task difficulty analysis
└── README.md
```


## Part 1: Synthetic Experiments

The synthetic experiments train small transformer models to understand optimal CoT length patterns on two algorithmic tasks:

### Available Datasets

1. **Arithmetic Dataset**: Addition problems with step-by-step solutions.
2. **Dynamic Programming Dataset**: Maximum Path Sum in a Number Triangle problem with bottom-up dp solutions.

### Quick Start

#### 1. Generate Dataset
Choose one of the two available datasets:

```bash
# Generate arithmetic dataset
python3 -m synthetic.dataset.gen_arith_data

# OR generate dynamic programming dataset  
python3 -m synthetic.dataset.gen_dp_data
```

#### 2. Generate Test Data
```bash
# Generate test data for the chosen dataset
python3 -m synthetic.dataset.gen_arith_test_data
# OR
python3 -m synthetic.dataset.gen_dp_test_data
```

#### 3. Train Model
Use the training script to train a transformer model:

```bash
# Basic training command
python3 synthetic/train.py --model_size=6 --device='cuda' --iter=25000 --T=80 --t=12

# Or use the provided script
bash synthetic/scripts/run_train.sh
```

**Training Parameters:**
- `--model_size`: Model size parameter (controls model dimensions)
- `--device`: Training device ('cuda', 'mps', or 'cpu')
- `--iter`: Number of training iterations
- `--T`: Maximum sequence length during training
- `--t`: Target CoT length during training

#### 4. Evaluate Model
After training, evaluate the model across different CoT lengths:

```bash
# Basic evaluation
python3 synthetic/eval.py --test_t=3 --test_T=32 --model_size=6 --t=12 --T=80 

# Or use the provided script for comprehensive evaluation
bash synthetic/scripts/run_eval.sh
```

**Evaluation Parameters:**
- `--test_t`: CoT length to evaluate
- `--test_T`: Maximum sequence length during evaluation
- `--model_size`: Model size (must match training)
- `--device`: Evaluation device

#### 5. View Sample Data
To inspect the generated datasets:

```bash
python3 -m synthetic.tokenizor
```

## Part 2: Real-world Analysis

The real-world component analyzes CoT length patterns in production LLMs on established benchmarks.

### Supported Datasets

1. **MATH500**: Mathematical reasoning problems
2. **WinoGrande**: Commonsense reasoning tasks

### Quick Start

#### 1. Generate Samples

**For MATH500:**
```bash
python3 real/run_math500.py --data math500.jsonl --out outputs --samples 30
```

**For WinoGrande:**
```bash
python3 real/run_winogrande.py --data winogrande_xs --out outputs --samples 30
```

**Common Parameters:**
- `--data`: Input dataset path/name
- `--out`: Output directory for results
- `--samples`: Number of samples to generate per question
- `--model`: Model name (default: qwen models)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max_tokens`: Maximum tokens per completion (default: 1024)
- `--max_retries`: Maximum API retry attempts (default: 5)
- `--debug`: Enable debug logging

#### 2. Analyze Results

After generating samples, analyze CoT length patterns:

**Length Analysis:**
```bash
# Analyze CoT length vs accuracy patterns
python3 real/eval_math500_cot_length.py --data_dir outputs/math500 --output_dir results
python3 real/eval_winogrande_cot_length.py --data_dir outputs/winogrande --output_dir results
```

**Task Difficulty Analysis:**
```bash
# Analyze optimal CoT length vs task difficulty correlation
python3 real/eval_math500_task_difficulty.py --data_dir outputs/math500 --output_dir difficulty_results
python3 real/eval_winogrande_task_difficulty.py --data_dir outputs/winogrande --output_dir difficulty_results
```

**Analysis Parameters:**
- `--data_dir`: Directory containing model outputs
- `--output_dir`: Directory to save analysis results
- `--model`: Specific model to analyze (optional, analyzes all if not specified)
- `--no-filter`: Skip filtering questions with all correct/incorrect answers


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Synthetic training component inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Real-world datasets: MATH500, WinoGrande, MMLU, GPQA benchmarks
