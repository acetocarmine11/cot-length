#!/bin/bash

# Define the list of test_t values
model_size=(3)

# Loop through the test_t values and run the Python script

for model_size in "${model_size[@]}"; do
    echo "Running train.py with --model_size=${model_size}"
    python3 synthetic/train.py --model_size=$model_size --device='cuda:1' --iter=25000 --T=24 --t=4 --model_type=looped_gpt2 --n_loop=2
done
