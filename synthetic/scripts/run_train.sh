#!/bin/bash

# Define the list of test_t values
model_size=(6)

# Loop through the test_t values and run the Python script

for model_size in "${model_size[@]}"; do
    echo "Running train.py with --model_size=${model_size}"
    python3 synthetic/train.py --model_size=$model_size --device='mps' --iter=25000 --T=80 --t=12
done
