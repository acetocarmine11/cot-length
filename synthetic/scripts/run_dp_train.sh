#!/bin/bash

# Define the list of test_t values
model_size=(6)
dataset="dp"  # Change this to "arithmetic" or "dp"

# Set T and t based on dataset
if [ "$dataset" = "dp" ]; then
    T=20
    t=10
elif [ "$dataset" = "arithmetic" ]; then
    T=80
    t=12
else
    echo "Unknown dataset: $dataset"
    exit 1
fi

# Loop through the test_t values and run the Python script

for model_size in "${model_size[@]}"; do
    echo "Running train.py with --model_size=${model_size} --dataset=${dataset} --T=${T} --t=${t}"
    python3 synthetic/train.py --model_size=$model_size --dataset=$dataset --device='mps' --iter=25000 --T=$T --t=$t
done
