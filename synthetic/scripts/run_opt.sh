#!/bin/bash

# Define the list of test_t values
model_size=(5 6 7)

# Loop through the test_t values and run the Python script

for model_size in "${model_size[@]}"; do
    echo "Running train.py with --model_size=${model_size}"
    python3 train_opt.py --model_size=$model_size --device='cuda:0' --iter=25000 --T=80 --t=12
done

# model_size=(8)

# for model_size in "${model_size[@]}"; do
#     echo "Running train.py with --model_size=${model_size}"
#     python3 train.py --model_size=$model_size --device='cuda:0' --iter=20000 --T=104
# done