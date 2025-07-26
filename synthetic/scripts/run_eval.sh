#!/bin/bash

# Define the list of test_t values
test_values=(1 2 3 4 5 6 7 8)
model_size=(8 9 10)

test_Ts=(8 16 24 32 40 48 56 64)


for model_size in "${model_size[@]}"; do
    for test_T in "${test_Ts[@]}"; do
        for test_t in "${test_values[@]}"; do
            echo "Running eval.py with --test_t=${test_t} --model_size=${model_size}"
            python3 eval.py --test_t=$test_t --model_size=$model_size --T=64 --test_T=$test_T --device=0 --caption="large_model_64"
        done
    done
done

# # Define the list of test_t values
# test_values=(1 2 3 4 5 6 7 8)
# model_size=(9 8 7)

# test_Ts=(8 16 24 32 40 48 56 64)


# for model_size in "${model_size[@]}"; do
#     for test_T in "${test_Ts[@]}"; do
#         for test_t in "${test_values[@]}"; do
#             echo "Running eval.py with --test_t=${test_t} --model_size=${model_size}"
#             python3 eval.py --test_t=$test_t --model_size=$model_size --T=128 --test_T=$test_T --device=0 --caption="large_model"
#         done
#     done
# done

#  128 120 112 104 96 88 80 72 64 