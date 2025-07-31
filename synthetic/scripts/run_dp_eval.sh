#!/bin/bash

# DP evaluation script
# Iterates through test_T from 8 to 32 and test_t from 1 to 4

echo "Starting DP evaluation with different test_T and test_t parameters..."
echo "============================================================"

# Counter for successful runs
successful_runs=0
total_runs=0

# Iterate through test_T from 8 to 32
for test_T in {8..32}; do
    # Iterate through test_t from 1 to 4
    for test_t in {1..4}; do
        echo ""
        echo "=========================================="
        echo "Running: test_T=$test_T, test_t=$test_t"
        echo "=========================================="
        
        # Run the evaluation command
        python eval.py --dp --model_size 5 --T 32 --t 4 --test_T=$test_T --test_t=$test_t
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "✓ Success: test_T=$test_T, test_t=$test_t"
            ((successful_runs++))
        else
            echo "✗ Failed: test_T=$test_T, test_t=$test_t"
        fi
        
        ((total_runs++))
        echo "Progress: $successful_runs/$total_runs successful runs"
    done
done

echo ""
echo "============================================================"
echo "Evaluation complete!"
echo "Successful runs: $successful_runs/$total_runs"
echo "Failed runs: $((total_runs - successful_runs))"
echo "============================================================" 