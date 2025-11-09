#!/usr/bin/env bash
# Local parameter sweep script - runs benchmarks sequentially
# No Slurm required - just runs on your local machine

set -e  # Exit on error

# Configuration
DATA_PATH="../assignment1-basics/data/TinyStoriesV2-GPT4-train.npy"
RESULTS_DIR="results"
VOCAB=10000
NUM_WARMUPS=5
NUM_TRIALS=10
NUM_STEPS=1
BATCH_SIZE=4
# FORWARD_ONLY=1  # Uncomment to enable forward-only mode

# Function to get model config
get_model_config() {
    case $1 in
        # small)   echo "256 1024 4 4" ;;
        # medium)  echo "256 1024 4 4" ;;
        # large)   echo "256 1024 4 4" ;;
        # xl)      echo "256 1024 4 4" ;;
        # 2.7B)    echo "256 1024 4 4" ;;
        # *)       echo ""; return 1 ;;
        small)   echo "768 3072 12 12" ;;
        medium)  echo "1024 4096 24 16" ;;
        large)   echo "1280 5120 36 20" ;;
        xl)      echo "1600 6400 48 25" ;;
        2.7B)    echo "2560 10240 32 32" ;;
        *)       echo ""; return 1 ;;
    esac
}

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Starting Benchmark Sweep"
echo "=========================================="
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Define sweeps
MODEL_SIZES=(small medium large xl 2.7B)
CONTEXT_LENGTHS=(128 256)

total_runs=$((${#MODEL_SIZES[@]} * ${#CONTEXT_LENGTHS[@]}))
current_run=0

# Sweep over model sizes and context lengths
for model_size in "${MODEL_SIZES[@]}"; do
    # Get model hyperparameters
    read -r d_model d_ff num_layers num_heads <<< "$(get_model_config $model_size)"

    for context_length in "${CONTEXT_LENGTHS[@]}"; do
        current_run=$((current_run + 1))

        echo "----------------------------------------"
        echo "Run $current_run/$total_runs"
        echo "Model: $model_size (d_model=$d_model, layers=$num_layers)"
        echo "Context: $context_length"
        echo "----------------------------------------"

        output_file="$RESULTS_DIR/bench_${model_size}_ctx${context_length}.json"

        uv run python cs336_systems/benchmarking_script.py \
            --model_size "$model_size" \
            --vocab "$VOCAB" \
            --context_length "$context_length" \
            --d_model "$d_model" \
            --d_ff "$d_ff" \
            --num_layers "$num_layers" \
            --num_heads "$num_heads" \
            --batch_size "$BATCH_SIZE" \
            --num_steps "$NUM_STEPS" \
            --num_warmups "$NUM_WARMUPS" \
            --num_trials "$NUM_TRIALS" \
            --data_path "$DATA_PATH" \
            --output "$output_file" \
            ${FORWARD_ONLY:+--forward_only}

        echo "✓ Saved to $output_file"
        echo ""
    done
done

echo "=========================================="
echo "Sweep Complete!"
echo "=========================================="
echo "Total runs: $total_runs"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "To generate summary tables, run:"
echo "  uv run python cs336_systems/aggregate_results.py --results-dir $RESULTS_DIR --output $RESULTS_DIR/summary.md"
echo ""
