#!/usr/bin/env bash
# Local parameter sweep script - runs benchmarks sequentially
# No Slurm required - just runs on your local machine

set -e  # Exit on error

# Configuration
DATA_PATH="../TinyStoriesV2-GPT4-train.npy"
RESULTS_DIR="results"
NSYS_DIR="nsys"
VOCAB=10000
NUM_WARMUPS=1
NUM_TRIALS=5
NUM_STEPS=1
BATCH_SIZE=4

# Run modes - set to 1 to enable
RUN_FORWARD_ONLY=1      # Run forward-only benchmarks
RUN_FORWARD_BACKWARD=1  # Run forward+backward benchmarks

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
mkdir -p "$NSYS_DIR"

echo "=========================================="
echo "Starting Benchmark Sweep"
echo "=========================================="
echo "Results will be saved to: $RESULTS_DIR"
echo "Nsys files will be saved to: $NSYS_DIR"
echo ""

# Define sweeps
MODEL_SIZES=(small medium large xl 2.7B)
CONTEXT_LENGTHS=(128 256 512 1024)

# Calculate total runs based on enabled modes
num_modes=0
[ "$RUN_FORWARD_ONLY" = "1" ] && num_modes=$((num_modes + 1))
[ "$RUN_FORWARD_BACKWARD" = "1" ] && num_modes=$((num_modes + 1))
total_runs=$((${#MODEL_SIZES[@]} * ${#CONTEXT_LENGTHS[@]} * num_modes))
current_run=0

# Sweep over model sizes and context lengths
for model_size in "${MODEL_SIZES[@]}"; do
    # Get model hyperparameters
    read -r d_model d_ff num_layers num_heads <<< "$(get_model_config $model_size)"

    for context_length in "${CONTEXT_LENGTHS[@]}"; do

        # Run forward-only mode if enabled
        if [ "$RUN_FORWARD_ONLY" = "1" ]; then
            current_run=$((current_run + 1))

            echo "----------------------------------------"
            echo "Run $current_run/$total_runs"
            echo "Model: $model_size (d_model=$d_model, layers=$num_layers)"
            echo "Context: $context_length"
            echo "Mode: Forward only"
            echo "----------------------------------------"

            output_file="$RESULTS_DIR/bench_${model_size}_ctx${context_length}_fwd_warmup${NUM_WARMUPS}.json"
            nsys_file="$NSYS_DIR/bench_${model_size}_ctx${context_length}_fwd_warmup${NUM_WARMUPS}"

            # uv run python cs336_systems/benchmarking_script.py \
            uv run nsys profile --trace=cuda,osrt,nvtx --pytorch=autograd-nvtx --python-backtrace=cuda -o "$nsys_file" --python-backtrace=cuda python cs336_systems/benchmarking_script.py \
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
                --forward_only

            echo "✓ Saved to $output_file"
            echo ""
        fi

        # Run forward+backward mode if enabled
        if [ "$RUN_FORWARD_BACKWARD" = "1" ]; then
            current_run=$((current_run + 1))

            echo "----------------------------------------"
            echo "Run $current_run/$total_runs"
            echo "Model: $model_size (d_model=$d_model, layers=$num_layers)"
            echo "Context: $context_length"
            echo "Mode: Forward + Backward"
            echo "----------------------------------------"

            output_file="$RESULTS_DIR/bench_${model_size}_ctx${context_length}_fwd_bwd_warmup${NUM_WARMUPS}.json"
            nsys_file="$NSYS_DIR/bench_${model_size}_ctx${context_length}_fwd_warmup${NUM_WARMUPS}"

            # uv run python cs336_systems/benchmarking_script.py \
            uv run nsys profile --trace=cuda,osrt,nvtx --pytorch=autograd-nvtx --python-backtrace=cuda -o "$nsys_file" --python-backtrace=cuda python cs336_systems/benchmarking_script.py \
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
                --output "$output_file"

            echo "✓ Saved to $output_file"
            echo ""
        fi
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
