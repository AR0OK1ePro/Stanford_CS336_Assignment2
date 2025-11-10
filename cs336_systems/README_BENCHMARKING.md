# Benchmarking Workflow Guide

## Overview

This directory contains scripts for benchmarking transformer models with parameter sweeps running locally.

## Files

- `benchmarking_script.py` - Main benchmarking script with command-line arguments
- `sweep_local.sh` - Local shell script for running parameter sweeps
- `aggregate_results.py` - Aggregates JSON results and generates markdown tables
- `run_benchmarking.sh` - Simple single-run example

## Quick Start

### 1. Single Benchmark Run

All hyperparameters must be specified:

```bash
uv run python cs336_systems/benchmarking_script.py \
    --model_size medium \
    --vocab 10000 \
    --context_length 256 \
    --d_model 512 \
    --d_ff 2048 \
    --num_layers 8 \
    --num_heads 8 \
    --batch_size 8 \
    --num_steps 10 \
    --num_warmups 2 \
    --num_trials 5 \
    --data_path ../assignment1-basics/data/TinyStoriesV2-GPT4-train.npy \
    --output results.json
```

**Forward-only mode** (skip backward pass):
```bash
# Add --forward_only flag to benchmark only the forward pass
uv run python cs336_systems/benchmarking_script.py [args...] --forward_only
```

Note: All arguments are required except `--forward_only`. The `--model_size` is just a label for the output JSON.

### 2. Local Parameter Sweep

**Run sweep:**

```bash
bash cs336_systems/sweep_local.sh
```

This will benchmark all combinations of:
- Model sizes: small, medium, large, xl, 2_7B
- Context lengths: 128, 256, 512

**Customize the sweep:**

Edit `sweep_local.sh` to modify:

```bash
# Model configurations (function get_model_config, lines 18-32)
get_model_config() {
    case $1 in
        small)   echo "768 3072 12 12" ;;      # d_model d_ff num_layers num_heads
        medium)  echo "1024 4096 24 16" ;;
        large)   echo "1280 5120 36 20" ;;
        xl)      echo "1600 6400 48 25" ;;
        2.7B)    echo "2560 10240 32 32" ;;
    esac
}

# Which sizes to sweep (line ~44)
MODEL_SIZES=(small medium large)

# Which context lengths to sweep (line ~45)
CONTEXT_LENGTHS=(128 256 512)

# Fixed parameters (lines 10-15)
VOCAB=10000
NUM_WARMUPS=5
NUM_TRIALS=10
NUM_STEPS=1
BATCH_SIZE=4
# FORWARD_ONLY=1  # Uncomment to enable forward-only mode
```

The script runs sequentially and saves each result to `results/bench_{model}_{context}.json`

### 3. Aggregate Results

After the sweep completes:

```bash
uv run python cs336_systems/aggregate_results.py \
    --results-dir results \
    --output benchmark_summary.md \
    --csv benchmark_results.csv
```

This generates:
- `benchmark_summary.md` - Markdown tables with results
- `benchmark_results.csv` - CSV file for further analysis

## Model Size Configurations

Model configurations are defined in `sweep_local.sh`:

| Size  | d_model | d_ff  | num_layers | num_heads |
|-------|---------|-------|------------|-----------|
| small | 768     | 3072  | 12         | 12        |
| medium| 1024    | 4096  | 24         | 16        |
| large | 1280    | 5120  | 36         | 20        |
| xl    | 1600    | 6400  | 48         | 25        |
| 2.7B  | 2560    | 10240 | 32         | 32        |

To add or modify configurations, edit the `get_model_config()` function in `sweep_local.sh` (lines 18-32).

## Output Format

### JSON Results (per configuration)

```json
{
  "model_size": "medium",
  "vocab": 10000,
  "context_length": 256,
  "d_model": 1024,
  "d_ff": 4096,
  "num_layers": 24,
  "num_heads": 16,
  "batch_size": 4,
  "num_steps": 1,
  "num_warmups": 5,
  "num_trials": 10,
  "forward_only": false,
  "trial_times_ms": [123.45, 124.32, 123.89, 124.01, 123.67, 123.55, 124.11, 123.78, 124.22, 123.91],
  "mean_time_ms": 123.89,
  "std_time_ms": 0.28,
  "device": "cuda:0"
}
```

### Markdown Summary

The aggregation script generates tables like:

```markdown
## Complete Results

| model_size | d_model | context_length | num_layers | num_heads | batch_size | forward_only | mean_time_ms | std_time_ms |
|------------|---------|----------------|------------|-----------|------------|--------------|--------------|-------------|
| small      | 768     | 128            | 12         | 12        | 4          | False        | 145.23       | 1.2         |
| small      | 256     | 128            | 12         | 12        | 4          | False        | 278.45       | 2.1         |
| medium     | 1024    | 128            | 24         | 16        | 4          | False        | 512.31       | 3.5         |
...

## Results by Model Size

### Model: small
| context_length | mean_time_ms | std_time_ms |
|----------------|--------------|-------------|
| 128            | 145.23       | 1.2         |
| 256            | 278.45       | 2.1         |
...

## Forward-Only vs Forward+Backward Comparison
(Shows this section if both forward_only=True and forward_only=False results exist)
```

## Customization Tips

### Add a New Model Configuration

Edit `sweep_local.sh` to add a new model size:

```bash
# In the get_model_config() function, add your custom config
get_model_config() {
    case $1 in
        small)   echo "768 3072 12 12" ;;
        custom)  echo "384 1536 6 6" ;;   # Add your config here
        medium)  echo "1024 4096 24 16" ;;
        # ... rest of configs
    esac
}

# Include it in the sweep
MODEL_SIZES=(small custom large)
```

### Enable Forward-Only Mode

To benchmark only forward passes (skip backward):

```bash
# In sweep_local.sh, uncomment this line (around line 15):
FORWARD_ONLY=1
```

### Sweep Over Different Parameters

You can also modify the script to sweep over other parameters like batch size:

```bash
BATCH_SIZES=(4 8 16)

for batch_size in "${BATCH_SIZES[@]}"; do
    # Run benchmark with different batch sizes
done
```

### Timing and Statistics

The benchmarking script uses:
- **Timer**: `timeit.default_timer()` for high-resolution timing
- **Statistics**: Computes mean and standard deviation across trials
- **Warmup runs**: Avoids cold-start timing issues
- **CUDA synchronization**: Ensures accurate GPU timing

### Custom Aggregation

Modify `aggregate_results.py` to add custom analysis:

```python
# Add custom metrics
df['params'] = df['d_model'] * df['num_layers']  # Approx total params
df['throughput'] = 1000 / df['mean_time_ms']     # Items/sec

# Filter by forward_only
forward_only_df = df[df['forward_only'] == True]
forward_backward_df = df[df['forward_only'] == False]

# Custom pivot tables
pivot = df.pivot_table(
    index='context_length',
    columns='model_size',
    values='mean_time_ms'
)
```

### Example: Complete Workflow

```bash
# 1. Customize sweep parameters
vim cs336_systems/sweep_local.sh

# 2. Run the sweep (takes time depending on configurations)
bash cs336_systems/sweep_local.sh

# 3. Aggregate results
uv run python cs336_systems/aggregate_results.py \
    --results-dir results \
    --output benchmark_summary.md \
    --csv benchmark_results.csv

# 4. View results
cat benchmark_summary.md
```

## Troubleshooting

**Script fails immediately:**
- Verify DATA_PATH exists: `ls ../assignment1-basics/data/TinyStoriesV2-GPT4-train.npy`
- Check Python environment: `uv sync`

**Out of memory:**
- Reduce `--batch_size` or `--context_length`
- Use smaller model sizes (small/medium instead of xl/2_7B)

**No results found:**
- Verify results directory: `ls -l results/`
- Check script output for errors

**Import errors:**
- Install dependencies: `uv sync`
- Verify cs336_basics is installed: `uv run python -c "import cs336_basics"`
