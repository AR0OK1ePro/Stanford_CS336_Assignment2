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

Note: All arguments are required. The `--model_size` is just a label for the output JSON.

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
# Model configurations (lines 17-22)
MODEL_CONFIGS[small]="256 1024 4 4"     # d_model d_ff num_layers num_heads
MODEL_CONFIGS[medium]="512 2048 8 8"
MODEL_CONFIGS[large]="768 3072 12 12"
MODEL_CONFIGS[xl]="1024 4096 24 16"
MODEL_CONFIGS[2_7B]="2560 10240 32 32"

# Which sizes to sweep (line 34)
MODEL_SIZES=(small medium large)

# Which context lengths to sweep (line 35)
CONTEXT_LENGTHS=(128 256 512)

# Fixed parameters (lines 10-14)
VOCAB=10000
NUM_WARMUPS=2
NUM_TRIALS=5
NUM_STEPS=10
BATCH_SIZE=8
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
| small | 256     | 1024  | 4          | 4         |
| medium| 512     | 2048  | 8          | 8         |
| large | 768     | 3072  | 12         | 12        |
| xl    | 1024    | 4096  | 24         | 16        |
| 2_7B  | 2560    | 10240 | 32         | 32        |

To add or modify configurations, edit the `MODEL_CONFIGS` associative array in `sweep_local.sh` (lines 17-22).

## Output Format

### JSON Results (per configuration)

```json
{
  "model_size": "medium",
  "vocab": 10000,
  "context_length": 256,
  "d_model": 512,
  "d_ff": 2048,
  "num_layers": 8,
  "num_heads": 8,
  "batch_size": 8,
  "num_steps": 10,
  "num_warmups": 2,
  "num_trials": 5,
  "trial_times_ms": [123.45, 124.32, 123.89, 124.01, 123.67],
  "mean_time_ms": 123.87,
  "device": "cuda:0"
}
```

### Markdown Summary

The aggregation script generates tables like:

```markdown
## Complete Results

| d_model | context_length | num_layers | num_heads | batch_size | mean_time_ms | std_time_ms |
|---------|----------------|------------|-----------|------------|--------------|-------------|
| 256     | 128            | 4          | 4         | 8          | 45.23        | 1.2         |
| 512     | 256            | 8          | 8         | 8          | 178.45       | 3.2         |
...
```

## Customization Tips

### Add a New Model Configuration

Edit `sweep_local.sh` to add a new model size:

```bash
# Add your custom config
MODEL_CONFIGS[custom]="384 1536 6 6"  # d_model=384, d_ff=1536, layers=6, heads=6

# Include it in the sweep
MODEL_SIZES=(small custom large)
```

### Sweep Over Different Parameters

You can also modify the script to sweep over other parameters like batch size:

```bash
BATCH_SIZES=(4 8 16)

for batch_size in "${BATCH_SIZES[@]}"; do
    # Run benchmark with different batch sizes
done
```

### Custom Aggregation

Modify `aggregate_results.py` to add custom analysis:

```python
# Add custom metrics
df['params'] = df['d_model'] * df['num_layers']  # Approx total params
df['throughput'] = 1000 / df['mean_time_ms']     # Items/sec

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
