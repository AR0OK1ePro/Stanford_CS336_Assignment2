import os
import cs336_basics.model as models
import torch
from typing import Callable
import timeit
from cs336_systems.torch_util import get_device
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch
import numpy as np
import numpy.typing as npt
import argparse
import json
import torch.cuda.nvtx as nvtx

def run_model(vocab: int, context_length: int, d_model: int, d_ff: int, num_layers: int,
              num_heads: int, batch_size: int, num_steps: int, dataset: npt.NDArray,
              forward_only: bool = False, rope_theta: float = 10000.0) -> Callable:
    # Define a model (with random weights)
    device = get_device()
    model = models.BasicsTransformerLM(vocab, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)
    # Define an input (random)
    x, y  = get_batch(dataset, batch_size, context_length, str(device))

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Data device: {x.device}")

    if forward_only:
        def run():
            # Run the model `num_steps` times (forward only)
            for _ in range(num_steps):
                logits = model(x)
                loss = cross_entropy(logits, y)
    else:
        def run():
            # Run the model `num_steps` times (forward + backward)
            for _ in range(num_steps):
                # Forward
                with nvtx.range("Forward pass"):
                    logits = model(x)
                with nvtx.range("Back pass"):
                    loss = cross_entropy(logits, y)
                    # Backward
                    loss.backward()
                # Clear gradients to avoid memory accumulation
                model.zero_grad()
    return run

def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times with statistics."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now using timeit.default_timer() for highest resolution
    timer = timeit.default_timer
    times: list[float] = []

    for _ in range(num_trials):  # Do it multiple times to capture variance
        start_time = timer()
        with nvtx.range("full run"):
            run()  # Actually perform computation
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
        end_time = timer()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # Compute statistics
    mean_time = np.mean(times)
    std_time = np.std(times, ddof=1)  # Sample standard deviation

    return times, mean_time, std_time

def main():
    parser = argparse.ArgumentParser(description="Benchmarking arguments")

    parser.add_argument('--model_size', type=str, help='Model size label for output')
    parser.add_argument('--vocab', type=int, required=True)
    parser.add_argument('--context_length', type=int, required=True)
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--d_ff', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--num_heads', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_steps', type=int, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num_warmups', type=int, required=True)
    parser.add_argument('--num_trials', type=int, required=True)
    parser.add_argument('--forward_only', action='store_true', help='Only run forward pass (no backward)')

    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    dataset = np.memmap(data_path, dtype=np.uint16, mode='r')
    data_size = len(dataset)
    if data_size < args.context_length + 1:
        raise ValueError(f"Dataset too small: {data_size} < {args.context_length + 1}")
    
    run_func = run_model(
        vocab=args.vocab,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        dataset=dataset,
        forward_only=args.forward_only
    )

    times, mean_time, std_time = benchmark('benchmark', run_func, num_warmups=args.num_warmups, num_trials=args.num_trials)

    # Save results to file
    results = {
        'model_size': args.model_size,
        'vocab': args.vocab,
        'context_length': args.context_length,
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'batch_size': args.batch_size,
        'num_steps': args.num_steps,
        'num_warmups': args.num_warmups,
        'num_trials': args.num_trials,
        'forward_only': args.forward_only,
        'trial_times_ms': times,
        'mean_time_ms': float(mean_time),
        'std_time_ms': float(std_time),
        'device': str(get_device())
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Mean time: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()