#!/usr/bin/env python3
"""
Aggregate benchmark results from multiple JSON files and generate markdown tables.

Usage:
    python aggregate_results.py --results-dir results --output summary.md
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON result files from a directory."""
    results = []
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"Warning: No JSON files found in {results_dir}")
        return results

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    print(f"Loaded {len(results)} result files")
    return results


def create_summary_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a pandas DataFrame from benchmark results."""
    if not results:
        return pd.DataFrame()

    # Extract relevant fields
    df_data = []
    for r in results:
        row = {
            'model_size': r.get('model_size'),
            'vocab': r.get('vocab'),
            'context_length': r.get('context_length'),
            'd_model': r.get('d_model'),
            'd_ff': r.get('d_ff'),
            'num_layers': r.get('num_layers'),
            'num_heads': r.get('num_heads'),
            'batch_size': r.get('batch_size'),
            'num_steps': r.get('num_steps'),
            'num_warmups': r.get('num_warmups'),
            'num_trials': r.get('num_trials'),
            'forward_only': r.get('forward_only', False),
            'mean_time_ms': r.get('mean_time_ms'),
            'std_time_ms': r.get('std_time_ms'),
            'device': r.get('device'),
        }

        # Calculate additional statistics if trial times are available
        if 'trial_times_ms' in r and r['trial_times_ms']:
            times = r['trial_times_ms']
            row['min_time_ms'] = min(times)
            row['max_time_ms'] = max(times)
            # Use std_time_ms from results if not already present
            if 'std_time_ms' not in r or r['std_time_ms'] is None:
                row['std_time_ms'] = np.std(times, ddof=1)

        df_data.append(row)

    df = pd.DataFrame(df_data)

    # Sort by model_size, context_length, num_warmups, and forward_only
    if not df.empty:
        sort_cols = ['d_model', 'context_length', 'num_warmups', 'forward_only']
        existing_cols = [c for c in sort_cols if c in df.columns]
        if existing_cols:
            df = df.sort_values(existing_cols)

    return df


def generate_markdown_tables(df: pd.DataFrame, output_file: Path):
    """Generate markdown tables from DataFrame."""
    with open(output_file, 'w') as f:
        f.write("# Benchmark Results Summary\n\n")

        if df.empty:
            f.write("No results found.\n")
            return

        # Full results table
        f.write("## Complete Results\n\n")

        # Select columns for display
        display_cols = ['model_size', 'd_model', 'context_length', 'num_layers', 'num_heads',
                       'batch_size', 'num_warmups', 'num_trials', 'forward_only', 'mean_time_ms', 'std_time_ms']

        available_cols = [c for c in display_cols if c in df.columns]
        display_df = df[available_cols].copy()

        # Round numeric columns
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        display_df[numeric_cols] = display_df[numeric_cols].round(2)

        f.write(display_df.to_markdown(index=False))
        f.write("\n\n")

        # Group by model_size for easier comparison
        if 'model_size' in df.columns and df['model_size'].notna().any():
            f.write("## Results by Model Size\n\n")

            for model_size in df['model_size'].dropna().unique():
                f.write(f"### Model: {model_size}\n\n")
                subset = df[df['model_size'] == model_size]

                if 'context_length' in subset.columns and 'mean_time_ms' in subset.columns:
                    # Simple table grouped by context length
                    summary_cols = ['context_length', 'num_warmups', 'forward_only', 'mean_time_ms', 'std_time_ms']
                    available_summary_cols = [c for c in summary_cols if c in subset.columns]
                    summary = subset[available_summary_cols].copy()
                    summary = summary.round(2)
                    f.write(summary.to_markdown(index=False))
                    f.write("\n\n")

        # Comparison: Forward vs Forward+Backward (if both present)
        if 'forward_only' in df.columns and df['forward_only'].nunique() > 1:
            f.write("## Forward-Only vs Forward+Backward Comparison\n\n")

            comparison_cols = ['model_size', 'context_length', 'num_warmups', 'forward_only', 'mean_time_ms', 'std_time_ms']
            available_comp_cols = [c for c in comparison_cols if c in df.columns]
            comp_df = df[available_comp_cols].copy()
            comp_df = comp_df.round(2)
            f.write(comp_df.to_markdown(index=False))
            f.write("\n\n")

        # Comparison by num_warmups (if multiple values present)
        if 'num_warmups' in df.columns and df['num_warmups'].nunique() > 1:
            f.write("## Effect of Number of Warmup Runs\n\n")

            warmup_cols = ['model_size', 'context_length', 'num_warmups', 'mean_time_ms', 'std_time_ms']
            available_warmup_cols = [c for c in warmup_cols if c in df.columns]
            warmup_df = df[available_warmup_cols].copy()
            warmup_df = warmup_df.sort_values(['model_size', 'context_length', 'num_warmups'])
            warmup_df = warmup_df.round(2)
            f.write(warmup_df.to_markdown(index=False))
            f.write("\n\n")

            # Pivot table for easier comparison
            if all(col in df.columns for col in ['model_size', 'num_warmups', 'mean_time_ms']):
                f.write("### Mean Time by Warmup Count\n\n")
                pivot = df.pivot_table(
                    index='model_size',
                    columns='num_warmups',
                    values='mean_time_ms',
                    aggfunc='mean'
                )
                pivot = pivot.round(2)
                f.write(pivot.to_markdown())
                f.write("\n\n")

        # Summary statistics
        f.write("## Summary Statistics\n\n")
        stats_cols = ['mean_time_ms']
        if 'std_time_ms' in df.columns:
            stats_cols.append('std_time_ms')
        stats_df = df[stats_cols].describe().round(2)
        f.write(stats_df.to_markdown())
        f.write("\n")

    print(f"Markdown summary saved to {output_file}")


def generate_csv(df: pd.DataFrame, output_file: Path):
    """Save results as CSV for further analysis."""
    if df.empty:
        print("No data to save to CSV")
        return

    df.to_csv(output_file, index=False)
    print(f"CSV results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark results and generate summary tables"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing benchmark result JSON files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_summary.md',
        help='Output markdown file path'
    )
    parser.add_argument(
        '--csv',
        type=str,
        help='Optional: Save results as CSV to this path'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return

    # Load results
    results = load_results(results_dir)

    if not results:
        print("No results to process")
        return

    # Create DataFrame
    df = create_summary_dataframe(results)

    # Generate outputs
    generate_markdown_tables(df, Path(args.output))

    if args.csv:
        generate_csv(df, Path(args.csv))

    # Print quick summary to console
    print("\n" + "="*50)
    print("Quick Summary:")
    print("="*50)
    if not df.empty and 'mean_time_ms' in df.columns:
        print(f"Total configurations: {len(df)}")
        print(f"Mean time: {df['mean_time_ms'].mean():.2f} ms")
        print(f"Min time: {df['mean_time_ms'].min():.2f} ms")
        print(f"Max time: {df['mean_time_ms'].max():.2f} ms")


if __name__ == '__main__':
    main()
