#!/usr/bin/env python3
"""
DL-MoM CLI Runner

Usage:
    dlmom run --suite A1 --bench gsm8k --out runs/2024-01-01_A1 --seeds 0 1 2
    dlmom aggregate --in runs/2024-01-01_A1 --out runs/2024-01-01_A1/aggregate.csv
    dlmom plot --in runs/2024-01-01_A1 --out runs/2024-01-01_A1/figs
"""

import argparse
import json
import os
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_suite_config(suite_name: str) -> List[Dict[str, Any]]:
    """Load experiment configurations for a suite."""
    config_path = Path(__file__).parent.parent / "configs" / "suites.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    defaults = config.get("defaults", {})
    suite_configs = config.get("suites", {}).get(suite_name, [])
    
    # Merge defaults with each experiment config
    merged = []
    for exp in suite_configs:
        merged_config = {**defaults, **exp}
        merged.append(merged_config)
    
    return merged


def run_experiment(
    exp_config: Dict[str, Any],
    bench: str,
    seed: int,
    output_dir: Path,
    device: str = "cuda",
    dtype: str = "bf16",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    exp_id = exp_config["id"]
    print(f"  Running {exp_id} (seed={seed})...")
    
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load benchmark data
    results = []
    start_time = time.time()
    
    # For now, return placeholder - actual implementation would run DLMoMLoop
    # This is a scaffold for the full experiment runner
    
    summary = {
        "exp_id": exp_id,
        "seed": seed,
        "bench": bench,
        "config": exp_config,
        "accuracy": 0.0,  # Placeholder
        "total_samples": 0,
        "wall_time_s": time.time() - start_time,
        "timestamp": datetime.now().isoformat(),
    }
    
    return summary


def cmd_run(args):
    """Run a suite of experiments."""
    print(f"DL-MoM Experiment Runner")
    print(f"=" * 50)
    print(f"Suite: {args.suite}")
    print(f"Benchmark: {args.bench}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {args.out}")
    print()
    
    # Create output directories
    output_dir = Path(args.out)
    (output_dir / "configs").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "summary").mkdir(exist_ok=True)
    
    # Load suite configuration
    experiments = load_suite_config(args.suite)
    print(f"Loaded {len(experiments)} experiments from suite {args.suite}")
    
    # Save experiment configs
    for exp in experiments:
        config_path = output_dir / "configs" / f"{exp['id']}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(exp, f)
    
    # Run experiments
    all_summaries = []
    for exp in experiments:
        for seed in args.seeds:
            try:
                summary = run_experiment(
                    exp_config=exp,
                    bench=args.bench,
                    seed=seed,
                    output_dir=output_dir,
                    device=args.device,
                    dtype=args.dtype,
                    max_samples=args.max_samples,
                )
                all_summaries.append(summary)
                
                # Save individual summary
                summary_path = output_dir / "summary" / f"{exp['id']}_seed{seed}.json"
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
                    
            except Exception as e:
                print(f"  ERROR in {exp['id']} seed={seed}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save combined summary
    combined_path = output_dir / "all_summaries.json"
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\nCompleted {len(all_summaries)} experiments")
    print(f"Results saved to {output_dir}")


def cmd_aggregate(args):
    """Aggregate results into CSV."""
    import csv
    
    input_dir = Path(args.input)
    output_path = Path(args.out)
    
    # Load all summaries
    summaries = []
    summary_dir = input_dir / "summary"
    for f in summary_dir.glob("*.json"):
        with open(f) as fp:
            summaries.append(json.load(fp))
    
    if not summaries:
        print(f"No summaries found in {summary_dir}")
        return
    
    # Get all keys
    all_keys = set()
    for s in summaries:
        all_keys.update(s.keys())
    
    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
        writer.writeheader()
        for s in summaries:
            # Flatten config dict if present
            row = {}
            for k, v in s.items():
                if isinstance(v, dict):
                    row[k] = json.dumps(v)
                else:
                    row[k] = v
            writer.writerow(row)
    
    print(f"Aggregated {len(summaries)} results to {output_path}")


def cmd_plot(args):
    """Generate plots from results."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib and pandas required for plotting")
        return
    
    input_dir = Path(args.input)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load aggregate CSV or summaries
    csv_path = input_dir / "aggregate.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        # Load from JSON
        summaries = []
        for f in (input_dir / "summary").glob("*.json"):
            with open(f) as fp:
                summaries.append(json.load(fp))
        df = pd.DataFrame(summaries)
    
    if df.empty:
        print("No data to plot")
        return
    
    # Plot 1: Accuracy vs Total Steps (Pareto)
    if "accuracy" in df.columns and "total_steps" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df["total_steps"], df["accuracy"])
        plt.xlabel("Total Steps")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Total Steps (Pareto)")
        plt.savefig(output_dir / f"accuracy_vs_total_steps.{args.format}")
        plt.close()
    
    # Plot 2: Accuracy vs Wall Clock
    if "accuracy" in df.columns and "wall_time_s" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df["wall_time_s"], df["accuracy"])
        plt.xlabel("Wall Time (s)")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Wall Clock")
        plt.savefig(output_dir / f"accuracy_vs_wallclock.{args.format}")
        plt.close()
    
    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DL-MoM Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run experiments")
    run_parser.add_argument("--suite", required=True, help="Suite name (A1, A2, etc.)")
    run_parser.add_argument("--bench", required=True, help="Benchmark name")
    run_parser.add_argument("--out", required=True, help="Output directory")
    run_parser.add_argument("--device", default="cuda", help="Device (cuda, hip, cpu)")
    run_parser.add_argument("--dtype", default="bf16", help="Data type")
    run_parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    run_parser.add_argument("--max-samples", type=int, default=None)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.set_defaults(func=cmd_run)
    
    # Aggregate subcommand
    agg_parser = subparsers.add_parser("aggregate", help="Aggregate results")
    agg_parser.add_argument("--in", dest="input", required=True, help="Input directory")
    agg_parser.add_argument("--out", required=True, help="Output CSV path")
    agg_parser.set_defaults(func=cmd_aggregate)
    
    # Plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Generate plots")
    plot_parser.add_argument("--in", dest="input", required=True, help="Input directory")
    plot_parser.add_argument("--out", required=True, help="Output directory")
    plot_parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    plot_parser.set_defaults(func=cmd_plot)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
