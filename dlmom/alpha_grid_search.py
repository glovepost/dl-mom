#!/usr/bin/env python3
"""
Alpha Grid Search for DL-MoM

Finds optimal entropy threshold (α) before running full ablations.
"""

import argparse
import json
import random
import sys
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gsm8k_subset(n_samples: int = 100):
    """Load a subset of GSM8K for grid search."""
    from datasets import load_dataset
    
    dataset = load_dataset("gsm8k", "main", split="test")
    samples = []
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break
        samples.append({
            "question": item["question"],
            "answer": item["answer"],
        })
    return samples


def run_alpha_grid_search(
    alpha_values: list,
    n_samples: int = 100,
    model_id: str = "gpt2",
    max_steps: int = 40,
    seed: int = 42,
    verbose: bool = False,
):
    """
    Run grid search over alpha values.
    
    Args:
        alpha_values: List of α values to test
        n_samples: Number of GSM8K samples
        model_id: HuggingFace model ID
        max_steps: Maximum steps per sample
        seed: Random seed
        verbose: Print per-step entropy for first sample
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dlmom.core.entropy_controller import EntropyController, normalized_entropy
    from dlmom.core.belief_packet import belief_packet_from_logits
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = model.to(device)
    model.eval()
    
    print(f"Loading GSM8K subset: {n_samples} samples")
    samples = load_gsm8k_subset(n_samples)
    
    results = {}
    
    for alpha in alpha_values:
        print(f"\n=== Testing α = {alpha} ===")
        
        controller = EntropyController(
            alpha=alpha,
            window_size=5,
            switch_cap=10,
            max_steps=max_steps,
            convergence_threshold=0.25,
            plateau_variance=0.005,
        )
        
        per_sample_results = []
        all_entropies = []
        
        for i, sample in tqdm(enumerate(samples), total=len(samples), desc=f"α={alpha}"):
            question = sample["question"]
            
            inputs = tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            controller.reset()
            sample_entropies = []
            latent_steps = 0
            explicit_steps = 0
            stopped_reason = "max_steps"
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                
                for step in range(max_steps):
                    entropy = normalized_entropy(logits).mean().item()
                    sample_entropies.append(entropy)
                    
                    # Check EOS
                    top_token_id = logits.argmax(dim=-1).item()
                    eos_generated = (top_token_id == tokenizer.eos_token_id)
                    
                    mode, should_stop = controller.step(entropy, eos_generated=eos_generated)
                    
                    # Debug output for first sample
                    if verbose and i == 0:
                        print(f"  Step {step}: H={entropy:.3f}, mode={mode}, stop={should_stop}")
                    
                    if should_stop:
                        if eos_generated:
                            stopped_reason = "eos"
                        elif entropy < controller.convergence_threshold:
                            stopped_reason = "converged"
                        else:
                            stopped_reason = "plateau"
                        break
                    
                    if mode == "latent":
                        latent_steps += 1
                    else:
                        explicit_steps += 1
                    
                    # Check EOS before continuing
                    if eos_generated:
                        stopped_reason = "eos"
                        break
                    
                    # Generate next token
                    packet = belief_packet_from_logits(logits, top_k=50)
                    top_token = torch.tensor([[top_token_id]], device=device)
                    outputs = model(input_ids=top_token)
                    logits = outputs.logits[:, -1, :]
            
            per_sample_results.append({
                "latent_steps": latent_steps,
                "explicit_steps": explicit_steps,
                "total_steps": latent_steps + explicit_steps,
                "switches": controller.state.switch_count,
                "stopped_reason": stopped_reason,
                "final_entropy": sample_entropies[-1] if sample_entropies else 1.0,
            })
            
            all_entropies.append({
                "min": min(sample_entropies) if sample_entropies else 1.0,
                "max": max(sample_entropies) if sample_entropies else 1.0,
                "mean": float(np.mean(sample_entropies)) if sample_entropies else 1.0,
                "final": sample_entropies[-1] if sample_entropies else 1.0,
            })
        
        # Compute summary
        timeout_count = sum(1 for r in per_sample_results if r["stopped_reason"] == "max_steps")
        
        results[alpha] = {
            "alpha": alpha,
            "summary": {
                "avg_latent_steps": float(np.mean([r["latent_steps"] for r in per_sample_results])),
                "avg_explicit_steps": float(np.mean([r["explicit_steps"] for r in per_sample_results])),
                "avg_total_steps": float(np.mean([r["total_steps"] for r in per_sample_results])),
                "avg_switches": float(np.mean([r["switches"] for r in per_sample_results])),
                "timeout_rate": timeout_count / n_samples,
                "stop_reasons": {
                    reason: sum(1 for r in per_sample_results if r["stopped_reason"] == reason)
                    for reason in ["eos", "converged", "plateau", "max_steps"]
                },
            },
            "entropy_stats": {
                "avg_min": float(np.mean([e["min"] for e in all_entropies])),
                "avg_max": float(np.mean([e["max"] for e in all_entropies])),
                "avg_mean": float(np.mean([e["mean"] for e in all_entropies])),
                "avg_final": float(np.mean([e["final"] for e in all_entropies])),
            },
            "n_samples": n_samples,
        }
        
        s = results[alpha]["summary"]
        e = results[alpha]["entropy_stats"]
        print(f"  Latent: {s['avg_latent_steps']:.1f}, "
              f"Explicit: {s['avg_explicit_steps']:.1f}, "
              f"Switches: {s['avg_switches']:.2f}, "
              f"Timeouts: {s['timeout_rate']:.1%}")
        print(f"  Entropy: min={e['avg_min']:.3f}, mean={e['avg_mean']:.3f}, final={e['avg_final']:.3f}")
        print(f"  Stop reasons: {s['stop_reasons']}")
    
    return results


def select_optimal_alpha(results: dict) -> float:
    """
    Select optimal α using explicit criteria:
    1. Timeout rate < 20% (hard constraint)
    2. Among valid, maximize latent_steps / total_steps ratio
    """
    valid = {
        a: r for a, r in results.items() 
        if r["summary"]["timeout_rate"] < 0.20
    }
    
    if not valid:
        print("WARNING: No α achieves <20% timeout rate. Selecting lowest timeout.")
        return min(results.keys(), key=lambda a: results[a]["summary"]["timeout_rate"])
    
    def latent_ratio(a):
        s = valid[a]["summary"]
        total = s["avg_total_steps"]
        return s["avg_latent_steps"] / total if total > 0 else 0
    
    return max(valid.keys(), key=latent_ratio)


def main():
    parser = argparse.ArgumentParser(description="Alpha Grid Search for DL-MoM")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/alpha_grid_search.json")
    parser.add_argument("--verbose", action="store_true", help="Print per-step entropy for first sample")
    args = parser.parse_args()
    
    print("DL-MoM Alpha Grid Search")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Samples: {args.samples}")
    print(f"Alpha values: {args.alpha}")
    print(f"Max steps: {args.max_steps}")
    print(f"Seed: {args.seed}")
    
    results = run_alpha_grid_search(
        alpha_values=args.alpha,
        n_samples=args.samples,
        model_id=args.model,
        max_steps=args.max_steps,
        seed=args.seed,
        verbose=args.verbose,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert float keys to strings for JSON
    json_results = {str(k): v for k, v in results.items()}
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Select optimal alpha
    best_alpha = select_optimal_alpha(results)
    print(f"\n{'=' * 50}")
    print(f"Recommended α: {best_alpha}")
    
    # Show comparison table
    print(f"\n{'α':<6} {'Latent':<8} {'Explicit':<10} {'Timeout':<10} {'Converged':<10}")
    print("-" * 50)
    for alpha in sorted(results.keys()):
        s = results[alpha]["summary"]
        sr = s["stop_reasons"]
        print(f"{alpha:<6.2f} {s['avg_latent_steps']:<8.1f} {s['avg_explicit_steps']:<10.1f} "
              f"{sr['max_steps']:<10} {sr['converged']:<10}")


if __name__ == "__main__":
    main()
