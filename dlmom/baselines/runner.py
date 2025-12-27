#!/usr/bin/env python3
"""
Baseline Runner for DL-MoM Comparison

Runs all baseline methods on a benchmark and saves results.
"""

import argparse
import json
import sys
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_baselines(
    methods: List[str],
    model,
    tokenizer,
    samples: List[Dict],
    n_samples: int = 100,
) -> Dict[str, Any]:
    """Run all baseline methods on samples."""
    from dlmom.baselines.methods import get_baseline
    
    results = {}
    
    for method in methods:
        print(f"\n=== Running {method} ===")
        
        baseline = get_baseline(method, model, tokenizer)
        method_results = []
        correct = 0
        
        for i, sample in enumerate(samples[:n_samples]):
            question = sample["question"]
            gold = sample.get("gold_answer", "")
            
            try:
                result = baseline.run(question)
                
                # Check correctness (simple exact match for now)
                is_correct = result.answer == gold
                if is_correct:
                    correct += 1
                
                method_results.append({
                    "sample_id": i,
                    "answer": result.answer,
                    "gold": gold,
                    "correct": is_correct,
                    "latency_s": result.latency_s,
                    "tokens": result.tokens_generated,
                })
            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                method_results.append({
                    "sample_id": i,
                    "error": str(e),
                })
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{n_samples}")
        
        accuracy = correct / len(method_results) if method_results else 0
        avg_latency = np.mean([r["latency_s"] for r in method_results if "latency_s" in r])
        avg_tokens = np.mean([r["tokens"] for r in method_results if "tokens" in r])
        
        results[method] = {
            "accuracy": accuracy,
            "avg_latency_s": avg_latency,
            "avg_tokens": avg_tokens,
            "n_samples": len(method_results),
            "n_correct": correct,
            "results": method_results,
        }
        
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Latency: {avg_latency:.2f}s")
        print(f"  Tokens: {avg_tokens:.1f}")
    
    return results


def load_samples(benchmark: str, n_samples: int = 100) -> List[Dict]:
    """Load benchmark samples with gold answers."""
    from datasets import load_dataset
    import re
    
    if benchmark == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
        samples = []
        for i, item in enumerate(dataset):
            if i >= n_samples:
                break
            
            # Extract gold answer from GSM8K format
            gold = ""
            match = re.search(r"####\s*(-?[\d,]+\.?\d*)", item["answer"])
            if match:
                gold = match.group(1).replace(",", "")
            
            samples.append({
                "question": item["question"],
                "answer": item["answer"],
                "gold_answer": gold,
            })
        return samples
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def main():
    parser = argparse.ArgumentParser(description="Baseline Runner")
    parser.add_argument("--methods", nargs="+", default=["direct", "cot", "self_consistency", "text_mas"])
    parser.add_argument("--bench", default="gsm8k")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Baseline Runner")
    print("=" * 50)
    print(f"Methods: {args.methods}")
    print(f"Benchmark: {args.bench}")
    print(f"Samples: {args.samples}")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    # Load samples
    print(f"Loading {args.bench} samples...")
    samples = load_samples(args.bench, args.samples)
    print(f"Loaded {len(samples)} samples")
    
    # Run baselines
    results = run_baselines(
        methods=args.methods,
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        n_samples=args.samples,
    )
    
    # Add metadata
    output = {
        "metadata": {
            "model": args.model,
            "benchmark": args.bench,
            "n_samples": args.samples,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    # Summary
    print(f"\n{'=' * 50}")
    print("Summary:")
    for method, data in results.items():
        print(f"  {method}: {data['accuracy']:.2%} accuracy, {data['avg_latency_s']:.2f}s latency")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
