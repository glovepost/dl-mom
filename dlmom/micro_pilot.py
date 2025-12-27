#!/usr/bin/env python3
"""
Micro-Pilot Test for DL-MoM

Runs a quick smoke test on n=50 GSM8K samples to validate the full pipeline.
"""

import argparse
import json
import os
import sys
import time
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_micro_pilot(
    n_samples: int = 50,
    model_id: str = "gpt2",
    device: str = "cuda",
    top_k: int = 50,
    alpha: float = 0.6,
):
    """Run micro-pilot test."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dlmom.core.loop import DLMoMLoop
    from datasets import load_dataset
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    # Create DL-MoM loop with single expert (for smoke test)
    loop = DLMoMLoop(
        experts=[model],
        tokenizer=tokenizer,
        top_k=top_k,
        alpha=alpha,
    )
    
    print(f"Loading GSM8K: {n_samples} samples")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    results = []
    total_time = 0
    
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break
        
        question = item["question"]
        
        # Tokenize
        inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Run loop
        start = time.time()
        with torch.no_grad():
            result = loop.run(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        elapsed = time.time() - start
        total_time += elapsed
        
        results.append({
            "sample_id": i,
            "total_steps": result.total_steps,
            "latent_steps": result.latent_steps,
            "explicit_tokens": result.explicit_tokens,
            "switches": result.switches,
            "wall_time_s": elapsed,
            "top_k_mass": result.final_packet.top_k_mass,
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_samples} "
                  f"(avg: {total_time / (i + 1):.2f}s/sample)")
    
    # Summary
    avg_latent = sum(r["latent_steps"] for r in results) / len(results)
    avg_explicit = sum(r["explicit_tokens"] for r in results) / len(results)
    avg_switches = sum(r["switches"] for r in results) / len(results)
    avg_time = total_time / len(results)
    
    summary = {
        "model": model_id,
        "n_samples": n_samples,
        "top_k": top_k,
        "alpha": alpha,
        "avg_latent_steps": avg_latent,
        "avg_explicit_tokens": avg_explicit,
        "avg_switches": avg_switches,
        "avg_time_s": avg_time,
        "total_time_s": total_time,
        "timestamp": datetime.now().isoformat(),
    }
    
    return results, summary


def main():
    parser = argparse.ArgumentParser(description="DL-MoM Micro-Pilot")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--output", default="runs/micro_pilot.json")
    args = parser.parse_args()
    
    print("DL-MoM Micro-Pilot Test")
    print("=" * 50)
    
    results, summary = run_micro_pilot(
        n_samples=args.samples,
        model_id=args.model,
        top_k=args.top_k,
        alpha=args.alpha,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"  Samples: {summary['n_samples']}")
    print(f"  Avg latent steps: {summary['avg_latent_steps']:.1f}")
    print(f"  Avg explicit tokens: {summary['avg_explicit_tokens']:.1f}")
    print(f"  Avg switches: {summary['avg_switches']:.2f}")
    print(f"  Avg time: {summary['avg_time_s']:.2f}s")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
