#!/usr/bin/env python3
"""
DL-MoM Ablation Runner

Runs ablation experiments with full accuracy measurement, timeout handling,
and comprehensive metric collection.
"""

import argparse
import json
import os
import sys
import time
import random
import signal
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============== Configuration ==============

BENCHMARK_CONFIGS = {
    "gsm8k": {
        "path": "gsm8k", "name": "main", "split": "test",
        "q_field": "question", "a_field": "answer",
        "answer_extractor": "gsm8k",
    },
    "math": {
        "path": "qwedsacf/competition_math", "name": None, "split": "train",
        "q_field": "problem", "a_field": "solution",
        "answer_extractor": "math",
    },
    "openbookqa": {
        "path": "openbookqa", "name": "main", "split": "test",
        "q_field": "question_stem", "a_field": "answerKey",
        "answer_extractor": "multiple_choice",
        "choices_field": "choices", # Dict with 'text' and 'label'
    },
    "mmlu": {
        "path": "cais/mmlu", "name": "all", "split": "test",
        "q_field": "question", "a_field": "answer",
        "answer_extractor": "multiple_choice_index", # Answer is 0-3 index
        "choices_field": "choices",
    },
    "mmlu-redux": {
        "path": "edinburgh-dawg/mmlu-redux", "name": "all", "split": "test",
        "q_field": "question", "a_field": "answer", 
        "answer_extractor": "multiple_choice_index",
        "choices_field": "choices",
    },
    "arc": {
        "path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "test",
        "q_field": "question", "a_field": "answerKey",
        "answer_extractor": "multiple_choice",
        "choices_field": "choices", # Dict with 'text' and 'label'
    },
    "ceval": {
        "path": "ceval/ceval-exam", "name": "computer_network", "split": "test", # Default subset
        "q_field": "question", "a_field": "answer",
        "answer_extractor": "multiple_choice",
    },
    "humaneval": {
        "path": "openai_humaneval", "name": None, "split": "test",
        "q_field": "prompt", "a_field": "canonical_solution",
        "answer_extractor": "code",
    },
    "mbpp": {
        "path": "mbpp", "name": None, "split": "test",
        "q_field": "text", "a_field": "code",
        "answer_extractor": "code",
    },
    "hellaswag": {
        "path": "Rowan/hellaswag", "name": None, "split": "validation",
        "q_field": "ctx", "a_field": "label",
        "answer_extractor": "multiple_choice_index",
        "choices_field": "endings",
    },
}


# ============== Utilities ==============

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SampleTimeoutError(Exception):
    """Raised when a single sample times out during processing."""
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for timing out operations (Unix only)."""
    def handler(signum, frame):
        raise SampleTimeoutError(f"Timed out after {seconds}s")
    
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract numerical answer from GSM8K format (#### <number>)."""
    import re
    
    # Try official GSM8K format first: #### <number>
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").rstrip(".")
    
    # Try "the answer is <number>" pattern
    match = re.search(r"(?:the answer is|answer is|answer:)\s*\$?(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").rstrip(".")
    
    # Fallback: try to find last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        # Filter out empty strings and clean
        cleaned = [n.replace(",", "").rstrip(".") for n in numbers if n.strip()]
        return cleaned[-1] if cleaned else None
    return None


def extract_math_answer(text: str) -> Optional[str]:
    """Extract answer from MATH format (latex boxed)."""
    import re
    
    # Look for \boxed{...} 
    # This is a simple regex, might fail on nested braces
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    
    # Fallback to last number? No, MATH is rigorous.
    return None


def extract_mcq_answer(text: str) -> Optional[str]:
    """Extract multiple choice answer (A/B/C/D)."""
    import re
    
    # Look for "Answer: (A)" or "The answer is A"
    match = re.search(r"(?:Answer:|answer is)\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # Check for standalone "A" at end of text? Risky.
    # Look for boxed single letter
    match = re.search(r"\\boxed\{([A-D])\}", text)
    if match:
        return match.group(1).upper()
        
    return None


def extract_code_block(text: str) -> Optional[str]:
    """Extract Python code from model output, handling markdown code blocks."""
    import re
    
    if not text:
        return ""
    
    # Try to find ```python ... ``` block first
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try generic ``` ... ``` block
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try to find Python code starting with 'def ' or 'class '
    # This handles cases where model outputs explanation then code
    lines = text.split('\n')
    code_start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('def ') or stripped.startswith('class ') or stripped.startswith('import ') or stripped.startswith('from '):
            code_start = i
            break
    
    if code_start is not None:
        # Return from code start to end
        return '\n'.join(lines[code_start:])
    
    # Last resort: return text as-is
    return text.strip()


def execute_code_with_tests(
    generated_code: str,
    test_code: str,
    entry_point: Optional[str] = None,
    timeout_seconds: int = 5,
    benchmark_type: str = "humaneval",
) -> Tuple[bool, str]:
    """
    Execute generated code with test cases in a sandboxed subprocess.
    
    Args:
        generated_code: The model-generated code completion
        test_code: Test code (HumanEval: check() function, MBPP: assert statements)
        entry_point: Function name for HumanEval (to alias as 'candidate')
        timeout_seconds: Execution timeout
        benchmark_type: 'humaneval' or 'mbpp'
        
    Returns:
        (passed: bool, error_message: str)
    """
    import subprocess
    import tempfile
    import os
    
    # Extract code from markdown if needed
    code = extract_code_block(generated_code)
    
    if benchmark_type == "humaneval":
        # HumanEval: code completes the function, test has check(candidate)
        # We need to: run code, alias entry_point as candidate, run check
        full_code = f"""
{code}

{test_code}

check({entry_point})
"""
    else:  # mbpp
        # MBPP: test_code is list of assert statements joined
        full_code = f"""
{code}

{test_code}
"""
    
    # Write to temp file and execute
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_path = f.name
        
        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        
        os.unlink(temp_path)
        
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr[:500]
            
    except subprocess.TimeoutExpired:
        if 'temp_path' in locals():
            os.unlink(temp_path)
        return False, "Execution timeout"
    except Exception as e:
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        return False, str(e)[:500]


def extract_answer(text: str, extractor: str) -> Optional[str]:
    """Generic answer extractor dispatcher."""
    if not text:
        return None
        
    if extractor == "gsm8k":
        return extract_gsm8k_answer(text)
    elif extractor == "math":
        return extract_math_answer(text)
    elif extractor == "multiple_choice":
        return extract_mcq_answer(text)
    elif extractor == "multiple_choice_index":
        # MMLU uses indices 0-3, we normalize to A-D
        # But prediction is usually letter.
        return extract_mcq_answer(text)
    elif extractor == "code":
        return None # Code eval not implemented in this runner
        
    return None


def answers_match(pred: Optional[str], gold: Optional[str], tolerance: float = 0.01) -> bool:
    """Check if predicted answer matches gold answer."""
    if pred is None or gold is None:
        return False
    
    # Clean both values
    pred = pred.strip().rstrip(".").rstrip(",")
    gold = gold.strip().rstrip(".").rstrip(",")
    
    try:
        pred_val = float(pred)
        gold_val = float(gold)
        # Use relative tolerance for large numbers
        if abs(gold_val) > 100:
            return abs(pred_val - gold_val) / abs(gold_val) <= tolerance
        return abs(pred_val - gold_val) <= tolerance
        return abs(pred_val - gold_val) <= tolerance
    except ValueError:
        # String matching for MCQ
        return pred.strip().lower() == gold.strip().lower()


# ============== Data Loading ==============

def load_samples(benchmark: str, n_samples: int = 500) -> List[Dict]:
    """Load benchmark samples."""
    from datasets import load_dataset
    
    if benchmark not in BENCHMARK_CONFIGS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARK_CONFIGS.keys())}")
    
    cfg = BENCHMARK_CONFIGS[benchmark]
    dataset = load_dataset(cfg["path"], cfg["name"], split=cfg["split"], trust_remote_code=True)
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break
            
        question = item[cfg["q_field"]]
        answer = item[cfg["a_field"]]
        
        # Handle MCQ formatting
        if "choices_field" in cfg:
            choices = item[cfg["choices_field"]]
            formatted_choices = []
            labels = []
            
            # Case 1: List of strings (MMLU style)
            if isinstance(choices, list): 
                 for idx, choice in enumerate(choices):
                     label = chr(65 + idx) # A, B, C...
                     formatted_choices.append(f"({label}) {choice}")
                     labels.append(label)
                 
                 # Answer is usually index for MMLU/HellaSwag
                 # Handle both int and string representations (HellaSwag uses strings like "3")
                 answer_idx = None
                 if isinstance(answer, int):
                     answer_idx = answer
                 elif isinstance(answer, str) and answer.isdigit():
                     answer_idx = int(answer)
                 
                 if answer_idx is not None and 0 <= answer_idx < len(labels):
                     answer = labels[answer_idx]
                     
            # Case 2: Dict with text/label lists (ARC/OBQA style)
            elif isinstance(choices, dict): 
                 texts = choices.get("text", [])
                 lbls = choices.get("label", [])
                 for t, l in zip(texts, lbls):
                     formatted_choices.append(f"({l}) {t}")
                     
            question = f"{question}\n" + "\n".join(formatted_choices)
            question += "\nAnswer:"
        
        # Build sample dict
        sample_data = {
            "question": question,
            "answer": str(answer), # Ensure string
            "extractor": cfg["answer_extractor"],
        }
        
        # Add test data for code benchmarks
        if cfg["answer_extractor"] == "code":
            if benchmark == "humaneval":
                sample_data["test_code"] = item.get("test", "")
                sample_data["entry_point"] = item.get("entry_point", "")
                sample_data["prompt"] = item.get("prompt", question)  # Original prompt
            elif benchmark == "mbpp":
                # MBPP has test_list as list of assert strings
                test_list = item.get("test_list", [])
                sample_data["test_code"] = "\n".join(test_list)
                sample_data["entry_point"] = None
                sample_data["prompt"] = question
        
        samples.append(sample_data)
    
    return samples


def load_experts(
    expert_names: List[str],
    device: torch.device,
    kv_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[torch.nn.Module], Any]:
    """
    Load multiple expert models for heterogeneous expert experiments.
    
    Args:
        expert_names: List of HuggingFace model names/paths
        device: Target device for models
        kv_config: KV compression config dict with keys:
            - method: 'none', 'kivi2bit', 'kivi4bit', 'minicache', etc.
            - k_bits: Key quantization bits (2 or 4)
            - v_bits: Value quantization bits (2 or 4)
    
    Returns:
        (list_of_models, tokenizer) - tokenizer is from the first model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if not expert_names:
        raise ValueError("At least one expert model is required")
    
    models = []
    tokenizer = None
    
    for i, model_name in enumerate(expert_names):
        print(f"  Loading expert {i+1}/{len(expert_names)}: {model_name}")
        
        # Load tokenizer from first model only
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # Use consistent dtype for all experts (float16 for compatibility)
        expert_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Determine compression method
        kv_method = kv_config.get("method", "none") if kv_config else "none"
        uses_kivi = kv_method in ["kivi2bit", "kivi4bit", "minicache+kivi2bit"]
        
        # Load KIVI model if requested and it's a Qwen model
        if uses_kivi and "qwen" in model_name.lower():
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "kivi"))
                from models.qwen_kivi import load_qwen2_kivi
                
                # Get bit-width from config
                k_bits = kv_config.get("k_bits", 2) if kv_config else 2
                v_bits = kv_config.get("v_bits", 2) if kv_config else 2
                
                model = load_qwen2_kivi(
                    model_name,
                    dtype=expert_dtype,
                    trust_remote_code=True,
                    k_bits=k_bits,
                    v_bits=v_bits,
                    group_size=32,
                    residual_length=32,
                )
                print(f"    [KIVI] Loaded with k_bits={k_bits}, v_bits={v_bits}")
            except Exception as e:
                print(f"    [KIVI] Failed: {e}, using standard model")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=expert_dtype,
                    trust_remote_code=True,
                )
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=expert_dtype,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=expert_dtype,
                    trust_remote_code=True,
                )
        
        model = model.to(device)
        model.eval()
        models.append(model)
    
    return models, tokenizer


# ============== Main Experiment ==============

def run_ablation_experiment(
    exp_config: Dict[str, Any],
    experts: List[torch.nn.Module],
    tokenizer,
    samples: List[Dict],
    seed: int = 0,
    device: torch.device = None,
    sample_timeout: int = 120,
    drift_tracker=None,
    is_reference: bool = False,
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Run a single ablation experiment.
    
    Args:
        experts: List of expert models (can be single model wrapped in list)
        drift_tracker: Optional DriftTracker instance for measuring KL/cosine drift
        is_reference: If True, record logits as reference baseline
    """
    from dlmom.core.loop import DLMoMLoop
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(seed)
    
    exp_id = exp_config.get("id", "unknown")
    
    # Extract config parameters with explicit None handling
    alpha = exp_config.get("alpha") if exp_config.get("alpha") is not None else 0.6
    k = exp_config.get("k") if exp_config.get("k") is not None else 50
    cap = exp_config.get("cap") if exp_config.get("cap") is not None else 10
    w = exp_config.get("w") if exp_config.get("w") is not None else 5
    tau = exp_config.get("tau") if exp_config.get("tau") is not None else 1.0
    lambda_val = exp_config.get("lambda")  # Can be None
    soft_mode = exp_config.get("soft_mode") if exp_config.get("soft_mode") is not None else "deterministic"
    kv_method = exp_config.get("kv") if exp_config.get("kv") is not None else "none"
    max_steps = exp_config.get("max_steps") if exp_config.get("max_steps") is not None else 40
    max_new_tokens = exp_config.get("max_new_tokens") if exp_config.get("max_new_tokens") is not None else 256
    
    # Gate mode for A2 experiments (threshold vs trend-based controller)
    gate_mode = exp_config.get("gate", "trend")
    
    # Communication mode for A3 experiments (embed/belief/logits)
    comm_mode = exp_config.get("comm", "belief")
    
    # Configure stochasticity
    gumbel_noise = (soft_mode == "gumbel")
    gumbel_tau = tau if gumbel_noise else 1.0
    dirichlet_lambda = lambda_val if soft_mode == "dirichlet" else None
    
    # Handle cap=-1 (unlimited)
    if cap < 0:
        cap = 1000
    
    # Create loop with all experts, gate_mode, and comm_mode
    loop = DLMoMLoop(
        experts=experts,
        tokenizer=tokenizer,
        top_k=k,
        alpha=alpha,
        window_size=w,
        switch_cap=cap,
        max_steps=max_steps,
        max_new_tokens=max_new_tokens,
        gumbel_noise=gumbel_noise,
        gumbel_tau=gumbel_tau,
        dirichlet_lambda=dirichlet_lambda,
        kv_method=kv_method,
        gate_mode=gate_mode,
        comm_mode=comm_mode,
    )
    
    # Run on samples
    results = []
    total_correct = 0
    total_time = 0
    timeouts = 0
    errors = 0
    
    for i, sample in tqdm(enumerate(samples), total=len(samples), desc=f"{exp_id} (s={seed})"):
        question = sample["question"]
        # Gold answer is already clean from load_samples() - just use it directly
        # For GSM8K/MATH, apply extraction; for others, it's already the final answer
        extractor_type = sample["extractor"]
        if extractor_type == "gsm8k":
            gold_answer = extract_gsm8k_answer(sample["answer"])
        elif extractor_type == "math":
            # MATH solutions contain \boxed{answer} - extract it
            gold_answer = extract_answer(sample["answer"], "math")
        else:
            gold_answer = sample["answer"]  # Already cleaned in load_samples()
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Prepare prompt
        prompt_text = None
        
        # 1. Try Chat Template (Standard for Instruct models)
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                # Force CoT by pre-filling assistant response
                messages = [
                    {"role": "user", "content": question},
                ]
                # Some tokenizers (like Qwen) handle generation prompts automatically
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # Append CoT trigger if not present
                if "step by step" not in prompt_text[-20:]: 
                     prompt_text += "Let's think step by step."
            except Exception as e:
                print(f"Chat template failed: {e}")
                prompt_text = None

        # 2. Fallback for Base Models / Missing Templates
        if prompt_text is None:
            # Generic format often works for base models + Phi-2
            prompt_text = f"Question: {question}\nLet's think step by step.\nAnswer:"

        if i == 0:
            pass  # Debug prompt print removed for cleaner output
        
        # Tokenize (using correct prompt_text)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run with timeout
        start = time.time()
        failure_mode = None
        
        # Collect logits if drift tracking is enabled
        should_collect_logits = drift_tracker is not None
        
        try:
            with timeout(sample_timeout):
                generated_text, loop_result = loop.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    collect_logits=should_collect_logits,
                )
        except SampleTimeoutError:
            failure_mode = "timeout"
            timeouts += 1
            generated_text = ""
            loop_result = None
        except Exception as e:
            # import traceback; traceback.print_exc()  # Debug print
            error_msg = str(e)
            if "NaN or Inf" in error_msg:
                failure_mode = "nan_detected"
            else:
                failure_mode = f"error: {error_msg[:100]}"
            errors += 1
            generated_text = ""
            loop_result = None
        
        # Handle drift tracking
        sample_kl_drift = None
        sample_cosine_drift = None
        sample_drift_events = None
        if drift_tracker is not None and loop_result is not None and loop_result.step_logits:
            sample_key = f"{seed}_{i}"
            if is_reference:
                # Record reference logits
                drift_tracker.record_reference(sample_key, loop_result.step_logits)
            else:
                # Compare against reference
                sample_kl_drift, sample_cosine_drift, sample_drift_events = drift_tracker.compute_drift(
                    sample_key, loop_result.step_logits
                )
        
        elapsed = time.time() - start
        total_time += elapsed
        
        # generated_text is already set from loop.generate()
        # generated_text = loop_result.generated_text if loop_result else ""
        
        # Use generic extractor based on benchmark config
        extractor_type = samples[i]["extractor"]
        pred_answer = extract_answer(generated_text, extractor_type) if generated_text else None
        
        # Evaluate correctness - use code execution for code benchmarks
        if extractor_type == "code" and generated_text and loop_result:
            # Pass@k: Execute generated code with test cases
            test_code = samples[i].get("test_code", "")
            entry_point = samples[i].get("entry_point")
            prompt = samples[i].get("prompt", "")
            
            # For HumanEval, we need to prepend the prompt (function signature)
            if entry_point:
                # HumanEval: The model often outputs explanation then a complete code block
                # extract_code_block will find the ```python block which has complete function
                # So we pass generated_text directly, not prompt + generated_text
                full_code = generated_text  # extract_code_block will find the code
                benchmark_type = "humaneval"
            else:
                # MBPP: just the generated code
                full_code = generated_text
                benchmark_type = "mbpp"
            
            passed, error_msg = execute_code_with_tests(
                full_code, test_code, entry_point,
                timeout_seconds=10, benchmark_type=benchmark_type
            )
            correct = passed
            pred_answer = "PASS" if passed else f"FAIL: {error_msg[:100]}"
        else:
            correct = answers_match(pred_answer, gold_answer) if loop_result else False
        
        total_correct += int(correct)
        
        # Check for decode failure (loop succeeded but no answer found)
        if failure_mode is None and generated_text and pred_answer is None:
            failure_mode = "decode_failure"
        
        # Get memory usage
        peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        # Build result dict
        result_dict = {
            "sample_id": i,
            "correct": correct,
            "pred_answer": pred_answer,
            "gold_answer": gold_answer,
            "generated_text": generated_text, # Added logging
            "wall_time_s": elapsed,
            "peak_mem_gb": peak_mem,
            "failure_mode": failure_mode,
        }
        
        if loop_result:
            result_dict.update({
                "total_steps": loop_result.total_steps,
                "latent_steps": loop_result.latent_steps,
                "explicit_tokens": loop_result.explicit_tokens,
                "switches": loop_result.switches,
                "stopped_reason": loop_result.stopped_reason,
                "top_k_mass": loop_result.final_packet.top_k_mass,
                "time_forward_s": sum(s.get("time_forward", 0) for s in loop_result.step_log),
                "time_consensus_s": sum(s.get("time_consensus", 0) for s in loop_result.step_log),
                "time_comm_s": sum(s.get("time_comm", 0) for s in loop_result.step_log),
                "avg_bridge_rate": np.mean([s.get("bridge_rate", 0) for s in loop_result.step_log]) if loop_result.step_log else 0,
                "kv_stats": loop_result.kv_compression_stats,
                "entropy_trace": [{"step": s["step"], "entropy": s["entropy"], "mode": s["mode"]} for s in loop_result.step_log],
                "drift_events": sample_drift_events if drift_tracker else None,
            })
            
            # Calculate bandwidth metrics (estimated)
            # A3 ablation metric: bytes per step
            # A3 ablation metric: bytes per step
            comm_mode = exp_config.get("comm", "belief")
            k_val = exp_config.get("k", 50)
            
            bytes_per_step = 0
            if comm_mode == "embed":
                # hidden_dim * 2 bytes (float16)
                hidden_dim = experts[0].config.hidden_size if experts else 4096 # fallback
                bytes_per_step = hidden_dim * 2
            elif comm_mode == "belief":
                # k * (4 + 4) bytes (int32 ids + float32 probs)
                bytes_per_step = k_val * 8
            elif comm_mode == "logits":
                # vocab * 4 bytes (float32)
                vocab_size = experts[0].config.vocab_size if experts else 32000
                bytes_per_step = vocab_size * 4
            
            result_dict["bytes_per_step"] = bytes_per_step
            result_dict["total_bytes"] = bytes_per_step * loop_result.latent_steps
        
        results.append(result_dict)
    
    # Compute summary statistics
    valid_results = [r for r in results if r["failure_mode"] is None]
    
    def safe_mean(key):
        vals = [r[key] for r in valid_results if key in r]
        return float(np.mean(vals)) if vals else 0
    
    def safe_std(key):
        vals = [r[key] for r in valid_results if key in r]
        return float(np.std(vals)) if vals else 0
    
    summary = {
        "exp_id": exp_id,
        "seed": seed,
        "config": exp_config,
        "n_samples": len(samples),
        "n_valid": len(valid_results),
        "n_timeouts": timeouts,
        "n_errors": errors,
        
        # Primary metrics
        "accuracy": total_correct / len(samples),
        "timeout_rate": timeouts / len(samples),
        "error_rate": errors / len(samples),
        
        # Timing
        "avg_wall_time_s": safe_mean("wall_time_s"),
        "std_wall_time_s": safe_std("wall_time_s"),
        "p50_wall_time_s": float(np.percentile([r["wall_time_s"] for r in valid_results], 50)) if valid_results else 0,
        "p90_wall_time_s": float(np.percentile([r["wall_time_s"] for r in valid_results], 90)) if valid_results else 0,
        "total_wall_time_s": total_time,
        
        # Throughput
        "samples_per_sec": len(samples) / total_time if total_time > 0 else 0,
        "steps_per_sec": sum(r["total_steps"] for r in valid_results) / total_time if total_time > 0 and valid_results else 0,
        
        # Loop behavior
        "avg_total_steps": safe_mean("total_steps"),
        "avg_latent_steps": safe_mean("latent_steps"),
        "avg_explicit_tokens": safe_mean("explicit_tokens"),
        "avg_switches": safe_mean("switches"),
        "std_switches": safe_std("switches"),
        "avg_top_k_mass": safe_mean("top_k_mass"),
        
        # Memory
        "avg_peak_mem_gb": safe_mean("peak_mem_gb"),
        "max_peak_mem_gb": max([r["peak_mem_gb"] for r in results]) if results else 0,
        
        # Communication
        "avg_bridge_rate": safe_mean("avg_bridge_rate"),
        "avg_bytes_per_step": safe_mean("bytes_per_step"),
        "avg_bytes_total": safe_mean("total_bytes"),
        
        "timestamp": datetime.now().isoformat(),
    }
    
    # Add drift statistics if tracker was used
    # Add drift statistics if tracker was used
    if drift_tracker is not None and not is_reference:
        kl_drift, cosine_drift, drift_events = drift_tracker.get_aggregate_drift()
        summary["kl_drift"] = kl_drift
        summary["cosine_drift"] = cosine_drift
        summary["drift_events"] = drift_events
        drift_tracker.reset_aggregates()  # Reset for next experiment
    
    return summary, results


def main():
    parser = argparse.ArgumentParser(description="DL-MoM Ablation Runner")
    parser.add_argument("--suite", required=True, help="Suite name (A1, A2, etc.)")
    parser.add_argument("--exp-id", help="Specific experiment ID")
    parser.add_argument("--bench", default="gsm8k", help="Benchmark name")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds")
    parser.add_argument("--model", default="gpt2", help="Model to use")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--timeout", type=int, default=120, help="Per-sample timeout")
    args = parser.parse_args()
    
    import yaml
    
    # Load config
    config_path = Path(__file__).parent / "configs" / "suites.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "configs" / "suites.yaml"
    
    if not config_path.exists():
        print(f"Error: Suite config not found at {config_path}")
        print("Expected locations:")
        print(f"  - {Path(__file__).parent / 'configs' / 'suites.yaml'}")
        print(f"  - {Path(__file__).parent.parent / 'configs' / 'suites.yaml'}")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    defaults = config.get("defaults", {})
    full_suite = config.get("suites", {}).get(args.suite, [])
    experiments = full_suite
    
    if args.exp_id:
        experiments = [e for e in experiments if e.get("id") == args.exp_id]
    
    if not experiments:
        print(f"No experiments found for suite {args.suite}")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if any experiment uses KIVI compression
    uses_kivi = any(
        exp.get("kv") in ["kivi2bit", "minicache+kivi2bit"]
        for exp in experiments
    )
    
    # Check if any experiment uses heterogeneous experts (A5 suite)
    uses_heterogeneous_experts = any(
        exp.get("experts") is not None
        for exp in experiments
    )
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if uses_heterogeneous_experts:
        # For heterogeneous expert experiments, load models per-experiment in the loop
        # Use tokenizer from the CLI model as fallback
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        experts = None  # Will be loaded per-experiment
    else:
        # Standard single-model loading
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load KIVI-enabled model if any experiment uses KIVI compression
        if uses_kivi and "qwen" in args.model.lower():
            print(f"[KIVI] Loading Qwen2ForCausalLM_KIVI for native compression...")
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "kivi"))
                from models.qwen_kivi import load_qwen2_kivi
                model = load_qwen2_kivi(
                    args.model,
                    dtype=torch.float16,  # KIVI kernels require Half precision
                    trust_remote_code=True,
                    k_bits=2,
                    v_bits=2,
                    group_size=32,
                    residual_length=32,
                )
                print(f"[KIVI] Successfully loaded {type(model).__name__}")
            except Exception as e:
                print(f"[KIVI] Failed to load KIVI model: {e}, falling back to standard model")
                uses_kivi = False
        
        if not uses_kivi or "qwen" not in args.model.lower():
            # Use SDPA (AOTriton-backed on ROCm) for better attention performance
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                )
        
        model = model.to(device)
        model.eval()
        
        # Ensure pad_token_id is set to avoid warnings
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            if model.generation_config.pad_token_id is None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id
        
        # Wrap single model in list for consistency with multi-expert API
        experts = [model]
    
    # Load samples
    samples = load_samples(args.bench, args.samples)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary").mkdir(exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    
    # Import rich CLI
    try:
        from dlmom.rich_cli import DLMoMCLI, ExperimentResult, Status
        use_rich = True
    except ImportError:
        use_rich = False
        print("[Warning] rich_cli not available, using basic output")
    
    # Initialize CLI
    if use_rich:
        cli = DLMoMCLI(quiet=False)
        cli.header(
            suite=args.suite,
            benchmark=args.bench,
            model=args.model,
            samples=args.samples,
            seeds=args.seeds,
            experiments=[e['id'] for e in experiments],
            device=str(device),
            reference=experiments[0]['id'] if experiments else None,
            configs=[{**defaults, **e} for e in experiments],
        )
    
    # Create drift tracker for measuring distribution shift
    from dlmom.core.drift_tracker import DriftTracker
    drift_tracker = DriftTracker(storage_dir=output_dir / "drift_refs")
    
    # Run experiments
    all_summaries = []
    start_time = time.time()
    
    # Track which experiments are reference (first exp is reference for each seed)
    reference_exp_id = None
    if full_suite: # Use full suite to find the canonical reference (e.g. A1.1)
        reference_exp_id = full_suite[0]['id']
    elif experiments: # Fallback
        reference_exp_id = experiments[0]['id']
    
    for exp in experiments:
        merged_exp = {**defaults, **exp}
        
        # Validate critical parameters for A3
        if "comm" in merged_exp and merged_exp["comm"] not in ["belief", "embed", "logits"]:
             print(f"Warning: Invalid comm mode {merged_exp['comm']}, defaulting to belief")
             merged_exp["comm"] = "belief"
             
        # Export experiment config per paper requirement
        config_dump_path = output_dir / "configs" / f"{merged_exp.get('id', 'unknown')}.yaml"
        config_dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_dump_path, 'w') as f:
            yaml.dump(merged_exp, f)
        
        # For heterogeneous experts (A5), load experts per-experiment
        exp_experts = experts
        exp_tokenizer = tokenizer
        if merged_exp.get("experts") is not None:
            expert_names = merged_exp["experts"]
            print(f"[A5] Loading {len(expert_names)} heterogeneous experts for {exp['id']}...")
            kv_config = {"method": merged_exp.get("kv", "none"), "k_bits": 2, "v_bits": 2}
            exp_experts, exp_tokenizer = load_experts(expert_names, device, kv_config=kv_config)
        
        # For A4 KV compression experiments, reload model if kv_method changes
        exp_kv_method = merged_exp.get("kv", "none")
        if args.suite and args.suite.upper() == "A4":
            # Build kv_config from experiment settings
            # For combined compression (minicache+kivi), use standard model with kv_compressor
            # - "minicache" or "minicache+kivi*": use standard model (TokenMerge needs raw cache)
            # - "kivi2bit" or "kivi4bit" only: use KIVI native model
            use_native_kivi = exp_kv_method in ["kivi2bit", "kivi4bit"]  # NOT minicache+kivi
            
            kv_config = {
                "method": exp_kv_method if use_native_kivi else "none",  # Only KIVI model if pure KIVI
                "k_bits": 4 if "4bit" in exp_kv_method else 2,
                "v_bits": 4 if "4bit" in exp_kv_method else 2,
            }
            
            # Check if we need to reload model
            needs_reload = (
                (use_native_kivi and not uses_kivi) or  # Need KIVI, don't have it
                (not use_native_kivi and uses_kivi) or  # Need standard, have KIVI
                exp_kv_method != "none"  # Any compression config change
            )
            if needs_reload:
                print(f"[A4] Reloading model for kv_method={exp_kv_method} (native_kivi={use_native_kivi})...")
                exp_experts, exp_tokenizer = load_experts([args.model], device, kv_config=kv_config)
        
        for seed in args.seeds:
            if not use_rich:
                print(f"\n>>> Running {exp['id']} (seed={seed})...")
            
            # First experiment is reference for drift tracking
            is_ref = (exp['id'] == reference_exp_id)
            
            summary, results = run_ablation_experiment(
                exp_config=merged_exp,
                experts=exp_experts,
                tokenizer=exp_tokenizer,
                samples=samples,
                seed=seed,
                device=device,
                sample_timeout=args.timeout,
                drift_tracker=drift_tracker,
                is_reference=is_ref,
            )
            
            all_summaries.append(summary)
            
            # Save results
            summary_path = output_dir / "summary" / f"{exp['id']}_seed{seed}.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            
            raw_path = output_dir / "raw" / f"{exp['id']}_seed{seed}.jsonl"
            with open(raw_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            
            # Report results via CLI
            if use_rich:
                exp_result = ExperimentResult(
                    exp_id=exp['id'],
                    seed=seed,
                    samples=[],  # We don't need to populate this for display
                )
                exp_result.accuracy = summary['accuracy'] * 100
                exp_result.mean_latent = summary['avg_latent_steps']
                exp_result.mean_switches = summary['avg_switches']
                exp_result.timeouts = summary['n_timeouts']
                exp_result.failures = summary['n_errors']
                exp_result.total_time = summary['total_wall_time_s']
                exp_result.kl_drift = summary.get('kl_drift')
                exp_result.cosine_drift = summary.get('cosine_drift')
                
                # Auto-detect issues
                if exp_result.accuracy == 0 and exp_result.failures == 0:
                    exp_result.status = Status.WARNING
                elif exp_result.failures > args.samples * 0.2:
                    exp_result.status = Status.WARNING
                else:
                    exp_result.status = Status.SUCCESS
                
                cli.results.append(exp_result)
                cli._print_result_line(exp_result)
            else:
                print(f"    Accuracy: {summary['accuracy']:.1%}, "
                      f"Latent: {summary['avg_latent_steps']:.1f}, "
                      f"Switches: {summary['avg_switches']:.2f}, "
                      f"Timeouts: {summary['n_timeouts']}")
        
        # Clean up per-experiment experts to free GPU memory (for heterogeneous experiments)
        if merged_exp.get("experts") is not None and exp_experts is not None:
            for model in exp_experts:
                del model
            torch.cuda.empty_cache()
            exp_experts = None
    
    # Save combined summary
    combined_path = output_dir / "all_summaries.json"
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    
    total_time = time.time() - start_time
    
    # Print summary and footer
    if use_rich:
        cli.summary()
        cli.export(output_dir)
        cli.footer(str(output_dir), total_time)
    else:
        print(f"\n{'=' * 50}")
        print(f"Completed {len(all_summaries)} experiments")
        print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

