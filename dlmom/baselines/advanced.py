"""
Advanced Baseline Methods for DL-MoM Comparison

Implements:
- ToT: Tree-of-Thought with self-evaluation
- LatentMAS: Integration with external LatentMAS repository
"""

import torch
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path


@dataclass
class BaselineResult:
    """Result from a baseline method."""
    answer: str
    raw_output: str
    latency_s: float
    tokens_generated: int
    method: str
    extra: Dict[str, Any] = None


class TreeOfThoughtBaseline:
    """
    Tree-of-Thought (ToT) baseline.
    
    Explores multiple reasoning paths using BFS/DFS with self-evaluation.
    Inspired by Yao et al. (2023) "Tree of Thoughts: Deliberate Problem Solving
    with Large Language Models"
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        n_branches: int = 3,
        max_depth: int = 3,
        eval_samples: int = 3,
        temperature: float = 0.8,
        max_new_tokens: int = 256,
    ):
        """
        Args:
            model: HuggingFace model
            tokenizer: Tokenizer
            n_branches: Number of branches to explore at each step
            max_depth: Maximum depth of the thought tree
            eval_samples: Number of samples for self-evaluation
            temperature: Generation temperature
            max_new_tokens: Max tokens per generation step
        """
        self.model = model
        self.tokenizer = tokenizer
        self.n_branches = n_branches
        self.max_depth = max_depth
        self.eval_samples = eval_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
    
    def run(self, question: str) -> BaselineResult:
        """Run Tree-of-Thought reasoning."""
        start = time.time()
        total_tokens = 0
        
        # Root node
        root_prompt = f"Question: {question}\n\nLet me break this down step by step.\n\nStep 1:"
        
        # BFS exploration
        current_level = [(root_prompt, [])]  # (prompt, path)
        all_paths = []
        
        for depth in range(self.max_depth):
            next_level = []
            
            for prompt, path in current_level:
                # Generate branches
                branches, tokens = self._generate_branches(prompt)
                total_tokens += tokens
                
                for branch in branches:
                    new_path = path + [branch]
                    new_prompt = prompt + " " + branch
                    
                    if depth < self.max_depth - 1:
                        new_prompt += f"\n\nStep {depth + 2}:"
                        next_level.append((new_prompt, new_path))
                    else:
                        # Leaf node - add conclusion
                        final_prompt = new_prompt + "\n\nTherefore, the answer is:"
                        conclusion, tokens = self._generate_single(final_prompt)
                        total_tokens += tokens
                        all_paths.append((new_path + [conclusion], new_prompt))
            
            current_level = next_level
            
            # Prune to top-k based on self-evaluation
            if len(current_level) > self.n_branches:
                current_level = self._prune_paths(current_level, question)
        
        # Select best path via self-evaluation
        best_path = self._select_best_path(all_paths, question)
        
        latency = time.time() - start
        
        # Extract answer
        raw_output = " ".join(best_path[0]) if best_path else ""
        answer = self._extract_answer(raw_output)
        
        return BaselineResult(
            answer=answer,
            raw_output=raw_output,
            latency_s=latency,
            tokens_generated=total_tokens,
            method="tree_of_thought",
            extra={
                "n_paths_explored": len(all_paths),
                "depth": self.max_depth,
                "branches": self.n_branches,
            },
        )
    
    def _generate_branches(self, prompt: str) -> Tuple[List[str], int]:
        """Generate multiple thought branches."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        branches = []
        total_tokens = 0
        
        with torch.no_grad():
            for _ in range(self.n_branches):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
                generated = outputs[0, inputs["input_ids"].shape[1]:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                
                # Stop at newline (one step at a time)
                text = text.split("\n")[0].strip()
                branches.append(text)
                total_tokens += len(generated)
        
        return branches, total_tokens
    
    def _generate_single(self, prompt: str) -> Tuple[str, int]:
        """Generate a single completion."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = outputs[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text, len(generated)
    
    def _prune_paths(
        self,
        paths: List[Tuple[str, List[str]]],
        question: str,
    ) -> List[Tuple[str, List[str]]]:
        """Prune paths based on self-evaluation scores."""
        if len(paths) <= self.n_branches:
            return paths
        
        scored = []
        for prompt, path in paths:
            score = self._evaluate_path(prompt, question)
            scored.append((score, prompt, path))
        
        # Sort by score descending
        scored.sort(reverse=True, key=lambda x: x[0])
        return [(p, path) for _, p, path in scored[:self.n_branches]]
    
    def _evaluate_path(self, path_prompt: str, question: str) -> float:
        """Self-evaluate a reasoning path (0-1 score)."""
        eval_prompt = f"""Question: {question}

Reasoning so far:
{path_prompt}

Rate this reasoning on a scale of 1-10 (1=wrong, 10=correct):
Rating:"""
        
        inputs = self.tokenizer(eval_prompt, return_tensors="pt", truncation=True, max_length=1024)
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = outputs[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        # Extract rating
        numbers = re.findall(r"\d+", text)
        if numbers:
            try:
                rating = min(10, max(1, int(numbers[0])))
                return rating / 10.0
            except:
                pass
        return 0.5  # Default
    
    def _select_best_path(
        self,
        all_paths: List[Tuple[List[str], str]],
        question: str,
    ) -> Optional[Tuple[List[str], str]]:
        """Select best path from all completed paths."""
        if not all_paths:
            return None
        
        if len(all_paths) == 1:
            return all_paths[0]
        
        # Evaluate each final path
        scored = []
        for path, prompt in all_paths:
            score = self._evaluate_path(prompt, question)
            scored.append((score, path, prompt))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return (scored[0][1], scored[0][2])
    
    def _extract_answer(self, text: str) -> str:
        """Extract answer from text."""
        numbers = re.findall(r"-?\d+\.?\d*", text)
        return numbers[-1] if numbers else ""


class LatentMASWrapper:
    """
    Wrapper for the external LatentMAS repository.
    
    Integrates with the LatentMAS method for multi-agent latent-space reasoning.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        latent_steps: int = 10,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: Device to run on
            latent_steps: Number of latent reasoning steps
            max_new_tokens: Max tokens for generation
            temperature: Generation temperature
        """
        self.model_name = model_name
        self.device = device
        self.latent_steps = latent_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Import from external repo
        latentmas_path = Path(__file__).parent.parent.parent / "external" / "LatentMAS"
        sys.path.insert(0, str(latentmas_path))
        
        self._method = None
        self._model = None
    
    def _lazy_init(self):
        """Lazy initialization of LatentMAS components."""
        if self._method is not None:
            return
        
        try:
            import argparse
            from models import ModelWrapper
            from methods.latent_mas import LatentMASMethod
            
            # Create minimal args
            args = argparse.Namespace(
                model_name=self.model_name,
                device=self.device,
                device2=self.device,
                latent_steps=self.latent_steps,
                max_new_tokens=self.max_new_tokens,
                latent_space_realign=False,
                use_second_HF_model=False,
                latent_only=False,
                sequential_info_only=True,
                enable_prefix_caching=False,
                method="latent_mas",
                prompt="sequential",
                think=False,
                task="gsm8k",
            )
            
            self._model = ModelWrapper(
                self.model_name,
                torch.device(self.device),
                use_vllm=False,
                args=args,
            )
            
            self._method = LatentMASMethod(
                self._model,
                latent_steps=self.latent_steps,
                judger_max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                generate_bs=1,
                args=args,
            )
        except ImportError as e:
            raise ImportError(f"Failed to import LatentMAS: {e}")
    
    def run(self, question: str, gold: str = "") -> BaselineResult:
        """Run LatentMAS on a question."""
        self._lazy_init()
        
        start = time.time()
        
        item = {
            "question": question,
            "gold": gold,
            "solution": "",
        }
        
        result = self._method.run_item(item)
        latency = time.time() - start
        
        # Count tokens (approximate from trace)
        total_tokens = 0
        for agent in result.get("agents", []):
            total_tokens += len(agent.get("input_ids", []))
        
        return BaselineResult(
            answer=result.get("prediction", ""),
            raw_output=result.get("raw_prediction", ""),
            latency_s=latency,
            tokens_generated=total_tokens,
            method="latent_mas",
            extra={
                "correct": result.get("correct", False),
                "agents": result.get("agents", []),
                "latent_steps": self.latent_steps,
            },
        )


# Simplified LatentMAS for non-vLLM environments
class LatentMASSimple:
    """
    Simplified LatentMAS implementation that doesn't require vLLM.
    
    Uses standard HuggingFace transformers for latent-space reasoning.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        n_agents: int = 3,
        latent_steps: int = 5,
        max_new_tokens: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_agents = n_agents
        self.latent_steps = latent_steps
        self.max_new_tokens = max_new_tokens
    
    def run(self, question: str) -> BaselineResult:
        """Run simplified LatentMAS."""
        start = time.time()
        total_tokens = 0
        
        # Build initial prompt
        prompt = f"Question: {question}\n\nLet's think through this step by step.\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Initial forward to get hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
            )
        
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        # Latent reasoning steps (simulate multi-agent)
        agent_hiddens = []
        for agent_idx in range(self.n_agents):
            # Each agent processes latent steps
            agent_hidden = last_hidden
            
            for _ in range(self.latent_steps):
                # Use hidden state as input embedding
                latent_embed = agent_hidden.unsqueeze(1)
                
                past_len = past[0][0].shape[2]
                attn_mask = torch.ones(1, past_len + 1, device=latent_embed.device)
                
                outputs = self.model(
                    inputs_embeds=latent_embed,
                    attention_mask=attn_mask,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                past = outputs.past_key_values
                agent_hidden = outputs.hidden_states[-1][:, -1, :]
            
            agent_hiddens.append(agent_hidden)
            total_tokens += self.latent_steps
        
        # Average agent hiddens for final reasoning
        combined = torch.stack(agent_hiddens, dim=0).mean(dim=0)
        
        # Generate final answer
        with torch.no_grad():
            # Use combined hidden as starting point
            final_embed = combined.unsqueeze(1)
            past_len = past[0][0].shape[2]
            attn_mask = torch.ones(1, past_len + 1, device=final_embed.device)
            
            # Generate tokens
            generated = []
            for _ in range(self.max_new_tokens):
                outputs = self.model(
                    inputs_embeds=final_embed,
                    attention_mask=attn_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                
                logits = outputs.logits[:, -1, :]
                next_token = logits.argmax(dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated.append(next_token.item())
                past = outputs.past_key_values
                
                # Get embedding for next input
                final_embed = self.model.get_input_embeddings()(next_token).unsqueeze(1)
                attn_mask = torch.ones(1, past[0][0].shape[2] + 1, device=final_embed.device)
                total_tokens += 1
        
        latency = time.time() - start
        
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)
        answer = self._extract_answer(raw_output)
        
        return BaselineResult(
            answer=answer,
            raw_output=raw_output,
            latency_s=latency,
            tokens_generated=total_tokens,
            method="latent_mas_simple",
            extra={
                "n_agents": self.n_agents,
                "latent_steps": self.latent_steps,
            },
        )
    
    def _extract_answer(self, text: str) -> str:
        """Extract answer from text."""
        numbers = re.findall(r"-?\d+\.?\d*", text)
        return numbers[-1] if numbers else ""


def get_advanced_baseline(method: str, model, tokenizer, **kwargs):
    """Factory function for advanced baselines."""
    if method == "tree_of_thought":
        return TreeOfThoughtBaseline(model, tokenizer, **kwargs)
    elif method == "latent_mas_simple":
        return LatentMASSimple(model, tokenizer, **kwargs)
    else:
        raise ValueError(f"Unknown advanced baseline: {method}")
