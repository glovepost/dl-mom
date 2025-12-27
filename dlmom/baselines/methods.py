"""
Baseline Methods for DL-MoM Comparison

Implements:
- Direct: Single-shot generation
- CoT: Chain-of-Thought prompting
- Self-Consistency: Multiple samples + majority vote
- Text-MAS: Text-based multi-agent with explicit communication
"""

import torch
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class BaselineResult:
    """Result from a baseline method."""
    answer: str
    raw_output: str
    latency_s: float
    tokens_generated: int
    method: str
    extra: Dict[str, Any] = None


class DirectBaseline:
    """
    Direct single-shot generation baseline.
    
    Simply generates answer directly from the question.
    """
    
    def __init__(self, model, tokenizer, max_new_tokens: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
    
    def run(self, question: str) -> BaselineResult:
        """Generate answer directly."""
        prompt = f"Question: {question}\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.time() - start
        
        generated = outputs[0, inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)
        answer = self._extract_answer(raw_output)
        
        return BaselineResult(
            answer=answer,
            raw_output=raw_output,
            latency_s=latency,
            tokens_generated=len(generated),
            method="direct",
        )
    
    def _extract_answer(self, text: str) -> str:
        """Extract numerical answer from text."""
        # Look for #### pattern (GSM8K format)
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
        if match:
            return match.group(1).replace(",", "")
        
        # Look for any number at the end
        numbers = re.findall(r"-?\d+\.?\d*", text)
        return numbers[-1] if numbers else ""


class CoTBaseline:
    """
    Chain-of-Thought baseline.
    
    Uses "Let's think step by step" prompting.
    """
    
    def __init__(self, model, tokenizer, max_new_tokens: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
    
    def run(self, question: str) -> BaselineResult:
        """Generate answer with CoT prompting."""
        prompt = f"Question: {question}\n\nLet's think step by step.\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.time() - start
        
        generated = outputs[0, inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)
        answer = self._extract_answer(raw_output)
        
        return BaselineResult(
            answer=answer,
            raw_output=raw_output,
            latency_s=latency,
            tokens_generated=len(generated),
            method="cot",
        )
    
    def _extract_answer(self, text: str) -> str:
        """Extract answer from CoT output."""
        # Look for "the answer is X" pattern
        match = re.search(r"(?:answer|result|=)\s*(?:is\s*)?\$?(-?[\d,]+\.?\d*)", text, re.I)
        if match:
            return match.group(1).replace(",", "")
        
        # Fallback: last number
        numbers = re.findall(r"-?\d+\.?\d*", text)
        return numbers[-1] if numbers else ""


class SelfConsistencyBaseline:
    """
    Self-Consistency baseline.
    
    Samples multiple CoT paths and aggregates via majority vote.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        n_samples: int = 5,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
    
    def run(self, question: str) -> BaselineResult:
        """Generate multiple samples and take majority vote."""
        prompt = f"Question: {question}\n\nLet's think step by step.\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        answers = []
        all_outputs = []
        total_tokens = 0
        
        start = time.time()
        with torch.no_grad():
            for _ in range(self.n_samples):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
                generated = outputs[0, inputs["input_ids"].shape[1]:]
                raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)
                
                all_outputs.append(raw_output)
                total_tokens += len(generated)
                
                answer = self._extract_answer(raw_output)
                if answer:
                    answers.append(answer)
        
        latency = time.time() - start
        
        # Majority vote
        final_answer = self._majority_vote(answers)
        
        return BaselineResult(
            answer=final_answer,
            raw_output="\n---\n".join(all_outputs),
            latency_s=latency,
            tokens_generated=total_tokens,
            method="self_consistency",
            extra={
                "n_samples": self.n_samples,
                "all_answers": answers,
                "unique_answers": list(set(answers)),
            },
        )
    
    def _extract_answer(self, text: str) -> str:
        """Extract answer from CoT output."""
        match = re.search(r"(?:answer|result|=)\s*(?:is\s*)?\$?(-?[\d,]+\.?\d*)", text, re.I)
        if match:
            return match.group(1).replace(",", "")
        
        numbers = re.findall(r"-?\d+\.?\d*", text)
        return numbers[-1] if numbers else ""
    
    def _majority_vote(self, answers: List[str]) -> str:
        """Return most common answer."""
        if not answers:
            return ""
        
        # Normalize (round floats)
        normalized = []
        for a in answers:
            try:
                val = float(a)
                normalized.append(str(int(val)) if val == int(val) else str(val))
            except ValueError:
                normalized.append(a)
        
        from collections import Counter
        counts = Counter(normalized)
        return counts.most_common(1)[0][0]


class TextMASBaseline:
    """
    Text-based Multi-Agent System baseline.
    
    Multiple "agents" discuss in text, then synthesize final answer.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        n_agents: int = 3,
        n_rounds: int = 2,
        max_new_tokens: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.max_new_tokens = max_new_tokens
    
    def run(self, question: str) -> BaselineResult:
        """Run multi-agent discussion."""
        agent_names = ["Analyst", "Checker", "Solver"][:self.n_agents]
        
        # Initial prompts for each agent
        discussion = []
        total_tokens = 0
        
        start = time.time()
        
        # Round 1: Initial responses
        for i, agent in enumerate(agent_names):
            prompt = self._build_prompt(question, agent, discussion, round_num=1)
            response, tokens = self._generate(prompt)
            
            discussion.append({
                "agent": agent,
                "round": 1,
                "response": response,
            })
            total_tokens += tokens
        
        # Subsequent rounds: Responses with context
        for round_num in range(2, self.n_rounds + 1):
            for i, agent in enumerate(agent_names):
                prompt = self._build_prompt(question, agent, discussion, round_num)
                response, tokens = self._generate(prompt)
                
                discussion.append({
                    "agent": agent,
                    "round": round_num,
                    "response": response,
                })
                total_tokens += tokens
        
        # Final synthesis
        synthesis_prompt = self._build_synthesis_prompt(question, discussion)
        final_response, tokens = self._generate(synthesis_prompt)
        total_tokens += tokens
        
        latency = time.time() - start
        
        answer = self._extract_answer(final_response)
        
        # Format discussion as raw output
        raw_output = ""
        for d in discussion:
            raw_output += f"[{d['agent']} R{d['round']}]: {d['response']}\n\n"
        raw_output += f"[Final]: {final_response}"
        
        return BaselineResult(
            answer=answer,
            raw_output=raw_output,
            latency_s=latency,
            tokens_generated=total_tokens,
            method="text_mas",
            extra={
                "n_agents": self.n_agents,
                "n_rounds": self.n_rounds,
                "discussion": discussion,
            },
        )
    
    def _build_prompt(
        self,
        question: str,
        agent: str,
        discussion: List[Dict],
        round_num: int,
    ) -> str:
        """Build prompt for an agent."""
        prompt = f"You are {agent}, a helpful assistant.\n\n"
        prompt += f"Question: {question}\n\n"
        
        if discussion:
            prompt += "Previous discussion:\n"
            for d in discussion:
                prompt += f"- {d['agent']}: {d['response'][:200]}...\n"
            prompt += "\n"
        
        if round_num == 1:
            prompt += f"Provide your initial analysis.\n"
        else:
            prompt += f"Given the discussion, provide your updated thoughts.\n"
        
        prompt += f"\n{agent}:"
        return prompt
    
    def _build_synthesis_prompt(
        self,
        question: str,
        discussion: List[Dict],
    ) -> str:
        """Build final synthesis prompt."""
        prompt = f"Question: {question}\n\n"
        prompt += "Team discussion:\n"
        for d in discussion:
            prompt += f"- {d['agent']}: {d['response'][:200]}...\n"
        prompt += "\nBased on the team's analysis, the final answer is:"
        return prompt
    
    def _generate(self, prompt: str) -> tuple:
        """Generate text and return (text, tokens)."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = outputs[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text, len(generated)
    
    def _extract_answer(self, text: str) -> str:
        """Extract answer."""
        numbers = re.findall(r"-?\d+\.?\d*", text)
        return numbers[-1] if numbers else ""


def get_baseline(method: str, model, tokenizer, **kwargs):
    """Factory function for baselines."""
    if method == "direct":
        return DirectBaseline(model, tokenizer, **kwargs)
    elif method == "cot":
        return CoTBaseline(model, tokenizer, **kwargs)
    elif method == "self_consistency":
        return SelfConsistencyBaseline(model, tokenizer, **kwargs)
    elif method == "text_mas":
        return TextMASBaseline(model, tokenizer, **kwargs)
    else:
        raise ValueError(f"Unknown baseline method: {method}")
