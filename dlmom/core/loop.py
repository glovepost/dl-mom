"""
Main DL-MoM Collaboration Loop.

Implements Algorithm 1 from the DL-MoM paper:
1. Latent Phase: Experts communicate via belief packets until entropy converges
2. Explicit Phase: Primary expert generates tokens from the final belief state

Stopping conditions: EOS token, max_steps reached, entropy convergence, or entropy plateau.
"""

import logging
import torch
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time

from transformers.cache_utils import DynamicCache

from .belief_packet import (
    BeliefPacket, 
    belief_packet_from_logits, 
    reconstruct_soft_input,
    apply_dirichlet_noise,
    get_comm_input,
)
from .entropy_controller import EntropyController, normalized_entropy
from .consensus import ContrastiveConsensus
from .kv_compression import get_compressor, KVCompressor
from .minicache import MiniCache

logger = logging.getLogger(__name__)


@dataclass
class StepOutput:
    """Output from one step of the collaboration loop."""
    packet: BeliefPacket
    entropy: float
    mode: str
    expert_entropies: List[float]
    bridge_rate: float
    time_forward: float
    time_consensus: float
    time_comm: float


@dataclass 
class LoopResult:
    """Final result from the collaboration loop."""
    final_packet: BeliefPacket
    total_steps: int
    latent_steps: int
    explicit_tokens: int
    switches: int
    wall_time: float
    step_log: List[Dict[str, Any]]
    kv_compression_stats: Dict[str, Any]
    stopped_reason: str  # "eos", "max_steps", "converged", "plateau"
    final_kv_caches: Optional[List[Tuple]] = None  # Persist context for generation
    step_logits: Optional[List[torch.Tensor]] = None  # For drift tracking


class DLMoMLoop:
    """
    Deep-Latent Mixture of Models collaboration loop.
    
    Coordinates multiple expert models to reason in latent space,
    using belief packets for communication and TIES consensus for merging.
    """
    
    def __init__(
        self,
        experts: List[torch.nn.Module],
        tokenizer,
        top_k: int = 50,
        alpha: float = 0.60,
        window_size: int = 5,
        switch_cap: int = 10,
        max_steps: int = 40,
        max_new_tokens: int = 256,
        trim_threshold: float = 0.1,
        gumbel_noise: bool = False,
        gumbel_tau: float = 1.0,
        dirichlet_lambda: Optional[float] = None,
        kv_method: str = "none",
        repetition_penalty: float = 1.2,
        gate_mode: str = "trend",
        comm_mode: str = "belief",
    ):
        """
        Args:
            experts: List of expert models (HuggingFace CausalLM). At least one required.
            tokenizer: Shared tokenizer
            top_k: Number of tokens in belief packets
            alpha: Entropy threshold for mode switching (see EntropyController)
            window_size: Entropy trend window size
            switch_cap: Maximum mode switches
            max_steps: Maximum latent steps
            max_new_tokens: Maximum explicit tokens to generate
            trim_threshold: TIES trim threshold
            gumbel_noise: Use Gumbel-Softmax for stochastic exploration
            gumbel_tau: Gumbel temperature
            dirichlet_lambda: Dirichlet noise concentration (None = no noise)
            kv_method: KV compression method ("none", "kivi", etc.)
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty)
            gate_mode: Controller mode - "threshold" for simple alpha threshold, "trend" for trend-based
            comm_mode: Communication protocol - "embed", "belief", or "logits" (Paper A3 suite)
        """
        if not experts:
            raise ValueError("At least one expert model is required")
        
        self.experts = experts
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.gumbel_noise = gumbel_noise
        self.gumbel_tau = gumbel_tau
        self.dirichlet_lambda = dirichlet_lambda
        self.kv_method = kv_method
        self.repetition_penalty = repetition_penalty
        self.gate_mode = gate_mode
        self.comm_mode = comm_mode
        
        # Configure controller based on gate_mode
        # - "threshold": Pure alpha threshold (no trend analysis)
        # - "trend": Trend-based with slope detection (default)
        if gate_mode == "threshold":
            self.controller = EntropyController(
                alpha=alpha,
                window_size=window_size,
                switch_cap=switch_cap,
                max_steps=max_steps,
                slope_threshold=0.0,  # Disable slope-based switching
                plateau_variance=0.0,  # Disable plateau detection
            )
        else:
            self.controller = EntropyController(
                alpha=alpha,
                window_size=window_size,
                switch_cap=switch_cap,
                max_steps=max_steps,
            )
        
        self.consensus = ContrastiveConsensus(trim_threshold=trim_threshold)
        
        # KIVI Integration:
        # - If expert is Qwen2ForCausalLM_KIVI: cache is KIVICache (9-tuple), skip compress/decompress
        # - Otherwise: cache is standard (K,V) tuples, use kv_compressor
        self.uses_kivi_model = self._detect_kivi_model(experts[0])
        
        if self.uses_kivi_model:
            # KIVI model handles compression internally - don't double compress
            # But track that KIVI is active for stats reporting
            self.kv_compressor = get_compressor("none")
            self._kivi_active = True  # Track that KIVI compression is happening internally
        else:
            # For combined mode (minicache+kivi2bit), extract the KIVI part for quantization
            if "+" in kv_method:
                # e.g. "minicache+kivi2bit" -> use "kivi2bit" compressor
                parts = kv_method.split("+")
                kivi_part = next((p for p in parts if "kivi" in p), "none")
                self.kv_compressor = get_compressor(kivi_part)
            elif kv_method == "minicache":
                # Pure minicache: no quantization, just sequence reduction
                self.kv_compressor = get_compressor("none")
            else:
                self.kv_compressor = get_compressor(kv_method)
            self._kivi_active = False
        
        # MiniCache/TokenMerge: Sequence-length compression (applied to standard caches only)
        self._minicache_active = "minicache" in kv_method and not self.uses_kivi_model
        if self._minicache_active:
            self.minicache = MiniCache(merge_ratio=0.5, min_tokens_before_merge=64)
        else:
            self.minicache = None
    
    def _get_kv_compression_stats(self) -> Dict[str, Any]:
        """Get KV compression stats, accounting for KIVI models and MiniCache."""
        stats = {}
        
        if self._kivi_active:
            stats = {
                "method": self.kv_method,
                "native_kivi": True,
                "note": "Compression handled internally by KIVI attention layers"
            }
        else:
            stats = self.kv_compressor.get_stats()
        
        # Add MiniCache stats if active
        if self._minicache_active and self.minicache:
            minicache_stats = self.minicache.get_stats()
            stats["minicache"] = minicache_stats
        
        return stats
    
    def _detect_kivi_model(self, model) -> bool:

        """Check if model is a KIVI-enabled model that handles compression internally."""
        model_class_name = type(model).__name__
        return "KIVI" in model_class_name or hasattr(model, 'uses_kivi_cache')
    
    def _get_embedding_layer(self, expert) -> torch.nn.Module:
        """Get embedding layer from expert model.
        
        Returns:
            The embedding layer (nn.Embedding or compatible module with num_embeddings)
        """
        layer = None
        if hasattr(expert, "transformer"):
            layer = expert.transformer.wte
        elif hasattr(expert, "model") and hasattr(expert.model, "embed_tokens"):
            layer = expert.model.embed_tokens
        elif hasattr(expert, "get_input_embeddings"):
            layer = expert.get_input_embeddings()
        
        if layer is None:
            raise ValueError("Cannot find embedding layer")
        if not hasattr(layer, 'num_embeddings'):
            raise ValueError(f"Embedding layer {type(layer)} lacks num_embeddings attribute")
        return layer

    def _compress_kv(self, past_key_values: Optional[Tuple]) -> Optional[Tuple]:
        """Compress KV cache using configured method.
        
        KIVI Integration:
        - If self.uses_kivi_model is True, KIVI handles compression internally in attention
        - The cache IS the KIVICache format, pass through unchanged
        """
        if past_key_values is None:
            return None
        
        # KIVI models handle compression internally, pass through
        if self.uses_kivi_model:
            return past_key_values
            
        # Handle DynamicCache
        if hasattr(past_key_values, 'to_legacy_cache'):
            past_key_values = past_key_values.to_legacy_cache()
        
        if self.kv_method == "none" and not self._minicache_active:
            return past_key_values
        
        compressed = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            k_c, v_c = k, v
            
            # 1. Apply TokenMerge sequence-length reduction FIRST (needs raw tensors)
            # This reduces KV cache size for multi-agent communication
            if self._minicache_active and self.minicache is not None:
                k_c, v_c = self.minicache.compress(k_c, v_c)
            
            # 2. Apply quantization-style compression AFTER sequence reduction
            # (KIVI compressor may transform tensor format)
            if self.kv_method != "none" and self.kv_method != "minicache":
                k_c, v_c = self.kv_compressor.compress(k_c, v_c, layer_idx=layer_idx)
            
            compressed.append((k_c, v_c))
        return tuple(compressed)
    
    def _decompress_kv(self, compressed_kv: Optional[Tuple]) -> Optional[Tuple]:
        """Decompress KV cache for model consumption.
        
        KIVI Integration:
        - If self.uses_kivi_model is True, cache IS KIVI format, pass through unchanged
        """
        if compressed_kv is None:
            return None
        
        # KIVI models handle compression internally, pass through
        if self.uses_kivi_model:
            return compressed_kv
        
        # Check if kv_compressor is identity (no quantization applied)
        # For combined mode (minicache+kivi), kv_compressor has the KIVI quantizer
        compressor_method = getattr(self.kv_compressor, 'method', 'none') if hasattr(self.kv_compressor, 'method') else 'none'
        if compressor_method == "none":
            # No quantization to decompress (pure minicache or none)
            return compressed_kv
        
        decompressed = []
        for k_c, v_c in compressed_kv:
            k, v = self.kv_compressor.decompress(k_c, v_c)
            decompressed.append((k, v))
        return tuple(decompressed)
    
    def _to_dynamic_cache(self, kv_tuple: Optional[Tuple]) -> Optional[DynamicCache]:
        """Convert tuple past_key_values to DynamicCache."""
        if kv_tuple is None:
            return None
        return DynamicCache.from_legacy_cache(kv_tuple)
    
    def _from_dynamic_cache(self, cache: Optional[DynamicCache]) -> Optional[Tuple]:
        """Convert DynamicCache back to tuple for compression."""
        if cache is None:
            return None
        return cache.to_legacy_cache()
    
    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        collect_logits: bool = False,
    ) -> LoopResult:
        """
        Run the DL-MoM collaboration loop (latent reasoning phase).
        
        Args:
            input_ids: [batch, seq] Input token IDs
            attention_mask: [batch, seq] Attention mask
            collect_logits: If True, collect merged logits at each step for drift tracking
        
        Returns:
            LoopResult with final packet and statistics
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Reset state
        self.controller.reset()
        self.kv_compressor.reset()
        
        kv_caches: List[Optional[Tuple]] = [None] * len(self.experts)
        step_log = []
        collected_logits: List[torch.Tensor] = []  # For drift tracking
        latent_steps = 0
        explicit_tokens = 0
        stopped_reason = "max_steps"
        current_seq_len = seq_len
        
        start_time = time.time()
        
        # Initial forward pass
        t0 = time.time()
        initial_logits = []
        last_hidden_state = None  # For embed mode
        for i, expert in enumerate(self.experts):
            outputs = expert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=(self.comm_mode == "embed"),
            )
            initial_logits.append(outputs.logits[:, -1, :])
            kv_caches[i] = self._compress_kv(outputs.past_key_values)
            # Capture hidden state from first expert for embed mode
            if i == 0 and self.comm_mode == "embed" and hasattr(outputs, 'hidden_states'):
                last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        time_forward = time.time() - t0
        
        # Merge initial logits
        t0 = time.time()
        merged_logits = self.consensus(initial_logits)
        
        # Check for NaN/Inf
        if torch.isnan(merged_logits).any() or torch.isinf(merged_logits).any():
            raise ValueError("NaN or Inf detected in merged_logits (initial)")
            
        time_consensus = time.time() - t0
        
        # Collect initial logits for drift tracking
        if collect_logits:
            collected_logits.append(merged_logits.detach().cpu())
        
        # Initial belief packet
        packet = belief_packet_from_logits(
            merged_logits,
            top_k=self.top_k,
            gumbel_noise=self.gumbel_noise,
            gumbel_tau=self.gumbel_tau,
        )
        
        # Apply optional Dirichlet noise
        if self.dirichlet_lambda is not None:
            packet = apply_dirichlet_noise(packet, self.dirichlet_lambda)
        
        # Get initial entropy from merged_logits (full vocab, not top-k packet)
        avg_entropy = normalized_entropy(merged_logits).mean().item()
        expert_entropies = [normalized_entropy(l).mean().item() 
                           for l in initial_logits]
        
        # Log initial step
        step_log.append({
            "step": 0,
            "mode": "latent",
            "entropy": avg_entropy,
            "expert_entropies": [e.item() if hasattr(e, 'item') else e for e in expert_entropies],
            "top_k_mass": packet.top_k_mass,
            "time_forward": time_forward,
            "time_consensus": time_consensus,
            "time_comm": 0.0,
        })
        
        # Main loop
        current_mode, should_stop = self.controller.step(avg_entropy)
        total_bridge_rate = 0.0
        
        for step_idx in range(1, self.max_steps + 1):
            if should_stop:
                stopped_reason = "converged"
                break
            
            # Prepare soft input for this step using selected communication protocol
            emb_layer = self._get_embedding_layer(self.experts[0])
            soft_input, bridge_rate = get_comm_input(
                comm_mode=self.comm_mode,
                packet=packet,
                merged_logits=merged_logits,
                embedding_layer=emb_layer,
                last_hidden_state=last_hidden_state,
            )
            total_bridge_rate += bridge_rate
            
            # Run expert forward passes with soft input
            t0 = time.time()
            step_logits = []
            for i, expert in enumerate(self.experts):
                past_kv = self._decompress_kv(kv_caches[i])
                # Convert tuple to DynamicCache for modern transformers models
                if past_kv is not None and isinstance(past_kv, tuple):
                    past_kv = self._to_dynamic_cache(past_kv)
                outputs = expert(
                    inputs_embeds=soft_input,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=(self.comm_mode == "embed"),
                )
                step_logits.append(outputs.logits[:, -1, :])
                kv_caches[i] = self._compress_kv(outputs.past_key_values)
                # Update hidden state from first expert for embed mode
                if i == 0 and self.comm_mode == "embed" and hasattr(outputs, 'hidden_states'):
                    last_hidden_state = outputs.hidden_states[-1][:, -1, :]
            time_forward = time.time() - t0
            
            # Consensus
            t0 = time.time()
            merged_logits = self.consensus(step_logits)
            
            # Check for NaN/Inf
            if torch.isnan(merged_logits).any() or torch.isinf(merged_logits).any():
                raise ValueError(f"NaN or Inf detected in merged_logits (step {step})")
                
            time_consensus = time.time() - t0
            
            # Collect logits for drift tracking
            if collect_logits:
                collected_logits.append(merged_logits.detach().cpu())
            
            # Update packet
            packet = belief_packet_from_logits(
                merged_logits,
                top_k=self.top_k,
                gumbel_noise=self.gumbel_noise,
                gumbel_tau=self.gumbel_tau,
            )
            
            if self.dirichlet_lambda is not None:
                packet = apply_dirichlet_noise(packet, self.dirichlet_lambda)
            
            # Compute entropy from merged_logits (full vocab, not top-k packet)
            avg_entropy = normalized_entropy(merged_logits).mean().item()
            expert_entropies = [normalized_entropy(l).mean().item() 
                               for l in step_logits]
            
            # Check for EOS
            top_token = packet.token_ids[0, 0].item()
            if top_token == self.tokenizer.eos_token_id and packet.probs[0, 0] > 0.9:
                stopped_reason = "eos"
                break
            
            latent_steps += 1
            current_seq_len += 1
            
            # Log step
            step_log.append({
                "step": step_idx,
                "mode": current_mode,
                "entropy": avg_entropy,
                "expert_entropies": [e.item() if hasattr(e, 'item') else e for e in expert_entropies],
                "top_k_mass": packet.top_k_mass,
                "bridge_rate": bridge_rate,
                "time_forward": time_forward,
                "time_consensus": time_consensus,
                "time_comm": 0.0,
            })
            
            # Update mode
            current_mode, should_stop = self.controller.step(avg_entropy)
        
        wall_time = time.time() - start_time
        
        return LoopResult(
            final_packet=packet,
            total_steps=latent_steps,
            latent_steps=latent_steps,
            explicit_tokens=explicit_tokens,
            switches=self.controller.state.switch_count,
            wall_time=wall_time,
            step_log=step_log,
            kv_compression_stats=self._get_kv_compression_stats(),
            stopped_reason=stopped_reason,
            final_kv_caches=kv_caches,
            step_logits=collected_logits if collect_logits else None,
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        collect_logits: bool = False,
    ) -> Tuple[str, LoopResult]:
        """
        Generate text using DL-MoM: latent reasoning followed by explicit decoding.
        
        Note: Currently only supports batch_size=1.
        
        Args:
            input_ids: [batch, seq] Input token IDs (batch_size must be 1)
            attention_mask: [batch, seq] Attention mask
            collect_logits: If True, collect logits for drift tracking
        
        Returns:
            (generated_text, loop_result)
        """
        # Enforce batch_size=1 constraint
        if input_ids.shape[0] != 1:
            raise ValueError("generate() currently only supports batch_size=1")
        
        device = input_ids.device
        
        # Phase 1: Latent reasoning
        loop_result = self.run(input_ids, attention_mask, collect_logits=collect_logits)
        
        # Phase 2: Explicit generation using primary expert
        expert = self.experts[0]
        
        generated_ids: List[int] = []
        
        # Get the top token from the belief packet as a hard token
        # (Using soft embeddings corrupts generation; hard tokens work correctly)
        top_token_id = loop_result.final_packet.token_ids[0, 0].item()
        
        # Start fresh - don't use the latent loop's KV cache as it contains
        # states from soft embeddings which corrupt attention patterns.
        # Instead, re-run from input_ids and append the top token from latent reasoning.
        past_kv = None
        
        # Track sequence length for attention mask
        if attention_mask is not None:
            current_seq_len = attention_mask.shape[1]
        else:
            current_seq_len = input_ids.shape[1]
        
        # Initial forward pass with input_ids to build clean KV cache
        if loop_result.latent_steps > 0:
            # Latent reasoning happened - use top token from packet as first generated token
            outputs = expert(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            past_kv = outputs.past_key_values
            
            # Add the latent reasoning result as first token (if not EOS)
            if top_token_id != self.tokenizer.eos_token_id:
                generated_ids.append(top_token_id)
                current_seq_len += 1
        else:
            # No latent reasoning - use the KV cache from run() directly
            past_kv = self._decompress_kv(loop_result.final_kv_caches[0]) if loop_result.final_kv_caches else None
            if past_kv is not None and isinstance(past_kv, tuple):
                past_kv = self._to_dynamic_cache(past_kv)
        
        for step in range(self.max_new_tokens):
            # Skip if we have no tokens to continue from
            if not generated_ids:
                # First iteration after latent reasoning - we already added top token
                # Need to generate next token conditioned on it
                next_input_ids = torch.tensor([[top_token_id]], device=device)
            else:
                next_input_ids = torch.tensor([[generated_ids[-1]]], device=device)
            
            # Build attention mask for current sequence length
            attn_mask = torch.ones(1, current_seq_len + 1, device=device)
            
            outputs = expert(input_ids=next_input_ids, past_key_values=past_kv,
                             attention_mask=attn_mask, use_cache=True)
            
            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            
            # Apply repetition penalty
            if self.repetition_penalty != 1.0 and generated_ids:
                for prev_token in set(generated_ids):
                    if logits[0, prev_token] > 0:
                        logits[0, prev_token] /= self.repetition_penalty
                    else:
                        logits[0, prev_token] *= self.repetition_penalty
            
            next_token = logits.argmax(dim=-1).item()
            
            if next_token == self.tokenizer.eos_token_id:
                break
            
            generated_ids.append(next_token)
            current_seq_len += 1
        
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text, loop_result
    
    def decode_final(self, packet: BeliefPacket) -> str:
        """Decode final answer from belief packet (single token)."""
        top_token_id = packet.token_ids[0, 0].item()
        return self.tokenizer.decode([top_token_id])
