"""
Soft Belief Packet Protocol for DL-MoM.

Enables cross-model latent communication via sparse (token_id, probability) tuples.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BeliefPacket:
    """
    Soft belief packet containing top-k token IDs and probabilities.
    
    Attributes:
        token_ids: [batch, k] Top-k token indices
        probs: [batch, k] Corresponding probabilities (sum to 1)
        source_vocab_size: Original vocabulary size (for OOV detection)
    """
    token_ids: torch.Tensor
    probs: torch.Tensor
    source_vocab_size: int
    
    def to(self, device: torch.device) -> "BeliefPacket":
        return BeliefPacket(
            token_ids=self.token_ids.to(device),
            probs=self.probs.to(device),
            source_vocab_size=self.source_vocab_size
        )
    
    @property
    def top_k_mass(self) -> float:
        """Probability mass captured by top-k (detached, for logging only)."""
        return self.probs.sum(dim=-1).mean().item()
    
    @property
    def entropy(self) -> float:
        """Normalized entropy of the belief distribution."""
        h = -torch.sum(self.probs * torch.log(self.probs + 1e-10), dim=-1)
        max_h = torch.log(torch.tensor(self.probs.shape[-1], dtype=torch.float32, device=self.probs.device))
        return (h / max_h).mean().item()
    
    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        return {
            "top_tokens": self.token_ids[0, :5].tolist(),
            "top_probs": [round(p, 4) for p in self.probs[0, :5].tolist()],
            "top_k_mass": round(self.top_k_mass, 4),
            "entropy": round(self.entropy, 4),
            "k": self.probs.shape[-1],
        }


def belief_packet_from_logits(
    logits: torch.Tensor,
    top_k: int = 50,
    temperature: float = 1.0,
    gumbel_noise: bool = False,
    gumbel_tau: float = 1.0,
) -> BeliefPacket:
    """
    Create a belief packet from logits.
    
    Args:
        logits: [batch, vocab_size] Raw logits
        top_k: Number of tokens to retain (-1 for full vocab)
        temperature: Softmax temperature
        gumbel_noise: If True, apply Gumbel-Softmax
        gumbel_tau: Gumbel temperature (only if gumbel_noise=True)
    
    Returns:
        BeliefPacket with top-k tokens and probabilities
    """
    if logits.dim() != 2:
        raise ValueError(f"Expected logits of shape [batch, vocab], got {logits.shape}")
    
    vocab_size = logits.shape[-1]
    
    # Handle k=-1 (full vocab) or k > vocab_size
    effective_k = vocab_size if top_k == -1 or top_k > vocab_size else top_k
    
    # Apply temperature
    scaled_logits = logits / temperature
    
    if gumbel_noise:
        # Gumbel-Softmax for stochastic exploration (numerically stable)
        u = torch.rand_like(scaled_logits).clamp(1e-10, 1 - 1e-10)
        gumbel = -torch.log(-torch.log(u))
        scaled_logits = (scaled_logits + gumbel) / gumbel_tau
    
    # Get probabilities
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Extract top-k
    top_probs, top_ids = torch.topk(probs, effective_k, dim=-1)
    
    # Renormalize
    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
    
    return BeliefPacket(
        token_ids=top_ids,
        probs=top_probs,
        source_vocab_size=vocab_size
    )


def reconstruct_soft_input(
    packet: BeliefPacket,
    embedding_layer: torch.nn.Embedding,
    receiver_vocab_size: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Reconstruct soft input embedding from belief packet.
    
    Args:
        packet: BeliefPacket from sender
        embedding_layer: Receiver's embedding layer
        receiver_vocab_size: Receiver's vocab size (for OOV detection)
    
    Returns:
        Tuple of (soft_embedding [batch, 1, hidden], bridge_rate)
    """
    token_ids = packet.token_ids
    probs = packet.probs
    
    # Detect and handle OOV tokens
    if receiver_vocab_size is not None:
        oov_mask = token_ids >= receiver_vocab_size
        bridge_rate = oov_mask.float().mean().item()
        
        if bridge_rate > 0:
            # Zero out OOV probabilities and renormalize
            valid_probs = probs.clone()
            valid_probs[oov_mask] = 0
            valid_mass = valid_probs.sum(dim=-1, keepdim=True)
            
            if (valid_mass == 0).any():
                raise ValueError("All tokens in belief packet are OOV - Text-Bridge required")
            
            probs = valid_probs / valid_mass
            token_ids = token_ids.clone()
            token_ids[oov_mask] = 0  # Safe: these have zero probability
    else:
        bridge_rate = 0.0
    
    # Get embeddings for all top-k tokens: [batch, k, hidden]
    embeddings = embedding_layer(token_ids)
    
    # Match dtype to avoid float32/float16 mismatch in einsum
    probs = probs.to(embeddings.dtype)
    
    # Weighted sum: [batch, hidden]
    soft_emb = torch.einsum("bk,bkh->bh", probs, embeddings)
    
    # Add sequence dimension: [batch, 1, hidden]
    return soft_emb.unsqueeze(1), bridge_rate


def apply_dirichlet_noise(
    packet: BeliefPacket,
    concentration: float = 50.0,
) -> BeliefPacket:
    """
    Apply Dirichlet noise to belief packet probabilities.
    
    Args:
        packet: Original belief packet
        concentration: Dirichlet concentration (lambda). Higher = less noise.
    
    Returns:
        New BeliefPacket with perturbed probabilities
    """
    # Add epsilon to prevent zero alpha (numerical stability)
    alpha = concentration * packet.probs + 1e-6
    
    # Sample from Dirichlet
    noisy_probs = torch.distributions.Dirichlet(alpha).sample()
    
    return BeliefPacket(
        token_ids=packet.token_ids,
        probs=noisy_probs,
        source_vocab_size=packet.source_vocab_size
    )


# ============================================================================
# Communication Protocol Functions (A3 Suite)
# ============================================================================

def get_comm_input(
    comm_mode: str,
    packet: BeliefPacket,
    merged_logits: torch.Tensor,
    embedding_layer: torch.nn.Embedding,
    last_hidden_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Get soft input embedding based on communication protocol.
    
    Paper Section 6.1 A3 - Communication Protocols:
    - embed: Raw embedding vectors (same-model only) - uses last hidden state
    - belief: Soft Belief Packets - weighted top-k embedding
    - logits: Full logits transmission - full vocab weighted embedding
    
    Args:
        comm_mode: "embed", "belief", or "logits"
        packet: BeliefPacket with top-k tokens and probs
        merged_logits: Full vocabulary logits [batch, vocab_size]
        embedding_layer: Model's embedding layer
        last_hidden_state: Last hidden state from previous step (for embed mode)
    
    Returns:
        (soft_embedding [batch, 1, hidden], bridge_rate)
    """
    if comm_mode == "embed":
        return comm_embed(last_hidden_state)
    elif comm_mode == "belief":
        return comm_belief(packet, embedding_layer)
    elif comm_mode == "logits":
        return comm_logits(merged_logits, embedding_layer)
    else:
        raise ValueError(f"Unknown comm_mode: {comm_mode}. Expected 'embed', 'belief', or 'logits'")


def comm_embed(
    last_hidden_state: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, float]:
    """
    A3.1 - Raw embedding vectors (same-model only).
    
    Transmits the full hidden state from the previous step.
    Maximum information but only works for same-model setups.
    Paper: "transmit fp16/bf16 embeddings for each latent step"
    
    Args:
        last_hidden_state: [batch, hidden_dim] from previous forward pass
    
    Returns:
        (embedding [batch, 1, hidden], bridge_rate=0.0)
    """
    if last_hidden_state is None:
        raise ValueError("comm_embed requires last_hidden_state from previous step")
    
    # Ensure correct shape [batch, 1, hidden]
    if last_hidden_state.dim() == 2:
        return last_hidden_state.unsqueeze(1), 0.0
    elif last_hidden_state.dim() == 3:
        # Take last position if sequence dim present
        return last_hidden_state[:, -1:, :], 0.0
    else:
        raise ValueError(f"Unexpected hidden state shape: {last_hidden_state.shape}")


def comm_belief(
    packet: BeliefPacket,
    embedding_layer: torch.nn.Embedding,
    receiver_vocab_size: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    """
    A3.2-A3.4 - Soft Belief Packets.
    
    Transmits (token_ids, probs) and reconstructs embedding on receiver side.
    Paper Equation: e_soft = Σ p_i · E_recv[token_id_i]
    
    Args:
        packet: BeliefPacket with top-k tokens and probabilities
        embedding_layer: Receiver's embedding layer
        receiver_vocab_size: For OOV detection
    
    Returns:
        (soft_embedding [batch, 1, hidden], bridge_rate)
    """
    # This is the existing reconstruct_soft_input implementation
    token_ids = packet.token_ids
    probs = packet.probs
    
    # Detect and handle OOV tokens
    if receiver_vocab_size is not None:
        oov_mask = token_ids >= receiver_vocab_size
        bridge_rate = oov_mask.float().mean().item()
        
        if bridge_rate > 0:
            valid_probs = probs.clone()
            valid_probs[oov_mask] = 0
            valid_mass = valid_probs.sum(dim=-1, keepdim=True)
            
            if (valid_mass == 0).any():
                raise ValueError("All tokens in belief packet are OOV - Text-Bridge required")
            
            probs = valid_probs / valid_mass
            token_ids = token_ids.clone()
            token_ids[oov_mask] = 0
    else:
        bridge_rate = 0.0
    
    # Get embeddings for all top-k tokens: [batch, k, hidden]
    embeddings = embedding_layer(token_ids)
    
    # Match dtype to avoid float32/float16 mismatch in einsum
    probs = probs.to(embeddings.dtype)
    
    # Weighted sum: [batch, hidden]
    soft_emb = torch.einsum("bk,bkh->bh", probs, embeddings)
    
    return soft_emb.unsqueeze(1), bridge_rate


def comm_logits(
    merged_logits: torch.Tensor,
    embedding_layer: torch.nn.Embedding,
) -> Tuple[torch.Tensor, float]:
    """
    A3.5 - Full logits transmission (upper bound).
    
    Transmits full logits and computes soft embedding over entire vocabulary.
    Most expensive but preserves complete distribution.
    Paper: "transmit logits... and reconstruct probabilities exactly"
    
    Args:
        merged_logits: [batch, vocab_size] full vocabulary logits
        embedding_layer: Embedding layer
    
    Returns:
        (soft_embedding [batch, 1, hidden], bridge_rate=0.0)
    """
    # Compute probabilities over full vocabulary (use float32 for stability)
    probs = torch.softmax(merged_logits.float(), dim=-1)
    
    # Get all embeddings: [vocab_size, hidden]
    all_embeddings = embedding_layer.weight
    
    # Cast probs to match embedding dtype
    probs = probs.to(all_embeddings.dtype)
    
    # Weighted sum over entire vocabulary: [batch, hidden]
    # This is expensive: O(batch * vocab_size * hidden_dim) matmul
    soft_emb = torch.matmul(probs, all_embeddings)
    
    return soft_emb.unsqueeze(1), 0.0
