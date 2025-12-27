"""
Contrastive TIES-Style Consensus Engine for DL-MoM.

Merges expert preferences in logit space using directional conflict resolution.

NOTE: Adapted from TIES-Merging (Yadav et al., 2023) which operates on parameter deltas.
This version operates on logits, using centering to create preference vectors.
The analogy holds because:
1. Centering creates "preference vectors" similar to task vectors
2. Trimming removes low-confidence predictions (noise)
3. Sign election resolves directional conflicts
4. Agreement-weighted merge preserves consensus

Key difference: No pretrained baseline exists in logit space, so we center
by subtracting per-expert mean instead of a shared reference.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Literal, Dict, Any


def contrastive_ties(
    logits_list: List[torch.Tensor],
    trim_threshold: float = 0.1,
    trim_mode: Literal["absolute", "std_relative"] = "std_relative",
    already_centered: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Merge multiple experts' logits using contrastive TIES-style consensus.
    
    Procedure:
    1. Center logits into preference vectors (subtract mean)
    2. Trim small-magnitude preferences (noise)
    3. Elect sign direction via voting
    4. Merge only experts agreeing with elected sign
    
    Args:
        logits_list: List of [batch, vocab_size] tensors from K experts
        trim_threshold: Magnitude threshold for trimming
        trim_mode: "absolute" or "std_relative" (threshold * std)
        already_centered: Skip centering if inputs are pre-normalized
        eps: Small value for numerical stability
    
    Returns:
        Merged logits [batch, vocab_size]
    """
    if len(logits_list) == 0:
        raise ValueError("logits_list cannot be empty")
    
    if len(logits_list) == 1:
        return logits_list[0]
    
    # Validate shapes
    shapes = [l.shape for l in logits_list]
    if len(set(shapes)) > 1:
        raise ValueError(f"All logits must have same shape, got {shapes}")
    
    # Stack: [K, batch, vocab]
    stacked = torch.stack(logits_list, dim=0)
    K = stacked.shape[0]
    
    # 1. Center each expert's logits (contrastive preferences)
    if already_centered:
        centered = stacked
    else:
        centered = stacked - stacked.mean(dim=-1, keepdim=True)
    
    # 2. Trim: zero out small-magnitude preferences
    if trim_mode == "std_relative":
        std = centered.std(dim=-1, keepdim=True) + eps
        threshold = trim_threshold * std
    else:
        threshold = trim_threshold
    
    mask = (centered.abs() > threshold).float()
    trimmed = centered * mask
    
    # 3. Elect: vote on sign direction per token
    sign_votes = torch.sign(trimmed).sum(dim=0)  # [batch, vocab]
    elected_sign = torch.sign(sign_votes)
    
    # Handle ties (sign_votes == 0) by defaulting to positive
    elected_sign = torch.where(elected_sign == 0, torch.ones_like(elected_sign), elected_sign)
    
    # 4. Merge: average only experts agreeing with elected sign
    agreement = (torch.sign(trimmed) == elected_sign).float()
    numerator = (trimmed * agreement).sum(dim=0)  # [batch, vocab]
    denominator = agreement.sum(dim=0).clamp(min=1.0)
    
    merged_preferences = numerator / (denominator + eps)
    
    # Restore baseline: use mean expert as reference (Option B from review)
    baseline = stacked.mean(dim=0)  # [batch, vocab]
    
    return baseline + merged_preferences


class ContrastiveConsensus:
    """
    Wrapper class for contrastive TIES consensus with configurable parameters.
    """
    
    def __init__(
        self,
        trim_threshold: float = 0.1,
        trim_mode: Literal["absolute", "std_relative"] = "std_relative",
        temperature_normalize: bool = True,
    ):
        """
        Args:
            trim_threshold: Magnitude threshold for noise trimming
            trim_mode: "absolute" or "std_relative" (threshold * std)
            temperature_normalize: If True, z-score normalize before merging
        """
        self.trim_threshold = trim_threshold
        self.trim_mode = trim_mode
        self.temperature_normalize = temperature_normalize
    
    def __call__(
        self,
        logits_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Merge expert logits.
        
        Args:
            logits_list: List of [batch, vocab_size] logits
        
        Returns:
            Merged logits [batch, vocab_size]
        """
        if self.temperature_normalize:
            # Z-score normalize each expert's logits
            normalized = []
            for logits in logits_list:
                mean = logits.mean(dim=-1, keepdim=True)
                std = logits.std(dim=-1, keepdim=True) + 1e-6
                normalized.append((logits - mean) / std)
            logits_list = normalized
            already_centered = True
        else:
            already_centered = False
        
        return contrastive_ties(
            logits_list,
            self.trim_threshold,
            self.trim_mode,
            already_centered=already_centered,
        )
    
    def merge_per_family(
        self,
        family_logits: Dict[str, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Hierarchical consensus: merge within families, then across families.
        
        For heterogeneous experts with different tokenizers, merge same-tokenizer
        experts first, then merge family results.
        
        Args:
            family_logits: Dict mapping family name to list of expert logits
        
        Returns:
            Final merged logits
        """
        if not family_logits:
            raise ValueError("family_logits cannot be empty")
        
        family_merged = []
        target_device = None
        
        for family_name, logits_list in family_logits.items():
            if not logits_list:
                continue
            
            if target_device is None:
                target_device = logits_list[0].device
            
            # Move to common device
            logits_list = [l.to(target_device) for l in logits_list]
            merged = self(logits_list)
            family_merged.append(merged)
        
        if len(family_merged) == 1:
            return family_merged[0]
        
        return self(family_merged)


def consensus_diagnostics(
    logits_list: List[torch.Tensor],
    merged: torch.Tensor,
) -> Dict[str, Any]:
    """
    Compute diagnostic metrics for consensus quality.
    
    Args:
        logits_list: Original expert logits
        merged: Result from contrastive_ties
    
    Returns:
        dict with agreement_rate, kl_from_mean, entropy_reduction
    """
    stacked = torch.stack(logits_list, dim=0)
    
    # Agreement: fraction of tokens where experts agree on sign
    centered = stacked - stacked.mean(dim=-1, keepdim=True)
    signs = torch.sign(centered)
    # Compare all to first expert
    agreement = (signs == signs[0:1]).float().mean(dim=0)
    avg_agreement = agreement.mean().item()
    
    # KL from mean expert
    mean_expert = stacked.mean(dim=0)
    mean_probs = F.softmax(mean_expert, dim=-1)
    merged_probs = F.softmax(merged, dim=-1)
    kl = F.kl_div(merged_probs.log(), mean_probs, reduction='batchmean').item()
    
    # Entropy
    mean_entropy = -(mean_probs * mean_probs.log().clamp(min=-100)).sum(dim=-1).mean().item()
    merged_entropy = -(merged_probs * merged_probs.log().clamp(min=-100)).sum(dim=-1).mean().item()
    
    return {
        "agreement_rate": avg_agreement,
        "kl_from_mean_expert": kl,
        "mean_entropy": mean_entropy,
        "merged_entropy": merged_entropy,
        "entropy_reduction": mean_entropy - merged_entropy,
    }
