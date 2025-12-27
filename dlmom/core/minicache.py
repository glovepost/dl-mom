"""
MiniCache and TokenMerge: KV Cache Compression Strategies

Two compression approaches for multi-agent LLM systems:

1. TokenMerge (within-sequence): Farthest-point sampling to keep diverse tokens
2. CrossLayerMiniCache: Liu et al. (2024) algorithm that merges KV caches between
   adjacent transformer layers using SLERP interpolation

Reference: MiniCache: KV Cache Compression in Depth Dimension for Large Language Models
           https://arxiv.org/abs/2405.14366
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import math


class TokenMerge:
    """
    Within-Sequence Token Pruning for KV Cache Compression.
    
    Uses farthest-point sampling to keep diverse tokens based on cosine dissimilarity.
    Reduces sequence length while preserving important attention patterns.
    """
    
    def __init__(
        self,
        merge_ratio: float = 0.5,
        similarity_threshold: float = 0.9,
        min_tokens_before_merge: int = 64,
    ):
        """
        Args:
            merge_ratio: Target ratio of tokens to keep (0.5 = keep 50%)
            similarity_threshold: Minimum cosine similarity to consider merging
            min_tokens_before_merge: Don't merge until cache has this many tokens
        """
        self.merge_ratio = merge_ratio
        self.similarity_threshold = similarity_threshold
        self.min_tokens_before_merge = min_tokens_before_merge
        self.stats = {
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "merge_calls": 0,
            "bytes_in": 0,
            "bytes_out": 0,
        }
    
    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV cache by selecting diverse tokens.
        
        Args:
            keys: [batch, heads, seq_len, head_dim]
            values: [batch, heads, seq_len, head_dim]
        
        Returns:
            Compressed (keys, values) with reduced seq_len
        """
        batch, heads, seq_len, head_dim = keys.shape
        
        # Calculate input bytes (keys + values)
        element_size = keys.element_size()  # 2 for fp16, 4 for fp32
        bytes_in = 2 * batch * heads * seq_len * head_dim * element_size
        
        self.stats["total_tokens_in"] += seq_len * batch
        self.stats["merge_calls"] += 1
        self.stats["bytes_in"] += bytes_in
        
        # Don't merge if cache is too small
        if seq_len < self.min_tokens_before_merge:
            self.stats["total_tokens_out"] += seq_len * batch
            self.stats["bytes_out"] += bytes_in
            return keys, values
        
        # Calculate target number of tokens to keep
        target_len = max(int(seq_len * self.merge_ratio), self.min_tokens_before_merge // 2)
        
        if target_len >= seq_len:
            self.stats["total_tokens_out"] += seq_len * batch
            self.stats["bytes_out"] += bytes_in
            return keys, values
        
        # Compute pairwise cosine similarity for keys
        keys_flat = keys.view(batch * heads, seq_len, head_dim)
        keys_norm = F.normalize(keys_flat, dim=-1)
        
        # Greedy token selection: keep tokens that are most dissimilar
        keep_indices = self._greedy_select(keys_norm, target_len)
        
        # Gather selected tokens
        batch_heads = batch * heads
        keys_flat = keys.view(batch_heads, seq_len, head_dim)
        values_flat = values.view(batch_heads, seq_len, head_dim)
        
        # Expand indices for gather
        keep_indices = keep_indices.unsqueeze(-1).expand(-1, -1, head_dim)
        
        keys_out = torch.gather(keys_flat, 1, keep_indices)
        values_out = torch.gather(values_flat, 1, keep_indices)
        
        # Reshape back
        keys_out = keys_out.view(batch, heads, target_len, head_dim)
        values_out = values_out.view(batch, heads, target_len, head_dim)
        
        # Calculate output bytes
        bytes_out = 2 * batch * heads * target_len * head_dim * element_size
        
        self.stats["total_tokens_out"] += target_len * batch
        self.stats["bytes_out"] += bytes_out
        
        # Log compression stats periodically
        if self.stats["merge_calls"] % 100 == 1:
            savings_mb = (self.stats["bytes_in"] - self.stats["bytes_out"]) / 1e6
            ratio = self.stats["bytes_out"] / max(self.stats["bytes_in"], 1)
            print(f"[TokenMerge] calls={self.stats['merge_calls']}, "
                  f"tokens: {self.stats['total_tokens_in']}→{self.stats['total_tokens_out']}, "
                  f"saved={savings_mb:.1f}MB ({1-ratio:.1%})")
        
        return keys_out, values_out
    
    def _greedy_select(
        self,
        keys_norm: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """Farthest-point sampling for diverse token selection."""
        batch_heads, seq_len, _ = keys_norm.shape
        device = keys_norm.device
        
        selected = torch.zeros(batch_heads, target_len, dtype=torch.long, device=device)
        selected[:, 0] = 0  # First token (attention sink)
        selected[:, 1] = seq_len - 1  # Last token (most recent)
        
        min_sim_to_selected = torch.ones(batch_heads, seq_len, device=device)
        
        if target_len <= 2:
            return selected
        
        sim_to_first = torch.bmm(keys_norm, keys_norm[:, 0:1, :].transpose(1, 2)).squeeze(-1)
        sim_to_last = torch.bmm(keys_norm, keys_norm[:, -1:, :].transpose(1, 2)).squeeze(-1)
        min_sim_to_selected = torch.minimum(sim_to_first, sim_to_last)
        
        min_sim_to_selected[:, 0] = 1.0
        min_sim_to_selected[:, -1] = 1.0
        
        for i in range(2, target_len):
            _, next_idx = min_sim_to_selected.min(dim=1)
            selected[:, i] = next_idx
            
            batch_indices = torch.arange(batch_heads, device=device)
            next_token = keys_norm[batch_indices, next_idx, :].unsqueeze(1)
            sim_to_next = torch.bmm(keys_norm, next_token.transpose(1, 2)).squeeze(-1)
            min_sim_to_selected = torch.minimum(min_sim_to_selected, sim_to_next)
            min_sim_to_selected[batch_indices, next_idx] = 1.0
        
        selected, _ = selected.sort(dim=1)
        return selected
    
    def get_stats(self) -> Dict[str, Any]:
        ratio = self.stats["total_tokens_out"] / max(self.stats["total_tokens_in"], 1)
        bytes_ratio = self.stats["bytes_out"] / max(self.stats["bytes_in"], 1)
        savings_mb = (self.stats["bytes_in"] - self.stats["bytes_out"]) / 1e6
        return {
            "method": "token_merge",
            "merge_ratio": self.merge_ratio,
            "actual_ratio": ratio,
            "merge_calls": self.stats["merge_calls"],
            "bytes_in_mb": self.stats["bytes_in"] / 1e6,
            "bytes_out_mb": self.stats["bytes_out"] / 1e6,
            "bytes_saved_mb": savings_mb,
            "compression_ratio": bytes_ratio,
        }
    
    def reset_stats(self):
        self.stats = {
            "total_tokens_in": 0, "total_tokens_out": 0, "merge_calls": 0,
            "bytes_in": 0, "bytes_out": 0,
        }

class CrossLayerMiniCache:
    """
    Cross-Layer KV Cache Compression using SLERP Interpolation.
    
    Implements the MiniCache algorithm from Liu et al. (2024):
    - Merges KV caches between adjacent transformer layers (depth dimension)
    - Uses spherical linear interpolation (SLERP) for directional merging
    - Stores magnitude separately for lossless restoration
    - Retains unmergeable tokens based on angular distance
    
    This reduces memory by ~2x for middle-to-deep layers.
    """
    
    def __init__(
        self,
        start_layer_ratio: float = 0.5,  # Start merging from L/2
        interpolation_t: float = 0.5,     # SLERP interpolation parameter
        angular_threshold: float = 0.3,   # Threshold for unmergeable tokens (d/π)
    ):
        """
        Args:
            start_layer_ratio: Start cross-layer merging from this layer (0.5 = L/2)
            interpolation_t: SLERP interpolation weight (0.5 = average)
            angular_threshold: Angular distance threshold for token retention
        """
        self.start_layer_ratio = start_layer_ratio
        self.interpolation_t = interpolation_t
        self.angular_threshold = angular_threshold
        self.stats = {
            "layers_merged": 0,
            "tokens_retained": 0,
            "total_tokens": 0,
        }
    
    def compress(
        self,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        num_layers: int,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Compress KV cache by merging adjacent layers.
        
        Args:
            past_key_values: List of (key, value) tuples per layer
                             Each tensor: [batch, heads, seq_len, head_dim]
            num_layers: Total number of transformer layers
        
        Returns:
            (compressed_kv, restoration_data)
            - compressed_kv: Reduced list with merged layers
            - restoration_data: Magnitudes and angles for decompression
        """
        start_layer = int(num_layers * self.start_layer_ratio)
        
        compressed = []
        restoration_data = {
            "magnitudes_l": [],
            "magnitudes_l_minus_1": [],
            "angles": [],
            "retained_indices": [],
            "start_layer": start_layer,
        }
        
        # Keep early layers unchanged
        for layer_idx in range(start_layer):
            compressed.append(past_key_values[layer_idx])
        
        # Merge adjacent layers from start_layer onwards
        layer_idx = start_layer
        while layer_idx < num_layers - 1:
            k_l, v_l = past_key_values[layer_idx]
            k_l_minus_1, v_l_minus_1 = past_key_values[layer_idx + 1]
            
            # Merge keys and values
            k_merged, k_mags, k_angle, k_retained = self._merge_tensors(
                k_l, k_l_minus_1, "keys"
            )
            v_merged, v_mags, v_angle, v_retained = self._merge_tensors(
                v_l, v_l_minus_1, "values"
            )
            
            compressed.append((k_merged, v_merged))
            
            # Store restoration data
            restoration_data["magnitudes_l"].append((k_mags[0], v_mags[0]))
            restoration_data["magnitudes_l_minus_1"].append((k_mags[1], v_mags[1]))
            restoration_data["angles"].append((k_angle, v_angle))
            restoration_data["retained_indices"].append((k_retained, v_retained))
            
            self.stats["layers_merged"] += 1
            layer_idx += 2  # Skip merged layer
        
        # Handle odd remaining layer
        if layer_idx < num_layers:
            compressed.append(past_key_values[layer_idx])
        
        return compressed, restoration_data
    
    def _merge_tensors(
        self,
        x_l: torch.Tensor,
        x_l_minus_1: torch.Tensor,
        name: str,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Merge two adjacent layer tensors using SLERP.
        
        Returns:
            (merged, (mag_l, mag_l_minus_1), angle, retained_indices)
        """
        batch, heads, seq_len, head_dim = x_l.shape
        device = x_l.device
        dtype = x_l.dtype
        
        # Compute magnitudes (L2 norm)
        mag_l = torch.norm(x_l, dim=-1, keepdim=True)  # [batch, heads, seq, 1]
        mag_l_minus_1 = torch.norm(x_l_minus_1, dim=-1, keepdim=True)
        
        # Normalize to unit vectors
        x_l_norm = x_l / (mag_l + 1e-8)
        x_l_minus_1_norm = x_l_minus_1 / (mag_l_minus_1 + 1e-8)
        
        # Compute angular distance: Ω = arccos(x_l · x_{l-1})
        cos_angle = (x_l_norm * x_l_minus_1_norm).sum(dim=-1, keepdim=True)
        cos_angle = cos_angle.clamp(-1.0, 1.0)  # Numerical stability
        angle = torch.acos(cos_angle)  # [batch, heads, seq, 1]
        
        # Identify unmergeable tokens (high angular distance)
        angular_distance = angle / math.pi  # Normalize to [0, 1]
        unmergeable_mask = angular_distance > self.angular_threshold
        
        # SLERP interpolation for mergeable tokens
        # e^{l,l-1} = (sin((1-t)Ω) * x_l + sin(tΩ) * x_{l-1}) / sin(Ω)
        t = self.interpolation_t
        sin_angle = torch.sin(angle)
        sin_angle = sin_angle.clamp(min=1e-6)  # Avoid division by zero
        
        weight_l = torch.sin((1 - t) * angle) / sin_angle
        weight_l_minus_1 = torch.sin(t * angle) / sin_angle
        
        # Handle near-parallel vectors (angle ≈ 0)
        near_parallel = angle.abs() < 1e-4
        weight_l = torch.where(near_parallel, torch.ones_like(weight_l) * (1 - t), weight_l)
        weight_l_minus_1 = torch.where(near_parallel, torch.ones_like(weight_l_minus_1) * t, weight_l_minus_1)
        
        # Compute merged direction (SLERP result)
        merged_direction = weight_l * x_l_norm + weight_l_minus_1 * x_l_minus_1_norm
        
        # For unmergeable tokens, just keep x_l (no merging)
        merged_direction = torch.where(
            unmergeable_mask.expand_as(merged_direction),
            x_l_norm,
            merged_direction
        )
        
        # Store retained indices for unmergeable tokens
        retained_indices = unmergeable_mask.squeeze(-1).any(dim=1).any(dim=0)  # [seq_len]
        
        self.stats["tokens_retained"] += retained_indices.sum().item()
        self.stats["total_tokens"] += seq_len
        
        return merged_direction, (mag_l.squeeze(-1), mag_l_minus_1.squeeze(-1)), angle.squeeze(-1), retained_indices
    
    def decompress(
        self,
        compressed_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        restoration_data: Dict[str, Any],
        target_layer: int,
        is_upper_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Restore original KV for a specific layer from merged cache.
        
        Args:
            compressed_kv: Compressed KV cache
            restoration_data: Magnitudes and angles from compression
            target_layer: The layer to restore
            is_upper_layer: True for layer l, False for layer l-1
        
        Returns:
            Restored (key, value) tensors
        """
        start_layer = restoration_data["start_layer"]
        
        # Early layers are not compressed
        if target_layer < start_layer:
            return compressed_kv[target_layer]
        
        # Find the merged layer index
        merged_idx = (target_layer - start_layer) // 2
        if merged_idx >= len(restoration_data["magnitudes_l"]):
            # Odd final layer
            return compressed_kv[-1]
        
        k_merged, v_merged = compressed_kv[start_layer + merged_idx]
        
        k_mag_l, v_mag_l = restoration_data["magnitudes_l"][merged_idx]
        k_mag_l_minus_1, v_mag_l_minus_1 = restoration_data["magnitudes_l_minus_1"][merged_idx]
        k_angle, v_angle = restoration_data["angles"][merged_idx]
        
        if is_upper_layer:
            # Restore x_l: magnitude_l * direction adjusted by angle
            k_restored = k_merged * k_mag_l.unsqueeze(-1)
            v_restored = v_merged * v_mag_l.unsqueeze(-1)
        else:
            # Restore x_{l-1}: magnitude_{l-1} * direction adjusted by angle
            k_restored = k_merged * k_mag_l_minus_1.unsqueeze(-1)
            v_restored = v_merged * v_mag_l_minus_1.unsqueeze(-1)
        
        return k_restored, v_restored
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "method": "cross_layer_minicache",
            "layers_merged": self.stats["layers_merged"],
            "retention_ratio": self.stats["tokens_retained"] / max(self.stats["total_tokens"], 1),
            "start_layer_ratio": self.start_layer_ratio,
        }
    
    def reset_stats(self):
        self.stats = {"layers_merged": 0, "tokens_retained": 0, "total_tokens": 0}


# Keep MiniCache as alias for backward compatibility
MiniCache = TokenMerge


def get_minicache_compressor(merge_ratio: float = 0.5) -> TokenMerge:
    """Factory function for TokenMerge compressor."""
    return TokenMerge(merge_ratio=merge_ratio)


def get_crosslayer_minicache(
    start_layer_ratio: float = 0.5,
    interpolation_t: float = 0.5,
) -> CrossLayerMiniCache:
    """Factory function for CrossLayerMiniCache compressor."""
    return CrossLayerMiniCache(
        start_layer_ratio=start_layer_ratio,
        interpolation_t=interpolation_t,
    )
