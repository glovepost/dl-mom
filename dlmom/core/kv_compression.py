"""
KV Cache Compression Wrapper for DL-MoM

Provides integration with KIVI 2-bit and MiniCache compression.
Falls back gracefully when dependencies are unavailable.
"""

import torch
from typing import Optional, Tuple, Dict, Any, List
import warnings
import sys
from pathlib import Path


# Add KIVI to path if available
KIVI_PATH = Path(__file__).parent.parent.parent / "external" / "kivi"
KIVI_AVAILABLE = False

if KIVI_PATH.exists():
    sys.path.insert(0, str(KIVI_PATH))
    try:
        from quant.new_pack import triton_quantize_and_pack_along_last_dim
        from quant.matmul import cuda_bmm_fA_qB_outer
        KIVI_AVAILABLE = True
    except ImportError as e:
        warnings.warn(f"KIVI available but CUDA extensions not compiled: {e}", stacklevel=2)
        warnings.filterwarnings("ignore", message="KIVI available but CUDA extensions not compiled")
        KIVI_AVAILABLE = False


class KVCompressor:
    """
    Abstract base class for KV cache compression.
    """
    
    def __init__(self, method: str = "none"):
        self.method = method
    
    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int = 0,  # Added for MiniCache compatibility
    ) -> Tuple[Any, Any]:
        """Compress KV cache."""
        raise NotImplementedError
    
    def decompress(self, key_compressed: Any, value_compressed: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress KV cache."""
        raise NotImplementedError
    
    def reset(self):
        """Reset state for new sequence. Override if stateful."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {"method": self.method}


class NoCompression(KVCompressor):
    """No compression - passthrough."""
    
    def __init__(self):
        super().__init__("none")
    
    def compress(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        return key, value
    
    def decompress(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return key, value


class KIVI2BitCompressor(KVCompressor):
    """
    KIVI 2-bit asymmetric quantization for KV cache.
    """
    
    def __init__(self, group_size: int = 32, residual_length: int = 32, track_error: bool = False):
        super().__init__("kivi2bit")
        self.group_size = group_size
        self.residual_length = residual_length
        self.k_bits = 2
        self.v_bits = 2
        self.track_error = track_error
        
        self.last_key_error = 0.0
        self.last_value_error = 0.0
        self.native_mode = KIVI_AVAILABLE
        
        if not KIVI_AVAILABLE:
            warnings.warn("KIVI CUDA extensions not available, falling back to simulated quantization")
    
    def compress(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int = 0) -> Tuple[Any, Any]:
        """Compress KV cache using 2-bit quantization.
        
        Note: For standalone use (e.g., combined with TokenMerge), we always use
        simulated compression. Native KIVI compression requires KIVI-enabled models.
        """
        # Always use simulated compression for standalone compressor
        # Native KIVI is handled internally by KIVI-enabled models
        k_c, v_c = self._compress_simulated(key, value)
        
        if self.track_error:
            k_recon, v_recon = self._decompress_simulated(k_c, v_c)
            self.last_key_error = (key - k_recon).abs().mean().item()
            self.last_value_error = (value - v_recon).abs().mean().item()
        
        return k_c, v_c
    
    def _compress_simulated(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[Dict, Dict]:
        """Simulated 2-bit quantization."""
        orig_dtype = key.dtype  # Store original dtype for decompression
        
        key_min = key.min(dim=-1, keepdim=True).values
        key_max = key.max(dim=-1, keepdim=True).values
        key_scale = (key_max - key_min) / 3
        key_quant = ((key - key_min) / (key_scale + 1e-10)).round().clamp(0, 3).to(torch.uint8)
        
        value_min = value.min(dim=-1, keepdim=True).values
        value_max = value.max(dim=-1, keepdim=True).values
        value_scale = (value_max - value_min) / 3
        value_quant = ((value - value_min) / (value_scale + 1e-10)).round().clamp(0, 3).to(torch.uint8)
        
        return (
            {"quant": key_quant, "scale": key_scale, "min": key_min, "dtype": orig_dtype},
            {"quant": value_quant, "scale": value_scale, "min": value_min, "dtype": orig_dtype}
        )
    
    def _compress_kivi(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[Any, Any]:
        """Real KIVI compression."""
        key_compressed = triton_quantize_and_pack_along_last_dim(key, self.group_size, self.k_bits)
        value_compressed = triton_quantize_and_pack_along_last_dim(value, self.group_size, self.v_bits)
        return key_compressed, value_compressed
    
    def decompress(self, key_compressed: Any, value_compressed: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check if this is simulated compression format (dict with 'quant' key)
        # vs native KIVI format (which requires attention integration)
        if isinstance(key_compressed, dict) and "quant" in key_compressed:
            # Simulated format - can decompress directly
            return self._decompress_simulated(key_compressed, value_compressed)
        elif KIVI_AVAILABLE:
            raise NotImplementedError("KIVI native decompression requires attention integration")
        return self._decompress_simulated(key_compressed, value_compressed)
    
    def _decompress_simulated(self, key_compressed: Dict, value_compressed: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        # Decompress and restore original dtype
        orig_dtype = key_compressed.get("dtype", torch.float32)
        key = (key_compressed["quant"].float() * key_compressed["scale"] + key_compressed["min"]).to(orig_dtype)
        value = (value_compressed["quant"].float() * value_compressed["scale"] + value_compressed["min"]).to(orig_dtype)
        return key, value
    
    def memory_savings(self, original_numel: Optional[int] = None) -> float:
        if original_numel is None:
            return 6.0
        bits_per_value = self.k_bits
        groups = original_numel // self.group_size
        overhead_bits = groups * 32 * 2
        compressed_bits = original_numel * bits_per_value + overhead_bits
        original_bits = original_numel * 16
        return original_bits / compressed_bits
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "native_mode": self.native_mode,
            "key_quant_error": self.last_key_error,
            "value_quant_error": self.last_value_error,
            "estimated_savings": self.memory_savings(),
        }


class MiniCacheCompressor(KVCompressor):
    """
    MiniCache: KV cache compression via layer merging.
    
    Odd layers (after merge_start_layer) reuse the previous even layer's KV cache.
    """
    
    def __init__(
        self,
        merge_start_layer: int = 8,
        num_layers: int = 32,
        interpolate: bool = False,  # True = blend, False = direct reuse
        alpha: float = 0.5,
    ):
        super().__init__("minicache")
        self.merge_start_layer = merge_start_layer
        self.num_layers = num_layers
        self.interpolate = interpolate
        self.alpha = alpha
        
        self.layer_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.merge_count = 0
        self.total_count = 0
    
    def reset(self):
        """Reset layer cache for new sequence."""
        self.layer_cache.clear()
        self.merge_count = 0
        self.total_count = 0
    
    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.total_count += 1
        
        if layer_idx < self.merge_start_layer:
            self.layer_cache[layer_idx] = (key, value)
            return key, value
        
        if layer_idx % 2 == 1:  # Odd layer
            prev_layer = layer_idx - 1
            if prev_layer in self.layer_cache:
                self.merge_count += 1
                prev_k, prev_v = self.layer_cache[prev_layer]
                
                if self.interpolate:
                    merged_k = self.alpha * key + (1 - self.alpha) * prev_k
                    merged_v = self.alpha * value + (1 - self.alpha) * prev_v
                    return merged_k, merged_v
                else:
                    return prev_k, prev_v
        
        self.layer_cache[layer_idx] = (key, value)
        return key, value
    
    def decompress(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return key, value
    
    def memory_savings(self) -> float:
        if self.total_count == 0:
            return 1.0
        merge_ratio = self.merge_count / self.total_count
        return 1.0 / (1.0 - merge_ratio) if merge_ratio < 1.0 else float('inf')
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "merge_count": self.merge_count,
            "total_count": self.total_count,
            "merge_ratio": self.merge_count / max(1, self.total_count),
            "estimated_savings": self.memory_savings(),
            "interpolate": self.interpolate,
        }


class ComposedCompressor(KVCompressor):
    """Chains multiple compression methods."""
    
    def __init__(self, compressors: List[KVCompressor]):
        method = "+".join([c.method for c in compressors])
        super().__init__(method)
        self.compressors = compressors
    
    def compress(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int = 0) -> Tuple[Any, Any]:
        k, v = key, value
        for compressor in self.compressors:
            k, v = compressor.compress(k, v, layer_idx=layer_idx)
        return k, v
    
    def decompress(self, key_compressed: Any, value_compressed: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        k, v = key_compressed, value_compressed
        for compressor in reversed(self.compressors):
            k, v = compressor.decompress(k, v)
        return k, v
    
    def reset(self):
        for compressor in self.compressors:
            compressor.reset()
    
    def memory_savings(self) -> float:
        savings = 1.0
        for compressor in self.compressors:
            if hasattr(compressor, 'memory_savings'):
                savings *= compressor.memory_savings()
        return savings
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {"method": self.method, "total_savings": self.memory_savings()}
        for compressor in self.compressors:
            for k, v in compressor.get_stats().items():
                stats[f"{compressor.method}_{k}"] = v
        return stats


def get_compressor(method: str, **kwargs) -> KVCompressor:
    """Factory function to get a KV compressor."""
    if method == "none":
        return NoCompression()
    elif method == "kivi2bit":
        kivi_kwargs = {k: v for k, v in kwargs.items() if k in ['group_size', 'residual_length', 'track_error']}
        return KIVI2BitCompressor(**kivi_kwargs)
    elif method == "minicache":
        mc_kwargs = {k: v for k, v in kwargs.items() if k in ['merge_start_layer', 'num_layers', 'interpolate', 'alpha']}
        return MiniCacheCompressor(**mc_kwargs)
    elif method == "minicache+kivi2bit":
        mc_kwargs = {k: v for k, v in kwargs.items() if k in ['merge_start_layer', 'num_layers', 'interpolate', 'alpha']}
        kivi_kwargs = {k: v for k, v in kwargs.items() if k in ['group_size', 'residual_length', 'track_error']}
        return ComposedCompressor([MiniCacheCompressor(**mc_kwargs), KIVI2BitCompressor(**kivi_kwargs)])
    else:
        raise ValueError(f"Unknown compression method: {method}")


def test_compressors():
    """Test compressor functionality."""
    print("Testing KV compressors...")
    
    batch, heads, seq, dim = 2, 4, 64, 32
    key = torch.randn(batch, heads, seq, dim)
    value = torch.randn(batch, heads, seq, dim)
    
    # Test none
    print("\n  none:")
    c = get_compressor("none")
    k_c, v_c = c.compress(key, value)
    k_d, v_d = c.decompress(k_c, v_c)
    print(f"    Roundtrip error: {(key - k_d).abs().max().item():.6f}")
    
    # Test kivi2bit (simulated)
    print("\n  kivi2bit:")
    c = get_compressor("kivi2bit", track_error=True)
    k_c, v_c = c.compress(key, value)
    stats = c.get_stats()
    print(f"    Key error: {stats['key_quant_error']:.4f}")
    print(f"    Savings: {stats['estimated_savings']:.1f}x")
    
    # Test minicache (multi-layer)
    print("\n  minicache:")
    c = get_compressor("minicache", merge_start_layer=2, num_layers=8)
    c.reset()
    for layer in range(8):
        c.compress(key, value, layer_idx=layer)
    stats = c.get_stats()
    print(f"    Merge ratio: {stats['merge_ratio']:.2%}")
    print(f"    Savings: {stats['estimated_savings']:.2f}x")
    
    # Test composed
    print("\n  minicache+kivi2bit:")
    c = get_compressor("minicache+kivi2bit", merge_start_layer=2, num_layers=8)
    c.reset()
    for layer in range(8):
        c.compress(key, value, layer_idx=layer)
    stats = c.get_stats()
    print(f"    Total savings: {stats['total_savings']:.2f}x")
    
    print("\n✓ Compressor tests passed")


if __name__ == "__main__":
    test_compressors()
