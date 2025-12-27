#!/usr/bin/env python3
"""
Phase 0: ROCm Environment Validation for DL-MoM.

Checks compatibility of core components on AMD gfx1151.
"""

import sys
import os

def check_pytorch():
    """Check PyTorch + ROCm."""
    print("=" * 50)
    print("1. PyTorch + ROCm")
    print("=" * 50)
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  HIP version: {getattr(torch.version, 'hip', 'N/A')}")
            # Quick inference test
            x = torch.randn(1, 768, device="cuda")
            y = torch.nn.Linear(768, 768).cuda()(x)
            print(f"  ✓ Basic tensor ops work")
        else:
            print(f"  ✗ GPU not available")
            return False
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_transformers():
    """Check Transformers basic inference."""
    print("\n" + "=" * 50)
    print("2. Transformers Basic Inference")
    print("=" * 50)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_id = "gpt2"  # Small model for testing
        print(f"  Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
        model = model.cuda()
        
        inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"  ✓ Model inference works")
        print(f"  Logits shape: {outputs.logits.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_flash_attention():
    """Check Flash Attention (AOTriton)."""
    print("\n" + "=" * 50)
    print("3. Flash Attention (AOTriton)")
    print("=" * 50)
    try:
        import torch
        
        # Check if AOTriton is enabled
        aotriton_enabled = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0") == "1"
        print(f"  AOTRITON_ENABLE_EXPERIMENTAL: {aotriton_enabled}")
        
        # Try SDPA
        from torch.nn.functional import scaled_dot_product_attention
        q = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16)
        
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            try:
                out = scaled_dot_product_attention(q, k, v)
                print(f"  ✓ Flash Attention works")
                return True
            except RuntimeError as e:
                print(f"  ⚠ Flash path failed, falling back: {e}")
        
        # Fallback to any SDPA
        out = scaled_dot_product_attention(q, k, v)
        print(f"  ✓ SDPA works (may not be Flash)")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_kivi():
    """Check KIVI 2-bit quantization availability."""
    print("\n" + "=" * 50)
    print("4. KIVI 2-bit Quantization")
    print("=" * 50)
    try:
        # KIVI is typically custom code, check if we have it
        kivi_path = os.path.join(os.path.dirname(__file__), "..", "external", "kivi")
        if os.path.exists(kivi_path):
            print(f"  ✓ KIVI found at {kivi_path}")
            return True
        else:
            print(f"  ⚠ KIVI not found (will skip A4.2/A4.4)")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_minicache():
    """Check MiniCache availability."""
    print("\n" + "=" * 50)
    print("5. MiniCache")
    print("=" * 50)
    try:
        minicache_path = os.path.join(os.path.dirname(__file__), "..", "external", "minicache")
        if os.path.exists(minicache_path):
            print(f"  ✓ MiniCache found at {minicache_path}")
            return True
        else:
            print(f"  ⚠ MiniCache not found (will skip A4.3/A4.4)")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_dlmom_modules():
    """Check DL-MoM core modules."""
    print("\n" + "=" * 50)
    print("6. DL-MoM Core Modules")
    print("=" * 50)
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        
        from dlmom.core.belief_packet import BeliefPacket, belief_packet_from_logits
        from dlmom.core.entropy_controller import EntropyController, normalized_entropy
        from dlmom.core.consensus import contrastive_ties, ContrastiveConsensus
        
        print(f"  ✓ belief_packet imported")
        print(f"  ✓ entropy_controller imported")
        print(f"  ✓ consensus imported")
        
        # Quick test
        import torch
        logits = torch.randn(2, 1000)
        packet = belief_packet_from_logits(logits, top_k=10)
        print(f"  ✓ belief_packet_from_logits works: {packet.token_ids.shape}")
        
        entropy = normalized_entropy(logits)
        print(f"  ✓ normalized_entropy works: {entropy.shape}")
        
        merged = contrastive_ties([logits, logits + 0.1])
        print(f"  ✓ contrastive_ties works: {merged.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_memory():
    """Check GPU memory availability."""
    print("\n" + "=" * 50)
    print("7. GPU Memory")
    print("=" * 50)
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"  Total: {total:.1f} GB")
            print(f"  Reserved: {reserved:.1f} GB")
            print(f"  Allocated: {allocated:.1f} GB")
            print(f"  Free: {total - reserved:.1f} GB")
            
            # Estimate if 3x 7B models fit
            model_size = 14  # GB per 7B bf16 model
            needed = 3 * model_size + 20  # 3 models + overhead
            if total > needed:
                print(f"  ✓ Sufficient for 3x 7B ensemble ({needed:.0f} GB needed)")
            else:
                print(f"  ⚠ May be tight for 3x 7B ensemble ({needed:.0f} GB needed)")
            return True
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("DL-MoM Phase 0: Environment Validation")
    print("=" * 50)
    
    results = {
        "pytorch": check_pytorch(),
        "transformers": check_transformers(),
        "flash_attention": check_flash_attention(),
        "kivi": check_kivi(),
        "minicache": check_minicache(),
        "dlmom_modules": check_dlmom_modules(),
        "memory": check_memory(),
    }
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, passed_check in results.items():
        status = "✓" if passed_check else "✗"
        print(f"  {status} {name}")
    
    print(f"\n  Passed: {passed}/{total}")
    
    if passed >= 4:  # Core components work
        print("\n  ➤ Environment validation PASSED. Proceed to Phase 1.")
        return 0
    else:
        print("\n  ➤ Environment validation FAILED. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
