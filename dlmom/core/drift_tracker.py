"""
Drift Tracker - Measures distribution shift from reference baseline.

Per DL-MoM paper Section 6.4:
- Logits KL drift: mean KL between next-token distributions at matched step indices
- Cosine drift: cosine distance between logits vectors

Usage:
    tracker = DriftTracker()
    
    # 1. Record reference baseline
    tracker.record_reference("sample_1", [logits_step0, logits_step1, ...])
    
    # 2. Compute drift for experiment runs
    kl, cosine = tracker.compute_drift("sample_1", [exp_logits_step0, exp_logits_step1, ...])
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """
    Compute KL(P || Q) from logits.
    
    Args:
        p_logits: Reference logits (the "true" distribution)
        q_logits: Experiment logits (the "approximation")
    
    Returns:
        KL divergence value
    """
    # Convert to probabilities
    p = F.softmax(p_logits.float(), dim=-1)
    q = F.softmax(q_logits.float(), dim=-1)
    
    # KL divergence: sum(p * log(p/q))
    # Add small epsilon for numerical stability
    eps = 1e-10
    kl = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)
    
    return kl.mean().item()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute cosine distance (1 - cosine_similarity) between logit vectors.
    """
    a = a.float().flatten()
    b = b.float().flatten()
    
    similarity = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
    return (1 - similarity).item()


class DriftTracker:
    """
    Tracks distribution drift from a reference baseline.
    
    The reference baseline is typically the first experiment in a suite
    (e.g., A1.1 deterministic, A4.1 no compression) representing the
    "canonical" behavior without any ablation modifications.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Args:
            storage_dir: Optional directory to persist reference logits
        """
        self.storage_dir = Path(storage_dir) if storage_dir else None
        
        # In-memory cache: sample_id -> list of logits tensors (one per step)
        self._reference_cache: Dict[str, List[torch.Tensor]] = {}
        
        # Aggregated drift stats per sample
        self._kl_drifts: List[float] = []
        self._cosine_drifts: List[float] = []
        self._drift_events: List[int] = []
    
    def record_reference(
        self,
        sample_id: str,
        step_logits: List[torch.Tensor],
        persist: bool = False,
    ):
        """
        Record reference logits for a sample.
        
        Args:
            sample_id: Unique identifier for this sample
            step_logits: List of logits tensors, one per generation step
            persist: If True and storage_dir is set, save to disk
        """
        # Detach and move to CPU to save GPU memory
        cpu_logits = [l.detach().cpu() for l in step_logits]
        self._reference_cache[sample_id] = cpu_logits
        
        if persist and self.storage_dir:
            self._save_reference(sample_id, cpu_logits)
    
    def compute_drift(
        self,
        sample_id: str,
        step_logits: List[torch.Tensor],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute drift between experiment logits and reference.
        
        Args:
            sample_id: Sample identifier (must have reference recorded)
            step_logits: Experiment logits to compare
        
        Returns:
            Tuple of (kl_drift, cosine_drift, drift_events) or (None, None, None) if no reference
        """
        ref_logits = self._reference_cache.get(sample_id)
        
        if ref_logits is None:
            # Try loading from disk
            if self.storage_dir:
                ref_logits = self._load_reference(sample_id)
                if ref_logits:
                    self._reference_cache[sample_id] = ref_logits
        
        if ref_logits is None:
            return None, None
        
        # Compare at matched step indices (only up to min length)
        n_steps = min(len(ref_logits), len(step_logits))
        
        if n_steps == 0:
            return None, None
        
        kl_values = []
        cosine_values = []
        
        for i in range(n_steps):
            ref = ref_logits[i]
            exp = step_logits[i].detach().cpu()
            
            # Only compare if shapes match
            if ref.shape == exp.shape:
                kl_values.append(kl_divergence(ref, exp))
                cosine_values.append(cosine_distance(ref, exp))
        
        if not kl_values:
            return None, None, None
        
        mean_kl = sum(kl_values) / len(kl_values)
        mean_cosine = sum(cosine_values) / len(cosine_values)
        
        # Count drift events (KL > threshold for >= 3 consecutive steps)
        # Default threshold based on "Compression stable KL < 0.05", so maybe 0.1 is a significant event?
        # User didn't specify threshold value, defaulting to 0.1
        threshold = 0.1
        drift_events = 0
        consecutive = 0
        for kl in kl_values:
            if kl > threshold:
                consecutive += 1
                if consecutive == 3: # Trigger event on 3rd step
                     drift_events += 1
                elif consecutive > 3: # Already counted, keep counting as same event? 
                     # "Drift > threshold for >= 3 steps". 
                     # Usually means each sequence counts as 1 event? 
                     # Or per-sample binary? "Count per sample".
                     # I will count *distinct sequences* of >=3 steps?
                     # Or just flag the sample? "Count per sample" implies integer.
                     # Let's count sequences.
                     pass
            else:
                consecutive = 0
                
        # Track for aggregate stats
        self._kl_drifts.append(mean_kl)
        self._cosine_drifts.append(mean_cosine)
        self._drift_events.append(drift_events)
        
        return mean_kl, mean_cosine, drift_events
    
    def get_aggregate_drift(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Get mean drift across all measured samples.
        
        Returns:
            (mean_kl_drift, mean_cosine_drift, mean_drift_events)
        """
        if not self._kl_drifts:
            return None, None, None
        
        mean_kl = sum(self._kl_drifts) / len(self._kl_drifts)
        mean_cosine = sum(self._cosine_drifts) / len(self._cosine_drifts)
        mean_events = sum(self._drift_events) / len(self._drift_events)
        
        return mean_kl, mean_cosine, mean_events
    
    def reset_aggregates(self):
        """Reset aggregate drift statistics (call between experiments)."""
        self._kl_drifts = []
        self._cosine_drifts = []
        self._drift_events = []
    
    def has_reference(self, sample_id: str) -> bool:
        """Check if reference exists for a sample."""
        if sample_id in self._reference_cache:
            return True
        if self.storage_dir:
            ref_path = self.storage_dir / f"ref_{sample_id}.pt"
            return ref_path.exists()
        return False
    
    def clear_references(self):
        """Clear all cached references."""
        self._reference_cache.clear()
    
    def _save_reference(self, sample_id: str, logits: List[torch.Tensor]):
        """Save reference logits to disk."""
        if not self.storage_dir:
            return
        
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        ref_path = self.storage_dir / f"ref_{sample_id}.pt"
        
        # Save as list of tensors
        torch.save(logits, ref_path)
    
    def _load_reference(self, sample_id: str) -> Optional[List[torch.Tensor]]:
        """Load reference logits from disk."""
        if not self.storage_dir:
            return None
        
        ref_path = self.storage_dir / f"ref_{sample_id}.pt"
        if not ref_path.exists():
            return None
        
        return torch.load(ref_path, weights_only=True)
