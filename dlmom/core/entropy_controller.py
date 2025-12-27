"""
Trend-Based Entropy Controller for DL-MoM.

Implements adaptive switching between latent (exploration) and explicit (exploitation) modes.
Based on SwiReasoning's relative entropy approach.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
from collections import deque


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized entropy H / log|V|.
    
    Args:
        logits: [batch, vocab_size] or [vocab_size]
    
    Returns:
        Normalized entropy in [0, 1]
    """
    # Convert to float32 for numerical stability (float16 causes entropy=0 issues)
    logits_f32 = logits.float()
    probs = F.softmax(logits_f32, dim=-1)
    log_probs = F.log_softmax(logits_f32, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    vocab_size = logits.shape[-1]
    max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32, device=logits.device))
    
    return entropy / max_entropy


@dataclass
class ControllerState:
    """Internal state of the entropy controller."""
    entropy_window: deque = field(default_factory=lambda: deque(maxlen=10))
    switch_count: int = 0
    current_mode: str = "latent"
    total_steps: int = 0
    steps_in_current_mode: int = 0
    reference_entropy: Optional[float] = None  # SwiReasoning-style reference


class EntropyController:
    """
    Trend-based entropy controller with SwiReasoning-style mode switching.
    
    Key features (from SwiReasoning):
    - Relative entropy comparison: switches when current entropy crosses reference
    - Window-based delay: must stay in mode for `window_size` steps before can switch
    - No absolute convergence threshold for early stopping
    - Updates reference entropy on each mode switch
    """
    
    def __init__(
        self,
        alpha: float = 0.60,
        window_size: int = 5,
        switch_cap: int = 10,
        max_steps: int = 50,
        convergence_threshold: float = 0.05,  # Lower - Qwen-Math is very confident
        slope_threshold: float = -0.02,
        plateau_variance: float = 0.005,
    ):
        """
        Args:
            alpha: Normalized entropy threshold (used for initial mode selection)
            window_size: Minimum steps in mode before allowing switch
            switch_cap: Maximum mode switches allowed
            max_steps: Hard cap on total steps (prevents infinite loops)
            convergence_threshold: Only stop if entropy is extremely low
            slope_threshold: Negative slope indicating convergence
            plateau_variance: Variance threshold for plateau detection
        """
        self.alpha = alpha
        self.window_size = window_size
        self.switch_cap = switch_cap
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
        self.slope_threshold = slope_threshold
        self.plateau_variance = plateau_variance
        
        self.state = ControllerState(entropy_window=deque(maxlen=window_size))
    
    def reset(self):
        """Reset controller state for new sample."""
        self.state = ControllerState(entropy_window=deque(maxlen=self.window_size))
    
    def _compute_trend(self) -> Tuple[float, float, float]:
        """
        Compute trend statistics for entropy window (pure torch).
        
        Returns:
            (slope, variance, mean) of recent entropies
        """
        if len(self.state.entropy_window) < 2:
            return 0.0, float('inf'), self.state.entropy_window[-1] if self.state.entropy_window else 1.0
        
        entropies = torch.tensor(list(self.state.entropy_window), dtype=torch.float32)
        n = len(entropies)
        x = torch.arange(n, dtype=torch.float32)
        
        x_mean, y_mean = x.mean(), entropies.mean()
        slope = ((x - x_mean) * (entropies - y_mean)).sum() / ((x - x_mean) ** 2).sum().clamp(min=1e-6)
        variance = entropies.var().item()
        
        return slope.item(), variance, y_mean.item()
    
    def step(
        self,
        entropy: float,
        eos_generated: bool = False,
    ) -> Tuple[str, bool]:
        """
        Process one entropy observation and decide next mode (SwiReasoning-style).
        
        Args:
            entropy: Normalized entropy for current step
            eos_generated: Whether EOS token was generated
        
        Returns:
            (mode, should_stop) where mode is "latent" or "explicit"
        """
        self.state.entropy_window.append(entropy)
        self.state.total_steps += 1
        self.state.steps_in_current_mode += 1
        
        # === STOPPING CONDITIONS ===
        
        # 0. EOS generated - always stop
        if eos_generated:
            return self.state.current_mode, True
        
        # 1. Hard step limit
        if self.state.total_steps >= self.max_steps:
            return self.state.current_mode, True
        
        # 2. Extremely low entropy (very confident) - lowered threshold
        if entropy < self.convergence_threshold:
            return "explicit", True
        
        # 3. Plateau detection
        slope, variance, mean_entropy = self._compute_trend()
        window_full = len(self.state.entropy_window) >= self.window_size
        
        if window_full and variance < self.plateau_variance and abs(slope) < 0.01:
            return self.state.current_mode, True
        
        # === MODE SWITCHING (SwiReasoning-style) ===
        
        # First step: set reference entropy and initial mode
        if self.state.total_steps == 1:
            self.state.reference_entropy = entropy
            self.state.current_mode = "latent" if entropy > self.alpha else "explicit"
            return self.state.current_mode, False
        
        # Check switch cap
        if self.state.switch_count >= self.switch_cap:
            return "explicit", False
        
        # Window-based switch delay (must stay in mode for window_size steps)
        allow_switch = self.state.steps_in_current_mode >= self.window_size
        
        # Relative entropy comparison (SwiReasoning-style)
        ref = self.state.reference_entropy
        
        if self.state.current_mode == "latent" and entropy < ref:
            # Entropy dropped below reference → switch to explicit
            self.state.switch_count += 1
            self.state.current_mode = "explicit"
            self.state.steps_in_current_mode = 0
            self.state.reference_entropy = entropy  # Update reference
            
        elif self.state.current_mode == "explicit" and entropy > ref and allow_switch:
            # Entropy rose above reference → switch back to latent
            self.state.switch_count += 1
            self.state.current_mode = "latent"
            self.state.steps_in_current_mode = 0
            self.state.reference_entropy = entropy  # Update reference
        
        return self.state.current_mode, False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics for logging."""
        slope, variance, mean = self._compute_trend()
        return {
            "switch_count": self.state.switch_count,
            "total_steps": self.state.total_steps,
            "current_mode": self.state.current_mode,
            "steps_in_mode": self.state.steps_in_current_mode,
            "reference_entropy": self.state.reference_entropy,
            "entropy_mean": mean,
            "entropy_slope": slope,
            "entropy_variance": variance,
            "window_size": len(self.state.entropy_window),
        }


def analyze_entropy_trajectory(entropies: list, controller: EntropyController) -> Dict[str, Any]:
    """
    Analyze an entropy trajectory post-hoc.
    
    Args:
        entropies: List of normalized entropy values
        controller: Controller instance (for threshold access)
    
    Returns:
        Analysis dict with statistics
    """
    import numpy as np
    entropies = np.array(entropies)
    
    return {
        "min": float(entropies.min()),
        "max": float(entropies.max()),
        "mean": float(entropies.mean()),
        "final": float(entropies[-1]) if len(entropies) > 0 else 0,
        "below_alpha_fraction": float((entropies < controller.alpha).mean()),
        "below_convergence_fraction": float((entropies < controller.convergence_threshold).mean()),
        "trend_slope": float(np.polyfit(np.arange(len(entropies)), entropies, 1)[0]) if len(entropies) > 1 else 0,
        "num_steps": len(entropies),
    }
