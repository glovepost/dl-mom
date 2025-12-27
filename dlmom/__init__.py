"""
DL-MoM: Deep-Latent Mixture of Models

Training-free architecture for latent-space multi-expert collaboration.
"""

from .core.belief_packet import BeliefPacket, reconstruct_soft_input
from .core.entropy_controller import EntropyController, normalized_entropy
from .core.consensus import contrastive_ties, ContrastiveConsensus

__version__ = "0.1.0"
__all__ = [
    "BeliefPacket",
    "reconstruct_soft_input",
    "EntropyController",
    "normalized_entropy",
    "contrastive_ties",
    "ContrastiveConsensus",
]
