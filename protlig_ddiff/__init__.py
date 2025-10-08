"""
Protein-Ligand Discrete Diffusion (protlig_ddiff) Package

A comprehensive package for discrete diffusion models applied to protein sequences.
Includes SUBS loss implementation, V100-compatible models, and training utilities.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import key components for easy access
from .models.transformer_v100 import DiscDiffModel
from .processing.graph_lib import Absorbing
from .processing.noise_lib import LogLinearNoise
from .processing.subs_loss import subs_loss, subs_loss_with_curriculum, compute_subs_metrics

__all__ = [
    "DiscDiffModel",
    "Absorbing", 
    "LogLinearNoise",
    "subs_loss",
    "subs_loss_with_curriculum",
    "compute_subs_metrics"
]
