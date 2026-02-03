"""
Neural Network Optimization module for trajectory planning.

This module provides:
- ConditionalDecoder: Latent-conditioned trajectory generator
- MLP: Deterministic baseline model
- PhysicsLayer: Physics simulation for loss computation
- Configuration and utilities
"""

from NN_opt.config import ROOT_DIR, training_config, model_config, trajectory_config

__all__ = ['ROOT_DIR', 'training_config', 'model_config', 'trajectory_config']
