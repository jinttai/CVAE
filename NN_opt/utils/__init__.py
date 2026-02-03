"""
Utility functions for NN optimization.

Includes data generation, visualization, losses, and logging utilities.
"""

from .logging_utils import setup_logger, get_logger, TrainingLogger

__all__ = ['setup_logger', 'get_logger', 'TrainingLogger']
