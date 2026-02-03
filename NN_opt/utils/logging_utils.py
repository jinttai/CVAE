"""
Logging utilities for NN optimization.

Provides consistent logging configuration across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with console and optional file output.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default settings.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return setup_logger(name)


class TrainingLogger:
    """
    Specialized logger for training progress.

    Provides methods for logging training metrics with consistent formatting.
    """

    def __init__(self, name: str, log_file: Optional[Path] = None):
        self.logger = setup_logger(name, log_file=log_file)

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        loss: float,
        duration: float,
        **extra_metrics
    ) -> None:
        """Log epoch completion with metrics."""
        extra_str = " | ".join(f"{k}: {v:.6f}" for k, v in extra_metrics.items())
        msg = f"Epoch [{epoch}/{total_epochs}] | Loss: {loss:.6f} | Time: {duration:.2f}s"
        if extra_str:
            msg += f" | {extra_str}"
        self.logger.info(msg)

    def log_validation(self, epoch: int, val_loss: float) -> None:
        """Log validation results."""
        self.logger.info(f"   >>> Validation Loss: {val_loss:.6f}")

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

    def log_nan_detected(self, epoch: int) -> None:
        """Log NaN/Inf detection."""
        self.logger.warning(f"NaN/Inf loss detected at epoch {epoch}, skipping update")

    def log_training_start(self, device: str, model_name: str) -> None:
        """Log training start."""
        self.logger.info(f"=== {model_name} Training Start on {device} ===")

    def log_training_end(self, total_time: float) -> None:
        """Log training completion."""
        self.logger.info(f"Training Finished. Total Time: {total_time:.2f}s")
