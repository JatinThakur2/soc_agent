"""Centralized logging configuration."""
import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Create a logger with both console and optional file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
