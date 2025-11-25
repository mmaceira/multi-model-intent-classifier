"""
Global Seeding Utility

This module provides a centralized function to set all random seeds for reproducibility
across the entire codebase. It ensures deterministic behavior for:
- Python's random module
- NumPy
- PyTorch (used by sentence-transformers)
- Python's hash randomization (via PYTHONHASHSEED)

This should be called at the start of all entry points (CLI scripts, API, training scripts)
to ensure reproducibility.

Example:
    >>> from src.utils.seed import set_global_seed
    >>> set_global_seed(42)
"""

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility.

    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (if available, used by sentence-transformers)
    - PYTHONHASHSEED environment variable

    Args:
        seed: Random seed value (default: 42)

    Notes:
        - This should be called at the start of entry points (scripts, API, etc.)
        - For libraries/modules, use logger.getLogger(__name__) instead of basicConfig
        - Setting PYTHONHASHSEED requires the environment variable to be set before
          Python starts, so this function sets it but warns if it's too late
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed (if available)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For reproducibility with CUDA
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch not available, skip
        pass

    # Set PYTHONHASHSEED (must be set before Python starts, but we try anyway)
    # This is a best-effort attempt; for full effect, set it before starting Python:
    # PYTHONHASHSEED=42 python script.py
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Note: If PYTHONHASHSEED was not set before Python started, this won't take effect
    # until the next Python process. For maximum reproducibility, users should set it
    # in their environment or shell before running scripts.


def get_seed_from_config(config: dict, default: int = 42) -> int:
    """Extract seed from configuration dictionary.

    Args:
        config: Configuration dictionary that may contain a 'seed' or 'GENERAL_SEED' key
        default: Default seed value if not found in config

    Returns:
        Seed value from config or default
    """
    return config.get("seed", config.get("GENERAL_SEED", default))
