"""
Utility functions for the evaluation package.

This module provides helper functions used across the evaluation package.
"""

import logging
from pathlib import Path
from typing import Union

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to the directory to ensure exists.
        
    Returns
    -------
    Path
        The path to the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def setup_logging(verbose: bool = True) -> logging.Logger:
    """Set up logging configuration.
    
    Parameters
    ----------
    verbose : bool, optional
        Whether to enable verbose logging, by default True.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(levelname)s | %(message)s',
        force=True  # Force reconfiguration of the root logger
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    return logger 