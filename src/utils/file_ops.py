"""Common file-system utilities used across the project.

This module provides centralized file-system operations to ensure consistency
across the project and avoid circular imports. It includes functions for
directory management and file path handling.

Functions:
- ensure_dir: Create directory path if it doesn't exist
"""

from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory path (including parents) if it does not exist and return a Path object.
    
    This is a centralized implementation to avoid subtle inconsistencies and
    circular imports when each module defines its own helper. The function
    handles both string and Path inputs, and creates all necessary parent
    directories.
    
    Parameters
    ----------
    path : Union[str, Path]
        The directory path to ensure exists. Can be either a string or a Path object.
        
    Returns
    -------
    Path
        The Path object pointing to the directory. This can be used for further
        path operations.
        
    Raises
    ------
    OSError
        If the directory cannot be created due to permission issues or other
        system-level problems.
        
    Examples
    --------
    >>> ensure_dir("output/results")
    PosixPath('output/results')
    
    >>> ensure_dir(Path("output/results"))
    PosixPath('output/results')
    
    >>> path = ensure_dir("output/results")
    >>> path / "file.txt"
    PosixPath('output/results/file.txt')
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
