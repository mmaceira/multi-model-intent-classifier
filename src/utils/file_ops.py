"""Common file-system utilities used across the project."""

from pathlib import Path
from typing import Union

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory path (including parents) if it does not exist and return a Path object.
    
    This is a centralized implementation to avoid subtle inconsistencies and
    circular imports when each module defines its own helper.
    
    Parameters
    ----------
    path : Union[str, Path]
        The directory path to ensure exists
        
    Returns
    -------
    Path
        The Path object pointing to the directory
        
    Examples
    --------
    >>> ensure_dir("output/results")
    PosixPath('output/results')
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
