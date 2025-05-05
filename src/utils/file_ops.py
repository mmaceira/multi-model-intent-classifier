"""Common file-system utilities used across the project."""

from pathlib import Path

def ensure_dir(path: str | Path) -> Path:
    """Create *path* (including parents) if it does not exist and return a ``Path``.
    
    Having a single implementation avoids subtle inconsistencies and
    circular imports when each module defines its own helper.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
