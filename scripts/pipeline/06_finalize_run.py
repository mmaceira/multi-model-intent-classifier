#!/usr/bin/env python
"""
06 – Run Finalization

This script normalizes the outputs for the current run into the canonical
`output/runs/<run_id>/...` layout and writes the reproducibility metadata
package (meta/* and manifest.json).
"""

import sys

from intent_classifier.utils.paths import get_repo_root

# Ensure repo root is on sys.path so that imports work when this is executed
repo_root = get_repo_root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def main() -> None:
    """Finalize the current run outputs."""
    from intent_classifier.pipeline.finalize import finalize_run

    finalize_run()


if __name__ == "__main__":
    main()
