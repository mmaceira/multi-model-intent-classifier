#!/usr/bin/env python
"""
Commit message format checker for pre-commit.

This script enforces a commit message format. You can customize the format
by modifying the validation logic below.
"""

import sys
from pathlib import Path


def check_commit_message():
    """Check if commit message follows the required format."""
    # Get commit message file path from pre-commit
    commit_msg_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    if not commit_msg_file or not commit_msg_file.exists():
        print("Error: Commit message file not found")
        return 1

    # Read commit message
    commit_msg = commit_msg_file.read_text(encoding="utf-8").strip()

    if not commit_msg:
        print("Error: Commit message is empty")
        return 1

    # Basic validation: ensure message is not too short
    if len(commit_msg) < 10:
        print("Error: Commit message is too short (minimum 10 characters)")
        return 1

    # Optional: enforce conventional commits format
    # Uncomment and customize as needed
    # conventional_commit_prefixes = [
    #     "feat:", "fix:", "docs:", "style:", "refactor:",
    #     "perf:", "test:", "chore:", "ci:", "build:"
    # ]
    #
    # first_line = commit_msg.split('\n')[0]
    # if not any(first_line.startswith(prefix) for prefix in conventional_commit_prefixes):
    #     print("Error: Commit message should follow conventional commits format")
    #     print(f"Expected format: <type>: <description>")
    #     print(f"Valid types: {', '.join(conventional_commit_prefixes)}")
    #     return 1

    return 0


if __name__ == "__main__":
    sys.exit(check_commit_message())
