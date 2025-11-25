#!/usr/bin/env python
"""
Pre-commit hook to prevent committing .env files.

This script checks if .env files are being staged and fails if they are.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Check if .env files are staged for commit."""
    repo_root = Path(__file__).resolve().parents[1]

    # Check if .env is staged
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
        staged_files = result.stdout.strip().split("\n") if result.stdout.strip() else []

        env_files = [f for f in staged_files if ".env" in f and not f.endswith(".template")]

        if env_files:
            print("❌ ERROR: Attempting to commit .env files!")
            print("\nThe following .env files are staged:")
            for f in env_files:
                print(f"  - {f}")
            print("\n💡 .env files should never be committed to the repository.")
            print("   Use .env.template as a template instead.")
            print("\nTo unstage these files:")
            print("  git reset HEAD <file>")
            sys.exit(1)

        print("✅ No .env files detected in staged changes.")
        sys.exit(0)

    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not check git status: {e}")
        # Don't fail if git is not available (e.g., in some CI environments)
        sys.exit(0)
    except FileNotFoundError:
        print("⚠️  Warning: git not found. Skipping .env check.")
        sys.exit(0)


if __name__ == "__main__":
    main()
