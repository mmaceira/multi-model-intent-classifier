#!/usr/bin/env python
"""
Pre-commit hook to prevent committing .env files.

This script checks if .env files are being staged and fails if they are.
"""

import subprocess
import sys


def main():
    """Check if .env files are staged for commit."""
    from intent_classifier.utils.paths import get_repo_root

    repo_root = get_repo_root()

    # Check if .env is staged (but allow deletions)
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-status"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
        staged_changes = result.stdout.strip().split("\n") if result.stdout.strip() else []

        # Filter out .env files that are being added or modified (but allow deletions)
        env_files = []
        for change in staged_changes:
            if not change.strip():
                continue
            # Format: STATUS\tFILENAME
            parts = change.split("\t", 1)
            if len(parts) == 2:
                status, filename = parts
                # Allow deletions (status 'D'), block additions ('A') and modifications ('M')
                if status != "D" and ".env" in filename and not filename.endswith(".template"):
                    env_files.append(filename)

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
