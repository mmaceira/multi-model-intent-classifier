# config/notebook_setup.py
# Configuration setup module for pipeline scripts
# This module loads and processes layered experiment configs using
# ``config/experiments/{dataset}/{variant}.yaml`` and exposes convenient
# variables for use in pipeline scripts.
import logging
import os
import random
from pathlib import Path

# Optional matplotlib import (only needed for plotting, which is in optional dependencies)
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend to prevent pop-ups
except ImportError:
    # matplotlib not installed - that's fine, we only need it for plotting
    pass

import numpy as np

# Import centralized config loader
from intent_classifier.utils.config_loader import load_config_with_metadata
from intent_classifier.utils.paths import get_repo_root

# Get repo root
repo_root = get_repo_root()

# Load config using centralized loader
config_metadata = load_config_with_metadata()
cfg = config_metadata["config"]
dataset_name = config_metadata["dataset_name"]
config_name = config_metadata["config_name"]
label_type = config_metadata["label_type"]
config_path = config_metadata["config_path"]
config_file = config_metadata["config_file"]

# Create specific variables from config sections
# This flattens the hierarchical config into module-level variables with prefixes

# Dictionary to store flattened variables for easy reference
config_vars = {}

# Process each section and create variables
for section_key, section_value in cfg.items():
    if isinstance(section_value, dict):
        for key, value in section_value.items():
            # Create variable name: uppercase with section prefix
            var_name = f"{section_key.upper()}_{key.upper()}"

            # Handle path creation for items in the paths section.
            # Paths are already derived from run_id in code; we only need to
            # convert them to absolute Paths rooted at the repository.
            if section_key == "paths" and isinstance(value, str):
                value = repo_root / value

            # Store in the global namespace and our tracking dictionary
            globals()[var_name] = value
            config_vars[var_name] = value

# Set frequently used variables as top-level for backward compatibility
# These are common variables that may be used directly in code
N_CLASSES = config_vars.get("DATASET_N_CLASSES")
N_SAMPLES_PER_CLASS = config_vars.get("DATASET_N_SAMPLES_PER_CLASS")
SEED = config_vars.get("GENERAL_SEED")
RUN_NAME = config_vars.get("GENERAL_RUN_NAME")
RAG_TOP_K = int(config_vars.get("MODEL_RAG_TOP_K", 25))  # Default to 25 if not found

# Path variables with shorter names, mapped to the new output schema.
# The central config loader always derives ``paths`` from ``run_id`` using
# ``compute_paths``, so we simply convert those to absolute Paths here.
_paths_cfg = cfg.get("paths", {})

DATA_EXPLORATION_DIR = repo_root / _paths_cfg.get("dataset_dir", "output/runs/unknown/dataset")
EMB_DIR = repo_root / _paths_cfg.get("features_dir", "output/runs/unknown/features")
MODELS_DIR = repo_root / _paths_cfg.get("models_dir", "output/runs/unknown/models")
PREDICTIONS_DIR = repo_root / _paths_cfg.get("eval_dir", "output/runs/unknown/eval")
RESULTS_DIR = repo_root / _paths_cfg.get("compare_dir", "output/runs/unknown/compare")

# Set environment variables
if N_CLASSES is not None:
    os.environ["N_CLASSES"] = str(N_CLASSES)

# Surface key paths for downstream modules without forcing them to import this module.
if EMB_DIR is not None:
    os.environ["EMBEDDINGS_DIR"] = str(EMB_DIR)
if MODELS_DIR is not None:
    os.environ.setdefault("MODELS_DIR", str(MODELS_DIR))

# Expose Ollama endpoint and embedding model to downstream components (e.g. embedders)
ollama_endpoint = config_vars.get("MODEL_OLLAMA_ENDPOINT")
if ollama_endpoint is not None:
    # Dedicated env var used by embedding backends
    os.environ.setdefault("MODEL_OLLAMA_ENDPOINT", str(ollama_endpoint))
    # Also populate standard Ollama base var if not already set
    os.environ.setdefault("OLLAMA_API_BASE", str(ollama_endpoint))

# Disable HuggingFace tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set random seeds
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)

# Print repository info
print(f"Repository Root: {repo_root}")
print(f"Configuration: {cfg}")

# Print all dynamically created variables grouped by section
print("\n=== Configuration Variables ===")
for section_key in sorted(cfg.keys()):
    print(f"\n[{section_key.upper()}]")
    section_vars = {k: v for k, v in config_vars.items() if k.startswith(f"{section_key.upper()}_")}
    for var_name in sorted(section_vars.keys()):
        print(f"  {var_name}: {section_vars[var_name]}")

# Create directories for all path variables
print("\n=== Creating Directories ===")
for var_name, value in config_vars.items():
    if var_name.startswith("PATHS_") and isinstance(value, Path):
        value.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {value}")

# Create a logger that can be used throughout the project
# Note: Do not call basicConfig here - this is a module, not an entry point.
# Entry points (pipeline scripts) should configure logging.
logger = logging.getLogger(__name__)
