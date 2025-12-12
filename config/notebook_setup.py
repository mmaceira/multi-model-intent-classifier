# config/notebook_setup.py
# Configuration setup module for pipeline scripts
# This module loads and processes config files from the structure:
# config/dataset/{dataset_name}/{config_name}.yaml
# Creates convenient variables for use in pipeline scripts
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

            # Handle path creation for items in the paths section
            if section_key == "paths":
                # If we detected label_type and dataset_name, add prefix to output paths
                if label_type and dataset_name and value.startswith("output/"):
                    # Extract the part after "output/" (e.g., "${general.run_name}/embeddings")
                    path_suffix = value.replace("output/", "", 1)
                    # Build new path: output/{label_type}/{dataset_name}/{path_suffix}
                    value = f"output/{label_type}/{dataset_name}/{path_suffix}"
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

# Path variables with shorter names for backward compatibility
DATA_EXPLORATION_DIR = config_vars.get("PATHS_DATA_EXPLORATION_DIR")
EMB_DIR = config_vars.get("PATHS_EMBEDDINGS_DIR")
MODELS_DIR = config_vars.get("PATHS_MODELS_DIR")
PREDICTIONS_DIR = config_vars.get("PATHS_PREDICTIONS_DIR")
RESULTS_DIR = config_vars.get("PATHS_RESULTS_DIR")

# Set environment variables
if N_CLASSES is not None:
    os.environ["N_CLASSES"] = str(N_CLASSES)

# Surface key paths for downstream modules without forcing them to import this module.
if EMB_DIR is not None:
    os.environ["EMBEDDINGS_DIR"] = str(EMB_DIR)
if MODELS_DIR is not None:
    os.environ.setdefault("MODELS_DIR", str(MODELS_DIR))

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
