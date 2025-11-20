# config/notebook_setup.py
# Configuration setup module for pipeline scripts
# This module loads and processes config.yaml, creating convenient variables for use in pipeline scripts
import logging
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import yaml

# infer repo root from the location of this file
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))  # allow `import src.*`

with open(repo_root / "config" / "config.yaml") as fp:
    cfg = yaml.safe_load(fp)


# Function to substitute ${var} with values from the config
def substitute_vars(value, config):
    if isinstance(value, str):
        # Find all ${section.var} patterns and replace them with values from config
        var_pattern = r"\${([^}]+)}"
        for var_path in re.findall(var_pattern, value):
            if "." in var_path:
                section, var = var_path.split(".", 1)
                if section in config and var in config[section]:
                    value = value.replace(f"${{{var_path}}}", str(config[section][var]))
        return value
    return value


# Apply variable substitution to all values in config
for section_key, section_value in cfg.items():
    if isinstance(section_value, dict):
        for key, value in section_value.items():
            if isinstance(value, str) and "${" in value:
                cfg[section_key][key] = substitute_vars(value, cfg)

# Create specific variables from config.yaml sections
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
RAG_TOP_K = int(config_vars.get("MODEL_RAG_TOP_K"))

# Path variables with shorter names for backward compatibility
DATA_EXPLORATION_DIR = config_vars.get("PATHS_DATA_EXPLORATION_DIR")
EMB_DIR = config_vars.get("PATHS_EMBEDDINGS_DIR")
MODELS_DIR = config_vars.get("PATHS_MODELS_DIR")
PREDICTIONS_DIR = config_vars.get("PATHS_PREDICTIONS_DIR")
RESULTS_DIR = config_vars.get("PATHS_RESULTS_DIR")

# Set environment variables
if N_CLASSES is not None:
    os.environ["N_CLASSES"] = str(N_CLASSES)

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

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s | %(message)s", handlers=[logging.StreamHandler()]
)

# Create a logger that can be used throughout the project
logger = logging.getLogger(__name__)
