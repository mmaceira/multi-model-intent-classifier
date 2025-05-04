# config/notebook_setup.py
from pathlib import Path
import os, yaml, sys, random, numpy as np
import re

# infer repo root from the location of this file
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))  # allow `import src.*`

with open(repo_root / "config" / "config.yaml") as fp:
    cfg = yaml.safe_load(fp)

# Function to substitute ${var} with values from the config
def substitute_vars(value, config):
    if isinstance(value, str):
        # Find all ${var} patterns and replace them with values from config
        var_pattern = r'\${([^}]+)}'
        for var_name in re.findall(var_pattern, value):
            if var_name in config:
                value = value.replace(f"${{{var_name}}}", str(config[var_name]))
        return value
    return value

# Apply variable substitution to all paths in config
for key in cfg:
    if isinstance(cfg[key], str) and "${" in cfg[key]:
        cfg[key] = substitute_vars(cfg[key], cfg)

DATA_EXPLORATION_DIR      = repo_root / cfg["data_exploration_dir"]
ARTIFACTS_DIR = repo_root / cfg["artifacts_dir"]
EMB_DIR       = repo_root / cfg["embeddings_dir"]
MODELS_DIR    = repo_root / cfg["models_dir"]
RESULTS_DIR   = repo_root / cfg["results_dir"]
N_CLASSES     = cfg["n_classes"]
CUTOFF_YEAR   = cfg["cutoff_year"]

os.environ["N_CLASSES"] = str(N_CLASSES)

# Disable HuggingFace tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = cfg["seed"]
random.seed(SEED)
np.random.seed(SEED)

# Print all variables
print(f"Repository Root: {repo_root}")
print(f"Configuration: {cfg}")
print(f"DATA_EXPLORATION_DIR: {DATA_EXPLORATION_DIR}")
print(f"ARTIFACTS_DIR: {ARTIFACTS_DIR}")
print(f"EMB_DIR: {EMB_DIR}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"RESULTS_DIR: {RESULTS_DIR}")
print(f"N_CLASSES: {N_CLASSES}")
print(f"CUTOFF_YEAR: {CUTOFF_YEAR}")
print(f"Environment N_CLASSES: {os.environ['N_CLASSES']}")
print(f"SEED: {SEED}")

# Create directories if they don't exist
DATA_EXPLORATION_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

