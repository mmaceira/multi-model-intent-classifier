"""Model Loader Utility

This module provides functions to load model configurations from YAML files and
instantiate model objects dynamically based on configuration parameters.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.algorithms.embedding_logreg import EmbeddingLogReg
from src.algorithms.linear_svm import LinearSVMBigrams, LinearSVMClassifier
from src.algorithms.naive_bayes import NaiveBayesClassifier
from src.algorithms.transformer_logreg import TransformerLogReg
from src.rag import load_centroid, load_kmajority, load_llm
from src.rag.adapter_sklearn import RagSklearnAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Dictionary mapping class names to their actual classes
MODEL_CLASSES = {
    "NaiveBayesClassifier": NaiveBayesClassifier,
    "LinearSVMClassifier": LinearSVMClassifier,
    "LinearSVMBigrams": LinearSVMBigrams,
    "TransformerLogReg": TransformerLogReg,
    "EmbeddingLogReg": EmbeddingLogReg,
    "OpenAIEmbedLogReg": EmbeddingLogReg,  # Backward compatibility alias
    "RagSklearnAdapter": RagSklearnAdapter,
}


def substitute_vars(value: Any, config: Dict[str, Any]) -> Any:
    """Replace variable references in string values with their actual values from config."""
    if isinstance(value, str) and "${" in value:
        import re

        var_pattern = r"\${([^}]+)}"
        for var_path in re.findall(var_pattern, value):
            if "." in var_path:
                section, var = var_path.split(".", 1)
                if section in config and var in config[section]:
                    value = value.replace(f"${{{var_path}}}", str(config[section][var]))
    return value


def process_config_vars(config_value: Any, main_config: Dict[str, Any]) -> Any:
    """Recursively process a configuration value to substitute variables."""
    if isinstance(config_value, dict):
        return {k: process_config_vars(v, main_config) for k, v in config_value.items()}
    elif isinstance(config_value, list):
        return [process_config_vars(v, main_config) for v in config_value]
    elif isinstance(config_value, str):
        return substitute_vars(config_value, main_config)
    return config_value


def load_rag_model(params: Dict[str, Any]) -> RagSklearnAdapter:
    """Create a RAG model instance based on configuration parameters."""
    # Import here to avoid circular import
    from src.rag.vector_store import VectorStore

    method = params.get("method")
    # Ensure top_k is an integer
    top_k = int(params.get("top_k", 25))

    logger.info(f"Initializing RAG model with method={method}, top_k={top_k}, params={params}")

    if method == "kmajority":
        use_openai = params.get("use_openai", False)
        logger.info(f"Loading KMajority RAG model with top_k={top_k}, use_openai={use_openai}")
        return RagSklearnAdapter(load_kmajority(top_k=top_k, use_openai=use_openai))

    elif method == "centroid":
        logger.info(f"Loading Centroid RAG model with top_k={top_k}")
        return RagSklearnAdapter(load_centroid(top_k=top_k))

    elif method == "llm":
        model_name = params.get("model", "ollama/llama3.1:8b")
        use_openai = params.get("use_openai", False)

        # Configure embedder for local embeddings
        embedder = None
        if not use_openai and "embedder_model" in params:
            embedder_model = params.get("embedder_model")
            logger.info(f"Setting up custom embedder using model: {embedder_model}")

            def embedder(texts):
                return VectorStore.embed(embedder_model, texts)

        logger.info(
            f"Loading LLM RAG model with top_k={top_k}, model={model_name}, use_openai={use_openai}"
        )
        return RagSklearnAdapter(
            load_llm(top_k=top_k, model=model_name, use_openai=use_openai, embedder=embedder)
        )

    raise ValueError(f"Unknown RAG method: {method}")


def load_tuned_hyperparameters(
    hyperparams_path: Optional[str] = None,
    config_name: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load tuned hyperparameters from individual model files.

    Each model has its own file: best_{model_name}.yaml
    This function loads all available hyperparameter files from the directory.

    Args:
        hyperparams_path: Optional path to the hyperparameters directory.
                        If None, uses default location based on config name
        config_name: Optional config file name (without .yaml extension).
                    If None, tries to infer from CONFIG_FILE env var or uses "config"

    Returns:
        Dict mapping model IDs to their tuned hyperparameters
    """
    repo_root = Path(__file__).resolve().parents[2]

    # Determine config name if not provided
    if config_name is None:
        import os

        config_file = os.environ.get("CONFIG_FILE", "config.yaml")
        config_name = Path(config_file).stem  # Remove .yaml extension

    # Default to config/hyperparameters/{config_name}/ directory
    if hyperparams_path is None:
        hyperparams_dir = repo_root / "config" / "hyperparameters" / config_name
    else:
        hyperparams_dir = Path(hyperparams_path)
        if hyperparams_dir.is_file():
            # If a file path was provided, use its parent directory
            hyperparams_dir = hyperparams_dir.parent

    if not hyperparams_dir.exists():
        logger.info(f"Hyperparameters directory not found: {hyperparams_dir}")
        logger.info("Models will use default hyperparameters from config.")
        return {}

    # Load all best_{model_name}.yaml files from the directory
    hyperparams = {}
    pattern = "best_*.yaml"
    for file_path in hyperparams_dir.glob(pattern):
        try:
            model_name = file_path.stem.replace("best_", "")  # Remove "best_" prefix
            with open(file_path) as f:
                params = yaml.safe_load(f)
            if params:  # Only add if file has content
                hyperparams[model_name] = params
                logger.debug(f"Loaded hyperparameters for {model_name} from {file_path.name}")
        except Exception as e:
            logger.warning(f"Failed to load hyperparameters from {file_path.name}: {e}")
            continue

    if hyperparams:
        logger.info(
            f"Loaded tuned hyperparameters from {len(hyperparams)} files in {hyperparams_dir}"
        )
    else:
        logger.info(f"No hyperparameter files found in {hyperparams_dir}")
        logger.info("Models will use default hyperparameters from config.")

    return hyperparams


def load_models_from_config(
    models_config_path: str = "config/models_config.yaml",
    main_config_path: str = "config/config.yaml",
    hyperparams_path: Optional[str] = None,
    config_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and instantiate models based on configuration.

    Args:
        models_config_path: Path to the models configuration file
        main_config_path: Path to the main configuration file for variable substitution
        hyperparams_path: Path to tuned hyperparameters file. If None, hyperparameter tuning
                         results are not loaded. If file doesn't exist, defaults are used.

    Returns:
        Dict[str, Any]: Dictionary mapping display names to model instances
    """
    # Resolve paths relative to repo root
    repo_root = Path(__file__).resolve().parents[2]
    models_config_path = repo_root / models_config_path
    main_config_path = repo_root / main_config_path

    # Load configurations
    with open(models_config_path) as f:
        models_config = yaml.safe_load(f)

    with open(main_config_path) as f:
        main_config = yaml.safe_load(f)

    # Determine config name from main_config_path if not provided
    if config_name is None:
        config_name = Path(
            main_config_path
        ).stem  # e.g., "config_tiny_dataset" from "config_tiny_dataset.yaml"

    # Load tuned hyperparameters if path is provided
    tuned_hyperparams = {}
    if hyperparams_path is not None:
        tuned_hyperparams = load_tuned_hyperparameters(hyperparams_path, config_name=config_name)
    else:
        # Try to load from default location based on config name
        tuned_hyperparams = load_tuned_hyperparameters(config_name=config_name)

    # Validate configuration structure
    if not isinstance(models_config, dict):
        raise ValueError(
            f"models_config.yaml must contain a YAML dictionary, got {type(models_config)}"
        )

    if "models" not in models_config:
        raise ValueError("models_config.yaml is missing a top-level 'models' key")

    if not isinstance(models_config["models"], dict):
        raise ValueError("models_config.yaml 'models' key must contain a dictionary")

    # Initialize results dictionary
    models = {}

    # Process each model configuration
    for model_id, model_config in models_config.get("models", {}).items():
        # Skip disabled models
        if not model_config.get("enabled", True):
            continue

        # Get model details
        class_name = model_config.get("class")
        display_name = model_config.get("name", model_id)
        params = process_config_vars(model_config.get("params", {}), main_config)

        # Override with tuned hyperparameters if available
        # Map model_id to hyperparameter key (they might differ)
        hyperparam_key = model_id
        # Handle special mappings
        if model_id == "naive_bayes":
            hyperparam_key = "naive_bayes"
        elif model_id == "linear_svm":
            hyperparam_key = "linear_svm"
        elif model_id == "linear_svm_bigrams":
            hyperparam_key = "linear_svm_bigrams"
        elif model_id == "transformer_logreg":
            hyperparam_key = "transformer_logreg"
        elif model_id in ("embedding_logreg", "openai_logreg"):
            # Support both new name (embedding_logreg) and old name (openai_logreg)
            hyperparam_key = "embedding_logreg"  # Use consistent key name
        elif model_id == "rag_kmajority":
            hyperparam_key = "rag_kmajority"
        elif model_id == "rag_centroid":
            hyperparam_key = "rag_centroid"
        elif model_id == "rag_llm":
            hyperparam_key = "rag_llm"

        if hyperparam_key in tuned_hyperparams:
            tuned_params = tuned_hyperparams[hyperparam_key]
            # Remove f1 score if present (it's metadata, not a hyperparameter)
            tuned_params = {k: v for k, v in tuned_params.items() if k != "f1"}

            # Convert C to Cs for LogReg models (they expect Cs as a sequence)
            if (
                class_name in ("TransformerLogReg", "EmbeddingLogReg", "OpenAIEmbedLogReg")
                and "C" in tuned_params
            ):
                tuned_params["Cs"] = [tuned_params.pop("C")]

            logger.info(f"Using tuned hyperparameters for {display_name}: {tuned_params}")
            # Merge tuned params with config params (tuned params take precedence)
            params = {**params, **tuned_params}

        # Special handling for RagSklearnAdapter
        if class_name == "RagSklearnAdapter":
            logger.info(
                f"Initializing {display_name} (class={class_name}) with parameters: {params}"
            )
            try:
                models[display_name] = load_rag_model(params)
            except Exception as e:
                logger.warning(f"Failed to initialize {display_name}: {e}. Skipping this model.")
                continue
        else:
            # Standard model instantiation
            model_class = MODEL_CLASSES.get(class_name)
            if model_class:
                logger.info(
                    f"Initializing {display_name} (class={class_name}) with parameters: {params}"
                )
                try:
                    models[display_name] = model_class(**params)
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize {display_name}: {e}. Skipping this model."
                    )
                    continue
            else:
                known_classes = ", ".join(MODEL_CLASSES.keys())
                logger.warning(
                    f"Unknown model class '{class_name}' for model '{model_id}'. "
                    f"Known classes: {known_classes}"
                )

    print(f"Loaded {len(models)} models from configuration:")
    for model_name in models:
        print(f"  - {model_name}")

    return models
