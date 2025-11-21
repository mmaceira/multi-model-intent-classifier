"""Model Loader Utility

This module provides functions to load model configurations from YAML files and
instantiate model objects dynamically based on configuration parameters.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from src.algorithms.linear_svm import LinearSVMBigrams, LinearSVMClassifier
from src.algorithms.naive_bayes import NaiveBayesClassifier
from src.algorithms.openai_logreg import OpenAIEmbedLogReg
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
    "OpenAIEmbedLogReg": OpenAIEmbedLogReg,
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


def load_models_from_config(
    models_config_path: str = "config/models_config.yaml",
    main_config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """Load and instantiate models based on configuration.

    Args:
        models_config_path: Path to the models configuration file
        main_config_path: Path to the main configuration file for variable substitution

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
