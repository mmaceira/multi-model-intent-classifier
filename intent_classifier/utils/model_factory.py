"""Model Factory

This module provides factory functions for creating model instances from configuration.
It handles the instantiation logic, separating it from configuration loading.

The factory uses the model registry to create instances dynamically based on class names.
"""

import logging
from typing import Any

from intent_classifier.rag import load_centroid, load_kmajority, load_llm
from intent_classifier.rag.adapter_sklearn import RagSklearnAdapter
from intent_classifier.utils.model_registry import (
    get_all_registered_models,
    get_model_class,
)

# Use module-level logger (no basicConfig - that's for entry points only)
logger = logging.getLogger(__name__)


def create_rag_model(params: dict[str, Any]) -> RagSklearnAdapter:
    """Create a RAG model instance based on configuration parameters.

    Args:
        params: Dictionary containing RAG model parameters:
            - method: One of "kmajority", "centroid", or "llm"
            - top_k: Number of nearest neighbors (default: 25)
            - use_openai: Whether to use OpenAI embeddings (default: False)
            - model: Model name for LLM method (default: "ollama/llama3.1:8b")
            - min_labels: Minimum distinct labels for LLM method (default: 4)

    Returns:
        RagSklearnAdapter instance wrapping the RAG model

    Raises:
        ValueError: If method is unknown
    """
    method = params.get("method")
    # Ensure top_k is an integer
    top_k = int(params.get("top_k", 25))

    logger.info(f"Creating RAG model with method={method}, top_k={top_k}, params={params}")

    if method == "kmajority":
        use_openai = params.get("use_openai", False)
        logger.info(f"Creating KMajority RAG model with top_k={top_k}, use_openai={use_openai}")
        return RagSklearnAdapter(load_kmajority(top_k=top_k, use_openai=use_openai))

    elif method == "centroid":
        logger.info(f"Creating Centroid RAG model with top_k={top_k}")
        return RagSklearnAdapter(load_centroid(top_k=top_k))

    elif method == "llm":
        # Get model name with fallback to LLM config default
        model_name = params.get("model")
        # Check if model is missing, None, or still contains unsubstituted variable
        if not model_name or model_name.startswith("${"):
            # Try to get from LLM config
            from intent_classifier.utils.config_loader import _load_llm_config

            llm_config = _load_llm_config()
            if llm_config and "ollama" in llm_config and "default_model" in llm_config["ollama"]:
                model_name = llm_config["ollama"]["default_model"]
                logger.info(f"Using LLM model from llm_config.yaml: {model_name}")
            else:
                # Final fallback
                model_name = "ollama/llama3.1:8b"
                logger.warning(
                    f"LLM model not found in params or config, using fallback: {model_name}"
                )
        use_openai = params.get(
            "use_openai", False
        )  # Kept for compatibility, but new impl uses TF-IDF
        min_labels = params.get("min_labels", 4)  # New parameter for minimum distinct labels
        prompt_style = params.get("prompt_style", "default")  # Prompt style: "default" or "short"

        logger.info(
            f"Creating LLM RAG model with top_k={top_k}, model={model_name}, "
            f"min_labels={min_labels}, prompt_style={prompt_style}"
        )

        return RagSklearnAdapter(
            load_llm(
                top_k=top_k,
                model=model_name,
                use_openai=use_openai,  # Kept for compatibility
                min_labels=min_labels,
                prompt_style=prompt_style,
            )
        )

    raise ValueError(f"Unknown RAG method: {method}")


def create_model_instance(class_name: str, params: dict[str, Any], display_name: str) -> Any:
    """Create a model instance from class name and parameters.

    Args:
        class_name: Name of the model class (must be registered in model registry)
        params: Dictionary of parameters to pass to model constructor
        display_name: Display name for logging purposes

    Returns:
        Model instance

    Raises:
        ValueError: If class_name is not registered
        Exception: If model instantiation fails
    """
    # Special handling for RagSklearnAdapter
    if class_name == "RagSklearnAdapter":
        logger.info(f"Creating {display_name} (class={class_name}) with parameters: {params}")
        return create_rag_model(params)

    # Standard model instantiation using registry
    model_class = get_model_class(class_name)
    if model_class:
        logger.info(f"Creating {display_name} (class={class_name}) with parameters: {params}")
        return model_class(**params)
    else:
        known_classes = ", ".join(sorted(get_all_registered_models().keys()))
        raise ValueError(
            f"Unknown model class '{class_name}' for model '{display_name}'. "
            f"Known classes: {known_classes}"
        )


def create_models_from_config(
    models_config: dict[str, Any],
    tuned_hyperparams: dict[str, dict[str, Any]],
    model_id_to_hyperparam_key: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create model instances from loaded configuration.

    This function handles the instantiation logic after configuration has been loaded.
    It merges default parameters with tuned hyperparameters and creates model instances.

    Args:
        models_config: Dictionary containing model configurations (from models_config.yaml)
        tuned_hyperparams: Dictionary mapping hyperparameter keys to tuned values
        model_id_to_hyperparam_key: Optional mapping from model_id to hyperparameter key.
                                   If None, uses model_id as key.

    Returns:
        Dictionary mapping display names to model instances

    Raises:
        ValueError: If configuration structure is invalid
    """
    # Validate configuration structure
    if not isinstance(models_config, dict):
        raise ValueError(f"models_config must be a dictionary, got {type(models_config)}")

    if "models" not in models_config:
        raise ValueError("models_config is missing a top-level 'models' key")

    if not isinstance(models_config["models"], dict):
        raise ValueError("models_config 'models' key must contain a dictionary")

    # Default mapping: model_id -> hyperparam_key (identity mapping)
    if model_id_to_hyperparam_key is None:
        model_id_to_hyperparam_key = {}

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
        params = model_config.get("params", {})

        # Get hyperparameter key (use mapping if provided, otherwise use model_id)
        hyperparam_key = model_id_to_hyperparam_key.get(model_id, model_id)

        # Override with tuned hyperparameters if available
        if hyperparam_key in tuned_hyperparams:
            tuned_params = tuned_hyperparams[hyperparam_key].copy()
            # Remove f1 score if present (it's metadata, not a hyperparameter)
            tuned_params = {k: v for k, v in tuned_params.items() if k != "f1"}

            # Convert C to Cs for LogReg models (they expect Cs as a sequence)
            if class_name in ("TransformerLogReg", "EmbeddingLogReg") and "C" in tuned_params:
                tuned_params["Cs"] = [tuned_params.pop("C")]

            logger.info(f"Using tuned hyperparameters for {display_name}: {tuned_params}")
            # Merge tuned params with config params (tuned params take precedence)
            params = {**params, **tuned_params}

        # Create model instance
        try:
            models[display_name] = create_model_instance(class_name, params, display_name)
        except Exception as e:
            logger.warning(f"Failed to create {display_name}: {e}. Skipping this model.")
            continue

    print(f"Created {len(models)} models from configuration:")
    for model_name in models:
        print(f"  - {model_name}")

    return models
