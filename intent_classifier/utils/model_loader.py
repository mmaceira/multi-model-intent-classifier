"""Model Loader Utility

This module provides functions to load model configurations from YAML files.
Model instantiation is handled by model_factory.py to separate loading from creation.
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from intent_classifier.utils.model_registry import discover_and_register_models

# Use module-level logger (no basicConfig - that's for entry points only)
logger = logging.getLogger(__name__)

# Import all algorithm classes to trigger their @register_model decorators
# This ensures models are registered when this module is imported
# These imports must stay here (after logger setup) to trigger registration
from intent_classifier.algorithms.embedding_logreg import EmbeddingLogReg  # noqa: F401, E402
from intent_classifier.algorithms.linear_svm import (  # noqa: F401, E402
    LinearSVMBigrams,
    LinearSVMClassifier,
)
from intent_classifier.algorithms.naive_bayes import NaiveBayesClassifier  # noqa: F401, E402
from intent_classifier.algorithms.transformer_logreg import TransformerLogReg  # noqa: F401, E402

# Auto-discover and register any additional models from algorithms package
# (This is a fallback for models that might not be explicitly imported above)
discover_and_register_models()


# Import centralized config processing functions
from intent_classifier.utils.config_loader import process_config_vars  # noqa: E402


def load_tuned_hyperparameters(
    hyperparams_path: str | None = None,
    config_name: str | None = None,
) -> dict[str, dict[str, Any]]:
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
    from intent_classifier.utils.paths import get_repo_root

    repo_root = get_repo_root()

    # Determine config name if not provided
    if config_name is None:
        import os

        config_file = os.environ.get("CONFIG_FILE")
        if config_file is None:
            # Try to find first available dataset config as default
            dataset_dir = repo_root / "config" / "dataset"
            if dataset_dir.exists():
                dataset_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
                if dataset_dirs:
                    first_dataset = sorted(dataset_dirs)[0].name
                    # Try tiny.yaml first, fallback to default.yaml
                    if (dataset_dir / first_dataset / "tiny.yaml").exists():
                        config_file = f"dataset/{first_dataset}/tiny.yaml"
                    elif (dataset_dir / first_dataset / "default.yaml").exists():
                        config_file = f"dataset/{first_dataset}/default.yaml"
        if config_file is None:
            config_file = "dataset/clinc150/tiny.yaml"  # Final fallback for backward compatibility
        # Extract config name from path like "dataset/clinc150/tiny.yaml" -> "tiny"
        config_name = Path(config_file).parts[-1].replace(".yaml", "")

    # Default to config/algorithm/hyperparameters/{config_name}/ directory
    if hyperparams_path is None:
        hyperparams_dir = repo_root / "config" / "algorithm" / "hyperparameters" / config_name
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
            with open(file_path, encoding="utf-8") as f:
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
    models_config_path: str = "config/algorithm/models_config.yaml",
    main_config_path: str | None = None,
    hyperparams_path: str | None = None,
    config_name: str | None = None,
) -> dict[str, Any]:
    """Load and instantiate models based on configuration.

    Args:
        models_config_path: Path to the models configuration file
        main_config_path: Path to the main configuration file for variable
                         substitution. If None, tries to discover from CONFIG_FILE
                         env var or first available dataset.
        hyperparams_path: Path to tuned hyperparameters file. If None, hyperparameter tuning
                         results are not loaded. If file doesn't exist, defaults are used.

    Returns:
        Dict[str, Any]: Dictionary mapping display names to model instances
    """
    from intent_classifier.utils.paths import get_repo_root

    # Resolve paths relative to repo root
    repo_root = get_repo_root()
    from intent_classifier.utils.paths import get_config_path

    models_config_path_obj = repo_root / models_config_path

    # Handle main_config_path discovery if not provided
    main_config_path_obj: Path
    if main_config_path is None:
        # Use centralized config discovery
        from intent_classifier.utils.config_loader import discover_config_file

        config_file = discover_config_file()
        main_config_path_obj = get_config_path(config_file)
    else:
        # Use get_config_path to handle paths that already start with "config/"
        main_config_path_obj = get_config_path(main_config_path)

    # Load configurations
    with open(models_config_path_obj, encoding="utf-8") as f:
        models_config = yaml.safe_load(f)

    # Convert main_config_path (absolute Path from get_config_path) to relative string
    # load_config expects paths starting with "config/" relative to repo root
    from intent_classifier.utils.config_loader import (
        load_config,
        parse_config_path,
        path_to_config_file_str,
    )

    # Use centralized helper to convert Path to config file string format
    config_file_str = path_to_config_file_str(main_config_path_obj)

    # Load main config using centralized loader to get merged LLM config
    # We disable variable substitution here and apply it later with process_config_vars
    main_config = load_config(config_file=config_file_str, apply_variable_substitution=False)

    # Determine config name from config_file_str if not provided
    if config_name is None:
        _, config_name = parse_config_path(config_file_str)

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
    for _model_id, model_config in models_config.get("models", {}).items():
        # Skip disabled models
        if not model_config.get("enabled", True):
            continue

        # Process config vars in params (before passing to factory)
        model_config["params"] = process_config_vars(model_config.get("params", {}), main_config)

    # Build mapping from model_id to hyperparameter key
    model_id_to_hyperparam_key = {}
    for model_id in models_config.get("models", {}).keys():
        # Handle special mappings
        if model_id == "naive_bayes":
            model_id_to_hyperparam_key[model_id] = "naive_bayes"
        elif model_id == "linear_svm":
            model_id_to_hyperparam_key[model_id] = "linear_svm"
        elif model_id == "linear_svm_bigrams":
            model_id_to_hyperparam_key[model_id] = "linear_svm_bigrams"
        elif model_id == "transformer_logreg":
            model_id_to_hyperparam_key[model_id] = "transformer_logreg"
        elif model_id == "embedding_logreg":
            model_id_to_hyperparam_key[model_id] = "embedding_logreg"
        elif model_id == "rag_kmajority":
            model_id_to_hyperparam_key[model_id] = "rag_kmajority"
        elif model_id == "rag_centroid":
            model_id_to_hyperparam_key[model_id] = "rag_centroid"
        elif model_id in (
            "rag_llm",
            "rag_llm_local",
            "rag_llm_local_short",
            "rag_llm_local_n8n",
            "rag_llm_openai",
        ):
            # Support both old single rag_llm and new separate local/openai variants
            model_id_to_hyperparam_key[model_id] = "rag_llm"
        else:
            # Default: use model_id as key
            model_id_to_hyperparam_key[model_id] = model_id

    # Use factory to create model instances
    from intent_classifier.utils.model_factory import create_models_from_config

    models = create_models_from_config(
        models_config=models_config,
        tuned_hyperparams=tuned_hyperparams,
        model_id_to_hyperparam_key=model_id_to_hyperparam_key,
    )

    return models


def load_persisted_model(
    model_identifier: str,
    models_dir: Path | None = None,
    embeddings_dir: Path | None = None,
) -> Any:
    """Load a persisted model from disk (for API/inference use).

    This function loads models that were saved during training. It handles:
    - Regular classifiers (with optional vectorizer wrapping)
    - RAG models (with index and metadata loading)

    Args:
        model_identifier: Model identifier (can be model ID, directory name, or file path)
        models_dir: Directory containing saved models. If None, uses default location.
        embeddings_dir: Directory containing embeddings. If None, uses default location.

    Returns:
        Loaded model ready for inference (accepts raw text)

    Raises:
        FileNotFoundError: If model file cannot be found
        ValueError: If model type is unknown
    """

    import joblib

    # Use paths utility for consistent path resolution
    from intent_classifier.utils.paths import get_embeddings_dir, get_models_dir

    if models_dir is None:
        models_dir = get_models_dir()
    else:
        models_dir = Path(models_dir).resolve()

    if embeddings_dir is None:
        embeddings_dir = get_embeddings_dir()
    else:
        embeddings_dir = Path(embeddings_dir).resolve()

    # Model info mapping (same as API loader, kept in sync with models_config.yaml)
    MODELS_INFO = {
        # Text classification models
        "naive_bayes": {"name": "Naive Bayes", "dir": "Naive Bayes", "type": "classifier"},
        "linear_svm": {"name": "Linear SVM", "dir": "Linear SVM", "type": "classifier"},
        "linear_svm_bigrams": {
            "name": "TF-IDF bigrams + SVM",
            "dir": "TF-IDF bigrams + SVM",
            "type": "classifier",
        },
        "transformer_logreg": {
            "name": "MiniLM + LogReg",
            "dir": "MiniLM + LogReg",
            "type": "classifier",
        },
        "embedding_logreg": {
            "name": "Embedding + LogReg",
            "dir": "Embedding + LogReg",
            "type": "classifier",
        },
        # RAG-based models
        "rag_kmajority": {
            "name": "RAG-kMajority",
            "dir": "RAG-kMajority",
            "type": "rag",
        },
        "rag_centroid": {
            "name": "RAG-CentroidNN",
            "dir": "RAG-CentroidNN",
            "type": "rag",
        },
        "rag_llm_local": {
            "name": "RAG-LLM (local-embeddings, default prompt)",
            "dir": "RAG-LLM (local-embeddings, default prompt)",
            "type": "rag",
        },
        "rag_llm_local_short": {
            "name": "RAG-LLM (local-embeddings, short prompt)",
            "dir": "RAG-LLM (local-embeddings, short prompt)",
            "type": "rag",
        },
        "rag_llm_local_n8n": {
            "name": "RAG-LLM (local-embeddings, n8n prompt)",
            "dir": "RAG-LLM (local-embeddings, n8n prompt)",
            "type": "rag",
        },
        "rag_llm_openai": {
            "name": "RAG-LLM (OpenAI-embeddings)",
            "dir": "RAG-LLM (OpenAI-embeddings)",
            "type": "rag",
        },
    }

    def _locate(model_id: str) -> Path:
        """Find model file path."""
        # Try direct .joblib or .pkl files
        for ext in [".joblib", ".pkl"]:
            direct = models_dir / f"{model_id}{ext}"
            if direct.exists():
                return direct

        # Try model IDs from MODELS_INFO
        if model_id in MODELS_INFO:
            model_dir = models_dir / MODELS_INFO[model_id]["dir"]
            for filename in ["model.joblib", "model.pkl"]:
                cand = model_dir / filename
                if cand.exists():
                    return cand

        # Try directory names
        for d in models_dir.iterdir():
            if d.is_dir() and d.name.lower() == model_id.lower():
                for filename in ["model.joblib", "model.pkl"]:
                    cand = d / filename
                    if cand.exists():
                        return cand

        raise FileNotFoundError(f"Model {model_id!r} not found in {models_dir}")

    model_path = _locate(model_identifier)

    # Get model info
    model_info = None
    for mid, info in MODELS_INFO.items():
        if mid == model_identifier or info["dir"] == model_path.parent.name:
            model_info = info
            break

    if model_info is None:
        # Try to infer from path
        model_info = {"type": "classifier"}  # Default assumption

    # Load model based on type
    if model_info.get("type") == "rag":
        # RAG models need special handling
        import json

        try:
            import faiss
        except ImportError:
            faiss = None

        # Determine embedding directory
        idx_dir = embeddings_dir
        if "openai" in model_identifier:
            index_path = idx_dir / "openai" / "index.faiss"
            meta_path = idx_dir / "openai" / "meta.jsonl"
        else:
            index_path = idx_dir / "sbert" / "index.faiss"
            meta_path = idx_dir / "sbert" / "meta.jsonl"

        index = None
        if faiss and index_path.exists():
            index = faiss.read_index(str(index_path))

        # Load passages from meta.jsonl
        passages = []
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        meta = json.loads(line)
                        if "text" in meta:
                            passages.append(meta["text"])
                    except json.JSONDecodeError:
                        continue

        # Load classifier
        classifier_obj = None
        if model_path.exists():
            try:
                if model_path.suffix == ".pkl":
                    import cloudpickle

                    with open(str(model_path), "rb") as f:  # type: ignore[assignment]
                        classifier_obj = cloudpickle.load(f)
                else:
                    classifier_obj = joblib.load(model_path)

                # Initialize RAG component if needed
                if hasattr(classifier_obj, "rag") and classifier_obj.rag is None:
                    from intent_classifier.rag import load_centroid, load_kmajority, load_llm
                    from intent_classifier.rag.vector_store import VectorStore

                    if "kmajority" in model_identifier:
                        rag_model = load_kmajority(top_k=5, use_openai="openai" in model_identifier)
                    elif "centroid" in model_identifier:
                        rag_model = load_centroid(use_openai="openai" in model_identifier)
                    else:  # LLM-based RAG
                        if "openai" in model_identifier:
                            from intent_classifier.utils.embeddings import EmbeddingGenerator

                            api_key = os.getenv("OPENAI_API_KEY")
                            if not api_key:
                                raise ValueError(
                                    "OPENAI_API_KEY environment variable must be set "
                                    "for OpenAI embeddings"
                                )
                            embedder = EmbeddingGenerator(
                                api_key=api_key, model="text-embedding-3-small", batch_size=50
                            )
                            rag_model = load_llm(
                                top_k=5,
                                model="ollama/llama3.1:8b",
                                embedder=embedder,
                                use_openai=True,
                            )
                        else:

                            def embedder(texts: list[str]) -> np.ndarray:  # type: ignore[misc]
                                return VectorStore.embed(
                                    "sentence-transformers/all-MiniLM-L6-v2", texts
                                )

                            rag_model = load_llm(
                                top_k=5,
                                model="ollama/llama3.1:8b",
                                embedder=embedder,
                                use_openai=False,
                            )

                    classifier_obj.rag = rag_model
                    classifier_obj.rag_clf = rag_model
            except Exception as e:
                logger.warning(f"Could not load RAG classifier: {e}")

        return {"index": index, "passages": passages, "model": classifier_obj}

    # Regular classifier
    # Try to load with joblib first, then cloudpickle
    try:
        if model_path.suffix == ".pkl":
            import cloudpickle

            with open(str(model_path), "rb") as f:  # type: ignore[assignment]
                artefact = cloudpickle.load(f)
        else:
            artefact = joblib.load(model_path)
    except Exception:
        # Fallback to cloudpickle
        import cloudpickle

        with open(str(model_path), "rb") as f:  # type: ignore[assignment]
            artefact = cloudpickle.load(f)

    # Wrap with vectorizer if needed
    from sklearn.pipeline import make_pipeline

    needs_wrap = not hasattr(artefact, "predict") or getattr(artefact, "_expects_vectors", False)
    if needs_wrap:
        # Try to find vectorizer
        for vec_name in ["vectorizer.joblib", "vectorizer.pkl"]:
            vec_path = model_path.with_name(vec_name)
            if vec_path.exists():
                if vec_path.suffix == ".pkl":
                    import cloudpickle

                    with open(str(vec_path), "rb") as f:  # type: ignore[assignment]
                        vectorizer = cloudpickle.load(f)
                else:
                    vectorizer = joblib.load(vec_path)
                artefact = make_pipeline(vectorizer, artefact)
                break
        else:
            # If no vectorizer found but model needs it, check if it's a Pipeline
            if not hasattr(artefact, "predict"):
                raise AttributeError(
                    f"Loaded object for '{model_identifier}' cannot classify raw text and "
                    f"no vectorizer found next to it."
                )

    return artefact
