"""Model Loader Utility

This module provides functions to load model configurations from YAML files and
instantiate model objects dynamically based on configuration parameters.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from intent_classifier.algorithms.embedding_logreg import EmbeddingLogReg
from intent_classifier.algorithms.linear_svm import LinearSVMBigrams, LinearSVMClassifier
from intent_classifier.algorithms.naive_bayes import NaiveBayesClassifier
from intent_classifier.algorithms.transformer_logreg import TransformerLogReg
from intent_classifier.rag import load_centroid, load_kmajority, load_llm
from intent_classifier.rag.adapter_sklearn import RagSklearnAdapter

# Use module-level logger (no basicConfig - that's for entry points only)
logger = logging.getLogger(__name__)

# Dictionary mapping class names to their actual classes
MODEL_CLASSES = {
    "NaiveBayesClassifier": NaiveBayesClassifier,
    "LinearSVMClassifier": LinearSVMClassifier,
    "LinearSVMBigrams": LinearSVMBigrams,
    "TransformerLogReg": TransformerLogReg,
    "EmbeddingLogReg": EmbeddingLogReg,
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
        use_openai = params.get(
            "use_openai", False
        )  # Kept for compatibility, but new impl uses TF-IDF
        min_labels = params.get("min_labels", 4)  # New parameter for minimum distinct labels

        logger.info(
            f"Loading LLM RAG model with top_k={top_k}, model={model_name}, min_labels={min_labels}"
        )

        return RagSklearnAdapter(
            load_llm(
                top_k=top_k,
                model=model_name,
                use_openai=use_openai,  # Kept for compatibility
                min_labels=min_labels,
            )
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
        elif model_id == "embedding_logreg":
            hyperparam_key = "embedding_logreg"
        elif model_id == "rag_kmajority":
            hyperparam_key = "rag_kmajority"
        elif model_id == "rag_centroid":
            hyperparam_key = "rag_centroid"
        elif model_id in ("rag_llm", "rag_llm_local", "rag_llm_openai"):
            # Support both old single rag_llm and new separate local/openai variants
            hyperparam_key = "rag_llm"

        if hyperparam_key in tuned_hyperparams:
            tuned_params = tuned_hyperparams[hyperparam_key]
            # Remove f1 score if present (it's metadata, not a hyperparameter)
            tuned_params = {k: v for k, v in tuned_params.items() if k != "f1"}

            # Convert C to Cs for LogReg models (they expect Cs as a sequence)
            if class_name in ("TransformerLogReg", "EmbeddingLogReg") and "C" in tuned_params:
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


def load_persisted_model(
    model_identifier: str,
    models_dir: Optional[Path] = None,
    embeddings_dir: Optional[Path] = None,
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

    # Model info mapping (same as API loader)
    MODELS_INFO = {
        "naive_bayes": {"name": "Naive Bayes", "dir": "Naive Bayes", "type": "classifier"},
        "linear_svm": {"name": "Linear SVM", "dir": "Linear SVM", "type": "classifier"},
        "tfidf_svm": {"name": "TF-IDF + SVM", "dir": "TF-IDF bigrams + SVM", "type": "classifier"},
        "minilm_logreg": {
            "name": "MiniLM + LogReg",
            "dir": "MiniLM + LogReg",
            "type": "classifier",
        },
        "rag_centroid": {"name": "RAG CentroidNN", "dir": "RAG-CentroidNN", "type": "rag"},
        "rag_kmajority": {"name": "RAG k-Majority", "dir": "RAG-kMajority", "type": "rag"},
        "rag_llm_local": {
            "name": "RAG LLM (Local)",
            "dir": "RAG-LLM (local-embeddings)",
            "type": "rag",
        },
        "rag_llm_openai": {
            "name": "RAG LLM (OpenAI)",
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
            with open(meta_path, "r") as f:
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

                    with open(model_path, "rb") as f:
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
                            from intent_classifier.embeddings.openai_embedder import OpenAIEmbedder

                            embedder = OpenAIEmbedder(model="text-embedding-3-small", batch_size=50)
                            rag_model = load_llm(
                                top_k=5,
                                model="ollama/llama3.1:8b",
                                embedder=embedder,
                                use_openai=True,
                            )
                        else:

                            def embedder(texts):
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

            with open(model_path, "rb") as f:
                artefact = cloudpickle.load(f)
        else:
            artefact = joblib.load(model_path)
    except Exception:
        # Fallback to cloudpickle
        import cloudpickle

        with open(model_path, "rb") as f:
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

                    with open(vec_path, "rb") as f:
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
