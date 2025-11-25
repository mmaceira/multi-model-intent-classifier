"""
Hyperparameter Tuning Script (Ray Tune)

This script performs distributed hyperparameter optimization for all text classification
models using Ray Tune. It supports tuning of:
- Naive Bayes (alpha)
- Linear SVM (C)
- Linear SVM Bigrams (C)
- Transformer LogReg (C) - MiniLM + Logistic Regression
- Embedding LogReg (C) - Flexible embeddings (SBERT or OpenAI) + Logistic Regression
- RAG KMajority (top_k)
- RAG Centroid (no hyperparameters, just evaluation)
- RAG LLM (top_k)

Key Features:
- Distributed hyperparameter search using Ray Tune
- Uses validation set for proper model selection (ML best practice)
- Reads global YAML configuration for dataset and experiment settings
- Automatic result saving in format compatible with model loader
- Supports all models in the pipeline

Usage:
    # Tune all models
    python scripts/tune_hyperparams.py --config config/config.yaml --all

    # Tune specific model
    python scripts/tune_hyperparams.py --config config/config.yaml --algo nb --num-samples 30

    # Tune with custom search space
    python scripts/tune_hyperparams.py --config config/config.yaml --algo svm --num-samples 50

This script follows ML best practices:
- Uses validation set for hyperparameter selection (not test set)
- Saves results that can be loaded by the main training pipeline
- Prevents data leakage by keeping test set completely separate
"""

import argparse
import os
import sys
from pathlib import Path

import ray
import yaml
from ray import tune
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Add repo root to path for imports
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

# Import from project
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from src.algorithms.embedding_logreg import EmbeddingLogReg  # noqa: E402
from src.algorithms.transformer_logreg import TransformerLogReg  # noqa: E402
from src.datasets.dataset import get_dataset  # noqa: E402
from src.rag import _OPENAI_DIR, _SBERT_DIR, load_centroid, load_kmajority, load_llm  # noqa: E402
from src.rag.adapter_sklearn import RagSklearnAdapter  # noqa: E402
from src.rag.vector_store import VectorStore  # noqa: E402


def train_nb(config, data=None):
    """Train Naive Bayes with hyperparameter tuning."""
    # Use pre-loaded data if provided, otherwise load it
    if data is None:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = get_dataset(
            dataset_name=config["dataset"].get("name", "clinc150"),
            use_oos=config["dataset"].get("use_oos", False),
            max_classes=config["dataset"].get("max_classes", None),
            max_train_samples=config["dataset"].get("max_train_samples", None),
            max_test_samples=config["dataset"].get("max_test_samples", None),
            seed=config.get("general", {}).get("seed", 42),
        )
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000, stop_words="english")),
            ("nb", MultinomialNB(alpha=config["alpha"])),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")
    tune.report({"f1": f1})


def train_svm(config, data=None):
    """Train Linear SVM with hyperparameter tuning."""
    # Use pre-loaded data if provided, otherwise load it
    if data is None:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = get_dataset(
            dataset_name=config["dataset"].get("name", "clinc150"),
            use_oos=config["dataset"].get("use_oos", False),
            max_classes=config["dataset"].get("max_classes", None),
            max_train_samples=config["dataset"].get("max_train_samples", None),
            max_test_samples=config["dataset"].get("max_test_samples", None),
            seed=config.get("general", {}).get("seed", 42),
        )
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, stop_words="english")),
            ("svm", LinearSVC(C=config["C"])),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")
    tune.report({"f1": f1})


def train_svm_bigrams(config, data=None):
    """Train Linear SVM with bigrams and hyperparameter tuning."""
    # Use pre-loaded data if provided, otherwise load it
    if data is None:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = get_dataset(
            dataset_name=config["dataset"].get("name", "clinc150"),
            use_oos=config["dataset"].get("use_oos", False),
            max_classes=config["dataset"].get("max_classes", None),
            max_train_samples=config["dataset"].get("max_train_samples", None),
            max_test_samples=config["dataset"].get("max_test_samples", None),
            seed=config.get("general", {}).get("seed", 42),
        )
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
                ),
            ),
            ("svm", LinearSVC(C=config["C"])),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")
    tune.report({"f1": f1})


def ensure_embeddings_built(X_train, y_train, use_openai=False, force_rebuild=False):
    """Ensure embeddings are built before tuning RAG models.

    Args:
        X_train: Training texts (for hyperparameter tuning, should be ONLY train, not train+val)
        y_train: Training labels
        use_openai: Whether to check/build OpenAI embeddings (default: False, uses SBERT)
        force_rebuild: If True, rebuild embeddings even if they exist (default: False)

    Returns:
        bool: True if embeddings exist or were built successfully
    """
    if use_openai:
        index_path = _OPENAI_DIR / "index.faiss"
        meta_path = _OPENAI_DIR / "meta.jsonl"
    else:
        index_path = _SBERT_DIR / "index.faiss"
        meta_path = _SBERT_DIR / "meta.jsonl"

    # Check if embeddings already exist
    if index_path.exists() and meta_path.exists() and not force_rebuild:
        print(f"✅ Embeddings already exist at {index_path}")
        return True

    # If force_rebuild, delete existing embeddings
    if force_rebuild and (index_path.exists() or meta_path.exists()):
        print("🔄 Force rebuild requested - removing existing embeddings...")
        if index_path.exists():
            index_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    # Build embeddings if they don't exist
    print(f"\n{'='*60}")
    print("Building Embeddings for RAG Models")
    print(f"{'='*60}")
    if force_rebuild:
        print("Force rebuild requested - building embeddings from provided data...")
    else:
        print("Embeddings not found - building them now...")
    print(f"Building embeddings for {len(X_train)} training samples")

    try:
        if use_openai:
            from src.embeddings.openai_embedder import OpenAIEmbedder

            print("Using OpenAI model: text-embedding-3-small")
            embedder = OpenAIEmbedder(model="text-embedding-3-small", batch_size=50)
            vectors = np.array(embedder.encode(X_train), dtype="float32")
        else:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"Using SBERT model: {model_name}")
            sbert = SentenceTransformer(model_name)
            vectors = sbert.encode(
                X_train, batch_size=64, show_progress_bar=True, convert_to_numpy=True
            ).astype("float32")

        print(f"Generated embeddings with shape: {vectors.shape}")
        print("Building metadata...")
        meta = []
        for i, (txt, label, vec) in enumerate(zip(X_train, y_train, vectors, strict=False)):
            meta.append({"id": i, "label": label, "text": txt, "vector": vec.tolist()})

        print("Building FAISS index...")
        (
            _SBERT_DIR.mkdir(parents=True, exist_ok=True)
            if not use_openai
            else _OPENAI_DIR.mkdir(parents=True, exist_ok=True)
        )
        VectorStore.build(vectors, meta, vectors.shape[1], index_path, meta_path)
        print(f"✅ Embeddings built and saved at {index_path}")
        return True
    except Exception as e:
        print(f"❌ Error building embeddings: {e}")
        return False


def train_rag_kmajority(config, data=None):
    """Train RAG KMajority with hyperparameter tuning (top_k).

    Follows same approach as other models: build index using ONLY training set,
    then evaluate on validation set. This prevents data leakage.
    """
    # Use pre-loaded data if provided, otherwise load it
    if data is None:
        X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
            dataset_name=config["dataset"].get("name", "clinc150"),
            use_oos=config["dataset"].get("use_oos", False),
            max_classes=config["dataset"].get("max_classes", None),
            max_train_samples=config["dataset"].get("max_train_samples", None),
            max_test_samples=config["dataset"].get("max_test_samples", None),
            seed=config.get("general", {}).get("seed", 42),
        )
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data

    # Embeddings should already be built in main() before tuning starts
    # Just verify they exist (they should, since we built them with train-only data)
    use_openai = False  # Using SBERT embeddings for hyperparameter tuning
    if use_openai:
        index_path = _OPENAI_DIR / "index.faiss"
        meta_path = _OPENAI_DIR / "meta.jsonl"
    else:
        index_path = _SBERT_DIR / "index.faiss"
        meta_path = _SBERT_DIR / "meta.jsonl"

    if not (index_path.exists() and meta_path.exists()):
        print(f"Warning: Embeddings not found at {index_path}")
        tune.report({"f1": 0.0})
        return

    try:
        rag_model = load_kmajority(top_k=config["top_k"], use_openai=use_openai)
        adapter = RagSklearnAdapter(rag_model)

        # RAG models don't need fit, but we need to ensure index is built
        # Evaluate on validation set (which was not used to build the index)
        preds = adapter.predict(X_val)
        f1 = f1_score(y_val, preds, average="macro")
        tune.report({"f1": f1})
    except Exception as e:
        # If RAG model fails, return low score
        print(f"Warning: RAG model failed: {e}")
        tune.report({"f1": 0.0})


def train_rag_centroid(config, data=None):
    """Train RAG Centroid with hyperparameter tuning (top_k).

    Follows same approach as other models: build index using ONLY training set,
    then evaluate on validation set. This prevents data leakage.
    """
    # Use pre-loaded data if provided, otherwise load it
    if data is None:
        X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
            dataset_name=config["dataset"].get("name", "clinc150"),
            use_oos=config["dataset"].get("use_oos", False),
            max_classes=config["dataset"].get("max_classes", None),
            max_train_samples=config["dataset"].get("max_train_samples", None),
            max_test_samples=config["dataset"].get("max_test_samples", None),
            seed=config.get("general", {}).get("seed", 42),
        )
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data

    # Embeddings should already be built in main() before tuning starts
    # Just verify they exist (they should, since we built them with train-only data)
    use_openai = False  # Using SBERT embeddings for hyperparameter tuning
    if use_openai:
        index_path = _OPENAI_DIR / "index.faiss"
        meta_path = _OPENAI_DIR / "meta.jsonl"
    else:
        index_path = _SBERT_DIR / "index.faiss"
        meta_path = _SBERT_DIR / "meta.jsonl"

    if not (index_path.exists() and meta_path.exists()):
        print(f"Warning: Embeddings not found at {index_path}")
        tune.report({"f1": 0.0})
        return

    try:
        rag_model = load_centroid(use_openai=use_openai)
        # Note: CentroidNN doesn't use top_k in the same way, but we'll tune it if supported
        adapter = RagSklearnAdapter(rag_model)
        # Evaluate on validation set (which was not used to build the index)
        preds = adapter.predict(X_val)
        f1 = f1_score(y_val, preds, average="macro")
        tune.report({"f1": f1})
    except Exception as e:
        print(f"Warning: RAG Centroid failed: {e}")
        tune.report({"f1": 0.0})


def train_transformer_logreg(config, data=None):
    """Train Transformer LogReg with hyperparameter tuning (C)."""
    # Use pre-loaded data if provided, otherwise load it
    if data is None:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = get_dataset(
            dataset_name=config["dataset"].get("name", "clinc150"),
            use_oos=config["dataset"].get("use_oos", False),
            max_classes=config["dataset"].get("max_classes", None),
            max_train_samples=config["dataset"].get("max_train_samples", None),
            max_test_samples=config["dataset"].get("max_test_samples", None),
            seed=config.get("general", {}).get("seed", 42),
        )
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data

    # Get model name from config
    model_name = config.get("model", {}).get(
        "sbert_model_name", "sentence-transformers/all-MiniLM-L6-v2"
    )
    # Remove "sentence-transformers/" prefix if present
    # (TransformerLogReg expects just the model name)
    if model_name.startswith("sentence-transformers/"):
        model_name = model_name.replace("sentence-transformers/", "")

    # Create model with single C value (bypasses internal GridSearchCV)
    # Set n_jobs=1 to avoid nested parallelism with Ray Tune (Ray handles parallelism)
    model = TransformerLogReg(
        model_name=model_name,
        Cs=[config["C"]],  # Single C value for this trial
        cv=5,
        n_jobs=1,  # Avoid nested parallelism - Ray Tune handles parallel trials
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")
    tune.report({"f1": f1})


def train_embedding_logreg(config, data=None):
    """Train Embedding LogReg with hyperparameter tuning (C).

    Supports both OpenAI embeddings (requires OPENAI_API_KEY) and SBERT embeddings
    (local, no API key). The backend is selected via config["use_openai"]
    (default: False for SBERT).
    """
    # Use pre-loaded data if provided, otherwise load it
    if data is None:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = get_dataset(
            dataset_name=config["dataset"].get("name", "clinc150"),
            use_oos=config["dataset"].get("use_oos", False),
            max_classes=config["dataset"].get("max_classes", None),
            max_train_samples=config["dataset"].get("max_train_samples", None),
            max_test_samples=config["dataset"].get("max_test_samples", None),
            seed=config.get("general", {}).get("seed", 42),
        )
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data

    # Get embedding backend (default: False = SBERT, no API key needed)
    use_openai = config.get("use_openai", False)

    # Get model name from config based on backend
    if use_openai:
        # Check if OpenAI API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  Skipping Embedding LogReg (OpenAI): OPENAI_API_KEY not set")
            tune.report({"f1": 0.0})
            return
        model_name = config.get("model", {}).get("openai_model_name", "text-embedding-3-small")
    else:
        # Use SBERT embeddings (local, no API key needed)
        api_key = None
        model_name = config.get("model", {}).get(
            "sbert_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )

    # Create model with single C value (bypasses internal GridSearchCV)
    # Set n_jobs=1 to avoid nested parallelism with Ray Tune (Ray handles parallelism)
    model = EmbeddingLogReg(
        use_openai=use_openai,
        model=model_name,
        api_key=api_key,
        Cs=[config["C"]],  # Single C value for this trial
        cv=5,
        n_jobs=1,  # Avoid nested parallelism - Ray Tune handles parallel trials
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")
    tune.report({"f1": f1})


# Backward compatibility alias
train_openai_logreg = train_embedding_logreg


def train_rag_llm(config, data=None):
    """Train RAG LLM with hyperparameter tuning (top_k).

    Follows same approach as other models: build index using ONLY training set,
    then evaluate on validation set. This prevents data leakage.
    """
    # Use pre-loaded data if provided, otherwise load it
    if data is None:
        X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
            dataset_name=config["dataset"].get("name", "clinc150"),
            use_oos=config["dataset"].get("use_oos", False),
            max_classes=config["dataset"].get("max_classes", None),
            max_train_samples=config["dataset"].get("max_train_samples", None),
            max_test_samples=config["dataset"].get("max_test_samples", None),
            seed=config.get("general", {}).get("seed", 42),
        )
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data

    # Embeddings should already be built in main() before tuning starts
    # Just verify they exist (they should, since we built them with train-only data)
    use_openai = False  # Using SBERT embeddings for hyperparameter tuning
    if use_openai:
        index_path = _OPENAI_DIR / "index.faiss"
        meta_path = _OPENAI_DIR / "meta.jsonl"
    else:
        index_path = _SBERT_DIR / "index.faiss"
        meta_path = _SBERT_DIR / "meta.jsonl"

    if not (index_path.exists() and meta_path.exists()):
        print(f"Warning: Embeddings not found at {index_path}")
        tune.report({"f1": 0.0})
        return

    try:
        # Load RAG LLM with tuned top_k parameter
        rag_model = load_llm(top_k=config["top_k"], use_openai=use_openai)
        adapter = RagSklearnAdapter(rag_model)
        # Evaluate on validation set (which was not used to build the index)
        preds = adapter.predict(X_val)
        f1 = f1_score(y_val, preds, average="macro")
        tune.report({"f1": f1})
    except Exception as e:
        # If RAG LLM fails, return low score
        print(f"Warning: RAG LLM failed: {e}")
        tune.report({"f1": 0.0})


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for text classification models"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--algo",
        choices=[
            "nb",
            "svm",
            "svm_bigrams",
            "transformer_logreg",
            "embedding_logreg",
            "openai_logreg",
            "rag_kmajority",
            "rag_centroid",
            "rag_llm",
            "all",
        ],
        default="all",
        help="Algorithm to tune (default: all)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=30, help="Number of hyperparameter samples (default: 30)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/hyperparams_tune",
        help="Output directory for results (default: output/hyperparams_tune)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to repo root
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / args.config
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")

    cfg = yaml.safe_load(config_path.read_text())

    # Extract config file name (without extension) for output directory naming
    config_name = config_path.stem  # e.g., "config_tiny_dataset" from "config_tiny_dataset.yaml"

    # Set up embeddings directory from config (needed for RAG models)
    repo_root = Path(__file__).resolve().parents[1]
    if "paths" in cfg and "embeddings_dir" in cfg["paths"]:
        # Resolve embeddings directory path (handle variable substitution)
        embeddings_dir = cfg["paths"]["embeddings_dir"]
        # Simple variable substitution for ${general.run_name}
        if "${general.run_name}" in embeddings_dir:
            run_name = cfg.get("general", {}).get("run_name", "default")
            embeddings_dir = embeddings_dir.replace("${general.run_name}", run_name)
        embeddings_dir_path = repo_root / embeddings_dir
        os.environ["EMBEDDINGS_DIR"] = str(embeddings_dir_path)
        # Update RAG module paths
        from src.rag import set_artifacts_dir

        set_artifacts_dir(embeddings_dir_path)

    # Update output directories to include config name
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also update config/hyperparameters path to include config name
    config_hyperparams_dir = repo_root / "config" / "hyperparameters" / config_name
    config_hyperparams_dir.mkdir(parents=True, exist_ok=True)

    print("\n📁 Output directories:")
    print(f"   - Config: {config_hyperparams_dir}")
    print(f"   - Output: {output_dir}")
    print(f"   (Using config: {config_path.name})")

    # Load dataset ONCE before starting Ray Tune to avoid rate limiting
    print("\n" + "=" * 60)
    print("Loading Dataset (once for all trials)")
    print("=" * 60)
    print("This prevents HuggingFace rate limiting when multiple trials run in parallel...")
    X_train, y_train, X_val, y_val, X_test, y_test, _ = get_dataset(
        dataset_name=cfg["dataset"].get("name", "clinc150"),
        use_oos=cfg["dataset"].get("use_oos", False),
        max_classes=cfg["dataset"].get("max_classes", None),
        max_train_samples=cfg["dataset"].get("max_train_samples", None),
        max_test_samples=cfg["dataset"].get("max_test_samples", None),
        seed=cfg.get("general", {}).get("seed", 42),
    )
    print(f"✅ Dataset loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")

    # Package data as tuple for passing to trials
    data = (X_train, y_train, X_val, y_val, X_test, y_test)

    # Initialize Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    results = {}

    # Define algorithms to tune
    algorithms = []
    if args.algo == "all":
        algorithms = [
            "nb",
            "svm",
            "svm_bigrams",
            "transformer_logreg",
            "embedding_logreg",
            "rag_kmajority",
            "rag_centroid",
            "rag_llm",
        ]
    else:
        algorithms = [args.algo]

    # Build embeddings once for RAG models (using ONLY train data to prevent data leakage)
    # This ensures all RAG trials use the same embeddings built from train-only data
    rag_algorithms = ["rag_kmajority", "rag_centroid", "rag_llm"]
    if any(algo in rag_algorithms for algo in algorithms):
        print("\n" + "=" * 60)
        print("Building Embeddings for RAG Models (train-only)")
        print("=" * 60)
        print("Building embeddings from training set only (not train+val)")
        print("This ensures validation set is truly unseen during hyperparameter tuning")
        if not ensure_embeddings_built(X_train, y_train, use_openai=False, force_rebuild=True):
            print("⚠️  Warning: Failed to build embeddings for RAG models")
            print("   RAG model tuning will be skipped")
            # Remove RAG algorithms from the list
            algorithms = [a for a in algorithms if a not in rag_algorithms]

    # Tune each algorithm
    for algo in algorithms:
        print(f"\n{'='*60}")
        print(f"Tuning {algo.upper()}")
        print(f"{'='*60}")

        try:
            if algo == "nb":
                analysis = tune.run(
                    tune.with_parameters(train_nb, data=data),
                    config={**cfg, "alpha": tune.loguniform(1e-3, 1.0)},
                    num_samples=args.num_samples,
                    resources_per_trial={"cpu": 1},
                    metric="f1",
                    mode="max",
                )
                best = analysis.get_best_config("f1", "max")
                results["naive_bayes"] = {"alpha": best["alpha"], "f1": analysis.best_result["f1"]}

            elif algo == "svm":
                analysis = tune.run(
                    tune.with_parameters(train_svm, data=data),
                    config={**cfg, "C": tune.loguniform(1e-3, 10)},
                    num_samples=args.num_samples,
                    resources_per_trial={"cpu": 1},
                    metric="f1",
                    mode="max",
                )
                best = analysis.get_best_config("f1", "max")
                results["linear_svm"] = {"C": best["C"], "f1": analysis.best_result["f1"]}

            elif algo == "svm_bigrams":
                analysis = tune.run(
                    tune.with_parameters(train_svm_bigrams, data=data),
                    config={**cfg, "C": tune.loguniform(1e-3, 10)},
                    num_samples=args.num_samples,
                    resources_per_trial={"cpu": 1},
                    metric="f1",
                    mode="max",
                )
                best = analysis.get_best_config("f1", "max")
                results["linear_svm_bigrams"] = {"C": best["C"], "f1": analysis.best_result["f1"]}

            elif algo == "transformer_logreg":
                # Memory-intensive model: use more CPUs per trial to reduce parallelism
                analysis = tune.run(
                    tune.with_parameters(train_transformer_logreg, data=data),
                    config={**cfg, "C": tune.loguniform(1e-3, 10)},
                    num_samples=args.num_samples,
                    resources_per_trial={
                        "cpu": 4
                    },  # More CPUs = fewer parallel trials = less memory pressure
                    max_concurrent_trials=2,  # Limit concurrent trials to prevent OOM
                    metric="f1",
                    mode="max",
                )
                best = analysis.get_best_config("f1", "max")
                results["transformer_logreg"] = {"C": best["C"], "f1": analysis.best_result["f1"]}

            elif algo in ("embedding_logreg", "openai_logreg"):
                # Support both new name (embedding_logreg) and old name (openai_logreg)
                # for backward compatibility. Default to SBERT embeddings
                # (use_openai=False) unless explicitly set
                use_openai = (
                    algo == "openai_logreg"
                )  # Old name defaults to OpenAI, new name defaults to SBERT

                # For embedding_logreg, default to SBERT (no API key needed)
                # For openai_logreg (backward compat), check API key
                if use_openai and not os.getenv("OPENAI_API_KEY"):
                    print("⚠️  Skipping Embedding LogReg (OpenAI): OPENAI_API_KEY not set")
                    print(
                        "   💡 Tip: Use 'embedding_logreg' with use_openai=False "
                        "for SBERT embeddings (no API key needed)"
                    )
                    continue

                # Memory-intensive model: run sequentially (max_concurrent_trials=1)
                # to prevent OOM
                analysis = tune.run(
                    tune.with_parameters(train_embedding_logreg, data=data),
                    config={**cfg, "C": tune.loguniform(1e-3, 10), "use_openai": use_openai},
                    num_samples=args.num_samples,
                    resources_per_trial={"cpu": 2},  # Reduced CPU allocation
                    max_concurrent_trials=1,  # Run sequentially to prevent OOM
                    metric="f1",
                    mode="max",
                )
                best = analysis.get_best_config("f1", "max")
                # Use consistent key name regardless of which algo name was used
                results["embedding_logreg"] = {
                    "C": best["C"],
                    "f1": analysis.best_result["f1"],
                    "use_openai": use_openai,
                }

            elif algo == "rag_kmajority":
                # Tune top_k for RAG models (smaller search space)
                analysis = tune.run(
                    tune.with_parameters(train_rag_kmajority, data=data),
                    config={**cfg, "top_k": tune.choice([5, 10, 15, 20, 25, 30])},
                    num_samples=min(args.num_samples, 6),  # Only 6 options
                    resources_per_trial={"cpu": 1},
                    metric="f1",
                    mode="max",
                )
                best = analysis.get_best_config("f1", "max")
                results["rag_kmajority"] = {
                    "top_k": int(best["top_k"]),
                    "f1": analysis.best_result["f1"],
                }

            elif algo == "rag_centroid":
                # CentroidNN doesn't have top_k parameter, but we'll still run it for consistency
                analysis = tune.run(
                    tune.with_parameters(train_rag_centroid, data=data),
                    config={**cfg},
                    num_samples=1,  # No hyperparameters to tune
                    resources_per_trial={"cpu": 1},
                    metric="f1",
                    mode="max",
                )
                best = analysis.get_best_config("f1", "max")
                results["rag_centroid"] = {"f1": analysis.best_result["f1"]}

            elif algo == "rag_llm":
                # Tune top_k for RAG LLM models (smaller search space)
                # Memory-intensive model: run sequentially (max_concurrent_trials=1)
                # to prevent OOM
                analysis = tune.run(
                    tune.with_parameters(train_rag_llm, data=data),
                    config={**cfg, "top_k": tune.choice([5, 10, 15, 20, 25, 30])},
                    num_samples=min(args.num_samples, 6),  # Only 6 options
                    resources_per_trial={"cpu": 2},  # Reduced CPU allocation
                    max_concurrent_trials=1,  # Run sequentially to prevent OOM
                    metric="f1",
                    mode="max",
                )
                best = analysis.get_best_config("f1", "max")
                results["rag_llm"] = {
                    "top_k": int(best["top_k"]),
                    "f1": analysis.best_result["f1"],
                }

            print(f"✅ Best {algo} hyperparameters: {best}")
            print(f"   F1 score: {analysis.best_result['f1']:.4f}")

        except Exception as e:
            print(f"❌ Error tuning {algo}: {e}")
            continue

    # Save individual files to config/hyperparameters/{config_name}/
    # (primary location for model loader)
    # Note: config_hyperparams_dir was already created in main() with config name

    # Clean up old unified file if it exists (we now use individual files only)
    old_unified_file = config_hyperparams_dir / "best_hyperparameters.yaml"
    if old_unified_file.exists():
        old_unified_file.unlink()
        print(f"🗑️  Removed old unified file: {old_unified_file.name}")

    saved_files = []
    for model_name, params in results.items():
        # Extract hyperparameters (exclude f1 score)
        hyperparams = {k: v for k, v in params.items() if k != "f1"}
        if hyperparams:
            file_path = config_hyperparams_dir / f"best_{model_name}.yaml"
            file_path.write_text(yaml.dump(hyperparams, default_flow_style=False))
            saved_files.append(file_path)

    print(f"\n✅ Hyperparameters saved to config: {config_hyperparams_dir}")
    print(f"   Saved {len(saved_files)} individual model files")

    # Also save individual files to output/hyperparams_tune/ for reference/backup
    for model_name, params in results.items():
        # Extract hyperparameters (exclude f1 score)
        hyperparams = {k: v for k, v in params.items() if k != "f1"}
        if hyperparams:
            (output_dir / f"best_{model_name}.yaml").write_text(
                yaml.dump(hyperparams, default_flow_style=False)
            )
    print(f"✅ Hyperparameters also saved to output: {output_dir} (for reference)")

    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Complete!")
    print("=" * 60)
    print("\nBest hyperparameters:")
    for model_name, params in results.items():
        hyperparams = {k: v for k, v in params.items() if k != "f1"}
        f1 = params.get("f1", "N/A")
        print(f"  {model_name}: {hyperparams} (F1: {f1:.4f})")
    print("\nResults saved as individual files to:")
    print(f"  - Config: {config_hyperparams_dir} (used by model loader)")
    print("    Files: best_*.yaml (one per model)")
    print(f"  - Output: {output_dir} (for reference)")
    print("\n💡 Next step: Run the training pipeline to use these hyperparameters:")
    print("   python scripts/pipeline/03_model_training.py")


if __name__ == "__main__":
    main()
