"""Model-specific hyperparameter tuning strategies.

This module contains the training functions for each model type used during
hyperparameter tuning with Ray Tune. Each function follows the same pattern:
- Accepts a config dict and optional pre-loaded data
- Trains the model with hyperparameters from config
- Evaluates on validation set
- Reports F1 score to Ray Tune

Functions:
- train_nb: Naive Bayes tuning
- train_svm: Linear SVM tuning
- train_svm_bigrams: Linear SVM with bigrams tuning
- train_transformer_logreg: Transformer + Logistic Regression tuning
- train_embedding_logreg: Embedding + Logistic Regression tuning
- train_rag_kmajority: RAG KMajority tuning
- train_rag_centroid: RAG Centroid tuning
- train_rag_llm: RAG LLM tuning
- ensure_embeddings_built: Helper to build embeddings for RAG models
"""

import os
from typing import Any

import numpy as np
from ray import tune
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from intent_classifier.algorithms.embedding_logreg import EmbeddingLogReg
from intent_classifier.algorithms.transformer_logreg import TransformerLogReg
from intent_classifier.datasets.dataset import get_dataset
from intent_classifier.rag import get_index_paths, load_centroid, load_kmajority, load_llm
from intent_classifier.rag.adapter_sklearn import RagSklearnAdapter
from intent_classifier.rag.vector_store import VectorStore


def train_nb(config: dict[str, Any], data: tuple | None = None) -> None:
    """Train Naive Bayes with hyperparameter tuning.

    Args:
        config: Configuration dict containing dataset settings and hyperparameter 'alpha'
        data: Optional pre-loaded data tuple (X_train, y_train, X_val, y_val, X_test, y_test)
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


def train_svm(config: dict[str, Any], data: tuple | None = None) -> None:
    """Train Linear SVM with hyperparameter tuning.

    Args:
        config: Configuration dict containing dataset settings and hyperparameter 'C'
        data: Optional pre-loaded data tuple (X_train, y_train, X_val, y_val, X_test, y_test)
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


def train_svm_bigrams(config: dict[str, Any], data: tuple | None = None) -> None:
    """Train Linear SVM with bigrams and hyperparameter tuning.

    Args:
        config: Configuration dict containing dataset settings and hyperparameter 'C'
        data: Optional pre-loaded data tuple (X_train, y_train, X_val, y_val, X_test, y_test)
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


def ensure_embeddings_built(
    X_train, y_train, use_openai: bool = False, force_rebuild: bool = False
) -> bool:
    """Ensure embeddings are built before tuning RAG models.

    Args:
        X_train: Training texts (for hyperparameter tuning, should be ONLY train, not train+val)
        y_train: Training labels
        use_openai: Whether to check/build OpenAI embeddings (default: False, uses SBERT)
        force_rebuild: If True, rebuild embeddings even if they exist (default: False)

    Returns:
        True if embeddings exist or were built successfully
    """
    # Resolve index/meta paths using current RAG configuration.
    # This respects EMBEDDINGS_DIR / RAG_EMBEDDINGS_DIR and any recent calls
    # to set_artifacts_dir, so tuning uses the same layout as the main pipeline
    # (e.g. output/runs/multilabel/nlu_plus/tiny/features/embeddings/sbert/...).
    index_path, meta_path = get_index_paths(use_openai=use_openai)

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
    print(f"\n{'=' * 60}")
    print("Building Embeddings for RAG Models")
    print(f"{'=' * 60}")
    if force_rebuild:
        print("Force rebuild requested - building embeddings from provided data...")
    else:
        print("Embeddings not found - building them now...")
    print(f"Building embeddings for {len(X_train)} training samples")

    try:
        if use_openai:
            from intent_classifier.utils.embeddings import EmbeddingGenerator

            print("Using OpenAI model: text-embedding-3-small")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable must be set for OpenAI embeddings"
                )
            embedder = EmbeddingGenerator(
                api_key=api_key, model="text-embedding-3-small", batch_size=50
            )
            vectors = np.array(embedder.generate_embeddings(X_train), dtype="float32")
        else:
            from sentence_transformers import SentenceTransformer

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
        # Ensure parent directory exists before writing artifacts
        index_path.parent.mkdir(parents=True, exist_ok=True)
        VectorStore.build(vectors, meta, vectors.shape[1], index_path, meta_path)
        print(f"✅ Embeddings built and saved at {index_path}")
        return True
    except Exception as e:
        print(f"❌ Error building embeddings: {e}")
        return False


def train_rag_kmajority(config: dict[str, Any], data: tuple | None = None) -> None:
    """Train RAG KMajority with hyperparameter tuning (top_k).

    Follows same approach as other models: build index using ONLY training set,
    then evaluate on validation set. This prevents data leakage.

    Args:
        config: Configuration dict containing dataset settings and hyperparameter 'top_k'
        data: Optional pre-loaded data tuple (X_train, y_train, X_val, y_val, X_test, y_test)
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
    # Resolve embeddings paths using current RAG configuration
    index_path, meta_path = get_index_paths(use_openai=use_openai)

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


def train_rag_centroid(config: dict[str, Any], data: tuple | None = None) -> None:
    """Train RAG Centroid with hyperparameter tuning (top_k).

    Follows same approach as other models: build index using ONLY training set,
    then evaluate on validation set. This prevents data leakage.

    Args:
        config: Configuration dict containing dataset settings
        data: Optional pre-loaded data tuple (X_train, y_train, X_val, y_val, X_test, y_test)
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
    # Resolve embeddings paths using current RAG configuration
    index_path, meta_path = get_index_paths(use_openai=use_openai)

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


def train_transformer_logreg(config: dict[str, Any], data: tuple | None = None) -> None:
    """Train Transformer LogReg with hyperparameter tuning (C).

    Args:
        config: Configuration dict containing dataset settings and hyperparameter 'C'
        data: Optional pre-loaded data tuple (X_train, y_train, X_val, y_val, X_test, y_test)
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


def train_embedding_logreg(config: dict[str, Any], data: tuple | None = None) -> None:
    """Train Embedding LogReg with hyperparameter tuning (C).

    Supports both OpenAI embeddings (requires OPENAI_API_KEY) and SBERT embeddings
    (local, no API key). The backend is selected via config["use_openai"]
    (default: False for SBERT).

    Args:
        config: Configuration dict containing dataset settings, hyperparameter 'C', and 'use_openai'
        data: Optional pre-loaded data tuple (X_train, y_train, X_val, y_val, X_test, y_test)
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


def train_rag_llm(config: dict[str, Any], data: tuple | None = None) -> None:
    """Train RAG LLM with hyperparameter tuning (top_k).

    Follows same approach as other models: build index using ONLY training set,
    then evaluate on validation set. This prevents data leakage.

    Args:
        config: Configuration dict containing dataset settings and hyperparameter 'top_k'
        data: Optional pre-loaded data tuple (X_train, y_train, X_val, y_val, X_test, y_test)
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
    # Resolve embeddings paths using current RAG configuration
    index_path, meta_path = get_index_paths(use_openai=use_openai)

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
