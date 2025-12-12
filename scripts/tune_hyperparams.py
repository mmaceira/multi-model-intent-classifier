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
    python scripts/tune_hyperparams.py --config config/dataset/clinc150/tiny.yaml --all

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
from pathlib import Path

import ray
import yaml
from ray import tune

# Import path utilities
from intent_classifier.utils.paths import get_repo_root  # noqa: E402

# Get repo root and add to path for imports
repo_root = get_repo_root()

# Import model-specific tuning strategies
from intent_classifier.datasets.dataset import get_dataset  # noqa: E402
from intent_classifier.hparam.strategies import (  # noqa: E402
    ensure_embeddings_built,
    train_embedding_logreg,
    train_nb,
    train_rag_centroid,
    train_rag_kmajority,
    train_rag_llm,
    train_svm,
    train_svm_bigrams,
    train_transformer_logreg,
)


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

    # Load config using centralized loader
    from intent_classifier.utils.config_loader import get_config_path, load_config

    config_file = args.config
    # Use get_config_path to handle paths that already start with "config/"
    if Path(config_file).is_absolute():
        config_path = Path(config_file)
    else:
        config_path = get_config_path(config_file)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Convert to relative path for config loader (relative to repo root)
    config_file = str(config_path.relative_to(repo_root))

    cfg = load_config(config_file)

    # Extract config file name (without extension) for output directory naming
    # Extract config name from path (e.g., "tiny" from "config/dataset/clinc150/tiny.yaml")
    from intent_classifier.utils.config_loader import parse_config_path

    _, config_name = parse_config_path(config_file)

    # Set up embeddings directory from config (needed for RAG models)
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
        from intent_classifier.rag import set_artifacts_dir

        set_artifacts_dir(embeddings_dir_path)

    # Update output directories to include config name
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also update config/algorithm/hyperparameters path to include config name
    config_hyperparams_dir = repo_root / "config" / "algorithm" / "hyperparameters" / config_name
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

            elif algo == "embedding_logreg":
                # Default to SBERT embeddings (use_openai=False) unless explicitly set
                use_openai = False

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

    # Save individual files to config/algorithm/hyperparameters/{config_name}/
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
