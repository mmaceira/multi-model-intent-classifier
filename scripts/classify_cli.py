#!/usr/bin/env python3
"""
Unified Intent Classifier CLI

This script provides a command-line interface for all intent classifier methods:
- Traditional ML: Naive Bayes, Linear SVM, TF-IDF + SVM, MiniLM + LogReg, Embedding + LogReg
- RAG methods: RAG-kMajority, RAG-CentroidNN, RAG-LLM

It loads models from the configuration and provides a unified interface for predictions.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from intent_classifier.utils.model_loader import (  # noqa: E402
    load_models_from_config,
    load_persisted_model,
)
from intent_classifier.utils.paths import get_models_dir  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified intent classifier CLI for all classifier methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a specific model by name
  %(prog)s --model-name "Naive Bayes" --text "What's the weather?"

  # Use RAG-LLM with specific LLM model
  %(prog)s --model-name "RAG-LLM (local-embeddings)" --text "What's the weather?" \\
      --llm-model "gpt-4o-mini"

  # List all available models
  %(prog)s --list-models

  # Use method-based selection (convenience for RAG methods)
  %(prog)s --method llm --text "What's the weather?"
        """,
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Specific model name from config (e.g., 'Naive Bayes', 'RAG-LLM (local-embeddings)'). "
        "Use --list-models to see all available models.",
    )
    parser.add_argument(
        "--method",
        default=None,
        choices=["kmajority", "centroid", "llm"],
        help="Convenience option for RAG methods: 'kmajority', 'centroid', or 'llm'. "
        "Only used if --model-name is not specified.",
    )
    parser.add_argument(
        "--text",
        required=False,
        help="Single query to classify. Required unless --list-models is used.",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="For RAG methods: prefer OpenAI embeddings variant (if available). "
        "Only used with --method option.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="For RAG-LLM: override LLM model to use (e.g., 'gpt-4o-mini', 'ollama/llama3.1:8b'). "
        "If not provided, uses default from config.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="json",
        help="Output format: 'json' (default) or 'text'",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit.",
    )
    return parser


def find_model_by_method(models: dict, method: str, use_openai: bool = False) -> tuple:
    """Find a model matching the specified RAG method.

    Args:
        models: Dictionary of model name -> model instance
        method: Method name ('kmajority', 'centroid', 'llm')
        use_openai: Whether to prefer OpenAI variant for LLM

    Returns:
        Tuple of (model_name, model_instance) or (None, None) if not found
    """
    method_lower = method.lower()

    # Map method names to config identifiers
    method_map = {
        "kmajority": ["rag_kmajority", "rag-kmajority", "rag kmajority", "kmajority"],
        "centroid": [
            "rag_centroid",
            "rag-centroidnn",
            "rag centroidnn",
            "rag-centroid",
            "centroid",
        ],
        "llm": ["rag_llm", "rag-llm"],
    }

    search_terms = method_map.get(method_lower, [method_lower])

    # Find matching models
    candidates = []
    for name, model in models.items():
        name_lower = name.lower()
        for term in search_terms:
            if term in name_lower:
                # For LLM, check if it matches the OpenAI preference
                if method_lower == "llm":
                    is_openai_variant = "openai" in name_lower
                    if use_openai and is_openai_variant:
                        candidates.insert(0, (name, model))  # Prefer OpenAI variant
                    elif not use_openai and not is_openai_variant:
                        candidates.insert(0, (name, model))  # Prefer local variant
                    else:
                        candidates.append((name, model))
                else:
                    candidates.append((name, model))
                break

    if candidates:
        return candidates[0]  # Return first match (prioritized by OpenAI preference)

    return None, None


def get_model_labels(model) -> list:
    """Extract label names from a model.

    Args:
        model: Model instance

    Returns:
        List of label names, or empty list if not available
    """
    # Try different ways to get labels
    if hasattr(model, "classes_"):
        return list(model.classes_)
    elif hasattr(model, "rag") and hasattr(model.rag, "labels"):
        return list(model.rag.labels)
    elif hasattr(model, "labels"):
        return list(model.labels)
    elif hasattr(model, "_classes"):
        return list(model._classes)
    return []


def main():
    """Main CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load models from config
    try:
        models = load_models_from_config()
        if not models:
            print("❌ Error: No models found in configuration.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading models: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # List models if requested
    if args.list_models:
        print("Available models:")
        for i, name in enumerate(sorted(models.keys()), 1):
            print(f"  {i}. {name}")
        sys.exit(0)

    # Validate text is provided
    if not args.text:
        print("❌ Error: --text is required (unless --list-models is used).", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Find the appropriate model
    if args.model_name:
        # Use specific model name
        if args.model_name not in models:
            print(
                f"❌ Error: Model '{args.model_name}' not found in configuration.",
                file=sys.stderr,
            )
            print("\nAvailable models:", file=sys.stderr)
            for name in sorted(models.keys()):
                print(f"  - {name}", file=sys.stderr)
            sys.exit(1)
        model_name = args.model_name
        model = models[model_name]

        # Try to load persisted model if available (for trained models)
        # RAG models work without training, but traditional ML models need to be trained
        models_dir = get_models_dir()
        model_path = models_dir / model_name / "model.pkl"
        if model_path.exists():
            try:
                # Try loading persisted model
                persisted_model = load_persisted_model(model_name, models_dir=models_dir)
                if persisted_model is not None:
                    model = persisted_model
            except Exception:
                # If loading fails, use the config model (might be RAG which doesn't need training)
                pass
        else:
            # Check if this is a traditional ML model that needs training
            is_rag_model = any(
                rag_term in model_name.lower() for rag_term in ["rag", "kmajority", "centroid"]
            )
            if not is_rag_model:
                print(
                    f"⚠️  Warning: Model '{model_name}' appears to be a traditional ML model "
                    f"that requires training. No persisted model found at {model_path}.",
                    file=sys.stderr,
                )
                print(
                    "   Please train the model first using the pipeline scripts.",
                    file=sys.stderr,
                )
                print(
                    "   RAG models (RAG-kMajority, RAG-CentroidNN, RAG-LLM) work without training.",
                    file=sys.stderr,
                )
    elif args.method:
        # Find by method (convenience for RAG methods)
        model_name, model = find_model_by_method(models, args.method, args.use_openai)
        if model is None:
            print(
                f"❌ Error: No {args.method} model found in configuration.",
                file=sys.stderr,
            )
            print("\nAvailable models:", file=sys.stderr)
            for name in sorted(models.keys()):
                print(f"  - {name}", file=sys.stderr)
            sys.exit(1)
    else:
        print("❌ Error: Either --model-name or --method must be specified.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # For RAG-LLM, override LLM model if specified
    if args.llm_model:
        # Check if this is a RAG-LLM model
        if hasattr(model, "rag") and hasattr(model.rag, "model"):
            from intent_classifier.rag.rag_llm import RagLLM

            # Get current parameters
            top_k = getattr(model.rag, "top_k", 10)
            min_labels = getattr(model.rag, "min_labels", 4)

            # Create new model with specified LLM
            new_rag = RagLLM.load_default(model=args.llm_model, top_k=top_k, min_labels=min_labels)
            from intent_classifier.rag.adapter_sklearn import RagSklearnAdapter

            model = RagSklearnAdapter(new_rag)
            model_name = f"{model_name} (llm={args.llm_model})"

    # Make prediction
    try:
        predictions = model.predict([args.text])
        prediction = predictions[0]

        # Get probabilities if available
        probas = None
        labels = get_model_labels(model)

        if hasattr(model, "predict_proba"):
            try:
                probas_array = model.predict_proba([args.text])
                if probas_array is not None and probas_array.shape[0] > 0:
                    if labels and len(labels) == probas_array.shape[1]:
                        probas = {
                            label: float(probas_array[0, i]) for i, label in enumerate(labels)
                        }
                    else:
                        # Fallback: use indices if labels not available
                        probas = {
                            f"class_{i}": float(probas_array[0, i])
                            for i in range(probas_array.shape[1])
                        }
            except Exception:
                # Probabilities not available or error
                pass

        # Format output
        if args.output_format == "json":
            output = {
                "model": model_name,
                "text": args.text,
                "prediction": prediction,
            }
            if probas:
                output["probabilities"] = probas
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(f"Model: {model_name}")
            print(f"Text: {args.text}")
            print(f"Prediction: {prediction}")
            if probas:
                print("\nTop Probabilities:")
                sorted_probas = sorted(probas.items(), key=lambda x: x[1], reverse=True)
                for label, prob in sorted_probas[:10]:
                    print(f"  {label}: {prob:.4f}")

    except Exception as e:
        print(f"❌ Error during prediction: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
