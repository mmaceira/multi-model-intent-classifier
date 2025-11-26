#!/usr/bin/env python3
"""
Unified RAG Classifier CLI

This script provides a command-line interface for all RAG classifier methods:
- RAG-kMajority
- RAG-CentroidNN
- RAG-LLM

It loads models from the configuration and provides a unified interface for predictions.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from intent_classifier.utils.model_loader import load_models_from_config  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(description="Unified RAG classifier CLI for all RAG methods.")
    parser.add_argument(
        "--method",
        required=True,
        choices=["kmajority", "centroid", "llm"],
        help="RAG method to use: 'kmajority', 'centroid', or 'llm'",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Single query to classify.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Specific model name from config (e.g., 'RAG-LLM (local-embeddings)'). "
        "If not provided, uses the first matching model of the specified method.",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="For LLM method: use OpenAI embeddings variant (if available). "
        "For other methods: use OpenAI embeddings.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="For LLM method: specific LLM model to use "
        "(e.g., 'gpt-4o-mini', 'ollama/llama3.1:8b'). "
        "If not provided, uses default from config.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="json",
        help="Output format: 'json' (default) or 'text'",
    )
    return parser


def find_model_by_method(models: dict, method: str, use_openai: bool = False) -> tuple:
    """Find a model matching the specified method.

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
        "kmajority": ["rag_kmajority", "rag-kmajority", "rag kmajority"],
        "centroid": ["rag_centroid", "rag-centroidnn", "rag centroidnn", "rag-centroid"],
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
        sys.exit(1)

    # Find the appropriate model
    if args.model_name:
        # Use specific model name
        if args.model_name not in models:
            print(
                f"❌ Error: Model '{args.model_name}' not found in configuration.",
                file=sys.stderr,
            )
            print(f"Available models: {', '.join(models.keys())}", file=sys.stderr)
            sys.exit(1)
        model_name = args.model_name
        model = models[model_name]
    else:
        # Find by method
        model_name, model = find_model_by_method(models, args.method, args.use_openai)
        if model is None:
            print(
                f"❌ Error: No {args.method} model found in configuration.",
                file=sys.stderr,
            )
            print(f"Available models: {', '.join(models.keys())}", file=sys.stderr)
            sys.exit(1)

    # For LLM method, override model if specified
    if args.method == "llm" and args.llm_model:
        # Check if the model has a model attribute we can override
        if hasattr(model, "rag") and hasattr(model.rag, "model"):
            # Create a new instance with the specified model
            from intent_classifier.rag.rag_llm import RagLLM

            # Get current parameters
            top_k = getattr(model.rag, "top_k", 10)
            min_labels = getattr(model.rag, "min_labels", 4)

            # Create new model with specified LLM
            new_rag = RagLLM.load_default(model=args.llm_model, top_k=top_k, min_labels=min_labels)
            from intent_classifier.rag.adapter_sklearn import RagSklearnAdapter

            model = RagSklearnAdapter(new_rag)
            model_name = f"{model_name} (model={args.llm_model})"

    # Make prediction
    try:
        predictions = model.predict([args.text])
        prediction = predictions[0]

        # Get probabilities if available
        probas = None
        if hasattr(model, "predict_proba"):
            try:
                probas_array = model.predict_proba([args.text])
                if probas_array is not None and probas_array.shape[0] > 0:
                    # Get label names from model if available
                    if hasattr(model, "rag") and hasattr(model.rag, "labels"):
                        labels = model.rag.labels
                        probas = {
                            label: float(probas_array[0, i]) for i, label in enumerate(labels)
                        }
            except Exception:
                pass  # Probabilities not available

        # Format output
        if args.output_format == "json":
            output = {
                "method": args.method,
                "model": model_name,
                "text": args.text,
                "prediction": prediction,
            }
            if probas:
                output["probabilities"] = probas
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(f"Method: {args.method}")
            print(f"Model: {model_name}")
            print(f"Text: {args.text}")
            print(f"Prediction: {prediction}")
            if probas:
                print("\nProbabilities:")
                for label, prob in sorted(probas.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {label}: {prob:.4f}")

    except Exception as e:
        print(f"❌ Error during prediction: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
