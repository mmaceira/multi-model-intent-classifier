#!/usr/bin/env python3
"""CLI for running intent classification with trained models."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cloudpickle

from intent_classifier.utils.label_utils import is_multilabel


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Intent classifier CLI for classifying text with trained models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a single text with a saved model
  %(prog)s --model-path artifacts/model.pkl --text "what's my account balance?"

  # Classify from a batch file
  %(prog)s --model-path artifacts/model.pkl --batch-file queries.txt --output text

  # Use JSON output (default)
  %(prog)s --model-path artifacts/model.pkl --text "reset my password" --output json
        """,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to saved model file (e.g., 'artifacts/model.pkl' or model directory)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Single text to classify (required if --batch-file not provided)",
    )
    parser.add_argument(
        "--batch-file",
        default=None,
        help="Path to file with one text per line to classify (optional)",
    )
    parser.add_argument(
        "--output",
        choices=["json", "text"],
        default="json",
        help="Output format: 'json' (default) or 'text'",
    )
    return parser


def get_model_labels(model: object) -> list[str]:
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


def main() -> None:
    """Main CLI entry point."""
    # Import heavy dependencies inside function for fast --help
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate that either --text or --batch-file is provided
    if not args.text and not args.batch_file:
        print("❌ Error: Either --text or --batch-file must be provided.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if args.text and args.batch_file:
        print("❌ Error: Cannot use both --text and --batch-file.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Load model from path
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print(f"❌ Error: Model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Try to load as a direct model file first
        if model_path.is_file():
            with open(model_path, "rb") as f:
                model = cloudpickle.load(f)
        else:
            # Try loading as a model directory (e.g., "Linear SVM/model.pkl")
            model_file = model_path / "model.pkl"
            if model_file.exists():
                with open(model_file, "rb") as f:
                    model = cloudpickle.load(f)
            else:
                # Path doesn't exist as file or directory with model.pkl
                print(
                    f"❌ Error: Model path not found: {model_path}\n"
                    "   Expected either a model file (.pkl or .joblib) "
                    "or a directory containing model.pkl",
                    file=sys.stderr,
                )
                sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Prepare texts to classify
    if args.batch_file:
        batch_path = Path(args.batch_file)
        if not batch_path.exists():
            print(f"❌ Error: Batch file not found: {batch_path}", file=sys.stderr)
            sys.exit(1)
        with open(batch_path, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [args.text]

    # Make predictions
    try:
        predictions = model.predict(texts)

        # Detect if model is multi-label
        is_multilabel_model = False
        if hasattr(model, "_is_multilabel"):
            is_multilabel_model = bool(getattr(model, "_is_multilabel", False))
        elif len(predictions) > 0:
            # Detect from prediction format
            is_multilabel_model = is_multilabel(predictions)

        # Get probabilities if available
        probas_list: list[dict[str, float]] | None = None
        labels = get_model_labels(model)

        if hasattr(model, "predict_proba"):
            try:
                probas_raw = model.predict_proba(texts)

                # Handle different probability formats
                if isinstance(probas_raw, list):
                    # Multi-label: list of arrays (one per label)
                    probas_list = []
                    for i in range(len(texts)):
                        probas = {}
                        for j, label in enumerate(labels):
                            if j < len(probas_raw):
                                # Get probability of positive class
                                if probas_raw[j].shape[1] > 1:
                                    probas[label] = float(probas_raw[j][i, 1])
                                else:
                                    probas[label] = float(probas_raw[j][i, 0])
                        probas_list.append(probas)
                elif probas_raw is not None and hasattr(probas_raw, "shape"):
                    # Single-label: 2D array (n_samples, n_classes)
                    if probas_raw.shape[0] > 0:
                        probas_list = []
                        for i in range(len(texts)):
                            if labels and len(labels) == probas_raw.shape[1]:
                                probas = {
                                    label: float(probas_raw[i, j]) for j, label in enumerate(labels)
                                }
                            else:
                                probas = {
                                    f"class_{j}": float(probas_raw[i, j])
                                    for j in range(probas_raw.shape[1])
                                }
                            probas_list.append(probas)
            except Exception:
                pass

        # Format output
        if args.output == "json":
            if len(texts) == 1:
                # Single prediction
                pred = predictions[0]
                output: dict[str, Any]
                if is_multilabel_model:
                    # Multi-label: pred is a list
                    if not isinstance(pred, (list, tuple)):
                        pred = [pred] if pred else []
                    output = {
                        "labels": list(pred),
                        "confidence": None,
                    }
                    if probas_list and probas_list[0]:
                        # Get confidence from top probability
                        top_prob = max(probas_list[0].values())
                        probas_json = {str(k): float(v) for k, v in probas_list[0].items()}
                        output["confidence"] = float(top_prob)
                        output["probabilities"] = probas_json
                else:
                    # Single-label: pred is a string
                    output = {
                        "label": str(pred),
                        "confidence": None,
                    }
                    if probas_list and probas_list[0]:
                        # Get confidence from top probability
                        top_prob = max(probas_list[0].values())
                        # Normalize probability keys/values to JSON-safe types
                        probas_json = {str(k): float(v) for k, v in probas_list[0].items()}
                        output["confidence"] = float(top_prob)
                        output["probabilities"] = probas_json
                print(json.dumps(output, ensure_ascii=False, indent=2))
            else:
                # Batch predictions
                results: list[dict[str, Any]] = []
                for i, (text, pred) in enumerate(zip(texts, predictions, strict=False)):
                    if is_multilabel_model:
                        # Multi-label: pred is a list
                        if not isinstance(pred, (list, tuple)):
                            pred = [pred] if pred else []
                        result: dict[str, Any] = {
                            "text": text,
                            "labels": list(pred),
                            "confidence": None,
                        }
                    else:
                        # Single-label: pred is a string
                        result = {"text": text, "label": str(pred), "confidence": None}

                    if probas_list and probas_list[i]:
                        top_prob = max(probas_list[i].values())
                        probas_json = {str(k): float(v) for k, v in probas_list[i].items()}
                        result["confidence"] = float(top_prob)
                        result["probabilities"] = probas_json
                    results.append(result)
                print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            # Text output
            for i, (text, pred) in enumerate(zip(texts, predictions, strict=False)):
                print(f"Text: {text}")
                if is_multilabel_model:
                    # Multi-label: pred is a list
                    if not isinstance(pred, (list, tuple)):
                        pred = [pred] if pred else []
                    if len(pred) == 0:
                        print("Labels: []")
                    elif len(pred) == 1:
                        print(f"Label: {pred[0]}")
                    else:
                        print(f"Labels: {', '.join(str(p) for p in pred)}")
                else:
                    # Single-label: pred is a string
                    print(f"Label: {pred}")

                if probas_list and probas_list[i]:
                    print("Top Probabilities:")
                    sorted_probas = sorted(probas_list[i].items(), key=lambda x: x[1], reverse=True)
                    for label, prob in sorted_probas[:5]:
                        print(f"  {label}: {prob:.4f}")
                if i < len(texts) - 1:
                    print()

    except Exception as e:
        print(f"❌ Error during prediction: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
