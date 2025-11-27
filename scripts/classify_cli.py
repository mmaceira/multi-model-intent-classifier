#!/usr/bin/env python3
"""CLI for running intent classification with trained models."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


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
            import cloudpickle

            with open(model_path, "rb") as f:
                model = cloudpickle.load(f)
        else:
            # Try loading as a model directory (e.g., "Linear SVM/model.pkl")
            model_file = model_path / "model.pkl"
            if model_file.exists():
                import cloudpickle

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
        with open(batch_path) as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [args.text]

    # Make predictions
    try:
        predictions = model.predict(texts)

        # Get probabilities if available
        probas_list = None
        labels = get_model_labels(model)

        if hasattr(model, "predict_proba"):
            try:
                probas_array = model.predict_proba(texts)
                if probas_array is not None and probas_array.shape[0] > 0:
                    probas_list = []
                    for i in range(len(texts)):
                        if labels and len(labels) == probas_array.shape[1]:
                            probas = {
                                label: float(probas_array[i, j]) for j, label in enumerate(labels)
                            }
                        else:
                            probas = {
                                f"class_{j}": float(probas_array[i, j])
                                for j in range(probas_array.shape[1])
                            }
                        probas_list.append(probas)
            except Exception:
                pass

        # Format output
        if args.output == "json":
            if len(texts) == 1:
                # Single prediction
                output = {
                    "label": predictions[0],
                    "confidence": None,
                }
                if probas_list and probas_list[0]:
                    # Get confidence from top probability
                    top_prob = max(probas_list[0].values())
                    output["confidence"] = top_prob
                    output["probabilities"] = probas_list[0]
                print(json.dumps(output, ensure_ascii=False, indent=2))
            else:
                # Batch predictions
                results = []
                for i, (text, pred) in enumerate(zip(texts, predictions, strict=False)):
                    result = {"text": text, "label": pred, "confidence": None}
                    if probas_list and probas_list[i]:
                        top_prob = max(probas_list[i].values())
                        result["confidence"] = top_prob
                        result["probabilities"] = probas_list[i]
                    results.append(result)
                print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            # Text output
            for i, (text, pred) in enumerate(zip(texts, predictions, strict=False)):
                print(f"Text: {text}")
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
