#!/usr/bin/env python3
"""
RAG-LLM Classifier CLI

This script provides a command-line interface for RAG-LLM classification.
It uses the standalone rag_llm implementation for flexible LLM provider support.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path so `rag_llm` can be imported when installed or run from source.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the RAG-LLM CLI.

    Exposed primarily for tests; console scripts should call :func:`main`.
    """
    parser = argparse.ArgumentParser(
        description="RAG-LLM classifier CLI for intent classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # OpenAI\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "  %(prog)s --provider openai --model gpt-4o-mini "
            "--labels data/labels.json --k 10 "
            '--text "reset my card pin"\n\n'
            "  # Ollama (local)\n"
            "  export OLLAMA_HOST=http://localhost:11434\n"
            "  %(prog)s --provider ollama --model llama3.1:8b "
            "--labels data/labels.json --k 10 "
            '--text "reset my card pin"\n'
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM provider: 'openai' or 'ollama' (default: ollama)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLM model identifier (e.g., 'gpt-4o-mini' for OpenAI, 'llama3.1:8b' for Ollama)",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help='Path to JSON file with label definitions (format: {"label_name": "description"})',
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of similar examples to retrieve (default: 10)",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text to classify",
    )

    return parser


def main(argv: list[str] | None = None):
    """Main CLI entry point."""
    # Import heavy dependencies inside function for fast --help
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Import rag_llm module (heavy import, done after --help)
    try:
        from rag_llm import Retriever, _load_examples, classify_single
    except ImportError as e:
        print(f"❌ Error: Could not import rag_llm module: {e}", file=sys.stderr)
        sys.exit(1)

    # Load labels from JSON file
    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"❌ Error: Labels file not found: {labels_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(labels_path) as f:
            label_defs = json.load(f)
        if not isinstance(label_defs, dict):
            print(
                "❌ Error: Labels file must contain a JSON object (dict)",
                file=sys.stderr,
            )
            sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in labels file: {e}", file=sys.stderr)
        sys.exit(1)

    # Load training examples
    try:
        examples, _ = _load_examples()
    except Exception as e:
        print(f"❌ Error loading training examples: {e}", file=sys.stderr)
        sys.exit(1)

    # Create retriever
    try:
        retriever = Retriever(examples)
    except Exception as e:
        print(f"❌ Error creating retriever: {e}", file=sys.stderr)
        sys.exit(1)

    # Build model identifier based on provider
    if args.provider == "ollama":
        model_id = f"ollama/{args.model}"
    else:
        model_id = args.model

    # Classify
    try:
        result = classify_single(
            model=model_id,
            query=args.text,
            retriever=retriever,
            label_defs=label_defs,
            k=args.k,
            m=4,  # Minimum labels (fixed for now)
        )

        # Output result as JSON
        output = {
            "label": result["label"],
            "confidence": result["confidence"],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"❌ Error during classification: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
