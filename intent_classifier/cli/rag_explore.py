#!/usr/bin/env python3
"""
RAG-LLM Exploration CLI

This script provides a command-line interface for exploring RAG-LLM behaviour.
It uses the shared implementation under intent_classifier.rag.rag_llm and
reloads training examples and label definitions from the current dataset
configuration on each run.
"""

import argparse
import json
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the RAG-LLM exploration CLI.

    Exposed primarily for tests; console scripts should call :func:`main`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "RAG-LLM classifier CLI for intent classification "
            "(explore retrieval-augmented LLM predictions)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # OpenAI with default prompt\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "  %(prog)s --provider openai --model gpt-4o-mini "
            "--k 10 "
            '--text "reset my card pin"\\n\\n'
            "  # Ollama with short prompt (faster/cheaper)\n"
            "  export OLLAMA_HOST=http://localhost:11434\n"
            "  %(prog)s --provider ollama --model llama3.1:8b "
            "--k 10 --prompt-style short "
            '--text "reset my card pin"\\n\\n'
            "  # Using n8n prompt style (Catalan, email cleaning)\n"
            "  %(prog)s --provider ollama --model llama3.1:8b "
            "--k 10 --prompt-style n8n_prompt "
            '--text "reset my card pin"\\n'
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
        "--k",
        type=int,
        default=10,
        help="Number of similar examples to retrieve (default: 10)",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["default", "short", "n8n_prompt"],
        default="default",
        help=(
            'Prompt style to use: "default" (full detailed), '
            '"short" (concise), or "n8n_prompt" '
            "(email cleaning + classification, Catalan). "
            "(default: default)"
        ),
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text to classify",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    # Import heavy dependencies inside function for fast --help
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Import RAG-LLM implementation (heavy import, done after --help)
    try:
        from intent_classifier.rag.rag_llm import Retriever, _load_examples, classify_single
    except ImportError as e:
        print(f"❌ Error: Could not import RAG-LLM implementation: {e}", file=sys.stderr)
        sys.exit(1)

    # Load training examples and label definitions from dataset
    try:
        examples, label_defs = _load_examples()
    except Exception as e:  # pragma: no cover - defensive
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
            prompt_style=args.prompt_style,
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
