#!/usr/bin/env python3
"""
Simple script to test Ollama and OpenAI endpoint connections.

This script performs a basic connection test to verify that:
- Ollama is running and accessible
- OpenAI API key is valid (if provided)

Configuration:
- Ollama endpoint: Set OLLAMA_API_BASE environment variable (default: http://localhost:11434)
  Example: export OLLAMA_API_BASE=http://localhost:11434
- OpenAI: Set OPENAI_API_KEY environment variable
- Model selection: The model name comes from the active layered config
  (base/providers/dataset/experiment) or can be overridden via command line arguments
"""

import argparse
import os
import sys

# Add project root to path
from intent_classifier.utils.paths import get_repo_root

project_root = get_repo_root()
sys.path.insert(0, str(project_root))

# Configure litellm logging before importing
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("LITELLM_SUPPRESS_LOGGING", "true")

# These imports must come after environment setup
from dotenv import load_dotenv  # noqa: E402
from litellm import completion  # noqa: E402

# Load environment variables
load_dotenv()

# Import config loader after environment setup
try:
    from intent_classifier.utils.config_loader import discover_config_file, load_config
except ImportError:
    # Fallback if config loader not available
    load_config = None  # type: ignore[assignment]
    discover_config_file = None  # type: ignore[assignment]


def normalize_endpoint(endpoint: str) -> str:
    """Normalize endpoint URL by adding http:// if protocol is missing.

    Args:
        endpoint: Endpoint URL (with or without protocol)

    Returns:
        Normalized endpoint URL with protocol
    """
    endpoint = endpoint.strip()
    if not endpoint.startswith(("http://", "https://")):
        # Default to http:// if no protocol specified
        endpoint = f"http://{endpoint}"
    return endpoint


def normalize_ollama_model(model: str) -> str:
    """Normalize Ollama model name by adding 'ollama/' prefix if missing.

    Args:
        model: Model identifier (e.g., "qwen2.5:14b" or "ollama/qwen2.5:14b")

    Returns:
        Normalized model identifier with 'ollama/' prefix
    """
    model = model.strip()
    if not model.startswith("ollama/"):
        model = f"ollama/{model}"
    return model


def check_ollama(
    model: str = "ollama/llama3.1:8b", endpoint: str | None = None
) -> tuple[bool, str]:
    """Test Ollama connection with a simple prompt.

    Args:
        model: Model identifier (e.g., "ollama/llama3.1:8b" or "qwen2.5:14b")
        endpoint: Optional endpoint URL (overrides OLLAMA_API_BASE env var)

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Normalize model name (add 'ollama/' prefix if missing)
    normalized_model = normalize_ollama_model(model)

    # Use provided endpoint or fall back to environment variable or default
    if endpoint:
        ollama_base = normalize_endpoint(endpoint)
    else:
        env_endpoint = os.getenv("OLLAMA_API_BASE")
        ollama_base = normalize_endpoint(env_endpoint) if env_endpoint else "http://localhost:11434"

    print(f"Testing Ollama connection ({normalized_model})...")
    print(f"  Endpoint: {ollama_base}")
    if model != normalized_model:
        print(f"  Note: Model name normalized from '{model}' to '{normalized_model}'")

    # Set environment variable for litellm (it reads OLLAMA_API_BASE automatically)
    os.environ["OLLAMA_API_BASE"] = ollama_base

    try:
        response = completion(
            model=normalized_model,
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Hello' in one word and then tell me what model you are using.",
                }
            ],
            temperature=0.0,
        )
        result = response["choices"][0]["message"]["content"].strip()
        return True, f"✅ Ollama connection successful! Response: {result}"
    except Exception as e:
        return False, f"❌ Ollama connection failed: {e}"


def check_openai(model: str = "gpt-4o-mini") -> tuple[bool, str]:
    """Test OpenAI connection with a simple prompt.

    Args:
        model: Model identifier (e.g., "gpt-4o-mini")

    Returns:
        Tuple of (success: bool, message: str)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "⚠️  OPENAI_API_KEY not found - skipping OpenAI test"

    print(f"Testing OpenAI connection ({model})...")
    try:
        response = completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Hello' in one word and then tell me what model you are using.",
                }
            ],
            temperature=0.0,
        )
        result = response["choices"][0]["message"]["content"].strip()
        return True, f"✅ OpenAI connection successful! Response: {result}"
    except Exception as e:
        return False, f"❌ OpenAI connection failed: {e}"


def main():
    """Run connection tests for Ollama and OpenAI."""
    parser = argparse.ArgumentParser(
        description="Test Ollama and OpenAI endpoint connections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings (reads from config file if available)
  %(prog)s

  # Test with custom Ollama endpoint and model
  %(prog)s --ollama-endpoint http://localhost:11434 --ollama-model ollama/llama3.2:3b

  # Endpoint without protocol (http:// will be added automatically)
  %(prog)s --ollama-endpoint example.com:11434 --ollama-model qwen2.5:14b

  # Model without 'ollama/' prefix (will be added automatically)
  %(prog)s --ollama-model qwen2.5:14b

  # Test with specific config file
  CONFIG_FILE=config/experiments/nlu_plus/default.yaml %(prog)s

Configuration Priority (highest to lowest):
  1. Command-line arguments (--ollama-endpoint, --ollama-model)
  2. Environment variables (OLLAMA_API_BASE)
  3. Config file (config/experiments/{dataset}/{variant}.yaml)
  4. Defaults (http://localhost:11434, ollama/llama3.1:8b)
        """,
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama model identifier. Overrides config file and defaults. "
        "The 'ollama/' prefix will be added automatically if missing.",
    )
    parser.add_argument(
        "--ollama-endpoint",
        default=None,
        help="Ollama endpoint URL. Overrides config file and OLLAMA_API_BASE env var. "
        "Protocol (http://) will be added automatically if missing.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model identifier (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Config file to read settings from (e.g., config/experiments/nlu_plus/default.yaml). "
        "If not specified, uses CONFIG_FILE env var or discovers default config.",
    )
    args = parser.parse_args()

    # Load config from file if available
    ollama_model_from_config = None
    ollama_endpoint_from_config = None
    config_source = None

    if load_config is not None:
        try:
            config_file = args.config_file or discover_config_file()
            config = load_config(config_file)
            config_source = config_file

            # Read from config if not overridden by command line
            # LLM settings are now in config/base/providers.yaml and merged into config["model"]
            if "model" in config:
                if args.ollama_model is None:
                    # Try model.llm_model first (if present), then check provider defaults
                    ollama_model_from_config = config["model"].get("llm_model")
                if args.ollama_endpoint is None:
                    # Try model.ollama_endpoint first (for backward
                    # compatibility), then check LLM config
                    ollama_endpoint_from_config = config["model"].get("ollama_endpoint")
        except Exception:
            # Silently fall back to defaults if config loading fails
            pass

    # Determine final values (command line > config > env > defaults)
    ollama_model = args.ollama_model or ollama_model_from_config or "ollama/llama3.1:8b"

    ollama_endpoint = (
        args.ollama_endpoint
        or ollama_endpoint_from_config
        or os.getenv("OLLAMA_API_BASE")
        or "http://localhost:11434"
    )

    print("=" * 70)
    print("LLM Endpoint Connection Test")
    print("=" * 70)
    print()

    # Show configuration
    print("Configuration:")
    if config_source:
        print(f"  Config file: {config_source}")
    print(f"  Ollama endpoint: {ollama_endpoint}")
    print(f"  Ollama model: {ollama_model}")
    print(f"  OpenAI model: {args.openai_model}")
    print()

    # Test Ollama
    print("-" * 70)
    ollama_success, ollama_msg = check_ollama(model=ollama_model, endpoint=ollama_endpoint)
    print(ollama_msg)
    print()

    # Test OpenAI
    print("-" * 70)
    openai_result = check_openai(model=args.openai_model)
    if openai_result[0] is None:
        print(openai_result[1])
    else:
        openai_success, openai_msg = openai_result
        print(openai_msg)
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Ollama: {'✅ PASSED' if ollama_success else '❌ FAILED'}")
    if openai_result[0] is None:
        print("OpenAI: ⏭️  SKIPPED (no API key)")
    else:
        openai_success = openai_result[0]
        print(f"OpenAI: {'✅ PASSED' if openai_success else '❌ FAILED'}")
    print("=" * 70)

    # Exit with error code if any test failed
    if not ollama_success or (openai_result[0] is not None and not openai_result[0]):
        sys.exit(1)


if __name__ == "__main__":
    main()
