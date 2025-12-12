#!/usr/bin/env python3
"""
Test script for RAG LLM predictions with Ollama and OpenAI.

This script tests the refactored RAG LLM classifier with both Ollama and OpenAI
to verify that all fixes are working correctly.
"""

import os
import sys

# Import path utilities
from intent_classifier.utils.paths import get_repo_root  # noqa: E402

# Add project root to path
project_root = get_repo_root()
sys.path.insert(0, str(project_root))

from intent_classifier.datasets.dataset import get_dataset  # noqa: E402
from intent_classifier.rag.rag_llm import RagLLM  # noqa: E402


def run_predictions(model_name: str, use_openai: bool = False, test_docs: list = None):
    """Test predictions with a specific model configuration.

    Args:
        model_name: Name/identifier for the model (for display)
        use_openai: Whether to use OpenAI embeddings
        test_docs: List of test documents to classify
    """
    print(f"\n{'='*70}")
    print(f"Testing {model_name}")
    print(f"{'='*70}")
    print("Configuration:")
    print(f"  - Model: {model_name}")
    print(f"  - Use OpenAI embeddings: {use_openai}")
    print(f"  - Test documents: {len(test_docs)}")
    print()

    try:
        # Load classifier
        print("Loading classifier...")
        classifier = RagLLM.load_default(use_openai=use_openai, model=model_name)
        print("✅ Classifier loaded successfully")
        print(f"   - Number of labels: {len(classifier.labels)}")
        print(f"   - Top K: {classifier.top_k}")
        print(f"   - Batch size: {classifier.batch_size}")
        print()

        # Make predictions
        print("Making predictions...")
        predictions = classifier.predict(test_docs)
        print("✅ Predictions completed")
        print()

        # Display results
        print("Results:")
        print("-" * 70)
        for i, (doc, pred) in enumerate(zip(test_docs, predictions, strict=False), 1):
            doc_preview = doc[:60] + "..." if len(doc) > 60 else doc
            print(f"{i}. {doc_preview}")
            print(f"   → {pred}")
        print("-" * 70)
        print()

        return True, predictions

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def main():
    """Main test function."""
    print("=" * 70)
    print("RAG LLM Prediction Test")
    print("=" * 70)
    print()

    # Load a small dataset for testing
    print("Loading test dataset...")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
            dataset_name="clinc150",
            use_oos=False,
            max_classes=10,
            max_train_samples=100,
            max_test_samples=50,
            seed=42,
        )
        print("✅ Dataset loaded:")
        print(f"   - Classes: {len(classes)}")
        print(f"   - Test samples: {len(X_test)}")
        print()

        # Use first 5 test documents for quick testing
        test_docs = X_test[:5]
        test_labels = y_test[:5]
        print(f"Using {len(test_docs)} test documents for prediction")
        print()

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        # Fallback to hardcoded test documents
        print("Using fallback test documents...")
        test_docs = [
            "What's the weather like today?",
            "Transfer $100 to my savings account",
            "Play some music",
            "What time is it?",
            "Book a flight to Paris",
        ]
        test_labels = None
        print()

    # Test 1: Ollama (local)
    print("\n" + "=" * 70)
    print("TEST 1: Ollama (Local Model)")
    print("=" * 70)
    ollama_success, ollama_preds = run_predictions(
        model_name="ollama/llama3.1:8b",
        use_openai=False,
        test_docs=test_docs,
    )

    # Test 2: OpenAI (if API key is available)
    print("\n" + "=" * 70)
    print("TEST 2: OpenAI (API Model)")
    print("=" * 70)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        print("✅ OPENAI_API_KEY found")
        openai_success, openai_preds = run_predictions(
            model_name="gpt-4o-mini",
            use_openai=True,
            test_docs=test_docs,
        )
    else:
        print("⚠️  OPENAI_API_KEY not found - skipping OpenAI test")
        print("   Set OPENAI_API_KEY environment variable to test OpenAI models")
        openai_success = None
        openai_preds = None

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Ollama (ollama/llama3.1:8b): {'✅ PASSED' if ollama_success else '❌ FAILED'}")
    if openai_success is not None:
        print(f"OpenAI (gpt-4o-mini): {'✅ PASSED' if openai_success else '❌ FAILED'}")
    else:
        print("OpenAI (gpt-4o-mini): ⏭️  SKIPPED (no API key)")

    if test_labels is not None:
        print("\nGround Truth Labels:")
        for i, (doc, true_label) in enumerate(zip(test_docs, test_labels, strict=False), 1):
            doc_preview = doc[:50] + "..." if len(doc) > 50 else doc
            print(f"  {i}. {doc_preview}")
            print(f"     True: {true_label}")
            if ollama_success:
                print(f"     Ollama: {ollama_preds[i-1]}")
            if openai_success:
                print(f"     OpenAI: {openai_preds[i-1]}")
            print()

    print("=" * 70)


if __name__ == "__main__":
    main()
