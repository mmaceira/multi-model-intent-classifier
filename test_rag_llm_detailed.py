#!/usr/bin/env python3
"""
Detailed test script for RAG LLM predictions with extensive logging.

Tests predictions with 1 text and 3 texts to diagnose issues.
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from intent_classifier.datasets.dataset import get_dataset  # noqa: E402
from intent_classifier.rag.rag_llm import RagLLM  # noqa: E402

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

# Enable debug logging for RAG LLM components
logging.getLogger("intent_classifier.rag.rag_llm").setLevel(logging.DEBUG)
logging.getLogger("intent_classifier.rag.rag_llm.classifier").setLevel(logging.DEBUG)
logging.getLogger("intent_classifier.rag.rag_llm.response_parser").setLevel(logging.DEBUG)
logging.getLogger("intent_classifier.rag.rag_llm.prompt_builder").setLevel(logging.DEBUG)
logging.getLogger("intent_classifier.rag.rag_llm.client").setLevel(logging.DEBUG)


def test_single_prediction(classifier, test_doc, true_label=None):
    """Test prediction with a single document."""
    print("\n" + "=" * 80)
    print("TEST: Single Document Prediction")
    print("=" * 80)
    print(f"Document: {test_doc}")
    if true_label:
        print(f"True Label: {true_label}")
    print()

    try:
        print("Making prediction...")
        predictions = classifier.predict([test_doc])
        predicted_label = predictions[0]

        print("\n" + "-" * 80)
        print("RESULT:")
        print(f"  Document: {test_doc}")
        if true_label:
            print(f"  True Label: {true_label}")
            print(f"  Predicted: {predicted_label}")
            print(f"  Correct: {'✅ YES' if predicted_label == true_label else '❌ NO'}")
        else:
            print(f"  Predicted: {predicted_label}")
        print("-" * 80)

        return predicted_label
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_batch_prediction(classifier, test_docs, true_labels=None):
    """Test prediction with a batch of documents."""
    print("\n" + "=" * 80)
    print(f"TEST: Batch Prediction ({len(test_docs)} documents)")
    print("=" * 80)
    for i, doc in enumerate(test_docs, 1):
        print(f"{i}. {doc}")
    if true_labels:
        print("\nTrue Labels:")
        for i, label in enumerate(true_labels, 1):
            print(f"  {i}. {label}")
    print()

    try:
        print("Making predictions...")
        predictions = classifier.predict(test_docs)

        print("\n" + "-" * 80)
        print("RESULTS:")
        print("-" * 80)
        correct_count = 0
        for i, (doc, pred) in enumerate(zip(test_docs, predictions, strict=False), 1):
            doc_preview = doc[:60] + "..." if len(doc) > 60 else doc
            print(f"\n{i}. Document: {doc_preview}")
            if true_labels:
                true_label = true_labels[i - 1]
                print(f"   True Label: {true_label}")
                print(f"   Predicted: {pred}")
                is_correct = pred == true_label
                print(f"   Correct: {'✅ YES' if is_correct else '❌ NO'}")
                if is_correct:
                    correct_count += 1
            else:
                print(f"   Predicted: {pred}")

        if true_labels:
            accuracy = correct_count / len(test_docs)
            print(f"\n{'='*80}")
            print(f"Accuracy: {correct_count}/{len(test_docs)} = {accuracy:.2%}")
            print(f"{'='*80}")

        return predictions
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main test function."""
    print("=" * 80)
    print("RAG LLM Detailed Prediction Test")
    print("=" * 80)
    print()

    # Load dataset
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
        print(f"   - Classes: {sorted(classes)}")
        print(f"   - Test samples: {len(X_test)}")
        print()
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback

        traceback.print_exc()
        return

    # Get test samples
    test_doc_1 = X_test[0]
    true_label_1 = y_test[0]

    test_docs_3 = X_test[:3]
    true_labels_3 = y_test[:3]

    test_docs_7 = X_test[:7]
    true_labels_7 = y_test[:7]

    test_docs_12 = X_test[:12]
    true_labels_12 = y_test[:12]

    print("Test Documents Selected:")
    print(f"  1. Single: '{test_doc_1}' (label: {true_label_1})")
    print("  2. Batch of 3:")
    for i, (doc, label) in enumerate(zip(test_docs_3, true_labels_3, strict=False), 1):
        print(f"     {i}. '{doc}' (label: {label})")
    print("  3. Batch of 7:")
    for i, (doc, label) in enumerate(zip(test_docs_7, true_labels_7, strict=False), 1):
        print(f"     {i}. '{doc}' (label: {label})")
    print("  4. Batch of 12:")
    for i, (doc, label) in enumerate(zip(test_docs_12, true_labels_12, strict=False), 1):
        print(f"     {i}. '{doc}' (label: {label})")
    print()

    # Load classifier
    print("=" * 80)
    print("Loading RAG LLM Classifier")
    print("=" * 80)
    try:
        use_openai = os.getenv("OPENAI_API_KEY") is not None
        model_name = "ollama/llama3.1:8b"

        print("Configuration:")
        print(f"  - Model: {model_name}")
        print(f"  - Use OpenAI embeddings: {use_openai}")
        print()

        classifier = RagLLM.load_default(use_openai=use_openai, model=model_name)
        print("✅ Classifier loaded successfully")
        print(f"   - Number of labels: {len(classifier.labels)}")
        print(f"   - Labels: {sorted(classifier.labels)}")
        print(f"   - Top K: {classifier.top_k}")
        print(f"   - Batch size: {classifier.batch_size}")
        print()
    except Exception as e:
        print(f"❌ Error loading classifier: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test 1: Single document
    pred_1 = test_single_prediction(classifier, test_doc_1, true_label_1)

    # Test 2: Batch of 3 documents
    preds_3 = test_batch_prediction(classifier, test_docs_3, true_labels_3)

    # Test 3: Batch of 7 documents
    preds_7 = test_batch_prediction(classifier, test_docs_7, true_labels_7)

    # Test 4: Batch of 12 documents
    preds_12 = test_batch_prediction(classifier, test_docs_12, true_labels_12)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Single prediction: {'✅ SUCCESS' if pred_1 else '❌ FAILED'}")
    print(f"Batch prediction (3): {'✅ SUCCESS' if preds_3 else '❌ FAILED'}")
    print(f"Batch prediction (7): {'✅ SUCCESS' if preds_7 else '❌ FAILED'}")
    print(f"Batch prediction (12): {'✅ SUCCESS' if preds_12 else '❌ FAILED'}")

    if pred_1 and true_label_1:
        status = "✅ CORRECT" if pred_1 == true_label_1 else "❌ INCORRECT"
        print(f"\nSingle document accuracy: {status}")

    if preds_3 and true_labels_3:
        correct = sum(1 for p, t in zip(preds_3, true_labels_3, strict=False) if p == t)
        print(f"\nBatch (3 documents) accuracy: {correct}/3 = {correct/3:.2%}")

    if preds_7 and true_labels_7:
        correct = sum(1 for p, t in zip(preds_7, true_labels_7, strict=False) if p == t)
        print(f"Batch (7 documents) accuracy: {correct}/7 = {correct/7:.2%}")

    if preds_12 and true_labels_12:
        correct = sum(1 for p, t in zip(preds_12, true_labels_12, strict=False) if p == t)
        print(f"Batch (12 documents) accuracy: {correct}/12 = {correct/12:.2%}")

    print("=" * 80)


if __name__ == "__main__":
    main()
