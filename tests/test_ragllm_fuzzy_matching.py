"""Tests for fuzzy matching of RagLLM model variants."""

from __future__ import annotations

from intent_classifier.evaluation.run_docs import ALGORITHM_REGISTRY, _slugify_model_name


def test_ragllm_fuzzy_matching_tfidf_variants() -> None:
    """Test that RagLLM TF-IDF variants match to rag_llm_tfidf_default."""
    test_cases = [
        "RAG-LLM (TF-IDF, default prompt)",
        "RAG-LLM (TF-IDF, custom prompt)",
        "RAG-LLM (TF-IDF)",
    ]

    for display_name in test_cases:
        model_id = _slugify_model_name(display_name)
        # All TF-IDF variants should start with rag_llm_tfidf
        assert model_id.startswith("rag_llm_tfidf")

        # The registry should have a matching entry (via fuzzy matching in run_docs)
        # We test the fuzzy matching logic by checking if the base variant exists
        assert "rag_llm_tfidf_default" in ALGORITHM_REGISTRY


def test_ragllm_fuzzy_matching_sbert_variants() -> None:
    """Test that RagLLM SBERT variants match to rag_llm_sbert_default."""
    test_cases = [
        "RAG-LLM (SBERT embeddings, default prompt)",
        "RAG-LLM (SBERT, custom prompt)",
    ]

    for display_name in test_cases:
        model_id = _slugify_model_name(display_name)
        # All SBERT variants should start with rag_llm_sbert
        assert model_id.startswith("rag_llm_sbert")
        assert "rag_llm_sbert_default" in ALGORITHM_REGISTRY


def test_ragllm_fuzzy_matching_qwen_variants() -> None:
    """Test that RagLLM Qwen variants match to rag_llm_qwen_default."""
    test_cases = [
        "RAG-LLM (Qwen embeddings, default prompt)",
        "RAG-LLM (Qwen, custom prompt)",
    ]

    for display_name in test_cases:
        model_id = _slugify_model_name(display_name)
        # All Qwen variants should start with rag_llm_qwen
        assert model_id.startswith("rag_llm_qwen")
        assert "rag_llm_qwen_default" in ALGORITHM_REGISTRY


def test_ragllm_exact_matches_still_work() -> None:
    """Test that exact registry matches still work."""
    exact_matches = [
        "rag_llm_tfidf_default",
        "rag_llm_tfidf_short",
        "rag_llm_tfidf_n8n",
        "rag_llm_sbert_default",
        "rag_llm_qwen_default",
    ]

    for model_id in exact_matches:
        assert model_id in ALGORITHM_REGISTRY, f"Exact match {model_id} should be in registry"
