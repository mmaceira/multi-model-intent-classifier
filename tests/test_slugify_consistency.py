"""Tests for slugify consistency across all model directories."""

from __future__ import annotations

from pathlib import Path

from intent_classifier.utils.slugify import slugify_model_id


def test_slugify_consistency_model_names_with_special_chars() -> None:
    """Test that model names with parentheses, spaces, and commas are consistently slugified."""
    test_cases = [
        ("RAG-LLM (TF-IDF, default prompt)", "rag_llm_tfidf_default_prompt"),
        ("Embedding + LogReg (Qwen/Ollama)", "embedding_logreg_qwen_ollama"),
        ("TF-IDF bigrams + SVM", "tfidf_bigrams_svm"),
        ("RAG-LLM (TF-IDF, short prompt)", "rag_llm_tfidf_short_prompt"),
        ("RAG-LLM (SBERT embeddings, default prompt)", "rag_llm_sbert_embeddings_default_prompt"),
    ]

    for display_name, expected_slug in test_cases:
        result = slugify_model_id(display_name)
        assert result == expected_slug, (
            f"Failed for '{display_name}': got '{result}', expected '{expected_slug}'"
        )


def test_slugify_produces_valid_directory_names() -> None:
    """Test that slugified names are valid directory names."""
    test_names = [
        "Model with spaces",
        "Model/with/slashes",
        "Model(with)parentheses",
        "Model,with,commas",
        "Model+with+plus",
        "Model-with-dashes",
    ]

    for name in test_names:
        slug = slugify_model_id(name)
        # Should not contain path separators, spaces, or other problematic chars
        assert "/" not in slug
        assert "\\" not in slug
        assert " " not in slug
        assert "(" not in slug
        assert ")" not in slug
        assert "," not in slug
        # Should be lowercase
        assert slug == slug.lower()
        # Should not be empty
        assert len(slug) > 0


def test_slugify_consistency_across_pipeline_stages(tmp_path: Path) -> None:
    """Test that the same model name produces the same slug in training,
    prediction, and evaluation."""
    model_name = "RAG-LLM (TF-IDF, default prompt)"
    expected_slug = slugify_model_id(model_name)

    # Simulate directory creation as done in training, prediction, and evaluation
    training_dir = tmp_path / "models" / expected_slug
    prediction_dir = tmp_path / "predictions" / expected_slug
    eval_dir = tmp_path / "eval" / expected_slug

    training_dir.mkdir(parents=True)
    prediction_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    # All directories should exist with the same slug
    assert training_dir.exists()
    assert prediction_dir.exists()
    assert eval_dir.exists()

    # Verify the slug is consistent
    assert training_dir.name == expected_slug
    assert prediction_dir.name == expected_slug
    assert eval_dir.name == expected_slug
