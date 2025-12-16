from intent_classifier.evaluation.run_docs import ALGORITHM_REGISTRY, _slugify_model_name


def test_slugify_model_name_basic() -> None:
    assert _slugify_model_name("TF-IDF bigrams + SVM") == "tfidf_bigrams_svm"
    assert _slugify_model_name("Embedding + LogReg (Qwen/Ollama)") == "embedding_logreg_qwen_ollama"


def test_algorithm_registry_contains_known_models() -> None:
    # Sanity check that a few core models are documented
    for display_name in [
        "Naive Bayes",
        "Linear SVM",
        "TF-IDF bigrams + SVM",
    ]:
        slug = _slugify_model_name(display_name)
        assert slug in ALGORITHM_REGISTRY, f"Missing registry entry for {display_name} ({slug})"
