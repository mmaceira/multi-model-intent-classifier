"""Unit tests for the standalone rag_llm module."""

from __future__ import annotations

from collections.abc import Iterable

from intent_classifier.rag.rag_llm import classifier as rag_llm


def _build_examples() -> list[rag_llm.Example]:
    """Helper dataset with multiple intents for deterministic retrieval tests."""
    return [
        rag_llm.Example("Book me a flight to London tomorrow", "travel"),
        rag_llm.Example("I want to reserve a hotel in Paris", "travel"),
        rag_llm.Example("Play some relaxing jazz music", "music"),
        rag_llm.Example("Add milk and bread to my grocery list", "shopping"),
        rag_llm.Example("Remind me to call mom at 5pm", "reminder"),
        rag_llm.Example("Turn off the living room lights", "smart_home"),
    ]


def _retriever(examples: Iterable[rag_llm.Example] | None = None) -> rag_llm.TfIdfRetriever:
    """Convenience wrapper that creates a retriever from helper data."""
    return rag_llm.TfIdfRetriever(list(examples) if examples else _build_examples())


def test_retrieval_enforces_min_labels_and_similarity_order():
    retriever = _retriever()
    selected = retriever.select_topk_with_min_labels("book my trip", k=4, m=3)

    assert len(selected) == 4
    assert len({ex.label for ex, _ in selected}) >= 3

    scores = [score for _, score in selected]
    assert all(scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1))


def test_retrieval_handles_insufficient_label_variety():
    # Dataset with only two labels; requesting m=3 should return everything without error.
    examples = [
        rag_llm.Example("Flight to NYC", "travel"),
        rag_llm.Example("Book hotel", "travel"),
        rag_llm.Example("Set an alarm", "reminder"),
    ]
    retriever = _retriever(examples)

    selected = retriever.select_topk_with_min_labels("Need a trip", k=5, m=3)

    assert len(selected) == len(examples)
    assert len({ex.label for ex, _ in selected}) == 2


def test_classify_single_repairs_invalid_json(monkeypatch):
    retriever = _retriever()

    responses = iter(
        [
            "not-json-at-all",
            '{"label":"travel","confidence":1.25}',
        ]
    )

    def fake_completion(*args, **kwargs):
        return {"choices": [{"message": {"content": next(responses)}}]}

    monkeypatch.setattr(rag_llm, "completion", fake_completion)

    result = rag_llm.classify_single(
        model="test-model",
        query="Need to book a holiday flight",
        retriever=retriever,
        label_defs={"travel": "Trips and reservations"},
        k=3,
        m=2,
    )

    assert result["label"] == "travel"
    assert 0.0 <= result["confidence"] <= 1.0
    assert "allowed_labels" in result
    assert "shots_used" in result
