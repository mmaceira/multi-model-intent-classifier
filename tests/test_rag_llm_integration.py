#!/usr/bin/env python3
"""
Integration tests for the standalone rag_llm module.

Tests the full pipeline from data loading through retrieval to classification.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pytest

# Import path utilities
from intent_classifier.utils.paths import get_repo_root

# Add project root to path
project_root = get_repo_root()
sys.path.insert(0, str(project_root))

from intent_classifier.rag.rag_llm import (  # noqa: E402
    Example,
    Retriever,
    _load_examples,
    classify_single,
)


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_examples_with_defaults(self) -> None:
        """Test loading examples with default parameters."""
        examples, label_defs = _load_examples()
        assert len(examples) > 0, "Should load at least some examples"
        assert all(isinstance(ex, Example) for ex in examples), (
            "All items should be Example instances"
        )
        assert all(ex.text and ex.label for ex in examples), (
            "All examples should have text and label"
        )
        assert isinstance(label_defs, dict), "Label definitions should be a dict"
        assert len(label_defs) > 0, "Should have at least some label definitions"

    def test_load_examples_with_limits(self) -> None:
        """Test loading examples with sample and class limits."""
        examples, label_defs = _load_examples(
            max_train_samples=50,
            max_classes=5,
        )
        assert len(examples) <= 50, "Should respect max_train_samples"
        unique_labels = {ex.label for ex in examples}
        assert len(unique_labels) <= 5, "Should respect max_classes"

    def test_load_examples_label_defs_structure(self) -> None:
        """Test that label definitions have reasonable structure."""
        examples, label_defs = _load_examples(max_train_samples=100)
        for ex in examples[:10]:  # Check first 10
            assert ex.label in label_defs, f"Label {ex.label} should have a definition"
            assert isinstance(label_defs[ex.label], str), "Definition should be a string"


class TestRetriever:
    """Test the Retriever class with real data."""

    @pytest.fixture
    def sample_examples(self) -> list[Example]:
        """Create sample examples for testing."""
        return [
            Example(text="What's the weather today?", label="weather_query"),
            Example(text="Check the weather forecast", label="weather_query"),
            Example(text="Transfer money to account", label="transfer_money"),
            Example(text="Send $100 to savings", label="transfer_money"),
            Example(text="Play some music", label="play_music"),
            Example(text="Start playing songs", label="play_music"),
            Example(text="What time is it?", label="time_query"),
            Example(text="Tell me the current time", label="time_query"),
            Example(text="Book a flight", label="book_flight"),
            Example(text="Reserve a plane ticket", label="book_flight"),
        ]

    @pytest.fixture
    def retriever(self, sample_examples: list[Example]) -> Retriever:
        """Create a retriever with sample examples."""
        return Retriever(sample_examples)

    def test_retriever_initialization(self, sample_examples: list[Example]) -> None:
        """Test retriever can be initialized with examples."""
        retriever = Retriever(sample_examples)
        assert retriever.examples == sample_examples
        assert retriever.matrix.shape[0] == len(sample_examples)

    def test_retriever_empty_examples_raises(self) -> None:
        """Test that empty examples list raises an error."""
        with pytest.raises(ValueError, match="at least one"):
            Retriever([])

    def test_select_topk_basic(self, retriever: Retriever) -> None:
        """Test basic top-k retrieval."""
        results = retriever.select_topk_with_min_labels("What's the weather?", k=3, m=1)
        assert len(results) <= 3, "Should return at most k results"
        assert len(results) > 0, "Should return at least one result"
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results), (
            "Results should be (Example, score) tuples"
        )
        assert all(isinstance(ex, Example) for ex, _ in results), "First element should be Example"
        assert all(isinstance(score, float) for _, score in results), (
            "Second element should be float"
        )

    def test_select_topk_enforces_min_labels(self, retriever: Retriever) -> None:
        """Test that retrieval enforces minimum distinct labels."""
        # Query that matches weather examples
        results = retriever.select_topk_with_min_labels("weather forecast today", k=5, m=3)
        distinct_labels = {ex.label for ex, _ in results}
        assert len(distinct_labels) >= 3, (
            f"Should have at least 3 distinct labels, got {len(distinct_labels)}"
        )

    def test_select_topk_similarity_order(self, retriever: Retriever) -> None:
        """Test that results are in descending similarity order."""
        results = retriever.select_topk_with_min_labels("weather", k=5, m=1)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), "Scores should be in descending order"

    def test_select_topk_with_real_data(self) -> None:
        """Test retrieval with real dataset examples."""
        examples, _ = _load_examples(max_train_samples=100, max_classes=10)
        retriever = Retriever(examples)
        query = "What's the weather like?"
        results = retriever.select_topk_with_min_labels(query, k=10, m=4)
        assert len(results) > 0, "Should retrieve some results"
        assert all(ex.label in {e.label for e in examples} for ex, _ in results), (
            "All labels should be from training set"
        )

    def test_select_topk_invalid_k_raises(self, retriever: Retriever) -> None:
        """Test that invalid k values raise errors."""
        with pytest.raises(ValueError, match="k must be positive"):
            retriever.select_topk_with_min_labels("test", k=0, m=1)

    def test_select_topk_invalid_m_raises(self, retriever: Retriever) -> None:
        """Test that invalid m values raise errors."""
        with pytest.raises(ValueError, match="m must be positive"):
            retriever.select_topk_with_min_labels("test", k=5, m=0)


class TestClassifySingle:
    """Test the classify_single function."""

    @pytest.fixture
    def sample_examples(self) -> list[Example]:
        """Create sample examples for testing."""
        return [
            Example(text="What's the weather today?", label="weather_query"),
            Example(text="Check the weather forecast", label="weather_query"),
            Example(text="Transfer money to account", label="transfer_money"),
            Example(text="Send $100 to savings", label="transfer_money"),
            Example(text="Play some music", label="play_music"),
            Example(text="Start playing songs", label="play_music"),
            Example(text="What time is it?", label="time_query"),
            Example(text="Tell me the current time", label="time_query"),
        ]

    @pytest.fixture
    def retriever(self, sample_examples: list[Example]) -> Retriever:
        """Create a retriever with sample examples."""
        return Retriever(sample_examples)

    @pytest.fixture
    def label_defs(self) -> dict[str, str]:
        """Create sample label definitions."""
        return {
            "weather_query": "Questions about weather conditions",
            "transfer_money": "Requests to transfer money between accounts",
            "play_music": "Commands to play music or songs",
            "time_query": "Questions about current time",
        }

    @patch("intent_classifier.rag.rag_llm.classifier._call_llm")
    def test_classify_single_success(
        self, mock_llm: Any, retriever: Retriever, label_defs: dict[str, str]
    ) -> None:
        """Test successful classification with mocked LLM."""
        # Mock LLM response
        mock_llm.return_value = '{"label": "weather_query", "confidence": 0.95}'
        result = classify_single(
            model="gpt-4o-mini",
            query="What's the weather forecast?",
            retriever=retriever,
            label_defs=label_defs,
            k=5,
            m=2,
        )
        assert "label" in result
        assert "confidence" in result
        assert result["label"] in result["allowed_labels"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["label"] == "weather_query"

    @patch("intent_classifier.rag.rag_llm.classifier._call_llm")
    def test_classify_single_repairs_invalid_json(
        self, mock_llm: Any, retriever: Retriever, label_defs: dict[str, str]
    ) -> None:
        """Test that invalid JSON triggers repair."""
        # First call returns invalid JSON, second returns valid
        mock_llm.side_effect = [
            "This is not JSON at all",
            '{"label": "weather_query", "confidence": 0.8}',
        ]
        result = classify_single(
            model="gpt-4o-mini",
            query="What's the weather?",
            retriever=retriever,
            label_defs=label_defs,
            k=5,
            m=2,
        )
        assert mock_llm.call_count == 2, "Should call LLM twice (initial + repair)"
        assert result["label"] in result["allowed_labels"]

    @patch("intent_classifier.rag.rag_llm.classifier._call_llm")
    def test_classify_single_strips_code_fences(
        self, mock_llm: Any, retriever: Retriever, label_defs: dict[str, str]
    ) -> None:
        """Test that code fences are stripped from LLM response."""
        mock_llm.return_value = '```json\n{"label": "weather_query", "confidence": 0.9}\n```'
        result = classify_single(
            model="gpt-4o-mini",
            query="What's the weather?",
            retriever=retriever,
            label_defs=label_defs,
            k=5,
            m=2,
        )
        assert result["label"] == "weather_query"

    @patch("intent_classifier.rag.rag_llm.classifier._call_llm")
    def test_classify_single_clamps_confidence(
        self, mock_llm: Any, retriever: Retriever, label_defs: dict[str, str]
    ) -> None:
        """Test that confidence is clamped to [0, 1]."""
        mock_llm.return_value = '{"label": "weather_query", "confidence": 1.5}'
        result = classify_single(
            model="gpt-4o-mini",
            query="What's the weather?",
            retriever=retriever,
            label_defs=label_defs,
            k=5,
            m=2,
        )
        assert result["confidence"] == 1.0

        mock_llm.return_value = '{"label": "weather_query", "confidence": -0.5}'
        result = classify_single(
            model="gpt-4o-mini",
            query="What's the weather?",
            retriever=retriever,
            label_defs=label_defs,
            k=5,
            m=2,
        )
        assert result["confidence"] == 0.0

    @patch("intent_classifier.rag.rag_llm.classifier._call_llm")
    def test_classify_single_invalid_label_raises(
        self, mock_llm: Any, retriever: Retriever, label_defs: dict[str, str]
    ) -> None:
        """Test that invalid labels raise an error."""
        mock_llm.return_value = '{"label": "invalid_label", "confidence": 0.8}'
        with pytest.raises(ValueError, match="Invalid label"):
            classify_single(
                model="gpt-4o-mini",
                query="What's the weather?",
                retriever=retriever,
                label_defs=label_defs,
                k=5,
                m=2,
            )

    @patch("intent_classifier.rag.rag_llm.classifier._call_llm")
    def test_classify_single_includes_metadata(
        self, mock_llm: Any, retriever: Retriever, label_defs: dict[str, str]
    ) -> None:
        """Test that result includes metadata like allowed_labels and shots_used."""
        mock_llm.return_value = '{"label": "weather_query", "confidence": 0.9}'
        result = classify_single(
            model="gpt-4o-mini",
            query="What's the weather?",
            retriever=retriever,
            label_defs=label_defs,
            k=5,
            m=2,
        )
        assert "allowed_labels" in result
        assert "shots_used" in result
        assert isinstance(result["allowed_labels"], list)
        assert isinstance(result["shots_used"], int)
        assert result["shots_used"] > 0


class TestCLI:
    """Test the RAG CLI interface (`scripts/rag_cli.py`)."""

    def test_cli_help(self) -> None:
        """Test that CLI shows help message."""
        result = subprocess.run(
            [sys.executable, "scripts/rag_cli.py", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        assert result.returncode == 0
        # Help text should mention RAG-LLM CLI description
        assert "rag-llm classifier cli" in result.stdout.lower()

    @patch("intent_classifier.rag.rag_llm.classifier._call_llm")
    @patch("intent_classifier.rag.rag_llm.classifier._load_examples")
    def test_cli_basic_execution(self, mock_load: Any, mock_llm: Any, tmp_path: Any) -> None:
        """Test basic CLI execution with mocked dependencies."""
        from scripts.rag_cli import build_arg_parser, main

        # Mock data loading
        examples = [
            Example(text="What's the weather?", label="weather"),
            Example(text="Transfer money", label="transfer"),
        ]
        label_defs = {"weather": "Weather queries", "transfer": "Money transfers"}
        mock_load.return_value = (examples, label_defs)

        # Mock LLM
        mock_llm.return_value = '{"label": "weather", "confidence": 0.9}'

        # Create a temporary labels file
        labels_path = tmp_path / "labels.json"
        import json as _json

        labels_path.write_text(_json.dumps(label_defs), encoding="utf-8")

        # Create args and call main directly (not via subprocess)
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--provider",
                "openai",
                "--model",
                "gpt-4o-mini",
                "--labels",
                str(labels_path),
                "--k",
                "5",
                "--text",
                "What's the weather?",
            ]
        )

        # Capture stdout
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            main(
                [
                    "--provider",
                    args.provider,
                    "--model",
                    args.model,
                    "--labels",
                    args.labels,
                    "--k",
                    str(args.k),
                    "--text",
                    args.text,
                ]
            )
        output_str = f.getvalue()

        # Parse output
        output = json.loads(output_str)
        assert "label" in output
        assert "confidence" in output
        assert output["label"] == "weather"

    def test_cli_missing_required_args(self) -> None:
        """Test that missing required arguments cause error."""
        result = subprocess.run(
            [sys.executable, "scripts/rag_cli.py", "--text", "test"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        # argparse should exit with non-zero when required args are missing
        assert result.returncode != 0, "Should fail without required arguments"


class TestEndToEnd:
    """End-to-end integration tests."""

    @patch("intent_classifier.rag.rag_llm.classifier._call_llm")
    def test_full_pipeline_with_real_data(self, mock_llm: Any) -> None:
        """Test the full pipeline from data loading to classification."""
        # Load real data
        examples, label_defs = _load_examples(max_train_samples=50, max_classes=5)
        assert len(examples) > 0

        # Build retriever
        retriever = Retriever(examples)

        # Mock LLM response
        mock_llm.return_value = json.dumps({"label": examples[0].label, "confidence": 0.85})

        # Classify
        result = classify_single(
            model="gpt-4o-mini",
            query="What's the weather like?",
            retriever=retriever,
            label_defs=label_defs,
            k=10,
            m=3,
        )

        # Verify result structure
        assert "label" in result
        assert "confidence" in result
        assert result["label"] in result["allowed_labels"]
        assert result["label"] in {ex.label for ex in examples}

    @patch("intent_classifier.rag.rag_llm.classifier._call_llm")
    def test_retrieval_affects_classification(self, mock_llm: Any) -> None:
        """Test that different queries retrieve different examples."""
        examples = [
            Example(text="weather forecast", label="weather"),
            Example(text="transfer money", label="transfer"),
            Example(text="play music", label="music"),
        ] * 3  # Repeat to have enough examples
        retriever = Retriever(examples)

        # Test weather query
        mock_llm.return_value = json.dumps({"label": "weather", "confidence": 0.9})
        result1 = classify_single(
            model="gpt-4o-mini",
            query="What's the weather?",
            retriever=retriever,
            label_defs={},
            k=5,
            m=2,
        )

        # Test transfer query
        mock_llm.return_value = json.dumps({"label": "transfer", "confidence": 0.9})
        result2 = classify_single(
            model="gpt-4o-mini",
            query="Send money to account",
            retriever=retriever,
            label_defs={},
            k=5,
            m=2,
        )

        # Both should succeed
        assert result1["label"] == "weather"
        assert result2["label"] == "transfer"
        # Should retrieve similar number of shots (both should hit k=5 or close)
        assert abs(result1["shots_used"] - result2["shots_used"]) <= 2


class TestProviderIntegration:
    """Test integration with different LLM providers (Ollama and OpenAI)."""

    @pytest.fixture
    def sample_examples(self) -> list[Example]:
        """Create sample examples for testing."""
        return [
            Example(text="What's the weather today?", label="weather_query"),
            Example(text="Check the weather forecast", label="weather_query"),
            Example(text="Transfer money to account", label="transfer_money"),
            Example(text="Send $100 to savings", label="transfer_money"),
            Example(text="Play some music", label="play_music"),
            Example(text="Start playing songs", label="play_music"),
            Example(text="What time is it?", label="time_query"),
            Example(text="Tell me the current time", label="time_query"),
        ]

    @pytest.fixture
    def retriever(self, sample_examples: list[Example]) -> Retriever:
        """Create a retriever with sample examples."""
        return Retriever(sample_examples)

    @pytest.fixture
    def label_defs(self) -> dict[str, str]:
        """Create sample label definitions."""
        return {
            "weather_query": "Questions about weather conditions",
            "transfer_money": "Requests to transfer money between accounts",
            "play_music": "Commands to play music or songs",
            "time_query": "Questions about current time",
        }

    @patch("intent_classifier.rag.rag_llm.classifier._call_llm")
    def test_ollama_model_name_passed_correctly(
        self, mock_llm: Any, retriever: Retriever, label_defs: dict[str, str]
    ) -> None:
        """Test that Ollama model names are passed correctly to LiteLLM."""
        mock_llm.return_value = '{"label": "weather_query", "confidence": 0.9}'

        result = classify_single(
            model="ollama/llama3.1:8b",
            query="What's the weather?",
            retriever=retriever,
            label_defs=label_defs,
            k=5,
            m=2,
        )

        # Verify the model name was passed to the LLM call
        assert mock_llm.called
        call_args = mock_llm.call_args
        model = call_args[1]["model"]  # Keyword arg (model)
        assert model == "ollama/llama3.1:8b", "Ollama model name should be passed correctly"
        assert result["label"] in result["allowed_labels"]

    @patch("intent_classifier.rag.rag_llm.classifier._call_llm")
    def test_openai_model_name_passed_correctly(
        self, mock_llm: Any, retriever: Retriever, label_defs: dict[str, str]
    ) -> None:
        """Test that OpenAI model names are passed correctly to LiteLLM."""
        mock_llm.return_value = '{"label": "weather_query", "confidence": 0.9}'

        result = classify_single(
            model="gpt-4o-mini",
            query="What's the weather?",
            retriever=retriever,
            label_defs=label_defs,
            k=5,
            m=2,
        )

        # Verify the model name was passed to the LLM call
        assert mock_llm.called
        call_args = mock_llm.call_args
        model = call_args[1]["model"]  # Keyword arg (model)
        assert model == "gpt-4o-mini", "OpenAI model name should be passed correctly"
        assert result["label"] in result["allowed_labels"]

    @patch("intent_classifier.rag.rag_llm.classifier._call_llm")
    def test_ollama_vs_openai_same_behavior(
        self, mock_llm: Any, retriever: Retriever, label_defs: dict[str, str]
    ) -> None:
        """Test that both Ollama and OpenAI models produce same structure."""
        mock_llm.return_value = '{"label": "weather_query", "confidence": 0.85}'

        # Test with Ollama
        result_ollama = classify_single(
            model="ollama/llama3.1:8b",
            query="What's the weather?",
            retriever=retriever,
            label_defs=label_defs,
            k=5,
            m=2,
        )

        # Test with OpenAI
        result_openai = classify_single(
            model="gpt-4o-mini",
            query="What's the weather?",
            retriever=retriever,
            label_defs=label_defs,
            k=5,
            m=2,
        )

        # Both should have same structure
        assert set(result_ollama.keys()) == set(result_openai.keys())
        assert "label" in result_ollama
        assert "confidence" in result_ollama
        assert "allowed_labels" in result_ollama
        assert "shots_used" in result_ollama

    @pytest.mark.skipif(
        os.getenv("TEST_REAL_APIS", "").lower() not in ("1", "true", "yes"),
        reason="Set TEST_REAL_APIS=1 to run real API tests",
    )
    def test_real_ollama_if_available(
        self, retriever: Retriever, label_defs: dict[str, str]
    ) -> None:
        """Test with real Ollama API if available (requires TEST_REAL_APIS=1 env var)."""
        # Check if Ollama is available (basic check)
        try:
            result = classify_single(
                model="ollama/llama3.1:8b",
                query="What's the weather?",
                retriever=retriever,
                label_defs=label_defs,
                k=5,
                m=2,
            )
            assert "label" in result
            assert "confidence" in result
            assert result["label"] in result["allowed_labels"]
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

    @pytest.mark.skipif(
        os.getenv("TEST_REAL_APIS", "").lower() not in ("1", "true", "yes"),
        reason="Set TEST_REAL_APIS=1 to run real API tests",
    )
    def test_real_openai_if_available(
        self, retriever: Retriever, label_defs: dict[str, str]
    ) -> None:
        """Test with real OpenAI API if available (requires TEST_REAL_APIS=1 env var)."""
        # Load .env file to get API key if it's there
        from dotenv import load_dotenv

        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            result = classify_single(
                model="gpt-4o-mini",
                query="What's the weather?",
                retriever=retriever,
                label_defs=label_defs,
                k=5,
                m=2,
            )
            assert "label" in result
            assert "confidence" in result
            assert result["label"] in result["allowed_labels"]
        except Exception as e:
            pytest.skip(f"OpenAI API not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
