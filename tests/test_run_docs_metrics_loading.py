"""Tests for metrics loading in run_docs generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from intent_classifier.evaluation.run_docs import generate_run_readme_and_model_cards
from intent_classifier.utils.slugify import slugify_model_id


def test_run_docs_loads_metrics_from_eval_dir(tmp_path: Path, monkeypatch) -> None:
    """Test that generate_run_readme_and_model_cards loads metrics from
    eval/<model_id>/test/test_metrics.json."""
    # Setup mock run directory structure
    run_dir = tmp_path / "output" / "runs" / "test_run"
    eval_dir = run_dir / "eval"
    compare_dir = run_dir / "compare"
    models_dir = run_dir / "models"

    eval_dir.mkdir(parents=True)
    compare_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    # Create a model with metrics in eval directory
    model_display_name = "Linear SVM"
    model_id = slugify_model_id(model_display_name)
    model_eval_dir = eval_dir / model_id / "test"
    model_eval_dir.mkdir(parents=True)

    # Write test metrics
    test_metrics = {
        "accuracy": 0.95,
        "macro_f1": 0.94,
        "micro_f1": 0.95,
        "weighted_f1": 0.94,
    }
    (model_eval_dir / "test_metrics.json").write_text(json.dumps(test_metrics), encoding="utf-8")

    # Create summary metrics CSV
    summary_df = pd.DataFrame(
        {
            "test_accuracy": [0.95],
            "test_macro_f1": [0.94],
        },
        index=[model_display_name],
    )
    summary_df.to_csv(compare_dir / "summary_metrics.csv")

    # Mock config loader
    def mock_load_config():
        return {
            "config": {
                "paths": {
                    "run_dir": str(run_dir),
                    "eval_dir": str(eval_dir),
                    "compare_dir": str(compare_dir),
                    "models_dir": str(models_dir),
                    "dataset_dir": str(run_dir / "dataset"),
                    "features_dir": str(run_dir / "features"),
                },
                "model": {},
                "resolved": {},
            },
            "dataset_name": "test_dataset",
            "config_name": "test",
            "label_type": "singlelabel",
            "config_path": "config/test.yaml",
        }

    monkeypatch.setattr(
        "intent_classifier.evaluation.run_docs.load_config_with_metadata", mock_load_config
    )

    # Generate docs with empty results - should load from files
    generate_run_readme_and_model_cards(results={model_display_name: {}})

    # Verify model card was created
    model_card = models_dir / model_id / "model_card.md"
    assert model_card.exists()

    # Verify metrics are in the card
    card_content = model_card.read_text(encoding="utf-8")
    assert "0.950" in card_content or "0.95" in card_content  # Accuracy should be present


def test_run_docs_fallback_to_summary_metrics(tmp_path: Path, monkeypatch) -> None:
    """Test that run_docs falls back to summary_metrics.csv when eval metrics are missing."""
    run_dir = tmp_path / "output" / "runs" / "test_run"
    eval_dir = run_dir / "eval"
    compare_dir = run_dir / "compare"
    models_dir = run_dir / "models"

    eval_dir.mkdir(parents=True)
    compare_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    model_display_name = "Naive Bayes"
    model_id = slugify_model_id(model_display_name)

    # Create summary metrics CSV only (no eval metrics)
    summary_df = pd.DataFrame(
        {
            "test_accuracy": [0.88],
            "test_macro_f1": [0.87],
            "test_micro_f1": [0.88],
            "test_weighted_f1": [0.87],
        },
        index=[model_display_name],
    )
    summary_df.to_csv(compare_dir / "summary_metrics.csv")

    # Mock config loader
    def mock_load_config():
        return {
            "config": {
                "paths": {
                    "run_dir": str(run_dir),
                    "eval_dir": str(eval_dir),
                    "compare_dir": str(compare_dir),
                    "models_dir": str(models_dir),
                    "dataset_dir": str(run_dir / "dataset"),
                    "features_dir": str(run_dir / "features"),
                },
                "model": {},
                "resolved": {},
            },
            "dataset_name": "test_dataset",
            "config_name": "test",
            "label_type": "singlelabel",
            "config_path": "config/test.yaml",
        }

    monkeypatch.setattr(
        "intent_classifier.evaluation.run_docs.load_config_with_metadata", mock_load_config
    )

    # Generate docs with empty results
    generate_run_readme_and_model_cards(results={model_display_name: {}})

    # Verify model card was created
    model_card = models_dir / model_id / "model_card.md"
    assert model_card.exists()

    # Verify metrics from summary are in the card
    card_content = model_card.read_text(encoding="utf-8")
    assert "0.880" in card_content or "0.88" in card_content
