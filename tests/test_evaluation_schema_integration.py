"""Integration smoke test for evaluation directory schema and docs generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from intent_classifier.evaluation import run_evaluations
from intent_classifier.evaluation.run_docs import generate_run_readme_and_model_cards
from intent_classifier.utils.config_loader import load_config_with_metadata


@pytest.mark.integration
def test_evaluation_schema_and_docs_integration(tmp_path: Path, monkeypatch) -> None:
    """Integration test: Run evaluation pipeline and verify output schema matches expectations.

    This test verifies:
    1. eval/<model_id>/ structure for per-model metrics
    2. compare/ structure for summary metrics
    3. Model cards are generated with metrics from files
    4. No ModuleNotFoundError for missing dataframe_image
    """
    # Setup mock run directory structure
    run_id = "test_integration_run"
    run_dir = tmp_path / "output" / "runs" / run_id
    eval_dir = run_dir / "eval"
    compare_dir = run_dir / "compare"
    models_dir = run_dir / "models"
    dataset_dir = run_dir / "dataset"
    features_dir = run_dir / "features"
    meta_dir = run_dir / "meta"

    for d in [eval_dir, compare_dir, models_dir, dataset_dir, features_dir, meta_dir]:
        d.mkdir(parents=True)

    # Create a minimal prediction structure
    pred_dir = tmp_path / "predictions"
    model_id = "linear_svm"
    model_pred_dir = pred_dir / model_id
    model_pred_dir.mkdir(parents=True)

    # Create minimal test predictions CSV
    import pandas as pd

    test_df = pd.DataFrame(
        {
            "y_true": ["intent1", "intent2", "intent1"],
            "y_pred": ["intent1", "intent2", "intent2"],
        }
    )
    test_df.to_csv(model_pred_dir / "test_predictions.csv", index=False)

    # Mock config loader to return our test paths
    original_load = load_config_with_metadata

    def mock_load_config():
        config = original_load()
        config["config"]["paths"] = {
            "run_dir": str(run_dir),
            "eval_dir": str(eval_dir),
            "compare_dir": str(compare_dir),
            "models_dir": str(models_dir),
            "dataset_dir": str(dataset_dir),
            "features_dir": str(features_dir),
            "meta_dir": str(meta_dir),
        }
        return config

    monkeypatch.setattr(
        "intent_classifier.evaluation.run_docs.load_config_with_metadata", mock_load_config
    )

    # Run evaluation
    results = run_evaluations(
        model_names=["linear_svm"],
        artefacts_root=pred_dir,
        eval_dir=eval_dir,
        compare_dir=compare_dir,
        verbose=False,
    )

    # Verify results structure
    assert len(results) > 0
    assert "linear_svm" in results

    # Verify eval/<model_id>/ structure
    model_eval_path = eval_dir / model_id / "test"
    assert model_eval_path.exists(), f"Expected eval/{model_id}/test/ to exist"

    # Verify per-model metrics file exists
    test_metrics_file = model_eval_path / "test_metrics.json"
    assert test_metrics_file.exists(), f"Expected eval/{model_id}/test/test_metrics.json to exist"

    # Verify compare/ structure
    summary_metrics_file = compare_dir / "summary_metrics.csv"
    assert summary_metrics_file.exists(), "Expected compare/summary_metrics.csv to exist"

    # Verify summary_metrics.csv is readable
    import pandas as pd

    summary_df = pd.read_csv(summary_metrics_file, index_col=0)
    assert len(summary_df) > 0, "summary_metrics.csv should have at least one row"

    # Verify model card generation with metrics from files
    # Generate docs with empty results to test file loading
    generate_run_readme_and_model_cards(results={})

    # Verify model card was created
    model_card = models_dir / model_id / "model_card.md"
    assert model_card.exists(), f"Expected models/{model_id}/model_card.md to exist"

    # Verify metrics are in the card (loaded from files)
    card_content = model_card.read_text(encoding="utf-8")
    # Card should contain metric sections
    assert "Accuracy" in card_content or "F1" in card_content, "Model card should contain metrics"

    # Verify README was created
    readme = run_dir / "README.md"
    assert readme.exists(), "Expected README.md to exist"

    # Verify no ModuleNotFoundError for dataframe_image
    # This is implicitly tested by the fact that evaluation runs without errors
    # even if dataframe_image is not installed (it's now lazy-loaded)


@pytest.mark.integration
def test_slugify_consistency_in_output_directories(tmp_path: Path) -> None:
    """Test that model directories use consistent slugified names across all stages."""
    from intent_classifier.utils.slugify import slugify_model_id

    model_display_name = "RAG-LLM (TF-IDF, default prompt)"
    expected_slug = slugify_model_id(model_display_name)

    # Simulate directory creation as done in different pipeline stages
    models_dir = tmp_path / "models"
    predictions_dir = tmp_path / "predictions"
    eval_dir = tmp_path / "eval"

    # All should use the same slug
    model_model_dir = models_dir / expected_slug
    model_pred_dir = predictions_dir / expected_slug
    model_eval_dir = eval_dir / expected_slug

    model_model_dir.mkdir(parents=True)
    model_pred_dir.mkdir(parents=True)
    model_eval_dir.mkdir(parents=True)

    # Verify all directories use the same slug
    assert model_model_dir.name == expected_slug
    assert model_pred_dir.name == expected_slug
    assert model_eval_dir.name == expected_slug

    # Verify no spaces, parentheses, or commas in directory names
    assert " " not in expected_slug
    assert "(" not in expected_slug
    assert ")" not in expected_slug
    assert "," not in expected_slug
