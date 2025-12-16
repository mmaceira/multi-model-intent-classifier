"""Tests for the finalize_run output schema and metadata package."""

from __future__ import annotations

import json

from intent_classifier.pipeline.finalize import _slugify_model_id, finalize_run
from intent_classifier.utils.config_loader import load_config_with_metadata
from intent_classifier.utils.paths import get_repo_root


def test_slugify_model_id_basic_cases() -> None:
    assert _slugify_model_id("TF-IDF bigrams + SVM") == "tf_idf_bigrams_svm"
    assert _slugify_model_id("MiniLM + LogReg") == "minilm_logreg"
    assert _slugify_model_id(" already_slugged_id ") == "already_slugged_id"
    assert _slugify_model_id("$$$") == "model"


def test_finalize_run_writes_meta_and_manifest(tmp_path, monkeypatch) -> None:
    """Run finalize_run and assert that core meta artefacts exist."""
    repo_root = get_repo_root()

    # Ensure we are finalising an existing tiny experiment run.
    monkeypatch.setenv("DATASET", "clinc150")
    monkeypatch.setenv("VARIANT", "tiny")
    monkeypatch.delenv("CONFIG_FILE", raising=False)

    metadata = load_config_with_metadata()
    cfg = metadata["config"]
    paths_cfg = cfg["paths"]
    run_dir = repo_root / paths_cfg["run_dir"]

    # Make sure the run directory exists for the purpose of this test.
    run_dir.mkdir(parents=True, exist_ok=True)

    finalize_run()

    # Core meta package
    meta_dir = run_dir / "meta"
    assert meta_dir.exists()
    assert (meta_dir / "config_resolved.yaml").exists()
    assert (meta_dir / "config_sources.json").exists()
    assert (meta_dir / "command.txt").exists()
    assert (meta_dir / "git.json").exists()
    assert (meta_dir / "env.json").exists()
    assert (meta_dir / "timestamps.json").exists()

    # Manifest at run root
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()

    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(content, list)
    # Each entry should have path / size / sha256 keys
    if content:
        entry = content[0]
        assert "path" in entry
        assert "sha256" in entry
