"""Integration smoke test for end-to-end pipeline outputs and run schema."""

from __future__ import annotations

import os
import subprocess

from intent_classifier.utils.config_loader import load_config_with_metadata
from intent_classifier.utils.paths import get_repo_root


def _run_pipeline_for_tiny_experiment(env_overrides: dict[str, str]) -> None:
    """Execute the tiny experiment pipeline using the configured scripts."""
    repo_root = get_repo_root()
    env_full = os.environ.copy()
    env_full.update(env_overrides)

    # Run the pipeline entrypoint which chains sub-steps and finalize.
    subprocess.run(
        ["uv", "run", "python", "-m", "scripts.pipeline.run_all"],
        cwd=repo_root,
        check=True,
        env=env_full,
    )


def test_pipeline_tiny_run_writes_core_run_schema(tmp_path, monkeypatch) -> None:
    """Run the tiny pipeline and assert core artefacts under output/runs/<run_id>/."""
    # Select the tiny experiment via environment
    env_overrides = {
        "DATASET": "clinc150",
        "VARIANT": "tiny",
    }
    monkeypatch.delenv("CONFIG_FILE", raising=False)

    # Run the pipeline (includes finalize_run at the end).
    _run_pipeline_for_tiny_experiment(env_overrides)

    meta = load_config_with_metadata()
    cfg = meta["config"]
    paths_cfg = cfg["paths"]

    repo_root = get_repo_root()
    run_dir = repo_root / paths_cfg["run_dir"]

    # Run directory should exist
    assert run_dir.exists()

    # Meta package
    meta_dir = run_dir / "meta"
    assert (meta_dir / "config_resolved.yaml").exists()

    # Compare summary metrics (format may be CSV or Parquet; accept either)
    compare_dir = run_dir / "compare"
    assert compare_dir.exists()
    summary_files = list(compare_dir.glob("summary_metrics.*"))
    assert summary_files, "Expected summary_metrics.* in compare/"
