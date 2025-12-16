"""Unit tests for configuration loading, merging, and path computation."""

from __future__ import annotations

from intent_classifier.utils.config_loader import (
    ConfigMetadata,
    _attach_providers_and_resolved,
    _deep_merge_dicts,
    _ensure_general_and_paths,
    detect_label_type,
    load_config_with_metadata,
)
from intent_classifier.utils.paths import compute_paths, get_repo_root


def test_deep_merge_dicts_recurses_and_overrides() -> None:
    base = {
        "section": {
            "a": 1,
            "nested": {"x": 1, "y": 2},
            "list_value": [1, 2],
        },
        "keep": "base",
    }
    override = {
        "section": {
            "b": 2,
            "nested": {"y": 999, "z": 3},
            "list_value": ["replaced"],
        },
        "keep": "override",
    }

    merged = _deep_merge_dicts(base, override)

    # Shallow keys
    assert merged["keep"] == "override"

    # Dicts are merged recursively
    assert merged["section"]["a"] == 1
    assert merged["section"]["b"] == 2
    assert merged["section"]["nested"] == {"x": 1, "y": 999, "z": 3}

    # Lists are replaced, not concatenated
    assert merged["section"]["list_value"] == ["replaced"]


def test_detect_label_type_uses_dataset_multilabel_flag() -> None:
    cfg_multi = {"dataset": {"multilabel": True}}
    cfg_single = {"dataset": {"multilabel": False}}

    assert detect_label_type(cfg_multi) == "multilabel"
    assert detect_label_type(cfg_single) == "singlelabel"


def test_compute_paths_uses_run_id_and_root() -> None:
    run_id = "singlelabel/clinc150/tiny"
    paths = compute_paths(run_id, root="output/runs")

    assert paths["run_dir"] == f"output/runs/{run_id}"
    assert paths["meta_dir"] == f"output/runs/{run_id}/meta"
    assert paths["dataset_dir"] == f"output/runs/{run_id}/dataset"
    assert paths["features_dir"] == f"output/runs/{run_id}/features"
    assert paths["models_dir"] == f"output/runs/{run_id}/models"
    assert paths["eval_dir"] == f"output/runs/{run_id}/eval"
    assert paths["compare_dir"] == f"output/runs/{run_id}/compare"
    assert paths["llm_logs_dir"] == f"output/runs/{run_id}/llm_logs"
    assert paths["figures_dir"] == f"output/runs/{run_id}/compare/figures"


def test_ensure_general_and_paths_derives_run_id_and_paths() -> None:
    config: dict[str, object] = {}
    dataset_name = "clinc150"
    config_name = "tiny"
    label_type = "singlelabel"

    _ensure_general_and_paths(
        config=config,
        dataset_name=dataset_name,
        config_name=config_name,
        label_type=label_type,
    )

    general = config.get("general", {})
    assert isinstance(general, dict)
    assert general["run_name"] == config_name
    assert general["run_id"] == f"{label_type}/{dataset_name}/{config_name}"

    paths = config.get("paths", {})
    assert isinstance(paths, dict)
    assert paths["run_dir"].endswith(f"{label_type}/{dataset_name}/{config_name}")


def test_attach_providers_and_resolved_uses_providers_defaults(monkeypatch) -> None:
    """Ensure resolved.embedding_model picks up provider defaults when requested.

    This test intentionally leaves ``llm_backend`` as ``None`` to document
    the new behaviour after the config refactor:

    - ``resolved.embedding_model`` should still come from the provider
      defaults when ``embedding_backend == "openai"``.
    - ``resolved.llm_model`` is allowed to be ``None`` unless an explicit
      ``llm_backend`` is configured (e.g. ``\"openai\"`` or ``\"ollama\"``).
    """
    # Isolate from real environment to keep the test deterministic.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    repo_root = get_repo_root()
    providers_path = repo_root / "config" / "base" / "providers.yaml"
    assert providers_path.exists(), "providers.yaml must exist for this test"

    # Start from a minimal config that only sets backends.
    config: dict[str, object] = {
        "model": {
            "embedding_backend": "openai",
            "llm_backend": None,
        }
    }

    _attach_providers_and_resolved(config)

    resolved = config.get("resolved", {})
    assert isinstance(resolved, dict)

    providers = config.get("providers", {})
    assert isinstance(providers, dict)

    # Embedding model should come from providers.openai.embed_default
    openai_provider = providers.get("openai", {})
    assert isinstance(openai_provider, dict)
    assert resolved["embedding_backend"] == "openai"
    assert resolved["embedding_model"] == openai_provider.get("embed_default")

    # New behaviour: with no explicit llm_backend configured, we do not
    # force an LLM model. Downstream code should gate LLM usage on
    # resolved.llm_backend / resolved.llm_model.
    assert resolved.get("llm_backend") is None
    assert resolved.get("llm_model") is None


def test_attach_providers_and_resolved_sets_llm_defaults_for_backends(
    monkeypatch,
) -> None:
    """Ensure resolved.llm_model uses provider defaults when llm_backend is set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    repo_root = get_repo_root()
    providers_path = repo_root / "config" / "base" / "providers.yaml"
    assert providers_path.exists(), "providers.yaml must exist for this test"

    # --- OpenAI backend ---
    config_openai: dict[str, object] = {
        "model": {
            "embedding_backend": "sbert",
            "sbert_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_backend": "openai",
        }
    }

    _attach_providers_and_resolved(config_openai)
    resolved_openai = config_openai.get("resolved", {})
    providers_openai = config_openai.get("providers", {})
    assert isinstance(resolved_openai, dict)
    assert isinstance(providers_openai, dict)

    openai_provider = providers_openai.get("openai", {})
    assert isinstance(openai_provider, dict)
    assert resolved_openai["llm_backend"] == "openai"
    assert resolved_openai["llm_model"] == openai_provider.get("llm_default")

    # --- Ollama backend ---
    config_ollama: dict[str, object] = {
        "model": {
            "embedding_backend": "sbert",
            "sbert_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_backend": "ollama",
        }
    }

    _attach_providers_and_resolved(config_ollama)
    resolved_ollama = config_ollama.get("resolved", {})
    providers_ollama = config_ollama.get("providers", {})
    assert isinstance(resolved_ollama, dict)
    assert isinstance(providers_ollama, dict)

    ollama_provider = providers_ollama.get("ollama", {})
    assert isinstance(ollama_provider, dict)
    assert resolved_ollama["llm_backend"] == "ollama"
    assert resolved_ollama["llm_model"] == ollama_provider.get("llm_default")


def test_load_config_with_metadata_attaches_resolved_and_paths(monkeypatch) -> None:
    """Smoke test end-to-end config loading for a tiny experiment."""
    # Use the layered experiment config for clinc150 tiny.
    monkeypatch.setenv("DATASET", "clinc150")
    monkeypatch.setenv("VARIANT", "tiny")
    monkeypatch.delenv("CONFIG_FILE", raising=False)

    meta: ConfigMetadata = load_config_with_metadata()

    assert meta["dataset_name"] == "clinc150"
    assert meta["config_name"] == "tiny"
    assert meta["label_type"] in {"singlelabel", "multilabel"}

    cfg = meta["config"]
    resolved = cfg.get("resolved", {})
    paths = cfg.get("paths", {})

    assert isinstance(resolved, dict)
    assert isinstance(paths, dict)

    # Resolved section should contain run_id, label_type, and a snapshot of paths.
    assert resolved.get("run_id") == cfg.get("general", {}).get("run_id")
    assert resolved.get("label_type") == meta["label_type"]
    assert isinstance(resolved.get("paths"), dict)


def test_load_config_with_metadata_resolved_metadata_consistency(monkeypatch) -> None:
    """Assert that resolved.paths, resolved.run_id, and resolved.label_type are consistent.

    This test ensures that the resolved metadata snapshot is properly populated
    and matches the computed paths and general metadata for a real experiment config.
    """
    monkeypatch.setenv("DATASET", "clinc150")
    monkeypatch.setenv("VARIANT", "tiny")
    monkeypatch.delenv("CONFIG_FILE", raising=False)

    meta: ConfigMetadata = load_config_with_metadata()
    cfg = meta["config"]

    general = cfg.get("general", {})
    assert isinstance(general, dict)

    paths = cfg.get("paths", {})
    assert isinstance(paths, dict)

    resolved = cfg.get("resolved", {})
    assert isinstance(resolved, dict)

    # Resolved.run_id should match general.run_id
    expected_run_id = general.get("run_id")
    assert expected_run_id is not None
    assert resolved.get("run_id") == expected_run_id

    # Resolved.label_type should match metadata.label_type
    assert resolved.get("label_type") == meta["label_type"]

    # Resolved.paths should be a snapshot of the computed paths
    resolved_paths = resolved.get("paths")
    assert isinstance(resolved_paths, dict)

    # Key paths should match between resolved.paths and config.paths
    for key in ["run_dir", "meta_dir", "dataset_dir", "features_dir", "models_dir"]:
        assert key in resolved_paths
        assert key in paths
        assert resolved_paths[key] == paths[key]

    # Run ID should be embedded in the path structure
    assert expected_run_id in str(resolved_paths["run_dir"])


def test_attach_providers_and_resolved_respects_ollama_endpoint_env_override(
    monkeypatch,
) -> None:
    """Ensure that environment variables override static Ollama endpoint config.

    This test verifies that MODEL_OLLAMA_ENDPOINT, OLLAMA_API_BASE, and
    OLLAMA_HOST environment variables take precedence over the static endpoint
    in providers.yaml, and that resolved.ollama_endpoint reflects the override.
    """
    # Clear any existing Ollama env vars first
    monkeypatch.delenv("MODEL_OLLAMA_ENDPOINT", raising=False)
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


def test_attach_providers_and_resolved_emits_warnings_for_legacy_keys(caplog, monkeypatch) -> None:
    """Ensure legacy model.* override keys still work but emit deprecation warnings."""
    caplog.set_level("WARNING")

    # Legacy embedding override for OpenAI.
    cfg_openai_legacy: dict[str, object] = {
        "model": {
            "embedding_backend": "openai",
            "openai_model_name": "legacy-embed-model",
        }
    }
    _attach_providers_and_resolved(cfg_openai_legacy)
    resolved_openai = cfg_openai_legacy.get("resolved", {})
    assert isinstance(resolved_openai, dict)
    assert resolved_openai["embedding_model"] == "legacy-embed-model"
    assert any("model.openai_model_name" in rec.getMessage() for rec in caplog.records)

    caplog.clear()

    # Legacy Ollama embedding override.
    cfg_ollama_legacy: dict[str, object] = {
        "model": {
            "embedding_backend": "ollama",
            "ollama_embedding_model_name": "legacy-ollama-embed",
        }
    }
    _attach_providers_and_resolved(cfg_ollama_legacy)
    resolved_ollama = cfg_ollama_legacy.get("resolved", {})
    assert isinstance(resolved_ollama, dict)
    assert resolved_ollama["embedding_model"] == "legacy-ollama-embed"
    assert any("model.ollama_embedding_model_name" in rec.getMessage() for rec in caplog.records)

    caplog.clear()

    # Legacy LLM model override when no backend is specified.
    cfg_llm_legacy: dict[str, object] = {
        "model": {
            "embedding_backend": "sbert",
            "sbert_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "legacy-llm-model",
        }
    }
    _attach_providers_and_resolved(cfg_llm_legacy)
    resolved_llm = cfg_llm_legacy.get("resolved", {})
    assert isinstance(resolved_llm, dict)
    assert resolved_llm["llm_model"] == "legacy-llm-model"
    assert any("model.llm_model" in rec.getMessage() for rec in caplog.records)

    repo_root = get_repo_root()
    providers_path = repo_root / "config" / "base" / "providers.yaml"
    assert providers_path.exists(), "providers.yaml must exist for this test"

    # Test MODEL_OLLAMA_ENDPOINT (highest priority)
    custom_endpoint = "http://custom-ollama-server:11434"
    monkeypatch.setenv("MODEL_OLLAMA_ENDPOINT", custom_endpoint)

    config: dict[str, object] = {
        "model": {
            "embedding_backend": "sbert",
            "sbert_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }
    }

    _attach_providers_and_resolved(config)

    resolved = config.get("resolved", {})
    assert isinstance(resolved, dict)
    assert resolved.get("ollama_endpoint") == custom_endpoint

    providers = config.get("providers", {})
    assert isinstance(providers, dict)
    ollama_provider = providers.get("ollama", {})
    assert isinstance(ollama_provider, dict)
    assert ollama_provider.get("endpoint") == custom_endpoint

    # Test OLLAMA_API_BASE (second priority)
    monkeypatch.delenv("MODEL_OLLAMA_ENDPOINT", raising=False)
    base_endpoint = "http://base-ollama:11434"
    monkeypatch.setenv("OLLAMA_API_BASE", base_endpoint)

    config2: dict[str, object] = {
        "model": {
            "embedding_backend": "sbert",
            "sbert_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }
    }

    _attach_providers_and_resolved(config2)

    resolved2 = config2.get("resolved", {})
    assert isinstance(resolved2, dict)
    assert resolved2.get("ollama_endpoint") == base_endpoint

    # Test OLLAMA_HOST (third priority)
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    host_endpoint = "http://host-ollama:11434"
    monkeypatch.setenv("OLLAMA_HOST", host_endpoint)

    config3: dict[str, object] = {
        "model": {
            "embedding_backend": "sbert",
            "sbert_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }
    }

    _attach_providers_and_resolved(config3)

    resolved3 = config3.get("resolved", {})
    assert isinstance(resolved3, dict)
    assert resolved3.get("ollama_endpoint") == host_endpoint

    # Clean up
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
