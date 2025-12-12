"""Tests for the FastAPI service in `scripts.api.main_api`.

If FastAPI is not installed (i.e., API extras not installed), this module is skipped.
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("fastapi not installed; skipping API tests", allow_module_level=True)

from scripts.api.main_api import MODELS_INFO, app


@pytest.fixture
def client(monkeypatch: Any) -> TestClient:
    """Create a TestClient with a stubbed `_get_model` to avoid loading real models."""

    from scripts import api as api_pkg  # noqa: F401
    from scripts.api import main_api as main_api_module

    class DummyModel:  # pylint: disable=missing-class-docstring
        def __init__(self) -> None:
            self.classes_ = ["a", "b"]

        def predict(self, texts: list[str]) -> list[str]:
            return ["a" for _ in texts]

        def predict_proba(self, texts: list[str]) -> Any:
            import numpy as np

            return np.array([[0.9, 0.1] for _ in texts])

    def fake_get_model(model_identifier: str) -> Any:
        return DummyModel()

    monkeypatch.setattr(main_api_module, "_get_model", fake_get_model)

    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_ready_endpoint(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert "ready" in data
    assert "models_loaded" in data


def test_predict(client: TestClient) -> None:
    """Test prediction endpoint works without authentication."""
    model_id = next(iter(MODELS_INFO.keys()))
    payload: dict[str, Any] = {"model_id": model_id, "text": "some example text"}
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in ["a", "b", "__ABSTAIN__"]
    assert "confidence" in data
    assert "request_id" in data
