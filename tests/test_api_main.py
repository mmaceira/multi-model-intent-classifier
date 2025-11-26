"""Tests for the FastAPI service in `scripts.api.main_api`.

If FastAPI is not installed (i.e., API extras not installed), this module is skipped.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

try:
    from fastapi.testclient import TestClient  # type: ignore
except ImportError:  # pragma: no cover
    pytest.skip("fastapi not installed; skipping API tests", allow_module_level=True)

from scripts.api.main_api import MODELS_INFO, app


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """Create a TestClient with a stubbed `_get_model` to avoid loading real models."""

    from scripts import api as api_pkg  # noqa: F401
    from scripts.api import main_api as main_api_module

    class DummyModel:
        def __init__(self) -> None:
            self.classes_ = ["a", "b"]

        def predict(self, texts: List[str]) -> List[str]:
            return ["a" for _ in texts]

        def predict_proba(self, texts: List[str]):
            import numpy as np

            return np.array([[0.9, 0.1] for _ in texts])

    def fake_get_model(model_identifier: str) -> Any:
        return DummyModel()

    monkeypatch.setattr(main_api_module, "_get_model", fake_get_model)

    # Ensure API_KEY is empty for most tests (no auth)
    monkeypatch.setattr(main_api_module, "API_KEY", None)

    return TestClient(app)


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_ready_endpoint(client: TestClient):
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert "ready" in data
    assert "models_loaded" in data


def test_predict_without_auth(client: TestClient):
    # With API_KEY disabled, prediction should work without headers
    model_id = next(iter(MODELS_INFO.keys()))
    payload: Dict[str, Any] = {"model_id": model_id, "text": "some example text"}
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in ["a", "b", "__ABSTAIN__"]
    assert "confidence" in data
    assert "request_id" in data


def test_predict_with_auth_required(monkeypatch):
    """When API_KEY is set, requests without key should be rejected."""
    from scripts.api import main_api as main_api_module

    # Stub model loader to avoid hitting real models
    class DummyModel:
        def __init__(self) -> None:
            self.classes_ = ["a", "b"]

        def predict(self, texts: List[str]) -> List[str]:
            return ["a" for _ in texts]

        def predict_proba(self, texts: List[str]):
            import numpy as np

            return np.array([[0.9, 0.1] for _ in texts])

    def fake_get_model(model_identifier: str) -> Any:
        return DummyModel()

    monkeypatch.setattr(main_api_module, "_get_model", fake_get_model)

    # Force API_KEY and re-create client to ensure middleware sees it
    monkeypatch.setattr(main_api_module, "API_KEY", "secret-key")
    test_client = TestClient(app)

    model_id = next(iter(MODELS_INFO.keys()))
    payload = {"model_id": model_id, "text": "some example text"}

    # Missing key → 401
    resp = test_client.post("/v1/predict", json=payload)
    assert resp.status_code == 401

    # Correct key via X-API-Key → 200
    resp = test_client.post(
        "/v1/predict",
        json=payload,
        headers={"X-API-Key": "secret-key"},
    )
    assert resp.status_code == 200
