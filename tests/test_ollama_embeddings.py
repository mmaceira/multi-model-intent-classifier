import json
from typing import Any

import numpy as np

from intent_classifier.utils.embeddings import LitellmOllamaEmbedder


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: Any) -> None:  # pragma: no cover - trivial
        return None


def test_litellm_ollama_embedder_encode_uses_ollama_http_api(monkeypatch) -> None:
    """LitellmOllamaEmbedder should call Ollama's /api/embeddings and return correct shape."""

    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout: float = 0) -> _FakeResponse:  # type: ignore[override]
        # Capture URL and payload for assertions
        captured["url"] = req.full_url
        body = req.data.decode("utf-8")
        payload = json.loads(body)

        # Model name should be normalised (no optional "ollama/" prefix)
        assert payload["model"] == "qwen3-embedding:latest"
        assert "prompt" in payload

        # Return a simple fixed-size embedding
        resp = {"embedding": [1.0, 2.0, 3.0]}
        return _FakeResponse(json.dumps(resp).encode("utf-8"))

    # Ensure base URL is taken from env when not passed explicitly
    monkeypatch.setenv("OLLAMA_API_BASE", "http://test-ollama:11434")

    # Import inside the test so we can patch urllib for this module's use
    import urllib.request as urllib_request

    monkeypatch.setattr(urllib_request, "urlopen", fake_urlopen)

    embedder = LitellmOllamaEmbedder(
        model="ollama/qwen3-embedding:latest",
        base_url=None,
        batch_size=2,
    )

    texts = ["hello", "world"]
    vectors = embedder.encode(texts, batch_size=2)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 3)
    assert captured["url"] == "http://test-ollama:11434/api/embeddings"
