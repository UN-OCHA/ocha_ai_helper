"""
Pytest configuration and fixtures. Mocks model2vec and onnxruntime so tests
run without loading real embedding/visual models. When API_BASE_URL is set,
tests run against that live API instead (e.g. a running container).
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# When API_BASE_URL is set we talk to a live server and skip loading the app.
if os.environ.get("API_BASE_URL"):
    _app = None
else:
    # Import Pillow before patch.dict snapshots sys.modules. patch.dict restores
    # that snapshot on exit and would otherwise drop PIL.Image imported during
    # app load, leaving the app bound to a stale module without image plugins.
    import PIL.Image  # noqa: F401

    def _make_mock_dedup_model():
        """
        Return a StaticModel mock whose encode() yields deterministic vectors.

        Vector layout (4-dim):
          index 0 (query)     : [1, 0, 0, 0]   -- reference direction
          index 1 (candidate) : [0.9, 0.1, 0, 0] -- high similarity
          index 2 (candidate) : [0, 1, 0, 0]   -- low similarity (orthogonal)
        Any additional indices repeat the low-similarity vector.
        """
        def _encode(texts, **kwargs):
            base = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ], dtype=np.float32)
            result = []
            for i, _ in enumerate(texts):
                result.append(base[min(i, 2)])
            return np.array(result, dtype=np.float32)

        model = MagicMock()
        model.encode = MagicMock(side_effect=_encode)
        return model

    def _make_mock_visual_session():
        """ORT session mock with fixed logits favoring Map."""
        session = MagicMock()
        input_meta = MagicMock()
        input_meta.name = "input"
        session.get_inputs.return_value = [input_meta]
        session.run.return_value = [
            np.array([[5.0, 0.1, 0.0]], dtype=np.float32)
        ]
        return session

    _mock_model2vec = MagicMock()
    _mock_model2vec.StaticModel.from_pretrained = MagicMock(
        return_value=_make_mock_dedup_model()
    )
    _mock_ort = MagicMock()
    _mock_ort.SessionOptions.return_value = MagicMock()
    _mock_ort.InferenceSession.return_value = _make_mock_visual_session()

    # Point at vendored artifact so Path.exists passes; ORT itself is mocked.
    _repo_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault(
        "VISUAL_MODEL_PATH",
        str(_repo_root / "html" / "models" / "visual" / "model.onnx"),
    )

    with patch.dict(
        sys.modules,
        {
            "model2vec": _mock_model2vec,
            "onnxruntime": _mock_ort,
        },
    ):
        from html.app import app as _app  # noqa: E402


@pytest.fixture(scope="session")
def app():
    """FastAPI app (loaded with mocked models), or None when using live API."""
    return _app


@pytest.fixture
def client(app):
    """HTTP client: live httpx client when API_BASE_URL is set, else TestClient(app)."""
    if os.environ.get("API_BASE_URL"):
        import httpx

        base_url = os.environ["API_BASE_URL"].rstrip("/")
        client_instance = httpx.Client(base_url=base_url, timeout=120.0)
        yield client_instance
        client_instance.close()
    else:
        from fastapi.testclient import TestClient

        yield TestClient(app)
