"""
Pytest configuration and fixtures. Mocks spacy and flashrank so tests run
without loading real language models. When API_BASE_URL is set, tests run
against that live API instead (e.g. a running container).
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# When API_BASE_URL is set we talk to a live server and skip loading the app.
if os.environ.get("API_BASE_URL"):
    _app = None
else:
    import httpx
    from fastapi.testclient import TestClient

    def _make_mock_doc(sents_texts=None, token_texts=None):
        """Build a mock spacy-like document with .sents and token iteration."""
        doc = MagicMock()
        if sents_texts is None:
            sents_texts = ["First sentence.", "Second sentence.", "Third sentence."]
        doc.sents = [MagicMock(text=s) for s in sents_texts]
        if token_texts is None:
            token_texts = ["hello", "world", ",", "this", "is", "a", "wide", "test"]
        tokens = []
        for t in token_texts:
            tok = MagicMock()
            tok.text = t
            tok.is_space = False
            tok.is_stop = False
            tok.is_punct = False
            tokens.append(tok)
        doc.__iter__ = lambda self: iter(tokens)
        return doc

    def _make_mock_nlp():
        return MagicMock(side_effect=lambda text: _make_mock_doc())

    def _make_mock_ranker():
        ranker = MagicMock()
        ranker.rerank = MagicMock(
            return_value=[
                {"text": "most relevant", "score": 0.95},
                {"text": "second", "score": 0.8},
            ]
        )
        return ranker

    _mock_spacy = MagicMock()
    _mock_spacy.load = MagicMock(return_value=_make_mock_nlp())
    _mock_flashrank = MagicMock()
    _mock_flashrank.Ranker = MagicMock(return_value=_make_mock_ranker())
    _mock_flashrank.RerankRequest = MagicMock()

    with patch.dict(
        sys.modules,
        {"spacy": _mock_spacy, "flashrank": _mock_flashrank},
    ):
        from html.app import app as _app  # noqa: E402


@pytest.fixture(scope="session")
def app():
    """FastAPI app (loaded with mocked spacy and flashrank), or None when using live API."""
    return _app


@pytest.fixture
def client(app):
    """HTTP client: live httpx client when API_BASE_URL is set, else TestClient(app)."""
    if os.environ.get("API_BASE_URL"):
        import httpx

        base_url = os.environ["API_BASE_URL"].rstrip("/")
        client_instance = httpx.Client(base_url=base_url, timeout=30.0)
        yield client_instance
        client_instance.close()
    else:
        from fastapi.testclient import TestClient

        yield TestClient(app)
