"""
Tests for the ReliefWeb helper API (app.py).
"""


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------


def test_health_status_returns_ok(client):
    """GET /status returns 200 and status ok."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# -----------------------------------------------------------------------------
# Text deduplicate embed
# -----------------------------------------------------------------------------


def test_text_deduplicate_embed_success(client):
    """POST /text/deduplicate/embed returns index-aligned embedding vectors."""
    texts = [
        "Green tea may improve health.",
        "Basketball is a popular sport.",
    ]
    payload = {"language": "en", "texts": texts}
    response = client.post("/text/deduplicate/embed", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert "took" in data
    assert isinstance(data["embeddings"], list)
    assert len(data["embeddings"]) == len(texts)
    assert all(isinstance(vec, list) for vec in data["embeddings"])
    assert all(isinstance(v, float) for vec in data["embeddings"] for v in vec)
    assert all(len(vec) > 0 for vec in data["embeddings"])


def test_text_deduplicate_embed_missing_texts(client):
    """POST /text/deduplicate/embed with empty texts returns 400."""
    payload = {"language": "en", "texts": []}
    response = client.post("/text/deduplicate/embed", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing texts"
