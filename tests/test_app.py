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


# -----------------------------------------------------------------------------
# Visual classify
# -----------------------------------------------------------------------------


def _tiny_png_base64():
    """Return base64 for a 2x2 RGB PNG."""
    import base64
    import io

    from PIL import Image

    image = Image.new("RGB", (2, 2), (255, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_visual_classify_success(client):
    """POST /visual/classify with a valid image returns label, probs, took."""
    payload = {"image_base64": _tiny_png_base64()}
    response = client.post("/visual/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ("Map", "Infographic", "Other")
    assert set(data["probabilities"].keys()) == {"Map", "Infographic", "Other"}
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-5
    assert "took" in data


def test_visual_classify_data_url_success(client):
    """POST /visual/classify accepts a data-URL base64 image."""
    payload = {"image_base64": f"data:image/png;base64,{_tiny_png_base64()}"}
    response = client.post("/visual/classify", json=payload)
    assert response.status_code == 200
    assert response.json()["label"] in ("Map", "Infographic", "Other")


def test_visual_classify_missing_image(client):
    """POST /visual/classify with empty image_base64 returns 400."""
    payload = {"image_base64": ""}
    response = client.post("/visual/classify", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing image_base64"


def test_visual_classify_invalid_base64(client):
    """POST /visual/classify with undecodable base64 returns 400."""
    payload = {"image_base64": "%%%not-base64%%%"}
    response = client.post("/visual/classify", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image_base64"


def test_visual_classify_invalid_image(client):
    """POST /visual/classify with valid base64 that is not an image returns 400."""
    import base64

    payload = {"image_base64": base64.b64encode(b"not an image").decode("ascii")}
    response = client.post("/visual/classify", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image"
