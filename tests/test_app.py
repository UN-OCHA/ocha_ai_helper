"""
Tests for the text manipulation API (app.py).
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
# Text split sentences
# -----------------------------------------------------------------------------


def test_text_split_sentences_success(client):
    """POST /text/split/sentences returns sentences and took."""
    payload = {"language": "en", "text": "First sentence. Second sentence. Third sentence."}
    response = client.post("/text/split/sentences", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "sentences" in data
    assert "took" in data
    assert isinstance(data["sentences"], list)
    assert len(data["sentences"]) == 3
    assert data["sentences"] == ["First sentence.", "Second sentence.", "Third sentence."]


def test_text_split_sentences_unsupported_language(client):
    """POST /text/split/sentences with unsupported language returns 400."""
    payload = {"language": "xx", "text": "Some text."}
    response = client.post("/text/split/sentences", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported language"


def test_text_split_sentences_missing_text(client):
    """POST /text/split/sentences with empty text returns 400."""
    payload = {"language": "en", "text": ""}
    response = client.post("/text/split/sentences", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing text"


# -----------------------------------------------------------------------------
# Text tokenize
# -----------------------------------------------------------------------------


def test_text_tokenize_text_success(client):
    """POST /text/tokenize/text returns tokens and took."""
    payload = {"language": "en", "text": "hello world, this is a world wide test"}
    response = client.post("/text/tokenize/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert "took" in data
    assert isinstance(data["tokens"], list)
    assert len(data["tokens"]) == 8
    assert data["tokens"] == ["hello", "world", ",", "this", "is", "a", "wide", "test"]


def test_text_tokenize_text_unsupported_language(client):
    """POST /text/tokenize/text with unsupported language returns 400."""
    payload = {"language": "xx", "text": "hello"}
    response = client.post("/text/tokenize/text", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported language"


def test_text_tokenize_text_missing_text(client):
    """POST /text/tokenize/text with empty text returns 400."""
    payload = {"language": "en", "text": ""}
    response = client.post("/text/tokenize/text", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing text"


# -----------------------------------------------------------------------------
# Text correlate keywords
# -----------------------------------------------------------------------------


def test_text_correlate_keywords_success(client):
    """POST /text/correlate/keywords returns keywords and took."""
    payload = {
        "language": "en",
        "text": "Some document about health and nutrition.",
        "keywords": ["health", "food"],
        "limit": 10,
    }
    response = client.post("/text/correlate/keywords", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "keywords" in data
    assert "took" in data
    assert isinstance(data["keywords"], list)
    # Result = doc tokens + ranked keywords; with mocks we get mock tokens + ranker hits
    assert len(data["keywords"]) > 0
    # With live API, "health" and "food" are relevant; with mocks, ranker returns fixed strings
    assert any(
        k in data["keywords"] for k in ("health", "food", "most relevant")
    ), "keywords should contain requested or ranked terms"


def test_text_correlate_keywords_unsupported_language(client):
    """POST /text/correlate/keywords with unsupported language returns 400."""
    payload = {
        "language": "xx",
        "text": "Some text.",
        "keywords": ["foo"],
    }
    response = client.post("/text/correlate/keywords", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported language"


def test_text_correlate_keywords_missing_text(client):
    """POST /text/correlate/keywords with empty text returns 400."""
    payload = {"language": "en", "text": "", "keywords": ["foo"]}
    response = client.post("/text/correlate/keywords", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing text"


def test_text_correlate_keywords_missing_keywords(client):
    """POST /text/correlate/keywords with empty keywords returns 400."""
    payload = {"language": "en", "text": "Some text.", "keywords": []}
    response = client.post("/text/correlate/keywords", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing keywords"


# -----------------------------------------------------------------------------
# Text correlate texts
# -----------------------------------------------------------------------------


def test_text_correlate_texts_success(client):
    """POST /text/correlate/texts returns ranked texts and took."""
    payload = {
        "language": "en",
        "text": "Query about climate.",
        "texts": ["Doc about food.", "Doc about climate change.", "Doc about sports."],
        "limit": 5,
    }
    response = client.post("/text/correlate/texts", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "texts" in data
    assert "took" in data
    assert isinstance(data["texts"], dict)
    assert data["texts"]
    scores = data["texts"]
    # With live API: climate doc ranks highest for "Query about climate."
    # With mocks: ranker returns fixed keys "most relevant", "second"
    if "Doc about climate change." in scores:
        assert scores["Doc about climate change."] == max(scores.values())
    else:
        assert all(isinstance(v, (int, float)) for v in scores.values())
        assert max(scores.values()) > 0


def test_text_correlate_texts_unsupported_language(client):
    """POST /text/correlate/texts with unsupported language returns 400."""
    payload = {
        "language": "xx",
        "text": "Query.",
        "texts": ["Doc one."],
    }
    response = client.post("/text/correlate/texts", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported language"


def test_text_correlate_texts_missing_text(client):
    """POST /text/correlate/texts with empty text returns 400."""
    payload = {"language": "en", "text": "", "texts": ["Doc."]}
    response = client.post("/text/correlate/texts", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing text"


def test_text_correlate_texts_missing_texts(client):
    """POST /text/correlate/texts with empty texts returns 400."""
    payload = {"language": "en", "text": "Query.", "texts": []}
    response = client.post("/text/correlate/texts", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing texts"


# -----------------------------------------------------------------------------
# Text deduplicate texts
# -----------------------------------------------------------------------------


def test_text_deduplicate_texts_success(client):
    """POST /text/deduplicate/texts returns index-aligned similarity scores."""
    texts = [
        "Green tea may improve health.",
        "Basketball is a popular sport.",
    ]
    payload = {
        "language": "en",
        "text": "Green tea has health benefits.",
        "texts": texts,
    }
    response = client.post("/text/deduplicate/texts", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "scores" in data
    assert "took" in data
    assert isinstance(data["scores"], list)
    # One score per candidate text, in the same order.
    assert len(data["scores"]) == len(texts)
    assert all(isinstance(v, float) for v in data["scores"])
    # First candidate (tea-related) is more similar than the second (sport).
    assert data["scores"][0] > data["scores"][1]


def test_text_deduplicate_texts_missing_text(client):
    """POST /text/deduplicate/texts with empty text returns 400."""
    payload = {"language": "en", "text": "", "texts": ["Doc."]}
    response = client.post("/text/deduplicate/texts", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing text"


def test_text_deduplicate_texts_missing_texts(client):
    """POST /text/deduplicate/texts with empty texts returns 400."""
    payload = {"language": "en", "text": "Query.", "texts": []}
    response = client.post("/text/deduplicate/texts", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing texts"


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
# Text deduplicate embeddings
# -----------------------------------------------------------------------------


def test_text_deduplicate_embeddings_success(client):
    """POST /text/deduplicate/embeddings returns index-aligned similarity scores."""
    # Vectors match the conftest mock layout: query ~ first candidate, orthogonal to second.
    payload = {
        "language": "en",
        "embedding": [1.0, 0.0, 0.0, 0.0],
        "embeddings": [
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
    }
    response = client.post("/text/deduplicate/embeddings", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "scores" in data
    assert "took" in data
    assert isinstance(data["scores"], list)
    assert len(data["scores"]) == 2
    assert all(isinstance(v, float) for v in data["scores"])
    assert data["scores"][0] > data["scores"][1]


def test_text_deduplicate_embeddings_missing_embedding(client):
    """POST /text/deduplicate/embeddings with empty embedding returns 400."""
    payload = {
        "language": "en",
        "embedding": [],
        "embeddings": [[1.0, 0.0]],
    }
    response = client.post("/text/deduplicate/embeddings", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing embedding"


def test_text_deduplicate_embeddings_missing_embeddings(client):
    """POST /text/deduplicate/embeddings with empty embeddings returns 400."""
    payload = {
        "language": "en",
        "embedding": [1.0, 0.0],
        "embeddings": [],
    }
    response = client.post("/text/deduplicate/embeddings", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing embeddings"


def test_text_deduplicate_embeddings_dimension_mismatch(client):
    """POST /text/deduplicate/embeddings with mismatched dims returns 400."""
    payload = {
        "language": "en",
        "embedding": [1.0, 0.0, 0.0, 0.0],
        "embeddings": [
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0],
        ],
    }
    response = client.post("/text/deduplicate/embeddings", json=payload)
    assert response.status_code == 400
    assert "dimension mismatch" in response.json()["detail"].lower()


# -----------------------------------------------------------------------------
# Text match lines
# -----------------------------------------------------------------------------


def test_text_match_lines_success(client):
    """POST /text/match/lines returns matching lines and took."""
    payload = {
        "language": "en",
        "text": "human rights and humanitarian action",
        "lines": [
            "Human rights are important.",
            "Humanitarian action in crisis.",
            "Sports and games.",
        ],
        "threshold": 70,
    }
    response = client.post("/text/match/lines", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "lines" in data
    assert "took" in data
    assert isinstance(data["lines"], list)
    # Lines about human rights / humanitarian action match at threshold 70; sports does not
    assert "Human rights are important." in data["lines"]
    assert "Humanitarian action in crisis." in data["lines"]
    assert "Sports and games." not in data["lines"]
    assert len(data["lines"]) == 2


def test_text_match_lines_exact_match(client):
    """POST /text/match/lines with exact line returns that line only."""
    payload = {
        "language": "en",
        "text": "exact phrase",
        "lines": ["exact phrase", "other line"],
        "threshold": 100,
    }
    response = client.post("/text/match/lines", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["lines"] == ["exact phrase"]
    assert "other line" not in data["lines"]


def test_text_match_lines_high_threshold_filters(client):
    """POST /text/match/lines with high threshold returns fewer matches."""
    payload = {
        "language": "en",
        "text": "human rights",
        "lines": ["Human rights matter.", "Something else."],
        "threshold": 95,
    }
    response = client.post("/text/match/lines", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["lines"], list)
    # Unrelated line must not match at high threshold
    assert "Something else." not in data["lines"]
    # At 95 we may get 0 or 1 match; if any, it should be the relevant line
    if data["lines"]:
        assert data["lines"] == ["Human rights matter."]


def test_text_match_lines_missing_text(client):
    """POST /text/match/lines with empty text returns 400."""
    payload = {"language": "en", "text": "", "lines": ["a line"], "threshold": 70}
    response = client.post("/text/match/lines", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing text"


def test_text_match_lines_missing_lines(client):
    """POST /text/match/lines with empty lines returns 400."""
    payload = {"language": "en", "text": "query", "lines": [], "threshold": 70}
    response = client.post("/text/match/lines", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing lines"


def test_text_match_lines_invalid_threshold_too_high(client):
    """POST /text/match/lines with threshold > 100 returns 400."""
    payload = {"language": "en", "text": "query", "lines": ["line"], "threshold": 101}
    response = client.post("/text/match/lines", json=payload)
    assert response.status_code == 400
    assert "Threshold" in response.json()["detail"]


def test_text_match_lines_invalid_threshold_negative(client):
    """POST /text/match/lines with negative threshold returns 400."""
    payload = {"language": "en", "text": "query", "lines": ["line"], "threshold": -1}
    response = client.post("/text/match/lines", json=payload)
    assert response.status_code == 400
    assert "Threshold" in response.json()["detail"]
