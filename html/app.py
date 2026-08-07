"""
This module provides a simple API with various text manipulation options.
"""
import re
import time
from typing import Dict, List

from collections import OrderedDict
from fastapi import FastAPI, HTTPException
from flashrank import Ranker, RerankRequest
from model2vec import StaticModel
import numpy as np
from pydantic import BaseModel
import spacy
from rapidfuzz import fuzz

# Load and set up the NLP language models.
nlp_pipeline_exclude = ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
nlp_models = {
    'en': spacy.load('en_core_web_sm', exclude=nlp_pipeline_exclude),
    'es': spacy.load('es_core_news_sm', exclude=nlp_pipeline_exclude),
    'fr': spacy.load('fr_core_news_sm', exclude=nlp_pipeline_exclude),
}
for model in nlp_models.values():
    model.enable_pipe('senter')

# Rankers.
english_ranker = Ranker()#model_name='ms-marco-TinyBERT-L-2-v2', cache_dir='/opt/models')
multilingual_reranker = Ranker(model_name='ms-marco-MultiBERT-L-12', cache_dir='/opt/models')
rankers = {
    'en': english_ranker,
    'es': multilingual_reranker,
    'fr': multilingual_reranker,
}

# Deduplication model: multilingual static embeddings, CPU-friendly.
dedup_model = StaticModel.from_pretrained('/opt/model2vec/potion-multilingual-128M')

#------------------------------------------------------------------------------#

# Get the time difference between now and a previous timestamp in milliseconds.
def took(start: float) -> float:
    """
    Get the time difference between now and a previous timestamp in milliseconds.

    Args:
        start (float): the stating time.

    Returns:
        float: the elapsed time in seconds.
    """

    return round((time.perf_counter() - start), 3)

# Split a text into sentences.
def split_text_into_sentences(text: str, language: str) -> List[str]:
    """
    Split a text into sentences.

    Args:
        text (str): the text to split into sentences.
        language (str): the language of the text as a ISO 639-2 code.

    Returns:
        List[str]: sentences.
    """

    # Process the text.
    document = nlp_models[language](text)

    # Retrieve the sentences.
    sentences = []
    for sentence in document.sents:
        # Clean the sentence.
        sentence = sentence.text.replace('\r', '').strip()
        # Split sentences with at least 2 consecutive line breaks.
        sentences += [part.strip() for part in re.split(r'\n{2,}', sentence)]

    # Remove empty strings.
    sentences = [sentence for sentence in sentences if sentence != '']
    return sentences

# Rank texts against a query.
def rank_texts(query: str, texts: List[str], language: str, limit: int) -> Dict[str, float]:
    """
    Split a text into sentences.

    Args:
        query (str): the query to compare the texts against.
        texts (List[str]): the texts to compare against the query.
        language (str): the language of the texts and query as a ISO 639-2 code.
        limit (int): the maximum number of most relevant texts to the query.

    Returns:
        Dict[str, float]: most relevant texts with their score.
    """
    passages = [{'text': text} for text in texts]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = rankers[language].rerank(rerank_request)
    ranked_texts = {result.get('text'): result.get('score') for result in results[:limit]}

    return ranked_texts

# Encode texts into embeddings for deduplication.
def encode_texts(texts: List[str]) -> np.ndarray:
    """
    Encode texts into embeddings using the deduplication model.

    Args:
        texts (List[str]): the texts to encode.

    Returns:
        np.ndarray: embedding matrix with one row per text.
    """
    # max_length=None disables truncation so full-length texts are embedded.
    return dedup_model.encode(texts, max_length=None)

# Score candidate embeddings against a query embedding via cosine similarity.
def score_embeddings(query_vec, candidate_vecs) -> List[float]:
    """
    Score candidate embeddings for similarity to a query embedding.

    Args:
        query_vec: the query embedding vector.
        candidate_vecs: the candidate embedding vectors.

    Returns:
        List[float]: cosine similarity scores, one per candidate, in input order.
    """
    query_arr = np.asarray(query_vec, dtype=np.float32)
    candidate_arr = np.asarray(candidate_vecs, dtype=np.float32)

    # Cosine similarity: dot product of unit-normalised vectors.
    query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-10)
    scores = []
    for vec in candidate_arr:
        vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
        scores.append(float(np.dot(query_norm, vec_norm)))

    return scores

# Score candidate texts for similarity to a source text using cosine similarity.
def deduplicate_texts(query: str, texts: List[str]) -> List[float]:
    """
    Score candidate texts for similarity to a source text.

    Args:
        query (str): the source text to compare candidates against.
        texts (List[str]): the candidate texts (potential duplicates).

    Returns:
        List[float]: cosine similarity scores, one per candidate text, in the
            same order as the input texts.
    """
    embeddings = encode_texts([query] + texts)
    return score_embeddings(embeddings[0], embeddings[1:])

#------------------------------------------------------------------------------#

# API.
app = FastAPI()

class Request(BaseModel):
    """
    A basic API request.

    Attributes:
        language (str): language of the content of the request as a ISO 639-2 code.
    """
    language: str

class Response(BaseModel):
    """
    A basic API response.

    Attributes:
        took (float): the elapsed time in seconds.
    """
    took: float

#------------------------------------------------------------------------------#

class HealthCheckResponse(BaseModel):
    """
    Health check response.

    Attributes:
        status (str): the health status of the app, defaults to "ok".
    """
    status: str

# Health check endpoint.
@app.get('/status', status_code=200)
def health_status() -> HealthCheckResponse:
    """
    A very simple health check endpoint.

    Returns:
        HealthCheckResponse: the health check response.
    """
    return HealthCheckResponse(status="ok")

#------------------------------------------------------------------------------#

class TextSplitRequest(Request):
    """
    A text splitting request.

    Attributes:
        text (str): the text to split.
    """
    text: str

class TextSplitSentencesResponse(Response):
    """
    A text splitting response.

    Attributes:
        sentences (List[str]): the sentences.
    """
    sentences: List[str]

# Endpoint to split a text into sentences.
@app.post('/text/split/sentences')
def text_split_sentences(request: TextSplitRequest) -> TextSplitSentencesResponse:
    """
    API endpoint callback to split a text into sentences.

    Args:
        request (TextSplitRequest): the text splitting request.

    Returns:
        TextSplitSentenceResponse: the text splitting response.
    """
    start_time = time.perf_counter()

    # Validate request.
    if request.language not in nlp_models:
        raise HTTPException(status_code=400, detail='Unsupported language')
    if not request.text:
        raise HTTPException(status_code=400, detail='Missing text')

    # Retrieve the sentences.
    sentences = split_text_into_sentences(request.text, request.language)

    return TextSplitSentencesResponse(sentences=sentences, took=took(start_time))

#------------------------------------------------------------------------------#

class TextTokenizeTextRequest(Request):
    """
    A text tokenizing request.

    Attributes:
        text (str): the text to tokenize.
    """
    text: str

class TextTokenizeSentencesResponse(Response):
    """
    A text tokenizing response.

    Attributes:
        tokens (List[str]): the tokens.
    """
    tokens: List[str]

# Endpoint to tokenize a text.
@app.post('/text/tokenize/text')
def text_tokenize_text(request: TextTokenizeTextRequest) -> TextTokenizeSentencesResponse:
    """
    API endpoint callback to tokenize a text.

    Args:
        request (TextTokenizeTextRequest): the text tokenizing request.

    Returns:
        TextTokenizeSentencesResponse: the text tokenizing response.
    """
    start_time = time.perf_counter()

    # Validate request.
    if request.language not in nlp_models:
        raise HTTPException(status_code=400, detail='Unsupported language')
    if not request.text:
        raise HTTPException(status_code=400, detail='Missing text')

    # Process the text.
    document = nlp_models[request.language](request.text)

    # Retrieve the tokens and ensure uniqueness (order preserved).
    tokens = [token.text.strip().lower() for token in document if not token.is_space]
    tokens = list(dict.fromkeys(tokens))

    return TextTokenizeSentencesResponse(tokens=tokens, took=took(start_time))

#------------------------------------------------------------------------------#

class TextCorrelateKeywordsRequest(Request):
    """
    A keyword correlation request.

    Attributes:
        text (str): the text to correlate.
        keywords (List[str]): the keywords to correlate.
        limit (int): the maximum number of keywords to return.
    """
    text: str
    keywords: List[str]
    limit: int = 50

class TextCorrelateKeywordsResponse(Response):
    """
    A keyword correlation response.

    Attributes:
        keywords (List[str]): the most relevant keywords.
    """
    keywords: List[str]

# Endpoint to correlate keywords to a text.
@app.post('/text/correlate/keywords')
def text_correlate_keywords(request: TextCorrelateKeywordsRequest) -> TextCorrelateKeywordsResponse:
    """
    API endpoint callback to correlate keywords to a text.

    Args:
        request (TextCorrelateKeywordsRequest): the keyword correlation request.

    Returns:
        TextCorrelateKeywordsResponse: the keyword correlation response.
    """
    start_time = time.perf_counter()

    # Validate request.
    if request.language not in rankers or request.language not in nlp_models:
        raise HTTPException(status_code=400, detail='Unsupported language')
    if not request.text:
        raise HTTPException(status_code=400, detail='Missing text')
    if not request.keywords or len(request.keywords) == 0:
        raise HTTPException(status_code=400, detail='Missing keywords')

    # Process the text and extract its keywords.
    document = nlp_models[request.language](request.text)
    keywords = [token.text for token in document if (
        not token.is_stop and
        not token.is_punct and
        not token.is_space
    )]

    # Find the most relevant terms for the text and add them to the
    # keywords extracted from the text.
    keywords += rank_texts(request.text, request.keywords, request.language, request.limit).keys()

    # Ensure uniqueness.
    keywords = list(OrderedDict.fromkeys(keywords))

    return TextCorrelateKeywordsResponse(keywords=keywords, took=took(start_time))

#------------------------------------------------------------------------------#

class TextCorrelateTextsRequest(Request):
    """
    A text correlation request.

    Attributes:
        text (str): the text to correlate.
        texts (List[str]): the texts to correlate.
        limit (int): the maximum number of texts to return.
    """
    text: str
    texts: List[str]
    limit: int = 50

class TextCorrelateTextsResponse(Response):
    """
    A text correlation response.

    Attributes:
        texts (List[str]): the most relevant tests.
    """
    texts: Dict[str, float]

# Endpoint to correlate texts to another text.
@app.post('/text/correlate/texts')
def text_correlate_texts(request: TextCorrelateTextsRequest) -> TextCorrelateTextsResponse:
    """
    API endpoint callback to correlate texts to a another text.

    Args:
        request (TextCorrelateTextsRequest): the text correlation request.

    Returns:
        TextCorrelateTextsResponse: the text correlation response.
    """
    start_time = time.perf_counter()

    # Validate request.
    if request.language not in rankers:
        raise HTTPException(status_code=400, detail='Unsupported language')
    if not request.text:
        raise HTTPException(status_code=400, detail='Missing text')
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=400, detail='Missing texts')

    # Order the texts by relevance to the other text and return the given limit
    # number of the most relevant ones.
    texts = rank_texts(request.text, request.texts, request.language, request.limit)

    return TextCorrelateTextsResponse(texts=texts, took=took(start_time))

#------------------------------------------------------------------------------#

class TextDeduplicateTextsRequest(Request):
    """
    A text deduplication request using raw texts.

    Attributes:
        text (str): the source text to check for duplicates.
        texts (List[str]): the candidate texts (potential duplicates).
    """
    text: str
    texts: List[str]

class TextDeduplicateScoresResponse(Response):
    """
    A text deduplication response with similarity scores.

    Attributes:
        scores (List[float]): cosine similarity scores, index-aligned with the
            request candidates.
    """
    scores: List[float]

# Endpoint to score potential duplicate texts against a source text.
@app.post('/text/deduplicate/texts')
def text_deduplicate_texts(request: TextDeduplicateTextsRequest) -> TextDeduplicateScoresResponse:
    """
    API endpoint callback to score potential duplicates against a source text.

    Args:
        request (TextDeduplicateTextsRequest): the text deduplication request.

    Returns:
        TextDeduplicateScoresResponse: the text deduplication response.
    """
    start_time = time.perf_counter()

    # Validate request.
    if not request.text:
        raise HTTPException(status_code=400, detail='Missing text')
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=400, detail='Missing texts')

    # Score each candidate by similarity to the source text (index-aligned).
    scores = deduplicate_texts(request.text, request.texts)

    return TextDeduplicateScoresResponse(scores=scores, took=took(start_time))

#------------------------------------------------------------------------------#

class TextDeduplicateEmbedRequest(Request):
    """
    A request to embed texts with the deduplication model.

    Attributes:
        texts (List[str]): the texts to embed.
    """
    texts: List[str]

class TextDeduplicateEmbedResponse(Response):
    """
    A response containing embeddings for the requested texts.

    Attributes:
        embeddings (List[List[float]]): embedding vectors, index-aligned with
            the request texts.
    """
    embeddings: List[List[float]]

# Endpoint to embed texts for later deduplication.
@app.post('/text/deduplicate/embed')
def text_deduplicate_embed(request: TextDeduplicateEmbedRequest) -> TextDeduplicateEmbedResponse:
    """
    API endpoint callback to embed texts with the deduplication model.

    Args:
        request (TextDeduplicateEmbedRequest): the embed request.

    Returns:
        TextDeduplicateEmbedResponse: the embed response.
    """
    start_time = time.perf_counter()

    # Validate request.
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=400, detail='Missing texts')

    embeddings = encode_texts(request.texts).tolist()

    return TextDeduplicateEmbedResponse(embeddings=embeddings, took=took(start_time))

#------------------------------------------------------------------------------#

class TextDeduplicateEmbeddingsRequest(Request):
    """
    A text deduplication request using stored embeddings.

    Attributes:
        embedding (List[float]): the source text embedding.
        embeddings (List[List[float]]): the candidate embeddings.
    """
    embedding: List[float]
    embeddings: List[List[float]]

# Endpoint to score potential duplicates using stored embeddings.
@app.post('/text/deduplicate/embeddings')
def text_deduplicate_embeddings(
    request: TextDeduplicateEmbeddingsRequest,
) -> TextDeduplicateScoresResponse:
    """
    API endpoint callback to score potential duplicates from embeddings.

    Args:
        request (TextDeduplicateEmbeddingsRequest): the embeddings request.

    Returns:
        TextDeduplicateScoresResponse: the text deduplication response.
    """
    start_time = time.perf_counter()

    # Validate request.
    if not request.embedding or len(request.embedding) == 0:
        raise HTTPException(status_code=400, detail='Missing embedding')
    if not request.embeddings or len(request.embeddings) == 0:
        raise HTTPException(status_code=400, detail='Missing embeddings')

    expected_dim = len(request.embedding)
    for index, candidate in enumerate(request.embeddings):
        if len(candidate) != expected_dim:
            raise HTTPException(
                status_code=400,
                detail=(
                    f'Embedding dimension mismatch at index {index}: '
                    f'expected {expected_dim}, got {len(candidate)}'
                ),
            )

    scores = score_embeddings(request.embedding, request.embeddings)

    return TextDeduplicateScoresResponse(scores=scores, took=took(start_time))

#------------------------------------------------------------------------------#

class TextMatchLinesRequest(Request):
    """
    A request to find lines that match against a given text.

    Attributes:
        text (str): The text to search for matches against
        lines (List[str]): The list of lines to search through
        threshold (int): Minimum similarity score (0-100) to consider a match
    """
    text: str
    lines: List[str]
    threshold: int = 70

class TextMatchLinesResponse(Response):
    """
    A response containing the lines that match against the given text.

    Attributes:
        lines (List[str]): List of matched lines in original order
    """
    lines: List[str]

# Endpoint to match lines against a text
@app.post('/text/match/lines')
def text_match(request: TextMatchLinesRequest) -> TextMatchLinesResponse:
    """
    API endpoint callback to find matching lines against a text.

    Args:
        request (TextMatchLinesRequest): The text matching request.

    Returns:
        TextMatchLinesResponse: The text matching response containing matched lines.
    """
    start_time = time.perf_counter()

    # Validate request
    if not request.text:
        raise HTTPException(status_code=400, detail='Missing text')
    if not request.lines:
        raise HTTPException(status_code=400, detail='Missing lines')
    if not 0 <= request.threshold <= 100:
        raise HTTPException(status_code=400, detail='Threshold must be between 0 and 100')

    # Find matching lines
    matched_lines = []
    for line in request.lines:
        if line.strip():  # Skip empty lines
            # Try different fuzzy matching methods for better results
            partial_score = fuzz.partial_ratio(line.strip(), request.text)
            token_set_score = fuzz.token_set_ratio(line.strip(), request.text)

            # Use the best score from the different methods
            best_score = max(partial_score, token_set_score)

            if best_score >= request.threshold:
                matched_lines.append(line.strip())

    # Remove duplicates while preserving order
    seen = set()
    unique_matches = []
    for line in matched_lines:
        if line not in seen:
            unique_matches.append(line)
            seen.add(line)

    return TextMatchLinesResponse(lines=unique_matches, took=took(start_time))
