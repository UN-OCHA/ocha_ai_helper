"""
OCHA AI Helper HTTP API: focused ML endpoints for OCHA sites.
"""
import time
from typing import List

from fastapi import FastAPI, HTTPException
from model2vec import StaticModel
import numpy as np
from pydantic import BaseModel

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
