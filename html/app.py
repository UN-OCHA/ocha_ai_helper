"""
This module provides a simple API with various text manipulation options.
"""
import re
import time
from typing import List

from collections import OrderedDict
from fastapi import FastAPI, HTTPException
from flashrank import Ranker, RerankRequest
from pydantic import BaseModel
import spacy

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
english_ranker = Ranker(model_name='ms-marco-TinyBERT-L-2-v2', cache_dir='/opt/models')
multilingual_reranker = Ranker(model_name='ms-marco-MultiBERT-L-12', cache_dir='/opt/models')
rankers = {
    'en': english_ranker,
    'es': multilingual_reranker,
    'fr': multilingual_reranker,
}

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
def rank_texts(query: str, texts: List[str], language: str, limit: int) -> List[str]:
    """
    Split a text into sentences.

    Args:
        query (str): the query to compare the texts against.
        texts (List[str]): the texts to compare against the query.
        language (str): the language of the texts and query as a ISO 639-2 code.
        limit (int): the maximum number of most relevant texts to the query.

    Returns:
        List[str]: most relevant texts.
    """
    passages = [{'text': text} for text in texts]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = rankers[language].rerank(rerank_request)
    ranked_texts = [result.get('text') for result in results[:limit]]

    return ranked_texts

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

# Health check endpoint.
@app.get('/status', status_code=200)
def health_status():
    """
    A very simple health check endpoint.
    """
    return {"status": "ok"}

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

    # Retrieve the tokens and ensure uniqueness.
    tokens = [token.text.strip().lower() for token in document if not token.is_space]
    tokens = list(set(tokens))

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
    keywords = [keyword.strip().lower() for keyword in request.keywords]
    keywords += rank_texts(request.text, keywords, request.language, request.limit)

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
    texts: List[str]

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
