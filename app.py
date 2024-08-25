from collections import OrderedDict
from fastapi import FastAPI, HTTPException
from flashrank import Ranker, RerankRequest
from pydantic import BaseModel
from typing import Any, Callable, List, TypeVar

import os
import re
import spacy
import time

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
english_ranker = Ranker()#model_name='ms-marco-TinyBERT-L-2-v2', cache_dir='./models'),
multilingual_reranker = Ranker(model_name='ms-marco-MultiBERT-L-12', cache_dir='./models')
rankers = {
    'en': english_ranker,
    'es': multilingual_reranker,
    'fr': multilingual_reranker,
}

#------------------------------------------------------------------------------#

# Get the time difference between now and a previous timestamp in milliseconds.
def took(start: float) -> int:
    return round((time.perf_counter() - start), 4)

# Split a text into sentences.
def split_text_into_sentences(text: str, language: str) -> List[str]:
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
    passages = [{'text': text} for text in texts]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = rankers[language].rerank(rerank_request)
    ranked_texts = [result.get('text') for result in results[:limit]]

    return ranked_texts

#------------------------------------------------------------------------------#

# API.
app = FastAPI()

class Request(BaseModel):
    language: str

class Response(BaseModel):
    took: float

#------------------------------------------------------------------------------#

class TextSplitRequest(Request):
    text: str

class TextSplitSentencesResponse(Response):
    sentences: List[str]

# Endpoint to split a text into sentences.
@app.post('/text/split/sentences')
def text_split_sentences(request: TextSplitRequest) -> TextSplitSentencesResponse:
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
    text: str

class TextTokenizeSentencesResponse(Response):
    tokens: List[str]

# Endpoint to tokenize a text.
@app.post('/text/tokenize/text')
def text_tokenize_text(request: TextSplitRequest) -> TextTokenizeSentencesResponse:
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
    question: str
    keywords: List[str]
    limit: int = 50

class TextCorrelateKeywordsResponse(Response):
    keywords: List[str]

# Endpoint to correlate keywords to a question.
@app.post('/text/correlate/keywords')
def text_compare_keywords(request: TextCorrelateKeywordsRequest) -> TextCorrelateKeywordsResponse:
    start_time = time.perf_counter()

    # Validate request.
    if request.language not in rankers or request.language not in nlp_models:
        raise HTTPException(status_code=400, detail='Unsupported language')
    if not request.question:
        raise HTTPException(status_code=400, detail='Missing question')
    if not request.keywords or len(request.keywords) == 0:
        raise HTTPException(status_code=400, detail='Missing keywords')

    # Process the question and extract its keywords.
    document = nlp_models[request.language](request.question)
    keywords = [token.text for token in document if not token.is_stop and not token.is_punct and not token.is_space]

    # Find the most relevant terms for the question and add them to the
    # keywords extracted from the question.
    # @todo, should be convert the keywords to lower case?
    keywords += rank_texts(request.question, request.keywords, request.language, request.limit)

    # Ensure uniqueness.
    keywords = list(OrderedDict.fromkeys(keywords))

    return TextCorrelateKeywordsResponse(keywords=keywords, took=took(start_time))

#------------------------------------------------------------------------------#

class TextCorrelateTextsRequest(Request):
    text: str
    texts: List[str]
    limit: int = 50

class TextCorrelateTextsResponse(Response):
    texts: List[str]

# Endpoint to correlate texts to another text.
@app.post('/text/correlate/texts')
def text_compare_keywords(request: TextCorrelateTextsRequest) -> TextCorrelateTextsResponse:
    start_time = time.perf_counter()

    # Validate request.
    if request.language not in rankers:
        raise HTTPException(status_code=400, detail='Unsupported language')
    if not request.text:
        raise HTTPException(status_code=400, detail='Missing text')
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=400, detail='Missing texts')

    # Find the most relevant terms for the question and add them to the
    # keywords extracted from the question.
    # @todo, should be convert the keywords to lower case?
    texts = rank_texts(request.text, request.texts, request.language, request.limit)

    return TextCorrelateTextsResponse(texts=texts, took=took(start_time))
