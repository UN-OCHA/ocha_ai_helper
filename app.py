import os
import re
import spacy
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Get the time difference between now and a previous timestamp in milliseconds.
def took(start):
    return round((time.perf_counter() - start), 4)

# Load and set up the NLP language models.
nlp_pipeline_exclude = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
nlp_models = {
    "en": spacy.load('en_core_web_sm', exclude=nlp_pipeline_exclude),
    "es": spacy.load('es_core_news_sm', exclude=nlp_pipeline_exclude),
    "fr": spacy.load('fr_core_news_sm', exclude=nlp_pipeline_exclude),
}
for model in nlp_models.values():
    model.enable_pipe("senter")

# API.
app = FastAPI()

class Response(BaseModel):
    took: float

class TextSplitRequest(BaseModel):
    language: str
    text: str

class TextSplitSentencesResponse(Response):
    sentences: List[str]

# Endpoint to split a text into sentences.
@app.post("/text/split/sentences")
def text_split_sentences(request: TextSplitRequest) -> TextSplitSentencesResponse:
    start_time = time.perf_counter()
    # Unsupported language.
    if request.language not in nlp_models:
        raise HTTPException(status_code=400, detail="Unrecognized language")
    # Process the text.
    document = nlp_models[request.language](request.text)
    # Retrieve the sentences.
    sentences = []
    for sentence in document.sents:
        # Clean the sentence.
        sentence = sentence.text.replace("\r", "").strip()
        # Split sentences with at least 2 consecutive line breaks.
        sentences += [part.strip() for part in re.split(r"\n{2,}", sentence)]
        # Remove empty strings.
        sentences = [sentence for sentence in sentences if sentence != ""]
    return TextSplitSentencesResponse(sentences=sentences, took=took(start_time))
