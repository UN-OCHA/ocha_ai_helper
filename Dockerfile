FROM python:3.12-slim

RUN apt-get update && apt-get install build-essential curl -y

# @todo use a requirements.txt file with fixed versions?
RUN pip install fastapi flashrank pydantic spacy uvicorn

RUN python3 -m spacy download en_core_web_sm && \
  python3 -m spacy download es_core_news_sm && \
  python3 -m spacy download fr_core_news_sm

COPY . /srv/www/html/

WORKDIR /srv/www/html/

CMD ["./server.sh"]
