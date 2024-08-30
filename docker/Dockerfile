# Build the code.
FROM python:3.12 AS builder

# Copy only the files necessary to build the code (see .dockerignore in root).
COPY . /srv/www

# Install dependencies. This will be in the `/root/.local` dir by default.
RUN pip install --upgrade pip setuptools wheel && \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install -r /srv/www/html/requirements.txt

# Generate the image.
FROM python:3.12-slim

ARG VCS_REF
ARG VCS_URL
ARG BUILD_DATE
ARG GITHUB_ACTOR
ARG GITHUB_REPOSITORY
ARG GITHUB_SHA
ARG GITHUB_REF

ENV NGINX_SERVERNAME=ocha-ai-helper.test \
    GIT_BLAME=$GITHUB_ACTOR \
    GIT_REPO=$GITHUB_REPOSITORY \
    GIT_SHA=$GITHUB_SHA \
    GIT_REF=$GITHUB_REF \
    GIT_MESSAGE=$GITHUB_MESSAGE

LABEL info.humanitarianresponse.build.date=$BUILD_DATE \
      info.humanitarianresponse.build.vcs-url=$VCS_URL \
      info.humanitarianresponse.build.vcs-ref=$VCS_REF

# Copy the installed python packages from the builder step.
COPY --from=builder /opt/venv /opt/venv

# Set environment variables to use the virtual environment.
ENV PATH=/opt/venv/bin:$PATH

# Download space models.
RUN mkdir -p /var/log/uvicorn /opt/models  && \
  python3 -m spacy download en_core_web_sm && \
  python3 -m spacy download es_core_news_sm && \
  python3 -m spacy download fr_core_news_sm

COPY --from=builder /srv/www/html /srv/www/html

WORKDIR /srv/www/html

CMD ["/srv/www/html/server.sh"]
