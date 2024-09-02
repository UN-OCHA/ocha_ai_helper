# Build the code.
FROM python:3.12 AS builder

ENV S6VERSION=2.2.0.3

# Copy only the files necessary to build the code (see .dockerignore in root).
COPY . /srv/www

# Fetch s6 init files, needed later.
RUN apt-get clean && apt-get update && apt-get -y install ca-certificates curl && \
    S6ARCH=$(uname -m | sed 's/x86_64/amd64/') && \
    echo "Installing s6 version: $S6VERSION for $S6ARCH" && \
    curl -o /tmp/s6-overlay.tar.gz -jkSL https://github.com/just-containers/s6-overlay/releases/download/v${S6VERSION}/s6-overlay-${S6ARCH}.tar.gz

# Install dependencies. This will be in the `/root/.local` dir by default.
RUN pip install --root-user-action=ignore --upgrade pip setuptools wheel && \
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

ENV GIT_BLAME=$GITHUB_ACTOR \
    GIT_REPO=$GITHUB_REPOSITORY \
    GIT_SHA=$GITHUB_SHA \
    GIT_REF=$GITHUB_REF \
    GIT_MESSAGE=$GITHUB_MESSAGE

LABEL info.humanitarianresponse.build.date=$BUILD_DATE \
      info.humanitarianresponse.build.vcs-url=$VCS_URL \
      info.humanitarianresponse.build.vcs-ref=$VCS_REF

# Copy the installed python packages from the builder step.
COPY --from=builder /opt/venv /opt/venv

# For the init service.
COPY --from=builder /tmp/s6-overlay.tar.gz /tmp/
COPY --from=builder /srv/www/docker/etc/services/run_uvicorn /tmp/

# Set environment variables to use the virtual environment.
ENV PATH=/opt/venv/bin:$PATH

RUN \
    apt-get clean && apt-get update && apt-get -y install netcat-openbsd procps && \
    tar xzf /tmp/s6-overlay.tar.gz -C / && \
    rm -f tar xzf /tmp/s6-overlay.tar.gz && \
    # Add some users.
    addgroup --system --gid 4000 appuser && \
    adduser --system --uid 4000 --gid 4000 --shell /sbin/nologin --comment 'Docker App User' --home /home/appuser --no-create-home appuser && \
    addgroup --system --gid 508 jenkins && \
    adduser --system --uid 508 --gid 508 --shell /sbin/nologin --comment 'Jenkins User' --home /var/lib/jenkins --no-create-home jenkins && \
    mkdir -p /etc/services.d/uvicorn /var/log/uvicorn /opt/models  && \
    mv /tmp/run_uvicorn /etc/services.d/uvicorn/run && \
    chown -R appuser:appuser /var/log/uvicorn /opt/models && \
    # Download space models.
    python3 -m spacy download en_core_web_sm && \
    python3 -m spacy download es_core_news_sm && \
    python3 -m spacy download fr_core_news_sm

COPY --from=builder /srv/www/html /srv/www/html

WORKDIR /srv/www/html

ENTRYPOINT ["/init"]

EXPOSE 8000
