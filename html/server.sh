#!/bin/bash

# @todo handle additional parameters.
uvicorn app:app \
  --host ${SERVER_HOST:-0.0.0.0} \
  --port ${SERVER_PORT:-80} \
  --log-config=log_config.yaml \
  --reload
