# OCHA AI Helper

Small HTTP API that runs ML helpers for OCHA sites (notably ReliefWeb): text embeddings, image classification, and similar CPU-friendly tasks. Callers own orchestration and scoring; this service exposes focused endpoints.

## Status

`GET /status` returns a simple health check.

Response:

```json
{ "status": "ok" }
```

## Text deduplicate embed

`POST /text/deduplicate/embed` returns embedding vectors for a list of texts (Model2Vec `potion-multilingual-128M`). Used for near-duplicate matching; cosine scoring is done by the caller (e.g. PHP/MariaDB on ReliefWeb).

Request:

```json
{
  "language": "en",
  "texts": ["First text to embed.", "Second text to embed."]
}
```

`language` is accepted for client compatibility but is not used by the multilingual model.

Response:

```json
{
  "embeddings": [[0.01, -0.02, "..."], [0.03, 0.04, "..."]],
  "took": 0.012
}
```

Embeddings are index-aligned with the request `texts`.

## Visual classify

`POST /visual/classify` classifies an image as `Map`, `Infographic`, or `Other` using a CPU-only ONNX model. The helper does **not** fetch remote URLs — send image bytes as base64 (raw or data-URL).

Request:

```json
{ "image_base64": "<raw or data-URL base64 of JPEG/PNG/WebP>" }
```

Response:

```json
{
  "label": "Map",
  "probabilities": { "Map": 0.92, "Infographic": 0.07, "Other": 0.01 },
  "took": 0.012
}
```

## Run locally

See [local/README.md](local/README.md).
