# Markov API

## Running

```sh
uv sync --locked
uv run python main.py               # configured external sources
uv run uvicorn main:app --reload     # development with external sources
```

API: http://127.0.0.1:8000/api/v1. Swagger: http://127.0.0.1:8000/docs.

Startup loads `data/model.pkl.gz`. If the snapshot is missing or unreadable, the
model trains on the corpus. Snapshot parameters take precedence over the service's
initial settings. Training texts are loaded from configured sources;
no local JSON corpus cache is used. The default dataset remains in Russian.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Application readiness |
| GET | `/model` | Settings, text count, vocabulary size, and context count |
| POST | `/generate` | Generate with `prefix` and optional `topic` |
| POST | `/train` | Train on `texts`; `replace: true` replaces the corpus |
| POST | `/settings` | Change settings |
| POST | `/train/stream` | Train with SSE progress |
| POST | `/settings/stream` | Change settings with SSE progress |
| POST | `/train/file/stream` | Upload a `.txt` file and train with SSE progress |

All paths in the table have the `/api/v1` prefix.

```sh
curl -X POST http://127.0.0.1:8000/api/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prefix": "", "topic": ""}'

curl -X POST http://127.0.0.1:8000/api/v1/train \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["A new text for training."], "replace": false}'
```

Prefixes and topics should use the language of the training corpus. Topic queries
are matched literally, without translation.

`texts` are split into lines, skipping blank lines; entirely blank items are
rejected. Unknown topics and invalid JSON requests return HTTP 422.

The file endpoint accepts multipart fields `file`, `replace`, and optional `texts`
(a JSON array). UTF-8, UTF-16 with BOM, and Windows-1251 are supported. Each nonempty
line is a separate text. The temporary file is deleted after processing.

SSE sends `start`, `progress`, `complete`, or `error` events. Percentages refer to
the current stage. Disconnecting does not cancel an operation that has started.
An operation failure after streaming begins is sent as an `error` event rather
than a new HTTP status.

## Settings

```json
{"tokenizer": "character", "n_gramm": 3, "min_frequency": 5, "max_length": 100}
```

- `tokenizer`: `character` or `regex`.
- `n_gramm`: a context of 1–50 tokens. The historical API field name is preserved;
  the model uses `context_size` internally.
- `min_frequency`: minimum token frequency, 1–100000.
- `max_length`: maximum number of new tokens, 1–10000.

Tokenizer type, frequency, and context changes rebuild the model; length changes
apply without training. Settings are shared by all clients and saved between runs.
Initial values are defined in `MarkovService`.

Topic selection uses texts containing the topic, ignoring case. The two most
recent topic indexes are cached. Corpus changes clear the cache.

## Structure and persistence

See [architecture](../ARCHITECTURE.md). `domain/transitions.py` stores a compact
index of full reversed contexts; short-context frequencies are summed over a range.
The `model.transitions` property expands the full table for debugging and can use
substantial memory. Use `contexts_count` for statistics.

Training and generation are protected by a lock. The service builds a new model
separately and publishes it only after a successful write. Version 1 snapshots
contain primitive data in gzip; the loader prohibits Python classes. Compatibility
with earlier version 1 snapshots and the texts required for retraining is preserved.

Use one worker. Snapshots and corpora in `data/` are excluded from the repository.
Persistence is disabled in test applications unless `model_path` is provided.

## Checks

```sh
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests
```
