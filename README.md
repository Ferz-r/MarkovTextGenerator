# Markov

An educational Markov chain text generator. The backend uses Python and FastAPI;
the interface uses HTML, CSS, and JavaScript with no build step.

Continue a text, choose a topic, train the model on your own texts or `.txt` files,
and change the context length, token frequency, and tokenizer type.
The model is saved between runs.

## Quick start

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```sh
cd backend
uv sync --locked
uv run python main.py
```

Once the model is ready, open http://127.0.0.1:8000.
API documentation: http://127.0.0.1:8000/docs.

If no saved model exists, the full `Mikimi/russian-wikipedia-top100k` dataset is
downloaded. The first run requires network access, time, and memory.
Sources are configured in `backend/src/markov/infrastructure/corpus.py`.
An existing snapshot is loaded without downloading or training.
The default training corpus remains in Russian; the interface language does not
change the language of generated text or translate topic queries.

## Docker

From the repository root:

```sh
docker build -t markov .
docker run -p 8000:8000 -v markov-data:/app/backend/data markov
```

Open http://127.0.0.1:8000 once the model is ready.
The image includes Python, backend dependencies, and frontend files.
Local virtual environments, corpora, and snapshots are excluded from the image.
The named `markov-data` volume preserves the model and Hugging Face cache between runs.
On the first run without a snapshot, the model trains on the configured dataset;
subsequent runs load the snapshot from the volume.

Uvicorn listens on `0.0.0.0` inside the container; the published port is available
at `127.0.0.1` on the host. It runs with one worker and no automatic reload.
The container runs as the `markov` user.

## Architecture

```text
backend/
├── main.py                  # entry point
├── src/markov/
│   ├── app.py               # FastAPI setup and lifecycle
│   ├── domain/              # model, index, settings, and tokenizers
│   ├── application/         # service: operations and synchronization
│   ├── infrastructure/      # files, snapshots, and external sources
│   ├── api/                 # HTTP schemas, routes, and SSE
│   ├── progress.py          # training progress messages
│   └── logging.py           # logging configuration
└── tests/
frontend/                    # interface with no npm dependencies
```

The algorithm is independent of the HTTP API and Uvicorn. Tokenizers implement a
shared abstract class. The model owns the corpus and validates its state during
restoration; the service manages access and persistence. See
[architecture](ARCHITECTURE.md) and [backend](backend/README.md).

## Checks

```sh
cd backend
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests
```

GitHub Actions runs these checks on Python 3.13 and 3.14.
Tests use small local corpora and do not download datasets.

## Limitations

This application is intended for local educational use. Run one worker: the model
and lock live in process memory. Training and generation run sequentially.
There is no authentication or corpus size limit. Publishing the source code does
not make the API ready for public internet access.

Safe updates temporarily keep both the old and new models in memory.
Large corpora and long contexts can require substantial memory.

The code is distributed under the [MIT license](LICENSE). The project license does
not cover external datasets; trained snapshots and downloaded texts are excluded from Git.
