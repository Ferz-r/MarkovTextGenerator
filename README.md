# Markov

An educational Markov chain text generator. The backend uses Python and FastAPI;
the interface uses HTML, CSS, and JavaScript with no build step.

Continue a text, choose a topic, train the model on your own texts or `.txt` files,
and change the context length, token frequency, and tokenizer type.
The model is saved between runs.

## Quick start: Docker Hub

With Docker running, pull the prebuilt image from
[Docker Hub](https://hub.docker.com/r/ferzr/markov_text_generator) and start it:

```sh
docker pull ferzr/markov_text_generator
docker run -p 8000:8000 -v markov-data:/app/backend/data ferzr/markov_text_generator
```

Once the model is ready, open http://127.0.0.1:8000.
API documentation: http://127.0.0.1:8000/docs.
No local Python or uv installation is required for this option.

The image includes Python, backend dependencies, and frontend files.
The named `markov-data` volume preserves the model and Hugging Face cache between runs.
If no saved model exists, both `Mikimi/russian-wikipedia-top100k` (Russian) and
`agentlans/wikipedia-paragraphs` (English) are downloaded and the model is trained
on their combined texts. The first run requires network access,
time, and memory. Subsequent runs load the snapshot from the volume.
The default training corpus contains Russian and English texts. The interface
language does not translate generated text or topic queries.
Changing the source list does not update an existing model snapshot automatically.

## Build the Docker image yourself

If you want to build from source or include your own changes, run these commands
from the repository root instead:

```sh
docker build -t markov .
docker run -p 8000:8000 -v markov-data:/app/backend/data markov
```

Local virtual environments, corpora, and snapshots are excluded from the image.
Uvicorn listens on `0.0.0.0` inside the container; open the app at
http://127.0.0.1:8000 on the host. It runs with one worker and no automatic reload.
The container runs as the `markov` user.

## Run locally without Docker

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```sh
cd backend
uv sync --locked
uv run python main.py
```

Sources are configured in `backend/src/markov/infrastructure/corpus.py`.
An existing snapshot is loaded without downloading or training.

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
