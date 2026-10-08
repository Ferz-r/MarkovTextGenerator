FROM python:3.14-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    HF_HOME=/app/backend/data/huggingface \
    PATH="/app/backend/.venv/bin:$PATH"

WORKDIR /app/backend

# Cache dependencies independently of application source changes.
COPY backend/pyproject.toml backend/uv.lock backend/README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY backend/src ./src
COPY backend/main.py ./main.py
COPY frontend /app/frontend
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

RUN useradd --create-home --uid 1000 markov \
    && mkdir -p /app/backend/data \
    && chown markov:markov /app/backend/data
USER markov

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
