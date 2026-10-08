import uvicorn

from markov.app import create_app
from markov.infrastructure.storage import MODEL_FILE
from markov.logging import configure_logging

app = create_app(model_path=MODEL_FILE)


def main() -> None:
    configure_logging()
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
