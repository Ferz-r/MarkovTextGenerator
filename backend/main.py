import argparse
from pathlib import Path

import uvicorn

from markov.app import create_app
from markov.infrastructure.corpus import load_demo_corpus
from markov.infrastructure.storage import MODEL_FILE
from markov.logging import configure_logging

app = create_app(model_path=MODEL_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Markov text generation server")
    parser.add_argument(
        "--demo", action="store_true", help="Use a small offline corpus"
    )
    args = parser.parse_args()
    configure_logging()
    application = app
    if args.demo:
        path = Path(__file__).resolve().parent / "data" / "demo-model.pkl.gz"
        application = create_app(load_demo_corpus, model_path=path)
    uvicorn.run(application, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
