import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from markov.api.routes import router
from markov.application.service import MarkovService
from markov.infrastructure.corpus import load_corpus
from markov.logging import configure_logging

logger = logging.getLogger("markov.app")


def create_app(
    corpus_loader: Callable[[], list[str]] = load_corpus,
    model_path: Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        try:
            service = None
            if model_path is not None and model_path.exists():
                try:
                    service = await run_in_threadpool(MarkovService.load, model_path)
                except Exception:
                    logger.exception(
                        "The model file is corrupted or incompatible; retraining"
                    )
            if service is None:
                logger.info("No trained model found. Loading the corpus and training")
                service = MarkovService(model_path=model_path)
                texts = await run_in_threadpool(corpus_loader)
                if not texts:
                    raise ValueError("The training corpus is empty")
                await run_in_threadpool(service.train, texts, replace=True)
        except Exception:
            logger.exception("Could not prepare the model")
            raise
        app.state.service = service
        try:
            yield
        finally:
            app.state.service = None
            logger.info("Application stopped")

    app = FastAPI(title="Markov API", version="0.1.0", lifespan=lifespan)
    app.include_router(router, prefix="/api/v1")
    frontend = Path(__file__).resolve().parents[3] / "frontend"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app
