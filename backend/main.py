import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from markov.api.routes import router
from markov.corpus import load_corpus
from markov.logging import configure_logging
from markov.service import MarkovService
from markov.storage import MODEL_FILE

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
                        "Файл модели повреждён или несовместим, обучаем заново"
                    )
            if service is None:
                logger.info("Обученная модель отсутствует. Загружаем корпус и обучаем")
                service = MarkovService(model_path=model_path)
                texts = await run_in_threadpool(corpus_loader)
                if not texts:
                    raise ValueError("Корпус обучения пуст")
                await run_in_threadpool(service.train, texts, replace=True)
        except Exception:
            logger.exception("Не удалось подготовить модель")
            raise
        app.state.service = service
        try:
            yield
        finally:
            app.state.service = None
            logger.info("Приложение остановлено")

    app = FastAPI(title="Markov API", version="0.1.0", lifespan=lifespan)
    app.include_router(router, prefix="/api/v1")
    frontend = Path(__file__).resolve().parent.parent / "frontend"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app(model_path=MODEL_FILE)


def main() -> None:
    configure_logging()
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
