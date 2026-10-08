import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from markov.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    TrainRequest,
)
from markov.api.stream import training_stream
from markov.application.service import MarkovService
from markov.domain.settings import ModelSettings, ModelStats
from markov.infrastructure.file_corpus import load_file_texts

router = APIRouter()


def get_service(request: Request) -> MarkovService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Модель ещё не загружена")
    return service


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    get_service(request)
    return {"status": "ok"}


@router.get("/model", response_model=ModelStats)
def model_stats(request: Request) -> ModelStats:
    return get_service(request).stats()


@router.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest, request: Request) -> GenerateResponse:
    try:
        text = get_service(request).generate(payload.prefix, payload.topic)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return GenerateResponse(text=text)


@router.post("/train", response_model=ModelStats)
def train(payload: TrainRequest, request: Request) -> ModelStats:
    return get_service(request).train(payload.texts, replace=payload.replace)


@router.post("/settings", response_model=ModelStats)
def configure(payload: ModelSettings, request: Request) -> ModelStats:
    return get_service(request).configure(payload)


@router.post("/settings/stream")
def configure_stream(payload: ModelSettings, request: Request) -> StreamingResponse:
    service = get_service(request)
    return training_stream(lambda: service.configure(payload))


@router.post("/train/stream")
def train_stream(payload: TrainRequest, request: Request) -> StreamingResponse:
    service = get_service(request)
    return training_stream(
        lambda: service.train(payload.texts, replace=payload.replace)
    )


@router.post("/train/file/stream")
async def train_file_stream(
    request: Request,
    file: Annotated[UploadFile, File()],
    replace: Annotated[bool, Form()] = False,
    texts: Annotated[str, Form()] = "[]",
) -> StreamingResponse:
    service = get_service(request)
    if not (file.filename or "").lower().endswith(".txt"):
        raise HTTPException(status_code=422, detail="Выберите файл с расширением .txt.")
    try:
        extra_texts = json.loads(texts)
        if not isinstance(extra_texts, list) or any(
            not isinstance(text, str) or not text.strip() for text in extra_texts
        ):
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422, detail="Некорректные дополнительные тексты."
        ) from None

    def save() -> Path:
        with NamedTemporaryFile(suffix=".txt", delete=False) as temporary:
            path = Path(temporary.name)
            try:
                shutil.copyfileobj(file.file, temporary)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        return path

    try:
        path = await run_in_threadpool(save)
    finally:
        await file.close()

    def operation() -> ModelStats:
        corpus = load_file_texts(path)
        if extra_texts:
            corpus.extend(TrainRequest(texts=extra_texts).texts)
        return service.train(corpus, replace=replace)

    return training_stream(operation, cleanup=lambda: path.unlink(missing_ok=True))
