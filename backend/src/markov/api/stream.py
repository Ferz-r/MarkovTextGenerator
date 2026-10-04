import json
import logging
from collections.abc import Callable, Iterator
from queue import Empty, Queue
from threading import Event, Thread

from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from markov.api.schemas import ModelStats
from markov.file_corpus import CorpusFileError
from markov.logging import listen_progress

logger = logging.getLogger("markov.api")


def training_stream(
    operation: Callable[[], ModelStats], cleanup: Callable[[], None] | None = None
) -> StreamingResponse:
    started = Event()

    def events() -> Iterator[str]:
        queue: Queue[tuple[str, dict]] = Queue()

        def worker() -> None:
            try:
                with listen_progress(
                    lambda progress: queue.put(("progress", progress))
                ):
                    result = operation()
                queue.put(("complete", result.model_dump()))
            except CorpusFileError as exc:
                queue.put(("error", {"message": str(exc)}))
            except Exception:
                logger.exception("Ошибка обучения в потоковом запросе")
                queue.put(
                    (
                        "error",
                        {
                            "message": "Не удалось завершить обучение. Подробности — в терминале сервера."
                        },
                    )
                )

            finally:
                if cleanup is not None:
                    cleanup()

        started.set()
        Thread(target=worker, daemon=True, name="markov-training").start()
        yield "event: start\ndata: {}\n\n"
        while True:
            try:
                event, data = queue.get(timeout=10)
            except Empty:
                yield ": heartbeat\n\n"
                continue
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            if event in ("complete", "error"):
                return

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(
            lambda: cleanup() if cleanup is not None and not started.is_set() else None
        ),
    )
