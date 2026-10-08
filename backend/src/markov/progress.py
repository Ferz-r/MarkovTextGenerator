import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

_progress_listener: ContextVar[Callable[[dict], None] | None] = ContextVar(
    "progress_listener", default=None
)


@contextmanager
def listen_progress(listener: Callable[[dict], None]) -> Generator[None, None, None]:
    token = _progress_listener.set(listener)
    try:
        yield
    finally:
        _progress_listener.reset(token)


class TrainingProgress:
    """Report progress at 10% intervals without flooding the terminal."""

    def __init__(self, stage: str, total: int):
        self.stage = stage
        self.total = total
        self.started = monotonic()
        self.next_percent = 10
        self.next_ui_percent = 1
        self.logger = logging.getLogger("markov.training")
        self.logger.info("%s: 0/%s (%s%%)", stage, total, 0 if total else 100)
        self._publish(0)

    def advance(self, completed: int) -> None:
        percent = completed * 100 // self.total if self.total else 100
        if percent >= self.next_ui_percent or completed == self.total:
            self._publish(completed)
            self.next_ui_percent = percent + 1
        if percent >= self.next_percent or completed == self.total:
            self.logger.info(
                "%s: %s/%s (%s%%), %.1f сек.",
                self.stage,
                completed,
                self.total,
                percent,
                monotonic() - self.started,
            )
            self.next_percent = (percent // 10 + 1) * 10

    def _publish(self, completed: int) -> None:
        listener = _progress_listener.get()
        if listener is not None:
            listener(
                {
                    "stage": self.stage,
                    "completed": completed,
                    "total": self.total,
                    "percent": completed * 100 // self.total if self.total else 100,
                    "elapsed": round(monotonic() - self.started, 1),
                }
            )
