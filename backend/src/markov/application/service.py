import logging
from pathlib import Path
from threading import RLock

from markov.domain.model import MarkovChain
from markov.domain.settings import ModelSettings, ModelStats, TokenizerKind
from markov.domain.tokenizers.base import Tokenizer
from markov.domain.tokenizers.character import CharacterTokenizer
from markov.domain.tokenizers.regex import RegexTokenizer
from markov.infrastructure.storage import read_snapshot, write_snapshot

logger = logging.getLogger("markov.service")


class MarkovService:
    """Serialize operations and publish model changes only after successful saving."""

    def __init__(
        self,
        min_frequency: int = 2,
        max_length: int = 300,
        n_gramm: int = 5,
        tokenizer_type: TokenizerKind = "regex",
        model_path: Path | None = None,
    ):
        self._settings = ModelSettings(
            tokenizer=tokenizer_type,
            n_gramm=n_gramm,
            min_frequency=min_frequency,
            max_length=max_length,
        )
        self._model = self._make_model(self._settings)
        self._lock = RLock()
        self._model_path = model_path

    @staticmethod
    def _make_model(settings: ModelSettings) -> MarkovChain:
        tokenizer_class = {"character": CharacterTokenizer, "regex": RegexTokenizer}[
            settings.tokenizer
        ]
        tokenizer = tokenizer_class(min_frequency=settings.min_frequency)
        return MarkovChain(
            tokenizer, max_length=settings.max_length, context_size=settings.n_gramm
        )

    @property
    def model(self) -> MarkovChain:
        """Expose the model for inspection; application operations go through the service."""
        return self._model

    @property
    def tokenizer(self) -> Tokenizer:
        return self._model.tokenizer

    def train(self, texts: list[str], replace: bool = False) -> ModelStats:
        if not texts or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ValueError("Training texts must be nonempty strings")
        with self._lock:
            logger.info("Training model: %s texts, replace=%s", len(texts), replace)
            corpus = list(texts) if replace else [*self._model.texts, *texts]
            candidate = self._make_model(self._settings)
            candidate.fit(corpus)
            self._persist(candidate, self._settings)
            self._model = candidate
            return self.stats()

    def generate(self, prefix: str, topic: str | None) -> str:
        with self._lock:
            return self._model.generate(prefix=prefix, topic=topic)

    def stats(self) -> ModelStats:
        with self._lock:
            return ModelStats(
                settings=self._settings.model_copy(),
                texts_count=self._model.texts_count,
                vocab_size=self.tokenizer.vocab_size,
                contexts_count=self._model.contexts_count,
            )

    def configure(self, settings: ModelSettings) -> ModelStats:
        with self._lock:
            rebuild = (
                settings.tokenizer != self._settings.tokenizer
                or settings.n_gramm != self._settings.n_gramm
                or settings.min_frequency != self._settings.min_frequency
            )
            if rebuild:
                candidate = self._make_model(settings)
                candidate.fit(list(self._model.texts))
                for topic in self._model.cached_topics:
                    candidate.prepare_topic(topic)
            else:
                candidate = self._model
            self._persist(candidate, settings)
            candidate.max_length = settings.max_length
            self._model = candidate
            self._settings = settings.model_copy()
            return self.stats()

    def _persist(self, model: MarkovChain, settings: ModelSettings) -> None:
        if self._model_path is not None:
            write_snapshot(
                self._model_path,
                {"settings": settings.model_dump(), **model.export_state()},
            )

    def save(self) -> None:
        with self._lock:
            self._persist(self._model, self._settings)

    @classmethod
    def load(cls, path: Path) -> "MarkovService":
        snapshot = read_snapshot(path)
        settings = ModelSettings.model_validate(snapshot["settings"])
        model = MarkovChain.from_state(snapshot, settings)
        service = cls(
            min_frequency=settings.min_frequency,
            max_length=settings.max_length,
            n_gramm=settings.n_gramm,
            tokenizer_type=settings.tokenizer,
            model_path=path,
        )
        service._model = model
        logger.info("Model loaded without training: %s", path)
        return service
