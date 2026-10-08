import logging
from pathlib import Path
from threading import RLock

from markov.api.schemas import ModelSettings, ModelStats, TokenizerKind
from markov.markov_chain import Markovka
from markov.rules import Tokenizer
from markov.storage import read_snapshot, write_snapshot
from markov.tokenizer import CharacterTokenizer, RegexTokenizer
from markov.transitions import TransitionIndex

logger = logging.getLogger("markov.service")


class MarkovService:
    def __init__(
        self,
        min_frequency: int = 3,
        max_length: int = 100,
        n_gramm: int = 15,
        tokenizer_type: TokenizerKind = "character",
        model_path: Path | None = None,
    ):
        self._settings = ModelSettings(
            tokenizer=tokenizer_type,
            n_gramm=n_gramm,
            min_frequency=min_frequency,
            max_length=max_length,
        )
        self.tokenizer = self._make_tokenizer(self._settings)
        self.model = Markovka(self.tokenizer, max_length=max_length, n_gramm=n_gramm)
        self._lock = RLock()
        self._texts_count = 0
        self._texts: list[str] = []
        self._model_path = model_path

    @staticmethod
    def _make_tokenizer(settings: ModelSettings) -> Tokenizer:
        tokenizer_class = {"character": CharacterTokenizer, "regex": RegexTokenizer}[
            settings.tokenizer
        ]
        return tokenizer_class(min_frequency=settings.min_frequency)

    def train(self, texts: list[str], replace: bool = False) -> ModelStats:
        with self._lock:
            logger.info("Обучение модели: %s текстов, replace=%s", len(texts), replace)
            if replace:
                self.model.fit(texts)
                self._texts = list(texts)
                self._texts_count = len(texts)
            else:
                self.model.update(texts)
                self._texts.extend(texts)
                self._texts_count += len(texts)
            self.save()
            stats = self.stats()
            logger.info(
                "Модель готова: словарь=%s, контексты=%s",
                stats.vocab_size,
                stats.contexts_count,
            )
            return stats

    def generate(self, prefix: str, topic: str | None) -> str:
        with self._lock:
            return self.model.generate(prefix=prefix, topic=topic)

    def stats(self) -> ModelStats:
        with self._lock:
            return ModelStats(
                settings=self._settings.model_copy(),
                texts_count=self._texts_count,
                vocab_size=self.tokenizer.vocab_size,
                contexts_count=self.model.contexts_count,
            )

    def configure(self, settings: ModelSettings) -> ModelStats:
        with self._lock:
            rebuild = (
                settings.tokenizer != self._settings.tokenizer
                or settings.n_gramm != self._settings.n_gramm
                or settings.min_frequency != self._settings.min_frequency
            )
            if rebuild:
                logger.info(
                    "Пересчёт модели: tokenizer=%s, n_gramm=%s, min_frequency=%s, max_length=%s",
                    settings.tokenizer,
                    settings.n_gramm,
                    settings.min_frequency,
                    settings.max_length,
                )
                tokenizer = self._make_tokenizer(settings)
                model = Markovka(
                    tokenizer, max_length=settings.max_length, n_gramm=settings.n_gramm
                )
                model.fit(self._texts)
                # Готовим уже используемые темы до завершения пересчёта настроек.
                for topic in self.model.cached_topics:
                    model.prepare_topic(topic)
                self.tokenizer = tokenizer
                self.model = model
            else:
                self.model.max_length = settings.max_length
            self._settings = settings.model_copy()
            self.save()
            return self.stats()

    def save(self) -> None:
        with self._lock:
            if self._model_path is None:
                return
            write_snapshot(
                self._model_path,
                {
                    "settings": self._settings.model_dump(),
                    "vocabulary": self.tokenizer.piece_to_token,
                    "texts": self._texts,
                    "transitions": self.model._frequencies.export_state(),
                    "topics": {
                        topic: index.export_state()
                        for topic, index in self.model._topic_transitions.items()
                    },
                },
            )

    @classmethod
    def load(cls, path: Path) -> "MarkovService":
        snapshot = read_snapshot(path)
        settings = ModelSettings.model_validate(snapshot["settings"])
        texts = snapshot["texts"]
        if (
            not isinstance(texts, list)
            or not texts
            or any(not isinstance(text, str) for text in texts)
        ):
            raise ValueError("Invalid training corpus in model snapshot")
        service = cls(
            min_frequency=settings.min_frequency,
            max_length=settings.max_length,
            n_gramm=settings.n_gramm,
            tokenizer_type=settings.tokenizer,
            model_path=path,
        )
        service.tokenizer.restore_vocabulary(snapshot["vocabulary"])
        model = service.model
        model._frequencies = TransitionIndex.from_state(snapshot["transitions"])
        if (
            model._frequencies._context_size != settings.n_gramm
            or model._frequencies._token_ids != sorted(service.tokenizer.token_to_piece)
        ):
            raise ValueError("Snapshot settings and transitions do not match")
        for topic, state in snapshot["topics"].items():
            if not isinstance(topic, str):
                raise TypeError("Invalid topic snapshot")
            index = TransitionIndex.from_state(state)
            if (
                index._context_size != settings.n_gramm
                or index._token_ids != model._frequencies._token_ids
            ):
                raise ValueError("Invalid topic transition index")
            model._topic_transitions[topic] = index
        if len(model._topic_transitions) > 2:
            raise ValueError("Invalid topic cache size")
        model._texts = list(texts)
        service._texts = texts
        service._texts_count = len(texts)
        logger.info("Модель загружена без обучения: %s", path)
        return service
