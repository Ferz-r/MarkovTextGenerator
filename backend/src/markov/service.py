import logging
from threading import RLock

from markov.api.schemas import ModelSettings, ModelStats
from markov.markov_chain import Markovka
from markov.tokenizer import CharacterTokenizer

logger = logging.getLogger("markov.service")


class MarkovService:
    def __init__(
        self,
        min_frequency: int = 3,
        max_length: int = 100,
        n_gramm: int = 15,
    ):
        self.tokenizer = CharacterTokenizer(min_frequency=min_frequency)
        self.model = Markovka(self.tokenizer, max_length=max_length, n_gramm=n_gramm)
        self._lock = RLock()
        self._texts_count = 0
        self._texts: list[str] = []
        self._settings = ModelSettings(
            n_gramm=n_gramm, min_frequency=min_frequency, max_length=max_length
        )

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
                settings.n_gramm != self._settings.n_gramm
                or settings.min_frequency != self._settings.min_frequency
            )
            if rebuild:
                logger.info(
                    "Пересчёт модели: n_gramm=%s, min_frequency=%s, max_length=%s",
                    settings.n_gramm,
                    settings.min_frequency,
                    settings.max_length,
                )
                tokenizer = CharacterTokenizer(min_frequency=settings.min_frequency)
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
            return self.stats()
