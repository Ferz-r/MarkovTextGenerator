import random
from collections import defaultdict

from rules import Tokenizer


class Markovka:
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: int,
        n_gramm: int = 3,
    ):
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._context_size = n_gramm

        self._texts: list[str] = []
        self._frequencies: dict[tuple[int, ...], dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def fit(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self._tokenizer.clear()
        self._rebuild_transitions()

    def update(self, texts: list[str]) -> None:
        if not texts:
            return
        self._texts.extend(texts)
        self._rebuild_transitions()

    def generate(self, prefix: str = "", topic: str | None = None) -> str:
        """Continue prefix using all texts or texts containing topic."""
        transitions = self._frequencies

        if topic is not None:
            topic = topic.strip().casefold()
            if not topic:
                raise ValueError("Topic must not be empty")

            topic_texts = []
            for text in self._texts:
                if topic in text.casefold():
                    topic_texts.append(text)

            if not topic_texts:
                raise ValueError(f"No training texts found for topic: {topic}")

            # Локальная таблица сохраняет общий словарь и обученную модель.
            transitions = self._count_transitions(topic_texts)

        tokens = [self._tokenizer.bos_id] * self._context_size

        prefix_tokens = self._tokenizer.encode(prefix)[
            1:-1
        ]  # [1:-1] чтобы убрать bos и eos из токенов промпта
        tokens.extend(prefix_tokens)
        generated_tokens = []

        for _ in range(self._max_length):
            context = tuple(tokens[-self._context_size :])
            next_token = self._sample_next_token(context, transitions)

            tokens.append(next_token)
            generated_tokens.append(next_token)

            if next_token == self._tokenizer.eos_id:
                break

        # Сохраняем исходное начало, даже если в нём есть неизвестные слова.
        return prefix + self._tokenizer.decode(generated_tokens)

    def _sample_next_token(
        self,
        context: tuple[int, ...],
        transitions: dict[tuple[int, ...], dict[int, int]],
    ) -> int:
        frequencies = transitions.get(context, {})

        tokens = []
        weights = []

        for next_token, count in frequencies.items():
            if next_token == self._tokenizer.unk_id:
                continue

            tokens.append(next_token)
            weights.append(count)

        if not tokens:
            return self._tokenizer.eos_id

        return random.choices(tokens, weights=weights)[0]

    def _rebuild_transitions(self) -> None:
        # Частота учитывает все тексты, включая предыдущие вызовы update().
        self._tokenizer.train(self._texts)
        self._frequencies = self._count_transitions(self._texts)

    def _count_transitions(
        self, texts: list[str]
    ) -> dict[tuple[int, ...], dict[int, int]]:
        transitions = defaultdict(lambda: defaultdict(int))

        # Старые UNK тоже пересчитываются, если слово теперь вошло в словарь.
        for text in texts:
            tokens = self._tokenizer.encode(text)

            tokens = [self._tokenizer.bos_id] * (self._context_size - 1) + tokens

            for i in range(self._context_size, len(tokens)):
                context = tuple(tokens[i - self._context_size : i])
                next_token = tokens[i]

                transitions[context][next_token] += 1

        return transitions

    @property
    def transitions(self) -> dict[tuple[int, ...], dict[int, int]]:
        return self._frequencies.copy()
