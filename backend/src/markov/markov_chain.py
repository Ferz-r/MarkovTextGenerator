import random
from collections import OrderedDict

from markov.logging import TrainingProgress
from markov.rules import Tokenizer
from markov.transitions import TransitionIndex


class Markovka:
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: int = 3000,
        n_gramm: int = 3,
    ):
        if n_gramm < 1:
            raise ValueError("n_gramm must be at least 1")
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._context_size = n_gramm

        self._texts: list[str] = []
        # Ограничиваем кеш: тематические таблицы могут занимать много памяти.
        self._topic_transitions: OrderedDict[str, TransitionIndex] = OrderedDict()
        self._frequencies = TransitionIndex(
            {}, sorted(tokenizer.token_to_piece), n_gramm
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
            transitions = self.prepare_topic(topic)

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

    def prepare_topic(self, topic: str) -> TransitionIndex:
        topic = topic.strip().casefold()
        if not topic:
            raise ValueError("Topic must not be empty")
        if topic in self._topic_transitions:
            self._topic_transitions.move_to_end(topic)
            return self._topic_transitions[topic]
        topic_texts = [text for text in self._texts if topic in text.casefold()]
        if not topic_texts:
            raise ValueError(f"No training texts found for topic: {topic}")
        transitions = self._count_transitions(
            topic_texts, stage=f"Переходы для темы «{topic}»"
        )
        self._topic_transitions[topic] = transitions
        if len(self._topic_transitions) > 2:
            self._topic_transitions.popitem(last=False)
        return transitions

    @property
    def cached_topics(self) -> tuple[str, ...]:
        return tuple(self._topic_transitions)

    def _sample_next_token(
        self,
        context: tuple[int, ...],
        transitions: TransitionIndex,
    ) -> int:
        # Начинаем с полного контекста, затем убираем самые старые токены.
        for size in range(len(context), 0, -1):
            shorter_context = context[-size:]
            frequencies = transitions.get(shorter_context, {})

            tokens = []
            weights = []

            for next_token, count in frequencies.items():
                if next_token == self._tokenizer.unk_id:
                    continue

                tokens.append(next_token)
                weights.append(count)

            if tokens:
                return random.choices(tokens, weights=weights)[0]

        return self._tokenizer.eos_id

    def _rebuild_transitions(self) -> None:
        self._topic_transitions.clear()
        # Частота учитывает все тексты, включая предыдущие вызовы update().
        self._tokenizer.train(self._texts)
        self._frequencies = self._count_transitions(self._texts)

    def _count_transitions(
        self, texts: list[str], stage: str = "Построение переходов"
    ) -> TransitionIndex:
        token_ids = sorted(self._tokenizer.token_to_piece)
        codes = {token: code for code, token in enumerate(token_ids)}
        bits = max(1, (len(token_ids) - 1).bit_length())
        oldest_shift = bits * (self._context_size - 1)
        bos = codes[self._tokenizer.bos_id]
        initial_context = 0
        for _ in range(self._context_size):
            initial_context = (initial_context << bits) | bos
        records: dict[int, int] = {}
        progress = TrainingProgress(stage, len(texts))
        for index, text in enumerate(texts, start=1):
            context = initial_context
            # BOS already fills the context. Characters and EOS are targets.
            for token in self._tokenizer.encode(text)[1:]:
                code = codes[token]
                record = (context << bits) | code
                records[record] = records.get(record, 0) + 1
                context = (code << oldest_shift) | (context >> bits)
            progress.advance(index)
        indexing = TrainingProgress(f"Индексирование: {stage}", len(records))
        transitions = TransitionIndex(records, token_ids, self._context_size)
        indexing.advance(len(records))
        return transitions

    @property
    def transitions(self) -> dict[tuple[int, ...], dict[int, int]]:
        return self._frequencies.copy()

    @property
    def contexts_count(self) -> int:
        return len(self._frequencies)

    @property
    def max_length(self) -> int:
        return self._max_length

    @max_length.setter
    def max_length(self, value: int) -> None:
        if value < 1:
            raise ValueError("max_length must be at least 1")
        self._max_length = value
