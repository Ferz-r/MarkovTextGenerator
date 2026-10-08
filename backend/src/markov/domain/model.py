import random
from collections import OrderedDict

from markov.domain.settings import ModelSettings
from markov.domain.tokenizers.base import Tokenizer
from markov.domain.tokenizers.character import CharacterTokenizer
from markov.domain.tokenizers.regex import RegexTokenizer
from markov.domain.transitions import TransitionIndex
from markov.progress import TrainingProgress


class MarkovChain:
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: int = 3000,
        context_size: int = 3,
    ):
        if context_size < 1:
            raise ValueError("context_size must be at least 1")
        self._tokenizer = tokenizer
        self.max_length = max_length
        self._context_size = context_size

        self._texts: list[str] = []
        # Limit the cache: topic tables can use substantial memory.
        self._topic_transitions: OrderedDict[str, TransitionIndex] = OrderedDict()
        self._frequencies = TransitionIndex(
            {}, sorted(tokenizer.token_to_piece), context_size
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
        ]  # [1:-1] removes bos and eos from the prompt tokens
        tokens.extend(prefix_tokens)
        generated_tokens = []

        for _ in range(self._max_length):
            context = tuple(tokens[-self._context_size :])
            next_token = self._sample_next_token(context, transitions)

            tokens.append(next_token)
            generated_tokens.append(next_token)

            if next_token == self._tokenizer.eos_id:
                break

        # Preserve the original prefix even if it contains unknown words.
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
            topic_texts, stage=f'Transitions for topic "{topic}"'
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
        # Start with the full context, then drop the oldest tokens.
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
        # Frequency includes all texts, including previous update() calls.
        self._tokenizer.train(self._texts)
        self._frequencies = self._count_transitions(self._texts)

    def _count_transitions(
        self, texts: list[str], stage: str = "Building transitions"
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
        indexing = TrainingProgress(f"Indexing: {stage}", len(records))
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

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(self._texts)

    @property
    def texts_count(self) -> int:
        return len(self._texts)

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def export_state(self) -> dict:
        return {
            "vocabulary": self._tokenizer.piece_to_token,
            "texts": list(self._texts),
            "transitions": self._frequencies.export_state(),
            "topics": {
                topic: index.export_state()
                for topic, index in self._topic_transitions.items()
            },
        }

    @classmethod
    def from_state(cls, state: dict, settings: ModelSettings) -> "MarkovChain":
        texts = state["texts"]
        if (
            not isinstance(texts, list)
            or not texts
            or any(not isinstance(text, str) or not text.strip() for text in texts)
        ):
            raise ValueError("Invalid training corpus in model snapshot")
        tokenizer_class = {"character": CharacterTokenizer, "regex": RegexTokenizer}[
            settings.tokenizer
        ]
        tokenizer = tokenizer_class(min_frequency=settings.min_frequency)
        tokenizer.restore_vocabulary(state["vocabulary"])
        model = cls(tokenizer, settings.max_length, settings.n_gramm)
        token_ids = sorted(tokenizer.token_to_piece)

        def restore_index(data: dict) -> TransitionIndex:
            index = TransitionIndex.from_state(data)
            if index.context_size != settings.n_gramm or index.token_ids != tuple(
                token_ids
            ):
                raise ValueError("Snapshot settings and transitions do not match")
            return index

        model._frequencies = restore_index(state["transitions"])
        topics = state["topics"]
        if not isinstance(topics, dict) or len(topics) > 2:
            raise ValueError("Invalid topic cache")
        for topic, data in topics.items():
            if (
                not isinstance(topic, str)
                or not topic
                or topic != topic.strip().casefold()
            ):
                raise ValueError("Invalid topic snapshot")
            model._topic_transitions[topic] = restore_index(data)
        model._texts = list(texts)
        return model
