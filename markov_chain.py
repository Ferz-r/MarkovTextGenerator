import random
from collections import defaultdict

from rules import Tokenizer


class Markovka:
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_lengh: int,
        n_gramm: int = 3,
    ):
        self._tokenizer = tokenizer
        self._max_lengh = max_lengh
        self._context_size = n_gramm

        self._texts: list[str] = []
        self._frequencies: dict[int, dict[int, int]] = defaultdict(
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

    def generate(self, prompt: str = "") -> str:
        tokens = [self._tokenizer.bos_id] * self._context_size

        prompt_tokens = self._tokenizer.encode(prompt)[
            1:-1
        ]  # [1:-1] чтобы убрать bos и eos из токенов промпта
        tokens.extend(prompt_tokens)
        generated_tokens = []

        for _ in range(self._max_lengh):
            context = tuple(tokens[-self._context_size :])
            next_token = self._sample_next_token(context)

            tokens.append(next_token)
            generated_tokens.append(next_token)

            if next_token == self._tokenizer.eos_id:
                break

        # Сохраняем исходный промпт, даже если в нём есть неизвестные слова.
        return prompt + self._tokenizer.decode(generated_tokens)

    def _sample_next_token(self, context: tuple[int, ...]) -> int:
        frequencies = self._frequencies.get(context, {})
        # print("frequecncies", frequencies)

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
        self._frequencies.clear()

        # Старые UNK тоже пересчитываются, если слово теперь вошло в словарь.
        for text in self._texts:
            tokens = self._tokenizer.encode(text)

            tokens = [self._tokenizer.bos_id] * (self._context_size - 1) + tokens

            for i in range(self._context_size, len(tokens)):
                context = tuple(tokens[i - self._context_size : i])
                next_token = tokens[i]

                self._frequencies[context][next_token] += 1

    @property
    def transitions(self) -> dict[int, dict[int, int]]:
        return self._frequencies.copy()
