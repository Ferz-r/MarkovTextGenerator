import random
from collections import defaultdict

from rules import Tokenizer


class Markovka:
    def __init__(self, tokenizer: Tokenizer, max_lengh: int):
        self._tokenizer = tokenizer
        self._max_lengh = max_lengh

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

    def generate(self) -> str:
        tokens = [self._tokenizer.bos_id]

        curent_token = self._tokenizer.bos_id

        for _ in range(self._max_lengh):
            next_token = self._sample_next_token(curent_token)

            tokens.append(next_token)

            if next_token == self._tokenizer.eos_id:
                break

            curent_token = next_token

        output = self._tokenizer.decode(tokens=tokens)

        return output

    def _rebuild_transitions(self) -> None:
        # Частота учитывает все тексты, включая предыдущие вызовы update().
        self._tokenizer.train(self._texts)
        self._frequencies.clear()

        # Старые UNK тоже пересчитываются, если слово теперь вошло в словарь.
        for text in self._texts:
            tokens = self._tokenizer.encode(text)

            for current_token, next_token in zip(tokens, tokens[1:]):
                self._frequencies[current_token][next_token] += 1

    def _sample_next_token(self, token: int) -> int:
        frequencies = self._frequencies.get(token, {})

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

    @property
    def transitions(self) -> dict[int, dict[int, int]]:
        return self._frequencies.copy()
