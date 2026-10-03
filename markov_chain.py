import random
from collections import defaultdict

from rules import Tokenizer


class Markovka:
    def __init__(self, tokenizer: Tokenizer, max_lengh: int):
        self._tokenizer = tokenizer
        self._max_lengh = max_lengh
        self._frequencies: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def fit(self, texts: list[str]) -> None:
        self._tokenizer.clear()
        self._tokenizer.train(texts)

        for text in texts:
            tokens = self._tokenizer.encode(text)

            # тут мы делаем окна, записываем пары токенов идущих друг за другом
            for current_token, next_token in zip(tokens, tokens[1:]):
                # тут мы в _frequencies записываем сколько next_token встречался с current_token
                self._frequencies[current_token][next_token] += 1
                # if current_token == 2 and next_token == 31:
                #     print(current_token, next_token)

    def _probabilities(self, token: int) -> dict[int, float]:
        freq = self._frequencies.get(token)

        if not freq:
            return {}

        # считаем общее количество встреченных токенов после текущего
        total = sum(freq.values())
        # возвращаем next_toten от token и вероятность его появления
        return {next_toten: count / total for next_toten, count in freq.items()}

    def _sample_next_token(self, token: int) -> int:
        probabilities = self._probabilities(token=token)

        if not probabilities:
            return self._tokenizer.eos_id

        tokens = list(probabilities.keys())
        weight = list(probabilities.values())

        next_token = random.choices(population=tokens, weights=weight)[0]

        return next_token

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

    @property
    def transitions(self) -> dict[int, dict[int, int]]:
        return self._frequencies.copy()
