import re
from collections import Counter

from rules import Tokenizer

# import tiktoken


class RegexTokenizer(Tokenizer):
    """Tokenize text into words, numbers, whitespace and punctuation."""

    _token_pattern = re.compile(
        r"\s+|\d+(?:[.,]\d+)*|[^\W_]+(?:[-'’][^\W_]+)*|[^\w\s]|_+",
        flags=re.UNICODE,
    )

    def __init__(
        self,
        min_frequency: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")

        self._min_frequency = min_frequency

    def _split(self, text: str) -> list[str]:
        return self._token_pattern.findall(text)

    def train(self, texts: list[str]) -> None:
        frequencies = Counter(piece for text in texts for piece in self._split(text))

        # Sorting makes token IDs reproducible for the same training corpus.
        pieces = sorted(
            (
                piece
                for piece, frequency in frequencies.items()
                if frequency >= self._min_frequency
                and piece not in self._piece_to_token
            ),
            key=lambda piece: (-frequencies[piece], piece),
        )

        for piece in pieces:
            token_id = self._next_token_id()
            self._piece_to_token[piece] = token_id
            self._token_to_piece[token_id] = piece

    def encode(self, text: str) -> list[int]:
        tokens = [self._bos_id]
        tokens.extend(
            self._piece_to_token.get(piece, self._unk_id) for piece in self._split(text)
        )
        tokens.append(self._eos_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        pieces = []

        for token in tokens:
            if token in (self._bos_id, self._eos_id):
                continue
            if token not in self._token_to_piece:
                raise ValueError(f"Unknown token ID: {token}")
            pieces.append(self._token_to_piece[token])

        return "".join(pieces)


class BPETokenizer(Tokenizer):
    def __init__(self, vocab_size: int = 100, **kwargs):
        super().__init__(**kwargs)

        self._target_vocab_size = vocab_size
        self._merges: dict[tuple[str, str], int] = {}

    def _get_pairs(self, symbols: list[str]) -> set[tuple[str, str]]:

        pairs = set()

        for i in range(len(symbols) - 1):
            pairs.add((symbols[i], symbols[i + 1]))

        return pairs

    def _get_most_frequent_pair(self, words: list[list[str]]):
        counter = Counter()
        for word in words:
            pairs = self._get_pairs(word)

            for pair in pairs:
                counter[pair] += 1

        if not counter:
            return None

        return counter.most_common(1)[0][0]

    def _merge_pair(self, symbols: list[str], pair: tuple[str, str]):
        result = []
        i = 0
        while i < len(symbols):
            if (
                i < len(symbols) - 1
                and symbols[i] == pair[0]
                and symbols[i + 1] == pair[1]
            ):
                result.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                result.append(symbols[i])
                i += 1

        return result

    def train(self, texts: list[str]):
        words = []

        for text in texts:
            print(1)
            for word in text.split():
                symbols = list(word)

                words.append(symbols)

                for symbol in symbols:
                    if symbol not in self._piece_to_token:
                        token_id = self._next_token_id()

                        self._piece_to_token[symbol] = token_id

                        self._token_to_piece[token_id] = symbol

        while self.vocab_size < self._target_vocab_size:
            pair = self._get_most_frequent_pair(words)
            if pair is None:
                break
            rank = len(self._merges)
            self._merges[pair] = rank
            new_token = "".join(pair)

            if new_token not in self._piece_to_token:
                token_id = self._next_token_id()

                self._piece_to_token[new_token] = token_id

                self._token_to_piece[token_id] = new_token
            for i, word in enumerate(words):
                words[i] = self._merge_pair(word, pair)

    def _apply_bpe(self, symbols: list[str]):
        while True:
            pairs = self._get_pairs(symbols)
            if not pairs:
                break

            available = [pair for pair in pairs if pair in self._merges]
            if not available:
                break

            best_pair = min(available, key=lambda x: self._merges[x])
            symbols = self._merge_pair(symbols, best_pair)

        return symbols

    def encode(self, text: str) -> list[int]:
        tokens = [self._bos_id]
        for word in text.split():
            symbols = list(word)
            symbols = self._apply_bpe(symbols)
            for symbol in symbols:
                tokens.append(self._piece_to_token.get(symbol, self._unk_id))
        tokens.append(self._eos_id)

        return tokens

    def decode(self, tokens: list[int]) -> str:
        pieces = []
        for token in tokens:
            if token in (self._bos_id, self._eos_id):
                continue
            pieces.append(self._token_to_piece[token])

        return " ".join(pieces)
