from abc import ABC, abstractmethod
from collections import Counter
from typing import Protocol

from markov.logging import TrainingProgress


class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...
    def train(self, texts: list[str]) -> None: ...


class Tokenizer(ABC):
    def __init__(
        self,
        bos_str: str = "<BOS>",
        eos_str: str = "<EOS>",
        unk_str: str = "<UNK>",
        bos_id: int = 0,
        eos_id: int = 1,
        unk_id: int = 2,
        min_frequency: int = 1,
    ):

        if len({bos_str, eos_str, unk_str}) != 3:
            raise ValueError("Special token strings must be unique")
        if len({bos_id, eos_id, unk_id}) != 3:
            raise ValueError("Special token IDs must be unique")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")

        self._bos_str = bos_str
        self._eos_str = eos_str
        self._unk_str = unk_str

        self._bos_id = bos_id
        self._eos_id = eos_id
        self._unk_id = unk_id

        self._piece_to_token: dict[str, int] = {
            bos_str: bos_id,
            eos_str: eos_id,
            unk_str: unk_id,
        }

        self._token_to_piece: dict[int, str] = {
            bos_id: bos_str,
            eos_id: eos_str,
            unk_id: unk_str,
        }
        self._min_frequency = min_frequency

        self._next_id = max(bos_id, eos_id, unk_id) + 1

    @abstractmethod
    def _split(self, text: str) -> list[str]: ...

    def clear(self) -> None:
        """Reset the vocabulary, keeping special tokens and configuration."""
        self._piece_to_token = {
            self._bos_str: self._bos_id,
            self._eos_str: self._eos_id,
            self._unk_str: self._unk_id,
        }
        self._token_to_piece = {
            self._bos_id: self._bos_str,
            self._eos_id: self._eos_str,
            self._unk_id: self._unk_str,
        }
        self._next_id = max(self._bos_id, self._eos_id, self._unk_id) + 1

    def train(self, texts: list[str]) -> None:
        frequencies = Counter()

        progress = TrainingProgress("Подсчёт частот словаря", len(texts))
        for index, text in enumerate(texts, start=1):
            pieces = self._split(text)
            frequencies.update(pieces)
            progress.advance(index)

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

        progress = TrainingProgress("Построение словаря", len(pieces))
        for index, piece in enumerate(pieces, start=1):
            token_id = self._next_token_id()
            self._piece_to_token[piece] = token_id
            self._token_to_piece[token_id] = piece
            progress.advance(index)

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

    def _next_token_id(self) -> int:
        token_id = self._next_id
        self._next_id += 1
        return token_id

    @property
    def vocab_size(self):
        return len(self._piece_to_token)

    @property
    def piece_to_token(self) -> dict[str, int]:
        return self._piece_to_token.copy()

    @property
    def token_to_piece(self) -> dict[int, str]:
        return self._token_to_piece.copy()

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def unk_id(self) -> int:
        return self._unk_id

    def restore_vocabulary(self, vocabulary: dict[str, int]) -> None:
        special = {
            self._bos_str: self._bos_id,
            self._eos_str: self._eos_id,
            self._unk_str: self._unk_id,
        }
        if (
            not isinstance(vocabulary, dict)
            or any(
                not isinstance(piece, str) or type(token) is not int
                for piece, token in vocabulary.items()
            )
            or any(vocabulary.get(piece) != token for piece, token in special.items())
            or len(set(vocabulary.values())) != len(vocabulary)
        ):
            raise ValueError("Invalid tokenizer vocabulary")
        self._piece_to_token = dict(vocabulary)
        self._token_to_piece = {token: piece for piece, token in vocabulary.items()}
        self._next_id = max(vocabulary.values()) + 1
