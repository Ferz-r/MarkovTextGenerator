from abc import ABC, abstractmethod
from typing import Protocol


class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


class Tokenizer(ABC):
    def __init__(
        self,
        bos_str: str = "<BOS>",
        eos_str: str = "<EOS>",
        unk_str: str = "<UNK>",
        bos_id: int = 0,
        eos_id: int = 1,
        unk_id: int = 2,
    ):
        if len({bos_str, eos_str, unk_str}) != 3:
            raise ValueError("Special token strings must be unique")
        if len({bos_id, eos_id, unk_id}) != 3:
            raise ValueError("Special token IDs must be unique")

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

        self._next_id = 3

    def _next_token_id(self) -> int:
        token_id = self._next_id
        self._next_id += 1
        return token_id

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass

    @property
    def vocab_size(self):
        return len(self._piece_to_token)
