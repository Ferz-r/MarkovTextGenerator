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
        bos_id: int = 0,
        eos_id: int = 1,
    ):
        self._bos_str = bos_str
        self._eos_str = eos_str

        self._bos_id = bos_id
        self._eos_id = eos_id

        self._piece_to_token: dict[str, int] = {
            bos_str: bos_id,
            eos_str: eos_id,
        }

        self._token_to_piece: dict[int, str] = {
            bos_id: bos_str,
            eos_id: eos_str,
        }

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass

    @property
    def vocab_size(self):
        return len(self._piece_to_token)
