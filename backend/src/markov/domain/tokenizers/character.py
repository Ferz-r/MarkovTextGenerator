from markov.domain.tokenizers.base import Tokenizer


class CharacterTokenizer(Tokenizer):
    def _split(self, text: str) -> list[str]:
        return list(text)
