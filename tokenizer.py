import re

from rules import Tokenizer

# import tiktoken


class RegexTokenizer(Tokenizer):
    """Tokenize text into words, numbers, whitespace and punctuation."""

    _token_pattern = re.compile(
        r"\s+|\d+(?:[.,]\d+)*|[^\W_]+(?:[-'’][^\W_]+)*|[^\w\s]|_+",
        flags=re.UNICODE,
    )

    def _split(self, text: str) -> list[str]:
        return self._token_pattern.findall(text)
