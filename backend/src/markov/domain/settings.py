from typing import Literal

from pydantic import BaseModel, Field

TokenizerKind = Literal["character", "regex"]


class ModelSettings(BaseModel):
    tokenizer: TokenizerKind = "regex"
    n_gramm: int = Field(ge=1, le=50, strict=True)
    min_frequency: int = Field(ge=1, le=100000, strict=True)
    max_length: int = Field(ge=1, le=10000, strict=True)


class ModelStats(BaseModel):
    texts_count: int
    vocab_size: int
    settings: ModelSettings
    contexts_count: int
