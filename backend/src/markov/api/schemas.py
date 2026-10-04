from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    prefix: str = Field(default="", max_length=10000)
    topic: str | None = Field(default=None, min_length=1, max_length=200)


class GenerateResponse(BaseModel):
    text: str


class TrainRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    replace: bool = False

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        if any(not text.strip() for text in texts):
            raise ValueError("Тексты должны быть непустыми")
        return texts


class ModelSettings(BaseModel):
    n_gramm: int = Field(ge=1, le=50, strict=True)
    min_frequency: int = Field(ge=1, le=100000, strict=True)
    max_length: int = Field(ge=1, le=10000, strict=True)


class ModelStats(BaseModel):
    texts_count: int
    vocab_size: int
    settings: ModelSettings
    contexts_count: int
