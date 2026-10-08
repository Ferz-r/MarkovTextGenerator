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
        return [
            line.strip() for text in texts for line in text.splitlines() if line.strip()
        ]
