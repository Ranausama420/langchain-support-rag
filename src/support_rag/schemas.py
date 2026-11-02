from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    model: str | None = None

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if len(value.strip()) < 3:
            raise ValueError("Enter a question of at least three characters")
        return value.strip()


class CompareRequest(AskRequest):
    models: list[str] = Field(min_length=1, max_length=3)

    @field_validator("models")
    @classmethod
    def distinct(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Choose distinct model aliases")
        return value


class Evidence(BaseModel):
    id: str
    text: str
    source: str
    section: str
    similarity: float


class Claim(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    citation_ids: list[str] = Field(min_length=1, max_length=12)


class Draft(BaseModel):
    insufficient_evidence: bool
    claims: list[Claim] = Field(default_factory=list, max_length=10)


class Answer(BaseModel):
    run_id: str
    model: str
    status: Literal["answered", "insufficient_evidence", "error"]
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    sources: list[Evidence] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    latency_ms: float = 0
    usage: dict = Field(default_factory=dict)
    error: str | None = None
