import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSpec(BaseModel):
    alias: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    provider: Literal["demo", "openai", "azure", "ollama"]
    model: str = Field(min_length=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    embedding_provider: Literal["demo", "openai", "ollama"] = "demo"
    embedding_model: str = ""
    chroma_persist_directory: Path = Path("data/chroma")
    manuals_directory: Path = Path("examples/manuals")
    runs_directory: Path = Path("data/runs")
    top_k: int = Field(default=4, ge=1, le=12)
    min_similarity: float = Field(default=0.15, ge=0, le=1)
    chunk_size: int = Field(default=900, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0)
    models_json: str = '[{"alias":"demo","provider":"demo","model":"extractive-v1"}]'
    default_model: str = "demo"
    model_timeout_seconds: float = Field(default=45, gt=0, le=300)
    openai_api_key: str = Field(default="", repr=False)
    azure_openai_api_key: str = Field(default="", repr=False)
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = ""
    ollama_base_url: str = "http://localhost:11434"
    store_trace_content: bool = False

    @property
    def models(self) -> dict[str, ModelSpec]:
        specs = [ModelSpec.model_validate(item) for item in json.loads(self.models_json)]
        if not specs or len({s.alias for s in specs}) != len(specs):
            raise ValueError("MODELS_JSON must have unique aliases and at least one model")
        return {s.alias: s for s in specs}

    @model_validator(mode="after")
    def validate_config(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.default_model not in self.models:
            raise ValueError("DEFAULT_MODEL must be an alias in MODELS_JSON")
        if self.embedding_provider != "demo" and not self.embedding_model:
            raise ValueError("Set EMBEDDING_MODEL for live embeddings")
        return self
