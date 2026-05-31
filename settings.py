from __future__ import annotations

import json
from typing import Any, Iterable

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    # Load defaults from .env in the project root.
    model_config = SettingsConfigDict(env_file=".env")

    # Core service configuration.
    database_url: str = Field(..., description="SQLAlchemy database URL")
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    openrouter_model_name: str = Field(..., description="Primary OpenRouter model")
    openrouter_fallback_models: list[str] = Field(
        default_factory=list, description="Fallback models for OpenRouter"
    )
    jina_api_key: str = Field(..., description="Jina API key")
    jina_base_url: str = Field(
        default="https://api.jina.ai/v1", description="Jina API base URL"
    )
    embedding_model_name: str = Field(..., description="Embedding model for ingest/query")
    embedding_dim: int = Field(..., description="Embedding vector dimension")
    rrf_vector_weight: float = Field(
        default=0.3, description="RRF weight for vector ranking"
    )
    rrf_bm25_weight: float = Field(
        default=0.7, description="RRF weight for BM25 ranking"
    )
    ingest_years: int = Field(
        default=2, description="Number of years of data to keep during ingestion"
    )
    ingest_csv_path: str = Field(
        default="hdb_search_agents/data/ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv",
        description="Default CSV path for ingestion",
    )
    braintrust_api_key: str | None = Field(
        default=None, description="BrainTrust API key for tracing (optional)"
    )
    braintrust_parent: str | None = Field(
        default=None, description="BrainTrust parent project identifier (optional)"
    )
    judge_openrouter_model_name: str = Field(
        default="z-ai/glm-5", description="OpenRouter model used as LLM judge in eval harness"
    )
    eval_project: str = Field(
        default="hdb-compare-agents-eval",
        description="Braintrust project name for the eval harness",
    )
    # Relative paths are resolved from the project root at runtime.
    # Absolute paths are used as-is.
    eval_dataset_path: str = Field(
        default="evals/datasets/hdb_compare_benchmark.yaml",
        description="Path to benchmark YAML dataset used by evals/run.py",
    )

    @field_validator("openrouter_fallback_models", mode="before")
    @classmethod
    def _split_fallback_models(cls, value: Any) -> Iterable[str]:
        # Accept JSON list or comma-separated string in env.
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return parsed
            # Fall back to CSV parsing when JSON isn't provided.
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value


    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Prefer dotenv for local dev, but allow explicit init values in tests.
        return (init_settings, dotenv_settings, env_settings, file_secret_settings)


settings = Settings()
