from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "mysql+aiomysql://research:research@localhost:3306/research_agent"
    api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "phi3"
    qdrant_url: str | None = None
    qdrant_path: str = "./qdrant_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-base"
    max_question_length: int = Field(default=2000, ge=1, le=20000)
    conversation_history_limit: int = Field(default=10, ge=0, le=100)
    history_token_budget: int = Field(default=500, ge=0, le=3000)
    max_upload_bytes: int = Field(default=20_000_000, ge=1024, le=100_000_000)
    upload_directory: str = "./data/uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
