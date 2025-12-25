from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / '.env', env_file_encoding='utf-8'
    )

    XAI_API_KEY: str
    XAI_API_URL: str | None = 'https://api.x.ai/v1'
    XAI_MODEL: str | None = 'grok-4-1-fast-reasoning'
    XAI_TEMP: float | None = 0.7
    XAI_MAX_TOKENS: int | None = 3000
    SENTENCE_MODEL: str | None = "paraphrase-multilingual-MiniLM-L12-v2"
    CHROMADB_HOST: str | None = "localhost"
    CHROMADB_PORT: int | None = 8010
    REDIS_HOST: str | None = "localhost"
    REDIS_PORT: int | None = 6379
    REDIS_CACHE_TTL: int | None = 600


@lru_cache
def get_settings():
    return Settings()
