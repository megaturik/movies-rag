from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / '.env', env_file_encoding='utf-8'
    )

    OPENAI_KEY: str
    OPENAI_API_URL: str
    CHROMADB_HOST: str | None = "localhost"
    CHROMADB_PORT: int | None = 8010
    BACKEND_CORS_ORIGINS: List[str] = ['http://127.0.0.1:3000']


@lru_cache
def get_settings():
    return Settings()
