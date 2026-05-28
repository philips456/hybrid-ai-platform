"""
Settings centralisés — chargés depuis .env
Utiliser: from configs.settings import settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # Base de données
    database_url: str = "postgresql://postgres:changeme@localhost:5432/hybrid_ai_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "hybrid_ai_db"
    postgres_user: str = "postgres"
    postgres_password: str = "changeme"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "hybrid_ai_platform"

    # LangSmith
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = True
    langchain_project: str = "hybrid-ai-platform"

    # API
    api_secret_key: str = "changeme_secret_key_min_32_chars"
    api_algorithm: str = "HS256"
    api_access_token_expire_minutes: int = 60
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Domaine applicatif
    domain: str = "telecom"  # telecom | finance | industry

    # Feedback Loop
    feedback_error_threshold: float = 0.15
    feedback_consecutive_periods: int = 5

    class Config:
        env_file = "configs/.env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
