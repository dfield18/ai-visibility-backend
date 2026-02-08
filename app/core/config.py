"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        DATABASE_URL: PostgreSQL connection string with asyncpg driver.
        OPENAI_API_KEY: OpenAI API key for GPT model access.
        GEMINI_API_KEY: Google Gemini API key.
        ENVIRONMENT: Current environment (development, staging, production).
        DEBUG: Enable debug mode.
        CORS_ORIGINS: List of allowed CORS origins.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    DATABASE_URL: str

    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def convert_database_url(cls, v: str) -> str:
        """Convert database URL to asyncpg format.

        Railway and other providers use postgres:// or postgresql://
        but asyncpg requires postgresql+asyncpg://
        """
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        if v.startswith("postgresql://") and "+asyncpg" not in v:
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    PERPLEXITY_API_KEY: str = ""
    GROK_API_KEY: str = ""
    LLAMA_API_KEY: str = ""
    SERPAPI_API_KEY: str = ""
    RESEND_API_KEY: str = ""  # Resend API key for email notifications
    CLERK_DOMAIN: str = ""  # e.g., "your-app.clerk.accounts.dev"
    CLERK_SECRET_KEY: str = ""  # Clerk secret key for backend API calls
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | List[str]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # Handle JSON-like string or comma-separated
            if v.startswith("["):
                import json
                return json.loads(v.replace("'", '"'))
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings singleton.
    """
    return Settings()


settings = get_settings()
