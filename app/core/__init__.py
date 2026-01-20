"""Core module containing configuration and database setup."""

from app.core.config import settings
from app.core.database import Base, get_db

__all__ = ["settings", "Base", "get_db"]
