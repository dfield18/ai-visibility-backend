"""Cached category model for storing domain classifications."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.database import Base


class CachedCategory(Base):
    """Cached domain category classification.

    Stores domain-to-category mappings from OpenAI to avoid
    repeated API calls for the same domains. These don't expire
    since domain categories rarely change.

    Attributes:
        id: Unique identifier (UUID).
        domain: The domain name (e.g., 'reddit.com').
        category: The classified category (e.g., 'Social Media').
        created_at: When the cache entry was created.
        updated_at: When the cache entry was last updated.
    """

    __tablename__ = "cached_categories"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    domain: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )
    category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        """String representation of the cached category."""
        return f"<CachedCategory(domain={self.domain}, category={self.category})>"
