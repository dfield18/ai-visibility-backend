"""MagicLinkToken model for passwordless authentication."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.database import Base


class MagicLinkToken(Base):
    """Token for magic link (passwordless) authentication.

    Tokens are single-use and expire after a set duration.
    The token itself is stored as a hash for security.

    Attributes:
        id: Unique identifier (UUID).
        email: Email address the token was sent to.
        token_hash: SHA-256 hash of the token (never store plain tokens).
        expires_at: When the token expires.
        used_at: When the token was used (null if unused).
        created_at: When the token was created.
    """

    __tablename__ = "magic_link_tokens"

    # Default expiration time (15 minutes)
    DEFAULT_EXPIRATION_MINUTES = 15

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(
        String(64),  # SHA-256 produces 64 hex characters
        nullable=False,
        unique=True,
        index=True,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        """String representation of the magic link token."""
        return f"<MagicLinkToken(id={self.id}, email={self.email})>"

    @property
    def is_expired(self) -> bool:
        """Check if the token has expired."""
        return datetime.utcnow() > self.expires_at.replace(tzinfo=None)

    @property
    def is_used(self) -> bool:
        """Check if the token has been used."""
        return self.used_at is not None

    @property
    def is_valid(self) -> bool:
        """Check if the token is valid (not expired and not used)."""
        return not self.is_expired and not self.is_used

    def mark_used(self) -> None:
        """Mark the token as used."""
        self.used_at = datetime.utcnow()

    @classmethod
    def get_expiration_time(cls, minutes: Optional[int] = None) -> datetime:
        """Calculate expiration timestamp.

        Args:
            minutes: Custom expiration time in minutes.

        Returns:
            datetime: Expiration timestamp.
        """
        expiration_minutes = minutes or cls.DEFAULT_EXPIRATION_MINUTES
        return datetime.utcnow() + timedelta(minutes=expiration_minutes)
