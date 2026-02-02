"""AuthAccount model for linking OAuth providers."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.user import User


class AuthAccount(Base):
    """Linked authentication account (Google, email, etc.).

    Allows users to sign in with multiple providers while maintaining
    a single user account.

    Attributes:
        id: Unique identifier (UUID).
        user_id: Reference to the user.
        provider: Auth provider name ('google', 'email').
        provider_account_id: Provider's unique ID for the user (e.g., Google sub).
        created_at: When the account was linked.
    """

    __tablename__ = "auth_accounts"

    # Provider constants
    PROVIDER_GOOGLE = "google"
    PROVIDER_EMAIL = "email"

    VALID_PROVIDERS = {PROVIDER_GOOGLE, PROVIDER_EMAIL}

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    provider_account_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,  # Null for email provider
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="auth_accounts",
    )

    __table_args__ = (
        # Ensure unique provider + provider_account_id combination
        Index(
            "ix_auth_accounts_provider_account",
            "provider",
            "provider_account_id",
            unique=True,
            postgresql_where=(provider_account_id.isnot(None)),
        ),
        Index("ix_auth_accounts_user_id", "user_id"),
    )

    def __repr__(self) -> str:
        """String representation of the auth account."""
        return f"<AuthAccount(id={self.id}, provider={self.provider}, user_id={self.user_id})>"
