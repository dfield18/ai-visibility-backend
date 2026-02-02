"""User model for authentication and billing."""

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.auth_account import AuthAccount
    from app.models.magic_link_token import MagicLinkToken
    from app.models.run import Run
    from app.models.subscription import Subscription
    from app.models.usage_record import UsageRecord


class User(Base):
    """User account for authentication and billing.

    Attributes:
        id: Unique identifier (UUID).
        email: User's email address (unique).
        name: User's display name.
        avatar_url: URL to user's profile picture (from OAuth).
        email_verified: Whether the email has been verified.
        stripe_customer_id: Stripe customer ID for billing.
        created_at: When the user was created.
        updated_at: When the user was last updated.
    """

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    avatar_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    email_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
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

    # Relationships
    auth_accounts: Mapped[List["AuthAccount"]] = relationship(
        "AuthAccount",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    subscriptions: Mapped[List["Subscription"]] = relationship(
        "Subscription",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    runs: Mapped[List["Run"]] = relationship(
        "Run",
        back_populates="user",
        lazy="selectin",
    )
    usage_records: Mapped[List["UsageRecord"]] = relationship(
        "UsageRecord",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """String representation of the user."""
        return f"<User(id={self.id}, email={self.email})>"

    @property
    def has_active_subscription(self) -> bool:
        """Check if user has an active subscription."""
        return any(
            sub.status == "active" for sub in self.subscriptions
        )

    @property
    def current_subscription(self) -> Optional["Subscription"]:
        """Get the user's current active subscription."""
        for sub in self.subscriptions:
            if sub.status == "active":
                return sub
        return None
