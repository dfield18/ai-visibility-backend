"""Subscription model for Stripe billing."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.user import User


class Subscription(Base):
    """Stripe subscription for a user.

    Tracks the user's subscription status, plan, and billing period.
    This data is synced from Stripe via webhooks.

    Attributes:
        id: Unique identifier (UUID).
        user_id: Reference to the user.
        stripe_subscription_id: Stripe's subscription ID.
        stripe_price_id: Stripe's price ID for the plan.
        status: Subscription status from Stripe.
        current_period_start: Start of current billing period.
        current_period_end: End of current billing period.
        cancel_at_period_end: Whether subscription will cancel at period end.
        canceled_at: When the subscription was canceled.
        created_at: When the subscription was created.
        updated_at: When the subscription was last updated.
    """

    __tablename__ = "subscriptions"

    # Status constants (from Stripe)
    STATUS_ACTIVE = "active"
    STATUS_PAST_DUE = "past_due"
    STATUS_CANCELED = "canceled"
    STATUS_INCOMPLETE = "incomplete"
    STATUS_INCOMPLETE_EXPIRED = "incomplete_expired"
    STATUS_TRIALING = "trialing"
    STATUS_UNPAID = "unpaid"
    STATUS_PAUSED = "paused"

    VALID_STATUSES = {
        STATUS_ACTIVE,
        STATUS_PAST_DUE,
        STATUS_CANCELED,
        STATUS_INCOMPLETE,
        STATUS_INCOMPLETE_EXPIRED,
        STATUS_TRIALING,
        STATUS_UNPAID,
        STATUS_PAUSED,
    }

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
    stripe_subscription_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    stripe_price_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=STATUS_INCOMPLETE,
    )
    current_period_start: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    current_period_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    cancel_at_period_end: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    canceled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
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
    user: Mapped["User"] = relationship(
        "User",
        back_populates="subscriptions",
    )

    __table_args__ = (
        Index("ix_subscriptions_user_id", "user_id"),
        Index("ix_subscriptions_status", "status"),
    )

    def __repr__(self) -> str:
        """String representation of the subscription."""
        return f"<Subscription(id={self.id}, status={self.status}, user_id={self.user_id})>"

    @property
    def is_active(self) -> bool:
        """Check if subscription is currently active."""
        return self.status in {self.STATUS_ACTIVE, self.STATUS_TRIALING}

    @property
    def is_canceled(self) -> bool:
        """Check if subscription has been canceled."""
        return self.status == self.STATUS_CANCELED or self.cancel_at_period_end

    @property
    def days_until_renewal(self) -> Optional[int]:
        """Calculate days until subscription renews."""
        if not self.current_period_end:
            return None
        delta = self.current_period_end - datetime.utcnow()
        return max(0, delta.days)
