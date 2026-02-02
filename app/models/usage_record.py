"""UsageRecord model for tracking API usage and billing."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Numeric
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.run import Run
    from app.models.user import User


class UsageRecord(Base):
    """Record of API usage for billing and limits.

    Tracks each run's resource consumption for usage-based billing
    and enforcing plan limits.

    Attributes:
        id: Unique identifier (UUID).
        user_id: Reference to the user.
        run_id: Reference to the run (optional, for tracking).
        credits_used: Number of credits/units consumed.
        cost: Actual cost in dollars.
        created_at: When the usage occurred.
    """

    __tablename__ = "usage_records"

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
    run_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    credits_used: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )
    cost: Mapped[float] = mapped_column(
        Numeric(10, 4),
        nullable=False,
        default=0.0,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="usage_records",
    )
    run: Mapped[Optional["Run"]] = relationship(
        "Run",
        back_populates="usage_record",
    )

    __table_args__ = (
        Index("ix_usage_records_user_id", "user_id"),
        Index("ix_usage_records_created_at", "created_at"),
        Index("ix_usage_records_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        """String representation of the usage record."""
        return f"<UsageRecord(id={self.id}, credits={self.credits_used}, user_id={self.user_id})>"
