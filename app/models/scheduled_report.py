"""Scheduled report model for automated visibility reports."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.user import User


class ScheduledReport(Base):
    """A scheduled report for automated visibility analysis.

    Attributes:
        id: Unique identifier (UUID).
        user_id: Reference to the user who created the report.
        name: User-defined report name.
        brand: The brand or category to analyze.
        search_type: Whether this is a brand or category search.
        prompts: List of prompts to test (JSON).
        competitors: List of competitors to track (JSON).
        providers: List of AI providers to use (JSON).
        temperatures: Temperature settings (JSON).
        repeats: Number of times to repeat each combination.
        frequency: How often to run (daily or weekly).
        day_of_week: Day to run for weekly reports (0=Monday, 6=Sunday).
        hour: Hour of day to run (0-23).
        timezone: User's timezone for scheduling.
        is_active: Whether the report is enabled.
        last_run_at: When the report was last run.
        next_run_at: When the report will next run.
        created_at: When the report was created.
        updated_at: When the report was last updated.
    """

    __tablename__ = "scheduled_reports"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    brand: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    search_type: Mapped[str] = mapped_column(
        String(20),
        default="brand",
        nullable=False,
    )
    prompts: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    competitors: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    providers: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    temperatures: Mapped[List[float]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    repeats: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )
    frequency: Mapped[str] = mapped_column(
        String(20),
        default="weekly",
        nullable=False,
    )
    day_of_week: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    hour: Mapped[int] = mapped_column(
        Integer,
        default=9,
        nullable=False,
    )
    timezone: Mapped[str] = mapped_column(
        String(50),
        default="UTC",
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    last_run_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    next_run_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
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

    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="scheduled_reports",
    )

    __table_args__ = (
        Index("ix_scheduled_reports_user_active", "user_id", "is_active"),
        Index("ix_scheduled_reports_next_run", "next_run_at"),
    )

    # Valid frequency values
    FREQUENCY_DAILY = "daily"
    FREQUENCY_WEEKLY = "weekly"

    VALID_FREQUENCIES = {FREQUENCY_DAILY, FREQUENCY_WEEKLY}

    def __repr__(self) -> str:
        """String representation of the scheduled report."""
        return f"<ScheduledReport(id={self.id}, name={self.name}, frequency={self.frequency})>"

    @property
    def total_calls(self) -> int:
        """Calculate total number of API calls per run."""
        return len(self.prompts) * len(self.providers) * len(self.temperatures) * self.repeats
