"""Run model for tracking visibility check runs."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.result import Result
    from app.models.session import Session
    from app.models.usage_record import UsageRecord
    from app.models.user import User


class Run(Base):
    """A visibility check run against AI providers.

    Each run represents a batch of API calls to check brand visibility
    across different prompts and providers.

    Attributes:
        id: Unique identifier (UUID).
        session_id: Reference to parent session.
        status: Current run status (queued, running, complete, failed, cancelled).
        brand: The brand being tracked.
        config: JSON configuration (prompts, competitors, providers, etc.).
        total_calls: Total number of API calls to make.
        completed_calls: Number of successfully completed calls.
        failed_calls: Number of failed calls.
        estimated_cost: Estimated cost before running.
        actual_cost: Actual cost after completion.
        cancelled: Whether the run was cancelled.
        created_at: When the run was created.
        completed_at: When the run finished.
        results: List of individual results.
        session: Parent session reference.
    """

    __tablename__ = "runs"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    session_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,  # Nullable for existing runs before user system
        index=True,
    )
    parent_run_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="queued",
        nullable=False,
    )
    brand: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    config: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    total_calls: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    completed_calls: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    failed_calls: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    estimated_cost: Mapped[float] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )
    actual_cost: Mapped[float] = mapped_column(
        Numeric(10, 4),
        default=0.00,
        nullable=False,
    )
    cancelled: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    results: Mapped[List["Result"]] = relationship(
        "Result",
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    session: Mapped["Session"] = relationship(
        "Session",
        back_populates="runs",
    )
    user: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="runs",
    )
    usage_record: Mapped[Optional["UsageRecord"]] = relationship(
        "UsageRecord",
        back_populates="run",
        uselist=False,
    )
    parent_run: Mapped[Optional["Run"]] = relationship(
        "Run",
        remote_side=[id],
        foreign_keys=[parent_run_id],
        back_populates="child_runs",
    )
    child_runs: Mapped[List["Run"]] = relationship(
        "Run",
        back_populates="parent_run",
        foreign_keys=[parent_run_id],
    )

    __table_args__ = (
        Index("ix_runs_session_status", "session_id", "status"),
        Index("ix_runs_created_at", "created_at"),
    )

    # Valid status values
    STATUS_QUEUED = "queued"
    STATUS_RUNNING = "running"
    STATUS_COMPLETE = "complete"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"

    VALID_STATUSES = {
        STATUS_QUEUED,
        STATUS_RUNNING,
        STATUS_COMPLETE,
        STATUS_FAILED,
        STATUS_CANCELLED,
    }

    def __repr__(self) -> str:
        """String representation of the run."""
        return f"<Run(id={self.id}, brand={self.brand}, status={self.status})>"

    @property
    def progress_percent(self) -> float:
        """Calculate completion percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.completed_calls + self.failed_calls) / self.total_calls * 100

    @property
    def is_complete(self) -> bool:
        """Check if run has finished (success, failure, or cancelled)."""
        return self.status in {
            self.STATUS_COMPLETE,
            self.STATUS_FAILED,
            self.STATUS_CANCELLED,
        }

    def mark_complete(self) -> None:
        """Mark the run as complete."""
        self.status = self.STATUS_COMPLETE
        self.completed_at = datetime.utcnow()

    def mark_failed(self) -> None:
        """Mark the run as failed."""
        self.status = self.STATUS_FAILED
        self.completed_at = datetime.utcnow()

    def mark_cancelled(self) -> None:
        """Mark the run as cancelled."""
        self.status = self.STATUS_CANCELLED
        self.cancelled = True
        self.completed_at = datetime.utcnow()
