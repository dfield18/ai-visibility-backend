"""SiteAudit model for LLM site optimization audits."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.session import Session
    from app.models.user import User


class SiteAudit(Base):
    """A site audit checking LLM/AI search optimization compatibility.

    Each audit analyzes a website for AI crawler access, meta directives,
    llms.txt support, structured data, content accessibility, and structure.

    Attributes:
        id: Unique identifier (UUID).
        session_id: Reference to parent session.
        user_id: Optional reference to authenticated user.
        url: The URL being audited.
        status: Current audit status (queued, running, complete, failed).
        results: JSON object containing all audit results.
        overall_score: Calculated score from 0-100.
        error_message: Error message if audit failed.
        created_at: When the audit was created.
        completed_at: When the audit finished.
    """

    __tablename__ = "site_audits"

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
        nullable=True,
        index=True,
    )
    url: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="queued",
        nullable=False,
    )
    results: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        default=None,
    )
    overall_score: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
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
    session: Mapped["Session"] = relationship(
        "Session",
        back_populates="site_audits",
    )
    user: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="site_audits",
    )

    __table_args__ = (
        Index("ix_site_audits_session_id", "session_id"),
        Index("ix_site_audits_url", "url"),
        Index("ix_site_audits_created_at", "created_at"),
    )

    # Valid status values
    STATUS_QUEUED = "queued"
    STATUS_RUNNING = "running"
    STATUS_COMPLETE = "complete"
    STATUS_FAILED = "failed"

    VALID_STATUSES = {
        STATUS_QUEUED,
        STATUS_RUNNING,
        STATUS_COMPLETE,
        STATUS_FAILED,
    }

    def __repr__(self) -> str:
        """String representation of the site audit."""
        return f"<SiteAudit(id={self.id}, url={self.url}, status={self.status})>"

    @property
    def is_complete(self) -> bool:
        """Check if audit has finished (success or failure)."""
        return self.status in {self.STATUS_COMPLETE, self.STATUS_FAILED}

    def mark_running(self) -> None:
        """Mark the audit as running."""
        self.status = self.STATUS_RUNNING

    def mark_complete(self, results: Dict[str, Any], score: int) -> None:
        """Mark the audit as complete with results."""
        self.status = self.STATUS_COMPLETE
        self.results = results
        self.overall_score = score
        self.completed_at = datetime.utcnow()

    def mark_failed(self, error: str) -> None:
        """Mark the audit as failed with error message."""
        self.status = self.STATUS_FAILED
        self.error_message = error
        self.completed_at = datetime.utcnow()
