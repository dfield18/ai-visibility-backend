"""Session model for tracking user sessions."""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, List
from uuid import uuid4

from sqlalchemy import DateTime, Index, Numeric, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.run import Run
    from app.models.site_audit import SiteAudit


class Session(Base):
    """User session for tracking visibility runs.

    Sessions are created when users start using the application and
    expire after 7 days of inactivity. They track total spending
    across all runs.

    Attributes:
        id: Unique identifier (UUID).
        session_id: Client-provided session identifier.
        total_spent: Total amount spent on API calls.
        created_at: When the session was created.
        expires_at: When the session expires.
        runs: List of visibility runs in this session.
    """

    __tablename__ = "sessions"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    session_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    total_spent: Mapped[float] = mapped_column(
        Numeric(10, 2),
        default=0.00,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.utcnow() + timedelta(days=7),
        nullable=False,
    )

    # Relationships
    runs: Mapped[List["Run"]] = relationship(
        "Run",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    site_audits: Mapped[List["SiteAudit"]] = relationship(
        "SiteAudit",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_sessions_expires_at", "expires_at"),
    )

    def __repr__(self) -> str:
        """String representation of the session."""
        return f"<Session(id={self.id}, session_id={self.session_id})>"

    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.utcnow() > self.expires_at.replace(tzinfo=None)

    def extend_expiration(self, days: int = 7) -> None:
        """Extend session expiration by specified days."""
        self.expires_at = datetime.utcnow() + timedelta(days=days)
