"""Result model for storing individual API call results."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.run import Run


class Result(Base):
    """Individual API call result.

    Stores the response from a single API call along with
    analysis of brand visibility.

    Attributes:
        id: Unique identifier (UUID).
        run_id: Reference to parent run.
        prompt: The prompt sent to the AI.
        provider: AI provider (openai, gemini).
        model: Specific model used.
        temperature: Temperature setting used.
        repeat_index: Index for repeated calls (0-based).
        response_text: Full response from the AI.
        error: Error message if call failed.
        brand_mentioned: Whether the brand was mentioned.
        competitors_mentioned: List of competitors found.
        response_type: Type of response (list, prose, unknown).
        tokens: Number of tokens used.
        cost: Cost of this API call.
        sources: List of source URLs/citations from the response.
        grounding_metadata: Grounding metadata from Gemini.
        brand_sentiment: How the AI describes the brand (strong_endorsement, neutral_mention, conditional, negative_comparison, not_mentioned).
        competitor_sentiments: Dict mapping competitor names to their sentiment classification.
        created_at: When the result was created.
        run: Parent run reference.
    """

    __tablename__ = "results"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    temperature: Mapped[float] = mapped_column(
        Numeric(3, 2),
        nullable=False,
    )
    repeat_index: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    response_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    error: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    brand_mentioned: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )
    competitors_mentioned: Mapped[Optional[List[Any]]] = mapped_column(
        JSON,
        nullable=True,
    )
    response_type: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )
    tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    cost: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )
    sources: Mapped[Optional[List[Any]]] = mapped_column(
        JSON,
        nullable=True,
    )
    grounding_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
    )
    brand_sentiment: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    competitor_sentiments: Mapped[Optional[Dict[str, str]]] = mapped_column(
        JSON,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    run: Mapped["Run"] = relationship(
        "Run",
        back_populates="results",
    )

    __table_args__ = (
        Index("ix_results_run_id", "run_id"),
        Index("ix_results_created_at", "created_at"),
    )

    # Valid provider values
    PROVIDER_OPENAI = "openai"
    PROVIDER_GEMINI = "gemini"

    VALID_PROVIDERS = {PROVIDER_OPENAI, PROVIDER_GEMINI}

    # Valid response types
    RESPONSE_LIST = "list"
    RESPONSE_PROSE = "prose"
    RESPONSE_UNKNOWN = "unknown"

    VALID_RESPONSE_TYPES = {RESPONSE_LIST, RESPONSE_PROSE, RESPONSE_UNKNOWN}

    # Valid sentiment values
    SENTIMENT_STRONG_ENDORSEMENT = "strong_endorsement"
    SENTIMENT_NEUTRAL_MENTION = "neutral_mention"
    SENTIMENT_CONDITIONAL = "conditional"
    SENTIMENT_NEGATIVE_COMPARISON = "negative_comparison"
    SENTIMENT_NOT_MENTIONED = "not_mentioned"

    VALID_SENTIMENTS = {
        SENTIMENT_STRONG_ENDORSEMENT,
        SENTIMENT_NEUTRAL_MENTION,
        SENTIMENT_CONDITIONAL,
        SENTIMENT_NEGATIVE_COMPARISON,
        SENTIMENT_NOT_MENTIONED,
    }

    def __repr__(self) -> str:
        """String representation of the result."""
        return f"<Result(id={self.id}, provider={self.provider}, brand_mentioned={self.brand_mentioned})>"

    @property
    def is_success(self) -> bool:
        """Check if the API call was successful."""
        return self.error is None and self.response_text is not None

    @property
    def competitor_count(self) -> int:
        """Count of competitors mentioned."""
        if self.competitors_mentioned is None:
            return 0
        return len(self.competitors_mentioned)
