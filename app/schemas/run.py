"""Pydantic schemas for run endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class RunRequest(BaseModel):
    """Request body for starting a visibility run.

    Attributes:
        session_id: Client session identifier.
        brand: Brand or category to analyze.
        search_type: Whether this is a brand or category search.
        prompts: List of prompts to test.
        competitors: List of competitors/brands to track.
        providers: AI providers to use.
        temperatures: Temperature settings for variation.
        repeats: Number of times to repeat each combination.
    """

    session_id: str = Field(..., min_length=1, max_length=255)
    brand: str = Field(..., min_length=1, max_length=255)
    search_type: Literal["brand", "category"] = Field(default="brand")
    prompts: List[str] = Field(..., min_length=1, max_length=10)
    competitors: List[str] = Field(..., min_length=1, max_length=10)
    providers: List[Literal["openai", "gemini", "anthropic", "perplexity", "ai_overviews"]] = Field(..., min_length=1)
    temperatures: List[float] = Field(..., min_length=1)
    repeats: int = Field(default=1, ge=1, le=10)
    openai_model: Literal["gpt-4o-mini", "gpt-4o"] = Field(default="gpt-4o-mini")
    anthropic_model: Literal["claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"] = Field(default="claude-haiku-4-5-20251001")
    country: str = Field(default="us", max_length=5)

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, v: List[str]) -> List[str]:
        """Validate prompts are non-empty strings."""
        return [p.strip() for p in v if p.strip()]

    @field_validator("competitors")
    @classmethod
    def validate_competitors(cls, v: List[str]) -> List[str]:
        """Validate competitors are non-empty strings."""
        return [c.strip() for c in v if c.strip()]

    @field_validator("temperatures")
    @classmethod
    def validate_temperatures(cls, v: List[float]) -> List[float]:
        """Validate temperatures are in valid range."""
        for temp in v:
            if not 0.0 <= temp <= 2.0:
                raise ValueError(f"Temperature {temp} must be between 0.0 and 2.0")
        return v

    @property
    def total_calls(self) -> int:
        """Calculate total number of API calls."""
        return len(self.prompts) * len(self.providers) * len(self.temperatures) * self.repeats


class RunResponse(BaseModel):
    """Response after starting a run.

    Attributes:
        run_id: Unique identifier for the run.
        status: Current status of the run.
        total_calls: Total number of API calls to make.
        estimated_cost: Estimated cost in dollars.
        estimated_duration_seconds: Estimated time to complete.
    """

    run_id: UUID
    status: str
    total_calls: int
    estimated_cost: float
    estimated_duration_seconds: int


class ProviderStats(BaseModel):
    """Statistics for a single provider."""

    mentioned: int
    total: int
    rate: float


class CompetitorStats(BaseModel):
    """Statistics for a single competitor."""

    count: int
    rate: float


class RunSummary(BaseModel):
    """Summary statistics for a run."""

    brand_mention_rate: float
    by_provider: Dict[str, ProviderStats]
    competitor_mentions: Dict[str, CompetitorStats]


class SourceItem(BaseModel):
    """Source citation from an LLM response."""

    url: str = ""
    title: str = ""


class GroundingSupport(BaseModel):
    """A grounding support linking response text to sources."""

    segment: str = ""
    chunk_indices: List[int] = []
    confidence_scores: List[float] = []


class GroundingMetadata(BaseModel):
    """Grounding metadata from Gemini API."""

    supports: List[GroundingSupport] = []
    search_queries: List[str] = []


class ResultItem(BaseModel):
    """Individual result from an API call."""

    id: UUID
    prompt: str
    provider: str
    model: str
    temperature: float
    repeat_index: int
    response_text: Optional[str]
    error: Optional[str]
    brand_mentioned: Optional[bool]
    competitors_mentioned: Optional[List[str]]
    response_type: Optional[str]
    tokens: Optional[int]
    cost: Optional[float]
    sources: Optional[List[SourceItem]] = None
    grounding_metadata: Optional[GroundingMetadata] = None
    brand_sentiment: Optional[str] = None
    competitor_sentiments: Optional[Dict[str, str]] = None
    all_brands_mentioned: Optional[List[str]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class RunStatusResponse(BaseModel):
    """Full status response for a run.

    Attributes:
        run_id: Unique identifier for the run.
        status: Current status (queued, running, complete, failed, cancelled).
        brand: Brand or category being analyzed.
        search_type: Whether this is a brand or category search.
        total_calls: Total number of API calls.
        completed_calls: Number of completed calls.
        failed_calls: Number of failed calls.
        progress_percent: Completion percentage.
        estimated_seconds_remaining: Estimated time remaining.
        actual_cost: Actual cost so far.
        created_at: When the run was created.
        completed_at: When the run completed (if applicable).
        summary: Aggregated statistics.
        results: List of individual results.
    """

    run_id: UUID
    status: str
    brand: str
    search_type: Literal["brand", "category"] = "brand"
    total_calls: int
    completed_calls: int
    failed_calls: int
    progress_percent: float
    estimated_seconds_remaining: Optional[int]
    actual_cost: float
    created_at: datetime
    completed_at: Optional[datetime]
    summary: Optional[RunSummary]
    results: List[ResultItem]


class CancelResponse(BaseModel):
    """Response after cancelling a run."""

    run_id: UUID
    status: str
    completed_calls: int
    cancelled_calls: int
    actual_cost: float


class AISummaryResponse(BaseModel):
    """Response containing AI-generated summary and recommendations of run results."""

    run_id: UUID
    summary: str
    recommendations: str = ""  # Prose-style strategy brief
    generated_at: datetime
