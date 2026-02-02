"""Pydantic schemas for scheduled report endpoints."""

from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ScheduledReportCreate(BaseModel):
    """Request body for creating a scheduled report.

    Attributes:
        name: User-defined report name.
        brand: Brand or category to analyze.
        search_type: Whether this is a brand or category search.
        prompts: List of prompts to test.
        competitors: List of competitors to track.
        providers: AI providers to use.
        temperatures: Temperature settings for variation.
        repeats: Number of times to repeat each combination.
        frequency: How often to run (daily or weekly).
        day_of_week: Day to run for weekly reports (0=Monday, 6=Sunday).
        hour: Hour of day to run (0-23).
        timezone: User's timezone for scheduling.
    """

    name: str = Field(..., min_length=1, max_length=255)
    brand: str = Field(..., min_length=1, max_length=255)
    search_type: Literal["brand", "category"] = Field(default="brand")
    prompts: List[str] = Field(..., min_length=1, max_length=10)
    competitors: List[str] = Field(..., min_length=1, max_length=10)
    providers: List[Literal["openai", "gemini", "anthropic", "perplexity", "ai_overviews"]] = Field(..., min_length=1)
    temperatures: List[float] = Field(..., min_length=1)
    repeats: int = Field(default=1, ge=1, le=10)
    frequency: Literal["daily", "weekly"] = Field(default="weekly")
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6)
    hour: int = Field(default=9, ge=0, le=23)
    timezone: str = Field(default="UTC", max_length=50)

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

    @field_validator("day_of_week")
    @classmethod
    def validate_day_of_week(cls, v: Optional[int], info) -> Optional[int]:
        """Validate day_of_week is set for weekly reports."""
        # Note: We can't access other fields in field_validator in Pydantic v2
        # This validation will be done in the endpoint
        return v


class ScheduledReportUpdate(BaseModel):
    """Request body for updating a scheduled report.

    All fields are optional - only provided fields will be updated.
    """

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    brand: Optional[str] = Field(None, min_length=1, max_length=255)
    search_type: Optional[Literal["brand", "category"]] = None
    prompts: Optional[List[str]] = Field(None, min_length=1, max_length=10)
    competitors: Optional[List[str]] = Field(None, min_length=1, max_length=10)
    providers: Optional[List[Literal["openai", "gemini", "anthropic", "perplexity", "ai_overviews"]]] = Field(None, min_length=1)
    temperatures: Optional[List[float]] = Field(None, min_length=1)
    repeats: Optional[int] = Field(None, ge=1, le=10)
    frequency: Optional[Literal["daily", "weekly"]] = None
    day_of_week: Optional[int] = Field(None, ge=0, le=6)
    hour: Optional[int] = Field(None, ge=0, le=23)
    timezone: Optional[str] = Field(None, max_length=50)

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate prompts are non-empty strings."""
        if v is None:
            return None
        return [p.strip() for p in v if p.strip()]

    @field_validator("competitors")
    @classmethod
    def validate_competitors(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate competitors are non-empty strings."""
        if v is None:
            return None
        return [c.strip() for c in v if c.strip()]

    @field_validator("temperatures")
    @classmethod
    def validate_temperatures(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate temperatures are in valid range."""
        if v is None:
            return None
        for temp in v:
            if not 0.0 <= temp <= 2.0:
                raise ValueError(f"Temperature {temp} must be between 0.0 and 2.0")
        return v


class ScheduledReportResponse(BaseModel):
    """Response model for a single scheduled report."""

    id: UUID
    user_id: UUID
    name: str
    brand: str
    search_type: Literal["brand", "category"]
    prompts: List[str]
    competitors: List[str]
    providers: List[str]
    temperatures: List[float]
    repeats: int
    frequency: Literal["daily", "weekly"]
    day_of_week: Optional[int]
    hour: int
    timezone: str
    is_active: bool
    last_run_at: Optional[datetime]
    next_run_at: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ScheduledReportListResponse(BaseModel):
    """Response model for listing scheduled reports."""

    reports: List[ScheduledReportResponse]
    total: int


class ToggleResponse(BaseModel):
    """Response after toggling a report's active status."""

    id: UUID
    is_active: bool
    next_run_at: Optional[datetime]


class RunNowResponse(BaseModel):
    """Response after triggering an immediate report run."""

    id: UUID
    run_id: UUID
    message: str
