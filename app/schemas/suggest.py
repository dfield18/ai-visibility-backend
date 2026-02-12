"""Pydantic schemas for the suggest endpoint."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SuggestRequest(BaseModel):
    """Request body for generating suggestions.

    Attributes:
        brand: The brand name or category to generate suggestions for.
        industry: Optional industry context for better suggestions.
        search_type: Whether this is a brand, category, or local business search.
        location: Location for local search type (city, neighborhood, etc.).
    """

    brand: str = Field(..., min_length=1, max_length=255, description="Brand name or category")
    industry: Optional[str] = Field(
        None, max_length=255, description="Optional industry context"
    )
    search_type: Literal["brand", "category", "local", "issue", "public_figure"] = Field(
        default="brand", description="Whether searching for a brand, category, or local business"
    )
    location: Optional[str] = Field(
        None, max_length=255, description="Location for local search (required for local search_type)"
    )

    @field_validator('location')
    @classmethod
    def validate_location(cls, v, info):
        """Validate that location is provided for local search type."""
        if info.data.get('search_type') == 'local' and not v:
            raise ValueError('Location is required for local search type')
        return v


class SuggestResponse(BaseModel):
    """Response containing generated suggestions.

    Attributes:
        brand: The brand that suggestions were generated for.
        prompts: List of suggested consumer search queries.
        competitors: List of identified competitors.
    """

    brand: str
    prompts: List[str]
    competitors: List[str]
