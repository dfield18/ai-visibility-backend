"""Pydantic schemas for the suggest endpoint."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SuggestRequest(BaseModel):
    """Request body for generating suggestions.

    Attributes:
        brand: The brand name or category to generate suggestions for.
        industry: Optional industry context for better suggestions.
        search_type: Whether this is a brand or category search.
    """

    brand: str = Field(..., min_length=1, max_length=255, description="Brand name or category")
    industry: Optional[str] = Field(
        None, max_length=255, description="Optional industry context"
    )
    search_type: Literal["brand", "category"] = Field(
        default="brand", description="Whether searching for a brand or category"
    )


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
