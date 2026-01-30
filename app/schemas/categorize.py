"""Pydantic schemas for the categorize endpoint."""

from typing import Dict, List

from pydantic import BaseModel, Field


class CategorizeRequest(BaseModel):
    """Request body for categorizing domains.

    Attributes:
        domains: List of domain names to categorize.
    """

    domains: List[str] = Field(
        ..., min_items=1, max_items=50, description="List of domain names to categorize"
    )


class CategorizeResponse(BaseModel):
    """Response containing domain categories.

    Attributes:
        categories: Dict mapping domain names to their category.
    """

    categories: Dict[str, str]
