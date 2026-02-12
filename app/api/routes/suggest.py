"""Suggest endpoint for generating prompts and competitors/brands."""

import re
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from sqlalchemy import select, and_

from app.core.database import DatabaseSession
from app.models.cached_suggestion import CachedSuggestion
from app.schemas.suggest import SuggestRequest, SuggestResponse
from app.services.openai_service import OpenAIService

router = APIRouter()

# Current year for prompt suggestions
CURRENT_YEAR = datetime.now().year


def update_years_in_prompts(prompts: List[str]) -> List[str]:
    """Update old years in prompts to the current year.

    Replaces any 2020-series year older than the current year.
    This ensures suggestions are for current/relevant time periods.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of prompts with updated years.
    """
    current = CURRENT_YEAR
    updated = []
    for prompt in prompts:
        # Replace any 202x year that is older than current year
        def replace_old_year(match: re.Match) -> str:
            year = int(match.group(0))
            return str(current) if year < current else match.group(0)
        updated_prompt = re.sub(r'\b(202\d)\b', replace_old_year, prompt)
        updated.append(updated_prompt)
    return updated


async def get_cached_suggestion(
    db: DatabaseSession,
    brand: str,
    search_type: str,
    industry: Optional[str],
) -> Optional[CachedSuggestion]:
    """Look up a cached suggestion by brand, search_type, and industry.

    Args:
        db: Database session.
        brand: Brand or category name (will be lowercased).
        search_type: Type of search ('brand' or 'category').
        industry: Optional industry context (will be lowercased).

    Returns:
        CachedSuggestion if found and not expired, None otherwise.
    """
    normalized_brand = brand.lower().strip()
    normalized_industry = industry.lower().strip() if industry else None

    # Build query conditions
    conditions = [
        CachedSuggestion.brand == normalized_brand,
        CachedSuggestion.search_type == search_type,
        CachedSuggestion.expires_at > datetime.utcnow(),
    ]

    if normalized_industry:
        conditions.append(CachedSuggestion.industry == normalized_industry)
    else:
        conditions.append(CachedSuggestion.industry.is_(None))

    result = await db.execute(
        select(CachedSuggestion).where(and_(*conditions))
    )
    return result.scalar_one_or_none()


async def save_cached_suggestion(
    db: DatabaseSession,
    brand: str,
    search_type: str,
    prompts: List[str],
    competitors: List[str],
    industry: Optional[str],
) -> CachedSuggestion:
    """Save a new cached suggestion to the database.

    Args:
        db: Database session.
        brand: Brand or category name.
        search_type: Type of search ('brand' or 'category').
        prompts: List of generated prompts.
        competitors: List of generated competitors/brands.
        industry: Optional industry context.

    Returns:
        The created CachedSuggestion instance.
    """
    cached = CachedSuggestion.create_with_ttl(
        brand=brand,
        search_type=search_type,
        prompts=prompts,
        competitors=competitors,
        industry=industry,
    )
    db.add(cached)
    await db.flush()
    return cached


@router.post("/suggest", response_model=SuggestResponse)
async def generate_suggestions(
    request: SuggestRequest,
    db: DatabaseSession,
) -> SuggestResponse:
    """Generate suggested prompts and competitors/brands.

    For brand searches: generates consumer search queries and identifies competitors.
    For category searches: generates search queries and identifies brands in that category.

    Results are cached for 30 days to reduce API calls.

    Args:
        request: SuggestRequest containing brand/category name, search_type, and optional industry.
        db: Database session for caching.

    Returns:
        SuggestResponse with generated prompts and competitors/brands.

    Raises:
        HTTPException: If suggestion generation fails.
    """
    # Check cache first (gracefully handle if table doesn't exist yet)
    cached = None
    cache_available = True
    try:
        cached = await get_cached_suggestion(
            db=db,
            brand=request.brand,
            search_type=request.search_type,
            industry=request.industry,
        )
    except Exception as e:
        # Cache table might not exist yet - continue without caching
        print(f"[Suggest] Cache lookup failed (table may not exist): {e}")
        cache_available = False

    if cached:
        print(f"[Suggest] Cache hit for '{request.brand}' ({request.search_type})")
        # Update years in cached prompts (in case they were cached last year)
        prompts = update_years_in_prompts(cached.prompts)
        return SuggestResponse(
            brand=request.brand,
            prompts=prompts,
            competitors=cached.competitors,
        )

    print(f"[Suggest] Cache miss for '{request.brand}' ({request.search_type}), generating...")

    try:
        service = OpenAIService()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=f"OpenAI service not available: {str(e)}",
        )

    try:
        if request.search_type == "local":
            # For local searches, generate prompts with location context
            # and list local businesses in that category/location
            prompts = await service.generate_local_prompts(
                category=request.brand,
                location=request.location,
            )
            # Update old years to current year in suggestions
            prompts = update_years_in_prompts(prompts)
            businesses = await service.generate_local_businesses(
                category=request.brand,
                location=request.location,
            )

            # Note: Local searches are not cached since they're location-specific
            # and locations have high cardinality

            return SuggestResponse(
                brand=request.brand,
                prompts=prompts,
                competitors=businesses,  # Local businesses to track
            )
        elif request.search_type == "category":
            # For category searches, generate prompts for the category
            # and list brands in that category
            prompts = await service.generate_category_prompts(
                category=request.brand,
                industry=request.industry,
            )
            # Update old years to current year in suggestions
            prompts = update_years_in_prompts(prompts)
            brands = await service.generate_category_brands(
                category=request.brand,
                industry=request.industry,
            )

            # Cache the result (if cache is available)
            if cache_available:
                try:
                    await save_cached_suggestion(
                        db=db,
                        brand=request.brand,
                        search_type=request.search_type,
                        prompts=prompts,
                        competitors=brands,
                        industry=request.industry,
                    )
                except Exception as e:
                    print(f"[Suggest] Failed to cache result: {e}")

            return SuggestResponse(
                brand=request.brand,
                prompts=prompts,
                competitors=brands,  # Brands to track for category search
            )
        else:
            # For brand searches, generate prompts and competitors
            prompts = await service.generate_prompts(
                brand=request.brand,
                industry=request.industry,
            )
            # Update old years to current year in suggestions
            prompts = update_years_in_prompts(prompts)
            competitors = await service.generate_competitors(
                brand=request.brand,
                industry=request.industry,
            )

            # Cache the result (if cache is available)
            if cache_available:
                try:
                    await save_cached_suggestion(
                        db=db,
                        brand=request.brand,
                        search_type=request.search_type,
                        prompts=prompts,
                        competitors=competitors,
                        industry=request.industry,
                    )
                except Exception as e:
                    print(f"[Suggest] Failed to cache result: {e}")

            return SuggestResponse(
                brand=request.brand,
                prompts=prompts,
                competitors=competitors,
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate suggestions: {str(e)}",
        )
