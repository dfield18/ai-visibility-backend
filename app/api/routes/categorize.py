"""Categorize endpoint for classifying domains using AI with caching."""

from uuid import uuid4

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.deps import DatabaseDep
from app.models.cached_category import CachedCategory
from app.schemas.categorize import CategorizeRequest, CategorizeResponse
from app.services.openai_service import OpenAIService

router = APIRouter()


@router.post("/categorize", response_model=CategorizeResponse)
async def categorize_domains(
    request: CategorizeRequest,
    db: DatabaseDep,
) -> CategorizeResponse:
    """Categorize domains into content categories using AI with caching.

    First checks the cache for existing categorizations, then uses OpenAI
    to classify only the uncached domains. New categorizations are stored
    in the cache for future use.

    Args:
        request: CategorizeRequest containing list of domains to categorize.
        db: Database session dependency.

    Returns:
        CategorizeResponse with dict mapping domains to categories.

    Raises:
        HTTPException: If categorization fails.
    """
    domains = request.domains
    categories: dict[str, str] = {}
    uncached_domains: list[str] = []

    # Check cache for existing categorizations
    if domains:
        result = await db.execute(
            select(CachedCategory).where(CachedCategory.domain.in_(domains))
        )
        cached_entries = result.scalars().all()

        # Build dict of cached categories
        cached_map = {entry.domain: entry.category for entry in cached_entries}

        # Separate cached and uncached domains
        for domain in domains:
            if domain in cached_map:
                categories[domain] = cached_map[domain]
            else:
                uncached_domains.append(domain)

    # If all domains were cached, return immediately
    if not uncached_domains:
        return CategorizeResponse(categories=categories)

    # Get OpenAI categorizations for uncached domains
    try:
        service = OpenAIService()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=f"OpenAI service not available: {str(e)}",
        )

    try:
        new_categories = await service.categorize_domains(domains=uncached_domains)

        # Cache the new categorizations
        for domain, category in new_categories.items():
            cached_entry = CachedCategory(
                id=uuid4(),
                domain=domain,
                category=category,
            )
            db.add(cached_entry)
            categories[domain] = category

        await db.commit()

        return CategorizeResponse(categories=categories)
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to categorize domains: {str(e)}",
        )
