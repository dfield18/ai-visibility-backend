"""Categorize endpoint for classifying domains using AI."""

from fastapi import APIRouter, HTTPException

from app.schemas.categorize import CategorizeRequest, CategorizeResponse
from app.services.openai_service import OpenAIService

router = APIRouter()


@router.post("/categorize", response_model=CategorizeResponse)
async def categorize_domains(request: CategorizeRequest) -> CategorizeResponse:
    """Categorize domains into content categories using AI.

    Uses OpenAI to classify domains that can't be categorized by predefined rules.

    Args:
        request: CategorizeRequest containing list of domains to categorize.

    Returns:
        CategorizeResponse with dict mapping domains to categories.

    Raises:
        HTTPException: If categorization fails.
    """
    try:
        service = OpenAIService()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=f"OpenAI service not available: {str(e)}",
        )

    try:
        categories = await service.categorize_domains(domains=request.domains)
        return CategorizeResponse(categories=categories)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to categorize domains: {str(e)}",
        )
