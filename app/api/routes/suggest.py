"""Suggest endpoint for generating prompts and competitors."""

from fastapi import APIRouter, HTTPException

from app.schemas.suggest import SuggestRequest, SuggestResponse
from app.services.openai_service import OpenAIService

router = APIRouter()


@router.post("/suggest", response_model=SuggestResponse)
async def generate_suggestions(request: SuggestRequest) -> SuggestResponse:
    """Generate suggested prompts and competitors for a brand.

    Uses GPT-4o to generate consumer search queries and identify
    competitors for the specified brand.

    Args:
        request: SuggestRequest containing brand name and optional industry.

    Returns:
        SuggestResponse with generated prompts and competitors.

    Raises:
        HTTPException: If suggestion generation fails.
    """
    try:
        service = OpenAIService()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=f"OpenAI service not available: {str(e)}",
        )

    # Generate prompts and competitors in parallel
    try:
        prompts = await service.generate_prompts(
            brand=request.brand,
            industry=request.industry,
        )
        competitors = await service.generate_competitors(
            brand=request.brand,
            industry=request.industry,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate suggestions: {str(e)}",
        )

    return SuggestResponse(
        brand=request.brand,
        prompts=prompts,
        competitors=competitors,
    )
