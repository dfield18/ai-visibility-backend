"""Suggest endpoint for generating prompts and competitors/brands."""

from fastapi import APIRouter, HTTPException

from app.schemas.suggest import SuggestRequest, SuggestResponse
from app.services.openai_service import OpenAIService

router = APIRouter()


@router.post("/suggest", response_model=SuggestResponse)
async def generate_suggestions(request: SuggestRequest) -> SuggestResponse:
    """Generate suggested prompts and competitors/brands.

    For brand searches: generates consumer search queries and identifies competitors.
    For category searches: generates search queries and identifies brands in that category.

    Args:
        request: SuggestRequest containing brand/category name, search_type, and optional industry.

    Returns:
        SuggestResponse with generated prompts and competitors/brands.

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

    try:
        if request.search_type == "category":
            # For category searches, generate prompts for the category
            # and list brands in that category
            prompts = await service.generate_category_prompts(
                category=request.brand,
                industry=request.industry,
            )
            brands = await service.generate_category_brands(
                category=request.brand,
                industry=request.industry,
            )
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
            competitors = await service.generate_competitors(
                brand=request.brand,
                industry=request.industry,
            )
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
