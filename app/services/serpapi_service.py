"""SerpAPI service for Google AI Overviews."""

from typing import Dict, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings


class SerpAPIResponse:
    """Structured response from SerpAPI.

    Attributes:
        text: The AI Overview text content.
        tokens_input: Estimated input tokens (based on query length).
        tokens_output: Estimated output tokens (based on response length).
        cost: Estimated cost in dollars.
        model: Model identifier.
    """

    def __init__(
        self,
        text: str,
        tokens_input: int,
        tokens_output: int,
        model: str = "google-ai-overview",
    ):
        self.text = text
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.model = model
        # SerpAPI pricing: $0.005 per search (approximate)
        self.cost = 0.005


class SerpAPIService:
    """Service for fetching Google AI Overviews via SerpAPI.

    Uses httpx for async HTTP calls.
    Includes retry logic for handling rate limits and transient errors.
    """

    BASE_URL = "https://serpapi.com/search"

    def __init__(self):
        """Initialize the SerpAPI service."""
        self.api_key = settings.SERPAPI_API_KEY
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY not configured")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
        reraise=True,
    )
    async def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,  # Not used for SerpAPI but kept for interface consistency
    ) -> SerpAPIResponse:
        """Fetch AI Overview from Google via SerpAPI.

        Args:
            prompt: The search query to send.
            temperature: Not used, kept for interface consistency.

        Returns:
            SerpAPIResponse with AI Overview text.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
            ValueError: If no AI Overview is available for the query.
        """
        params = {
            "q": prompt,
            "api_key": self.api_key,
            "engine": "google",
            "gl": "us",
            "hl": "en",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(self.BASE_URL, params=params)
            if not response.is_success:
                print(f"[SerpAPI] Error response: {response.status_code} - {response.text}")
            response.raise_for_status()
            data = response.json()

        # Extract AI Overview from response
        ai_overview = data.get("ai_overview")

        if not ai_overview:
            # Try alternative field names that SerpAPI might use
            ai_overview = data.get("answer_box", {}).get("ai_overview")

        if not ai_overview:
            # Check for featured snippet as fallback
            featured_snippet = data.get("featured_snippet")
            if featured_snippet:
                text = featured_snippet.get("snippet", "")
                if not text:
                    text = featured_snippet.get("description", "")
            else:
                # No AI Overview available for this query
                raise ValueError(f"No AI Overview available for query: {prompt}")
        else:
            # Extract text from AI Overview
            if isinstance(ai_overview, dict):
                text = ai_overview.get("text", "")
                if not text:
                    # Try to get text from nested structure
                    text_blocks = ai_overview.get("text_blocks", [])
                    if text_blocks:
                        text = " ".join([block.get("text", "") for block in text_blocks if block.get("text")])
                    else:
                        # Try snippet
                        text = ai_overview.get("snippet", "")
            elif isinstance(ai_overview, str):
                text = ai_overview
            else:
                text = str(ai_overview)

        if not text:
            raise ValueError(f"Empty AI Overview for query: {prompt}")

        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        tokens_input = len(prompt) // 4
        tokens_output = len(text) // 4

        return SerpAPIResponse(
            text=text,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model="google-ai-overview",
        )
