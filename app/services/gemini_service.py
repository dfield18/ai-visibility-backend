"""Gemini service for Google AI API calls."""

from typing import Dict, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings


class GeminiResponse:
    """Structured response from Gemini API.

    Attributes:
        text: The generated text content.
        tokens_input: Number of input tokens.
        tokens_output: Number of output tokens.
        cost: Estimated cost in dollars.
        model: Model used for generation.
    """

    def __init__(
        self,
        text: str,
        tokens_input: int,
        tokens_output: int,
        model: str = "gemini-2.0-flash",
    ):
        self.text = text
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.model = model
        # Gemini Flash pricing: $0.000075/1K input, $0.0003/1K output
        self.cost = (tokens_input * 0.000075 / 1000) + (tokens_output * 0.0003 / 1000)


class GeminiService:
    """Service for interacting with Google's Gemini API.

    Uses httpx for async HTTP calls.
    Includes retry logic for handling rate limits and transient errors.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    MODEL = "gemini-2.0-flash"

    def __init__(self):
        """Initialize the Gemini service."""
        self.api_key = settings.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
        reraise=True,
    )
    async def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
    ) -> GeminiResponse:
        """Generate content using Gemini.

        Args:
            prompt: The prompt to send.
            temperature: Sampling temperature (0.0-2.0).

        Returns:
            GeminiResponse with generated text and usage stats.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        url = f"{self.BASE_URL}/models/{self.MODEL}:generateContent"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
            },
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                params={"key": self.api_key},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        # Extract text from response
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in Gemini response")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        text = parts[0].get("text", "") if parts else ""

        # Extract token counts from usage metadata
        usage = data.get("usageMetadata", {})
        tokens_input = usage.get("promptTokenCount", 0)
        tokens_output = usage.get("candidatesTokenCount", 0)

        return GeminiResponse(
            text=text,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=self.MODEL,
        )
