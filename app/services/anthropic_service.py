"""Anthropic service for Claude API calls."""

from typing import Dict, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings


class AnthropicResponse:
    """Structured response from Anthropic API.

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
        model: str = "claude-sonnet-4-20250514",
    ):
        self.text = text
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.model = model
        # Claude 3.5 Sonnet pricing: $0.003/1K input, $0.015/1K output
        self.cost = (tokens_input * 0.003 / 1000) + (tokens_output * 0.015 / 1000)


class AnthropicService:
    """Service for interacting with Anthropic's Claude API.

    Uses httpx for async HTTP calls.
    Includes retry logic for handling rate limits and transient errors.
    """

    BASE_URL = "https://api.anthropic.com/v1"
    MODEL = "claude-sonnet-4-20250514"
    API_VERSION = "2023-06-01"

    def __init__(self):
        """Initialize the Anthropic service."""
        self.api_key = settings.ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }

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
    ) -> AnthropicResponse:
        """Generate content using Claude.

        Args:
            prompt: The prompt to send.
            temperature: Sampling temperature (0.0-1.0).

        Returns:
            AnthropicResponse with generated text and usage stats.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        payload = {
            "model": self.MODEL,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": min(temperature, 1.0),  # Claude max temp is 1.0
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.BASE_URL}/messages",
                headers=self._get_headers(),
                json=payload,
            )
            if not response.is_success:
                print(f"[Anthropic] Error response: {response.status_code} - {response.text}")
            response.raise_for_status()
            data = response.json()

        # Extract text from response
        content = data.get("content", [])
        text = ""
        for block in content:
            if block.get("type") == "text":
                text += block.get("text", "")

        # Extract token counts from usage
        usage = data.get("usage", {})
        tokens_input = usage.get("input_tokens", 0)
        tokens_output = usage.get("output_tokens", 0)

        return AnthropicResponse(
            text=text,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=self.MODEL,
        )
