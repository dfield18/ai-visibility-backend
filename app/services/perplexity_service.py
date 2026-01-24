"""Perplexity service for Sonar API calls."""

from typing import Dict, List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings


class PerplexityResponse:
    """Structured response from Perplexity API.

    Attributes:
        text: The generated text content.
        tokens_input: Number of input tokens.
        tokens_output: Number of output tokens.
        cost: Estimated cost in dollars.
        model: Model used for generation.
        sources: List of source citations (URL and title).
    """

    def __init__(
        self,
        text: str,
        tokens_input: int,
        tokens_output: int,
        model: str = "sonar",
        sources: Optional[List[Dict[str, str]]] = None,
    ):
        self.text = text
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.model = model
        self.sources = sources or []
        # Perplexity Sonar pricing: ~$0.001/1K input, $0.001/1K output
        self.cost = (tokens_input * 0.001 / 1000) + (tokens_output * 0.001 / 1000)


class PerplexityService:
    """Service for interacting with Perplexity's Sonar API.

    Uses httpx for async HTTP calls.
    Includes retry logic for handling rate limits and transient errors.
    """

    BASE_URL = "https://api.perplexity.ai"
    MODEL = "sonar"

    def __init__(self):
        """Initialize the Perplexity service."""
        self.api_key = settings.PERPLEXITY_API_KEY
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not configured")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
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
        country: str = "us",
    ) -> PerplexityResponse:
        """Generate content using Perplexity Sonar.

        Args:
            prompt: The prompt to send.
            temperature: Sampling temperature (0.0-2.0).
            country: Two-letter ISO country code for location filtering (e.g., 'us', 'gb').

        Returns:
            PerplexityResponse with generated text and usage stats.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        payload = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "web_search_options": {
                "user_location": {
                    "country": country.upper(),
                }
            },
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )
            if not response.is_success:
                print(f"[Perplexity] Error response: {response.status_code} - {response.text}")
            response.raise_for_status()
            data = response.json()

        # Extract text from response (OpenAI-compatible format)
        choices = data.get("choices", [])
        text = ""
        if choices:
            message = choices[0].get("message", {})
            text = message.get("content", "")

        # Extract token counts from usage
        usage = data.get("usage", {})
        tokens_input = usage.get("prompt_tokens", 0)
        tokens_output = usage.get("completion_tokens", 0)

        # Extract citations/sources
        sources = []

        # Try search_results first (includes title, url, date)
        search_results = data.get("search_results", [])
        if search_results:
            for result in search_results:
                sources.append({
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                })
        else:
            # Fall back to citations array (just URLs)
            citations = data.get("citations", [])
            for url in citations:
                if isinstance(url, str):
                    sources.append({
                        "url": url,
                        "title": "",
                    })

        print(f"[Perplexity] Response with {len(sources)} sources")

        return PerplexityResponse(
            text=text,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=self.MODEL,
            sources=sources,
        )
