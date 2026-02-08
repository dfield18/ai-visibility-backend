"""Grok (xAI) service for Grok API calls."""

from typing import Dict, List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings


class GrokResponse:
    """Structured response from Grok API.

    Attributes:
        text: The generated text content.
        tokens_input: Number of input tokens.
        tokens_output: Number of output tokens.
        cost: Estimated cost in dollars.
        model: Model used for generation.
        sources: List of source citations (URL and title).
    """

    # Pricing per 1M tokens (xAI pricing as of 2024)
    # grok-2: $2 input, $10 output per 1M tokens
    # grok-2-mini: $0.30 input, $0.50 output per 1M tokens (estimated)
    MODEL_PRICING = {
        "grok-2": {"input": 0.002, "output": 0.010},  # per 1K tokens
        "grok-2-mini": {"input": 0.0003, "output": 0.0005},  # per 1K tokens
        "grok-3": {"input": 0.003, "output": 0.015},  # estimated
    }

    def __init__(
        self,
        text: str,
        tokens_input: int,
        tokens_output: int,
        model: str = "grok-2",
        sources: Optional[List[Dict[str, str]]] = None,
    ):
        self.text = text
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.model = model
        self.sources = sources or []
        # Calculate cost based on model
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["grok-2"])
        self.cost = (tokens_input * pricing["input"] / 1000) + (tokens_output * pricing["output"] / 1000)


class GrokService:
    """Service for interacting with xAI's Grok API.

    Uses httpx for async HTTP calls.
    Grok API is OpenAI-compatible, so it uses similar endpoints and format.
    Includes retry logic for handling rate limits and transient errors.
    """

    BASE_URL = "https://api.x.ai/v1"
    MODEL = "grok-2"  # Default model

    def __init__(self):
        """Initialize the Grok service."""
        self.api_key = settings.GROK_API_KEY
        if not self.api_key:
            raise ValueError("GROK_API_KEY not configured")
        print(f"[Grok] Service initialized with API key: {self.api_key[:10]}...")

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
        model: str = "grok-2",
    ) -> GrokResponse:
        """Generate content using Grok with web search capabilities.

        Grok has real-time access to X (Twitter) data and web information.

        Args:
            prompt: The prompt to send.
            temperature: Sampling temperature (0.0-2.0).
            model: Grok model to use (grok-2 default).

        Returns:
            GrokResponse with generated text and usage stats.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        # Prepend instruction for real-time search
        search_prompt = (
            f"Search for the most current and up-to-date information to answer this question. "
            f"Include any relevant sources you find: {prompt}"
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": search_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 2048,
        }

        print(f"[Grok] Making API call to {self.BASE_URL}/chat/completions with model {model}")

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                )
                print(f"[Grok] Got response with status: {response.status_code}")
                if not response.is_success:
                    print(f"[Grok] Error response: {response.status_code} - {response.text}")
                response.raise_for_status()
                data = response.json()
                print(f"[Grok] Successfully parsed JSON response")
        except httpx.TimeoutException as e:
            print(f"[Grok] TIMEOUT ERROR: {e}")
            raise
        except httpx.HTTPStatusError as e:
            print(f"[Grok] HTTP ERROR: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            print(f"[Grok] UNEXPECTED ERROR: {type(e).__name__}: {e}")
            raise

        # Extract content from response (OpenAI-compatible format)
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # Extract any sources/citations if available in the response
        # Grok doesn't have a standardized citation format yet, so we'll parse from content
        sources = []

        # Try to extract URLs from the response text
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls_found = re.findall(url_pattern, content)
        seen_urls = set()
        for url in urls_found:
            # Clean up URL (remove trailing punctuation)
            url = url.rstrip('.,;:!?)')
            if url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "url": url,
                    "title": "",  # Grok doesn't provide titles
                })

        print(f"[Grok] Response with {len(sources)} extracted sources")

        return GrokResponse(
            text=content,
            tokens_input=usage.get("prompt_tokens", 0),
            tokens_output=usage.get("completion_tokens", 0),
            model=model,
            sources=sources,
        )
