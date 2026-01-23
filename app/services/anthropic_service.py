"""Anthropic service for Claude API calls."""

from typing import Any, Dict, List, Optional

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
        sources: List of source citations (URL and title).
    """

    # Pricing per 1K tokens
    MODEL_PRICING = {
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    }

    # Web search pricing: $10 per 1,000 searches = $0.01 per search
    WEB_SEARCH_COST = 0.01

    def __init__(
        self,
        text: str,
        tokens_input: int,
        tokens_output: int,
        model: str = "claude-3-haiku-20240307",
        sources: Optional[List[Dict[str, str]]] = None,
        web_search_requests: int = 0,
    ):
        self.text = text
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.model = model
        self.sources = sources or []
        # Calculate cost based on model + web search
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["claude-3-haiku-20240307"])
        self.cost = (
            (tokens_input * pricing["input"] / 1000) +
            (tokens_output * pricing["output"] / 1000) +
            (web_search_requests * self.WEB_SEARCH_COST)
        )


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
        model: str = "claude-sonnet-4-20250514",
    ) -> AnthropicResponse:
        """Generate content using Claude with web search enabled.

        Args:
            prompt: The prompt to send.
            temperature: Sampling temperature (0.0-1.0).
            model: Claude model to use (claude-sonnet-4-20250514 recommended for web search).

        Returns:
            AnthropicResponse with generated text, usage stats, and sources.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        payload = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": min(temperature, 1.0),  # Claude max temp is 1.0
            # Enable web search tool for real-time information and citations
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }],
        }

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{self.BASE_URL}/messages",
                headers=self._get_headers(),
                json=payload,
            )
            if not response.is_success:
                print(f"[Anthropic] Error response: {response.status_code} - {response.text}")
            response.raise_for_status()
            data = response.json()

        # Extract text and citations from response
        content = data.get("content", [])
        text_parts = []
        sources = []
        seen_urls = set()

        for block in content:
            block_type = block.get("type")

            if block_type == "text":
                text_parts.append(block.get("text", ""))

                # Extract citations from text blocks
                citations = block.get("citations", [])
                for citation in citations:
                    if citation.get("type") == "web_search_result_location":
                        url = citation.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            sources.append({
                                "url": url,
                                "title": citation.get("title", ""),
                            })

            elif block_type == "web_search_tool_result":
                # Also extract sources from search results
                search_content = block.get("content", [])
                for result in search_content:
                    if result.get("type") == "web_search_result":
                        url = result.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            sources.append({
                                "url": url,
                                "title": result.get("title", ""),
                            })

        text = "".join(text_parts)

        # Extract token counts and web search usage
        usage = data.get("usage", {})
        tokens_input = usage.get("input_tokens", 0)
        tokens_output = usage.get("output_tokens", 0)
        server_tool_use = usage.get("server_tool_use", {})
        web_search_requests = server_tool_use.get("web_search_requests", 0)

        print(f"[Anthropic] Response with {len(sources)} sources, {web_search_requests} web searches")

        return AnthropicResponse(
            text=text,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=model,
            sources=sources,
            web_search_requests=web_search_requests,
        )
