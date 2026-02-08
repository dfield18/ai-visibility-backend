"""Llama (Meta) service for Llama API calls."""

from typing import Dict, List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings


class LlamaResponse:
    """Structured response from Llama API.

    Attributes:
        text: The generated text content.
        tokens_input: Number of input tokens.
        tokens_output: Number of output tokens.
        cost: Estimated cost in dollars.
        model: Model used for generation.
        sources: List of source citations (URL and title).
    """

    # Pricing per 1M tokens (estimated)
    MODEL_PRICING = {
        "Llama-4-Scout-17B-16E-Instruct-FP8": {"input": 0.0002, "output": 0.0008},
        "Llama-4-Maverick-17B-128E-Instruct-FP8": {"input": 0.0003, "output": 0.001},
        "meta-llama/Llama-3.3-70B-Instruct": {"input": 0.0005, "output": 0.001},
        "llama3.1-70b": {"input": 0.0005, "output": 0.001},
        "llama3.1-8b": {"input": 0.0001, "output": 0.0002},
    }

    def __init__(
        self,
        text: str,
        tokens_input: int,
        tokens_output: int,
        model: str = "llama3.1-70b",
        sources: Optional[List[Dict[str, str]]] = None,
    ):
        self.text = text
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.model = model
        self.sources = sources or []
        # Calculate cost based on model
        pricing = self.MODEL_PRICING.get(model, {"input": 0.0003, "output": 0.001})
        self.cost = (tokens_input * pricing["input"] / 1000) + (tokens_output * pricing["output"] / 1000)


class LlamaService:
    """Service for interacting with Meta's Llama API.

    Uses httpx for async HTTP calls.
    Llama API is OpenAI-compatible, so it uses similar endpoints and format.
    Includes retry logic for handling rate limits and transient errors.
    """

    # Meta's official Llama API (OpenAI-compatible)
    BASE_URL = "https://api.llama.com/compat/v1"
    MODEL = "llama3.1-70b"  # Default model

    def __init__(self):
        """Initialize the Llama service."""
        self.api_key = settings.LLAMA_API_KEY
        if not self.api_key:
            raise ValueError("LLAMA_API_KEY not configured")
        print(f"[Llama] Service initialized with API key: {self.api_key[:10]}...")

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
        model: str = "llama3.1-70b",
    ) -> LlamaResponse:
        """Generate content using Llama.

        Args:
            prompt: The prompt to send.
            temperature: Sampling temperature (0.0-2.0).
            model: Llama model to use.

        Returns:
            LlamaResponse with generated text and usage stats.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 2048,
        }

        print(f"[Llama] Making API call with model {model}")

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                )
                print(f"[Llama] Got response with status: {response.status_code}")
                if not response.is_success:
                    print(f"[Llama] Error response: {response.status_code} - {response.text}")
                response.raise_for_status()
                data = response.json()
                print(f"[Llama] Successfully parsed JSON response")
        except httpx.TimeoutException as e:
            print(f"[Llama] TIMEOUT ERROR: {e}")
            raise
        except httpx.HTTPStatusError as e:
            print(f"[Llama] HTTP ERROR: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            print(f"[Llama] UNEXPECTED ERROR: {type(e).__name__}: {e}")
            raise

        # Extract content from response (OpenAI-compatible format)
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # Llama doesn't have web search, so no sources
        sources = []

        print(f"[Llama] Response received, length: {len(content)}")

        return LlamaResponse(
            text=content,
            tokens_input=usage.get("prompt_tokens", 0),
            tokens_output=usage.get("completion_tokens", 0),
            model=model,
            sources=sources,
        )
