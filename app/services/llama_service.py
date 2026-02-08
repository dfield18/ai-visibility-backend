"""Llama service for Groq API calls.

Uses Groq (groq.com) for fast Llama model inference.
"""

import logging
from typing import Dict, List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class LlamaResponse:
    """Structured response from Groq/Llama API.

    Attributes:
        text: The generated text content.
        tokens_input: Number of input tokens.
        tokens_output: Number of output tokens.
        cost: Estimated cost in dollars.
        model: Model used for generation.
        sources: List of source citations (URL and title).
    """

    # Groq pricing per 1M tokens (as of 2024)
    # https://groq.com/pricing/
    MODEL_PRICING = {
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
        "meta-llama/llama-4-maverick-17b-128e-instruct": {"input": 0.20, "output": 0.60},
        "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11, "output": 0.34},
    }

    def __init__(
        self,
        text: str,
        tokens_input: int,
        tokens_output: int,
        model: str = "llama-3.3-70b-versatile",
        sources: Optional[List[Dict[str, str]]] = None,
    ):
        self.text = text
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.model = model
        self.sources = sources or []
        # Calculate cost based on model (prices are per 1M tokens)
        pricing = self.MODEL_PRICING.get(model, {"input": 0.59, "output": 0.79})
        self.cost = (tokens_input * pricing["input"] / 1_000_000) + (tokens_output * pricing["output"] / 1_000_000)


class LlamaService:
    """Service for interacting with Groq API for Llama models.

    Uses httpx for async HTTP calls.
    Groq API is OpenAI-compatible, so it uses similar endpoints and format.
    Includes retry logic for handling rate limits and transient errors.
    """

    # Groq API base URL (OpenAI-compatible)
    BASE_URL = "https://api.groq.com/openai/v1"
    MODEL = "llama-3.3-70b-versatile"  # Default model - best quality Llama on Groq

    def __init__(self):
        """Initialize the Llama/Groq service."""
        self.api_key = settings.LLAMA_API_KEY
        if not self.api_key:
            raise ValueError("LLAMA_API_KEY not configured")
        logger.info(f"[Llama/Groq] Service initialized with API key: {self.api_key[:10]}...")

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
        model: str = "llama-3.3-70b-versatile",
    ) -> LlamaResponse:
        """Generate content using Llama via Groq.

        Args:
            prompt: The prompt to send.
            temperature: Sampling temperature (0.0-2.0).
            model: Llama model to use on Groq.

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

        logger.info(f"[Llama/Groq] Making API call to {self.BASE_URL}/chat/completions")
        logger.info(f"[Llama/Groq] Model: {model}, Temperature: {temperature}")
        logger.debug(f"[Llama/Groq] Prompt length: {len(prompt)} chars")

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                )
                logger.info(f"[Llama/Groq] Response status: {response.status_code}")

                if not response.is_success:
                    logger.error(f"[Llama/Groq] Error response: {response.status_code}")
                    logger.error(f"[Llama/Groq] Error body: {response.text}")
                    response.raise_for_status()

                data = response.json()
                logger.info(f"[Llama/Groq] Successfully parsed JSON response")
                logger.debug(f"[Llama/Groq] Response keys: {list(data.keys())}")

        except httpx.TimeoutException as e:
            logger.error(f"[Llama/Groq] TIMEOUT ERROR after 90s: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"[Llama/Groq] HTTP ERROR: {e.response.status_code}")
            logger.error(f"[Llama/Groq] HTTP ERROR body: {e.response.text}")
            raise
        except httpx.ConnectError as e:
            logger.error(f"[Llama/Groq] CONNECTION ERROR: {e}")
            raise
        except Exception as e:
            logger.error(f"[Llama/Groq] UNEXPECTED ERROR: {type(e).__name__}: {e}")
            raise

        # Extract content from response (OpenAI-compatible format)
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        logger.info(f"[Llama/Groq] Response received successfully")
        logger.info(f"[Llama/Groq] Content length: {len(content)} chars")
        logger.info(f"[Llama/Groq] Tokens - input: {tokens_in}, output: {tokens_out}")

        # Groq/Llama doesn't have web search, so no sources
        sources = []

        return LlamaResponse(
            text=content,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            model=model,
            sources=sources,
        )
