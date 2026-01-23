"""Gemini service for Google AI API calls."""

from typing import Any, Dict, List, Optional

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
        sources: List of source citations (URL and title).
        grounding_metadata: Full grounding metadata including supports with confidence scores.
    """

    def __init__(
        self,
        text: str,
        tokens_input: int,
        tokens_output: int,
        model: str = "gemini-2.0-flash",
        sources: Optional[List[Dict[str, str]]] = None,
        grounding_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.model = model
        self.sources = sources or []
        self.grounding_metadata = grounding_metadata
        # Gemini Flash pricing: $0.000075/1K input, $0.0003/1K output
        # Grounding adds ~$35/1K requests
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
            # Enable Google Search grounding for web search and citations
            # Use google_search for newer models (2.0+), google_search_retrieval for older
            "tools": [{"google_search": {}}],
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                params={"key": self.api_key},
                json=payload,
            )
            if not response.is_success:
                print(f"[Gemini] Error response: {response.status_code} - {response.text}")
            response.raise_for_status()
            data = response.json()

        # Extract text from response
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in Gemini response")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        text = parts[0].get("text", "") if parts else ""

        # Extract sources from grounding metadata
        sources = []
        grounding_metadata = candidates[0].get("groundingMetadata", {})
        grounding_chunks = grounding_metadata.get("groundingChunks", [])

        # Debug: log the full grounding metadata structure
        print(f"[Gemini] Grounding metadata keys: {grounding_metadata.keys()}")

        for chunk in grounding_chunks:
            web_chunk = chunk.get("web", {})
            if web_chunk:
                url = web_chunk.get("uri", "")
                title = web_chunk.get("title", "")

                # Check if this is a Vertex AI Search proxy URL and try to extract real URL
                if "vertexaisearch.cloud.google.com" in url:
                    # Try to parse query params for the real URL
                    try:
                        from urllib.parse import urlparse, parse_qs
                        parsed = urlparse(url)
                        query_params = parse_qs(parsed.query)
                        # Common param names for redirect URLs
                        for param in ["url", "q", "redirect", "target", "dest"]:
                            if param in query_params:
                                url = query_params[param][0]
                                break
                    except Exception as e:
                        print(f"[Gemini] Could not parse proxy URL: {e}")

                sources.append({
                    "url": url,
                    "title": title,
                })

        # Also check for sources in searchEntryPoint or supportingDocuments
        search_entry = grounding_metadata.get("searchEntryPoint", {})
        if search_entry:
            print(f"[Gemini] Search entry point: {search_entry}")

        # Check retrievalMetadata for additional sources
        retrieval_metadata = grounding_metadata.get("retrievalMetadata", {})
        if retrieval_metadata:
            print(f"[Gemini] Retrieval metadata: {retrieval_metadata}")

        # Check webSearchQueries for context
        web_queries = grounding_metadata.get("webSearchQueries", [])
        if web_queries:
            print(f"[Gemini] Web search queries: {web_queries}")

        # Build structured grounding metadata with supports (confidence scores)
        grounding_supports = grounding_metadata.get("groundingSupports", [])
        structured_grounding = None
        if grounding_supports or grounding_chunks:
            structured_grounding = {
                "supports": [
                    {
                        "segment": support.get("segment", {}).get("text", ""),
                        "chunk_indices": support.get("groundingChunkIndices", []),
                        "confidence_scores": support.get("confidenceScores", []),
                    }
                    for support in grounding_supports
                ],
                "search_queries": grounding_metadata.get("webSearchQueries", []),
            }

        print(f"[Gemini] Found {len(sources)} sources, {len(grounding_supports)} grounding supports")

        # Extract token counts from usage metadata
        usage = data.get("usageMetadata", {})
        tokens_input = usage.get("promptTokenCount", 0)
        tokens_output = usage.get("candidatesTokenCount", 0)

        return GeminiResponse(
            text=text,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=self.MODEL,
            sources=sources,
            grounding_metadata=structured_grounding,
        )
