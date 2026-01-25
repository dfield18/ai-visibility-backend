"""SerpAPI service for Google AI Overviews."""

import json
from typing import Dict, List, Optional

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
        sources: List of source citations (URL and title).
    """

    def __init__(
        self,
        text: str,
        tokens_input: int,
        tokens_output: int,
        model: str = "google-ai-overview",
        sources: Optional[List[Dict[str, str]]] = None,
    ):
        self.text = text
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.model = model
        self.sources = sources or []
        # SerpAPI pricing: $0.005 per search (approximate)
        self.cost = 0.005


class SerpAPIService:
    """Service for fetching Google AI Overviews via SerpAPI.

    Uses httpx for async HTTP calls.
    Includes retry logic for handling rate limits and transient errors.
    """

    BASE_URL = "https://serpapi.com/search.json"

    def __init__(self):
        """Initialize the SerpAPI service."""
        self.api_key = settings.SERPAPI_API_KEY
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY not configured")

    # Map country codes to language codes
    COUNTRY_TO_LANGUAGE = {
        "us": "en",
        "gb": "en",
        "ca": "en",
        "au": "en",
        "de": "de",
        "fr": "fr",
        "es": "es",
        "it": "it",
        "nl": "nl",
        "br": "pt",
        "mx": "es",
        "in": "en",
        "jp": "ja",
        "kr": "ko",
        "sg": "en",
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
        temperature: float = 0.7,  # Not used for SerpAPI but kept for interface consistency
        country: str = "us",
    ) -> SerpAPIResponse:
        """Fetch AI Overview from Google via SerpAPI.

        Args:
            prompt: The search query to send.
            temperature: Not used, kept for interface consistency.
            country: Country code for filtering search results (e.g., 'us', 'gb', 'de').

        Returns:
            SerpAPIResponse with AI Overview text.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
            ValueError: If no AI Overview is available for the query.
        """
        # Get language for the country, default to English
        language = self.COUNTRY_TO_LANGUAGE.get(country.lower(), "en")

        params = {
            "q": prompt,
            "api_key": self.api_key,
            "engine": "google",
            "gl": country.lower(),
            "hl": language,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(self.BASE_URL, params=params)
            if not response.is_success:
                print(f"[SerpAPI] Error response: {response.status_code} - {response.text}")
            response.raise_for_status()
            data = response.json()

        # Debug: log full response
        print(f"[SerpAPI] ===== FULL RESPONSE for '{prompt[:50]}...' =====")
        print(json.dumps(data, indent=2, default=str))
        print(f"[SerpAPI] ===== END FULL RESPONSE =====")

        text = ""
        sources = []

        # Check for lazy-loaded AI Overview (requires second API call)
        page_token = data.get("page_token")
        if page_token:
            print(f"[SerpAPI] Warning: AI Overview is lazy-loaded (page_token present). Content may be incomplete.")

        # Try multiple possible field names for AI Overview
        ai_overview = data.get("ai_overview")

        if ai_overview:
            print(f"[SerpAPI] Found ai_overview field with keys: {list(ai_overview.keys()) if isinstance(ai_overview, dict) else 'string'}")
            if isinstance(ai_overview, dict):
                text_parts = []

                # Collect text from all available fields
                if ai_overview.get("text"):
                    text_parts.append(ai_overview.get("text"))
                if ai_overview.get("snippet"):
                    text_parts.append(ai_overview.get("snippet"))
                if ai_overview.get("answer"):
                    text_parts.append(ai_overview.get("answer"))

                # Get text from text_blocks (often contains the full content)
                # text_blocks can have types: heading, paragraph, list, expandable, comparison
                text_blocks = ai_overview.get("text_blocks", [])
                if text_blocks:
                    for block in text_blocks:
                        if isinstance(block, dict):
                            block_type = block.get("type", "paragraph")
                            # Primary field is 'snippet' according to SerpAPI docs
                            block_text = block.get("snippet") or block.get("text")

                            if block_text:
                                # Format short paragraphs without periods as headings (markdown bold)
                                is_heading = (
                                    block_type == "heading" or
                                    (len(block_text) < 60 and not block_text.endswith('.') and not block_text.endswith(':'))
                                )
                                if is_heading:
                                    text_parts.append(f"**{block_text}**")
                                else:
                                    text_parts.append(block_text)

                            # Handle nested list items within text_block (type: list)
                            block_list = block.get("list", [])
                            if block_list:
                                for list_item in block_list:
                                    if isinstance(list_item, dict):
                                        item_text = list_item.get("snippet") or list_item.get("title") or list_item.get("text")
                                        if item_text:
                                            text_parts.append(f"• {item_text}")
                                    elif isinstance(list_item, str):
                                        text_parts.append(f"• {list_item}")
                        elif isinstance(block, str):
                            text_parts.append(block)

                # Get text from list items if present
                list_items = ai_overview.get("list", []) or ai_overview.get("items", [])
                if list_items:
                    for item in list_items:
                        if isinstance(item, dict):
                            item_text = item.get("title", "") or item.get("text", "") or item.get("snippet", "")
                            if item_text:
                                text_parts.append(f"• {item_text}")
                        elif isinstance(item, str):
                            text_parts.append(f"• {item}")

                # Get text from source blocks
                source_blocks = ai_overview.get("source", [])
                if source_blocks and isinstance(source_blocks, list):
                    for s in source_blocks:
                        if isinstance(s, dict) and s.get("snippet"):
                            text_parts.append(s.get("snippet"))

                # Combine all text parts, removing duplicates while preserving order
                seen = set()
                unique_parts = []
                for part in text_parts:
                    if part and part not in seen:
                        seen.add(part)
                        unique_parts.append(part)
                text = "\n\n".join(unique_parts)

                # Extract sources from ai_overview (check multiple locations)
                ai_sources = ai_overview.get("sources", []) or ai_overview.get("source", [])
                if ai_sources and isinstance(ai_sources, list):
                    for src in ai_sources:
                        if isinstance(src, dict):
                            sources.append({
                                "url": src.get("link", "") or src.get("url", ""),
                                "title": src.get("title", "") or src.get("name", ""),
                            })

                # Extract references from inside ai_overview (this is where SerpAPI puts them)
                ai_references = ai_overview.get("references", [])
                if ai_references and isinstance(ai_references, list):
                    print(f"[SerpAPI] Found {len(ai_references)} references inside ai_overview")
                    for ref in ai_references:
                        if isinstance(ref, dict):
                            ref_url = ref.get("link", "") or ref.get("url", "")
                            ref_title = ref.get("title", "") or ref.get("name", "")
                            if ref_url and not any(s.get("url") == ref_url for s in sources):
                                sources.append({"url": ref_url, "title": ref_title})

            elif isinstance(ai_overview, str):
                text = ai_overview

        # Also check root-level references field (fallback)
        references = data.get("references", [])
        if references and isinstance(references, list):
            print(f"[SerpAPI] Found {len(references)} root-level references")
            for ref in references:
                if isinstance(ref, dict):
                    ref_url = ref.get("link", "") or ref.get("url", "")
                    ref_title = ref.get("title", "") or ref.get("name", "")
                    if ref_url and not any(s.get("url") == ref_url for s in sources):
                        sources.append({"url": ref_url, "title": ref_title})

        # Try answer_box if no ai_overview
        if not text:
            answer_box = data.get("answer_box")
            if answer_box:
                print(f"[SerpAPI] Found answer_box field: {list(answer_box.keys()) if isinstance(answer_box, dict) else type(answer_box)}")
                if isinstance(answer_box, dict):
                    text = answer_box.get("answer", "") or answer_box.get("snippet", "") or answer_box.get("result", "")
                    if not text and answer_box.get("contents"):
                        contents = answer_box.get("contents", {})
                        if isinstance(contents, dict):
                            text = contents.get("answer", "") or contents.get("snippet", "")
                    # Extract source from answer_box
                    if answer_box.get("link"):
                        sources.append({
                            "url": answer_box.get("link", ""),
                            "title": answer_box.get("title", ""),
                        })

        # Try knowledge_graph
        if not text:
            knowledge_graph = data.get("knowledge_graph")
            if knowledge_graph:
                print(f"[SerpAPI] Found knowledge_graph field")
                if isinstance(knowledge_graph, dict):
                    text = knowledge_graph.get("description", "") or knowledge_graph.get("snippet", "")
                    # Extract source from knowledge_graph
                    if knowledge_graph.get("source") and isinstance(knowledge_graph.get("source"), dict):
                        kg_source = knowledge_graph.get("source")
                        sources.append({
                            "url": kg_source.get("link", ""),
                            "title": kg_source.get("name", ""),
                        })

        # Try featured_snippet as fallback
        if not text:
            featured_snippet = data.get("featured_snippet")
            if featured_snippet:
                print(f"[SerpAPI] Found featured_snippet field")
                if isinstance(featured_snippet, dict):
                    text = featured_snippet.get("snippet", "") or featured_snippet.get("description", "")
                    # Extract source from featured_snippet
                    if featured_snippet.get("link"):
                        sources.append({
                            "url": featured_snippet.get("link", ""),
                            "title": featured_snippet.get("title", ""),
                        })

        # Try organic_results first result as last resort
        if not text:
            organic_results = data.get("organic_results", [])
            if organic_results and len(organic_results) > 0:
                print(f"[SerpAPI] Using first organic result as fallback")
                first_result = organic_results[0]
                text = first_result.get("snippet", "")
                # Add first few organic results as sources
                for result in organic_results[:3]:
                    sources.append({
                        "url": result.get("link", ""),
                        "title": result.get("title", ""),
                    })

        if not text:
            print(f"[SerpAPI] No content found for query: {prompt}")
            raise ValueError(f"No AI Overview or similar content available for query: {prompt}")

        # If no sources yet, add top organic results as sources
        if not sources:
            organic_results = data.get("organic_results", [])
            for result in organic_results[:3]:
                sources.append({
                    "url": result.get("link", ""),
                    "title": result.get("title", ""),
                })

        # Filter out empty sources
        sources = [s for s in sources if s.get("url")]

        print(f"[SerpAPI] Response with {len(sources)} sources")

        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        tokens_input = len(prompt) // 4
        tokens_output = len(text) // 4

        return SerpAPIResponse(
            text=text,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model="google-ai-overview",
            sources=sources,
        )
