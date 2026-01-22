"""OpenAI service for GPT-4o API calls."""

import json
import re
from typing import Any, Dict, List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings


class OpenAIResponse:
    """Structured response from OpenAI API.

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
        model: str = "gpt-4o",
        sources: Optional[List[Dict[str, str]]] = None,
    ):
        self.text = text
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.model = model
        self.sources = sources or []
        # GPT-4o pricing: $0.005/1K input, $0.015/1K output
        # Web search adds ~$25/1K searches
        self.cost = (tokens_input * 0.005 / 1000) + (tokens_output * 0.015 / 1000)


class OpenAIService:
    """Service for interacting with OpenAI's GPT-4o API.

    Uses httpx for async HTTP calls instead of the sync SDK.
    Includes retry logic for handling rate limits and transient errors.
    """

    BASE_URL = "https://api.openai.com/v1"
    MODEL = "gpt-4o"

    def __init__(self):
        """Initialize the OpenAI service."""
        self.api_key = settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not configured")

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
    async def chat_completion(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> OpenAIResponse:
        """Make a chat completion request to GPT-4o.

        Args:
            user_prompt: The user message to send.
            system_prompt: Optional system message for context.
            temperature: Sampling temperature (0.0-2.0).

        Returns:
            OpenAIResponse with generated text and usage stats.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.MODEL,
            "messages": messages,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return OpenAIResponse(
            text=content,
            tokens_input=usage.get("prompt_tokens", 0),
            tokens_output=usage.get("completion_tokens", 0),
            model=self.MODEL,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
        reraise=True,
    )
    async def search_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
    ) -> OpenAIResponse:
        """Make a request using Responses API with web search for citations.

        Args:
            prompt: The search query/prompt to send.
            temperature: Sampling temperature (0.0-2.0).

        Returns:
            OpenAIResponse with generated text, usage stats, and sources.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        # Prepend instruction to search the web for current information
        search_prompt = f"Search the web for current, up-to-date information to answer: {prompt}"

        payload = {
            "model": self.MODEL,
            "input": search_prompt,
            "tools": [{"type": "web_search_preview"}],
            "tool_choice": "required",  # Force the model to use web search
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{self.BASE_URL}/responses",
                headers=self._get_headers(),
                json=payload,
            )
            if not response.is_success:
                print(f"[OpenAI] Error response: {response.status_code} - {response.text}")
            response.raise_for_status()
            data = response.json()

        # Debug: log raw response structure
        print(f"[OpenAI] Raw response keys: {data.keys()}")

        # Extract text from output
        text = ""
        sources = []

        # Parse the output array for text and citations
        output = data.get("output", [])
        print(f"[OpenAI] Output items: {len(output)}")

        for item in output:
            item_type = item.get("type")
            print(f"[OpenAI] Output item type: {item_type}")

            if item_type == "message":
                content = item.get("content", [])
                for content_item in content:
                    content_type = content_item.get("type")
                    print(f"[OpenAI] Content item type: {content_type}")

                    if content_type == "output_text":
                        text = content_item.get("text", "")
                        # Extract annotations (citations)
                        annotations = content_item.get("annotations", [])
                        print(f"[OpenAI] Found {len(annotations)} annotations")
                        for annotation in annotations:
                            ann_type = annotation.get("type")
                            if ann_type == "url_citation":
                                sources.append({
                                    "url": annotation.get("url", ""),
                                    "title": annotation.get("title", ""),
                                })
                            else:
                                print(f"[OpenAI] Unknown annotation type: {ann_type}")

            # Also check for web_search_call results which may contain sources
            elif item_type == "web_search_call":
                print(f"[OpenAI] Web search call found: {item}")
                # Extract search results if available
                if "status" in item:
                    print(f"[OpenAI] Web search status: {item.get('status')}")
            else:
                # Log any other output types we might be missing
                print(f"[OpenAI] Other output item: {item}")

        # If no sources found via annotations, check for alternative locations
        if not sources:
            # Check if there's a 'citations' key at the top level
            if "citations" in data:
                print(f"[OpenAI] Found top-level citations: {len(data['citations'])}")
                for citation in data.get("citations", []):
                    sources.append({
                        "url": citation.get("url", ""),
                        "title": citation.get("title", citation.get("name", "")),
                    })

            # Check in output_text for inline citations
            if not sources and text:
                # Log a sample of the text to see if citations are embedded differently
                print(f"[OpenAI] Text sample (first 200 chars): {text[:200]}...")

        # Extract token counts from usage
        usage = data.get("usage", {})
        tokens_input = usage.get("input_tokens", 0)
        tokens_output = usage.get("output_tokens", 0)

        print(f"[OpenAI] Search completed with {len(sources)} sources")

        return OpenAIResponse(
            text=text,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=self.MODEL,
            sources=sources,
        )

    async def generate_prompts(self, brand: str, industry: Optional[str] = None) -> List[str]:
        """Generate category-based prompts for a brand.

        Args:
            brand: The brand name.
            industry: Optional industry context.

        Returns:
            List of suggested search queries (category-based, not brand-specific).
        """
        system_prompt = (
            "You are a market research assistant helping brands understand their "
            "visibility in AI-generated recommendations."
        )

        industry_context = f" in the {industry} industry" if industry else ""
        user_prompt = (
            f"For the brand '{brand}'{industry_context}, identify the main product/service "
            f"categories they operate in, then generate exactly 5 consumer search queries for those categories.\n\n"
            f"IMPORTANT RULES:\n"
            f"- Do NOT include the brand name '{brand}' in any search query\n"
            f"- Queries should be generic category searches like 'best running shoes' NOT 'best Nike shoes'\n"
            f"- Focus on discovery/recommendation queries consumers naturally ask\n"
            f"- Include different intents: best overall, specific use cases, comparisons\n"
            f"- At least ONE query must be a 'how to' or informational question (e.g., 'how to choose running shoes', "
            f"'what to look for in a laptop') - these trigger Google AI Overviews\n"
            f"- If the brand spans multiple categories, include queries for each\n\n"
            f"Examples of GOOD queries: 'best running shoes', 'how to choose running shoes for beginners', "
            f"'most comfortable athletic shoes', 'what to look for in wireless headphones'\n"
            f"Examples of BAD queries: 'best Nike shoes', 'Nike vs Adidas', 'is Nike good'\n\n"
            f"Return as JSON array of exactly 5 strings, no explanation."
        )

        try:
            response = await self.chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
            )
            return self._parse_json_array(response.text)
        except Exception as e:
            print(f"OpenAI prompt generation failed: {e}")
            return self._get_fallback_prompts(brand)

    async def generate_competitors(self, brand: str, industry: Optional[str] = None) -> List[str]:
        """Generate list of competitors for a brand.

        Args:
            brand: The brand name.
            industry: Optional industry context.

        Returns:
            List of competitor brand names.
        """
        system_prompt = (
            "You are a market research assistant. "
            "Generate consumer search queries and competitor analysis."
        )

        industry_context = f" in the {industry} industry" if industry else ""
        user_prompt = (
            f"List 6-8 main competitors to {brand}{industry_context}. "
            f"Return as JSON array of strings only, no explanation."
        )

        try:
            response = await self.chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
            )
            return self._parse_json_array(response.text)
        except Exception as e:
            print(f"OpenAI competitor generation failed: {e}")
            return self._get_fallback_competitors(brand)

    def _parse_json_array(self, text: str) -> List[str]:
        """Parse a JSON array from text, handling markdown code blocks.

        Args:
            text: Text that may contain a JSON array.

        Returns:
            Parsed list of strings.
        """
        # Remove markdown code blocks if present
        text = re.sub(r"```json?\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(item) for item in result]
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract array portion
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return [str(item) for item in result]
            except json.JSONDecodeError:
                pass

        return []

    def _get_fallback_prompts(self, brand: str) -> List[str]:
        """Get fallback prompts when API fails."""
        # Category-based fallbacks for known brands (5 each, with at least 1 informational query)
        known_categories = {
            "nike": [
                "best running shoes",
                "how to choose running shoes for beginners",
                "most comfortable athletic shoes",
                "best basketball shoes",
                "top workout clothes",
            ],
            "adidas": [
                "best running shoes",
                "how to choose soccer cleats",
                "most comfortable athletic shoes",
                "best workout clothes",
                "top athletic sneakers",
            ],
            "apple": [
                "best smartphones",
                "how to choose a laptop for students",
                "best wireless earbuds",
                "top smartwatches",
                "best tablets for work",
            ],
            "samsung": [
                "best smartphones",
                "how to choose a smart TV",
                "best android phones",
                "top wireless earbuds",
                "best tablets",
            ],
            "coca-cola": [
                "best soft drinks",
                "what are the healthiest sodas",
                "top soda brands",
                "best refreshing beverages",
                "top cola drinks",
            ],
            "toyota": [
                "best reliable cars",
                "how to choose a family SUV",
                "top fuel efficient vehicles",
                "most reliable sedans",
                "top hybrid cars",
            ],
            "mcdonald's": [
                "best fast food restaurants",
                "how to eat healthy at fast food",
                "top burger chains",
                "best quick service restaurants",
                "top fast food for families",
            ],
        }

        # Return known categories or generic fallbacks
        return known_categories.get(brand.lower(), [
            "best products in category",
            "how to choose the right product",
            "top recommended options",
            "best value options",
            "most popular choices",
        ])

    def _get_fallback_competitors(self, brand: str) -> List[str]:
        """Get fallback competitors based on known brands."""
        known_competitors = {
            "nike": ["Adidas", "ASICS", "Hoka", "New Balance", "Brooks", "Saucony"],
            "adidas": ["Nike", "Puma", "ASICS", "New Balance", "Reebok", "Under Armour"],
            "apple": ["Samsung", "Google", "Microsoft", "Huawei", "OnePlus", "Sony"],
            "samsung": ["Apple", "Google", "Huawei", "OnePlus", "Xiaomi", "Sony"],
            "coca-cola": ["Pepsi", "Dr Pepper", "Sprite", "Fanta", "7-Up", "Mountain Dew"],
            "pepsi": ["Coca-Cola", "Dr Pepper", "Sprite", "7-Up", "Mountain Dew", "Fanta"],
            "toyota": ["Honda", "Ford", "Chevrolet", "Nissan", "Hyundai", "Mazda"],
            "mcdonald's": ["Burger King", "Wendy's", "Taco Bell", "KFC", "Subway", "Chick-fil-A"],
        }
        return known_competitors.get(brand.lower(), ["Competitor 1", "Competitor 2", "Competitor 3"])