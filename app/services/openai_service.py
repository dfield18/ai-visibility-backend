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

    # Pricing per 1K tokens
    MODEL_PRICING = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }

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
        # Calculate cost based on model
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["gpt-4o"])
        self.cost = (tokens_input * pricing["input"] / 1000) + (tokens_output * pricing["output"] / 1000)


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
        model: str = "gpt-4o-mini",
    ) -> OpenAIResponse:
        """Make a request using Responses API with web search for citations.

        Args:
            prompt: The search query/prompt to send.
            temperature: Sampling temperature (0.0-2.0).
            model: OpenAI model to use (gpt-4o-mini or gpt-4o).

        Returns:
            OpenAIResponse with generated text, usage stats, and sources.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        # Prepend instruction to search the web for current information
        search_prompt = f"Search the web for the most current and up-to-date information in 2024-2025 to answer this question. Include sources and citations: {prompt}"

        payload = {
            "model": model,
            "input": search_prompt,
            "tools": [
                {
                    "type": "web_search_preview",
                    "search_context_size": "high",  # Request more search context
                }
            ],
            "tool_choice": {"type": "web_search_preview"},  # Force specific tool
            "temperature": temperature,
        }
        print(f"[OpenAI] Request payload tools: {payload['tools']}")
        print(f"[OpenAI] Request payload tool_choice: {payload['tool_choice']}")

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
        print(f"[OpenAI] Tools in response: {data.get('tools')}")
        print(f"[OpenAI] Tool choice in response: {data.get('tool_choice')}")

        # Extract text from output
        text = ""
        sources = []

        # Parse the output array for text and citations
        output = data.get("output", [])
        print(f"[OpenAI] Output items: {len(output)}")
        print(f"[OpenAI] Full output structure: {output}")

        for idx, item in enumerate(output):
            item_type = item.get("type")
            print(f"[OpenAI] Output[{idx}] type: {item_type}, keys: {item.keys()}")

            if item_type == "message":
                content = item.get("content", [])
                for content_item in content:
                    content_type = content_item.get("type")
                    print(f"[OpenAI] Content item type: {content_type}, keys: {content_item.keys()}")

                    if content_type == "output_text":
                        text = content_item.get("text", "")
                        # Extract annotations (citations)
                        annotations = content_item.get("annotations", [])
                        print(f"[OpenAI] Found {len(annotations)} annotations")
                        if annotations:
                            print(f"[OpenAI] First annotation: {annotations[0]}")
                        for annotation in annotations:
                            ann_type = annotation.get("type")
                            if ann_type == "url_citation":
                                sources.append({
                                    "url": annotation.get("url", ""),
                                    "title": annotation.get("title", ""),
                                })

            # Check for web_search_call which may contain sources
            elif item_type == "web_search_call":
                print(f"[OpenAI] Web search call: status={item.get('status')}")

        # Check for 'sources' field at various levels (returns ALL URLs consulted)
        # This is different from inline citations - it's ALL sources the model looked at
        if "sources" in data:
            print(f"[OpenAI] Found top-level sources: {data['sources']}")
            for source in data.get("sources", []):
                if isinstance(source, dict):
                    sources.append({
                        "url": source.get("url", ""),
                        "title": source.get("title", ""),
                    })
                elif isinstance(source, str):
                    sources.append({"url": source, "title": ""})

        # Check for 'citations' at top level
        if "citations" in data:
            print(f"[OpenAI] Found top-level citations: {data['citations']}")
            for citation in data.get("citations", []):
                sources.append({
                    "url": citation.get("url", ""),
                    "title": citation.get("title", citation.get("name", "")),
                })

        # Log text sample for debugging
        if text:
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
            model=model,
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

    async def generate_category_prompts(self, category: str, industry: Optional[str] = None) -> List[str]:
        """Generate search prompts for a product category.

        Args:
            category: The product category (e.g., 'cars', 'shoes', 'laptops').
            industry: Optional industry context.

        Returns:
            List of suggested search queries for the category.
        """
        system_prompt = (
            "You are a market research assistant helping understand "
            "brand visibility in AI-generated recommendations for product categories."
        )

        industry_context = f" in the {industry} industry" if industry else ""
        user_prompt = (
            f"For the product category '{category}'{industry_context}, generate exactly 5 consumer search queries.\n\n"
            f"IMPORTANT RULES:\n"
            f"- Queries should be generic category searches consumers naturally ask\n"
            f"- Focus on discovery/recommendation queries\n"
            f"- Include different intents: best overall, specific use cases, comparisons, buying guides\n"
            f"- At least ONE query must be a 'how to' or informational question (e.g., 'how to choose a laptop', "
            f"'what to look for when buying a car') - these trigger Google AI Overviews\n\n"
            f"Examples for 'laptops': 'best laptops', 'how to choose a laptop for students', "
            f"'best budget laptops', 'laptops for video editing', 'what specs matter in a laptop'\n\n"
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
            print(f"OpenAI category prompt generation failed: {e}")
            return self._get_fallback_category_prompts(category)

    async def generate_category_brands(self, category: str, industry: Optional[str] = None) -> List[str]:
        """Generate list of brands in a product category.

        Args:
            category: The product category (e.g., 'cars', 'shoes', 'laptops').
            industry: Optional industry context.

        Returns:
            List of brand names in the category.
        """
        system_prompt = (
            "You are a market research assistant. "
            "Identify leading brands in product categories."
        )

        industry_context = f" in the {industry} industry" if industry else ""
        user_prompt = (
            f"List 8-10 leading brands in the '{category}' category{industry_context}. "
            f"Include a mix of premium, mid-range, and value brands. "
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
            print(f"OpenAI category brands generation failed: {e}")
            return self._get_fallback_category_brands(category)

    def _get_fallback_category_prompts(self, category: str) -> List[str]:
        """Get fallback prompts for categories when API fails."""
        known_categories = {
            "cars": [
                "best cars",
                "how to choose a car",
                "most reliable cars",
                "best family cars",
                "fuel efficient cars",
            ],
            "shoes": [
                "best shoes",
                "how to choose running shoes",
                "most comfortable shoes",
                "best walking shoes",
                "shoes for standing all day",
            ],
            "laptops": [
                "best laptops",
                "how to choose a laptop",
                "best laptops for students",
                "laptops for video editing",
                "best budget laptops",
            ],
            "smartphones": [
                "best smartphones",
                "how to choose a phone",
                "best phones for photography",
                "best budget phones",
                "phones with best battery life",
            ],
            "restaurants": [
                "best restaurants",
                "how to find good restaurants",
                "best restaurants near me",
                "top rated restaurants",
                "restaurants for special occasions",
            ],
        }
        return known_categories.get(category.lower(), [
            f"best {category}",
            f"how to choose {category}",
            f"top rated {category}",
            f"best {category} reviews",
            f"{category} buying guide",
        ])

    def _get_fallback_category_brands(self, category: str) -> List[str]:
        """Get fallback brands for categories when API fails."""
        known_brands = {
            "cars": ["Toyota", "Honda", "Ford", "BMW", "Mercedes-Benz", "Tesla", "Chevrolet", "Hyundai"],
            "shoes": ["Nike", "Adidas", "New Balance", "ASICS", "Brooks", "Hoka", "Saucony", "Puma"],
            "laptops": ["Apple", "Dell", "HP", "Lenovo", "ASUS", "Microsoft", "Acer", "Samsung"],
            "smartphones": ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "Sony", "Motorola", "Nothing"],
            "restaurants": ["McDonald's", "Starbucks", "Chipotle", "Chick-fil-A", "Subway", "Panera", "Wendy's", "Taco Bell"],
        }
        return known_brands.get(category.lower(), [f"Brand 1", f"Brand 2", f"Brand 3", f"Brand 4", f"Brand 5"])

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

    async def classify_brand_sentiment(
        self,
        response_text: str,
        brand: str,
        competitors: List[str],
    ) -> Dict[str, Any]:
        """Classify how an AI response describes a brand and its competitors.

        Uses GPT-4o-mini to analyze the sentiment/framing of brand mentions.

        Args:
            response_text: The AI-generated response text to analyze.
            brand: The primary brand to classify.
            competitors: List of competitor brands to also classify.

        Returns:
            Dict with 'brand_sentiment' and 'competitor_sentiments' keys.
            brand_sentiment is one of: strong_endorsement, neutral_mention, conditional, negative_comparison, not_mentioned
            competitor_sentiments is a dict mapping competitor name to sentiment.
        """
        if not response_text or not response_text.strip():
            return {
                "brand_sentiment": "not_mentioned",
                "competitor_sentiments": {c: "not_mentioned" for c in competitors},
            }

        # Build the list of all brands to analyze
        all_brands = [brand] + competitors

        system_prompt = """You are a brand sentiment classifier. Analyze how AI responses describe brands.

For each brand mentioned (or not mentioned) in the text, classify the sentiment as one of:
- strong_endorsement: Brand is clearly recommended, praised, or positioned as a top choice
- neutral_mention: Brand is mentioned factually without strong positive or negative framing
- conditional: Brand is mentioned with caveats, limitations, or "it depends" framing
- negative_comparison: Brand is mentioned unfavorably or positioned worse than competitors
- not_mentioned: Brand does not appear in the response

Return ONLY a valid JSON object with no markdown formatting."""

        brands_list = ", ".join(f'"{b}"' for b in all_brands)
        user_prompt = f"""Analyze this AI response and classify how it describes each brand.

RESPONSE TEXT:
{response_text[:4000]}

BRANDS TO CLASSIFY: {brands_list}

Return a JSON object where keys are brand names and values are sentiment classifications.
Example format:
{{"Nike": "strong_endorsement", "Adidas": "neutral_mention", "Puma": "not_mentioned"}}

Classify ALL brands listed, including those not mentioned (use "not_mentioned" for those)."""

        try:
            response = await self.chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,  # Low temperature for consistent classification
            )

            # Parse the response
            result_text = response.text.strip()
            # Remove markdown code blocks if present
            result_text = re.sub(r"```json?\s*", "", result_text)
            result_text = re.sub(r"```\s*", "", result_text)
            result_text = result_text.strip()

            sentiments = json.loads(result_text)

            # Extract brand sentiment (case-insensitive lookup)
            brand_sentiment = "not_mentioned"
            for key, value in sentiments.items():
                if key.lower() == brand.lower():
                    brand_sentiment = value
                    break

            # Extract competitor sentiments
            competitor_sentiments = {}
            for comp in competitors:
                comp_sentiment = "not_mentioned"
                for key, value in sentiments.items():
                    if key.lower() == comp.lower():
                        comp_sentiment = value
                        break
                competitor_sentiments[comp] = comp_sentiment

            # Validate sentiment values
            valid_sentiments = {
                "strong_endorsement",
                "neutral_mention",
                "conditional",
                "negative_comparison",
                "not_mentioned",
            }

            if brand_sentiment not in valid_sentiments:
                brand_sentiment = "neutral_mention" if brand.lower() in response_text.lower() else "not_mentioned"

            for comp, sent in competitor_sentiments.items():
                if sent not in valid_sentiments:
                    competitor_sentiments[comp] = "neutral_mention" if comp.lower() in response_text.lower() else "not_mentioned"

            return {
                "brand_sentiment": brand_sentiment,
                "competitor_sentiments": competitor_sentiments,
            }

        except Exception as e:
            print(f"[OpenAI] Sentiment classification failed: {e}")
            # Fallback: simple mention detection
            brand_sentiment = "neutral_mention" if brand.lower() in response_text.lower() else "not_mentioned"
            competitor_sentiments = {}
            for comp in competitors:
                competitor_sentiments[comp] = "neutral_mention" if comp.lower() in response_text.lower() else "not_mentioned"

            return {
                "brand_sentiment": brand_sentiment,
                "competitor_sentiments": competitor_sentiments,
            }

    async def extract_all_brands(
        self,
        response_text: str,
        primary_brand: str,
    ) -> List[str]:
        """Extract all brand/company names mentioned in a response.

        Uses GPT-4o-mini to identify all brand and company names in the text,
        not just the tracked competitors.

        Args:
            response_text: The AI-generated response text to analyze.
            primary_brand: The primary brand being tracked (to ensure it's included).

        Returns:
            List of brand/company names found in the response, in order of appearance.
        """
        if not response_text or not response_text.strip():
            return []

        system_prompt = """You are a brand/company name extractor. Your job is to identify all brand names, company names, product names, and service names mentioned in the text.

Rules:
- Include all commercial brands, companies, products, and services
- Preserve the exact capitalization/spelling used in the text
- Return names in the order they appear in the text
- Do NOT include generic terms (e.g., "smartphone", "search engine")
- Do NOT include people's names unless they are brand names
- Return ONLY a JSON array of strings, no other text"""

        user_prompt = f"""Extract all brand/company/product names from this text:

{response_text[:4000]}

Return a JSON array of brand names in order of appearance. Example: ["Apple", "Samsung", "Google Pixel"]"""

        try:
            response = await self.chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
            )

            result_text = response.text.strip()
            # Remove markdown code blocks if present
            result_text = re.sub(r"```json?\s*", "", result_text)
            result_text = re.sub(r"```\s*", "", result_text)
            result_text = result_text.strip()

            brands = json.loads(result_text)

            if not isinstance(brands, list):
                return []

            # Ensure primary brand is included if it appears in text
            if primary_brand and primary_brand.lower() in response_text.lower():
                primary_found = any(b.lower() == primary_brand.lower() for b in brands)
                if not primary_found:
                    # Find position of primary brand to insert in correct order
                    primary_pos = response_text.lower().find(primary_brand.lower())
                    inserted = False
                    for i, b in enumerate(brands):
                        b_pos = response_text.lower().find(b.lower())
                        if b_pos > primary_pos:
                            brands.insert(i, primary_brand)
                            inserted = True
                            break
                    if not inserted:
                        brands.append(primary_brand)

            return brands

        except Exception as e:
            print(f"[OpenAI] Brand extraction failed: {e}")
            # Fallback: just return primary brand if mentioned
            if primary_brand and primary_brand.lower() in response_text.lower():
                return [primary_brand]
            return []

    async def generate_results_summary(
        self,
        brand: str,
        search_type: str,
        results_data: str,
    ) -> str:
        """Generate an AI summary of visibility run results.

        Args:
            brand: The brand or category being analyzed.
            search_type: Either 'brand' or 'category'.
            results_data: Formatted string of all results data.

        Returns:
            An executive summary of the results.
        """
        entity_type = "category" if search_type == "category" else "brand"

        system_prompt = (
            "You are an AI visibility analyst. Your job is to analyze how brands appear "
            "in AI-generated responses across different AI providers. "
            "You produce executive-ready summaries that are clear, accurate, and actionable. "
            "You prioritize correct interpretation of quantitative data, scannability, "
            "and business-relevant insights over cautious or academic language."
        )

        user_prompt = f"""
Analyze the following AI visibility data for the {entity_type} "{brand}" and produce
an executive summary suitable for a product dashboard or brand intelligence report.

The data shows how multiple AI providers (e.g., OpenAI ChatGPT, Google Gemini,
Anthropic Claude, Perplexity, Google AI Overviews) responded to relevant prompts.
Each result includes whether {brand} was mentioned, which competitors appeared,
and any cited sources.

DATA:
{results_data}

OUTPUT FORMAT:

1. Begin with a **single bolded headline sentence** that clearly states the dominant
   conclusion about {brand}'s AI visibility (e.g., strong, moderate, weak, or provider-specific).

2. Follow with **4–5 short paragraph blocks**, separated by line breaks.
   Each paragraph should begin with a **bolded lead-in phrase** (not bullet points)
   that guides the reader.

CONTENT TO COVER:
- **Overall visibility**: How frequently {brand} is mentioned or recommended across AI providers
- **Provider differences**: Any meaningful over- or under-representation by specific providers
- **Competitive context**: Which competitors appear most often and how {brand} compares
- **Source patterns**: Commonly cited source types or notable gaps in sourcing
- **Actionable takeaway**: One concrete, practical recommendation to improve AI visibility

INTERPRETATION GUIDANCE (CRITICAL):
- Determine visibility strength based on the proportion of providers mentioning {brand},
  relative to the total number of providers analyzed (which may range from 1 to 6).
- Use the following framing guidelines:
  • Presence in a clear majority of responses (≈70% or more) = **strong AI visibility**
  • Presence in roughly half of responses = **moderate AI visibility**
  • Presence in a small minority of responses = **weak AI visibility**
- Do NOT describe visibility as "uneven", "mixed", or "inconsistent" if the brand appears
  in a strong majority of responses.
- Absence from a single provider should be framed as a **specific gap or opportunity**,
  not as overall inconsistency.
- Overall tone and headline should reflect the **dominant quantitative pattern**, not exceptions.

STYLE AND FORMATTING RULES:
- Do NOT use bullet points or numbered lists in the output
- Use line breaks between paragraphs for readability
- Avoid hedging or softening language (e.g., "may indicate", "appears to be", "somewhat")
- Avoid downplaying strong performance with qualifiers like "only" or "moderate" when data supports strength
- Be specific, comparative, and decisive
- Write for a brand, SEO, or growth lead—not an academic audience
- Do not restate the prompt or describe the methodology
"""

        try:
            response = await self.chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.5,  # Lower temperature for more consistent summaries
            )
            return response.text
        except Exception as e:
            print(f"[OpenAI] Summary generation failed: {e}")
            return ""