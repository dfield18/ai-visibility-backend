"""Run executor service for parallel API call execution."""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import engine
from app.models.result import Result
from app.models.run import Run
from app.services.anthropic_service import AnthropicService
from app.services.gemini_service import GeminiService
from app.services.grok_service import GrokService
from app.services.llama_service import LlamaService
from app.services.openai_service import OpenAIService
from app.services.perplexity_service import PerplexityService
from app.services.serpapi_service import SerpAPIService
from app.services.result_processor import (
    check_brand_mentioned,
    classify_response_type,
    extract_competitors_mentioned,
)


class RunExecutor:
    """Executes visibility runs with parallel API calls.

    Manages concurrent API calls with semaphore-based rate limiting,
    handles errors gracefully, and updates the database in real-time.
    """

    MAX_CONCURRENT = 25  # Maximum parallel API calls

    def __init__(self):
        """Initialize the executor with API services."""
        self.openai_service: Optional[OpenAIService] = None
        self.gemini_service: Optional[GeminiService] = None
        self.anthropic_service: Optional[AnthropicService] = None
        self.perplexity_service: Optional[PerplexityService] = None
        self.grok_service: Optional[GrokService] = None
        self.llama_service: Optional[LlamaService] = None
        self.serpapi_service: Optional[SerpAPIService] = None
        self.semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)

        # Initialize services if API keys are available
        try:
            self.openai_service = OpenAIService()
        except ValueError as e:
            print(f"OpenAI service not available: {e}")

        try:
            self.gemini_service = GeminiService()
        except ValueError as e:
            print(f"Gemini service not available: {e}")

        try:
            self.anthropic_service = AnthropicService()
        except ValueError as e:
            print(f"Anthropic service not available: {e}")

        try:
            self.perplexity_service = PerplexityService()
        except ValueError as e:
            print(f"Perplexity service not available: {e}")

        try:
            self.grok_service = GrokService()
        except ValueError as e:
            print(f"Grok service not available: {e}")

        try:
            self.llama_service = LlamaService()
        except ValueError as e:
            print(f"Llama service not available: {e}")

        try:
            self.serpapi_service = SerpAPIService()
        except ValueError as e:
            print(f"SerpAPI service not available: {e}")

    def _get_session_factory(self) -> async_sessionmaker:
        """Create a new session factory."""
        return async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def execute_run(self, run_id: UUID, config: Dict[str, Any]) -> None:
        """Execute a visibility run in the background.

        Args:
            run_id: The UUID of the run to execute.
            config: Run configuration containing prompts, providers, etc.
        """
        print(f"[Executor] Starting run {run_id}")

        session_factory = self._get_session_factory()

        # Update run status to running
        async with session_factory() as session:
            run = await session.get(Run, run_id)
            if not run:
                print(f"[Executor] Run {run_id} not found")
                return

            run.status = Run.STATUS_RUNNING
            await session.commit()

        # Extract config
        brand = config.get("brand", "")
        prompts = config.get("prompts", [])
        competitors = config.get("competitors", [])
        providers = config.get("providers", [])
        temperatures = config.get("temperatures", [0.7])
        repeats = config.get("repeats", 1)
        openai_model = config.get("openai_model", "gpt-4o-mini")
        anthropic_model = config.get("anthropic_model", "claude-haiku-4-5-20251001")
        grok_model = config.get("grok_model", "grok-3")
        llama_model = config.get("llama_model", "llama-3.3-70b-versatile")
        country = config.get("country", "us")
        search_type = config.get("search_type", "brand")

        # Build task list
        tasks = []
        for prompt in prompts:
            for provider in providers:
                for temp in temperatures:
                    for repeat_idx in range(repeats):
                        tasks.append({
                            "prompt": prompt,
                            "provider": provider,
                            "temperature": temp,
                            "repeat_index": repeat_idx,
                        })

        print(f"[Executor] Run {run_id}: {len(tasks)} tasks to execute")

        # Execute tasks with concurrency control
        # Each task gets its own session to avoid conflicts
        async def execute_task(task: Dict[str, Any]) -> Tuple[bool, float]:
            """Execute a single task and return (success, cost)."""
            async with self.semaphore:
                return await self._execute_single_task(
                    session_factory=session_factory,
                    run_id=run_id,
                    brand=brand,
                    competitors=competitors,
                    task=task,
                    openai_model=openai_model,
                    anthropic_model=anthropic_model,
                    grok_model=grok_model,
                    llama_model=llama_model,
                    country=country,
                    search_type=search_type,
                )

        # Run all tasks concurrently
        results = await asyncio.gather(
            *[execute_task(task) for task in tasks],
            return_exceptions=True,
        )

        # Count exceptions (tasks that failed outside normal error handling)
        exception_count = sum(1 for r in results if isinstance(r, Exception))
        for result in results:
            if isinstance(result, Exception):
                print(f"[Executor] Task exception: {result}")

        # Final update to run status (counters already updated incrementally)
        async with session_factory() as session:
            run = await session.get(Run, run_id)
            if run:
                # Check if cancelled
                if run.cancelled:
                    run.status = Run.STATUS_CANCELLED
                else:
                    run.status = Run.STATUS_COMPLETE

                # Add any exception failures that weren't counted
                run.failed_calls = (run.failed_calls or 0) + exception_count
                run.completed_at = datetime.utcnow()
                await session.commit()

                print(f"[Executor] Run {run_id} completed: {run.completed_calls} successful, {run.failed_calls} failed")

    async def _execute_single_task(
        self,
        session_factory: async_sessionmaker,
        run_id: UUID,
        brand: str,
        competitors: List[str],
        task: Dict[str, Any],
        openai_model: str = "gpt-4o-mini",
        anthropic_model: str = "claude-haiku-4-5-20251001",
        grok_model: str = "grok-3",
        llama_model: str = "llama-3.3-70b-versatile",
        country: str = "us",
        search_type: str = "brand",
    ) -> Tuple[bool, float]:
        """Execute a single API call task.

        Args:
            session_factory: Factory for creating database sessions.
            run_id: The run ID.
            brand: Brand to check for.
            competitors: List of competitors to check for.
            task: Task configuration dict.
            openai_model: OpenAI model to use.
            anthropic_model: Anthropic model to use.
            country: Country code for filtering search results.
            search_type: The search type (brand, category, etc.).

        Returns:
            Tuple of (success: bool, cost: float)
        """
        prompt = task["prompt"]
        provider = task["provider"]
        temperature = task["temperature"]
        repeat_index = task["repeat_index"]

        # Check if run was cancelled
        async with session_factory() as session:
            run = await session.get(Run, run_id)
            if run and run.cancelled:
                print(f"[Executor] Run {run_id} cancelled, skipping task")
                return (False, 0.0)

        result = Result(
            run_id=run_id,
            prompt=prompt,
            provider=provider,
            model="",
            temperature=temperature,
            repeat_index=repeat_index,
        )

        success = False
        cost = 0.0

        try:
            if provider == "openai" and self.openai_service:
                # Try Responses API with web search first, fall back to chat completion
                try:
                    response = await self.openai_service.search_completion(
                        prompt=prompt,
                        temperature=temperature,
                        model=openai_model,
                    )
                    result.sources = response.sources
                except Exception as e:
                    print(f"[Executor] OpenAI Responses API failed, falling back to chat completion: {e}")
                    response = await self.openai_service.chat_completion(
                        user_prompt=prompt,
                        temperature=temperature,
                    )
                    result.sources = []
                result.response_text = response.text
                result.model = response.model
                result.tokens = response.tokens_input + response.tokens_output
                result.cost = response.cost
                cost = response.cost

            elif provider == "gemini" and self.gemini_service:
                response = await self.gemini_service.generate_content(
                    prompt=prompt,
                    temperature=temperature,
                )
                result.response_text = response.text
                result.model = response.model
                result.tokens = response.tokens_input + response.tokens_output
                result.cost = response.cost
                result.sources = response.sources  # Sources from Google Search grounding
                result.grounding_metadata = response.grounding_metadata  # Grounding supports with confidence
                cost = response.cost

            elif provider == "anthropic" and self.anthropic_service:
                response = await self.anthropic_service.generate_content(
                    prompt=prompt,
                    temperature=temperature,
                    model=anthropic_model,
                )
                result.response_text = response.text
                result.model = response.model
                result.tokens = response.tokens_input + response.tokens_output
                result.cost = response.cost
                result.sources = response.sources  # Sources from web search citations
                cost = response.cost

            elif provider == "perplexity" and self.perplexity_service:
                response = await self.perplexity_service.generate_content(
                    prompt=prompt,
                    temperature=temperature,
                    country=country,
                )
                result.response_text = response.text
                result.model = response.model
                result.tokens = response.tokens_input + response.tokens_output
                result.cost = response.cost
                result.sources = response.sources
                cost = response.cost

            elif provider == "ai_overviews" and self.serpapi_service:
                response = await self.serpapi_service.generate_content(
                    prompt=prompt,
                    temperature=temperature,
                    country=country,
                )
                result.response_text = response.text
                result.model = response.model
                result.tokens = response.tokens_input + response.tokens_output
                result.cost = response.cost
                result.sources = response.sources
                cost = response.cost

            elif provider == "grok":
                print(f"[Executor] Grok provider requested, service available: {self.grok_service is not None}")
                if not self.grok_service:
                    raise ValueError("Grok service not initialized - check GROK_API_KEY")
                try:
                    response = await self.grok_service.generate_content(
                        prompt=prompt,
                        temperature=temperature,
                        model=grok_model,
                    )
                    result.response_text = response.text
                    result.model = response.model
                    result.tokens = response.tokens_input + response.tokens_output
                    result.cost = response.cost
                    result.sources = response.sources
                    cost = response.cost
                    print(f"[Executor] Grok call successful, response length: {len(response.text)}")
                except Exception as e:
                    print(f"[Executor] Grok call FAILED with error: {type(e).__name__}: {e}")
                    raise

            elif provider == "llama":
                logger.info(f"[Executor] Llama provider requested, service available: {self.llama_service is not None}")
                if not self.llama_service:
                    raise ValueError("Llama service not initialized - check LLAMA_API_KEY")
                try:
                    logger.info(f"[Executor] Calling Llama/Groq API for prompt: {prompt[:50]}...")
                    response = await self.llama_service.generate_content(
                        prompt=prompt,
                        temperature=temperature,
                        model=llama_model,
                    )
                    result.response_text = response.text
                    result.model = response.model
                    result.tokens = response.tokens_input + response.tokens_output
                    result.cost = response.cost
                    result.sources = response.sources
                    cost = response.cost
                    logger.info(f"[Executor] Llama/Groq call successful, model: {response.model}, response length: {len(response.text)}")
                except Exception as e:
                    logger.error(f"[Executor] Llama/Groq call FAILED with error: {type(e).__name__}: {e}")
                    raise

            else:
                raise ValueError(f"Provider {provider} not available")

            # Analyze response
            result.brand_mentioned = check_brand_mentioned(result.response_text, brand)
            result.competitors_mentioned = extract_competitors_mentioned(
                result.response_text, competitors
            )
            result.response_type = classify_response_type(result.response_text)

            # Classify sentiment and extract all brands using OpenAI
            if self.openai_service and result.response_text:
                try:
                    sentiment_result = await self.openai_service.classify_brand_sentiment(
                        response_text=result.response_text,
                        brand=brand,
                        competitors=competitors,
                        sources=result.sources,
                    )
                    result.brand_sentiment = sentiment_result.get("brand_sentiment", "not_mentioned")
                    result.competitor_sentiments = sentiment_result.get("competitor_sentiments", {})
                    result.source_brand_sentiments = sentiment_result.get("source_brand_sentiments")

                    # For category (industry) reports, override brand_sentiment with
                    # the average sentiment across mentioned brands instead of the
                    # sentiment toward the category name itself.
                    if search_type == "category" and result.competitor_sentiments:
                        _SENT_SCORES = {
                            "strong_endorsement": 5,
                            "positive_endorsement": 4,
                            "neutral_mention": 3,
                            "conditional": 2,
                            "negative_comparison": 1,
                        }
                        _SCORE_TO_SENT = {v: k for k, v in _SENT_SCORES.items()}
                        scored = [
                            _SENT_SCORES[s]
                            for s in result.competitor_sentiments.values()
                            if s in _SENT_SCORES
                        ]
                        if scored:
                            avg = sum(scored) / len(scored)
                            # Round to nearest sentiment level
                            nearest = min(_SCORE_TO_SENT.keys(), key=lambda x: abs(x - avg))
                            result.brand_sentiment = _SCORE_TO_SENT[nearest]

                except Exception as e:
                    print(f"[Executor] Sentiment classification failed: {e}")
                    # Fallback to simple detection
                    result.brand_sentiment = "neutral_mention" if result.brand_mentioned else "not_mentioned"
                    result.competitor_sentiments = {
                        c: "neutral_mention" if c in (result.competitors_mentioned or []) else "not_mentioned"
                        for c in competitors
                    }
                    result.source_brand_sentiments = None

                # Extract all brand mentions for position calculation
                try:
                    all_brands = await self.openai_service.extract_all_brands(
                        response_text=result.response_text,
                        primary_brand=brand,
                    )
                    result.all_brands_mentioned = all_brands
                except Exception as e:
                    print(f"[Executor] Brand extraction failed: {e}")
                    # Fallback to tracked brands only
                    result.all_brands_mentioned = [brand] + (result.competitors_mentioned or []) if result.brand_mentioned else (result.competitors_mentioned or [])
            else:
                # No OpenAI service available, use simple fallback
                result.brand_sentiment = "neutral_mention" if result.brand_mentioned else "not_mentioned"
                result.competitor_sentiments = {
                    c: "neutral_mention" if c in (result.competitors_mentioned or []) else "not_mentioned"
                    for c in competitors
                }
                result.source_brand_sentiments = None
                result.all_brands_mentioned = [brand] + (result.competitors_mentioned or []) if result.brand_mentioned else (result.competitors_mentioned or [])

            success = True

            print(
                f"[Executor] Task completed: {provider} | "
                f"brand_mentioned={result.brand_mentioned} | "
                f"brand_sentiment={result.brand_sentiment} | "
                f"cost=${cost:.4f}"
            )

        except Exception as e:
            result.error = str(e)
            print(f"[Executor] Task failed: {provider} | error={e}")

        # Save result and update run progress in its own session
        async with session_factory() as session:
            session.add(result)

            # Update run progress atomically to avoid race conditions
            if success:
                await session.execute(
                    update(Run)
                    .where(Run.id == run_id)
                    .values(
                        completed_calls=Run.completed_calls + 1,
                        actual_cost=Run.actual_cost + Decimal(str(cost)),
                    )
                )
            else:
                await session.execute(
                    update(Run)
                    .where(Run.id == run_id)
                    .values(
                        failed_calls=Run.failed_calls + 1,
                        actual_cost=Run.actual_cost + Decimal(str(cost)),
                    )
                )

            await session.commit()

        return (success, cost)
