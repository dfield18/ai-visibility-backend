"""Run executor service for parallel API call execution."""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import engine
from app.models.result import Result
from app.models.run import Run
from app.services.anthropic_service import AnthropicService
from app.services.gemini_service import GeminiService
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

    MAX_CONCURRENT = 10  # Maximum parallel API calls

    def __init__(self):
        """Initialize the executor with API services."""
        self.openai_service: Optional[OpenAIService] = None
        self.gemini_service: Optional[GeminiService] = None
        self.anthropic_service: Optional[AnthropicService] = None
        self.perplexity_service: Optional[PerplexityService] = None
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
    ) -> Tuple[bool, float]:
        """Execute a single API call task.

        Args:
            session_factory: Factory for creating database sessions.
            run_id: The run ID.
            brand: Brand to check for.
            competitors: List of competitors to check for.
            task: Task configuration dict.

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
                result.sources = []  # Gemini doesn't provide sources
                cost = response.cost

            elif provider == "anthropic" and self.anthropic_service:
                response = await self.anthropic_service.generate_content(
                    prompt=prompt,
                    temperature=temperature,
                )
                result.response_text = response.text
                result.model = response.model
                result.tokens = response.tokens_input + response.tokens_output
                result.cost = response.cost
                result.sources = []  # Anthropic doesn't provide sources
                cost = response.cost

            elif provider == "perplexity" and self.perplexity_service:
                response = await self.perplexity_service.generate_content(
                    prompt=prompt,
                    temperature=temperature,
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
                )
                result.response_text = response.text
                result.model = response.model
                result.tokens = response.tokens_input + response.tokens_output
                result.cost = response.cost
                result.sources = response.sources
                cost = response.cost

            else:
                raise ValueError(f"Provider {provider} not available")

            # Analyze response
            result.brand_mentioned = check_brand_mentioned(result.response_text, brand)
            result.competitors_mentioned = extract_competitors_mentioned(
                result.response_text, competitors
            )
            result.response_type = classify_response_type(result.response_text)
            success = True

            print(
                f"[Executor] Task completed: {provider} | "
                f"brand_mentioned={result.brand_mentioned} | "
                f"cost=${cost:.4f}"
            )

        except Exception as e:
            result.error = str(e)
            print(f"[Executor] Task failed: {provider} | error={e}")

        # Save result and update run progress in its own session
        async with session_factory() as session:
            session.add(result)

            # Update run progress incrementally
            run = await session.get(Run, run_id)
            if run:
                if success:
                    run.completed_calls = (run.completed_calls or 0) + 1
                else:
                    run.failed_calls = (run.failed_calls or 0) + 1
                run.actual_cost = (run.actual_cost or Decimal("0")) + Decimal(str(cost))

            await session.commit()

        return (success, cost)
