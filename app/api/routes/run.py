"""Run endpoints for starting, monitoring, and cancelling visibility runs."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import DatabaseDep
from app.models.result import Result
from app.models.run import Run
from app.models.session import Session
from app.schemas.run import (
    AISummaryResponse,
    CancelResponse,
    CompetitorStats,
    ExtendRunRequest,
    ExtendRunResponse,
    ProviderStats,
    ResultItem,
    RunRequest,
    RunResponse,
    RunStatusResponse,
    RunSummary,
)
from app.services.openai_service import OpenAIService
from app.services.executor import RunExecutor
from app.services.result_processor import estimate_duration_seconds, estimate_run_cost
from app.core.auth import OptionalUser

router = APIRouter()

# Global executor instance
executor = RunExecutor()


@router.post("/run", response_model=RunResponse)
async def start_run(
    request: RunRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseDep,
    user: OptionalUser = None,
) -> RunResponse:
    """Start a brand visibility analysis run.

    Creates a run record and starts background execution.
    Returns immediately with run_id for polling.

    Args:
        request: RunRequest with brand, prompts, providers, etc.
        background_tasks: FastAPI background tasks.
        db: Database session.
        user: Optional authenticated user for billing enforcement.

    Returns:
        RunResponse with run_id and estimated cost/duration.

    Raises:
        HTTPException: If validation fails or run cannot be created.
    """
    # Free tier enforcement: check subscription status
    FREE_PROVIDERS = {"openai", "gemini"}
    is_paid = False

    if user:
        try:
            import stripe
            from app.core.config import settings
            stripe.api_key = getattr(settings, 'STRIPE_API_KEY', '') or ''
            if stripe.api_key:
                customers = stripe.Customer.search(
                    query=f'metadata["clerk_user_id"]:"{user.user_id}"',
                )
                if customers.data:
                    subs = stripe.Subscription.list(
                        customer=customers.data[0].id,
                        status='active',
                        limit=1,
                    )
                    is_paid = bool(subs.data)
        except Exception:
            pass  # Default to free tier if Stripe unavailable

    if not is_paid:
        # Restrict to free providers
        non_free = [p for p in request.providers if p not in FREE_PROVIDERS]
        if non_free:
            raise HTTPException(
                status_code=403,
                detail=f"Free tier only supports {', '.join(FREE_PROVIDERS)} providers. Upgrade to Pro for access to {', '.join(non_free)}.",
            )

    # Validate prompt limit
    if len(request.prompts) > 20:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum 20 prompts per run. You submitted {len(request.prompts)}.",
        )

    # Validate total calls
    total_calls = request.total_calls
    if total_calls > 300:
        raise HTTPException(
            status_code=400,
            detail=f"Total calls ({total_calls}) exceeds maximum of 300",
        )

    # Get or create session
    result = await db.execute(
        select(Session).where(Session.session_id == request.session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        session = Session(
            session_id=request.session_id,
            expires_at=datetime.utcnow() + timedelta(days=7),
        )
        db.add(session)
        await db.flush()

    # Calculate estimates
    estimated_cost = estimate_run_cost(
        num_prompts=len(request.prompts),
        num_providers=len(request.providers),
        num_temperatures=len(request.temperatures),
        num_repeats=request.repeats,
        providers=request.providers,
    )
    estimated_duration = estimate_duration_seconds(total_calls)

    # Create run config
    config = {
        "brand": request.brand,
        "search_type": request.search_type,
        "prompts": request.prompts,
        "competitors": request.competitors,
        "providers": request.providers,
        "temperatures": request.temperatures,
        "repeats": request.repeats,
        "openai_model": request.openai_model,
        "anthropic_model": request.anthropic_model,
        "country": request.country,
    }

    # Create run record
    run = Run(
        session_id=session.id,
        status=Run.STATUS_QUEUED,
        brand=request.brand,
        config=config,
        total_calls=total_calls,
        estimated_cost=estimated_cost,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    # Start background execution
    background_tasks.add_task(executor.execute_run, run.id, config)

    return RunResponse(
        run_id=run.id,
        status=run.status,
        total_calls=total_calls,
        estimated_cost=estimated_cost,
        estimated_duration_seconds=estimated_duration,
    )


@router.get("/run/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: UUID, db: DatabaseDep) -> RunStatusResponse:
    """Get the status and results of a run.

    For runs with child runs (extensions), this aggregates results from all related runs.

    Args:
        run_id: The UUID of the run.
        db: Database session.

    Returns:
        RunStatusResponse with full status, summary, and results.

    Raises:
        HTTPException: If run not found.
    """
    # Fetch run with results and child runs
    result = await db.execute(
        select(Run)
        .options(
            selectinload(Run.results),
            selectinload(Run.child_runs).selectinload(Run.results),
        )
        .where(Run.id == run_id)
    )
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Aggregate results from this run and all child runs
    all_results: List[Result] = list(run.results)
    child_run_ids: List[UUID] = []
    has_running_extension = False
    total_calls = run.total_calls
    completed_calls = run.completed_calls
    failed_calls = run.failed_calls
    actual_cost = float(run.actual_cost)

    for child_run in run.child_runs:
        child_run_ids.append(child_run.id)
        all_results.extend(child_run.results)
        total_calls += child_run.total_calls
        completed_calls += child_run.completed_calls
        failed_calls += child_run.failed_calls
        actual_cost += float(child_run.actual_cost)
        if child_run.status in {Run.STATUS_QUEUED, Run.STATUS_RUNNING}:
            has_running_extension = True

    # Calculate progress
    completed = completed_calls + failed_calls
    progress_percent = (completed / total_calls * 100) if total_calls > 0 else 0

    # Estimate remaining time
    estimated_remaining: Optional[int] = None
    if run.status == Run.STATUS_RUNNING and completed > 0:
        elapsed = (datetime.utcnow() - run.created_at.replace(tzinfo=None)).total_seconds()
        rate = completed / elapsed if elapsed > 0 else 0
        remaining_calls = total_calls - completed
        estimated_remaining = int(remaining_calls / rate) if rate > 0 else None

    # Build summary from successful results (all runs combined)
    summary = _build_summary(all_results, run.brand) if all_results else None

    # Convert results to response format
    result_items = [
        ResultItem(
            id=r.id,
            prompt=r.prompt,
            provider=r.provider,
            model=r.model,
            temperature=float(r.temperature),
            repeat_index=r.repeat_index,
            response_text=r.response_text,
            error=r.error,
            brand_mentioned=r.brand_mentioned,
            competitors_mentioned=r.competitors_mentioned,
            response_type=r.response_type,
            tokens=r.tokens,
            cost=float(r.cost) if r.cost else None,
            sources=r.sources,
            grounding_metadata=r.grounding_metadata,
            brand_sentiment=r.brand_sentiment,
            competitor_sentiments=r.competitor_sentiments,
            all_brands_mentioned=r.all_brands_mentioned,
            source_brand_sentiments=r.source_brand_sentiments,
            created_at=r.created_at,
        )
        for r in sorted(all_results, key=lambda x: x.created_at)
    ]

    # Get search_type from config (default to 'brand' for backwards compatibility)
    search_type = run.config.get("search_type", "brand") if run.config else "brand"

    # Build extension info
    extension_info = None
    if run.parent_run_id or child_run_ids:
        extension_info = {
            "parent_run_id": str(run.parent_run_id) if run.parent_run_id else None,
            "child_run_ids": [str(cid) for cid in child_run_ids],
            "has_running_extension": has_running_extension,
        }

    return RunStatusResponse(
        run_id=run.id,
        status=run.status,
        brand=run.brand,
        search_type=search_type,
        total_calls=total_calls,
        completed_calls=completed_calls,
        failed_calls=failed_calls,
        progress_percent=round(progress_percent, 1),
        estimated_seconds_remaining=estimated_remaining,
        actual_cost=actual_cost,
        created_at=run.created_at,
        completed_at=run.completed_at,
        summary=summary,
        results=result_items,
        config=run.config,
        extension_info=extension_info,
    )


@router.post("/run/{run_id}/extend", response_model=ExtendRunResponse, status_code=201)
async def extend_run(
    run_id: UUID,
    request: ExtendRunRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseDep,
) -> ExtendRunResponse:
    """Extend a completed run with new prompts, competitors, or providers.

    Creates a child run that only executes new combinations not already
    present in the parent run. Results are aggregated when fetching status.

    Args:
        run_id: The UUID of the parent run to extend.
        request: ExtendRunRequest with new prompts/competitors/providers.
        background_tasks: FastAPI background tasks.
        db: Database session.

    Returns:
        ExtendRunResponse with child run_id and estimated cost/duration.

    Raises:
        HTTPException: If run not found, not complete, or no new combinations.
    """
    # Fetch parent run with child runs to check for running extensions
    result = await db.execute(
        select(Run)
        .options(selectinload(Run.child_runs))
        .where(Run.id == run_id)
    )
    parent_run = result.scalar_one_or_none()

    if not parent_run:
        raise HTTPException(status_code=404, detail="Run not found")

    if parent_run.status != Run.STATUS_COMPLETE:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot extend run with status '{parent_run.status}'. Run must be complete.",
        )

    # Check for running extensions
    for child in parent_run.child_runs:
        if child.status in {Run.STATUS_QUEUED, Run.STATUS_RUNNING}:
            raise HTTPException(
                status_code=400,
                detail="Cannot extend run while another extension is in progress.",
            )

    # Validate request has at least one addition
    if not request.add_prompts and not request.add_competitors and not request.add_providers:
        raise HTTPException(
            status_code=400,
            detail="At least one of add_prompts, add_competitors, or add_providers must be provided.",
        )

    # Get parent config
    parent_config = parent_run.config or {}
    parent_prompts = set(parent_config.get("prompts", []))
    parent_competitors = set(parent_config.get("competitors", []))
    parent_providers = set(parent_config.get("providers", []))
    parent_temperatures = parent_config.get("temperatures", [0.7])
    parent_repeats = parent_config.get("repeats", 1)

    # Calculate new items (filter out duplicates)
    new_prompts = [p for p in (request.add_prompts or []) if p not in parent_prompts]
    new_competitors = [c for c in (request.add_competitors or []) if c not in parent_competitors]
    new_providers = [p for p in (request.add_providers or []) if p not in parent_providers]

    # Build combined sets for child run
    all_prompts = list(parent_prompts) + new_prompts
    all_competitors = list(parent_competitors) + new_competitors
    all_providers = list(parent_providers) + new_providers

    # Calculate which prompt/provider combinations are new
    # New combinations = (new prompts x all providers) + (all prompts x new providers)
    # But we need to avoid double-counting (new prompts x new providers)
    new_combinations = set()

    # New prompts with all providers
    for prompt in new_prompts:
        for provider in all_providers:
            for temp in parent_temperatures:
                for repeat in range(parent_repeats):
                    new_combinations.add((prompt, provider, temp, repeat))

    # All prompts with new providers (excluding already counted new prompts)
    for prompt in parent_prompts:
        for provider in new_providers:
            for temp in parent_temperatures:
                for repeat in range(parent_repeats):
                    new_combinations.add((prompt, provider, temp, repeat))

    if not new_combinations:
        raise HTTPException(
            status_code=400,
            detail="No new combinations to run. All provided prompts/providers already exist in the run.",
        )

    total_new_calls = len(new_combinations)

    # Check against 300 call limit (considering parent + existing children + new)
    existing_total = parent_run.total_calls
    for child in parent_run.child_runs:
        existing_total += child.total_calls

    if existing_total + total_new_calls > 300:
        remaining_capacity = 300 - existing_total
        raise HTTPException(
            status_code=400,
            detail=f"Extension would exceed 300 call limit. Current: {existing_total}, Requested: {total_new_calls}, Remaining capacity: {remaining_capacity}",
        )

    # Calculate estimates for the child run
    # Group providers for cost estimation
    child_providers = set()
    for _, provider, _, _ in new_combinations:
        child_providers.add(provider)

    estimated_cost = estimate_run_cost(
        num_prompts=len(set(p for p, _, _, _ in new_combinations)),
        num_providers=len(child_providers),
        num_temperatures=len(parent_temperatures),
        num_repeats=parent_repeats,
        providers=list(child_providers),
    )
    # Adjust cost based on actual number of calls vs estimated
    estimated_calls_from_formula = (
        len(set(p for p, _, _, _ in new_combinations))
        * len(child_providers)
        * len(parent_temperatures)
        * parent_repeats
    )
    if estimated_calls_from_formula > 0:
        estimated_cost = estimated_cost * total_new_calls / estimated_calls_from_formula

    estimated_duration = estimate_duration_seconds(total_new_calls)

    # Build child run config
    # Only include prompts and providers that are part of new combinations
    child_prompts = list(set(p for p, _, _, _ in new_combinations))
    child_provider_list = list(child_providers)

    child_config = {
        "brand": parent_run.brand,
        "search_type": parent_config.get("search_type", "brand"),
        "prompts": child_prompts,
        "competitors": all_competitors,  # Use all competitors for analysis
        "providers": child_provider_list,
        "temperatures": parent_temperatures,
        "repeats": parent_repeats,
        # Track which combinations to actually execute
        "_new_combinations": [
            {"prompt": p, "provider": prov, "temperature": t, "repeat": r}
            for p, prov, t, r in new_combinations
        ],
    }

    # Create child run
    child_run = Run(
        session_id=parent_run.session_id,
        user_id=parent_run.user_id,
        parent_run_id=parent_run.id,
        status=Run.STATUS_QUEUED,
        brand=parent_run.brand,
        config=child_config,
        total_calls=total_new_calls,
        estimated_cost=estimated_cost,
    )
    db.add(child_run)
    await db.commit()
    await db.refresh(child_run)

    # Start background execution
    background_tasks.add_task(executor.execute_run, child_run.id, child_config)

    return ExtendRunResponse(
        run_id=child_run.id,
        status=child_run.status,
        total_calls=total_new_calls,
        estimated_cost=round(estimated_cost, 4),
        estimated_duration_seconds=estimated_duration,
    )


@router.post("/run/{run_id}/cancel", response_model=CancelResponse)
async def cancel_run(run_id: UUID, db: DatabaseDep) -> CancelResponse:
    """Cancel a running or queued run.

    Args:
        run_id: The UUID of the run to cancel.
        db: Database session.

    Returns:
        CancelResponse with final stats.

    Raises:
        HTTPException: If run not found or cannot be cancelled.
    """
    run = await db.get(Run, run_id)

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status not in {Run.STATUS_QUEUED, Run.STATUS_RUNNING}:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel run with status '{run.status}'",
        )

    # Mark as cancelled
    run.cancelled = True
    run.status = Run.STATUS_CANCELLED
    run.completed_at = datetime.utcnow()
    await db.commit()

    # Calculate cancelled calls
    completed = run.completed_calls + run.failed_calls
    cancelled_calls = run.total_calls - completed

    return CancelResponse(
        run_id=run.id,
        status=run.status,
        completed_calls=run.completed_calls,
        cancelled_calls=cancelled_calls,
        actual_cost=float(run.actual_cost),
    )


@router.get("/run/{run_id}/ai-summary", response_model=AISummaryResponse)
async def get_ai_summary(run_id: UUID, db: DatabaseDep) -> AISummaryResponse:
    """Generate an AI summary of the run results.

    Args:
        run_id: The UUID of the run.
        db: Database session.

    Returns:
        AISummaryResponse with the AI-generated summary.

    Raises:
        HTTPException: If run not found or not complete.
    """
    # Fetch run with results
    result = await db.execute(
        select(Run)
        .options(selectinload(Run.results))
        .where(Run.id == run_id)
    )
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status != Run.STATUS_COMPLETE:
        raise HTTPException(
            status_code=400,
            detail=f"Run must be complete to generate summary. Current status: {run.status}",
        )

    # Filter to successful results only
    successful_results = [r for r in run.results if r.error is None and r.response_text]

    if not successful_results:
        raise HTTPException(status_code=400, detail="No successful results to summarize")

    # Format results data for the AI
    results_data = _format_results_for_ai(successful_results, run.brand)

    # Get search_type from config
    search_type = run.config.get("search_type", "brand") if run.config else "brand"

    # Generate summary and recommendations using OpenAI
    try:
        openai_service = OpenAIService()
        result = await openai_service.generate_results_summary(
            brand=run.brand,
            search_type=search_type,
            results_data=results_data,
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI service not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")

    if not result.get("summary"):
        raise HTTPException(status_code=500, detail="Failed to generate summary")

    return AISummaryResponse(
        run_id=run_id,
        summary=result["summary"],
        recommendations=result.get("recommendations", ""),
        generated_at=datetime.utcnow(),
    )


def _provider_display_name(provider: str) -> str:
    """Map internal provider ID to user-facing model name."""
    return {
        "openai": "ChatGPT",
        "anthropic": "Claude",
        "gemini": "Gemini",
        "perplexity": "Perplexity",
        "grok": "Grok",
        "llama": "Llama",
        "ai_overviews": "AI Overviews",
    }.get(provider, provider)


def _format_results_for_ai(results: List[Result], brand: str) -> str:
    """Format results into a readable string for AI analysis.

    Includes pre-computed aggregate statistics so the LLM can reference
    exact numbers without having to count individual results.

    Args:
        results: List of successful Result objects.
        brand: The brand being analyzed.

    Returns:
        Formatted string with all results data.
    """
    lines = []
    lines.append(f"Brand/Category: {brand}")
    lines.append(f"Total Responses: {len(results)}")
    lines.append("")

    # === Pre-computed aggregate statistics ===
    lines.append("=== AGGREGATE STATISTICS (use these exact numbers) ===")

    # Overall brand mention rate
    mentioned_count = sum(1 for r in results if r.brand_mentioned)
    mention_rate = (mentioned_count / len(results) * 100) if results else 0
    lines.append(f"Overall Brand Visibility: {brand} mentioned in {mentioned_count}/{len(results)} responses ({mention_rate:.1f}%)")

    # Per-provider stats
    by_provider: Dict[str, List[Result]] = {}
    for r in results:
        if r.provider not in by_provider:
            by_provider[r.provider] = []
        by_provider[r.provider].append(r)

    lines.append("")
    lines.append("Brand Visibility by Provider:")
    for provider, provider_results in by_provider.items():
        p_mentioned = sum(1 for r in provider_results if r.brand_mentioned)
        p_rate = (p_mentioned / len(provider_results) * 100) if provider_results else 0
        lines.append(f"  {_provider_display_name(provider)}: {p_mentioned}/{len(provider_results)} ({p_rate:.1f}%)")

    # Competitor mention rates
    competitor_counts: Dict[str, int] = {}
    for r in results:
        if r.competitors_mentioned:
            for comp in r.competitors_mentioned:
                competitor_counts[comp] = competitor_counts.get(comp, 0) + 1

    if competitor_counts:
        lines.append("")
        lines.append("Competitor Mention Rates:")
        for comp, count in sorted(competitor_counts.items(), key=lambda x: -x[1]):
            c_rate = (count / len(results) * 100) if results else 0
            lines.append(f"  {comp}: {count}/{len(results)} ({c_rate:.1f}%)")

    # Sentiment distribution
    sentiment_counts: Dict[str, int] = {}
    for r in results:
        if r.brand_sentiment:
            sentiment_counts[r.brand_sentiment] = sentiment_counts.get(r.brand_sentiment, 0) + 1

    if sentiment_counts:
        lines.append("")
        lines.append(f"Brand Sentiment Distribution for {brand}:")
        for sent, count in sorted(sentiment_counts.items(), key=lambda x: -x[1]):
            s_rate = (count / len(results) * 100) if results else 0
            lines.append(f"  {sent}: {count} ({s_rate:.1f}%)")

    lines.append("")
    lines.append("=== INDIVIDUAL RESPONSES ===")
    lines.append("")

    for provider, provider_results in by_provider.items():
        lines.append(f"=== {_provider_display_name(provider)} ({len(provider_results)} responses) ===")

        for r in provider_results:
            lines.append(f"\nPrompt: {r.prompt}")
            lines.append(f"Brand Mentioned: {'Yes' if r.brand_mentioned else 'No'}")
            if r.competitors_mentioned:
                lines.append(f"Competitors Mentioned: {', '.join(r.competitors_mentioned)}")
            else:
                lines.append("Competitors Mentioned: None")
            lines.append(f"Response Type: {r.response_type or 'Unknown'}")

            # Include truncated response text
            if r.response_text:
                truncated = r.response_text[:500] + "..." if len(r.response_text) > 500 else r.response_text
                lines.append(f"Response Preview: {truncated}")

            # Include sources if available
            if r.sources:
                source_titles = [s.get("title", s.get("url", ""))[:50] for s in r.sources[:3]]
                lines.append(f"Sources: {', '.join(source_titles)}")

            lines.append("")

    return "\n".join(lines)


def _build_summary(results: List[Result], brand: str) -> Optional[RunSummary]:
    """Build summary statistics from results.

    Args:
        results: List of Result objects.
        brand: The brand being tracked.

    Returns:
        RunSummary with aggregated statistics, or None if no successful results.
    """
    # Filter to successful results only
    successful = [r for r in results if r.error is None and r.response_text]

    if not successful:
        return None

    # Calculate overall brand mention rate
    mentioned_count = sum(1 for r in successful if r.brand_mentioned)
    brand_mention_rate = mentioned_count / len(successful) if successful else 0

    # Calculate by provider
    by_provider: Dict[str, ProviderStats] = {}
    providers = set(r.provider for r in successful)

    for provider in providers:
        provider_results = [r for r in successful if r.provider == provider]
        provider_mentioned = sum(1 for r in provider_results if r.brand_mentioned)
        by_provider[provider] = ProviderStats(
            mentioned=provider_mentioned,
            total=len(provider_results),
            rate=provider_mentioned / len(provider_results) if provider_results else 0,
        )

    # Calculate competitor mentions
    competitor_mentions: Dict[str, CompetitorStats] = {}
    all_competitors: Dict[str, int] = {}

    for r in successful:
        if r.competitors_mentioned:
            for comp in r.competitors_mentioned:
                all_competitors[comp] = all_competitors.get(comp, 0) + 1

    for comp, count in all_competitors.items():
        competitor_mentions[comp] = CompetitorStats(
            count=count,
            rate=count / len(successful) if successful else 0,
        )

    return RunSummary(
        brand_mention_rate=round(brand_mention_rate, 2),
        by_provider=by_provider,
        competitor_mentions=competitor_mentions,
    )
