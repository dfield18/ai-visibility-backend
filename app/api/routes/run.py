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
    CancelResponse,
    CompetitorStats,
    ProviderStats,
    ResultItem,
    RunRequest,
    RunResponse,
    RunStatusResponse,
    RunSummary,
)
from app.services.executor import RunExecutor
from app.services.result_processor import estimate_duration_seconds, estimate_run_cost

router = APIRouter()

# Global executor instance
executor = RunExecutor()


@router.post("/run", response_model=RunResponse)
async def start_run(
    request: RunRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseDep,
) -> RunResponse:
    """Start a brand visibility analysis run.

    Creates a run record and starts background execution.
    Returns immediately with run_id for polling.

    Args:
        request: RunRequest with brand, prompts, providers, etc.
        background_tasks: FastAPI background tasks.
        db: Database session.

    Returns:
        RunResponse with run_id and estimated cost/duration.

    Raises:
        HTTPException: If validation fails or run cannot be created.
    """
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
        "prompts": request.prompts,
        "competitors": request.competitors,
        "providers": request.providers,
        "temperatures": request.temperatures,
        "repeats": request.repeats,
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

    Args:
        run_id: The UUID of the run.
        db: Database session.

    Returns:
        RunStatusResponse with full status, summary, and results.

    Raises:
        HTTPException: If run not found.
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

    # Calculate progress
    completed = run.completed_calls + run.failed_calls
    progress_percent = (completed / run.total_calls * 100) if run.total_calls > 0 else 0

    # Estimate remaining time
    estimated_remaining: Optional[int] = None
    if run.status == Run.STATUS_RUNNING and completed > 0:
        elapsed = (datetime.utcnow() - run.created_at.replace(tzinfo=None)).total_seconds()
        rate = completed / elapsed if elapsed > 0 else 0
        remaining_calls = run.total_calls - completed
        estimated_remaining = int(remaining_calls / rate) if rate > 0 else None

    # Build summary from successful results
    summary = _build_summary(run.results, run.brand) if run.results else None

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
            created_at=r.created_at,
        )
        for r in sorted(run.results, key=lambda x: x.created_at)
    ]

    return RunStatusResponse(
        run_id=run.id,
        status=run.status,
        brand=run.brand,
        total_calls=run.total_calls,
        completed_calls=run.completed_calls,
        failed_calls=run.failed_calls,
        progress_percent=round(progress_percent, 1),
        estimated_seconds_remaining=estimated_remaining,
        actual_cost=float(run.actual_cost),
        created_at=run.created_at,
        completed_at=run.completed_at,
        summary=summary,
        results=result_items,
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
