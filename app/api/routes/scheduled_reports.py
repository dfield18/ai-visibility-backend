"""Scheduled reports API endpoints for managing automated reports."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import DatabaseDep
from app.core.auth import CurrentUser
from app.models.scheduled_report import ScheduledReport
from app.models.session import Session
from app.models.run import Run
from app.schemas.scheduled_report import (
    RunNowResponse,
    ScheduledReportCreate,
    ScheduledReportListResponse,
    ScheduledReportResponse,
    ScheduledReportUpdate,
    ToggleResponse,
)
from app.services.executor import RunExecutor
from app.services.user_service import get_or_create_user_from_clerk

router = APIRouter()

# Global executor instance
executor = RunExecutor()


def calculate_next_run(
    frequency: str,
    hour: int,
    timezone: str,
    day_of_week: Optional[int] = None,
    from_time: Optional[datetime] = None,
) -> datetime:
    """Calculate the next run time for a scheduled report.

    Args:
        frequency: 'daily' or 'weekly'
        hour: Hour of day to run (0-23)
        timezone: User's timezone
        day_of_week: Day of week for weekly reports (0=Monday, 6=Sunday)
        from_time: Calculate next run from this time (default: now)

    Returns:
        Next run datetime in UTC
    """
    try:
        tz = ZoneInfo(timezone)
    except KeyError:
        tz = ZoneInfo("UTC")

    now = from_time or datetime.now(tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)

    # Start with today at the specified hour
    next_run = now.replace(hour=hour, minute=0, second=0, microsecond=0)

    if frequency == "daily":
        # If the time has already passed today, schedule for tomorrow
        if next_run <= now:
            next_run += timedelta(days=1)
    else:  # weekly
        if day_of_week is None:
            day_of_week = 0  # Default to Monday

        # Calculate days until next occurrence
        current_day = now.weekday()
        days_ahead = day_of_week - current_day

        if days_ahead < 0 or (days_ahead == 0 and next_run <= now):
            days_ahead += 7

        next_run += timedelta(days=days_ahead)

    # Convert to UTC for storage
    return next_run.astimezone(ZoneInfo("UTC"))


@router.get("/scheduled-reports", response_model=ScheduledReportListResponse)
async def list_scheduled_reports(
    db: DatabaseDep,
    user: CurrentUser,
) -> ScheduledReportListResponse:
    """List all scheduled reports for the current user.

    Args:
        db: Database session
        user: Authenticated user

    Returns:
        List of scheduled reports
    """
    # Get or create user in database
    db_user = await get_or_create_user_from_clerk(db, user)
    await db.commit()

    # Fetch user's scheduled reports
    result = await db.execute(
        select(ScheduledReport)
        .where(ScheduledReport.user_id == db_user.id)
        .order_by(ScheduledReport.created_at.desc())
    )
    reports = result.scalars().all()

    return ScheduledReportListResponse(
        reports=[ScheduledReportResponse.model_validate(r) for r in reports],
        total=len(reports),
    )


MAX_SCHEDULED_REPORTS = 3


@router.post("/scheduled-reports", response_model=ScheduledReportResponse, status_code=201)
async def create_scheduled_report(
    request: ScheduledReportCreate,
    db: DatabaseDep,
    user: CurrentUser,
) -> ScheduledReportResponse:
    """Create a new scheduled report.

    Args:
        request: Report configuration
        db: Database session
        user: Authenticated user

    Returns:
        Created scheduled report

    Raises:
        HTTPException: If validation fails or limit exceeded
    """
    # Validate day_of_week for weekly reports
    if request.frequency == "weekly" and request.day_of_week is None:
        raise HTTPException(
            status_code=400,
            detail="day_of_week is required for weekly reports",
        )

    # Get or create user in database
    db_user = await get_or_create_user_from_clerk(db, user)

    # Check if user has reached the report limit
    existing_count_result = await db.execute(
        select(ScheduledReport)
        .where(ScheduledReport.user_id == db_user.id)
    )
    existing_count = len(existing_count_result.scalars().all())

    if existing_count >= MAX_SCHEDULED_REPORTS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum of {MAX_SCHEDULED_REPORTS} scheduled reports allowed. Please delete an existing report to create a new one.",
        )

    # Calculate next run time
    next_run = calculate_next_run(
        frequency=request.frequency,
        hour=request.hour,
        timezone=request.timezone,
        day_of_week=request.day_of_week,
    )

    # Create the scheduled report
    report = ScheduledReport(
        user_id=db_user.id,
        name=request.name,
        brand=request.brand,
        search_type=request.search_type,
        prompts=request.prompts,
        competitors=request.competitors,
        providers=request.providers,
        temperatures=request.temperatures,
        repeats=request.repeats,
        frequency=request.frequency,
        day_of_week=request.day_of_week,
        hour=request.hour,
        timezone=request.timezone,
        is_active=True,
        next_run_at=next_run,
    )

    db.add(report)
    await db.commit()
    await db.refresh(report)

    return ScheduledReportResponse.model_validate(report)


@router.get("/scheduled-reports/{report_id}", response_model=ScheduledReportResponse)
async def get_scheduled_report(
    report_id: UUID,
    db: DatabaseDep,
    user: CurrentUser,
) -> ScheduledReportResponse:
    """Get a scheduled report by ID.

    Args:
        report_id: Report UUID
        db: Database session
        user: Authenticated user

    Returns:
        Scheduled report

    Raises:
        HTTPException: If report not found or not owned by user
    """
    db_user = await get_or_create_user_from_clerk(db, user)
    await db.commit()

    report = await db.get(ScheduledReport, report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if report.user_id != db_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    return ScheduledReportResponse.model_validate(report)


@router.put("/scheduled-reports/{report_id}", response_model=ScheduledReportResponse)
async def update_scheduled_report(
    report_id: UUID,
    request: ScheduledReportUpdate,
    db: DatabaseDep,
    user: CurrentUser,
) -> ScheduledReportResponse:
    """Update a scheduled report.

    Args:
        report_id: Report UUID
        request: Updated fields
        db: Database session
        user: Authenticated user

    Returns:
        Updated scheduled report

    Raises:
        HTTPException: If report not found, not owned by user, or validation fails
    """
    db_user = await get_or_create_user_from_clerk(db, user)
    await db.commit()

    report = await db.get(ScheduledReport, report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if report.user_id != db_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Track if scheduling fields changed
    schedule_changed = False

    # Update fields
    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if field in ("frequency", "hour", "timezone", "day_of_week"):
            schedule_changed = True
        setattr(report, field, value)

    # Validate day_of_week for weekly reports
    if report.frequency == "weekly" and report.day_of_week is None:
        raise HTTPException(
            status_code=400,
            detail="day_of_week is required for weekly reports",
        )

    # Recalculate next run if scheduling changed
    if schedule_changed and report.is_active:
        report.next_run_at = calculate_next_run(
            frequency=report.frequency,
            hour=report.hour,
            timezone=report.timezone,
            day_of_week=report.day_of_week,
        )

    await db.commit()
    await db.refresh(report)

    return ScheduledReportResponse.model_validate(report)


@router.delete("/scheduled-reports/{report_id}", status_code=204)
async def delete_scheduled_report(
    report_id: UUID,
    db: DatabaseDep,
    user: CurrentUser,
) -> None:
    """Delete a scheduled report.

    Args:
        report_id: Report UUID
        db: Database session
        user: Authenticated user

    Raises:
        HTTPException: If report not found or not owned by user
    """
    db_user = await get_or_create_user_from_clerk(db, user)
    await db.commit()

    report = await db.get(ScheduledReport, report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if report.user_id != db_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    await db.delete(report)
    await db.commit()


@router.post("/scheduled-reports/{report_id}/toggle", response_model=ToggleResponse)
async def toggle_scheduled_report(
    report_id: UUID,
    db: DatabaseDep,
    user: CurrentUser,
) -> ToggleResponse:
    """Toggle a scheduled report's active status.

    Args:
        report_id: Report UUID
        db: Database session
        user: Authenticated user

    Returns:
        Updated active status and next run time

    Raises:
        HTTPException: If report not found or not owned by user
    """
    db_user = await get_or_create_user_from_clerk(db, user)
    await db.commit()

    report = await db.get(ScheduledReport, report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if report.user_id != db_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Toggle active status
    report.is_active = not report.is_active

    # Update next run time if activating
    if report.is_active:
        report.next_run_at = calculate_next_run(
            frequency=report.frequency,
            hour=report.hour,
            timezone=report.timezone,
            day_of_week=report.day_of_week,
        )

    await db.commit()
    await db.refresh(report)

    return ToggleResponse(
        id=report.id,
        is_active=report.is_active,
        next_run_at=report.next_run_at if report.is_active else None,
    )


@router.post("/scheduled-reports/{report_id}/run-now", response_model=RunNowResponse)
async def run_report_now(
    report_id: UUID,
    background_tasks: BackgroundTasks,
    db: DatabaseDep,
    user: CurrentUser,
) -> RunNowResponse:
    """Trigger an immediate run of a scheduled report.

    Args:
        report_id: Report UUID
        background_tasks: FastAPI background tasks
        db: Database session
        user: Authenticated user

    Returns:
        Run ID and confirmation message

    Raises:
        HTTPException: If report not found or not owned by user
    """
    db_user = await get_or_create_user_from_clerk(db, user)
    await db.commit()

    report = await db.get(ScheduledReport, report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if report.user_id != db_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Create a session for the run
    session = Session(
        session_id=f"scheduled-report-{report.id}-{datetime.utcnow().timestamp()}",
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    db.add(session)
    await db.flush()

    # Create run config
    config = {
        "brand": report.brand,
        "search_type": report.search_type,
        "prompts": report.prompts,
        "competitors": report.competitors,
        "providers": report.providers,
        "temperatures": report.temperatures,
        "repeats": report.repeats,
    }

    total_calls = (
        len(report.prompts) *
        len(report.providers) *
        len(report.temperatures) *
        report.repeats
    )

    # Create run record
    run = Run(
        session_id=session.id,
        user_id=db_user.id,
        status=Run.STATUS_QUEUED,
        brand=report.brand,
        config=config,
        total_calls=total_calls,
    )
    db.add(run)

    # Update last_run_at
    report.last_run_at = datetime.utcnow()

    await db.commit()
    await db.refresh(run)

    # Start background execution
    background_tasks.add_task(executor.execute_run, run.id, config)

    return RunNowResponse(
        id=report.id,
        run_id=run.id,
        message=f"Report '{report.name}' started. Run ID: {run.id}",
    )
