"""Scheduler service for running scheduled reports using APScheduler."""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import engine
from app.models.run import Run
from app.models.scheduled_report import ScheduledReport
from app.models.session import Session
from app.models.user import User
from app.services.executor import RunExecutor


class SchedulerService:
    """Service for managing scheduled report execution.

    Uses APScheduler to run reports at their scheduled times.
    Reports are executed using the RunExecutor and results are
    emailed to users.
    """

    def __init__(self):
        """Initialize the scheduler service."""
        self.scheduler = AsyncIOScheduler()
        self.executor = RunExecutor()
        self._session_factory: Optional[async_sessionmaker] = None

    def _get_session_factory(self) -> async_sessionmaker:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                bind=engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._session_factory

    async def start(self) -> None:
        """Start the scheduler and load all active scheduled reports."""
        print("[Scheduler] Starting scheduler service...")

        # Add the main job that checks for and runs scheduled reports
        self.scheduler.add_job(
            self._check_and_run_reports,
            CronTrigger(minute="*/5"),  # Check every 5 minutes
            id="check_scheduled_reports",
            replace_existing=True,
        )

        self.scheduler.start()
        print("[Scheduler] Scheduler started successfully")

        # Run an initial check
        asyncio.create_task(self._check_and_run_reports())

    async def shutdown(self) -> None:
        """Shutdown the scheduler gracefully."""
        print("[Scheduler] Shutting down scheduler service...")
        self.scheduler.shutdown(wait=True)
        print("[Scheduler] Scheduler shutdown complete")

    async def _check_and_run_reports(self) -> None:
        """Check for reports that need to run and execute them."""
        print("[Scheduler] Checking for scheduled reports to run...")

        session_factory = self._get_session_factory()

        async with session_factory() as session:
            # Find all active reports where next_run_at is in the past
            now = datetime.now(ZoneInfo("UTC"))
            result = await session.execute(
                select(ScheduledReport)
                .where(ScheduledReport.is_active == True)
                .where(ScheduledReport.next_run_at <= now)
            )
            reports = result.scalars().all()

            if not reports:
                print("[Scheduler] No reports ready to run")
                return

            print(f"[Scheduler] Found {len(reports)} report(s) to run")

            for report in reports:
                try:
                    await self._execute_report(session, report)
                except Exception as e:
                    print(f"[Scheduler] Error executing report {report.id}: {e}")

            await session.commit()

    async def _execute_report(
        self,
        session: AsyncSession,
        report: ScheduledReport,
    ) -> None:
        """Execute a single scheduled report.

        Args:
            session: Database session
            report: The scheduled report to execute
        """
        print(f"[Scheduler] Executing report '{report.name}' (ID: {report.id})")

        # Create a session for the run
        run_session = Session(
            session_id=f"scheduled-report-{report.id}-{datetime.utcnow().timestamp()}",
            expires_at=datetime.utcnow() + timedelta(days=7),
        )
        session.add(run_session)
        await session.flush()

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
            session_id=run_session.id,
            user_id=report.user_id,
            status=Run.STATUS_QUEUED,
            brand=report.brand,
            config=config,
            total_calls=total_calls,
        )
        session.add(run)
        await session.flush()

        # Update report timestamps
        report.last_run_at = datetime.now(ZoneInfo("UTC"))
        report.next_run_at = self._calculate_next_run(report)

        await session.commit()

        # Execute the run in the background
        asyncio.create_task(self._run_and_notify(report.id, run.id, config))

        print(f"[Scheduler] Started run {run.id} for report '{report.name}'")

    async def _run_and_notify(
        self,
        report_id: UUID,
        run_id: UUID,
        config: dict,
    ) -> None:
        """Execute a run and send email notification when complete.

        Args:
            report_id: The scheduled report ID
            run_id: The run ID
            config: Run configuration
        """
        try:
            # Execute the run
            await self.executor.execute_run(run_id, config)

            # Wait for completion and send email
            await self._wait_and_send_email(report_id, run_id)

        except Exception as e:
            print(f"[Scheduler] Error in run_and_notify for run {run_id}: {e}")

    async def _wait_and_send_email(
        self,
        report_id: UUID,
        run_id: UUID,
        max_wait_seconds: int = 3600,
    ) -> None:
        """Wait for a run to complete and send the email notification.

        Args:
            report_id: The scheduled report ID
            run_id: The run ID
            max_wait_seconds: Maximum time to wait for completion
        """
        session_factory = self._get_session_factory()
        start_time = datetime.utcnow()

        while True:
            async with session_factory() as session:
                run = await session.get(Run, run_id)

                if not run:
                    print(f"[Scheduler] Run {run_id} not found")
                    return

                if run.is_complete:
                    print(f"[Scheduler] Run {run_id} completed with status: {run.status}")

                    # Load report and user for email
                    report = await session.get(ScheduledReport, report_id)
                    if report:
                        user = await session.get(User, report.user_id)
                        if user and user.email:
                            try:
                                # Import here to avoid circular imports
                                from app.services.email_service import email_service
                                await email_service.send_report_email(
                                    to_email=user.email,
                                    report=report,
                                    run=run,
                                )
                            except Exception as e:
                                print(f"[Scheduler] Failed to send email: {e}")
                    return

            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > max_wait_seconds:
                print(f"[Scheduler] Timeout waiting for run {run_id}")
                return

            # Wait before checking again
            await asyncio.sleep(10)

    def _calculate_next_run(self, report: ScheduledReport) -> datetime:
        """Calculate the next run time for a report.

        Args:
            report: The scheduled report

        Returns:
            Next run datetime in UTC
        """
        try:
            tz = ZoneInfo(report.timezone)
        except KeyError:
            tz = ZoneInfo("UTC")

        now = datetime.now(tz)

        # Start with today/now at the specified hour
        next_run = now.replace(hour=report.hour, minute=0, second=0, microsecond=0)

        if report.frequency == "daily":
            # Schedule for tomorrow
            next_run += timedelta(days=1)
        else:  # weekly
            day_of_week = report.day_of_week if report.day_of_week is not None else 0
            current_day = now.weekday()
            days_ahead = day_of_week - current_day

            # Always schedule for next week
            if days_ahead <= 0:
                days_ahead += 7

            next_run += timedelta(days=days_ahead)

        # Convert to UTC
        return next_run.astimezone(ZoneInfo("UTC"))


# Global scheduler instance
scheduler_service = SchedulerService()
