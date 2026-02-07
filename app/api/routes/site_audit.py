"""Site audit endpoints for LLM optimization analysis."""

from datetime import datetime, timedelta
from typing import List
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import select

from app.api.deps import DatabaseDep
from app.models.session import Session
from app.models.site_audit import SiteAudit
from app.schemas.site_audit import (
    Recommendation,
    SiteAuditListResponse,
    SiteAuditRequest,
    SiteAuditResponse,
    SiteAuditResult,
    AuditResults,
)
from app.services.site_audit_service import site_audit_service

router = APIRouter()


@router.post("/site-audit", response_model=SiteAuditResponse, status_code=201)
async def create_site_audit(
    request: SiteAuditRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseDep,
) -> SiteAuditResponse:
    """Start a new site audit.

    Creates an audit record and starts background analysis.
    Returns immediately with audit_id for polling.

    Args:
        request: SiteAuditRequest with URL and session_id.
        background_tasks: FastAPI background tasks.
        db: Database session.

    Returns:
        SiteAuditResponse with audit_id and status.

    Raises:
        HTTPException: If validation fails or audit cannot be created.
    """
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

    # Create audit record
    audit = SiteAudit(
        session_id=session.id,
        url=request.url,
        status=SiteAudit.STATUS_QUEUED,
    )
    db.add(audit)
    await db.commit()
    await db.refresh(audit)

    # Start background audit
    background_tasks.add_task(site_audit_service.run_audit, audit.id)

    return SiteAuditResponse(
        audit_id=audit.id,
        status=audit.status,
        url=audit.url,
    )


@router.get("/site-audit/{audit_id}", response_model=SiteAuditResult)
async def get_site_audit(audit_id: UUID, db: DatabaseDep) -> SiteAuditResult:
    """Get the status and results of a site audit.

    Args:
        audit_id: The UUID of the audit.
        db: Database session.

    Returns:
        SiteAuditResult with full status and results.

    Raises:
        HTTPException: If audit not found.
    """
    audit = await db.get(SiteAudit, audit_id)

    if not audit:
        raise HTTPException(status_code=404, detail="Audit not found")

    # Parse results if available
    results = None
    recommendations = None

    if audit.results:
        try:
            results = AuditResults(**audit.results)
            # Regenerate recommendations from results
            recommendations = site_audit_service._generate_recommendations(results)
        except Exception:
            # If parsing fails, return raw results
            pass

    return SiteAuditResult(
        audit_id=audit.id,
        url=audit.url,
        status=audit.status,
        overall_score=audit.overall_score,
        results=results,
        recommendations=recommendations,
        error_message=audit.error_message,
        created_at=audit.created_at,
        completed_at=audit.completed_at,
    )


@router.get("/site-audits", response_model=SiteAuditListResponse)
async def list_site_audits(
    session_id: str,
    db: DatabaseDep,
) -> SiteAuditListResponse:
    """List all site audits for a session.

    Args:
        session_id: The client session identifier.
        db: Database session.

    Returns:
        SiteAuditListResponse with list of audits.
    """
    # Find session
    result = await db.execute(
        select(Session).where(Session.session_id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        return SiteAuditListResponse(audits=[], total=0)

    # Get audits for session
    result = await db.execute(
        select(SiteAudit)
        .where(SiteAudit.session_id == session.id)
        .order_by(SiteAudit.created_at.desc())
    )
    audits = result.scalars().all()

    audit_results: List[SiteAuditResult] = []
    for audit in audits:
        results = None
        recommendations = None

        if audit.results:
            try:
                results = AuditResults(**audit.results)
                recommendations = site_audit_service._generate_recommendations(results)
            except Exception:
                pass

        audit_results.append(
            SiteAuditResult(
                audit_id=audit.id,
                url=audit.url,
                status=audit.status,
                overall_score=audit.overall_score,
                results=results,
                recommendations=recommendations,
                error_message=audit.error_message,
                created_at=audit.created_at,
                completed_at=audit.completed_at,
            )
        )

    return SiteAuditListResponse(
        audits=audit_results,
        total=len(audit_results),
    )
