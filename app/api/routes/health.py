"""Health check endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from app.api.deps import DatabaseDep

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response schema.

    Attributes:
        status: Overall health status.
        database: Database connection status.
        timestamp: Current server time in ISO format.
    """

    status: str
    database: str
    timestamp: str


@router.get("/health", response_model=HealthResponse)
async def health_check(db: DatabaseDep) -> HealthResponse:
    """Check application health and database connectivity.

    Returns:
        HealthResponse: Current health status of the application.

    Raises:
        HTTPException: If database is unreachable.
    """
    # Test database connection
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {str(e)}",
        )

    return HealthResponse(
        status="healthy",
        database=db_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/health/live")
async def liveness_check() -> dict:
    """Kubernetes liveness probe endpoint.

    Returns:
        dict: Simple alive status.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_check(db: DatabaseDep) -> dict:
    """Kubernetes readiness probe endpoint.

    Verifies that the application is ready to accept traffic
    by checking database connectivity.

    Returns:
        dict: Ready status.

    Raises:
        HTTPException: If not ready to accept traffic.
    """
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        raise HTTPException(status_code=503, detail="Not ready")
