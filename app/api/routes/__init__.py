"""API route definitions."""

from fastapi import APIRouter

from app.api.routes import health, run, suggest

router = APIRouter()

# Include all route modules
router.include_router(health.router, tags=["health"])
router.include_router(suggest.router, tags=["suggest"])
router.include_router(run.router, tags=["run"])
