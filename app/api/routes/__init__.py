"""API route definitions."""

from fastapi import APIRouter

from app.api.routes import admin, billing, categorize, health, run, scheduled_reports, site_audit, suggest, users

router = APIRouter()

# Include all route modules
router.include_router(health.router, tags=["health"])
router.include_router(suggest.router, tags=["suggest"])
router.include_router(run.router, tags=["run"])
router.include_router(categorize.router, tags=["categorize"])
router.include_router(scheduled_reports.router, tags=["scheduled-reports"])
router.include_router(site_audit.router, tags=["site-audit"])
router.include_router(users.router, tags=["users"])
router.include_router(billing.router, tags=["billing"])
router.include_router(admin.router, prefix="/admin", tags=["admin"])
