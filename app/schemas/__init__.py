"""Pydantic schemas for API request/response validation."""

from app.schemas.suggest import SuggestRequest, SuggestResponse
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
from app.schemas.scheduled_report import (
    RunNowResponse,
    ScheduledReportCreate,
    ScheduledReportListResponse,
    ScheduledReportResponse,
    ScheduledReportUpdate,
    ToggleResponse,
)
from app.schemas.site_audit import (
    SiteAuditRequest,
    SiteAuditResponse,
    SiteAuditResult,
    SiteAuditListResponse,
)

__all__ = [
    "SuggestRequest",
    "SuggestResponse",
    "RunRequest",
    "RunResponse",
    "RunStatusResponse",
    "CancelResponse",
    "RunSummary",
    "ProviderStats",
    "CompetitorStats",
    "ResultItem",
    "ScheduledReportCreate",
    "ScheduledReportUpdate",
    "ScheduledReportResponse",
    "ScheduledReportListResponse",
    "ToggleResponse",
    "RunNowResponse",
    "SiteAuditRequest",
    "SiteAuditResponse",
    "SiteAuditResult",
    "SiteAuditListResponse",
]
