"""Pydantic schemas for site audit endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class SiteAuditRequest(BaseModel):
    """Request body for starting a site audit.

    Attributes:
        url: The URL to audit.
        session_id: Client session identifier.
    """

    url: str = Field(..., min_length=1, max_length=2048)
    session_id: str = Field(..., min_length=1, max_length=255)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            v = f"https://{v}"
        return v


class SiteAuditResponse(BaseModel):
    """Response after starting a site audit.

    Attributes:
        audit_id: Unique identifier for the audit.
        status: Current status of the audit.
        url: The URL being audited.
    """

    audit_id: UUID
    status: str
    url: str


class CrawlerStatus(BaseModel):
    """Status of a specific AI crawler's access."""

    name: str
    user_agent: str
    allowed: bool
    rule: Optional[str] = None


class RobotsTxtResult(BaseModel):
    """Results from robots.txt analysis."""

    found: bool
    crawlers: List[CrawlerStatus]
    raw_content: Optional[str] = None


class MetaDirectivesResult(BaseModel):
    """Results from meta directives analysis."""

    has_noai: bool
    has_noimageai: bool
    x_robots_tag: Optional[str] = None
    meta_robots_content: Optional[str] = None


class LlmsTxtResult(BaseModel):
    """Results from llms.txt analysis."""

    found: bool
    content: Optional[str] = None
    parsed_sections: Optional[Dict[str, str]] = None


class StructuredDataItem(BaseModel):
    """A single structured data item found on the page."""

    type: str
    properties: Dict[str, Any]


class StructuredDataResult(BaseModel):
    """Results from structured data analysis."""

    has_json_ld: bool
    has_open_graph: bool
    has_twitter_cards: bool
    json_ld_types: List[str]
    json_ld_items: List[StructuredDataItem]
    open_graph_tags: Dict[str, str]
    twitter_card_tags: Dict[str, str]


class ContentAccessibilityResult(BaseModel):
    """Results from content accessibility analysis."""

    initial_html_length: int
    has_noscript_content: bool
    estimated_ssr: bool
    content_snippet: Optional[str] = None


class HeadingItem(BaseModel):
    """A heading element found on the page."""

    level: int
    text: str


class ContentStructureResult(BaseModel):
    """Results from content structure analysis."""

    has_valid_heading_hierarchy: bool
    headings: List[HeadingItem]
    has_header: bool
    has_main: bool
    has_footer: bool
    has_article: bool
    has_nav: bool
    semantic_elements_count: int


class Recommendation(BaseModel):
    """A recommendation for improving LLM optimization."""

    category: str
    priority: str  # high, medium, low
    title: str
    description: str


class AuditResults(BaseModel):
    """Complete audit results."""

    robots_txt: RobotsTxtResult
    meta_directives: MetaDirectivesResult
    llms_txt: LlmsTxtResult
    structured_data: StructuredDataResult
    content_accessibility: ContentAccessibilityResult
    content_structure: ContentStructureResult


class SiteAuditResult(BaseModel):
    """Full site audit result.

    Attributes:
        audit_id: Unique identifier for the audit.
        url: The URL that was audited.
        status: Current status of the audit.
        overall_score: Calculated score from 0-100.
        results: Detailed audit results by category.
        recommendations: List of recommendations for improvement.
        error_message: Error message if audit failed.
        created_at: When the audit was created.
        completed_at: When the audit completed.
    """

    audit_id: UUID
    url: str
    status: str
    overall_score: Optional[int] = None
    results: Optional[AuditResults] = None
    recommendations: Optional[List[Recommendation]] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SiteAuditListResponse(BaseModel):
    """Response containing a list of site audits."""

    audits: List[SiteAuditResult]
    total: int
