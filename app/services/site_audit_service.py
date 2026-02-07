"""Site audit service for analyzing LLM/AI search optimization."""

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from uuid import UUID

import httpx
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import engine
from app.models.site_audit import SiteAudit
from app.schemas.site_audit import (
    AuditResults,
    ContentAccessibilityResult,
    ContentStructureResult,
    CrawlerStatus,
    HeadingItem,
    LlmsTxtResult,
    MetaDirectivesResult,
    Recommendation,
    RobotsTxtResult,
    StructuredDataItem,
    StructuredDataResult,
)


# AI crawlers to check in robots.txt
AI_CRAWLERS = [
    {"name": "GPTBot", "user_agent": "GPTBot", "company": "OpenAI"},
    {"name": "ChatGPT-User", "user_agent": "ChatGPT-User", "company": "OpenAI"},
    {"name": "ClaudeBot", "user_agent": "ClaudeBot", "company": "Anthropic"},
    {"name": "Claude-Web", "user_agent": "Claude-Web", "company": "Anthropic"},
    {"name": "PerplexityBot", "user_agent": "PerplexityBot", "company": "Perplexity"},
    {"name": "Google-Extended", "user_agent": "Google-Extended", "company": "Google"},
    {"name": "CCBot", "user_agent": "CCBot", "company": "Common Crawl"},
    {"name": "Applebot-Extended", "user_agent": "Applebot-Extended", "company": "Apple"},
]

# Scoring weights
SCORE_WEIGHTS = {
    "ai_crawler_access": 30,
    "meta_directives": 15,
    "llms_txt": 10,
    "structured_data": 20,
    "content_accessibility": 15,
    "content_structure": 10,
}


class SiteAuditService:
    """Service for running LLM site optimization audits."""

    REQUEST_TIMEOUT = 30.0  # seconds

    def __init__(self):
        """Initialize the audit service."""
        self.http_client: Optional[httpx.AsyncClient] = None

    def _get_session_factory(self) -> async_sessionmaker:
        """Create a new session factory."""
        return async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self.http_client is None or self.http_client.is_closed:
            self.http_client = httpx.AsyncClient(
                timeout=self.REQUEST_TIMEOUT,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; SiteAuditBot/1.0; +https://aivisibility.app)"
                },
            )
        return self.http_client

    async def run_audit(self, audit_id: UUID) -> None:
        """Run a complete site audit.

        Args:
            audit_id: The UUID of the audit to run.
        """
        print(f"[SiteAudit] Starting audit {audit_id}")

        session_factory = self._get_session_factory()

        # Get the audit and update status to running
        async with session_factory() as session:
            audit = await session.get(SiteAudit, audit_id)
            if not audit:
                print(f"[SiteAudit] Audit {audit_id} not found")
                return

            audit.mark_running()
            await session.commit()
            url = audit.url

        try:
            # Parse base URL
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"

            # Run all analysis tasks concurrently
            (
                (html_content, response_headers),
                robots_result,
                llms_result,
            ) = await asyncio.gather(
                self._fetch_page(url),
                self._analyze_robots_txt(base_url),
                self._fetch_llms_txt(base_url),
            )

            # Analyze the fetched HTML
            meta_result = self._analyze_meta_directives(html_content, response_headers)
            structured_data_result = self._analyze_structured_data(html_content)
            content_accessibility_result = self._analyze_content_accessibility(html_content)
            content_structure_result = self._analyze_content_structure(html_content)

            # Build results object
            results = AuditResults(
                robots_txt=robots_result,
                meta_directives=meta_result,
                llms_txt=llms_result,
                structured_data=structured_data_result,
                content_accessibility=content_accessibility_result,
                content_structure=content_structure_result,
            )

            # Calculate score
            score = self._calculate_score(results)

            # Generate recommendations
            recommendations = self._generate_recommendations(results)

            # Update audit with results
            async with session_factory() as session:
                audit = await session.get(SiteAudit, audit_id)
                if audit:
                    audit.mark_complete(
                        results=results.model_dump(),
                        score=score,
                    )
                    await session.commit()

            print(f"[SiteAudit] Audit {audit_id} completed with score {score}")

        except Exception as e:
            print(f"[SiteAudit] Audit {audit_id} failed: {e}")
            async with session_factory() as session:
                audit = await session.get(SiteAudit, audit_id)
                if audit:
                    audit.mark_failed(str(e))
                    await session.commit()

    async def _fetch_page(self, url: str) -> Tuple[str, Dict[str, str]]:
        """Fetch a page and return HTML content and headers.

        Args:
            url: The URL to fetch.

        Returns:
            Tuple of (html_content, response_headers)
        """
        client = await self._get_http_client()
        response = await client.get(url)
        response.raise_for_status()

        headers = {k.lower(): v for k, v in response.headers.items()}
        return response.text, headers

    async def _analyze_robots_txt(self, base_url: str) -> RobotsTxtResult:
        """Analyze robots.txt for AI crawler permissions.

        Args:
            base_url: The base URL of the site.

        Returns:
            RobotsTxtResult with crawler status information.
        """
        robots_url = urljoin(base_url, "/robots.txt")
        client = await self._get_http_client()

        try:
            response = await client.get(robots_url)
            if response.status_code != 200:
                # No robots.txt means all crawlers are allowed
                return RobotsTxtResult(
                    found=False,
                    crawlers=[
                        CrawlerStatus(
                            name=c["name"],
                            user_agent=c["user_agent"],
                            allowed=True,
                            rule=None,
                        )
                        for c in AI_CRAWLERS
                    ],
                )

            content = response.text
            crawlers = self._parse_robots_txt(content)

            return RobotsTxtResult(
                found=True,
                crawlers=crawlers,
                raw_content=content[:5000] if len(content) > 5000 else content,
            )

        except Exception as e:
            print(f"[SiteAudit] Failed to fetch robots.txt: {e}")
            # On error, assume allowed
            return RobotsTxtResult(
                found=False,
                crawlers=[
                    CrawlerStatus(
                        name=c["name"],
                        user_agent=c["user_agent"],
                        allowed=True,
                        rule=None,
                    )
                    for c in AI_CRAWLERS
                ],
            )

    def _parse_robots_txt(self, content: str) -> List[CrawlerStatus]:
        """Parse robots.txt content and check AI crawler permissions.

        Args:
            content: The robots.txt content.

        Returns:
            List of CrawlerStatus for each AI crawler.
        """
        lines = content.lower().split("\n")
        results = []

        for crawler in AI_CRAWLERS:
            user_agent = crawler["user_agent"].lower()
            allowed = True
            matching_rule = None
            in_matching_section = False
            in_wildcard_section = False
            wildcard_disallow = None

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("user-agent:"):
                    agent = line.split(":", 1)[1].strip()
                    if agent == user_agent:
                        in_matching_section = True
                        in_wildcard_section = False
                    elif agent == "*":
                        in_wildcard_section = True
                        in_matching_section = False
                    else:
                        in_matching_section = False
                        in_wildcard_section = False

                elif line.startswith("disallow:"):
                    path = line.split(":", 1)[1].strip()
                    if in_matching_section and path == "/":
                        allowed = False
                        matching_rule = f"Disallow: /"
                    elif in_wildcard_section and path == "/" and wildcard_disallow is None:
                        wildcard_disallow = f"Disallow: / (via User-agent: *)"

                elif line.startswith("allow:"):
                    path = line.split(":", 1)[1].strip()
                    if in_matching_section and path == "/":
                        allowed = True
                        matching_rule = f"Allow: /"

            # If no specific rule found, use wildcard rule
            if matching_rule is None and wildcard_disallow is not None:
                allowed = False
                matching_rule = wildcard_disallow

            results.append(
                CrawlerStatus(
                    name=crawler["name"],
                    user_agent=crawler["user_agent"],
                    allowed=allowed,
                    rule=matching_rule,
                )
            )

        return results

    async def _fetch_llms_txt(self, base_url: str) -> LlmsTxtResult:
        """Check for and fetch llms.txt file.

        Args:
            base_url: The base URL of the site.

        Returns:
            LlmsTxtResult with file content if found.
        """
        llms_url = urljoin(base_url, "/llms.txt")
        client = await self._get_http_client()

        try:
            response = await client.get(llms_url)
            if response.status_code != 200:
                return LlmsTxtResult(found=False)

            content = response.text
            parsed_sections = self._parse_llms_txt(content)

            return LlmsTxtResult(
                found=True,
                content=content[:10000] if len(content) > 10000 else content,
                parsed_sections=parsed_sections,
            )

        except Exception:
            return LlmsTxtResult(found=False)

    def _parse_llms_txt(self, content: str) -> Dict[str, str]:
        """Parse llms.txt content into sections.

        Args:
            content: The llms.txt content.

        Returns:
            Dict of section name to content.
        """
        sections = {}
        current_section = "default"
        current_content = []

        for line in content.split("\n"):
            # Check for section headers (e.g., "# Section Name")
            if line.startswith("# "):
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = line[2:].strip().lower().replace(" ", "_")
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _analyze_meta_directives(
        self, html: str, headers: Dict[str, str]
    ) -> MetaDirectivesResult:
        """Analyze meta tags and headers for AI directives.

        Args:
            html: The HTML content.
            headers: Response headers.

        Returns:
            MetaDirectivesResult with directive information.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Check meta robots tags
        meta_robots_content = None
        has_noai = False
        has_noimageai = False

        for meta in soup.find_all("meta", attrs={"name": re.compile(r"^robots?$", re.I)}):
            content = meta.get("content", "").lower()
            meta_robots_content = content
            if "noai" in content:
                has_noai = True
            if "noimageai" in content:
                has_noimageai = True

        # Check X-Robots-Tag header
        x_robots_tag = headers.get("x-robots-tag", "")
        if x_robots_tag:
            if "noai" in x_robots_tag.lower():
                has_noai = True
            if "noimageai" in x_robots_tag.lower():
                has_noimageai = True

        return MetaDirectivesResult(
            has_noai=has_noai,
            has_noimageai=has_noimageai,
            x_robots_tag=x_robots_tag if x_robots_tag else None,
            meta_robots_content=meta_robots_content,
        )

    def _analyze_structured_data(self, html: str) -> StructuredDataResult:
        """Analyze structured data on the page.

        Args:
            html: The HTML content.

        Returns:
            StructuredDataResult with structured data information.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Find JSON-LD scripts
        json_ld_items = []
        json_ld_types = []

        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            item_type = item.get("@type", "Unknown")
                            json_ld_types.append(item_type)
                            json_ld_items.append(
                                StructuredDataItem(
                                    type=item_type,
                                    properties=self._extract_schema_properties(item),
                                )
                            )
                elif isinstance(data, dict):
                    item_type = data.get("@type", "Unknown")
                    json_ld_types.append(item_type)
                    json_ld_items.append(
                        StructuredDataItem(
                            type=item_type,
                            properties=self._extract_schema_properties(data),
                        )
                    )
            except json.JSONDecodeError:
                continue

        # Find Open Graph tags
        open_graph_tags = {}
        for meta in soup.find_all("meta", property=re.compile(r"^og:")):
            prop = meta.get("property", "")
            content = meta.get("content", "")
            if prop and content:
                open_graph_tags[prop] = content

        # Find Twitter Card tags
        twitter_card_tags = {}
        for meta in soup.find_all("meta", attrs={"name": re.compile(r"^twitter:")}):
            name = meta.get("name", "")
            content = meta.get("content", "")
            if name and content:
                twitter_card_tags[name] = content

        return StructuredDataResult(
            has_json_ld=len(json_ld_items) > 0,
            has_open_graph=len(open_graph_tags) > 0,
            has_twitter_cards=len(twitter_card_tags) > 0,
            json_ld_types=list(set(json_ld_types)),
            json_ld_items=json_ld_items[:10],  # Limit to first 10
            open_graph_tags=open_graph_tags,
            twitter_card_tags=twitter_card_tags,
        )

    def _extract_schema_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key properties from schema.org data.

        Args:
            data: The JSON-LD data.

        Returns:
            Dict of key properties (limited depth).
        """
        result = {}
        skip_keys = {"@context", "@graph"}

        for key, value in data.items():
            if key in skip_keys:
                continue
            if isinstance(value, (str, int, float, bool)):
                result[key] = value
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (str, int, float, bool)):
                    result[key] = value[:5]  # Limit list size
                else:
                    result[key] = f"[{len(value)} items]"
            elif isinstance(value, dict):
                result[key] = value.get("@type", "[object]")

        return result

    def _analyze_content_accessibility(self, html: str) -> ContentAccessibilityResult:
        """Analyze content accessibility for LLM indexing.

        Args:
            html: The HTML content.

        Returns:
            ContentAccessibilityResult with accessibility information.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Calculate initial HTML length (proxy for SSR)
        html_length = len(html)

        # Check for noscript content
        noscript_tags = soup.find_all("noscript")
        has_noscript = any(
            len(tag.get_text(strip=True)) > 50 for tag in noscript_tags
        )

        # Estimate if SSR (heuristic: substantial content in body)
        body = soup.find("body")
        body_text = body.get_text(strip=True) if body else ""
        estimated_ssr = len(body_text) > 500

        # Get content snippet
        content_snippet = None
        main_content = soup.find("main") or soup.find("article") or body
        if main_content:
            text = main_content.get_text(strip=True)[:500]
            content_snippet = text if len(text) > 50 else None

        return ContentAccessibilityResult(
            initial_html_length=html_length,
            has_noscript_content=has_noscript,
            estimated_ssr=estimated_ssr,
            content_snippet=content_snippet,
        )

    def _analyze_content_structure(self, html: str) -> ContentStructureResult:
        """Analyze HTML content structure.

        Args:
            html: The HTML content.

        Returns:
            ContentStructureResult with structure information.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Find all headings
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                text = heading.get_text(strip=True)
                if text:
                    headings.append(HeadingItem(level=level, text=text[:100]))

        # Check heading hierarchy validity
        has_valid_hierarchy = self._check_heading_hierarchy(headings)

        # Check for semantic elements
        has_header = soup.find("header") is not None
        has_main = soup.find("main") is not None
        has_footer = soup.find("footer") is not None
        has_article = soup.find("article") is not None
        has_nav = soup.find("nav") is not None

        semantic_count = sum([has_header, has_main, has_footer, has_article, has_nav])

        return ContentStructureResult(
            has_valid_heading_hierarchy=has_valid_hierarchy,
            headings=headings[:20],  # Limit to first 20
            has_header=has_header,
            has_main=has_main,
            has_footer=has_footer,
            has_article=has_article,
            has_nav=has_nav,
            semantic_elements_count=semantic_count,
        )

    def _check_heading_hierarchy(self, headings: List[HeadingItem]) -> bool:
        """Check if heading hierarchy is valid.

        Args:
            headings: List of headings found on the page.

        Returns:
            True if hierarchy is valid (starts with h1, no skips > 1 level).
        """
        if not headings:
            return True

        # Should start with h1
        if headings[0].level != 1:
            return False

        # Check for level skips
        prev_level = 0
        for heading in headings:
            if heading.level > prev_level + 1 and prev_level > 0:
                return False
            prev_level = heading.level

        return True

    def _calculate_score(self, results: AuditResults) -> int:
        """Calculate overall audit score.

        Args:
            results: The audit results.

        Returns:
            Score from 0-100.
        """
        scores = {}

        # AI Crawler Access (30%)
        allowed_count = sum(1 for c in results.robots_txt.crawlers if c.allowed)
        total_crawlers = len(results.robots_txt.crawlers)
        scores["ai_crawler_access"] = (allowed_count / total_crawlers) if total_crawlers > 0 else 1.0

        # Meta Directives (15%)
        if results.meta_directives.has_noai:
            scores["meta_directives"] = 0.0
        elif results.meta_directives.has_noimageai:
            scores["meta_directives"] = 0.5
        else:
            scores["meta_directives"] = 1.0

        # llms.txt (10%)
        scores["llms_txt"] = 1.0 if results.llms_txt.found else 0.0

        # Structured Data (20%)
        structured_score = 0.0
        if results.structured_data.has_json_ld:
            structured_score += 0.5
        if results.structured_data.has_open_graph:
            structured_score += 0.25
        if results.structured_data.has_twitter_cards:
            structured_score += 0.25
        scores["structured_data"] = structured_score

        # Content Accessibility (15%)
        if results.content_accessibility.estimated_ssr:
            accessibility_score = 0.7
        else:
            accessibility_score = 0.3
        if results.content_accessibility.has_noscript_content:
            accessibility_score += 0.3
        scores["content_accessibility"] = min(accessibility_score, 1.0)

        # Content Structure (10%)
        structure_score = 0.0
        if results.content_structure.has_valid_heading_hierarchy:
            structure_score += 0.5
        structure_score += min(results.content_structure.semantic_elements_count / 5, 0.5)
        scores["content_structure"] = structure_score

        # Calculate weighted total
        total = sum(
            scores[key] * SCORE_WEIGHTS[key]
            for key in SCORE_WEIGHTS
        )

        return round(total)

    def _generate_recommendations(self, results: AuditResults) -> List[Recommendation]:
        """Generate recommendations based on audit results.

        Args:
            results: The audit results.

        Returns:
            List of recommendations.
        """
        recommendations = []

        # Check blocked crawlers
        blocked_crawlers = [c for c in results.robots_txt.crawlers if not c.allowed]
        if blocked_crawlers:
            names = ", ".join(c.name for c in blocked_crawlers[:3])
            recommendations.append(
                Recommendation(
                    category="robots_txt",
                    priority="high",
                    title="Blocked AI Crawlers Detected",
                    description=f"The following AI crawlers are blocked in robots.txt: {names}. "
                    "Consider allowing these crawlers to improve visibility in AI search results.",
                )
            )

        # Check meta directives
        if results.meta_directives.has_noai:
            recommendations.append(
                Recommendation(
                    category="meta_directives",
                    priority="high",
                    title="noai Meta Directive Found",
                    description="Your site has a 'noai' meta directive that prevents AI systems from using your content. "
                    "Remove this if you want to appear in AI-generated search results.",
                )
            )
        elif results.meta_directives.has_noimageai:
            recommendations.append(
                Recommendation(
                    category="meta_directives",
                    priority="medium",
                    title="noimageai Meta Directive Found",
                    description="Your site blocks AI image training. This may limit some AI features but won't prevent text indexing.",
                )
            )

        # Check llms.txt
        if not results.llms_txt.found:
            recommendations.append(
                Recommendation(
                    category="llms_txt",
                    priority="medium",
                    title="No llms.txt File Found",
                    description="Consider adding an llms.txt file to provide instructions to LLMs about your site. "
                    "This emerging standard helps AI systems better understand and represent your content.",
                )
            )

        # Check structured data
        if not results.structured_data.has_json_ld:
            recommendations.append(
                Recommendation(
                    category="structured_data",
                    priority="high",
                    title="No JSON-LD Structured Data",
                    description="Add JSON-LD structured data (Schema.org) to help AI systems understand your content. "
                    "Common types include Organization, Product, Article, and FAQ.",
                )
            )
        if not results.structured_data.has_open_graph:
            recommendations.append(
                Recommendation(
                    category="structured_data",
                    priority="medium",
                    title="Missing Open Graph Tags",
                    description="Add Open Graph meta tags to improve how your content appears when shared or indexed.",
                )
            )

        # Check content accessibility
        if not results.content_accessibility.estimated_ssr:
            recommendations.append(
                Recommendation(
                    category="content_accessibility",
                    priority="high",
                    title="Limited Server-Side Rendered Content",
                    description="Your page appears to rely heavily on client-side JavaScript. "
                    "Consider server-side rendering or static generation to ensure AI crawlers can access your content.",
                )
            )

        # Check content structure
        if not results.content_structure.has_valid_heading_hierarchy:
            recommendations.append(
                Recommendation(
                    category="content_structure",
                    priority="medium",
                    title="Invalid Heading Hierarchy",
                    description="Your page has heading structure issues. Ensure you start with an h1 and don't skip heading levels.",
                )
            )
        if results.content_structure.semantic_elements_count < 3:
            recommendations.append(
                Recommendation(
                    category="content_structure",
                    priority="low",
                    title="Limited Semantic HTML",
                    description="Use more semantic HTML elements (header, main, footer, article, nav) to help AI understand your page structure.",
                )
            )

        return recommendations


# Singleton instance
site_audit_service = SiteAuditService()
