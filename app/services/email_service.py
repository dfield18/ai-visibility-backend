"""Email service for sending report notifications using Resend."""

import asyncio
import html
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Any
from urllib.parse import urlparse

from app.core.config import settings

if TYPE_CHECKING:
    from app.models.run import Run
    from app.models.scheduled_report import ScheduledReport


class EmailService:
    """Service for sending email notifications.

    Uses Resend API for email delivery. If RESEND_API_KEY is not configured,
    emails will be logged but not sent.
    """

    def __init__(self):
        """Initialize the email service."""
        self.api_key = getattr(settings, 'RESEND_API_KEY', '')
        self._client = None

        if self.api_key:
            try:
                import resend
                resend.api_key = self.api_key
                self._client = resend
                print("[Email] Resend email service initialized")
            except ImportError:
                print("[Email] Warning: resend package not installed, run: pip install resend")
        else:
            print("[Email] Warning: RESEND_API_KEY not configured, emails will be logged only")

    def _send_sync(self, from_email: str, to_email: str, subject: str, html_content: str) -> dict:
        """Synchronous email send - to be run in thread pool."""
        return self._client.Emails.send({
            "from": from_email,
            "to": [to_email],
            "subject": subject,
            "html": html_content,
        })

    async def send_report_email(
        self,
        to_email: str,
        report: "ScheduledReport",
        run: "Run",
    ) -> bool:
        """Send a report completion email."""
        print(f"[Email] Preparing email for {to_email}, report: {report.name}, run: {run.id}")

        try:
            summary = self._build_comprehensive_summary(run)
        except Exception as e:
            print(f"[Email] Error building summary: {e}")
            import traceback
            traceback.print_exc()
            summary = self._empty_summary()

        subject = f"AI Visibility Report: {report.brand} - {datetime.now().strftime('%B %d, %Y')}"
        html_content = self._generate_html_email(
            report_name=report.name,
            brand=report.brand,
            summary=summary,
            run_id=str(run.id),
            frequency=report.frequency,
        )

        if not self._client:
            print(f"[Email] Would send email to {to_email}: {subject}")
            return True

        try:
            from_email = "AI Visibility <onboarding@resend.dev>"
            result = await asyncio.to_thread(
                self._send_sync,
                from_email,
                to_email,
                subject,
                html_content,
            )
            print(f"[Email] Report email sent to {to_email}, result: {result}")
            return True
        except Exception as e:
            print(f"[Email] Failed to send email to {to_email}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _empty_summary(self) -> Dict:
        """Return empty summary structure."""
        return {
            "total_responses": 0,
            "visibility_rate": 0,
            "by_provider": {},
            "top_competitors": [],
            "sentiment": {},
            "sources": [],
            "insights": [],
        }

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path.split('/')[0]
            return domain.lower().replace('www.', '')
        except:
            return url

    def _build_comprehensive_summary(self, run: "Run") -> Dict:
        """Build comprehensive summary statistics from a run."""
        results = run.results or []
        successful = [r for r in results if r.error is None and r.response_text]

        if not successful:
            return self._empty_summary()

        # === OVERVIEW STATS ===
        mentioned_count = sum(1 for r in successful if r.brand_mentioned)
        visibility_rate = mentioned_count / len(successful) if successful else 0

        # By provider
        by_provider = {}
        providers = set(r.provider for r in successful)
        for provider in providers:
            provider_results = [r for r in successful if r.provider == provider]
            provider_mentioned = sum(1 for r in provider_results if r.brand_mentioned)
            by_provider[provider] = {
                "mentioned": provider_mentioned,
                "total": len(provider_results),
                "rate": provider_mentioned / len(provider_results) if provider_results else 0,
            }

        # Top competitors
        competitor_counts: Dict[str, int] = {}
        for r in successful:
            if r.competitors_mentioned:
                for comp in r.competitors_mentioned:
                    competitor_counts[comp] = competitor_counts.get(comp, 0) + 1

        top_competitors = sorted(
            competitor_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # === SENTIMENT ANALYSIS ===
        sentiment_counts: Dict[str, int] = Counter()
        sentiment_labels = {
            "strong_endorsement": "Strongly Recommended",
            "positive_endorsement": "Recommended",
            "neutral_mention": "Neutral Mention",
            "conditional": "Conditional/With Caveats",
            "negative_comparison": "Not Recommended",
            "not_mentioned": "Not Mentioned",
        }

        for r in successful:
            if r.brand_sentiment:
                sentiment_counts[r.brand_sentiment] += 1
            elif r.brand_mentioned:
                sentiment_counts["neutral_mention"] += 1
            else:
                sentiment_counts["not_mentioned"] += 1

        # Calculate sentiment score (1-5 scale)
        sentiment_scores = {
            "strong_endorsement": 5,
            "positive_endorsement": 4,
            "neutral_mention": 3,
            "conditional": 2,
            "negative_comparison": 1,
            "not_mentioned": 0,
        }

        total_sentiment_score = 0
        sentiment_response_count = 0
        for sentiment, count in sentiment_counts.items():
            if sentiment != "not_mentioned":
                total_sentiment_score += sentiment_scores.get(sentiment, 3) * count
                sentiment_response_count += count

        avg_sentiment = total_sentiment_score / sentiment_response_count if sentiment_response_count > 0 else 0

        sentiment_data = {
            "counts": {sentiment_labels.get(k, k): v for k, v in sentiment_counts.items()},
            "average_score": avg_sentiment,
            "total_with_sentiment": sentiment_response_count,
        }

        # === SOURCES ANALYSIS ===
        source_counts: Dict[str, int] = Counter()
        for r in successful:
            if r.sources:
                for source in r.sources:
                    url = source.get('url', '') if isinstance(source, dict) else str(source)
                    if url:
                        domain = self._extract_domain(url)
                        source_counts[domain] += 1

        top_sources = source_counts.most_common(10)
        total_sources = sum(source_counts.values())

        # === INSIGHTS ===
        insights = []

        # Visibility insight
        if visibility_rate >= 0.7:
            insights.append(f"Strong visibility: {run.brand} is mentioned in {int(visibility_rate * 100)}% of AI responses.")
        elif visibility_rate >= 0.4:
            insights.append(f"Moderate visibility: {run.brand} appears in {int(visibility_rate * 100)}% of responses. Room for improvement.")
        elif visibility_rate > 0:
            insights.append(f"Low visibility: {run.brand} only appears in {int(visibility_rate * 100)}% of responses. Consider SEO optimization.")
        else:
            insights.append(f"No visibility: {run.brand} was not mentioned in any AI responses. Urgent attention needed.")

        # Provider insights
        best_provider = max(by_provider.items(), key=lambda x: x[1]['rate'], default=(None, {'rate': 0}))
        worst_provider = min(by_provider.items(), key=lambda x: x[1]['rate'], default=(None, {'rate': 0}))

        provider_labels = {
            "openai": "ChatGPT",
            "gemini": "Google Gemini",
            "anthropic": "Claude",
            "perplexity": "Perplexity",
            "ai_overviews": "Google AI Overviews",
        }

        if best_provider[0] and best_provider[1]['rate'] > 0:
            insights.append(f"Best performing: {provider_labels.get(best_provider[0], best_provider[0])} ({int(best_provider[1]['rate'] * 100)}% visibility)")

        if worst_provider[0] and len(by_provider) > 1 and worst_provider[1]['rate'] < best_provider[1]['rate']:
            insights.append(f"Needs attention: {provider_labels.get(worst_provider[0], worst_provider[0])} ({int(worst_provider[1]['rate'] * 100)}% visibility)")

        # Sentiment insight
        if avg_sentiment >= 4:
            insights.append("Excellent sentiment: AI models are recommending your brand positively.")
        elif avg_sentiment >= 3:
            insights.append("Neutral sentiment: Your brand is mentioned but not strongly recommended.")
        elif avg_sentiment > 0:
            insights.append("Sentiment concern: Some AI responses show negative or conditional sentiment.")

        # Competitor insight
        if top_competitors:
            top_comp = top_competitors[0]
            insights.append(f"Top competitor: {top_comp[0]} mentioned {top_comp[1]} times.")

        return {
            "total_responses": len(successful),
            "visibility_rate": visibility_rate,
            "by_provider": by_provider,
            "top_competitors": top_competitors,
            "sentiment": sentiment_data,
            "sources": top_sources,
            "total_sources": total_sources,
            "insights": insights,
        }

    def _generate_html_email(
        self,
        report_name: str,
        brand: str,
        summary: Dict,
        run_id: str,
        frequency: str,
    ) -> str:
        """Generate HTML email content with tabs-like sections."""
        visibility_pct = int(summary.get("visibility_rate", 0) * 100)
        safe_report_name = html.escape(report_name)
        safe_brand = html.escape(brand)

        # Provider labels
        provider_labels = {
            "openai": "ChatGPT",
            "gemini": "Google Gemini",
            "anthropic": "Claude",
            "perplexity": "Perplexity",
            "ai_overviews": "Google AI Overviews",
        }

        # === OVERVIEW SECTION ===
        provider_rows = ""
        for provider, stats in summary.get("by_provider", {}).items():
            label = provider_labels.get(provider, html.escape(provider))
            rate_pct = int(stats.get("rate", 0) * 100)
            # Color based on rate
            if rate_pct >= 70:
                rate_color = "#059669"  # green
            elif rate_pct >= 40:
                rate_color = "#d97706"  # amber
            else:
                rate_color = "#dc2626"  # red
            provider_rows += f"""
                <tr>
                    <td style="padding: 12px 8px; border-bottom: 1px solid #e5e7eb;">{label}</td>
                    <td style="padding: 12px 8px; border-bottom: 1px solid #e5e7eb; text-align: center;">
                        <span style="color: {rate_color}; font-weight: 600;">{rate_pct}%</span>
                    </td>
                    <td style="padding: 12px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; color: #6b7280;">
                        {stats.get('mentioned', 0)}/{stats.get('total', 0)}
                    </td>
                </tr>
            """

        # === SENTIMENT SECTION ===
        sentiment_data = summary.get("sentiment", {})
        sentiment_counts = sentiment_data.get("counts", {})
        avg_sentiment = sentiment_data.get("average_score", 0)

        # Sentiment bar colors
        sentiment_colors = {
            "Strongly Recommended": "#059669",
            "Recommended": "#10b981",
            "Neutral Mention": "#6b7280",
            "Conditional/With Caveats": "#f59e0b",
            "Not Recommended": "#dc2626",
            "Not Mentioned": "#d1d5db",
        }

        sentiment_rows = ""
        total_sentiment = sum(sentiment_counts.values()) or 1
        for sentiment, count in sentiment_counts.items():
            if count > 0:
                pct = int((count / total_sentiment) * 100)
                color = sentiment_colors.get(sentiment, "#6b7280")
                sentiment_rows += f"""
                    <tr>
                        <td style="padding: 8px 0; font-size: 13px;">{sentiment}</td>
                        <td style="padding: 8px 0; width: 60%;">
                            <div style="background: #e5e7eb; border-radius: 4px; height: 8px; overflow: hidden;">
                                <div style="background: {color}; height: 100%; width: {pct}%;"></div>
                            </div>
                        </td>
                        <td style="padding: 8px 0; text-align: right; font-size: 13px; color: #6b7280;">{count}</td>
                    </tr>
                """

        # Sentiment score display
        if avg_sentiment >= 4:
            sentiment_label = "Excellent"
            sentiment_color = "#059669"
        elif avg_sentiment >= 3:
            sentiment_label = "Good"
            sentiment_color = "#10b981"
        elif avg_sentiment >= 2:
            sentiment_label = "Fair"
            sentiment_color = "#f59e0b"
        else:
            sentiment_label = "Needs Work"
            sentiment_color = "#dc2626"

        # === SOURCES SECTION ===
        sources = summary.get("sources", [])
        total_sources = summary.get("total_sources", 0)

        source_rows = ""
        for i, (domain, count) in enumerate(sources[:8], 1):
            safe_domain = html.escape(domain)
            source_rows += f"""
                <tr>
                    <td style="padding: 8px 0; border-bottom: 1px solid #f3f4f6;">
                        <span style="color: #9ca3af; margin-right: 8px;">{i}.</span>
                        <span style="color: #4A7C59;">{safe_domain}</span>
                    </td>
                    <td style="padding: 8px 0; border-bottom: 1px solid #f3f4f6; text-align: right; color: #6b7280;">
                        {count} citations
                    </td>
                </tr>
            """

        # === COMPETITORS SECTION ===
        competitor_rows = ""
        for comp, count in summary.get("top_competitors", []):
            safe_comp = html.escape(comp)
            competitor_rows += f"""
                <tr>
                    <td style="padding: 8px 0; border-bottom: 1px solid #f3f4f6;">{safe_comp}</td>
                    <td style="padding: 8px 0; border-bottom: 1px solid #f3f4f6; text-align: right; color: #6b7280;">
                        {count} mentions
                    </td>
                </tr>
            """

        # === INSIGHTS SECTION ===
        insights = summary.get("insights", [])
        insight_items = ""
        for insight in insights:
            safe_insight = html.escape(insight)
            insight_items += f'<li style="margin-bottom: 8px; color: #374151;">{safe_insight}</li>'

        # Get frontend URL
        cors_origins = getattr(settings, 'CORS_ORIGINS', None) or []
        frontend_url = cors_origins[0] if cors_origins else "http://localhost:3000"
        results_url = f"{frontend_url}/results/{run_id}"

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f3f4f6;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background-color: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">

            <!-- Header -->
            <div style="background-color: #4A7C59; padding: 24px; text-align: center;">
                <h1 style="margin: 0; color: white; font-size: 24px; font-weight: 600;">
                    AI Visibility Report
                </h1>
                <p style="margin: 8px 0 0; color: #E8F0E8; font-size: 14px;">
                    {safe_report_name}
                </p>
            </div>

            <!-- Main Content -->
            <div style="padding: 24px;">

                <!-- Hero: Visibility Score -->
                <div style="text-align: center; margin-bottom: 32px; padding: 24px; background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%); border-radius: 12px; border: 1px solid #bbf7d0;">
                    <p style="margin: 0 0 8px; color: #6b7280; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">Overall Visibility</p>
                    <p style="margin: 0; font-size: 56px; font-weight: 700; color: #4A7C59;">
                        {visibility_pct}%
                    </p>
                    <p style="margin: 12px 0 0; color: #6b7280; font-size: 14px;">
                        {safe_brand} mentioned in <strong>{summary.get('total_responses', 0)}</strong> AI responses
                    </p>
                </div>

                <!-- SECTION: Overview (Visibility by Platform) -->
                <div style="margin-bottom: 32px; padding: 20px; background-color: #fafafa; border-radius: 8px;">
                    <div style="display: flex; align-items: center; margin-bottom: 16px;">
                        <div style="width: 32px; height: 32px; background-color: #4A7C59; border-radius: 6px; display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                            <span style="color: white; font-size: 16px;">📊</span>
                        </div>
                        <h2 style="margin: 0; font-size: 16px; font-weight: 600; color: #111827;">
                            Overview
                        </h2>
                    </div>
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        <thead>
                            <tr style="border-bottom: 2px solid #e5e7eb;">
                                <th style="padding: 8px; text-align: left; font-weight: 500; color: #6b7280;">AI Platform</th>
                                <th style="padding: 8px; text-align: center; font-weight: 500; color: #6b7280;">Visibility</th>
                                <th style="padding: 8px; text-align: right; font-weight: 500; color: #6b7280;">Mentions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {provider_rows if provider_rows else '<tr><td colspan="3" style="padding: 16px; text-align: center; color: #9ca3af;">No data available</td></tr>'}
                        </tbody>
                    </table>
                </div>

                <!-- SECTION: Sentiment -->
                <div style="margin-bottom: 32px; padding: 20px; background-color: #fafafa; border-radius: 8px;">
                    <div style="display: flex; align-items: center; margin-bottom: 16px;">
                        <div style="width: 32px; height: 32px; background-color: #8b5cf6; border-radius: 6px; display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                            <span style="color: white; font-size: 16px;">💬</span>
                        </div>
                        <h2 style="margin: 0; font-size: 16px; font-weight: 600; color: #111827;">
                            Sentiment Analysis
                        </h2>
                    </div>
                    <div style="text-align: center; margin-bottom: 16px; padding: 12px; background: white; border-radius: 6px;">
                        <p style="margin: 0 0 4px; font-size: 12px; color: #6b7280;">Average Sentiment Score</p>
                        <p style="margin: 0; font-size: 24px; font-weight: 700; color: {sentiment_color};">
                            {avg_sentiment:.1f}/5 - {sentiment_label}
                        </p>
                    </div>
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        {sentiment_rows if sentiment_rows else '<tr><td colspan="3" style="padding: 16px; text-align: center; color: #9ca3af;">No sentiment data</td></tr>'}
                    </table>
                </div>

                <!-- SECTION: Sources -->
                <div style="margin-bottom: 32px; padding: 20px; background-color: #fafafa; border-radius: 8px;">
                    <div style="display: flex; align-items: center; margin-bottom: 16px;">
                        <div style="width: 32px; height: 32px; background-color: #0ea5e9; border-radius: 6px; display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                            <span style="color: white; font-size: 16px;">🔗</span>
                        </div>
                        <h2 style="margin: 0; font-size: 16px; font-weight: 600; color: #111827;">
                            Top Sources Cited
                        </h2>
                        <span style="margin-left: auto; font-size: 12px; color: #6b7280;">{total_sources} total citations</span>
                    </div>
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        {source_rows if source_rows else '<tr><td colspan="2" style="padding: 16px; text-align: center; color: #9ca3af;">No sources cited</td></tr>'}
                    </table>
                </div>

                <!-- SECTION: Competitors -->
                <div style="margin-bottom: 32px; padding: 20px; background-color: #fafafa; border-radius: 8px;">
                    <div style="display: flex; align-items: center; margin-bottom: 16px;">
                        <div style="width: 32px; height: 32px; background-color: #f59e0b; border-radius: 6px; display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                            <span style="color: white; font-size: 16px;">🏢</span>
                        </div>
                        <h2 style="margin: 0; font-size: 16px; font-weight: 600; color: #111827;">
                            Competitor Mentions
                        </h2>
                    </div>
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        {competitor_rows if competitor_rows else '<tr><td colspan="2" style="padding: 16px; text-align: center; color: #9ca3af;">No competitors mentioned</td></tr>'}
                    </table>
                </div>

                <!-- SECTION: Key Insights -->
                <div style="margin-bottom: 24px; padding: 20px; background: linear-gradient(135deg, #fef3c7 0%, #fef9c3 100%); border-radius: 8px; border: 1px solid #fde68a;">
                    <div style="display: flex; align-items: center; margin-bottom: 16px;">
                        <div style="width: 32px; height: 32px; background-color: #f59e0b; border-radius: 6px; display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                            <span style="color: white; font-size: 16px;">💡</span>
                        </div>
                        <h2 style="margin: 0; font-size: 16px; font-weight: 600; color: #111827;">
                            Key Insights
                        </h2>
                    </div>
                    <ul style="margin: 0; padding: 0 0 0 20px; font-size: 14px;">
                        {insight_items if insight_items else '<li style="color: #9ca3af;">No insights available</li>'}
                    </ul>
                </div>

                <!-- CTA Button -->
                <div style="text-align: center; margin-top: 32px;">
                    <a href="{results_url}"
                       style="display: inline-block; padding: 14px 32px; background-color: #4A7C59; color: white; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 15px;">
                        View Full Report →
                    </a>
                    <p style="margin: 12px 0 0; font-size: 12px; color: #9ca3af;">
                        See detailed responses, all sources, and more
                    </p>
                </div>
            </div>

            <!-- Footer -->
            <div style="padding: 20px 24px; background-color: #f9fafb; border-top: 1px solid #e5e7eb; text-align: center;">
                <p style="margin: 0; font-size: 12px; color: #6b7280;">
                    This is a {frequency} automated report from AI Visibility Tracker.
                </p>
                <p style="margin: 8px 0 0; font-size: 12px; color: #9ca3af;">
                    You're receiving this because you set up automated reporting.
                </p>
            </div>
        </div>
    </div>
</body>
</html>
        """


# Global email service instance
email_service = EmailService()
