"""Email service for sending report notifications using Resend."""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

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
        """Send a report completion email.

        Args:
            to_email: Recipient email address
            report: The scheduled report
            run: The completed run

        Returns:
            True if email was sent successfully, False otherwise
        """
        print(f"[Email] Preparing email for {to_email}, report: {report.name}, run: {run.id}")

        # Build summary statistics
        try:
            summary = self._build_summary(run)
        except Exception as e:
            print(f"[Email] Error building summary: {e}")
            import traceback
            traceback.print_exc()
            summary = {
                "total_responses": 0,
                "visibility_rate": 0,
                "by_provider": {},
                "top_competitors": [],
            }

        # Generate email content
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
            print(f"[Email] Summary: visibility_rate={summary.get('visibility_rate', 0):.1%}, total_responses={summary.get('total_responses', 0)}")
            return True

        try:
            # Run synchronous Resend API call in thread pool to avoid blocking
            from_email = "AI Visibility <reports@aivis.io>"
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

    def _build_summary(self, run: "Run") -> Dict:
        """Build summary statistics from a run.

        Args:
            run: The completed run

        Returns:
            Dictionary with summary statistics
        """
        results = run.results or []
        successful = [r for r in results if r.error is None and r.response_text]

        if not successful:
            return {
                "total_responses": 0,
                "visibility_rate": 0,
                "by_provider": {},
                "top_competitors": [],
            }

        # Calculate visibility rate
        mentioned_count = sum(1 for r in successful if r.brand_mentioned)
        visibility_rate = mentioned_count / len(successful) if successful else 0

        # Calculate by provider
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

        # Get top competitors
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

        return {
            "total_responses": len(successful),
            "visibility_rate": visibility_rate,
            "by_provider": by_provider,
            "top_competitors": top_competitors,
        }

    def _generate_html_email(
        self,
        report_name: str,
        brand: str,
        summary: Dict,
        run_id: str,
        frequency: str,
    ) -> str:
        """Generate HTML email content.

        Args:
            report_name: Name of the scheduled report
            brand: Brand being analyzed
            summary: Summary statistics
            run_id: The run ID for linking
            frequency: Report frequency

        Returns:
            HTML email content
        """
        visibility_pct = int(summary.get("visibility_rate", 0) * 100)

        # Format provider breakdown
        provider_rows = ""
        for provider, stats in summary.get("by_provider", {}).items():
            provider_label = {
                "openai": "ChatGPT",
                "gemini": "Google Gemini",
                "anthropic": "Claude",
                "perplexity": "Perplexity",
                "ai_overviews": "Google AI Overviews",
            }.get(provider, provider)
            rate_pct = int(stats.get("rate", 0) * 100)
            provider_rows += f"""
                <tr>
                    <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb;">{provider_label}</td>
                    <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb; text-align: right;">{rate_pct}%</td>
                    <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb; text-align: right; color: #6b7280;">
                        {stats.get('mentioned', 0)}/{stats.get('total', 0)}
                    </td>
                </tr>
            """

        # Format competitor mentions
        competitor_items = ""
        for comp, count in summary.get("top_competitors", []):
            competitor_items += f'<li style="margin-bottom: 4px;">{comp}: {count} mentions</li>'

        if not competitor_items:
            competitor_items = '<li style="color: #6b7280;">No competitor mentions detected</li>'

        # Get frontend URL
        frontend_url = settings.CORS_ORIGINS[0] if settings.CORS_ORIGINS else "http://localhost:3000"
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
                    {report_name}
                </p>
            </div>

            <!-- Main Content -->
            <div style="padding: 24px;">
                <!-- Visibility Score -->
                <div style="text-align: center; margin-bottom: 24px; padding: 20px; background-color: #f9fafb; border-radius: 8px;">
                    <p style="margin: 0 0 8px; color: #6b7280; font-size: 14px;">Overall Visibility</p>
                    <p style="margin: 0; font-size: 48px; font-weight: 700; color: #4A7C59;">
                        {visibility_pct}%
                    </p>
                    <p style="margin: 8px 0 0; color: #6b7280; font-size: 12px;">
                        {brand} mentioned in {summary.get('total_responses', 0)} AI responses
                    </p>
                </div>

                <!-- Provider Breakdown -->
                <div style="margin-bottom: 24px;">
                    <h2 style="margin: 0 0 12px; font-size: 16px; font-weight: 600; color: #111827;">
                        Visibility by AI Platform
                    </h2>
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        <thead>
                            <tr style="border-bottom: 2px solid #e5e7eb;">
                                <th style="padding: 8px 0; text-align: left; font-weight: 500; color: #6b7280;">Platform</th>
                                <th style="padding: 8px 0; text-align: right; font-weight: 500; color: #6b7280;">Rate</th>
                                <th style="padding: 8px 0; text-align: right; font-weight: 500; color: #6b7280;">Mentions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {provider_rows}
                        </tbody>
                    </table>
                </div>

                <!-- Competitor Mentions -->
                <div style="margin-bottom: 24px;">
                    <h2 style="margin: 0 0 12px; font-size: 16px; font-weight: 600; color: #111827;">
                        Top Competitor Mentions
                    </h2>
                    <ul style="margin: 0; padding: 0 0 0 20px; font-size: 14px; color: #374151;">
                        {competitor_items}
                    </ul>
                </div>

                <!-- CTA Button -->
                <div style="text-align: center; margin-top: 32px;">
                    <a href="{results_url}"
                       style="display: inline-block; padding: 12px 24px; background-color: #4A7C59; color: white; text-decoration: none; border-radius: 8px; font-weight: 500; font-size: 14px;">
                        View Full Report
                    </a>
                </div>
            </div>

            <!-- Footer -->
            <div style="padding: 16px 24px; background-color: #f9fafb; border-top: 1px solid #e5e7eb; text-align: center;">
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
