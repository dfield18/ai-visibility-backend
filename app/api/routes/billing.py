"""Billing endpoints for Stripe webhook handling and subscription status."""

import stripe
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from app.core.config import settings
from app.core.database import DatabaseSession
from app.core.auth import get_current_user, ClerkUser

router = APIRouter()

# Initialize Stripe
stripe.api_key = settings.STRIPE_API_KEY if hasattr(settings, 'STRIPE_API_KEY') else ""


class BillingStatusResponse(BaseModel):
    hasSubscription: bool = False
    reportsUsed: int = 0
    reportsLimit: int = 1
    subscriptionStatus: str = "none"
    currentPeriodEnd: Optional[str] = None


@router.get("/billing/status", response_model=BillingStatusResponse)
async def get_billing_status(
    db: DatabaseSession,
    user: ClerkUser = None,
) -> BillingStatusResponse:
    """Get billing status for the current user.

    Returns subscription status and usage information.
    Falls back to free tier if Stripe is not configured.
    """
    if not user:
        return BillingStatusResponse()

    if not stripe.api_key:
        return BillingStatusResponse()

    try:
        # Search for Stripe customer by clerk user ID
        customers = stripe.Customer.search(
            query=f'metadata["clerk_user_id"]:"{user.user_id}"',
        )

        if not customers.data:
            return BillingStatusResponse()

        customer = customers.data[0]

        # Check active subscriptions
        subscriptions = stripe.Subscription.list(
            customer=customer.id,
            status='active',
            limit=1,
        )

        if subscriptions.data:
            sub = subscriptions.data[0]
            return BillingStatusResponse(
                hasSubscription=True,
                reportsUsed=0,  # TODO: Track from database
                reportsLimit=999,
                subscriptionStatus=sub.status,
                currentPeriodEnd=str(sub.current_period_end),
            )

        # Check for past_due subscriptions
        past_due_subs = stripe.Subscription.list(
            customer=customer.id,
            status='past_due',
            limit=1,
        )

        if past_due_subs.data:
            sub = past_due_subs.data[0]
            return BillingStatusResponse(
                hasSubscription=True,
                reportsUsed=0,
                reportsLimit=999,
                subscriptionStatus='past_due',
                currentPeriodEnd=str(sub.current_period_end),
            )

        return BillingStatusResponse()

    except Exception as e:
        print(f"[Billing] Error fetching status: {e}")
        return BillingStatusResponse()


@router.post("/billing/webhook")
async def stripe_webhook(request: Request, db: DatabaseSession):
    """Handle Stripe webhook events.

    Processes subscription lifecycle events to keep the database in sync.
    """
    if not stripe.api_key:
        raise HTTPException(status_code=503, detail="Stripe not configured")

    webhook_secret = getattr(settings, 'STRIPE_WEBHOOK_SECRET', '')
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature', '')

    try:
        if webhook_secret:
            event = stripe.Webhook.construct_event(
                payload, sig_header, webhook_secret
            )
        else:
            import json
            event = stripe.Event.construct_from(
                json.loads(payload), stripe.api_key
            )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event.type
    print(f"[Billing] Webhook received: {event_type}")

    if event_type == 'checkout.session.completed':
        session = event.data.object
        clerk_user_id = session.get('metadata', {}).get('clerk_user_id') or session.get('client_reference_id')
        customer_id = session.get('customer')

        if clerk_user_id and customer_id:
            # Update customer metadata with clerk user ID
            try:
                stripe.Customer.modify(
                    customer_id,
                    metadata={'clerk_user_id': clerk_user_id},
                )
                print(f"[Billing] Linked customer {customer_id} to user {clerk_user_id}")
            except Exception as e:
                print(f"[Billing] Failed to update customer metadata: {e}")

    elif event_type in [
        'customer.subscription.created',
        'customer.subscription.updated',
        'customer.subscription.deleted',
    ]:
        subscription = event.data.object
        customer_id = subscription.get('customer')
        status = subscription.get('status')
        print(f"[Billing] Subscription {subscription.get('id')} status: {status}")
        # TODO: Update subscription record in database

    elif event_type == 'invoice.payment_succeeded':
        invoice = event.data.object
        print(f"[Billing] Payment succeeded for invoice {invoice.get('id')}")

    elif event_type == 'invoice.payment_failed':
        invoice = event.data.object
        print(f"[Billing] Payment failed for invoice {invoice.get('id')}")

    return {"status": "ok"}
