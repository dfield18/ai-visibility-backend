"""User profile routes."""

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import CurrentUser
from app.core.config import settings
from app.core.database import get_db
from app.services.user_service import get_or_create_user_from_clerk

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/sync-profile")
async def sync_profile(
    clerk_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Sync user profile from Clerk, including email address.

    This endpoint fetches the user's full profile from Clerk's Backend API
    and updates the local database with the correct email address.
    """
    if not settings.CLERK_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CLERK_SECRET_KEY not configured",
        )

    # Fetch user details from Clerk Backend API
    clerk_api_url = f"https://api.clerk.com/v1/users/{clerk_user.user_id}"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            clerk_api_url,
            headers={
                "Authorization": f"Bearer {settings.CLERK_SECRET_KEY}",
                "Content-Type": "application/json",
            },
        )

        if response.status_code != 200:
            print(f"[SyncProfile] Clerk API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to fetch user from Clerk: {response.status_code}",
            )

        clerk_data = response.json()

    # Extract primary email
    email_addresses = clerk_data.get("email_addresses", [])
    primary_email_id = clerk_data.get("primary_email_address_id")

    primary_email = None
    for email_obj in email_addresses:
        if email_obj.get("id") == primary_email_id:
            primary_email = email_obj.get("email_address")
            break

    # Fallback to first email if no primary
    if not primary_email and email_addresses:
        primary_email = email_addresses[0].get("email_address")

    if not primary_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No email address found in Clerk profile",
        )

    # Get or create user and update email
    user = await get_or_create_user_from_clerk(db, clerk_user)

    old_email = user.email
    user.email = primary_email
    user.name = f"{clerk_data.get('first_name', '')} {clerk_data.get('last_name', '')}".strip() or user.name
    user.email_verified = True

    await db.commit()

    print(f"[SyncProfile] Updated user {user.id}: email {old_email} -> {primary_email}")

    return {
        "success": True,
        "message": "Profile synced successfully",
        "email": primary_email,
        "name": user.name,
    }
