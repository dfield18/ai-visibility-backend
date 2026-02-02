"""User service for managing users in the database."""

from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import ClerkUser
from app.models.user import User


async def get_user_by_clerk_id(
    db: AsyncSession,
    clerk_user_id: str,
) -> Optional[User]:
    """Get a user by their Clerk user ID.

    Args:
        db: Database session
        clerk_user_id: The Clerk user ID (from JWT sub claim)

    Returns:
        User if found, None otherwise
    """
    # We store Clerk user ID in the auth_accounts table
    from app.models.auth_account import AuthAccount

    result = await db.execute(
        select(User)
        .join(AuthAccount)
        .where(AuthAccount.provider_account_id == clerk_user_id)
    )
    return result.scalar_one_or_none()


async def get_user_by_email(
    db: AsyncSession,
    email: str,
) -> Optional[User]:
    """Get a user by their email address.

    Args:
        db: Database session
        email: User's email address

    Returns:
        User if found, None otherwise
    """
    result = await db.execute(
        select(User).where(User.email == email)
    )
    return result.scalar_one_or_none()


async def get_user_by_id(
    db: AsyncSession,
    user_id: UUID,
) -> Optional[User]:
    """Get a user by their database ID.

    Args:
        db: Database session
        user_id: User's database UUID

    Returns:
        User if found, None otherwise
    """
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def get_or_create_user_from_clerk(
    db: AsyncSession,
    clerk_user: ClerkUser,
) -> User:
    """Get or create a user from Clerk authentication.

    If the user already exists (by Clerk ID or email), returns the existing user.
    Otherwise, creates a new user and links the Clerk account.

    Args:
        db: Database session
        clerk_user: Authenticated user from Clerk JWT

    Returns:
        The user (existing or newly created)
    """
    from app.models.auth_account import AuthAccount

    # First, try to find by Clerk ID
    user = await get_user_by_clerk_id(db, clerk_user.user_id)
    if user:
        return user

    # Try to find by email
    if clerk_user.email:
        user = await get_user_by_email(db, clerk_user.email)
        if user:
            # Link the Clerk account to existing user
            auth_account = AuthAccount(
                user_id=user.id,
                provider="clerk",
                provider_account_id=clerk_user.user_id,
            )
            db.add(auth_account)
            await db.flush()
            return user

    # Create new user
    user = User(
        email=clerk_user.email or f"{clerk_user.user_id}@clerk.user",
        name=clerk_user.full_name,
        email_verified=clerk_user.email is not None,
    )
    db.add(user)
    await db.flush()

    # Create auth account link
    auth_account = AuthAccount(
        user_id=user.id,
        provider="clerk",
        provider_account_id=clerk_user.user_id,
    )
    db.add(auth_account)
    await db.flush()

    return user


async def update_user_stripe_customer_id(
    db: AsyncSession,
    user_id: UUID,
    stripe_customer_id: str,
) -> User:
    """Update a user's Stripe customer ID.

    Args:
        db: Database session
        user_id: User's database UUID
        stripe_customer_id: Stripe customer ID

    Returns:
        Updated user

    Raises:
        ValueError: If user not found
    """
    user = await get_user_by_id(db, user_id)
    if not user:
        raise ValueError(f"User not found: {user_id}")

    user.stripe_customer_id = stripe_customer_id
    await db.flush()
    return user
