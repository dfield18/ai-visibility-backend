"""Clerk JWT authentication for FastAPI."""

from typing import Annotated, Optional

import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.core.config import settings


class ClerkUser(BaseModel):
    """Authenticated user from Clerk JWT."""

    user_id: str
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None

    @property
    def full_name(self) -> Optional[str]:
        """Get user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name or self.last_name


# Cache for Clerk's JWKS (JSON Web Key Set)
_jwks_cache: dict = {}


async def get_clerk_jwks() -> dict:
    """Fetch Clerk's public keys for JWT verification.

    Returns cached keys if available, otherwise fetches from Clerk.
    """
    global _jwks_cache

    if _jwks_cache:
        return _jwks_cache

    # Fetch JWKS from Clerk
    jwks_url = f"https://{settings.CLERK_DOMAIN}/.well-known/jwks.json"

    async with httpx.AsyncClient() as client:
        response = await client.get(jwks_url)
        response.raise_for_status()
        _jwks_cache = response.json()

    return _jwks_cache


def get_public_key(token: str, jwks: dict) -> str:
    """Get the public key for verifying the JWT.

    Args:
        token: The JWT token
        jwks: The JSON Web Key Set

    Returns:
        The public key in PEM format
    """
    # Get the key ID from the token header
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")

    if not kid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing key ID",
        )

    # Find the matching key
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return jwt.algorithms.RSAAlgorithm.from_jwk(key)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unable to find matching key",
    )


security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
) -> ClerkUser:
    """Validate Clerk JWT and return the authenticated user.

    Args:
        credentials: Bearer token from Authorization header

    Returns:
        ClerkUser with user information

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        # Get Clerk's public keys
        jwks = await get_clerk_jwks()
        public_key = get_public_key(token, jwks)

        # Verify and decode the token
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={
                "verify_aud": False,  # Clerk doesn't use standard audience
                "verify_iss": True,
            },
            issuer=f"https://{settings.CLERK_DOMAIN}",
        )

        # Extract user information
        return ClerkUser(
            user_id=payload.get("sub"),
            email=payload.get("email"),
            first_name=payload.get("first_name"),
            last_name=payload.get("last_name"),
        )

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
) -> Optional[ClerkUser]:
    """Get the current user if authenticated, otherwise return None.

    Use this for routes that work with or without authentication.
    """
    if not credentials:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


# Type aliases for dependency injection
CurrentUser = Annotated[ClerkUser, Depends(get_current_user)]
OptionalUser = Annotated[Optional[ClerkUser], Depends(get_optional_user)]
