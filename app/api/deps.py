"""FastAPI dependencies for route handlers."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

# Type alias for database session dependency
DatabaseDep = Annotated[AsyncSession, Depends(get_db)]


# Additional dependencies can be added here as needed
# Examples:
# - get_current_user
# - get_api_key
# - rate_limiter
