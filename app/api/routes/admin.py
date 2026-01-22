"""Admin routes for database management."""

from fastapi import APIRouter, HTTPException

from app.core.database import Base, engine

router = APIRouter()


@router.post("/reset-database")
async def reset_database():
    """Reset the database by dropping and recreating all tables.

    WARNING: This will delete all data!

    Returns:
        dict: Success message.
    """
    try:
        async with engine.begin() as conn:
            # Drop all tables
            await conn.run_sync(Base.metadata.drop_all)
            # Recreate all tables
            await conn.run_sync(Base.metadata.create_all)

        return {"message": "Database reset successfully. All tables recreated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {str(e)}")
