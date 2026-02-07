"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api import router
from app.core.config import settings
from app.core.database import AsyncSessionLocal


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown events.

    Performs database connection test on startup, starts scheduler,
    and handles cleanup on shutdown.

    Args:
        app: FastAPI application instance.

    Yields:
        None: Control passes to the application.
    """
    # Startup
    print("Starting AI Visibility Backend...")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug mode: {settings.DEBUG}")

    # Test database connection and verify schema
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            print("Database connection successful!")

            # Verify parent_run_id column exists
            result = await session.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'runs' AND column_name = 'parent_run_id'
            """))
            if not result.fetchone():
                print("Adding parent_run_id column to runs table...")
                await session.execute(text(
                    "ALTER TABLE runs ADD COLUMN parent_run_id UUID REFERENCES runs(id) ON DELETE SET NULL"
                ))
                await session.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_runs_parent_run_id ON runs (parent_run_id)"
                ))
                await session.commit()
                print("Added parent_run_id column successfully!")
            else:
                print("parent_run_id column already exists")
    except Exception as e:
        print(f"Warning: Database setup issue: {e}")
        print("The application will start but some operations may fail.")

    # Start scheduler service for automated reports
    try:
        from app.services.scheduler import scheduler_service
        await scheduler_service.start()
    except Exception as e:
        print(f"Warning: Scheduler service failed to start: {e}")
        print("Scheduled reports will not run automatically.")

    yield

    # Shutdown
    print("Shutting down AI Visibility Backend...")

    # Stop scheduler
    try:
        from app.services.scheduler import scheduler_service
        await scheduler_service.shutdown()
    except Exception as e:
        print(f"Warning: Scheduler shutdown error: {e}")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance.
    """
    app = FastAPI(
        title="AI Visibility Backend",
        description="Backend API for AI Brand Visibility Tracker",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS - use settings for configurable origins
    cors_origins = settings.CORS_ORIGINS + [
        "https://ai-visibility-frontend-rho.vercel.app",
        "https://ai-visibility-frontend.vercel.app",
    ]
    # Remove duplicates while preserving order
    cors_origins = list(dict.fromkeys(cors_origins))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Include API routes
    app.include_router(router, prefix="/api/v1")

    # Root endpoint
    @app.get("/")
    async def root() -> dict:
        """Root endpoint with API information.

        Returns:
            dict: Basic API information and links.
        """
        return {
            "name": "AI Visibility Backend",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


# Create application instance
app = create_application()
