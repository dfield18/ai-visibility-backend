#!/bin/bash
set -e

echo "=== Start script beginning ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "DATABASE_URL is set: $(if [ -n "$DATABASE_URL" ]; then echo 'yes'; else echo 'no'; fi)"
echo "DATABASE_URL prefix: ${DATABASE_URL:0:15}..."

echo ""
echo "=== Checking alembic setup ==="
ls -la alembic/versions/ || echo "Warning: Could not list alembic versions"

echo ""
echo "=== Running database migrations ==="
python -m alembic current || echo "No current revision (fresh database)"

# Try to run migrations, but don't fail if they error
if python -m alembic upgrade head; then
    echo "=== Migration complete ==="
    python -m alembic current
else
    echo "=== Migration failed, applying schema fixes directly ==="
    # Fallback: Apply missing columns directly via psycopg
    python << 'PYEOF'
import os
import psycopg2

db_url = os.environ.get("DATABASE_URL", "")
# Convert Railway's postgres:// to postgresql:// for psycopg2
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
# Remove asyncpg if present
db_url = db_url.replace("+asyncpg", "")

print(f"Connecting to database...")
try:
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cur = conn.cursor()

    # Check and add brand_sentiment column
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'results' AND column_name = 'brand_sentiment'
    """)
    if not cur.fetchone():
        print("Adding brand_sentiment column...")
        cur.execute("ALTER TABLE results ADD COLUMN brand_sentiment VARCHAR(50)")
        print("Added brand_sentiment column")
    else:
        print("brand_sentiment column already exists")

    # Check and add competitor_sentiments column
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'results' AND column_name = 'competitor_sentiments'
    """)
    if not cur.fetchone():
        print("Adding competitor_sentiments column...")
        cur.execute("ALTER TABLE results ADD COLUMN competitor_sentiments JSONB")
        print("Added competitor_sentiments column")
    else:
        print("competitor_sentiments column already exists")

    # Check and add all_brands_mentioned column
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'results' AND column_name = 'all_brands_mentioned'
    """)
    if not cur.fetchone():
        print("Adding all_brands_mentioned column...")
        cur.execute("ALTER TABLE results ADD COLUMN all_brands_mentioned JSONB")
        print("Added all_brands_mentioned column")
    else:
        print("all_brands_mentioned column already exists")

    # Check and add parent_run_id column to runs table
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'runs' AND column_name = 'parent_run_id'
    """)
    if not cur.fetchone():
        print("Adding parent_run_id column to runs...")
        cur.execute("ALTER TABLE runs ADD COLUMN parent_run_id UUID REFERENCES runs(id) ON DELETE SET NULL")
        cur.execute("CREATE INDEX ix_runs_parent_run_id ON runs (parent_run_id)")
        print("Added parent_run_id column")
    else:
        print("parent_run_id column already exists")

    # Check and create cached_suggestions table
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_name = 'cached_suggestions'
    """)
    if not cur.fetchone():
        print("Creating cached_suggestions table...")
        cur.execute("""
            CREATE TABLE cached_suggestions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                brand VARCHAR(255) NOT NULL,
                search_type VARCHAR(50) NOT NULL DEFAULT 'brand',
                industry VARCHAR(255),
                prompts JSONB NOT NULL DEFAULT '[]',
                competitors JSONB NOT NULL DEFAULT '[]',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                expires_at TIMESTAMPTZ NOT NULL
            )
        """)
        cur.execute("CREATE INDEX ix_cached_suggestions_lookup ON cached_suggestions (brand, search_type, industry)")
        cur.execute("CREATE INDEX ix_cached_suggestions_expires_at ON cached_suggestions (expires_at)")
        print("Created cached_suggestions table")
    else:
        print("cached_suggestions table already exists")

    cur.close()
    conn.close()
    print("Schema fixes applied successfully")
except Exception as e:
    print(f"Error applying schema fixes: {e}")
    # Don't fail startup - the app might still work
PYEOF
fi

echo ""
echo "=== Starting server ==="
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
