#!/bin/bash
set -e

echo "=== Start script beginning ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "DATABASE_URL is set: $(if [ -n "$DATABASE_URL" ]; then echo 'yes'; else echo 'no'; fi)"
echo "DATABASE_URL prefix: ${DATABASE_URL:0:15}..."

echo ""
echo "=== Checking alembic setup ==="
ls -la alembic/versions/

echo ""
echo "=== Running database migrations ==="
alembic current || echo "No current revision (fresh database)"
alembic upgrade head
echo "=== Migration complete ==="
alembic current

echo ""
echo "=== Starting server ==="
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
