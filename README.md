# AI Visibility Backend

A FastAPI backend for tracking brand visibility across AI language models. This service allows you to monitor how AI assistants recommend and mention brands in response to user queries.

## Tech Stack

- **FastAPI** - Modern, fast web framework for building APIs
- **PostgreSQL 15** - Robust relational database with asyncpg driver
- **SQLAlchemy 2.0** - Async ORM with modern Python type hints
- **Alembic** - Database migration tool
- **Pydantic** - Data validation using Python type annotations
- **Docker** - Containerization for local development

## Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose
- Git

## Local Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-visibility-backend
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ai_visibility
OPENAI_API_KEY=sk-proj-your-key-here
GEMINI_API_KEY=your-gemini-key-here
ENVIRONMENT=development
DEBUG=True
CORS_ORIGINS=["http://localhost:3000"]
```

### 5. Start PostgreSQL with Docker

```bash
docker-compose up -d
```

This starts:
- PostgreSQL on port 5432
- pgAdmin on port 5050 (optional, access at http://localhost:5050)
  - Email: admin@admin.com
  - Password: admin

### 6. Run Database Migrations

```bash
alembic upgrade head
```

### 7. Start the Development Server

```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

## API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/api/v1/health` | GET | Health check with database status |
| `/api/v1/health/live` | GET | Kubernetes liveness probe |
| `/api/v1/health/ready` | GET | Kubernetes readiness probe |

## Database Migrations

### Create a New Migration

```bash
alembic revision --autogenerate -m "Description of changes"
```

### Apply Migrations

```bash
alembic upgrade head
```

### Rollback Migration

```bash
alembic downgrade -1
```

### View Migration History

```bash
alembic history
```

## Project Structure

```
ai-visibility-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Pydantic Settings configuration
│   │   └── database.py      # Async SQLAlchemy setup
│   ├── models/              # SQLAlchemy ORM models
│   │   ├── __init__.py
│   │   ├── session.py       # User session tracking
│   │   ├── run.py           # Visibility check runs
│   │   └── result.py        # Individual API call results
│   ├── schemas/             # Pydantic request/response schemas
│   │   └── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py          # FastAPI dependencies
│   │   └── routes/          # API route handlers
│   │       ├── __init__.py
│   │       └── health.py    # Health check endpoints
│   └── services/            # Business logic layer
│       └── __init__.py
├── alembic/
│   ├── versions/            # Migration scripts
│   ├── env.py               # Alembic environment config
│   └── script.py.mako       # Migration template
├── tests/
│   └── __init__.py
├── alembic.ini              # Alembic configuration
├── docker-compose.yml       # Local development services
├── Dockerfile               # Production container build
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
└── README.md
```

## Database Schema

### Sessions Table
Tracks user sessions with spending limits.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| session_id | VARCHAR(255) | Client session identifier |
| total_spent | DECIMAL(10,2) | Total API spending |
| created_at | TIMESTAMP | Creation time |
| expires_at | TIMESTAMP | Expiration time (7 days) |

### Runs Table
Tracks visibility check batches.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| session_id | UUID | Foreign key to sessions |
| status | VARCHAR(20) | queued/running/complete/failed/cancelled |
| brand | VARCHAR(255) | Brand being tracked |
| config | JSON | Run configuration |
| total_calls | INTEGER | Total API calls planned |
| completed_calls | INTEGER | Successful calls |
| failed_calls | INTEGER | Failed calls |
| estimated_cost | DECIMAL(10,4) | Pre-run cost estimate |
| actual_cost | DECIMAL(10,4) | Post-run actual cost |
| cancelled | BOOLEAN | Cancellation flag |
| created_at | TIMESTAMP | Creation time |
| completed_at | TIMESTAMP | Completion time |

### Results Table
Stores individual API call results.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| run_id | UUID | Foreign key to runs |
| prompt | TEXT | Query sent to AI |
| provider | VARCHAR(50) | openai/gemini |
| model | VARCHAR(100) | Specific model used |
| temperature | DECIMAL(3,2) | Temperature setting |
| repeat_index | INTEGER | Repeat iteration |
| response_text | TEXT | AI response |
| error | TEXT | Error message if failed |
| brand_mentioned | BOOLEAN | Brand visibility |
| competitors_mentioned | JSON | Array of competitors |
| response_type | VARCHAR(20) | list/prose/unknown |
| tokens | INTEGER | Tokens used |
| cost | DECIMAL(10,4) | Call cost |
| created_at | TIMESTAMP | Creation time |

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
# Install dev dependencies
pip install black isort

# Format code
black app tests
isort app tests
```

### Type Checking

```bash
pip install mypy
mypy app
```

## Next Steps

This is Phase 1 - the foundation. Future phases will add:

1. **Session Management API** - Create/manage user sessions
2. **Run Management API** - Start/cancel/monitor visibility runs
3. **AI Provider Integration** - OpenAI and Gemini API calls
4. **Results Analysis API** - Query and analyze results
5. **Background Tasks** - Async job processing with Celery/ARQ
6. **Rate Limiting** - Protect against abuse
7. **Authentication** - API key or JWT auth

## Troubleshooting

### Database Connection Failed

1. Ensure Docker is running: `docker ps`
2. Check PostgreSQL container: `docker-compose logs postgres`
3. Verify DATABASE_URL in `.env`

### Migration Errors

1. Check if database exists: `docker-compose exec postgres psql -U postgres -l`
2. Reset migrations: `alembic downgrade base && alembic upgrade head`

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

## License

MIT
