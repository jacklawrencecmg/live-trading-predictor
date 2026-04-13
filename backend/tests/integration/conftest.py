"""
Integration test fixtures.

Integration tests require live services:
  - PostgreSQL (DATABASE_URL env var must point to a real database)
  - Redis (REDIS_URL env var must point to a real Redis instance)

In CI, these tests are skipped unless the `INTEGRATION_TESTS=1` environment
variable is set. This prevents them from running in the unit-test fast path.

To run integration tests locally:

    cd backend
    INTEGRATION_TESTS=1 pytest tests/integration/ -v

Or with docker-compose services running:

    docker-compose up -d postgres redis
    INTEGRATION_TESTS=1 pytest tests/integration/ -v
"""

import asyncio
import os

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ── Guard: skip all integration tests unless explicitly opted in ──────────────
INTEGRATION_ENABLED = os.environ.get("INTEGRATION_TESTS", "").strip() in ("1", "true", "yes")

if not INTEGRATION_ENABLED:
    collect_ignore_glob = ["test_*.py"]  # noqa: F841 — tells pytest to skip this dir


def pytest_collection_modifyitems(config, items):
    """Skip all integration tests if INTEGRATION_TESTS is not set."""
    if not INTEGRATION_ENABLED:
        skip_marker = pytest.mark.skip(
            reason="Integration tests disabled. Set INTEGRATION_TESTS=1 to enable."
        )
        for item in items:
            if "integration" in str(item.fspath):
                item.add_marker(skip_marker)


# ── Database URL (real PostgreSQL, not SQLite) ────────────────────────────────
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/options_research_test",
)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session():
    """Live PostgreSQL session. Rolls back after each test."""
    engine = create_async_engine(DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        from app.core.database import Base
        import app.models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)

    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        yield session
        await session.rollback()

    await engine.dispose()


@pytest.fixture(scope="session")
def redis_client():
    """Live Redis client for integration tests."""
    import redis.asyncio as aioredis
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/1")  # DB 1 = test
    return aioredis.from_url(redis_url)
