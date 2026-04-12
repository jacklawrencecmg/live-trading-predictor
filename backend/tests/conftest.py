import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# ---------------------------------------------------------------------------
# Stub out redis before any app module is imported so that test environments
# without the redis package can still load the full app import graph.
# ---------------------------------------------------------------------------
_redis_stub = types.ModuleType("redis")
_redis_async_stub = types.ModuleType("redis.asyncio")


class _FakeRedis:
    """Minimal async Redis stand-in for tests that don't touch Redis."""
    async def get(self, key):
        return None
    async def set(self, key, value, ex=None):
        pass
    async def close(self):
        pass


_redis_async_stub.Redis = _FakeRedis
_redis_async_stub.from_url = MagicMock(return_value=_FakeRedis())
_redis_stub.asyncio = _redis_async_stub
sys.modules.setdefault("redis", _redis_stub)
sys.modules.setdefault("redis.asyncio", _redis_async_stub)

# ---------------------------------------------------------------------------
# Stub out asyncpg so that app.core.database can be imported without a live
# PostgreSQL connection. The engine is created at module level; stubbing the
# dialect package prevents ModuleNotFoundError during collection.
# ---------------------------------------------------------------------------
_asyncpg_stub = types.ModuleType("asyncpg")
sys.modules.setdefault("asyncpg", _asyncpg_stub)

from app.core.database import Base
import app.models  # noqa: F401 — registers all models with Base.metadata

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
