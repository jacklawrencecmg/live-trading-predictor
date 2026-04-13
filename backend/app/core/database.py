from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

# Pool tuning — suitable for a research platform running <10 concurrent requests.
# Increase pool_size / max_overflow for higher-concurrency deployments.
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,   # discard stale connections before use
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,    # recycle connections every 30 min
)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    """FastAPI dependency — yields a transactional async session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
