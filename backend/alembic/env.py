import asyncio
from logging.config import fileConfig

from sqlalchemy.ext.asyncio import create_async_engine
from alembic import context
from app.core.config import settings
from app.core.database import Base

# ── Import ALL models so Alembic can detect them in autogenerate mode ─────────
# Add any new model module here when you create it.
import app.models.trade           # noqa: F401
import app.models.position        # noqa: F401
import app.models.audit_log       # noqa: F401
import app.models.backtest        # noqa: F401
import app.models.feature_row     # noqa: F401
import app.models.option_snapshot # noqa: F401
import app.models.regime_label    # noqa: F401
# Governance models (registered separately via their own Base/mixin):
try:
    import app.governance.models  # noqa: F401
except ImportError:
    pass

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (emits SQL to stdout)."""
    url = settings.database_url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against a live DB using the async engine."""
    connectable = create_async_engine(settings.database_url)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
