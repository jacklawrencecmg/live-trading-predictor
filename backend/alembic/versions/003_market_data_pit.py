"""Add point-in-time market data schema

Revision ID: 003_market_data_pit
Revises: 002_governance
Create Date: 2026-04-12

New tables
----------
market_data_sources   Source registry (yfinance, polygon, demo, …)
bar_ingest_batches    Ingest session grouping for atomic rollback
market_bars           Bi-temporal OHLCV with revision chain
bar_corrections       Immutable correction ledger
option_quotes         Bi-temporal options quotes with revision chain
research_snapshots    Named PIT snapshot definitions for reproducible research

Columns added to existing tables
---------------------------------
inference_events.data_available_at   TIMESTAMPTZ — latest available_at of input data
inference_events.outcome_quality     VARCHAR(16) — 'final'|'preliminary'|'corrected'
model_versions.went_live_at          TIMESTAMPTZ — when model status → 'active'
model_versions.went_offline_at       TIMESTAMPTZ — when model status left 'active'

Design notes
------------
* All new timestamp columns use TIMESTAMP WITH TIME ZONE (timezone=True).
* market_bars and option_quotes carry both is_current and superseded_at so that
  point-in-time queries can be expressed without self-joins:

      -- Current state:
      WHERE is_current = TRUE

      -- DB state as of wall-clock time T_wall:
      WHERE ingested_at <= :T_wall
        AND (is_current OR superseded_at > :T_wall)

* Partial indexes on is_current trim B-tree scan cost for the hot path.
  These use postgresql_where and are no-ops on non-Postgres engines.

* market_data_sources is seeded with four built-in sources in the upgrade()
  function so that FKs resolve immediately after migration.
"""

from alembic import op
import sqlalchemy as sa

revision = "003_market_data_pit"
down_revision = "002_governance"
branch_labels = None
depends_on = None

# ── Helper: TIMESTAMP WITH TIME ZONE ─────────────────────────────────────────
_tstz = sa.TIMESTAMP(timezone=True)


def upgrade() -> None:
    # =========================================================================
    # market_data_sources
    # =========================================================================
    op.create_table(
        "market_data_sources",
        sa.Column("source_id",       sa.String(32), primary_key=True),
        sa.Column("display_name",    sa.String(128), nullable=False),
        sa.Column("typical_delay_s", sa.Integer, nullable=False, server_default="0"),
        sa.Column("max_staleness_s", sa.Integer, nullable=False, server_default="900"),
        sa.Column("is_real_time",    sa.Boolean, nullable=False, server_default="false"),
        sa.Column("notes",           sa.Text, nullable=True),
        sa.Column("created_at",      _tstz, nullable=False,
                  server_default=sa.func.now()),
    )

    # Seed built-in sources — add new providers here as they are integrated.
    op.execute("""
        INSERT INTO market_data_sources
            (source_id, display_name, typical_delay_s, max_staleness_s, is_real_time, notes)
        VALUES
            ('yfinance',
             'Yahoo Finance (yfinance)',
             900,
             1800,
             false,
             '15-minute delayed. available_at = event_time + bar_duration + 900s. '
             'Historical available_at is estimated; not exact.'),
            ('polygon',
             'Polygon.io',
             0,
             60,
             false,
             'Real-time via websocket; REST historical has zero nominal delay. '
             'Requires paid subscription for options data.'),
            ('alpaca',
             'Alpaca Markets Data',
             0,
             60,
             true,
             'Real-time websocket feed. available_at = event_time for live bars.'),
            ('demo',
             'Synthetic demo data',
             0,
             86400,
             false,
             'Seeded synthetic data for local development. No real market information.')
        ON CONFLICT (source_id) DO NOTHING;
    """)

    # =========================================================================
    # bar_ingest_batches
    # =========================================================================
    op.create_table(
        "bar_ingest_batches",
        sa.Column("id",             sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("source_id",      sa.String(32),
                  sa.ForeignKey("market_data_sources.source_id"), nullable=False),
        sa.Column("symbol",         sa.String(20), nullable=True),
        sa.Column("timeframe",      sa.String(10), nullable=True),
        sa.Column("started_at",     _tstz, nullable=False,
                  server_default=sa.func.now()),
        sa.Column("completed_at",   _tstz, nullable=True),
        sa.Column("rows_written",   sa.Integer, nullable=True),
        sa.Column("rows_skipped",   sa.Integer, nullable=False, server_default="0"),
        sa.Column("rows_corrected", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status",         sa.String(16), nullable=False,
                  server_default="running"),
        sa.Column("error_detail",   sa.Text, nullable=True),
    )
    op.create_index("ix_bib_symbol",         "bar_ingest_batches", ["symbol"])
    op.create_index("ix_bib_source_started", "bar_ingest_batches",
                    ["source_id", "started_at"])
    op.create_index("ix_bib_status",         "bar_ingest_batches", ["status"])

    # =========================================================================
    # market_bars
    # =========================================================================
    op.create_table(
        "market_bars",
        # Identity
        sa.Column("id",              sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("symbol",          sa.String(20), nullable=False),
        sa.Column("timeframe",       sa.String(10), nullable=False),

        # Tri-temporal
        sa.Column("event_time",      _tstz, nullable=False),
        sa.Column("available_at",    _tstz, nullable=True),
        sa.Column("ingested_at",     _tstz, nullable=False,
                  server_default=sa.func.now()),

        # Provenance
        sa.Column("source_id",       sa.String(32),
                  sa.ForeignKey("market_data_sources.source_id"), nullable=False),
        sa.Column("ingest_batch_id", sa.BigInteger,
                  sa.ForeignKey("bar_ingest_batches.id"), nullable=True),

        # Bar lifecycle
        sa.Column("bar_status",      sa.String(16), nullable=False,
                  server_default="CLOSED"),

        # OHLCV
        sa.Column("open",            sa.Float, nullable=False),
        sa.Column("high",            sa.Float, nullable=False),
        sa.Column("low",             sa.Float, nullable=False),
        sa.Column("close",           sa.Float, nullable=False),
        sa.Column("volume",          sa.Float, nullable=False),
        sa.Column("vwap",            sa.Float, nullable=True),
        sa.Column("trade_count",     sa.Integer, nullable=True),

        # Staleness
        sa.Column("staleness_s",     sa.Float, nullable=True),

        # Adjustments
        sa.Column("split_factor",    sa.Float, nullable=False, server_default="1.0"),
        sa.Column("div_factor",      sa.Float, nullable=False, server_default="1.0"),
        sa.Column("is_adjusted",     sa.Boolean, nullable=False,
                  server_default="false"),

        # Revision chain
        sa.Column("revision_seq",    sa.SmallInteger, nullable=False,
                  server_default="1"),
        sa.Column("is_current",      sa.Boolean, nullable=False,
                  server_default="true"),
        sa.Column("superseded_at",   _tstz, nullable=True),
        sa.Column("superseded_by",   sa.BigInteger,
                  sa.ForeignKey("market_bars.id"), nullable=True),

        sa.UniqueConstraint(
            "symbol", "timeframe", "event_time", "source_id", "revision_seq",
            name="uq_market_bar_revision",
        ),
    )

    # Standard indexes
    op.create_index("ix_mbar_symbol_ingested",  "market_bars",
                    ["symbol", "ingested_at"])
    op.create_index("ix_mbar_source_event",     "market_bars",
                    ["source_id", "event_time"])
    op.create_index("ix_mbar_batch",            "market_bars", ["ingest_batch_id"])

    # Partial indexes (PostgreSQL) — hot-path queries on current rows only.
    op.execute("""
        CREATE INDEX ix_mbar_symbol_tf_event_cur
            ON market_bars (symbol, timeframe, event_time)
            WHERE is_current = TRUE;
    """)
    op.execute("""
        CREATE INDEX ix_mbar_symbol_tf_avail_cur
            ON market_bars (symbol, timeframe, available_at)
            WHERE is_current = TRUE;
    """)

    # =========================================================================
    # bar_corrections
    # =========================================================================
    op.create_table(
        "bar_corrections",
        sa.Column("id",                  sa.BigInteger, primary_key=True,
                  autoincrement=True),
        sa.Column("original_bar_id",     sa.BigInteger,
                  sa.ForeignKey("market_bars.id"), nullable=False),
        sa.Column("replacement_bar_id",  sa.BigInteger,
                  sa.ForeignKey("market_bars.id"), nullable=True),
        sa.Column("corrected_at",        _tstz, nullable=False,
                  server_default=sa.func.now()),
        sa.Column("correction_type",     sa.String(32), nullable=False),
        sa.Column("initiated_by",        sa.String(64), nullable=True),
        sa.Column("reason",              sa.Text, nullable=True),
        sa.Column("changed_fields_json", sa.Text, nullable=True),
    )
    op.create_index("ix_bcorr_original",    "bar_corrections", ["original_bar_id"])
    op.create_index("ix_bcorr_replacement", "bar_corrections", ["replacement_bar_id"])
    op.create_index("ix_bcorr_corrected",   "bar_corrections", ["corrected_at"])

    # =========================================================================
    # option_quotes
    # =========================================================================
    op.create_table(
        "option_quotes",
        # Identity
        sa.Column("id",                 sa.BigInteger, primary_key=True,
                  autoincrement=True),
        sa.Column("underlying_symbol",  sa.String(20), nullable=False),
        sa.Column("option_symbol",      sa.String(50), nullable=True),
        sa.Column("expiry",             sa.String(12), nullable=False),
        sa.Column("strike",             sa.Float, nullable=False),
        sa.Column("option_type",        sa.String(4), nullable=False),
        sa.Column("dte",                sa.SmallInteger, nullable=True),

        # Tri-temporal
        sa.Column("event_time",         _tstz, nullable=True),
        sa.Column("available_at",       _tstz, nullable=False),
        sa.Column("ingested_at",        _tstz, nullable=False,
                  server_default=sa.func.now()),

        # Provenance
        sa.Column("source_id",          sa.String(32),
                  sa.ForeignKey("market_data_sources.source_id"), nullable=False),
        sa.Column("ingest_batch_id",    sa.BigInteger,
                  sa.ForeignKey("bar_ingest_batches.id"), nullable=True),

        # Quote data
        sa.Column("underlying_price",   sa.Float, nullable=True),
        sa.Column("bid",                sa.Float, nullable=True),
        sa.Column("ask",                sa.Float, nullable=True),
        sa.Column("last",               sa.Float, nullable=True),
        sa.Column("volume",             sa.Integer, nullable=True),
        sa.Column("open_interest",      sa.Integer, nullable=True),

        # Implied vol and Greeks
        sa.Column("implied_volatility", sa.Float, nullable=True),
        sa.Column("delta",              sa.Float, nullable=True),
        sa.Column("gamma",              sa.Float, nullable=True),
        sa.Column("theta",              sa.Float, nullable=True),
        sa.Column("vega",               sa.Float, nullable=True),
        sa.Column("rho",                sa.Float, nullable=True),

        # Chain aggregates
        sa.Column("iv_rank",            sa.Float, nullable=True),
        sa.Column("iv_skew",            sa.Float, nullable=True),
        sa.Column("pc_volume_ratio",    sa.Float, nullable=True),
        sa.Column("pc_oi_ratio",        sa.Float, nullable=True),
        sa.Column("gamma_exposure",     sa.Float, nullable=True),

        # Data quality
        sa.Column("spread_pct",         sa.Float, nullable=True),
        sa.Column("is_stale",           sa.Boolean, nullable=False,
                  server_default="false"),
        sa.Column("staleness_s",        sa.Float, nullable=True),
        sa.Column("is_illiquid",        sa.Boolean, nullable=False,
                  server_default="false"),

        # Revision chain
        sa.Column("revision_seq",       sa.SmallInteger, nullable=False,
                  server_default="1"),
        sa.Column("is_current",         sa.Boolean, nullable=False,
                  server_default="true"),
        sa.Column("superseded_at",      _tstz, nullable=True),
        sa.Column("superseded_by",      sa.BigInteger,
                  sa.ForeignKey("option_quotes.id"), nullable=True),

        sa.UniqueConstraint(
            "underlying_symbol", "expiry", "strike", "option_type",
            "available_at", "source_id", "revision_seq",
            name="uq_option_quote_revision",
        ),
    )

    op.create_index("ix_oq_underlying_expiry_avail", "option_quotes",
                    ["underlying_symbol", "expiry", "available_at"])
    op.create_index("ix_oq_source_avail",            "option_quotes",
                    ["source_id", "available_at"])
    op.create_index("ix_oq_batch",                   "option_quotes",
                    ["ingest_batch_id"])

    # Partial index for the L7 join hot path.
    op.execute("""
        CREATE INDEX ix_oq_underlying_avail_cur
            ON option_quotes (underlying_symbol, available_at)
            WHERE is_current = TRUE;
    """)

    # =========================================================================
    # research_snapshots
    # =========================================================================
    op.create_table(
        "research_snapshots",
        sa.Column("id",                 sa.Integer, primary_key=True,
                  autoincrement=True),
        sa.Column("name",               sa.String(128), nullable=False,
                  unique=True),
        sa.Column("description",        sa.Text, nullable=True),
        sa.Column("as_of_time",         _tstz, nullable=False),
        sa.Column("symbols_json",       sa.Text, nullable=False),
        sa.Column("timeframes_json",    sa.Text, nullable=True),
        sa.Column("sources_json",       sa.Text, nullable=True),
        sa.Column("bar_count",          sa.Integer, nullable=True),
        sa.Column("option_quote_count", sa.Integer, nullable=True),
        sa.Column("created_at",         _tstz, nullable=False,
                  server_default=sa.func.now()),
        sa.Column("created_by",         sa.String(64), nullable=False,
                  server_default="system"),
        sa.Column("is_locked",          sa.Boolean, nullable=False,
                  server_default="false"),
    )
    op.create_index("ix_rsnap_as_of", "research_snapshots", ["as_of_time"])

    # =========================================================================
    # Extend inference_events
    # =========================================================================
    op.add_column(
        "inference_events",
        sa.Column("data_available_at", _tstz, nullable=True),
        # The available_at of the most recent data used for this inference.
        # Enables post-hoc check: was inference based on stale or correct data?
        # Populated by InferenceLogService when logging results.
    )
    op.add_column(
        "inference_events",
        sa.Column("outcome_quality", sa.String(16), nullable=True),
        # 'final'        — outcome is settled (bar closed and confirmed)
        # 'preliminary'  — outcome recorded before confirmation (e.g. after-hours)
        # 'corrected'    — outcome was back-filled then subsequently revised
        # NULL           — outcome not yet recorded
    )
    op.create_index(
        "ix_inf_data_avail", "inference_events", ["data_available_at"]
    )

    # =========================================================================
    # Extend model_versions
    # =========================================================================
    op.add_column(
        "model_versions",
        sa.Column("went_live_at", _tstz, nullable=True),
        # Timestamp when this model's status transitioned to 'active'.
        # Point-in-time query: which model was live at time T?
        #   WHERE went_live_at <= :T
        #     AND (went_offline_at IS NULL OR went_offline_at > :T)
    )
    op.add_column(
        "model_versions",
        sa.Column("went_offline_at", _tstz, nullable=True),
        # Timestamp when this model's status left 'active'.
    )
    op.create_index(
        "ix_mv_went_live", "model_versions", ["model_name", "went_live_at"]
    )


def downgrade() -> None:
    # Extend rollbacks first (columns on existing tables)
    op.drop_index("ix_mv_went_live",     table_name="model_versions")
    op.drop_column("model_versions",     "went_offline_at")
    op.drop_column("model_versions",     "went_live_at")

    op.drop_index("ix_inf_data_avail",   table_name="inference_events")
    op.drop_column("inference_events",   "outcome_quality")
    op.drop_column("inference_events",   "data_available_at")

    # Drop tables in reverse FK dependency order
    op.drop_table("research_snapshots")

    op.execute("DROP INDEX IF EXISTS ix_oq_underlying_avail_cur")
    op.drop_index("ix_oq_batch",                   table_name="option_quotes")
    op.drop_index("ix_oq_source_avail",            table_name="option_quotes")
    op.drop_index("ix_oq_underlying_expiry_avail", table_name="option_quotes")
    op.drop_table("option_quotes")

    op.drop_index("ix_bcorr_corrected",   table_name="bar_corrections")
    op.drop_index("ix_bcorr_replacement", table_name="bar_corrections")
    op.drop_index("ix_bcorr_original",    table_name="bar_corrections")
    op.drop_table("bar_corrections")

    op.execute("DROP INDEX IF EXISTS ix_mbar_symbol_tf_avail_cur")
    op.execute("DROP INDEX IF EXISTS ix_mbar_symbol_tf_event_cur")
    op.drop_index("ix_mbar_batch",           table_name="market_bars")
    op.drop_index("ix_mbar_source_event",    table_name="market_bars")
    op.drop_index("ix_mbar_symbol_ingested", table_name="market_bars")
    op.drop_table("market_bars")

    op.drop_index("ix_bib_status",         table_name="bar_ingest_batches")
    op.drop_index("ix_bib_source_started", table_name="bar_ingest_batches")
    op.drop_index("ix_bib_symbol",         table_name="bar_ingest_batches")
    op.drop_table("bar_ingest_batches")

    op.drop_table("market_data_sources")
