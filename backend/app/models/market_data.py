"""
Point-in-time-correct market data models.

Tables
------
MarketDataSource   Source registry — providers and their delay characteristics
BarIngestBatch     Session-level grouping of ingestion runs (enables atomic rollback)
MarketBar          Bi-temporal OHLCV bars with revision chain
BarCorrection      Immutable correction ledger (split/dividend/error history)
OptionQuote        Bi-temporal options quotes with revision chain
ResearchSnapshot   Named PIT snapshots for reproducible research runs

Temporal model
--------------
Every data row carries three timestamps:

  event_time   When the market event occurred (bar open time, quote exchange ts).
               This is ground truth — it never changes even if data is corrected.

  available_at When the data became available to consumers.
               For live bars:       event_time + bar_duration + provider_delay
               For yfinance (15m):  event_time + 5m + 900s
               For historical fill: ingested_at (conservative — we didn't know
                                    about it until we fetched it)
               This is the timestamp to filter on for training set construction.

  ingested_at  Wall-clock time of the DB INSERT.
               Immutable. Used for audit and "DB state as of time T" queries.

Point-in-time queries
---------------------
Current data (live inference):
    WHERE is_current = TRUE

Data available as of market time T (correct training set boundary):
    WHERE available_at <= :T AND is_current = TRUE

DB state as of wall-clock time T_wall (audit / replay):
    WHERE ingested_at <= :T_wall
      AND (is_current OR superseded_at > :T_wall)

Lookahead guard — options join (L7)
------------------------------------
When joining option_quotes to market_bars for feature computation:

    JOIN option_quotes oq
      ON oq.underlying_symbol = mb.symbol
     AND oq.available_at <= mb.event_time          -- ← REQUIRED. Never ">".
     AND oq.is_current = TRUE

Using oq.available_at > mb.event_time leaks future options state into bar features
and is the most common source of lookahead bias in options research.

Idempotency
-----------
Unique constraints encode the idempotency key for each table:

  market_bars:   (symbol, timeframe, event_time, source_id, revision_seq)
  option_quotes: (underlying_symbol, expiry, strike, option_type,
                  available_at, source_id, revision_seq)

First write sets revision_seq=1. Retrying the same payload hits the unique constraint
and should be handled by the ingestion layer with ON CONFLICT DO NOTHING (skip).
A correction inserts revision_seq=2 and marks the old row is_current=FALSE.
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)

from app.core.database import Base


# ── Source registry ───────────────────────────────────────────────────────────

class MarketDataSource(Base):
    """
    Registry of data providers.

    One row per source.  Seeded in migration 003; add new rows as providers
    are integrated.  The source_id short code ('yfinance', 'polygon', …) is
    used as a FK in MarketBar and OptionQuote to carry full provenance.
    """

    __tablename__ = "market_data_sources"

    source_id        = Column(String(32), primary_key=True)
    # Short code: 'yfinance' | 'polygon' | 'alpaca' | 'cboe' | 'demo'

    display_name     = Column(String(128), nullable=False)
    typical_delay_s  = Column(Integer, nullable=False, default=0)
    # Typical seconds between event_time+bar_duration and when data is readable.
    # yfinance = 900 (15 min), polygon realtime = 0, polygon historical = 0.

    max_staleness_s  = Column(Integer, nullable=False, default=900)
    # Alert threshold: if age of latest data exceeds this, raise a freshness alert.

    is_real_time     = Column(Boolean, nullable=False, default=False)
    # True only if the source provides sub-second latency (e.g. websocket feed).

    notes            = Column(Text, nullable=True)
    created_at       = Column(DateTime(timezone=True), nullable=False,
                               default=datetime.utcnow)


# ── Ingest batch (session grouping) ──────────────────────────────────────────

class BarIngestBatch(Base):
    """
    Groups all rows written in a single ingestion run.

    Enables:
    - Atomic rollback: set status='rolled_back', mark all member bars
      is_current=FALSE to revert a bad batch without deleting rows.
    - Batch statistics: how many rows were written vs skipped (idempotent).
    - Provenance: which process/cron created a set of bars.
    """

    __tablename__ = "bar_ingest_batches"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    source_id       = Column(String(32),
                              ForeignKey("market_data_sources.source_id"),
                              nullable=False)
    symbol          = Column(String(20), nullable=True, index=True)
    timeframe       = Column(String(10), nullable=True)

    started_at      = Column(DateTime(timezone=True), nullable=False,
                              default=datetime.utcnow)
    completed_at    = Column(DateTime(timezone=True), nullable=True)

    rows_written    = Column(Integer, nullable=True)
    rows_skipped    = Column(Integer, nullable=False, default=0)
    # rows_skipped: rows that hit the unique constraint → already existed → no-op
    rows_corrected  = Column(Integer, nullable=False, default=0)
    # rows_corrected: rows that replaced an existing current row with a new revision

    status          = Column(String(16), nullable=False, default="running", index=True)
    # 'running' | 'completed' | 'failed' | 'rolled_back'
    error_detail    = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_bib_source_started", "source_id", "started_at"),
    )


# ── Market bars (OHLCV, bi-temporal + revision) ───────────────────────────────

class MarketBar(Base):
    """
    Point-in-time-correct OHLCV bar.

    Revision chain
    --------------
    Original ingest:  revision_seq=1, is_current=TRUE
    First correction: INSERT revision_seq=2, is_current=TRUE
                      UPDATE old row: is_current=FALSE, superseded_at=now(),
                                      superseded_by=new_row.id, bar_status='CORRECTED'
    Second correction: INSERT revision_seq=3, … (repeat)

    The BarCorrection table records every correction event with field-level diffs.

    Adjustment factors
    ------------------
    split_factor and div_factor are cumulative multipliers relative to the
    exchange-native price series.  Adjusted close = close * split_factor * div_factor.
    When is_adjusted=TRUE, the stored OHLCV values already incorporate these factors.
    """

    __tablename__ = "market_bars"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)

    # ── Identity ──────────────────────────────────────────────────────────────
    symbol          = Column(String(20), nullable=False)
    timeframe       = Column(String(10), nullable=False)
    # '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d'

    # ── Tri-temporal ──────────────────────────────────────────────────────────
    event_time      = Column(DateTime(timezone=True), nullable=False)
    # Bar open time in market time (UTC).  Never changed by corrections.

    available_at    = Column(DateTime(timezone=True), nullable=True)
    # When consumers may use this bar.
    # = event_time + bar_duration + source.typical_delay_s
    # NULL means "unknown" — treat as ingested_at for safety.

    ingested_at     = Column(DateTime(timezone=True), nullable=False,
                              default=datetime.utcnow)
    # Wall-clock time of DB write.  Immutable after INSERT.

    # ── Provenance ────────────────────────────────────────────────────────────
    source_id       = Column(String(32),
                              ForeignKey("market_data_sources.source_id"),
                              nullable=False)
    ingest_batch_id = Column(BigInteger,
                              ForeignKey("bar_ingest_batches.id"),
                              nullable=True)

    # ── Bar lifecycle ─────────────────────────────────────────────────────────
    bar_status      = Column(String(16), nullable=False, default="CLOSED")
    # PARTIAL    — bar still open (live feed)
    # CLOSED     — bar finalised, canonical state
    # CORRECTED  — this row has been superseded (is_current will be FALSE)
    # BACKFILLED — gap-filled from secondary source
    # INVALID    — vendor-flagged bad data

    # ── OHLCV ─────────────────────────────────────────────────────────────────
    open            = Column(Float, nullable=False)
    high            = Column(Float, nullable=False)
    low             = Column(Float, nullable=False)
    close           = Column(Float, nullable=False)
    volume          = Column(Float, nullable=False)
    vwap            = Column(Float, nullable=True)
    trade_count     = Column(Integer, nullable=True)

    # ── Staleness ─────────────────────────────────────────────────────────────
    staleness_s     = Column(Float, nullable=True)
    # ingested_at - available_at in seconds.
    # Positive = late (normal for historical backfill).
    # Negative = impossible unless clocks are skewed.

    # ── Price adjustments ─────────────────────────────────────────────────────
    split_factor    = Column(Float, nullable=False, default=1.0)
    div_factor      = Column(Float, nullable=False, default=1.0)
    is_adjusted     = Column(Boolean, nullable=False, default=False)

    # ── Revision chain ────────────────────────────────────────────────────────
    revision_seq    = Column(SmallInteger, nullable=False, default=1)
    # Starts at 1. Each correction increments by 1.

    is_current      = Column(Boolean, nullable=False, default=True)
    # FALSE for all rows except the latest revision.

    superseded_at   = Column(DateTime(timezone=True), nullable=True)
    # Wall-clock time when this row was replaced by a correction.

    superseded_by   = Column(BigInteger,
                              ForeignKey("market_bars.id"),
                              nullable=True)
    # FK to the replacement row. NULL if this is the current revision.

    __table_args__ = (
        UniqueConstraint(
            "symbol", "timeframe", "event_time", "source_id", "revision_seq",
            name="uq_market_bar_revision",
        ),
        # Primary read path: symbol + timeframe + time window, current rows only.
        Index(
            "ix_mbar_symbol_tf_event_cur",
            "symbol", "timeframe", "event_time",
            postgresql_where="is_current = TRUE",
        ),
        # Training set boundary query: available_at <= :T
        Index(
            "ix_mbar_symbol_tf_avail_cur",
            "symbol", "timeframe", "available_at",
            postgresql_where="is_current = TRUE",
        ),
        # Audit / wall-clock replay: ingested_at <= :T_wall
        Index("ix_mbar_symbol_ingested", "symbol", "ingested_at"),
        # Batch rollback
        Index("ix_mbar_batch", "ingest_batch_id"),
        # Source-level freshness checks
        Index("ix_mbar_source_event", "source_id", "event_time"),
    )


# ── Bar correction ledger ─────────────────────────────────────────────────────

class BarCorrection(Base):
    """
    Immutable ledger of every correction applied to a MarketBar.

    Never UPDATE or DELETE rows in this table.

    correction_type values
    ----------------------
    SPLIT_ADJUSTMENT     Post-split backward price normalisation
    DIVIDEND_ADJUSTMENT  Ex-dividend price adjustment
    DATA_ERROR           Vendor error corrected (price spike, tick error)
    LATE_DATA            Bar arrived after gap; replaces a gap-fill or absent row
    PARTIAL_TO_FINAL     Live partial bar promoted to final close
    """

    __tablename__ = "bar_corrections"

    id                  = Column(BigInteger, primary_key=True, autoincrement=True)

    original_bar_id     = Column(BigInteger,
                                  ForeignKey("market_bars.id"),
                                  nullable=False)
    replacement_bar_id  = Column(BigInteger,
                                  ForeignKey("market_bars.id"),
                                  nullable=True)
    # NULL until the replacement row has been written.

    corrected_at        = Column(DateTime(timezone=True), nullable=False,
                                  default=datetime.utcnow)
    correction_type     = Column(String(32), nullable=False)
    initiated_by        = Column(String(64), nullable=True)
    # 'auto_ingest' | 'manual' | 'vendor_restatement' | 'split_processor'
    reason              = Column(Text, nullable=True)

    changed_fields_json = Column(Text, nullable=True)
    # JSON: {"close": {"from": 100.0, "to": 99.5},
    #        "split_factor": {"from": 1.0, "to": 2.0}}

    __table_args__ = (
        Index("ix_bcorr_original",    "original_bar_id"),
        Index("ix_bcorr_replacement", "replacement_bar_id"),
        Index("ix_bcorr_corrected",   "corrected_at"),
    )


# ── Options quotes (bi-temporal + revision) ───────────────────────────────────

class OptionQuote(Base):
    """
    Point-in-time-correct options quote.

    Unlike OHLCV bars, options quotes are instantaneous snapshots of a
    single strike at a moment in time.

    available_at semantics
    ----------------------
    For real-time feeds: available_at ≈ event_time (sub-second latency).
    For yfinance historical: available_at = ingested_at (we don't know when
    the exchange published it, so we conservatively assume "now").
    For CBOE EOD data: available_at = exchange_close_time of that session.

    Lookahead guard (L7) — CRITICAL
    --------------------------------
    When joining to market_bars for feature engineering, you MUST use:

        option_quotes.available_at <= market_bars.event_time

    Never: option_quotes.available_at > market_bars.event_time
           option_quotes.event_time  > market_bars.event_time

    Either of those conditions leaks options data from after the bar opened,
    introducing lookahead bias into any features derived from that join.
    """

    __tablename__ = "option_quotes"

    id                  = Column(BigInteger, primary_key=True, autoincrement=True)

    # ── Identity ──────────────────────────────────────────────────────────────
    underlying_symbol   = Column(String(20), nullable=False)
    option_symbol       = Column(String(50), nullable=True)  # OCC symbol
    expiry              = Column(String(12), nullable=False)  # 'YYYY-MM-DD'
    strike              = Column(Float, nullable=False)
    option_type         = Column(String(4), nullable=False)  # 'call' | 'put'
    dte                 = Column(SmallInteger, nullable=True)
    # Days-to-expiry at available_at.  Computed on insert; helps range scans.

    # ── Tri-temporal ──────────────────────────────────────────────────────────
    event_time          = Column(DateTime(timezone=True), nullable=True)
    # Exchange-embedded timestamp of this quote, if known.

    available_at        = Column(DateTime(timezone=True), nullable=False)
    # When our system could query this quote.  The correct timestamp for
    # feature engineering join conditions.

    ingested_at         = Column(DateTime(timezone=True), nullable=False,
                                  default=datetime.utcnow)

    # ── Provenance ────────────────────────────────────────────────────────────
    source_id           = Column(String(32),
                                  ForeignKey("market_data_sources.source_id"),
                                  nullable=False)
    ingest_batch_id     = Column(BigInteger,
                                  ForeignKey("bar_ingest_batches.id"),
                                  nullable=True)

    # ── Quote data ────────────────────────────────────────────────────────────
    underlying_price    = Column(Float, nullable=True)
    bid                 = Column(Float, nullable=True)
    ask                 = Column(Float, nullable=True)
    last                = Column(Float, nullable=True)
    volume              = Column(Integer, nullable=True)
    open_interest       = Column(Integer, nullable=True)

    # ── Implied vol and Greeks ────────────────────────────────────────────────
    implied_volatility  = Column(Float, nullable=True)
    delta               = Column(Float, nullable=True)
    gamma               = Column(Float, nullable=True)
    theta               = Column(Float, nullable=True)
    vega                = Column(Float, nullable=True)
    rho                 = Column(Float, nullable=True)

    # ── Chain-level aggregates (snapshot-wide, not per-strike) ────────────────
    iv_rank             = Column(Float, nullable=True)
    iv_skew             = Column(Float, nullable=True)   # put_iv - call_iv
    pc_volume_ratio     = Column(Float, nullable=True)
    pc_oi_ratio         = Column(Float, nullable=True)
    gamma_exposure      = Column(Float, nullable=True)   # chain GEX proxy

    # ── Data quality flags (computed on insert) ───────────────────────────────
    spread_pct          = Column(Float, nullable=True)
    # (ask - bid) / mid.  NULL if bid or ask is missing.
    is_stale            = Column(Boolean, nullable=False, default=False)
    staleness_s         = Column(Float, nullable=True)
    # available_at - reference_bar_event_time.  Populated when joined to a bar.
    is_illiquid         = Column(Boolean, nullable=False, default=False)
    # True if bid==ask==0 or open_interest==0 at time of snapshot.

    # ── Revision chain ────────────────────────────────────────────────────────
    revision_seq        = Column(SmallInteger, nullable=False, default=1)
    is_current          = Column(Boolean, nullable=False, default=True)
    superseded_at       = Column(DateTime(timezone=True), nullable=True)
    superseded_by       = Column(BigInteger,
                                  ForeignKey("option_quotes.id"),
                                  nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "underlying_symbol", "expiry", "strike", "option_type",
            "available_at", "source_id", "revision_seq",
            name="uq_option_quote_revision",
        ),
        # Primary lookup: current quotes for a symbol at a point in time (L7 join).
        Index(
            "ix_oq_underlying_avail_cur",
            "underlying_symbol", "available_at",
            postgresql_where="is_current = TRUE",
        ),
        # Expiry-filtered queries (near-term options selection).
        Index(
            "ix_oq_underlying_expiry_avail",
            "underlying_symbol", "expiry", "available_at",
        ),
        # Staleness and freshness checks per source.
        Index("ix_oq_source_avail", "source_id", "available_at"),
        # Batch rollback.
        Index("ix_oq_batch", "ingest_batch_id"),
    )


# ── Research snapshot registry ────────────────────────────────────────────────

class ResearchSnapshot(Base):
    """
    Named point-in-time snapshot for reproducible research.

    A ResearchSnapshot records a single as_of_time that serves as the
    data availability cutoff for all tables in a research run.  Any
    training set, backtest, or published result that claims to be
    "point-in-time correct" must be associated with a ResearchSnapshot
    and apply the filter:

        WHERE available_at <= :as_of_time AND is_current = TRUE

    Locking
    -------
    Set is_locked=TRUE before publishing or committing to results.
    A locked snapshot must not be modified; create a new one instead.
    Locked snapshots are the unit of reproducibility: two users querying
    the same locked snapshot with the same SQL will get the same rows.

    Caveat
    ------
    Point-in-time correctness is only guaranteed for data ingested AFTER
    this schema was in place.  Historical backfills from yfinance have
    approximate available_at values (see DATA_LINEAGE.md §Limitations).
    """

    __tablename__ = "research_snapshots"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    name            = Column(String(128), nullable=False, unique=True)
    description     = Column(Text, nullable=True)

    as_of_time      = Column(DateTime(timezone=True), nullable=False)
    # Cutoff: all data used must have available_at <= as_of_time.

    # Scope definition
    symbols_json    = Column(Text, nullable=False)    # JSON: ["SPY", "QQQ"]
    timeframes_json = Column(Text, nullable=True)     # JSON: ["5m"]
    sources_json    = Column(Text, nullable=True)     # JSON: ["yfinance"]

    # Row counts captured at creation (for integrity verification)
    bar_count           = Column(Integer, nullable=True)
    option_quote_count  = Column(Integer, nullable=True)

    created_at      = Column(DateTime(timezone=True), nullable=False,
                              default=datetime.utcnow)
    created_by      = Column(String(64), nullable=False, default="system")
    is_locked       = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_rsnap_as_of", "as_of_time"),
    )
