"""
US equity market session calendar.

Provides trading-day detection, market-hours classification, and
bar-closed determination for NYSE/NASDAQ-scheduled instruments.

No external calendar library is required — NYSE holidays are computed
algorithmically for any year.  All public functions accept datetime objects;
timezone-aware inputs are converted to ET for session boundary comparisons.

Usage
-----
from app.data_ingestion.session_calendar import (
    is_bar_closed, is_within_session, compute_available_at,
    TIMEFRAME_SECONDS, BAR_CLOSE_BUFFER_S,
)
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

# ── Constants ─────────────────────────────────────────────────────────────────

ET  = ZoneInfo("America/New_York")
UTC = timezone.utc

# NYSE regular session (Eastern Time)
_SESSION_OPEN  = time(9, 30)
_SESSION_CLOSE = time(16, 0)

# Pre-market / after-hours window accepted when include_extended_hours=True
_PREMARKET_OPEN   = time(4, 0)
_AFTERHOURS_CLOSE = time(20, 0)

#: Seconds per timeframe string.  Add new entries here when new timeframes
#: are supported; every other calendar function derives from this dict.
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m":  60,
    "5m":  300,
    "15m": 900,
    "30m": 1_800,
    "1h":  3_600,
    "4h":  14_400,
    "1d":  86_400,
}

#: Extra seconds after nominal bar close before the bar is marked CLOSED.
#: Covers exchange dissemination lag, vendor processing, and network jitter.
BAR_CLOSE_BUFFER_S: int = 5


# ── Holiday calendar ──────────────────────────────────────────────────────────

def _easter(year: int) -> date:
    """Gregorian Easter date (Anonymous / Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month, day = divmod(h + el - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th occurrence (1-based) of ISO weekday in year/month."""
    first = date(year, month, 1)
    offset = (weekday - first.isoweekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of ISO weekday in year/month."""
    # start from the last day of the month
    last = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    offset = (last.isoweekday() - weekday) % 7
    return last - timedelta(days=offset)


def _observed(d: date) -> date:
    """Standard US federal observance: Sunday → Monday, Saturday → Friday."""
    dow = d.weekday()  # 0=Mon, 5=Sat, 6=Sun
    if dow == 6:
        return d + timedelta(days=1)
    if dow == 5:
        return d - timedelta(days=1)
    return d


def nyse_holidays(year: int) -> frozenset[date]:
    """
    Return the set of NYSE market holidays for the given calendar year.

    Covers: New Year's Day, MLK Day, Presidents Day, Good Friday,
    Memorial Day, Juneteenth (from 2022), Independence Day,
    Labor Day, Thanksgiving, Christmas Day.
    """
    easter = _easter(year)
    candidates: list[date] = [
        _observed(date(year, 1, 1)),          # New Year's Day
        _nth_weekday(year, 1, 1, 3),          # MLK Day: 3rd Mon Jan
        _nth_weekday(year, 2, 1, 3),          # Presidents Day: 3rd Mon Feb
        easter - timedelta(days=2),            # Good Friday (no substitution)
        _last_weekday(year, 5, 1),            # Memorial Day: last Mon May
        _observed(date(year, 7, 4)),           # Independence Day
        _nth_weekday(year, 9, 1, 1),          # Labor Day: 1st Mon Sep
        _nth_weekday(year, 11, 4, 4),         # Thanksgiving: 4th Thu Nov
        _observed(date(year, 12, 25)),         # Christmas Day
    ]
    # Juneteenth observed from 2022 onward
    if year >= 2022:
        candidates.append(_observed(date(year, 6, 19)))

    return frozenset(candidates)


def is_trading_day(d: date) -> bool:
    """Return True if *d* is a regular NYSE/NASDAQ trading day."""
    if d.weekday() >= 5:          # Saturday = 5, Sunday = 6
        return False
    return d not in nyse_holidays(d.year)


# ── Timeframe helpers ─────────────────────────────────────────────────────────

def bar_duration_s(timeframe: str) -> int:
    """Return the duration of *timeframe* in seconds."""
    try:
        return TIMEFRAME_SECONDS[timeframe]
    except KeyError:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}. "
            f"Supported: {sorted(TIMEFRAME_SECONDS)}"
        )


def bar_nominal_close(event_time: datetime, timeframe: str) -> datetime:
    """Return the nominal close time for a bar (event_time + duration)."""
    return event_time + timedelta(seconds=bar_duration_s(timeframe))


# ── Bar-closed determination ──────────────────────────────────────────────────

def is_bar_closed(
    event_time: datetime,
    timeframe: str,
    now_utc: datetime | None = None,
    buffer_s: int = BAR_CLOSE_BUFFER_S,
) -> bool:
    """
    Return True if the bar that opened at *event_time* is fully closed.

    A bar is considered closed when:

        bar_nominal_close(event_time, timeframe) + buffer_s <= now

    The *buffer_s* guard (default 5 s) accommodates exchange-to-vendor
    dissemination lag so that the bar's OHLCV data has had time to
    propagate before the bar is marked as final.

    Parameters
    ----------
    event_time
        Bar open time.  Timezone-aware or naive UTC both accepted.
    timeframe
        Timeframe string ('1m', '5m', etc.).
    now_utc
        Reference time.  Defaults to ``datetime.utcnow()`` (naive UTC).
        Pass an explicit value in tests to get deterministic results.
    buffer_s
        Guard buffer in seconds.  Default 5.  Must be >= 0.
    """
    if now_utc is None:
        now_utc = datetime.utcnow()

    # Normalise both sides to naive UTC for comparison
    def _naive(dt: datetime) -> datetime:
        if dt.tzinfo is not None:
            return dt.astimezone(UTC).replace(tzinfo=None)
        return dt

    close_naive = _naive(bar_nominal_close(event_time, timeframe))
    now_naive   = _naive(now_utc)

    # Strict: close + buffer must be strictly <= now
    return close_naive + timedelta(seconds=buffer_s) <= now_naive


# ── Session / trading-hours classification ────────────────────────────────────

def is_within_session(
    event_time: datetime,
    *,
    include_extended_hours: bool = False,
) -> bool:
    """
    Return True if *event_time* falls within an NYSE trading session.

    For intraday timeframes: the bar open must be within session hours on a
    trading day.  Daily bars only require the date to be a trading day.

    Parameters
    ----------
    event_time
        Bar open time (UTC-aware or naive UTC).
    include_extended_hours
        If True, pre-market (04:00–09:30 ET) and after-hours
        (16:00–20:00 ET) bars are accepted in addition to regular session.
    """
    # Convert to ET for boundary comparison
    if event_time.tzinfo is None:
        # Assume naive = UTC
        et_dt = event_time.replace(tzinfo=UTC).astimezone(ET)
    else:
        et_dt = event_time.astimezone(ET)

    d = et_dt.date()
    if not is_trading_day(d):
        return False

    t = et_dt.time()
    if include_extended_hours:
        return _PREMARKET_OPEN <= t < _AFTERHOURS_CLOSE
    return _SESSION_OPEN <= t < _SESSION_CLOSE


# ── available_at computation ──────────────────────────────────────────────────

def compute_available_at(
    event_time: datetime,
    timeframe: str,
    typical_delay_s: int,
) -> datetime:
    """
    Estimate when bar data became available to downstream consumers.

    Formula::

        available_at = event_time + bar_duration + typical_delay_s

    For yfinance (typical_delay_s = 900): a 5-min bar that opened at
    09:30 ET becomes available at 09:30 + 5m + 15m = 09:50 ET.

    This estimate is preferable to ``ingested_at`` for historical backfills
    where data may be fetched months after the bar closed.

    Returns a **naive UTC** datetime to match the existing ORM convention.
    """
    duration = bar_duration_s(timeframe)
    # Strip timezone for storage consistency (model stores naive UTC)
    base = event_time.replace(tzinfo=None) if event_time.tzinfo else event_time
    return base + timedelta(seconds=duration + typical_delay_s)
