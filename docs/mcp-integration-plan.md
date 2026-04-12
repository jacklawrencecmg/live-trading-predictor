# MCP Integration Plan — options-research

## 1. Current State

Every external system is called directly from application code with no
abstraction layer:

| Call site | External system | Abstraction |
|-----------|-----------------|-------------|
| `market_data.py:55` | yfinance OHLCV | none — direct `yf.Ticker().history()` |
| `market_data.py:76` | yfinance quote | none — direct `yf.Ticker().fast_info` |
| `options_service.py:118` | yfinance chain | none — direct `yf.Ticker().option_chain()` |
| `ingestion_service.py:166` | yfinance backfill | none — lazy import in function body |
| `backtest_service.py:134` | yfinance backtest | none — lazy import in function body |
| `governance/alerts.py:89` | alert delivery | DB-only — no outbound channel |
| `paper_trader.py:21` | broker execution | DB/Redis — no real-fill adapter |

**Problems this creates:**
- Swapping data providers requires editing 5 files instead of 1
- Tests either call yfinance (slow, flaky, network) or mock 5 separate call sites
- No way to add a second data source (e.g., real-time WebSocket feed) without
  rewriting all consumers
- Governance alerts never reach operators — they exist only in the DB

---

## 2. Decision Framework

Two separate concerns, often confused:

```
┌──────────────────────────────────────────────────────────────────┐
│  PROVIDER INTERFACES                                             │
│  What the application code talks to at RUNTIME                   │
│  Goal: vendor-agnostic, testable, injectable                     │
│                                                                  │
│  RUNTIME                          TEST                           │
│  YFinanceMarketDataProvider  ←→   NullMarketDataProvider         │
│  YFinanceOptionsProvider     ←→   NullOptionsProvider            │
│  PaperBrokerProvider         ←→   NullBrokerProvider             │
│  WebhookAlertChannel         ←→   NullAlertChannel               │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  MCP SERVERS                                                     │
│  What CLAUDE CODE talks to during DEVELOPMENT                    │
│  Goal: give Claude live project state for debugging              │
│                                                                  │
│  claude code  →  options-research MCP  →  DB + Redis             │
│  claude code  →  github MCP            →  GitHub Issues/PRs      │
│  claude code  →  fetch MCP             →  broker/provider docs   │
└──────────────────────────────────────────────────────────────────┘
```

**Use a provider interface when:**
- The system is on the runtime path (data fetching, order submission, alerts)
- You expect to swap vendors (yfinance → Polygon, paper → Alpaca)
- You need a no-network stub for unit testing

**Use an MCP server when:**
- Claude needs to query live state during a coding or debugging session
- The information is in an external system (GitHub, browser docs)
- You want Claude to be able to inspect the running application

---

## 3. Provider Interface Architecture

Four protocols defined in `backend/app/providers/protocols.py`:

### 3.1 MarketDataProvider

```
MarketDataProvider (Protocol)
  ├── get_candles(symbol, interval, period) → pd.DataFrame
  ├── get_quote(symbol) → Quote
  └── stream_quotes(symbol, interval_seconds) → AsyncIterator[Quote]

Implementations:
  YFinanceMarketDataProvider   ← current default   (yfinance_market.py)
  PolygonMarketDataProvider    ← production path   (not yet written)
  AlpacaMarketDataProvider     ← alternative       (not yet written)
  NullMarketDataProvider       ← tests             (null_providers.py)
```

**Migration target:** Replace the 5 direct yfinance call sites with a single
injected `MarketDataProvider`. The FastAPI lifespan creates one instance and
stores it in `app.state.market_data`. Tests inject `NullMarketDataProvider`.

### 3.2 OptionsChainProvider

```
OptionsChainProvider (Protocol)
  ├── get_expirations(symbol) → List[str]
  ├── get_chain(symbol, expiry) → OptionsChainSnapshot
  └── get_atm_iv(symbol) → float

Implementations:
  YFinanceOptionsProvider          ← current default  (yfinance_options.py)
  TDAmeritradeOptionsProvider      ← not yet written
  IBKROptionsProvider              ← not yet written
  NullOptionsProvider              ← tests            (null_providers.py)
```

**Note:** yfinance options data is delayed (15 min typical) and computes greeks
locally via Black-Scholes. Production use requires a paid options feed with
real-time greeks.

### 3.3 BrokerProvider

```
BrokerProvider (Protocol)
  ├── is_paper: bool  (property)
  ├── submit_order(order) → OrderResult
  ├── cancel_order(order_id) → bool
  ├── get_positions() → List[dict]
  └── get_account() → AccountInfo

Implementations:
  PaperBrokerProvider    ← current (DB + Redis)  (paper_broker.py)
  AlpacaBrokerProvider   ← sandbox + live        (not yet written)
  IBKRBrokerProvider     ← not yet written
  NullBrokerProvider     ← tests                 (null_providers.py)
```

**Critical constraint:** The risk gate (`check_all_risks`) is NOT inside
`BrokerProvider`. It remains in `risk_manager.py` and is called before
`submit_order`. Broker implementations are pure execution adapters.

### 3.4 AlertChannel

```
AlertChannel (Protocol)
  └── dispatch(alert: AlertPayload) → None  # must never raise

Implementations:
  NullAlertChannel      ← dev default        (alert_channels.py)
  WebhookAlertChannel   ← generic HTTP POST  (alert_channels.py)
  SlackAlertChannel     ← Slack webhook      (alert_channels.py)
  PagerDutyAlertChannel ← critical-only      (not yet written)
```

**Registration:** Channels are registered at startup via
`build_alert_channels_from_env()`. Multiple channels can be active
simultaneously. A channel failure must never affect the trading path.

---

## 4. MCP Server Architecture

### 4.1 options-research (custom, in-repo)

**File:** `backend/app/mcp/server.py`
**Transport:** stdio (launched by Claude Code as a subprocess)
**Purpose:** Expose live application state to Claude Code during development

| Tool | What it returns | When to use |
|------|-----------------|-------------|
| `get_risk_summary` | capital, P&L, kill switch, limits | Debugging trade execution |
| `get_governance_alerts` | DB alerts by severity | Investigating model degradation |
| `get_latest_inference` | last inference bundle for a symbol | Verifying signal pipeline |
| `get_market_quote` | live quote via configured provider | Confirming data feed is alive |
| `get_options_chain` | chain summary (nearest 5 strikes) | Checking chain quality |
| `get_system_health` | aggregated health snapshot | First call in any incident |
| `get_active_positions` | open paper positions | Verifying execution |

**Security:** stdio only — never exposed on a network port. DB credentials
come from the same `.env` file as the application. No additional credentials
needed.

### 4.2 github (standard MCP server)

**Package:** `@modelcontextprotocol/server-github`
**Credential:** `GITHUB_TOKEN` environment variable (Personal Access Token)
**Purpose:** Claude can read and create issues, review PRs, and link code
changes to work items.

**Workflow examples:**
- When fixing a bug, Claude can find the related GitHub issue and add a
  comment with the fix summary
- When implementing a feature, Claude can read the issue for acceptance criteria
- Claude can create issues for medium/low audit findings

### 4.3 fetch (standard MCP server)

**Package:** `@modelcontextprotocol/server-fetch`
**Credentials:** None
**Purpose:** Claude can fetch external documentation during development

**Workflow examples:**
- Fetch Alpaca API docs when writing `AlpacaBrokerProvider`
- Fetch Polygon API docs when writing `PolygonMarketDataProvider`
- Fetch PagerDuty Events API v2 spec when writing `PagerDutyAlertChannel`

---

## 5. Security Model

### Credential isolation

```
Runtime credentials (in .env, never in code):
  DATABASE_URL            — PostgreSQL connection
  REDIS_URL               — Redis connection
  ALERT_WEBHOOK_URL       — outbound alert webhook
  ALERT_SLACK_WEBHOOK_URL — Slack incoming webhook
  ALERT_WEBHOOK_SECRET    — webhook HMAC secret

MCP credentials (in shell environment, not .env):
  GITHUB_TOKEN            — GitHub PAT (read-only for dev, write for issue creation)

Provider credentials (in .env when provider requires auth):
  POLYGON_API_KEY         — when switching from yfinance to Polygon
  ALPACA_API_KEY          — for AlpacaBrokerProvider
  ALPACA_API_SECRET       — for AlpacaBrokerProvider
  ALPACA_BASE_URL         — "https://paper-api.alpaca.markets" for sandbox
  IBKR_PORT               — for IBKRBrokerProvider
```

### Separation of concerns

- The options-research MCP server is **read-only** (no trading path calls)
- Provider implementations are **instantiated at startup** — no credential
  access in hot paths
- `BrokerProvider.is_paper` must return `True` in any test or CI context;
  any live broker adapter must check `ALPACA_BASE_URL` to confirm sandbox mode

### CORS

`main.py` currently allows all origins (`allow_origins=["*"]`). Before any
live broker integration, this must be restricted to the frontend origin only.
The MCP server does not use HTTP — stdio only.

---

## 6. Testability

### Before provider interfaces (current state)

Tests that call any inference, decision, or backtest code implicitly call
yfinance. CI failures are often network failures, not code failures.

### After provider interfaces

```python
# conftest.py
@pytest.fixture
def market_provider():
    return NullMarketDataProvider(base_price=450.0, seed=42)

@pytest.fixture
def options_provider():
    return NullOptionsProvider(base_price=450.0, atm_iv=0.18)

@pytest.fixture
def broker_provider():
    return NullBrokerProvider()

# test_inference.py
async def test_inference_returns_bundle(market_provider, options_provider):
    df = await market_provider.get_candles("SPY", "5m", "5d")
    result = run_inference(df, "SPY", options_features=None)
    assert result.action in ("buy", "sell", "abstain")
    assert 0.0 <= result.tradeable_confidence <= 1.0
```

No network calls. Tests run in <1s. Deterministic results (fixed seed).

### Broker rejection testing

```python
async def test_risk_violation_blocks_order():
    broker = NullBrokerProvider(reject_symbols=["AAPL"])
    with pytest.raises(BrokerError):
        await broker.submit_order(OrderRequest(symbol="AAPL", side="buy", quantity=1))
```

---

## 7. Migration Path

Phased migration to avoid breaking changes:

### Phase 1 (now) — introduce interfaces, don't break existing code

- ✅ Provider protocols defined (`protocols.py`)
- ✅ YFinance implementations written (wrap existing code exactly)
- ✅ Null implementations written (for tests)
- ✅ MCP server written (`mcp/server.py`)
- ✅ MCP servers configured (`.claude/settings.json`)
- ⬜ Add `mcp` to requirements.txt ← done

### Phase 2 — inject providers at startup

Wire providers into `app.state` in `main.py`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.market_data = YFinanceMarketDataProvider()
    app.state.options = YFinanceOptionsProvider()
    app.state.broker = PaperBrokerProvider()
    for ch in build_alert_channels_from_env():
        GovernanceAlertService.register_channel(ch)
    yield
```

Add `Depends()` accessors:
```python
def get_market_provider(request: Request) -> MarketDataProvider:
    return request.app.state.market_data
```

Refactor `market_data.py`, `options_service.py`, and `ingestion_service.py`
to accept an injected `MarketDataProvider` instead of importing yfinance
directly. The existing function signatures become thin wrappers.

### Phase 3 — swap providers as needed

To switch market data to Polygon:
1. Write `PolygonMarketDataProvider(api_key=settings.polygon_api_key)`
2. Change one line in `lifespan()`: `app.state.market_data = PolygonMarketDataProvider(...)`
3. No other changes required

To connect a real broker sandbox:
1. Write `AlpacaBrokerProvider(api_key, api_secret, base_url)`
2. Change one line in `lifespan()`: `app.state.broker = AlpacaBrokerProvider(...)`
3. Ensure `is_paper` returns `True` when `base_url` is the paper endpoint

---

## 8. Vendor Shortlist

When ready to replace yfinance, evaluate these vendors:

### Market data
| Vendor | Free tier | Latency | Options | Notes |
|--------|-----------|---------|---------|-------|
| yfinance | yes | 15m delayed | yes | current; no SLA |
| Polygon.io | limited | real-time | yes | good REST + WebSocket |
| Alpaca Data v2 | yes (limited) | real-time | no | bundled with broker |
| Tradier | yes | real-time | yes | also has broker API |
| CBOE LiveVol | no | real-time | excellent | expensive |

### Broker sandbox
| Vendor | Paper API | Live API | Options | Notes |
|--------|-----------|----------|---------|-------|
| Alpaca | yes | yes | no (equities only) | simple REST |
| Tastytrade | yes | yes | yes | options-native |
| IBKR | yes (TWS) | yes | yes | most complete; complex |
| TD Ameritrade/Schwab | yes | yes | yes | institutional quality |

### Alerting
| Vendor | Free tier | Latency | Notes |
|--------|-----------|---------|-------|
| Webhook (generic) | yes | ms | works with Zapier/Make |
| Slack incoming webhook | yes | ms | recommended for dev teams |
| PagerDuty | limited | ms | critical-only escalation |
| Datadog | limited | ms | bundles monitoring + alerts |

---

## 9. What Was NOT MCP-Connected (and Why)

| System | Decision | Reason |
|--------|----------|--------|
| PostgreSQL | Provider interface, not MCP | DB is runtime dependency, not developer tool |
| Redis | Provider interface, not MCP | Same; also the options-research MCP reads Redis indirectly |
| Market data feed | Provider interface | Runtime; Null stub sufficient for tests |
| Paper broker | Provider interface | Runtime; MCP gives read-only inspection |
| ML model | No abstraction yet | Model is in-process; model serving (MLflow) is a future concern |
| Feature store | No abstraction yet | Features computed on-demand; Phase 4 |
| Monitoring (Datadog/Grafana) | MCP when needed | Not yet running; add when there is a deployed environment to monitor |
