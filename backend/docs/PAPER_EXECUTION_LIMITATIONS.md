# Paper Execution Limitations

**Scope:** This document describes the known limitations of the paper-trading
options simulator (`app/paper_trading/options_simulator/`) that affect the
fidelity of simulated executions relative to real-market fills. Read this
before interpreting any P&L results from paper simulation.

The companion document `simulator_limitations.md` covers backtesting methodology
limitations (data quality, lookback bias, etc.). This document focuses specifically
on **execution realism**.

---

## 1. Fill Price Assumptions

### 1.1 No Order Book Depth
The simulator has no knowledge of bid/ask size, market depth, or queue position.
Fill methods (MIDPOINT, MIDPOINT_PLUS_SLIPPAGE, BID_ASK, CONSERVATIVE) apply a
single price to the entire order regardless of size. In practice:

- Large orders (multiple contracts) consume depth and receive price degradation
  beyond the quoted spread.
- Even small orders in illiquid names may not fill at mid; the quoted bid/ask
  may not be executable if the size is 1 contract at that level.

**Consequence:** Paper fills are systematically optimistic for multi-contract
orders in names with thin option chains.

### 1.2 Simultaneous Multi-Leg Fills Are Not Real
Spread structures (debit spreads, credit spreads) simulate both legs filling at
the same instant. Real spread orders are subject to legging risk: the first leg
may fill while the second moves against you. Contingent/combo orders reduce but
do not eliminate this risk.

**Consequence:** Spread fills in simulation are better than achievable in
practice, particularly in fast-moving markets or during catalyst events.

### 1.3 Mid-Price Is Often Not Achievable
MIDPOINT fill mode assumes fills at exactly (bid + ask) / 2. In practice:

- Market makers widen spreads dynamically; the "mid" at order submission may
  differ from the mid at fill time.
- For options priced below $1.00, the minimum tick ($0.01 or $0.05) constrains
  achievable mid fills.
- During high-volatility periods, spreads widen and mid fills become less reliable.

MIDPOINT_PLUS_SLIPPAGE (fill at mid ± slippage_factor × spread) is more realistic
for most use cases. CONSERVATIVE (fill outside bid/ask) models thin or illiquid
conditions.

**Note from CLAUDE.md NI:** The default fill method must not be MIDPOINT in
production simulation contexts. Use MIDPOINT_PLUS_SLIPPAGE or BID_ASK as defaults.

### 1.4 No Partial Fills
Every order either fills completely or not at all. Partial fills do not occur in
simulation. In practice, especially for multi-contract orders or illiquid names,
partial fills are common and change the realized position size.

### 1.5 Fill Timing Within the Bar
The simulator processes fills at the moment `update_positions()` is called. It
does not model intrabar price variation. A limit order that would have been filled
intrabar at a favorable price is not distinguished from one that fills at the bar
close price. This means:

- Good-till-cancel (GTC) and intraday limit orders are not supported.
- All fills are treated as market orders at the prevailing quote.

---

## 2. Options Pricing and Greeks

### 2.1 No Greeks Tracking
The simulator tracks P&L based on quoted prices only. It does not maintain a
live Greeks surface. As a result:

- Delta-equivalent exposure is not tracked dynamically.
- Gamma risk near expiry is not quantified.
- Vega risk from IV expansion/contraction is captured only through quoted prices,
  not through a model.

### 2.2 No Volatility Surface
There is no term structure, skew, or smile model. The simulator uses whatever
IV is passed with the quote (if any). It does not model:

- Skew steepening after a large move (put skew typically increases post-drop).
- Term structure flattening approaching expiry.
- Volatility crush after earnings or events.

### 2.3 Volatility Crush Is Not Simulated
Buying options into an event (earnings, FOMC) and then experiencing IV crush is
a major source of real P&L drag that is not modeled unless the input quotes
explicitly reflect the crushed IV. If backtesting through historical events,
ensure your historical option quotes capture the post-event IV reduction.

### 2.4 Pin Risk at Expiry
The simulator's intrinsic value calculation at expiry (when `use_intrinsic_at_expiry=True`)
does not model pin risk — the scenario where the underlying closes within $0.50
of the short strike, creating uncertain assignment. In practice, being "pinned"
at the short strike of a spread introduces significant uncertainty about the
position's final value and hedge requirements.

---

## 3. Assignment and Exercise

### 3.1 Early Exercise Not Modeled
American-style options can be exercised early. The simulator does not model:

- Early assignment of short calls when a dividend is declared.
- Rational early exercise of deep ITM options when time value approaches zero.
- The cost of covering an unexpected assignment over a weekend or holiday.

### 3.2 Expiry Assignment Is Binary
At expiry, the simulator classifies legs as ITM (assigned) or OTM (expired
worthless) based on the final spot price vs strike. It does not model:

- Automatic exercise thresholds (options ≥ $0.01 ITM are typically auto-exercised).
- Counterparty-driven exercise decisions for options that are barely ITM.
- The capital requirement to accept or make delivery on assigned stock.

---

## 4. Timing and Session Rules

### 4.1 Quote Staleness Not Detected
The simulator does not validate that the quotes passed to `update_positions()`
are fresh. If stale quotes are fed (e.g., from a delayed data source), P&L
calculations will be incorrect. The caller is responsible for ensuring quote
recency.

### 4.2 Session Rule Precision
Session rules (no-trade windows near open/close) are evaluated against wall-clock
time using Eastern timezone. In backtesting contexts where bars are synthetic or
compressed, the wall-clock check may not correspond to the bar's actual time-of-day.
Callers running intraday backtests should ensure bar timestamps correctly reflect
Eastern market time.

### 4.3 No Pre-Market or After-Hours Simulation
The simulator enforces `market_open = 09:30` and `market_close = 16:00` Eastern.
Options traded in extended hours are not supported. Positions held overnight carry
gap risk that is captured only through the difference in quotes at next open.

---

## 5. Capital and Margin

### 5.1 No Margin or Buying Power Check
The simulator does not maintain a buying power or margin account. It will open
positions without verifying that sufficient capital exists. In live trading,
credit spreads require margin equal to the spread width minus credit received;
debit spreads require the full premium upfront.

**Consequence:** Paper results may reflect more positions open simultaneously
than a real account with limited capital could support.

### 5.2 No Buying Power Reduction on Opening
Opening a new position does not reduce available capital for subsequent positions
(beyond the `max_open_risk` check in RiskGuard). This allows the simulator to
over-allocate relative to a real account with a fixed capital base.

---

## 6. Fees and Costs

### 6.1 Fixed Commission Model
The fee model (`FeeConfig`) uses a flat per-contract commission plus a regulatory
fee. It does not model:

- Payment-for-order-flow (PFOF) dynamics, which affect effective spread cost.
- Exchange-specific fees that vary by option class (equity, index, ETF).
- Clearing and settlement fees beyond the regulatory pass-through.
- Margin interest on credit spread requirements held overnight.

### 6.2 Bid-Ask Friction Underestimation
Commissions in `FeeConfig` capture explicit fees, but the implicit cost of the
bid-ask spread is only captured through the fill method. With MIDPOINT fills,
the implicit half-spread cost is not included. Use MIDPOINT_PLUS_SLIPPAGE or
BID_ASK to more accurately reflect total transaction friction.

---

## 7. Risk Model Gaps

### 7.1 Correlation and Portfolio Greeks Not Tracked
When multiple positions are open, the simulator tracks aggregate open risk as
the sum of max-loss per position. It does not compute:

- Net delta of the portfolio.
- Cross-gamma or cross-vega across positions.
- Correlation between positions in related underlyings.

A real options book with 5 open positions in correlated names may have much
higher effective exposure than the sum of their individual max-losses.

### 7.2 Tail Risk and Gap Risk
The max-loss figures used for risk tracking assume orderly markets. Overnight
gaps, halt-opens, or extreme moves can cause realized losses exceeding the
theoretical max on debit structures (due to wide fills at open) or credit
structures (due to pin/assignment outcomes that the spread model doesn't capture).

### 7.3 Cooldown and Kill Switch Are Intraday Only
The `cooldown_minutes` and `kill_switch` controls in `RiskConfig` are evaluated
against the current session clock. `reset_daily()` clears the cooldown (but not
the kill switch) at session start. These controls do not persist across days or
sessions in persistent storage; they must be re-initialized or reconstructed when
the simulator is restarted.

---

## 8. Data Quality Dependencies

### 8.1 Options Chain Data Is Caller-Supplied
The simulator does not fetch live or historical options chain data. All quotes
(`LegQuote`) are provided by the caller. Simulation quality is bounded by the
quality of the input data:

- Stale NBBO quotes produce unrealistic fills.
- Missing IV data removes volatility-based P&L decomposition.
- Synthetic chains (computed from spot + Black-Scholes) introduce model error.

### 8.2 No Earnings or Corporate Action Awareness
The simulator has no knowledge of upcoming earnings, dividends, splits, or
mergers. These events cause:

- Option pricing discontinuities (IV spike before earnings, crush after).
- Adjusted strikes and multipliers post-split.
- Accelerated time decay around known catalyst dates.

Callers are responsible for avoiding or accounting for these events in backtests.

---

## Summary of Fidelity Tiers

| Aspect | Simulator Fidelity | Notes |
|--------|--------------------|-------|
| Fill price (liquid, 1 contract) | Moderate–High | MIDPOINT_PLUS_SLIPPAGE recommended |
| Fill price (illiquid / multi-contract) | Low | No depth model |
| Spread leg synchronization | Overoptimistic | No legging risk |
| P&L mark-to-market | Moderate | Depends on quote quality |
| Greeks / risk decomposition | None | Not tracked |
| Volatility surface dynamics | None | No skew or term structure |
| Early exercise / assignment | None | Not modeled |
| Capital / margin enforcement | None | No buying power check |
| Session time enforcement | Moderate | Wall-clock only, not bar-time |
| Fee accuracy | Moderate | Fixed model; no PFOF |
| Portfolio-level risk | Low | Additive max-loss only |

---

*Last updated: 2026-04-12*
