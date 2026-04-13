# Paper Options Simulator — Known Limitations

This document describes the inherent gaps between the paper simulator and real options
execution. Every item listed here represents a way the simulator can be **more optimistic
than reality** and should be understood before drawing conclusions from simulation results.

---

## Fill Realism

| # | Limitation | Impact | Severity |
|---|-----------|--------|----------|
| F1 | **MIDPOINT fill is optimistic.** In practice, retail orders rarely fill at the mid for single-leg options; the true achievable price sits closer to the natural side for all but the most liquid names (SPY, QQQ). | Understates entry cost by ~15–30% of the spread for single-leg trades. | High |
| F2 | **Spread width is not dynamically adjusted.** The simulator uses the quote at the moment of the open signal. For 5-minute bars, the live spread at order time may be wider (low-volume periods, news, pre/post-market overhang). | Can understate actual entry cost, especially at open or near events. | High |
| F3 | **No order-book depth modeling.** Sizes > 1 contract may move the market. The simulator assumes fills at the same price regardless of size. | Understates market impact for any size > 1. | Medium |
| F4 | **Closing fills use the same fill model as opens.** In practice, unwinding a winning position requires lifting offers; closing losers can be done at bid. The asymmetry is only partially captured by CONSERVATIVE mode. | Can overstate closing proceeds on profitable positions. | Medium |
| F5 | **No roll modeling.** The simulator does not support rolling a position to a later expiry. Exiting near-DTE and re-opening is the only supported continuation path. | Limits strategy realism for managed positions. | Low |

---

## Options Mechanics

| # | Limitation | Impact | Severity |
|---|-----------|--------|----------|
| M1 | **American-style early exercise not modeled.** Short puts on dividend-paying stocks can be assigned early. The simulator only evaluates assignment at expiry (DTE = 0). | Can miss early-assignment losses on short equity puts, especially near ex-dividend dates. | High |
| M2 | **Pin risk is flagged but not resolved.** When the underlying pins near a strike at expiry, real outcome is uncertain (the short may or may not be assigned). The simulator settles at intrinsic value. | Can misstate P&L by up to one spread width in pin scenarios. | High |
| M3 | **Dividend risk not modeled.** Deep ITM calls may be exercised early to capture dividend. Simulator does not check dividend schedules. | Can overstate P&L for positions straddling ex-dividend dates. | Medium |
| M4 | **Volatility crush after events not modeled.** Earnings, Fed announcements, and macro events cause IV to collapse after the catalyst. The simulator uses the current IV/quote for all marks. | Significantly overstates value of long volatility positions held through events. | High |
| M5 | **Greeks P&L decomposition absent.** The simulator tracks total P&L only. No delta/gamma/theta/vega decomposition is produced. | Cannot attribute P&L to individual Greek contributions; makes hedging analysis impossible. | Low |
| M6 | **Skew and surface not modeled.** The IV for OTM puts and calls is treated as flat (ATM IV used everywhere). In practice, implied vol is higher for downside puts and lower for upside calls. | Understates cost of OTM put protection; understates credit for OTM put sales. | Medium |
| M7 | **No physical delivery simulation.** Equity options can result in 100 shares per contract at expiry. The simulator does not track stock positions or margin calls from stock delivery. | Limits realism of any expiry simulation involving exercise. | Medium |

---

## Risk Controls

| # | Limitation | Impact | Severity |
|---|-----------|--------|----------|
| R1 | **Margin requirements not enforced.** Credit spreads, naked shorts, and long options each carry different margin requirements. The simulator only tracks `max_daily_loss` and `max_open_risk` in dollar terms. | Simulator can open positions that a real account would reject due to insufficient margin. | High |
| R2 | **Buying power effect not tracked.** The simulator does not maintain a capital account. There is no check that sufficient capital exists to support open risk. | Overstates trade capacity for underfunded accounts. | High |
| R3 | **Position-level Greeks limits absent.** Many production risk systems limit net delta, gamma, and vega exposure. The simulator has no Greek-based gates. | Can accumulate excessive directional or volatility exposure without constraint. | Medium |
| R4 | **No circuit-breaker on underlying price move.** A large gap move in the underlying can blow through stop-loss levels between update ticks. | Stop losses are checked at update frequency (1 bar), not intrabar. | Medium |

---

## Data and Timing

| # | Limitation | Impact | Severity |
|---|-----------|--------|----------|
| D1 | **Quotes are assumed point-in-time and current.** The simulator does not model quote staleness. If market data is delayed or the last bar has old prices, fills will be based on stale mid-prices. | Can fill at better-than-available prices if chain data lags. | High |
| D2 | **No execution latency modeled.** Real orders have submission → acknowledgement → fill latency (100ms–2s for retail). Signals based on the closing bar of bar N will not fill until early bar N+1 at best. | The simulator opens at bar N's close price, not bar N+1's open. | Medium |
| D3 | **No partial fill modeling.** The simulator assumes all legs fill completely on the first attempt. Real multi-leg orders can partially fill or have one leg fill while another doesn't. | Overstates fill reliability, especially for spread orders. | Medium |
| D4 | **Calendar and holiday handling is minimal.** The session rules use static open/close times. Early closes, extended hours, and market halts are not handled. | Simulator may attempt fills on days markets are closed or during halts. | Low |

---

## Regulatory and Tax

| # | Limitation | Impact | Severity |
|---|-----------|--------|----------|
| T1 | **Wash-sale rule not applied.** Realizing a loss and reopening a substantially identical position within 30 days triggers wash-sale disallowance in the US. | Reported P&L can differ from tax-reported P&L. | Low (paper only) |
| T2 | **PDT (Pattern Day Trader) rule not enforced.** Accounts under $25K are limited to 3 day trades in a rolling 5-session window. The simulator does not track or block round-trip same-day trades. | Can execute strategies that would be blocked for small accounts. | Medium |
| T3 | **FINRA/SEC regulatory fees are approximate.** The `regulatory_fee_per_contract` config field is a fixed estimate. Actual fees vary with trade volume, exchange, and quarterly adjustments. | Minor P&L impact; sub-dollar discrepancy per trade. | Low |

---

## Backtesting Context

| # | Limitation | Impact | Severity |
|---|-----------|--------|----------|
| B1 | **No look-ahead protection at the simulator level.** The simulator itself is forward-safe (it only acts on signals it receives), but the signals produced by the inference layer may contain look-ahead leakage if features are not properly shifted. See `FALSE_CONFIDENCE_AUDIT.md`. | Can produce unrealistically strong backtest results if signals leak future information. | Critical |
| B2 | **Survivorship bias in underlying selection.** Backtesting only on symbols that still exist today implicitly selects for companies that survived. | Overstates strategy performance vs a live selection process. | High |
| B3 | **Low regime-specific sample sizes.** EVENT_RISK and LIQUIDITY_POOR regimes may have too few occurrences per symbol to produce statistically reliable regime-conditional P&L estimates. | Regime-segmented performance tables may be noisy. See `FALSE_CONFIDENCE_AUDIT.md`. | High |

---

## Summary: Severity Ranking

| Severity | Count | Examples |
|----------|-------|---------|
| **Critical** | 1 | B1 (look-ahead leakage in signals) |
| **High** | 10 | F1, F2, M1, M2, M4, R1, R2, D1, B2, B3 |
| **Medium** | 8 | F3, F4, M3, M6, M7, R3, R4, D2, D3, T2 |
| **Low** | 5 | F5, M5, D4, T1, T3 |

---

*Last updated: 2025. Review whenever the simulator config defaults or fill engine logic changes.*
