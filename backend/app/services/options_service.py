import asyncio
import json
import math
from datetime import datetime, date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq

from app.core.redis_client import get_redis
from app.schemas.options import OptionContract, OptionsChain, OptionsChainRow

RISK_FREE_RATE = 0.05
CACHE_TTL = 120


# ---------- Black-Scholes math ----------

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        return max(K - S, 0)
    d1 = _d1(S, K, T, r, sigma)
    d2_val = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2_val)
    return K * math.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict:
    if T <= 1e-6 or sigma <= 1e-6:
        itm = (S > K) if option_type == "call" else (S < K)
        return {
            "delta": 1.0 if (option_type == "call" and itm) else (-1.0 if (option_type == "put" and itm) else 0.0),
            "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0,
        }
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    nd1 = norm.pdf(d1)
    sqrt_T = math.sqrt(T)

    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    gamma = nd1 / (S * sigma * sqrt_T)
    vega = S * nd1 * sqrt_T / 100  # per 1% move in IV
    theta_call = (
        -S * nd1 * sigma / (2 * sqrt_T)
        - r * K * math.exp(-r * T) * norm.cdf(d2)
    ) / 365
    theta = theta_call if option_type == "call" else (
        theta_call + r * K * math.exp(-r * T) / 365
    )
    rho_call = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    rho = rho_call if option_type == "call" else -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


def implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float, option_type: str
) -> float:
    if T <= 0 or market_price <= 0:
        return 0.0
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price <= intrinsic:
        return 0.01

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        iv = brentq(objective, 1e-4, 10.0, xtol=1e-6, maxiter=200)
        return float(iv)
    except Exception:
        return 0.3  # fallback


def _time_to_expiry(expiry_str: str) -> float:
    """Returns T in years."""
    try:
        exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        today = date.today()
        days = (exp - today).days
        return max(days, 0) / 365.0
    except Exception:
        return 0.0


# ---------- Options chain ----------

async def fetch_options_chain(symbol: str, expiry: Optional[str] = None) -> OptionsChain:
    cache_key = f"options:{symbol}:{expiry or 'nearest'}"
    _redis = None
    try:
        _redis = await get_redis()
        cached = await _redis.get(cache_key)
        if cached:
            return OptionsChain(**json.loads(cached))
    except Exception:
        _redis = None  # Redis unavailable — skip cache

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _build_chain, symbol, expiry)
    if _redis is not None:
        try:
            await _redis.setex(cache_key, CACHE_TTL, result.model_dump_json())
        except Exception:
            pass
    return result


def _build_chain(symbol: str, expiry: Optional[str]) -> OptionsChain:
    ticker = yf.Ticker(symbol)
    spot = float(ticker.fast_info.last_price or 100)

    exps = ticker.options
    if not exps:
        return OptionsChain(
            symbol=symbol, underlying_price=spot, expiry="N/A",
            rows=[], iv_rank=0.0, put_call_ratio=1.0, atm_iv=0.2
        )

    chosen_exp = expiry if (expiry and expiry in exps) else exps[0]
    chain = ticker.option_chain(chosen_exp)
    calls_df = chain.calls.copy()
    puts_df = chain.puts.copy()

    T = _time_to_expiry(chosen_exp)
    r = RISK_FREE_RATE

    # Compute strikes near the money (within 5%)
    strikes_calls = set(calls_df["strike"].tolist())
    strikes_puts = set(puts_df["strike"].tolist())
    all_strikes = sorted(strikes_calls | strikes_puts)
    near_strikes = [k for k in all_strikes if abs(k / spot - 1) <= 0.10]
    if not near_strikes:
        near_strikes = all_strikes

    rows = []
    call_vol_total = 0
    put_vol_total = 0
    atm_iv = 0.2

    for strike in near_strikes:
        call_row = calls_df[calls_df["strike"] == strike]
        put_row = puts_df[puts_df["strike"] == strike]

        call_contract = None
        put_contract = None

        if not call_row.empty:
            r_ = call_row.iloc[0]
            mid = (float(r_["bid"]) + float(r_["ask"])) / 2 if r_["bid"] > 0 else float(r_["lastPrice"])
            iv = float(r_.get("impliedVolatility", 0)) or implied_volatility(mid, spot, strike, T, r, "call")
            greeks = bs_greeks(spot, strike, T, r, iv, "call")
            intrinsic = max(spot - strike, 0)
            call_vol_total += int(r_["volume"]) if not pd.isna(r_["volume"]) else 0
            call_contract = OptionContract(
                symbol=f"{symbol}{chosen_exp.replace('-','')}C{int(strike*1000):08d}",
                strike=strike, expiry=chosen_exp, option_type="call",
                bid=float(r_["bid"]), ask=float(r_["ask"]), mid=mid,
                last=float(r_["lastPrice"]),
                volume=int(r_["volume"]) if not pd.isna(r_["volume"]) else 0,
                open_interest=int(r_["openInterest"]) if not pd.isna(r_["openInterest"]) else 0,
                iv=round(iv, 4), delta=round(greeks["delta"], 4),
                gamma=round(greeks["gamma"], 6), theta=round(greeks["theta"], 4),
                vega=round(greeks["vega"], 4), rho=round(greeks["rho"], 4),
                in_the_money=bool(r_.get("inTheMoney", False)),
                intrinsic_value=round(intrinsic, 4),
                time_value=round(max(mid - intrinsic, 0), 4),
            )
            if abs(strike / spot - 1) < 0.01:
                atm_iv = iv

        if not put_row.empty:
            r_ = put_row.iloc[0]
            mid = (float(r_["bid"]) + float(r_["ask"])) / 2 if r_["bid"] > 0 else float(r_["lastPrice"])
            iv = float(r_.get("impliedVolatility", 0)) or implied_volatility(mid, spot, strike, T, r, "put")
            greeks = bs_greeks(spot, strike, T, r, iv, "put")
            intrinsic = max(strike - spot, 0)
            put_vol_total += int(r_["volume"]) if not pd.isna(r_["volume"]) else 0
            put_contract = OptionContract(
                symbol=f"{symbol}{chosen_exp.replace('-','')}P{int(strike*1000):08d}",
                strike=strike, expiry=chosen_exp, option_type="put",
                bid=float(r_["bid"]), ask=float(r_["ask"]), mid=mid,
                last=float(r_["lastPrice"]),
                volume=int(r_["volume"]) if not pd.isna(r_["volume"]) else 0,
                open_interest=int(r_["openInterest"]) if not pd.isna(r_["openInterest"]) else 0,
                iv=round(iv, 4), delta=round(greeks["delta"], 4),
                gamma=round(greeks["gamma"], 6), theta=round(greeks["theta"], 4),
                vega=round(greeks["vega"], 4), rho=round(greeks["rho"], 4),
                in_the_money=bool(r_.get("inTheMoney", False)),
                intrinsic_value=round(intrinsic, 4),
                time_value=round(max(mid - intrinsic, 0), 4),
            )

        rows.append(OptionsChainRow(strike=strike, call=call_contract, put=put_contract))

    pcr = (put_vol_total / call_vol_total) if call_vol_total > 0 else 1.0

    # IV Rank: compare atm_iv to 52-week range (approximate via hist vol)
    iv_rank = _compute_iv_rank(ticker, atm_iv)

    return OptionsChain(
        symbol=symbol,
        underlying_price=round(spot, 2),
        expiry=chosen_exp,
        rows=rows,
        iv_rank=round(iv_rank, 4),
        put_call_ratio=round(pcr, 4),
        atm_iv=round(atm_iv, 4),
    )


def _compute_iv_rank(ticker, current_iv: float) -> float:
    try:
        hist = ticker.history(period="1y", interval="1d")
        if len(hist) < 20:
            return 0.5
        returns = hist["Close"].pct_change().dropna()
        rolling_vol = returns.rolling(20).std() * math.sqrt(252)
        low_vol = float(rolling_vol.min())
        high_vol = float(rolling_vol.max())
        if high_vol == low_vol:
            return 0.5
        return (current_iv - low_vol) / (high_vol - low_vol)
    except Exception:
        return 0.5


async def get_expirations(symbol: str) -> list:
    loop = asyncio.get_event_loop()
    ticker = yf.Ticker(symbol)
    exps = await loop.run_in_executor(None, lambda: ticker.options)
    return list(exps) if exps else []
