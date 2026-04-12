import pytest
import math
from app.services.options_service import bs_price, bs_greeks, implied_volatility, _time_to_expiry


def test_bs_call_price_positive():
    price = bs_price(S=100, K=100, T=30/365, r=0.05, sigma=0.2, option_type="call")
    assert price > 0


def test_bs_put_call_parity():
    S, K, T, r, sigma = 100, 100, 30/365, 0.05, 0.2
    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    # C - P = S - K*e^(-rT)
    lhs = call - put
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 0.01


def test_call_delta_range():
    greeks = bs_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.2, option_type="call")
    assert 0 < greeks["delta"] < 1


def test_put_delta_range():
    greeks = bs_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.2, option_type="put")
    assert -1 < greeks["delta"] < 0


def test_implied_vol_roundtrip():
    S, K, T, r, sigma = 100, 100, 30/365, 0.05, 0.25
    market_price = bs_price(S, K, T, r, sigma, "call")
    iv = implied_volatility(market_price, S, K, T, r, "call")
    assert abs(iv - sigma) < 0.001


def test_gamma_positive():
    greeks = bs_greeks(100, 100, 30/365, 0.05, 0.2, "call")
    assert greeks["gamma"] > 0


def test_expired_option():
    price = bs_price(S=100, K=95, T=0, r=0.05, sigma=0.2, option_type="call")
    assert abs(price - 5.0) < 0.01
