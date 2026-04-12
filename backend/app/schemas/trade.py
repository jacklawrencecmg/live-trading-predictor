from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class TradeCreate(BaseModel):
    symbol: str
    option_symbol: Optional[str] = None
    action: str
    quantity: int
    price: Optional[float] = None  # None = use market
    strike: Optional[float] = None
    expiry: Optional[str] = None
    option_type: Optional[str] = None


class TradeOut(BaseModel):
    id: int
    symbol: str
    option_symbol: Optional[str]
    action: str
    quantity: int
    price: float
    executed_at: datetime
    model_prob_up: Optional[float]
    model_prob_down: Optional[float]
    model_confidence: Optional[float]

    class Config:
        from_attributes = True


class PositionOut(BaseModel):
    id: int
    symbol: str
    option_symbol: Optional[str]
    quantity: int
    avg_cost: float
    current_price: Optional[float]
    unrealized_pnl: float
    realized_pnl: float
    is_open: bool

    class Config:
        from_attributes = True


class PortfolioSummary(BaseModel):
    capital: float
    cash: float
    positions_value: float
    total_value: float
    daily_pnl: float
    daily_pnl_pct: float
    total_pnl: float
    open_positions: int
    kill_switch_active: bool
