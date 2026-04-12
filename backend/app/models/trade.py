from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Enum
from sqlalchemy.orm import relationship
import enum
from app.core.database import Base


class TradeAction(str, enum.Enum):
    BUY_TO_OPEN = "BTO"
    SELL_TO_OPEN = "STO"
    BUY_TO_CLOSE = "BTC"
    SELL_TO_CLOSE = "STC"


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    option_symbol = Column(String(50), nullable=True)  # None = underlying
    action = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    underlying_price = Column(Float, nullable=True)
    strike = Column(Float, nullable=True)
    expiry = Column(String(12), nullable=True)
    option_type = Column(String(4), nullable=True)  # call / put
    delta = Column(Float, nullable=True)
    iv = Column(Float, nullable=True)
    model_prob_up = Column(Float, nullable=True)
    model_prob_down = Column(Float, nullable=True)
    model_confidence = Column(Float, nullable=True)
    executed_at = Column(DateTime, default=datetime.utcnow)
    is_paper = Column(Boolean, default=True)
