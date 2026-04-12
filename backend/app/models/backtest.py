from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from app.core.database import Base


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False)
    interval = Column(String(10), nullable=False)
    start_date = Column(String(12), nullable=False)
    end_date = Column(String(12), nullable=False)
    n_folds = Column(Integer, nullable=False)
    train_size = Column(Integer, nullable=False)
    test_size = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    log_loss = Column(Float, nullable=True)
    magnitude_mae = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    total_return = Column(Float, nullable=True)
    n_trades = Column(Integer, nullable=True)
    fold_results = Column(JSON, nullable=True)
    calibration_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
