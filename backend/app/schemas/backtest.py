from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class BacktestRequest(BaseModel):
    symbol: str = "SPY"
    interval: str = "5m"
    period: str = "60d"
    n_folds: int = 5
    train_size: int = 200
    test_size: int = 50
    confidence_threshold: float = 0.60


class BacktestResultOut(BaseModel):
    id: int
    symbol: str
    interval: str
    start_date: str
    end_date: str
    n_folds: int
    accuracy: Optional[float]
    brier_score: Optional[float]
    log_loss: Optional[float]
    magnitude_mae: Optional[float]
    sharpe_ratio: Optional[float]
    total_return: Optional[float]
    n_trades: Optional[int]
    fold_results: Optional[List[Dict]]
    calibration_data: Optional[Dict]

    class Config:
        from_attributes = True
