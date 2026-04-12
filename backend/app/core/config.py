from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/options_research"
    redis_url: str = "redis://localhost:6379/0"
    secret_key: str = "dev-secret-key"

    # Risk defaults
    max_daily_loss_pct: float = 0.02       # 2% of capital
    max_position_size_pct: float = 0.05    # 5% per position
    cooldown_minutes: int = 15
    starting_capital: float = 100_000.0
    kill_switch: bool = False

    # Market data
    default_symbol: str = "SPY"
    candle_interval: str = "5m"
    candle_lookback_days: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
