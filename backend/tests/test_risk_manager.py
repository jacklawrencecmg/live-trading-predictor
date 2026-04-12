import pytest
import pytest_asyncio


@pytest.mark.asyncio
async def test_kill_switch():
    from unittest.mock import AsyncMock, patch
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    mock_redis.delete = AsyncMock()

    with patch("app.services.risk_manager.get_redis", return_value=mock_redis):
        from app.services.risk_manager import set_kill_switch, is_kill_switch_active
        mock_redis.get.return_value = "1"
        assert await is_kill_switch_active()


@pytest.mark.asyncio
async def test_risk_violation_on_large_trade():
    from unittest.mock import AsyncMock, patch, MagicMock
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)  # kill switch off
    mock_redis.setex = AsyncMock()

    with patch("app.services.risk_manager.get_redis", return_value=mock_redis):
        with patch("app.services.risk_manager.get_capital", return_value=100_000.0):
            with patch("app.services.risk_manager.get_daily_pnl", return_value=0.0):
                from app.services.risk_manager import check_all_risks, RiskViolation
                # 10% position (above 5% limit)
                with pytest.raises(RiskViolation, match="Position size"):
                    await check_all_risks("SPY", 10_001.0)
