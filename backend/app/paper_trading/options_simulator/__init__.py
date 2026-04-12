"""
Options paper-execution simulator.

Entry point: PaperOptionsSimulator
Config:      SimulatorConfig
"""

from app.paper_trading.options_simulator.simulator import PaperOptionsSimulator
from app.paper_trading.options_simulator.config import (
    SimulatorConfig,
    FillConfig,
    FeeConfig,
    ContractSelectionConfig,
    SessionConfig,
    ExitConfig,
    RiskConfig,
    FillMethod,
)
from app.paper_trading.options_simulator.models import (
    SimPosition,
    PositionState,
    LegQuote,
    ExitEvent,
    OpenResult,
    ExecutionEvent,
    EventType,
)

__all__ = [
    "PaperOptionsSimulator",
    "SimulatorConfig",
    "FillConfig",
    "FeeConfig",
    "ContractSelectionConfig",
    "SessionConfig",
    "ExitConfig",
    "RiskConfig",
    "FillMethod",
    "SimPosition",
    "PositionState",
    "LegQuote",
    "ExitEvent",
    "OpenResult",
    "ExecutionEvent",
    "EventType",
]
