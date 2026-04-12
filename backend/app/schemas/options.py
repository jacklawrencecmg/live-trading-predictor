from pydantic import BaseModel
from typing import List, Optional


class OptionContract(BaseModel):
    symbol: str
    strike: float
    expiry: str
    option_type: str  # "call" or "put"
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    open_interest: int
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    in_the_money: bool
    intrinsic_value: float
    time_value: float


class OptionsChainRow(BaseModel):
    strike: float
    call: Optional[OptionContract]
    put: Optional[OptionContract]


class OptionsChain(BaseModel):
    symbol: str
    underlying_price: float
    expiry: str
    rows: List[OptionsChainRow]
    iv_rank: float
    put_call_ratio: float
    atm_iv: float
