from fastapi import APIRouter, Query
from typing import Optional, List
from app.services.options_service import fetch_options_chain, get_expirations
from app.schemas.options import OptionsChain

router = APIRouter()


@router.get("/chain/{symbol}", response_model=OptionsChain)
async def get_options_chain(symbol: str, expiry: Optional[str] = Query(None)):
    return await fetch_options_chain(symbol.upper(), expiry)


@router.get("/expirations/{symbol}", response_model=List[str])
async def get_expirations_route(symbol: str):
    return await get_expirations(symbol.upper())
