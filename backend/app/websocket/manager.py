import asyncio
import json
import time
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.market_data import fetch_candles, fetch_quote

websocket_router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}  # symbol -> sockets

    async def connect(self, ws: WebSocket, symbol: str):
        await ws.accept()
        if symbol not in self.connections:
            self.connections[symbol] = set()
        self.connections[symbol].add(ws)

    def disconnect(self, ws: WebSocket, symbol: str):
        if symbol in self.connections:
            self.connections[symbol].discard(ws)

    async def broadcast(self, symbol: str, data: dict):
        if symbol not in self.connections:
            return
        dead = set()
        for ws in list(self.connections[symbol]):
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.connections[symbol].discard(ws)


manager = ConnectionManager()


@websocket_router.websocket("/ws/market/{symbol}")
async def market_feed(websocket: WebSocket, symbol: str):
    symbol = symbol.upper()
    await manager.connect(websocket, symbol)
    try:
        while True:
            try:
                quote = await fetch_quote(symbol)
                candles = await fetch_candles(symbol, "5m", "1d")
                last_candle = candles.candles[-1] if candles.candles else None

                await websocket.send_json({
                    "type": "quote",
                    "symbol": symbol,
                    "price": quote.price,
                    "change": quote.change,
                    "change_pct": quote.change_pct,
                    "volume": quote.volume,
                    "timestamp": int(time.time()),
                })

                if last_candle:
                    await websocket.send_json({
                        "type": "candle",
                        "symbol": symbol,
                        "candle": last_candle.model_dump(),
                    })
            except WebSocketDisconnect:
                break
            except Exception:
                pass

            # Poll every 15 seconds
            await asyncio.sleep(15)

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, symbol)
