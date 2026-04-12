"""
Live streaming loop — polls for new bars at the configured interval.
Runs as an asyncio background task.
"""

import asyncio
import logging
from typing import Set

from app.data_ingestion.ingestion_service import ingest_latest

logger = logging.getLogger(__name__)

_streaming_symbols: Set[str] = set()
_running = False


async def _poll_loop(symbol: str, timeframe: str, poll_seconds: int):
    while _running:
        try:
            count = await ingest_latest(symbol, timeframe)
            logger.debug("Streamed %d bars for %s/%s", count, symbol, timeframe)
        except Exception as exc:
            logger.error("Stream error for %s: %s", symbol, exc)
        await asyncio.sleep(poll_seconds)


async def start_streaming(symbols: list, timeframe: str = "5m", poll_seconds: int = 60):
    global _running
    _running = True
    tasks = [
        asyncio.create_task(_poll_loop(s, timeframe, poll_seconds))
        for s in symbols
    ]
    logger.info("Started streaming for %s", symbols)
    return tasks


def stop_streaming():
    global _running
    _running = False
