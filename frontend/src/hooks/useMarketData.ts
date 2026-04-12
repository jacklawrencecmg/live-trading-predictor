import { useState, useEffect, useCallback } from "react";
import { getQuote, getCandles, MarketQuote, Candle } from "@/lib/api";

export function useMarketData(symbol: string) {
  const [quote, setQuote] = useState<MarketQuote | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!symbol) return;
    setLoading(true);
    setError(null);
    try {
      const [qRes, cRes] = await Promise.all([
        getQuote(symbol),
        getCandles(symbol, "5m", "5d"),
      ]);
      setQuote(qRes.data);
      setCandles(cRes.data.candles);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { quote, candles, loading, error, refresh };
}
