import { useEffect, useRef, useCallback, useState } from "react";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export interface WSMessage {
  type: "quote" | "candle";
  symbol: string;
  [key: string]: any;
}

export function useWebSocket(symbol: string, onMessage: (msg: WSMessage) => void) {
  const ws = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    if (!symbol) return;

    const connect = () => {
      const socket = new WebSocket(`${WS_URL}/ws/market/${symbol}`);
      ws.current = socket;

      socket.onopen = () => setConnected(true);
      socket.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data) as WSMessage;
          onMessageRef.current(msg);
        } catch {}
      };
      socket.onclose = () => {
        setConnected(false);
        // Reconnect after 5s
        setTimeout(connect, 5000);
      };
      socket.onerror = () => socket.close();
    };

    connect();
    return () => {
      ws.current?.close();
    };
  }, [symbol]);

  return { connected };
}
