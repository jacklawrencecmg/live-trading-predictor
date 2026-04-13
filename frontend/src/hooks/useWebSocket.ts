import { useEffect, useRef, useCallback, useState } from "react";

// Derive WebSocket base URL from the browser's own origin so this works in
// Codespaces, tunnels, and production without any env configuration.
// The /ws/* rewrite in next.config.js proxies the connection to the backend.
function getWsBase(): string {
  if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
  if (typeof window === "undefined") return "ws://localhost:3000";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}`;
}

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
      const socket = new WebSocket(`${getWsBase()}/ws/market/${symbol}`);
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
