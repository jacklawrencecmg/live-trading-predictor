#!/bin/bash
# Runs on every Codespace start and resume.
# Starts Redis, the FastAPI backend, and the Next.js frontend.
WS=/workspaces/options-research

# ── Redis ─────────────────────────────────────────────────────────────────────
echo "[services] Starting Redis..."
sudo service redis-server start 2>/dev/null \
  || sudo redis-server --daemonize yes --logfile /tmp/redis.log
sleep 1

# ── PostgreSQL ────────────────────────────────────────────────────────────────
# The postgres devcontainer feature manages pg as a systemd/init service.
# It usually starts automatically; this is a safety net.
echo "[services] Ensuring PostgreSQL is running..."
sudo service postgresql start 2>/dev/null || true
# Wait up to 10s for postgres to accept connections
for i in $(seq 1 10); do
  pg_isready -U postgres -q && break
  sleep 1
done

# ── Backend ───────────────────────────────────────────────────────────────────
echo "[services] Starting backend on :8000..."
pkill -f "uvicorn app.main" 2>/dev/null || true
sleep 1

cd "$WS/backend"
# Settings default to localhost postgres/redis — no extra env vars needed.
nohup python -m uvicorn app.main:app \
  --host 0.0.0.0 --port 8000 --reload \
  > /tmp/backend.log 2>&1 &
echo "  backend PID=$! — logs: tail -f /tmp/backend.log"

# ── Frontend ──────────────────────────────────────────────────────────────────
echo "[services] Starting frontend on :3000..."
pkill -f "next-server\|next dev" 2>/dev/null || true
sleep 1

cd "$WS/frontend"
nohup npm run dev > /tmp/frontend.log 2>&1 &
echo "  frontend PID=$! — logs: tail -f /tmp/frontend.log"

echo ""
echo "[services] All services launched. Give them ~15s to be ready."
echo "  Frontend: https://\$CODESPACE_NAME-3000.\$GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN"
echo "  Backend:  http://localhost:8000/health"
echo "  Logs:     tail -f /tmp/backend.log /tmp/frontend.log"
