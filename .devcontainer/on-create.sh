#!/bin/bash
# Runs ONCE when the Codespace is first created.
# No set -e: each step reports failure but continues so one bad step
# doesn't block the rest of setup.

WS=/workspaces/options-research

echo "=== [1/4] Installing Redis ==="
sudo apt-get update -qq && sudo apt-get install -y -qq redis-server \
  && echo "Redis installed." \
  || echo "WARNING: Redis install failed — it may already be present."

echo "=== [2/4] Creating options_research database ==="
# Wait up to 20s for the postgres feature to have postgres ready.
for i in $(seq 1 20); do
  pg_isready -U postgres -q 2>/dev/null && break
  sleep 1
done
sudo -u postgres createdb options_research 2>/dev/null \
  && echo "Database created." \
  || echo "Database already exists or postgres not ready — skipping."

echo "=== [3/4] Installing Python dependencies ==="
pip install --quiet -r "$WS/backend/requirements.txt" \
  && echo "Python deps installed." \
  || echo "WARNING: pip install had errors — check requirements.txt."

echo "=== [4/4] Installing Node dependencies ==="
npm install --prefix "$WS/frontend" --silent \
  && echo "Node deps installed." \
  || echo "WARNING: npm install had errors."

echo ""
echo "=== Setup complete. Services will start automatically. ==="
