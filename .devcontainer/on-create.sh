#!/bin/bash
# Runs ONCE when the Codespace is first created.
# No set -e — each step logs failure but continues.

# Resolve workspace root relative to this script — works for any repo name.
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() { echo ""; echo "=== $* ==="; }

step "[1/4] Installing Redis + PostgreSQL"
sudo apt-get update -qq
sudo apt-get install -y redis-server postgresql postgresql-contrib \
  && echo "  OK" || echo "  WARN: apt install had errors"

step "[2/4] Configuring PostgreSQL"
sudo service postgresql start || true
for i in $(seq 1 20); do
  pg_isready -U postgres -q 2>/dev/null && break; sleep 1
done
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';" 2>/dev/null || true
sudo -u postgres createdb options_research 2>/dev/null \
  && echo "  DB created" || echo "  DB already exists — skipping"

step "[3/4] Installing Python deps"
pip install -q -r "$WS/backend/requirements.txt" \
  && echo "  OK" || echo "  WARN: pip install had errors — check requirements.txt"

step "[4/4] Installing Node deps"
npm install --prefix "$WS/frontend" --silent \
  && echo "  OK" || echo "  WARN: npm install had errors"

echo ""
echo "=== on-create complete ==="
