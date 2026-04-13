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

# Switch to trust auth so no password is needed for local connections.
# This lets psql -U postgres work without sudo -u postgres.
sudo sed -i \
  's/^\(local\s\+all\s\+\w\+\s\+\)peer/\1trust/g' \
  /etc/postgresql/*/main/pg_hba.conf 2>/dev/null || true

sudo service postgresql restart || true

# Wait for postgres to accept connections
for i in $(seq 1 20); do
  pg_isready -U postgres -q 2>/dev/null && break; sleep 1
done

# Create DB and set password (no sudo -u postgres needed with trust auth)
psql -U postgres -c "ALTER USER postgres PASSWORD 'postgres';" 2>/dev/null || true
psql -U postgres -c "CREATE DATABASE options_research;" 2>/dev/null \
  && echo "  DB created" \
  || echo "  DB already exists — skipping"

step "[3/4] Installing Python deps"
pip install -q -r "$WS/backend/requirements.txt" \
  && echo "  OK" || echo "  WARN: pip install had errors"

step "[4/4] Installing Node deps"
npm install --prefix "$WS/frontend" --silent \
  && echo "  OK" || echo "  WARN: npm install had errors"

echo ""
echo "=== on-create complete ==="
