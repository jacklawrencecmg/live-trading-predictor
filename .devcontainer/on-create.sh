#!/bin/bash
# Runs ONCE when the Codespace is first created.

set -e
WS=/workspaces/options-research

echo "=== [1/5] Installing system packages (Redis + PostgreSQL) ==="
sudo apt-get update -qq
sudo apt-get install -y -qq redis-server postgresql postgresql-contrib

echo "=== [2/5] Configuring PostgreSQL ==="
sudo service postgresql start
# Wait for postgres to accept connections
for i in $(seq 1 15); do
  pg_isready -U postgres -q 2>/dev/null && break
  sleep 1
done
# Set password and create database
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';" 2>/dev/null || true
sudo -u postgres createdb options_research 2>/dev/null \
  && echo "Database 'options_research' created." \
  || echo "Database already exists — skipping."

echo "=== [3/5] Installing Python dependencies ==="
pip install --quiet -r "$WS/backend/requirements.txt"

echo "=== [4/5] Installing Node dependencies ==="
npm install --prefix "$WS/frontend" --silent

echo "=== [5/5] Done ==="
echo ""
echo "Setup complete. Services will auto-start via postStartCommand."
