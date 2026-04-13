#!/bin/bash
# Runs ONCE when the Codespace is first created.
set -e

WS=/workspaces/options-research

echo "=== [1/4] Installing Redis ==="
sudo apt-get update -qq
sudo apt-get install -y -qq redis-server

echo "=== [2/4] Creating options_research database ==="
# The postgres feature creates the 'postgres' superuser with password 'postgres'.
# We just need to create our database.
sudo -u postgres createdb options_research 2>/dev/null \
  && echo "Database created." \
  || echo "Database already exists — skipping."

echo "=== [3/4] Installing Python dependencies ==="
pip install --quiet -r "$WS/backend/requirements.txt"

echo "=== [4/4] Installing Node dependencies ==="
npm install --prefix "$WS/frontend" --silent

echo ""
echo "=== Setup complete. Services will start automatically. ==="
