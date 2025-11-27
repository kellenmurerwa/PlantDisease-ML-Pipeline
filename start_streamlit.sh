#!/usr/bin/env bash
set -euo pipefail

# Use the port provided by Render (fallback to 8000 for local runs)
PORT="${PORT:-8000}"

exec streamlit run ui/streamlit_app.py --server.port "$PORT" --server.address 0.0.0.0
