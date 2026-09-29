#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOKEH_HOST="${BOKEH_HOST:-127.0.0.1}"
export BOKEH_PORT="${BOKEH_PORT:-7860}"
cd "$PROJECT_ROOT"
exec .venv/bin/python -m competition_app
