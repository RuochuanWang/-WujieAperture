#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_DEPTH_PRO="$PROJECT_ROOT/depth_anything_demo/third_party/depth_pro_src/ml-depth-pro-main"
LOCAL_DEPTH_CHECKPOINT="$LOCAL_DEPTH_PRO/checkpoints/depth_pro.pt"

echo "[1/5] Creating Python environment..."
"$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo "[2/5] Updating installer..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo "[3/5] Installing PyTorch and application dependencies..."
"$VENV_PYTHON" -m pip install torch==2.9.0 torchvision==0.24.0
"$VENV_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements-app.txt"

echo "[4/5] Preparing DepthPro..."
if [[ -f "$LOCAL_DEPTH_PRO/src/depth_pro/__init__.py" ]]; then
  echo "Using bundled DepthPro source: $LOCAL_DEPTH_PRO"
else
  "$VENV_PYTHON" -m pip install "depth_pro @ git+https://github.com/apple/ml-depth-pro.git@9efe5c1" || \
    "$VENV_PYTHON" -m pip install "https://github.com/apple/ml-depth-pro/archive/9efe5c1.zip"
fi

echo "[5/5] Preparing model checkpoint..."
mkdir -p "$PROJECT_ROOT/checkpoints"
if [[ ! -f "$PROJECT_ROOT/checkpoints/depth_pro.pt" && ! -f "$LOCAL_DEPTH_CHECKPOINT" ]]; then
  curl -fL "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt" \
    -o "$PROJECT_ROOT/checkpoints/depth_pro.pt"
fi

"$VENV_PYTHON" "$PROJECT_ROOT/scripts/verify_install.py"
echo "Setup complete. Run: ./scripts/start.sh"
