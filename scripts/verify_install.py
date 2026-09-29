from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPTH_SOURCE = (
    ROOT
    / "depth_anything_demo"
    / "third_party"
    / "depth_pro_src"
    / "ml-depth-pro-main"
    / "src"
)
LOCAL_DEPTH_CHECKPOINT = LOCAL_DEPTH_SOURCE.parent / "checkpoints" / "depth_pro.pt"
if LOCAL_DEPTH_SOURCE.is_dir():
    sys.path.insert(0, str(LOCAL_DEPTH_SOURCE))

checks = {
    "VATD bokeh model": ROOT / "weights" / "VATD_weight.pth",
    "EBB bokeh model": ROOT / "weights" / "EBB_weight.pth",
    "Web interface": ROOT / "competition_app" / "static" / "index.html",
}

failed = False
for label, path in checks.items():
    exists = path.is_file() and path.stat().st_size > 0
    print(f"[{'OK' if exists else 'MISSING'}] {label}: {path}")
    failed |= not exists

depth_checkpoints = [ROOT / "checkpoints" / "depth_pro.pt", LOCAL_DEPTH_CHECKPOINT]
depth_checkpoint = next((path for path in depth_checkpoints if path.is_file()), None)
print(f"[{'OK' if depth_checkpoint else 'MISSING'}] DepthPro model: {depth_checkpoint or depth_checkpoints}")
failed |= depth_checkpoint is None

for package in ["torch", "torchvision", "fastapi", "uvicorn", "depth_pro"]:
    exists = importlib.util.find_spec(package) is not None
    print(f"[{'OK' if exists else 'MISSING'}] Python package: {package}")
    failed |= not exists

if failed:
    print("Installation is incomplete. See the missing entries above.", file=sys.stderr)
    raise SystemExit(1)

import torch

print(f"[INFO] PyTorch: {torch.__version__}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
print("Wujie Aperture installation verified.")
