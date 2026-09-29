from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import torch
from PIL import Image


class DepthProUnavailableError(RuntimeError):
    """Raised when the optional DepthPro runtime is not ready."""


class DepthProPredictor:
    """Lazy wrapper around Apple's official DepthPro implementation.

    DepthPro returns metric depth (farther pixels have larger values). The bokeh
    renderer consumes a relative inverse-depth-like map, so conversion and robust
    percentile normalization happen in ``predict_inverse_depth``.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        source_path: Path | None = None,
        precision: str = "auto",
        low_percentile: float = 1.0,
        high_percentile: float = 99.0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.source_path = Path(source_path) if source_path is not None else None
        self.precision_name = precision
        self.low_percentile = float(low_percentile)
        self.high_percentile = float(high_percentile)
        self._model: Any | None = None
        self._transform: Any | None = None
        self._lock = RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _resolve_precision(self) -> torch.dtype:
        if self.precision_name == "float32" or self.device.type != "cuda":
            return torch.float32
        if self.precision_name in {"auto", "float16"}:
            return torch.float16
        raise ValueError(
            "DEPTH_PRO_PRECISION must be one of: auto, float16, float32"
        )

    def _add_local_source(self) -> bool:
        if self.source_path is None or not self.source_path.is_dir():
            return False
        source = str(self.source_path.resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        return True

    def runtime_available(self) -> bool:
        self._add_local_source()
        return importlib.util.find_spec("depth_pro") is not None

    @property
    def runtime_source(self) -> str:
        if self.source_path is not None and self.source_path.is_dir():
            return "local"
        return "installed"

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not self.checkpoint_path.is_file():
                raise DepthProUnavailableError(
                    f"未找到 DepthPro 权重：{self.checkpoint_path}。"
                    "请运行 scripts/setup.ps1（Windows）或 scripts/setup.sh（Linux/macOS）。"
                )
            self._add_local_source()
            try:
                import depth_pro
                from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
            except ImportError as exc:
                raise DepthProUnavailableError(
                    "未安装 DepthPro。请先运行项目的一键安装脚本，"
                    "或执行 pip install -r requirements-app.txt。"
                ) from exc

            config = replace(
                DEFAULT_MONODEPTH_CONFIG_DICT,
                checkpoint_uri=str(self.checkpoint_path.resolve()),
            )
            precision = self._resolve_precision()
            try:
                model, transform = depth_pro.create_model_and_transforms(
                    config=config,
                    device=self.device,
                    precision=precision,
                )
            except Exception as exc:
                raise DepthProUnavailableError(
                    f"DepthPro 初始化失败：{exc}"
                ) from exc
            self._model = model.eval()
            self._transform = transform

    @torch.inference_mode()
    def predict_inverse_depth(
        self,
        image: Image.Image,
    ) -> tuple[np.ndarray, dict[str, float]]:
        self.load()
        assert self._model is not None and self._transform is not None

        with self._lock:
            transformed = self._transform(image.convert("RGB"))
            prediction = self._model.infer(transformed)
            metric_depth = prediction["depth"].detach().float().cpu().numpy()
            focal = prediction.get("focallength_px")
            focal_px = (
                float(focal.detach().float().cpu().item())
                if torch.is_tensor(focal)
                else float(focal or 0.0)
            )

        metric_depth = np.asarray(metric_depth, dtype=np.float32).squeeze()
        finite = np.isfinite(metric_depth) & (metric_depth > 1e-6)
        if not finite.any():
            raise ValueError("DepthPro 未产生有效深度值。")

        safe_depth = metric_depth.copy()
        far_fill = float(np.max(metric_depth[finite]))
        safe_depth[~finite] = far_fill
        inverse_depth = 1.0 / np.maximum(safe_depth, 1e-4)
        valid_inverse = inverse_depth[finite]
        low = float(np.percentile(valid_inverse, self.low_percentile))
        high = float(np.percentile(valid_inverse, self.high_percentile))
        if high <= low + 1e-8:
            raise ValueError("DepthPro 深度图变化过小，无法生成可靠的景深效果。")

        normalized = np.clip((inverse_depth - low) / (high - low), 0.0, 1.0)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        stats = {
            "metric_depth_min_m": float(np.min(metric_depth[finite])),
            "metric_depth_max_m": float(np.max(metric_depth[finite])),
            "focal_length_px": focal_px,
            "inverse_depth_p01": low,
            "inverse_depth_p99": high,
        }
        return np.ascontiguousarray(normalized, dtype=np.float32), stats
