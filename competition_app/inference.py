from __future__ import annotations

import hashlib
import io
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Literal, Mapping
from uuid import uuid4

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor

from render_photos import (
    focus_depth_from_point,
    load_physical_renderer,
    resize_rgb,
    tensor_to_pil,
)

from .depth_pro_backend import DepthProPredictor


ROOT_DIR = Path(__file__).resolve().parents[1]
PHYSICAL_OUTPUT_MODE: Literal["physical"] = "physical"
LOCAL_DEPTH_PRO_ROOT = (
    ROOT_DIR
    / "depth_anything_demo"
    / "third_party"
    / "depth_pro_src"
    / "ml-depth-pro-main"
)
LOCAL_DEPTH_PRO_SOURCE = LOCAL_DEPTH_PRO_ROOT / "src"
LOCAL_DEPTH_PRO_CHECKPOINT = LOCAL_DEPTH_PRO_ROOT / "checkpoints" / "depth_pro.pt"
PROJECT_DEPTH_PRO_CHECKPOINT = ROOT_DIR / "checkpoints" / "depth_pro.pt"


def _default_depth_checkpoint() -> Path:
    if PROJECT_DEPTH_PRO_CHECKPOINT.is_file():
        return PROJECT_DEPTH_PRO_CHECKPOINT
    return LOCAL_DEPTH_PRO_CHECKPOINT


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PipelineSettings:
    vatd_checkpoint_path: Path = ROOT_DIR / "weights" / "VATD_weight.pth"
    ebb_checkpoint_path: Path = ROOT_DIR / "weights" / "EBB_weight.pth"
    depth_checkpoint_path: Path = _default_depth_checkpoint()
    depth_source_path: Path = LOCAL_DEPTH_PRO_SOURCE
    device_name: str = "auto"
    max_long_side: int = 768
    max_custom_long_side: int = 2048
    amp: bool = True
    depth_precision: str = "auto"
    focus_patch_ratio: float = 0.04
    depth_cache_size: int = 4

    @classmethod
    def from_env(cls) -> "PipelineSettings":
        def resolve_path(value: str | os.PathLike[str]) -> Path:
            path = Path(value)
            return path if path.is_absolute() else ROOT_DIR / path

        default_long_side = int(os.getenv("BOKEH_MAX_LONG_SIDE", "768"))
        max_custom_long_side = max(
            default_long_side,
            int(os.getenv("BOKEH_MAX_CUSTOM_LONG_SIDE", "2048")),
        )

        return cls(
            vatd_checkpoint_path=resolve_path(
                os.getenv(
                    "BOKEH_VATD_CHECKPOINT",
                    os.getenv("BOKEH_CHECKPOINT", ROOT_DIR / "weights" / "VATD_weight.pth"),
                )
            ),
            ebb_checkpoint_path=resolve_path(
                os.getenv("BOKEH_EBB_CHECKPOINT", ROOT_DIR / "weights" / "EBB_weight.pth")
            ),
            depth_checkpoint_path=resolve_path(
                os.getenv("DEPTH_PRO_CHECKPOINT", _default_depth_checkpoint())
            ),
            depth_source_path=resolve_path(
                os.getenv("DEPTH_PRO_SOURCE_DIR", LOCAL_DEPTH_PRO_SOURCE)
            ),
            device_name=os.getenv("BOKEH_DEVICE", "auto"),
            max_long_side=default_long_side,
            max_custom_long_side=max_custom_long_side,
            amp=_env_bool("BOKEH_AMP", True),
            depth_precision=os.getenv("DEPTH_PRO_PRECISION", "auto"),
            focus_patch_ratio=float(os.getenv("BOKEH_FOCUS_PATCH_RATIO", "0.04")),
            depth_cache_size=max(1, int(os.getenv("BOKEH_DEPTH_CACHE_SIZE", "4"))),
        )


@dataclass
class RenderResult:
    image_bytes: bytes
    width: int
    height: int
    f_number: float
    focus_mode: str
    elapsed_seconds: float
    depth_seconds: float
    render_seconds: float
    depth_stats: dict[str, float]
    engine: str = "vatd"
    output_mode: Literal["physical"] = PHYSICAL_OUTPUT_MODE


@dataclass
class DepthEstimateResult:
    image_bytes: bytes
    depth_id: str
    width: int
    height: int
    elapsed_seconds: float
    depth_stats: dict[str, float]


@dataclass
class _CachedDepth:
    image: Image.Image
    image_digest: bytes
    normalized_depth: np.ndarray
    processing_long_side: int


class BokehPipeline:
    """Thread-safe, lazy single-image inference pipeline."""

    def __init__(self, settings: PipelineSettings | None = None) -> None:
        self.settings = settings or PipelineSettings.from_env()
        self.device = self._select_device(self.settings.device_name)
        self._renderers: dict[str, torch.nn.Module] = {}
        self._configs: dict[str, dict[str, Any]] = {}
        self._depth = DepthProPredictor(
            checkpoint_path=self.settings.depth_checkpoint_path,
            device=self.device,
            source_path=self.settings.depth_source_path,
            precision=self.settings.depth_precision,
        )
        self._depth_cache: OrderedDict[str, _CachedDepth] = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        requested = requested.strip().lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("BOKEH_DEVICE=cuda，但当前 PyTorch 未检测到 CUDA。")
        if requested not in {"cuda", "cpu", "mps"}:
            raise ValueError("BOKEH_DEVICE must be auto, cuda, cpu, or mps")
        return torch.device(requested)

    @property
    def model_loaded(self) -> bool:
        return bool(self._renderers)

    @staticmethod
    def _normalize_engine(engine: str) -> str:
        normalized = engine.strip().lower()
        if normalized not in {"vatd", "ebb"}:
            raise ValueError("渲染引擎须为 VATD 或 EBB。")
        return normalized

    def _engine_spec(self, engine: str) -> tuple[Path, str, str]:
        if engine == "vatd":
            return self.settings.vatd_checkpoint_path, str(
                ROOT_DIR / "train_vatd_physical_v3.yml"
            ), "raw"
        return self.settings.ebb_checkpoint_path, "", "ema"

    def readiness(self) -> dict[str, Any]:
        depth_package = self._depth.runtime_available()
        vatd_available = self.settings.vatd_checkpoint_path.is_file()
        ebb_available = self.settings.ebb_checkpoint_path.is_file()
        depth_ready = self.settings.depth_checkpoint_path.is_file() and depth_package
        ready = vatd_available and ebb_available and depth_ready
        return {
            "ready": ready,
            "device": str(self.device),
            "cuda_name": (
                torch.cuda.get_device_name(0)
                if self.device.type == "cuda"
                else None
            ),
            "bokeh_checkpoint": vatd_available,
            "models": {
                "vatd": {
                    "available": vatd_available,
                    "loaded": "vatd" in self._renderers,
                    "label": "VATD 连续光圈",
                    "aperture_range": [1.2, 16.0],
                },
                "ebb": {
                    "available": ebb_available,
                    "loaded": "ebb" in self._renderers,
                    "label": "EBB f/1.8 专项",
                    "aperture_range": [1.8, 1.8],
                },
            },
            "depth_pro_package": depth_package,
            "depth_pro_checkpoint": self.settings.depth_checkpoint_path.is_file(),
            "depth_pro_source": self._depth.runtime_source,
            "model_loaded": self.model_loaded,
            "depth_model_loaded": self._depth.is_loaded,
            "max_long_side": self.settings.max_long_side,
            "max_custom_long_side": self.settings.max_custom_long_side,
            "output_mode": PHYSICAL_OUTPUT_MODE,
        }

    def load(self, engine: str = "vatd") -> None:
        engine = self._normalize_engine(engine)
        with self._lock:
            if engine in self._renderers:
                return
            checkpoint_path, config_path, weights = self._engine_spec(engine)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"未找到 {engine.upper()} 景深渲染权重：{checkpoint_path}"
                )
            renderer, cfg = load_physical_renderer(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                weights=weights,
                device=self.device,
                overrides=[],
            )
            self._renderers[engine] = renderer
            self._configs[engine] = cfg

    @staticmethod
    def decode_image(data: bytes) -> Image.Image:
        try:
            with Image.open(io.BytesIO(data)) as opened:
                opened.load()
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except Exception as exc:
            raise ValueError("无法读取图片，请使用 JPG、PNG 或 WebP 文件。") from exc
        if image.width < 64 or image.height < 64:
            raise ValueError("图片尺寸过小，宽高至少为 64 像素。")
        return image

    def _resolve_long_side(self, requested: int | None) -> int:
        value = self.settings.max_long_side if requested is None else int(requested)
        if value < 256:
            raise ValueError("处理尺寸的最长边不能小于 256 像素。")
        if value > self.settings.max_custom_long_side:
            raise ValueError(
                f"处理尺寸的最长边不能超过 {self.settings.max_custom_long_side} 像素。"
            )
        if value % 4 != 0:
            raise ValueError("处理尺寸的最长边须为 4 的倍数。")
        return value

    def _prepare_image(
        self,
        image: Image.Image,
        max_long_side: int | None = None,
    ) -> Image.Image:
        rgb = np.asarray(image, dtype=np.uint8)
        resized = resize_rgb(
            rgb,
            self._resolve_long_side(max_long_side),
            divisor=4,
        )
        return Image.fromarray(resized, mode="RGB")

    @staticmethod
    def _depth_preview_bytes(depth: np.ndarray) -> bytes:
        preview = np.rint(np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(preview, mode="L").save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    @staticmethod
    def _image_digest(image: Image.Image) -> bytes:
        return hashlib.sha256(image.tobytes()).digest()

    @torch.inference_mode()
    def estimate_depth(
        self,
        image: Image.Image,
        max_long_side: int | None = None,
    ) -> DepthEstimateResult:
        started = time.perf_counter()
        with self._lock:
            processing_long_side = self._resolve_long_side(max_long_side)
            prepared = self._prepare_image(image, processing_long_side)
            normalized_depth, depth_stats = self._depth.predict_inverse_depth(prepared)
            depth_id = uuid4().hex
            self._depth_cache[depth_id] = _CachedDepth(
                image=prepared,
                image_digest=self._image_digest(prepared),
                normalized_depth=normalized_depth,
                processing_long_side=processing_long_side,
            )
            while len(self._depth_cache) > self.settings.depth_cache_size:
                self._depth_cache.popitem(last=False)

        return DepthEstimateResult(
            image_bytes=self._depth_preview_bytes(normalized_depth),
            depth_id=depth_id,
            width=prepared.width,
            height=prepared.height,
            elapsed_seconds=time.perf_counter() - started,
            depth_stats=depth_stats,
        )

    def _cached_depth(self, depth_id: str) -> _CachedDepth:
        cached = self._depth_cache.get(depth_id)
        if cached is None:
            raise ValueError("深度结果已失效，请重新生成深度图。")
        self._depth_cache.move_to_end(depth_id)
        return cached

    @staticmethod
    def _validate_parameters(
        f_number: float,
        focus_point: tuple[float, float] | None,
        engine: str,
    ) -> None:
        if not 1.2 <= float(f_number) <= 16.0:
            raise ValueError("光圈值须在 f/1.2 到 f/16 之间。")
        if focus_point is not None and not all(0.0 <= v <= 1.0 for v in focus_point):
            raise ValueError("焦点坐标须在图片范围内。")
        if engine == "ebb" and abs(float(f_number) - 1.8) > 1e-6:
            raise ValueError("EBB 专项引擎仅在 f/1.8 上完成训练与验证，请使用 f/1.8。")

    @staticmethod
    def _physical_tensor(renderer_output: Mapping[str, Any]) -> torch.Tensor:
        """Return only the explicit renderer output; never fall back to a final image."""
        if "coarse" not in renderer_output:
            raise RuntimeError(
                "显式物理渲染器未返回 coarse；为避免误用神经最终结果，处理已中止。"
            )
        physical = renderer_output["coarse"]
        if not torch.is_tensor(physical) or physical.ndim != 4:
            raise RuntimeError("显式物理渲染器返回了无效的 coarse 张量。")
        return physical

    @torch.inference_mode()
    def render(
        self,
        image: Image.Image,
        f_number: float,
        focus_point: tuple[float, float] | None = None,
        engine: str = "vatd",
        progress: Callable[[str], None] | None = None,
        depth_id: str | None = None,
        max_long_side: int | None = None,
    ) -> RenderResult:
        engine = self._normalize_engine(engine)
        self._validate_parameters(f_number, focus_point, engine)
        started = time.perf_counter()
        progress = progress or (lambda _: None)

        with self._lock:
            progress("正在准备图像")
            if depth_id:
                cached = self._cached_depth(depth_id)
                requested_long_side = (
                    cached.processing_long_side
                    if max_long_side is None
                    else max_long_side
                )
                prepared = self._prepare_image(image, requested_long_side)
                if self._image_digest(prepared) != cached.image_digest:
                    raise ValueError("深度结果与当前图片不匹配，请重新生成深度图。")
                image = cached.image
                depth_np = cached.normalized_depth
                depth_stats: dict[str, float] = {}
                depth_seconds = 0.0
            else:
                image = self._prepare_image(image, max_long_side)
                progress("DepthPro 正在理解空间层次")
                depth_started = time.perf_counter()
                depth_np, depth_stats = self._depth.predict_inverse_depth(image)
                depth_seconds = time.perf_counter() - depth_started
            width, height = image.size

            progress("正在计算连续光圈与散焦半径")
            self.load(engine)
            renderer = self._renderers[engine]
            source = to_tensor(image).unsqueeze(0).to(self.device)
            depth = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(self.device)
            focus_depth = (
                focus_depth_from_point(
                    depth,
                    focus_point,
                    self.settings.focus_patch_ratio,
                )
                if focus_point is not None
                else None
            )
            f_tensor = torch.tensor(
                [float(f_number)], dtype=torch.float32, device=self.device
            )

            progress("正在渲染中")
            render_started = time.perf_counter()
            amp_enabled = bool(self.settings.amp and self.device.type == "cuda")
            amp_dtype = (
                torch.bfloat16
                if amp_enabled and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            with torch.autocast(
                device_type=self.device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                physical_outputs = renderer(
                    source=source,
                    depth=depth,
                    f_number=f_tensor,
                    focus_depth=focus_depth,
                )
                physical = self._physical_tensor(physical_outputs)
            rendered = tensor_to_pil(physical[0])
            render_seconds = time.perf_counter() - render_started

            buffer = io.BytesIO()
            rendered.save(buffer, format="PNG", optimize=True)
            elapsed = time.perf_counter() - started
            return RenderResult(
                image_bytes=buffer.getvalue(),
                width=width,
                height=height,
                f_number=float(f_number),
                focus_mode="manual" if focus_point is not None else "auto",
                elapsed_seconds=elapsed,
                depth_seconds=depth_seconds,
                render_seconds=render_seconds,
                depth_stats=depth_stats,
                engine=engine,
            )

    def recover_from_oom(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
