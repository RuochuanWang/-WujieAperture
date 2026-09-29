from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from dataset.explicit_bokeh_dataset import build_pos_map
from train import build_model, build_renderer, load_yaml_config


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEPTH_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
DEPTH_ANYTHING_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Render a folder of ordinary photos with the finalized first-paper VATD model."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="weights/VATD_weight.pth")
    parser.add_argument("--weights", choices=["raw", "ema"], default="raw")
    parser.add_argument(
        "--config",
        default="",
        help="Optional YAML override. The packaged VATD checkpoint already embeds its config.",
    )
    parser.add_argument(
        "--f-numbers",
        type=float,
        nargs="+",
        default=[1.8, 2.8, 4.0, 5.6, 8.0],
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--max-long-side",
        type=int,
        default=1536,
        help="Resize larger inputs while preserving aspect ratio; 0 keeps the original size.",
    )
    parser.add_argument(
        "--depth-dir",
        default="",
        help="Read precomputed relative inverse-depth maps instead of running Depth Anything V2.",
    )
    parser.add_argument(
        "--depth-anything-root",
        default="/root/autodl-tmp/Depth-Anything-V2",
        help="Path to the official Depth-Anything-V2 repository.",
    )
    parser.add_argument("--depth-checkpoint", default="")
    parser.add_argument(
        "--depth-encoder",
        choices=sorted(DEPTH_ANYTHING_CONFIGS),
        default="vitl",
    )
    parser.add_argument("--depth-input-size", type=int, default=518)
    parser.add_argument("--depth-low-percentile", type=float, default=1.0)
    parser.add_argument("--depth-high-percentile", type=float, default=99.0)
    parser.add_argument(
        "--invert-depth",
        action="store_true",
        help="Use only if a depth backend produces far-high rather than near-high values.",
    )
    parser.add_argument(
        "--focus-file",
        default="",
        help='Optional JSON mapping, e.g. {"1.jpg": [0.50, 0.42], "2": [0.35, 0.55]}.',
    )
    parser.add_argument(
        "--focus-patch-ratio",
        type=float,
        default=0.04,
        help="Side length of the robust depth patch around a manual focus point.",
    )
    parser.add_argument("--panel-width", type=int, default=320)
    parser.add_argument("--save-coarse", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Temporary YAML config overrides used only for this render.",
    )
    return parser


def apply_config_overrides(cfg: dict, overrides: list[str]) -> dict:
    cfg = dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set value {item!r}; expected KEY=VALUE")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty config key in --set value {item!r}")
        cfg[key] = yaml.safe_load(raw_value)
    return cfg


def natural_key(path: Path) -> list[Any]:
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", path.name)]


def list_images(folder: Path) -> list[Path]:
    paths = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(paths, key=natural_key)


def resolve_checkpoint(
    checkpoint: Any,
    weights: str,
) -> tuple[dict[str, torch.Tensor], Optional[dict]]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state dict or a dictionary containing 'model'.")
    if weights == "ema":
        if "ema_model" not in checkpoint:
            raise KeyError("Checkpoint does not contain EMA weights; use --weights raw.")
        return checkpoint["ema_model"], checkpoint.get("cfg")
    if "model" in checkpoint:
        return checkpoint["model"], checkpoint.get("cfg")
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, None
    raise KeyError("Checkpoint does not contain a model state dict.")


def load_render_model(
    checkpoint_path: Path,
    config_path: str,
    weights: str,
    device: torch.device,
    overrides: list[str],
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, embedded_cfg = resolve_checkpoint(checkpoint, weights)
    cfg = load_yaml_config(config_path) if config_path else embedded_cfg
    if cfg is None:
        raise ValueError("Checkpoint has no embedded config; pass --config.")
    cfg = apply_config_overrides(cfg, overrides)

    model = build_model(dict(cfg), device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, dict(cfg)


def load_physical_renderer(
    checkpoint_path: Path,
    config_path: str,
    weights: str,
    device: torch.device,
    overrides: list[str],
) -> tuple[torch.nn.Module, dict]:
    """Load the calibrated explicit renderer without constructing the neural corrector."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, embedded_cfg = resolve_checkpoint(checkpoint, weights)
    cfg = load_yaml_config(config_path) if config_path else embedded_cfg
    if cfg is None:
        raise ValueError("Checkpoint has no embedded config; pass --config.")
    cfg = apply_config_overrides(cfg, overrides)

    renderer = build_renderer(dict(cfg), device)
    renderer_state = {}
    for key, value in state_dict.items():
        normalized_key = key.removeprefix("module.")
        if normalized_key.startswith("renderer."):
            renderer_state[normalized_key.removeprefix("renderer.")] = value
    if not renderer_state:
        raise KeyError("Checkpoint does not contain explicit renderer weights.")
    renderer.load_state_dict(renderer_state, strict=True)
    renderer.eval()
    return renderer, dict(cfg)


class DepthAnythingPredictor:
    def __init__(
        self,
        repo_root: Path,
        checkpoint_path: Path,
        encoder: str,
        input_size: int,
        device: torch.device,
    ) -> None:
        if not repo_root.exists():
            raise FileNotFoundError(
                f"Depth Anything V2 repository not found: {repo_root}\n"
                "Pass --depth-anything-root or use --depth-dir with precomputed maps."
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Depth Anything V2 checkpoint not found: {checkpoint_path}\n"
                "Pass --depth-checkpoint explicitly."
            )

        sys.path.insert(0, str(repo_root))
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as exc:
            raise ImportError(
                f"Could not import depth_anything_v2 from {repo_root}. "
                "Use the official Depth-Anything-V2 repository."
            ) from exc

        model = DepthAnythingV2(**DEPTH_ANYTHING_CONFIGS[encoder])
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval()
        self.input_size = int(input_size)

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        # The official infer_image API follows OpenCV's BGR convention.
        bgr = rgb[..., ::-1].copy()
        depth = self.model.infer_image(bgr, input_size=self.input_size)
        return np.asarray(depth, dtype=np.float32)


class PrecomputedDepthPredictor:
    def __init__(self, depth_dir: Path) -> None:
        if not depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
        self.depth_dir = depth_dir

    def __call__(self, _: np.ndarray, image_path: Path) -> np.ndarray:
        candidates = [self.depth_dir / f"{image_path.stem}{suffix}" for suffix in DEPTH_SUFFIXES]
        depth_path = next((path for path in candidates if path.exists()), None)
        if depth_path is None:
            raise FileNotFoundError(f"No depth map found for {image_path.name} in {self.depth_dir}")
        with Image.open(depth_path) as image:
            return np.asarray(image.convert("F"), dtype=np.float32)


def default_depth_checkpoint(repo_root: Path, encoder: str) -> Path:
    return repo_root / "checkpoints" / f"depth_anything_v2_{encoder}.pth"


def resize_rgb(rgb: np.ndarray, max_long_side: int, divisor: int = 4) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = 1.0
    if max_long_side > 0 and max(height, width) > max_long_side:
        scale = max_long_side / float(max(height, width))

    new_height = max(divisor, int(round(height * scale)) // divisor * divisor)
    new_width = max(divisor, int(round(width * scale)) // divisor * divisor)
    if (new_height, new_width) == (height, width):
        return rgb

    image = Image.fromarray(rgb)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def normalize_inverse_depth(
    depth: np.ndarray,
    output_size: tuple[int, int],
    low_percentile: float,
    high_percentile: float,
    invert: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError("Depth prediction contains no finite values.")

    valid = depth[finite]
    low = float(np.percentile(valid, low_percentile))
    high = float(np.percentile(valid, high_percentile))
    if high <= low + 1e-8:
        raise ValueError(f"Depth prediction is nearly constant: low={low}, high={high}")

    normalized = np.nan_to_num((depth - low) / (high - low), nan=0.0, posinf=1.0, neginf=0.0)
    normalized = np.clip(normalized, 0.0, 1.0)
    if invert:
        normalized = 1.0 - normalized

    width, height = output_size
    if normalized.shape != (height, width):
        depth_image = Image.fromarray(normalized, mode="F")
        depth_image = depth_image.resize((width, height), Image.Resampling.BILINEAR)
        normalized = np.asarray(depth_image, dtype=np.float32).copy()

    return np.ascontiguousarray(normalized), {"raw_low": low, "raw_high": high}


def load_focus_points(path: str) -> dict[str, tuple[float, float]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    points: dict[str, tuple[float, float]] = {}
    for key, value in raw.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Focus point for {key!r} must be [x, y].")
        x, y = float(value[0]), float(value[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"Focus point for {key!r} must lie in [0, 1].")
        points[str(key)] = (x, y)
    return points


def lookup_focus_point(
    points: dict[str, tuple[float, float]],
    image_path: Path,
) -> Optional[tuple[float, float]]:
    return points.get(image_path.name, points.get(image_path.stem))


def focus_depth_from_point(
    depth: torch.Tensor,
    point: tuple[float, float],
    patch_ratio: float,
) -> torch.Tensor:
    _, _, height, width = depth.shape
    center_x = int(round(point[0] * (width - 1)))
    center_y = int(round(point[1] * (height - 1)))
    radius = max(2, int(round(min(height, width) * patch_ratio * 0.5)))
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    values = depth[:, :, y0:y1, x0:x1].reshape(depth.shape[0], -1)
    return values.median(dim=1).values.view(-1, 1, 1, 1).clamp(1e-4, 1.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def aperture_folder(f_number: float) -> str:
    return f"f_{f_number:g}".replace(".", "_")


def save_image(image: Image.Image, path: Path, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=jpeg_quality, subsampling=0)
    else:
        image.save(path)


def make_panel(
    items: list[tuple[str, Image.Image]],
    width: int,
    focus_point: Optional[tuple[float, float]],
) -> Image.Image:
    label_height = 28
    tiles: list[Image.Image] = []
    for index, (label, image) in enumerate(items):
        ratio = width / float(image.width)
        resized = image.resize(
            (width, max(1, int(round(image.height * ratio)))),
            Image.Resampling.LANCZOS,
        )
        tile = Image.new("RGB", (width, resized.height + label_height), "white")
        tile.paste(resized, (0, label_height))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 7), label, fill="black")
        if index == 0 and focus_point is not None:
            x = int(round(focus_point[0] * (resized.width - 1)))
            y = label_height + int(round(focus_point[1] * (resized.height - 1)))
            draw.line((x - 8, y, x + 8, y), fill=(255, 50, 20), width=2)
            draw.line((x, y - 8, x, y + 8), fill=(255, 50, 20), width=2)
        tiles.append(tile)

    max_height = max(tile.height for tile in tiles)
    panel = Image.new("RGB", (sum(tile.width for tile in tiles), max_height), "white")
    left = 0
    for tile in tiles:
        panel.paste(tile, (left, 0))
        left += tile.width
    return panel


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if any(value <= 0.0 for value in args.f_numbers):
        raise ValueError("All f-numbers must be positive.")
    if not (0.0 <= args.depth_low_percentile < args.depth_high_percentile <= 100.0):
        raise ValueError("Depth percentiles must satisfy 0 <= low < high <= 100.")
    if args.focus_patch_ratio <= 0.0:
        raise ValueError("--focus-patch-ratio must be positive.")

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model, cfg = load_render_model(
        Path(args.checkpoint),
        args.config,
        args.weights,
        device,
        args.overrides,
    )
    if args.depth_dir:
        depth_predictor: Any = PrecomputedDepthPredictor(Path(args.depth_dir))
        depth_backend = f"precomputed:{args.depth_dir}"
    else:
        repo_root = Path(args.depth_anything_root)
        checkpoint_path = (
            Path(args.depth_checkpoint)
            if args.depth_checkpoint
            else default_depth_checkpoint(repo_root, args.depth_encoder)
        )
        depth_predictor = DepthAnythingPredictor(
            repo_root=repo_root,
            checkpoint_path=checkpoint_path,
            encoder=args.depth_encoder,
            input_size=args.depth_input_size,
            device=device,
        )
        depth_backend = f"depth_anything_v2:{args.depth_encoder}"

    images = list_images(input_dir)
    if not images:
        raise ValueError(f"No supported images found in {input_dir}")
    focus_points = load_focus_points(args.focus_file)

    for folder in ["input", "depth", "panels"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    for f_number in args.f_numbers:
        (output_dir / aperture_folder(f_number)).mkdir(parents=True, exist_ok=True)
        if args.save_coarse:
            (output_dir / f"coarse_{aperture_folder(f_number)}").mkdir(
                parents=True, exist_ok=True
            )

    manifest: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "weights": args.weights,
        "checkpoint_experiment": cfg.get("experiment_name", ""),
        "config_overrides": args.overrides,
        "depth_backend": depth_backend,
        "f_numbers": args.f_numbers,
        "images": [],
    }

    amp_enabled = bool(args.amp and device.type == "cuda")
    for index, image_path in enumerate(images, start=1):
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        rgb = resize_rgb(rgb, args.max_long_side, divisor=4)
        source_pil = Image.fromarray(rgb, mode="RGB")
        width, height = source_pil.size

        if args.depth_dir:
            raw_depth = depth_predictor(rgb, image_path)
        else:
            raw_depth = depth_predictor(rgb)
        depth_np, depth_stats = normalize_inverse_depth(
            raw_depth,
            output_size=(width, height),
            low_percentile=args.depth_low_percentile,
            high_percentile=args.depth_high_percentile,
            invert=args.invert_depth,
        )

        source = to_tensor(source_pil).unsqueeze(0).to(device)
        depth = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(device)
        pos_map = build_pos_map(width, height).unsqueeze(0).to(device)
        focus_point = lookup_focus_point(focus_points, image_path)
        focus_depth = (
            focus_depth_from_point(depth, focus_point, args.focus_patch_ratio)
            if focus_point is not None
            else None
        )

        source_path = output_dir / "input" / f"{image_path.stem}.png"
        depth_path = output_dir / "depth" / f"{image_path.stem}.png"
        save_image(source_pil, source_path, args.jpeg_quality)
        depth_preview = Image.fromarray(
            np.round(depth_np * 255.0).astype(np.uint8), mode="L"
        )
        save_image(depth_preview, depth_path, args.jpeg_quality)

        panel_items: list[tuple[str, Image.Image]] = [
            ("Input", source_pil),
            ("DA-V2 inverse depth", depth_preview.convert("RGB")),
        ]
        output_paths: dict[str, str] = {}
        focus_value: Optional[float] = None

        for f_number in args.f_numbers:
            f_tensor = torch.tensor([f_number], dtype=torch.float32, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                model_out = model(
                    source=source,
                    depth=depth,
                    f_number=f_tensor,
                    pos_map=pos_map,
                    focus_depth=focus_depth,
                )

            rendered = tensor_to_pil(model_out["output"][0])
            rendered_path = (
                output_dir / aperture_folder(f_number) / f"{image_path.stem}.png"
            )
            save_image(rendered, rendered_path, args.jpeg_quality)
            output_paths[f"{f_number:g}"] = str(rendered_path)
            panel_items.append((f"f/{f_number:g}", rendered))
            focus_value = float(model_out["focus_depth"][0, 0, 0, 0].float().item())

            if args.save_coarse:
                coarse = tensor_to_pil(model_out["coarse"][0])
                coarse_path = (
                    output_dir
                    / f"coarse_{aperture_folder(f_number)}"
                    / f"{image_path.stem}.png"
                )
                save_image(coarse, coarse_path, args.jpeg_quality)

        panel = make_panel(panel_items, args.panel_width, focus_point)
        panel_path = output_dir / "panels" / f"{image_path.stem}.jpg"
        save_image(panel, panel_path, args.jpeg_quality)

        manifest["images"].append(
            {
                "input": str(image_path),
                "processed_size": [width, height],
                "focus_point": list(focus_point) if focus_point is not None else None,
                "focus_depth": focus_value,
                "depth_stats": depth_stats,
                "depth": str(depth_path),
                "outputs": output_paths,
                "panel": str(panel_path),
            }
        )
        print(
            f"[{index:02d}/{len(images):02d}] {image_path.name}: "
            f"size={width}x{height} focus={focus_value:.4f} saved={panel_path}"
        )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Finished {len(images)} images.")
    print(f"Results: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
