from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


@dataclass(frozen=True)
class SampleRecord:
    scene_id: str
    source_path: Path
    depth_path: Path
    target_path: Path
    f_number: float
    mask_path: Optional[Path] = None
    softmask_path: Optional[Path] = None


def parse_fnumber_from_folder(folder_name: str) -> float:
    """
    Convert folder names like:
        '1_8' -> 1.8
        '2_8' -> 2.8
        '8'   -> 8.0
    """
    if "_" in folder_name:
        return float(folder_name.replace("_", "."))
    return float(folder_name)


def build_pos_map(width: int, height: int) -> Tensor:
    """
    Return a 2xHxW positional map in [0, 1].
    Channel 0: x-coordinate
    Channel 1: y-coordinate (top=1, bottom=0, same convention as original repo)
    """
    if width > height:
        crop_dist = (1.0 - (height / width)) / 2.0
        x_lin = torch.linspace(0.0, 1.0, width)
        y_lin = torch.linspace(1.0 - crop_dist, crop_dist, height)
    elif width < height:
        crop_dist = (1.0 - (width / height)) / 2.0
        x_lin = torch.linspace(crop_dist, 1.0 - crop_dist, width)
        y_lin = torch.linspace(1.0, 0.0, height)
    else:
        x_lin = torch.linspace(0.0, 1.0, width)
        y_lin = torch.linspace(1.0, 0.0, height)

    yy, xx = torch.meshgrid(y_lin, x_lin, indexing="ij")
    return torch.stack([xx, yy], dim=0)  # [2, H, W]


def center_crop_pil(img: Image.Image, new_width: int, new_height: int) -> Image.Image:
    width, height = img.size
    left = max((width - new_width) // 2, 0)
    top = max((height - new_height) // 2, 0)
    right = left + new_width
    bottom = top + new_height
    return img.crop((left, top, right, bottom))


def crop_all_to_common_min_size(
    source: Image.Image,
    depth: Image.Image,
    target: Image.Image,
    mask: Optional[Image.Image] = None,
    softmask: Optional[Image.Image] = None,
) -> Tuple[Image.Image, Image.Image, Image.Image, Optional[Image.Image], Optional[Image.Image]]:
    widths = [source.size[0], depth.size[0], target.size[0]]
    heights = [source.size[1], depth.size[1], target.size[1]]

    if mask is not None:
        widths.append(mask.size[0])
        heights.append(mask.size[1])
    if softmask is not None:
        widths.append(softmask.size[0])
        heights.append(softmask.size[1])

    new_w = min(widths)
    new_h = min(heights)

    source = center_crop_pil(source, new_w, new_h)
    depth = center_crop_pil(depth, new_w, new_h)
    target = center_crop_pil(target, new_w, new_h)
    mask = center_crop_pil(mask, new_w, new_h) if mask is not None else None
    softmask = center_crop_pil(softmask, new_w, new_h) if softmask is not None else None

    return source, depth, target, mask, softmask


def crop_to_divisible_tensor(x: Tensor, divisor: int) -> Tensor:
    """
    x: [C, H, W]
    """
    _, h, w = x.shape
    new_h = h - (h % divisor)
    new_w = w - (w % divisor)

    top = max((h - new_h) // 2, 0)
    left = max((w - new_w) // 2, 0)

    return x[:, top:top + new_h, left:left + new_w]


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_gray(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def align_rgb_mean_to_reference(
    source: Image.Image,
    reference: Image.Image,
    eps: float = 1e-6,
) -> Image.Image:
    """
    Match the source image's global RGB mean to the reference image.

    This mirrors the EBB exposure alignment used by BokehDiff.
    """
    source_arr = np.asarray(source, dtype=np.float32) / 255.0
    reference_arr = np.asarray(reference, dtype=np.float32) / 255.0

    source_mean = source_arr.mean(axis=(0, 1), keepdims=True)
    reference_mean = reference_arr.mean(axis=(0, 1), keepdims=True)
    modifier = reference_mean / np.clip(source_mean, eps, 1.0)

    aligned = np.clip(source_arr * modifier, 0.0, 1.0)
    aligned_uint8 = np.round(aligned * 255.0).astype(np.uint8)
    return Image.fromarray(aligned_uint8)


def normalize_depth_tensor(depth: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Normalize depth to [0, 1] per image.
    depth: [1, H, W]
    """
    d_min = depth.amin(dim=(1, 2), keepdim=True)
    d_max = depth.amax(dim=(1, 2), keepdim=True)
    return (depth - d_min) / (d_max - d_min + eps)


class ExplicitBokehDataset(Dataset):
    """
    Dataset for either of these directory structures:

    VATD-style:
        root/
            train/
                input/
                depth/
                1_8/
                2_8/
                8/
                masks/       (optional)
                softmasks/   (optional)

    EBB-style:
        root/
            train/
                original/
                depth/
                bokeh/
                masks/       (optional)
                softmasks/   (optional)

    Files with the same basename are treated as the same scene.
    """

    TARGET_FOLDER_PATTERN = re.compile(r"^\d+(?:_\d+)?$")
    VALID_LAYOUTS = {"auto", "vatd", "ebb"}
    EBB_F_NUMBER = 1.8

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        divisor: int = 4,
        normalize_depth: bool = True,
        include_masks: bool = True,
        return_all_targets_per_scene: bool = False,
        layout: str = "auto",
        align_source_to_target_mean: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.divisor = divisor
        self.normalize_depth = normalize_depth
        self.include_masks = include_masks
        self.return_all_targets_per_scene = return_all_targets_per_scene

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.layout = self._resolve_layout(layout)
        self.align_source_to_target_mean = (
            self.layout == "ebb" if align_source_to_target_mean is None else bool(align_source_to_target_mean)
        )

        self.input_dir = self.split_dir / ("input" if self.layout == "vatd" else "original")
        self.depth_dir = self.split_dir / "depth"
        self.mask_dir = self.split_dir / "masks"
        self.softmask_dir = self.split_dir / "softmasks"

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Missing input directory: {self.input_dir}")
        if not self.depth_dir.exists():
            raise FileNotFoundError(f"Missing depth directory: {self.depth_dir}")

        self.target_dirs = self._find_target_dirs()
        if len(self.target_dirs) == 0:
            raise FileNotFoundError(f"No aperture target folders found under {self.split_dir}")

        self.records = self._build_records()

    def _resolve_layout(self, layout: str) -> str:
        layout = str(layout).lower()
        if layout not in self.VALID_LAYOUTS:
            raise ValueError(f"Unsupported layout '{layout}'. Expected one of {sorted(self.VALID_LAYOUTS)}")

        has_vatd = (
            (self.split_dir / "input").exists()
            and (self.split_dir / "depth").exists()
            and len(self._find_vatd_target_dirs()) > 0
        )
        has_ebb = (
            (self.split_dir / "original").exists()
            and (self.split_dir / "depth").exists()
            and (self.split_dir / "bokeh").exists()
        )

        if layout == "vatd":
            if not has_vatd:
                raise FileNotFoundError(f"Split {self.split_dir} does not match VATD layout")
            return layout

        if layout == "ebb":
            if not has_ebb:
                raise FileNotFoundError(f"Split {self.split_dir} does not match EBB layout")
            return layout

        if has_vatd and has_ebb:
            raise RuntimeError(
                f"Split {self.split_dir} matches both VATD and EBB layouts; "
                "pass layout='vatd' or layout='ebb' explicitly."
            )
        if has_vatd:
            return "vatd"
        if has_ebb:
            return "ebb"

        raise FileNotFoundError(
            f"Could not detect dataset layout under {self.split_dir}. "
            "Expected VATD-style directories (input/depth/<f-number>) or "
            "EBB-style directories (original/depth/bokeh)."
        )

    def _find_vatd_target_dirs(self) -> List[Path]:
        target_dirs: List[Path] = []
        for p in sorted(self.split_dir.iterdir()):
            if p.is_dir() and self.TARGET_FOLDER_PATTERN.match(p.name):
                target_dirs.append(p)
        return target_dirs

    def _find_target_dirs(self) -> List[Path]:
        if self.layout == "vatd":
            return self._find_vatd_target_dirs()
        return [self.split_dir / "bokeh"]

    def _collect_files(self, folder: Path) -> Dict[str, Path]:
        files = {}
        for p in sorted(folder.iterdir()):
            if p.is_file():
                files[p.stem] = p
        return files

    def _build_records(self) -> List[SampleRecord]:
        input_files = self._collect_files(self.input_dir)
        depth_files = self._collect_files(self.depth_dir)
        mask_files = self._collect_files(self.mask_dir) if (self.include_masks and self.mask_dir.exists()) else {}
        softmask_files = self._collect_files(self.softmask_dir) if (self.include_masks and self.softmask_dir.exists()) else {}

        records: List[SampleRecord] = []

        for target_dir in self.target_dirs:
            f_number = (
                parse_fnumber_from_folder(target_dir.name)
                if self.layout == "vatd"
                else self.EBB_F_NUMBER
            )
            target_files = self._collect_files(target_dir)

            common_ids = sorted(set(input_files) & set(depth_files) & set(target_files))
            if len(common_ids) == 0:
                continue

            for scene_id in common_ids:
                records.append(
                    SampleRecord(
                        scene_id=scene_id,
                        source_path=input_files[scene_id],
                        depth_path=depth_files[scene_id],
                        target_path=target_files[scene_id],
                        f_number=f_number,
                        mask_path=mask_files.get(scene_id, None),
                        softmask_path=softmask_files.get(scene_id, None),
                    )
                )

        if len(records) == 0:
            raise RuntimeError(f"No valid samples found in {self.split_dir}")

        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_sample_images(
        self,
        record: SampleRecord,
    ) -> Tuple[Image.Image, Image.Image, Image.Image, Optional[Image.Image], Optional[Image.Image]]:
        source = load_rgb(record.source_path)
        depth = load_gray(record.depth_path)
        target = load_rgb(record.target_path)

        mask = load_gray(record.mask_path) if record.mask_path is not None else None
        softmask = load_gray(record.softmask_path) if record.softmask_path is not None else None

        source, depth, target, mask, softmask = crop_all_to_common_min_size(
            source, depth, target, mask, softmask
        )

        return source, depth, target, mask, softmask

    def __getitem__(self, index: int) -> Dict[str, Tensor | float | str]:
        record = self.records[index]
        source_pil, depth_pil, target_pil, mask_pil, softmask_pil = self._load_sample_images(record)

        if self.align_source_to_target_mean:
            source_pil = align_rgb_mean_to_reference(source_pil, target_pil)

        source = to_tensor(source_pil)          # [3, H, W], [0,1]
        depth = to_tensor(depth_pil)[:1]        # [1, H, W], [0,1] if PNG/JPG grayscale
        target = to_tensor(target_pil)          # [3, H, W], [0,1]

        if self.normalize_depth:
            depth = normalize_depth_tensor(depth)

        source = crop_to_divisible_tensor(source, self.divisor)
        depth = crop_to_divisible_tensor(depth, self.divisor)
        target = crop_to_divisible_tensor(target, self.divisor)

        mask = None
        if mask_pil is not None:
            mask = to_tensor(mask_pil)[:1]
            mask = crop_to_divisible_tensor(mask, self.divisor)

        softmask = None
        if softmask_pil is not None:
            softmask = to_tensor(softmask_pil)[:1]
            softmask = crop_to_divisible_tensor(softmask, self.divisor)

        _, h, w = source.shape
        pos_map = build_pos_map(w, h)

        sample = {
            "scene_id": record.scene_id,
            "source": source,
            "depth": depth,
            "target": target,
            "f_number": torch.tensor(record.f_number, dtype=torch.float32),
            "pos_map": pos_map,
        }

        if mask is not None:
            sample["mask"] = mask
        if softmask is not None:
            sample["softmask"] = softmask

        return sample
