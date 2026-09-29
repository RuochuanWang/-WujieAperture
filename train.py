from __future__ import annotations

import argparse
import copy
from functools import partial
import inspect
import json
import random
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
try:
    from torch.amp import GradScaler, autocast as _autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast as _autocast

    _AMP_HAS_DEVICE_ARG = False
else:
    _AMP_HAS_DEVICE_ARG = True
from torch.utils.data import DataLoader, WeightedRandomSampler, default_collate
from torchvision.transforms.functional import to_pil_image

from dataset.explicit_bokeh_dataset import ExplicitBokehDataset
from method.config import bokehlicious_size_builder
from method.corrector import BokehCorrector
from method.explicit_bokeh_net import ExplicitBokehNet
from method.explicit_renderer import RelativeExplicitRenderer
from method.losses import ExplicitBokehLoss
from method.repair_prior import LocalRepairPrior

try:
    import yaml
except ImportError as e:
    raise ImportError("Please install pyyaml first: pip install pyyaml") from e


def amp_autocast(device_type: str, dtype: torch.dtype, enabled: bool):
    if _AMP_HAS_DEVICE_ARG:
        return _autocast(device_type=device_type, dtype=dtype, enabled=enabled)
    return _autocast(dtype=dtype, enabled=enabled and device_type == "cuda")


def build_grad_scaler(enabled: bool) -> GradScaler:
    if _AMP_HAS_DEVICE_ARG:
        return GradScaler(device="cuda", enabled=enabled)
    return GradScaler(enabled=enabled)


# =========================
# Config / parsing
# =========================

def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a dict at top level: {config_path}")
    return cfg


def build_parser(defaults: Optional[dict] = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser("Formal training for ExplicitBokehNet")
    parser.add_argument("--config", type=str, default=defaults.get("config", ""))
    parser.add_argument("--resume", type=str, default=defaults.get("resume", ""))
    return parser


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="")
    pre_parser.add_argument("--resume", type=str, default="")
    pre_args, _ = pre_parser.parse_known_args()

    defaults = {}
    if pre_args.config:
        defaults = load_yaml_config(pre_args.config)
        defaults["config"] = pre_args.config
    if pre_args.resume:
        defaults["resume"] = pre_args.resume

    parser = build_parser(defaults=defaults)
    return parser.parse_args()


def validate_config(cfg: dict):
    required_top = [
        "experiment_name",
        "root",
        "train_split",
        "val_split",
        "size",
        "num_workers",
        "base_scale",
        "max_radius",
        "num_bins",
        "lambda_rec",
        "lambda_coarse",
        "lambda_gate",
        "lambda_preserve",
        "weight_decay",
        "grad_clip",
        "seed",
        "save_dir",
        "use_softmask_for_rec",
        "use_mask_for_rec",
        "stages",
    ]
    for k in required_top:
        if k not in cfg:
            raise KeyError(f"Missing required config key: '{k}'")

    if not isinstance(cfg["stages"], list) or len(cfg["stages"]) == 0:
        raise ValueError("Config must contain a non-empty list: 'stages'")

    stage_required = ["name", "epochs", "batch_size", "crop_h", "crop_w", "lr"]
    for i, st in enumerate(cfg["stages"]):
        for k in stage_required:
            if k not in st:
                raise KeyError(f"Missing required stage key '{k}' in stages[{i}]")

    if bool(cfg.get("require_fixed_renderer_calibration", False)):
        learnable_keys = (
            "learnable_kappa",
            "learnable_focus_bias",
            "learnable_source_strength",
        )
        enabled = [key for key in learnable_keys if bool(cfg.get(key, False))]
        if enabled:
            raise ValueError(
                "Fixed renderer calibration required, but these options are learnable: "
                + ", ".join(enabled)
            )

    unified_photo = bool(cfg.get("unified_photometric_calibration_enabled", False))
    if float(cfg.get("lambda_affine", 0.0)) > 0.0 and not unified_photo:
        raise ValueError("lambda_affine > 0 requires unified_photometric_calibration_enabled=true")


def normalize_aperture_weight_cfg(weight_cfg: Optional[dict]) -> Dict[float, float]:
    if not weight_cfg:
        return {}

    normalized = {}
    for k, v in weight_cfg.items():
        normalized[float(k)] = float(v)
    return normalized


def quantize_f_number_key(value: float) -> float:
    return round(float(value), 1)


# =========================
# Repro / utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


def random_crop_batch_spatial(batch: Dict[str, Any], crop_h: int, crop_w: int) -> Dict[str, Any]:
    spatial_keys = [k for k, v in batch.items() if torch.is_tensor(v) and v.dim() == 4]
    if len(spatial_keys) == 0:
        return batch

    ref = batch[spatial_keys[0]]
    _, _, h, w = ref.shape

    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop size ({crop_h}, {crop_w}) exceeds batch spatial size ({h}, {w})")

    top = random.randint(0, h - crop_h) if h > crop_h else 0
    left = random.randint(0, w - crop_w) if w > crop_w else 0

    cropped = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() == 4:
            cropped[k] = v[:, :, top:top + crop_h, left:left + crop_w]
        else:
            cropped[k] = v
    return cropped


def random_crop_sample_spatial(sample: Dict[str, Any], crop_h: int, crop_w: int) -> Dict[str, Any]:
    """Crop one variable-size sample before DataLoader tries to stack a batch."""
    spatial_keys = [k for k, v in sample.items() if torch.is_tensor(v) and v.dim() == 3]
    if len(spatial_keys) == 0:
        return sample

    ref = sample[spatial_keys[0]]
    _, h, w = ref.shape
    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop size ({crop_h}, {crop_w}) exceeds sample spatial size ({h}, {w})")

    top = random.randint(0, h - crop_h) if h > crop_h else 0
    left = random.randint(0, w - crop_w) if w > crop_w else 0
    cropped = {}
    for key, value in sample.items():
        if torch.is_tensor(value) and value.dim() == 3:
            cropped[key] = value[:, top:top + crop_h, left:left + crop_w]
        else:
            cropped[key] = value
    return cropped


def collate_random_crops(
    samples: List[Dict[str, Any]],
    crop_h: int,
    crop_w: int,
) -> Dict[str, Any]:
    cropped_samples = [random_crop_sample_spatial(sample, crop_h, crop_w) for sample in samples]
    return default_collate(cropped_samples)


def center_crop_batch_spatial(batch: Dict[str, Any], crop_h: int, crop_w: int) -> Dict[str, Any]:
    spatial_keys = [k for k, v in batch.items() if torch.is_tensor(v) and v.dim() == 4]
    if len(spatial_keys) == 0:
        return batch

    ref = batch[spatial_keys[0]]
    _, _, h, w = ref.shape

    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop size ({crop_h}, {crop_w}) exceeds batch spatial size ({h}, {w})")

    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)

    cropped = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() == 4:
            cropped[k] = v[:, :, top:top + crop_h, left:left + crop_w]
        else:
            cropped[k] = v
    return cropped


def maybe_center_crop_batch_spatial(batch: Dict[str, Any], crop_h: int, crop_w: int) -> Dict[str, Any]:
    if crop_h > 0 and crop_w > 0:
        return center_crop_batch_spatial(batch, crop_h, crop_w)
    return batch


def get_scene_id_for_log(batch: Dict[str, Any]) -> str:
    scene_id = batch["scene_id"]
    if isinstance(scene_id, (list, tuple)):
        return str(scene_id[0])
    return str(scene_id)


def make_vis_panel(batch: Dict[str, Tensor], out: Dict[str, Tensor], max_radius: float) -> Tensor:
    source = batch["source"][0].detach().float().cpu().clamp(0, 1)
    target = batch["target"][0].detach().float().cpu().clamp(0, 1)
    depth = batch["depth"][0].detach().float().cpu().clamp(0, 1)

    coarse = out["coarse"][0].detach().float().cpu().clamp(0, 1)
    output = out["output"][0].detach().float().cpu().clamp(0, 1)

    radius = out["radius_map"][0].detach().float().cpu().clamp(0, max_radius) / max(max_radius, 1e-6)
    repair = out["repair_map"][0].detach().float().cpu().clamp(0, 1)
    gate = out["gate"][0].detach().float().cpu().clamp(0, 1)

    depth_vis = depth.repeat(3, 1, 1)
    radius_vis = radius.repeat(3, 1, 1)
    repair_vis = repair.repeat(3, 1, 1)
    gate_vis = gate.repeat(3, 1, 1)

    panel = torch.cat(
        [source, depth_vis, radius_vis, repair_vis, gate_vis, coarse, output, target],
        dim=2
    )
    return panel


def save_vis_panel(batch: Dict[str, Tensor], out: Dict[str, Tensor], save_path: Path, max_radius: float):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    panel = make_vis_panel(batch, out, max_radius=max_radius)
    img = to_pil_image(panel)
    img.save(str(save_path))


# =========================
# Metrics (self-contained)
# =========================

def _check_4d(x: Tensor, name: str):
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.dim() != 4:
        raise ValueError(f"{name} must be 4D [B,C,H,W], got shape={tuple(x.shape)}")


def _make_gaussian_kernel(
    window_size: int,
    sigma: float,
    channels: int,
    device,
    dtype,
) -> Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()

    kernel_2d = g[:, None] * g[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()

    kernel = kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    return kernel


def psnr_batch(pred: Tensor, target: Tensor, data_range: float = 1.0, eps: float = 1e-8) -> Tensor:
    _check_4d(pred, "pred")
    _check_4d(target, "target")
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))


def ssim_batch(
    pred: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-12,
) -> Tensor:
    _check_4d(pred, "pred")
    _check_4d(target, "target")
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")

    _, c, _, _ = pred.shape
    kernel = _make_gaussian_kernel(window_size, sigma, c, pred.device, pred.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(pred, kernel, padding=padding, groups=c)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=c)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, kernel, padding=padding, groups=c) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, kernel, padding=padding, groups=c) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, kernel, padding=padding, groups=c) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + eps
    )
    return ssim_map.flatten(1).mean(dim=1)


def image_metrics_batch(pred: Tensor, target: Tensor, data_range: float = 1.0) -> Dict[str, float]:
    psnr = psnr_batch(pred, target, data_range=data_range)
    ssim = ssim_batch(pred, target, data_range=data_range)
    return {
        "psnr_mean": float(psnr.mean().item()),
        "ssim_mean": float(ssim.mean().item()),
    }


def compute_quality_metrics(out: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Dict[str, float]:
    coarse = out["coarse"].detach().float().clamp(0.0, 1.0)
    output = out["output"].detach().float().clamp(0.0, 1.0)
    target = batch["target"].detach().float().clamp(0.0, 1.0)

    coarse_err = float((coarse - target).abs().mean().item())
    output_err = float((output - target).abs().mean().item())
    improve = coarse_err - output_err

    coarse_metrics = image_metrics_batch(coarse, target)
    output_metrics = image_metrics_batch(output, target)

    return {
        "coarse_err": coarse_err,
        "output_err": output_err,
        "improve": improve,
        "coarse_psnr": coarse_metrics["psnr_mean"],
        "coarse_ssim": coarse_metrics["ssim_mean"],
        "output_psnr": output_metrics["psnr_mean"],
        "output_ssim": output_metrics["ssim_mean"],
    }


def tensor_mean_or_zero(value: Optional[Tensor]) -> float:
    if value is None:
        return 0.0
    return float(value.detach().mean().item())


def tensor_tree_is_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value.detach().float()).all().item())
    if isinstance(value, dict):
        return all(tensor_tree_is_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensor_tree_is_finite(child) for child in value)
    return True


def first_nonfinite_name(name: str, value: Any) -> Optional[str]:
    if torch.is_tensor(value):
        finite = bool(torch.isfinite(value.detach().float()).all().item())
        return None if finite else name
    if isinstance(value, dict):
        for key, child in value.items():
            found = first_nonfinite_name(f"{name}.{key}", child)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = first_nonfinite_name(f"{name}[{index}]", child)
            if found is not None:
                return found
    return None


def gradients_are_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is not None and not bool(torch.isfinite(param.grad.detach().float()).all().item()):
            return False
    return True


def first_nonfinite_gradient_summary(model: nn.Module) -> Optional[str]:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        finite = torch.isfinite(grad)
        if bool(finite.all().item()):
            continue
        nan_count = int(torch.isnan(grad).sum().item())
        inf_count = int(torch.isinf(grad).sum().item())
        finite_grad = grad[finite]
        if finite_grad.numel() > 0:
            finite_min = float(finite_grad.min().item())
            finite_max = float(finite_grad.max().item())
        else:
            finite_min = float("nan")
            finite_max = float("nan")
        return (
            f"{name} shape={tuple(param.shape)} nan={nan_count} inf={inf_count} "
            f"finite_min={finite_min:.6e} finite_max={finite_max:.6e}"
        )
    return None


def first_nonfinite_parameter_name(model: nn.Module) -> Optional[str]:
    for name, param in model.named_parameters():
        if not bool(torch.isfinite(param.detach().float()).all().item()):
            return name
    return None


def clone_eval_model(model: nn.Module, cfg: dict, device: torch.device) -> nn.Module:
    ema_model = build_model(cfg, device)
    ema_model.load_state_dict(model.state_dict(), strict=True)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def load_initial_model_weights(
    model: nn.Module,
    checkpoint: Dict[str, Any],
    weights: str,
) -> tuple[str, List[str]]:
    weights = str(weights).lower()
    if weights not in {"raw", "ema"}:
        raise ValueError("init_weights must be 'raw' or 'ema'")

    state_key = "ema_model" if weights == "ema" else "model"
    if state_key not in checkpoint:
        raise ValueError(f"Initial checkpoint has no {state_key!r} state dict")

    incompatible = model.load_state_dict(checkpoint[state_key], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Initial checkpoint is not architecture-compatible. "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return weights, list(incompatible.missing_keys)


def should_use_ema_model(ema_model: Optional[nn.Module], global_epoch: int, ema_start_epoch: int) -> bool:
    return ema_model is not None and global_epoch >= ema_start_epoch


@torch.no_grad()
def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float):
    decay = float(decay)
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()

    for key, ema_value in ema_state.items():
        model_value = model_state[key].detach()
        if not torch.is_floating_point(ema_value):
            ema_value.copy_(model_value)
            continue
        ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)


# =========================
# Model / optimizer / io
# =========================

def build_corrector(
    size: str = "small",
    in_chans: int = 9,
    use_checkpoints: Optional[bool] = None,
    extra_kwargs: Optional[dict] = None,
) -> BokehCorrector:
    cfg = bokehlicious_size_builder(size)
    cfg["in_chans"] = int(in_chans)
    if use_checkpoints is not None:
        cfg["use_checkpoints"] = [bool(use_checkpoints) for _ in cfg["use_checkpoints"]]
    if extra_kwargs:
        cfg.update(extra_kwargs)

    valid_keys = set(inspect.signature(BokehCorrector.__init__).parameters.keys())
    valid_keys.discard("self")
    filtered_cfg = {k: v for k, v in cfg.items() if k in valid_keys}
    return BokehCorrector(**filtered_cfg)


def build_renderer(cfg: dict, device: torch.device) -> RelativeExplicitRenderer:
    """Build only the explicit physical renderer from a training config."""
    source_strength_init = float(cfg.get("source_strength_init", 0.125))
    source_f_number = cfg.get("source_f_number")
    if source_f_number is not None:
        source_f_number = float(source_f_number)
        if source_f_number <= 0.0:
            raise ValueError("Config key 'source_f_number' must be positive")
        derived_source_strength = 1.0 / source_f_number
        if "source_strength_init" in cfg and abs(source_strength_init - derived_source_strength) > 1e-6:
            raise ValueError(
                "source_strength_init must equal 1/source_f_number: "
                f"got {source_strength_init} vs {derived_source_strength}"
            )
        source_strength_init = derived_source_strength

    return RelativeExplicitRenderer(
        ref_f_number=float(cfg.get("ref_f_number", 8.0)),
        base_scale=float(cfg["base_scale"]),
        focal_length=float(cfg.get("focal_length", 1.0)),
        max_radius=float(cfg["max_radius"]),
        num_bins=int(cfg["num_bins"]),
        renderer_kernel=str(cfg.get("renderer_kernel", "gaussian")),
        renderer_padding=str(cfg.get("renderer_padding", "zero")),
        renderer_mode=str(cfg.get("renderer_mode", "gather")),
        renderer_linear_light=bool(cfg.get("renderer_linear_light", False)),
        renderer_depth_layers=int(cfg.get("renderer_depth_layers", 12)),
        acceptable_coc_radius=float(cfg.get("acceptable_coc_radius", 0.0)),
        coc_transition_width=float(cfg.get("coc_transition_width", 0.0)),
        depth_guidance_enabled=bool(cfg.get("depth_guidance_enabled", False)),
        depth_guidance_iterations=int(cfg.get("depth_guidance_iterations", 1)),
        depth_guidance_color_sigma=float(cfg.get("depth_guidance_color_sigma", 0.12)),
        depth_guidance_depth_sigma=float(cfg.get("depth_guidance_depth_sigma", 0.08)),
        depth_guidance_blend=float(cfg.get("depth_guidance_blend", 0.5)),
        layered_radius_interpolation=bool(cfg.get("layered_radius_interpolation", False)),
        layered_coverage_mode=str(cfg.get("layered_coverage_mode", "normalize")),
        radiance_weighting_strength=float(cfg.get("radiance_weighting_strength", 0.0)),
        radiance_weighting_kernel=int(cfg.get("radiance_weighting_kernel", 15)),
        radiance_weighting_threshold=float(cfg.get("radiance_weighting_threshold", 0.15)),
        radiance_weighting_gamma=float(cfg.get("radiance_weighting_gamma", 1.0)),
        highlight_recovery_strength=float(cfg.get("highlight_recovery_strength", 0.0)),
        highlight_recovery_threshold=float(cfg.get("highlight_recovery_threshold", 0.75)),
        highlight_recovery_gamma=float(cfg.get("highlight_recovery_gamma", 2.0)),
        focus_mode=str(cfg.get("focus_mode", "center_weighted_median")),
        focus_patch_ratio=float(cfg.get("focus_patch_ratio", 0.25)),
        trim_ratio=float(cfg.get("trim_ratio", 0.10)),
        inverse_depth_floor=float(cfg.get("inverse_depth_floor", 1.0)),
        depth_representation=str(cfg.get("depth_representation", "metric_depth")),
        learnable_kappa=bool(cfg.get("learnable_kappa", True)),
        learnable_focus_bias=bool(cfg.get("learnable_focus_bias", True)),
        learnable_source_strength=bool(cfg.get("learnable_source_strength", True)),
        source_strength_init=source_strength_init,
    ).to(device)


def build_model(cfg: dict, device: torch.device) -> ExplicitBokehNet:
    corrector = build_corrector(
        cfg["size"],
        in_chans=9,
        use_checkpoints=cfg.get("corrector_use_checkpoints"),
        extra_kwargs={
            "use_coc_modulation": bool(cfg.get("use_coc_modulation", True)),
            "coc_condition_chans": int(cfg.get("coc_condition_chans", 2)),
            "coc_mod_hidden_chans": int(cfg.get("coc_mod_hidden_chans", 16)),
            "use_time_modulation": bool(cfg.get("use_time_modulation", True)),
            "time_mod_hidden_chans": int(cfg.get("time_mod_hidden_chans", 64)),
            "use_prior_gate": bool(cfg.get("use_prior_gate", True)),
            "prior_gate_lambda": float(cfg.get("prior_gate_lambda", 1.0)),
            "use_aperture_gate": bool(cfg.get("use_aperture_gate", False)),
            "aperture_gate_init_offset": float(cfg.get("aperture_gate_init_offset", 0.0)),
            "aperture_gate_init_slope": float(cfg.get("aperture_gate_init_slope", 0.0)),
        },
    ).to(device)
    if "corrector_gate_bias" in cfg:
        corrector.out_stage.bias.data[3] = float(cfg["corrector_gate_bias"])

    renderer = build_renderer(cfg, device)

    repair_prior = LocalRepairPrior(
        alpha=float(cfg.get("repair_prior_alpha", 4.0)),
        beta=float(cfg.get("repair_prior_beta", 2.0)),
        gamma=float(cfg.get("repair_prior_gamma", 0.0)),
        eta=float(cfg.get("repair_prior_eta", 0.0)),
        tau=float(cfg.get("repair_prior_tau", 2.5)),
        max_radius=float(cfg["max_radius"]),
        normalize_grad=bool(cfg.get("repair_prior_normalize_grad", True)),
        aperture_aware=bool(cfg.get("repair_prior_aperture_aware", False)),
        learnable_core=bool(cfg.get("repair_prior_learnable_core", True)),
    ).to(device)

    model = ExplicitBokehNet(
        corrector=corrector,
        renderer=renderer,
        repair_prior=repair_prior,
        use_signed_delta=False,
        normalize_radius_for_input=True,
        unified_photometric_calibration_enabled=bool(
            cfg.get("unified_photometric_calibration_enabled", False)
        ),
        unified_photometric_hidden_chans=int(cfg.get("unified_photometric_hidden_chans", 64)),
        unified_photometric_max_log_gain=float(cfg.get("unified_photometric_max_log_gain", 0.20)),
        unified_photometric_max_bias=float(cfg.get("unified_photometric_max_bias", 0.10)),
        flow_noise_std=float(cfg.get("flow_noise_std", 0.02)),
        flow_zero_prob=float(cfg.get("flow_zero_prob", 0.5)),
        flow_use_bridge_noise=bool(cfg.get("flow_use_bridge_noise", True)),
        bound_residual_proposal=bool(cfg.get("bound_residual_proposal", False)),
        residual_proposal_scale=float(cfg.get("residual_proposal_scale", 1.0)),
    ).to(device)
    return model


def build_aperture_sampler(train_ds, aperture_sampling_weights: Dict[float, float]):
    if not aperture_sampling_weights:
        return None

    sample_weights = []
    for record in train_ds.records:
        weight = aperture_sampling_weights.get(quantize_f_number_key(record.f_number), 1.0)
        sample_weights.append(weight)

    if len(sample_weights) == 0:
        return None

    if max(sample_weights) - min(sample_weights) < 1e-8:
        return None

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_train_loader(
    train_ds,
    batch_size: int,
    num_workers: int,
    aperture_sampling_weights: Optional[Dict[float, float]] = None,
    pre_collate_crop: Optional[tuple[int, int]] = None,
) -> DataLoader:
    sampler = build_aperture_sampler(train_ds, aperture_sampling_weights or {})
    collate_fn = None
    if pre_collate_crop is not None:
        crop_h, crop_w = pre_collate_crop
        collate_fn = partial(collate_random_crops, crop_h=int(crop_h), crop_w=int(crop_w))
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )


def build_val_loader(val_ds, num_workers: int) -> DataLoader:
    return DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def compute_epoch_lr(
    epoch_in_stage: int,
    stage_epochs: int,
    lr_max: float,
    lr_min: float,
    scheduler_name: str,
    warmup_epochs: int,
) -> float:
    if scheduler_name == "none" or stage_epochs <= 0:
        return lr_max

    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    warmup_epochs = max(int(warmup_epochs), 0)
    if warmup_epochs > 0 and epoch_in_stage < warmup_epochs:
        # Leave room for the first post-warmup epoch to reach lr_max.
        warmup_progress = float(epoch_in_stage + 1) / float(warmup_epochs + 1)
        return lr_min + (lr_max - lr_min) * warmup_progress

    cosine_epochs = max(stage_epochs - warmup_epochs, 1)
    cosine_pos = max(epoch_in_stage - warmup_epochs, 0)
    progress = float(cosine_pos) / float(max(cosine_epochs - 1, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine


def build_optimizer(model: ExplicitBokehNet, cfg: dict) -> torch.optim.Optimizer:
    optimizer_name = str(cfg.get("optimizer", "adam")).lower()
    betas_cfg = cfg.get("betas", [0.9, 0.999])
    if not isinstance(betas_cfg, (list, tuple)) or len(betas_cfg) != 2:
        raise ValueError("Config key 'betas' must be a list/tuple with length 2")
    betas = (float(betas_cfg[0]), float(betas_cfg[1]))

    first_stage_lr = float(cfg["stages"][0]["lr"])
    weight_decay = float(cfg["weight_decay"])

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=first_stage_lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=first_stage_lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def save_checkpoint(
    save_path: Path,
    model: ExplicitBokehNet,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    stage_idx: int,
    epoch_in_stage: int,
    global_epoch: int,
    global_step: int,
    best_val_output_psnr: float,
    best_full_output_psnr: Optional[float] = None,
    best_crop_output_psnr: Optional[float] = None,
    best_full_output_ssim: Optional[float] = None,
    ema_model: Optional[nn.Module] = None,
    scaler: Optional[GradScaler] = None,
    epoch_complete: bool = False,
):
    bad_param = first_nonfinite_parameter_name(model)
    if bad_param is not None:
        raise RuntimeError(f"Refusing to save checkpoint with non-finite model parameter: {bad_param}")
    if ema_model is not None:
        bad_ema_param = first_nonfinite_parameter_name(ema_model)
        if bad_ema_param is not None:
            raise RuntimeError(f"Refusing to save checkpoint with non-finite EMA parameter: {bad_ema_param}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
        "stage_idx": stage_idx,
        "epoch_in_stage": epoch_in_stage,
        "global_epoch": global_epoch,
        "global_step": global_step,
        "best_val_output_psnr": best_val_output_psnr,
        "epoch_complete": bool(epoch_complete),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if best_full_output_psnr is not None:
        ckpt["best_full_output_psnr"] = best_full_output_psnr
    if best_crop_output_psnr is not None:
        ckpt["best_crop_output_psnr"] = best_crop_output_psnr
    if best_full_output_ssim is not None:
        ckpt["best_full_output_ssim"] = best_full_output_ssim
    if ema_model is not None:
        ckpt["ema_model"] = ema_model.state_dict()
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, str(save_path))


# =========================
# Validation
# =========================

@torch.no_grad()
def run_validation(
    model: ExplicitBokehNet,
    criterion: ExplicitBokehLoss,
    val_loader: DataLoader,
    device: torch.device,
    crop_h: int = 0,
    crop_w: int = 0,
    max_samples: int = 0,
    balance_apertures: bool = False,
) -> Dict[str, Any]:
    model.eval()

    loss_total_sum = 0.0
    loss_coarse_sum = 0.0
    loss_rec_sum = 0.0
    loss_fm_sum = 0.0
    loss_mse_sum = 0.0
    loss_ssim_sum = 0.0
    loss_gate_sum = 0.0
    loss_gate_sup_sum = 0.0
    loss_preserve_sum = 0.0
    loss_endpoint_sum = 0.0
    loss_affine_sum = 0.0
    loss_photo_sum = 0.0
    gate_target_mean_sum = 0.0
    coarse_err_sum = 0.0
    output_err_sum = 0.0
    improve_sum = 0.0
    coarse_psnr_sum = 0.0
    output_psnr_sum = 0.0
    coarse_ssim_sum = 0.0
    output_ssim_sum = 0.0
    kappa_sum = 0.0
    aperture_radius_sum = 0.0
    lens_strength_sum = 0.0
    aperture_scale_sum = 0.0
    source_strength_sum = 0.0
    photo_gain_sum = 0.0
    photo_bias_abs_sum = 0.0
    gate_mean_sum = 0.0
    aperture_gate_bias_sum = 0.0
    proposal_l1_sum = 0.0
    applied_l1_sum = 0.0
    count = 0

    aperture_limits: dict[float, int] = {}
    aperture_selected: dict[float, int] = defaultdict(int)
    if balance_apertures and max_samples > 0 and hasattr(val_loader.dataset, "records"):
        apertures = sorted({
            quantize_f_number_key(record.f_number)
            for record in val_loader.dataset.records
        })
        if apertures:
            quotient, remainder = divmod(max_samples, len(apertures))
            aperture_limits = {
                aperture: quotient + (1 if index < remainder else 0)
                for index, aperture in enumerate(apertures)
            }

    aperture_metric_sums: dict[float, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "coarse_psnr": 0.0,
            "output_psnr": 0.0,
            "coarse_ssim": 0.0,
            "output_ssim": 0.0,
            "gate_mean": 0.0,
            "aperture_gate_bias": 0.0,
            "gate_target_mean": 0.0,
            "proposal_l1": 0.0,
            "applied_l1": 0.0,
        }
    )

    for batch in val_loader:
        if max_samples > 0 and count >= max_samples:
            break
        aperture_key = quantize_f_number_key(float(batch["f_number"][0].item()))
        if aperture_limits and aperture_selected[aperture_key] >= aperture_limits.get(aperture_key, 0):
            continue
        aperture_selected[aperture_key] += 1
        batch = move_batch_to_device(batch, device)
        batch = maybe_center_crop_batch_spatial(batch, crop_h, crop_w)

        out = model(
            source=batch["source"],
            depth=batch["depth"],
            f_number=batch["f_number"],
            pos_map=batch["pos_map"],
            mask=batch.get("mask"),
            target=batch["target"],
        )

        losses = criterion(out, batch)
        metrics = compute_quality_metrics(out, batch)

        loss_total_sum += float(losses["loss_total"].item())
        loss_coarse_sum += float(losses["loss_coarse"].item())
        loss_rec_sum += float(losses["loss_rec"].item())
        loss_fm_sum += float(losses["loss_fm"].item())
        loss_mse_sum += float(losses["loss_mse"].item())
        loss_ssim_sum += float(losses["loss_ssim"].item())
        loss_gate_sum += float(losses["loss_gate"].item())
        loss_gate_sup_sum += float(losses["loss_gate_sup"].item())
        loss_preserve_sum += float(losses["loss_preserve"].item())
        loss_endpoint_sum += float(losses["loss_endpoint"].item())
        loss_affine_sum += float(losses["loss_affine"].item())
        loss_photo_sum += float(losses["loss_photo"].item())
        gate_target_mean_sum += float(losses["gate_target_mean"].item())
        coarse_err_sum += metrics["coarse_err"]
        output_err_sum += metrics["output_err"]
        improve_sum += metrics["improve"]
        coarse_psnr_sum += metrics["coarse_psnr"]
        output_psnr_sum += metrics["output_psnr"]
        coarse_ssim_sum += metrics["coarse_ssim"]
        output_ssim_sum += metrics["output_ssim"]
        kappa_sum += tensor_mean_or_zero(out.get("kappa"))
        aperture_radius_sum += tensor_mean_or_zero(out.get("aperture_radius"))
        lens_strength_sum += tensor_mean_or_zero(out.get("lens_strength"))
        aperture_scale_sum += tensor_mean_or_zero(out.get("aperture_scale"))
        source_strength_sum += tensor_mean_or_zero(out.get("source_strength"))
        photo_gain_sum += tensor_mean_or_zero(out.get("photo_gain"))
        photo_bias = out.get("photo_bias")
        photo_bias_abs_sum += 0.0 if photo_bias is None else float(photo_bias.detach().abs().mean().item())
        gate_mean = float(out["gate"].detach().float().mean().item())
        aperture_gate_bias = tensor_mean_or_zero(out.get("aperture_gate_bias"))
        proposal_l1 = float(out["residual_proposal"].detach().float().abs().mean().item())
        applied_l1 = float(out["delta_rgb"].detach().float().abs().mean().item())
        gate_mean_sum += gate_mean
        aperture_gate_bias_sum += aperture_gate_bias
        proposal_l1_sum += proposal_l1
        applied_l1_sum += applied_l1
        aperture_stats = aperture_metric_sums[aperture_key]
        aperture_stats["count"] += 1.0
        aperture_stats["coarse_psnr"] += metrics["coarse_psnr"]
        aperture_stats["output_psnr"] += metrics["output_psnr"]
        aperture_stats["coarse_ssim"] += metrics["coarse_ssim"]
        aperture_stats["output_ssim"] += metrics["output_ssim"]
        aperture_stats["gate_mean"] += gate_mean
        aperture_stats["aperture_gate_bias"] += aperture_gate_bias
        aperture_stats["gate_target_mean"] += float(losses["gate_target_mean"].item())
        aperture_stats["proposal_l1"] += proposal_l1
        aperture_stats["applied_l1"] += applied_l1
        count += 1

    if count == 0:
        return {
            "val_loss_total": 0.0,
            "val_loss_coarse": 0.0,
            "val_loss_rec": 0.0,
            "val_loss_fm": 0.0,
            "val_loss_mse": 0.0,
            "val_loss_ssim": 0.0,
            "val_loss_gate": 0.0,
            "val_loss_gate_sup": 0.0,
            "val_loss_preserve": 0.0,
            "val_loss_endpoint": 0.0,
            "val_loss_affine": 0.0,
            "val_loss_photo": 0.0,
            "val_gate_target_mean": 0.0,
            "val_coarse_err": 0.0,
            "val_output_err": 0.0,
            "val_improve": 0.0,
            "val_coarse_psnr": 0.0,
            "val_output_psnr": 0.0,
            "val_coarse_ssim": 0.0,
            "val_output_ssim": 0.0,
            "val_kappa": 0.0,
            "val_aperture_radius": 0.0,
            "val_lens_strength": 0.0,
            "val_aperture_scale": 0.0,
            "val_source_strength": 0.0,
            "val_photo_gain": 1.0,
            "val_photo_bias_abs": 0.0,
            "val_gate_mean": 0.0,
            "val_aperture_gate_bias": 0.0,
            "val_proposal_l1": 0.0,
            "val_applied_l1": 0.0,
            "val_by_aperture": {},
        }

    by_aperture = {}
    for aperture_key, sums in sorted(aperture_metric_sums.items()):
        aperture_count = max(sums["count"], 1.0)
        by_aperture[f"f/{aperture_key:.1f}"] = {
            key: value / aperture_count
            for key, value in sums.items()
            if key != "count"
        }

    return {
        "val_loss_total": loss_total_sum / count,
        "val_loss_coarse": loss_coarse_sum / count,
        "val_loss_rec": loss_rec_sum / count,
        "val_loss_fm": loss_fm_sum / count,
        "val_loss_mse": loss_mse_sum / count,
        "val_loss_ssim": loss_ssim_sum / count,
        "val_loss_gate": loss_gate_sum / count,
        "val_loss_gate_sup": loss_gate_sup_sum / count,
        "val_loss_preserve": loss_preserve_sum / count,
        "val_loss_endpoint": loss_endpoint_sum / count,
        "val_loss_affine": loss_affine_sum / count,
        "val_loss_photo": loss_photo_sum / count,
        "val_gate_target_mean": gate_target_mean_sum / count,
        "val_coarse_err": coarse_err_sum / count,
        "val_output_err": output_err_sum / count,
        "val_improve": improve_sum / count,
        "val_coarse_psnr": coarse_psnr_sum / count,
        "val_output_psnr": output_psnr_sum / count,
        "val_coarse_ssim": coarse_ssim_sum / count,
        "val_output_ssim": output_ssim_sum / count,
        "val_kappa": kappa_sum / count,
        "val_aperture_radius": aperture_radius_sum / count,
        "val_lens_strength": lens_strength_sum / count,
        "val_aperture_scale": aperture_scale_sum / count,
        "val_source_strength": source_strength_sum / count,
        "val_photo_gain": photo_gain_sum / count,
        "val_photo_bias_abs": photo_bias_abs_sum / count,
        "val_gate_mean": gate_mean_sum / count,
        "val_aperture_gate_bias": aperture_gate_bias_sum / count,
        "val_proposal_l1": proposal_l1_sum / count,
        "val_applied_l1": applied_l1_sum / count,
        "val_by_aperture": by_aperture,
    }


# =========================
# Stage / epoch training
# =========================

def merge_stage_cfg(global_cfg: dict, stage_cfg: dict) -> dict:
    merged = copy.deepcopy(global_cfg)
    merged.update(stage_cfg)
    return merged


def rebuild_criterion_for_stage(global_cfg: dict, stage_cfg: dict) -> ExplicitBokehLoss:
    merged_cfg = merge_stage_cfg(global_cfg, stage_cfg)
    return ExplicitBokehLoss(
        lambda_rec=float(merged_cfg["lambda_rec"]),
        lambda_fm=float(merged_cfg.get("lambda_fm", 0.0)),
        lambda_mse=float(merged_cfg.get("lambda_mse", 0.0)),
        lambda_ssim=float(merged_cfg.get("lambda_ssim", 0.0)),
        lambda_coarse=float(merged_cfg.get("lambda_coarse", 0.0)),
        lambda_gate=float(merged_cfg["lambda_gate"]),
        lambda_gate_sup=float(merged_cfg.get("lambda_gate_sup", 0.0)),
        lambda_preserve=float(merged_cfg["lambda_preserve"]),
        lambda_endpoint=float(merged_cfg.get("lambda_endpoint", 0.0)),
        lambda_affine=float(merged_cfg.get("lambda_affine", 0.0)),
        lambda_photo=float(merged_cfg.get("lambda_photo", 0.0)),
        gate_target_threshold=float(merged_cfg.get("gate_target_threshold", 0.03)),
        gate_target_quantile=float(merged_cfg.get("gate_target_quantile", 0.75)),
        gate_target_kernel_size=int(merged_cfg.get("gate_target_kernel_size", 5)),
        affine_teacher_max_log_gain=float(
            merged_cfg.get(
                "affine_teacher_max_log_gain",
                merged_cfg.get("unified_photometric_max_log_gain", 0.20),
            )
        ),
        affine_teacher_max_bias=float(
            merged_cfg.get(
                "affine_teacher_max_bias",
                merged_cfg.get("unified_photometric_max_bias", 0.10),
            )
        ),
        ssim_max_size=int(merged_cfg.get("ssim_max_size", 512)),
        use_softmask_for_rec=bool(merged_cfg.get("use_softmask_for_rec", False)),
        use_mask_for_rec=bool(merged_cfg.get("use_mask_for_rec", False)),
        detach_repair_map=True,
        aperture_rec_weights=normalize_aperture_weight_cfg(merged_cfg.get("aperture_rec_weights")),
    )


def train_one_epoch(
    model: ExplicitBokehNet,
    criterion: ExplicitBokehLoss,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    stage_cfg: dict,
    global_cfg: dict,
    stage_idx: int,
    epoch_in_stage: int,
    global_epoch: int,
    global_step: int,
    vis_dir: Path,
    ckpt_dir: Path,
    best_val_output_psnr: float,
    best_full_output_psnr: float,
    best_crop_output_psnr: float,
    best_full_output_ssim: float,
    ema_model: Optional[nn.Module] = None,
    ema_decay: float = 0.999,
    ema_start_epoch: int = 0,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler: Optional[GradScaler] = None,
) -> int:
    model.train()

    crop_h = int(stage_cfg["crop_h"])
    crop_w = int(stage_cfg["crop_w"])
    print_every = int(stage_cfg.get("print_every", 100))
    vis_every = int(stage_cfg.get("vis_every", 1000))
    ckpt_every = int(stage_cfg.get("ckpt_every", 4000))
    grad_clip = float(global_cfg["grad_clip"])
    stage_name = str(stage_cfg["name"])
    fail_on_nonfinite = bool(stage_cfg.get("fail_on_nonfinite", global_cfg.get("fail_on_nonfinite", False)))
    accum_steps = max(int(stage_cfg.get("accum_steps", global_cfg.get("accum_steps", 1))), 1)
    max_batches = max(int(stage_cfg.get("max_batches_per_epoch", 0)), 0)
    batches_this_epoch = min(len(train_loader), max_batches) if max_batches > 0 else len(train_loader)

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= batches_this_epoch:
            break
        batch = random_crop_batch_spatial(batch, crop_h, crop_w)
        batch = move_batch_to_device(batch, device)
        scene_id_for_log = get_scene_id_for_log(batch)
        f_number_for_log = float(batch["f_number"][0].item())

        with amp_autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(
                source=batch["source"],
                depth=batch["depth"],
                f_number=batch["f_number"],
                pos_map=batch["pos_map"],
                mask=batch.get("mask"),
                target=batch["target"] if criterion.lambda_fm > 0.0 else None,
            )
            losses = criterion(out, batch)
        loss_total = losses["loss_total"]

        if not tensor_tree_is_finite(out) or not tensor_tree_is_finite(losses):
            bad_out = first_nonfinite_name("out", out)
            bad_loss = first_nonfinite_name("losses", losses)
            message = (
                f"[NonFinite] stage={stage_name} epoch={global_epoch:03d} "
                f"step={global_step + 1:06d} scene_id={scene_id_for_log} "
                f"f_number={f_number_for_log:.4f} bad_out={bad_out} bad_loss={bad_loss}"
            )
            print(message)
            if fail_on_nonfinite:
                raise FloatingPointError(message)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            continue

        metrics = compute_quality_metrics(out, batch)

        loss_for_backward = loss_total / accum_steps
        if scaler is not None and amp_enabled:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == batches_this_epoch)
        if should_step:
            step_is_safe = True
            if scaler is not None and amp_enabled:
                scaler.unscale_(optimizer)

            if not gradients_are_finite(model):
                step_is_safe = False
                bad_grad = first_nonfinite_gradient_summary(model)
                message = (
                    f"[NonFiniteGrad] stage={stage_name} epoch={global_epoch:03d} "
                    f"step={global_step + 1:06d} scene_id={scene_id_for_log} "
                    f"f_number={f_number_for_log:.4f} bad_grad={bad_grad}"
                )
                print(message)
                if fail_on_nonfinite:
                    raise FloatingPointError(message)

            if step_is_safe and grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(grad_norm.detach().float()).all():
                    step_is_safe = False
                    message = (
                        f"[NonFiniteGradNorm] stage={stage_name} epoch={global_epoch:03d} "
                        f"step={global_step + 1:06d} scene_id={scene_id_for_log} "
                        f"f_number={f_number_for_log:.4f} grad_norm={float(grad_norm.detach().float().item())}"
                    )
                    print(message)
                    if fail_on_nonfinite:
                        raise FloatingPointError(message)

            if step_is_safe:
                if scaler is not None and amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                bad_param = first_nonfinite_parameter_name(model)
                if bad_param is not None:
                    raise RuntimeError(f"Non-finite model parameter after optimizer step: {bad_param}")

                if ema_model is not None and global_epoch >= ema_start_epoch:
                    update_ema_model(ema_model, model, decay=ema_decay)
                    bad_ema_param = first_nonfinite_parameter_name(ema_model)
                    if bad_ema_param is not None:
                        raise RuntimeError(f"Non-finite EMA parameter after update: {bad_ema_param}")
            elif scaler is not None and amp_enabled:
                scaler.update()

            optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if print_every > 0 and global_step % print_every == 0:
            print(
                f"[Train] stage={stage_name} epoch={global_epoch:03d} "
                f"epoch_in_stage={epoch_in_stage:03d} step={global_step:06d} "
                f"scene_id={scene_id_for_log} "
                f"f_number={f_number_for_log:.4f} "
                f"loss_total={float(loss_total.item()):.6f} "
                f"loss_coarse={float(losses['loss_coarse'].item()):.6f} "
                f"loss_rec={float(losses['loss_rec'].item()):.6f} "
                f"loss_fm={float(losses['loss_fm'].item()):.6f} "
                f"loss_mse={float(losses['loss_mse'].item()):.6f} "
                f"loss_ssim={float(losses['loss_ssim'].item()):.6f} "
                f"loss_gate={float(losses['loss_gate'].item()):.6f} "
                f"loss_gate_sup={float(losses['loss_gate_sup'].item()):.6f} "
                f"loss_preserve={float(losses['loss_preserve'].item()):.6f} "
                f"loss_endpoint={float(losses['loss_endpoint'].item()):.6f} "
                f"loss_affine={float(losses['loss_affine'].item()):.6f} "
                f"loss_photo={float(losses['loss_photo'].item()):.6f} "
                f"lr={get_current_lr(optimizer):.7f} "
                f"coarse_err={metrics['coarse_err']:.6f} "
                f"output_err={metrics['output_err']:.6f} "
                f"improve={metrics['improve']:.6f} "
                f"coarse_psnr={metrics['coarse_psnr']:.4f} "
                f"output_psnr={metrics['output_psnr']:.4f} "
                f"coarse_ssim={metrics['coarse_ssim']:.4f} "
                f"output_ssim={metrics['output_ssim']:.4f} "
                f"gate_mean={float(out['gate'].mean().item()):.6f} "
                f"aperture_gate_bias={tensor_mean_or_zero(out.get('aperture_gate_bias')):.6f} "
                f"gate_target_mean={float(losses['gate_target_mean'].item()):.6f} "
                f"gate_max={float(out['gate'].max().item()):.6f} "
                f"residual_l1={float(out['delta_rgb'].abs().mean().item()):.6f} "
                f"radius_mean={float(out['radius_map'].mean().item()):.6f} "
                f"radius_max={float(out['radius_map'].max().item()):.6f} "
                f"kappa={tensor_mean_or_zero(out.get('kappa')):.6f} "
                f"aperture_radius={tensor_mean_or_zero(out.get('aperture_radius')):.6f} "
                f"lens_strength={tensor_mean_or_zero(out.get('lens_strength')):.6f} "
                f"aperture_scale={tensor_mean_or_zero(out.get('aperture_scale')):.6f} "
                f"source_strength={tensor_mean_or_zero(out.get('source_strength')):.6f}"
                f" photo_gain={tensor_mean_or_zero(out.get('photo_gain')):.6f} "
                f"photo_bias_abs={float(out['photo_bias'].detach().abs().mean().item()):.6f}"
            )

        if vis_every > 0 and global_step % vis_every == 0:
            vis_path = vis_dir / (
                f"train_stage_{stage_name}_e{global_epoch:03d}_s{global_step:06d}"
                f"_scene_{scene_id_for_log}_f{f_number_for_log:.4f}.png"
            )
            save_vis_panel(batch, out, vis_path, max_radius=float(global_cfg["max_radius"]))
            print(f"[Vis] saved to {vis_path}")

        if ckpt_every > 0 and global_step % ckpt_every == 0:
            ckpt_path = ckpt_dir / f"stage_{stage_name}_step_{global_step:06d}.pt"
            ckpt_ema_model = ema_model if should_use_ema_model(ema_model, global_epoch, ema_start_epoch) else None
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                global_cfg,
                stage_idx,
                epoch_in_stage,
                global_epoch,
                global_step,
                best_val_output_psnr,
                best_full_output_psnr=best_full_output_psnr,
                best_crop_output_psnr=best_crop_output_psnr,
                best_full_output_ssim=best_full_output_ssim,
                ema_model=ckpt_ema_model,
                scaler=scaler,
                epoch_complete=False,
            )
            print(f"[Ckpt] saved to {ckpt_path}")

    return global_step


# =========================
# Main
# =========================

def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    if args.resume:
        cfg["resume"] = args.resume

    validate_config(cfg)
    set_seed(int(cfg["seed"]))

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(cfg["save_dir"]) / cfg["experiment_name"]
    ckpt_dir = exp_dir / "checkpoints"
    vis_dir = exp_dir / "visuals"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"Using device: {device}")
    print(f"Experiment dir: {exp_dir}")
    print(f"Resolved config: {json.dumps(cfg, ensure_ascii=False, indent=2)}")

    train_ds = ExplicitBokehDataset(
        root=cfg["root"],
        split=cfg["train_split"],
        divisor=4,
        normalize_depth=True,
        include_masks=True,
        layout=str(cfg.get("dataset_layout", "auto")),
        align_source_to_target_mean=cfg.get("align_source_to_target_mean"),
    )

    val_ds = ExplicitBokehDataset(
        root=cfg["root"],
        split=cfg["val_split"],
        divisor=4,
        normalize_depth=True,
        include_masks=True,
        layout=str(cfg.get("dataset_layout", "auto")),
        align_source_to_target_mean=cfg.get("align_source_to_target_mean"),
    )

    val_loader = build_val_loader(val_ds, int(cfg["num_workers"]))

    model = build_model(cfg, device)
    ema_enabled = bool(cfg.get("ema_enabled", False))
    ema_decay = float(cfg.get("ema_decay", 0.999))
    ema_start_epoch = int(cfg.get("ema_start_epoch", 3))
    ema_model = clone_eval_model(model, cfg, device) if ema_enabled else None
    ema_initialized = ema_model is not None and ema_start_epoch <= 1

    criterion = rebuild_criterion_for_stage(cfg, cfg)
    
    optimizer = build_optimizer(model, cfg)
    amp_enabled = bool(cfg.get("amp_enabled", False)) and device.type == "cuda"
    amp_dtype_name = str(cfg.get("amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    scaler = build_grad_scaler(enabled=amp_enabled and amp_dtype == torch.float16)

    resume_path = str(cfg.get("resume", "")).strip()
    init_checkpoint_path = str(cfg.get("init_checkpoint", "")).strip()
    init_weights = str(cfg.get("init_weights", "ema")).lower()
    if resume_path and init_checkpoint_path:
        raise ValueError("Use either resume or init_checkpoint, not both.")

    start_stage_idx = 0
    start_epoch_in_stage = 0
    global_epoch = 0
    global_step = 0
    best_val_output_psnr = float("-inf")
    best_full_output_psnr = float("-inf")
    best_crop_output_psnr = float("-inf")
    best_full_output_ssim = float("-inf")

    last_stage_idx = 0
    last_epoch_in_stage = 0

    if init_checkpoint_path:
        init_checkpoint = torch.load(init_checkpoint_path, map_location=device)
        loaded_weights, missing_keys = load_initial_model_weights(model, init_checkpoint, init_weights)
        if ema_model is not None:
            ema_model.load_state_dict(model.state_dict(), strict=True)
            ema_initialized = True
        print(
            f"[Init] Loaded {loaded_weights} model weights from {init_checkpoint_path}; "
            f"new_parameters={missing_keys}"
        )

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if ema_model is not None:
            if "ema_model" in ckpt:
                ema_model.load_state_dict(ckpt["ema_model"], strict=True)
                ema_initialized = True
            else:
                ema_model.load_state_dict(ckpt["model"], strict=True)
                ema_initialized = int(ckpt.get("global_epoch", 0)) >= ema_start_epoch
        start_stage_idx = int(ckpt.get("stage_idx", 0))
        default_epoch_complete = "_step_" not in Path(resume_path).name
        epoch_complete = bool(ckpt.get("epoch_complete", default_epoch_complete))
        start_epoch_in_stage = int(ckpt.get("epoch_in_stage", 0)) + int(epoch_complete)
        global_epoch = int(ckpt.get("global_epoch", 0))
        if not epoch_complete:
            # Mid-epoch recovery repeats that epoch from its beginning.
            global_epoch = max(global_epoch - 1, 0)
        global_step = int(ckpt.get("global_step", 0))
        best_val_output_psnr = float(ckpt.get("best_val_output_psnr", float("-inf")))
        best_full_output_psnr = float(ckpt.get("best_full_output_psnr", best_val_output_psnr))
        best_crop_output_psnr = float(ckpt.get("best_crop_output_psnr", float("-inf")))
        best_full_output_ssim = float(ckpt.get("best_full_output_ssim", float("-inf")))
        last_stage_idx = start_stage_idx
        last_epoch_in_stage = int(ckpt.get("epoch_in_stage", 0))

        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in ckpt:
            torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng_state_all"]])

        print(f"[Resume] Loaded checkpoint from {resume_path}")
        print(
            f"[Resume] start_stage_idx={start_stage_idx}, "
            f"start_epoch_in_stage={start_epoch_in_stage}, "
            f"epoch_complete={epoch_complete}, "
            f"global_epoch={global_epoch}, global_step={global_step}, "
            f"best_full_output_psnr={best_full_output_psnr:.4f}, "
            f"best_crop_output_psnr={best_crop_output_psnr:.4f}, "
            f"best_full_output_ssim={best_full_output_ssim:.4f}"
        )

    for stage_idx, raw_stage_cfg in enumerate(cfg["stages"]):
        if stage_idx < start_stage_idx:
            continue

        stage_cfg = merge_stage_cfg(cfg, raw_stage_cfg)
        stage_name = str(stage_cfg["name"])
        stage_epochs = int(stage_cfg["epochs"])
        stage_batch_size = int(stage_cfg["batch_size"])
        stage_lr = float(stage_cfg["lr"])
        stage_min_lr = float(stage_cfg.get("min_lr", cfg.get("min_lr", stage_lr)))
        scheduler_name = str(stage_cfg.get("scheduler", cfg.get("scheduler", "none"))).lower()
        warmup_epochs = int(stage_cfg.get("warmup_epochs", cfg.get("warmup_epochs", 0)))
        val_crop_max_samples = int(stage_cfg.get("val_crop_max_samples", cfg.get("val_crop_max_samples", 0)))
        val_full_max_samples = int(stage_cfg.get("val_full_max_samples", cfg.get("val_full_max_samples", 0)))
        val_full_every = max(int(stage_cfg.get("val_full_every", cfg.get("val_full_every", 1))), 1)
        raw_validation_epochs = stage_cfg.get("validation_epochs")
        validation_epochs = None
        if raw_validation_epochs is not None:
            if not isinstance(raw_validation_epochs, (list, tuple)):
                raise TypeError(
                    f"validation_epochs for stage '{stage_name}' must be a list of 1-based epoch indices"
                )
            validation_epochs = {int(epoch) for epoch in raw_validation_epochs}
            invalid_validation_epochs = sorted(
                epoch for epoch in validation_epochs if epoch < 1 or epoch > stage_epochs
            )
            if invalid_validation_epochs:
                raise ValueError(
                    f"validation_epochs for stage '{stage_name}' contains out-of-range values: "
                    f"{invalid_validation_epochs}; valid range is [1, {stage_epochs}]"
                )
        aperture_sampling_weights = normalize_aperture_weight_cfg(
            stage_cfg.get("aperture_sampling_weights", cfg.get("aperture_sampling_weights"))
        )
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total_params = sum(parameter.numel() for parameter in model.parameters())
        criterion = rebuild_criterion_for_stage(cfg, stage_cfg)

        train_loader = build_train_loader(
            train_ds,
            stage_batch_size,
            int(cfg["num_workers"]),
            aperture_sampling_weights=aperture_sampling_weights,
            pre_collate_crop=(int(stage_cfg["crop_h"]), int(stage_cfg["crop_w"]))
            if str(cfg.get("dataset_layout", "auto")).lower() == "ebb"
            else None,
        )

        print(
            f"\n[Stage Start] idx={stage_idx} name={stage_name} "
            f"crop=({stage_cfg['crop_h']},{stage_cfg['crop_w']}) "
            f"epochs={stage_epochs} batch_size={stage_batch_size} lr={stage_lr} "
            f"min_lr={stage_min_lr} scheduler={scheduler_name} "
            f"warmup_epochs={warmup_epochs} accum_steps={int(stage_cfg.get('accum_steps', cfg.get('accum_steps', 1)))} "
            f"max_batches={int(stage_cfg.get('max_batches_per_epoch', 0))} "
            f"validation_epochs={sorted(validation_epochs) if validation_epochs is not None else 'every'} "
            f"trainable_params={trainable_params}/{total_params} "
            f"lambda_coarse={float(stage_cfg.get('lambda_coarse', cfg.get('lambda_coarse', 0.0))):.4f}\n"
        )

        epoch_begin = start_epoch_in_stage if stage_idx == start_stage_idx else 0

        if epoch_begin >= stage_epochs:
            print(f"[Stage Skip] idx={stage_idx} name={stage_name} already finished by resume checkpoint.")
            continue

        for epoch_in_stage in range(epoch_begin, stage_epochs):
            global_epoch += 1
            last_stage_idx = stage_idx
            last_epoch_in_stage = epoch_in_stage

            if ema_model is not None and not ema_initialized and global_epoch >= ema_start_epoch:
                ema_model.load_state_dict(model.state_dict(), strict=True)
                ema_initialized = True
                print(f"[EMA] Initialized from current model at global_epoch={global_epoch}")

            epoch_lr = compute_epoch_lr(
                epoch_in_stage=epoch_in_stage,
                stage_epochs=stage_epochs,
                lr_max=stage_lr,
                lr_min=stage_min_lr,
                scheduler_name=scheduler_name,
                warmup_epochs=warmup_epochs,
            )
            set_optimizer_lr(optimizer, epoch_lr)

            global_step = train_one_epoch(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                device=device,
                stage_cfg=stage_cfg,
                global_cfg=cfg,
                stage_idx=stage_idx,
                epoch_in_stage=epoch_in_stage,
                global_epoch=global_epoch,
                global_step=global_step,
                vis_dir=vis_dir,
                ckpt_dir=ckpt_dir,
                best_val_output_psnr=best_val_output_psnr,
                best_full_output_psnr=best_full_output_psnr,
                best_crop_output_psnr=best_crop_output_psnr,
                best_full_output_ssim=best_full_output_ssim,
                ema_model=ema_model,
                ema_decay=ema_decay,
                ema_start_epoch=ema_start_epoch,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )

            ema_eval_model = ema_model if should_use_ema_model(ema_model, global_epoch, ema_start_epoch) else None

            latest_path = ckpt_dir / "latest.pt"
            save_checkpoint(
                latest_path,
                model,
                optimizer,
                cfg,
                stage_idx,
                epoch_in_stage,
                global_epoch,
                global_step,
                best_val_output_psnr,
                best_full_output_psnr=best_full_output_psnr,
                best_crop_output_psnr=best_crop_output_psnr,
                best_full_output_ssim=best_full_output_ssim,
                ema_model=ema_eval_model,
                scaler=scaler,
                epoch_complete=True,
            )

            local_epoch_number = epoch_in_stage + 1
            if validation_epochs is not None and local_epoch_number not in validation_epochs:
                print(
                    f"[Val] skipped by validation_epochs at stage={stage_name} "
                    f"epoch={global_epoch:03d} epoch_in_stage={local_epoch_number:03d}; "
                    f"scheduled={sorted(validation_epochs)}"
                )
                continue

            eval_model = ema_eval_model if ema_eval_model is not None else model

            val_crop_stats = run_validation(
                model=eval_model,
                criterion=criterion,
                val_loader=val_loader,
                device=device,
                crop_h=int(stage_cfg["crop_h"]),
                crop_w=int(stage_cfg["crop_w"]),
                max_samples=val_crop_max_samples,
                balance_apertures=True,
            )
            run_full_validation = (
                (epoch_in_stage + 1) % val_full_every == 0
                or epoch_in_stage == stage_epochs - 1
            )
            val_full_stats = None
            if run_full_validation:
                val_full_stats = run_validation(
                    model=eval_model,
                    criterion=criterion,
                    val_loader=val_loader,
                    device=device,
                    crop_h=0,
                    crop_w=0,
                    max_samples=val_full_max_samples,
                    balance_apertures=val_full_max_samples > 0,
                )

            print(
                f"[ValCrop] stage={stage_name} epoch={global_epoch:03d} "
                f"epoch_in_stage={epoch_in_stage:03d} "
                f"loss_total={val_crop_stats['val_loss_total']:.6f} "
                f"loss_coarse={val_crop_stats['val_loss_coarse']:.6f} "
                f"loss_rec={val_crop_stats['val_loss_rec']:.6f} "
                f"loss_fm={val_crop_stats['val_loss_fm']:.6f} "
                f"loss_mse={val_crop_stats['val_loss_mse']:.6f} "
                f"loss_ssim={val_crop_stats['val_loss_ssim']:.6f} "
                f"loss_gate={val_crop_stats['val_loss_gate']:.6f} "
                f"loss_gate_sup={val_crop_stats['val_loss_gate_sup']:.6f} "
                f"loss_preserve={val_crop_stats['val_loss_preserve']:.6f} "
                f"loss_endpoint={val_crop_stats['val_loss_endpoint']:.6f} "
                f"loss_affine={val_crop_stats['val_loss_affine']:.6f} "
                f"loss_photo={val_crop_stats['val_loss_photo']:.6f} "
                f"coarse_err={val_crop_stats['val_coarse_err']:.6f} "
                f"output_err={val_crop_stats['val_output_err']:.6f} "
                f"improve={val_crop_stats['val_improve']:.6f} "
                f"coarse_psnr={val_crop_stats['val_coarse_psnr']:.4f} "
                f"output_psnr={val_crop_stats['val_output_psnr']:.4f} "
                f"coarse_ssim={val_crop_stats['val_coarse_ssim']:.4f} "
                f"output_ssim={val_crop_stats['val_output_ssim']:.4f} "
                f"kappa={val_crop_stats['val_kappa']:.6f} "
                f"aperture_radius={val_crop_stats['val_aperture_radius']:.6f} "
                f"lens_strength={val_crop_stats['val_lens_strength']:.6f} "
                f"aperture_scale={val_crop_stats['val_aperture_scale']:.6f} "
                f"source_strength={val_crop_stats['val_source_strength']:.6f} "
                f"photo_gain={val_crop_stats['val_photo_gain']:.6f} "
                f"photo_bias_abs={val_crop_stats['val_photo_bias_abs']:.6f} "
                f"gate_mean={val_crop_stats['val_gate_mean']:.6f} "
                f"gate_target_mean={val_crop_stats['val_gate_target_mean']:.6f} "
                f"proposal_l1={val_crop_stats['val_proposal_l1']:.6f} "
                f"applied_l1={val_crop_stats['val_applied_l1']:.6f}"
            )
            if val_full_stats is not None:
                print(
                    f"[ValFull] stage={stage_name} epoch={global_epoch:03d} "
                    f"epoch_in_stage={epoch_in_stage:03d} "
                    f"loss_total={val_full_stats['val_loss_total']:.6f} "
                    f"loss_coarse={val_full_stats['val_loss_coarse']:.6f} "
                    f"loss_rec={val_full_stats['val_loss_rec']:.6f} "
                    f"loss_fm={val_full_stats['val_loss_fm']:.6f} "
                    f"loss_mse={val_full_stats['val_loss_mse']:.6f} "
                    f"loss_ssim={val_full_stats['val_loss_ssim']:.6f} "
                    f"loss_gate={val_full_stats['val_loss_gate']:.6f} "
                    f"loss_gate_sup={val_full_stats['val_loss_gate_sup']:.6f} "
                    f"loss_preserve={val_full_stats['val_loss_preserve']:.6f} "
                    f"loss_endpoint={val_full_stats['val_loss_endpoint']:.6f} "
                    f"loss_affine={val_full_stats['val_loss_affine']:.6f} "
                    f"loss_photo={val_full_stats['val_loss_photo']:.6f} "
                    f"coarse_err={val_full_stats['val_coarse_err']:.6f} "
                    f"output_err={val_full_stats['val_output_err']:.6f} "
                    f"improve={val_full_stats['val_improve']:.6f} "
                    f"coarse_psnr={val_full_stats['val_coarse_psnr']:.4f} "
                    f"output_psnr={val_full_stats['val_output_psnr']:.4f} "
                    f"coarse_ssim={val_full_stats['val_coarse_ssim']:.4f} "
                    f"output_ssim={val_full_stats['val_output_ssim']:.4f} "
                    f"kappa={val_full_stats['val_kappa']:.6f} "
                    f"aperture_radius={val_full_stats['val_aperture_radius']:.6f} "
                    f"lens_strength={val_full_stats['val_lens_strength']:.6f} "
                    f"aperture_scale={val_full_stats['val_aperture_scale']:.6f} "
                    f"source_strength={val_full_stats['val_source_strength']:.6f} "
                    f"photo_gain={val_full_stats['val_photo_gain']:.6f} "
                    f"photo_bias_abs={val_full_stats['val_photo_bias_abs']:.6f} "
                    f"gate_mean={val_full_stats['val_gate_mean']:.6f} "
                    f"aperture_gate_bias={val_full_stats['val_aperture_gate_bias']:.6f} "
                    f"gate_target_mean={val_full_stats['val_gate_target_mean']:.6f} "
                    f"proposal_l1={val_full_stats['val_proposal_l1']:.6f} "
                    f"applied_l1={val_full_stats['val_applied_l1']:.6f}"
                )
                for aperture_key, aperture_stats in val_full_stats["val_by_aperture"].items():
                    print(
                        f"[ValFullAperture] stage={stage_name} epoch={global_epoch:03d} "
                        f"aperture={aperture_key} "
                        f"coarse_psnr={aperture_stats['coarse_psnr']:.4f} "
                        f"output_psnr={aperture_stats['output_psnr']:.4f} "
                        f"coarse_ssim={aperture_stats['coarse_ssim']:.4f} "
                        f"output_ssim={aperture_stats['output_ssim']:.4f} "
                        f"gate_mean={aperture_stats['gate_mean']:.6f} "
                        f"aperture_gate_bias={aperture_stats['aperture_gate_bias']:.6f} "
                        f"gate_target_mean={aperture_stats['gate_target_mean']:.6f} "
                        f"proposal_l1={aperture_stats['proposal_l1']:.6f} "
                        f"applied_l1={aperture_stats['applied_l1']:.6f}"
                    )
            else:
                print(
                    f"[ValFull] skipped at stage={stage_name} epoch={global_epoch:03d}; "
                    f"frequency={val_full_every}"
                )

            if val_crop_stats["val_output_psnr"] > best_crop_output_psnr:
                best_crop_output_psnr = val_crop_stats["val_output_psnr"]
                best_crop_path = ckpt_dir / "best_crop.pt"
                save_checkpoint(
                    best_crop_path,
                    model,
                    optimizer,
                    cfg,
                    stage_idx,
                    epoch_in_stage,
                    global_epoch,
                    global_step,
                    best_val_output_psnr,
                    best_full_output_psnr=best_full_output_psnr,
                    best_crop_output_psnr=best_crop_output_psnr,
                    best_full_output_ssim=best_full_output_ssim,
                    ema_model=ema_eval_model,
                    scaler=scaler,
                    epoch_complete=True,
                )
                print(
                    f"[BestCrop] Updated best checkpoint: {best_crop_path} "
                    f"(val_output_psnr={best_crop_output_psnr:.4f})"
                )

            if val_full_stats is not None and val_full_stats["val_output_psnr"] > best_full_output_psnr:
                best_full_output_psnr = val_full_stats["val_output_psnr"]
                best_val_output_psnr = best_full_output_psnr
                best_path = ckpt_dir / "best.pt"
                best_full_path = ckpt_dir / "best_full.pt"
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    cfg,
                    stage_idx,
                    epoch_in_stage,
                    global_epoch,
                    global_step,
                    best_val_output_psnr,
                    best_full_output_psnr=best_full_output_psnr,
                    best_crop_output_psnr=best_crop_output_psnr,
                    best_full_output_ssim=best_full_output_ssim,
                    ema_model=ema_eval_model,
                    scaler=scaler,
                    epoch_complete=True,
                )
                save_checkpoint(
                    best_full_path,
                    model,
                    optimizer,
                    cfg,
                    stage_idx,
                    epoch_in_stage,
                    global_epoch,
                    global_step,
                    best_val_output_psnr,
                    best_full_output_psnr=best_full_output_psnr,
                    best_crop_output_psnr=best_crop_output_psnr,
                    best_full_output_ssim=best_full_output_ssim,
                    ema_model=ema_eval_model,
                    scaler=scaler,
                    epoch_complete=True,
                )
                print(
                    f"[BestFull] Updated best checkpoint: {best_full_path} "
                    f"(val_output_psnr={best_full_output_psnr:.4f})"
                )

            if val_full_stats is not None and val_full_stats["val_output_ssim"] > best_full_output_ssim:
                best_full_output_ssim = val_full_stats["val_output_ssim"]
                best_full_ssim_path = ckpt_dir / "best_full_ssim.pt"
                save_checkpoint(
                    best_full_ssim_path,
                    model,
                    optimizer,
                    cfg,
                    stage_idx,
                    epoch_in_stage,
                    global_epoch,
                    global_step,
                    best_val_output_psnr,
                    best_full_output_psnr=best_full_output_psnr,
                    best_crop_output_psnr=best_crop_output_psnr,
                    best_full_output_ssim=best_full_output_ssim,
                    ema_model=ema_eval_model,
                    scaler=scaler,
                    epoch_complete=True,
                )
                print(
                    f"[BestFullSSIM] Updated best checkpoint: {best_full_ssim_path} "
                    f"(val_output_ssim={best_full_output_ssim:.4f})"
                )

            with torch.no_grad():
                first_val_batch = next(iter(val_loader))
                first_val_batch = move_batch_to_device(first_val_batch, device)
                first_val_batch = center_crop_batch_spatial(
                    first_val_batch,
                    int(stage_cfg["crop_h"]),
                    int(stage_cfg["crop_w"]),
                )
                vis_model = ema_eval_model if ema_eval_model is not None else model
                first_val_out = vis_model(
                    source=first_val_batch["source"],
                    depth=first_val_batch["depth"],
                    f_number=first_val_batch["f_number"],
                    pos_map=first_val_batch["pos_map"],
                    mask=first_val_batch.get("mask"),
                )

            val_scene_id = get_scene_id_for_log(first_val_batch)
            val_f_number = float(first_val_batch["f_number"][0].item())
            val_vis_path = vis_dir / (
                f"val_stage_{stage_name}_e{global_epoch:03d}"
                f"_scene_{val_scene_id}_f{val_f_number:.4f}.png"
            )
            save_vis_panel(first_val_batch, first_val_out, val_vis_path, max_radius=float(cfg["max_radius"]))
            print(f"[Val Vis] saved to {val_vis_path}")

            # Refresh latest after validation so it contains current best metrics.
            save_checkpoint(
                latest_path,
                model,
                optimizer,
                cfg,
                stage_idx,
                epoch_in_stage,
                global_epoch,
                global_step,
                best_val_output_psnr,
                best_full_output_psnr=best_full_output_psnr,
                best_crop_output_psnr=best_crop_output_psnr,
                best_full_output_ssim=best_full_output_ssim,
                ema_model=ema_eval_model,
                scaler=scaler,
                epoch_complete=True,
            )

        stage_last_path = ckpt_dir / f"stage_{stage_name}_last.pt"
        save_checkpoint(
            stage_last_path,
            model,
            optimizer,
            cfg,
            stage_idx,
            stage_epochs - 1,
            global_epoch,
            global_step,
            best_val_output_psnr,
            best_full_output_psnr=best_full_output_psnr,
            best_crop_output_psnr=best_crop_output_psnr,
            best_full_output_ssim=best_full_output_ssim,
            ema_model=ema_eval_model,
            scaler=scaler,
            epoch_complete=True,
        )
        print(f"[Stage End] Saved {stage_last_path}")

    final_path = ckpt_dir / "final.pt"
    final_ema_model = ema_model if should_use_ema_model(ema_model, global_epoch, ema_start_epoch) else None
    save_checkpoint(
        final_path,
        model,
        optimizer,
        cfg,
        last_stage_idx,
        last_epoch_in_stage,
        global_epoch,
        global_step,
        best_val_output_psnr,
        best_full_output_psnr=best_full_output_psnr,
        best_crop_output_psnr=best_crop_output_psnr,
        best_full_output_ssim=best_full_output_ssim,
        ema_model=final_ema_model,
        scaler=scaler,
        epoch_complete=True,
    )
    print(f"Training finished. Final checkpoint saved to {final_path}")
    print(f"Best full-image val_output_psnr = {best_full_output_psnr:.4f}")
    print(f"Best crop val_output_psnr = {best_crop_output_psnr:.4f}")
    print(f"Best full-image val_output_ssim = {best_full_output_ssim:.4f}")


if __name__ == "__main__":
    main()
