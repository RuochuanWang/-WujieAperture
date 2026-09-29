from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from method.metrics import ssim_batch


def quantize_f_number_key(value: float) -> float:
    return round(float(value), 1)


def weighted_l1_loss_per_sample(
    pred: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """
    pred, target: [B, C, H, W]
    weight      : [B, 1, H, W] or [B, C, H, W]
    """
    diff = (pred - target).abs()

    if weight is None:
        return diff.flatten(1).mean(dim=1)

    if weight.dim() != 4:
        raise ValueError(f"weight must be 4D, got shape {tuple(weight.shape)}")

    if weight.shape[1] == 1 and pred.shape[1] > 1:
        weight = weight.expand(-1, pred.shape[1], -1, -1)

    if weight.shape != pred.shape:
        raise ValueError(
            f"Expanded weight shape {tuple(weight.shape)} must match pred shape {tuple(pred.shape)}"
        )

    weighted = (diff * weight).flatten(1).sum(dim=1)
    denom = weight.flatten(1).sum(dim=1) + eps
    return weighted / denom


def weighted_mse_loss_per_sample(
    pred: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """Per-sample MSE with the same optional spatial weighting as L1."""
    diff_sq = (pred - target).square()

    if weight is None:
        return diff_sq.flatten(1).mean(dim=1)

    if weight.dim() != 4:
        raise ValueError(f"weight must be 4D, got shape {tuple(weight.shape)}")

    if weight.shape[1] == 1 and pred.shape[1] > 1:
        weight = weight.expand(-1, pred.shape[1], -1, -1)

    if weight.shape != pred.shape:
        raise ValueError(
            f"Expanded weight shape {tuple(weight.shape)} must match pred shape {tuple(pred.shape)}"
        )

    weighted = (diff_sq * weight).flatten(1).sum(dim=1)
    denom = weight.flatten(1).sum(dim=1) + eps
    return weighted / denom


def structural_loss_per_sample(
    pred: Tensor,
    target: Tensor,
    max_size: int = 512,
) -> Tensor:
    """Compute 1-SSIM in float32, optionally at a bounded resolution."""
    # The criterion is called inside the training autocast context. Explicitly
    # disable it here because SSIM's local variance calculation is sensitive
    # to BF16 rounding even when its inputs were first converted to float32.
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred_float = pred.float().clamp(0.0, 1.0)
        target_float = target.float().clamp(0.0, 1.0)
        max_size = int(max_size)
        if max_size > 0 and max(pred_float.shape[-2:]) > max_size:
            scale = float(max_size) / float(max(pred_float.shape[-2:]))
            size = (
                max(int(round(pred_float.shape[-2] * scale)), 11),
                max(int(round(pred_float.shape[-1] * scale)), 11),
            )
            pred_float = F.interpolate(pred_float, size=size, mode="bilinear", align_corners=False)
            target_float = F.interpolate(target_float, size=size, mode="bilinear", align_corners=False)
        return 1.0 - ssim_batch(pred_float, target_float).clamp(-1.0, 1.0)


@torch.no_grad()
def fit_bounded_rgb_affine(
    source: Tensor,
    target: Tensor,
    max_log_gain: float,
    max_bias: float,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit a bounded per-image, per-channel affine transform in closed form."""
    with torch.autocast(device_type=source.device.type, enabled=False):
        x = source.detach().float()
        y = target.detach().float()
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        y_mean = y.mean(dim=(2, 3), keepdim=True)
        covariance = ((x - x_mean) * (y - y_mean)).mean(dim=(2, 3), keepdim=True)
        variance = (x - x_mean).square().mean(dim=(2, 3), keepdim=True)

        min_gain = torch.exp(x.new_tensor(-float(max_log_gain)))
        max_gain = torch.exp(x.new_tensor(float(max_log_gain)))
        gain = covariance / (variance + eps)
        gain = torch.maximum(torch.minimum(gain, max_gain), min_gain)
        bias = (y_mean - gain * x_mean).clamp(-float(max_bias), float(max_bias))
        log_gain = torch.log(gain.clamp_min(eps))
    return gain, bias, log_gain


class ExplicitBokehLoss(nn.Module):
    """Reconstruction, residual-flow, structural, gate, and photometric losses."""

    def __init__(
        self,
        lambda_coarse: float = 0.0,
        lambda_rec: float = 1.0,
        lambda_fm: float = 0.0,
        lambda_mse: float = 0.0,
        lambda_ssim: float = 0.0,
        lambda_gate: float = 0.1,
        lambda_gate_sup: float = 0.0,
        lambda_preserve: float = 0.5,
        lambda_endpoint: float = 0.0,
        lambda_affine: float = 0.0,
        lambda_photo: float = 0.0,
        gate_target_threshold: float = 0.03,
        gate_target_quantile: float = 0.75,
        gate_target_kernel_size: int = 5,
        affine_teacher_max_log_gain: float = 0.20,
        affine_teacher_max_bias: float = 0.10,
        ssim_max_size: int = 512,
        use_softmask_for_rec: bool = False,
        use_mask_for_rec: bool = False,
        detach_repair_map: bool = True,
        aperture_rec_weights: Optional[Dict[float, float]] = None,
    ) -> None:
        super().__init__()
        self.lambda_coarse = lambda_coarse
        self.lambda_rec = lambda_rec
        self.lambda_fm = lambda_fm
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_gate = lambda_gate
        self.lambda_gate_sup = lambda_gate_sup
        self.lambda_preserve = lambda_preserve
        self.lambda_endpoint = lambda_endpoint
        self.lambda_affine = lambda_affine
        self.lambda_photo = lambda_photo
        self.gate_target_threshold = float(gate_target_threshold)
        self.gate_target_quantile = min(max(float(gate_target_quantile), 0.0), 1.0)
        self.gate_target_kernel_size = max(int(gate_target_kernel_size), 1)
        if self.gate_target_kernel_size % 2 == 0:
            raise ValueError("gate_target_kernel_size must be odd")
        self.affine_teacher_max_log_gain = float(affine_teacher_max_log_gain)
        self.affine_teacher_max_bias = float(affine_teacher_max_bias)
        self.ssim_max_size = int(ssim_max_size)
        self.use_softmask_for_rec = use_softmask_for_rec
        self.use_mask_for_rec = use_mask_for_rec
        self.detach_repair_map = detach_repair_map
        self.aperture_rec_weights = {
            quantize_f_number_key(k): float(v) for k, v in (aperture_rec_weights or {}).items()
        }

    def _get_rec_weight(self, batch: Dict[str, Tensor]) -> Optional[Tensor]:
        if self.use_softmask_for_rec and "softmask" in batch:
            return batch["softmask"]
        if self.use_mask_for_rec and "mask" in batch:
            return batch["mask"]
        return None

    def _get_aperture_sample_weight(self, batch: Dict[str, Tensor], ref_tensor: Tensor) -> Tensor:
        f_number = batch.get("f_number")
        if f_number is None or len(self.aperture_rec_weights) == 0:
            return torch.ones(ref_tensor.shape[0], device=ref_tensor.device, dtype=ref_tensor.dtype)

        weights = []
        for value in f_number.view(-1).detach().cpu().tolist():
            weights.append(self.aperture_rec_weights.get(quantize_f_number_key(value), 1.0))
        return torch.tensor(weights, device=ref_tensor.device, dtype=ref_tensor.dtype)

    def forward(self, model_out: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = model_out["output"]
        coarse = model_out["coarse"]
        gate = model_out["gate"]
        repair_map = model_out["repair_map"]
        target = batch["target"]

        if self.detach_repair_map:
            repair_map = repair_map.detach()

        repair_map = repair_map.clamp(0.0, 1.0)
        outside_repair = (1.0 - repair_map).clamp(0.0, 1.0)

        rec_weight = self._get_rec_weight(batch)

        loss_coarse_per_sample = weighted_l1_loss_per_sample(coarse, target, rec_weight)
        loss_rec_per_sample = weighted_l1_loss_per_sample(output, target, rec_weight)
        loss_mse_per_sample = weighted_mse_loss_per_sample(output, target, rec_weight)

        sample_weight = self._get_aperture_sample_weight(batch, output)
        loss_coarse = (loss_coarse_per_sample * sample_weight).sum() / (sample_weight.sum() + 1e-6)
        loss_rec = (loss_rec_per_sample * sample_weight).sum() / (sample_weight.sum() + 1e-6)
        loss_mse = (loss_mse_per_sample * sample_weight).sum() / (sample_weight.sum() + 1e-6)

        if self.lambda_ssim > 0.0:
            loss_ssim_per_sample = structural_loss_per_sample(
                output,
                target,
                max_size=self.ssim_max_size,
            )
            loss_ssim = (loss_ssim_per_sample * sample_weight.float()).sum() / (
                sample_weight.float().sum() + 1e-6
            )
        else:
            loss_ssim = output.new_tensor(0.0)

        # Fit the global exposure/color component first, then define a local
        # repair support from the remaining spatial error. Unlike a fixed
        # quantile, an absolute threshold does not force every aperture to open
        # the same fraction of its gate.
        coarse_physical = model_out.get("coarse_physical")
        photo_log_gain = model_out.get("photo_log_gain")
        photo_bias = model_out.get("photo_bias")
        affine_teacher_gain = None
        affine_teacher_bias = None
        affine_teacher_log_gain = None
        repair_support = None
        needs_affine_teacher = coarse_physical is not None and any(
            value > 0.0
            for value in (
                self.lambda_affine,
                self.lambda_gate_sup,
                self.lambda_fm,
                self.lambda_endpoint,
                self.lambda_gate,
                self.lambda_preserve,
            )
        )
        if needs_affine_teacher:
            affine_teacher_gain, affine_teacher_bias, affine_teacher_log_gain = fit_bounded_rgb_affine(
                coarse_physical,
                target,
                max_log_gain=self.affine_teacher_max_log_gain,
                max_bias=self.affine_teacher_max_bias,
            )
            with torch.no_grad():
                teacher_coarse = (
                    coarse_physical.detach().float() * affine_teacher_gain + affine_teacher_bias
                )
                residual_need = (target.detach().float() - teacher_coarse).abs().mean(dim=1, keepdim=True)
                if self.gate_target_kernel_size > 1:
                    residual_need = F.avg_pool2d(
                        residual_need,
                        kernel_size=self.gate_target_kernel_size,
                        stride=1,
                        padding=self.gate_target_kernel_size // 2,
                    )
                if self.gate_target_quantile > 0.0:
                    threshold = torch.quantile(
                        residual_need.flatten(1),
                        q=self.gate_target_quantile,
                        dim=1,
                    ).view(-1, 1, 1, 1)
                else:
                    threshold = residual_need.new_full(
                        (residual_need.shape[0], 1, 1, 1),
                        self.gate_target_threshold,
                    )
                repair_support = (residual_need >= threshold).to(dtype=output.dtype)

        # Flow matching learns the complete residual velocity field. The gate
        # is trained separately to decide where that field is applied, so FM
        # supervision must not disappear outside a thresholded support.
        flow_velocity = model_out.get("flow_velocity_tau")
        flow_target = model_out.get("flow_target_velocity")
        if flow_velocity is not None and flow_target is not None:
            loss_fm = weighted_l1_loss_per_sample(
                flow_velocity.float(),
                flow_target.float(),
            )
            loss_fm = (loss_fm * sample_weight).sum() / (sample_weight.sum() + 1e-6)
        else:
            loss_fm = output.new_tensor(0.0)

        endpoint_velocity = model_out.get("endpoint_velocity")
        if self.lambda_endpoint > 0.0 and endpoint_velocity is not None:
            endpoint_target = (target - coarse).detach()
            if repair_support is not None:
                endpoint_target = endpoint_target * repair_support.to(endpoint_target.dtype)
            # endpoint_velocity is the applied update g * v. Supervising the
            # applied update avoids asking an ungated proposal to compensate
            # for a fractional gate.
            loss_endpoint = weighted_l1_loss_per_sample(
                endpoint_velocity.float(),
                endpoint_target.float(),
            )
            loss_endpoint = (loss_endpoint * sample_weight).sum() / (sample_weight.sum() + 1e-6)
        else:
            loss_endpoint = output.new_tensor(0.0)

        if (
            self.lambda_affine > 0.0
            and photo_log_gain is not None
            and photo_bias is not None
            and affine_teacher_log_gain is not None
            and affine_teacher_bias is not None
        ):
            affine_error = (photo_log_gain.float() - affine_teacher_log_gain).abs().flatten(1).mean(dim=1)
            affine_error = affine_error + (
                photo_bias.float() - affine_teacher_bias
            ).abs().flatten(1).mean(dim=1)
            loss_affine = (affine_error * sample_weight.float()).sum() / (
                sample_weight.float().sum() + 1e-6
            )
        else:
            loss_affine = output.new_tensor(0.0)

        if (
            self.lambda_gate_sup > 0.0
            and repair_support is not None
        ):
            with torch.autocast(device_type=gate.device.type, enabled=False):
                gate_prob = gate.float().clamp(1e-5, 1.0 - 1e-5)
                gate_sup_per_sample = F.binary_cross_entropy(
                    gate_prob,
                    repair_support.float(),
                    reduction="none",
                ).flatten(1).mean(dim=1)
            loss_gate_sup = (gate_sup_per_sample * sample_weight.float()).sum() / (
                sample_weight.float().sum() + 1e-6
            )
            gate_target_mean = repair_support.float().mean()
        else:
            loss_gate_sup = output.new_tensor(0.0)
            gate_target_mean = (
                repair_support.float().mean()
                if repair_support is not None
                else output.new_tensor(0.0)
            )

        # A gate may open where either the physical prior or the target-derived
        # local support indicates renderer uncertainty.
        if repair_support is not None:
            allowed_repair = torch.maximum(repair_map, repair_support.to(repair_map.dtype))
            outside_allowed = (1.0 - allowed_repair).clamp(0.0, 1.0)
            preserve_region = (1.0 - repair_support).clamp(0.0, 1.0)
        else:
            outside_allowed = outside_repair
            preserve_region = outside_repair
        loss_gate = (gate * outside_allowed).mean()

        loss_preserve = ((output - coarse).abs() * preserve_region).mean()

        if photo_log_gain is not None and photo_bias is not None:
            loss_photo = photo_log_gain.abs().mean() + photo_bias.abs().mean()
        else:
            loss_photo = output.new_tensor(0.0)

        total = (
            self.lambda_coarse * loss_coarse
            + self.lambda_rec * loss_rec
            + self.lambda_fm * loss_fm
            + self.lambda_mse * loss_mse
            + self.lambda_ssim * loss_ssim
            + self.lambda_gate * loss_gate
            + self.lambda_gate_sup * loss_gate_sup
            + self.lambda_preserve * loss_preserve
            + self.lambda_endpoint * loss_endpoint
            + self.lambda_affine * loss_affine
            + self.lambda_photo * loss_photo
        )

        return {
            "loss_total": total,
            "loss_coarse": loss_coarse.detach(),
            "loss_rec": loss_rec.detach(),
            "loss_fm": loss_fm.detach(),
            "loss_mse": loss_mse.detach(),
            "loss_ssim": loss_ssim.detach(),
            "loss_gate": loss_gate.detach(),
            "loss_gate_sup": loss_gate_sup.detach(),
            "loss_preserve": loss_preserve.detach(),
            "loss_endpoint": loss_endpoint.detach(),
            "loss_affine": loss_affine.detach(),
            "loss_photo": loss_photo.detach(),
            "gate_target_mean": gate_target_mean.detach(),
        }
