from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


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


def psnr_batch(
    pred: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """
    pred, target: [B, C, H, W] in [0,1]
    return: [B]
    """
    _check_4d(pred, "pred")
    _check_4d(target, "target")

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")

    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    psnr = 10.0 * torch.log10((data_range ** 2) / (mse + eps))
    return psnr


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
    """
    pred, target: [B, C, H, W] in [0,1]
    return: [B]
    """
    _check_4d(pred, "pred")
    _check_4d(target, "target")

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")

    b, c, h, w = pred.shape
    kernel = _make_gaussian_kernel(
        window_size=window_size,
        sigma=sigma,
        channels=c,
        device=pred.device,
        dtype=pred.dtype,
    )

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


def image_metrics_batch(
    pred: Tensor,
    target: Tensor,
    data_range: float = 1.0,
) -> dict:
    psnr = psnr_batch(pred, target, data_range=data_range)
    ssim = ssim_batch(pred, target, data_range=data_range)

    return {
        "psnr_batch": psnr,
        "ssim_batch": ssim,
        "psnr_mean": float(psnr.mean().item()),
        "ssim_mean": float(ssim.mean().item()),
    }
