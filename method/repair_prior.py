from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def normalize_per_sample(x: Tensor, eps: float = 1e-6) -> Tensor:
    """
    x: [B, 1, H, W]
    normalize to [0,1] per sample
    """
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def gradient_magnitude(x: Tensor) -> Tensor:
    """
    x: [B, 1, H, W]
    simple finite-difference gradient magnitude
    """
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]

    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")

    g = torch.sqrt(dx * dx + dy * dy + 1e-12)
    return g


class LocalRepairPrior(nn.Module):
    """
    M_r = sigmoid(
        alpha * grad_term
      + beta * radius_term
      + gamma * image_edge_term
      + eta * mask_boundary_term
      - tau
    )

    grad_term:   normalized per sample
    radius_term: normalized by a FIXED global max radius
    """

    def __init__(
        self,
        alpha: float = 4.0,
        beta: float = 2.0,
        gamma: float = 0.0,
        eta: float = 0.0,
        tau: float = 2.5,
        max_radius: float = 24.0,
        normalize_grad: bool = True,
        aperture_aware: bool = False,
        learnable_core: bool = True,
        alpha_range: tuple[float, float] = (2.0, 6.0),
        beta_range: tuple[float, float] = (1.0, 4.0),
        tau_range: tuple[float, float] = (1.5, 3.5),
    ) -> None:
        super().__init__()
        self.learnable_core = learnable_core
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.tau_range = tau_range
        self.gamma = gamma
        self.eta = eta
        self.max_radius = max_radius
        self.normalize_grad = normalize_grad
        self.aperture_aware = bool(aperture_aware)

        if self.learnable_core:
            self.alpha_raw = nn.Parameter(self._raw_from_value(alpha, alpha_range))
            self.beta_raw = nn.Parameter(self._raw_from_value(beta, beta_range))
            self.tau_raw = nn.Parameter(self._raw_from_value(tau, tau_range))
        else:
            self.alpha_const = float(alpha)
            self.beta_const = float(beta)
            self.tau_const = float(tau)

    @staticmethod
    def _raw_from_value(value: float, value_range: tuple[float, float]) -> Tensor:
        lo, hi = value_range
        if not hi > lo:
            raise ValueError(f"Invalid value range: {value_range}")
        normalized = (float(value) - lo) / (hi - lo)
        normalized = min(max(normalized, 1e-4), 1.0 - 1e-4)
        raw = math.log(normalized / (1.0 - normalized))
        return torch.tensor(raw, dtype=torch.float32)

    @staticmethod
    def _bounded_value(raw: Tensor, value_range: tuple[float, float]) -> Tensor:
        lo, hi = value_range
        return lo + (hi - lo) * torch.sigmoid(raw)

    def current_alpha(self) -> Tensor:
        if self.learnable_core:
            return self._bounded_value(self.alpha_raw, self.alpha_range)
        return torch.tensor(float(self.alpha_const))

    def current_beta(self) -> Tensor:
        if self.learnable_core:
            return self._bounded_value(self.beta_raw, self.beta_range)
        return torch.tensor(float(self.beta_const))

    def current_tau(self) -> Tensor:
        if self.learnable_core:
            return self._bounded_value(self.tau_raw, self.tau_range)
        return torch.tensor(float(self.tau_const))

    def forward(
        self,
        delta: Tensor,
        radius_map: Tensor,
        signed_radius_map: Tensor | None = None,
        source: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        radius_term = radius_map / max(self.max_radius, 1e-6)
        radius_term = radius_term.clamp(0.0, 1.0)

        if self.aperture_aware and signed_radius_map is not None:
            # CoC gradients must retain their absolute aperture magnitude.
            # Per-image normalization would make a tiny f/8 edge as strong as
            # a large f/1.8 discontinuity and incorrectly open the repair gate.
            grad_delta = gradient_magnitude(signed_radius_map)
            grad_delta = (grad_delta / max(self.max_radius, 1e-6)).clamp(0.0, 1.0)
            radius_context = F.max_pool2d(radius_term, kernel_size=5, stride=1, padding=2)
        else:
            grad_delta = gradient_magnitude(delta)
            if self.normalize_grad:
                grad_delta = normalize_per_sample(grad_delta)
            radius_context = torch.ones_like(radius_term)

        alpha = self.current_alpha().to(device=delta.device, dtype=delta.dtype)
        beta = self.current_beta().to(device=delta.device, dtype=delta.dtype)
        tau = self.current_tau().to(device=delta.device, dtype=delta.dtype)

        logits = alpha * grad_delta + beta * radius_term

        if self.gamma > 0.0 and source is not None:
            gray = source.mean(dim=1, keepdim=True)
            image_edge = gradient_magnitude(gray)
            image_edge = normalize_per_sample(image_edge)
            logits = logits + self.gamma * image_edge * radius_context

        if self.eta > 0.0 and mask is not None:
            mask_boundary = gradient_magnitude(mask)
            mask_boundary = normalize_per_sample(mask_boundary)
            logits = logits + self.eta * mask_boundary * radius_context

        logits = logits - tau
        repair_map = torch.sigmoid(logits)
        return repair_map
