from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _build_gaussian_kernel(radius: float, device, dtype) -> Tensor:
    if radius < 0.5:
        return torch.tensor([[1.0]], device=device, dtype=dtype)

    sigma = max(radius / 2.0, 0.5)
    kernel_size = int(2 * math.ceil(3 * sigma) + 1)

    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()

    kernel = g[:, None] * g[None, :]
    kernel = kernel / kernel.sum()
    return kernel


def _build_disk_kernel(
    radius: float,
    device,
    dtype,
) -> Tensor:
    if radius < 0.5:
        return torch.tensor([[1.0]], device=device, dtype=dtype)

    kernel_size = int(2 * math.ceil(radius) + 1)
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dist = torch.sqrt(xx * xx + yy * yy)
    # Approximate sub-pixel aperture coverage over a one-pixel transition.
    # A hard binary edge visibly aliases when the CoC radius changes between
    # neighboring bins, especially around small circular highlights.
    kernel = (radius + 0.5 - dist).clamp(0.0, 1.0)

    if float(kernel.sum().item()) <= 0.0:
        kernel[kernel_size // 2, kernel_size // 2] = 1.0

    kernel = kernel / kernel.sum()
    return kernel


def _build_blur_kernel(
    radius: float,
    kernel_type: str,
    device,
    dtype,
) -> Tensor:
    if kernel_type == "gaussian":
        return _build_gaussian_kernel(radius, device, dtype)
    if kernel_type == "disk":
        return _build_disk_kernel(radius, device, dtype)
    raise ValueError(f"Unsupported renderer_kernel: {kernel_type}")


def _apply_blur(
    x: Tensor,
    radius: float,
    kernel_type: str = "gaussian",
    padding_mode: str = "zero",
) -> Tensor:
    """
    x: [1, C, H, W]
    """
    if radius < 0.5:
        return x

    kernel = _build_blur_kernel(
        radius,
        kernel_type,
        x.device,
        x.dtype,
    )
    k = kernel.shape[0]
    c = x.shape[1]

    kernel = kernel.view(1, 1, k, k).repeat(c, 1, 1, 1)

    pad = k // 2
    if padding_mode == "zero":
        return F.conv2d(x, kernel, padding=pad, groups=c)
    if padding_mode not in {"reflect", "replicate"}:
        raise ValueError(f"Unsupported renderer_padding: {padding_mode}")

    effective_padding = padding_mode
    if padding_mode == "reflect" and (x.shape[-2] <= pad or x.shape[-1] <= pad):
        effective_padding = "replicate"
    x = F.pad(x, (pad, pad, pad, pad), mode=effective_padding)
    return F.conv2d(x, kernel, padding=0, groups=c)


def _srgb_to_linear(x: Tensor) -> Tensor:
    """Convert normalized sRGB values to linear radiance."""
    x = x.clamp(0.0, 1.0)
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055).clamp_min(0.0).pow(2.4),
    )


def _linear_to_srgb(x: Tensor) -> Tensor:
    """Convert normalized linear radiance to sRGB values."""
    x = x.clamp(0.0, 1.0)
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * x.clamp_min(1e-12).pow(1.0 / 2.4) - 0.055,
    )


def _inverse_softplus(y: float) -> Tensor:
    """
    Initialize a raw parameter such that softplus(raw) ~= y.
    """
    y_tensor = torch.tensor(float(y), dtype=torch.float32)
    if float(y) > 20.0:
        return y_tensor
    return torch.log(torch.expm1(y_tensor))


def _inverse_sigmoid(y: float) -> Tensor:
    """
    Initialize a raw parameter such that sigmoid(raw) ~= y.
    """
    y = min(max(float(y), 1e-4), 1.0 - 1e-4)
    return torch.tensor(math.log(y / (1.0 - y)), dtype=torch.float32)


def _safe_inverse_depth(depth: Tensor, depth_floor: float = 1.0) -> Tensor:
    """
    Depth from the current VATD pipeline is per-sample normalized to [0, 1],
    so using 1 / depth directly is numerically unstable and physically dubious.

    We therefore use a stable, explicit pseudo-inverse-depth mapping:

        d_inv = 1 / (depth + depth_floor)

    This remains monotonic in depth, preserves the explicit mathematical
    relation, and avoids singular behavior near 0.
    """
    return 1.0 / (depth.clamp_min(0.0) + depth_floor)


def _depth_to_inverse_coordinate(
    depth: Tensor,
    depth_floor: float = 1.0,
    depth_representation: str = "metric_depth",
) -> Tensor:
    """
    Convert the dataset depth map into a near-high inverse-depth coordinate.

    Supported conventions:
      - inverse_like: depth is already near-high / far-low, e.g. common
        monocular depth visualizations.
      - metric_depth: depth is far-high / near-low, so use a stable inverse.
    """
    if depth_representation == "inverse_like":
        return depth.clamp(0.0, 1.0)
    if depth_representation == "metric_depth":
        return _safe_inverse_depth(depth, depth_floor=depth_floor)
    raise ValueError(
        f"Unsupported depth_representation: {depth_representation}. "
        "Expected 'inverse_like' or 'metric_depth'."
    )


class RelativeExplicitRenderer(nn.Module):
    """
    Explicit controllable coarse renderer.

    Compared with the original relative baseline, this version keeps the
    explicit control chain and follows the thin-lens aperture parameterization:

        aperture_scale(N) = relu(1 / N - a_s)
        signed_delta(x)   = 1 / z(x) - 1 / z_f
        q_t(x)            = kappa * aperture_scale(N) * signed_delta(x)
        radius(x)         = clip(|q_t(x)|, 0, r_max)

    a_s is a learned effective inverse source f-number. This keeps the
    source-to-target control chain explicit without assuming the input image is
    exactly a physical f/8 capture.
    """

    def __init__(
        self,
        ref_f_number: float = 8.0,
        base_scale: float = 18.0,
        focal_length: float = 1.0,
        max_radius: float = 24.0,
        num_bins: int = 8,
        renderer_kernel: str = "gaussian",
        renderer_padding: str = "zero",
        renderer_mode: str = "gather",
        renderer_linear_light: bool = False,
        renderer_depth_layers: int = 12,
        acceptable_coc_radius: float = 0.0,
        coc_transition_width: float = 0.0,
        depth_guidance_enabled: bool = False,
        depth_guidance_iterations: int = 1,
        depth_guidance_color_sigma: float = 0.12,
        depth_guidance_depth_sigma: float = 0.08,
        depth_guidance_blend: float = 0.5,
        layered_radius_interpolation: bool = False,
        layered_coverage_mode: str = "normalize",
        radiance_weighting_strength: float = 0.0,
        radiance_weighting_kernel: int = 15,
        radiance_weighting_threshold: float = 0.15,
        radiance_weighting_gamma: float = 1.0,
        highlight_recovery_strength: float = 0.0,
        highlight_recovery_threshold: float = 0.75,
        highlight_recovery_gamma: float = 2.0,
        focus_mode: str = "center_weighted_median",
        focus_patch_ratio: float = 0.25,
        trim_ratio: float = 0.10,
        inverse_depth_floor: float = 1.0,
        depth_representation: str = "metric_depth",
        learnable_kappa: bool = True,
        learnable_focus_bias: bool = True,
        learnable_source_strength: bool = True,
        source_strength_init: float = 0.125,
    ) -> None:
        super().__init__()
        self.ref_f_number = ref_f_number
        self.base_scale = base_scale
        self.focal_length = focal_length
        self.max_radius = max_radius
        self.num_bins = num_bins
        self.renderer_kernel = renderer_kernel
        self.renderer_padding = renderer_padding
        self.renderer_mode = str(renderer_mode)
        self.renderer_linear_light = bool(renderer_linear_light)
        self.renderer_depth_layers = int(renderer_depth_layers)
        self.acceptable_coc_radius = float(acceptable_coc_radius)
        self.coc_transition_width = float(coc_transition_width)
        self.depth_guidance_enabled = bool(depth_guidance_enabled)
        self.depth_guidance_iterations = int(depth_guidance_iterations)
        self.depth_guidance_color_sigma = float(depth_guidance_color_sigma)
        self.depth_guidance_depth_sigma = float(depth_guidance_depth_sigma)
        self.depth_guidance_blend = float(depth_guidance_blend)
        self.layered_radius_interpolation = bool(layered_radius_interpolation)
        self.layered_coverage_mode = str(layered_coverage_mode)
        self.radiance_weighting_strength = float(radiance_weighting_strength)
        self.radiance_weighting_kernel = int(radiance_weighting_kernel)
        self.radiance_weighting_threshold = float(radiance_weighting_threshold)
        self.radiance_weighting_gamma = float(radiance_weighting_gamma)
        self.highlight_recovery_strength = float(highlight_recovery_strength)
        self.highlight_recovery_threshold = float(highlight_recovery_threshold)
        self.highlight_recovery_gamma = float(highlight_recovery_gamma)
        self.focus_mode = focus_mode
        self.focus_patch_ratio = focus_patch_ratio
        self.trim_ratio = trim_ratio
        self.inverse_depth_floor = inverse_depth_floor
        self.depth_representation = str(depth_representation)
        self.learnable_source_strength = learnable_source_strength

        radius_bins = torch.linspace(0.0, max_radius, num_bins)
        self.register_buffer("radius_bins", radius_bins)

        if self.renderer_kernel not in {"gaussian", "disk"}:
            raise ValueError(f"Unsupported renderer_kernel: {self.renderer_kernel}")
        if self.renderer_padding not in {"zero", "reflect", "replicate"}:
            raise ValueError(f"Unsupported renderer_padding: {self.renderer_padding}")
        if self.renderer_mode not in {"gather", "layered"}:
            raise ValueError(f"Unsupported renderer_mode: {self.renderer_mode}")
        if self.renderer_depth_layers < 2:
            raise ValueError("renderer_depth_layers must be at least 2")
        if self.acceptable_coc_radius < 0.0:
            raise ValueError("acceptable_coc_radius must be non-negative")
        if self.coc_transition_width < 0.0:
            raise ValueError("coc_transition_width must be non-negative")
        if self.depth_guidance_iterations < 1:
            raise ValueError("depth_guidance_iterations must be at least 1")
        if self.depth_guidance_color_sigma <= 0.0 or self.depth_guidance_depth_sigma <= 0.0:
            raise ValueError("depth-guidance sigmas must be positive")
        if not 0.0 <= self.depth_guidance_blend <= 1.0:
            raise ValueError("depth_guidance_blend must be in [0, 1]")
        if self.layered_coverage_mode not in {"normalize", "source"}:
            raise ValueError("layered_coverage_mode must be 'normalize' or 'source'")
        if self.radiance_weighting_strength < 0.0:
            raise ValueError("radiance_weighting_strength must be non-negative")
        if self.radiance_weighting_kernel < 3 or self.radiance_weighting_kernel % 2 == 0:
            raise ValueError("radiance_weighting_kernel must be an odd integer of at least 3")
        if not 0.0 <= self.radiance_weighting_threshold < 1.0:
            raise ValueError("radiance_weighting_threshold must be in [0, 1)")
        if self.radiance_weighting_gamma <= 0.0:
            raise ValueError("radiance_weighting_gamma must be positive")
        if self.highlight_recovery_strength < 0.0:
            raise ValueError("highlight_recovery_strength must be non-negative")
        if not 0.0 <= self.highlight_recovery_threshold < 1.0:
            raise ValueError("highlight_recovery_threshold must be in [0, 1)")
        if self.highlight_recovery_gamma <= 0.0:
            raise ValueError("highlight_recovery_gamma must be positive")

        init_kappa = float(base_scale)
        init_scale = _inverse_softplus(init_kappa)
        if learnable_kappa:
            self.log_kappa = nn.Parameter(init_scale.clone())
        else:
            self.register_buffer("log_kappa", init_scale)

        if learnable_focus_bias:
            self.focus_bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("focus_bias", torch.zeros(1))

        source_strength_init = min(max(float(source_strength_init), 0.0), 1.0)
        if learnable_source_strength:
            self.source_strength_raw = nn.Parameter(_inverse_sigmoid(source_strength_init))
        else:
            self.register_buffer("source_strength_const", torch.tensor(source_strength_init, dtype=torch.float32))

    def effective_kappa(self) -> Tensor:
        # Keep the calibration factor positive while allowing optimization.
        return F.softplus(self.log_kappa) + 1e-6

    def effective_source_strength(self) -> Tensor:
        if self.learnable_source_strength:
            return torch.sigmoid(self.source_strength_raw)
        return self.source_strength_const.clamp(0.0, 1.0)

    def _lens_strength_scalar(self, f_number: float) -> float:
        aperture_radius = float(self.focal_length) / (2.0 * max(float(f_number), 1e-6))
        return aperture_radius * float(self.focal_length)

    def _get_center_patch(self, depth: Tensor) -> Tensor:
        _, _, h, w = depth.shape
        patch_h = max(int(round(h * self.focus_patch_ratio)), 8)
        patch_w = max(int(round(w * self.focus_patch_ratio)), 8)

        top = max((h - patch_h) // 2, 0)
        left = max((w - patch_w) // 2, 0)
        return depth[:, :, top:top + patch_h, left:left + patch_w]

    def _robust_focus_value(self, values: Tensor, fallback: Tensor) -> Tensor:
        valid = values[values > 1e-6]
        if valid.numel() == 0:
            return fallback

        sorted_vals, _ = torch.sort(valid)

        if self.focus_mode in {"center_median", "center_weighted_median"}:
            return sorted_vals.median()

        if self.focus_mode == "center_trimmed_median":
            n = sorted_vals.numel()
            trim = int(n * self.trim_ratio)
            if trim * 2 < n:
                sorted_vals = sorted_vals[trim:n - trim]
            return sorted_vals.median()

        raise NotImplementedError(f"Unknown focus mode: {self.focus_mode}")

    def estimate_focus_depth(
        self,
        depth: Tensor,
        subject_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        depth: [B, 1, H, W]
        subject_mask: [B, 1, H, W], optional subject-region mask
        return: [B, 1, 1, 1]

        The estimator remains explicit and lightweight:
        1. if subject mask is available, prefer subject-region depth
        2. otherwise use a center patch
        3. compute a robust statistic on valid pixels
        4. fall back to global valid median
        5. allow a tiny learnable bias for dataset-level calibration
        """
        b = depth.shape[0]
        center_patch = self._get_center_patch(depth)
        focus_values = []

        for i in range(b):
            if subject_mask is not None:
                subject_vals = depth[i][subject_mask[i] > 0.5]
            else:
                subject_vals = depth[i].new_empty(0)

            if subject_vals.numel() > 0:
                candidate_vals = subject_vals.reshape(-1)
            else:
                candidate_vals = center_patch[i].reshape(-1)

            global_vals = depth[i].reshape(-1)
            global_valid = global_vals[global_vals > 1e-6]
            if global_valid.numel() > 0:
                global_fallback = global_valid.median()
            else:
                global_fallback = torch.tensor(0.5, device=depth.device, dtype=depth.dtype)

            fd = self._robust_focus_value(candidate_vals, global_fallback)
            focus_values.append(fd)

        focus_depth = torch.stack(focus_values).view(b, 1, 1, 1)
        focus_depth = (focus_depth + self.focus_bias.view(1, 1, 1, 1)).clamp(1e-4, 1.0)
        return focus_depth

    def compute_aperture_radius(self, f_number: Tensor) -> Tensor:
        """
        f_number: [B]
        return: [B, 1, 1, 1]
        """
        aperture_radius = float(self.focal_length) / (2.0 * f_number.clamp_min(1e-6))
        return aperture_radius.view(-1, 1, 1, 1)

    def compute_aperture_scale(self, f_number: Tensor) -> Tensor:
        """
        Differential continuous source-to-target aperture scale.

        f_number: [B]
        return: [B, 1, 1, 1]
        """
        target_strength = 1.0 / f_number.clamp_min(1e-6)
        source_strength = self.effective_source_strength().to(device=f_number.device, dtype=f_number.dtype)
        aperture_scale = (target_strength - source_strength).clamp_min(0.0)
        return aperture_scale.view(-1, 1, 1, 1)

    @staticmethod
    def _shift_replicate(x: Tensor, dy: int, dx: int) -> Tensor:
        """Shift a tensor by one pixel without wrapping across image borders."""
        pad_top = max(dy, 0)
        pad_bottom = max(-dy, 0)
        pad_left = max(dx, 0)
        pad_right = max(-dx, 0)
        padded = F.pad(
            x,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="replicate",
        )
        top = pad_bottom
        left = pad_right
        return padded[:, :, top:top + x.shape[-2], left:left + x.shape[-1]]

    def _edge_aware_refine_inverse_depth(
        self,
        inv_depth: Tensor,
        guide_image: Optional[Tensor],
    ) -> Tensor:
        """
        Suppress small monocular-depth fluctuations without smoothing across
        strong RGB or depth discontinuities. This stabilizes near-focus CoC at
        object interiors while preserving the foreground silhouette.
        """
        if not self.depth_guidance_enabled or guide_image is None:
            return inv_depth

        guide = guide_image.to(device=inv_depth.device, dtype=inv_depth.dtype).clamp(0.0, 1.0)
        refined = inv_depth
        offsets = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),             (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        )
        color_sigma = max(self.depth_guidance_color_sigma, 1e-6)
        depth_sigma = max(self.depth_guidance_depth_sigma, 1e-6)

        for _ in range(self.depth_guidance_iterations):
            weighted_sum = refined
            weight_sum = torch.ones_like(refined)
            for dy, dx in offsets:
                shifted_depth = self._shift_replicate(refined, dy, dx)
                shifted_guide = self._shift_replicate(guide, dy, dx)
                color_distance = (guide - shifted_guide).abs().mean(dim=1, keepdim=True)
                depth_distance = (refined - shifted_depth).abs()
                weight = torch.exp(-color_distance / color_sigma - depth_distance / depth_sigma)
                weighted_sum = weighted_sum + weight * shifted_depth
                weight_sum = weight_sum + weight

            filtered = weighted_sum / weight_sum.clamp_min(1e-6)
            refined = refined + self.depth_guidance_blend * (filtered - refined)

        return refined

    def compute_radius_map(
        self,
        depth: Tensor,
        f_number: Tensor,
        focus_depth: Optional[Tensor] = None,
        subject_mask: Optional[Tensor] = None,
        guide_image: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        depth: [B, 1, H, W]
        f_number: [B]
        """
        if focus_depth is None:
            focus_depth = self.estimate_focus_depth(depth, subject_mask=subject_mask)

        inv_depth = _depth_to_inverse_coordinate(
            depth,
            depth_floor=self.inverse_depth_floor,
            depth_representation=self.depth_representation,
        )
        inv_depth = self._edge_aware_refine_inverse_depth(inv_depth, guide_image)
        inv_focus_depth = _depth_to_inverse_coordinate(
            focus_depth,
            depth_floor=self.inverse_depth_floor,
            depth_representation=self.depth_representation,
        )
        signed_delta = inv_depth - inv_focus_depth

        aperture_radius = self.compute_aperture_radius(f_number)
        aperture_scale = self.compute_aperture_scale(f_number)
        kappa = self.effective_kappa().view(1, 1, 1, 1).to(depth.dtype).to(depth.device)

        aperture_scale = aperture_scale.to(depth.dtype).to(depth.device)
        signed_coc = kappa * aperture_scale * signed_delta

        raw_radius = signed_coc.abs()
        if self.acceptable_coc_radius > 0.0 and self.coc_transition_width > 0.0:
            excess_radius = (raw_radius - self.acceptable_coc_radius).clamp_min(0.0)
            transition = (
                excess_radius / self.coc_transition_width
            ).clamp(0.0, 1.0)
            transition = transition * transition * (3.0 - 2.0 * transition)
            # The acceptable CoC is the image-space depth-of-field tolerance,
            # so only the radius beyond that tolerance should be rendered.
            # Multiplying the raw CoC here would add the tolerated radius back
            # after the transition and create a visible blur jump at edges.
            radius_map = excess_radius * transition
        else:
            # Preserve the legacy behavior when no transition width is set.
            radius_map = (raw_radius - self.acceptable_coc_radius).clamp_min(0.0)
        radius_map = radius_map.clamp(0.0, self.max_radius)
        signed_radius_map = radius_map * signed_coc.sign()
        source_strength = self.effective_source_strength().view(1, 1, 1, 1).to(depth.dtype).to(depth.device)

        return {
            "delta": signed_delta,
            "focus_depth": focus_depth,
            "radius_map": radius_map,
            "signed_radius_map": signed_radius_map,
            "signed_coc": signed_coc,
            "inv_depth": inv_depth,
            "inv_focus_depth": inv_focus_depth,
            "kappa": kappa.expand(depth.shape[0], 1, 1, 1),
            "aperture_radius": aperture_radius.to(depth.dtype).to(depth.device),
            "lens_strength": aperture_scale,
            "aperture_scale": aperture_scale,
            "source_strength": source_strength.expand(depth.shape[0], 1, 1, 1),
        }

    def render_quantized(self, source: Tensor, radius_map: Tensor) -> Tensor:
        """
        source: [B, 3, H, W]
        radius_map: [B, 1, H, W]

        The renderer remains explicit and bank-based, but we avoid hard argmin
        assignment. Instead, each pixel interpolates linearly between its two
        neighboring blur bins. This preserves the explicit aperture-to-radius
        mapping while giving the coarse branch a meaningful training signal.
        """
        b = source.shape[0]
        outputs = []
        bins = self.radius_bins.to(device=source.device, dtype=source.dtype)
        num_bins = len(bins)

        for i in range(b):
            x = source[i:i + 1]      # [1, 3, H, W]
            r = radius_map[i, 0]     # [H, W]
            h, w = r.shape

            blurred_bank = []
            for rb in bins:
                blurred_bank.append(
                    _apply_blur(
                        x,
                        float(rb.item()),
                        kernel_type=self.renderer_kernel,
                        padding_mode=self.renderer_padding,
                    )[0]
                )
            blurred_bank = torch.stack(blurred_bank, dim=0)  # [K, 3, H, W]

            upper_idx = torch.bucketize(r.reshape(-1), bins)
            upper_idx = upper_idx.clamp(1, num_bins - 1)
            lower_idx = upper_idx - 1

            lower_bins = bins[lower_idx].view(h, w)
            upper_bins = bins[upper_idx].view(h, w)
            denom = (upper_bins - lower_bins).clamp_min(1e-6)
            alpha = ((r - lower_bins) / denom).clamp(0.0, 1.0)

            pixels = h * w
            pixel_idx = torch.arange(pixels, device=source.device)
            blurred_flat = blurred_bank.view(num_bins, 3, pixels)

            lower_img = blurred_flat[lower_idx, :, pixel_idx].view(h, w, 3).permute(2, 0, 1)
            upper_img = blurred_flat[upper_idx, :, pixel_idx].view(h, w, 3).permute(2, 0, 1)

            alpha = alpha.unsqueeze(0)
            out = (1.0 - alpha) * lower_img + alpha * upper_img

            outputs.append(out.unsqueeze(0))

        return torch.cat(outputs, dim=0)

    def _blur_at_scalar_radius(self, x: Tensor, radius: Tensor) -> Tensor:
        """
        Blur one layer at a scalar CoC radius while retaining gradients through
        the interpolation weight. Kernel radii remain fixed bank entries.
        """
        bins = self.radius_bins.to(device=x.device, dtype=x.dtype)
        radius = radius.to(device=x.device, dtype=x.dtype).clamp(0.0, self.max_radius)

        upper_idx = torch.bucketize(radius.detach().reshape(1), bins)[0]
        upper_idx = upper_idx.clamp(1, len(bins) - 1)
        lower_idx = upper_idx - 1

        lower_radius = bins[lower_idx]
        upper_radius = bins[upper_idx]
        blend = ((radius - lower_radius) / (upper_radius - lower_radius).clamp_min(1e-6)).clamp(0.0, 1.0)

        lower = _apply_blur(
            x,
            float(lower_radius.item()),
            kernel_type=self.renderer_kernel,
            padding_mode=self.renderer_padding,
        )
        upper = _apply_blur(
            x,
            float(upper_radius.item()),
            kernel_type=self.renderer_kernel,
            padding_mode=self.renderer_padding,
        )
        return lower + blend * (upper - lower)

    def _filter_layer_with_radius_interpolation(
        self,
        image: Tensor,
        occupancy: Tensor,
        radius_map: Tensor,
        radiance_weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Scatter one depth layer through the radius bins touched by its pixels.
        In particular, zero-CoC pixels stay in the identity bin even when other
        pixels assigned to the same depth layer require visible defocus.
        """
        eps = 1e-6
        bins = self.radius_bins.to(device=image.device, dtype=image.dtype)
        bin_step = (bins[1] - bins[0]).clamp_min(eps)
        occupied = occupancy > 0.0
        radius_min = torch.where(
            occupied,
            radius_map,
            torch.full_like(radius_map, float("inf")),
        ).amin()
        radius_max = torch.where(
            occupied,
            radius_map,
            torch.full_like(radius_map, float("-inf")),
        ).amax()
        start_idx = int(torch.floor(radius_min.detach() / bin_step).clamp(0, len(bins) - 1).item())
        end_idx = int(torch.ceil(radius_max.detach() / bin_step).clamp(0, len(bins) - 1).item())

        filtered_premultiplied = torch.zeros_like(image)
        filtered_color_weight = torch.zeros_like(occupancy)
        filtered_occupancy = torch.zeros_like(occupancy)
        for bin_idx in range(start_idx, end_idx + 1):
            bin_radius = bins[bin_idx]
            radius_weight = (1.0 - (radius_map - bin_radius).abs() / bin_step).clamp(0.0, 1.0)
            bin_occupancy = occupancy * radius_weight
            color_weight = bin_occupancy * radiance_weight
            packed = torch.cat(
                [image * color_weight, color_weight, bin_occupancy],
                dim=1,
            )
            filtered = _apply_blur(
                packed,
                float(bin_radius.item()),
                kernel_type=self.renderer_kernel,
                padding_mode=self.renderer_padding,
            )
            filtered_premultiplied = filtered_premultiplied + filtered[:, :3]
            filtered_color_weight = filtered_color_weight + filtered[:, 3:4]
            filtered_occupancy = filtered_occupancy + filtered[:, 4:5]

        return filtered_premultiplied, filtered_color_weight, filtered_occupancy

    def _compute_radiance_weight(self, image: Tensor, radius_map: Tensor) -> Tensor:
        """Estimate a bounded radiance prior for color integration only."""
        if self.radiance_weighting_strength <= 0.0:
            return torch.ones_like(radius_map)

        luminance = (
            0.2126 * image[:, 0:1]
            + 0.7152 * image[:, 1:2]
            + 0.0722 * image[:, 2:3]
        )
        kernel = self.radiance_weighting_kernel
        pad = kernel // 2
        padding_mode = (
            "reflect"
            if luminance.shape[-2] > pad and luminance.shape[-1] > pad
            else "replicate"
        )
        local_mean = F.avg_pool2d(
            F.pad(luminance, (pad, pad, pad, pad), mode=padding_mode),
            kernel_size=kernel,
            stride=1,
        )
        relative_contrast = (
            (luminance - local_mean).clamp_min(0.0) / (local_mean + 0.05)
        )
        threshold = self.radiance_weighting_threshold
        highlight = (
            (relative_contrast - threshold) / max(1.0 - threshold, 1e-6)
        ).clamp(0.0, 1.0)
        highlight = highlight.pow(self.radiance_weighting_gamma)
        defocus = (radius_map / max(self.max_radius, 1e-6)).clamp(0.0, 1.0)
        return 1.0 + self.radiance_weighting_strength * highlight * defocus

    def _recover_defocused_highlights(self, image: Tensor, radius_map: Tensor) -> Tensor:
        """Recover compressed LDR radiance before aperture integration."""
        if self.highlight_recovery_strength <= 0.0:
            return image

        luminance = (
            0.2126 * image[:, 0:1]
            + 0.7152 * image[:, 1:2]
            + 0.0722 * image[:, 2:3]
        )
        threshold = self.highlight_recovery_threshold
        highlight = ((luminance - threshold) / max(1.0 - threshold, 1e-6)).clamp(0.0, 1.0)
        highlight = highlight.pow(self.highlight_recovery_gamma)

        defocus = (radius_map / max(self.max_radius, 1e-6)).clamp(0.0, 1.0)
        gain = 1.0 + self.highlight_recovery_strength * highlight * defocus
        return image * gain

    def _depth_layer_memberships(self, inv_depth: Tensor) -> Tensor:
        """
        Quantize each pixel to one ordered inverse-depth layer.

        The depth estimator is frozen and its maps are inputs rather than
        optimized variables, so hard assignment is intentional here. It keeps
        every visible surface opaque before aperture filtering; soft assignment
        would split one surface into several translucent copies and allow the
        far background to leak through a focused foreground.
        """
        depth_min = inv_depth.amin(dim=(2, 3), keepdim=True)
        depth_max = inv_depth.amax(dim=(2, 3), keepdim=True)
        normalized = (inv_depth - depth_min) / (depth_max - depth_min).clamp_min(1e-6)
        indices = torch.round(normalized * (self.renderer_depth_layers - 1)).long()
        indices = indices.clamp(0, self.renderer_depth_layers - 1)

        memberships = F.one_hot(
            indices[:, 0],
            num_classes=self.renderer_depth_layers,
        ).permute(0, 3, 1, 2)
        return memberships.to(dtype=inv_depth.dtype)

    def render_layered(
        self,
        source: Tensor,
        inv_depth: Tensor,
        radius_map: Tensor,
        radiance_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Occlusion-aware disk rendering through ordered depth layers.

        For every layer, geometry coverage and color-integration weights are
        filtered separately. Optional radiance weights affect only the color
        integral; geometric visibility remains controlled by occupancy. Layers
        are then composited from near to far, preventing foreground color
        leakage without averaging sparse highlight radiance out of its CoC.
        """
        memberships = self._depth_layer_memberships(inv_depth)
        if radiance_weight is None:
            radiance_weight = torch.ones_like(radius_map)
        if radiance_weight.shape != radius_map.shape:
            raise ValueError(
                "radiance_weight must match radius_map: "
                f"{radiance_weight.shape} vs {radius_map.shape}"
            )
        outputs = []
        eps = 1e-6

        for batch_idx in range(source.shape[0]):
            image = source[batch_idx:batch_idx + 1]
            sample_radius = radius_map[batch_idx:batch_idx + 1]
            sample_radiance_weight = radiance_weight[batch_idx:batch_idx + 1]
            sample_memberships = memberships[batch_idx:batch_idx + 1]

            accumulated = torch.zeros_like(image)
            accumulated_opacity = torch.zeros_like(sample_radius)
            transmittance = torch.ones_like(sample_radius)

            # inverse depth is near-high, hence the reversed layer order.
            for layer_idx in reversed(range(self.renderer_depth_layers)):
                occupancy = sample_memberships[:, layer_idx:layer_idx + 1]
                mass = occupancy.sum()
                if float(mass.detach().item()) <= 0.0:
                    continue

                if self.layered_radius_interpolation:
                    (
                        filtered_premultiplied,
                        filtered_color_weight,
                        filtered_occupancy,
                    ) = (
                        self._filter_layer_with_radius_interpolation(
                            image,
                            occupancy,
                            sample_radius,
                            sample_radiance_weight,
                        )
                    )
                else:
                    layer_radius = (sample_radius * occupancy).sum() / mass.clamp_min(eps)
                    color_weight = occupancy * sample_radiance_weight
                    premultiplied = image * color_weight
                    packed = torch.cat(
                        [premultiplied, color_weight, occupancy],
                        dim=1,
                    )
                    filtered = self._blur_at_scalar_radius(packed, layer_radius)
                    filtered_premultiplied = filtered[:, :3]
                    filtered_color_weight = filtered[:, 3:4]
                    filtered_occupancy = filtered[:, 4:5]

                filtered_occupancy = filtered_occupancy.clamp(0.0, 1.0)
                layer_color = filtered_premultiplied / filtered_color_weight.clamp_min(eps)

                contribution = transmittance * filtered_occupancy
                accumulated = accumulated + contribution * layer_color
                accumulated_opacity = accumulated_opacity + contribution
                transmittance = transmittance * (1.0 - filtered_occupancy)

            if self.layered_coverage_mode == "source":
                # Fill only rays left uncovered by the layered approximation.
                # Unlike global opacity normalization, this cannot amplify a
                # small bright foreground contribution into a white contour.
                output = accumulated + transmittance * image
            else:
                normalized = accumulated / accumulated_opacity.clamp_min(eps)
                output = torch.where(accumulated_opacity > eps, normalized, image)
            outputs.append(output)

        return torch.cat(outputs, dim=0)

    def forward(
        self,
        source: Tensor,
        depth: Tensor,
        f_number: Tensor,
        focus_depth: Optional[Tensor] = None,
        subject_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        radius_dict = self.compute_radius_map(
            depth,
            f_number,
            focus_depth=focus_depth,
            subject_mask=subject_mask,
            guide_image=source,
        )
        render_source = _srgb_to_linear(source) if self.renderer_linear_light else source
        render_source = self._recover_defocused_highlights(
            render_source,
            radius_dict["radius_map"],
        )
        radiance_weight = self._compute_radiance_weight(
            render_source,
            radius_dict["radius_map"],
        )
        if self.renderer_mode == "layered":
            coarse = self.render_layered(
                render_source,
                radius_dict["inv_depth"],
                radius_dict["radius_map"],
                radiance_weight=radiance_weight,
            )
        else:
            coarse = self.render_quantized(render_source, radius_dict["radius_map"])
        if self.renderer_linear_light:
            coarse = _linear_to_srgb(coarse)

        return {
            "coarse": coarse,
            "delta": radius_dict["delta"],
            "focus_depth": radius_dict["focus_depth"],
            "radius_map": radius_dict["radius_map"],
            "signed_radius_map": radius_dict["signed_radius_map"],
            "signed_coc": radius_dict["signed_coc"],
            "inv_depth": radius_dict["inv_depth"],
            "inv_focus_depth": radius_dict["inv_focus_depth"],
            "kappa": radius_dict["kappa"],
            "aperture_radius": radius_dict["aperture_radius"],
            "lens_strength": radius_dict["lens_strength"],
            "aperture_scale": radius_dict["aperture_scale"],
            "source_strength": radius_dict["source_strength"],
            "radiance_weight": radiance_weight,
        }
