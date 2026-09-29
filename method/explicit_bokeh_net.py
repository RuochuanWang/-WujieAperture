from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from method.corrector import BokehCorrector
from method.explicit_renderer import RelativeExplicitRenderer
from method.repair_prior import LocalRepairPrior


class UnifiedPhotometricCalibrator(nn.Module):
    """Predict one global RGB affine lens correction before local flow repair."""

    def __init__(
        self,
        hidden_chans: int = 64,
        max_log_gain: float = 0.20,
        max_bias: float = 0.10,
    ) -> None:
        super().__init__()
        hidden_chans = max(int(hidden_chans), 8)
        self.max_log_gain = float(max_log_gain)
        self.max_bias = float(max_bias)
        self.net = nn.Sequential(
            nn.Linear(14, hidden_chans),
            nn.SiLU(),
            nn.Linear(hidden_chans, hidden_chans),
            nn.SiLU(),
            nn.Linear(hidden_chans, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        coarse: Tensor,
        source: Tensor,
        f_number: Tensor,
        aperture_scale: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Global affine corrections are small, so estimate them in FP32 even
        # when the image backbone is trained with BF16 autocast.
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            coarse_float = coarse.float()
            source_float = source.float()
            condition = torch.cat(
                [
                    coarse_float.mean(dim=(2, 3)),
                    coarse_float.std(dim=(2, 3), unbiased=False),
                    source_float.mean(dim=(2, 3)),
                    source_float.std(dim=(2, 3), unbiased=False),
                    (1.0 / f_number.float().clamp_min(1e-6)).view(-1, 1),
                    aperture_scale.float().reshape(aperture_scale.shape[0], -1)[:, :1],
                ],
                dim=1,
            )
            params = self.net(condition)
            log_gain_raw, bias_raw = params.chunk(2, dim=1)
            log_gain = torch.tanh(log_gain_raw) * self.max_log_gain
            gain = torch.exp(log_gain).view(-1, 3, 1, 1)
            bias = (torch.tanh(bias_raw) * self.max_bias).view(-1, 3, 1, 1)
            calibrated = coarse_float * gain + bias
        return calibrated, gain, bias, log_gain.view(-1, 3, 1, 1)


class ExplicitBokehNet(nn.Module):
    """
    Full model:
        source, depth, f_number, pos_map
            -> explicit renderer -> coarse, delta, radius_map
            -> repair prior     -> repair_map
            -> residual flow    -> bounded proposal, gate
            -> applied velocity -> gate * proposal
            -> final            -> coarse + applied velocity

    The corrector receives [calibrated_source, calibrated_coarse,
    residual_state]. Source supplies a photometrically aligned detail
    reference, while CoC maps control the feature hierarchy and the repair
    prior biases the gate. Raw depth is not passed to the corrector.
    """

    def __init__(
        self,
        corrector: BokehCorrector,
        renderer: Optional[RelativeExplicitRenderer] = None,
        repair_prior: Optional[LocalRepairPrior] = None,
        use_signed_delta: bool = False,
        normalize_radius_for_input: bool = True,
        unified_photometric_calibration_enabled: bool = False,
        unified_photometric_hidden_chans: int = 64,
        unified_photometric_max_log_gain: float = 0.20,
        unified_photometric_max_bias: float = 0.10,
        flow_noise_std: float = 0.02,
        flow_zero_prob: float = 0.5,
        flow_use_bridge_noise: bool = True,
        bound_residual_proposal: bool = False,
        residual_proposal_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.corrector = corrector
        self.renderer = renderer if renderer is not None else RelativeExplicitRenderer()
        self.repair_prior = repair_prior if repair_prior is not None else LocalRepairPrior()
        self.use_signed_delta = use_signed_delta
        self.normalize_radius_for_input = normalize_radius_for_input
        self.unified_photometric_calibration_enabled = bool(unified_photometric_calibration_enabled)
        self.unified_photometric_calibrator = (
            UnifiedPhotometricCalibrator(
                hidden_chans=unified_photometric_hidden_chans,
                max_log_gain=unified_photometric_max_log_gain,
                max_bias=unified_photometric_max_bias,
            )
            if self.unified_photometric_calibration_enabled
            else None
        )
        self.flow_noise_std = float(flow_noise_std)
        self.flow_zero_prob = float(flow_zero_prob)
        self.flow_use_bridge_noise = flow_use_bridge_noise
        self.bound_residual_proposal = bool(bound_residual_proposal)
        self.residual_proposal_scale = max(float(residual_proposal_scale), 1e-6)

    def _prepare_residual_proposal(self, proposal: Tensor) -> Tensor:
        if not self.bound_residual_proposal:
            return proposal
        scale = self.residual_proposal_scale
        return scale * torch.tanh(proposal / scale)

    def _prepare_radius_features(
        self,
        delta: Tensor,
        radius_map: Tensor,
        signed_radius_map: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        max_radius = max(float(self.renderer.max_radius), 1e-6)

        if self.normalize_radius_for_input:
            radius_feat = (radius_map / max_radius).clamp(0.0, 1.0)
        else:
            radius_feat = radius_map

        if signed_radius_map is not None:
            if self.normalize_radius_for_input:
                # Stable signed defocus feature for the corrector:
                # preserve foreground/background sign, but keep the dynamic
                # range bounded by the explicit blur radius.
                delta_feat = (signed_radius_map / max_radius).clamp(-1.0, 1.0)
            else:
                delta_feat = signed_radius_map
        elif self.use_signed_delta:
            delta_feat = torch.sign(delta)
        else:
            # Fallback for older renderers without signed radius outputs.
            delta_feat = torch.tanh(delta)

        return radius_feat, delta_feat

    def _prepare_coc_condition(
        self,
        delta: Tensor,
        radius_map: Tensor,
        signed_radius_map: Optional[Tensor],
    ) -> Tensor:
        radius_feat, delta_feat = self._prepare_radius_features(delta, radius_map, signed_radius_map)
        return torch.cat([radius_feat, delta_feat], dim=1)

    def _prepare_flow_input(
        self,
        source: Tensor,
        coarse: Tensor,
        z: Tensor,
    ) -> Tensor:
        # Source supplies details that the explicit blur can no longer recover;
        # CoC modulation remains responsible for aperture control.
        return torch.cat([source, coarse, z], dim=1)  # 3 + 3 + 3 = 9

    def _sample_bridge_start(self, residual: Tensor) -> Tensor:
        if (not self.flow_use_bridge_noise) or self.flow_noise_std <= 0.0:
            return torch.zeros_like(residual)

        noise = torch.randn_like(residual) * self.flow_noise_std
        if self.flow_zero_prob <= 0.0:
            return noise
        if self.flow_zero_prob >= 1.0:
            return torch.zeros_like(residual)

        keep_zero = torch.rand(
            residual.shape[0],
            1,
            1,
            1,
            device=residual.device,
            dtype=residual.dtype,
        ) < self.flow_zero_prob
        return torch.where(keep_zero, torch.zeros_like(residual), noise)

    def forward(
        self,
        source: Tensor,
        depth: Tensor,
        f_number: Tensor,
        pos_map: Tensor,
        mask: Optional[Tensor] = None,
        focus_depth: Optional[Tensor] = None,
        target: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        explicit_out = self.renderer(
            source=source,
            depth=depth,
            f_number=f_number,
            focus_depth=focus_depth,
            subject_mask=mask,
        )

        coarse_physical = explicit_out["coarse"]
        delta = explicit_out["delta"]
        radius_map = explicit_out["radius_map"]
        signed_radius_map = explicit_out.get("signed_radius_map")
        focus_depth = explicit_out["focus_depth"]

        if self.unified_photometric_calibrator is not None:
            coarse, photo_gain, photo_bias, photo_log_gain = self.unified_photometric_calibrator(
                coarse=coarse_physical,
                source=source,
                f_number=f_number,
                aperture_scale=explicit_out["aperture_scale"],
            )
            # Keep the detail reference in the same photometric domain as the
            # calibrated coarse image. For a normalized linear blur operator,
            # R(g * I + b) = g * R(I) + b, so the same affine is consistent.
            source_reference = source.float() * photo_gain + photo_bias
        else:
            coarse = coarse_physical
            photo_gain = coarse.new_ones(coarse.shape[0], 3, 1, 1)
            photo_bias = coarse.new_zeros(coarse.shape[0], 3, 1, 1)
            photo_log_gain = coarse.new_zeros(coarse.shape[0], 3, 1, 1)
            source_reference = source

        repair_map = self.repair_prior(
            delta,
            radius_map,
            signed_radius_map=signed_radius_map,
            source=source,
            mask=mask,
        )
        max_radius = max(float(self.renderer.max_radius), 1e-6)
        att_range_factor = (radius_map / max_radius).mean(dim=(1, 2, 3))
        coc_condition = self._prepare_coc_condition(
            delta=delta,
            radius_map=radius_map,
            signed_radius_map=signed_radius_map,
        )

        z_zero = torch.zeros_like(coarse)
        tau_zero = torch.zeros(coarse.shape[0], device=coarse.device, dtype=coarse.dtype)
        corrector_in = self._prepare_flow_input(source_reference, coarse, z_zero)
        residual_proposal, gate = self.corrector(
            features=corrector_in,
            pos_map=pos_map,
            att_range_factor=att_range_factor.detach(),
            coc_condition=coc_condition.detach(),
            repair_prior=repair_map.detach(),
            tau=tau_zero,
            aperture_scale=explicit_out["aperture_scale"],
        )
        residual_proposal = self._prepare_residual_proposal(residual_proposal)
        residual_velocity = gate * residual_proposal
        output = coarse + residual_velocity

        flow_velocity_tau = None
        flow_gate_tau = None
        flow_applied_velocity_tau = None
        flow_target_velocity = None
        flow_tau = None
        flow_z0 = None
        if target is not None and self.training:
            residual_target = (target - coarse).detach()
            flow_z0 = self._sample_bridge_start(residual_target)
            flow_tau = torch.rand(
                source.shape[0],
                1,
                1,
                1,
                device=source.device,
                dtype=source.dtype,
            )
            flow_z_tau = (1.0 - flow_tau) * flow_z0 + flow_tau * residual_target
            # Source is a fixed observation in both branches; detach it here to
            # avoid retaining a second renderer graph without changing gradients.
            flow_input_tau = self._prepare_flow_input(
                source_reference.detach(),
                coarse.detach(),
                flow_z_tau,
            )
            flow_proposal_tau, flow_gate_tau = self.corrector(
                features=flow_input_tau,
                pos_map=pos_map,
                att_range_factor=att_range_factor.detach(),
                coc_condition=coc_condition.detach(),
                repair_prior=repair_map.detach(),
                tau=flow_tau,
                aperture_scale=explicit_out["aperture_scale"],
            )
            flow_proposal_tau = self._prepare_residual_proposal(flow_proposal_tau)
            flow_velocity_tau = flow_proposal_tau
            flow_applied_velocity_tau = flow_gate_tau * flow_proposal_tau
            flow_target_velocity = residual_target - flow_z0

        return {
            "output": output,
            "source_reference": source_reference,
            "coarse": coarse,
            "coarse_physical": coarse_physical,
            "delta": delta,
            "radius_map": radius_map,
            "signed_radius_map": signed_radius_map if signed_radius_map is not None else torch.sign(delta) * radius_map,
            "signed_coc": explicit_out.get("signed_coc"),
            "focus_depth": focus_depth,
            "repair_map": repair_map,
            "repair_map_physical": repair_map,
            "delta_rgb": residual_velocity,
            "residual_proposal": residual_proposal,
            "residual_velocity": residual_velocity,
            # Endpoint supervision applies to the actual image update, while
            # flow_velocity_tau remains the ungated residual velocity field.
            "endpoint_velocity": residual_velocity,
            "gate": gate,
            "aperture_gate_bias": self.corrector.compute_aperture_gate_bias(
                explicit_out["aperture_scale"],
                gate,
            ),
            "flow_velocity_tau": flow_velocity_tau,
            "flow_gate_tau": flow_gate_tau,
            "flow_applied_velocity_tau": flow_applied_velocity_tau,
            "flow_target_velocity": flow_target_velocity,
            "flow_tau": flow_tau,
            "kappa": explicit_out.get("kappa"),
            "aperture_radius": explicit_out.get("aperture_radius"),
            "lens_strength": explicit_out.get("lens_strength"),
            "aperture_scale": explicit_out.get("aperture_scale"),
            "source_strength": explicit_out.get("source_strength"),
            "photo_gain": photo_gain,
            "photo_bias": photo_bias,
            "photo_log_gain": photo_log_gain,
        }
