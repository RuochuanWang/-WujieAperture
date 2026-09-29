from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import LayerNorm
from torch.nn.init import trunc_normal_

from method.blocks import BlockMod, ResidualBlock
from method.nn_util import IdentityMod, SkipConnection, PatchEmbedIR, PatchUnEmbedIR


class SpatialAffineModulator(nn.Module):
    """
    Lightweight spatial FiLM-style modulation from CoC conditions.

    The final layer is zero-initialized so enabling the module starts as an
    identity mapping: x -> x.
    """

    def __init__(self, in_chans: int, out_chans: int, hidden_chans: int = 16):
        super().__init__()
        hidden_chans = max(int(hidden_chans), 4)
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_chans, out_chans * 2, kernel_size=1, padding=0, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if condition.shape[-2:] != target.shape[-2:]:
            condition = F.interpolate(
                condition,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        condition = condition.to(device=target.device, dtype=target.dtype)
        gamma_beta = self.net(condition)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return gamma, beta


class TimeAffineModulator(nn.Module):
    """
    Per-sample scale/shift modulation from the residual-flow time t.
    """

    def __init__(self, out_chans: int, hidden_chans: int = 64):
        super().__init__()
        hidden_chans = max(int(hidden_chans), 8)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_chans),
            nn.SiLU(),
            nn.Linear(hidden_chans, out_chans * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tau: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if tau.dim() > 1:
            tau = tau.reshape(tau.shape[0], -1)[:, :1]
        else:
            tau = tau.view(-1, 1)
        tau = tau.to(device=target.device, dtype=target.dtype)
        gamma_beta = self.net(tau).view(tau.shape[0], -1, 1, 1)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return gamma, beta


class BokehCorrector(nn.Module):
    """
    Bokehlicious-style backbone used only as a local corrector.

    Input:
        features : [B, C_in, H, W]
        pos_map  : [B, 2, H, W]
        optional CoC condition : [B, 2, H, W]
        optional repair prior  : [B, 1, H, W]

    Output:
        delta_rgb : [B, 3, H, W]
        gate      : [B, 1, H, W] in [0,1]
    """

    def __init__(
        self,
        in_chans=9,
        out_chans=4,
        img_range=1.0,
        in_stage_use_pos_map: bool = True,
        u_width=32,
        u_depth=2,
        u_block_config=None,
        u_skip_connections=None,
        enc_blk_nums=None,
        enc_blks_use_pos_map=None,
        embed_dims=None,
        depths=None,
        num_heads=None,
        init_values=None,
        heads_ranges=None,
        mlp_ratios=None,
        drop_path_rate=0.1,
        norm_layer=LayerNorm,
        patch_norm=True,
        use_checkpoints=None,
        chunkwise_recurrents=None,
        layerscales=None,
        layer_init_values=1e-6,
        positional_dfe=False,
        use_dfe_norm_layer=True,
        positional_conv_last=False,
        dec_blk_nums=None,
        dec_blks_use_pos_map=None,
        out_stage_use_pos_map=True,
        use_coc_modulation: bool = False,
        coc_condition_chans: int = 2,
        coc_mod_hidden_chans: int = 16,
        use_time_modulation: bool = False,
        time_mod_hidden_chans: int = 64,
        use_prior_gate: bool = False,
        prior_gate_lambda: float = 1.0,
        use_aperture_gate: bool = False,
        aperture_gate_init_offset: float = 0.0,
        aperture_gate_init_slope: float = 0.0,
    ):
        super().__init__()

        if embed_dims is None:
            raise ValueError("embed_dims must be provided")
        if depths is None:
            raise ValueError("depths must be provided")
        if num_heads is None:
            raise ValueError("num_heads must be provided")

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.img_range = img_range

        self.in_stage_use_pos_map = in_stage_use_pos_map

        self.u_width = u_width
        self.u_depth = u_depth
        self.u_block_config = u_block_config or {
            "dw_expand": 1.0,
            "ffn_expand": 2.0,
            "drop_out_rate": 0.0,
            "attention_type": "CA",
            "activation_type": "GELU",
            "kernel_size": 3,
            "inverted_conv": True,
        }

        self.u_skip_connections = u_skip_connections or [True for _ in range(u_depth)]
        self.enc_blk_nums = enc_blk_nums or [1 for _ in range(u_depth)]
        self.enc_blks_use_pos_map = enc_blks_use_pos_map or [True for _ in range(u_depth)]

        self.num_blocks = len(embed_dims)
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.init_values = init_values or [2 for _ in range(self.num_blocks)]
        self.heads_ranges = heads_ranges or [6 for _ in range(self.num_blocks)]
        self.mlp_ratios = mlp_ratios or [2 for _ in range(self.num_blocks)]
        self.drop_path_rate = drop_path_rate
        self.chunkwise_recurrents = chunkwise_recurrents or [True for _ in range(self.num_blocks)]
        self.layerscales = layerscales or [False for _ in range(self.num_blocks)]
        self.use_checkpoints = use_checkpoints or [False for _ in range(self.num_blocks)]
        self.positional_dfe = positional_dfe
        self.layer_init_values = layer_init_values
        self.patch_norm = patch_norm
        self.norm_layer = norm_layer
        self.use_dfe_norm_layer = use_dfe_norm_layer
        self.positional_conv_last = positional_conv_last

        self.dec_blk_nums = dec_blk_nums or [1 for _ in range(u_depth)]
        self.dec_blks_use_pos_map = dec_blks_use_pos_map or [True for _ in range(u_depth)]
        self.out_stage_use_pos_map = out_stage_use_pos_map
        self.use_coc_modulation = bool(use_coc_modulation)
        self.coc_condition_chans = int(coc_condition_chans)
        self.use_time_modulation = bool(use_time_modulation)
        self.use_prior_gate = bool(use_prior_gate)
        self.prior_gate_lambda = float(prior_gate_lambda)
        self.use_aperture_gate = bool(use_aperture_gate)
        if self.use_aperture_gate:
            init_slope = max(float(aperture_gate_init_slope), 1e-6)
            slope_raw = math.log(math.expm1(init_slope))
            self.aperture_gate_offset = nn.Parameter(
                torch.tensor(float(aperture_gate_init_offset), dtype=torch.float32)
            )
            self.aperture_gate_slope_raw = nn.Parameter(
                torch.tensor(slope_raw, dtype=torch.float32)
            )
        else:
            self.register_parameter("aperture_gate_offset", None)
            self.register_parameter("aperture_gate_slope_raw", None)
        extra_in_channels = 2 if self.in_stage_use_pos_map else 0
        self.in_stage = nn.Conv2d(
            in_channels=self.in_chans + extra_in_channels,
            out_channels=self.u_width,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )
        self.in_stage_2 = nn.Conv2d(
            in_channels=self.u_width,
            out_channels=self.u_width,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.enc_coc_mods = nn.ModuleList()
        self.enc_time_mods = nn.ModuleList()

        chan = self.u_width
        u_depths = range(0, self.u_depth)
        for num, depth, use_pos_map in zip(self.enc_blk_nums, u_depths, self.enc_blks_use_pos_map):
            self.enc_coc_mods.append(self._make_coc_mod(chan, coc_mod_hidden_chans))
            self.enc_time_mods.append(self._make_time_mod(chan, time_mod_hidden_chans))
            self.encoders.append(
                nn.ModuleList(
                    [BlockMod(chan, **self.u_block_config, depth=depth, use_pos_map=use_pos_map) for _ in range(num)]
                )
            )
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan = chan * 2

        self.conv_prep = nn.Conv2d(chan, self.embed_dims[0], 3, 1, 1)
        self.body_coc_mod = self._make_coc_mod(self.embed_dims[0], coc_mod_hidden_chans)
        self.body_time_mod = self._make_time_mod(self.embed_dims[0], time_mod_hidden_chans)

        # corrector input is not pure RGB, so use zero mean
        self.register_buffer("mean", torch.zeros(1, 1, 1, 1))

        self.patch_embed = PatchEmbedIR(
            embed_dim=self.embed_dims[0],
            norm_layer=self.norm_layer if self.patch_norm else None,
        )
        self.patch_unembed = PatchUnEmbedIR(embed_dim=self.embed_dims[-1])

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        self.blocks = nn.ModuleList()
        for i_block in range(self.num_blocks):
            self.blocks.append(
                ResidualBlock(
                    embed_dim=self.embed_dims[i_block],
                    embed_dim_next=(
                        self.embed_dims[i_block + 1]
                        if i_block < self.num_blocks - 1
                        else self.embed_dims[i_block]
                    ),
                    depth=self.depths[i_block],
                    num_heads=self.num_heads[i_block],
                    init_value=self.init_values[i_block],
                    heads_range=self.heads_ranges[i_block],
                    ffn_dim=int(self.mlp_ratios[i_block] * self.embed_dims[i_block]),
                    drop_path=dpr[sum(self.depths[:i_block]):sum(self.depths[:i_block + 1])],
                    norm_layer=self.norm_layer,
                    use_checkpoint=self.use_checkpoints[i_block],
                    layerscale=self.layerscales[i_block],
                    layer_init_values=self.layer_init_values,
                    use_pos_map=self.positional_dfe,
                )
            )

        self.norm = nn.LayerNorm(self.embed_dims[-1], eps=1e-6) if self.use_dfe_norm_layer else IdentityMod()
        self.conv_after_body = nn.Conv2d(self.embed_dims[-1], self.embed_dims[0], 3, 1, 1)

        conv_last_extra_channels = 2 if self.positional_conv_last else 0
        self.conv_last = nn.Conv2d(self.embed_dims[0] + conv_last_extra_channels, chan, 3, 1, 1)

        self.decoders = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.dec_coc_mods = nn.ModuleList()
        self.dec_time_mods = nn.ModuleList()
        for num, use_pos_map, skip_connection, depth in zip(
            self.dec_blk_nums,
            self.dec_blks_use_pos_map,
            self.u_skip_connections,
            reversed(list(u_depths)),
        ):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=chan, out_channels=2 * chan, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            self.skips.append(SkipConnection() if skip_connection else IdentityMod())
            chan = chan // 2
            self.dec_coc_mods.append(self._make_coc_mod(chan, coc_mod_hidden_chans))
            self.dec_time_mods.append(self._make_time_mod(chan, time_mod_hidden_chans))
            self.decoders.append(
                nn.ModuleList(
                    [BlockMod(chan, **self.u_block_config, use_pos_map=use_pos_map, depth=depth) for _ in range(num)]
                )
            )

        extra_out_channels = 2 if self.out_stage_use_pos_map else 0
        self.out_stage = nn.Conv2d(
            in_channels=self.u_width + extra_out_channels,
            out_channels=self.out_chans,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )

        self.apply(self._init_weights)
        self._init_condition_mods()
        self._init_output_head()

    def _make_coc_mod(self, out_chans: int, hidden_chans: int) -> nn.Module:
        if not self.use_coc_modulation:
            return IdentityMod()
        return SpatialAffineModulator(self.coc_condition_chans, out_chans, hidden_chans)

    def _make_time_mod(self, out_chans: int, hidden_chans: int) -> nn.Module:
        if not self.use_time_modulation:
            return IdentityMod()
        return TimeAffineModulator(out_chans, hidden_chans)

    def _apply_condition_mod(
        self,
        x: Tensor,
        coc_mod: nn.Module,
        time_mod: nn.Module,
        coc_condition: Tensor | None,
        tau: Tensor | None,
    ) -> Tensor:
        gamma = None
        beta = None
        if self.use_coc_modulation and coc_condition is not None:
            gamma, beta = coc_mod(coc_condition, x)
        if self.use_time_modulation and tau is not None:
            gamma_t, beta_t = time_mod(tau, x)
            gamma = gamma_t if gamma is None else gamma + gamma_t
            beta = beta_t if beta is None else beta + beta_t
        if gamma is None or beta is None:
            return x
        return (1.0 + gamma) * x + beta

    def compute_aperture_gate_bias(
        self,
        aperture_scale: Tensor | None,
        reference: Tensor,
    ) -> Tensor:
        if not self.use_aperture_gate or aperture_scale is None:
            return reference.new_zeros(reference.shape[0], 1, 1, 1)
        aperture = aperture_scale.float().reshape(aperture_scale.shape[0], -1)[:, :1]
        aperture = aperture.view(-1, 1, 1, 1)
        slope = F.softplus(self.aperture_gate_slope_raw.float())
        bias = self.aperture_gate_offset.float() + slope * aperture
        return bias.to(device=reference.device, dtype=torch.float32)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except Exception:
                pass

    def _init_condition_mods(self):
        for module in self.modules():
            if isinstance(module, SpatialAffineModulator):
                nn.init.zeros_(module.net[-1].weight)
                nn.init.zeros_(module.net[-1].bias)
            elif isinstance(module, TimeAffineModulator):
                nn.init.zeros_(module.net[-1].weight)
                nn.init.zeros_(module.net[-1].bias)

    def _init_output_head(self):
        """
        Conservative but trainable initialization:
              - output head weights are tiny, not exactly zero
              - delta_rgb starts near 0
              - gate starts near sigmoid(-1.5) ~= 0.182

        This keeps output close to coarse at initialization,
        while still allowing gradients to flow into the backbone.
        """
        nn.init.normal_(self.out_stage.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.out_stage.bias)

        # channels 0,1,2 => delta_rgb biases stay at 0
        # channel 3 => gate logits
        self.out_stage.bias.data[3] = -1.5

    def forward_features(
        self,
        x: Tensor,
        pos_map: Tensor | None = None,
        att_range_factor: Tensor | None = None,
    ) -> Tensor:
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        # keep dynamic attention path alive, but do not let it own control
        if att_range_factor is None:
            att_range_factor = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

        for block in self.blocks:
            x = block(x, x_size, pos_map=pos_map, att_range_factor=att_range_factor)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(
        self,
        features: Tensor,
        pos_map: Tensor | None = None,
        att_range_factor: Tensor | None = None,
        coc_condition: Tensor | None = None,
        repair_prior: Tensor | None = None,
        tau: Tensor | None = None,
        aperture_scale: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        features: [B, C_in, H, W]
        pos_map : [B, 2, H, W]
        """
        mean = self.mean.to(device=features.device, dtype=features.dtype)
        x_in = (features - mean) * self.img_range

        x = torch.cat((x_in, pos_map), dim=1) if self.in_stage_use_pos_map and pos_map is not None else x_in
        x = self.in_stage(x)
        x = self.in_stage_2(x)

        encs = []
        for encoder, down, use_pos_map, depth, coc_mod, time_mod in zip(
            self.encoders,
            self.downs,
            self.enc_blks_use_pos_map,
            range(0, self.u_depth),
            self.enc_coc_mods,
            self.enc_time_mods,
        ):
            if pos_map is not None and use_pos_map:
                pos_map_e = F.interpolate(
                    pos_map,
                    scale_factor=1 / (2 ** depth),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                pos_map_e = None

            x = self._apply_condition_mod(x, coc_mod, time_mod, coc_condition, tau)
            for blk in encoder:
                x = blk(x, pos_map=pos_map_e)
            encs.append(x)
            x = down(x)

        x_prep = self.conv_prep(x)
        x_prep_mod = self._apply_condition_mod(
            x_prep,
            self.body_coc_mod,
            self.body_time_mod,
            coc_condition,
            tau,
        )

        if pos_map is not None and (self.positional_dfe or self.positional_conv_last):
            pos_map_t = F.interpolate(
                pos_map,
                scale_factor=1 / (2 ** self.u_depth),
                mode="bilinear",
                align_corners=False,
            )
        else:
            pos_map_t = None

        x_after_body = self.forward_features(
            x_prep_mod,
            pos_map=pos_map_t,
            att_range_factor=att_range_factor,
        )
        res = self.conv_after_body(x_after_body) + x_prep

        if self.positional_conv_last and pos_map_t is not None:
            res = torch.cat((res, pos_map_t), dim=1)

        x = x + self.conv_last(res)

        for decoder, up, skip, enc_skip, use_pos_map_d, depth, coc_mod, time_mod in zip(
            self.decoders,
            self.ups,
            self.skips,
            encs[::-1],
            self.dec_blks_use_pos_map,
            reversed(range(0, self.u_depth)),
            self.dec_coc_mods,
            self.dec_time_mods,
        ):
            if pos_map is not None and use_pos_map_d:
                pos_map_d = F.interpolate(
                    pos_map,
                    scale_factor=1 / (2 ** depth),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                pos_map_d = None

            x = up(x)
            x = skip(x, enc_skip)
            x = self._apply_condition_mod(x, coc_mod, time_mod, coc_condition, tau)
            for blk in decoder:
                x = blk(x, pos_map=pos_map_d)

        if self.out_stage_use_pos_map and pos_map is not None:
            x = torch.cat((x, pos_map), dim=1)

        out = self.out_stage(x)
        delta_rgb = out[:, :3]
        gate_logits = out[:, 3:4]
        if self.use_prior_gate and repair_prior is not None:
            # The explicit prior is conditioning, not a trainable shortcut.
            # Compute logit in FP32: 1 - 1e-4 rounds to 1 in BF16 and would
            # otherwise create division-by-zero and non-finite gradients.
            prior = repair_prior.detach()
            if prior.shape[-2:] != gate_logits.shape[-2:]:
                prior = F.interpolate(prior, size=gate_logits.shape[-2:], mode="bilinear", align_corners=False)
            prior = prior.to(device=gate_logits.device, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
            prior_logit = torch.logit(prior, eps=1e-4)
            gate_logits = gate_logits.float() + self.prior_gate_lambda * prior_logit
        gate_logits = gate_logits.float() + self.compute_aperture_gate_bias(
            aperture_scale,
            gate_logits,
        )
        gate = torch.sigmoid(gate_logits.float()).to(dtype=out.dtype)

        return delta_rgb, gate
