from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import iTransformer


# =============================================================================
# WSPR: iTransformer-backed history-anchored NWP residual adapter
#
# Main idea:
#   y_final = y_history_backbone + reliability_gate * delta_nwp
#
# This version intentionally keeps the architecture small and auditable:
#   - iTransformer is the history backbone.
#   - Future NWP never predicts the whole PV power directly.
#   - Future NWP only provides a bounded residual correction.
#   - Future OT is never consumed; only leading known_future_dim channels can pass.
#   - If use_future_covariates=0, the adapter is disabled and the model is a
#     pure iTransformer history-only forecaster.
#
# Ablations supported through --wspr_ablation:
#   none             : reliability-gated residual adapter
#   no_reliability   : plain residual correction without learned reliability gate
#   no_nwp_residual  : same data contract, but residual correction disabled
#   no_physics/no_routing/no_state_routing: accepted as aliases for none/no-op
# =============================================================================


class HistoryStats:
    @staticmethod
    def target_bounds(x_hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target = x_hist[:, :, -1]
        low_k = max(1, target.size(1) // 4)
        high_k = max(1, target.size(1) // 8)
        low = target.topk(low_k, dim=1, largest=False).values.mean(dim=1, keepdim=True)
        high = target.topk(high_k, dim=1, largest=True).values.mean(dim=1, keepdim=True)
        std = target.std(dim=1, unbiased=False, keepdim=True)
        lower = low.unsqueeze(1)
        upper = torch.maximum(high + 0.25 * std + 0.05, low + 0.35).unsqueeze(1)
        return lower, upper

    @staticmethod
    def summary(x_hist: torch.Tensor) -> torch.Tensor:
        mean = x_hist.mean(dim=1)
        std = x_hist.std(dim=1, unbiased=False)
        last = x_hist[:, -1, :]
        slope = x_hist[:, -1, :] - x_hist[:, 0, :]
        amplitude = x_hist.amax(dim=1) - x_hist.amin(dim=1)
        return torch.cat([mean, std, last, slope, amplitude], dim=-1)

    @staticmethod
    def target_aux(x_hist: torch.Tensor) -> torch.Tensor:
        target = x_hist[:, :, -1]
        lookback = min(8, target.size(1))
        recent = target[:, -lookback:].mean(dim=1, keepdim=True)
        last = target[:, -1:]
        slope = (target[:, -1:] - target[:, -lookback].unsqueeze(-1)) / max(1, lookback - 1)
        volatility = target[:, -lookback:].std(dim=1, unbiased=False, keepdim=True)
        return torch.cat([last, recent, slope, volatility], dim=-1)


class ReliabilityAwareNWPAdapter(nn.Module):
    def __init__(self, enc_in: int, known_future_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.enc_in = enc_in
        self.known_future_dim = max(0, known_future_dim)

        hidden = max(64, d_model * 2)
        context_dim = enc_in * 5 + 4
        self.context = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )

        step_dim = self.known_future_dim + d_model + 4
        self.step_net = nn.Sequential(
            nn.LayerNorm(step_dim),
            nn.Linear(step_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(hidden, 1)
        self.reliability_head = nn.Linear(hidden, 1)
        self.max_residual_logit = nn.Parameter(torch.tensor(-0.25))
        self.gate_floor_weight = nn.Parameter(torch.zeros(1, hidden))
        self.gate_floor_bias = nn.Parameter(torch.tensor([-0.5]))

    @staticmethod
    def future_stats(future_nwp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        variability = future_nwp.std(dim=-1, unbiased=False, keepdim=True)
        magnitude = future_nwp.abs().mean(dim=-1, keepdim=True)
        return variability, magnitude

    def forward(
        self,
        x_hist: torch.Tensor,
        future_nwp: torch.Tensor,
        history_prediction: torch.Tensor,
        nwp_available: torch.Tensor,
        ablation: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, horizon_len, _ = future_nwp.shape

        hist_summary = HistoryStats.summary(x_hist)
        target_aux = HistoryStats.target_aux(x_hist)
        context = self.context(torch.cat([hist_summary, target_aux], dim=-1))
        context_steps = context.unsqueeze(1).expand(-1, horizon_len, -1)

        lead = torch.linspace(0.0, 1.0, horizon_len, device=future_nwp.device, dtype=future_nwp.dtype)
        lead = lead.view(1, horizon_len, 1).expand(batch, -1, -1)
        future_var, future_mag = self.future_stats(future_nwp)
        history_level = history_prediction.detach()
        step_input = torch.cat([future_nwp, context_steps, lead, future_var, future_mag, history_level], dim=-1)
        step_input = torch.nan_to_num(step_input, nan=0.0, posinf=5.0, neginf=-5.0)

        hidden = self.step_net(step_input)
        lower, upper = HistoryStats.target_bounds(x_hist)
        span = (upper - lower).clamp_min(0.10)
        max_scale = 0.80 * torch.sigmoid(self.max_residual_logit) + 0.05
        raw_delta = max_scale * span * torch.tanh(self.delta_head(hidden))

        available = nwp_available.view(batch, 1, 1).to(dtype=future_nwp.dtype)
        if ablation == "no_nwp_residual" or self.known_future_dim == 0:
            gate = torch.zeros_like(raw_delta)
        elif ablation == "no_reliability":
            gate = available.expand_as(raw_delta)
        else:
            reliability = torch.sigmoid(self.reliability_head(hidden))
            if horizon_len <= 24:
                # Short horizons are still strongly history-driven; keep the
                # stable calibrated gate instead of over-reacting to NWP noise.
                gate_floor = torch.full_like(reliability, 0.15)
                horizon_prior = (1.0 - 0.15 * lead).clamp(0.70, 1.0)
            else:
                # Longer horizons benefit more from NWP. Learn the gate floor
                # from the same residual evidence and relax horizon decay only
                # when that evidence is strong.
                gate_floor = 0.25 * torch.sigmoid(
                    F.linear(hidden, self.gate_floor_weight, self.gate_floor_bias)
                )
                floor_strength = (gate_floor / 0.25).clamp(0.0, 1.0)
                horizon_decay = 0.35 - 0.20 * floor_strength
                horizon_prior = (1.0 - horizon_decay * lead).clamp(0.55, 1.0)
            calibrated_reliability = gate_floor + (1.0 - gate_floor) * reliability
            gate = calibrated_reliability * horizon_prior * available

        gate = gate.clamp(0.0, 1.0)
        return gate * raw_delta, gate


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.use_future_covariates = int(getattr(configs, "use_future_covariates", 0))
        configured_known = int(getattr(configs, "known_future_dim", -1))
        if configured_known < 0:
            configured_known = max(0, self.enc_in - 1)
        self.known_future_dim = max(0, min(configured_known, self.enc_in - 1))
        self.reliability_target_idx = int(getattr(configs, "reliability_target_idx", -1))
        self.ablation = str(getattr(configs, "wspr_ablation", "none")).lower()
        self.residual_l1_weight = float(getattr(configs, "residual_l1_weight", 0.0))

        # PatchTST is intentionally used as the strong history backbone.
        self.backbone = iTransformer.Model(configs)
        self.adapter = ReliabilityAwareNWPAdapter(
            enc_in=self.enc_in,
            known_future_dim=self.known_future_dim,
            d_model=configs.d_model,
            dropout=configs.dropout,
        )

        self.auxiliary_loss: Optional[torch.Tensor] = None
        self.last_history_forecast: Optional[torch.Tensor] = None
        self.last_nwp_delta: Optional[torch.Tensor] = None
        self.last_residual_gate: Optional[torch.Tensor] = None
        self.last_nwp_reliability: Optional[torch.Tensor] = None

        print(
            "WSPR loaded: iTransformer history backbone + "
            "horizon-aware adaptive reliability NWP residual adapter"
        )

    def set_reliability_target(self, target: Optional[torch.Tensor]) -> None:
        # v8 does not use future reliability labels as supervision.
        return None

    def _select_future_window(self, x_dec: torch.Tensor) -> torch.Tensor:
        return x_dec[:, -self.pred_len :, :]

    def _sanitize_future(self, future_raw: torch.Tensor) -> torch.Tensor:
        safe = torch.zeros_like(future_raw)
        if self.use_future_covariates and self.known_future_dim > 0:
            safe[:, :, : self.known_future_dim] = future_raw[:, :, : self.known_future_dim]
        if 0 <= self.reliability_target_idx < safe.size(-1):
            safe[:, :, self.reliability_target_idx : self.reliability_target_idx + 1] = 0.0
        safe[:, :, -1:] = 0.0
        return safe

    def _future_available(self, future_safe: torch.Tensor) -> torch.Tensor:
        if self.use_future_covariates == 0 or self.known_future_dim == 0:
            return torch.zeros(future_safe.size(0), 1, device=future_safe.device, dtype=future_safe.dtype)
        known = future_safe[:, :, : self.known_future_dim]
        available = known.detach().abs().mean(dim=(1, 2), keepdim=False) > 1e-6
        return available.to(dtype=future_safe.dtype).view(-1, 1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        backbone_out = self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        backbone_out = backbone_out[:, -self.pred_len :, :]
        history_prediction = backbone_out[:, :, -1:]

        future_safe = self._sanitize_future(self._select_future_window(x_dec))
        nwp_available = self._future_available(future_safe)
        future_nwp = future_safe[:, :, : self.known_future_dim]

        if self.known_future_dim > 0:
            delta, gate = self.adapter(
                x_hist=x_enc,
                future_nwp=future_nwp,
                history_prediction=history_prediction,
                nwp_available=nwp_available,
                ablation=self.ablation,
            )
        else:
            delta = torch.zeros_like(history_prediction)
            gate = torch.zeros_like(history_prediction)

        target_prediction = history_prediction + delta

        if self.c_out == self.enc_in:
            outputs = backbone_out.clone()
        else:
            outputs = torch.zeros(
                backbone_out.size(0),
                backbone_out.size(1),
                self.c_out,
                device=backbone_out.device,
                dtype=backbone_out.dtype,
            )
            copy_dim = min(outputs.size(-1) - 1, future_safe.size(-1) - 1, self.known_future_dim)
            if copy_dim > 0:
                outputs[:, :, :copy_dim] = future_safe[:, :, :copy_dim]
        outputs[:, :, -1:] = target_prediction

        residual_l1 = (delta.abs() * nwp_available.view(nwp_available.size(0), 1, 1)).mean()
        if self.training and self.residual_l1_weight > 0:
            self.auxiliary_loss = self.residual_l1_weight * residual_l1
        else:
            self.auxiliary_loss = None

        self.last_history_forecast = history_prediction.detach()
        self.last_nwp_delta = delta.detach()
        self.last_residual_gate = gate.detach()
        self.last_nwp_reliability = gate.detach()
        return outputs

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise ValueError("WSPR v8 currently supports forecasting tasks only.")
