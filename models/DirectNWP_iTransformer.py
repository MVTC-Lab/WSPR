from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from models import iTransformer
from models.WSPR import HistoryStats


# =============================================================================
# Training-setting-matched, gradient-repaired Direct-NWP comparator.
#
# This isolated comparator preserves the legacy Direct-NWP formulation:
#
#   y_final = DirectHead(history_context, future_nwp, history_prediction)
#
# It directly predicts absolute target power and does not add a residual to the
# PatchTST base forecast.  The only conceptual repair relative to
# the detached-gradient predecessor is that history_prediction is NOT detached before it
# conditions the direct head.  Consequently, target loss can update PatchTST.
# This release-only rename does not change the numerical operations.
# =============================================================================


class DirectNWPHead(nn.Module):
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
        self.target_head = nn.Linear(hidden, 1)

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
    ) -> torch.Tensor:
        batch, horizon_len, _ = future_nwp.shape

        hist_summary = HistoryStats.summary(x_hist)
        target_aux = HistoryStats.target_aux(x_hist)
        context = self.context(torch.cat([hist_summary, target_aux], dim=-1))
        context_steps = context.unsqueeze(1).expand(-1, horizon_len, -1)

        lead = torch.linspace(0.0, 1.0, horizon_len, device=future_nwp.device, dtype=future_nwp.dtype)
        lead = lead.view(1, horizon_len, 1).expand(batch, -1, -1)
        future_var, future_mag = self.future_stats(future_nwp)
        step_input = torch.cat(
            [future_nwp, context_steps, lead, future_var, future_mag, history_prediction],
            dim=-1,
        )
        step_input = torch.nan_to_num(step_input, nan=0.0, posinf=5.0, neginf=-5.0)

        hidden = self.step_net(step_input)
        lower, upper = HistoryStats.target_bounds(x_hist)
        span = (upper - lower).clamp_min(0.10)
        direct_target = lower + span * torch.sigmoid(self.target_head(hidden))

        available = nwp_available.view(batch, 1, 1).to(dtype=future_nwp.dtype)
        return available * direct_target + (1.0 - available) * history_prediction


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

        self.backbone = iTransformer.Model(configs)
        self.direct_head = DirectNWPHead(
            enc_in=self.enc_in,
            known_future_dim=self.known_future_dim,
            d_model=configs.d_model,
            dropout=configs.dropout,
        )

        self.auxiliary_loss: Optional[torch.Tensor] = None
        self.last_history_forecast: Optional[torch.Tensor] = None
        self.last_direct_prediction: Optional[torch.Tensor] = None
        self.last_nwp_delta: Optional[torch.Tensor] = None
        self.last_raw_nwp_delta: Optional[torch.Tensor] = None
        self.last_residual_gate: Optional[torch.Tensor] = None
        self.last_nwp_reliability: Optional[torch.Tensor] = None
        self.last_future_nwp: Optional[torch.Tensor] = None

        print(
            "Training-setting-matched, gradient-repaired Direct-NWP comparator loaded: "
            "PatchTST backbone + direct absolute-power target head"
        )

    def set_reliability_target(self, target: Optional[torch.Tensor]) -> None:
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
            target_prediction = self.direct_head(
                x_hist=x_enc,
                future_nwp=future_nwp,
                history_prediction=history_prediction,
                nwp_available=nwp_available,
            )
        else:
            target_prediction = history_prediction

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

        # This delta is diagnostic only.  It is never added to the base forecast.
        direct_delta = target_prediction - history_prediction
        available = nwp_available.view(nwp_available.size(0), 1, 1)
        self.auxiliary_loss = None
        self.last_history_forecast = history_prediction.detach()
        self.last_direct_prediction = target_prediction.detach()
        self.last_nwp_delta = direct_delta.detach()
        self.last_raw_nwp_delta = direct_delta.detach()
        self.last_residual_gate = available.expand_as(history_prediction).detach()
        self.last_nwp_reliability = self.last_residual_gate
        self.last_future_nwp = future_nwp.detach()
        return outputs

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise ValueError("Direct-NWP comparator supports forecasting tasks only.")
