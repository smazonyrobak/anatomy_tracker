"""Freshly initialized spatial coarse proposal on smooth plane quotients."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    FULL_FRAME_STATE_SIZE,
    full_frame_state_to_components,
)


COARSE_PROPOSAL_V6_HEAD_SCHEMA = "anatomy-tracker.coarse-proposal-head/v6"
COARSE_PROPOSAL_V6_PROBABILITIES_CALIBRATED = False
COARSE_PROPOSAL_V6_GEOMETRY = (
    "rp2-projector-n-outer-n",
    "invariant-closest-plane-vector-d-times-n",
    "horizontal-quotient-roll-v-and-u-outer-n",
)


class AntipodalPlaneProposalV6(nn.Module):
    """Spatially aware mixture of normalized full-catalogue categoricals."""

    def __init__(
        self,
        feature_channels: int,
        proposal_channels: int = 16,
        mixture_components: int = 8,
        spatial_bins_h_w: tuple[int, int] = (4, 4),
        offset_scale_um: float = 10000.0,
    ):
        super().__init__()
        if min(feature_channels, proposal_channels, mixture_components) < 1:
            raise ValueError("proposal dimensions must be positive")
        if (
            len(spatial_bins_h_w) != 2
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                for value in spatial_bins_h_w
            )
        ):
            raise ValueError("proposal spatial bins must be two positive integers")
        if not math.isfinite(offset_scale_um) or offset_scale_um <= 0.0:
            raise ValueError("proposal offset scale must be finite and positive")

        self.feature_channels = int(feature_channels)
        self.proposal_channels = int(proposal_channels)
        self.mixture_components = int(mixture_components)
        self.spatial_bins_h_w = tuple(spatial_bins_h_w)
        self.offset_scale_um = float(offset_scale_um)
        context_inputs = self.feature_channels * math.prod(self.spatial_bins_h_w)

        self.source_context = nn.Sequential(
            nn.Linear(context_inputs, self.feature_channels),
            nn.GELU(),
        )
        self.mixture_logit = nn.Linear(self.feature_channels, mixture_components)
        self.normal_query = nn.Linear(
            self.feature_channels, mixture_components * proposal_channels
        )
        self.offset_query = nn.Linear(
            self.feature_channels, mixture_components * proposal_channels
        )
        self.roll_query = nn.Linear(
            self.feature_channels, mixture_components * proposal_channels
        )
        self.normal_embedding = nn.Sequential(
            nn.Linear(9, proposal_channels),
            nn.GELU(),
            nn.Linear(proposal_channels, proposal_channels),
        )
        self.offset_embedding = nn.Sequential(
            nn.Linear(3, proposal_channels),
            nn.GELU(),
            nn.Linear(proposal_channels, proposal_channels),
        )
        self.roll_embedding = nn.Sequential(
            nn.Linear(12, proposal_channels),
            nn.GELU(),
            nn.Linear(proposal_channels, proposal_channels),
        )
        for head in (
            self.mixture_logit,
            self.normal_query,
            self.offset_query,
            self.roll_query,
        ):
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)

    def _geometry(
        self,
        cell_states: torch.Tensor,
        support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center, frame, _ = full_frame_state_to_components(cell_states)
        u, v, normal = frame.unbind(dim=-1)
        origin = torch.as_tensor(
            support_origin_ap_dv_ml_um,
            device=cell_states.device,
            dtype=cell_states.dtype,
        )
        if origin.shape != (3,) or not bool(torch.isfinite(origin).all()):
            raise ValueError("proposal support origin must be one finite 3-vector")

        normal_projector = torch.einsum("bki,bkj->bkij", normal, normal).flatten(2)
        signed_offset = ((center - origin) * normal).sum(dim=-1, keepdim=True)
        closest_plane_vector = signed_offset * normal / self.offset_scale_um
        invariant_roll = torch.cat(
            (v, torch.einsum("bki,bkj->bkij", u, normal).flatten(2)), dim=-1
        )
        return normal_projector, closest_plane_vector, invariant_roll

    def forward(
        self,
        source_features: torch.Tensor,
        cell_states: torch.Tensor,
        cell_log_mass: torch.Tensor,
        support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        *,
        expected_catalogue_cell_count: int,
    ) -> dict[str, object]:
        if source_features.ndim != 4 or source_features.shape[1] != self.feature_channels:
            raise ValueError("proposal source features must have shape (B,F,h,w)")
        if any(
            size < bins
            for size, bins in zip(source_features.shape[-2:], self.spatial_bins_h_w)
        ):
            raise ValueError("proposal feature map must cover every spatial bin")
        if (
            cell_states.ndim != 3
            or cell_states.shape[0] != source_features.shape[0]
            or cell_states.shape[-1] != FULL_FRAME_STATE_SIZE
        ):
            raise ValueError("proposal cell states must have shape (B,K,12)")
        batch, cells = cell_states.shape[:2]
        if (
            not isinstance(expected_catalogue_cell_count, int)
            or isinstance(expected_catalogue_cell_count, bool)
            or expected_catalogue_cell_count < 1
            or cells != expected_catalogue_cell_count
        ):
            raise ValueError("proposal requires the declared complete catalogue")

        probability_dtype = (
            torch.float32
            if source_features.dtype in (torch.float16, torch.bfloat16)
            else source_features.dtype
        )
        log_mass = torch.as_tensor(
            cell_log_mass,
            device=source_features.device,
            dtype=probability_dtype,
        )
        if log_mass.shape != (batch, cells) or not bool(torch.isfinite(log_mass).all()):
            raise ValueError("proposal cell log mass must be finite with shape (B,K)")
        if not torch.allclose(
            torch.logsumexp(log_mass, dim=1),
            torch.zeros(batch, device=log_mass.device, dtype=log_mass.dtype),
            atol=2e-6,
            rtol=0.0,
        ):
            raise ValueError("complete-catalogue cell log mass must have unit mass")

        geometry_dtype = self.normal_embedding[0].weight.dtype
        geometry_states = cell_states.to(
            device=source_features.device, dtype=geometry_dtype
        )
        normal, offset, roll = self._geometry(
            geometry_states, support_origin_ap_dv_ml_um
        )
        normal_embedding = self.normal_embedding(normal)
        offset_embedding = self.offset_embedding(offset)
        roll_embedding = self.roll_embedding(roll)

        pooled = F.adaptive_avg_pool2d(
            source_features, self.spatial_bins_h_w
        ).flatten(1)
        context = self.source_context(pooled.to(self.source_context[0].weight))
        mixture_log_probability = F.log_softmax(
            self.mixture_logit(context).to(probability_dtype), dim=1
        )

        def query(head: nn.Linear) -> torch.Tensor:
            return head(context).reshape(
                batch, self.mixture_components, self.proposal_channels
            )

        component_cell_log_score = self.proposal_channels ** -0.5 * (
            torch.einsum("bld,bkd->blk", query(self.normal_query), normal_embedding)
            + torch.einsum("bld,bkd->blk", query(self.offset_query), offset_embedding)
            + torch.einsum("bld,bkd->blk", query(self.roll_query), roll_embedding)
        )
        component_cell_log_probability = F.log_softmax(
            log_mass[:, None] + component_cell_log_score.to(probability_dtype), dim=2
        )
        component_cell_probability = component_cell_log_probability.exp()
        cell_log_probability = torch.logsumexp(
            mixture_log_probability[:, :, None] + component_cell_log_probability,
            dim=1,
        )
        cell_probability = cell_log_probability.exp()
        entropy = -(cell_probability * cell_log_probability).sum(dim=1)
        component_entropy = -(
            component_cell_probability * component_cell_log_probability
        ).sum(dim=2)

        return {
            "schema_version": COARSE_PROPOSAL_V6_HEAD_SCHEMA,
            "geometry_contract": COARSE_PROPOSAL_V6_GEOMETRY,
            "catalogue_complete": True,
            "catalogue_cell_count": cells,
            "probability_scope": "all_declared_catalogue_cells",
            "probabilities_calibrated": COARSE_PROPOSAL_V6_PROBABILITIES_CALIBRATED,
            "mixture_log_probability": mixture_log_probability,
            "mixture_probability": mixture_log_probability.exp(),
            "component_cell_log_score": component_cell_log_score,
            "component_cell_log_probability": component_cell_log_probability,
            "component_cell_probability": component_cell_probability,
            "component_entropy": component_entropy,
            "raw_full_catalogue_cell_log_probability": cell_log_probability,
            "raw_full_catalogue_cell_probability": cell_probability,
            "cell_log_probability": cell_log_probability,
            "cell_probability": cell_probability,
            "entropy": entropy,
            "normalized_entropy": entropy / max(math.log(cells), 1.0e-12),
        }


__all__ = [
    "AntipodalPlaneProposalV6",
    "COARSE_PROPOSAL_V6_GEOMETRY",
    "COARSE_PROPOSAL_V6_HEAD_SCHEMA",
    "COARSE_PROPOSAL_V6_PROBABILITIES_CALIBRATED",
]
