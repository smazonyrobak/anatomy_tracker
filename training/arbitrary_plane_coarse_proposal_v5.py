"""Randomly initialized scalable proposal over antipodal plane/frame cells."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    FULL_FRAME_STATE_SIZE,
    full_frame_state_to_components,
)


COARSE_PROPOSAL_V5_SCHEMA = "anatomy-tracker.antipodal-coarse-proposal/v5"
COARSE_PROPOSAL_PROBABILITIES_CALIBRATED = False
COARSE_PROPOSAL_GEOMETRY = (
    "canonical-rp2-normal",
    "signed-normal-offset",
    "canonical-inplane-roll-frame",
)


class AntipodalPlaneProposalV5(nn.Module):
    """Mixture of factorized normal/offset/roll energies over any cell set.

    The head evaluates only catalogue geometry, never an atlas render.  Its
    discrete posterior is therefore cheap over the full production catalogue.
    Multiple factorized components permit separated correlated modes while the
    downstream exact renderer remains responsible for top-M reranking.
    """

    def __init__(
        self,
        feature_channels: int,
        proposal_channels: int = 16,
        mixture_components: int = 8,
        offset_scale_um: float = 10000.0,
    ):
        super().__init__()
        if min(feature_channels, proposal_channels, mixture_components) < 1:
            raise ValueError("proposal dimensions must be positive")
        if not math.isfinite(offset_scale_um) or offset_scale_um <= 0.0:
            raise ValueError("proposal offset scale must be finite and positive")
        self.feature_channels = int(feature_channels)
        self.proposal_channels = int(proposal_channels)
        self.mixture_components = int(mixture_components)
        self.offset_scale_um = float(offset_scale_um)

        self.source_context = nn.Sequential(
            nn.Linear(feature_channels, feature_channels),
            nn.GELU(),
        )
        self.mixture_logit = nn.Linear(feature_channels, mixture_components)
        self.normal_query = nn.Linear(
            feature_channels, mixture_components * proposal_channels
        )
        self.offset_query = nn.Linear(
            feature_channels, mixture_components * proposal_channels
        )
        self.roll_query = nn.Linear(
            feature_channels, mixture_components * proposal_channels
        )
        self.normal_embedding = nn.Sequential(
            nn.Linear(3, proposal_channels),
            nn.GELU(),
            nn.Linear(proposal_channels, proposal_channels),
        )
        self.offset_embedding = nn.Sequential(
            nn.Linear(1, proposal_channels),
            nn.GELU(),
            nn.Linear(proposal_channels, proposal_channels),
        )
        self.roll_embedding = nn.Sequential(
            nn.Linear(6, proposal_channels),
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
        largest = normal.abs().argmax(dim=-1, keepdim=True)
        sign = torch.where(
            torch.gather(normal, -1, largest) < 0.0,
            -torch.ones_like(largest, dtype=normal.dtype),
            torch.ones_like(largest, dtype=normal.dtype),
        )
        canonical_normal = normal * sign
        canonical_u = u * sign
        origin = torch.as_tensor(
            support_origin_ap_dv_ml_um,
            device=cell_states.device,
            dtype=cell_states.dtype,
        )
        if origin.shape != (3,) or not bool(torch.isfinite(origin).all()):
            raise ValueError("proposal support origin must be one finite 3-vector")
        signed_offset = (
            ((center - origin) * normal).sum(dim=-1, keepdim=True) * sign
            / self.offset_scale_um
        )
        roll_frame = torch.cat((canonical_u, v), dim=-1)
        return canonical_normal, signed_offset, roll_frame

    def forward(
        self,
        source_features: torch.Tensor,
        cell_states: torch.Tensor,
        cell_log_mass: torch.Tensor,
        support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
    ) -> dict[str, torch.Tensor | bool | str]:
        if source_features.ndim != 4 or source_features.shape[1] != self.feature_channels:
            raise ValueError("proposal source features must have shape (B,F,h,w)")
        if (
            cell_states.ndim != 3
            or cell_states.shape[0] != source_features.shape[0]
            or cell_states.shape[-1] != FULL_FRAME_STATE_SIZE
        ):
            raise ValueError("proposal cell states must have shape (B,K,12)")
        batch, cells = cell_states.shape[:2]
        if cells < 1:
            raise ValueError("proposal requires at least one catalogue cell")
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

        pooled = source_features.mean(dim=(-2, -1)).to(self.source_context[0].weight)
        context = self.source_context(pooled)
        mixture_log_probability = F.log_softmax(
            self.mixture_logit(context).to(probability_dtype), dim=-1
        )

        def query(head: nn.Linear) -> torch.Tensor:
            return head(context).reshape(
                batch, self.mixture_components, self.proposal_channels
            )

        scale = self.proposal_channels ** -0.5
        component_cell_log_score = scale * (
            torch.einsum("bld,bkd->blk", query(self.normal_query), normal_embedding)
            + torch.einsum("bld,bkd->blk", query(self.offset_query), offset_embedding)
            + torch.einsum("bld,bkd->blk", query(self.roll_query), roll_embedding)
        )
        cell_log_score = torch.logsumexp(
            mixture_log_probability[..., None]
            + component_cell_log_score.to(probability_dtype),
            dim=1,
        )
        cell_log_unnormalized_mass = log_mass + cell_log_score
        cell_log_probability = F.log_softmax(cell_log_unnormalized_mass, dim=1)
        probability = cell_log_probability.exp()
        entropy = -(probability * cell_log_probability).sum(dim=1)
        normalized_entropy = entropy / max(math.log(cells), 1.0e-12)
        return {
            "schema_version": COARSE_PROPOSAL_V5_SCHEMA,
            "geometry_contract": COARSE_PROPOSAL_GEOMETRY,
            "mixture_log_probability": mixture_log_probability,
            "cell_log_score": cell_log_score,
            "cell_log_unnormalized_mass": cell_log_unnormalized_mass,
            "cell_log_probability": cell_log_probability,
            "cell_probability": probability,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "probabilities_calibrated": COARSE_PROPOSAL_PROBABILITIES_CALIBRATED,
        }
