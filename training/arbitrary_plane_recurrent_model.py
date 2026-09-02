"""Randomly initialized arbitrary-plane retrieval and recurrent refinement."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from training.arbitrary_plane_deformation_primitives import (
    AFFINE_FREE_DEFORMATION_TENSOR_KEYS,
    identity_pixel_map_yx,
    inactive_affine_free_deformation,
    warp_tensor_with_map_yx,
)
from training.arbitrary_plane_coarse_proposal_v5 import (
    AntipodalPlaneProposalV5,
)
from training.arbitrary_plane_full_frame_primitives import (
    FULL_FRAME_STATE_SIZE,
    FULL_FRAME_UPDATE_SIZE,
    compose_full_frame_state,
    full_frame_state_from_components,
    full_frame_state_to_components,
    render_finite_thickness_plane,
    so3_exp_map,
)


MODEL_INPUT_CHANNELS = 3
PLANE_TANGENT_COORDINATES = (
    "normal_tangent_along_local_u_rad",
    "normal_tangent_along_local_v_rad",
    "translation_along_local_normal_um",
)
PROBABILITIES_CALIBRATED = False
RETRIEVAL_TAIL_SCOPE = "complete_catalogue_at_retrieval"
_VERIFIED_CATALOGUE_FEATURE_CACHE_TOKEN = object()


def local_correlation(
    source: torch.Tensor,
    atlas: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Channel-cosine correlations for every local atlas displacement."""
    if source.ndim != 4 or atlas.shape != source.shape:
        raise ValueError("source and atlas features must share shape (B,C,H,W)")
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
        raise ValueError("correlation radius must be a nonnegative integer")
    output_dtype = source.dtype
    work_dtype = (
        torch.float32
        if source.dtype in (torch.float16, torch.bfloat16)
        else source.dtype
    )
    source = F.normalize(source.to(work_dtype), dim=1, eps=1e-6)
    atlas = F.normalize(atlas.to(work_dtype), dim=1, eps=1e-6)
    batch, channels, height, width = source.shape
    side = 2 * radius + 1
    patches = F.unfold(atlas, side, padding=radius).reshape(
        batch, channels, side * side, height, width
    )
    return (source[:, :, None] * patches).sum(dim=1).to(output_dtype)


def _pair_evidence(
    source: torch.Tensor,
    atlas: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    return torch.cat(
        (
            source,
            atlas,
            torch.abs(source - atlas),
            local_correlation(source, atlas, radius),
        ),
        dim=1,
    )


def canonicalize_representation_raster(
    rendered: torch.Tensor,
    representation_to_canonical_raster_affine: torch.Tensor,
) -> torch.Tensor:
    """Resample representation rasters into the common observed-image raster.

    The affine follows ``affine_grid`` convention: canonical output coordinates
    map to coordinates in the representation-rendered input. Exact signed-axis
    reflections therefore use diagonal entries of ``-1`` with zero translation.
    """
    if rendered.ndim != 6:
        raise ValueError("representation renders must have shape (B,K,R,C,H,W)")
    affine = torch.as_tensor(
        representation_to_canonical_raster_affine,
        device=rendered.device,
        dtype=rendered.dtype,
    )
    if affine.shape != rendered.shape[:3] + (2, 3) or not bool(
        torch.isfinite(affine).all()
    ):
        raise ValueError("raster affines must be finite with shape (B,K,R,2,3)")
    diagonal = torch.diagonal(affine[..., :2], dim1=-2, dim2=-1)
    off_diagonal = torch.stack((affine[..., 0, 1], affine[..., 1, 0]), dim=-1)
    if not bool(((diagonal == -1.0) | (diagonal == 1.0)).all()) or bool(
        (off_diagonal != 0.0).any()
    ) or bool((affine[..., :, 2] != 0.0).any()):
        raise ValueError(
            "raster affines must enumerate exact identity/horizontal/vertical/both flips"
        )
    flat = rendered.reshape(-1, *rendered.shape[3:])
    grid = F.affine_grid(
        affine.reshape(-1, 2, 3), flat.shape, align_corners=False
    )
    canonical = F.grid_sample(
        flat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return canonical.reshape_as(rendered)


def compose_antipodal_plane_frame_residual(
    state: torch.Tensor,
    residual: torch.Tensor,
    support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    """Compose a plane-tangent residual followed by finite-frame corrections.

    Residuals are ``[normal_tangent_u_rad, normal_tangent_v_rad, normal_offset_um,
    roll_rad, translation_u_um, translation_v_um, delta_log_basis_u,
    delta_log_basis_v, delta_shear]``. Plane rotation and offset are transported
    about one fixed physical support origin before roll, in-plane translation,
    and basis changes are right-composed in the rotated local frame.
    """
    state = torch.as_tensor(state)
    residual = torch.as_tensor(residual)
    if state.shape[-1:] != (FULL_FRAME_STATE_SIZE,):
        raise ValueError("full-frame state must end in 12 values")
    if residual.shape != state.shape[:-1] + (FULL_FRAME_UPDATE_SIZE,):
        raise ValueError("plane/frame residual must align with state and end in nine values")
    if state.device != residual.device or state.dtype != residual.dtype:
        raise ValueError("state and residual must share one device and dtype")
    if not torch.is_floating_point(residual) or not bool(torch.isfinite(residual).all()):
        raise ValueError("plane/frame residual must be finite floating point")

    center, frame, basis = full_frame_state_to_components(state)
    origin = torch.as_tensor(
        support_origin_ap_dv_ml_um, device=state.device, dtype=state.dtype
    )
    if origin.shape != (3,) or not bool(torch.isfinite(origin).all()):
        raise ValueError("support origin must be one finite physical 3-vector")
    u, v, normal = frame.unbind(dim=-1)
    tangent = residual[..., 0, None] * u + residual[..., 1, None] * v
    plane_rotation = so3_exp_map(torch.cross(normal, tangent, dim=-1))
    rotated_frame = plane_rotation @ frame
    rotated_normal = rotated_frame[..., :, 2]
    relative = center - origin
    signed_offset = (relative * normal).sum(dim=-1)
    inplane_center = relative - signed_offset[..., None] * normal
    rotated_inplane_center = (
        plane_rotation @ inplane_center[..., None]
    ).squeeze(-1)
    rotated_center = (
        origin
        + (signed_offset + residual[..., 2])[..., None] * rotated_normal
        + rotated_inplane_center
    )
    rotated_state = full_frame_state_from_components(
        rotated_center, rotated_frame, basis
    )
    local_update = torch.zeros_like(residual)
    local_update[..., 2] = residual[..., 3]
    local_update[..., 3:5] = residual[..., 4:6]
    local_update[..., 6:] = residual[..., 6:]
    return compose_full_frame_state(rotated_state, local_update)


class ConvGRUCell(nn.Module):
    """One spatial recurrent cell shared across all pose refinements."""

    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        joined = input_channels + hidden_channels
        self.gates = nn.Conv2d(joined, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(joined, hidden_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = torch.sigmoid(
            self.gates(torch.cat((inputs, hidden), dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((inputs, reset * hidden), dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


class ArbitraryPlaneRetrievalRefinementModel(nn.Module):
    """Cell-marginalized coarse retrieval and shared pose-only refinement.

    Each physical antipodal plane cell contains all declared finite-raster
    representations. Representation weights have unit mass within a cell; the
    physical cell measure is added exactly once after marginalization. Scored
    chunks remain unnormalized. Probabilities and an exact retrieval-time tail
    are exposed only after verifying unique, complete catalogue cell IDs.

    Every cell owns one canonical continuous state. Equivalent ``x/W,y/H``
    raster representations are generated from that state as external affine
    nuisance transforms and marginalized before one canonical update is
    composed. They never evolve as independent reflected continuous states,
    whose finite-grid centre correction would be nonlinear. The
    local Gaussian covers the two normal-tangent coordinates and normal offset;
    separated coarse modes remain separated.

    The pose-only API remains available. The joint wrapper may inject a fresh
    affine-free SVF decoder after a fixed pose-capture prefix; the decoder uses
    this same recurrent context, and its absolute map warps the next freshly
    rendered finite-thickness atlas before correlation.
    """

    def __init__(
        self,
        atlas_channels: int,
        feature_channels: int = 16,
        hidden_channels: int = 32,
        correlation_radius: int = 2,
        update_limits: tuple[float, ...] = (
            0.18,
            0.18,
            600.0,
            0.18,
            600.0,
            600.0,
            0.12,
            0.12,
            0.12,
        ),
        plane_tangent_scales: tuple[float, float, float] = (0.18, 0.18, 600.0),
        proposal_count: int | None = None,
        proposal_channels: int = 16,
        proposal_mixture_components: int = 8,
        proposal_offset_scale_um: float = 10000.0,
    ):
        super().__init__()
        if atlas_channels < 1 or feature_channels < 1 or hidden_channels < 1:
            raise ValueError("model channel counts must be positive")
        if correlation_radius < 0:
            raise ValueError("correlation radius must be nonnegative")
        if len(update_limits) != FULL_FRAME_UPDATE_SIZE or any(
            value <= 0.0 for value in update_limits
        ):
            raise ValueError("update limits must contain nine positive values")
        if len(plane_tangent_scales) != 3 or any(
            value <= 0.0 for value in plane_tangent_scales
        ):
            raise ValueError("plane tangent scales must contain three positive values")
        if proposal_count is not None and (
            not isinstance(proposal_count, int)
            or isinstance(proposal_count, bool)
            or proposal_count < 1
        ):
            raise ValueError("proposal count must be a positive integer or None")

        self.atlas_channels = int(atlas_channels)
        self.correlation_radius = int(correlation_radius)
        self.proposal_count = proposal_count
        self.histology_stem = nn.Sequential(
            nn.Conv2d(MODEL_INPUT_CHANNELS, feature_channels, 5, padding=2),
            nn.GroupNorm(1, feature_channels),
            nn.GELU(),
        )
        self.atlas_stem = nn.Sequential(
            nn.Conv2d(atlas_channels, feature_channels, 5, padding=2),
            nn.GroupNorm(1, feature_channels),
            nn.GELU(),
        )
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, stride=2, padding=1),
            nn.GroupNorm(1, feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, 3, stride=2, padding=1),
            nn.GroupNorm(1, feature_channels),
            nn.GELU(),
        )
        pair_channels = 3 * feature_channels + (2 * correlation_radius + 1) ** 2
        self.retrieval_pair_encoder = nn.Sequential(
            nn.Conv2d(pair_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.candidate_log_likelihood = nn.Linear(hidden_channels, 1, bias=False)
        self.candidate_update = nn.Linear(hidden_channels, FULL_FRAME_UPDATE_SIZE)
        self.candidate_plane_cholesky = nn.Linear(hidden_channels, 6)

        self.refinement_pair_encoder = nn.Sequential(
            nn.Conv2d(pair_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.recurrent_cell = ConvGRUCell(hidden_channels, hidden_channels)
        self.recurrent_update = nn.Linear(hidden_channels, FULL_FRAME_UPDATE_SIZE)
        self.recurrent_log_likelihood = nn.Linear(hidden_channels, 1, bias=False)
        self.recurrent_plane_cholesky = nn.Linear(hidden_channels, 6)

        self.register_buffer("update_limits", torch.tensor(update_limits))
        self.register_buffer(
            "plane_tangent_scales", torch.tensor(plane_tangent_scales)
        )
        self.coarse_proposal = (
            None
            if proposal_count is None
            else AntipodalPlaneProposalV5(
                feature_channels,
                proposal_channels=proposal_channels,
                mixture_components=proposal_mixture_components,
                offset_scale_um=proposal_offset_scale_um,
            )
        )
        self._complete_catalogue_feature_cache = None
        for head in (
            self.candidate_log_likelihood,
            self.candidate_update,
            self.recurrent_update,
            self.recurrent_log_likelihood,
        ):
            nn.init.normal_(head.weight, std=1e-3)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        nn.init.normal_(self.candidate_plane_cholesky.weight, std=1e-3)
        nn.init.zeros_(self.candidate_plane_cholesky.bias)
        nn.init.constant_(self.candidate_plane_cholesky.bias[:3], -2.0)
        nn.init.normal_(self.recurrent_plane_cholesky.weight, std=1e-3)
        nn.init.zeros_(self.recurrent_plane_cholesky.bias)
        nn.init.constant_(self.recurrent_plane_cholesky.bias[:3], -2.0)

    def encode_histology(
        self,
        image: torch.Tensor,
        outline: torch.Tensor,
        outline_available: torch.Tensor,
    ) -> torch.Tensor:
        """Encode image, optional outline, and an explicit availability plane."""
        if image.ndim != 4 or image.shape[1] != 1 or outline.shape != image.shape:
            raise ValueError("image and outline must share shape (B,1,H,W)")
        available = torch.as_tensor(
            outline_available, device=image.device, dtype=image.dtype
        )
        if available.shape == (image.shape[0],):
            available = available[:, None]
        if available.shape != (image.shape[0], 1):
            raise ValueError("outline availability must have shape (B,) or (B,1)")
        availability_plane = available[:, :, None, None].expand_as(image)
        encoded = self.histology_stem(
            torch.cat((image, outline * availability_plane, availability_plane), dim=1)
        )
        return self.shared_encoder(encoded)

    def _encode_atlas(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != self.atlas_channels:
            raise ValueError("rendered atlas must have shape (B,C,H,W)")
        return self.shared_encoder(self.atlas_stem(image))

    @staticmethod
    def _expanded_axial_schedule(
        value: torch.Tensor,
        batch: int,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.ndim == 1:
            return value[None].expand(batch * count, -1)
        if value.ndim == 2 and value.shape[0] == batch:
            return value[:, None].expand(batch, count, -1).reshape(batch * count, -1)
        raise ValueError("axial schedules must have shape (S,) or (B,S)")

    def _render_states(
        self,
        atlas_volume: torch.Tensor,
        states: torch.Tensor,
        output_shape_h_w: tuple[int, int],
        origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        axial_offsets_um: torch.Tensor,
        axial_weights: torch.Tensor,
    ) -> torch.Tensor:
        if states.ndim != 3 or states.shape[-1] != FULL_FRAME_STATE_SIZE:
            raise ValueError("states must have shape (B,N,12)")
        batch, count = states.shape[:2]
        offsets = self._expanded_axial_schedule(
            axial_offsets_um, batch, count, atlas_volume.device, atlas_volume.dtype
        )
        weights = self._expanded_axial_schedule(
            axial_weights, batch, count, atlas_volume.device, atlas_volume.dtype
        )
        tolerance = 16.0 * torch.finfo(atlas_volume.dtype).eps * torch.maximum(
            offsets.abs().amax(), offsets.new_tensor(1.0)
        )
        if not torch.allclose(
            offsets,
            -offsets.flip(-1),
            atol=float(tolerance),
            rtol=0.0,
        ) or not torch.allclose(
            weights,
            weights.flip(-1),
            atol=16.0 * torch.finfo(atlas_volume.dtype).eps,
            rtol=0.0,
        ):
            raise ValueError(
                "antipodal finite-thickness rendering requires symmetric offsets and weights"
            )
        rendered = render_finite_thickness_plane(
            atlas_volume,
            states.reshape(batch * count, FULL_FRAME_STATE_SIZE),
            output_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            offsets,
            weights,
        )
        return rendered.reshape(batch, count, self.atlas_channels, *output_shape_h_w)

    def _render_representations(
        self,
        atlas_volume: torch.Tensor,
        cell_states: torch.Tensor,
        raster_affine: torch.Tensor,
        output_shape_h_w: tuple[int, int],
        origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        axial_offsets_um: torch.Tensor,
        axial_weights: torch.Tensor,
    ) -> torch.Tensor:
        if cell_states.ndim != 3 or cell_states.shape[-1] != FULL_FRAME_STATE_SIZE:
            raise ValueError("cell states must have shape (B,K,12)")
        batch, cells = cell_states.shape[:2]
        affine = torch.as_tensor(
            raster_affine, device=atlas_volume.device, dtype=atlas_volume.dtype
        )
        if affine.ndim != 5 or affine.shape[:2] != (batch, cells) or affine.shape[-2:] != (2, 3):
            raise ValueError("raster affines must have shape (B,K,R,2,3)")
        representations = affine.shape[2]
        rendered = self._render_states(
            atlas_volume,
            cell_states,
            output_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            axial_offsets_um,
            axial_weights,
        )
        represented = rendered[:, :, None].expand(
            batch,
            cells,
            representations,
            self.atlas_channels,
            *output_shape_h_w,
        )
        return canonicalize_representation_raster(represented, affine)

    @staticmethod
    def _warp_representations_with_cell_map(
        rendered: torch.Tensor,
        cell_map_yx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one canonical output-to-input deformation to every cell representation."""
        if rendered.ndim != 6:
            raise ValueError("representation renders must have shape (B,K,R,C,H,W)")
        batch, cells, representations, channels, height, width = rendered.shape
        if cell_map_yx.shape != (batch, cells, 2, height, width):
            raise ValueError("cell deformation maps must have shape (B,K,2,H,W)")
        expanded_map = cell_map_yx[:, :, None].expand(
            batch, cells, representations, 2, height, width
        )
        return warp_tensor_with_map_yx(
            rendered.reshape(batch * cells * representations, channels, height, width),
            expanded_map.reshape(batch * cells * representations, 2, height, width),
        ).reshape_as(rendered)

    def _pair_context(
        self,
        source_features: torch.Tensor,
        rendered_atlas: torch.Tensor,
        encoder: nn.Module,
    ) -> torch.Tensor:
        batch, count, channels, height, width = rendered_atlas.shape
        atlas_features = self._encode_atlas(
            rendered_atlas.reshape(batch * count, channels, height, width)
        )
        source = source_features[:, None].expand(
            batch, count, *source_features.shape[1:]
        ).reshape(batch * count, *source_features.shape[1:])
        context = encoder(
            _pair_evidence(source, atlas_features, self.correlation_radius)
        )
        return context.reshape(batch, count, *context.shape[1:])

    @contextmanager
    def use_complete_catalogue_feature_cache(
        self,
        atlas_features: torch.Tensor,
        cell_id: torch.Tensor,
        retrieval_shape_h_w: tuple[int, int],
        *,
        _verification_token=None,
    ):
        """Activate one verified, same-checkpoint atlas cache for one eval call."""
        if self.training:
            raise ValueError("complete-catalogue feature caches are inference-only")
        if self._complete_catalogue_feature_cache is not None:
            raise RuntimeError("complete-catalogue feature cache contexts cannot be nested")
        features = torch.as_tensor(atlas_features)
        ids = torch.as_tensor(cell_id, device=features.device)
        already_verified = _verification_token is _VERIFIED_CATALOGUE_FEATURE_CACHE_TOKEN
        if (
            features.ndim != 5
            or features.shape[0] < 1
            or not torch.is_floating_point(features)
            or ids.ndim != 1
            or ids.shape[0] != features.shape[0]
            or ids.dtype == torch.bool
            or torch.is_floating_point(ids)
            or len(retrieval_shape_h_w) != 2
            or min(retrieval_shape_h_w) < 4
            or (
                not already_verified
                and (
                    not bool(torch.isfinite(features).all())
                    or not torch.equal(
                        ids.to(torch.long),
                        torch.arange(ids.numel(), device=ids.device),
                    )
                )
            )
        ):
            raise ValueError("cached atlas features require complete canonical cell coverage")
        self._complete_catalogue_feature_cache = {
            "atlas_features": features,
            "cell_id": ids.to(torch.long),
            "retrieval_shape_h_w": tuple(int(value) for value in retrieval_shape_h_w),
            "already_verified": bool(already_verified),
        }
        try:
            yield
        finally:
            self._complete_catalogue_feature_cache = None

    def score_catalogue_feature_chunk(
        self,
        source_features: torch.Tensor,
        atlas_features: torch.Tensor,
        cell_id: torch.Tensor,
        cell_log_mass: torch.Tensor,
        representation_log_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Exactly score one cached atlas-feature chunk without approximation."""
        if self.training:
            raise ValueError("cached atlas features cannot be used during training")
        features = torch.as_tensor(
            atlas_features, device=source_features.device, dtype=source_features.dtype
        )
        if features.ndim != 5 or features.shape[2:] != source_features.shape[1:]:
            raise ValueError("cached atlas features must have shape (K,R,F,h,w)")
        cells, representations = features.shape[:2]
        ids = torch.as_tensor(cell_id, device=source_features.device)
        if (
            ids.ndim != 1
            or ids.shape[0] != cells
            or ids.dtype == torch.bool
            or torch.is_floating_point(ids)
            or torch.unique(ids).numel() != cells
        ):
            raise ValueError("cached cell IDs must be one unique integer vector")
        ids = ids.to(torch.long)
        batch = source_features.shape[0]
        probability_dtype = (
            torch.float32
            if source_features.dtype in (torch.float16, torch.bfloat16)
            else source_features.dtype
        )
        log_mass = torch.as_tensor(
            cell_log_mass, device=source_features.device, dtype=probability_dtype
        )
        log_weight = torch.as_tensor(
            representation_log_weight,
            device=source_features.device,
            dtype=probability_dtype,
        )
        if log_mass.shape != (batch, cells) or not bool(torch.isfinite(log_mass).all()):
            raise ValueError("cell log mass must be finite with shape (B,K)")
        if log_weight.shape != (batch, cells, representations) or not bool(
            torch.isfinite(log_weight).all()
        ):
            raise ValueError("representation log weights must be finite with shape (B,K,R)")
        if not torch.allclose(
            torch.logsumexp(log_weight, dim=2),
            torch.zeros_like(log_mass),
            atol=2e-6,
            rtol=0.0,
        ):
            raise ValueError("representation weights must have unit mass within every cell")
        expanded = features[None].expand(batch, *features.shape).reshape(
            batch * cells * representations, *features.shape[2:]
        )
        source = source_features[:, None, None].expand(
            batch, cells, representations, *source_features.shape[1:]
        ).reshape(batch * cells * representations, *source_features.shape[1:])
        context = self.retrieval_pair_encoder(
            _pair_evidence(source, expanded, self.correlation_radius)
        ).reshape(batch, cells, representations, -1, *source_features.shape[-2:])
        representation_log_score = self.candidate_log_likelihood(
            context.mean(dim=(-2, -1))
        ).squeeze(-1)
        cell_log_evidence = torch.logsumexp(
            log_weight + representation_log_score.to(probability_dtype), dim=2
        )
        return {
            "cell_id": ids,
            "representation_log_score": representation_log_score,
            "cell_log_evidence": cell_log_evidence,
            "cell_log_unnormalized_mass": log_mass + cell_log_evidence,
        }

    def _bounded_update(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw) * self.update_limits.to(raw)[None]

    def _plane_cholesky(self, raw: torch.Tensor) -> torch.Tensor:
        work = raw.float() if raw.dtype in (torch.float16, torch.bfloat16) else raw
        diagonal = F.softplus(work[..., :3]) + 1e-4
        off_diagonal = 0.25 * torch.tanh(work[..., 3:])
        zero = torch.zeros_like(diagonal[..., 0])
        normalized = torch.stack(
            (
                torch.stack((diagonal[..., 0], zero, zero), dim=-1),
                torch.stack((off_diagonal[..., 0], diagonal[..., 1], zero), dim=-1),
                torch.stack(
                    (
                        off_diagonal[..., 1],
                        off_diagonal[..., 2],
                        diagonal[..., 2],
                    ),
                    dim=-1,
                ),
            ),
            dim=-2,
        )
        return normalized * self.plane_tangent_scales.to(work)[..., :, None]

    def score_catalogue_chunk(
        self,
        source_features: torch.Tensor,
        atlas_volume: torch.Tensor,
        cell_id: torch.Tensor,
        cell_states: torch.Tensor,
        cell_log_mass: torch.Tensor,
        representation_log_weight: torch.Tensor,
        representation_to_canonical_raster_affine: torch.Tensor,
        output_shape_h_w: tuple[int, int],
        origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        axial_offsets_um: torch.Tensor,
        axial_weights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score cells without normalizing a possibly incomplete catalogue chunk."""
        if cell_states.ndim != 3 or cell_states.shape[-1] != FULL_FRAME_STATE_SIZE:
            raise ValueError("cell states must have shape (B,K,12)")
        batch, cells = cell_states.shape[:2]
        representations = representation_to_canonical_raster_affine.shape[2]
        if source_features.shape[0] != batch:
            raise ValueError("source features and cell states must share batch size")
        ids = torch.as_tensor(cell_id, device=source_features.device)
        if ids.ndim != 1 or ids.shape[0] != cells or ids.dtype == torch.bool or torch.is_floating_point(ids):
            raise ValueError("cell IDs must be one integer vector of length K")
        ids = ids.to(torch.long)
        if torch.unique(ids).numel() != cells:
            raise ValueError("cell IDs must be unique within each scored chunk")

        probability_dtype = (
            torch.float32
            if source_features.dtype in (torch.float16, torch.bfloat16)
            else source_features.dtype
        )
        log_mass = torch.as_tensor(
            cell_log_mass, device=source_features.device, dtype=probability_dtype
        )
        log_weight = torch.as_tensor(
            representation_log_weight,
            device=source_features.device,
            dtype=probability_dtype,
        )
        if log_mass.shape != (batch, cells) or not bool(torch.isfinite(log_mass).all()):
            raise ValueError("cell log mass must be finite with shape (B,K)")
        if log_weight.shape != (batch, cells, representations) or not bool(
            torch.isfinite(log_weight).all()
        ):
            raise ValueError("representation log weights must be finite with shape (B,K,R)")
        if not torch.allclose(
            torch.logsumexp(log_weight, dim=2),
            torch.zeros_like(log_mass),
            atol=2e-6,
            rtol=0.0,
        ):
            raise ValueError("representation weights must have unit mass within every cell")

        canonical_render = self._render_representations(
            atlas_volume,
            cell_states,
            representation_to_canonical_raster_affine,
            output_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            axial_offsets_um,
            axial_weights,
        )
        context = self._pair_context(
            source_features,
            canonical_render.reshape(
                batch,
                cells * representations,
                self.atlas_channels,
                *output_shape_h_w,
            ),
            self.retrieval_pair_encoder,
        ).reshape(batch, cells, representations, -1, *source_features.shape[-2:])
        pooled = context.mean(dim=(-2, -1))
        representation_log_score = self.candidate_log_likelihood(pooled).squeeze(-1)
        score_work = representation_log_score.to(probability_dtype)
        cell_log_evidence = torch.logsumexp(log_weight + score_work, dim=2)
        cell_log_unnormalized_mass = log_mass + cell_log_evidence

        raw_canonical_update = self.candidate_update(pooled)
        canonical_update = torch.tanh(raw_canonical_update) * self.update_limits.to(
            raw_canonical_update
        )
        representation_log_conditional = (
            log_weight + score_work - cell_log_evidence[..., None]
        )
        accumulation_dtype = (
            torch.float32
            if canonical_update.dtype in (torch.float16, torch.bfloat16)
            else canonical_update.dtype
        )
        representation_probability = representation_log_conditional.exp().to(
            accumulation_dtype
        )
        cell_update = (
            representation_probability[..., None]
            * canonical_update.to(accumulation_dtype)
        ).sum(dim=2).to(cell_states)
        with torch.autocast(device_type=cell_states.device.type, enabled=False):
            refined_state = compose_antipodal_plane_frame_residual(
                cell_states.reshape(-1, FULL_FRAME_STATE_SIZE),
                cell_update.reshape(-1, FULL_FRAME_UPDATE_SIZE),
                support_origin_ap_dv_ml_um,
            ).reshape(batch, cells, FULL_FRAME_STATE_SIZE)

        # Normal-offset variance is measured in um^2 and routinely exceeds
        # float16's finite range. Keep the complete probabilistic head and its
        # mixture moments out of autocast; these tensors do not affect the
        # point-state update above.
        with torch.autocast(device_type=cell_states.device.type, enabled=False):
            raw_cholesky = self.candidate_plane_cholesky(
                pooled.to(self.candidate_plane_cholesky.weight)
            )
            canonical_cholesky = self._plane_cholesky(raw_cholesky)
            covariance_dtype = torch.promote_types(
                canonical_cholesky.dtype, cell_states.dtype
            )
            if covariance_dtype in (torch.float16, torch.bfloat16):
                covariance_dtype = torch.float32
            canonical_cholesky = canonical_cholesky.to(dtype=covariance_dtype)
            canonical_covariance = (
                canonical_cholesky @ canonical_cholesky.transpose(-1, -2)
            )
            difference = (
                canonical_update.to(dtype=covariance_dtype)[..., :3]
                - cell_update.to(dtype=covariance_dtype)[..., None, :3]
            )
            cell_covariance = (
                representation_probability.to(dtype=covariance_dtype)[..., None, None]
                * (
                    canonical_covariance
                    + difference[..., :, None] @ difference[..., None, :]
                )
            ).sum(dim=2)
        return {
            "cell_id": ids,
            "representation_log_score": representation_log_score,
            "representation_log_weight": log_weight,
            "representation_log_conditional_within_cell": representation_log_conditional,
            "cell_log_evidence": cell_log_evidence,
            "cell_log_unnormalized_mass": cell_log_unnormalized_mass,
            "initial_representation_canonical_residual": canonical_update.to(cell_states),
            "initial_cell_canonical_residual": cell_update,
            "initial_cell_refined_state": refined_state,
            "initial_representation_canonical_plane_covariance": canonical_covariance,
            "initial_cell_canonical_plane_covariance": cell_covariance,
        }

    @staticmethod
    def normalize_complete_catalogue(
        chunks: dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
        expected_cell_count: int,
        top_k: int,
    ) -> dict[str, torch.Tensor | bool | str]:
        """Verify complete cell coverage, then normalize and compute an exact tail."""
        if isinstance(chunks, dict):
            chunks = (chunks,)
        if not chunks:
            raise ValueError("at least one scored catalogue chunk is required")
        ids = torch.cat(tuple(chunk["cell_id"] for chunk in chunks), dim=0)
        masses = torch.cat(
            tuple(chunk["cell_log_unnormalized_mass"] for chunk in chunks), dim=1
        )
        if (
            not isinstance(expected_cell_count, int)
            or isinstance(expected_cell_count, bool)
            or expected_cell_count < 1
        ):
            raise ValueError("expected cell count must be a positive integer")
        expected = torch.arange(expected_cell_count, device=ids.device)
        order = torch.argsort(ids)
        if ids.numel() != expected_cell_count or not torch.equal(ids[order], expected):
            raise ValueError(
                "catalogue normalization requires every expected unique cell ID exactly once"
            )
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= expected_cell_count:
            raise ValueError("top_k must select between one and all physical cells")
        sorted_mass = masses[:, order]
        log_probability = F.log_softmax(sorted_mass, dim=1)
        top_catalogue_index = torch.argsort(
            log_probability, dim=1, descending=True, stable=True
        )[:, :top_k]
        top_source_index = order[top_catalogue_index]
        top_log_probability = torch.gather(log_probability, 1, top_catalogue_index)
        probability = log_probability.exp()
        selected = torch.zeros_like(probability, dtype=torch.bool).scatter(
            1, top_catalogue_index, True
        )
        retained = probability.masked_fill(~selected, 0.0).sum(dim=1).clamp(0.0, 1.0)
        omitted = probability.masked_fill(selected, 0.0).sum(dim=1).clamp(0.0, 1.0)
        return {
            "retrieval_cell_id": expected,
            "retrieval_cell_log_probability": log_probability,
            "retrieval_cell_probability": probability,
            "retrieval_topk_catalogue_index": top_catalogue_index,
            "retrieval_topk_source_index": top_source_index,
            "retrieval_topk_cell_id": expected[top_catalogue_index],
            "retrieval_topk_log_probability": top_log_probability,
            "retrieval_topk_retained_probability": retained,
            "retrieval_omitted_probability": omitted,
            "catalogue_complete": True,
            "probabilities_calibrated": PROBABILITIES_CALIBRATED,
            "retrieval_tail_scope": RETRIEVAL_TAIL_SCOPE,
        }

    def refine(
        self,
        source_features: torch.Tensor,
        atlas_volume: torch.Tensor,
        initial_states: torch.Tensor,
        initial_joint_log_probability: torch.Tensor,
        representation_to_canonical_raster_affine: torch.Tensor,
        output_shape_h_w: tuple[int, int],
        origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        axial_offsets_um: torch.Tensor,
        axial_weights: torch.Tensor,
        steps: int,
        *,
        deformation_decoder: nn.Module | None = None,
        pose_only_steps: int | None = None,
        dense_deformation_supervision_weight: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Marginalize raster nuisance states before each shared recurrent update.

        When a deformation decoder is supplied, its absolute affine-free state is
        decoded from the same recurrent context and fed into the next correlation
        render. The fixed iteration gate prevents pose/deformation confounding
        during the declared pose-capture prefix. Dense-censored rows use an
        exact detached identity feedback map at every iteration.
        """
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("refinement steps must be a positive integer")
        if initial_states.ndim != 3 or initial_states.shape[-1] != FULL_FRAME_STATE_SIZE:
            raise ValueError("initial states must have shape (B,K,12)")
        batch, cells = initial_states.shape[:2]
        representations = representation_to_canonical_raster_affine.shape[2]
        if initial_joint_log_probability.shape != (batch, cells, representations):
            raise ValueError("initial joint log probability must have shape (B,K,R)")
        if deformation_decoder is None:
            if pose_only_steps is not None:
                raise ValueError("pose_only_steps requires a deformation decoder")
        elif (
            not isinstance(pose_only_steps, int)
            or isinstance(pose_only_steps, bool)
            or not 0 <= pose_only_steps <= steps + 1
        ):
            raise ValueError("pose_only_steps must be between zero and T")
        if dense_deformation_supervision_weight is None:
            dense_weight = source_features.new_ones(batch)
        else:
            dense_weight = torch.as_tensor(
                dense_deformation_supervision_weight,
                device=source_features.device,
                dtype=source_features.dtype,
            )
            if dense_weight.shape != (batch,) or not bool(
                torch.isfinite(dense_weight).all()
                and (dense_weight >= 0.0).all()
                and (dense_weight <= 1.0).all()
            ):
                raise ValueError(
                    "dense deformation supervision weight must be finite in [0,1] with shape (B,)"
                )
        feedback_enabled = dense_weight > 0.0
        feedback_enabled_map = feedback_enabled[:, None, None, None, None]
        source = source_features[:, None, None].expand(
            batch, cells, representations, *source_features.shape[1:]
        ).reshape(batch * cells * representations, *source_features.shape[1:])
        hidden = source.new_zeros(
            batch * cells * representations,
            self.recurrent_cell.candidate.out_channels,
            *source.shape[-2:],
        )
        state = initial_states
        state_sequence = [state]
        updates = []
        log_scores = []
        representation_log_conditionals = []
        representation_contexts = []
        representation_plane_covariances = []
        cell_plane_covariances = []
        deformation_outputs = []
        deformation_cell_contexts = []
        deformation_representation_probabilities = []
        deformation_active = []
        deformation_feedback_maps = []
        feedback_map = None
        initial_cell_log_probability = torch.logsumexp(
            initial_joint_log_probability, dim=2
        )
        representation_log_conditional = (
            initial_joint_log_probability - initial_cell_log_probability[..., None]
        )
        cell_log_score_increment = torch.zeros_like(initial_cell_log_probability)
        for iteration in range(steps):
            if deformation_decoder is not None:
                probability = representation_log_conditional.exp().detach()
                representation_hidden = hidden.reshape(
                    batch,
                    cells,
                    representations,
                    *hidden.shape[1:],
                )
                cell_context = (
                    probability.to(representation_hidden)[..., None, None, None]
                    * representation_hidden
                ).sum(dim=2)
                flat_context = cell_context.reshape(
                    batch * cells, *cell_context.shape[2:]
                )
                active = iteration >= pose_only_steps
                decoded = (
                    deformation_decoder(flat_context, output_shape_h_w)
                    if active
                    else inactive_affine_free_deformation(
                        flat_context, output_shape_h_w
                    )
                )
                decoded_feedback_map = decoded["forward_map_yx_px"].reshape(
                    batch, cells, 2, *output_shape_h_w
                )
                identity_feedback_map = identity_pixel_map_yx(
                    batch * cells,
                    output_shape_h_w,
                    device=decoded_feedback_map.device,
                    dtype=decoded_feedback_map.dtype,
                ).reshape(batch, cells, 2, *output_shape_h_w)
                feedback_map = torch.where(
                    feedback_enabled_map,
                    decoded_feedback_map,
                    identity_feedback_map.detach(),
                )
                deformation_outputs.append(decoded)
                deformation_cell_contexts.append(cell_context)
                deformation_representation_probabilities.append(probability)
                deformation_active.append(active)
                deformation_feedback_maps.append(feedback_map)
            rendered = self._render_representations(
                atlas_volume,
                state,
                representation_to_canonical_raster_affine,
                output_shape_h_w,
                origin_ap_dv_ml_um,
                voxel_size_ap_dv_ml_um,
                axial_offsets_um,
                axial_weights,
            )
            if feedback_map is not None:
                rendered = self._warp_representations_with_cell_map(
                    rendered, feedback_map.to(rendered)
                )
            atlas_features = self._encode_atlas(
                rendered.reshape(
                    batch * cells * representations,
                    self.atlas_channels,
                    *output_shape_h_w,
                )
            )
            evidence = self.refinement_pair_encoder(
                _pair_evidence(source, atlas_features, self.correlation_radius)
            )
            hidden = self.recurrent_cell(evidence, hidden)
            representation_contexts.append(
                hidden.reshape(
                    batch,
                    cells,
                    representations,
                    *hidden.shape[1:],
                )
            )
            pooled = hidden.mean(dim=(-2, -1))
            canonical_update = self._bounded_update(
                self.recurrent_update(pooled)
            ).reshape(
                batch, cells, representations, FULL_FRAME_UPDATE_SIZE
            )
            log_score = self.recurrent_log_likelihood(pooled).reshape(
                batch, cells, representations
            )
            joint_within_cell = representation_log_conditional + log_score
            step_cell_log_score = torch.logsumexp(joint_within_cell, dim=2)
            representation_log_conditional = (
                joint_within_cell - step_cell_log_score[..., None]
            )
            cell_log_score_increment = cell_log_score_increment + step_cell_log_score
            accumulation_dtype = (
                torch.float32
                if canonical_update.dtype in (torch.float16, torch.bfloat16)
                else canonical_update.dtype
            )
            cell_update = (
                representation_log_conditional.exp().to(accumulation_dtype)[..., None]
                * canonical_update.to(accumulation_dtype)
            ).sum(dim=2).to(state)
            with torch.autocast(device_type=state.device.type, enabled=False):
                raw_cholesky = self.recurrent_plane_cholesky(
                    pooled.to(self.recurrent_plane_cholesky.weight)
                )
                canonical_cholesky = self._plane_cholesky(raw_cholesky)
                covariance_dtype = torch.promote_types(
                    canonical_cholesky.dtype, state.dtype
                )
                if covariance_dtype in (torch.float16, torch.bfloat16):
                    covariance_dtype = torch.float32
                canonical_cholesky = canonical_cholesky.to(dtype=covariance_dtype)
                canonical_covariance = (
                    canonical_cholesky @ canonical_cholesky.transpose(-1, -2)
                ).reshape(batch, cells, representations, 3, 3)
                plane_difference = (
                    canonical_update.to(dtype=covariance_dtype)[..., :3]
                    - cell_update.to(dtype=covariance_dtype)[..., None, :3]
                )
                cell_covariance = (
                    representation_log_conditional.exp().to(
                        dtype=covariance_dtype
                    )[..., None, None]
                    * (
                        canonical_covariance
                        + plane_difference[..., :, None]
                        @ plane_difference[..., None, :]
                    )
                ).sum(dim=2)
            with torch.autocast(device_type=state.device.type, enabled=False):
                state = compose_antipodal_plane_frame_residual(
                    state.reshape(-1, FULL_FRAME_STATE_SIZE),
                    cell_update.reshape(-1, FULL_FRAME_UPDATE_SIZE),
                    support_origin_ap_dv_ml_um,
                ).reshape(batch, cells, FULL_FRAME_STATE_SIZE)
            updates.append(cell_update)
            log_scores.append(log_score)
            representation_log_conditionals.append(representation_log_conditional)
            representation_plane_covariances.append(canonical_covariance)
            cell_plane_covariances.append(cell_covariance)
            state_sequence.append(state)

        if deformation_decoder is not None:
            final_probability_for_deformation = (
                representation_log_conditional.exp().detach()
            )
            final_representation_hidden_for_deformation = hidden.reshape(
                batch,
                cells,
                representations,
                *hidden.shape[1:],
            )
            final_cell_context_for_deformation = (
                final_probability_for_deformation.to(
                    final_representation_hidden_for_deformation
                )[..., None, None, None]
                * final_representation_hidden_for_deformation
            ).sum(dim=2)
            flat_final_context_for_deformation = (
                final_cell_context_for_deformation.reshape(
                    batch * cells,
                    *final_cell_context_for_deformation.shape[2:],
                )
            )
            final_active_for_deformation = steps >= pose_only_steps
            final_decoded_for_deformation = (
                deformation_decoder(
                    flat_final_context_for_deformation, output_shape_h_w
                )
                if final_active_for_deformation
                else inactive_affine_free_deformation(
                    flat_final_context_for_deformation, output_shape_h_w
                )
            )
            decoded_feedback_map = final_decoded_for_deformation[
                "forward_map_yx_px"
            ].reshape(batch, cells, 2, *output_shape_h_w)
            identity_feedback_map = identity_pixel_map_yx(
                batch * cells,
                output_shape_h_w,
                device=decoded_feedback_map.device,
                dtype=decoded_feedback_map.dtype,
            ).reshape(batch, cells, 2, *output_shape_h_w)
            feedback_map = torch.where(
                feedback_enabled_map,
                decoded_feedback_map,
                identity_feedback_map.detach(),
            )
            deformation_outputs.append(final_decoded_for_deformation)
            deformation_cell_contexts.append(
                final_cell_context_for_deformation
            )
            deformation_representation_probabilities.append(
                final_probability_for_deformation
            )
            deformation_active.append(final_active_for_deformation)
            deformation_feedback_maps.append(feedback_map)

        final_render = self._render_representations(
            atlas_volume,
            state,
            representation_to_canonical_raster_affine,
            output_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            axial_offsets_um,
            axial_weights,
        )
        final_feedback_render = (
            final_render
            if feedback_map is None
            else self._warp_representations_with_cell_map(
                final_render, feedback_map.to(final_render)
            )
        )
        final_atlas_features = self._encode_atlas(
            final_feedback_render.reshape(
                batch * cells * representations,
                self.atlas_channels,
                *output_shape_h_w,
            )
        )
        final_evidence = self.refinement_pair_encoder(
            _pair_evidence(source, final_atlas_features, self.correlation_radius)
        )
        final_hidden = self.recurrent_cell(final_evidence, hidden)
        representation_contexts.append(
            final_hidden.reshape(
                batch,
                cells,
                representations,
                *final_hidden.shape[1:],
            )
        )
        final_log_likelihood = self.recurrent_log_likelihood(
            final_hidden.mean(dim=(-2, -1))
        ).reshape(batch, cells, representations)
        final_joint_within_cell = representation_log_conditional + final_log_likelihood
        final_cell_log_score = torch.logsumexp(final_joint_within_cell, dim=2)
        final_representation_log_conditional = (
            final_joint_within_cell - final_cell_log_score[..., None]
        )
        final_pooled = final_hidden.mean(dim=(-2, -1))
        with torch.autocast(device_type=state.device.type, enabled=False):
            final_raw_cholesky = self.recurrent_plane_cholesky(
                final_pooled.to(self.recurrent_plane_cholesky.weight)
            )
            final_canonical_cholesky = self._plane_cholesky(final_raw_cholesky)
            covariance_dtype = torch.promote_types(
                final_canonical_cholesky.dtype, state.dtype
            )
            if covariance_dtype in (torch.float16, torch.bfloat16):
                covariance_dtype = torch.float32
            final_canonical_cholesky = final_canonical_cholesky.to(
                dtype=covariance_dtype
            )
            final_representation_covariance = (
                final_canonical_cholesky
                @ final_canonical_cholesky.transpose(-1, -2)
            ).reshape(batch, cells, representations, 3, 3)
            final_cell_covariance = (
                final_representation_log_conditional.exp().to(
                    dtype=covariance_dtype
                )[..., None, None]
                * final_representation_covariance
            ).sum(dim=2)
        conditional_cell_log_probability = F.log_softmax(
            initial_cell_log_probability
            + cell_log_score_increment
            + final_cell_log_score,
            dim=1,
        )
        conditional_joint_log_probability = (
            conditional_cell_log_probability[..., None]
            + final_representation_log_conditional
        )
        result = {
            "refined_cell_state_sequence": torch.stack(state_sequence, dim=2),
            "refinement_cell_update_sequence": torch.stack(updates, dim=2),
            "refinement_representation_log_score_sequence": torch.stack(log_scores, dim=3),
            "refinement_representation_log_conditional_sequence": torch.stack(
                representation_log_conditionals, dim=3
            ),
            "refinement_representation_context_sequence": torch.stack(
                representation_contexts, dim=3
            ),
            "refinement_representation_canonical_plane_covariance_sequence": torch.stack(
                representation_plane_covariances, dim=3
            ),
            "refinement_cell_canonical_plane_covariance_sequence": torch.stack(
                cell_plane_covariances, dim=2
            ),
            "final_cell_state": state,
            "final_canonical_render": final_render,
            "final_representation_log_score": final_log_likelihood,
            "final_representation_log_conditional_within_cell": final_representation_log_conditional,
            "final_representation_canonical_plane_covariance": final_representation_covariance,
            "final_cell_canonical_plane_covariance": final_cell_covariance,
            "conditional_within_topk_representation_log_probability": conditional_joint_log_probability,
            "conditional_within_topk_cell_log_probability": conditional_cell_log_probability,
            "conditional_within_topk_cell_probability": conditional_cell_log_probability.exp(),
        }
        if deformation_decoder is not None:
            result.update(
                {
                    "joint_deformation_output_sequences": {
                        f"{key}_sequence": torch.stack(
                            [
                                output[key].reshape(
                                    batch, cells, *output[key].shape[1:]
                                )
                                for output in deformation_outputs
                            ],
                            dim=2,
                        )
                        for key in AFFINE_FREE_DEFORMATION_TENSOR_KEYS
                    },
                    "joint_deformation_cell_context_sequence": torch.stack(
                        deformation_cell_contexts, dim=2
                    ),
                    "joint_deformation_representation_probability_sequence": torch.stack(
                        deformation_representation_probabilities, dim=3
                    ),
                    "joint_deformation_active_sequence": torch.tensor(
                        deformation_active,
                        device=state.device,
                        dtype=torch.bool,
                    ),
                    "joint_deformation_feedback_map_yx_px_sequence": torch.stack(
                        deformation_feedback_maps, dim=2
                    ),
                    "joint_deformation_feedback_enabled_mask": feedback_enabled,
                    "joint_final_feedback_deformed_canonical_render": final_feedback_render,
                }
            )
        return result

    def forward_proposed(
        self,
        image: torch.Tensor,
        outline: torch.Tensor,
        outline_available: torch.Tensor,
        atlas_volume: torch.Tensor,
        cell_id: torch.Tensor,
        cell_states: torch.Tensor,
        cell_log_mass: torch.Tensor,
        representation_log_weight: torch.Tensor,
        representation_to_canonical_raster_affine: torch.Tensor,
        output_shape_h_w: tuple[int, int],
        retrieval_shape_h_w: tuple[int, int],
        origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        axial_offsets_um: torch.Tensor,
        axial_weights: torch.Tensor,
        *,
        expected_catalogue_cell_count: int,
        catalogue_chunk_size: int,
        top_k: int = 4,
        refinement_steps: int = 3,
        training_truth_catalogue_index: torch.Tensor | None = None,
        deformation_decoder: nn.Module | None = None,
        pose_only_steps: int | None = None,
        dense_deformation_supervision_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | bool | str]:
        """Propose globally, exactly rerank top-M finite renders, then refine."""
        if self.coarse_proposal is None or self.proposal_count is None:
            raise RuntimeError("the scalable proposal head is not configured")
        if self._complete_catalogue_feature_cache is not None:
            raise ValueError(
                "amortized proposals use exact row-specific finite renders, not a feature cache"
            )
        if (
            not isinstance(catalogue_chunk_size, int)
            or isinstance(catalogue_chunk_size, bool)
            or catalogue_chunk_size < 1
        ):
            raise ValueError("catalogue chunk size must be a positive integer")
        if len(retrieval_shape_h_w) != 2 or min(retrieval_shape_h_w) < 4:
            raise ValueError("retrieval spatial sizes must both be at least four")
        if (
            cell_states.shape[1] != expected_catalogue_cell_count
            or not 1 <= top_k <= self.proposal_count <= expected_catalogue_cell_count
        ):
            raise ValueError("proposal/top-k counts must fit the declared catalogue")
        ids = torch.as_tensor(cell_id, device=image.device)
        expected_ids = torch.arange(expected_catalogue_cell_count, device=image.device)
        if ids.dtype == torch.bool or torch.is_floating_point(ids) or not torch.equal(
            ids.to(torch.long), expected_ids
        ):
            raise ValueError(
                "scalable proposal requires canonical contiguous local cell IDs"
            )

        retrieval_image = F.interpolate(
            image, retrieval_shape_h_w, mode="bilinear", align_corners=False
        )
        retrieval_outline = F.interpolate(
            outline, retrieval_shape_h_w, mode="bilinear", align_corners=False
        )
        coarse_source = self.encode_histology(
            retrieval_image, retrieval_outline, outline_available
        )
        proposal = self.coarse_proposal(
            coarse_source,
            cell_states,
            cell_log_mass,
            support_origin_ap_dv_ml_um,
        )
        proposal_log_probability = proposal["cell_log_probability"]
        proposal_probability = proposal["cell_probability"]
        proposal_source_index = torch.argsort(
            proposal_log_probability, dim=1, descending=True, stable=True
        )[:, : self.proposal_count]

        def gather(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            expanded = index.reshape(
                *index.shape, *([1] * (value.ndim - 2))
            ).expand(*index.shape, *value.shape[2:])
            return torch.gather(value, 1, expanded)

        proposed_states = gather(cell_states, proposal_source_index)
        proposed_log_mass = gather(cell_log_mass, proposal_source_index)
        proposed_log_weight = gather(
            representation_log_weight, proposal_source_index
        )
        proposed_affine = gather(
            representation_to_canonical_raster_affine, proposal_source_index
        )
        exact_log_evidence_chunks = []
        for start in range(0, self.proposal_count, catalogue_chunk_size):
            stop = min(start + catalogue_chunk_size, self.proposal_count)

            def score_evidence(
                source: torch.Tensor,
                volume: torch.Tensor,
                states: torch.Tensor,
                log_mass: torch.Tensor,
                log_weight: torch.Tensor,
                affine: torch.Tensor,
            ) -> torch.Tensor:
                return self.score_catalogue_chunk(
                    source,
                    volume,
                    torch.arange(states.shape[1], device=source.device),
                    states,
                    log_mass,
                    log_weight,
                    affine,
                    retrieval_shape_h_w,
                    origin_ap_dv_ml_um,
                    voxel_size_ap_dv_ml_um,
                    support_origin_ap_dv_ml_um,
                    axial_offsets_um,
                    axial_weights,
                )["cell_log_evidence"]

            arguments = (
                coarse_source,
                atlas_volume,
                proposed_states[:, start:stop],
                proposed_log_mass[:, start:stop],
                proposed_log_weight[:, start:stop],
                proposed_affine[:, start:stop],
            )
            exact_log_evidence_chunks.append(
                checkpoint(score_evidence, *arguments, use_reentrant=False)
                if self.training
                else score_evidence(*arguments)
            )
        exact_log_evidence = torch.cat(exact_log_evidence_chunks, dim=1)
        proposal_topm_log_probability = gather(
            proposal_log_probability, proposal_source_index
        )
        exact_rerank_log_probability = F.log_softmax(
            proposal_topm_log_probability + exact_log_evidence.float(), dim=1
        )
        exact_topk_position = torch.argsort(
            exact_rerank_log_probability, dim=1, descending=True, stable=True
        )[:, :top_k]
        honest_topk_index = gather(proposal_source_index, exact_topk_position)
        honest_topk_log_probability = gather(
            proposal_log_probability, honest_topk_index
        )

        def selected_mass(index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            selected = torch.zeros_like(proposal_probability, dtype=torch.bool).scatter(
                1, index, True
            )
            retained = proposal_probability.masked_fill(~selected, 0.0).sum(dim=1)
            omitted = proposal_probability.masked_fill(selected, 0.0).sum(dim=1)
            return retained.clamp(0.0, 1.0), omitted.clamp(0.0, 1.0)

        honest_retained, honest_omitted = selected_mass(honest_topk_index)
        top_indices = honest_topk_index.clone()
        teacher_forced = torch.zeros(
            image.shape[0], device=image.device, dtype=torch.bool
        )
        if training_truth_catalogue_index is not None:
            if not self.training:
                raise ValueError("truth-forced refinement is training-only")
            truth_index = torch.as_tensor(
                training_truth_catalogue_index, device=image.device, dtype=torch.long
            )
            if truth_index.shape != (image.shape[0],) or bool(
                ((truth_index < 0) | (truth_index >= expected_catalogue_cell_count)).any()
            ):
                raise ValueError("training truth catalogue indices must have shape (B,)")
            teacher_forced = ~top_indices.eq(truth_index[:, None]).any(dim=1)
            top_indices[teacher_forced, -1] = truth_index[teacher_forced]
        top_log_probability = gather(proposal_log_probability, top_indices)
        retained, omitted = selected_mass(top_indices)
        proposal_topm_retained, proposal_topm_omitted = selected_mass(
            proposal_source_index
        )

        retrieval = {
            "retrieval_cell_id": expected_ids,
            "retrieval_cell_log_probability": proposal_log_probability,
            "retrieval_cell_probability": proposal_probability,
            "retrieval_topk_catalogue_index": top_indices,
            "retrieval_topk_source_index": top_indices,
            "retrieval_topk_cell_id": top_indices,
            "retrieval_topk_log_probability": top_log_probability,
            "retrieval_topk_retained_probability": retained,
            "retrieval_omitted_probability": omitted,
            "honest_retrieval_topk_catalogue_index": honest_topk_index,
            "honest_retrieval_topk_cell_id": honest_topk_index,
            "honest_retrieval_topk_log_probability": honest_topk_log_probability,
            "honest_retrieval_topk_retained_probability": honest_retained,
            "honest_retrieval_omitted_probability": honest_omitted,
            "retrieval_teacher_forced_mask": teacher_forced,
            "proposal_topm_catalogue_index": proposal_source_index,
            "proposal_topm_cell_id": proposal_source_index,
            "proposal_topm_log_probability": proposal_topm_log_probability,
            "proposal_topm_retained_probability": proposal_topm_retained,
            "proposal_omitted_probability": proposal_topm_omitted,
            "proposal_mixture_log_probability": proposal[
                "mixture_log_probability"
            ],
            "proposal_entropy": proposal["entropy"],
            "proposal_normalized_entropy": proposal["normalized_entropy"],
            "exact_rerank_cell_log_evidence": exact_log_evidence,
            "exact_rerank_conditional_log_probability": exact_rerank_log_probability,
            "exact_rerank_topk_proposal_position": exact_topk_position,
            "catalogue_complete": True,
            "probabilities_calibrated": False,
            "retrieval_tail_scope": (
                "complete_amortized_proposal; exact_finite_likelihood_top_m_only"
            ),
            "proposal_probability_scope": "all_supplied_catalogue_cells",
            "exact_rerank_probability_scope": "conditional_within_proposed_top_m",
        }

        top_states = gather(cell_states, top_indices)
        top_log_mass = gather(cell_log_mass, top_indices)
        top_log_weight = gather(representation_log_weight, top_indices)
        top_affine = gather(representation_to_canonical_raster_affine, top_indices)
        full_source = (
            coarse_source
            if tuple(image.shape[-2:]) == tuple(output_shape_h_w)
            and tuple(retrieval_shape_h_w) == tuple(output_shape_h_w)
            else self.encode_histology(image, outline, outline_available)
        )
        top_chunk = self.score_catalogue_chunk(
            full_source,
            atlas_volume,
            torch.arange(top_k, device=image.device),
            top_states,
            top_log_mass,
            top_log_weight,
            top_affine,
            output_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            support_origin_ap_dv_ml_um,
            axial_offsets_um,
            axial_weights,
        )
        initial_states = top_chunk["initial_cell_refined_state"]
        initial_joint_log_probability = (
            top_log_probability[..., None]
            + top_chunk["representation_log_conditional_within_cell"]
        )
        refinement = self.refine(
            full_source,
            atlas_volume,
            initial_states,
            initial_joint_log_probability,
            top_affine,
            output_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            support_origin_ap_dv_ml_um,
            axial_offsets_um,
            axial_weights,
            refinement_steps,
            deformation_decoder=deformation_decoder,
            pose_only_steps=pose_only_steps,
            dense_deformation_supervision_weight=dense_deformation_supervision_weight,
        )
        return {
            "cell_id": ids,
            "cell_log_score": proposal["cell_log_score"],
            "cell_log_unnormalized_mass": proposal[
                "cell_log_unnormalized_mass"
            ],
            **retrieval,
            "topk_initial_representation_log_score": top_chunk[
                "representation_log_score"
            ],
            "topk_initial_representation_log_conditional_within_cell": top_chunk[
                "representation_log_conditional_within_cell"
            ],
            "topk_initial_cell_state": initial_states,
            "topk_initial_cell_canonical_plane_covariance": top_chunk[
                "initial_cell_canonical_plane_covariance"
            ],
            "refinement_probability_scope": "conditional_within_retrieved_topk",
            "retrieval_execution": (
                "amortized_antipodal_proposal_plus_exact_finite_top_m_rerank"
            ),
            "retrieval_shape_h_w": torch.tensor(
                retrieval_shape_h_w, device=image.device
            ),
            "catalogue_chunk_size": int(catalogue_chunk_size),
            "proposal_count": int(self.proposal_count),
            **refinement,
        }

    def forward_streamed(
        self,
        image: torch.Tensor,
        outline: torch.Tensor,
        outline_available: torch.Tensor,
        atlas_volume: torch.Tensor,
        cell_id: torch.Tensor,
        cell_states: torch.Tensor,
        cell_log_mass: torch.Tensor,
        representation_log_weight: torch.Tensor,
        representation_to_canonical_raster_affine: torch.Tensor,
        output_shape_h_w: tuple[int, int],
        retrieval_shape_h_w: tuple[int, int],
        origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        axial_offsets_um: torch.Tensor,
        axial_weights: torch.Tensor,
        *,
        expected_catalogue_cell_count: int,
        catalogue_chunk_size: int,
        top_k: int = 4,
        refinement_steps: int = 3,
        training_truth_catalogue_index: torch.Tensor | None = None,
        deformation_decoder: nn.Module | None = None,
        pose_only_steps: int | None = None,
        dense_deformation_supervision_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | bool | str]:
        """Checkpoint low-resolution catalogue chunks, then refine only top-K."""
        if self.coarse_proposal is not None:
            return self.forward_proposed(
                image,
                outline,
                outline_available,
                atlas_volume,
                cell_id,
                cell_states,
                cell_log_mass,
                representation_log_weight,
                representation_to_canonical_raster_affine,
                output_shape_h_w,
                retrieval_shape_h_w,
                origin_ap_dv_ml_um,
                voxel_size_ap_dv_ml_um,
                support_origin_ap_dv_ml_um,
                axial_offsets_um,
                axial_weights,
                expected_catalogue_cell_count=expected_catalogue_cell_count,
                catalogue_chunk_size=catalogue_chunk_size,
                top_k=top_k,
                refinement_steps=refinement_steps,
                training_truth_catalogue_index=training_truth_catalogue_index,
                deformation_decoder=deformation_decoder,
                pose_only_steps=pose_only_steps,
                dense_deformation_supervision_weight=(
                    dense_deformation_supervision_weight
                ),
            )
        if (
            not isinstance(catalogue_chunk_size, int)
            or isinstance(catalogue_chunk_size, bool)
            or catalogue_chunk_size < 1
        ):
            raise ValueError("catalogue chunk size must be a positive integer")
        if len(retrieval_shape_h_w) != 2 or min(retrieval_shape_h_w) < 4:
            raise ValueError("retrieval spatial sizes must both be at least four")
        if cell_states.shape[1] != expected_catalogue_cell_count:
            raise ValueError("streamed retrieval requires the declared complete catalogue")

        retrieval_image = F.interpolate(
            image, retrieval_shape_h_w, mode="bilinear", align_corners=False
        )
        retrieval_outline = F.interpolate(
            outline, retrieval_shape_h_w, mode="bilinear", align_corners=False
        )
        coarse_source = self.encode_histology(
            retrieval_image, retrieval_outline, outline_available
        )
        feature_cache = self._complete_catalogue_feature_cache
        if feature_cache is not None:
            if self.training:
                raise ValueError("complete-catalogue feature caches are inference-only")
            incoming_ids = torch.as_tensor(cell_id, device=image.device).to(torch.long)
            cached_features = feature_cache["atlas_features"]
            if (
                feature_cache["retrieval_shape_h_w"] != tuple(retrieval_shape_h_w)
                or cached_features.shape[0] != expected_catalogue_cell_count
                or cached_features.shape[2:] != coarse_source.shape[1:]
                or (
                    not feature_cache["already_verified"]
                    and not torch.equal(
                        feature_cache["cell_id"].to(image.device), incoming_ids
                    )
                )
            ):
                raise ValueError("active feature cache does not match this complete catalogue call")
        chunks = []
        for start in range(0, expected_catalogue_cell_count, catalogue_chunk_size):
            stop = min(start + catalogue_chunk_size, expected_catalogue_cell_count)
            ids = torch.as_tensor(cell_id, device=image.device)[start:stop]

            def score_mass(
                source: torch.Tensor,
                volume: torch.Tensor,
                states: torch.Tensor,
                log_mass: torch.Tensor,
                log_weight: torch.Tensor,
                affine: torch.Tensor,
                chunk_ids: torch.Tensor,
            ) -> torch.Tensor:
                return self.score_catalogue_chunk(
                    source,
                    volume,
                    chunk_ids,
                    states,
                    log_mass,
                    log_weight,
                    affine,
                    retrieval_shape_h_w,
                    origin_ap_dv_ml_um,
                    voxel_size_ap_dv_ml_um,
                    support_origin_ap_dv_ml_um,
                    axial_offsets_um,
                    axial_weights,
                )["cell_log_unnormalized_mass"]

            if feature_cache is None:
                arguments = (
                    coarse_source,
                    atlas_volume,
                    cell_states[:, start:stop],
                    cell_log_mass[:, start:stop],
                    representation_log_weight[:, start:stop],
                    representation_to_canonical_raster_affine[:, start:stop],
                    ids,
                )
                mass = (
                    checkpoint(score_mass, *arguments, use_reentrant=False)
                    if self.training
                    else score_mass(*arguments)
                )
            else:
                mass = self.score_catalogue_feature_chunk(
                    coarse_source,
                    cached_features[start:stop],
                    ids,
                    cell_log_mass[:, start:stop],
                    representation_log_weight[:, start:stop],
                )["cell_log_unnormalized_mass"]
            chunks.append({"cell_id": ids, "cell_log_unnormalized_mass": mass})

        retrieval = self.normalize_complete_catalogue(
            chunks, expected_catalogue_cell_count, top_k
        )
        retrieval.update(
            {
                "honest_retrieval_topk_catalogue_index": retrieval[
                    "retrieval_topk_catalogue_index"
                ],
                "honest_retrieval_topk_cell_id": retrieval[
                    "retrieval_topk_cell_id"
                ],
                "honest_retrieval_topk_log_probability": retrieval[
                    "retrieval_topk_log_probability"
                ],
                "honest_retrieval_topk_retained_probability": retrieval[
                    "retrieval_topk_retained_probability"
                ],
                "honest_retrieval_omitted_probability": retrieval[
                    "retrieval_omitted_probability"
                ],
            }
        )
        teacher_forced = torch.zeros(
            image.shape[0], device=image.device, dtype=torch.bool
        )
        if training_truth_catalogue_index is not None:
            if not self.training:
                raise ValueError("truth-forced refinement is training-only")
            truth_index = torch.as_tensor(
                training_truth_catalogue_index, device=image.device, dtype=torch.long
            )
            if truth_index.shape != (image.shape[0],) or bool(
                ((truth_index < 0) | (truth_index >= expected_catalogue_cell_count)).any()
            ):
                raise ValueError("training truth catalogue indices must have shape (B,)")
            forced_catalogue_index = retrieval[
                "retrieval_topk_catalogue_index"
            ].clone()
            teacher_forced = ~forced_catalogue_index.eq(truth_index[:, None]).any(dim=1)
            forced_catalogue_index[teacher_forced, -1] = truth_index[teacher_forced]
            source_by_catalogue = torch.argsort(
                torch.as_tensor(cell_id, device=image.device)
            )
            forced_source_index = source_by_catalogue[forced_catalogue_index]
            forced_log_probability = torch.gather(
                retrieval["retrieval_cell_log_probability"],
                1,
                forced_catalogue_index,
            )
            selected = torch.zeros_like(
                retrieval["retrieval_cell_probability"], dtype=torch.bool
            ).scatter(1, forced_catalogue_index, True)
            probability = retrieval["retrieval_cell_probability"]
            retrieval.update(
                {
                    "retrieval_topk_catalogue_index": forced_catalogue_index,
                    "retrieval_topk_source_index": forced_source_index,
                    "retrieval_topk_cell_id": retrieval["retrieval_cell_id"][
                        forced_catalogue_index
                    ],
                    "retrieval_topk_log_probability": forced_log_probability,
                    "retrieval_topk_retained_probability": probability.masked_fill(
                        ~selected, 0.0
                    ).sum(dim=1),
                    "retrieval_omitted_probability": probability.masked_fill(
                        selected, 0.0
                    ).sum(dim=1),
                }
            )
        retrieval["retrieval_teacher_forced_mask"] = teacher_forced
        top_indices = retrieval["retrieval_topk_source_index"]

        def gather_cells(value: torch.Tensor) -> torch.Tensor:
            index = top_indices.reshape(
                *top_indices.shape, *([1] * (value.ndim - 2))
            ).expand(*top_indices.shape, *value.shape[2:])
            return torch.gather(value, 1, index)

        top_states = gather_cells(cell_states)
        top_log_mass = gather_cells(cell_log_mass)
        top_log_weight = gather_cells(representation_log_weight)
        top_affine = gather_cells(representation_to_canonical_raster_affine)
        full_source = (
            coarse_source
            if tuple(image.shape[-2:]) == tuple(output_shape_h_w)
            and tuple(retrieval_shape_h_w) == tuple(output_shape_h_w)
            else self.encode_histology(image, outline, outline_available)
        )
        top_chunk = self.score_catalogue_chunk(
            full_source,
            atlas_volume,
            torch.arange(top_k, device=image.device),
            top_states,
            top_log_mass,
            top_log_weight,
            top_affine,
            output_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            support_origin_ap_dv_ml_um,
            axial_offsets_um,
            axial_weights,
        )
        initial_states = top_chunk["initial_cell_refined_state"]
        initial_joint_log_probability = (
            retrieval["retrieval_topk_log_probability"][..., None]
            + top_chunk["representation_log_conditional_within_cell"]
        )
        refinement = self.refine(
            full_source,
            atlas_volume,
            initial_states,
            initial_joint_log_probability,
            top_affine,
            output_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            support_origin_ap_dv_ml_um,
            axial_offsets_um,
            axial_weights,
            refinement_steps,
            deformation_decoder=deformation_decoder,
            pose_only_steps=pose_only_steps,
            dense_deformation_supervision_weight=dense_deformation_supervision_weight,
        )
        return {
            "cell_id": torch.as_tensor(cell_id, device=image.device),
            "cell_log_unnormalized_mass": torch.cat(
                [chunk["cell_log_unnormalized_mass"] for chunk in chunks], dim=1
            ),
            **retrieval,
            "topk_initial_representation_log_score": top_chunk[
                "representation_log_score"
            ],
            "topk_initial_representation_log_conditional_within_cell": top_chunk[
                "representation_log_conditional_within_cell"
            ],
            "topk_initial_cell_state": initial_states,
            "topk_initial_cell_canonical_plane_covariance": top_chunk[
                "initial_cell_canonical_plane_covariance"
            ],
            "refinement_probability_scope": "conditional_within_retrieved_topk",
            "retrieval_execution": (
                "cached_complete_catalogue_features_exact"
                if feature_cache is not None
                else "checkpointed_low_resolution_chunks"
            ),
            "retrieval_shape_h_w": torch.tensor(
                retrieval_shape_h_w, device=image.device
            ),
            "catalogue_chunk_size": int(catalogue_chunk_size),
            **refinement,
        }

    def forward(
        self,
        image: torch.Tensor,
        outline: torch.Tensor,
        outline_available: torch.Tensor,
        atlas_volume: torch.Tensor,
        cell_id: torch.Tensor,
        cell_states: torch.Tensor,
        cell_log_mass: torch.Tensor,
        representation_log_weight: torch.Tensor,
        representation_to_canonical_raster_affine: torch.Tensor,
        output_shape_h_w: tuple[int, int],
        origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        axial_offsets_um: torch.Tensor,
        axial_weights: torch.Tensor,
        *,
        expected_catalogue_cell_count: int,
        top_k: int = 4,
        refinement_steps: int = 3,
        deformation_decoder: nn.Module | None = None,
        pose_only_steps: int | None = None,
        dense_deformation_supervision_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | bool | str]:
        source_features = self.encode_histology(image, outline, outline_available)
        chunk = self.score_catalogue_chunk(
            source_features,
            atlas_volume,
            cell_id,
            cell_states,
            cell_log_mass,
            representation_log_weight,
            representation_to_canonical_raster_affine,
            output_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            support_origin_ap_dv_ml_um,
            axial_offsets_um,
            axial_weights,
        )
        retrieval = self.normalize_complete_catalogue(
            chunk, expected_catalogue_cell_count, top_k
        )
        top_indices = retrieval["retrieval_topk_source_index"]

        def gather_cells(value: torch.Tensor) -> torch.Tensor:
            index = top_indices.reshape(
                *top_indices.shape, *([1] * (value.ndim - 2))
            ).expand(*top_indices.shape, *value.shape[2:])
            return torch.gather(value, 1, index)

        initial_states = gather_cells(chunk["initial_cell_refined_state"])
        initial_joint_log_probability = (
            retrieval["retrieval_topk_log_probability"][..., None]
            + gather_cells(chunk["representation_log_conditional_within_cell"])
        )
        top_raster_affine = gather_cells(
            torch.as_tensor(
                representation_to_canonical_raster_affine,
                device=initial_states.device,
                dtype=atlas_volume.dtype,
            )
        )
        refinement = self.refine(
            source_features,
            atlas_volume,
            initial_states,
            initial_joint_log_probability,
            top_raster_affine,
            output_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            support_origin_ap_dv_ml_um,
            axial_offsets_um,
            axial_weights,
            refinement_steps,
            deformation_decoder=deformation_decoder,
            pose_only_steps=pose_only_steps,
            dense_deformation_supervision_weight=dense_deformation_supervision_weight,
        )
        return {
            **chunk,
            **retrieval,
            "topk_initial_cell_state": initial_states,
            "topk_initial_cell_canonical_plane_covariance": gather_cells(
                chunk["initial_cell_canonical_plane_covariance"]
            ),
            "refinement_probability_scope": "conditional_within_retrieved_topk",
            **refinement,
        }
