"""Convert provenance-bound v3 rows and catalogue cells into joint-model tensors."""

from __future__ import annotations

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components,
    full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import (
    allen_index_to_physical_um_points,
    allen_index_to_physical_um_vectors,
    physical_ouv_to_frame,
    quicknii_to_allen_points,
    quicknii_to_allen_vectors,
)


def physical_state_from_quicknii_ouv_v3(
    quicknii_ouv,
    atlas_shape_ap_dv_ml,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
):
    quicknii = torch.as_tensor(quicknii_ouv, dtype=torch.float64).reshape(3, 3)
    origin = torch.as_tensor(origin_ap_dv_ml_um, dtype=torch.float64)
    spacing = torch.as_tensor(voxel_size_ap_dv_ml_um, dtype=torch.float64)
    physical_ouv = torch.cat(
        (
            allen_index_to_physical_um_points(
                quicknii_to_allen_points(quicknii[0], tuple(atlas_shape_ap_dv_ml)),
                origin,
                spacing,
            ),
            allen_index_to_physical_um_vectors(
                quicknii_to_allen_vectors(quicknii[1]), spacing
            ),
            allen_index_to_physical_um_vectors(
                quicknii_to_allen_vectors(quicknii[2]), spacing
            ),
        )
    )
    center, frame, basis = physical_ouv_to_frame(physical_ouv)
    return full_frame_state_from_components(center, frame, basis)


def training_row_to_tensors_v3(
    row,
    *,
    atlas_shape_ap_dv_ml,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    device=None,
):
    arrays = row["arrays"]
    channels = torch.from_numpy(
        np.asarray(arrays["model_input_channels_float32"])
    ).permute(2, 0, 1)[None]
    velocity = torch.from_numpy(
        np.asarray(arrays["truth_section_pullback_stationary_velocity_yx_px_float64"])
    ).permute(2, 0, 1)[None]
    pullback = torch.from_numpy(
        np.asarray(arrays["truth_section_pullback_map_yx_px_float64"])
    ).permute(2, 0, 1)[None]
    deformation_valid = torch.from_numpy(
        np.asarray(arrays["truth_section_deformation_valid_mask"])
    )[None, None]
    correspondence_valid = torch.from_numpy(
        np.asarray(arrays["target_valid_correspondence_mask"])
    )[None, None]
    abstention = torch.from_numpy(
        np.asarray(arrays["target_correspondence_abstention_mask"])
    )[None, None]
    correspondence_weight = torch.from_numpy(
        np.asarray(arrays["target_correspondence_weight_float32"])
    )[None, None]
    loss_weight = (
        deformation_valid & correspondence_valid & ~abstention
    ).to(correspondence_weight) * correspondence_weight
    support_contract = row.get("upstream_reference", {}).get(
        "support_supervision_contract",
        {
            "point_pose_supervision_weight": 1.0,
            "dense_deformation_supervision_weight": 1.0,
        },
    )
    pose_weight = float(support_contract["point_pose_supervision_weight"])
    dense_weight = float(
        support_contract["dense_deformation_supervision_weight"]
    )
    loss_weight = loss_weight * dense_weight
    state = physical_state_from_quicknii_ouv_v3(
        row["canonical_effective_quicknii_ouv_float64"],
        atlas_shape_ap_dv_ml,
        origin_ap_dv_ml_um,
        voxel_size_ap_dv_ml_um,
    )[None]
    tensors = {
        "image": channels[:, :1],
        "outline": channels[:, 1:2],
        "outline_available": channels[:, 2].mean(dim=(-2, -1)),
        "truth_state": state,
        "pose_supervision_weight": torch.tensor([pose_weight], dtype=torch.float32),
        "truth_stationary_velocity_yx_px": velocity,
        "truth_pullback_map_yx_px": pullback,
        "deformation_weight": loss_weight,
    }
    return {
        "tensors": {
            name: value.to(device=device, dtype=torch.float32)
            if torch.is_floating_point(value)
            else value.to(device=device)
            for name, value in tensors.items()
        },
        "provenance": {
            "training_row_id": row["training_row_id"],
            "synthetic_realization_id": row["synthetic_realization_id"],
            "animal_id": row["lineage"]["animal_id"],
            "specimen_id": row["lineage"]["specimen_id"],
            "experiment_id": row["lineage"]["experiment_id"],
            "selected_mode": row["selected_mode"],
            "reflection_state": row["reflection_state"],
            "point_pose_supervision_identifiable": bool(pose_weight),
            "dense_deformation_supervision_identifiable": bool(dense_weight),
        },
    }


def nearest_catalogue_cell_v3(truth_state, catalogue):
    truth = torch.as_tensor(truth_state, dtype=torch.float64)
    if truth.ndim == 1:
        truth = truth[None]
    states = catalogue["tensors"]["cell_states"][0].to(
        device=truth.device, dtype=torch.float64
    )
    support_origin = torch.as_tensor(
        catalogue["support_geometry"]["support_origin_ap_dv_ml_um"],
        device=truth.device,
        dtype=torch.float64,
    )
    candidate_center, candidate_frame, _ = full_frame_state_to_components(states)
    truth_center, truth_frame, _ = full_frame_state_to_components(truth)
    candidate_normal = candidate_frame[:, :, 2]
    truth_normal = truth_frame[:, :, 2]
    dot = truth_normal @ candidate_normal.T
    sign = torch.where(dot < 0.0, -1.0, 1.0)
    normal_angle = torch.acos(dot.abs().clamp(0.0, 1.0))
    truth_offset = ((truth_center - support_origin) * truth_normal).sum(dim=-1)
    candidate_offset = (
        (candidate_center - support_origin) * candidate_normal
    ).sum(dim=-1)
    offset_error = (truth_offset[:, None] * sign - candidate_offset[None]).abs()
    offset_table = torch.as_tensor(
        catalogue["arrays"]["normal_offset_table_um_float64"],
        device=truth.device,
        dtype=torch.float64,
    )
    offset_step = torch.diff(offset_table, dim=1).abs().median().clamp_min(1.0)
    normal_scale = max(
        float(
            catalogue["coverage_audit"][
                "max_observed_rp2_angular_covering_radius_rad"
            ]
        ),
        1e-3,
    )
    truth_u = truth_frame[:, :, 0]
    aligned_candidate_u = candidate_frame[:, :, 0][None].expand(truth.shape[0], -1, -1)
    aligned_candidate_u = torch.where(
        (sign < 0.0)[..., None], -aligned_candidate_u, aligned_candidate_u
    )
    roll_error = torch.acos(
        (truth_u[:, None] * aligned_candidate_u)
        .sum(dim=-1)
        .clamp(min=-1.0, max=1.0)
    )
    roll_scale = np.pi / catalogue["counts"]["roll_count"]
    cost = (
        (normal_angle / normal_scale).square()
        + (offset_error / offset_step).square()
        + (roll_error / roll_scale).square()
    )
    return cost.argmin(dim=1)


def truth_index_within_topk_v3(retrieval_topk_cell_id, truth_catalogue_cell_id):
    match = retrieval_topk_cell_id == truth_catalogue_cell_id[:, None]
    return torch.where(
        match.any(dim=1), match.to(torch.int64).argmax(dim=1), -torch.ones(match.shape[0], dtype=torch.int64, device=match.device)
    )
