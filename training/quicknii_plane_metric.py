"""Shared DeepSlice corresponding-pixel physical plane-distance metric."""

from __future__ import annotations

import numpy as np
import torch


QUICKNII_PIXEL_GRID_SHAPE = (299, 299)
QUICKNII_SHAPE_ML_AP_DV = (456.0, 528.0, 320.0)
QUICKNII_PLANE_DISTANCE_CONTRACT = {
    "version": "deepslice-corresponding-pixel-plane-distance-v1",
    "grid_shape": QUICKNII_PIXEL_GRID_SHAPE,
    "grid_samples": "pixel centers: (column + 0.5) / width, (row + 0.5) / height",
    "coordinate_frame": "QuickNII ML-AP-DV in Allen CCFv3 25-um voxels",
    "reference_mask": (
        "nearest Allen annotation voxel at each reference-plane pixel; label > 0"
    ),
    "statistic": (
        "mean Euclidean separation of corresponding predicted/reference CCF points"
    ),
    "physical_scale_um_per_voxel": 25.0,
}


def quicknii_pixel_points(ouv: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    grid_x, grid_y = np.meshgrid(x, y)
    values = np.asarray(ouv, dtype=np.float64)
    return values[:3] + grid_x[..., None] * values[3:6] + grid_y[..., None] * values[6:9]


def brain_masked_plane_distance(
    ground_truth_ouv: np.ndarray,
    predicted_ouv: np.ndarray,
    brain_mask: np.ndarray,
) -> float:
    mask = np.asarray(brain_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("The plane-distance metric needs a non-empty 2-D brain mask")
    ground_truth = quicknii_pixel_points(ground_truth_ouv, mask.shape)
    predicted = quicknii_pixel_points(predicted_ouv, mask.shape)
    return float(np.linalg.norm(predicted[mask] - ground_truth[mask], axis=1).mean())


def annotation_brain_mask(
    ground_truth_ouv: np.ndarray,
    annotation_ap_dv_ml: np.ndarray,
    shape: tuple[int, int] = QUICKNII_PIXEL_GRID_SHAPE,
) -> np.ndarray:
    quicknii = quicknii_pixel_points(ground_truth_ouv, shape)
    atlas = np.stack(
        (
            QUICKNII_SHAPE_ML_AP_DV[1] - quicknii[..., 1],
            QUICKNII_SHAPE_ML_AP_DV[2] - quicknii[..., 2],
            quicknii[..., 0],
        ),
        axis=-1,
    )
    indices = np.rint(atlas).astype(np.int64)
    valid = np.all(indices >= 0, axis=-1) & np.all(
        indices < np.asarray(annotation_ap_dv_ml.shape), axis=-1
    )
    mask = np.zeros(shape, dtype=bool)
    inside = indices[valid]
    mask[valid] = annotation_ap_dv_ml[inside[:, 0], inside[:, 1], inside[:, 2]] > 0
    return mask


def torch_quicknii_pixel_points(
    ouv: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    values = torch.as_tensor(ouv)
    if values.shape[-1:] != (9,):
        raise ValueError("QuickNII OUV must end in nine coordinates")
    height, width = shape
    x = (torch.arange(width, device=values.device, dtype=values.dtype) + 0.5) / width
    y = (torch.arange(height, device=values.device, dtype=values.dtype) + 0.5) / height
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return (
        values[..., None, None, :3]
        + grid_x[..., None] * values[..., None, None, 3:6]
        + grid_y[..., None] * values[..., None, None, 6:9]
    )


def torch_annotation_brain_mask(
    ground_truth_ouv: torch.Tensor,
    annotation_ap_dv_ml: torch.Tensor,
    shape: tuple[int, int] = QUICKNII_PIXEL_GRID_SHAPE,
) -> torch.Tensor:
    quicknii = torch_quicknii_pixel_points(ground_truth_ouv, shape)
    atlas = torch.stack(
        (
            QUICKNII_SHAPE_ML_AP_DV[1] - quicknii[..., 1],
            QUICKNII_SHAPE_ML_AP_DV[2] - quicknii[..., 2],
            quicknii[..., 0],
        ),
        dim=-1,
    )
    indices = torch.round(atlas).long()
    annotation_shape = torch.as_tensor(
        annotation_ap_dv_ml.shape, device=indices.device
    )
    valid = ((indices >= 0) & (indices < annotation_shape)).all(dim=-1)
    mask = torch.zeros_like(valid)
    inside = indices[valid]
    mask[valid] = annotation_ap_dv_ml[
        inside[:, 0], inside[:, 1], inside[:, 2]
    ] > 0
    support = mask.reshape(-1, *shape).sum(dim=(1, 2))
    if bool((support == 0).any()):
        raise ValueError("The plane-distance metric needs a non-empty 2-D brain mask")
    return mask


def torch_brain_masked_plane_distance(
    ground_truth_ouv: torch.Tensor,
    predicted_ouv: torch.Tensor,
    brain_mask: torch.Tensor,
) -> torch.Tensor:
    truth = torch.as_tensor(ground_truth_ouv)
    predicted = torch.as_tensor(
        predicted_ouv, device=truth.device, dtype=truth.dtype
    )
    mask = torch.as_tensor(brain_mask, device=truth.device, dtype=torch.bool)
    if truth.shape != predicted.shape or truth.shape[-1:] != (9,):
        raise ValueError("predicted and reference QuickNII OUV must have matching shapes")
    expected_mask_shape = (*truth.shape[:-1], *mask.shape[-2:])
    if mask.shape != expected_mask_shape:
        raise ValueError("QuickNII OUV batch dimensions must match the brain mask")
    support = mask.sum(dim=(-2, -1))
    if bool((support == 0).any()):
        raise ValueError("The plane-distance metric needs a non-empty 2-D brain mask")
    delta = torch_quicknii_pixel_points(predicted - truth, mask.shape[-2:])
    distance = torch.linalg.vector_norm(delta, dim=-1)
    return (distance * mask).sum(dim=(-2, -1)) / support
