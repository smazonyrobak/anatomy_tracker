"""Pure subject-section coordinate sampling and plane-fit primitives."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_acquisition_v2 import _array_receipt, _payload_sha256


SUBJECT_CENTRE_PLANE_FIT_V2_SCHEMA = "anatomy-tracker.subject-centre-plane-fit/v2"
SUBJECT_CENTRE_PLANE_FIT_V2_ALGORITHM = "quicknii-orthogonal-design-row-major-float64/v2"


def sample_coordinate_rasters_v2(
    scalar_volume: torch.Tensor,
    annotation_volume: torch.Tensor,
    allen_coordinate_rasters_float32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample S arbitrary Allen-index rasters with the frozen v2 renderer semantics."""
    scalar = torch.as_tensor(scalar_volume)
    annotation = torch.as_tensor(annotation_volume, device=scalar.device)
    points = torch.as_tensor(allen_coordinate_rasters_float32, device=scalar.device)
    if scalar.ndim != 3 or scalar.dtype != torch.float32:
        raise ValueError("scalar_volume must be one float32 AP-DV-ML volume")
    if annotation.shape != scalar.shape or annotation.dtype != torch.int64:
        raise ValueError("annotation_volume must be one aligned int64 AP-DV-ML volume")
    if points.ndim != 4 or points.shape[-1] != 3 or points.dtype != torch.float32:
        raise ValueError("coordinate rasters must have float32 shape [S,H,W,3]")

    depth, height, width = scalar.shape
    grid = torch.stack(
        (
            points[..., 2] / (width - 1) * 2 - 1,
            points[..., 1] / (height - 1) * 2 - 1,
            points[..., 0] / (depth - 1) * 2 - 1,
        ),
        -1,
    )[:, None]
    samples = F.grid_sample(
        scalar[None, None].expand(points.shape[0], -1, -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0, 0]

    indices = torch.round(points).long()
    valid = (
        (indices[..., 0] >= 0)
        & (indices[..., 0] < depth)
        & (indices[..., 1] >= 0)
        & (indices[..., 1] < height)
        & (indices[..., 2] >= 0)
        & (indices[..., 2] < width)
    )
    clipped = torch.stack(
        (
            indices[..., 0].clamp(0, depth - 1),
            indices[..., 1].clamp(0, height - 1),
            indices[..., 2].clamp(0, width - 1),
        ),
        -1,
    )
    labels = annotation[clipped[..., 0], clipped[..., 1], clipped[..., 2]]
    return samples, torch.where(valid, labels, torch.zeros_like(labels))


def _row_major_sum(values: np.ndarray) -> np.ndarray:
    return np.add.reduce(np.ascontiguousarray(values).reshape(-1, values.shape[-1]), axis=0)


def _fit_identity_payload(result: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": result["schema_version"],
        "algorithm": result["algorithm"],
        "sampling_contract": result["sampling_contract"],
        "output_shape_h_w": result["output_shape_h_w"],
        "input_coordinate_raster_receipt": result["input_coordinate_raster_receipt"],
        "array_receipts": result["array_receipts"],
        "diagnostics": result["diagnostics"],
    }


def fit_subject_centre_plane_and_residual_v2(
    centre_ccf_physical_raster_float64: np.ndarray,
) -> dict[str, object]:
    """Fit physical O+sU+tV to every immutable pixel and retain the residual field."""
    points = np.asarray(centre_ccf_physical_raster_float64)
    if points.ndim != 3 or points.shape[-1] != 3 or points.dtype != np.float64:
        raise ValueError("centre raster must have float64 shape [H,W,3]")
    height, width = points.shape[:2]
    if height < 2 or width < 2 or not np.isfinite(points).all():
        raise ValueError("centre raster must be finite with H,W >= 2")
    points = np.ascontiguousarray(points)
    s = np.arange(width, dtype=np.float64) / width
    t = np.arange(height, dtype=np.float64) / height
    ds, dt = s - s.mean(), t - t.mean()
    edge_u = _row_major_sum(points * ds[None, :, None]) / (height * np.sum(ds * ds))
    edge_v = _row_major_sum(points * dt[:, None, None]) / (width * np.sum(dt * dt))
    mean = _row_major_sum(points) / (height * width)
    origin = mean - s.mean() * edge_u - t.mean() * edge_v
    fitted = np.ascontiguousarray(
        origin[None, None] + s[None, :, None] * edge_u + t[:, None, None] * edge_v
    )
    residual = np.ascontiguousarray(points - fitted)
    physical_ouv = np.ascontiguousarray(np.concatenate((origin, edge_u, edge_v)))
    normal = np.cross(edge_u, edge_v)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= np.finfo(np.float64).eps * max(
        float(np.linalg.norm(edge_u) * np.linalg.norm(edge_v)), 1.0
    ):
        raise ValueError("fitted physical O/U/V plane is degenerate")
    normal /= normal_norm
    normal_component = residual @ normal
    squared = np.sum(residual * residual, axis=-1)
    tangential_squared = np.maximum(squared - normal_component * normal_component, 0.0)
    diagnostics = {
        "residual_rms_um": float(np.sqrt(np.mean(squared))),
        "residual_max_um": float(np.sqrt(np.max(squared))),
        "normal_residual_rms_um": float(np.sqrt(np.mean(normal_component * normal_component))),
        "normal_residual_max_abs_um": float(np.max(np.abs(normal_component))),
        "tangential_residual_rms_um": float(np.sqrt(np.mean(tangential_squared))),
        "tangential_residual_max_um": float(np.sqrt(np.max(tangential_squared))),
    }
    arrays = {
        "physical_ouv_ap_dv_ml_um_float64": physical_ouv,
        "fitted_coordinate_raster_ap_dv_ml_um_float64": fitted,
        "residual_coordinate_field_ap_dv_ml_um_float64": residual,
        "fitted_plane_unit_normal_ap_dv_ml_float64": np.ascontiguousarray(normal),
    }
    result = {
        "schema_version": SUBJECT_CENTRE_PLANE_FIT_V2_SCHEMA,
        "algorithm": SUBJECT_CENTRE_PLANE_FIT_V2_ALGORITHM,
        "sampling_contract": "QuickNII s=x/W, t=y/H; complete immutable raster; no tissue mask",
        "output_shape_h_w": [height, width],
        "input_coordinate_raster_receipt": _array_receipt(points),
        "arrays": arrays,
        "array_receipts": {name: _array_receipt(value) for name, value in arrays.items()},
        "diagnostics": diagnostics,
    }
    result["subject_centre_plane_fit_id"] = _payload_sha256(_fit_identity_payload(result))
    return result


def verify_subject_centre_plane_fit_v2(
    result: dict[str, object], centre_ccf_physical_raster_float64: np.ndarray
) -> None:
    expected_top = {
        "schema_version",
        "algorithm",
        "sampling_contract",
        "output_shape_h_w",
        "input_coordinate_raster_receipt",
        "arrays",
        "array_receipts",
        "diagnostics",
        "subject_centre_plane_fit_id",
    }
    expected_arrays = {
        "physical_ouv_ap_dv_ml_um_float64",
        "fitted_coordinate_raster_ap_dv_ml_um_float64",
        "residual_coordinate_field_ap_dv_ml_um_float64",
        "fitted_plane_unit_normal_ap_dv_ml_float64",
    }
    expected_diagnostics = {
        "residual_rms_um",
        "residual_max_um",
        "normal_residual_rms_um",
        "normal_residual_max_abs_um",
        "tangential_residual_rms_um",
        "tangential_residual_max_um",
    }
    replay = fit_subject_centre_plane_and_residual_v2(
        centre_ccf_physical_raster_float64
    )
    if (
        set(result) != expected_top
        or set(result.get("arrays", {})) != expected_arrays
        or set(result.get("array_receipts", {})) != expected_arrays
        or set(result.get("diagnostics", {})) != expected_diagnostics
        or result["schema_version"] != SUBJECT_CENTRE_PLANE_FIT_V2_SCHEMA
        or result["algorithm"] != SUBJECT_CENTRE_PLANE_FIT_V2_ALGORITHM
        or result["input_coordinate_raster_receipt"]
        != _array_receipt(centre_ccf_physical_raster_float64)
        or result["array_receipts"] != replay["array_receipts"]
        or result["diagnostics"] != replay["diagnostics"]
        or result["subject_centre_plane_fit_id"]
        != replay["subject_centre_plane_fit_id"]
        or any(
            not np.array_equal(result["arrays"][name], replay["arrays"][name])
            for name in expected_arrays
        )
    ):
        raise ValueError("subject centre plane fit receipt or identity does not match")
