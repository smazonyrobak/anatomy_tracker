import copy

import numpy as np
import pytest
import torch

from training.arbitrary_plane_geometry import render_arbitrary_plane
from training.arbitrary_plane_subject_section_v2 import (
    fit_subject_centre_plane_and_residual_v2,
    sample_coordinate_rasters_v2,
    verify_subject_centre_plane_fit_v2,
)


def test_coordinate_sampler_exactly_matches_renderer_including_outside_and_large_labels():
    volume = torch.arange(5 * 6 * 7, dtype=torch.float32).reshape(5, 6, 7)
    labels = torch.arange(5 * 6 * 7, dtype=torch.int64).reshape(5, 6, 7) + 2**40
    center = torch.tensor([1.3, 2.0, 2.5], dtype=torch.float32)
    frame = torch.eye(3, dtype=torch.float32)
    basis = torch.tensor([[8.0, 0.0], [0.0, 7.0]], dtype=torch.float32)
    image, rendered_labels = render_arbitrary_plane(
        volume, center, frame, basis, (6, 7), labels
    )
    s, t = torch.arange(7, dtype=torch.float32) / 7, torch.arange(6, dtype=torch.float32) / 6
    tt, ss = torch.meshgrid(t, s, indexing="ij")
    points = center + torch.matmul(frame[:, :2] @ basis, (torch.stack((ss, tt), -1) - 0.5).unsqueeze(-1)).squeeze(-1)
    sampled, sampled_labels = sample_coordinate_rasters_v2(volume, labels, points[None])

    assert torch.equal(sampled[0], image[0, 0])
    assert torch.equal(sampled_labels[0], rendered_labels[0, 0])
    assert (sampled_labels > 2**40).any()
    assert (sampled_labels == 0).any()


def test_coordinate_sampler_uses_ties_to_even_annotation_rounding():
    volume = torch.zeros((3, 3, 4), dtype=torch.float32)
    labels = torch.arange(36, dtype=torch.int64).reshape(3, 3, 4) + 2**40
    points = torch.tensor(
        [[[[0.5, 0.0, 0.5], [1.5, 0.0, 1.5], [2.5, 0.0, 2.5]]]],
        dtype=torch.float32,
    )
    _, sampled_labels = sample_coordinate_rasters_v2(volume, labels, points)

    expected = labels[
        torch.tensor([0, 2, 2]), torch.zeros(3, dtype=torch.long), torch.tensor([0, 2, 2])
    ]
    assert torch.equal(sampled_labels.reshape(-1), expected)


def test_planar_fit_recovers_analytic_ouv_and_has_nearly_zero_residual():
    height, width = 9, 13
    origin = np.asarray([100.0, 200.0, 300.0])
    edge_u = np.asarray([70.0, -20.0, 10.0])
    edge_v = np.asarray([5.0, 60.0, -30.0])
    s, t = np.arange(width) / width, np.arange(height) / height
    points = origin + s[None, :, None] * edge_u + t[:, None, None] * edge_v
    result = fit_subject_centre_plane_and_residual_v2(points.astype(np.float64))

    assert np.allclose(result["arrays"]["physical_ouv_ap_dv_ml_um_float64"], np.concatenate((origin, edge_u, edge_v)), atol=2e-13, rtol=0)
    assert result["diagnostics"]["residual_max_um"] < 3e-13
    verify_subject_centre_plane_fit_v2(result, points.astype(np.float64))


def test_nonlinear_residual_reconstructs_the_complete_input_raster():
    height, width = 7, 11
    s, t = np.arange(width) / width, np.arange(height) / height
    ss, tt = s[None, :], t[:, None]
    points = np.empty((height, width, 3), dtype=np.float64)
    points[..., 0] = 10 + 8 * ss + 2 * tt + 3 * ss * tt
    points[..., 1] = 20 - ss + 6 * tt + 4 * ss**2
    points[..., 2] = 30 + 2 * ss - tt + 5 * tt**2
    result = fit_subject_centre_plane_and_residual_v2(points)
    arrays = result["arrays"]

    assert np.allclose(arrays["fitted_coordinate_raster_ap_dv_ml_um_float64"] + arrays["residual_coordinate_field_ap_dv_ml_um_float64"], points, atol=4e-15, rtol=0)
    assert result["diagnostics"]["residual_rms_um"] > 0


def test_fit_uses_full_raster_and_rejects_any_mask_like_extra_axis():
    points = np.zeros((4, 5, 3), dtype=np.float64)
    points[..., 0] = np.arange(5)[None]
    points[..., 1] = np.arange(4)[:, None]
    fit_subject_centre_plane_and_residual_v2(points)
    with pytest.raises(ValueError, match="shape"):
        fit_subject_centre_plane_and_residual_v2(points[None])
    with pytest.raises(ValueError, match="degenerate"):
        fit_subject_centre_plane_and_residual_v2(np.zeros((4, 5, 3), dtype=np.float64))


def test_fit_receipts_and_combined_id_reject_array_and_metadata_tamper():
    rng = np.random.default_rng(4)
    points = rng.normal(size=(5, 8, 3)).astype(np.float64)
    result = fit_subject_centre_plane_and_residual_v2(points)
    verify_subject_centre_plane_fit_v2(result, points)
    changed = copy.deepcopy(result)
    changed["arrays"]["residual_coordinate_field_ap_dv_ml_um_float64"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="receipt or identity"):
        verify_subject_centre_plane_fit_v2(changed, points)
    changed = copy.deepcopy(result)
    changed["diagnostics"]["residual_max_um"] += 1
    with pytest.raises(ValueError, match="receipt or identity"):
        verify_subject_centre_plane_fit_v2(changed, points)

    changed = copy.deepcopy(result)
    changed["extra"] = 1
    with pytest.raises(ValueError, match="receipt or identity"):
        verify_subject_centre_plane_fit_v2(changed, points)
