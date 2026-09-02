import copy
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import training.arbitrary_plane_catalogue_v3 as catalogue
from training.arbitrary_plane_support import build_annotation_support_index


def _inputs():
    shape = (28, 22, 26)
    z, y, x = np.indices(shape)
    support = (
        ((z - 13.0) / 11.0) ** 2
        + ((y - 10.0) / 8.0) ** 2
        + ((x - 12.0) / 9.5) ** 2
        <= 1.0
    )
    arguments = {
        "normal_count": 48,
        "offset_count": 9,
        "roll_count": 8,
        "raster_shape_h_w": (96, 128),
        "raster_physical_span_y_x_um": (7200.0, 9600.0),
    }
    return support, (-100.0, 30.0, 250.0), (25.0, 30.0, 20.0), arguments


def _make():
    support, origin, spacing, arguments = _inputs()
    return catalogue.make_arbitrary_plane_catalogue_v3(
        support, origin, spacing, **arguments
    )


def test_proper_frames_intersecting_offsets_and_stable_cells():
    artifact = _make()
    arrays = artifact["arrays"]
    count = artifact["counts"]["cell_count"]
    assert np.array_equal(arrays["cell_id_int64"], np.arange(count))
    states = arrays["cell_states_float64"]
    u, v = states[:, 3:6], states[:, 6:9]
    normals = arrays["cell_normal_ap_dv_ml_float64"]
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(np.cross(u, v), normals, atol=1.0e-12)
    assert np.all(arrays["cell_support_intersection_margin_um_float64"] >= -1e-10)
    assert np.all(np.exp(states[:, 9:11]) > 0.0)
    assert np.allclose(
        np.exp(states[0, 9:11]),
        artifact["support_geometry"]["raster_physical_span_y_x_um"][::-1],
    )


def test_masses_representations_and_model_ready_batch_shapes():
    artifact = _make()
    arrays, tensors = artifact["arrays"], artifact["tensors"]
    count = artifact["counts"]["cell_count"]
    assert np.isclose(np.exp(arrays["cell_log_mass_float64"]).sum(), 1.0)
    assert np.allclose(
        np.exp(arrays["representation_log_weight_float64"]).sum(axis=1), 1.0
    )
    affine = arrays["representation_to_canonical_raster_affine_float64"]
    assert np.array_equal(
        affine[0, 0], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    assert np.array_equal(
        affine[0, 1], np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    assert tensors["cell_id"].shape == (count,)
    assert tensors["cell_states"].shape == (1, count, 12)
    assert tensors["cell_log_mass"].shape == (1, count)
    assert tensors["representation_log_weight"].shape == (1, count, 2)
    assert tensors["representation_to_canonical_raster_affine"].shape == (
        1,
        count,
        2,
        2,
        3,
    )
    image = torch.arange(35.0).reshape(1, 1, 5, 7)
    theta = tensors["representation_to_canonical_raster_affine"][0, 0].float()
    grid = F.affine_grid(theta, (2, 1, 5, 7), align_corners=False)
    represented = F.grid_sample(image.expand(2, -1, -1, -1), grid, align_corners=False)
    assert torch.allclose(represented[0], image[0], atol=1e-5, rtol=0.0)
    assert torch.allclose(represented[1], image[0].flip(-1), atol=1e-5, rtol=0.0)


def test_rp2_and_joint_plane_coverage_over_arbitrary_intersections():
    artifact = _make()
    arrays = artifact["arrays"]
    normals = arrays["cell_normal_ap_dv_ml_float64"]
    offsets = arrays["cell_signed_offset_um_float64"]
    rolls = arrays["cell_roll_rad_float64"]
    rng = np.random.default_rng(721)
    max_normal = max_offset = max_roll = 0.0
    normal_table = normals[:: 9 * 8]
    offset_table = arrays["normal_offset_table_um_float64"]
    support, origin, spacing, _ = _inputs()
    support_points = np.asarray(origin) + (
        np.argwhere(support).astype(np.float64) + 0.5
    ) * np.asarray(spacing)
    support_origin = np.asarray(
        artifact["support_geometry"]["support_origin_ap_dv_ml_um"]
    )
    for _ in range(512):
        truth_normal = rng.normal(size=3)
        truth_normal /= np.linalg.norm(truth_normal)
        if truth_normal[2] < 0.0:
            truth_normal *= -1.0
        normal_index = int(np.argmax(np.abs(normal_table @ truth_normal)))
        truth_point = support_points[int(rng.integers(0, len(support_points)))]
        truth_offset = float((truth_point - support_origin) @ truth_normal)
        truth_roll = rng.uniform(0.0, 2.0 * math.pi)
        angular = np.arccos(np.clip(np.abs(normals @ truth_normal), 0.0, 1.0))
        roll_error = np.abs((rolls - truth_roll + math.pi) % (2.0 * math.pi) - math.pi)
        offset_scale = np.ptp(offset_table[normal_index]) / 8.0 + 1e-12
        score = (angular / 0.45) ** 2 + ((offsets - truth_offset) / offset_scale) ** 2 + (
            roll_error / (math.pi / 8.0)
        ) ** 2
        selected = int(np.argmin(score))
        max_normal = max(max_normal, float(angular[selected]))
        max_offset = max(max_offset, float(abs(offsets[selected] - truth_offset)))
        max_roll = max(max_roll, float(roll_error[selected]))
    assert artifact["coverage_audit"][
        "max_observed_rp2_angular_covering_radius_rad"
    ] < 0.31
    assert max_normal < 0.42
    assert max_offset < 180.0
    assert max_roll <= math.pi / 8.0 + 1e-12


def test_deterministic_receipt_replay_and_tamper_rejection():
    support, origin, spacing, arguments = _inputs()
    first = catalogue.make_arbitrary_plane_catalogue_v3(
        support, origin, spacing, **arguments
    )
    second = catalogue.replay_arbitrary_plane_catalogue_v3(
        first, support, origin, spacing, **arguments
    )
    assert catalogue.catalogue_receipt_v3(first) == catalogue.catalogue_receipt_v3(
        second
    )
    catalogue.verify_arbitrary_plane_catalogue_v3(
        first, support, origin, spacing, **arguments
    )
    changed = copy.deepcopy(first)
    changed["arrays"]["cell_states_float64"][0, 0] += 1.0
    with pytest.raises(ValueError, match="replay"):
        catalogue.verify_arbitrary_plane_catalogue_v3(
            changed, support, origin, spacing, **arguments
        )
    changed = copy.deepcopy(first)
    changed["tensors"]["cell_states"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="replay"):
        catalogue.verify_arbitrary_plane_catalogue_v3(
            changed, support, origin, spacing, **arguments
        )


def test_nonzero_support_and_explicit_counts_are_required():
    support, origin, spacing, arguments = _inputs()
    with pytest.raises(ValueError):
        catalogue.make_arbitrary_plane_catalogue_v3(
            np.zeros_like(support), origin, spacing, **arguments
        )
    with pytest.raises(ValueError):
        catalogue.make_arbitrary_plane_catalogue_v3(
            support, origin, spacing, **{**arguments, "normal_count": 0}
        )
    assert "filesystem" in _make()["provenance"]["dependency_contract"]


def test_authenticated_support_index_route_is_exact_and_voxel_scalable():
    support, origin, spacing, arguments = _inputs()
    index = build_annotation_support_index(
        support.astype(np.uint8),
        atlas_id="catalogue-fixture",
        atlas_version="v1",
        source_uri="file:///catalogue-annotation.nrrd",
        source_sha256="b" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=spacing,
        origin_um=origin,
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    indexed_arguments = {
        **arguments,
        "normal_count": 16,
        "offset_count": 5,
        "roll_count": 4,
        "support_index": index,
    }
    artifact = catalogue.make_arbitrary_plane_catalogue_v3(
        None, origin, spacing, **indexed_arguments
    )
    assert artifact["algorithm"] == catalogue.CATALOGUE_V3_SUPPORT_INDEX_ALGORITHM
    assert artifact["support_geometry"]["support_index_sha256"] == index[
        "support_index_sha256"
    ]
    assert len(artifact["support_geometry"]["normal_support_interval_receipts"]) == 16
    assert artifact["coverage_audit"]["all_cells_support_membership_certified"]
    assert np.all(
        artifact["arrays"]["cell_support_intersection_margin_um_float64"] >= 0.0
    )
    catalogue.verify_arbitrary_plane_catalogue_v3(
        artifact, None, origin, spacing, **indexed_arguments
    )


def test_authenticated_route_uses_certificate_canonical_normal_everywhere(monkeypatch):
    support, origin, spacing, arguments = _inputs()
    index = build_annotation_support_index(
        support.astype(np.uint8),
        atlas_id="catalogue-canonical-normal-fixture",
        atlas_version="v1",
        source_uri="file:///catalogue-canonical-normal-annotation.nrrd",
        source_sha256="c" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=spacing,
        origin_um=origin,
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    problematic = np.asarray(
        ((-0.80, 0.10, 0.59), (0.20, -0.91, 0.31)), dtype=np.float64
    )
    problematic /= np.linalg.norm(problematic, axis=1, keepdims=True)
    original_normals = catalogue._normals
    monkeypatch.setattr(
        catalogue,
        "_normals",
        lambda count: problematic.copy() if count == 2 else original_normals(count),
    )
    artifact = catalogue.make_arbitrary_plane_catalogue_v3(
        None,
        origin,
        spacing,
        support_index=index,
        normal_count=2,
        offset_count=5,
        roll_count=4,
        raster_shape_h_w=arguments["raster_shape_h_w"],
        raster_physical_span_y_x_um=arguments["raster_physical_span_y_x_um"],
    )
    canonical = catalogue.acquisition.support_projection_bounds(problematic, index)[
        "normal_rp2"
    ]
    observed = artifact["arrays"]["cell_normal_ap_dv_ml_float64"][:: 5 * 4]
    assert np.array_equal(observed, canonical)
    assert np.all(
        artifact["arrays"]["cell_support_intersection_margin_um_float64"] >= 0.0
    )
