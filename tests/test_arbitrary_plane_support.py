import copy

import numpy as np
import pytest

from training.arbitrary_plane_support import (
    INTERSECTION_ALGORITHM,
    PROJECTION_ALGORITHM,
    SUPPORT_INDEX_ALGORITHM,
    SUPPORT_INDEX_SCHEMA,
    build_annotation_support_index,
    plane_interval_membership_certificate,
    replay_annotation_support_index,
    support_projection_bounds,
    verify_annotation_support_index,
)


def _index(annotation, spacing=(10.0, 20.0, 30.0)):
    return build_annotation_support_index(
        annotation,
        atlas_id="fixture-atlas",
        atlas_version="fixture-v1",
        source_uri="file:///fixtures/annotation.npy",
        source_sha256="3" * 64,
        source_entity_type="atlas",
        voxel_size_um=spacing,
        origin_um=(-35.0, 11.0, 23.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )


def _full_point_bounds(annotation, normals, index):
    points = np.argwhere(annotation != 0).astype(np.float64)
    spacing = np.asarray(index["voxel_size_um"])
    origin = np.asarray(index["origin_um"])
    projection_origin = np.asarray(index["projection_origin_um"])
    physical = origin + (points + 0.5) * spacing - projection_origin
    half_extent = np.abs(normals) @ (spacing / 2.0)
    projected = normals @ physical.T
    return np.stack((projected.min(1) - half_extent, projected.max(1) + half_extent), axis=1)


def _ap_line_endpoints(mask):
    present = mask.any(axis=0)
    dv_ml = np.argwhere(present)
    lower = np.argmax(mask, axis=0)[present]
    upper = mask.shape[0] - 1 - np.argmax(mask[::-1], axis=0)[present]
    return np.unique(
        np.vstack((np.column_stack((lower, dv_ml)), np.column_stack((upper, dv_ml)))), axis=0
    )


def test_anisotropic_box_has_analytical_exact_bounds():
    annotation = np.zeros((7, 6, 5), dtype=np.uint16)
    annotation[1:5, 2:5, 1:4] = 7
    index = _index(annotation)
    normals = np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 2.0, -3.0]])
    result = support_projection_bounds(normals, index)
    expected = _full_point_bounds(annotation, result["normal_rp2"], index)

    assert index["schema_version"] == SUPPORT_INDEX_SCHEMA
    assert index["algorithm"] == SUPPORT_INDEX_ALGORITHM
    assert index["component_count"] == 1
    assert index["source"]["source_entity_type"] == "atlas"
    assert index["source"]["source_sha256_semantics"] == "raw source bytes"
    assert index["atlas"]["coordinate_axis_directions"] == ["posterior", "inferior", "right"]
    assert index["components"][0]["affine_rank"] == 3
    assert index["components"][0]["hull_vertex_count"] == 8
    assert result["algorithm"] == PROJECTION_ALGORITHM
    assert np.allclose(result["component_bounds_um"][:, 0], expected, atol=1e-12)
    assert np.array_equal(result["global_bounds_um"], result["component_bounds_um"][:, 0])


def test_connected_nonconvex_support_projects_to_an_intersection_interval():
    annotation = np.zeros((7, 7, 3), dtype=np.uint8)
    annotation[1:6, 1, 1] = 1
    annotation[5, 1:6, 1] = 1
    index = _index(annotation, spacing=(11.0, 17.0, 23.0))
    normal = np.asarray([1.0, 1.0, 0.25])
    projection = support_projection_bounds(normal, index)
    lower, upper = projection["global_bounds_um"]
    offsets = lower + np.linspace(0.01, 0.99, 101) * (upper - lower)
    certificate = plane_interval_membership_certificate(
        np.repeat(projection["normal_rp2"][None], len(offsets), axis=0), offsets, index
    )
    canonical = certificate["normal_rp2"]
    points = (
        np.asarray(index["origin_um"])
        + (np.argwhere(annotation != 0).astype(np.float64) + 0.5)
        * np.asarray(index["voxel_size_um"])
        - np.asarray(index["projection_origin_um"])
    )
    half_extent = np.abs(canonical) @ (np.asarray(index["voxel_size_um"]) / 2.0)
    brute_intersection = np.any(
        np.abs(np.sum(points[None] * canonical[:, None], axis=-1) - offsets[:, None])
        <= half_extent[:, None],
        axis=1,
    )

    assert index["component_count"] == 1
    assert index["projection_interval_mode"] == "single-connected-component-fast-path"
    assert certificate["algorithm"] == INTERSECTION_ALGORITHM
    assert certificate["intersects"].all()
    assert brute_intersection.all()


def test_disconnected_components_retain_a_true_projection_gap():
    annotation = np.zeros((9, 3, 3), dtype=np.uint8)
    annotation[1, 1, 1] = 1
    annotation[7, 1, 1] = 1
    index = _index(annotation, spacing=(10.0, 10.0, 10.0))
    projection = support_projection_bounds([1.0, 0.0, 0.0], index)
    intervals = projection["component_bounds_um"]
    first = float(intervals[0].mean())
    gap = float((intervals[0, 1] + intervals[1, 0]) / 2.0)
    second = float(intervals[1].mean())
    certificate = plane_interval_membership_certificate(
        np.repeat([[1.0, 0.0, 0.0]], 3, axis=0), [first, gap, second], index
    )

    assert index["component_count"] == 2
    assert index["projection_interval_mode"] == "per-connected-component-interval-union"
    assert [component["affine_rank"] for component in index["components"]] == [0, 0]
    assert intervals[0, 1] < intervals[1, 0]
    assert certificate["intersects"].tolist() == [True, False, True]
    assert certificate["component_membership"].tolist() == [[True, False], [False, False], [False, True]]
    assert projection["global_bounds_um"][0] < gap < projection["global_bounds_um"][1]


def test_antipodal_planes_have_identical_bounds_and_certificates():
    annotation = np.zeros((8, 7, 6), dtype=np.uint8)
    annotation[1:7, 1:6, 1:5] = 1
    index = _index(annotation)
    normal = np.asarray([0.2, -0.7, 0.5])
    lower, upper = support_projection_bounds(normal, index)["global_bounds_um"]
    offset = float(lower + 0.37 * (upper - lower))
    raw_offset = offset * np.linalg.norm(normal)
    projection = support_projection_bounds(np.stack((normal, -normal)), index)
    certificate = plane_interval_membership_certificate(
        np.stack((normal, -normal)), [raw_offset, -raw_offset], index
    )

    assert np.array_equal(projection["normal_rp2"][0], projection["normal_rp2"][1])
    assert np.array_equal(projection["component_bounds_um"][0], projection["component_bounds_um"][1])
    assert np.array_equal(certificate["normal_rp2"][0], certificate["normal_rp2"][1])
    assert certificate["signed_offset_um"][0] == certificate["signed_offset_um"][1]
    assert certificate["intersects"].tolist() == [True, True]


def test_seeded_hull_bounds_match_exact_ap_line_endpoint_extrema():
    ap, dv, ml = np.mgrid[:25, :21, :19]
    annotation = (
        ((ap - 12) / 10) ** 2 + ((dv - 10) / 8) ** 2 + ((ml - 9) / 7) ** 2 <= 1
    ).astype(np.uint8)
    annotation[4:11, 9:14, 8:13] = 0
    index = _index(annotation, spacing=(25.0, 17.0, 31.0))
    endpoints = _ap_line_endpoints(annotation != 0)
    generator = np.random.default_rng(62017)
    normals = generator.normal(size=(1024, 3))
    projection = support_projection_bounds(normals, index)
    canonical = projection["normal_rp2"]
    spacing = np.asarray(index["voxel_size_um"])
    physical = (
        np.asarray(index["origin_um"])
        + (endpoints.astype(np.float64) + 0.5) * spacing
        - np.asarray(index["projection_origin_um"])
    )
    half_extent = np.abs(canonical) @ (spacing / 2.0)
    projected = canonical @ physical.T
    endpoint_bounds = np.stack(
        (projected.min(1) - half_extent, projected.max(1) + half_extent), axis=1
    )

    assert index["components"][0]["line_endpoint_count"] == len(endpoints)
    assert index["components"][0]["hull_vertex_count"] < len(endpoints)
    assert np.allclose(projection["component_bounds_um"][:, 0], endpoint_bounds, atol=1e-12)


def test_support_index_hashes_reject_array_and_metadata_tampering():
    annotation = np.zeros((7, 6, 5), dtype=np.uint8)
    annotation[1:6, 1:5, 1:4] = 1
    index = _index(annotation)
    changed_endpoints = copy.deepcopy(index)
    changed_endpoints["component_line_endpoint_indices"][0][0, 0] += 1
    changed_hull = copy.deepcopy(index)
    changed_hull["component_hull_indices"][0][0, 0] += 1
    changed_metadata = copy.deepcopy(index)
    changed_metadata["origin_um"][0] += 1.0

    verify_annotation_support_index(index)
    with pytest.raises(ValueError, match="endpoint hash"):
        verify_annotation_support_index(changed_endpoints)
    with pytest.raises(ValueError, match="hull hash"):
        verify_annotation_support_index(changed_hull)
    with pytest.raises(ValueError, match="metadata"):
        verify_annotation_support_index(changed_metadata)


def test_index_build_and_replay_are_deterministic_and_provenance_bound():
    annotation = np.zeros((11, 9, 7), dtype=np.uint16)
    annotation[1:10, 1:8, 1:6] = 42
    annotation[1:4, 1:3, 1:3] = 0
    first = _index(annotation)
    repeated = _index(annotation.copy())
    replayed = replay_annotation_support_index(annotation, first)
    changed = annotation.copy()
    changed[1, 7, 5] = 0

    assert first["support_index_sha256"] == repeated["support_index_sha256"]
    assert first["support_index_sha256"] == replayed["support_index_sha256"]
    assert first["support_mask_sha256"] == repeated["support_mask_sha256"]
    assert first["source"]["source_sha256"] == "3" * 64
    assert first["source"]["annotation_array_sha256"] == repeated["source"]["annotation_array_sha256"]
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            first["component_line_endpoint_indices"], repeated["component_line_endpoint_indices"]
        )
    )
    assert all(
        np.array_equal(left, right)
        for left, right in zip(first["component_hull_indices"], repeated["component_hull_indices"])
    )
    with pytest.raises(ValueError, match="did not reproduce"):
        replay_annotation_support_index(changed, first)

    with pytest.raises(ValueError, match="raw source bytes"):
        build_annotation_support_index(
            annotation,
            atlas_id="fixture-atlas",
            atlas_version="fixture-v1",
            source_uri="file:///fixtures/annotation.npy",
            source_sha256="not-a-sha256",
            source_entity_type="atlas",
            voxel_size_um=(10.0, 20.0, 30.0),
            origin_um=(-35.0, 11.0, 23.0),
            coordinate_axis_directions=("posterior", "inferior", "right"),
        )


def test_projection_and_certificate_are_chunk_size_invariant():
    ap, dv, ml = np.mgrid[:23, :19, :17]
    annotation = (
        ((ap - 11) / 9) ** 2 + ((dv - 9) / 7) ** 2 + ((ml - 8) / 6) ** 2 <= 1
    ).astype(np.uint8)
    index = _index(annotation)
    generator = np.random.default_rng(81023)
    normals = generator.normal(size=(2053, 3))
    default = support_projection_bounds(normals, index)
    one_at_a_time = support_projection_bounds(normals, index, normal_batch_size=1)
    irregular = support_projection_bounds(normals, index, normal_batch_size=137)
    offsets = default["global_bounds_um"].mean(axis=1)
    certificate_default = plane_interval_membership_certificate(default["normal_rp2"], offsets, index)
    certificate_irregular = plane_interval_membership_certificate(
        default["normal_rp2"], offsets, index, normal_batch_size=113
    )

    assert index["projection_query"]["normal_batch_size"] == 1024
    assert default["projection_sha256"] == one_at_a_time["projection_sha256"]
    assert default["projection_sha256"] == irregular["projection_sha256"]
    assert np.array_equal(default["component_bounds_um"], one_at_a_time["component_bounds_um"])
    assert np.array_equal(default["component_bounds_um"], irregular["component_bounds_um"])
    assert certificate_default["certificate_sha256"] == certificate_irregular["certificate_sha256"]
    assert np.array_equal(certificate_default["intersects"], certificate_irregular["intersects"])
