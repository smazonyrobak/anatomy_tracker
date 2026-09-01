import copy
import inspect

import numpy as np
import pytest
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates

import training.arbitrary_plane_image_candidate_scalar as scalar_adapter
from training.arbitrary_plane_image_information import (
    common_lattice_map_yx,
    constant_within_support_null,
    dewarp_target_for_scoring,
    four_corner_safe_mask,
    mind_descriptor,
    mind_parameters,
    rank_candidate_scores,
    resample_common_lattice_intensity,
    resample_common_lattice_support,
    scale_candidate_raster,
    score_mind_candidates,
    score_support_penalized_mind_candidates,
    support_penalized_score,
    target_score_masks,
)


def _texture():
    y, x = np.indices((65, 67))
    return np.ascontiguousarray(((7 * y + 11 * x) % 23) / 22.0, dtype=np.float64)


def test_target_dewarp_and_candidate_scaling_preserve_frozen_dtype_order():
    image = np.arange(12, dtype=np.float32).reshape(3, 4) / 12
    y, x = np.indices(image.shape, dtype=np.float32)
    dewarped = dewarp_target_for_scoring(image, np.stack((x, y)))
    assert dewarped.dtype == np.float64
    assert np.array_equal(dewarped, image.astype(np.float64))
    raw = np.array([[0.0, 6.0, 146.5, 287.0, 300.0]], dtype=np.float32)
    scaled = scale_candidate_raster(raw)
    assert scaled.dtype == np.float64
    assert np.array_equal(scaled, [[0.0, 0.0, 0.5, 1.0, 1.0]])


def test_four_corner_safety_rejects_false_and_formally_zero_weight_corners():
    source = np.ones((4, 4), dtype=bool)
    source[0, 1] = False
    pixel_map = np.array([[[0.0, 1.25, 3.0]], [[0.0, 1.25, 1.0]]], dtype=np.float32)
    safe = four_corner_safe_mask(source, np.ones((1, 3), bool), pixel_map)
    assert safe.tolist() == [[False, True, False]]


def test_mind_parameters_masks_identity_and_chunking_are_exact():
    parameters = mind_parameters(40.0)
    assert parameters["axial_step_px"] == 3
    assert parameters["diagonal_step_px"] == 2
    assert parameters["gaussian_radius_px"] == 4
    assert parameters["offsets_dy_dx"] == (
        (-3, 0),
        (3, 0),
        (0, -3),
        (0, 3),
        (-2, -2),
        (-2, 2),
        (2, -2),
        (2, 2),
    )
    assert parameters["kernel"].dtype == np.float64
    assert np.isclose(parameters["kernel"].sum(), 1.0, rtol=0.0, atol=3e-16)

    image = _texture()
    domain = np.zeros(image.shape, bool)
    domain[10:55, 10:57] = True
    descriptor, vbar = mind_descriptor(image, domain, 40.0)
    assert descriptor.shape == (8, *image.shape)
    assert np.all(np.max(descriptor, axis=0) == 1.0)
    assert vbar > 0.0
    candidates = np.stack((image, np.roll(image, 1, 1), np.roll(image, 2, 0)))
    first = score_mind_candidates(image, candidates, domain, 40.0, chunk_size=1)
    second = score_mind_candidates(image, candidates, domain, 40.0, chunk_size=3)
    assert first["scores"][0] == 1.0
    assert np.array_equal(first["scores"], second["scores"])
    assert np.array_equal(first["candidate_vbar"], second["candidate_vbar"])
    assert "support" not in inspect.signature(score_mind_candidates).parameters


def test_target_masks_match_literal_erosion_and_context_construction():
    shape = (45, 51)
    y, x = np.indices(shape, dtype=np.float32)
    pixel_map = np.stack((x, y))
    source_map = np.ones(shape, bool)
    fixed_map = np.ones(shape, bool)
    source_valid = np.zeros(shape, bool)
    source_valid[5:40, 6:45] = True
    fixed_valid = np.ones(shape, bool)
    masks = target_score_masks(
        pixel_map, source_map, fixed_map, source_valid, fixed_valid, 100.0
    )
    footprint = mind_parameters(100.0)["footprint"]
    assert np.array_equal(
        masks["core"],
        binary_erosion(
            masks["visible"],
            structure=footprint,
            iterations=1,
            border_value=0,
            origin=0,
            brute_force=False,
        ),
    )
    safe_eroded = binary_erosion(
        masks["map_safe"],
        structure=footprint,
        iterations=1,
        border_value=0,
        origin=0,
        brute_force=False,
    )
    expected_context = (
        distance_transform_edt(~masks["visible"], sampling=(100.0, 100.0)) <= 1000.0
    ) & safe_eroded
    assert np.array_equal(masks["context"], expected_context)
    assert np.array_equal(masks["boundary_ring"], masks["context"] & ~masks["visible"])


def test_support_null_preserves_exterior_and_penalty_is_separate():
    image = np.array([[0.1, 0.2, 0.9], [0.8, 0.4, 0.7]], dtype=np.float64)
    support = np.array([[True, True, False], [False, True, False]])
    flattened, mean = constant_within_support_null(image, support)
    assert mean == np.mean(image[support])
    assert np.all(flattened[support] == mean)
    assert np.array_equal(flattened[~support], image[~support])
    loss = np.full(image.shape, 0.25)
    domain = np.ones(image.shape, bool)
    visible = np.array([[True, True, False], [True, False, False]])
    score, fraction = support_penalized_score(loss, domain, visible, support)
    assert fraction == 1 / 3
    assert score == 1.0 - (1.0 + 5 * 0.25) / 6

    target = _texture()
    candidates = np.stack((target, np.roll(target, 1, 1)))
    domain = np.zeros(target.shape, bool)
    domain[10:55, 10:57] = True
    visible = domain.copy()
    supports = np.ones(candidates.shape, bool)
    supports[1, 25:30, 25:30] = False
    first = score_support_penalized_mind_candidates(
        target, candidates, domain, visible, supports, 40.0, chunk_size=1
    )
    second = score_support_penalized_mind_candidates(
        target, candidates, domain, visible, supports, 40.0, chunk_size=2
    )
    assert np.array_equal(first["scores"], second["scores"])
    assert np.array_equal(
        first["candidate_exterior_fractions"], second["candidate_exterior_fractions"]
    )


def test_common_lattice_uses_yx_float64_and_ties_to_even_support():
    image = np.arange(24, dtype=np.float64).reshape(4, 6)
    coordinates = common_lattice_map_yx(image.shape, 1.0, 2.0)
    assert coordinates.dtype == np.float64 and coordinates.flags.c_contiguous
    y, x = np.indices(image.shape, dtype=np.float64)
    assert np.array_equal(coordinates[0], image.shape[0] / 2 + (y - image.shape[0] / 2) * 2)
    assert np.array_equal(coordinates[1], image.shape[1] / 2 + (x - image.shape[1] / 2) * 2)
    expected = map_coordinates(
        image,
        np.stack((coordinates[0], coordinates[1]), axis=0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    assert np.array_equal(resample_common_lattice_intensity(image, coordinates), expected)
    support = np.array([[True, True, False, True]])
    custom = np.array(
        [[[0.5, 0.5, 0.5, 0.5]], [[0.5, 1.5, 2.5, 4.5]]], dtype=np.float64
    )
    result = resample_common_lattice_support(support, custom)
    assert result.tolist() == [[True, False, False, False]]

    native = np.array(
        [[0.0, 0.2, 0.8, 1.0], [0.1, 0.4, 0.9, 0.7]], dtype=np.float64
    )
    native_support = np.array([[True, True, False, False], [True, False, False, False]])
    native_map = common_lattice_map_yx(native.shape, 1.0, 0.75)
    flattened_native = constant_within_support_null(native, native_support)[0]
    correct = resample_common_lattice_intensity(flattened_native, native_map)
    resampled = resample_common_lattice_intensity(native, native_map)
    resampled_support = resample_common_lattice_support(native_support, native_map)
    wrong = constant_within_support_null(resampled, resampled_support)[0]
    assert not np.array_equal(correct, wrong)


def test_ranking_is_conservative_and_candidate_ids_are_canonical():
    scores = np.array([0.8, 0.8 - 0.5e-12, 0.8 - 2.0e-12, 0.1])
    result = rank_candidate_scores(scores, ["z", "a", "b", "c"], "z")
    assert result["tied_maximum_candidate_ids"] == ["a", "z"]
    assert not result["top1"]
    assert result["true_rank"] == 2
    assert result["selected_candidate_id"] is None


def test_candidate_scalar_adapter_rejects_annotation_mismatch(monkeypatch):
    annotation = np.array([[1, 0], [2, 3]], dtype=np.int64)
    support = annotation != 0
    receipts = {"annotation": {"x": 1}, "brain_mask": {"x": 2}, "scalar": {"x": 3}}
    rendered = {
        "scalar": np.ones((2, 2), dtype=np.float32),
        "annotation": annotation,
        "brain_mask": support,
        "array_receipts": receipts,
    }
    monkeypatch.setattr(scalar_adapter, "_validate_prepared_context", lambda context: None)
    monkeypatch.setattr(
        scalar_adapter, "_render_finite_arbitrary_plane_trusted", lambda *args: rendered
    )
    context = {"scalar_tensor": object(), "annotation_tensor": object()}
    parent = {"geometry": {"id": "truth"}}
    candidate = {
        "geometry_storage": "candidate",
        "geometry": {"id": "decoy"},
        "rendered_annotation": annotation.copy(),
        "brain_mask": support.copy(),
        "render_array_receipts": receipts,
    }
    assert scalar_adapter.render_candidate_scalar(context, candidate, parent) is rendered
    candidate["rendered_annotation"][0, 0] = 9
    try:
        scalar_adapter.render_candidate_scalar(context, candidate, parent)
    except ValueError as error:
        assert "annotation/support" in str(error)
    else:
        raise AssertionError("annotation mismatch was accepted")


def test_candidate_scalar_bank_binds_exact_order_count_and_parent(monkeypatch):
    annotation = np.array([[1, 0], [2, 3]], dtype=np.int64)
    support = annotation != 0
    receipts = {"annotation": {"x": 1}, "brain_mask": {"x": 2}, "scalar": {"x": 3}}
    rendered = {
        "scalar": np.ones((2, 2), dtype=np.float32),
        "annotation": annotation,
        "brain_mask": support,
        "array_receipts": receipts,
    }
    monkeypatch.setattr(scalar_adapter, "_validate_prepared_context", lambda context: None)
    monkeypatch.setattr(
        scalar_adapter, "_render_finite_arbitrary_plane_trusted", lambda *args: rendered
    )
    candidate = {
        "geometry_storage": "candidate",
        "geometry": {"id": "decoy"},
        "rendered_annotation": annotation,
        "brain_mask": support,
        "render_array_receipts": receipts,
    }
    candidates = []
    for index in range(40):
        item = copy.deepcopy(candidate)
        item["candidate_id"] = f"c{index:02d}"
        candidates.append(item)
    parent = {"geometry": {"id": "truth"}}
    bank = {
        "candidates": candidates,
        "ordered_candidate_ids": [item["candidate_id"] for item in candidates],
        "truth_parent_geometry": parent["geometry"],
    }
    context = {"scalar_tensor": object(), "annotation_tensor": object()}
    result = scalar_adapter.render_candidate_bank_scalars(context, bank, parent)
    assert result["candidate_ids"] == bank["ordered_candidate_ids"]
    tampered = copy.deepcopy(bank)
    tampered["candidates"][0], tampered["candidates"][1] = (
        tampered["candidates"][1],
        tampered["candidates"][0],
    )
    with pytest.raises(ValueError, match="order/count"):
        scalar_adapter.render_candidate_bank_scalars(context, tampered, parent)
    wrong_parent = {"geometry": {"id": "other"}}
    with pytest.raises(ValueError, match="truth geometry"):
        scalar_adapter.render_candidate_bank_scalars(context, bank, wrong_parent)
