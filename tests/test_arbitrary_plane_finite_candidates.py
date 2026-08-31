import copy
import hashlib
import inspect
import math
from types import MappingProxyType

import numpy as np
import pytest

import training.arbitrary_plane_finite_candidates as finite_candidates
from training.arbitrary_plane_finite_candidates import (
    DEFAULT_CANDIDATE_ROOT_SEED,
    DEFAULT_SHUFFLE_ROOT_SEED,
    EFFECTIVE_PLANE_TOLERANCE_UM,
    align_rp2_pose_to_reference,
    derive_finite_candidate_seed,
    make_arbitrary_plane_finite_candidate_bank,
    make_arbitrary_plane_finite_candidate_bank_from_context,
    minimal_normal_rotation,
    prepare_arbitrary_plane_finite_candidate_context,
    render_finite_candidate_annotation,
    replay_arbitrary_plane_finite_candidate_bank,
    replay_arbitrary_plane_finite_candidate_bank_from_context,
    transport_finite_candidate_pose,
    verify_arbitrary_plane_finite_candidate_bank,
    verify_arbitrary_plane_finite_candidate_bank_from_context,
)
from training.arbitrary_plane_rendered_generator import (
    component_interval_union,
    effective_renderer_sampling_arrays,
    finite_plane_raster_geometry,
    make_finite_arbitrary_plane_render,
)
from training.arbitrary_plane_support import build_annotation_support_index


def _atlas():
    shape = (33, 31, 35)
    annotation = np.zeros(shape, dtype=np.uint16)
    annotation[2:-2, 2:-2, 2:-2] = 7
    annotation[7:17, 5:15, 4:18] = 19
    annotation[17:29, 15:27, 18:31] = 41
    ap, dv, ml = np.indices(shape)
    template = (100 + 3 * ap + 5 * dv + 7 * ml).astype(np.uint16)
    support = build_annotation_support_index(
        annotation,
        atlas_id="candidate-fixture-ccf",
        atlas_version="fixture-v1",
        source_uri="file:///fixture/annotation.nrrd",
        source_sha256="3" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(100.0, 100.0, 100.0),
        origin_um=(-1700.0, -1500.0, -1800.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    return template, annotation, support


def _parent(seed=1, output_shape=(48, 64), generator_source_commit=None):
    template, annotation, support = _atlas()
    parent = make_finite_arbitrary_plane_render(
        template,
        annotation,
        support,
        "development",
        seed,
        output_shape,
        sample_index=3,
        margin_um=(250.0, 250.0),
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="4" * 64,
        template_decoder="fixture",
        annotation_decoder="fixture",
        minimum_brain_pixels=64,
        generator_source_commit=generator_source_commit,
    )
    return parent, template, annotation, support


def _bank():
    parent, template, annotation, support = _parent()
    bank = make_arbitrary_plane_finite_candidate_bank(parent, annotation, support)
    return bank, parent, template, annotation, support


def _fixed_geometry(normal, support, output_shape=(37, 49), roll=0.37):
    interval = component_interval_union(normal, support)[0]
    return finite_plane_raster_geometry(
        normal,
        float(interval.mean()),
        roll,
        support,
        output_shape,
        (250.0, 250.0),
    )


def test_rp2_antipodes_produce_identical_sign_aligned_transport():
    _, annotation, support = _atlas()
    parent = _fixed_geometry([1.0, -2.0, 3.0], support)
    n = np.asarray([0.6, 0.1, -0.2])
    d = float(parent["signed_offset_um"] + 83.0)
    first = transport_finite_candidate_pose(parent, support, n, d, 0.19)
    second = transport_finite_candidate_pose(parent, support, -n, -d, 0.19)
    aligned_first = align_rp2_pose_to_reference(n, d, parent["normal_rp2_ap_dv_ml"])
    aligned_second = align_rp2_pose_to_reference(-n, -d, parent["normal_rp2_ap_dv_ml"])

    assert np.array_equal(aligned_first[0], aligned_second[0])
    assert aligned_first[1] == aligned_second[1]
    assert first == second
    assert render_finite_candidate_annotation(annotation, first)[0].dtype == np.int64


def test_exactly_orthogonal_rp2_antipodes_use_the_same_deterministic_tie():
    reference = np.asarray([1.0, 0.0, 0.0])
    first = align_rp2_pose_to_reference([0.0, 2.0, 0.0], 14.0, reference)
    second = align_rp2_pose_to_reference([0.0, -2.0, 0.0], -14.0, reference)

    assert np.array_equal(first[0], [0.0, 1.0, 0.0])
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1] == 7.0


def test_transport_uses_predeclared_center_formula_and_design_plane_equation():
    _, _, support = _atlas()
    parent = _fixed_geometry([0.3, -0.4, 0.8], support)
    frame0 = np.asarray(parent["frame_ap_dv_ml_physical"])
    ouv0 = np.asarray(parent["physical_ouv_ap_dv_ml_um"])
    c0 = ouv0[:3] + 0.5 * (ouv0[3:6] + ouv0[6:9])
    n0 = np.asarray(parent["normal_rp2_ap_dv_ml"])
    d0 = float(parent["signed_offset_um"])
    theta = math.radians(7.0)
    nc = math.cos(theta) * n0 + math.sin(theta) * frame0[:, 0]
    dc = d0 + 250.0
    candidate = transport_finite_candidate_pose(parent, support, nc, dc, -0.23)
    q = np.asarray(support["projection_origin_um"])
    expected = q + dc * nc + minimal_normal_rotation(n0, nc) @ (c0 - q - d0 * n0)

    assert np.allclose(candidate["center_ap_dv_ml_um"], expected, atol=1e-11)
    assert abs(np.dot(nc, np.asarray(candidate["center_ap_dv_ml_um"]) - q) - dc) <= 1e-9
    assert candidate["design_plane_equation"]["absolute_tolerance_um"] == 1e-9
    assert abs(candidate["effective_plane_equation"]["residual_um"]) <= candidate[
        "effective_plane_equation"
    ]["absolute_tolerance_um"]
    assert candidate["effective_plane_equation"]["absolute_tolerance_um"] == 0.01
    assert EFFECTIVE_PLANE_TOLERANCE_UM == 0.01


def test_explicit_roll_is_about_transported_normal_and_preserves_canvas_basis():
    _, _, support = _atlas()
    parent = _fixed_geometry([0.0, 0.0, 1.0], support)
    normal = np.asarray(parent["normal_rp2_ap_dv_ml"])
    offset = float(parent["signed_offset_um"])
    candidate = transport_finite_candidate_pose(parent, support, normal, offset, math.pi / 2)
    parent_frame = np.asarray(parent["frame_ap_dv_ml_physical"])
    frame = np.asarray(candidate["frame_ap_dv_ml_physical"])
    parent_ouv = np.asarray(parent["physical_ouv_ap_dv_ml_um"])
    candidate_ouv = np.asarray(candidate["physical_ouv_ap_dv_ml_um"])

    assert np.allclose(frame[:, 0], parent_frame[:, 1], atol=1e-12)
    assert np.allclose(frame[:, 1], -parent_frame[:, 0], atol=1e-12)
    assert np.allclose(frame[:, 2], parent_frame[:, 2], atol=1e-12)
    assert np.allclose(np.linalg.norm(candidate_ouv[3:6]), np.linalg.norm(parent_ouv[3:6]))
    assert np.allclose(np.linalg.norm(candidate_ouv[6:9]), np.linalg.norm(parent_ouv[6:9]))
    assert candidate["inplane_basis_u_v_um"] == pytest.approx(
        np.asarray(parent_frame[:, :2]).T
        @ np.stack((parent_ouv[3:6], parent_ouv[6:9]), axis=-1)
    )
    assert candidate["reflection_state"] == parent["reflection_state"]


def test_bank_has_exact_composition_and_reuses_truth_geometry_and_raster_bytes():
    bank, parent, _, annotation, support = _bank()
    replayed = replay_arbitrary_plane_finite_candidate_bank(bank, parent, annotation, support)
    truth = next(candidate for candidate in bank["candidates"] if candidate["candidate_class"] == "truth")
    counts = {
        name: sum(candidate["candidate_class"] == name for candidate in bank["candidates"])
        for name in (
            "truth",
            "offset_only",
            "normal_angle_only",
            "roll_only",
            "coupled_local",
            "global_hard_negative",
        )
    }

    assert counts == {
        "truth": 1,
        "offset_only": 6,
        "normal_angle_only": 16,
        "roll_only": 6,
        "coupled_local": 5,
        "global_hard_negative": 6,
    }
    assert bank["truth_parent_geometry"] == parent["geometry"]
    for key in parent["raster"]:
        assert bank["truth_parent_raster"][key].dtype == parent["raster"][key].dtype
        assert bank["truth_parent_raster"][key].tobytes() == parent["raster"][key].tobytes()
    assert truth["rendered_annotation"].tobytes() == parent["raster"]["annotation"].tobytes()
    assert truth["brain_mask"].tobytes() == parent["raster"]["brain_mask"].tobytes()
    assert truth["truth_parent_binding"]["finite_plane_render_id"] == parent["finite_plane_render_id"]
    assert bank["ordered_candidate_ids"] == replayed["ordered_candidate_ids"]
    assert len(set(bank["ordered_candidate_ids"])) == 40
    assert len({candidate["geometry_uniqueness_sha256"] for candidate in bank["candidates"]}) == 40
    assert bank["generator"]["learned_checkpoint_dependencies"] == []
    assert bank["generator"]["previous_model_dependencies"] == []
    assert bank["generator"]["pretrained_feature_dependencies"] == []


def test_candidate_grid_and_label_render_use_x_over_w_y_over_h_not_inclusive_endpoints():
    _, annotation, support = _atlas()
    parent = _fixed_geometry([0.0, 1.0, 0.0], support, output_shape=(19, 23))
    candidate = transport_finite_candidate_pose(
        parent,
        support,
        parent["normal_rp2_ap_dv_ml"],
        parent["signed_offset_um"] + 100.0,
        0.17,
    )
    arrays = effective_renderer_sampling_arrays(candidate, annotation.shape)
    points = arrays["coordinate_raster_allen_index_float32"]
    ouv = arrays["allen_index_ouv_ap_dv_ml_float32"]
    height, width = candidate["output_shape_h_w"]
    labels, _ = render_finite_candidate_annotation(annotation, candidate)
    rounded = np.rint(points).astype(np.int64)
    valid = np.ones((height, width), dtype=bool)
    for axis, size in enumerate(annotation.shape):
        valid &= (rounded[..., axis] >= 0) & (rounded[..., axis] < size)
    clipped = np.stack(
        [np.clip(rounded[..., axis], 0, size - 1) for axis, size in enumerate(annotation.shape)],
        axis=-1,
    )
    expected_labels = np.where(
        valid, annotation[clipped[..., 0], clipped[..., 1], clipped[..., 2]], 0
    )

    assert np.array_equal(points[0, 0], ouv[:3])
    assert np.allclose(
        points[-1, -1],
        ouv[:3] + (width - 1) / width * ouv[3:6] + (height - 1) / height * ouv[6:9],
        atol=2e-5,
    )
    assert not np.allclose(points[-1, -1], ouv[:3] + ouv[3:6] + ouv[6:9])
    assert np.array_equal(labels, expected_labels)
    assert candidate["sampling_contract"] == "quicknii-raster-index-x-over-W-y-over-H-v1"


def test_candidate_acceptance_has_no_target_or_target_overlap_dependency():
    bank, _, _, _, _ = _bank()
    signature = inspect.signature(make_arbitrary_plane_finite_candidate_bank)
    zero_target = np.zeros(bank["truth_parent_raster"]["brain_mask"].shape, dtype=bool)

    assert "target" not in signature.parameters
    assert "target_mask" not in signature.parameters
    assert bank["acceptance_contract"]["candidate_target_overlap_used"] is False
    assert bank["acceptance_contract"]["target_or_target_mask_argument"] is None
    assert all(candidate["brain_pixel_count"] >= 64 for candidate in bank["candidates"])
    assert all(np.logical_and(candidate["brain_mask"], zero_target).sum() == 0 for candidate in bank["candidates"])


def test_prepared_annotation_is_hashed_once_across_generation_verification_and_replay(monkeypatch):
    parent, _, annotation, support = _parent()
    full_shape_hashes = 0
    original = finite_candidates._array_sha256

    def counted(array):
        nonlocal full_shape_hashes
        if np.asarray(array).shape == annotation.shape:
            full_shape_hashes += 1
        return original(array)

    monkeypatch.setattr(finite_candidates, "_array_sha256", counted)
    context = prepare_arbitrary_plane_finite_candidate_context(annotation, support)
    first = make_arbitrary_plane_finite_candidate_bank_from_context(parent, context, support)
    second = make_arbitrary_plane_finite_candidate_bank_from_context(parent, context, support)
    verify_arbitrary_plane_finite_candidate_bank_from_context(first, parent, context, support)
    replayed = replay_arbitrary_plane_finite_candidate_bank_from_context(first, parent, context, support)

    assert full_shape_hashes == 1
    assert context["annotation"].flags.writeable is False
    assert first["finite_candidate_bank_id"] == second["finite_candidate_bank_id"]
    assert first["finite_candidate_bank_id"] == replayed["finite_candidate_bank_id"]
    assert first["generator"]["resolved_config"]["prepared_annotation_context_sha256"] == context[
        "prepared_context_sha256"
    ]


def test_prepared_candidate_entrypoints_bind_finite_parent_source_commit():
    source_commit, wrong_commit = "a" * 40, "b" * 40
    parent, _, annotation, support = _parent(generator_source_commit=source_commit)
    context = prepare_arbitrary_plane_finite_candidate_context(annotation, support)
    with pytest.raises(ValueError, match="source commit does not match"):
        make_arbitrary_plane_finite_candidate_bank_from_context(
            parent,
            context,
            support,
            finite_parent_generator_source_commit=wrong_commit,
        )
    bank = make_arbitrary_plane_finite_candidate_bank_from_context(
        parent,
        context,
        support,
        finite_parent_generator_source_commit=source_commit,
    )
    verify_arbitrary_plane_finite_candidate_bank_from_context(
        bank,
        parent,
        context,
        support,
        finite_parent_generator_source_commit=source_commit,
    )
    replayed = replay_arbitrary_plane_finite_candidate_bank_from_context(
        bank,
        parent,
        context,
        support,
        finite_parent_generator_source_commit=source_commit,
    )
    assert replayed["finite_candidate_bank_id"] == bank["finite_candidate_bank_id"]
    for operation in (
        verify_arbitrary_plane_finite_candidate_bank_from_context,
        replay_arbitrary_plane_finite_candidate_bank_from_context,
    ):
        with pytest.raises(ValueError, match="source commit does not match"):
            operation(
                bank,
                parent,
                context,
                support,
                finite_parent_generator_source_commit=wrong_commit,
            )


def test_noncontext_candidate_entrypoints_bind_finite_parent_source_commit():
    source_commit, wrong_commit = "a" * 40, "b" * 40
    parent, _, annotation, support = _parent(generator_source_commit=source_commit)
    with pytest.raises(ValueError, match="source commit does not match"):
        make_arbitrary_plane_finite_candidate_bank(
            parent,
            annotation,
            support,
            finite_parent_generator_source_commit=wrong_commit,
        )
    bank = make_arbitrary_plane_finite_candidate_bank(
        parent,
        annotation,
        support,
        finite_parent_generator_source_commit=source_commit,
    )
    verify_arbitrary_plane_finite_candidate_bank(
        bank,
        parent,
        annotation,
        support,
        finite_parent_generator_source_commit=source_commit,
    )
    replayed = replay_arbitrary_plane_finite_candidate_bank(
        bank,
        parent,
        annotation,
        support,
        finite_parent_generator_source_commit=source_commit,
    )
    assert replayed["finite_candidate_bank_id"] == bank["finite_candidate_bank_id"]
    for operation in (
        verify_arbitrary_plane_finite_candidate_bank,
        replay_arbitrary_plane_finite_candidate_bank,
    ):
        with pytest.raises(ValueError, match="source commit does not match"):
            operation(
                bank,
                parent,
                annotation,
                support,
                finite_parent_generator_source_commit=wrong_commit,
            )


def test_prepared_annotation_context_rejects_mutation_and_forgery():
    parent, _, annotation, support = _parent()
    context = prepare_arbitrary_plane_finite_candidate_context(annotation, support)
    with pytest.raises(ValueError):
        context["annotation"][0, 0, 0] = 7
    forged = MappingProxyType({**context, "prepared_context_sha256": "0" * 64})
    with pytest.raises(ValueError, match="identity or receipt"):
        make_arbitrary_plane_finite_candidate_bank_from_context(parent, forged, support)
    with pytest.raises(ValueError, match="forged"):
        make_arbitrary_plane_finite_candidate_bank_from_context(parent, dict(context), support)


@pytest.mark.parametrize(
    "tamper",
    (
        "truth-geometry",
        "truth-scalar",
        "candidate-array",
        "candidate-geometry",
        "candidate-order",
        "attempts",
        "shuffle-seed",
        "dependency",
        "source-hash",
        "identity",
    ),
)
def test_verifier_detects_truth_candidate_order_array_and_identity_tampering(tamper):
    bank, parent, _, annotation, support = _bank()
    changed = copy.deepcopy(bank)
    if tamper == "truth-geometry":
        changed["truth_parent_geometry"]["signed_offset_um"] += 1.0
    elif tamper == "truth-scalar":
        changed["truth_parent_raster"]["scalar"][0, 0] += 1.0
    elif tamper == "candidate-array":
        changed["candidates"][0]["rendered_annotation"][0, 0] += 1
    elif tamper == "candidate-geometry":
        decoy = next(candidate for candidate in changed["candidates"] if "geometry" in candidate)
        decoy["geometry"]["effective_physical_ouv_ap_dv_ml_um"][0] += 1.0
    elif tamper == "candidate-order":
        changed["candidates"][0], changed["candidates"][1] = (
            changed["candidates"][1],
            changed["candidates"][0],
        )
    elif tamper == "attempts":
        changed["candidate_attempts"][0]["accepted"] = False
    elif tamper == "shuffle-seed":
        changed["shuffle_field_stream_seed_uint64"] = "0x0000000000000000"
    elif tamper == "dependency":
        changed["generator"]["previous_model_dependencies"] = ["legacy"]
    elif tamper == "source-hash":
        changed["generator"]["implementation"]["loaded_source_sha256"]["candidate_generator"] = "0" * 64
    else:
        changed["candidates"][0]["candidate_id"] = "0" * 64

    with pytest.raises(ValueError):
        verify_arbitrary_plane_finite_candidate_bank(changed, parent, annotation, support)


def test_local_invalid_case_preserves_hash_bound_rejection_history():
    parent, _, annotation, support = _parent(seed=1001)
    with pytest.raises(ValueError) as rejected:
        make_arbitrary_plane_finite_candidate_bank(parent, annotation, support)

    message = str(rejected.value)
    assert "finite candidate case rejected" in message
    assert '"candidate_attempts_sha256"' in message
    assert '"accepted":false' in message
    assert "fixed local slot" in message


@pytest.mark.parametrize(
    "normal",
    (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, -2.0, 3.0],
        [1e-7, 1.0, -2e-7],
    ),
)
def test_ap_dv_ml_general_and_extreme_oblique_parents_transport_on_same_canvas(normal):
    _, annotation, support = _atlas()
    parent = _fixed_geometry(normal, support, output_shape=(41, 55), roll=1.13)
    candidate = transport_finite_candidate_pose(
        parent,
        support,
        parent["normal_rp2_ap_dv_ml"],
        parent["signed_offset_um"] + 100.0,
        math.radians(10.0),
    )
    labels, brain = render_finite_candidate_annotation(annotation, candidate)
    frame = np.asarray(candidate["frame_ap_dv_ml_physical"])

    assert candidate["output_shape_h_w"] == parent["output_shape_h_w"]
    assert candidate["physical_pixel_pitch_u_v_um"] == [
        parent["reference_aspect_policy"]["pixel_pitch_u_um"],
        parent["reference_aspect_policy"]["pixel_pitch_v_um"],
    ]
    assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0, abs=1e-12)
    assert labels.shape == tuple(parent["output_shape_h_w"])
    assert int(brain.sum()) >= 64


def test_default_root_seeds_are_the_predeclared_roots():
    assert DEFAULT_CANDIDATE_ROOT_SEED == int("caad1da7e0000001", 16)
    assert DEFAULT_SHUFFLE_ROOT_SEED == int("5c0e5eed00000001", 16)


def test_final_order_seed_uses_the_frozen_class_and_bank_slot_literals():
    base_id = "a" * 64
    payload = (
        "arbitrary-plane-finite-candidates/v1\0"
        "0x5c0e5eed00000001\0"
        f"{base_id}\0final-order\0bank\0" + "0"
    )
    expected = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")
    assert derive_finite_candidate_seed(
        DEFAULT_SHUFFLE_ROOT_SEED, base_id, "final-order", "bank", 0
    ) == expected
