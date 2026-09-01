from collections.abc import Mapping

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_section_processing_v2 as processing
import training.arbitrary_plane_subject_slab_v2 as subject_slab
import training.arbitrary_plane_synthetic_generator_v2 as slab
from training.arbitrary_plane_support import build_annotation_support_index


SHAPE = (7, 8)
PITCH = (2.0, 5.0)


def _plan(
    mode="standard",
    *,
    root_seed=3,
    animal_index=1,
    animal_id="animal-A",
    section_id="section-A",
    section_index=2,
    split="train",
):
    return processing.sample_section_processing_plan_v2(
        SHAPE,
        PITCH,
        root_seed=root_seed,
        split=split,
        animal_index=animal_index,
        section_index=section_index,
        animal_id=animal_id,
        section_id=section_id,
        deformation_mode=mode,
        coarse_spacing_yx_um=(8.0, 10.0),
        fine_spacing_yx_um=(4.0, 5.0),
        coarse_padding_um=40.0,
        fine_padding_um=20.0,
        a0_um=1.0,
        cycle_max_um=1.0,
        maximum_displacement_um=10.0,
        component_derivative_abs_max=2.0,
        gradient_frobenius_bound_max=2.0,
        divergence_abs_bound_max=2.0,
        speed_l2_bound_um_max=10.0,
        minimum_halo_um=0.0,
        affine_residual_max_um=1.0e-5,
    )


@pytest.fixture(scope="module")
def standard_plan():
    return _plan()


@pytest.fixture(scope="module")
def identity_plan():
    return _plan("identity")


def _source(*, finite_invalid_coordinate=True):
    y, x = np.indices(SHAPE)
    scalar = (10 * y + x).astype(np.float32)
    annotation = (1 + 10 * y + x).astype(np.int64)
    support = np.ones(SHAPE, dtype=bool)
    support[3, 3] = False
    weight = support.astype(np.float32)
    coordinates = np.stack(
        (100.0 + 2.0 * y, 200.0 + 3.0 * x, 300.0 + y - x), axis=-1
    ).astype(np.float64)
    if not finite_invalid_coordinate:
        coordinates[3, 3] = np.nan
    raster = {
        "scalar": scalar,
        "centre_plane_annotation": annotation,
        "centre_plane_support_mask": support,
        "slab_brain_occupancy": weight,
        "slab_observable_support_mask": support,
        "slab_modal_annotation": annotation + 100,
        "slab_label_purity": weight,
        "centre_label_support_weight": weight,
        "slab_supervision_weight_or_abstention": {
            "dense_correspondence_weight": weight,
            "abstention_mask": ~support,
        },
    }
    return raster, coordinates


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, order="C")
    return value


def _verify_plan(
    plan,
    animal_id="animal-A",
    section_id="section-A",
    section_index=2,
    split="train",
):
    processing.verify_section_processing_plan_v2(
        plan,
        expected_image_shape_h_w=SHAPE,
        expected_pixel_pitch_y_x_um=PITCH,
        expected_split=split,
        expected_animal_index=1,
        expected_section_index=section_index,
        expected_animal_id=animal_id,
        expected_section_id=section_id,
    )


def test_rng_is_section_scoped_and_labels_never_enter_random_streams(standard_plan):
    relabelled = _plan(animal_id="renamed-animal", section_id="renamed-section")
    next_section = _plan(section_index=3)
    assert relabelled["rng_sources"] == standard_plan["rng_sources"]
    assert np.array_equal(
        relabelled["state"]["raw_coarse_coefficients"],
        standard_plan["state"]["raw_coarse_coefficients"],
    )
    assert np.array_equal(
        relabelled["state"]["raw_fine_coefficients"],
        standard_plan["state"]["raw_fine_coefficients"],
    )
    assert relabelled["section_processing_plan_id"] != standard_plan[
        "section_processing_plan_id"
    ]
    assert next_section["rng_sources"] != standard_plan["rng_sources"]
    for key in (
        "learned_checkpoint_dependencies",
        "pretrained_feature_dependencies",
        "previous_model_dependencies",
    ):
        assert standard_plan["resolved_config"][key] == ()


@pytest.mark.parametrize(
    ("root_seed", "animal_index", "section_index", "attempt"),
    [
        (1.0, 1, 2, 0),
        (True, 1, 2, 0),
        (3, 1.0, 2, 0),
        (3, np.bool_(False), 2, 0),
        (3, 1, 2.0, 0),
        (3, 1, np.bool_(True), 0),
        (3, 1, 2, 0.0),
        (3, 1, 2, True),
    ],
)
def test_section_rng_rejects_noninteger_provenance_coordinates(
    root_seed, animal_index, section_index, attempt
):
    with pytest.raises(ValueError):
        processing.derive_section_processing_seed_v2(
            root_seed,
            "train",
            animal_index,
            section_index,
            "section-processing",
            "coarse-svf",
            attempt,
        )


@pytest.mark.parametrize(
    ("stage", "field"),
    [(7, "coarse-svf"), ("", "coarse-svf"), ("section-processing", object()),
     ("section-processing", "")],
)
def test_section_rng_rejects_nonstring_or_empty_domain_names(stage, field):
    with pytest.raises(ValueError):
        processing.derive_section_processing_seed_v2(
            3, "train", 1, 2, stage, field, 0
        )


def test_section_rng_and_plan_accept_numpy_integer_coordinates():
    assert processing.derive_section_processing_seed_v2(
        np.uint64(3),
        "train",
        np.int64(1),
        np.int64(2),
        "section-processing",
        "coarse-svf",
        np.int64(0),
    ) == processing.derive_section_processing_seed_v2(
        3, "train", 1, 2, "section-processing", "coarse-svf", 0
    )
    plan = _plan(
        "identity",
        root_seed=np.uint64(3),
        animal_index=np.int64(1),
        section_index=np.int64(2),
    )
    assert plan["provenance"]["root_seed_uint64"] == "0x0000000000000003"
    assert plan["provenance"]["animal_index"] == 1
    assert plan["provenance"]["section_index"] == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"root_seed": 1.0},
        {"root_seed": np.bool_(True)},
        {"animal_index": 1.0},
        {"animal_index": True},
        {"section_index": 2.0},
        {"section_index": np.bool_(False)},
    ],
)
def test_section_plan_rejects_noninteger_provenance_coordinates(overrides):
    with pytest.raises(ValueError):
        _plan("identity", **overrides)


@pytest.mark.parametrize(
    ("expected_animal_index", "expected_section_index"),
    [(1.0, 2), (True, 2), (1, 2.0), (1, np.bool_(False))],
)
def test_section_verifier_rejects_noninteger_authoritative_indices(
    identity_plan, expected_animal_index, expected_section_index
):
    with pytest.raises(ValueError):
        processing.verify_section_processing_plan_v2(
            identity_plan,
            expected_image_shape_h_w=SHAPE,
            expected_pixel_pitch_y_x_um=PITCH,
            expected_split="train",
            expected_animal_index=expected_animal_index,
            expected_section_index=expected_section_index,
            expected_animal_id="animal-A",
            expected_section_id="section-A",
        )


def test_arbitrary_nonempty_split_names_are_bound_without_changing_label_independence():
    split = "animal-heldout-synthetic-test"
    plan = _plan("identity", split=split)
    _verify_plan(plan, split=split)
    assert plan["provenance"]["split"] == split
    assert plan["rng_sources"] != _plan("identity", split="train")["rng_sources"]
    with pytest.raises(ValueError, match="invalid"):
        _plan("identity", split="")


def test_plan_binds_half_fine_grid_affine_free_realization_and_replay(standard_plan):
    _verify_plan(standard_plan)
    state = standard_plan["state"]
    assert np.array_equal(
        state["grid_maximum_spacing_yx_um"], state["fine_spacing_yx_um"] / 2.0
    )
    accepted_index = standard_plan["realization"]["accepted_candidate_index"]
    audits = standard_plan["realization"]["candidate_audits"]
    assert len(audits) == accepted_index + 1
    assert not any(audit["accepted"] for audit in audits[:accepted_index])
    assert audits[accepted_index]["accepted"]
    field = processing._accepted_field(standard_plan)
    velocity = field(state["projection_grid_yx_um"], return_gradient=False)
    affine = processing._physical_affine(
        state["projection_grid_yx_um"],
        velocity,
        state["domain_center_yx_um"],
        state["domain_half_extent_yx_um"],
    )
    assert np.max(np.abs(affine)) <= 1.0e-5
    assert audits[accepted_index]["gate_values"]["forward_jacobian_det_min"] > 0.0
    assert audits[accepted_index]["gate_values"]["inverse_jacobian_det_min"] > 0.0
    assert processing.section_processing_plan_receipt_v2(
        processing.replay_section_processing_plan_v2(standard_plan)
    ) == processing.section_processing_plan_receipt_v2(standard_plan)

    tampered = _thaw(standard_plan)
    tampered["state"]["accepted_coarse_coefficients_yx_um"][0, 0, 0] += 1.0
    with pytest.raises(ValueError):
        _verify_plan(tampered)
    with pytest.raises(ValueError):
        processing.verify_section_processing_plan_v2(
            standard_plan,
            expected_image_shape_h_w=SHAPE,
            expected_pixel_pitch_y_x_um=(5.0, 2.0),
            expected_split="train",
            expected_animal_index=1,
            expected_section_index=2,
            expected_animal_id="animal-A",
            expected_section_id="section-A",
        )


def test_rk4_step_orientation_certificate_rejects_fold_risk(standard_plan):
    safe = processing.rk4_step_orientation_certificate_2d_v2(1.0, 8)
    unsafe = processing.rk4_step_orientation_certificate_2d_v2(2.0, 1)
    assert safe["rk4_step_orientation_certified"]
    assert not unsafe["rk4_step_orientation_certified"]

    state = standard_plan["state"]
    coarse_shape = state["projected_coarse_unit_coefficients"].shape[:2]
    coarse_modes = processing._affine_modes(
        coarse_shape,
        state["coarse_origin_yx_um"],
        state["coarse_spacing_yx_um"],
        state["domain_center_yx_um"],
        processing._taper(coarse_shape),
    )
    coarse_response = np.stack(
        [
            processing._physical_affine(
                state["projection_grid_yx_um"],
                processing.cubic_bspline_velocity_2d_v2(
                    state["projection_grid_yx_um"],
                    mode,
                    state["coarse_origin_yx_um"],
                    state["coarse_spacing_yx_um"],
                ),
                state["domain_center_yx_um"],
                state["domain_half_extent_yx_um"],
            )
            for mode in coarse_modes
        ],
        1,
    )
    limits = {
        "jacobian_det_min": -1.0e9,
        "jacobian_det_max": 1.0e9,
        "cycle_max_um": 1.0e9,
        "maximum_displacement_um": 1.0e9,
        "component_derivative_abs_max": 1.0e9,
        "gradient_frobenius_bound": 1.0e9,
        "divergence_abs_bound": 1.0e9,
        "speed_l2_bound_um": 1.0e9,
        "physical_affine_residual_max_um": 1.0e9,
        "minimum_halo_um": -1.0e9,
        "rk4_step_jacobian_perturbation_bound": 1.0,
    }
    audit, _, _, _ = processing._candidate_audit(
        state,
        state["projected_coarse_unit_coefficients"],
        state["projected_fine_unit_coefficients"],
        coarse_modes,
        coarse_response,
        2.0,
        1,
        limits,
    )
    assert "rk4_step_orientation_certificate" in audit["failed_gates"]
    assert not audit["accepted"]


def test_identity_point_map_is_bitwise_and_standard_direction_is_inverse(
    identity_plan, standard_plan, monkeypatch
):
    signed = np.asarray([[-0.0, 0.0], [1.25, -2.5]], dtype=np.float32)
    mapped = processing.subject_to_processed_points_yx_um_v2(signed, identity_plan)
    pulled = processing.processed_to_subject_points_yx_um_v2(signed, identity_plan)
    assert mapped.dtype == signed.dtype and mapped.tobytes() == signed.tobytes()
    assert pulled.dtype == signed.dtype and pulled.tobytes() == signed.tobytes()

    centres = processing._processed_pixel_geometry(standard_plan, None)[
        "processed_pixel_centres_yx_um"
    ]
    assert np.array_equal(centres[0, 0], np.asarray(PITCH) / 2.0)
    assert np.array_equal(centres[1, 0] - centres[0, 0], [PITCH[0], 0.0])
    assert np.array_equal(centres[0, 1] - centres[0, 0], [0.0, PITCH[1]])
    subject = processing.processed_to_subject_points_yx_um_v2(centres, standard_plan)
    restored = processing.subject_to_processed_points_yx_um_v2(subject, standard_plan)
    wrong_direction = processing.subject_to_processed_points_yx_um_v2(
        centres, standard_plan
    )
    assert np.max(np.linalg.norm(restored - centres, axis=-1)) < 1.0e-5
    assert not np.allclose(subject, wrong_direction)
    source_index = processing._processed_pixel_geometry(standard_plan, None)[
        "source_index_yx"
    ]
    assert np.array_equal(source_index, subject / np.asarray(PITCH) - 0.5)
    assert not np.allclose(source_index, subject / np.asarray(PITCH[::-1]) - 0.5)

    constant_velocity = np.asarray([2.0, -3.0], dtype=np.float64)

    def constant_field(points, *, return_gradient=False):
        velocity = np.broadcast_to(constant_velocity, np.asarray(points).shape)
        if not return_gradient:
            return velocity
        return velocity, np.zeros(np.asarray(points).shape[:-1] + (2, 2))

    monkeypatch.setattr(processing, "_accepted_field", lambda plan: constant_field)
    anchors = np.asarray([[1.25, -2.5], [7.0, 11.0]], dtype=np.float64)
    assert np.array_equal(
        processing.subject_to_processed_points_yx_um_v2(anchors, standard_plan),
        anchors + constant_velocity,
    )
    assert np.array_equal(
        processing.processed_to_subject_points_yx_um_v2(anchors, standard_plan),
        anchors - constant_velocity,
    )


def test_strict_coordinate_interpolation_preserves_missing_and_closed_edge_rule():
    coordinates = np.zeros((3, 3, 3), dtype=np.float64)
    y, x = np.indices((3, 3))
    coordinates[..., 0] = y
    coordinates[..., 1] = x
    coordinates[..., 2] = y + x
    source_valid = np.ones((3, 3), dtype=bool)
    source_valid[1, 1] = False
    coordinates[1, 1] = np.nan
    queries = np.asarray([[[0.0, 0.0], [0.25, 0.25], [-0.1, 0.0], [2.0, 2.0]]])
    sampled, valid = processing._strict_valid_bilinear_coordinates(
        coordinates, source_valid, queries
    )
    assert valid.tolist() == [[True, False, False, True]]
    assert np.array_equal(sampled[0, 0], coordinates[0, 0])
    assert np.isnan(sampled[0, 1]).all()
    assert np.isnan(sampled[0, 2]).all()
    assert np.array_equal(sampled[0, 3], coordinates[2, 2])


def test_physical_pixel_metric_accepts_rotated_rectangles_and_rejects_shear_or_bowing():
    y, x = np.indices((4, 5))
    origin = np.asarray([11.0, 23.0, 37.0])
    y_vector = np.asarray([1.2, 1.6, 0.0])
    x_vector = np.asarray([-2.4, 1.8, 0.0])
    coordinates = (
        origin[None, None]
        + y[..., None] * y_vector
        + x[..., None] * x_vector
    )
    pitch, y_steps, x_steps = processing._orthogonal_section_pixel_metric(coordinates)
    assert np.allclose(pitch, [2.0, 3.0], rtol=0.0, atol=1.0e-14)
    assert np.allclose(y_steps, 2.0, rtol=0.0, atol=1.0e-14)
    assert np.allclose(x_steps, 3.0, rtol=0.0, atol=1.0e-14)

    sheared = (
        origin[None, None]
        + y[..., None] * y_vector
        + x[..., None] * (x_vector + 0.25 * y_vector)
    )
    with pytest.raises(ValueError, match="constant orthogonal parallelogram"):
        processing._orthogonal_section_pixel_metric(sheared)

    bowed = coordinates.copy()
    bowed[2:, 2:, 2] += 0.1
    with pytest.raises(ValueError, match="constant orthogonal parallelogram"):
        processing._orthogonal_section_pixel_metric(bowed)


def test_identity_render_is_bitwise_and_pose_reference_is_separate(identity_plan):
    raster, coordinates = _source(finite_invalid_coordinate=True)
    source_receipt = {"subject_slab_render_id": "source-slab", "receipt": "exact"}
    pose_reference = {"plane_pose_id": "immutable-pose", "anatomy_id": "immutable-anatomy"}
    render = processing._make_section_processing_render_from_arrays_v2(
        raster,
        coordinates,
        identity_plan,
        source_stage_receipt=source_receipt,
        pose_anatomy_reference=pose_reference,
    )
    processing._verify_section_processing_render_from_arrays_v2(
        render,
        raster,
        coordinates,
        identity_plan,
        source_stage_receipt=source_receipt,
        pose_anatomy_reference=pose_reference,
    )
    output = processing._flatten_section_raster(render["raster"])
    source = processing._flatten_section_raster(raster)
    assert all(processing._byte_equal(output[name], source[name]) for name in source)
    assert processing._byte_equal(
        render["mapped_ccf_physical_coordinates_ap_dv_ml_um"], coordinates
    )
    assert render["pose_anatomy_policy"]["pose_anatomy_reference"] == pose_reference
    assert render["pose_anatomy_policy"][
        "processing_warp_is_separate_from_plane_pose"
    ]


def test_mixed_interpolation_has_independent_known_values_through_render(
    monkeypatch, standard_plan
):
    raster, coordinates = _source(finite_invalid_coordinate=False)
    raster["centre_plane_annotation"][3, 3] = 0
    raster["slab_modal_annotation"][3, 3] = 0
    y, x = np.indices(SHAPE)
    source_index = np.stack((y, x), axis=-1).astype(np.float64)
    source_index[0, 0] = [0.5, 0.5]
    source_index[0, 1] = [2.5, 2.5]
    source_index[0, 2] = [-0.25, 1.0]
    source_index[0, 3] = [SHAPE[0] - 1.0, SHAPE[1] - 1.0]
    pitch = np.asarray(PITCH, dtype=np.float64)
    centres = np.stack(
        np.meshgrid(
            (np.arange(SHAPE[0]) + 0.5) * pitch[0],
            (np.arange(SHAPE[1]) + 0.5) * pitch[1],
            indexing="ij",
        ),
        -1,
    )
    subject_points = (source_index + 0.5) * pitch

    monkeypatch.setattr(
        processing,
        "_processed_pixel_geometry",
        lambda plan, batch_size: {
            "processed_pixel_centres_yx_um": np.asarray(centres, dtype="<f8"),
            "subject_pullback_points_yx_um": np.asarray(subject_points, dtype="<f8"),
            "source_index_yx": np.asarray(source_index, dtype="<f8"),
        },
    )
    source_receipt = {"subject_slab_render_id": "known-value-source"}
    pose_reference = {"plane_pose_id": "known-value-pose"}
    render = processing._make_section_processing_render_from_arrays_v2(
        raster,
        coordinates,
        standard_plan,
        source_stage_receipt=source_receipt,
        pose_anatomy_reference=pose_reference,
    )
    processing._verify_section_processing_render_from_arrays_v2(
        render,
        raster,
        coordinates,
        standard_plan,
        source_stage_receipt=source_receipt,
        pose_anatomy_reference=pose_reference,
    )
    output = processing._flatten_section_raster(render["raster"])
    state = render["state"]
    mapped = render["mapped_ccf_physical_coordinates_ap_dv_ml_um"]
    assert output["scalar"][0, 0] == np.float32(5.5)
    assert output["scalar"][0, 2] == np.float32(0.75)
    assert output["scalar"][0, 3] == raster["scalar"][-1, -1]
    assert output["centre_plane_annotation"][0, 0] == raster[
        "centre_plane_annotation"
    ][1, 1]
    assert output["centre_plane_annotation"][0, 1] == 0
    assert output["centre_plane_annotation"][0, 2] == raster[
        "centre_plane_annotation"
    ][0, 1]
    expected_coordinate = (
        coordinates[0, 0]
        + coordinates[0, 1]
        + coordinates[1, 0]
        + coordinates[1, 1]
    ) / 4.0
    assert np.array_equal(mapped[0, 0], expected_coordinate)
    assert np.isnan(mapped[0, 1]).all()
    assert np.isnan(mapped[0, 2]).all()
    assert np.array_equal(mapped[0, 3], coordinates[-1, -1])
    assert state["dense_coordinate_valid_mask"][0, [0, 1, 2, 3]].tolist() == [
        True,
        False,
        False,
        True,
    ]
    assert not state["bilinear_domain_valid_mask"][0, 2]
    assert state["nearest_domain_valid_mask"][0, 2]
    assert np.array_equal(
        output["dense_correspondence_abstention_mask"],
        ~state["dense_coordinate_valid_mask"],
    )
    assert np.all(
        output["dense_correspondence_weight"][~state["dense_coordinate_valid_mask"]]
        == 0.0
    )
    assert not processing._byte_equal(output["scalar"], raster["scalar"])


def test_nonidentity_render_warps_all_fields_and_authenticates_replay(standard_plan):
    raster, coordinates = _source(finite_invalid_coordinate=False)
    source_receipt = {"subject_slab_render_id": "source-slab", "receipt": "exact"}
    pose_reference = {"plane_pose_id": "immutable-pose", "anatomy_id": "immutable-anatomy"}
    render = processing._make_section_processing_render_from_arrays_v2(
        raster,
        coordinates,
        standard_plan,
        source_stage_receipt=source_receipt,
        pose_anatomy_reference=pose_reference,
    )
    processing._verify_section_processing_render_from_arrays_v2(
        render,
        raster,
        coordinates,
        standard_plan,
        source_stage_receipt=source_receipt,
        pose_anatomy_reference=pose_reference,
    )
    output = processing._flatten_section_raster(render["raster"])
    dense_valid = render["state"]["dense_coordinate_valid_mask"]
    mapped = render["mapped_ccf_physical_coordinates_ap_dv_ml_um"]
    assert set(output) == processing._SECTION_RASTER_KEYS
    assert np.isnan(mapped[~dense_valid]).all()
    assert np.isfinite(mapped[dense_valid]).all()
    assert np.all(output["dense_correspondence_weight"][~dense_valid] == 0.0)
    assert np.array_equal(
        output["dense_correspondence_abstention_mask"], ~dense_valid
    )
    assert render["state"]["source_index_yx"].shape == SHAPE + (2,)
    assert set(render["interpolation_semantics"]) == {
        "scalar",
        "continuous_supervision",
        "categorical_labels",
        "support_masks",
        "mapped_ccf_coordinates",
        "dense_correspondence_weight",
        "dense_correspondence_abstention_mask",
    }
    replay = processing._replay_section_processing_render_from_arrays_v2(
        render,
        raster,
        coordinates,
        standard_plan,
        source_stage_receipt=source_receipt,
        pose_anatomy_reference=pose_reference,
    )
    assert processing.section_processing_render_receipt_v2(
        replay
    ) == processing.section_processing_render_receipt_v2(render)

    tampered = _thaw(render)
    tampered["raster"]["scalar"][0, 0] += 1.0
    with pytest.raises(ValueError):
        processing._verify_section_processing_render_from_arrays_v2(
            tampered,
            raster,
            coordinates,
            standard_plan,
            source_stage_receipt=source_receipt,
            pose_anatomy_reference=pose_reference,
        )
    with pytest.raises(ValueError):
        processing._verify_section_processing_render_from_arrays_v2(
            render,
            raster,
            coordinates,
            standard_plan,
            source_stage_receipt={**source_receipt, "extra": "tamper"},
            pose_anatomy_reference=pose_reference,
        )


def test_public_maker_authenticates_subject_slab_before_render(monkeypatch, standard_plan):
    import training.arbitrary_plane_subject_slab_v2 as subject_slab

    raster, coordinates = _source(finite_invalid_coordinate=False)
    calls = []

    def authenticated(source, context, precursor, *, subject_plan):
        calls.append((source, context, precursor, subject_plan))

    source = {
        "precursor_reference": {
            "split": "train",
            "animal_index": 1,
            "plane_sample_index": 2,
            "animal_id": "animal-A",
        }
    }
    monkeypatch.setattr(subject_slab, "verify_subject_slab_render_v2", authenticated)
    monkeypatch.setattr(
        processing,
        "_subject_slab_inputs",
        lambda source, plan: (
            raster,
            coordinates,
            {"subject_slab_render_id": "authenticated-source"},
            {"plane_pose_id": "preserved"},
        ),
    )
    render = processing.make_section_processing_render_v2(
        source,
        standard_plan,
        {"context": "authoritative"},
        {"precursor": "authoritative"},
        subject_plan={"subject_plan": "authoritative"},
    )
    assert len(calls) == 1
    assert render["source_input_reference"]["source_stage_receipt"] == {
        "subject_slab_render_id": "authenticated-source"
    }


def test_real_authenticated_public_chain_and_upstream_tamper_rejection():
    annotation = np.zeros((9, 8, 7), dtype=np.uint16)
    annotation[1:8, 1:7, 1:6] = 7
    ap, dv, ml = np.indices(annotation.shape)
    scalar = (1 + ap + 2 * dv + 3 * ml).astype(np.float32)
    support = build_annotation_support_index(
        annotation,
        atlas_id="section-processing-chain-fixture",
        atlas_version="fixture-v1",
        source_uri="file:///fixture/annotation.nrrd",
        source_sha256="3" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(11.0, 17.0, 29.0),
        origin_um=(-71.0, 23.0, 107.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    context = acquisition.prepare_arbitrary_plane_acquisition_context_v2(
        scalar,
        annotation,
        support,
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="4" * 64,
        template_decoder="fixture decoder",
        annotation_decoder="fixture decoder",
    )
    alignments = []
    precursor = source = subject_physical = pitch = None
    for plane_sample_index in (0, 2, 12, 17):
        current_precursor = slab.make_v2_generic_global_reference_slab_render(
            context,
            "development",
            "0x415154564f320001",
            plane_sample_index,
            "reference",
            nominal_cut_thickness_um=10.0,
            axial_step_um_max=10.0,
            animal_id="chain-animal",
            animal_index=4,
            specimen_id="chain-specimen",
            experiment_id="chain-experiment",
        )
        current_source = subject_slab.make_subject_slab_render_v2(
            context, current_precursor, subject_plan=None
        )
        coordinate = current_source["coordinate_map"]
        centre_index = coordinate["kernel"]["centre_index"]
        current_physical = coordinate["arrays"][
            "subject_physical_coordinates_ap_dv_ml_um_float64"
        ][centre_index]
        current_pitch, _, _ = (
            processing._orthogonal_section_pixel_metric(current_physical)
        )
        y_vectors = current_physical[1:] - current_physical[:-1]
        x_vectors = current_physical[:, 1:] - current_physical[:, :-1]
        y_reference = y_vectors.mean(axis=(0, 1))
        x_reference = x_vectors.mean(axis=(0, 1))
        parallelogram = (
            current_physical[1:, 1:]
            - current_physical[1:, :-1]
            - current_physical[:-1, 1:]
            + current_physical[:-1, :-1]
        )
        relative_residual = max(
            np.linalg.norm(y_vectors - y_reference, axis=-1).max()
            / current_pitch[0],
            np.linalg.norm(x_vectors - x_reference, axis=-1).max()
            / current_pitch[1],
            np.linalg.norm(parallelogram, axis=-1).max() / current_pitch.max(),
            abs(y_reference @ x_reference)
            / (np.linalg.norm(y_reference) * np.linalg.norm(x_reference)),
        )
        assert relative_residual < (
            0.75 * processing.SECTION_PROCESSING_V2_PIXEL_METRIC_RELATIVE_TOLERANCE
        )
        normal = np.cross(y_reference, x_reference)
        alignments.append(np.max(np.abs(normal / np.linalg.norm(normal))))
        if plane_sample_index == 0:
            precursor = current_precursor
            source = current_source
            subject_physical = current_physical
            pitch = current_pitch
    assert min(alignments) < 0.7
    assert max(alignments) > 0.98
    plan = processing.sample_section_processing_plan_v2(
        tuple(subject_physical.shape[:2]),
        tuple(pitch),
        root_seed=19,
        split="development",
        animal_index=4,
        section_index=0,
        animal_id="chain-animal",
        section_id="processing-section-0",
        deformation_mode="identity",
    )
    render = processing.make_section_processing_render_v2(
        source,
        plan,
        context,
        precursor,
        subject_plan=None,
    )
    processing.verify_section_processing_render_v2(
        render,
        source,
        plan,
        context,
        precursor,
        subject_plan=None,
    )
    assert plan["resolved_config"]["section_id_policy"] == (
        "processing-stage provenance identifier bound to this plan and render, "
        "excluded from RNG, and not claimed as upstream subject-slab provenance"
    )
    assert render["source_input_reference"]["pose_anatomy_reference"][
        "precursor_reference"
    ]["plane_sample_index"] == 0

    tampered = _thaw(source)
    tampered["coordinate_map"]["arrays"][
        "subject_physical_coordinates_ap_dv_ml_um_float64"
    ][centre_index, 0, 0, 0] += 1.0
    with pytest.raises(ValueError):
        processing.make_section_processing_render_v2(
            tampered,
            plan,
            context,
            precursor,
            subject_plan=None,
        )
