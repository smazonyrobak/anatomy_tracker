import copy
import inspect

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_acquisition_window_v3 as window
import training.arbitrary_plane_observation_v3 as observation


ROOT_SEED = "0x4f42535633000001"
BASE = {
    "subject_plan": {"synthetic_animal_id": "synthetic-animal-007"},
    "root_seed": ROOT_SEED,
    "split": "train",
    "split_index": 1,
    "animal_index": 7,
    "animal_id": "animal-007",
    "section_index": 3,
    "observation_index": 2,
    "modality": "brightfield-nissl-like",
}


def _live_arrays(render):
    raster, state = render["raster"], render["state"]
    return {
        "scalar": raster["scalar"],
        "labels": raster["slab_modal_annotation"],
        "centre": raster["centre_plane_support_mask"],
        "occupancy": raster["slab_brain_occupancy"],
        "tissue": raster["slab_observable_support_mask"],
        "weight": raster["slab_supervision_weight_or_abstention"][
            "dense_correspondence_weight"
        ],
        "abstention": raster["slab_supervision_weight_or_abstention"][
            "abstention_mask"
        ],
        "mapped": render["mapped_ccf_physical_coordinates_ap_dv_ml_um"],
        "bilinear": state["bilinear_domain_valid_mask"],
        "nearest": state["nearest_domain_valid_mask"],
        "dense": state["dense_coordinate_valid_mask"],
        "processed_centres": state["processed_pixel_centres_yx_um"],
        "source_index": state["source_index_yx"],
    }


def _parent_receipt(render):
    payload = {
        "section_processing_render_id": render["section_processing_render_id"],
        "array_receipts": {
            name: acquisition._array_receipt(array)
            for name, array in _live_arrays(render).items()
        },
    }
    return {**payload, "receipt_sha256": acquisition._payload_sha256(payload)}


def _refresh(inputs):
    render = inputs["processed_render"]
    render["authenticated_test_receipt"] = _parent_receipt(render)


def _inputs():
    height, width = 256, 256
    y, x = np.indices((height, width))
    tissue = ((x - 127.5) / 105.0) ** 2 + ((y - 127.5) / 92.0) ** 2 <= 1.0
    centre = tissue & (x >= 18) & (x < 238)
    scalar = (80.0 + 1.7 * y + 0.8 * x + 7.0 * np.sin(x / 5.0)).astype(
        np.float32
    )
    labels = np.zeros((height, width), dtype=np.int64)
    labels[tissue & (x < 96)] = 11
    labels[tissue & (x >= 96) & (x < 164)] = 23
    labels[tissue & (x >= 164)] = 37
    dense = tissue & (x < 235) & (y > 10)
    weight = np.zeros((height, width), dtype=np.float32)
    weight[dense] = np.where((x[dense] + y[dense]) % 3, 1.0, 0.25).astype(
        np.float32
    )
    mapped = np.stack(
        (100.0 + 18.0 * y, 200.0 + 21.0 * x, 300.0 + 3.0 * x + 2.0 * y),
        axis=-1,
    ).astype(np.float64)
    mapped[~dense] = np.nan
    design = np.array(
        [[100.0, 200.0, 300.0], [0.0, 1024.0, 0.0], [1024.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    section_plan = {
        "section_processing_plan_id": "section-plan-id",
        "section_processing_realization_id": "section-realization-id",
        "synthetic_section_processing_id": "synthetic-section-id",
        "resolved_config": {
            "deformation_mode": "identity",
            "pixel_pitch_y_x_um": [4.0, 8.0],
        },
        "provenance": {
            "split": "train",
            "animal_index": 7,
            "animal_id": "animal-007",
            "section_index": 3,
            "section_id": "section-003",
        },
    }
    subject = {
        "subject_slab_render_id": "subject-slab-id",
        "synthetic_animal_id": "synthetic-animal-007",
    }
    precursor = {
        "v2_plane_realization_id": "v2-plane-id",
        "slab_render_id": "precursor-slab-id",
        "generator": {"resolved_config": {"sample_index": 3}},
        "provenance": {
            "animal_id": "animal-007",
            "animal_index": 7,
            "specimen_id": "specimen-007-A",
            "experiment_id": "experiment-2026-007",
        },
    }
    pose = {
        "precursor_reference": {
            "global_reference_grid_id": "global-reference-grid-id"
        },
        "centre_plane_fit": {
            "arrays": {
                "physical_ouv_ap_dv_ml_um_float64": design.reshape(-1).copy()
            }
        },
    }
    render = {
        "section_processing_render_id": "section-render-id",
        "pose_anatomy_policy": {"pose_anatomy_reference": pose},
        "raster": {
            "scalar": scalar,
            "slab_modal_annotation": labels,
            "centre_plane_support_mask": centre,
            "slab_brain_occupancy": np.where(
                tissue, 0.25 + 0.75 * (x / (width - 1)), 0.0
            ).astype(np.float32),
            "slab_observable_support_mask": tissue,
            "slab_supervision_weight_or_abstention": {
                "dense_correspondence_weight": weight,
                "abstention_mask": ~dense,
            },
        },
        "state": {
            "bilinear_domain_valid_mask": np.ones((height, width), dtype=bool),
            "nearest_domain_valid_mask": np.ones((height, width), dtype=bool),
            "dense_coordinate_valid_mask": dense,
            "processed_pixel_centres_yx_um": np.stack(
                ((y + 0.5) * 4.0, (x + 0.5) * 8.0), axis=-1
            ).astype(np.float64),
            "source_index_yx": np.stack((y, x), axis=-1).astype(np.float64),
        },
        "mapped_ccf_physical_coordinates_ap_dv_ml_um": mapped,
    }
    result = {
        "processed_render": render,
        "subject_slab_render": subject,
        "section_processing_plan": section_plan,
        "prepared_context": {"context": "authenticated-test-context"},
        "precursor": precursor,
    }
    _refresh(result)
    return result


def _fake_verify_parent(
    render,
    subject,
    plan,
    prepared_context,
    precursor,
    *,
    subject_plan,
    batch_size=None,
    subject_to_ccf_mapper=None,
):
    if (
        render["authenticated_test_receipt"] != _parent_receipt(render)
        or subject["subject_slab_render_id"] != "subject-slab-id"
        or plan["provenance"]["animal_id"] != precursor["provenance"]["animal_id"]
        or subject_plan["synthetic_animal_id"] != subject["synthetic_animal_id"]
        or prepared_context["context"] != "authenticated-test-context"
    ):
        raise ValueError("authenticated section-processing parent changed")


@pytest.fixture(autouse=True)
def patched_parent_verifier(monkeypatch):
    monkeypatch.setattr(
        observation,
        "_verify_section_processing_render_with_mapper_v2",
        _fake_verify_parent,
    )
    monkeypatch.setattr(observation, "section_processing_render_receipt_v2", _parent_receipt)
    monkeypatch.setattr(
        observation.section_processing,
        "_accepted_field",
        lambda plan: lambda points, return_gradient=False: np.zeros_like(points),
    )


@pytest.fixture
def inputs():
    return _inputs()


def _plan(inputs, sample_index=3):
    return window.sample_acquisition_window_plan_v3(
        root_seed=ROOT_SEED,
        split="train",
        sample_index=sample_index,
    )


def _make(inputs, plan=None, **changes):
    arguments = {**BASE, **changes}
    if plan is None:
        plan = _plan(inputs)
    return observation.make_arbitrary_plane_observation_v3(
        inputs["processed_render"],
        inputs["subject_slab_render"],
        inputs["section_processing_plan"],
        inputs["prepared_context"],
        inputs["precursor"],
        plan,
        **arguments,
    )


def _verify(artifact, inputs, plan=None, **changes):
    arguments = {**BASE, **changes}
    if plan is None:
        plan = _plan(inputs)
    observation.verify_arbitrary_plane_observation_v3(
        artifact,
        inputs["processed_render"],
        inputs["subject_slab_render"],
        inputs["section_processing_plan"],
        inputs["prepared_context"],
        inputs["precursor"],
        plan,
        **arguments,
    )


def test_api_requires_separate_plan_and_never_accepts_lineage_labels_as_rng_inputs(inputs):
    parameters = inspect.signature(
        observation.make_arbitrary_plane_observation_v3
    ).parameters
    assert "acquisition_window_plan" in parameters
    assert not {"specimen_id", "experiment_id", "synthetic_animal_id"} & set(
        parameters
    )
    artifact = _make(inputs)
    binding = artifact["acquisition_window_realization"]["source_binding"]
    assert set(binding["upstream_realization_ids"]) == set(
        window.UPSTREAM_REALIZATION_ID_FIELDS
    )
    assert binding["upstream_realization_ids"]["v2_plane_realization_id"] == (
        "v2-plane-id"
    )
    assert binding["section_processing_receipt"] == _parent_receipt(
        inputs["processed_render"]
    )
    assert artifact["acquisition_window_realization"]["lineage"] == {
        name: artifact["lineage"][name] for name in window.LINEAGE_FIELDS
    }
    assert "crop" not in artifact["engineering_priors"]
    assert artifact["engineering_priors"]["acquisition_window"][
        "parent_conditioning"
    ].startswith("none")
    plan = _plan(inputs)
    changed = copy.deepcopy(plan)
    changed["content_scale"] += 0.01
    with pytest.raises(ValueError, match="does not replay"):
        _make(inputs, changed)
    wrong_lineage = _plan(inputs, sample_index=4)
    with pytest.raises(ValueError, match="split, sample"):
        _make(inputs, wrong_lineage)
    independent_window_rng = window.sample_acquisition_window_plan_v3(
        root_seed="0x57494e444f570304", split="train", sample_index=3
    )
    independent = _make(inputs, independent_window_rng)
    assert (
        independent["acquisition_window_realization"]["window_plan"]["provenance"][
            "root_seed_uint64"
        ]
        != independent["rng_provenance"]["root_seed_uint64"]
    )


def test_parent_is_verified_before_the_one_window_application(inputs, monkeypatch):
    state = {"verified": False, "applications": 0}
    original_verify = observation._verify_section_processing_render_with_mapper_v2
    original_apply = window.apply_acquisition_window_v3

    def verify(*args, **kwargs):
        original_verify(*args, **kwargs)
        state["verified"] = True

    def apply(*args, **kwargs):
        assert state["verified"]
        state["applications"] += 1
        optical = kwargs["optical_slab_support_mass"]
        assert np.any((optical > 0.0) & (optical < 1.0))
        fov = kwargs["global_reference_fov_uv_um"]
        design = kwargs["design_quicknii_ouv"]
        assert fov[0] == fov[1]
        assert np.allclose(
            fov,
            [
                np.linalg.norm(design[1]) * 255.0 / 256.0,
                np.linalg.norm(design[2]) * 255.0 / 256.0,
            ],
            rtol=1.0e-12,
            atol=1.0e-9,
        )
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(observation, "_verify_section_processing_render_with_mapper_v2", verify)
    monkeypatch.setattr(observation.acquisition_window, "apply_acquisition_window_v3", apply)
    _make(inputs)
    assert state == {"verified": True, "applications": 1}


def test_authenticated_parent_token_fans_out_without_heavy_parent_replay(
    inputs, monkeypatch
):
    token = observation.authenticate_observation_parent_v3(
        inputs["processed_render"],
        inputs["subject_slab_render"],
        inputs["section_processing_plan"],
        inputs["prepared_context"],
        inputs["precursor"],
        subject_plan=BASE["subject_plan"],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared descendants must not replay the heavy parent")

    monkeypatch.setattr(
        observation, "_verify_section_processing_render_with_mapper_v2", forbidden
    )
    first = _make(inputs, authenticated_parent_v3=token)
    second = _make(
        inputs,
        authenticated_parent_v3=token,
        observation_index=BASE["observation_index"] + 1,
    )
    assert first["parent_authentication_v3"] == token
    assert second["parent_authentication_v3"] == token
    assert first["rng_provenance"]["observation_index"] == BASE["observation_index"]
    assert second["rng_provenance"]["observation_index"] == (
        BASE["observation_index"] + 1
    )
    changed = copy.deepcopy(token)
    changed["section_processing_render_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="authentication receipt changed"):
        _make(inputs, authenticated_parent_v3=changed)


def test_authenticated_lineage_is_exposed_but_label_changes_do_not_change_rng(inputs):
    plan = _plan(inputs)
    baseline = _make(inputs, plan)
    lineage = baseline["lineage"]
    assert lineage["animal_id"] == "animal-007"
    assert lineage["specimen_id"] == "specimen-007-A"
    assert lineage["experiment_id"] == "experiment-2026-007"
    assert lineage["synthetic_animal_id"] == "synthetic-animal-007"
    assert all(
        key not in repr(baseline["rng_sources"])
        for key in (
            "animal-007",
            "specimen-007-A",
            "experiment-2026-007",
            "synthetic-animal-007",
        )
    )
    renamed = copy.deepcopy(inputs)
    renamed["precursor"]["provenance"].update(
        {
            "animal_id": "animal-renamed",
            "specimen_id": "specimen-renamed",
            "experiment_id": "experiment-renamed",
        }
    )
    renamed["section_processing_plan"]["provenance"]["animal_id"] = "animal-renamed"
    renamed["subject_slab_render"]["synthetic_animal_id"] = "synthetic-renamed"
    renamed_subject = {"synthetic_animal_id": "synthetic-renamed"}
    renamed_artifact = _make(
        renamed,
        plan,
        animal_id="animal-renamed",
        subject_plan=renamed_subject,
    )
    assert renamed_artifact["rng_sources"] == baseline["rng_sources"]
    assert renamed_artifact["array_receipts"] == baseline["array_receipts"]
    assert renamed_artifact["lineage"] != baseline["lineage"]
    assert (
        renamed_artifact["acquired_observation_id"]
        != baseline["acquired_observation_id"]
    )
    assert renamed_artifact["observation_bundle_id"] != baseline["observation_bundle_id"]


def test_damage_is_sampled_on_full_parent_then_transformed_with_all_supervision(
    inputs, monkeypatch
):
    selected = None
    for observation_index in range(64):
        plan = _plan(inputs)
        artifact = _make(inputs, plan, observation_index=observation_index)
        if any(
            event["category"] == "physical-loss"
            for event in artifact["parent_damage_geometry"]["events"]
        ):
            selected = artifact
            break
    assert selected is not None
    captured = {}
    original_apply = observation._apply_verified_window

    def apply(plan, source_arrays, array_roles, **kwargs):
        captured.update({name: np.array(value, copy=True) for name, value in source_arrays.items()})
        return original_apply(plan, source_arrays, array_roles, **kwargs)

    monkeypatch.setattr(observation, "_apply_verified_window", apply)
    selected = _make(inputs, _plan(inputs), observation_index=observation_index)
    geometry = selected["parent_damage_geometry"]
    parent_shape = inputs["processed_render"]["raster"]["scalar"].shape
    assert geometry["sampling_domain"].startswith("full authenticated")
    assert tuple(geometry["parent_shape_h_w"]) == parent_shape
    assert geometry["parent_tissue_pixel_count"] == int(
        inputs["processed_render"]["raster"]["slab_observable_support_mask"].sum()
    )
    for event in geometry["events"]:
        yy, xx = event["center_y_x"]
        assert 0 <= yy < parent_shape[0] and 0 <= xx < parent_shape[1]
    transformed = selected["acquisition_window_realization"]["arrays"]
    arrays = selected["arrays"]
    parent_physical = captured["parent_physical_loss_mask"]
    assert parent_physical.any()
    assert not captured["source_tissue_mask"][parent_physical].any()
    assert not captured["source_scalar_float32"][parent_physical].view(np.uint32).any()
    assert not captured["source_label_int64"][parent_physical].any()
    assert not captured["source_correspondence_mask"][parent_physical].any()
    assert not captured["source_dense_valid_mask"][parent_physical].any()
    assert not captured["source_dense_weight_float32"][parent_physical].view(np.uint32).any()
    assert captured["source_abstention_mask"][parent_physical].all()
    assert np.array_equal(
        arrays["damage_union_mask"],
        arrays["physical_loss_mask"]
        | arrays["occlusion_mask"]
        | arrays["appearance_artifact_mask"],
    )
    assert np.array_equal(
        arrays["physical_loss_mask"],
        transformed["parent_physical_loss_mask"]
        & arrays["parent_sampling_domain_mask"],
    )
    assert not np.any(
        arrays["physical_loss_mask"] & arrays["source_tissue_ground_truth_mask"]
    )
    assert selected["parameters"]["appearance"]["synthesis_domain"].startswith(
        "fixed acquisition canvas"
    )


def test_fixed_canvas_validity_and_optional_brush_descendants_are_coherent(inputs):
    artifact = _make(inputs)
    assert "arbitrary_plane_acquisition_v2.py" in artifact[
        "implementation_source_sha256"
    ]
    arrays = artifact["arrays"]
    assert all(array.shape[:2] == (192, 256) for array in arrays.values())
    dense_valid = arrays["processed_dense_coordinate_valid_mask"]
    mapped = arrays["processed_mapped_ccf_physical_coordinates_canvas_float64"]
    weight = arrays["source_dense_correspondence_weight_float32"]
    assert np.isfinite(mapped).all()
    assert not mapped[~dense_valid].any()
    assert not weight[~dense_valid].view(np.uint32).any()
    assert not np.any(
        dense_valid & ~arrays["parent_sampling_domain_mask"]
    )
    raw = arrays["raw_acquired_image_float32"]
    accurate = artifact["descendants"]["smart-brush-accurate"]["arrays"]
    imperfect = artifact["descendants"]["smart-brush-imperfect"]["arrays"]
    absent = artifact["descendants"]["smart-brush-absent"]["arrays"]
    assert np.array_equal(
        accurate["selected_input_mask"], arrays["observable_footprint_mask"]
    )
    assert not accurate["model_input_image_float32"][
        ~accurate["selected_input_mask"]
    ].view(np.uint32).any()
    assert np.array_equal(absent["model_input_image_float32"], raw)
    raw_descendant = artifact["descendants"]["raw"]
    absent_descendant = artifact["descendants"]["smart-brush-absent"]
    policy = artifact["descendant_sampling_policy"]
    assert policy["canonical_trainable_raw_background_mode"] == "smart-brush-absent"
    assert policy["raw_background_equivalent_modes"] == [
        "raw",
        "smart-brush-absent",
    ]
    assert raw_descendant["trainable"] is False
    assert absent_descendant["trainable"] is True
    assert sum(
        artifact["descendants"][mode]["trainable"]
        for mode in policy["raw_background_equivalent_modes"]
    ) == 1
    assert np.array_equal(
        raw_descendant["arrays"]["model_input_image_float32"],
        absent_descendant["arrays"]["model_input_image_float32"],
    )
    assert np.array_equal(
        imperfect["brush_mask_error_mask"],
        imperfect["selected_input_mask"] ^ arrays["observable_footprint_mask"],
    )
    deformation_valid = arrays["truth_section_deformation_valid_mask"]
    canvas_y, canvas_x = np.indices(deformation_valid.shape, dtype=np.float64)
    canvas_identity = np.stack((canvas_y, canvas_x), axis=-1)
    assert np.allclose(
        arrays["truth_section_pullback_map_yx_px_float64"][deformation_valid],
        canvas_identity[deformation_valid],
        rtol=0.0,
        atol=1.0e-10,
    )
    assert not arrays[
        "truth_section_pullback_stationary_velocity_yx_px_float64"
    ].view(np.uint64).any()


def test_pullback_velocity_sign_yx_order_and_window_scaling(inputs, monkeypatch):
    deformed = copy.deepcopy(inputs)
    deformed["section_processing_plan"]["resolved_config"][
        "deformation_mode"
    ] = "bspline-svf"
    deformed["processed_render"]["state"]["source_index_yx"] = (
        deformed["processed_render"]["state"]["source_index_yx"]
        + np.array((-2.0 / 4.0, -6.0 / 8.0), dtype=np.float64)
    )
    _refresh(deformed)

    evaluated = {}

    def accepted_field(plan):
        def field(points, return_gradient=False):
            evaluated["points"] = np.array(points, copy=True)
            value = np.empty_like(points, dtype=np.float64)
            value[..., 0] = 2.0
            value[..., 1] = 6.0
            return value

        return field

    monkeypatch.setattr(observation.section_processing, "_accepted_field", accepted_field)
    full_inside_plan = window.sample_acquisition_window_plan_v3(
        root_seed=4,
        split="train",
        sample_index=3,
    )
    assert window._sampling_grid(full_inside_plan)[2].all()
    artifact = _make(deformed, full_inside_plan)
    assert np.array_equal(
        evaluated["points"],
        deformed["processed_render"]["state"]["processed_pixel_centres_yx_um"],
    )
    arrays = artifact["arrays"]
    valid = arrays["truth_section_deformation_valid_mask"]
    inverse_linear = np.asarray(
        artifact["acquisition_window_realization"]["window_plan"][
            "parent_to_canvas_affine_float64"
        ],
        dtype=np.float64,
    )[:2, :2]
    parent_pullback_yx = np.array([-2.0 / 4.0, -6.0 / 8.0])
    expected_xy = inverse_linear @ parent_pullback_yx[::-1]
    expected_yx = expected_xy[::-1]
    assert valid.any()
    gauge = artifact["deformation_pose_gauge"]
    removed = np.asarray(
        gauge["removed_affine_coefficients_yx_float64"], dtype=np.float64
    )
    nested = artifact["acquisition_window_realization"]["arrays"]
    windowed_velocity = nested[
        "truth_section_pullback_stationary_velocity_yx_px_float64"
    ]
    windowed_velocity_valid = nested[
        "truth_section_pullback_stationary_velocity_yx_px_float64"
        + window.VALIDITY_SUFFIX
    ]
    assert np.allclose(
        windowed_velocity[windowed_velocity_valid],
        expected_yx,
        rtol=0.0,
        atol=1.0e-12,
    )
    expected_residual, expected_removed, _ = (
        observation.deformation_gauge.uniform_canvas_affine_projection_yx_v3(
            windowed_velocity
        )
    )
    assert np.allclose(removed, expected_removed, rtol=0.0, atol=1.0e-12)
    assert np.allclose(
        arrays["truth_section_pullback_stationary_velocity_yx_px_float64"],
        expected_residual,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert gauge["projection_weighting"].startswith("fixed uniform full canvas")


def test_bilinear_boundary_invalidity_is_abstained_but_gauge_field_is_full_canvas(inputs):
    boundary = copy.deepcopy(inputs)
    boundary["processed_render"]["state"]["bilinear_domain_valid_mask"][:, -8:] = False
    _refresh(boundary)
    artifact = _make(boundary)
    nested = artifact["acquisition_window_realization"]["arrays"]
    scalar_valid = nested[
        "source_scalar_float32" + window.VALIDITY_SUFFIX
    ]
    scalar_abstention = nested[
        "source_scalar_float32" + window.ABSTENTION_SUFFIX
    ]
    assert (~scalar_valid).any()
    assert np.array_equal(scalar_abstention, ~scalar_valid)
    assert not artifact["arrays"]["source_scalar_canvas_float32"][
        ~scalar_valid
    ].view(np.uint32).any()
    assert artifact["arrays"]["source_dense_correspondence_abstention_mask"][
        ~scalar_valid
    ].all()
    assert not artifact["arrays"]["valid_correspondence_mask"][~scalar_valid].any()
    deformation_valid = artifact["arrays"]["truth_section_deformation_valid_mask"]
    assert not deformation_valid[~scalar_valid].any()
    target_map = artifact["arrays"]["truth_section_pullback_map_yx_px_float64"]
    target_velocity = artifact["arrays"][
        "truth_section_pullback_stationary_velocity_yx_px_float64"
    ]
    assert np.isfinite(target_map).all() and np.isfinite(target_velocity).all()
    assert np.any(target_map[~deformation_valid] != 0.0)
    assert (
        artifact["deformation_pose_gauge"]["diagnostics"][
            "postprojection_affine_coefficient_max_abs"
        ]
        < 1.0e-12
    )
    nested = artifact["acquisition_window_realization"]["arrays"]
    for name in (
        "truth_section_pullback_map_yx_px_float64",
        "truth_section_pullback_stationary_velocity_yx_px_float64",
    ):
        assert np.array_equal(
            nested[name + window.ABSTENTION_SUFFIX],
            ~nested[name + window.VALIDITY_SUFFIX],
        )


def test_empty_view_is_retained_with_pose_abstention_and_background(inputs):
    sparse = copy.deepcopy(inputs)
    render = sparse["processed_render"]
    shape = render["raster"]["scalar"].shape
    tissue = np.zeros(shape, dtype=bool)
    tissue[0, 0] = True
    render["raster"]["centre_plane_support_mask"] = tissue.copy()
    render["raster"]["slab_brain_occupancy"] = tissue.astype(np.float32)
    render["raster"]["slab_observable_support_mask"] = tissue.copy()
    render["raster"]["slab_modal_annotation"] = tissue.astype(np.int64)
    weight = np.zeros(shape, dtype=np.float32)
    weight[0, 0] = np.float32(1)
    render["raster"]["slab_supervision_weight_or_abstention"] = {
        "dense_correspondence_weight": weight,
        "abstention_mask": ~tissue,
    }
    render["state"]["dense_coordinate_valid_mask"] = tissue.copy()
    mapped = np.zeros(shape + (3,), dtype=np.float64)
    mapped[:] = np.nan
    mapped[0, 0] = [100.0, 200.0, 300.0]
    render["mapped_ccf_physical_coordinates_ap_dv_ml_um"] = mapped
    _refresh(sparse)
    selected = None
    for observation_index in range(64):
        plan = _plan(sparse)
        artifact = _make(sparse, plan, observation_index=observation_index)
        if artifact["pose_supervision"]["empty_canvas_tissue"]:
            selected = artifact
            break
    assert selected is not None
    assert selected["pose_supervision"]["pose_abstention"] is True
    assert selected["pose_supervision"]["empty_canvas_tissue"] is True
    assert selected["pose_supervision"]["pose_abstention_reason"].startswith(
        "empty sampled tissue"
    )
    assert selected["parameters"]["appearance"]["normalization"]["method"] == (
        "empty-window-no-tissue"
    )
    assert selected["arrays"]["acquired_background_float32"].any()
    assert np.array_equal(
        selected["arrays"]["raw_acquired_image_float32"],
        selected["arrays"]["acquired_background_float32"],
    )
    assert not selected["descendants"]["smart-brush-accurate"]["arrays"][
        "selected_input_mask"
    ].any()


def test_exact_replay_and_tamper_rejection(inputs):
    plan = _plan(inputs)
    artifact = _make(inputs, plan)
    replay = observation.replay_arbitrary_plane_observation_v3(
        artifact,
        inputs["processed_render"],
        inputs["subject_slab_render"],
        inputs["section_processing_plan"],
        inputs["prepared_context"],
        inputs["precursor"],
        plan,
        **BASE,
    )
    assert observation.observation_bundle_receipt_v3(artifact) == (
        observation.observation_bundle_receipt_v3(replay)
    )
    _verify(artifact, inputs, plan)
    changed = copy.deepcopy(artifact)
    changed["arrays"]["raw_acquired_image_float32"][0, 0] += np.float32(0.1)
    with pytest.raises(ValueError, match="receipt|arrays"):
        _verify(changed, inputs, plan)
    changed = copy.deepcopy(artifact)
    changed["arrays"]["truth_section_pullback_map_yx_px_float64"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="receipt|arrays"):
        _verify(changed, inputs, plan)
    changed = copy.deepcopy(artifact)
    changed["lineage"]["specimen_id"] = "wrong-specimen"
    with pytest.raises(ValueError, match="receipt"):
        _verify(changed, inputs, plan)
    changed = copy.deepcopy(artifact)
    changed["acquisition_window_realization"]["arrays"][
        "source_scalar_float32"
    ][0, 0] += np.float32(1)
    with pytest.raises(ValueError, match="nested"):
        _verify(changed, inputs, plan)
    changed = copy.deepcopy(artifact)
    first_stream = next(iter(changed["rng_sources"]["damage"].values()))
    first_stream["seed_uint64"] = "0x0000000000000000"
    with pytest.raises(ValueError, match="receipt"):
        _verify(changed, inputs, plan)


def test_v3_rng_is_exact_acquisition_seed_parity_and_ignores_numeric_lineage():
    provenance = {
        "root_seed_uint64": ROOT_SEED,
        "split": "train",
        "sample_index": 3,
        "split_index": 999,
        "animal_index": 888,
        "section_index": 777,
        "observation_index": 666,
    }
    receipts = {}
    actual = observation._rng(
        provenance, "appearance", "seed-parity", receipts, attempt=2
    )
    seed = acquisition.derive_v2_field_seed(
        ROOT_SEED, "train", 3, "appearance", "seed-parity", 2
    )
    expected = np.random.Generator(np.random.PCG64DXSM(seed))
    assert actual.random() == expected.random()
    receipt = receipts["appearance/seed-parity/attempt-2"]
    assert receipt["derivation_tuple"] == [
        acquisition.V2_RNG_DOMAIN,
        acquisition.V2_SCHEMA,
        "train",
        ROOT_SEED.removeprefix("0x"),
        "3",
        "appearance",
        "seed-parity",
        "2",
    ]
    assert receipt["tuple_encoding"].startswith("uint32 big-endian")
    assert receipt["person"] == "AP-ACQ-V2"
    assert receipt["seed_endian"] == "unsigned big-endian uint64"
    changed = {**provenance, "animal_index": 1, "observation_index": 2}
    assert observation._rng(
        changed, "appearance", "seed-parity", {}, attempt=2
    ).random() == (
        expected := np.random.Generator(np.random.PCG64DXSM(seed))
    ).random()


def test_forced_anti_shortcut_returns_levels_toward_global_mean(monkeypatch):
    original_rng = observation._rng

    class FixedEnable:
        def random(self):
            return 0.1

    class FixedFraction:
        def uniform(self, *interval):
            assert interval == (0.5, 1.0)
            return 0.75

    def forced_rng(provenance, stage, field, receipts, attempt=0):
        original = original_rng(provenance, stage, field, receipts, attempt)
        if field == "label-boundary-anti-shortcut-enable":
            return FixedEnable()
        if field == "label-boundary-return-fraction":
            return FixedFraction()
        return original

    monkeypatch.setattr(observation, "_rng", forced_rng)
    scalar = np.arange(36, dtype=np.float32).reshape(6, 6)
    tissue = np.ones((6, 6), dtype=bool)
    labels = np.where(np.indices((6, 6))[1] < 3, 11, 23).astype(np.int64)
    provenance = {
        "root_seed_uint64": ROOT_SEED,
        "split": "train",
        "sample_index": 3,
    }
    receipts = {}
    _, parameters = observation._appearance(
        scalar,
        labels,
        tissue,
        "brightfield-nissl-like",
        provenance,
        receipts,
        observation._engineering_priors(),
    )
    conditioning = parameters["label_conditioning"]
    sampled = np.asarray(conditioning["sampled_region_levels"])
    effective = np.asarray(conditioning["effective_region_levels"])
    mean = conditioning["global_tissue_mean"]
    assert conditioning["anti_boundary_shortcut_return_to_global_mean"] is True
    assert conditioning["sampled_return_fraction"] == 0.75
    assert conditioning["effective_return_fraction"] == 0.75
    assert np.allclose(effective, 0.25 * sampled + 0.75 * mean)
    assert {
        "appearance/label-boundary-anti-shortcut-enable/attempt-0",
        "appearance/label-boundary-return-fraction/attempt-0",
        "appearance/label-region-levels/attempt-0",
    } <= set(receipts)


def test_internal_holes_are_genuinely_interior_component_aware_and_tiny_safe():
    tissue = np.ones((31, 31), dtype=bool)
    rngs = [
        np.random.Generator(np.random.PCG64DXSM(seed)) for seed in (11, 12, 13)
    ]
    mask, parameters = observation._internal_hole_mask(
        tissue,
        tissue.copy(),
        3,
        rngs,
        [0.05, 0.08],
        200,
    )
    interior = observation.scipy.ndimage.binary_erosion(tissue, iterations=1)
    assert mask.any()
    assert not np.any(mask & ~interior)
    assert parameters["requested_component_count"] == 3
    assert parameters["realized_component_count"] == 3
    assert all(component["retained_pixel_count"] > 0 for component in parameters["components"])
    tiny = np.ones((1, 1), dtype=bool)
    tiny_mask, tiny_parameters = observation._internal_hole_mask(
        tiny,
        tiny.copy(),
        3,
        [np.random.Generator(np.random.PCG64DXSM(seed)) for seed in (21, 22, 23)],
        [0.05, 0.08],
        1,
    )
    assert not tiny_mask.any()
    assert tiny_parameters["realized_component_count"] == 0
    assert tiny_parameters["component_support_limited"] is True
    assert tiny_parameters["no_rejection_or_redraw"] is True


def test_damage_named_stream_keys_do_not_depend_on_sampled_outcome():
    tissue = np.ones((31, 31), dtype=bool)
    priors = observation._engineering_priors()
    key_sets = []
    strata = set()
    for sample_index in range(8):
        receipts = {}
        _, parameters = observation._sample_parent_damage(
            tissue,
            {
                "root_seed_uint64": ROOT_SEED,
                "split": "train",
                "sample_index": sample_index,
            },
            receipts,
            priors,
            "brightfield-nissl-like",
        )
        key_sets.append(set(receipts))
        strata.add(parameters["stratum"])
    assert len(strata) > 1
    assert all(keys == key_sets[0] for keys in key_sets[1:])
    assert len([key for key in key_sets[0] if key.endswith("-category/attempt-0")]) == 6


def test_forced_brush_has_zero_to_two_total_named_events_and_iou_is_audit_only(
    monkeypatch,
):
    original_rng = observation._rng

    class FixedCount:
        def integers(self, low, high):
            assert (low, high) == (0, 3)
            return 2

    class FixedKind:
        def __init__(self, kind):
            self.kind = kind

        def choice(self, values):
            assert set(values.tolist()) == {"gap", "island"}
            return self.kind

    def forced_rng(provenance, stage, field, receipts, attempt=0):
        original = original_rng(provenance, stage, field, receipts, attempt)
        if field == "island-or-gap-event-count":
            return FixedCount()
        if field.endswith("event-00-kind"):
            return FixedKind("gap")
        if field.endswith("event-01-kind"):
            return FixedKind("island")
        return original

    monkeypatch.setattr(observation, "_rng", forced_rng)
    y, x = np.indices((64, 64))
    footprint = (y - 32) ** 2 + (x - 32) ** 2 <= 20**2
    provenance = {
        "root_seed_uint64": ROOT_SEED,
        "split": "train",
        "sample_index": 3,
    }
    receipts = {}
    _, parameters = observation._imperfect_brush_mask_v3(
        footprint, provenance, receipts, observation._engineering_priors()
    )
    assert parameters["requested_island_or_gap_event_count"] == 2
    active = [event for event in parameters["events"] if event["active"]]
    assert [event["kind"] for event in active] == ["gap", "island"]
    assert parameters["fixed_named_event_stream_slots"] == 2
    assert 0 <= parameters["realized_island_or_gap_event_count"] <= 2
    assert parameters["quality_iou_role"] == "audit-statistic-only"
    assert parameters["iou_acceptance_rejection_or_redraw"] is False
    assert parameters["superseded_v1_accepted_iou_interval"] == [0.70, 0.98]
    assert {
        "brush/island-or-gap-event-count/attempt-0",
        "brush/island-or-gap-event-00-kind/attempt-0",
        "brush/island-or-gap-event-00-parameters/attempt-0",
        "brush/island-or-gap-event-01-kind/attempt-0",
        "brush/island-or-gap-event-01-parameters/attempt-0",
    } <= set(receipts)
