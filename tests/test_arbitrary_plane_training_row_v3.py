import copy

import numpy as np
import pytest

import test_arbitrary_plane_observation_v3 as observation_fixture
import training.arbitrary_plane_observation_v3 as observation
import training.arbitrary_plane_training_row_v3 as training_row


@pytest.fixture(autouse=True)
def authenticated_parent(monkeypatch):
    monkeypatch.setattr(
        observation,
        "_verify_section_processing_render_with_mapper_v2",
        observation_fixture._fake_verify_parent,
    )
    monkeypatch.setattr(
        observation,
        "section_processing_render_receipt_v2",
        observation_fixture._parent_receipt,
    )
    monkeypatch.setattr(
        observation.section_processing,
        "_accepted_field",
        lambda plan: lambda points, return_gradient=False: np.zeros_like(points),
    )


@pytest.fixture
def source():
    inputs = observation_fixture._inputs()
    return observation_fixture._make(inputs), inputs


def _rows_for_states(artifact):
    rows = {}
    for index in range(128):
        row = training_row.make_arbitrary_plane_training_row_v3(artifact, index)
        rows.setdefault(row["reflection_state"], row)
        if set(rows) == set(training_row.REFLECTION_STATES):
            return rows
    raise AssertionError("both reflection states were not sampled")


def test_one_uniform_trainable_mode_no_vertical_and_paired_group(source):
    artifact, _ = source
    seen = set()
    for index in range(96):
        row = training_row.make_arbitrary_plane_training_row_v3(artifact, index)
        seen.add(row["selected_mode"])
        assert row["selected_mode"] in training_row.TRAINABLE_MODES
        assert row["selected_mode"] != "raw"
        assert row["reflection_state"] in {"none", "horizontal"}
        assert row["reflection_representation_index"] in {0, 1}
        assert set(row["paired_mode_reflected_receipts"]) == set(
            training_row.TRAINABLE_MODES
        )
        assert isinstance(row["paired_view_group_id"], str)
    assert seen == set(training_row.TRAINABLE_MODES)


def test_exact_horizontal_map_vector_ccf_and_ouv_conjugation(source):
    artifact, _ = source
    horizontal = _rows_for_states(artifact)["horizontal"]
    arrays = horizontal["arrays"]
    source_arrays = artifact["arrays"]
    width = arrays["truth_section_pullback_map_yx_px_float64"].shape[1]
    expected_map = source_arrays[
        "truth_section_pullback_map_yx_px_float64"
    ][:, ::-1].copy()
    expected_map[..., 1] = width - 1.0 - expected_map[..., 1]
    expected_velocity = source_arrays[
        "truth_section_pullback_stationary_velocity_yx_px_float64"
    ][:, ::-1].copy()
    expected_velocity[..., 1] *= -1.0
    assert np.array_equal(
        arrays["truth_section_pullback_map_yx_px_float64"], expected_map
    )
    assert np.array_equal(
        arrays["truth_section_pullback_stationary_velocity_yx_px_float64"],
        expected_velocity,
    )
    assert np.array_equal(
        arrays["target_ccf_coordinates_ap_dv_ml_um_float64"],
        source_arrays[
            "processed_mapped_ccf_physical_coordinates_canvas_float64"
        ][:, ::-1],
    )
    canonical = np.asarray(horizontal["canonical_effective_quicknii_ouv_float64"])
    observed = np.asarray(horizontal["observed_effective_quicknii_ouv_float64"])
    assert np.array_equal(observed[1], -canonical[1])
    assert np.array_equal(observed[2], canonical[2])
    assert np.allclose(
        observed[0], canonical[0] + ((width - 1.0) / width) * canonical[1]
    )
    assert horizontal["proper_physical_pose_unchanged"] == canonical.tolist()
    assert horizontal["deformation_pose_gauge_reference"] == (
        training_row.deformation_gauge.deformation_pose_gauge_reference_v3(
            artifact["deformation_pose_gauge"]
        )
    )


def test_channels_preserve_black_exterior_and_raw_absent_semantics(source):
    artifact, _ = source
    found = {}
    for index in range(128):
        row = training_row.make_arbitrary_plane_training_row_v3(artifact, index)
        found.setdefault(row["selected_mode"], row)
        if set(found) == set(training_row.TRAINABLE_MODES):
            break
    for mode, row in found.items():
        horizontal = row["reflection_state"] == "horizontal"
        descendant = artifact["descendants"][mode]
        expected_image = descendant["arrays"]["model_input_image_float32"]
        selected = descendant["arrays"]["selected_input_mask"]
        if horizontal:
            expected_image = expected_image[:, ::-1]
            selected = selected[:, ::-1]
        channels = row["arrays"]["model_input_channels_float32"]
        assert np.array_equal(channels[..., 0], expected_image)
        assert np.all(channels[..., 2] == float(descendant["brush_available"]))
        if descendant["brush_available"]:
            assert not channels[..., 0][~selected].view(np.uint32).any()
        else:
            raw = artifact["arrays"]["raw_acquired_image_float32"]
            if horizontal:
                raw = raw[:, ::-1]
            assert np.array_equal(channels[..., 0], raw)


def test_numeric_rng_ignores_labels_but_identities_bind_lineage(source):
    baseline, inputs = source
    renamed = copy.deepcopy(inputs)
    renamed["precursor"]["provenance"].update(
        {
            "animal_id": "renamed-animal",
            "specimen_id": "renamed-specimen",
            "experiment_id": "renamed-experiment",
        }
    )
    renamed["section_processing_plan"]["provenance"]["animal_id"] = "renamed-animal"
    renamed["subject_slab_render"]["synthetic_animal_id"] = "renamed-synthetic"
    renamed_artifact = observation_fixture._make(
        renamed,
        animal_id="renamed-animal",
        subject_plan={"synthetic_animal_id": "renamed-synthetic"},
    )
    first = training_row.make_arbitrary_plane_training_row_v3(baseline, 11)
    second = training_row.make_arbitrary_plane_training_row_v3(renamed_artifact, 11)
    assert first["selected_mode"] == second["selected_mode"]
    assert first["reflection_state"] == second["reflection_state"]
    assert first["rng_sources"] == second["rng_sources"]
    assert all(
        np.array_equal(first["arrays"][name], second["arrays"][name])
        for name in first["arrays"]
    )
    assert first["training_row_id"] != second["training_row_id"]
    assert first["synthetic_realization_id"] != second["synthetic_realization_id"]
    assert first["reflection_realization_id"] != second["reflection_realization_id"]


def test_exact_replay_and_source_or_row_tamper_rejection(source):
    artifact, _ = source
    row = training_row.make_arbitrary_plane_training_row_v3(artifact, 4)
    replay = training_row.replay_arbitrary_plane_training_row_v3(row, artifact, 4)
    assert training_row.training_row_receipt_v3(row) == (
        training_row.training_row_receipt_v3(replay)
    )
    training_row.verify_arbitrary_plane_training_row_v3(row, artifact, 4)
    changed = copy.deepcopy(row)
    changed["arrays"]["model_input_channels_float32"][0, 0, 0] += np.float32(1)
    with pytest.raises(ValueError, match="replay"):
        training_row.verify_arbitrary_plane_training_row_v3(changed, artifact, 4)
    changed_source = copy.deepcopy(artifact)
    changed_source["arrays"]["raw_acquired_image_float32"][0, 0] += np.float32(1)
    with pytest.raises(ValueError, match="observation"):
        training_row.make_arbitrary_plane_training_row_v3(changed_source, 4)


def test_stage_declares_no_prior_models_or_features(source):
    row = training_row.make_arbitrary_plane_training_row_v3(source[0], 0)
    assert row["prior_model_dependencies"] == []
    assert row["prior_feature_dependencies"] == []
    assert row["prior_pseudolabel_dependencies"] == []
