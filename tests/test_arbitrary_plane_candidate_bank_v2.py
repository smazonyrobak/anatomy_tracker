import copy
from collections import Counter
from collections.abc import Mapping

import numpy as np
import pytest

import test_arbitrary_plane_pose_v2 as pose_tests
import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_candidate_bank_v2 as candidate_v2
import training.arbitrary_plane_pose_v2 as pose_v2
import training.arbitrary_plane_realization_v2 as realization_v2


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return copy.deepcopy(value)


@pytest.fixture(scope="module")
def prepared_context():
    return pose_tests.prepared_context.__wrapped__()


def _final(
    context,
    *,
    horizontal=True,
    vertical=True,
    intersect=True,
    animal_id="candidate-animal",
    changed_target=False,
):
    final = pose_tests._final_realization(context, horizontal, vertical)
    final["provenance"]["animal_id"] = animal_id
    if intersect:
        cropped = np.array(
            [
                [-21.5, 40.0, 136.0],
                [0.0, 0.0, 116.0],
                [0.0, 85.0, 0.0],
            ],
            dtype=np.float64,
        )
        parent_shape = tuple(final["frame_transform"]["parent_shape_h_w"])
        top, left = final["frame_transform"]["top_left_y_x"]
        height, width = final["frame_transform"]["output_shape_h_w"]
        parent = np.array(cropped, copy=True)
        parent[1] *= parent_shape[1] / width
        parent[2] *= parent_shape[0] / height
        parent[0] -= (left / parent_shape[1]) * parent[1]
        parent[0] -= (top / parent_shape[0]) * parent[2]
        cropped, model = realization_v2._crop_and_reflect_ouv(
            parent,
            parent_shape,
            (top, left),
            (height, width),
            horizontal,
            vertical,
        )
        arrays = final["frame_transform"]["arrays"]
        arrays["full_raster_best_fit_physical_ouv_ap_dv_ml_um_float64"] = parent
        arrays["cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64"] = cropped
        arrays["model_raster_physical_ouv_ap_dv_ml_um_float64"] = model

    frame = final["frame_transform"]
    frame["array_receipts"] = {
        name: acquisition._array_receipt(value)
        for name, value in frame["arrays"].items()
    }
    frame["frame_transform_id"] = acquisition._payload_sha256(
        {key: value for key, value in frame.items() if key not in {"arrays", "frame_transform_id"}}
    )
    final["paired_mode_sensitivity_reference"]["frame_transform_id"] = frame[
        "frame_transform_id"
    ]
    provenance = final["provenance"]
    support_receipt = {
        "subject_support_resolution_id": "candidate-resolution-id",
        "plan_identity_payload": {
            "support_resolution_plan_id": "candidate-resolution-plan-id",
            "lineage": {
                "split": provenance["split"],
                "animal_id": animal_id,
                "animal_index": provenance["animal_index"],
                "specimen_id": "candidate-specimen",
                "experiment_id": "candidate-experiment",
            },
            "configuration": {
                "split_index": provenance["split_index"],
                "animal_index": provenance["animal_index"],
                "section_index": provenance["section_index"],
                "plane_stratum": "reference",
            },
        },
        "resolution_identity_payload": {},
    }
    final["upstream_reference"][
        "support_resolution_plan_id"
    ] = "candidate-resolution-plan-id"
    final["upstream_reference"]["live_receipt_bindings"]["support_resolution"] = {
        "receipt_payload": support_receipt,
        "receipt_sha256": acquisition._payload_sha256(support_receipt),
    }
    nominal = realization_v2._quicknii_map(
        frame["arrays"]["model_raster_physical_ouv_ap_dv_ml_um_float64"],
        tuple(frame["output_shape_h_w"]),
    )
    final["factor_truth"]["arrays"][
        "nominal_physical_map_ap_dv_ml_um_float64"
    ] = nominal
    final["factor_truth"]["array_receipts"][
        "nominal_physical_map_ap_dv_ml_um_float64"
    ] = acquisition._array_receipt(nominal)
    if changed_target:
        name = "source_label_ground_truth_crop_int64"
        final["targets"][name] = np.full(
            final["model_input"]["spatial_shape_h_w"], 91, dtype=np.int64
        )
        final["target_array_receipts"][name] = acquisition._array_receipt(
            final["targets"][name]
        )
    final["synthetic_realization_id"] = acquisition._payload_sha256(
        realization_v2._identity_payload(final)
    )
    final["receipt_sha256"] = acquisition._payload_sha256(
        realization_v2.synthetic_realization_receipt_v2(final)
    )
    return final


@pytest.fixture(scope="module")
def accepted(prepared_context):
    final = _final(prepared_context)
    truth = pose_v2.make_arbitrary_plane_pose_truth_v2(final, prepared_context)
    bank = candidate_v2.make_arbitrary_plane_candidate_bank_v2(
        truth, final, prepared_context
    )
    return final, truth, bank


def test_fixed_schedule_exact_truth_reflection_replay_and_provenance(
    prepared_context, accepted
):
    final, truth, bank = accepted
    candidate_v2.verify_arbitrary_plane_candidate_bank_v2(
        bank, truth, final, prepared_context
    )
    counts = Counter(item["candidate_class"] for item in bank["candidates"])
    assert counts == {
        "truth": 1,
        "offset_only": 6,
        "normal_angle_only": 16,
        "roll_only": 6,
        "coupled_local": 5,
        "global_hard_negative": 6,
    }
    truth_candidate = next(
        item for item in bank["candidates"] if item["candidate_class"] == "truth"
    )
    for name in (
        "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64",
        "model_raster_physical_ouv_ap_dv_ml_um_float64",
    ):
        assert np.array_equal(truth_candidate["arrays"][name], truth["arrays"][name])
    expected_model = candidate_v2._reflect_ouv(
        truth_candidate["arrays"][
            "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64"
        ],
        tuple(final["frame_transform"]["output_shape_h_w"]),
        True,
        True,
    )
    assert np.array_equal(
        expected_model,
        truth_candidate["arrays"]["model_raster_physical_ouv_ap_dv_ml_um_float64"],
    )
    assert np.array_equal(
        candidate_v2._physical_raster(
            truth_candidate["arrays"][
                "model_raster_physical_ouv_ap_dv_ml_um_float64"
            ],
            tuple(final["frame_transform"]["output_shape_h_w"]),
        ),
        final["factor_truth"]["arrays"][
            "nominal_physical_map_ap_dv_ml_um_float64"
        ],
    )
    rendered = candidate_v2.render_physical_ouv_annotation_v2(
        prepared_context,
        truth_candidate["arrays"]["model_raster_physical_ouv_ap_dv_ml_um_float64"],
        tuple(final["frame_transform"]["output_shape_h_w"]),
    )
    assert np.array_equal(
        rendered["rendered_annotation_int64"],
        truth_candidate["arrays"]["rendered_annotation_int64"],
    )
    assert bank["truth_evaluability"]["evaluable"] is True
    assert truth_candidate["brain_pixel_count"] > 0
    assert bank["source_lineage"]["specimen_id"] == "candidate-specimen"
    assert bank["source_lineage"]["experiment_id"] == "candidate-experiment"
    assert bank["source_lineage"]["plane_stratum"] == "reference"
    assert bank["model_grid_reference"]["target_arrays_or_receipts_accessed"] is False
    assert bank["scope"]["posterior_mass_claim"] is False
    assert bank["scope"]["semantic_scores_present"] is False
    assert len({
        item["array_receipts"][
            "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64"
        ]["array_sha256"]
        for item in bank["candidates"]
    }) == 40


def test_numeric_streams_ignore_animal_label_and_target_values(
    prepared_context, accepted
):
    _, _, bank = accepted
    changed_final = _final(
        prepared_context,
        animal_id="renamed-candidate-animal",
        changed_target=True,
    )
    changed_truth = pose_v2.make_arbitrary_plane_pose_truth_v2(
        changed_final, prepared_context
    )
    changed_bank = candidate_v2.make_arbitrary_plane_candidate_bank_v2(
        changed_truth, changed_final, prepared_context
    )
    assert changed_bank["canonical_candidate_ids"] == bank["canonical_candidate_ids"]
    assert changed_bank["final_order_canonical_indices"] == bank[
        "final_order_canonical_indices"
    ]
    assert changed_bank["candidate_bank_id"] != bank["candidate_bank_id"]
    base = candidate_v2.derive_candidate_bank_seed_v2(
        3, 1, 2, 3, 4, 5, "offset_only", 0, 0
    )
    assert base != candidate_v2.derive_candidate_bank_seed_v2(
        3, 1, 9, 3, 4, 5, "offset_only", 0, 0
    )
    with pytest.raises(TypeError):
        candidate_v2.derive_candidate_bank_seed_v2(
            3, 1, 2.0, 3, 4, 5, "offset_only", 0, 0
        )
    with pytest.raises(TypeError):
        candidate_v2.derive_candidate_bank_seed_v2(
            3, True, 2, 3, 4, 5, "offset_only", 0, 0
        )


def test_disconnected_interval_union_uses_cumulative_length_measure():
    intervals = np.array([[-2.0, 0.0], [10.0, 16.0]], dtype=np.float64)
    offset, index, fraction = candidate_v2._offset_at_fraction(intervals, 0.5)
    assert (offset, index, fraction) == (12.0, 1, 0.5)
    offset, index, fraction = candidate_v2._offset_at_measure(intervals, 8.5)
    assert (offset, index, fraction) == (-1.5, 0, 0.0625)
    state = candidate_v2._measure_state(intervals, 13.0)
    assert state["input_is_member"] is True
    assert state["measure_um"] == 5.0
    assert state["measure_fraction"] == 0.625


def test_zero_support_truth_is_retained_and_marked_unevaluable(prepared_context):
    final = _final(prepared_context, intersect=False, horizontal=False, vertical=False)
    truth = pose_v2.make_arbitrary_plane_pose_truth_v2(final, prepared_context)
    bank = candidate_v2.make_arbitrary_plane_candidate_bank_v2(
        truth, final, prepared_context
    )
    assert bank["truth_evaluability"]["evaluable"] is False
    truth_candidate = next(
        item for item in bank["candidates"] if item["candidate_class"] == "truth"
    )
    assert truth_candidate["brain_pixel_count"] == 0
    assert truth_candidate["finite_raster_support"] is False
    assert any(
        item["candidate_class"] != "truth" and not item["finite_raster_support"]
        for item in bank["candidates"]
    )
    assert len(bank["candidates"]) == 40


def test_array_and_coherently_rereceipted_lineage_tamper_are_rejected(
    prepared_context, accepted
):
    final, truth, bank = accepted
    tampered = _thaw(bank)
    tampered["candidates"][0]["arrays"]["rendered_annotation_int64"][0, 0] += 1
    with pytest.raises(ValueError):
        candidate_v2.verify_arbitrary_plane_candidate_bank_v2(
            tampered, truth, final, prepared_context
        )

    invalid_final = _thaw(final)
    binding = invalid_final["upstream_reference"]["live_receipt_bindings"][
        "support_resolution"
    ]
    binding["receipt_payload"]["plan_identity_payload"]["configuration"][
        "plane_stratum"
    ] = "ordinary"
    binding["receipt_sha256"] = acquisition._payload_sha256(
        binding["receipt_payload"]
    )
    invalid_final["synthetic_realization_id"] = acquisition._payload_sha256(
        realization_v2._identity_payload(invalid_final)
    )
    invalid_final["receipt_sha256"] = acquisition._payload_sha256(
        realization_v2.synthetic_realization_receipt_v2(invalid_final)
    )
    invalid_truth = pose_v2.make_arbitrary_plane_pose_truth_v2(
        invalid_final, prepared_context
    )
    with pytest.raises(ValueError, match="frozen v2 strata"):
        candidate_v2.make_arbitrary_plane_candidate_bank_v2(
            invalid_truth, invalid_final, prepared_context
        )

    tampered = _thaw(bank)
    tampered["source_lineage"]["specimen_id"] = "wrong-specimen"
    tampered["candidate_bank_id"] = acquisition._payload_sha256(
        candidate_v2._bank_identity(tampered)
    )
    tampered["receipt_sha256"] = acquisition._payload_sha256(
        candidate_v2.arbitrary_plane_candidate_bank_receipt_v2(tampered)
    )
    with pytest.raises(ValueError):
        candidate_v2.verify_arbitrary_plane_candidate_bank_v2(
            tampered, truth, final, prepared_context
        )
