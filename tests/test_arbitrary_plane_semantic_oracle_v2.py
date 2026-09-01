from copy import deepcopy
from inspect import signature

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_semantic_oracle_v2 as oracle


def _fixture(
    monkeypatch, pitch=(10.0, 10.0), horizontal=False, vertical=False
):
    shape = (8, 10)
    y, x = np.indices(shape)
    source_target = np.where(x < 5, 11, 29).astype(np.int64)
    source_valid = np.ones(shape, dtype=bool)
    target, valid = source_target.copy(), source_valid.copy()
    if horizontal:
        target, valid = np.flip(target, axis=1), np.flip(valid, axis=1)
    if vertical:
        target, valid = np.flip(target, axis=0), np.flip(valid, axis=0)
    target, valid = np.ascontiguousarray(target), np.ascontiguousarray(valid)
    weight = np.ones(shape, dtype=np.float32)
    pre_ouv = np.asarray(
        [[100.0, 200.0, 300.0], [100.0, 0.0, 0.0], [0.0, 80.0, 0.0]],
        dtype=np.float64,
    )
    model_ouv = oracle._reflected_ouv(
        pre_ouv, shape, horizontal=horizontal, vertical=vertical
    )
    nominal = oracle._physical_raster(model_ouv, shape)
    allen = np.zeros((*shape, 3), dtype=np.float32)
    physical_receipt = acquisition._array_receipt(nominal)
    allen_receipt = acquisition._array_receipt(allen)
    truth_index = 13
    candidates = []
    for index in range(40):
        is_truth = index == truth_index
        labels = target.copy() if is_truth else np.zeros(shape, dtype=np.int64)
        brain = labels != 0
        candidate_ouv = model_ouv.copy()
        if not is_truth:
            candidate_ouv[0, 2] += 25.0 * (index + 1)
        arrays = {
            "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64": pre_ouv.copy()
            if is_truth
            else candidate_ouv,
            "model_raster_physical_ouv_ap_dv_ml_um_float64": candidate_ouv,
            "rendered_annotation_int64": labels,
            "brain_mask": brain,
        }
        candidates.append(
            {
                "candidate_id": f"candidate-{index:02d}",
                "candidate_class": "truth" if is_truth else "offset_only",
                "slot": 0 if is_truth else index,
                "pose": {
                    "actual_normal_ap_dv_ml": [0.0, 0.0, 1.0],
                    "actual_signed_offset_um": 0.0 if is_truth else 25.0 * (index + 1),
                    "roll_delta_rad_from_truth": 0.0,
                },
                "render_contract": {
                    "physical_coordinate_raster_receipt": physical_receipt,
                    "allen_index_coordinate_raster_float32_receipt": allen_receipt,
                    "target_overlap_used_for_construction_or_acceptance": False,
                },
                "brain_pixel_count": int(brain.sum()),
                "finite_raster_support": bool(brain.any()),
                "infinite_plane_support_envelope": {
                    "plane_intersects_support_envelope": True,
                },
                "arrays": arrays,
                "array_receipts": {
                    name: acquisition._array_receipt(value)
                    for name, value in arrays.items()
                },
            }
        )
    ordered_ids = [candidate["candidate_id"] for candidate in candidates]
    source_lineage = {
        "support_resolution_plan_id": "support-plan",
        "support_resolution_receipt_sha256": "1" * 64,
        "split": "development",
        "split_index": 3,
        "animal_id": "animal-five",
        "animal_index": 5,
        "specimen_id": "specimen-five-a",
        "experiment_id": "experiment-five",
        "section_index": 41,
        "plane_stratum": "general_oblique",
    }
    bank = {
        "candidate_bank_id": "2" * 64,
        "receipt_sha256": "3" * 64,
        "ordered_candidate_ids": ordered_ids,
        "candidates": candidates,
        "source_lineage": source_lineage,
        "reflection_state": {
            "horizontal_reflection": horizontal,
            "vertical_reflection": vertical,
        },
        "model_grid_reference": {
            "output_shape_h_w": list(shape),
            "shape_sources": [
                "frame_transform.output_shape_h_w",
                "model_input.spatial_shape_h_w",
            ],
            "target_arrays_or_receipts_accessed": False,
        },
        "truth_evaluability": {
            "independent_truth_brain_pixel_count": int(target.size),
            "evaluable": True,
            "zero_support_policy": "mark unevaluable; never redraw or inspect target overlap",
        },
    }
    plan_payload = {
        "domain": "anatomy-tracker.observation-plan/v2",
        "fixture": "semantic-oracle-v2",
    }
    plan_id = acquisition._payload_sha256(plan_payload)
    crop_payload = {
        "domain": "anatomy-tracker.observation-crop-window/v2",
        "observation_plan_id": plan_id,
        "crop_window": {"processed_pixel_pitch_y_x_um": list(pitch)},
    }
    crop_id = acquisition._payload_sha256(crop_payload)
    acquired_payload = {
        "domain": "anatomy-tracker.acquired-observation/v2",
        "observation_plan_id": plan_id,
        "crop_window_id": crop_id,
        "parameters": {},
        "acquisition_rng_sources": {},
        "array_receipts": {
            "source_label_ground_truth_crop_int64": acquisition._array_receipt(
                source_target
            ),
            "valid_correspondence_mask": acquisition._array_receipt(source_valid),
        },
    }
    acquired_id = acquisition._payload_sha256(acquired_payload)
    descendant_payloads = {
        mode: {
            "domain": "anatomy-tracker.observation-descendant/v2",
            "schema_version": "fixture/v2",
            "mode": mode,
            "trainable": mode != "raw",
            "brush_available": mode in {
                "smart-brush-accurate",
                "smart-brush-imperfect",
            },
            "acquired_observation_id": acquired_id,
            "background_policy": "fixture",
            "parameters": {},
            "array_receipts": {},
        }
        for mode in oracle._OBSERVATION_DESCENDANT_MODES
    }
    descendant_ids = {
        mode: acquisition._payload_sha256(payload)
        for mode, payload in descendant_payloads.items()
    }
    bundle_payload = {
        "domain": "anatomy-tracker.observation-bundle/v2",
        "observation_plan_id": plan_id,
        "acquired_observation_id": acquired_id,
        "descendant_ids": descendant_ids,
        "brush_rng_sources": {},
    }
    bundle_id = acquisition._payload_sha256(bundle_payload)
    observation_payload = {
        "observation_plan_id": plan_id,
        "crop_window_id": crop_id,
        "acquired_observation_id": acquired_id,
        "observation_bundle_id": bundle_id,
        "plan_identity_payload": plan_payload,
        "crop_identity_payload": crop_payload,
        "acquired_identity_payload": acquired_payload,
        "bundle_identity_payload": bundle_payload,
        "descendant_identity_payloads": descendant_payloads,
    }
    observation_binding = {
        "receipt_payload": observation_payload,
        "receipt_sha256": acquisition._payload_sha256(observation_payload),
    }
    target_receipts = {
        "source_label_ground_truth_crop_int64": acquisition._array_receipt(target),
        "valid_correspondence_mask": acquisition._array_receipt(valid),
        "valid_correspondence_weight_float32": acquisition._array_receipt(weight),
    }
    final = {
        "targets": {
            "source_label_ground_truth_crop_int64": target,
            "valid_correspondence_mask": valid,
            "valid_correspondence_weight_float32": weight,
        },
        "target_array_receipts": target_receipts,
        "frame_transform": {
            "crop_window_id": crop_id,
            "output_shape_h_w": list(shape),
            "horizontal_reflection": horizontal,
            "vertical_reflection": vertical,
            "frame_transform_id": "frame-transform",
            "arrays": {
                "model_raster_physical_ouv_ap_dv_ml_um_float64": model_ouv,
            },
        },
        "factor_truth": {
            "arrays": {
                "nominal_physical_map_ap_dv_ml_um_float64": nominal,
            },
            "array_receipts": {
                "nominal_physical_map_ap_dv_ml_um_float64": acquisition._array_receipt(
                    nominal
                ),
            },
        },
        "model_input": {
            "spatial_shape_h_w": list(shape),
            "channels_float32": np.zeros((3, *shape), dtype=np.float32),
        },
        "upstream_reference": {
            "observation_bundle_id": bundle_id,
            "observation_receipt_sha256": observation_binding["receipt_sha256"],
            "acquired_observation_id": acquired_id,
            "crop_window_id": crop_id,
            "live_receipt_bindings": {
                "observation_bundle": observation_binding,
            }
        },
        "provenance": {
            "split": "development",
            "split_index": 3,
            "animal_id": "animal-five",
            "animal_index": 5,
            "section_index": 41,
            "observation_index": 7,
            "realization_index": 11,
        },
        "synthetic_realization_id": "4" * 64,
        "receipt_sha256": "5" * 64,
        "training_row_id": "6" * 64,
    }
    truth_plane = np.asarray(
        [0.0, 0.0, 1.0, 0.0], dtype=np.float64
    )
    truth_plane.setflags(write=False)
    pose = {
        "finite_plane_pose_truth_id": "7" * 64,
        "receipt_sha256": "8" * 64,
        "reflection_state": {
            "horizontal_reflection": horizontal,
            "vertical_reflection": vertical,
        },
        "arrays": {
            "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64": pre_ouv,
            "model_raster_physical_ouv_ap_dv_ml_um_float64": model_ouv,
            "actual_plane_normal_and_signed_offset_um_float64": truth_plane,
        },
    }
    context = {
        "v2_context_sha256": "9" * 64,
        "receipt": {
            "support_index_sha256": "a" * 64,
            "annotation_array_sha256": "b" * 64,
        },
    }

    monkeypatch.setattr(
        oracle.candidate_v2,
        "verify_arbitrary_plane_candidate_bank_v2",
        lambda *args: None,
    )

    def rerender(prepared_context, ouv, output_shape):
        assert prepared_context is context
        assert np.array_equal(ouv, model_ouv)
        assert tuple(output_shape) == shape
        return {
            "rendered_annotation_int64": target.copy(),
            "brain_mask": np.ones(shape, dtype=bool),
            "physical_coordinate_raster_receipt": physical_receipt,
            "allen_index_coordinate_raster_float32_receipt": allen_receipt,
            "annotation_array_sha256": context["receipt"]["annotation_array_sha256"],
            "sampling_contract": "O+(x/W)U+(y/H)V",
        }

    monkeypatch.setattr(
        oracle.candidate_v2, "render_physical_ouv_annotation_v2", rerender
    )
    return bank, pose, final, context, truth_index


def test_primary_result_preserves_raw_ranking_controls_and_source_lineage(monkeypatch):
    bank, pose, final, context, truth_index = _fixture(monkeypatch)
    result = oracle.make_arbitrary_plane_semantic_oracle_result_v2(
        bank, pose, final, context
    )

    assert tuple(signature(oracle._score_semantic_arrays_v2).parameters) == (
        "target_labels",
        "candidate_labels",
        "fixed_valid_mask",
        "pixel_pitch_um",
    )
    assert result["candidate_reference"]["truth_candidate_index"] == truth_index
    assert result["ranking"]["top1"] is True
    assert result["ranking"]["top3"] is True
    assert result["ranking"]["coverage_adjusted_top1"] is True
    assert result["ranking"]["true_rank"] == 1
    assert result["selected_pose_error"]["normal_geodesic_angle_deg"] == 0.0
    assert result["selected_pose_error"]["sign_aligned_offset_error_um"] == 0.0
    assert result["selected_pose_error"]["absolute_wrapped_roll_error_deg"] == 0.0
    assert result["selected_pose_error"]["corresponding_point_rms_um"] == 0.0
    assert result["provenance"]["source_lineage"]["specimen_id"] == "specimen-five-a"
    assert result["provenance"]["source_lineage"]["experiment_id"] == "experiment-five"
    assert result["provenance"]["source_lineage"]["plane_stratum"] == "general_oblique"
    assert all(record["passed"] for record in result["exact_controls"].values())
    assert result["scope"]["valid_correspondence_weight_used"] is False
    assert result["scope"]["posterior_or_probability_claim"] is False
    assert not result["scores"]["arrays"]["semantic_score_float64"].flags.writeable


def test_exact_replay_verify_and_raw_score_tamper_rejection(monkeypatch):
    bank, pose, final, context, _ = _fixture(monkeypatch)
    result = oracle.make_arbitrary_plane_semantic_oracle_result_v2(
        bank, pose, final, context
    )
    replayed = oracle.replay_arbitrary_plane_semantic_oracle_result_v2(
        result, bank, pose, final, context
    )
    assert replayed["receipt_sha256"] == result["receipt_sha256"]
    oracle.verify_arbitrary_plane_semantic_oracle_result_v2(
        result, bank, pose, final, context
    )

    changed = dict(result)
    changed_scores = dict(result["scores"])
    changed_arrays = dict(result["scores"]["arrays"])
    changed_semantic = np.array(changed_arrays["semantic_score_float64"], copy=True)
    changed_semantic[0] = 0.5
    changed_arrays["semantic_score_float64"] = changed_semantic
    changed_scores["arrays"] = changed_arrays
    changed["scores"] = changed_scores
    with pytest.raises(ValueError, match="raw score array or receipt"):
        oracle.verify_arbitrary_plane_semantic_oracle_result_v2(
            changed, bank, pose, final, context
        )


def test_forbidden_weight_and_model_channels_do_not_change_scores(monkeypatch):
    bank, pose, final, context, _ = _fixture(monkeypatch)
    first = oracle.make_arbitrary_plane_semantic_oracle_result_v2(
        bank, pose, final, context
    )
    changed = dict(final)
    changed_targets = dict(final["targets"])
    changed_targets["valid_correspondence_weight_float32"] = np.zeros_like(
        changed_targets["valid_correspondence_weight_float32"]
    )
    changed["targets"] = changed_targets
    changed_model_input = dict(final["model_input"])
    changed_model_input["channels_float32"] = np.ones_like(
        changed_model_input["channels_float32"]
    )
    changed["model_input"] = changed_model_input
    second = oracle.make_arbitrary_plane_semantic_oracle_result_v2(
        bank, pose, changed, context
    )

    assert first["receipt_sha256"] == second["receipt_sha256"]
    assert all(
        np.array_equal(first["scores"]["arrays"][name], second["scores"]["arrays"][name])
        for name in oracle._SCORE_ARRAY_KEYS
    )


def test_reflected_targets_are_bound_back_to_pre_reflection_observation(monkeypatch):
    bank, pose, final, context, _ = _fixture(
        monkeypatch, horizontal=True, vertical=True
    )
    result = oracle.make_arbitrary_plane_semantic_oracle_result_v2(
        bank, pose, final, context
    )
    assert result["exact_controls"]["crop_reflection_binding"]["passed"] is True
    assert result["ranking"]["top1"] is True


def test_coherently_rereceipted_and_reidentified_target_is_rejected(monkeypatch):
    bank, pose, final, context, _ = _fixture(monkeypatch)
    changed = dict(final)
    changed_targets = dict(final["targets"])
    changed_labels = np.array(
        changed_targets["source_label_ground_truth_crop_int64"], copy=True
    )
    changed_labels[0, 0] = 47
    changed_targets["source_label_ground_truth_crop_int64"] = changed_labels
    changed["targets"] = changed_targets
    changed_receipts = dict(final["target_array_receipts"])
    changed_receipts["source_label_ground_truth_crop_int64"] = acquisition._array_receipt(
        changed_labels
    )
    changed["target_array_receipts"] = changed_receipts
    changed["synthetic_realization_id"] = "c" * 64
    changed["receipt_sha256"] = "d" * 64

    with pytest.raises(ValueError, match="observation trust root"):
        oracle.make_arbitrary_plane_semantic_oracle_result_v2(
            bank, pose, changed, context
        )


def test_nested_observation_identity_payload_cannot_change_under_stale_ids(monkeypatch):
    bank, pose, final, context, _ = _fixture(monkeypatch)
    changed = deepcopy(final)
    binding = changed["upstream_reference"]["live_receipt_bindings"][
        "observation_bundle"
    ]
    payload = binding["receipt_payload"]
    payload["crop_identity_payload"]["crop_window"][
        "processed_pixel_pitch_y_x_um"
    ] = [12.0, 12.0]
    binding["receipt_sha256"] = acquisition._payload_sha256(payload)
    changed["upstream_reference"]["observation_receipt_sha256"] = binding[
        "receipt_sha256"
    ]

    with pytest.raises(ValueError, match="observation receipt identity changed"):
        oracle.make_arbitrary_plane_semantic_oracle_result_v2(
            bank, pose, changed, context
        )


def test_nested_acquired_target_receipt_cannot_change_under_stale_id(monkeypatch):
    bank, pose, final, context, _ = _fixture(monkeypatch)
    changed = deepcopy(final)
    binding = changed["upstream_reference"]["live_receipt_bindings"][
        "observation_bundle"
    ]
    payload = binding["receipt_payload"]
    payload["acquired_identity_payload"]["array_receipts"][
        "source_label_ground_truth_crop_int64"
    ]["array_sha256"] = "f" * 64
    binding["receipt_sha256"] = acquisition._payload_sha256(payload)
    changed["upstream_reference"]["observation_receipt_sha256"] = binding[
        "receipt_sha256"
    ]

    with pytest.raises(ValueError, match="observation receipt identity changed"):
        oracle.make_arbitrary_plane_semantic_oracle_result_v2(
            bank, pose, changed, context
        )


def test_zero_support_truth_cannot_be_counted_as_top1_or_top3(monkeypatch):
    bank, pose, final, context, truth_index = _fixture(monkeypatch)
    truth = bank["candidates"][truth_index]
    zero_brain = np.zeros_like(truth["arrays"]["brain_mask"])
    truth["arrays"]["brain_mask"] = zero_brain
    truth["array_receipts"]["brain_mask"] = acquisition._array_receipt(zero_brain)
    truth["brain_pixel_count"] = 0
    truth["finite_raster_support"] = False
    bank["truth_evaluability"] = {
        "independent_truth_brain_pixel_count": 0,
        "evaluable": False,
        "zero_support_policy": "mark unevaluable; never redraw or inspect target overlap",
    }
    truth_labels = truth["arrays"]["rendered_annotation_int64"]

    def rerender(prepared_context, ouv, output_shape):
        assert prepared_context is context
        assert tuple(output_shape) == truth_labels.shape
        return {
            "rendered_annotation_int64": truth_labels.copy(),
            "brain_mask": zero_brain.copy(),
            "physical_coordinate_raster_receipt": truth["render_contract"][
                "physical_coordinate_raster_receipt"
            ],
            "allen_index_coordinate_raster_float32_receipt": truth[
                "render_contract"
            ]["allen_index_coordinate_raster_float32_receipt"],
            "annotation_array_sha256": context["receipt"][
                "annotation_array_sha256"
            ],
            "sampling_contract": "O+(x/W)U+(y/H)V",
        }

    monkeypatch.setattr(
        oracle.candidate_v2, "render_physical_ouv_annotation_v2", rerender
    )
    result = oracle.make_arbitrary_plane_semantic_oracle_result_v2(
        bank, pose, final, context
    )

    assert result["ranking"]["raw_top1_before_coverage_policy"] is True
    assert result["ranking"]["raw_top3_before_coverage_policy"] is True
    assert result["ranking"]["top1"] is False
    assert result["ranking"]["top3"] is False
    assert result["ranking"]["coverage_adjusted_top1"] is False
    assert result["ranking"]["coverage_adjusted_top3"] is False


def test_anisotropic_authenticated_pitch_is_rejected(monkeypatch):
    bank, pose, final, context, _ = _fixture(monkeypatch, pitch=(10.0, 11.0))
    with pytest.raises(ValueError, match="authenticated isotropic pixel pitch"):
        oracle.make_arbitrary_plane_semantic_oracle_result_v2(
            bank, pose, final, context
        )
