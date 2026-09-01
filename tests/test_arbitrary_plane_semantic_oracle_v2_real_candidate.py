import numpy as np

import test_arbitrary_plane_candidate_bank_v2 as candidate_tests
import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_candidate_bank_v2 as candidate_v2
import training.arbitrary_plane_observation_v2 as observation_v2
import training.arbitrary_plane_pose_v2 as pose_v2
import training.arbitrary_plane_realization_v2 as realization_v2
import training.arbitrary_plane_semantic_oracle_v2 as oracle_v2


def _observation_receipt(final, source_labels, source_valid):
    frame = final["frame_transform"]
    shape = tuple(frame["output_shape_h_w"])
    plan_payload = {
        "domain": "anatomy-tracker.observation-plan/v2",
        "schema_version": observation_v2.OBSERVATION_V2_SCHEMA,
        "algorithm": observation_v2.OBSERVATION_V2_ALGORITHM,
        "implementation_source_sha256": {"fixture": "1" * 64},
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "runtime_dependencies": {"numpy_version": np.__version__},
        "asset_dependencies": {
            "learned_checkpoint_dependencies": [],
            "pretrained_feature_dependencies": [],
            "previous_model_dependencies": [],
        },
        "upstream_reference": {"fixture": "real-candidate-compatibility"},
        "provenance": acquisition._json_value(final["provenance"]),
        "modality": "brightfield-nissl-like",
        "modality_model": {"fixture": True},
        "engineering_priors": {"fixture": True},
        "disclosure": {"fixture": True},
    }
    plan_id = acquisition._payload_sha256(plan_payload)
    nominal = final["factor_truth"]["arrays"][
        "nominal_physical_map_ap_dv_ml_um_float64"
    ]
    crop_payload = {
        "domain": "anatomy-tracker.observation-crop-window/v2",
        "observation_plan_id": plan_id,
        "crop_window": {
            "parent_shape_h_w": list(frame["parent_shape_h_w"]),
            "top_left_y_x": list(frame["top_left_y_x"]),
            "output_shape_h_w": list(shape),
            "processed_pixel_pitch_y_x_um": [10.0, 10.0],
            "processed_mapped_ccf_coordinate_crop_receipt": acquisition._array_receipt(
                nominal
            ),
        },
    }
    crop_id = acquisition._payload_sha256(crop_payload)
    acquired_payload = {
        "domain": "anatomy-tracker.acquired-observation/v2",
        "observation_plan_id": plan_id,
        "crop_window_id": crop_id,
        "parameters": {"fixture": "deterministic"},
        "acquisition_rng_sources": {},
        "array_receipts": {
            "source_label_ground_truth_crop_int64": acquisition._array_receipt(
                source_labels
            ),
            "valid_correspondence_mask": acquisition._array_receipt(source_valid),
        },
    }
    acquired_id = acquisition._payload_sha256(acquired_payload)
    descendant_payloads = {}
    descendant_ids = {}
    zeros = np.zeros(shape, dtype=np.float32)
    false = np.zeros(shape, dtype=bool)
    for mode in observation_v2.DESCENDANT_MODES:
        payload = {
            "domain": observation_v2.OBSERVATION_DESCENDANT_V2_SCHEMA,
            "schema_version": observation_v2.OBSERVATION_DESCENDANT_V2_SCHEMA,
            "mode": mode,
            "trainable": mode != "raw",
            "brush_available": mode
            in {"smart-brush-accurate", "smart-brush-imperfect"},
            "acquired_observation_id": acquired_id,
            "background_policy": "fixture",
            "parameters": {},
            "array_receipts": {
                "model_input_image_float32": acquisition._array_receipt(zeros),
                "selected_input_mask": acquisition._array_receipt(false),
                "brush_mask_error_mask": acquisition._array_receipt(false),
            },
        }
        descendant_payloads[mode] = payload
        descendant_ids[mode] = acquisition._payload_sha256(payload)
    bundle_payload = {
        "domain": "anatomy-tracker.observation-bundle/v2",
        "observation_plan_id": plan_id,
        "acquired_observation_id": acquired_id,
        "descendant_ids": descendant_ids,
        "brush_rng_sources": {},
    }
    bundle_id = acquisition._payload_sha256(bundle_payload)
    receipt = {
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
    return receipt


def test_real_candidate_bank_horizontal_vertical_oracle_make_replay_verify():
    context = candidate_tests.prepared_context.__wrapped__()
    final = candidate_tests._final(
        context, horizontal=True, vertical=True, intersect=True
    )
    frame = final["frame_transform"]
    shape = tuple(frame["output_shape_h_w"])
    rendered = candidate_v2.render_physical_ouv_annotation_v2(
        context,
        frame["arrays"]["model_raster_physical_ouv_ap_dv_ml_um_float64"],
        shape,
    )
    target = np.array(rendered["rendered_annotation_int64"], copy=True)
    valid = np.ascontiguousarray(target != 0)
    assert valid.any()
    final["targets"]["source_label_ground_truth_crop_int64"] = target
    final["targets"]["valid_correspondence_mask"] = valid
    final["target_array_receipts"][
        "source_label_ground_truth_crop_int64"
    ] = acquisition._array_receipt(target)
    final["target_array_receipts"][
        "valid_correspondence_mask"
    ] = acquisition._array_receipt(valid)

    source_target = np.ascontiguousarray(np.flip(np.flip(target, axis=0), axis=1))
    source_valid = np.ascontiguousarray(np.flip(np.flip(valid, axis=0), axis=1))
    observation_receipt = _observation_receipt(final, source_target, source_valid)
    frame["crop_window_id"] = observation_receipt["crop_window_id"]
    frame["frame_transform_id"] = acquisition._payload_sha256(
        {
            key: value
            for key, value in frame.items()
            if key not in {"arrays", "frame_transform_id"}
        }
    )
    final["paired_mode_sensitivity_reference"]["frame_transform_id"] = frame[
        "frame_transform_id"
    ]
    binding = {
        "receipt_payload": observation_receipt,
        "receipt_sha256": acquisition._payload_sha256(observation_receipt),
    }
    upstream = final["upstream_reference"]
    upstream["observation_bundle_id"] = observation_receipt[
        "observation_bundle_id"
    ]
    upstream["observation_receipt_sha256"] = binding["receipt_sha256"]
    upstream["acquired_observation_id"] = observation_receipt[
        "acquired_observation_id"
    ]
    upstream["crop_window_id"] = observation_receipt["crop_window_id"]
    upstream["live_receipt_bindings"]["observation_bundle"] = binding
    final["synthetic_realization_id"] = acquisition._payload_sha256(
        realization_v2._identity_payload(final)
    )
    final["receipt_sha256"] = acquisition._payload_sha256(
        realization_v2.synthetic_realization_receipt_v2(final)
    )

    pose = pose_v2.make_arbitrary_plane_pose_truth_v2(final, context)
    bank = candidate_v2.make_arbitrary_plane_candidate_bank_v2(
        pose, final, context
    )
    candidate_v2.verify_arbitrary_plane_candidate_bank_v2(
        bank, pose, final, context
    )
    result = oracle_v2.make_arbitrary_plane_semantic_oracle_result_v2(
        bank, pose, final, context
    )
    replayed = oracle_v2.replay_arbitrary_plane_semantic_oracle_result_v2(
        result, bank, pose, final, context
    )
    oracle_v2.verify_arbitrary_plane_semantic_oracle_result_v2(
        result, bank, pose, final, context
    )

    assert result["receipt_sha256"] == replayed["receipt_sha256"]
    assert result["candidate_reference"]["candidate_count"] == 40
    assert result["coverage"]["evaluable"] is True
    assert all(record["passed"] for record in result["exact_controls"].values())
    assert result["exact_controls"]["crop_reflection_binding"]["evidence"][
        "horizontal_reflection"
    ] is True
    assert result["exact_controls"]["crop_reflection_binding"]["evidence"][
        "vertical_reflection"
    ] is True
    assert observation_receipt["observation_plan_id"] == acquisition._payload_sha256(
        observation_receipt["plan_identity_payload"]
    )
    assert observation_receipt["crop_window_id"] == acquisition._payload_sha256(
        observation_receipt["crop_identity_payload"]
    )
    assert observation_receipt[
        "acquired_observation_id"
    ] == acquisition._payload_sha256(
        observation_receipt["acquired_identity_payload"]
    )
    assert observation_receipt[
        "observation_bundle_id"
    ] == acquisition._payload_sha256(
        observation_receipt["bundle_identity_payload"]
    )
