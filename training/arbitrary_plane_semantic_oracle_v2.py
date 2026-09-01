"""Model-free semantic ranking of one authenticated v2 candidate bank."""

from __future__ import annotations

import math
from collections.abc import Mapping
from inspect import signature
from pathlib import Path

import numpy as np
import scipy

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_candidate_bank_v2 as candidate_v2
from training.arbitrary_plane_semantic_oracle import (
    finite_point_error,
    rank_candidate_ids,
    rp2_plane_error,
    score_semantic_candidates,
)


SEMANTIC_ORACLE_RESULT_V2_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-semantic-oracle-result/v2"
)
SEMANTIC_ORACLE_RESULT_V2_ALGORITHM = (
    "fixed-valid-region-balanced-atlas-label-candidate-ranking/v2"
)
PITCH_ISOTROPY_RTOL = 1.0e-12
MINIMUM_INDIVIDUAL_REGION_PIXELS = 16
SMOOTHING_SIGMA_UM = 75.0
NUMERICAL_SCORER_INPUTS = (
    "target_labels",
    "candidate_labels",
    "fixed_valid_mask",
    "pixel_pitch_um",
)
FORBIDDEN_NUMERICAL_SCORER_INPUTS = (
    "model_input_image_or_channels",
    "selected_input_mask",
    "brush_availability_or_error",
    "valid_correspondence_weight",
    "mapped_physical_coordinates_or_residuals",
    "pose_or_candidate_metadata",
    "candidate_ids_classes_order_or_proposal_deltas",
    "candidate_brain_or_support_masks",
)
EXACT_CONTROL_NAMES = (
    "candidate_order_permutation_equivariance",
    "truth_independent_atlas_rerender",
    "truth_nominal_coordinate_grid",
    "crop_reflection_binding",
    "rp2_sign_equivalence",
    "strict_scorer_input_exclusion_and_fixed_denominator",
)
_SCORE_ARRAY_KEYS = {
    "semantic_score_float64",
    "raw_id_agreement_float64",
    "mask_dice_float64",
}
_RESULT_KEYS = {
    "schema_version",
    "algorithm",
    "implementation_source_sha256",
    "implementation_source_sha256_canonicalization",
    "runtime_dependencies",
    "asset_dependencies",
    "scope",
    "upstream_reference",
    "provenance",
    "scorer_input_contract",
    "target_reference",
    "candidate_reference",
    "scores",
    "ranking",
    "selected_pose_error",
    "coverage",
    "exact_controls",
    "semantic_oracle_result_id",
    "receipt_sha256",
}
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "arbitrary_plane_semantic_oracle_v2.py",
    "arbitrary_plane_semantic_oracle.py",
    "arbitrary_plane_candidate_bank_v2.py",
    "arbitrary_plane_pose_v2.py",
    "arbitrary_plane_realization_v2.py",
    "arbitrary_plane_acquisition_v2.py",
)
_OBSERVATION_RECEIPT_KEYS = {
    "observation_plan_id",
    "crop_window_id",
    "acquired_observation_id",
    "observation_bundle_id",
    "plan_identity_payload",
    "crop_identity_payload",
    "acquired_identity_payload",
    "bundle_identity_payload",
    "descendant_identity_payloads",
}
_OBSERVATION_DESCENDANT_MODES = (
    "raw",
    "smart-brush-accurate",
    "smart-brush-imperfect",
    "smart-brush-absent",
)


def _source_hashes() -> dict[str, str]:
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _score_semantic_arrays_v2(
    target_labels: np.ndarray,
    candidate_labels: np.ndarray,
    fixed_valid_mask: np.ndarray,
    pixel_pitch_um: float,
) -> dict[str, np.ndarray | float | int]:
    """The entire numerical scorer boundary: exactly four non-metadata inputs."""
    return score_semantic_candidates(
        target_labels, candidate_labels, fixed_valid_mask, pixel_pitch_um
    )


def _byte_equal(left: object, right: object) -> bool:
    left_array, right_array = np.asarray(left), np.asarray(right)
    return (
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and np.ascontiguousarray(left_array).tobytes(order="C")
        == np.ascontiguousarray(right_array).tobytes(order="C")
    )


def _physical_raster(ouv: np.ndarray, shape_h_w: tuple[int, int]) -> np.ndarray:
    values = np.asarray(ouv, dtype=np.float64)
    height, width = shape_h_w
    if values.shape != (3, 3) or min(height, width) < 2:
        raise ValueError("semantic-oracle O/U/V or raster shape is invalid")
    y, x = np.indices((height, width), dtype=np.float64)
    return np.ascontiguousarray(
        values[0][None, None]
        + (x / width)[..., None] * values[1][None, None]
        + (y / height)[..., None] * values[2][None, None]
    )


def _reflected_ouv(
    pre_reflection_ouv: np.ndarray,
    shape_h_w: tuple[int, int],
    horizontal: bool,
    vertical: bool,
) -> np.ndarray:
    height, width = shape_h_w
    result = np.array(pre_reflection_ouv, dtype=np.float64, copy=True, order="C")
    if horizontal:
        result[0] = result[0] + ((width - 1.0) / width) * result[1]
        result[1] = -result[1]
    if vertical:
        result[0] = result[0] + ((height - 1.0) / height) * result[2]
        result[2] = -result[2]
    return np.ascontiguousarray(result)


def _observation_receipt_payload(
    final_realization: Mapping[str, object],
) -> tuple[dict[str, object], Mapping[str, object]]:
    try:
        binding = final_realization["upstream_reference"]["live_receipt_bindings"][
            "observation_bundle"
        ]
        payload = acquisition._json_value(binding["receipt_payload"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("semantic target lacks its authenticated observation receipt") from error
    try:
        if set(payload) != _OBSERVATION_RECEIPT_KEYS:
            raise ValueError("semantic target observation receipt structure changed")
        plan = payload["plan_identity_payload"]
        crop = payload["crop_identity_payload"]
        acquired = payload["acquired_identity_payload"]
        bundle = payload["bundle_identity_payload"]
        descendants = payload["descendant_identity_payloads"]
        if not all(
            isinstance(value, Mapping)
            for value in (plan, crop, acquired, bundle, descendants)
        ):
            raise ValueError("semantic target observation identity payload changed")
        plan_id = acquisition._payload_sha256(plan)
        crop_id = acquisition._payload_sha256(crop)
        acquired_id = acquisition._payload_sha256(acquired)
        if set(descendants) != set(_OBSERVATION_DESCENDANT_MODES):
            raise ValueError("semantic target observation descendants changed")
        if not all(
            isinstance(descendants[name], Mapping)
            for name in _OBSERVATION_DESCENDANT_MODES
        ):
            raise ValueError("semantic target observation descendant payload changed")
        descendant_ids = {
            name: acquisition._payload_sha256(descendants[name])
            for name in _OBSERVATION_DESCENDANT_MODES
        }
        bundle_id = acquisition._payload_sha256(bundle)
        valid = (
            plan.get("domain") == "anatomy-tracker.observation-plan/v2"
            and crop.get("domain") == "anatomy-tracker.observation-crop-window/v2"
            and acquired.get("domain") == "anatomy-tracker.acquired-observation/v2"
            and bundle.get("domain") == "anatomy-tracker.observation-bundle/v2"
            and all(
                descendants[name].get("domain")
                == "anatomy-tracker.observation-descendant/v2"
                and descendants[name].get("mode") == name
                and descendants[name].get("acquired_observation_id") == acquired_id
                for name in _OBSERVATION_DESCENDANT_MODES
            )
            and payload["observation_plan_id"] == plan_id
            and crop.get("observation_plan_id") == plan_id
            and payload["crop_window_id"] == crop_id
            and acquired.get("observation_plan_id") == plan_id
            and acquired.get("crop_window_id") == crop_id
            and payload["acquired_observation_id"] == acquired_id
            and bundle.get("observation_plan_id") == plan_id
            and bundle.get("acquired_observation_id") == acquired_id
            and acquisition._json_value(bundle.get("descendant_ids"))
            == acquisition._json_value(descendant_ids)
            and payload["observation_bundle_id"] == bundle_id
            and binding["receipt_sha256"] == acquisition._payload_sha256(payload)
            and binding["receipt_sha256"]
            == final_realization["upstream_reference"]["observation_receipt_sha256"]
            and bundle_id
            == final_realization["upstream_reference"]["observation_bundle_id"]
            and acquired_id
            == final_realization["upstream_reference"]["acquired_observation_id"]
            and crop_id
            == final_realization["upstream_reference"]["crop_window_id"]
            and crop_id == final_realization["frame_transform"]["crop_window_id"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError("semantic target observation receipt identity is incomplete") from error
    if not valid:
        raise ValueError("semantic target observation receipt identity changed")
    return payload, binding


def _processed_pixel_pitch(
    final_realization: Mapping[str, object],
    observation_payload: Mapping[str, object],
    observation_binding: Mapping[str, object],
) -> tuple[float, dict[str, object]]:
    try:
        crop = observation_payload["crop_identity_payload"]["crop_window"]
        pitch = np.asarray(crop["processed_pixel_pitch_y_x_um"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("semantic target lacks its authenticated processed-pixel pitch") from error
    if (
        pitch.shape != (2,)
        or not np.isfinite(pitch).all()
        or np.any(pitch <= 0.0)
        or not np.isclose(
            pitch[0], pitch[1], rtol=PITCH_ISOTROPY_RTOL, atol=0.0
        )
    ):
        raise ValueError("semantic target requires one authenticated isotropic pixel pitch")
    reference = {
        "pixel_pitch_y_x_um": pitch.tolist(),
        "selected_pixel_pitch_um": float(pitch[0]),
        "isotropy_relative_tolerance": PITCH_ISOTROPY_RTOL,
        "crop_window_id": observation_payload["crop_window_id"],
        "observation_bundle_receipt_sha256": observation_binding["receipt_sha256"],
        "source": "authenticated observation crop processed_pixel_pitch_y_x_um",
        "model_ouv_pitch_used": False,
    }
    return float(pitch[0]), reference


def _scoring_arrays(
    candidate_bank: Mapping[str, object],
    final_realization: Mapping[str, object],
    observation_payload: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    target = np.asarray(
        final_realization["targets"]["source_label_ground_truth_crop_int64"]
    )
    valid = np.asarray(final_realization["targets"]["valid_correspondence_mask"])
    receipts = final_realization["target_array_receipts"]
    horizontal = bool(final_realization["frame_transform"]["horizontal_reflection"])
    vertical = bool(final_realization["frame_transform"]["vertical_reflection"])
    unreflected_target = np.asarray(target)
    unreflected_valid = np.asarray(valid)
    if vertical:
        unreflected_target = np.flip(unreflected_target, axis=0)
        unreflected_valid = np.flip(unreflected_valid, axis=0)
    if horizontal:
        unreflected_target = np.flip(unreflected_target, axis=1)
        unreflected_valid = np.flip(unreflected_valid, axis=1)
    unreflected_target = np.ascontiguousarray(unreflected_target)
    unreflected_valid = np.ascontiguousarray(unreflected_valid)
    try:
        acquired_receipts = observation_payload["acquired_identity_payload"][
            "array_receipts"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("authenticated observation target receipts are incomplete") from error
    if (
        target.dtype != np.dtype(np.int64)
        or target.ndim != 2
        or valid.dtype != np.dtype(bool)
        or valid.shape != target.shape
        or not valid.any()
        or np.any(target < 0)
        or np.any(target[valid] == 0)
        or acquisition._json_value(receipts["source_label_ground_truth_crop_int64"])
        != acquisition._json_value(acquisition._array_receipt(target))
        or acquisition._json_value(receipts["valid_correspondence_mask"])
        != acquisition._json_value(acquisition._array_receipt(valid))
        or acquisition._json_value(
            acquired_receipts["source_label_ground_truth_crop_int64"]
        )
        != acquisition._json_value(acquisition._array_receipt(unreflected_target))
        or acquisition._json_value(acquired_receipts["valid_correspondence_mask"])
        != acquisition._json_value(acquisition._array_receipt(unreflected_valid))
    ):
        raise ValueError(
            "semantic target labels or fixed-valid mask no longer match their observation trust root"
        )
    candidates = list(candidate_bank["candidates"])
    ordered_ids = [str(value) for value in candidate_bank["ordered_candidate_ids"]]
    if (
        len(candidates) != 40
        or len(ordered_ids) != 40
        or len(set(ordered_ids)) != 40
        or [str(item["candidate_id"]) for item in candidates] != ordered_ids
    ):
        raise ValueError("semantic oracle requires forty unique ordered candidates")
    labels = np.stack(
        [np.asarray(item["arrays"]["rendered_annotation_int64"]) for item in candidates]
    )
    if (
        labels.dtype != np.dtype(np.int64)
        or labels.shape != (40, *target.shape)
        or np.any(labels < 0)
    ):
        raise ValueError("candidate semantic label stack changed")
    truth_indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate["candidate_class"] == "truth"
    ]
    if len(truth_indices) != 1:
        raise ValueError("candidate bank must contain exactly one truth")
    return (
        np.ascontiguousarray(target),
        np.ascontiguousarray(valid),
        np.ascontiguousarray(labels),
        ordered_ids,
        truth_indices[0],
    )


def _control_record(
    name: str, passed: bool, evidence: Mapping[str, object]
) -> dict[str, object]:
    payload = {
        "control": name,
        "passed": bool(passed),
        "evidence": acquisition._json_value(evidence),
    }
    return {
        **payload,
        "evidence_receipt_sha256": acquisition._payload_sha256(payload),
    }


def _score_arrays(score: Mapping[str, object]) -> dict[str, np.ndarray]:
    result = {
        "semantic_score_float64": np.ascontiguousarray(
            score["semantic_score"], dtype=np.float64
        ),
        "raw_id_agreement_float64": np.ascontiguousarray(
            score["raw_id_agreement"], dtype=np.float64
        ),
        "mask_dice_float64": np.ascontiguousarray(
            score["mask_dice"], dtype=np.float64
        ),
    }
    if any(
        value.shape != (40,)
        or not np.isfinite(value).all()
        or np.any((value < 0.0) | (value > 1.0))
        for value in result.values()
    ):
        raise ValueError("semantic score landscapes must be finite forty-vectors in [0,1]")
    return result


def _identity_payload(result: Mapping[str, object]) -> dict[str, object]:
    return acquisition._json_value(
        {
            key: value
            for key, value in result.items()
            if key
            not in {
                "scores",
                "semantic_oracle_result_id",
                "receipt_sha256",
            }
        }
        | {
            "scores": {
                key: value
                for key, value in result["scores"].items()
                if key != "arrays"
            }
        }
    )


def arbitrary_plane_semantic_oracle_result_receipt_v2(
    result: Mapping[str, object],
) -> dict[str, object]:
    return {
        "semantic_oracle_result_id": result["semantic_oracle_result_id"],
        "identity_payload": _identity_payload(result),
    }


def make_arbitrary_plane_semantic_oracle_result_v2(
    candidate_bank: Mapping[str, object],
    pose_truth: Mapping[str, object],
    final_realization: Mapping[str, object],
    prepared_context: Mapping[str, object],
) -> Mapping[str, object]:
    """Score all forty candidates without exposing metadata to the scorer."""
    candidate_v2.verify_arbitrary_plane_candidate_bank_v2(
        candidate_bank, pose_truth, final_realization, prepared_context
    )
    observation_payload, observation_binding = _observation_receipt_payload(
        final_realization
    )
    target, valid, candidate_labels, ordered_ids, truth_index = _scoring_arrays(
        candidate_bank, final_realization, observation_payload
    )
    pitch, pitch_reference = _processed_pixel_pitch(
        final_realization, observation_payload, observation_binding
    )
    raw_score = _score_semantic_arrays_v2(target, candidate_labels, valid, pitch)
    if (
        type(raw_score["channel_count"]) is not int
        or raw_score["channel_count"] <= 0
        or not np.isclose(
            float(raw_score["smoothing_sigma_px"]),
            SMOOTHING_SIGMA_UM / pitch,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ValueError("semantic scorer channel count or physical smoothing changed")
    score_arrays = _score_arrays(raw_score)
    truth_id = ordered_ids[truth_index]
    ranking = rank_candidate_ids(
        score_arrays["semantic_score_float64"], ordered_ids, truth_id
    )
    ranking["top3"] = bool(ranking["true_rank"] <= 3)

    permutation = np.arange(39, -1, -1, dtype=np.int64)
    permuted = _score_semantic_arrays_v2(
        target, candidate_labels[permutation], valid, pitch
    )
    permuted_arrays = _score_arrays(permuted)
    inverse = np.argsort(permutation)
    order_scores_equal = all(
        _byte_equal(permuted_arrays[name][inverse], score_arrays[name])
        for name in _SCORE_ARRAY_KEYS
    )
    permuted_ranking = rank_candidate_ids(
        permuted_arrays["semantic_score_float64"],
        [ordered_ids[index] for index in permutation],
        truth_id,
    )
    invariant_rank_fields = (
        "truth_score",
        "top1",
        "true_rank",
        "reciprocal_rank",
        "true_versus_decoy_win_fraction",
        "true_score_margin",
        "tied_maximum_candidate_ids",
        "selected_candidate_id",
    )
    order_rank_equal = all(
        acquisition._json_value(permuted_ranking[name])
        == acquisition._json_value(ranking[name])
        for name in invariant_rank_fields
    )

    candidates = list(candidate_bank["candidates"])
    truth_candidate = candidates[truth_index]
    truth_arrays = truth_candidate["arrays"]
    truth_model_ouv = np.asarray(
        truth_arrays["model_raster_physical_ouv_ap_dv_ml_um_float64"]
    )
    rerendered = candidate_v2.render_physical_ouv_annotation_v2(
        prepared_context, truth_model_ouv, target.shape
    )
    rerender_equal = (
        _byte_equal(rerendered["rendered_annotation_int64"], candidate_labels[truth_index])
        and _byte_equal(rerendered["brain_mask"], truth_arrays["brain_mask"])
        and acquisition._json_value(
            rerendered["physical_coordinate_raster_receipt"]
        )
        == acquisition._json_value(
            truth_candidate["render_contract"]["physical_coordinate_raster_receipt"]
        )
        and acquisition._json_value(
            rerendered["allen_index_coordinate_raster_float32_receipt"]
        )
        == acquisition._json_value(
            truth_candidate["render_contract"][
                "allen_index_coordinate_raster_float32_receipt"
            ]
        )
        and rerendered["annotation_array_sha256"]
        == prepared_context["receipt"]["annotation_array_sha256"]
        and truth_candidate["render_contract"][
            "target_overlap_used_for_construction_or_acceptance"
        ]
        is False
    )

    pose_arrays = pose_truth["arrays"]
    pose_model_ouv = np.asarray(
        pose_arrays["model_raster_physical_ouv_ap_dv_ml_um_float64"]
    )
    final_model_ouv = np.asarray(
        final_realization["frame_transform"]["arrays"][
            "model_raster_physical_ouv_ap_dv_ml_um_float64"
        ]
    )
    nominal = _physical_raster(truth_model_ouv, target.shape)
    saved_nominal = np.asarray(
        final_realization["factor_truth"]["arrays"][
            "nominal_physical_map_ap_dv_ml_um_float64"
        ]
    )
    nominal_equal = (
        _byte_equal(truth_model_ouv, pose_model_ouv)
        and _byte_equal(truth_model_ouv, final_model_ouv)
        and _byte_equal(nominal, saved_nominal)
        and acquisition._json_value(
            final_realization["factor_truth"]["array_receipts"][
                "nominal_physical_map_ap_dv_ml_um_float64"
            ]
        )
        == acquisition._json_value(acquisition._array_receipt(saved_nominal))
    )

    pre_ouv = np.asarray(
        truth_arrays[
            "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64"
        ]
    )
    pose_pre_ouv = np.asarray(
        pose_arrays[
            "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64"
        ]
    )
    horizontal = bool(pose_truth["reflection_state"]["horizontal_reflection"])
    vertical = bool(pose_truth["reflection_state"]["vertical_reflection"])
    reflected = _reflected_ouv(pre_ouv, target.shape, horizontal, vertical)
    reflection_equal = (
        _byte_equal(pre_ouv, pose_pre_ouv)
        and _byte_equal(reflected, truth_model_ouv)
        and candidate_bank["reflection_state"]["horizontal_reflection"]
        is horizontal
        and candidate_bank["reflection_state"]["vertical_reflection"] is vertical
        and final_realization["frame_transform"]["horizontal_reflection"]
        is horizontal
        and final_realization["frame_transform"]["vertical_reflection"] is vertical
    )

    truth_plane = np.array(
        pose_arrays["actual_plane_normal_and_signed_offset_um_float64"],
        dtype=np.float64,
        copy=True,
    )
    antipodal_error = rp2_plane_error(
        truth_plane[:3],
        float(truth_plane[3]),
        -truth_plane[:3],
        -float(truth_plane[3]),
    )
    rp2_equal = all(value == 0.0 for value in antipodal_error.values())
    scorer_signature_equal = (
        tuple(signature(_score_semantic_arrays_v2).parameters)
        == NUMERICAL_SCORER_INPUTS
    )

    controls = {
        "candidate_order_permutation_equivariance": _control_record(
            "candidate_order_permutation_equivariance",
            order_scores_equal and order_rank_equal,
            {
                "permutation": permutation.tolist(),
                "original_score_array_receipts": {
                    name: acquisition._array_receipt(value)
                    for name, value in score_arrays.items()
                },
                "permuted_then_restored_score_array_receipts": {
                    name: acquisition._array_receipt(permuted_arrays[name][inverse])
                    for name in _SCORE_ARRAY_KEYS
                },
                "coordinate_free_ranking_fields": list(invariant_rank_fields),
            },
        ),
        "truth_independent_atlas_rerender": _control_record(
            "truth_independent_atlas_rerender",
            rerender_equal,
            {
                "truth_candidate_id": truth_id,
                "truth_label_receipt": acquisition._array_receipt(
                    candidate_labels[truth_index]
                ),
                "rerendered_label_receipt": acquisition._array_receipt(
                    rerendered["rendered_annotation_int64"]
                ),
                "annotation_array_sha256": rerendered["annotation_array_sha256"],
                "target_overlap_used": False,
            },
        ),
        "truth_nominal_coordinate_grid": _control_record(
            "truth_nominal_coordinate_grid",
            nominal_equal,
            {
                "truth_model_ouv_receipt": acquisition._array_receipt(truth_model_ouv),
                "derived_nominal_map_receipt": acquisition._array_receipt(nominal),
                "saved_nominal_map_receipt": acquisition._array_receipt(saved_nominal),
                "sampling_contract": "O+(x/W)U+(y/H)V",
            },
        ),
        "crop_reflection_binding": _control_record(
            "crop_reflection_binding",
            reflection_equal,
            {
                "horizontal_reflection": horizontal,
                "vertical_reflection": vertical,
                "reflection_order": ["horizontal", "vertical"],
                "pre_reflection_ouv_receipt": acquisition._array_receipt(pre_ouv),
                "recomputed_model_ouv_receipt": acquisition._array_receipt(reflected),
            },
        ),
        "rp2_sign_equivalence": _control_record(
            "rp2_sign_equivalence",
            rp2_equal,
            {
                "truth_plane": truth_plane.tolist(),
                "antipodal_plane": (-truth_plane).tolist(),
                "rp2_error": antipodal_error,
            },
        ),
        "strict_scorer_input_exclusion_and_fixed_denominator": _control_record(
            "strict_scorer_input_exclusion_and_fixed_denominator",
            scorer_signature_equal,
            {
                "numerical_scorer_parameters": list(
                    signature(_score_semantic_arrays_v2).parameters
                ),
                "allowed_inputs": list(NUMERICAL_SCORER_INPUTS),
                "forbidden_inputs": list(FORBIDDEN_NUMERICAL_SCORER_INPUTS),
                "fixed_valid_mask_receipt": acquisition._array_receipt(valid),
                "candidate_specific_mask_argument_present": False,
                "valid_correspondence_weight_accessed": False,
                "candidate_metadata_joined_after_numerical_scoring": True,
            },
        ),
    }
    failed_controls = [name for name, record in controls.items() if not record["passed"]]
    if failed_controls:
        raise ValueError(f"semantic-oracle exact controls failed: {failed_controls}")

    selected_id = ranking["selected_candidate_id"]
    if selected_id is None:
        selected_error = {
            "available": False,
            "reason": "maximum score is tied; no unique selected candidate",
        }
    else:
        selected = candidates[ordered_ids.index(selected_id)]
        selected_pose = selected["pose"]
        plane_error = rp2_plane_error(
            truth_plane[:3],
            float(truth_plane[3]),
            np.asarray(selected_pose["actual_normal_ap_dv_ml"]),
            float(selected_pose["actual_signed_offset_um"]),
        )
        point_error = finite_point_error(
            truth_model_ouv.reshape(9),
            np.asarray(
                selected["arrays"][
                    "model_raster_physical_ouv_ap_dv_ml_um_float64"
                ]
            ).reshape(9),
            valid,
        )
        roll = float(selected_pose["roll_delta_rad_from_truth"])
        selected_error = {
            "available": True,
            "selected_candidate_id": selected_id,
            "selected_candidate_class": selected["candidate_class"],
            **plane_error,
            "absolute_wrapped_roll_error_deg": abs(
                math.degrees(math.atan2(math.sin(roll), math.cos(roll)))
            ),
            **point_error,
        }

    truth_brain_pixels = int(truth_candidate["brain_pixel_count"])
    truth_evaluable = candidate_bank["truth_evaluability"]
    if (
        truth_evaluable["independent_truth_brain_pixel_count"] != truth_brain_pixels
        or truth_evaluable["evaluable"] is not (truth_brain_pixels > 0)
    ):
        raise ValueError("candidate truth evaluability no longer matches its rendered crop")
    evaluable = truth_brain_pixels > 0
    ranking = acquisition._json_value(ranking)
    ranking["raw_top1_before_coverage_policy"] = bool(ranking["top1"])
    ranking["raw_top3_before_coverage_policy"] = bool(ranking["top3"])
    ranking["top1"] = bool(evaluable and ranking["raw_top1_before_coverage_policy"])
    ranking["top3"] = bool(evaluable and ranking["raw_top3_before_coverage_policy"])
    ranking["coverage_adjusted_top1"] = ranking["top1"]
    ranking["coverage_adjusted_top3"] = ranking["top3"]

    target_receipts = final_realization["target_array_receipts"]
    large_ids = np.asarray(raw_score["target_large_region_ids"], dtype=np.int64)
    small_ids = np.asarray(raw_score["target_small_region_ids"], dtype=np.int64)
    channel_reference = {
        "minimum_individual_region_pixels": MINIMUM_INDIVIDUAL_REGION_PIXELS,
        "large_region_ids": large_ids.tolist(),
        "small_pooled_region_ids": small_ids.tolist(),
        "large_region_ids_receipt": acquisition._array_receipt(large_ids),
        "small_pooled_region_ids_receipt": acquisition._array_receipt(small_ids),
        "channel_count": int(raw_score["channel_count"]),
    }
    candidate_summaries = [
        {
            "candidate_id": candidate["candidate_id"],
            "candidate_class": candidate["candidate_class"],
            "slot": candidate["slot"],
            "brain_pixel_count": candidate["brain_pixel_count"],
            "finite_raster_support": candidate["finite_raster_support"],
            "plane_intersects_support_envelope": candidate[
                "infinite_plane_support_envelope"
            ]["plane_intersects_support_envelope"],
        }
        for candidate in candidates
    ]
    score_receipts = {
        name: acquisition._array_receipt(value)
        for name, value in score_arrays.items()
    }
    context_receipt = acquisition._json_value(prepared_context["receipt"])
    result = {
        "schema_version": SEMANTIC_ORACLE_RESULT_V2_SCHEMA,
        "algorithm": SEMANTIC_ORACLE_RESULT_V2_ALGORITHM,
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "runtime_dependencies": {
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "scorer": "training.arbitrary_plane_semantic_oracle.score_semantic_candidates",
        },
        "asset_dependencies": {
            "learned_checkpoint_dependencies": [],
            "pretrained_feature_dependencies": [],
            "previous_model_dependencies": [],
        },
        "scope": {
            "model_free": True,
            "forced_truth_finite_ranking_premise_only": True,
            "posterior_or_probability_claim": False,
            "calibrated_uncertainty_claim": False,
            "semantic_score_is_probability": False,
            "valid_correspondence_weight_used": False,
            "weighted_sensitivity_status": "separate deferred descriptive artifact",
            "benchmark_or_final_test_claim": False,
        },
        "upstream_reference": {
            "candidate_bank_id": candidate_bank["candidate_bank_id"],
            "candidate_bank_receipt_sha256": candidate_bank["receipt_sha256"],
            "finite_plane_pose_truth_id": pose_truth["finite_plane_pose_truth_id"],
            "finite_plane_pose_truth_receipt_sha256": pose_truth["receipt_sha256"],
            "synthetic_realization_id": final_realization["synthetic_realization_id"],
            "synthetic_realization_receipt_sha256": final_realization["receipt_sha256"],
            "training_row_id": final_realization["training_row_id"],
            "frame_transform_id": final_realization["frame_transform"]["frame_transform_id"],
            "v2_context_sha256": prepared_context["v2_context_sha256"],
            "prepared_context_receipt_sha256": acquisition._payload_sha256(
                context_receipt
            ),
            "support_index_sha256": prepared_context["receipt"][
                "support_index_sha256"
            ],
            "annotation_array_sha256": prepared_context["receipt"][
                "annotation_array_sha256"
            ],
        },
        "provenance": {
            "source_lineage": acquisition._json_value(candidate_bank["source_lineage"]),
            "observation_and_realization": acquisition._json_value(
                final_realization["provenance"]
            ),
        },
        "scorer_input_contract": {
            "allowed_inputs": list(NUMERICAL_SCORER_INPUTS),
            "forbidden_inputs": list(FORBIDDEN_NUMERICAL_SCORER_INPUTS),
            "target_labels_receipt": acquisition._json_value(
                target_receipts["source_label_ground_truth_crop_int64"]
            ),
            "fixed_valid_mask_receipt": acquisition._json_value(
                target_receipts["valid_correspondence_mask"]
            ),
            "candidate_label_stack_receipt": acquisition._array_receipt(
                candidate_labels
            ),
            "pixel_pitch_reference": pitch_reference,
            "candidate_ids_and_metadata_joined_after_scoring_only": True,
        },
        "target_reference": {
            "labels_receipt": acquisition._json_value(
                target_receipts["source_label_ground_truth_crop_int64"]
            ),
            "fixed_valid_mask_receipt": acquisition._json_value(
                target_receipts["valid_correspondence_mask"]
            ),
            "shape_h_w": list(target.shape),
            "fixed_valid_pixel_count": int(valid.sum()),
            "fixed_valid_nonzero_region_id_count": int(
                np.unique(target[valid]).size
            ),
            "channel_reference": channel_reference,
            "pixel_pitch_reference": pitch_reference,
        },
        "candidate_reference": {
            "candidate_count": 40,
            "ordered_candidate_ids": ordered_ids,
            "truth_candidate_id": truth_id,
            "truth_candidate_index": truth_index,
            "candidate_label_stack_receipt": acquisition._array_receipt(
                candidate_labels
            ),
            "model_grid_reference": acquisition._json_value(
                candidate_bank["model_grid_reference"]
            ),
            "candidate_summaries": candidate_summaries,
        },
        "scores": {
            "arrays": score_arrays,
            "array_receipts": score_receipts,
            "channel_count": int(raw_score["channel_count"]),
            "smoothing_sigma_px": float(raw_score["smoothing_sigma_px"]),
        },
        "ranking": ranking,
        "selected_pose_error": selected_error,
        "coverage": {
            "target_fixed_valid_pixel_count": int(valid.sum()),
            "independent_truth_brain_pixel_count": truth_brain_pixels,
            "evaluable": evaluable,
            "failure_reason": None
            if evaluable
            else "independent truth atlas render has zero finite-crop brain support",
            "zero_support_truth_policy": "save raw ranking, count top1/top3 false, never redraw",
        },
        "exact_controls": controls,
    }
    result["semantic_oracle_result_id"] = acquisition._payload_sha256(
        _identity_payload(result)
    )
    result["receipt_sha256"] = acquisition._payload_sha256(
        arbitrary_plane_semantic_oracle_result_receipt_v2(result)
    )
    return acquisition._freeze_value(result)


def replay_arbitrary_plane_semantic_oracle_result_v2(
    result: Mapping[str, object],
    candidate_bank: Mapping[str, object],
    pose_truth: Mapping[str, object],
    final_realization: Mapping[str, object],
    prepared_context: Mapping[str, object],
) -> Mapping[str, object]:
    if result.get("schema_version") != SEMANTIC_ORACLE_RESULT_V2_SCHEMA:
        raise ValueError("unsupported semantic-oracle v2 result")
    return make_arbitrary_plane_semantic_oracle_result_v2(
        candidate_bank, pose_truth, final_realization, prepared_context
    )


def verify_arbitrary_plane_semantic_oracle_result_v2(
    result: Mapping[str, object],
    candidate_bank: Mapping[str, object],
    pose_truth: Mapping[str, object],
    final_realization: Mapping[str, object],
    prepared_context: Mapping[str, object],
) -> None:
    if (
        set(result) != _RESULT_KEYS
        or result.get("schema_version") != SEMANTIC_ORACLE_RESULT_V2_SCHEMA
        or result.get("algorithm") != SEMANTIC_ORACLE_RESULT_V2_ALGORITHM
        or acquisition._json_value(result.get("implementation_source_sha256"))
        != _source_hashes()
        or result.get("implementation_source_sha256_canonicalization")
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or any(result.get("asset_dependencies", {}).values())
        or result.get("scope", {}).get("posterior_or_probability_claim") is not False
        or result.get("scope", {}).get("semantic_score_is_probability") is not False
        or result.get("scope", {}).get("valid_correspondence_weight_used") is not False
        or set(result.get("scores", {}))
        != {"arrays", "array_receipts", "channel_count", "smoothing_sigma_px"}
        or set(result.get("scores", {}).get("arrays", {})) != _SCORE_ARRAY_KEYS
        or set(result.get("scores", {}).get("array_receipts", {}))
        != _SCORE_ARRAY_KEYS
        or set(result.get("exact_controls", {})) != set(EXACT_CONTROL_NAMES)
    ):
        raise ValueError("semantic-oracle v2 result structure or scope changed")
    for name, values in result["scores"]["arrays"].items():
        array = np.asarray(values)
        if (
            array.dtype != np.dtype(np.float64)
            or array.shape != (40,)
            or not np.isfinite(array).all()
            or np.any((array < 0.0) | (array > 1.0))
            or acquisition._json_value(result["scores"]["array_receipts"][name])
            != acquisition._json_value(acquisition._array_receipt(array))
        ):
            raise ValueError("semantic-oracle raw score array or receipt changed")
    for name, record in result["exact_controls"].items():
        payload = {
            "control": name,
            "passed": record.get("passed"),
            "evidence": acquisition._json_value(record.get("evidence")),
        }
        if (
            set(record) != {
                "control",
                "passed",
                "evidence",
                "evidence_receipt_sha256",
            }
            or record.get("control") != name
            or record.get("passed") is not True
            or not isinstance(record.get("evidence"), Mapping)
            or not record["evidence"]
            or record.get("evidence_receipt_sha256")
            != acquisition._payload_sha256(payload)
        ):
            raise ValueError("semantic-oracle exact control or evidence changed")
    if (
        result["semantic_oracle_result_id"]
        != acquisition._payload_sha256(_identity_payload(result))
        or result["receipt_sha256"]
        != acquisition._payload_sha256(
            arbitrary_plane_semantic_oracle_result_receipt_v2(result)
        )
    ):
        raise ValueError("semantic-oracle result identity or receipt changed")
    expected = replay_arbitrary_plane_semantic_oracle_result_v2(
        result, candidate_bank, pose_truth, final_realization, prepared_context
    )
    if acquisition._json_value(
        arbitrary_plane_semantic_oracle_result_receipt_v2(result)
    ) != acquisition._json_value(
        arbitrary_plane_semantic_oracle_result_receipt_v2(expected)
    ):
        raise ValueError("semantic-oracle deterministic receipt replay changed")
    if any(
        not _byte_equal(result["scores"]["arrays"][name], expected["scores"]["arrays"][name])
        for name in _SCORE_ARRAY_KEYS
    ):
        raise ValueError("semantic-oracle raw score replay changed")
