"""Shape-preserving semantic null and development-only engineering gates."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_semantic_oracle_v2 as primary_v2
from training.arbitrary_plane_semantic_oracle import rank_candidate_ids


SEMANTIC_NULL_RESULT_V2_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-semantic-null-result/v2"
)
SEMANTIC_NULL_RESULT_V2_ALGORITHM = (
    "within-fixed-valid-label-permutation-and-original-truth-ranking/v2"
)
SEMANTIC_GATE_SUMMARY_V2_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-semantic-engineering-gate-summary/v2"
)
SEMANTIC_GATE_SUMMARY_V2_ALGORITHM = (
    "planned-case-failure-adverse-stratified-development-gates/v2"
)
SEMANTIC_NULL_RNG_DOMAIN = "anatomy-tracker.semantic-null-rng/v2"
DEFAULT_SEMANTIC_NULL_ROOT_SEED = 0x53454D4E554C4C32
PLANE_STRATA = (
    "reference",
    "near_AP",
    "near_DV",
    "near_ML",
    "general_oblique",
    "edge_or_partial",
)
CARDINAL_STRATA = ("near_AP", "near_DV", "near_ML")
FROZEN_PANEL_ANIMAL_COUNT = 4
FROZEN_PANEL_CASE_COUNT = FROZEN_PANEL_ANIMAL_COUNT * len(PLANE_STRATA)
_NO_LEARNED_ASSET_DEPENDENCIES = {
    "learned_checkpoint_dependencies": [],
    "pretrained_feature_dependencies": [],
    "previous_model_dependencies": [],
}
_PRIMARY_SCOPE_CONTRACT = {
    "model_free": True,
    "forced_truth_finite_ranking_premise_only": True,
    "posterior_or_probability_claim": False,
    "calibrated_uncertainty_claim": False,
    "semantic_score_is_probability": False,
    "valid_correspondence_weight_used": False,
    "weighted_sensitivity_status": "separate deferred descriptive artifact",
    "benchmark_or_final_test_claim": False,
}
_NULL_SCOPE_CONTRACT = {
    "development_shape_preserving_negative_control_only": True,
    "model_training_or_benchmark_claim": False,
    "posterior_or_probability_claim": False,
    "calibrated_uncertainty_claim": False,
    "weighted_correspondence_sensitivity_included": False,
    "original_truth_id_is_ranked": True,
}
_NULL_RNG_COORDINATES = (
    "split_index",
    "animal_index",
    "section_index",
    "observation_index",
    "realization_index",
    "null_index",
)
_NULL_RNG_EXCLUSIONS = (
    "split",
    "animal_id",
    "specimen_id",
    "experiment_id",
    "artifact_ids",
    "target_or_candidate_content",
)
_NULL_SCORE_ARRAY_KEYS = {
    "semantic_score_float64",
    "raw_id_agreement_float64",
    "mask_dice_float64",
}
_NULL_CONTROL_NAMES = {
    "numeric_lineage_seed_exclusion",
    "within_fixed_valid_permutation_bijection",
    "label_histogram_and_channel_preservation",
    "fixed_denominator_and_candidate_bank_identity",
}
_NULL_RESULT_KEYS = {
    "schema_version",
    "algorithm",
    "implementation_source_sha256",
    "implementation_source_sha256_canonicalization",
    "runtime_dependencies",
    "asset_dependencies",
    "scope",
    "upstream_reference",
    "provenance",
    "rng_contract",
    "target_reference",
    "candidate_reference",
    "scores",
    "ranking",
    "exact_controls",
    "semantic_null_result_id",
    "receipt_sha256",
}
_SUMMARY_KEYS = {
    "schema_version",
    "algorithm",
    "implementation_source_sha256",
    "implementation_source_sha256_canonicalization",
    "runtime_dependencies",
    "scope",
    "panel_reference",
    "thresholds",
    "case_contributions",
    "metrics",
    "gates",
    "passed",
    "semantic_gate_summary_id",
    "receipt_sha256",
}
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "arbitrary_plane_semantic_oracle_null_gate_v2.py",
    "arbitrary_plane_semantic_oracle_v2.py",
    "arbitrary_plane_semantic_oracle.py",
    "arbitrary_plane_candidate_bank_v2.py",
    "arbitrary_plane_acquisition_v2.py",
)


def _source_hashes() -> dict[str, str]:
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _uint64(value: int | str, name: str) -> int:
    if isinstance(value, str):
        if (
            len(value) != 18
            or not value.startswith("0x")
            or any(character not in "0123456789abcdef" for character in value[2:])
        ):
            raise ValueError(f"{name} must be canonical uint64 hexadecimal")
        return int(value[2:], 16)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer or canonical hexadecimal")
    result = int(value)
    if result < 0 or result >= 1 << 64:
        raise ValueError(f"{name} must fit uint64")
    return result


def _canonical_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be canonical lowercase SHA-256")
    return value


def _exact_no_learned_dependencies(value: object) -> bool:
    return acquisition._json_value(value) == _NO_LEARNED_ASSET_DEPENDENCIES


def _valid_receipt_metadata(
    receipt: object, dtype: np.dtype, shape: tuple[int, ...]
) -> bool:
    if not isinstance(receipt, Mapping):
        return False
    payload = acquisition._json_value(receipt)
    try:
        _canonical_sha256(payload["array_sha256"], "array receipt SHA-256")
    except (KeyError, TypeError, ValueError):
        return False
    return (
        set(payload) == {"dtype", "shape", "array_sha256"}
        and payload["dtype"] == np.dtype(dtype).str
        and payload["shape"] == list(shape)
    )


def _score_block_self_consistent(
    result: Mapping[str, object], expected_keys: set[str]
) -> bool:
    scores = result["scores"]
    arrays = scores["arrays"]
    receipts = scores["array_receipts"]
    return (
        set(arrays) == expected_keys
        and set(receipts) == expected_keys
        and all(
            np.asarray(array).dtype == np.dtype(np.float64)
            and np.asarray(array).shape == (40,)
            and np.isfinite(array).all()
            and np.all((np.asarray(array) >= 0.0) & (np.asarray(array) <= 1.0))
            and acquisition._json_value(receipts[name])
            == acquisition._json_value(acquisition._array_receipt(array))
            for name, array in arrays.items()
        )
        and type(scores["channel_count"]) is int
        and scores["channel_count"] > 0
        and np.isfinite(scores["smoothing_sigma_px"])
        and scores["smoothing_sigma_px"] > 0.0
    )


def derive_semantic_null_seed_v2(
    root_seed: int | str,
    split_index: int,
    animal_index: int,
    section_index: int,
    observation_index: int,
    realization_index: int,
    null_index: int = 0,
) -> int:
    """Derive the null permutation from numeric lineage and nothing else."""
    root = _uint64(root_seed, "semantic null root seed")
    coordinates = tuple(
        _uint64(value, name)
        for value, name in zip(
            (
                split_index,
                animal_index,
                section_index,
                observation_index,
                realization_index,
                null_index,
            ),
            (
                "split_index",
                "animal_index",
                "section_index",
                "observation_index",
                "realization_index",
                "null_index",
            ),
            strict=True,
        )
    )
    components = (
        SEMANTIC_NULL_RNG_DOMAIN,
        SEMANTIC_NULL_RESULT_V2_SCHEMA,
        f"0x{root:016x}",
        *(str(value) for value in coordinates),
    )
    encoded = b"".join(
        len(component.encode("utf-8")).to_bytes(4, "big")
        + component.encode("utf-8")
        for component in components
    )
    return int.from_bytes(
        hashlib.blake2b(encoded, digest_size=8, person=b"AP-NULL-V2").digest(),
        "big",
    )


def _byte_equal(left: object, right: object) -> bool:
    left_array, right_array = np.asarray(left), np.asarray(right)
    return (
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and np.ascontiguousarray(left_array).tobytes(order="C")
        == np.ascontiguousarray(right_array).tobytes(order="C")
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
    arrays = {
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
        array.shape != (40,)
        or not np.isfinite(array).all()
        or np.any((array < 0.0) | (array > 1.0))
        for array in arrays.values()
    ):
        raise ValueError("semantic null score landscapes must be finite forty-vectors")
    return arrays


def _null_identity(result: Mapping[str, object]) -> dict[str, object]:
    return acquisition._json_value(
        {
            key: value
            for key, value in result.items()
            if key not in {"scores", "semantic_null_result_id", "receipt_sha256"}
        }
        | {
            "scores": {
                key: value
                for key, value in result["scores"].items()
                if key != "arrays"
            }
        }
    )


def arbitrary_plane_semantic_null_result_receipt_v2(
    result: Mapping[str, object],
) -> dict[str, object]:
    return {
        "semantic_null_result_id": result["semantic_null_result_id"],
        "identity_payload": _null_identity(result),
    }


def _primary_arrays(
    primary_result: Mapping[str, object],
    candidate_bank: Mapping[str, object],
    final_realization: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[str], str]:
    target = np.asarray(
        final_realization["targets"]["source_label_ground_truth_crop_int64"]
    )
    valid = np.asarray(final_realization["targets"]["valid_correspondence_mask"])
    candidates = list(candidate_bank["candidates"])
    labels = np.stack(
        [np.asarray(item["arrays"]["rendered_annotation_int64"]) for item in candidates]
    )
    ordered_ids = [str(value) for value in candidate_bank["ordered_candidate_ids"]]
    truth_id = str(primary_result["candidate_reference"]["truth_candidate_id"])
    pitch = float(
        primary_result["target_reference"]["pixel_pitch_reference"][
            "selected_pixel_pitch_um"
        ]
    )
    if (
        target.dtype != np.dtype(np.int64)
        or valid.dtype != np.dtype(bool)
        or target.ndim != 2
        or valid.shape != target.shape
        or labels.dtype != np.dtype(np.int64)
        or labels.shape != (40, *target.shape)
        or len(ordered_ids) != 40
        or len(set(ordered_ids)) != 40
        or [str(item["candidate_id"]) for item in candidates] != ordered_ids
        or ordered_ids.count(truth_id) != 1
        or not np.isfinite(pitch)
        or pitch <= 0.0
        or acquisition._json_value(
            primary_result["target_reference"]["labels_receipt"]
        )
        != acquisition._json_value(acquisition._array_receipt(target))
        or acquisition._json_value(
            primary_result["target_reference"]["fixed_valid_mask_receipt"]
        )
        != acquisition._json_value(acquisition._array_receipt(valid))
        or acquisition._json_value(
            primary_result["candidate_reference"]["candidate_label_stack_receipt"]
        )
        != acquisition._json_value(acquisition._array_receipt(labels))
    ):
        raise ValueError("semantic null inputs no longer match the verified primary result")
    return target, valid, labels, pitch, ordered_ids, truth_id


def make_arbitrary_plane_semantic_null_result_v2(
    primary_result: Mapping[str, object],
    candidate_bank: Mapping[str, object],
    pose_truth: Mapping[str, object],
    final_realization: Mapping[str, object],
    prepared_context: Mapping[str, object],
    *,
    null_root_seed: int | str = DEFAULT_SEMANTIC_NULL_ROOT_SEED,
) -> Mapping[str, object]:
    """Permute labels within the fixed-valid mask and rank the original truth ID."""
    primary_v2.verify_arbitrary_plane_semantic_oracle_result_v2(
        primary_result,
        candidate_bank,
        pose_truth,
        final_realization,
        prepared_context,
    )
    target, valid, candidate_labels, pitch, ordered_ids, truth_id = _primary_arrays(
        primary_result, candidate_bank, final_realization
    )
    provenance = final_realization["provenance"]
    coordinate_names = _NULL_RNG_COORDINATES[:-1]
    coordinates = {
        name: _uint64(provenance[name], name) for name in coordinate_names
    }
    root = _uint64(null_root_seed, "semantic null root seed")
    seed = derive_semantic_null_seed_v2(
        root, *(coordinates[name] for name in coordinate_names), 0
    )
    permutation_rng = np.random.Generator(np.random.PCG64DXSM(seed))
    valid_indices = np.flatnonzero(valid.reshape(-1, order="C")).astype(np.int64)
    permutation = np.ascontiguousarray(
        permutation_rng.permutation(len(valid_indices)), dtype=np.int64
    )
    null_target = np.array(target, copy=True, order="C")
    source_values = target.reshape(-1, order="C")[valid_indices]
    null_target.reshape(-1, order="C")[valid_indices] = source_values[permutation]
    changed_count = int(np.count_nonzero(null_target[valid] != target[valid]))
    degenerate = changed_count == 0

    raw_score = primary_v2._score_semantic_arrays_v2(
        null_target, candidate_labels, valid, pitch
    )
    score_arrays = _score_arrays(raw_score)
    ranking = rank_candidate_ids(
        score_arrays["semantic_score_float64"], ordered_ids, truth_id
    )
    ranking["top3"] = bool(ranking["true_rank"] <= 3)
    original_ids, original_counts = np.unique(target[valid], return_counts=True)
    null_ids, null_counts = np.unique(null_target[valid], return_counts=True)
    primary_channels = primary_result["target_reference"]["channel_reference"]
    channel_equal = (
        raw_score["target_large_region_ids"].tolist()
        == list(primary_channels["large_region_ids"])
        and raw_score["target_small_region_ids"].tolist()
        == list(primary_channels["small_pooled_region_ids"])
        and int(raw_score["channel_count"]) == primary_channels["channel_count"]
        and float(raw_score["smoothing_sigma_px"])
        == primary_result["scores"]["smoothing_sigma_px"]
    )
    permutation_equal = (
        permutation.shape == (len(valid_indices),)
        and np.array_equal(np.sort(permutation), np.arange(len(valid_indices)))
        and np.array_equal(null_target[~valid], target[~valid])
        and null_target.shape == target.shape
    )
    histogram_equal = np.array_equal(original_ids, null_ids) and np.array_equal(
        original_counts, null_counts
    )
    primary_valid_receipt = primary_result["target_reference"][
        "fixed_valid_mask_receipt"
    ]
    denominator_equal = (
        acquisition._json_value(primary_valid_receipt)
        == acquisition._json_value(acquisition._array_receipt(valid))
        and acquisition._json_value(
            primary_result["candidate_reference"]["candidate_label_stack_receipt"]
        )
        == acquisition._json_value(acquisition._array_receipt(candidate_labels))
        and ordered_ids == list(primary_result["candidate_reference"]["ordered_candidate_ids"])
    )
    rng_contract = {
        "domain": SEMANTIC_NULL_RNG_DOMAIN,
        "null_root_seed_uint64": f"0x{root:016x}",
        "numeric_coordinates": coordinates | {"null_index": 0},
        "excluded_coordinates": list(_NULL_RNG_EXCLUSIONS),
        "seed_uint64": f"0x{seed:016x}",
        "generator": "numpy.random.PCG64DXSM",
        "redraw_count": 0,
    }
    controls = {
        "numeric_lineage_seed_exclusion": _control_record(
            "numeric_lineage_seed_exclusion",
            set(rng_contract["numeric_coordinates"])
            == {*coordinate_names, "null_index"},
            rng_contract,
        ),
        "within_fixed_valid_permutation_bijection": _control_record(
            "within_fixed_valid_permutation_bijection",
            permutation_equal,
            {
                "valid_index_count": len(valid_indices),
                "valid_indices_receipt": acquisition._array_receipt(valid_indices),
                "permutation_receipt": acquisition._array_receipt(permutation),
                "outside_fixed_valid_unchanged": np.array_equal(
                    null_target[~valid], target[~valid]
                ),
                "redraw_count": 0,
            },
        ),
        "label_histogram_and_channel_preservation": _control_record(
            "label_histogram_and_channel_preservation",
            histogram_equal and channel_equal,
            {
                "original_label_ids": original_ids.tolist(),
                "original_label_counts": original_counts.tolist(),
                "null_label_ids": null_ids.tolist(),
                "null_label_counts": null_counts.tolist(),
                "primary_channel_reference": acquisition._json_value(primary_channels),
                "null_large_region_ids": raw_score["target_large_region_ids"].tolist(),
                "null_small_region_ids": raw_score["target_small_region_ids"].tolist(),
                "null_channel_count": int(raw_score["channel_count"]),
            },
        ),
        "fixed_denominator_and_candidate_bank_identity": _control_record(
            "fixed_denominator_and_candidate_bank_identity",
            denominator_equal,
            {
                "fixed_valid_mask_receipt": acquisition._array_receipt(valid),
                "candidate_label_stack_receipt": acquisition._array_receipt(
                    candidate_labels
                ),
                "candidate_bank_id": candidate_bank["candidate_bank_id"],
                "ordered_candidate_ids": ordered_ids,
                "truth_candidate_id": truth_id,
            },
        ),
    }
    failed = [name for name, record in controls.items() if not record["passed"]]
    if failed:
        raise ValueError(f"semantic-null exact controls failed: {failed}")
    score_receipts = {
        name: acquisition._array_receipt(array)
        for name, array in score_arrays.items()
    }
    result = {
        "schema_version": SEMANTIC_NULL_RESULT_V2_SCHEMA,
        "algorithm": SEMANTIC_NULL_RESULT_V2_ALGORITHM,
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "runtime_dependencies": {"numpy_version": np.__version__},
        "asset_dependencies": acquisition._json_value(
            _NO_LEARNED_ASSET_DEPENDENCIES
        ),
        "scope": acquisition._json_value(_NULL_SCOPE_CONTRACT),
        "upstream_reference": {
            "semantic_oracle_result_id": primary_result["semantic_oracle_result_id"],
            "semantic_oracle_result_receipt_sha256": primary_result["receipt_sha256"],
            "candidate_bank_id": candidate_bank["candidate_bank_id"],
            "candidate_bank_receipt_sha256": candidate_bank["receipt_sha256"],
            "synthetic_realization_id": final_realization["synthetic_realization_id"],
            "synthetic_realization_receipt_sha256": final_realization["receipt_sha256"],
            "finite_plane_pose_truth_id": pose_truth["finite_plane_pose_truth_id"],
            "v2_context_sha256": prepared_context["v2_context_sha256"],
        },
        "provenance": acquisition._json_value(primary_result["provenance"]),
        "rng_contract": rng_contract,
        "target_reference": {
            "shape_h_w": list(target.shape),
            "original_target_labels_receipt": acquisition._array_receipt(target),
            "null_target_labels_receipt": acquisition._array_receipt(null_target),
            "fixed_valid_mask_receipt": acquisition._array_receipt(valid),
            "valid_index_count": len(valid_indices),
            "valid_indices_receipt": acquisition._array_receipt(valid_indices),
            "permutation_receipt": acquisition._array_receipt(permutation),
            "changed_valid_pixel_count": changed_count,
            "changed_valid_pixel_fraction": float(changed_count / len(valid_indices)),
            "degenerate": degenerate,
            "degeneracy_reason": "permutation left all valid label values unchanged"
            if degenerate
            else None,
            "degenerate_policy": "record and retain; never redraw",
            "outside_fixed_valid_unchanged": True,
            "label_histogram_preserved": True,
            "channel_definition_preserved": True,
            "scorer_denominator_preserved": True,
            "primary_channel_reference": acquisition._json_value(primary_channels),
        },
        "candidate_reference": {
            "candidate_count": 40,
            "ordered_candidate_ids": ordered_ids,
            "truth_candidate_id": truth_id,
            "candidate_label_stack_receipt": acquisition._array_receipt(
                candidate_labels
            ),
        },
        "scores": {
            "arrays": score_arrays,
            "array_receipts": score_receipts,
            "channel_count": int(raw_score["channel_count"]),
            "smoothing_sigma_px": float(raw_score["smoothing_sigma_px"]),
        },
        "ranking": acquisition._json_value(ranking),
        "exact_controls": controls,
    }
    result["semantic_null_result_id"] = acquisition._payload_sha256(
        _null_identity(result)
    )
    result["receipt_sha256"] = acquisition._payload_sha256(
        arbitrary_plane_semantic_null_result_receipt_v2(result)
    )
    return acquisition._freeze_value(result)


def replay_arbitrary_plane_semantic_null_result_v2(
    result: Mapping[str, object],
    primary_result: Mapping[str, object],
    candidate_bank: Mapping[str, object],
    pose_truth: Mapping[str, object],
    final_realization: Mapping[str, object],
    prepared_context: Mapping[str, object],
) -> Mapping[str, object]:
    if result.get("schema_version") != SEMANTIC_NULL_RESULT_V2_SCHEMA:
        raise ValueError("unsupported semantic-null v2 result")
    return make_arbitrary_plane_semantic_null_result_v2(
        primary_result,
        candidate_bank,
        pose_truth,
        final_realization,
        prepared_context,
        null_root_seed=result["rng_contract"]["null_root_seed_uint64"],
    )


def _valid_control_records(
    records: Mapping[str, object], expected_names: set[str]
) -> bool:
    if set(records) != expected_names:
        return False
    for name, record in records.items():
        payload = {
            "control": name,
            "passed": record.get("passed"),
            "evidence": acquisition._json_value(record.get("evidence")),
        }
        if (
            set(record)
            != {"control", "passed", "evidence", "evidence_receipt_sha256"}
            or record.get("control") != name
            or record.get("passed") is not True
            or not isinstance(record.get("evidence"), Mapping)
            or not record["evidence"]
            or record.get("evidence_receipt_sha256")
            != acquisition._payload_sha256(payload)
        ):
            return False
    return True


def _primary_reference_contract(result: Mapping[str, object]) -> bool:
    target = result["target_reference"]
    candidates = result["candidate_reference"]
    scorer = result["scorer_input_contract"]
    coverage = result["coverage"]
    channels = target["channel_reference"]
    pitch = target["pixel_pitch_reference"]
    shape = tuple(target["shape_h_w"])
    candidate_ids = list(candidates["ordered_candidate_ids"])
    truth_id = candidates["truth_candidate_id"]
    truth_index = candidates["truth_candidate_index"]
    summaries = list(candidates["candidate_summaries"])
    large_ids = np.asarray(channels["large_region_ids"], dtype=np.int64)
    small_ids = np.asarray(channels["small_pooled_region_ids"], dtype=np.int64)
    pitch_y_x = np.asarray(pitch["pixel_pitch_y_x_um"], dtype=np.float64)
    source_lineage = result["provenance"]["source_lineage"]
    observation = result["provenance"]["observation_and_realization"]
    if (
        set(target)
        != {
            "labels_receipt",
            "fixed_valid_mask_receipt",
            "shape_h_w",
            "fixed_valid_pixel_count",
            "fixed_valid_nonzero_region_id_count",
            "channel_reference",
            "pixel_pitch_reference",
        }
        or set(candidates)
        != {
            "candidate_count",
            "ordered_candidate_ids",
            "truth_candidate_id",
            "truth_candidate_index",
            "candidate_label_stack_receipt",
            "model_grid_reference",
            "candidate_summaries",
        }
        or set(scorer)
        != {
            "allowed_inputs",
            "forbidden_inputs",
            "target_labels_receipt",
            "fixed_valid_mask_receipt",
            "candidate_label_stack_receipt",
            "pixel_pitch_reference",
            "candidate_ids_and_metadata_joined_after_scoring_only",
        }
        or set(coverage)
        != {
            "target_fixed_valid_pixel_count",
            "independent_truth_brain_pixel_count",
            "evaluable",
            "failure_reason",
            "zero_support_truth_policy",
        }
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
        or type(target["fixed_valid_pixel_count"]) is not int
        or not 0 < target["fixed_valid_pixel_count"] <= int(np.prod(shape))
        or type(target["fixed_valid_nonzero_region_id_count"]) is not int
        or target["fixed_valid_nonzero_region_id_count"] <= 0
        or not _valid_receipt_metadata(
            target["labels_receipt"], np.dtype(np.int64), shape
        )
        or not _valid_receipt_metadata(
            target["fixed_valid_mask_receipt"], np.dtype(bool), shape
        )
        or candidates["candidate_count"] != 40
        or len(candidate_ids) != 40
        or any(not isinstance(value, str) or not value for value in candidate_ids)
        or len(set(candidate_ids)) != 40
        or candidate_ids.count(truth_id) != 1
        or type(truth_index) is not int
        or truth_index != candidate_ids.index(truth_id)
        or not _valid_receipt_metadata(
            candidates["candidate_label_stack_receipt"],
            np.dtype(np.int64),
            (40, *shape),
        )
        or len(summaries) != 40
        or [item.get("candidate_id") for item in summaries] != candidate_ids
        or sum(item.get("candidate_class") == "truth" for item in summaries) != 1
        or summaries[truth_index].get("candidate_class") != "truth"
        or tuple(scorer["allowed_inputs"])
        != tuple(primary_v2.NUMERICAL_SCORER_INPUTS)
        or tuple(scorer["forbidden_inputs"])
        != tuple(primary_v2.FORBIDDEN_NUMERICAL_SCORER_INPUTS)
        or scorer["candidate_ids_and_metadata_joined_after_scoring_only"] is not True
        or acquisition._json_value(scorer["target_labels_receipt"])
        != acquisition._json_value(target["labels_receipt"])
        or acquisition._json_value(scorer["fixed_valid_mask_receipt"])
        != acquisition._json_value(target["fixed_valid_mask_receipt"])
        or acquisition._json_value(scorer["candidate_label_stack_receipt"])
        != acquisition._json_value(candidates["candidate_label_stack_receipt"])
        or acquisition._json_value(scorer["pixel_pitch_reference"])
        != acquisition._json_value(pitch)
        or set(channels)
        != {
            "minimum_individual_region_pixels",
            "large_region_ids",
            "small_pooled_region_ids",
            "large_region_ids_receipt",
            "small_pooled_region_ids_receipt",
            "channel_count",
        }
        or channels["minimum_individual_region_pixels"]
        != primary_v2.MINIMUM_INDIVIDUAL_REGION_PIXELS
        or large_ids.ndim != 1
        or small_ids.ndim != 1
        or len(set(large_ids.tolist())) != len(large_ids)
        or len(set(small_ids.tolist())) != len(small_ids)
        or set(large_ids.tolist()) & set(small_ids.tolist())
        or acquisition._json_value(channels["large_region_ids_receipt"])
        != acquisition._array_receipt(large_ids)
        or acquisition._json_value(channels["small_pooled_region_ids_receipt"])
        != acquisition._array_receipt(small_ids)
        or channels["channel_count"] != len(large_ids) + bool(len(small_ids))
        or result["scores"]["channel_count"] != channels["channel_count"]
        or set(pitch)
        != {
            "pixel_pitch_y_x_um",
            "selected_pixel_pitch_um",
            "isotropy_relative_tolerance",
            "crop_window_id",
            "observation_bundle_receipt_sha256",
            "source",
            "model_ouv_pitch_used",
        }
        or pitch_y_x.shape != (2,)
        or not np.isfinite(pitch_y_x).all()
        or np.any(pitch_y_x <= 0.0)
        or not np.isclose(
            pitch_y_x[0], pitch_y_x[1], rtol=primary_v2.PITCH_ISOTROPY_RTOL, atol=0.0
        )
        or pitch["selected_pixel_pitch_um"] != float(pitch_y_x[0])
        or pitch["isotropy_relative_tolerance"] != primary_v2.PITCH_ISOTROPY_RTOL
        or pitch["source"]
        != "authenticated observation crop processed_pixel_pitch_y_x_um"
        or pitch["model_ouv_pitch_used"] is not False
        or type(coverage["target_fixed_valid_pixel_count"]) is not int
        or coverage["target_fixed_valid_pixel_count"]
        != target["fixed_valid_pixel_count"]
        or type(coverage["independent_truth_brain_pixel_count"]) is not int
        or coverage["independent_truth_brain_pixel_count"] < 0
        or coverage["evaluable"]
        is not (coverage["independent_truth_brain_pixel_count"] > 0)
        or coverage["failure_reason"]
        != (
            None
            if coverage["evaluable"]
            else "independent truth atlas render has zero finite-crop brain support"
        )
        or coverage["zero_support_truth_policy"]
        != "save raw ranking, count top1/top3 false, never redraw"
        or summaries[truth_index].get("brain_pixel_count")
        != coverage["independent_truth_brain_pixel_count"]
        or source_lineage["plane_stratum"] not in PLANE_STRATA
        or type(source_lineage["animal_index"]) is not int
        or source_lineage["animal_id"] is None
        or source_lineage["animal_index"] != observation["animal_index"]
        or acquisition._json_value(source_lineage["animal_id"])
        != acquisition._json_value(observation["animal_id"])
        or any(
            type(observation[name]) is not int for name in _NULL_RNG_COORDINATES[:-1]
        )
    ):
        return False
    return True


def _null_reference_contract(result: Mapping[str, object]) -> bool:
    rng = result["rng_contract"]
    target = result["target_reference"]
    candidates = result["candidate_reference"]
    coordinates = rng["numeric_coordinates"]
    shape = tuple(target["shape_h_w"])
    valid_count = target["valid_index_count"]
    changed_count = target["changed_valid_pixel_count"]
    candidate_ids = list(candidates["ordered_candidate_ids"])
    observation = result["provenance"]["observation_and_realization"]
    root = _uint64(rng["null_root_seed_uint64"], "semantic null root seed")
    if (
        set(rng)
        != {
            "domain",
            "null_root_seed_uint64",
            "numeric_coordinates",
            "excluded_coordinates",
            "seed_uint64",
            "generator",
            "redraw_count",
        }
        or rng["domain"] != SEMANTIC_NULL_RNG_DOMAIN
        or rng["null_root_seed_uint64"] != f"0x{root:016x}"
        or set(coordinates) != set(_NULL_RNG_COORDINATES)
        or any(type(coordinates[name]) is not int for name in _NULL_RNG_COORDINATES)
        or coordinates["null_index"] != 0
        or tuple(rng["excluded_coordinates"]) != _NULL_RNG_EXCLUSIONS
        or rng["seed_uint64"]
        != f"0x{derive_semantic_null_seed_v2(root, *(coordinates[name] for name in _NULL_RNG_COORDINATES)):016x}"
        or rng["generator"] != "numpy.random.PCG64DXSM"
        or rng["redraw_count"] != 0
        or any(
            observation[name] != coordinates[name]
            for name in _NULL_RNG_COORDINATES[:-1]
        )
        or set(target)
        != {
            "shape_h_w",
            "original_target_labels_receipt",
            "null_target_labels_receipt",
            "fixed_valid_mask_receipt",
            "valid_index_count",
            "valid_indices_receipt",
            "permutation_receipt",
            "changed_valid_pixel_count",
            "changed_valid_pixel_fraction",
            "degenerate",
            "degeneracy_reason",
            "degenerate_policy",
            "outside_fixed_valid_unchanged",
            "label_histogram_preserved",
            "channel_definition_preserved",
            "scorer_denominator_preserved",
            "primary_channel_reference",
        }
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
        or type(valid_count) is not int
        or not 0 < valid_count <= int(np.prod(shape))
        or not _valid_receipt_metadata(
            target["original_target_labels_receipt"], np.dtype(np.int64), shape
        )
        or not _valid_receipt_metadata(
            target["null_target_labels_receipt"], np.dtype(np.int64), shape
        )
        or not _valid_receipt_metadata(
            target["fixed_valid_mask_receipt"], np.dtype(bool), shape
        )
        or not _valid_receipt_metadata(
            target["valid_indices_receipt"], np.dtype(np.int64), (valid_count,)
        )
        or not _valid_receipt_metadata(
            target["permutation_receipt"], np.dtype(np.int64), (valid_count,)
        )
        or type(changed_count) is not int
        or not 0 <= changed_count <= valid_count
        or target["changed_valid_pixel_fraction"] != changed_count / valid_count
        or target["degenerate"] is not (changed_count == 0)
        or target["degeneracy_reason"]
        != (
            "permutation left all valid label values unchanged"
            if changed_count == 0
            else None
        )
        or target["degenerate_policy"] != "record and retain; never redraw"
        or target["outside_fixed_valid_unchanged"] is not True
        or target["label_histogram_preserved"] is not True
        or target["channel_definition_preserved"] is not True
        or target["scorer_denominator_preserved"] is not True
        or not isinstance(target["primary_channel_reference"], Mapping)
        or result["scores"]["channel_count"]
        != target["primary_channel_reference"]["channel_count"]
        or set(candidates)
        != {
            "candidate_count",
            "ordered_candidate_ids",
            "truth_candidate_id",
            "candidate_label_stack_receipt",
        }
        or candidates["candidate_count"] != 40
        or len(candidate_ids) != 40
        or any(not isinstance(value, str) or not value for value in candidate_ids)
        or len(set(candidate_ids)) != 40
        or candidate_ids.count(candidates["truth_candidate_id"]) != 1
        or not _valid_receipt_metadata(
            candidates["candidate_label_stack_receipt"],
            np.dtype(np.int64),
            (40, *shape),
        )
    ):
        return False
    return True


def _null_control_evidence_consistent(result: Mapping[str, object]) -> bool:
    controls = result["exact_controls"]
    rng = result["rng_contract"]
    target = result["target_reference"]
    candidates = result["candidate_reference"]
    permutation = controls["within_fixed_valid_permutation_bijection"]["evidence"]
    histogram = controls["label_histogram_and_channel_preservation"]["evidence"]
    denominator = controls["fixed_denominator_and_candidate_bank_identity"]["evidence"]
    original_ids = list(histogram["original_label_ids"])
    original_counts = list(histogram["original_label_counts"])
    null_ids = list(histogram["null_label_ids"])
    null_counts = list(histogram["null_label_counts"])
    return (
        acquisition._json_value(
            controls["numeric_lineage_seed_exclusion"]["evidence"]
        )
        == acquisition._json_value(rng)
        and set(permutation)
        == {
            "valid_index_count",
            "valid_indices_receipt",
            "permutation_receipt",
            "outside_fixed_valid_unchanged",
            "redraw_count",
        }
        and permutation["valid_index_count"] == target["valid_index_count"]
        and acquisition._json_value(permutation["valid_indices_receipt"])
        == acquisition._json_value(target["valid_indices_receipt"])
        and acquisition._json_value(permutation["permutation_receipt"])
        == acquisition._json_value(target["permutation_receipt"])
        and permutation["outside_fixed_valid_unchanged"] is True
        and permutation["redraw_count"] == 0
        and original_ids == null_ids
        and original_counts == null_counts
        and len(original_ids) == len(original_counts)
        and len(set(original_ids)) == len(original_ids)
        and all(type(value) is int for value in original_ids)
        and all(type(value) is int and value > 0 for value in original_counts)
        and sum(original_counts) == target["valid_index_count"]
        and acquisition._json_value(histogram["primary_channel_reference"])
        == acquisition._json_value(target["primary_channel_reference"])
        and list(histogram["null_large_region_ids"])
        == list(target["primary_channel_reference"]["large_region_ids"])
        and list(histogram["null_small_region_ids"])
        == list(target["primary_channel_reference"]["small_pooled_region_ids"])
        and histogram["null_channel_count"]
        == target["primary_channel_reference"]["channel_count"]
        and acquisition._json_value(denominator["fixed_valid_mask_receipt"])
        == acquisition._json_value(target["fixed_valid_mask_receipt"])
        and acquisition._json_value(denominator["candidate_label_stack_receipt"])
        == acquisition._json_value(candidates["candidate_label_stack_receipt"])
        and denominator["candidate_bank_id"]
        == result["upstream_reference"]["candidate_bank_id"]
        and list(denominator["ordered_candidate_ids"])
        == list(candidates["ordered_candidate_ids"])
        and denominator["truth_candidate_id"] == candidates["truth_candidate_id"]
    )


def _null_self_audit(result: Mapping[str, object]) -> dict[str, bool]:
    audit = {"structure": False, "receipt": False, "controls": False, "ranking": False}
    try:
        arrays = result["scores"]["arrays"]
        audit["structure"] = (
            set(result) == _NULL_RESULT_KEYS
            and result["schema_version"] == SEMANTIC_NULL_RESULT_V2_SCHEMA
            and result["algorithm"] == SEMANTIC_NULL_RESULT_V2_ALGORITHM
            and acquisition._json_value(result["implementation_source_sha256"])
            == _source_hashes()
            and result["implementation_source_sha256_canonicalization"]
            == acquisition.V2_SOURCE_SHA256_CANONICALIZATION
            and _exact_no_learned_dependencies(result["asset_dependencies"])
            and acquisition._json_value(result["scope"]) == _NULL_SCOPE_CONTRACT
            and _score_block_self_consistent(result, _NULL_SCORE_ARRAY_KEYS)
            and _null_reference_contract(result)
        )
        audit["receipt"] = (
            result["semantic_null_result_id"]
            == acquisition._payload_sha256(_null_identity(result))
            and result["receipt_sha256"]
            == acquisition._payload_sha256(
                arbitrary_plane_semantic_null_result_receipt_v2(result)
            )
        )
        audit["controls"] = _valid_control_records(
            result["exact_controls"], _NULL_CONTROL_NAMES
        ) and _null_control_evidence_consistent(result)
        derived = rank_candidate_ids(
            np.asarray(arrays["semantic_score_float64"]),
            list(result["candidate_reference"]["ordered_candidate_ids"]),
            result["candidate_reference"]["truth_candidate_id"],
        )
        audit["ranking"] = (
            set(result["ranking"]) == {*derived, "top3"}
            and all(
                acquisition._json_value(result["ranking"][name])
                == acquisition._json_value(value)
                for name, value in derived.items()
            )
            and result["ranking"]["top3"] is (derived["true_rank"] <= 3)
        )
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        pass
    return audit


def verify_arbitrary_plane_semantic_null_result_v2(
    result: Mapping[str, object],
    primary_result: Mapping[str, object],
    candidate_bank: Mapping[str, object],
    pose_truth: Mapping[str, object],
    final_realization: Mapping[str, object],
    prepared_context: Mapping[str, object],
) -> None:
    audit = _null_self_audit(result)
    if not all(audit.values()):
        raise ValueError(f"semantic-null structure, receipt, control, or ranking changed: {audit}")
    expected = replay_arbitrary_plane_semantic_null_result_v2(
        result,
        primary_result,
        candidate_bank,
        pose_truth,
        final_realization,
        prepared_context,
    )
    if acquisition._json_value(
        arbitrary_plane_semantic_null_result_receipt_v2(result)
    ) != acquisition._json_value(
        arbitrary_plane_semantic_null_result_receipt_v2(expected)
    ) or any(
        not _byte_equal(result["scores"]["arrays"][name], expected["scores"]["arrays"][name])
        for name in _NULL_SCORE_ARRAY_KEYS
    ):
        raise ValueError("semantic-null deterministic replay changed")


def _primary_self_audit(result: Mapping[str, object]) -> dict[str, bool]:
    audit = {"structure": False, "receipt": False, "controls": False, "ranking": False}
    try:
        arrays = result["scores"]["arrays"]
        audit["structure"] = (
            set(result) == primary_v2._RESULT_KEYS
            and result["schema_version"] == primary_v2.SEMANTIC_ORACLE_RESULT_V2_SCHEMA
            and result["algorithm"] == primary_v2.SEMANTIC_ORACLE_RESULT_V2_ALGORITHM
            and acquisition._json_value(result["implementation_source_sha256"])
            == primary_v2._source_hashes()
            and result["implementation_source_sha256_canonicalization"]
            == acquisition.V2_SOURCE_SHA256_CANONICALIZATION
            and _exact_no_learned_dependencies(result["asset_dependencies"])
            and acquisition._json_value(result["scope"]) == _PRIMARY_SCOPE_CONTRACT
            and _score_block_self_consistent(result, primary_v2._SCORE_ARRAY_KEYS)
            and _primary_reference_contract(result)
        )
        audit["receipt"] = (
            result["semantic_oracle_result_id"]
            == acquisition._payload_sha256(primary_v2._identity_payload(result))
            and result["receipt_sha256"]
            == acquisition._payload_sha256(
                primary_v2.arbitrary_plane_semantic_oracle_result_receipt_v2(result)
            )
        )
        audit["controls"] = _valid_control_records(
            result["exact_controls"], set(primary_v2.EXACT_CONTROL_NAMES)
        )
        derived = rank_candidate_ids(
            np.asarray(arrays["semantic_score_float64"]),
            list(result["candidate_reference"]["ordered_candidate_ids"]),
            result["candidate_reference"]["truth_candidate_id"],
        )
        coverage = bool(result["coverage"]["evaluable"])
        raw_top3 = derived["true_rank"] <= 3
        expected_ranking_keys = {
            *derived,
            "top3",
            "raw_top1_before_coverage_policy",
            "raw_top3_before_coverage_policy",
            "coverage_adjusted_top1",
            "coverage_adjusted_top3",
        }
        audit["ranking"] = (
            set(result["ranking"]) == expected_ranking_keys
            and all(
                acquisition._json_value(result["ranking"][name])
                == acquisition._json_value(value)
                for name, value in derived.items()
                if name != "top1"
            )
            and result["ranking"]["raw_top1_before_coverage_policy"]
            is bool(derived["top1"])
            and result["ranking"]["raw_top3_before_coverage_policy"] is raw_top3
            and result["ranking"]["top1"] is (coverage and derived["top1"])
            and result["ranking"]["top3"] is (coverage and raw_top3)
            and result["ranking"]["coverage_adjusted_top1"]
            is (coverage and derived["top1"])
            and result["ranking"]["coverage_adjusted_top3"]
            is (coverage and raw_top3)
        )
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        pass
    return audit


def _primary_null_binding_self_consistent(
    primary: Mapping[str, object], null: Mapping[str, object]
) -> bool:
    try:
        primary_upstream = primary["upstream_reference"]
        null_upstream = null["upstream_reference"]
        primary_target = primary["target_reference"]
        null_target = null["target_reference"]
        primary_candidates = primary["candidate_reference"]
        null_candidates = null["candidate_reference"]
        observation = primary["provenance"]["observation_and_realization"]
        coordinates = null["rng_contract"]["numeric_coordinates"]
        return (
            acquisition._json_value(null["provenance"])
            == acquisition._json_value(primary["provenance"])
            and null_upstream["semantic_oracle_result_id"]
            == primary["semantic_oracle_result_id"]
            and null_upstream["semantic_oracle_result_receipt_sha256"]
            == primary["receipt_sha256"]
            and null_upstream["candidate_bank_id"]
            == primary_upstream["candidate_bank_id"]
            and null_upstream["candidate_bank_receipt_sha256"]
            == primary_upstream["candidate_bank_receipt_sha256"]
            and null_upstream["synthetic_realization_id"]
            == primary_upstream["synthetic_realization_id"]
            and null_upstream["synthetic_realization_receipt_sha256"]
            == primary_upstream["synthetic_realization_receipt_sha256"]
            and null_upstream["finite_plane_pose_truth_id"]
            == primary_upstream["finite_plane_pose_truth_id"]
            and null_upstream["v2_context_sha256"]
            == primary_upstream["v2_context_sha256"]
            and acquisition._json_value(null_target["original_target_labels_receipt"])
            == acquisition._json_value(primary_target["labels_receipt"])
            and acquisition._json_value(null_target["fixed_valid_mask_receipt"])
            == acquisition._json_value(primary_target["fixed_valid_mask_receipt"])
            and tuple(null_target["shape_h_w"]) == tuple(primary_target["shape_h_w"])
            and acquisition._json_value(null_target["primary_channel_reference"])
            == acquisition._json_value(primary_target["channel_reference"])
            and null_target["valid_index_count"]
            == primary_target["fixed_valid_pixel_count"]
            and null_candidates["candidate_count"]
            == primary_candidates["candidate_count"]
            and tuple(null_candidates["ordered_candidate_ids"])
            == tuple(primary_candidates["ordered_candidate_ids"])
            and null_candidates["truth_candidate_id"]
            == primary_candidates["truth_candidate_id"]
            and acquisition._json_value(
                null_candidates["candidate_label_stack_receipt"]
            )
            == acquisition._json_value(
                primary_candidates["candidate_label_stack_receipt"]
            )
            and null["scores"]["channel_count"]
            == primary["scores"]["channel_count"]
            and null["scores"]["smoothing_sigma_px"]
            == primary["scores"]["smoothing_sigma_px"]
            and all(
                coordinates[name] == observation[name]
                for name in _NULL_RNG_COORDINATES[:-1]
            )
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _case_metrics(cases: list[dict[str, object]]) -> dict[str, object]:
    count = len(cases)
    if count == 0:
        return {
            "scheduled_count": 0,
            "evaluable_count": 0,
            "primary_top1_count": 0,
            "primary_top1_rate": None,
            "primary_top3_count": 0,
            "primary_top3_rate": None,
            "median_coverage_adjusted_rank": None,
            "median_coverage_adjusted_win_fraction": None,
            "null_original_truth_top1_count": 0,
            "null_original_truth_top1_rate": None,
            "null_original_truth_mean_reciprocal_rank": None,
            "execution_complete_rate": None,
            "artifact_contract_self_consistent_rate": None,
            "exact_controls_complete_rate": None,
            "combined_complete_rate": None,
        }
    return {
        "scheduled_count": count,
        "evaluable_count": sum(bool(case["evaluable"]) for case in cases),
        "primary_top1_count": sum(bool(case["primary_top1"]) for case in cases),
        "primary_top1_rate": float(
            np.mean([bool(case["primary_top1"]) for case in cases])
        ),
        "primary_top3_count": sum(bool(case["primary_top3"]) for case in cases),
        "primary_top3_rate": float(
            np.mean([bool(case["primary_top3"]) for case in cases])
        ),
        "median_coverage_adjusted_rank": float(
            np.median([int(case["primary_rank"]) for case in cases])
        ),
        "median_coverage_adjusted_win_fraction": float(
            np.median([float(case["primary_win_fraction"]) for case in cases])
        ),
        "null_original_truth_top1_count": sum(
            bool(case["null_top1"]) for case in cases
        ),
        "null_original_truth_top1_rate": float(
            np.mean([bool(case["null_top1"]) for case in cases])
        ),
        "null_original_truth_mean_reciprocal_rank": float(
            np.mean([float(case["null_reciprocal_rank"]) for case in cases])
        ),
        "execution_complete_rate": float(
            np.mean([bool(case["execution_complete"]) for case in cases])
        ),
        "artifact_contract_self_consistent_rate": float(
            np.mean(
                [bool(case["artifact_contract_self_consistent"]) for case in cases]
            )
        ),
        "exact_controls_complete_rate": float(
            np.mean([bool(case["exact_controls_complete"]) for case in cases])
        ),
        "combined_complete_rate": float(
            np.mean([bool(case["combined_complete"]) for case in cases])
        ),
    }


def _summary_identity(summary: Mapping[str, object]) -> dict[str, object]:
    return acquisition._json_value(
        {
            key: value
            for key, value in summary.items()
            if key not in {"semantic_gate_summary_id", "receipt_sha256"}
        }
    )


def arbitrary_plane_semantic_gate_summary_receipt_v2(
    summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "semantic_gate_summary_id": summary["semantic_gate_summary_id"],
        "identity_payload": _summary_identity(summary),
    }


def _planned_cases(plan: list[Mapping[str, object]]) -> list[dict[str, object]]:
    if len(plan) != FROZEN_PANEL_CASE_COUNT:
        raise ValueError(
            f"frozen semantic panel must schedule exactly {FROZEN_PANEL_CASE_COUNT} cases"
        )
    normalized = []
    for item in plan:
        if set(item) != {"case_id", "plane_stratum", "animal_index", "animal_id"}:
            raise ValueError("each planned case must have the exact four-field schema")
        case_id = item["case_id"]
        stratum = item["plane_stratum"]
        animal_index = _uint64(item["animal_index"], "planned animal_index")
        animal_id = acquisition._json_value(item["animal_id"])
        if (
            not isinstance(case_id, str)
            or not case_id
            or stratum not in PLANE_STRATA
            or animal_id is None
            or animal_id == ""
        ):
            raise ValueError("planned case identity, stratum, or animal is invalid")
        normalized.append(
            {
                "case_id": case_id,
                "plane_stratum": stratum,
                "animal_index": animal_index,
                "animal_id": animal_id,
            }
        )
    if not normalized or len({item["case_id"] for item in normalized}) != len(normalized):
        raise ValueError("planned case IDs must be nonempty and unique")
    index_to_id: dict[int, str] = {}
    id_to_index: dict[str, int] = {}
    strata_by_animal: dict[tuple[int, str], Counter[str]] = {}
    for item in normalized:
        animal_id_receipt = acquisition._payload_sha256(
            {"animal_id": acquisition._json_value(item["animal_id"])}
        )
        animal_index = item["animal_index"]
        if (
            animal_index in index_to_id
            and index_to_id[animal_index] != animal_id_receipt
        ) or (
            animal_id_receipt in id_to_index
            and id_to_index[animal_id_receipt] != animal_index
        ):
            raise ValueError("animal_index and animal_id must have a one-to-one mapping")
        index_to_id[animal_index] = animal_id_receipt
        id_to_index[animal_id_receipt] = animal_index
        key = (animal_index, animal_id_receipt)
        strata_by_animal.setdefault(key, Counter())[item["plane_stratum"]] += 1
    exact_strata = Counter({stratum: 1 for stratum in PLANE_STRATA})
    if (
        len(index_to_id) != FROZEN_PANEL_ANIMAL_COUNT
        or len(id_to_index) != FROZEN_PANEL_ANIMAL_COUNT
        or len(strata_by_animal) != FROZEN_PANEL_ANIMAL_COUNT
        or any(counts != exact_strata for counts in strata_by_animal.values())
        or Counter(item["plane_stratum"] for item in normalized)
        != Counter({stratum: FROZEN_PANEL_ANIMAL_COUNT for stratum in PLANE_STRATA})
    ):
        raise ValueError(
            "frozen semantic panel requires four animals with every stratum exactly once"
        )
    return normalized


def make_arbitrary_plane_semantic_gate_summary_v2(
    planned_cases: list[Mapping[str, object]],
    case_records: list[Mapping[str, object]],
    *,
    panel_id: str,
) -> Mapping[str, object]:
    """Aggregate every planned case with failure-adverse engineering metrics."""
    panel_id = _canonical_sha256(panel_id, "semantic engineering panel_id")
    plan = _planned_cases(planned_cases)
    records: dict[str, Mapping[str, object]] = {}
    for record in case_records:
        if set(record) != {
            "case_id",
            "status",
            "primary_result",
            "null_result",
            "failure",
        }:
            raise ValueError("semantic case record has the wrong schema")
        case_id = record["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("semantic case record ID must be a nonempty string")
        if case_id in records:
            raise ValueError("semantic case records contain a duplicate case ID")
        records[case_id] = record
    planned_ids = {item["case_id"] for item in plan}
    if not set(records) <= planned_ids:
        raise ValueError("semantic case records contain an unplanned case")

    contributions = []
    for planned in plan:
        case_id = planned["case_id"]
        record = records.get(case_id)
        missing = record is None
        status = "execution_failure" if missing else record["status"]
        if status not in {"completed", "execution_failure"}:
            raise ValueError("semantic case status is invalid")
        primary_result = None if missing else record["primary_result"]
        null_result = None if missing else record["null_result"]
        failure = (
            {"stage": "case-record", "reason": "planned case has no case record"}
            if missing
            else acquisition._json_value(record["failure"])
        )
        if status == "execution_failure":
            if not missing and (
                primary_result is not None
                or null_result is not None
                or not isinstance(failure, dict)
                or not failure
            ):
                raise ValueError("execution failure record is incomplete or contains results")
        elif (
            not isinstance(primary_result, Mapping)
            or not isinstance(null_result, Mapping)
            or failure is not None
        ):
            raise ValueError("completed case must contain primary and null results only")

        primary_audit = (
            _primary_self_audit(primary_result)
            if isinstance(primary_result, Mapping)
            else {name: False for name in ("structure", "receipt", "controls", "ranking")}
        )
        null_audit = (
            _null_self_audit(null_result)
            if isinstance(null_result, Mapping)
            else {name: False for name in ("structure", "receipt", "controls", "ranking")}
        )
        execution_complete = status == "completed"
        lineage_equal = False
        binding_equal = False
        if execution_complete:
            try:
                lineage = primary_result["provenance"]["source_lineage"]
                lineage_equal = (
                    lineage["plane_stratum"] == planned["plane_stratum"]
                    and lineage["animal_index"] == planned["animal_index"]
                    and acquisition._json_value(lineage["animal_id"])
                    == acquisition._json_value(planned["animal_id"])
                )
                binding_equal = _primary_null_binding_self_consistent(
                    primary_result, null_result
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
        artifact_contract_self_consistent = (
            execution_complete
            and primary_audit["structure"]
            and primary_audit["receipt"]
            and primary_audit["ranking"]
            and null_audit["structure"]
            and null_audit["receipt"]
            and null_audit["ranking"]
            and lineage_equal
            and binding_equal
        )
        controls_complete = (
            execution_complete
            and primary_audit["controls"]
            and null_audit["controls"]
        )
        combined_complete = (
            execution_complete
            and artifact_contract_self_consistent
            and controls_complete
        )
        evaluable = bool(combined_complete and primary_result["coverage"]["evaluable"])
        primary_top1 = bool(
            evaluable and primary_result["ranking"]["coverage_adjusted_top1"]
        )
        primary_top3 = bool(
            evaluable and primary_result["ranking"]["coverage_adjusted_top3"]
        )
        primary_rank = int(primary_result["ranking"]["true_rank"]) if evaluable else 40
        primary_win = (
            float(primary_result["ranking"]["true_versus_decoy_win_fraction"])
            if evaluable
            else 0.0
        )
        null_top1 = bool(null_result["ranking"]["top1"]) if combined_complete else True
        null_rr = (
            float(null_result["ranking"]["reciprocal_rank"])
            if combined_complete
            else 1.0
        )
        contributions.append(
            {
                **planned,
                "status": status,
                "failure": failure,
                "primary_result_id": primary_result.get("semantic_oracle_result_id")
                if isinstance(primary_result, Mapping)
                else None,
                "primary_result_receipt_sha256": primary_result.get("receipt_sha256")
                if isinstance(primary_result, Mapping)
                else None,
                "null_result_id": null_result.get("semantic_null_result_id")
                if isinstance(null_result, Mapping)
                else None,
                "null_result_receipt_sha256": null_result.get("receipt_sha256")
                if isinstance(null_result, Mapping)
                else None,
                "execution_complete": execution_complete,
                "artifact_contract_self_consistent": artifact_contract_self_consistent,
                "exact_controls_complete": controls_complete,
                "combined_complete": combined_complete,
                "evaluable": evaluable,
                "primary_top1": primary_top1,
                "primary_top3": primary_top3,
                "primary_rank": primary_rank,
                "primary_win_fraction": primary_win,
                "null_top1": null_top1,
                "null_reciprocal_rank": null_rr,
            }
        )

    by_stratum = {
        stratum: _case_metrics(
            [case for case in contributions if case["plane_stratum"] == stratum]
        )
        for stratum in PLANE_STRATA
    }
    animal_groups: dict[tuple[int, str], list[dict[str, object]]] = {}
    animal_references: dict[tuple[int, str], object] = {}
    for case in contributions:
        animal_hash = acquisition._payload_sha256(
            {"animal_id": acquisition._json_value(case["animal_id"])}
        )
        key = (int(case["animal_index"]), animal_hash)
        animal_groups.setdefault(key, []).append(case)
        animal_references[key] = case["animal_id"]
    by_animal = [
        {
            "animal_index": key[0],
            "animal_id": acquisition._json_value(animal_references[key]),
            "animal_id_receipt_sha256": key[1],
            "metrics": _case_metrics(animal_groups[key]),
        }
        for key in sorted(animal_groups)
    ]
    grouped_cases = {
        "reference": [
            case for case in contributions if case["plane_stratum"] == "reference"
        ],
        "pooled_cardinal": [
            case for case in contributions if case["plane_stratum"] in CARDINAL_STRATA
        ],
        "general_oblique": [
            case
            for case in contributions
            if case["plane_stratum"] == "general_oblique"
        ],
        "edge_or_partial": [
            case
            for case in contributions
            if case["plane_stratum"] == "edge_or_partial"
        ],
    }
    group_metrics = {name: _case_metrics(cases) for name, cases in grouped_cases.items()}
    overall = _case_metrics(contributions)
    thresholds = {
        "reference": {
            "primary_top1_rate_minimum": 0.80,
            "median_coverage_adjusted_rank_maximum": 1.0,
            "median_coverage_adjusted_win_fraction_minimum": 0.95,
        },
        "pooled_cardinal": {
            "primary_top1_rate_minimum": 0.60,
            "primary_top3_rate_minimum": 0.85,
        },
        "general_oblique": {
            "primary_top1_rate_minimum": 0.60,
            "primary_top3_rate_minimum": 0.85,
        },
        "edge_or_partial": {
            "primary_top1_rate_minimum": 0.60,
            "primary_top3_rate_minimum": 0.80,
            "median_coverage_adjusted_rank_maximum": 3.0,
        },
        "shape_preserving_null": {
            "original_truth_top1_rate_maximum": 0.10,
            "original_truth_mean_reciprocal_rank_maximum": 0.15,
        },
        "completeness": {
            "execution_rate_required": 1.0,
            "artifact_contract_self_consistent_rate_required": 1.0,
            "exact_controls_rate_required": 1.0,
            "combined_rate_required": 1.0,
        },
    }

    def nonempty(name: str) -> bool:
        return group_metrics[name]["scheduled_count"] > 0

    gates = {
        "reference": bool(
            nonempty("reference")
            and group_metrics["reference"]["primary_top1_rate"] >= 0.80
            and group_metrics["reference"]["median_coverage_adjusted_rank"] <= 1.0
            and group_metrics["reference"][
                "median_coverage_adjusted_win_fraction"
            ]
            >= 0.95
        ),
        "pooled_cardinal": bool(
            nonempty("pooled_cardinal")
            and group_metrics["pooled_cardinal"]["primary_top1_rate"] >= 0.60
            and group_metrics["pooled_cardinal"]["primary_top3_rate"] >= 0.85
        ),
        "general_oblique": bool(
            nonempty("general_oblique")
            and group_metrics["general_oblique"]["primary_top1_rate"] >= 0.60
            and group_metrics["general_oblique"]["primary_top3_rate"] >= 0.85
        ),
        "edge_or_partial": bool(
            nonempty("edge_or_partial")
            and group_metrics["edge_or_partial"]["primary_top1_rate"] >= 0.60
            and group_metrics["edge_or_partial"]["primary_top3_rate"] >= 0.80
            and group_metrics["edge_or_partial"][
                "median_coverage_adjusted_rank"
            ]
            <= 3.0
        ),
        "shape_preserving_null": bool(
            overall["null_original_truth_top1_rate"] <= 0.10
            and overall["null_original_truth_mean_reciprocal_rank"] <= 0.15
        ),
        "execution_receipt_control_completeness": bool(
            overall["execution_complete_rate"] == 1.0
            and overall["artifact_contract_self_consistent_rate"] == 1.0
            and overall["exact_controls_complete_rate"] == 1.0
            and overall["combined_complete_rate"] == 1.0
        ),
    }
    plan_payload = {"panel_id": panel_id, "planned_cases": plan}
    summary = {
        "schema_version": SEMANTIC_GATE_SUMMARY_V2_SCHEMA,
        "algorithm": SEMANTIC_GATE_SUMMARY_V2_ALGORITHM,
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "runtime_dependencies": {"numpy_version": np.__version__},
        "scope": {
            "small_development_engineering_premise_only": True,
            "benchmark_or_final_test_claim": False,
            "posterior_or_probability_claim": False,
            "inferential_case_level_confidence_intervals_reported": False,
            "animal_is_preserved_as_future_statistical_unit": True,
            "weighted_correspondence_sensitivity_included": False,
            "aggregate_artifact_checks_are_self_consistency_only": True,
            "artifact_hashes_are_not_authentication_against_hostile_coherent_resigning": True,
            "scientific_qualification_requires_live_deterministic_primary_and_null_verification": True,
            "intended_panel_runner_performs_live_verification_at_case_creation_and_strict_replay": True,
            "failure_adverse_imputation": {
                "primary_top1_top3": False,
                "primary_rank": 40,
                "primary_win_fraction": 0.0,
                "null_top1": True,
                "null_reciprocal_rank": 1.0,
            },
        },
        "panel_reference": {
            **plan_payload,
            "planned_case_count": len(plan),
            "planned_panel_receipt_sha256": acquisition._payload_sha256(plan_payload),
            "missing_case_record_policy": "retain as execution failure",
        },
        "thresholds": thresholds,
        "case_contributions": contributions,
        "metrics": {
            "overall": overall,
            "gate_groups": group_metrics,
            "by_stratum": by_stratum,
            "by_synthetic_animal": by_animal,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    summary["semantic_gate_summary_id"] = acquisition._payload_sha256(
        _summary_identity(summary)
    )
    summary["receipt_sha256"] = acquisition._payload_sha256(
        arbitrary_plane_semantic_gate_summary_receipt_v2(summary)
    )
    return acquisition._freeze_value(summary)


def replay_arbitrary_plane_semantic_gate_summary_v2(
    summary: Mapping[str, object],
    planned_cases: list[Mapping[str, object]],
    case_records: list[Mapping[str, object]],
    *,
    expected_panel_id: str,
) -> Mapping[str, object]:
    expected_panel_id = _canonical_sha256(
        expected_panel_id, "expected semantic engineering panel ID"
    )
    if summary.get("schema_version") != SEMANTIC_GATE_SUMMARY_V2_SCHEMA:
        raise ValueError("unsupported semantic engineering-gate summary")
    if summary.get("panel_reference", {}).get("panel_id") != expected_panel_id:
        raise ValueError("semantic engineering panel ID differs from external expectation")
    return make_arbitrary_plane_semantic_gate_summary_v2(
        planned_cases,
        case_records,
        panel_id=expected_panel_id,
    )


def verify_arbitrary_plane_semantic_gate_summary_v2(
    summary: Mapping[str, object],
    planned_cases: list[Mapping[str, object]],
    case_records: list[Mapping[str, object]],
    *,
    expected_panel_id: str,
) -> None:
    expected_panel_id = _canonical_sha256(
        expected_panel_id, "expected semantic engineering panel ID"
    )
    if (
        set(summary) != _SUMMARY_KEYS
        or summary.get("schema_version") != SEMANTIC_GATE_SUMMARY_V2_SCHEMA
        or summary.get("algorithm") != SEMANTIC_GATE_SUMMARY_V2_ALGORITHM
        or acquisition._json_value(summary.get("implementation_source_sha256"))
        != _source_hashes()
        or summary.get("implementation_source_sha256_canonicalization")
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or summary.get("scope", {}).get("benchmark_or_final_test_claim") is not False
        or summary.get("scope", {}).get("posterior_or_probability_claim") is not False
        or summary.get("scope", {}).get("weighted_correspondence_sensitivity_included")
        is not False
        or summary.get("scope", {}).get(
            "aggregate_artifact_checks_are_self_consistency_only"
        )
        is not True
        or summary.get("scope", {}).get(
            "artifact_hashes_are_not_authentication_against_hostile_coherent_resigning"
        )
        is not True
        or summary.get("scope", {}).get(
            "scientific_qualification_requires_live_deterministic_primary_and_null_verification"
        )
        is not True
        or summary.get("scope", {}).get(
            "intended_panel_runner_performs_live_verification_at_case_creation_and_strict_replay"
        )
        is not True
        or summary.get("panel_reference", {}).get("panel_id") != expected_panel_id
        or summary["semantic_gate_summary_id"]
        != acquisition._payload_sha256(_summary_identity(summary))
        or summary["receipt_sha256"]
        != acquisition._payload_sha256(
            arbitrary_plane_semantic_gate_summary_receipt_v2(summary)
        )
    ):
        raise ValueError("semantic engineering-gate summary structure or receipt changed")
    expected = replay_arbitrary_plane_semantic_gate_summary_v2(
        summary,
        planned_cases,
        case_records,
        expected_panel_id=expected_panel_id,
    )
    if acquisition._json_value(
        arbitrary_plane_semantic_gate_summary_receipt_v2(summary)
    ) != acquisition._json_value(
        arbitrary_plane_semantic_gate_summary_receipt_v2(expected)
    ):
        raise ValueError("semantic engineering-gate deterministic replay changed")
