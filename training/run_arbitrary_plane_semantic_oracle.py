"""Run the frozen 64-case, model-free arbitrary-plane semantic oracle."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

import nrrd
import numpy as np

from training.arbitrary_plane_finite_candidates import (
    finite_candidate_bank_receipt,
    make_arbitrary_plane_finite_candidate_bank_from_context,
    prepare_arbitrary_plane_finite_candidate_context,
    transport_finite_candidate_pose,
)
from training.arbitrary_plane_rendered_generator import (
    effective_renderer_sampling_arrays,
    finite_render_receipt,
    make_finite_arbitrary_plane_render_from_context,
    prepare_finite_render_context,
)
from training.arbitrary_plane_semantic_oracle import (
    EXACT_CONTROL_NAMES,
    build_oracle_target,
    canonical_payload_sha256,
    frozen_orientation_family,
    rank_candidate_ids,
    rp2_plane_error,
    score_semantic_candidates,
    semantic_gate_summary,
    shuffled_target_index,
)
from training.arbitrary_plane_support import (
    build_annotation_support_index,
    verify_annotation_support_index,
)
from training.arbitrary_plane_synthetic_generator import (
    ABSENT_OUTLINE,
    ACCURATE_OUTLINE,
    IMPERFECT_OUTLINE,
    make_arbitrary_plane_synthetic_realization,
    synthetic_realization_receipt,
)


RUNNER_SCHEMA = "anatomy-tracker.arbitrary-plane-semantic-oracle-run/v1"
CASE_SEED_ALGORITHM = "arbitrary-plane-oracle-cases/v1"
CASE_ROOT_SEED = 0xA11E5EED00000001
CASE_ROOT_SEED_HEX = "0xa11e5eed00000001"
CASE_FIELDS = ("finite-parent", "synthetic", "outline")
CASE_COUNT = 64
OUTPUT_SHAPE = (192, 256)
MARGIN_UM = (250.0, 250.0)
MAXIMUM_CASE_ATTEMPTS = 4096
MAXIMUM_LIVE_CANDIDATE_BANKS = 3
OUTLINE_MODES = (ACCURATE_OUTLINE, IMPERFECT_OUTLINE, ABSENT_OUTLINE)
ORIENTATION_COUNTS = {"near_AP": 12, "near_DV": 12, "near_ML": 12, "general_oblique": 28}
ATLAS_TEMPLATE_SHA256 = "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b"
ATLAS_ANNOTATION_SHA256 = "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42"
ATLAS_TEMPLATE_URI = "data/Allen Brain Atlas 25um/average_template_25.nrrd"
ATLAS_ANNOTATION_URI = "data/Allen Brain Atlas 25um/annotation_25.nrrd"
ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "publication" / "arbitrary_plane_oracle_pose_ranking_preflight.yaml"
DEFAULT_OUTPUT = Path(
    os.environ.get(
        "ANATOMY_TRACKER_ORACLE_OUTPUT",
        str(ROOT / "build" / "arbitrary_plane_semantic_oracle_raw"),
    )
).expanduser().resolve()
CONTROL_SCHEMAS = {
    name: f"anatomy-tracker.arbitrary-plane-semantic-oracle-control.{name}/v1"
    for name in EXACT_CONTROL_NAMES
}
CONTROL_KEYS = {
    "exact_replay": {
        "schema", "control", "case_index", "case_payload_sha256",
        "replayed_case_payload_sha256", "target_receipt_sha256", "candidate_bank_id",
        "raw_scores_sha256", "paired_outline_semantic_receipts", "passed",
    },
    "candidate_order_permutation_equivariance": {
        "schema", "control", "case_index", "permutation", "original_ordered_candidate_ids",
        "permuted_ordered_candidate_ids", "truth_candidate_id", "original_scores_sha256",
        "permuted_scores", "permuted_scores_sha256", "original_ranking",
        "permuted_ranking", "passed",
    },
    "rp2_sign_equivalence": {
        "schema", "control", "case_index", "candidate_receipts", "passed",
    },
    "truth_metadata_coordinate_channel_exclusion": {
        "schema", "control", "case_index", "scorer_signature", "scorer_source_sha256",
        "forbidden_source_tokens", "forbidden_matches", "passed",
    },
    "xy_over_wh_coordinate_contract": {
        "schema", "control", "case_index", "candidate_receipts", "passed",
    },
}
SYNTHETIC_ELIGIBILITY_REJECTIONS = {
    "finite parent has too little tissue for a synthetic realization",
    "ordinary synthetic stratum does not meet the predeclared clean-brain-pixel gate",
    "no G1 realization passed every predeclared topology, cycle, displacement, and FOV gate",
    "G2 realization failed all deterministic information-content rejection attempts",
    "G3 realization failed all deterministic damage/visibility rejection attempts",
    "imperfect outline did not meet its predeclared IoU gate",
}
SOURCE_RELATIVE_PATHS = (
    "training/run_arbitrary_plane_semantic_oracle.py",
    "training/arbitrary_plane_finite_candidates.py",
    "training/arbitrary_plane_semantic_oracle.py",
    "training/arbitrary_plane_rendered_generator.py",
    "training/arbitrary_plane_synthetic_generator.py",
    "training/arbitrary_plane_support.py",
    "training/arbitrary_plane_geometry.py",
    "training/arbitrary_plane_manifest.py",
    "training/arbitrary_plane_synthetic_ops.py",
    "training/arbitrary_plane_synthetic_observation.py",
    "publication/arbitrary_plane_oracle_pose_ranking_preflight.yaml",
    "publication/arbitrary_plane_synthetic_preflight.yaml",
)
CONFIG_KEYS = {
    "schema", "source_commit", "repository", "source_sha256", "checkout_source_sha256",
    "source_hash_contract", "preflight_sha256",
    "case_root_seed", "case_seed_algorithm", "case_count", "output_shape_h_w",
    "margin_u_v_um", "maximum_case_rejection_attempts", "memory_contract",
    "orientation_counts", "outline_modes", "atlas_assets", "shuffled_mapping",
    "animal_id", "specimen_id", "experiment_id", "learned_checkpoint_dependencies",
    "previous_model_dependencies", "pretrained_feature_dependencies",
    "deepslice_ground_truth_accessed", "real_lab_histology_accessed",
    "final_test_animals_accessed", "environment", "resolved_config_sha256",
}
RESULT_KEYS = {
    "schema", "resolved_config", "support_index_sha256",
    "prepared_render_context_sha256", "prepared_candidate_annotation_context_sha256",
    "prepared_render_asset_receipt", "prepared_candidate_annotation_receipt",
    "primary_case_payload_sha256", "shuffled_case_payload_sha256", "exact_controls",
    "semantic_gate", "interpretation", "result_payload_sha256",
}
INTERPRETATION = (
    "model-free renderer/atlas-label self-discriminability development gate only; "
    "not image-model evidence, benchmarking, qualification, or final-test evaluation"
)
SOURCE_HASH_CONTRACT = {
    "source_sha256": "SHA-256 of raw Git blob bytes at source_commit",
    "checkout_source_sha256": (
        "SHA-256 of raw loaded checkout bytes; Git clean/smudge and EOL filters may differ"
    ),
    "relationship": (
        "repository worktree_clean=true proves the checkout is Git-clean; generator receipts "
        "bind checkout bytes while source_sha256 remains cross-platform canonical"
    ),
}


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
        default=_json_scalar,
    ).encode("utf-8")


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_receipt(array: np.ndarray) -> dict[str, object]:
    value = np.asarray(array)
    if value.dtype == np.bool_:
        payload = np.packbits(value.reshape(-1, order="C"), bitorder="little").tobytes()
        dtype = "|b1"
        extra = {"bitorder": "little"}
    else:
        normalized_dtype = value.dtype.newbyteorder("<")
        payload = np.ascontiguousarray(value.astype(normalized_dtype, copy=False)).tobytes()
        dtype = normalized_dtype.str
        extra = {}
    digest = hashlib.sha256()
    digest.update(_canonical_bytes({"dtype": dtype, "shape": list(value.shape), **extra}))
    digest.update(payload)
    return {"dtype": value.dtype.str, "shape": list(value.shape), "array_sha256": digest.hexdigest()}


def _atomic_json(path: Path, value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        allow_nan=False,
        ensure_ascii=False,
        default=_json_scalar,
    ).encode("utf-8") + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace nonidentical frozen output: {path}")
        return digest
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return digest


def _atomic_bytes(path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace nonidentical frozen output: {path}")
        return digest
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return digest


def _mask_binary_receipt(case_index: int, mask: np.ndarray) -> dict[str, object]:
    value = np.asarray(mask, dtype=bool)
    payload = np.packbits(value.reshape(-1, order="C"), bitorder="little").tobytes()
    return {
        "encoding": "numpy.packbits-C-order-little-bitorder/v1",
        "relative_path": f"masks/case-{int(case_index):03d}.bin",
        "shape_h_w": list(value.shape),
        "bit_count": int(value.size),
        "byte_count": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_mask_sidecar(output: Path, target: dict[str, object], mask: np.ndarray) -> None:
    receipt = target["fixed_valid_mask_binary"]
    value = np.asarray(mask, dtype=bool)
    payload = np.packbits(value.reshape(-1, order="C"), bitorder="little").tobytes()
    if receipt != _mask_binary_receipt(target["source_case_index"], value):
        raise ValueError("fixed-valid mask binary does not match its frozen target receipt")
    if _atomic_bytes(output / receipt["relative_path"], payload) != receipt["payload_sha256"]:
        raise ValueError("fixed-valid mask binary file hash changed during atomic write")


def _seed_hex(value: int) -> str:
    value = int(value)
    if not 0 <= value <= np.iinfo(np.uint64).max:
        raise ValueError("seed must fit uint64")
    return f"0x{value:016x}"


def derive_case_seed(field: str, case_index: int, attempt: int, root_seed: int = CASE_ROOT_SEED) -> int:
    field, case_index, attempt = str(field), int(case_index), int(attempt)
    if field not in CASE_FIELDS or not 0 <= case_index < CASE_COUNT or attempt < 0:
        raise ValueError("field, case index, or case attempt is outside the frozen design")
    payload = (
        f"{CASE_SEED_ALGORITHM}\0{_seed_hex(root_seed)}\0{field}\0{case_index}\0{attempt}"
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def case_seed_lineage(case_index: int, attempt: int) -> dict[str, str]:
    return {field: _seed_hex(derive_case_seed(field, case_index, attempt)) for field in CASE_FIELDS}


def expected_orientation(case_index: int) -> str:
    case_index = int(case_index)
    if not 0 <= case_index < CASE_COUNT:
        raise ValueError("case index must be 0 through 63")
    return "near_AP" if case_index < 12 else "near_DV" if case_index < 24 else "near_ML" if case_index < 36 else "general_oblique"


def orientation_accepts(case_index: int, normal_ap_dv_ml: np.ndarray) -> bool:
    normal = np.asarray(normal_ap_dv_ml, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(
        np.linalg.norm(normal), 1.0, rtol=0.0, atol=1e-9
    ):
        raise ValueError("candidate parent normal must be a finite unit vector")
    family = expected_orientation(case_index)
    if family == "general_oblique":
        return bool(np.max(np.abs(normal)) < 0.90)
    return bool(abs(normal[{"near_AP": 0, "near_DV": 1, "near_ML": 2}[family]]) >= 0.90)


def shuffled_case_cycle() -> list[int]:
    cycle = [0]
    for _ in range(1, CASE_COUNT):
        cycle.append(shuffled_target_index(cycle[-1]))
    if len(set(cycle)) != CASE_COUNT or shuffled_target_index(cycle[-1]) != cycle[0]:
        raise RuntimeError("the frozen shuffled mapping does not form one 64-case cycle")
    return cycle


def _exception_receipt(error: Exception) -> dict[str, object]:
    message = str(error)
    prefix = "finite candidate case rejected: "
    parsed = None
    if message.startswith(prefix):
        parsed = json.loads(message[len(prefix) :])
    return {
        "exception_type": type(error).__name__,
        "message": message,
        "structured_candidate_rejection": parsed,
    }


def _target_receipt(case_index: int, paired_view_group_id: str, target: dict[str, object], scores: dict[str, object]) -> dict[str, object]:
    channel = {
        "large_ids": np.asarray(scores["target_large_region_ids"], dtype=np.int64).tolist(),
        "small_ids": np.asarray(scores["target_small_region_ids"], dtype=np.int64).tolist(),
        "minimum_individual_region_pixels": 16,
        "channel_count": int(scores["channel_count"]),
        "source": "target-defined raw nonzero Allen IDs only",
    }
    receipt = {
        "source_case_index": int(case_index),
        "paired_view_group_id": paired_view_group_id,
        "labels_receipt": _array_receipt(target["labels"]),
        "mask_receipt": _array_receipt(target["fixed_valid_mask"]),
        "fixed_valid_mask_binary": _mask_binary_receipt(
            case_index, target["fixed_valid_mask"]
        ),
        "fixed_valid_pixel_count": int(np.asarray(target["fixed_valid_mask"], dtype=bool).sum()),
        "channel_receipt": channel,
        "channel_receipt_sha256": canonical_payload_sha256(channel),
        "pixel_pitch_um": float(target["pixel_pitch_um"]),
    }
    receipt["target_receipt_sha256"] = canonical_payload_sha256(receipt)
    return receipt


def _score_payload(scores: dict[str, object]) -> dict[str, object]:
    return {
        "semantic": np.asarray(scores["semantic_score"], dtype=np.float64).tolist(),
        "raw_ID_agreement": np.asarray(scores["raw_id_agreement"], dtype=np.float64).tolist(),
        "mask_only_Dice": np.asarray(scores["mask_dice"], dtype=np.float64).tolist(),
        "channel_count": int(scores["channel_count"]),
        "smoothing_sigma_px": float(scores["smoothing_sigma_px"]),
    }


def _effective_ouv(bank: dict[str, object], candidate: dict[str, object]) -> np.ndarray:
    geometry = bank["truth_parent_geometry"] if candidate["candidate_class"] == "truth" else candidate["geometry"]
    return np.asarray(geometry["effective_physical_ouv_ap_dv_ml_um"], dtype=np.float64)


def _finite_point_error_with_evidence(
    truth_effective_ouv: np.ndarray,
    selected_effective_ouv: np.ndarray,
    fixed_valid_mask: np.ndarray,
) -> dict[str, object]:
    truth = np.asarray(truth_effective_ouv, dtype=np.float64)
    selected = np.asarray(selected_effective_ouv, dtype=np.float64)
    valid = np.asarray(fixed_valid_mask, dtype=bool)
    if (
        truth.shape != (9,)
        or selected.shape != (9,)
        or not np.isfinite(truth).all()
        or not np.isfinite(selected).all()
        or valid.ndim != 2
        or not valid.any()
    ):
        raise ValueError("point-error evidence requires finite O/U/V vectors and a nonempty mask")
    height, width = valid.shape
    s = np.arange(width, dtype=np.float64) / width
    t = np.arange(height, dtype=np.float64) / height
    tt, ss = np.meshgrid(t, s, indexing="ij")
    truth_points = truth[:3] + ss[..., None] * truth[3:6] + tt[..., None] * truth[6:9]
    selected_points = selected[:3] + ss[..., None] * selected[3:6] + tt[..., None] * selected[6:9]
    error = np.asarray(
        np.linalg.norm(truth_points - selected_points, axis=-1)[valid], dtype=np.float64
    )
    valid_yx = np.asarray(np.argwhere(valid), dtype=np.int32)
    sorted_error = np.sort(error)
    position = 0.95 * (error.size - 1)
    lower_index, upper_index = int(np.floor(position)), int(np.ceil(position))
    quantile = {
        "q": 0.95,
        "position": float(position),
        "lower_index": lower_index,
        "upper_index": upper_index,
        "upper_weight": float(position - lower_index),
        "lower_value_um": float(sorted_error[lower_index]),
        "upper_value_um": float(sorted_error[upper_index]),
    }
    evidence = {
        "schema": "anatomy-tracker.finite-corresponding-point-error-evidence/v1",
        "fixed_valid_mask_receipt": _array_receipt(valid),
        "fixed_valid_pixel_count": int(error.size),
        "fixed_valid_yx_receipt": _array_receipt(valid_yx),
        "truth_effective_physical_ouv_ap_dv_ml_um": truth.tolist(),
        "truth_effective_ouv_receipt": _array_receipt(truth),
        "selected_effective_physical_ouv_ap_dv_ml_um": selected.tolist(),
        "selected_effective_ouv_receipt": _array_receipt(selected),
        "corresponding_point_error_receipt": _array_receipt(error),
        "squared_error_sum_um2": float(np.sum(error * error, dtype=np.float64)),
        "p95_linear_quantile": quantile,
    }
    return {
        "corresponding_point_rms_um": float(np.sqrt(np.mean(error * error))),
        "corresponding_point_p95_um": float(
            np.quantile(error, 0.95, method="linear")
        ),
        "evidence": evidence,
        "evidence_sha256": canonical_payload_sha256(evidence),
    }


def _ranking_payload(bank: dict[str, object], scores: dict[str, object], valid: np.ndarray) -> dict[str, object]:
    truth_index = next(index for index, item in enumerate(bank["candidates"]) if item["candidate_class"] == "truth")
    candidate_ids = bank["ordered_candidate_ids"]
    ranking = rank_candidate_ids(
        np.asarray(scores["semantic_score"]), candidate_ids, candidate_ids[truth_index]
    )
    selected_index = ranking["selected_index"]
    if selected_index is None:
        ranking["selected_pose_error"] = None
    else:
        truth = bank["candidates"][truth_index]
        selected = bank["candidates"][selected_index]
        truth_pose, selected_pose = truth["physical_pose"], selected["physical_pose"]
        ranking["selected_pose_error"] = {
            **rp2_plane_error(
                truth_pose["normal_rp2_sign_aligned_ap_dv_ml"],
                truth_pose["signed_offset_um"],
                selected_pose["normal_rp2_sign_aligned_ap_dv_ml"],
                selected_pose["signed_offset_um"],
            ),
            **_finite_point_error_with_evidence(
                _effective_ouv(bank, truth), _effective_ouv(bank, selected), valid
            ),
        }
    return ranking


def _semantic_arrays(bank: dict[str, object]) -> np.ndarray:
    labels = np.stack([item["rendered_annotation"] for item in bank["candidates"]])
    if labels.shape != (40, *OUTPUT_SHAPE) or labels.dtype != np.int64:
        raise ValueError("finite candidate bank does not expose forty native 192x256 int64 label rasters")
    return labels


def _primary_record(
    case_index: int,
    attempt: int,
    rejection_attempts: list[dict[str, object]],
    parent: dict[str, object],
    descendants: list[dict[str, object]],
    bank: dict[str, object],
    target: dict[str, object],
    scores: dict[str, object],
) -> dict[str, object]:
    paired_id = descendants[0]["paired_view_group_id"]
    truth = next(item for item in bank["candidates"] if item["candidate_class"] == "truth")
    outline_assignment = {
        "field_stream_seed_uint64": case_seed_lineage(case_index, attempt)["outline"],
        "assignment": "three frozen explicit paired-counterfactual mode strings",
        "ordered_modes": list(OUTLINE_MODES),
    }
    record = {
        "schema": "anatomy-tracker.arbitrary-plane-semantic-oracle-case/v1",
        "case_index": int(case_index),
        "case_root_seed": CASE_ROOT_SEED_HEX,
        "accepted_case_attempt_index": int(attempt),
        "accepted_case_field_stream_seed_uint64": case_seed_lineage(case_index, attempt),
        "case_rejection_attempts": rejection_attempts,
        "case_rejection_attempts_sha256": canonical_payload_sha256(rejection_attempts),
        "orientation": frozen_orientation_family(case_index, parent["geometry"]["normal_rp2_ap_dv_ml"]),
        "truth_normal_ap_dv_ml": copy.deepcopy(parent["geometry"]["normal_rp2_ap_dv_ml"]),
        "truth_signed_offset_um": float(parent["geometry"]["signed_offset_um"]),
        "parent_plane_realization_id": parent["plane_realization_id"],
        "finite_parent_receipt": finite_render_receipt(parent),
        "paired_view_group_id": paired_id,
        "outline_assignment": outline_assignment,
        "outline_assignment_sha256": canonical_payload_sha256(outline_assignment),
        "outline_descendant_ids": [item["synthetic_realization_id"] for item in descendants],
        "outline_descendants": [
            {
                "mode": item["outline"]["parameters"]["mode"],
                "synthetic_realization_id": item["synthetic_realization_id"],
                "synthetic_receipt": synthetic_realization_receipt(item),
                "oracle_target_labels_receipt": _array_receipt(build_oracle_target(item)["labels"]),
                "oracle_target_mask_receipt": _array_receipt(build_oracle_target(item)["fixed_valid_mask"]),
            }
            for item in descendants
        ],
        "candidate_bank_id": bank["finite_candidate_bank_id"],
        "candidate_bank_receipt_sha256": bank["finite_candidate_receipt_sha256"],
        "candidate_bank_receipt": finite_candidate_bank_receipt(bank),
        "ordered_candidate_ids": copy.deepcopy(bank["ordered_candidate_ids"]),
        "truth_candidate_id": truth["candidate_id"],
        "target": _target_receipt(case_index, paired_id, target, scores),
        "scores": _score_payload(scores),
        "ranking": _ranking_payload(bank, scores, target["fixed_valid_mask"]),
        "provenance": {
            "animal_id": parent["provenance"]["animal_id"],
            "specimen_id": parent["provenance"]["specimen_id"],
            "experiment_id": parent["provenance"]["experiment_id"],
            "atlas": copy.deepcopy(parent["provenance"]["atlas"]),
            "annotation_source": copy.deepcopy(parent["provenance"]["annotation_source"]),
        },
        "data_access": {
            "allen_synthetic_development_only": True,
            "deepslice_ground_truth_accessed": False,
            "real_lab_histology_accessed": False,
            "final_test_animals_accessed": False,
        },
        "reporting_strata": {
            "orientation_family": frozen_orientation_family(
                case_index, parent["geometry"]["normal_rp2_ap_dv_ml"]
            ),
            "appearance_family": descendants[0]["g2"]["parameters"]["source_family"],
            "damage_event_types": [
                event["type"] for event in descendants[0]["g3"]["parameters"]["events"]
            ],
            "damage_event_count": int(descendants[0]["g3"]["parameters"]["event_count"]),
            "damage_union_fraction": float(
                descendants[0]["g3"]["parameters"]["gates"]["union_damage_fraction"]
            ),
            "parent_brain_pixel_count": int(parent["acceptance_contract"]["brain_pixel_count"]),
            "fixed_valid_pixel_count": int(
                np.asarray(target["fixed_valid_mask"], dtype=bool).sum()
            ),
        },
    }
    record["case_payload_sha256"] = canonical_payload_sha256(record)
    return record


def build_oracle_case(
    case_index: int,
    render_context: dict[str, object],
    candidate_context: dict[str, object],
    support_index: dict[str, object],
    source_commit: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build one accepted case; return its JSON record, target arrays, and one pending bank."""
    rejections = []
    for attempt in range(MAXIMUM_CASE_ATTEMPTS):
        seeds = case_seed_lineage(case_index, attempt)
        try:
            parent = make_finite_arbitrary_plane_render_from_context(
                render_context,
                "development",
                int(seeds["finite-parent"], 16),
                OUTPUT_SHAPE,
                sample_index=case_index,
                margin_um=MARGIN_UM,
                animal_id=None,
                specimen_id=None,
                experiment_id=None,
                max_rejection_attempts=1,
                minimum_brain_pixels=64,
                generator_source_commit=source_commit,
            )
        except RuntimeError as error:
            if not str(error).startswith("No nonempty finite tissue raster in "):
                raise
            rejections.append({"attempt_index": attempt, "field_stream_seed_uint64": seeds, "stage": "finite-parent", "reason": _exception_receipt(error)})
            continue
        normal = np.asarray(parent["geometry"]["normal_rp2_ap_dv_ml"], dtype=np.float64)
        if not orientation_accepts(case_index, normal):
            rejections.append(
                {
                    "attempt_index": attempt,
                    "field_stream_seed_uint64": seeds,
                    "stage": "orientation-stratum",
                    "expected_orientation": expected_orientation(case_index),
                    "normal_rp2_ap_dv_ml": normal.tolist(),
                    "finite_parent_receipt": finite_render_receipt(parent),
                }
            )
            continue
        try:
            bank = make_arbitrary_plane_finite_candidate_bank_from_context(
                parent,
                candidate_context,
                support_index,
                finite_parent_generator_source_commit=source_commit,
            )
        except ValueError as error:
            if not str(error).startswith("finite candidate case rejected: "):
                raise
            rejections.append({"attempt_index": attempt, "field_stream_seed_uint64": seeds, "stage": "finite-candidate-bank", "finite_parent_receipt": finite_render_receipt(parent), "reason": _exception_receipt(error)})
            continue
        descendants = []
        current_mode = None
        try:
            for current_mode in OUTLINE_MODES:
                descendants.append(
                    make_arbitrary_plane_synthetic_realization(
                        parent,
                        support_index,
                        root_seed=int(seeds["synthetic"], 16),
                        sample_index=case_index,
                        synthetic_stratum="ordinary",
                        outline_mode=current_mode,
                    )
                )
        except ValueError as error:
            if str(error) not in SYNTHETIC_ELIGIBILITY_REJECTIONS:
                raise
            rejections.append(
                {
                    "attempt_index": attempt,
                    "field_stream_seed_uint64": seeds,
                    "stage": "synthetic-eligibility",
                    "outline_mode": current_mode,
                    "finite_parent_receipt": finite_render_receipt(parent),
                    "candidate_bank_id": bank["finite_candidate_bank_id"],
                    "candidate_bank_receipt_sha256": bank["finite_candidate_receipt_sha256"],
                    "reason": _exception_receipt(error),
                }
            )
            continue
        paired_ids = {item["paired_view_group_id"] for item in descendants}
        if len(paired_ids) != 1 or [item["outline"]["parameters"]["mode"] for item in descendants] != list(OUTLINE_MODES):
            raise ValueError("three explicit outlines did not retain one paired-view group")
        targets = [build_oracle_target(item) for item in descendants]
        if any(
            first["pixel_pitch_um"] != targets[0]["pixel_pitch_um"]
            or not np.array_equal(first["labels"], targets[0]["labels"])
            or not np.array_equal(first["fixed_valid_mask"], targets[0]["fixed_valid_mask"])
            for first in targets[1:]
        ):
            raise ValueError("paired outline descendants changed semantic truth")
        target = targets[0]
        candidate_labels = _semantic_arrays(bank)
        scores = score_semantic_candidates(
            target["labels"], candidate_labels, target["fixed_valid_mask"], target["pixel_pitch_um"]
        )
        record = _primary_record(case_index, attempt, rejections, parent, descendants, bank, target, scores)
        pending = {"case_index": case_index, "bank": bank, "candidate_labels": candidate_labels, "primary_scores": scores}
        return record, target, pending
    raise RuntimeError(f"case {case_index} exhausted {MAXIMUM_CASE_ATTEMPTS} frozen attempts")


def replay_oracle_case(
    expected_record: dict[str, object],
    expected_target: dict[str, object],
    expected_pending: dict[str, object],
    render_context: dict[str, object],
    candidate_context: dict[str, object],
    support_index: dict[str, object],
    source_commit: str,
) -> dict[str, object]:
    case_index = int(expected_record["case_index"])
    record, target, pending = build_oracle_case(case_index, render_context, candidate_context, support_index, source_commit)
    passed = (
        record == expected_record
        and np.array_equal(target["labels"], expected_target["labels"])
        and np.array_equal(target["fixed_valid_mask"], expected_target["fixed_valid_mask"])
        and target["pixel_pitch_um"] == expected_target["pixel_pitch_um"]
        and np.array_equal(pending["candidate_labels"], expected_pending["candidate_labels"])
    )
    evidence = {
        "schema": CONTROL_SCHEMAS["exact_replay"],
        "control": "exact_replay",
        "case_index": case_index,
        "case_payload_sha256": expected_record["case_payload_sha256"],
        "replayed_case_payload_sha256": record["case_payload_sha256"],
        "target_receipt_sha256": expected_record["target"]["target_receipt_sha256"],
        "candidate_bank_id": expected_record["candidate_bank_id"],
        "raw_scores_sha256": canonical_payload_sha256(expected_record["scores"]),
        "paired_outline_semantic_receipts": [
            [item["oracle_target_labels_receipt"]["array_sha256"], item["oracle_target_mask_receipt"]["array_sha256"]]
            for item in expected_record["outline_descendants"]
        ],
        "passed": bool(passed),
    }
    if not passed:
        raise ValueError(f"case {case_index} failed exact replay")
    return evidence


def _case_controls(
    record: dict[str, object],
    target: dict[str, object],
    pending: dict[str, object],
    support_index: dict[str, object],
) -> dict[str, dict[str, object]]:
    bank, labels, primary = pending["bank"], pending["candidate_labels"], pending["primary_scores"]
    reverse = np.arange(39, -1, -1)
    permuted = score_semantic_candidates(
        target["labels"], labels[reverse], target["fixed_valid_mask"], target["pixel_pitch_um"]
    )
    permutation_passed = all(
        np.array_equal(np.asarray(permuted[key]), np.asarray(primary[key])[reverse])
        for key in ("semantic_score", "raw_id_agreement", "mask_dice")
    ) and all(
        np.array_equal(np.asarray(permuted[key]), np.asarray(primary[key]))
        for key in ("target_large_region_ids", "target_small_region_ids")
    )
    candidate_ids = list(bank["ordered_candidate_ids"])
    permuted_candidate_ids = [candidate_ids[index] for index in reverse]
    truth_candidate_id = str(record["truth_candidate_id"])
    original_ranking = rank_candidate_ids(
        np.asarray(primary["semantic_score"]), candidate_ids, truth_candidate_id
    )
    permuted_ranking = rank_candidate_ids(
        np.asarray(permuted["semantic_score"]), permuted_candidate_ids, truth_candidate_id
    )
    invariant_keys = (
        "truth_score", "top1", "true_rank", "reciprocal_rank",
        "true_versus_decoy_win_fraction", "true_score_margin",
        "tied_maximum_candidate_ids", "selected_candidate_id",
    )
    permutation_passed &= all(
        original_ranking[key] == permuted_ranking[key] for key in invariant_keys
    )

    rp2_receipts = []
    rp2_passed = True
    for candidate in bank["candidates"]:
        pose = candidate["physical_pose"]
        first = transport_finite_candidate_pose(
            bank["truth_parent_geometry"], support_index, pose["normal_rp2_sign_aligned_ap_dv_ml"], pose["signed_offset_um"], pose["roll_delta_rad_from_parallel_transport"]
        )
        second = transport_finite_candidate_pose(
            bank["truth_parent_geometry"], support_index, -np.asarray(pose["normal_rp2_sign_aligned_ap_dv_ml"]), -float(pose["signed_offset_um"]), pose["roll_delta_rad_from_parallel_transport"]
        )
        equal = first == second
        rp2_passed &= equal
        rp2_receipts.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_pose_sha256": candidate["pose_sha256"],
                "geometry_storage": candidate["geometry_storage"],
                "stored_candidate_geometry_sha256": candidate[
                    "candidate_geometry_sha256"
                ],
                "positive_geometry_sha256": first["candidate_geometry_sha256"],
                "antipodal_geometry_sha256": second["candidate_geometry_sha256"],
                "equal": bool(equal),
            }
        )

    scorer_signature = list(inspect.signature(score_semantic_candidates).parameters)
    scorer_source = inspect.getsource(score_semantic_candidates)
    forbidden = ("candidate_id", "truth_index", "coordinate", "geometry", "normal", "offset", "roll")
    forbidden_matches = [token for token in forbidden if token in scorer_source]
    metadata_passed = scorer_signature == ["target_labels", "candidate_labels", "fixed_valid_mask", "pixel_pitch_um"] and not forbidden_matches

    xy_receipts = []
    xy_passed = True
    atlas_shape = tuple(support_index["annotation_shape"])
    for candidate in bank["candidates"]:
        geometry = bank["truth_parent_geometry"] if candidate["candidate_class"] == "truth" else candidate["geometry"]
        arrays = effective_renderer_sampling_arrays(geometry, atlas_shape)
        points = arrays["coordinate_raster_allen_index_float32"]
        ouv = arrays["allen_index_ouv_ap_dv_ml_float32"]
        height, width = geometry["output_shape_h_w"]
        s = np.arange(width, dtype=np.float32) / np.float32(width)
        t = np.arange(height, dtype=np.float32) / np.float32(height)
        tt, ss = np.meshgrid(t, s, indexing="ij")
        expected_grid = (
            ouv[:3]
            + ss[..., None] * ouv[3:6]
            + tt[..., None] * ouv[6:9]
        ).astype(np.float32, copy=False)
        residual = float(np.max(np.abs(points - expected_grid)))
        inclusive_gap = float(np.linalg.norm(points[-1, -1] - (ouv[:3] + ouv[3:6] + ouv[6:9])))
        source_receipt = geometry["array_receipts"].get(
            "effective_coordinate_raster_allen_index_float32",
            geometry["array_receipts"].get("coordinate_raster_allen_index_float32"),
        )
        actual_receipt = _array_receipt(points)
        passed = (
            geometry["sampling_contract"] == "quicknii-raster-index-x-over-W-y-over-H-v1"
            and actual_receipt == source_receipt
            and residual <= 5e-4
            and inclusive_gap > 1e-6
        )
        xy_passed &= passed
        xy_receipts.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_coordinate_raster_receipt": copy.deepcopy(source_receipt),
                "reconstructed_xy_over_wh_grid_receipt": _array_receipt(expected_grid),
                "output_shape_h_w": [height, width],
                "grid_point_count": int(height * width),
                "maximum_absolute_residual_allen_index": residual,
                "inclusive_endpoint_gap_allen_index": inclusive_gap,
                "equal_within_float32_tolerance": bool(passed),
            }
        )

    case_index = int(record["case_index"])
    return {
        "candidate_order_permutation_equivariance": {
            "schema": CONTROL_SCHEMAS["candidate_order_permutation_equivariance"],
            "control": "candidate_order_permutation_equivariance",
            "case_index": case_index,
            "permutation": reverse.tolist(),
            "original_ordered_candidate_ids": candidate_ids,
            "permuted_ordered_candidate_ids": permuted_candidate_ids,
            "truth_candidate_id": truth_candidate_id,
            "original_scores_sha256": canonical_payload_sha256(_score_payload(primary)),
            "permuted_scores": _score_payload(permuted),
            "permuted_scores_sha256": canonical_payload_sha256(_score_payload(permuted)),
            "original_ranking": original_ranking,
            "permuted_ranking": permuted_ranking,
            "passed": bool(permutation_passed),
        },
        "rp2_sign_equivalence": {
            "schema": CONTROL_SCHEMAS["rp2_sign_equivalence"],
            "control": "rp2_sign_equivalence",
            "case_index": case_index,
            "candidate_receipts": rp2_receipts,
            "passed": bool(rp2_passed),
        },
        "truth_metadata_coordinate_channel_exclusion": {
            "schema": CONTROL_SCHEMAS["truth_metadata_coordinate_channel_exclusion"],
            "control": "truth_metadata_coordinate_channel_exclusion",
            "case_index": case_index,
            "scorer_signature": scorer_signature,
            "scorer_source_sha256": hashlib.sha256(scorer_source.encode("utf-8")).hexdigest(),
            "forbidden_source_tokens": list(forbidden),
            "forbidden_matches": forbidden_matches,
            "passed": bool(metadata_passed),
        },
        "xy_over_wh_coordinate_contract": {
            "schema": CONTROL_SCHEMAS["xy_over_wh_coordinate_contract"],
            "control": "xy_over_wh_coordinate_contract",
            "case_index": case_index,
            "candidate_receipts": xy_receipts,
            "passed": bool(xy_passed),
        },
    }


def _shuffled_record(pending: dict[str, object], target_record: dict[str, object], target: dict[str, object]) -> dict[str, object]:
    bank = pending["bank"]
    source_index = int(pending["case_index"])
    expected_target_index = shuffled_target_index(source_index)
    if int(target_record["case_index"]) != expected_target_index:
        raise ValueError("shuffled stream supplied the wrong target case")
    scores = score_semantic_candidates(
        target["labels"], pending["candidate_labels"], target["fixed_valid_mask"], target["pixel_pitch_um"]
    )
    recomputed_target_receipt = _target_receipt(
        expected_target_index,
        str(target_record["paired_view_group_id"]),
        target,
        scores,
    )
    if recomputed_target_receipt != target_record["target"]:
        raise ValueError("shuffled target arrays or pitch do not match the frozen target receipt")
    truth = next(item for item in bank["candidates"] if item["candidate_class"] == "truth")
    record = {
        "schema": "anatomy-tracker.arbitrary-plane-semantic-oracle-shuffled/v1",
        "case_index": source_index,
        "paired_view_group_id": pending["primary_record"]["paired_view_group_id"],
        "candidate_bank_id": bank["finite_candidate_bank_id"],
        "candidate_bank_receipt_sha256": bank["finite_candidate_receipt_sha256"],
        "ordered_candidate_ids": copy.deepcopy(bank["ordered_candidate_ids"]),
        "truth_candidate_id": truth["candidate_id"],
        "target": recomputed_target_receipt,
        "scores": _score_payload(scores),
        "ranking": _ranking_payload(bank, scores, target["fixed_valid_mask"]),
        "mapping": "target=(case_index+17)%64; candidate bank and order unchanged",
    }
    record["shuffled_payload_sha256"] = canonical_payload_sha256(record)
    return record


def _write_control_sidecar(output: Path, evidence: dict[str, object]) -> dict[str, object]:
    name = str(evidence.get("control"))
    case_index = int(evidence.get("case_index", -1))
    if (
        name not in EXACT_CONTROL_NAMES
        or evidence.get("schema") != CONTROL_SCHEMAS.get(name)
        or set(evidence) != CONTROL_KEYS.get(name)
        or not 0 <= case_index < CASE_COUNT
        or type(evidence.get("passed")) is not bool
    ):
        raise ValueError("control evidence must bind one frozen name, case index, and strict Boolean")
    relative = Path("controls") / name / f"case-{case_index:03d}.json"
    return {
        "case_index": case_index,
        "relative_path": relative.as_posix(),
        "passed": evidence["passed"],
        "payload_sha256": canonical_payload_sha256(evidence),
        "file_sha256": _atomic_json(output / relative, evidence),
    }


def _control_record(name: str, case_evidence: list[dict[str, object]]) -> dict[str, object]:
    if name not in EXACT_CONTROL_NAMES:
        raise ValueError("unknown exact control")
    references = sorted(copy.deepcopy(case_evidence), key=lambda item: int(item["case_index"]))
    exact_indices = [int(item["case_index"]) for item in references] == list(range(CASE_COUNT))
    passed = exact_indices and all(item.get("passed") is True for item in references)
    evidence = {
        "control": name,
        "checked_cases": len(references),
        "case_evidence": references,
    }
    return {
        "passed": bool(passed),
        "evidence": evidence,
        "evidence_receipt_sha256": canonical_payload_sha256({"control": name, "passed": bool(passed), "evidence": evidence}),
    }


def _gate_view(record: dict[str, object]) -> dict[str, object]:
    return {key: copy.deepcopy(record[key]) for key in (
        "case_index", "parent_plane_realization_id", "paired_view_group_id", "outline_descendant_ids",
        "candidate_bank_id", "ordered_candidate_ids", "truth_candidate_id", "truth_normal_ap_dv_ml",
        "orientation", "target", "scores",
    ) if key in record}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def repository_state() -> dict[str, object]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("semantic-oracle execution requires a clean worktree")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    head = _git("rev-parse", "HEAD")
    upstream_head = _git("rev-parse", "@{upstream}")
    if branch == "HEAD" or head != upstream_head:
        raise RuntimeError("semantic-oracle execution requires a branch whose HEAD equals its upstream")
    return {
        "branch": branch,
        "upstream": upstream,
        "head": head,
        "upstream_head": upstream_head,
        "worktree_clean": True,
    }


def _source_hash_receipts(source_commit: str) -> tuple[dict[str, str], dict[str, str]]:
    git_blobs = {
        relative_path: _git_blob_sha256(source_commit, relative_path)
        for relative_path in SOURCE_RELATIVE_PATHS
    }
    checkout_bytes = {
        relative_path: _file_sha256(ROOT / relative_path)
        for relative_path in SOURCE_RELATIVE_PATHS
    }
    return git_blobs, checkout_bytes


def _build_allen_support_index(annotation: np.ndarray) -> dict[str, object]:
    return build_annotation_support_index(
        annotation,
        atlas_id="Allen CCFv3",
        atlas_version="2017 25um",
        source_uri=ATLAS_ANNOTATION_URI,
        source_sha256=ATLAS_ANNOTATION_SHA256,
        source_entity_type="atlas",
        voxel_size_um=(25.0, 25.0, 25.0),
        origin_um=(0.0, 0.0, 0.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )


def _load_authenticated_allen_support_index() -> dict[str, object]:
    annotation_path = ROOT / ATLAS_ANNOTATION_URI
    if _file_sha256(annotation_path) != ATLAS_ANNOTATION_SHA256:
        raise ValueError("raw Allen annotation does not match the frozen preflight hash")
    annotation = nrrd.read(str(annotation_path), index_order="F")[0]
    support = _build_allen_support_index(annotation)
    verify_annotation_support_index(support)
    return support


def load_allen_contexts() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    template_path = ROOT / ATLAS_TEMPLATE_URI
    annotation_path = ROOT / ATLAS_ANNOTATION_URI
    if _file_sha256(template_path) != ATLAS_TEMPLATE_SHA256 or _file_sha256(annotation_path) != ATLAS_ANNOTATION_SHA256:
        raise ValueError("raw Allen atlas assets do not match the frozen preflight hashes")
    template = nrrd.read(str(template_path), index_order="F")[0]
    annotation = nrrd.read(str(annotation_path), index_order="F")[0]
    support = _build_allen_support_index(annotation)
    render_context = prepare_finite_render_context(
        template,
        annotation,
        support,
        scalar_source_uri=ATLAS_TEMPLATE_URI,
        scalar_source_sha256=ATLAS_TEMPLATE_SHA256,
        scalar_source_entity_type="atlas-template",
        template_decoder="pynrrd 1.1.3",
        template_index_order="F",
        annotation_decoder="pynrrd 1.1.3",
        annotation_index_order="F",
    )
    candidate_context = prepare_arbitrary_plane_finite_candidate_context(annotation, support)
    return support, render_context, candidate_context


def _require_sha256(value: object, name: str) -> str:
    digest = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return digest


def _require_model_independence(generator: dict[str, object], name: str) -> None:
    for key in (
        "learned_checkpoint_dependencies",
        "previous_model_dependencies",
        "pretrained_feature_dependencies",
    ):
        if generator.get(key) != []:
            raise ValueError(f"{name} does not preserve empty {key}")


def _verify_finite_parent_identity(
    parent: object,
    name: str,
    support_index_sha256: str | None = None,
) -> dict[str, object]:
    if not isinstance(parent, dict):
        raise ValueError(f"{name} is not a finite-parent receipt")
    required = {
        "schema_version", "generator_algorithm", "split", "support_index_sha256",
        "generator", "provenance", "provenance_sha256", "rejection_attempts",
        "rejection_attempts_sha256", "plane_realization_id",
        "finite_plane_geometry_sha256", "rendered_artifacts_sha256",
    }
    if not required.issubset(parent):
        raise ValueError(f"{name} lacks finite-parent identity fields")
    provenance = parent["provenance"]
    if (
        parent["schema_version"] != "anatomy-tracker.finite-arbitrary-plane-render/v1"
        or parent["generator_algorithm"] != "uniform-rp2-component-union-finite-render/v1"
        or parent["split"] != "development"
        or not isinstance(provenance, dict)
        or any(
            provenance.get(key) is not None
            for key in ("animal_id", "specimen_id", "experiment_id")
        )
    ):
        raise ValueError(f"{name} changed finite-parent protocol or subject identity")
    for key in (
        "support_index_sha256", "provenance_sha256", "rejection_attempts_sha256",
        "plane_realization_id", "finite_plane_geometry_sha256", "rendered_artifacts_sha256",
    ):
        _require_sha256(parent[key], f"{name} {key}")
    if support_index_sha256 is not None and parent["support_index_sha256"] != support_index_sha256:
        raise ValueError(f"{name} changed support-index identity")
    if parent["provenance_sha256"] != canonical_payload_sha256(provenance):
        raise ValueError(f"{name} provenance hash does not replay")
    if (
        not isinstance(parent["rejection_attempts"], list)
        or parent["rejection_attempts_sha256"]
        != canonical_payload_sha256(parent["rejection_attempts"])
    ):
        raise ValueError(f"{name} internal rejection receipt does not replay")
    generator = parent["generator"]
    if not isinstance(generator, dict):
        raise ValueError(f"{name} lacks a generator receipt")
    _require_model_independence(generator, name)
    resolved = generator.get("resolved_config")
    implementation = generator.get("implementation")
    if (
        not isinstance(resolved, dict)
        or generator.get("resolved_config_sha256") != canonical_payload_sha256(resolved)
        or not isinstance(implementation, dict)
        or resolved.get("support_index_sha256") != parent["support_index_sha256"]
        or resolved.get("output_shape_h_w") != list(OUTPUT_SHAPE)
        or resolved.get("margin_u_v_um") != list(MARGIN_UM)
        or resolved.get("max_rejection_attempts") != 1
        or resolved.get("minimum_brain_pixels") != 64
        or any(
            resolved.get(key) is not None
            for key in ("animal_id", "specimen_id", "experiment_id")
        )
    ):
        raise ValueError(f"{name} generator identities do not replay")
    if "implementation_sha256" in implementation:
        implementation_payload = {
            key: value for key, value in implementation.items() if key != "implementation_sha256"
        }
        if implementation["implementation_sha256"] != canonical_payload_sha256(
            implementation_payload
        ):
            raise ValueError(f"{name} implementation hash does not replay")
    return parent


def _verify_rejection_reason(stage: str, rejection: dict[str, object]) -> None:
    reason = rejection.get("reason")
    if not isinstance(reason, dict) or set(reason) != {
        "exception_type", "message", "structured_candidate_rejection"
    }:
        raise ValueError("case rejection reason does not have the exact exception receipt")
    message = reason["message"]
    structured = reason["structured_candidate_rejection"]
    if not isinstance(message, str):
        raise ValueError("case rejection reason message must be a string")
    if stage == "finite-parent":
        if (
            reason["exception_type"] != "RuntimeError"
            or structured is not None
            or not message.startswith("No nonempty finite tissue raster in ")
        ):
            raise ValueError("finite-parent rejection is not a frozen stochastic eligibility failure")
        return
    if stage == "synthetic-eligibility":
        if (
            reason["exception_type"] != "ValueError"
            or structured is not None
            or message not in SYNTHETIC_ELIGIBILITY_REJECTIONS
        ):
            raise ValueError("synthetic rejection is not a frozen stochastic eligibility failure")
        return
    if stage != "finite-candidate-bank":
        raise ValueError("only exception-bearing rejection stages may contain a reason receipt")
    if (
        reason["exception_type"] != "ValueError"
        or not isinstance(structured, dict)
        or set(structured) != {
            "schema", "reason", "candidate_attempts", "candidate_attempts_sha256"
        }
        or structured["schema"] != "anatomy-tracker.finite-candidate-case-rejection/v1"
        or not isinstance(structured["reason"], str)
        or not structured["reason"]
        or not isinstance(structured["candidate_attempts"], list)
        or not structured["candidate_attempts"]
        or not all(isinstance(item, dict) for item in structured["candidate_attempts"])
        or structured["candidate_attempts_sha256"]
        != canonical_payload_sha256(structured["candidate_attempts"])
        or message != "finite candidate case rejected: " + _canonical_bytes(structured).decode("utf-8")
    ):
        raise ValueError("candidate-bank rejection is not a hash-bound stochastic eligibility failure")


def _verify_saved_ranking(
    record: dict[str, object],
    derived: dict[str, object],
    name: str,
    bank: dict[str, object],
    fixed_valid_mask: np.ndarray,
) -> None:
    ranking = record.get("ranking")
    expected_keys = set(derived) | {"selected_pose_error"}
    if not isinstance(ranking, dict) or set(ranking) != expected_keys:
        raise ValueError(f"{name} ranking lacks the complete frozen fields")
    if {key: value for key, value in ranking.items() if key != "selected_pose_error"} != derived:
        raise ValueError(f"saved {name} ranking does not match its raw score landscape")
    pose_error = ranking["selected_pose_error"]
    if derived["selected_candidate_id"] is None:
        if pose_error is not None:
            raise ValueError(f"tied {name} ranking must have null selected pose error")
        return
    required = {
        "normal_geodesic_angle_deg",
        "sign_aligned_offset_error_um",
        "corresponding_point_rms_um",
        "corresponding_point_p95_um",
        "evidence",
        "evidence_sha256",
    }
    if not isinstance(pose_error, dict) or set(pose_error) != required:
        raise ValueError(f"unique {name} ranking lacks complete physical pose errors")
    candidates = bank.get("candidates", [])
    truth = next(
        (item for item in candidates if item.get("candidate_id") == record["truth_candidate_id"]),
        None,
    )
    selected = next(
        (item for item in candidates if item.get("candidate_id") == derived["selected_candidate_id"]),
        None,
    )
    if truth is None or selected is None:
        raise ValueError(f"{name} selected pose is not present in the frozen candidate bank")
    expected_plane = rp2_plane_error(
        truth["physical_pose"]["normal_rp2_sign_aligned_ap_dv_ml"],
        truth["physical_pose"]["signed_offset_um"],
        selected["physical_pose"]["normal_rp2_sign_aligned_ap_dv_ml"],
        selected["physical_pose"]["signed_offset_um"],
    )
    if any(
        not np.isclose(
            float(pose_error[key]), float(expected_plane[key]), rtol=0.0, atol=1e-12
        )
        for key in expected_plane
    ):
        raise ValueError(f"saved {name} normal or offset error does not match its selected pose")
    truth_ouv = _effective_ouv(bank, truth)
    selected_ouv = _effective_ouv(bank, selected)
    valid = np.asarray(fixed_valid_mask, dtype=bool)
    if (
        _array_receipt(valid) != record["target"]["mask_receipt"]
        or int(valid.sum()) != int(record["target"]["fixed_valid_pixel_count"])
    ):
        raise ValueError(f"saved {name} mask is not bound to its target receipt")
    expected_point = _finite_point_error_with_evidence(truth_ouv, selected_ouv, valid)
    saved_point = {
        key: value for key, value in pose_error.items() if key not in expected_plane
    }
    if saved_point != expected_point:
        raise ValueError(
            f"saved {name} point-error metrics and evidence do not replay from the exact mask"
        )


def _verify_score_payload(record: dict[str, object], name: str) -> None:
    scores = record.get("scores")
    target = record.get("target", {})
    if not isinstance(scores, dict) or set(scores) != {
        "semantic", "raw_ID_agreement", "mask_only_Dice", "channel_count",
        "smoothing_sigma_px",
    }:
        raise ValueError(f"{name} score payload does not have the five frozen fields")
    arrays = [np.asarray(scores[key], dtype=np.float64) for key in (
        "semantic", "raw_ID_agreement", "mask_only_Dice"
    )]
    if any(array.shape != (40,) or not np.isfinite(array).all() for array in arrays) or any(
        np.any((array < 0.0) | (array > 1.0)) for array in arrays
    ):
        raise ValueError(f"{name} raw score vectors must contain forty finite unit-interval values")
    pitch = float(target.get("pixel_pitch_um", np.nan))
    channel_count = int(target.get("channel_receipt", {}).get("channel_count", -1))
    if (
        int(scores["channel_count"]) != channel_count
        or channel_count <= 0
        or not np.isfinite(pitch)
        or pitch <= 0.0
        or not np.isclose(float(scores["smoothing_sigma_px"]), 75.0 / pitch, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(f"{name} channel count or physical smoothing scale does not match its target")


def _verify_primary_record(
    record: dict[str, object],
    case_index: int,
    support_index_sha256: str | None = None,
) -> None:
    required = {
        "schema", "case_index", "case_root_seed", "accepted_case_attempt_index",
        "accepted_case_field_stream_seed_uint64", "case_rejection_attempts",
        "case_rejection_attempts_sha256", "orientation", "truth_normal_ap_dv_ml",
        "truth_signed_offset_um", "parent_plane_realization_id", "finite_parent_receipt",
        "paired_view_group_id", "outline_assignment", "outline_assignment_sha256",
        "outline_descendant_ids", "outline_descendants", "candidate_bank_id",
        "candidate_bank_receipt_sha256", "candidate_bank_receipt",
        "ordered_candidate_ids", "truth_candidate_id", "target", "scores", "ranking",
        "provenance", "data_access", "reporting_strata", "case_payload_sha256",
    }
    if set(record) != required or record.get("schema") != "anatomy-tracker.arbitrary-plane-semantic-oracle-case/v1" or int(record.get("case_index", -1)) != case_index:
        raise ValueError("primary sidecar does not match the complete production schema")
    attempt = int(record["accepted_case_attempt_index"])
    rejections = record["case_rejection_attempts"]
    if (
        record["case_root_seed"] != CASE_ROOT_SEED_HEX
        or not 0 <= attempt < MAXIMUM_CASE_ATTEMPTS
        or not isinstance(rejections, list)
        or len(rejections) != attempt
        or record["accepted_case_field_stream_seed_uint64"] != case_seed_lineage(case_index, attempt)
        or record["case_rejection_attempts_sha256"] != canonical_payload_sha256(rejections)
    ):
        raise ValueError("primary case seed or rejection lineage does not replay")
    for rejection_index, rejection in enumerate(rejections):
        if (
            int(rejection.get("attempt_index", -1)) != rejection_index
            or rejection.get("field_stream_seed_uint64") != case_seed_lineage(case_index, rejection_index)
            or rejection.get("stage") not in {
                "finite-parent", "orientation-stratum", "finite-candidate-bank",
                "synthetic-eligibility",
            }
        ):
            raise ValueError("primary case rejection log is incomplete or out of order")
        stage = rejection["stage"]
        expected_keys = {
            "finite-parent": {"attempt_index", "field_stream_seed_uint64", "stage", "reason"},
            "orientation-stratum": {
                "attempt_index", "field_stream_seed_uint64", "stage", "expected_orientation",
                "normal_rp2_ap_dv_ml", "finite_parent_receipt",
            },
            "finite-candidate-bank": {
                "attempt_index", "field_stream_seed_uint64", "stage", "finite_parent_receipt",
                "reason",
            },
            "synthetic-eligibility": {
                "attempt_index", "field_stream_seed_uint64", "stage", "outline_mode",
                "finite_parent_receipt", "candidate_bank_id",
                "candidate_bank_receipt_sha256", "reason",
            },
        }[stage]
        if set(rejection) != expected_keys:
            raise ValueError("case rejection does not match its strict stage-specific schema")
        if stage != "orientation-stratum":
            _verify_rejection_reason(stage, rejection)
        if stage != "finite-parent":
            rejected_parent = _verify_finite_parent_identity(
                rejection["finite_parent_receipt"],
                f"case {case_index} rejected finite parent",
                support_index_sha256,
            )
        if stage == "orientation-stratum":
            rejected_normal = np.asarray(rejection["normal_rp2_ap_dv_ml"], dtype=np.float64)
            parent_normal = np.asarray(
                rejected_parent.get("geometry", {}).get("normal_rp2_ap_dv_ml"),
                dtype=np.float64,
            )
            if (
                rejection["expected_orientation"] != expected_orientation(case_index)
                or rejected_normal.shape != (3,)
                or not np.isfinite(rejected_normal).all()
                or not np.isclose(np.linalg.norm(rejected_normal), 1.0, rtol=0.0, atol=1e-9)
                or parent_normal.shape != (3,)
                or not np.array_equal(rejected_normal, parent_normal)
                or orientation_accepts(case_index, rejected_normal)
            ):
                raise ValueError("orientation rejection does not bind a rejected frozen case stratum")
        if stage == "synthetic-eligibility" and (
            rejection.get("outline_mode") not in OUTLINE_MODES
            or re.fullmatch(r"[0-9a-f]{64}", str(rejection.get("candidate_bank_id"))) is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(rejection.get("candidate_bank_receipt_sha256"))
            )
            is None
        ):
            raise ValueError("synthetic rejection is not a frozen stochastic eligibility failure")
    normal = np.asarray(record["truth_normal_ap_dv_ml"], dtype=np.float64)
    if record["orientation"] != frozen_orientation_family(case_index, normal) or not np.isfinite(float(record["truth_signed_offset_um"])):
        raise ValueError("primary orientation or truth plane is invalid")
    parent = _verify_finite_parent_identity(
        record["finite_parent_receipt"], "accepted finite parent", support_index_sha256
    )
    bank = record["candidate_bank_receipt"]
    if (
        parent.get("plane_realization_id") != record["parent_plane_realization_id"]
        or parent.get("split") != "development"
        or parent.get("provenance", {}).get("animal_id") is not None
        or parent.get("provenance", {}).get("specimen_id") is not None
        or parent.get("provenance", {}).get("experiment_id") is not None
    ):
        raise ValueError("finite-parent receipt or null subject provenance is incomplete")
    if (
        not isinstance(bank, dict)
        or bank.get("finite_candidate_bank_id") != record["candidate_bank_id"]
        or canonical_payload_sha256(bank) != record["candidate_bank_receipt_sha256"]
        or bank.get("finite_parent_receipt") != parent
        or bank.get("support_index_sha256") != parent["support_index_sha256"]
    ):
        raise ValueError("candidate-bank receipt does not match its frozen identity")
    bank_generator = bank.get("generator", {})
    _require_model_independence(bank_generator, "candidate bank")
    bank_resolved = bank_generator.get("resolved_config")
    bank_implementation = bank_generator.get("implementation")
    if (
        not isinstance(bank_resolved, dict)
        or bank_generator.get("resolved_config_sha256")
        != canonical_payload_sha256(bank_resolved)
        or not isinstance(bank_implementation, dict)
        or bank_generator.get("implementation_sha256")
        != canonical_payload_sha256(bank_implementation)
    ):
        raise ValueError("candidate-bank generator identities do not replay")
    candidates = bank.get("candidates")
    ordered_ids = [str(value) for value in record["ordered_candidate_ids"]]
    if (
        not isinstance(candidates, list)
        or len(candidates) != 40
        or len(set(ordered_ids)) != 40
        or [str(item.get("candidate_id")) for item in candidates] != ordered_ids
        or bank.get("ordered_candidate_ids") != ordered_ids
    ):
        raise ValueError("candidate-bank sidecar lacks forty ordered candidate receipts")
    truths = [item for item in candidates if item.get("candidate_class") == "truth"]
    if len(truths) != 1 or truths[0].get("candidate_id") != record["truth_candidate_id"]:
        raise ValueError("candidate-bank truth identity is incomplete")
    _require_sha256(record["candidate_bank_id"], "candidate bank ID")
    _require_sha256(record["candidate_bank_receipt_sha256"], "candidate bank receipt")

    assignment = record["outline_assignment"]
    if (
        record["outline_assignment_sha256"] != canonical_payload_sha256(assignment)
        or assignment.get("field_stream_seed_uint64") != case_seed_lineage(case_index, attempt)["outline"]
        or assignment.get("ordered_modes") != list(OUTLINE_MODES)
    ):
        raise ValueError("outline assignment does not match its independent seed lineage")
    descendants = record["outline_descendants"]
    descendant_ids = record["outline_descendant_ids"]
    if (
        not isinstance(descendants, list)
        or len(descendants) != 3
        or len(set(descendant_ids)) != 3
        or [item.get("mode") for item in descendants] != list(OUTLINE_MODES)
        or [item.get("synthetic_realization_id") for item in descendants] != descendant_ids
    ):
        raise ValueError("primary sidecar lacks exactly three explicit outline descendants")
    for descendant in descendants:
        receipt = descendant.get("synthetic_receipt")
        if (
            not isinstance(receipt, dict)
            or receipt.get("synthetic_realization_id") != descendant["synthetic_realization_id"]
            or receipt.get("paired_view_group_id") != record["paired_view_group_id"]
        ):
            raise ValueError("synthetic outline receipt is not paired to the base case")
        _require_model_independence(receipt.get("generator", {}), "synthetic descendant")

    target = record["target"]
    target_keys = {
        "source_case_index", "paired_view_group_id", "labels_receipt", "mask_receipt",
        "fixed_valid_mask_binary", "fixed_valid_pixel_count", "channel_receipt",
        "channel_receipt_sha256", "pixel_pitch_um", "target_receipt_sha256",
    }
    if (
        not isinstance(target, dict)
        or set(target) != target_keys
        or int(target["source_case_index"]) != case_index
        or target["paired_view_group_id"] != record["paired_view_group_id"]
        or target["labels_receipt"].get("shape") != list(OUTPUT_SHAPE)
        or target["mask_receipt"].get("shape") != list(OUTPUT_SHAPE)
        or not 0 < int(target["fixed_valid_pixel_count"]) <= OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]
        or target["channel_receipt_sha256"] != canonical_payload_sha256(target["channel_receipt"])
        or target["target_receipt_sha256"] != canonical_payload_sha256(
            {key: value for key, value in target.items() if key != "target_receipt_sha256"}
        )
    ):
        raise ValueError("primary semantic target receipt is incomplete")
    for descendant in descendants:
        if (
            descendant["oracle_target_labels_receipt"] != target["labels_receipt"]
            or descendant["oracle_target_mask_receipt"] != target["mask_receipt"]
        ):
            raise ValueError("paired outline target receipts are not semantically invariant")
    _verify_score_payload(record, "primary")

    provenance = record["provenance"]
    if set(provenance) != {"animal_id", "specimen_id", "experiment_id", "atlas", "annotation_source"} or any(
        provenance[key] is not None for key in ("animal_id", "specimen_id", "experiment_id")
    ):
        raise ValueError("primary provenance must preserve explicit null subject identifiers")
    access = record["data_access"]
    if access != {
        "allen_synthetic_development_only": True,
        "deepslice_ground_truth_accessed": False,
        "real_lab_histology_accessed": False,
        "final_test_animals_accessed": False,
    }:
        raise ValueError("primary sidecar accessed data outside the frozen development scope")
    strata = record["reporting_strata"]
    if (
        set(strata) != {
            "orientation_family", "appearance_family", "damage_event_types",
            "damage_event_count", "damage_union_fraction", "parent_brain_pixel_count",
            "fixed_valid_pixel_count",
        }
        or strata["orientation_family"] != record["orientation"]
        or int(strata["damage_event_count"]) != len(strata["damage_event_types"])
        or not 0.0 <= float(strata["damage_union_fraction"]) <= 1.0
        or int(strata["parent_brain_pixel_count"]) < 64
        or int(strata["fixed_valid_pixel_count"]) != int(target["fixed_valid_pixel_count"])
    ):
        raise ValueError("primary appearance, damage, or support reporting strata are incomplete")


def _verify_shuffled_record(record: dict[str, object], primary: dict[str, object], case_index: int) -> None:
    required = {
        "schema", "case_index", "paired_view_group_id", "candidate_bank_id",
        "candidate_bank_receipt_sha256", "ordered_candidate_ids", "truth_candidate_id",
        "target", "scores", "ranking", "mapping", "shuffled_payload_sha256",
    }
    if (
        set(record) != required
        or record.get("schema") != "anatomy-tracker.arbitrary-plane-semantic-oracle-shuffled/v1"
        or int(record.get("case_index", -1)) != case_index
        or record["paired_view_group_id"] != primary["paired_view_group_id"]
        or record["candidate_bank_id"] != primary["candidate_bank_id"]
        or record["candidate_bank_receipt_sha256"] != primary["candidate_bank_receipt_sha256"]
        or record["ordered_candidate_ids"] != primary["ordered_candidate_ids"]
        or record["truth_candidate_id"] != primary["truth_candidate_id"]
        or record["mapping"] != "target=(case_index+17)%64; candidate bank and order unchanged"
    ):
        raise ValueError("shuffled sidecar does not match the complete production schema")
    _verify_score_payload(record, "shuffled")


def _verify_control_payload(
    name: str,
    item: dict[str, object],
    primary: dict[str, object],
    authenticated_support_index: dict[str, object] | None = None,
) -> None:
    case_index = int(primary["case_index"])
    if (
        set(item) != CONTROL_KEYS[name]
        or item.get("schema") != CONTROL_SCHEMAS[name]
        or item.get("control") != name
        or int(item.get("case_index", -1)) != case_index
        or item.get("passed") is not True
    ):
        raise ValueError("exact control evidence does not match its strict per-control schema")
    if name == "exact_replay":
        expected_outlines = [
            [
                descendant["oracle_target_labels_receipt"]["array_sha256"],
                descendant["oracle_target_mask_receipt"]["array_sha256"],
            ]
            for descendant in primary["outline_descendants"]
        ]
        if (
            item["case_payload_sha256"] != primary["case_payload_sha256"]
            or item["replayed_case_payload_sha256"] != primary["case_payload_sha256"]
            or item["target_receipt_sha256"] != primary["target"]["target_receipt_sha256"]
            or item["candidate_bank_id"] != primary["candidate_bank_id"]
            or item["raw_scores_sha256"] != canonical_payload_sha256(primary["scores"])
            or item["paired_outline_semantic_receipts"] != expected_outlines
        ):
            raise ValueError("exact replay evidence is not cross-bound to the primary case")
        return
    if name == "candidate_order_permutation_equivariance":
        permutation = list(range(39, -1, -1))
        ordered = primary["ordered_candidate_ids"]
        expected_scores = {
            **primary["scores"],
            **{
                key: [primary["scores"][key][index] for index in permutation]
                for key in ("semantic", "raw_ID_agreement", "mask_only_Dice")
            },
        }
        expected_permuted_ids = [ordered[index] for index in permutation]
        original_ranking = rank_candidate_ids(
            np.asarray(primary["scores"]["semantic"]), ordered, primary["truth_candidate_id"]
        )
        permuted_ranking = rank_candidate_ids(
            np.asarray(expected_scores["semantic"]), expected_permuted_ids, primary["truth_candidate_id"]
        )
        if (
            item["permutation"] != permutation
            or item["original_ordered_candidate_ids"] != ordered
            or item["permuted_ordered_candidate_ids"] != expected_permuted_ids
            or item["truth_candidate_id"] != primary["truth_candidate_id"]
            or item["original_scores_sha256"] != canonical_payload_sha256(primary["scores"])
            or item["permuted_scores"] != expected_scores
            or item["permuted_scores_sha256"] != canonical_payload_sha256(expected_scores)
            or item["original_ranking"] != original_ranking
            or item["permuted_ranking"] != permuted_ranking
        ):
            raise ValueError("permutation evidence does not reindex the primary bank and raw scores")
        return
    if name == "rp2_sign_equivalence":
        if (
            authenticated_support_index is None
            or authenticated_support_index.get("support_index_sha256")
            != primary["candidate_bank_receipt"]["support_index_sha256"]
        ):
            raise ValueError("RP2 verification requires the authenticated frozen support index")
        parent_geometry = primary["candidate_bank_receipt"]["truth_parent_geometry"]
        candidates = primary["candidate_bank_receipt"]["candidates"]
        receipts = item["candidate_receipts"]
        if not isinstance(receipts, list) or len(receipts) != 40:
            raise ValueError("RP2 evidence must contain exactly forty candidates")
        for candidate, receipt in zip(candidates, receipts, strict=True):
            if (
                set(receipt) != {
                    "candidate_id", "source_pose_sha256", "geometry_storage",
                    "stored_candidate_geometry_sha256", "positive_geometry_sha256",
                    "antipodal_geometry_sha256", "equal",
                }
                or receipt["candidate_id"] != candidate["candidate_id"]
                or receipt["source_pose_sha256"] != candidate["pose_sha256"]
                or receipt["geometry_storage"] != candidate["geometry_storage"]
                or receipt["stored_candidate_geometry_sha256"]
                != candidate["candidate_geometry_sha256"]
                or receipt["positive_geometry_sha256"] != receipt["antipodal_geometry_sha256"]
                or receipt["equal"] is not True
            ):
                raise ValueError("RP2 evidence is not an exact ordered antipodal comparison")
            if candidate["candidate_class"] == "truth":
                if (
                    receipt["geometry_storage"] != "truth_parent_geometry"
                    or receipt["stored_candidate_geometry_sha256"]
                    != primary["finite_parent_receipt"]["finite_plane_geometry_sha256"]
                ):
                    raise ValueError("RP2 truth evidence is not bound to the finite parent")
            elif receipt["geometry_storage"] != "candidate":
                raise ValueError("RP2 decoy evidence is not bound to its stored geometry")
            _require_sha256(
                receipt["stored_candidate_geometry_sha256"], "stored RP2 geometry"
            )
            _require_sha256(receipt["positive_geometry_sha256"], "RP2 geometry")
            pose = candidate["physical_pose"]
            normal = np.asarray(
                pose["normal_rp2_sign_aligned_ap_dv_ml"], dtype=np.float64
            )
            offset = float(pose["signed_offset_um"])
            roll = float(pose["roll_delta_rad_from_parallel_transport"])
            positive = transport_finite_candidate_pose(
                parent_geometry,
                authenticated_support_index,
                normal,
                offset,
                roll,
            )
            antipodal = transport_finite_candidate_pose(
                parent_geometry,
                authenticated_support_index,
                -normal,
                -offset,
                roll,
            )
            if (
                positive != antipodal
                or receipt["positive_geometry_sha256"]
                != positive["candidate_geometry_sha256"]
                or receipt["antipodal_geometry_sha256"]
                != antipodal["candidate_geometry_sha256"]
            ):
                raise ValueError(
                    "RP2 evidence does not replay from the authenticated support index"
                )
        return
    if name == "truth_metadata_coordinate_channel_exclusion":
        source = inspect.getsource(score_semantic_candidates)
        signature = list(inspect.signature(score_semantic_candidates).parameters)
        forbidden = ["candidate_id", "truth_index", "coordinate", "geometry", "normal", "offset", "roll"]
        if (
            item["scorer_signature"] != signature
            or item["scorer_source_sha256"] != hashlib.sha256(source.encode("utf-8")).hexdigest()
            or item["forbidden_source_tokens"] != forbidden
            or item["forbidden_matches"] != []
        ):
            raise ValueError("metadata-exclusion evidence does not bind the exact scorer source")
        return
    candidates = primary["candidate_bank_receipt"]["candidates"]
    receipts = item["candidate_receipts"]
    if not isinstance(receipts, list) or len(receipts) != 40:
        raise ValueError("x/W,y/H evidence must contain exactly forty candidates")
    bank = primary["candidate_bank_receipt"]
    for candidate, receipt in zip(candidates, receipts, strict=True):
        geometry = bank["truth_parent_geometry"] if candidate["candidate_class"] == "truth" else candidate["geometry"]
        source = geometry["array_receipts"].get(
            "effective_coordinate_raster_allen_index_float32",
            geometry["array_receipts"].get("coordinate_raster_allen_index_float32"),
        )
        height, width = geometry["output_shape_h_w"]
        ouv = np.asarray(geometry["effective_allen_index_ouv_ap_dv_ml"], dtype=np.float32)
        s = np.arange(width, dtype=np.float32) / np.float32(width)
        t = np.arange(height, dtype=np.float32) / np.float32(height)
        tt, ss = np.meshgrid(t, s, indexing="ij")
        reconstructed = (
            ouv[:3] + ss[..., None] * ouv[3:6] + tt[..., None] * ouv[6:9]
        ).astype(np.float32, copy=False)
        reconstructed_receipt = _array_receipt(reconstructed)
        if (
            set(receipt) != {
                "candidate_id", "source_coordinate_raster_receipt",
                "reconstructed_xy_over_wh_grid_receipt", "output_shape_h_w",
                "grid_point_count", "maximum_absolute_residual_allen_index",
                "inclusive_endpoint_gap_allen_index", "equal_within_float32_tolerance",
            }
            or receipt["candidate_id"] != candidate["candidate_id"]
            or receipt["source_coordinate_raster_receipt"] != source
            or receipt["reconstructed_xy_over_wh_grid_receipt"] != reconstructed_receipt
            or receipt["output_shape_h_w"] != list(OUTPUT_SHAPE)
            or int(receipt["grid_point_count"]) != OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]
            or receipt["reconstructed_xy_over_wh_grid_receipt"].get("shape") != [*OUTPUT_SHAPE, 3]
            or not np.isfinite(float(receipt["maximum_absolute_residual_allen_index"]))
            or float(receipt["maximum_absolute_residual_allen_index"]) > 5e-4
            or not np.isfinite(float(receipt["inclusive_endpoint_gap_allen_index"]))
            or float(receipt["inclusive_endpoint_gap_allen_index"]) <= 1e-6
            or receipt["equal_within_float32_tolerance"] is not True
        ):
            raise ValueError("x/W,y/H evidence is not bound to the full ordered coordinate grids")
        _require_sha256(
            receipt["reconstructed_xy_over_wh_grid_receipt"].get("array_sha256"),
            "reconstructed x/W,y/H grid",
        )


def _verify_resolved_config(config: object) -> dict[str, object]:
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("resolved config does not have the exact production keyset")
    payload = {key: value for key, value in config.items() if key != "resolved_config_sha256"}
    if config["resolved_config_sha256"] != canonical_payload_sha256(payload):
        raise ValueError("resolved config hash does not match")
    repository = config["repository"]
    if (
        not isinstance(repository, dict)
        or set(repository) != {
            "branch", "upstream", "head", "upstream_head", "worktree_clean"
        }
        or repository["branch"] != "codex/joint-registration"
        or not isinstance(repository["upstream"], str)
        or not repository["upstream"].endswith("/codex/joint-registration")
        or repository["worktree_clean"] is not True
        or re.fullmatch(r"[0-9a-f]{40}", str(repository["head"])) is None
        or repository["head"] != repository["upstream_head"]
        or config["source_commit"] != repository["head"]
    ):
        raise ValueError("resolved repository receipt is not a clean frozen branch at its upstream")
    source_hashes = config["source_sha256"]
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_RELATIVE_PATHS):
        raise ValueError("resolved source-hash map does not have the exact frozen files")
    checkout_source_hashes = config["checkout_source_sha256"]
    if (
        not isinstance(checkout_source_hashes, dict)
        or set(checkout_source_hashes) != set(SOURCE_RELATIVE_PATHS)
    ):
        raise ValueError("resolved checkout-source map does not have the exact loaded files")
    for relative_path in SOURCE_RELATIVE_PATHS:
        digest = _require_sha256(source_hashes[relative_path], relative_path)
        if digest != _git_blob_sha256(config["source_commit"], relative_path):
            raise ValueError(f"source hash does not match the recorded Git blob: {relative_path}")
        _require_sha256(
            checkout_source_hashes[relative_path], f"loaded checkout {relative_path}"
        )
    preflight_path = "publication/arbitrary_plane_oracle_pose_ranking_preflight.yaml"
    if config["preflight_sha256"] != source_hashes[preflight_path]:
        raise ValueError("preflight hash is not bound to the exact source-hash map")
    memory_contract = {
        "maximum_live_candidate_banks": MAXIMUM_LIVE_CANDIDATE_BANKS,
        "reason": "one preceding shuffled bank, one current bank, and one transient exact-replay bank; never all 64 banks",
        "candidate_or_synthetic_arrays_saved_to_json": False,
    }
    atlas_assets = {
        "template": {
            "uri": ATLAS_TEMPLATE_URI,
            "raw_sha256": ATLAS_TEMPLATE_SHA256,
            "decoder": "pynrrd 1.1.3",
            "index_order": "F",
        },
        "annotation": {
            "uri": ATLAS_ANNOTATION_URI,
            "raw_sha256": ATLAS_ANNOTATION_SHA256,
            "decoder": "pynrrd 1.1.3",
            "index_order": "F",
        },
    }
    environment = config["environment"]
    if (
        config["schema"] != RUNNER_SCHEMA
        or config["case_root_seed"] != CASE_ROOT_SEED_HEX
        or config["case_seed_algorithm"] != CASE_SEED_ALGORITHM
        or config["case_count"] != CASE_COUNT
        or config["output_shape_h_w"] != list(OUTPUT_SHAPE)
        or config["margin_u_v_um"] != list(MARGIN_UM)
        or config["maximum_case_rejection_attempts"] != MAXIMUM_CASE_ATTEMPTS
        or config["source_hash_contract"] != SOURCE_HASH_CONTRACT
        or config["memory_contract"] != memory_contract
        or config["orientation_counts"] != ORIENTATION_COUNTS
        or config["outline_modes"] != list(OUTLINE_MODES)
        or config["atlas_assets"] != atlas_assets
        or config["shuffled_mapping"] != "(i+17)%64"
        or any(config[key] is not None for key in ("animal_id", "specimen_id", "experiment_id"))
        or any(
            config[key] != []
            for key in (
                "learned_checkpoint_dependencies", "previous_model_dependencies",
                "pretrained_feature_dependencies",
            )
        )
        or any(
            config[key] is not False
            for key in (
                "deepslice_ground_truth_accessed", "real_lab_histology_accessed",
                "final_test_animals_accessed",
            )
        )
        or not isinstance(environment, dict)
        or set(environment) != {"python", "numpy", "scipy", "torch", "pynrrd"}
        or any(not isinstance(value, str) or not value for value in environment.values())
        or environment["pynrrd"] != "1.1.3"
    ):
        raise ValueError("resolved config is not the frozen standalone development protocol")
    return config


def _verify_result_contexts(
    result: dict[str, object],
    config: dict[str, object],
    primary: list[dict[str, object]],
) -> None:
    support_sha256 = _require_sha256(result["support_index_sha256"], "support index")
    render = result["prepared_render_asset_receipt"]
    candidate = result["prepared_candidate_annotation_receipt"]
    if (
        not isinstance(render, dict)
        or set(render) != {
            "schema", "support_index_sha256", "template_decoded", "scalar_conversion",
            "annotation_decoded", "annotation_sampling", "scalar_source",
        }
        or render["schema"] != "anatomy-tracker.prepared-finite-render-context/v1"
        or render["support_index_sha256"] != support_sha256
        or result["prepared_render_context_sha256"] != canonical_payload_sha256(render)
        or not isinstance(candidate, dict)
        or set(candidate) != {"schema", "support_index_sha256", "annotation"}
        or candidate["schema"] != "anatomy-tracker.prepared-finite-candidate-annotation/v1"
        or candidate["support_index_sha256"] != support_sha256
        or result["prepared_candidate_annotation_context_sha256"]
        != canonical_payload_sha256(candidate)
    ):
        raise ValueError("prepared atlas contexts do not match their frozen receipts")
    shape = [528, 320, 456]
    template = render["template_decoded"]
    annotation = render["annotation_decoded"]
    conversion = render["scalar_conversion"]
    sampling = render["annotation_sampling"]
    scalar_source = render["scalar_source"]
    candidate_annotation = candidate["annotation"]
    if (
        not isinstance(template, dict)
        or set(template) != {"decoder", "index_order", "dtype", "shape", "array_sha256"}
        or not isinstance(annotation, dict)
        or set(annotation) != {"decoder", "index_order", "dtype", "shape", "array_sha256"}
        or not isinstance(conversion, dict)
        or set(conversion) != {"operation", "normalization", "dtype", "shape", "array_sha256"}
        or not isinstance(sampling, dict)
        or set(sampling) != {
            "operation", "losslessness", "full_volume_copy", "rendered_output_dtype"
        }
        or not isinstance(scalar_source, dict)
        or set(scalar_source) != {
            "source_entity_type", "uri", "source_sha256", "source_sha256_semantics"
        }
        or not isinstance(candidate_annotation, dict)
        or set(candidate_annotation) != {"dtype", "shape", "array_sha256", "storage"}
    ):
        raise ValueError("prepared atlas receipts do not have their exact production schemas")
    for receipt, name in ((template, "decoded template"), (annotation, "decoded annotation")):
        if (
            receipt["shape"] != shape
            or not isinstance(receipt["dtype"], str)
            or not receipt["dtype"]
            or receipt["decoder"] != "pynrrd 1.1.3"
            or receipt["index_order"] != "F"
        ):
            raise ValueError(f"{name} identity changed")
        _require_sha256(receipt["array_sha256"], name)
    if (
        conversion["operation"] != "numpy.array(dtype=<f4, copy=True, order=C)"
        or conversion["normalization"] != "none"
        or conversion["dtype"] != "<f4"
        or conversion["shape"] != shape
        or sampling["full_volume_copy"] != "none"
        or sampling["rendered_output_dtype"] != "<i8"
        or scalar_source != {
            "source_entity_type": "atlas-template",
            "uri": config["atlas_assets"]["template"]["uri"],
            "source_sha256": config["atlas_assets"]["template"]["raw_sha256"],
            "source_sha256_semantics": "raw source bytes",
        }
        or candidate_annotation["dtype"] != annotation["dtype"]
        or candidate_annotation["shape"] != annotation["shape"]
        or candidate_annotation["array_sha256"] != annotation["array_sha256"]
        or candidate_annotation["storage"] != "owned immutable C-order bytes"
    ):
        raise ValueError("prepared raw/decoded atlas identities are not mutually bound")
    _require_sha256(conversion["array_sha256"], "converted scalar atlas")
    source_hashes = config["checkout_source_sha256"]
    candidate_sources = {
        "candidate_generator": source_hashes["training/arbitrary_plane_finite_candidates.py"],
        "finite_renderer": source_hashes["training/arbitrary_plane_rendered_generator.py"],
        "geometry": source_hashes["training/arbitrary_plane_geometry.py"],
        "manifest": source_hashes["training/arbitrary_plane_manifest.py"],
        "support": source_hashes["training/arbitrary_plane_support.py"],
        "predeclared_protocol": source_hashes[
            "publication/arbitrary_plane_oracle_pose_ranking_preflight.yaml"
        ],
    }
    synthetic_sources = {
        "generator": source_hashes["training/arbitrary_plane_synthetic_generator.py"],
        "ops": source_hashes["training/arbitrary_plane_synthetic_ops.py"],
        "observation": source_hashes["training/arbitrary_plane_synthetic_observation.py"],
        "finite_renderer": source_hashes["training/arbitrary_plane_rendered_generator.py"],
        "predeclared_config": source_hashes[
            "publication/arbitrary_plane_synthetic_preflight.yaml"
        ],
    }
    renderer_dependencies = {
        name: source_hashes[f"training/{name}"]
        for name in (
            "arbitrary_plane_geometry.py", "arbitrary_plane_manifest.py",
            "arbitrary_plane_support.py",
        )
    }
    for case_index, record in enumerate(primary):
        parent = record["finite_parent_receipt"]
        bank = record["candidate_bank_receipt"]
        parent_provenance = parent["provenance"]
        annotation_source = parent_provenance.get("annotation_source", {})
        parent_generator = parent["generator"]
        parent_config = parent_generator.get("resolved_config", {})
        parent_implementation = parent_generator.get("implementation", {})
        bank_generator = bank.get("generator", {})
        bank_config = bank_generator.get("resolved_config", {})
        bank_implementation = bank_generator.get("implementation", {})
        if (
            parent["support_index_sha256"] != support_sha256
            or parent_provenance.get("annotation_decoded") != annotation
            or parent_provenance.get("scalar_source", {}).get("decoded") != template
            or parent_provenance.get("scalar_source", {}).get("float_conversion") != conversion
            or annotation_source.get("annotation_uri")
            != config["atlas_assets"]["annotation"]["uri"]
            or annotation_source.get("source_sha256")
            != config["atlas_assets"]["annotation"]["raw_sha256"]
            or annotation_source.get("annotation_array_sha256") != annotation["array_sha256"]
            or record["provenance"].get("atlas") != parent_provenance.get("atlas")
            or record["provenance"].get("annotation_source") != annotation_source
            or parent_config.get("support_index_sha256") != support_sha256
            or parent_config.get("prepared_context_sha256")
            != result["prepared_render_context_sha256"]
            or parent_config.get("annotation_array_sha256") != annotation["array_sha256"]
            or parent_config.get("template_decoded_array_sha256") != template["array_sha256"]
            or parent_config.get("scalar_source_uri") != ATLAS_TEMPLATE_URI
            or parent_config.get("scalar_source_sha256") != ATLAS_TEMPLATE_SHA256
            or parent_config.get("output_shape_h_w") != list(OUTPUT_SHAPE)
            or parent_config.get("margin_u_v_um") != list(MARGIN_UM)
            or parent_config.get("max_rejection_attempts") != 1
            or parent_config.get("minimum_brain_pixels") != 64
            or parent_config.get("numpy_version") != config["environment"]["numpy"]
            or parent_config.get("torch_version") != config["environment"]["torch"]
            or any(parent_config.get(key) is not None for key in (
                "animal_id", "specimen_id", "experiment_id"
            ))
            or parent_implementation.get("source_commit") != config["source_commit"]
            or parent_implementation.get("loaded_source_sha256")
            != source_hashes["training/arbitrary_plane_rendered_generator.py"]
            or parent_implementation.get("loaded_dependency_source_sha256")
            != renderer_dependencies
            or bank.get("support_index_sha256") != support_sha256
            or bank.get("finite_parent_receipt") != parent
            or bank.get("provenance") != parent_provenance
            or bank_config.get("support_index_sha256") != support_sha256
            or bank_config.get("annotation_array_sha256") != annotation["array_sha256"]
            or bank_config.get("prepared_annotation_context_sha256")
            != result["prepared_candidate_annotation_context_sha256"]
            or bank_config.get("output_shape_h_w") != list(OUTPUT_SHAPE)
            or bank_implementation.get("loaded_source_sha256") != candidate_sources
            or bank_implementation.get("numpy_version") != config["environment"]["numpy"]
            or bank_implementation.get("torch_version") != config["environment"]["torch"]
        ):
            raise ValueError(f"case {case_index} is not bound to the frozen source and atlas contexts")
        for descendant in record["outline_descendants"]:
            synthetic = descendant["synthetic_receipt"]
            synthetic_generator = synthetic.get("generator", {})
            implementation = synthetic_generator.get("implementation", {})
            resolved = synthetic_generator.get("resolved_config")
            if (
                synthetic.get("support_index_sha256") != support_sha256
                or synthetic.get("provenance") != parent_provenance
                or implementation.get("loaded_source_sha256") != synthetic_sources
                or implementation.get("numpy_version") != config["environment"]["numpy"]
                or implementation.get("scipy_version") != config["environment"]["scipy"]
                or synthetic_generator.get("implementation_sha256")
                != canonical_payload_sha256(implementation)
                or not isinstance(resolved, dict)
                or synthetic_generator.get("resolved_config_sha256")
                != canonical_payload_sha256(resolved)
            ):
                raise ValueError(
                    f"case {case_index} synthetic receipt is not bound to frozen sources"
                )


def _write_and_verify_result(
    output: Path,
    result: dict[str, object],
    authenticated_support_index: dict[str, object],
) -> dict[str, object]:
    _atomic_json(output / "result.json", result)
    verified = _verify_written_result(output, authenticated_support_index)
    if verified != result:
        raise ValueError("mandatory post-write verification did not reproduce the in-memory result")
    return verified


def run_oracle(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output = Path(output)
    repository = repository_state()
    source_commit = repository["head"]
    source_hashes, checkout_source_hashes = _source_hash_receipts(source_commit)
    config = {
        "schema": RUNNER_SCHEMA,
        "source_commit": source_commit,
        "repository": repository,
        "source_sha256": source_hashes,
        "checkout_source_sha256": checkout_source_hashes,
        "source_hash_contract": SOURCE_HASH_CONTRACT,
        "preflight_sha256": source_hashes[
            "publication/arbitrary_plane_oracle_pose_ranking_preflight.yaml"
        ],
        "case_root_seed": CASE_ROOT_SEED_HEX,
        "case_seed_algorithm": CASE_SEED_ALGORITHM,
        "case_count": CASE_COUNT,
        "output_shape_h_w": list(OUTPUT_SHAPE),
        "margin_u_v_um": list(MARGIN_UM),
        "maximum_case_rejection_attempts": MAXIMUM_CASE_ATTEMPTS,
        "memory_contract": {
            "maximum_live_candidate_banks": MAXIMUM_LIVE_CANDIDATE_BANKS,
            "reason": "one preceding shuffled bank, one current bank, and one transient exact-replay bank; never all 64 banks",
            "candidate_or_synthetic_arrays_saved_to_json": False,
        },
        "orientation_counts": ORIENTATION_COUNTS,
        "outline_modes": list(OUTLINE_MODES),
        "atlas_assets": {
            "template": {"uri": ATLAS_TEMPLATE_URI, "raw_sha256": ATLAS_TEMPLATE_SHA256, "decoder": "pynrrd 1.1.3", "index_order": "F"},
            "annotation": {"uri": ATLAS_ANNOTATION_URI, "raw_sha256": ATLAS_ANNOTATION_SHA256, "decoder": "pynrrd 1.1.3", "index_order": "F"},
        },
        "shuffled_mapping": "(i+17)%64",
        "animal_id": None,
        "specimen_id": None,
        "experiment_id": None,
        "learned_checkpoint_dependencies": [],
        "previous_model_dependencies": [],
        "pretrained_feature_dependencies": [],
        "deepslice_ground_truth_accessed": False,
        "real_lab_histology_accessed": False,
        "final_test_animals_accessed": False,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": version("scipy"),
            "torch": version("torch"),
            "pynrrd": version("pynrrd"),
        },
    }
    config["resolved_config_sha256"] = canonical_payload_sha256(config)
    _atomic_json(output / "resolved_config.json", config)
    print(json.dumps({"event": "config-frozen", "resolved_config_sha256": config["resolved_config_sha256"]}), flush=True)
    support, render_context, candidate_context = load_allen_contexts()
    print(json.dumps({"event": "contexts-ready", "support_index_sha256": support["support_index_sha256"]}), flush=True)

    primary_records: dict[int, dict[str, object]] = {}
    shuffled_records: dict[int, dict[str, object]] = {}
    evidence_references = {name: [] for name in EXACT_CONTROL_NAMES}
    first_target = None
    first_target_record = None
    pending = None
    for case_index in shuffled_case_cycle():
        record, target, current = build_oracle_case(case_index, render_context, candidate_context, support, source_commit)
        current["primary_record"] = record
        primary_records[case_index] = record
        _write_mask_sidecar(output, record["target"], target["fixed_valid_mask"])
        primary_file_sha256 = _atomic_json(
            output / "primary" / f"case-{case_index:03d}.json", record
        )
        print(
            json.dumps(
                {
                    "event": "base-ready",
                    "case_index": case_index,
                    "orientation": record["orientation"],
                    "accepted_attempt_index": record["accepted_case_attempt_index"],
                    "primary_file_sha256": primary_file_sha256,
                }
            ),
            flush=True,
        )
        replay = replay_oracle_case(record, target, current, render_context, candidate_context, support, source_commit)
        replay["control"] = "exact_replay"
        evidence_references["exact_replay"].append(_write_control_sidecar(output, replay))
        controls = _case_controls(record, target, current, support)
        for name, item in controls.items():
            item["control"] = name
            evidence_references[name].append(_write_control_sidecar(output, item))
        if case_index == 0:
            first_target = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in target.items()}
            first_target_record = record
        if pending is not None:
            shuffled = _shuffled_record(pending, record, target)
            shuffled_records[int(pending["case_index"])] = shuffled
            shuffled_hash = _atomic_json(output / "shuffled" / f"case-{int(pending['case_index']):03d}.json", shuffled)
            print(json.dumps({"event": "case-complete", "case_index": int(pending["case_index"]), "primary_sha256": primary_records[int(pending["case_index"])]["case_payload_sha256"], "shuffled_file_sha256": shuffled_hash}), flush=True)
        pending = current
    if pending is None or first_target is None or first_target_record is None:
        raise RuntimeError("frozen case stream produced no cases")
    shuffled = _shuffled_record(pending, first_target_record, first_target)
    shuffled_records[int(pending["case_index"])] = shuffled
    shuffled_hash = _atomic_json(output / "shuffled" / f"case-{int(pending['case_index']):03d}.json", shuffled)
    print(json.dumps({"event": "case-complete", "case_index": int(pending["case_index"]), "primary_sha256": primary_records[int(pending["case_index"])]["case_payload_sha256"], "shuffled_file_sha256": shuffled_hash}), flush=True)

    exact_controls = {
        name: _control_record(name, items) for name, items in evidence_references.items()
    }
    primary_gate = [_gate_view(primary_records[index]) for index in range(CASE_COUNT)]
    shuffled_gate = [_gate_view(shuffled_records[index]) for index in range(CASE_COUNT)]
    summary = semantic_gate_summary(primary_gate, shuffled_gate, exact_controls=exact_controls)
    result = {
        "schema": RUNNER_SCHEMA,
        "resolved_config": config,
        "support_index_sha256": support["support_index_sha256"],
        "prepared_render_context_sha256": render_context["prepared_context_sha256"],
        "prepared_candidate_annotation_context_sha256": candidate_context["prepared_context_sha256"],
        "prepared_render_asset_receipt": _plain(render_context["asset_receipt"]),
        "prepared_candidate_annotation_receipt": _plain(candidate_context["receipt"]),
        "primary_case_payload_sha256": [primary_records[index]["case_payload_sha256"] for index in range(CASE_COUNT)],
        "shuffled_case_payload_sha256": [shuffled_records[index]["shuffled_payload_sha256"] for index in range(CASE_COUNT)],
        "exact_controls": exact_controls,
        "semantic_gate": summary,
        "interpretation": INTERPRETATION,
    }
    result["result_payload_sha256"] = canonical_payload_sha256(result)
    _write_and_verify_result(output, result, support)
    print(json.dumps({"event": "run-complete", "passed": summary["passed"], "result_payload_sha256": result["result_payload_sha256"]}), flush=True)
    return result


def _load_verified_masks(
    output: Path, primary: list[dict[str, object]]
) -> list[np.ndarray]:
    masks = []
    for case_index, record in enumerate(primary):
        target = record["target"]
        receipt = target["fixed_valid_mask_binary"]
        expected_path = f"masks/case-{case_index:03d}.bin"
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {
                "encoding", "relative_path", "shape_h_w", "bit_count", "byte_count",
                "payload_sha256",
            }
            or receipt["encoding"] != "numpy.packbits-C-order-little-bitorder/v1"
            or receipt["relative_path"] != expected_path
            or receipt["shape_h_w"] != list(OUTPUT_SHAPE)
            or int(receipt["bit_count"]) != OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]
            or int(receipt["byte_count"]) != (OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1] + 7) // 8
        ):
            raise ValueError("fixed-valid mask binary receipt is not the exact frozen encoding")
        payload = (output / expected_path).read_bytes()
        if (
            len(payload) != receipt["byte_count"]
            or hashlib.sha256(payload).hexdigest() != receipt["payload_sha256"]
        ):
            raise ValueError("fixed-valid mask binary does not match its payload receipt")
        mask = np.unpackbits(
            np.frombuffer(payload, dtype=np.uint8), bitorder="little",
            count=int(receipt["bit_count"]),
        ).astype(bool, copy=False).reshape(OUTPUT_SHAPE)
        if (
            _array_receipt(mask) != target["mask_receipt"]
            or int(mask.sum()) != int(target["fixed_valid_pixel_count"])
        ):
            raise ValueError("fixed-valid mask binary does not replay its target mask receipt")
        masks.append(mask)
    return masks


def _verify_written_result(
    output: Path,
    authenticated_support_index: dict[str, object],
) -> dict[str, object]:
    output = Path(output)
    verify_annotation_support_index(authenticated_support_index)
    expected_files = {"resolved_config.json", "result.json"}
    expected_files.update(f"primary/case-{index:03d}.json" for index in range(CASE_COUNT))
    expected_files.update(f"shuffled/case-{index:03d}.json" for index in range(CASE_COUNT))
    expected_files.update(f"masks/case-{index:03d}.bin" for index in range(CASE_COUNT))
    expected_files.update(
        f"controls/{name}/case-{index:03d}.json"
        for name in EXACT_CONTROL_NAMES
        for index in range(CASE_COUNT)
    )
    actual_files = {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(
            f"output tree differs from the exact frozen file set; missing={sorted(expected_files - actual_files)}, unexpected={sorted(actual_files - expected_files)}"
        )
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    if (
        not isinstance(result, dict)
        or set(result) != RESULT_KEYS
        or result.get("schema") != RUNNER_SCHEMA
        or result.get("interpretation") != INTERPRETATION
    ):
        raise ValueError("result does not have the exact production schema")
    payload = {key: value for key, value in result.items() if key != "result_payload_sha256"}
    if result["result_payload_sha256"] != canonical_payload_sha256(payload):
        raise ValueError("result payload hash does not match")
    resolved_config = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    if resolved_config != result.get("resolved_config"):
        raise ValueError("resolved config sidecar does not match the result")
    _verify_resolved_config(resolved_config)
    primary = [json.loads((output / "primary" / f"case-{index:03d}.json").read_text(encoding="utf-8")) for index in range(CASE_COUNT)]
    shuffled = [json.loads((output / "shuffled" / f"case-{index:03d}.json").read_text(encoding="utf-8")) for index in range(CASE_COUNT)]
    for item in primary:
        payload = {key: value for key, value in item.items() if key != "case_payload_sha256"}
        if item["case_payload_sha256"] != canonical_payload_sha256(payload):
            raise ValueError("primary case payload hash does not match")
    for item in shuffled:
        payload = {key: value for key, value in item.items() if key != "shuffled_payload_sha256"}
        if item["shuffled_payload_sha256"] != canonical_payload_sha256(payload):
            raise ValueError("shuffled case payload hash does not match")
    for case_index, item in enumerate(primary):
        _verify_primary_record(item, case_index, result["support_index_sha256"])
    for case_index, item in enumerate(shuffled):
        _verify_shuffled_record(item, primary[case_index], case_index)
        if item["target"] != primary[shuffled_target_index(case_index)]["target"]:
            raise ValueError("shuffled target does not match its frozen source case")
    if (
        authenticated_support_index["support_index_sha256"]
        != result["support_index_sha256"]
    ):
        raise ValueError("result does not match the authenticated frozen support index")
    masks = _load_verified_masks(output, primary)
    _verify_result_contexts(result, resolved_config, primary)
    if [item["case_payload_sha256"] for item in primary] != result["primary_case_payload_sha256"] or [item["shuffled_payload_sha256"] for item in shuffled] != result["shuffled_case_payload_sha256"]:
        raise ValueError("case sidecars do not match the frozen result")
    if not isinstance(result["exact_controls"], dict) or set(result["exact_controls"]) != set(
        EXACT_CONTROL_NAMES
    ):
        raise ValueError("result does not contain exactly the five frozen controls")
    for name in EXACT_CONTROL_NAMES:
        aggregate = result["exact_controls"].get(name)
        if not isinstance(aggregate, dict):
            raise ValueError("result lacks one named exact control")
        references = aggregate.get("evidence", {}).get("case_evidence", [])
        if len(references) != CASE_COUNT:
            raise ValueError("exact control lacks 64 full evidence sidecar references")
        for case_index, reference in enumerate(references):
            expected_relative = (Path("controls") / name / f"case-{case_index:03d}.json").as_posix()
            if int(reference.get("case_index", -1)) != case_index or reference.get("relative_path") != expected_relative:
                raise ValueError("exact control sidecar reference order or path changed")
            path = output / expected_relative
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != reference.get("file_sha256"):
                raise ValueError("exact control evidence file hash does not match")
            item = json.loads(payload)
            if (
                item.get("control") != name
                or int(item.get("case_index", -1)) != case_index
                or type(item.get("passed")) is not bool
                or item["passed"] != reference.get("passed")
                or canonical_payload_sha256(item) != reference.get("payload_sha256")
            ):
                raise ValueError("exact control evidence payload does not match its reference")
            _verify_control_payload(
                name, item, primary[case_index], authenticated_support_index
            )
        if _control_record(name, references) != aggregate:
            raise ValueError("exact control aggregate does not replay from its evidence sidecars")
    replayed = semantic_gate_summary([_gate_view(item) for item in primary], [_gate_view(item) for item in shuffled], exact_controls=result["exact_controls"])
    for case_index, (saved, derived) in enumerate(
        zip(primary, replayed["primary_rankings"], strict=True)
    ):
        _verify_saved_ranking(
            saved,
            derived,
            "primary",
            saved["candidate_bank_receipt"],
            masks[case_index],
        )
    for case_index, (saved, derived) in enumerate(
        zip(shuffled, replayed["shuffled_rankings"], strict=True)
    ):
        _verify_saved_ranking(
            saved,
            derived,
            "shuffled",
            primary[case_index]["candidate_bank_receipt"],
            masks[shuffled_target_index(case_index)],
        )
    if replayed != result["semantic_gate"]:
        raise ValueError("semantic gate does not replay from raw sidecars")
    return result


def verify_written_result(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    return _verify_written_result(output, _load_authenticated_allen_support_index())


if __name__ == "__main__":
    run_oracle()
