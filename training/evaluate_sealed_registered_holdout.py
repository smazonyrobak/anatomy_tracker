"""Final-only evaluation on the sealed Allen S2P experiments.

This module is intentionally independent of every training and model-selection
entry point.  Its outputs are test reports, never training metadata.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import queue
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from source.atlas_pose_runtime import (
    ATLAS_POSE_SEALED_BENCHMARK_ID as SEALED_BENCHMARK_ID,
    ATLAS_POSE_SEALED_EXPERIMENT_COUNT as EXPECTED_EXPERIMENTS,
    ATLAS_POSE_SEALED_SECTION_COUNT as EXPECTED_SECTIONS,
    ATLAS_POSE_SEALED_SOURCE_FILES as SEALED_SOURCE_FILES,
    ATLAS_POSE_SEALED_SPLIT as SEALED_SPLIT,
    automatic_brain_mask,
    run_atlas_pose_candidate_onnx,
    verify_atlas_pose_candidate_bundle,
)
from source.deepslice_runtime import (
    load_deepslice_onnx_sessions,
    preprocess_deepslice_images,
    run_deepslice_inference,
)
from source.registered_image_quality import (
    load_registered_image_quality_manifest,
)
from training.atlas_pose_release_contract import (
    POSE_AXES,
    RELEASE_CONFIDENCE,
    RELEASE_REFERENCE,
    evaluation_domains,
    paired_animal_bootstrap,
    paired_animal_joint_superiority,
    release_quality_gate,
    validate_complete_method_cohort,
)


# Final-only v7 comparison against frozen DeepSlice modes on specimen-separated registered sections.
ROOT = Path(__file__).resolve().parents[1]
ACQUISITION_ROOT = Path(
    os.environ.get(
        "ALLEN_REGISTERED_ROOT",
        "J:/AtlasPoseTraining_v7/allen_registered_full_quicknii_ras_v2_20260811",
    )
)
ANNOTATION_PATH = Path(
    os.environ.get(
        "ALLEN_ANNOTATION_25_PATH",
        ROOT / "data/Allen Brain Atlas 25um/annotation_25.nrrd",
    )
)
ATLAS_POSE_MODEL = os.environ.get("ATLAS_POSE_SEALED_MODEL")
SEALED_EVALUATION_STATE_ROOT = (
    Path.home() / "AppData/Local/Proprietary Anatomy Tracker/AtlasPose Sealed Evaluation"
    if os.name == "nt"
    else Path.home() / ".local/state/proprietary-anatomy-tracker/atlaspose-sealed-evaluation"
)
SEALED_CLAIM_NAME = f"{SEALED_BENCHMARK_ID}.claim.json"
SEALED_RECEIPT_NAME = f"{SEALED_BENCHMARK_ID}.receipt.json"
SEALED_RECOVERY_MODE = "diagnostic-empty-annotation-mask-v1"
SEALED_RECOVERABLE_FAILURE = (
    "ValueError: The plane-distance metric needs a non-empty 2-D brain mask"
)
ATLAS_POSE_IMAGE_BATCH = 16
VOXEL_UM = 25.0
QUICKNII_SHAPE_ML_AP_DV = np.asarray((456.0, 528.0, 320.0))
ATLAS_CENTER_ML_DV = np.asarray((227.5, 159.5))
BREGMA_AP_INDEX = 216.0
OUV_COLUMNS = ("ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz")
POSE_TOLERANCES = {
    "ap_um": (50.0, 100.0, 250.0),
    "lr_deg": (1.0, 2.0, 5.0),
    "dv_deg": (1.0, 2.0, 5.0),
}
AP_BANDS = (
    ("above_+500_um", 500.0, np.inf),
    ("+500_to_0_um", 0.0, 500.0),
    ("0_to_-1000_um", -1000.0, 0.0),
    ("-1000_to_-2000_um", -2000.0, -1000.0),
    ("-2000_to_-3000_um", -3000.0, -2000.0),
    ("-3000_to_-4000_um", -4000.0, -3000.0),
    ("-4000_to_-4500_um", -4500.0, -4000.0),
    ("below_-4500_um", -np.inf, -4500.0),
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_hash_tree(value) -> bool:
    return bool(value) and (
        _is_sha256(value)
        or isinstance(value, dict)
        and all(isinstance(key, str) and _valid_hash_tree(child) for key, child in value.items())
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _immutable_copy(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256(destination) != expected_sha256:
            raise RuntimeError(f"Frozen candidate snapshot differs from its commitment: {destination}")
        return
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Candidate changed while it was being frozen: {source}")
    os.replace(temporary, destination)


def _distribution_version(name: str) -> str:
    return importlib.metadata.version(name)


def evaluator_environment_commitment() -> dict:
    deepslice = importlib.util.find_spec("DeepSlice")
    if deepslice is None or deepslice.submodule_search_locations is None:
        raise RuntimeError("The validated DeepSlice package is unavailable")
    package = Path(next(iter(deepslice.submodule_search_locations)))
    source_paths = (
        Path(__file__),
        ROOT / "training" / "atlas_pose_release_contract.py",
        ROOT / "source" / "atlas_pose_runtime.py",
        ROOT / "source" / "deepslice_runtime.py",
        ROOT / "source" / "registered_image_quality.py",
        *sorted(package.rglob("*.py")),
    )
    model_paths = {
        "primary": ROOT / "models/DeepSlice/deepslice_mouse_primary_opset18.onnx",
        "secondary": ROOT / "models/DeepSlice/deepslice_mouse_secondary_opset18.onnx",
    }
    missing = [path for path in (*source_paths, *model_paths.values()) if not path.is_file()]
    if missing:
        raise RuntimeError(f"Sealed evaluator dependencies are unavailable: {missing}")
    source_hashes = {
        (
            path.resolve().relative_to(ROOT).as_posix()
            if path.resolve().is_relative_to(ROOT)
            else f"DeepSlice/{path.relative_to(package).as_posix()}"
        ): sha256(path)
        for path in source_paths
    }
    payload = {
        "contract_version": 1,
        "source_sha256": source_hashes,
        "deepslice_model_sha256": {
            name: sha256(path) for name, path in model_paths.items()
        },
        "dependencies": {
            "python": platform.python_version(),
            **{
                name: _distribution_version(name)
                for name in (
                    "DeepSlice",
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "scikit-image",
                    "tensorflow",
                    "h5py",
                    "requests",
                    "protobuf",
                    "lxml",
                    "urllib3",
                    "Pillow",
                    "opencv-python",
                    "onnxruntime-directml",
                )
            },
        },
    }
    payload["commitment_sha256"] = _canonical_json_sha256(payload)
    return payload


def freeze_candidate(model_path: Path) -> dict:
    model_path = Path(model_path)
    model_sha256, metadata_sha256, metadata = verify_atlas_pose_candidate_bundle(model_path)
    registered = metadata.get("registered_data", {})
    registered_commitment = registered.get("sha256")
    source_commitment = {
        name: (registered_commitment or {}).get(name) for name in SEALED_SOURCE_FILES
    }
    if (
        set(registered_commitment or {})
        != {*SEALED_SOURCE_FILES, "nonsealed_image_tree_sha256"}
        or not _valid_hash_tree(source_commitment)
        or not _is_sha256((registered_commitment or {}).get("nonsealed_image_tree_sha256"))
        or SEALED_SPLIT not in registered.get("excluded_from_selection", [])
        or not _valid_hash_tree(metadata.get("source_sha256"))
        or not _valid_hash_tree(metadata.get("manifest_sha256"))
        or not _valid_hash_tree(metadata.get("atlas_data_sha256"))
        or metadata.get("git", {}).get("tracked_source_dirty") is not False
    ):
        raise RuntimeError(
            "AtlasPose candidate lacks a clean training/source/atlas commitment or sealed exclusion"
        )
    snapshot = SEALED_EVALUATION_STATE_ROOT / "candidate" / model_sha256
    frozen_model = snapshot / "atlas_pose.onnx"
    frozen_metadata = snapshot / "atlas_pose.json"
    _immutable_copy(model_path, frozen_model, model_sha256)
    _immutable_copy(model_path.with_suffix(".json"), frozen_metadata, metadata_sha256)
    presealed = {
        "contract_version": 1,
        "benchmark_id": SEALED_BENCHMARK_ID,
        "model_sha256": model_sha256,
        "metadata_sha256": metadata_sha256,
        "training_source_sha256": metadata["source_sha256"],
        "training_data_sha256": {
            "synthetic_manifests": metadata.get("manifest_sha256"),
            "registered_data": registered_commitment,
            "atlas_data": metadata.get("atlas_data_sha256"),
        },
        "sealed_source_sha256": source_commitment,
        "evaluator_environment": evaluator_environment_commitment(),
    }
    presealed_path = snapshot / "PRESEALED_COMMITMENT.json"
    if presealed_path.is_file():
        if json.loads(presealed_path.read_text(encoding="utf-8")) != presealed:
            raise RuntimeError("Frozen candidate presealed commitment changed")
    else:
        _atomic_json(presealed_path, presealed)
    return {
        "model_path": frozen_model,
        "metadata_path": frozen_metadata,
        "presealed_path": presealed_path,
        "model_sha256": model_sha256,
        "metadata_sha256": metadata_sha256,
        "metadata": metadata,
        "presealed": presealed,
        "presealed_sha256": sha256(presealed_path),
    }


def claim_sealed_benchmark(frozen: dict) -> tuple[dict, Path, Path]:
    SEALED_EVALUATION_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    claim_path = SEALED_EVALUATION_STATE_ROOT / SEALED_CLAIM_NAME
    receipt_path = SEALED_EVALUATION_STATE_ROOT / SEALED_RECEIPT_NAME
    claim = {
        "contract_version": 1,
        "benchmark_id": SEALED_BENCHMARK_ID,
        "model_sha256": frozen["model_sha256"],
        "metadata_sha256": frozen["metadata_sha256"],
        "presealed_commitment_sha256": frozen["presealed_sha256"],
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_access_permitted_after_claim_only": True,
    }
    try:
        with claim_path.open("x", encoding="utf-8") as stream:
            json.dump(claim, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise RuntimeError(
            f"SEALED EVALUATION REFUSED: {SEALED_BENCHMARK_ID} was already consumed"
        ) from error
    _atomic_json(
        receipt_path,
        {
            "contract_version": 1,
            "benchmark_id": SEALED_BENCHMARK_ID,
            "claim_sha256": sha256(claim_path),
            "model_sha256": frozen["model_sha256"],
            "status": "claimed",
        },
    )
    return claim, claim_path, receipt_path


def load_frozen_candidate_for_recovery(model_path: Path) -> dict:
    model_sha256, metadata_sha256, metadata = verify_atlas_pose_candidate_bundle(model_path)
    snapshot = SEALED_EVALUATION_STATE_ROOT / "candidate" / model_sha256
    frozen_model = snapshot / "atlas_pose.onnx"
    frozen_metadata = snapshot / "atlas_pose.json"
    presealed_path = snapshot / "PRESEALED_COMMITMENT.json"
    if (
        not frozen_model.is_file()
        or not frozen_metadata.is_file()
        or not presealed_path.is_file()
        or sha256(frozen_model) != model_sha256
        or sha256(frozen_metadata) != metadata_sha256
    ):
        raise RuntimeError("SEALED RECOVERY REFUSED: the original frozen candidate is unavailable")
    presealed = json.loads(presealed_path.read_text(encoding="utf-8"))
    registered = metadata.get("registered_data", {}).get("sha256")
    expected = {
        "contract_version": 1,
        "benchmark_id": SEALED_BENCHMARK_ID,
        "model_sha256": model_sha256,
        "metadata_sha256": metadata_sha256,
        "training_source_sha256": metadata.get("source_sha256"),
        "training_data_sha256": {
            "synthetic_manifests": metadata.get("manifest_sha256"),
            "registered_data": registered,
            "atlas_data": metadata.get("atlas_data_sha256"),
        },
        "sealed_source_sha256": {
            name: (registered or {}).get(name) for name in SEALED_SOURCE_FILES
        },
    }
    if any(presealed.get(key) != value for key, value in expected.items()):
        raise RuntimeError("SEALED RECOVERY REFUSED: the original commitment does not bind this candidate")
    return {
        "model_path": frozen_model,
        "metadata_path": frozen_metadata,
        "presealed_path": presealed_path,
        "model_sha256": model_sha256,
        "metadata_sha256": metadata_sha256,
        "metadata": metadata,
        "presealed": presealed,
        "presealed_sha256": sha256(presealed_path),
    }


def recover_failed_sealed_benchmark(
    frozen: dict,
    acquisition_root: Path,
) -> tuple[dict, Path, Path, dict]:
    claim_path = SEALED_EVALUATION_STATE_ROOT / SEALED_CLAIM_NAME
    receipt_path = SEALED_EVALUATION_STATE_ROOT / SEALED_RECEIPT_NAME
    if not claim_path.is_file() or not receipt_path.is_file():
        raise RuntimeError("SEALED RECOVERY REFUSED: the failed claim and receipt are required")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    failed_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        claim.get("model_sha256") != frozen["model_sha256"]
        or claim.get("metadata_sha256") != frozen["metadata_sha256"]
        or claim.get("presealed_commitment_sha256") != frozen["presealed_sha256"]
        or failed_receipt.get("status") != "failed"
        or failed_receipt.get("failure") != SEALED_RECOVERABLE_FAILURE
        or failed_receipt.get("claim_sha256") != sha256(claim_path)
        or failed_receipt.get("model_sha256") != frozen["model_sha256"]
        or failed_receipt.get("presealed_commitment_sha256") != frozen["presealed_sha256"]
    ):
        raise RuntimeError("SEALED RECOVERY REFUSED: the failure is not the audited diagnostic failure")
    output_root = Path(acquisition_root) / "SEALED_FINAL_EVALUATION"
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError("SEALED RECOVERY REFUSED: sealed result artifacts already exist")

    recovery_root = SEALED_EVALUATION_STATE_ROOT / "recovery" / frozen["model_sha256"]
    failed_claim_path = recovery_root / "FAILED_ATTEMPT_CLAIM.json"
    failed_receipt_path = recovery_root / "FAILED_ATTEMPT_RECEIPT.json"
    _immutable_copy(claim_path, failed_claim_path, sha256(claim_path))
    _immutable_copy(receipt_path, failed_receipt_path, sha256(receipt_path))
    current_environment = evaluator_environment_commitment()
    commitment = {
        "contract_version": 1,
        "benchmark_id": SEALED_BENCHMARK_ID,
        "recovery_mode": SEALED_RECOVERY_MODE,
        "model_sha256": frozen["model_sha256"],
        "metadata_sha256": frozen["metadata_sha256"],
        "presealed_commitment_sha256": frozen["presealed_sha256"],
        "original_claim_sha256": sha256(failed_claim_path),
        "failed_attempt_receipt_sha256": sha256(failed_receipt_path),
        "failed_attempt_error": SEALED_RECOVERABLE_FAILURE,
        "original_evaluator_environment_sha256": frozen["presealed"][
            "evaluator_environment"
        ]["commitment_sha256"],
        "recovery_evaluator_environment": current_environment,
        "repair_scope": (
            "An empty annotation mask now makes only the optional QuickNII plane-distance "
            "diagnostic unavailable; AP/L-R/D-V predictions and release metrics are unchanged."
        ),
        "sealed_result_artifacts_existed_before_recovery": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    commitment_path = recovery_root / "SEALED_RECOVERY_COMMITMENT.json"
    if commitment_path.exists():
        raise RuntimeError("SEALED RECOVERY REFUSED: this audited recovery was already consumed")
    _atomic_json(commitment_path, commitment)
    recovery = {
        "commitment": commitment,
        "commitment_path": commitment_path,
        "commitment_sha256": sha256(commitment_path),
        "failed_claim_path": failed_claim_path,
        "failed_receipt_path": failed_receipt_path,
        "failed_receipt_sha256": sha256(failed_receipt_path),
    }
    _atomic_json(
        receipt_path,
        {
            "contract_version": 1,
            "benchmark_id": SEALED_BENCHMARK_ID,
            "claim_sha256": sha256(claim_path),
            "model_sha256": frozen["model_sha256"],
            "presealed_commitment_sha256": frozen["presealed_sha256"],
            "sealed_recovery_commitment_sha256": recovery["commitment_sha256"],
            "failed_attempt_receipt_sha256": recovery["failed_receipt_sha256"],
            "status": "claimed_recovery",
        },
    )
    return claim, claim_path, receipt_path, recovery


def verify_source_commitment(root: Path, annotation_path: Path, frozen: dict) -> dict:
    expected = frozen["presealed"]["sealed_source_sha256"]
    actual = {name: sha256(Path(root) / name) for name in SEALED_SOURCE_FILES}
    if actual != expected:
        raise RuntimeError("SEALED INFERENCE REFUSED: acquisition source differs from the frozen candidate")
    atlas_expected = frozen["metadata"].get("atlas_data_sha256", {}).get("annotation_25.nrrd")
    if not _is_sha256(atlas_expected) or sha256(annotation_path) != atlas_expected:
        raise RuntimeError("SEALED INFERENCE REFUSED: Allen annotation differs from the candidate")
    return actual


def verify_complete_sealed_image_hashes(root: Path) -> str:
    records = validate_sealed_boundary(read_jsonl(root / "sections.jsonl"))
    download_rows = read_jsonl(root / "downloads.jsonl")
    downloads = {int(row["section_image_id"]): row["sha256"] for row in download_rows}
    if len(downloads) != len(download_rows) or any(not _is_sha256(value) for value in downloads.values()):
        raise ValueError("Download manifest has duplicate IDs or invalid SHA-256 values")
    if len(records) != EXPECTED_SECTIONS or len({int(row["experiment_id"]) for row in records}) != EXPECTED_EXPERIMENTS:
        raise RuntimeError("SEALED INFERENCE REFUSED: sealed image cohort is incomplete")
    verified = []
    for record in records:
        section_id = int(record["section_image_id"])
        path = root / record["relative_path"]
        expected = downloads.get(section_id)
        if expected is None or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"SEALED INFERENCE REFUSED: image checksum failed for section {section_id}")
        verified.append((section_id, expected))
    return hashlib.sha256(
        json.dumps(sorted(verified), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_sealed_boundary(records: list[dict]) -> list[dict]:
    """Return sealed records while rejecting specimen or experiment leakage."""
    required = {
        "section_image_id",
        "experiment_id",
        "specimen_id",
        "split",
        "section_number",
        "quicknii_ouv",
        "relative_path",
        "in_training_ap_domain",
    }
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Section record is missing {sorted(missing)}")

    sealed = [record for record in records if record["split"] == SEALED_SPLIT]
    if not sealed:
        raise ValueError("No sealed Allen S2P records were found")
    sealed_specimens = {int(record["specimen_id"]) for record in sealed}
    sealed_experiments = {int(record["experiment_id"]) for record in sealed}
    leaked = [
        record
        for record in records
        if record["split"] != SEALED_SPLIT
        and (
            int(record["specimen_id"]) in sealed_specimens
            or int(record["experiment_id"]) in sealed_experiments
        )
    ]
    if leaked:
        raise ValueError("A sealed specimen or experiment also occurs in a train/validation/test split")

    section_ids = [int(record["section_image_id"]) for record in sealed]
    if len(section_ids) != len(set(section_ids)):
        raise ValueError("The sealed manifest contains duplicate section_image_id values")
    for record in sealed:
        ouv = np.asarray(record["quicknii_ouv"], dtype=np.float64)
        if ouv.shape != (9,) or not np.isfinite(ouv).all():
            raise ValueError(f"Invalid recorded QuickNII OUV for section {record['section_image_id']}")
    return sorted(
        sealed,
        key=lambda record: (
            int(record["experiment_id"]),
            int(record["section_number"]),
            int(record["section_image_id"]),
        ),
    )


def ordered_experiment_groups(records: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for record in records:
        groups.setdefault(int(record["experiment_id"]), []).append(record)
    return {
        experiment_id: sorted(
            groups[experiment_id],
            key=lambda record: (int(record["section_number"]), int(record["section_image_id"])),
        )
        for experiment_id in sorted(groups)
    }


def load_sealed_holdout(root: Path) -> tuple[list[dict], dict[int, dict], dict, dict]:
    records = validate_sealed_boundary(read_jsonl(root / "sections.jsonl"))
    source_section_ids = {int(record["section_image_id"]) for record in records}
    source_experiment_ids = {int(record["experiment_id"]) for record in records}
    if len(records) != EXPECTED_SECTIONS or len(source_experiment_ids) != EXPECTED_EXPERIMENTS:
        raise RuntimeError(
            "SEALED INFERENCE REFUSED: acquisition manifest does not contain the complete sealed set"
        )
    datasets = read_jsonl(root / "datasets.jsonl")
    sealed_datasets = [record for record in datasets if record["split"] == SEALED_SPLIT]
    sealed_specimens = {int(record["specimen_id"]) for record in sealed_datasets}
    sealed_experiments = {int(record["experiment_id"]) for record in sealed_datasets}
    if any(
        record["split"] != SEALED_SPLIT
        and (
            int(record["specimen_id"]) in sealed_specimens
            or int(record["experiment_id"]) in sealed_experiments
        )
        for record in datasets
    ):
        raise ValueError("A sealed dataset specimen or experiment crosses the split boundary")
    dataset_by_experiment = {
        int(record["experiment_id"]): record
        for record in sealed_datasets
    }
    if set(dataset_by_experiment) != source_experiment_ids:
        raise ValueError("Sealed dataset and section manifests disagree")
    if any(
        int(record["specimen_id"]) != int(dataset_by_experiment[int(record["experiment_id"])]["specimen_id"])
        for record in records
    ):
        raise ValueError("Sealed dataset and section specimen IDs disagree")
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    expected_ids = set(map(int, provenance["sealed_deepslice_s2p_experiment_ids"]))
    if set(dataset_by_experiment) != expected_ids:
        raise ValueError("The sealed experiment set differs from acquisition provenance")
    quality, approved_section_ids, _ = load_registered_image_quality_manifest(root)
    records = [
        record
        for record in records
        if int(record["section_image_id"]) in approved_section_ids
    ]
    if len(records) != EXPECTED_SECTIONS:
        raise RuntimeError(
            "SEALED INFERENCE REFUSED: quality filtering removed one or more sealed sections"
        )
    quality = {
        **quality,
        "sealed_source_record_count": len(source_section_ids),
        "sealed_approved_record_count": len(records),
        "sealed_rejected_records": [
            record
            for record in quality["rejected_records"]
            if int(record["section_image_id"]) in source_section_ids
        ],
    }
    return records, dataset_by_experiment, provenance, quality


def require_complete_sealed_images(
    root: Path,
    records: list[dict],
    expected_sections: int = EXPECTED_SECTIONS,
    expected_experiments: int = EXPECTED_EXPERIMENTS,
) -> list[Path]:
    paths = [root / record["relative_path"] for record in records]
    present = {
        folder / entry.name
        for folder in {path.parent for path in paths}
        if folder.is_dir()
        for entry in os.scandir(folder)
        if entry.is_file() and entry.stat().st_size > 0
    }
    missing = [path for path in paths if path not in present]
    experiment_count = len({int(record["experiment_id"]) for record in records})
    if len(records) != expected_sections or experiment_count != expected_experiments or missing:
        raise RuntimeError(
            "SEALED INFERENCE REFUSED: expected "
            f"{expected_sections} images from {expected_experiments} experiments; found "
            f"{len(records)} records from {experiment_count} experiments and "
            f"{len(records) - len(missing)} present images."
        )
    return paths


# Duplicated deliberately so final coordinate conversion does not depend on trainer helpers.
def quicknii_to_tracker_pose(ouv: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(np.asarray(ouv, dtype=np.float64))
    if values.shape[1] != 9:
        raise ValueError("QuickNII OUV must have nine values")
    origin = values[:, :3]
    normal = np.cross(values[:, 3:6], values[:, 6:9])
    normal[normal[:, 1] < 0.0] *= -1.0
    if np.any(np.abs(normal[:, 1]) < 1e-9):
        raise ValueError("QuickNII OUV contains a non-coronal plane")
    ap_per_ml = normal[:, 0] / normal[:, 1]
    ap_per_dv = -normal[:, 2] / normal[:, 1]
    origin_ml = origin[:, 0]
    origin_ap = QUICKNII_SHAPE_ML_AP_DV[1] - origin[:, 1]
    origin_dv = QUICKNII_SHAPE_ML_AP_DV[2] - origin[:, 2]
    ap_index = (
        origin_ap
        + ap_per_ml * (ATLAS_CENTER_ML_DV[0] - origin_ml)
        + ap_per_dv * (ATLAS_CENTER_ML_DV[1] - origin_dv)
    )
    pose = np.column_stack(
        (
            (BREGMA_AP_INDEX - ap_index) * VOXEL_UM,
            np.degrees(np.arctan(ap_per_ml)),
            np.degrees(np.arctan(ap_per_dv)),
        )
    )
    return pose[0] if np.asarray(ouv).ndim == 1 else pose


def quicknii_pixel_points(ouv: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    grid_x, grid_y = np.meshgrid(x, y)
    values = np.asarray(ouv, dtype=np.float64)
    return values[:3] + grid_x[..., None] * values[3:6] + grid_y[..., None] * values[6:9]


def brain_masked_plane_distance(
    ground_truth_ouv: np.ndarray,
    predicted_ouv: np.ndarray,
    brain_mask: np.ndarray,
) -> float:
    mask = np.asarray(brain_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("The plane-distance metric needs a non-empty 2-D brain mask")
    ground_truth = quicknii_pixel_points(ground_truth_ouv, mask.shape)
    predicted = quicknii_pixel_points(predicted_ouv, mask.shape)
    return float(np.linalg.norm(predicted[mask] - ground_truth[mask], axis=1).mean())


def annotation_brain_mask(
    ground_truth_ouv: np.ndarray,
    annotation_ap_dv_ml: np.ndarray,
    shape: tuple[int, int] = (299, 299),
) -> np.ndarray:
    quicknii = quicknii_pixel_points(ground_truth_ouv, shape)
    atlas = np.stack(
        (
            QUICKNII_SHAPE_ML_AP_DV[1] - quicknii[..., 1],
            QUICKNII_SHAPE_ML_AP_DV[2] - quicknii[..., 2],
            quicknii[..., 0],
        ),
        axis=-1,
    )
    indices = np.rint(atlas).astype(np.int64)
    valid = np.all(indices >= 0, axis=-1) & np.all(indices < np.asarray(annotation_ap_dv_ml.shape), axis=-1)
    mask = np.zeros(shape, dtype=bool)
    inside = indices[valid]
    mask[valid] = annotation_ap_dv_ml[inside[:, 0], inside[:, 1], inside[:, 2]] > 0
    return mask


def ap_band(ap_um: float) -> str:
    for name, lower, upper in AP_BANDS:
        if lower < ap_um <= upper or (lower == -np.inf and ap_um <= upper):
            return name
    raise AssertionError(f"No AP band for {ap_um}")


def _calibration(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    variance = float(np.var(prediction))
    slope = float(np.cov(prediction, target, ddof=0)[0, 1] / variance) if variance > 0.0 else None
    intercept = float(target.mean() - slope * prediction.mean()) if slope is not None else None
    correlation = float(np.corrcoef(prediction, target)[0, 1]) if len(target) > 1 and variance > 0.0 and np.var(target) > 0.0 else None
    return {
        "definition": "ordinary least squares: observed = intercept + slope * predicted",
        "slope": slope,
        "intercept": intercept,
        "pearson_r": correlation,
        "r_squared": None if correlation is None else correlation**2,
        "observed_mean": float(target.mean()),
        "predicted_mean": float(prediction.mean()),
    }


def _error_summary(target: np.ndarray, prediction: np.ndarray, tolerances: tuple[float, ...]) -> dict:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "bias": float(error.mean()),
        "p95_absolute_error": float(np.percentile(absolute, 95.0)),
        "median_absolute_error": float(np.median(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "calibration": _calibration(target, prediction),
        "coverage": {f"within_{value:g}": float(np.mean(absolute <= value)) for value in tolerances},
    }


def summarize_predictions(rows: pd.DataFrame) -> dict:
    result = {"count": int(len(rows)), "tracker_pose": {}}
    for axis in POSE_AXES:
        result["tracker_pose"][axis] = _error_summary(
            rows[f"gt_{axis}"].to_numpy(),
            rows[f"pred_{axis}"].to_numpy(),
            POSE_TOLERANCES[axis],
        )
    with_ouv = rows[rows["plane_distance_voxels"].notna()]
    if len(with_ouv):
        component_metrics = {}
        for column in OUV_COLUMNS:
            component_metrics[column] = _error_summary(
                with_ouv[f"gt_{column}"].to_numpy(),
                with_ouv[f"pred_{column}"].to_numpy(),
                (),
            )
        distance = with_ouv["plane_distance_voxels"].to_numpy()
        result["quicknii_ouv"] = {
            "count": int(len(with_ouv)),
            "component_metrics_voxels": component_metrics,
            "mean_9_component_mae_voxels": float(
                np.mean([component_metrics[column]["mae"] for column in OUV_COLUMNS])
            ),
            "brain_masked_plane_distance": {
                "mean_voxels": float(distance.mean()),
                "median_voxels": float(np.median(distance)),
                "p95_voxels": float(np.percentile(distance, 95.0)),
                "mean_um": float(distance.mean() * VOXEL_UM),
                "median_um": float(np.median(distance) * VOXEL_UM),
                "p95_um": float(np.percentile(distance, 95.0) * VOXEL_UM),
            },
        }
    else:
        result["quicknii_ouv"] = {
            "count": 0,
            "unavailable_reason": "This predictor exposes tracker pose, not nine QuickNII OUV values.",
        }
    return result


def report_scopes(rows: pd.DataFrame) -> dict:
    return {
        "aggregate": summarize_predictions(rows),
        "per_experiment": {
            str(experiment_id): summarize_predictions(group)
            for experiment_id, group in rows.groupby("experiment_id", sort=True)
        },
        "per_ap_band": {
            band: summarize_predictions(group)
            for band, group in rows.groupby("ap_band", sort=False)
        },
        "per_product": {
            str(product): summarize_predictions(group)
            for product, group in rows.groupby("product", sort=True)
        },
    }


def _ordered_table(records: list[dict], predictions: list[dict]) -> pd.DataFrame:
    predicted = {Path(str(row["Filenames"])).name.casefold(): row for row in predictions}
    names = [Path(record["relative_path"]).name.casefold() for record in records]
    if len(predicted) != len(records) or set(predicted) != set(names):
        raise ValueError("DeepSlice predictions do not match the sealed experiment images")
    table = pd.DataFrame([predicted[name] for name in names])
    table["nr"] = [int(record["section_number"]) for record in records]
    return table


def _propagate_angles(table: pd.DataFrame) -> pd.DataFrame:
    from DeepSlice.coord_post_processing import angle_methods

    result = table.copy()
    for _ in range(2):
        result = angle_methods.propagate_angles(result, "weighted_mean", "mouse")
    return result


def run_deepslice_modes(records: list[dict], paths: list[Path]) -> tuple[dict[str, np.ndarray], dict]:
    """Run published AI, MEns-AI, and MEns-AI-CI states for one experiment."""
    messages = queue.SimpleQueue()
    ensemble_records, version, hashes, _, runtime = run_deepslice_inference(
        list(map(str, paths)), messages, threading.Event()
    )
    ensemble = _ordered_table(records, ensemble_records)

    inputs, widths, heights = preprocess_deepslice_images(list(map(str, paths)))
    force_cpu = runtime["backend"] == "ONNX Runtime CPU"
    sessions, _, _, _ = load_deepslice_onnx_sessions(force_cpu)
    primary_values = sessions["primary"].run(["Identity:0"], {"images": inputs})[0]
    primary_records = [
        {
            "Filenames": path.name,
            **dict(zip(OUV_COLUMNS, map(float, values))),
            "width": int(width),
            "height": int(height),
        }
        for path, values, width, height in zip(paths, primary_values, widths, heights)
    ]
    primary = _ordered_table(records, primary_records)

    ai = _propagate_angles(primary)
    mens_ai = _propagate_angles(ensemble)
    from DeepSlice.coord_post_processing import spacing_and_indexing

    mens_ai_ci = spacing_and_indexing.enforce_section_ordering(mens_ai.copy())
    mens_ai_ci = spacing_and_indexing.space_according_to_index(
        mens_ai_ci,
        section_thickness=None,
        voxel_size=VOXEL_UM,
        suppress=True,
        species="mouse",
    )
    return (
        {
            "deepslice_ai": ai.loc[:, OUV_COLUMNS].to_numpy(np.float64),
            "deepslice_mens_ai": mens_ai.loc[:, OUV_COLUMNS].to_numpy(np.float64),
            "deepslice_mens_ai_ci": mens_ai_ci.loc[:, OUV_COLUMNS].to_numpy(np.float64),
        },
        {
            "version": version,
            "model_sha256": hashes,
            "runtime": runtime,
            "mode_definitions": {
                "deepslice_ai": "validated primary model followed by two-pass weighted angle integration",
                "deepslice_mens_ai": "validated primary/secondary OUV mean followed by two-pass weighted angle integration",
                "deepslice_mens_ai_ci": "MEns-AI followed by official cutting-index order and inferred-spacing adjustment",
            },
        },
    )


def run_atlas_pose(records: list[dict], paths: list[Path], model_path: Path) -> tuple[np.ndarray, dict]:
    import cv2

    predictions = []
    runtimes = []
    for start in range(0, len(paths), ATLAS_POSE_IMAGE_BATCH):
        batch_paths = paths[start : start + ATLAS_POSE_IMAGE_BATCH]
        images = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in batch_paths]
        if any(image is None for image in images):
            raise ValueError("AtlasPose could not read a sealed image")
        masks = [automatic_brain_mask(image) for image in images]
        prediction, runtime = run_atlas_pose_candidate_onnx(images, masks, model_path)
        predictions.append(prediction)
        runtimes.append(runtime)
    prediction = np.concatenate(predictions)
    if prediction.shape != (len(records), 3) or not np.isfinite(prediction).all():
        raise ValueError("AtlasPose returned the wrong number of predictions")
    immutable_fields = (
        "model_sha256",
        "metadata_sha256",
        "architecture",
        "preprocessing_version",
        "preprocessing_contract_sha256",
    )
    if any(
        runtime.get(field) != runtimes[0].get(field)
        for runtime in runtimes[1:]
        for field in immutable_fields
    ):
        raise RuntimeError("AtlasPose candidate or preprocessing changed during sealed inference")
    return prediction, {
        "model_sha256": runtimes[0]["model_sha256"],
        "metadata_sha256": runtimes[0]["metadata_sha256"],
        "device": runtimes[0]["device"],
        "onnxruntime_version": runtimes[0]["onnxruntime_version"],
        "architecture": runtimes[0]["architecture"],
        "preprocessing_version": runtimes[0]["preprocessing_version"],
        "batch_size": ATLAS_POSE_IMAGE_BATCH,
        "batch_count": len(runtimes),
        "inference_seconds": float(sum(runtime["inference_seconds"] for runtime in runtimes)),
        "gpu_fallback_reasons": list(
            dict.fromkeys(
                runtime["gpu_fallback_reason"]
                for runtime in runtimes
                if runtime["gpu_fallback_reason"]
            )
        ),
    }


def prediction_rows(
    records: list[dict],
    method: str,
    predicted_pose: np.ndarray,
    predicted_ouv: np.ndarray | None,
    annotation: np.ndarray,
) -> list[dict]:
    ground_truth_ouv = np.asarray([record["quicknii_ouv"] for record in records], dtype=np.float64)
    ground_truth_pose = quicknii_to_tracker_pose(ground_truth_ouv)
    predicted_pose = np.asarray(predicted_pose, dtype=np.float64)
    if predicted_pose.shape != ground_truth_pose.shape or not np.isfinite(predicted_pose).all():
        raise ValueError(f"{method} produced incomplete or non-finite pose predictions")
    if predicted_ouv is not None:
        predicted_ouv = np.asarray(predicted_ouv, dtype=np.float64)
        if predicted_ouv.shape != ground_truth_ouv.shape or not np.isfinite(predicted_ouv).all():
            raise ValueError(f"{method} produced incomplete or non-finite QuickNII predictions")
    output = []
    for index, record in enumerate(records):
        in_training_domain = -4500.0 <= float(ground_truth_pose[index, 0]) <= 500.0
        if bool(record["in_training_ap_domain"]) != in_training_domain:
            raise ValueError("Recorded AP-domain label disagrees with exact QuickNII ground truth")
        row = {
            "sealed": True,
            "split": SEALED_SPLIT,
            "method": method,
            "experiment_id": int(record["experiment_id"]),
            "specimen_id": int(record["specimen_id"]),
            "section_image_id": int(record["section_image_id"]),
            "section_number": int(record["section_number"]),
            "relative_path": record["relative_path"],
            "product": record["product"],
            "ap_band": ap_band(float(ground_truth_pose[index, 0])),
            "in_training_ap_domain": in_training_domain,
        }
        for column, name in enumerate(POSE_AXES):
            row[f"gt_{name}"] = float(ground_truth_pose[index, column])
            row[f"pred_{name}"] = float(predicted_pose[index, column])
            row[f"error_{name}"] = float(predicted_pose[index, column] - ground_truth_pose[index, column])
            row[f"absolute_error_{name}"] = abs(row[f"error_{name}"])
        for column, name in enumerate(OUV_COLUMNS):
            row[f"gt_{name}"] = float(ground_truth_ouv[index, column])
            row[f"pred_{name}"] = None if predicted_ouv is None else float(predicted_ouv[index, column])
        if predicted_ouv is None:
            row["plane_distance_voxels"] = None
            row["plane_distance_um"] = None
        else:
            mask = annotation_brain_mask(ground_truth_ouv[index], annotation)
            if mask.any():
                distance = brain_masked_plane_distance(
                    ground_truth_ouv[index], predicted_ouv[index], mask
                )
                row["plane_distance_voxels"] = distance
                row["plane_distance_um"] = distance * VOXEL_UM
            else:
                row["plane_distance_voxels"] = None
                row["plane_distance_um"] = None
        output.append(row)
    return output


def _canonical_json_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sealed_release_report(
    primary_table: pd.DataFrame,
    comparisons: list[dict],
    joint_superiority: dict,
    model_sha256: str,
    metadata_sha256: str,
    preprocessing_contract_sha256: str,
    training_source_sha256: dict,
    training_data_sha256: dict,
    sealed_data_sha256: dict,
    sealed_metrics_sha256: str,
    sealed_predictions_sha256: str,
    evaluator_sha256: str,
    evaluator_environment_sha256: str,
    presealed_commitment_sha256: str,
    sealed_claim_sha256: str,
    consumption_receipt_sha256: str,
    created_utc: str,
    recovery: dict | None = None,
) -> dict:
    atlas_rows = primary_table[primary_table["method"] == "atlas_pose"]
    quality = release_quality_gate(atlas_rows)
    comparison_by_metric = {
        comparison["metric"]: comparison
        for comparison in comparisons
        if comparison["candidate"] == "atlas_pose"
        and comparison["reference"] == RELEASE_REFERENCE
    }
    required_metrics = tuple(f"absolute_error_{axis}" for axis in POSE_AXES)
    if set(required_metrics) - comparison_by_metric.keys():
        raise ValueError("Sealed release report is missing AtlasPose/DeepSlice paired comparisons")
    component_passed = {
        axis: bool(
            comparison_by_metric[f"absolute_error_{axis}"]["delta_candidate_minus_reference"] < 0.0
            and comparison_by_metric[f"absolute_error_{axis}"]["probability_candidate_lower_error"]
            >= RELEASE_CONFIDENCE
        )
        for axis in POSE_AXES
    }
    simultaneous_passed = bool(
        joint_superiority.get("candidate") == "atlas_pose"
        and joint_superiority.get("reference") == RELEASE_REFERENCE
        and joint_superiority.get("simultaneous_superiority_passed") is True
    )
    release_approved = bool(
        quality["all_gates_passed"]
        and all(component_passed.values())
        and simultaneous_passed
    )
    payload = {
        "release_report_version": 4 if recovery else 3,
        "sealed": True,
        "benchmark_role": "final_release_gate",
        "created_utc": created_utc,
        "model_sha256": model_sha256,
        "metadata_sha256": metadata_sha256,
        "preprocessing_contract_sha256": preprocessing_contract_sha256,
        "training_source_sha256": training_source_sha256,
        "training_data_sha256": training_data_sha256,
        "sealed_data_sha256": sealed_data_sha256,
        "sealed_metrics_sha256": sealed_metrics_sha256,
        "sealed_predictions_sha256": sealed_predictions_sha256,
        "evaluator_sha256": evaluator_sha256,
        "evaluator_environment_sha256": evaluator_environment_sha256,
        "presealed_commitment_sha256": presealed_commitment_sha256,
        "sealed_claim_sha256": sealed_claim_sha256,
        "consumption_receipt_sha256": consumption_receipt_sha256,
        "quality_gate": quality,
        "deepslice_reference": RELEASE_REFERENCE,
        "deepslice_confidence_threshold": RELEASE_CONFIDENCE,
        "deepslice_component_passed": component_passed,
        "deepslice_simultaneous_superiority": joint_superiority,
        "deepslice_comparisons": {
            axis: comparison_by_metric[f"absolute_error_{axis}"] for axis in POSE_AXES
        },
        "release_approved": release_approved,
        "promotion_ready": release_approved,
    }
    if recovery:
        payload.update(
            {
                "sealed_recovery_commitment_sha256": recovery["commitment_sha256"],
                "failed_attempt_receipt_sha256": recovery["failed_receipt_sha256"],
            }
        )
    payload["release_integrity_sha256"] = _canonical_json_sha256(payload)
    return payload


# Consumes the sealed cohort once; outputs cannot select or tune a candidate.
def run_evaluation(
    acquisition_root: Path = ACQUISITION_ROOT,
    atlas_pose_model: Path | None = Path(ATLAS_POSE_MODEL) if ATLAS_POSE_MODEL else None,
) -> Path:
    if atlas_pose_model is None:
        raise ValueError("SEALED EVALUATION REFUSED: a frozen AtlasPose candidate is mandatory")
    acquisition_root = Path(acquisition_root)
    recovery_mode = os.environ.get("ATLAS_POSE_SEALED_RECOVERY")
    if recovery_mode:
        if recovery_mode != SEALED_RECOVERY_MODE:
            raise ValueError(f"SEALED RECOVERY REFUSED: unknown recovery mode {recovery_mode!r}")
        frozen = load_frozen_candidate_for_recovery(Path(atlas_pose_model))
        _, claim_path, receipt_path, recovery = recover_failed_sealed_benchmark(
            frozen, acquisition_root
        )
    else:
        frozen = freeze_candidate(Path(atlas_pose_model))
        _, claim_path, receipt_path = claim_sealed_benchmark(frozen)
        recovery = None
    claim_sha256 = sha256(claim_path)
    active_environment = (
        recovery["commitment"]["recovery_evaluator_environment"]
        if recovery
        else frozen["presealed"]["evaluator_environment"]
    )
    try:
        source_hashes = verify_source_commitment(acquisition_root, ANNOTATION_PATH, frozen)
        image_tree_sha256 = verify_complete_sealed_image_hashes(acquisition_root)
        records, datasets, acquisition_provenance, image_quality = load_sealed_holdout(acquisition_root)
        records = [
            {
                **record,
                "product": "+".join(
                    map(str, datasets[int(record["experiment_id"])].get("product_ids", []))
                ) or "unknown",
            }
            for record in records
        ]
        paths = require_complete_sealed_images(
            acquisition_root,
            records,
            expected_sections=EXPECTED_SECTIONS,
        )

        import nrrd

        annotation = nrrd.read(str(ANNOTATION_PATH))[0]
        all_rows = []
        deepslice_runtime = {}
        atlas_pose_runtime = {}
        path_by_section = {
            int(record["section_image_id"]): path for record, path in zip(records, paths)
        }
        for experiment_id, experiment_records in ordered_experiment_groups(records).items():
            experiment_paths = [
                path_by_section[int(record["section_image_id"])] for record in experiment_records
            ]
            modes, runtime = run_deepslice_modes(experiment_records, experiment_paths)
            deepslice_runtime[str(experiment_id)] = runtime
            for method, ouv in modes.items():
                pose = quicknii_to_tracker_pose(ouv)
                all_rows.extend(prediction_rows(experiment_records, method, pose, ouv, annotation))
            pose, runtime = run_atlas_pose(
                experiment_records, experiment_paths, frozen["model_path"]
            )
            atlas_pose_runtime[str(experiment_id)] = runtime
            all_rows.extend(
                prediction_rows(experiment_records, "atlas_pose", pose, None, annotation)
            )

        methods = (
            "deepslice_ai",
            "deepslice_mens_ai",
            "deepslice_mens_ai_ci",
            "atlas_pose",
        )
        table = pd.DataFrame(all_rows)
        validate_complete_method_cohort(table, records, methods)
        primary_table, out_of_domain_table = evaluation_domains(table)
        metrics = {
            "primary_in_training_ap_domain": {
                method: report_scopes(primary_table[primary_table["method"] == method])
                for method in methods
            },
            "out_of_domain": {
                method: report_scopes(out_of_domain_table[out_of_domain_table["method"] == method])
                for method in methods
            } if not out_of_domain_table.empty else None,
        }
        comparison_metrics = (
            "absolute_error_ap_um",
            "absolute_error_lr_deg",
            "absolute_error_dv_deg",
        )
        comparisons = []
        for candidate, reference in (
            ("deepslice_mens_ai", "deepslice_ai"),
            ("deepslice_mens_ai_ci", "deepslice_mens_ai"),
            ("atlas_pose", RELEASE_REFERENCE),
        ):
            for metric in comparison_metrics:
                comparisons.append(
                    paired_animal_bootstrap(primary_table, candidate, reference, metric)
                )
            if candidate != "atlas_pose":
                plane_rows = primary_table[primary_table["plane_distance_um"].notna()]
                comparisons.append(
                    paired_animal_bootstrap(
                        plane_rows, candidate, reference, "plane_distance_um"
                    )
                )
        joint_superiority = paired_animal_joint_superiority(
            primary_table,
            "atlas_pose",
            RELEASE_REFERENCE,
            comparison_metrics,
        )

        if (
            sha256(frozen["model_path"]) != frozen["model_sha256"]
            or sha256(frozen["metadata_path"]) != frozen["metadata_sha256"]
            or verify_source_commitment(acquisition_root, ANNOTATION_PATH, frozen) != source_hashes
            or verify_complete_sealed_image_hashes(acquisition_root) != image_tree_sha256
            or evaluator_environment_commitment() != active_environment
        ):
            raise RuntimeError("SEALED EVALUATION REFUSED: candidate or evaluation source changed")

        run_id = frozen["model_sha256"][:12]
        output = acquisition_root / "SEALED_FINAL_EVALUATION" / run_id
        output.mkdir(parents=True, exist_ok=False)
        predictions_path = output / "SEALED_predictions.csv"
        table.to_csv(predictions_path, index=False)
        created_utc = datetime.now(timezone.utc).isoformat()
        report = {
            "sealed": True,
            "benchmark_role": "final_test_only",
            "benchmark_id": SEALED_BENCHMARK_ID,
            "prohibited_uses": [
                "training",
                "validation",
                "model_selection",
                "hyperparameter_tuning",
                "early_stopping",
                "augmentation_selection",
            ],
            "created_utc": created_utc,
            "section_count": len(records),
            "in_training_ap_domain_section_count": int(
                primary_table["section_image_id"].nunique()
            ),
            "out_of_domain_section_count": int(
                out_of_domain_table["section_image_id"].nunique()
            ),
            "experiment_count": len(ordered_experiment_groups(records)),
            "coordinate_ground_truth": "exact Allen section alignment2d/alignment3d converted to recorded QuickNII OUV",
            "plane_distance_definition": {
                "unit": "25 um Allen CCF voxels",
                "grid": "299 x 299 pixel centers",
                "mask": "ground-truth plane samples whose nearest Allen annotation voxel is inside brain",
                "statistic": "mean Euclidean distance between corresponding predicted and ground-truth CCF points",
            },
            "source": {
                "acquisition_root": str(acquisition_root.resolve()),
                **source_hashes,
                "sealed_image_tree_sha256": image_tree_sha256,
                "registered_image_quality": image_quality,
                "acquisition_provenance": acquisition_provenance,
            },
            "predictors": {
                "deepslice": deepslice_runtime,
                "atlas_pose": {
                    "enabled": True,
                    "model_path": str(frozen["model_path"]),
                    "model_sha256": frozen["model_sha256"],
                    "metadata_sha256": frozen["metadata_sha256"],
                    "runtime": atlas_pose_runtime,
                },
            },
            "metrics": metrics,
            "animal_level_paired_bootstrap": comparisons,
            "animal_level_joint_superiority": joint_superiority,
            "selection_statement": "This globally consumed result cannot be reused by a trainer or model-selection entry point.",
            "presealed_commitment_sha256": frozen["presealed_sha256"],
            "sealed_claim_sha256": claim_sha256,
            "evaluator_sha256": sha256(Path(__file__)),
            "evaluator_environment_sha256": active_environment["commitment_sha256"],
        }
        metrics_path = output / "SEALED_metrics.json"
        metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        receipt = {
            "contract_version": 1,
            "benchmark_id": SEALED_BENCHMARK_ID,
            "claim_sha256": claim_sha256,
            "model_sha256": frozen["model_sha256"],
            "presealed_commitment_sha256": frozen["presealed_sha256"],
            "sealed_predictions_sha256": sha256(predictions_path),
            "sealed_metrics_sha256": sha256(metrics_path),
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if recovery:
            receipt.update(
                {
                    "sealed_recovery_commitment_sha256": recovery["commitment_sha256"],
                    "failed_attempt_receipt_sha256": recovery["failed_receipt_sha256"],
                }
            )
        _atomic_json(receipt_path, receipt)
        artifact_paths = {
            "PRESEALED_COMMITMENT.json": frozen["presealed_path"],
            "SEALED_CLAIM.json": claim_path,
            "SEALED_CONSUMPTION_RECEIPT.json": receipt_path,
        }
        if recovery:
            artifact_paths.update(
                {
                    "SEALED_RECOVERY_COMMITMENT.json": recovery["commitment_path"],
                    "FAILED_ATTEMPT_CLAIM.json": recovery["failed_claim_path"],
                    "FAILED_ATTEMPT_RECEIPT.json": recovery["failed_receipt_path"],
                }
            )
        for name, source_path in artifact_paths.items():
            _immutable_copy(source_path, output / name, sha256(source_path))

        training_data_sha256 = frozen["presealed"]["training_data_sha256"]
        sealed_data_sha256 = {
            **source_hashes,
            "sealed_image_tree_sha256": image_tree_sha256,
        }
        release = sealed_release_report(
            primary_table,
            comparisons,
            joint_superiority,
            frozen["model_sha256"],
            frozen["metadata_sha256"],
            frozen["metadata"].get("preprocessing_contract_sha256"),
            frozen["metadata"].get("source_sha256"),
            training_data_sha256,
            sealed_data_sha256,
            sha256(metrics_path),
            sha256(predictions_path),
            sha256(Path(__file__)),
            active_environment["commitment_sha256"],
            frozen["presealed_sha256"],
            claim_sha256,
            sha256(receipt_path),
            created_utc,
            recovery,
        )
        (output / "DO_NOT_USE_FOR_MODEL_SELECTION.txt").write_text(
            "SEALED FINAL TEST OUTPUT. Do not use these results for training, tuning, early stopping, or model selection.\n",
            encoding="utf-8",
        )
        (output / "RELEASE_REPORT.json").write_text(
            json.dumps(release, indent=2), encoding="utf-8"
        )
        return output
    except BaseException as error:
        failed = {
                "contract_version": 1,
                "benchmark_id": SEALED_BENCHMARK_ID,
                "claim_sha256": claim_sha256,
                "model_sha256": frozen["model_sha256"],
                "presealed_commitment_sha256": frozen["presealed_sha256"],
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure": f"{type(error).__name__}: {error}",
            }
        if recovery:
            failed.update(
                {
                    "sealed_recovery_commitment_sha256": recovery["commitment_sha256"],
                    "failed_attempt_receipt_sha256": recovery["failed_receipt_sha256"],
                }
            )
        _atomic_json(receipt_path, failed)
        raise


if __name__ == "__main__":
    print(run_evaluation())
