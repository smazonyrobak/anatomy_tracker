"""Final-only evaluation on the sealed Allen S2P experiments.

This module is intentionally independent of every training and model-selection
entry point.  Its outputs are test reports, never training metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from source.registered_image_quality import (
    REGISTERED_IMAGE_QUALITY_MANIFEST,
    load_registered_image_quality_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
ACQUISITION_ROOT = Path(
    os.environ.get(
        "ALLEN_REGISTERED_ROOT",
        "J:/AtlasPoseTraining_v7/allen_s2p_pilot_d100_stratified",
    )
)
ANNOTATION_PATH = Path(
    os.environ.get(
        "ALLEN_ANNOTATION_25_PATH",
        ROOT / "data/Allen Brain Atlas 25um/annotation_25.nrrd",
    )
)
ATLAS_POSE_MODEL = os.environ.get("ATLAS_POSE_SEALED_MODEL")
SEALED_SPLIT = "sealed_deepslice_s2p"
EXPECTED_SECTIONS = 1400
EXPECTED_EXPERIMENTS = 10
ATLAS_POSE_IMAGE_BATCH = 16
VOXEL_UM = 25.0
QUICKNII_SHAPE_ML_AP_DV = np.asarray((456.0, 528.0, 320.0))
ATLAS_CENTER_ML_DV = np.asarray((227.5, 159.5))
BREGMA_AP_INDEX = 216.0
OUV_COLUMNS = ("ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz")
POSE_AXES = ("ap_um", "lr_deg", "dv_deg")
POSE_TOLERANCES = {
    "ap_um": (50.0, 100.0, 250.0),
    "lr_deg": (1.0, 2.0, 5.0),
    "dv_deg": (1.0, 2.0, 5.0),
}
RELEASE_GATE_THRESHOLDS = {
    "mean_ap_um": 60.0,
    "mean_lr_deg": 0.90,
    "mean_dv_deg": 1.75,
    "absolute_ap_bias_um": 25.0,
    "ap_p95_um": 150.0,
    "worst_ap_band_mae_um": 90.0,
    "worst_product_mae_um": 90.0,
}
RELEASE_REFERENCE = "deepslice_mens_ai_ci"
RELEASE_CONFIDENCE = 0.95
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


def quicknii_to_tracker_pose(ouv: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(np.asarray(ouv, dtype=np.float64))
    if values.shape[1] != 9:
        raise ValueError("QuickNII OUV must have nine values")
    origin = values[:, :3]
    normal = np.cross(values[:, 3:6], values[:, 6:9])
    normal[normal[:, 1] < 0.0] *= -1.0
    if np.any(np.abs(normal[:, 1]) < 1e-9):
        raise ValueError("QuickNII OUV contains a non-coronal plane")
    ap_per_ml = -normal[:, 0] / normal[:, 1]
    ap_per_dv = -normal[:, 2] / normal[:, 1]
    origin_ml = QUICKNII_SHAPE_ML_AP_DV[0] - origin[:, 0]
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
            QUICKNII_SHAPE_ML_AP_DV[0] - quicknii[..., 0],
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


def paired_animal_bootstrap(
    rows: pd.DataFrame,
    candidate: str,
    reference: str,
    metric: str,
    iterations: int = 10_000,
    seed: int = 94731,
) -> dict:
    selected = rows[rows["method"].isin((candidate, reference))]
    pivot = selected.pivot(index=["specimen_id", "section_image_id"], columns="method", values=metric).dropna()
    if candidate not in pivot or reference not in pivot:
        raise ValueError(f"No paired {metric} values for {candidate} and {reference}")
    animal_delta = (pivot[candidate] - pivot[reference]).groupby(level="specimen_id").mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(animal_delta, (iterations, len(animal_delta)), replace=True).mean(axis=1)
    return {
        "unit": "specimen_id",
        "animal_count": int(len(animal_delta)),
        "paired_section_count": int(len(pivot)),
        "metric": metric,
        "candidate": candidate,
        "reference": reference,
        "delta_candidate_minus_reference": float(animal_delta.mean()),
        "bootstrap_95_ci": np.percentile(draws, (2.5, 97.5)).tolist(),
        "probability_candidate_lower_error": float(np.mean(draws < 0.0)),
        "iterations": int(iterations),
        "seed": int(seed),
    }


def _tracker_module():
    source = str(ROOT / "source")
    if source not in sys.path:
        sys.path.insert(0, source)
    import proprietary_trajectory_tool

    return proprietary_trajectory_tool


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
    tracker = _tracker_module()
    messages = queue.SimpleQueue()
    ensemble_records, version, hashes, _, runtime = tracker.run_deepslice_inference(
        list(map(str, paths)), messages, threading.Event()
    )
    ensemble = _ordered_table(records, ensemble_records)

    inputs, widths, heights = tracker.preprocess_deepslice_images(list(map(str, paths)))
    force_cpu = runtime["backend"] == "ONNX Runtime CPU"
    sessions, _, _, _ = tracker.load_deepslice_onnx_sessions(force_cpu)
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

    source = str(ROOT / "source")
    if source not in sys.path:
        sys.path.insert(0, source)
    from atlas_pose_runtime import automatic_brain_mask, run_atlas_pose_candidate_onnx

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
    if prediction.shape != (len(records), 3):
        raise ValueError("AtlasPose returned the wrong number of predictions")
    return prediction, {
        "model_sha256": runtimes[0]["model_sha256"],
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
            distance = brain_masked_plane_distance(ground_truth_ouv[index], predicted_ouv[index], mask)
            row["plane_distance_voxels"] = distance
            row["plane_distance_um"] = distance * VOXEL_UM
        output.append(row)
    return output


def evaluation_domains(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    in_domain = table[table["in_training_ap_domain"].astype(bool)]
    if in_domain.empty:
        raise ValueError("The sealed holdout contains no sections inside the trained AP domain")
    return in_domain, table[~table["in_training_ap_domain"].astype(bool)]


def _canonical_json_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _ap_500um_band(ap_um: float) -> str:
    index = 9 if ap_um == 500.0 else int(np.floor((ap_um + 4500.0) / 500.0))
    low = -4500 + 500 * index
    return f"{low}:{low + 500}"


def release_quality_gate(rows: pd.DataFrame) -> dict:
    frame = rows[rows["in_training_ap_domain"].astype(bool)].copy()
    if frame.empty or set(frame["method"]) != {"atlas_pose"}:
        raise ValueError("Release quality gate requires in-domain AtlasPose rows only")
    frame["ap_500um_band"] = frame["gt_ap_um"].map(_ap_500um_band)

    component_mae = {}
    for axis in POSE_AXES:
        frame[f"absolute_{axis}"] = (frame[f"pred_{axis}"] - frame[f"gt_{axis}"]).abs()
        per_bin = frame.groupby(["specimen_id", "ap_500um_band"])[f"absolute_{axis}"].mean()
        per_specimen = per_bin.groupby(level="specimen_id").mean()
        component_mae[axis] = float(per_specimen.mean())

    frame["signed_ap_error"] = frame["pred_ap_um"] - frame["gt_ap_um"]
    per_bin_bias = frame.groupby(["specimen_id", "ap_500um_band"])["signed_ap_error"].mean()
    absolute_ap_bias = float(abs(per_bin_bias.groupby(level="specimen_id").mean().mean()))
    band_mae = (
        frame.groupby(["ap_500um_band", "specimen_id"])["absolute_ap_um"]
        .mean()
        .groupby(level="ap_500um_band")
        .mean()
    )
    product_mae = (
        frame.groupby(["product", "specimen_id"])["absolute_ap_um"]
        .mean()
        .groupby(level="product")
        .mean()
    )
    values = {
        "mean_ap_um": component_mae["ap_um"],
        "mean_lr_deg": component_mae["lr_deg"],
        "mean_dv_deg": component_mae["dv_deg"],
        "absolute_ap_bias_um": absolute_ap_bias,
        "ap_p95_um": float(np.percentile(frame["absolute_ap_um"], 95.0)),
        "worst_ap_band_mae_um": float(band_mae.max()),
        "worst_product_mae_um": float(product_mae.max()),
    }
    passed = {
        name: bool(values[name] <= threshold)
        for name, threshold in RELEASE_GATE_THRESHOLDS.items()
    }
    return {
        "values": values,
        "thresholds": dict(RELEASE_GATE_THRESHOLDS),
        "passed": passed,
        "all_gates_passed": all(passed.values()),
        "worst_ap_band": str(band_mae.idxmax()),
        "worst_product": str(product_mae.idxmax()),
    }


def sealed_release_report(
    primary_table: pd.DataFrame,
    comparisons: list[dict],
    model_sha256: str,
    metadata_sha256: str,
    preprocessing_contract_sha256: str,
    training_source_sha256: dict,
    training_data_sha256: dict,
    sealed_data_sha256: dict,
    sealed_metrics_sha256: str,
    evaluator_sha256: str,
    created_utc: str,
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
    release_approved = bool(quality["all_gates_passed"] and all(component_passed.values()))
    payload = {
        "release_report_version": 2,
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
        "evaluator_sha256": evaluator_sha256,
        "quality_gate": quality,
        "deepslice_reference": RELEASE_REFERENCE,
        "deepslice_confidence_threshold": RELEASE_CONFIDENCE,
        "deepslice_component_passed": component_passed,
        "deepslice_comparisons": {
            axis: comparison_by_metric[f"absolute_error_{axis}"] for axis in POSE_AXES
        },
        "release_approved": release_approved,
        "promotion_ready": release_approved,
    }
    payload["release_integrity_sha256"] = _canonical_json_sha256(payload)
    return payload


def run_evaluation(
    acquisition_root: Path = ACQUISITION_ROOT,
    atlas_pose_model: Path | None = Path(ATLAS_POSE_MODEL) if ATLAS_POSE_MODEL else None,
) -> Path:
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
        expected_sections=image_quality["sealed_approved_record_count"],
    )

    import nrrd

    annotation = nrrd.read(str(ANNOTATION_PATH))[0]
    all_rows = []
    deepslice_runtime = {}
    atlas_pose_runtime = {}
    path_by_section = {int(record["section_image_id"]): path for record, path in zip(records, paths)}
    for experiment_id, experiment_records in ordered_experiment_groups(records).items():
        experiment_paths = [path_by_section[int(record["section_image_id"])] for record in experiment_records]
        modes, runtime = run_deepslice_modes(experiment_records, experiment_paths)
        deepslice_runtime[str(experiment_id)] = runtime
        for method, ouv in modes.items():
            pose = quicknii_to_tracker_pose(ouv)
            all_rows.extend(prediction_rows(experiment_records, method, pose, ouv, annotation))
        if atlas_pose_model is not None:
            pose, runtime = run_atlas_pose(experiment_records, experiment_paths, atlas_pose_model)
            atlas_pose_runtime[str(experiment_id)] = runtime
            all_rows.extend(prediction_rows(experiment_records, "atlas_pose", pose, None, annotation))

    table = pd.DataFrame(all_rows)
    primary_table, out_of_domain_table = evaluation_domains(table)
    methods = list(dict.fromkeys(table["method"]))
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
    comparisons = []
    pairs = [
        ("deepslice_mens_ai", "deepslice_ai"),
        ("deepslice_mens_ai_ci", "deepslice_mens_ai"),
    ]
    if "atlas_pose" in methods:
        pairs.append(("atlas_pose", "deepslice_mens_ai_ci"))
    for candidate, reference in pairs:
        for metric in ("absolute_error_ap_um", "absolute_error_lr_deg", "absolute_error_dv_deg"):
            comparisons.append(paired_animal_bootstrap(primary_table, candidate, reference, metric))
        if candidate != "atlas_pose":
            comparisons.append(
                paired_animal_bootstrap(primary_table, candidate, reference, "plane_distance_um")
            )

    input_hash = sha256(acquisition_root / "sections.jsonl")
    model_hash = sha256(atlas_pose_model) if atlas_pose_model is not None else "deepslice-only"
    atlas_pose_metadata = None
    atlas_pose_metadata_hash = None
    if atlas_pose_model is not None:
        metadata_path = atlas_pose_model.with_suffix(".json")
        atlas_pose_metadata_hash = sha256(metadata_path)
        atlas_pose_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if atlas_pose_metadata.get("sha256") != model_hash:
            raise RuntimeError("AtlasPose sealed candidate metadata does not bind its model")
    run_id = hashlib.sha256(f"{input_hash}:{model_hash}".encode()).hexdigest()[:12]
    output = acquisition_root / "SEALED_FINAL_EVALUATION" / run_id
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "SEALED_predictions.csv", index=False)
    report = {
        "sealed": True,
        "benchmark_role": "final_test_only",
        "prohibited_uses": [
            "training",
            "validation",
            "model_selection",
            "hyperparameter_tuning",
            "early_stopping",
            "augmentation_selection",
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "section_count": len(records),
        "in_training_ap_domain_section_count": int(
            primary_table["section_image_id"].nunique()
        ),
        "out_of_domain_section_count": int(out_of_domain_table["section_image_id"].nunique()),
        "experiment_count": len(ordered_experiment_groups(records)),
        "coordinate_ground_truth": "exact Allen section alignment2d/alignment3d converted to recorded QuickNII OUV",
        "plane_distance_definition": {
            "unit": "25 um Allen CCF voxels",
            "grid": "299 x 299 pixel centers",
            "mask": "ground-truth plane samples whose nearest Allen annotation voxel is inside brain",
            "statistic": "mean Euclidean distance between corresponding predicted and ground-truth CCF points",
        },
        "source": {
            "acquisition_root": str(acquisition_root),
            "sections_sha256": input_hash,
            "datasets_sha256": sha256(acquisition_root / "datasets.jsonl"),
            "provenance_sha256": sha256(acquisition_root / "provenance.json"),
            "downloads_sha256": sha256(acquisition_root / "downloads.jsonl"),
            "registered_image_quality_manifest_sha256": sha256(
                acquisition_root / REGISTERED_IMAGE_QUALITY_MANIFEST
            ),
            "registered_image_quality": image_quality,
            "acquisition_provenance": acquisition_provenance,
        },
        "predictors": {
            "deepslice": deepslice_runtime,
            "atlas_pose": {
                "enabled": atlas_pose_model is not None,
                "model_path": None if atlas_pose_model is None else str(atlas_pose_model),
                "model_sha256": None if atlas_pose_model is None else model_hash,
                "runtime": atlas_pose_runtime,
            },
        },
        "metrics": metrics,
        "animal_level_paired_bootstrap": comparisons,
        "selection_statement": "No ranking in this report is consumed by any trainer or model-selection code.",
        "evaluator_sha256": sha256(Path(__file__)),
    }
    metrics_path = output / "SEALED_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if atlas_pose_model is not None:
        training_data_sha256 = {
            "synthetic_manifests": atlas_pose_metadata.get("manifest_sha256"),
            "registered_data": atlas_pose_metadata.get("registered_data", {}).get("sha256"),
            "atlas_data": atlas_pose_metadata.get("atlas_data_sha256"),
        }
        sealed_data_sha256 = {
            key: report["source"][key]
            for key in (
                "sections_sha256",
                "datasets_sha256",
                "provenance_sha256",
                "downloads_sha256",
                "registered_image_quality_manifest_sha256",
            )
        }
        release = sealed_release_report(
            primary_table,
            comparisons,
            model_hash,
            atlas_pose_metadata_hash,
            atlas_pose_metadata.get("preprocessing_contract_sha256"),
            atlas_pose_metadata.get("source_sha256"),
            training_data_sha256,
            sealed_data_sha256,
            sha256(metrics_path),
            sha256(Path(__file__)),
            report["created_utc"],
        )
        (output / "RELEASE_REPORT.json").write_text(
            json.dumps(release, indent=2),
            encoding="utf-8",
        )
    (output / "DO_NOT_USE_FOR_MODEL_SELECTION.txt").write_text(
        "SEALED FINAL TEST OUTPUT. Do not use these results for training, tuning, early stopping, or model selection.\n",
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    print(run_evaluation())
