from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import queue
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import cv2
import nrrd
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import tifffile
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.interpolate import Rbf
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

SOURCE_MODULE_DIR = str(Path(__file__).resolve().parent)
if SOURCE_MODULE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_MODULE_DIR)
from atlas_pose_runtime import (
    automatic_brain_mask,
    brain_mask_affine,
    fuse_pose_predictions,
    QUICKNII_COORDINATE_CONTRACT_VERSION,
    run_atlas_pose_evaluated_onnx as run_atlas_pose_onnx,
)
from deepslice_runtime import (
    quicknii_to_tracker_alignment,
    run_deepslice_inference,
)
from diffeomorphic_registration_runtime import (
    DiffeomorphicRegistrationRejected,
    run_classical_diffeomorphic_registration,
    run_diffeomorphic_registration,
    verify_diffeomorphic_attestation_inputs,
    verify_classical_registration_backend,
    verify_diffeomorphic_model_bundle,
)
from nonlinear_registration import NonlinearWarpAttestation, SliceAtlasTransform2D
from probe_constraints import (
    InfeasibleProbeConstraint,
    ProbeInsertionConstraint,
    SlicePlane,
    atlas_points_to_stereotaxic_um,
    direction_from_attack_angle,
    fit_probe_ray,
    fit_observed_probe_ray,
    insertion_plan_plane_feasibility,
    prepare_insertion_surface_entries,
    score_candidate_slice_plane,
    stereotaxic_to_volume as probe_stereotaxic_to_volume,
)


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
RESOURCE_DIR = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else APP_DIR
INSTALL_ROOT = APP_DIR.parent.parent if getattr(sys, "frozen", False) and APP_DIR.parent.name.lower() == "tools" else APP_DIR
DEFAULT_ATLAS_FOLDER = INSTALL_ROOT / "data" / "Allen Brain Atlas 25um"
VOXEL_UM = 25.0
ALLEN_CCF_25_SHAPE_AP_DV_ML = (528, 320, 456)
# IBL bregma estimate in Allen CCF space: ML, AP, DV in um; this NRRD is indexed AP, DV, ML.
DEFAULT_BREGMA_UM_ML_AP_DV = np.array([5739.0, 5400.0, 332.0], dtype=np.float64)
DEFAULT_BREGMA_VOXEL_AP_DV_ML = (
    np.array(
        [
            DEFAULT_BREGMA_UM_ML_AP_DV[1],
            DEFAULT_BREGMA_UM_ML_AP_DV[2],
            DEFAULT_BREGMA_UM_ML_AP_DV[0],
        ]
    )
    / VOXEL_UM
)
STEREOTAXIC_AXIS_SIGN_AP_DV_ML = np.array([-1.0, -1.0, 1.0], dtype=np.float64)
DEEPSLICE_REVIEW_AP_UM = 400.0
DEEPSLICE_REVIEW_TILT_DEG = 5.0
DEEPSLICE_REVIEW_SURFACE_RMS_PX = 8.0
POSE_ENGINE_DEEPSLICE = "DeepSlice"
POSE_ENGINE_OWN_CNN = "AtlasPose (evaluated)"
POSE_ENGINE_WEIGHTED = "Weighted vote (AtlasPose + DeepSlice)"
POSE_ENGINES = (POSE_ENGINE_DEEPSLICE, POSE_ENGINE_OWN_CNN, POSE_ENGINE_WEIGHTED)
DEFAULT_OWN_CNN_WEIGHT = 0.20
OWN_CNN_MODEL_PATH = RESOURCE_DIR / "models" / "AtlasPose" / "atlas_pose.onnx"
NONLINEAR_MODEL_PATH = (
    RESOURCE_DIR / "models" / "DiffeomorphicRegistration" / "diffeomorphic.onnx"
)
MIND_CANONICAL_SIZE = (171, 120)
MIND_AP_PRIOR_WEIGHT = 0.001
MIND_TILT_PRIOR_WEIGHT = 0.0005
MIND_SURFACE_WEIGHT = 0.05
MIND_PROBE_GEOMETRY_WEIGHT = 1.0
SESSION_ARCHIVE_FORMAT = "Proprietary Anatomy Tracker session"
SESSION_ARCHIVE_VERSION = 1
SESSION_STATE_FIELDS = (
    "rotation_deg",
    "flip_horizontal",
    "flip_vertical",
    "curve_points",
    "atlas_plane",
    "atlas_index",
    "atlas_tilt_ml_deg",
    "atlas_tilt_dv_deg",
    "atlas_landmarks",
    "slice_landmarks",
    "brain_outline_points",
    "brain_outline_segment_starts",
    "brain_outline_closed",
    "brain_brush_strokes",
    "auto_alignment_score",
    "auto_alignment_global",
    "auto_alignment_extent",
    "auto_alignment_method",
    "auto_alignment_engine",
    "auto_alignment_scope",
    "auto_alignment_run_id",
    "manual_refined_from_run_id",
    "auto_alignment_diagnostics",
    "alignment_source_sha256",
    "deepslice_raw_ensemble_ouv",
    "deepslice_version",
    "deepslice_model_hashes",
    "deepslice_ensemble_disagreement",
)
SURFACE_EDIT_ACTIONS = {"brain_outline_edit", "brain_outline_delete", "brain_outline_erase", "brain_outline_insert"}
SURFACE_ACTIONS = SURFACE_EDIT_ACTIONS | {"brain_outline", "brain_brush"}
CHANNEL_KEY_COLUMNS = ["probe_name", "probe_channel_number"]
ANATOMY_MAPPING_COLUMNS = [
    "structure_id",
    "structure_name",
    "structure_acronym",
    "ccf_ap_index",
    "ccf_dv_index",
    "ccf_ml_index",
    "atlas_region_id",
    "atlas_region",
    "atlas_acronym",
    "atlas_ap",
    "atlas_dv",
    "atlas_ml",
    "stereotaxic_ap_um",
    "stereotaxic_dv_um",
    "stereotaxic_ml_um",
    "trajectory_distance_um",
    "probe_type",
    "anatomy_source",
    "anatomy_assignment_method",
    "anatomy_mapped_at",
]
PROBE_COLORS = (
    (40, 181, 246),
    (255, 153, 51),
    (171, 108, 255),
    (46, 204, 113),
    (255, 91, 137),
    (240, 209, 70),
)
PROBE_PHYSICAL_LENGTH_UM = {
    "Neuropixels 1.0": 10000.0,
    "Neuropixels 2.0 single-shank": 10000.0,
    "Neuropixels 2.0 four-shank": 10000.0,
}
PROBE_TIP_TO_Y0_CONTACT_UM = {
    # SpikeGLX/chanMap y=0 is the first recording row. These offsets are the
    # physical-tip coordinates in the ProbeInterface Neuropixels catalogue.
    "Neuropixels 1.0": 220.0,
    "Neuropixels 2.0 single-shank": 217.0,
    "Neuropixels 2.0 four-shank": 217.0,
}


def probe_mapping_coordinates(
    brain_entry: np.ndarray,
    deepest_mark: np.ndarray,
    surface_direction: np.ndarray,
    endpoint_mode: str,
    insertion_depth_um: float | None,
    channel_vertical_um: np.ndarray,
    probe_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    direction = np.asarray(surface_direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    entry = np.asarray(brain_entry, dtype=np.float64)
    mark = np.asarray(deepest_mark, dtype=np.float64)
    observed_depth_um = float((entry - mark) @ direction * VOXEL_UM)
    if endpoint_mode == "known_insertion_depth":
        tip_depth_um = float(insertion_depth_um)
        if tip_depth_um + VOXEL_UM < observed_depth_um:
            raise ValueError(
                f"Insertion depth ({tip_depth_um:.0f} um) is shallower than the deepest marked electrode "
                f"point ({observed_depth_um:.0f} um)."
            )
    elif endpoint_mode == "deepest_mark_is_tip":
        tip_depth_um = observed_depth_um
    else:
        raise ValueError("Choose how the physical probe tip should be located.")
    physical_length_um = PROBE_PHYSICAL_LENGTH_UM[probe_type]
    if not 0.0 < tip_depth_um <= physical_length_um:
        raise ValueError(
            f"Surface-to-tip depth must be between 0 and {physical_length_um:.0f} um for {probe_type}."
        )
    tip = entry - direction * (tip_depth_um / VOXEL_UM)
    tip_to_y0_um = PROBE_TIP_TO_Y0_CONTACT_UM[probe_type]
    y0_contact = tip + direction * (tip_to_y0_um / VOXEL_UM)
    channel_distance_from_tip_um = tip_to_y0_um + np.asarray(channel_vertical_um, dtype=np.float64)
    if np.any(channel_distance_from_tip_um < 0.0) or np.any(
        channel_distance_from_tip_um > physical_length_um
    ):
        raise ValueError(f"Channel geometry extends outside the {physical_length_um:.0f} um physical shank.")
    coordinates = tip[None, :] + direction[None, :] * (
        channel_distance_from_tip_um[:, None] / VOXEL_UM
    )
    return tip, y0_contact, coordinates, tip_depth_um, observed_depth_um

pg.setConfigOptions(imageAxisOrder="row-major", background="#0f131a", foreground="#d7e7f5")


def probe_color(probe_name: str) -> tuple[int, int, int]:
    digits = "".join(character for character in probe_name if character.isdigit())
    index = int(digits) if digits else sum(probe_name.encode("utf-8"))
    return PROBE_COLORS[index % len(PROBE_COLORS)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def prepare_pose_inputs(
    image_jobs: list[tuple],
    temporary_folder: str,
    progress_messages: queue.SimpleQueue,
    cancel_event: threading.Event,
) -> tuple[list[str], dict[str, dict], dict[str, dict[str, np.ndarray]]]:
    image_paths = []
    input_crops = {}
    prepared_inputs = {}
    for sequence, (
        source_path,
        rotation_deg,
        flip_horizontal,
        flip_vertical,
        surface_points,
        selection_crop,
        outline_closed,
        selection_mask,
    ) in enumerate(image_jobs):
        if cancel_event.is_set():
            raise InterruptedError
        progress_messages.put((2, f"Preparing model input {sequence + 1} / {len(image_jobs)}..."))
        path = Path(source_path)
        output_path = Path(temporary_folder) / f"slice_{sequence:04d}.png"
        raw = (
            tifffile.imread(str(path))
            if path.suffix.lower() in {".tif", ".tiff"}
            else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        )
        source_gray = as_gray(raw)
        source_height, source_width = source_gray.shape[:2]
        display_raw, display_scale = downsample_for_display(source_gray)
        image, _ = transform_slice_image(
            normalize_u8(display_raw),
            rotation_deg,
            flip_horizontal,
            flip_vertical,
        )
        if selection_mask is not None and selection_mask.shape == image.shape:
            brain_mask = np.asarray(selection_mask, dtype=np.uint8) > 0
        elif outline_closed:
            brain_mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.fillPoly(brain_mask, [np.rint(np.asarray(surface_points)).astype(np.int32)], 1)
            brain_mask = brain_mask.astype(bool)
        else:
            brain_mask = automatic_brain_mask(image)
        if selection_crop is not None:
            crop = selection_crop
        else:
            mask_y, mask_x = np.nonzero(brain_mask)
            crop = surface_crop_bounds(
                [
                    (float(mask_x.min()), float(mask_y.min())),
                    (float(mask_x.max()), float(mask_y.max())),
                ],
                image.shape,
                0.06,
            )
        x0, y0, x1, y1 = crop
        if not cv2.imwrite(str(output_path), image[y0:y1, x0:x1]):
            raise RuntimeError(f"Could not prepare {path.name} for automatic alignment")
        image_paths.append(str(output_path))
        prepared_inputs[output_path.name] = {"image": image, "brain_mask": brain_mask}
        input_crops[output_path.name] = {
            "source_path": str(path),
            "source_image_sha256": file_sha256(path),
            "source_size_bytes": int(path.stat().st_size),
            "source_modified_ns": int(path.stat().st_mtime_ns),
            "rotation_deg": float(rotation_deg),
            "flip_horizontal": bool(flip_horizontal),
            "flip_vertical": bool(flip_vertical),
            "outline_closed": bool(outline_closed),
            "coordinate_frame": "oriented_downsampled_display_pixels",
            "trusted_surface_points_oriented_display_px": [
                [float(point[0]), float(point[1])] for point in surface_points
            ],
            "crop_x0_oriented_display_px": int(x0),
            "crop_y0_oriented_display_px": int(y0),
            "crop_x1_oriented_display_px": int(x1),
            "crop_y1_oriented_display_px": int(y1),
            "oriented_display_width_px": int(image.shape[1]),
            "oriented_display_height_px": int(image.shape[0]),
            "source_raw_width_px": int(source_width),
            "source_raw_height_px": int(source_height),
            "display_downsample_factor": float(display_scale),
            "model_input_png_sha256": file_sha256(output_path),
        }
    return image_paths, input_crops, prepared_inputs


def combine_pose_predictions(
    image_paths: list[str],
    input_crops: dict[str, dict],
    component_records: dict[str, dict[str, dict]],
    deepslice_disagreement: dict[str, dict[str, float]],
    engine: str,
    own_cnn_weight: float,
    bregma_ap_index: float,
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    use_deepslice = engine in (POSE_ENGINE_DEEPSLICE, POSE_ENGINE_WEIGHTED)
    use_own_cnn = engine in (POSE_ENGINE_OWN_CNN, POSE_ENGINE_WEIGHTED)
    records = []
    disagreement = {}
    for path in image_paths:
        filename = Path(path).name
        components = {}
        initial_matrix = None
        if use_deepslice:
            prediction = component_records[POSE_ENGINE_DEEPSLICE][filename]
            atlas_index, tilt_lr, tilt_dv, matrix = quicknii_to_tracker_alignment(
                prediction,
                ALLEN_CCF_25_SHAPE_AP_DV_ML,
            )
            if use_deepslice:
                crop = input_crops[filename]
                full_to_crop = np.asarray(
                    [
                        [1.0, 0.0, -float(crop["crop_x0_oriented_display_px"])],
                        [0.0, 1.0, -float(crop["crop_y0_oriented_display_px"])],
                        [0.0, 0.0, 1.0],
                    ]
                )
                initial_matrix = matrix @ full_to_crop
            deepslice_pose = np.asarray(
                [(bregma_ap_index - atlas_index) * VOXEL_UM, tilt_lr, tilt_dv],
                dtype=np.float64,
            )
            components[POSE_ENGINE_DEEPSLICE] = {
                "pose_ap_um_lr_deg_dv_deg": deepslice_pose.tolist(),
                "raw_ensemble_ouv_quicknii_ml_ap_dv": list(prediction["raw_ensemble_ouv"]),
                "ensemble_disagreement": dict(deepslice_disagreement[filename]),
            }
        if use_own_cnn:
            own_pose = np.asarray(
                component_records[POSE_ENGINE_OWN_CNN][filename]["pose_ap_um_lr_deg_dv_deg"],
                dtype=np.float64,
            )
            components[POSE_ENGINE_OWN_CNN] = {
                "pose_ap_um_lr_deg_dv_deg": own_pose.tolist(),
                "orientation_inverted": bool(
                    component_records[POSE_ENGINE_OWN_CNN][filename]["orientation_inverted"]
                ),
                "orientation_inverted_logit": float(
                    component_records[POSE_ENGINE_OWN_CNN][filename]["orientation_inverted_logit"]
                ),
            }

        if engine == POSE_ENGINE_DEEPSLICE:
            pose = deepslice_pose
            model_difference = dict(deepslice_disagreement[filename])
        elif engine == POSE_ENGINE_OWN_CNN:
            pose = own_pose
            model_difference = {}
        else:
            pose = fuse_pose_predictions(
                np.stack([deepslice_pose, own_pose]),
                np.asarray([1.0 - own_cnn_weight, own_cnn_weight]),
            )
            component_delta = np.abs(deepslice_pose - own_pose)
            internal = deepslice_disagreement[filename]
            model_difference = {
                "ap_um": max(float(component_delta[0]), float(internal["ap_um"])),
                "lr_deg": max(float(component_delta[1]), float(internal["lr_deg"])),
                "dv_deg": max(float(component_delta[2]), float(internal["dv_deg"])),
            }
        disagreement[filename] = model_difference
        records.append(
            {
                "Filenames": filename,
                "pose_ap_um_lr_deg_dv_deg": pose.tolist(),
                "predicted_atlas_index": float(bregma_ap_index - pose[0] / VOXEL_UM),
                "predicted_tilt_lr_deg": float(pose[1]),
                "predicted_tilt_dv_deg": float(pose[2]),
                "initial_slice_to_atlas": None if initial_matrix is None else initial_matrix.tolist(),
                "initial_orientation_inverted": bool(
                    components.get(POSE_ENGINE_OWN_CNN, {}).get("orientation_inverted", False)
                ),
                "model_uncertainty": dict(
                    component_records.get(POSE_ENGINE_OWN_CNN, {}).get(filename, {}).get("model_uncertainty", {})
                ),
                "component_predictions": components,
                "fusion": (
                    {
                        "method": "AP weighted arithmetic; tilts weighted in plane-normal space",
                        "own_cnn_weight": float(own_cnn_weight),
                        "deepslice_weight": float(1.0 - own_cnn_weight),
                    }
                    if engine == POSE_ENGINE_WEIGHTED
                    else None
                ),
            }
        )
    return records, disagreement


def prepare_and_run_pose_predictions(
    image_jobs: list[tuple],
    engine: str,
    own_cnn_weight: float,
    bregma_ap_index: float,
    progress_messages: queue.SimpleQueue,
    cancel_event: threading.Event,
    own_cnn_model_path: str | Path = OWN_CNN_MODEL_PATH,
) -> tuple[list[dict], dict[str, dict[str, float]], dict, dict[str, dict[str, np.ndarray]]]:
    if engine not in POSE_ENGINES:
        raise ValueError(f"Unknown pose-prediction engine: {engine}")
    if not 0.0 <= own_cnn_weight <= 1.0:
        raise ValueError("Own CNN vote weight must be between 0 and 1")
    use_deepslice = engine in (POSE_ENGINE_DEEPSLICE, POSE_ENGINE_WEIGHTED)
    use_own_cnn = engine in (POSE_ENGINE_OWN_CNN, POSE_ENGINE_WEIGHTED)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="trajectory_pose_") as temporary_folder:
        image_paths, input_crops, prepared_inputs = prepare_pose_inputs(
            image_jobs,
            temporary_folder,
            progress_messages,
            cancel_event,
        )
        component_records: dict[str, dict[str, dict]] = {}
        component_provenance = {}
        component_runtimes = {}
        deepslice_disagreement: dict[str, dict[str, float]] = {}
        if use_deepslice:
            deepslice_records, version, hashes, deepslice_disagreement, deepslice_runtime = run_deepslice_inference(
                image_paths,
                progress_messages,
                cancel_event,
            )
            component_records[POSE_ENGINE_DEEPSLICE] = {
                Path(str(record["Filenames"])).name: record for record in deepslice_records
            }
            component_runtimes[POSE_ENGINE_DEEPSLICE] = deepslice_runtime
            component_provenance[POSE_ENGINE_DEEPSLICE] = {
                "version": version,
                "model_sha256": hashes,
                "backend": deepslice_runtime.get("backend"),
                "device": deepslice_runtime.get("device"),
                "onnxruntime_version": deepslice_runtime.get("onnxruntime_version"),
                "gpu_fallback_reason": deepslice_runtime.get("gpu_fallback_reason"),
            }
        if use_own_cnn:
            progress_messages.put((28 if use_deepslice else 12, "Running own atlas-pose CNN..."))
            own_prediction, own_runtime = run_atlas_pose_onnx(
                [prepared_inputs[Path(path).name]["image"] for path in image_paths],
                [prepared_inputs[Path(path).name]["brain_mask"] for path in image_paths],
                own_cnn_model_path,
                cancel_event,
            )
            benchmark = own_runtime["metadata"].get("real_histology_benchmark", {})
            model_uncertainty = dict(benchmark.get("raw_in_domain", {}))
            component_records[POSE_ENGINE_OWN_CNN] = {
                Path(path).name: {
                    "Filenames": Path(path).name,
                    "pose_ap_um_lr_deg_dv_deg": [float(value) for value in pose],
                    "orientation_inverted": bool(own_runtime["orientation_inverted"][index]),
                    "orientation_inverted_logit": float(own_runtime["orientation_inverted_logit"][index]),
                    "model_uncertainty": model_uncertainty,
                }
                for index, (path, pose) in enumerate(zip(image_paths, own_prediction))
            }
            component_runtimes[POSE_ENGINE_OWN_CNN] = own_runtime
            component_provenance[POSE_ENGINE_OWN_CNN] = {
                key: own_runtime.get(key)
                for key in (
                    "architecture",
                    "model_sha256",
                    "backend",
                    "device",
                    "onnxruntime_version",
                    "gpu_fallback_reason",
                    "preprocessing_version",
                    "automatic_brain_mask_version",
                    "preprocessing_contract_sha256",
                )
            }
        if cancel_event.is_set():
            raise InterruptedError
        records, disagreement = combine_pose_predictions(
            image_paths,
            input_crops,
            component_records,
            deepslice_disagreement,
            engine,
            own_cnn_weight,
            bregma_ap_index,
        )

    tilts = np.asarray(
        [[record["predicted_tilt_lr_deg"], record["predicted_tilt_dv_deg"]] for record in records]
    )
    fallback_reasons = [
        value.get("gpu_fallback_reason") for value in component_runtimes.values() if value.get("gpu_fallback_reason")
    ]
    runtime_info = {
        "engine": engine,
        "coordinate_contract": QUICKNII_COORDINATE_CONTRACT_VERSION,
        "component_provenance": component_provenance,
        "component_runtimes": component_runtimes,
        "fusion": records[0]["fusion"],
        "backend": " + ".join(value.get("backend", "unknown") for value in component_runtimes.values()),
        "device": " + ".join(value.get("device", "unknown") for value in component_runtimes.values()),
        "onnxruntime_version": " + ".join(
            dict.fromkeys(str(value.get("onnxruntime_version", "unknown")) for value in component_runtimes.values())
        ),
        "gpu_fallback_reason": "; ".join(fallback_reasons) or None,
        "inference_seconds": float(
            sum(float(value.get("inference_seconds", 0.0)) for value in component_runtimes.values())
        ),
        "total_backend_seconds": float(time.perf_counter() - started),
        "preintegration_tilt_spread_deg": np.ptp(tilts, axis=0).tolist() if len(tilts) > 1 else [0.0, 0.0],
        "input_crops": input_crops,
    }
    return records, disagreement, runtime_info, prepared_inputs


def _integer_series(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    rounded = np.rint(numeric.to_numpy(dtype=float)).astype(np.int64)
    if not np.allclose(numeric.to_numpy(dtype=float), rounded):
        raise ValueError(f"{label} contains non-integer channel identifiers")
    return pd.Series(rounded, index=values.index, dtype="int64")


def canonical_channel_keys(table: pd.DataFrame, *, units: bool = False) -> pd.DataFrame:
    table = table.copy()
    if "probe_name" not in table.columns:
        if "probe" not in table.columns:
            raise ValueError("Metadata needs probe_name or probe")
        table["probe_name"] = "imec" + _integer_series(table["probe"], "probe").astype(str)
    table["probe_name"] = table["probe_name"].astype(str).str.strip()
    if "probe" in table.columns:
        expected_probe_name = "imec" + _integer_series(table["probe"], "probe").astype(str)
        if not np.array_equal(expected_probe_name.to_numpy(), table["probe_name"].to_numpy()):
            raise ValueError("probe and probe_name disagree")

    if "probe_channel_number" not in table.columns:
        source = "peak_channel" if units else "ks_channel_id"
        if source not in table.columns:
            raise ValueError(f"Metadata needs probe_channel_number or {source}")
        table["probe_channel_number"] = _integer_series(table[source], source)
    else:
        table["probe_channel_number"] = _integer_series(
            table["probe_channel_number"], "probe_channel_number"
        )

    if units and "peak_channel" in table.columns:
        peak_channel = _integer_series(table["peak_channel"], "peak_channel")
        if not np.array_equal(peak_channel.to_numpy(), table["probe_channel_number"].to_numpy()):
            raise ValueError("unit peak_channel and probe_channel_number disagree")
    if not units and "ks_channel_id" in table.columns:
        ks_channel_id = _integer_series(table["ks_channel_id"], "ks_channel_id")
        if not np.array_equal(ks_channel_id.to_numpy(), table["probe_channel_number"].to_numpy()):
            raise ValueError("ks_channel_id and probe_channel_number disagree")

    if not units:
        if "probe_horizontal_position" not in table.columns and "x_um" in table.columns:
            table["probe_horizontal_position"] = pd.to_numeric(table["x_um"], errors="raise")
        if "probe_vertical_position" not in table.columns and "y_um" in table.columns:
            table["probe_vertical_position"] = pd.to_numeric(table["y_um"], errors="raise")
        if "structure_acronym" not in table.columns and "atlas_acronym" in table.columns:
            table["structure_acronym"] = table["atlas_acronym"]
        duplicates = table.duplicated(CHANNEL_KEY_COLUMNS, keep=False)
        if duplicates.any():
            keys = table.loc[duplicates, CHANNEL_KEY_COLUMNS].drop_duplicates().head(10).to_dict("records")
            raise ValueError(f"Duplicate probe/channel keys in channels.csv: {keys}")
    return table


def attach_peak_channel_metadata(channels: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    # Channel numbers repeat across probes, so the composite key is mandatory.
    channels = canonical_channel_keys(channels)
    units = canonical_channel_keys(units, units=True)
    if "unit_key" in units.columns and units["unit_key"].duplicated().any():
        raise ValueError("units.csv contains duplicate unit_key values")

    copy_columns = [
        name
        for name in [
            "probe_horizontal_position",
            "probe_vertical_position",
            "probe_shank",
            *ANATOMY_MAPPING_COLUMNS,
        ]
        if name in channels.columns
    ]
    units = units.drop(columns=[name for name in copy_columns if name in units.columns])
    before = len(units)
    units = units.merge(
        channels[CHANNEL_KEY_COLUMNS + copy_columns],
        on=CHANNEL_KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    if len(units) != before:
        raise ValueError("Peak-channel join changed the number of units")
    unresolved = units[
        units["probe_horizontal_position"].isna()
        | units["probe_vertical_position"].isna()
    ]
    if len(unresolved):
        keys = unresolved[["unit_key", *CHANNEL_KEY_COLUMNS]].head(10).to_dict("records")
        raise ValueError(f"Units have peak channels absent from channels.csv: {keys}")
    return units


def write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_anatomy_sidecars(data_folder: Path, channels: pd.DataFrame, units: pd.DataFrame) -> None:
    anatomy_dir = data_folder / "anatomy"
    anatomy_dir.mkdir(exist_ok=True)
    channel_columns = [
        name
        for name in [
            *CHANNEL_KEY_COLUMNS,
            "probe_horizontal_position",
            "probe_vertical_position",
            "probe_shank",
            *ANATOMY_MAPPING_COLUMNS,
        ]
        if name in channels.columns
    ]
    unit_columns = [
        name
        for name in [
            "unit_key",
            "unit_id",
            *CHANNEL_KEY_COLUMNS,
            "peak_channel_index",
            "probe_horizontal_position",
            "probe_vertical_position",
            "probe_shank",
            *ANATOMY_MAPPING_COLUMNS,
        ]
        if name in units.columns
    ]
    write_csv_atomic(channels[channel_columns], anatomy_dir / "channel_brain_regions.csv")
    write_csv_atomic(units[unit_columns], anatomy_dir / "unit_brain_region_assignments.csv")


def verify_staged_mapping_outputs(
    staging_root: Path,
    channel_rows: int,
    unit_rows: int,
    probe_name: str,
) -> dict[Path, str]:
    channels = pd.read_csv(staging_root / "channels.csv")
    units = pd.read_csv(staging_root / "units.csv")
    channel_sidecar = pd.read_csv(staging_root / "anatomy" / "channel_brain_regions.csv")
    unit_sidecar = pd.read_csv(staging_root / "anatomy" / "unit_brain_region_assignments.csv")
    if len(channels) != channel_rows or len(channel_sidecar) != channel_rows:
        raise RuntimeError("Staged channel mapping row count changed")
    if len(units) != unit_rows or len(unit_sidecar) != unit_rows:
        raise RuntimeError("Staged unit mapping row count changed")
    try:
        pd.testing.assert_frame_equal(
            channel_sidecar.reset_index(drop=True),
            channels[channel_sidecar.columns].reset_index(drop=True),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            unit_sidecar.reset_index(drop=True),
            units[unit_sidecar.columns].reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise RuntimeError("Staged anatomy sidecars do not match their primary CSVs") from exc
    for table, label in ((channels, "channels"), (units, "units")):
        missing = [name for name in ANATOMY_MAPPING_COLUMNS if name not in table.columns]
        if missing:
            raise RuntimeError(f"Staged {label} mapping is missing columns: {', '.join(missing)}")
        canonical_channel_keys(table, units=label == "units")

    manifest_path = staging_root / "anatomy" / f"proprietary_trajectory_manifest_{probe_name}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("probe_name") != probe_name or not isinstance(manifest.get("slices"), list):
        raise RuntimeError("Staged trajectory manifest does not describe this mapping")
    anatomy_root = (staging_root / "anatomy").resolve()
    for slice_record in manifest["slices"]:
        transform_record = slice_record.get("slice_atlas_transform")
        if not transform_record or not transform_record.get("nonlinear"):
            continue
        sidecar = transform_record.get("sidecar")
        if not sidecar:
            raise RuntimeError("Staged nonlinear transform has no sidecar reference")
        sidecar_path = (anatomy_root / sidecar["relative_path"]).resolve()
        sidecar_path.relative_to(anatomy_root)
        if file_sha256(sidecar_path) != sidecar["sha256"]:
            raise RuntimeError("Staged nonlinear transform sidecar checksum failed")
        restored = SliceAtlasTransform2D.load_npz(sidecar_path)
        restored.check_invariants()
        attestation = restored.nonlinear_attestation
        if restored.nonlinear is None or attestation is None:
            raise RuntimeError("Staged nonlinear transform sidecar contains no accepted warp")
        if (
            sidecar.get("coordinate_convention") != restored.coordinate_convention
            or sidecar.get("model_sha256") != attestation.model_sha256
            or sidecar.get("manifest_sha256") != attestation.manifest_sha256
            or sidecar.get("source_image_sha256") != attestation.source_image_sha256
            or sidecar.get("atlas_image_sha256") != attestation.atlas_image_sha256
            or sidecar.get("moving_affine_sha256") != attestation.moving_affine_sha256
            or sidecar.get("runtime_gate_version") != attestation.runtime_gate_version
            or sidecar.get("pixel_spacing_um") != attestation.pixel_spacing_um
        ):
            raise RuntimeError("Staged nonlinear transform metadata disagrees with its sidecar")

    files = sorted(path for path in staging_root.rglob("*") if path.is_file())
    return {path.relative_to(staging_root): file_sha256(path) for path in files}


def promote_staged_mapping_outputs(
    staging_root: Path,
    data_folder: Path,
    backup_dir: Path,
    staged_hashes: dict[Path, str],
    replace_file=os.replace,
) -> None:
    originals: dict[Path, Path | None] = {}
    for relative_path in staged_hashes:
        destination = data_folder / relative_path
        if destination.exists():
            backup = backup_dir / "replaced" / relative_path
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
            if file_sha256(backup) != file_sha256(destination):
                raise RuntimeError(f"Could not verify rollback backup for {relative_path}")
            originals[relative_path] = backup
        else:
            originals[relative_path] = None

    def is_manifest(relative_path: Path) -> bool:
        return (
            relative_path.parent == Path("anatomy")
            and relative_path.name.startswith("proprietary_trajectory_manifest_")
            and relative_path.suffix == ".json"
        )

    # The manifest is the transaction's commit marker: every file it references
    # is installed and checksum-verified before the manifest becomes visible.
    promotion_order = sorted(
        staged_hashes,
        key=lambda path: (is_manifest(path), path.as_posix()),
    )
    promoted: list[Path] = []
    try:
        for relative_path in promotion_order:
            expected_sha256 = staged_hashes[relative_path]
            source = staging_root / relative_path
            destination = data_folder / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            promoted.append(relative_path)
            replace_file(source, destination)
            if file_sha256(destination) != expected_sha256:
                raise RuntimeError(f"Promoted mapping checksum failed for {relative_path}")
    except Exception as promotion_error:
        rollback_errors = []
        for relative_path in reversed(promoted):
            try:
                destination = data_folder / relative_path
                backup = originals[relative_path]
                if backup is None:
                    if destination.exists():
                        destination.unlink()
                else:
                    shutil.copy2(backup, destination)
                    if file_sha256(destination) != file_sha256(backup):
                        raise RuntimeError(f"rollback checksum failed for {relative_path}")
            except Exception as rollback_error:
                rollback_errors.append(str(rollback_error))
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                f"Mapping promotion failed and rollback was incomplete: {details}"
            ) from promotion_error
        raise


def as_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if image.ndim == 3:
        if image.shape[-1] in (3, 4):
            image = image[..., :3].astype(np.float32).mean(axis=-1)
        else:
            image = image[0]
    return image


def normalize_u8(image: np.ndarray) -> np.ndarray:
    image = as_gray(image).astype(np.float32, copy=False)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = np.percentile(image[finite], [0.2, 99.8])
    if hi <= lo:
        lo = float(np.nanmin(image))
        hi = float(np.nanmax(image))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    image = np.clip((image - lo) * 255.0 / (hi - lo), 0, 255)
    return image.astype(np.uint8)


def downsample_for_display(image: np.ndarray, max_side: int = 1800) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    factor = max(1, int(np.ceil(max(h, w) / max_side)))
    if factor == 1:
        return image, 1.0
    resized = cv2.resize(
        image,
        (max(1, round(w / factor)), max(1, round(h / factor))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, float(factor)


def apply_curve(image_u8: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    points = sorted((float(x), float(y)) for x, y in points)
    xs = np.array([0.0] + [x for x, _ in points] + [255.0], dtype=np.float32)
    ys = np.array([0.0] + [y for _, y in points] + [255.0], dtype=np.float32)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    lut = np.interp(np.arange(256, dtype=np.float32), xs, ys)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[image_u8]


def slice_geometry_matrix(
    image_shape: tuple[int, int],
    angle_deg: float,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> tuple[tuple[int, int], np.ndarray]:
    h, w = image_shape[:2]
    if abs(angle_deg) < 0.05:
        out_h, out_w = h, w
        matrix = np.eye(3, dtype=np.float64)
    else:
        center = ((w - 1.0) / 2.0, (h - 1.0) / 2.0)
        rotation = np.eye(3, dtype=np.float64)
        rotation[:2, :] = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
        corners = np.array([[0.0, 0.0, 1.0], [w - 1.0, 0.0, 1.0], [0.0, h - 1.0, 1.0], [w - 1.0, h - 1.0, 1.0]])
        rotated_corners = (rotation @ corners.T).T[:, :2]
        min_xy = rotated_corners.min(axis=0)
        max_xy = rotated_corners.max(axis=0)
        out_w = max(1, int(np.ceil(max_xy[0] - min_xy[0] + 1.0)))
        out_h = max(1, int(np.ceil(max_xy[1] - min_xy[1] + 1.0)))
        translate = np.array([[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]])
        matrix = translate @ rotation

    if flip_horizontal:
        matrix = np.array([[-1.0, 0.0, out_w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ matrix
    if flip_vertical:
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, out_h - 1.0], [0.0, 0.0, 1.0]]) @ matrix
    return (out_h, out_w), matrix


def transform_slice_image(
    image_u8: np.ndarray,
    angle_deg: float,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    out_shape, matrix = slice_geometry_matrix(image_u8.shape[:2], angle_deg, flip_horizontal, flip_vertical)
    if out_shape == image_u8.shape[:2] and np.allclose(matrix, np.eye(3)):
        return image_u8.copy(), matrix
    out_h, out_w = out_shape
    transformed = cv2.warpAffine(
        image_u8,
        matrix[:2, :].astype(np.float32),
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return transformed.astype(np.uint8), matrix


def transform_points(points: list[tuple[float, float]] | np.ndarray, matrix: np.ndarray) -> list[tuple[float, float]]:
    if len(points) == 0:
        return []
    hom = np.column_stack([np.asarray(points, dtype=np.float64), np.ones(len(points), dtype=np.float64)])
    mapped = (matrix @ hom.T).T
    return [(float(x), float(y)) for x, y in mapped[:, :2]]


def red_rgba(image_u8: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*image_u8.shape, 4), dtype=np.uint8)
    rgba[..., 0] = image_u8
    rgba[..., 3] = np.where(image_u8 > 0, np.maximum(image_u8, 35), 0).astype(np.uint8)
    return rgba


def gray_rgba(image_u8: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*image_u8.shape, 4), dtype=np.uint8)
    rgba[..., 0] = image_u8
    rgba[..., 1] = image_u8
    rgba[..., 2] = image_u8
    rgba[..., 3] = 255
    return rgba


def resample_closed_contour(contour: np.ndarray, point_count: int) -> np.ndarray:
    contour_points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    segment_vectors = np.roll(contour_points, -1, axis=0) - contour_points
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    valid = segment_lengths > 0
    contour_points = contour_points[valid]
    segment_vectors = segment_vectors[valid]
    segment_lengths = segment_lengths[valid]
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    targets = np.linspace(0.0, cumulative[-1], int(point_count), endpoint=False)
    segments = np.searchsorted(cumulative[1:], targets, side="right")
    fractions = (targets - cumulative[segments]) / segment_lengths[segments]
    return contour_points[segments] + segment_vectors[segments] * fractions[:, None]


def smart_brain_surface_selection(
    image: np.ndarray,
    foreground_points: list[tuple[float, float]],
    background_points: list[tuple[float, float]],
    brush_radius: float,
    outline_point_count: int = 50,
    max_size: int = 1000,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    if not foreground_points:
        raise RuntimeError("Paint at least one foreground brush stroke on the brain")
    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    small_width = max(2, round(width * scale))
    small_height = max(2, round(height * scale))
    small = cv2.resize(normalize_u8(image), (small_width, small_height), interpolation=cv2.INTER_AREA)
    foreground = np.rint(np.asarray(foreground_points, dtype=np.float32) * scale).astype(np.int32)
    background = np.rint(np.asarray(background_points, dtype=np.float32) * scale).astype(np.int32)
    foreground[:, 0] = np.clip(foreground[:, 0], 0, small_width - 1)
    foreground[:, 1] = np.clip(foreground[:, 1], 0, small_height - 1)
    if len(background):
        background[:, 0] = np.clip(background[:, 0], 0, small_width - 1)
        background[:, 1] = np.clip(background[:, 1], 0, small_height - 1)
    radius = max(3, round(float(brush_radius) * scale))

    blurred = cv2.GaussianBlur(small, (0, 0), 3.0)
    _, bright = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    candidates: list[np.ndarray] = []
    for binary in (bright.astype(np.uint8), (1 - bright).astype(np.uint8)):
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if 0.005 * small.size <= area <= 0.90 * small.size:
                candidates.append((labels == label).astype(np.uint8))

    if candidates:
        def candidate_score(candidate: np.ndarray) -> float:
            distance = cv2.distanceTransform(1 - candidate, cv2.DIST_L2, 3)
            foreground_score = sum(max(0.0, 4.0 * radius - float(distance[y, x])) for x, y in foreground)
            background_penalty = sum(float(candidate[y, x]) for x, y in background) * 6.0 * radius
            touches = sum(
                (
                    bool(np.any(candidate[0])),
                    bool(np.any(candidate[-1])),
                    bool(np.any(candidate[:, 0])),
                    bool(np.any(candidate[:, -1])),
                )
            )
            return foreground_score - background_penalty - touches * radius * len(foreground)

        prior = max(candidates, key=candidate_score)
    else:
        prior = np.zeros_like(small, dtype=np.uint8)
        for x, y in foreground:
            cv2.circle(prior, (int(x), int(y)), radius * 3, 1, -1)

    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(small)
    local_background = cv2.GaussianBlur(small, (0, 0), 5.0)
    texture = cv2.normalize(cv2.absdiff(small, local_background), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    features = cv2.merge([small, clahe, texture])
    grabcut_mask = np.full(small.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[prior > 0] = cv2.GC_PR_FGD
    grabcut_mask[:3] = cv2.GC_BGD
    grabcut_mask[-3:] = cv2.GC_BGD
    grabcut_mask[:, :3] = cv2.GC_BGD
    grabcut_mask[:, -3:] = cv2.GC_BGD
    for x, y in foreground:
        cv2.circle(grabcut_mask, (int(x), int(y)), radius, cv2.GC_PR_FGD, -1)
        cv2.circle(grabcut_mask, (int(x), int(y)), max(2, radius // 3), cv2.GC_FGD, -1)
    for x, y in background:
        cv2.circle(grabcut_mask, (int(x), int(y)), radius, cv2.GC_BGD, -1)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        features,
        grabcut_mask,
        None,
        background_model,
        foreground_model,
        3,
        cv2.GC_INIT_WITH_MASK,
    )
    selected = np.isin(grabcut_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(selected, 8)
    if count <= 1:
        raise RuntimeError("The painted region did not produce a foreground object")
    label = max(range(1, count), key=lambda value: np.count_nonzero((labels == value) & (prior > 0)))
    selected = (labels == label).astype(np.uint8)
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No closed surface was found around the painted region")
    contour = max(contours, key=cv2.contourArea)
    selected.fill(0)
    cv2.drawContours(selected, [contour], -1, 1, -1)
    surface = resample_closed_contour(contour, outline_point_count) / scale
    selection = cv2.resize(selected, (width, height), interpolation=cv2.INTER_NEAREST)
    return [(float(x), float(y)) for x, y in surface], selection


def canonical_mind_input(
    image: np.ndarray,
    brain_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(brain_mask, dtype=bool)
    y, x = np.nonzero(mask)
    if len(x) < 64:
        raise ValueError("The selected brain surface does not enclose enough tissue")
    image_crop = np.asarray(image)[y.min() : y.max() + 1, x.min() : x.max() + 1]
    mask_crop = mask[y.min() : y.max() + 1, x.min() : x.max() + 1]
    width, height = MIND_CANONICAL_SIZE
    canonical = cv2.resize(
        image_crop.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    canonical_mask = cv2.resize(
        mask_crop.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    values = canonical[canonical_mask]
    low, high = np.percentile(values, [1.0, 99.0])
    canonical = np.clip((canonical - low) / max(float(high - low), 1e-4), 0.0, 1.0)
    canonical[~canonical_mask] = 0.0
    return canonical.astype(np.float32), canonical_mask


def mind_descriptor(image: np.ndarray) -> np.ndarray:
    differences = []
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        shifted = cv2.warpAffine(
            image,
            np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        differences.append(cv2.GaussianBlur((image - shifted) ** 2, (0, 0), 1.0))
    differences = np.stack(differences)
    local_variance = np.mean(differences, axis=0)
    return np.exp(-differences / (local_variance + 1e-4)).astype(np.float32)


def mind_distance(
    first_descriptor: np.ndarray,
    first_mask: np.ndarray,
    second_descriptor: np.ndarray,
    second_mask: np.ndarray,
) -> tuple[float, float]:
    overlap = cv2.erode(
        (np.asarray(first_mask) & np.asarray(second_mask)).astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
    ).astype(bool)
    if np.count_nonzero(overlap) < 64:
        return float("inf"), float("inf")
    union = np.asarray(first_mask) | np.asarray(second_mask)
    texture = float(np.mean(np.abs(first_descriptor[:, overlap] - second_descriptor[:, overlap])))
    surface = float(np.count_nonzero(np.asarray(first_mask) ^ np.asarray(second_mask)) / np.count_nonzero(union))
    return texture, surface


def ap_candidate_indices(
    predicted_index: float,
    atlas_length: int,
    bounds: tuple[int, int] | None,
    step: int,
    local_radius: int = 16,
) -> list[int]:
    if bounds is None:
        center = float(np.clip(predicted_index, 0, atlas_length - 1))
        minimum = max(0, int(np.floor(center)) - local_radius)
        maximum = min(atlas_length - 1, int(np.ceil(center)) + local_radius)
    else:
        minimum, maximum = sorted(int(value) for value in bounds)
        minimum = int(np.clip(minimum, 0, atlas_length - 1))
        maximum = int(np.clip(maximum, 0, atlas_length - 1))
    candidates = list(range(minimum, maximum + 1, max(1, int(step))))
    if candidates[-1] != maximum:
        candidates.append(maximum)
    return candidates


def solve_ordered_lattice(
    lattices: dict[int, dict[int, tuple[float, dict]]],
    anterior_to_posterior: list[int],
) -> tuple[dict[int, int], float]:
    assignments = {
        session_index: min(values, key=lambda ap: values[ap][0])
        for session_index, values in lattices.items()
    }
    ordered = list(dict.fromkeys(index for index in anterior_to_posterior if index in lattices))
    if len(ordered) < 2:
        return assignments, float(sum(lattices[index][ap][0] for index, ap in assignments.items()))

    costs = {ap: lattices[ordered[0]][ap][0] for ap in lattices[ordered[0]]}
    paths = {ap: [ap] for ap in costs}
    for session_index in ordered[1:]:
        next_costs = {}
        next_paths = {}
        for ap, (score, _) in lattices[session_index].items():
            predecessors = [previous for previous in costs if previous < ap]
            if not predecessors:
                continue
            previous = min(predecessors, key=costs.get)
            next_costs[ap] = costs[previous] + score
            next_paths[ap] = [*paths[previous], ap]
        costs, paths = next_costs, next_paths
    if not costs:
        raise ValueError("The AP search range is too narrow for the selected slice order")
    last = min(costs, key=costs.get)
    assignments.update(dict(zip(ordered, paths[last])))
    total = sum(lattices[index][ap][0] for index, ap in assignments.items())
    return assignments, float(total)


def solve_probe_constrained_lattice(
    lattices: dict[int, dict[int, tuple[float, dict]]],
    anterior_to_posterior: list[int],
    probe_constraints: dict[str, dict],
    candidate_atlas_points,
    candidate_brain_mask,
    surface_dv,
    bregma_voxel: np.ndarray,
    volume_shape: tuple[int, int, int],
    tilt_lr_deg: float,
    tilt_dv_deg: float,
    *,
    quick: bool = False,
    beam_width: int = 12,
) -> tuple[dict[int, int], float, dict]:
    """Solve the existing AP lattice jointly with optional physical probe plans.

    ``candidate_atlas_points`` returns the observed probe points mapped into a
    candidate atlas slice.  The ordinary lattice is called directly when no
    enabled constraint is supplied; that branch deliberately performs no new
    arithmetic so existing automatic alignment remains exactly unchanged.
    """
    enabled = {
        name: specification
        for name, specification in probe_constraints.items()
        if specification["constraint"].enabled
    }
    if not enabled:
        assignments, total = solve_ordered_lattice(lattices, anterior_to_posterior)
        return assignments, total, {"applied": False, "probes": {}}

    baseline, _ = solve_ordered_lattice(lattices, anterior_to_posterior)
    point_cache: dict[tuple[str, int, int], np.ndarray] = {}
    fit_cache: dict[tuple[tuple[tuple[int, int], ...], bool, int | None], dict] = {}

    def atlas_points(probe_name: str, session_index: int, ap: int) -> np.ndarray:
        key = probe_name, session_index, int(ap)
        if key not in point_cache:
            point_cache[key] = np.asarray(
                candidate_atlas_points(
                    probe_name,
                    session_index,
                    int(ap),
                    float(tilt_lr_deg),
                    float(tilt_dv_deg),
                ),
                dtype=np.float64,
            ).reshape(-1, 2)
        return point_cache[key]

    def fit_assignment(
        assignments: dict[int, int],
        *,
        search_fit: bool,
        max_starts: int | None = None,
    ):
        cache_key = (tuple(sorted(assignments.items())), search_fit, max_starts)
        if cache_key in fit_cache:
            return fit_cache[cache_key]
        fits = {}
        for probe_name, specification in enabled.items():
            observations = {}
            for session_index, ap in assignments.items():
                points = atlas_points(probe_name, session_index, ap)
                if len(points):
                    observations[session_index] = atlas_points_to_stereotaxic_um(
                        points,
                        SlicePlane(ap, tilt_lr_deg, tilt_dv_deg),
                        bregma_voxel,
                        volume_shape,
                        VOXEL_UM,
                    )
            if sum(len(points) for points in observations.values()) < 2:
                raise InfeasibleProbeConstraint(
                    f"{probe_name} needs at least two trajectory observations before its surgical constraint can be used"
                )
            if search_fit:
                fits[probe_name] = fit_probe_ray(
                    observations,
                    specification["constraint"],
                    surface_dv,
                    max_starts=max_starts,
                )
            else:
                fits[probe_name] = fit_observed_probe_ray(
                    observations,
                    specification["constraint"],
                    surface_dv,
                )
        fit_cache[cache_key] = fits
        return fits

    def geometry_score(
        probe_name: str,
        session_index: int,
        ap: int,
        fit,
    ) -> tuple[float, bool]:
        points = atlas_points(probe_name, session_index, ap)
        if not len(points):
            return 0.0, True
        result = score_candidate_slice_plane(
            points,
            SlicePlane(ap, tilt_lr_deg, tilt_dv_deg),
            fit,
            bregma_voxel,
            volume_shape,
            voxel_um=VOXEL_UM,
            brain_mask=candidate_brain_mask(
                int(ap),
                float(tilt_lr_deg),
                float(tilt_dv_deg),
            ),
        )
        return float(result["score"]), bool(result["feasible"])

    # The unconstrained result and coherent AP shifts are deterministic seeds.
    # Coherent shifts matter because a surgical plan often disambiguates an
    # otherwise plausible anterior/posterior image match across several slices.
    seeds = [baseline]
    offsets = sorted(
        {
            int(ap - baseline[index])
            for index, values in lattices.items()
            for ap in values
        }
    )
    if len(offsets) > 33:
        offsets = [offsets[index] for index in np.linspace(0, len(offsets) - 1, 33).round().astype(int)]
    for offset in offsets:
        shifted = {
            index: {
                ap: (abs(ap - (baseline[index] + offset)), {})
                for ap in values
            }
            for index, values in lattices.items()
        }
        try:
            assignment, _ = solve_ordered_lattice(shifted, anterior_to_posterior)
        except ValueError:
            continue
        seeds.append(assignment)
    for rank in range(min(8, min(len(values) for values in lattices.values()))):
        ranked = {
            index: sorted(values, key=lambda ap: values[ap][0])[rank]
            for index, values in lattices.items()
        }
        if all(
            ranked[earlier] < ranked[later]
            for earlier, later in zip(anterior_to_posterior, anterior_to_posterior[1:])
            if earlier in ranked and later in ranked
        ):
            seeds.append(ranked)

    # Cover non-coherent AP corrections without enumerating the Cartesian
    # product. A bounded beam evaluates exact hard-bound observed regressions.
    ordered_sessions = sorted(
        lattices,
        key=lambda index: (len(lattices[index]), index),
    )
    beam = [({}, 0.0)]
    for session_index in ordered_sessions:
        expanded = []
        ranked_aps = sorted(
            lattices[session_index],
            key=lambda ap: lattices[session_index][ap][0],
        )
        if len(ranked_aps) > 24:
            structural = ranked_aps[:12]
            spread = [
                ranked_aps[index]
                for index in np.linspace(0, len(ranked_aps) - 1, 12).round().astype(int)
            ]
            ranked_aps = list(dict.fromkeys([*structural, *spread]))
        for partial, partial_score in beam:
            for ap in ranked_aps:
                assignment = {**partial, session_index: ap}
                order_valid = all(
                    assignment[earlier] < assignment[later]
                    for earlier, later in zip(
                        anterior_to_posterior,
                        anterior_to_posterior[1:],
                    )
                    if earlier in assignment and later in assignment
                )
                if not order_valid:
                    continue
                structural = partial_score + float(lattices[session_index][ap][0])
                geometry = 0.0
                if len(assignment) == len(lattices):
                    try:
                        fits = fit_assignment(assignment, search_fit=True, max_starts=6)
                    except InfeasibleProbeConstraint:
                        continue
                    geometry = MIND_PROBE_GEOMETRY_WEIGHT * float(
                        np.mean([fit.loss for fit in fits.values()])
                    )
                expanded.append((assignment, structural, geometry))
        if not expanded:
            break
        expanded.sort(key=lambda item: (item[2], item[1]))
        beam = [
            (assignment, structural)
            for assignment, structural, _ in expanded[: max(1, int(beam_width))]
        ]
    seeds.extend(
        assignment
        for assignment, _ in beam
        if len(assignment) == len(lattices)
    )
    unique_seeds = list({
        tuple(sorted(assignment.items())): assignment
        for assignment in seeds
    }.values())

    quick_feasible = []
    failures = []
    for assignment in unique_seeds:
        try:
            fits = fit_assignment(assignment, search_fit=True, max_starts=6)
        except InfeasibleProbeConstraint as exc:
            failures.append(str(exc))
            continue
        structural = float(sum(lattices[index][ap][0] for index, ap in assignment.items()))
        geometry = float(np.mean([fit.loss for fit in fits.values()]))
        quick_feasible.append(
            (structural + MIND_PROBE_GEOMETRY_WEIGHT * geometry, assignment, fits)
        )
    if not quick_feasible:
        detail = failures[0] if failures else "no feasible candidate assignment"
        raise InfeasibleProbeConstraint(
            f"No atlas pose satisfies the enabled surgical probe constraints: {detail}"
        )
    if quick:
        _, assignments, fits = min(quick_feasible, key=lambda item: item[0])
    else:
        feasible = []
        for _, assignment, _ in sorted(quick_feasible, key=lambda item: item[0]):
            try:
                fits = fit_assignment(assignment, search_fit=False)
            except InfeasibleProbeConstraint as exc:
                failures.append(str(exc))
                continue
            structural = float(sum(lattices[index][ap][0] for index, ap in assignment.items()))
            geometry = float(np.mean([fit.loss for fit in fits.values()]))
            feasible.append(
                (structural + MIND_PROBE_GEOMETRY_WEIGHT * geometry, assignment, fits)
            )
            if len(feasible) == 4:
                break
        if not feasible:
            raise InfeasibleProbeConstraint(
                "No atlas pose satisfies the enabled surgical probe constraints: "
                + failures[-1]
            )
        _, assignments, fits = min(feasible, key=lambda item: item[0])

    # Alternate a fitted constrained ray with the ordered AP lattice.  This is
    # bounded and deterministic; it permits individual AP corrections while
    # retaining partial order and the caller's exactly shared cutting tilt.
    for _ in range(1 if quick else 3):
        constrained = {}
        for session_index, values in lattices.items():
            constrained[session_index] = {}
            for ap, (score, components) in values.items():
                candidate_scores = []
                candidate_feasible = True
                for probe_name, fit in fits.items():
                    value, valid = geometry_score(probe_name, session_index, ap, fit)
                    if len(atlas_points(probe_name, session_index, ap)):
                        candidate_scores.append(value)
                    candidate_feasible &= valid
                geometry = float(np.mean(candidate_scores)) if candidate_scores else 0.0
                constrained[session_index][ap] = (
                    float(score + MIND_PROBE_GEOMETRY_WEIGHT * geometry)
                    if candidate_feasible
                    else float("inf"),
                    components,
                )
        updated, _ = solve_ordered_lattice(constrained, anterior_to_posterior)
        if not np.isfinite(sum(constrained[index][ap][0] for index, ap in updated.items())):
            raise InfeasibleProbeConstraint(
                "No ordered atlas assignment satisfies the enabled surgical probe constraints"
            )
        if updated == assignments:
            break
        try:
            updated_fits = fit_assignment(
                updated,
                search_fit=quick,
                max_starts=6 if quick else None,
            )
        except InfeasibleProbeConstraint:
            # Per-slice scores can prefer a combination whose joint regression
            # is invalid. Retain the last jointly validated solution.
            break
        assignments, fits = updated, updated_fits

    structural_total = float(sum(lattices[index][ap][0] for index, ap in assignments.items()))
    geometry_loss = float(np.mean([fit.loss for fit in fits.values()]))
    diagnostics = {
        "applied": True,
        "weight": MIND_PROBE_GEOMETRY_WEIGHT,
        "seed_count": len(unique_seeds),
        "geometry_loss": geometry_loss,
        "probes": {
            name: {
                "entry_ap_dv_ml_um": fit.entry_ap_dv_ml_um.tolist(),
                "direction_ap_dv_ml": fit.direction_ap_dv_ml.tolist(),
                "angle_deg": float(fit.angle_deg),
                "loss": float(fit.loss),
                "diagnostics": fit.diagnostics,
            }
            for name, fit in fits.items()
        },
    }
    return (
        assignments,
        structural_total + MIND_PROBE_GEOMETRY_WEIGHT * geometry_loss,
        diagnostics,
    )


def surface_crop_bounds(
    surface_points: list[tuple[float, float]],
    image_shape: tuple[int, int],
    margin_fraction: float,
) -> tuple[int, int, int, int]:
    points = np.asarray(surface_points, dtype=np.float64)
    height, width = image_shape[:2]
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    margin = np.maximum(12.0, span * float(margin_fraction))
    x0 = int(np.clip(np.floor(minimum[0] - margin[0]), 0, width - 1))
    y0 = int(np.clip(np.floor(minimum[1] - margin[1]), 0, height - 1))
    x1 = int(np.clip(np.ceil(maximum[0] + margin[0] + 1.0), x0 + 1, width))
    y1 = int(np.clip(np.ceil(maximum[1] + margin[1] + 1.0), y0 + 1, height))
    return x0, y0, x1, y1


def alignment_review_reasons(disagreement: dict, diagnostics: dict) -> list[str]:
    reasons = []
    if diagnostics.get("alignment_run_stale", False):
        reasons.append("alignment input changed; rerun required")
    if disagreement.get("ap_um", 0.0) >= DEEPSLICE_REVIEW_AP_UM:
        reasons.append("model AP disagreement")
    if max(disagreement.get("lr_deg", 0.0), disagreement.get("dv_deg", 0.0)) >= DEEPSLICE_REVIEW_TILT_DEG:
        reasons.append("model tilt disagreement")
    if abs(diagnostics.get("ap_search_shift_um", 0.0)) >= DEEPSLICE_REVIEW_AP_UM:
        reasons.append("large AP refinement")
    if diagnostics.get("pose_search_boundary"):
        reasons.append(
            "best match is at AP search boundary"
            if diagnostics.get("pose_search_explicit_bounds")
            else "best match is at local AP-search edge; set an explicit range"
        )
    if diagnostics.get("pose_search_flat"):
        reasons.append("ambiguous AP score curve")
    if diagnostics.get("surface_rms_after_atlas_px", 0.0) > DEEPSLICE_REVIEW_SURFACE_RMS_PX:
        reasons.append("poor surface fit")
    return reasons


def fit_surface_scale_translation(
    slice_to_atlas: np.ndarray,
    surface_points: list[tuple[float, float]],
    atlas_brain_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    points = np.asarray(surface_points, dtype=np.float64)
    if len(points) < 8:
        raise ValueError("At least 8 trusted surface points are required")
    mask = np.asarray(atlas_brain_mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("The predicted atlas plane has no brain surface")
    largest_area = max(cv2.contourArea(contour) for contour in contours)
    boundary = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(
        boundary,
        [contour for contour in contours if cv2.contourArea(contour) >= 0.02 * largest_area],
        -1,
        1,
        1,
    )
    distance = cv2.distanceTransform(1 - boundary, cv2.DIST_L2, 5)
    boundary_y, boundary_x = np.nonzero(boundary)
    boundary_points = np.column_stack([boundary_x, boundary_y]).astype(np.float64)
    mapped = np.asarray(transform_points(surface_points, slice_to_atlas), dtype=np.float64)
    center = mapped.mean(axis=0)
    height, width = mask.shape

    def corrected(parameters: np.ndarray) -> np.ndarray:
        scale = float(np.exp(parameters[0]))
        return scale * (mapped - center) + center + parameters[1:3]

    def surface_distances(parameters: np.ndarray) -> np.ndarray:
        candidate = corrected(parameters)
        return map_coordinates(
            distance,
            [candidate[:, 1], candidate[:, 0]],
            order=1,
            mode="constant",
            cval=float(np.hypot(height, width)),
            prefilter=False,
        )

    def residuals(parameters: np.ndarray) -> np.ndarray:
        return np.r_[
            surface_distances(parameters) / 2.0,
            parameters[0] / 0.18,
            parameters[1] / (0.10 * width),
            parameters[2] / (0.10 * height),
        ]

    lower = np.asarray([np.log(0.60), -0.25 * width, -0.25 * height])
    upper = np.asarray([np.log(1.60), 0.25 * width, 0.25 * height])
    atlas_center = (boundary_points.min(axis=0) + boundary_points.max(axis=0)) / 2.0
    atlas_span = np.maximum(np.ptp(boundary_points, axis=0), 1.0)
    mapped_span = np.maximum(np.ptp(mapped, axis=0), 1.0)
    extent_scale = float(np.clip(np.median(atlas_span / mapped_span), 0.60, 1.60))
    center_translation = atlas_center - center
    starts = [
        np.zeros(3, dtype=np.float64),
        np.asarray([np.log(extent_scale), *center_translation]),
        np.asarray([0.0, *center_translation]),
        np.asarray([np.log(extent_scale), 0.0, 0.0]),
    ]
    results = [
        least_squares(
            residuals,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=160,
        )
        for start in starts
    ]
    initial = starts[0]
    successful = [result for result in results if result.success]
    parameters = min(successful, key=lambda result: result.cost).x if successful else initial
    before = surface_distances(initial)
    after = surface_distances(parameters)
    if np.sqrt(np.mean(after**2)) > np.sqrt(np.mean(before**2)):
        parameters = initial
        after = before
    scale = float(np.exp(parameters[0]))
    tx, ty = (float(value) for value in parameters[1:3])
    correction = np.asarray(
        [
            [scale, 0.0, center[0] + tx - scale * center[0]],
            [0.0, scale, center[1] + ty - scale * center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return correction @ np.asarray(slice_to_atlas, dtype=np.float64), {
        "scale": scale,
        "translation_x_atlas_px": tx,
        "translation_y_atlas_px": ty,
        "rms_before_atlas_px": float(np.sqrt(np.mean(before**2))),
        "rms_after_atlas_px": float(np.sqrt(np.mean(after**2))),
        "trusted_point_count": int(len(points)),
    }


def orientation_preserving_slice_to_atlas(
    slice_to_atlas: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Keep the user-selected display orientation authoritative."""
    matrix = np.asarray(slice_to_atlas, dtype=np.float64)
    if np.linalg.det(matrix[:2, :2]) > 0.0:
        return matrix
    return brain_mask_affine(source_mask, target_mask)


def refine_pose_search(
    converted: dict[int, tuple[float, float, float, np.ndarray | None]],
    records_by_session: dict[int, dict],
    atlas_volume: np.ndarray,
    annotation_volume: np.ndarray,
    prepared_inputs: dict[str, dict[str, np.ndarray]],
    disagreement: dict[str, dict[str, float]],
    ap_bounds: tuple[int, int] | None,
    order_snapshot: list[int],
    progress_messages: queue.SimpleQueue | None,
    cancel_event: threading.Event | None,
    *,
    global_alignment: bool,
    probe_constraints: dict[str, dict] | None = None,
    bregma_voxel: np.ndarray | None = None,
    trusted_surface_points: dict[int, list[tuple[float, float]]] | None = None,
    cortical_region_ids: frozenset[int] | None = None,
) -> tuple[dict[int, tuple[int, float, float]], dict[int, dict], tuple[float, float] | None]:
    sources = {}
    filenames = {}
    for session_index, record in records_by_session.items():
        filename = Path(str(record["Filenames"])).name
        prepared = prepared_inputs[filename]
        canonical, mask = canonical_mind_input(prepared["image"], prepared["brain_mask"])
        sources[session_index] = (mind_descriptor(canonical), mask)
        filenames[session_index] = filename

    active_probe_constraints = {
        name: specification
        for name, specification in (probe_constraints or {}).items()
        if specification["constraint"].enabled
    }
    observed_probe_constraints = {
        name: specification
        for name, specification in active_probe_constraints.items()
        if sum(len(points) for points in specification["display_points_by_session"].values()) >= 2
    }
    if active_probe_constraints and bregma_voxel is None:
        raise ValueError("Bregma is required when surgical probe constraints are enabled")
    if active_probe_constraints and trusted_surface_points is None:
        raise ValueError("Trusted surface points are required when surgical probe constraints are enabled")
    if active_probe_constraints and not cortical_region_ids:
        raise ValueError("Cortical atlas structure IDs are required when surgical probe constraints are enabled")
    bregma = None if bregma_voxel is None else np.asarray(bregma_voxel, dtype=np.float64)
    candidate_transform_cache: dict[tuple[int, int, float, float], np.ndarray] = {}

    def surface_dv(ap_um: float, ml_um: float) -> float:
        assert bregma is not None
        ap = int(round(ap_um / (-VOXEL_UM) + bregma[0]))
        ml = int(round(ml_um / VOXEL_UM + bregma[2]))
        if not (0 <= ap < annotation_volume.shape[0] and 0 <= ml < annotation_volume.shape[2]):
            return float("nan")
        inside = np.flatnonzero(annotation_volume[ap, :, ml] > 0)
        if not len(inside):
            return float("nan")
        if int(annotation_volume[ap, inside.min(), ml]) not in cortical_region_ids:
            return float("nan")
        return float((inside.min() - bregma[1]) * -VOXEL_UM)

    insertion_surface_entries = {
        name: prepare_insertion_surface_entries(specification["constraint"], surface_dv)
        for name, specification in active_probe_constraints.items()
    }

    def candidate_probe_atlas_points(
        probe_name: str,
        session_index: int,
        ap: int,
        tilt_lr: float,
        tilt_dv: float,
    ) -> np.ndarray:
        display_points = np.asarray(
            active_probe_constraints[probe_name]["display_points_by_session"].get(
                session_index,
                (),
            ),
            dtype=np.float64,
        ).reshape(-1, 2)
        if not len(display_points):
            return np.empty((0, 2), dtype=np.float64)
        key = session_index, int(ap), float(tilt_lr), float(tilt_dv)
        matrix = candidate_transform_cache.get(key)
        if matrix is None:
            target_mask = coronal_oblique_slice(
                annotation_volume,
                int(ap),
                float(tilt_lr),
                float(tilt_dv),
                order=0,
            ) > 0
            matrix = converted[session_index][3]
            if matrix is None:
                filename = filenames[session_index]
                matrix = brain_mask_affine(
                    prepared_inputs[filename]["brain_mask"],
                    target_mask,
                )
            else:
                filename = filenames[session_index]
                matrix = orientation_preserving_slice_to_atlas(
                    matrix,
                    prepared_inputs[filename]["brain_mask"],
                    target_mask,
                )
            matrix, _ = fit_surface_scale_translation(
                matrix,
                trusted_surface_points[session_index],
                target_mask,
            )
            candidate_transform_cache[key] = matrix
        return np.asarray(transform_points(display_points, matrix), dtype=np.float64)

    def atlas_reference(ap: int, tilt_lr: float, tilt_dv: float) -> tuple[np.ndarray, np.ndarray] | None:
        plane = coronal_oblique_slice_resampled(
            atlas_volume,
            int(ap),
            float(tilt_lr),
            float(tilt_dv),
            order=1,
        )
        mask = coronal_oblique_slice_resampled(
            annotation_volume,
            int(ap),
            float(tilt_lr),
            float(tilt_dv),
            order=0,
        ) > 0
        if np.count_nonzero(mask) < 64:
            return None
        canonical, canonical_mask = canonical_mind_input(plane, mask)
        return mind_descriptor(canonical), canonical_mask

    evaluated_poses = {session_index: 0 for session_index in converted}

    def candidates_for(session_index: int, step: int) -> list[int]:
        if active_probe_constraints and ap_bounds is None:
            return ap_candidate_indices(
                converted[session_index][0],
                atlas_volume.shape[0],
                None,
                step,
                48,
            )
        record = records_by_session[session_index]
        uncertainty = record.get("model_uncertainty", {})
        radius_um = max(
            400.0,
            float(uncertainty.get("ap_p95_um", 0.0)),
            float(disagreement.get(filenames[session_index], {}).get("ap_um", 0.0)),
        )
        radius = int(np.clip(np.ceil(radius_um / VOXEL_UM), 16, 32))
        return ap_candidate_indices(
            converted[session_index][0],
            atlas_volume.shape[0],
            ap_bounds,
            step,
            radius,
        )

    def lattice(
        tilt_lr: float,
        tilt_dv: float,
        candidates: dict[int, list[int]],
    ) -> dict[int, dict[int, tuple[float, dict]]]:
        candidate_sets = {index: set(values) for index, values in candidates.items()}
        result = {index: {} for index in candidates}
        for ap in sorted(set().union(*candidate_sets.values())):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError
            reference = atlas_reference(ap, tilt_lr, tilt_dv)
            for session_index, ap_values in candidate_sets.items():
                if ap not in ap_values:
                    continue
                predicted_ap, predicted_lr, predicted_dv, _ = converted[session_index]
                model_difference = disagreement.get(filenames[session_index], {})
                model_uncertainty = records_by_session[session_index].get("model_uncertainty", {})
                ap_sigma = max(
                    8.0,
                    float(model_difference.get("ap_um", 0.0)) / VOXEL_UM,
                    float(model_uncertainty.get("ap_mae_um", 0.0)) / VOXEL_UM,
                )
                if ap_bounds is not None:
                    minimum, maximum = sorted(ap_bounds)
                    if predicted_ap < minimum or predicted_ap > maximum:
                        distance = min(abs(predicted_ap - minimum), abs(predicted_ap - maximum))
                        ap_sigma = max(ap_sigma, float(maximum - minimum), 2.0 * distance)
                lr_sigma = max(
                    3.0,
                    float(model_difference.get("lr_deg", 0.0)),
                    float(model_uncertainty.get("lr_mae_deg", 0.0)),
                )
                dv_sigma = max(
                    3.0,
                    float(model_difference.get("dv_deg", 0.0)),
                    float(model_uncertainty.get("dv_mae_deg", 0.0)),
                )
                source_descriptor, source_mask = sources[session_index]
                if reference is None:
                    texture_distance = surface_distance = float("inf")
                else:
                    texture_distance, surface_distance = mind_distance(
                        source_descriptor,
                        source_mask,
                        *reference,
                    )
                structural_distance = texture_distance + MIND_SURFACE_WEIGHT * surface_distance
                plan_feasibility = {
                    name: insertion_plan_plane_feasibility(
                        specification["constraint"],
                        SlicePlane(ap, tilt_lr, tilt_dv),
                        bregma,
                        atlas_volume.shape,
                        surface_dv,
                        VOXEL_UM,
                        insertion_surface_entries[name],
                    )
                    for name, specification in active_probe_constraints.items()
                } if active_probe_constraints else {}
                if plan_feasibility and not all(value["feasible"] for value in plan_feasibility.values()):
                    structural_distance = float("inf")
                ap_prior = MIND_AP_PRIOR_WEIGHT * ((ap - predicted_ap) / ap_sigma) ** 2
                tilt_prior = MIND_TILT_PRIOR_WEIGHT * (
                    ((tilt_lr - predicted_lr) / lr_sigma) ** 2
                    + ((tilt_dv - predicted_dv) / dv_sigma) ** 2
                )
                result[session_index][int(ap)] = (
                    float(structural_distance + ap_prior + tilt_prior),
                    {
                        "mind_texture_distance": float(texture_distance),
                        "surface_shape_distance": float(surface_distance),
                        "model_ap_prior": float(ap_prior),
                        "model_tilt_prior": float(tilt_prior),
                        "probe_plan_feasibility": plan_feasibility,
                    },
                )
                evaluated_poses[session_index] += 1
        return result

    def coarse_tilts(center: float) -> list[float]:
        values = set(float(value) for value in np.arange(-12.0, 12.1, 4.0))
        values.add(float(np.clip(center, -20.0, 20.0)))
        if abs(center) > 12.0:
            values.update(float(np.clip(center + offset, -20.0, 20.0)) for offset in (-4.0, 4.0))
        return sorted(values)

    def search_group(
        session_indices: list[int],
        ordered_indices: list[int],
        progress_start: int,
        progress_end: int,
    ) -> tuple[dict[int, tuple[int, float, float]], dict[int, dict], tuple[float, float]]:
        predicted_tilt = np.mean(
            np.asarray([(converted[index][1], converted[index][2]) for index in session_indices]),
            axis=0,
        )
        coarse_candidates = {
            index: candidates_for(index, 4)
            for index in session_indices
        }

        def solve_candidates(values, ordered, tilt_lr, tilt_dv, *, quick=False):
            if not active_probe_constraints:
                assignments, total = solve_ordered_lattice(values, ordered)
                return assignments, total, None
            plan_diagnostics = {
                "applied": True,
                "mode": "alignment_feasibility_only",
                "probes": {
                    name: {
                        "constraint": asdict(specification["constraint"]),
                        "trajectory_observations_used": name in observed_probe_constraints,
                    }
                    for name, specification in active_probe_constraints.items()
                },
            }
            if not observed_probe_constraints:
                assignments, total = solve_ordered_lattice(values, ordered)
                return assignments, total, plan_diagnostics
            assert bregma is not None
            assignments, total, diagnostics = solve_probe_constrained_lattice(
                values,
                ordered,
                observed_probe_constraints,
                candidate_probe_atlas_points,
                lambda ap, lr, dv: coronal_oblique_slice(
                    annotation_volume,
                    int(ap),
                    float(lr),
                    float(dv),
                    order=0,
                ) > 0,
                surface_dv,
                bregma,
                tuple(atlas_volume.shape),
                float(tilt_lr),
                float(tilt_dv),
                quick=quick,
            )
            diagnostics["mode"] = "alignment_feasibility_and_observed_geometry_hard_bounds"
            for name, value in plan_diagnostics["probes"].items():
                diagnostics["probes"].setdefault(name, value)
            return assignments, total, diagnostics

        initial_lattice = lattice(float(predicted_tilt[0]), float(predicted_tilt[1]), coarse_candidates)
        initial_assignment, _ = solve_ordered_lattice(initial_lattice, ordered_indices)
        fixed_candidates = {index: [ap] for index, ap in initial_assignment.items()}
        tilt_pairs = [
            (tilt_lr, tilt_dv)
            for tilt_lr in coarse_tilts(float(predicted_tilt[0]))
            for tilt_dv in coarse_tilts(float(predicted_tilt[1]))
        ]
        coarse_results = []
        for sequence, (tilt_lr, tilt_dv) in enumerate(tilt_pairs, start=1):
            if progress_messages is not None:
                progress_messages.put(
                    (
                        progress_start + round((progress_end - progress_start) * 0.40 * sequence / len(tilt_pairs)),
                        f"Searching atlas tilt {sequence} / {len(tilt_pairs)}...",
                    )
                )
            values = lattice(tilt_lr, tilt_dv, fixed_candidates)
            assignments, total = solve_ordered_lattice(values, ordered_indices)
            coarse_results.append((total, tilt_lr, tilt_dv, assignments, values))
        if active_probe_constraints:
            screened_coarse = []
            constraint_failures = []
            # A surgical constraint must jointly choose AP and tilt. Screening
            # only the unconstrained AP winner falsely rejects feasible poses.
            # Search the structurally credible tilt quartile with each slice's
            # complete coarse AP lattice, then perform the exact refinement.
            ranked_coarse = sorted(coarse_results, key=lambda item: item[0])
            primary_count = max(12, len(ranked_coarse) // 4)
            for _, tilt_lr, tilt_dv, _, _ in ranked_coarse[:primary_count]:
                values = lattice(tilt_lr, tilt_dv, coarse_candidates)
                try:
                    assignments, total, geometry = solve_candidates(
                        values, ordered_indices, tilt_lr, tilt_dv, quick=True
                    )
                except InfeasibleProbeConstraint as exc:
                    constraint_failures.append(str(exc))
                    continue
                screened_coarse.append(
                    (total, tilt_lr, tilt_dv, assignments, geometry, values)
                )
            # If the image-ranked tilts conflict with the surgical plan, finish
            # the bounded grid before declaring the user's constraints infeasible.
            if not screened_coarse:
                for _, tilt_lr, tilt_dv, _, _ in ranked_coarse[primary_count:]:
                    values = lattice(tilt_lr, tilt_dv, coarse_candidates)
                    try:
                        assignments, total, geometry = solve_candidates(
                            values, ordered_indices, tilt_lr, tilt_dv, quick=True
                        )
                    except InfeasibleProbeConstraint as exc:
                        constraint_failures.append(str(exc))
                        continue
                    screened_coarse.append(
                        (total, tilt_lr, tilt_dv, assignments, geometry, values)
                    )
            if not screened_coarse:
                raise InfeasibleProbeConstraint(
                    "No coarse atlas tilt satisfies the surgical probe constraints: "
                    + constraint_failures[0]
                )
            constrained_coarse = []
            for _, tilt_lr, tilt_dv, _, _, values in sorted(
                screened_coarse, key=lambda item: item[0]
            )[:4]:
                assignments, total, geometry = solve_candidates(
                    values, ordered_indices, tilt_lr, tilt_dv
                )
                constrained_coarse.append(
                    (total, tilt_lr, tilt_dv, assignments, geometry)
                )
            _, coarse_lr, coarse_dv, _, _ = min(
                constrained_coarse, key=lambda item: item[0]
            )
        else:
            _, coarse_lr, coarse_dv, _, _ = min(coarse_results, key=lambda item: item[0])

        updated_lattice = lattice(coarse_lr, coarse_dv, coarse_candidates)
        updated_assignment, _, _ = solve_candidates(
            updated_lattice, ordered_indices, coarse_lr, coarse_dv
        )
        fine_candidates = {
            index: [
                ap
                for ap in candidates_for(index, 1)
                if abs(ap - updated_assignment[index]) <= 3
            ]
            for index in session_indices
        }
        fine_tilts = [
            (float(coarse_lr + lr_offset), float(coarse_dv + dv_offset))
            for lr_offset in range(-3, 4)
            for dv_offset in range(-3, 4)
        ]
        fine_results = []
        for sequence, (tilt_lr, tilt_dv) in enumerate(fine_tilts, start=1):
            if progress_messages is not None:
                progress_messages.put(
                    (
                        progress_start
                        + round(
                            (progress_end - progress_start)
                            * (0.40 + 0.40 * sequence / len(fine_tilts))
                        ),
                        f"Refining atlas pose {sequence} / {len(fine_tilts)}...",
                    )
                )
            values = lattice(tilt_lr, tilt_dv, fine_candidates)
            assignments, total = solve_ordered_lattice(values, ordered_indices)
            fine_results.append((total, tilt_lr, tilt_dv, assignments, values))
        if active_probe_constraints:
            screened_fine = []
            constraint_failures = []
            for _, tilt_lr, tilt_dv, _, values in sorted(
                fine_results, key=lambda item: item[0]
            ):
                try:
                    assignments, total, geometry = solve_candidates(
                        values, ordered_indices, tilt_lr, tilt_dv, quick=True
                    )
                except InfeasibleProbeConstraint as exc:
                    constraint_failures.append(str(exc))
                    continue
                screened_fine.append(
                    (total, tilt_lr, tilt_dv, assignments, geometry)
                )
            if not screened_fine:
                raise InfeasibleProbeConstraint(
                    "No refined atlas tilt satisfies the surgical probe constraints: "
                    + constraint_failures[0]
                )
            constrained_fine = []
            for _, tilt_lr, tilt_dv, _, _ in sorted(
                screened_fine, key=lambda item: item[0]
            )[:4]:
                values = next(
                    item[4]
                    for item in fine_results
                    if item[1] == tilt_lr and item[2] == tilt_dv
                )
                assignments, total, geometry = solve_candidates(
                    values, ordered_indices, tilt_lr, tilt_dv
                )
                constrained_fine.append(
                    (total, tilt_lr, tilt_dv, assignments, geometry)
                )
            _, best_lr, best_dv, _, _ = min(
                constrained_fine, key=lambda item: item[0]
            )
        else:
            _, best_lr, best_dv, _, _ = min(fine_results, key=lambda item: item[0])

        final_candidates = {
            index: candidates_for(index, 1)
            for index in session_indices
        }
        final_lattice = lattice(best_lr, best_dv, final_candidates)
        assignments, _, probe_geometry = solve_candidates(
            final_lattice, ordered_indices, best_lr, best_dv
        )
        group_pose = {
            index: (int(ap), float(best_lr), float(best_dv))
            for index, ap in assignments.items()
        }
        group_diagnostics = {}
        for index, ap in assignments.items():
            values = final_lattice[index]
            alternatives = sorted(
                score
                for candidate_ap, (score, _) in values.items()
                if abs(candidate_ap - ap) >= 2
            )
            structural_score, components = values[ap]
            if not np.isfinite(structural_score):
                raise ValueError("The AP search range contains no usable atlas brain sections")
            structural_margin = (
                float(alternatives[0] - structural_score) if alternatives else float("nan")
            )
            minimum, maximum = min(values), max(values)
            group_diagnostics[index] = {
                **components,
                "pose_search_final_ap_candidate_count": int(len(values)),
                "pose_search_evaluated_pose_count": int(evaluated_poses[index]),
                "pose_search_boundary": bool(ap in (minimum, maximum)),
                "pose_search_explicit_bounds": ap_bounds is not None,
                "pose_search_method": (
                    "coarse-to-fine MIND + disagreement-weighted model prior"
                    + (
                        " + jointly fitted surgical probe constraints"
                        if probe_geometry is not None
                        else ""
                    )
                ),
            }
            group_diagnostics[index].update(
                pose_search_score=float(structural_score),
                pose_search_margin=structural_margin,
                pose_search_flat=bool(
                    np.isfinite(structural_margin) and structural_margin < 0.0003
                ),
            )
            if probe_geometry is not None:
                group_diagnostics[index].update(
                    pose_search_structural_score=float(structural_score),
                    pose_search_structural_margin=structural_margin,
                    probe_geometry_constraints=probe_geometry,
                )
        return group_pose, group_diagnostics, (float(best_lr), float(best_dv))

    groups = [list(converted)] if global_alignment else [[index] for index in converted]
    pose = {}
    diagnostics = {}
    shared_tilt = None
    for group_number, session_indices in enumerate(groups):
        progress_start = 30 + round(60 * group_number / len(groups))
        progress_end = 30 + round(60 * (group_number + 1) / len(groups))
        group_order = [index for index in order_snapshot if index in session_indices]
        group_pose, group_diagnostics, group_tilt = search_group(
            session_indices,
            group_order,
            progress_start,
            progress_end,
        )
        pose.update(group_pose)
        diagnostics.update(group_diagnostics)
        if global_alignment:
            shared_tilt = group_tilt
    return pose, diagnostics, shared_tilt


def solve_pose_alignment(
    records: list[dict],
    filename_to_session: dict[str, int],
    annotation_volume: np.ndarray,
    outline_snapshot: dict[int, list[tuple[float, float]]],
    ap_bounds: tuple[int, int] | None,
    order_snapshot: list[int],
    runtime_info: dict,
    cancel_event: threading.Event | None = None,
    *,
    global_alignment: bool,
    atlas_volume: np.ndarray,
    prepared_inputs: dict[str, dict[str, np.ndarray]],
    disagreement: dict[str, dict[str, float]],
    progress_messages: queue.SimpleQueue | None = None,
    probe_constraints: dict[str, dict] | None = None,
    bregma_voxel: np.ndarray | None = None,
    cortical_region_ids: frozenset[int] | None = None,
) -> tuple[list[tuple], tuple[float, float] | None]:
    records_by_session = {
        filename_to_session[Path(str(record["Filenames"])).name]: dict(record)
        for record in records
    }
    if len(records_by_session) != len(filename_to_session):
        raise RuntimeError("The pose predictor did not return exactly one result for every input slice")

    converted = {}
    input_crops = runtime_info.get("input_crops", {})
    for index, record in records_by_session.items():
        atlas_index = float(record["predicted_atlas_index"])
        tilt_ml = float(record["predicted_tilt_lr_deg"])
        tilt_dv = float(record["predicted_tilt_dv_deg"])
        matrix = record.get("initial_slice_to_atlas")
        converted[index] = (
            atlas_index,
            tilt_ml,
            tilt_dv,
            None if matrix is None else np.asarray(matrix, dtype=np.float64),
        )

    pose, search_diagnostics, shared_tilt = refine_pose_search(
        converted,
        records_by_session,
        atlas_volume,
        annotation_volume,
        prepared_inputs,
        disagreement,
        ap_bounds,
        order_snapshot,
        progress_messages,
        cancel_event,
        global_alignment=global_alignment,
        probe_constraints=probe_constraints,
        bregma_voxel=bregma_voxel,
        trusted_surface_points=outline_snapshot,
        cortical_region_ids=cortical_region_ids,
    )

    prepared = []
    for session_index, (raw_atlas_index, tilt_ml, tilt_dv, matrix) in converted.items():
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError
        atlas_index, tilt_ml, tilt_dv = pose[session_index]
        atlas_mask = coronal_oblique_slice(
            annotation_volume,
            atlas_index,
            float(tilt_ml),
            float(tilt_dv),
            order=0,
        ) > 0
        if matrix is None:
            filename = Path(str(records_by_session[session_index]["Filenames"])).name
            matrix = brain_mask_affine(
                prepared_inputs[filename]["brain_mask"],
                atlas_mask,
            )
        else:
            filename = Path(str(records_by_session[session_index]["Filenames"])).name
            matrix = orientation_preserving_slice_to_atlas(
                matrix,
                prepared_inputs[filename]["brain_mask"],
                atlas_mask,
            )
        matrix, surface_fit = fit_surface_scale_translation(
            matrix,
            outline_snapshot[session_index],
            atlas_mask,
        )
        filename = Path(str(records_by_session[session_index]["Filenames"])).name
        affine_transform = SliceAtlasTransform2D(
            matrix,
            prepared_inputs[filename]["image"].shape,
            atlas_mask.shape,
        )
        affine_transform.check_invariants()
        diagnostics = {
            "raw_model_ap_index": float(raw_atlas_index),
            "raw_model_pose_ap_um_lr_deg_dv_deg": list(
                records_by_session[session_index]["pose_ap_um_lr_deg_dv_deg"]
            ),
            "component_predictions": records_by_session[session_index]["component_predictions"],
            "prediction_fusion": records_by_session[session_index]["fusion"],
            "model_uncertainty": records_by_session[session_index].get("model_uncertainty", {}),
            "initial_orientation_inverted": bool(
                records_by_session[session_index].get("initial_orientation_inverted", False)
            ),
            "refined_ap_index": float(atlas_index),
            "ap_search_shift_index": float(atlas_index - raw_atlas_index),
            "ap_search_bounds_index": None if ap_bounds is None else list(ap_bounds),
            "order_constraint_applied": session_index in order_snapshot,
            "alignment_batch_session_indices": sorted(filename_to_session.values()),
            "order_constraint_session_indices": list(order_snapshot),
            "runtime_backend": runtime_info.get("backend", "unknown"),
            "coordinate_contract": runtime_info["coordinate_contract"],
            "runtime_device": runtime_info.get("device", "unknown"),
            "alignment_run_id": runtime_info.get("alignment_run_id"),
            "alignment_scope": "global" if global_alignment else "single",
            "onnxruntime_version": runtime_info.get("onnxruntime_version"),
            "gpu_fallback_reason": runtime_info.get("gpu_fallback_reason"),
            "inference_seconds": runtime_info.get("inference_seconds"),
            "total_backend_seconds": runtime_info.get("total_backend_seconds"),
            "preintegration_tilt_spread_deg": runtime_info.get(
                "preintegration_tilt_spread_deg",
                [0.0, 0.0],
            ),
            "input_crop": input_crops.get(Path(str(records_by_session[session_index]["Filenames"])).name),
            **search_diagnostics[session_index],
            **{f"surface_{key}": value for key, value in surface_fit.items()},
        }
        if not prepared:
            diagnostics["alignment_batch_inputs"] = input_crops
        prepared.append(
            (
                session_index,
                atlas_index,
                float(tilt_ml),
                float(tilt_dv),
                affine_transform,
                records_by_session[session_index],
                diagnostics,
            )
        )
    return prepared, shared_tilt


def prepare_run_and_solve_alignment(
    image_jobs: list[tuple],
    filename_to_session: dict[str, int],
    bregma_ap_index: float,
    atlas_volume: np.ndarray,
    annotation_volume: np.ndarray,
    outline_snapshot: dict[int, list[tuple[float, float]]],
    ap_bounds: tuple[int, int] | None,
    order_snapshot: list[int],
    alignment_run_id: str,
    global_alignment: bool,
    engine: str,
    own_cnn_weight: float,
    progress_messages: queue.SimpleQueue,
    cancel_event: threading.Event,
    probe_constraints: dict[str, dict] | None = None,
    bregma_voxel: np.ndarray | None = None,
    cortical_region_ids: frozenset[int] | None = None,
) -> tuple[str, dict, dict[str, dict[str, float]], dict, list[tuple], tuple[float, float] | None]:
    alignment_started = time.perf_counter()
    records, disagreement, runtime_info, prepared_inputs = prepare_and_run_pose_predictions(
        image_jobs,
        engine,
        own_cnn_weight,
        bregma_ap_index,
        progress_messages,
        cancel_event,
    )
    runtime_info["alignment_run_id"] = alignment_run_id
    if cancel_event.is_set():
        raise InterruptedError
    progress_messages.put((30, "Searching corresponding Allen atlas anatomy..."))
    solver_started = time.perf_counter()
    prepared, shared_tilt = solve_pose_alignment(
        records,
        filename_to_session,
        annotation_volume,
        outline_snapshot,
        ap_bounds,
        order_snapshot,
        runtime_info,
        cancel_event,
        global_alignment=global_alignment,
        atlas_volume=atlas_volume,
        prepared_inputs=prepared_inputs,
        disagreement=disagreement,
        progress_messages=progress_messages,
        probe_constraints=probe_constraints,
        bregma_voxel=bregma_voxel,
        cortical_region_ids=cortical_region_ids,
    )
    runtime_info["alignment_solver_seconds"] = float(time.perf_counter() - solver_started)
    for *_, diagnostics in prepared:
        diagnostics["nonlinear_refinement"] = {
            "requested": False,
            "status": "not-run",
            "reason": "Use Fit current slice to atlas after reviewing the affine alignment",
        }
    runtime_info["total_alignment_seconds"] = float(time.perf_counter() - alignment_started)
    for *_, diagnostics in prepared:
        diagnostics["alignment_solver_seconds"] = runtime_info["alignment_solver_seconds"]
        diagnostics["total_alignment_seconds"] = runtime_info["total_alignment_seconds"]
    if cancel_event.is_set():
        raise InterruptedError
    progress_messages.put((100, "Alignment ready"))
    return engine, runtime_info["component_provenance"], disagreement, runtime_info, prepared, shared_tilt


def fit_slice_anatomy_to_atlas(
    image_job: tuple,
    affine_transform: SliceAtlasTransform2D,
    atlas_index: int,
    tilt_ml: float,
    tilt_dv: float,
    atlas_volume: np.ndarray,
    annotation_volume: np.ndarray,
    nonlinear_model_path: str | None,
    progress_messages: queue.SimpleQueue,
    cancel_event: threading.Event,
) -> tuple[SliceAtlasTransform2D, dict]:
    if affine_transform.nonlinear is not None:
        raise ValueError("This slice already has a nonlinear anatomical fit")
    progress_messages.put((5, "Preparing the aligned histology slice..."))
    with tempfile.TemporaryDirectory(prefix="trajectory_nonlinear_") as temporary_folder:
        image_paths, input_crops, prepared_inputs = prepare_pose_inputs(
            [image_job], temporary_folder, progress_messages, cancel_event
        )
        filename = Path(image_paths[0]).name
        display_image = prepared_inputs[filename]["image"]
        display_mask = prepared_inputs[filename]["brain_mask"].astype(np.uint8)
        fixed_atlas = coronal_oblique_slice(
            atlas_volume, atlas_index, tilt_ml, tilt_dv, order=1
        )
        fixed_mask = (
            coronal_oblique_slice(
                annotation_volume, atlas_index, tilt_ml, tilt_dv, order=0
            )
            > 0
        )
        moving_affine = affine_transform.render_display_image_in_atlas(display_image)
        moving_mask = (
            affine_transform.render_display_image_in_atlas(
                display_mask, interpolation=cv2.INTER_NEAREST
            )
            > 0
        )
        progress_messages.put((35, "Fitting internal anatomy on the fixed atlas plane..."))
        source_sha256 = input_crops[filename]["source_image_sha256"]
        try:
            if nonlinear_model_path is None:
                warp, accepted = run_classical_diffeomorphic_registration(
                    fixed_atlas,
                    moving_affine,
                    fixed_mask,
                    moving_mask,
                    pixel_spacing_um=VOXEL_UM,
                    source_image_sha256=source_sha256,
                    progress_messages=progress_messages,
                    cancel_event=cancel_event,
                )
            else:
                warp, accepted = run_diffeomorphic_registration(
                    fixed_atlas,
                    moving_affine,
                    fixed_mask,
                    moving_mask,
                    nonlinear_model_path,
                    pixel_spacing_um=VOXEL_UM,
                    source_image_sha256=source_sha256,
                )
        except DiffeomorphicRegistrationRejected as exc:
            return affine_transform, {
                "requested": True,
                "status": "rejected",
                "reason": "; ".join(exc.failures),
                "rejection_categories": list(exc.categories),
                "mapping_blocking": bool(
                    {"wrong_plane", "affine_input"}.intersection(exc.categories)
                ),
                "runtime": exc.diagnostics,
            }
        if cancel_event.is_set():
            raise InterruptedError
        attestation = NonlinearWarpAttestation.from_runtime(
            warp, fixed_mask, moving_mask, accepted
        )
        transform = SliceAtlasTransform2D(
            affine_transform.display_to_affine_atlas_h,
            affine_transform.display_shape,
            affine_transform.atlas_shape,
            warp,
            attestation,
        )
        transform.check_invariants()
        progress_messages.put((100, "Anatomical fit ready"))
        return transform, {
            "requested": True,
            "status": "accepted",
            "reason": None,
            "rejection_categories": [],
            "mapping_blocking": False,
            "runtime": accepted,
        }


def atlas_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "coronal":
        return volume[index, :, :]
    if plane == "horizontal":
        return volume[:, index, :]
    return volume[:, :, index]


def coronal_oblique_slice(
    volume: np.ndarray,
    index: int,
    tilt_ml_deg: float,
    tilt_dv_deg: float,
    *,
    order: int,
) -> np.ndarray:
    dv_size, ml_size = volume.shape[1:]
    dv, ml = np.mgrid[0:dv_size, 0:ml_size].astype(np.float64)
    ap = (
        float(index)
        + np.tan(np.deg2rad(tilt_ml_deg)) * (ml - (ml_size - 1) / 2.0)
        + np.tan(np.deg2rad(tilt_dv_deg)) * (dv - (dv_size - 1) / 2.0)
    )
    return map_coordinates(
        volume,
        [ap, dv, ml],
        order=order,
        mode="constant",
        cval=0,
        prefilter=False,
    )


def coronal_oblique_slice_resampled(
    volume: np.ndarray,
    index: int,
    tilt_ml_deg: float,
    tilt_dv_deg: float,
    *,
    order: int,
) -> np.ndarray:
    width, height = MIND_CANONICAL_SIZE
    dv = np.linspace(0.0, volume.shape[1] - 1.0, height)
    ml = np.linspace(0.0, volume.shape[2] - 1.0, width)
    dv, ml = np.meshgrid(dv, ml, indexing="ij")
    ap = (
        float(index)
        + np.tan(np.deg2rad(tilt_ml_deg)) * (ml - (volume.shape[2] - 1) / 2.0)
        + np.tan(np.deg2rad(tilt_dv_deg)) * (dv - (volume.shape[1] - 1) / 2.0)
    )
    return map_coordinates(
        volume,
        [ap, dv, ml],
        order=order,
        mode="constant",
        cval=0,
        prefilter=False,
    )


def plane_axis(plane: str) -> int:
    return {"coronal": 0, "horizontal": 1, "sagittal": 2}[plane]


def plane_axis_name(plane: str) -> str:
    return {"coronal": "AP", "horizontal": "DV", "sagittal": "ML"}[plane]


def volume_to_stereotaxic_um(coord: np.ndarray, bregma_voxel: np.ndarray) -> np.ndarray:
    return (np.asarray(coord, dtype=np.float64) - bregma_voxel.astype(np.float64)) * VOXEL_UM * STEREOTAXIC_AXIS_SIGN_AP_DV_ML


def point_to_volume(
    point: tuple[float, float],
    plane: str,
    index: int,
    volume_shape: tuple[int, int, int] | None = None,
    tilt_ml_deg: float = 0.0,
    tilt_dv_deg: float = 0.0,
) -> np.ndarray:
    x, y = point
    if plane == "coronal":
        ap = float(index)
        if volume_shape is not None:
            ap += np.tan(np.deg2rad(tilt_ml_deg)) * (x - (volume_shape[2] - 1) / 2.0)
            ap += np.tan(np.deg2rad(tilt_dv_deg)) * (y - (volume_shape[1] - 1) / 2.0)
        return np.array([ap, y, x], dtype=np.float64)
    if plane == "horizontal":
        return np.array([y, index, x], dtype=np.float64)
    return np.array([y, x, index], dtype=np.float64)


def section_plane_corners(
    shape: tuple[int, int, int],
    plane: str,
    index: int,
    tilt_ml_deg: float = 0.0,
    tilt_dv_deg: float = 0.0,
) -> np.ndarray:
    ap_max, dv_max, ml_max = (float(size - 1) for size in shape)
    if plane == "coronal":
        corners = [(0.0, 0.0), (0.0, ml_max), (dv_max, ml_max), (dv_max, 0.0)]
        return np.asarray(
            [
                point_to_volume(
                    (ml, dv),
                    plane,
                    index,
                    shape,
                    tilt_ml_deg,
                    tilt_dv_deg,
                )
                for dv, ml in corners
            ],
            dtype=np.float32,
        )
    if plane == "horizontal":
        return np.asarray(
            [[0.0, index, 0.0], [0.0, index, ml_max], [ap_max, index, ml_max], [ap_max, index, 0.0]],
            dtype=np.float32,
        )
    return np.asarray(
        [[0.0, 0.0, index], [0.0, dv_max, index], [ap_max, dv_max, index], [ap_max, 0.0, index]],
        dtype=np.float32,
    )


def volume_to_gl(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    mesh_ap = ALLEN_CCF_25_SHAPE_AP_DV_ML[0] - points[:, 0]
    mesh_dv = ALLEN_CCF_25_SHAPE_AP_DV_ML[1] - points[:, 1]
    return np.column_stack([points[:, 2], mesh_ap, mesh_dv])


def plane_label_paths(corners: np.ndarray, text: str) -> list[np.ndarray]:
    path = QtGui.QPainterPath()
    font = QtGui.QFont("Sans Serif")
    font.setPixelSize(24)
    path.addText(0.0, 0.0, font, text)
    bounds = path.boundingRect()
    if bounds.width() <= 0 or bounds.height() <= 0:
        return []

    corners = np.asarray(corners, dtype=np.float32)
    across = corners[1] - corners[0]
    down = corners[3] - corners[0]
    across_length = float(np.linalg.norm(across))
    down_length = float(np.linalg.norm(down))
    if across_length == 0 or down_length == 0:
        return []
    across_unit = across / across_length
    down_unit = down / down_length
    normal = np.cross(across_unit, down_unit)
    normal_length = float(np.linalg.norm(normal))
    if normal_length:
        normal /= normal_length

    label_height = min(16.0, down_length * 0.06)
    label_width = min(across_length * 0.32, label_height * bounds.width() / bounds.height())
    origin = corners[0] + across * 0.035 + down * 0.045 + normal * 1.2
    paths: list[np.ndarray] = []
    for polygon in path.toSubpathPolygons(QtGui.QTransform()):
        points = np.asarray([(point.x(), point.y()) for point in polygon], dtype=np.float32)
        if len(points) < 2:
            continue
        x = (points[:, 0] - bounds.left()) / bounds.width()
        y = (points[:, 1] - bounds.top()) / bounds.height()
        paths.append(origin + x[:, None] * label_width * across_unit + y[:, None] * label_height * down_unit)
    return paths


@dataclass
class ProbeTrace:
    atlas_points: list[tuple[float, float]] = field(default_factory=list)
    slice_points: list[tuple[float, float]] = field(default_factory=list)
    volume_points: list[list[float]] = field(default_factory=list)
    signal_values: list[float] = field(default_factory=list)


@dataclass
class SliceSession:
    name: str
    path: str = ""
    display_scale: float = 1.0
    raw_display: np.ndarray | None = None
    adjusted: np.ndarray | None = None
    rotated: np.ndarray | None = None
    weight_image: np.ndarray | None = None
    rotation_deg: float = 0.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    slice_transform: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    curve_points: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (255.0, 255.0)])
    atlas_plane: str = "coronal"
    atlas_index: int = 0
    atlas_tilt_ml_deg: float = 0.0
    atlas_tilt_dv_deg: float = 0.0
    atlas_landmarks: list[tuple[float, float]] = field(default_factory=list)
    slice_landmarks: list[tuple[float, float]] = field(default_factory=list)
    brain_outline_points: list[tuple[float, float]] = field(default_factory=list)
    brain_outline_segment_starts: list[int] = field(default_factory=lambda: [0])
    brain_outline_closed: bool = False
    brain_brush_strokes: list[tuple[bool, list[tuple[float, float]]]] = field(default_factory=list)
    brain_brush_selection_mask: np.ndarray | None = None
    brain_outline_undo_stack: list[tuple] = field(default_factory=list, repr=False)
    probe_traces: dict[str, ProbeTrace] = field(default_factory=dict)
    point_history: list[str] = field(default_factory=list)
    auto_alignment_score: float | None = None
    auto_alignment_global: bool = False
    auto_alignment_extent: str | None = None
    auto_alignment_method: str | None = None
    auto_alignment_engine: str | None = None
    auto_alignment_scope: str | None = None
    auto_alignment_run_id: str | None = None
    manual_refined_from_run_id: str | None = None
    auto_alignment_diagnostics: dict | None = None
    alignment_source_sha256: str | None = None
    deepslice_raw_ensemble_ouv: list[float] | None = None
    deepslice_version: str | None = None
    deepslice_model_hashes: dict[str, str] | None = None
    deepslice_ensemble_disagreement: dict[str, float] | None = None
    transformed_overlay: np.ndarray | None = None
    slice_atlas_transform: SliceAtlasTransform2D | None = None
    slice_to_atlas_x: Rbf | None = None
    slice_to_atlas_y: Rbf | None = None
    atlas_to_slice_x: Rbf | None = None
    atlas_to_slice_y: Rbf | None = None


def map_session_display_to_atlas(session: SliceSession, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if session.slice_atlas_transform is not None:
        return session.slice_atlas_transform.map_display_to_atlas(points)
    if session.slice_to_atlas_x is None or session.slice_to_atlas_y is None:
        raise ValueError("Slice-to-atlas transform is unavailable")
    return np.column_stack(
        (
            session.slice_to_atlas_x(points[:, 0], points[:, 1]),
            session.slice_to_atlas_y(points[:, 0], points[:, 1]),
        )
    )


def map_session_atlas_to_display(session: SliceSession, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if session.slice_atlas_transform is not None:
        return session.slice_atlas_transform.map_atlas_to_display(points)
    if session.atlas_to_slice_x is None or session.atlas_to_slice_y is None:
        raise ValueError("Atlas-to-slice transform is unavailable")
    return np.column_stack(
        (
            session.atlas_to_slice_x(points[:, 0], points[:, 1]),
            session.atlas_to_slice_y(points[:, 0], points[:, 1]),
        )
    )


def render_session_slice_in_atlas(
    session: SliceSession,
    display_image: np.ndarray,
    atlas_shape: tuple[int, int],
) -> np.ndarray:
    if session.slice_atlas_transform is not None:
        if session.slice_atlas_transform.atlas_shape != tuple(atlas_shape):
            raise ValueError("Stored slice transform does not match the active atlas canvas")
        return session.slice_atlas_transform.render_display_image_in_atlas(display_image)
    if session.atlas_to_slice_x is None or session.atlas_to_slice_y is None:
        raise ValueError("Atlas-to-slice transform is unavailable")
    yy, xx = np.mgrid[: atlas_shape[0], : atlas_shape[1]].astype(np.float32)
    return cv2.remap(
        display_image,
        session.atlas_to_slice_x(xx, yy).astype(np.float32),
        session.atlas_to_slice_y(xx, yy).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


class ImagePanel(QtWidgets.QWidget):
    clicked = QtCore.Signal(float, float)
    brush_stroke = QtCore.Signal(list, bool)
    outline_drag_started = QtCore.Signal(int)
    outline_point_moved = QtCore.Signal(int, float, float)
    outline_point_deleted = QtCore.Signal(int)

    def __init__(self, title: str) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = QtWidgets.QLabel(title)
        self.title.setStyleSheet("font-weight:600;")
        layout.addWidget(self.title)

        self.widget = pg.GraphicsLayoutWidget()
        self.widget.setBackground("#05070a")
        layout.addWidget(self.widget, 1)
        self.view = self.widget.addViewBox(lockAspect=True)
        self.view.invertY(True)
        self.base_item = pg.ImageItem(axisOrder="row-major")
        self.overlay_item = pg.ImageItem(axisOrder="row-major")
        self.overlay_item.setZValue(5)
        self.overlay_item.hide()
        self.selection_item = pg.ImageItem(axisOrder="row-major")
        self.selection_item.setZValue(12)
        self.selection_item.hide()
        self.landmark_item = pg.ScatterPlotItem(size=10, brush=pg.mkBrush("#ffe66d"), pen=pg.mkPen("#111820", width=1))
        self.probe_item = pg.ScatterPlotItem(
            size=9,
            brush=pg.mkBrush(*PROBE_COLORS[0]),
            pen=pg.mkPen("#ffffff", width=1),
        )
        self.outline_item = pg.PlotDataItem(
            pen=pg.mkPen("#5de4ff", width=2),
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush("#5de4ff"),
            symbolPen=pg.mkPen("#052c36"),
        )
        self.landmark_item.setZValue(20)
        self.probe_item.setZValue(25)
        self.outline_item.setZValue(18)
        self.selected_outline_item = pg.ScatterPlotItem(
            size=14,
            brush=pg.mkBrush("#ffcc4d"),
            pen=pg.mkPen("#ffffff", width=2),
        )
        self.selected_outline_item.setZValue(21)
        self.brush_item = pg.PlotDataItem(pen=pg.mkPen("#53ffae", width=7))
        self.brush_item.setZValue(22)
        self.brush_cursor_item = QtWidgets.QGraphicsEllipseItem()
        brush_cursor_pen = pg.mkPen("#d9fff0", width=2)
        brush_cursor_pen.setCosmetic(True)
        self.brush_cursor_item.setPen(brush_cursor_pen)
        self.brush_cursor_item.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        self.brush_cursor_item.setZValue(23)
        self.brush_cursor_item.hide()
        self.view.addItem(self.base_item)
        self.view.addItem(self.overlay_item)
        self.view.addItem(self.selection_item)
        self.view.addItem(self.landmark_item)
        self.view.addItem(self.probe_item)
        self.view.addItem(self.outline_item)
        self.view.addItem(self.selected_outline_item)
        self.view.addItem(self.brush_item)
        self.view.addItem(self.brush_cursor_item)
        self.labels: list[pg.TextItem] = []
        self.image_shape: tuple[int, int] | None = None
        self.brush_enabled = False
        self.brush_erase_only = False
        self.outline_editable = False
        self.outline_points: list[tuple[float, float]] = []
        self.selected_outline_index: int | None = None
        self.active_outline_drag_index: int | None = None
        self.outline_drag_snapshot_started = False
        self.brush_radius = 180.0
        self.active_brush_points: list[tuple[float, float]] = []
        self.active_brush_exclude = False
        self.widget.viewport().installEventFilter(self)
        self.widget.scene().sigMouseClicked.connect(self._mouse_clicked)

    def set_base(self, image: np.ndarray | None) -> None:
        if image is None:
            self.base_item.clear()
            self.overlay_item.hide()
            self.selection_item.hide()
            self.image_shape = None
            return
        previous_shape = self.image_shape
        self.image_shape = image.shape[:2]
        self.base_item.setImage(image, autoLevels=False, levels=(0, 255))
        self.base_item.setRect(QtCore.QRectF(0, 0, image.shape[1], image.shape[0]))
        if previous_shape != self.image_shape:
            self.view.autoRange()

    def set_overlay(self, image: np.ndarray | None, opacity: float = 0.55) -> None:
        if image is None:
            self.overlay_item.hide()
            return
        rgba = red_rgba(image) if image.ndim == 2 else image
        self.overlay_item.setImage(rgba, autoLevels=False)
        self.overlay_item.setRect(QtCore.QRectF(0, 0, rgba.shape[1], rgba.shape[0]))
        self.overlay_item.setOpacity(opacity)
        self.overlay_item.show()

    def set_overlay_opacity(self, opacity: float) -> None:
        self.overlay_item.setOpacity(opacity)

    def set_selection_mask(self, mask: np.ndarray | None) -> None:
        if mask is None or self.image_shape is None:
            self.selection_item.hide()
            return
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask > 0] = (45, 135, 220, 42)
        height, width = self.image_shape
        self.selection_item.setImage(rgba, autoLevels=False)
        self.selection_item.setRect(QtCore.QRectF(0, 0, width, height))
        self.selection_item.show()

    def set_brush_enabled(self, enabled: bool) -> None:
        self.brush_enabled = bool(enabled)
        self.widget.viewport().setCursor(
            QtCore.Qt.CursorShape.CrossCursor if enabled else QtCore.Qt.CursorShape.ArrowCursor
        )
        if not enabled:
            self.active_brush_points.clear()
            self.brush_item.setData([], [])
            self.brush_cursor_item.hide()

    def set_brush_erase_only(self, enabled: bool) -> None:
        self.brush_erase_only = bool(enabled)

    def set_outline_editable(self, enabled: bool) -> None:
        self.outline_editable = bool(enabled)
        if not enabled:
            self.active_outline_drag_index = None
            self.clear_outline_selection()

    def clear_outline_selection(self) -> None:
        self.selected_outline_index = None
        self.selected_outline_item.setData([])

    def delete_selected_outline_point(self) -> None:
        if not self.outline_editable or self.selected_outline_index is None:
            return
        index = self.selected_outline_index
        self.clear_outline_selection()
        self.outline_point_deleted.emit(index)

    def set_brush_radius(self, radius: float) -> None:
        self.brush_radius = float(radius)

    def _show_brush_cursor(self, point: tuple[float, float] | None) -> None:
        if point is None:
            self.brush_cursor_item.hide()
            return
        radius = self.brush_radius
        self.brush_cursor_item.setRect(point[0] - radius, point[1] - radius, 2 * radius, 2 * radius)
        self.brush_cursor_item.show()

    def set_points(
        self,
        landmarks: list[tuple[float, float]],
        probes: list[tuple[tuple[float, float], tuple[int, int, int], bool]],
    ) -> None:
        self.landmark_item.setData([{"pos": point} for point in landmarks])
        self.probe_item.setData(
            [
                {
                    "pos": point,
                    "size": 10 if selected else 7,
                    "brush": pg.mkBrush(*color, 255 if selected else 150),
                    "pen": pg.mkPen("#ffffff" if selected else "#1d2733", width=1),
                }
                for point, color, selected in probes
            ]
        )
        for label in self.labels:
            self.view.removeItem(label)
        self.labels.clear()
        for i, (x, y) in enumerate(landmarks, start=1):
            label = pg.TextItem(
                str(i),
                color="#fff4a3",
                anchor=(-0.25, 1.1),
            )
            font = QtGui.QFont()
            font.setPointSize(11)
            font.setBold(True)
            label.setFont(font)
            label.setZValue(30)
            label.setPos(x, y)
            self.view.addItem(label)
            self.labels.append(label)

    def set_outline(
        self,
        points: list[tuple[float, float]],
        segment_starts: list[int] | None = None,
        closed: bool = False,
    ) -> None:
        self.outline_points = list(points)
        if self.selected_outline_index is not None and self.selected_outline_index >= len(points):
            self.clear_outline_selection()
        starts = sorted(set(segment_starts or [0]))
        starts = [start for start in starts if 0 <= start < len(points)]
        starts = starts or ([0] if points else [])
        display_points: list[tuple[float, float]] = []
        for segment_index, start in enumerate(starts):
            end = starts[segment_index + 1] if segment_index + 1 < len(starts) else len(points)
            display_points.extend(points[start:end])
            if closed and len(starts) == 1 and end > start:
                display_points.append(points[start])
            if end < len(points):
                display_points.append((np.nan, np.nan))
        self.outline_item.setData(
            [point[0] for point in display_points],
            [point[1] for point in display_points],
        )
        if self.selected_outline_index is not None:
            self.selected_outline_item.setData([{"pos": points[self.selected_outline_index]}])

    def _nearest_outline_point(self, point: tuple[float, float]) -> int | None:
        if not self.outline_points:
            return None
        distances = np.linalg.norm(np.asarray(self.outline_points) - np.asarray(point), axis=1)
        pixel_width, pixel_height = self.view.viewPixelSize()
        index = int(np.argmin(distances))
        return index if distances[index] <= 12.0 * max(pixel_width, pixel_height) else None

    def _mouse_clicked(self, event: QtGui.QMouseEvent) -> None:
        if self.brush_enabled or event.button() != QtCore.Qt.MouseButton.LeftButton or self.image_shape is None:
            return
        point = self.view.mapSceneToView(event.scenePos())
        x = float(point.x())
        y = float(point.y())
        h, w = self.image_shape
        if 0 <= x < w and 0 <= y < h:
            self.clicked.emit(x, y)

    def _brush_point(self, event: QtGui.QMouseEvent) -> tuple[float, float] | None:
        if self.image_shape is None:
            return None
        scene_point = self.widget.mapToScene(event.position().toPoint())
        point = self.view.mapSceneToView(scene_point)
        x, y = float(point.x()), float(point.y())
        height, width = self.image_shape
        if 0 <= x < width and 0 <= y < height:
            return x, y
        return None

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.widget.viewport() and self.brush_enabled:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
                point = self._brush_point(event)
                self._show_brush_cursor(point)
                if point is not None:
                    self.active_brush_points = [point]
                    self.active_brush_exclude = self.brush_erase_only or bool(
                        event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
                    )
                    color = "#ff6b7a" if self.active_brush_exclude else "#53ffae"
                    self.brush_item.setPen(pg.mkPen(color, width=7))
                    self.brush_item.setData([point[0]], [point[1]])
                    return True
            elif event.type() == QtCore.QEvent.Type.MouseMove and self.active_brush_points:
                point = self._brush_point(event)
                self._show_brush_cursor(point)
                if point is not None:
                    previous = np.asarray(self.active_brush_points[-1])
                    if np.linalg.norm(np.asarray(point) - previous) >= max(1.0, self.brush_radius * 0.15):
                        self.active_brush_points.append(point)
                        self.brush_item.setData(
                            [value[0] for value in self.active_brush_points],
                            [value[1] for value in self.active_brush_points],
                        )
                    return True
            elif event.type() == QtCore.QEvent.Type.MouseMove:
                self._show_brush_cursor(self._brush_point(event))
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease and self.active_brush_points:
                point = self._brush_point(event)
                self._show_brush_cursor(point)
                if point is not None:
                    self.active_brush_points.append(point)
                stroke = self.active_brush_points.copy()
                exclude = self.active_brush_exclude
                self.active_brush_points.clear()
                self.brush_item.setData([], [])
                self.brush_stroke.emit(stroke, exclude)
                return True
        if watched is self.widget.viewport() and self.outline_editable:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
                point = self._brush_point(event)
                index = None if point is None else self._nearest_outline_point(point)
                if index is None:
                    self.clear_outline_selection()
                    return False
                self.selected_outline_index = index
                self.active_outline_drag_index = index
                self.outline_drag_snapshot_started = False
                self.selected_outline_item.setData([{"pos": self.outline_points[index]}])
                return True
            if event.type() == QtCore.QEvent.Type.MouseMove and self.active_outline_drag_index is not None:
                point = self._brush_point(event)
                if point is not None:
                    index = self.active_outline_drag_index
                    if not self.outline_drag_snapshot_started:
                        self.outline_drag_started.emit(index)
                        self.outline_drag_snapshot_started = True
                    self.outline_points[index] = point
                    self.selected_outline_item.setData([{"pos": point}])
                    self.outline_point_moved.emit(index, point[0], point[1])
                return True
            if event.type() == QtCore.QEvent.Type.MouseButtonRelease and self.active_outline_drag_index is not None:
                point = self._brush_point(event)
                index = self.active_outline_drag_index
                self.active_outline_drag_index = None
                if point is not None and self.outline_drag_snapshot_started:
                    self.outline_point_moved.emit(index, point[0], point[1])
                self.outline_drag_snapshot_started = False
                return True
        return super().eventFilter(watched, event)


class CurveCanvas(QtWidgets.QWidget):
    points_changed = QtCore.Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(100)
        self.setMouseTracking(True)
        self.points: list[tuple[float, float]] = [(0.0, 0.0), (255.0, 255.0)]
        self.hist: np.ndarray | None = None
        self.drag_index: int | None = None

    def set_histogram(self, image_u8: np.ndarray | None) -> None:
        if image_u8 is None:
            self.hist = None
        else:
            hist, _ = np.histogram(image_u8.ravel(), bins=128, range=(0, 256))
            hist = hist.astype(np.float32)
            hist = cv2.GaussianBlur(hist[None, :], (0, 0), 1.5).ravel()
            hist = np.sqrt(hist)
            self.hist = hist / hist.max() if hist.max() > 0 else hist
        self.update()

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self.points = [(float(x), float(y)) for x, y in points]
        self.update()

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#0f131a"))
        graph = self._graph_rect()
        painter.setPen(QtGui.QPen(QtGui.QColor("#2d3a4c"), 1))
        painter.drawRect(graph)
        painter.setPen(QtGui.QPen(QtGui.QColor("#263342"), 1, QtCore.Qt.PenStyle.DotLine))
        for fraction in (0.25, 0.5, 0.75):
            x = graph.left() + fraction * graph.width()
            y = graph.top() + fraction * graph.height()
            painter.drawLine(QtCore.QPointF(x, graph.top()), QtCore.QPointF(x, graph.bottom()))
            painter.drawLine(QtCore.QPointF(graph.left(), y), QtCore.QPointF(graph.right(), y))
        if self.hist is not None:
            histogram = QtGui.QPainterPath(QtCore.QPointF(graph.left(), graph.bottom()))
            for i, value in enumerate(self.hist):
                x = graph.left() + i / max(1, len(self.hist) - 1) * graph.width()
                y = graph.bottom() - float(value) * graph.height()
                histogram.lineTo(x, y)
            histogram.lineTo(graph.right(), graph.bottom())
            histogram.closeSubpath()
            painter.setPen(QtGui.QPen(QtGui.QColor("#6f8da6"), 1.5))
            painter.setBrush(QtGui.QColor(76, 111, 137, 95))
            painter.drawPath(histogram)
        sorted_points = sorted(self.points)
        polyline = QtGui.QPolygonF([self._data_to_pos(x, y) for x, y in sorted_points])
        painter.setPen(QtGui.QPen(QtGui.QColor("#49b9ff"), 2))
        painter.drawPolyline(polyline)
        painter.setBrush(QtGui.QColor("#49b9ff"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
        for x, y in self.points:
            pos = self._data_to_pos(x, y)
            painter.drawEllipse(pos, 5, 5)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        pos = event.position()
        distances = [QtCore.QLineF(pos, self._data_to_pos(x, y)).length() for x, y in self.points]
        if event.button() == QtCore.Qt.MouseButton.RightButton and distances and min(distances) <= 12:
            index = int(np.argmin(distances))
            if len(self.points) > 2 and index not in (0, len(self.points) - 1):
                self.points.pop(index)
                self.points = sorted(self.points)
                self.update()
                self.points_changed.emit(self.points)
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if distances and min(distances) <= 12:
            self.drag_index = int(np.argmin(distances))

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton or len(self.points) >= 12:
            return
        if any(QtCore.QLineF(event.position(), self._data_to_pos(x, y)).length() <= 12 for x, y in self.points):
            return
        self.points.append(self._pos_to_data(event.position()))
        self.points = sorted(self.points)
        self.update()
        self.points_changed.emit(self.points)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.drag_index is None:
            return
        x, y = self._pos_to_data(event.position())
        self.points[self.drag_index] = (x, y)
        self.update()
        self.points_changed.emit(sorted(self.points))

    def mouseReleaseEvent(self, _: QtGui.QMouseEvent) -> None:
        if self.drag_index is None:
            return
        self.drag_index = None
        self.points = sorted(self.points)
        self.points_changed.emit(self.points)

    def _graph_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(28, 8, max(1, self.width() - 38), max(1, self.height() - 32))

    def _data_to_pos(self, x: float, y: float) -> QtCore.QPointF:
        graph = self._graph_rect()
        return QtCore.QPointF(
            graph.left() + np.clip(x, 0, 255) / 255.0 * graph.width(),
            graph.bottom() - np.clip(y, 0, 255) / 255.0 * graph.height(),
        )

    def _pos_to_data(self, pos: QtCore.QPointF) -> tuple[float, float]:
        graph = self._graph_rect()
        x = (pos.x() - graph.left()) / graph.width() * 255.0
        y = (graph.bottom() - pos.y()) / graph.height() * 255.0
        return float(np.clip(x, 0, 255)), float(np.clip(y, 0, 255))


class CurveEditor(QtWidgets.QWidget):
    points_changed = QtCore.Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self._updating = False
        self.points: list[tuple[float, float]] = [(0.0, 0.0), (255.0, 255.0)]

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        row = QtWidgets.QHBoxLayout()
        self.point_count = QtWidgets.QSpinBox()
        self.point_count.setKeyboardTracking(False)
        self.point_count.setRange(2, 12)
        self.point_count.setValue(2)
        row.addWidget(QtWidgets.QLabel("Points"))
        row.addWidget(self.point_count)
        curve_help = QtWidgets.QLabel("Drag • double-click add • right-click remove")
        curve_help.setToolTip("Drag curve points; double-click to add a point; right-click an interior point to remove it")
        row.addWidget(curve_help)
        row.addStretch(1)
        self.reset_btn = QtWidgets.QPushButton("Reset linear")
        row.addWidget(self.reset_btn)
        layout.addLayout(row)

        self.canvas = CurveCanvas()

        self.table = QtWidgets.QTableWidget(2, 2)
        self.table.setHorizontalHeaderLabels(["Input", "Output"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumWidth(150)

        self.editor_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.editor_splitter.setChildrenCollapsible(False)
        self.editor_splitter.setHandleWidth(6)
        self.editor_splitter.addWidget(self.canvas)
        self.editor_splitter.addWidget(self.table)
        self.editor_splitter.setStretchFactor(0, 3)
        self.editor_splitter.setStretchFactor(1, 1)
        self.editor_splitter.setSizes([420, 180])
        layout.addWidget(self.editor_splitter, 1)

        self.point_count.valueChanged.connect(self._set_count)
        self.table.cellChanged.connect(self._table_changed)
        self.canvas.points_changed.connect(self._canvas_changed)
        self.reset_btn.clicked.connect(self._reset_linear)
        self.set_points(self.points)

    def set_histogram(self, image_u8: np.ndarray | None) -> None:
        self.canvas.set_histogram(image_u8)

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self._updating = True
        self.points = sorted((float(x), float(y)) for x, y in points)
        self.point_count.setValue(len(self.points))
        self.table.setRowCount(len(self.points))
        for row, (x, y) in enumerate(self.points):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{x:.1f}"))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{y:.1f}"))
        self._updating = False
        self._refresh_plot()

    def _set_count(self, count: int) -> None:
        if self._updating:
            return
        if count == len(self.points):
            return
        xs = np.linspace(0, 255, count)
        ys = np.interp(xs, [p[0] for p in self.points], [p[1] for p in self.points])
        self.set_points(list(zip(xs, ys)))
        self.points_changed.emit(self.points)

    def _table_changed(self, *_: object) -> None:
        if self._updating:
            return
        points: list[tuple[float, float]] = []
        for row in range(self.table.rowCount()):
            x_item = self.table.item(row, 0)
            y_item = self.table.item(row, 1)
            if x_item is None or y_item is None:
                continue
            x = float(x_item.text().replace(",", "."))
            y = float(y_item.text().replace(",", "."))
            points.append((np.clip(x, 0, 255), np.clip(y, 0, 255)))
        if len(points) >= 2:
            self.points = sorted(points)
            self._refresh_plot()
            self.points_changed.emit(self.points)

    def _canvas_changed(self, points: list[tuple[float, float]]) -> None:
        if self._updating:
            return
        self.set_points(points)
        self.points_changed.emit(self.points)

    def _reset_linear(self) -> None:
        self.set_points([(0.0, 0.0), (255.0, 255.0)])
        self.points_changed.emit(self.points)

    def _refresh_plot(self) -> None:
        self.canvas.set_points(self.points)


class TrajectoryTrackerWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        *,
        default_atlas_folder: str | Path = DEFAULT_ATLAS_FOLDER,
        default_slices_folder: str | Path = "",
        default_run_folder: str | Path = "",
    ) -> None:
        super().__init__()
        self.setWindowTitle("Proprietary neuropixels trajectory tracker")
        self.resize(1780, 980)
        self.atlas_folder = Path(default_atlas_folder)
        self.atlas_file_hashes: dict[str, str] = {}
        self.query_file_hash: str | None = None
        self.default_slices_folder = Path(default_slices_folder) if str(default_slices_folder).strip() else Path()
        self.atlas_volume: np.ndarray | None = None
        self.annotation_volume: np.ndarray | None = None
        self.bregma_voxel = DEFAULT_BREGMA_VOXEL_AP_DV_ML.copy()
        self.region_names: dict[int, tuple[str, str]] = {}
        self.cortical_region_ids: set[int] = set()
        self.current_atlas_image: np.ndarray | None = None
        self.sessions: list[SliceSession] = []
        self.current_session_index = -1
        self.probe_constraints: dict[str, ProbeInsertionConstraint] = {}
        self._loading_probe_constraints = False
        self.probe_endpoint_settings: dict[str, tuple[str | None, float]] = {}
        self._loading_probe_endpoint = False
        self.dynamic_gl_items: list[object] = []
        self.brain_mesh_item: gl.GLMeshItem | None = None
        self.auto_alignment_busy = False
        self.alignment_executor = ThreadPoolExecutor(max_workers=1)
        self._alignment_cancel_event: threading.Event | None = None
        self._alignment_timer: QtCore.QTimer | None = None
        self._alignment_progress: QtWidgets.QProgressDialog | None = None
        self._session_cache_dirs: list[tempfile.TemporaryDirectory] = []

        self._build_ui(default_run_folder)
        self._build_session_menu()
        if self.atlas_folder.exists():
            self.load_atlas_folder(self.atlas_folder)

    def _build_session_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        self.load_session_action = file_menu.addAction("Open session...")
        self.load_session_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.save_session_action = file_menu.addAction("Save session...")
        self.save_session_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.load_session_action.triggered.connect(self._load_session_dialog)
        self.save_session_action.triggered.connect(self._save_session_dialog)

    def _build_ui(self, default_run_folder: str | Path) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)

        self.workspace_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setHandleWidth(8)
        layout.addWidget(self.workspace_splitter, 1)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QGridLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setHorizontalSpacing(6)
        controls_layout.setVerticalSpacing(6)

        def resizable_panel(widget: QtWidgets.QWidget, minimum_width: int) -> QtWidgets.QScrollArea:
            panel = QtWidgets.QScrollArea()
            panel.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            panel.setWidgetResizable(True)
            panel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            panel.setMinimumWidth(minimum_width)
            panel.setWidget(widget)
            return panel

        self.controls_scroll = QtWidgets.QScrollArea()
        self.controls_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.controls_scroll.setMinimumHeight(130)
        self.controls_scroll.setWidget(controls)
        self.workspace_splitter.addWidget(self.controls_scroll)

        self.atlas_path = QtWidgets.QLineEdit(str(self.atlas_folder))
        self.load_atlas_btn = QtWidgets.QPushButton("Load atlas")
        self.browse_atlas_btn = QtWidgets.QPushButton("Browse")
        atlas_setup = QtWidgets.QGroupBox("Atlas")
        atlas_setup_layout = QtWidgets.QGridLayout(atlas_setup)
        atlas_setup_layout.addWidget(self.atlas_path, 0, 0, 1, 3)
        atlas_setup_layout.addWidget(self.browse_atlas_btn, 0, 3)
        atlas_setup_layout.addWidget(self.load_atlas_btn, 0, 4)

        self.plane_box = QtWidgets.QComboBox()
        self.plane_box.addItems(["coronal", "sagittal", "horizontal"])
        self.axis_label = QtWidgets.QLabel("AP position")
        self.axis_position_um = QtWidgets.QSpinBox()
        self.axis_position_um.setKeyboardTracking(False)
        self.axis_position_um.setRange(-999999, 999999)
        self.axis_position_um.setSingleStep(int(VOXEL_UM))
        self.axis_position_um.setSuffix(" um")
        self.axis_position_um.setToolTip(
            "Stereotaxic coordinate relative to bregma. For AP, 0 is bregma; anterior is positive and posterior is negative."
        )
        atlas_setup_layout.addWidget(QtWidgets.QLabel("Plane"), 1, 0)
        atlas_setup_layout.addWidget(self.plane_box, 1, 1)
        atlas_setup_layout.addWidget(self.axis_label, 1, 2)
        atlas_setup_layout.addWidget(self.axis_position_um, 1, 3, 1, 2)
        atlas_setup_layout.setColumnStretch(0, 1)
        atlas_setup_layout.setColumnStretch(1, 2)
        atlas_setup_layout.setColumnStretch(2, 2)

        self.add_slice_btn = QtWidgets.QPushButton("Add slices")
        self.previous_slice_btn = QtWidgets.QPushButton("‹")
        self.previous_slice_btn.setToolTip("Previous slice (Ctrl+Left)")
        self.previous_slice_btn.setFixedWidth(32)
        self.slice_list = QtWidgets.QComboBox()
        self.slice_list.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.slice_list.setMinimumContentsLength(20)
        self.next_slice_btn = QtWidgets.QPushButton("›")
        self.next_slice_btn.setToolTip("Next slice (Ctrl+Right)")
        self.next_slice_btn.setFixedWidth(32)
        self.slice_position = QtWidgets.QLabel("0 / 0")
        slice_picker = QtWidgets.QWidget()
        slice_picker_layout = QtWidgets.QHBoxLayout(slice_picker)
        slice_picker_layout.setContentsMargins(0, 0, 0, 0)
        slice_picker_layout.setSpacing(4)
        slice_picker_layout.addWidget(self.previous_slice_btn)
        slice_picker_layout.addWidget(self.slice_list, 1)
        slice_picker_layout.addWidget(self.next_slice_btn)
        slice_picker_layout.addWidget(self.slice_position)
        self.rotation = QtWidgets.QDoubleSpinBox()
        self.rotation.setKeyboardTracking(False)
        self.rotation.setRange(-3600.0, 3600.0)
        self.rotation.setDecimals(1)
        self.rotation.setSingleStep(0.1)
        self.rotation.setSuffix(" deg")
        self.flip_horizontal = QtWidgets.QCheckBox("H")
        self.flip_horizontal.setToolTip(
            "Flip the displayed histology horizontally. The displayed A-to-P viewing orientation is authoritative."
        )
        self.flip_vertical = QtWidgets.QCheckBox("V")
        self.flip_vertical.setToolTip(
            "Flip the displayed histology vertically. The displayed A-to-P viewing orientation is authoritative."
        )
        slice_setup = QtWidgets.QGroupBox("Slices")
        slice_setup_layout = QtWidgets.QGridLayout(slice_setup)
        slice_setup_layout.addWidget(self.add_slice_btn, 0, 0)
        slice_setup_layout.addWidget(slice_picker, 0, 1, 1, 4)
        slice_setup_layout.addWidget(QtWidgets.QLabel("Rotation"), 1, 0)
        slice_setup_layout.addWidget(self.rotation, 1, 1)
        slice_setup_layout.addWidget(QtWidgets.QLabel("Flip"), 1, 2)
        slice_setup_layout.addWidget(self.flip_horizontal, 1, 3)
        slice_setup_layout.addWidget(self.flip_vertical, 1, 4)
        slice_setup_layout.setColumnStretch(1, 1)
        self.setup_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.setup_splitter.setChildrenCollapsible(False)
        self.setup_splitter.setHandleWidth(8)
        self.atlas_setup_panel = resizable_panel(atlas_setup, 260)
        self.slice_setup_panel = resizable_panel(slice_setup, 260)
        self.setup_splitter.addWidget(self.atlas_setup_panel)
        self.setup_splitter.addWidget(self.slice_setup_panel)
        self.setup_splitter.setSizes([900, 900])
        controls_layout.addWidget(self.setup_splitter, 0, 0, 1, 2)

        alignment_group = QtWidgets.QGroupBox("Alignment")
        alignment_group_layout = QtWidgets.QVBoxLayout(alignment_group)
        self.alignment_tabs = QtWidgets.QTabWidget()
        alignment_group_layout.addWidget(self.alignment_tabs)

        manual_tab = QtWidgets.QWidget()
        manual_layout = QtWidgets.QGridLayout(manual_tab)
        manual_help = QtWidgets.QLabel(
            "Set AP and cutting tilts on the atlas, then click corresponding landmarks on the atlas and histology."
        )
        manual_help.setWordWrap(True)
        manual_help.setStyleSheet("color:#9fb4c8;")
        self.landmark_mode = QtWidgets.QPushButton("Add corresponding landmarks")
        self.probe_mode = QtWidgets.QPushButton("Mark probe on slice")
        self.probe_mode.setToolTip(
            "Add probe observations directly on the histology before automatic alignment. "
            "Their atlas/3D coordinates are calculated only after an alignment is accepted."
        )
        self.auto_outline_mode = QtWidgets.QPushButton("Add / edit points")
        self.smart_surface_mode = QtWidgets.QPushButton("Smart brush")
        self.erase_surface_mode = QtWidgets.QPushButton("Erase points")
        self.landmark_mode.setCheckable(True)
        self.probe_mode.setCheckable(True)
        self.auto_outline_mode.setCheckable(True)
        self.smart_surface_mode.setCheckable(True)
        self.erase_surface_mode.setCheckable(True)
        mode_button_style = "QPushButton:checked { background:#2b6f95; border:2px solid #80d4ff; color:#ffffff; }"
        self.landmark_mode.setStyleSheet(mode_button_style)
        self.probe_mode.setStyleSheet(mode_button_style)
        self.auto_outline_mode.setStyleSheet(mode_button_style)
        self.smart_surface_mode.setStyleSheet(mode_button_style)
        self.erase_surface_mode.setStyleSheet(mode_button_style)
        self.auto_outline_mode.setToolTip(
            "Click empty space to add a point; click and drag an existing point to move it; Delete removes the selected point"
        )
        self.smart_surface_mode.setToolTip(
            "Paint over the brain to select the contrast-defined object; hold Shift while painting to subtract"
        )
        self.erase_surface_mode.setToolTip("Paint across unreliable surface points to remove them from alignment")
        self.landmark_mode.setChecked(True)
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.landmark_mode)
        self.mode_group.addButton(self.probe_mode)
        self.mode_group.addButton(self.auto_outline_mode)
        self.mode_group.addButton(self.smart_surface_mode)
        self.mode_group.addButton(self.erase_surface_mode)
        self.transform_btn = QtWidgets.QPushButton("Transform slice to atlas")
        self.undo_point_btn = QtWidgets.QPushButton("Undo landmark")
        self.clear_points_btn = QtWidgets.QPushButton("Clear landmarks")
        manual_layout.addWidget(manual_help, 0, 0, 1, 2)
        manual_layout.addWidget(self.landmark_mode, 1, 0)
        manual_layout.addWidget(self.transform_btn, 1, 1)
        manual_layout.addWidget(self.undo_point_btn, 2, 0)
        manual_layout.addWidget(self.clear_points_btn, 2, 1)
        manual_layout.setColumnStretch(0, 1)
        manual_layout.setColumnStretch(1, 1)
        manual_layout.setRowStretch(3, 1)
        self.alignment_tabs.addTab(manual_tab, "Manual alignment")

        automatic_tab = QtWidgets.QWidget()
        automatic_layout = QtWidgets.QGridLayout(automatic_tab)
        automatic_help = QtWidgets.QLabel(
            "Select at least 8 reliable outer-surface points. The selected model initializes the pose, then a bounded "
            "modality-independent atlas search matches internal anatomy; the surface fixes scale independently "
            "of the display brightness curve."
        )
        automatic_help.setWordWrap(True)
        automatic_help.setStyleSheet("color:#9fb4c8;")
        self.pose_engine = QtWidgets.QComboBox()
        self.pose_engine.addItems(POSE_ENGINES)
        self.pose_engine.setToolTip(
            "DeepSlice is the default. AtlasPose passed all absolute-quality gates and beat DeepSlice on AP and L-R; "
            "its small D-V advantage was statistically inconclusive."
        )
        self.own_cnn_weight = QtWidgets.QSpinBox()
        self.own_cnn_weight.setRange(1, 99)
        self.own_cnn_weight.setValue(round(DEFAULT_OWN_CNN_WEIGHT * 100))
        self.own_cnn_weight.setSuffix("% own CNN")
        self.own_cnn_weight.setEnabled(False)
        self.own_cnn_weight.setToolTip("Weighted vote only: remaining weight is assigned to DeepSlice")
        engine_controls = QtWidgets.QWidget()
        engine_layout = QtWidgets.QHBoxLayout(engine_controls)
        engine_layout.setContentsMargins(0, 0, 0, 0)
        engine_layout.addWidget(QtWidgets.QLabel("Pose model"))
        engine_layout.addWidget(self.pose_engine, 1)
        engine_layout.addWidget(self.own_cnn_weight)
        self.brush_radius = QtWidgets.QSpinBox()
        self.brush_radius.setRange(20, 1000)
        self.brush_radius.setSingleStep(20)
        self.brush_radius.setValue(180)
        self.brush_radius.setSuffix(" px")
        self.brush_radius.setToolTip("Brush radius in displayed-slice pixels")
        self.outline_point_count = QtWidgets.QSpinBox()
        self.outline_point_count.setKeyboardTracking(False)
        self.outline_point_count.setRange(8, 500)
        self.outline_point_count.setSingleStep(5)
        self.outline_point_count.setValue(50)
        self.outline_point_count.setSuffix(" pts")
        self.outline_point_count.setToolTip("Exact number of evenly spaced points generated by automatic brain selection")
        brush_size = QtWidgets.QWidget()
        brush_size_layout = QtWidgets.QHBoxLayout(brush_size)
        brush_size_layout.setContentsMargins(0, 0, 0, 0)
        brush_size_layout.addWidget(QtWidgets.QLabel("Brush"))
        brush_size_layout.addWidget(self.brush_radius)
        brush_size_layout.addWidget(QtWidgets.QLabel("Auto-selection"))
        brush_size_layout.addWidget(self.outline_point_count)
        self.new_outline_segment_btn = QtWidgets.QPushButton("New segment")
        self.auto_undo_point_btn = QtWidgets.QPushButton("Undo edit")
        self.auto_clear_points_btn = QtWidgets.QPushButton("Clear points")
        automatic_layout.addWidget(automatic_help, 0, 0, 1, 4)
        automatic_layout.addWidget(engine_controls, 1, 0, 1, 4)
        automatic_layout.addWidget(self.smart_surface_mode, 2, 0)
        automatic_layout.addWidget(brush_size, 2, 1, 1, 3)
        automatic_layout.addWidget(self.auto_outline_mode, 3, 0, 1, 2)
        automatic_layout.addWidget(self.erase_surface_mode, 3, 2, 1, 2)
        automatic_layout.addWidget(self.new_outline_segment_btn, 4, 0)
        automatic_layout.addWidget(self.auto_undo_point_btn, 4, 1, 1, 2)
        automatic_layout.addWidget(self.auto_clear_points_btn, 4, 3)
        self.auto_align_btn = QtWidgets.QPushButton("Auto-align current")
        self.auto_align_btn.setToolTip(
            "Align this outlined slice independently; its AP and both cutting tilts are estimated from this slice alone"
        )
        self.auto_align_btn.setEnabled(False)
        self.auto_align_all_btn = QtWidgets.QPushButton("Auto-align all")
        self.auto_align_all_btn.setToolTip(
            "Align all outlined slices together with one shared L-R/D-V cutting angle and a separate AP per slice"
        )
        self.auto_align_all_btn.setEnabled(False)
        automatic_layout.addWidget(self.auto_align_btn, 5, 0, 1, 2)
        automatic_layout.addWidget(self.auto_align_all_btn, 5, 2, 1, 2)
        try:
            verify_diffeomorphic_model_bundle(NONLINEAR_MODEL_PATH)
            self._nonlinear_backend = "learned"
            self._nonlinear_bundle_error = None
        except Exception:
            verify_classical_registration_backend()
            self._nonlinear_backend = "classical"
            self._nonlinear_bundle_error = None
        self.fit_anatomy_btn = QtWidgets.QPushButton("Fit current slice to atlas (nonlinear)")
        self.fit_anatomy_btn.setEnabled(False)
        self.fit_anatomy_btn.setToolTip(
            "After reviewing the AP, tilts and affine scale, deform local histology anatomy onto this fixed atlas plane. "
            "Only results passing conservative geometry and correspondence gates are used; a rejected result keeps "
            "the affine alignment for review."
        )
        self.nonlinear_model_status = QtWidgets.QLabel(
            "Learned diffeomorphic model"
            if self._nonlinear_backend == "learned"
            else "Validated bounded B-spline anatomical fit"
        )
        self.nonlinear_model_status.setStyleSheet("color:#9fb4c8;")
        self.nonlinear_model_status.setWordWrap(True)
        self.nonlinear_model_status.setToolTip(
            "Uses the installed learned model when source-approved; otherwise uses deterministic "
            "multimodal B-spline registration with the same geometry and correspondence rejection gates."
        )
        automatic_layout.addWidget(self.fit_anatomy_btn, 6, 0, 1, 2)
        automatic_layout.addWidget(self.nonlinear_model_status, 6, 2, 1, 2)
        self.limit_auto_align_ap = QtWidgets.QCheckBox("Limit AP search")
        self.limit_auto_align_ap.setToolTip(
            "Evaluate atlas candidates only inside this stereotaxic AP interval. This is a genuine bounded "
            "anatomical search, not clipping after prediction. Bregma is 0; anterior is positive."
        )
        self.auto_align_ap_min = QtWidgets.QSpinBox()
        self.auto_align_ap_max = QtWidgets.QSpinBox()
        for control in (self.auto_align_ap_min, self.auto_align_ap_max):
            control.setKeyboardTracking(False)
            control.setRange(-999999, 999999)
            control.setSingleStep(int(VOXEL_UM))
            control.setSuffix(" um")
            control.setEnabled(False)
        self.auto_align_ap_min.setPrefix("From ")
        self.auto_align_ap_max.setPrefix("To ")
        automatic_layout.addWidget(self.limit_auto_align_ap, 7, 0)
        automatic_layout.addWidget(self.auto_align_ap_min, 7, 1)
        automatic_layout.addWidget(self.auto_align_ap_max, 7, 2)
        order_label = QtWidgets.QLabel(
            "Optional AP-order constraint: drag slices into known anterior → posterior order and check only those "
            "whose relative order is known. Leave all unchecked to apply no order constraint."
        )
        order_label.setWordWrap(True)
        order_label.setStyleSheet("color:#9fb4c8;")
        automatic_layout.addWidget(order_label, 8, 0, 1, 4)
        self.auto_slice_order = QtWidgets.QListWidget()
        self.auto_slice_order.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.auto_slice_order.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.auto_slice_order.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.auto_slice_order.setMinimumHeight(76)
        self.auto_slice_order.setMaximumHeight(112)
        automatic_layout.addWidget(self.auto_slice_order, 9, 0, 1, 3)
        order_buttons = QtWidgets.QWidget()
        order_buttons_layout = QtWidgets.QVBoxLayout(order_buttons)
        order_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.auto_order_up_btn = QtWidgets.QPushButton("Move earlier")
        self.auto_order_down_btn = QtWidgets.QPushButton("Move later")
        order_buttons_layout.addWidget(self.auto_order_up_btn)
        order_buttons_layout.addWidget(self.auto_order_down_btn)
        automatic_layout.addWidget(order_buttons, 9, 3)
        self.alignment_summary = QtWidgets.QLabel("Auto-align: not run")
        self.alignment_summary.setToolTip(
            "Selected-model pose, component disagreement and atlas-refinement diagnostics; REVIEW is a guardrail"
        )
        self.alignment_summary.setWordWrap(True)
        automatic_layout.addWidget(self.alignment_summary, 10, 0, 1, 4)
        for column in range(4):
            automatic_layout.setColumnStretch(column, 1)
        automatic_layout.setRowStretch(11, 1)
        self.alignment_tabs.addTab(automatic_tab, "Automatic alignment")

        self.probe_type = QtWidgets.QComboBox()
        self.probe_type.addItems(["Neuropixels 1.0", "Neuropixels 2.0 single-shank", "Neuropixels 2.0 four-shank"])
        self.probe_name = QtWidgets.QComboBox()
        self.run_folder = QtWidgets.QLineEdit(str(default_run_folder))
        self.browse_run_btn = QtWidgets.QPushButton("Browse")
        self.map_btn = QtWidgets.QPushButton("Map channels/units")
        self.undo_mapping_btn = QtWidgets.QPushButton("Undo file mapping")
        for button in (
            self.load_atlas_btn,
            self.add_slice_btn,
            self.transform_btn,
            self.auto_align_btn,
            self.auto_align_all_btn,
            self.map_btn,
        ):
            button.setProperty("role", "primary")
        probe_group = QtWidgets.QGroupBox("Probe mapping")
        probe_layout = QtWidgets.QGridLayout(probe_group)
        probe_layout.addWidget(QtWidgets.QLabel("Run folder"), 0, 0)
        probe_layout.addWidget(self.run_folder, 0, 1, 1, 2)
        probe_layout.addWidget(self.browse_run_btn, 0, 3)
        probe_layout.addWidget(QtWidgets.QLabel("Probe type"), 1, 0)
        probe_layout.addWidget(self.probe_type, 1, 1, 1, 3)
        probe_layout.addWidget(QtWidgets.QLabel("Active probe (edit / map)"), 2, 0)
        probe_layout.addWidget(self.probe_name, 2, 1, 1, 3)

        constraint_group = QtWidgets.QGroupBox("Optional surgical constraints")
        constraint_layout = QtWidgets.QGridLayout(constraint_group)
        self.use_probe_constraints = QtWidgets.QCheckBox("Use for the next automatic slice alignment")
        self.use_probe_constraints.setToolTip(
            "Restrict automatic slice alignment using the planned bregma-centred insertion target and "
            "angle/depth limits. Changes take effect only after Auto-align current/all is run again; "
            "they never modify an existing alignment or pull the later observed trajectory regression."
        )
        constraint_layout.addWidget(self.use_probe_constraints, 0, 0, 1, 6)
        self.probe_constraint_scope = QtWidgets.QLabel(
            "1. Mark this probe on the raw slices.  2. Set the surgical plan.  "
            "3. Rerun Auto-align current/all. Existing alignments never change automatically."
        )
        self.probe_constraint_scope.setStyleSheet("color:#7fbbe8;")
        self.probe_constraint_scope.setWordWrap(True)
        constraint_layout.addWidget(self.probe_constraint_scope, 1, 0, 1, 6)

        self.insertion_ap_um = QtWidgets.QSpinBox()
        self.insertion_ml_um = QtWidgets.QSpinBox()
        for control in (self.insertion_ap_um, self.insertion_ml_um):
            control.setKeyboardTracking(False)
            control.setRange(-20000, 20000)
            control.setSingleStep(int(VOXEL_UM))
            control.setSuffix(" um")
        self.insertion_ap_um.setToolTip("Planned AP coordinate from bregma; anterior is positive.")
        self.insertion_ml_um.setToolTip(
            "Planned ML coordinate from bregma; positive increases toward the animal's left "
            "in the Allen CCF axis convention used by this app."
        )
        self.insertion_radius_um = QtWidgets.QSpinBox()
        self.insertion_radius_um.setKeyboardTracking(False)
        self.insertion_radius_um.setRange(0, 5000)
        self.insertion_radius_um.setSingleStep(int(VOXEL_UM))
        self.insertion_radius_um.setValue(250)
        self.insertion_radius_um.setSuffix(" um")
        self.insertion_radius_um.setToolTip(
            "Allowed radial uncertainty around the planned AP/ML insertion coordinate."
        )
        constraint_layout.addWidget(QtWidgets.QLabel("Target AP"), 2, 0)
        constraint_layout.addWidget(self.insertion_ap_um, 2, 1)
        constraint_layout.addWidget(QtWidgets.QLabel("Target ML"), 2, 2)
        constraint_layout.addWidget(self.insertion_ml_um, 2, 3)
        constraint_layout.addWidget(QtWidgets.QLabel("Radius"), 2, 4)
        constraint_layout.addWidget(self.insertion_radius_um, 2, 5)

        self.attack_angle_deg = QtWidgets.QDoubleSpinBox()
        self.attack_angle_tolerance_deg = QtWidgets.QDoubleSpinBox()
        for control in (self.attack_angle_deg, self.attack_angle_tolerance_deg):
            control.setKeyboardTracking(False)
            control.setRange(0.0, 90.0)
            control.setDecimals(1)
            control.setSingleStep(0.5)
            control.setSuffix(" deg")
            control.setToolTip("Probe angle: 0 degrees is completely horizontal and 90 degrees is vertical.")
        self.attack_angle_deg.setValue(90.0)
        self.attack_angle_tolerance_deg.setValue(5.0)
        constraint_layout.addWidget(QtWidgets.QLabel("Attack angle (0 horizontal / 90 vertical)"), 3, 0)
        constraint_layout.addWidget(self.attack_angle_deg, 3, 1)
        constraint_layout.addWidget(QtWidgets.QLabel("+/-"), 3, 2)
        constraint_layout.addWidget(self.attack_angle_tolerance_deg, 3, 3)

        self.limit_insertion_depth = QtWidgets.QCheckBox("Limit feasible insertion depth")
        self.max_insertion_depth_um = QtWidgets.QSpinBox()
        self.max_insertion_depth_um.setKeyboardTracking(False)
        self.max_insertion_depth_um.setRange(int(VOXEL_UM), 50000)
        self.max_insertion_depth_um.setSingleStep(100)
        self.max_insertion_depth_um.setValue(10000)
        self.max_insertion_depth_um.setSuffix(" um")
        depth_help = (
            "Auto-alignment constraint only: reject candidate slice poses that would place a marked "
            "electrode point deeper than this distance from the cortical entry. Leave off to use only "
            "the selected probe's physical shank limit. Changes apply after rerunning auto-alignment."
        )
        self.limit_insertion_depth.setToolTip(depth_help)
        self.max_insertion_depth_um.setToolTip(depth_help)
        constraint_layout.addWidget(self.limit_insertion_depth, 3, 4)
        constraint_layout.addWidget(self.max_insertion_depth_um, 3, 5)

        self.probe_fit_summary = QtWidgets.QLabel("Fit: add at least two trajectory points")
        self.probe_fit_summary.setStyleSheet("color:#9fb4c8;")
        self.probe_fit_summary.setWordWrap(True)
        constraint_layout.addWidget(self.probe_fit_summary, 4, 0, 1, 6)
        for column in (1, 3, 5):
            constraint_layout.setColumnStretch(column, 1)
        probe_layout.addWidget(constraint_group, 3, 0, 1, 4)

        probe_layout.addWidget(self.map_btn, 4, 0, 1, 2)
        probe_layout.addWidget(self.undo_mapping_btn, 4, 2, 1, 2)

        self.endpoint_reference = QtWidgets.QComboBox()
        self.endpoint_reference.addItem("Known insertion depth from brain surface", "known_insertion_depth")
        self.endpoint_reference.addItem("Deepest marked dot is the probe tip", "deepest_mark_is_tip")
        self.endpoint_reference.setCurrentIndex(-1)
        self.endpoint_reference.setPlaceholderText("Choose how to locate the probe tip")
        self.mapping_insertion_depth_um = QtWidgets.QDoubleSpinBox()
        self.mapping_insertion_depth_um.setKeyboardTracking(False)
        self.mapping_insertion_depth_um.setRange(1.0, 10000.0)
        self.mapping_insertion_depth_um.setDecimals(1)
        self.mapping_insertion_depth_um.setSuffix(" um")
        self.mapping_insertion_depth_um.setValue(3000.0)
        self.mapping_insertion_depth_um.setEnabled(False)
        tip_help = (
            "Choose whether the physical tip is placed at the known surgical insertion depth from the "
            "brain surface, or at the deepest electrode dot marked on the histology. Recording-contact "
            "positions are then placed automatically from the selected probe geometry."
        )
        self.endpoint_reference.setToolTip(tip_help)
        self.mapping_insertion_depth_um.setToolTip(
            "Exact distance travelled from the fitted cortical entry to the physical probe tip during surgery."
        )
        probe_layout.addWidget(QtWidgets.QLabel("Probe tip location"), 5, 0)
        probe_layout.addWidget(self.endpoint_reference, 5, 1, 1, 3)
        probe_layout.addWidget(QtWidgets.QLabel("Insertion depth from surface"), 6, 0)
        probe_layout.addWidget(self.mapping_insertion_depth_um, 6, 1)

        self.point_counts = QtWidgets.QLabel("Transform atlas 0 / slice 0 | Probe 0")
        self.point_counts.setStyleSheet("color:#9fb4c8;")
        self.brightness_weighting = QtWidgets.QCheckBox("Brightness-weighted trajectory")
        probe_layout.addWidget(self.brightness_weighting, 6, 2, 1, 2)
        self.probe_undo_point_btn = QtWidgets.QPushButton("Undo trajectory point")
        self.probe_clear_points_btn = QtWidgets.QPushButton("Clear trajectory")
        probe_layout.addWidget(QtWidgets.QLabel("Trajectory"), 7, 0)
        probe_layout.addWidget(self.probe_mode, 7, 1, 1, 3)
        probe_layout.addWidget(self.probe_undo_point_btn, 8, 0, 1, 2)
        probe_layout.addWidget(self.probe_clear_points_btn, 8, 2, 1, 2)
        for column in range(4):
            probe_layout.setColumnStretch(column, 1)
        probe_layout.setRowStretch(9, 1)
        self.workflow_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.workflow_splitter.setChildrenCollapsible(False)
        self.workflow_splitter.setHandleWidth(8)
        self.alignment_panel = resizable_panel(alignment_group, 380)
        self.probe_mapping_panel = resizable_panel(probe_group, 360)
        self.workflow_splitter.addWidget(self.alignment_panel)
        self.workflow_splitter.addWidget(self.probe_mapping_panel)
        self.workflow_splitter.setSizes([1040, 740])
        controls_layout.addWidget(self.workflow_splitter, 1, 0, 1, 2)

        self.status = QtWidgets.QLabel("Idle")
        self.status.setStyleSheet("color:#9fb4c8;")
        self.status.setWordWrap(True)
        feedback_layout = QtWidgets.QHBoxLayout()
        feedback_layout.addWidget(self.point_counts)
        feedback_layout.addStretch(1)
        feedback_layout.addWidget(self.status, 2)
        controls_layout.addLayout(feedback_layout, 2, 0, 1, 2)
        controls_layout.setRowMinimumHeight(0, 104)
        controls_layout.setRowMinimumHeight(1, 365)
        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        self.workspace_splitter.addWidget(splitter)

        atlas_group = QtWidgets.QGroupBox("Atlas")
        atlas_layout = QtWidgets.QVBoxLayout(atlas_group)
        self.atlas_panel = ImagePanel("Atlas / transformed slice comparison")
        atlas_layout.addWidget(self.atlas_panel, 1)
        self.atlas_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.atlas_opacity.setRange(0, 100)
        self.atlas_opacity.setValue(65)
        self.atlas_opacity.setEnabled(False)
        self.atlas_opacity.setToolTip("Blend between the transformed histology slice and the atlas")
        self.atlas_opacity_value = QtWidgets.QLabel("65% atlas")
        atlas_blend_row = QtWidgets.QHBoxLayout()
        atlas_blend_row.addWidget(QtWidgets.QLabel("Slice"))
        atlas_blend_row.addWidget(self.atlas_opacity, 1)
        atlas_blend_row.addWidget(QtWidgets.QLabel("Atlas"))
        atlas_blend_row.addWidget(self.atlas_opacity_value)
        atlas_layout.addLayout(atlas_blend_row)
        self.section_scroll = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        section_row = QtWidgets.QHBoxLayout()
        section_row.addWidget(QtWidgets.QLabel("Section position"))
        section_row.addWidget(self.section_scroll, 1)
        atlas_layout.addLayout(section_row)
        self.atlas_tilt_ml = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.atlas_tilt_ml.setRange(-450, 450)
        self.atlas_tilt_ml.setValue(0)
        self.atlas_tilt_ml.setToolTip("Coronal AP tilt across the left-right (ML) axis")
        self.atlas_tilt_ml_value = QtWidgets.QLabel("0.0°")
        tilt_ml_row = QtWidgets.QHBoxLayout()
        tilt_ml_row.addWidget(QtWidgets.QLabel("Coronal L–R tilt"))
        tilt_ml_row.addWidget(self.atlas_tilt_ml, 1)
        tilt_ml_row.addWidget(self.atlas_tilt_ml_value)
        atlas_layout.addLayout(tilt_ml_row)
        self.atlas_tilt_dv = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.atlas_tilt_dv.setRange(-450, 450)
        self.atlas_tilt_dv.setValue(0)
        self.atlas_tilt_dv.setToolTip("Coronal AP tilt across the dorsal-ventral (DV) axis")
        self.atlas_tilt_dv_value = QtWidgets.QLabel("0.0°")
        tilt_dv_row = QtWidgets.QHBoxLayout()
        tilt_dv_row.addWidget(QtWidgets.QLabel("Coronal D–V tilt"))
        tilt_dv_row.addWidget(self.atlas_tilt_dv, 1)
        tilt_dv_row.addWidget(self.atlas_tilt_dv_value)
        atlas_layout.addLayout(tilt_dv_row)
        splitter.addWidget(atlas_group)

        slice_group = QtWidgets.QGroupBox("Slice")
        slice_layout = QtWidgets.QVBoxLayout(slice_group)
        self.slice_panel = ImagePanel("Brain slice")
        self.curve_editor = CurveEditor()
        self.curve_group = QtWidgets.QGroupBox("Contrast curve — display only")
        curve_group_layout = QtWidgets.QVBoxLayout(self.curve_group)
        curve_group_layout.addWidget(self.curve_editor)
        self.slice_workspace_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.slice_workspace_splitter.setChildrenCollapsible(True)
        self.slice_workspace_splitter.setHandleWidth(8)
        self.slice_workspace_splitter.addWidget(self.slice_panel)
        self.slice_workspace_splitter.addWidget(self.curve_group)
        self.slice_workspace_splitter.setStretchFactor(0, 4)
        self.slice_workspace_splitter.setStretchFactor(1, 1)
        self.slice_workspace_splitter.setSizes([520, 170])
        slice_layout.addWidget(self.slice_workspace_splitter, 1)
        splitter.addWidget(slice_group)

        view3d_group = QtWidgets.QGroupBox("3D trajectory")
        view3d_layout = QtWidgets.QVBoxLayout(view3d_group)
        self.view3d = gl.GLViewWidget()
        self.view3d.setBackgroundColor("#05070a")
        self.view3d.setCameraPosition(pos=QtGui.QVector3D(228, 264, 160), distance=760, elevation=22, azimuth=35)
        view3d_layout.addWidget(self.view3d, 1)
        brain_opacity_row = QtWidgets.QHBoxLayout()
        self.brain_opacity_label = QtWidgets.QLabel("Brain opacity")
        self.brain_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brain_opacity.setRange(0, 100)
        self.brain_opacity.setValue(45)
        self.brain_opacity_value = QtWidgets.QLabel("45%")
        self.show_all_slice_planes = QtWidgets.QCheckBox("All slice planes")
        self.show_all_slice_planes.setToolTip(
            "Show the saved AP and cutting plane for every loaded slice instead of only the current slice"
        )
        self.reset_3d_view_btn = QtWidgets.QPushButton("Reset 3D view")
        brain_opacity_row.addWidget(self.brain_opacity_label)
        brain_opacity_row.addWidget(self.brain_opacity, 1)
        brain_opacity_row.addWidget(self.brain_opacity_value)
        view3d_layout.addLayout(brain_opacity_row)
        view3d_actions_row = QtWidgets.QHBoxLayout()
        view3d_actions_row.addWidget(self.show_all_slice_planes)
        view3d_actions_row.addStretch(1)
        view3d_actions_row.addWidget(self.reset_3d_view_btn)
        view3d_layout.addLayout(view3d_actions_row)
        splitter.addWidget(view3d_group)
        splitter.setSizes([620, 620, 540])
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([560, 420])

        self.atlas_panel.clicked.connect(self._atlas_clicked)
        self.slice_panel.clicked.connect(self._slice_clicked)
        self.browse_atlas_btn.clicked.connect(self._browse_atlas)
        self.load_atlas_btn.clicked.connect(lambda: self.load_atlas_folder(Path(self.atlas_path.text().strip())))
        self.plane_box.currentTextChanged.connect(self._plane_changed)
        self.section_scroll.valueChanged.connect(self._section_changed)
        self.atlas_tilt_ml.valueChanged.connect(self._atlas_tilt_changed)
        self.atlas_tilt_dv.valueChanged.connect(self._atlas_tilt_changed)
        self.axis_position_um.valueChanged.connect(self._axis_um_changed)
        self.add_slice_btn.clicked.connect(self._load_slice_dialog)
        self.slice_list.currentIndexChanged.connect(self._switch_slice)
        self.previous_slice_btn.clicked.connect(lambda: self._step_slice(-1))
        self.next_slice_btn.clicked.connect(lambda: self._step_slice(1))
        self.rotation.valueChanged.connect(self._rotation_changed)
        self.flip_horizontal.toggled.connect(self._slice_geometry_changed)
        self.flip_vertical.toggled.connect(self._slice_geometry_changed)
        self.curve_editor.points_changed.connect(self._curve_changed)
        self.transform_btn.clicked.connect(self.transform_current_slice)
        self.auto_align_btn.clicked.connect(self._auto_align_clicked)
        self.auto_align_all_btn.clicked.connect(self._auto_align_all_clicked)
        self.fit_anatomy_btn.clicked.connect(self._fit_current_slice_anatomy_clicked)
        self.new_outline_segment_btn.clicked.connect(self.start_new_surface_segment)
        self.auto_order_up_btn.clicked.connect(lambda: self._move_auto_order_item(-1))
        self.auto_order_down_btn.clicked.connect(lambda: self._move_auto_order_item(1))
        self.auto_slice_order.itemClicked.connect(self._auto_order_slice_clicked)
        self.auto_slice_order.itemChanged.connect(self._auto_order_constraint_changed)
        self.auto_slice_order.model().rowsMoved.connect(self._auto_order_constraint_changed)
        self.pose_engine.currentTextChanged.connect(self._pose_engine_changed)
        self.limit_auto_align_ap.toggled.connect(self.auto_align_ap_min.setEnabled)
        self.limit_auto_align_ap.toggled.connect(self.auto_align_ap_max.setEnabled)
        self.undo_point_btn.clicked.connect(lambda: self._undo_for_mode(self.landmark_mode))
        self.clear_points_btn.clicked.connect(lambda: self._clear_for_mode(self.landmark_mode))
        self.auto_undo_point_btn.clicked.connect(self.undo_last_point)
        self.auto_clear_points_btn.clicked.connect(self.clear_current_points)
        self.probe_undo_point_btn.clicked.connect(lambda: self._undo_for_mode(self.probe_mode))
        self.probe_clear_points_btn.clicked.connect(lambda: self._clear_for_mode(self.probe_mode))
        self.atlas_opacity.valueChanged.connect(self._atlas_opacity_changed)
        self.brain_opacity.valueChanged.connect(self._brain_opacity_changed)
        self.show_all_slice_planes.toggled.connect(self._refresh_3d)
        self.reset_3d_view_btn.clicked.connect(self._reset_3d_camera)
        self.browse_run_btn.clicked.connect(self._browse_run)
        self.run_folder.editingFinished.connect(self._refresh_probe_names)
        self.probe_name.currentTextChanged.connect(self._probe_selection_changed)
        self.probe_type.currentTextChanged.connect(self._probe_type_changed)
        self.use_probe_constraints.toggled.connect(self._probe_constraints_changed)
        self.insertion_ap_um.valueChanged.connect(self._probe_constraints_changed)
        self.insertion_ml_um.valueChanged.connect(self._probe_constraints_changed)
        self.insertion_radius_um.valueChanged.connect(self._probe_constraints_changed)
        self.attack_angle_deg.valueChanged.connect(self._probe_constraints_changed)
        self.attack_angle_tolerance_deg.valueChanged.connect(self._probe_constraints_changed)
        self.limit_insertion_depth.toggled.connect(self._probe_constraints_changed)
        self.max_insertion_depth_um.valueChanged.connect(self._probe_constraints_changed)
        self.endpoint_reference.currentIndexChanged.connect(
            self._probe_reference_changed
        )
        self.mapping_insertion_depth_um.valueChanged.connect(self._mapping_insertion_depth_changed)
        self.map_btn.clicked.connect(self._map_channels_units_clicked)
        self.undo_mapping_btn.clicked.connect(self.undo_file_mapping)
        self.brightness_weighting.toggled.connect(self._trajectory_weighting_changed)
        self.mode_group.buttonClicked.connect(self._point_target_changed)
        self.alignment_tabs.currentChanged.connect(self._alignment_tab_changed)
        self.brush_radius.valueChanged.connect(self.slice_panel.set_brush_radius)
        self.outline_point_count.valueChanged.connect(self._outline_point_count_changed)
        self.slice_panel.brush_stroke.connect(self._smart_surface_stroke)
        self.slice_panel.outline_drag_started.connect(self._surface_point_drag_started)
        self.slice_panel.outline_point_moved.connect(self._surface_point_moved)
        self.slice_panel.outline_point_deleted.connect(self._surface_point_deleted)
        self.slice_panel.set_brush_radius(self.brush_radius.value())
        self.undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.undo_last_point)
        self.previous_slice_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Left"), self)
        self.previous_slice_shortcut.activated.connect(lambda: self._step_slice(-1))
        self.next_slice_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Right"), self)
        self.next_slice_shortcut.activated.connect(lambda: self._step_slice(1))
        self.delete_surface_point_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self)
        self.delete_surface_point_shortcut.activated.connect(self.slice_panel.delete_selected_outline_point)
        self.backspace_surface_point_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Backspace), self)
        self.backspace_surface_point_shortcut.activated.connect(self.slice_panel.delete_selected_outline_point)
        self._update_slice_navigation()
        self._refresh_probe_names()

    def _map_channels_units_clicked(self) -> None:
        try:
            self.map_channels_units()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Anatomy mapping failed", str(exc))
            self.status.setText(f"Mapping failed: {exc}")

    def _browse_atlas(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select atlas folder", self.atlas_path.text())
        if path:
            self.atlas_path.setText(path)

    def _browse_run(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select preprocessed run folder", self.run_folder.text())
        if path:
            self.run_folder.setText(path)
            self._refresh_probe_names()

    def _save_session_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save complete tracker session",
            str(self.default_slices_folder / "trajectory_session.attracker"),
            "Anatomy Tracker session (*.attracker)",
        )
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".attracker":
            destination = destination.with_suffix(".attracker")
        try:
            self.save_session_file(destination)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Session save failed", str(exc))
            return
        self.status.setText(f"Saved complete session: {destination}")

    def _load_session_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open tracker session",
            str(self.default_slices_folder),
            "Anatomy Tracker session (*.attracker)",
        )
        if not path:
            return
        try:
            self.load_session_file(Path(path))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Session load failed", str(exc))

    def save_session_file(self, destination: Path) -> None:
        if self.auto_alignment_busy:
            raise RuntimeError("Wait for automatic alignment to finish or cancel it before saving")
        destination = Path(destination)
        temporary = destination.with_name(destination.name + ".tmp")
        order = [
            {
                "session_index": int(self.auto_slice_order.item(row).data(QtCore.Qt.ItemDataRole.UserRole)),
                "checked": self.auto_slice_order.item(row).checkState() == QtCore.Qt.CheckState.Checked,
            }
            for row in range(self.auto_slice_order.count())
        ]
        state = {
            "format": SESSION_ARCHIVE_FORMAT,
            "version": SESSION_ARCHIVE_VERSION,
            "saved_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "atlas_folder": str(self.atlas_folder),
            "atlas_file_hashes": self.atlas_file_hashes,
            "query_file_hash": self.query_file_hash,
            "run_folder": self.run_folder.text(),
            "probe_type": self.probe_type.currentText(),
            "probe_names": [self.probe_name.itemText(index) for index in range(self.probe_name.count())],
            "active_probe": self._active_probe_name(),
            "probe_constraints": {
                name: asdict(constraint) for name, constraint in self.probe_constraints.items()
            },
            "probe_endpoint_settings": {
                name: [mode, depth]
                for name, (mode, depth) in self.probe_endpoint_settings.items()
            },
            "current_session_index": self.current_session_index,
            "alignment_tab": self.alignment_tabs.currentIndex(),
            "pose_engine": self.pose_engine.currentText(),
            "own_cnn_weight": self.own_cnn_weight.value(),
            "limit_ap_search": self.limit_auto_align_ap.isChecked(),
            "ap_search_min_um": self.auto_align_ap_min.value(),
            "ap_search_max_um": self.auto_align_ap_max.value(),
            "outline_point_count": self.outline_point_count.value(),
            "brush_radius": self.brush_radius.value(),
            "brightness_weighting": self.brightness_weighting.isChecked(),
            "atlas_opacity": self.atlas_opacity.value(),
            "brain_opacity": self.brain_opacity.value(),
            "show_all_slice_planes": self.show_all_slice_planes.isChecked(),
            "auto_slice_order": order,
            "splitters": {
                "workspace": self.workspace_splitter.sizes(),
                "setup": self.setup_splitter.sizes(),
                "workflow": self.workflow_splitter.sizes(),
                "slice_workspace": self.slice_workspace_splitter.sizes(),
            },
            "sessions": [],
        }
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                with tempfile.TemporaryDirectory() as transform_folder:
                    for index, session in enumerate(self.sessions):
                        source = Path(session.path)
                        if not source.is_file():
                            raise FileNotFoundError(f"Loaded slice is missing: {source}")
                        image_member = f"slices/{index:04d}{source.suffix.lower()}"
                        archive.write(source, image_member)
                        record = {
                            "name": session.name,
                            "original_path": str(source),
                            "image_member": image_member,
                            "image_sha256": file_sha256(source),
                            "fields": {
                                name: json_value(getattr(session, name))
                                for name in SESSION_STATE_FIELDS
                            },
                            "point_history": list(session.point_history),
                            "probe_traces": {
                                name: json_value(asdict(trace))
                                for name, trace in session.probe_traces.items()
                            },
                        }
                        if session.brain_brush_selection_mask is not None:
                            mask_member = f"state/{index:04d}_brush_mask.npz"
                            payload = io.BytesIO()
                            np.savez_compressed(
                                payload,
                                mask=np.asarray(session.brain_brush_selection_mask, dtype=np.uint8),
                            )
                            archive.writestr(mask_member, payload.getvalue())
                            record["brush_mask_member"] = mask_member
                        if session.slice_atlas_transform is not None:
                            transform_member = f"state/{index:04d}_slice_atlas_transform.npz"
                            transform_path = Path(transform_folder) / f"{index:04d}.npz"
                            session.slice_atlas_transform.save_npz(transform_path)
                            archive.write(transform_path, transform_member)
                            record["transform_member"] = transform_member
                        state["sessions"].append(record)
                archive.writestr(
                    "session.json",
                    json.dumps(json_value(state), indent=2, allow_nan=False).encode("utf-8"),
                )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load_session_file(self, source: Path) -> None:
        if self.auto_alignment_busy:
            raise RuntimeError("Wait for automatic alignment to finish or cancel it before opening a session")
        source = Path(source)
        cache = tempfile.TemporaryDirectory(prefix="anatomy_tracker_session_")
        cache_path = Path(cache.name)
        try:
            with zipfile.ZipFile(source, "r") as archive:
                state = json.loads(archive.read("session.json"))
                if state.get("format") != SESSION_ARCHIVE_FORMAT or state.get("version") != SESSION_ARCHIVE_VERSION:
                    raise ValueError("Unsupported or invalid Anatomy Tracker session file")
                saved_hashes = state.get("atlas_file_hashes", {})
                if saved_hashes and self.atlas_file_hashes and saved_hashes != self.atlas_file_hashes:
                    raise ValueError("This session was saved with a different atlas installation")
                new_sessions = []
                for index, record in enumerate(state["sessions"]):
                    image_member = record["image_member"]
                    suffix = Path(image_member).suffix
                    image_path = cache_path / f"slice_{index:04d}{suffix}"
                    with archive.open(image_member) as input_file, open(image_path, "wb") as output_file:
                        shutil.copyfileobj(input_file, output_file)
                    if file_sha256(image_path) != record["image_sha256"]:
                        raise ValueError(f"Embedded slice failed its integrity check: {record['name']}")
                    session = SliceSession(name=record["name"], path=str(image_path))
                    self._load_session_image(session)
                    for name, value in record["fields"].items():
                        if name in SESSION_STATE_FIELDS:
                            setattr(session, name, value)
                    session.point_history = list(record.get("point_history", []))
                    session.probe_traces = {
                        name: ProbeTrace(**trace)
                        for name, trace in record.get("probe_traces", {}).items()
                    }
                    session.adjusted = apply_curve(session.raw_display, session.curve_points)
                    session.rotated, session.slice_transform = transform_slice_image(
                        session.adjusted,
                        session.rotation_deg,
                        session.flip_horizontal,
                        session.flip_vertical,
                    )
                    session.weight_image, _ = transform_slice_image(
                        session.raw_display,
                        session.rotation_deg,
                        session.flip_horizontal,
                        session.flip_vertical,
                    )
                    if mask_member := record.get("brush_mask_member"):
                        with np.load(io.BytesIO(archive.read(mask_member)), allow_pickle=False) as values:
                            session.brain_brush_selection_mask = values["mask"].astype(bool)
                    if transform_member := record.get("transform_member"):
                        transform_path = cache_path / f"transform_{index:04d}.npz"
                        transform_path.write_bytes(archive.read(transform_member))
                        session.slice_atlas_transform = SliceAtlasTransform2D.load_npz(transform_path)
                    elif min(len(session.atlas_landmarks), len(session.slice_landmarks)) >= 3:
                        atlas_points = np.asarray(session.atlas_landmarks, dtype=np.float64)
                        slice_points = np.asarray(
                            transform_points(session.slice_landmarks, session.slice_transform),
                            dtype=np.float64,
                        )
                        session.slice_to_atlas_x = Rbf(
                            slice_points[:, 0], slice_points[:, 1], atlas_points[:, 0], function="thin_plate", smooth=0.0
                        )
                        session.slice_to_atlas_y = Rbf(
                            slice_points[:, 0], slice_points[:, 1], atlas_points[:, 1], function="thin_plate", smooth=0.0
                        )
                        session.atlas_to_slice_x = Rbf(
                            atlas_points[:, 0], atlas_points[:, 1], slice_points[:, 0], function="thin_plate", smooth=0.0
                        )
                        session.atlas_to_slice_y = Rbf(
                            atlas_points[:, 0], atlas_points[:, 1], slice_points[:, 1], function="thin_plate", smooth=0.0
                        )
                    new_sessions.append(session)
        except Exception:
            cache.cleanup()
            raise

        atlas_folder = Path(state.get("atlas_folder", ""))
        if self.atlas_volume is None and atlas_folder.is_dir():
            self.load_atlas_folder(atlas_folder)
        if state.get("atlas_file_hashes") and self.atlas_file_hashes != state["atlas_file_hashes"]:
            cache.cleanup()
            raise ValueError("The atlas required by this session is not available")

        for old_cache in self._session_cache_dirs:
            old_cache.cleanup()
        self._session_cache_dirs = [cache]
        self.sessions = new_sessions
        self.current_session_index = -1
        self.probe_constraints = {
            name: ProbeInsertionConstraint(**value)
            for name, value in state.get("probe_constraints", {}).items()
        }
        self.probe_endpoint_settings = {
            name: (value[0], float(value[1]))
            for name, value in state.get("probe_endpoint_settings", {}).items()
        }
        self.run_folder.setText(state.get("run_folder", ""))
        self.probe_type.blockSignals(True)
        self.probe_type.setCurrentText(state.get("probe_type", self.probe_type.currentText()))
        self.probe_type.blockSignals(False)
        self.pose_engine.setCurrentText(state.get("pose_engine", self.pose_engine.currentText()))
        self.own_cnn_weight.setValue(state.get("own_cnn_weight", self.own_cnn_weight.value()))
        self.limit_auto_align_ap.setChecked(state.get("limit_ap_search", False))
        self.auto_align_ap_min.setValue(state.get("ap_search_min_um", self.auto_align_ap_min.value()))
        self.auto_align_ap_max.setValue(state.get("ap_search_max_um", self.auto_align_ap_max.value()))
        self.outline_point_count.setValue(state.get("outline_point_count", 50))
        self.brush_radius.setValue(state.get("brush_radius", self.brush_radius.value()))
        self.brightness_weighting.setChecked(state.get("brightness_weighting", False))
        self.atlas_opacity.setValue(state.get("atlas_opacity", 65))
        self.brain_opacity.setValue(state.get("brain_opacity", 45))
        self.show_all_slice_planes.setChecked(state.get("show_all_slice_planes", False))
        self.alignment_tabs.setCurrentIndex(state.get("alignment_tab", 1))
        for name, sizes in state.get("splitters", {}).items():
            splitter = getattr(self, f"{name}_splitter", None)
            if splitter is not None:
                splitter.setSizes(sizes)

        self.slice_list.blockSignals(True)
        self.slice_list.clear()
        for session in self.sessions:
            self.slice_list.addItem(session.name, session.path)
            self.slice_list.setItemData(
                self.slice_list.count() - 1,
                f"Embedded from {source}\nOriginal: {next(record['original_path'] for record in state['sessions'] if record['name'] == session.name)}",
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        self.slice_list.blockSignals(False)
        self._update_auto_order_labels()
        ordered = state.get("auto_slice_order", [])
        self.auto_slice_order.blockSignals(True)
        for target_row, item_state in enumerate(ordered):
            source_row = next(
                (
                    row for row in range(self.auto_slice_order.count())
                    if self.auto_slice_order.item(row).data(QtCore.Qt.ItemDataRole.UserRole) == item_state["session_index"]
                ),
                -1,
            )
            if source_row >= 0:
                item = self.auto_slice_order.takeItem(source_row)
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked if item_state["checked"] else QtCore.Qt.CheckState.Unchecked
                )
                self.auto_slice_order.insertItem(target_row, item)
        self.auto_slice_order.blockSignals(False)

        self._refresh_probe_names()
        required_probe_names = set(state.get("probe_names", [])) | set(self.probe_constraints)
        required_probe_names.update(
            name for session in self.sessions for name in session.probe_traces
        )
        existing = {self.probe_name.itemText(index) for index in range(self.probe_name.count())}
        for name in sorted(required_probe_names - existing):
            color = probe_color(name)
            swatch = QtGui.QPixmap(12, 12)
            swatch.fill(QtGui.QColor(*color))
            self.probe_name.addItem(QtGui.QIcon(swatch), name)
        active_probe = state.get("active_probe", "")
        if active_probe:
            self.probe_name.setCurrentText(active_probe)
        index = int(np.clip(state.get("current_session_index", 0), 0, max(0, len(self.sessions) - 1)))
        if self.sessions:
            self.slice_list.setCurrentIndex(index)
            self._switch_slice(index)
        else:
            self._update_slice_navigation()
        self._probe_selection_changed()
        self.status.setText(f"Restored complete session from {source}")

    def load_atlas_folder(self, folder: Path) -> None:
        template = folder / "average_template_25.nrrd"
        annotation = folder / "annotation_25.nrrd"
        if not template.exists() or not annotation.exists():
            QtWidgets.QMessageBox.warning(self, "Atlas files missing", f"Missing average_template_25.nrrd or annotation_25.nrrd in:\n{folder}")
            return
        self.status.setText("Loading atlas")
        QtWidgets.QApplication.processEvents()
        try:
            atlas_volume = nrrd.read(str(template))[0]
            annotation_volume = nrrd.read(str(annotation))[0]
            bregma_voxel = self._default_bregma_for_shape(atlas_volume.shape)
            if annotation_volume.shape != atlas_volume.shape:
                raise ValueError(
                    f"Atlas template shape {atlas_volume.shape} and annotation shape "
                    f"{annotation_volume.shape} do not match"
                )
            region_names = self._load_region_names(folder / "query.csv")
            cortical_region_ids = self._load_cortical_region_ids(folder / "query.csv")
            query_file_hash = file_sha256(folder / "query.csv") if (folder / "query.csv").exists() else None
            atlas_file_hashes = {
                template.name: file_sha256(template),
                annotation.name: file_sha256(annotation),
            }
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Atlas load failed", str(exc))
            self.status.setText(f"Atlas load failed: {exc}")
            return
        atlas_changed = bool(self.atlas_file_hashes and atlas_file_hashes != self.atlas_file_hashes)
        affected_sessions = [
            session
            for session in self.sessions
            if session.slice_atlas_transform is not None
            or session.slice_to_atlas_x is not None
            or any(trace.volume_points for trace in session.probe_traces.values())
        ]
        if atlas_changed and affected_sessions:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Replace atlas?",
                "The selected atlas differs from the one used for existing alignments. "
                "Replace it and clear all atlas-dependent transforms and derived probe coordinates?",
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                self.status.setText("Atlas replacement cancelled")
                return
            for session in affected_sessions:
                self._clear_slice_transform(session)
        self.atlas_folder = folder
        self.atlas_path.setText(str(folder))
        self.atlas_volume = atlas_volume
        self.annotation_volume = annotation_volume
        self.bregma_voxel = bregma_voxel
        self.atlas_file_hashes = atlas_file_hashes
        self.query_file_hash = query_file_hash
        self._setup_auto_align_ap_range()
        self.region_names = region_names
        self.cortical_region_ids = cortical_region_ids
        self._setup_3d_static(folder)
        self._set_plane_limits()
        self._refresh_atlas()
        self._refresh_3d()
        self._refresh_point_counts()
        self.status.setText(f"Loaded atlas: {folder}")

    def _default_bregma_for_shape(self, shape: tuple[int, int, int]) -> np.ndarray:
        if tuple(shape) != ALLEN_CCF_25_SHAPE_AP_DV_ML:
            raise ValueError(
                f"Expected Allen CCFv3 25 um atlas shape {ALLEN_CCF_25_SHAPE_AP_DV_ML}, got {tuple(shape)}"
            )
        return DEFAULT_BREGMA_VOXEL_AP_DV_ML.copy()

    def _load_region_names(self, query_path: Path) -> dict[int, tuple[str, str]]:
        if not query_path.exists():
            return {}
        table = pd.read_csv(query_path)
        names: dict[int, tuple[str, str]] = {}
        for row in table.itertuples(index=False):
            region_id = int(getattr(row, "id"))
            names[region_id] = (str(getattr(row, "name", region_id)), str(getattr(row, "acronym", region_id)))
        return names

    def _load_cortical_region_ids(self, query_path: Path) -> set[int]:
        if not query_path.exists():
            return set()
        table = pd.read_csv(query_path, usecols=["id", "structure_id_path"])
        paths = table["structure_id_path"].fillna("").astype(str)
        return set(table.loc[paths.str.contains("/315/", regex=False), "id"].astype(int))

    def _setup_3d_static(self, folder: Path) -> None:
        self.view3d.clear()
        self.brain_mesh_item = None
        self._reset_3d_camera()
        grid = gl.GLGridItem()
        grid.setSize(x=456, y=528, z=1)
        grid.setSpacing(x=50, y=50, z=1)
        grid.translate(228, 264, 0)
        self.view3d.addItem(grid)
        self._add_volume_box()
        mesh_path = folder / "atlas_meshdata.pkl"
        if mesh_path.exists():
            with open(mesh_path, "rb") as handle:
                mesh_data = pickle.load(handle)
            if mesh_data is not None:
                self.brain_mesh_item = gl.GLMeshItem(
                    meshdata=mesh_data,
                    color=self._brain_mesh_color(),
                    smooth=True,
                    shader="balloon",
                )
                self.brain_mesh_item.setGLOptions("additive")
                self.view3d.addItem(self.brain_mesh_item)
        else:
            self.status.setText(f"Whole-brain mesh missing: {mesh_path}")
        self._refresh_3d()

    def _brain_mesh_color(self) -> tuple[float, float, float, float]:
        return (0.55, 0.62, 0.72, self.brain_opacity.value() / 100.0)

    def _brain_opacity_changed(self, value: int) -> None:
        self.brain_opacity_value.setText(f"{value}%")
        if self.brain_mesh_item is not None:
            self.brain_mesh_item.setVisible(value > 0)
            self.brain_mesh_item.setColor(self._brain_mesh_color())

    def _reset_3d_camera(self) -> None:
        center = QtGui.QVector3D(228, 264, 160)
        if self.atlas_volume is not None:
            ap, dv, ml = self.atlas_volume.shape
            center = QtGui.QVector3D(ml / 2, ap / 2, dv / 2)
        self.view3d.setCameraPosition(pos=center, distance=760, elevation=22, azimuth=35)

    def _add_volume_box(self) -> None:
        if self.atlas_volume is None:
            return
        ap, dv, ml = (size - 1 for size in self.atlas_volume.shape)
        corners = np.array(
            [
                [0, 0, 0],
                [ap, 0, 0],
                [ap, dv, 0],
                [0, dv, 0],
                [0, 0, ml],
                [ap, 0, ml],
                [ap, dv, ml],
                [0, dv, ml],
            ],
            dtype=np.float32,
        )
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        line_points = []
        for a, b in edges:
            line_points.extend([corners[a], corners[b]])
        item = gl.GLLinePlotItem(pos=volume_to_gl(np.array(line_points)), color=(0.7, 0.85, 1.0, 0.5), width=1, mode="lines")
        self.view3d.addItem(item)

    def _set_plane_limits(self) -> None:
        if self.atlas_volume is None:
            return
        axis = plane_axis(self.plane_box.currentText())
        index = int(np.clip(round(float(self.bregma_voxel[axis])), 0, self.atlas_volume.shape[axis] - 1))
        self.section_scroll.blockSignals(True)
        self.section_scroll.setRange(0, self.atlas_volume.shape[axis] - 1)
        self.section_scroll.setValue(index)
        self.section_scroll.blockSignals(False)
        self._update_axis_control(index)
        self._update_tilt_controls()

    def _current_atlas_tilts(self) -> tuple[float, float]:
        if self.plane_box.currentText() != "coronal":
            return 0.0, 0.0
        return self.atlas_tilt_ml.value() / 10.0, self.atlas_tilt_dv.value() / 10.0

    def _update_tilt_controls(self) -> None:
        enabled = self.plane_box.currentText() == "coronal"
        self.atlas_tilt_ml.setEnabled(enabled)
        self.atlas_tilt_dv.setEnabled(enabled)
        self.atlas_tilt_ml_value.setEnabled(enabled)
        self.atlas_tilt_dv_value.setEnabled(enabled)

    def _atlas_tilt_changed(self) -> None:
        tilt_ml, tilt_dv = self._current_atlas_tilts()
        self.atlas_tilt_ml_value.setText(f"{self.atlas_tilt_ml.value() / 10.0:+.1f}°")
        self.atlas_tilt_dv_value.setText(f"{self.atlas_tilt_dv.value() / 10.0:+.1f}°")
        session = self.current_session()
        if session is not None:
            manually_overridden = session.auto_alignment_engine is not None
            session.atlas_tilt_ml_deg = tilt_ml
            session.atlas_tilt_dv_deg = tilt_dv
            if manually_overridden:
                self._detach_auto_alignment_for_manual_pose(session)
            self._recompute_session_volume_points(session)
        self._refresh_atlas()
        self._refresh_3d()
        if session is not None and manually_overridden:
            self.status.setText(
                "Atlas tilt adjusted manually; any nonlinear map was invalidated and the affine alignment was retained."
            )

    def _index_to_um(self, index: int) -> int:
        axis = plane_axis(self.plane_box.currentText())
        return int(round((index - float(self.bregma_voxel[axis])) * VOXEL_UM * STEREOTAXIC_AXIS_SIGN_AP_DV_ML[axis]))

    def _ap_index_to_um(self, index: int) -> int:
        return int(round((index - float(self.bregma_voxel[0])) * VOXEL_UM * STEREOTAXIC_AXIS_SIGN_AP_DV_ML[0]))

    def _ap_um_to_index(self, value_um: int) -> int:
        if self.atlas_volume is None:
            return 0
        index = int(round(value_um / (VOXEL_UM * STEREOTAXIC_AXIS_SIGN_AP_DV_ML[0]) + float(self.bregma_voxel[0])))
        return int(np.clip(index, 0, self.atlas_volume.shape[0] - 1))

    def _setup_auto_align_ap_range(self) -> None:
        if self.atlas_volume is None:
            return
        values = (self._ap_index_to_um(0), self._ap_index_to_um(self.atlas_volume.shape[0] - 1))
        minimum, maximum = min(values), max(values)
        for control in (self.auto_align_ap_min, self.auto_align_ap_max):
            control.setRange(minimum, maximum)
        self.auto_align_ap_min.setValue(minimum)
        self.auto_align_ap_max.setValue(maximum)

    def _auto_align_index_bounds(self) -> tuple[int, int]:
        if self.atlas_volume is None:
            return 0, 0
        first = self._ap_um_to_index(self.auto_align_ap_min.value())
        second = self._ap_um_to_index(self.auto_align_ap_max.value())
        return min(first, second), max(first, second)

    def _um_to_index(self, value_um: int) -> int:
        if self.atlas_volume is None:
            return 0
        axis = plane_axis(self.plane_box.currentText())
        index = int(round(value_um / (VOXEL_UM * STEREOTAXIC_AXIS_SIGN_AP_DV_ML[axis]) + float(self.bregma_voxel[axis])))
        return int(np.clip(index, 0, self.atlas_volume.shape[axis] - 1))

    def _update_axis_control(self, index: int) -> None:
        if self.atlas_volume is None:
            return
        plane = self.plane_box.currentText()
        axis = plane_axis(plane)
        min_um = self._index_to_um(0)
        max_um = self._index_to_um(self.atlas_volume.shape[axis] - 1)
        if min_um > max_um:
            min_um, max_um = max_um, min_um
        if plane == "coronal":
            self.axis_label.setText("AP from bregma (+ anterior)")
        else:
            self.axis_label.setText(f"{plane_axis_name(plane)} from bregma")
        self.axis_position_um.blockSignals(True)
        self.axis_position_um.setRange(min_um, max_um)
        self.axis_position_um.setValue(self._index_to_um(index))
        self.axis_position_um.blockSignals(False)

    def _plane_changed(self) -> None:
        self._set_plane_limits()
        session = self.current_session()
        if session is not None:
            if session.auto_alignment_engine is not None:
                self._mark_alignment_run_stale(session, "contributor atlas plane changed")
                self._clear_slice_transform(session)
            session.atlas_plane = self.plane_box.currentText()
            session.atlas_index = self.section_scroll.value()
            self._recompute_session_volume_points(session)
        self._refresh_atlas()
        self._refresh_3d()
        self._refresh_point_counts()

    def _section_changed(self, value: int) -> None:
        self.section_scroll.blockSignals(True)
        self.section_scroll.setValue(value)
        self.section_scroll.blockSignals(False)
        self._update_axis_control(value)
        session = self.current_session()
        if session is not None:
            manually_overridden = session.auto_alignment_engine is not None
            session.atlas_plane = self.plane_box.currentText()
            session.atlas_index = value
            if manually_overridden:
                self._detach_auto_alignment_for_manual_pose(session)
            self._recompute_session_volume_points(session)
        self._refresh_atlas()
        self._refresh_3d()
        if session is not None and manually_overridden:
            self.status.setText(
                "Atlas position adjusted manually; any nonlinear map was invalidated and the affine alignment was retained."
            )

    def _axis_um_changed(self, value_um: int) -> None:
        self._section_changed(self._um_to_index(value_um))

    def _atlas_opacity_changed(self, value: int) -> None:
        self.atlas_opacity_value.setText(f"{value}% atlas")
        self.atlas_panel.set_overlay_opacity(value / 100.0)

    def _refresh_atlas(self) -> None:
        if self.atlas_volume is None or self.annotation_volume is None:
            return
        plane = self.plane_box.currentText()
        index = self.section_scroll.value()
        tilt_ml, tilt_dv = self._current_atlas_tilts()
        if plane == "coronal" and (tilt_ml != 0.0 or tilt_dv != 0.0):
            self.current_atlas_image = normalize_u8(
                coronal_oblique_slice(self.atlas_volume, index, tilt_ml, tilt_dv, order=1)
            )
        else:
            self.current_atlas_image = normalize_u8(atlas_slice(self.atlas_volume, plane, index))
        session = self.current_session()
        overlay_available = (
            session is not None
            and session.transformed_overlay is not None
            and session.atlas_plane == plane
            and session.atlas_index == index
        )
        self.atlas_opacity.setEnabled(overlay_available)
        self.atlas_opacity_value.setEnabled(overlay_available)
        if overlay_available:
            self.atlas_panel.set_base(session.transformed_overlay)
            self.atlas_panel.set_overlay(gray_rgba(self.current_atlas_image), self.atlas_opacity.value() / 100.0)
        else:
            self.atlas_panel.set_base(self.current_atlas_image)
            self.atlas_panel.set_overlay(None)
        self._refresh_points()

    def _load_slice_dialog(self) -> None:
        start = str(self.default_slices_folder) if self.default_slices_folder else ""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select brain slices",
            start,
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)",
        )
        if not paths:
            return
        first_index = len(self.sessions)
        self.slice_list.blockSignals(True)
        for path in paths:
            self.load_slice(Path(path), select=False)
        self.slice_list.setCurrentIndex(first_index)
        self.slice_list.blockSignals(False)
        self.default_slices_folder = Path(paths[0]).parent
        self._switch_slice(first_index)
        self.status.setText(f"Loaded {len(paths)} slices")

    def load_slice(self, path: Path, *, select: bool = True) -> None:
        session = SliceSession(
            name=path.name,
            path=str(path),
            atlas_plane=self.plane_box.currentText(),
            atlas_index=self.section_scroll.value(),
            atlas_tilt_ml_deg=self._current_atlas_tilts()[0],
            atlas_tilt_dv_deg=self._current_atlas_tilts()[1],
        )
        self.sessions.append(session)
        index = len(self.sessions) - 1
        self.slice_list.blockSignals(True)
        self.slice_list.addItem(session.name, session.path)
        self.slice_list.setItemData(index, session.path, QtCore.Qt.ItemDataRole.ToolTipRole)
        self.slice_list.blockSignals(False)
        self._update_auto_order_labels()
        if select:
            self.slice_list.setCurrentIndex(index)
            self._switch_slice(index)
            self.status.setText(f"Loaded slice: {path.name}")
        else:
            self._update_slice_navigation()

    def _load_session_image(self, session: SliceSession) -> bool:
        path = Path(session.path)
        source_changed = (
            session.alignment_source_sha256 is not None
            and file_sha256(path) != session.alignment_source_sha256
        )
        if source_changed:
            self._mark_alignment_run_stale(session, "source image content changed")
            self._clear_slice_transform(session)
        raw = tifffile.imread(str(path)) if path.suffix.lower() in {".tif", ".tiff"} else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        raw = as_gray(raw)
        display_raw, scale = downsample_for_display(raw)
        session.display_scale = scale
        session.raw_display = normalize_u8(display_raw)
        return source_changed

    def _switch_slice(self, index: int) -> None:
        if index < 0 or index >= len(self.sessions):
            return
        previous_index = self.current_session_index
        if 0 <= previous_index < len(self.sessions) and previous_index != index:
            previous = self.sessions[previous_index]
            previous.raw_display = None
            previous.adjusted = None
            previous.rotated = None
            previous.weight_image = None
        self.current_session_index = index
        self.slice_panel.clear_outline_selection()
        session = self.sessions[index]
        source_changed = False
        if session.raw_display is None:
            source_changed = self._load_session_image(session)
        self.rotation.blockSignals(True)
        self.rotation.setValue(session.rotation_deg)
        self.rotation.blockSignals(False)
        self.flip_horizontal.blockSignals(True)
        self.flip_horizontal.setChecked(session.flip_horizontal)
        self.flip_horizontal.blockSignals(False)
        self.flip_vertical.blockSignals(True)
        self.flip_vertical.setChecked(session.flip_vertical)
        self.flip_vertical.blockSignals(False)
        self.curve_editor.set_points(session.curve_points)
        self.curve_editor.set_histogram(session.raw_display)
        self.plane_box.blockSignals(True)
        self.plane_box.setCurrentText(session.atlas_plane)
        self.plane_box.blockSignals(False)
        self._set_plane_limits()
        self.section_scroll.blockSignals(True)
        self.section_scroll.setValue(session.atlas_index)
        self.section_scroll.blockSignals(False)
        self.atlas_tilt_ml.blockSignals(True)
        self.atlas_tilt_dv.blockSignals(True)
        self.atlas_tilt_ml.setValue(round(session.atlas_tilt_ml_deg * 10))
        self.atlas_tilt_dv.setValue(round(session.atlas_tilt_dv_deg * 10))
        self.atlas_tilt_ml.blockSignals(False)
        self.atlas_tilt_dv.blockSignals(False)
        self.atlas_tilt_ml_value.setText(f"{session.atlas_tilt_ml_deg:+.1f}°")
        self.atlas_tilt_dv_value.setText(f"{session.atlas_tilt_dv_deg:+.1f}°")
        self._update_tilt_controls()
        self._update_axis_control(session.atlas_index)
        self._update_slice_image()
        self._update_slice_navigation()
        self._refresh_3d()
        if source_changed:
            self.status.setText(
                f"{session.name} changed on disk; its alignment and derived probe coordinates were invalidated."
            )

    def _step_slice(self, offset: int) -> None:
        index = self.current_session_index + offset
        if 0 <= index < len(self.sessions):
            self.slice_list.setCurrentIndex(index)

    def _update_slice_navigation(self) -> None:
        count = len(self.sessions)
        index = self.current_session_index
        self.slice_position.setText(f"{index + 1 if index >= 0 else 0} / {count}")
        self.previous_slice_btn.setEnabled(index > 0)
        self.next_slice_btn.setEnabled(0 <= index < count - 1)

    def current_session(self) -> SliceSession | None:
        if 0 <= self.current_session_index < len(self.sessions):
            return self.sessions[self.current_session_index]
        return None

    def _active_probe_name(self) -> str:
        return self.probe_name.currentText().strip()

    def _probe_constraint(self, probe_name: str) -> ProbeInsertionConstraint | None:
        return self.probe_constraints.get(probe_name)

    def _effective_probe_constraint(
        self,
        constraint: ProbeInsertionConstraint,
    ) -> ProbeInsertionConstraint:
        physical = PROBE_PHYSICAL_LENGTH_UM[self.probe_type.currentText()]
        maximum = (
            physical
            if constraint.maximum_insertion_depth_um is None
            else min(float(constraint.maximum_insertion_depth_um), physical)
        )
        return ProbeInsertionConstraint(
            enabled=constraint.enabled,
            ap_um=constraint.ap_um,
            ml_um=constraint.ml_um,
            radius_um=constraint.radius_um,
            angle_deg=constraint.angle_deg,
            angle_tolerance_deg=constraint.angle_tolerance_deg,
            maximum_insertion_depth_um=maximum,
        )

    def _set_probe_constraint_controls(self, constraint: ProbeInsertionConstraint | None) -> None:
        constraint = constraint or ProbeInsertionConstraint(
            radius_um=250.0,
            angle_deg=90.0,
            angle_tolerance_deg=5.0,
        )
        self._loading_probe_constraints = True
        try:
            self.use_probe_constraints.setChecked(constraint.enabled)
            self.insertion_ap_um.setValue(round(constraint.ap_um))
            self.insertion_ml_um.setValue(round(constraint.ml_um))
            self.insertion_radius_um.setValue(round(max(constraint.radius_um, 0.0)))
            self.attack_angle_deg.setValue(constraint.angle_deg)
            self.attack_angle_tolerance_deg.setValue(constraint.angle_tolerance_deg)
            self.limit_insertion_depth.setChecked(constraint.maximum_insertion_depth_um is not None)
            if constraint.maximum_insertion_depth_um is not None:
                self.max_insertion_depth_um.setValue(round(constraint.maximum_insertion_depth_um))
        finally:
            self._loading_probe_constraints = False
        self._update_probe_constraint_controls()

    def _probe_constraints_changed(self, *_: object) -> None:
        if self._loading_probe_constraints:
            return
        probe_name = self._active_probe_name()
        if not probe_name:
            return
        previous = self.probe_constraints.get(probe_name)
        updated = ProbeInsertionConstraint(
            enabled=self.use_probe_constraints.isChecked(),
            ap_um=float(self.insertion_ap_um.value()),
            ml_um=float(self.insertion_ml_um.value()),
            radius_um=float(self.insertion_radius_um.value()),
            angle_deg=float(self.attack_angle_deg.value()),
            angle_tolerance_deg=float(self.attack_angle_tolerance_deg.value()),
            maximum_insertion_depth_um=(
                float(self.max_insertion_depth_um.value())
                if self.limit_insertion_depth.isChecked()
                else None
            ),
        )
        self.probe_constraints[probe_name] = updated
        if previous != updated:
            for session in self.sessions:
                self._mark_probe_constrained_alignment_stale(
                    session,
                    probe_name,
                    f"{probe_name} surgical constraint changed",
                )
        self._update_probe_constraint_controls()
        self._update_probe_fit_summary()
        self.status.setText(
            f"{probe_name} surgical constraint updated. Existing results are unchanged; rerun Auto-align current/all to apply it."
        )
        self._refresh_3d()

    def _update_probe_constraint_controls(self) -> None:
        available = bool(self._active_probe_name())
        enabled = available and self.use_probe_constraints.isChecked()
        self.use_probe_constraints.setEnabled(available)
        for control in (
            self.insertion_ap_um,
            self.insertion_ml_um,
            self.insertion_radius_um,
            self.attack_angle_deg,
            self.attack_angle_tolerance_deg,
            self.limit_insertion_depth,
        ):
            control.setEnabled(enabled)
        self.max_insertion_depth_um.setEnabled(enabled and self.limit_insertion_depth.isChecked())

    def _probe_reference_changed(self, *_: object) -> None:
        self.mapping_insertion_depth_um.setEnabled(
            self.endpoint_reference.currentData() == "known_insertion_depth"
        )
        if not self._loading_probe_endpoint and self._active_probe_name():
            self.probe_endpoint_settings[self._active_probe_name()] = (
                self.endpoint_reference.currentData(),
                self.mapping_insertion_depth_um.value(),
            )
        self._update_probe_fit_summary()

    def _mapping_insertion_depth_changed(self, *_: object) -> None:
        if not self._loading_probe_endpoint and self._active_probe_name():
            self.probe_endpoint_settings[self._active_probe_name()] = (
                self.endpoint_reference.currentData(),
                self.mapping_insertion_depth_um.value(),
            )
        self._update_probe_fit_summary()

    def _set_probe_endpoint_controls(self, probe_name: str) -> None:
        mode, depth = self.probe_endpoint_settings.get(probe_name, (None, 3000.0))
        self._loading_probe_endpoint = True
        try:
            self.endpoint_reference.setCurrentIndex(self.endpoint_reference.findData(mode))
            self.mapping_insertion_depth_um.setValue(depth)
        finally:
            self._loading_probe_endpoint = False
        self.mapping_insertion_depth_um.setEnabled(mode == "known_insertion_depth")

    def _probe_type_changed(self, *_: object) -> None:
        for probe_name, constraint in self.probe_constraints.items():
            if not constraint.enabled:
                continue
            for session in self.sessions:
                self._mark_probe_constrained_alignment_stale(
                    session,
                    probe_name,
                    f"{probe_name} physical probe type changed",
                )
        self._update_probe_fit_summary()
        self._refresh_3d()

    def _surface_dv_um(self, ap_um: float, ml_um: float) -> float:
        if self.annotation_volume is None:
            return float("nan")
        ap_index = int(round(ap_um / (VOXEL_UM * STEREOTAXIC_AXIS_SIGN_AP_DV_ML[0]) + self.bregma_voxel[0]))
        ml_index = int(round(ml_um / (VOXEL_UM * STEREOTAXIC_AXIS_SIGN_AP_DV_ML[2]) + self.bregma_voxel[2]))
        if not (0 <= ap_index < self.annotation_volume.shape[0] and 0 <= ml_index < self.annotation_volume.shape[2]):
            return float("nan")
        inside = np.flatnonzero(self.annotation_volume[ap_index, :, ml_index] > 0)
        if not len(inside):
            return float("nan")
        surface_id = int(self.annotation_volume[ap_index, inside.min(), ml_index])
        if surface_id not in self.cortical_region_ids:
            return float("nan")
        return float(volume_to_stereotaxic_um(
            np.array([ap_index, inside.min(), ml_index], dtype=np.float64), self.bregma_voxel
        )[1])

    def _update_probe_fit_summary(self, *_: object) -> None:
        probe_name = self._active_probe_name()
        if not probe_name:
            self.probe_fit_summary.setStyleSheet("color:#9fb4c8;")
            self.probe_fit_summary.setText("Fit: select a probe")
            return
        points = self.all_probe_volume_points(probe_name)
        raw_count = sum(
            len(trace.slice_points)
            for session in self.sessions
            if (trace := session.probe_traces.get(probe_name)) is not None
        )
        try:
            entry, deep_endpoint, surface_direction = self.probe_brain_geometry(probe_name)
        except InfeasibleProbeConstraint as exc:
            self.probe_fit_summary.setStyleSheet("color:#ff8c8c;")
            self.probe_fit_summary.setText(f"Observed fit infeasible: {exc}")
            self.probe_fit_summary.setToolTip(str(exc))
            return
        self.probe_fit_summary.setStyleSheet("color:#9fb4c8;")
        self.probe_fit_summary.setToolTip("")
        if entry is None or deep_endpoint is None or surface_direction is None:
            if raw_count:
                constraint = self._probe_constraint(probe_name)
                suffix = (
                    " Surgical constraints require at least two observations across the selected alignment batch."
                    if constraint is not None and constraint.enabled and raw_count < 2
                    else " Rerun automatic alignment to solve atlas coordinates."
                )
                self.probe_fit_summary.setText(
                    f"Pre-alignment observations: {raw_count}.{suffix}"
                )
            else:
                self.probe_fit_summary.setText("Fit: mark the probe on at least two slice locations")
            return
        stereo_entry = volume_to_stereotaxic_um(entry, self.bregma_voxel)
        stereo_deep = volume_to_stereotaxic_um(deep_endpoint, self.bregma_voxel)
        stereo_deep_direction = -surface_direction * STEREOTAXIC_AXIS_SIGN_AP_DV_ML
        stereo_deep_direction /= np.linalg.norm(stereo_deep_direction)
        angle = np.degrees(np.arcsin(np.clip(abs(stereo_deep_direction[1]), 0.0, 1.0)))
        depth = float(np.linalg.norm(stereo_deep - stereo_entry))
        constraint = self._probe_constraint(probe_name)
        constrained_sessions = [
            session
            for session in self.sessions
            if (geometry := (session.auto_alignment_diagnostics or {}).get(
                "probe_geometry_constraints", {}
            )).get("applied")
            and probe_name in geometry.get("probes", {})
        ]
        if constraint is not None and constraint.enabled:
            applied = bool(constrained_sessions) and not any(
                (session.auto_alignment_diagnostics or {}).get("alignment_run_stale", False)
                for session in constrained_sessions
            )
            lifecycle = (
                " | surgical constraint applied by auto-alignment"
                if applied
                else " | surgical constraint pending — rerun Auto-align"
            )
        else:
            lifecycle = ""
        self.probe_fit_summary.setText(
            f"Observed fit: AP {stereo_entry[0]:+.0f} um | ML {stereo_entry[2]:+.0f} um | "
            f"angle {angle:.1f} deg | deepest mark {depth:.0f} um | {len(points)} points"
            + (
                f" | mapping tip depth {self.mapping_insertion_depth_um.value():.0f} um"
                if self.endpoint_reference.currentData() == "known_insertion_depth"
                else " | mapping tip = deepest mark"
                if self.endpoint_reference.currentData() == "deepest_mark_is_tip"
                else " | choose probe tip location before mapping"
            )
            + lifecycle
        )

    @staticmethod
    def _probe_trace(
        session: SliceSession,
        probe_name: str,
        *,
        create: bool = False,
    ) -> ProbeTrace | None:
        if create:
            return session.probe_traces.setdefault(probe_name, ProbeTrace())
        return session.probe_traces.get(probe_name)

    def _curve_changed(self, points: list[tuple[float, float]]) -> None:
        session = self.current_session()
        if session is None:
            return
        session.curve_points = [(float(x), float(y)) for x, y in points]
        session.adjusted = apply_curve(session.raw_display, session.curve_points)
        session.rotated, _ = transform_slice_image(
            session.adjusted,
            session.rotation_deg,
            session.flip_horizontal,
            session.flip_vertical,
        )
        self.slice_panel.set_base(session.rotated)
        if session.slice_atlas_transform is not None or session.atlas_to_slice_x is not None:
            self._refresh_transformed_overlay(session)
            self._refresh_atlas()
        self.status.setText("Brightness curve updated; alignment coordinates unchanged.")

    def _rotation_changed(self, value: float) -> None:
        session = self.current_session()
        if session is None:
            return
        self._apply_slice_geometry(float(value), self.flip_horizontal.isChecked(), self.flip_vertical.isChecked())

    def _slice_geometry_changed(self) -> None:
        self._apply_slice_geometry(self.rotation.value(), self.flip_horizontal.isChecked(), self.flip_vertical.isChecked())

    def _apply_slice_geometry(self, rotation_deg: float, flip_horizontal: bool, flip_vertical: bool) -> None:
        session = self.current_session()
        if session is None:
            return
        if (
            abs(session.rotation_deg - float(rotation_deg)) < 0.05
            and session.flip_horizontal == bool(flip_horizontal)
            and session.flip_vertical == bool(flip_vertical)
        ):
            return
        old_slice_atlas_transform = session.slice_atlas_transform
        had_transform = (
            session.slice_atlas_transform is not None
            or (session.slice_to_atlas_x is not None and session.atlas_to_slice_x is not None)
        )
        horizontal_changed = session.flip_horizontal != bool(flip_horizontal)
        vertical_changed = session.flip_vertical != bool(flip_vertical)
        rotation_changed = abs(session.rotation_deg - float(rotation_deg)) >= 0.05
        if session.auto_alignment_engine is not None:
            self._mark_alignment_run_stale(session, "contributor slice geometry changed")
        if session.brain_brush_selection_mask is not None and session.raw_display is not None:
            new_shape, new_transform = slice_geometry_matrix(
                session.raw_display.shape,
                rotation_deg,
                flip_horizontal,
                flip_vertical,
            )
            old_to_new = new_transform @ np.linalg.inv(session.slice_transform)
            session.brain_brush_selection_mask = cv2.warpAffine(
                session.brain_brush_selection_mask.astype(np.uint8),
                old_to_new[:2].astype(np.float32),
                (new_shape[1], new_shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        session.rotation_deg = float(rotation_deg)
        session.flip_horizontal = bool(flip_horizontal)
        session.flip_vertical = bool(flip_vertical)
        preserve_auto_transform = had_transform and old_slice_atlas_transform is not None and not rotation_changed
        self._update_slice_image(clear_transform=not preserve_auto_transform)
        if session.brain_brush_strokes:
            self._apply_smart_surface_selection(session, session.brain_brush_strokes)
        if preserve_auto_transform:
            if old_slice_atlas_transform.nonlinear is not None:
                session.slice_atlas_transform = SliceAtlasTransform2D(
                    old_slice_atlas_transform.display_to_affine_atlas_h,
                    session.rotated.shape,
                    old_slice_atlas_transform.atlas_shape,
                )
                diagnostics = dict(session.auto_alignment_diagnostics or {})
                diagnostics["nonlinear_refinement"] = {
                    "requested": True,
                    "status": "invalidated",
                    "reason": "Slice orientation changed after nonlinear anatomical fitting",
                }
                session.auto_alignment_diagnostics = diagnostics
            session.atlas_tilt_ml_deg *= -1.0 if horizontal_changed else 1.0
            session.atlas_tilt_dv_deg *= -1.0 if vertical_changed else 1.0
            session.transformed_overlay = None
            self._recompute_probe_points_from_slice_points(session)
            self._refresh_transformed_overlay(session)
            self._sync_current_pose_controls(session)
            self._refresh_atlas()
            self._refresh_points()
            self._refresh_3d()
            suffix = " Refit internal anatomy." if old_slice_atlas_transform.nonlinear is not None else ""
            self.status.setText(
                "Slice flip changed the atlas orientation; alignment and probe coordinates were recomputed."
                + suffix
            )
            return
        n_pairs = min(len(session.atlas_landmarks), len(session.slice_landmarks))
        if had_transform and n_pairs >= 3 and self._rebuild_slice_transform(session):
            self._recompute_probe_points_from_slice_points(session)
            self._refresh_atlas()
            self._refresh_points()
            self._refresh_3d()
            self.status.setText("Slice geometry changed; transform and probe coordinates were rebuilt.")
        elif had_transform:
            self.status.setText("Slice geometry changed; run auto-align again.")
        else:
            self.status.setText("Slice geometry changed; transform landmarks were moved with the slice.")

    def _sync_current_pose_controls(self, session: SliceSession) -> None:
        self.atlas_tilt_ml.blockSignals(True)
        self.atlas_tilt_dv.blockSignals(True)
        self.atlas_tilt_ml.setValue(round(session.atlas_tilt_ml_deg * 10))
        self.atlas_tilt_dv.setValue(round(session.atlas_tilt_dv_deg * 10))
        self.atlas_tilt_ml.blockSignals(False)
        self.atlas_tilt_dv.blockSignals(False)
        self.atlas_tilt_ml_value.setText(f"{session.atlas_tilt_ml_deg:+.1f}Â°")
        self.atlas_tilt_dv_value.setText(f"{session.atlas_tilt_dv_deg:+.1f}Â°")

    def _update_slice_image(self, *, clear_transform: bool = False) -> None:
        session = self.current_session()
        if session is None or session.raw_display is None:
            return
        session.adjusted = apply_curve(session.raw_display, session.curve_points)
        session.rotated, session.slice_transform = transform_slice_image(
            session.adjusted,
            session.rotation_deg,
            session.flip_horizontal,
            session.flip_vertical,
        )
        session.weight_image, _ = transform_slice_image(
            session.raw_display,
            session.rotation_deg,
            session.flip_horizontal,
            session.flip_vertical,
        )
        if clear_transform:
            self._clear_slice_transform(session)
        for trace in session.probe_traces.values():
            trace.signal_values = [
                self._probe_point_signal(session, point) for point in trace.slice_points
            ]
        self.slice_panel.set_base(session.rotated)
        if not clear_transform and (
            session.slice_atlas_transform is not None or session.atlas_to_slice_x is not None
        ):
            self._refresh_transformed_overlay(session)
        self._refresh_atlas()
        self._refresh_points()

    def _clear_derived_probe_coordinates(self, session: SliceSession) -> None:
        for trace in session.probe_traces.values():
            trace.atlas_points.clear()
            trace.volume_points.clear()

    def _clear_auto_alignment_metadata(self, session: SliceSession) -> None:
        session.auto_alignment_score = None
        session.auto_alignment_global = False
        session.auto_alignment_extent = None
        session.auto_alignment_method = None
        session.auto_alignment_engine = None
        session.auto_alignment_scope = None
        session.auto_alignment_run_id = None
        session.manual_refined_from_run_id = None
        session.auto_alignment_diagnostics = None
        session.alignment_source_sha256 = None
        session.deepslice_raw_ensemble_ouv = None
        session.deepslice_version = None
        session.deepslice_model_hashes = None
        session.deepslice_ensemble_disagreement = None

    def _clear_slice_transform(self, session: SliceSession) -> None:
        session.transformed_overlay = None
        session.slice_atlas_transform = None
        session.slice_to_atlas_x = None
        session.slice_to_atlas_y = None
        session.atlas_to_slice_x = None
        session.atlas_to_slice_y = None
        self._clear_auto_alignment_metadata(session)
        self._clear_derived_probe_coordinates(session)

    @staticmethod
    def _source_binding_error(session: SliceSession) -> str | None:
        if session.alignment_source_sha256 is None:
            return None
        path = Path(session.path)
        if not path.is_file():
            return f"{session.name}: source image is missing"
        if file_sha256(path) != session.alignment_source_sha256:
            return f"{session.name}: source image changed after alignment"
        return None

    def _verify_nonlinear_binding(
        self,
        session: SliceSession,
        transform: SliceAtlasTransform2D,
    ) -> None:
        attestation = transform.nonlinear_attestation
        if transform.nonlinear is None or attestation is None:
            raise RuntimeError(f"{session.name} has no accepted nonlinear warp evidence")
        source_error = self._source_binding_error(session)
        if source_error is not None:
            raise RuntimeError(source_error)
        if file_sha256(Path(session.path)) != attestation.source_image_sha256:
            raise RuntimeError(f"{session.name} nonlinear evidence does not match its source image")
        if attestation.acceptance_diagnostics.get("backend") == "bounded_bspline_mattes_mi_v1":
            classical_model_sha256, classical_manifest_sha256, _ = (
                verify_classical_registration_backend()
            )
            model_sha256, manifest_sha256 = (
                classical_model_sha256,
                classical_manifest_sha256,
            )
        else:
            model_sha256, manifest_sha256, _ = verify_diffeomorphic_model_bundle(
                NONLINEAR_MODEL_PATH
            )
        if (
            attestation.model_sha256 != model_sha256
            or attestation.manifest_sha256 != manifest_sha256
        ):
            raise RuntimeError(f"{session.name} nonlinear evidence does not match the installed model bundle")
        fixed_atlas = coronal_oblique_slice(
            self.atlas_volume,
            session.atlas_index,
            session.atlas_tilt_ml_deg,
            session.atlas_tilt_dv_deg,
            order=1,
        )
        path = Path(session.path)
        raw = (
            tifffile.imread(str(path))
            if path.suffix.lower() in {".tif", ".tiff"}
            else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        )
        display_raw, _ = downsample_for_display(as_gray(raw))
        display_image, _ = transform_slice_image(
            normalize_u8(display_raw),
            session.rotation_deg,
            session.flip_horizontal,
            session.flip_vertical,
        )
        affine = SliceAtlasTransform2D(
            transform.display_to_affine_atlas_h,
            transform.display_shape,
            transform.atlas_shape,
        )
        moving_affine = affine.render_display_image_in_atlas(display_image)
        verify_diffeomorphic_attestation_inputs(
            transform.nonlinear,
            attestation,
            fixed_atlas,
            moving_affine,
        )
        transform.check_invariants()

    def _detach_auto_alignment_for_manual_pose(self, session: SliceSession) -> None:
        source_run_id = session.manual_refined_from_run_id or session.auto_alignment_run_id
        if session.auto_alignment_scope != "manual-refined":
            session.manual_refined_from_run_id = session.auto_alignment_run_id
            session.auto_alignment_run_id = (
                f"{session.auto_alignment_run_id or 'alignment'}_manual_{time.time_ns()}"
            )
            session.auto_alignment_global = False
            session.auto_alignment_scope = "manual-refined"
            session.auto_alignment_method = (
                f"{session.auto_alignment_method} + manual pose refinement"
                if session.auto_alignment_method
                else f"Manual pose refinement derived from automatic-alignment run {source_run_id}"
            )
        diagnostics = dict(session.auto_alignment_diagnostics or {})
        diagnostics["alignment_run_id"] = session.auto_alignment_run_id
        diagnostics["alignment_scope"] = "manual-refined"
        diagnostics["shared_tilt_lr_dv_deg"] = None
        diagnostics["manual_refined_from_run_id"] = source_run_id
        diagnostics["manual_refined_pose"] = {
            "atlas_index": int(session.atlas_index),
            "tilt_lr_deg": float(session.atlas_tilt_ml_deg),
            "tilt_dv_deg": float(session.atlas_tilt_dv_deg),
        }
        if (
            session.slice_atlas_transform is not None
            and session.slice_atlas_transform.nonlinear is not None
        ):
            transform = session.slice_atlas_transform
            session.slice_atlas_transform = SliceAtlasTransform2D(
                transform.display_to_affine_atlas_h,
                transform.display_shape,
                transform.atlas_shape,
            )
            diagnostics["nonlinear_refinement"] = {
                "requested": True,
                "status": "invalidated",
                "reason": "Atlas pose was edited after nonlinear refinement; affine alignment retained",
            }
            session.transformed_overlay = None
            self._recompute_probe_points_from_slice_points(session)
            self._refresh_transformed_overlay(session)
        session.auto_alignment_diagnostics = diagnostics

    def _mark_alignment_run_stale(self, session: SliceSession, reason: str) -> None:
        linked_run_ids = {
            run_id
            for run_id in (session.auto_alignment_run_id, session.manual_refined_from_run_id)
            if run_id is not None
        }
        if not linked_run_ids:
            return
        for member in self.sessions:
            member_run_ids = {member.auto_alignment_run_id, member.manual_refined_from_run_id}
            if linked_run_ids.isdisjoint(member_run_ids):
                continue
            diagnostics = dict(member.auto_alignment_diagnostics or {})
            reasons = list(diagnostics.get("stale_reasons", []))
            if reason not in reasons:
                reasons.append(reason)
            diagnostics["alignment_run_stale"] = True
            diagnostics["stale_reasons"] = reasons
            member.auto_alignment_diagnostics = diagnostics

    def _mark_probe_constrained_alignment_stale(
        self,
        session: SliceSession,
        probe_name: str,
        reason: str,
    ) -> None:
        geometry = (session.auto_alignment_diagnostics or {}).get(
            "probe_geometry_constraints",
            {},
        )
        if geometry.get("applied") and probe_name in geometry.get("probes", {}):
            self._mark_alignment_run_stale(session, reason)

    def _slice_raw_to_display_points(self, session: SliceSession, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return transform_points(points, session.slice_transform)

    def _slice_display_to_raw_point(self, session: SliceSession, point: tuple[float, float]) -> tuple[float, float]:
        inverse = np.linalg.inv(session.slice_transform)
        x, y, _ = inverse @ np.array([point[0], point[1], 1.0], dtype=np.float64)
        return float(x), float(y)

    def _atlas_clicked(self, x: float, y: float) -> None:
        session = self.current_session()
        if session is None:
            return
        if self.smart_surface_mode.isChecked() or self.auto_outline_mode.isChecked() or self.erase_surface_mode.isChecked():
            self.status.setText("Draw the brain outline on the histology panel, then click Auto-align.")
        elif self.landmark_mode.isChecked():
            self._invalidate_transform_after_landmark_edit(session)
            session.atlas_landmarks.append((x, y))
            session.point_history.append("atlas_landmark")
            self.status.setText(f"Added atlas transform landmark {len(session.atlas_landmarks)}")
        else:
            self._add_probe_point(atlas_point=(x, y), slice_raw_point=None)
        self._refresh_points()

    def _slice_clicked(self, x: float, y: float) -> None:
        session = self.current_session()
        if session is None:
            return
        raw_point = self._slice_display_to_raw_point(session, (x, y))
        if self.smart_surface_mode.isChecked():
            return
        if self.auto_outline_mode.isChecked():
            self._insert_surface_point(session, raw_point)
        elif self.landmark_mode.isChecked():
            self._invalidate_transform_after_landmark_edit(session)
            session.slice_landmarks.append(raw_point)
            session.point_history.append("slice_landmark")
            self.status.setText(f"Added slice transform landmark {len(session.slice_landmarks)}")
        else:
            self._add_probe_point(atlas_point=None, slice_raw_point=raw_point)
        self._refresh_points()

    def _invalidate_transform_after_landmark_edit(self, session: SliceSession) -> None:
        if (
            session.slice_atlas_transform is None
            and session.slice_to_atlas_x is None
            and session.atlas_to_slice_x is None
            and session.transformed_overlay is None
        ):
            return
        self._mark_alignment_run_stale(session, "contributor transform landmarks changed")
        self._clear_slice_transform(session)
        self.atlas_panel.set_overlay(None)
        self._refresh_atlas()
        self._refresh_3d()
        self.status.setText("Transform landmarks changed; run transform again before adding new probe points.")

    def _invalidate_auto_alignment_after_surface_edit(self, session: SliceSession) -> None:
        if session.auto_alignment_engine is None:
            return
        self._mark_alignment_run_stale(session, "contributor trusted surface changed")
        self._clear_slice_transform(session)
        if session is self.current_session():
            self.atlas_panel.set_overlay(None)
            self._refresh_atlas()
            self._refresh_3d()

    def _add_probe_point(
        self,
        *,
        atlas_point: tuple[float, float] | None,
        slice_raw_point: tuple[float, float] | None,
    ) -> None:
        session = self.current_session()
        if session is None:
            return
        probe_name = self.probe_name.currentText().strip()
        if not probe_name:
            QtWidgets.QMessageBox.warning(
                self,
                "Probe required",
                "Select the probe whose trajectory point you are adding.",
            )
            return
        aligned = session.slice_atlas_transform is not None or (
            session.slice_to_atlas_x is not None and session.atlas_to_slice_x is not None
        )
        if not aligned:
            if slice_raw_point is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Slice observation required",
                    "Before alignment, mark the probe directly on the histology slice. Atlas coordinates do not exist yet.",
                )
                return
            trace = self._probe_trace(session, probe_name, create=True)
            assert trace is not None
            trace.slice_points.append(slice_raw_point)
            trace.signal_values.append(self._probe_point_signal(session, slice_raw_point))
            session.point_history.append(f"probe:{probe_name}")
            self._update_probe_fit_summary()
            self._refresh_points()
            self.status.setText(
                f"Added pre-alignment {probe_name} observation {len(trace.slice_points)} on {session.name}. "
                "Its atlas/3D position will be solved by the next automatic alignment."
            )
            return
        slice_display_point: tuple[float, float] | None = None
        if atlas_point is None and slice_raw_point is not None:
            slice_display_point = self._slice_raw_to_display_points(session, [slice_raw_point])[0]
            atlas_point = tuple(
                map_session_display_to_atlas(session, np.asarray([slice_display_point]))[0]
            )
        if slice_raw_point is None and atlas_point is not None:
            slice_display_point = tuple(
                map_session_atlas_to_display(session, np.asarray([atlas_point]))[0]
            )
            slice_raw_point = self._slice_display_to_raw_point(session, slice_display_point)
        if atlas_point is None or slice_raw_point is None:
            return
        trace = self._probe_trace(session, probe_name, create=True)
        assert trace is not None
        trace.atlas_points.append(atlas_point)
        trace.slice_points.append(slice_raw_point)
        if self.atlas_volume is not None:
            trace.volume_points.append(
                point_to_volume(
                    atlas_point,
                    session.atlas_plane,
                    session.atlas_index,
                    self.atlas_volume.shape,
                    session.atlas_tilt_ml_deg,
                    session.atlas_tilt_dv_deg,
                ).tolist()
            )
        trace.signal_values.append(self._probe_point_signal(session, slice_raw_point))
        session.point_history.append(f"probe:{probe_name}")
        self._mark_probe_constrained_alignment_stale(
            session,
            probe_name,
            f"{probe_name} trajectory observations changed",
        )
        self._update_probe_fit_summary()
        self._refresh_3d()
        self.status.setText(f"Added {probe_name} trajectory point {len(trace.slice_points)}")

    def _probe_point_signal(self, session: SliceSession, slice_raw_point: tuple[float, float]) -> float:
        if session.weight_image is None:
            return 1.0
        slice_point = self._slice_raw_to_display_points(session, [slice_raw_point])[0]
        x, y = int(round(slice_point[0])), int(round(slice_point[1]))
        image = session.weight_image
        y0 = max(0, y - 3)
        y1 = min(image.shape[0], y + 4)
        x0 = max(0, x - 3)
        x1 = min(image.shape[1], x + 4)
        if x0 >= x1 or y0 >= y1:
            return 1.0
        return float(np.percentile(image[y0:y1, x0:x1], 75))

    def _smart_surface_stroke(self, display_points: list[tuple[float, float]], exclude: bool) -> None:
        session = self.current_session()
        if session is None or session.weight_image is None or not display_points:
            return
        if self.erase_surface_mode.isChecked():
            self._erase_surface_points(session, display_points)
            return
        raw_points = [self._slice_display_to_raw_point(session, point) for point in display_points]
        strokes = [*session.brain_brush_strokes, (bool(exclude), raw_points)]
        self.setCursor(QtCore.Qt.CursorShape.WaitCursor)
        action = "Subtracting from" if exclude else "Growing"
        self.status.setText(f"{action} the smart brain selection from local contrast...")
        try:
            point_count = self._apply_smart_surface_selection(session, strokes)
        except (RuntimeError, cv2.error) as exc:
            self.status.setText(f"Smart surface selection failed: {exc}")
            return
        finally:
            self.unsetCursor()
        session.point_history.append("brain_brush")
        self._invalidate_auto_alignment_after_surface_edit(session)
        operation = "refined" if any(stroke_excludes for stroke_excludes, _ in strokes) else "selected"
        self.status.setText(
            f"Smart brush {operation} the brain object and created {point_count} editable surface points. "
            "Paint again to add evidence or Shift-paint to subtract."
        )
        self._refresh_points()

    def _store_surface_edit_undo(self, session: SliceSession) -> None:
        session.brain_outline_undo_stack.append(
            (
                session.brain_outline_points.copy(),
                session.brain_outline_segment_starts.copy(),
                session.brain_outline_closed,
                [(exclude, points.copy()) for exclude, points in session.brain_brush_strokes],
                None if session.brain_brush_selection_mask is None else session.brain_brush_selection_mask.copy(),
                session.point_history.copy(),
            )
        )

    def _detach_smart_surface(self, session: SliceSession) -> None:
        # Point edits change the trusted boundary, not the tissue mask used for brightness-independent inference.
        session.brain_brush_strokes.clear()
        session.point_history = [action for action in session.point_history if action != "brain_brush"]

    def _surface_point_drag_started(self, index: int) -> None:
        session = self.current_session()
        if session is None or not 0 <= index < len(session.brain_outline_points):
            return
        self._store_surface_edit_undo(session)
        self._detach_smart_surface(session)
        session.point_history.append("brain_outline_edit")
        self._invalidate_auto_alignment_after_surface_edit(session)
        self._refresh_point_counts()

    def _surface_point_moved(self, index: int, x: float, y: float) -> None:
        session = self.current_session()
        if session is None or not 0 <= index < len(session.brain_outline_points):
            return
        session.brain_outline_points[index] = self._slice_display_to_raw_point(session, (x, y))
        self.slice_panel.set_outline(
            self._slice_raw_to_display_points(session, session.brain_outline_points),
            session.brain_outline_segment_starts,
            session.brain_outline_closed,
        )
        self.status.setText(f"Moved surface point {index + 1}; run auto-alignment again")

    def _surface_point_deleted(self, index: int) -> None:
        session = self.current_session()
        if session is None or not 0 <= index < len(session.brain_outline_points):
            return
        self._store_surface_edit_undo(session)
        starts = sorted(set(session.brain_outline_segment_starts or [0]))
        starts = [start for start in starts if 0 <= start < len(session.brain_outline_points)] or [0]
        segments: list[list[tuple[float, float]]] = []
        for segment_index, start in enumerate(starts):
            end = starts[segment_index + 1] if segment_index + 1 < len(starts) else len(session.brain_outline_points)
            segment = session.brain_outline_points[start:end]
            if start <= index < end:
                segment = segment[: index - start] + segment[index - start + 1 :]
            if segment:
                segments.append(segment)
        session.brain_outline_points = [point for segment in segments for point in segment]
        session.brain_outline_segment_starts = [
            int(start) for start in np.cumsum([0] + [len(segment) for segment in segments[:-1]])
        ] or [0]
        self._detach_smart_surface(session)
        session.point_history.append("brain_outline_delete")
        self._invalidate_auto_alignment_after_surface_edit(session)
        self._refresh_points()
        self.status.setText(
            f"Deleted surface point {index + 1}; its neighboring points were reconnected automatically"
        )

    def _insert_surface_point(self, session: SliceSession, raw_point: tuple[float, float]) -> None:
        self._store_surface_edit_undo(session)
        points = session.brain_outline_points
        starts = sorted(set(session.brain_outline_segment_starts or [0]))
        starts = [start for start in starts if 0 <= start < len(points)] or ([0] if points else [])
        display_points = np.asarray(self._slice_raw_to_display_points(session, points), dtype=np.float64)
        display_point = np.asarray(self._slice_raw_to_display_points(session, [raw_point])[0], dtype=np.float64)
        candidates: list[tuple[float, int]] = []
        for segment_index, start in enumerate(starts):
            end = starts[segment_index + 1] if segment_index + 1 < len(starts) else len(points)
            for first in range(start, end - 1):
                a = display_points[first]
                b = display_points[first + 1]
                vector = b - a
                fraction = np.clip(np.dot(display_point - a, vector) / (np.dot(vector, vector) + 1e-12), 0.0, 1.0)
                distance = float(np.linalg.norm(display_point - (a + fraction * vector)))
                candidates.append((distance, first + 1))
        if session.brain_outline_closed and len(starts) == 1 and len(points) > 1:
            a = display_points[-1]
            b = display_points[0]
            vector = b - a
            fraction = np.clip(np.dot(display_point - a, vector) / (np.dot(vector, vector) + 1e-12), 0.0, 1.0)
            distance = float(np.linalg.norm(display_point - (a + fraction * vector)))
            candidates.append((distance, len(points)))
        if candidates:
            insertion_index = min(candidates, key=lambda candidate: candidate[0])[1]
        elif points:
            insertion_index = int(np.argmin(np.linalg.norm(display_points - display_point, axis=1))) + 1
        else:
            insertion_index = 0
        points.insert(insertion_index, raw_point)
        session.brain_outline_segment_starts = [
            start + 1 if start >= insertion_index and start > 0 else start for start in (starts or [0])
        ]
        self._detach_smart_surface(session)
        session.point_history.append("brain_outline_insert")
        self._invalidate_auto_alignment_after_surface_edit(session)
        self.status.setText(
            f"Inserted surface point {insertion_index + 1} between its nearest contour neighbors"
        )

    def _erase_surface_points(
        self,
        session: SliceSession,
        stroke: list[tuple[float, float]],
    ) -> None:
        if not session.brain_outline_points:
            self.status.setText("No surface points to erase on the current slice")
            return
        display = np.asarray(self._slice_raw_to_display_points(session, session.brain_outline_points))
        stroke_points = np.asarray(stroke, dtype=np.float64)
        remove = np.min(np.linalg.norm(display[:, None, :] - stroke_points[None, :, :], axis=2), axis=1) <= self.brush_radius.value()
        if not np.any(remove):
            self.status.setText("The eraser did not touch any surface points")
            return

        self._store_surface_edit_undo(session)
        starts = sorted(set(session.brain_outline_segment_starts or [0]))
        starts = [start for start in starts if 0 <= start < len(remove)] or [0]
        segments: list[list[tuple[float, float]]] = []
        for segment_index, start in enumerate(starts):
            end = starts[segment_index + 1] if segment_index + 1 < len(starts) else len(remove)
            segment_points = session.brain_outline_points[start:end]
            segment_keep = ~remove[start:end]
            if session.brain_outline_closed and len(starts) == 1 and np.any(~segment_keep) and np.any(segment_keep):
                cut = int(np.flatnonzero(~segment_keep)[0])
                order = np.r_[np.arange(cut + 1, len(segment_keep)), np.arange(0, cut + 1)]
                segment_keep = segment_keep[order]
                segment_points = [segment_points[index] for index in order]
            run: list[tuple[float, float]] = []
            for point, keep in zip(segment_points, segment_keep):
                if keep:
                    run.append(point)
                elif run:
                    segments.append(run)
                    run = []
            if run:
                segments.append(run)
        session.brain_outline_points = [point for segment in segments for point in segment]
        session.brain_outline_segment_starts = [
            int(start) for start in np.cumsum([0] + [len(segment) for segment in segments[:-1]])
        ] or [0]
        session.brain_outline_closed = False
        session.brain_brush_strokes.clear()
        session.point_history = [
            action for action in session.point_history if action not in SURFACE_ACTIONS
        ]
        session.point_history.extend("brain_outline" for _ in session.brain_outline_points)
        session.point_history.append("brain_outline_erase")
        self._invalidate_auto_alignment_after_surface_edit(session)
        self._refresh_points()
        self.status.setText(
            f"Removed {int(np.count_nonzero(remove))} unreliable surface points; gaps are excluded from alignment"
        )

    def _compute_smart_surface_selection(
        self,
        session: SliceSession,
        strokes: list[tuple[bool, list[tuple[float, float]]]],
    ) -> tuple[list[tuple[float, float]], np.ndarray | None]:
        foreground_points: list[tuple[float, float]] = []
        background_points: list[tuple[float, float]] = []
        for exclude, raw_points in strokes:
            display_points = self._slice_raw_to_display_points(session, raw_points)
            (background_points if exclude else foreground_points).extend(display_points)
        if not foreground_points:
            return [], None
        surface, selection_mask = smart_brain_surface_selection(
            session.weight_image,
            foreground_points,
            background_points,
            self.brush_radius.value(),
            self.outline_point_count.value(),
        )
        return [self._slice_display_to_raw_point(session, point) for point in surface], selection_mask

    def _apply_smart_surface_selection(
        self,
        session: SliceSession,
        strokes: list[tuple[bool, list[tuple[float, float]]]],
    ) -> int:
        surface, selection_mask = self._compute_smart_surface_selection(session, strokes)
        session.brain_brush_strokes = strokes
        session.brain_outline_points = surface
        session.brain_outline_segment_starts = [0]
        session.brain_outline_closed = bool(surface)
        session.brain_brush_selection_mask = selection_mask
        return len(surface)

    def _outline_point_count_changed(self, point_count: int) -> None:
        session = self.current_session()
        if session is None or session.weight_image is None or not session.brain_outline_closed:
            return
        if session.brain_brush_selection_mask is not None and session.brain_brush_strokes:
            mask = session.brain_brush_selection_mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                return
            surface = resample_closed_contour(max(contours, key=cv2.contourArea), point_count)
            image_height, image_width = session.weight_image.shape[:2]
            mask_height, mask_width = mask.shape
            surface *= (image_width / mask_width, image_height / mask_height)
        else:
            surface = resample_closed_contour(
                np.asarray(self._slice_raw_to_display_points(session, session.brain_outline_points)),
                point_count,
            )
            session.point_history = [action for action in session.point_history if action != "brain_outline"]
            session.point_history.extend("brain_outline" for _ in range(point_count))
        self._invalidate_auto_alignment_after_surface_edit(session)
        session.brain_outline_points = [
            self._slice_display_to_raw_point(session, (float(x), float(y))) for x, y in surface
        ]
        self._refresh_points()
        self.status.setText(f"Resampled the current closed outline to exactly {point_count} evenly spaced points")

    def start_new_surface_segment(self) -> None:
        session = self.current_session()
        if session is None:
            return
        self.alignment_tabs.setCurrentIndex(1)
        self.auto_outline_mode.setChecked(True)
        self._point_target_changed()
        point_count = len(session.brain_outline_points)
        if point_count == 0:
            session.brain_outline_segment_starts = [0]
        elif not session.brain_outline_segment_starts or session.brain_outline_segment_starts[-1] != point_count:
            session.brain_outline_segment_starts.append(point_count)
        session.brain_outline_closed = False
        self.status.setText("Started a separate trusted surface arc; the gap will not be drawn as anatomy.")

    def _update_auto_order_labels(self) -> None:
        previous_order: list[int] = []
        checked: set[int] = set()
        selected_item = self.auto_slice_order.currentItem()
        selected_session_index = None if selected_item is None else selected_item.data(QtCore.Qt.ItemDataRole.UserRole)
        for row in range(self.auto_slice_order.count()):
            item = self.auto_slice_order.item(row)
            session_index = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if session_index is None:
                continue
            session_index = int(session_index)
            previous_order.append(session_index)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                checked.add(session_index)
        outlined_indices = {
            index
            for index, session in enumerate(self.sessions)
            if len(session.brain_outline_points) >= 8
        }
        eligible = [
            index
            for index in previous_order
            if index in outlined_indices
        ]
        eligible.extend(
            index
            for index in range(len(self.sessions))
            if index in outlined_indices and index not in eligible
        )
        self.auto_slice_order.blockSignals(True)
        self.auto_slice_order.clear()
        for session_index in eligible:
            session = self.sessions[session_index]
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.UserRole, session_index)
            item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsDragEnabled
            )
            item.setCheckState(
                QtCore.Qt.CheckState.Checked if session_index in checked else QtCore.Qt.CheckState.Unchecked
            )
            item.setText(session.name)
            item.setToolTip(f"{session.path}\n{len(session.brain_outline_points)} trusted surface points")
            self.auto_slice_order.addItem(item)
            if session_index == selected_session_index:
                self.auto_slice_order.setCurrentItem(item)
        self.auto_slice_order.blockSignals(False)

    def _outlined_auto_sessions(self) -> list[tuple[int, SliceSession]]:
        outlined: list[tuple[int, SliceSession]] = []
        for row in range(self.auto_slice_order.count()):
            item = self.auto_slice_order.item(row)
            session_index = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if (
                session_index is not None
                and 0 <= int(session_index) < len(self.sessions)
                and len(self.sessions[int(session_index)].brain_outline_points) >= 8
            ):
                outlined.append((int(session_index), self.sessions[int(session_index)]))
        return outlined

    def _auto_order_constraint_session_indices(self) -> list[int]:
        constrained: list[int] = []
        for row in range(self.auto_slice_order.count()):
            item = self.auto_slice_order.item(row)
            session_index = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if (
                item.checkState() == QtCore.Qt.CheckState.Checked
                and session_index is not None
                and 0 <= int(session_index) < len(self.sessions)
            ):
                constrained.append(int(session_index))
        return constrained

    def _auto_order_slice_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        session_index = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if session_index is not None and 0 <= int(session_index) < len(self.sessions):
            self.slice_list.setCurrentIndex(int(session_index))

    def _auto_order_constraint_changed(self, *_: object) -> None:
        count = len(self._auto_order_constraint_session_indices())
        self.status.setText(
            f"AP-order constraint uses {count} slice{'s' if count != 1 else ''}."
            if count >= 2
            else "AP-order constraint is off; check at least two slices to enable it."
        )

    def _move_auto_order_item(self, offset: int) -> None:
        row = self.auto_slice_order.currentRow()
        if row < 0 or self.auto_slice_order.count() == 0:
            return
        target = int(np.clip(row + offset, 0, self.auto_slice_order.count() - 1))
        if target == row:
            return
        item = self.auto_slice_order.takeItem(row)
        self.auto_slice_order.insertItem(target, item)
        self.auto_slice_order.setCurrentRow(target)
        self._update_auto_order_labels()
        self._refresh_point_counts()

    def _alignment_tab_changed(self, index: int) -> None:
        self.curve_group.setVisible(index == 0)
        if index == 0:
            self.landmark_mode.setChecked(True)
        else:
            self.smart_surface_mode.setChecked(True)
        self._point_target_changed()
        self._refresh_points()

    def _undo_for_mode(self, button: QtWidgets.QPushButton) -> None:
        if button is self.landmark_mode:
            self.alignment_tabs.setCurrentIndex(0)
        elif button is self.auto_outline_mode:
            self.alignment_tabs.setCurrentIndex(1)
        button.setChecked(True)
        self.undo_last_point()

    def _clear_for_mode(self, button: QtWidgets.QPushButton) -> None:
        if button is self.landmark_mode:
            self.alignment_tabs.setCurrentIndex(0)
        elif button is self.auto_outline_mode:
            self.alignment_tabs.setCurrentIndex(1)
        button.setChecked(True)
        self.clear_current_points()

    def _probe_spots(
        self,
        session: SliceSession,
        *,
        atlas: bool,
    ) -> list[tuple[tuple[float, float], tuple[int, int, int], bool]]:
        selected_probe = self.probe_name.currentText().strip()
        spots = []
        for probe_name, trace in sorted(session.probe_traces.items()):
            points = (
                trace.atlas_points
                if atlas
                else self._slice_raw_to_display_points(session, trace.slice_points)
            )
            spots.extend(
                (point, probe_color(probe_name), probe_name == selected_probe)
                for point in points
            )
        return spots

    def _refresh_points(self) -> None:
        session = self.current_session()
        if session is None:
            self.atlas_panel.set_points([], [])
            self.slice_panel.set_points([], [])
            self.atlas_panel.set_outline([])
            self.slice_panel.set_outline([])
            self.slice_panel.set_selection_mask(None)
            self._refresh_point_counts()
            return
        manual_active = self.alignment_tabs.currentIndex() == 0
        automatic_active = self.alignment_tabs.currentIndex() == 1
        self.atlas_panel.set_points(
            session.atlas_landmarks if manual_active else [],
            self._probe_spots(session, atlas=True),
        )
        self.slice_panel.set_points(
            self._slice_raw_to_display_points(session, session.slice_landmarks) if manual_active else [],
            self._probe_spots(session, atlas=False),
        )
        self.atlas_panel.set_outline([])
        self.slice_panel.set_outline(
            self._slice_raw_to_display_points(session, session.brain_outline_points) if automatic_active else [],
            session.brain_outline_segment_starts,
            session.brain_outline_closed,
        )
        self.slice_panel.set_selection_mask(
            session.brain_brush_selection_mask if automatic_active else None
        )
        self._refresh_point_counts()

    def _refresh_point_counts(self) -> None:
        session = self.current_session()
        if session is None:
            self.point_counts.setText("Surface 0 | Transform atlas 0 / slice 0 | Probe 0")
            self.auto_align_btn.setEnabled(False)
            self.auto_align_all_btn.setEnabled(False)
            self._update_nonlinear_fit_button()
            self.alignment_summary.setText("Auto-align: not run")
            return
        n_pairs = min(len(session.atlas_landmarks), len(session.slice_landmarks))
        selected_probe = self.probe_name.currentText().strip()
        selected_trace = session.probe_traces.get(selected_probe)
        selected_count = 0 if selected_trace is None else len(selected_trace.slice_points)
        total_probe_points = sum(len(trace.slice_points) for trace in session.probe_traces.values())
        probe_count_text = (
            f"Probe {selected_probe} {selected_count} / all probes {total_probe_points}"
            if selected_probe
            else f"Probe not selected / all probes {total_probe_points}"
        )
        self.point_counts.setText(
            f"Surface {len(session.brain_outline_points)} | "
            f"Transform atlas {len(session.atlas_landmarks)} / slice {len(session.slice_landmarks)} ({n_pairs} pairs) | "
            f"{probe_count_text}"
        )
        self.auto_align_btn.setEnabled(
            session.rotated is not None
            and session.weight_image is not None
            and len(session.brain_outline_points) >= 8
            and self.atlas_volume is not None
            and self.annotation_volume is not None
            and self.plane_box.currentText() == "coronal"
            and not self.auto_alignment_busy
        )
        self._update_auto_order_labels()
        self.auto_align_all_btn.setEnabled(
            len(self._outlined_auto_sessions()) >= 2
            and self.atlas_volume is not None
            and self.annotation_volume is not None
            and self.plane_box.currentText() == "coronal"
            and not self.auto_alignment_busy
        )
        self._update_nonlinear_fit_button()
        if session.auto_alignment_engine is not None:
            ap_um = int(round((session.atlas_index - float(self.bregma_voxel[0])) * VOXEL_UM * STEREOTAXIC_AXIS_SIGN_AP_DV_ML[0]))
            scope = session.auto_alignment_scope or ("global" if session.auto_alignment_global else "single")
            diagnostics = session.auto_alignment_diagnostics or {}
            disagreement = diagnostics.get("model_disagreement", {})
            disagreement_text = ""
            if disagreement:
                disagreement_text = (
                    f" | models differ: AP {disagreement['ap_um']:.0f} um, L-R {disagreement['lr_deg']:.1f}°, "
                    f"D-V {disagreement['dv_deg']:.1f}°"
                )
            search_shift = abs(float(diagnostics.get("ap_search_shift_um", 0.0)))
            constraint_text = f" | AP search shift {search_shift:.0f} um" if search_shift >= 0.5 else ""
            scale_text = f" | surface scale {float(diagnostics.get('surface_scale', 1.0)):.3f}x"
            reasons = alignment_review_reasons(disagreement, diagnostics)
            surgical = diagnostics.get("probe_geometry_constraints", {})
            surgical_text = ""
            if surgical.get("applied"):
                surgical_text = " | surgical constraints applied"
                if diagnostics.get("alignment_run_stale", False):
                    surgical_text = " | surgical constraints changed — RERUN AUTO-ALIGN"
            nonlinear = diagnostics.get("nonlinear_refinement", {})
            nonlinear_status = nonlinear.get("status", "not-run")
            nonlinear_text = {
                "accepted": " | experimental nonlinear gates passed",
                "rejected": " | nonlinear rejected; affine retained",
                "invalidated": " | nonlinear invalidated; affine retained",
                "not-run": " | affine-only",
            }.get(nonlinear_status, f" | nonlinear {nonlinear_status}")
            if nonlinear.get("mapping_blocking", False):
                nonlinear_text = " | pose/input rejection; mapping blocked"
            if nonlinear_status == "rejected" and nonlinear.get("reason"):
                reasons.append(str(nonlinear["reason"]))
            review_text = f" — REVIEW: {', '.join(reasons)}" if reasons else ""
            self.alignment_summary.setText(
                f"{session.auto_alignment_engine} {scope}: AP {ap_um:+d} um | L-R {session.atlas_tilt_ml_deg:+.1f}° | "
                f"D-V {session.atlas_tilt_dv_deg:+.1f}°{scale_text}{constraint_text}{disagreement_text}"
                f"{surgical_text}{nonlinear_text}{review_text}"
            )
        else:
            self.alignment_summary.setText("Auto-align: not run")

    def _point_target_changed(self, *_: object) -> None:
        erase_active = self.alignment_tabs.currentIndex() == 1 and self.erase_surface_mode.isChecked()
        point_edit_active = self.alignment_tabs.currentIndex() == 1 and self.auto_outline_mode.isChecked()
        brush_active = self.alignment_tabs.currentIndex() == 1 and (
            self.smart_surface_mode.isChecked() or erase_active
        )
        self.slice_panel.set_brush_erase_only(erase_active)
        self.slice_panel.set_brush_enabled(brush_active)
        self.slice_panel.set_outline_editable(point_edit_active)
        self._refresh_atlas()
        if erase_active:
            self.status.setText("Surface eraser: paint across points on folds, tears, or other unreliable edges.")
        elif brush_active:
            self.status.setText(
                "Smart surface brush: paint over the brain object; hold Shift while painting to subtract."
            )
        elif self.auto_outline_mode.isChecked():
            self.status.setText(
                "Click only trustworthy outer-surface arcs. Use New surface segment across folds, tears, or missing edges."
            )
        elif self.landmark_mode.isChecked():
            self.status.setText("Point target: transform landmarks")
        else:
            selected_probe = self.probe_name.currentText().strip()
            self.status.setText(
                f"Point target: {selected_probe} trajectory"
                if selected_probe
                else "Point target: select a probe before adding trajectory points"
            )

    def _trajectory_weighting_changed(self, enabled: bool) -> None:
        self._refresh_3d()
        self._update_probe_fit_summary()
        self.status.setText("Brightness-weighted trajectory on" if enabled else "Brightness-weighted trajectory off")

    def undo_last_point(self) -> None:
        session = self.current_session()
        if session is None:
            return
        if (
            self.alignment_tabs.currentIndex() == 1
            and session.brain_outline_undo_stack
            and session.point_history
            and session.point_history[-1] in SURFACE_EDIT_ACTIONS
        ):
            points, starts, closed, strokes, selection_mask, point_history = session.brain_outline_undo_stack.pop()
            session.brain_outline_points = points
            session.brain_outline_segment_starts = starts
            session.brain_outline_closed = closed
            session.brain_brush_strokes = strokes
            session.brain_brush_selection_mask = selection_mask
            session.point_history = point_history
            self._invalidate_auto_alignment_after_surface_edit(session)
            self._refresh_points()
            self.status.setText("Undid the last surface point edit")
            return
        if self.smart_surface_mode.isChecked():
            if not session.brain_brush_strokes:
                self.status.setText("No smart brush stroke to undo on current slice")
                return
            strokes = session.brain_brush_strokes[:-1]
            history_index = next(
                (
                    index
                    for index in range(len(session.point_history) - 1, -1, -1)
                    if session.point_history[index] == "brain_brush"
                ),
                None,
            )
            self.setCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                self._apply_smart_surface_selection(session, strokes)
            finally:
                self.unsetCursor()
            if history_index is not None:
                session.point_history.pop(history_index)
            self._invalidate_auto_alignment_after_surface_edit(session)
            self.status.setText("Undid the last smart surface brush stroke")
            self._refresh_points()
            return
        if self.auto_outline_mode.isChecked():
            allowed = {"brain_outline"}
        elif self.landmark_mode.isChecked():
            allowed = {"atlas_landmark", "slice_landmark"}
        else:
            probe_name = self._active_probe_name()
            allowed = {f"probe:{probe_name}"} if probe_name else set()
        history_index = next(
            (index for index in range(len(session.point_history) - 1, -1, -1) if session.point_history[index] in allowed),
            None,
        )
        if history_index is None:
            self.status.setText("No points to undo on current slice")
            return
        action = session.point_history.pop(history_index)
        if action == "brain_outline" and session.brain_outline_points:
            self._invalidate_auto_alignment_after_surface_edit(session)
            session.brain_outline_points.pop()
            point_count = len(session.brain_outline_points)
            session.brain_outline_segment_starts = [
                start for start in session.brain_outline_segment_starts if start < point_count
            ] or [0]
            session.brain_outline_closed = False
            self.status.setText("Undid trusted surface point")
        elif action == "atlas_landmark" and session.atlas_landmarks:
            session.atlas_landmarks.pop()
            self._invalidate_transform_after_landmark_edit(session)
            self.status.setText("Undid atlas transform landmark")
        elif action == "slice_landmark" and session.slice_landmarks:
            session.slice_landmarks.pop()
            self._invalidate_transform_after_landmark_edit(session)
            self.status.setText("Undid slice transform landmark")
        elif action.startswith("probe:"):
            probe_name = action.partition(":")[2]
            trace = session.probe_traces.get(probe_name)
            if trace is not None and trace.slice_points:
                if trace.atlas_points:
                    trace.atlas_points.pop()
                trace.slice_points.pop()
                if trace.volume_points:
                    trace.volume_points.pop()
                if trace.signal_values:
                    trace.signal_values.pop()
                self._mark_probe_constrained_alignment_stale(
                    session,
                    probe_name,
                    f"{probe_name} trajectory observations changed",
                )
                self.status.setText(f"Undid {probe_name} trajectory point")
        self._refresh_atlas()
        self._refresh_points()
        self._update_probe_fit_summary()
        self._refresh_3d()

    def clear_current_points(self) -> None:
        session = self.current_session()
        if session is None:
            return
        if self.smart_surface_mode.isChecked() or self.auto_outline_mode.isChecked() or self.erase_surface_mode.isChecked():
            self._invalidate_auto_alignment_after_surface_edit(session)
            session.brain_outline_points.clear()
            session.brain_outline_segment_starts = [0]
            session.brain_outline_closed = False
            session.brain_brush_strokes.clear()
            session.brain_brush_selection_mask = None
            session.brain_outline_undo_stack.clear()
            session.point_history = [
                action for action in session.point_history if action not in SURFACE_ACTIONS
            ]
            self.status.setText("Cleared the surface selection on the current slice")
        elif self.landmark_mode.isChecked():
            session.atlas_landmarks.clear()
            session.slice_landmarks.clear()
            session.point_history = [action for action in session.point_history if action not in {"atlas_landmark", "slice_landmark"}]
            self._invalidate_transform_after_landmark_edit(session)
            self.status.setText("Cleared transform landmarks on current slice")
        else:
            probe_name = self._active_probe_name()
            trace = session.probe_traces.get(probe_name)
            if trace is not None:
                trace.atlas_points.clear()
                trace.slice_points.clear()
                trace.volume_points.clear()
                trace.signal_values.clear()
                self._mark_probe_constrained_alignment_stale(
                    session,
                    probe_name,
                    f"{probe_name} trajectory observations changed",
                )
            token = f"probe:{probe_name}"
            session.point_history = [action for action in session.point_history if action != token]
            self.status.setText(
                f"Cleared {probe_name} trajectory on current slice"
                if probe_name
                else "Select a probe before clearing trajectory points"
            )
        self._refresh_atlas()
        self._refresh_points()
        self._update_probe_fit_summary()
        self._refresh_3d()

    def transform_current_slice(self) -> None:
        session = self.current_session()
        if session is None or session.rotated is None or self.current_atlas_image is None:
            return
        n = self._rebuild_slice_transform(session)
        if n is None:
            return
        self._recompute_probe_points_from_slice_points(session)
        self.probe_mode.setChecked(True)
        self._refresh_atlas()
        self._refresh_3d()
        self.status.setText(f"Transformed {session.name} using {n} point pairs")

    def _auto_align_clicked(self) -> None:
        try:
            self.auto_align_current_slice()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Auto-alignment failed", str(exc))
            self.status.setText(f"Auto-alignment failed: {exc}")

    def _auto_align_all_clicked(self) -> None:
        try:
            self.auto_align_all_slices()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Global auto-alignment failed", str(exc))
            self.status.setText(f"Global auto-alignment failed: {exc}")

    def _set_auto_constraint_controls_enabled(self, enabled: bool) -> None:
        self.pose_engine.setEnabled(enabled)
        self.own_cnn_weight.setEnabled(enabled and self.pose_engine.currentText() == POSE_ENGINE_WEIGHTED)
        self._update_nonlinear_fit_button(enabled)
        self.limit_auto_align_ap.setEnabled(enabled)
        self.auto_align_ap_min.setEnabled(enabled and self.limit_auto_align_ap.isChecked())
        self.auto_align_ap_max.setEnabled(enabled and self.limit_auto_align_ap.isChecked())
        self.auto_slice_order.setEnabled(enabled)
        self.auto_order_up_btn.setEnabled(enabled)
        self.auto_order_down_btn.setEnabled(enabled)

    def _pose_engine_changed(self, engine: str) -> None:
        self.own_cnn_weight.setEnabled(not self.auto_alignment_busy and engine == POSE_ENGINE_WEIGHTED)

    def _update_nonlinear_fit_button(self, controls_enabled: bool = True) -> None:
        session = self.current_session()
        transform = None if session is None else session.slice_atlas_transform
        self.fit_anatomy_btn.setEnabled(
            controls_enabled
            and not self.auto_alignment_busy
            and self._nonlinear_bundle_error is None
            and transform is not None
            and transform.nonlinear is None
            and session.atlas_plane == "coronal"
            and self.atlas_volume is not None
            and self.annotation_volume is not None
        )

    def _automatic_alignment_image_job(self, session: SliceSession) -> tuple:
        display_outline = self._slice_raw_to_display_points(
            session,
            session.brain_outline_points,
        )
        selection_crop = None
        if (
            session.brain_brush_strokes
            and session.brain_brush_selection_mask is not None
            and np.any(session.brain_brush_selection_mask)
        ):
            selection_y, selection_x = np.nonzero(session.brain_brush_selection_mask)
            selection_crop = surface_crop_bounds(
                [
                    (float(selection_x.min()), float(selection_y.min())),
                    (float(selection_x.max()), float(selection_y.max())),
                ],
                session.brain_brush_selection_mask.shape,
                0.04,
            )
        return (
            session.path,
            session.rotation_deg,
            session.flip_horizontal,
            session.flip_vertical,
            display_outline,
            selection_crop,
            session.brain_outline_closed,
            (
                None
                if session.brain_brush_selection_mask is None
                else session.brain_brush_selection_mask.copy()
            ),
        )

    def _fit_current_slice_anatomy_clicked(self) -> None:
        try:
            self._fit_current_slice_anatomy()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Anatomical fitting failed", str(exc))
            self.status.setText(f"Anatomical fitting failed: {exc}")

    def _fit_current_slice_anatomy(self) -> None:
        session = self.current_session()
        if (
            self.auto_alignment_busy
            or session is None
            or self.atlas_volume is None
            or self.annotation_volume is None
        ):
            return
        transform = session.slice_atlas_transform
        if transform is None:
            raise RuntimeError("Auto-align this slice before fitting its internal anatomy")
        if transform.nonlinear is not None:
            raise RuntimeError("This slice already has an accepted nonlinear anatomical fit")
        if session.atlas_plane != "coronal":
            raise RuntimeError("Nonlinear anatomical fitting currently supports coronal sections only")
        if self._nonlinear_backend == "learned":
            verify_diffeomorphic_model_bundle(NONLINEAR_MODEL_PATH)
        else:
            verify_classical_registration_backend()

        session_index = self.current_session_index
        source_sha256 = file_sha256(Path(session.path))
        atlas_snapshot = (
            id(self.atlas_volume),
            id(self.annotation_volume),
            self.atlas_volume.shape,
            tuple(self.bregma_voxel),
        )
        geometry_snapshot = (
            id(transform),
            session.atlas_index,
            session.atlas_tilt_ml_deg,
            session.atlas_tilt_dv_deg,
            session.rotation_deg,
            session.flip_horizontal,
            session.flip_vertical,
            source_sha256,
        )
        messages: queue.SimpleQueue = queue.SimpleQueue()
        cancel_event = threading.Event()
        progress = QtWidgets.QProgressDialog(
            f"Fitting {session.name} to the fixed atlas plane...",
            "Cancel",
            0,
            100,
            self,
        )
        progress.setWindowTitle("Nonlinear anatomical fit")
        progress.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.canceled.connect(cancel_event.set)
        progress.show()

        self.auto_alignment_busy = True
        self._set_auto_constraint_controls_enabled(False)
        self._refresh_point_counts()
        self.status.setText(
            f"Fitting internal anatomy for {session.name}; AP and tilts remain fixed and the interface remains available."
        )
        future = self.alignment_executor.submit(
            fit_slice_anatomy_to_atlas,
            self._automatic_alignment_image_job(session),
            transform,
            session.atlas_index,
            session.atlas_tilt_ml_deg,
            session.atlas_tilt_dv_deg,
            self.atlas_volume,
            self.annotation_volume,
            str(NONLINEAR_MODEL_PATH) if self._nonlinear_backend == "learned" else None,
            messages,
            cancel_event,
        )
        timer = QtCore.QTimer(self)
        timer.setInterval(100)

        def poll() -> None:
            while True:
                try:
                    value, label = messages.get_nowait()
                except queue.Empty:
                    break
                progress.setValue(value)
                progress.setLabelText(label)
            if not future.done():
                return
            timer.stop()
            try:
                if cancel_event.is_set():
                    raise InterruptedError
                fitted_transform, nonlinear_result = future.result()
                if session_index >= len(self.sessions) or self.sessions[session_index] is not session:
                    raise RuntimeError("The slice list changed while anatomical fitting was running; result discarded")
                current_atlas = (
                    id(self.atlas_volume),
                    id(self.annotation_volume),
                    None if self.atlas_volume is None else self.atlas_volume.shape,
                    tuple(self.bregma_voxel),
                )
                current_geometry = (
                    id(session.slice_atlas_transform),
                    session.atlas_index,
                    session.atlas_tilt_ml_deg,
                    session.atlas_tilt_dv_deg,
                    session.rotation_deg,
                    session.flip_horizontal,
                    session.flip_vertical,
                    file_sha256(Path(session.path)),
                )
                if current_atlas != atlas_snapshot or current_geometry != geometry_snapshot:
                    raise RuntimeError("The slice or atlas changed while anatomical fitting was running; result discarded")

                if nonlinear_result["status"] == "accepted":
                    self._verify_nonlinear_binding(session, fitted_transform)
                diagnostics = dict(session.auto_alignment_diagnostics or {})
                diagnostics["nonlinear_refinement"] = nonlinear_result
                session.auto_alignment_diagnostics = diagnostics
                session.slice_atlas_transform = fitted_transform
                session.transformed_overlay = None
                if nonlinear_result["status"] == "accepted":
                    suffix = " + nonlinear anatomical fit"
                    if not (session.auto_alignment_method or "").endswith(suffix):
                        session.auto_alignment_method = (session.auto_alignment_method or "Affine alignment") + suffix
                    self._recompute_probe_points_from_slice_points(session)
                    if self.current_session_index == session_index:
                        self._refresh_transformed_overlay(session)
                        self._refresh_atlas()
                    self._refresh_3d()
                    self.status.setText(
                        f"Anatomical fit accepted for {session.name}; AP and tilts were kept fixed."
                    )
                else:
                    self.status.setText(
                        f"Anatomical fit rejected for {session.name}; the affine alignment was retained: "
                        f"{nonlinear_result['reason']}"
                    )
            except InterruptedError:
                self.status.setText("Anatomical fitting cancelled; the affine alignment was kept unchanged.")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Anatomical fitting failed", str(exc))
                self.status.setText(f"Anatomical fitting failed: {exc}")
            finally:
                self._finish_auto_alignment_ui()

        timer.timeout.connect(poll)
        self._alignment_timer = timer
        self._alignment_progress = progress
        self._alignment_cancel_event = cancel_event
        timer.start()

    def auto_align_current_slice(self) -> None:
        session = self.current_session()
        if session is None:
            return
        if len(session.brain_outline_points) < 8:
            QtWidgets.QMessageBox.warning(
                self,
                "Surface required",
                "Select at least 8 reliable outer-surface points before automatic alignment.",
            )
            return
        self._start_auto_alignment([self.current_session_index], global_alignment=False)

    def auto_align_all_slices(self) -> None:
        outlined_indices = [index for index, _ in self._outlined_auto_sessions()]
        if len(outlined_indices) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "Outlined slices required",
                "Select at least 8 reliable surface points on at least two slices.",
            )
            return
        self._start_auto_alignment(outlined_indices, global_alignment=True)

    def _start_auto_alignment(self, session_indices: list[int], *, global_alignment: bool) -> None:
        if self.auto_alignment_busy or self.atlas_volume is None or self.annotation_volume is None or not session_indices:
            return
        if self.plane_box.currentText() != "coronal":
            QtWidgets.QMessageBox.warning(
                self,
                "Coronal sections only",
                "Automatic pose models currently support coronal mouse-brain sections only.",
            )
            return

        session_indices = list(dict.fromkeys(session_indices))
        if any(len(self.sessions[index].brain_outline_points) < 8 for index in session_indices):
            QtWidgets.QMessageBox.warning(
                self,
                "Surface required",
                "Every slice in an automatic alignment must have at least 8 trusted surface points.",
            )
            return

        ap_bounds = self._auto_align_index_bounds() if self.limit_auto_align_ap.isChecked() else None
        order_snapshot = [
            index
            for index in self._auto_order_constraint_session_indices()
            if global_alignment and index in session_indices
        ]
        if len(order_snapshot) < 2:
            order_snapshot = []
        engine = self.pose_engine.currentText()
        own_cnn_weight = self.own_cnn_weight.value() / 100.0
        if engine in (POSE_ENGINE_OWN_CNN, POSE_ENGINE_WEIGHTED) and (
            not OWN_CNN_MODEL_PATH.is_file() or not OWN_CNN_MODEL_PATH.with_suffix(".json").is_file()
        ):
            raise RuntimeError(
                "AtlasPose model bundle is unavailable. Expected atlas_pose.onnx and atlas_pose.json at: "
                f"{OWN_CNN_MODEL_PATH.parent}. "
                "Choose DeepSlice until the evaluated model is installed."
            )
        alignment_run_id = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        atlas_snapshot = (
            id(self.atlas_volume),
            id(self.annotation_volume),
            str(self.atlas_folder),
            self.atlas_volume.shape,
            tuple(self.bregma_voxel),
        )
        image_jobs = []
        filename_to_session: dict[str, int] = {}
        geometry_snapshot: dict[int, tuple] = {}
        outline_snapshot: dict[int, list[tuple[float, float]]] = {}
        enabled_probe_constraints = {
            name: self._effective_probe_constraint(constraint)
            for name, constraint in self.probe_constraints.items()
            if constraint.enabled
        }
        insufficient_probe_observations = {
            name: sum(
                len(trace.slice_points)
                for session_index in session_indices
                if (trace := self.sessions[session_index].probe_traces.get(name)) is not None
            )
            for name in enabled_probe_constraints
        }
        insufficient_probe_observations = {
            name: count
            for name, count in insufficient_probe_observations.items()
            if count < 2
        }
        if insufficient_probe_observations:
            details = ", ".join(
                f"{name}: {count}" for name, count in sorted(insufficient_probe_observations.items())
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Probe observations required",
                "Enabled surgical constraints are solved from probe marks on the histology. "
                "Add at least two marks for every enabled probe within the slices being aligned. "
                f"Current counts: {details}.",
            )
            return
        probe_constraints_snapshot = tuple(sorted(enabled_probe_constraints.items()))
        probe_constraint_snapshot = {
            name: {
                "constraint": constraint,
                "display_points_by_session": {
                    session_index: self._slice_raw_to_display_points(
                        self.sessions[session_index],
                        trace.slice_points,
                    )
                    for session_index in session_indices
                    if (
                        trace := self.sessions[session_index].probe_traces.get(name)
                    ) is not None
                    and trace.slice_points
                },
            }
            for name, constraint in enabled_probe_constraints.items()
        }
        def alignment_input_snapshot(session: SliceSession) -> tuple:
            source = Path(session.path)
            source_stat = source.stat()
            selection_digest = (
                None
                if session.brain_brush_selection_mask is None
                else hashlib.sha256(
                    np.ascontiguousarray(session.brain_brush_selection_mask).view(np.uint8)
                ).hexdigest()
            )
            return (
                session.path,
                source_stat.st_size,
                source_stat.st_mtime_ns,
                session.rotation_deg,
                session.flip_horizontal,
                session.flip_vertical,
                tuple(session.brain_outline_points),
                tuple(session.brain_outline_segment_starts),
                session.brain_outline_closed,
                session.atlas_plane,
                session.atlas_index,
                session.atlas_tilt_ml_deg,
                session.atlas_tilt_dv_deg,
                tuple(session.atlas_landmarks),
                tuple(session.slice_landmarks),
                selection_digest,
                id(session.slice_atlas_transform),
                id(session.slice_to_atlas_x),
                id(session.atlas_to_slice_x),
                tuple(
                    (
                        name,
                        tuple(session.probe_traces.get(name, ProbeTrace()).slice_points),
                    )
                    for name in sorted(enabled_probe_constraints)
                ),
                tuple(
                    (name, tuple(sorted(asdict(constraint).items())))
                    for name, constraint in sorted(enabled_probe_constraints.items())
                ),
            )

        for sequence, session_index in enumerate(session_indices):
            session = self.sessions[session_index]
            filename = f"slice_{sequence:04d}.png"
            image_jobs.append(self._automatic_alignment_image_job(session))
            filename_to_session[filename] = session_index
            geometry_snapshot[session_index] = alignment_input_snapshot(session)
            outline_snapshot[session_index] = image_jobs[-1][4]

        messages: queue.SimpleQueue = queue.SimpleQueue()
        cancel_event = threading.Event()
        progress = QtWidgets.QProgressDialog(
            f"Preparing {engine}...",
            "Cancel",
            0,
            100,
            self,
        )
        progress.setWindowTitle("Automatic alignment")
        progress.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.canceled.connect(cancel_event.set)
        progress.show()

        self.auto_alignment_busy = True
        self._set_auto_constraint_controls_enabled(False)
        self._refresh_point_counts()
        scope = f"{len(session_indices)} outlined slices" if global_alignment else self.sessions[session_indices[0]].name
        constraint_names = ", ".join(sorted(enabled_probe_constraints))
        constraint_text = (
            f" with surgical geometry from {constraint_names}"
            if constraint_names
            else ""
        )
        order_text = " + partial AP order" if order_snapshot else ""
        bounds_text = " + AP bounds" if ap_bounds is not None else ""
        shared_text = " + shared tilt" if global_alignment else ""
        self.status.setText(
            f"{engine} and atlas refinement are jointly aligning {scope}{constraint_text}"
            f"{bounds_text}{order_text}{shared_text}; the interface remains available."
        )
        future = self.alignment_executor.submit(
            prepare_run_and_solve_alignment,
            image_jobs,
            filename_to_session,
            float(self.bregma_voxel[0]),
            self.atlas_volume,
            self.annotation_volume,
            outline_snapshot,
            ap_bounds,
            order_snapshot,
            alignment_run_id,
            global_alignment,
            engine,
            own_cnn_weight,
            messages,
            cancel_event,
            probe_constraint_snapshot,
            self.bregma_voxel.copy(),
            frozenset(self.cortical_region_ids),
        )
        timer = QtCore.QTimer(self)
        timer.setInterval(100)

        def poll() -> None:
            while True:
                try:
                    value, label = messages.get_nowait()
                except queue.Empty:
                    break
                progress.setValue(value)
                progress.setLabelText(label)
            if not future.done():
                return
            timer.stop()
            was_cancelled = cancel_event.is_set()
            try:
                if was_cancelled:
                    raise InterruptedError
                (
                    completed_engine,
                    component_provenance,
                    disagreement,
                    runtime_info,
                    prepared,
                    shared_tilt,
                ) = future.result()
                current_atlas = (
                    id(self.atlas_volume),
                    id(self.annotation_volume),
                    str(self.atlas_folder),
                    None if self.atlas_volume is None else self.atlas_volume.shape,
                    tuple(self.bregma_voxel),
                )
                if current_atlas != atlas_snapshot:
                    raise RuntimeError("The atlas changed while automatic alignment was running; result discarded")
                current_constraints = tuple(
                    (name, self._effective_probe_constraint(constraint))
                    for name, constraint in sorted(self.probe_constraints.items())
                    if constraint.enabled
                )
                if current_constraints != probe_constraints_snapshot:
                    raise RuntimeError(
                        "A surgical probe constraint changed while automatic alignment was running; result discarded"
                    )
                for session_index, expected in geometry_snapshot.items():
                    session = self.sessions[session_index]
                    if alignment_input_snapshot(session) != expected:
                        raise RuntimeError(
                            f"{session.name} was edited while automatic alignment was running; result discarded"
                        )
                self._apply_auto_alignment_results(
                    prepared,
                    completed_engine,
                    component_provenance,
                    disagreement,
                    runtime_info,
                    ap_bounds,
                    order_snapshot,
                    global_alignment=global_alignment,
                    shared_tilt=shared_tilt,
                )
            except InterruptedError:
                self.status.setText("Automatic alignment cancelled; previous results were kept unchanged.")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Automatic alignment failed", str(exc))
                self.status.setText(f"Automatic alignment failed: {exc}")
            finally:
                self._finish_auto_alignment_ui()

        timer.timeout.connect(poll)
        self._alignment_timer = timer
        self._alignment_progress = progress
        self._alignment_cancel_event = cancel_event
        timer.start()

    def _finish_auto_alignment_ui(self) -> None:
        timer, self._alignment_timer = self._alignment_timer, None
        progress, self._alignment_progress = self._alignment_progress, None
        self._alignment_cancel_event = None
        self.auto_alignment_busy = False
        if timer is not None:
            timer.stop()
            timer.timeout.disconnect()
            timer.deleteLater()
        if progress is not None:
            progress.blockSignals(True)
            progress.close()
            progress.deleteLater()
        self._set_auto_constraint_controls_enabled(True)
        self._refresh_point_counts()

    def _apply_auto_alignment_results(
        self,
        prepared: list[tuple],
        engine: str,
        component_provenance: dict,
        disagreement: dict[str, dict[str, float]],
        runtime_info: dict,
        ap_bounds: tuple[int, int] | None,
        order_snapshot: list[int],
        *,
        global_alignment: bool,
        shared_tilt: tuple[float, float] | None,
    ) -> None:
        if self.atlas_volume is None or self.annotation_volume is None:
            raise RuntimeError("Atlas was unloaded while automatic alignment was running")
        if not prepared:
            raise RuntimeError("Automatic alignment returned no slices")
        batch_names = [
            self.sessions[index].name
            for index in prepared[0][6]["alignment_batch_session_indices"]
        ]
        order_names = [self.sessions[index].name for index in order_snapshot]
        staged = []
        for (
            session_index,
            atlas_index,
            tilt_ml,
            tilt_dv,
            transform,
            prediction,
            original_diagnostics,
        ) in prepared:
            session = self.sessions[session_index]
            diagnostics = dict(original_diagnostics)
            if transform.atlas_shape != tuple(self.atlas_volume.shape[1:]):
                raise RuntimeError(f"{session.name} transform does not match the loaded atlas canvas")
            source_sha256 = diagnostics["input_crop"]["source_image_sha256"]
            if file_sha256(Path(session.path)) != source_sha256:
                raise RuntimeError(f"{session.name} changed on disk while automatic alignment was running")
            diagnostics["source_image_sha256"] = source_sha256
            diagnostics["raw_model_ap_um"] = float(diagnostics["raw_model_pose_ap_um_lr_deg_dv_deg"][0])
            diagnostics["refined_ap_um"] = float(
                (diagnostics["refined_ap_index"] - float(self.bregma_voxel[0]))
                * VOXEL_UM
                * STEREOTAXIC_AXIS_SIGN_AP_DV_ML[0]
            )
            diagnostics["ap_search_shift_um"] = float(
                diagnostics.pop("ap_search_shift_index")
                * VOXEL_UM
                * STEREOTAXIC_AXIS_SIGN_AP_DV_ML[0]
            )
            diagnostics["alignment_batch_slices"] = batch_names
            diagnostics["order_constraint_anterior_to_posterior"] = order_names
            diagnostics["shared_tilt_lr_dv_deg"] = (
                None if shared_tilt is None else [float(shared_tilt[0]), float(shared_tilt[1])]
            )
            nonlinear_result = diagnostics.setdefault(
                "nonlinear_refinement",
                {
                    "requested": False,
                    "status": "not-run",
                    "reason": "Nonlinear refinement was not requested",
                },
            )
            diagnostics["component_provenance"] = component_provenance
            diagnostics["model_disagreement"] = dict(
                disagreement[Path(str(prediction["Filenames"])).name]
            )
            probe_updates = {}
            for probe_name, trace in session.probe_traces.items():
                display_points = self._slice_raw_to_display_points(session, trace.slice_points)
                atlas_array = (
                    transform.map_display_to_atlas(np.asarray(display_points, dtype=np.float64))
                    if display_points
                    else np.empty((0, 2), dtype=np.float64)
                )
                if not np.isfinite(atlas_array).all():
                    raise RuntimeError(f"{session.name} produced non-finite probe coordinates")
                atlas_points = [tuple(map(float, point)) for point in atlas_array]
                volume_points = [
                    point_to_volume(
                        point,
                        "coronal",
                        atlas_index,
                        self.atlas_volume.shape,
                        tilt_ml,
                        tilt_dv,
                    ).tolist()
                    for point in atlas_points
                ]
                probe_updates[probe_name] = (atlas_points, volume_points)
            deepslice_component = prediction["component_predictions"].get(POSE_ENGINE_DEEPSLICE)
            deepslice_provenance = component_provenance.get(POSE_ENGINE_DEEPSLICE)
            staged.append(
                (
                    session,
                    int(atlas_index),
                    float(tilt_ml),
                    float(tilt_dv),
                    transform,
                    diagnostics,
                    probe_updates,
                    float(diagnostics["surface_rms_after_atlas_px"]),
                    nonlinear_result["status"] == "accepted",
                    None
                    if deepslice_component is None
                    else list(deepslice_component["raw_ensemble_ouv_quicknii_ml_ap_dv"]),
                    None if deepslice_provenance is None else deepslice_provenance.get("version"),
                    None
                    if deepslice_provenance is None
                    else dict(deepslice_provenance.get("model_sha256", {})),
                    None
                    if deepslice_component is None
                    else dict(deepslice_component["ensemble_disagreement"]),
                )
            )

        for probe_name, constraint in self.probe_constraints.items():
            if not constraint.enabled:
                continue
            observations = {
                session.name: volume_to_stereotaxic_um(
                    np.asarray(probe_updates[probe_name][1], dtype=np.float64),
                    self.bregma_voxel,
                )
                for session, _, _, _, _, diagnostics, probe_updates, *_ in staged
                if probe_name in probe_updates
                and probe_updates[probe_name][1]
                and diagnostics.get("probe_geometry_constraints", {})
                .get("probes", {})
                .get(probe_name, {})
                .get("trajectory_observations_used", True)
            }
            if sum(len(points) for points in observations.values()) < 2:
                continue
            observed_fit = fit_observed_probe_ray(
                observations,
                self._effective_probe_constraint(constraint),
                self._surface_dv_um,
            )
            for _, _, _, _, _, diagnostics, _, *_ in staged:
                geometry = diagnostics.get("probe_geometry_constraints", {})
                if geometry.get("applied") and probe_name in geometry.get("probes", {}):
                    geometry["probes"][probe_name]["post_alignment_observed_fit"] = {
                        "entry_ap_dv_ml_um": observed_fit.entry_ap_dv_ml_um.tolist(),
                        "direction_ap_dv_ml": observed_fit.direction_ap_dv_ml.tolist(),
                        "angle_deg": float(observed_fit.angle_deg),
                        "diagnostics": observed_fit.diagnostics,
                    }

        for (
            session,
            atlas_index,
            tilt_ml,
            tilt_dv,
            transform,
            diagnostics,
            probe_updates,
            alignment_score,
            nonlinear_accepted,
            deepslice_raw,
            deepslice_version,
            deepslice_hashes,
            deepslice_disagreement,
        ) in staged:
            session.atlas_plane = "coronal"
            session.atlas_index = atlas_index
            session.atlas_tilt_ml_deg = tilt_ml
            session.atlas_tilt_dv_deg = tilt_dv
            session.slice_atlas_transform = transform
            session.slice_to_atlas_x = None
            session.slice_to_atlas_y = None
            session.atlas_to_slice_x = None
            session.atlas_to_slice_y = None
            session.auto_alignment_score = alignment_score
            session.auto_alignment_global = global_alignment
            session.auto_alignment_extent = "internal_anatomy_and_trusted_surface"
            session.auto_alignment_method = (
                f"{engine} initialization + bounded MIND atlas search + trusted-surface calibration"
                + (" with exact shared-tilt integration" if global_alignment else "")
                + (
                    " + experimental nonlinear anatomical refinement"
                    if nonlinear_accepted
                    else ""
                )
            )
            session.auto_alignment_engine = engine
            session.auto_alignment_scope = "global" if global_alignment else "single"
            session.auto_alignment_run_id = diagnostics["alignment_run_id"]
            session.manual_refined_from_run_id = None
            session.auto_alignment_diagnostics = diagnostics
            session.alignment_source_sha256 = diagnostics["source_image_sha256"]
            session.deepslice_raw_ensemble_ouv = deepslice_raw
            session.deepslice_version = deepslice_version
            session.deepslice_model_hashes = deepslice_hashes
            session.deepslice_ensemble_disagreement = deepslice_disagreement
            session.transformed_overlay = None
            for probe_name, (atlas_points, volume_points) in probe_updates.items():
                trace = session.probe_traces[probe_name]
                trace.atlas_points = atlas_points
                trace.volume_points = volume_points

        self._switch_slice(self.current_session_index)
        self._refresh_3d()
        review_sessions = [
            self.sessions[session_index]
            for session_index, *_ in prepared
            if alignment_review_reasons(
                self.sessions[session_index].auto_alignment_diagnostics.get("model_disagreement", {}),
                self.sessions[session_index].auto_alignment_diagnostics,
            )
        ]
        nonlinear_results = [
            session.auto_alignment_diagnostics["nonlinear_refinement"]
            for session, *_ in staged
        ]
        accepted_count = sum(result["status"] == "accepted" for result in nonlinear_results)
        rejected = [result for result in nonlinear_results if result["status"] == "rejected"]
        blocked = [result for result in rejected if result.get("mapping_blocking", False)]
        if accepted_count == len(nonlinear_results):
            nonlinear_text = " Experimental nonlinear refinement passed runtime gates."
        elif blocked:
            nonlinear_text = (
                f" Pose/input rejection for {len(blocked)} slice(s); mapping is blocked until alignment is rerun: "
                f"{blocked[0]['reason']}."
            )
        elif rejected:
            nonlinear_text = (
                f" Experimental nonlinear refinement rejected for {len(rejected)} slice(s); affine results were retained for review: "
                f"{rejected[0]['reason']}."
            )
        else:
            nonlinear_text = f" Experimental nonlinear refinement not run: {nonlinear_results[0]['reason']}."
        if global_alignment:
            order_text = " + AP order" if len(order_snapshot) >= 2 else ""
            bounds_text = " + AP range" if ap_bounds is not None else ""
            pre_spread = runtime_info.get("preintegration_tilt_spread_deg", [0.0, 0.0])
            review_text = (
                f" REVIEW {len(review_sessions)} slice(s)."
                if review_sessions
                else ""
            )
            self.status.setText(
                f"{engine} + atlas search aligned {len(prepared)} outlined slices{bounds_text}{order_text}; exact shared L-R "
                f"{shared_tilt[0]:+.1f}°, D-V {shared_tilt[1]:+.1f}° (independent spread before integration "
                f"{pre_spread[0]:.1f}° / {pre_spread[1]:.1f}°; {runtime_info.get('device', 'unknown')})."
                f"{review_text}{nonlinear_text}"
            )
        else:
            session = self.sessions[prepared[0][0]]
            diagnostics = session.auto_alignment_diagnostics
            disagreement = diagnostics.get("model_disagreement", {})
            shift = diagnostics["ap_search_shift_um"]
            shift_text = f", AP search shift {shift:+.0f} um" if abs(shift) >= 0.5 else ""
            reasons = alignment_review_reasons(disagreement, diagnostics)
            review_text = f" REVIEW: {', '.join(reasons)}." if reasons else ""
            self.status.setText(
                f"{engine} + atlas search aligned {session.name}: AP {self._ap_index_to_um(session.atlas_index):+d} um, "
                f"L-R {session.atlas_tilt_ml_deg:+.1f}°, D-V {session.atlas_tilt_dv_deg:+.1f}°, "
                f"surface scale {diagnostics['surface_scale']:.3f}x{shift_text}; "
                f"{runtime_info.get('device', 'unknown')}.{review_text}{nonlinear_text}"
            )

    def _rebuild_slice_transform(self, session: SliceSession) -> int | None:
        if session.rotated is None or self.current_atlas_image is None:
            return None
        n = min(len(session.atlas_landmarks), len(session.slice_landmarks))
        if n < 3:
            QtWidgets.QMessageBox.warning(self, "More points needed", "Add at least 3 corresponding points on the atlas and slice.")
            return None
        atlas_points = np.asarray(session.atlas_landmarks[:n], dtype=np.float64)
        slice_points = np.asarray(self._slice_raw_to_display_points(session, session.slice_landmarks[:n]), dtype=np.float64)
        session.slice_to_atlas_x = Rbf(slice_points[:, 0], slice_points[:, 1], atlas_points[:, 0], function="thin_plate", smooth=0.0)
        session.slice_to_atlas_y = Rbf(slice_points[:, 0], slice_points[:, 1], atlas_points[:, 1], function="thin_plate", smooth=0.0)
        session.atlas_to_slice_x = Rbf(atlas_points[:, 0], atlas_points[:, 1], slice_points[:, 0], function="thin_plate", smooth=0.0)
        session.atlas_to_slice_y = Rbf(atlas_points[:, 0], atlas_points[:, 1], slice_points[:, 1], function="thin_plate", smooth=0.0)
        session.slice_atlas_transform = None
        self._clear_auto_alignment_metadata(session)
        self._refresh_transformed_overlay(session)
        return n

    def _refresh_transformed_overlay(self, session: SliceSession) -> None:
        if session.rotated is None or self.current_atlas_image is None:
            return
        if session.slice_atlas_transform is None and session.atlas_to_slice_x is None:
            return
        session.transformed_overlay = render_session_slice_in_atlas(
            session,
            session.rotated,
            self.current_atlas_image.shape,
        )

    def _recompute_probe_points_from_slice_points(self, session: SliceSession) -> None:
        if session.slice_atlas_transform is None and session.slice_to_atlas_x is None:
            return
        for trace in session.probe_traces.values():
            display_points = self._slice_raw_to_display_points(session, trace.slice_points)
            trace.atlas_points = [
                tuple(map(float, point))
                for point in map_session_display_to_atlas(session, np.asarray(display_points))
            ] if display_points else []
        self._recompute_session_volume_points(session)
        self._update_probe_fit_summary()

    def _recompute_session_volume_points(self, session: SliceSession) -> None:
        if self.atlas_volume is None:
            return
        for trace in session.probe_traces.values():
            trace.volume_points = [
                point_to_volume(
                    point,
                    session.atlas_plane,
                    session.atlas_index,
                    self.atlas_volume.shape,
                    session.atlas_tilt_ml_deg,
                    session.atlas_tilt_dv_deg,
                ).tolist()
                for point in trace.atlas_points
            ]
        self._update_probe_fit_summary()

    def all_probe_volume_points(self, probe_name: str) -> np.ndarray:
        points = []
        for session in self.sessions:
            if (session.auto_alignment_diagnostics or {}).get(
                "nonlinear_refinement", {}
            ).get("mapping_blocking", False):
                continue
            trace = session.probe_traces.get(probe_name)
            if trace is not None:
                points.extend(trace.volume_points)
        return np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def all_probe_signal_values(self, probe_name: str) -> np.ndarray:
        values = []
        for session in self.sessions:
            if (session.auto_alignment_diagnostics or {}).get(
                "nonlinear_refinement", {}
            ).get("mapping_blocking", False):
                continue
            trace = session.probe_traces.get(probe_name)
            if trace is not None:
                values.extend(trace.signal_values)
        return np.asarray(values, dtype=np.float64)

    def probe_regression_weights(self, probe_name: str, n_points: int) -> np.ndarray:
        if not self.brightness_weighting.isChecked():
            return np.ones(n_points, dtype=np.float64)
        values = self.all_probe_signal_values(probe_name)
        if len(values) != n_points or n_points < 2:
            return np.ones(n_points, dtype=np.float64)
        values = np.nan_to_num(values, nan=np.nanmedian(values), posinf=np.nanmax(values), neginf=np.nanmin(values))
        lo, hi = np.percentile(values, [10, 95])
        if hi <= lo:
            return np.ones(n_points, dtype=np.float64)
        normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        return 0.15 + 0.85 * normalized**2

    def probe_regression(self, probe_name: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        points = self.all_probe_volume_points(probe_name)
        if len(points) < 2:
            return None, None
        weights = self.probe_regression_weights(probe_name, len(points))
        center = np.average(points, axis=0, weights=weights)
        _, _, vh = np.linalg.svd((points - center) * np.sqrt(weights[:, None]), full_matrices=False)
        direction = vh[0]
        direction = direction / np.linalg.norm(direction)
        if direction[1] > 0:
            direction = -direction
        return center, direction

    def probe_brain_geometry(self, probe_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        points = self.all_probe_volume_points(probe_name)
        center, surface_direction = self.probe_regression(probe_name)
        if center is None or surface_direction is None or len(points) < 2:
            return None, None, None
        projection = (points - center) @ surface_direction
        deep_endpoint = center + surface_direction * projection.min()
        entry_parameter = projection.max()
        if self.annotation_volume is not None:
            radius = int(np.ceil(np.linalg.norm(self.annotation_volume.shape)))
            parameters = np.arange(-radius, radius + 1, dtype=np.float64)
            ray = center[None, :] + parameters[:, None] * surface_direction[None, :]
            indices = np.rint(ray).astype(int)
            in_bounds = np.all((indices >= 0) & (indices < np.asarray(self.annotation_volume.shape)), axis=1)
            inside = np.zeros(len(parameters), dtype=bool)
            valid = indices[in_bounds]
            inside[in_bounds] = self.annotation_volume[valid[:, 0], valid[:, 1], valid[:, 2]] > 0
            if np.any(inside):
                entry_parameter = max(entry_parameter, float(parameters[inside].max()))
        entry = center + surface_direction * entry_parameter
        physical_length_voxels = PROBE_PHYSICAL_LENGTH_UM[self.probe_type.currentText()] / VOXEL_UM
        depth_from_entry = float((entry - deep_endpoint) @ surface_direction)
        if depth_from_entry > physical_length_voxels:
            raise InfeasibleProbeConstraint(
                f"Observed trajectory exceeds the selected probe's "
                f"{PROBE_PHYSICAL_LENGTH_UM[self.probe_type.currentText()] / 1000.0:g} mm physical shank"
            )
        return entry, deep_endpoint, surface_direction

    def probe_line_geometry(self, probe_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        entry, deep_endpoint, surface_direction = self.probe_brain_geometry(probe_name)
        if entry is None or deep_endpoint is None or surface_direction is None:
            return None, None, None
        above_brain = entry + surface_direction * 20.0
        return above_brain, deep_endpoint, surface_direction

    def _refresh_3d(self) -> None:
        for item in self.dynamic_gl_items:
            self.view3d.removeItem(item)
        self.dynamic_gl_items.clear()
        if self.atlas_volume is not None and self.sessions:
            aligned_sessions = [
                (index, session)
                for index, session in enumerate(self.sessions)
                if session.brain_outline_points
                and (
                    session.slice_atlas_transform is not None
                    or (
                        session.slice_to_atlas_x is not None
                        and session.slice_to_atlas_y is not None
                        and session.atlas_to_slice_x is not None
                        and session.atlas_to_slice_y is not None
                    )
                )
            ]
            if self.show_all_slice_planes.isChecked():
                visible_sessions = aligned_sessions
            else:
                visible_sessions = [
                    item for item in aligned_sessions if item[0] == self.current_session_index
                ]
            palette = [
                (0.15, 0.85, 1.0),
                (1.0, 0.55, 0.18),
                (0.62, 0.42, 1.0),
                (0.22, 0.92, 0.58),
                (1.0, 0.34, 0.67),
                (0.88, 0.88, 0.22),
            ]
            for session_index, session in visible_sessions:
                is_current = session_index == self.current_session_index
                rgb = (0.15, 0.85, 1.0) if is_current else palette[session_index % len(palette)]
                corners = volume_to_gl(
                    section_plane_corners(
                        self.atlas_volume.shape,
                        session.atlas_plane,
                        session.atlas_index,
                        session.atlas_tilt_ml_deg,
                        session.atlas_tilt_dv_deg,
                    )
                )
                plane_item = gl.GLMeshItem(
                    vertexes=corners,
                    faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
                    color=(*rgb, 0.22 if is_current else 0.11),
                    smooth=False,
                    shader="shaded",
                )
                plane_item.setGLOptions("translucent")
                outline = gl.GLLinePlotItem(
                    pos=np.vstack([corners, corners[0]]),
                    color=(*rgb, 1.0 if is_current else 0.72),
                    width=2 if is_current else 1,
                    antialias=True,
                )
                self.view3d.addItem(plane_item)
                self.view3d.addItem(outline)
                self.dynamic_gl_items.extend([plane_item, outline])
                label = f"{session_index + 1}: {session.name}"
                for label_path in plane_label_paths(corners, label):
                    label_item = gl.GLLinePlotItem(
                        pos=label_path,
                        color=(*rgb, 1.0),
                        width=1.5,
                        antialias=True,
                    )
                    self.view3d.addItem(label_item)
                    self.dynamic_gl_items.append(label_item)
        probe_names = sorted(
            {
                probe_name
                for session in self.sessions
                for probe_name, trace in session.probe_traces.items()
                if trace.volume_points
            }
        )
        selected_probe = self._active_probe_name()
        for probe_name in probe_names:
            points = self.all_probe_volume_points(probe_name)
            if len(points) == 0:
                continue
            selected = probe_name == selected_probe
            rgb = np.asarray(probe_color(probe_name), dtype=np.float32) / 255.0
            alpha = 1.0 if selected else 0.58
            weights = self.probe_regression_weights(probe_name, len(points))
            scatter = gl.GLScatterPlotItem(
                pos=volume_to_gl(points),
                color=(*rgb, alpha),
                size=(7.0 if selected else 5.0) + (6.0 if selected else 3.0) * weights,
            )
            self.view3d.addItem(scatter)
            self.dynamic_gl_items.append(scatter)
            try:
                above_brain, deep_endpoint, _ = self.probe_line_geometry(probe_name)
            except InfeasibleProbeConstraint:
                continue
            if above_brain is None or deep_endpoint is None:
                continue
            line = gl.GLLinePlotItem(
                pos=volume_to_gl(np.vstack([above_brain, deep_endpoint])),
                color=(*rgb, alpha),
                width=4 if selected else 2,
                antialias=True,
            )
            endpoints = gl.GLScatterPlotItem(
                pos=volume_to_gl(np.vstack([above_brain, deep_endpoint])),
                color=(*rgb, alpha),
                size=np.asarray([11.0, 13.0] if selected else [8.0, 10.0], dtype=np.float32),
            )
            self.view3d.addItem(line)
            self.view3d.addItem(endpoints)
            self.dynamic_gl_items.extend([line, endpoints])

        for constraint_probe, constraint in sorted(self.probe_constraints.items()):
            if not constraint.enabled:
                continue
            selected = constraint_probe == selected_probe
            constraint = self._effective_probe_constraint(constraint)
            rgb = np.asarray(probe_color(constraint_probe), dtype=np.float32) / 255.0
            warning = " | target area partly outside dorsal cortex"
            if selected and warning in self.probe_fit_summary.text():
                self.probe_fit_summary.setText(self.probe_fit_summary.text().replace(warning, ""))
                self.probe_fit_summary.setToolTip("")
            angles = np.linspace(0.0, 2.0 * np.pi, 97)
            ring_stereotaxic = np.column_stack(
                [
                    constraint.ap_um + constraint.radius_um * np.cos(angles),
                    np.zeros_like(angles),
                    constraint.ml_um + constraint.radius_um * np.sin(angles),
                ]
            )
            ring_stereotaxic[:, 1] = [
                self._surface_dv_um(ap_um, ml_um)
                for ap_um, ml_um in ring_stereotaxic[:, (0, 2)]
            ]
            valid_ring = np.isfinite(ring_stereotaxic[:, 1])
            for valid_indices in np.split(
                np.flatnonzero(valid_ring),
                np.flatnonzero(np.diff(np.flatnonzero(valid_ring)) > 1) + 1,
            ) if np.any(valid_ring) else ():
                if len(valid_indices) < 2:
                    continue
                ring_volume = probe_stereotaxic_to_volume(
                    ring_stereotaxic[valid_indices], self.bregma_voxel, VOXEL_UM
                )
                ring_item = gl.GLLinePlotItem(
                    pos=volume_to_gl(ring_volume),
                    color=(*rgb, 0.9 if selected else 0.5),
                    width=2 if selected else 1,
                    antialias=True,
                )
                self.view3d.addItem(ring_item)
                self.dynamic_gl_items.append(ring_item)
            if selected and not np.all(valid_ring):
                if warning not in self.probe_fit_summary.text():
                    self.probe_fit_summary.setText(self.probe_fit_summary.text() + warning)
                self.probe_fit_summary.setToolTip(
                    "Part of the insertion-radius boundary lies outside the dorsal Isocortex surface; "
                    "the 3D view shows only anatomically valid boundary arcs."
                )

            angle_low = max(0.0, constraint.angle_deg - constraint.angle_tolerance_deg)
            angle_high = min(90.0, constraint.angle_deg + constraint.angle_tolerance_deg)
            depth_um = min(float(constraint.maximum_insertion_depth_um), 4000.0)
            cone_paths = []
            entry_dv_um = self._surface_dv_um(constraint.ap_um, constraint.ml_um)
            if np.isfinite(entry_dv_um):
                for angle_deg, alpha in ((angle_low, 0.45), (angle_high, 0.75)):
                    horizontal = depth_um * np.cos(np.deg2rad(angle_deg))
                    ventral = depth_um * np.sin(np.deg2rad(angle_deg))
                    for azimuth in np.linspace(0.0, 2.0 * np.pi, 9)[:-1]:
                        endpoint = np.array(
                            [
                                constraint.ap_um + horizontal * np.cos(azimuth),
                                entry_dv_um - ventral,
                                constraint.ml_um + horizontal * np.sin(azimuth),
                            ]
                        )
                        path = np.vstack(
                            [
                                [constraint.ap_um, entry_dv_um, constraint.ml_um],
                                endpoint,
                            ]
                        )
                        volume = probe_stereotaxic_to_volume(path, self.bregma_voxel, VOXEL_UM)
                        cone_paths.append((volume, alpha))
            elif selected:
                if warning not in self.probe_fit_summary.text():
                    self.probe_fit_summary.setText(self.probe_fit_summary.text() + warning)
                self.probe_fit_summary.setToolTip(
                    "The planned insertion center is outside the dorsal Isocortex surface; "
                    "valid parts of the uncertainty boundary remain visible."
                )
            for volume, alpha in cone_paths:
                cone_item = gl.GLLinePlotItem(
                    pos=volume_to_gl(volume),
                    color=(*rgb, alpha if selected else alpha * 0.55),
                    width=1,
                    antialias=True,
                )
                self.view3d.addItem(cone_item)
                self.dynamic_gl_items.append(cone_item)

    def _resolve_data_folder(self, run_folder: Path) -> Path:
        if (run_folder / "channels.csv").exists() and (run_folder / "units.csv").exists():
            return run_folder
        return run_folder / "preprocessed_data"

    def _refresh_probe_names(self) -> None:
        selected = self.probe_name.currentText()
        blocker = QtCore.QSignalBlocker(self.probe_name)
        self.probe_name.clear()
        data_folder = self._resolve_data_folder(Path(self.run_folder.text().strip()))
        channels_path = data_folder / "channels.csv"
        if not channels_path.exists():
            del blocker
            self._probe_selection_changed()
            return
        try:
            channels = canonical_channel_keys(pd.read_csv(channels_path))
        except Exception as exc:
            del blocker
            self.status.setText(f"Could not read probe names: {exc}")
            self._probe_selection_changed()
            return
        names = sorted(channels["probe_name"].dropna().astype(str).unique())
        for name in names:
            color = probe_color(name)
            swatch = QtGui.QPixmap(12, 12)
            swatch.fill(QtGui.QColor(*color))
            self.probe_name.addItem(QtGui.QIcon(swatch), name)
        if selected in names:
            self.probe_name.setCurrentText(selected)
        del blocker
        self._probe_selection_changed()

    def _probe_selection_changed(self, *_: object) -> None:
        probe_name = self._active_probe_name()
        self.probe_mode.setEnabled(bool(probe_name))
        self.map_btn.setEnabled(bool(probe_name))
        self.probe_undo_point_btn.setText(
            f"Undo {probe_name} point" if probe_name else "Undo trajectory point"
        )
        self.probe_clear_points_btn.setText(
            f"Clear {probe_name} trajectory" if probe_name else "Clear trajectory"
        )
        self._set_probe_constraint_controls(self._probe_constraint(probe_name))
        self._set_probe_endpoint_controls(probe_name)
        self._refresh_points()
        self._refresh_3d()
        self._update_probe_fit_summary()
        if self.probe_mode.isChecked():
            self._point_target_changed()

    def _sample_region(self, coord: np.ndarray) -> tuple[int | None, str, str, tuple[int | None, int | None, int | None]]:
        if self.annotation_volume is None:
            return None, "", "", (None, None, None)
        index = np.rint(np.asarray(coord, dtype=float)).astype(int)
        if np.any(index < 0) or np.any(index >= np.asarray(self.annotation_volume.shape)):
            return None, "", "", (None, None, None)
        ap, dv, ml = (int(value) for value in index)
        region_id = int(self.annotation_volume[ap, dv, ml])
        if region_id == 0:
            return None, "", "", (ap, dv, ml)
        name, acronym = self.region_names.get(region_id, (str(region_id), ""))
        return region_id, name, acronym, (ap, dv, ml)

    def map_channels_units(self) -> None:
        run_folder = Path(self.run_folder.text().strip())
        data_folder = self._resolve_data_folder(run_folder)
        channels_path = data_folder / "channels.csv"
        units_path = data_folder / "units.csv"
        if not channels_path.exists() or not units_path.exists():
            QtWidgets.QMessageBox.warning(self, "CSV files missing", f"Missing channels.csv or units.csv in:\n{data_folder}")
            return
        selected_probe = self._active_probe_name()
        if not selected_probe:
            QtWidgets.QMessageBox.warning(self, "Probe missing", "Select imec0, imec1, or another available probe.")
            return
        probe_sessions = [
            session
            for session in self.sessions
            if (
                (trace := self._probe_trace(session, selected_probe)) is not None
                and trace.volume_points
            )
        ]
        stale_sessions = [
            session.name
            for session in probe_sessions
            if (session.auto_alignment_diagnostics or {}).get("alignment_run_stale", False)
        ]
        if stale_sessions:
            QtWidgets.QMessageBox.warning(
                self,
                "Auto-alignment rerun required",
                "A contributor to an automatic alignment run changed after fitting. Rerun alignment before "
                f"mapping these probe-bearing slices: {', '.join(stale_sessions)}",
            )
            return
        source_errors = [
            error
            for session in probe_sessions
            if (error := self._source_binding_error(session)) is not None
        ]
        if source_errors:
            QtWidgets.QMessageBox.warning(
                self,
                "Source image changed",
                "Rerun alignment before mapping:\n" + "\n".join(source_errors),
            )
            return
        blocked_sessions = [
            session.name
            for session in probe_sessions
            if (session.auto_alignment_diagnostics or {})
            .get("nonlinear_refinement", {})
            .get("mapping_blocking", False)
        ]
        if blocked_sessions:
            QtWidgets.QMessageBox.warning(
                self,
                "Alignment rejected",
                "The nonlinear stage rejected the pose or affine input for these probe-bearing slices. "
                "Rerun alignment before mapping: " + ", ".join(blocked_sessions),
            )
            return
        review_sessions = [
            session
            for session in probe_sessions
            if (
                (result := (session.auto_alignment_diagnostics or {}).get("nonlinear_refinement", {}))
                .get("status") == "rejected"
                and not result.get("mapping_blocking", False)
                and not result.get("affine_review_acknowledged_at")
            )
        ]
        if review_sessions:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Review affine fallback",
                "Experimental nonlinear refinement was rejected for "
                + ", ".join(session.name for session in review_sessions)
                + ". The affine alignments remain usable, but inspect their overlays before mapping. "
                "Continue with the reviewed affine results?",
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                self.status.setText("Mapping cancelled; affine fallback was not approved.")
                return
            acknowledged_at = datetime.now().astimezone().isoformat()
            for session in review_sessions:
                diagnostics = dict(session.auto_alignment_diagnostics or {})
                result = dict(diagnostics["nonlinear_refinement"])
                result["affine_review_acknowledged_at"] = acknowledged_at
                diagnostics["nonlinear_refinement"] = result
                session.auto_alignment_diagnostics = diagnostics
        points = self.all_probe_volume_points(selected_probe)
        try:
            brain_entry, deepest_mark, surface_direction = self.probe_brain_geometry(selected_probe)
        except InfeasibleProbeConstraint as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Surgical constraint infeasible",
                str(exc),
            )
            return
        if brain_entry is None or deepest_mark is None or surface_direction is None or len(points) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "Probe line missing",
                f"Add at least two {selected_probe} points before mapping channels.",
            )
            return
        if self.probe_type.currentText() == "Neuropixels 2.0 four-shank":
            QtWidgets.QMessageBox.warning(
                self,
                "Four-shank mapping needs orientation",
                "This trajectory fit has no probe-roll measurement, so four-shank contacts cannot be assigned "
                "to atlas regions unambiguously.",
            )
            return
        endpoint_reference = self.endpoint_reference.currentData()
        if endpoint_reference is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Probe tip location required",
                "Choose either the known surgical insertion depth or 'deepest marked dot is the probe tip'.",
            )
            return

        channels = canonical_channel_keys(pd.read_csv(channels_path))
        units = canonical_channel_keys(pd.read_csv(units_path), units=True)
        selected = channels["probe_name"].eq(selected_probe)
        if not selected.any():
            QtWidgets.QMessageBox.warning(self, "Probe missing", f"{selected_probe} is not present in {channels_path}")
            return
        if "probe_vertical_position" not in channels.columns:
            raise ValueError("channels.csv has no probe_vertical_position/y_um geometry")
        distance_um = pd.to_numeric(
            channels.loc[selected, "probe_vertical_position"], errors="raise"
        ).to_numpy(dtype=float)
        try:
            physical_tip, y0_contact, coords, insertion_depth_um, observed_depth_um = probe_mapping_coordinates(
                brain_entry,
                deepest_mark,
                surface_direction,
                str(endpoint_reference),
                self.mapping_insertion_depth_um.value()
                if endpoint_reference == "known_insertion_depth"
                else None,
                distance_um,
                self.probe_type.currentText(),
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Probe geometry infeasible", str(exc))
            return
        sampled = [self._sample_region(coord) for coord in coords]
        stereotaxic = np.asarray([volume_to_stereotaxic_um(coord, self.bregma_voxel) for coord in coords])
        mapped_at = datetime.now().isoformat(timespec="seconds")
        assignments = {
            "structure_id": [item[0] for item in sampled],
            "structure_name": [item[1] for item in sampled],
            "structure_acronym": [item[2] for item in sampled],
            "ccf_ap_index": [item[3][0] for item in sampled],
            "ccf_dv_index": [item[3][1] for item in sampled],
            "ccf_ml_index": [item[3][2] for item in sampled],
            "atlas_region_id": [item[0] for item in sampled],
            "atlas_region": [item[1] for item in sampled],
            "atlas_acronym": [item[2] for item in sampled],
            "atlas_ap": [item[3][0] for item in sampled],
            "atlas_dv": [item[3][1] for item in sampled],
            "atlas_ml": [item[3][2] for item in sampled],
            "stereotaxic_ap_um": stereotaxic[:, 0],
            "stereotaxic_dv_um": stereotaxic[:, 1],
            "stereotaxic_ml_um": stereotaxic[:, 2],
            "trajectory_distance_um": PROBE_TIP_TO_Y0_CONTACT_UM[self.probe_type.currentText()] + distance_um,
            "probe_type": self.probe_type.currentText(),
            "anatomy_source": "proprietary_trajectory_tracker",
            "anatomy_assignment_method": "peak_channel_on_trajectory_centerline",
            "anatomy_mapped_at": mapped_at,
        }
        for name, values in assignments.items():
            if name not in channels.columns:
                channels[name] = pd.Series(pd.NA, index=channels.index, dtype="object")
            channels.loc[selected, name] = values

        units = attach_peak_channel_metadata(channels, units)
        selected_units = units["probe_name"].eq(selected_probe)
        if not units.loc[selected_units, "structure_acronym"].fillna("").astype(str).str.len().gt(0).any():
            QtWidgets.QMessageBox.warning(
                self,
                "No atlas structures assigned",
                f"The {selected_probe} trajectory did not intersect a labelled atlas structure. No files were changed.",
            )
            return

        staging_root = Path(tempfile.mkdtemp(prefix=".trajectory_mapping_", dir=data_folder))
        try:
            write_csv_atomic(channels, staging_root / "channels.csv")
            write_csv_atomic(units, staging_root / "units.csv")
            write_anatomy_sidecars(staging_root, channels, units)
            self._write_manifest(
                staging_root,
                selected_probe,
                str(endpoint_reference),
                brain_entry,
                deepest_mark,
                physical_tip,
                y0_contact,
                surface_direction,
                insertion_depth_um,
                observed_depth_um,
            )
            staged_hashes = verify_staged_mapping_outputs(
                staging_root,
                len(channels),
                len(units),
                selected_probe,
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_dir = data_folder / "anatomy" / "backups" / f"{timestamp}_{selected_probe}"
            promote_staged_mapping_outputs(
                staging_root,
                data_folder,
                backup_dir,
                staged_hashes,
            )
        finally:
            if staging_root.parent.resolve() != data_folder.resolve():
                raise RuntimeError("Refusing to remove a mapping stage outside the data folder")
            shutil.rmtree(staging_root, ignore_errors=True)
        mapped_channels = channels.loc[selected, "structure_acronym"].fillna("").astype(str).str.len().gt(0).sum()
        mapped_units = units.loc[selected_units, "structure_acronym"].fillna("").astype(str).str.len().gt(0).sum()
        self.status.setText(
            f"Mapped {selected_probe}: {mapped_channels}/{selected.sum()} channels, "
            f"{mapped_units}/{selected_units.sum()} units; assignments saved by peak channel"
        )

    def undo_file_mapping(self) -> None:
        run_folder = Path(self.run_folder.text().strip())
        data_folder = self._resolve_data_folder(run_folder)
        paths = [data_folder / "channels.csv", data_folder / "units.csv"]
        existing_paths = [path for path in paths if path.exists()]
        if not existing_paths:
            QtWidgets.QMessageBox.warning(self, "CSV files missing", f"Missing channels.csv and units.csv in:\n{data_folder}")
            return

        tables: list[tuple[Path, pd.DataFrame, list[str]]] = []
        for path in existing_paths:
            table = pd.read_csv(path)
            drop_cols = [col for col in ANATOMY_MAPPING_COLUMNS if col in table.columns]
            if drop_cols:
                tables.append((path, table, drop_cols))

        if not tables:
            self.status.setText(f"No anatomy mapping columns found in {data_folder}")
            return

        anatomy_dir = data_folder / "anatomy"
        sidecars = [
            anatomy_dir / "channel_brain_regions.csv",
            anatomy_dir / "unit_brain_region_assignments.csv",
            *sorted(anatomy_dir.glob("proprietary_trajectory_manifest_*.json")),
            *sorted((anatomy_dir / "slice_atlas_transforms").glob("*.npz")),
        ]
        sidecars = [path for path in sidecars if path.exists()]
        summary = "\n".join(f"{path.name}: {', '.join(cols)}" for path, _, cols in tables)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Undo file mapping",
            "Remove only these anatomy mapping columns?\n\n"
            f"{summary}\n\n"
            "Timestamped backups will be written beside each CSV before any file is changed.",
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            self.status.setText("Undo file mapping cancelled")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        undo_log = {"data_folder": str(data_folder), "timestamp": timestamp, "files": []}
        backup_dir = anatomy_dir / "backups" / f"{timestamp}_undo_all"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for path, table, drop_cols in tables:
            backup = backup_dir / path.name
            shutil.copy2(path, backup)
            write_csv_atomic(table.drop(columns=drop_cols), path)
            undo_log["files"].append({"csv": str(path), "backup": str(backup), "removed_columns": drop_cols})

        for path in sidecars:
            backup = backup_dir / path.name
            shutil.copy2(path, backup)
            path.unlink()
            undo_log["files"].append({"removed": str(path), "backup": str(backup)})

        log_path = anatomy_dir / f"undo_file_mapping_{timestamp}.json"
        log_path.write_text(json.dumps(undo_log, indent=2), encoding="utf-8")
        self.status.setText(f"Removed anatomy mapping; recoverable backups are in {backup_dir}")

    def _write_manifest(
        self,
        data_folder: Path,
        probe_name: str,
        endpoint_mode: str,
        brain_entry: np.ndarray,
        deepest_mark: np.ndarray,
        physical_tip: np.ndarray,
        y0_contact: np.ndarray,
        surface_direction: np.ndarray,
        insertion_depth_um: float,
        observed_depth_um: float,
    ) -> None:
        anatomy_dir = data_folder / "anatomy"
        anatomy_dir.mkdir(exist_ok=True)

        def output_trace(session: SliceSession) -> ProbeTrace:
            return self._probe_trace(session, probe_name) or ProbeTrace()

        for session in self.sessions:
            if not output_trace(session).volume_points:
                continue
            source_error = self._source_binding_error(session)
            if source_error is not None:
                raise RuntimeError(source_error)
            nonlinear_result = (session.auto_alignment_diagnostics or {}).get(
                "nonlinear_refinement", {}
            )
            if nonlinear_result.get("mapping_blocking", False):
                raise RuntimeError(f"{session.name} has a pose/input rejection; rerun alignment")
            if (
                nonlinear_result.get("status") == "rejected"
                and not nonlinear_result.get("affine_review_acknowledged_at")
            ):
                raise RuntimeError(f"{session.name} affine fallback has not been explicitly reviewed")

        nonlinear_sidecars = {}
        for session_index, session in enumerate(self.sessions):
            transform = session.slice_atlas_transform
            if transform is None or transform.nonlinear is None:
                continue
            self._verify_nonlinear_binding(session, transform)
            sidecar_dir = anatomy_dir / "slice_atlas_transforms"
            sidecar_dir.mkdir(exist_ok=True)
            identity = hashlib.sha256(
                f"{session_index}|{Path(session.path).resolve()}".encode("utf-8")
            ).hexdigest()[:16]
            sidecar_path = sidecar_dir / f"slice_atlas_transform_{identity}.npz"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{sidecar_path.stem}_",
                suffix=".npz",
                dir=sidecar_dir,
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            try:
                transform.save_npz(temporary_path)
                restored = SliceAtlasTransform2D.load_npz(temporary_path)
                restored.check_invariants()
                os.replace(temporary_path, sidecar_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
            attestation = transform.nonlinear_attestation
            assert attestation is not None
            nonlinear_sidecars[id(session)] = {
                "relative_path": sidecar_path.relative_to(anatomy_dir).as_posix(),
                "sha256": file_sha256(sidecar_path),
                "coordinate_convention": transform.coordinate_convention,
                "model_sha256": attestation.model_sha256,
                "manifest_sha256": attestation.manifest_sha256,
                "source_image_sha256": attestation.source_image_sha256,
                "atlas_image_sha256": attestation.atlas_image_sha256,
                "moving_affine_sha256": attestation.moving_affine_sha256,
                "runtime_gate_version": attestation.runtime_gate_version,
                "pixel_spacing_um": attestation.pixel_spacing_um,
            }

        alignment_runs = {}
        for session in self.sessions:
            diagnostics = session.auto_alignment_diagnostics or {}
            if session.auto_alignment_run_id is None:
                continue
            run = alignment_runs.setdefault(
                session.auto_alignment_run_id,
                {
                    "scope": diagnostics.get("alignment_scope", session.auto_alignment_scope),
                    "contributors": diagnostics.get("alignment_batch_slices", []),
                    "order_constraint_anterior_to_posterior": diagnostics.get(
                        "order_constraint_anterior_to_posterior",
                        [],
                    ),
                    "ap_search_bounds_index": diagnostics.get("ap_search_bounds_index"),
                    "shared_tilt_lr_dv_deg": diagnostics.get("shared_tilt_lr_dv_deg"),
                    "runtime_backend": diagnostics.get("runtime_backend"),
                    "runtime_device": diagnostics.get("runtime_device"),
                    "onnxruntime_version": diagnostics.get("onnxruntime_version"),
                    "gpu_fallback_reason": diagnostics.get("gpu_fallback_reason"),
                    "stale": bool(diagnostics.get("alignment_run_stale", False)),
                    "stale_reasons": diagnostics.get("stale_reasons", []),
                    "input_snapshot": diagnostics.get("alignment_batch_inputs", {}),
                    "pose_prediction_engine": session.auto_alignment_engine,
                    "pose_prediction_provenance": diagnostics.get("component_provenance"),
                    "prediction_fusion": diagnostics.get("prediction_fusion"),
                    "deepslice_version": session.deepslice_version,
                    "model_sha256": session.deepslice_model_hashes,
                },
            )
            if diagnostics.get("alignment_batch_inputs"):
                run["input_snapshot"] = diagnostics["alignment_batch_inputs"]
        alignment_engines = sorted(
            {
                session.auto_alignment_engine
                for session in self.sessions
                if session.auto_alignment_engine is not None
            }
        )
        manifest = {
            "created_at": datetime.now().astimezone().isoformat(),
            "probe_name": probe_name,
            "atlas_folder": str(self.atlas_folder),
            "atlas_sha256": self.atlas_file_hashes,
            "region_lookup_query_csv_sha256": self.query_file_hash,
            "voxel_um": VOXEL_UM,
            "bregma_um_mlapdv": DEFAULT_BREGMA_UM_ML_AP_DV.tolist(),
            "bregma_voxel_ap_dv_ml": self.bregma_voxel.tolist(),
            "stereotaxic_origin": "bregma",
            "stereotaxic_ap_convention": "0 at bregma; anterior positive; posterior negative",
            "stereotaxic_axis_sign_ap_dv_ml": STEREOTAXIC_AXIS_SIGN_AP_DV_ML.tolist(),
            "probe_type": self.probe_type.currentText(),
            "insertion_constraint": (
                None
                if (constraint := self._probe_constraint(probe_name)) is None
                else asdict(constraint)
            ),
            "observed_probe_fit_uses_surgical_constraint": False,
            "probe_attack_angle_convention": "0 degrees horizontal; 90 degrees vertical",
            "insertion_target_coordinate_frame": (
                "bregma-centred stereotaxic AP/ML um; uncertainty circle projected onto the Allen annotation surface"
            ),
            "channel_identity": ["probe_name", "probe_channel_number"],
            "unit_assignment": "structure_acronym inherited from the unit peak probe channel",
            "trajectory_sampling": "shank centerline at each recording contact; horizontal position is retained but no probe-roll estimate is available",
            "vertical_reference": (
                "probe_vertical_position is relative to the lowest chanMap recording row; "
                "the physical tip offset is supplied by the selected Neuropixels geometry"
            ),
            "probe_tip_location_mode": endpoint_mode,
            "insertion_depth_from_surface_um": insertion_depth_um,
            "deepest_mark_depth_from_surface_um": observed_depth_um,
            "physical_tip_to_y0_contact_um": PROBE_TIP_TO_Y0_CONTACT_UM[self.probe_type.currentText()],
            "brightness_weighted_trajectory": self.brightness_weighting.isChecked(),
            "automatic_alignment_engine": alignment_engines[0] if len(alignment_engines) == 1 else None,
            "automatic_alignment_engines": alignment_engines,
            "automatic_alignment_runs": alignment_runs,
            "brain_entry_voxel_ap_dv_ml": brain_entry.tolist(),
            "brain_entry_stereotaxic_um_ap_dv_ml": volume_to_stereotaxic_um(brain_entry, self.bregma_voxel).tolist(),
            "deepest_mark_voxel_ap_dv_ml": deepest_mark.tolist(),
            "deepest_mark_stereotaxic_um_ap_dv_ml": volume_to_stereotaxic_um(deepest_mark, self.bregma_voxel).tolist(),
            "physical_tip_voxel_ap_dv_ml": physical_tip.tolist(),
            "physical_tip_stereotaxic_um_ap_dv_ml": volume_to_stereotaxic_um(physical_tip, self.bregma_voxel).tolist(),
            "y0_contact_voxel_ap_dv_ml": y0_contact.tolist(),
            "y0_contact_stereotaxic_um_ap_dv_ml": volume_to_stereotaxic_um(y0_contact, self.bregma_voxel).tolist(),
            "surface_direction_ap_dv_ml": surface_direction.tolist(),
            "slices": [
                {
                    "name": session.name,
                    "path": session.path,
                    "alignment_source_sha256": session.alignment_source_sha256,
                    "display_scale": session.display_scale,
                    "rotation_deg": session.rotation_deg,
                    "flip_horizontal": session.flip_horizontal,
                    "flip_vertical": session.flip_vertical,
                    "atlas_plane": session.atlas_plane,
                    "atlas_index": session.atlas_index,
                    "atlas_tilt_ml_deg": session.atlas_tilt_ml_deg,
                    "atlas_tilt_dv_deg": session.atlas_tilt_dv_deg,
                    "atlas_landmarks": session.atlas_landmarks,
                    "slice_landmarks": self._slice_raw_to_display_points(session, session.slice_landmarks),
                    "slice_landmarks_raw": session.slice_landmarks,
                    "brain_outline_points": self._slice_raw_to_display_points(session, session.brain_outline_points),
                    "brain_outline_points_raw": session.brain_outline_points,
                    "brain_outline_segment_starts": session.brain_outline_segment_starts,
                    "brain_outline_closed": session.brain_outline_closed,
                    "brain_brush_strokes_raw": session.brain_brush_strokes,
                    "auto_alignment_method": session.auto_alignment_method,
                    "auto_alignment_score": session.auto_alignment_score,
                    "auto_alignment_affine": (
                        None
                        if session.slice_atlas_transform is None
                        else session.slice_atlas_transform.display_to_affine_atlas_h.tolist()
                    ),
                    "slice_atlas_transform": {
                        "coordinate_convention": session.slice_atlas_transform.coordinate_convention,
                        "nonlinear": session.slice_atlas_transform.nonlinear is not None,
                        "nonlinear_status": (
                            (session.auto_alignment_diagnostics or {})
                            .get("nonlinear_refinement", {})
                            .get("status", "not-run")
                        ),
                        "sidecar": nonlinear_sidecars.get(id(session)),
                    }
                    if session.slice_atlas_transform is not None
                    else None,
                    "auto_alignment_global": session.auto_alignment_global,
                    "auto_alignment_extent": session.auto_alignment_extent,
                    "auto_alignment_engine": session.auto_alignment_engine,
                    "auto_alignment_scope": session.auto_alignment_scope,
                    "auto_alignment_run_id": session.auto_alignment_run_id,
                    "manual_refined_from_run_id": session.manual_refined_from_run_id,
                    "auto_alignment_diagnostics": (
                        None
                        if session.auto_alignment_diagnostics is None
                        else {
                            key: value
                            for key, value in session.auto_alignment_diagnostics.items()
                            if key != "alignment_batch_inputs"
                        }
                    ),
                    "deepslice_raw_ensemble_ouv_quicknii_ml_ap_dv": session.deepslice_raw_ensemble_ouv,
                    "deepslice_shared_angle_ouv_quicknii_ml_ap_dv": None,
                    "deepslice_version": session.deepslice_version,
                    "deepslice_model_sha256": session.deepslice_model_hashes,
                    "deepslice_ensemble_disagreement": session.deepslice_ensemble_disagreement,
                    "probe_atlas_points": output_trace(session).atlas_points,
                    "probe_slice_points": self._slice_raw_to_display_points(
                        session,
                        output_trace(session).slice_points,
                    ),
                    "probe_slice_points_raw": output_trace(session).slice_points,
                    "probe_volume_points": output_trace(session).volume_points,
                    "probe_signal_values": output_trace(session).signal_values,
                }
                for session in self.sessions
            ],
        }
        manifest_path = anatomy_dir / f"proprietary_trajectory_manifest_{probe_name}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._alignment_cancel_event is not None:
            self._alignment_cancel_event.set()
        self._finish_auto_alignment_ui()
        self.alignment_executor.shutdown(wait=False, cancel_futures=True)
        for cache in self._session_cache_dirs:
            cache.cleanup()
        self._session_cache_dirs.clear()
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication([])
    app.setStyleSheet(
        """
        QWidget { background:#161b22; color:#d7e7f5; font-size:10pt; }
        QGroupBox { border:1px solid #2d3a4c; border-radius:7px; margin-top:8px; padding:8px; }
        QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget {
            background:#0f131a; border:1px solid #2d3a4c; border-radius:5px; padding:4px;
        }
        QHeaderView::section { background:#1b2634; color:#d7e7f5; padding:5px; border:1px solid #2d3a4c; }
        QTabWidget::pane { background:#161b22; border:1px solid #41627f; }
        QTabBar::tab {
            background:#1b2634; color:#d7e7f5; border:1px solid #41627f;
            border-bottom:none; padding:7px 16px; margin-right:2px;
        }
        QTabBar::tab:selected { background:#2b6f95; color:#ffffff; }
        QTabBar::tab:hover:!selected { background:#24415a; color:#ffffff; }
        QPushButton { background:#1b2634; border:1px solid #33475b; border-radius:6px; padding:7px 12px; }
        QPushButton:hover { background:#25384a; }
        QPushButton[role="primary"] { background:#245f82; border-color:#4387ad; }
        QPushButton[role="primary"]:hover { background:#2b7199; }
        QPushButton:focus { border:2px solid #80d4ff; }
        QPushButton:disabled { background:#1b232d; border-color:#293544; color:#667789; }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTableWidget:focus {
            border:1px solid #80d4ff;
        }
        QSplitter::handle { background:#2d3a4c; }
        QSplitter::handle:hover { background:#49b9ff; }
        """
    )
    window = TrajectoryTrackerWindow(
        default_atlas_folder=os.environ.get("TRAJECTORY_ATLAS_FOLDER", str(DEFAULT_ATLAS_FOLDER)),
        default_slices_folder=os.environ.get("TRAJECTORY_SLICES_FOLDER", ""),
        default_run_folder=os.environ.get("TRAJECTORY_RUN_FOLDER", ""),
    )
    window.showMaximized()
    app.exec()


if __name__ == "__main__":
    main()
