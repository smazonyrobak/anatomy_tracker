"""Landmark QA for dense registration on the frozen real session 722 archive.

This evaluates the nonlinear slice-to-atlas coordinate transform conditional on
the AP position and tilts saved in the session.  It does not evaluate AP/tilt
detection and does not establish performance on independent animals.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import nrrd
import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from atlas_pose_runtime import brain_mask_affine
from dense_registration_runtime import run_dense_registration
from nonlinear_registration import SliceAtlasTransform2D
from proprietary_trajectory_tool import (
    VOXEL_UM,
    as_gray,
    coronal_oblique_slice,
    dense_registration_canvases,
    downsample_for_display,
    fit_surface_scale_translation,
    normalize_u8,
    point_to_volume,
    registration_brain_mask,
    transform_points,
    transform_slice_image,
)


ARCHIVE_SHA256 = "c40f98e113e7d216f95b7152ccf55132cdaafa5f7401aebe6dee7062f1d0a64b"
SESSION_LANDMARK_COUNTS = {0: 32, 1: 35, 4: 25, 5: 25, 6: 29, 7: 34, 8: 25, 9: 25}
EXPECTED_LANDMARK_COUNT = 230
REPORT_SCHEMA = "real-session-dense-registration-landmark-qa-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_state(archive: zipfile.ZipFile) -> dict:
    state = json.loads(archive.read("session.json"))
    if state.get("format") != "Proprietary Anatomy Tracker session" or state.get("version") != 1:
        raise ValueError("Unsupported Anatomy Tracker session archive")
    return state


def load_public_archive_manifest(path: str | Path, expected_sha256: str = ARCHIVE_SHA256) -> dict:
    path = Path(path)
    archive_sha256 = sha256_file(path)
    if archive_sha256 != expected_sha256:
        raise ValueError(
            f"Session archive SHA-256 is {archive_sha256}, expected frozen 722 archive {expected_sha256}"
        )
    with zipfile.ZipFile(path) as archive:
        state = _archive_state(archive)
    sessions = []
    for record in state["sessions"]:
        fields = {
            name: value
            for name, value in record["fields"].items()
            if name not in {"atlas_landmarks", "slice_landmarks"}
        }
        sessions.append(
            {
                "name": record["name"],
                "image_member": record["image_member"],
                "image_sha256": record["image_sha256"],
                "brush_mask_member": record.get("brush_mask_member"),
                "fields": fields,
            }
        )
    return {
        "archive_path": str(path.resolve()),
        "archive_sha256": archive_sha256,
        "saved_utc": state.get("saved_utc"),
        "atlas_folder": state["atlas_folder"],
        "atlas_file_hashes": state["atlas_file_hashes"],
        "sessions": sessions,
    }


def reveal_landmarks(path: str | Path, session_index: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read human references only after model inference for the slice has finished."""
    with zipfile.ZipFile(path) as archive:
        fields = _archive_state(archive)["sessions"][session_index]["fields"]
    atlas = np.asarray(fields["atlas_landmarks"], dtype=np.float64)
    moving = np.asarray(fields["slice_landmarks"], dtype=np.float64)
    score_count = SESSION_LANDMARK_COUNTS[session_index]
    if len(atlas) < score_count or len(moving) < score_count:
        raise ValueError(f"Session {session_index} has fewer than {score_count} frozen landmark pairs")
    return atlas[:score_count], moving[:score_count], {
        "atlas_orphan_count": int(len(atlas) - score_count),
        "slice_orphan_count": int(len(moving) - score_count),
    }


def _decode_image(member: str, payload: bytes) -> np.ndarray:
    if Path(member).suffix.lower() in {".tif", ".tiff"}:
        return tifffile.imread(io.BytesIO(payload))
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode embedded slice {member}")
    return image


def load_atlas(manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    folder = Path(manifest["atlas_folder"])
    template_path = folder / "average_template_25.nrrd"
    annotation_path = folder / "annotation_25.nrrd"
    for path in (template_path, annotation_path):
        expected = manifest["atlas_file_hashes"][path.name]
        if sha256_file(path) != expected:
            raise ValueError(f"Atlas file differs from the session-pinned copy: {path}")
    atlas = nrrd.read(str(template_path))[0]
    annotation = nrrd.read(str(annotation_path))[0]
    if atlas.shape != annotation.shape:
        raise ValueError("Atlas template and annotation volumes have different shapes")
    return atlas, annotation


def reconstruct_case(
    archive_path: str | Path,
    public_record: dict,
    atlas_volume: np.ndarray,
    annotation_volume: np.ndarray,
) -> dict:
    fields = public_record["fields"]
    if fields["atlas_plane"] != "coronal":
        raise ValueError("Real-session dense-registration QA supports coronal sections only")
    with zipfile.ZipFile(archive_path) as archive:
        image_bytes = archive.read(public_record["image_member"])
        if hashlib.sha256(image_bytes).hexdigest() != public_record["image_sha256"]:
            raise ValueError(f"Embedded image failed its checksum: {public_record['name']}")
        selection_mask = None
        if public_record["brush_mask_member"]:
            with np.load(io.BytesIO(archive.read(public_record["brush_mask_member"])), allow_pickle=False) as values:
                selection_mask = values["mask"].astype(bool)

    display_raw, display_scale = downsample_for_display(as_gray(_decode_image(public_record["image_member"], image_bytes)))
    raw_display = normalize_u8(display_raw)
    moving_image, slice_transform = transform_slice_image(
        raw_display,
        fields["rotation_deg"],
        fields["flip_horizontal"],
        fields["flip_vertical"],
    )
    surface = transform_points(fields["brain_outline_points"], slice_transform)
    moving_mask = registration_brain_mask(
        moving_image,
        surface,
        fields["brain_outline_closed"],
        selection_mask,
    )
    atlas_index = int(fields["atlas_index"])
    tilt_ml = float(fields["atlas_tilt_ml_deg"])
    tilt_dv = float(fields["atlas_tilt_dv_deg"])
    atlas_image = coronal_oblique_slice(atlas_volume, atlas_index, tilt_ml, tilt_dv, order=1)
    atlas_labels = coronal_oblique_slice(annotation_volume, atlas_index, tilt_ml, tilt_dv, order=0)
    atlas_mask = atlas_labels > 0
    base_h = brain_mask_affine(moving_mask, atlas_mask)
    if len(surface) >= 8:
        base_h, surface_fit = fit_surface_scale_translation(base_h, surface, atlas_mask)
    else:
        surface_fit = None
    canvases = dense_registration_canvases(
        moving_image,
        moving_mask,
        atlas_image,
        atlas_mask,
        base_h,
    )
    return {
        "name": public_record["name"],
        "display_scale": display_scale,
        "moving_image": moving_image,
        "moving_mask": moving_mask,
        "slice_transform": slice_transform,
        "atlas_index": atlas_index,
        "tilt_ml_deg": tilt_ml,
        "tilt_dv_deg": tilt_dv,
        "atlas_image": atlas_image,
        "atlas_labels": atlas_labels,
        "atlas_mask": atlas_mask,
        "base_h": base_h,
        "surface_fit": surface_fit,
        "canvases": canvases,
    }


def _homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    mapped = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def _point_mask_values(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    finite = np.isfinite(points).all(axis=1)
    safe = np.where(finite[:, None], points, 0.0)
    inside = (
        finite
        & (safe[:, 0] >= 0.0)
        & (safe[:, 0] <= mask.shape[1] - 1.0)
        & (safe[:, 1] >= 0.0)
        & (safe[:, 1] <= mask.shape[0] - 1.0)
    )
    values = cv2.remap(
        mask.astype(np.uint8),
        safe[:, 0].astype(np.float32).reshape(-1, 1),
        safe[:, 1].astype(np.float32).reshape(-1, 1),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).ravel().astype(bool)
    return inside & values


def _error_summary(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "count": int(len(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.quantile(errors, 0.95)),
        "maximum": float(errors.max()),
        "mean": float(errors.mean()),
    }


def score_landmarks(
    case: dict,
    transform: SliceAtlasTransform2D,
    atlas_landmarks: np.ndarray,
    raw_slice_landmarks: np.ndarray,
    atlas_volume_shape: tuple[int, int, int],
) -> dict:
    display_landmarks = np.asarray(transform_points(raw_slice_landmarks, case["slice_transform"]))
    affine_prediction = _homography(display_landmarks, case["base_h"])
    nonlinear_prediction = transform.map_display_to_atlas(display_landmarks)
    target_finite = np.isfinite(atlas_landmarks).all(axis=1)
    target_labeled = _point_mask_values(case["atlas_mask"], atlas_landmarks)
    prediction_finite = np.isfinite(nonlinear_prediction).all(axis=1)
    valid = target_finite & target_labeled & prediction_finite

    nonlinear_delta = nonlinear_prediction - atlas_landmarks
    affine_delta = affine_prediction - atlas_landmarks
    nonlinear_px = np.linalg.norm(nonlinear_delta[valid], axis=1)
    affine_px = np.linalg.norm(affine_delta[valid], axis=1)
    nonlinear_mm = []
    affine_mm = []
    for target, nonlinear, affine in zip(
        atlas_landmarks[valid], nonlinear_prediction[valid], affine_prediction[valid]
    ):
        target_volume = point_to_volume(
            tuple(target), "coronal", case["atlas_index"], atlas_volume_shape,
            case["tilt_ml_deg"], case["tilt_dv_deg"],
        )
        nonlinear_volume = point_to_volume(
            tuple(nonlinear), "coronal", case["atlas_index"], atlas_volume_shape,
            case["tilt_ml_deg"], case["tilt_dv_deg"],
        )
        affine_volume = point_to_volume(
            tuple(affine), "coronal", case["atlas_index"], atlas_volume_shape,
            case["tilt_ml_deg"], case["tilt_dv_deg"],
        )
        nonlinear_mm.append(np.linalg.norm(nonlinear_volume - target_volume) * VOXEL_UM / 1000.0)
        affine_mm.append(np.linalg.norm(affine_volume - target_volume) * VOXEL_UM / 1000.0)
    return {
        "landmark_count": int(len(atlas_landmarks)),
        "invalid_target_count": int(np.count_nonzero(~(target_finite & target_labeled))),
        "nonfinite_prediction_count": int(np.count_nonzero(~prediction_finite)),
        "valid_count": int(np.count_nonzero(valid)),
        "nonlinear_error_px": np.asarray(nonlinear_px),
        "affine_error_px": np.asarray(affine_px),
        "nonlinear_error_mm": np.asarray(nonlinear_mm),
        "affine_error_mm": np.asarray(affine_mm),
        "signed_error_xy_px": nonlinear_delta[valid],
        "target_xy": atlas_landmarks,
        "affine_prediction_xy": affine_prediction,
        "nonlinear_prediction_xy": nonlinear_prediction,
        "valid": valid,
    }


def _public_slice_report(result: dict) -> dict:
    score = result["score"]
    diagnostics = result["diagnostics"]
    nonlinear_px = _error_summary(score["nonlinear_error_px"])
    affine_px = _error_summary(score["affine_error_px"])
    return {
        "session_index": result["session_index"],
        "name": result["case"]["name"],
        "pose": {
            "atlas_index": result["case"]["atlas_index"],
            "tilt_ml_deg": result["case"]["tilt_ml_deg"],
            "tilt_dv_deg": result["case"]["tilt_dv_deg"],
        },
        "landmark_count": score["landmark_count"],
        "invalid_target_count": score["invalid_target_count"],
        "nonfinite_prediction_count": score["nonfinite_prediction_count"],
        "invalid_or_nonfinite_count": score["landmark_count"] - score["valid_count"],
        "orphan_landmarks": result["orphans"],
        "nonlinear_error_px": nonlinear_px,
        "nonlinear_error_mm": _error_summary(score["nonlinear_error_mm"]),
        "affine_error_px": affine_px,
        "affine_error_mm": _error_summary(score["affine_error_mm"]),
        "p95_improvement_over_affine_fraction": float(
            (affine_px["p95"] - nonlinear_px["p95"]) / max(affine_px["p95"], 1e-12)
        ),
        "signed_bias_px": {
            "ml_x": float(score["signed_error_xy_px"][:, 0].mean()),
            "dv_y": float(score["signed_error_xy_px"][:, 1].mean()),
        },
        "deformation": {
            "fold_count": diagnostics["fold_count"],
            "minimum_jacobian": diagnostics["minimum_jacobian"],
            "inverse_cycle_p95_px": diagnostics["inverse_cycle_p95_px"],
            "inverse_cycle_max_px": diagnostics["inverse_cycle_max_px"],
        },
    }


def summarize_results(results: list[dict], manifest: dict, model_path: Path, metadata_path: Path) -> dict:
    nonlinear_px = np.concatenate([item["score"]["nonlinear_error_px"] for item in results])
    affine_px = np.concatenate([item["score"]["affine_error_px"] for item in results])
    nonlinear_mm = np.concatenate([item["score"]["nonlinear_error_mm"] for item in results])
    affine_mm = np.concatenate([item["score"]["affine_error_mm"] for item in results])
    signed = np.concatenate([item["score"]["signed_error_xy_px"] for item in results])
    per_slice = [_public_slice_report(item) for item in results]
    nonlinear_summary = _error_summary(nonlinear_px)
    affine_summary = _error_summary(affine_px)
    improvement = (affine_summary["p95"] - nonlinear_summary["p95"]) / max(affine_summary["p95"], 1e-12)
    deformation = {
        "fold_count": int(sum(item["diagnostics"]["fold_count"] for item in results)),
        "minimum_jacobian": float(min(item["diagnostics"]["minimum_jacobian"] for item in results)),
        "worst_inverse_cycle_p95_px": float(max(item["diagnostics"]["inverse_cycle_p95_px"] for item in results)),
        "worst_inverse_cycle_max_px": float(max(item["diagnostics"]["inverse_cycle_max_px"] for item in results)),
    }
    invalid = sum(item["score"]["landmark_count"] - item["score"]["valid_count"] for item in results)
    gates = {
        "all_eight_complete_slices_executed": len(results) == len(SESSION_LANDMARK_COUNTS),
        "all_230_landmark_pairs_scored": sum(item["score"]["valid_count"] for item in results) == EXPECTED_LANDMARK_COUNT,
        "all_targets_labeled_and_predictions_finite": invalid == 0,
        "pooled_median_at_most_2px": nonlinear_summary["median"] <= 2.0,
        "pooled_p95_at_most_5px": nonlinear_summary["p95"] <= 5.0,
        "every_slice_median_at_most_3px": all(item["nonlinear_error_px"]["median"] <= 3.0 for item in per_slice),
        "absolute_ml_bias_at_most_1px": abs(float(signed[:, 0].mean())) <= 1.0,
        "absolute_dv_bias_at_most_1px": abs(float(signed[:, 1].mean())) <= 1.0,
        "zero_folds": deformation["fold_count"] == 0,
        "minimum_jacobian_at_least_0_01": deformation["minimum_jacobian"] >= 0.01,
        "cycle_p95_at_most_2px": deformation["worst_inverse_cycle_p95_px"] <= 2.0,
        "cycle_max_at_most_5px": deformation["worst_inverse_cycle_max_px"] <= 5.0,
        "p95_not_worse_than_affine": nonlinear_summary["p95"] <= affine_summary["p95"],
        "p95_improves_at_least_20pct_when_affine_exceeds_5px": affine_summary["p95"] <= 5.0 or improvement >= 0.20,
    }
    return {
        "schema": REPORT_SCHEMA,
        "scope": (
            "Conditional nonlinear 2D warp QA at the session-saved AP position and tilts. "
            "Landmarks were hidden until after inference. This is not AP/tilt detection QA and "
            "not an independent-animal real-histology accuracy claim."
        ),
        "archive": {
            "path": manifest["archive_path"],
            "sha256": manifest["archive_sha256"],
            "saved_utc": manifest["saved_utc"],
        },
        "model": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
        },
        "runtime": {
            name: results[0]["runtime_metadata"].get(name)
            for name in (
                "method",
                "provider",
                "candidate_checkpoint_file_sha256",
                "preprocessing_contract",
                "map_contract",
            )
        },
        "summary": {
            "slice_count": len(results),
            "landmark_count": int(len(nonlinear_px)),
            "invalid_or_nonfinite_count": int(invalid),
            "nonlinear_error_px": nonlinear_summary,
            "nonlinear_error_mm": _error_summary(nonlinear_mm),
            "affine_error_px": affine_summary,
            "affine_error_mm": _error_summary(affine_mm),
            "p95_improvement_over_affine_fraction": float(improvement),
            "signed_bias_px": {"ml_x": float(signed[:, 0].mean()), "dv_y": float(signed[:, 1].mean())},
            "deformation": deformation,
        },
        "hard_gates": {"passed": all(gates.values()), "checks": gates},
        "strong_target": {
            "passed": nonlinear_summary["median"] <= 1.0 and nonlinear_summary["p95"] <= 3.0,
            "checks": {"pooled_median_at_most_1px": nonlinear_summary["median"] <= 1.0, "pooled_p95_at_most_3px": nonlinear_summary["p95"] <= 3.0},
        },
        "per_slice": per_slice,
    }


def _boundary(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    return ((values != np.roll(values, 1, 0)) | (values != np.roll(values, 1, 1))) & (values > 0)


def _panel(image: np.ndarray, title: str) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 2:
        values = cv2.cvtColor(values, cv2.COLOR_GRAY2BGR)
    scale = min(456.0 / values.shape[1], 320.0 / values.shape[0])
    resized = cv2.resize(
        values,
        (max(1, round(values.shape[1] * scale)), max(1, round(values.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    output = np.zeros((350, 456, 3), np.uint8)
    y0 = 30 + (320 - resized.shape[0]) // 2
    x0 = (456 - resized.shape[1]) // 2
    output[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    cv2.putText(output, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return output


def _overlay(atlas: np.ndarray, moving: np.ndarray, labels: np.ndarray) -> np.ndarray:
    fixed = normalize_u8(atlas)
    moved = normalize_u8(moving)
    image = np.stack((fixed, ((fixed.astype(np.uint16) + moved) // 2).astype(np.uint8), moved), axis=-1)
    image[_boundary(labels)] = (255, 190, 20)
    return image


def write_montage(results: list[dict], destination: str | Path) -> None:
    rows = []
    for result in results:
        case, score, transform = result["case"], result["score"], result["transform"]
        atlas_canvas, _, affine_canvas, _ = case["canvases"]
        nonlinear_canvas = transform.render_display_image_in_atlas(case["moving_image"])
        raw = normalize_u8(case["moving_image"])
        raw_mask = case["moving_mask"].astype(np.uint8) * 255
        raw_panel = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(raw_panel, contours, -1, (255, 170, 0), 2)
        atlas_panel = cv2.cvtColor(normalize_u8(atlas_canvas), cv2.COLOR_GRAY2BGR)
        atlas_panel[_boundary(case["atlas_labels"])] = (255, 190, 20)
        affine_overlay = _overlay(atlas_canvas, affine_canvas, case["atlas_labels"])
        nonlinear_overlay = _overlay(atlas_canvas, nonlinear_canvas, case["atlas_labels"])
        for target, predicted, valid in zip(score["target_xy"], score["nonlinear_prediction_xy"], score["valid"]):
            if valid:
                target_xy = tuple(np.rint(target).astype(int))
                predicted_xy = tuple(np.rint(predicted).astype(int))
                cv2.arrowedLine(nonlinear_overlay, target_xy, predicted_xy, (0, 220, 255), 1, tipLength=0.25)
                cv2.circle(nonlinear_overlay, target_xy, 2, (50, 255, 50), -1)
        rows.append(
            np.hstack(
                (
                    _panel(raw_panel, f"{case['name']} oriented slice + mask"),
                    _panel(atlas_panel, "saved-pose atlas + region boundaries"),
                    _panel(affine_overlay, "affine baseline"),
                    _panel(nonlinear_overlay, "learned warp; green target / yellow residual"),
                )
            )
        )
    if not cv2.imwrite(str(destination), np.vstack(rows)):
        raise OSError(f"Could not write montage to {destination}")


def write_markdown(report: dict, destination: str | Path, montage_name: str) -> None:
    summary = report["summary"]
    lines = [
        "# Real-session 722 dense-registration QA",
        "",
        report["scope"],
        "",
        f"Hard gates: **{'PASS' if report['hard_gates']['passed'] else 'FAIL'}**  ",
        f"Strong target: **{'PASS' if report['strong_target']['passed'] else 'FAIL'}**  ",
        f"Landmarks: {summary['landmark_count']} across {summary['slice_count']} slices  ",
        f"Learned warp: median {summary['nonlinear_error_px']['median']:.3f} px, p95 {summary['nonlinear_error_px']['p95']:.3f} px  ",
        f"Affine baseline: median {summary['affine_error_px']['median']:.3f} px, p95 {summary['affine_error_px']['p95']:.3f} px  ",
        f"P95 improvement: {100.0 * summary['p95_improvement_over_affine_fraction']:.1f}%",
        "",
        f"![Visual QA montage]({montage_name})",
        "",
        "| Slice | pairs | median px | p95 px | affine p95 px | invalid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["per_slice"]:
        lines.append(
            f"| {item['name']} | {item['landmark_count']} | "
            f"{item['nonlinear_error_px']['median']:.3f} | {item['nonlinear_error_px']['p95']:.3f} | "
            f"{item['affine_error_px']['p95']:.3f} | "
            f"{item['invalid_or_nonfinite_count']} |"
        )
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_archive(
    archive_path: str | Path,
    model_path: str | Path,
    metadata_path: str | Path,
    expected_model_sha256: str,
    expected_metadata_sha256: str,
    output_folder: str | Path,
    *,
    expected_archive_sha256: str = ARCHIVE_SHA256,
    registration_runner=run_dense_registration,
) -> dict:
    manifest = load_public_archive_manifest(archive_path, expected_archive_sha256)
    atlas_volume, annotation_volume = load_atlas(manifest)
    results = []
    for session_index in SESSION_LANDMARK_COUNTS:
        case = reconstruct_case(
            archive_path,
            manifest["sessions"][session_index],
            atlas_volume,
            annotation_volume,
        )
        atlas_canvas, atlas_mask_canvas, slice_canvas, slice_mask_canvas = case["canvases"]
        inference = registration_runner(
            model_path,
            atlas_canvas,
            atlas_mask_canvas,
            slice_canvas,
            slice_mask_canvas,
            expected_model_sha256=expected_model_sha256,
            expected_metadata_sha256=expected_metadata_sha256,
            metadata_path=metadata_path,
        )
        transform = SliceAtlasTransform2D(
            case["base_h"],
            case["moving_image"].shape,
            atlas_canvas.shape,
            inference["atlas_to_affine_xy"],
            inference["affine_to_atlas_xy"],
            inference["valid_atlas_mask"],
            inference["registration_metadata_json"],
        )
        atlas_landmarks, slice_landmarks, orphans = reveal_landmarks(archive_path, session_index)
        score = score_landmarks(case, transform, atlas_landmarks, slice_landmarks, atlas_volume.shape)
        results.append(
            {
                "session_index": session_index,
                "case": case,
                "transform": transform,
                "score": score,
                "orphans": orphans,
                "diagnostics": inference["metadata"]["output_diagnostics"],
                "runtime_metadata": inference["metadata"],
            }
        )
        print(
            f"{case['name']}: {score['valid_count']} pairs, "
            f"median {_error_summary(score['nonlinear_error_px'])['median']:.3f} px, "
            f"p95 {_error_summary(score['nonlinear_error_px'])['p95']:.3f} px",
            flush=True,
        )

    report = summarize_results(results, manifest, Path(model_path), Path(metadata_path))
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    json_path = output_folder / "real-722-landmark-qa.json"
    montage_path = output_folder / "real-722-landmark-qa.png"
    markdown_path = output_folder / "real-722-landmark-qa.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_montage(results, montage_path)
    write_markdown(report, markdown_path, montage_path.name)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--archive-sha256", default=ARCHIVE_SHA256)
    args = parser.parse_args()
    report = evaluate_archive(
        args.archive,
        args.model,
        args.metadata,
        args.model_sha256,
        args.metadata_sha256,
        args.output_folder,
        expected_archive_sha256=args.archive_sha256,
    )
    summary = report["summary"]["nonlinear_error_px"]
    print(
        f"Hard QA {'PASSED' if report['hard_gates']['passed'] else 'FAILED'}: "
        f"median {summary['median']:.3f} px, p95 {summary['p95']:.3f} px",
        flush=True,
    )
    return 0 if report["hard_gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
