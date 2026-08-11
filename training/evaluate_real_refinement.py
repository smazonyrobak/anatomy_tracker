from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import nrrd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
import proprietary_trajectory_tool as tracker


BENCHMARK_ROOT = Path("G:/DeepSlice_GroundTruth_Evaluation")
MODEL = ROOT / "models/AtlasPose/atlas_pose.onnx"
MODEL_DIGEST = hashlib.sha256(MODEL.read_bytes()).hexdigest()
MODEL_METADATA = json.loads(MODEL.with_suffix(".json").read_text(encoding="utf-8"))
PREDICTIONS = BENCHMARK_ROOT / "results" / MODEL_DIGEST[:12] / "predictions.csv"
OUTPUT = PREDICTIONS.parent / "production_refinement_subset"
ATLAS = ROOT / "data/Allen Brain Atlas 25um/average_template_25.nrrd"
ANNOTATION = ROOT / "data/Allen Brain Atlas 25um/annotation_25.nrrd"
AXES = ("ap_um", "lr_deg", "dv_deg")
TOLERANCE = np.asarray((250.0, 2.0, 2.0))
QUANTILES = (0.1, 0.5, 0.9)


def representative_subset(table: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for _, rows in table[table["in_training_ap_domain"]].groupby("dataset", sort=True):
        available = rows.copy()
        for quantile in QUANTILES:
            target = rows["gt_ap_um"].quantile(quantile)
            index = (available["gt_ap_um"] - target).abs().idxmin()
            selected.append(index)
            available = available.drop(index)
    return table.loc[selected].sort_values(["dataset", "gt_ap_um"]).reset_index(drop=True)


def prepare_input(path: str) -> tuple[np.ndarray, np.ndarray]:
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    display, _ = tracker.downsample_for_display(tracker.as_gray(raw))
    image = tracker.normalize_u8(display)
    return image, tracker.automatic_brain_mask(image)


def summarize(table: pd.DataFrame, prefix: str) -> dict:
    target = table[[f"gt_{axis}" for axis in AXES]].to_numpy()
    prediction = table[[f"{prefix}_{axis}" for axis in AXES]].to_numpy()
    error = prediction - target
    absolute = np.abs(error)
    return {
        axis: {
            "bias": float(error[:, column].mean()),
            "mae": float(absolute[:, column].mean()),
            "median_absolute_error": float(np.median(absolute[:, column])),
            "p95_absolute_error": float(np.percentile(absolute[:, column], 95.0)),
        }
        for column, axis in enumerate(AXES)
    }


def main() -> None:
    table = representative_subset(pd.read_csv(PREDICTIONS))
    atlas = nrrd.read(str(ATLAS))[0]
    annotation = nrrd.read(str(ANNOTATION))[0]
    converted = {}
    records = {}
    prepared = {}
    disagreement = {}
    for index, row in table.iterrows():
        filename = f"benchmark_{index:03d}.png"
        image, brain_mask = prepare_input(row["image_path"])
        converted[index] = (
            216.0 - float(row["prediction_ap_um"]) / tracker.VOXEL_UM,
            float(row["prediction_lr_deg"]),
            float(row["prediction_dv_deg"]),
            None,
        )
        records[index] = {
            "Filenames": filename,
            "model_uncertainty": MODEL_METADATA["real_histology_benchmark"]["raw_in_domain"],
        }
        prepared[filename] = {"image": image, "brain_mask": brain_mask}
        disagreement[filename] = {}

    started = time.perf_counter()
    refined, diagnostics, _ = tracker.refine_pose_search(
        converted,
        records,
        atlas,
        annotation,
        prepared,
        disagreement,
        None,
        [],
        None,
        threading.Event(),
        global_alignment=False,
    )
    elapsed = time.perf_counter() - started
    for index, (ap_index, tilt_lr, tilt_dv) in refined.items():
        table.loc[index, [f"refined_{axis}" for axis in AXES]] = (
            (216.0 - ap_index) * tracker.VOXEL_UM,
            tilt_lr,
            tilt_dv,
        )
        for key, value in diagnostics[index].items():
            table.loc[index, f"diagnostic_{key}"] = value

    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "predictions.csv", index=False)
    report = {
        "source_predictions": str(PREDICTIONS),
        "selection": {
            "count": len(table),
            "method": "Per-dataset in-domain AP 10th, 50th, and 90th percentile representatives",
            "datasets": table["dataset"].value_counts().sort_index().to_dict(),
            "ap_range_um": [float(table["gt_ap_um"].min()), float(table["gt_ap_um"].max())],
        },
        "refinement": {
            "function": "source.proprietary_trajectory_tool.refine_pose_search",
            "mode": "single-slice, no explicit AP bounds, no slice-order constraint",
            "brain_mask": "production automatic_brain_mask on the production display-normalized image",
            "elapsed_seconds": elapsed,
            "seconds_per_slice": elapsed / len(table),
            "boundary_fraction": float(table["diagnostic_pose_search_boundary"].mean()),
            "flat_score_fraction": float(table["diagnostic_pose_search_flat"].mean()),
        },
        "raw": summarize(table, "prediction"),
        "refined": summarize(table, "refined"),
        "comparison": {
            "axis_improved_count": {
                axis: int(
                    (
                        (table[f"refined_{axis}"] - table[f"gt_{axis}"]).abs()
                        < (table[f"prediction_{axis}"] - table[f"gt_{axis}"]).abs()
                    ).sum()
                )
                for axis in AXES
            },
            "overall_normalized_error_improved_count": int(
                (
                    np.mean(
                        np.abs(
                            table[[f"refined_{axis}" for axis in AXES]].to_numpy()
                            - table[[f"gt_{axis}" for axis in AXES]].to_numpy()
                        )
                        / TOLERANCE,
                        axis=1,
                    )
                    < np.mean(
                        np.abs(
                            table[[f"prediction_{axis}" for axis in AXES]].to_numpy()
                            - table[[f"gt_{axis}" for axis in AXES]].to_numpy()
                        )
                        / TOLERANCE,
                        axis=1,
                    )
                ).sum()
            ),
        },
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
