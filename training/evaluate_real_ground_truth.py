from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
import atlas_pose_runtime
from atlas_pose_runtime import automatic_brain_mask, fuse_pose_predictions, preprocess_atlas_pose_image


GROUND_TRUTH_ROOT = Path(os.environ.get("DEEPSLICE_GT_ROOT", "G:/DeepSlice_GroundTruth_Evaluation"))
MODEL_PATH = Path(os.environ.get("ATLAS_POSE_MODEL", ROOT / "models/AtlasPose/atlas_pose.onnx"))
BATCH_SIZE = int(os.environ.get("ATLAS_POSE_EVAL_BATCH_SIZE", "32"))
OUV = ("ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz")
OPERATORS = (
    "Expert_1",
    "Expert_2",
    "Intermediate_1",
    "Intermediate_2",
    "Novice_1",
    "Novice_2",
    "Novice_3",
)
DATASETS = {
    "CAMKII": "CamKII",
    "DAB": "bAmyloid",
    "GLTa": "GLT1a",
    "ISH": "Calb1",
    "Myelin": "Myelin",
    "PCP2": "PcP2",
    "PITX3": "Pitx3",
}
BENCHMARK_ROLE = {
    **dict.fromkeys(("CamKII", "GLT1a", "PcP2"), "development"),
    **dict.fromkeys(("Myelin", "Pitx3", "Calb1", "bAmyloid"), "test"),
}
DEVELOPMENT_EXCLUSIONS = {
    ("GLT1a", "641_2002_2568_nm01_s109_10x_a.png"),
    ("GLT1a", "641_2002_2568_nm01_s114_10x_a.png"),
    ("GLT1a", "641_2002_2568_nm01_s119_10x_a.png"),
    ("PcP2", "1261_pcp2_tta_lacz_xgal_nr_s173.png"),
    ("PcP2", "1261_pcp2_tta_lacz_xgal_nr_s176.png"),
}
DEEPSLICE_VARIANTS = {
    "ai": "AI",
    "mens_ai": "MENS-AI",
    "mens_ai_ci": "MENS-AI-CI",
}
AXES = ("ap_um", "lr_deg", "dv_deg")
TOLERANCE = np.asarray((250.0, 2.0, 2.0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filename_key(value: str) -> str:
    return Path(str(value).replace("\\", "/")).name.casefold()


def quicknii_to_tracker_pose(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    origin = values[:, :3]
    normal = np.cross(values[:, 3:6], values[:, 6:9])
    normal[normal[:, 1] < 0.0] *= -1.0
    if np.any(np.abs(normal[:, 1]) < 1e-9):
        raise ValueError("Ground truth contains a non-coronal plane")
    ap_per_ml = normal[:, 0] / normal[:, 1]
    ap_per_dv = -normal[:, 2] / normal[:, 1]
    origin_ml = origin[:, 0]
    origin_ap = 528.0 - origin[:, 1]
    origin_dv = 320.0 - origin[:, 2]
    ap_index = origin_ap + ap_per_ml * (227.5 - origin_ml) + ap_per_dv * (159.5 - origin_dv)
    return np.column_stack(
        (
            -(ap_index - 216.0) * 25.0,
            np.degrees(np.arctan(ap_per_ml)),
            np.degrees(np.arctan(ap_per_dv)),
        )
    )


def load_ground_truth() -> pd.DataFrame:
    human_root = GROUND_TRUTH_ROOT / "extracted/human/Operator_Alignments"
    image_root = GROUND_TRUTH_ROOT / "extracted/datasets"
    deepslice_root = GROUND_TRUTH_ROOT / "extracted/deepslice/DeepSlice_Alignments"
    records = []
    for csv_name, dataset in DATASETS.items():
        operator_tables = []
        names = None
        for operator in OPERATORS:
            table = pd.read_csv(human_root / operator / f"{csv_name}.csv", usecols=["Filenames", *OUV])
            table["key"] = table["Filenames"].map(filename_key)
            if table["key"].duplicated().any():
                raise ValueError(f"Duplicate filename in {operator}/{csv_name}.csv")
            table = table.set_index("key").sort_index()
            names = table.index if names is None else names
            if not table.index.equals(names):
                raise ValueError(f"Operator filename mismatch in {operator}/{csv_name}.csv")
            operator_tables.append(table.loc[names, OUV].to_numpy(np.float64))

        alignments = np.stack(operator_tables)
        consensus_ouv = alignments.mean(axis=0)
        consensus_pose = quicknii_to_tracker_pose(consensus_ouv)
        operator_pose = quicknii_to_tracker_pose(alignments.reshape(-1, 9)).reshape(len(OPERATORS), -1, 3)
        operator_sd = operator_pose.std(axis=0, ddof=1)

        image_paths = {
            path.name.casefold(): path
            for path in (image_root / dataset).iterdir()
            if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        }
        if set(image_paths) != set(names):
            raise ValueError(f"Image/annotation filename mismatch in {dataset}")

        deepslice = {}
        for variant, folder in DEEPSLICE_VARIANTS.items():
            table = pd.read_csv(deepslice_root / folder / f"{dataset}.csv", usecols=["Filenames", *OUV])
            table["key"] = table["Filenames"].map(filename_key)
            table = table.set_index("key")
            if set(table.index) != set(names) or table.index.duplicated().any():
                raise ValueError(f"DeepSlice {folder} filename mismatch in {dataset}")
            ouv = table.loc[names, OUV].to_numpy(np.float64)
            deepslice[variant] = (ouv, quicknii_to_tracker_pose(ouv))

        for index, name in enumerate(names):
            record = {
                "dataset": dataset,
                "filename": image_paths[name].name,
                "image_path": str(image_paths[name]),
                "benchmark_role": BENCHMARK_ROLE[dataset],
                "official_usable": (dataset, name) not in DEVELOPMENT_EXCLUSIONS,
                **{f"consensus_{field}": consensus_ouv[index, column] for column, field in enumerate(OUV)},
            }
            for column, axis in enumerate(AXES):
                record[f"gt_{axis}"] = consensus_pose[index, column]
                record[f"human_operator_sd_{axis}"] = operator_sd[index, column]
                for variant, (_, pose) in deepslice.items():
                    record[f"deepslice_{variant}_{axis}"] = pose[index, column]
            for variant, (ouv, _) in deepslice.items():
                record.update(
                    {f"deepslice_{variant}_{field}": ouv[index, column] for column, field in enumerate(OUV)}
                )
            records.append(record)

    result = pd.DataFrame(records).sort_values(["dataset", "filename"]).reset_index(drop=True)
    if len(result) != 315 or result[[f"gt_{axis}" for axis in AXES]].isna().any().any():
        raise ValueError(f"Expected 315 complete ground-truth rows, found {len(result)}")
    if set(result["benchmark_role"]) != {"development", "test"}:
        raise ValueError("Published DeepSlice brain-level split is incomplete")
    usable_counts = result[result["official_usable"]].groupby("benchmark_role").size().to_dict()
    if usable_counts != {"development": 119, "test": 191}:
        raise ValueError(f"Published DeepSlice usable split must be 119 development / 191 test, found {usable_counts}")
    result["in_training_ap_domain"] = result["gt_ap_um"].between(-4500.0, 500.0)
    return result


def create_session() -> tuple[ort.InferenceSession, str]:
    available = ort.get_available_providers()
    provider = next(
        name
        for name in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
        if name in available
    )
    options = ort.SessionOptions()
    if provider == "DmlExecutionProvider":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]
    session = ort.InferenceSession(str(MODEL_PATH), sess_options=options, providers=providers)
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    if model_input.name != "images" or model_input.shape[1:] != [3, 299, 299]:
        raise ValueError(f"Unexpected model input contract: {model_input.name} {model_input.shape}")
    if model_output.name != "pose_ap_um_lr_deg_dv_deg" or model_output.shape[-1] != 3:
        raise ValueError(f"Unexpected model output contract: {model_output.name} {model_output.shape}")
    return session, provider


def predict(table: pd.DataFrame, session: ort.InferenceSession) -> tuple[np.ndarray, pd.DataFrame]:
    predictions = []
    diagnostics = []
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for start in range(0, len(table), BATCH_SIZE):
        rows = table.iloc[start : start + BATCH_SIZE]
        inputs = []
        for path in rows["image_path"].map(Path):
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Could not read {path}")
            mask = automatic_brain_mask(image)
            network_input = preprocess_atlas_pose_image(image, mask)
            if network_input.shape != (3, 299, 299) or network_input.dtype != np.float32:
                raise ValueError(f"Invalid deployable preprocessing output for {path}")
            y, x = np.nonzero(mask)
            bounding_area = (x.max() - x.min() + 1) * (y.max() - y.min() + 1)
            component_count = cv2.connectedComponents(mask.astype(np.uint8), 8)[0] - 1
            diagnostics.append(
                {
                    "mask_area_fraction": float(mask.mean()),
                    "mask_bbox_fill": float(mask.sum() / bounding_area),
                    "mask_component_count": int(component_count),
                    "mask_touches_border": bool(
                        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
                    ),
                    "network_input_mean": float(network_input.mean()),
                    "network_input_std": float(network_input.std()),
                }
            )
            inputs.append(network_input)
        batch = np.ascontiguousarray(np.stack(inputs))
        predictions.append(session.run([output_name], {input_name: batch})[0])
    prediction = np.concatenate(predictions).astype(np.float64)
    if prediction.shape != (315, 3) or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid ONNX prediction array: {prediction.shape}")
    return prediction, pd.DataFrame(diagnostics)


def summarize(table: pd.DataFrame, prediction_prefix: str) -> dict:
    target = table[[f"gt_{axis}" for axis in AXES]].to_numpy()
    prediction = table[[f"{prediction_prefix}_{axis}" for axis in AXES]].to_numpy()
    error = prediction - target
    absolute = np.abs(error)
    result = {"count": len(table), "within_all_tolerances_fraction": float(np.all(absolute <= TOLERANCE, axis=1).mean())}
    for column, axis in enumerate(AXES):
        result[axis] = {
            "bias": float(error[:, column].mean()),
            "mae": float(absolute[:, column].mean()),
            "median_absolute_error": float(np.median(absolute[:, column])),
            "rmse": float(np.sqrt(np.mean(error[:, column] ** 2))),
            "p95_absolute_error": float(np.percentile(absolute[:, column], 95.0)),
            "within_tolerance_fraction": float(np.mean(absolute[:, column] <= TOLERANCE[column])),
        }
    result["mean_normalized_mae"] = float(np.mean(absolute.mean(axis=0) / TOLERANCE))
    return result


def report_scopes(table: pd.DataFrame, prediction_prefix: str) -> dict:
    in_domain = table[table["in_training_ap_domain"]]
    usable = table[table["official_usable"]]
    development = usable[usable["benchmark_role"] == "development"]
    test = usable[usable["benchmark_role"] == "test"]
    by_dataset = {}
    for dataset, rows in table.groupby("dataset", sort=True):
        published_rows = rows[rows["official_usable"]]
        by_dataset[dataset] = {"all_public": summarize(rows, prediction_prefix)}
        if len(published_rows):
            by_dataset[dataset]["published_usable"] = summarize(published_rows, prediction_prefix)
        if published_rows["in_training_ap_domain"].any():
            by_dataset[dataset]["published_usable_in_training_ap_domain"] = summarize(
                published_rows[published_rows["in_training_ap_domain"]], prediction_prefix
            )
    return {
        "all_ood_inclusive": summarize(table, prediction_prefix),
        "in_training_ap_domain": summarize(in_domain, prediction_prefix),
        "published_usable": summarize(usable, prediction_prefix),
        "development": summarize(development, prediction_prefix),
        "development_in_training_ap_domain": summarize(
            development[development["in_training_ap_domain"]], prediction_prefix
        ),
        "test": summarize(test, prediction_prefix),
        "test_in_training_ap_domain": summarize(test[test["in_training_ap_domain"]], prediction_prefix),
        "by_dataset": by_dataset,
    }


def paired_comparison(table: pd.DataFrame, deepslice_prefix: str) -> dict:
    target = table[[f"gt_{axis}" for axis in AXES]].to_numpy()
    own = np.abs(table[[f"prediction_{axis}" for axis in AXES]].to_numpy() - target)
    deepslice = np.abs(table[[f"{deepslice_prefix}_{axis}" for axis in AXES]].to_numpy() - target)
    return {
        "count": len(table),
        "mae_delta_own_minus_deepslice": {
            axis: float((own[:, column] - deepslice[:, column]).mean()) for column, axis in enumerate(AXES)
        },
        "own_lower_absolute_error_fraction": {
            axis: float(np.mean(own[:, column] < deepslice[:, column])) for column, axis in enumerate(AXES)
        },
    }


def compact_pose_metrics(summary: dict) -> dict:
    return {
        "count": summary["count"],
        "ap_mae_um": summary["ap_um"]["mae"],
        "ap_p95_um": summary["ap_um"]["p95_absolute_error"],
        "lr_mae_deg": summary["lr_deg"]["mae"],
        "lr_p95_deg": summary["lr_deg"]["p95_absolute_error"],
        "dv_mae_deg": summary["dv_deg"]["mae"],
        "dv_p95_deg": summary["dv_deg"]["p95_absolute_error"],
        "mean_normalized_mae": summary["mean_normalized_mae"],
    }


def main() -> None:
    model_metadata = json.loads(MODEL_PATH.with_suffix(".json").read_text(encoding="utf-8"))
    table = load_ground_truth()
    session, provider = create_session()
    prediction, mask_diagnostics = predict(table, session)
    for column, axis in enumerate(AXES):
        table[f"prediction_{axis}"] = prediction[:, column]
        table[f"error_{axis}"] = prediction[:, column] - table[f"gt_{axis}"]
        table[f"absolute_error_{axis}"] = table[f"error_{axis}"].abs()
    own_pose = table[[f"prediction_{axis}" for axis in AXES]].to_numpy()
    deepslice_pose = table[[f"deepslice_mens_ai_{axis}" for axis in AXES]].to_numpy()
    weighted_prefixes = {}
    for own_weight_percent in range(10, 100, 10):
        own_weight = own_weight_percent / 100.0
        fused = np.stack(
            [
                fuse_pose_predictions(
                    np.stack((deepslice, own)),
                    np.asarray((1.0 - own_weight, own_weight)),
                )
                for deepslice, own in zip(deepslice_pose, own_pose)
            ]
        )
        prefix = f"weighted_{own_weight_percent:02d}"
        weighted_prefixes[own_weight_percent] = prefix
        for column, axis in enumerate(AXES):
            table[f"{prefix}_{axis}"] = fused[:, column]
    table = pd.concat((table, mask_diagnostics), axis=1)

    model_digest = sha256(MODEL_PATH)
    output = GROUND_TRUTH_ROOT / "results" / model_digest[:12]
    output.mkdir(parents=True, exist_ok=True)
    ground_truth_columns = [
        "dataset",
        "filename",
        "image_path",
        "benchmark_role",
        "official_usable",
        "in_training_ap_domain",
        *[f"consensus_{field}" for field in OUV],
        *[f"gt_{axis}" for axis in AXES],
        *[f"human_operator_sd_{axis}" for axis in AXES],
    ]
    table[ground_truth_columns].to_csv(output / "consensus_ground_truth.csv", index=False)
    table.to_csv(output / "predictions.csv", index=False)
    table[["dataset", "filename", *mask_diagnostics.columns]].to_csv(output / "mask_diagnostics.csv", index=False)

    input_metadata = session.get_inputs()[0]
    output_metadata = session.get_outputs()[0]
    development = table[
        table["official_usable"] & (table["benchmark_role"] == "development")
    ]
    test = table[table["official_usable"] & (table["benchmark_role"] == "test")]
    weighted_vote_sweep = {
        str(own_weight_percent): report_scopes(table, prefix)
        for own_weight_percent, prefix in weighted_prefixes.items()
    }
    best_own_weight_percent = min(
        weighted_prefixes,
        key=lambda value: weighted_vote_sweep[str(value)]["development_in_training_ap_domain"][
            "mean_normalized_mae"
        ],
    )
    metrics = {
        "source": {
            "figshare_article": 22802411,
            "doi": "10.25949/22802411.v1",
            "paper": "https://www.nature.com/articles/s41467-023-41645-4",
            "dataset": "https://figshare.com/articles/dataset/22802411",
            "consensus_method": "Arithmetic mean of seven operators' nine O/U/V values, then pose conversion",
            "archive_sha256": {
                path.name: sha256(path) for path in sorted((GROUND_TRUTH_ROOT / "archives").glob("*.zip"))
            },
        },
        "coordinate_convention": {
            "output": ["AP from bregma (um; anterior positive)", "L-R tilt (deg)", "D-V tilt (deg)"],
            "note": "Tracker D-V sign is opposite upstream DeepSlice calculate_angles(); values here use tracker convention.",
            "training_ap_domain_um": [-4500.0, 500.0],
            "all_image_ap_range_um": [float(table["gt_ap_um"].min()), float(table["gt_ap_um"].max())],
        },
        "model": {
            "path": str(MODEL_PATH),
            "sha256": model_digest,
            "input": {"name": input_metadata.name, "shape": input_metadata.shape},
            "output": {"name": output_metadata.name, "shape": output_metadata.shape},
            "onnxruntime_version": ort.__version__,
            "provider": provider,
        },
        "reproducibility": {
            "preprocessor": str(Path(atlas_pose_runtime.__file__).resolve()),
            "preprocessor_sha256": sha256(Path(atlas_pose_runtime.__file__).resolve()),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "real_benchmark_used_for_training_or_model_selection": bool(
                model_metadata.get("real_benchmark_informed_final_iteration", False)
            ),
        },
        "human_operator_pose_sd": {
            axis: {
                "mean": float(table[f"human_operator_sd_{axis}"].mean()),
                "p95": float(table[f"human_operator_sd_{axis}"].quantile(0.95)),
            }
            for axis in AXES
        },
        "own_model": report_scopes(table, "prediction"),
        "published_deepslice": {
            variant: report_scopes(table, f"deepslice_{variant}") for variant in DEEPSLICE_VARIANTS
        },
        "weighted_vote_sweep": weighted_vote_sweep,
        "weighted_vote_selection": {
            "deepslice_component": "MENS-AI (ensemble with shared tilt; no cutting-index order)",
            "criterion": "lowest mean normalized MAE on official development brains within AP +500 to -4500 um",
            "selected_own_weight_percent": best_own_weight_percent,
        },
        "paired_own_vs_deepslice": {
            variant: {
                "development": paired_comparison(development, f"deepslice_{variant}"),
                "test": paired_comparison(test, f"deepslice_{variant}"),
            }
            for variant in DEEPSLICE_VARIANTS
        },
        "mask_diagnostics": {
            dataset: {
                "count": len(rows),
                "area_fraction_median": float(rows["mask_area_fraction"].median()),
                "area_fraction_range": [float(rows["mask_area_fraction"].min()), float(rows["mask_area_fraction"].max())],
                "bbox_fill_median": float(rows["mask_bbox_fill"].median()),
                "touches_border_count": int(rows["mask_touches_border"].sum()),
            }
            for dataset, rows in table.groupby("dataset", sort=True)
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    own_development = metrics["own_model"]["development_in_training_ap_domain"]
    own_test = metrics["own_model"]["test_in_training_ap_domain"]
    deepslice_development = metrics["published_deepslice"]["mens_ai"][
        "development_in_training_ap_domain"
    ]
    deepslice_test = metrics["published_deepslice"]["mens_ai_ci"]["test_in_training_ap_domain"]
    weighted_development = metrics["weighted_vote_sweep"][str(best_own_weight_percent)][
        "development_in_training_ap_domain"
    ]
    weighted_test = metrics["weighted_vote_sweep"][str(best_own_weight_percent)][
        "test_in_training_ap_domain"
    ]
    model_metadata["real_histology_benchmark"] = {
        "source": "DeepSlice published seven-operator consensus ground truth",
        "model_sha256": model_digest,
        "scope": "AP +500 to -4500 um",
        "raw_in_domain": compact_pose_metrics(own_test),
        "published_deepslice_raw_in_domain": compact_pose_metrics(
            metrics["published_deepslice"]["mens_ai"]["test_in_training_ap_domain"]
        ),
        "selected_weighted_vote": {
            "own_weight_percent": best_own_weight_percent,
            **compact_pose_metrics(weighted_test),
        },
        "development": {
            "own_model": compact_pose_metrics(own_development),
            "published_deepslice_mens_ai": compact_pose_metrics(deepslice_development),
        },
        "retrospective_published_test": {
            "note": "Previously inspected by this project; not an untouched test set.",
            "own_model": compact_pose_metrics(own_test),
            "published_deepslice_mens_ai_ci": compact_pose_metrics(deepslice_test),
        },
        "development_selected_weighted_vote": {
            "own_weight_percent": best_own_weight_percent,
            **compact_pose_metrics(weighted_development),
        },
    }
    MODEL_PATH.with_suffix(".json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "provider": provider,
                "development_images": len(development),
                "published_test_images": len(test),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
