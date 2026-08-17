from __future__ import annotations

import numpy as np
import pandas as pd

from source.atlas_pose_runtime import ATLAS_POSE_RELEASE_GATE_THRESHOLDS


# Final AtlasPose comparisons are paired at section level and resampled at the independent animal level.
POSE_AXES = ("ap_um", "lr_deg", "dv_deg")
RELEASE_GATE_THRESHOLDS = ATLAS_POSE_RELEASE_GATE_THRESHOLDS
RELEASE_REFERENCE = "deepslice_mens_ai_ci"
RELEASE_CONFIDENCE = 0.95
ANIMAL_BOOTSTRAP_ITERATIONS = 10_000
ANIMAL_BOOTSTRAP_SEED = 68431


def release_statistics_equal(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            release_statistics_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            release_statistics_equal(a, b) for a, b in zip(left, right)
        )
    if (
        isinstance(left, (int, float, np.integer, np.floating))
        and not isinstance(left, (bool, np.bool_))
        and isinstance(right, (int, float, np.integer, np.floating))
        and not isinstance(right, (bool, np.bool_))
    ):
        return bool(np.isclose(left, right, rtol=1e-12, atol=1e-9))
    return left == right


def validate_complete_method_cohort(
    table: pd.DataFrame,
    records: list[dict],
    methods: tuple[str, ...],
) -> None:
    expected = {int(record["section_image_id"]) for record in records}
    if table.duplicated(["method", "section_image_id"]).any():
        raise ValueError("Sealed predictions contain duplicate method/section rows")
    if set(table["method"]) != set(methods):
        raise ValueError("Sealed predictions do not contain the exact predictor set")
    finite_columns = [
        f"{prefix}_{axis}"
        for prefix in ("gt", "pred", "error", "absolute_error")
        for axis in POSE_AXES
    ]
    if not np.isfinite(table[finite_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Sealed predictions contain non-finite pose values")
    for method in methods:
        rows = table[table["method"] == method]
        if len(rows) != len(expected) or set(rows["section_image_id"].astype(int)) != expected:
            raise ValueError(f"{method} does not cover the complete sealed cohort")


def evaluation_domains(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    in_domain = table[table["in_training_ap_domain"].astype(bool)]
    if in_domain.empty:
        raise ValueError("The sealed holdout contains no sections inside the trained AP domain")
    return in_domain, table[~table["in_training_ap_domain"].astype(bool)]


# Bootstrap draws animals, never treating correlated slices as independent replicates.
def paired_animal_bootstrap(
    rows: pd.DataFrame,
    candidate: str,
    reference: str,
    metric: str,
    iterations: int = 10_000,
    seed: int = 94731,
) -> dict:
    selected = rows[rows["method"].isin((candidate, reference))]
    if selected.duplicated(["method", "specimen_id", "section_image_id"]).any():
        raise ValueError(f"Duplicate paired {metric} values")
    pivot = selected.pivot(index=["specimen_id", "section_image_id"], columns="method", values=metric)
    if (
        candidate not in pivot
        or reference not in pivot
        or pivot[[candidate, reference]].isna().any().any()
        or not np.isfinite(pivot[[candidate, reference]].to_numpy(dtype=np.float64)).all()
        or len(pivot) * 2 != len(selected)
    ):
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


def paired_animal_joint_superiority(
    rows: pd.DataFrame,
    candidate: str,
    reference: str,
    metrics: tuple[str, ...],
    iterations: int = 10_000,
    seed: int = 94731,
) -> dict:
    selected = rows[rows["method"].isin((candidate, reference))]
    if selected.duplicated(["method", "specimen_id", "section_image_id"]).any():
        raise ValueError("Duplicate rows in paired joint comparison")
    section_index = ["specimen_id", "section_image_id"]
    deltas = []
    expected_index = None
    for metric in metrics:
        pivot = selected.pivot(index=section_index, columns="method", values=metric)
        if (
            candidate not in pivot
            or reference not in pivot
            or pivot[[candidate, reference]].isna().any().any()
            or not np.isfinite(pivot[[candidate, reference]].to_numpy(dtype=np.float64)).all()
            or len(pivot) * 2 != len(selected)
        ):
            raise ValueError(f"Incomplete paired cohort for {metric}")
        if expected_index is not None and not pivot.index.equals(expected_index):
            raise ValueError("Paired metrics describe different sealed cohorts")
        expected_index = pivot.index
        deltas.append((pivot[candidate] - pivot[reference]).groupby(level="specimen_id").mean())
    animal_delta = np.column_stack([delta.to_numpy(dtype=np.float64) for delta in deltas])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(animal_delta), (iterations, len(animal_delta)))
    draws = animal_delta[sampled].mean(axis=1)
    point = animal_delta.mean(axis=0)
    upper95 = np.percentile(draws, 95.0, axis=0)
    probability = float(np.mean(np.all(draws < 0.0, axis=1)))
    return {
        "unit": "specimen_id",
        "animal_count": int(len(animal_delta)),
        "paired_section_count": int(len(expected_index)),
        "metrics": list(metrics),
        "candidate": candidate,
        "reference": reference,
        "delta_candidate_minus_reference": dict(zip(metrics, point.tolist())),
        "one_sided_upper95": dict(zip(metrics, upper95.tolist())),
        "probability_all_components_lower_error": probability,
        "confidence_threshold": RELEASE_CONFIDENCE,
        "simultaneous_superiority_passed": bool(
            np.all(point < 0.0) and probability >= RELEASE_CONFIDENCE
        ),
        "iterations": int(iterations),
        "seed": int(seed),
    }


def _ap_500um_band(ap_um: float) -> str:
    index = 9 if ap_um == 500.0 else int(np.floor((ap_um + 4500.0) / 500.0))
    low = -4500 + 500 * index
    return f"{low}:{low + 500}"


def _lr_bin(value: float) -> str:
    if value < -1.5:
        return "lt_-1.5"
    return "gt_1.5" if value > 1.5 else "-1.5_to_1.5"


def _dv_bin(value: float) -> str:
    if value < -7.0:
        return "lt_-7"
    return "gt_-2" if value > -2.0 else "-7_to_-2"


def _balanced_animal_errors(frame: pd.DataFrame) -> pd.DataFrame:
    values = []
    for specimen_id, specimen in frame.groupby("specimen_id", sort=True):
        bands = [
            np.column_stack(
                [np.abs(band[f"pred_{axis}"] - band[f"gt_{axis}"]) for axis in POSE_AXES]
            ).mean(axis=0)
            for _, band in specimen.groupby("ap_500um_band", sort=True)
        ]
        values.append((int(specimen_id), *np.asarray(bands).mean(axis=0)))
    return pd.DataFrame(values, columns=("specimen_id", *POSE_AXES)).set_index("specimen_id")


def _group_component_p90(frame: pd.DataFrame) -> tuple[dict, np.ndarray]:
    groupers = {
        "product": frame["product"].astype(str),
        "ap_band": frame["ap_500um_band"],
        "lr_bin": frame["gt_lr_deg"].map(_lr_bin),
        "dv_bin": frame["gt_dv_deg"].map(_dv_bin),
    }
    report = {}
    tails = []
    for family, labels in groupers.items():
        report[family] = {}
        for label in sorted(labels.unique()):
            errors = _balanced_animal_errors(frame[labels == label])
            tail = np.percentile(errors.to_numpy(), 90.0, axis=0)
            report[family][str(label)] = {
                "animal_count": int(len(errors)),
                "component_p90": dict(zip(POSE_AXES, tail.tolist())),
            }
            tails.append(tail)
    return report, np.max(np.asarray(tails), axis=0)


# This is a release decision only; training and checkpoint selection must not call it on sealed data.
def release_quality_gate(rows: pd.DataFrame) -> dict:
    frame = rows[rows["in_training_ap_domain"].astype(bool)].copy()
    if frame.empty or set(frame["method"]) != {"atlas_pose"}:
        raise ValueError("Release quality gate requires in-domain AtlasPose rows only")
    frame["ap_500um_band"] = frame["gt_ap_um"].map(_ap_500um_band)
    for axis in POSE_AXES:
        frame[f"absolute_{axis}"] = (frame[f"pred_{axis}"] - frame[f"gt_{axis}"]).abs()
    animal_errors = _balanced_animal_errors(frame)
    component_mae = animal_errors.mean(axis=0)
    per_animal_p90 = np.percentile(animal_errors.to_numpy(), 90.0, axis=0)
    group_component_p90, worst_group_p90 = _group_component_p90(frame)
    rng = np.random.default_rng(ANIMAL_BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(animal_errors), (ANIMAL_BOOTSTRAP_ITERATIONS, len(animal_errors)))
    ap_bootstrap_upper95 = float(
        np.percentile(animal_errors["ap_um"].to_numpy()[sampled].mean(axis=1), 95.0)
    )
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
    worst_band_mae = float(band_mae.max())
    worst_product_mae = float(product_mae.max())
    worst_band = next(
        str(label)
        for label, value in band_mae.items()
        if np.isclose(value, worst_band_mae, rtol=1e-12, atol=1e-9)
    )
    worst_product = next(
        str(label)
        for label, value in product_mae.items()
        if np.isclose(value, worst_product_mae, rtol=1e-12, atol=1e-9)
    )
    values = {
        "mean_ap_um": float(component_mae["ap_um"]),
        "mean_lr_deg": float(component_mae["lr_deg"]),
        "mean_dv_deg": float(component_mae["dv_deg"]),
        "absolute_ap_bias_um": absolute_ap_bias,
        "ap_p95_um": float(np.percentile(frame["absolute_ap_um"], 95.0)),
        "worst_ap_band_mae_um": worst_band_mae,
        "worst_product_mae_um": worst_product_mae,
        "ap_bootstrap_upper95_um": ap_bootstrap_upper95,
        "per_animal_p90_ap_um": float(per_animal_p90[0]),
        "per_animal_p90_lr_deg": float(per_animal_p90[1]),
        "per_animal_p90_dv_deg": float(per_animal_p90[2]),
        "worst_group_p90_ap_um": float(worst_group_p90[0]),
        "worst_group_p90_lr_deg": float(worst_group_p90[1]),
        "worst_group_p90_dv_deg": float(worst_group_p90[2]),
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
        "worst_ap_band": worst_band,
        "worst_product": worst_product,
        "animal_count": int(len(animal_errors)),
        "group_component_p90": group_component_p90,
    }
