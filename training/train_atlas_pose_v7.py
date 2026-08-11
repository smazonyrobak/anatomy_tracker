from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.atlas_pose_models_v7 import (
    BACKBONES,
    AtlasPoseV7,
    AtlasPoseV7Export,
    atlas_pose_v7_loss,
)
from training.registered_section_dataset import RegisteredSectionDataset
from training.synthetic_atlas import (
    APPEARANCE_MANIFEST_KEYS,
    COHORT_NAMES,
    IMAGE_SIZE,
    SyntheticAtlas,
    load_manifest,
    make_manifest,
    paired_appearance_manifest,
)
from source.atlas_pose_runtime import ATLAS_POSE_PREPROCESSING_VERSION


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("ATLAS_POSE_V7_WORKSPACE", "J:/AtlasPoseTraining_v7"))
ATLAS_FOLDER = Path(os.environ.get("ATLAS_POSE_ATLAS", ROOT / "data" / "Allen Brain Atlas 25um"))
REGISTERED_ROOT = Path(
    os.environ.get("ATLAS_POSE_REGISTERED", WORKSPACE / "allen_s2p_pilot_d100_stratified")
)
SEEDS = {"train": 73191, "validation": 19841, "test": 49157, "paired": 90821}
COMPONENT_SCALES = np.asarray((60.0, 2.0, 2.0), dtype=np.float64)

SCHEDULE = {
    "ablation_20k": 20_000,
    "surviving_heads_100k": 100_000,
    "backbones_100k": 100_000,
    "final_unique_views": 1_000_000,
}
ABLATIONS_20K = {
    "renderer_minimal": {"renderer": "minimal", "head": "direct", "consistency": 0.0, "anatomy": 0.0},
    "renderer_v7": {"renderer": "v7", "head": "direct", "consistency": 0.0, "anatomy": 0.0},
    "head_direct": {"renderer": "v7", "head": "direct", "consistency": 0.15, "anatomy": 0.20},
    "head_binned": {"renderer": "v7", "head": "binned", "consistency": 0.15, "anatomy": 0.20},
    "head_ouv": {"renderer": "v7", "head": "ouv", "consistency": 0.15, "anatomy": 0.20},
    "no_consistency": {"renderer": "v7", "head": "binned", "consistency": 0.0, "anatomy": 0.20},
    "no_anatomy": {"renderer": "v7", "head": "binned", "consistency": 0.15, "anatomy": 0.0},
    "full": {"renderer": "v7", "head": "binned", "consistency": 0.15, "anatomy": 0.20},
}
DEFAULTS = {
    "architecture": "convnextv2_tiny",
    "renderer": "v7",
    "head": "binned",
    "consistency": 0.15,
    "anatomy": 0.20,
    "registered_fraction": 0.20,
    "batch_size": 12,
    "evaluation_batch_size": 24,
    "data_workers": 2,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "warmup_fraction": 0.05,
    "ema_decay": 0.999,
    "gradient_clip": 1.0,
    "validation_interval": 10_000,
    "validation_count": 1_024,
    "early_stopping_patience": 6,
    "early_stopping_min_delta": 0.002,
}


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest_sha256(manifest: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(manifest):
        values = np.ascontiguousarray(manifest[name])
        digest.update(name.encode())
        digest.update(values.dtype.str.encode())
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def ensure_fixed_manifest(workspace: Path, split: str, count: int, seed: int) -> tuple[dict[str, np.ndarray], Path]:
    folder = Path(workspace) / "manifests"
    path = folder / f"{split}_{count}_{seed}.npz"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".part.npz")
        np.savez_compressed(temporary, **make_manifest(count, split, seed))
        os.replace(temporary, path)
    manifest = load_manifest(path)
    record = {"count": count, "seed": seed, "split": split, "sha256": manifest_sha256(manifest)}
    metadata_path = path.with_suffix(".json")
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != record:
        raise RuntimeError(f"Fixed latent manifest changed: {path}")
    metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return manifest, path


def ensure_paired_manifest(
    workspace: Path,
    base: dict[str, np.ndarray],
    split: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], Path]:
    count = len(base["ap_um"])
    folder = Path(workspace) / "manifests"
    path = folder / f"{split}_{count}_{seed}_paired.npz"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".part.npz")
        np.savez_compressed(temporary, **paired_appearance_manifest(base, seed))
        os.replace(temporary, path)
    paired = load_manifest(path)
    for key in base.keys() - set(APPEARANCE_MANIFEST_KEYS):
        if not np.array_equal(base[key], paired[key]):
            raise RuntimeError(f"Paired view changed geometric latent {key}")
    return paired, path


def renderer_variant(manifest: dict[str, np.ndarray], variant: str) -> dict[str, np.ndarray]:
    if variant == "v7":
        return manifest
    result = {key: np.array(value, copy=True) for key, value in manifest.items()}
    result["cohort"].fill(0)
    for key in ("warp", "flaw_mask", "contrast_invert", "sensor_enabled"):
        result[key].fill(False)
    for key in ("occlusion_type", "damage_mode"):
        result[key].fill(0)
    for key in (
        "anatomy_mix", "anatomy_edge_strength", "contrast_offset", "exposure_strength",
        "tile_strength", "tile_seam_strength", "bias_coefficients", "sensor_noise",
        "speck_density", "blowout_strength", "background_level", "background_texture",
    ):
        result[key].fill(0.0)
    for key in ("contrast_gain", "contrast_gamma"):
        result[key].fill(1.0)
    return result


def registered_style(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    severity = float(rng.choice((0.0, 0.35, 0.65, 1.0), p=(0.15, 0.60, 0.20, 0.05)))
    if severity == 0.0:
        return image.copy()
    pixels = image.astype(np.float32) / 255.0
    height, width = pixels.shape[:2]
    y, x = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)].astype(np.float32)
    coefficients = rng.normal(0.0, 0.13 * severity, 6)
    local_bias = coefficients[0] * x + coefficients[1] * y + coefficients[2] * x * y
    local_bias += coefficients[3] * x * x + coefficients[4] * y * y + coefficients[5]
    gamma_xy = rng.normal(0.0, 0.18 * severity, 2)
    local_gamma = np.exp(gamma_xy[0] * x + gamma_xy[1] * y)
    pixels = np.clip(pixels, 0.0, 1.0) ** local_gamma[..., None] if pixels.ndim == 3 else np.clip(pixels, 0.0, 1.0) ** local_gamma
    pixels *= np.exp(local_bias)[..., None] if pixels.ndim == 3 else np.exp(local_bias)

    tile_height, tile_width = rng.integers(32, 97, 2)
    tile_y, tile_x = np.arange(height) // tile_height, np.arange(width) // tile_width
    grid_shape = (int(tile_y.max()) + 1, int(tile_x.max()) + 1)
    tile_gain = np.exp(rng.normal(0.0, 0.11 * severity, grid_shape))[tile_y[:, None], tile_x]
    tile_offset = rng.normal(0.0, 0.045 * severity, grid_shape)[tile_y[:, None], tile_x]
    tile_gamma = np.exp(rng.normal(0.0, 0.09 * severity, grid_shape))[tile_y[:, None], tile_x]
    pixels = np.clip(pixels, 0.0, 1.0) ** (tile_gamma[..., None] if pixels.ndim == 3 else tile_gamma)
    pixels = pixels * (tile_gain[..., None] if pixels.ndim == 3 else tile_gain)
    pixels += tile_offset[..., None] if pixels.ndim == 3 else tile_offset
    seams = ((np.arange(height)[:, None] % tile_height) < 2) | ((np.arange(width)[None, :] % tile_width) < 2)
    pixels += seams[..., None] * rng.uniform(-0.06, 0.08) * severity if pixels.ndim == 3 else seams * rng.uniform(-0.06, 0.08) * severity

    if rng.random() < 0.18:
        pixels = 1.0 - pixels
    if rng.random() < 0.45:
        sigma = float(rng.uniform(0.35, 1.1) * severity)
        pixels = cv2.GaussianBlur(pixels, (0, 0), sigma)
    noise = rng.normal(0.0, rng.uniform(0.004, 0.025) * severity, (height, width))
    pixels += noise[..., None] if pixels.ndim == 3 else noise
    specks = rng.random((height, width)) < rng.uniform(0.00005, 0.0007) * severity
    pixels += specks[..., None] * rng.uniform(0.3, 0.9) if pixels.ndim == 3 else specks * rng.uniform(0.3, 0.9)
    if rng.random() < 0.25:
        center_x, center_y = rng.uniform(-0.8, 0.8, 2)
        radius = rng.uniform(0.02, 0.12)
        blowout = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2.0 * radius**2))
        pixels += blowout[..., None] * rng.uniform(0.25, 0.8) if pixels.ndim == 3 else blowout * rng.uniform(0.25, 0.8)
    return np.clip(pixels * 255.0, 0.0, 255.0).astype(np.uint8)


def build_registered_loaders(
    manifest_root: Path,
    atlas_folder: Path,
    batch_size: int,
    validation_batch_size: int,
    paired: bool,
    workers: int,
) -> tuple[DataLoader, DataLoader]:
    train = RegisteredSectionDataset(
        manifest_root,
        atlas_folder,
        split="train",
        augmentation=registered_style,
        seed=SEEDS["train"],
        views=2 if paired else 1,
    )
    validation = RegisteredSectionDataset(manifest_root, atlas_folder, split="validation", views=1)
    if any(record["split"] != "train" for record in train.records):
        raise RuntimeError("Registered training loader contains a non-training specimen")
    if any(record["split"] != "validation" for record in validation.records):
        raise RuntimeError("Registered validation loader contains a non-validation specimen")
    generator = torch.Generator().manual_seed(SEEDS["train"])
    options = {"num_workers": workers, "pin_memory": torch.cuda.is_available(), "persistent_workers": workers > 0}
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator, **options),
        DataLoader(validation, batch_size=validation_batch_size, shuffle=False, **options),
    )


def vicreg_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    invariance = F.mse_loss(first, second)
    first = first - first.mean(0)
    second = second - second.mean(0)
    denominator = max(first.shape[0] - 1, 1)
    standard_deviation = 0.5 * (
        F.relu(1.0 - torch.sqrt(first.var(0, unbiased=False) + 1e-4)).mean()
        + F.relu(1.0 - torch.sqrt(second.var(0, unbiased=False) + 1e-4)).mean()
    )
    covariance_first = first.T @ first / denominator
    covariance_second = second.T @ second / denominator
    diagonal = torch.eye(first.shape[1], device=first.device, dtype=torch.bool)
    covariance = 0.5 * (
        covariance_first.masked_select(~diagonal).square().mean()
        + covariance_second.masked_select(~diagonal).square().mean()
    )
    return 25.0 * invariance + 25.0 * standard_deviation + covariance


def anatomy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cross_entropy = F.cross_entropy(logits, target)
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).to(probabilities.dtype)
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = (probabilities + one_hot).sum(dim=(0, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return cross_entropy + dice


def training_objective(
    model: torch.nn.Module,
    images: torch.Tensor,
    pose: torch.Tensor,
    orientation: torch.Tensor,
    anatomy: torch.Tensor | None,
    consistency_weight: float,
    anatomy_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    views = images.unbind(1) if images.ndim == 5 else (images,)
    outputs = [model.training_outputs(view) for view in views]
    pose_loss = torch.stack([atlas_pose_v7_loss(output, pose, orientation) for output in outputs]).mean()
    feature_consistency = (
        vicreg_loss(outputs[0]["pooled_features"], outputs[1]["pooled_features"])
        if len(outputs) == 2 and consistency_weight > 0.0
        else pose_loss.new_zeros(())
    )
    prediction_consistency = (
        F.smooth_l1_loss(
            (outputs[0]["image_frame_pose"] - outputs[1]["image_frame_pose"])
            / pose.new_tensor((2500.0, 20.0, 20.0)),
            torch.zeros_like(outputs[0]["image_frame_pose"]),
            beta=0.10,
        )
        if len(outputs) == 2 and consistency_weight > 0.0
        else pose_loss.new_zeros(())
    )
    anatomy_component = (
        torch.stack([anatomy_loss(output["anatomy_logits"], anatomy) for output in outputs]).mean()
        if anatomy is not None and anatomy_weight > 0.0
        else pose_loss.new_zeros(())
    )
    total = pose_loss + consistency_weight * (feature_consistency + prediction_consistency) + anatomy_weight * anatomy_component
    return total, {
        "total": float(total.detach()),
        "pose": float(pose_loss.detach()),
        "feature_consistency": float(feature_consistency.detach()),
        "prediction_consistency": float(prediction_consistency.detach()),
        "anatomy": float(anatomy_component.detach()),
    }


def ema_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def update_ema(
    ema: dict[str, torch.Tensor],
    model: torch.nn.Module,
    decay: float,
    updates: int,
) -> None:
    decay = min(decay, (1.0 + updates) / (10.0 + updates))
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            ema[name].mul_(decay).add_(value, alpha=1.0 - decay)
        else:
            ema[name].copy_(value)


def cosine_learning_rate(step: int, total_steps: int, warmup_steps: int, peak: float) -> float:
    if step < warmup_steps:
        return peak * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def component_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - target
    absolute = np.abs(error)
    slopes = []
    for column in range(3):
        centered = target[:, column] - target[:, column].mean()
        slopes.append(float(np.dot(centered, prediction[:, column] - prediction[:, column].mean()) / max(np.dot(centered, centered), 1e-12)))
    return {
        "count": int(len(target)),
        "mae": absolute.mean(0).tolist(),
        "p95": np.percentile(absolute, 95, axis=0).tolist(),
        "bias": error.mean(0).tolist(),
        "calibration_slope": slopes,
        "selection_score": float(np.mean(absolute.mean(0) / COMPONENT_SCALES)),
    }


def stratified_pose_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    cohort: np.ndarray | None = None,
) -> dict:
    report = {"overall": component_metrics(target, prediction), "ap_500um_bands": {}, "tilt_bands": {}}
    for low in np.arange(-4500.0, 500.0, 500.0):
        high = low + 500.0
        selected = (target[:, 0] >= low) & (target[:, 0] < high if high < 500.0 else target[:, 0] <= high)
        if selected.any():
            report["ap_500um_bands"][f"{low:g}:{high:g}"] = component_metrics(target[selected], prediction[selected])
    magnitude = np.abs(target[:, 1:]).max(1)
    for low, high in ((0.0, 5.0), (5.0, 15.0), (15.0, 25.0), (25.0, 35.0001)):
        selected = (magnitude >= low) & (magnitude < high)
        if selected.any():
            report["tilt_bands"][f"{low:g}:{min(high, 35.0):g}"] = component_metrics(target[selected], prediction[selected])
    if cohort is not None:
        report["artifact_severity"] = {
            str(COHORT_NAMES[level]): component_metrics(target[cohort == level], prediction[cohort == level])
            for level in range(4)
            if np.any(cohort == level)
        }
    return report


def paired_invariance(
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> dict:
    shift = np.abs(first - second)
    degradation = np.abs(second - target) - np.abs(first - target)
    return {
        "mean_absolute_prediction_shift": shift.mean(0).tolist(),
        "p95_absolute_prediction_shift": np.percentile(shift, 95, axis=0).tolist(),
        "mean_absolute_error_change": degradation.mean(0).tolist(),
    }


def registered_report(rows: list[dict]) -> dict:
    target = np.asarray([[row[f"target_{axis}"] for axis in ("ap", "lr", "dv")] for row in rows])
    prediction = np.asarray([[row[f"prediction_{axis}"] for axis in ("ap", "lr", "dv")] for row in rows])
    report = {"overall": component_metrics(target, prediction), "per_specimen": {}, "per_product": {}}
    for field, destination in (("specimen_id", "per_specimen"), ("product", "per_product")):
        for value in sorted({str(row[field]) for row in rows}):
            selected = np.asarray([str(row[field]) == value for row in rows])
            report[destination][value] = component_metrics(target[selected], prediction[selected])
    return report


def validation_selection_metric(rows: list[dict]) -> float:
    if not rows or any(row["split"] != "validation" for row in rows):
        raise RuntimeError("Model selection accepts only non-empty registered validation-specimen rows")
    per_animal = []
    for specimen in sorted({int(row["specimen_id"]) for row in rows}):
        selected = [row for row in rows if int(row["specimen_id"]) == specimen]
        target = np.asarray([[row[f"target_{axis}"] for axis in ("ap", "lr", "dv")] for row in selected])
        prediction = np.asarray([[row[f"prediction_{axis}"] for axis in ("ap", "lr", "dv")] for row in selected])
        per_animal.append(np.abs(prediction - target).mean(0) / COMPONENT_SCALES)
    return float(np.asarray(per_animal).mean())


def animal_bootstrap_comparison(
    target: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    specimen_id: np.ndarray,
    iterations: int = 10_000,
    seed: int = 41171,
) -> dict:
    animals = np.unique(specimen_id)
    component_differences = np.asarray([
        np.abs(candidate[specimen_id == animal] - target[specimen_id == animal]).mean(0)
        - np.abs(reference[specimen_id == animal] - target[specimen_id == animal]).mean(0)
        for animal in animals
    ])
    differences = (component_differences / COMPONENT_SCALES).mean(1)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(differences), (iterations, len(differences)))
    bootstrap = differences[samples].mean(1)
    component_bootstrap = component_differences[samples].mean(1)
    components = {}
    for index, name in enumerate(("ap_um", "lr_deg", "dv_deg")):
        components[name] = {
            "candidate_minus_reference": float(component_differences[:, index].mean()),
            "ci95": np.percentile(component_bootstrap[:, index], (2.5, 97.5)).tolist(),
            "probability_candidate_better": float(np.mean(component_bootstrap[:, index] < 0.0)),
        }
    return {
        "animal_count": int(len(animals)),
        "candidate_minus_reference": float(differences.mean()),
        "ci95": np.percentile(bootstrap, (2.5, 97.5)).tolist(),
        "probability_candidate_better": float(np.mean(bootstrap < 0.0)),
        "components": components,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_diagnostic_plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("AP (um)", "L-R (deg)", "D-V (deg)")
    mae = report["overall"]["mae"]
    p95 = report["overall"]["p95"]
    figure, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    for index, axis in enumerate(axes):
        axis.bar(("MAE", "P95"), (mae[index], p95[index]), color=("#168AAD", "#76C893"))
        axis.set_title(names[index])
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


@torch.inference_mode()
def evaluate_registered(model: torch.nn.Module, loader: DataLoader, device: torch.device, split: str) -> tuple[dict, list[dict]]:
    if split not in {"validation", "test", "sealed_deepslice_s2p"}:
        raise ValueError("Registered evaluation split is not reportable")
    model.eval()
    rows = []
    datasets = loader.dataset.datasets
    for batch in loader:
        image = batch["image"][:, 0] if batch["image"].ndim == 5 else batch["image"]
        prediction = model(image.to(device, non_blocking=True)).float().cpu().numpy()
        target = batch["pose"].numpy()
        for index in range(len(target)):
            experiment = int(batch["experiment_id"][index])
            product = "+".join(map(str, datasets[experiment].get("product_ids", []))) or "unknown"
            rows.append({
                "split": split,
                "specimen_id": int(batch["specimen_id"][index]),
                "experiment_id": experiment,
                "section_image_id": int(batch["section_image_id"][index]),
                "product": product,
                "target_ap": float(target[index, 0]),
                "target_lr": float(target[index, 1]),
                "target_dv": float(target[index, 2]),
                "prediction_ap": float(prediction[index, 0]),
                "prediction_lr": float(prediction[index, 1]),
                "prediction_dv": float(prediction[index, 2]),
            })
    return registered_report(rows), rows


@torch.inference_mode()
def evaluate_synthetic(
    model: torch.nn.Module,
    renderer: SyntheticAtlas,
    manifest: dict[str, np.ndarray],
    paired: dict[str, np.ndarray] | None,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    model.eval()
    predictions, paired_predictions, targets = [], [], []
    for start in range(0, len(manifest["ap_um"]), batch_size):
        count = min(batch_size, len(manifest["ap_um"]) - start)
        image, _, target = renderer.batch(manifest, start, count)
        predictions.append(model(image).float().cpu().numpy())
        targets.append(target.float().cpu().numpy())
        if paired is not None:
            paired_image, _, _ = renderer.batch(paired, start, count)
            paired_predictions.append(model(paired_image).float().cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    report = stratified_pose_metrics(target, prediction, manifest["cohort"])
    if paired_predictions:
        report["artifact_invariance"] = paired_invariance(target, prediction, np.concatenate(paired_predictions))
    rows = [
        {
            "latent_index": index,
            "cohort": str(COHORT_NAMES[int(manifest["cohort"][index])]),
            "target_ap": float(target[index, 0]),
            "target_lr": float(target[index, 1]),
            "target_dv": float(target[index, 2]),
            "prediction_ap": float(prediction[index, 0]),
            "prediction_lr": float(prediction[index, 1]),
            "prediction_dv": float(prediction[index, 2]),
        }
        for index in range(len(target))
    ]
    return report, rows


def _registered_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["image"].to(device, non_blocking=True),
        batch["pose"].to(device, non_blocking=True),
        torch.zeros(len(batch["pose"]), device=device),
        batch["anatomy"].to(device, non_blocking=True),
    )


def _synthetic_batch(
    renderer: SyntheticAtlas,
    manifest: dict[str, np.ndarray],
    paired: dict[str, np.ndarray] | None,
    start: int,
    count: int,
    anatomy_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    rendered = renderer.batch(manifest, start, count, return_anatomy=anatomy_enabled)
    image, _, pose = rendered[:3]
    anatomy = rendered[3] if anatomy_enabled else None
    if paired is not None:
        second = renderer.batch(paired, start, count)[0]
        image = torch.stack((image, second), dim=1)
    rotation = torch.from_numpy((np.abs(manifest["rotation_deg"][start : start + count]) > 90.0).astype(np.float32)).to(image.device)
    return image, pose, rotation, anatomy


def _swap_to_ema(model: torch.nn.Module, ema: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    current = ema_state(model)
    model.load_state_dict(ema)
    return current


def train_experiment(
    model: torch.nn.Module,
    renderer: SyntheticAtlas,
    train_manifest: dict[str, np.ndarray],
    paired_manifest: dict[str, np.ndarray] | None,
    validation_manifest: dict[str, np.ndarray],
    validation_paired: dict[str, np.ndarray] | None,
    registered_train: DataLoader,
    registered_validation: DataLoader,
    config: dict,
    run_folder: Path,
    device: torch.device,
) -> dict:
    run_folder.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    synthetic_steps = math.ceil(len(train_manifest["ap_um"]) / config["batch_size"])
    total_steps = math.ceil(synthetic_steps / (1.0 - config["registered_fraction"]))
    warmup_steps = round(total_steps * config["warmup_fraction"])
    ema = ema_state(model)
    rng = np.random.default_rng(SEEDS["train"])
    registered_iterator = iter(registered_train)
    synthetic_start = 0
    step = 0
    next_validation = min(config["validation_interval"], len(train_manifest["ap_um"]))
    best_score = float("inf")
    stale = 0
    history = []
    best_checkpoint = run_folder / "best.pt"

    while synthetic_start < len(train_manifest["ap_um"]):
        use_registered = bool(rng.random() < config["registered_fraction"])
        if use_registered:
            try:
                batch = next(registered_iterator)
            except StopIteration:
                registered_iterator = iter(registered_train)
                batch = next(registered_iterator)
            images, pose, orientation, anatomy = _registered_batch(batch, device)
        else:
            count = min(config["batch_size"], len(train_manifest["ap_um"]) - synthetic_start)
            images, pose, orientation, anatomy = _synthetic_batch(
                renderer,
                train_manifest,
                paired_manifest,
                synthetic_start,
                count,
                config["anatomy"] > 0.0,
            )
            synthetic_start += count

        learning_rate = cosine_learning_rate(step, total_steps, warmup_steps, config["learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss, components = training_objective(
                model,
                images,
                pose,
                orientation,
                anatomy,
                config["consistency"],
                config["anatomy"],
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        step += 1
        update_ema(ema, model, config["ema_decay"], step)

        if synthetic_start >= next_validation or synthetic_start == len(train_manifest["ap_um"]):
            current = _swap_to_ema(model, ema)
            synthetic_metrics, _ = evaluate_synthetic(
                model, renderer, validation_manifest, validation_paired, config["evaluation_batch_size"]
            )
            registered_metrics, registered_rows = evaluate_registered(
                model, registered_validation, device, "validation"
            )
            score = validation_selection_metric(registered_rows)
            model.load_state_dict(current)
            record = {
                "step": step,
                "unique_synthetic_views": synthetic_start,
                "learning_rate": learning_rate,
                "training": components,
                "validation_selection_score": score,
                "synthetic": synthetic_metrics,
                "registered": registered_metrics,
            }
            history.append(record)
            (run_folder / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if score < best_score - config["early_stopping_min_delta"]:
                best_score = score
                stale = 0
                torch.save({"model": {key: value.cpu() for key, value in ema.items()}, "config": config, "record": record}, best_checkpoint)
                _write_csv(run_folder / "validation_registered.csv", registered_rows)
                write_diagnostic_plot(registered_metrics, run_folder / "validation_registered.png")
            else:
                stale += 1
            next_validation += config["validation_interval"]
            if stale >= config["early_stopping_patience"]:
                break

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return {
        "best_selection_score": best_score,
        "selection_split": "validation",
        "best_checkpoint": str(best_checkpoint),
        "steps": step,
        "unique_synthetic_views": synthetic_start,
        "stopped_early": synthetic_start < len(train_manifest["ap_um"]),
    }


def export_onnx(
    model: torch.nn.Module,
    output_folder: Path,
    metadata: dict,
    example: torch.Tensor | None = None,
) -> dict:
    output_folder.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    wrapper = AtlasPoseV7Export(model).eval()
    example = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE) if example is None else example.cpu()
    reference = [value.detach().numpy() for value in wrapper(example)]
    temporary = output_folder / "atlas_pose.part.onnx"
    model_path = output_folder / "atlas_pose.onnx"
    torch.onnx.export(
        wrapper,
        example,
        temporary,
        input_names=["images"],
        output_names=["pose_ap_um_lr_deg_dv_deg", "orientation_inverted_logit"],
        dynamic_axes={
            "images": {0: "batch"},
            "pose_ap_um_lr_deg_dv_deg": {0: "batch"},
            "orientation_inverted_logit": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(temporary))
    session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].name != "images" or session.get_inputs()[0].shape != ["batch", 3, IMAGE_SIZE, IMAGE_SIZE]:
        raise RuntimeError("Exported input contract differs from the tracker runtime")
    if [output.name for output in session.get_outputs()] != ["pose_ap_um_lr_deg_dv_deg", "orientation_inverted_logit"]:
        raise RuntimeError("Exported output contract differs from the tracker runtime")
    actual = session.run(None, {"images": example.numpy()})
    differences = [float(np.max(np.abs(left - right))) for left, right in zip(reference, actual)]
    if differences[0] > 0.05 or differences[1] > 1e-4:
        raise RuntimeError(f"ONNX verification failed: {differences}")
    os.replace(temporary, model_path)
    metadata = {
        **metadata,
        "sha256": file_sha256(model_path),
        "input": {"name": "images", "shape": ["batch", 3, IMAGE_SIZE, IMAGE_SIZE]},
        "outputs": [
            {"name": "pose_ap_um_lr_deg_dv_deg", "shape": ["batch", 3]},
            {"name": "orientation_inverted_logit", "shape": ["batch"]},
        ],
        "onnx_opset": 17,
        "onnxruntime_version": ort.__version__,
        "preprocessing_version": ATLAS_POSE_PREPROCESSING_VERSION,
        "preprocessing_source_sha256": file_sha256(ROOT / "source" / "atlas_pose_runtime.py"),
        "architecture": metadata.get("config", {}).get("architecture"),
        "pose_representation": metadata.get("config", {}).get("head"),
        "verification_max_absolute_difference": differences,
        "source_sha256": {
            path.name: file_sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("atlas_pose_models_v7.py"),
                Path(__file__).with_name("synthetic_atlas.py"),
                Path(__file__).with_name("registered_section_dataset.py"),
            )
        },
    }
    (output_folder / "atlas_pose.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    provenance = {
        "model_sha256": metadata["sha256"],
        "source_sha256": metadata["source_sha256"],
        "manifest_sha256": metadata.get("manifest_sha256"),
        "selection_split": metadata.get("selection_split"),
        "training_splits": ["synthetic_train", "registered_train"],
        "selection_data": "synthetic_validation diagnostics and registered validation specimens",
        "excluded_from_selection": ["registered_test", "sealed_deepslice_s2p"],
    }
    (output_folder / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return metadata


def promote_export(export_folder: Path, destination: Path = ROOT / "models" / "AtlasPose") -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("atlas_pose.onnx", "atlas_pose.json"):
        shutil.copy2(Path(export_folder) / name, destination / name)


def experiment_config(name: str, samples: int, overrides: dict | None = None) -> dict:
    config = {**DEFAULTS, **(overrides or {})}
    config.update({"name": name, "samples": int(samples)})
    return config


def run_experiment(config: dict, export: bool = False) -> dict:
    run_folder = WORKSPACE / "runs" / config["name"]
    train, train_path = ensure_fixed_manifest(WORKSPACE, "train", config["samples"], SEEDS["train"])
    validation, validation_path = ensure_fixed_manifest(
        WORKSPACE, "validation", config["validation_count"], SEEDS["validation"]
    )
    train = renderer_variant(train, config["renderer"])
    validation = renderer_variant(validation, config["renderer"])
    paired_train = paired_validation = None
    paired_paths = []
    if config["consistency"] > 0.0:
        paired_train, paired_train_path = ensure_paired_manifest(WORKSPACE, train, "train", SEEDS["paired"])
        paired_validation, paired_validation_path = ensure_paired_manifest(
            WORKSPACE, validation, "validation", SEEDS["paired"] + 1
        )
        paired_paths = [paired_train_path, paired_validation_path]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = SyntheticAtlas(ATLAS_FOLDER, str(device))
    registered_train, registered_validation = build_registered_loaders(
        REGISTERED_ROOT,
        ATLAS_FOLDER,
        config["batch_size"],
        config["evaluation_batch_size"],
        paired=config["consistency"] > 0.0,
        workers=config["data_workers"],
    )
    torch.manual_seed(SEEDS["train"])
    model = AtlasPoseV7(config["architecture"], pretrained=True, pose_representation=config["head"])
    result = train_experiment(
        model,
        renderer,
        train,
        paired_train,
        validation,
        paired_validation,
        registered_train,
        registered_validation,
        config,
        run_folder,
        device,
    )
    result.update({
        "config": config,
        "manifest_sha256": {
            str(path): file_sha256(path) for path in (train_path, validation_path, *paired_paths)
        },
        "registered_data": {
            "root": str(REGISTERED_ROOT),
            "sha256": {
                name: file_sha256(REGISTERED_ROOT / name)
                for name in ("datasets.jsonl", "sections.jsonl", "provenance.json")
            },
            "training_split": "train",
            "selection_split": "validation",
            "excluded_from_selection": ["test", "sealed_deepslice_s2p"],
        },
    })
    (run_folder / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if export:
        export_onnx(model, run_folder / "export", result)
    return result


def _winner(results: list[dict]) -> dict:
    if any(result["selection_split"] != "validation" for result in results):
        raise RuntimeError("Only validation results may select an experiment")
    return min(results, key=lambda result: result["best_selection_score"])


def bootstrap_run_comparison(candidate: dict, reference: dict, output: Path) -> dict:
    def read(path: Path) -> dict[int, dict]:
        with path.open(newline="", encoding="utf-8") as stream:
            return {int(row["section_image_id"]): row for row in csv.DictReader(stream)}

    candidate_rows = read(Path(candidate["best_checkpoint"]).parent / "validation_registered.csv")
    reference_rows = read(Path(reference["best_checkpoint"]).parent / "validation_registered.csv")
    identifiers = sorted(candidate_rows.keys() & reference_rows.keys())
    target = np.asarray([[float(candidate_rows[key][f"target_{axis}"]) for axis in ("ap", "lr", "dv")] for key in identifiers])
    candidate_pose = np.asarray([[float(candidate_rows[key][f"prediction_{axis}"]) for axis in ("ap", "lr", "dv")] for key in identifiers])
    reference_pose = np.asarray([[float(reference_rows[key][f"prediction_{axis}"]) for axis in ("ap", "lr", "dv")] for key in identifiers])
    specimens = np.asarray([int(candidate_rows[key]["specimen_id"]) for key in identifiers])
    result = animal_bootstrap_comparison(target, candidate_pose, reference_pose, specimens)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def held_out_reports(result: dict) -> dict:
    config = result["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AtlasPoseV7(config["architecture"], pretrained=False, pose_representation=config["head"]).to(device)
    checkpoint = torch.load(result["best_checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    run_folder = Path(result["best_checkpoint"]).parent
    reports = {}
    for split in ("test", "sealed_deepslice_s2p"):
        dataset = RegisteredSectionDataset(
            REGISTERED_ROOT,
            ATLAS_FOLDER,
            split=split,
            include_sealed=split == "sealed_deepslice_s2p",
        )
        loader = DataLoader(
            dataset,
            batch_size=config["evaluation_batch_size"],
            shuffle=False,
            num_workers=config["data_workers"],
        )
        report, rows = evaluate_registered(model, loader, device, split)
        reports[split] = report
        _write_csv(run_folder / f"{split}_registered.csv", rows)
        write_diagnostic_plot(report, run_folder / f"{split}_registered.png")
    test_manifest, _ = ensure_fixed_manifest(WORKSPACE, "test", 8_192, SEEDS["test"])
    paired_test, _ = ensure_paired_manifest(WORKSPACE, test_manifest, "test", SEEDS["paired"] + 2)
    renderer = SyntheticAtlas(ATLAS_FOLDER, str(device))
    synthetic_report, synthetic_rows = evaluate_synthetic(
        model, renderer, test_manifest, paired_test, config["evaluation_batch_size"]
    )
    reports["synthetic_test"] = synthetic_report
    _write_csv(run_folder / "synthetic_test.csv", synthetic_rows)
    (run_folder / "held_out_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports


def run_schedule() -> dict:
    ablations = [
        run_experiment(experiment_config(f"20k_{name}", SCHEDULE["ablation_20k"], values))
        for name, values in ABLATIONS_20K.items()
    ]
    head_scores = {
        head: min(
            result["best_selection_score"]
            for result in ablations
            if result["config"]["head"] == head and result["config"]["renderer"] == "v7"
        )
        for head in ("direct", "binned", "ouv")
    }
    surviving_heads = sorted(head_scores, key=head_scores.get)[:2]
    head_results = [
        run_experiment(
            experiment_config(
                f"100k_head_{head}",
                SCHEDULE["surviving_heads_100k"],
                {"head": head},
            )
        )
        for head in surviving_heads
    ]
    winning_head_result = _winner(head_results)
    best_head = winning_head_result["config"]["head"]
    head_comparison = bootstrap_run_comparison(
        winning_head_result,
        next(result for result in head_results if result is not winning_head_result),
        WORKSPACE / "head_animal_bootstrap.json",
    )
    backbone_results = [
        run_experiment(
            experiment_config(
                f"100k_backbone_{architecture}_{best_head}",
                SCHEDULE["backbones_100k"],
                {"architecture": architecture, "head": best_head},
            )
        )
        for architecture in BACKBONES
    ]
    winning_backbone_result = _winner(backbone_results)
    best_backbone = winning_backbone_result["config"]["architecture"]
    runner_up_backbone = sorted(backbone_results, key=lambda result: result["best_selection_score"])[1]
    backbone_comparison = bootstrap_run_comparison(
        winning_backbone_result,
        runner_up_backbone,
        WORKSPACE / "backbone_animal_bootstrap.json",
    )
    final = run_experiment(
        experiment_config(
            f"final_{best_backbone}_{best_head}",
            SCHEDULE["final_unique_views"],
            {"architecture": best_backbone, "head": best_head, "validation_interval": 50_000},
        ),
        export=True,
    )
    final_reports = held_out_reports(final)
    summary = {
        "ablation_winner": _winner(ablations)["config"]["name"],
        "surviving_heads": surviving_heads,
        "best_head": best_head,
        "best_backbone": best_backbone,
        "head_animal_bootstrap": head_comparison,
        "backbone_animal_bootstrap": backbone_comparison,
        "held_out_reports": final_reports,
        "final": final,
    }
    (WORKSPACE / "schedule_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    name = os.environ.get("ATLAS_POSE_V7_EXPERIMENT")
    if name:
        overrides = ABLATIONS_20K.get(name, {})
        samples = int(os.environ.get("ATLAS_POSE_V7_SAMPLES", SCHEDULE["ablation_20k"]))
        config = experiment_config(name, samples, overrides)
        environment_overrides = {
            "ATLAS_POSE_V7_ARCHITECTURE": ("architecture", str),
            "ATLAS_POSE_V7_HEAD": ("head", str),
            "ATLAS_POSE_V7_RENDERER": ("renderer", str),
            "ATLAS_POSE_V7_BATCH_SIZE": ("batch_size", int),
            "ATLAS_POSE_V7_EVALUATION_BATCH_SIZE": ("evaluation_batch_size", int),
            "ATLAS_POSE_V7_DATA_WORKERS": ("data_workers", int),
            "ATLAS_POSE_V7_VALIDATION_COUNT": ("validation_count", int),
            "ATLAS_POSE_V7_CONSISTENCY": ("consistency", float),
            "ATLAS_POSE_V7_ANATOMY": ("anatomy", float),
            "ATLAS_POSE_V7_REGISTERED_FRACTION": ("registered_fraction", float),
        }
        for variable, (key, conversion) in environment_overrides.items():
            if variable in os.environ:
                config[key] = conversion(os.environ[variable])
        run_experiment(config, export=os.environ.get("ATLAS_POSE_V7_EXPORT") == "1")
    else:
        run_schedule()


if __name__ == "__main__":
    main()
