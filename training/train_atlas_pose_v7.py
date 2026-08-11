from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.atlas_pose_models_v7 import (
    BACKBONES,
    PHYSICAL_POSE_LOSS_SCALE,
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
from source.atlas_pose_runtime import (
    ATLAS_POSE_PREPROCESSING_VERSION,
    atlas_pose_preprocessing_contract_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("ATLAS_POSE_V7_WORKSPACE", "J:/AtlasPoseTraining_v7"))
ATLAS_FOLDER = Path(os.environ.get("ATLAS_POSE_ATLAS", ROOT / "data" / "Allen Brain Atlas 25um"))
REGISTERED_ROOT = Path(
    os.environ.get("ATLAS_POSE_REGISTERED", WORKSPACE / "allen_registered_full_20260811")
)
SEEDS = {"train": 73191, "validation": 19841, "test": 49157, "paired": 90821}
COMPARISON_SEEDS = (73191, 41777, 90217)
FINAL_GATE_THRESHOLDS = {
    "mean_ap_um": 60.0,
    "mean_lr_deg": 0.90,
    "mean_dv_deg": 1.75,
    "absolute_ap_bias_um": 25.0,
    "ap_p95_um": 150.0,
    "worst_ap_band_mae_um": 90.0,
    "worst_product_mae_um": 90.0,
}
COMPONENT_SCALES = np.asarray(PHYSICAL_POSE_LOSS_SCALE, dtype=np.float64)
VALIDATION_COMPONENT_GATES = np.asarray(
    tuple(FINAL_GATE_THRESHOLDS[name] for name in ("mean_ap_um", "mean_lr_deg", "mean_dv_deg")),
    dtype=np.float64,
)
FAMILY_CONFIDENCE = 0.95
HEAD_TIE_PRIORITY = ("binned", "direct", "ouv")
BACKBONE_TIE_PRIORITY = ("convnext_tiny", "maxvit_tiny", "xception")
_SYNTHETIC_VALIDATION_CACHE: dict[
    tuple[str, str, str | None, int],
    tuple[torch.Tensor, torch.Tensor | None, torch.Tensor],
] = {}

SCHEDULE = {
    "head_screen_20k": 20_000,
    "ablation_20k": 20_000,
    "surviving_heads_100k": 100_000,
    "backbones_100k": 100_000,
    "final_unique_views": 1_000_000,
}
ABLATIONS_20K = {
    "renderer_minimal": {"renderer": "minimal"},
    "no_consistency": {"consistency": 0.0},
    "no_anatomy": {"anatomy": 0.0},
}
DEFAULTS = {
    "architecture": "convnext_tiny",
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


def atlas_data_hashes(folder: Path) -> dict[str, str]:
    names = ("average_template_25.nrrd", "annotation_25.nrrd", "query.csv", "atlas_labels.pkl")
    missing = [name for name in names if not (Path(folder) / name).is_file()]
    if missing:
        raise RuntimeError(f"Atlas data is incomplete; missing {missing}")
    return {name: file_sha256(Path(folder) / name) for name in names}


def module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def pretrained_backbone_provenance(model: AtlasPoseV7, architecture: str) -> dict:
    config = dict(model.encoder.backbone.pretrained_cfg)
    license_name = str(config.get("license", "")).lower()
    if license_name != "apache-2.0":
        raise RuntimeError(
            f"AtlasPose candidate {BACKBONES[architecture]} has non-approved pretrained license "
            f"{license_name or 'missing'}"
        )
    return {
        "candidate": architecture,
        "timm_model_id": BACKBONES[architecture],
        "architecture": config.get("architecture"),
        "tag": config.get("tag"),
        "url": config.get("url"),
        "hf_hub_id": config.get("hf_hub_id"),
        "license": license_name,
        "initial_state_sha256": module_state_sha256(model.encoder.backbone),
    }


def training_environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def registered_data_hashes(root: Path) -> dict[str, str]:
    names = ("datasets.jsonl", "sections.jsonl", "provenance.json", "downloads.jsonl")
    missing = [name for name in names if not (Path(root) / name).is_file()]
    if missing:
        raise RuntimeError(f"Registered dataset is incomplete; missing {missing}")
    return {name: file_sha256(Path(root) / name) for name in names}


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
    path = folder / f"{split}_{count}_{seed}_{manifest_sha256(base)[:12]}_paired.npz"
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
    if variant != "minimal":
        raise ValueError(f"Unknown AtlasPose renderer variant: {variant}")
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
    seed: int,
    anatomy_enabled: bool,
) -> tuple[DataLoader, DataLoader]:
    train = RegisteredSectionDataset(
        manifest_root,
        atlas_folder,
        split="train",
        augmentation=registered_style,
        seed=seed,
        views=2 if paired else 1,
        include_anatomy=anatomy_enabled,
        cache_static=anatomy_enabled,
    )
    validation = RegisteredSectionDataset(
        manifest_root,
        atlas_folder,
        split="validation",
        views=1,
        include_anatomy=False,
        cache_images=True,
    )
    if any(record["split"] != "train" for record in train.records):
        raise RuntimeError("Registered training loader contains a non-training specimen")
    if any(record["split"] != "validation" for record in validation.records):
        raise RuntimeError("Registered validation loader contains a non-validation specimen")
    generator = torch.Generator().manual_seed(seed)
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
    include_anatomy = anatomy is not None and anatomy_weight > 0.0
    outputs = [model.training_outputs(view, include_anatomy=include_anatomy) for view in views]
    pose_terms = [
        atlas_pose_v7_loss(output, pose, orientation, return_components=True)
        for output in outputs
    ]
    pose_loss = torch.stack([total for total, _ in pose_terms]).mean()
    pose_components = {
        name: torch.stack([components[name] for _, components in pose_terms]).mean()
        for name in pose_terms[0][1]
    }
    feature_consistency = (
        vicreg_loss(outputs[0]["pooled_features"], outputs[1]["pooled_features"])
        if len(outputs) == 2 and consistency_weight > 0.0
        else pose_loss.new_zeros(())
    )
    prediction_consistency = (
        F.smooth_l1_loss(
            (outputs[0]["image_frame_pose"] - outputs[1]["image_frame_pose"])
            / pose.new_tensor(COMPONENT_SCALES),
            torch.zeros_like(outputs[0]["image_frame_pose"]),
            beta=1.0,
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
        **{name: float(value.detach()) for name, value in pose_components.items()},
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
    groups: dict[tuple[torch.device, torch.dtype], tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            targets, sources = groups.setdefault((value.device, value.dtype), ([], []))
            targets.append(ema[name])
            sources.append(value)
        else:
            ema[name].copy_(value)
    for targets, sources in groups.values():
        torch._foreach_mul_(targets, decay)
        torch._foreach_add_(targets, sources, alpha=1.0 - decay)


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
    report = {
        "overall": component_metrics(target, prediction),
        "per_specimen": {},
        "per_product": {},
        "ap_500um_bands": {},
    }
    for field, destination in (("specimen_id", "per_specimen"), ("product", "per_product")):
        for value in sorted({str(row[field]) for row in rows}):
            selected = np.asarray([str(row[field]) == value for row in rows])
            report[destination][value] = component_metrics(target[selected], prediction[selected])
    for band in sorted({_ap_500um_band(float(row["target_ap"])) for row in rows}):
        selected = np.asarray([_ap_500um_band(float(row["target_ap"])) == band for row in rows])
        report["ap_500um_bands"][band] = component_metrics(target[selected], prediction[selected])
    report["worst_ap_bin"] = max(
        report["ap_500um_bands"],
        key=lambda band: report["ap_500um_bands"][band]["selection_score"],
    )
    return report


def _row_in_training_ap_domain(row: dict) -> bool:
    expected = -4500.0 <= float(row["target_ap"]) <= 500.0
    if "in_training_ap_domain" not in row:
        return expected
    recorded = row["in_training_ap_domain"]
    if isinstance(recorded, str):
        recorded = recorded.casefold() == "true"
    if bool(recorded) != expected:
        raise RuntimeError("Registered AP-domain label disagrees with the physical target")
    return expected


def registered_domain_reports(rows: list[dict]) -> dict:
    in_domain = [row for row in rows if _row_in_training_ap_domain(row)]
    out_of_domain = [row for row in rows if not _row_in_training_ap_domain(row)]
    if not in_domain:
        raise RuntimeError("Registered evaluation contains no sections in the trained AP domain")
    return {
        "primary_in_training_ap_domain": registered_report(in_domain),
        "out_of_domain": registered_report(out_of_domain) if out_of_domain else None,
        "counts": {"in_training_ap_domain": len(in_domain), "out_of_domain": len(out_of_domain)},
    }


def _ap_500um_band(ap_um: float) -> str:
    index = 9 if ap_um == 500.0 else int(math.floor((ap_um + 4500.0) / 500.0))
    low = -4500 + 500 * index
    return f"{low}:{low + 500}"


def balanced_animal_component_errors(
    rows: list[dict],
    required_split: str = "validation",
) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for row in rows if _row_in_training_ap_domain(row)]
    if not rows or any(row["split"] != required_split for row in rows):
        raise RuntimeError(
            f"Expected non-empty registered {required_split}-specimen rows"
        )
    animals = np.asarray(sorted({int(row["specimen_id"]) for row in rows}))
    per_animal = []
    for specimen in animals:
        specimen_rows = [row for row in rows if int(row["specimen_id"]) == specimen]
        per_bin = []
        for band in sorted({_ap_500um_band(float(row["target_ap"])) for row in specimen_rows}):
            selected = [
                row for row in specimen_rows if _ap_500um_band(float(row["target_ap"])) == band
            ]
            target = np.asarray(
                [[float(row[f"target_{axis}"]) for axis in ("ap", "lr", "dv")] for row in selected]
            )
            prediction = np.asarray(
                [[float(row[f"prediction_{axis}"]) for axis in ("ap", "lr", "dv")] for row in selected]
            )
            per_bin.append(np.abs(prediction - target).mean(0))
        per_animal.append(np.asarray(per_bin).mean(0))
    return animals, np.asarray(per_animal)


def _selection_summary_from_animal_errors(errors: np.ndarray) -> dict:
    component_mae = errors.mean(0)
    passed = component_mae <= VALIDATION_COMPONENT_GATES
    return {
        "component_mae": dict(zip(("ap_um", "lr_deg", "dv_deg"), component_mae.tolist())),
        "component_gates": dict(
            zip(("ap_um", "lr_deg", "dv_deg"), VALIDATION_COMPONENT_GATES.tolist())
        ),
        "component_passed": dict(zip(("ap_um", "lr_deg", "dv_deg"), passed.tolist())),
        "all_mean_gates_passed": bool(passed.all()),
        "worst_gate_ratio": float(np.max(component_mae / VALIDATION_COMPONENT_GATES)),
        "composite_score": float((errors / COMPONENT_SCALES).mean()),
    }


def validation_selection_summary(rows: list[dict]) -> dict:
    _, errors = balanced_animal_component_errors(rows)
    return _selection_summary_from_animal_errors(errors)


def _balanced_ap_bias(rows: list[dict], required_split: str) -> float:
    rows = [row for row in rows if _row_in_training_ap_domain(row)]
    animals = sorted({int(row["specimen_id"]) for row in rows})
    per_animal = []
    for specimen in animals:
        specimen_rows = [row for row in rows if int(row["specimen_id"]) == specimen]
        per_bin = []
        for band in sorted({_ap_500um_band(float(row["target_ap"])) for row in specimen_rows}):
            selected = [
                row for row in specimen_rows if _ap_500um_band(float(row["target_ap"])) == band
            ]
            per_bin.append(
                np.mean(
                    [float(row["prediction_ap"]) - float(row["target_ap"]) for row in selected]
                )
            )
        per_animal.append(np.mean(per_bin))
    if not rows or any(row["split"] != required_split for row in rows):
        raise RuntimeError(f"Expected non-empty registered {required_split}-specimen rows")
    return float(abs(np.mean(per_animal)))


def _worst_group_ap_mae(rows: list[dict], group) -> tuple[str, float]:
    groups = sorted({str(group(row)) for row in rows})
    values = {}
    for value in groups:
        selected = [row for row in rows if str(group(row)) == value]
        specimens = sorted({int(row["specimen_id"]) for row in selected})
        values[value] = float(
            np.mean(
                [
                    np.mean(
                        [
                            abs(float(row["prediction_ap"]) - float(row["target_ap"]))
                            for row in selected
                            if int(row["specimen_id"]) == specimen
                        ]
                    )
                    for specimen in specimens
                ]
            )
        )
    worst = max(values, key=values.get)
    return worst, values[worst]


def final_acceptance_summary(rows: list[dict], required_split: str) -> dict:
    in_domain = [row for row in rows if _row_in_training_ap_domain(row)]
    _, errors = balanced_animal_component_errors(in_domain, required_split)
    selection = _selection_summary_from_animal_errors(errors)
    ap_absolute = np.asarray(
        [abs(float(row["prediction_ap"]) - float(row["target_ap"])) for row in in_domain]
    )
    worst_band, worst_band_mae = _worst_group_ap_mae(
        in_domain,
        lambda row: _ap_500um_band(float(row["target_ap"])),
    )
    worst_product, worst_product_mae = _worst_group_ap_mae(
        in_domain,
        lambda row: row["product"],
    )
    component = selection["component_mae"]
    values = {
        "mean_ap_um": component["ap_um"],
        "mean_lr_deg": component["lr_deg"],
        "mean_dv_deg": component["dv_deg"],
        "absolute_ap_bias_um": _balanced_ap_bias(in_domain, required_split),
        "ap_p95_um": float(np.percentile(ap_absolute, 95.0)),
        "worst_ap_band_mae_um": worst_band_mae,
        "worst_product_mae_um": worst_product_mae,
    }
    passed = {
        name: bool(values[name] <= threshold)
        for name, threshold in FINAL_GATE_THRESHOLDS.items()
    }
    return {
        "split": required_split,
        "in_training_ap_domain_count": len(in_domain),
        "values": values,
        "thresholds": dict(FINAL_GATE_THRESHOLDS),
        "passed": passed,
        "all_gates_passed": all(passed.values()),
        "worst_gate_ratio": max(
            values[name] / threshold for name, threshold in FINAL_GATE_THRESHOLDS.items()
        ),
        "worst_ap_band": worst_band,
        "worst_product": worst_product,
        "selection_summary": selection,
    }


def validation_selection_key(summary: dict) -> tuple[float, float, float]:
    return (
        summary["composite_score"],
        summary["worst_gate_ratio"],
        max(summary["component_mae"].values()),
    )


def validation_selection_improved(
    candidate: tuple[float, float, float],
    reference: tuple[float, float, float],
    minimum_delta: float,
) -> bool:
    for candidate_value, reference_value in zip(candidate, reference):
        if candidate_value < reference_value - minimum_delta:
            return True
        if candidate_value > reference_value + minimum_delta:
            return False
    return False


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
    if split not in {"validation", "test"}:
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
                "in_training_ap_domain": bool(batch["in_training_ap_domain"][index]),
            })
    return registered_report(rows), rows


@torch.inference_mode()
def _synthetic_validation_images(
    renderer: SyntheticAtlas,
    manifest: dict[str, np.ndarray],
    paired: dict[str, np.ndarray] | None,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    key = (
        str(renderer.atlas_folder),
        manifest_sha256(manifest),
        manifest_sha256(paired) if paired is not None else None,
        int(batch_size),
    )
    if key not in _SYNTHETIC_VALIDATION_CACHE:
        images, paired_images, targets = [], [], []
        for start in range(0, len(manifest["ap_um"]), batch_size):
            count = min(batch_size, len(manifest["ap_um"]) - start)
            image, _, target = renderer.batch(manifest, start, count)
            images.append(image[:, 0].float().cpu().clone())
            targets.append(target.float().cpu())
            if paired is not None:
                paired_image = renderer.batch(paired, start, count)[0]
                paired_images.append(paired_image[:, 0].float().cpu().clone())
        _SYNTHETIC_VALIDATION_CACHE[key] = (
            torch.cat(images),
            torch.cat(paired_images) if paired_images else None,
            torch.cat(targets),
        )
    return _SYNTHETIC_VALIDATION_CACHE[key]


@torch.inference_mode()
def evaluate_synthetic(
    model: torch.nn.Module,
    renderer: SyntheticAtlas,
    manifest: dict[str, np.ndarray],
    paired: dict[str, np.ndarray] | None,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    model.eval()
    cached_images, cached_paired, cached_targets = _synthetic_validation_images(
        renderer, manifest, paired, batch_size
    )
    predictions, paired_predictions = [], []
    for start in range(0, len(manifest["ap_um"]), batch_size):
        count = min(batch_size, len(manifest["ap_um"]) - start)
        image = cached_images[start : start + count].to(renderer.device)
        image = image[:, None].repeat(1, 3, 1, 1)
        predictions.append(model(image).float().cpu().numpy())
        if cached_paired is not None:
            paired_image = cached_paired[start : start + count].to(renderer.device)
            paired_image = paired_image[:, None].repeat(1, 3, 1, 1)
            paired_predictions.append(model(paired_image).float().cpu().numpy())
    prediction = np.concatenate(predictions)
    target = cached_targets.numpy()
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


def _registered_batch(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return (
        batch["image"].to(device, non_blocking=True),
        batch["pose"].to(device, non_blocking=True),
        torch.zeros(len(batch["pose"]), device=device),
        batch["anatomy"].to(device, non_blocking=True) if "anatomy" in batch else None,
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
    rng = np.random.default_rng(config["training_seed"])
    registered_iterator = iter(registered_train)
    synthetic_start = 0
    step = 0
    next_validation = min(config["validation_interval"], len(train_manifest["ap_um"]))
    best_score = float("inf")
    best_selection_key = (float("inf"), float("inf"), float("inf"))
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
            selection = validation_selection_summary(registered_rows)
            final_gate = final_acceptance_summary(registered_rows, "validation")
            score = selection["composite_score"]
            selection_key = validation_selection_key(selection)
            model.load_state_dict(current)
            record = {
                "step": step,
                "unique_synthetic_views": synthetic_start,
                "learning_rate": learning_rate,
                "training": components,
                "validation_selection_score": score,
                "validation_selection": selection,
                "validation_final_gate": final_gate,
                "synthetic": synthetic_metrics,
                "registered": registered_metrics,
            }
            history.append(record)
            (run_folder / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if validation_selection_improved(
                selection_key,
                best_selection_key,
                config["early_stopping_min_delta"],
            ):
                best_score = score
                best_selection_key = selection_key
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
        "best_selection": checkpoint["record"]["validation_selection"],
        "validation_final_gate": checkpoint["record"]["validation_final_gate"],
        "selection_split": "validation",
        "best_checkpoint": str(best_checkpoint),
        "steps": step,
        "unique_synthetic_views": synthetic_start,
        "stopped_early": synthetic_start < len(train_manifest["ap_um"]),
    }


def representative_onnx_batch() -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, IMAGE_SIZE)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    ellipse = ((x / 0.82).square() + (y / 0.62).square() <= 1.0).float()
    anatomy = (0.45 + 0.22 * torch.cos(10.0 * x) + 0.18 * torch.sin(8.0 * y)) * ellipse
    gradient = ((x + 1.0) * 0.5) * ellipse
    checker = ((torch.floor((x + 1.0) * 12) + torch.floor((y + 1.0) * 12)) % 2) * ellipse
    images = torch.stack((anatomy, 1.0 - anatomy, gradient, 0.7 * anatomy + 0.3 * checker))
    return images[:, None].repeat(1, 3, 1, 1).clamp(0.0, 1.0).float()


def _onnx_verification_session(model_path: Path, provider: str):
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = (
        [(provider, {"device_id": 0}), "CPUExecutionProvider"]
        if provider in {"CUDAExecutionProvider", "DmlExecutionProvider"}
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"ONNX verification did not activate {provider}")
    return session


def export_onnx(
    model: torch.nn.Module,
    output_folder: Path,
    metadata: dict,
    example: torch.Tensor | None = None,
) -> dict:
    output_folder.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    wrapper = AtlasPoseV7Export(model).eval()
    example = representative_onnx_batch() if example is None else example.cpu()
    if example.ndim != 4 or tuple(example.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("ONNX verification needs a [batch, 3, 299, 299] image batch")
    if float(example.std()) < 0.01:
        raise ValueError("ONNX verification batch must contain representative image variation")
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
    available = set(ort.get_available_providers())
    providers = ["CPUExecutionProvider"] + [
        provider
        for provider in ("CUDAExecutionProvider", "DmlExecutionProvider")
        if provider in available
    ]
    verification = {}
    for provider in providers:
        session = _onnx_verification_session(temporary, provider)
        if session.get_inputs()[0].name != "images" or session.get_inputs()[0].shape != ["batch", 3, IMAGE_SIZE, IMAGE_SIZE]:
            raise RuntimeError("Exported input contract differs from the tracker runtime")
        if [output.name for output in session.get_outputs()] != ["pose_ap_um_lr_deg_dv_deg", "orientation_inverted_logit"]:
            raise RuntimeError("Exported output contract differs from the tracker runtime")
        actual = session.run(None, {"images": example.numpy()})
        maximum = [float(np.max(np.abs(left - right))) for left, right in zip(reference, actual)]
        mean = [float(np.mean(np.abs(left - right))) for left, right in zip(reference, actual)]
        if maximum[0] > 0.05 or maximum[1] > 1e-4:
            raise RuntimeError(f"ONNX verification failed on {provider}: {maximum}")
        verification[provider] = {
            "mean_absolute_difference": mean,
            "max_absolute_difference": maximum,
        }
    differences = verification["CPUExecutionProvider"]["max_absolute_difference"]
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
        "preprocessing_contract_sha256": atlas_pose_preprocessing_contract_sha256(),
        "architecture": metadata.get("config", {}).get("architecture"),
        "pose_representation": metadata.get("config", {}).get("head"),
        "verification_max_absolute_difference": differences,
        "verification_by_provider": verification,
        "verification_sample_count": len(example),
        "verification_input_sha256": hashlib.sha256(
            np.ascontiguousarray(example.numpy()).tobytes()
        ).hexdigest(),
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
        "registered_data_sha256": metadata.get("registered_data", {}).get("sha256"),
        "atlas_data_sha256": metadata.get("atlas_data_sha256"),
        "training_environment": metadata.get("training_environment"),
        "pretrained_backbone": metadata.get("pretrained_backbone"),
        "selection_split": metadata.get("selection_split"),
        "training_splits": ["synthetic_train", "registered_train"],
        "selection_data": "synthetic_validation diagnostics and registered validation specimens",
        "excluded_from_selection": ["registered_test", "sealed_deepslice_s2p"],
    }
    (output_folder / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return metadata


def _canonical_json_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def promote_export(
    export_folder: Path,
    release_report: Path,
    destination: Path = ROOT / "models" / "AtlasPose",
) -> dict[str, str]:
    export_folder = Path(export_folder)
    release_report = Path(release_report)
    evidence_sha256 = file_sha256(release_report)
    report = json.loads(release_report.read_text(encoding="utf-8"))
    integrity = report.pop("release_integrity_sha256", None)
    if integrity != _canonical_json_sha256(report):
        raise RuntimeError("Release report integrity check failed")
    component_passed = report.get("deepslice_component_passed", {})
    quality = report.get("quality_gate", {})
    if (
        report.get("release_report_version") != 2
        or report.get("benchmark_role") != "final_release_gate"
        or not report.get("release_approved")
        or not report.get("promotion_ready")
        or not report.get("sealed")
        or not quality.get("all_gates_passed")
        or not quality.get("passed")
        or not all(quality["passed"].values())
        or set(component_passed) != {"ap_um", "lr_deg", "dv_deg"}
        or not all(component_passed.values())
    ):
        raise RuntimeError("AtlasPose promotion refused by the sealed release report")
    model = export_folder / "atlas_pose.onnx"
    metadata_path = export_folder / "atlas_pose.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if report.get("model_sha256") != file_sha256(model):
        raise RuntimeError("Release report does not describe this AtlasPose model")
    if (
        report.get("metadata_sha256") != file_sha256(metadata_path)
        or report.get("preprocessing_contract_sha256")
        != metadata.get("preprocessing_contract_sha256")
        or report.get("training_source_sha256") != metadata.get("source_sha256")
        or report.get("training_data_sha256")
        != {
            "synthetic_manifests": metadata.get("manifest_sha256"),
            "registered_data": metadata.get("registered_data", {}).get("sha256"),
            "atlas_data": metadata.get("atlas_data_sha256"),
        }
    ):
        raise RuntimeError("Release report does not bind the AtlasPose metadata and training data")
    sealed_metrics = release_report.with_name("SEALED_metrics.json")
    if report.get("sealed_metrics_sha256") != file_sha256(sealed_metrics):
        raise RuntimeError("Release report does not match its sealed metrics")
    sealed = json.loads(sealed_metrics.read_text(encoding="utf-8"))
    if (
        report.get("evaluator_sha256") != sealed.get("evaluator_sha256")
        or report.get("sealed_data_sha256")
        != {
            key: sealed.get("source", {}).get(key)
            for key in (
                "sections_sha256",
                "datasets_sha256",
                "provenance_sha256",
                "downloads_sha256",
                "registered_image_quality_manifest_sha256",
            )
        }
    ):
        raise RuntimeError("Release report does not bind the sealed evaluator and data")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("atlas_pose.onnx", "atlas_pose.json", "provenance.json"):
        shutil.copy2(export_folder / name, destination / name)
    shutil.copy2(release_report, destination / "RELEASE_REPORT.json")
    shutil.copy2(sealed_metrics, destination / "SEALED_metrics.json")
    return {
        "APPROVED_ATLAS_POSE_MODEL_SHA256": file_sha256(model),
        "APPROVED_ATLAS_POSE_METADATA_SHA256": file_sha256(metadata_path),
        "APPROVED_ATLAS_POSE_EVIDENCE_SHA256": evidence_sha256,
    }


def experiment_config(
    name: str,
    samples: int,
    overrides: dict | None = None,
    training_seed: int = SEEDS["train"],
) -> dict:
    config = {**DEFAULTS, **(overrides or {})}
    config.update({"name": name, "samples": int(samples), "training_seed": int(training_seed)})
    return config


def run_experiment(config: dict, export: bool = False) -> dict:
    run_folder = WORKSPACE / "runs" / config["name"]
    registered_hashes = registered_data_hashes(REGISTERED_ROOT)
    atlas_hashes = atlas_data_hashes(ATLAS_FOLDER)
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
        seed=config["training_seed"],
        anatomy_enabled=config["anatomy"] > 0.0,
    )
    torch.manual_seed(config["training_seed"])
    model = AtlasPoseV7(config["architecture"], pretrained=True, pose_representation=config["head"])
    pretrained_provenance = pretrained_backbone_provenance(model, config["architecture"])
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
            "sha256": registered_hashes,
            "training_split": "train",
            "selection_split": "validation",
            "excluded_from_selection": ["test", "sealed_deepslice_s2p"],
        },
        "atlas_data_sha256": atlas_hashes,
        "training_environment": training_environment(),
        "pretrained_backbone": pretrained_provenance,
    })
    if registered_data_hashes(REGISTERED_ROOT) != registered_hashes:
        raise RuntimeError("Registered dataset changed during training")
    if atlas_data_hashes(ATLAS_FOLDER) != atlas_hashes:
        raise RuntimeError("Atlas data changed during training")
    result_path = run_folder / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if export:
        if not result["validation_final_gate"]["all_gates_passed"]:
            raise RuntimeError("Model export refused because the final validation gates failed")
        reports = held_out_reports(result)
        result["held_out_reports"] = reports
        if not reports["test"]["final_gate"]["all_gates_passed"]:
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            raise RuntimeError("Model export refused because the registered test gates failed")
        export_onnx(model, run_folder / "export", result)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def seed_animal_component_errors(
    results: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not results or any(result["selection_split"] != "validation" for result in results):
        raise RuntimeError("Architecture selection accepts only registered validation results")
    result_by_seed = {int(result["config"]["training_seed"]): result for result in results}
    seeds = tuple(sorted(result_by_seed))
    if len(result_by_seed) != len(results) or set(seeds) != set(COMPARISON_SEEDS):
        raise RuntimeError("Architecture selection requires the three prespecified training seeds")
    run_errors = []
    expected_sections = None
    expected_animals = None
    for seed in seeds:
        result = result_by_seed[seed]
        path = Path(result["best_checkpoint"]).parent / "validation_registered.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        sections = {
            int(row["section_image_id"]): (
                int(row["specimen_id"]),
                *(float(row[f"target_{axis}"]) for axis in ("ap", "lr", "dv")),
                str(row["product"]),
                str(row.get("in_training_ap_domain", "True")).casefold(),
                row["split"],
            )
            for row in rows
        }
        if expected_sections is None:
            expected_sections = sections
        elif sections != expected_sections:
            raise RuntimeError("Compared seeds did not evaluate identical validation sections and targets")
        animals, errors = balanced_animal_component_errors(rows)
        if expected_animals is None:
            expected_animals = animals
        elif not np.array_equal(animals, expected_animals):
            raise RuntimeError("Compared seeds did not evaluate identical validation animals")
        run_errors.append(errors)
    return np.asarray(seeds), expected_animals, np.asarray(run_errors)


def seed_group_selection_summary(results: list[dict]) -> dict:
    _, _, errors = seed_animal_component_errors(results)
    return _selection_summary_from_animal_errors(errors.mean(0))


def bootstrap_seed_group_comparison(
    candidate: list[dict],
    reference: list[dict],
    output: Path,
    iterations: int = 10_000,
    seed: int = 41171,
) -> dict:
    candidate_seeds, candidate_animals, candidate_errors = seed_animal_component_errors(candidate)
    reference_seeds, reference_animals, reference_errors = seed_animal_component_errors(reference)
    if not np.array_equal(candidate_seeds, reference_seeds):
        raise RuntimeError("Compared model families did not use identical training seeds")
    if not np.array_equal(candidate_animals, reference_animals):
        raise RuntimeError("Compared model families did not evaluate identical validation animals")
    differences = candidate_errors - reference_errors
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(0, len(candidate_seeds), (iterations, len(candidate_seeds)))
    animal_draws = rng.integers(0, len(candidate_animals), (iterations, len(candidate_animals)))
    sampled = differences[seed_draws[:, :, None], animal_draws[:, None, :]].mean(axis=(1, 2))
    sampled_composite = (sampled / COMPONENT_SCALES).mean(1)
    point_components = differences.mean(axis=(0, 1))
    point_composite = float((point_components / COMPONENT_SCALES).mean())
    components = {}
    for index, name in enumerate(("ap_um", "lr_deg", "dv_deg")):
        components[name] = {
            "candidate_minus_reference": float(point_components[index]),
            "ci95": np.percentile(sampled[:, index], (2.5, 97.5)).tolist(),
            "probability_candidate_better": float(np.mean(sampled[:, index] < 0.0)),
        }
    result = {
        "unit": "paired training_seed x specimen_id",
        "seed_count": len(candidate_seeds),
        "animal_count": len(candidate_animals),
        "candidate_minus_reference": point_composite,
        "ci95": np.percentile(sampled_composite, (2.5, 97.5)).tolist(),
        "probability_candidate_better": float(np.mean(sampled_composite < 0.0)),
        "components": components,
        "iterations": int(iterations),
        "bootstrap_seed": int(seed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_comparison_group(name: str, samples: int, overrides: dict) -> list[dict]:
    return [
        run_experiment(
            experiment_config(
                f"{name}_seed{seed}",
                samples,
                overrides,
                training_seed=seed,
            )
        )
        for seed in COMPARISON_SEEDS
    ]


def select_model_family(
    groups: dict[str, list[dict]],
    label: str,
    tie_priority: tuple[str, ...],
    output_prefix: Path,
) -> dict:
    if len(groups) < 2 or set(groups) - set(tie_priority):
        raise RuntimeError(f"Missing prespecified {label} tie priority")
    summaries = {
        candidate: seed_group_selection_summary(results)
        for candidate, results in groups.items()
    }
    point_order = sorted(
        groups,
        key=lambda candidate: (validation_selection_key(summaries[candidate]), tie_priority.index(candidate)),
    )
    point_best = point_order[0]
    comparisons = {}
    tied = [point_best]
    pairwise_confidence = 1.0 - (1.0 - FAMILY_CONFIDENCE) / (len(groups) - 1)
    for opponent in point_order[1:]:
        comparison = bootstrap_seed_group_comparison(
            groups[point_best],
            groups[opponent],
            output_prefix.with_name(f"{output_prefix.name}_{point_best}_vs_{opponent}.json"),
        )
        comparisons[opponent] = comparison
        if comparison["probability_candidate_better"] < pairwise_confidence:
            tied.append(opponent)
    winner = min(tied, key=tie_priority.index)
    runner_up = next(candidate for candidate in point_order if candidate != winner)
    return {
        "label": label,
        "winner": winner,
        "runner_up": runner_up,
        "point_estimate_best": point_best,
        "summaries": summaries,
        "comparisons_from_point_estimate_best": comparisons,
        "confidence_threshold": FAMILY_CONFIDENCE,
        "bonferroni_pairwise_confidence": pairwise_confidence,
        "tie_priority": list(tie_priority),
        "statistically_tied": tied,
        "decision": (
            "hierarchical_bootstrap"
            if tied == [point_best]
            else "prespecified_tie_priority"
        ),
    }


def held_out_reports(result: dict) -> dict:
    config = result["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AtlasPoseV7(config["architecture"], pretrained=False, pose_representation=config["head"]).to(device)
    checkpoint = torch.load(result["best_checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    run_folder = Path(result["best_checkpoint"]).parent
    reports = {}
    split = "test"
    dataset = RegisteredSectionDataset(
        REGISTERED_ROOT,
        ATLAS_FOLDER,
        split=split,
        include_anatomy=False,
        cache_images=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["evaluation_batch_size"],
        shuffle=False,
        num_workers=config["data_workers"],
    )
    _, rows = evaluate_registered(model, loader, device, split)
    reports[split] = {
        **registered_domain_reports(rows),
        "final_gate": final_acceptance_summary(rows, split),
    }
    _write_csv(run_folder / f"{split}_registered.csv", rows)
    write_diagnostic_plot(
        reports[split]["primary_in_training_ap_domain"],
        run_folder / f"{split}_registered.png",
    )
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
    head_screen = {
        head: run_comparison_group(
            f"20k_head_{head}",
            SCHEDULE["head_screen_20k"],
            {"head": head},
        )
        for head in ("direct", "binned", "ouv")
    }
    head_summaries = {
        head: seed_group_selection_summary(results) for head, results in head_screen.items()
    }
    surviving_heads = sorted(
        head_summaries,
        key=lambda head: (
            validation_selection_key(head_summaries[head]),
            HEAD_TIE_PRIORITY.index(head),
        ),
    )[:2]
    head_results = {
        head: run_comparison_group(
            f"100k_head_{head}",
            SCHEDULE["surviving_heads_100k"],
            {"head": head},
        )
        for head in surviving_heads
    }
    head_decision = select_model_family(
        head_results,
        "pose head",
        HEAD_TIE_PRIORITY,
        WORKSPACE / "head_hierarchical_bootstrap",
    )
    best_head = head_decision["winner"]
    backbone_results = {"convnext_tiny": head_results[best_head]}
    backbone_results.update({
        architecture: run_comparison_group(
            f"100k_backbone_{architecture}_{best_head}",
            SCHEDULE["backbones_100k"],
            {"architecture": architecture, "head": best_head},
        )
        for architecture in BACKBONES
        if architecture != "convnext_tiny"
    })
    backbone_decision = select_model_family(
        backbone_results,
        "backbone",
        BACKBONE_TIE_PRIORITY,
        WORKSPACE / "backbone_hierarchical_bootstrap",
    )
    best_backbone = backbone_decision["winner"]
    ablation_control_config = {"architecture": best_backbone, "head": best_head}
    ablation_control = run_comparison_group(
        "20k_ablation_control",
        SCHEDULE["ablation_20k"],
        ablation_control_config,
    )
    ablations = {
        name: run_comparison_group(
            f"20k_ablation_{name}",
            SCHEDULE["ablation_20k"],
            {**ablation_control_config, **values},
        )
        for name, values in ABLATIONS_20K.items()
    }
    ablation_comparisons = {
        name: bootstrap_seed_group_comparison(
            results,
            ablation_control,
            WORKSPACE / f"ablation_{name}_vs_control.json",
        )
        for name, results in ablations.items()
    }
    final = run_experiment(
        experiment_config(
            f"final_{best_backbone}_{best_head}",
            SCHEDULE["final_unique_views"],
            {"architecture": best_backbone, "head": best_head, "validation_interval": 50_000},
            training_seed=COMPARISON_SEEDS[0],
        ),
        export=True,
    )
    final_reports = final["held_out_reports"]
    summary = {
        "comparison_seeds": COMPARISON_SEEDS,
        "head_screen": head_summaries,
        "surviving_heads": surviving_heads,
        "head_final": head_decision["summaries"],
        "best_head": best_head,
        "head_decision": head_decision,
        "backbones": backbone_decision["summaries"],
        "best_backbone": best_backbone,
        "backbone_decision": backbone_decision,
        "ablation_control": seed_group_selection_summary(ablation_control),
        "ablations": {
            name: seed_group_selection_summary(results) for name, results in ablations.items()
        },
        "ablation_hierarchical_bootstrap": ablation_comparisons,
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
            "ATLAS_POSE_V7_TRAINING_SEED": ("training_seed", int),
        }
        for variable, (key, conversion) in environment_overrides.items():
            if variable in os.environ:
                config[key] = conversion(os.environ[variable])
        run_experiment(config, export=os.environ.get("ATLAS_POSE_V7_EXPORT") == "1")
    else:
        run_schedule()


if __name__ == "__main__":
    main()
