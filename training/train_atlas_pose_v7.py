from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from training.atlas_pose_models_v7 import (
    BACKBONES,
    PHYSICAL_POSE_LOSS_SCALE,
    AtlasPoseV7,
    AtlasPoseV7Export,
    atlas_pose_v7_loss,
)
from training.atlas_pose_release_contract import (
    POSE_AXES,
    RELEASE_CONFIDENCE,
    RELEASE_GATE_THRESHOLDS,
    RELEASE_REFERENCE,
    evaluation_domains,
    paired_animal_bootstrap,
    paired_animal_joint_superiority,
    release_quality_gate,
    release_statistics_equal,
    validate_complete_method_cohort,
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
    ATLAS_POSE_SEALED_SOURCE_FILES,
    ATLAS_POSE_SEALED_SPLIT,
    AUTOMATIC_BRAIN_MASK_VERSION,
    QUICKNII_COORDINATE_CONTRACT_VERSION,
    atlas_pose_preprocessing_contract_sha256,
    verify_atlas_pose_release_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("ATLAS_POSE_V7_WORKSPACE", "J:/AtlasPoseTraining_v7"))
ATLAS_FOLDER = Path(os.environ.get("ATLAS_POSE_ATLAS", ROOT / "data" / "Allen Brain Atlas 25um"))
REGISTERED_ROOT = Path(
    os.environ.get(
        "ATLAS_POSE_REGISTERED",
        WORKSPACE / "allen_registered_full_quicknii_ras_v2_20260811",
    )
)
SEEDS = {"train": 73191, "validation": 19841, "test": 49157, "paired": 90821}
COMPARISON_SEEDS = (73191, 41777, 90217)
FINAL_GATE_THRESHOLDS = RELEASE_GATE_THRESHOLDS
COVERAGE_REQUIREMENTS = {
    "animals": 20,
    "animals_per_required_product": 10,
    "animals_per_ap_band": 20,
    "animals_per_lr_bin": 10,
    "animals_per_dv_bin": 10,
}
REQUIRED_PRODUCTS = ("5", "8")
TRUSTED_REGISTERED_PRODUCTS = ("5",)
DIAGNOSTIC_ONLY_REGISTERED_PRODUCTS = ("8",)
REGISTERED_LABEL_POLICY = {
    "supervised_and_selected_products": list(TRUSTED_REGISTERED_PRODUCTS),
    "diagnostic_only_products": list(DIAGNOSTIC_ONLY_REGISTERED_PRODUCTS),
    "product_5_role": "Allen ConnProj serial two-photon block-face registration",
    "product_8_role": "Allen ConnTG slide-mounted affine; diagnostic only",
}
SYNTHETIC_GATE_THRESHOLDS = {
    "mean_ap_um": 60.0,
    "mean_lr_deg": 0.90,
    "mean_dv_deg": 1.75,
    "worst_artifact_mae_ap_um": 90.0,
    "worst_artifact_mae_lr_deg": 1.50,
    "worst_artifact_mae_dv_deg": 2.50,
    "worst_tilt_mae_ap_um": 90.0,
    "worst_tilt_mae_lr_deg": 1.50,
    "worst_tilt_mae_dv_deg": 2.50,
    "mean_invariance_shift_ap_um": 60.0,
    "mean_invariance_shift_lr_deg": 0.90,
    "mean_invariance_shift_dv_deg": 1.75,
}
COMPONENT_SCALES = np.asarray(PHYSICAL_POSE_LOSS_SCALE, dtype=np.float64)
VALIDATION_COMPONENT_GATES = np.asarray(
    tuple(FINAL_GATE_THRESHOLDS[name] for name in ("mean_ap_um", "mean_lr_deg", "mean_dv_deg")),
    dtype=np.float64,
)
FAMILY_CONFIDENCE = 0.95
_SYNTHETIC_VALIDATION_CACHE: dict[
    tuple[str, str, str | None, int],
    tuple[torch.Tensor, torch.Tensor | None, torch.Tensor],
] = {}

DEFAULTS = {
    "architecture": "convnext_tiny",
    "renderer": "v7",
    "head": "binned",
    "consistency": 0.15,
    "anatomy": 0.20,
    "registered_fraction": 0.50,
    "batch_size": 12,
    "evaluation_batch_size": 24,
    "data_workers": 8,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "warmup_fraction": 0.05,
    "ema_decay": 0.999,
    "gradient_clip": 1.0,
    "validation_interval": 1_000,
    "validation_count": 1_024,
    "early_stopping_patience": 6,
    "early_stopping_min_delta": 0.002,
}


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def training_source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(__file__).with_name("atlas_pose_models_v7.py"),
        Path(__file__).with_name("synthetic_atlas.py"),
        Path(__file__).with_name("registered_section_dataset.py"),
        Path(__file__).with_name("atlas_pose_release_contract.py"),
        ROOT / "source" / "atlas_pose_runtime.py",
        ROOT / "source" / "registered_image_quality.py",
    )
    return {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in paths}


def git_source_provenance() -> dict:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked = tuple(training_source_hashes())
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all", "--", *tracked),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "tracked_source_dirty": bool(status), "tracked_source_status": status}


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
    root = Path(root)
    names = ATLAS_POSE_SEALED_SOURCE_FILES
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Registered dataset is incomplete; missing {missing}")
    hashes = {name: file_sha256(root / name) for name in names}
    receipt_path = root / ".atlas_pose_cache" / "registered_data_hashes_v1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
    receipt = receipt if isinstance(receipt, dict) else {}
    cached_images = receipt.get("images", [])
    cached_images = cached_images if isinstance(cached_images, list) else []
    cache_matches_manifests = (
        receipt.get("version") == 1
        and receipt.get("source_manifest_sha256") == hashes
    )
    sections = [
        json.loads(line)
        for line in (root / "sections.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    download_rows = [
        json.loads(line)
        for line in (root / "downloads.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    downloads = {int(row["section_image_id"]): row["sha256"] for row in download_rows}
    if len(downloads) != len(download_rows):
        raise RuntimeError("Registered download manifest contains duplicate section IDs")
    sections = [record for record in sections if record["split"] != ATLAS_POSE_SEALED_SPLIT]
    cache_matches_manifests &= len(cached_images) == len(sections)
    image_hashes = []
    verified_images = []
    for index, record in enumerate(sections):
        section_id = int(record["section_image_id"])
        relative_path = str(record["relative_path"]).replace("\\", "/")
        path = root / relative_path
        expected = downloads.get(section_id)
        if expected is None or not path.is_file():
            raise RuntimeError(f"Registered image checksum failed for section {section_id}")
        stat = path.stat()
        identity = {
            "section_image_id": section_id,
            "relative_path": relative_path,
            "sha256": expected,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        cached = cached_images[index] if cache_matches_manifests else None
        if cached != identity:
            actual = file_sha256(path)
            verified_stat = path.stat()
            if (verified_stat.st_size, verified_stat.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError(f"Registered image changed during checksum for section {section_id}")
            if actual != expected:
                raise RuntimeError(f"Registered image checksum failed for section {section_id}")
        verified_images.append(identity)
        image_hashes.append((section_id, expected))
    if not image_hashes:
        raise RuntimeError("Registered dataset contains no non-sealed images")
    hashes["nonsealed_image_tree_sha256"] = hashlib.sha256(
        json.dumps(sorted(image_hashes), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not cache_matches_manifests or verified_images != cached_images:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=receipt_path.parent,
            prefix=f".{receipt_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(
                {
                    "version": 1,
                    "source_manifest_sha256": {name: hashes[name] for name in names},
                    "images": verified_images,
                },
                stream,
                separators=(",", ":"),
                sort_keys=True,
            )
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, receipt_path)
    return hashes


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


def _registered_product(datasets: dict[int, dict], experiment_id: int) -> str:
    return "+".join(map(str, datasets[experiment_id].get("product_ids", []))) or "unknown"


def registered_rows_for_products(rows: list[dict], products: tuple[str, ...]) -> list[dict]:
    products = set(map(str, products))
    selected = [
        row
        for row in rows
        if set(str(row["product"]).split("+")).issubset(products)
    ]
    if not selected:
        raise RuntimeError(f"Registered evaluation contains no rows for products {sorted(products)}")
    return selected


def registered_sampling_weights(dataset: RegisteredSectionDataset) -> torch.Tensor:
    groups = []
    section_counts = {}
    product_specimens = {}
    for record in dataset.records:
        product = _registered_product(dataset.datasets, int(record["experiment_id"]))
        group = product, int(record["specimen_id"])
        groups.append(group)
        section_counts[group] = section_counts.get(group, 0) + 1
        product_specimens.setdefault(product, set()).add(group[1])
    return torch.as_tensor(
        [
            1.0 / (len(product_specimens[product]) * section_counts[(product, specimen)])
            for product, specimen in groups
        ],
        dtype=torch.float64,
    )


def build_registered_loaders(
    manifest_root: Path,
    atlas_folder: Path,
    batch_size: int,
    validation_batch_size: int,
    paired: bool,
    workers: int,
    seed: int,
    anatomy_enabled: bool,
    supervised_product_ids: tuple[int, ...],
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
        allowed_product_ids=supervised_product_ids,
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
    if any(
        not set(train.datasets[int(record["experiment_id"])]["product_ids"]).issubset(
            supervised_product_ids
        )
        for record in train.records
    ):
        raise RuntimeError("Registered training loader contains a diagnostic-only product")
    if any(record["split"] != "validation" for record in validation.records):
        raise RuntimeError("Registered validation loader contains a non-validation specimen")
    sampler = WeightedRandomSampler(
        registered_sampling_weights(train),
        num_samples=len(train),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    options = {"num_workers": workers, "pin_memory": torch.cuda.is_available(), "persistent_workers": workers > 0}
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed + 1),
            **options,
        ),
        DataLoader(
            validation,
            batch_size=validation_batch_size,
            shuffle=False,
            generator=torch.Generator().manual_seed(seed + 2),
            **options,
        ),
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


def rotation_180_counterfactual_diagnostics(
    prediction: np.ndarray,
    rotated_prediction: np.ndarray,
    orientation_logit: np.ndarray,
    rotated_orientation_logit: np.ndarray,
) -> dict:
    shift = np.abs(rotated_prediction - prediction)
    return {
        "role": "diagnostic_only_not_used_for_selection",
        "mean_absolute_prediction_shift": shift.mean(0).tolist(),
        "p95_absolute_prediction_shift": np.percentile(shift, 95, axis=0).tolist(),
        "orientation_logit_sign_flip_fraction": float(
            np.mean(np.signbit(orientation_logit) != np.signbit(rotated_orientation_logit))
        ),
        "mean_absolute_orientation_logit_sum": float(
            np.mean(np.abs(orientation_logit + rotated_orientation_logit))
        ),
    }


def synthetic_acceptance_summary(report: dict) -> dict:
    artifact = report.get("artifact_severity", {})
    tilt = report.get("tilt_bands", {})
    invariance = report.get("artifact_invariance")
    expected_artifact = {str(name) for name in COHORT_NAMES}
    expected_tilt = {"0:5", "5:15", "15:25", "25:35"}
    coverage = {
        "artifact_cohorts": set(artifact) == expected_artifact,
        "tilt_bands": set(tilt) == expected_tilt,
        "paired_artifact_invariance": invariance is not None,
    }
    overall = np.asarray(report["overall"]["mae"], dtype=np.float64)
    artifact_worst = (
        np.max(np.asarray([group["mae"] for group in artifact.values()]), axis=0)
        if artifact
        else np.full(3, np.inf)
    )
    tilt_worst = (
        np.max(np.asarray([group["mae"] for group in tilt.values()]), axis=0)
        if tilt
        else np.full(3, np.inf)
    )
    invariant = (
        np.asarray(invariance["mean_absolute_prediction_shift"], dtype=np.float64)
        if invariance is not None
        else np.full(3, np.inf)
    )
    values = dict(
        zip(
            SYNTHETIC_GATE_THRESHOLDS,
            np.concatenate((overall, artifact_worst, tilt_worst, invariant)).tolist(),
        )
    )
    passed = {
        name: bool(values[name] <= threshold)
        for name, threshold in SYNTHETIC_GATE_THRESHOLDS.items()
    }
    return {
        "coverage": {"passed": coverage, "eligible": all(coverage.values())},
        "values": values,
        "thresholds": dict(SYNTHETIC_GATE_THRESHOLDS),
        "passed": passed,
        "all_performance_gates_passed": all(passed.values()),
        "all_gates_passed": all(coverage.values()) and all(passed.values()),
        "worst_gate_ratio": max(
            values[name] / threshold for name, threshold in SYNTHETIC_GATE_THRESHOLDS.items()
        ),
    }


def specimen_median_tilt_diagnostics(rows: list[dict]) -> dict:
    errors = []
    target_ranges = []
    for specimen in sorted({int(row["specimen_id"]) for row in rows}):
        selected = [row for row in rows if int(row["specimen_id"]) == specimen]
        target = np.asarray([[row["target_lr"], row["target_dv"]] for row in selected])
        prediction = np.asarray([[row["prediction_lr"], row["prediction_dv"]] for row in selected])
        errors.append(np.abs(np.median(prediction, axis=0) - np.median(target, axis=0)))
        target_ranges.append(np.ptp(target, axis=0))
    errors = np.asarray(errors)
    return {
        "role": "diagnostic_only_not_used_for_selection",
        "unit": "specimen_id",
        "specimen_count": len(errors),
        "mae_deg": dict(zip(("lr", "dv"), errors.mean(0).tolist())),
        "p90_deg": dict(zip(("lr", "dv"), np.percentile(errors, 90, axis=0).tolist())),
        "maximum_within_specimen_target_range_deg": dict(
            zip(("lr", "dv"), np.max(np.asarray(target_ranges), axis=0).tolist())
        ),
    }


def registered_report(rows: list[dict]) -> dict:
    target = np.asarray([[row[f"target_{axis}"] for axis in ("ap", "lr", "dv")] for row in rows])
    prediction = np.asarray([[row[f"prediction_{axis}"] for axis in ("ap", "lr", "dv")] for row in rows])
    report = {
        "overall": component_metrics(target, prediction),
        "per_specimen": {},
        "per_product": {},
        "ap_500um_bands": {},
        "nonselection_diagnostics": {
            "specimen_median_tilt": specimen_median_tilt_diagnostics(rows),
        },
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


def _lr_bin(value: float) -> str:
    if value < -1.5:
        return "lt_-1.5"
    return "gt_1.5" if value > 1.5 else "-1.5_to_1.5"


def _dv_bin(value: float) -> str:
    if value < -7.0:
        return "lt_-7"
    return "gt_-2" if value > -2.0 else "-7_to_-2"


def evaluated_rows_sha256(rows: list[dict]) -> str:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("split", "synthetic")),
            int(row.get("specimen_id", 0)),
            int(row.get("experiment_id", 0)),
            int(row.get("section_image_id", row.get("latent_index", 0))),
        ),
    )
    return hashlib.sha256(
        json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def registered_coverage_summary(
    rows: list[dict],
    required_products: tuple[str, ...] = REQUIRED_PRODUCTS,
) -> dict:
    specimens = {int(row["specimen_id"]) for row in rows}
    product_animals = {
        product: {
            int(row["specimen_id"])
            for row in rows
            if product in str(row["product"]).split("+")
        }
        for product in required_products
    }
    ap_band_animals = {
        band: {
            int(row["specimen_id"])
            for row in rows
            if _ap_500um_band(float(row["target_ap"])) == band
        }
        for band in (f"{low}:{low + 500}" for low in range(-4500, 500, 500))
    }
    lr_bin_animals = {
        label: {
            int(row["specimen_id"])
            for row in rows
            if _lr_bin(float(row["target_lr"])) == label
        }
        for label in ("lt_-1.5", "-1.5_to_1.5", "gt_1.5")
    }
    dv_bin_animals = {
        label: {
            int(row["specimen_id"])
            for row in rows
            if _dv_bin(float(row["target_dv"])) == label
        }
        for label in ("lt_-7", "-7_to_-2", "gt_-2")
    }
    counts = {
        "animals": len(specimens),
        "animals_by_required_product": {
            product: len(values) for product, values in product_animals.items()
        },
        "animals_by_ap_band": {label: len(values) for label, values in ap_band_animals.items()},
        "animals_by_lr_bin": {label: len(values) for label, values in lr_bin_animals.items()},
        "animals_by_dv_bin": {label: len(values) for label, values in dv_bin_animals.items()},
    }
    passed = {
        "animals": counts["animals"] >= COVERAGE_REQUIREMENTS["animals"],
        "required_products": min(counts["animals_by_required_product"].values())
        >= COVERAGE_REQUIREMENTS["animals_per_required_product"],
        "ap_bands": min(counts["animals_by_ap_band"].values())
        >= COVERAGE_REQUIREMENTS["animals_per_ap_band"],
        "lr_bins": min(counts["animals_by_lr_bin"].values())
        >= COVERAGE_REQUIREMENTS["animals_per_lr_bin"],
        "dv_bins": min(counts["animals_by_dv_bin"].values())
        >= COVERAGE_REQUIREMENTS["animals_per_dv_bin"],
    }
    return {
        "requirements": dict(COVERAGE_REQUIREMENTS),
        "required_products": list(required_products),
        "counts": counts,
        "passed": passed,
        "eligible": all(passed.values()),
    }


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


def _selection_summary_from_component_mae(component_mae: np.ndarray) -> dict:
    component_mae = np.asarray(component_mae, dtype=np.float64)
    passed = component_mae <= VALIDATION_COMPONENT_GATES
    return {
        "component_mae": dict(zip(("ap_um", "lr_deg", "dv_deg"), component_mae.tolist())),
        "component_gates": dict(
            zip(("ap_um", "lr_deg", "dv_deg"), VALIDATION_COMPONENT_GATES.tolist())
        ),
        "component_passed": dict(zip(("ap_um", "lr_deg", "dv_deg"), passed.tolist())),
        "all_mean_gates_passed": bool(passed.all()),
        "worst_gate_ratio": float(np.max(component_mae / VALIDATION_COMPONENT_GATES)),
        "composite_score": float((component_mae / COMPONENT_SCALES).mean()),
    }


def validation_selection_summary(rows: list[dict]) -> dict:
    _, errors = balanced_animal_component_errors(rows)
    return _selection_summary_from_component_mae(errors.mean(0))


def registered_rows_for_release_gate(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["atlas_pose"] * len(rows),
            "specimen_id": [int(row["specimen_id"]) for row in rows],
            "section_image_id": [int(row["section_image_id"]) for row in rows],
            "product": [str(row["product"]) for row in rows],
            "in_training_ap_domain": [_row_in_training_ap_domain(row) for row in rows],
            "gt_ap_um": [float(row["target_ap"]) for row in rows],
            "gt_lr_deg": [float(row["target_lr"]) for row in rows],
            "gt_dv_deg": [float(row["target_dv"]) for row in rows],
            "pred_ap_um": [float(row["prediction_ap"]) for row in rows],
            "pred_lr_deg": [float(row["prediction_lr"]) for row in rows],
            "pred_dv_deg": [float(row["prediction_dv"]) for row in rows],
        }
    )


def final_acceptance_summary(
    rows: list[dict],
    required_split: str,
    required_products: tuple[str, ...] = REQUIRED_PRODUCTS,
) -> dict:
    quality_rows = registered_rows_for_release_gate(rows)
    in_domain = [
        row
        for row, include in zip(rows, quality_rows["in_training_ap_domain"])
        if include
    ]
    if not in_domain or any(row["split"] != required_split for row in in_domain):
        raise RuntimeError(f"Expected non-empty registered {required_split}-specimen rows")
    quality = release_quality_gate(quality_rows)
    component_mae = np.asarray(
        [quality["values"][name] for name in ("mean_ap_um", "mean_lr_deg", "mean_dv_deg")]
    )
    selection = _selection_summary_from_component_mae(component_mae)
    coverage = registered_coverage_summary(in_domain, required_products)
    return {
        "split": required_split,
        "in_training_ap_domain_count": len(in_domain),
        "animal_count": quality["animal_count"],
        "evaluated_rows_sha256": evaluated_rows_sha256(in_domain),
        "coverage": coverage,
        "values": quality["values"],
        "thresholds": quality["thresholds"],
        "passed": quality["passed"],
        "all_performance_gates_passed": quality["all_gates_passed"],
        "all_gates_passed": coverage["eligible"] and quality["all_gates_passed"],
        "worst_gate_ratio": max(
            quality["values"][name] / threshold
            for name, threshold in quality["thresholds"].items()
        ),
        "worst_ap_band": quality["worst_ap_band"],
        "worst_product": quality["worst_product"],
        "group_component_p90": quality["group_component_p90"],
        "selection_summary": selection,
    }


def validation_selection_key(
    summary: dict,
    final_gate: dict,
) -> tuple[float, float, float]:
    return (
        0.0 if final_gate["all_performance_gates_passed"] else 1.0,
        final_gate["worst_gate_ratio"],
        summary["composite_score"],
    )


def checkpoint_validation_key(
    summary: dict,
    registered_gate: dict,
    synthetic_gate: dict,
) -> tuple[float, float, float]:
    return (
        max(registered_gate["worst_gate_ratio"], synthetic_gate["worst_gate_ratio"]),
        registered_gate["worst_gate_ratio"],
        summary["composite_score"],
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


def checkpoint_selection_improved(
    candidate_eligible: bool,
    reference_eligible: bool,
    candidate: tuple[float, float, float],
    reference: tuple[float, float, float],
    minimum_delta: float,
) -> bool:
    if candidate_eligible != reference_eligible:
        return candidate_eligible
    return validation_selection_improved(candidate, reference, minimum_delta)


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
def evaluate_registered(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    split: str,
    counterfactual_180: bool = False,
) -> tuple[dict, list[dict]]:
    if split not in {"validation", "test"}:
        raise ValueError("Registered evaluation split is not reportable")
    model.eval()
    rows = []
    counterfactual = []
    datasets = loader.dataset.datasets
    for batch in loader:
        image = batch["image"][:, 0] if batch["image"].ndim == 5 else batch["image"]
        image = image.to(device, non_blocking=True)
        if counterfactual_180:
            prediction_tensor, orientation_logit = model.forward_with_orientation(image)
            rotated_prediction, rotated_orientation_logit = model.forward_with_orientation(
                torch.rot90(image, 2, (-2, -1))
            )
            counterfactual.append(
                (
                    prediction_tensor.float().cpu().numpy(),
                    rotated_prediction.float().cpu().numpy(),
                    orientation_logit.float().cpu().numpy(),
                    rotated_orientation_logit.float().cpu().numpy(),
                )
            )
        else:
            prediction_tensor = model(image)
        prediction = prediction_tensor.float().cpu().numpy()
        target = batch["pose"].numpy()
        for index in range(len(target)):
            experiment = int(batch["experiment_id"][index])
            product = _registered_product(datasets, experiment)
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
    report = registered_report(rows)
    if counterfactual:
        values = [np.concatenate([batch[index] for batch in counterfactual]) for index in range(4)]
        report["nonselection_diagnostics"]["rotation_180_counterfactual"] = (
            rotation_180_counterfactual_diagnostics(*values)
        )
    return report, rows


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
    best_checkpoint_eligible = False
    stale = 0
    history = []
    best_checkpoint = run_folder / "best.pt"
    last_checkpoint = run_folder / "last.pt"
    interval_component_sums = {source: {} for source in ("registered", "synthetic")}
    interval_batch_counts = {source: 0 for source in interval_component_sums}

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
        source = "registered" if use_registered else "synthetic"
        interval_batch_counts[source] += 1
        for name, value in components.items():
            interval_component_sums[source][name] = (
                interval_component_sums[source].get(name, 0.0) + value
            )

        if synthetic_start >= next_validation or synthetic_start == len(train_manifest["ap_um"]):
            current = _swap_to_ema(model, ema)
            synthetic_metrics, _ = evaluate_synthetic(
                model, renderer, validation_manifest, validation_paired, config["evaluation_batch_size"]
            )
            registered_metrics, registered_rows = evaluate_registered(
                model, registered_validation, device, "validation", counterfactual_180=True
            )
            trusted_rows = registered_rows_for_products(
                registered_rows, TRUSTED_REGISTERED_PRODUCTS
            )
            trusted_metrics = registered_report(trusted_rows)
            selection = validation_selection_summary(trusted_rows)
            final_gate = final_acceptance_summary(
                trusted_rows,
                "validation",
                TRUSTED_REGISTERED_PRODUCTS,
            )
            raw_all_products_gate = final_acceptance_summary(
                registered_rows,
                "validation",
            )
            raw_all_products_gate["role"] = "diagnostic_only_not_used_for_selection"
            synthetic_gate = synthetic_acceptance_summary(synthetic_metrics)
            checkpoint_eligible = bool(
                final_gate["all_gates_passed"] and synthetic_gate["all_gates_passed"]
            )
            score = selection["composite_score"]
            selection_key = checkpoint_validation_key(selection, final_gate, synthetic_gate)
            model.load_state_dict(current)
            training_summary = {
                source: {
                    "batch_count": interval_batch_counts[source],
                    **{
                        name: total / max(interval_batch_counts[source], 1)
                        for name, total in interval_component_sums[source].items()
                    },
                }
                for source in interval_component_sums
            }
            record = {
                "step": step,
                "unique_synthetic_views": synthetic_start,
                "learning_rate": learning_rate,
                "training": training_summary,
                "validation_selection_score": score,
                "validation_selection": selection,
                "validation_final_gate": final_gate,
                "validation_raw_all_products_diagnostic_gate": raw_all_products_gate,
                "synthetic_validation_gate": synthetic_gate,
                "checkpoint_eligible": checkpoint_eligible,
                "synthetic": synthetic_metrics,
                "registered": registered_metrics,
                "registered_trusted": trusted_metrics,
            }
            history.append(record)
            interval_component_sums = {source: {} for source in interval_component_sums}
            interval_batch_counts = {source: 0 for source in interval_batch_counts}
            (run_folder / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            checkpoint_payload = {
                "model": {key: value.cpu() for key, value in ema.items()},
                "config": config,
                "record": record,
            }
            torch.save(checkpoint_payload, last_checkpoint)
            _write_csv(run_folder / "validation_registered_last.csv", registered_rows)
            if checkpoint_selection_improved(
                checkpoint_eligible,
                best_checkpoint_eligible,
                selection_key,
                best_selection_key,
                config["early_stopping_min_delta"],
            ):
                best_score = score
                best_selection_key = selection_key
                best_checkpoint_eligible = checkpoint_eligible
                stale = 0
                torch.save(checkpoint_payload, best_checkpoint)
                _write_csv(run_folder / "validation_registered.csv", registered_rows)
                write_diagnostic_plot(trusted_metrics, run_folder / "validation_registered.png")
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
        "synthetic_validation_gate": checkpoint["record"]["synthetic_validation_gate"],
        "validation_raw_all_products_diagnostic_gate": checkpoint["record"][
            "validation_raw_all_products_diagnostic_gate"
        ],
        "validation_checkpoint_eligible": checkpoint["record"]["checkpoint_eligible"],
        "selection_split": "validation",
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
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
    source_hashes = metadata.get("training_source_sha256") or training_source_hashes()
    git_provenance = metadata.get("git") or git_source_provenance()
    if git_provenance.get("tracked_source_dirty") is not False:
        raise RuntimeError("AtlasPose export requires tracked-clean training source")
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
        "automatic_brain_mask_version": AUTOMATIC_BRAIN_MASK_VERSION,
        "quicknii_coordinate_contract": QUICKNII_COORDINATE_CONTRACT_VERSION,
        "preprocessing_contract_sha256": atlas_pose_preprocessing_contract_sha256(),
        "architecture": metadata.get("config", {}).get("architecture"),
        "pose_representation": metadata.get("config", {}).get("head"),
        "verification_max_absolute_difference": differences,
        "verification_by_provider": verification,
        "verification_sample_count": len(example),
        "verification_input_sha256": hashlib.sha256(
            np.ascontiguousarray(example.numpy()).tobytes()
        ).hexdigest(),
        "source_sha256": source_hashes,
        "git": git_provenance,
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
        "git": metadata.get("git"),
        "selection_split": metadata.get("selection_split"),
        "training_splits": ["synthetic_train", "registered_train"],
        "selection_data": (
            "registered validation specimens with locked coverage/tail gates and synthetic "
            "validation robustness eligibility"
        ),
        "excluded_from_selection": ["registered_test", ATLAS_POSE_SEALED_SPLIT],
    }
    (output_folder / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return metadata


def promote_export(
    export_folder: Path,
    release_report: Path,
    destination: Path = ROOT / "models" / "AtlasPose",
) -> dict[str, str]:
    export_folder = Path(export_folder)
    release_report = Path(release_report)
    model = export_folder / "atlas_pose.onnx"
    model_sha256, metadata_sha256, evidence_sha256, _, report = (
        verify_atlas_pose_release_bundle(model, release_report)
    )
    sealed_predictions = release_report.with_name("SEALED_predictions.csv")
    prediction_table = pd.read_csv(sealed_predictions)
    methods = tuple(sorted(prediction_table["method"].unique()))
    validate_complete_method_cohort(
        prediction_table,
        [{"section_image_id": value} for value in prediction_table["section_image_id"].unique()],
        methods,
    )
    primary_table, _ = evaluation_domains(prediction_table)
    recomputed_quality = release_quality_gate(
        primary_table[primary_table["method"] == "atlas_pose"]
    )
    recomputed_comparisons = {
        axis: paired_animal_bootstrap(
            primary_table,
            "atlas_pose",
            RELEASE_REFERENCE,
            f"absolute_error_{axis}",
        )
        for axis in POSE_AXES
    }
    recomputed_component_passed = {
        axis: bool(
            comparison["delta_candidate_minus_reference"] < 0.0
            and comparison["probability_candidate_lower_error"] >= RELEASE_CONFIDENCE
        )
        for axis, comparison in recomputed_comparisons.items()
    }
    recomputed_joint = paired_animal_joint_superiority(
        primary_table,
        "atlas_pose",
        RELEASE_REFERENCE,
        tuple(f"absolute_error_{axis}" for axis in POSE_AXES),
    )
    if (
        not release_statistics_equal(report.get("quality_gate"), recomputed_quality)
        or not release_statistics_equal(
            report.get("deepslice_comparisons"), recomputed_comparisons
        )
        or report.get("deepslice_component_passed") != recomputed_component_passed
        or not release_statistics_equal(
            report.get("deepslice_simultaneous_superiority"), recomputed_joint
        )
    ):
        raise RuntimeError("AtlasPose release metrics do not reconcile with raw sealed predictions")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("atlas_pose.onnx", "atlas_pose.json", "provenance.json"):
        shutil.copy2(export_folder / name, destination / name)
    for name in (
        "RELEASE_REPORT.json",
        "SEALED_metrics.json",
        "SEALED_predictions.csv",
        "PRESEALED_COMMITMENT.json",
        "SEALED_CLAIM.json",
        "SEALED_CONSUMPTION_RECEIPT.json",
    ):
        shutil.copy2(release_report.with_name(name), destination / name)
    return {
        "APPROVED_ATLAS_POSE_MODEL_SHA256": model_sha256,
        "APPROVED_ATLAS_POSE_METADATA_SHA256": metadata_sha256,
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
    source_hashes = training_source_hashes()
    git_provenance = git_source_provenance()
    registered_hashes = registered_data_hashes(REGISTERED_ROOT)
    atlas_hashes = atlas_data_hashes(ATLAS_FOLDER)
    train, train_path = ensure_fixed_manifest(WORKSPACE, "train", config["samples"], SEEDS["train"])
    validation, validation_path = ensure_fixed_manifest(
        WORKSPACE, "validation", config["validation_count"], SEEDS["validation"]
    )
    train = renderer_variant(train, config["renderer"])
    validation = renderer_variant(validation, config["renderer"])
    paired_train = None
    paired_validation, paired_validation_path = ensure_paired_manifest(
        WORKSPACE, validation, "validation", SEEDS["paired"] + 1
    )
    paired_paths = [paired_validation_path]
    if config["consistency"] > 0.0:
        paired_train, paired_train_path = ensure_paired_manifest(WORKSPACE, train, "train", SEEDS["paired"])
        paired_paths.insert(0, paired_train_path)
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
        supervised_product_ids=tuple(map(int, TRUSTED_REGISTERED_PRODUCTS)),
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
            "label_policy": REGISTERED_LABEL_POLICY,
            "excluded_from_selection": ["test", ATLAS_POSE_SEALED_SPLIT],
        },
        "atlas_data_sha256": atlas_hashes,
        "training_environment": training_environment(),
        "pretrained_backbone": pretrained_provenance,
        "training_source_sha256": source_hashes,
        "git": git_provenance,
    })
    if registered_data_hashes(REGISTERED_ROOT) != registered_hashes:
        raise RuntimeError("Registered dataset changed during training")
    if atlas_data_hashes(ATLAS_FOLDER) != atlas_hashes:
        raise RuntimeError("Atlas data changed during training")
    if training_source_hashes() != source_hashes:
        raise RuntimeError("AtlasPose source changed during training")
    result_path = run_folder / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if export:
        if not result["validation_checkpoint_eligible"]:
            raise RuntimeError(
                "Model export refused because registered or synthetic validation eligibility failed"
            )
        reports = held_out_reports(result)
        result["held_out_reports"] = reports
        if (
            not reports["test"]["final_gate"]["all_gates_passed"]
            or not reports["synthetic_test_gate"]["all_gates_passed"]
        ):
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            raise RuntimeError("Model export refused because registered or synthetic test gates failed")
        export_onnx(model, run_folder / "export", result)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _seed_group_validation_data(
    results: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[dict]]]:
    if not results or any(result["selection_split"] != "validation" for result in results):
        raise RuntimeError("Architecture selection accepts only registered validation results")
    result_by_seed = {int(result["config"]["training_seed"]): result for result in results}
    seeds = tuple(sorted(result_by_seed))
    if len(result_by_seed) != len(results) or set(seeds) != set(COMPARISON_SEEDS):
        raise RuntimeError("Architecture selection requires the three prespecified training seeds")
    run_errors = []
    run_rows = []
    expected_sections = None
    expected_animals = None
    for seed in seeds:
        result = result_by_seed[seed]
        path = Path(result["best_checkpoint"]).parent / "validation_registered.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows = registered_rows_for_products(rows, TRUSTED_REGISTERED_PRODUCTS)
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
        run_rows.append(rows)
    return np.asarray(seeds), expected_animals, np.asarray(run_errors), run_rows


def seed_animal_component_errors(
    results: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seeds, animals, errors, _ = _seed_group_validation_data(results)
    return seeds, animals, errors


def seed_group_selection_summary(results: list[dict]) -> dict:
    _, _, errors, run_rows = _seed_group_validation_data(results)
    selection = _selection_summary_from_component_mae(errors.mean((0, 1)))
    registered_gate = final_acceptance_summary(
        [row for rows in run_rows for row in rows],
        "validation",
        TRUSTED_REGISTERED_PRODUCTS,
    )
    synthetic_by_seed = {
        str(result["config"]["training_seed"]): result["synthetic_validation_gate"]
        for result in results
    }
    synthetic_passed = all(gate["all_gates_passed"] for gate in synthetic_by_seed.values())
    synthetic_worst = max(
        gate["worst_gate_ratio"] for gate in synthetic_by_seed.values()
    )
    return {
        **selection,
        "validation_final_gate": registered_gate,
        "synthetic_validation_by_seed": synthetic_by_seed,
        "model_family_gate": {
            "all_performance_gates_passed": bool(
                registered_gate["all_performance_gates_passed"] and synthetic_passed
            ),
            "all_gates_passed": bool(registered_gate["all_gates_passed"] and synthetic_passed),
            "worst_gate_ratio": max(registered_gate["worst_gate_ratio"], synthetic_worst),
        },
    }


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
        key=lambda candidate: (
            validation_selection_key(
                summaries[candidate], summaries[candidate]["model_family_gate"]
            ),
            tie_priority.index(candidate),
        ),
    )
    point_best = point_order[0]
    comparisons = {}
    tied = [point_best]
    pairwise_confidence = 1.0 - (1.0 - FAMILY_CONFIDENCE) / (len(groups) - 1)
    point_gate = summaries[point_best]["model_family_gate"]
    for opponent in point_order[1:]:
        comparison = bootstrap_seed_group_comparison(
            groups[point_best],
            groups[opponent],
            output_prefix.with_name(f"{output_prefix.name}_{point_best}_vs_{opponent}.json"),
        )
        comparisons[opponent] = comparison
        opponent_gate = summaries[opponent]["model_family_gate"]
        same_gate_tier = (
            point_gate["all_performance_gates_passed"]
            == opponent_gate["all_performance_gates_passed"]
            and np.isclose(
                point_gate["worst_gate_ratio"],
                opponent_gate["worst_gate_ratio"],
                rtol=1e-12,
                atol=1e-9,
            )
        )
        if same_gate_tier and comparison["probability_candidate_better"] < pairwise_confidence:
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
    trusted_rows = registered_rows_for_products(rows, TRUSTED_REGISTERED_PRODUCTS)
    raw_all_products_gate = final_acceptance_summary(rows, split)
    raw_all_products_gate["role"] = "diagnostic_only_not_used_for_release"
    reports[split] = {
        **registered_domain_reports(rows),
        "trusted_registered": registered_domain_reports(trusted_rows),
        "final_gate": final_acceptance_summary(
            trusted_rows,
            split,
            TRUSTED_REGISTERED_PRODUCTS,
        ),
        "raw_all_products_diagnostic_gate": raw_all_products_gate,
    }
    _write_csv(run_folder / f"{split}_registered.csv", rows)
    write_diagnostic_plot(
        reports[split]["trusted_registered"]["primary_in_training_ap_domain"],
        run_folder / f"{split}_registered.png",
    )
    test_manifest, _ = ensure_fixed_manifest(WORKSPACE, "test", 8_192, SEEDS["test"])
    paired_test, _ = ensure_paired_manifest(WORKSPACE, test_manifest, "test", SEEDS["paired"] + 2)
    renderer = SyntheticAtlas(ATLAS_FOLDER, str(device))
    synthetic_report, synthetic_rows = evaluate_synthetic(
        model, renderer, test_manifest, paired_test, config["evaluation_batch_size"]
    )
    reports["synthetic_test"] = synthetic_report
    reports["synthetic_test_gate"] = {
        **synthetic_acceptance_summary(synthetic_report),
        "evaluated_rows_sha256": evaluated_rows_sha256(synthetic_rows),
    }
    _write_csv(run_folder / "synthetic_test.csv", synthetic_rows)
    (run_folder / "held_out_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports


def main() -> None:
    name = os.environ.get("ATLAS_POSE_V7_EXPERIMENT")
    if not name:
        raise RuntimeError("ATLAS_POSE_V7_EXPERIMENT must name one explicitly orchestrated run")
    samples = int(os.environ.get("ATLAS_POSE_V7_SAMPLES", 20_000))
    config = experiment_config(name, samples, {})
    environment_overrides = {
        "ATLAS_POSE_V7_ARCHITECTURE": ("architecture", str),
        "ATLAS_POSE_V7_HEAD": ("head", str),
        "ATLAS_POSE_V7_RENDERER": ("renderer", str),
        "ATLAS_POSE_V7_BATCH_SIZE": ("batch_size", int),
        "ATLAS_POSE_V7_EVALUATION_BATCH_SIZE": ("evaluation_batch_size", int),
        "ATLAS_POSE_V7_DATA_WORKERS": ("data_workers", int),
        "ATLAS_POSE_V7_VALIDATION_COUNT": ("validation_count", int),
        "ATLAS_POSE_V7_VALIDATION_INTERVAL": ("validation_interval", int),
        "ATLAS_POSE_V7_CONSISTENCY": ("consistency", float),
        "ATLAS_POSE_V7_ANATOMY": ("anatomy", float),
        "ATLAS_POSE_V7_REGISTERED_FRACTION": ("registered_fraction", float),
        "ATLAS_POSE_V7_TRAINING_SEED": ("training_seed", int),
    }
    for variable, (key, conversion) in environment_overrides.items():
        if variable in os.environ:
            config[key] = conversion(os.environ[variable])
    run_experiment(config, export=os.environ.get("ATLAS_POSE_V7_EXPORT") == "1")


if __name__ == "__main__":
    main()
