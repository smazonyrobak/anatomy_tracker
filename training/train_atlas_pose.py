from __future__ import annotations

import hashlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

from atlas_pose_models import MODEL_SPECS, AtlasPoseRegressor, PhysicalPoseOutput, set_backbone_trainable
from synthetic_atlas import IMAGE_SIZE, TARGET_CENTER, TARGET_SCALE, SyntheticAtlas, load_manifest, make_manifest, save_manifest


ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "data" / "Allen Brain Atlas 25um"
WORKSPACE = Path(os.environ.get("ATLAS_POSE_WORKSPACE", "G:/AtlasPoseTraining"))
MANIFESTS = WORKSPACE / "manifests_v3"
RESULTS = WORKSPACE / "results_v6"
CACHE = WORKSPACE / "cache"
MODEL_OUTPUT = ROOT / "models" / "AtlasPose"
BATCH_SIZE = 32
VALIDATION_COUNT = 5000
INTERIM_VALIDATION_COUNT = 1000
TEST_COUNT = 5000
STAGES = (5000, 10000, 15000, 20000, 30000)
SEEDS = {"train": 73191, "validation": 19841, "test": 49157}
ACCEPTABLE_ERROR = np.asarray([250.0, 2.0, 2.0], dtype=np.float32)
LOSS_WEIGHTS = (1.0, 2.0, 2.0)
ORIENTATION_LOSS_WEIGHT = 0.35


def ensure_manifests() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    counts = {"train": 100000, "validation": VALIDATION_COUNT, "test": TEST_COUNT}
    values = {}
    for split, count in counts.items():
        path = MANIFESTS / f"{split}_{count}.npz"
        if not path.exists():
            save_manifest(path, make_manifest(count, split, SEEDS[split]))
        values[split] = load_manifest(path)
    return values["train"], values["validation"], values["test"]


def ensure_image_cache(
    manifests: tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fingerprint = hashlib.sha256()
    for path in (
        Path(__file__).with_name("synthetic_atlas.py"),
        ATLAS / "average_template_25.nrrd",
        ATLAS / "annotation_25.nrrd",
    ):
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                fingerprint.update(block)
    for manifest in manifests:
        for name in sorted(manifest):
            values = np.ascontiguousarray(manifest[name])
            fingerprint.update(name.encode("utf-8"))
            fingerprint.update(values.dtype.str.encode("ascii"))
            fingerprint.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            fingerprint.update(values.tobytes())
    cache_folder = CACHE / fingerprint.hexdigest()[:16]
    cache_folder.mkdir(parents=True, exist_ok=True)
    generator = None
    cached = []
    for split, manifest in zip(("train", "validation", "test"), manifests):
        count = len(manifest["ap_um"])
        path = cache_folder / f"{split}_{count}_uint8.npy"
        if not path.exists():
            generator = generator or SyntheticAtlas(ATLAS)
            temporary = path.with_suffix(".partial.npy")
            images = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=(count, IMAGE_SIZE, IMAGE_SIZE))
            for start in range(0, count, 256):
                batch_count = min(256, count - start)
                image, _, _ = generator.batch(manifest, start, batch_count)
                images[start : start + batch_count] = (
                    image[:, 0].mul(255.0).round().byte().cpu().numpy()
                )
                if (start + batch_count) % 5000 == 0 or start + batch_count == count:
                    print(f"cached {split}: {start + batch_count:,}/{count:,}", flush=True)
            images.flush()
            del images
            temporary.replace(path)
        cached.append(np.load(path, mmap_mode="r"))
    return tuple(cached)


def load_batch(
    images: np.ndarray,
    manifest: dict[str, np.ndarray],
    start: int,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image = torch.from_numpy(np.array(images[start : start + count], copy=True)).cuda().float().div_(255.0)
    image = image[:, None].expand(-1, 3, -1, -1)
    physical_target = torch.from_numpy(
        np.column_stack(
            (
                manifest["ap_um"][start : start + count],
                manifest["tilt_lr_deg"][start : start + count],
                manifest["tilt_dv_deg"][start : start + count],
            )
        )
    ).cuda()
    center = torch.as_tensor(TARGET_CENTER, device="cuda")
    scale = torch.as_tensor(TARGET_SCALE, device="cuda")
    rotation = manifest["rotation_deg"][start : start + count]
    orientation_target = torch.from_numpy((np.abs(rotation) > 90.0).astype(np.float32)).cuda()
    return image, (physical_target - center) / scale, physical_target, orientation_target


@torch.inference_mode()
def evaluate(
    model: AtlasPoseRegressor,
    images: np.ndarray,
    manifest: dict[str, np.ndarray],
    count: int,
) -> tuple[dict, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    component_loss_sum = np.zeros(3, dtype=np.float64)
    orientation_correct = 0
    center = torch.as_tensor(TARGET_CENTER, device="cuda")
    scale = torch.as_tensor(TARGET_SCALE, device="cuda")
    for start in range(0, count, BATCH_SIZE):
        batch_count = min(BATCH_SIZE, count - start)
        image, normalized_target, physical_target, orientation_target = load_batch(images, manifest, start, batch_count)
        with torch.autocast("cuda", dtype=torch.float16):
            image_frame_prediction, orientation_logit = model.forward_with_orientation(image)
            orientation_sign = torch.where(orientation_logit > 0.0, -1.0, 1.0)[:, None]
            normalized_prediction = torch.cat(
                (image_frame_prediction[:, :1], image_frame_prediction[:, 1:] * orientation_sign),
                dim=1,
            )
            component_loss = F.smooth_l1_loss(
                normalized_prediction,
                normalized_target,
                beta=0.10,
                reduction="none",
            ).mean(0)
        predictions.append((normalized_prediction * scale + center).float().cpu().numpy())
        targets.append(physical_target.cpu().numpy())
        component_loss_sum += component_loss.float().cpu().numpy() * batch_count
        orientation_correct += int(((orientation_logit > 0) == (orientation_target > 0.5)).sum())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    error = prediction - target
    absolute = np.abs(error)
    mae = absolute.mean(axis=0)
    rmse = np.sqrt((error**2).mean(axis=0))
    p95 = np.percentile(absolute, 95, axis=0)
    normalized_mae = mae / ACCEPTABLE_ERROR
    component_loss = component_loss_sum / count
    metrics = {
        "loss": float(component_loss.mean()),
        "training_objective": float(np.mean(component_loss * np.asarray(LOSS_WEIGHTS))),
        "component_loss": component_loss.tolist(),
        "orientation_accuracy": float(orientation_correct / count),
        "mae": mae.tolist(),
        "rmse": rmse.tolist(),
        "p95": p95.tolist(),
        "normalized_mae": normalized_mae.tolist(),
        "selection_score": float(normalized_mae.mean() + 0.25 * normalized_mae.max()),
    }
    return metrics, np.column_stack([target, prediction])


def train_one(
    architecture: str,
    train_count: int,
    train_images: np.ndarray,
    validation_images: np.ndarray,
    train_manifest: dict[str, np.ndarray],
    validation_manifest: dict[str, np.ndarray],
    *,
    early_stopping: bool = False,
) -> tuple[AtlasPoseRegressor, dict]:
    torch.manual_seed(9000 + list(MODEL_SPECS).index(architecture))
    model = AtlasPoseRegressor(architecture, pretrained=True).cuda()
    set_backbone_trainable(model, False)
    head_parameters = list(model.head.parameters()) + list(model.orientation_head.parameters())
    optimizer = torch.optim.AdamW(head_parameters, lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    loss_weights = torch.tensor(LOSS_WEIGHTS, device="cuda")
    warmup_samples = min(500, max(BATCH_SIZE, train_count // 20))
    validation_interval = 5000 if train_count > 30000 else max(1000, train_count // 6)
    best_score = float("inf")
    best_state = None
    stale_checks = 0
    history = []
    start_time = time.perf_counter()
    samples_seen = 0
    backbone_unfrozen = False

    while samples_seen < train_count:
        if not backbone_unfrozen and samples_seen >= warmup_samples:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": 1e-4},
                    {"params": head_parameters, "lr": 3e-4},
                ],
                weight_decay=1e-4,
            )
            backbone_unfrozen = True
        batch_count = min(BATCH_SIZE, train_count - samples_seen)
        image, normalized_target, _, orientation_target = load_batch(
            train_images,
            train_manifest,
            samples_seen,
            batch_count,
        )
        model.train()
        if samples_seen < warmup_samples:
            model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction, orientation_logit = model.forward_with_orientation(image)
            target_sign = torch.where(orientation_target > 0.5, -1.0, 1.0)[:, None]
            image_frame_target = torch.cat(
                (normalized_target[:, :1], normalized_target[:, 1:] * target_sign),
                dim=1,
            )
            component_loss = F.smooth_l1_loss(
                prediction,
                image_frame_target,
                beta=0.10,
                reduction="none",
            ).mean(0)
            pose_loss = (component_loss * loss_weights).mean()
            orientation_loss = F.binary_cross_entropy_with_logits(orientation_logit, orientation_target)
            loss = pose_loss + ORIENTATION_LOSS_WEIGHT * orientation_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        samples_seen += batch_count

        if samples_seen % validation_interval < BATCH_SIZE or samples_seen == train_count:
            validation_metrics, _ = evaluate(
                model,
                validation_images,
                validation_manifest,
                INTERIM_VALIDATION_COUNT,
            )
            record = {
                "samples_seen": samples_seen,
                "train_loss": float(loss.detach()),
                "train_component_loss": component_loss.detach().float().cpu().tolist(),
                "train_orientation_loss": float(orientation_loss.detach()),
                "validation": validation_metrics,
            }
            history.append(record)
            print(
                f"{architecture} {train_count}: {samples_seen}/{train_count} "
                f"train={float(loss.detach()):.5f} val={validation_metrics['training_objective']:.5f} "
                f"MAE={validation_metrics['mae']}",
                flush=True,
            )
            if validation_metrics["selection_score"] < best_score:
                best_score = validation_metrics["selection_score"]
                best_state = deepcopy(model.state_dict())
                stale_checks = 0
            else:
                stale_checks += 1
            if early_stopping and samples_seen >= 30000 and stale_checks >= 4:
                break

    model.load_state_dict(best_state)
    validation_metrics, validation_predictions = evaluate(
        model,
        validation_images,
        validation_manifest,
        VALIDATION_COUNT,
    )
    report = {
        "architecture": architecture,
        "backbone": MODEL_SPECS[architecture][0],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "requested_training_images": train_count,
        "training_images_seen": samples_seen,
        "unique_training_images": True,
        "elapsed_seconds": time.perf_counter() - start_time,
        "best_validation": validation_metrics,
        "history": history,
    }
    output = RESULTS / f"{train_count}" / architecture
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "best.pt")
    np.savez_compressed(output / "validation_predictions.npz", values=validation_predictions)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_history(report, output / "loss.png")
    return model, report


def plot_history(report: dict, path: Path) -> None:
    samples = [row["samples_seen"] for row in report["history"]]
    training = [row["train_loss"] for row in report["history"]]
    validation = [row["validation"]["training_objective"] for row in report["history"]]
    mae = np.asarray([row["validation"]["mae"] for row in report["history"]])
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(samples, training, label="training")
    axes[0, 0].plot(samples, validation, label="validation")
    axes[0, 0].set(xlabel="unique training images", ylabel="normalized Huber loss")
    axes[0, 0].legend()
    for axis, column, label in zip(axes.flat[1:], range(3), ("AP (µm)", "L–R (°)", "D–V (°)")):
        axis.plot(samples, mae[:, column])
        axis.set(xlabel="unique training images", ylabel=f"validation {label} MAE")
    figure.suptitle(f"{report['architecture']} — {report['requested_training_images']:,} images")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_stage_comparison(train_count: int, reports: dict[str, dict]) -> None:
    labels = list(reports)
    mae = np.asarray([reports[name]["best_validation"]["mae"] for name in labels])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for column, title in enumerate(("AP MAE (µm)", "L–R MAE (°)", "D–V MAE (°)")):
        axes[column].bar(labels, mae[:, column])
        axes[column].set_title(title)
        axes[column].tick_params(axis="x", rotation=20)
    figure.suptitle(f"Held-out validation after {train_count:,} unique images")
    figure.tight_layout()
    figure.savefig(RESULTS / f"comparison_{train_count}.png", dpi=160)
    plt.close(figure)


def run_stage(train_count: int) -> dict[str, dict]:
    manifests = ensure_manifests()
    train_manifest, validation_manifest, _ = manifests
    train_images, validation_images, _ = ensure_image_cache(manifests)
    reports = {}
    for architecture in MODEL_SPECS:
        model, reports[architecture] = train_one(
            architecture,
            train_count,
            train_images,
            validation_images,
            train_manifest,
            validation_manifest,
        )
        del model
        torch.cuda.empty_cache()
    plot_stage_comparison(train_count, reports)
    (RESULTS / f"comparison_{train_count}.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports


def select_winner() -> str:
    reports = json.loads((RESULTS / "comparison_30000.json").read_text(encoding="utf-8"))
    return min(reports, key=lambda name: reports[name]["best_validation"]["selection_score"])


def export_model(
    model: AtlasPoseRegressor,
    architecture: str,
    report: dict,
    test_metrics: dict,
    test_images: np.ndarray,
) -> Path:
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    physical = PhysicalPoseOutput(
        model.eval().cpu(),
        torch.as_tensor(TARGET_CENTER),
        torch.as_tensor(TARGET_SCALE),
    ).eval()
    temporary = WORKSPACE / "atlas_pose_fp32.onnx"
    example = torch.from_numpy(np.array(test_images[:16], copy=True)).float().div_(255.0)
    example = example[:, None].expand(-1, 3, -1, -1)
    torch.onnx.export(
        physical,
        example,
        temporary,
        input_names=["images"],
        output_names=["pose_ap_um_lr_deg_dv_deg", "orientation_inverted_logit"],
        dynamic_axes={
            "images": {0: "batch"},
            "pose_ap_um_lr_deg_dv_deg": {0: "batch"},
            "orientation_inverted_logit": {0: "batch"},
        },
        opset_version=18,
        dynamo=False,
    )
    fp32 = onnx.load(temporary)
    model_path = MODEL_OUTPUT / "atlas_pose.onnx"
    onnx.save(fp32, model_path)
    temporary.unlink()
    onnx.checker.check_model(onnx.load(model_path))
    torch_pose, torch_orientation = (value.detach().numpy() for value in physical(example))
    conversion_validation = {}
    tolerance = np.asarray([1.0, 0.02, 0.02], dtype=np.float32)
    for provider in ("CPUExecutionProvider", "DmlExecutionProvider"):
        if provider not in ort.get_available_providers():
            continue
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=[provider])
        onnx_pose, onnx_orientation = session.run(None, {"images": example.numpy()})
        difference = np.abs(torch_pose - onnx_pose)
        orientation_difference = np.abs(torch_orientation - onnx_orientation)
        maximum_difference = difference.max(axis=0)
        if np.any(maximum_difference > tolerance) or orientation_difference.max() > 0.001:
            raise RuntimeError(
                f"ONNX {provider} conversion exceeded tolerance: "
                f"pose={maximum_difference}, orientation={orientation_difference.max()}"
            )
        conversion_validation[provider] = {
            "pose_mean_abs_difference": difference.mean(axis=0).tolist(),
            "pose_max_abs_difference": maximum_difference.tolist(),
            "orientation_logit_mean_abs_difference": float(orientation_difference.mean()),
            "orientation_logit_max_abs_difference": float(orientation_difference.max()),
        }
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {
        "architecture": architecture,
        "precision": "float32",
        "preprocessing_version": "smart-mask-scale-invariant-v1",
        "preprocessing_source_sha256": hashlib.sha256(
            (ROOT / "source" / "atlas_pose_runtime.py").read_bytes()
        ).hexdigest(),
        "input": [None, 3, IMAGE_SIZE, IMAGE_SIZE],
        "output": ["AP from bregma (um; anterior positive)", "L-R tilt (deg)", "D-V tilt (deg)"],
        "auxiliary_output": "orientation_inverted_logit",
        "sha256": digest,
        "real_benchmark_informed_final_iteration": True,
        "real_benchmark_note": (
            "The first v5 real-histology run was untouched. Its domain-gap result informed the v6 "
            "full-range AP stratification, realistic tilt mixture, and contrast-polarity augmentation."
        ),
        "onnx_conversion_validation": conversion_validation,
        "training": report,
        "held_out_test": test_metrics,
    }
    (MODEL_OUTPUT / "atlas_pose.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


def train_final(architecture: str | None = None) -> dict:
    manifests = ensure_manifests()
    train_manifest, validation_manifest, test_manifest = manifests
    train_images, validation_images, test_images = ensure_image_cache(manifests)
    architecture = architecture or select_winner()
    model, report = train_one(
        architecture,
        100000,
        train_images,
        validation_images,
        train_manifest,
        validation_manifest,
        early_stopping=True,
    )
    test_metrics, test_predictions = evaluate(model, test_images, test_manifest, TEST_COUNT)
    output = RESULTS / "100000" / architecture
    np.savez_compressed(output / "test_predictions.npz", values=test_predictions)
    (output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    model_path = export_model(model, architecture, report, test_metrics, test_images)
    return {"architecture": architecture, "model": str(model_path), "test": test_metrics}


if __name__ == "__main__":
    for stage in STAGES:
        run_stage(stage)
    print(json.dumps(train_final(), indent=2))
