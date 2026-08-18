"""Run the frozen, development-only initializer-foundation qualification.

This diagnostic trains only the independent source encoder and probabilistic
pose head.  It neither compares architectures nor calls the recurrent,
pairwise, atlas-encoder, or dense-registration paths.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import training.independent_joint_data as independent_data
from training.independent_joint_model import IndependentJointModel
from training.quicknii_plane_metric import (
    QUICKNII_PIXEL_GRID_SHAPE,
    QUICKNII_PLANE_DISTANCE_CONTRACT,
    QUICKNII_SHAPE_ML_AP_DV,
    torch_annotation_brain_mask,
    torch_brain_masked_plane_distance,
)
from training.synthetic_registration import SyntheticRegistrationGenerator
from training.train_independent_joint import (
    _atomic_save,
    _rng_state,
    _set_rng_state,
    initializer_pose_losses,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
SCHEMA_VERSION = 1
PURPOSE = "development-only-cold-start-initializer-foundation-qualification"
FORMAT = "independent-initializer-foundation-v1"
SOURCE_FILES = (
    "training/independent_joint_model.py",
    "training/independent_joint_data.py",
    "training/train_independent_joint.py",
    "training/synthetic_registration.py",
    "training/quicknii_plane_metric.py",
    "training/run_independent_initializer_foundation.py",
    "source/dense_registration_preprocessing.py",
)
OUTLINE_MODES = {
    name: index for index, name in enumerate(independent_data.OUTLINE_MODE_NAMES)
}


def _canonical(value):
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if torch.is_tensor(value):
        return _canonical(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_sha256(path: str | Path) -> str:
    source = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(source).hexdigest()


def _binary_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _named_parameter_sha256(model: torch.nn.Module, names) -> str:
    selected = set(names)
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if name not in selected:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_subset_sha256(state: dict[str, torch.Tensor], names) -> str:
    selected = set(names)
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name not in selected:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _update_initializer_ema(
    ema: dict[str, torch.Tensor],
    model: torch.nn.Module,
    decay: float,
    trainable_names,
) -> None:
    selected = set(trainable_names)
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if name in selected and value.is_floating_point():
                ema[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema[name].copy_(value)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validate_rng_state_types(state: dict) -> None:
    if not isinstance(state.get("torch"), torch.ByteTensor):
        raise RuntimeError("foundation CPU torch RNG state is not a ByteTensor")
    cuda = state.get("cuda")
    if cuda is not None and (
        not isinstance(cuda, list)
        or any(not isinstance(value, torch.ByteTensor) for value in cuda)
    ):
        raise RuntimeError("foundation CUDA RNG state is not a list of CPU ByteTensors")


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(_canonical(payload), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_foundation_config(path: str | Path) -> dict:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    contract = raw.pop("contract_sha256")
    if _canonical_sha256(raw) != contract:
        raise ValueError("initializer-foundation config hash differs from its payload")
    if raw.get("schema_version") != SCHEMA_VERSION or raw.get("frozen") is not True:
        raise ValueError("initializer-foundation config is not frozen schema v1")
    if raw.get("purpose") != PURPOSE or raw.get("role") != "diagnostic-not-architecture-selection":
        raise ValueError("initializer-foundation role or purpose changed")
    if any(raw.get(name) is not False for name in (
        "product5_access", "calibration_access", "final_test_access"
    )) or raw.get("learned_checkpoint_dependencies") != []:
        raise ValueError("qualification must be synthetic-only and cold-start")
    if int(raw.get("seed", -1)) != 4322:
        raise ValueError("qualification seed must remain 4322")
    if raw.get("device") != "auto" or raw["paths"] != {
        "atlas_repo_relative": "data/Allen Brain Atlas 25um",
        "run_root_env": "ATLAS_JOINT_RUN_ROOT",
    }:
        raise ValueError("foundation device or path contract changed")
    atlas_relative = Path(raw["paths"]["atlas_repo_relative"])
    if atlas_relative.is_absolute() or ".." in atlas_relative.parts:
        raise ValueError("atlas path must remain repository-relative")
    if raw["model"] != {
        "class": "training.independent_joint_model.IndependentJointModel",
        "expected_parameter_count": 1369070,
        "kwargs": {
            "pyramid_channels": [24, 40, 64, 96],
            "pose_context_features": 192,
            "pair_features": 96,
            "hidden_channels": 96,
            "integration_steps": 6,
            "maximum_pose_delta": [750.0, 7.5, 7.5],
            "maximum_translation_pixels": 32.0,
            "minimum_scale": 0.4,
            "maximum_scale": 2.0,
            "maximum_velocity_fraction": 0.12,
        },
    }:
        raise ValueError("foundation model payload changed")

    data = raw["data"]
    if (
        data.get("source") != "synthetic_ccf"
        or data.get("split") != "train"
        or data.get("pose_regime") != "standard"
        or int(data.get("total_unique_views", -1)) != 2500
        or int(data.get("batch_size", -1)) != 2
        or int(data.get("optimizer_updates", -1)) != 1250
        or int(data.get("negative_candidates_unused", -1)) != 1
    ):
        raise ValueError("foundation data workload changed")
    cycle = data.get("stratum_cycle", [])
    if len(cycle) != 10 or cycle.count("clean") != 7 or cycle.count("mild") != 3:
        raise ValueError("foundation strata must remain exactly 70% clean and 30% mild")
    if data.get("outline_probabilities") != {
        "accurate": 0.35, "imperfect": 0.35, "absent": 0.30
    }:
        raise ValueError("foundation outline curriculum changed")
    if any(
        data.get(name) != value
        for name, value in {
            "base_seed": 104322,
            "seed_stride": 7919,
            "outline_seed": 204322,
            "stratum_probabilities": {"clean": 0.70, "mild": 0.30},
        }.items()
    ):
        raise ValueError("foundation data schedule seeds or proportions changed")
    nuisance = data["source_view_nuisance"]
    if nuisance != {
        "first_half_rotation_abs_max_deg": 30.0,
        "first_half_scale": [0.8, 1.2],
        "final_rotation_abs_max_deg": 90.0,
        "final_scale": [0.7, 1.3],
        "second_half_schedule": "linear",
    }:
        raise ValueError("foundation nuisance schedule changed")

    training = raw["training"]
    required_training = {
        "optimizer": "AdamW",
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "amp_initial_scale": 512.0,
        "warmup_views": 500,
        "learning_rate_after_warmup": "hold",
        "gaussian_nll_weight_start": 0.01,
        "gaussian_nll_weight_end": 0.05,
        "encoder_gradient_clip_norm": 5.0,
        "head_gradient_clip_norm": 5.0,
        "ema_decay": 0.99,
    }
    if any(training.get(name) != value for name, value in required_training.items()):
        raise ValueError("foundation optimizer, loss, clipping, or EMA contract changed")
    if (
        training.get("amp") is not True
        or training.get("loss_weights") != {
            "initializer_categorical": 1.0,
            "initializer_sub_bin": 0.5,
            "initializer_plane_anchor": 0.25,
        }
        or int(training.get("checkpoint_every_views", -1)) != 100
        or training.get("resume") is not True
    ):
        raise ValueError("foundation loss, AMP, checkpoint, or resume contract changed")
    development = raw["development"]
    if development.get("evaluation_views") != [0, 500, 1000, 1500, 2000, 2500]:
        raise ValueError("foundation development schedule changed")
    if int(development.get("count_per_outline", -1)) < 8:
        raise ValueError("each fixed outline panel needs at least eight views")
    if any(
        development.get(name) != value
        for name, value in {
            "partition": "development",
            "source": "synthetic_ccf",
            "pose_regime": "standard",
            "count_per_outline": 8,
            "seed": 404322,
            "stratum": "mild",
            "primary_outline_mode": "absent",
            "state": "ema",
            "physical_metric": "deepslice-corresponding-pixel-plane-distance-v1",
        }.items()
    ):
        raise ValueError("foundation development panel contract changed")
    gates = raw["gates"]
    if gates != {
        "interim_views": 1000,
        "interim_overall_physical_reduction_minimum": 0.15,
        "final_views": 2500,
        "final_overall_physical_reduction_minimum": 0.35,
        "final_each_axis_reduction_minimum": 0.20,
        "final_absent_physical_reduction_minimum": 0.25,
        "nonfinite_count_maximum": 0,
        "postwarm_median_clip_factor_minimum": 0.10,
        "each_group_clipped_fraction_maximum": 0.75,
        "stop_on_interim_failure": True,
    }:
        raise ValueError("foundation stop/go gates changed")
    expected_sources = raw["lineage"]["source_sha256"]
    if set(expected_sources) != set(SOURCE_FILES):
        raise ValueError("initializer-foundation source lineage is incomplete")
    for relative, expected in expected_sources.items():
        if _source_sha256(REPOSITORY_ROOT / relative) != expected:
            raise ValueError(f"source lineage changed: {relative}")
    raw["contract_sha256"] = contract
    raw["config_file_sha256"] = _source_sha256(path)
    return raw


def nuisance_limits(view_index: int, total_views: int) -> dict[str, float]:
    half = total_views // 2
    if view_index < half:
        progress = 0.0
    else:
        progress = (view_index - half) / max(total_views - half - 1, 1)
    return {
        "rotation_abs_max_deg": 30.0 + 60.0 * progress,
        "scale_minimum": 0.8 - 0.1 * progress,
        "scale_maximum": 1.2 + 0.1 * progress,
        "second_half_progress": progress,
    }


def _slice_outline_plan(plan: dict, start: int, stop: int) -> dict:
    mode = np.asarray(plan["mode"])[start:stop].copy()
    result = {
        "mode_probabilities": np.asarray(plan["mode_probabilities"]).copy(),
        "mode_counts": np.bincount(mode, minlength=3).astype(np.int64),
        "mode": mode,
        "morphology_px": np.asarray(plan["morphology_px"])[start:stop].copy(),
        "jitter_amplitude_px": np.asarray(plan["jitter_amplitude_px"])[start:stop].copy(),
        "jitter_seed": np.asarray(plan["jitter_seed"])[start:stop].copy(),
        "sample_receipt_sha256": list(plan["sample_receipt_sha256"][start:stop]),
    }
    result["plan_sha256"] = independent_data._payload_sha256(result)
    return result


def _forced_outline_plan(mode: int, count: int, seed: int) -> dict:
    rng = np.random.default_rng(int(seed))
    imperfect = mode == OUTLINE_MODES["imperfect"]
    morphology = np.zeros(count, np.int8)
    jitter = np.zeros(count, np.float32)
    if imperfect:
        morphology = rng.choice(np.asarray((-3, -2, -1, 1, 2, 3), np.int8), count)
        jitter = rng.uniform(0.5, 2.5, count).astype(np.float32)
    jitter_seed = rng.integers(
        0, np.iinfo(np.uint64).max, count, dtype=np.uint64, endpoint=True
    )
    plan = {
        "mode_probabilities": independent_data.OUTLINE_MODE_PROBABILITIES.astype(np.float32),
        "mode_counts": np.eye(3, dtype=np.int64)[mode] * count,
        "mode": np.full(count, mode, np.int8),
        "morphology_px": morphology,
        "jitter_amplitude_px": jitter,
        "jitter_seed": jitter_seed,
    }
    plan["sample_receipt_sha256"] = [
        independent_data._payload_sha256(
            {
                "mode": mode,
                "morphology_px": int(morphology[item]),
                "jitter_amplitude_px": float(jitter[item]),
                "jitter_seed": int(jitter_seed[item]),
                "gap_count": int(imperfect),
                "island_count": int(imperfect),
                "contract": independent_data.OUTLINE_CURRICULUM_CONTRACT,
            }
        )
        for item in range(count)
    ]
    plan["plan_sha256"] = independent_data._payload_sha256(plan)
    return plan


def _set_source_view(manifest: dict, seed: int, limits: dict) -> dict:
    manifest = copy.deepcopy(manifest)
    count = int(manifest["sample_count"])
    rng = independent_data._rng(seed, "initializer-foundation-source-view")
    manifest["source_view_rotation_deg"] = independent_data._stratified_uniform(
        rng, count, -limits["rotation_abs_max_deg"], limits["rotation_abs_max_deg"]
    )
    manifest["source_view_scale"] = independent_data._stratified_uniform(
        rng, count, limits["scale_minimum"], limits["scale_maximum"]
    )
    manifest["foundation_source_view_limits"] = dict(limits)
    return manifest


def training_manifest(synthetic, config: dict, update: int, outline_plan: dict) -> dict:
    data = config["data"]
    batch_size = int(data["batch_size"])
    seed = int(data["base_seed"]) + update * int(data["seed_stride"])
    stratum = data["stratum_cycle"][update % len(data["stratum_cycle"])]
    manifest = synthetic.make_manifest(
        batch_size,
        "train",
        seed,
        stratum,
        int(data["negative_candidates_unused"]),
        pose_regime="standard",
    )
    start = update * batch_size
    manifest = copy.deepcopy(manifest)
    rng = independent_data._rng(seed, "initializer-foundation-source-view")
    rotation_unit = independent_data._stratified_uniform(rng, batch_size, -1.0, 1.0)
    scale_unit = independent_data._stratified_uniform(rng, batch_size, 0.0, 1.0)
    limits = [nuisance_limits(start + row, int(data["total_unique_views"])) for row in range(batch_size)]
    manifest["source_view_rotation_deg"] = np.asarray(
        [rotation_unit[row] * limits[row]["rotation_abs_max_deg"] for row in range(batch_size)],
        np.float32,
    )
    manifest["source_view_scale"] = np.asarray(
        [
            limits[row]["scale_minimum"]
            + scale_unit[row] * (limits[row]["scale_maximum"] - limits[row]["scale_minimum"])
            for row in range(batch_size)
        ],
        np.float32,
    )
    manifest["foundation_source_view_limits"] = limits
    manifest["outline_plan"] = _slice_outline_plan(
        outline_plan, start, start + batch_size
    )
    manifest["manifest_sha256"] = independent_data._payload_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def _view_receipts(manifest: dict, update: int) -> list[dict]:
    records = []
    for row in range(int(manifest["sample_count"])):
        record = {
            "view_index": update * int(manifest["sample_count"]) + row,
            "optimizer_update": update,
            "row": row,
            "manifest_sha256": manifest["manifest_sha256"],
            "generator_manifest_sha256": manifest["generator_manifest"]["manifest_sha256"],
            "stratum": manifest["stratum"],
            "pose_regime": manifest["pose_regime"],
            "true_pose": np.asarray(manifest["true_pose"])[row],
            "source_view_rotation_deg": float(manifest["source_view_rotation_deg"][row]),
            "source_view_scale": float(manifest["source_view_scale"][row]),
            "outline_mode": int(manifest["outline_plan"]["mode"][row]),
            "outline_receipt_sha256": manifest["outline_plan"]["sample_receipt_sha256"][row],
        }
        record["view_content_sha256"] = _canonical_sha256(
            {
                name: record[name]
                for name in (
                    "manifest_sha256",
                    "generator_manifest_sha256",
                    "row",
                    "true_pose",
                    "source_view_rotation_deg",
                    "source_view_scale",
                    "outline_mode",
                    "outline_receipt_sha256",
                )
            }
        )
        record["view_provenance_sha256"] = _canonical_sha256(record)
        records.append(record)
    return records


def _pose_to_quicknii_ouv(pose: torch.Tensor) -> torch.Tensor:
    pose = torch.as_tensor(pose)
    ap_um, tilt_lr_deg, tilt_dv_deg = pose.unbind(dim=-1)
    slope_lr = torch.tan(torch.deg2rad(tilt_lr_deg))
    slope_dv = torch.tan(torch.deg2rad(tilt_dv_deg))
    ap_index = independent_data.BREGMA_AP_INDEX - ap_um / independent_data.VOXEL_UM
    ml_center = (QUICKNII_SHAPE_ML_AP_DV[0] - 1.0) / 2.0
    dv_center = (QUICKNII_SHAPE_ML_AP_DV[2] - 1.0) / 2.0
    origin_ap = ap_index - slope_lr * ml_center - slope_dv * dv_center
    zeros = torch.zeros_like(ap_um)
    return torch.stack(
        (
            zeros,
            zeros + QUICKNII_SHAPE_ML_AP_DV[1] - origin_ap,
            zeros + QUICKNII_SHAPE_ML_AP_DV[2],
            zeros + QUICKNII_SHAPE_ML_AP_DV[0],
            -QUICKNII_SHAPE_ML_AP_DV[0] * slope_lr,
            zeros,
            zeros,
            -QUICKNII_SHAPE_ML_AP_DV[2] * slope_dv,
            zeros - QUICKNII_SHAPE_ML_AP_DV[2],
        ),
        dim=-1,
    )


def _panel_batch(batch: dict) -> dict:
    names = (
        "source_image", "source_mask", "mask_available", "input_outline_mode",
        "input_outline_receipt_sha256", "true_pose", "sample_manifest_sha256",
        "data_contract_sha256", "source_type", "data_split",
    )
    return {
        name: value.detach().cpu() if torch.is_tensor(value) else value
        for name, value in batch.items() if name in names
    }


def _development_setup(config: dict, synthetic, generator):
    settings = config["development"]
    count = int(settings["count_per_outline"])
    seed = int(settings["seed"])
    base = synthetic.make_manifest(
        count, "validation", seed, settings["stratum"], 1, pose_regime="standard"
    )
    base = _set_source_view(
        base,
        seed,
        {
            "rotation_abs_max_deg": 90.0,
            "scale_minimum": 0.7,
            "scale_maximum": 1.3,
            "second_half_progress": 1.0,
        },
    )
    manifests, batches = {}, {}
    for name, mode in OUTLINE_MODES.items():
        manifest = copy.deepcopy(base)
        manifest["outline_plan"] = _forced_outline_plan(mode, count, seed + mode)
        manifest["manifest_sha256"] = independent_data._payload_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        manifests[name] = manifest
        batches[name] = _panel_batch(synthetic.batch(manifest))
    true_pose = batches["absent"]["true_pose"].to(generator.device, torch.float64)
    truth_ouv = _pose_to_quicknii_ouv(true_pose)
    brain_mask = torch_annotation_brain_mask(
        truth_ouv, generator.annotation, QUICKNII_PIXEL_GRID_SHAPE
    ).cpu()
    panel_contract = {
        "version": 1,
        "purpose": "fixed-synthetic-development-diagnostic-only",
        "partition": "development",
        "product5_access": False,
        "calibration_access": False,
        "final_test_access": False,
        "primary_outline_mode": "absent",
        "count_per_outline": count,
        "manifest_sha256": {name: value["manifest_sha256"] for name, value in manifests.items()},
        "input_sha256": {
            name: {
                key: _tensor_sha256(batch[key])
                for key in ("source_image", "source_mask", "mask_available", "true_pose")
            }
            for name, batch in batches.items()
        },
        "brain_mask_sha256": _tensor_sha256(brain_mask),
        "physical_plane_distance_contract": QUICKNII_PLANE_DISTANCE_CONTRACT,
        "evaluation_views": settings["evaluation_views"],
    }
    panel_contract["contract_sha256"] = _canonical_sha256(panel_contract)
    return panel_contract, batches, brain_mask


def _development_evaluator(panel_contract, batches, brain_mask, device):
    def evaluate(model: IndependentJointModel, views: int) -> dict:
        by_mode, raw, all_prediction, all_truth = {}, [], [], []
        nonfinite = 0
        for mode_name in independent_data.OUTLINE_MODE_NAMES:
            cpu = batches[mode_name]
            image = cpu["source_image"].to(device)
            outline = cpu["source_mask"].to(device)
            available = cpu["mask_available"].to(device)
            truth = cpu["true_pose"].to(device)
            output = model.initialize(image, outline, available)
            nonfinite += sum(
                int((~torch.isfinite(value)).sum()) for value in output.values()
            )
            prediction = output["pose"]
            physical = torch_brain_masked_plane_distance(
                _pose_to_quicknii_ouv(truth.to(torch.float64)),
                _pose_to_quicknii_ouv(prediction.to(torch.float64)),
                brain_mask.to(device),
            ) * independent_data.VOXEL_UM
            absolute = (prediction - truth).abs()
            by_mode[mode_name] = {
                "physical_corresponding_plane_error_um": float(physical.mean()),
                "ap_mae_um": float(absolute[:, 0].mean()),
                "lr_mae_deg": float(absolute[:, 1].mean()),
                "dv_mae_deg": float(absolute[:, 2].mean()),
                "sample_count": len(truth),
            }
            all_prediction.append(prediction.detach().cpu())
            all_truth.append(truth.detach().cpu())
            for item in range(len(truth)):
                record = {
                    "panel_contract_sha256": panel_contract["contract_sha256"],
                    "views": views,
                    "outline_mode": mode_name,
                    "sample_index": item,
                    "sample_manifest_sha256": cpu["sample_manifest_sha256"],
                    "input_outline_receipt_sha256": cpu["input_outline_receipt_sha256"][item],
                    "true_pose": truth[item].detach().cpu(),
                    "predicted_pose": prediction[item].detach().cpu(),
                    "pose_cholesky": output["pose_cholesky"][item].detach().cpu(),
                    "pose_covariance": output["pose_covariance"][item].detach().cpu(),
                    "physical_corresponding_plane_error_um": float(physical[item]),
                }
                record["record_provenance_sha256"] = _canonical_sha256(
                    {
                        "panel_contract_sha256": panel_contract["contract_sha256"],
                        "outline_mode": mode_name,
                        "sample_index": item,
                        "sample_manifest_sha256": cpu["sample_manifest_sha256"],
                    }
                )
                raw.append(record)
        prediction = torch.cat(all_prediction)
        truth = torch.cat(all_truth)
        physical_values = torch.tensor(
            [record["physical_corresponding_plane_error_um"] for record in raw]
        )
        absolute = (prediction - truth).abs()
        overall = {
            "physical_corresponding_plane_error_um": float(physical_values.mean()),
            "ap_mae_um": float(absolute[:, 0].mean()),
            "lr_mae_deg": float(absolute[:, 1].mean()),
            "dv_mae_deg": float(absolute[:, 2].mean()),
            "sample_count": len(truth),
        }
        result = {
            "partition": "development",
            "fresh_checkpoint_views": int(views),
            "model_state": "ema",
            "panel_contract_sha256": panel_contract["contract_sha256"],
            "primary_outline_mode": "absent",
            "overall": overall,
            "by_outline_mode": by_mode,
            "nonfinite_output_count": nonfinite,
            "raw_predictions": raw,
        }
        result["panel_manifest_sha256"] = _canonical_sha256(result)
        return result

    return evaluate


def _payload_is_finite(value) -> bool:
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return not np.issubdtype(value.dtype, np.floating) or bool(np.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_payload_is_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_payload_is_finite(item) for item in value)
    return True


def _validate_gradient_records(records: list[dict], config: dict) -> None:
    data, training = config["data"], config["training"]
    batch_size = int(data["batch_size"])
    total_updates = int(data["optimizer_updates"])
    warmup_updates = int(training["warmup_views"]) // batch_size
    base_learning_rate = float(training["learning_rate"])
    start = float(training["gaussian_nll_weight_start"])
    end = float(training["gaussian_nll_weight_end"])
    for index, record in enumerate(records):
        if (
            int(record.get("update", -1)) != index + 1
            or int(record.get("views_after", -1)) != (index + 1) * batch_size
        ):
            raise RuntimeError("foundation gradient sequence is invalid")
        expected_lr = _learning_rate(index, warmup_updates, base_learning_rate)
        expected_gaussian = _gaussian_weight(index, total_updates, start, end)
        if (
            not _payload_is_finite(record)
            or not math.isclose(float(record["learning_rate"]), expected_lr, rel_tol=1e-12)
            or not math.isclose(
                float(record["gaussian_nll_weight"]), expected_gaussian, rel_tol=1e-12
            )
        ):
            raise RuntimeError("foundation gradient schedule or telemetry is invalid")
        for group, limit_name in (
            ("encoder", "encoder_gradient_clip_norm"),
            ("head", "head_gradient_clip_norm"),
        ):
            values = record[group]
            pre = float(values["preclip_norm"])
            post = float(values["postclip_norm"])
            factor = float(values["clip_factor"])
            limit = float(training[limit_name])
            clipped = pre > limit
            expected_factor = min(1.0, post / pre) if clipped else 1.0
            expected_post = (
                pre * min(1.0, limit / (pre + 1e-6)) if clipped else pre
            )
            if (
                pre < 0.0
                or post < 0.0
                or not 0.0 <= factor <= 1.0
                or bool(values.get("clipped")) != clipped
                or not math.isclose(post, expected_post, rel_tol=5e-5, abs_tol=1e-5)
                or not math.isclose(factor, expected_factor, rel_tol=1e-9, abs_tol=1e-12)
                or (clipped and post > limit * (1.0 + 1e-5))
            ):
                raise RuntimeError("foundation gradient clipping telemetry is invalid")


def _validate_evaluations(
    evaluations: list[dict],
    views: int,
    panel_contract: dict,
    config: dict,
) -> None:
    expected_views = [
        int(value)
        for value in config["development"]["evaluation_views"]
        if int(value) <= views
    ]
    if [int(panel.get("fresh_checkpoint_views", -1)) for panel in evaluations] != expected_views:
        raise RuntimeError("foundation checkpoint development sequence is invalid")
    expected_per_mode = int(config["development"]["count_per_outline"])
    expected_total = expected_per_mode * len(independent_data.OUTLINE_MODE_NAMES)
    contract_sha256 = panel_contract["contract_sha256"]
    metrics = (
        "physical_corresponding_plane_error_um",
        "ap_mae_um",
        "lr_mae_deg",
        "dv_mae_deg",
    )
    for panel in evaluations:
        unhashed = {
            name: value for name, value in panel.items() if name != "panel_manifest_sha256"
        }
        if (
            panel.get("panel_manifest_sha256") != _canonical_sha256(unhashed)
            or panel.get("panel_contract_sha256") != contract_sha256
            or panel.get("model_state") != "ema"
            or panel.get("partition") != "development"
            or panel.get("primary_outline_mode") != "absent"
        ):
            raise RuntimeError("foundation development panel provenance is invalid")
        by_mode = panel.get("by_outline_mode", {})
        raw = panel.get("raw_predictions", [])
        if (
            set(by_mode) != set(independent_data.OUTLINE_MODE_NAMES)
            or int(panel["overall"].get("sample_count", -1)) != expected_total
            or len(raw) != expected_total
        ):
            raise RuntimeError("foundation development panel sample counts are invalid")
        nonfinite = int(panel.get("nonfinite_output_count", -1))
        if nonfinite < 0:
            raise RuntimeError("foundation development nonfinite count is invalid")
        for mode in independent_data.OUTLINE_MODE_NAMES:
            mode_records = [record for record in raw if record.get("outline_mode") == mode]
            if (
                int(by_mode[mode].get("sample_count", -1)) != expected_per_mode
                or len(mode_records) != expected_per_mode
                or sorted(int(record.get("sample_index", -1)) for record in mode_records)
                != list(range(expected_per_mode))
            ):
                raise RuntimeError("foundation development outline counts are invalid")
            for record in mode_records:
                provenance = {
                    "panel_contract_sha256": contract_sha256,
                    "outline_mode": mode,
                    "sample_index": record["sample_index"],
                    "sample_manifest_sha256": record["sample_manifest_sha256"],
                }
                if (
                    record.get("panel_contract_sha256") != contract_sha256
                    or int(record.get("views", -1))
                    != int(panel["fresh_checkpoint_views"])
                    or record.get("record_provenance_sha256")
                    != _canonical_sha256(provenance)
                ):
                    raise RuntimeError("foundation development raw provenance is invalid")
        metric_payload = {
            name: panel["overall"][name] for name in metrics
        }
        metric_payload.update(
            {
                f"{mode}:{name}": by_mode[mode][name]
                for mode in independent_data.OUTLINE_MODE_NAMES
                for name in metrics
            }
        )
        if nonfinite == 0 and (
            not _payload_is_finite(metric_payload) or not _payload_is_finite(raw)
        ):
            raise RuntimeError("foundation development panel hid a nonfinite output")


def _reduction(baseline, current) -> float | None:
    baseline, current = float(baseline), float(current)
    if not math.isfinite(baseline) or not math.isfinite(current) or baseline <= 0.0:
        return None
    return (baseline - current) / baseline


def _gradient_summary(records: list[dict], warmup_views: int) -> dict:
    postwarm = [record for record in records if record["views_after"] > warmup_views]
    factors = [
        min(record["encoder"]["clip_factor"], record["head"]["clip_factor"])
        for record in postwarm
    ]
    by_group = {}
    for group in ("encoder", "head"):
        values = [record[group]["clip_factor"] for record in postwarm]
        clipped = [
            bool(record[group].get("clipped", record[group]["clip_factor"] < 1.0))
            for record in postwarm
        ]
        by_group[group] = {
            "updates": len(values),
            "median_clip_factor": None if not values else float(np.median(values)),
            "clipped_fraction": None if not clipped else float(np.mean(clipped)),
        }
    return {
        "postwarm_update_count": len(postwarm),
        "median_clip_factor": None if not factors else float(np.median(factors)),
        "by_group": by_group,
    }


def qualification_status(
    evaluations: list[dict],
    gradient_records: list[dict],
    nonfinite_training_count: int,
    config: dict,
) -> dict:
    indexed = {int(panel["fresh_checkpoint_views"]): panel for panel in evaluations}
    views = max(indexed) if indexed else 0
    baseline = indexed.get(0)
    gates = config["gates"]
    checks = {}

    def add(name, observed, threshold, passed):
        checks[name] = {
            "observed": observed, "threshold": threshold, "passed": bool(passed)
        }

    panel_nonfinite = sum(int(panel["nonfinite_output_count"]) for panel in evaluations)
    total_nonfinite = int(nonfinite_training_count) + panel_nonfinite
    add(
        "nonfinite_count",
        total_nonfinite,
        gates["nonfinite_count_maximum"],
        total_nonfinite <= gates["nonfinite_count_maximum"],
    )
    interim_views = int(gates["interim_views"])
    interim = indexed.get(interim_views)
    if views >= interim_views:
        observed = None
        if baseline is not None and interim is not None:
            observed = _reduction(
                baseline["overall"]["physical_corresponding_plane_error_um"],
                interim["overall"]["physical_corresponding_plane_error_um"],
            )
        add(
            "interim_overall_physical_reduction",
            observed,
            gates["interim_overall_physical_reduction_minimum"],
            observed is not None
            and observed >= gates["interim_overall_physical_reduction_minimum"],
        )
    final_views = int(gates["final_views"])
    final = indexed.get(final_views)
    if views >= final_views:
        for name, key, threshold in (
            ("final_overall_physical_reduction", "physical_corresponding_plane_error_um", gates["final_overall_physical_reduction_minimum"]),
            ("final_ap_reduction", "ap_mae_um", gates["final_each_axis_reduction_minimum"]),
            ("final_lr_reduction", "lr_mae_deg", gates["final_each_axis_reduction_minimum"]),
            ("final_dv_reduction", "dv_mae_deg", gates["final_each_axis_reduction_minimum"]),
        ):
            observed = None
            if baseline is not None and final is not None:
                observed = _reduction(baseline["overall"][key], final["overall"][key])
            add(name, observed, threshold, observed is not None and observed >= threshold)
        observed = None
        if baseline is not None and final is not None:
            observed = _reduction(
                baseline["by_outline_mode"]["absent"]["physical_corresponding_plane_error_um"],
                final["by_outline_mode"]["absent"]["physical_corresponding_plane_error_um"],
            )
        add(
            "final_absent_physical_reduction",
            observed,
            gates["final_absent_physical_reduction_minimum"],
            observed is not None and observed >= gates["final_absent_physical_reduction_minimum"],
        )
        summary = _gradient_summary(gradient_records, int(config["training"]["warmup_views"]))
        median = summary["median_clip_factor"]
        add(
            "postwarm_median_clip_factor",
            median,
            gates["postwarm_median_clip_factor_minimum"],
            median is not None and median >= gates["postwarm_median_clip_factor_minimum"],
        )
        for group in ("encoder", "head"):
            fraction = summary["by_group"][group]["clipped_fraction"]
            add(
                f"{group}_clipped_fraction",
                fraction,
                gates["each_group_clipped_fraction_maximum"],
                fraction is not None
                and fraction <= gates["each_group_clipped_fraction_maximum"],
            )
    if total_nonfinite:
        decision = "stop"
    elif views >= interim_views and not checks["interim_overall_physical_reduction"]["passed"]:
        decision = "stop"
    elif views >= final_views:
        decision = "go" if all(item["passed"] for item in checks.values()) else "stop"
    elif views >= interim_views:
        decision = "continue"
    else:
        decision = "pending"
    return {
        "views": views,
        "decision": decision,
        "checks": checks,
        "gradient_summary": _gradient_summary(
            gradient_records, int(config["training"]["warmup_views"])
        ),
    }


def _initializer_parameter_groups(model: IndependentJointModel):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder = list(model.pyramid.slice_stem.parameters()) + list(model.pyramid.levels.parameters())
    head = list(model.pose_head.parameters())
    for parameter in encoder + head:
        parameter.requires_grad_(True)
    encoder_ids = {id(value) for value in encoder}
    head_ids = {id(value) for value in head}
    if encoder_ids.intersection(head_ids):
        raise RuntimeError("initializer encoder and head parameter groups overlap")
    trainable = {id(value) for value in model.parameters() if value.requires_grad}
    if trainable != encoder_ids | head_ids:
        raise RuntimeError("initializer-only trainable parameter boundary changed")
    return encoder, head


def _parameter_grad_norm(parameters) -> float:
    values = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    return 0.0 if not values else float(torch.stack(values).norm())


def _learning_rate(update: int, warmup_updates: int, base: float) -> float:
    return base if update >= warmup_updates else base * (update + 1) / warmup_updates


def _gaussian_weight(update: int, total_updates: int, start: float, end: float) -> float:
    progress = update / max(total_updates - 1, 1)
    return start + (end - start) * progress


def _validate_optimizer_contract(optimizer, encoder, head, config: dict) -> None:
    if not isinstance(optimizer, torch.optim.AdamW) or len(optimizer.param_groups) != 2:
        raise RuntimeError("foundation optimizer structure changed")
    expected_parameters = (encoder, head)
    expected_weight_decay = float(config["training"]["weight_decay"])
    for group, parameters in zip(optimizer.param_groups, expected_parameters):
        if (
            [id(value) for value in group["params"]] != [id(value) for value in parameters]
            or float(group["weight_decay"]) != expected_weight_decay
            or tuple(group["betas"]) != (0.9, 0.999)
            or float(group["eps"]) != 1e-8
            or bool(group["amsgrad"])
            or bool(group["maximize"])
        ):
            raise RuntimeError("foundation optimizer hyperparameters changed")


def _evaluate_ema(model, ema, evaluator, views):
    raw = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(ema)
    model.eval()
    try:
        with torch.no_grad():
            return evaluator(model, views)
    finally:
        model.load_state_dict(raw)
        model.train()


def _resolve_paths(config: dict) -> tuple[Path, Path]:
    atlas = (REPOSITORY_ROOT / config["paths"]["atlas_repo_relative"]).resolve()
    variable = config["paths"]["run_root_env"]
    if not os.environ.get(variable):
        raise RuntimeError(f"set {variable} before running the qualification")
    return atlas, Path(os.environ[variable]).resolve()


def run_initializer_foundation(
    config_path: str | Path,
    *,
    max_updates_this_call: int | None = None,
) -> dict:
    config = load_foundation_config(config_path)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_name = config["device"]
    device = "cuda" if device_name == "auto" and torch.cuda.is_available() else (
        "cpu" if device_name == "auto" else device_name
    )
    training = config["training"]
    amp_enabled = bool(training["amp"] and str(device).startswith("cuda"))
    if str(device).startswith("cuda"):
        gpu_index = torch.cuda.current_device() if torch.device(device).index is None else torch.device(device).index
        gpu_properties = torch.cuda.get_device_properties(gpu_index)
        gpu = {
            "index": gpu_index,
            "name": gpu_properties.name,
            "capability": [gpu_properties.major, gpu_properties.minor],
            "total_memory_bytes": gpu_properties.total_memory,
        }
    else:
        gpu = None
    execution_environment = {
        "requested_device": device_name,
        "resolved_device": str(torch.device(device)),
        "amp_requested": bool(training["amp"]),
        "amp_enabled": amp_enabled,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": gpu,
    }
    model = IndependentJointModel(**config["model"]["kwargs"]).to(device)
    if sum(value.numel() for value in model.parameters()) != int(config["model"]["expected_parameter_count"]):
        raise RuntimeError("foundation model parameter count changed")
    initial_state_sha256 = _state_sha256(model)
    encoder, head = _initializer_parameter_groups(model)
    atlas_folder, run_root = _resolve_paths(config)
    generator = SyntheticRegistrationGenerator(atlas_folder, device=device)
    synthetic = independent_data.IndependentSyntheticData(generator)
    data = config["data"]
    total_views = int(data["total_unique_views"])
    batch_size = int(data["batch_size"])
    total_updates = int(data["optimizer_updates"])
    outline_plan = independent_data._outline_plan(
        total_views, int(data["outline_seed"]), "initializer-foundation-global"
    )
    expected_outline_counts = np.asarray((875, 875, 750), np.int64)
    if not np.array_equal(outline_plan["mode_counts"], expected_outline_counts):
        raise RuntimeError("global foundation outline plan is not exactly 35/35/30")
    panel_contract, panel_batches, brain_mask = _development_setup(
        config, synthetic, generator
    )
    evaluator = _development_evaluator(
        panel_contract, panel_batches, brain_mask, torch.device(device)
    )

    output_folder = run_root / config["name"]
    latest_path = output_folder / "latest.pt"
    final_path = output_folder / "final.pt"
    receipt_path = output_folder / "qualification_receipt.json"
    source_hashes = {
        relative: _source_sha256(REPOSITORY_ROOT / relative) for relative in SOURCE_FILES
    }
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    frozen_names = [name for name, value in model.named_parameters() if not value.requires_grad]
    frozen_parameter_state_sha256 = _named_parameter_sha256(model, frozen_names)
    frozen_ema_names = sorted(set(model.state_dict()) - set(trainable_names))
    frozen_ema_initial_state_sha256 = _state_subset_sha256(
        model.state_dict(), frozen_ema_names
    )
    setup = {
        "version": 1,
        "purpose": PURPOSE,
        "role": "diagnostic-not-architecture-selection",
        "product5_access": False,
        "calibration_access": False,
        "final_test_access": False,
        "learned_checkpoint_dependencies": [],
        "execution_environment": execution_environment,
        "config": config,
        "source_sha256": source_hashes,
        "initial_state_sha256": initial_state_sha256,
        "atlas_contract": generator.contract,
        "synthetic_data_contract": synthetic.contract,
        "global_outline_plan_sha256": outline_plan["plan_sha256"],
        "global_outline_mode_counts": outline_plan["mode_counts"],
        "development_panel": panel_contract,
        "model_execution_path": "encode_source-plus-probabilistic-pose-head-only",
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "frozen_parameter_initial_state_sha256": frozen_parameter_state_sha256,
        "ema_trainable_parameter_names": trainable_names,
        "frozen_ema_state_names": frozen_ema_names,
        "frozen_ema_initial_state_sha256": frozen_ema_initial_state_sha256,
    }
    setup["setup_sha256"] = _canonical_sha256(setup)
    canonical_setup = _canonical(setup)
    previous_receipt = None
    if receipt_path.is_file():
        previous_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        previous_setup = {
            name: previous_receipt.get(name) for name in canonical_setup
        }
        if previous_setup != canonical_setup:
            raise RuntimeError("existing foundation receipt differs from the frozen setup")
        if not latest_path.is_file():
            raise RuntimeError("foundation receipt exists without its latest checkpoint")

    optimizer = torch.optim.AdamW(
        ({"params": encoder}, {"params": head}),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    _validate_optimizer_contract(optimizer, encoder, head, config)
    scaler = torch.amp.GradScaler(
        str(device).split(":")[0],
        enabled=amp_enabled,
        init_scale=float(training["amp_initial_scale"]),
    )
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    update = views = nonfinite_training_count = 0
    gradient_records: list[dict] = []
    view_receipts: list[dict] = []
    evaluations: list[dict] = []
    status = "running"

    if latest_path.is_file() and bool(training["resume"]):
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("format") != FORMAT
            or checkpoint.get("setup_sha256") != setup["setup_sha256"]
            or checkpoint.get("learned_checkpoint_dependencies") != []
        ):
            raise RuntimeError("foundation checkpoint lineage differs from this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        ema = {name: value.to(device) for name, value in checkpoint["ema"].items()}
        if _state_subset_sha256(ema, frozen_ema_names) != frozen_ema_initial_state_sha256:
            raise RuntimeError("foundation checkpoint changed a frozen EMA state entry")
        update = int(checkpoint["update"])
        views = int(checkpoint["views"])
        gradient_records = checkpoint["gradient_records"]
        view_receipts = checkpoint["view_receipts"]
        evaluations = checkpoint["evaluations"]
        nonfinite_training_count = int(checkpoint["nonfinite_training_count"])
        status = checkpoint["status"]
        if status not in {"running", "paused", "go", "stop"}:
            raise RuntimeError("foundation checkpoint status is invalid")
        if (
            update < 0
            or update > total_updates
            or views != update * batch_size
            or len(gradient_records) != update
            or len(view_receipts) != views
        ):
            raise RuntimeError("foundation checkpoint update/view counters disagree")
        _validate_gradient_records(gradient_records, config)
        _validate_optimizer_contract(optimizer, encoder, head, config)
        resume_learning_rate = (
            float(training["learning_rate"])
            if update == 0
            else float(gradient_records[-1]["learning_rate"])
        )
        for group in optimizer.param_groups:
            group["lr"] = resume_learning_rate
        for index, record in enumerate(view_receipts):
            recorded_sha256 = record.get("view_provenance_sha256")
            content_sha256 = _canonical_sha256(
                {
                    name: record.get(name)
                    for name in (
                        "manifest_sha256",
                        "generator_manifest_sha256",
                        "row",
                        "true_pose",
                        "source_view_rotation_deg",
                        "source_view_scale",
                        "outline_mode",
                        "outline_receipt_sha256",
                    )
                }
            )
            unhashed = {
                name: value
                for name, value in record.items()
                if name != "view_provenance_sha256"
            }
            if (
                int(record.get("view_index", -1)) != index
                or int(record.get("optimizer_update", -1)) != index // batch_size
                or int(record.get("row", -1)) != index % batch_size
                or record.get("view_content_sha256") != content_sha256
                or recorded_sha256 != _canonical_sha256(unhashed)
            ):
                raise RuntimeError("foundation checkpoint view sequence is invalid")
        identities = [record["view_content_sha256"] for record in view_receipts]
        if len(set(identities)) != len(identities):
            raise RuntimeError("foundation checkpoint contains repeated training views")
        _validate_evaluations(evaluations, views, panel_contract, config)
        if previous_receipt is not None:
            progress = previous_receipt.get("progress", {})
            receipt_update = int(progress.get("optimizer_updates", -1))
            receipt_views = int(progress.get("unique_views", -1))
            if receipt_update > update or receipt_views > views:
                raise RuntimeError("foundation receipt progress is ahead of its checkpoint")
            same_progress = receipt_update == update and receipt_views == views
            same_status = previous_receipt.get("status") == status
            if same_progress and same_status and previous_receipt.get(
                "latest_checkpoint_sha256"
            ) != _binary_sha256(latest_path):
                raise RuntimeError("foundation receipt checkpoint hash differs")
        _validate_rng_state_types(checkpoint["rng_state"])
        _set_rng_state(checkpoint["rng_state"])

    def save(current_status: str, qualification: dict) -> dict:
        _validate_gradient_records(gradient_records, config)
        _validate_evaluations(evaluations, views, panel_contract, config)
        current_frozen_sha256 = _named_parameter_sha256(model, frozen_names)
        if current_frozen_sha256 != frozen_parameter_state_sha256:
            raise RuntimeError("a frozen pair/recurrent/dense parameter changed")
        current_frozen_ema_sha256 = _state_subset_sha256(ema, frozen_ema_names)
        if current_frozen_ema_sha256 != frozen_ema_initial_state_sha256:
            raise RuntimeError("a frozen EMA state entry changed")
        payload = {
            "format": FORMAT,
            "setup_sha256": setup["setup_sha256"],
            "learned_checkpoint_dependencies": [],
            "model": model.state_dict(),
            "ema": ema,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": _rng_state(),
            "update": update,
            "views": views,
            "status": current_status,
            "qualification_model_state": "ema",
            "nonfinite_training_count": nonfinite_training_count,
            "gradient_records": gradient_records,
            "view_receipts": view_receipts,
            "evaluations": evaluations,
            "qualification": qualification,
            "frozen_parameter_state_sha256": current_frozen_sha256,
            "frozen_ema_state_sha256": current_frozen_ema_sha256,
        }
        _atomic_save(payload, latest_path)
        if current_status in {"go", "stop"}:
            _atomic_save(payload, final_path)
        receipt = {
            **setup,
            "status": current_status,
            "progress": {"optimizer_updates": update, "unique_views": views},
            "nonfinite_training_count": nonfinite_training_count,
            "gradient_records": gradient_records,
            "training_view_receipts": view_receipts,
            "development_evaluations": evaluations,
            "qualification": qualification,
            "latest_checkpoint": str(latest_path),
            "latest_checkpoint_sha256": _binary_sha256(latest_path),
            "final_checkpoint": str(final_path) if final_path.is_file() else None,
            "final_checkpoint_sha256": _binary_sha256(final_path) if final_path.is_file() else None,
        }
        _atomic_json(receipt, receipt_path)
        return receipt

    if status in {"go", "stop"}:
        resumed_qualification = qualification_status(
            evaluations, gradient_records, nonfinite_training_count, config
        )
        if resumed_qualification["decision"] != status:
            raise RuntimeError("terminal foundation checkpoint contradicts its gates")
        save(status, resumed_qualification)
        return {
            "receipt_path": receipt_path,
            "checkpoint_folder": output_folder,
            "status": status,
            "views": views,
            "qualification": resumed_qualification,
        }
    if not evaluations:
        baseline = _evaluate_ema(model, ema, evaluator, 0)
        evaluations.append(baseline)
        _atomic_json(baseline, output_folder / "development_views_0000.json")
        baseline_qualification = qualification_status(
            evaluations, gradient_records, 0, config
        )
        if baseline_qualification["decision"] == "stop":
            status = "stop"
            save(status, baseline_qualification)
            return {
                "receipt_path": receipt_path,
                "checkpoint_folder": output_folder,
                "status": status,
                "views": views,
                "qualification": baseline_qualification,
            }
        save("running", baseline_qualification)

    call_stop = total_updates
    if max_updates_this_call is not None:
        if int(max_updates_this_call) < 0:
            raise ValueError("max_updates_this_call cannot be negative")
        call_stop = min(total_updates, update + int(max_updates_this_call))
    evaluation_views = set(int(value) for value in config["development"]["evaluation_views"])
    warmup_updates = int(training["warmup_views"]) // batch_size
    while update < call_stop:
        manifest = training_manifest(synthetic, config, update, outline_plan)
        batch = synthetic.batch(manifest)
        if batch["source_type"] != "synthetic_ccf" or batch["data_split"] != "train" or batch["pose_regime"] != "standard":
            raise RuntimeError("foundation training opened a nonstandard or nontraining stream")
        receipts = _view_receipts(manifest, update)
        known = {record["view_content_sha256"] for record in view_receipts}
        if any(record["view_content_sha256"] in known for record in receipts):
            raise RuntimeError("foundation attempted to reuse a training view")
        learning_rate = _learning_rate(
            update, warmup_updates, float(training["learning_rate"])
        )
        gaussian_weight = _gaussian_weight(
            update,
            total_updates,
            float(training["gaussian_nll_weight_start"]),
            float(training["gaussian_nll_weight_end"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with torch.amp.autocast(device_type=str(device).split(":")[0], enabled=amp_enabled):
            initialization = model.initialize(
                batch["source_image"], batch["source_mask"], batch["mask_available"]
            )
            components = initializer_pose_losses(
                initialization, batch["true_pose"], model
            )
            loss = (
                float(training["loss_weights"]["initializer_categorical"])
                * components["initializer_categorical"]
                + float(training["loss_weights"]["initializer_sub_bin"])
                * components["initializer_sub_bin"]
                + gaussian_weight * components["initializer_gaussian_nll"]
                + float(training["loss_weights"]["initializer_plane_anchor"])
                * components["initializer_plane_anchor"]
            )
        output_nonfinite = sum(
            int((~torch.isfinite(value)).sum()) for value in initialization.values()
        )
        if not bool(torch.isfinite(loss)) or output_nonfinite:
            nonfinite_training_count += max(output_nonfinite, 1)
            status = "stop"
            qualification = qualification_status(
                evaluations, gradient_records, nonfinite_training_count, config
            )
            save(status, qualification)
            break
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        encoder_pre = float(torch.nn.utils.clip_grad_norm_(
            encoder, float(training["encoder_gradient_clip_norm"])
        ))
        head_pre = float(torch.nn.utils.clip_grad_norm_(
            head, float(training["head_gradient_clip_norm"])
        ))
        encoder_post = _parameter_grad_norm(encoder)
        head_post = _parameter_grad_norm(head)
        if not all(math.isfinite(value) for value in (encoder_pre, head_pre, encoder_post, head_post)):
            nonfinite_training_count += 1
            status = "stop"
            qualification = qualification_status(
                evaluations, gradient_records, nonfinite_training_count, config
            )
            save(status, qualification)
            break
        scaler.step(optimizer)
        scaler.update()
        _update_initializer_ema(
            ema, model, float(training["ema_decay"]), trainable_names
        )
        encoder_clipped = encoder_pre > float(training["encoder_gradient_clip_norm"])
        head_clipped = head_pre > float(training["head_gradient_clip_norm"])
        encoder_factor = (
            min(1.0, encoder_post / encoder_pre) if encoder_clipped else 1.0
        )
        head_factor = min(1.0, head_post / head_pre) if head_clipped else 1.0
        update += 1
        views += batch_size
        view_receipts.extend(receipts)
        gradient_records.append(
            {
                "update": update,
                "views_after": views,
                "learning_rate": learning_rate,
                "gaussian_nll_weight": gaussian_weight,
                "loss": float(loss.detach()),
                "encoder": {
                    "preclip_norm": encoder_pre,
                    "postclip_norm": encoder_post,
                    "clip_factor": encoder_factor,
                    "clipped": encoder_clipped,
                },
                "head": {
                    "preclip_norm": head_pre,
                    "postclip_norm": head_post,
                    "clip_factor": head_factor,
                    "clipped": head_clipped,
                },
            }
        )
        if views in evaluation_views:
            panel = _evaluate_ema(model, ema, evaluator, views)
            evaluations.append(panel)
            _atomic_json(panel, output_folder / f"development_views_{views:04d}.json")
            qualification = qualification_status(
                evaluations, gradient_records, nonfinite_training_count, config
            )
            if qualification["decision"] == "stop":
                status = "stop"
                save(status, qualification)
                break
        checkpoint_every = int(training["checkpoint_every_views"])
        if views % checkpoint_every == 0 or views in evaluation_views:
            save(
                "running",
                qualification_status(
                    evaluations, gradient_records, nonfinite_training_count, config
                ),
            )

    if status != "stop":
        qualification = qualification_status(
            evaluations, gradient_records, nonfinite_training_count, config
        )
        if update == total_updates:
            status = qualification["decision"]
            if status not in {"go", "stop"}:
                raise RuntimeError("completed foundation run lacks its final fixed panel")
        else:
            status = "paused"
        save(status, qualification)
    return {
        "receipt_path": receipt_path,
        "checkpoint_folder": output_folder,
        "status": status,
        "views": views,
        "qualification": qualification,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m training.run_independent_initializer_foundation CONFIG.json"
        )
    print(
        json.dumps(
            _canonical(run_initializer_foundation(sys.argv[1])),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
