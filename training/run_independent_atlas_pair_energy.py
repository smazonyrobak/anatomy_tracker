"""Run the frozen oracle atlas-pair energy premise without protected data."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.independent_atlas_pair_energy import (
    MODEL_INPUT_SHAPE,
    AtlasPairEnergyModel,
    atlas_pair_loss,
    parameter_count,
)
from training.quicknii_plane_metric import (
    QUICKNII_PIXEL_GRID_SHAPE,
    QUICKNII_SHAPE_ML_AP_DV,
    torch_annotation_brain_mask,
    torch_brain_masked_plane_distance,
)
from training.synthetic_registration import (
    AP_MAX_UM,
    AP_MIN_UM,
    BREGMA_AP_INDEX,
    VOXEL_UM,
    SyntheticRegistrationGenerator,
    _payload_sha256,
    split_ap_indices,
)


ROOT = Path(__file__).parents[1]
PURPOSE = "development-only-oracle-atlas-pair-energy-premise"
FORMAT = "independent-atlas-pair-energy-state-v1"
SOURCE_FILES = (
    "training/independent_atlas_pair_energy.py",
    "training/run_independent_atlas_pair_energy.py",
    "training/synthetic_registration.py",
    "training/quicknii_plane_metric.py",
    "source/dense_registration_preprocessing.py",
)
_QUALIFICATION_CAPABILITY = object()
QUALIFICATION_SEEDS = (1204322, 1304322)
TRUTH_AP_MARGIN_UM = 500.0
TRUTH_TILT_ABS_MAX_DEG = 25.0
REALIZATION_DOMAIN = "atlas-pair-oracle-realization-v1"


def _canonical(value):
    if torch.is_tensor(value):
        return _canonical(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _canonical_sha256(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: _sha256(ROOT / name) for name in SOURCE_FILES}


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(value) + b"\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def inspect_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    commitment = raw.pop("contract_sha256", None)
    if commitment != _canonical_sha256(raw):
        raise ValueError("atlas-pair config commitment differs from its payload")
    required = {
        "schema_version": 1,
        "frozen": True,
        "name": "independent-oracle-atlas-pair-energy-1500-v1",
        "purpose": PURPOSE,
        "role": "causal-premise-not-model-selection",
        "product5_access": False,
        "calibration_access": False,
        "final_test_access": False,
        "learned_checkpoint_dependencies": [],
        "optimizer_seed": 1404322,
        "device": "auto",
        "paths": {
            "atlas_repo_relative": "data/Allen Brain Atlas 25um",
            "run_root_env": "ATLAS_JOINT_RUN_ROOT",
        },
    }
    if any(raw.get(name) != value for name, value in required.items()):
        raise ValueError("atlas-pair independence or purpose contract changed")
    if set(raw) != {
        *required,
        "model",
        "data",
        "training",
        "search",
        "gates",
        "lineage",
    }:
        raise ValueError("atlas-pair config fields changed")
    if raw.get("model") != {
        "class": "training.independent_atlas_pair_energy.AtlasPairEnergyModel",
        "expected_parameter_count": 271450,
        "maximum_parameter_count": 1500000,
        "input_shape": [160, 232],
        "correlation_levels": [8, 16],
        "correlation_radius": 4,
        "candidate_chunk_size": 8,
        "candidate_pose_input": False,
    }:
        raise ValueError("atlas-pair model contract changed")
    data = raw.get("data", {})
    if data != {
        "source": "synthetic_ccf",
        "split": "train",
        "stratum": "clean",
        "oracle_source": "moving_raw_uint8_pre_source_view",
        "source_view_rotation_deg": 0.0,
        "source_view_scale": 1.0,
        "source_resampling_count_before_fixed_downsample": 0,
        "outline_mode": "absent",
        "train_seed": 1004322,
        "train_count": 2048,
        "development_seed": 1104322,
        "development_count": 48,
        "qualification_seeds": [1204322, 1304322],
        "qualification_count_per_seed": 48,
        "qualification_generation": "only-after-final-source-config-checkpoint-freeze",
        "panel_balance": "six-interior-ap-strata-times-eight-signed-tilt-configurations",
        "premise_truth_ap_margin_um": 500.0,
        "premise_truth_tilt_abs_max_deg": 25.0,
        "truth_support_role": "interior-only-causal-premise-not-outer-domain-evidence",
    }:
        raise ValueError("atlas-pair data or untouched-seed contract changed")
    training = raw.get("training", {})
    if training != {
        "optimizer": "AdamW",
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "batch_size": 2,
        "max_updates": 1500,
        "amp": True,
        "amp_initial_scale": 512.0,
        "gradient_clip_norm": 5.0,
        "resume_every_updates": 25,
        "development_updates": [500, 1000, 1500],
        "candidate_count": 16,
        "loss": "ranking+0.25*two-scale-ranking+0.25*posterior-point",
    }:
        raise ValueError("atlas-pair training budget changed")
    search = raw.get("search", {})
    if search != {
        "coarse_shape": [9, 5, 5],
        "top_k": 3,
        "refinement_rounds": 3,
        "neighborhood_shape": [3, 3, 3],
        "maximum_candidate_evaluations_per_slice": 468,
        "continuous_refinement": False,
    }:
        raise ValueError("atlas-pair search contract changed")
    gates = raw.get("gates", {})
    if gates != {
        "nonfinite_count_maximum": 0,
        "invalid_render_count_maximum": 0,
        "truth_in_set_correct_minimum_per_48": 46,
        "ap_mae_um_maximum": 250.0,
        "lr_mae_deg_maximum": 3.0,
        "dv_mae_deg_maximum": 3.0,
        "physical_improvement_over_constant_prior_minimum": 0.50,
        "broken_pair_correct_maximum_per_48": 12,
        "order_energy_atol": 1e-6,
        "order_energy_rtol": 1e-6,
        "ten_slice_p95_seconds_maximum": 180.0,
    }:
        raise ValueError("atlas-pair gates changed")
    if raw.get("lineage", {}).get("source_sha256") != source_hashes():
        raise ValueError("atlas-pair source lineage changed")
    raw["contract_sha256"] = commitment
    raw["config_file_sha256"] = _sha256(config_path)
    raw["config_path"] = str(config_path)
    return raw


def _take_manifest(manifest: dict, indices: np.ndarray) -> dict:
    count = len(manifest["ap_index"])
    result = {}
    for name, value in manifest.items():
        if isinstance(value, np.ndarray) and value.ndim and len(value) == count:
            result[name] = value[indices]
        elif name != "manifest_sha256":
            result[name] = value
    result["manifest_sha256"] = _payload_sha256(result)
    return result


def _commit_manifest(manifest: dict) -> dict:
    result = {name: value for name, value in manifest.items() if name != "manifest_sha256"}
    result["manifest_sha256"] = _payload_sha256(result)
    return result


def _premise_ap_pool() -> tuple[np.ndarray, np.ndarray]:
    pool = np.asarray(split_ap_indices("train"), np.float32)
    ap_um = (BREGMA_AP_INDEX - pool) * VOXEL_UM
    keep = (ap_um >= AP_MIN_UM + TRUTH_AP_MARGIN_UM) & (
        ap_um <= AP_MAX_UM - TRUTH_AP_MARGIN_UM
    )
    return pool[keep], ap_um[keep]


def training_manifest(generator: SyntheticRegistrationGenerator, config: dict) -> dict:
    settings = config["data"]
    count = int(settings["train_count"])
    seed = int(settings["train_seed"])
    manifest = generator.make_manifest(count, "train", seed, "clean")
    rng = np.random.default_rng(seed + 17)
    pool, _ = _premise_ap_pool()
    repeats = math.ceil(count / len(pool))
    manifest["ap_index"] = np.concatenate(
        [rng.permutation(pool) for _ in range(repeats)]
    )[:count].astype(np.float32)
    lr = np.linspace(
        -TRUTH_TILT_ABS_MAX_DEG,
        TRUTH_TILT_ABS_MAX_DEG,
        count,
        endpoint=False,
        dtype=np.float32,
    ) + TRUTH_TILT_ABS_MAX_DEG / count
    dv = lr.copy()
    manifest["tilt_lr_deg"] = lr[rng.permutation(count)]
    manifest["tilt_dv_deg"] = dv[rng.permutation(count)]
    manifest["ap_um"] = ((BREGMA_AP_INDEX - manifest["ap_index"]) * VOXEL_UM).astype(np.float32)
    return _commit_manifest(manifest)


def balanced_panel_manifest(
    generator: SyntheticRegistrationGenerator,
    seed: int,
    count: int = 48,
    *,
    qualification_capability=None,
) -> dict:
    if count != 48:
        raise ValueError("frozen atlas-pair panels contain exactly 48 sources")
    if seed in QUALIFICATION_SEEDS and (
        not isinstance(qualification_capability, dict)
        or qualification_capability.get("guard") is not _QUALIFICATION_CAPABILITY
        or seed not in qualification_capability.get("qualification_seeds", ())
        or count != qualification_capability.get("qualification_count_per_seed")
        or not qualification_capability.get("checkpoint_file_sha256")
        or not qualification_capability.get("freeze_payload_sha256")
        or not qualification_capability.get("data_lineage_sha256")
    ):
        raise RuntimeError("qualification manifest generation is forbidden before final freeze")
    manifest = generator.make_manifest(count, "train", seed, "clean")
    pool, ap_um = _premise_ap_pool()
    order = np.argsort(ap_um)
    strata = np.array_split(order, 6)
    rng = np.random.default_rng(seed + 31)
    chosen = [int(rng.choice(value)) for value in strata]
    tilts = np.asarray(
        [
            (-13.25, -18.25), (-13.25, 18.25), (13.25, -18.25), (13.25, 18.25),
            (-13.25, 0.0), (13.25, 0.0), (0.0, -18.25), (0.0, 18.25),
        ],
        np.float32,
    )
    manifest["ap_index"] = np.repeat(pool[chosen], 8).astype(np.float32)
    manifest["ap_um"] = np.repeat(ap_um[chosen], 8).astype(np.float32)
    manifest["tilt_lr_deg"] = np.tile(tilts[:, 0], 6)
    manifest["tilt_dv_deg"] = np.tile(tilts[:, 1], 6)
    return _commit_manifest(manifest)


def oracle_source(pair: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "moving_raw_uint8" not in pair:
        raise RuntimeError("oracle source requires generator QA pre-view tensor")
    source = pair["moving_raw_uint8"].float() / 255.0
    source = F.interpolate(source, MODEL_INPUT_SHAPE, mode="bilinear", align_corners=False)
    mask = torch.zeros_like(source, dtype=torch.bool)
    available = torch.zeros(len(source), 1, 1, 1, device=source.device)
    return source, mask, available


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def realization_manifest(
    parent: dict, sample_index: int, generator_contract_sha256: str
) -> tuple[dict, dict]:
    parent_payload = {
        name: value for name, value in parent.items() if name != "manifest_sha256"
    }
    if parent.get("manifest_sha256") != _payload_sha256(parent_payload):
        raise RuntimeError("parent synthetic manifest commitment differs from its payload")
    if parent.get("contract_sha256") != generator_contract_sha256:
        raise RuntimeError("parent synthetic manifest belongs to a different generator")
    if not 0 <= sample_index < len(parent["ap_index"]):
        raise IndexError("synthetic realization index is outside the parent manifest")
    payload = {
        "domain": REALIZATION_DOMAIN,
        "generator_contract_sha256": generator_contract_sha256,
        "parent_manifest_sha256": parent["manifest_sha256"],
        "sample_index": int(sample_index),
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).digest()
    realization_seed = int.from_bytes(digest[:8], "little")
    realization_id = hashlib.sha256(b"synthetic-realization-id\0" + digest).hexdigest()
    child = _take_manifest(parent, np.asarray([sample_index]))
    child["parent_seed"] = int(parent["seed"])
    child["seed"] = realization_seed
    child["parent_manifest_sha256"] = parent["manifest_sha256"]
    child["realization_index"] = int(sample_index)
    child["realization_seed"] = realization_seed
    child["synthetic_realization_id"] = realization_id
    child = _commit_manifest(child)
    return child, {
        "animal_id": -1,
        "specimen_id": -1,
        "synthetic_realization_id": realization_id,
        "realization_index": int(sample_index),
        "realization_seed": realization_seed,
        "parent_seed": int(parent["seed"]),
        "parent_manifest_sha256": parent["manifest_sha256"],
        "realization_manifest_sha256": child["manifest_sha256"],
        "generator_contract_sha256": generator_contract_sha256,
    }


def oracle_realizations(
    generator: SyntheticRegistrationGenerator,
    manifest: dict,
    indices: np.ndarray,
    cache: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    sources, masks, availability, records = [], [], [], []
    contract_sha256 = generator.contract["contract_sha256"]
    for sample_index in map(int, indices):
        child, record = realization_manifest(manifest, sample_index, contract_sha256)
        cache_key = record["synthetic_realization_id"]
        cached = None if cache is None else cache.get(cache_key)
        if cached is None:
            pair = generator.batch(child, qa=True)
            source, mask, available = oracle_source(pair)
            record["moving_raw_uint8_sha256"] = _tensor_sha256(
                pair["moving_raw_uint8"]
            )
            record["source_160x232_sha256"] = _tensor_sha256(source)
            cached = (source.cpu(), mask.cpu(), available.cpu(), dict(record))
            if cache is not None:
                cache[cache_key] = cached
        source_cpu, mask_cpu, available_cpu, record = cached
        sources.append(source_cpu.to(generator.device))
        masks.append(mask_cpu.to(generator.device))
        availability.append(available_cpu.to(generator.device))
        records.append(dict(record))
    return (
        torch.cat(sources),
        torch.cat(masks),
        torch.cat(availability),
        records,
    )


def _pose_from_manifest(manifest: dict, device) -> torch.Tensor:
    return torch.as_tensor(
        np.column_stack(
            (manifest["ap_um"], manifest["tilt_lr_deg"], manifest["tilt_dv_deg"])
        ),
        device=device,
        dtype=torch.float32,
    )


def _pose_valid(pose: np.ndarray) -> bool:
    return bool(
        AP_MIN_UM <= pose[0] <= AP_MAX_UM
        and -35.0 <= pose[1] <= 35.0
        and -35.0 <= pose[2] <= 35.0
    )


def _global_negative(rng: np.random.Generator, truth: np.ndarray, used: set) -> np.ndarray:
    for _ in range(1000):
        pose = np.asarray(
            [rng.uniform(AP_MIN_UM, AP_MAX_UM), rng.uniform(-35.0, 35.0), rng.uniform(-35.0, 35.0)],
            np.float32,
        )
        key = tuple(np.round(pose, 4))
        separation = np.abs((pose - truth) / np.asarray((500.0, 10.0, 10.0)))
        if key not in used and bool(np.all(separation >= 1.0)):
            used.add(key)
            return pose
    raise RuntimeError("could not draw a distinct global pose negative")


def candidate_pose_table(
    truth_pose: torch.Tensor, seed: int, sample_indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, list[list[str]]]:
    offsets = np.asarray(
        [
            (-125.0, 0.0, 0.0), (125.0, 0.0, 0.0), (-500.0, 0.0, 0.0), (500.0, 0.0, 0.0),
            (0.0, -2.5, 0.0), (0.0, 2.5, 0.0), (0.0, -10.0, 0.0), (0.0, 10.0, 0.0),
            (0.0, 0.0, -2.5), (0.0, 0.0, 2.5), (0.0, 0.0, -10.0), (0.0, 0.0, 10.0),
        ],
        np.float32,
    )
    rows, targets, kinds = [], [], []
    for truth_tensor, sample_index in zip(truth_pose.detach().cpu(), sample_indices):
        truth = truth_tensor.numpy().astype(np.float32)
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(f"atlas-pair-candidates:{seed}:{int(sample_index)}".encode()).digest()[:8], "little")
        )
        used = {tuple(np.round(truth, 4))}
        values, names = [truth], ["truth"]
        for offset in offsets:
            pose = truth + offset
            key = tuple(np.round(pose, 4))
            if not _pose_valid(pose) or key in used:
                raise RuntimeError("truth pose violates the frozen exact-axis candidate support")
            used.add(key)
            values.append(pose)
            names.append("axis")
        for _ in range(3):
            values.append(_global_negative(rng, truth, used))
            names.append("joint-global")
        permutation = rng.permutation(16)
        rows.append(np.asarray(values, np.float32)[permutation])
        names = [names[int(value)] for value in permutation]
        kinds.append(names)
        targets.append(int(np.flatnonzero(permutation == 0)[0]))
    return (
        torch.as_tensor(np.stack(rows), device=truth_pose.device),
        torch.as_tensor(targets, device=truth_pose.device, dtype=torch.long),
        kinds,
    )


def render_candidate_poses(
    renderer: SyntheticRegistrationGenerator, candidate_pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, candidates = candidate_pose.shape[:2]
    flat = candidate_pose.flatten(0, 1).to(renderer.device)
    image, mask, _ = renderer.render_planes(
        BREGMA_AP_INDEX - flat[:, 0] / VOXEL_UM, flat[:, 1], flat[:, 2]
    )
    image = F.interpolate(image, MODEL_INPUT_SHAPE, mode="bilinear", align_corners=False)
    mask = F.interpolate(mask.float(), MODEL_INPUT_SHAPE, mode="nearest") > 0.5
    return image.reshape(batch, candidates, 1, *MODEL_INPUT_SHAPE), mask.reshape(batch, candidates, 1, *MODEL_INPUT_SHAPE)


def training_indices(update: int, count: int, batch_size: int, seed: int) -> np.ndarray:
    offset = update * batch_size
    epoch, start = divmod(offset, count)
    first = np.random.default_rng(seed + epoch).permutation(count)
    if start + batch_size <= count:
        return first[start : start + batch_size]
    second = np.random.default_rng(seed + epoch + 1).permutation(count)
    return np.concatenate((first[start:], second[: batch_size - (count - start)]))


def data_lineage(
    generator: SyntheticRegistrationGenerator,
    train_manifest: dict,
    development_manifest: dict,
    config: dict,
) -> dict:
    lineage = {
        "generator_contract_sha256": generator.contract["contract_sha256"],
        "generator_contract": dict(generator.contract),
        "atlas_sha256": {
            name: generator.contract[name]
            for name in (
                "average_template_sha256",
                "annotation_sha256",
                "query_sha256",
            )
        },
        "train_manifest_sha256": train_manifest["manifest_sha256"],
        "development_manifest_sha256": development_manifest["manifest_sha256"],
        "train_seed": config["data"]["train_seed"],
        "development_seed": config["data"]["development_seed"],
        "reserved_qualification_seeds": config["data"]["qualification_seeds"],
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "oracle_source_contract": {
            "tensor": "moving_raw_uint8.float()/255 before source-view transform",
            "source_view_rotation_deg": 0.0,
            "source_view_scale": 1.0,
            "pre_downsample_resampling_count": 0,
            "fixed_downsample": "bilinear-align_corners-false-160x232",
            "source_mask": "all-zero",
            "mask_available": "all-zero",
        },
        "candidate_pose_is_scorer_input": False,
        "truth_support_role": config["data"]["truth_support_role"],
        "source_realization_contract": {
            "domain": REALIZATION_DOMAIN,
            "unit": "one committed child manifest and one generator batch per source",
            "seed_inputs": [
                "generator_contract_sha256",
                "parent_manifest_sha256",
                "sample_index",
            ],
            "batch_position_invariant": True,
            "animal_id_sentinel": -1,
            "specimen_id_sentinel": -1,
        },
        "learned_checkpoint_dependencies": [],
    }
    lineage["data_lineage_sha256"] = _canonical_sha256(lineage)
    return lineage


def write_manifest_receipt(
    path: Path,
    role: str,
    manifest: dict,
    generator: SyntheticRegistrationGenerator,
    config: dict,
    post_freeze_binding: dict | None = None,
) -> Path:
    payload = {
        "format": "independent-atlas-pair-synthetic-manifest-v1",
        "purpose": PURPOSE,
        "role": role,
        "source_sha256": source_hashes(),
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "generator_contract_sha256": generator.contract["contract_sha256"],
        "manifest": manifest,
    }
    if post_freeze_binding is not None:
        payload["post_freeze_binding"] = post_freeze_binding
    payload["manifest_file_payload_sha256"] = _canonical_sha256(payload)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != _canonical(payload):
            raise RuntimeError("existing synthetic manifest receipt differs")
    else:
        _atomic_json(path, payload)
    return path


def write_training_trajectory(
    path: Path,
    history: list[dict],
    checkpoint_path: Path,
    config: dict,
) -> Path:
    payload = {
        "format": "independent-atlas-pair-training-trajectory-v1",
        "purpose": PURPOSE,
        "source_sha256": source_hashes(),
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "checkpoint_file_sha256": _sha256(checkpoint_path),
        "applied_optimizer_updates": len(history),
        "training_history_sha256": _canonical_sha256(history),
        "training_history": history,
    }
    payload["trajectory_file_payload_sha256"] = _canonical_sha256(payload)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != _canonical(payload):
            raise RuntimeError("existing training trajectory differs")
    else:
        _atomic_json(path, payload)
    return path


def _score_pose_set(
    model: AtlasPairEnergyModel,
    renderer: SyntheticRegistrationGenerator,
    source_features,
    poses: torch.Tensor,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    pieces = {name: [] for name in ("energy", "energy8", "energy16")}
    invalid = []
    for start in range(0, len(poses), chunk_size):
        part = poses[start : start + chunk_size]
        image, mask = render_candidate_poses(renderer, part[None])
        output = model.score_encoded(source_features, image[0], mask[0])
        for name in pieces:
            pieces[name].append(output[name])
        finite_image = torch.isfinite(image[0]).flatten(1).all(1)
        empty_mask = ~mask[0].flatten(1).any(1)
        invalid.append(~finite_image | empty_mask)
    return {
        **{name: torch.cat(value) for name, value in pieces.items()},
        "invalid_render": torch.cat(invalid),
    }


def coarse_to_fine_search(
    model: AtlasPairEnergyModel,
    renderer: SyntheticRegistrationGenerator,
    source_image: torch.Tensor,
    source_mask: torch.Tensor,
    available: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict]:
    if len(source_image) != 1:
        raise ValueError("coarse-to-fine search processes one source at a time")
    source_features = model.encode_source(source_image, source_mask, available)
    ap = torch.linspace(AP_MIN_UM, AP_MAX_UM, 9, device=source_image.device)
    tilt = torch.linspace(-35.0, 35.0, 5, device=source_image.device)
    poses = torch.cartesian_prod(ap, tilt, tilt)
    spacings = (625.0, 17.5, 17.5)
    evaluations = 0
    stages = []
    nonfinite = 0
    invalid = 0
    for stage in range(4):
        output = _score_pose_set(
            model, renderer, source_features, poses, config["model"]["candidate_chunk_size"]
        )
        energy = output["energy"]
        evaluations += len(poses)
        stage_nonfinite = sum(
            int((~torch.isfinite(output[name])).sum())
            for name in ("energy", "energy8", "energy16")
        )
        stage_invalid = int(output["invalid_render"].sum())
        nonfinite += stage_nonfinite
        invalid += stage_invalid
        stages.append(
            {
                "stage": stage,
                "lattice_spacing": spacings,
                "candidate_pose": poses,
                "energy": output["energy"],
                "energy8": output["energy8"],
                "energy16": output["energy16"],
                "invalid_render": output["invalid_render"],
                "nonfinite_count": stage_nonfinite,
                "invalid_render_count": stage_invalid,
            }
        )
        selection_energy = torch.nan_to_num(
            energy, nan=torch.inf, posinf=torch.inf, neginf=torch.inf
        ).masked_fill(output["invalid_render"], torch.inf)
        if stage == 3:
            selected = int(selection_energy.argmin())
            if evaluations > config["search"]["maximum_candidate_evaluations_per_slice"]:
                raise RuntimeError("coarse-to-fine search exceeded its frozen evaluation budget")
            return poses[selected], {
                "candidate_evaluations": evaluations,
                "selected_pose": poses[selected],
                "minimum_energy": energy[selected],
                "minimum_energy8": output["energy8"][selected],
                "minimum_energy16": output["energy16"][selected],
                "nonfinite_count": nonfinite,
                "invalid_render_count": invalid,
                "stages": stages,
            }
        centers = poses[
            selection_energy.topk(config["search"]["top_k"], largest=False).indices
        ]
        offsets = torch.cartesian_prod(
            source_image.new_tensor((-spacings[0] / 2, 0.0, spacings[0] / 2)),
            source_image.new_tensor((-spacings[1] / 2, 0.0, spacings[1] / 2)),
            source_image.new_tensor((-spacings[2] / 2, 0.0, spacings[2] / 2)),
        )
        proposals = (centers[:, None] + offsets[None]).reshape(-1, 3)
        valid = (
            (proposals[:, 0] >= AP_MIN_UM) & (proposals[:, 0] <= AP_MAX_UM)
            & (proposals[:, 1].abs() <= 35.0) & (proposals[:, 2].abs() <= 35.0)
        )
        poses = torch.unique(proposals[valid], dim=0)
        spacings = tuple(value / 2 for value in spacings)
    raise AssertionError("unreachable")


def _pose_to_quicknii_ouv(pose: torch.Tensor) -> torch.Tensor:
    pose = torch.as_tensor(pose)
    ap_um, tilt_lr_deg, tilt_dv_deg = pose.unbind(-1)
    slope_lr = torch.tan(torch.deg2rad(tilt_lr_deg))
    slope_dv = torch.tan(torch.deg2rad(tilt_dv_deg))
    ap_index = BREGMA_AP_INDEX - ap_um / VOXEL_UM
    ml_center = (QUICKNII_SHAPE_ML_AP_DV[0] - 1.0) / 2.0
    dv_center = (QUICKNII_SHAPE_ML_AP_DV[2] - 1.0) / 2.0
    origin_ap = ap_index - slope_lr * ml_center - slope_dv * dv_center
    zeros = torch.zeros_like(ap_um)
    return torch.stack(
        (
            zeros, zeros + QUICKNII_SHAPE_ML_AP_DV[1] - origin_ap,
            zeros + QUICKNII_SHAPE_ML_AP_DV[2], zeros + QUICKNII_SHAPE_ML_AP_DV[0],
            -QUICKNII_SHAPE_ML_AP_DV[0] * slope_lr, zeros, zeros,
            -QUICKNII_SHAPE_ML_AP_DV[2] * slope_dv, zeros - QUICKNII_SHAPE_ML_AP_DV[2],
        ),
        dim=-1,
    )


def _derangement(truth: torch.Tensor, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    identity = np.arange(len(truth))
    truth_numpy = truth.detach().cpu().numpy()
    for _ in range(1000):
        order = rng.permutation(len(truth))
        if np.all(order != identity) and np.all(np.any(truth_numpy[order] != truth_numpy, axis=1)):
            return order
    raise RuntimeError("could not construct a pose-changing source derangement")


def _order_equivariance(
    model: AtlasPairEnergyModel,
    batch: dict,
    output: dict,
    seed: int,
    chunk_size: int,
    atol: float,
    rtol: float,
) -> dict:
    candidates = batch["candidate_image"].shape[1]
    permutation = torch.as_tensor(
        np.random.default_rng(seed).permutation(candidates), device=batch["source_image"].device
    )
    permuted = model(
        batch["source_image"], batch["source_mask"], batch["mask_available"],
        batch["candidate_image"][:, permutation], batch["candidate_mask"][:, permutation],
        candidate_chunk_size=chunk_size,
    )
    inverse = permutation.argsort()
    differences = {
        name: float((permuted[name][:, inverse] - output[name]).abs().max())
        for name in ("energy", "energy8", "energy16")
    }
    allclose = all(
        torch.allclose(permuted[name][:, inverse], output[name], atol=atol, rtol=rtol)
        for name in ("energy", "energy8", "energy16")
    )
    original_pose = batch["candidate_pose"].gather(
        1, output["energy"].argmin(1)[:, None, None].expand(-1, 1, 3)
    )[:, 0]
    permuted_pose = batch["candidate_pose"][:, permutation].gather(
        1, permuted["energy"].argmin(1)[:, None, None].expand(-1, 1, 3)
    )[:, 0]
    return {
        "sample_count": len(batch["source_image"]),
        "maximum_energy_difference": max(differences.values()),
        "per_scale_maximum_difference": differences,
        "energies_allclose": bool(allclose),
        "top1_unchanged": bool(torch.equal(original_pose, permuted_pose)),
        "decoded_pose_maximum_difference": float((original_pose - permuted_pose).abs().max()),
    }


def _materialize_sources(
    generator: SyntheticRegistrationGenerator,
    manifest: dict,
    batch_size: int,
) -> dict:
    sources, truths, realization = [], [], []
    for start in range(0, len(manifest["ap_index"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(manifest["ap_index"])))
        piece = _take_manifest(manifest, indices)
        source, mask, available, records = oracle_realizations(
            generator, manifest, indices
        )
        sources.append(source.cpu())
        truths.append(_pose_from_manifest(piece, "cpu"))
        realization.extend(records)
        if bool(mask.any()) or bool(available.any()):
            raise RuntimeError("oracle atlas-pair premise requires absent masks")
    return {
        "source_image": torch.cat(sources),
        "true_pose": torch.cat(truths),
        "realization": realization,
        "panel_manifest_sha256": manifest["manifest_sha256"],
        "generator_contract_sha256": generator.contract["contract_sha256"],
        "atlas_sha256": {
            name: generator.contract[name]
            for name in (
                "average_template_sha256",
                "annotation_sha256",
                "query_sha256",
            )
        },
    }


def _fixed_candidate_evaluation(
    model: AtlasPairEnergyModel,
    renderer: SyntheticRegistrationGenerator,
    materialized: dict,
    seed: int,
    config: dict,
) -> dict:
    device = renderer.device
    source_all = materialized["source_image"].to(device)
    truth_all = materialized["true_pose"].to(device)
    source_order = _derangement(truth_all, seed + 101)
    counts = {"normal": 0, "broken_atlas_binding": 0, "broken_source_pairing": 0}
    nonfinite = 0
    invalid_render = 0
    raw = []
    order_batches = []
    batch_size = config["training"]["batch_size"]
    chunk_size = config["model"]["candidate_chunk_size"]
    for start in range(0, len(truth_all), batch_size):
        stop = min(start + batch_size, len(truth_all))
        indices = np.arange(start, stop)
        truth = truth_all[start:stop]
        candidate_pose, target, kinds = candidate_pose_table(truth, seed, indices)
        candidate_image, candidate_mask = render_candidate_poses(renderer, candidate_pose)
        invalid = (
            ~torch.isfinite(candidate_image).flatten(2).all(2)
            | ~candidate_mask.flatten(2).any(2)
        )
        invalid_render += int(invalid.sum())
        batch = {
            "source_image": source_all[start:stop],
            "source_mask": torch.zeros_like(source_all[start:stop], dtype=torch.bool),
            "mask_available": torch.zeros(stop - start, 1, 1, 1, device=device),
            "candidate_image": candidate_image,
            "candidate_mask": candidate_mask,
            "candidate_pose": candidate_pose,
        }
        normal = model(
            batch["source_image"], batch["source_mask"], batch["mask_available"],
            candidate_image, candidate_mask, candidate_chunk_size=chunk_size,
        )
        shifted_image = candidate_image.roll(1, 1)
        shifted_mask = candidate_mask.roll(1, 1)
        broken_atlas = model(
            batch["source_image"], batch["source_mask"], batch["mask_available"],
            shifted_image, shifted_mask, candidate_chunk_size=chunk_size,
        )
        source_indices = torch.as_tensor(source_order[start:stop], device=device)
        broken_source = model(
            source_all[source_indices], torch.zeros_like(source_all[start:stop], dtype=torch.bool),
            batch["mask_available"], candidate_image, candidate_mask,
            candidate_chunk_size=chunk_size,
        )
        outputs = {
            "normal": normal, "broken_atlas_binding": broken_atlas,
            "broken_source_pairing": broken_source,
        }
        for name, output in outputs.items():
            counts[name] += int((output["energy"].argmin(1) == target).sum())
            nonfinite += sum(int((~torch.isfinite(value)).sum()) for value in output.values())
        order_batches.append(
            _order_equivariance(
                model,
                batch,
                normal,
                seed + 202 + start,
                chunk_size,
                config["gates"]["order_energy_atol"],
                config["gates"]["order_energy_rtol"],
            )
        )
        for row in range(stop - start):
            raw.append(
                {
                    "sample_index": start + row,
                    "realization": materialized["realization"][start + row],
                    "latent_pose_id": hashlib.sha256(
                        _canonical_bytes({"panel": materialized["panel_manifest_sha256"], "pose": truth[row]})
                    ).hexdigest(),
                    "true_pose": truth[row],
                    "candidate_pose": candidate_pose[row],
                    "candidate_kind": kinds[row],
                    "target_index": int(target[row]),
                    "normal_energy": normal["energy"][row],
                    "normal_energy8": normal["energy8"][row],
                    "normal_energy16": normal["energy16"][row],
                    "broken_atlas_binding_energy": broken_atlas["energy"][row],
                    "broken_atlas_binding_energy8": broken_atlas["energy8"][row],
                    "broken_atlas_binding_energy16": broken_atlas["energy16"][row],
                    "broken_source_pairing_energy": broken_source["energy"][row],
                    "broken_source_pairing_energy8": broken_source["energy8"][row],
                    "broken_source_pairing_energy16": broken_source["energy16"][row],
                    "invalid_render": invalid[row],
                }
            )
    order_receipt = {
        "evaluated_sample_count": sum(value["sample_count"] for value in order_batches),
        "maximum_energy_difference": max(
            value["maximum_energy_difference"] for value in order_batches
        ),
        "energies_allclose": all(value["energies_allclose"] for value in order_batches),
        "top1_unchanged": all(value["top1_unchanged"] for value in order_batches),
        "decoded_pose_maximum_difference": max(
            value["decoded_pose_maximum_difference"] for value in order_batches
        ),
        "batch_receipts": order_batches,
    }
    return {
        "correct": counts,
        "order_equivariance": order_receipt,
        "nonfinite_count": nonfinite,
        "invalid_render_count": invalid_render,
        "raw": raw,
    }


def evaluate_panel(
    model: AtlasPairEnergyModel,
    renderer: SyntheticRegistrationGenerator,
    materialized: dict,
    seed: int,
    config: dict,
    *,
    free_search: bool,
) -> dict:
    model.eval()
    with torch.no_grad():
        fixed = _fixed_candidate_evaluation(model, renderer, materialized, seed, config)
        result = {
            "seed": seed,
            "sample_count": len(materialized["true_pose"]),
            "panel_manifest_sha256": materialized["panel_manifest_sha256"],
            "generator_contract_sha256": materialized["generator_contract_sha256"],
            "atlas_sha256": materialized["atlas_sha256"],
            "source_realizations": materialized["realization"],
            "fixed_candidates": fixed,
        }
        if free_search:
            predictions, receipts, seconds = [], [], []
            for item, source_cpu in enumerate(materialized["source_image"]):
                source = source_cpu[None].to(renderer.device)
                mask = torch.zeros_like(source, dtype=torch.bool)
                available = torch.zeros(1, 1, 1, 1, device=renderer.device)
                if renderer.device.type == "cuda":
                    torch.cuda.synchronize(renderer.device)
                started = time.perf_counter()
                prediction, receipt = coarse_to_fine_search(
                    model, renderer, source, mask, available, config
                )
                receipt["sample_index"] = item
                receipt["synthetic_realization_id"] = materialized["realization"][
                    item
                ]["synthetic_realization_id"]
                if renderer.device.type == "cuda":
                    torch.cuda.synchronize(renderer.device)
                seconds.append(time.perf_counter() - started)
                predictions.append(prediction.cpu())
                receipts.append(receipt)
            prediction = torch.stack(predictions).to(renderer.device)
            truth = materialized["true_pose"].to(renderer.device)
            absolute = (prediction - truth).abs()
            truth_ouv = _pose_to_quicknii_ouv(truth.double())
            prediction_ouv = _pose_to_quicknii_ouv(prediction.double())
            brain_mask = torch_annotation_brain_mask(
                truth_ouv, renderer.annotation, QUICKNII_PIXEL_GRID_SHAPE
            )
            physical = torch_brain_masked_plane_distance(
                truth_ouv, prediction_ouv, brain_mask
            ) * VOXEL_UM
            prior_pose = truth.mean(0, keepdim=True).expand_as(truth)
            prior = torch_brain_masked_plane_distance(
                truth_ouv, _pose_to_quicknii_ouv(prior_pose.double()), brain_mask
            ) * VOXEL_UM
            search_nonfinite = sum(value["nonfinite_count"] for value in receipts)
            search_invalid = sum(value["invalid_render_count"] for value in receipts)
            metric_values = (prediction, absolute, physical, prior)
            metric_nonfinite = sum(
                int((~torch.isfinite(value)).sum()) for value in metric_values
            )
            result["free_search"] = {
                "mae": absolute.mean(0),
                "absolute_pose_error": absolute,
                "physical_error_um": physical.mean(),
                "physical_error_um_per_slice": physical,
                "constant_prior_physical_error_um": prior.mean(),
                "constant_prior_physical_error_um_per_slice": prior,
                "physical_improvement_over_constant_prior": 1.0 - physical.mean() / prior.mean(),
                "seconds_per_slice": seconds,
                "ten_slice_projected_p95_seconds": 10.0 * float(np.quantile(seconds, 0.95)),
                "predicted_pose": prediction,
                "true_pose": truth,
                "search_receipts": receipts,
                "scorer_nonfinite_count": search_nonfinite,
                "metric_nonfinite_count": metric_nonfinite,
                "nonfinite_count": search_nonfinite + metric_nonfinite,
                "invalid_render_count": search_invalid,
            }
        result["result_sha256"] = _canonical_sha256(result)
        return result


def panel_status(result: dict, config: dict, *, require_search: bool) -> dict:
    gates = config["gates"]
    fixed = result["fixed_candidates"]
    order = fixed["order_equivariance"]
    search = result.get("free_search", {})
    nonfinite = fixed["nonfinite_count"] + search.get("nonfinite_count", 0)
    invalid_render = fixed["invalid_render_count"] + search.get(
        "invalid_render_count", 0
    )
    checks = {
        "nonfinite": nonfinite <= gates["nonfinite_count_maximum"],
        "invalid_render": invalid_render <= gates["invalid_render_count_maximum"],
        "truth_in_set": fixed["correct"]["normal"] >= gates["truth_in_set_correct_minimum_per_48"],
        "broken_atlas_binding": fixed["correct"]["broken_atlas_binding"] <= gates["broken_pair_correct_maximum_per_48"],
        "broken_source_pairing": fixed["correct"]["broken_source_pairing"] <= gates["broken_pair_correct_maximum_per_48"],
        "order_equivariance": (
            order["top1_unchanged"] and order["energies_allclose"]
            and order["decoded_pose_maximum_difference"] <= gates["order_energy_atol"]
            and order["evaluated_sample_count"] == result["sample_count"]
        ),
    }
    if require_search:
        checks.update(
            ap_mae=float(search["mae"][0]) <= gates["ap_mae_um_maximum"],
            lr_mae=float(search["mae"][1]) <= gates["lr_mae_deg_maximum"],
            dv_mae=float(search["mae"][2]) <= gates["dv_mae_deg_maximum"],
            physical_improvement=float(search["physical_improvement_over_constant_prior"])
            >= gates["physical_improvement_over_constant_prior_minimum"],
            runtime=float(search["ten_slice_projected_p95_seconds"])
            <= gates["ten_slice_p95_seconds_maximum"],
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "status_sha256": _canonical_sha256(checks),
    }


def _freeze_payload(config: dict, checkpoint_path: Path, checkpoint: dict) -> dict:
    lineage = checkpoint["data_lineage"]
    return {
        "format": "independent-atlas-pair-qualification-freeze-v1",
        "purpose": PURPOSE,
        "source_sha256": source_hashes(),
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_file_sha256": _sha256(checkpoint_path),
        "training_history_sha256": checkpoint["training_history_sha256"],
        "applied_optimizer_updates": len(checkpoint["training_history"]),
        "data_lineage_sha256": lineage["data_lineage_sha256"],
        "generator_contract_sha256": lineage["generator_contract_sha256"],
        "atlas_sha256": lineage["atlas_sha256"],
        "final_update": config["training"]["max_updates"],
        "qualification_seeds": config["data"]["qualification_seeds"],
        "qualification_count_per_seed": config["data"]["qualification_count_per_seed"],
        "learned_checkpoint_dependencies": [],
    }


def _verify_final_checkpoint(config: dict, checkpoint_path: Path) -> dict:
    if not checkpoint_path.is_file():
        raise RuntimeError("qualification freeze requires the final checkpoint")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError("qualification freeze requires a readable final checkpoint") from error
    checkpoint_lineage = checkpoint.get("data_lineage", {})
    lineage_commitment = checkpoint_lineage.pop("data_lineage_sha256", None)
    lineage_valid = lineage_commitment == _canonical_sha256(checkpoint_lineage)
    if lineage_commitment is not None:
        checkpoint_lineage["data_lineage_sha256"] = lineage_commitment
    generator_contract = checkpoint_lineage.get("generator_contract", {})
    generator_payload = {
        name: value
        for name, value in generator_contract.items()
        if name != "contract_sha256"
    }
    generator_valid = (
        generator_contract.get("contract_sha256")
        == _payload_sha256(generator_payload)
        == checkpoint_lineage.get("generator_contract_sha256")
        and checkpoint_lineage.get("atlas_sha256")
        == {
            name: generator_contract.get(name)
            for name in (
                "average_template_sha256",
                "annotation_sha256",
                "query_sha256",
            )
        }
    )
    history = checkpoint.get("training_history", [])
    history_valid = (
        len(history) == checkpoint.get("update")
        and checkpoint.get("training_history_sha256") == _canonical_sha256(history)
        and all(
            value.get("update") == item
            and value.get("optimizer_step_applied") is True
            for item, value in enumerate(history, 1)
        )
    )
    if (
        checkpoint.get("format") != "independent-atlas-pair-energy-final-v1"
        or checkpoint.get("config_contract_sha256") != config["contract_sha256"]
        or checkpoint.get("config_file_sha256") != config["config_file_sha256"]
        or checkpoint.get("source_sha256") != source_hashes()
        or checkpoint.get("learned_checkpoint_dependencies") != []
        or checkpoint.get("update") != config["training"]["max_updates"]
        or not isinstance(checkpoint.get("model"), dict)
        or not checkpoint["model"]
        or not lineage_valid
        or not generator_valid
        or not history_valid
    ):
        raise RuntimeError("qualification freeze checkpoint is not the committed final state")
    return checkpoint


def freeze_qualification(run_folder: Path, config: dict, checkpoint_path: Path) -> Path:
    checkpoint = _verify_final_checkpoint(config, checkpoint_path)
    payload = _freeze_payload(config, checkpoint_path, checkpoint)
    payload["freeze_payload_sha256"] = _canonical_sha256(payload)
    path = run_folder / "qualification_freeze.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != _canonical(payload):
            raise RuntimeError("existing qualification freeze differs")
    else:
        _atomic_json(path, payload)
    return path


def verified_qualification_capability(
    run_folder: Path, config: dict, checkpoint_path: Path
):
    checkpoint = _verify_final_checkpoint(config, checkpoint_path)
    freeze_path = run_folder / "qualification_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("qualification tensors are forbidden before final freeze")
    observed = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = _freeze_payload(config, checkpoint_path, checkpoint)
    commitment = observed.pop("freeze_payload_sha256", None)
    if observed != _canonical(expected) or commitment != _canonical_sha256(expected):
        raise RuntimeError("qualification freeze no longer matches source/config/checkpoint")
    return {
        "guard": _QUALIFICATION_CAPABILITY,
        "freeze_payload_sha256": commitment,
        "source_sha256": expected["source_sha256"],
        "config_contract_sha256": expected["config_contract_sha256"],
        "checkpoint_file_sha256": expected["checkpoint_file_sha256"],
        "generator_contract_sha256": expected["generator_contract_sha256"],
        "data_lineage_sha256": expected["data_lineage_sha256"],
        "qualification_seeds": tuple(expected["qualification_seeds"]),
        "qualification_count_per_seed": expected["qualification_count_per_seed"],
    }


def qualification_manifests(
    generator: SyntheticRegistrationGenerator,
    config: dict,
    run_folder: Path,
    checkpoint_path: Path,
) -> list[dict]:
    capability = verified_qualification_capability(
        run_folder, config, checkpoint_path
    )
    if (
        not isinstance(capability, dict)
        or capability.get("guard") is not _QUALIFICATION_CAPABILITY
        or capability.get("generator_contract_sha256")
        != generator.contract["contract_sha256"]
        or capability.get("source_sha256") != source_hashes()
        or capability.get("config_contract_sha256") != config["contract_sha256"]
    ):
        raise RuntimeError("qualification tensors require verified post-checkpoint capability")
    return [
        balanced_panel_manifest(
            generator,
            seed,
            config["data"]["qualification_count_per_seed"],
            qualification_capability=capability,
        )
        for seed in config["data"]["qualification_seeds"]
    ]


def _device(config: dict) -> torch.device:
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run(config_path: str | Path) -> Path:
    config = inspect_config(config_path)
    device = _device(config)
    run_root = Path(os.environ.get("ATLAS_JOINT_RUN_ROOT", ROOT / "training-runs"))
    run_folder = run_root / config["name"]
    run_folder.mkdir(parents=True, exist_ok=True)
    atlas_path = ROOT / config["paths"]["atlas_repo_relative"]
    generator = SyntheticRegistrationGenerator(atlas_path, device)
    train_manifest = training_manifest(generator, config)
    dev_manifest = balanced_panel_manifest(
        generator, config["data"]["development_seed"], config["data"]["development_count"]
    )
    lineage = data_lineage(generator, train_manifest, dev_manifest, config)
    setup = {
        "format": "independent-atlas-pair-energy-setup-v1",
        "purpose": PURPOSE,
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": source_hashes(),
        "data_lineage": lineage,
        "learned_checkpoint_dependencies": [],
    }
    setup["setup_payload_sha256"] = _canonical_sha256(setup)
    setup_path = run_folder / "setup_receipt.json"
    if setup_path.exists():
        if json.loads(setup_path.read_text(encoding="utf-8")) != _canonical(setup):
            raise RuntimeError("existing atlas-pair setup differs from frozen lineage")
    else:
        _atomic_json(setup_path, setup)
    write_manifest_receipt(
        run_folder / "training_manifest.json",
        "training",
        train_manifest,
        generator,
        config,
    )
    write_manifest_receipt(
        run_folder / "development_manifest.json",
        "development",
        dev_manifest,
        generator,
        config,
    )
    random.seed(config["optimizer_seed"])
    np.random.seed(config["optimizer_seed"])
    torch.manual_seed(config["optimizer_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["optimizer_seed"])
    model = AtlasPairEnergyModel().to(device)
    if parameter_count(model) != config["model"]["expected_parameter_count"]:
        raise RuntimeError("atlas-pair parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    amp = bool(config["training"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler(
        device.type, enabled=amp, init_scale=config["training"]["amp_initial_scale"]
    )
    resume_path = run_folder / "resume_state.pt"
    update = 0
    development = []
    training_history = []
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        history = state.get("training_history", [])
        if (
            state.get("format") != FORMAT
            or state.get("config_contract_sha256") != config["contract_sha256"]
            or state.get("config_file_sha256") != config["config_file_sha256"]
            or state.get("source_sha256") != source_hashes()
            or state.get("learned_checkpoint_dependencies") != []
            or state.get("data_lineage") != lineage
            or len(history) != int(state.get("update", -1))
            or state.get("training_history_sha256") != _canonical_sha256(history)
            or any(
                value.get("update") != item
                or value.get("optimizer_step_applied") is not True
                for item, value in enumerate(history, 1)
            )
        ):
            raise RuntimeError("atlas-pair resume state lineage differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        update = int(state["update"])
        development = state["development"]
        training_history = history
    chunk_size = config["model"]["candidate_chunk_size"]
    source_cache = {}
    while update < config["training"]["max_updates"]:
        indices = training_indices(
            update, config["data"]["train_count"], config["training"]["batch_size"],
            config["data"]["train_seed"],
        )
        manifest = _take_manifest(train_manifest, indices)
        source, source_mask, available, realization = oracle_realizations(
            generator, train_manifest, indices, source_cache
        )
        truth = _pose_from_manifest(manifest, device)
        candidate_pose, target, _ = candidate_pose_table(
            truth, config["data"]["train_seed"], indices
        )
        candidate_image, candidate_mask = render_candidate_poses(generator, candidate_pose)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            output = model(
                source, source_mask, available, candidate_image, candidate_mask,
                candidate_chunk_size=chunk_size,
            )
            losses = atlas_pair_loss(output, candidate_pose, truth, target)
        if not bool(torch.isfinite(losses["total"])) or any(
            not bool(torch.isfinite(value).all()) for value in output.values()
        ):
            raise RuntimeError("atlas-pair training produced nonfinite values")
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["training"]["gradient_clip_norm"]
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("atlas-pair training produced nonfinite unscaled gradients")
        tracked_parameter = next(model.parameters())
        step_before = int(optimizer.state.get(tracked_parameter, {}).get("step", 0))
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        step_after = int(optimizer.state.get(tracked_parameter, {}).get("step", 0))
        scale_after = float(scaler.get_scale())
        if step_after != step_before + 1 or scale_after < scale_before:
            raise RuntimeError("AMP skipped the frozen optimizer update")
        update += 1
        training_history.append(
            {
                "update": update,
                "sample_indices": indices.tolist(),
                "source_realization": realization,
                "candidate_pose_sha256": _tensor_sha256(candidate_pose),
                "target_index": target.detach().cpu().tolist(),
                "loss": {
                    name: float(losses[name].detach())
                    for name in ("total", "ranking", "auxiliary_ranking", "point")
                },
                "unscaled_gradient_norm": float(grad_norm),
                "amp_scale_before": scale_before,
                "amp_scale_after": scale_after,
                "optimizer_step_before": step_before,
                "optimizer_step_after": step_after,
                "optimizer_step_applied": True,
            }
        )
        if update in config["training"]["development_updates"]:
            materialized = _materialize_sources(
                generator, dev_manifest, config["training"]["batch_size"]
            )
            evaluation = evaluate_panel(
                model, generator, materialized, config["data"]["development_seed"],
                config, free_search=False,
            )
            evaluation["update"] = update
            evaluation["status"] = panel_status(evaluation, config, require_search=False)
            development.append(evaluation)
            _atomic_json(run_folder / f"development_{update:04d}.json", evaluation)
        if update % config["training"]["resume_every_updates"] == 0 or update == config["training"]["max_updates"]:
            _atomic_torch(
                resume_path,
                {
                    "format": FORMAT,
                    "config_contract_sha256": config["contract_sha256"],
                    "config_file_sha256": config["config_file_sha256"],
                    "source_sha256": source_hashes(),
                    "learned_checkpoint_dependencies": [],
                    "data_lineage": lineage,
                    "update": update,
                    "training_history": training_history,
                    "training_history_sha256": _canonical_sha256(training_history),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "development": development,
                },
            )
    final_path = run_folder / "final_checkpoint.pt"
    final_payload = {
        "format": "independent-atlas-pair-energy-final-v1",
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": source_hashes(),
        "learned_checkpoint_dependencies": [],
        "data_lineage": lineage,
        "update": update,
        "training_history": training_history,
        "training_history_sha256": _canonical_sha256(training_history),
        "model": model.state_dict(),
    }
    if final_path.exists():
        existing = torch.load(final_path, map_location="cpu", weights_only=False)
        if (
            existing.get("config_contract_sha256") != config["contract_sha256"]
            or existing.get("config_file_sha256") != config["config_file_sha256"]
            or existing.get("source_sha256") != source_hashes()
            or existing.get("learned_checkpoint_dependencies") != []
            or existing.get("data_lineage") != lineage
            or existing.get("update") != update
            or len(existing.get("training_history", [])) != update
            or existing.get("training_history_sha256")
            != _canonical_sha256(existing.get("training_history", []))
            or existing.get("training_history_sha256")
            != _canonical_sha256(training_history)
        ):
            raise RuntimeError("existing final checkpoint differs")
        model.load_state_dict(existing["model"])
    else:
        _atomic_torch(final_path, final_payload)
    trajectory_path = write_training_trajectory(
        run_folder / "training_trajectory.json",
        training_history,
        final_path,
        config,
    )
    freeze_qualification(run_folder, config, final_path)
    receipt_path = run_folder / "diagnostic_receipt.json"
    if receipt_path.exists():
        existing_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        uncommitted = {name: value for name, value in existing_receipt.items() if name != "receipt_sha256"}
        if (
            existing_receipt.get("receipt_sha256") != _canonical_sha256(uncommitted)
            or existing_receipt.get("config_contract_sha256") != config["contract_sha256"]
            or existing_receipt.get("config_file_sha256") != config["config_file_sha256"]
            or existing_receipt.get("source_sha256") != source_hashes()
            or existing_receipt.get("checkpoint_file_sha256") != _sha256(final_path)
            or existing_receipt.get("training_trajectory_file_sha256")
            != _sha256(trajectory_path)
            or existing_receipt.get("data_lineage") != _canonical(lineage)
        ):
            raise RuntimeError("existing atlas-pair receipt lineage differs")
        return run_folder
    capability = verified_qualification_capability(run_folder, config, final_path)
    qualification = []
    for manifest in qualification_manifests(
        generator, config, run_folder, final_path
    ):
        if verified_qualification_capability(run_folder, config, final_path) != capability:
            raise RuntimeError("qualification freeze changed before tensor generation")
        seed = int(manifest["seed"])
        qualification_manifest_path = write_manifest_receipt(
            run_folder / f"qualification_manifest_seed_{seed}.json",
            "post-freeze-qualification",
            manifest,
            generator,
            config,
            {
                "checkpoint_file_sha256": capability["checkpoint_file_sha256"],
                "freeze_payload_sha256": capability["freeze_payload_sha256"],
                "data_lineage_sha256": capability["data_lineage_sha256"],
            },
        )
        qualification_path = run_folder / f"qualification_seed_{seed}.json"
        qualification_lineage = {
            "source_sha256": source_hashes(),
            "config_contract_sha256": config["contract_sha256"],
            "config_file_sha256": config["config_file_sha256"],
            "checkpoint_file_sha256": _sha256(final_path),
            "data_lineage_sha256": lineage["data_lineage_sha256"],
            "generator_contract_sha256": lineage["generator_contract_sha256"],
            "atlas_sha256": lineage["atlas_sha256"],
            "panel_manifest_sha256": manifest["manifest_sha256"],
            "panel_manifest_file_sha256": _sha256(qualification_manifest_path),
        }
        if qualification_path.exists():
            result = json.loads(qualification_path.read_text(encoding="utf-8"))
            result_commitment = result.pop("qualification_file_payload_sha256", None)
            if (
                result.get("lineage") != qualification_lineage
                or result_commitment != _canonical_sha256(result)
            ):
                raise RuntimeError("existing qualification result lineage differs")
            result["qualification_file_payload_sha256"] = result_commitment
            qualification.append(result)
            continue
        materialized = _materialize_sources(
            generator, manifest, config["training"]["batch_size"]
        )
        result = evaluate_panel(model, generator, materialized, seed, config, free_search=True)
        if verified_qualification_capability(run_folder, config, final_path) != capability:
            raise RuntimeError("qualification freeze changed during tensor evaluation")
        result["status"] = panel_status(result, config, require_search=True)
        result["lineage"] = qualification_lineage
        result["qualification_file_payload_sha256"] = _canonical_sha256(result)
        qualification.append(result)
        _atomic_json(qualification_path, result)
    receipt = {
        "format": "independent-atlas-pair-energy-receipt-v1",
        "purpose": PURPOSE,
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": source_hashes(),
        "checkpoint_file_sha256": _sha256(final_path),
        "training_trajectory_file_sha256": _sha256(trajectory_path),
        "training_history_sha256": _canonical_sha256(training_history),
        "data_lineage": lineage,
        "qualification": qualification,
        "passed": all(value["status"]["passed"] for value in qualification),
        "learned_checkpoint_dependencies": [],
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    _atomic_json(receipt_path, receipt)
    return run_folder


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit("usage: python -m training.run_independent_atlas_pair_energy CONFIG")
    print(run(arguments[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
