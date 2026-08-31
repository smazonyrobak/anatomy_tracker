"""Run the frozen paired global-versus-spatial atlas-pair diagnostic."""

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

from training.independent_atlas_pair_energy import (
    _local_correlation,
    atlas_pair_loss,
    parameter_count,
)
from training.independent_atlas_pair_spatial_aggregation import (
    EXPECTED_PARAMETER_COUNT,
    AtlasPairEnergyGlobalAggregationControl,
    AtlasPairEnergyHaar2x2SpatialAggregation,
    _masked_global_haar_statistics,
)
from training.quicknii_plane_metric import (
    QUICKNII_PIXEL_GRID_SHAPE,
    torch_annotation_brain_mask,
    torch_brain_masked_plane_distance,
)
from training.run_independent_atlas_pair_energy import (
    _commit_manifest,
    _pose_from_manifest,
    _pose_to_quicknii_ouv,
    _take_manifest,
    candidate_pose_table,
    coarse_to_fine_search,
    oracle_realizations,
    render_candidate_poses,
    training_indices,
    training_manifest,
)
from training.synthetic_registration import (
    AP_MAX_UM,
    AP_MIN_UM,
    BREGMA_AP_INDEX,
    VOXEL_UM,
    SyntheticRegistrationGenerator,
    split_ap_indices,
)


ROOT = Path(__file__).parents[1]
PURPOSE = "development-only-paired-spatial-aggregation-causal-diagnostic"
ROLE = "causal-rescue-premise-not-model-selection"
FAMILY = "independent-oracle-atlas-pair-spatial-aggregation-pair-1500-r1404322-v1"
RESUME_FORMAT = "independent-atlas-pair-spatial-aggregation-paired-state-v1"
FINAL_FORMAT = "independent-atlas-pair-spatial-aggregation-paired-final-v1"
FREEZE_FORMAT = "independent-atlas-pair-spatial-aggregation-joint-freeze-v1"
RECEIPT_FORMAT = "independent-atlas-pair-spatial-aggregation-paired-receipt-v1"
QUALIFICATION_SEEDS = (1604322, 1704322)
CONSUMED_QUALIFICATION_SEEDS = (1204322, 1304322, 1504322)
TRUTH_AP_MARGIN_UM = 500.0
TRUTH_TILT_ABS_MAX_DEG = 25.0
_QUALIFICATION_CAPABILITY = object()

CONSUMED_SOURCE_SHA256 = {
    "training/independent_atlas_pair_energy.py": (
        "6187cb051d048d1e5eec3137b9edc6ac09706cecffcf989507951708681589ec"
    ),
    "training/run_independent_atlas_pair_energy.py": (
        "21c73f88a48ca87ac0a44ff022993eea5dc2cfbb1f8c72237ec4b05fa4445b19"
    ),
}
SOURCE_FILES = (
    "training/independent_atlas_pair_spatial_aggregation.py",
    "training/run_independent_atlas_pair_spatial_aggregation.py",
    "training/independent_atlas_pair_energy.py",
    "training/run_independent_atlas_pair_energy.py",
    "training/synthetic_registration.py",
    "training/quicknii_plane_metric.py",
    "source/dense_registration_preprocessing.py",
)


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
    observed = {name: _sha256(ROOT / name) for name in SOURCE_FILES}
    for name, expected in CONSUMED_SOURCE_SHA256.items():
        if observed[name] != expected:
            raise RuntimeError(f"consumed source changed: {name}")
    return observed


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(payload) + b"\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_immutable_json(path: Path, payload) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != _canonical(payload):
            raise RuntimeError(f"existing immutable receipt differs: {path.name}")
        return
    _atomic_json(path, payload)


def _expected_config() -> dict:
    return {
        "schema_version": 1,
        "frozen": True,
        "name": FAMILY,
        "purpose": PURPOSE,
        "role": ROLE,
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
        "family": {
            "self_hash": "contract_sha256-over-complete-payload-excluding-contract_sha256",
            "arm_order": ["null", "treatment"],
            "null": {
                "class": "training.independent_atlas_pair_spatial_aggregation.AtlasPairEnergyGlobalAggregationControl",
                "aggregation": "global-mean-max-plus-243-exact-zero-contrast-inputs",
            },
            "treatment": {
                "class": "training.independent_atlas_pair_spatial_aggregation.AtlasPairEnergyHaar2x2SpatialAggregation",
                "aggregation": "global-mean-max-plus-top-bottom-left-right-diagonal-haar-contrasts",
            },
            "initialization": "instantiate-both-then-strict-load-complete-null-state-into-treatment",
            "initial_state_requirement": "bit-identical-all-tensors-and-state-schema",
            "claim_scope": "causal-fixed-2x2-haar-correlation-contrast-access-rescue-only",
            "parameter_matching_caveat": "null-243-contrast-columns-are-dormant-exact-zero-inputs; state-schema-and-parameter-count-match-but-functional-input-rank-is-intentionally-unequal",
        },
        "model": {
            "expected_parameter_count_each": 271780,
            "maximum_parameter_count_each": 1500000,
            "input_shape": [160, 232],
            "correlation_levels": [8, 16],
            "correlation_radius": 4,
            "candidate_chunk_size": 8,
            "candidate_pose_input": False,
            "statistics_dimension_each_level": 405,
            "shared_global_statistics_dimension": 162,
            "contrast_statistics_dimension": 243,
            "head": "LayerNorm(405)-Linear(405,25)-GELU-Linear(25,1)",
        },
        "data": {
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
            "qualification_seeds": [1604322, 1704322],
            "consumed_or_forbidden_qualification_seeds": [1204322, 1304322, 1504322],
            "qualification_count_per_seed": 48,
            "qualification_generation": "only-after-both-final-states-jointly-freeze",
            "panel_balance": "six-interior-ap-strata-times-eight-signed-tilt-configurations",
            "premise_truth_ap_margin_um": 500.0,
            "premise_truth_tilt_abs_max_deg": 25.0,
            "truth_support_role": "interior-only-causal-premise-not-outer-domain-evidence",
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 0.0002,
            "weight_decay": 0.0001,
            "batch_size": 2,
            "max_updates": 1500,
            "paired_presentations": 3000,
            "amp": True,
            "amp_initial_scale": 512.0,
            "amp_required_to_stay": 512.0,
            "gradient_clip_norm": 5.0,
            "gradient_clipping_fraction_gate": False,
            "resume_every_updates": 25,
            "development_updates": [500, 1000, 1500],
            "candidate_count": 16,
            "loss": "ranking+0.25*two-scale-ranking+0.25*posterior-point",
            "step_barrier": "both-unscaled-finite-checked-and-clipped-before-either-optimizer-step",
            "persistence_unit": "completed-paired-update-only",
            "early_stopping": False,
            "budget_extension": False,
        },
        "search": {
            "coarse_shape": [9, 5, 5],
            "top_k": 3,
            "refinement_rounds": 3,
            "neighborhood_shape": [3, 3, 3],
            "maximum_candidate_evaluations_per_slice": 468,
            "continuous_refinement": False,
        },
        "evaluation": {
            "development": "paired-fixed-candidates-only",
            "qualification_fixed_candidates": "one-shared-source-and-candidate-tensor-set-for-both-arms",
            "qualification_free_search": "always-run-unchanged-search-for-both-arms",
            "qualification_free_search_arm_order": "alternate-null-first-on-even-sample-treatment-first-on-odd-sample",
            "cross_arm_energy_delta_gate": False,
            "no_selective_missing": "exactly-48-source-keyed-predictions-errors-and-search-receipts-per-arm-per-qualification-seed",
            "pooled_96_role": "descriptive-only-cannot-rescue-failed-seed",
            "gate_roles": {
                "treatment": "all-absolute-integrity-control-accuracy-and-runtime-gates",
                "null": "integrity-control-runtime-plus-causal-ranking-contrast; other-accuracy-descriptive",
            },
        },
        "gates": {
            "nonfinite_count_maximum": 0,
            "invalid_render_count_maximum": 0,
            "truth_in_set_correct_minimum_per_48": 46,
            "ap_mae_um_maximum": 250.0,
            "lr_mae_deg_maximum": 3.0,
            "dv_mae_deg_maximum": 3.0,
            "physical_improvement_over_constant_prior_minimum": 0.50,
            "broken_pair_correct_maximum_per_48": 12,
            "order_energy_atol": 0.000001,
            "order_energy_rtol": 0.000001,
            "ten_slice_p95_seconds_maximum": 180.0,
            "causal": {
                "paired_rows_per_seed": 48,
                "treatment_truth_in_set_minimum": 46,
                "null_truth_in_set_maximum": 45,
                "net_corrections_minimum": 8,
                "exact_two_sided_mcnemar_p_maximum": 0.01,
            },
        },
        "integrity": {
            "first_global_statistics_exact_dimension": 162,
            "null_contrast_statistics_exact_zero_dimension": 243,
            "paired_input_identity_required_updates": 1500,
            "optimizer_steps_required": [1, 1500],
            "amp_scale_required": 512.0,
            "final_must_equal_resume": True,
            "paired_qualification_rows_required": 96,
        },
        "interpretation": {
            "integrity_failure": "invalid-stop",
            "both_fail": "insufficient-stop",
            "treatment_pass_null_fail_paired_pass": "causal-rescue-only-authorize-independent-confirmation",
            "both_pass": "no-spatial-necessity-claim",
            "one_seed_pass": "family-fail",
            "ranking_causal_pass_but_treatment_free_search_fail": "local-fixed-haar-mechanism-supported-end-to-end-no-go",
            "prohibitions": [
                "no-early-stop",
                "no-extension",
                "no-gate-change",
                "no-seed-substitution",
                "no-protected-data",
                "no-promotion-from-this-family",
            ],
        },
        "lineage": {"source_sha256": source_hashes()},
    }


def inspect_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    commitment = raw.pop("contract_sha256", None)
    if raw != _expected_config():
        raise ValueError("paired spatial-aggregation config differs from the frozen family")
    if commitment != _canonical_sha256(raw):
        raise ValueError("paired spatial-aggregation family self-hash differs")
    raw["contract_sha256"] = commitment
    raw["family_self_sha256"] = commitment
    raw["config_file_sha256"] = _sha256(config_path)
    raw["config_path"] = str(config_path)
    return raw


def _assert_frozen_files(config: dict) -> dict[str, str]:
    expected_sources = config.get("lineage", {}).get("source_sha256")
    observed_sources = source_hashes()
    config_path = Path(config.get("config_path", ""))
    if (
        observed_sources != expected_sources
        or not config_path.is_file()
        or _sha256(config_path) != config.get("config_file_sha256")
    ):
        raise RuntimeError("frozen source or config bytes changed during the paired run")
    return expected_sources


def _premise_ap_pool() -> tuple[np.ndarray, np.ndarray]:
    pool = np.asarray(split_ap_indices("train"), np.float32)
    ap_um = (BREGMA_AP_INDEX - pool) * VOXEL_UM
    keep = (ap_um >= AP_MIN_UM + TRUTH_AP_MARGIN_UM) & (
        ap_um <= AP_MAX_UM - TRUTH_AP_MARGIN_UM
    )
    return pool[keep], ap_um[keep]


def balanced_panel_manifest(
    generator: SyntheticRegistrationGenerator,
    seed: int,
    count: int = 48,
    *,
    qualification_capability=None,
) -> dict:
    if count != 48:
        raise ValueError("frozen paired panels contain exactly 48 sources")
    if seed in CONSUMED_QUALIFICATION_SEEDS:
        raise RuntimeError("consumed qualification seed is forbidden in this family")
    if seed in QUALIFICATION_SEEDS and (
        not isinstance(qualification_capability, dict)
        or qualification_capability.get("guard") is not _QUALIFICATION_CAPABILITY
        or seed not in qualification_capability.get("qualification_seeds", ())
        or count != qualification_capability.get("qualification_count_per_seed")
        or not qualification_capability.get("joint_final_file_sha256")
        or not qualification_capability.get("freeze_payload_sha256")
        or not qualification_capability.get("data_lineage_sha256")
    ):
        raise RuntimeError("qualification manifest generation is forbidden before joint freeze")
    manifest = generator.make_manifest(count, "train", seed, "clean")
    pool, ap_um = _premise_ap_pool()
    strata = np.array_split(np.argsort(ap_um), 6)
    rng = np.random.default_rng(seed + 31)
    chosen = [int(rng.choice(value)) for value in strata]
    tilts = np.asarray(
        [
            (-13.25, -18.25),
            (-13.25, 18.25),
            (13.25, -18.25),
            (13.25, 18.25),
            (-13.25, 0.0),
            (13.25, 0.0),
            (0.0, -18.25),
            (0.0, 18.25),
        ],
        np.float32,
    )
    manifest["ap_index"] = np.repeat(pool[chosen], 8).astype(np.float32)
    manifest["ap_um"] = np.repeat(ap_um[chosen], 8).astype(np.float32)
    manifest["tilt_lr_deg"] = np.tile(tilts[:, 0], 6)
    manifest["tilt_dv_deg"] = np.tile(tilts[:, 1], 6)
    return _commit_manifest(manifest)


def _device(config: dict) -> torch.device:
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def initialize_pair(config: dict, device: torch.device):
    _seed_everything(int(config["optimizer_seed"]), device)
    null = AtlasPairEnergyGlobalAggregationControl().to(device)
    treatment = AtlasPairEnergyHaar2x2SpatialAggregation().to(device)
    incompatibility = treatment.load_state_dict(null.state_dict(), strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError("paired strict initialization reported incompatible state keys")
    if parameter_count(null) != EXPECTED_PARAMETER_COUNT or parameter_count(
        treatment
    ) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("paired spatial-aggregation parameter count changed")
    null_state = null.state_dict()
    treatment_state = treatment.state_dict()
    if list(null_state) != list(treatment_state):
        raise RuntimeError("paired model state schemas differ")
    null_hash = _state_dict_sha256(null_state)
    treatment_hash = _state_dict_sha256(treatment_state)
    if null_hash != treatment_hash or any(
        not torch.equal(null_state[name], treatment_state[name]) for name in null_state
    ):
        raise RuntimeError("paired models are not bit-identical after strict initialization")
    return {"null": null, "treatment": treatment}, {
        "seed": int(config["optimizer_seed"]),
        "parameter_count_each": EXPECTED_PARAMETER_COUNT,
        "state_schema": list(null_state),
        "state_schema_sha256": _canonical_sha256(list(null_state)),
        "null_state_sha256": null_hash,
        "treatment_state_sha256": treatment_hash,
        "full_initial_state_equal": True,
    }


def _optimizers_and_scalers(models: dict, config: dict, device: torch.device):
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        for name, model in models.items()
    }
    amp = bool(config["training"]["amp"] and device.type == "cuda")
    scalers = {
        name: torch.amp.GradScaler(
            device.type,
            enabled=amp,
            init_scale=config["training"]["amp_initial_scale"],
        )
        for name in models
    }
    return optimizers, scalers, amp


def _rng_state(device: torch.device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def _restore_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def data_lineage(
    generator: SyntheticRegistrationGenerator,
    train_manifest: dict,
    development_manifest: dict,
    config: dict,
) -> dict:
    frozen_sources = _assert_frozen_files(config)
    lineage = {
        "family_self_sha256": config["family_self_sha256"],
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": frozen_sources,
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
        "reserved_fresh_qualification_seeds": config["data"]["qualification_seeds"],
        "consumed_or_forbidden_qualification_seeds": config["data"][
            "consumed_or_forbidden_qualification_seeds"
        ],
        "oracle_source_contract": {
            "tensor": "moving_raw_uint8.float()/255 before source-view transform",
            "source_view_rotation_deg": 0.0,
            "source_view_scale": 1.0,
            "pre_downsample_resampling_count": 0,
            "fixed_downsample": "bilinear-align_corners-false-160x232",
            "source_mask": "all-zero",
            "mask_available": "all-zero",
        },
        "shared_training_contract": {
            "source_candidate_generation_count_per_update": 1,
            "same_tensor_objects_presented_to_both_arms": True,
            "candidate_pose_is_scorer_input": False,
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
    frozen_sources = _assert_frozen_files(config)
    payload = {
        "format": "independent-atlas-pair-spatial-aggregation-manifest-v1",
        "purpose": PURPOSE,
        "role": role,
        "family_self_sha256": config["family_self_sha256"],
        "source_sha256": frozen_sources,
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "generator_contract_sha256": generator.contract["contract_sha256"],
        "manifest": manifest,
    }
    if post_freeze_binding is not None:
        payload["post_freeze_binding"] = post_freeze_binding
    payload["manifest_file_payload_sha256"] = _canonical_sha256(payload)
    _write_immutable_json(path, payload)
    return path


def _materialize_sources(
    generator: SyntheticRegistrationGenerator, manifest: dict, batch_size: int
) -> dict:
    sources, truths, realization = [], [], []
    for start in range(0, len(manifest["ap_index"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(manifest["ap_index"])))
        piece = _take_manifest(manifest, indices)
        source, mask, available, records = oracle_realizations(
            generator, manifest, indices
        )
        if bool(mask.any()) or bool(available.any()):
            raise RuntimeError("paired oracle diagnostic requires absent masks")
        sources.append(source.cpu())
        truths.append(_pose_from_manifest(piece, "cpu"))
        realization.extend(records)
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


def _training_batch(
    update: int,
    generator: SyntheticRegistrationGenerator,
    train_manifest: dict,
    config: dict,
    source_cache: dict,
) -> dict:
    indices = training_indices(
        update,
        config["data"]["train_count"],
        config["training"]["batch_size"],
        config["data"]["train_seed"],
    )
    manifest = _take_manifest(train_manifest, indices)
    source, source_mask, available, realization = oracle_realizations(
        generator, train_manifest, indices, source_cache
    )
    truth = _pose_from_manifest(manifest, generator.device)
    candidate_pose, target, kinds = candidate_pose_table(
        truth, config["data"]["train_seed"], indices
    )
    candidate_image, candidate_mask = render_candidate_poses(generator, candidate_pose)
    return {
        "sample_indices": indices,
        "source_image": source,
        "source_mask": source_mask,
        "mask_available": available,
        "true_pose": truth,
        "candidate_pose": candidate_pose,
        "candidate_image": candidate_image,
        "candidate_mask": candidate_mask,
        "target_index": target,
        "candidate_kind": kinds,
        "source_realization": realization,
    }


def _input_commitments(batch: dict) -> dict:
    return {
        name: _tensor_sha256(batch[name])
        for name in (
            "source_image",
            "source_mask",
            "mask_available",
            "true_pose",
            "candidate_pose",
            "candidate_image",
            "candidate_mask",
            "target_index",
        )
    }


def _statistics_for_model(model, batch: dict, use_haar: bool) -> dict:
    source8, source16 = model.encode_source(
        batch["source_image"], batch["source_mask"], batch["mask_available"]
    )
    atlas_image = batch["candidate_image"].flatten(0, 1)
    atlas_mask = batch["candidate_mask"].flatten(0, 1)
    atlas8, atlas16 = model.encode_atlas(atlas_image, atlas_mask)
    repeats = len(atlas8) // len(source8)
    result = {}
    for level, source, atlas in (
        ("8", source8.repeat_interleave(repeats, 0), atlas8),
        ("16", source16.repeat_interleave(repeats, 0), atlas16),
    ):
        result[level] = _masked_global_haar_statistics(
            _local_correlation(source, atlas, model.radius),
            atlas_mask,
            use_haar_coefficients=use_haar,
        )
    return result


def initial_statistics_integrity(models: dict, batch: dict) -> dict:
    modes = {name: model.training for name, model in models.items()}
    for model in models.values():
        model.eval()
    with torch.no_grad():
        null = _statistics_for_model(models["null"], batch, False)
        treatment = _statistics_for_model(models["treatment"], batch, True)
    for name, model in models.items():
        model.train(modes[name])
    levels = {}
    for level in ("8", "16"):
        levels[level] = {
            "first_162_exact": bool(
                torch.equal(null[level][:, :162], treatment[level][:, :162])
            ),
            "null_last_243_exact_zero": bool(
                torch.equal(null[level][:, 162:], torch.zeros_like(null[level][:, 162:]))
            ),
            "null_statistics_sha256": _tensor_sha256(null[level]),
            "treatment_statistics_sha256": _tensor_sha256(treatment[level]),
        }
    result = {
        "levels": levels,
        "first_162_exact_across_modes": all(
            value["first_162_exact"] for value in levels.values()
        ),
        "null_contrasts_exact_zero": all(
            value["null_last_243_exact_zero"] for value in levels.values()
        ),
    }
    result["integrity_sha256"] = _canonical_sha256(result)
    if not result["first_162_exact_across_modes"] or not result[
        "null_contrasts_exact_zero"
    ]:
        raise RuntimeError("paired spatial-statistics integrity failed")
    return result


def _gradient_is_finite(model) -> bool:
    return all(
        value.grad is None or bool(torch.isfinite(value.grad).all())
        for value in model.parameters()
    )


def _optimizer_step(optimizer, model) -> int:
    parameters = [value for value in model.parameters() if value.requires_grad]
    observed = []
    missing = 0
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        if "step" in state:
            observed.append(int(state["step"]))
        else:
            missing += 1
    if not observed and missing == len(parameters):
        return 0
    if missing or len(set(observed)) != 1 or len(observed) != len(parameters):
        raise RuntimeError("optimizer state is missing or disagrees across trainable parameters")
    return observed[0]


def paired_optimizer_update(
    models: dict,
    optimizers: dict,
    scalers: dict,
    batch: dict,
    config: dict,
    amp: bool,
) -> dict:
    """Apply one barrier-synchronized update from one shared tensor batch."""
    if list(models) != ["null", "treatment"]:
        raise ValueError("paired arm order must remain null then treatment")
    commitments_before = _input_commitments(batch)
    for name in models:
        optimizers[name].zero_grad(set_to_none=True)
        models[name].train()
    device_type = batch["source_image"].device.type
    outputs, losses = {}, {}
    with torch.amp.autocast(device_type=device_type, enabled=amp):
        for name, model in models.items():
            outputs[name] = model(
                batch["source_image"],
                batch["source_mask"],
                batch["mask_available"],
                batch["candidate_image"],
                batch["candidate_mask"],
                candidate_chunk_size=config["model"]["candidate_chunk_size"],
            )
            losses[name] = atlas_pair_loss(
                outputs[name],
                batch["candidate_pose"],
                batch["true_pose"],
                batch["target_index"],
            )
    for name in models:
        if not bool(torch.isfinite(losses[name]["total"])) or any(
            not bool(torch.isfinite(value).all()) for value in outputs[name].values()
        ):
            raise RuntimeError(f"{name} arm produced nonfinite forward values")
        scalers[name].scale(losses[name]["total"]).backward()

    grad_norms = {}
    scale_before = {}
    step_before = {}
    clip_threshold = float(config["training"]["gradient_clip_norm"])
    for name, model in models.items():
        scale_before[name] = float(scalers[name].get_scale())
        if amp and scale_before[name] != float(
            config["training"]["amp_required_to_stay"]
        ):
            raise RuntimeError(f"{name} AMP scale departed from the frozen value")
        scalers[name].unscale_(optimizers[name])
        if not _gradient_is_finite(model):
            raise RuntimeError(f"{name} arm produced nonfinite unscaled gradients")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"{name} arm produced nonfinite gradient norm")
        grad_norms[name] = float(grad_norm)
        step_before[name] = _optimizer_step(optimizers[name], model)

    # Both arms have crossed every forward/backward/unscale/finite/clip barrier.
    for name in models:
        scalers[name].step(optimizers[name])
    for name in models:
        scalers[name].update()

    step_after = {
        name: _optimizer_step(optimizers[name], model)
        for name, model in models.items()
    }
    scale_after = {name: float(scalers[name].get_scale()) for name in models}
    for name in models:
        if step_after[name] != step_before[name] + 1:
            raise RuntimeError(f"{name} optimizer skipped the paired update")
        if amp and scale_after[name] != float(
            config["training"]["amp_required_to_stay"]
        ):
            raise RuntimeError(f"{name} AMP scale changed during the frozen run")
    commitments_after = _input_commitments(batch)
    if commitments_after != commitments_before:
        raise RuntimeError("a model mutated the shared paired input tensors")
    return {
        "amp_enabled": amp,
        "finite_forward": {"null": True, "treatment": True},
        "finite_unscaled_gradients": {"null": True, "treatment": True},
        "paired_barrier_completed_before_steps": True,
        "loss": {
            arm: {
                name: float(losses[arm][name].detach())
                for name in ("total", "ranking", "auxiliary_ranking", "point")
            }
            for arm in models
        },
        "unscaled_gradient_norm": grad_norms,
        "gradient_clipped": {
            name: grad_norms[name] > clip_threshold for name in models
        },
        "amp_scale_before": scale_before,
        "amp_scale_after": scale_after,
        "optimizer_step_before": step_before,
        "optimizer_step_after": step_after,
        "optimizer_step_applied": {name: True for name in models},
        "input_sha256": commitments_before,
        "paired_input_identity": True,
    }


def _history_is_valid(history: list[dict], update: int, config: dict) -> bool:
    amp_modes = {value.get("amp_enabled") for value in history}
    required_scale = (
        float(config["training"]["amp_required_to_stay"])
        if amp_modes == {True}
        else 1.0
    )
    return (
        len(history) == update
        and [value.get("update") for value in history] == list(range(1, update + 1))
        and all(value.get("paired_input_identity") is True for value in history)
        and amp_modes.issubset({True, False})
        and len(amp_modes) <= 1
        and all(
            value.get("finite_forward") == {"null": True, "treatment": True}
            and value.get("finite_unscaled_gradients")
            == {"null": True, "treatment": True}
            and value.get("paired_barrier_completed_before_steps") is True
            for value in history
        )
        and all(
            value.get("optimizer_step_applied") == {"null": True, "treatment": True}
            for value in history
        )
        and all(
            value.get("optimizer_step_before")
            == {"null": item - 1, "treatment": item - 1}
            for item, value in enumerate(history, 1)
        )
        and all(
            value.get("optimizer_step_after", {}).get(arm) == item
            for item, value in enumerate(history, 1)
            for arm in ("null", "treatment")
        )
        and all(
            value.get("amp_scale_before", {}).get(arm) == required_scale
            and value.get("amp_scale_after", {}).get(arm) == required_scale
            for value in history
            for arm in ("null", "treatment")
        )
        and all(
            set(value.get("loss", {})) == {"null", "treatment"}
            and all(
                set(value["loss"][arm])
                == {"total", "ranking", "auxiliary_ranking", "point"}
                and all(math.isfinite(number) for number in value["loss"][arm].values())
                for arm in ("null", "treatment")
            )
            and set(value.get("unscaled_gradient_norm", {}))
            == {"null", "treatment"}
            and all(
                math.isfinite(value["unscaled_gradient_norm"][arm])
                and value["unscaled_gradient_norm"][arm] >= 0.0
                for arm in ("null", "treatment")
            )
            and set(value.get("gradient_clipped", {}))
            == {"null", "treatment"}
            for value in history
        )
    )


def _resume_payload(
    config: dict,
    lineage: dict,
    models: dict,
    optimizers: dict,
    scalers: dict,
    device: torch.device,
    update: int,
    history: list[dict],
    development: list[dict],
    initialization: dict,
    statistics_integrity: dict,
) -> dict:
    frozen_sources = _assert_frozen_files(config)
    if not _history_is_valid(history, update, config):
        raise RuntimeError("paired history is not a complete update prefix")
    model_state = {name: model.state_dict() for name, model in models.items()}
    optimizer_state = {
        name: optimizer.state_dict() for name, optimizer in optimizers.items()
    }
    scaler_state = {name: scaler.state_dict() for name, scaler in scalers.items()}
    rng_state = _rng_state(device)
    state_commitments = {
        "optimizer_state_sha256": _canonical_sha256(optimizer_state),
        "scaler_state_sha256": _canonical_sha256(scaler_state),
        "rng_state_sha256": _canonical_sha256(rng_state),
        "development_sha256": _canonical_sha256(development),
    }
    return {
        "format": RESUME_FORMAT,
        "purpose": PURPOSE,
        "family_self_sha256": config["family_self_sha256"],
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": frozen_sources,
        "learned_checkpoint_dependencies": [],
        "data_lineage": lineage,
        "initialization": initialization,
        "initial_statistics_integrity": statistics_integrity,
        "update": update,
        "training_history": history,
        "training_history_sha256": _canonical_sha256(history),
        "model": model_state,
        "model_state_sha256": {
            name: _state_dict_sha256(model.state_dict())
            for name, model in models.items()
        },
        "optimizer": optimizer_state,
        "scaler": scaler_state,
        "development": development,
        "rng_state": rng_state,
        "resume_state_commitments": state_commitments,
    }


def _load_joint_resume(
    path: Path,
    config: dict,
    lineage: dict,
    models: dict,
    optimizers: dict,
    scalers: dict,
    device: torch.device,
    initialization: dict,
) -> tuple[int, list[dict], list[dict], dict]:
    frozen_sources = _assert_frozen_files(config)
    state = torch.load(path, map_location="cpu", weights_only=False)
    history = state.get("training_history", [])
    update = int(state.get("update", -1))
    expected_state_keys = {
        "format",
        "purpose",
        "family_self_sha256",
        "config_contract_sha256",
        "config_file_sha256",
        "source_sha256",
        "learned_checkpoint_dependencies",
        "data_lineage",
        "initialization",
        "initial_statistics_integrity",
        "update",
        "training_history",
        "training_history_sha256",
        "model",
        "model_state_sha256",
        "optimizer",
        "scaler",
        "development",
        "rng_state",
        "resume_state_commitments",
    }
    expected_header = {
        "format": RESUME_FORMAT,
        "purpose": PURPOSE,
        "family_self_sha256": config["family_self_sha256"],
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": frozen_sources,
        "learned_checkpoint_dependencies": [],
        "data_lineage": lineage,
        "initialization": initialization,
    }
    if set(state) != expected_state_keys or any(
        state.get(name) != value for name, value in expected_header.items()
    ):
        raise RuntimeError("joint resume state lineage differs")
    if (
        not 0 <= update <= config["training"]["max_updates"]
        or state.get("training_history_sha256") != _canonical_sha256(history)
        or not _history_is_valid(history, update, config)
        or set(state.get("model", {})) != {"null", "treatment"}
        or set(state.get("optimizer", {})) != {"null", "treatment"}
        or set(state.get("scaler", {})) != {"null", "treatment"}
        or set(state.get("model_state_sha256", {})) != {"null", "treatment"}
        or state.get("resume_state_commitments")
        != {
            "optimizer_state_sha256": _canonical_sha256(state.get("optimizer", {})),
            "scaler_state_sha256": _canonical_sha256(state.get("scaler", {})),
            "rng_state_sha256": _canonical_sha256(state.get("rng_state", {})),
            "development_sha256": _canonical_sha256(state.get("development", [])),
        }
        or not state.get("initial_statistics_integrity", {}).get(
            "first_162_exact_across_modes"
        )
        or not state.get("initial_statistics_integrity", {}).get(
            "null_contrasts_exact_zero"
        )
    ):
        raise RuntimeError("joint resume state is incomplete or invalid")
    for name in models:
        if _state_dict_sha256(state["model"][name]) != state[
            "model_state_sha256"
        ][name]:
            raise RuntimeError("joint resume model-state commitment differs")
        models[name].load_state_dict(state["model"][name], strict=True)
        optimizers[name].load_state_dict(state["optimizer"][name])
        scalers[name].load_state_dict(state["scaler"][name])
        if _optimizer_step(optimizers[name], models[name]) != update:
            raise RuntimeError("joint resume optimizer step differs from paired update")
        expected_scale = (
            float(config["training"]["amp_required_to_stay"])
            if config["training"]["amp"] and device.type == "cuda"
            else 1.0
        )
        if float(scalers[name].get_scale()) != expected_scale:
            raise RuntimeError("joint resume scaler differs from the frozen AMP state")
    _restore_rng_state(state["rng_state"], device)
    return (
        update,
        history,
        state.get("development", []),
        state["initial_statistics_integrity"],
    )


def _candidate_table_sha256(
    pose: torch.Tensor, target: int, kinds: list[str]
) -> str:
    return _canonical_sha256(
        {
            "candidate_pose": pose,
            "target_index": int(target),
            "candidate_kind": kinds,
        }
    )


def _truth_rank_margin(energy: torch.Tensor, target: int) -> tuple[int, float]:
    truth = energy[target]
    other = torch.cat((energy[:target], energy[target + 1 :]))
    rank = 1 + int((other < truth).sum())
    return rank, float((other.min() - truth).detach())


def _arm_fixed_row(outputs: dict, row: int, target: int) -> dict:
    normal = outputs["normal"]
    rank, margin = _truth_rank_margin(normal["energy"][row], target)
    top1 = int(normal["energy"][row].argmin())
    result = {
        "normal_energy": normal["energy"][row],
        "normal_energy8": normal["energy8"][row],
        "normal_energy16": normal["energy16"][row],
        "top1_index": top1,
        "top1_correct": top1 == target,
        "truth_rank": rank,
        "truth_margin": margin,
    }
    for mode in ("broken_atlas_binding", "broken_source_pairing"):
        result[f"{mode}_energy"] = outputs[mode]["energy"][row]
        result[f"{mode}_energy8"] = outputs[mode]["energy8"][row]
        result[f"{mode}_energy16"] = outputs[mode]["energy16"][row]
        result[f"{mode}_top1_index"] = int(outputs[mode]["energy"][row].argmin())
        result[f"{mode}_correct"] = result[f"{mode}_top1_index"] == target
    return result


def _order_equivariance(
    model,
    batch: dict,
    output: dict,
    seed: int,
    chunk_size: int,
    atol: float,
    rtol: float,
) -> dict:
    candidates = batch["candidate_image"].shape[1]
    permutation = torch.as_tensor(
        np.random.default_rng(seed).permutation(candidates),
        device=batch["source_image"].device,
    )
    permuted = model(
        batch["source_image"],
        batch["source_mask"],
        batch["mask_available"],
        batch["candidate_image"][:, permutation],
        batch["candidate_mask"][:, permutation],
        candidate_chunk_size=chunk_size,
    )
    inverse = permutation.argsort()
    differences = {
        name: float((permuted[name][:, inverse] - output[name]).abs().max())
        for name in ("energy", "energy8", "energy16")
    }
    allclose = all(
        torch.allclose(
            permuted[name][:, inverse], output[name], atol=atol, rtol=rtol
        )
        for name in ("energy", "energy8", "energy16")
    )
    original = output["energy"].argmin(1)
    reordered = permutation[permuted["energy"].argmin(1)]
    original_pose = batch["candidate_pose"].gather(
        1, original[:, None, None].expand(-1, 1, 3)
    )[:, 0]
    permuted_pose = batch["candidate_pose"][:, permutation].gather(
        1,
        permuted["energy"].argmin(1)[:, None, None].expand(-1, 1, 3),
    )[:, 0]
    return {
        "sample_count": len(batch["source_image"]),
        "maximum_energy_difference": max(differences.values()),
        "per_scale_maximum_difference": differences,
        "energies_allclose": bool(allclose),
        "top1_unchanged": bool(torch.equal(original, reordered)),
        "decoded_pose_maximum_difference": float(
            (original_pose - permuted_pose).abs().max()
        ),
    }


def _pair_key(
    role: str,
    seed: int,
    materialized: dict,
    sample_index: int,
    truth: torch.Tensor,
    candidate_pose: torch.Tensor,
    target: int,
    kinds: list[str],
) -> dict:
    realization = materialized["realization"][sample_index]
    latent_pose_id = hashlib.sha256(
        _canonical_bytes(
            {"panel": materialized["panel_manifest_sha256"], "pose": truth}
        )
    ).hexdigest()
    key = {
        "panel_role": role,
        "panel_seed": int(seed),
        "panel_manifest_sha256": materialized["panel_manifest_sha256"],
        "sample_index": int(sample_index),
        "synthetic_realization_id": realization["synthetic_realization_id"],
        "realization_manifest_sha256": realization["realization_manifest_sha256"],
        "moving_raw_uint8_sha256": realization["moving_raw_uint8_sha256"],
        "source_160x232_sha256": realization["source_160x232_sha256"],
        "latent_pose_id": latent_pose_id,
        "candidate_table_sha256": _candidate_table_sha256(
            candidate_pose, target, kinds
        ),
    }
    key["pair_key_sha256"] = _canonical_sha256(key)
    return key


def exact_mcnemar(null_correct, treatment_correct) -> dict:
    null_correct = np.asarray(null_correct, dtype=bool)
    treatment_correct = np.asarray(treatment_correct, dtype=bool)
    if null_correct.shape != treatment_correct.shape:
        raise ValueError("McNemar inputs must have matching shapes")
    corrected = int((~null_correct & treatment_correct).sum())
    regressed = int((null_correct & ~treatment_correct).sum())
    discordant = corrected + regressed
    if discordant:
        tail = sum(
            math.comb(discordant, item)
            for item in range(min(corrected, regressed) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "null_wrong_treatment_correct": corrected,
        "null_correct_treatment_wrong": regressed,
        "discordant_count": discordant,
        "net_corrections": corrected - regressed,
        "exact_two_sided_p": p_value,
    }


def _paired_fixed_candidate_evaluation(
    models: dict,
    renderer: SyntheticRegistrationGenerator,
    materialized: dict,
    seed: int,
    role: str,
    config: dict,
) -> dict:
    device = renderer.device
    source_all = materialized["source_image"].to(device)
    truth_all = materialized["true_pose"].to(device)
    rng = np.random.default_rng(seed + 101)
    identity = np.arange(len(truth_all))
    truth_numpy = truth_all.detach().cpu().numpy()
    for _ in range(1000):
        source_order = rng.permutation(len(truth_all))
        if np.all(source_order != identity) and np.all(
            np.any(truth_numpy[source_order] != truth_numpy, axis=1)
        ):
            break
    else:
        raise RuntimeError("could not construct paired source derangement")

    counts = {
        arm: {"normal": 0, "broken_atlas_binding": 0, "broken_source_pairing": 0}
        for arm in models
    }
    nonfinite = {arm: 0 for arm in models}
    invalid_render = 0
    raw = []
    order_batches = {arm: [] for arm in models}
    paired_input_batches = []
    batch_size = config["training"]["batch_size"]
    chunk_size = config["model"]["candidate_chunk_size"]
    for start in range(0, len(truth_all), batch_size):
        stop = min(start + batch_size, len(truth_all))
        indices = np.arange(start, stop)
        truth = truth_all[start:stop]
        candidate_pose, target, kinds = candidate_pose_table(truth, seed, indices)
        candidate_image, candidate_mask = render_candidate_poses(
            renderer, candidate_pose
        )
        invalid = (
            ~torch.isfinite(candidate_image).flatten(2).all(2)
            | ~candidate_mask.flatten(2).any(2)
        )
        invalid_render += int(invalid.sum())
        batch = {
            "source_image": source_all[start:stop],
            "source_mask": torch.zeros_like(
                source_all[start:stop], dtype=torch.bool
            ),
            "mask_available": torch.zeros(
                stop - start, 1, 1, 1, device=device
            ),
            "candidate_image": candidate_image,
            "candidate_mask": candidate_mask,
            "candidate_pose": candidate_pose,
        }
        shifted_image = candidate_image.roll(1, 1)
        shifted_mask = candidate_mask.roll(1, 1)
        source_indices = torch.as_tensor(source_order[start:stop], device=device)
        broken_source_image = source_all[source_indices]
        broken_source_mask = torch.zeros_like(
            source_all[start:stop], dtype=torch.bool
        )
        input_before = {
            name: _tensor_sha256(batch[name])
            for name in (
                "source_image",
                "source_mask",
                "mask_available",
                "candidate_image",
                "candidate_mask",
                "candidate_pose",
            )
        }
        input_before.update(
            broken_atlas_image=_tensor_sha256(shifted_image),
            broken_atlas_mask=_tensor_sha256(shifted_mask),
            broken_source_image=_tensor_sha256(broken_source_image),
            broken_source_mask=_tensor_sha256(broken_source_mask),
        )
        outputs = {}
        for arm, model in models.items():
            normal = model(
                batch["source_image"],
                batch["source_mask"],
                batch["mask_available"],
                batch["candidate_image"],
                batch["candidate_mask"],
                candidate_chunk_size=chunk_size,
            )
            broken_atlas = model(
                batch["source_image"],
                batch["source_mask"],
                batch["mask_available"],
                shifted_image,
                shifted_mask,
                candidate_chunk_size=chunk_size,
            )
            broken_source = model(
                broken_source_image,
                broken_source_mask,
                batch["mask_available"],
                batch["candidate_image"],
                batch["candidate_mask"],
                candidate_chunk_size=chunk_size,
            )
            outputs[arm] = {
                "normal": normal,
                "broken_atlas_binding": broken_atlas,
                "broken_source_pairing": broken_source,
            }
            for mode, output in outputs[arm].items():
                counts[arm][mode] += int(
                    (output["energy"].argmin(1) == target).sum()
                )
                nonfinite[arm] += sum(
                    int((~torch.isfinite(value)).sum()) for value in output.values()
                )
            order_batches[arm].append(
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
        input_after = {
            name: _tensor_sha256(batch[name])
            for name in (
                "source_image",
                "source_mask",
                "mask_available",
                "candidate_image",
                "candidate_mask",
                "candidate_pose",
            )
        }
        input_after.update(
            broken_atlas_image=_tensor_sha256(shifted_image),
            broken_atlas_mask=_tensor_sha256(shifted_mask),
            broken_source_image=_tensor_sha256(broken_source_image),
            broken_source_mask=_tensor_sha256(broken_source_mask),
        )
        if input_before != input_after:
            raise RuntimeError("paired evaluator mutated shared fixed-candidate inputs")
        paired_input_batches.append(
            {
                "sample_indices": indices.tolist(),
                "shared_tensor_sha256": input_before,
                "both_arms_received_same_tensor_objects": True,
            }
        )
        for row in range(stop - start):
            target_index = int(target[row])
            arm_rows = {
                arm: _arm_fixed_row(outputs[arm], row, target_index)
                for arm in models
            }
            null_correct = arm_rows["null"]["top1_correct"]
            treatment_correct = arm_rows["treatment"]["top1_correct"]
            transition = (
                "null-wrong-treatment-correct"
                if not null_correct and treatment_correct
                else "null-correct-treatment-wrong"
                if null_correct and not treatment_correct
                else "both-correct"
                if null_correct
                else "both-wrong"
            )
            net_correction = int(treatment_correct) - int(null_correct)
            raw.append(
                {
                    "pair_key": _pair_key(
                        role,
                        seed,
                        materialized,
                        start + row,
                        truth[row],
                        candidate_pose[row],
                        target_index,
                        kinds[row],
                    ),
                    "true_pose": truth[row],
                    "candidate_pose": candidate_pose[row],
                    "candidate_kind": kinds[row],
                    "target_index": target_index,
                    "invalid_render": invalid[row],
                    "null": arm_rows["null"],
                    "treatment": arm_rows["treatment"],
                    "transition": transition,
                    "net_correction": net_correction,
                }
            )

    order = {}
    for arm, batches in order_batches.items():
        order[arm] = {
            "evaluated_sample_count": sum(value["sample_count"] for value in batches),
            "maximum_energy_difference": max(
                value["maximum_energy_difference"] for value in batches
            ),
            "energies_allclose": all(value["energies_allclose"] for value in batches),
            "top1_unchanged": all(value["top1_unchanged"] for value in batches),
            "decoded_pose_maximum_difference": max(
                value["decoded_pose_maximum_difference"] for value in batches
            ),
            "batch_receipts": batches,
        }
    null_correct = [value["null"]["top1_correct"] for value in raw]
    treatment_correct = [value["treatment"]["top1_correct"] for value in raw]
    mcnemar = exact_mcnemar(null_correct, treatment_correct)
    pair_keys = [value["pair_key"]["pair_key_sha256"] for value in raw]
    return {
        "correct": counts,
        "nonfinite_count": nonfinite,
        "invalid_render_count": invalid_render,
        "order_equivariance": order,
        "mcnemar": mcnemar,
        "paired_row_count": len(raw),
        "unique_pair_key_count": len(set(pair_keys)),
        "shared_input_batch_count": len(paired_input_batches),
        "paired_input_identity": True,
        "shared_input_batches": paired_input_batches,
        "raw": raw,
    }


def _arm_free_search_metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    seconds: list[float],
    receipts: list[dict],
    renderer: SyntheticRegistrationGenerator,
) -> dict:
    absolute = (prediction - truth).abs()
    truth_ouv = _pose_to_quicknii_ouv(truth.double())
    prediction_ouv = _pose_to_quicknii_ouv(prediction.double())
    brain_mask = torch_annotation_brain_mask(
        truth_ouv, renderer.annotation, QUICKNII_PIXEL_GRID_SHAPE
    )
    physical = (
        torch_brain_masked_plane_distance(truth_ouv, prediction_ouv, brain_mask)
        * VOXEL_UM
    )
    prior_pose = truth.mean(0, keepdim=True).expand_as(truth)
    prior = (
        torch_brain_masked_plane_distance(
            truth_ouv, _pose_to_quicknii_ouv(prior_pose.double()), brain_mask
        )
        * VOXEL_UM
    )
    scorer_nonfinite = sum(value["nonfinite_count"] for value in receipts)
    invalid = sum(value["invalid_render_count"] for value in receipts)
    metric_nonfinite = sum(
        int((~torch.isfinite(value)).sum())
        for value in (prediction, absolute, physical, prior)
    )
    return {
        "mae": absolute.mean(0),
        "absolute_pose_error": absolute,
        "physical_error_um": physical.mean(),
        "physical_error_um_per_slice": physical,
        "constant_prior_physical_error_um": prior.mean(),
        "constant_prior_physical_error_um_per_slice": prior,
        "physical_improvement_over_constant_prior": 1.0
        - physical.mean() / prior.mean(),
        "seconds_per_slice": seconds,
        "ten_slice_projected_p95_seconds": 10.0
        * float(np.quantile(seconds, 0.95)),
        "predicted_pose": prediction,
        "true_pose": truth,
        "search_receipts": receipts,
        "scorer_nonfinite_count": scorer_nonfinite,
        "metric_nonfinite_count": metric_nonfinite,
        "nonfinite_count": scorer_nonfinite + metric_nonfinite,
        "invalid_render_count": invalid,
    }


def _paired_free_search(
    models: dict,
    renderer: SyntheticRegistrationGenerator,
    materialized: dict,
    fixed: dict,
    config: dict,
) -> dict:
    predictions = {arm: [] for arm in models}
    receipts = {arm: [] for arm in models}
    seconds = {arm: [] for arm in models}
    raw = []
    for item, source_cpu in enumerate(materialized["source_image"]):
        source = source_cpu[None].to(renderer.device)
        mask = torch.zeros_like(source, dtype=torch.bool)
        available = torch.zeros(1, 1, 1, 1, device=renderer.device)
        source_hash = _tensor_sha256(source)
        if source_hash != materialized["realization"][item][
            "source_160x232_sha256"
        ]:
            raise RuntimeError("materialized free-search source differs from its provenance hash")
        arm_order = ("null", "treatment") if item % 2 == 0 else (
            "treatment",
            "null",
        )
        per_arm = {}
        for arm in arm_order:
            if _tensor_sha256(source) != source_hash:
                raise RuntimeError("free-search source changed between paired arms")
            if renderer.device.type == "cuda":
                torch.cuda.synchronize(renderer.device)
            started = time.perf_counter()
            prediction, receipt = coarse_to_fine_search(
                models[arm], renderer, source, mask, available, config
            )
            if renderer.device.type == "cuda":
                torch.cuda.synchronize(renderer.device)
            elapsed = time.perf_counter() - started
            source_key = {
                **fixed["raw"][item]["pair_key"],
                "source_tensor_sha256": source_hash,
            }
            receipt["source_key"] = source_key
            receipt["arm"] = arm
            predictions[arm].append(prediction.detach().cpu())
            receipts[arm].append(receipt)
            seconds[arm].append(elapsed)
            per_arm[arm] = {
                "predicted_pose": prediction.detach().cpu(),
                "absolute_pose_error": (
                    prediction.detach().cpu() - materialized["true_pose"][item]
                ).abs(),
                "seconds": elapsed,
                "search_receipt": receipt,
            }
        raw.append(
            {
                "source_key": {
                    **fixed["raw"][item]["pair_key"],
                    "source_tensor_sha256": source_hash,
                },
                "true_pose": materialized["true_pose"][item],
                "null": per_arm["null"],
                "treatment": per_arm["treatment"],
            }
        )
    truth = materialized["true_pose"].to(renderer.device)
    result = {"raw": raw, "paired_source_identity": True}
    for arm in models:
        prediction = torch.stack(predictions[arm]).to(renderer.device)
        result[arm] = _arm_free_search_metrics(
            prediction, truth, seconds[arm], receipts[arm], renderer
        )
        for item, row in enumerate(raw):
            row[arm]["physical_error_um"] = result[arm][
                "physical_error_um_per_slice"
            ][item]
            row[arm]["constant_prior_physical_error_um"] = result[arm][
                "constant_prior_physical_error_um_per_slice"
            ][item]
    return result


def paired_evaluate_panel(
    models: dict,
    renderer: SyntheticRegistrationGenerator,
    materialized: dict,
    seed: int,
    role: str,
    config: dict,
    *,
    free_search: bool,
) -> dict:
    modes = {name: model.training for name, model in models.items()}
    for model in models.values():
        model.eval()
    with torch.no_grad():
        fixed = _paired_fixed_candidate_evaluation(
            models, renderer, materialized, seed, role, config
        )
        result = {
            "panel_role": role,
            "seed": int(seed),
            "sample_count": len(materialized["true_pose"]),
            "panel_manifest_sha256": materialized["panel_manifest_sha256"],
            "generator_contract_sha256": materialized[
                "generator_contract_sha256"
            ],
            "atlas_sha256": materialized["atlas_sha256"],
            "source_realizations": materialized["realization"],
            "fixed_candidates": fixed,
        }
        if free_search:
            result["free_search"] = _paired_free_search(
                models, renderer, materialized, fixed, config
            )
    for name, model in models.items():
        model.train(modes[name])
    result["result_sha256"] = _canonical_sha256(result)
    return result


def _absolute_arm_status(
    result: dict, arm: str, config: dict, *, require_search: bool
) -> dict:
    gates = config["gates"]
    fixed = result["fixed_candidates"]
    order = fixed["order_equivariance"][arm]
    search = result.get("free_search", {}).get(arm, {})
    nonfinite = fixed["nonfinite_count"][arm] + search.get("nonfinite_count", 0)
    invalid = fixed["invalid_render_count"] + search.get(
        "invalid_render_count", 0
    )
    checks = {
        "nonfinite": nonfinite <= gates["nonfinite_count_maximum"],
        "invalid_render": invalid <= gates["invalid_render_count_maximum"],
        "truth_in_set": fixed["correct"][arm]["normal"]
        >= gates["truth_in_set_correct_minimum_per_48"],
        "broken_atlas_binding": fixed["correct"][arm]["broken_atlas_binding"]
        <= gates["broken_pair_correct_maximum_per_48"],
        "broken_source_pairing": fixed["correct"][arm]["broken_source_pairing"]
        <= gates["broken_pair_correct_maximum_per_48"],
        "order_equivariance": (
            order["top1_unchanged"]
            and order["energies_allclose"]
            and order["maximum_energy_difference"] <= gates["order_energy_atol"]
            and order["decoded_pose_maximum_difference"]
            <= gates["order_energy_atol"]
            and order["evaluated_sample_count"] == result["sample_count"]
        ),
    }
    if require_search:
        checks.update(
            ap_mae=float(search["mae"][0]) <= gates["ap_mae_um_maximum"],
            lr_mae=float(search["mae"][1]) <= gates["lr_mae_deg_maximum"],
            dv_mae=float(search["mae"][2]) <= gates["dv_mae_deg_maximum"],
            physical_improvement=float(
                search["physical_improvement_over_constant_prior"]
            )
            >= gates["physical_improvement_over_constant_prior_minimum"],
            runtime=float(search["ten_slice_projected_p95_seconds"])
            <= gates["ten_slice_p95_seconds_maximum"],
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "status_sha256": _canonical_sha256(checks),
    }


def paired_panel_status(
    result: dict, config: dict, *, require_search: bool
) -> dict:
    fixed = result["fixed_candidates"]
    causal_gates = config["gates"]["causal"]
    arm_status = {
        arm: _absolute_arm_status(result, arm, config, require_search=require_search)
        for arm in ("null", "treatment")
    }
    treatment_fixed_status = _absolute_arm_status(
        result, "treatment", config, require_search=False
    )
    pair_keys = [
        value["pair_key"]["pair_key_sha256"] for value in fixed["raw"]
    ]
    try:
        recomputed_counts = {
            arm: {
                "normal": sum(bool(value[arm]["top1_correct"]) for value in fixed["raw"]),
                "broken_atlas_binding": sum(
                    bool(value[arm]["broken_atlas_binding_correct"])
                    for value in fixed["raw"]
                ),
                "broken_source_pairing": sum(
                    bool(value[arm]["broken_source_pairing_correct"])
                    for value in fixed["raw"]
                ),
            }
            for arm in ("null", "treatment")
        }
        null_flags = [bool(value["null"]["top1_correct"]) for value in fixed["raw"]]
        treatment_flags = [
            bool(value["treatment"]["top1_correct"]) for value in fixed["raw"]
        ]
        recomputed_mcnemar = exact_mcnemar(null_flags, treatment_flags)
        transitions_exact = all(
            value.get("net_correction")
            == int(value["treatment"]["top1_correct"])
            - int(value["null"]["top1_correct"])
            and value.get("transition")
            == (
                "null-wrong-treatment-correct"
                if not value["null"]["top1_correct"]
                and value["treatment"]["top1_correct"]
                else "null-correct-treatment-wrong"
                if value["null"]["top1_correct"]
                and not value["treatment"]["top1_correct"]
                else "both-correct"
                if value["null"]["top1_correct"]
                else "both-wrong"
            )
            for value in fixed["raw"]
        )
        fixed_statistics_exact = (
            recomputed_counts == fixed["correct"]
            and recomputed_mcnemar == fixed["mcnemar"]
            and transitions_exact
            and sum(value["net_correction"] for value in fixed["raw"])
            == recomputed_mcnemar["net_corrections"]
        )
    except (KeyError, TypeError, ValueError):
        fixed_statistics_exact = False
    if require_search:
        free = result.get("free_search", {})
        free_raw = free.get("raw", [])
        free_search_complete = (
            len(free_raw) == result["sample_count"]
            and all(
                set(value) == {"source_key", "true_pose", "null", "treatment"}
                and value["source_key"].get("pair_key_sha256") == pair_keys[item]
                for item, value in enumerate(free_raw)
            )
            and all(
                len(free.get(arm, {}).get("predicted_pose", []))
                == result["sample_count"]
                and len(free.get(arm, {}).get("absolute_pose_error", []))
                == result["sample_count"]
                and len(free.get(arm, {}).get("seconds_per_slice", []))
                == result["sample_count"]
                and len(free.get(arm, {}).get("search_receipts", []))
                == result["sample_count"]
                and all(
                    receipt.get("source_key", {}).get("pair_key_sha256")
                    == pair_keys[item]
                    for item, receipt in enumerate(
                        free.get(arm, {}).get("search_receipts", [])
                    )
                )
                for arm in ("null", "treatment")
            )
        )
    else:
        free_search_complete = True
    integrity_checks = {
        "paired_row_count": fixed["paired_row_count"]
        == causal_gates["paired_rows_per_seed"],
        "unique_pair_keys": fixed["unique_pair_key_count"]
        == causal_gates["paired_rows_per_seed"],
        "raw_pair_keys_complete": len(pair_keys)
        == len(set(pair_keys))
        == causal_gates["paired_rows_per_seed"],
        "paired_fixed_input_identity": fixed["paired_input_identity"] is True,
        "fixed_statistics_recomputed": fixed_statistics_exact,
        "paired_free_search_source_identity": (
            not require_search
            or result["free_search"]["paired_source_identity"] is True
        ),
        "no_selective_missing_free_search": free_search_complete,
        "nonfinite": all(
            value["checks"]["nonfinite"] for value in arm_status.values()
        ),
        "invalid_render": all(
            value["checks"]["invalid_render"] for value in arm_status.values()
        ),
    }
    null_checks = arm_status["null"]["checks"]
    null_control_integrity_runtime_checks = {
        name: null_checks[name]
        for name in (
            "nonfinite",
            "invalid_render",
            "broken_atlas_binding",
            "broken_source_pairing",
            "order_equivariance",
        )
    }
    if require_search:
        null_control_integrity_runtime_checks["runtime"] = null_checks["runtime"]
    null_control_integrity_runtime_passed = all(
        null_control_integrity_runtime_checks.values()
    )
    mcnemar = fixed["mcnemar"]
    causal_checks = {
        "pairing_integrity": all(integrity_checks.values()),
        "null_control_integrity_runtime": null_control_integrity_runtime_passed,
        "treatment_passes_46": fixed["correct"]["treatment"]["normal"]
        >= causal_gates["treatment_truth_in_set_minimum"],
        "null_fails_46": fixed["correct"]["null"]["normal"]
        <= causal_gates["null_truth_in_set_maximum"],
        "net_corrections": mcnemar["net_corrections"]
        >= causal_gates["net_corrections_minimum"],
        "mcnemar": mcnemar["exact_two_sided_p"]
        <= causal_gates["exact_two_sided_mcnemar_p_maximum"],
    }
    causal_passed = all(causal_checks.values())
    if not all(integrity_checks.values()):
        branch = "integrity-failure-invalid-stop"
    elif arm_status["null"]["passed"] and arm_status["treatment"]["passed"]:
        branch = "both-pass-no-spatial-necessity-claim"
    elif (
        arm_status["treatment"]["passed"]
        and not arm_status["null"]["passed"]
        and causal_passed
    ):
        branch = "causal-rescue-only-authorize-independent-confirmation"
    elif (
        causal_passed
        and treatment_fixed_status["passed"]
        and not arm_status["treatment"]["passed"]
    ):
        branch = "local-fixed-haar-mechanism-supported-end-to-end-no-go"
    elif not arm_status["null"]["passed"] and not arm_status["treatment"]["passed"]:
        branch = "both-fail-insufficient-stop"
    else:
        branch = "seed-fail-no-causal-rescue"
    payload = {
        "passed": arm_status["treatment"]["passed"] and causal_passed,
        "arm_absolute": arm_status,
        "treatment_fixed_panel": treatment_fixed_status,
        "integrity": {"passed": all(integrity_checks.values()), "checks": integrity_checks},
        "null_control_integrity_runtime": {
            "passed": null_control_integrity_runtime_passed,
            "checks": null_control_integrity_runtime_checks,
        },
        "causal": {"passed": causal_passed, "checks": causal_checks},
        "interpretation_branch": branch,
    }
    payload["status_sha256"] = _canonical_sha256(payload)
    return payload


def _training_integrity(
    history: list[dict],
    config: dict,
    initialization: dict,
    statistics_integrity: dict,
    final_resume_equal: bool,
) -> dict:
    maximum = config["training"]["max_updates"]
    required_scale = float(config["training"]["amp_required_to_stay"])
    checks = {
        "full_initial_state_equal": initialization.get("full_initial_state_equal")
        is True,
        "first_162_statistics_exact": statistics_integrity.get(
            "first_162_exact_across_modes"
        )
        is True,
        "null_contrasts_exact_zero": statistics_integrity.get(
            "null_contrasts_exact_zero"
        )
        is True,
        "paired_input_identity_all_updates": len(history) == maximum
        and all(value.get("paired_input_identity") is True for value in history),
        "finite_forward_all_updates": len(history) == maximum
        and all(
            value.get("finite_forward") == {"null": True, "treatment": True}
            for value in history
        ),
        "finite_unscaled_gradients_all_updates": len(history) == maximum
        and all(
            value.get("finite_unscaled_gradients")
            == {"null": True, "treatment": True}
            for value in history
        ),
        "paired_barrier_all_updates": len(history) == maximum
        and all(
            value.get("paired_barrier_completed_before_steps") is True
            for value in history
        ),
        "steps_exact_1_through_1500": [value.get("update") for value in history]
        == list(range(1, maximum + 1))
        and all(
            value.get("optimizer_step_after")
            == {"null": item, "treatment": item}
            for item, value in enumerate(history, 1)
        ),
        "amp_scale_exact_512": len(history) == maximum
        and all(value.get("amp_enabled") is True for value in history)
        and all(
            value.get("amp_scale_before")
            == {"null": required_scale, "treatment": required_scale}
            and value.get("amp_scale_after")
            == {"null": required_scale, "treatment": required_scale}
            for value in history
        ),
        "final_equals_resume": final_resume_equal,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "clipping_recorded_without_fraction_gate": True,
        "clipped_update_count": {
            arm: sum(bool(value["gradient_clipped"][arm]) for value in history)
            for arm in ("null", "treatment")
        },
        "integrity_sha256": _canonical_sha256(checks),
    }


def _final_payload(
    config: dict,
    lineage: dict,
    models: dict,
    history: list[dict],
    development: list[dict],
    initialization: dict,
    statistics_integrity: dict,
    resume_path: Path,
) -> dict:
    frozen_sources = _assert_frozen_files(config)
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    current_hashes = {
        name: _state_dict_sha256(model.state_dict()) for name, model in models.items()
    }
    final_resume_equal = (
        resume.get("update") == config["training"]["max_updates"]
        and resume.get("training_history_sha256") == _canonical_sha256(history)
        and resume.get("model_state_sha256") == current_hashes
        and all(
            _state_dict_sha256(resume["model"][name]) == current_hashes[name]
            for name in models
        )
    )
    integrity = _training_integrity(
        history,
        config,
        initialization,
        statistics_integrity,
        final_resume_equal,
    )
    if not integrity["passed"]:
        raise RuntimeError("paired training integrity failed before finalization")
    return {
        "format": FINAL_FORMAT,
        "purpose": PURPOSE,
        "family_self_sha256": config["family_self_sha256"],
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": frozen_sources,
        "learned_checkpoint_dependencies": [],
        "data_lineage": lineage,
        "initialization": initialization,
        "initial_statistics_integrity": statistics_integrity,
        "update": config["training"]["max_updates"],
        "training_history": history,
        "training_history_sha256": _canonical_sha256(history),
        "development": development,
        "model": {name: model.state_dict() for name, model in models.items()},
        "model_state_sha256": current_hashes,
        "resume_file_sha256": _sha256(resume_path),
        "resume_state_commitments": resume["resume_state_commitments"],
        "training_integrity": integrity,
        "joint_final_state_complete": True,
    }


def _verify_joint_final(
    config: dict, final_path: Path, resume_path: Path | None = None
) -> dict:
    frozen_sources = _assert_frozen_files(config)
    if not final_path.is_file():
        raise RuntimeError("joint freeze requires the paired final checkpoint")
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    history = final.get("training_history", [])
    lineage = final.get("data_lineage", {})
    lineage_payload = {
        name: value for name, value in lineage.items() if name != "data_lineage_sha256"
    }
    lineage_valid = lineage.get("data_lineage_sha256") == _canonical_sha256(
        lineage_payload
    )
    statistics_integrity = final.get("initial_statistics_integrity", {})
    statistics_commitment = statistics_integrity.get("integrity_sha256")
    statistics_payload = {
        name: value
        for name, value in statistics_integrity.items()
        if name != "integrity_sha256"
    }
    initialization = final.get("initialization", {})
    expected_final_keys = {
        "format",
        "purpose",
        "family_self_sha256",
        "config_contract_sha256",
        "config_file_sha256",
        "source_sha256",
        "learned_checkpoint_dependencies",
        "data_lineage",
        "initialization",
        "initial_statistics_integrity",
        "update",
        "training_history",
        "training_history_sha256",
        "development",
        "model",
        "model_state_sha256",
        "resume_file_sha256",
        "resume_state_commitments",
        "training_integrity",
        "joint_final_state_complete",
    }
    if (
        set(final) != expected_final_keys
        or final.get("format") != FINAL_FORMAT
        or final.get("purpose") != PURPOSE
        or final.get("family_self_sha256") != config["family_self_sha256"]
        or final.get("config_contract_sha256") != config["contract_sha256"]
        or final.get("config_file_sha256") != config["config_file_sha256"]
        or final.get("source_sha256") != frozen_sources
        or final.get("learned_checkpoint_dependencies") != []
        or final.get("update") != config["training"]["max_updates"]
        or final.get("joint_final_state_complete") is not True
        or set(final.get("model", {})) != {"null", "treatment"}
        or set(final.get("model_state_sha256", {})) != {"null", "treatment"}
        or not all(
            _state_dict_sha256(final["model"][name])
            == final["model_state_sha256"][name]
            for name in ("null", "treatment")
        )
        or final.get("training_history_sha256") != _canonical_sha256(history)
        or not _history_is_valid(history, len(history), config)
        or len(history) != config["training"]["max_updates"]
        or not lineage_valid
        or lineage.get("family_self_sha256") != config["family_self_sha256"]
        or lineage.get("config_contract_sha256") != config["contract_sha256"]
        or lineage.get("config_file_sha256") != config["config_file_sha256"]
        or lineage.get("source_sha256") != frozen_sources
        or lineage.get("learned_checkpoint_dependencies") != []
        or lineage.get("train_seed") != config["data"]["train_seed"]
        or lineage.get("development_seed") != config["data"]["development_seed"]
        or lineage.get("reserved_fresh_qualification_seeds")
        != config["data"]["qualification_seeds"]
        or lineage.get("consumed_or_forbidden_qualification_seeds")
        != config["data"]["consumed_or_forbidden_qualification_seeds"]
        or statistics_commitment != _canonical_sha256(statistics_payload)
        or statistics_integrity.get("first_162_exact_across_modes") is not True
        or statistics_integrity.get("null_contrasts_exact_zero") is not True
        or initialization.get("full_initial_state_equal") is not True
        or initialization.get("null_state_sha256")
        != initialization.get("treatment_state_sha256")
        or initialization.get("state_schema") != list(final["model"]["null"])
        or list(final["model"]["null"]) != list(final["model"]["treatment"])
    ):
        raise RuntimeError("paired final checkpoint is not the committed joint state")
    final_resume_equal = False
    if resume_path is not None:
        if not resume_path.is_file() or _sha256(resume_path) != final.get(
            "resume_file_sha256"
        ):
            raise RuntimeError("paired final checkpoint no longer matches joint resume")
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        optimizer_state_integrity = {}
        for arm in ("null", "treatment"):
            optimizer_state = resume.get("optimizer", {}).get(arm, {})
            parameter_ids = [
                value
                for group in optimizer_state.get("param_groups", [])
                for value in group.get("params", [])
            ]
            state = optimizer_state.get("state", {})
            steps = {
                int(value["step"])
                for value in state.values()
                if "step" in value
            }
            optimizer_state_integrity[arm] = (
                len(parameter_ids) == len(set(parameter_ids))
                == len(final["model"][arm])
                and set(parameter_ids) == set(state)
                and all("step" in value for value in state.values())
                and steps == {config["training"]["max_updates"]}
            )
        scaler_scales = {
            arm: float(resume.get("scaler", {}).get(arm, {}).get("scale", float("nan")))
            for arm in ("null", "treatment")
        }
        expected_resume_keys = {
            "format",
            "purpose",
            "family_self_sha256",
            "config_contract_sha256",
            "config_file_sha256",
            "source_sha256",
            "learned_checkpoint_dependencies",
            "data_lineage",
            "initialization",
            "initial_statistics_integrity",
            "update",
            "training_history",
            "training_history_sha256",
            "model",
            "model_state_sha256",
            "optimizer",
            "scaler",
            "development",
            "rng_state",
            "resume_state_commitments",
        }
        observed_resume_commitments = {
            "optimizer_state_sha256": _canonical_sha256(
                resume.get("optimizer", {})
            ),
            "scaler_state_sha256": _canonical_sha256(resume.get("scaler", {})),
            "rng_state_sha256": _canonical_sha256(resume.get("rng_state", {})),
            "development_sha256": _canonical_sha256(
                resume.get("development", [])
            ),
        }
        if (
            set(resume) != expected_resume_keys
            or resume.get("format") != RESUME_FORMAT
            or resume.get("purpose") != PURPOSE
            or resume.get("family_self_sha256") != config["family_self_sha256"]
            or resume.get("config_contract_sha256") != config["contract_sha256"]
            or resume.get("config_file_sha256") != config["config_file_sha256"]
            or resume.get("source_sha256") != frozen_sources
            or resume.get("learned_checkpoint_dependencies") != []
            or resume.get("data_lineage") != lineage
            or resume.get("initialization") != initialization
            or resume.get("initial_statistics_integrity")
            != statistics_integrity
            or resume.get("update") != final["update"]
            or resume.get("training_history") != history
            or resume.get("training_history_sha256")
            != _canonical_sha256(resume.get("training_history", []))
            or resume.get("training_history_sha256")
            != final["training_history_sha256"]
            or _canonical_sha256(resume.get("development", []))
            != _canonical_sha256(final.get("development", []))
            or set(resume.get("rng_state", {}))
            != {"python", "numpy", "torch_cpu", "torch_cuda"}
            or set(resume.get("model", {})) != {"null", "treatment"}
            or set(resume.get("model_state_sha256", {}))
            != {"null", "treatment"}
            or set(resume.get("optimizer", {})) != {"null", "treatment"}
            or set(resume.get("scaler", {})) != {"null", "treatment"}
            or resume.get("resume_state_commitments")
            != observed_resume_commitments
            or final.get("resume_state_commitments")
            != observed_resume_commitments
            or resume.get("model_state_sha256") != final["model_state_sha256"]
            or any(
                _state_dict_sha256(resume["model"][name])
                != final["model_state_sha256"][name]
                for name in ("null", "treatment")
            )
            or optimizer_state_integrity != {"null": True, "treatment": True}
            or scaler_scales
            != {
                "null": float(config["training"]["amp_required_to_stay"]),
                "treatment": float(config["training"]["amp_required_to_stay"]),
            }
        ):
            raise RuntimeError("joint final states differ from the persisted resume states")
        final_resume_equal = True
    expected_integrity = _training_integrity(
        history,
        config,
        initialization,
        statistics_integrity,
        final_resume_equal,
    )
    if final.get("training_integrity") != expected_integrity or not expected_integrity[
        "passed"
    ]:
        raise RuntimeError("paired final training-integrity payload differs on recomputation")
    return final


def _freeze_payload(
    config: dict, final_path: Path, resume_path: Path, final: dict
) -> dict:
    frozen_sources = _assert_frozen_files(config)
    lineage = final["data_lineage"]
    return {
        "format": FREEZE_FORMAT,
        "purpose": PURPOSE,
        "family_self_sha256": config["family_self_sha256"],
        "source_sha256": frozen_sources,
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "joint_final_file": final_path.name,
        "joint_final_file_sha256": _sha256(final_path),
        "joint_resume_file_sha256": _sha256(resume_path),
        "resume_state_commitments": final["resume_state_commitments"],
        "both_model_state_sha256": final["model_state_sha256"],
        "training_history_sha256": final["training_history_sha256"],
        "data_lineage_sha256": lineage["data_lineage_sha256"],
        "generator_contract_sha256": lineage["generator_contract_sha256"],
        "atlas_sha256": lineage["atlas_sha256"],
        "final_update": config["training"]["max_updates"],
        "qualification_seeds": config["data"]["qualification_seeds"],
        "qualification_count_per_seed": config["data"][
            "qualification_count_per_seed"
        ],
        "both_final_states_jointly_frozen": True,
        "learned_checkpoint_dependencies": [],
    }


def freeze_qualification(
    run_folder: Path, config: dict, final_path: Path, resume_path: Path
) -> Path:
    final = _verify_joint_final(config, final_path, resume_path)
    payload = _freeze_payload(config, final_path, resume_path, final)
    payload["freeze_payload_sha256"] = _canonical_sha256(payload)
    path = run_folder / "qualification_joint_freeze.json"
    _write_immutable_json(path, payload)
    return path


def verified_qualification_capability(
    run_folder: Path, config: dict, final_path: Path, resume_path: Path
) -> dict:
    final = _verify_joint_final(config, final_path, resume_path)
    freeze_path = run_folder / "qualification_joint_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("qualification tensors are forbidden before both final states freeze")
    observed = json.loads(freeze_path.read_text(encoding="utf-8"))
    commitment = observed.pop("freeze_payload_sha256", None)
    expected = _freeze_payload(config, final_path, resume_path, final)
    if observed != _canonical(expected) or commitment != _canonical_sha256(expected):
        raise RuntimeError("joint qualification freeze no longer matches its bindings")
    return {
        "guard": _QUALIFICATION_CAPABILITY,
        "freeze_payload_sha256": commitment,
        "source_sha256": expected["source_sha256"],
        "family_self_sha256": expected["family_self_sha256"],
        "config_contract_sha256": expected["config_contract_sha256"],
        "joint_final_file_sha256": expected["joint_final_file_sha256"],
        "both_model_state_sha256": expected["both_model_state_sha256"],
        "generator_contract_sha256": expected["generator_contract_sha256"],
        "data_lineage_sha256": expected["data_lineage_sha256"],
        "qualification_seeds": tuple(expected["qualification_seeds"]),
        "qualification_count_per_seed": expected["qualification_count_per_seed"],
    }


def qualification_manifests(
    generator: SyntheticRegistrationGenerator,
    config: dict,
    run_folder: Path,
    final_path: Path,
    resume_path: Path,
) -> list[dict]:
    frozen_sources = _assert_frozen_files(config)
    capability = verified_qualification_capability(
        run_folder, config, final_path, resume_path
    )
    if (
        capability.get("guard") is not _QUALIFICATION_CAPABILITY
        or capability.get("generator_contract_sha256")
        != generator.contract["contract_sha256"]
        or capability.get("source_sha256") != frozen_sources
        or capability.get("config_contract_sha256") != config["contract_sha256"]
    ):
        raise RuntimeError("qualification generation lacks the joint capability")
    return [
        balanced_panel_manifest(
            generator,
            seed,
            config["data"]["qualification_count_per_seed"],
            qualification_capability=capability,
        )
        for seed in config["data"]["qualification_seeds"]
    ]


def family_status(qualification: list[dict], training_integrity: dict) -> dict:
    seeds = [int(value["seed"]) for value in qualification]
    statuses = [value["status"] for value in qualification]
    exact_rows = sum(
        value["fixed_candidates"]["paired_row_count"] for value in qualification
    )
    exact_free_rows = sum(
        len(value.get("free_search", {}).get("raw", []))
        for value in qualification
    )
    fixed_pair_keys = [
        row["pair_key"]["pair_key_sha256"]
        for panel in qualification
        for row in panel.get("fixed_candidates", {}).get("raw", [])
    ]
    free_pair_keys = [
        row["source_key"]["pair_key_sha256"]
        for panel in qualification
        for row in panel.get("free_search", {}).get("raw", [])
    ]
    integrity = (
        training_integrity.get("passed") is True
        and seeds == list(QUALIFICATION_SEEDS)
        and exact_rows == 96
        and exact_free_rows == 96
        and len(set(fixed_pair_keys)) == 96
        and free_pair_keys == fixed_pair_keys
        and all(value["integrity"]["passed"] for value in statuses)
    )
    full_pass = integrity and all(
        value["arm_absolute"]["treatment"]["passed"]
        and value["causal"]["passed"]
        for value in statuses
    )
    if not integrity:
        branch = "integrity-failure-invalid-stop"
    elif full_pass:
        branch = "causal-rescue-only-authorize-independent-confirmation"
    elif all(
        value["arm_absolute"]["null"]["passed"]
        and value["arm_absolute"]["treatment"]["passed"]
        for value in statuses
    ):
        branch = "both-pass-no-spatial-necessity-claim"
    elif all(
        value.get("interpretation_branch")
        == "local-fixed-haar-mechanism-supported-end-to-end-no-go"
        and value["treatment_fixed_panel"]["passed"]
        for value in statuses
    ):
        branch = "local-fixed-haar-mechanism-supported-end-to-end-no-go"
    elif sum(value["passed"] for value in statuses) == 1:
        branch = "one-seed-pass-family-fail"
    elif all(
        not value["arm_absolute"]["null"]["passed"]
        and not value["arm_absolute"]["treatment"]["passed"]
        for value in statuses
    ):
        branch = "both-fail-insufficient-stop"
    else:
        branch = "family-fail-no-causal-rescue"
    result = {
        "passed": full_pass,
        "integrity_passed": integrity,
        "qualification_seeds_in_frozen_order": seeds,
        "paired_qualification_rows": exact_rows,
        "paired_free_search_rows": exact_free_rows,
        "unique_family_pair_keys": len(set(fixed_pair_keys)),
        "per_seed_pass": {
            str(value["seed"]): value["status"]["passed"] for value in qualification
        },
        "interpretation_branch": branch,
        "independent_confirmation_authorized": full_pass,
        "protected_data_access_authorized": False,
        "promotion_authorized": False,
        "pooled_96_is_descriptive_only": True,
    }
    result["family_status_sha256"] = _canonical_sha256(result)
    return result


def _trajectory_receipt(
    path: Path,
    config: dict,
    history: list[dict],
    final_path: Path,
    resume_path: Path,
) -> Path:
    frozen_sources = _assert_frozen_files(config)
    payload = {
        "format": "independent-atlas-pair-spatial-aggregation-paired-trajectory-v1",
        "purpose": PURPOSE,
        "family_self_sha256": config["family_self_sha256"],
        "source_sha256": frozen_sources,
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "joint_final_file_sha256": _sha256(final_path),
        "joint_resume_file_sha256": _sha256(resume_path),
        "applied_paired_updates": len(history),
        "arm_presentations": 2 * len(history),
        "training_history_sha256": _canonical_sha256(history),
        "training_history": history,
    }
    payload["trajectory_file_payload_sha256"] = _canonical_sha256(payload)
    _write_immutable_json(path, payload)
    return path


def _qualification_result_semantics_valid(result: dict, config: dict) -> bool:
    expected_keys = {
        "panel_role",
        "seed",
        "sample_count",
        "panel_manifest_sha256",
        "generator_contract_sha256",
        "atlas_sha256",
        "source_realizations",
        "fixed_candidates",
        "free_search",
        "result_sha256",
        "status",
        "lineage",
        "qualification_file_payload_sha256",
    }
    if set(result) != expected_keys:
        return False
    core = {
        name: value
        for name, value in result.items()
        if name
        not in {
            "result_sha256",
            "status",
            "lineage",
            "qualification_file_payload_sha256",
        }
    }
    uncommitted = {
        name: value
        for name, value in result.items()
        if name != "qualification_file_payload_sha256"
    }
    return (
        result.get("panel_role") == "qualification"
        and result.get("seed") in QUALIFICATION_SEEDS
        and result.get("sample_count") == 48
        and result.get("result_sha256") == _canonical_sha256(core)
        and result.get("qualification_file_payload_sha256")
        == _canonical_sha256(uncommitted)
        and result.get("status")
        == paired_panel_status(result, config, require_search=True)
    )


def _existing_receipt_is_valid(path: Path, config: dict, final_path: Path) -> bool:
    frozen_sources = _assert_frozen_files(config)
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text(encoding="utf-8"))
    commitment = receipt.pop("receipt_sha256", None)
    if commitment != _canonical_sha256(receipt):
        raise RuntimeError("existing paired receipt commitment differs")
    if (
        receipt.get("format") != RECEIPT_FORMAT
        or receipt.get("purpose") != PURPOSE
        or receipt.get("role") != ROLE
        or receipt.get("family_self_sha256") != config["family_self_sha256"]
        or receipt.get("config_contract_sha256") != config["contract_sha256"]
        or receipt.get("config_file_sha256") != config["config_file_sha256"]
        or receipt.get("source_sha256") != frozen_sources
        or receipt.get("joint_final_file_sha256") != _sha256(final_path)
        or receipt.get("learned_checkpoint_dependencies") != []
        or receipt.get("product5_access") is not False
        or receipt.get("calibration_access") is not False
        or receipt.get("final_test_access") is not False
        or [value.get("seed") for value in receipt.get("qualification", [])]
        != list(QUALIFICATION_SEEDS)
        or not all(
            _qualification_result_semantics_valid(value, config)
            for value in receipt.get("qualification", [])
        )
        or receipt.get("family_status")
        != family_status(
            receipt.get("qualification", []), receipt.get("training_integrity", {})
        )
    ):
        raise RuntimeError("existing paired receipt lineage or statistics differ")
    return True


def run(config_path: str | Path) -> Path:
    config = inspect_config(config_path)
    frozen_sources = _assert_frozen_files(config)
    device = _device(config)
    if config["training"]["amp"] and device.type != "cuda":
        raise RuntimeError(
            "the frozen paired family requires CUDA AMP with scale 512; CUDA is unavailable"
        )
    run_root = Path(
        os.environ.get(
            config["paths"]["run_root_env"], ROOT / "training-runs"
        )
    )
    run_folder = run_root / config["name"]
    run_folder.mkdir(parents=True, exist_ok=True)
    generator = SyntheticRegistrationGenerator(
        ROOT / config["paths"]["atlas_repo_relative"], device
    )
    train_manifest = training_manifest(generator, config)
    development_manifest = balanced_panel_manifest(
        generator,
        config["data"]["development_seed"],
        config["data"]["development_count"],
    )
    lineage = data_lineage(
        generator, train_manifest, development_manifest, config
    )
    models, initialization = initialize_pair(config, device)
    optimizers, scalers, amp = _optimizers_and_scalers(models, config, device)
    if not amp:
        raise RuntimeError("the frozen paired family did not enter CUDA AMP")

    setup = {
        "format": "independent-atlas-pair-spatial-aggregation-paired-setup-v1",
        "purpose": PURPOSE,
        "role": ROLE,
        "family_self_sha256": config["family_self_sha256"],
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": frozen_sources,
        "data_lineage": lineage,
        "initialization": initialization,
        "learned_checkpoint_dependencies": [],
    }
    setup["setup_payload_sha256"] = _canonical_sha256(setup)
    _write_immutable_json(run_folder / "setup_receipt.json", setup)
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
        development_manifest,
        generator,
        config,
    )

    resume_path = run_folder / "joint_resume_state.pt"
    update = 0
    history = []
    development = []
    statistics_integrity = None
    if resume_path.exists():
        update, history, development, statistics_integrity = _load_joint_resume(
            resume_path,
            config,
            lineage,
            models,
            optimizers,
            scalers,
            device,
            initialization,
        )

    source_cache = {}
    while update < config["training"]["max_updates"]:
        batch = _training_batch(
            update, generator, train_manifest, config, source_cache
        )
        if statistics_integrity is None:
            statistics_integrity = initial_statistics_integrity(models, batch)
        step = paired_optimizer_update(
            models, optimizers, scalers, batch, config, amp
        )
        update += 1
        history.append(
            {
                "update": update,
                "sample_indices": batch["sample_indices"].tolist(),
                "source_realization": batch["source_realization"],
                "candidate_pose_sha256": _tensor_sha256(batch["candidate_pose"]),
                "candidate_image_sha256": _tensor_sha256(batch["candidate_image"]),
                "candidate_mask_sha256": _tensor_sha256(batch["candidate_mask"]),
                "target_index": batch["target_index"].detach().cpu().tolist(),
                **step,
            }
        )
        if update in config["training"]["development_updates"]:
            materialized = _materialize_sources(
                generator,
                development_manifest,
                config["training"]["batch_size"],
            )
            evaluation = paired_evaluate_panel(
                models,
                generator,
                materialized,
                config["data"]["development_seed"],
                "development",
                config,
                free_search=False,
            )
            evaluation["update"] = update
            evaluation["status"] = paired_panel_status(
                evaluation, config, require_search=False
            )
            development.append(evaluation)
            evaluation["development_file_payload_sha256"] = _canonical_sha256(
                evaluation
            )
            _write_immutable_json(
                run_folder / f"paired_development_{update:04d}.json", evaluation
            )
        if (
            update % config["training"]["resume_every_updates"] == 0
            or update == config["training"]["max_updates"]
        ):
            _atomic_torch(
                resume_path,
                _resume_payload(
                    config,
                    lineage,
                    models,
                    optimizers,
                    scalers,
                    device,
                    update,
                    history,
                    development,
                    initialization,
                    statistics_integrity,
                ),
            )

    if statistics_integrity is None or not resume_path.is_file():
        raise RuntimeError("paired run reached finalization without a complete resume state")
    final_path = run_folder / "joint_final_checkpoint.pt"
    if final_path.exists():
        final = _verify_joint_final(config, final_path, resume_path)
        for name in models:
            if _state_dict_sha256(models[name].state_dict()) != final[
                "model_state_sha256"
            ][name]:
                raise RuntimeError("existing final checkpoint differs from current paired state")
    else:
        _atomic_torch(
            final_path,
            _final_payload(
                config,
                lineage,
                models,
                history,
                development,
                initialization,
                statistics_integrity,
                resume_path,
            ),
        )
        final = _verify_joint_final(config, final_path, resume_path)

    trajectory_path = _trajectory_receipt(
        run_folder / "paired_training_trajectory.json",
        config,
        history,
        final_path,
        resume_path,
    )
    freeze_qualification(run_folder, config, final_path, resume_path)
    receipt_path = run_folder / "diagnostic_receipt.json"
    if _existing_receipt_is_valid(receipt_path, config, final_path):
        return run_folder

    capability = verified_qualification_capability(
        run_folder, config, final_path, resume_path
    )
    qualification = []
    manifests = qualification_manifests(
        generator, config, run_folder, final_path, resume_path
    )
    for manifest in manifests:
        if (
            verified_qualification_capability(
                run_folder, config, final_path, resume_path
            )
            != capability
        ):
            raise RuntimeError("joint freeze changed before qualification materialization")
        seed = int(manifest["seed"])
        manifest_path = write_manifest_receipt(
            run_folder / f"qualification_manifest_seed_{seed}.json",
            "post-joint-freeze-qualification",
            manifest,
            generator,
            config,
            {
                "joint_final_file_sha256": capability[
                    "joint_final_file_sha256"
                ],
                "both_model_state_sha256": capability["both_model_state_sha256"],
                "freeze_payload_sha256": capability["freeze_payload_sha256"],
                "data_lineage_sha256": capability["data_lineage_sha256"],
            },
        )
        result_path = run_folder / f"paired_qualification_seed_{seed}.json"
        result_lineage = {
            "family_self_sha256": config["family_self_sha256"],
            "source_sha256": _assert_frozen_files(config),
            "config_contract_sha256": config["contract_sha256"],
            "config_file_sha256": config["config_file_sha256"],
            "joint_final_file_sha256": _sha256(final_path),
            "both_model_state_sha256": final["model_state_sha256"],
            "data_lineage_sha256": lineage["data_lineage_sha256"],
            "generator_contract_sha256": lineage["generator_contract_sha256"],
            "atlas_sha256": lineage["atlas_sha256"],
            "panel_manifest_sha256": manifest["manifest_sha256"],
            "panel_manifest_file_sha256": _sha256(manifest_path),
        }
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result_commitment = result.pop("qualification_file_payload_sha256", None)
            if (
                result.get("lineage") != result_lineage
                or result_commitment != _canonical_sha256(result)
            ):
                raise RuntimeError("existing paired qualification result differs")
            result["qualification_file_payload_sha256"] = result_commitment
            if not _qualification_result_semantics_valid(result, config):
                raise RuntimeError(
                    "existing paired qualification statistics do not recompute"
                )
            qualification.append(result)
            continue
        materialized = _materialize_sources(
            generator, manifest, config["training"]["batch_size"]
        )
        result = paired_evaluate_panel(
            models,
            generator,
            materialized,
            seed,
            "qualification",
            config,
            free_search=True,
        )
        if (
            verified_qualification_capability(
                run_folder, config, final_path, resume_path
            )
            != capability
        ):
            raise RuntimeError("joint freeze changed during qualification evaluation")
        result["status"] = paired_panel_status(
            result, config, require_search=True
        )
        result["lineage"] = result_lineage
        result["qualification_file_payload_sha256"] = _canonical_sha256(result)
        if not _qualification_result_semantics_valid(result, config):
            raise RuntimeError("fresh paired qualification statistics failed audit")
        _write_immutable_json(result_path, result)
        qualification.append(result)

    training_integrity = final["training_integrity"]
    status = family_status(qualification, training_integrity)
    receipt = {
        "format": RECEIPT_FORMAT,
        "purpose": PURPOSE,
        "role": ROLE,
        "family_self_sha256": config["family_self_sha256"],
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": _assert_frozen_files(config),
        "joint_final_file_sha256": _sha256(final_path),
        "joint_resume_file_sha256": _sha256(resume_path),
        "paired_training_trajectory_file_sha256": _sha256(trajectory_path),
        "both_model_state_sha256": final["model_state_sha256"],
        "training_history_sha256": final["training_history_sha256"],
        "data_lineage": lineage,
        "initialization": initialization,
        "initial_statistics_integrity": statistics_integrity,
        "training_integrity": training_integrity,
        "qualification": qualification,
        "family_status": status,
        "passed": status["passed"],
        "product5_access": False,
        "calibration_access": False,
        "final_test_access": False,
        "learned_checkpoint_dependencies": [],
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    _write_immutable_json(receipt_path, receipt)
    return run_folder


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit(
            "usage: python -m training.run_independent_atlas_pair_spatial_aggregation CONFIG"
        )
    print(run(arguments[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
