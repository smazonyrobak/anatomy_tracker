"""Run one frozen, development-only cold-start architecture screen.

Usage::

    python -m training.run_independent_architecture_screen PROTOCOL.json ARCHITECTURE.json

The protocol is shared by all candidate families.  The second file is one of
the already frozen independent architecture preflight configs.  This module
never opens calibration/final-test data and never initializes from a learned
checkpoint; it only validates the checkpoint produced by its own run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import training.independent_joint_data as independent_data
from training.independent_joint_data import IndependentProduct5Data, IndependentSyntheticData
from training.preflight_independent_architectures import (
    build_model,
    load_frozen_config as load_architecture_config,
)
from training.synthetic_registration import STRATA, SyntheticRegistrationGenerator
from training.train_independent_joint import (
    ACCURACY_POSE_SCALE,
    dense_registration_losses,
    independent_joint_forward,
    raw_prediction_records,
    shuffle_candidates,
    train_independent_joint,
    _validate_development_panel,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_PURPOSE = "development-only-cold-start-architecture-screen"
SOURCE_FILES = (
    "training/independent_joint_model.py",
    "training/independent_joint_variants.py",
    "training/independent_joint_data.py",
    "training/train_independent_joint.py",
    "training/preflight_independent_architectures.py",
    "training/synthetic_registration.py",
    "training/registered_section_dataset.py",
    "training/run_independent_architecture_screen.py",
    "source/dense_registration_preprocessing.py",
    "source/atlas_pose_runtime.py",
    "source/registered_image_quality.py",
)
OUTLINE_MODES = {name: index for index, name in enumerate(independent_data.OUTLINE_MODE_NAMES)}
PAIRED_INVARIANT_TENSORS = (
    "true_pose",
    "truth_fixed_image",
    "truth_fixed_mask",
    "truth_fixed_labels",
    "truth_svf",
    "truth_fixed_to_source_map",
    "truth_source_to_fixed_map",
    "truth_fixed_valid_mask",
    "truth_source_valid_mask",
    "truth_source_labels",
    "truth_source_tissue_mask",
    "truth_source_damage_mask",
    "truth_source_brush_mask",
    "truth_source_view_h",
    "truth_generator_similarity_h",
    "truth_similarity_h",
    "truth_similarity_parameters",
    "truth_source_view_parameters",
    "candidate_pose",
    "candidate_fixed_image",
    "candidate_fixed_mask",
    "candidate_fixed_labels",
    "candidate_in_training_domain",
    "candidate_dense_truth_valid",
)
PANEL_INPUT_AND_RANKING_TENSORS = (
    "source_image",
    "source_mask",
    "mask_available",
    "input_outline_mode",
    "true_pose",
    "candidate_pose",
    "candidate_fixed_image",
    "candidate_fixed_mask",
    "candidate_fixed_labels",
    "candidate_in_training_domain",
    "candidate_dense_truth_valid",
    "listwise_target_index",
    "candidate_permutation",
)
HIGH_PANEL_TENSORS = PANEL_INPUT_AND_RANKING_TENSORS + tuple(
    name for name in PAIRED_INVARIANT_TENSORS if name not in PANEL_INPUT_AND_RANKING_TENSORS
)
METRIC_BATCH_TENSORS = (
    "true_pose",
    "listwise_target_index",
    "input_outline_mode",
    "truth_fixed_valid_mask",
    "truth_source_valid_mask",
    "truth_fixed_to_source_map",
    "truth_source_to_fixed_map",
    "truth_similarity_parameters",
    "truth_svf",
    "truth_fixed_labels",
    "truth_source_labels",
)


def _canonical(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def _tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(np.asarray(tensor.shape, np.int64).tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_bundle_sha256(batch: dict, names) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update(_tensor_sha256(batch[name]).encode())
    return digest.hexdigest()


def _batch_to(batch: dict, device) -> dict:
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def _sample_slice(batch: dict, start: int, stop: int) -> dict:
    """Slice one contiguous group of sources while keeping each candidate set intact."""
    count = len(batch["true_pose"])
    sliced = {}
    for name, value in batch.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == count:
            sliced[name] = value[start:stop]
        elif isinstance(value, np.ndarray) and value.ndim and value.shape[0] == count:
            sliced[name] = value[start:stop]
        elif isinstance(value, (list, tuple)) and len(value) == count:
            sliced[name] = value[start:stop]
        else:
            sliced[name] = value
    return sliced


def _metric_output_to_cpu(output: dict, sample_offset: int) -> dict:
    dense = output["dense"]
    return {
        "settled_pose": output["settled_pose"].detach().cpu(),
        "ranking_logits_masked": output["ranking_logits_masked"].detach().cpu(),
        "dense_sample_index": output["dense_sample_index"].detach().cpu()
        + int(sample_offset),
        "dense": None
        if dense is None
        else {
            name: value.detach().cpu() if torch.is_tensor(value) else value
            for name, value in dense.items()
        },
    }


def _concatenate_metric_outputs(parts: list[dict]) -> dict:
    dense_parts = [part["dense"] for part in parts if part["dense"] is not None]
    dense = None
    if dense_parts:
        keys = set(dense_parts[0])
        if any(set(part) != keys for part in dense_parts):
            raise RuntimeError("development chunks returned different dense fields")
        dense = {}
        for name in keys:
            values = [part[name] for part in dense_parts]
            if not all(torch.is_tensor(value) and value.ndim for value in values):
                raise RuntimeError("development dense fields must be batch-leading tensors")
            dense[name] = torch.cat(values, dim=0)
    return {
        "settled_pose": torch.cat([part["settled_pose"] for part in parts], dim=0),
        "ranking_logits_masked": torch.cat(
            [part["ranking_logits_masked"] for part in parts], dim=0
        ),
        "dense_sample_index": torch.cat(
            [part["dense_sample_index"] for part in parts], dim=0
        ),
        "dense": dense,
    }


def _metric_state_to_device(output: dict, batch: dict, device):
    moved_output = {
        "settled_pose": output["settled_pose"].to(device),
        "ranking_logits_masked": output["ranking_logits_masked"].to(device),
        "dense_sample_index": output["dense_sample_index"].to(device),
        "dense": None
        if output["dense"] is None
        else {
            name: value.to(device) if torch.is_tensor(value) else value
            for name, value in output["dense"].items()
        },
    }
    moved_batch = {
        name: batch[name].to(device)
        for name in METRIC_BATCH_TENSORS
        if name in batch
    }
    return moved_output, moved_batch


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def load_screen_protocol(path: str | Path) -> dict:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    contract = raw.pop("contract_sha256")
    if raw.get("schema_version") != PROTOCOL_SCHEMA_VERSION or raw.get("frozen") is not True:
        raise ValueError("architecture-screen protocol is not frozen schema v1")
    if raw.get("purpose") != PROTOCOL_PURPOSE:
        raise ValueError("architecture-screen protocol is not development-only")
    if raw.get("calibration_access") is not False or raw.get("final_test_access") is not False:
        raise ValueError("development screen cannot access calibration/final-test data")
    if raw.get("learned_checkpoint_dependencies") != []:
        raise ValueError("development screen must start without learned dependencies")
    if _canonical_sha256(raw) != contract:
        raise ValueError("architecture-screen protocol hash differs from its frozen payload")
    if raw["paths"]["product5_root_env"] != "ATLAS_PRODUCT5_ROOT":
        raise ValueError("Product-5 root must come from ATLAS_PRODUCT5_ROOT")
    if raw["paths"]["run_root_env"] != "ATLAS_JOINT_RUN_ROOT":
        raise ValueError("run root must come from ATLAS_JOINT_RUN_ROOT")
    atlas_relative = Path(raw["paths"]["atlas_repo_relative"])
    if atlas_relative.is_absolute() or ".." in atlas_relative.parts:
        raise ValueError("atlas path must be repository-relative")
    training = raw["training"]
    if not 1 <= int(training["steps"]) <= 5000:
        raise ValueError("development architecture screen must use 1-5000 planned steps")
    if int(training["recurrent_steps"]) != 3:
        raise ValueError("architecture comparison is frozen at T=3")
    if set(training["curriculum"]) - {
        "regular_synthetic", "high_tilt", "product5"
    }:
        raise ValueError("unknown training curriculum stream")
    for stream in ("regular_synthetic", "high_tilt"):
        provider = raw["providers"][stream]
        if not 1 <= int(provider["batch_size"]) <= 16:
            raise ValueError("synthetic screen batch size is outside the compact protocol")
        if not provider["strata"] or set(provider["strata"]) - set(STRATA):
            raise ValueError("synthetic provider has an unknown stratum")
    if not 1 <= int(raw["providers"]["product5"]["batch_size"]) <= 16:
        raise ValueError("Product-5 screen batch size is outside the compact protocol")
    development = raw["development"]
    evaluate_every = int(development["evaluate_every"])
    if not 0 < evaluate_every <= int(training["steps"]) or int(training["steps"]) % evaluate_every:
        raise ValueError("planned screen must end on a fixed development evaluation")
    if not 27 <= int(development["product5_count"]) <= 64:
        raise ValueError("Product-5 panel needs at least eight no-outline samples")
    if not 8 <= int(development["paired_outline"]["count"]) <= 32:
        raise ValueError("paired outline panel must use 8-32 common base samples")
    if not 8 <= int(development["high_tilt_count"]) <= 32:
        raise ValueError("all-absent high-tilt panel must use 8-32 samples")
    if not 1 <= int(development["evaluation_sample_chunk_size"]) <= 4:
        raise ValueError("development evaluation sample chunks must contain 1-4 sources")
    expected_sources = raw["lineage"]["source_sha256"]
    if set(expected_sources) != set(SOURCE_FILES):
        raise ValueError("architecture-screen source lineage is incomplete")
    for relative_path, expected in expected_sources.items():
        if _source_sha256(REPOSITORY_ROOT / relative_path) != expected:
            raise ValueError(f"source lineage changed: {relative_path}")
    raw["contract_sha256"] = contract
    raw["config_file_sha256"] = _source_sha256(path)
    return raw


def _resolve_environment(protocol: dict) -> tuple[Path, Path, Path]:
    atlas = (REPOSITORY_ROOT / protocol["paths"]["atlas_repo_relative"]).resolve()
    product_variable = protocol["paths"]["product5_root_env"]
    run_variable = protocol["paths"]["run_root_env"]
    if not os.environ.get(product_variable) or not os.environ.get(run_variable):
        raise RuntimeError(f"set {product_variable} and {run_variable} before running the screen")
    return atlas, Path(os.environ[product_variable]).resolve(), Path(os.environ[run_variable]).resolve()


def _provider_functions(protocol, synthetic, product5):
    providers = protocol["providers"]

    def synthetic_provider(name, high_tilt):
        settings = providers[name]

        def provide(step, counter):
            seed = int(settings["base_seed"]) + int(counter) * int(settings["seed_stride"])
            stratum = settings["strata"][int(counter) % len(settings["strata"])]
            if high_tilt:
                return synthetic.generate_high_tilt(
                    int(settings["batch_size"]), "train", seed, stratum
                )
            return synthetic.generate(
                int(settings["batch_size"]), "train", seed, stratum,
                pose_regime="standard",
            )

        return provide

    product_settings = providers["product5"]

    def provide_product5(step, counter):
        seed = int(product_settings["base_seed"]) + int(counter) * int(
            product_settings["seed_stride"]
        )
        return product5.generate(int(product_settings["batch_size"]), seed, int(counter))

    return {
        "regular_synthetic": synthetic_provider("regular_synthetic", False),
        "high_tilt": synthetic_provider("high_tilt", True),
        "product5": provide_product5,
    }


def _outline_plan(mode: int, seed: int, count: int) -> dict:
    rng = np.random.default_rng(int(seed))
    imperfect = mode == OUTLINE_MODES["imperfect"]
    morphology = np.zeros(count, np.int8)
    jitter = np.zeros(count, np.float32)
    if imperfect:
        morphology = rng.choice(
            np.asarray((-3, -2, -1, 1, 2, 3), np.int8), count
        )
        jitter = rng.uniform(0.5, 2.5, count).astype(np.float32)
    jitter_seed = rng.integers(
        0, np.iinfo(np.uint64).max, count, dtype=np.uint64, endpoint=True
    )
    plan = {
        "mode_probabilities": independent_data.OUTLINE_MODE_PROBABILITIES.astype(np.float32),
        "mode_counts": np.eye(3, dtype=np.int64)[mode] * int(count),
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


def _with_outline_mode(manifest: dict, mode: int, seed: int) -> dict:
    manifest = copy.deepcopy(manifest)
    manifest["outline_plan"] = _outline_plan(
        mode, seed, int(manifest["sample_count"])
    )
    manifest["manifest_sha256"] = independent_data._payload_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def _paired_outline_batches(synthetic, settings: dict):
    base = synthetic.make_manifest(
        int(settings["count"]),
        "validation",
        int(settings["seed"]),
        settings["stratum"],
        pose_regime="standard",
    )
    manifests, batches = {}, {}
    for name, mode in OUTLINE_MODES.items():
        manifest = _with_outline_mode(base, mode, int(settings["seed"]) + mode)
        manifests[name] = manifest
        batches[name] = _batch_to(
            shuffle_candidates(
                synthetic.batch(manifest), int(settings["shuffle_seed"]), 0
            ),
            "cpu",
        )
    reference = batches["accurate"]
    for name in ("imperfect", "absent"):
        for key in PAIRED_INVARIANT_TENSORS:
            if not torch.equal(reference[key], batches[name][key]):
                raise RuntimeError(f"paired outline panel changed exact truth: {key}")
    return manifests, batches


def _select_dense(output: dict, batch: dict, selected: torch.Tensor):
    dense_index = output["dense_sample_index"]
    keep = torch.isin(dense_index, selected)
    rows = keep.nonzero(as_tuple=False).flatten()
    indices = dense_index.index_select(0, rows)
    if not len(rows):
        return None, indices
    dense_count = len(dense_index)
    dense = {
        name: value.index_select(0, rows)
        if torch.is_tensor(value) and value.ndim and value.shape[0] == dense_count
        else value
        for name, value in output["dense"].items()
    }
    return dense, indices


def _panel_metrics(output: dict, batch: dict, selected: torch.Tensor) -> dict:
    selected = selected.long()
    pose_scale = batch["true_pose"].new_tensor(ACCURACY_POSE_SCALE)
    pose_error = (
        (output["settled_pose"].index_select(0, selected) - batch["true_pose"].index_select(0, selected))
        / pose_scale
    )
    ranking = F.cross_entropy(
        output["ranking_logits_masked"].index_select(0, selected),
        batch["listwise_target_index"].index_select(0, selected),
    )
    metrics = {
        "pose_normalized_mean_l2": float(pose_error.square().sum(1).sqrt().mean().cpu()),
        "ranking_nll": float(ranking.cpu()),
        "dense_map": None,
        "dense_region": None,
        "dense_validity": None,
    }
    dense, dense_indices = _select_dense(output, batch, selected)
    if dense is not None:
        losses = dense_registration_losses(dense, batch, dense_indices)
        metrics.update(
            {
                "dense_map": float(
                    (losses["dense_map_forward"] + losses["dense_map_inverse"]).cpu()
                ),
                "dense_region": float(
                    (losses["dense_region_dice"] + losses["dense_region_boundary"]).cpu()
                ),
                "dense_validity": float(losses["dense_validity"].cpu()),
            }
        )
    return metrics


def _metric_score(metrics: dict, weights: dict) -> float:
    return float(
        weights["pose"] * metrics["pose_normalized_mean_l2"]
        + weights["ranking"] * metrics["ranking_nll"]
        + weights["dense_map"] * (metrics["dense_map"] or 0.0)
        + weights["dense_region"] * (metrics["dense_region"] or 0.0)
        + weights["dense_validity"] * (metrics["dense_validity"] or 0.0)
    )


def _mode_metrics(output, batch):
    result = {}
    for name, mode in OUTLINE_MODES.items():
        selected = (batch["input_outline_mode"] == mode).nonzero(as_tuple=False).flatten()
        if len(selected):
            result[name] = _panel_metrics(output, batch, selected)
    return result


def _development_setup(protocol, synthetic, validation_product5):
    settings = protocol["development"]
    positions = validation_product5.fixed_validation_positions(
        int(settings["product5_count"]), int(settings["product5_seed"])
    )
    product_provenance = validation_product5.provenance_manifest(positions)
    product_batch = _batch_to(
        shuffle_candidates(
            validation_product5.batch_positions(
                positions,
                int(settings["product5_seed"]),
                int(settings["product5_candidate_schedule_step"]),
            ),
            int(settings["product5_shuffle_seed"]),
            0,
        ),
        "cpu",
    )
    if int((product_batch["input_outline_mode"] == OUTLINE_MODES["absent"]).sum()) < 8:
        raise RuntimeError("fixed Product-5 panel has fewer than eight no-outline samples")
    paired_manifests, paired_batches = _paired_outline_batches(
        synthetic, settings["paired_outline"]
    )
    high = settings["high_tilt"]
    high_manifest = synthetic.make_manifest(
        int(settings["high_tilt_count"]),
        "validation",
        int(high["seed"]),
        high["stratum"],
        pose_regime="high_tilt",
    )
    high_manifest = _with_outline_mode(
        high_manifest, OUTLINE_MODES["absent"], int(high["seed"]) + 1
    )
    high_batch = _batch_to(
        shuffle_candidates(
            synthetic.batch(high_manifest), int(high["shuffle_seed"]), 0
        ),
        "cpu",
    )
    if not torch.all(high_batch["input_outline_mode"] == OUTLINE_MODES["absent"]):
        raise RuntimeError("high-tilt development panel must be entirely no-outline")
    paired_base = paired_manifests["accurate"].get("generator_manifest", {})
    paired_base_sha256 = paired_base.get("manifest_sha256")
    if not paired_base_sha256:
        paired_base_sha256 = independent_data._payload_sha256(paired_base)
    contract = {
        "version": 1,
        "purpose": "animal-disjoint-development-selection-only",
        "historically_consumed": True,
        "untouched_final_test": False,
        "calibration_access": False,
        "final_test_access": False,
        "protocol_contract_sha256": protocol["contract_sha256"],
        "product5_positions": np.asarray(positions, np.int64),
        "product5_provenance": product_provenance,
        "product5_batch_manifest_sha256": product_batch["batch_manifest_sha256"],
        "product5_panel_tensor_sha256": _tensor_bundle_sha256(
            product_batch,
            PANEL_INPUT_AND_RANKING_TENSORS
            + ("animal_id", "specimen_id", "experiment_id", "section_image_id"),
        ),
        "paired_outline_manifest_sha256": {
            name: manifest["manifest_sha256"] for name, manifest in paired_manifests.items()
        },
        "paired_outline_base_generator_manifest_sha256": paired_base_sha256,
        "paired_outline_unmasked_source_sha256": _tensor_sha256(
            paired_batches["absent"]["source_image"]
        ),
        "paired_outline_input_tensor_sha256": {
            name: _tensor_bundle_sha256(
                batch,
                ("source_image", "source_mask", "mask_available", "input_outline_mode"),
            )
            for name, batch in paired_batches.items()
        },
        "paired_outline_invariant_tensor_sha256": {
            name: _tensor_bundle_sha256(batch, PAIRED_INVARIANT_TENSORS)
            for name, batch in paired_batches.items()
        },
        "paired_outline_damage_truth_sha256": {
            name: _tensor_sha256(batch["truth_source_damage_mask"])
            for name, batch in paired_batches.items()
        },
        "high_tilt_manifest_sha256": high_manifest["manifest_sha256"],
        "high_tilt_panel_tensor_sha256": _tensor_bundle_sha256(
            high_batch, HIGH_PANEL_TENSORS
        ),
        "high_tilt_outline_mode": "absent",
        "panel_storage": "cpu-resident;one-panel-at-a-time-device-transfer",
        "evaluation_chunking": {
            "sample_chunk_size": int(settings["evaluation_sample_chunk_size"]),
            "candidate_policy": "all-candidates-for-each-source-preserved",
            "sample_order": "contiguous-original-order",
        },
        "metric": settings["metric"],
        "primary_endpoint": "absent/no-user-mask",
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract, product_batch, paired_batches, high_batch


def _development_evaluator(
    renderer,
    architecture_contract_sha256,
    panel_contract,
    product_batch,
    paired_batches,
    high_batch,
    receipt_folder: Path | None = None,
):
    metric_config = panel_contract["metric"]
    chunk_contract = panel_contract["evaluation_chunking"]
    sample_chunk_size = int(chunk_contract["sample_chunk_size"])

    def evaluate(model, step):
        panels = {}
        real_raw = []
        synthetic_raw = []
        chunk_receipts = {}

        def run(name, cpu_batch):
            count = len(cpu_batch["true_pose"])
            if count <= 0:
                raise RuntimeError("development panels cannot be empty")
            candidate_count = int(cpu_batch["candidate_pose"].shape[1])
            metric_parts = []
            panel_records = []
            observed_chunks = []
            for start in range(0, count, sample_chunk_size):
                stop = min(start + sample_chunk_size, count)
                batch = _batch_to(_sample_slice(cpu_batch, start, stop), renderer.device)
                if int(batch["candidate_pose"].shape[1]) != candidate_count:
                    raise RuntimeError("development chunk changed a source candidate set")
                output = independent_joint_forward(model, batch, renderer)
                records = raw_prediction_records(output, batch)
                for local_item, record in enumerate(records):
                    item = start + local_item
                    if record["source_type"] == "synthetic_ccf":
                        record["synthetic_sample_index"] = item
                        record["record_provenance_sha256"] = _canonical_sha256(
                            {
                                "source_type": record["source_type"],
                                "data_contract_sha256": record["data_contract_sha256"],
                                "sample_manifest_sha256": record["sample_manifest_sha256"],
                                "sample_index": item,
                            }
                        )
                        if name.startswith("paired_outline_"):
                            record["paired_base_sample_provenance_sha256"] = _canonical_sha256(
                                {
                                    "base_generator_manifest_sha256": panel_contract[
                                        "paired_outline_base_generator_manifest_sha256"
                                    ],
                                    "sample_index": item,
                                }
                            )
                    record.update(
                        {
                            "panel_name": name,
                            "panel_sample_index": item,
                            "panel_contract_sha256": panel_contract["contract_sha256"],
                            "architecture_contract_sha256": architecture_contract_sha256,
                        }
                    )
                panel_records.extend(records)
                metric_parts.append(_metric_output_to_cpu(output, start))
                observed_chunks.append(stop - start)
                del output, batch
            if cpu_batch["source_type"] == "allen_registered_product5":
                real_raw.extend(panel_records)
            else:
                synthetic_raw.extend(panel_records)
            metric_output, metric_batch = _metric_state_to_device(
                _concatenate_metric_outputs(metric_parts), cpu_batch, renderer.device
            )
            selected = torch.arange(count, device=renderer.device)
            panels[name] = {
                "all": _panel_metrics(metric_output, metric_batch, selected),
                "by_outline_mode": _mode_metrics(metric_output, metric_batch),
            }
            chunk_receipts[name] = {
                "sample_count": count,
                "candidate_count_per_sample": candidate_count,
                "observed_sample_chunks": observed_chunks,
                "maximum_candidates_per_forward": sample_chunk_size * candidate_count,
            }
            return panels[name]

        product = run("product5", product_batch)
        paired = {name: run(f"paired_outline_{name}", batch)["all"] for name, batch in paired_batches.items()}
        high = run("high_tilt", high_batch)
        absent = paired["absent"]
        paired_delta = {
            name: {
                component: None
                if value is None or absent[component] is None
                else value - absent[component]
                for component, value in metrics.items()
            }
            for name, metrics in paired.items()
        }
        product_primary = product["by_outline_mode"].get("absent")
        high_primary = high["by_outline_mode"].get("absent")
        if product_primary is None or high_primary is None:
            raise RuntimeError("fixed development panels must contain absent/no-user-mask samples")
        weights = metric_config["component_weights"]
        panel_weights = metric_config["primary_panel_weights"]
        selection_metric = (
            panel_weights["product5_absent"] * _metric_score(product_primary, weights)
            + panel_weights["paired_outline_absent"] * _metric_score(absent, weights)
            + panel_weights["high_tilt_absent"] * _metric_score(high_primary, weights)
        )
        animal_ids = sorted({int(record["animal_id"]) for record in real_raw})
        result = {
            "partition": "validation",
            "fresh_checkpoint_step": int(step),
            "panel_contract_sha256": panel_contract["contract_sha256"],
            "animal_ids": animal_ids,
            "selection_metric": float(selection_metric),
            "primary_endpoint": "absent/no-user-mask",
            "architecture_contract_sha256": architecture_contract_sha256,
            "metric_components": panels,
            "paired_outline_metrics": paired,
            "paired_outline_deltas_vs_absent": paired_delta,
            "evaluation_chunking": {
                **chunk_contract,
                "panels": chunk_receipts,
            },
            "raw_predictions": real_raw,
            "synthetic_raw_predictions": synthetic_raw,
        }
        result["panel_manifest_sha256"] = _canonical_sha256(
            {
                "panel_contract_sha256": panel_contract["contract_sha256"],
                "architecture_contract_sha256": architecture_contract_sha256,
                "fresh_checkpoint_step": int(step),
                "selection_metric": float(selection_metric),
                "raw_predictions": real_raw,
                "synthetic_raw_predictions": synthetic_raw,
            }
        )
        if receipt_folder is not None:
            latest_path = receipt_folder / "development_panel_latest.json"
            _atomic_json(result, latest_path)
        return result

    return evaluate


def _materialize_best_development(
    result: dict,
    checkpoint_folder: Path,
    panel_contract_sha256: str,
    train_animals,
) -> dict | None:
    receipt_path = checkpoint_folder / "development_panel_best.json"
    checkpoint_path = result.get("best_checkpoint")
    best_metric = float(result["best_metric"])
    if checkpoint_path is None:
        if math.isfinite(best_metric) or receipt_path.is_file():
            raise RuntimeError("development best receipt exists without a trainer best checkpoint")
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise RuntimeError("trainer reported a missing best checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("format") != "independent-joint-cold-start-v1"
        or checkpoint.get("learned_checkpoint_dependencies") != []
        or checkpoint.get("checkpoint_selection_state") != "ema"
        or checkpoint.get("lineage") != result["lineage"]
    ):
        raise RuntimeError("best checkpoint is not from this cold-start EMA screen")
    panel = checkpoint.get("development_panel")
    if panel is None:
        raise RuntimeError("best checkpoint does not contain its selecting development panel")
    metric, _ = _validate_development_panel(
        panel,
        int(checkpoint["step"]),
        train_animals,
        panel_contract_sha256,
    )
    if not math.isclose(metric, best_metric, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
        float(checkpoint["best_metric"]), best_metric, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("best checkpoint and trainer selection metric disagree")
    receipt = copy.deepcopy(panel)
    receipt.update(
        {
            "trainer_best_checkpoint": str(checkpoint_path),
            "trainer_best_checkpoint_sha256": _binary_sha256(checkpoint_path),
            "trainer_lineage_sha256": result["lineage"]["lineage_sha256"],
            "checkpoint_selection_state": "ema",
        }
    )
    _atomic_json(receipt, receipt_path)
    return receipt


def run_architecture_screen(
    protocol_path: str | Path,
    architecture_path: str | Path,
) -> dict:
    protocol = load_screen_protocol(protocol_path)
    architecture = load_architecture_config(architecture_path)
    if int(protocol["seed"]) != int(architecture["workload"]["seed"]):
        raise ValueError("protocol and frozen architecture random seeds differ")
    model = build_model(architecture)
    if getattr(model, "learned_weight_dependencies", ()):
        raise RuntimeError("architecture screen cannot use learned weights")
    atlas_folder, product5_root, run_root = _resolve_environment(protocol)
    device_name = protocol["device"]
    device = "cuda" if device_name == "auto" and torch.cuda.is_available() else (
        "cpu" if device_name == "auto" else device_name
    )
    model = model.to(device)
    generator = SyntheticRegistrationGenerator(atlas_folder, device=device)
    synthetic = IndependentSyntheticData(generator)
    training_product5 = IndependentProduct5Data(
        product5_root, atlas_folder, generator, split="train"
    )
    validation_product5 = IndependentProduct5Data(
        product5_root, atlas_folder, generator, split="validation"
    )
    train_animals = {int(value) for value in training_product5.contract["specimen_ids"]}
    validation_animals = {int(value) for value in validation_product5.contract["specimen_ids"]}
    if not train_animals or not validation_animals or train_animals.intersection(validation_animals):
        raise RuntimeError("Product-5 train/development animals are not animal-disjoint")

    panel_contract, product_batch, paired_batches, high_batch = _development_setup(
        protocol, synthetic, validation_product5
    )
    panel_animals = set(int(value) for value in product_batch["animal_id"].detach().cpu())
    if (
        not panel_animals
        or not panel_animals.issubset(validation_animals)
        or panel_animals.intersection(train_animals)
    ):
        raise RuntimeError("fixed development panel is outside its Product-5 validation animals")
    panel_registry = run_root / f"development_panel_{protocol['contract_sha256']}.json"
    if panel_registry.is_file():
        previous = json.loads(panel_registry.read_text(encoding="utf-8"))
        if previous != _canonical(panel_contract):
            raise RuntimeError("shared development panel differs from its frozen registry")
    else:
        _atomic_json(panel_contract, panel_registry)

    architecture_name = architecture["name"]
    checkpoint_folder = run_root / protocol["name"] / architecture_name
    source_hashes = {
        relative: _source_sha256(REPOSITORY_ROOT / relative) for relative in SOURCE_FILES
    }
    setup_receipt = {
        "version": 1,
        "purpose": PROTOCOL_PURPOSE,
        "calibration_access": False,
        "final_test_access": False,
        "learned_checkpoint_dependencies": [],
        "protocol": protocol,
        "architecture": architecture,
        "architecture_initial_state_sha256": _state_sha256(model),
        "source_sha256": source_hashes,
        "atlas_contract": generator.contract,
        "data_contract_sha256": {
            "regular_synthetic": synthetic.contract["contract_sha256"],
            "high_tilt": synthetic.contract["contract_sha256"],
            "product5": training_product5.contract["contract_sha256"],
        },
        "train_animal_ids": sorted(train_animals),
        "validation_animal_ids": sorted(validation_animals),
        "training_product5_provenance": training_product5.provenance_manifest(),
        "development_panel": panel_contract,
        "development_panel_registry": str(panel_registry),
    }
    setup_identity = copy.deepcopy(setup_receipt)
    setup_receipt["screen_setup_sha256"] = _canonical_sha256(setup_identity)
    receipt_path = checkpoint_folder / "screen_receipt.json"
    previous_receipt = None
    if receipt_path.is_file():
        previous_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        previous_identity = {
            name: previous_receipt.get(name) for name in setup_identity
        }
        if (
            previous_receipt.get("screen_setup_sha256")
            != setup_receipt["screen_setup_sha256"]
            or _canonical_sha256(previous_identity) != setup_receipt["screen_setup_sha256"]
        ):
            raise RuntimeError("existing screen receipt differs from this frozen setup")
    else:
        _atomic_json(setup_receipt, receipt_path)

    providers = _provider_functions(protocol, synthetic, training_product5)
    evaluator = _development_evaluator(
        generator,
        architecture["contract_sha256"],
        panel_contract,
        product_batch,
        paired_batches,
        high_batch,
        checkpoint_folder,
    )
    training = protocol["training"]
    result = train_independent_joint(
        model,
        generator,
        providers,
        {
            "regular_synthetic": synthetic.contract,
            "high_tilt": synthetic.contract,
            "product5": training_product5.contract,
        },
        checkpoint_folder,
        int(training["steps"]),
        seed=int(protocol["seed"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        amp=bool(training["amp"]),
        ema_decay=float(training["ema_decay"]),
        curriculum=tuple(training["curriculum"]),
        loss_weights=training.get("loss_weights"),
        recurrent_steps=3,
        warmup_steps=int(training["warmup_steps"]),
        minimum_learning_rate_fraction=float(training["minimum_learning_rate_fraction"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        checkpoint_interval=int(training["checkpoint_interval"]),
        max_steps_this_call=training.get("max_steps_this_call"),
        resume=bool(training["resume"]),
        development_evaluator=evaluator,
        evaluate_every=int(protocol["development"]["evaluate_every"]),
        development_panel_contract_sha256=panel_contract["contract_sha256"],
        train_animal_ids=sorted(train_animals),
    )
    training_result_receipt = {
        key: _canonical(value)
        for key, value in result.items()
        if key not in {"lineage"}
    }
    last_training_records = result["raw_predictions"]
    if not last_training_records and previous_receipt is not None:
        last_training_records = previous_receipt.get(
            "last_training_raw_predictions", []
        )
        training_result_receipt["raw_predictions"] = last_training_records
        if training_result_receipt.get("last_loss") is None:
            training_result_receipt["last_loss"] = previous_receipt.get(
                "training_result", {}
            ).get("last_loss")
    setup_receipt["training_result"] = training_result_receipt
    setup_receipt["training_lineage"] = result["lineage"]
    setup_receipt["last_training_raw_predictions"] = last_training_records
    latest_development = checkpoint_folder / "development_panel_latest.json"
    best_development = checkpoint_folder / "development_panel_best.json"
    best_panel = _materialize_best_development(
        result,
        checkpoint_folder,
        panel_contract["contract_sha256"],
        sorted(train_animals),
    )
    if latest_development.is_file():
        latest_panel = json.loads(latest_development.read_text(encoding="utf-8"))
        if (
            latest_panel.get("panel_contract_sha256") != panel_contract["contract_sha256"]
            or latest_panel.get("architecture_contract_sha256")
            != architecture["contract_sha256"]
            or int(latest_panel.get("fresh_checkpoint_step", -1)) > int(result["step"])
        ):
            raise RuntimeError("latest development receipt is stale or from another screen")
    setup_receipt["development_receipts"] = {
        "latest": str(latest_development) if latest_development.is_file() else None,
        "best": str(best_development) if best_panel is not None else None,
    }
    setup_receipt["best_development_panel"] = best_panel
    _atomic_json(setup_receipt, receipt_path)
    return {
        "receipt_path": receipt_path,
        "checkpoint_folder": checkpoint_folder,
        "panel_contract_sha256": panel_contract["contract_sha256"],
        "train_animal_ids": sorted(train_animals),
        "validation_animal_ids": sorted(validation_animals),
        "training_result": result,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python -m training.run_independent_architecture_screen "
            "PROTOCOL.json ARCHITECTURE.json"
        )
    print(
        json.dumps(
            _canonical(run_architecture_screen(sys.argv[1], sys.argv[2])),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
