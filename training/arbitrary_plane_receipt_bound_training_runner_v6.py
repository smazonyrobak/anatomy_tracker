"""Standalone receipt-bound runner for fresh v6 arbitrary-plane training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_allen_atlas_binding_v6 as allen_atlas_v6
import training.arbitrary_plane_catalogue_binding_v3 as catalogue_binding_v3
import training.arbitrary_plane_catalogue_runtime_v6 as catalogue_runtime_v6
import training.arbitrary_plane_finite_row_binding_v6 as finite_row_binding_v6
import training.arbitrary_plane_staged_trainer_v6 as staged_trainer_v6
import training.arbitrary_plane_training_data_v6 as training_data_v6


RECEIPT_BOUND_TRAINING_RUN_V6_SCHEMA = (
    "anatomy-tracker.receipt-bound-training-run/v6"
)
RECEIPT_BOUND_TRAINING_STATE_V6_SCHEMA = (
    "anatomy-tracker.receipt-bound-training-state/v6"
)
RECEIPT_BOUND_TRAINING_STEP_REPORT_V6_SCHEMA = (
    "anatomy-tracker.receipt-bound-training-step-report/v6"
)
RECEIPT_BOUND_TRAINING_RAW_OUTPUT_V6_SCHEMA = (
    "anatomy-tracker.receipt-bound-training-raw-output/v6"
)
RECEIPT_BOUND_TRAINING_TRANSACTION_V6_SCHEMA = (
    "anatomy-tracker.receipt-bound-training-transaction/v6"
)
ATLAS_BINDING_V6_SCHEMA = allen_atlas_v6.ALLEN_ATLAS_BINDING_V6_SCHEMA
ANIMAL_AWARE_SAMPLING_V6_SCHEMA = (
    "anatomy-tracker.animal-aware-row-sampling/v6"
)
FULL_CATALOGUE_CELL_COUNT_V6 = 98_304
RAW_UNCALIBRATED = "raw_uncalibrated"
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (
    "training/arbitrary_plane_receipt_bound_training_runner_v6.py",
    "training/arbitrary_plane_allen_atlas_binding_v6.py",
    "training/arbitrary_plane_training_data_v6.py",
    "training/arbitrary_plane_finite_row_binding_v6.py",
    "training/arbitrary_plane_manifest.py",
    "training/arbitrary_plane_support.py",
    "training/arbitrary_plane_rendered_generator.py",
    "training/arbitrary_plane_acquisition_v2.py",
    "training/arbitrary_plane_catalogue_runtime_v6.py",
    "training/arbitrary_plane_catalogue_binding_v3.py",
    "training/arbitrary_plane_catalogue_v3.py",
    "training/arbitrary_plane_geometry.py",
    "training/arbitrary_plane_full_frame_primitives.py",
    "training/arbitrary_plane_recurrent_model.py",
    "training/arbitrary_plane_staged_trainer_v6.py",
)
_RUNNER_CONFIG_KEYS = {
    "batch_size",
    "row_selection_seed",
    "archive_checkpoint_interval_steps",
}
_RUN_MANIFEST_KEYS = {
    "schema_version",
    "run_id",
    "data_role",
    "retrieval_scope",
    "git_commit",
    "source_sha256",
    "execution",
    "catalogue",
    "atlas",
    "training_data",
    "finite_psf_capability",
    "model_kwargs",
    "training_config",
    "runner_config",
    "seed_record",
    "row_sampling_policy",
    "training_commit_contract",
    "initialization",
    "prior_model_weight_dependencies",
    "prior_feature_dependencies",
    "prior_pseudolabel_dependencies",
    "probabilities_calibrated",
    "probability_status",
    "release_qualifying",
    "receipt_sha256",
}
_RUN_STATE_KEYS = {
    "schema_version",
    "run_id",
    "run_manifest_receipt_sha256",
    "global_step",
    "latest_checkpoint",
    "committed_steps",
    "immutable_archives",
    "probabilities_calibrated",
    "probability_status",
    "release_qualifying",
    "receipt_sha256",
}
_FIVE_IDS = (
    "animal_id",
    "specimen_id",
    "experiment_id",
    "synthetic_animal_id",
    "section_id",
)
_RESUME_SLOTS = (
    "checkpoints/resume_slot_0.pt",
    "checkpoints/resume_slot_1.pt",
)
_TRANSACTION_KEYS = {
    "schema_version",
    "run_id",
    "run_manifest_receipt_sha256",
    "step",
    "previous_run_state_receipt_sha256",
    "previous_transaction_receipt_sha256",
    "latest_checkpoint",
    "archive_checkpoint",
    "committed_step",
    "probabilities_calibrated",
    "probability_status",
    "release_qualifying",
    "receipt_sha256",
}
_TRAINER_OUTPUT_KEYS = {
    "schema_version",
    "step",
    "phase",
    "objective",
    "preclip_gradient_norm",
    "optimizer_step_applied",
    "catalogue_cell_count",
    "probabilities_calibrated",
    "probability_status",
    "refinement_ready_row_count",
    "refinement_abstained_row_count",
    "losses",
    "receipt_sha256",
}
_DTYPES = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


def _plain(value):
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def _hash_json(value) -> str:
    return hashlib.sha256(
        json.dumps(
            _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _with_receipt(payload: Mapping[str, object]) -> dict[str, object]:
    plain = _plain(payload)
    return {**plain, "receipt_sha256": _hash_json(plain)}


def _payload(value: Mapping[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if key != "receipt_sha256"}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and not (set(value) - set("0123456789abcdef"))
    )


def _i_path(path, *, must_exist: bool = False) -> Path:
    target = Path(path).resolve()
    if os.path.splitdrive(str(target))[0].upper() != "I:":
        raise ValueError("receipt-bound v6 training paths must be on I:")
    if must_exist and not target.exists():
        raise FileNotFoundError(target)
    return target


def _same_device(left: str | torch.device, right: str | torch.device) -> bool:
    left = torch.device(left)
    right = torch.device(right)
    if left.type != right.type:
        return False
    if left.type != "cuda":
        return left == right
    left_index = torch.cuda.current_device() if left.index is None else left.index
    right_index = torch.cuda.current_device() if right.index is None else right.index
    return left_index == right_index


def _run_file(run_root: Path, relative_path: str) -> Path:
    target = (run_root / relative_path).resolve()
    try:
        target.relative_to(run_root)
    except ValueError as error:
        raise ValueError("run artifact path escapes its I:-only run directory") from error
    return target


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    content = (
        json.dumps(
            _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, content)


def _immutable_json(path: Path, value: Mapping[str, object]) -> None:
    content = (
        json.dumps(
            _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("an immutable v6 run artifact changed during replay")
        return
    _atomic_bytes(path, content)


def _atomic_numpy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    torch.save(value, temporary)
    with temporary.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _fsync_existing_file(path: Path) -> None:
    with path.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _copy_immutable(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f"{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    with source.open("rb") as read_stream, temporary.open("xb") as write_stream:
        shutil.copyfileobj(read_stream, write_stream, length=1024 * 1024)
        write_stream.flush()
        os.fsync(write_stream.fileno())
    os.replace(temporary, target)


def _source_receipts() -> dict[str, str]:
    return {
        name: hashlib.sha256((_SOURCE_ROOT / name).read_bytes()).hexdigest()
        for name in _SOURCE_FILES
    }


def _declared_source_files() -> tuple[str, ...]:
    return tuple(sorted(set(_SOURCE_FILES) | set(staged_trainer_v6._SOURCE_FILES)))


def _verify_declared_sources_match_git_commit(commit: str) -> None:
    """Require every executable source byte to be recoverable from ``commit``."""
    for name in _declared_source_files():
        result = subprocess.run(
            ["git", "-C", str(_SOURCE_ROOT), "show", f"{commit}:{name}"],
            check=True,
            capture_output=True,
        )
        if result.stdout != (_SOURCE_ROOT / name).read_bytes():
            raise ValueError(
                f"declared v6 source differs from recorded git commit: {name}"
            )


def _current_git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(_SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip().lower()
    if len(value) not in (40, 64) or set(value) - set("0123456789abcdef"):
        raise ValueError("the current git commit is not an exact hexadecimal object ID")
    return value


def _assert_no_forbidden_dependencies(value: object) -> None:
    forbidden_names = (
        "candidate_bank",
        "training_bank",
        "legacy_prediction",
        "legacy_feature",
        "prior_weight",
        "prior_feature",
        "pseudolabel",
    )
    dependency_names = {
        "learned_dependencies",
        "prior_model_weight_dependencies",
        "prior_feature_dependencies",
        "prior_pseudolabel_dependencies",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(name in normalized for name in forbidden_names):
                if item not in (None, [], {}, ()):
                    raise ValueError("v6 runner configuration contains a forbidden dependency")
            if normalized in dependency_names and item not in (None, [], {}, ()):
                raise ValueError("v6 runner configuration contains a learned dependency")
            _assert_no_forbidden_dependencies(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_forbidden_dependencies(item)


def _validated_runner_config(value: Mapping[str, object]) -> dict[str, int]:
    config = dict(value)
    if set(config) != _RUNNER_CONFIG_KEYS:
        raise ValueError(
            f"v6 runner config keys must be exactly {sorted(_RUNNER_CONFIG_KEYS)}"
        )
    if any(
        not isinstance(config[key], int) or isinstance(config[key], bool)
        for key in _RUNNER_CONFIG_KEYS
    ):
        raise ValueError("v6 runner counts and row-selection seed must be integers")
    if (
        config["batch_size"] < 1
        or config["archive_checkpoint_interval_steps"] < 1
        or config["row_selection_seed"] < 0
    ):
        raise ValueError("v6 runner counts must be positive and its seed nonnegative")
    return config


def _row_identity(row: Mapping[str, object], index: int) -> dict[str, object]:
    lineage = row.get("lineage", {})
    if any(
        not isinstance(lineage.get(key), str) or not lineage[key]
        for key in _FIVE_IDS
    ):
        raise ValueError("each frozen v6 row must preserve all five exact IDs")
    identity = {
        "row_index": int(index),
        **{key: lineage[key] for key in _FIVE_IDS},
        "training_row_id": row.get("training_row_id"),
        "training_row_receipt_sha256": row.get("receipt_sha256"),
    }
    if (
        not isinstance(identity["training_row_id"], str)
        or not identity["training_row_id"]
        or not _is_sha256(identity["training_row_receipt_sha256"])
    ):
        raise ValueError("a frozen v6 row lacks its exact row ID or receipt")
    return identity


def _training_data_record(
    cache_directory: Path,
    expected_manifest_receipt_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if not _is_sha256(expected_manifest_receipt_sha256):
        raise ValueError("a trusted frozen cache-manifest receipt is required")
    cache_manifest = finite_row_binding_v6.load_frozen_row_cache_manifest_v6(
        cache_directory,
        expected_manifest_receipt_sha256=expected_manifest_receipt_sha256,
    )
    records = cache_manifest["rows"]
    if (
        not records
        or cache_manifest.get("receipt_sha256")
        != expected_manifest_receipt_sha256
        or cache_manifest.get("row_count") != len(records)
        or cache_manifest.get("generation_lineage", {}).get("split") != "train"
        or any(record.get("lineage", {}).get("split") != "train" for record in records)
    ):
        raise ValueError(
            "v6 optimization requires one nonempty exact train-split frozen cache manifest"
        )
    identities = [
        _row_identity(
            {
                "lineage": record["lineage"],
                "training_row_id": record["training_row_id"],
                "receipt_sha256": record["training_row_receipt_sha256"],
            },
            index,
        )
        for index, record in enumerate(records)
    ]
    selection = {
        "schema_version": finite_row_binding_v6.FROZEN_ROWS_V6_SCHEMA,
        "training_data_manifest_receipt_sha256": expected_manifest_receipt_sha256,
        "cache_manifest_receipt_sha256": expected_manifest_receipt_sha256,
        "generator_binding_receipt_sha256": cache_manifest["generator_binding"][
            "receipt_sha256"
        ],
        "generation_lineage_sha256": cache_manifest["generator_binding"][
            "generation_lineage_sha256"
        ],
        "row_indices": list(range(len(records))),
        "training_row_ids": [item["training_row_id"] for item in identities],
        "training_row_receipts_sha256": [
            item["training_row_receipt_sha256"] for item in identities
        ],
    }
    return (
        {
            "cache_directory": str(cache_directory),
            "training_data_manifest_receipt_sha256": expected_manifest_receipt_sha256,
            "cache_manifest_receipt_sha256": expected_manifest_receipt_sha256,
            "initial_full_selection_receipt_sha256": (
                finite_row_binding_v6.frozen_row_selection_receipt_v6(selection)
            ),
            "generator_binding_receipt_sha256": selection[
                "generator_binding_receipt_sha256"
            ],
            "generation_lineage_sha256": selection[
                "generation_lineage_sha256"
            ],
            "optimization_split": "train",
            "row_count": len(records),
            "ordered_row_identities": identities,
        },
        cache_manifest,
    )


def _sampling_policy(
    training_data: Mapping[str, object], runner_config: Mapping[str, object]
) -> dict[str, object]:
    animal_count = len(
        {item["animal_id"] for item in training_data["ordered_row_identities"]}
    )
    return _with_receipt(
        {
            "schema_version": ANIMAL_AWARE_SAMPLING_V6_SCHEMA,
            "algorithm": "sha256-ranked-animal-cycles-with-hashed-within-animal-row/v6",
            "population": "every row in the exact frozen training-data manifest",
            "training_data_manifest_receipt_sha256": training_data[
                "training_data_manifest_receipt_sha256"
            ],
            "row_count": training_data["row_count"],
            "animal_count": animal_count,
            "batch_size": runner_config["batch_size"],
            "row_selection_seed": runner_config["row_selection_seed"],
            "without_repeated_animal_within_each_complete_animal_cycle": True,
            "within_animal_row_selection": "SHA-256 modulo exact sorted row indices",
        }
    )


def sample_training_row_indices_v6(
    manifest: Mapping[str, object], step: int
) -> list[int]:
    """Select a reproducible animal-balanced batch without mutable RNG state."""
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("sampling step must be a nonnegative integer")
    table = manifest["training_data"]["ordered_row_identities"]
    groups: dict[str, list[int]] = {}
    for item in table:
        groups.setdefault(item["animal_id"], []).append(int(item["row_index"]))
    animals = sorted(groups)
    if not animals:
        raise ValueError("animal-aware v6 sampling requires at least one animal")
    selected = []
    batch_size = int(manifest["runner_config"]["batch_size"])
    seed = int(manifest["runner_config"]["row_selection_seed"])
    for cycle_start in range(0, batch_size, len(animals)):
        cycle = cycle_start // len(animals)
        ranked = sorted(
            animals,
            key=lambda animal: _hash_json(
                {
                    "domain": ANIMAL_AWARE_SAMPLING_V6_SCHEMA,
                    "run_id": manifest["run_id"],
                    "seed": seed,
                    "step": step,
                    "cycle": cycle,
                    "animal_id": animal,
                }
            ),
        )
        for animal in ranked[: min(len(animals), batch_size - cycle_start)]:
            rows = sorted(groups[animal])
            digest = _hash_json(
                {
                    "domain": f"{ANIMAL_AWARE_SAMPLING_V6_SCHEMA}/within-animal",
                    "run_id": manifest["run_id"],
                    "seed": seed,
                    "step": step,
                    "cycle": cycle,
                    "animal_id": animal,
                }
            )
            selected.append(rows[int(digest, 16) % len(rows)])
    return selected


def _training_commit_contract(interval: int) -> dict[str, object]:
    return _with_receipt(
        {
            "rolling_checkpoint_paths": list(_RESUME_SLOTS),
            "rolling_checkpoint_frequency": "every committed step",
            "immutable_archive_interval_steps": interval,
            "immutable_step_reports": True,
            "immutable_raw_trainer_outputs": True,
            "atomic_step_transaction_directories": True,
            "publication_order": [
                "unreferenced-alternate-rolling-checkpoint-and-fsync",
                "complete-hidden-step-transaction-directory-and-fsync",
                "atomic-step-transaction-directory-rename",
                "atomic-run-state-replacement",
            ],
            "crash_recovery": (
                "adopt one complete authenticated step transaction after a pre-state "
                "crash; hidden incomplete transactions remain unreferenced"
            ),
        }
    )


def _training_run_binding(manifest: Mapping[str, object]) -> dict[str, str]:
    return {
        "run_manifest_receipt_sha256": manifest["receipt_sha256"],
        "atlas_binding_receipt_sha256": manifest["atlas"]["binding"][
            "receipt_sha256"
        ],
        "training_data_manifest_receipt_sha256": manifest["training_data"][
            "training_data_manifest_receipt_sha256"
        ],
    }


def _scientific_manifest_payload(payload: Mapping[str, object]) -> dict[str, object]:
    value = dict(payload)
    value.pop("run_id", None)
    return value


def _checkpoint_record_at_path(
    path: Path,
    relative_path: str,
    checkpoint: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    return {
        "relative_path": relative_path,
        "file_sha256": _file_sha256(path),
        "checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
        "trainer_manifest_receipt_sha256": checkpoint["manifest"][
            "receipt_sha256"
        ],
        "global_step": int(checkpoint["global_step"]),
        **_training_run_binding(manifest),
    }


def _checkpoint_record(
    run_root: Path,
    relative_path: str,
    checkpoint: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    return _checkpoint_record_at_path(
        _run_file(run_root, relative_path), relative_path, checkpoint, manifest
    )


def _initial_run_state(
    manifest: Mapping[str, object], checkpoint: Mapping[str, object]
) -> dict[str, object]:
    return _with_receipt(
        {
            "schema_version": RECEIPT_BOUND_TRAINING_STATE_V6_SCHEMA,
            "run_id": manifest["run_id"],
            "run_manifest_receipt_sha256": manifest["receipt_sha256"],
            "global_step": 0,
            "latest_checkpoint": checkpoint,
            "committed_steps": [],
            "immutable_archives": [],
            "probabilities_calibrated": False,
            "probability_status": RAW_UNCALIBRATED,
            "release_qualifying": False,
        }
    )


def initialize_receipt_bound_training_run_v6(
    run_directory,
    *,
    cache_directory,
    expected_training_data_manifest_receipt_sha256: str,
    allen_atlas_bundle,
    finite_psf_capability,
    model_kwargs,
    training_config,
    runner_config,
    device="cuda",
    expected_git_commit: str | None = None,
) -> dict[str, object]:
    """Create an atomic I:-only run from a fresh random v6 trainer."""
    run_root = _i_path(run_directory)
    cache_root = _i_path(cache_directory, must_exist=True)
    if run_root.exists():
        raise FileExistsError("a fresh v6 run directory must not already exist")
    config = _validated_runner_config(runner_config)
    _assert_no_forbidden_dependencies(model_kwargs)
    _assert_no_forbidden_dependencies(training_config)
    finite_row_binding_v6.verify_finite_psf_model_capability_v6(
        finite_psf_capability
    )
    allen_atlas_v6.verify_bound_allen_atlas_v6(allen_atlas_bundle)
    resolved_allen = allen_atlas_v6.resolve_bound_allen_atlas_v6(
        allen_atlas_bundle
    )
    allen_atlas_v6.verify_pinned_allen_raw_sources_v6(
        resolved_allen["binding"]
    )
    catalogue = resolved_allen["catalogue"]
    support_index = resolved_allen["support_index"]
    atlas = resolved_allen["atlas_volume_float32"]
    atlas_binding = _plain(resolved_allen["binding"])
    catalogue_binding_v3.verify_catalogue_binding_v3(catalogue)
    catalogue_runtime = catalogue_runtime_v6.make_complete_catalogue_runtime_v6(
        catalogue,
        expected_catalogue_receipt_sha256=catalogue["receipt_sha256"],
        device=device,
        dtype=torch.float32,
    )
    catalogue_runtime_v6.verify_complete_catalogue_runtime_v6(catalogue_runtime)
    runtime_binding = _plain(catalogue_runtime.binding)
    canonical_device = runtime_binding["device"]
    if catalogue_runtime.cell_count != FULL_CATALOGUE_CELL_COUNT_V6:
        raise ValueError("runner requires the complete 98,304-cell runtime")
    training_data, _ = _training_data_record(
        cache_root, expected_training_data_manifest_receipt_sha256
    )
    commit = _current_git_commit()
    if expected_git_commit is not None and expected_git_commit.lower() != commit:
        raise ValueError("current git commit differs from the trusted expected commit")
    _verify_declared_sources_match_git_commit(commit)

    staging = run_root.with_name(
        f".{run_root.name}.initializing-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True)
    try:
        for name in ("inputs", "checkpoints", "transactions"):
            (staging / name).mkdir()
        catalogue_relative = "inputs/complete_catalogue.pt"
        atlas_relative = "inputs/atlas_volume_float32.npy"
        support_index_relative = "inputs/allen_support_index.pt"
        _atomic_torch(staging / catalogue_relative, catalogue)
        _atomic_numpy(staging / atlas_relative, atlas)
        _atomic_torch(staging / support_index_relative, support_index)
        catalogue_record = {
            "relative_path": catalogue_relative,
            "file_sha256": _file_sha256(staging / catalogue_relative),
            "catalogue_id": catalogue["catalogue_id"],
            "catalogue_receipt_sha256": catalogue["receipt_sha256"],
            "cell_count": FULL_CATALOGUE_CELL_COUNT_V6,
            "runtime_binding": runtime_binding,
        }
        atlas_record = {
            "relative_path": atlas_relative,
            "file_sha256": _file_sha256(staging / atlas_relative),
            "support_index_relative_path": support_index_relative,
            "support_index_file_sha256": _file_sha256(
                staging / support_index_relative
            ),
            "binding": atlas_binding,
        }
        policy = _sampling_policy(training_data, config)
        manifest_core = {
            "schema_version": RECEIPT_BOUND_TRAINING_RUN_V6_SCHEMA,
            "data_role": "development-training",
            "retrieval_scope": "complete_98304_cell_catalogue",
            "git_commit": commit,
            "source_sha256": _source_receipts(),
            "execution": {
                "device": canonical_device,
                "catalogue_dtype": runtime_binding["dtype"],
                "atlas_channels": int(atlas.shape[0]),
            },
            "catalogue": catalogue_record,
            "atlas": atlas_record,
            "training_data": training_data,
            "finite_psf_capability": _plain(finite_psf_capability),
            "model_kwargs": _plain(model_kwargs),
            "training_config": _plain(training_config),
            "runner_config": _plain(config),
            "seed_record": {
                "model_initialization_seed": training_config.get("seed"),
                "row_selection_seed": config["row_selection_seed"],
                "cache_generation_lineage_sha256": training_data[
                    "generation_lineage_sha256"
                ],
            },
            "row_sampling_policy": policy,
            "training_commit_contract": _training_commit_contract(
                config["archive_checkpoint_interval_steps"]
            ),
            "initialization": "fresh_random_only",
            "prior_model_weight_dependencies": [],
            "prior_feature_dependencies": [],
            "prior_pseudolabel_dependencies": [],
            "probabilities_calibrated": False,
            "probability_status": RAW_UNCALIBRATED,
            "release_qualifying": False,
        }
        run_id = _hash_json(
            {
                "domain": RECEIPT_BOUND_TRAINING_RUN_V6_SCHEMA,
                "scientific_manifest": manifest_core,
            }
        )
        manifest = _with_receipt({**manifest_core, "run_id": run_id})
        trainer_state = staged_trainer_v6.initialize_staged_trainer_v6(
            catalogue_runtime,
            manifest["execution"]["atlas_channels"],
            model_kwargs,
            training_config,
            training_run_binding=_training_run_binding(manifest),
            device=canonical_device,
        )
        _atomic_json(staging / "run_manifest.json", manifest)
        checkpoint_relative = _RESUME_SLOTS[0]
        staged_trainer_v6.save_staged_checkpoint_v6(
            trainer_state, staging / checkpoint_relative
        )
        _fsync_existing_file(staging / checkpoint_relative)
        checkpoint_payload = staged_trainer_v6.load_staged_checkpoint_v6(
            staging / checkpoint_relative
        )
        checkpoint = _checkpoint_record(
            staging, checkpoint_relative, checkpoint_payload, manifest
        )
        run_state = _initial_run_state(manifest, checkpoint)
        _atomic_json(staging / "run_state.json", run_state)
        _verify_manifest(manifest)
        _verify_allen_inputs(staging, manifest)
        _verify_training_data(manifest)
        _verify_run_state(staging, manifest, run_state)
        os.replace(staging, run_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "run_directory": str(run_root),
        "manifest": manifest,
        "run_state": run_state,
    }


def _read_receipted_json(path: Path, schema: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != schema
        or value.get("receipt_sha256") != _hash_json(_payload(value))
    ):
        raise ValueError(f"{schema} receipt is invalid")
    return value


def _verify_manifest(manifest: Mapping[str, object]) -> None:
    payload = _payload(manifest) if isinstance(manifest, Mapping) else {}
    scientific = _scientific_manifest_payload(payload)
    if (
        set(manifest) != _RUN_MANIFEST_KEYS
        or manifest.get("schema_version") != RECEIPT_BOUND_TRAINING_RUN_V6_SCHEMA
        or manifest.get("receipt_sha256") != _hash_json(payload)
        or manifest.get("run_id")
        != _hash_json(
            {
                "domain": RECEIPT_BOUND_TRAINING_RUN_V6_SCHEMA,
                "scientific_manifest": scientific,
            }
        )
        or manifest.get("data_role") != "development-training"
        or manifest.get("retrieval_scope") != "complete_98304_cell_catalogue"
        or manifest.get("source_sha256") != _source_receipts()
        or manifest.get("initialization") != "fresh_random_only"
        or manifest.get("probabilities_calibrated") is not False
        or manifest.get("probability_status") != RAW_UNCALIBRATED
        or manifest.get("release_qualifying") is not False
        or manifest.get("prior_model_weight_dependencies") != []
        or manifest.get("prior_feature_dependencies") != []
        or manifest.get("prior_pseudolabel_dependencies") != []
    ):
        raise ValueError("receipt-bound v6 run manifest is invalid")
    _verify_declared_sources_match_git_commit(manifest["git_commit"])
    config = _validated_runner_config(manifest["runner_config"])
    training_data = manifest.get("training_data", {})
    if (
        not _is_sha256(
            training_data.get("training_data_manifest_receipt_sha256")
        )
        or training_data.get("training_data_manifest_receipt_sha256")
        != training_data.get("cache_manifest_receipt_sha256")
        or training_data.get("row_count")
        != len(training_data.get("ordered_row_identities", []))
        or training_data.get("row_count", 0) < 1
        or training_data.get("optimization_split") != "train"
        or manifest.get("row_sampling_policy")
        != _sampling_policy(training_data, config)
        or manifest.get("seed_record")
        != {
            "model_initialization_seed": manifest["training_config"].get("seed"),
            "row_selection_seed": config["row_selection_seed"],
            "cache_generation_lineage_sha256": training_data.get(
                "generation_lineage_sha256"
            ),
        }
        or manifest.get("training_commit_contract")
        != _training_commit_contract(config["archive_checkpoint_interval_steps"])
    ):
        raise ValueError("v6 data, seed, or sampling binding is invalid")
    for index, identity in enumerate(training_data["ordered_row_identities"]):
        if identity != _row_identity(
            {
                "lineage": {key: identity.get(key) for key in _FIVE_IDS},
                "training_row_id": identity.get("training_row_id"),
                "receipt_sha256": identity.get("training_row_receipt_sha256"),
            },
            index,
        ):
            raise ValueError("v6 ordered row identity table is invalid")
    atlas_binding = manifest.get("atlas", {}).get("binding", {})
    bound_catalogue = atlas_binding.get("catalogue", {})
    if (
        atlas_binding.get("schema_version") != ATLAS_BINDING_V6_SCHEMA
        or atlas_binding.get("receipt_sha256")
        != _hash_json(_payload(atlas_binding))
        or atlas_binding.get("receipt_sha256")
        not in set(
            allen_atlas_v6.ALLEN_ATLAS_BINDING_RECEIPT_BY_RASTER_V6.values()
        )
        or atlas_binding.get("decoded_atlas_receipt")
        != allen_atlas_v6.ATLAS_FLOAT32_RECEIPT_V6
        or bound_catalogue.get("cell_count")
        != FULL_CATALOGUE_CELL_COUNT_V6
        or bound_catalogue.get("catalogue_id")
        != manifest.get("catalogue", {}).get("catalogue_id")
        or bound_catalogue.get("catalogue_receipt_sha256")
        != manifest.get("catalogue", {}).get("catalogue_receipt_sha256")
        or manifest.get("catalogue", {}).get("cell_count")
        != FULL_CATALOGUE_CELL_COUNT_V6
        or manifest.get("catalogue", {}).get("runtime_binding", {}).get(
            "cell_count"
        )
        != FULL_CATALOGUE_CELL_COUNT_V6
        or manifest.get("execution", {}).get("device")
        != manifest["catalogue"]["runtime_binding"].get("device")
        or manifest.get("execution", {}).get("catalogue_dtype")
        != manifest["catalogue"]["runtime_binding"].get("dtype")
        or manifest.get("execution", {}).get("atlas_channels") != 2
        or not isinstance(
            manifest.get("atlas", {}).get("support_index_relative_path"), str
        )
        or not _is_sha256(
            manifest.get("atlas", {}).get("support_index_file_sha256")
        )
    ):
        raise ValueError("v6 atlas or complete-catalogue manifest binding is invalid")
    _assert_no_forbidden_dependencies(manifest["model_kwargs"])
    _assert_no_forbidden_dependencies(manifest["training_config"])
    finite_row_binding_v6.verify_finite_psf_model_capability_v6(
        manifest["finite_psf_capability"]
    )


def _verify_allen_inputs(
    run_root: Path, manifest: Mapping[str, object]
) -> tuple[Mapping[str, object], Mapping[str, object], np.ndarray, object]:
    atlas_record = manifest["atlas"]
    catalogue_record = manifest["catalogue"]
    allen_atlas_v6.verify_pinned_allen_raw_sources_v6(
        atlas_record["binding"]
    )
    atlas_path = _run_file(run_root, atlas_record["relative_path"])
    support_path = _run_file(
        run_root, atlas_record["support_index_relative_path"]
    )
    catalogue_path = _run_file(run_root, catalogue_record["relative_path"])
    if (
        _file_sha256(atlas_path) != atlas_record["file_sha256"]
        or _file_sha256(support_path)
        != atlas_record["support_index_file_sha256"]
        or _file_sha256(catalogue_path) != catalogue_record["file_sha256"]
    ):
        raise ValueError("a stored Allen v6 input file receipt is invalid")
    with atlas_path.open("rb") as stream:
        atlas = np.load(stream, allow_pickle=False)
    atlas = np.ascontiguousarray(atlas)
    support_index = torch.load(
        support_path, map_location="cpu", weights_only=False
    )
    catalogue = torch.load(
        catalogue_path, map_location="cpu", weights_only=False
    )
    bundle = allen_atlas_v6.restore_bound_allen_atlas_v6(
        atlas_volume_float32=atlas,
        support_index=support_index,
        catalogue=catalogue,
        binding=atlas_record["binding"],
    )
    dtype = _DTYPES.get(manifest["execution"]["catalogue_dtype"])
    if dtype is None:
        raise ValueError("stored v6 catalogue dtype is unsupported")
    runtime = catalogue_runtime_v6.make_complete_catalogue_runtime_v6(
        catalogue,
        expected_catalogue_receipt_sha256=catalogue_record[
            "catalogue_receipt_sha256"
        ],
        device=manifest["execution"]["device"],
        dtype=dtype,
    )
    if (
        catalogue.get("catalogue_id") != catalogue_record["catalogue_id"]
        or catalogue.get("receipt_sha256")
        != catalogue_record["catalogue_receipt_sha256"]
        or _plain(runtime.binding) != catalogue_record["runtime_binding"]
    ):
        raise ValueError("stored complete catalogue differs from its runtime binding")
    return catalogue, support_index, atlas, runtime


def _verify_training_data(
    manifest: Mapping[str, object]
) -> dict[str, object]:
    expected = manifest["training_data"]
    observed, cache_manifest = _training_data_record(
        _i_path(expected["cache_directory"], must_exist=True),
        expected["training_data_manifest_receipt_sha256"],
    )
    if observed != expected:
        raise ValueError("frozen v6 training data changed after pre-run binding")
    return cache_manifest


def _verify_trainer_checkpoint(
    run_root: Path,
    record: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    path = _run_file(run_root, record["relative_path"])
    if (
        _file_sha256(path) != record.get("file_sha256")
        or record.get("run_manifest_receipt_sha256")
        != manifest["receipt_sha256"]
        or {
            key: record.get(key)
            for key in (
                "run_manifest_receipt_sha256",
                "atlas_binding_receipt_sha256",
                "training_data_manifest_receipt_sha256",
            )
        }
        != _training_run_binding(manifest)
    ):
        raise ValueError("v6 checkpoint file or runner binding is invalid")
    checkpoint = staged_trainer_v6.load_staged_checkpoint_v6(path)
    trainer_manifest = checkpoint["manifest"]
    if (
        checkpoint.get("receipt_sha256")
        != record.get("checkpoint_receipt_sha256")
        or checkpoint.get("global_step") != record.get("global_step")
        or trainer_manifest.get("receipt_sha256")
        != record.get("trainer_manifest_receipt_sha256")
        or trainer_manifest.get("training_run_binding")
        != _training_run_binding(manifest)
        or trainer_manifest.get("catalogue_binding")
        != manifest["catalogue"]["runtime_binding"]
        or trainer_manifest.get("model_kwargs") != manifest["model_kwargs"]
        or trainer_manifest.get("training_config") != manifest["training_config"]
        or trainer_manifest.get("atlas_channels")
        != manifest["execution"]["atlas_channels"]
        or checkpoint.get("probabilities_calibrated") is not False
        or checkpoint.get("uncertainty_status") != RAW_UNCALIBRATED
        or checkpoint.get("learned_dependencies")
        != {"model_weights": [], "features": [], "pseudolabels": []}
    ):
        raise ValueError("v6 trainer checkpoint differs from the run manifest")
    return checkpoint


def _artifact_record(path: Path, relative_path: str, value) -> dict[str, object]:
    return {
        "relative_path": relative_path,
        "file_sha256": _file_sha256(path),
        "receipt_sha256": value["receipt_sha256"],
    }


def _verify_step_artifacts(
    run_root: Path,
    manifest: Mapping[str, object],
    run_state: Mapping[str, object],
    checkpoint: Mapping[str, object],
) -> list[dict[str, object]]:
    previous = manifest["receipt_sha256"]
    previous_transaction = manifest["receipt_sha256"]
    table = manifest["training_data"]["ordered_row_identities"]
    all_identities = []
    ledger = checkpoint.get("training_step_ledger")
    if not isinstance(ledger, list) or len(ledger) != run_state["global_step"]:
        raise ValueError("v6 checkpoint step ledger is incomplete")
    archive_by_step = {
        item["global_step"]: item for item in run_state["immutable_archives"]
    }
    for step, record in enumerate(run_state["committed_steps"]):
        transaction_record = record.get("transaction")
        if not isinstance(transaction_record, Mapping):
            raise ValueError("a committed v6 step lacks its atomic transaction")
        transaction_path = _run_file(
            run_root, transaction_record.get("relative_path", "")
        )
        report_path = _run_file(run_root, record["report"]["relative_path"])
        raw_path = _run_file(run_root, record["raw_output"]["relative_path"])
        if (
            _file_sha256(transaction_path)
            != transaction_record.get("file_sha256")
            or _file_sha256(report_path) != record["report"]["file_sha256"]
            or _file_sha256(raw_path) != record["raw_output"]["file_sha256"]
        ):
            raise ValueError("a committed v6 step artifact file changed")
        transaction = _read_receipted_json(
            transaction_path, RECEIPT_BOUND_TRAINING_TRANSACTION_V6_SCHEMA
        )
        report = _read_receipted_json(
            report_path, RECEIPT_BOUND_TRAINING_STEP_REPORT_V6_SCHEMA
        )
        raw = _read_receipted_json(
            raw_path, RECEIPT_BOUND_TRAINING_RAW_OUTPUT_V6_SCHEMA
        )
        indices = sample_training_row_indices_v6(manifest, step)
        identities = [table[index] for index in indices]
        archive = archive_by_step.get(step + 1)
        archive_receipt = (
            archive["checkpoint_receipt_sha256"] if archive is not None else None
        )
        ledger_entry = ledger[step]
        trainer_output = raw.get("trainer_output")
        losses = trainer_output.get("losses") if isinstance(trainer_output, Mapping) else None
        numeric_losses = (
            [item for item in losses.values() if isinstance(item, (int, float)) and not isinstance(item, bool)]
            if isinstance(losses, Mapping)
            else []
        )
        row_receipts_sha256 = _hash_json(
            {
                "schema_version": staged_trainer_v6.ROW_RECEIPTS_V6_SCHEMA,
                "row_receipts": report.get("row_receipts"),
            }
        )
        committed_without_transaction = {
            key: value for key, value in record.items() if key != "transaction"
        }
        if (
            set(transaction) != _TRANSACTION_KEYS
            or transaction_record.get("receipt_sha256")
            != transaction.get("receipt_sha256")
            or transaction.get("run_id") != manifest["run_id"]
            or transaction.get("run_manifest_receipt_sha256")
            != manifest["receipt_sha256"]
            or transaction.get("step") != step
            or transaction.get("previous_transaction_receipt_sha256")
            != previous_transaction
            or not _is_sha256(
                transaction.get("previous_run_state_receipt_sha256")
            )
            or transaction.get("committed_step")
            != committed_without_transaction
            or not isinstance(transaction.get("latest_checkpoint"), Mapping)
            or transaction.get("latest_checkpoint", {}).get(
                "checkpoint_receipt_sha256"
            )
            != record.get("committed_checkpoint_receipt_sha256")
            or (
                step == run_state["global_step"] - 1
                and transaction.get("latest_checkpoint")
                != run_state["latest_checkpoint"]
            )
            or transaction.get("archive_checkpoint") != archive
            or transaction.get("probabilities_calibrated") is not False
            or transaction.get("probability_status") != RAW_UNCALIBRATED
            or transaction.get("release_qualifying") is not False
            or record.get("step") != step
            or record.get("selection_receipt_sha256")
            != report.get("selection_receipt_sha256")
            or record["report"].get("receipt_sha256")
            != report.get("receipt_sha256")
            or record["raw_output"].get("receipt_sha256")
            != raw.get("receipt_sha256")
            or record.get("committed_checkpoint_receipt_sha256")
            != report.get("committed_checkpoint_receipt_sha256")
            or record.get("archive_checkpoint_receipt_sha256")
            != report.get("archive_checkpoint_receipt_sha256")
            or report.get("archive_checkpoint_receipt_sha256") != archive_receipt
            or (
                step == run_state["global_step"] - 1
                and report.get("committed_checkpoint_receipt_sha256")
                != run_state["latest_checkpoint"]["checkpoint_receipt_sha256"]
            )
            or report.get("step") != step
            or raw.get("step") != step
            or report.get("run_id") != manifest["run_id"]
            or raw.get("run_id") != manifest["run_id"]
            or report.get("run_manifest_receipt_sha256")
            != manifest["receipt_sha256"]
            or raw.get("run_manifest_receipt_sha256")
            != manifest["receipt_sha256"]
            or report.get("previous_step_report_receipt_sha256") != previous
            or report.get("row_indices") != indices
            or report.get("row_identities") != identities
            or report.get("training_row_ids")
            != [item["training_row_id"] for item in identities]
            or report.get("training_row_receipts_sha256")
            != [item["training_row_receipt_sha256"] for item in identities]
            or report.get("training_data_manifest_receipt_sha256")
            != manifest["training_data"]["training_data_manifest_receipt_sha256"]
            or raw.get("training_data_manifest_receipt_sha256")
            != manifest["training_data"]["training_data_manifest_receipt_sha256"]
            or raw.get("selection_receipt_sha256")
            != report.get("selection_receipt_sha256")
            or report.get("trainer_output_receipt_sha256")
            != trainer_output.get("receipt_sha256")
            or report.get("raw_output_receipt_sha256")
            != raw.get("receipt_sha256")
            or report.get("trainer_step_ledger_receipt_sha256")
            != ledger_entry.get("receipt_sha256")
            or report.get("row_receipts_sha256")
            != ledger_entry.get("row_receipts_sha256")
            or report.get("row_receipts_sha256") != row_receipts_sha256
            or report.get("input_mode") != ledger_entry.get("input_mode")
            or not isinstance(trainer_output, Mapping)
            or set(trainer_output) != _TRAINER_OUTPUT_KEYS
            or trainer_output.get("receipt_sha256")
            != _hash_json(_payload(trainer_output))
            or trainer_output.get("receipt_sha256")
            != ledger_entry.get("trainer_output_receipt_sha256")
            or trainer_output.get("schema_version")
            != staged_trainer_v6.STAGED_TRAINER_V6_SCHEMA
            or trainer_output.get("step") != step
            or trainer_output.get("phase") != ledger_entry.get("phase")
            or trainer_output.get("optimizer_step_applied") is not True
            or trainer_output.get("catalogue_cell_count")
            != FULL_CATALOGUE_CELL_COUNT_V6
            or trainer_output.get("probabilities_calibrated") is not False
            or trainer_output.get("probability_status") != RAW_UNCALIBRATED
            or trainer_output.get("refinement_ready_row_count")
            != ledger_entry.get("refinement_ready_row_count")
            or trainer_output.get("refinement_abstained_row_count")
            != ledger_entry.get("refinement_abstained_row_count")
            or not isinstance(trainer_output.get("objective"), (int, float))
            or isinstance(trainer_output.get("objective"), bool)
            or not np.isfinite(float(trainer_output.get("objective", np.nan)))
            or float(trainer_output.get("objective", -1.0)) < 0.0
            or not isinstance(
                trainer_output.get("preclip_gradient_norm"), (int, float)
            )
            or isinstance(trainer_output.get("preclip_gradient_norm"), bool)
            or not np.isfinite(
                float(trainer_output.get("preclip_gradient_norm", np.nan))
            )
            or float(trainer_output.get("preclip_gradient_norm", -1.0)) < 0.0
            or not isinstance(losses, Mapping)
            or not isinstance(losses.get("total"), (int, float))
            or isinstance(losses.get("total"), bool)
            or float(losses.get("total", np.nan))
            != float(trainer_output.get("objective", np.nan))
            or any(not np.isfinite(float(item)) for item in numeric_losses)
            or report.get("probability_status") != RAW_UNCALIBRATED
            or raw.get("probability_status") != RAW_UNCALIBRATED
            or report.get("probabilities_calibrated") is not False
            or raw.get("probabilities_calibrated") is not False
            or report.get("release_qualifying") is not False
            or raw.get("release_qualifying") is not False
        ):
            raise ValueError("committed v6 step ledger failed exact replay")
        all_identities.extend(identities)
        previous = report["receipt_sha256"]
        previous_transaction = transaction["receipt_sha256"]
    return all_identities


def _verify_run_state(
    run_root: Path,
    manifest: Mapping[str, object],
    run_state: Mapping[str, object],
) -> dict[str, object]:
    payload = _payload(run_state) if isinstance(run_state, Mapping) else {}
    if (
        set(run_state) != _RUN_STATE_KEYS
        or run_state.get("schema_version")
        != RECEIPT_BOUND_TRAINING_STATE_V6_SCHEMA
        or run_state.get("receipt_sha256") != _hash_json(payload)
        or run_state.get("run_id") != manifest["run_id"]
        or run_state.get("run_manifest_receipt_sha256")
        != manifest["receipt_sha256"]
        or run_state.get("global_step") != len(run_state.get("committed_steps", []))
        or run_state.get("probabilities_calibrated") is not False
        or run_state.get("probability_status") != RAW_UNCALIBRATED
        or run_state.get("release_qualifying") is not False
    ):
        raise ValueError("receipt-bound v6 run state is invalid")
    checkpoint = _verify_trainer_checkpoint(
        run_root, run_state["latest_checkpoint"], manifest
    )
    if checkpoint["global_step"] != run_state["global_step"]:
        raise ValueError("v6 run state and rolling checkpoint steps differ")
    expected_identities = _verify_step_artifacts(
        run_root, manifest, run_state, checkpoint
    )
    provenance = checkpoint.get("provenance_records", [])
    if len(provenance) != len(expected_identities) or any(
        any(observed.get(key) != expected[key] for key in _FIVE_IDS)
        or observed.get("training_row_id") != expected["training_row_id"]
        or observed.get("training_row_receipt_sha256")
        != expected["training_row_receipt_sha256"]
        for observed, expected in zip(provenance, expected_identities)
    ):
        raise ValueError("v6 checkpoint provenance differs from sampled row identities")
    expected_archive_steps = []
    for record in run_state["immutable_archives"]:
        archive = _verify_trainer_checkpoint(run_root, record, manifest)
        if archive["global_step"] != record["global_step"]:
            raise ValueError("immutable v6 checkpoint archive step differs")
        expected_archive_steps.append(record["global_step"])
    interval = manifest["runner_config"]["archive_checkpoint_interval_steps"]
    if expected_archive_steps != list(
        range(interval, run_state["global_step"] + 1, interval)
    ):
        raise ValueError("immutable v6 checkpoint archive schedule is incomplete")
    return checkpoint


def load_receipt_bound_training_run_v6(
    run_directory,
    *,
    expected_run_manifest_receipt_sha256: str | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """Verify every bound input and restore only the authenticated rolling state."""
    run_root = _i_path(run_directory, must_exist=True)
    manifest = _read_receipted_json(
        run_root / "run_manifest.json", RECEIPT_BOUND_TRAINING_RUN_V6_SCHEMA
    )
    _verify_manifest(manifest)
    if (
        expected_run_manifest_receipt_sha256 is not None
        and manifest["receipt_sha256"] != expected_run_manifest_receipt_sha256
    ):
        raise ValueError("v6 run manifest differs from the trusted expected receipt")
    if device is not None and not _same_device(
        device, manifest["execution"]["device"]
    ):
        raise ValueError("v6 resume device differs from its pre-run binding")
    catalogue, support_index, atlas, runtime = _verify_allen_inputs(
        run_root, manifest
    )
    cache_manifest = _verify_training_data(manifest)
    run_state = _read_receipted_json(
        run_root / "run_state.json", RECEIPT_BOUND_TRAINING_STATE_V6_SCHEMA
    )
    checkpoint = _verify_run_state(run_root, manifest, run_state)
    trainer_state = staged_trainer_v6.restore_staged_trainer_v6(
        checkpoint,
        runtime,
        training_run_binding=_training_run_binding(manifest),
        device=manifest["execution"]["device"],
    )
    return {
        "run_directory": str(run_root),
        "manifest": manifest,
        "run_state": run_state,
        "catalogue": catalogue,
        "catalogue_runtime": runtime,
        "atlas_volume": atlas,
        "support_index": support_index,
        "training_cache_manifest": cache_manifest,
        "trainer_state": trainer_state,
    }


def _verify_selected_batch(
    manifest: Mapping[str, object],
    selected: list[int],
    frozen: Mapping[str, object],
    batch: Mapping[str, object],
) -> list[dict[str, object]]:
    table = manifest["training_data"]["ordered_row_identities"]
    identities = [table[index] for index in selected]
    source = batch.get("frozen_row_source")
    if (
        frozen.get("training_data_manifest_receipt_sha256")
        != manifest["training_data"]["training_data_manifest_receipt_sha256"]
        or frozen.get("cache_manifest_receipt_sha256")
        != manifest["training_data"]["cache_manifest_receipt_sha256"]
        or frozen.get("row_indices") != selected
        or frozen.get("training_row_ids")
        != [item["training_row_id"] for item in identities]
        or frozen.get("training_row_receipts_sha256")
        != [item["training_row_receipt_sha256"] for item in identities]
        or not _is_sha256(frozen.get("selection_receipt_sha256"))
        or not isinstance(source, Mapping)
        or source.get("training_data_manifest_receipt_sha256")
        != manifest["training_data"]["training_data_manifest_receipt_sha256"]
        or source.get("cache_manifest_receipt_sha256")
        != manifest["training_data"]["cache_manifest_receipt_sha256"]
        or source.get("selection_receipt_sha256")
        != frozen.get("selection_receipt_sha256")
        or source.get("row_indices") != selected
        or source.get("training_row_ids") != frozen.get("training_row_ids")
        or source.get("training_row_receipts_sha256")
        != frozen.get("training_row_receipts_sha256")
    ):
        raise ValueError("v6 minibatch selection differs from the bound frozen cache")
    provenance = batch.get("provenance")
    if not isinstance(provenance, list) or len(provenance) != len(identities):
        raise ValueError("v6 minibatch provenance count changed")
    for expected, observed in zip(identities, provenance):
        if any(observed.get(key) != expected[key] for key in _FIVE_IDS) or (
            observed.get("training_row_id") != expected["training_row_id"]
            or observed.get("training_row_receipt_sha256")
            != expected["training_row_receipt_sha256"]
        ):
            raise ValueError("v6 minibatch failed exact five-ID preservation")
    return identities


def _next_resume_slot(run_state: Mapping[str, object]) -> str:
    current = run_state["latest_checkpoint"]["relative_path"]
    if current not in _RESUME_SLOTS:
        raise ValueError("current v6 rolling checkpoint is not in an authenticated slot")
    return _RESUME_SLOTS[1 - _RESUME_SLOTS.index(current)]


def _transaction_relative_directory(step: int) -> str:
    return f"transactions/step_{step:08d}"


def _previous_transaction_receipt(
    manifest: Mapping[str, object], run_state: Mapping[str, object]
) -> str:
    if not run_state["committed_steps"]:
        return manifest["receipt_sha256"]
    return run_state["committed_steps"][-1]["transaction"]["receipt_sha256"]


def _state_after_transaction(
    manifest: Mapping[str, object],
    old_state: Mapping[str, object],
    transaction: Mapping[str, object],
    transaction_record: Mapping[str, object],
) -> dict[str, object]:
    committed = {
        **_plain(transaction["committed_step"]),
        "transaction": _plain(transaction_record),
    }
    archives = list(old_state["immutable_archives"])
    if transaction["archive_checkpoint"] is not None:
        archives.append(_plain(transaction["archive_checkpoint"]))
    return _with_receipt(
        {
            "schema_version": RECEIPT_BOUND_TRAINING_STATE_V6_SCHEMA,
            "run_id": manifest["run_id"],
            "run_manifest_receipt_sha256": manifest["receipt_sha256"],
            "global_step": int(old_state["global_step"]) + 1,
            "latest_checkpoint": _plain(transaction["latest_checkpoint"]),
            "committed_steps": [*old_state["committed_steps"], committed],
            "immutable_archives": archives,
            "probabilities_calibrated": False,
            "probability_status": RAW_UNCALIBRATED,
            "release_qualifying": False,
        }
    )


def _adopt_completed_transaction(context: dict[str, object]) -> bool:
    run_root = Path(context["run_directory"])
    manifest = context["manifest"]
    old_state = context["run_state"]
    step = int(old_state["global_step"])
    relative_directory = _transaction_relative_directory(step)
    transaction_path = _run_file(
        run_root, f"{relative_directory}/transaction.json"
    )
    if not transaction_path.is_file():
        return False
    transaction = _read_receipted_json(
        transaction_path, RECEIPT_BOUND_TRAINING_TRANSACTION_V6_SCHEMA
    )
    if (
        set(transaction) != _TRANSACTION_KEYS
        or transaction.get("run_id") != manifest["run_id"]
        or transaction.get("run_manifest_receipt_sha256")
        != manifest["receipt_sha256"]
        or transaction.get("step") != step
        or transaction.get("previous_run_state_receipt_sha256")
        != old_state["receipt_sha256"]
        or transaction.get("previous_transaction_receipt_sha256")
        != _previous_transaction_receipt(manifest, old_state)
    ):
        raise ValueError("completed v6 transaction does not extend the current run state")
    transaction_record = _artifact_record(
        transaction_path,
        f"{relative_directory}/transaction.json",
        transaction,
    )
    new_state = _state_after_transaction(
        manifest, old_state, transaction, transaction_record
    )
    checkpoint = _verify_run_state(run_root, manifest, new_state)
    _atomic_json(run_root / "run_state.json", new_state)
    context["run_state"] = new_state
    context["trainer_state"] = staged_trainer_v6.restore_staged_trainer_v6(
        checkpoint,
        context["catalogue_runtime"],
        training_run_binding=_training_run_binding(manifest),
        device=manifest["execution"]["device"],
    )
    return True


def _apply_one_step(context: dict[str, object]) -> None:
    run_root = Path(context["run_directory"])
    manifest = context["manifest"]
    old_state = context["run_state"]
    if _adopt_completed_transaction(context):
        return
    trainer_state = context["trainer_state"]
    step = int(old_state["global_step"])
    selected = sample_training_row_indices_v6(manifest, step)
    frozen = training_data_v6.load_frozen_training_rows_v6(
        manifest["training_data"]["cache_directory"],
        selected,
        expected_manifest_receipt_sha256=manifest["training_data"][
            "training_data_manifest_receipt_sha256"
        ],
    )
    geometry = manifest["atlas"]["binding"]["geometry"]
    batch = training_data_v6.model_ready_rows_v6(
        frozen,
        context["catalogue"],
        context["catalogue_runtime"],
        context["atlas_volume"],
        origin_ap_dv_ml_um=geometry["origin_ap_dv_ml_um"],
        voxel_size_ap_dv_ml_um=geometry["voxel_size_ap_dv_ml_um"],
        finite_psf_capability=manifest["finite_psf_capability"],
        expected_training_data_manifest_receipt_sha256=manifest[
            "training_data"
        ]["training_data_manifest_receipt_sha256"],
    )
    identities = _verify_selected_batch(manifest, selected, frozen, batch)
    trainer_output = staged_trainer_v6.train_staged_step_v6(trainer_state, batch)
    if (
        trainer_output.get("step") != step
        or trainer_output.get("optimizer_step_applied") is not True
        or trainer_output.get("probabilities_calibrated") is not False
        or trainer_output.get("probability_status") != RAW_UNCALIBRATED
        or trainer_state.get("global_step") != step + 1
    ):
        raise RuntimeError("v6 trainer returned a non-bound or calibrated step")
    ledger_entry = trainer_state.get("training_step_ledger", [])[-1]
    row_receipts_sha256 = _hash_json(
        {
            "schema_version": staged_trainer_v6.ROW_RECEIPTS_V6_SCHEMA,
            "row_receipts": _plain(batch["row_receipts"]),
        }
    )
    if (
        set(trainer_output) != _TRAINER_OUTPUT_KEYS
        or trainer_output.get("receipt_sha256")
        != _hash_json(_payload(trainer_output))
        or ledger_entry.get("step") != step
        or ledger_entry.get("frozen_row_selection")
        != _plain(batch["frozen_row_source"])
        or ledger_entry.get("row_receipts_sha256") != row_receipts_sha256
        or ledger_entry.get("trainer_output_receipt_sha256")
        != trainer_output.get("receipt_sha256")
    ):
        raise RuntimeError("v6 trainer output is not bound into its checkpoint ledger")

    checkpoint_relative = _next_resume_slot(old_state)
    checkpoint_path = _run_file(run_root, checkpoint_relative)
    staged_trainer_v6.save_staged_checkpoint_v6(trainer_state, checkpoint_path)
    _fsync_existing_file(checkpoint_path)
    checkpoint_payload = staged_trainer_v6.load_staged_checkpoint_v6(
        checkpoint_path
    )
    checkpoint = _checkpoint_record(
        run_root, checkpoint_relative, checkpoint_payload, manifest
    )
    relative_directory = _transaction_relative_directory(step)
    final_directory = _run_file(run_root, relative_directory)
    if final_directory.exists():
        raise FileExistsError("an unauthenticated v6 transaction already exists")
    transaction_root = run_root / "transactions"
    staging = transaction_root / (
        f".step_{step:08d}.building-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    interval = manifest["runner_config"]["archive_checkpoint_interval_steps"]
    archive = None
    try:
        if (step + 1) % interval == 0:
            archive_relative = f"{relative_directory}/archive_checkpoint.pt"
            archive_path = staging / "archive_checkpoint.pt"
            _copy_immutable(checkpoint_path, archive_path)
            archive_payload = staged_trainer_v6.load_staged_checkpoint_v6(
                archive_path
            )
            if (
                archive_payload["receipt_sha256"]
                != checkpoint_payload["receipt_sha256"]
            ):
                raise ValueError(
                    "immutable v6 archive differs from its rolling checkpoint"
                )
            archive = _checkpoint_record_at_path(
                archive_path, archive_relative, archive_payload, manifest
            )

        raw_output = _with_receipt(
            {
                "schema_version": RECEIPT_BOUND_TRAINING_RAW_OUTPUT_V6_SCHEMA,
                "run_id": manifest["run_id"],
                "run_manifest_receipt_sha256": manifest["receipt_sha256"],
                "step": step,
                "training_data_manifest_receipt_sha256": manifest[
                    "training_data"
                ]["training_data_manifest_receipt_sha256"],
                "selection_receipt_sha256": frozen[
                    "selection_receipt_sha256"
                ],
                "trainer_output": _plain(trainer_output),
                "probabilities_calibrated": False,
                "probability_status": RAW_UNCALIBRATED,
                "release_qualifying": False,
            }
        )
        raw_relative = f"{relative_directory}/raw_output.json"
        raw_path = staging / "raw_output.json"
        _immutable_json(raw_path, raw_output)
        previous = (
            old_state["committed_steps"][-1]["report"]["receipt_sha256"]
            if old_state["committed_steps"]
            else manifest["receipt_sha256"]
        )
        report = _with_receipt(
            {
                "schema_version": RECEIPT_BOUND_TRAINING_STEP_REPORT_V6_SCHEMA,
                "run_id": manifest["run_id"],
                "run_manifest_receipt_sha256": manifest["receipt_sha256"],
                "step": step,
                "previous_step_report_receipt_sha256": previous,
                "training_data_manifest_receipt_sha256": manifest[
                    "training_data"
                ]["training_data_manifest_receipt_sha256"],
                "selection_receipt_sha256": frozen[
                    "selection_receipt_sha256"
                ],
                "row_indices": selected,
                "training_row_ids": frozen["training_row_ids"],
                "training_row_receipts_sha256": frozen[
                    "training_row_receipts_sha256"
                ],
                "row_identities": identities,
                "input_mode": list(batch["input_mode"]),
                "row_receipts": _plain(batch["row_receipts"]),
                "row_receipts_sha256": row_receipts_sha256,
                "trainer_step_ledger_receipt_sha256": ledger_entry[
                    "receipt_sha256"
                ],
                "trainer_output_receipt_sha256": trainer_output[
                    "receipt_sha256"
                ],
                "raw_output_receipt_sha256": raw_output["receipt_sha256"],
                "committed_checkpoint_receipt_sha256": checkpoint[
                    "checkpoint_receipt_sha256"
                ],
                "archive_checkpoint_receipt_sha256": (
                    archive["checkpoint_receipt_sha256"] if archive else None
                ),
                "probabilities_calibrated": False,
                "probability_status": RAW_UNCALIBRATED,
                "release_qualifying": False,
            }
        )
        report_relative = f"{relative_directory}/report.json"
        report_path = staging / "report.json"
        _immutable_json(report_path, report)
        committed = {
            "step": step,
            "selection_receipt_sha256": frozen["selection_receipt_sha256"],
            "committed_checkpoint_receipt_sha256": checkpoint[
                "checkpoint_receipt_sha256"
            ],
            "archive_checkpoint_receipt_sha256": (
                archive["checkpoint_receipt_sha256"] if archive else None
            ),
            "report": _artifact_record(report_path, report_relative, report),
            "raw_output": _artifact_record(raw_path, raw_relative, raw_output),
        }
        transaction = _with_receipt(
            {
                "schema_version": RECEIPT_BOUND_TRAINING_TRANSACTION_V6_SCHEMA,
                "run_id": manifest["run_id"],
                "run_manifest_receipt_sha256": manifest["receipt_sha256"],
                "step": step,
                "previous_run_state_receipt_sha256": old_state[
                    "receipt_sha256"
                ],
                "previous_transaction_receipt_sha256": (
                    _previous_transaction_receipt(manifest, old_state)
                ),
                "latest_checkpoint": checkpoint,
                "archive_checkpoint": archive,
                "committed_step": committed,
                "probabilities_calibrated": False,
                "probability_status": RAW_UNCALIBRATED,
                "release_qualifying": False,
            }
        )
        transaction_path = staging / "transaction.json"
        _immutable_json(transaction_path, transaction)
        transaction_record = _artifact_record(
            transaction_path,
            f"{relative_directory}/transaction.json",
            transaction,
        )
        os.replace(staging, final_directory)
        new_state = _state_after_transaction(
            manifest, old_state, transaction, transaction_record
        )
        _verify_run_state(run_root, manifest, new_state)
        _atomic_json(run_root / "run_state.json", new_state)
        context["run_state"] = new_state
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def run_receipt_bound_training_steps_v6(
    run_directory,
    *,
    step_count: int = 1,
    expected_run_manifest_receipt_sha256: str | None = None,
) -> dict[str, object]:
    """Apply and durably publish a positive number of authenticated v6 steps."""
    if not isinstance(step_count, int) or isinstance(step_count, bool) or step_count < 1:
        raise ValueError("step_count must be a positive integer")
    context = load_receipt_bound_training_run_v6(
        run_directory,
        expected_run_manifest_receipt_sha256=expected_run_manifest_receipt_sha256,
    )
    for _ in range(step_count):
        _apply_one_step(context)
    return load_receipt_bound_training_run_v6(
        run_directory,
        expected_run_manifest_receipt_sha256=context["manifest"][
            "receipt_sha256"
        ],
    )


__all__ = [
    "ANIMAL_AWARE_SAMPLING_V6_SCHEMA",
    "ATLAS_BINDING_V6_SCHEMA",
    "RECEIPT_BOUND_TRAINING_RAW_OUTPUT_V6_SCHEMA",
    "RECEIPT_BOUND_TRAINING_RUN_V6_SCHEMA",
    "RECEIPT_BOUND_TRAINING_STATE_V6_SCHEMA",
    "RECEIPT_BOUND_TRAINING_STEP_REPORT_V6_SCHEMA",
    "initialize_receipt_bound_training_run_v6",
    "load_receipt_bound_training_run_v6",
    "run_receipt_bound_training_steps_v6",
    "sample_training_row_indices_v6",
]
