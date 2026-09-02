"""Prepared, resumable I:-only development runner for the v3 joint model."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_catalogue_v3 as catalogue_v3
import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_row_cache_v3 as row_cache_v3
import training.arbitrary_plane_staged_training as staged_training
from training.arbitrary_plane_joint_model import ArbitraryPlaneJointModel
from training.arbitrary_plane_training_bank_v3 import (
    TRAINING_CANDIDATE_BANK_SCOPE,
    make_training_candidate_batch_v3,
)


TRAINING_RUN_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-training-run/v3"
TRAINING_RUN_STATE_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-training-run-state/v3"
TRAINING_STEP_REPORT_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-training-step-report/v3"
TRAINING_COMMIT_CONTRACT_V3_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-training-commit-contract/v3"
)
ATLAS_BINDING_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-atlas-binding/v3"
FINITE_PSF_CONTRACT_V3_SCHEMA = "anatomy-tracker.finite-psf-training-contract/v3"
FINITE_PSF_TRAINING_SOURCE_V4_SCHEMA = (
    "anatomy-tracker.finite-psf-training-schedule-source/v4"
)
ROW_SAMPLING_POLICY_V3_SCHEMA = (
    "anatomy-tracker.frozen-cache-row-sampling-policy/v3"
)
DEFAULT_CHECKPOINT_COMMIT_INTERVAL_ATTEMPTS = 25
DEVELOPMENT_DATA_ROLE = staged_training.DEVELOPMENT_DATA_ROLE
RUNNER_CONFIG_KEYS = {
    "target_applied_steps",
    "batch_size",
    "candidate_bank_size",
    "row_selection_seed",
    "candidate_bank_root_seed",
    "axial_offsets_um",
    "axial_weights",
    "archive_checkpoint_interval_applied_steps",
    "checkpoint_commit_interval_attempts",
}
RUNNER_SOURCE_FILES = (
    "training/arbitrary_plane_psf_v4.py",
    "training/arbitrary_plane_row_cache_v3.py",
    "training/arbitrary_plane_training_runner_v3.py",
)


def _complete_model_kwargs(model_kwargs):
    signature = inspect.signature(ArbitraryPlaneJointModel)
    if not isinstance(model_kwargs, dict) or set(model_kwargs) - set(signature.parameters):
        raise ValueError("model kwargs contain unknown constructor fields")
    complete = {}
    for name, parameter in signature.parameters.items():
        if name in model_kwargs:
            complete[name] = model_kwargs[name]
        elif parameter.default is not inspect.Parameter.empty:
            complete[name] = parameter.default
        else:
            raise ValueError("model kwargs omit a required constructor field")
    return _plain(complete)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    return value


def _canonical_json(value):
    return json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_json(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _i_path(path):
    resolved = Path(path).resolve()
    if os.path.splitdrive(str(resolved))[0].upper() != "I:":
        raise ValueError("arbitrary-plane training runs and inputs must use only I:")
    return resolved


def _relative_child(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("training-run asset escapes its I:-drive run directory")
    return path


def _atomic_json(path, value):
    target = _i_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical_json(value))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _atomic_npy(path, value):
    target = _i_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _atomic_npz(path, arrays):
    target = _i_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: np.ascontiguousarray(value)
                for name, value in sorted(arrays.items())
            },
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _source_receipts():
    root = Path(__file__).resolve().parents[1]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in RUNNER_SOURCE_FILES
    }


@contextmanager
def _exclusive_run_lock(run_root):
    lock_path = _i_path(run_root) / ".training_runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RuntimeError("another process already owns this training run") from error
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _verify_catalogue(catalogue):
    arrays = catalogue.get("arrays", {})
    tensor_map = {
        "cell_id": "cell_id_int64",
        "cell_states": "cell_states_float64",
        "cell_log_mass": "cell_log_mass_float64",
        "representation_log_weight": "representation_log_weight_float64",
        "representation_to_canonical_raster_affine": "representation_to_canonical_raster_affine_float64",
    }
    if (
        catalogue.get("schema_version") != catalogue_v3.CATALOGUE_V3_SCHEMA
        or catalogue.get("array_receipts")
        != {name: catalogue_v3._array_receipt(value) for name, value in arrays.items()}
        or catalogue.get("receipt_sha256")
        != catalogue_v3._hash(catalogue_v3.catalogue_receipt_v3(catalogue))
        or set(catalogue.get("tensors", {})) != set(tensor_map)
    ):
        raise ValueError("catalogue failed its live v3 receipt")
    for tensor_name, array_name in tensor_map.items():
        expected = torch.from_numpy(np.asarray(arrays[array_name]))
        if tensor_name != "cell_id":
            expected = expected[None]
        if not torch.equal(torch.as_tensor(catalogue["tensors"][tensor_name]).cpu(), expected):
            raise ValueError("catalogue tensors differ from receipt-bound arrays")
    return True


def make_atlas_binding_v3(atlas_volume, *, source_assets, preprocessing):
    atlas = np.ascontiguousarray(_tensor_to_numpy(atlas_volume))
    assets = []
    for asset in source_assets:
        path = _i_path(asset["path"])
        observed = _file_sha256(path)
        if observed != asset["sha256"]:
            raise ValueError("atlas source-asset hash differs")
        assets.append(
            {
                "path": str(path),
                "role": str(asset["role"]),
                "sha256": observed,
            }
        )
    if atlas.ndim != 4 or atlas.shape[0] < 1 or not assets or not isinstance(preprocessing, dict):
        raise ValueError("atlas binding requires a channel-first volume, source assets, and preprocessing")
    payload = {
        "schema_version": ATLAS_BINDING_V3_SCHEMA,
        "array_receipt": acquisition_v2._array_receipt(atlas),
        "source_assets": assets,
        "preprocessing": _plain(preprocessing),
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    return {**payload, "receipt_sha256": _hash_json(payload)}


def verify_atlas_binding_v3(binding, atlas_volume):
    atlas = np.ascontiguousarray(_tensor_to_numpy(atlas_volume))
    payload = {key: value for key, value in binding.items() if key != "receipt_sha256"}
    if (
        binding.get("receipt_sha256") != _hash_json(payload)
        or payload.get("schema_version") != ATLAS_BINDING_V3_SCHEMA
        or payload.get("array_receipt") != acquisition_v2._array_receipt(atlas)
        or not payload.get("source_assets")
        or any(
            payload.get(name) != []
            for name in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    ):
        raise ValueError("atlas binding is invalid or learned-dependent")
    return True


def _finite_psf_contract(config):
    offsets = np.asarray(config["axial_offsets_um"], dtype=np.float64)
    weights = np.asarray(config["axial_weights"], dtype=np.float64)
    offset_tolerance = 16.0 * np.finfo(np.float32).eps * max(
        float(np.abs(offsets).max(initial=0.0)), 1.0
    )
    weight_tolerance = 16.0 * np.finfo(np.float32).eps
    payload = {
        "schema_version": FINITE_PSF_CONTRACT_V3_SCHEMA,
        "axial_offsets_um": offsets.tolist(),
        "axial_weights": weights.tolist(),
        "axial_offsets_receipt": acquisition_v2._array_receipt(offsets),
        "axial_weights_receipt": acquisition_v2._array_receipt(weights),
        "validation_dtype": "float32",
        "offset_symmetry_atol": offset_tolerance,
        "weight_symmetry_atol": weight_tolerance,
        "strictly_positive_weights": True,
        "unit_mass_atol": 1.0e-7,
        "normalization": "strictly positive symmetric discrete unit mass",
    }
    return {**payload, "receipt_sha256": _hash_json(payload)}


def _per_row_finite_psf_training_source(
    cache_manifest,
    capability,
    render_mode,
    axial_sample_count,
):
    psf_v4.verify_finite_psf_model_capability_v4(capability)
    allowed_sample_counts = {
        "finite_boxcar": psf_v4.PRODUCTION_AXIAL_SAMPLE_COUNT,
        "centre_plane_ablation": 1,
    }
    if (
        render_mode not in allowed_sample_counts
        or not isinstance(axial_sample_count, int)
        or isinstance(axial_sample_count, bool)
        or axial_sample_count != allowed_sample_counts[render_mode]
    ):
        raise ValueError("v4 run PSF render mode or sample count is unsupported")
    payload = {
        "schema_version": FINITE_PSF_TRAINING_SOURCE_V4_SCHEMA,
        "schedule_source": "authenticated-per-row",
        "training_row_schema_version": psf_v4.TRAINING_ROW_V4_SCHEMA,
        "training_row_contract_field": "finite_psf_contract",
        "render_mode": render_mode,
        "row_schedule_shape": [axial_sample_count],
        "finite_psf_capability_receipt_sha256": capability[
            "receipt_sha256"
        ],
        "frozen_cache_manifest_receipt_sha256": cache_manifest[
            "receipt_sha256"
        ],
        "per_row_receipt_binding": (
            "each ordered cache training_row_receipt_sha256 authenticates its exact "
            "finite_psf_contract and slab_observation_v4_receipt_sha256"
        ),
        "global_schedule_fallback": None,
        "unknown_thickness_policy": "reject",
    }
    return {**payload, "receipt_sha256": _hash_json(payload)}


def _row_sampling_policy(cache_manifest, runner_config, training_config):
    payload = {
        "schema_version": ROW_SAMPLING_POLICY_V3_SCHEMA,
        "algorithm": "seeded-PCG64DXSM-uniform-without-replacement/v3",
        "population": "every row in the exact frozen cache manifest",
        "frozen_cache_manifest_receipt_sha256": cache_manifest["receipt_sha256"],
        "frozen_cache_row_count": int(cache_manifest["row_count"]),
        "batch_size": int(runner_config["batch_size"]),
        "row_selection_seed": str(runner_config["row_selection_seed"]),
        "seed_derivation_domain": "anatomy-tracker.training-row-order/v3",
        "phase_policy": "identical uniform full-cache mixture in pose-warmup and joint phases",
        "pose_warmup_eligibility": (
            "identity-pose and nonidentity-deformed rows are both eligible; "
            "the deformation decoder is frozen and only the pose objective is optimized"
        ),
        "joint_phase_eligibility": (
            "identity-pose and nonidentity-deformed rows are both eligible"
        ),
        "pose_warmup_steps": int(training_config["pose_warmup_steps"]),
        "phase_aware_resampling": False,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    return {**payload, "receipt_sha256": _hash_json(payload)}


def _training_commit_contract(runner_config):
    payload = {
        "schema_version": TRAINING_COMMIT_CONTRACT_V3_SCHEMA,
        "checkpoint_commit_interval_attempts": int(
            runner_config["checkpoint_commit_interval_attempts"]
        ),
        "resume_checkpoint_slots": [
            "checkpoints/resume_slot_0.pt",
            "checkpoints/resume_slot_1.pt",
        ],
        "publication_order": [
            "unreferenced-checkpoint-write-and-fsync",
            "unreferenced-per-attempt-report-write-and-fsync",
            "atomic-run-state-replacement",
        ],
        "commit_boundaries": [
            "attempt-interval",
            "archive-applied-step",
            "target-applied-step",
            "bounded-call-end",
            "safe-pre-update-exception",
        ],
        "crash_recovery": (
            "load the last authenticated run state and deterministically replay only "
            "attempts that were not published by that state"
        ),
        "per_attempt_reports_preserved": True,
    }
    return {**payload, "receipt_sha256": _hash_json(payload)}


def _validate_runner_config(
    config,
    *,
    catalogue_cell_count,
    cache_row_count,
    training_top_k,
    pose_warmup_steps,
    refinement_steps,
    joint_pose_only_steps,
    per_row_finite_psf=False,
):
    config = _plain(config)
    config.setdefault(
        "checkpoint_commit_interval_attempts",
        DEFAULT_CHECKPOINT_COMMIT_INTERVAL_ATTEMPTS,
    )
    offsets = np.asarray(config.get("axial_offsets_um", ()), dtype=np.float64)
    weights = np.asarray(config.get("axial_weights", ()), dtype=np.float64)
    offset_tolerance = 16.0 * np.finfo(np.float32).eps * max(
        float(np.abs(offsets).max(initial=0.0)), 1.0
    )
    weight_tolerance = 16.0 * np.finfo(np.float32).eps
    if (
        set(config) != RUNNER_CONFIG_KEYS
        or not isinstance(config["target_applied_steps"], int)
        or isinstance(config["target_applied_steps"], bool)
        or config["target_applied_steps"] < 1
        or not isinstance(config["batch_size"], int)
        or isinstance(config["batch_size"], bool)
        or not 1 <= config["batch_size"] <= cache_row_count
        or not isinstance(config["candidate_bank_size"], int)
        or isinstance(config["candidate_bank_size"], bool)
        or not 5 <= config["candidate_bank_size"] < catalogue_cell_count
        or not isinstance(training_top_k, int)
        or isinstance(training_top_k, bool)
        or not 1 <= training_top_k <= config["candidate_bank_size"]
        or not isinstance(refinement_steps, int)
        or isinstance(refinement_steps, bool)
        or refinement_steps < 1
        or not isinstance(pose_warmup_steps, int)
        or isinstance(pose_warmup_steps, bool)
        or pose_warmup_steps < 1
        or not isinstance(joint_pose_only_steps, int)
        or isinstance(joint_pose_only_steps, bool)
        or not 0 <= joint_pose_only_steps <= refinement_steps + 1
        or (
            config["target_applied_steps"] > pose_warmup_steps
            and joint_pose_only_steps >= refinement_steps + 1
        )
        or not isinstance(config["archive_checkpoint_interval_applied_steps"], int)
        or isinstance(config["archive_checkpoint_interval_applied_steps"], bool)
        or config["archive_checkpoint_interval_applied_steps"] < 1
        or not isinstance(config["checkpoint_commit_interval_attempts"], int)
        or isinstance(config["checkpoint_commit_interval_attempts"], bool)
        or config["checkpoint_commit_interval_attempts"] < 1
        or (
            per_row_finite_psf
            and (offsets.ndim != 1 or weights.ndim != 1 or offsets.size or weights.size)
        )
        or (
            not per_row_finite_psf
            and (
                offsets.ndim != 1
                or weights.shape != offsets.shape
                or offsets.size < 1
                or not np.isfinite(offsets).all()
                or not np.isfinite(weights).all()
                or np.any(weights <= 0.0)
                or not math.isclose(
                    float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-7
                )
                or not np.allclose(
                    offsets, -offsets[::-1], rtol=0.0, atol=offset_tolerance
                )
                or not np.allclose(
                    weights, weights[::-1], rtol=0.0, atol=weight_tolerance
                )
            )
        )
    ):
        raise ValueError("runner config must specify bounded conditional training and normalized PSF weights")
    _canonical_json(config)
    return config


def _save_catalogue_inputs(run_root, catalogue):
    arrays_relative = "inputs/catalogue_arrays.npz"
    metadata_relative = "inputs/catalogue_metadata.json"
    metadata = {
        key: value for key, value in catalogue.items() if key not in ("arrays", "tensors")
    }
    _atomic_npz(run_root / arrays_relative, catalogue["arrays"])
    _atomic_json(run_root / metadata_relative, metadata)
    return {
        "metadata_relative_path": metadata_relative,
        "metadata_file_sha256": _file_sha256(run_root / metadata_relative),
        "arrays_relative_path": arrays_relative,
        "arrays_file_sha256": _file_sha256(run_root / arrays_relative),
        "catalogue_id": catalogue["catalogue_id"],
        "catalogue_receipt_sha256": catalogue["receipt_sha256"],
        "catalogue_cell_count": int(catalogue["counts"]["cell_count"]),
    }


def _load_catalogue_inputs(run_root, record):
    metadata_path = _relative_child(run_root, record["metadata_relative_path"])
    arrays_path = _relative_child(run_root, record["arrays_relative_path"])
    if (
        _file_sha256(metadata_path) != record["metadata_file_sha256"]
        or _file_sha256(arrays_path) != record["arrays_file_sha256"]
    ):
        raise ValueError("persisted catalogue input hash differs")
    catalogue = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as stored:
        catalogue["arrays"] = {
            name: np.ascontiguousarray(stored[name]) for name in stored.files
        }
    arrays = catalogue["arrays"]
    catalogue["tensors"] = {
        "cell_id": torch.from_numpy(arrays["cell_id_int64"]),
        "cell_states": torch.from_numpy(arrays["cell_states_float64"])[None],
        "cell_log_mass": torch.from_numpy(arrays["cell_log_mass_float64"])[None],
        "representation_log_weight": torch.from_numpy(
            arrays["representation_log_weight_float64"]
        )[None],
        "representation_to_canonical_raster_affine": torch.from_numpy(
            arrays["representation_to_canonical_raster_affine_float64"]
        )[None],
    }
    _verify_catalogue(catalogue)
    if (
        catalogue["catalogue_id"] != record["catalogue_id"]
        or catalogue["receipt_sha256"] != record["catalogue_receipt_sha256"]
        or catalogue["counts"]["cell_count"] != record["catalogue_cell_count"]
    ):
        raise ValueError("persisted catalogue identity differs from run manifest")
    return catalogue


def _manifest_payload(manifest):
    return {key: value for key, value in manifest.items() if key != "receipt_sha256"}


def _with_receipt(payload):
    payload = _plain(payload)
    return {**payload, "receipt_sha256": _hash_json(payload)}


def _scientific_run_id_payload(core):
    payload = dict(core)
    payload.pop("training_commit_contract", None)
    runner_config = dict(payload["runner_config"])
    runner_config.pop("checkpoint_commit_interval_attempts", None)
    payload["runner_config"] = runner_config
    return payload


def initialize_training_run_v3(
    run_directory,
    *,
    cache_directory,
    expected_generator_binding,
    catalogue,
    atlas_volume,
    atlas_source_assets,
    atlas_preprocessing,
    model_kwargs,
    training_config,
    runner_config,
    device="cuda",
    finite_psf_capability=None,
):
    """Freeze all inputs and create a fresh random initialization checkpoint."""
    run_root = _i_path(run_directory)
    model_kwargs = _complete_model_kwargs(model_kwargs)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError("training-run directory must be empty before initialization")
    cache_root = _i_path(cache_directory)
    cache_manifest = row_cache_v3.load_training_row_cache_manifest_v3(
        cache_root, expected_generator_binding=expected_generator_binding
    )
    if cache_manifest["status"] != row_cache_v3.FROZEN_CACHE_STATUS or cache_manifest["row_count"] < 1:
        raise ValueError("training requires a nonempty, fully audited frozen row cache")
    cache_audit = row_cache_v3.audit_training_row_cache_v3(cache_root)
    expected_row_schema = (
        psf_v4.TRAINING_ROW_V4_SCHEMA
        if finite_psf_capability is not None
        else "anatomy-tracker.arbitrary-plane-training-row/v3"
    )
    if cache_audit.get("training_row_schema_versions") != [expected_row_schema]:
        raise ValueError("frozen cache row PSF capability differs from the runner")
    finite_psf_render_mode = None
    finite_psf_axial_sample_count = None
    if finite_psf_capability is not None:
        modes = cache_audit.get("finite_psf_render_modes")
        sample_counts = cache_audit.get("finite_psf_axial_sample_counts")
        if len(modes or ()) != 1 or len(sample_counts or ()) != 1:
            raise ValueError(
                "one v4 run/cache must use one PSF render mode and sample count"
            )
        finite_psf_render_mode = modes[0]
        finite_psf_axial_sample_count = sample_counts[0]
    _verify_catalogue(catalogue)
    runner_config = _validate_runner_config(
        runner_config,
        catalogue_cell_count=int(catalogue["counts"]["cell_count"]),
        cache_row_count=int(cache_manifest["row_count"]),
        training_top_k=training_config.get("top_k"),
        pose_warmup_steps=training_config.get("pose_warmup_steps"),
        refinement_steps=training_config.get("refinement_steps"),
        joint_pose_only_steps=training_config.get("joint_pose_only_steps"),
        per_row_finite_psf=finite_psf_capability is not None,
    )
    if finite_psf_capability is not None:
        psf_v4.verify_finite_psf_model_capability_v4(finite_psf_capability)
    atlas = np.ascontiguousarray(_tensor_to_numpy(atlas_volume), dtype=np.float32)
    atlas_binding = make_atlas_binding_v3(
        atlas,
        source_assets=atlas_source_assets,
        preprocessing=atlas_preprocessing,
    )
    geometry = catalogue["support_geometry"]
    if tuple(atlas.shape[-3:]) != tuple(geometry["support_mask_receipt"]["shape"]):
        raise ValueError("atlas volume shape differs from the catalogue support asset")
    run_root.mkdir(parents=True, exist_ok=True)
    for name in ("inputs", "checkpoints", "reports"):
        (run_root / name).mkdir(exist_ok=True)
    catalogue_record = _save_catalogue_inputs(run_root, catalogue)
    atlas_relative = "inputs/atlas_volume_float32.npy"
    _atomic_npy(run_root / atlas_relative, atlas)
    state = staged_training.initialize_staged_training(
        model_kwargs,
        training_config,
        catalogue_id=catalogue["catalogue_id"],
        catalogue_receipt_sha256=catalogue["receipt_sha256"],
        catalogue_cell_count=int(catalogue["counts"]["cell_count"]),
        generator_ids=cache_manifest["generator_binding"]["generator_ids"],
        device=device,
        finite_psf_capability=finite_psf_capability,
    )
    core = {
        "schema_version": TRAINING_RUN_V3_SCHEMA,
        "data_role": DEVELOPMENT_DATA_ROLE,
        "retrieval_scope": TRAINING_CANDIDATE_BANK_SCOPE,
        "cache": {
            "directory": str(cache_root),
            "manifest_receipt_sha256": cache_manifest["receipt_sha256"],
            "row_count": cache_manifest["row_count"],
            "generator_binding_receipt_sha256": cache_manifest["generator_binding"][
                "receipt_sha256"
            ],
        },
        "catalogue": catalogue_record,
        "atlas": {
            "relative_path": atlas_relative,
            "file_sha256": _file_sha256(run_root / atlas_relative),
            "binding": atlas_binding,
        },
        "model_kwargs": _plain(model_kwargs),
        "training_config": _plain(training_config),
        "runner_config": runner_config,
        "row_sampling_policy": _row_sampling_policy(
            cache_manifest, runner_config, training_config
        ),
        "training_commit_contract": _training_commit_contract(runner_config),
        "seed_record": {
            "model_initialization_seed": _plain(training_config["seed"]),
            "row_selection_seed": runner_config["row_selection_seed"],
            "candidate_bank_root_seed": runner_config["candidate_bank_root_seed"],
            "cache_generation_seeds": cache_manifest["seed_record"],
        },
        "execution_device": str(torch.device(device)),
        "staged_training_binding": state["binding"],
        "runner_source_sha256": _source_receipts(),
        "forbidden_sources": [
            "public benchmark",
            "validation animals",
            "external-validation animals",
            "final-test animals",
        ],
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    if finite_psf_capability is not None:
        core["finite_psf_capability"] = _plain(finite_psf_capability)
        core["finite_psf_training_schedule_source"] = (
            _per_row_finite_psf_training_source(
                cache_manifest,
                finite_psf_capability,
                finite_psf_render_mode,
                finite_psf_axial_sample_count,
            )
        )
    else:
        core["finite_psf_contract"] = _finite_psf_contract(runner_config)
    core["run_id"] = _hash_json(
        {
            "domain": TRAINING_RUN_V3_SCHEMA,
            "scientific_core": _scientific_run_id_payload(core),
        }
    )
    manifest = _with_receipt(core)
    _atomic_json(run_root / "run_manifest.json", manifest)
    checkpoint_relative = "checkpoints/resume_slot_0.pt"
    staged_training.save_staged_training_checkpoint(state, run_root / checkpoint_relative)
    checkpoint_sha = _file_sha256(run_root / checkpoint_relative)
    run_state = _with_receipt(
        {
            "schema_version": TRAINING_RUN_STATE_V3_SCHEMA,
            "run_id": manifest["run_id"],
            "run_manifest_receipt_sha256": manifest["receipt_sha256"],
            "attempt_count": 0,
            "applied_step_count": 0,
            "latest_checkpoint": {
                "relative_path": checkpoint_relative,
                "file_sha256": checkpoint_sha,
            },
            "committed_reports": [],
        }
    )
    _atomic_json(run_root / "run_state.json", run_state)
    return manifest, run_state


def _load_manifest(run_root):
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    payload = _manifest_payload(manifest)
    identity_core = dict(payload)
    identity_core.pop("run_id", None)
    if (
        manifest.get("receipt_sha256") != _hash_json(payload)
        or payload.get("schema_version") != TRAINING_RUN_V3_SCHEMA
        or payload.get("run_id")
        != _hash_json(
            {
                "domain": TRAINING_RUN_V3_SCHEMA,
                "scientific_core": _scientific_run_id_payload(identity_core),
            }
        )
        or payload.get("data_role") != DEVELOPMENT_DATA_ROLE
        or payload.get("retrieval_scope") != TRAINING_CANDIDATE_BANK_SCOPE
        or payload.get("runner_source_sha256") != _source_receipts()
        or any(
            payload.get(name) != []
            for name in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    ):
        raise ValueError("training-run manifest failed authentication")
    capability = payload.get("finite_psf_capability")
    if capability is None:
        if (
            "finite_psf_capability" in payload
            or "finite_psf_training_schedule_source" in payload
            or payload.get("staged_training_binding", {}).get(
                "finite_psf_capability"
            )
            is not None
            or payload.get("finite_psf_contract")
            != _finite_psf_contract(payload.get("runner_config", {}))
        ):
            raise ValueError("legacy training run has an incomplete PSF capability binding")
    else:
        psf_v4.verify_finite_psf_model_capability_v4(capability)
        _validate_runner_config(
            payload["runner_config"],
            catalogue_cell_count=int(payload["catalogue"]["catalogue_cell_count"]),
            cache_row_count=int(payload["cache"]["row_count"]),
            training_top_k=payload["training_config"].get("top_k"),
            pose_warmup_steps=payload["training_config"].get("pose_warmup_steps"),
            refinement_steps=payload["training_config"].get("refinement_steps"),
            joint_pose_only_steps=payload["training_config"].get(
                "joint_pose_only_steps"
            ),
            per_row_finite_psf=True,
        )
        if "finite_psf_contract" in payload:
            raise ValueError("v4 training manifest must not bind one global PSF schedule")
        if payload.get("staged_training_binding", {}).get(
            "finite_psf_capability"
        ) != capability:
            raise ValueError("staged trainer finite-PSF capability differs")
    cache = payload["cache"]
    cache_manifest = row_cache_v3.load_training_row_cache_manifest_v3(
        cache["directory"],
        expected_receipt_sha256=cache["manifest_receipt_sha256"],
    )
    expected_finite_psf_training_source = None
    if capability is not None:
        finite_psf_training_source = payload.get(
            "finite_psf_training_schedule_source"
        )
        schedule_shape = (
            finite_psf_training_source.get("row_schedule_shape")
            if isinstance(finite_psf_training_source, dict)
            else None
        )
        if not isinstance(schedule_shape, list) or len(schedule_shape) != 1:
            raise ValueError("v4 training schedule source binding is missing")
        expected_finite_psf_training_source = (
            _per_row_finite_psf_training_source(
                cache_manifest,
                capability,
                finite_psf_training_source.get("render_mode"),
                schedule_shape[0],
            )
        )
    if (
        cache_manifest["status"] != row_cache_v3.FROZEN_CACHE_STATUS
        or cache_manifest["row_count"] != cache["row_count"]
        or cache_manifest["generator_binding"]["receipt_sha256"]
        != cache["generator_binding_receipt_sha256"]
        or payload.get("row_sampling_policy")
        != _row_sampling_policy(
            cache_manifest,
            payload.get("runner_config", {}),
            payload.get("training_config", {}),
        )
        or payload.get("training_commit_contract")
        != _training_commit_contract(payload.get("runner_config", {}))
        or (
            capability is not None
            and payload.get("finite_psf_training_schedule_source")
            != expected_finite_psf_training_source
        )
    ):
        raise ValueError("training-run frozen cache binding differs")
    return manifest, cache_manifest


def _load_run_state(run_root, manifest):
    run_state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    payload = _manifest_payload(run_state)
    reports = payload.get("committed_reports", [])
    if (
        run_state.get("receipt_sha256") != _hash_json(payload)
        or payload.get("schema_version") != TRAINING_RUN_STATE_V3_SCHEMA
        or payload.get("run_id") != manifest["run_id"]
        or payload.get("run_manifest_receipt_sha256") != manifest["receipt_sha256"]
        or payload.get("attempt_count") != len(reports)
        or payload.get("applied_step_count") < 0
    ):
        raise ValueError("training-run state failed authentication")
    checkpoint = payload["latest_checkpoint"]
    checkpoint_path = _relative_child(run_root, checkpoint["relative_path"])
    if _file_sha256(checkpoint_path) != checkpoint["file_sha256"]:
        raise ValueError("latest training checkpoint hash differs")
    applied = 0
    loaded_reports = []
    for attempt_index, record in enumerate(reports):
        report_path = _relative_child(run_root, record["relative_path"])
        if _file_sha256(report_path) != record["file_sha256"]:
            raise ValueError("committed training-step report hash differs")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_payload = _manifest_payload(report)
        if (
            report.get("receipt_sha256") != _hash_json(report_payload)
            or report["receipt_sha256"] != record["report_receipt_sha256"]
            or report_payload.get("attempt_index") != attempt_index
            or report_payload.get("run_id") != manifest["run_id"]
            or report_payload.get("retrieval_scope") != TRAINING_CANDIDATE_BANK_SCOPE
            or report_payload.get("global_step_before") != applied
            or report_payload.get("global_step_after") not in (applied, applied + 1)
            or bool(
                report_payload["training_report"]["optimizer_step_applied"]
            )
            != (report_payload.get("global_step_after") == applied + 1)
            or report_payload["training_report"].get("retrieval_scope")
            != TRAINING_CANDIDATE_BANK_SCOPE
            or any(
                report_payload.get(name) != []
                for name in (
                    "prior_model_weight_dependencies",
                    "prior_feature_dependencies",
                    "prior_pseudolabel_dependencies",
                )
            )
            or any(
                receipt.get("inference_scope") is not False
                for receipt in report_payload.get(
                    "training_candidate_bank_receipts", []
                )
            )
        ):
            raise ValueError("committed training-step report failed authentication")
        loaded_reports.append(report)
        archive_checkpoint = report_payload.get("archive_checkpoint")
        if archive_checkpoint is not None:
            archive_path = _relative_child(
                run_root, archive_checkpoint["relative_path"]
            )
            if _file_sha256(archive_path) != archive_checkpoint["file_sha256"]:
                raise ValueError("immutable archive checkpoint hash differs")
        applied = int(report_payload["global_step_after"])
    if (
        applied != payload["applied_step_count"]
        or (
            reports
            and payload["latest_checkpoint"]
            != json.loads(
                _relative_child(run_root, reports[-1]["relative_path"]).read_text(
                    encoding="utf-8"
                )
            )["checkpoint"]
        )
    ):
        raise ValueError("training report ledger and applied step count differ")
    return run_state, checkpoint_path, loaded_reports


def load_training_run_v3(run_directory):
    run_root = _i_path(run_directory)
    manifest, cache_manifest = _load_manifest(run_root)
    catalogue = _load_catalogue_inputs(run_root, manifest["catalogue"])
    atlas_record = manifest["atlas"]
    atlas_path = _relative_child(run_root, atlas_record["relative_path"])
    if _file_sha256(atlas_path) != atlas_record["file_sha256"]:
        raise ValueError("persisted atlas input hash differs")
    atlas = np.load(atlas_path, allow_pickle=False)
    verify_atlas_binding_v3(atlas_record["binding"], atlas)
    run_state, checkpoint_path, training_reports = _load_run_state(
        run_root, manifest
    )
    state = staged_training.load_staged_training_checkpoint(
        checkpoint_path,
        device=manifest["execution_device"],
        expected_binding=manifest["staged_training_binding"],
        training_report_ledger=training_reports,
    )
    if int(state["global_step"]) != int(run_state["applied_step_count"]):
        raise ValueError("checkpoint global step differs from committed run state")
    return {
        "run_root": run_root,
        "manifest": manifest,
        "cache_manifest": cache_manifest,
        "catalogue": catalogue,
        "atlas_volume": atlas,
        "run_state": run_state,
        "training_reports": training_reports,
        "training_state": state,
    }


def make_training_run_export_receipt_v3(run_directory):
    """Export only after the compact checkpoint and full report ledger agree."""
    context = load_training_run_v3(run_directory)
    checkpoint = (
        context["run_root"]
        / context["run_state"]["latest_checkpoint"]["relative_path"]
    )
    return staged_training.make_staged_training_export_receipt_v3(
        checkpoint,
        training_report_ledger=context["training_reports"],
    )


def _row_indices(run_id, root_seed, step, row_count, batch_size):
    payload = {
        "domain": "anatomy-tracker.training-row-order/v3",
        "run_id": run_id,
        "root_seed": str(root_seed),
        "global_step": int(step),
    }
    seed = int(_hash_json(payload)[:16], 16)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    return rng.choice(row_count, size=batch_size, replace=False).tolist()


def _commit_fault_injection_point(point):
    """Test hook: production execution intentionally does nothing here."""


def _fsync_existing_file(path):
    with _i_path(path).open("r+b") as handle:
        os.fsync(handle.fileno())


def _next_resume_checkpoint_relative(old_run_state):
    slots = (
        "checkpoints/resume_slot_0.pt",
        "checkpoints/resume_slot_1.pt",
    )
    current = old_run_state["latest_checkpoint"]["relative_path"]
    if current not in slots:
        raise ValueError("committed resume checkpoint is not one of the frozen slots")
    return slots[1] if current == slots[0] else slots[0]


def _pending_attempt_payload(context, train_report, row_indices, batch, attempt_index):
    manifest = context["manifest"]
    state = context["training_state"]
    return {
        "schema_version": TRAINING_STEP_REPORT_V3_SCHEMA,
        "run_id": manifest["run_id"],
        "run_manifest_receipt_sha256": manifest["receipt_sha256"],
        "attempt_index": int(attempt_index),
        "global_step_before": int(train_report["step"]),
        "global_step_after": int(state["global_step"]),
        "row_cache_manifest_receipt_sha256": manifest["cache"][
            "manifest_receipt_sha256"
        ],
        "row_indices": list(row_indices),
        "row_identity": _plain(batch["row_identity"]),
        "retrieval_scope": TRAINING_CANDIDATE_BANK_SCOPE,
        "training_candidate_bank_receipts": _plain(
            batch["training_candidate_bank_receipts"]
        ),
        "training_report": _plain(train_report),
        "optimizer_learning_rates_after": [
            float(group["lr"]) for group in state["optimizer"].param_groups
        ],
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }


def _commit_attempt_batch(context, pending_payloads):
    if not pending_payloads:
        return []
    run_root = context["run_root"]
    manifest = context["manifest"]
    state = context["training_state"]
    old_run_state = context["run_state"]
    attempt_index = int(old_run_state["attempt_count"])
    if [payload["attempt_index"] for payload in pending_payloads] != list(
        range(attempt_index, attempt_index + len(pending_payloads))
    ):
        raise ValueError("pending training attempts are not one contiguous ledger suffix")
    checkpoint_relative = _next_resume_checkpoint_relative(old_run_state)
    checkpoint_path = run_root / checkpoint_relative
    staged_training.save_staged_training_checkpoint(state, checkpoint_path)
    _fsync_existing_file(checkpoint_path)
    checkpoint_sha = _file_sha256(run_root / checkpoint_relative)
    interval = int(
        manifest["runner_config"]["archive_checkpoint_interval_applied_steps"]
    )
    target = int(manifest["runner_config"]["target_applied_steps"])
    last_payload = pending_payloads[-1]
    applied = bool(last_payload["training_report"]["optimizer_step_applied"])
    archive_checkpoint = None
    if applied and (int(state["global_step"]) % interval == 0 or int(state["global_step"]) == target):
        archive_relative = f"checkpoints/archive_step_{state['global_step']:08d}.pt"
        archive_path = run_root / archive_relative
        staged_training.save_staged_training_checkpoint(
            state, archive_path
        )
        _fsync_existing_file(archive_path)
        archive_checkpoint = {
            "relative_path": archive_relative,
            "file_sha256": _file_sha256(archive_path),
        }
    checkpoint_record = {
        "relative_path": checkpoint_relative,
        "file_sha256": checkpoint_sha,
    }
    _commit_fault_injection_point("after_checkpoint")
    reports = []
    report_records = []
    for offset, pending_payload in enumerate(pending_payloads):
        report = _with_receipt(
            {
                **pending_payload,
                "checkpoint": checkpoint_record,
                "archive_checkpoint": (
                    archive_checkpoint
                    if offset + 1 == len(pending_payloads)
                    else None
                ),
            }
        )
        report_relative = (
            f"reports/attempt_{pending_payload['attempt_index']:08d}.json"
        )
        report_path = run_root / report_relative
        _atomic_json(report_path, report)
        reports.append(report)
        report_records.append(
            {
                "relative_path": report_relative,
                "file_sha256": _file_sha256(report_path),
                "report_receipt_sha256": report["receipt_sha256"],
            }
        )
        _commit_fault_injection_point("after_report")
    _commit_fault_injection_point("after_reports")
    run_state_payload = {
        "schema_version": TRAINING_RUN_STATE_V3_SCHEMA,
        "run_id": manifest["run_id"],
        "run_manifest_receipt_sha256": manifest["receipt_sha256"],
        "attempt_count": attempt_index + len(pending_payloads),
        "applied_step_count": int(state["global_step"]),
        "latest_checkpoint": checkpoint_record,
        "committed_reports": [
            *old_run_state["committed_reports"],
            *report_records,
        ],
    }
    run_state = _with_receipt(run_state_payload)
    _atomic_json(run_root / "run_state.json", run_state)
    _commit_fault_injection_point("after_run_state")
    context["run_state"] = run_state
    context["training_reports"].extend(reports)
    return reports


def _run_training_attempts_locked_v3(run_directory, max_attempts):
    context = load_training_run_v3(run_directory)
    manifest = context["manifest"]
    config = manifest["runner_config"]
    target = int(config["target_applied_steps"])
    atlas = torch.as_tensor(
        context["atlas_volume"],
        device=manifest["execution_device"],
        dtype=torch.float32,
    )
    reports = []
    pending_payloads = []
    commit_interval = int(config["checkpoint_commit_interval_attempts"])
    for _ in range(max_attempts):
        step = int(context["training_state"]["global_step"])
        if step >= target:
            break
        preparation_rng_state = staged_training._rng_state()
        try:
            indices = _row_indices(
                manifest["run_id"],
                config["row_selection_seed"],
                step,
                context["cache_manifest"]["row_count"],
                config["batch_size"],
            )
            rows = row_cache_v3.load_training_rows_v3(
                manifest["cache"]["directory"],
                indices,
                expected_manifest_receipt_sha256=manifest["cache"][
                    "manifest_receipt_sha256"
                ],
            )
            geometry = context["catalogue"]["support_geometry"]
            full_batch = staged_training.model_ready_rows_v3(
                rows,
                context["catalogue"],
                atlas,
                origin_ap_dv_ml_um=geometry["origin_ap_dv_ml_um"],
                voxel_size_ap_dv_ml_um=geometry["voxel_size_ap_dv_ml_um"],
                support_origin_ap_dv_ml_um=geometry[
                    "support_origin_ap_dv_ml_um"
                ],
                axial_offsets_um=config["axial_offsets_um"],
                axial_weights=config["axial_weights"],
                device=manifest["execution_device"],
                data_role=DEVELOPMENT_DATA_ROLE,
                finite_psf_capability=manifest.get("finite_psf_capability"),
            )
            bank_root_seed = _hash_json(
                {
                    "root_seed": str(config["candidate_bank_root_seed"]),
                    "run_id": manifest["run_id"],
                    "global_step": step,
                }
            )
            batch = make_training_candidate_batch_v3(
                full_batch,
                context["catalogue"],
                bank_size=int(config["candidate_bank_size"]),
                root_seed=bank_root_seed,
            )
            if batch["catalogue_scope"] != TRAINING_CANDIDATE_BANK_SCOPE:
                raise RuntimeError(
                    "production trainer must never reinterpret sampled banks as a posterior"
                )
        except BaseException:
            staged_training._restore_rng_state(preparation_rng_state)
            reports.extend(_commit_attempt_batch(context, pending_payloads))
            pending_payloads = []
            raise
        train_report = staged_training.train_staged_step(
            context["training_state"], batch
        )
        attempt_index = int(context["run_state"]["attempt_count"]) + len(
            pending_payloads
        )
        pending_payloads.append(
            _pending_attempt_payload(
                context,
                train_report,
                indices,
                batch,
                attempt_index,
            )
        )
        applied = bool(train_report["optimizer_step_applied"])
        global_step = int(context["training_state"]["global_step"])
        archive_interval = int(
            config["archive_checkpoint_interval_applied_steps"]
        )
        archive_due = applied and (
            global_step % archive_interval == 0 or global_step == target
        )
        if (
            len(pending_payloads) >= commit_interval
            or archive_due
            or global_step >= target
        ):
            reports.extend(_commit_attempt_batch(context, pending_payloads))
            pending_payloads = []
    reports.extend(_commit_attempt_batch(context, pending_payloads))
    return reports


def run_training_attempts_v3(run_directory, *, max_attempts):
    """Run bounded atomic attempts under a process-exclusive run lock."""
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 0:
        raise ValueError("max_attempts must be a nonnegative integer")
    with _exclusive_run_lock(_i_path(run_directory)):
        return _run_training_attempts_locked_v3(run_directory, max_attempts)


def run_training_until_target_v3(run_directory):
    """Resume a prepared run until its frozen applied-step target is reached."""
    while True:
        context = load_training_run_v3(run_directory)
        remaining = (
            int(context["manifest"]["runner_config"]["target_applied_steps"])
            - int(context["training_state"]["global_step"])
        )
        if remaining <= 0:
            return context["run_state"]
        reports = run_training_attempts_v3(run_directory, max_attempts=remaining)
        if reports and any(report["training_report"]["optimizer_step_applied"] for report in reports):
            continue
        raise RuntimeError("all bounded attempts overflowed without an optimizer step")
