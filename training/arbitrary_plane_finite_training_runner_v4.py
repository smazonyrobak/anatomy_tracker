"""Authenticated, resumable finite-thickness v4 training-run ledger.

The v3 runner is used only for generic catalogue, atlas, path, and atomic-file
primitives.  This module owns distinct v4 run/state/report schemas and consumes
only the strict finite v4 row-cache API.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_row_cache_v4 as row_cache_v4
import training.arbitrary_plane_staged_training as staged_training
import training.arbitrary_plane_training_runner_v3 as runner_primitives
from training.arbitrary_plane_training_bank_v3 import (
    TRAINING_CANDIDATE_BANK_SCOPE,
    make_training_candidate_batch_v3,
    verify_training_candidate_bank_receipt_v3,
)


FINITE_TRAINING_RUN_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-finite-training-run/v4"
)
FINITE_TRAINING_RUN_STATE_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-finite-training-run-state/v4"
)
FINITE_TRAINING_STEP_REPORT_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-finite-training-step-report/v4"
)
FINITE_TRAINING_COMMIT_CONTRACT_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-finite-training-commit-contract/v4"
)
FINITE_ROW_SAMPLING_POLICY_V4_SCHEMA = (
    "anatomy-tracker.finite-cache-row-sampling-policy/v4"
)
FINITE_PSF_TRAINING_SOURCE_V4_SCHEMA = (
    "anatomy-tracker.finite-psf-training-schedule-source/v4"
)
FINITE_TRAINING_RUN_EXPORT_RECEIPT_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-finite-training-run-export-receipt/v4"
)
FINITE_STAGED_TRAINING_EXPORT_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-finite-staged-training-export/v4"
)
FINITE_TRAINING_REPORT_LEDGER_EVIDENCE_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-finite-training-report-ledger-evidence/v4"
)
UNCALIBRATED_STATUS = "absent-uncalibrated"
DEVELOPMENT_DATA_ROLE = staged_training.DEVELOPMENT_DATA_ROLE
DEFAULT_CHECKPOINT_COMMIT_INTERVAL_ATTEMPTS = 25
_RESUME_CHECKPOINT_RELATIVE_PATHS = {
    "checkpoints/resume_slot_0.pt",
    "checkpoints/resume_slot_1.pt",
}
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (
    "training/arbitrary_plane_finite_training_runner_v4.py",
    "training/arbitrary_plane_row_cache_v4.py",
    "training/arbitrary_plane_psf_v4.py",
    "training/arbitrary_plane_training_runner_v3.py",
    "training/arbitrary_plane_staged_training.py",
    "training/arbitrary_plane_batch_v3.py",
    "training/arbitrary_plane_training_bank_v3.py",
    "training/arbitrary_plane_coarse_proposal_v5.py",
    "training/arbitrary_plane_joint_model.py",
)
_RUN_MANIFEST_KEYS = {
    "schema_version",
    "run_id",
    "data_role",
    "retrieval_scope",
    "cache",
    "catalogue",
    "atlas",
    "model_kwargs",
    "training_config",
    "runner_config",
    "row_sampling_policy",
    "training_commit_contract",
    "seed_record",
    "execution_device",
    "staged_training_binding",
    "runner_source_sha256",
    "finite_psf_capability",
    "finite_psf_training_schedule_source",
    "forbidden_sources",
    "prior_model_weight_dependencies",
    "prior_feature_dependencies",
    "prior_pseudolabel_dependencies",
    "receipt_sha256",
}
_RUN_STATE_KEYS = {
    "schema_version",
    "run_id",
    "run_manifest_receipt_sha256",
    "attempt_count",
    "applied_step_count",
    "latest_checkpoint",
    "committed_reports",
    "receipt_sha256",
}


def _plain(value):
    return runner_primitives._plain(value)


def _hash_json(value):
    return runner_primitives._hash_json(value)


def _file_sha256(path):
    return runner_primitives._file_sha256(path)


def _i_path(path):
    return runner_primitives._i_path(path)


def _source_receipts():
    return {
        name: hashlib.sha256((_SOURCE_ROOT / name).read_bytes()).hexdigest()
        for name in _SOURCE_FILES
    }


def _with_receipt(payload):
    payload = _plain(payload)
    return {**payload, "receipt_sha256": _hash_json(payload)}


def _payload(receipted):
    return {key: value for key, value in receipted.items() if key != "receipt_sha256"}


def _finite_psf_training_source(cache_manifest):
    contract = cache_manifest["finite_psf_run_contract"]
    row_cache_v4.verify_finite_psf_cache_run_contract_v4(contract)
    payload = {
        "schema_version": FINITE_PSF_TRAINING_SOURCE_V4_SCHEMA,
        "schedule_source": "authenticated-per-row",
        "training_row_schema_version": psf_v4.TRAINING_ROW_V4_SCHEMA,
        "training_row_contract_field": "finite_psf_contract",
        "finite_psf_capability_receipt_sha256": cache_manifest[
            "finite_psf_capability"
        ]["receipt_sha256"],
        "finite_psf_cache_run_contract": _plain(contract),
        "frozen_cache_manifest_receipt_sha256": cache_manifest[
            "receipt_sha256"
        ],
        "ordered_row_schedule_binding": _plain(cache_manifest["freeze_audit"]),
        "global_schedule_fallback": None,
        "unknown_thickness_policy": "reject",
    }
    return _with_receipt(payload)


def _row_sampling_policy(cache_manifest, runner_config, training_config):
    payload = {
        "schema_version": FINITE_ROW_SAMPLING_POLICY_V4_SCHEMA,
        "algorithm": "seeded-PCG64DXSM-uniform-without-replacement/v4",
        "population": "every row in the exact frozen finite v4 cache manifest",
        "frozen_cache_manifest_receipt_sha256": cache_manifest[
            "receipt_sha256"
        ],
        "frozen_cache_row_count": int(cache_manifest["row_count"]),
        "batch_size": int(runner_config["batch_size"]),
        "row_selection_seed": str(runner_config["row_selection_seed"]),
        "seed_derivation_domain": "anatomy-tracker.finite-training-row-order/v4",
        "phase_policy": "identical uniform full-cache mixture in pose-warmup and joint phases",
        "marginal_or_empty_row_policy": (
            "retained in the sampling population at zero point-pose and/or dense "
            "loss exactly as authenticated by each row; never redraw, drop, or filter"
        ),
        "pose_warmup_steps": int(training_config["pose_warmup_steps"]),
        "phase_aware_resampling": False,
        "finite_psf_cache_run_contract_receipt_sha256": cache_manifest[
            "finite_psf_run_contract"
        ]["receipt_sha256"],
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    return _with_receipt(payload)


def _training_commit_contract(runner_config):
    payload = {
        "schema_version": FINITE_TRAINING_COMMIT_CONTRACT_V4_SCHEMA,
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
            "load the last authenticated v4 run state and deterministically replay "
            "only attempts not published by that state"
        ),
        "per_attempt_reports_preserved": True,
    }
    return _with_receipt(payload)


def _validate_runner_config(
    config,
    *,
    catalogue_cell_count,
    cache_row_count,
    training_config,
):
    validated = runner_primitives._validate_runner_config(
        config,
        catalogue_cell_count=catalogue_cell_count,
        cache_row_count=cache_row_count,
        training_top_k=training_config.get("top_k"),
        pose_warmup_steps=training_config.get("pose_warmup_steps"),
        refinement_steps=training_config.get("refinement_steps"),
        joint_pose_only_steps=training_config.get("joint_pose_only_steps"),
        per_row_finite_psf=True,
    )
    if validated["axial_offsets_um"] != [] or validated["axial_weights"] != []:
        raise ValueError("finite v4 runner forbids a global PSF schedule")
    return validated


def _cache_record(cache_root, cache_manifest):
    return {
        "directory": str(cache_root),
        "schema_version": row_cache_v4.ROW_CACHE_V4_SCHEMA,
        "manifest_receipt_sha256": cache_manifest["receipt_sha256"],
        "generator_binding_receipt_sha256": cache_manifest[
            "generator_binding"
        ]["receipt_sha256"],
        "row_count": int(cache_manifest["row_count"]),
        "finite_psf_cache_run_contract_receipt_sha256": cache_manifest[
            "finite_psf_run_contract"
        ]["receipt_sha256"],
        "freeze_audit": _plain(cache_manifest["freeze_audit"]),
    }


def _scientific_run_id_payload(core):
    payload = dict(core)
    payload.pop("training_commit_contract", None)
    runner_config = dict(payload["runner_config"])
    runner_config.pop("checkpoint_commit_interval_attempts", None)
    payload["runner_config"] = runner_config
    return payload


def _checkpoint_context(manifest):
    return {
        "run_manifest_receipt_sha256": manifest["receipt_sha256"],
        "finite_psf_capability_receipt_sha256": manifest[
            "finite_psf_capability"
        ]["receipt_sha256"],
        "finite_psf_cache_run_contract_receipt_sha256": manifest[
            "finite_psf_training_schedule_source"
        ]["finite_psf_cache_run_contract"]["receipt_sha256"],
    }


def _checkpoint_record(manifest, relative_path, file_sha256):
    return {
        "relative_path": str(relative_path),
        "file_sha256": str(file_sha256),
        **_checkpoint_context(manifest),
    }


def initialize_finite_training_run_v4(
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
):
    """Freeze a strict finite cache and create a fresh random model state."""
    run_root = _i_path(run_directory)
    cache_root = _i_path(cache_directory)
    model_kwargs = runner_primitives._complete_model_kwargs(model_kwargs)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError("finite v4 training-run directory must be empty")
    cache_manifest = row_cache_v4.load_training_row_cache_manifest_v4(
        cache_root, expected_generator_binding=expected_generator_binding
    )
    if (
        cache_manifest["status"] != row_cache_v4.FROZEN_CACHE_STATUS
        or cache_manifest["row_count"] < 1
    ):
        raise ValueError("finite training requires a nonempty frozen v4 row cache")
    cache_audit = row_cache_v4.audit_training_row_cache_v4(cache_root)
    if (
        not cache_audit["all_rows_authenticated"]
        or cache_audit["temporary_file_count"] != 0
        or cache_audit["row_count"] != cache_manifest["row_count"]
    ):
        raise ValueError("finite v4 row cache failed its full pre-run audit")
    capability = cache_manifest["finite_psf_capability"]
    psf_v4.verify_finite_psf_model_capability_v4(capability)
    runner_primitives._verify_catalogue(catalogue)
    runner_config = _validate_runner_config(
        runner_config,
        catalogue_cell_count=int(catalogue["counts"]["cell_count"]),
        cache_row_count=int(cache_manifest["row_count"]),
        training_config=training_config,
    )
    atlas = np.ascontiguousarray(
        runner_primitives._tensor_to_numpy(atlas_volume), dtype=np.float32
    )
    atlas_binding = runner_primitives.make_atlas_binding_v3(
        atlas,
        source_assets=atlas_source_assets,
        preprocessing=atlas_preprocessing,
    )
    geometry = catalogue["support_geometry"]
    if tuple(atlas.shape[-3:]) != tuple(geometry["support_mask_receipt"]["shape"]):
        raise ValueError("atlas volume shape differs from catalogue support")
    run_root.mkdir(parents=True, exist_ok=True)
    for name in ("inputs", "checkpoints", "reports"):
        (run_root / name).mkdir(exist_ok=True)
    catalogue_record = runner_primitives._save_catalogue_inputs(
        run_root, catalogue
    )
    atlas_relative = "inputs/atlas_volume_float32.npy"
    runner_primitives._atomic_npy(run_root / atlas_relative, atlas)
    state = staged_training.initialize_staged_training(
        model_kwargs,
        training_config,
        catalogue_id=catalogue["catalogue_id"],
        catalogue_receipt_sha256=catalogue["receipt_sha256"],
        catalogue_cell_count=int(catalogue["counts"]["cell_count"]),
        generator_ids=cache_manifest["generator_binding"]["generator_ids"],
        device=device,
        finite_psf_capability=capability,
    )
    core = {
        "schema_version": FINITE_TRAINING_RUN_V4_SCHEMA,
        "data_role": DEVELOPMENT_DATA_ROLE,
        "retrieval_scope": TRAINING_CANDIDATE_BANK_SCOPE,
        "cache": _cache_record(cache_root, cache_manifest),
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
            "candidate_bank_root_seed": runner_config[
                "candidate_bank_root_seed"
            ],
            "cache_generation_seeds": _plain(cache_manifest["seed_record"]),
        },
        "execution_device": str(torch.device(device)),
        "staged_training_binding": state["binding"],
        "runner_source_sha256": _source_receipts(),
        "finite_psf_capability": _plain(capability),
        "finite_psf_training_schedule_source": _finite_psf_training_source(
            cache_manifest
        ),
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
    core["run_id"] = _hash_json(
        {
            "domain": FINITE_TRAINING_RUN_V4_SCHEMA,
            "scientific_core": _scientific_run_id_payload(core),
        }
    )
    manifest = _with_receipt(core)
    if set(manifest) != _RUN_MANIFEST_KEYS:
        raise RuntimeError("finite v4 run manifest implementation is incomplete")
    runner_primitives._atomic_json(run_root / "run_manifest.json", manifest)
    checkpoint_relative = "checkpoints/resume_slot_0.pt"
    staged_training.save_staged_training_checkpoint(
        state, run_root / checkpoint_relative
    )
    checkpoint = _checkpoint_record(
        manifest,
        checkpoint_relative,
        _file_sha256(run_root / checkpoint_relative),
    )
    run_state = _with_receipt(
        {
            "schema_version": FINITE_TRAINING_RUN_STATE_V4_SCHEMA,
            "run_id": manifest["run_id"],
            "run_manifest_receipt_sha256": manifest["receipt_sha256"],
            "attempt_count": 0,
            "applied_step_count": 0,
            "latest_checkpoint": checkpoint,
            "committed_reports": [],
        }
    )
    runner_primitives._atomic_json(run_root / "run_state.json", run_state)
    return manifest, run_state


def _load_manifest(run_root):
    manifest = json.loads(
        (run_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    payload = _payload(manifest) if isinstance(manifest, dict) else {}
    identity_core = dict(payload)
    identity_core.pop("run_id", None)
    if (
        set(manifest) != _RUN_MANIFEST_KEYS
        or manifest.get("receipt_sha256") != _hash_json(payload)
        or payload.get("schema_version") != FINITE_TRAINING_RUN_V4_SCHEMA
        or payload.get("run_id")
        != _hash_json(
            {
                "domain": FINITE_TRAINING_RUN_V4_SCHEMA,
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
        raise ValueError("finite v4 training-run manifest failed authentication")
    capability = payload["finite_psf_capability"]
    psf_v4.verify_finite_psf_model_capability_v4(capability)
    cache_record = payload["cache"]
    cache_manifest = row_cache_v4.load_training_row_cache_manifest_v4(
        cache_record["directory"],
        expected_receipt_sha256=cache_record["manifest_receipt_sha256"],
    )
    _validate_runner_config(
        payload["runner_config"],
        catalogue_cell_count=int(payload["catalogue"]["catalogue_cell_count"]),
        cache_row_count=int(cache_record["row_count"]),
        training_config=payload["training_config"],
    )
    if (
        cache_manifest["status"] != row_cache_v4.FROZEN_CACHE_STATUS
        or cache_record != _cache_record(Path(cache_record["directory"]), cache_manifest)
        or cache_manifest["finite_psf_capability"] != capability
        or payload.get("finite_psf_training_schedule_source")
        != _finite_psf_training_source(cache_manifest)
        or payload.get("row_sampling_policy")
        != _row_sampling_policy(
            cache_manifest,
            payload["runner_config"],
            payload["training_config"],
        )
        or payload.get("training_commit_contract")
        != _training_commit_contract(payload["runner_config"])
        or payload.get("staged_training_binding", {}).get(
            "finite_psf_capability"
        )
        != capability
        or payload.get("staged_training_binding", {}).get("generator_ids")
        != sorted(cache_manifest["generator_binding"]["generator_ids"])
    ):
        raise ValueError("finite v4 run/cache/capability binding differs")
    return manifest, cache_manifest


def _verify_checkpoint_record(
    run_root,
    manifest,
    checkpoint,
    *,
    immutable_file=True,
    resume_slot=False,
):
    context = _checkpoint_context(manifest)
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"relative_path", "file_sha256", *context}
        or any(checkpoint.get(key) != value for key, value in context.items())
        or not isinstance(checkpoint.get("relative_path"), str)
        or not isinstance(checkpoint.get("file_sha256"), str)
        or len(checkpoint["file_sha256"]) != 64
        or bool(set(checkpoint["file_sha256"].lower()) - set("0123456789abcdef"))
        or (
            resume_slot
            and checkpoint["relative_path"] not in _RESUME_CHECKPOINT_RELATIVE_PATHS
        )
    ):
        raise ValueError("finite v4 checkpoint context differs from its run")
    checkpoint_path = runner_primitives._relative_child(
        run_root, checkpoint["relative_path"]
    )
    if immutable_file and _file_sha256(checkpoint_path) != checkpoint["file_sha256"]:
        raise ValueError("latest finite training checkpoint hash differs")
    return checkpoint_path


def _load_run_state(run_root, manifest):
    run_state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    payload = _payload(run_state) if isinstance(run_state, dict) else {}
    reports = payload.get("committed_reports", [])
    if (
        set(run_state) != _RUN_STATE_KEYS
        or run_state.get("receipt_sha256") != _hash_json(payload)
        or payload.get("schema_version") != FINITE_TRAINING_RUN_STATE_V4_SCHEMA
        or payload.get("run_id") != manifest["run_id"]
        or payload.get("run_manifest_receipt_sha256")
        != manifest["receipt_sha256"]
        or payload.get("attempt_count") != len(reports)
        or not isinstance(payload.get("applied_step_count"), int)
        or payload["applied_step_count"] < 0
    ):
        raise ValueError("finite v4 training-run state failed authentication")
    checkpoint_path = _verify_checkpoint_record(
        run_root,
        manifest,
        payload["latest_checkpoint"],
        resume_slot=True,
    )
    applied = 0
    loaded_reports = []
    for attempt_index, record in enumerate(reports):
        report_path = runner_primitives._relative_child(
            run_root, record["relative_path"]
        )
        if _file_sha256(report_path) != record["file_sha256"]:
            raise ValueError("committed finite training-step report hash differs")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_payload = _payload(report)
        row_identity = report_payload.get("row_identity", [])
        expected_row_indices = _row_indices(
            manifest["run_id"],
            manifest["runner_config"]["row_selection_seed"],
            applied,
            manifest["cache"]["row_count"],
            manifest["runner_config"]["batch_size"],
        )
        if (
            report.get("receipt_sha256") != _hash_json(report_payload)
            or report["receipt_sha256"] != record["report_receipt_sha256"]
            or report_payload.get("schema_version")
            != FINITE_TRAINING_STEP_REPORT_V4_SCHEMA
            or report_payload.get("attempt_index") != attempt_index
            or report_payload.get("run_id") != manifest["run_id"]
            or report_payload.get("run_manifest_receipt_sha256")
            != manifest["receipt_sha256"]
            or report_payload.get("retrieval_scope")
            != TRAINING_CANDIDATE_BANK_SCOPE
            or report_payload.get("global_step_before") != applied
            or report_payload.get("global_step_after") not in (applied, applied + 1)
            or report_payload.get("row_cache_manifest_receipt_sha256")
            != manifest["cache"]["manifest_receipt_sha256"]
            or report_payload.get("row_indices") != expected_row_indices
            or not isinstance(row_identity, list)
            or len(row_identity) != len(expected_row_indices)
            or any(not isinstance(identity.get("finite_psf"), dict) for identity in row_identity)
            or report_payload.get("ordered_row_finite_psf_identity_sha256")
            != _hash_json([identity["finite_psf"] for identity in row_identity])
            or bool(report_payload["training_report"]["optimizer_step_applied"])
            != (report_payload["global_step_after"] == applied + 1)
            or report_payload.get("finite_psf_cache_run_contract_receipt_sha256")
            != manifest["cache"][
                "finite_psf_cache_run_contract_receipt_sha256"
            ]
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
            raise ValueError("finite v4 training-step report failed authentication")
        # Resume slots are intentionally reused.  Historical reports authenticate
        # the checkpoint hash that was current when they were committed, but only
        # the latest slot can still be required to contain those bytes.
        _verify_checkpoint_record(
            run_root,
            manifest,
            report_payload["checkpoint"],
            immutable_file=False,
            resume_slot=True,
        )
        archive = report_payload.get("archive_checkpoint")
        if archive is not None:
            _verify_checkpoint_record(run_root, manifest, archive)
        loaded_reports.append(report)
        applied = int(report_payload["global_step_after"])
    if (
        applied != payload["applied_step_count"]
        or (
            reports
            and payload["latest_checkpoint"]
            != json.loads(
                runner_primitives._relative_child(
                    run_root, reports[-1]["relative_path"]
                ).read_text(encoding="utf-8")
            )["checkpoint"]
        )
    ):
        raise ValueError("finite training report ledger and step count differ")
    return run_state, checkpoint_path, loaded_reports


def _verify_training_report_ledger_correspondence_v4(
    checkpoint,
    reports,
    *,
    expected_run_id,
    expected_run_manifest_receipt_sha256,
):
    staged_training.verify_staged_training_checkpoint_payload_v3(
        checkpoint, replay_initialization=False
    )
    binding = checkpoint["binding"]
    compact_ledger = checkpoint["training_step_ledger"]
    identity_by_row_id = {
        identity["training_row_id"]: identity
        for identity in checkpoint["row_identity_records"]
    }
    applied = 0
    report_receipts = []
    report_chain = _hash_json(
        {
            "domain": f"{FINITE_TRAINING_REPORT_LEDGER_EVIDENCE_V4_SCHEMA}/genesis",
            "run_id": expected_run_id,
            "run_manifest_receipt_sha256": (
                expected_run_manifest_receipt_sha256
            ),
        }
    )
    for attempt_index, report in enumerate(reports):
        report_payload = _payload(report) if isinstance(report, dict) else {}
        report_receipt = report.get("receipt_sha256") if isinstance(report, dict) else None
        row_identity = report_payload.get("row_identity")
        receipts = report_payload.get("training_candidate_bank_receipts")
        scope = report_payload.get("retrieval_scope")
        before = report_payload.get("global_step_before")
        after = report_payload.get("global_step_after")
        train_report = report_payload.get("training_report", {})
        was_applied = bool(train_report.get("optimizer_step_applied"))
        if (
            report_receipt != _hash_json(report_payload)
            or report_payload.get("schema_version")
            != FINITE_TRAINING_STEP_REPORT_V4_SCHEMA
            or report_payload.get("attempt_index") != attempt_index
            or report_payload.get("run_id") != expected_run_id
            or report_payload.get("run_manifest_receipt_sha256")
            != expected_run_manifest_receipt_sha256
            or before != applied
            or after not in (applied, applied + 1)
            or was_applied != (after == applied + 1)
            or train_report.get("retrieval_scope") != scope
            or not isinstance(row_identity, list)
            or not row_identity
            or not isinstance(receipts, list)
            or scope != TRAINING_CANDIDATE_BANK_SCOPE
            or len(receipts) != len(row_identity)
        ):
            raise ValueError("finite training report ledger correspondence failed")
        for identity, receipt in zip(row_identity, receipts):
            verify_training_candidate_bank_receipt_v3(
                receipt,
                expected_catalogue_id=binding["catalogue_id"],
                expected_catalogue_receipt_sha256=binding[
                    "catalogue_receipt_sha256"
                ],
                expected_training_row_id=identity["training_row_id"],
                expected_training_row_receipt_sha256=identity[
                    "training_row_receipt_sha256"
                ],
                expected_training_row_identity_sha256=_hash_json(identity),
            )
        if was_applied:
            ledger_payload = {
                "step": applied,
                "catalogue_scope": scope,
                "training_row_ids": [
                    identity["training_row_id"] for identity in row_identity
                ],
                "training_row_receipt_sha256": [
                    identity["training_row_receipt_sha256"]
                    for identity in row_identity
                ],
                "training_row_identity_sha256": [
                    _hash_json(identity) for identity in row_identity
                ],
                "training_candidate_bank_receipt_sha256": [
                    receipt["receipt_sha256"] for receipt in receipts
                ],
            }
            if (
                applied >= len(compact_ledger)
                or any(
                    compact_ledger[applied].get(key) != value
                    for key, value in ledger_payload.items()
                )
                or any(
                    identity_by_row_id.get(identity["training_row_id"])
                    != identity
                    for identity in row_identity
                )
            ):
                raise ValueError("finite checkpoint and report ledgers differ")
            applied += 1
        report_receipts.append(report_receipt)
        report_chain = _hash_json(
            {
                "domain": (
                    f"{FINITE_TRAINING_REPORT_LEDGER_EVIDENCE_V4_SCHEMA}/ordered-report"
                ),
                "previous_chain_sha256": report_chain,
                "attempt_index": attempt_index,
                "report_receipt_sha256": report_receipt,
            }
        )
    if applied != int(checkpoint["global_step"]):
        raise ValueError("finite checkpoint/report applied-step counts differ")
    payload = {
        "schema_version": FINITE_TRAINING_REPORT_LEDGER_EVIDENCE_V4_SCHEMA,
        "run_id": expected_run_id,
        "run_manifest_receipt_sha256": expected_run_manifest_receipt_sha256,
        "report_count": len(reports),
        "applied_step_count": applied,
        "ordered_report_receipts_sha256": _hash_json(report_receipts),
        "final_report_chain_sha256": report_chain,
    }
    return _with_receipt(payload)


def _load_staged_training_checkpoint_v4(
    checkpoint_path,
    *,
    device,
    expected_binding,
    reports,
    run_id,
    run_manifest_receipt_sha256,
):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    staged_training.verify_staged_training_checkpoint_payload_v3(
        checkpoint,
        expected_binding=expected_binding,
        replay_initialization=False,
    )
    _verify_training_report_ledger_correspondence_v4(
        checkpoint,
        reports,
        expected_run_id=run_id,
        expected_run_manifest_receipt_sha256=run_manifest_receipt_sha256,
    )
    binding = checkpoint["binding"]
    state = staged_training.initialize_staged_training(
        binding["model_kwargs"],
        binding["training_config"],
        catalogue_id=binding["catalogue_id"],
        catalogue_receipt_sha256=binding["catalogue_receipt_sha256"],
        catalogue_cell_count=binding["catalogue_cell_count"],
        generator_ids=binding["generator_ids"],
        device=device,
        finite_psf_capability=binding["finite_psf_capability"],
    )
    if state["initialization_receipt"] != checkpoint["initialization_receipt"]:
        raise ValueError("finite fresh initialization receipt does not replay")
    state["model"].load_state_dict(checkpoint["model_state"])
    state["optimizer"].load_state_dict(checkpoint["optimizer_state"])
    state["scaler"].load_state_dict(checkpoint["scaler_state"])
    state["global_step"] = int(checkpoint["global_step"])
    state["row_identity_records"] = list(checkpoint["row_identity_records"])
    state["training_step_ledger"] = list(checkpoint["training_step_ledger"])
    staged_training._restore_rng_state(checkpoint["rng_state"])
    return state, checkpoint


def _make_finite_staged_training_export_receipt_v4(
    checkpoint_path,
    checkpoint,
    reports,
    *,
    run_id,
    run_manifest_receipt_sha256,
):
    staged_training.verify_staged_training_checkpoint_payload_v3(checkpoint)
    records = checkpoint["row_identity_records"]
    if int(checkpoint["global_step"]) < 1 or not records:
        raise ValueError("finite inference export requires applied training")
    evidence = _verify_training_report_ledger_correspondence_v4(
        checkpoint,
        reports,
        expected_run_id=run_id,
        expected_run_manifest_receipt_sha256=run_manifest_receipt_sha256,
    )
    model_receipts = staged_training._model_state_receipts(
        checkpoint["model_state"]
    )
    payload = {
        "schema_version": FINITE_STAGED_TRAINING_EXPORT_V4_SCHEMA,
        "staged_training_schema_version": staged_training.STAGED_TRAINING_SCHEMA,
        "staged_checkpoint_path": str(checkpoint_path),
        "staged_checkpoint_file_sha256": _file_sha256(checkpoint_path),
        "binding": _plain(checkpoint["binding"]),
        "initialization_receipt": _plain(checkpoint["initialization_receipt"]),
        "global_step": int(checkpoint["global_step"]),
        "model_state_receipts": model_receipts,
        "model_state_sha256": staged_training._model_state_receipt_sha256(
            model_receipts
        ),
        "row_identity_record_count": len(records),
        "row_identity_records_sha256": _hash_json(records),
        "candidate_bank_receipt_storage": "immutable-finite-v4-run-reports-only",
        "sampled_bank_step_count": len(checkpoint["training_step_ledger"]),
        "training_step_ledger_count": len(checkpoint["training_step_ledger"]),
        "training_step_ledger_sha256": _hash_json(
            checkpoint["training_step_ledger"]
        ),
        "training_step_ledger_summary": _plain(
            checkpoint["training_step_ledger_summary"]
        ),
        "training_report_ledger_evidence": evidence,
        "training_row_ids": _plain(checkpoint["seen_training_row_ids"]),
        "training_animal_ids": _plain(checkpoint["seen_animal_ids"]),
        "training_specimen_ids": _plain(checkpoint["seen_specimen_ids"]),
        "training_experiment_ids": _plain(checkpoint["seen_experiment_ids"]),
        "learned_dependency_arrays": _plain(
            checkpoint["learned_dependency_arrays"]
        ),
    }
    return _with_receipt(payload)


def verify_finite_staged_training_export_receipt_v4(
    receipt,
    *,
    model_kwargs,
    catalogue_id,
    catalogue_receipt_sha256,
    catalogue_cell_count,
    model_state_sha256,
    require_source_file=False,
):
    payload = _payload(receipt) if isinstance(receipt, dict) else {}
    binding = payload.get("binding", {})
    evidence = payload.get("training_report_ledger_evidence", {})
    ledger_summary = payload.get("training_step_ledger_summary", {})
    valid = (
        isinstance(receipt, dict)
        and receipt.get("receipt_sha256") == _hash_json(payload)
        and payload.get("schema_version")
        == FINITE_STAGED_TRAINING_EXPORT_V4_SCHEMA
        and payload.get("staged_training_schema_version")
        == staged_training.STAGED_TRAINING_SCHEMA
        and isinstance(binding, dict)
        and binding.get("schema_version") == staged_training.STAGED_TRAINING_SCHEMA
        and binding.get("source_sha256") == staged_training._source_receipts()
        and binding.get("model_kwargs") == _plain(model_kwargs)
        and binding.get("catalogue_id") == str(catalogue_id)
        and binding.get("catalogue_receipt_sha256")
        == str(catalogue_receipt_sha256)
        and binding.get("catalogue_cell_count") == int(catalogue_cell_count)
        and binding.get("prior_model_weight_dependencies") == []
        and binding.get("prior_feature_dependencies") == []
        and binding.get("prior_pseudolabel_dependencies") == []
        and payload.get("model_state_sha256") == model_state_sha256
        and isinstance(payload.get("model_state_receipts"), dict)
        and payload.get("model_state_sha256")
        == staged_training._model_state_receipt_sha256(
            payload["model_state_receipts"]
        )
        and isinstance(payload.get("global_step"), int)
        and payload["global_step"] >= 1
        and payload.get("sampled_bank_step_count") == payload["global_step"]
        and payload.get("training_step_ledger_count") == payload["global_step"]
        and payload.get("candidate_bank_receipt_storage")
        == "immutable-finite-v4-run-reports-only"
        and isinstance(ledger_summary, dict)
        and ledger_summary.get("schema_version")
        == staged_training.TRAINING_STEP_LEDGER_SCHEMA
        and ledger_summary.get("entry_count") == payload["global_step"]
        and ledger_summary.get("compact_ledger_sha256")
        == payload.get("training_step_ledger_sha256")
        and ledger_summary.get("receipt_sha256")
        == _hash_json(_payload(ledger_summary))
        and isinstance(evidence, dict)
        and evidence.get("schema_version")
        == FINITE_TRAINING_REPORT_LEDGER_EVIDENCE_V4_SCHEMA
        and evidence.get("applied_step_count") == payload["global_step"]
        and evidence.get("receipt_sha256") == _hash_json(_payload(evidence))
        and isinstance(evidence.get("run_id"), str)
        and bool(evidence["run_id"])
        and isinstance(evidence.get("run_manifest_receipt_sha256"), str)
        and len(evidence["run_manifest_receipt_sha256"]) == 64
        and isinstance(payload.get("row_identity_record_count"), int)
        and payload["row_identity_record_count"] >= 1
        and all(
            isinstance(payload.get(name), list) and bool(payload[name])
            for name in (
                "training_row_ids",
                "training_animal_ids",
                "training_specimen_ids",
                "training_experiment_ids",
            )
        )
        and payload.get("learned_dependency_arrays")
        == {
            "prior_model_weights": [],
            "prior_features": [],
            "prior_pseudolabels": [],
        }
        and all(
            isinstance(payload.get(name), str)
            and len(payload[name]) == 64
            and not (set(payload[name].lower()) - set("0123456789abcdef"))
            for name in (
                "staged_checkpoint_file_sha256",
                "row_identity_records_sha256",
                "training_step_ledger_sha256",
            )
        )
    )
    if valid:
        try:
            psf_v4.verify_finite_psf_model_capability_v4(
                binding["finite_psf_capability"]
            )
        except (KeyError, TypeError, ValueError):
            valid = False
    if valid and require_source_file:
        try:
            checkpoint_path = _i_path(payload["staged_checkpoint_path"])
            run_root = checkpoint_path.parent.parent
            context = load_finite_training_run_v4(run_root)
            latest = (
                run_root
                / context["run_state"]["latest_checkpoint"]["relative_path"]
            ).resolve()
            _, checkpoint = _load_staged_training_checkpoint_v4(
                latest,
                device=context["manifest"]["execution_device"],
                expected_binding=context["manifest"]["staged_training_binding"],
                reports=context["training_reports"],
                run_id=context["manifest"]["run_id"],
                run_manifest_receipt_sha256=context["manifest"]["receipt_sha256"],
            )
            valid = (
                checkpoint_path == latest
                and _file_sha256(checkpoint_path)
                == payload["staged_checkpoint_file_sha256"]
                and _make_finite_staged_training_export_receipt_v4(
                    checkpoint_path,
                    checkpoint,
                    context["training_reports"],
                    run_id=context["manifest"]["run_id"],
                    run_manifest_receipt_sha256=context["manifest"][
                        "receipt_sha256"
                    ],
                )
                == receipt
            )
        except (KeyError, OSError, TypeError, ValueError):
            valid = False
    if not valid:
        raise ValueError("finite staged-training export receipt is invalid")
    return True


def load_finite_training_run_v4(run_directory):
    run_root = _i_path(run_directory)
    manifest, cache_manifest = _load_manifest(run_root)
    catalogue = runner_primitives._load_catalogue_inputs(
        run_root, manifest["catalogue"]
    )
    atlas_record = manifest["atlas"]
    atlas_path = runner_primitives._relative_child(
        run_root, atlas_record["relative_path"]
    )
    if _file_sha256(atlas_path) != atlas_record["file_sha256"]:
        raise ValueError("persisted finite-run atlas input hash differs")
    atlas = np.load(atlas_path, allow_pickle=False)
    runner_primitives.verify_atlas_binding_v3(atlas_record["binding"], atlas)
    run_state, checkpoint_path, reports = _load_run_state(run_root, manifest)
    state, _ = _load_staged_training_checkpoint_v4(
        checkpoint_path,
        device=manifest["execution_device"],
        expected_binding=manifest["staged_training_binding"],
        reports=reports,
        run_id=manifest["run_id"],
        run_manifest_receipt_sha256=manifest["receipt_sha256"],
    )
    if int(state["global_step"]) != int(run_state["applied_step_count"]):
        raise ValueError("finite checkpoint step differs from committed run state")
    return {
        "run_root": run_root,
        "manifest": manifest,
        "cache_manifest": cache_manifest,
        "catalogue": catalogue,
        "atlas_volume": atlas,
        "run_state": run_state,
        "training_reports": reports,
        "training_state": state,
    }


def make_finite_training_run_export_receipt_v4(run_directory):
    context = load_finite_training_run_v4(run_directory)
    checkpoint = (
        context["run_root"]
        / context["run_state"]["latest_checkpoint"]["relative_path"]
    )
    _, checkpoint_payload = _load_staged_training_checkpoint_v4(
        checkpoint,
        device=context["manifest"]["execution_device"],
        expected_binding=context["manifest"]["staged_training_binding"],
        reports=context["training_reports"],
        run_id=context["manifest"]["run_id"],
        run_manifest_receipt_sha256=context["manifest"]["receipt_sha256"],
    )
    staged_receipt = _make_finite_staged_training_export_receipt_v4(
        checkpoint,
        checkpoint_payload,
        context["training_reports"],
        run_id=context["manifest"]["run_id"],
        run_manifest_receipt_sha256=context["manifest"]["receipt_sha256"],
    )
    payload = {
        "schema_version": FINITE_TRAINING_RUN_EXPORT_RECEIPT_V4_SCHEMA,
        "run_id": context["manifest"]["run_id"],
        "run_manifest_receipt_sha256": context["manifest"]["receipt_sha256"],
        "run_state_receipt_sha256": context["run_state"]["receipt_sha256"],
        "frozen_cache_manifest_receipt_sha256": context["cache_manifest"][
            "receipt_sha256"
        ],
        "finite_psf_capability": _plain(
            context["manifest"]["finite_psf_capability"]
        ),
        "finite_psf_cache_run_contract": _plain(
            context["cache_manifest"]["finite_psf_run_contract"]
        ),
        "checkpoint": _plain(context["run_state"]["latest_checkpoint"]),
        "staged_training_export_receipt": _plain(staged_receipt),
        "calibration": {
            "status": UNCALIBRATED_STATUS,
            "calibration_receipt": None,
        },
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    return _with_receipt(payload)


def verify_finite_training_run_export_receipt_v4(receipt, run_directory):
    expected = make_finite_training_run_export_receipt_v4(run_directory)
    if receipt != expected:
        raise ValueError("finite v4 training-run export receipt changed")
    return True


def _row_indices(run_id, root_seed, step, row_count, batch_size):
    seed = int(
        _hash_json(
            {
                "domain": "anatomy-tracker.finite-training-row-order/v4",
                "run_id": run_id,
                "root_seed": str(root_seed),
                "global_step": int(step),
            }
        )[:16],
        16,
    )
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    return rng.choice(row_count, size=batch_size, replace=False).tolist()


def _commit_fault_injection_point(point):
    """Test hook: production execution intentionally does nothing."""


def _next_resume_checkpoint_relative(old_run_state):
    slots = ("checkpoints/resume_slot_0.pt", "checkpoints/resume_slot_1.pt")
    current = old_run_state["latest_checkpoint"]["relative_path"]
    if current not in slots:
        raise ValueError("finite resume checkpoint is not in a frozen slot")
    return slots[1] if current == slots[0] else slots[0]


def _pending_attempt_payload(context, train_report, row_indices, batch, attempt_index):
    manifest = context["manifest"]
    state = context["training_state"]
    row_psf = [identity["finite_psf"] for identity in batch["row_identity"]]
    pose_weight = torch.as_tensor(batch["pose_supervision_weight"])
    dense_weight = torch.as_tensor(batch["dense_deformation_supervision_weight"])
    return {
        "schema_version": FINITE_TRAINING_STEP_REPORT_V4_SCHEMA,
        "run_id": manifest["run_id"],
        "run_manifest_receipt_sha256": manifest["receipt_sha256"],
        "attempt_index": int(attempt_index),
        "global_step_before": int(train_report["step"]),
        "global_step_after": int(state["global_step"]),
        "row_cache_manifest_receipt_sha256": manifest["cache"][
            "manifest_receipt_sha256"
        ],
        "finite_psf_cache_run_contract_receipt_sha256": manifest["cache"][
            "finite_psf_cache_run_contract_receipt_sha256"
        ],
        "row_indices": list(row_indices),
        "row_identity": _plain(batch["row_identity"]),
        "ordered_row_finite_psf_identity_sha256": _hash_json(row_psf),
        "supervision_weight_summary": {
            "batch_row_count": len(batch["row_identity"]),
            "pose_positive_row_count": int((pose_weight > 0).sum().item()),
            "pose_weight_sum": float(pose_weight.sum().item()),
            "dense_positive_row_count": int((dense_weight > 0).sum().item()),
            "dense_weight_sum": float(dense_weight.sum().item()),
            "zero_weight_rows_retained": True,
        },
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
    first_attempt = int(old_run_state["attempt_count"])
    if [payload["attempt_index"] for payload in pending_payloads] != list(
        range(first_attempt, first_attempt + len(pending_payloads))
    ):
        raise ValueError("finite pending attempts are not one ledger suffix")
    checkpoint_relative = _next_resume_checkpoint_relative(old_run_state)
    checkpoint_path = run_root / checkpoint_relative
    staged_training.save_staged_training_checkpoint(state, checkpoint_path)
    runner_primitives._fsync_existing_file(checkpoint_path)
    checkpoint = _checkpoint_record(
        manifest, checkpoint_relative, _file_sha256(checkpoint_path)
    )
    interval = int(
        manifest["runner_config"]["archive_checkpoint_interval_applied_steps"]
    )
    target = int(manifest["runner_config"]["target_applied_steps"])
    last = pending_payloads[-1]
    applied = bool(last["training_report"]["optimizer_step_applied"])
    archive = None
    if applied and (
        int(state["global_step"]) % interval == 0
        or int(state["global_step"]) == target
    ):
        archive_relative = f"checkpoints/archive_step_{state['global_step']:08d}.pt"
        archive_path = run_root / archive_relative
        staged_training.save_staged_training_checkpoint(state, archive_path)
        runner_primitives._fsync_existing_file(archive_path)
        archive = _checkpoint_record(
            manifest, archive_relative, _file_sha256(archive_path)
        )
    _commit_fault_injection_point("after_checkpoint")
    reports = []
    report_records = []
    for offset, pending in enumerate(pending_payloads):
        report = _with_receipt(
            {
                **pending,
                "checkpoint": checkpoint,
                "archive_checkpoint": archive if offset + 1 == len(pending_payloads) else None,
            }
        )
        relative = f"reports/attempt_{pending['attempt_index']:08d}.json"
        path = run_root / relative
        runner_primitives._atomic_json(path, report)
        reports.append(report)
        report_records.append(
            {
                "relative_path": relative,
                "file_sha256": _file_sha256(path),
                "report_receipt_sha256": report["receipt_sha256"],
            }
        )
        _commit_fault_injection_point("after_report")
    _commit_fault_injection_point("after_reports")
    run_state = _with_receipt(
        {
            "schema_version": FINITE_TRAINING_RUN_STATE_V4_SCHEMA,
            "run_id": manifest["run_id"],
            "run_manifest_receipt_sha256": manifest["receipt_sha256"],
            "attempt_count": first_attempt + len(pending_payloads),
            "applied_step_count": int(state["global_step"]),
            "latest_checkpoint": checkpoint,
            "committed_reports": [
                *old_run_state["committed_reports"],
                *report_records,
            ],
        }
    )
    runner_primitives._atomic_json(run_root / "run_state.json", run_state)
    _commit_fault_injection_point("after_run_state")
    context["run_state"] = run_state
    context["training_reports"].extend(reports)
    return reports


def _run_finite_training_attempts_locked_v4(run_directory, max_attempts):
    context = load_finite_training_run_v4(run_directory)
    manifest = context["manifest"]
    config = manifest["runner_config"]
    target = int(config["target_applied_steps"])
    atlas = torch.as_tensor(
        context["atlas_volume"],
        device=manifest["execution_device"],
        dtype=torch.float32,
    )
    reports = []
    pending = []
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
            rows = row_cache_v4.load_training_rows_v4(
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
                axial_offsets_um=(),
                axial_weights=(),
                device=manifest["execution_device"],
                data_role=DEVELOPMENT_DATA_ROLE,
                finite_psf_capability=manifest["finite_psf_capability"],
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
                raise RuntimeError("finite trainer cannot reinterpret a sampled bank")
        except BaseException:
            staged_training._restore_rng_state(preparation_rng_state)
            reports.extend(_commit_attempt_batch(context, pending))
            pending = []
            raise
        train_report = staged_training.train_staged_step(
            context["training_state"], batch
        )
        attempt_index = int(context["run_state"]["attempt_count"]) + len(pending)
        pending.append(
            _pending_attempt_payload(
                context, train_report, indices, batch, attempt_index
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
            len(pending) >= commit_interval
            or archive_due
            or global_step >= target
        ):
            reports.extend(_commit_attempt_batch(context, pending))
            pending = []
    reports.extend(_commit_attempt_batch(context, pending))
    return reports


def run_finite_training_attempts_v4(run_directory, *, max_attempts):
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < 0
    ):
        raise ValueError("max_attempts must be a nonnegative integer")
    with runner_primitives._exclusive_run_lock(_i_path(run_directory)):
        return _run_finite_training_attempts_locked_v4(
            run_directory, max_attempts
        )


def run_finite_training_until_target_v4(run_directory):
    while True:
        context = load_finite_training_run_v4(run_directory)
        remaining = int(
            context["manifest"]["runner_config"]["target_applied_steps"]
        ) - int(context["training_state"]["global_step"])
        if remaining <= 0:
            return context["run_state"]
        # Near the target, ``remaining`` may be one even though AMP legitimately
        # needs several same-step retries while its scaler backs off.  Keep a
        # bounded operational retry window without allowing an overshoot.
        attempt_budget = max(
            remaining,
            int(
                context["manifest"]["runner_config"][
                    "checkpoint_commit_interval_attempts"
                ]
            ),
        )
        reports = run_finite_training_attempts_v4(
            run_directory, max_attempts=attempt_budget
        )
        if reports and any(
            report["training_report"]["optimizer_step_applied"]
            for report in reports
        ):
            continue
        raise RuntimeError("all finite training attempts overflowed")


__all__ = [
    "FINITE_PSF_TRAINING_SOURCE_V4_SCHEMA",
    "FINITE_ROW_SAMPLING_POLICY_V4_SCHEMA",
    "FINITE_TRAINING_COMMIT_CONTRACT_V4_SCHEMA",
    "FINITE_TRAINING_RUN_EXPORT_RECEIPT_V4_SCHEMA",
    "FINITE_TRAINING_RUN_STATE_V4_SCHEMA",
    "FINITE_TRAINING_RUN_V4_SCHEMA",
    "FINITE_TRAINING_STEP_REPORT_V4_SCHEMA",
    "initialize_finite_training_run_v4",
    "load_finite_training_run_v4",
    "make_finite_training_run_export_receipt_v4",
    "run_finite_training_attempts_v4",
    "run_finite_training_until_target_v4",
    "verify_finite_staged_training_export_receipt_v4",
    "verify_finite_training_run_export_receipt_v4",
]
