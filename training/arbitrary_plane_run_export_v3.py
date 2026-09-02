"""Export one authenticated training-run milestone as standalone v3 inference."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
import torch

import training.arbitrary_plane_inference_v3 as inference_v3
import training.arbitrary_plane_training_runner_v3 as runner_v3


TRAINING_RUN_INFERENCE_EXPORT_V3_SCHEMA = (
    "anatomy-tracker.training-run-inference-export/v3"
)
TRAINING_RUN_DATASET_PROVENANCE_V3_SCHEMA = (
    "anatomy-tracker.training-run-dataset-provenance/v3"
)
TRAINING_RUN_REPORT_LEDGER_BINDING_V3_SCHEMA = (
    "anatomy-tracker.training-run-report-ledger-binding/v3"
)
UNCALIBRATED_STATUS = "absent-uncalibrated"


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("export receipts require finite values")
    return value


def _sha(value):
    return hashlib.sha256(
        json.dumps(
            _plain(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _i_path(path):
    target = Path(path).resolve()
    if os.path.splitdrive(str(target))[0].upper() != "I:":
        raise ValueError("training-run inference exports must use only I:")
    return target


def _atomic_torch_save_new_i(path, payload):
    target = _i_path(path)
    if os.path.lexists(target):
        raise FileExistsError("inference export target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError("inference export target already exists") from error
        except OSError as error:
            if os.path.lexists(target):
                raise FileExistsError("inference export target already exists") from error
            if os.name != "nt":
                raise RuntimeError(
                    "atomic no-overwrite checkpoint publication is unavailable"
                ) from error
            try:
                os.rename(temporary, target)
            except FileExistsError as rename_error:
                raise FileExistsError(
                    "inference export target already exists"
                ) from rename_error
        if temporary.exists():
            temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return target


def _verify_semantics_match_run(atlas_semantics, atlas_binding):
    expected_assets = [
        {
            "asset_role": asset["role"],
            "uri": asset["path"],
            "sha256": asset["sha256"],
        }
        for asset in atlas_binding["source_assets"]
    ]
    if (
        not isinstance(atlas_semantics, dict)
        or atlas_semantics.get("source_assets") != expected_assets
        or atlas_semantics.get("normalization_parameters")
        != atlas_binding["preprocessing"]
    ):
        raise ValueError(
            "atlas semantics do not match the authenticated training-run atlas binding"
        )
    return True


def _report_ledger_binding(context):
    run_state = context["run_state"]
    reports = context["training_reports"]
    payload = {
        "schema_version": TRAINING_RUN_REPORT_LEDGER_BINDING_V3_SCHEMA,
        "run_id": context["manifest"]["run_id"],
        "run_manifest_receipt_sha256": context["manifest"]["receipt_sha256"],
        "run_state_receipt_sha256": run_state["receipt_sha256"],
        "attempt_count": int(run_state["attempt_count"]),
        "applied_step_count": int(run_state["applied_step_count"]),
        "committed_report_records": _plain(run_state["committed_reports"]),
        "ordered_report_receipts_sha256": _sha(
            [report["receipt_sha256"] for report in reports]
        ),
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def _dataset_provenance(context, training_receipt, latest_checkpoint_path):
    manifest = context["manifest"]
    cache = context["cache_manifest"]
    run_state = context["run_state"]
    ledger = _report_ledger_binding(context)
    run_binding = {
        "run_id": manifest["run_id"],
        "run_manifest_receipt_sha256": manifest["receipt_sha256"],
        "run_state_receipt_sha256": run_state["receipt_sha256"],
        "run_status": (
            "completed"
            if int(run_state["applied_step_count"])
            >= int(manifest["runner_config"]["target_applied_steps"])
            else "milestone"
        ),
        "target_applied_steps": int(
            manifest["runner_config"]["target_applied_steps"]
        ),
        "applied_step_count": int(run_state["applied_step_count"]),
        "latest_staged_checkpoint": {
            **_plain(run_state["latest_checkpoint"]),
            "absolute_path": str(latest_checkpoint_path),
        },
        "training_report_ledger": ledger,
        "training_receipt_sha256": training_receipt["receipt_sha256"],
    }
    payload = {
        "schema_version": TRAINING_RUN_DATASET_PROVENANCE_V3_SCHEMA,
        "data_role": cache["data_role"],
        "frozen_cache": {
            "directory": manifest["cache"]["directory"],
            "manifest_receipt_sha256": cache["receipt_sha256"],
            "status": cache["status"],
            "row_count": int(cache["row_count"]),
            "freeze_audit": _plain(cache["freeze_audit"]),
            "generation_config": _plain(cache["generation_config"]),
            "seed_record": _plain(cache["seed_record"]),
            "generator_binding": _plain(cache["generator_binding"]),
        },
        "run_binding": run_binding,
        "catalogue_binding": {
            "catalogue_id": context["catalogue"]["catalogue_id"],
            "catalogue_receipt_sha256": context["catalogue"]["receipt_sha256"],
            "cell_count": int(context["catalogue"]["counts"]["cell_count"]),
        },
        "atlas_binding_receipt_sha256": manifest["atlas"]["binding"][
            "receipt_sha256"
        ],
        "runner_source_sha256": _plain(manifest["runner_source_sha256"]),
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    if "finite_psf_capability" in manifest:
        payload["finite_psf_capability"] = _plain(
            manifest["finite_psf_capability"]
        )
        payload["finite_psf_training_schedule_source"] = _plain(
            manifest["finite_psf_training_schedule_source"]
        )
    else:
        payload["finite_psf_contract_receipt_sha256"] = manifest[
            "finite_psf_contract"
        ]["receipt_sha256"]
    return {**payload, "receipt_sha256": _sha(payload)}


def _make_provenance(context, training_receipt, latest_checkpoint_path):
    return {
        "initialization": "fresh_random",
        "architecture_source": inference_v3.ARCHITECTURE_MODULE,
        "dataset_provenance": [
            _dataset_provenance(
                context, training_receipt, latest_checkpoint_path
            )
        ],
        "animal_specimen_experiment_id_contract": (
            "exact authenticated animal_id, specimen_id, and experiment_id values "
            "from the frozen V3 training-row cache and staged training receipt"
        ),
        "prior_trained_model_dependencies": [],
        "prior_model_feature_dependencies": [],
        "pseudolabel_dependencies": [],
    }


def verify_training_run_inference_export_report_v3(report):
    payload = (
        {key: value for key, value in report.items() if key != "receipt_sha256"}
        if isinstance(report, dict)
        else {}
    )
    checkpoint = payload.get("checkpoint", {})
    safe_load = payload.get("safe_load_verification", {})
    provenance = payload.get("dataset_provenance", {})
    ledger = provenance.get("run_binding", {}).get("training_report_ledger", {})
    ledger_payload = {
        key: value for key, value in ledger.items() if key != "receipt_sha256"
    }
    provenance_payload = {
        key: value for key, value in provenance.items() if key != "receipt_sha256"
    }
    valid = (
        payload.get("schema_version") == TRAINING_RUN_INFERENCE_EXPORT_V3_SCHEMA
        and payload.get("export_status") in {"completed", "milestone"}
        and payload.get("calibration")
        == {"status": UNCALIBRATED_STATUS, "calibration_receipt": None}
        and payload.get("prior_model_weight_dependencies") == []
        and payload.get("prior_feature_dependencies") == []
        and payload.get("prior_pseudolabel_dependencies") == []
        and isinstance(provenance, dict)
        and provenance.get("schema_version")
        == TRAINING_RUN_DATASET_PROVENANCE_V3_SCHEMA
        and provenance.get("receipt_sha256") == _sha(provenance_payload)
        and ledger.get("schema_version")
        == TRAINING_RUN_REPORT_LEDGER_BINDING_V3_SCHEMA
        and ledger.get("receipt_sha256") == _sha(ledger_payload)
        and payload.get("training_receipt", {}).get("receipt_sha256")
        == provenance.get("run_binding", {}).get("training_receipt_sha256")
        and checkpoint.get("checkpoint_id") == safe_load.get("checkpoint_id")
        and checkpoint.get("checkpoint_binding_id")
        == safe_load.get("checkpoint_binding_id")
        and checkpoint.get("model_state_sha256")
        == safe_load.get("model_state_sha256")
        and checkpoint.get("file_sha256") == safe_load.get("file_sha256")
        and checkpoint.get("inference_contract_receipt_sha256")
        == safe_load.get("inference_contract_receipt_sha256")
        and report.get("receipt_sha256") == _sha(payload)
    )
    if not valid:
        raise ValueError("training-run inference export report failed authentication")
    inference_v3.verify_inference_contract_v3(payload["inference_contract"])
    return True


def export_training_run_to_inference_checkpoint_v3(
    run_directory,
    checkpoint_path,
    *,
    atlas_semantics,
    annotation_volume_ap_dv_ml=None,
    safe_load_device="cpu",
):
    """Freeze and safe-load the exact current completed or milestone run state."""
    target = _i_path(checkpoint_path)
    if os.path.lexists(target):
        raise FileExistsError("inference export target already exists")
    run_root = runner_v3._i_path(run_directory)
    created = False
    with runner_v3._exclusive_run_lock(run_root):
        context = runner_v3.load_training_run_v3(run_root)
        manifest = context["manifest"]
        run_state = context["run_state"]
        latest_checkpoint_path = (
            context["run_root"]
            / run_state["latest_checkpoint"]["relative_path"]
        ).resolve(strict=True)
        training_receipt = runner_v3.make_training_run_export_receipt_v3(
            run_root
        )
        if (
            training_receipt["staged_checkpoint_path"]
            != str(latest_checkpoint_path)
            or training_receipt["staged_checkpoint_file_sha256"]
            != run_state["latest_checkpoint"]["file_sha256"]
            or int(training_receipt["global_step"])
            != int(run_state["applied_step_count"])
            or training_receipt["binding"]
            != manifest["staged_training_binding"]
        ):
            raise ValueError(
                "staged export receipt is not the exact latest authenticated run state"
            )
        _verify_semantics_match_run(
            atlas_semantics, manifest["atlas"]["binding"]
        )
        geometry = context["catalogue"]["support_geometry"]
        capability = manifest.get("finite_psf_capability")
        if capability is None:
            psf = manifest["finite_psf_contract"]
            offsets = psf["axial_offsets_um"]
            weights = psf["axial_weights"]
        else:
            offsets = weights = None
        inference_contract = inference_v3.make_inference_contract_v3(
            context["atlas_volume"],
            geometry["origin_ap_dv_ml_um"],
            geometry["voxel_size_ap_dv_ml_um"],
            offsets,
            weights,
            atlas_semantics=atlas_semantics,
            annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
            finite_psf_capability=capability,
        )
        provenance = _make_provenance(
            context, training_receipt, latest_checkpoint_path
        )
        checkpoint = inference_v3.make_arbitrary_plane_joint_checkpoint_v3(
            context["training_state"]["model"],
            manifest["model_kwargs"],
            context["catalogue"],
            provenance,
            training_receipt,
            inference_contract=inference_contract,
            calibration_receipt=None,
        )
        _atomic_torch_save_new_i(target, checkpoint)
        created = True
        try:
            loaded = inference_v3.load_arbitrary_plane_inference_v3(
                target,
                context["catalogue"],
                device=safe_load_device,
            )
        except BaseException:
            if created and target.is_file():
                target.unlink()
            raise
        dataset_provenance = provenance["dataset_provenance"][0]
        export_status = dataset_provenance["run_binding"]["run_status"]
        report_payload = {
            "schema_version": TRAINING_RUN_INFERENCE_EXPORT_V3_SCHEMA,
            "export_status": export_status,
            "exporter_source_sha256": _file_sha256(Path(__file__).resolve()),
            "dataset_provenance": dataset_provenance,
            "training_receipt": _plain(training_receipt),
            "inference_contract": _plain(inference_contract),
            "checkpoint": {
                "path": str(target),
                "file_sha256": loaded["checkpoint_file_sha256"],
                "checkpoint_id": loaded["checkpoint_id"],
                "checkpoint_binding_id": loaded["checkpoint_binding_id"],
                "model_state_sha256": loaded["model_state_sha256"],
                "inference_contract_receipt_sha256": inference_contract[
                    "receipt_sha256"
                ],
            },
            "safe_load_verification": {
                "device": loaded["device"],
                "catalogue_id": loaded["catalogue_id"],
                "file_sha256": loaded["checkpoint_file_sha256"],
                "checkpoint_id": loaded["checkpoint_id"],
                "checkpoint_binding_id": loaded["checkpoint_binding_id"],
                "model_state_sha256": loaded["model_state_sha256"],
                "inference_contract_receipt_sha256": loaded[
                    "inference_contract"
                ]["receipt_sha256"],
                "verified": True,
            },
            "calibration": {
                "status": UNCALIBRATED_STATUS,
                "calibration_receipt": None,
            },
            "prior_model_weight_dependencies": [],
            "prior_feature_dependencies": [],
            "prior_pseudolabel_dependencies": [],
        }
        report = {**report_payload, "receipt_sha256": _sha(report_payload)}
        try:
            verify_training_run_inference_export_report_v3(report)
        except BaseException:
            if target.is_file():
                target.unlink()
            raise
        return report
