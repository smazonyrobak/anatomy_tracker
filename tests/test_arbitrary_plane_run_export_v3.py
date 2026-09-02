import copy
import hashlib
import inspect
import json

import numpy as np
import pytest
import torch

import arbitrary_plane_production_v3_fixtures as fixture
import training.arbitrary_plane_inference_v3 as inference_v3
import training.arbitrary_plane_row_cache_v3 as row_cache_v3
import training.arbitrary_plane_run_export_v3 as export_v3
import training.arbitrary_plane_training_runner_v3 as runner_v3
from training.arbitrary_plane_joint_model import ArbitraryPlaneJointModel


def _prepared(tmp_path, target=2):
    cache = tmp_path / "cache"
    binding = fixture.generator_binding()
    row_cache_v3.initialize_training_row_cache_v3(
        cache,
        generator_binding=binding,
        generation_config={"row_count": 1, "plane_domain": "all brain-intersecting"},
        seed_record={"root_seed": "0xabc", "subject_seed": "0xdef"},
    )
    row_cache_v3.append_training_rows_v3(cache, [fixture.row(0)])
    frozen = row_cache_v3.freeze_training_row_cache_v3(cache)
    atlas_source = tmp_path / "allen-source.bin"
    atlas_source.write_bytes(b"authenticated Allen source fixture")
    run = tmp_path / "run"
    manifest, _ = runner_v3.initialize_training_run_v3(
        run,
        cache_directory=cache,
        expected_generator_binding=binding,
        catalogue=fixture.catalogue(),
        atlas_volume=fixture.atlas(),
        atlas_source_assets=(
            {
                "path": str(atlas_source),
                "role": "Allen template and annotation test asset",
                "sha256": hashlib.sha256(atlas_source.read_bytes()).hexdigest(),
            },
        ),
        atlas_preprocessing={"normalization": "fixed deterministic fixture"},
        model_kwargs=fixture.model_kwargs(),
        training_config=fixture.training_config(),
        runner_config=fixture.runner_config(target),
        device="cpu",
    )
    runner_v3.run_training_attempts_v3(run, max_attempts=1)
    return run, frozen, manifest


def _semantics(manifest):
    binding = manifest["atlas"]["binding"]
    return {
        "schema_version": inference_v3.ATLAS_SEMANTICS_V3_SCHEMA,
        "atlas_name": "Allen fixture atlas",
        "atlas_version": "fixture-v1",
        "processed_channel_names": ["template", "boundary"],
        "processed_channel_recipes": [
            "authenticated deterministic fixture template",
            "authenticated deterministic fixture boundary",
        ],
        "source_assets": [
            {
                "asset_role": asset["role"],
                "uri": asset["path"],
                "sha256": asset["sha256"],
            }
            for asset in binding["source_assets"]
        ],
        "source_format": "fixture binary transformed to C-AP-DV-ML float32",
        "nrrd_index_order": "F",
        "array_axis_order": ["AP", "DV", "ML"],
        "positive_axis_directions": ["posterior", "ventral", "right"],
        "voxel_center_convention": "origin is the centre of voxel index [0,0,0]",
        "normalization_parameters": binding["preprocessing"],
    }


def test_export_milestone_is_safe_loaded_and_binds_exact_run(tmp_path):
    run, frozen, manifest = _prepared(tmp_path)
    target = tmp_path / "exports" / "milestone.pt"
    annotation = np.ones((10, 10, 10), dtype=np.int32)
    report = export_v3.export_training_run_to_inference_checkpoint_v3(
        run,
        target,
        atlas_semantics=_semantics(manifest),
        annotation_volume_ap_dv_ml=annotation,
    )

    assert export_v3.verify_training_run_inference_export_report_v3(report)
    assert report["export_status"] == "milestone"
    assert report["calibration"] == {
        "status": export_v3.UNCALIBRATED_STATUS,
        "calibration_receipt": None,
    }
    provenance = report["dataset_provenance"]
    assert set(runner_v3.load_training_run_v3(run)["manifest"]["model_kwargs"]) == set(
        inspect.signature(ArbitraryPlaneJointModel).parameters
    )
    assert runner_v3.load_training_run_v3(run)["manifest"]["model_kwargs"][
        "deformation_support_floor"
    ] == 1e-4
    assert provenance["frozen_cache"]["manifest_receipt_sha256"] == frozen[
        "receipt_sha256"
    ]
    assert provenance["frozen_cache"]["generator_binding"] == frozen[
        "generator_binding"
    ]
    assert provenance["run_binding"]["run_manifest_receipt_sha256"] == manifest[
        "receipt_sha256"
    ]
    assert provenance["run_binding"]["applied_step_count"] == 1
    assert provenance["run_binding"]["training_report_ledger"]["attempt_count"] == 1
    assert report["inference_contract"]["atlas_assets"][
        "annotation_volume_receipt"
    ]["shape"] == [10, 10, 10]
    assert report["inference_contract"]["finite_psf"]["axial_offsets_um"] == [
        -0.5,
        0.0,
        0.5,
    ]
    assert report["training_receipt"]["staged_checkpoint_file_sha256"] == (
        provenance["run_binding"]["latest_staged_checkpoint"]["file_sha256"]
    )
    context = runner_v3.load_training_run_v3(run)
    loaded = inference_v3.load_arbitrary_plane_inference_v3(
        target, context["catalogue"]
    )
    assert loaded["checkpoint_id"] == report["checkpoint"]["checkpoint_id"]
    assert loaded["model_state_sha256"] == report["training_receipt"][
        "model_state_sha256"
    ]
    checkpoint = torch.load(target, map_location="cpu", weights_only=True)
    assert checkpoint["calibration_receipt"] is None
    assert checkpoint["provenance"]["prior_trained_model_dependencies"] == []
    assert checkpoint["provenance"]["dataset_provenance"] == [provenance]

    tampered_report = copy.deepcopy(report)
    tampered_report["export_status"] = "completed"
    with pytest.raises(ValueError, match="report failed authentication"):
        export_v3.verify_training_run_inference_export_report_v3(tampered_report)


def test_export_rejects_wrong_semantics_annotation_non_i_and_overwrite(tmp_path):
    run, _, manifest = _prepared(tmp_path)
    semantics = _semantics(manifest)

    wrong_semantics = copy.deepcopy(semantics)
    wrong_semantics["normalization_parameters"] = {"normalization": "wrong"}
    with pytest.raises(ValueError, match="semantics do not match"):
        export_v3.export_training_run_to_inference_checkpoint_v3(
            run,
            tmp_path / "wrong-semantics.pt",
            atlas_semantics=wrong_semantics,
        )

    with pytest.raises(ValueError, match="annotation asset"):
        export_v3.export_training_run_to_inference_checkpoint_v3(
            run,
            tmp_path / "wrong-annotation.pt",
            atlas_semantics=semantics,
            annotation_volume_ap_dv_ml=np.ones((9, 10, 10), dtype=np.int32),
        )

    with pytest.raises(ValueError, match="only I"):
        export_v3.export_training_run_to_inference_checkpoint_v3(
            run,
            "C:\\forbidden-export-v3.pt",
            atlas_semantics=semantics,
        )

    occupied = tmp_path / "occupied.pt"
    occupied.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError, match="already exists"):
        export_v3.export_training_run_to_inference_checkpoint_v3(
            run,
            occupied,
            atlas_semantics=semantics,
        )
    assert occupied.read_bytes() == b"do not overwrite"


def test_export_rejects_tampered_run_and_exported_checkpoint(tmp_path):
    run, _, manifest = _prepared(tmp_path)
    report_path = run / "reports" / "attempt_00000000.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["global_step_after"] = 99
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="report hash"):
        export_v3.export_training_run_to_inference_checkpoint_v3(
            run,
            tmp_path / "tampered-run.pt",
            atlas_semantics=_semantics(manifest),
        )

    clean_root = tmp_path / "clean"
    clean_run, _, clean_manifest = _prepared(clean_root)
    target = clean_root / "export.pt"
    export_v3.export_training_run_to_inference_checkpoint_v3(
        clean_run,
        target,
        atlas_semantics=_semantics(clean_manifest),
    )
    checkpoint = torch.load(target, map_location="cpu", weights_only=True)
    first_name = next(iter(checkpoint["state_dict"]))
    checkpoint["state_dict"][first_name] = checkpoint["state_dict"][
        first_name
    ].clone()
    checkpoint["state_dict"][first_name].view(-1)[0] += 1.0
    torch.save(checkpoint, target)
    catalogue = runner_v3.load_training_run_v3(clean_run)["catalogue"]
    with pytest.raises(ValueError, match="checkpoint or dependency binding"):
        inference_v3.load_arbitrary_plane_inference_v3(target, catalogue)
