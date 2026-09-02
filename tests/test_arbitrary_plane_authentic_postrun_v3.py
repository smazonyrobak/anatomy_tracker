import copy
import hashlib
import json

import numpy as np
import pytest

import arbitrary_plane_production_v3_fixtures as fixture
import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_inference_v3 as inference_v3
import training.arbitrary_plane_row_cache_v3 as row_cache_v3
import training.arbitrary_plane_training_row_v3 as training_row_v3
import training.arbitrary_plane_training_runner_v3 as runner_v3
import training.run_arbitrary_plane_authentic_postrun_v3 as postrun_v3
from training.verify_arbitrary_plane_authentic_package_v3 import (
    verify_arbitrary_plane_authentic_package_v3,
)


def _row(index, split, mode):
    row = copy.deepcopy(fixture.row(index, split=split))
    row["selected_mode"] = mode
    if mode == "smart-brush-absent":
        row["arrays"]["model_input_channels_float32"][..., 1:] = 0.0
    row["array_receipts"] = {
        name: acquisition_v2._array_receipt(value)
        for name, value in row["arrays"].items()
    }
    row["receipt_sha256"] = acquisition_v2._payload_sha256(
        training_row_v3.training_row_receipt_v3(row)
    )
    return row


def _frozen_cache(path, rows):
    row_cache_v3.initialize_training_row_cache_v3(
        path,
        generator_binding=fixture.generator_binding(),
        generation_config={
            "plane_domain": "all brain-intersecting",
            "row_count": len(rows),
        },
        seed_record={"root_seed": "0x1234", "purpose": "postrun-test"},
    )
    row_cache_v3.append_training_rows_v3(path, rows)
    return row_cache_v3.freeze_training_row_cache_v3(path)


def _prepared_run(root, target, attempts):
    catalogue = fixture.catalogue()
    atlas = fixture.atlas()
    training_cache = root / "training-cache"
    _frozen_cache(
        training_cache,
        [_row(0, "development-training", "smart-brush-accurate")],
    )
    source = root / "atlas-source.bin"
    source.write_bytes(b"authenticated Allen fixture for postrun")
    run = root / "training-run"
    manifest, _ = runner_v3.initialize_training_run_v3(
        run,
        cache_directory=training_cache,
        expected_generator_binding=fixture.generator_binding(),
        catalogue=catalogue,
        atlas_volume=atlas,
        atlas_source_assets=(
            {
                "path": str(source),
                "role": "test atlas",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        ),
        atlas_preprocessing={"normalization": "fixed postrun test"},
        model_kwargs=fixture.model_kwargs(),
        training_config=fixture.training_config(),
        runner_config=fixture.runner_config(target=target),
        device="cpu",
    )
    if attempts:
        runner_v3.run_training_attempts_v3(run, max_attempts=attempts)
    semantics = {
        "schema_version": inference_v3.ATLAS_SEMANTICS_V3_SCHEMA,
        "atlas_name": "postrun test atlas",
        "atlas_version": "test-v1",
        "processed_channel_names": ["intensity", "boundary"],
        "processed_channel_recipes": ["fixed fixture", "fixed boundary"],
        "source_assets": [
            {
                "asset_role": asset["role"],
                "uri": asset["path"],
                "sha256": asset["sha256"],
            }
            for asset in manifest["atlas"]["binding"]["source_assets"]
        ],
        "source_format": "synthetic test tensor",
        "nrrd_index_order": "F",
        "array_axis_order": ["AP", "DV", "ML"],
        "positive_axis_directions": ["positive AP", "positive DV", "positive ML"],
        "voxel_center_convention": "integer array coordinates denote voxel centres",
        "normalization_parameters": manifest["atlas"]["binding"]["preprocessing"],
    }
    return run, catalogue, atlas, semantics


def test_completed_postrun_binds_exact_checkpoint_cache_and_every_dev_row(tmp_path):
    run, catalogue, atlas, semantics = _prepared_run(tmp_path / "complete", 1, 1)
    development_cache = tmp_path / "complete" / "development-cache"
    _frozen_cache(
        development_cache,
        [
            _row(1, "development-evaluation", "smart-brush-accurate"),
            _row(2, "development-evaluation", "smart-brush-imperfect"),
            _row(3, "development-evaluation", "smart-brush-absent"),
        ],
    )
    output = tmp_path / "complete" / "postrun"
    bundle = postrun_v3.run_arbitrary_plane_authentic_postrun_v3(
        run,
        development_cache,
        output,
        atlas_semantics=semantics,
        development_evaluation_animal_ids=("animal-3", "animal-1", "animal-2"),
        annotation_volume_ap_dv_ml=np.ones(atlas.shape[-3:], dtype=np.int64),
        top_k=3,
        refinement_steps=1,
        pose_only_steps=0,
        retrieval_shape_h_w=(4, 4),
        catalogue_chunk_size=2,
        gauss_hermite_order=3,
        evaluation_seed=719,
        feature_cache_build_chunk_size=2,
        device="cpu",
    )
    assert bundle["run_binding"]["status"] == "completed"
    assert bundle["run_binding"]["applied_step_count"] == 1
    assert bundle["development_cache_binding"]["selected_row_indices"] == [0, 1, 2]
    assert bundle["configuration"]["development_evaluation_animal_ids"] == [
        "animal-1",
        "animal-2",
        "animal-3",
    ]
    assert verify_arbitrary_plane_authentic_package_v3(output)
    evaluation_report = json.loads(
        (output / "internal_development_evaluation" / "development_evaluation_report.json").read_text()
    )
    assert evaluation_report["row_accounting"]["selected_row_count"] == 3
    assert evaluation_report["row_accounting"]["reported_row_count"] == 3
    assert evaluation_report["row_accounting"]["no_rows_dropped"] is True
    assert len(list((output / "internal_development_evaluation" / "raw_predictions").glob("*.pt"))) == 3
    loaded = inference_v3.load_arbitrary_plane_inference_v3(
        output / "inference" / "completed_checkpoint.pt", catalogue
    )
    cache = inference_v3.load_arbitrary_plane_catalogue_feature_cache_v3(
        output / "inference" / "complete_catalogue_features.pt", loaded, catalogue
    )
    assert cache["cache_receipt"]["checkpoint_binding"]["checkpoint_id"] == loaded[
        "checkpoint_id"
    ]
    with pytest.raises(FileExistsError, match="must be new"):
        postrun_v3.run_arbitrary_plane_authentic_postrun_v3(
            run,
            development_cache,
            output,
            atlas_semantics=semantics,
            development_evaluation_animal_ids=("animal-1", "animal-2", "animal-3"),
        )
    with (output / "inference" / "complete_catalogue_features.pt").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="cache|file hash"):
        verify_arbitrary_plane_authentic_package_v3(output)


def test_incomplete_run_is_rejected_before_postrun_output_is_created(tmp_path):
    run, _, _, semantics = _prepared_run(tmp_path / "incomplete", 2, 1)
    development_cache = tmp_path / "incomplete" / "development-cache"
    _frozen_cache(
        development_cache,
        [_row(1, "development-evaluation", "smart-brush-accurate")],
    )
    output = tmp_path / "incomplete" / "postrun"
    with pytest.raises(ValueError, match="exact completed training target"):
        postrun_v3.run_arbitrary_plane_authentic_postrun_v3(
            run,
            development_cache,
            output,
            atlas_semantics=semantics,
            development_evaluation_animal_ids=("animal-1",),
        )
    assert not output.exists()
