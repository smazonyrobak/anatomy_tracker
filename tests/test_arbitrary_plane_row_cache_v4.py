import copy
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_deformation_gauge_v4 as deformation_gauge_v4
import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_row_cache_v4 as cache_v4


def _digest(label):
    return acquisition_v2._payload_sha256({"finite-cache-v4-test": label})


@pytest.fixture
def i_cache_root():
    parent = Path("I:/AnatomyTracker/test_tmp/finite_cache_v4")
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / uuid.uuid4().hex
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _run_contract(render_mode="finite_boxcar"):
    return cache_v4.make_finite_psf_cache_run_contract_v4(render_mode)


def _generation_lineage():
    return {
        "schema_version": cache_v4.GENERATION_LINEAGE_V4_SCHEMA,
        "generation_run_id": "finite-cache-v4-test-run",
        "source_commit": "1" * 40,
        "split": "development-cache-v4",
        "parent_dataset_receipt_sha256": _digest("parent-dataset"),
    }


def _binding(render_mode="finite_boxcar"):
    generation_config = {
        "schema_version": "anatomy-tracker.finite-cache-test-generator/v4",
        "algorithm": "finite-cache-test-generator/v4",
        "row_count": 2,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    return cache_v4.make_generator_binding_v4(
        generator_ids=[generation_config["algorithm"]],
        source_sha256={"fixture.py": _digest("source")},
        geometry_gauge_contract=(
            deformation_gauge_v4.direct_deformation_target_contract_v4()
        ),
        generation_config=generation_config,
        seed_record={"root_seed_uint64": "0x0000000000001234"},
        generation_lineage=_generation_lineage(),
        finite_psf_run_contract=_run_contract(render_mode),
    )


def _row(index, thickness=25.0, render_mode="finite_boxcar"):
    height = width = 6
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    arrays = {
        "model_input_channels_float32": np.stack(
            (
                np.full((height, width), 0.5, np.float32),
                np.ones((height, width), np.float32),
                np.ones((height, width), np.float32),
            ),
            axis=-1,
        ),
        "source_label_ground_truth_canvas_int64": np.ones(
            (height, width), np.int64
        ),
        "source_tissue_ground_truth_mask": np.ones((height, width), bool),
        "target_ccf_coordinates_ap_dv_ml_um_float64": np.zeros(
            (height, width, 3), np.float64
        ),
        "target_valid_correspondence_mask": np.ones((height, width), bool),
        "target_correspondence_weight_float32": np.ones(
            (height, width), np.float32
        ),
        "target_correspondence_abstention_mask": np.zeros(
            (height, width), bool
        ),
        "truth_section_pullback_map_yx_px_float64": np.stack(
            (y, x), axis=-1
        ).astype(np.float64),
        "truth_section_pullback_stationary_velocity_yx_px_float64": np.zeros(
            (height, width, 2), np.float64
        ),
        "truth_section_deformation_valid_mask": np.ones((height, width), bool),
    }
    finite_psf = psf_v4.make_finite_psf_schedule_v4(
        render_mode,
        thickness,
        thickness_selection_sha256=_digest(f"selection-{index}"),
    )
    slab_receipt = _digest(f"slab-{index}")
    contract = {
        **finite_psf,
        "slab_observation_v4_receipt_sha256": slab_receipt,
    }
    gauge_contract = deformation_gauge_v4.direct_deformation_target_contract_v4()
    row = {
        "schema_version": psf_v4.TRAINING_ROW_V4_SCHEMA,
        "source_observation_receipt_sha256": _digest(f"observation-{index}"),
        "lineage": {
            "animal_id": f"animal-{index}",
            "specimen_id": f"specimen-{index}",
            "experiment_id": f"experiment-{index}",
            "synthetic_animal_id": f"synthetic-animal-{index}",
            "section_id": f"section-{index}",
            "split": "development-cache-v4",
        },
        "upstream_reference": {
            "schema_version": "anatomy-tracker.finite-cache-test-generator/v4",
            "algorithm": "finite-cache-test-generator/v4",
            "implementation_source_sha256": {
                "fixture.py": _digest("source")
            },
            "slab_observation_id": _digest(f"slab-id-{index}"),
            "centre_plane_targets_receipt_sha256": _digest(f"centre-{index}"),
            "slab_observation_v4_receipt_sha256": slab_receipt,
            "finite_psf_sha256": finite_psf["finite_psf_sha256"],
            "finite_psf_capability_sha256": finite_psf[
                "finite_psf_capability_sha256"
            ],
        },
        "numeric_rng_provenance": {"sample_index": index},
        "rng_sources": {"root_seed_uint64": "0x0000000000001234"},
        "selected_mode": "smart-brush-imperfect",
        "selected_descendant_id": _digest(f"descendant-{index}"),
        "deformation_pose_gauge_reference": {
            **gauge_contract,
            "direct_deformation_target_id": _digest(f"gauge-id-{index}"),
            "receipt_sha256": _digest(f"gauge-receipt-{index}"),
        },
        "reflection_state": "none",
        "reflection_representation_index": 0,
        "reflection_representation_affine_xy_float64": np.eye(3).tolist(),
        "canonical_effective_quicknii_ouv_float64": [
            [3.0, 3.0, 5.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ],
        "observed_effective_quicknii_ouv_float64": [
            [3.0, 3.0, 5.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ],
        "proper_physical_pose_unchanged": [
            [3.0, 3.0, 5.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ],
        "prior_model_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
        "reflection_transform_id": _digest(f"reflection-transform-{index}"),
        "reflection_realization_id": _digest(f"reflection-realization-{index}"),
        "paired_view_group_id": _digest(f"paired-view-{index}"),
        "synthetic_realization_id": _digest(f"realization-{index}"),
        "paired_mode_reflected_receipts": {},
        "arrays": arrays,
        "array_receipts": {
            name: acquisition_v2._array_receipt(value)
            for name, value in arrays.items()
        },
        "finite_psf_contract": contract,
    }
    row["training_row_id"] = acquisition_v2._payload_sha256(
        {
            "domain": psf_v4.TRAINING_ROW_V4_SCHEMA,
            "synthetic_realization_id": row["synthetic_realization_id"],
            "array_receipts": row["array_receipts"],
            "finite_psf_sha256": contract["finite_psf_sha256"],
            "slab_observation_v4_receipt_sha256": slab_receipt,
        }
    )
    row["receipt_sha256"] = acquisition_v2._payload_sha256(
        psf_v4.training_row_receipt_v4(row)
    )
    psf_v4.verify_training_row_v4(
        row, capability=psf_v4.finite_psf_model_capability_v4()
    )
    return row


def _initialize(path, render_mode="finite_boxcar"):
    binding = _binding(render_mode)
    manifest = cache_v4.initialize_training_row_cache_v4(
        path, generator_binding=binding
    )
    return binding, manifest


def test_binding_and_manifest_freeze_exact_v4_capability_and_full_inputs(i_cache_root):
    binding, manifest = _initialize(i_cache_root)
    assert cache_v4.verify_generator_binding_v4(binding)
    assert manifest["schema_version"] == cache_v4.ROW_CACHE_V4_SCHEMA
    assert manifest["training_row_schema_version"] == psf_v4.TRAINING_ROW_V4_SCHEMA
    assert manifest["finite_psf_capability_sha256"] == (
        cache_v4.EXPECTED_FINITE_PSF_CAPABILITY_SHA256
    )
    assert manifest["generation_config"] == binding["generation_config"]
    assert manifest["seed_record"] == binding["seed_record"]
    assert manifest["generation_lineage"] == binding["generation_lineage"]
    assert manifest["finite_psf_run_contract"]["render_mode"] == "finite_boxcar"
    assert manifest["finite_psf_run_contract"]["axial_sample_count"] == 9
    with pytest.raises(ValueError, match="only on I"):
        cache_v4.initialize_training_row_cache_v4(
            Path("C:/finite-cache-v4-forbidden"), generator_binding=binding
        )


def test_append_load_audit_freeze_preserve_ordered_psf_slab_and_thickness(i_cache_root):
    binding, _ = _initialize(i_cache_root)
    rows = [_row(0, 25.0), _row(1, 100.0)]
    manifest = cache_v4.append_training_rows_v4(i_cache_root, rows)
    assert manifest["row_count"] == 2
    assert [record["finite_psf_sha256"] for record in manifest["rows"]] == [
        row["finite_psf_contract"]["finite_psf_sha256"] for row in rows
    ]
    assert [
        record["slab_observation_v4_receipt_sha256"]
        for record in manifest["rows"]
    ] == [
        row["finite_psf_contract"]["slab_observation_v4_receipt_sha256"]
        for row in rows
    ]
    assert [record["nominal_cut_thickness_um"] for record in manifest["rows"]] == [
        25.0,
        100.0,
    ]
    loaded = cache_v4.load_training_rows_v4(i_cache_root, indices=[1, 0])
    assert [row["training_row_id"] for row in loaded] == [
        rows[1]["training_row_id"],
        rows[0]["training_row_id"],
    ]
    audit = cache_v4.audit_training_row_cache_v4(i_cache_root)
    assert audit["all_rows_authenticated"]
    assert audit["row_count"] == 2
    assert audit["nominal_cut_thickness_um_min"] == 25.0
    assert audit["nominal_cut_thickness_um_max"] == 100.0
    frozen = cache_v4.freeze_training_row_cache_v4(i_cache_root)
    assert frozen["status"] == cache_v4.FROZEN_CACHE_STATUS
    assert frozen["freeze_audit"]["ordered_finite_psf_sha256"] == (
        audit["ordered_finite_psf_sha256"]
    )
    assert cache_v4.load_training_row_cache_manifest_v4(
        i_cache_root,
        expected_generator_binding=binding,
        expected_receipt_sha256=frozen["receipt_sha256"],
    ) == frozen
    with pytest.raises(ValueError, match="cannot be appended"):
        cache_v4.append_training_rows_v4(i_cache_root, [_row(2, 50.0)])


def test_append_is_idempotent_and_rejects_forged_reused_identity(i_cache_root):
    _initialize(i_cache_root)
    row = _row(0, 50.0)
    first = cache_v4.append_training_rows_v4(i_cache_root, [row])
    second = cache_v4.append_training_rows_v4(i_cache_root, [row])
    assert second == first
    changed = _row(1, 75.0)
    changed["training_row_id"] = row["training_row_id"]
    changed["receipt_sha256"] = acquisition_v2._payload_sha256(
        psf_v4.training_row_receipt_v4(changed)
    )
    with pytest.raises(ValueError, match="identity differs from its exact inputs"):
        cache_v4.append_training_rows_v4(i_cache_root, [changed])


def test_batch_preflight_rejects_v3_and_mixed_s9_s1_without_partial_append(i_cache_root):
    _initialize(i_cache_root)
    finite = _row(0, 25.0)
    ablation = _row(1, 0.0, "centre_plane_ablation")
    with pytest.raises(ValueError, match="mixed finite-PSF"):
        cache_v4.append_training_rows_v4(i_cache_root, [finite, ablation])
    assert cache_v4.load_training_row_cache_manifest_v4(i_cache_root)["row_count"] == 0
    legacy = copy.deepcopy(finite)
    legacy["schema_version"] = "anatomy-tracker.arbitrary-plane-training-row/v3"
    with pytest.raises(ValueError, match="only authenticated training-row/v4"):
        cache_v4.append_training_rows_v4(i_cache_root, [legacy])
    assert cache_v4.load_training_row_cache_manifest_v4(i_cache_root)["row_count"] == 0


def test_centre_plane_ablation_is_separate_s1_cache(i_cache_root):
    _initialize(i_cache_root, "centre_plane_ablation")
    ablation = _row(0, 0.0, "centre_plane_ablation")
    manifest = cache_v4.append_training_rows_v4(i_cache_root, [ablation])
    assert manifest["finite_psf_run_contract"]["axial_sample_count"] == 1
    assert manifest["rows"][0]["nominal_cut_thickness_um"] == 0.0
    with pytest.raises(ValueError, match="mixed finite-PSF"):
        cache_v4.append_training_rows_v4(i_cache_root, [_row(1, 25.0)])


def test_freeze_and_append_enforce_declared_generation_row_count(i_cache_root):
    _initialize(i_cache_root)
    cache_v4.append_training_rows_v4(i_cache_root, [_row(0)])
    with pytest.raises(ValueError, match="declared row count"):
        cache_v4.freeze_training_row_cache_v4(i_cache_root)
    cache_v4.append_training_rows_v4(i_cache_root, [_row(1, 50.0)])
    with pytest.raises(ValueError, match="exceed"):
        cache_v4.append_training_rows_v4(i_cache_root, [_row(2, 75.0)])


def test_learned_dependency_and_non_development_lineage_are_rejected(i_cache_root):
    with pytest.raises(ValueError, match="learned dependencies"):
        cache_v4.make_generator_binding_v4(
            generator_ids=["bad-generator"],
            source_sha256={"bad.py": _digest("bad-source")},
            geometry_gauge_contract=(
                deformation_gauge_v4.direct_deformation_target_contract_v4()
            ),
            generation_config={
                "schema_version": "bad/v4",
                "algorithm": "bad-generator",
                "row_count": 1,
                "prior_model_weight_dependencies": ["legacy.pt"],
            },
            seed_record={"seed": 1},
            generation_lineage=_generation_lineage(),
            finite_psf_run_contract=_run_contract(),
        )
    _initialize(i_cache_root)
    row = _row(0)
    row["lineage"]["split"] = "final-test"
    row["receipt_sha256"] = acquisition_v2._payload_sha256(
        psf_v4.training_row_receipt_v4(row)
    )
    with pytest.raises(ValueError, match="development splits"):
        cache_v4.append_training_rows_v4(i_cache_root, [row])


def test_crash_before_manifest_commit_resumes_from_same_authenticated_row(
    i_cache_root, monkeypatch
):
    _initialize(i_cache_root)
    row = _row(0, 55.0)
    original_atomic_json = cache_v4._atomic_json

    def fail_manifest(path, value):
        if Path(path).name == "manifest.json" and value.get("row_count") == 1:
            raise OSError("simulated manifest commit interruption")
        return original_atomic_json(path, value)

    monkeypatch.setattr(cache_v4, "_atomic_json", fail_manifest)
    with pytest.raises(OSError, match="simulated"):
        cache_v4.append_training_rows_v4(i_cache_root, [row])
    assert cache_v4.load_training_row_cache_manifest_v4(i_cache_root)["row_count"] == 0
    monkeypatch.setattr(cache_v4, "_atomic_json", original_atomic_json)
    resumed = cache_v4.append_training_rows_v4(i_cache_root, [row])
    assert resumed["row_count"] == 1
    assert cache_v4.audit_training_row_cache_v4(i_cache_root)[
        "all_rows_authenticated"
    ]


@pytest.mark.parametrize("target", ["metadata", "arrays"])
def test_row_file_tampering_is_detected(i_cache_root, target):
    _initialize(i_cache_root)
    manifest = cache_v4.append_training_rows_v4(i_cache_root, [_row(0)])
    record = manifest["rows"][0]
    path = i_cache_root / record[f"{target}_relative_path"]
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="file hash differs"):
        cache_v4.load_training_rows_v4(i_cache_root)


def test_rehashed_manifest_cannot_mix_run_contract_or_ordered_row_metadata(i_cache_root):
    _initialize(i_cache_root)
    cache_v4.append_training_rows_v4(i_cache_root, [_row(0, 25.0)])
    manifest_path = i_cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["finite_psf_run_contract"]["axial_sample_count"] = 1
    manifest["receipt_sha256"] = cache_v4._hash_json(
        cache_v4._manifest_payload(manifest)
    )
    cache_v4._atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="cache-run contract changed"):
        cache_v4.load_training_row_cache_manifest_v4(i_cache_root)


def test_unrecorded_final_row_file_fails_audit_but_tmp_is_reported(i_cache_root):
    _initialize(i_cache_root)
    cache_v4.append_training_rows_v4(i_cache_root, [_row(0)])
    temporary = i_cache_root / "rows" / "interrupted.json.tmp"
    temporary.write_text("partial", encoding="utf-8")
    assert cache_v4.audit_training_row_cache_v4(i_cache_root)[
        "temporary_file_count"
    ] == 1
    orphan = i_cache_root / "rows" / "orphan.json"
    orphan.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unrecorded finalized"):
        cache_v4.audit_training_row_cache_v4(i_cache_root)
