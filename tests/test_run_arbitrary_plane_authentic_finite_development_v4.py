import copy
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

import training.arbitrary_plane_catalogue_v3 as catalogue_v3
import training.arbitrary_plane_finite_composite_v4 as composite_v4
import training.arbitrary_plane_finite_training_runner_v4 as runner_v4
import training.arbitrary_plane_row_cache_v4 as row_cache_v4
import training.run_arbitrary_plane_authentic_finite_development_v4 as run
from training.arbitrary_plane_rendered_generator import prepare_finite_render_context
from training.arbitrary_plane_support import build_annotation_support_index


def test_smoke_and_pilot_profiles_are_exact_s9_i_only_and_learned_independent():
    assert all(
        path.drive.upper() == "I:"
        for path in (
            run.ANATOMY_ROOT,
            run.REPOSITORY_ROOT,
            run.TEMPLATE_PATH,
            run.ANNOTATION_PATH,
        )
    )
    for profile, expected_counts in (
        ("smoke", (12, 12, 6, 6, 24)),
        ("pilot", (3072, 2048, 384, 256, 8000)),
    ):
        config = run.finite_development_configuration_v4(profile)
        assert run.verify_finite_development_configuration_v4(config)
        assert config["render_mode"] == "finite_boxcar"
        assert config["axial_sample_count"] == 9
        assert config["finite_psf_capability_sha256"] == (
            row_cache_v4.EXPECTED_FINITE_PSF_CAPABILITY_SHA256
        )
        assert config["nominal_cut_thickness_policy"] == {
            "kind": "authenticated-per-row-closed-interval",
            "minimum_um": 25.0,
            "maximum_um": 100.0,
        }
        assert config["runner_config"]["axial_offsets_um"] == []
        assert config["runner_config"]["axial_weights"] == []
        assert config["all_brain_intersecting_plane_domain"]
        assert config["marginal_or_empty_rows_retained"]
        assert not config["smart_brush_required"]
        assert set(config["trainable_input_modes"]) == {
            "smart-brush-accurate",
            "smart-brush-imperfect",
            "smart-brush-absent",
        }
        assert all(
            Path(config[name]).drive.upper() == "I:"
            for name in (
                "output_root",
                "temp_root",
                "training_cache",
                "internal_development_cache",
                "training_run",
                "configuration_snapshot",
                "partition_audit",
            )
        )
        train = config["partitions"]["training"]
        development = config["partitions"]["internal_development"]
        assert (
            train["pose_row_count"],
            train["joint_row_count"],
            development["pose_row_count"],
            development["joint_row_count"],
            config["runner_config"]["target_applied_steps"],
        ) == expected_counts
        assert train["identity_prefix"] != development["identity_prefix"]
        assert train["pose_root_seed_uint64"] != development["pose_root_seed_uint64"]
        assert train["joint_root_seed_uint64"] != development["joint_root_seed_uint64"]
        assert config["prior_model_weight_dependencies"] == []
        assert config["prior_feature_dependencies"] == []
        assert config["prior_pseudolabel_dependencies"] == []
        if profile == "pilot":
            assert config["runner_config"]["batch_size"] == 2
            assert config["training_config"]["pose_warmup_steps"] == 2000
            assert config["training_config"]["amp_initial_scale"] == 256.0
            assert (
                config["runner_config"]["target_applied_steps"]
                * config["runner_config"]["batch_size"]
                == 16000
            )


def test_s1_ablation_is_separately_named_and_cannot_enter_s9_orchestrator():
    ablation = run.zero_thickness_ablation_configuration_v4()
    assert run.verify_zero_thickness_ablation_configuration_v4(ablation)
    assert ablation["render_mode"] == "centre_plane_ablation"
    assert ablation["nominal_cut_thickness_um"] == 0.0
    assert ablation["axial_sample_count"] == 1
    assert ablation["finite_psf_capability_sha256"] == (
        row_cache_v4.EXPECTED_FINITE_PSF_CAPABILITY_SHA256
    )
    assert not ablation["finite_development_orchestrator_compatible"]
    assert ablation["must_never_share_cache_or_run_with_s9"]
    with pytest.raises(ValueError, match="separately named ablation"):
        run.finite_development_configuration_v4("zero-thickness")
    changed = copy.deepcopy(run.FINITE_SMOKE_CONFIGURATION_V4)
    changed["render_mode"] = "centre_plane_ablation"
    changed["axial_sample_count"] = 1
    with pytest.raises(ValueError, match="differs from its frozen profile"):
        run.verify_finite_development_configuration_v4(changed)


def _manifest(config, partition_name):
    declaration = config["partitions"][partition_name]
    count = declaration["pose_row_count"] + declaration["joint_row_count"]
    modes = config["trainable_input_modes"]
    records = []
    for index in range(count):
        animal = index // config["sections_per_animal"]
        prefix = declaration["identity_prefix"]
        records.append(
            {
                "selected_mode": modes[index % len(modes)],
                "lineage": {
                    "animal_id": f"{prefix}-animal-{animal}",
                    "specimen_id": f"{prefix}-specimen-{animal}",
                    "experiment_id": f"{prefix}-experiment-{animal}",
                    "synthetic_animal_id": f"{prefix}-synthetic-animal-{animal}",
                    "section_id": f"{prefix}-section-{index}",
                    "split": declaration["split"],
                },
            }
        )
    return {
        "status": "FROZEN",
        "row_count": count,
        "rows": records,
        "receipt_sha256": f"{partition_name:0<64}"[:64],
        "generator_binding": {"receipt_sha256": f"binding-{partition_name:0<64}"[:64]},
        "finite_psf_capability_sha256": (
            row_cache_v4.EXPECTED_FINITE_PSF_CAPABILITY_SHA256
        ),
        "finite_psf_run_contract": (
            row_cache_v4.make_finite_psf_cache_run_contract_v4("finite_boxcar")
        ),
        "generation_lineage": {
            "generation_run_id": declaration["generation_run_id"],
            "split": declaration["split"],
            "source_commit": "1" * 40,
        },
    }


def _cache_audit(manifest):
    return {
        "row_count": manifest["row_count"],
        "receipt_sha256": "a" * 64,
        "all_rows_authenticated": True,
        "learned_dependencies": [],
    }


def test_partition_audit_proves_no_drop_mode_coverage_and_animal_disjointness():
    config = run.FINITE_SMOKE_CONFIGURATION_V4
    manifests = {
        name: _manifest(config, name)
        for name in ("training", "internal_development")
    }
    audits = {name: _cache_audit(manifest) for name, manifest in manifests.items()}
    report = run._audit_partition_pair(config, manifests, audits)
    assert report["all_partition_ids_disjoint"]
    assert report["all_marginal_or_empty_rows_retained"]
    assert report["partitions"]["training"]["all_logical_rows_retained"]
    assert report["partitions"]["internal_development"][
        "all_logical_rows_retained"
    ]
    assert all(not values for values in report["cross_partition_identity_overlap"].values())
    assert sum(report["partitions"]["training"]["mode_counts"].values()) == 24

    lost = copy.deepcopy(manifests)
    lost["training"]["rows"].pop()
    lost["training"]["row_count"] -= 1
    with pytest.raises(RuntimeError, match="count, mode, PSF, or provenance"):
        run._audit_partition_pair(config, lost, audits)

    overlapping = copy.deepcopy(manifests)
    for name in (
        "animal_id",
        "specimen_id",
        "experiment_id",
        "synthetic_animal_id",
        "section_id",
    ):
        overlapping["internal_development"]["rows"][0]["lineage"][name] = (
            manifests["training"]["rows"][0]["lineage"][name]
        )
    with pytest.raises(RuntimeError, match="lineage IDs overlap"):
        run._audit_partition_pair(config, overlapping, audits)


def test_i_only_atomic_snapshot_is_crash_resumable_and_exact():
    root = Path("I:/AnatomyTracker/test_tmp/finite_development_orchestrator_v4")
    path = root / uuid.uuid4().hex / "snapshot.json"
    try:
        path.parent.mkdir(parents=True)
        path.with_name(path.name + ".tmp").write_text("interrupted", encoding="utf-8")
        run._write_or_verify_json(path, {"value": 1})
        assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}
        run._write_or_verify_json(path, {"value": 1})
        with pytest.raises(RuntimeError, match="differs from frozen inputs"):
            run._write_or_verify_json(path, {"value": 2})
        with pytest.raises(ValueError, match="only on I"):
            run._write_or_verify_json(Path("C:/forbidden-finite-v4.json"), {})
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)


def test_real_finite_curricula_cache_resume_and_runner_initialization_integrate():
    root = Path("I:/AnatomyTracker/test_tmp/finite_development_orchestrator_v4") / uuid.uuid4().hex
    try:
        annotation = np.zeros((17, 15, 13), dtype=np.uint16)
        annotation[2:15, 3:13, 1:11] = 7
        annotation[6:11, 6:10, 4:8] = 19
        ap, dv, ml = np.indices(annotation.shape)
        template = (100 + 3 * ap + 5 * dv + 7 * ml).astype(np.uint16)
        support = build_annotation_support_index(
            annotation,
            atlas_id="finite-orchestrator-fixture",
            atlas_version="fixture-v1",
            source_uri="file:///finite-orchestrator-annotation.nrrd",
            source_sha256="3" * 64,
            source_entity_type="atlas-annotation",
            voxel_size_um=(11.0, 17.0, 29.0),
            origin_um=(-71.0, 23.0, 107.0),
            coordinate_axis_directions=("posterior", "inferior", "right"),
        )
        prepared = prepare_finite_render_context(
            template,
            annotation,
            support,
            scalar_source_uri="file:///finite-orchestrator-template.nrrd",
            scalar_source_sha256="4" * 64,
            template_decoder="fixture",
            template_index_order="F",
            annotation_decoder="fixture",
            annotation_index_order="F",
        )
        config = copy.deepcopy(run.FINITE_SMOKE_CONFIGURATION_V4)
        config["output_shape_h_w"] = [33, 35]
        config["sections_per_animal"] = 3
        partition = config["partitions"]["training"]
        partition["pose_row_count"] = 3
        partition["joint_row_count"] = 3
        _, binding = run._partition_generation(
            prepared, config, "training", "5" * 40
        )
        assert row_cache_v4.verify_generator_binding_v4(binding)
        assert binding["finite_psf_run_contract"] == (
            row_cache_v4.make_finite_psf_cache_run_contract_v4("finite_boxcar")
        )
        assert binding["generation_lineage"]["source_commit"] == "5" * 40
        cache = root / "cache"
        manifest, audit = composite_v4.resume_finite_composite_cache_v4(
            cache, prepared, binding, chunk_size=2
        )
        resumed, resumed_audit = composite_v4.resume_finite_composite_cache_v4(
            cache, prepared, binding, chunk_size=2
        )
        assert resumed == manifest
        assert resumed_audit == audit
        assert manifest["row_count"] == 6
        assert {record["selected_mode"] for record in manifest["rows"]} == set(
            config["trainable_input_modes"]
        )
        assert manifest["finite_psf_run_contract"]["axial_sample_count"] == 9

        catalogue = catalogue_v3.make_arbitrary_plane_catalogue_v3(
            None,
            (-71.0, 23.0, 107.0),
            (11.0, 17.0, 29.0),
            support_index=support,
            normal_count=4,
            offset_count=2,
            roll_count=2,
            raster_shape_h_w=(33, 35),
            raster_physical_span_y_x_um=(400.0, 600.0),
        )
        intensity = template.astype(np.float32)
        intensity = (intensity - intensity[annotation != 0].min()) / (
            intensity[annotation != 0].max() - intensity[annotation != 0].min()
        )
        intensity[annotation == 0] = 0.0
        atlas = np.stack((intensity, (annotation != 0).astype(np.float32)), axis=0)
        root.mkdir(parents=True, exist_ok=True)
        source = root / "allen-fixture.bin"
        source.write_bytes(b"finite orchestrator authenticated atlas fixture")
        runner_config = {
            "target_applied_steps": 2,
            "batch_size": 2,
            "candidate_bank_size": 15,
            "row_selection_seed": "0x0000000000000701",
            "candidate_bank_root_seed": "0x0000000000000702",
            "axial_offsets_um": [],
            "axial_weights": [],
            "archive_checkpoint_interval_applied_steps": 1,
            "checkpoint_commit_interval_attempts": 1,
        }
        training_config = {
            "seed": 17,
            "pose_warmup_steps": 1,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "top_k": 2,
            "refinement_steps": 1,
            "joint_pose_only_steps": 1,
            "retrieval_shape_h_w": (16, 16),
            "catalogue_chunk_size": 16,
            "amp": False,
            "amp_initial_scale": 65536.0,
            "gradient_clip_norm": 5.0,
        }
        model_kwargs = copy.deepcopy(run.MODEL_KWARGS)
        model_kwargs["feature_channels"] = 8
        model_kwargs["hidden_channels"] = 16
        run_manifest, run_state = runner_v4.initialize_finite_training_run_v4(
            root / "run",
            cache_directory=cache,
            expected_generator_binding=binding,
            catalogue=catalogue,
            atlas_volume=atlas,
            atlas_source_assets=[
                {
                    "path": str(source),
                    "role": "authenticated fixture",
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            ],
            atlas_preprocessing={"normalization": "deterministic fixture"},
            model_kwargs=model_kwargs,
            training_config=training_config,
            runner_config=runner_config,
            device="cpu",
        )
        assert run_manifest["finite_psf_training_schedule_source"][
            "finite_psf_cache_run_contract"
        ]["axial_sample_count"] == 9
        assert run_manifest["cache"]["manifest_receipt_sha256"] == manifest[
            "receipt_sha256"
        ]
        assert run_manifest["prior_model_weight_dependencies"] == []
        assert run_state["applied_step_count"] == 0
        loaded = runner_v4.load_finite_training_run_v4(root / "run")
        assert loaded["run_state"] == run_state
    finally:
        shutil.rmtree(root, ignore_errors=True)
