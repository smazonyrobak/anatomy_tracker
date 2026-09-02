"""Authenticated finite-thickness arbitrary-plane development orchestration.

This module prepares two synthetic-development profiles and never accesses a
public benchmark, external-validation animal, or final-test animal.  Every
cache, temporary file, checkpoint, and report is confined to I:\\AnatomyTracker.
The production profile is finite boxcar S=9.  The zero-thickness S=1 ablation
has a separately named configuration and is never admitted to these caches.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path


ANATOMY_ROOT = Path(r"I:\AnatomyTracker")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ATLAS_ROOT = ANATOMY_ROOT / "data" / "Allen Brain Atlas 25um"
TEMPLATE_PATH = ATLAS_ROOT / "average_template_25.nrrd"
ANNOTATION_PATH = ATLAS_ROOT / "annotation_25.nrrd"
TEMPLATE_SHA256 = "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b"
ANNOTATION_SHA256 = "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42"
PYNRRD_VERSION = "1.1.3"

FINITE_DEVELOPMENT_CONFIG_V4_SCHEMA = (
    "anatomy-tracker.authentic-finite-development-config/v4"
)
FINITE_DEVELOPMENT_PARTITION_AUDIT_V4_SCHEMA = (
    "anatomy-tracker.authentic-finite-development-partition-audit/v4"
)
ZERO_THICKNESS_ABLATION_CONFIG_V4_SCHEMA = (
    "anatomy-tracker.zero-thickness-ablation-config/v4"
)
FINITE_RENDER_MODE = "finite_boxcar"
FINITE_AXIAL_SAMPLE_COUNT = 9
FINITE_PSF_CAPABILITY_SHA256 = (
    "bcd6441a685e902fb5b59e85bb7003ef3261207d906a0b9390d4a219c3ae3d3e"
)
ZERO_THICKNESS_RENDER_MODE = "centre_plane_ablation"
ZERO_THICKNESS_AXIAL_SAMPLE_COUNT = 1

MODEL_KWARGS = {
    "atlas_channels": 2,
    "feature_channels": 32,
    "hidden_channels": 64,
    "correlation_radius": 2,
    "update_limits": (0.18, 0.18, 600.0, 0.18, 600.0, 600.0, 0.12, 0.12, 0.12),
    "plane_tangent_scales": (0.18, 0.18, 600.0),
    "max_velocity_fraction_yx": (0.08, 0.08),
    "deformation_integration_steps": 7,
    "deformation_support_floor": 1.0e-4,
    "deformation_maximum_velocity_gradient": 0.35,
}


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
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


def _with_receipt(payload):
    payload = _plain(payload)
    return {**payload, "receipt_sha256": _hash_json(payload)}


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _i_path(path):
    resolved = Path(path).resolve()
    if os.path.splitdrive(str(resolved))[0].upper() != "I:":
        raise ValueError("finite development artifacts must be stored only on I:")
    return resolved


def _atomic_json(path, value):
    target = _i_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical_json(value))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def _make_configuration(
    profile,
    *,
    output_shape_h_w,
    sections_per_animal,
    train_pose_rows,
    train_joint_rows,
    development_pose_rows,
    development_joint_rows,
    cache_chunk_size,
    catalogue_config,
    training_config,
    runner_config,
):
    output_root = ANATOMY_ROOT / "runs" / f"arbitrary_plane_finite_v4_{profile}_001"
    payload = {
        "schema_version": FINITE_DEVELOPMENT_CONFIG_V4_SCHEMA,
        "profile": profile,
        "scientific_scope": (
            "internal synthetic development only; not validation, qualification, "
            "benchmarking, calibration, external validation, or final test"
        ),
        "render_mode": FINITE_RENDER_MODE,
        "axial_sample_count": FINITE_AXIAL_SAMPLE_COUNT,
        "finite_psf_capability_sha256": FINITE_PSF_CAPABILITY_SHA256,
        "nominal_cut_thickness_policy": {
            "kind": "authenticated-per-row-closed-interval",
            "minimum_um": 25.0,
            "maximum_um": 100.0,
        },
        "output_root": str(output_root),
        "temp_root": str(ANATOMY_ROOT / "tmp" / output_root.name),
        "training_cache": str(output_root / "training_cache"),
        "internal_development_cache": str(
            output_root / "internal_development_cache"
        ),
        "training_run": str(output_root / "training_run"),
        "configuration_snapshot": str(output_root / "configuration_v4.json"),
        "partition_audit": str(output_root / "partition_audit_v4.json"),
        "output_shape_h_w": list(output_shape_h_w),
        "sections_per_animal": int(sections_per_animal),
        "cache_chunk_size": int(cache_chunk_size),
        "partitions": {
            "training": {
                "split": "train",
                "identity_prefix": f"finite-v4-{profile}-train",
                "generation_run_id": f"finite-v4-{profile}-train-001",
                "pose_row_count": int(train_pose_rows),
                "joint_row_count": int(train_joint_rows),
                "pose_root_seed_uint64": "0x2026090200001001",
                "joint_root_seed_uint64": "0x2026090200001002",
            },
            "internal_development": {
                "split": "development",
                "identity_prefix": f"finite-v4-{profile}-internal-development",
                "generation_run_id": (
                    f"finite-v4-{profile}-internal-development-001"
                ),
                "pose_row_count": int(development_pose_rows),
                "joint_row_count": int(development_joint_rows),
                "pose_root_seed_uint64": "0x2026090200001003",
                "joint_root_seed_uint64": "0x2026090200001004",
            },
        },
        "catalogue_config": copy.deepcopy(catalogue_config),
        "model_kwargs": _plain(MODEL_KWARGS),
        "training_config": copy.deepcopy(training_config),
        "runner_config": copy.deepcopy(runner_config),
        "atlas_source_assets": {
            "template": {
                "path": str(TEMPLATE_PATH),
                "sha256": TEMPLATE_SHA256,
            },
            "annotation": {
                "path": str(ANNOTATION_PATH),
                "sha256": ANNOTATION_SHA256,
            },
            "decoder": {"package": "pynrrd", "version": PYNRRD_VERSION},
        },
        "all_brain_intersecting_plane_domain": True,
        "marginal_or_empty_rows_retained": True,
        "smart_brush_required": False,
        "catalogue_capture_audit_status": (
            "separate namespaced v4 follow-up; the v3 cache audit is never reinterpreted"
        ),
        "trainable_input_modes": [
            "smart-brush-accurate",
            "smart-brush-imperfect",
            "smart-brush-absent",
        ],
        "input_mode_semantics": {
            "smart-brush-accurate": "exact black exterior",
            "smart-brush-imperfect": "imperfect user mask",
            "smart-brush-absent": "unmasked acquired/raw background",
        },
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
        "public_benchmark_accessed": False,
        "external_validation_accessed": False,
        "final_test_accessed": False,
    }
    return _with_receipt(payload)


FINITE_SMOKE_CONFIGURATION_V4 = _make_configuration(
    "smoke",
    output_shape_h_w=(96, 96),
    sections_per_animal=6,
    train_pose_rows=12,
    train_joint_rows=12,
    development_pose_rows=6,
    development_joint_rows=6,
    cache_chunk_size=3,
    catalogue_config={
        "normal_count": 24,
        "offset_count": 4,
        "roll_count": 4,
        "raster_shape_h_w": [96, 96],
        "raster_physical_span_y_x_um": [12000.0, 12000.0],
    },
    training_config={
        "seed": 20260902,
        "pose_warmup_steps": 8,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "top_k": 4,
        "refinement_steps": 3,
        "joint_pose_only_steps": 2,
        "retrieval_shape_h_w": [48, 48],
        "catalogue_chunk_size": 128,
        "amp": True,
        "amp_initial_scale": 65536.0,
        "gradient_clip_norm": 5.0,
    },
    runner_config={
        "target_applied_steps": 24,
        "batch_size": 2,
        "candidate_bank_size": 64,
        "row_selection_seed": "0x2026090200001101",
        "candidate_bank_root_seed": "0x2026090200001102",
        "axial_offsets_um": [],
        "axial_weights": [],
        "archive_checkpoint_interval_applied_steps": 12,
        "checkpoint_commit_interval_attempts": 4,
    },
)

FINITE_PILOT_CONFIGURATION_V4 = _make_configuration(
    "pilot",
    output_shape_h_w=(160, 160),
    sections_per_animal=16,
    train_pose_rows=3072,
    train_joint_rows=2048,
    development_pose_rows=384,
    development_joint_rows=256,
    cache_chunk_size=16,
    catalogue_config={
        "normal_count": 384,
        "offset_count": 16,
        "roll_count": 16,
        "raster_shape_h_w": [160, 160],
        "raster_physical_span_y_x_um": [12000.0, 12000.0],
    },
    training_config={
        "seed": 20260902,
        "pose_warmup_steps": 2000,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "top_k": 4,
        "refinement_steps": 3,
        "joint_pose_only_steps": 2,
        "retrieval_shape_h_w": [48, 48],
        "catalogue_chunk_size": 512,
        "amp": True,
        "amp_initial_scale": 256.0,
        "gradient_clip_norm": 5.0,
    },
    runner_config={
        "target_applied_steps": 8000,
        "batch_size": 2,
        "candidate_bank_size": 512,
        "row_selection_seed": "0x2026090200001201",
        "candidate_bank_root_seed": "0x2026090200001202",
        "axial_offsets_um": [],
        "axial_weights": [],
        "archive_checkpoint_interval_applied_steps": 500,
        "checkpoint_commit_interval_attempts": 50,
    },
)

ZERO_THICKNESS_ABLATION_CONFIGURATION_V4 = _with_receipt(
    {
        "schema_version": ZERO_THICKNESS_ABLATION_CONFIG_V4_SCHEMA,
        "name": "separate-zero-thickness-s1-ablation",
        "render_mode": ZERO_THICKNESS_RENDER_MODE,
        "nominal_cut_thickness_um": 0.0,
        "axial_sample_count": ZERO_THICKNESS_AXIAL_SAMPLE_COUNT,
        "finite_psf_capability_sha256": FINITE_PSF_CAPABILITY_SHA256,
        "finite_development_orchestrator_compatible": False,
        "reference_entrypoint": (
            "training.run_arbitrary_plane_authentic_development_v3:main"
        ),
        "separate_output_root_required": True,
        "must_never_share_cache_or_run_with_s9": True,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
)


def finite_development_configuration_v4(profile):
    if profile == "smoke":
        return copy.deepcopy(FINITE_SMOKE_CONFIGURATION_V4)
    if profile == "pilot":
        return copy.deepcopy(FINITE_PILOT_CONFIGURATION_V4)
    if profile in ("zero-thickness", "ablation", "centre_plane_ablation"):
        raise ValueError(
            "S=1 zero thickness is a separately named ablation configuration and "
            "cannot enter the S=9 finite-development orchestrator"
        )
    raise ValueError("finite development profile must be smoke or pilot")


def zero_thickness_ablation_configuration_v4():
    return copy.deepcopy(ZERO_THICKNESS_ABLATION_CONFIGURATION_V4)


def verify_zero_thickness_ablation_configuration_v4(config):
    expected = zero_thickness_ablation_configuration_v4()
    if config != expected or config["receipt_sha256"] != _hash_json(
        {key: value for key, value in config.items() if key != "receipt_sha256"}
    ):
        raise ValueError("zero-thickness S=1 ablation configuration changed")
    return True


def verify_finite_development_configuration_v4(config):
    expected = finite_development_configuration_v4(config.get("profile"))
    if config != expected:
        raise ValueError("finite development configuration differs from its frozen profile")
    if (
        config["render_mode"] != FINITE_RENDER_MODE
        or config["axial_sample_count"] != FINITE_AXIAL_SAMPLE_COUNT
        or config["finite_psf_capability_sha256"]
        != FINITE_PSF_CAPABILITY_SHA256
        or config["runner_config"]["axial_offsets_um"] != []
        or config["runner_config"]["axial_weights"] != []
        or config["receipt_sha256"]
        != _hash_json({key: value for key, value in config.items() if key != "receipt_sha256"})
        or any(
            config[name] != []
            for name in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    ):
        raise ValueError("finite development configuration is invalid")
    for name in (
        "output_root",
        "temp_root",
        "training_cache",
        "internal_development_cache",
        "training_run",
        "configuration_snapshot",
        "partition_audit",
    ):
        _i_path(config[name])
    for path in (REPOSITORY_ROOT, TEMPLATE_PATH, ANNOTATION_PATH):
        _i_path(path)
    training = config["partitions"]["training"]
    development = config["partitions"]["internal_development"]
    if (
        training["identity_prefix"] == development["identity_prefix"]
        or training["generation_run_id"] == development["generation_run_id"]
        or training["pose_root_seed_uint64"] == development["pose_root_seed_uint64"]
        or training["joint_root_seed_uint64"] == development["joint_root_seed_uint64"]
    ):
        raise ValueError("training and internal-development identities must be disjoint")
    return True


def _repository_head():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip().lower()
    if len(value) != 40 or set(value) - set("0123456789abcdef"):
        raise RuntimeError("current repository HEAD is not a canonical Git commit")
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise RuntimeError("finite development must launch from a clean committed worktree")
    return value


def _prepare_atlas():
    import nrrd
    import numpy as np

    from training.arbitrary_plane_rendered_generator import prepare_finite_render_context
    from training.arbitrary_plane_support import (
        build_annotation_support_index,
        verify_annotation_support_index,
    )

    if (
        importlib.metadata.version("pynrrd") != PYNRRD_VERSION
        or _file_sha256(TEMPLATE_PATH) != TEMPLATE_SHA256
        or _file_sha256(ANNOTATION_PATH) != ANNOTATION_SHA256
    ):
        raise RuntimeError("pinned Allen decoder or raw source hashes differ")
    template = nrrd.read(str(TEMPLATE_PATH), index_order="F")[0]
    annotation = nrrd.read(str(ANNOTATION_PATH), index_order="F")[0]
    support_mask = annotation != 0
    observed_q01, observed_q99 = np.quantile(template[support_mask], (0.01, 0.99))
    if (float(observed_q01), float(observed_q99)) != (9.0, 273.0):
        raise RuntimeError("Allen in-support intensity quantiles differ from q01=9, q99=273")
    intensity = np.clip(
        (template.astype(np.float32) - np.float32(9.0)) / np.float32(264.0),
        np.float32(0.0),
        np.float32(1.0),
    )
    intensity[~support_mask] = np.float32(0.0)
    atlas = np.ascontiguousarray(
        np.stack((intensity, support_mask.astype(np.float32)), axis=0),
        dtype=np.float32,
    )
    if np.any(atlas[:, ~support_mask] != 0.0):
        raise RuntimeError("atlas exterior is not exact zero")
    support = build_annotation_support_index(
        annotation,
        atlas_id="Allen CCFv3",
        atlas_version="2017 25um",
        source_uri=str(ANNOTATION_PATH.resolve()),
        source_sha256=ANNOTATION_SHA256,
        source_entity_type="atlas-annotation",
        voxel_size_um=(25.0, 25.0, 25.0),
        origin_um=(0.0, 0.0, 0.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    verify_annotation_support_index(support)
    prepared = prepare_finite_render_context(
        template,
        annotation,
        support,
        scalar_source_uri=str(TEMPLATE_PATH.resolve()),
        scalar_source_sha256=TEMPLATE_SHA256,
        scalar_source_entity_type="atlas-template",
        scalar_dtype="float32",
        template_decoder="pynrrd==1.1.3 nrrd.read(index_order=F)",
        template_index_order="F",
        annotation_decoder="pynrrd==1.1.3 nrrd.read(index_order=F)",
        annotation_index_order="F",
    )
    return atlas, support, prepared


def _partition_generation(prepared, config, partition_name, source_commit):
    import training.arbitrary_plane_finite_composite_v4 as finite_composite_v4
    import training.arbitrary_plane_finite_joint_curriculum_v5 as finite_joint_v5
    import training.arbitrary_plane_finite_pose_curriculum_v4 as finite_pose_v4

    partition = config["partitions"][partition_name]
    common = {
        "start_index": 0,
        "output_shape_h_w": tuple(config["output_shape_h_w"]),
        "sections_per_animal": config["sections_per_animal"],
        "split": partition["split"],
        "finite_parent_generator_source_commit": source_commit,
        "finite_slab_generator_source_commit": source_commit,
    }
    pose_config = finite_pose_v4.finite_pose_curriculum_generation_config_v4(
        prepared,
        root_seed=partition["pose_root_seed_uint64"],
        row_count=partition["pose_row_count"],
        identity_prefix=f"{partition['identity_prefix']}-pose",
        **common,
    )
    joint_config = finite_joint_v5.finite_joint_curriculum_generation_config_v5(
        prepared,
        root_seed=partition["joint_root_seed_uint64"],
        row_count=partition["joint_row_count"],
        identity_prefix=f"{partition['identity_prefix']}-joint",
        render_mode=FINITE_RENDER_MODE,
        nominal_cut_thickness_um=None,
        **common,
    )
    composite_config = finite_composite_v4.make_finite_composite_generation_config_v4(
        pose_config, joint_config
    )
    binding = finite_composite_v4.make_finite_composite_generator_binding_v4(
        composite_config,
        generation_run_id=partition["generation_run_id"],
        source_commit=source_commit,
    )
    return composite_config, binding


def _audit_partition_pair(config, manifests, cache_audits):
    import training.arbitrary_plane_row_cache_v4 as row_cache_v4

    expected_modes = set(config["trainable_input_modes"])
    expected_run_contract = row_cache_v4.make_finite_psf_cache_run_contract_v4(
        FINITE_RENDER_MODE
    )
    identity_sets = {}
    partitions = {}
    source_commits = set()
    for partition_name in ("training", "internal_development"):
        manifest = manifests[partition_name]
        audit = cache_audits[partition_name]
        declaration = config["partitions"][partition_name]
        expected_count = declaration["pose_row_count"] + declaration["joint_row_count"]
        modes = [record["selected_mode"] for record in manifest["rows"]]
        if (
            manifest["status"] != "FROZEN"
            or manifest["row_count"] != expected_count
            or audit["row_count"] != expected_count
            or manifest["finite_psf_run_contract"]["render_mode"] != FINITE_RENDER_MODE
            or manifest["finite_psf_run_contract"]["axial_sample_count"]
            != FINITE_AXIAL_SAMPLE_COUNT
            or manifest["finite_psf_run_contract"] != expected_run_contract
            or manifest["finite_psf_capability_sha256"]
            != FINITE_PSF_CAPABILITY_SHA256
            or manifest["finite_psf_capability_sha256"]
            != row_cache_v4.EXPECTED_FINITE_PSF_CAPABILITY_SHA256
            or manifest["generation_lineage"]["generation_run_id"]
            != declaration["generation_run_id"]
            or manifest["generation_lineage"]["split"] != declaration["split"]
            or set(modes) != expected_modes
            or not audit["all_rows_authenticated"]
            or audit["learned_dependencies"] != []
        ):
            raise RuntimeError("finite cache violated count, mode, PSF, or provenance contract")
        identity_sets[partition_name] = {
            name: {record["lineage"][name] for record in manifest["rows"]}
            for name in (
                "animal_id",
                "specimen_id",
                "experiment_id",
                "synthetic_animal_id",
                "section_id",
            )
        }
        source_commits.add(manifest["generation_lineage"]["source_commit"])
        if any(
            record["lineage"]["split"] != declaration["split"]
            for record in manifest["rows"]
        ):
            raise RuntimeError("finite cache row split differs from its partition")
        partitions[partition_name] = {
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "cache_audit_receipt_sha256": audit["receipt_sha256"],
            "generator_binding_receipt_sha256": manifest["generator_binding"][
                "receipt_sha256"
            ],
            "row_count": manifest["row_count"],
            "logical_row_count": expected_count,
            "all_logical_rows_retained": manifest["row_count"] == expected_count,
            "mode_counts": {mode: modes.count(mode) for mode in sorted(expected_modes)},
            "animal_count": len(identity_sets[partition_name]["animal_id"]),
            "specimen_count": len(identity_sets[partition_name]["specimen_id"]),
            "experiment_count": len(identity_sets[partition_name]["experiment_id"]),
        }
    overlap = {
        name: sorted(
            identity_sets["training"][name]
            & identity_sets["internal_development"][name]
        )
        for name in identity_sets["training"]
    }
    if any(overlap.values()):
        raise RuntimeError("training and internal-development lineage IDs overlap")
    if len(source_commits) != 1:
        raise RuntimeError("finite cache partitions bind different source commits")
    payload = {
        "schema_version": FINITE_DEVELOPMENT_PARTITION_AUDIT_V4_SCHEMA,
        "configuration_receipt_sha256": config["receipt_sha256"],
        "render_mode": FINITE_RENDER_MODE,
        "axial_sample_count": FINITE_AXIAL_SAMPLE_COUNT,
        "finite_psf_capability_sha256": (
            row_cache_v4.EXPECTED_FINITE_PSF_CAPABILITY_SHA256
        ),
        "finite_psf_run_contract_receipt_sha256": expected_run_contract[
            "receipt_sha256"
        ],
        "repository_source_commit": next(iter(source_commits)),
        "partitions": partitions,
        "cross_partition_identity_overlap": overlap,
        "all_partition_ids_disjoint": True,
        "all_marginal_or_empty_rows_retained": True,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
        "public_benchmark_accessed": False,
        "external_validation_accessed": False,
        "final_test_accessed": False,
    }
    return _with_receipt(payload)


def _write_or_verify_json(path, value):
    target = _i_path(path)
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != _plain(value):
            raise RuntimeError("existing finite development artifact differs from frozen inputs")
    else:
        _atomic_json(target, value)


def prepare_finite_development_run_v4(config):
    """Generate/resume authenticated caches and initialize, but do not train, a run."""
    verify_finite_development_configuration_v4(config)
    expected_temp = _i_path(config["temp_root"])
    if any(
        not os.environ.get(name)
        or Path(os.environ[name]).resolve() != expected_temp
        for name in ("TEMP", "TMP")
    ):
        raise RuntimeError(f"set TEMP and TMP exactly to {expected_temp} before this run")
    source_commit = _repository_head()
    expected_temp.mkdir(parents=True, exist_ok=True)
    _i_path(config["output_root"]).mkdir(parents=True, exist_ok=True)
    _write_or_verify_json(config["configuration_snapshot"], config)

    import training.arbitrary_plane_finite_composite_v4 as finite_composite_v4
    import training.arbitrary_plane_row_cache_v4 as row_cache_v4
    from training.arbitrary_plane_catalogue_v3 import make_arbitrary_plane_catalogue_v3
    from training.arbitrary_plane_finite_training_runner_v4 import (
        initialize_finite_training_run_v4,
        load_finite_training_run_v4,
    )

    atlas, support, prepared = _prepare_atlas()
    manifests = {}
    cache_audits = {}
    bindings = {}
    for partition_name, cache_key in (
        ("training", "training_cache"),
        ("internal_development", "internal_development_cache"),
    ):
        _, binding = _partition_generation(
            prepared, config, partition_name, source_commit
        )
        manifests[partition_name], cache_audits[partition_name] = (
            finite_composite_v4.resume_finite_composite_cache_v4(
                config[cache_key],
                prepared,
                binding,
                chunk_size=config["cache_chunk_size"],
            )
        )
        bindings[partition_name] = binding
        row_cache_v4.load_training_row_cache_manifest_v4(
            config[cache_key],
            expected_generator_binding=binding,
            expected_receipt_sha256=manifests[partition_name]["receipt_sha256"],
        )
    partition_audit = _audit_partition_pair(config, manifests, cache_audits)
    _write_or_verify_json(config["partition_audit"], partition_audit)

    catalogue = make_arbitrary_plane_catalogue_v3(
        None,
        (0.0, 0.0, 0.0),
        (25.0, 25.0, 25.0),
        support_index=support,
        **{
            key: tuple(value) if key.endswith("_h_w") or key.endswith("_y_x_um") else value
            for key, value in config["catalogue_config"].items()
        },
    )
    training_run = _i_path(config["training_run"])
    if not (training_run / "run_manifest.json").exists():
        _, state = initialize_finite_training_run_v4(
            training_run,
            cache_directory=config["training_cache"],
            expected_generator_binding=bindings["training"],
            catalogue=catalogue,
            atlas_volume=atlas,
            atlas_source_assets=[
                {
                    "path": str(TEMPLATE_PATH.resolve()),
                    "role": "Allen CCFv3 2017 25um average template raw source",
                    "sha256": TEMPLATE_SHA256,
                },
                {
                    "path": str(ANNOTATION_PATH.resolve()),
                    "role": "Allen CCFv3 2017 25um annotation raw source",
                    "sha256": ANNOTATION_SHA256,
                },
            ],
            atlas_preprocessing={
                "decoder": {
                    "package": "pynrrd",
                    "version": PYNRRD_VERSION,
                    "index_order": "F",
                },
                "axis_order": "channel,AP,DV,ML",
                "dtype": "float32",
                "intensity_channel": {
                    "quantile_population": "annotation != 0",
                    "observed_q01": 9.0,
                    "observed_q99": 273.0,
                    "transform": "clip((x-9)/(273-9),0,1)",
                    "exterior": "exact float32 zero where annotation == 0",
                },
                "support_channel": "float32(annotation != 0)",
                "prior_model_weight_dependencies": [],
                "prior_feature_dependencies": [],
                "prior_pseudolabel_dependencies": [],
            },
            model_kwargs=config["model_kwargs"],
            training_config=config["training_config"],
            runner_config=config["runner_config"],
            device="cuda",
        )
    else:
        state = load_finite_training_run_v4(training_run)["run_state"]
    return {
        "configuration": copy.deepcopy(config),
        "repository_head": source_commit,
        "training_manifest": manifests["training"],
        "internal_development_manifest": manifests["internal_development"],
        "partition_audit": partition_audit,
        "training_run_state": state,
    }


def run_finite_development_v4(config):
    """Prepare the selected immutable profile and train to its frozen target."""
    from training.arbitrary_plane_finite_training_runner_v4 import (
        run_finite_training_until_target_v4,
    )

    prepared = prepare_finite_development_run_v4(config)
    prepared["training_run_state"] = run_finite_training_until_target_v4(
        config["training_run"]
    )
    return prepared


def main():
    run_finite_development_v4(FINITE_PILOT_CONFIGURATION_V4)


if __name__ == "__main__":
    main()


__all__ = [
    "ANATOMY_ROOT",
    "FINITE_AXIAL_SAMPLE_COUNT",
    "FINITE_DEVELOPMENT_CONFIG_V4_SCHEMA",
    "FINITE_PILOT_CONFIGURATION_V4",
    "FINITE_PSF_CAPABILITY_SHA256",
    "FINITE_RENDER_MODE",
    "FINITE_SMOKE_CONFIGURATION_V4",
    "ZERO_THICKNESS_ABLATION_CONFIGURATION_V4",
    "finite_development_configuration_v4",
    "prepare_finite_development_run_v4",
    "run_finite_development_v4",
    "verify_finite_development_configuration_v4",
    "verify_zero_thickness_ablation_configuration_v4",
    "zero_thickness_ablation_configuration_v4",
]
