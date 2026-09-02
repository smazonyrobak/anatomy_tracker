"""Authenticated finite-pose then finite-joint composite cache generation."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import training.arbitrary_plane_deformation_gauge_v4 as deformation_gauge_v4
import training.arbitrary_plane_finite_joint_curriculum_v5 as finite_joint_v5
import training.arbitrary_plane_finite_pose_curriculum_v4 as finite_pose_v4
import training.arbitrary_plane_row_cache_v4 as row_cache_v4


FINITE_COMPOSITE_CURRICULUM_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-finite-composite-curriculum/v4"
)
FINITE_COMPOSITE_CURRICULUM_V4_ALGORITHM = (
    "finite-pose-v4-then-finite-joint-v5-single-homogeneous-cache/v4"
)
FINITE_COMPOSITE_ROW_ORDER_POLICY = (
    "all finite identity-G1 pose rows in logical-index order, followed by all "
    "finite affine-free joint rows in logical-index order"
)
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_SELF_SOURCE = "training/arbitrary_plane_finite_composite_v4.py"


def _source_sha256():
    sources = {}
    for component_sources in (
        finite_pose_v4._source_sha256(),
        finite_joint_v5._source_sha256(),
        {
            _SELF_SOURCE: hashlib.sha256(
                (_SOURCE_ROOT / _SELF_SOURCE).read_bytes()
            ).hexdigest()
        },
    ):
        for name, digest in component_sources.items():
            if name in sources and sources[name] != digest:
                raise ValueError("finite composite component source hashes disagree")
            sources[name] = digest
    return dict(sorted(sources.items()))


def _no_learned_dependencies(config):
    return all(
        config.get(name) == []
        for name in (
            "prior_model_weight_dependencies",
            "prior_feature_dependencies",
            "prior_pseudolabel_dependencies",
        )
    )


def make_finite_composite_generation_config_v4(
    finite_pose_generation_config,
    finite_joint_generation_config,
):
    pose = copy.deepcopy(finite_pose_generation_config)
    joint = copy.deepcopy(finite_joint_generation_config)
    if (
        pose.get("schema_version")
        != finite_pose_v4.FINITE_POSE_CURRICULUM_V4_SCHEMA
        or pose.get("algorithm")
        != finite_pose_v4.FINITE_POSE_CURRICULUM_V4_ALGORITHM
        or joint.get("schema_version")
        != finite_joint_v5.FINITE_JOINT_CURRICULUM_V5_SCHEMA
        or joint.get("algorithm")
        != finite_joint_v5.FINITE_JOINT_CURRICULUM_V5_ALGORITHM
        or any(
            isinstance(config.get("row_count"), bool)
            or not isinstance(config.get("row_count"), int)
            or config["row_count"] < 1
            for config in (pose, joint)
        )
        or any(
            pose.get(name) != joint.get(name)
            for name in (
                "prepared_context_sha256",
                "support_index_sha256",
                "output_shape_h_w",
                "split",
            )
        )
        or pose.get("finite_psf_render_mode") != "finite_boxcar"
        or joint.get("render_mode") != "finite_boxcar"
        or pose.get("finite_psf_capability")
        != joint.get("finite_psf_model_capability")
        or not _no_learned_dependencies(pose)
        or not _no_learned_dependencies(joint)
    ):
        raise ValueError(
            "finite pose/joint configs are incompatible or learned-dependent"
        )
    pose_count = int(pose["row_count"])
    joint_count = int(joint["row_count"])
    return {
        "schema_version": FINITE_COMPOSITE_CURRICULUM_V4_SCHEMA,
        "algorithm": FINITE_COMPOSITE_CURRICULUM_V4_ALGORITHM,
        "row_count": pose_count + joint_count,
        "generator_ids": [
            finite_pose_v4.FINITE_POSE_CURRICULUM_V4_ALGORITHM,
            finite_joint_v5.FINITE_JOINT_CURRICULUM_V5_ALGORITHM,
        ],
        "prepared_context_sha256": pose["prepared_context_sha256"],
        "support_index_sha256": pose["support_index_sha256"],
        "output_shape_h_w": pose["output_shape_h_w"],
        "split": pose["split"],
        "finite_psf_run_contract": (
            row_cache_v4.make_finite_psf_cache_run_contract_v4(
                "finite_boxcar"
            )
        ),
        "component_generation_configs": {
            "finite_identity_pose_curriculum": pose,
            "finite_nonidentity_joint_curriculum": joint,
        },
        "component_row_counts": {
            "finite_identity_pose_curriculum": pose_count,
            "finite_nonidentity_joint_curriculum": joint_count,
        },
        "row_order_policy": FINITE_COMPOSITE_ROW_ORDER_POLICY,
        "single_frozen_cache": True,
        "marginal_or_empty_row_policy": (
            "retain every logical row exactly once; zero supervision weights remain "
            "zero and rows are never redrawn, dropped, or phase-filtered"
        ),
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }


def make_finite_composite_generator_binding_v4(
    composite_generation_config,
    *,
    generation_run_id,
    source_commit,
):
    config = copy.deepcopy(composite_generation_config)
    components = config.get("component_generation_configs", {})
    if (
        config.get("schema_version") != FINITE_COMPOSITE_CURRICULUM_V4_SCHEMA
        or config.get("algorithm") != FINITE_COMPOSITE_CURRICULUM_V4_ALGORITHM
        or set(components)
        != {
            "finite_identity_pose_curriculum",
            "finite_nonidentity_joint_curriculum",
        }
        or config
        != make_finite_composite_generation_config_v4(
            components["finite_identity_pose_curriculum"],
            components["finite_nonidentity_joint_curriculum"],
        )
    ):
        raise ValueError("finite composite generation config is invalid or stale")
    pose = components["finite_identity_pose_curriculum"]
    joint = components["finite_nonidentity_joint_curriculum"]
    seed_record = {
        "finite_pose_root_seed_uint64": pose["root_seed_uint64"],
        "finite_pose_start_index": int(pose["start_index"]),
        "finite_joint_root_seed_uint64": joint["root_seed_uint64"],
        "finite_joint_start_index": int(joint["start_index"]),
        "finite_parent_generator_source_commit": pose[
            "finite_parent_generator_source_commit"
        ],
        "finite_slab_generator_source_commit": pose[
            "finite_slab_generator_source_commit"
        ],
    }
    if (
        joint["finite_parent_generator_source_commit"]
        != seed_record["finite_parent_generator_source_commit"]
        or joint["finite_slab_generator_source_commit"]
        != seed_record["finite_slab_generator_source_commit"]
    ):
        raise ValueError("finite composite component source commits disagree")
    lineage = {
        "schema_version": row_cache_v4.GENERATION_LINEAGE_V4_SCHEMA,
        "generation_run_id": str(generation_run_id),
        "source_commit": str(source_commit).lower(),
        "split": config["split"],
    }
    return row_cache_v4.make_generator_binding_v4(
        generator_ids=config["generator_ids"],
        source_sha256=_source_sha256(),
        geometry_gauge_contract=(
            deformation_gauge_v4.direct_deformation_target_contract_v4()
        ),
        generation_config=config,
        seed_record=seed_record,
        generation_lineage=lineage,
        finite_psf_run_contract=config["finite_psf_run_contract"],
    )


def _component_rows(prepared_context, config, start_index, row_count):
    arguments = {
        "root_seed": config["root_seed_uint64"],
        "start_index": int(start_index),
        "row_count": int(row_count),
        "output_shape_h_w": tuple(config["output_shape_h_w"]),
        "identity_prefix": config["identity_prefix"],
        "sections_per_animal": config["sections_per_animal"],
        "split": config["split"],
        "stratum": config["stratum"],
        "margin_um": tuple(config["margin_u_v_um"]),
        "minimum_brain_pixels": config["minimum_brain_pixels"],
        "maximum_rejection_attempts": config["maximum_rejection_attempts"],
        "finite_parent_generator_source_commit": config[
            "finite_parent_generator_source_commit"
        ],
        "finite_slab_generator_source_commit": config[
            "finite_slab_generator_source_commit"
        ],
    }
    if config["schema_version"] == finite_pose_v4.FINITE_POSE_CURRICULUM_V4_SCHEMA:
        arguments["maximum_pose_rejection_attempts"] = config[
            "maximum_pose_rejection_attempts"
        ]
        return finite_pose_v4.make_finite_pose_curriculum_training_rows_v4(
            prepared_context, **arguments
        )
    arguments.update(
        {
            "amplitude_band_cycle": tuple(config["amplitude_band_cycle"]),
            "maximum_joint_rejection_attempts": config[
                "maximum_joint_rejection_attempts"
            ],
            "render_mode": config["render_mode"],
            "nominal_cut_thickness_um": config["nominal_cut_thickness_um"],
        }
    )
    return finite_joint_v5.make_finite_joint_curriculum_training_rows_v5(
        prepared_context, **arguments
    )


def make_finite_composite_training_rows_v4(
    prepared_context,
    composite_generation_config,
    *,
    start_index=0,
    row_count=None,
):
    config = copy.deepcopy(composite_generation_config)
    components = config.get("component_generation_configs", {})
    if config.get("schema_version") != FINITE_COMPOSITE_CURRICULUM_V4_SCHEMA:
        raise ValueError("finite composite generation config schema differs")
    total = int(config["row_count"])
    start = int(start_index)
    count = total - start if row_count is None else int(row_count)
    if start < 0 or count < 0 or start + count > total:
        raise IndexError("finite composite row range is outside its declaration")
    pose = components["finite_identity_pose_curriculum"]
    joint = components["finite_nonidentity_joint_curriculum"]
    pose_count = int(pose["row_count"])
    rows = []
    cursor = start
    stop = start + count
    if cursor < min(stop, pose_count):
        component_count = min(stop, pose_count) - cursor
        rows.extend(
            _component_rows(
                prepared_context,
                pose,
                int(pose["start_index"]) + cursor,
                component_count,
            )
        )
        cursor += component_count
    if cursor < stop:
        local = cursor - pose_count
        rows.extend(
            _component_rows(
                prepared_context,
                joint,
                int(joint["start_index"]) + local,
                stop - cursor,
            )
        )
    if len(rows) != count:
        raise RuntimeError("finite composite generator violated its no-drop row count")
    return rows


def resume_finite_composite_cache_v4(
    cache_directory,
    prepared_context,
    generator_binding,
    *,
    chunk_size=16,
):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("finite composite cache chunk size must be positive")
    row_cache_v4.verify_generator_binding_v4(generator_binding)
    config = generator_binding["generation_config"]
    if config.get("schema_version") != FINITE_COMPOSITE_CURRICULUM_V4_SCHEMA:
        raise ValueError("finite composite cache requires its exact v4 config")
    root = row_cache_v4._i_path(cache_directory)
    if not (root / "manifest.json").exists():
        row_cache_v4.initialize_training_row_cache_v4(
            root, generator_binding=generator_binding
        )
    manifest = row_cache_v4.load_training_row_cache_manifest_v4(
        root, expected_generator_binding=generator_binding
    )
    target = int(config["row_count"])
    while (
        manifest["status"] == row_cache_v4.OPEN_CACHE_STATUS
        and manifest["row_count"] < target
    ):
        start = int(manifest["row_count"])
        rows = make_finite_composite_training_rows_v4(
            prepared_context,
            config,
            start_index=start,
            row_count=min(chunk_size, target - start),
        )
        manifest = row_cache_v4.append_training_rows_v4(root, rows)
    if manifest["status"] == row_cache_v4.OPEN_CACHE_STATUS:
        manifest = row_cache_v4.freeze_training_row_cache_v4(root)
    audit = row_cache_v4.audit_training_row_cache_v4(root)
    if audit["row_count"] != target:
        raise RuntimeError("finite composite cache row count differs from declaration")
    return manifest, audit


__all__ = [
    "FINITE_COMPOSITE_CURRICULUM_V4_ALGORITHM",
    "FINITE_COMPOSITE_CURRICULUM_V4_SCHEMA",
    "FINITE_COMPOSITE_ROW_ORDER_POLICY",
    "make_finite_composite_generation_config_v4",
    "make_finite_composite_generator_binding_v4",
    "make_finite_composite_training_rows_v4",
    "resume_finite_composite_cache_v4",
]
