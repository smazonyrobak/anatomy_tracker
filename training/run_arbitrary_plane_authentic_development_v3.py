"""Authentic direct-target V4 zero-thickness development run.

Run only with TEMP and TMP set to
I:\\AnatomyTracker\\tmp\\arbitrary_plane_v4_zero_thickness_development_001.
Every generated cache, checkpoint, report, and temporary file is then confined
to I:\\AnatomyTracker.  This is synthetic development training, not a public
benchmark, qualification run, or final-test evaluation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ANATOMY_ROOT = Path(r"I:\AnatomyTracker")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ATLAS_ROOT = ANATOMY_ROOT / "data" / "Allen Brain Atlas 25um"
TEMPLATE_PATH = ATLAS_ROOT / "average_template_25.nrrd"
ANNOTATION_PATH = ATLAS_ROOT / "annotation_25.nrrd"
TEMPLATE_SHA256 = "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b"
ANNOTATION_SHA256 = "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42"
PYNRRD_VERSION = "1.1.3"

OUTPUT_ROOT = (
    ANATOMY_ROOT / "runs" / "arbitrary_plane_v4_zero_thickness_development_001"
)
TRAIN_CACHE = OUTPUT_ROOT / "training_cache_5120"
DEVELOPMENT_CACHE = OUTPUT_ROOT / "internal_development_cache_640"
TRAINING_RUN = OUTPUT_ROOT / "training_run"
TRAIN_CAPTURE_AUDIT = OUTPUT_ROOT / "training_catalogue_capture_audit.json"
DEVELOPMENT_CAPTURE_AUDIT = (
    OUTPUT_ROOT / "internal_development_catalogue_capture_audit.json"
)
TEMP_ROOT = (
    ANATOMY_ROOT / "tmp" / "arbitrary_plane_v4_zero_thickness_development_001"
)

OUTPUT_SHAPE_H_W = (160, 160)
SECTIONS_PER_ANIMAL = 16
CACHE_CHUNK_SIZE = 48
CACHE_GENERATION_WORKERS = 4
TRAIN_POSE_ROWS = 3072
TRAIN_JOINT_ROWS = 2048
DEVELOPMENT_POSE_ROWS = 384
DEVELOPMENT_JOINT_ROWS = 256

CATALOGUE_CONFIG = {
    "normal_count": 384,
    "offset_count": 16,
    "roll_count": 16,
    "raster_shape_h_w": OUTPUT_SHAPE_H_W,
    "raster_physical_span_y_x_um": (12000.0, 12000.0),
}
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
    "proposal_count": None,
    "proposal_channels": 16,
    "proposal_mixture_components": 8,
    "proposal_offset_scale_um": 10000.0,
}
TRAINING_CONFIG = {
    "seed": 20260902,
    "pose_warmup_steps": 1000,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-4,
    "top_k": 4,
    "refinement_steps": 3,
    "joint_pose_only_steps": 2,
    "retrieval_shape_h_w": (48, 48),
    "catalogue_chunk_size": 512,
    "amp": True,
    "amp_initial_scale": 65536.0,
    "gradient_clip_norm": 5.0,
}
RUNNER_CONFIG = {
    "target_applied_steps": 4000,
    "batch_size": 4,
    "candidate_bank_size": 512,
    "row_selection_seed": "0x2026090200000101",
    "candidate_bank_root_seed": "0x2026090200000102",
    "axial_offsets_um": [0.0],
    "axial_weights": [1.0],
    "archive_checkpoint_interval_applied_steps": 250,
    "checkpoint_commit_interval_attempts": 25,
}


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture_audit(
    output_path,
    cache_path,
    cache_manifest,
    catalogue,
    atlas_shape,
    capture_v3,
):
    if output_path.exists():
        report = json.loads(output_path.read_text(encoding="utf-8"))
        capture_v3.verify_catalogue_capture_audit_report_v3(report)
        if (
            report["row_cache_binding"]["manifest_receipt_sha256"]
            != cache_manifest["receipt_sha256"]
            or report["catalogue_binding"]["receipt_sha256"]
            != catalogue["receipt_sha256"]
            or report["source_sha256"] != capture_v3._source_sha256()
            or report["model_capture_contract"]["update_limits"]
            != list(MODEL_KWARGS["update_limits"])
            or report["model_capture_contract"]["refinement_steps"]
            != TRAINING_CONFIG["refinement_steps"]
        ):
            raise RuntimeError("catalogue-capture report differs from frozen inputs")
        return report
    report = capture_v3.audit_catalogue_capture_v3(
        cache_path,
        catalogue,
        atlas_shape_ap_dv_ml=atlas_shape,
        origin_ap_dv_ml_um=catalogue["support_geometry"][
            "origin_ap_dv_ml_um"
        ],
        voxel_size_ap_dv_ml_um=catalogue["support_geometry"][
            "voxel_size_ap_dv_ml_um"
        ],
        update_limits=MODEL_KWARGS["update_limits"],
        refinement_steps=TRAINING_CONFIG["refinement_steps"],
        expected_cache_manifest_receipt_sha256=cache_manifest["receipt_sha256"],
        expected_catalogue_receipt_sha256=catalogue["receipt_sha256"],
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, allow_nan=False, separators=(",", ":"), sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    return report


def _component_rows(module, prepared, config, start_index, row_count):
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
    }
    if "maximum_parent_geometry_retries" in config:
        arguments["maximum_parent_geometry_retries"] = config[
            "maximum_parent_geometry_retries"
        ]
        return module.make_pose_curriculum_training_rows_v3(prepared, **arguments)
    arguments["amplitude_band_cycle"] = tuple(config["amplitude_band_cycle"])
    arguments["maximum_joint_rejection_attempts"] = config[
        "maximum_joint_rejection_attempts"
    ]
    return module.make_joint_curriculum_training_rows_v3(prepared, **arguments)


def _ordered_component_rows(
    executor,
    module,
    prepared,
    config,
    start_index,
    row_count,
):
    if executor is None or int(row_count) == 1:
        return _component_rows(
            module,
            prepared,
            config,
            start_index,
            row_count,
        )
    futures = [
        executor.submit(
            _component_rows,
            module,
            prepared,
            config,
            int(start_index) + offset,
            1,
        )
        for offset in range(int(row_count))
    ]
    return [future.result()[0] for future in futures]


def _resume_composite_cache(
    cache_path,
    prepared,
    row_cache,
    pose_curriculum,
    joint_curriculum,
    pose_config,
    joint_config,
    composite_config,
    composite_binding,
    repository_head,
    *,
    executor=None,
):
    if not (cache_path / "manifest.json").exists():
        row_cache.initialize_training_row_cache_v3(
            cache_path,
            generator_binding=composite_binding,
            generation_config=composite_config,
            seed_record={
                "pose_root_seed_uint64": pose_config["root_seed_uint64"],
                "joint_root_seed_uint64": joint_config["root_seed_uint64"],
                "finite_parent_generator_source_commit": repository_head,
            },
        )
    manifest = row_cache.load_training_row_cache_manifest_v3(
        cache_path, expected_generator_binding=composite_binding
    )
    pose_count = int(pose_config["row_count"])
    target_count = pose_count + int(joint_config["row_count"])
    while manifest["status"] == row_cache.OPEN_CACHE_STATUS and manifest["row_count"] < target_count:
        index = int(manifest["row_count"])
        if index < pose_count:
            config = pose_config
            module = pose_curriculum
            local_index = index
            count = min(CACHE_CHUNK_SIZE, pose_count - local_index)
        else:
            config = joint_config
            module = joint_curriculum
            local_index = index - pose_count
            count = min(CACHE_CHUNK_SIZE, target_count - index)
        rows = _ordered_component_rows(
            executor,
            module,
            prepared,
            config,
            int(config["start_index"]) + local_index,
            count,
        )
        manifest = row_cache.append_training_rows_v3(cache_path, rows)
        print(
            json.dumps(
                {
                    "cache": str(cache_path),
                    "row_count": manifest["row_count"],
                    "target_row_count": target_count,
                    "manifest_receipt_sha256": manifest["receipt_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if manifest["status"] == row_cache.OPEN_CACHE_STATUS:
        manifest = row_cache.freeze_training_row_cache_v3(cache_path)
    audit = row_cache.audit_training_row_cache_v3(cache_path)
    if audit["row_count"] != target_count:
        raise RuntimeError("frozen composite cache row count differs from its declaration")
    return manifest


def main():
    expected_temp = TEMP_ROOT.resolve()
    if any(
        not os.environ.get(name)
        or Path(os.environ[name]).resolve() != expected_temp
        for name in ("TEMP", "TMP")
    ):
        raise RuntimeError(f"set TEMP and TMP exactly to {expected_temp} before this run")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    import nrrd
    import numpy as np

    from training.arbitrary_plane_catalogue_v3 import make_arbitrary_plane_catalogue_v3
    import training.arbitrary_plane_catalogue_capture_audit_v3 as capture_v3
    import training.arbitrary_plane_joint_curriculum_v3 as joint_curriculum
    import training.arbitrary_plane_pose_curriculum_v3 as pose_curriculum
    import training.arbitrary_plane_row_cache_v3 as row_cache
    from training.arbitrary_plane_rendered_generator import prepare_finite_render_context
    from training.arbitrary_plane_support import (
        build_annotation_support_index,
        verify_annotation_support_index,
    )
    from training.arbitrary_plane_training_runner_v3 import (
        initialize_training_run_v3,
        load_training_run_v3,
        run_training_until_target_v3,
    )

    repository_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    if len(repository_head) != 40 or any(character not in "0123456789abcdef" for character in repository_head):
        raise RuntimeError("current repository HEAD is not a canonical Git commit")
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
    del template, annotation, intensity, support_mask

    def configs(split, pose_rows, joint_rows, prefix, pose_seed, joint_seed):
        pose_config = pose_curriculum.pose_curriculum_generation_config_v3(
            prepared,
            root_seed=pose_seed,
            start_index=0,
            row_count=pose_rows,
            output_shape_h_w=OUTPUT_SHAPE_H_W,
            identity_prefix=f"{prefix}-pose",
            sections_per_animal=SECTIONS_PER_ANIMAL,
            split=split,
            finite_parent_generator_source_commit=repository_head,
        )
        joint_config = joint_curriculum.joint_curriculum_generation_config_v3(
            prepared,
            root_seed=joint_seed,
            start_index=0,
            row_count=joint_rows,
            output_shape_h_w=OUTPUT_SHAPE_H_W,
            identity_prefix=f"{prefix}-joint",
            sections_per_animal=SECTIONS_PER_ANIMAL,
            split=split,
            finite_parent_generator_source_commit=repository_head,
        )
        composite = joint_curriculum.composite_curriculum_generation_config_v3(
            pose_config, joint_config
        )
        return (
            pose_config,
            joint_config,
            composite,
            joint_curriculum.composite_curriculum_generator_binding_v3(composite),
        )

    train_configs = configs(
        "train",
        TRAIN_POSE_ROWS,
        TRAIN_JOINT_ROWS,
        "authv3-train",
        "0x2026090200000001",
        "0x2026090200000002",
    )
    development_configs = configs(
        "development",
        DEVELOPMENT_POSE_ROWS,
        DEVELOPMENT_JOINT_ROWS,
        "authv3-internal-development",
        "0x2026090200000003",
        "0x2026090200000004",
    )
    with ThreadPoolExecutor(
        max_workers=CACHE_GENERATION_WORKERS,
        thread_name_prefix="arbitrary-plane-cache",
    ) as cache_executor:
        train_manifest = _resume_composite_cache(
            TRAIN_CACHE,
            prepared,
            row_cache,
            pose_curriculum,
            joint_curriculum,
            *train_configs,
            repository_head,
            executor=cache_executor,
        )
        development_manifest = _resume_composite_cache(
            DEVELOPMENT_CACHE,
            prepared,
            row_cache,
            pose_curriculum,
            joint_curriculum,
            *development_configs,
            repository_head,
            executor=cache_executor,
        )
    del prepared

    if not (TRAINING_RUN / "run_manifest.json").exists():
        catalogue = make_arbitrary_plane_catalogue_v3(
            None,
            (0.0, 0.0, 0.0),
            (25.0, 25.0, 25.0),
            support_index=support,
            **CATALOGUE_CONFIG,
        )
        train_capture = _capture_audit(
            TRAIN_CAPTURE_AUDIT,
            TRAIN_CACHE,
            train_manifest,
            catalogue,
            atlas.shape[-3:],
            capture_v3,
        )
        development_capture = _capture_audit(
            DEVELOPMENT_CAPTURE_AUDIT,
            DEVELOPMENT_CACHE,
            development_manifest,
            catalogue,
            atlas.shape[-3:],
            capture_v3,
        )
        runner_config = pose_curriculum.single_plane_curriculum_runner_config_v3(
            RUNNER_CONFIG
        )
        initialize_training_run_v3(
            TRAINING_RUN,
            cache_directory=TRAIN_CACHE,
            expected_generator_binding=train_configs[3],
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
            model_kwargs=MODEL_KWARGS,
            training_config=TRAINING_CONFIG,
            runner_config=runner_config,
            device="cuda",
        )
        print(
            json.dumps(
                {
                    "training_catalogue_capture_audit": str(TRAIN_CAPTURE_AUDIT),
                    "training_catalogue_capture_receipt_sha256": train_capture[
                        "receipt_sha256"
                    ],
                    "internal_development_catalogue_capture_audit": str(
                        DEVELOPMENT_CAPTURE_AUDIT
                    ),
                    "internal_development_catalogue_capture_receipt_sha256": (
                        development_capture["receipt_sha256"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del catalogue
    else:
        load_training_run_v3(TRAINING_RUN)
    del atlas, support

    print(
        json.dumps(
            {
                "repository_head": repository_head,
                "training_cache": str(TRAIN_CACHE),
                "training_cache_receipt_sha256": train_manifest["receipt_sha256"],
                "internal_development_cache": str(DEVELOPMENT_CACHE),
                "internal_development_cache_receipt_sha256": development_manifest[
                    "receipt_sha256"
                ],
                "training_run": str(TRAINING_RUN),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    state = run_training_until_target_v3(TRAINING_RUN)
    print(
        json.dumps(
            {
                "training_run": str(TRAINING_RUN),
                "applied_step_count": state["applied_step_count"],
                "target_applied_steps": RUNNER_CONFIG["target_applied_steps"],
                "run_state_receipt_sha256": state["receipt_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
