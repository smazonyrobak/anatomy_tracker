import hashlib

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_catalogue_v3 as catalogue_v3
import training.arbitrary_plane_deformation_gauge_v3 as deformation_gauge_v3
import training.arbitrary_plane_row_cache_v3 as row_cache_v3
import training.arbitrary_plane_training_row_v3 as training_row_v3


def row(index=0, split="development"):
    height = width = 8
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    identity = np.stack((yy, xx), axis=-1)
    image = np.linspace(0.0, 1.0, height * width, dtype=np.float32).reshape(
        height, width
    )
    arrays = {
        "model_input_channels_float32": np.stack(
            (image, np.ones_like(image), np.ones_like(image)), axis=-1
        ),
        "source_label_ground_truth_canvas_int64": np.ones(
            (height, width), dtype=np.int64
        ),
        "source_tissue_ground_truth_mask": np.ones((height, width), dtype=bool),
        "target_ccf_coordinates_ap_dv_ml_um_float64": np.zeros(
            (height, width, 3), dtype=np.float64
        ),
        "target_valid_correspondence_mask": np.ones((height, width), dtype=bool),
        "target_correspondence_weight_float32": np.ones(
            (height, width), dtype=np.float32
        ),
        "target_correspondence_abstention_mask": np.zeros(
            (height, width), dtype=bool
        ),
        "truth_section_pullback_map_yx_px_float64": identity,
        "truth_section_pullback_stationary_velocity_yx_px_float64": np.zeros(
            (height, width, 2), dtype=np.float64
        ),
        "truth_section_deformation_valid_mask": np.ones(
            (height, width), dtype=bool
        ),
    }
    artifact = {
        "schema_version": training_row_v3.TRAINING_ROW_V3_SCHEMA,
        "source_observation_receipt_sha256": f"observation-{index}",
        "lineage": {
            "animal_id": f"animal-{index}",
            "specimen_id": f"specimen-{index}",
            "experiment_id": f"experiment-{index}",
            "synthetic_animal_id": f"synthetic-animal-{index}",
            "section_id": f"section-{index}",
            "split": split,
        },
        "upstream_reference": {"fixture": index},
        "numeric_rng_provenance": {"root_seed": "fixture", "index": index},
        "rng_sources": {"mode": f"mode-{index}", "reflection": f"reflection-{index}"},
        "selected_mode": "smart-brush-accurate",
        "selected_descendant_id": f"descendant-{index}",
        "deformation_pose_gauge_reference": {
            "schema_version": deformation_gauge_v3.DEFORMATION_GAUGE_V3_SCHEMA,
            "algorithm": deformation_gauge_v3.DEFORMATION_GAUGE_V3_ALGORITHM,
            "projection_weighting": row_cache_v3.DEFORMATION_GAUGE_PROJECTION_WEIGHTING,
            "deformation_pose_gauge_id": hashlib.sha256(
                f"gauge-id-{index}".encode("ascii")
            ).hexdigest(),
            "receipt_sha256": hashlib.sha256(
                f"gauge-receipt-{index}".encode("ascii")
            ).hexdigest(),
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
        "reflection_transform_id": f"transform-{index}",
        "reflection_realization_id": f"reflection-realization-{index}",
        "paired_view_group_id": f"paired-{index}",
        "synthetic_realization_id": f"realization-{index}",
        "paired_mode_reflected_receipts": {},
        "training_row_id": f"row-{index}",
        "arrays": arrays,
    }
    artifact["array_receipts"] = {
        name: acquisition_v2._array_receipt(value) for name, value in arrays.items()
    }
    artifact["receipt_sha256"] = acquisition_v2._payload_sha256(
        training_row_v3.training_row_receipt_v3(artifact)
    )
    return artifact


def generator_binding():
    return row_cache_v3.make_generator_binding_v3(
        generator_ids=("arbitrary-plane-generator-v3-test",),
        source_sha256={
            "training/generator.py": hashlib.sha256(b"generator-source").hexdigest()
        },
        geometry_gauge_contract={
            "schema_version": deformation_gauge_v3.DEFORMATION_GAUGE_V3_SCHEMA,
            "algorithm": deformation_gauge_v3.DEFORMATION_GAUGE_V3_ALGORITHM,
            "projection_weighting": row_cache_v3.DEFORMATION_GAUGE_PROJECTION_WEIGHTING,
        },
        generator_config={"raster_shape_h_w": [8, 8], "arbitrary_plane": True},
    )


def catalogue():
    return catalogue_v3.make_arbitrary_plane_catalogue_v3(
        np.ones((10, 10, 10), dtype=bool),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        normal_count=2,
        offset_count=3,
        roll_count=1,
        raster_shape_h_w=(8, 8),
        raster_physical_span_y_x_um=(8.0, 8.0),
    )


def atlas():
    values = np.arange(2 * 10 * 10 * 10, dtype=np.float32).reshape(2, 10, 10, 10)
    return values / values.max()


def model_kwargs():
    return {
        "atlas_channels": 2,
        "feature_channels": 4,
        "hidden_channels": 6,
        "correlation_radius": 1,
        "update_limits": (0.08, 0.08, 0.5, 0.08, 0.5, 0.5, 0.04, 0.04, 0.04),
        "plane_tangent_scales": (0.08, 0.08, 0.5),
        "max_velocity_fraction_yx": (0.05, 0.04),
        "deformation_integration_steps": 3,
        "deformation_support_floor": 1e-4,
        "deformation_maximum_velocity_gradient": 0.35,
        "proposal_count": None,
        "proposal_channels": 16,
        "proposal_mixture_components": 8,
        "proposal_offset_scale_um": 10000.0,
    }


def training_config():
    return {
        "seed": 173,
        "pose_warmup_steps": 1,
        "learning_rate": 2e-3,
        "weight_decay": 0.0,
        "top_k": 3,
        "refinement_steps": 1,
        "joint_pose_only_steps": 0,
        "retrieval_shape_h_w": (4, 4),
        "catalogue_chunk_size": 2,
        "amp": False,
        "amp_initial_scale": 128.0,
        "gradient_clip_norm": 5.0,
    }


def runner_config(target=2):
    return {
        "target_applied_steps": target,
        "batch_size": 1,
        "candidate_bank_size": 5,
        "row_selection_seed": "row-order-v3-test",
        "candidate_bank_root_seed": "candidate-bank-v3-test",
        "axial_offsets_um": (-0.5, 0.0, 0.5),
        "axial_weights": (0.25, 0.5, 0.25),
        "archive_checkpoint_interval_applied_steps": 2,
    }
