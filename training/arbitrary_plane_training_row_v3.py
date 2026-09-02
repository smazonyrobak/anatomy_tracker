"""Receipt-bound training rows with raster-only horizontal reflection."""

from __future__ import annotations

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_deformation_gauge_v3 as deformation_gauge
import training.arbitrary_plane_observation_v3 as observation


TRAINING_ROW_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-training-row/v3"
TRAINABLE_MODES = (
    "smart-brush-accurate",
    "smart-brush-imperfect",
    "smart-brush-absent",
)
REFLECTION_STATES = ("none", "horizontal")
_ARRAY_KEYS = {
    "model_input_channels_float32",
    "source_label_ground_truth_canvas_int64",
    "source_tissue_ground_truth_mask",
    "target_ccf_coordinates_ap_dv_ml_um_float64",
    "target_valid_correspondence_mask",
    "target_correspondence_weight_float32",
    "target_correspondence_abstention_mask",
    "truth_section_pullback_map_yx_px_float64",
    "truth_section_pullback_stationary_velocity_yx_px_float64",
    "truth_section_deformation_valid_mask",
}


def _array_receipts(arrays):
    return {name: acquisition._array_receipt(value) for name, value in arrays.items()}


def _byte_equal(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes()
        == np.ascontiguousarray(right).tobytes()
    )


def _verify_observation(artifact):
    if (
        artifact.get("receipt_sha256")
        != acquisition._payload_sha256(observation.observation_bundle_receipt_v3(artifact))
        or artifact.get("array_receipts") != _array_receipts(artifact.get("arrays", {}))
        or set(artifact.get("descendants", {})) != set(observation.DESCENDANT_MODES)
    ):
        raise ValueError("observation v3 live receipt or arrays changed")
    for mode, descendant in artifact["descendants"].items():
        if (
            descendant.get("array_receipts")
            != _array_receipts(descendant.get("arrays", {}))
            or descendant.get("descendant_id")
            != acquisition._payload_sha256(observation._descendant_identity(descendant))
            or descendant.get("mode") != mode
        ):
            raise ValueError("observation descendant receipt changed")


def _rng(artifact, realization_index, field):
    provenance = artifact["rng_provenance"]
    payload = {
        "domain": "anatomy-tracker.training-row-rng/v3",
        "root_seed_uint64": provenance["root_seed_uint64"],
        "split": provenance["split"],
        "split_index": provenance["split_index"],
        "animal_index": provenance["animal_index"],
        "section_index": provenance["section_index"],
        "observation_index": provenance["observation_index"],
        "realization_index": realization_index,
        "field": field,
    }
    seed = int(acquisition._payload_sha256(payload)[:16], 16)
    return np.random.Generator(np.random.PCG64DXSM(seed)), {
        **payload,
        "derived_seed_uint64": f"0x{seed:016x}",
    }


def _outline(mask):
    mask = np.asarray(mask, dtype=bool)
    eroded = mask.copy()
    eroded[1:] &= mask[:-1]
    eroded[:-1] &= mask[1:]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    eroded[[0, -1], :] = False
    eroded[:, [0, -1]] = False
    return np.ascontiguousarray(mask & ~eroded)


def _reflect(array, horizontal):
    value = np.asarray(array)
    return np.ascontiguousarray(value[:, ::-1] if horizontal else value)


def _input_channels(descendant, horizontal):
    arrays = descendant["arrays"]
    image = _reflect(arrays["model_input_image_float32"], horizontal).astype(np.float32)
    outline = _reflect(_outline(arrays["selected_input_mask"]), horizontal)
    availability = np.full(
        image.shape, float(descendant["brush_available"]), dtype=np.float32
    )
    return np.ascontiguousarray(
        np.stack((image, outline.astype(np.float32), availability), axis=-1),
        dtype=np.float32,
    )


def _reflection_geometry(artifact, state):
    canonical = np.asarray(
        artifact["deformation_pose_gauge"][
            "pose_adjusted_effective_quicknii_ouv_float64"
        ],
        dtype=np.float64,
    )
    width = artifact["arrays"]["raw_acquired_image_float32"].shape[1]
    if state == "none":
        affine = np.eye(3, dtype=np.float64)
        observed = canonical.copy()
        index = 0
    else:
        affine = np.array(
            [[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        observed = canonical.copy()
        observed[0] = canonical[0] + ((width - 1.0) / width) * canonical[1]
        observed[1] = -canonical[1]
        index = 1
    return canonical, observed, affine, index


def training_row_receipt_v3(artifact):
    return {
        key: artifact[key]
        for key in (
            "schema_version",
            "source_observation_receipt_sha256",
            "lineage",
            "upstream_reference",
            "numeric_rng_provenance",
            "rng_sources",
            "selected_mode",
            "selected_descendant_id",
            "deformation_pose_gauge_reference",
            "reflection_state",
            "reflection_representation_index",
            "reflection_representation_affine_xy_float64",
            "canonical_effective_quicknii_ouv_float64",
            "observed_effective_quicknii_ouv_float64",
            "proper_physical_pose_unchanged",
            "prior_model_dependencies",
            "prior_feature_dependencies",
            "prior_pseudolabel_dependencies",
            "reflection_transform_id",
            "reflection_realization_id",
            "paired_view_group_id",
            "synthetic_realization_id",
            "paired_mode_reflected_receipts",
            "array_receipts",
            "training_row_id",
        )
    }


def make_arbitrary_plane_training_row_v3(observation_artifact, realization_index):
    _verify_observation(observation_artifact)
    realization_index = acquisition._nonnegative_integer(
        realization_index, "realization_index"
    )
    mode_rng, mode_receipt = _rng(observation_artifact, realization_index, "mode")
    reflection_rng, reflection_receipt = _rng(
        observation_artifact, realization_index, "horizontal-reflection"
    )
    mode = TRAINABLE_MODES[int(mode_rng.integers(0, len(TRAINABLE_MODES)))]
    reflection_state = REFLECTION_STATES[int(reflection_rng.integers(0, 2))]
    horizontal = reflection_state == "horizontal"
    descendant = observation_artifact["descendants"][mode]
    source = observation_artifact["arrays"]
    deformation_valid = _reflect(source["truth_section_deformation_valid_mask"], horizontal)
    pullback_map = _reflect(source["truth_section_pullback_map_yx_px_float64"], horizontal).copy()
    pullback_velocity = _reflect(
        source["truth_section_pullback_stationary_velocity_yx_px_float64"], horizontal
    ).copy()
    width = pullback_map.shape[1]
    if horizontal:
        pullback_map[..., 1] = width - 1.0 - pullback_map[..., 1]
        pullback_velocity[..., 1] *= -1.0
    arrays = {
        "model_input_channels_float32": _input_channels(descendant, horizontal),
        "source_label_ground_truth_canvas_int64": _reflect(
            source["source_label_ground_truth_canvas_int64"], horizontal
        ),
        "source_tissue_ground_truth_mask": _reflect(
            source["source_tissue_ground_truth_mask"], horizontal
        ),
        "target_ccf_coordinates_ap_dv_ml_um_float64": _reflect(
            source["processed_mapped_ccf_physical_coordinates_canvas_float64"], horizontal
        ),
        "target_valid_correspondence_mask": _reflect(
            source["valid_correspondence_mask"], horizontal
        ),
        "target_correspondence_weight_float32": _reflect(
            source["valid_correspondence_weight_float32"], horizontal
        ),
        "target_correspondence_abstention_mask": _reflect(
            source["source_dense_correspondence_abstention_mask"], horizontal
        ),
        "truth_section_pullback_map_yx_px_float64": np.ascontiguousarray(pullback_map),
        "truth_section_pullback_stationary_velocity_yx_px_float64": np.ascontiguousarray(
            pullback_velocity
        ),
        "truth_section_deformation_valid_mask": deformation_valid,
    }
    canonical, observed, affine, representation_index = _reflection_geometry(
        observation_artifact, reflection_state
    )
    transform_id = acquisition._payload_sha256(
        {
            "domain": "anatomy-tracker.raster-reflection-transform/v3",
            "state": reflection_state,
            "canvas_width": width,
            "representation_affine_xy": affine.tolist(),
        }
    )
    numeric = {
        **observation_artifact["rng_provenance"],
        "realization_index": realization_index,
    }
    paired_group = acquisition._payload_sha256(
        {
            "domain": "anatomy-tracker.paired-view-group/v3",
            "observation_bundle_id": observation_artifact["observation_bundle_id"],
            "reflection_transform_id": transform_id,
            "numeric_rng_provenance": numeric,
        }
    )
    paired_receipts = {
        paired_mode: acquisition._array_receipt(
            _input_channels(observation_artifact["descendants"][paired_mode], horizontal)
        )
        for paired_mode in TRAINABLE_MODES
    }
    artifact = {
        "schema_version": TRAINING_ROW_V3_SCHEMA,
        "source_observation_receipt_sha256": observation_artifact["receipt_sha256"],
        "lineage": observation_artifact["lineage"],
        "upstream_reference": observation_artifact["upstream_reference"],
        "numeric_rng_provenance": numeric,
        "rng_sources": {"mode": mode_receipt, "reflection": reflection_receipt},
        "selected_mode": mode,
        "selected_descendant_id": descendant["descendant_id"],
        "deformation_pose_gauge_reference": (
            deformation_gauge.deformation_pose_gauge_reference_v3(
                observation_artifact["deformation_pose_gauge"]
            )
        ),
        "reflection_state": reflection_state,
        "reflection_representation_index": representation_index,
        "reflection_representation_affine_xy_float64": affine.tolist(),
        "canonical_effective_quicknii_ouv_float64": canonical.tolist(),
        "observed_effective_quicknii_ouv_float64": observed.tolist(),
        "proper_physical_pose_unchanged": canonical.tolist(),
        "prior_model_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
        "reflection_transform_id": transform_id,
        "reflection_realization_id": acquisition._payload_sha256(
            {
                "domain": "anatomy-tracker.raster-reflection-realization/v3",
                "reflection_transform_id": transform_id,
                "source_observation_receipt_sha256": observation_artifact[
                    "receipt_sha256"
                ],
                "observation_bundle_id": observation_artifact[
                    "observation_bundle_id"
                ],
                "numeric_rng_provenance": numeric,
            }
        ),
        "paired_view_group_id": paired_group,
        "paired_mode_reflected_receipts": paired_receipts,
        "arrays": arrays,
        "array_receipts": _array_receipts(arrays),
    }
    artifact["synthetic_realization_id"] = acquisition._payload_sha256(
        {
            "domain": "anatomy-tracker.synthetic-training-realization/v3",
            "source_observation_receipt_sha256": artifact[
                "source_observation_receipt_sha256"
            ],
            "lineage": artifact["lineage"],
            "selected_descendant_id": artifact["selected_descendant_id"],
            "reflection_realization_id": artifact["reflection_realization_id"],
        }
    )
    artifact["training_row_id"] = acquisition._payload_sha256(
        {
            "domain": "anatomy-tracker.training-row/v3",
            "synthetic_realization_id": artifact["synthetic_realization_id"],
            "array_receipts": artifact["array_receipts"],
        }
    )
    artifact["receipt_sha256"] = acquisition._payload_sha256(
        training_row_receipt_v3(artifact)
    )
    return artifact


def replay_arbitrary_plane_training_row_v3(artifact, observation_artifact, realization_index):
    return make_arbitrary_plane_training_row_v3(observation_artifact, realization_index)


def verify_arbitrary_plane_training_row_v3(artifact, observation_artifact, realization_index):
    replay = make_arbitrary_plane_training_row_v3(observation_artifact, realization_index)
    if (
        set(artifact) != set(replay)
        or artifact.get("receipt_sha256")
        != acquisition._payload_sha256(training_row_receipt_v3(artifact))
        or artifact.get("array_receipts") != _array_receipts(artifact.get("arrays", {}))
        or set(artifact.get("arrays", {})) != _ARRAY_KEYS
        or training_row_receipt_v3(artifact) != training_row_receipt_v3(replay)
        or any(not _byte_equal(artifact["arrays"][name], replay["arrays"][name]) for name in _ARRAY_KEYS)
    ):
        raise ValueError("training row v3 does not replay exactly")
