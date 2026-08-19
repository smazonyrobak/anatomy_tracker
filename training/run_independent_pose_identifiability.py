"""Cold-start diagnostic for pose representation and nuisance invariance."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import training.independent_joint_data as independent_data
import training.run_independent_initializer_foundation as foundation
from training.independent_joint_model import IndependentJointModel, project_pose_to_domain
from training.independent_joint_variants import (
    IndependentJointSimilarityCanonicalizedModel,
    IndependentJointSpatialMomentModel,
    IndependentJointSpatialMomentSimilarityCanonicalizedModel,
)
from training.quicknii_plane_metric import (
    QUICKNII_PIXEL_GRID_SHAPE,
    QUICKNII_PLANE_DISTANCE_CONTRACT,
    torch_annotation_brain_mask,
    torch_brain_masked_plane_distance,
)
from training.synthetic_registration import SyntheticRegistrationGenerator, split_ap_indices
from training.train_independent_joint import _atomic_save, _rng_state, _set_rng_state


REPOSITORY_ROOT = Path(__file__).parents[1]
SCHEMA_VERSION = 1
PURPOSE = "development-only-cold-start-pose-identifiability-diagnostic"
FORMAT = "independent-pose-identifiability-resume-state-v1"
SOURCE_FILES = (
    "training/independent_joint_model.py",
    "training/independent_joint_data.py",
    "training/train_independent_joint.py",
    "training/synthetic_registration.py",
    "training/quicknii_plane_metric.py",
    "training/run_independent_initializer_foundation.py",
    "training/run_independent_pose_identifiability.py",
    "source/dense_registration_preprocessing.py",
)
MODEL_KWARGS = {
    "pyramid_channels": [24, 40, 64, 96],
    "pose_context_features": 192,
    "pair_features": 96,
    "hidden_channels": 96,
    "integration_steps": 6,
    "maximum_pose_delta": [750.0, 7.5, 7.5],
    "maximum_translation_pixels": 32.0,
    "minimum_scale": 0.4,
    "maximum_scale": 2.0,
    "maximum_velocity_fraction": 0.12,
}
MODEL_CONTRACTS = {
    "training.independent_joint_model.IndependentJointModel": {
        "factory": IndependentJointModel,
        "name": "independent-pose-identifiability-300-r4322-v1",
        "expected_parameter_count": 1_369_070,
        "extra_source_files": (),
        "source_view_supervision": False,
    },
    "training.independent_joint_variants.IndependentJointSpatialMomentModel": {
        "factory": IndependentJointSpatialMomentModel,
        "name": "independent-spatial-moment-pose-identifiability-300-r4322-v1",
        "expected_parameter_count": 1_373_338,
        "extra_source_files": ("training/independent_joint_variants.py",),
        "source_view_supervision": False,
    },
    "training.independent_joint_variants.IndependentJointSimilarityCanonicalizedModel": {
        "factory": IndependentJointSimilarityCanonicalizedModel,
        "name": "independent-supervised-similarity-pose-identifiability-300-r4322-v1",
        "expected_parameter_count": 1_373_904,
        "extra_source_files": ("training/independent_joint_variants.py",),
        "source_view_supervision": True,
    },
    "training.independent_joint_variants.IndependentJointSpatialMomentSimilarityCanonicalizedModel": {
        "factory": IndependentJointSpatialMomentSimilarityCanonicalizedModel,
        "name": "independent-spatial-moment-supervised-similarity-pose-identifiability-300-r4322-v1",
        "expected_parameter_count": 1_378_172,
        "extra_source_files": ("training/independent_joint_variants.py",),
        "source_view_supervision": True,
    },
}


def _model_contract(config: dict) -> dict:
    class_name = config.get("model", {}).get("class")
    if class_name not in MODEL_CONTRACTS:
        raise ValueError("pose-identifiability model class is not allowlisted")
    return MODEL_CONTRACTS[class_name]


def _source_files(config: dict) -> tuple[str, ...]:
    return SOURCE_FILES + tuple(_model_contract(config)["extra_source_files"])


def latent_pose_table(config: dict) -> np.ndarray:
    data = config["data"]
    signs = np.asarray(data["signed_tilt_pairs"], np.float32)
    poses = np.asarray(
        [
            (ap, lr_sign * data["tilt_lr_abs_deg"], dv_sign * data["tilt_dv_abs_deg"])
            for ap in data["ap_um"]
            for lr_sign, dv_sign in signs
        ],
        np.float32,
    )
    if poses.shape != (24, 3):
        raise RuntimeError("pose-identifiability latent table must contain 24 poses")
    return poses


def nuisance_transform_tables(config: dict) -> dict[str, list[np.ndarray]]:
    settings = config["data"]["nuisance_transforms"]
    def panel(rotations, scales, pair_ids):
        pair_ids = np.asarray(pair_ids, np.int64)
        return np.column_stack(
            (
                np.asarray(rotations, np.float32)[pair_ids // len(scales)],
                np.asarray(scales, np.float32)[pair_ids % len(scales)],
            )
        ).astype(np.float32)

    return {
        "seen": [
            panel(settings["seen_rotation_deg"], settings["seen_scale"], pair_ids)
            for pair_ids in settings["seen_pair_assignment"]
        ],
        "held": [
            panel(settings["held_rotation_deg"], settings["held_scale"], pair_ids)
            for pair_ids in settings["held_pair_assignment"]
        ],
    }


def nuisance_shortcut_accuracy(config: dict, partition: str) -> np.ndarray:
    poses = latent_pose_table(config)
    transforms = np.concatenate(nuisance_transform_tables(config)[partition])
    labels = np.tile(np.column_stack((np.repeat(np.arange(6), 4), poses[:, 1] > 0, poses[:, 2] > 0)), (2, 1))
    _, pair_id = np.unique(transforms, axis=0, return_inverse=True)
    return np.asarray([
        sum(np.bincount(axis[pair_id == value].astype(np.int64)).max()
            for value in np.unique(pair_id)) / len(pair_id)
        for axis in labels.T
    ])


def inspect_pose_identifiability_config(path: str | Path) -> dict:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    contract = raw.pop("contract_sha256")
    if foundation._canonical_sha256(raw) != contract:
        raise ValueError("pose-identifiability config hash differs from its payload")
    if raw.get("schema_version") != SCHEMA_VERSION or raw.get("frozen") is not True:
        raise ValueError("pose-identifiability config is not frozen schema v1")
    if raw.get("purpose") != PURPOSE or raw.get("role") != "diagnostic-not-model-selection":
        raise ValueError("pose-identifiability purpose or role changed")
    if any(raw.get(name) is not False for name in (
        "product5_access", "calibration_access", "final_test_access"
    )) or raw.get("learned_checkpoint_dependencies") != []:
        raise ValueError("pose-identifiability must remain synthetic-only and cold-start")
    if raw.get("seed") != 4322 or raw.get("device") != "auto":
        raise ValueError("pose-identifiability seed or device contract changed")
    if raw.get("paths") != {
        "atlas_repo_relative": "data/Allen Brain Atlas 25um",
        "run_root_env": "ATLAS_JOINT_RUN_ROOT",
    }:
        raise ValueError("pose-identifiability path contract changed")
    atlas_relative = Path(raw["paths"]["atlas_repo_relative"])
    if atlas_relative.is_absolute() or ".." in atlas_relative.parts:
        raise ValueError("atlas path must remain repository-relative")
    model_contract = _model_contract(raw)
    if raw.get("name") != model_contract["name"] or raw.get("model") != {
        "class": raw["model"]["class"],
        "expected_parameter_count": model_contract["expected_parameter_count"],
        "kwargs": MODEL_KWARGS,
    }:
        raise ValueError("pose-identifiability model payload changed")

    data = raw["data"]
    required_data = {
        "source": "synthetic_ccf",
        "split": "train",
        "stratum": "clean",
        "pose_regime": "standard",
        "latent_pose_count": 24,
        "base_seed": 304322,
        "held_base_seed": 404322,
        "ap_um": [-4175.0, -3100.0, -2175.0, -1100.0, -175.0, 400.0],
        "tilt_lr_abs_deg": 13.25,
        "tilt_dv_abs_deg": 18.25,
        "signed_tilt_pairs": [[-1, -1], [-1, 1], [1, -1], [1, 1]],
        "sub_bin_target_policy": "nonzero-within-nearest-classifier-bin-on-every-axis",
        "held_generator_realization": "disjoint-from-seen",
        "outline_mode": "absent",
        "seen_panels": 2,
        "held_panels": 2,
        "training_panel_schedule": "alternate-seen-panel-by-update-parity",
    }
    if any(data.get(name) != value for name, value in required_data.items()):
        raise ValueError("pose-identifiability latent data contract changed")
    nuisance = data.get("nuisance_transforms", {})
    if any(nuisance.get(name) != value for name, value in {
        "seen_rotation_deg": [-30.0, -10.0, 10.0, 30.0],
        "seen_scale": [0.8, 0.95, 1.05, 1.2],
        "held_rotation_deg": [-20.0, 0.0, 20.0],
        "held_scale": [0.875, 1.125],
    }.items()):
        raise ValueError("pose-identifiability nuisance transform contract changed")
    if np.asarray(nuisance.get("seen_pair_assignment")).shape != (2, 24) or np.asarray(
        nuisance.get("held_pair_assignment")
    ).shape != (2, 24):
        raise ValueError("pose-identifiability nuisance assignment shape changed")
    poses = latent_pose_table(raw)
    ap_indices = np.rint(
        independent_data.BREGMA_AP_INDEX - poses[:, 0] / independent_data.VOXEL_UM
    ).astype(np.int32)
    if not np.isin(ap_indices, split_ap_indices("train")).all():
        raise ValueError("a fixed AP pose is outside the synthetic training split")
    if not np.allclose(poses[:, 1:].mean(0), 0.0):
        raise ValueError("signed LR/DV tilt balance changed")
    centers = (
        np.linspace(independent_data.AP_MIN_UM, independent_data.AP_MAX_UM, 41),
        np.linspace(-35.0, 35.0, 29),
        np.linspace(-35.0, 35.0, 29),
    )
    residual = np.column_stack([
        poses[:, axis] - axis_centers[np.abs(poses[:, axis, None] - axis_centers).argmin(1)]
        for axis, axis_centers in enumerate(centers)
    ])
    if np.any(np.isclose(residual, 0.0)) or np.any(
        np.abs(residual) > np.asarray((62.5, 1.25, 1.25)) + 1e-6
    ):
        raise ValueError("sub-bin targets must remain nonzero and inside their nearest bins")
    tables = nuisance_transform_tables(raw)
    if any(len(panels) != 2 or any(panel.shape != (24, 2) for panel in panels)
           for panels in tables.values()):
        raise ValueError("fixed nuisance panel cardinality changed")
    seen_values = {tuple(row) for panel in tables["seen"] for row in panel.tolist()}
    held_values = {tuple(row) for panel in tables["held"] for row in panel.tolist()}
    if seen_values.intersection(held_values):
        raise ValueError("seen and held nuisance transforms overlap")
    if any(np.array_equal(panels[0][row], panels[1][row]) for panels in tables.values() for row in range(24)):
        raise ValueError("each pose requires two distinct transforms per partition")
    if not np.array_equal(
        np.bincount(np.asarray(nuisance["seen_pair_assignment"]).reshape(-1), minlength=16),
        np.full(16, 3),
    ) or not np.array_equal(
        np.bincount(np.asarray(nuisance["held_pair_assignment"]).reshape(-1), minlength=6),
        np.full(6, 8),
    ):
        raise ValueError("nuisance Cartesian pairs must remain exactly balanced")
    if np.any(nuisance_shortcut_accuracy(raw, "seen") > np.asarray((0.50, 2 / 3, 2 / 3)) + 1e-9):
        raise ValueError("seen nuisance pairs predict pose labels too accurately")

    training = raw["training"]
    loss_weights = {"categorical": 1.0, "sub_bin_residual": 0.5}
    source_view_contract = None
    if model_contract["source_view_supervision"]:
        loss_weights["source_view_supervision"] = 1.0
        source_view_contract = {
            "canonicalizer_initialization_seed": 12731,
            "targets": ["source_view_rotation_deg", "source_view_scale"],
            "loss": "smooth-l1-normalized-rotation-and-log-scale",
            "pose_gradient_to_canonicalizer": "blocked-at-sampling-parameters",
            "gradient_clipping": "separate-pose-and-canonicalizer-groups-at-5.0",
            "anatomical_pose_target_access": False,
        }
    expected_training = {
        "optimizer": "AdamW",
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "amp": True,
        "amp_initial_scale": 512.0,
        "max_updates": 300,
        "loss_weights": loss_weights,
        "gradient_clip_norm": 5.0,
        "gradient_clip_warmup_updates": 30,
        "resume": True,
        "resume_state_every_updates": 10,
    }
    if source_view_contract is not None:
        expected_training["source_view_supervision_contract"] = source_view_contract
    if training != expected_training:
        raise ValueError("pose-identifiability training contract changed")
    if raw.get("evaluation") != {
        "model_state": "raw-current",
        "decode": "argmax-bin-center-plus-bounded-sub-bin-residual",
        "also_record_public_soft_decode": True,
        "physical_metric": "deepslice-corresponding-pixel-plane-distance-v1",
        "raw_predictions": True,
        "evaluate_at_update": 300,
    }:
        raise ValueError("pose-identifiability evaluation contract changed")
    expected_gates = {
        "seen_ap_bin_accuracy_minimum": 0.95,
        "seen_lr_bin_accuracy_minimum": 0.90,
        "seen_dv_bin_accuracy_minimum": 0.90,
        "held_ap_mae_um_maximum": 250.0,
        "held_lr_mae_deg_maximum": 3.0,
        "held_dv_mae_deg_maximum": 3.0,
        "held_physical_improvement_over_constant_prior_minimum": 0.50,
        "held_prediction_to_truth_sd_ratio_minimum_each_axis": 0.75,
        "seen_residual_improvement_over_zero_minimum_each_axis": 0.20,
        "held_residual_improvement_over_zero_minimum_each_axis": 0.20,
        "nonfinite_count_maximum": 0,
        "postwarm_gradient_clipped_fraction_strict_maximum": 0.50,
    }
    if model_contract["source_view_supervision"]:
        expected_gates.update(
            {
                "seen_source_view_rotation_mae_deg_maximum": 2.0,
                "seen_source_view_scale_mae_maximum": 0.03,
                "held_source_view_rotation_mae_deg_maximum": 3.0,
                "held_source_view_scale_mae_maximum": 0.05,
            }
        )
    if raw.get("gates") != expected_gates:
        raise ValueError("pose-identifiability gates changed")
    expected_sources = raw["lineage"]["source_sha256"]
    if set(expected_sources) != set(_source_files(raw)):
        raise ValueError("pose-identifiability source lineage is incomplete")
    raw["contract_sha256"] = contract
    raw["config_file_sha256"] = foundation._source_sha256(path)
    return raw


def load_pose_identifiability_config(path: str | Path) -> dict:
    """Load an executable config and fail closed on current-source drift."""
    raw = inspect_pose_identifiability_config(path)
    for relative, expected in raw["lineage"]["source_sha256"].items():
        if foundation._source_sha256(REPOSITORY_ROOT / relative) != expected:
            raise ValueError(f"source lineage changed: {relative}")
    return raw


def _absent_outline_plan(count: int, seed: int) -> dict:
    return foundation._forced_outline_plan(
        foundation.OUTLINE_MODES["absent"], count, seed
    )


def fixed_panel_manifests(synthetic, config: dict) -> dict[str, list[dict]]:
    data = config["data"]
    poses = latent_pose_table(config)
    bases = {
        "seen": synthetic.make_manifest(
            len(poses), "train", int(data["base_seed"]), "clean", 1,
            pose_regime="standard",
        ),
        "held": synthetic.make_manifest(
            len(poses), "train", int(data["held_base_seed"]), "clean", 1,
            pose_regime="standard",
        ),
    }
    for base in bases.values():
        generator_manifest = base["generator_manifest"]
        generator_manifest["ap_um"] = poses[:, 0].copy()
        generator_manifest["ap_index"] = (
            independent_data.BREGMA_AP_INDEX - poses[:, 0] / independent_data.VOXEL_UM
        ).astype(np.float32)
        generator_manifest["tilt_lr_deg"] = poses[:, 1].copy()
        generator_manifest["tilt_dv_deg"] = poses[:, 2].copy()
        generator_manifest["manifest_sha256"] = independent_data._payload_sha256(
            {key: value for key, value in generator_manifest.items() if key != "manifest_sha256"}
        )
    transforms = nuisance_transform_tables(config)
    result = {"seen": [], "held": []}
    for kind in result:
        for panel_index, transform in enumerate(transforms[kind]):
            manifest = copy.deepcopy(bases[kind])
            manifest.pop("wrong_candidate_offset", None)
            manifest.pop("negative_count", None)
            manifest["true_pose"] = poses.copy()
            manifest["source_view_rotation_deg"] = transform[:, 0].copy()
            manifest["source_view_scale"] = transform[:, 1].copy()
            manifest["outline_plan"] = _absent_outline_plan(
                len(poses), int(data["base_seed"]) + 100 + 10 * (kind == "held") + panel_index
            )
            manifest["pose_identifiability_panel"] = {
                "kind": kind, "panel_index": panel_index,
            }
            manifest["manifest_sha256"] = independent_data._payload_sha256(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            )
            result[kind].append(manifest)
    return result


def _panel_batch(pair: dict, manifest: dict, data_contract_sha256: str) -> dict:
    input_image, input_mask = independent_data._synthetic_outline_input(
        pair, manifest["outline_plan"]
    )
    source = independent_data._apply_source_view(
        pair, manifest, input_image, input_mask
    )
    return {
        "source_image": source["source_image"].detach().cpu(),
        "source_mask": source["source_mask"].detach().cpu(),
        "mask_available": source["mask_available"].detach().cpu(),
        "true_pose": torch.as_tensor(manifest["true_pose"], dtype=torch.float32),
        "truth_source_view_parameters": source[
            "truth_source_view_parameters"
        ].detach().cpu(),
        "manifest_sha256": manifest["manifest_sha256"],
        "generator_manifest_sha256": manifest["generator_manifest"]["manifest_sha256"],
        "outline_plan_sha256": manifest["outline_plan"]["plan_sha256"],
        "data_contract_sha256": data_contract_sha256,
        "source_view_rotation_deg": np.asarray(manifest["source_view_rotation_deg"]).copy(),
        "source_view_scale": np.asarray(manifest["source_view_scale"]).copy(),
    }


def _prepare_fixed_panels(config: dict, synthetic, generator):
    manifests = fixed_panel_manifests(synthetic, config)
    if manifests["seen"][0]["generator_manifest"]["manifest_sha256"] == manifests["held"][0]["generator_manifest"]["manifest_sha256"]:
        raise RuntimeError("held evaluation must use a disjoint generator realization")
    pairs = {
        kind: generator.batch(panels[0]["generator_manifest"], qa=True)
        for kind, panels in manifests.items()
    }
    batches = {
        kind: [_panel_batch(pairs[kind], manifest, synthetic.contract["contract_sha256"])
               for manifest in panels]
        for kind, panels in manifests.items()
    }
    truth = batches["seen"][0]["true_pose"].to(generator.device, torch.float64)
    brain_mask = torch_annotation_brain_mask(
        foundation._pose_to_quicknii_ouv(truth),
        generator.annotation,
        QUICKNII_PIXEL_GRID_SHAPE,
    ).cpu()
    contract = {
        "version": 1,
        "purpose": PURPOSE,
        "partition": "fixed-synthetic-development-diagnostic",
        "latent_pose_sha256": foundation._tensor_sha256(batches["seen"][0]["true_pose"]),
        "manifests": {
            kind: [manifest["manifest_sha256"] for manifest in panels]
            for kind, panels in manifests.items()
        },
        "generator_realization_sha256": {
            kind: panels[0]["generator_manifest"]["manifest_sha256"]
            for kind, panels in manifests.items()
        },
        "input_sha256": {
            kind: [
                {name: foundation._tensor_sha256(panel[name])
                 for name in (
                     "source_image", "source_mask", "mask_available", "true_pose",
                     "truth_source_view_parameters",
                 )}
                for panel in panels
            ]
            for kind, panels in batches.items()
        },
        "brain_mask_sha256": foundation._tensor_sha256(brain_mask),
        "physical_plane_distance_contract": QUICKNII_PLANE_DISTANCE_CONTRACT,
        "outline_mode": "absent",
        "source_view_supervision_target": (
            "exact-synthetic-source-view-rotation-deg-and-scale-only;"
            "physical-AP-LR-DV-pose-excluded"
        ),
        "learned_checkpoint_dependencies": [],
    }
    contract["contract_sha256"] = foundation._canonical_sha256(contract)
    return contract, batches, brain_mask


def _pose_parameter_group(model: IndependentJointModel) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = [
        model.pyramid.slice_stem,
        model.pyramid.levels,
        model.pose_head.context,
        model.pose_head.ap_logits,
        model.pose_head.lr_logits,
        model.pose_head.dv_logits,
        model.pose_head.residual,
    ]
    if hasattr(model.pose_head, "spatial_attention_logits"):
        modules.append(model.pose_head.spatial_attention_logits)
    if hasattr(model, "source_view_canonicalizer"):
        modules.append(model.source_view_canonicalizer)
    parameters = [parameter for module in modules for parameter in module.parameters()]
    for parameter in parameters:
        parameter.requires_grad_(True)
    if len({id(value) for value in parameters}) != len(parameters):
        raise RuntimeError("pose-identifiability trainable parameter groups overlap")
    if {id(value) for value in model.parameters() if value.requires_grad} != {
        id(value) for value in parameters
    }:
        raise RuntimeError("pose-identifiability trainable boundary changed")
    if any(value.requires_grad for value in model.pose_head.local_cholesky.parameters()):
        raise RuntimeError("probabilistic covariance head must remain frozen")
    return parameters


def _clip_training_gradients(
    model: IndependentJointModel,
    parameters: list[torch.nn.Parameter],
    maximum_norm: float,
) -> dict[str, float | bool]:
    if not hasattr(model, "source_view_canonicalizer"):
        preclip = float(torch.nn.utils.clip_grad_norm_(parameters, maximum_norm))
        postclip = foundation._parameter_grad_norm(parameters)
        clipped = preclip > maximum_norm
        return {
            "preclip_norm": preclip,
            "postclip_norm": postclip,
            "clip_factor": min(1.0, postclip / preclip) if clipped and preclip else 1.0,
            "clipped": clipped,
        }

    canonicalizer_parameters = list(model.source_view_canonicalizer.parameters())
    canonicalizer_ids = {id(value) for value in canonicalizer_parameters}
    pose_parameters = [
        value for value in parameters if id(value) not in canonicalizer_ids
    ]
    if not pose_parameters or len(pose_parameters) + len(canonicalizer_parameters) != len(parameters):
        raise RuntimeError("canonicalizer and pose clipping groups do not partition training")
    pose_preclip = float(
        torch.nn.utils.clip_grad_norm_(pose_parameters, maximum_norm)
    )
    canonicalizer_preclip = float(
        torch.nn.utils.clip_grad_norm_(canonicalizer_parameters, maximum_norm)
    )
    pose_postclip = foundation._parameter_grad_norm(pose_parameters)
    canonicalizer_postclip = foundation._parameter_grad_norm(
        canonicalizer_parameters
    )
    pose_clipped = pose_preclip > maximum_norm
    canonicalizer_clipped = canonicalizer_preclip > maximum_norm
    preclip = math.sqrt(pose_preclip**2 + canonicalizer_preclip**2)
    postclip = math.sqrt(pose_postclip**2 + canonicalizer_postclip**2)
    return {
        "preclip_norm": preclip,
        "postclip_norm": postclip,
        "clip_factor": min(
            1.0,
            pose_postclip / pose_preclip if pose_clipped and pose_preclip else 1.0,
            canonicalizer_postclip / canonicalizer_preclip
            if canonicalizer_clipped and canonicalizer_preclip else 1.0,
        ),
        "clipped": pose_clipped or canonicalizer_clipped,
        "pose_preclip_norm": pose_preclip,
        "pose_postclip_norm": pose_postclip,
        "pose_clip_factor": (
            min(1.0, pose_postclip / pose_preclip)
            if pose_clipped and pose_preclip else 1.0
        ),
        "pose_clipped": pose_clipped,
        "canonicalizer_preclip_norm": canonicalizer_preclip,
        "canonicalizer_postclip_norm": canonicalizer_postclip,
        "canonicalizer_clip_factor": (
            min(1.0, canonicalizer_postclip / canonicalizer_preclip)
            if canonicalizer_clipped and canonicalizer_preclip else 1.0
        ),
        "canonicalizer_clipped": canonicalizer_clipped,
    }


def categorical_residual_loss(
    output: dict[str, torch.Tensor], truth: torch.Tensor, model: IndependentJointModel
) -> dict[str, torch.Tensor]:
    centers = (
        model.pose_head.ap_centers,
        model.pose_head.tilt_centers,
        model.pose_head.tilt_centers,
    )
    logits = (output["ap_logits"], output["lr_logits"], output["dv_logits"])
    target_bins = [
        (truth[:, axis, None] - axis_centers[None]).abs().argmin(1)
        for axis, axis_centers in enumerate(centers)
    ]
    categorical = torch.stack(
        [F.cross_entropy(axis_logits, target) for axis_logits, target in zip(logits, target_bins)]
    ).mean()
    selected = torch.stack(
        [axis_centers.index_select(0, target) for axis_centers, target in zip(centers, target_bins)],
        dim=1,
    )
    maximum = model.pose_head.maximum_residual.to(truth)
    sub_bin = F.smooth_l1_loss(
        output["continuous_residual"] / maximum,
        (truth - selected) / maximum,
    )
    return {
        "categorical": categorical,
        "sub_bin_residual": sub_bin,
        "target_bins": torch.stack(target_bins, dim=1),
    }


def source_view_supervision_loss(
    output: dict[str, torch.Tensor],
    truth_source_view_parameters: torch.Tensor,
    model: IndependentJointModel,
) -> dict[str, torch.Tensor]:
    """Supervise only the exact synthetic nuisance; anatomical pose is absent."""
    canonicalizer = model.source_view_canonicalizer
    target_rotation_deg = truth_source_view_parameters[:, 0]
    target_scale = truth_source_view_parameters[:, 1]
    target_log_scale = torch.log(target_scale)
    predicted = torch.stack(
        (
            output["source_view_rotation_deg"]
            / canonicalizer.maximum_rotation_deg,
            output["source_view_log_scale"]
            / canonicalizer.maximum_log_scale,
        ),
        dim=1,
    )
    target = torch.stack(
        (
            target_rotation_deg / canonicalizer.maximum_rotation_deg,
            target_log_scale / canonicalizer.maximum_log_scale,
        ),
        dim=1,
    )
    return {
        "loss": F.smooth_l1_loss(predicted, target),
        "rotation_absolute_error_deg": (
            output["source_view_rotation_deg"] - target_rotation_deg
        ).abs(),
        "scale_absolute_error": (
            output["source_view_scale"] - target_scale
        ).abs(),
        "log_scale_absolute_error": (
            output["source_view_log_scale"] - target_log_scale
        ).abs(),
    }


def _decoded_prediction(output: dict, model: IndependentJointModel):
    centers = (
        model.pose_head.ap_centers,
        model.pose_head.tilt_centers,
        model.pose_head.tilt_centers,
    )
    logits = (output["ap_logits"], output["lr_logits"], output["dv_logits"])
    bins = torch.stack([value.argmax(1) for value in logits], dim=1)
    selected = torch.stack(
        [axis_centers.index_select(0, bins[:, axis]) for axis, axis_centers in enumerate(centers)],
        dim=1,
    )
    return project_pose_to_domain(selected + output["continuous_residual"]), bins


def _bin_center_pose(bins: torch.Tensor, model: IndependentJointModel) -> torch.Tensor:
    centers = (
        model.pose_head.ap_centers,
        model.pose_head.tilt_centers,
        model.pose_head.tilt_centers,
    )
    return torch.stack(
        [axis_centers.index_select(0, bins[:, axis]) for axis, axis_centers in enumerate(centers)],
        dim=1,
    )


def _evaluate_panels(model, panel_contract, panels, brain_mask, device) -> dict:
    model.eval()
    result = {"panel_contract_sha256": panel_contract["contract_sha256"]}
    nonfinite = 0
    has_canonicalizer = hasattr(model, "source_view_canonicalizer")
    for kind in ("seen", "held"):
        raw, predictions, truths, zero_residual, predicted_bins, target_bins, physical = [], [], [], [], [], [], []
        source_view_errors = []
        for panel_index, cpu in enumerate(panels[kind]):
            image = cpu["source_image"].to(device)
            outline = cpu["source_mask"].to(device)
            available = cpu["mask_available"].to(device)
            truth = cpu["true_pose"].to(device)
            output = model.initialize(image, outline, available)
            prediction, bins = _decoded_prediction(output, model)
            targets = categorical_residual_loss(output, truth, model)["target_bins"]
            zero_pose = _bin_center_pose(targets, model)
            distance = torch_brain_masked_plane_distance(
                foundation._pose_to_quicknii_ouv(truth.to(torch.float64)),
                foundation._pose_to_quicknii_ouv(prediction.to(torch.float64)),
                brain_mask.to(device),
            ) * independent_data.VOXEL_UM
            view_components = None
            if has_canonicalizer:
                view_components = source_view_supervision_loss(
                    output,
                    cpu["truth_source_view_parameters"].to(device),
                    model,
                )
                source_view_errors.append(
                    torch.stack(
                        (
                            view_components["rotation_absolute_error_deg"],
                            view_components["scale_absolute_error"],
                            view_components["log_scale_absolute_error"],
                        ),
                        dim=1,
                    ).detach().cpu()
                )
            finite_outputs = (
                prediction, output["pose"], output["ap_logits"],
                output["lr_logits"], output["dv_logits"], distance,
            )
            if has_canonicalizer:
                finite_outputs += (
                    output["source_view_rotation_deg"],
                    output["source_view_scale"],
                    output["source_view_log_scale"],
                )
            nonfinite += sum(
                int((~torch.isfinite(value)).sum())
                for value in finite_outputs
            )
            predictions.append(prediction.detach().cpu())
            truths.append(truth.detach().cpu())
            zero_residual.append(zero_pose.detach().cpu())
            predicted_bins.append(bins.detach().cpu())
            target_bins.append(targets.detach().cpu())
            physical.append(distance.detach().cpu())
            for item in range(len(truth)):
                record = {
                    "panel_contract_sha256": panel_contract["contract_sha256"],
                    "transform_partition": kind,
                    "panel_index": panel_index,
                    "latent_pose_index": item,
                    "manifest_sha256": cpu["manifest_sha256"],
                    "source_view_rotation_deg": float(cpu["source_view_rotation_deg"][item]),
                    "source_view_scale": float(cpu["source_view_scale"][item]),
                    "true_pose": truth[item].detach().cpu(),
                    "predicted_pose": prediction[item].detach().cpu(),
                    "zero_residual_oracle_bin_pose": zero_pose[item].detach().cpu(),
                    "public_soft_decoded_pose": output["pose"][item].detach().cpu(),
                    "predicted_bins": bins[item].detach().cpu(),
                    "target_bins": targets[item].detach().cpu(),
                    "physical_corresponding_plane_error_um": float(distance[item]),
                }
                if view_components is not None:
                    record.update(
                        {
                            "predicted_source_view_rotation_deg": float(
                                output["source_view_rotation_deg"][item]
                            ),
                            "predicted_source_view_scale": float(
                                output["source_view_scale"][item]
                            ),
                            "source_view_rotation_absolute_error_deg": float(
                                view_components["rotation_absolute_error_deg"][item]
                            ),
                            "source_view_scale_absolute_error": float(
                                view_components["scale_absolute_error"][item]
                            ),
                            "source_view_log_scale_absolute_error": float(
                                view_components["log_scale_absolute_error"][item]
                            ),
                        }
                    )
                record["record_provenance_sha256"] = foundation._canonical_sha256(record)
                raw.append(record)
        prediction = torch.cat(predictions)
        truth = torch.cat(truths)
        zero_pose = torch.cat(zero_residual)
        bins = torch.cat(predicted_bins)
        targets = torch.cat(target_bins)
        absolute = (prediction - truth).abs()
        zero_absolute = (zero_pose - truth).abs()
        result[kind] = {
            "sample_count": len(truth),
            "bin_accuracy": (bins == targets).float().mean(0),
            "mae": absolute.mean(0),
            "zero_residual_oracle_bin_mae": zero_absolute.mean(0),
            "residual_improvement_over_zero": 1.0 - absolute.mean(0) / zero_absolute.mean(0),
            "prediction_sd": prediction.std(0, unbiased=False),
            "truth_sd": truth.std(0, unbiased=False),
            "physical_corresponding_plane_error_um": torch.cat(physical).mean(),
            "raw_predictions": raw,
        }
        if has_canonicalizer:
            errors = torch.cat(source_view_errors)
            result[kind]["source_view_canonicalization"] = {
                "rotation_mae_deg": errors[:, 0].mean(),
                "scale_mae": errors[:, 1].mean(),
                "log_scale_mae": errors[:, 2].mean(),
            }
    held_truth = torch.cat([panel["true_pose"] for panel in panels["held"]])
    prior = held_truth.mean(0, keepdim=True).expand_as(held_truth)
    prior_physical = torch_brain_masked_plane_distance(
        foundation._pose_to_quicknii_ouv(held_truth.to(torch.float64)),
        foundation._pose_to_quicknii_ouv(prior.to(torch.float64)),
        brain_mask.repeat(len(panels["held"]), 1, 1),
    ) * independent_data.VOXEL_UM
    result["held"]["constant_prior_pose"] = held_truth.mean(0)
    result["held"]["constant_prior_physical_error_um"] = prior_physical.mean()
    predicted_physical = result["held"]["physical_corresponding_plane_error_um"]
    result["held"]["physical_improvement_over_constant_prior"] = (
        1.0 - predicted_physical / prior_physical.mean()
    )
    result["nonfinite_output_count"] = nonfinite
    result["result_sha256"] = foundation._canonical_sha256(result)
    return result


def qualification_status(evaluation: dict | None, gradients: list[dict], nonfinite: int, config: dict) -> dict:
    gates = config["gates"]
    if evaluation is None:
        if nonfinite:
            return {
                "decision": "stop",
                "classification": "numerical-failure",
                "checks": {"nonfinite": {"passed": False, "observed": int(nonfinite)}},
                "observed": {"nonfinite_count": int(nonfinite)},
            }
        return {"decision": "pending", "classification": "not-yet-evaluated", "checks": {}}
    seen = evaluation["seen"]
    held = evaluation["held"]
    ratios = torch.as_tensor(held["prediction_sd"]) / torch.as_tensor(held["truth_sd"])
    postwarm = [
        record for record in gradients
        if int(record["update"]) > int(config["training"]["gradient_clip_warmup_updates"])
    ]
    clipped_fraction = None if not postwarm else sum(bool(value["clipped"]) for value in postwarm) / len(postwarm)
    observed = {
        "seen_ap_bin_accuracy": float(torch.as_tensor(seen["bin_accuracy"])[0]),
        "seen_lr_bin_accuracy": float(torch.as_tensor(seen["bin_accuracy"])[1]),
        "seen_dv_bin_accuracy": float(torch.as_tensor(seen["bin_accuracy"])[2]),
        "held_ap_mae_um": float(torch.as_tensor(held["mae"])[0]),
        "held_lr_mae_deg": float(torch.as_tensor(held["mae"])[1]),
        "held_dv_mae_deg": float(torch.as_tensor(held["mae"])[2]),
        "held_physical_improvement_over_constant_prior": float(held["physical_improvement_over_constant_prior"]),
        "held_prediction_to_truth_sd_ratio_each_axis": ratios,
        "seen_residual_improvement_over_zero_each_axis": seen["residual_improvement_over_zero"],
        "held_residual_improvement_over_zero_each_axis": held["residual_improvement_over_zero"],
        "nonfinite_count": int(nonfinite) + int(evaluation["nonfinite_output_count"]),
        "postwarm_gradient_clipped_fraction": clipped_fraction,
    }
    if "source_view_canonicalization" in seen:
        observed.update(
            {
                "seen_source_view_rotation_mae_deg": float(
                    seen["source_view_canonicalization"]["rotation_mae_deg"]
                ),
                "seen_source_view_scale_mae": float(
                    seen["source_view_canonicalization"]["scale_mae"]
                ),
                "held_source_view_rotation_mae_deg": float(
                    held["source_view_canonicalization"]["rotation_mae_deg"]
                ),
                "held_source_view_scale_mae": float(
                    held["source_view_canonicalization"]["scale_mae"]
                ),
            }
        )
    checks = {
        "seen_ap_bin_accuracy": observed["seen_ap_bin_accuracy"] >= gates["seen_ap_bin_accuracy_minimum"],
        "seen_lr_bin_accuracy": observed["seen_lr_bin_accuracy"] >= gates["seen_lr_bin_accuracy_minimum"],
        "seen_dv_bin_accuracy": observed["seen_dv_bin_accuracy"] >= gates["seen_dv_bin_accuracy_minimum"],
        "held_ap_mae_um": observed["held_ap_mae_um"] <= gates["held_ap_mae_um_maximum"],
        "held_lr_mae_deg": observed["held_lr_mae_deg"] <= gates["held_lr_mae_deg_maximum"],
        "held_dv_mae_deg": observed["held_dv_mae_deg"] <= gates["held_dv_mae_deg_maximum"],
        "held_physical_improvement": observed["held_physical_improvement_over_constant_prior"] >= gates["held_physical_improvement_over_constant_prior_minimum"],
        "held_prediction_sd": bool(torch.isfinite(ratios).all() and (ratios >= gates["held_prediction_to_truth_sd_ratio_minimum_each_axis"]).all()),
        "seen_residual_improvement": bool(
            torch.isfinite(torch.as_tensor(seen["residual_improvement_over_zero"])).all()
            and (torch.as_tensor(seen["residual_improvement_over_zero"]) >= gates["seen_residual_improvement_over_zero_minimum_each_axis"]).all()
        ),
        "held_residual_improvement": bool(
            torch.isfinite(torch.as_tensor(held["residual_improvement_over_zero"])).all()
            and (torch.as_tensor(held["residual_improvement_over_zero"]) >= gates["held_residual_improvement_over_zero_minimum_each_axis"]).all()
        ),
        "nonfinite": observed["nonfinite_count"] <= gates["nonfinite_count_maximum"],
        "postwarm_clipping": clipped_fraction is not None and clipped_fraction < gates["postwarm_gradient_clipped_fraction_strict_maximum"],
    }
    canonicalizer_check_names = ()
    if _model_contract(config)["source_view_supervision"]:
        canonicalizer_check_names = (
            "seen_source_view_rotation",
            "seen_source_view_scale",
            "held_source_view_rotation",
            "held_source_view_scale",
        )
        checks.update(
            {
                "seen_source_view_rotation": observed[
                    "seen_source_view_rotation_mae_deg"
                ] <= gates["seen_source_view_rotation_mae_deg_maximum"],
                "seen_source_view_scale": observed[
                    "seen_source_view_scale_mae"
                ] <= gates["seen_source_view_scale_mae_maximum"],
                "held_source_view_rotation": observed[
                    "held_source_view_rotation_mae_deg"
                ] <= gates["held_source_view_rotation_mae_deg_maximum"],
                "held_source_view_scale": observed[
                    "held_source_view_scale_mae"
                ] <= gates["held_source_view_scale_mae_maximum"],
            }
        )
    check_observed = {
        "seen_ap_bin_accuracy": observed["seen_ap_bin_accuracy"],
        "seen_lr_bin_accuracy": observed["seen_lr_bin_accuracy"],
        "seen_dv_bin_accuracy": observed["seen_dv_bin_accuracy"],
        "held_ap_mae_um": observed["held_ap_mae_um"],
        "held_lr_mae_deg": observed["held_lr_mae_deg"],
        "held_dv_mae_deg": observed["held_dv_mae_deg"],
        "held_physical_improvement": observed["held_physical_improvement_over_constant_prior"],
        "held_prediction_sd": observed["held_prediction_to_truth_sd_ratio_each_axis"],
        "seen_residual_improvement": observed["seen_residual_improvement_over_zero_each_axis"],
        "held_residual_improvement": observed["held_residual_improvement_over_zero_each_axis"],
        "nonfinite": observed["nonfinite_count"],
        "postwarm_clipping": observed["postwarm_gradient_clipped_fraction"],
    }
    if canonicalizer_check_names:
        check_observed.update(
            {
                "seen_source_view_rotation": observed[
                    "seen_source_view_rotation_mae_deg"
                ],
                "seen_source_view_scale": observed["seen_source_view_scale_mae"],
                "held_source_view_rotation": observed[
                    "held_source_view_rotation_mae_deg"
                ],
                "held_source_view_scale": observed["held_source_view_scale_mae"],
            }
        )
    seen_pass = all(checks[name] for name in (
        "seen_ap_bin_accuracy", "seen_lr_bin_accuracy", "seen_dv_bin_accuracy",
        "seen_residual_improvement",
    ))
    held_pass = all(checks[name] for name in (
        "held_ap_mae_um", "held_lr_mae_deg", "held_dv_mae_deg",
        "held_physical_improvement", "held_prediction_sd",
        "held_residual_improvement",
    ))
    seen_canonicalizer_pass = all(
        checks[name]
        for name in canonicalizer_check_names
        if name.startswith("seen_")
    )
    held_canonicalizer_pass = all(
        checks[name]
        for name in canonicalizer_check_names
        if name.startswith("held_")
    )
    if not checks["nonfinite"]:
        classification = "numerical-failure"
    elif not seen_canonicalizer_pass:
        classification = "source-view-canonicalizer-not-identified-on-seen-transforms"
    elif not held_canonicalizer_pass:
        classification = (
            "source-view-canonicalizer-identified-on-seen-but-held-transform-"
            "generalization-insufficient"
        )
    elif not seen_pass:
        classification = "pose-representation-not-identifiable-on-seen-transforms"
    elif not held_pass:
        classification = "pose-representation-identifiable-but-held-transform-invariance-insufficient"
    elif not checks["postwarm_clipping"]:
        classification = "pose-representation-and-invariance-demonstrated-but-training-stability-gate-failed"
    else:
        classification = "pose-representation-and-held-transform-invariance-demonstrated"
    return {
        "decision": "go" if all(checks.values()) else "stop",
        "classification": classification,
        "checks": {
            name: {"passed": passed, "observed": check_observed[name]}
            for name, passed in checks.items()
        },
        "observed": observed,
    }


def _resolve_paths(config: dict) -> tuple[Path, Path]:
    atlas = (REPOSITORY_ROOT / config["paths"]["atlas_repo_relative"]).resolve()
    variable = config["paths"]["run_root_env"]
    if not os.environ.get(variable):
        raise RuntimeError(f"set {variable} before running the diagnostic")
    return atlas, Path(os.environ[variable]).resolve()


def run_pose_identifiability(config_path: str | Path, *, max_updates_this_call: int | None = None) -> dict:
    config = load_pose_identifiability_config(config_path)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    requested = config["device"]
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else (
        "cpu" if requested == "auto" else requested
    )
    device = torch.device(device)
    training = config["training"]
    amp_enabled = bool(training["amp"] and device.type == "cuda")
    model_contract = _model_contract(config)
    model_class = model_contract["factory"]
    uses_source_view_supervision = model_contract["source_view_supervision"]
    model = model_class(**config["model"]["kwargs"]).to(device)
    if sum(value.numel() for value in model.parameters()) != int(config["model"]["expected_parameter_count"]):
        raise RuntimeError("pose-identifiability model parameter count changed")
    initial_state_sha256 = foundation._state_sha256(model)
    parameters = _pose_parameter_group(model)
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    frozen_names = [name for name, value in model.named_parameters() if not value.requires_grad]
    frozen_sha256 = foundation._named_parameter_sha256(model, frozen_names)
    atlas_folder, run_root = _resolve_paths(config)
    generator = SyntheticRegistrationGenerator(atlas_folder, device=device)
    synthetic = independent_data.IndependentSyntheticData(generator)
    panel_contract, panels, brain_mask = _prepare_fixed_panels(config, synthetic, generator)
    output_folder = run_root / config["name"]
    state_path = output_folder / "resume_state.pt"
    receipt_path = output_folder / "diagnostic_receipt.json"
    evaluation_path = output_folder / "fixed_panel_predictions.json"
    source_hashes = {
        relative: foundation._source_sha256(REPOSITORY_ROOT / relative)
        for relative in _source_files(config)
    }
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {"name": properties.name, "total_memory_bytes": properties.total_memory}
    setup = {
        "version": 1,
        "purpose": PURPOSE,
        "role": "diagnostic-not-model-selection",
        "product5_access": False,
        "calibration_access": False,
        "final_test_access": False,
        "learned_checkpoint_dependencies": [],
        "artifact_policy": "single-atomic-resume-state-not-a-selected-model-checkpoint",
        "source_view_supervision": {
            "enabled": uses_source_view_supervision,
            "canonicalizer_initialization_seed": (
                model.source_view_canonicalizer.initialization_seed
                if uses_source_view_supervision else None
            ),
            "targets": (
                ["synthetic_source_view_rotation_deg", "synthetic_source_view_scale"]
                if uses_source_view_supervision else []
            ),
            "anatomical_pose_target_access": False,
            "loss": (
                "smooth-l1-on-normalized-rotation-and-log-scale"
                if uses_source_view_supervision else None
            ),
            "pose_gradient_to_canonicalizer": (
                "blocked-at-sampling-parameters"
                if uses_source_view_supervision else None
            ),
            "gradient_clipping": (
                "separate-pose-and-canonicalizer-groups-at-5.0"
                if uses_source_view_supervision else None
            ),
            "attribution_gates": (
                {
                    name: config["gates"][name]
                    for name in (
                        "seen_source_view_rotation_mae_deg_maximum",
                        "seen_source_view_scale_mae_maximum",
                        "held_source_view_rotation_mae_deg_maximum",
                        "held_source_view_scale_mae_maximum",
                    )
                }
                if uses_source_view_supervision else {}
            ),
            "weight": training["loss_weights"].get("source_view_supervision"),
        },
        "config": config,
        "source_sha256": source_hashes,
        "initial_state_sha256": initial_state_sha256,
        "atlas_contract": generator.contract,
        "synthetic_data_contract": synthetic.contract,
        "fixed_panel_contract": panel_contract,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "frozen_parameter_initial_sha256": frozen_sha256,
        "execution_environment": {
            "requested_device": requested,
            "resolved_device": str(device),
            "amp_enabled": amp_enabled,
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "gpu": gpu,
        },
    }
    setup["setup_sha256"] = foundation._canonical_sha256(setup)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=amp_enabled, init_scale=float(training["amp_initial_scale"])
    )
    update = nonfinite = 0
    gradients: list[dict] = []
    evaluation = None
    status = "running"
    previous_receipt = None
    if receipt_path.is_file():
        previous_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if {name: previous_receipt.get(name) for name in setup} != foundation._canonical(setup):
            raise RuntimeError("existing pose-identifiability receipt differs from frozen setup")
        if not state_path.is_file():
            raise RuntimeError("diagnostic receipt exists without its atomic resume state")
    if state_path.is_file() and bool(training["resume"]):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("format") != FORMAT or state.get("setup_sha256") != setup["setup_sha256"]:
            raise RuntimeError("pose-identifiability resume lineage differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        update = int(state["update"])
        nonfinite = int(state["nonfinite_training_count"])
        gradients = state["gradient_records"]
        evaluation = state["evaluation"]
        status = state["status"]
        if update < 0 or update > int(training["max_updates"]) or len(gradients) != update:
            raise RuntimeError("pose-identifiability resume counters disagree")
        if foundation._named_parameter_sha256(model, frozen_names) != frozen_sha256:
            raise RuntimeError("a frozen pose-identifiability parameter changed")
        foundation._validate_rng_state_types(state["rng_state"])
        _set_rng_state(state["rng_state"])

    def save(current_status: str, qualification: dict) -> None:
        if foundation._named_parameter_sha256(model, frozen_names) != frozen_sha256:
            raise RuntimeError("a frozen pose-identifiability parameter changed")
        payload = {
            "format": FORMAT,
            "setup_sha256": setup["setup_sha256"],
            "learned_checkpoint_dependencies": [],
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": _rng_state(),
            "update": update,
            "status": current_status,
            "nonfinite_training_count": nonfinite,
            "gradient_records": gradients,
            "evaluation": evaluation,
            "qualification": qualification,
        }
        _atomic_save(payload, state_path)
        receipt = {
            **setup,
            "status": current_status,
            "progress": {"optimizer_updates": update, "sample_presentations": update * 24},
            "nonfinite_training_count": nonfinite,
            "gradient_records": gradients,
            "evaluation": evaluation,
            "qualification": qualification,
            "resume_state": str(state_path),
            "resume_state_sha256": foundation._binary_sha256(state_path),
        }
        foundation._atomic_json(receipt, receipt_path)

    if status in {"go", "stop"}:
        qualification = qualification_status(evaluation, gradients, nonfinite, config)
        if qualification["decision"] != status:
            raise RuntimeError("terminal pose-identifiability state contradicts its gates")
        save(status, qualification)
        return {"status": status, "updates": update, "qualification": qualification,
                "receipt_path": receipt_path, "state_path": state_path}
    total = int(training["max_updates"])
    call_stop = total
    if max_updates_this_call is not None:
        if int(max_updates_this_call) < 0:
            raise ValueError("max_updates_this_call cannot be negative")
        call_stop = min(total, update + int(max_updates_this_call))
    while update < call_stop:
        panel_index = update % 2
        cpu = panels["seen"][panel_index]
        image = cpu["source_image"].to(device)
        outline = cpu["source_mask"].to(device)
        available = cpu["mask_available"].to(device)
        truth = cpu["true_pose"].to(device)
        truth_source_view = (
            cpu["truth_source_view_parameters"].to(device)
            if uses_source_view_supervision else None
        )
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            output = model.initialize(image, outline, available)
            components = categorical_residual_loss(output, truth, model)
            loss = (
                float(training["loss_weights"]["categorical"]) * components["categorical"]
                + float(training["loss_weights"]["sub_bin_residual"]) * components["sub_bin_residual"]
            )
            view_components = None
            if uses_source_view_supervision:
                view_components = source_view_supervision_loss(
                    output, truth_source_view, model
                )
                loss = loss + float(
                    training["loss_weights"]["source_view_supervision"]
                ) * view_components["loss"]
        finite_outputs = (
            output["ap_logits"], output["lr_logits"], output["dv_logits"],
            output["continuous_residual"],
        )
        if uses_source_view_supervision:
            finite_outputs += (
                output["source_view_rotation_deg"],
                output["source_view_scale"],
                output["source_view_log_scale"],
            )
        output_nonfinite = sum(
            int((~torch.isfinite(value)).sum())
            for value in finite_outputs
        )
        if not bool(torch.isfinite(loss)) or output_nonfinite:
            nonfinite += max(1, output_nonfinite)
            status = "stop"
            break
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clipping = _clip_training_gradients(
            model, parameters, float(training["gradient_clip_norm"])
        )
        preclip = float(clipping["preclip_norm"])
        postclip = float(clipping["postclip_norm"])
        if not math.isfinite(preclip) or not math.isfinite(postclip):
            nonfinite += 1
            status = "stop"
            break
        scaler.step(optimizer)
        scaler.update()
        update += 1
        clipped = bool(clipping["clipped"])
        gradient_record = {
            "update": update,
            "seen_panel_index": panel_index,
            "loss": float(loss.detach()),
            "categorical_loss": float(components["categorical"].detach()),
            "sub_bin_residual_loss": float(components["sub_bin_residual"].detach()),
            "preclip_norm": preclip,
            "postclip_norm": postclip,
            "clip_factor": float(clipping["clip_factor"]),
            "clipped": clipped,
        }
        if view_components is not None:
            gradient_record.update(
                {
                    name: value
                    for name, value in clipping.items()
                    if name.startswith(("pose_", "canonicalizer_"))
                }
            )
            gradient_record.update(
                {
                    "source_view_supervision_loss": float(
                        view_components["loss"].detach()
                    ),
                    "source_view_rotation_mae_deg": float(
                        view_components["rotation_absolute_error_deg"].mean().detach()
                    ),
                    "source_view_scale_mae": float(
                        view_components["scale_absolute_error"].mean().detach()
                    ),
                }
            )
        gradients.append(gradient_record)
        if update % int(training["resume_state_every_updates"]) == 0:
            save("running", qualification_status(None, gradients, nonfinite, config))

    if status != "stop" and update == total:
        with torch.no_grad():
            evaluation = _evaluate_panels(
                model, panel_contract, panels, brain_mask, device
            )
        foundation._atomic_json(evaluation, evaluation_path)
        qualification = qualification_status(evaluation, gradients, nonfinite, config)
        status = qualification["decision"]
    elif status == "stop":
        qualification = {
            "decision": "stop",
            "classification": "numerical-failure",
            "checks": {},
            "observed": {"nonfinite_count": nonfinite},
        }
    else:
        status = "paused"
        qualification = qualification_status(None, gradients, nonfinite, config)
    save(status, qualification)
    return {"status": status, "updates": update, "qualification": qualification,
            "receipt_path": receipt_path, "state_path": state_path}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m training.run_independent_pose_identifiability CONFIG.json")
    print(json.dumps(foundation._canonical(run_pose_identifiability(sys.argv[1])), indent=2))
