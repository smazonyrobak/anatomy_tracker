import copy
import hashlib
import math

import numpy as np
import pytest
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_inference_v3 as inference_v3
import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_row_cache_v3 as row_cache_v3
import training.arbitrary_plane_run_export_v3 as run_export_v3
import training.arbitrary_plane_staged_training as staged
import training.arbitrary_plane_training_row_v3 as training_row_v3
import test_arbitrary_plane_inference_v3 as inference_fixture
import test_arbitrary_plane_run_export_v3 as export_fixture
import arbitrary_plane_production_v3_fixtures as production_fixture
import training.arbitrary_plane_training_runner_v3 as runner_v3
from training.arbitrary_plane_batch_v3 import physical_state_from_quicknii_ouv_v3
from training.arbitrary_plane_full_frame_primitives import (
    render_finite_thickness_plane,
)


def test_capability_is_schedule_agnostic_and_arbitrary_offsets_are_exactly_antipodal():
    capability = psf_v4.finite_psf_model_capability_v4()
    assert capability["runtime_schedule_scope"] == (
        "caller-explicit-exact-inference-session-or-feature-cache-bound"
    )
    contract = psf_v4.make_finite_psf_schedule_v4(
        "finite_boxcar",
        72.77212654910907,
        thickness_selection_sha256="a" * 64,
    )
    offsets = np.asarray(contract["axial_offsets_um"])
    assert np.array_equal(offsets, -offsets[::-1])
    assert psf_v4.verify_finite_psf_schedule_v4(contract)


def test_v4_finalizer_requires_exact_slab_source_bindings_and_never_relabels_v3():
    capability = psf_v4.finite_psf_model_capability_v4()
    schedule = psf_v4.make_finite_psf_schedule_v4(
        "finite_boxcar",
        50.0,
        thickness_selection_sha256="1" * 64,
    )
    slab = {
        "slab_observation_id": "2" * 64,
        "centre_plane_targets_receipt_sha256": "3" * 64,
        "finite_psf": schedule,
        "receipt_sha256": "4" * 64,
    }
    legacy = production_fixture.row(0)
    with pytest.raises(ValueError, match="unfinalized slab-derived v4 row"):
        psf_v4.finalize_training_row_v4(
            legacy,
            slab,
            capability=capability,
        )
    row_like = copy.deepcopy(legacy)
    row_like["schema_version"] = psf_v4.TRAINING_ROW_V4_SCHEMA
    row_like.pop("training_row_id")
    row_like.pop("receipt_sha256")
    row_like["source_observation_receipt_sha256"] = "5" * 64
    row_like["synthetic_realization_id"] = "6" * 64
    row_like["arrays"]["source_label_ground_truth_canvas_int64"] = np.full_like(
        row_like["arrays"]["source_label_ground_truth_canvas_int64"], 7
    )
    row_like["array_receipts"] = {
        name: acquisition_v2._array_receipt(value)
        for name, value in row_like["arrays"].items()
    }
    row_like["upstream_reference"].update(
        {
            "slab_observation_id": slab["slab_observation_id"],
            "centre_plane_targets_receipt_sha256": slab[
                "centre_plane_targets_receipt_sha256"
            ],
            "slab_observation_v4_receipt_sha256": slab["receipt_sha256"],
            "finite_psf_sha256": schedule["finite_psf_sha256"],
            "finite_psf_capability_sha256": schedule[
                "finite_psf_capability_sha256"
            ],
        }
    )
    finalized = psf_v4.finalize_training_row_v4(
        row_like,
        slab,
        capability=capability,
    )
    assert psf_v4.verify_training_row_v4(finalized, capability=capability)
    assert finalized["finite_psf_contract"] == {
        **schedule,
        "slab_observation_v4_receipt_sha256": slab["receipt_sha256"],
    }
    assert all(
        np.array_equal(finalized["arrays"][name], row_like["arrays"][name])
        for name in row_like["arrays"]
    )
    assert not np.array_equal(
        finalized["arrays"]["source_label_ground_truth_canvas_int64"],
        legacy["arrays"]["source_label_ground_truth_canvas_int64"],
    )
    extra = copy.deepcopy(finalized)
    extra["unreceipted_extra"] = "must reject"
    with pytest.raises(ValueError, match="receipt or arrays changed"):
        psf_v4.verify_training_row_v4(extra, capability=capability)
    changed = copy.deepcopy(slab)
    changed["slab_observation_id"] = "5" * 64
    with pytest.raises(ValueError, match="does not bind"):
        psf_v4.finalize_training_row_v4(
            row_like,
            changed,
            capability=capability,
        )


def _row(index, thickness, *, schema=psf_v4.TRAINING_ROW_V4_SCHEMA):
    height = width = 8
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
        "truth_section_pullback_map_yx_px_float64": np.stack((y, x), axis=-1).astype(
            np.float64
        ),
        "truth_section_pullback_stationary_velocity_yx_px_float64": np.zeros(
            (height, width, 2), np.float64
        ),
        "truth_section_deformation_valid_mask": np.ones((height, width), bool),
    }
    row = {
        "schema_version": schema,
        "training_row_id": f"row-{index}",
        "synthetic_realization_id": f"realization-{index}",
        "source_observation_receipt_sha256": "1" * 64,
        "upstream_reference": {},
        "numeric_rng_provenance": {},
        "rng_sources": {},
        "prior_model_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
        "selected_mode": "smart-brush-accurate",
        "selected_descendant_id": f"descendant-{index}",
        "deformation_pose_gauge_reference": {
            "schema_version": "anatomy-tracker.deformation-pose-gauge/v3",
            "algorithm": (
                "uniform-canvas-affine-svf-projection-and-pose-recomposition/v3"
            ),
            "projection_weighting": "fixed uniform full canvas, matching decoder gauge",
            "deformation_pose_gauge_id": "d" * 64,
            "receipt_sha256": "e" * 64,
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
        "reflection_transform_id": f"reflection-transform-{index}",
        "reflection_realization_id": f"reflection-realization-{index}",
        "paired_view_group_id": f"paired-view-{index}",
        "paired_mode_reflected_receipts": {},
        "lineage": {
            "animal_id": f"animal-{index}",
            "specimen_id": f"specimen-{index}",
            "experiment_id": f"experiment-{index}",
            "synthetic_animal_id": f"synthetic-animal-{index}",
            "section_id": f"section-{index}",
            "split": "development",
        },
        "arrays": arrays,
    }
    row["array_receipts"] = {
        name: acquisition_v2._array_receipt(value) for name, value in arrays.items()
    }
    if schema == psf_v4.TRAINING_ROW_V4_SCHEMA:
        row["finite_psf_contract"] = {
            **psf_v4.make_finite_psf_schedule_v4(
                "finite_boxcar",
                thickness,
                thickness_selection_sha256=f"{index + 2:x}" * 64,
            ),
            "slab_observation_v4_receipt_sha256": f"{index + 4:x}" * 64,
        }
        row["upstream_reference"].update(
            {
                "slab_observation_id": f"{index + 10:x}" * 64,
                "centre_plane_targets_receipt_sha256": f"{index + 12:x}" * 64,
                "slab_observation_v4_receipt_sha256": row[
                    "finite_psf_contract"
                ]["slab_observation_v4_receipt_sha256"],
                "finite_psf_sha256": row["finite_psf_contract"][
                    "finite_psf_sha256"
                ],
                "finite_psf_capability_sha256": row[
                    "finite_psf_contract"
                ]["finite_psf_capability_sha256"],
            }
        )
        receipt = psf_v4.training_row_receipt_v4(row)
    else:
        receipt = training_row_v3.training_row_receipt_v3(row)
    row["receipt_sha256"] = acquisition_v2._payload_sha256(receipt)
    return row


def _catalogue(row, atlas_shape, origin, spacing):
    truth = physical_state_from_quicknii_ouv_v3(
        row["canonical_effective_quicknii_ouv_float64"],
        atlas_shape,
        origin,
        spacing,
    )
    other = truth.clone()
    other[0] += 50.0
    cells = torch.stack((truth, other))[None]
    return {
        "catalogue_id": "catalogue-psf-v4-test",
        "receipt_sha256": "catalogue-psf-v4-receipt-test",
        "support_geometry": {
            "origin_ap_dv_ml_um": list(origin),
            "voxel_size_ap_dv_ml_um": list(spacing),
            "support_origin_ap_dv_ml_um": truth[:3].tolist(),
            "support_mask_receipt": {
                "shape": list(atlas_shape),
                "dtype": np.dtype(bool).str,
                "sha256": "fixture",
            },
            "raster_shape_h_w": [8, 8],
            "raster_physical_span_y_x_um": [240.0, 200.0],
        },
        "arrays": {
            "normal_offset_table_um_float64": np.asarray(
                [[-50.0, 0.0, 50.0]], dtype=np.float64
            )
        },
        "coverage_audit": {
            "max_observed_rp2_angular_covering_radius_rad": 0.5
        },
        "counts": {"cell_count": 2, "roll_count": 2},
        "tensors": {
            "cell_id": torch.tensor([0, 1]),
            "cell_states": cells,
            "cell_log_mass": torch.full((1, 2), -math.log(2.0)),
            "representation_log_weight": torch.full((1, 2, 2), -math.log(2.0)),
            "representation_to_canonical_raster_affine": torch.eye(2, 3).expand(
                1, 2, 2, 2, 3
            ),
        },
    }


def _replace_row_psf(row, render_mode, thickness, selection_sha256):
    changed = copy.deepcopy(row)
    source_receipt = changed["finite_psf_contract"][
        "slab_observation_v4_receipt_sha256"
    ]
    changed["finite_psf_contract"] = {
        **psf_v4.make_finite_psf_schedule_v4(
            render_mode,
            thickness,
            thickness_selection_sha256=selection_sha256,
        ),
        "slab_observation_v4_receipt_sha256": source_receipt,
    }
    changed["upstream_reference"].update(
        {
            "finite_psf_sha256": changed["finite_psf_contract"][
                "finite_psf_sha256"
            ],
            "finite_psf_capability_sha256": changed[
                "finite_psf_contract"
            ]["finite_psf_capability_sha256"],
        }
    )
    changed["receipt_sha256"] = acquisition_v2._payload_sha256(
        psf_v4.training_row_receipt_v4(changed)
    )
    return changed


def test_per_row_schedules_batch_and_render_exactly_like_individual_rows():
    capability = psf_v4.finite_psf_model_capability_v4()
    rows = [_row(0, 25.0), _row(1, 100.0)]
    atlas_shape = (11, 12, 13)
    origin = (100.0, 200.0, 300.0)
    spacing = (25.0, 30.0, 35.0)
    catalogue = _catalogue(rows[0], atlas_shape, origin, spacing)
    atlas = torch.arange(2 * np.prod(atlas_shape), dtype=torch.float32).reshape(
        2, *atlas_shape
    )
    batch = staged.model_ready_rows_v3(
        rows,
        catalogue,
        atlas,
        origin_ap_dv_ml_um=origin,
        voxel_size_ap_dv_ml_um=spacing,
        support_origin_ap_dv_ml_um=tuple(
            catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]
        ),
        axial_offsets_um=[],
        axial_weights=[],
        finite_psf_capability=capability,
    )
    assert batch["axial_offsets_um"].shape == (2, 9)
    assert batch["axial_weights"].shape == (2, 9)
    assert batch["row_identity"][0]["finite_psf"]["finite_psf_sha256"] == (
        rows[0]["finite_psf_contract"]["finite_psf_sha256"]
    )
    together = render_finite_thickness_plane(
        atlas,
        batch["truth_state"],
        (8, 8),
        origin,
        spacing,
        batch["axial_offsets_um"],
        batch["axial_weights"],
    )
    separate = torch.cat(
        [
            render_finite_thickness_plane(
                atlas,
                batch["truth_state"][index],
                (8, 8),
                origin,
                spacing,
                batch["axial_offsets_um"][index],
                batch["axial_weights"][index],
            )
            for index in range(2)
        ]
    )
    assert torch.allclose(together, separate, atol=0.0, rtol=0.0)


def test_one_v4_batch_rejects_mixed_production_and_ablation_before_tensor_cat():
    capability = psf_v4.finite_psf_model_capability_v4()
    production = _row(0, 25.0)
    ablation = _replace_row_psf(
        _row(1, 100.0),
        "centre_plane_ablation",
        0.0,
        "e" * 64,
    )
    atlas_shape = (11, 12, 13)
    origin = (100.0, 200.0, 300.0)
    spacing = (25.0, 30.0, 35.0)
    catalogue = _catalogue(production, atlas_shape, origin, spacing)
    with pytest.raises(ValueError, match="one render mode and axial sample count"):
        staged.model_ready_rows_v3(
            [production, ablation],
            catalogue,
            torch.ones(2, *atlas_shape),
            origin_ap_dv_ml_um=origin,
            voxel_size_ap_dv_ml_um=spacing,
            support_origin_ap_dv_ml_um=tuple(
                catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]
            ),
            axial_offsets_um=[],
            axial_weights=[],
            finite_psf_capability=capability,
        )


def test_v4_row_has_no_silent_schedule_default_and_rejects_tampering():
    row = _row(0, 50.0)
    assert row_cache_v3.verify_cached_training_row_v3(row)
    kwargs = {
        "atlas_shape_ap_dv_ml": (11, 12, 13),
        "origin_ap_dv_ml_um": (100.0, 200.0, 300.0),
        "voxel_size_ap_dv_ml_um": (25.0, 30.0, 35.0),
    }
    with pytest.raises(ValueError, match="explicit finite-PSF capability"):
        staged.training_row_to_tensors_v3(row, **kwargs)
    changed = copy.deepcopy(row)
    changed["finite_psf_contract"]["nominal_cut_thickness_um"] = 55.0
    with pytest.raises(ValueError, match="receipt"):
        staged.training_row_to_tensors_v3(
            changed,
            **kwargs,
            finite_psf_capability=psf_v4.finite_psf_model_capability_v4(),
        )
    unknown = psf_v4.make_finite_psf_schedule_v4
    with pytest.raises(ValueError, match="unsupported"):
        unknown(
            "finite_boxcar",
            125.0,
            thickness_selection_sha256="f" * 64,
        )


def test_frozen_v3_row_keeps_exact_global_single_plane_schedule():
    row = _row(0, 0.0, schema=training_row_v3.TRAINING_ROW_V3_SCHEMA)
    before = copy.deepcopy(row)
    atlas_shape = (11, 12, 13)
    origin = (100.0, 200.0, 300.0)
    spacing = (25.0, 30.0, 35.0)
    catalogue = _catalogue(row, atlas_shape, origin, spacing)
    batch = staged.model_ready_rows_v3(
        [row],
        catalogue,
        torch.ones(2, *atlas_shape),
        origin_ap_dv_ml_um=origin,
        voxel_size_ap_dv_ml_um=spacing,
        support_origin_ap_dv_ml_um=tuple(
            catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]
        ),
        axial_offsets_um=[0.0],
        axial_weights=[1.0],
    )
    assert torch.equal(batch["axial_offsets_um"], torch.tensor([0.0]))
    assert torch.equal(batch["axial_weights"], torch.tensor([1.0]))
    assert "finite_psf" not in batch["row_identity"][0]
    assert row["receipt_sha256"] == before["receipt_sha256"]
    assert all(
        np.array_equal(row["arrays"][name], before["arrays"][name])
        for name in row["arrays"]
    )


def _v4_checkpoint(tmp_path):
    capability = psf_v4.finite_psf_model_capability_v4()
    catalogue = inference_fixture._catalogue()
    config = inference_fixture._config()
    state = staged.initialize_staged_training(
        config,
        {
            "seed": 193,
            "pose_warmup_steps": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "top_k": 1,
            "refinement_steps": 1,
            "joint_pose_only_steps": 0,
            "retrieval_shape_h_w": (8, 8),
            "catalogue_chunk_size": 1,
            "amp": False,
            "amp_initial_scale": 128.0,
            "gradient_clip_norm": 5.0,
        },
        catalogue_id=catalogue["catalogue_id"],
        catalogue_receipt_sha256=catalogue["receipt_sha256"],
        catalogue_cell_count=int(catalogue["counts"]["cell_count"]),
        generator_ids=("synthetic-generator-v4",),
        device="cpu",
        finite_psf_capability=capability,
    )
    identity = {
        "training_row_id": "training-row-v4",
        "training_row_receipt_sha256": "a" * 64,
        "synthetic_realization_id": "realization-v4",
        "animal_id": "animal-v4",
        "specimen_id": "specimen-v4",
        "experiment_id": "experiment-v4",
        "synthetic_animal_id": "synthetic-animal-v4",
        "section_id": "section-v4",
        "split": "development",
        "finite_psf": {
            "finite_psf_sha256": "b" * 64,
            "slab_observation_v4_receipt_sha256": "c" * 64,
            "render_mode": "finite_boxcar",
            "nominal_cut_thickness_um": 25.0,
        },
    }
    ledger_payload = {
        "step": 0,
        "catalogue_scope": "complete catalogue posterior/inference scope",
        "training_row_ids": [identity["training_row_id"]],
        "training_row_receipt_sha256": [
            identity["training_row_receipt_sha256"]
        ],
        "training_row_identity_sha256": [staged._hash_json(identity)],
        "training_candidate_bank_receipt_sha256": [],
    }
    state["global_step"] = 1
    state["row_identity_records"] = [identity]
    state["training_step_ledger"] = [
        {**ledger_payload, "entry_sha256": staged._hash_json(ledger_payload)}
    ]
    staged_path = tmp_path / "staged-v4.pt"
    staged.save_staged_training_checkpoint(state, staged_path)
    training_receipt = staged.make_staged_training_export_receipt_v3(staged_path)
    atlas = torch.arange(2 * 6 * 6 * 6, dtype=torch.float32).reshape(
        2, 6, 6, 6
    ) / 100.0
    annotation = torch.zeros(6, 6, 6, dtype=torch.long)
    runtime = psf_v4.runtime_schedule_contract_v4(
        np.linspace(-12.5, 12.5, 9),
        np.asarray(psf_v4.PRODUCTION_INTEGER_MASSES) / 16.0,
        capability=capability,
    )
    inference_contract = inference_v3.make_inference_contract_v3(
        atlas,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        None,
        None,
        atlas_semantics=inference_fixture._atlas_semantics(),
        annotation_volume_ap_dv_ml=annotation,
        finite_psf_capability=capability,
    )
    checkpoint = inference_v3.make_arbitrary_plane_joint_checkpoint_v3(
        state["model"],
        config,
        catalogue,
        inference_fixture._provenance(),
        training_receipt,
        inference_contract=inference_contract,
    )
    return capability, catalogue, checkpoint, atlas, annotation, training_receipt


def test_checkpoint_and_feature_cache_bind_capability_and_exact_runtime_psf(
    tmp_path,
):
    (
        capability,
        catalogue,
        checkpoint,
        atlas,
        annotation,
        training_receipt,
    ) = _v4_checkpoint(tmp_path)
    assert checkpoint["inference_contract"]["finite_psf_capability"] == capability
    assert "finite_psf_runtime_contract" not in checkpoint["inference_contract"]
    legacy_contract = inference_v3.make_inference_contract_v3(
        atlas,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        [0.0],
        [1.0],
        atlas_semantics=inference_fixture._atlas_semantics(),
        annotation_volume_ap_dv_ml=annotation,
    )
    checkpoint_model = staged.ArbitraryPlaneJointModel(
        **inference_fixture._config()
    )
    checkpoint_model.load_state_dict(checkpoint["state_dict"])
    with pytest.raises(ValueError, match="capabilities differ"):
        inference_v3.make_arbitrary_plane_joint_checkpoint_v3(
            checkpoint_model,
            inference_fixture._config(),
            catalogue,
            inference_fixture._provenance(),
            training_receipt,
            inference_contract=legacy_contract,
        )
    checkpoint_path = tmp_path / "joint-v4.pt"
    torch.save(checkpoint, checkpoint_path)
    loaded = inference_v3.load_arbitrary_plane_inference_v3(
        checkpoint_path, catalogue
    )
    alternate_runtime = psf_v4.runtime_schedule_contract_v4(
        np.linspace(-25.0, 25.0, 9),
        np.asarray(psf_v4.PRODUCTION_INTEGER_MASSES) / 16.0,
        capability=capability,
    )
    feature_cache = inference_v3.make_arbitrary_plane_catalogue_feature_cache_v3(
        loaded,
        atlas,
        catalogue,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        alternate_runtime["axial_offsets_um"],
        alternate_runtime["axial_weights"],
        tmp_path / "alternate-runtime-cache.pt",
        retrieval_shape_h_w=(8, 8),
        build_chunk_size=1,
        annotation_volume_ap_dv_ml=annotation,
    )
    assert feature_cache["cache_receipt"]["render_and_storage_recipe"][
        "finite_psf_runtime_contract"
    ] == alternate_runtime
    checkpoint_runtime = psf_v4.runtime_schedule_contract_v4(
        np.linspace(-12.5, 12.5, 9),
        np.asarray(psf_v4.PRODUCTION_INTEGER_MASSES) / 16.0,
        capability=capability,
    )
    alternate_session = inference_v3.prepare_arbitrary_plane_inference_session_v3(
        loaded,
        atlas,
        catalogue,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        alternate_runtime["axial_offsets_um"],
        alternate_runtime["axial_weights"],
        annotation_volume_ap_dv_ml=annotation,
        catalogue_feature_cache=feature_cache,
    )
    checkpoint_session = inference_v3.prepare_arbitrary_plane_inference_session_v3(
        loaded,
        atlas,
        catalogue,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        checkpoint_runtime["axial_offsets_um"],
        checkpoint_runtime["axial_weights"],
        annotation_volume_ap_dv_ml=annotation,
    )
    assert alternate_session["runtime_inference_contract"] != (
        checkpoint_session["runtime_inference_contract"]
    )
    with pytest.raises(ValueError, match="runtime PSF differs from session"):
        inference_v3.prepare_arbitrary_plane_inference_session_v3(
            loaded,
            atlas,
            catalogue,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            checkpoint_runtime["axial_offsets_um"],
            checkpoint_runtime["axial_weights"],
            annotation_volume_ap_dv_ml=annotation,
            catalogue_feature_cache=feature_cache,
        )


def test_inference_capability_rejects_unknown_thickness_and_tampering():
    capability = psf_v4.finite_psf_model_capability_v4()
    atlas = torch.ones(2, 6, 6, 6)
    weights = np.asarray(psf_v4.PRODUCTION_INTEGER_MASSES) / 16.0
    contract = inference_v3.make_inference_contract_v3(
        atlas,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        None,
        None,
        atlas_semantics=inference_fixture._atlas_semantics(),
        finite_psf_capability=capability,
    )
    with pytest.raises(ValueError, match="unsupported"):
        inference_v3.make_runtime_inference_contract_v4(
            contract,
            np.linspace(-62.5, 62.5, 9),
            weights,
        )
    changed = copy.deepcopy(contract)
    changed["finite_psf_capability"]["production"]["axial_sample_count"] = 7
    changed["receipt_sha256"] = inference_v3._sha(
        {key: value for key, value in changed.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="failed verification"):
        inference_v3.verify_inference_contract_v3(changed)


def test_runner_rejects_v3_cache_declared_as_v4_capability(tmp_path):
    cache = tmp_path / "legacy-cache"
    row_cache_v3.initialize_training_row_cache_v3(
        cache,
        generator_binding=production_fixture.generator_binding(),
        generation_config={
            "row_count": 2,
            "plane_domain": "all brain-intersecting",
        },
        seed_record={"root_seed": "0xabc", "subject_seed": "0xdef"},
    )
    row_cache_v3.append_training_rows_v3(
        cache, [production_fixture.row(0), production_fixture.row(1)]
    )
    row_cache_v3.freeze_training_row_cache_v3(cache)
    atlas_source = tmp_path / "allen-source.bin"
    atlas_source.write_bytes(b"authenticated Allen source fixture")
    with pytest.raises(ValueError, match="cache row PSF capability differs"):
        runner_v3.initialize_training_run_v3(
            tmp_path / "run",
            cache_directory=cache,
            expected_generator_binding=production_fixture.generator_binding(),
            catalogue=production_fixture.catalogue(),
            atlas_volume=production_fixture.atlas(),
            atlas_source_assets=(
                {
                    "path": str(atlas_source),
                    "role": "Allen test asset",
                    "sha256": hashlib.sha256(atlas_source.read_bytes()).hexdigest(),
                },
            ),
            atlas_preprocessing={"normalization": "fixed deterministic fixture"},
            model_kwargs=production_fixture.model_kwargs(),
            training_config=production_fixture.training_config(),
            runner_config=production_fixture.runner_config(1),
            device="cpu",
            finite_psf_capability=psf_v4.finite_psf_model_capability_v4(),
        )


def test_runner_and_export_bind_per_row_training_but_schedule_agnostic_checkpoint(
    tmp_path,
):
    capability = psf_v4.finite_psf_model_capability_v4()
    rows = []
    for index, thickness in enumerate((25.0, 100.0)):
        row = production_fixture.row(index)
        row["schema_version"] = psf_v4.TRAINING_ROW_V4_SCHEMA
        row["finite_psf_contract"] = {
            **psf_v4.make_finite_psf_schedule_v4(
                "finite_boxcar",
                thickness,
                thickness_selection_sha256=f"{index + 6:x}" * 64,
            ),
            "slab_observation_v4_receipt_sha256": f"{index + 8:x}" * 64,
        }
        row["upstream_reference"].update(
            {
                "slab_observation_id": f"{index + 10:x}" * 64,
                "centre_plane_targets_receipt_sha256": f"{index + 12:x}" * 64,
                "slab_observation_v4_receipt_sha256": row[
                    "finite_psf_contract"
                ]["slab_observation_v4_receipt_sha256"],
                "finite_psf_sha256": row["finite_psf_contract"][
                    "finite_psf_sha256"
                ],
                "finite_psf_capability_sha256": row[
                    "finite_psf_contract"
                ]["finite_psf_capability_sha256"],
            }
        )
        row["receipt_sha256"] = acquisition_v2._payload_sha256(
            psf_v4.training_row_receipt_v4(row)
        )
        rows.append(row)
    atlas_source = tmp_path / "v4-allen-source.bin"
    atlas_source.write_bytes(b"authenticated Allen source fixture")
    config = production_fixture.runner_config(1)
    config["axial_offsets_um"] = []
    config["axial_weights"] = []
    mixed_cache = tmp_path / "mixed-v4-cache"
    row_cache_v3.initialize_training_row_cache_v3(
        mixed_cache,
        generator_binding=production_fixture.generator_binding(),
        generation_config={
            "row_count": 2,
            "plane_domain": "all brain-intersecting",
        },
        seed_record={"root_seed": "0xabc", "subject_seed": "0xdef"},
    )
    row_cache_v3.append_training_rows_v3(
        mixed_cache,
        [
            rows[0],
            _replace_row_psf(
                rows[1],
                "centre_plane_ablation",
                0.0,
                "f" * 64,
            ),
        ],
    )
    row_cache_v3.freeze_training_row_cache_v3(mixed_cache)
    with pytest.raises(ValueError, match="one PSF render mode and sample count"):
        runner_v3.initialize_training_run_v3(
            tmp_path / "mixed-v4-run",
            cache_directory=mixed_cache,
            expected_generator_binding=production_fixture.generator_binding(),
            catalogue=production_fixture.catalogue(),
            atlas_volume=production_fixture.atlas(),
            atlas_source_assets=(
                {
                    "path": str(atlas_source),
                    "role": "Allen test asset",
                    "sha256": hashlib.sha256(
                        atlas_source.read_bytes()
                    ).hexdigest(),
                },
            ),
            atlas_preprocessing={"normalization": "fixed deterministic fixture"},
            model_kwargs=production_fixture.model_kwargs(),
            training_config=production_fixture.training_config(),
            runner_config=config,
            device="cpu",
            finite_psf_capability=capability,
        )
    cache = tmp_path / "v4-cache"
    row_cache_v3.initialize_training_row_cache_v3(
        cache,
        generator_binding=production_fixture.generator_binding(),
        generation_config={
            "row_count": 2,
            "plane_domain": "all brain-intersecting",
        },
        seed_record={"root_seed": "0xabc", "subject_seed": "0xdef"},
    )
    row_cache_v3.append_training_rows_v3(cache, rows)
    row_cache_v3.freeze_training_row_cache_v3(cache)
    manifest, _ = runner_v3.initialize_training_run_v3(
        tmp_path / "v4-run",
        cache_directory=cache,
        expected_generator_binding=production_fixture.generator_binding(),
        catalogue=production_fixture.catalogue(),
        atlas_volume=production_fixture.atlas(),
        atlas_source_assets=(
            {
                "path": str(atlas_source),
                "role": "Allen test asset",
                "sha256": hashlib.sha256(atlas_source.read_bytes()).hexdigest(),
            },
        ),
        atlas_preprocessing={"normalization": "fixed deterministic fixture"},
        model_kwargs=production_fixture.model_kwargs(),
        training_config=production_fixture.training_config(),
        runner_config=config,
        device="cpu",
        finite_psf_capability=capability,
    )
    assert manifest["finite_psf_capability"] == capability
    assert manifest["finite_psf_training_schedule_source"][
        "schedule_source"
    ] == "authenticated-per-row"
    assert manifest["finite_psf_training_schedule_source"][
        "global_schedule_fallback"
    ] is None
    assert "finite_psf_contract" not in manifest
    assert "finite_psf_runtime_contract" not in manifest
    assert manifest["staged_training_binding"]["finite_psf_capability"] == capability
    reports = runner_v3.run_training_attempts_v3(
        tmp_path / "v4-run", max_attempts=1
    )
    assert len(reports) == 1
    assert all("finite_psf" in identity for identity in reports[0]["row_identity"])
    export_path = tmp_path / "v4-export.pt"
    export_report = run_export_v3.export_training_run_to_inference_checkpoint_v3(
        tmp_path / "v4-run",
        export_path,
        atlas_semantics=export_fixture._semantics(manifest),
    )
    inference_contract = export_report["inference_contract"]
    assert inference_contract["finite_psf_capability"] == capability
    assert inference_contract["finite_psf"] == (
        inference_v3.CHECKPOINT_FINITE_PSF_CAPABILITY_SCOPE_V4
    )
    assert "axial_offsets_um" not in inference_contract["finite_psf"]
    loaded = inference_v3.load_arbitrary_plane_inference_v3(
        export_path,
        runner_v3.load_training_run_v3(tmp_path / "v4-run")["catalogue"],
    )
    assert loaded["inference_contract"] == inference_contract
