import copy
import math

import numpy as np
import pytest
import torch

import training.arbitrary_plane_staged_training as staged
import training.arbitrary_plane_training_bank_v3 as training_bank_v3
from training.arbitrary_plane_batch_v3 import physical_state_from_quicknii_ouv_v3
from training.arbitrary_plane_deformation_primitives import identity_pixel_map_yx
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components,
    full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import positive_inplane_basis


def _state(center):
    return full_frame_state_from_components(
        torch.tensor(center, dtype=torch.float32),
        torch.eye(3),
        positive_inplane_basis(
            torch.log(torch.tensor((4.0, 3.5))), torch.tensor(0.02)
        ),
    )


def _model_kwargs():
    return {
        "atlas_channels": 2,
        "feature_channels": 4,
        "hidden_channels": 6,
        "correlation_radius": 1,
        "update_limits": (0.08, 0.08, 0.5, 0.08, 0.5, 0.5, 0.04, 0.04, 0.04),
        "plane_tangent_scales": (0.08, 0.08, 0.5),
        "max_velocity_fraction_yx": (0.05, 0.04),
        "deformation_integration_steps": 3,
    }


def _config(seed=173):
    return {
        "seed": seed,
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


def _training_state(seed=173):
    return staged.initialize_staged_training(
        _model_kwargs(),
        _config(seed),
        catalogue_id="catalogue-v3-test",
        catalogue_receipt_sha256="catalogue-receipt-v3-test",
        catalogue_cell_count=5,
        generator_ids=("generator-v3-test", "observation-v3-test"),
        device="cpu",
    )


def _batch():
    torch.manual_seed(613)
    batch_size, cells, height, width = 1, 5, 8, 8
    states = torch.stack(
        [
            torch.stack(
                [_state((4.6 + 0.2 * cell, 5.0, 5.1)) for cell in range(cells)]
            )
        ]
    )
    affine = torch.eye(2, 3).expand(batch_size, cells, 2, 2, 3).clone()
    identity = identity_pixel_map_yx(batch_size, (height, width))
    truth_map = identity.clone()
    truth_map[:, 0] += 0.08
    truth_velocity = torch.zeros(batch_size, 2, height, width)
    truth_velocity[:, 0, 2:6, 2:6] = 0.04
    return {
        "data_role": staged.DEVELOPMENT_DATA_ROLE,
        "catalogue_id": "catalogue-v3-test",
        "catalogue_receipt_sha256": "catalogue-receipt-v3-test",
        "full_catalogue_cell_count": cells,
        "catalogue_scope": training_bank_v3.COMPLETE_CATALOGUE_SCOPE,
        "row_identity": [
            {
                "training_row_id": "row-0",
                "training_row_receipt_sha256": "row-receipt-0",
                "synthetic_realization_id": "realization-0",
                "animal_id": "animal-0",
                "specimen_id": "specimen-0",
                "experiment_id": "experiment-0",
                "synthetic_animal_id": "synthetic-animal-0",
                "section_id": "section-0",
                "split": "development",
            }
        ],
        "image": torch.rand(batch_size, 1, height, width),
        "outline": torch.ones(batch_size, 1, height, width),
        "outline_available": torch.ones(batch_size),
        "atlas_volume": torch.rand(2, 10, 10, 10),
        "cell_id": torch.arange(cells),
        "cell_states": states,
        "cell_log_mass": torch.full((batch_size, cells), -math.log(cells)),
        "representation_log_weight": torch.full(
            (batch_size, cells, 2), -math.log(2.0)
        ),
        "representation_to_canonical_raster_affine": affine,
        "output_shape_h_w": (height, width),
        "origin_ap_dv_ml_um": (0.0, 0.0, 0.0),
        "voxel_size_ap_dv_ml_um": (1.0, 1.0, 1.0),
        "support_origin_ap_dv_ml_um": (5.0, 5.0, 5.0),
        "axial_offsets_um": torch.tensor([-0.5, 0.0, 0.5]),
        "axial_weights": torch.tensor([0.25, 0.5, 0.25]),
        "truth_state": states[:, 0].clone(),
        "truth_catalogue_cell_index": torch.tensor([0]),
        "truth_catalogue_cell_source_index": torch.tensor([0]),
        "truth_catalogue_cell_id": torch.tensor([0]),
        "truth_stationary_velocity_yx_px": truth_velocity,
        "truth_pullback_map_yx_px": truth_map,
        "deformation_weight": torch.ones(batch_size, 1, height, width),
    }


def _sampled_batch():
    batch = _batch()
    states = batch["cell_states"][0].double()
    centers, frames, _ = full_frame_state_to_components(states)
    normals = frames[:, :, 2]
    support_origin = torch.tensor(batch["support_origin_ap_dv_ml_um"], dtype=torch.float64)
    catalogue = {
        "catalogue_id": "catalogue-v3-test",
        "receipt_sha256": "catalogue-receipt-v3-test",
        "counts": {"cell_count": states.shape[0], "roll_count": states.shape[0]},
        "coverage_audit": {
            "max_observed_rp2_angular_covering_radius_rad": 0.5,
        },
        "support_geometry": {
            "support_origin_ap_dv_ml_um": support_origin.tolist(),
        },
        "arrays": {
            "cell_normal_ap_dv_ml_float64": normals.numpy(),
            "cell_signed_offset_um_float64": (
                (centers - support_origin) * normals
            ).sum(dim=-1).numpy(),
            "cell_states_float64": states.numpy(),
            "normal_offset_table_um_float64": np.array(
                [[-1.0, 0.0, 1.0]], dtype=np.float64
            ),
        },
    }
    return training_bank_v3.make_training_candidate_batch_v3(
        batch, catalogue, bank_size=5, root_seed="sampled-step-test"
    )


def _deformation_state(state):
    return {
        name: value.detach().clone()
        for name, value in state["model"].deformation_decoder.state_dict().items()
    }


def test_pose_phase_is_bit_exact_for_deformation_then_joint_updates_it():
    state = _training_state()
    before = _deformation_state(state)
    pose_report = staged.train_staged_step(state, _batch())
    after_pose = _deformation_state(state)
    assert pose_report["phase"] == "pose-warmup"
    assert not pose_report["deformation_decoder_called"]
    assert not pose_report["deformation_loss_enabled"]
    assert all(torch.equal(before[name], after_pose[name]) for name in before)
    assert all(
        not parameter.requires_grad
        for parameter in state["model"].deformation_decoder.parameters()
    )

    joint_report = staged.train_staged_step(state, _batch())
    after_joint = _deformation_state(state)
    assert joint_report["phase"] == "joint"
    assert joint_report["deformation_decoder_called"]
    assert joint_report["deformation_loss_enabled"]
    assert joint_report["truth_in_topk_fraction"] == 1.0
    assert any(not torch.equal(after_pose[name], after_joint[name]) for name in before)
    assert all(
        parameter.requires_grad
        for parameter in state["model"].deformation_decoder.parameters()
    )
    assert math.isfinite(joint_report["objective"])
    assert all(math.isfinite(value) for value in joint_report["losses"].values())


def test_staged_training_reports_honest_miss_while_refining_the_truth_cell():
    config = _config(seed=197)
    config["top_k"] = 1
    state = staged.initialize_staged_training(
        _model_kwargs(),
        config,
        catalogue_id="catalogue-v3-test",
        catalogue_receipt_sha256="catalogue-receipt-v3-test",
        catalogue_cell_count=5,
        generator_ids=("generator-v3-test", "observation-v3-test"),
        device="cpu",
    )
    batch = _batch()
    staged.train_staged_step(state, batch)
    preview = staged._model_forward(state, batch, "joint")
    honest_cell = int(preview["pose"]["honest_retrieval_topk_cell_id"][0, 0])
    truth_cell = next(cell for cell in range(3) if cell != honest_cell)
    batch["truth_catalogue_cell_index"] = torch.tensor([truth_cell])
    batch["truth_catalogue_cell_source_index"] = torch.tensor([truth_cell])
    batch["truth_catalogue_cell_id"] = torch.tensor([truth_cell])
    batch["truth_state"] = batch["cell_states"][:, truth_cell].clone()

    report = staged.train_staged_step(state, batch)
    assert report["truth_in_topk_fraction"] == 0.0
    assert report["truth_forced_refinement_fraction"] == 1.0
    assert math.isfinite(report["losses"]["initial_plane_mixture_nll"])
    assert math.isfinite(report["losses"]["final_plane_mixture_nll"])
    assert math.isfinite(report["losses"]["final_landmark_mixture_nll"])
    assert report["losses"]["deformation_eligible_fraction"] == 1.0


def test_checkpoint_resume_is_exact_and_binds_all_provenance(tmp_path):
    state = _training_state(seed=281)
    batch = _batch()
    staged.train_staged_step(state, batch)
    path = tmp_path / "staged.pt"
    staged.save_staged_training_checkpoint(state, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["binding"]["catalogue_id"] == "catalogue-v3-test"
    assert checkpoint["binding"]["catalogue_receipt_sha256"] == "catalogue-receipt-v3-test"
    assert checkpoint["binding"]["catalogue_cell_count"] == 5
    assert checkpoint["binding"]["generator_ids"] == [
        "generator-v3-test",
        "observation-v3-test",
    ]
    assert checkpoint["binding"]["config_id"]
    assert set(checkpoint["binding"]["source_sha256"]) == set(staged.SOURCE_FILES)
    assert checkpoint["seen_training_row_ids"] == ["row-0"]
    assert checkpoint["seen_animal_ids"] == ["animal-0"]
    assert checkpoint["seen_specimen_ids"] == ["specimen-0"]
    assert checkpoint["seen_experiment_ids"] == ["experiment-0"]
    assert not any(checkpoint["learned_dependency_arrays"].values())
    assert len(checkpoint["training_step_ledger"]) == 1

    resumed = staged.load_staged_training_checkpoint(
        path, device="cpu", expected_binding=state["binding"]
    )
    original_report = staged.train_staged_step(state, batch)
    resumed = staged.load_staged_training_checkpoint(
        path, device="cpu", expected_binding=state["binding"]
    )
    resumed_report = staged.train_staged_step(resumed, batch)
    assert original_report == resumed_report
    assert state["global_step"] == resumed["global_step"] == 2
    for name, value in state["model"].state_dict().items():
        assert torch.equal(value, resumed["model"].state_dict()[name]), name


def test_inference_export_receipt_binds_file_state_ids_and_step_ledger(tmp_path):
    state = _training_state(seed=287)
    staged.train_staged_step(state, _batch())
    path = tmp_path / "staged-export.pt"
    staged.save_staged_training_checkpoint(state, path)
    receipt = staged.make_staged_training_export_receipt_v3(path)
    assert staged.verify_staged_training_export_receipt_v3(
        receipt,
        model_kwargs=_model_kwargs(),
        catalogue_id="catalogue-v3-test",
        catalogue_receipt_sha256="catalogue-receipt-v3-test",
        catalogue_cell_count=5,
        model_state_sha256=receipt["model_state_sha256"],
        require_source_file=True,
    )
    assert receipt["training_animal_ids"] == ["animal-0"]
    assert receipt["training_step_ledger_count"] == 1
    tampered = copy.deepcopy(receipt)
    tampered["global_step"] = 2
    with pytest.raises(ValueError, match="export receipt"):
        staged.verify_staged_training_export_receipt_v3(
            tampered,
            model_kwargs=_model_kwargs(),
            catalogue_id="catalogue-v3-test",
            catalogue_receipt_sha256="catalogue-receipt-v3-test",
            catalogue_cell_count=5,
            model_state_sha256=receipt["model_state_sha256"],
        )


def test_one_row_id_cannot_alias_multiple_row_receipts(tmp_path):
    state = _training_state(seed=289)
    staged.train_staged_step(state, _batch())
    conflicting = _batch()
    conflicting["row_identity"][0]["training_row_receipt_sha256"] = "different"
    with pytest.raises(ValueError, match="multiple row receipts"):
        staged.train_staged_step(state, conflicting)
    state["row_identity_records"].append(conflicting["row_identity"][0])
    with pytest.raises(ValueError, match="multiple row receipts"):
        staged.save_staged_training_checkpoint(state, tmp_path / "alias.pt")


def _attempt_report(batch, train_report):
    payload = {
        "schema_version": "anatomy-tracker.arbitrary-plane-training-step-report/v3",
        "run_id": "run-v3-test",
        "run_manifest_receipt_sha256": "f" * 64,
        "attempt_index": 0,
        "global_step_before": 0,
        "global_step_after": 1,
        "row_identity": copy.deepcopy(batch["row_identity"]),
        "retrieval_scope": batch["catalogue_scope"],
        "training_candidate_bank_receipts": copy.deepcopy(
            batch["training_candidate_bank_receipts"]
        ),
        "training_report": copy.deepcopy(train_report),
    }
    return {**payload, "receipt_sha256": staged._hash_json(payload)}


def test_sampled_training_bank_mapping_is_verified_and_report_bound(tmp_path):
    state = _training_state(seed=293)
    batch = _sampled_batch()
    report = staged.train_staged_step(state, batch)
    assert report["optimizer_step_applied"]
    assert report["retrieval_scope"] == batch["training_candidate_bank_scope"]
    assert "training_candidate_bank_receipts" not in state
    assert state["training_step_ledger"][0][
        "training_candidate_bank_receipt_sha256"
    ] == [item["receipt_sha256"] for item in batch["training_candidate_bank_receipts"]]

    path = tmp_path / "sampled-staged.pt"
    staged.save_staged_training_checkpoint(state, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert "training_candidate_bank_receipts" not in checkpoint
    assert checkpoint["training_step_ledger_summary"]["entry_count"] == 1
    assert checkpoint["training_step_ledger_summary"]["final_chain_sha256"] == (
        checkpoint["training_step_ledger"][0]["chain_sha256"]
    )
    with pytest.raises(ValueError, match="authenticated training-run reports"):
        staged.load_staged_training_checkpoint(path, device="cpu")
    attempt = _attempt_report(batch, report)
    resumed = staged.load_staged_training_checkpoint(
        path,
        device="cpu",
        expected_binding=state["binding"],
        training_report_ledger=[attempt],
    )
    assert resumed["training_step_ledger"] == state["training_step_ledger"]

    tampered = copy.deepcopy(batch)
    tampered["selected_full_catalogue_indices"][0, -1] = 0
    with pytest.raises(ValueError, match="differ from their receipt"):
        staged.train_staged_step(resumed, tampered)

    bad_report = copy.deepcopy(attempt)
    bad_report["training_candidate_bank_receipts"][0]["bank_size"] += 1
    report_payload = {
        key: value for key, value in bad_report.items() if key != "receipt_sha256"
    }
    bad_report["receipt_sha256"] = staged._hash_json(report_payload)
    with pytest.raises(ValueError, match="receipt is invalid"):
        staged.load_staged_training_checkpoint(
            path, device="cpu", training_report_ledger=[bad_report]
        )

    bad_ledger = copy.deepcopy(checkpoint)
    bad_ledger["training_step_ledger"][0]["step"] = 7
    bad_ledger_path = tmp_path / "sampled-bad-ledger.pt"
    torch.save(bad_ledger, bad_ledger_path)
    with pytest.raises(ValueError, match="ledger failed"):
        staged.load_staged_training_checkpoint(bad_ledger_path, device="cpu")


def test_checkpoint_history_scales_with_compact_steps_not_bank_payloads(tmp_path):
    state = _training_state(seed=307)
    batch = _sampled_batch()
    staged.train_staged_step(state, batch)
    identity = batch["row_identity"][0]

    def extend_to(step_count):
        while len(state["training_step_ledger"]) < step_count:
            step = len(state["training_step_ledger"])
            payload = {
                "step": step,
                "catalogue_scope": training_bank_v3.TRAINING_CANDIDATE_BANK_SCOPE,
                "training_row_ids": [identity["training_row_id"]],
                "training_row_receipt_sha256": [
                    identity["training_row_receipt_sha256"]
                ],
                "training_row_identity_sha256": [staged._hash_json(identity)],
                "training_candidate_bank_receipt_sha256": [
                    staged._hash_json({"synthetic-bank-step": step})
                ],
            }
            state["training_step_ledger"].append(
                staged._training_step_ledger_entry_v3(
                    state["binding"], state["training_step_ledger"], payload
                )
            )
        state["global_step"] = step_count

    extend_to(8)
    small_path = tmp_path / "compact-8.pt"
    staged.save_staged_training_checkpoint(state, small_path)
    extend_to(64)
    large_path = tmp_path / "compact-64.pt"
    staged.save_staged_training_checkpoint(state, large_path)
    compact = torch.load(large_path, map_location="cpu", weights_only=False)
    assert "training_candidate_bank_receipts" not in compact
    assert compact["training_step_ledger_summary"]["entry_count"] == 64
    assert all(
        set(entry)
        == {
            "step",
            "catalogue_scope",
            "training_row_ids",
            "training_row_receipt_sha256",
            "training_row_identity_sha256",
            "training_candidate_bank_receipt_sha256",
            "entry_sha256",
            "previous_chain_sha256",
            "chain_sha256",
        }
        and all(
            len(receipt_sha256) == 64
            for receipt_sha256 in entry[
                "training_candidate_bank_receipt_sha256"
            ]
        )
        for entry in compact["training_step_ledger"]
    )
    assert large_path.stat().st_size - small_path.stat().st_size < 56 * 2048


def test_dependency_isolation_and_development_only_guards(tmp_path):
    state = _training_state()
    assert state["binding"]["prior_model_weight_dependencies"] == []
    assert state["binding"]["prior_feature_dependencies"] == []
    assert state["binding"]["prior_pseudolabel_dependencies"] == []
    forbidden = _batch()
    forbidden["data_role"] = "final-test"
    with pytest.raises(ValueError, match="forbidden"):
        staged.train_staged_step(state, forbidden)
    with pytest.raises(ValueError, match="only on I"):
        staged.save_staged_training_checkpoint(state, r"C:\forbidden\model.pt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is unavailable")
def test_cuda_amp_uses_conservative_scale_and_skips_overflow_without_advancing():
    batch = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in _batch().items()
    }
    config = _config(seed=331)
    config["amp"] = True
    finite = staged.initialize_staged_training(
        _model_kwargs(),
        config,
        catalogue_id="catalogue-v3-test",
        catalogue_receipt_sha256="catalogue-receipt-v3-test",
        catalogue_cell_count=5,
        generator_ids=("generator-v3-test", "observation-v3-test"),
        device="cuda",
    )
    report = staged.train_staged_step(finite, batch)
    assert report["optimizer_step_applied"]
    assert not report["amp_overflow"]
    assert report["amp_scale_before"] == 128.0
    assert finite["global_step"] == 1

    overflow_config = _config(seed=337)
    overflow_config["amp"] = True
    overflow_config["amp_initial_scale"] = 3.0e38
    overflow = staged.initialize_staged_training(
        _model_kwargs(),
        overflow_config,
        catalogue_id="catalogue-v3-test",
        catalogue_receipt_sha256="catalogue-receipt-v3-test",
        catalogue_cell_count=5,
        generator_ids=("generator-v3-test", "observation-v3-test"),
        device="cuda",
    )
    before = {
        name: value.detach().clone()
        for name, value in overflow["model"].state_dict().items()
    }
    report = staged.train_staged_step(overflow, batch)
    assert not report["optimizer_step_applied"]
    assert report["amp_overflow"]
    assert report["amp_scale_after"] < report["amp_scale_before"]
    assert overflow["global_step"] == 0
    assert overflow["row_identity_records"] == []
    assert all(
        torch.equal(before[name], value)
        for name, value in overflow["model"].state_dict().items()
    )


def test_v3_rows_become_model_ready_without_learned_dependencies():
    height = width = 8
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    identity = np.stack((y, x), axis=-1).astype(np.float64)
    arrays = {
        "model_input_channels_float32": np.stack(
            (
                np.full((height, width), 0.5, np.float32),
                np.ones((height, width), np.float32),
                np.ones((height, width), np.float32),
            ),
            axis=-1,
        ),
        "target_valid_correspondence_mask": np.ones((height, width), bool),
        "target_correspondence_weight_float32": np.ones(
            (height, width), np.float32
        ),
        "target_correspondence_abstention_mask": np.zeros((height, width), bool),
        "truth_section_deformation_valid_mask": np.ones((height, width), bool),
        "truth_section_pullback_map_yx_px_float64": identity,
        "truth_section_pullback_stationary_velocity_yx_px_float64": np.zeros(
            (height, width, 2), np.float64
        ),
    }
    row = {
        "schema_version": "anatomy-tracker.arbitrary-plane-training-row/v3",
        "training_row_id": "row-adapter",
        "synthetic_realization_id": "realization-adapter",
        "source_observation_receipt_sha256": "observation-adapter",
        "upstream_reference": {},
        "numeric_rng_provenance": {},
        "rng_sources": {},
        "prior_model_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
        "selected_mode": "smart-brush-accurate",
        "selected_descendant_id": "descendant-adapter",
        "deformation_pose_gauge_reference": {
            "schema_version": "anatomy-tracker.deformation-pose-gauge/v3",
            "algorithm": (
                "uniform-canvas-affine-svf-projection-and-pose-recomposition/v3"
            ),
            "projection_weighting": (
                "fixed uniform full canvas, matching decoder gauge"
            ),
            "deformation_pose_gauge_id": "0" * 64,
            "receipt_sha256": "1" * 64,
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
        "reflection_transform_id": "reflection-transform-adapter",
        "reflection_realization_id": "reflection-realization-adapter",
        "paired_view_group_id": "paired-view-adapter",
        "paired_mode_reflected_receipts": {},
        "lineage": {
            "animal_id": "a",
            "specimen_id": "s",
            "experiment_id": "e",
            "synthetic_animal_id": "sa",
            "section_id": "sec",
            "split": "development",
        },
        "arrays": arrays,
    }
    row["array_receipts"] = {
        name: staged.acquisition_v2._array_receipt(value)
        for name, value in arrays.items()
    }
    row["receipt_sha256"] = staged.acquisition_v2._payload_sha256(
        staged.training_row_v3.training_row_receipt_v3(row)
    )
    atlas_shape = (11, 12, 13)
    origin = (100.0, 200.0, 300.0)
    spacing = (25.0, 30.0, 35.0)
    truth = physical_state_from_quicknii_ouv_v3(
        row["canonical_effective_quicknii_ouv_float64"],
        atlas_shape,
        origin,
        spacing,
    )
    cells = torch.stack((truth, _state((7.0, 7.0, 7.0))))[None]
    catalogue = {
        "catalogue_id": "catalogue-adapter-test",
        "receipt_sha256": "catalogue-adapter-receipt-test",
        "support_geometry": {
            "origin_ap_dv_ml_um": list(origin),
            "voxel_size_ap_dv_ml_um": list(spacing),
            "support_origin_ap_dv_ml_um": truth[:3].tolist(),
            "support_mask_receipt": {
                "shape": list(atlas_shape),
                "dtype": np.dtype(bool).str,
                "sha256": "fixture",
            },
            "raster_shape_h_w": [height, width],
            "raster_physical_span_y_x_um": [240.0, 200.0],
        },
        "arrays": {
            "normal_offset_table_um_float64": np.array(
                [[-2.0, 0.0, 2.0]], dtype=np.float64
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
        }
    }
    batch = staged.model_ready_rows_v3(
        [row],
        catalogue,
        torch.rand(2, *atlas_shape),
        origin_ap_dv_ml_um=origin,
        voxel_size_ap_dv_ml_um=spacing,
        support_origin_ap_dv_ml_um=tuple(truth[:3].tolist()),
        axial_offsets_um=(-0.5, 0.0, 0.5),
        axial_weights=(0.25, 0.5, 0.25),
    )
    assert batch["image"].shape == (1, 1, height, width)
    assert batch["outline"].shape == (1, 1, height, width)
    assert batch["truth_catalogue_cell_index"].item() == 0
    assert batch["truth_catalogue_cell_source_index"].item() == 0
    assert batch["truth_catalogue_cell_id"].item() == 0
    assert torch.allclose(batch["truth_state"][0], truth.to(torch.float32))
    assert torch.equal(batch["deformation_weight"], torch.ones(1, 1, height, width))

    changed = copy.deepcopy(row)
    changed["arrays"]["model_input_channels_float32"][0, 0, 0] += 0.25
    with pytest.raises(ValueError, match="receipt or arrays changed"):
        staged.model_ready_rows_v3(
            [changed],
            catalogue,
            torch.rand(2, *atlas_shape),
            origin_ap_dv_ml_um=origin,
            voxel_size_ap_dv_ml_um=spacing,
            support_origin_ap_dv_ml_um=tuple(truth[:3].tolist()),
            axial_offsets_um=(-0.5, 0.0, 0.5),
            axial_weights=(0.25, 0.5, 0.25),
        )

    contaminated = copy.deepcopy(row)
    contaminated["prior_model_dependencies"] = ["legacy.pt"]
    with pytest.raises(ValueError, match="learned dependencies"):
        staged.model_ready_rows_v3(
            [contaminated],
            catalogue,
            torch.rand(2, *atlas_shape),
            origin_ap_dv_ml_um=origin,
            voxel_size_ap_dv_ml_um=spacing,
            support_origin_ap_dv_ml_um=tuple(truth[:3].tolist()),
            axial_offsets_um=(-0.5, 0.0, 0.5),
            axial_weights=(0.25, 0.5, 0.25),
        )
    missing_contract = copy.deepcopy(row)
    del missing_contract["prior_pseudolabel_dependencies"]
    with pytest.raises(ValueError, match="learned dependencies"):
        staged.model_ready_rows_v3(
            [missing_contract],
            catalogue,
            torch.rand(2, *atlas_shape),
            origin_ap_dv_ml_um=origin,
            voxel_size_ap_dv_ml_um=spacing,
            support_origin_ap_dv_ml_um=tuple(truth[:3].tolist()),
            axial_offsets_um=(-0.5, 0.0, 0.5),
            axial_weights=(0.25, 0.5, 0.25),
        )
