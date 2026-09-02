import copy

import pytest
import torch
from torch import nn

import training.arbitrary_plane_staged_trainer_v6 as trainer


class _Runtime:
    cell_count = trainer.FULL_CATALOGUE_CELL_COUNT_V6
    binding = {
        "schema_version": "anatomy-tracker.complete-catalogue-runtime/v6",
        "catalogue_id": "complete-v6",
        "catalogue_receipt_sha256": "a" * 64,
        "cell_count": trainer.FULL_CATALOGUE_CELL_COUNT_V6,
        "representation_count": 1,
        "device": "cpu",
        "dtype": "torch.float32",
        "support_origin_ap_dv_ml_um": (0.0, 0.0, 0.0),
    }


class _CatalogueBatch:
    batch_size = 2


class _Pose(nn.Module):
    def __init__(self, logits, render_budget):
        super().__init__()
        self.logits = logits
        self.cascade_max_rendered_cells_per_sample = render_budget
        self.proposal_calls = 0
        self.rerank_calls = 0

    def _q(self, batch):
        value = self.logits.log_softmax(0).expand(batch, -1)
        return value - torch.logsumexp(value, dim=1, keepdim=True)

    def forward_proposal_only(self, image, outline, available, catalogue, shape):
        self.proposal_calls += 1
        return {
            "raw_full_catalogue_cell_log_probability": self._q(image.shape[0]),
            "atlas_render_count": 0,
            "cascade_boundary": "full_catalogue_proposal_only",
            "probabilities_calibrated": False,
        }

    def forward_proposed(self, image, outline, available, atlas, catalogue, shape, origin, voxel, offsets, weights, **kwargs):
        self.rerank_calls += 1
        q = self._q(image.shape[0])
        honest_index = torch.tensor([[0, 1], [0, 1]])
        honest_valid = torch.ones(2, 2, dtype=torch.bool)
        honest_log = torch.log_softmax(q[:, :2], dim=1)
        training_index = torch.tensor([[0, 1, 0], [0, 1, 2]])
        training_valid = torch.tensor([[True, True, False], [True, True, True]])
        teacher_logits = torch.stack((q[0, [0, 1, 0]], q[1, [0, 1, 2]]))
        teacher_log = torch.log_softmax(teacher_logits.masked_fill(~training_valid, -torch.inf), dim=1)
        return {
            "raw_full_catalogue_proposal_log_probability": q,
            "probability_status": "raw_uncalibrated",
            "training_truth_leakage_into_honest_hybrid": False,
            "training_truth_forced_mask": torch.tensor([False, True]),
            "training_selected_catalogue_index": training_index,
            "training_selected_valid_mask": training_valid,
            "honest_hybrid_posterior": {
                "selection_scope": "honest_proposal_plus_adaptive_closure_no_truth",
                "selected_catalogue_index": honest_index,
                "selected_valid_mask": honest_valid,
                "selected_conditional_log_probability": honest_log,
            },
            "training_teacher_forced_hybrid_posterior": {
                "selected_catalogue_index": training_index,
                "selected_valid_mask": training_valid,
                "selected_conditional_log_probability": teacher_log,
            },
        }


class _Model(nn.Module):
    def __init__(self, runtime, atlas_channels, pose_only_steps, **kwargs):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(trainer.FULL_CATALOGUE_CELL_COUNT_V6))
        self.pose_model = _Pose(self.logits, kwargs.get("cascade_max_rendered_cells_per_sample", 64))
        self.pose_only_steps = pose_only_steps
        self.joint_calls = 0

    def forward(self, *args, **kwargs):
        self.joint_calls += 1
        return {
            "objective_source": self.logits[0].square() + self.logits[1],
            "refinement_ready_mask": torch.ones(2, dtype=torch.bool),
            "refinement_source_batch_index": torch.arange(2),
            "refinement_performed": True,
            "refined_output": {
                "deformation_gating_audit": {"pose_only_steps": self.pose_only_steps},
                "deformation_active_sequence": torch.arange(kwargs["refinement_steps"] + 1) >= self.pose_only_steps,
            },
        }


def _config():
    return {
        "seed": 7,
        "proposal_only_steps": 1,
        "pose_rerank_steps": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "proposal_top_m": 2,
        "top_k": 2,
        "refinement_steps": 2,
        "joint_pose_only_steps": 1,
        "retrieval_shape_h_w": (4, 4),
        "amp": False,
        "amp_initial_scale": 16.0,
        "gradient_clip_norm": 10.0,
        "proposal_loss_weight": 1.0,
        "rerank_loss_weight": 1.0,
    }


def _run_binding():
    return {
        "run_manifest_receipt_sha256": "1" * 64,
        "atlas_binding_receipt_sha256": "2" * 64,
        "training_data_manifest_receipt_sha256": "3" * 64,
    }


def _batch():
    image = torch.ones(2, 1, 4, 4)
    outline = torch.stack((torch.zeros(1, 4, 4), torch.ones(1, 4, 4)))
    frozen_source = {
        "schema_version": trainer.FROZEN_ROWS_V6_SCHEMA,
        "training_data_manifest_receipt_sha256": "3" * 64,
        "cache_manifest_receipt_sha256": "3" * 64,
        "generator_binding_receipt_sha256": "4" * 64,
        "generation_lineage_sha256": "5" * 64,
        "row_indices": [0, 1],
        "training_row_ids": ["row0", "row1"],
        "training_row_receipts_sha256": ["a" * 64, "c" * 64],
    }
    frozen_source["selection_receipt_sha256"] = trainer._hash_json(frozen_source)
    return {
        "image": image,
        "outline": outline,
        "outline_available": torch.tensor([1.0, 0.0]),
        "input_mode": ["raw", "imperfect-mask"],
        "provenance": [
            {
                "specimen_id": "s0", "animal_id": "a0", "experiment_id": "e0",
                "section_id": "x0", "synthetic_animal_id": "sa0",
                "training_row_id": "row0", "training_row_receipt_sha256": "a" * 64,
                "provenance_sha256": "b" * 64,
            },
            {
                "specimen_id": "s1", "animal_id": "a1", "experiment_id": "e1",
                "section_id": "x1", "synthetic_animal_id": "sa1",
                "training_row_id": "row1", "training_row_receipt_sha256": "c" * 64,
            },
        ],
        "frozen_row_source": frozen_source,
        "row_receipts": [
            {
                "training_row_id": "row0",
                "training_row_receipt_sha256": "a" * 64,
                "synthetic_realization_id": "d" * 64,
            },
            {
                "training_row_id": "row1",
                "training_row_receipt_sha256": "c" * 64,
                "synthetic_realization_id": "e" * 64,
            },
        ],
        "catalogue_batch": _CatalogueBatch(),
        "atlas_volume": torch.zeros(1),
        "output_shape_h_w": (4, 4),
        "origin_ap_dv_ml_um": (0.0, 0.0, 0.0),
        "voxel_size_ap_dv_ml_um": (1.0, 1.0, 1.0),
        "axial_offsets_um": torch.tensor([0.0]),
        "axial_weights": torch.tensor([1.0]),
        "truth_catalogue_index": torch.tensor([0, 2]),
        "truth_state": torch.zeros(2, 12),
        "truth_stationary_velocity_yx_px": torch.zeros(2, 2, 4, 4),
        "truth_pullback_map_yx_px": torch.zeros(2, 2, 4, 4),
        "deformation_weight": torch.ones(2, 1, 4, 4),
        "retrieval_supervision_weight": torch.ones(2),
        "pose_supervision_weight": torch.ones(2),
        "dense_deformation_supervision_weight": torch.ones(2),
    }


def _state(monkeypatch, native_loss=None):
    monkeypatch.setattr(trainer, "verify_complete_catalogue_runtime_v6", lambda runtime: True)
    monkeypatch.setattr(trainer, "verify_bound_complete_catalogue_batch_v6", lambda batch, expected_runtime: {})
    monkeypatch.setattr(trainer, "ArbitraryPlaneJointModelV6", _Model)
    if native_loss is None:
        native_loss = lambda output, *args, **kwargs: {
            "total": output["objective_source"],
            "probabilities_calibrated": False,
            "probability_status": "raw_uncalibrated",
        }
    monkeypatch.setattr(trainer, "arbitrary_plane_joint_loss_v6", native_loss)
    return trainer.initialize_staged_trainer_v6(
        _Runtime(), 1, {}, _config(), training_run_binding=_run_binding(), device="cpu"
    )


def test_three_stage_schedule_uses_zero_render_then_honest_rerank_then_native_joint_loss(monkeypatch):
    calls = []

    def joint_loss(output, *args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs["expected_catalogue_cell_count"] == 98_304
        return {
            "total": output["objective_source"],
            "probabilities_calibrated": False,
            "probability_status": "raw_uncalibrated",
        }

    state = _state(monkeypatch, joint_loss)
    reports = [trainer.train_staged_step_v6(state, _batch()) for _ in range(3)]
    assert [report["phase"] for report in reports] == ["proposal-only", "pose-rerank", "joint"]
    assert state["model"].pose_model.proposal_calls == 1
    assert state["model"].pose_model.rerank_calls == 1
    assert state["model"].joint_calls == 1
    assert state["model"].pose_only_steps == 1
    assert len(calls) == 1
    assert all(report["probability_status"] == "raw_uncalibrated" for report in reports)


def test_rerank_rejects_teacher_addition_on_an_honest_hit(monkeypatch):
    state = _state(monkeypatch)
    state["global_step"] = 1
    original = state["model"].pose_model.forward_proposed

    def invalid(*args, **kwargs):
        output = original(*args, **kwargs)
        output["training_truth_forced_mask"] = torch.tensor([True, True])
        return output

    state["model"].pose_model.forward_proposed = invalid
    with pytest.raises(RuntimeError, match="only on honest misses"):
        trainer.train_staged_step_v6(state, _batch())


def test_joint_all_abstained_batch_keeps_retrieval_training_and_records_r0(monkeypatch):
    state = _state(monkeypatch)
    state["global_step"] = 2

    def all_abstained(*args, **kwargs):
        return {
            "objective_source": state["model"].logits[0],
            "refinement_ready_mask": torch.zeros(2, dtype=torch.bool),
            "refinement_source_batch_index": torch.empty(0, dtype=torch.long),
            "refinement_performed": False,
            "refined_output": None,
        }

    state["model"].forward = all_abstained
    report = trainer.train_staged_step_v6(state, _batch())
    assert report["refinement_ready_row_count"] == 0
    assert report["refinement_abstained_row_count"] == 2
    assert state["training_step_ledger"][-1]["refinement_ready_row_count"] == 0


@pytest.mark.parametrize("field", ["input_mode", "provenance"])
def test_batch_requires_declared_input_semantics_and_exact_five_id_provenance(monkeypatch, field):
    state = _state(monkeypatch)
    batch = _batch()
    if field == "input_mode":
        batch[field][0] = "unknown"
    else:
        batch[field][0]["extra"] = "not-exact"
    with pytest.raises(ValueError):
        trainer.train_staged_step_v6(state, batch)


def test_batch_requires_run_bound_frozen_row_selection(monkeypatch):
    state = _state(monkeypatch)
    batch = _batch()
    batch.pop("frozen_row_source")
    with pytest.raises(ValueError, match="authenticated frozen-row selection"):
        trainer.train_staged_step_v6(state, batch)

    batch = _batch()
    batch["frozen_row_source"]["training_data_manifest_receipt_sha256"] = "9" * 64
    batch["frozen_row_source"]["cache_manifest_receipt_sha256"] = "9" * 64
    batch["frozen_row_source"]["selection_receipt_sha256"] = trainer._hash_json(
        {
            key: value
            for key, value in batch["frozen_row_source"].items()
            if key != "selection_receipt_sha256"
        }
    )
    with pytest.raises(ValueError, match="selection receipt or run binding"):
        trainer.train_staged_step_v6(state, batch)

    batch = _batch()
    batch["frozen_row_source"]["selection_receipt_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="selection receipt or run binding"):
        trainer.train_staged_step_v6(state, batch)

    batch = _batch()
    batch["provenance"][0]["training_row_id"] = "different"
    with pytest.raises(ValueError, match="provenance differs"):
        trainer.train_staged_step_v6(state, batch)

    batch = _batch()
    batch["row_receipts"][0]["training_row_receipt_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="row receipts differ"):
        trainer.train_staged_step_v6(state, batch)


def test_black_exterior_mode_accepts_nonzero_tissue_interior_with_boundary_only_outline(monkeypatch):
    state = _state(monkeypatch)
    batch = _batch()
    image = torch.zeros(2, 1, 5, 5)
    image[0, 0, 1:4, 1:4] = 1.0
    outline = torch.zeros_like(image)
    outline[0, 0, 1, 1:4] = 1.0
    outline[0, 0, 3, 1:4] = 1.0
    outline[0, 0, 1:4, 1] = 1.0
    outline[0, 0, 1:4, 3] = 1.0
    batch["image"] = image
    batch["outline"] = outline
    batch["input_mode"] = ["black-exterior", "raw"]
    batch["outline_available"] = torch.tensor([1.0, 0.0])
    report = trainer.train_staged_step_v6(state, batch)
    assert report["phase"] == "proposal-only"


def test_joint_schedule_rejects_nonboolean_active_sequence(monkeypatch):
    state = _state(monkeypatch)
    state["global_step"] = 2
    original = state["model"].forward

    def nonboolean(*args, **kwargs):
        output = original(*args, **kwargs)
        output["refined_output"]["deformation_active_sequence"] = torch.tensor([0.0, 1.0, 1.0])
        return output

    state["model"].forward = nonboolean
    with pytest.raises(RuntimeError, match="fixed pose-only"):
        trainer.train_staged_step_v6(state, _batch())


def test_v6_checkpoint_binds_full_catalogue_provenance_dependencies_and_raw_uncertainty(monkeypatch):
    state = _state(monkeypatch)
    trainer.train_staged_step_v6(state, _batch())
    checkpoint = trainer.make_staged_checkpoint_v6(state)
    assert trainer.verify_staged_checkpoint_v6(checkpoint, verify_sources=False)
    assert checkpoint["manifest"]["catalogue_cell_count"] == 98_304
    assert checkpoint["seen_section_ids"] == ["x0", "x1"]
    assert checkpoint["seen_synthetic_animal_ids"] == ["sa0", "sa1"]
    assert checkpoint["provenance_records"][0]["training_row_receipt_sha256"] == "a" * 64
    assert checkpoint["learned_dependencies"] == {"model_weights": [], "features": [], "pseudolabels": []}
    assert checkpoint["uncertainty_status"] == "raw_uncalibrated"
    assert checkpoint["manifest"]["training_run_binding"] == _run_binding()
    assert checkpoint["manifest"]["release_qualifying"] is False
    assert checkpoint["manifest"]["atlas_bytes_verified_by_trainer"] is False
    assert trainer._is_sha256(
        checkpoint["training_step_ledger"][0]["row_receipts_sha256"]
    )
    assert trainer._is_sha256(
        checkpoint["training_step_ledger"][0][
            "trainer_output_receipt_sha256"
        ]
    )

    tampered = copy.deepcopy(checkpoint)
    tampered["provenance_records"][0]["section_id"] = "changed"
    with pytest.raises(ValueError):
        trainer.verify_staged_checkpoint_v6(tampered, verify_sources=False)


def test_initializer_rejects_any_catalogue_other_than_98304(monkeypatch):
    runtime = _Runtime()
    runtime.cell_count = 98_303
    monkeypatch.setattr(trainer, "verify_complete_catalogue_runtime_v6", lambda value: True)
    with pytest.raises(ValueError, match="98,304"):
        trainer.initialize_staged_trainer_v6(runtime, 1, {}, _config(), training_run_binding=_run_binding(), device="cpu")


@pytest.mark.parametrize("binding_update", [{"device": "meta"}, {"dtype": "torch.int64"}])
def test_initializer_requires_runtime_device_and_floating_dtype_match(monkeypatch, binding_update):
    runtime = _Runtime()
    runtime.binding = {**runtime.binding, **binding_update}
    monkeypatch.setattr(trainer, "verify_complete_catalogue_runtime_v6", lambda value: True)
    with pytest.raises(ValueError, match="runtime (device|dtype)"):
        trainer.initialize_staged_trainer_v6(runtime, 1, {}, _config(), training_run_binding=_run_binding(), device="cpu")


def test_custom_joint_loss_injection_is_not_a_public_initializer_option():
    with pytest.raises(TypeError, match="joint_loss_fn"):
        trainer.initialize_staged_trainer_v6(
            _Runtime(), 1, {}, _config(),
            training_run_binding=_run_binding(), device="cpu", joint_loss_fn=lambda output: output,
        )


def test_refinement_schedule_requires_at_least_one_step(monkeypatch):
    config = _config()
    config["refinement_steps"] = 0
    config["joint_pose_only_steps"] = 0
    monkeypatch.setattr(trainer, "verify_complete_catalogue_runtime_v6", lambda value: True)
    with pytest.raises(ValueError, match="pose-only prefix"):
        trainer.initialize_staged_trainer_v6(
            _Runtime(), 1, {}, config, training_run_binding=_run_binding(), device="cpu"
        )


@pytest.mark.parametrize("model_kwargs,budget", [({}, 64), ({"cascade_max_rendered_cells_per_sample": 4}, 4)])
def test_initializer_rejects_proposal_width_above_default_or_overridden_render_budget(monkeypatch, model_kwargs, budget):
    config = _config()
    config["proposal_top_m"] = budget + 1
    config["top_k"] = 1
    monkeypatch.setattr(trainer, "verify_complete_catalogue_runtime_v6", lambda value: True)
    with pytest.raises(ValueError, match="effective v6 cascade render budget"):
        trainer.initialize_staged_trainer_v6(
            _Runtime(), 1, model_kwargs, config,
            training_run_binding=_run_binding(), device="cpu",
        )


def test_manifest_freezes_effective_render_budget(monkeypatch):
    monkeypatch.setattr(trainer, "verify_complete_catalogue_runtime_v6", lambda value: True)
    monkeypatch.setattr(trainer, "ArbitraryPlaneJointModelV6", _Model)
    state = trainer.initialize_staged_trainer_v6(
        _Runtime(), 1, {"cascade_max_rendered_cells_per_sample": 7}, _config(),
        training_run_binding=_run_binding(), device="cpu",
    )
    assert state["manifest"]["cascade_max_rendered_cells_per_sample"] == 7


def test_restore_rejects_a_different_training_run_binding(monkeypatch):
    state = _state(monkeypatch)
    checkpoint = trainer.make_staged_checkpoint_v6(state)
    different = {**_run_binding(), "atlas_binding_receipt_sha256": "4" * 64}
    with pytest.raises(ValueError, match="training-run binding differs"):
        trainer.restore_staged_trainer_v6(
            checkpoint, _Runtime(), training_run_binding=different, device="cpu"
        )


def test_checkpoint_snapshot_does_not_alias_later_live_training_state(monkeypatch):
    state = _state(monkeypatch)
    trainer.train_staged_step_v6(state, _batch())
    checkpoint = trainer.make_staged_checkpoint_v6(state)
    saved_model = checkpoint["model_state"]["logits"].clone()
    saved_optimizer_receipt = trainer._hash_json(trainer._object_receipt(checkpoint["optimizer_state"]))
    assert checkpoint["model_state"]["logits"].data_ptr() != state["model"].logits.data_ptr()

    trainer.train_staged_step_v6(state, _batch())
    assert torch.equal(checkpoint["model_state"]["logits"], saved_model)
    assert trainer._hash_json(trainer._object_receipt(checkpoint["optimizer_state"])) == saved_optimizer_receipt
    assert trainer.verify_staged_checkpoint_v6(checkpoint, verify_sources=False)
