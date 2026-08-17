import copy
import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from training import train_joint_pose_registration as trainer
from training.dense_registration_model import compose_pixel_maps, identity_pixel_map


class TinyJointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose_initializer = nn.Linear(3, 3)
        self.registrar = nn.Linear(1, 1)
        self.review_head = nn.Linear(7, 4)

    def initialize(self, pose_image):
        features = pose_image.mean((-2, -1))
        return {
            "pose": self.pose_initializer(features),
            "pose_features": features,
        }

    def refine_once(self, fixed_atlas, moving_slice, current_pose, pose_features):
        fixed_mean = fixed_atlas.mean((-2, -1))[:, :1]
        context = torch.cat((current_pose / current_pose.new_tensor((60, 1, 1)), pose_features, fixed_mean), dim=1)
        output = self.review_head(context)
        return {
            "pose": current_pose + output[:, :3],
            "compatibility_logit": output[:, 3],
        }

    def review_once(self, fixed_atlas, moving_slice, current_pose, pose_features):
        output = self.refine_once(
            fixed_atlas, moving_slice, current_pose, pose_features
        )
        return output["pose"], output["compatibility_logit"]

    def register_final_pose(
        self,
        fixed_atlas,
        moving_slice,
        final_pose,
        pose_features,
        map_domain_receipt,
    ):
        y, x = torch.meshgrid(
            torch.arange(fixed_atlas.shape[-2], device=fixed_atlas.device),
            torch.arange(fixed_atlas.shape[-1], device=fixed_atlas.device),
            indexing="ij",
        )
        pixel_map = torch.stack((x, y)).to(fixed_atlas).unsqueeze(0)
        pixel_map = pixel_map.expand(len(fixed_atlas), -1, -1, -1)
        return {
            "pose": final_pose,
            "fixed_atlas": fixed_atlas,
            "fixed_to_source_model_map": pixel_map,
            "map_space": "source-model-canvas",
            "map_domain_receipt": map_domain_receipt,
        }

    def set_training_stage(self, stage):
        if stage not in {"review", "geometry", "joint"}:
            raise ValueError(stage)
        for parameter in self.parameters():
            parameter.requires_grad_(stage == "joint")
        if stage != "joint":
            for parameter in self.review_head.parameters():
                parameter.requires_grad_(True)
        if stage == "geometry":
            for module in (self.pose_initializer, self.registrar):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)


def tiny_batch(seed=1, batch_size=2, candidates=2):
    generator = torch.Generator().manual_seed(int(seed) % (2**31))
    height = width = 16
    true_pose = torch.randn(batch_size, 3, generator=generator)
    pose_image = torch.randn(batch_size, 3, height, width, generator=generator)
    fixed = torch.ones(batch_size, 1, height, width)
    mask = torch.ones_like(fixed, dtype=torch.bool)
    moving = torch.randn(batch_size, 1, height, width, generator=generator)
    initial_pose = true_pose + torch.tensor((25.0, 0.5, -0.5))
    wrong_pose = true_pose[:, None] + torch.randn(
        batch_size, candidates, 3, generator=generator
    )
    wrong_fixed = torch.zeros(batch_size, candidates, 1, height, width)
    return {
        "pose_image": pose_image,
        "true_pose": true_pose,
        "initial_pose": initial_pose,
        "wrong_candidate_pose": wrong_pose,
        "wrong_candidate_fixed": wrong_fixed,
        "wrong_candidate_fixed_mask": torch.ones_like(wrong_fixed, dtype=torch.bool),
        "wrong_candidate_dense_target_valid": torch.zeros(
            batch_size, candidates, dtype=torch.bool
        ),
        "true_dense_target_valid": torch.ones(batch_size, dtype=torch.bool),
        "fixed": fixed,
        "fixed_mask": mask,
        "fixed_visible_mask": mask,
        "fixed_damage_mask": torch.zeros_like(mask),
        "initial_fixed": fixed * 0.5,
        "initial_fixed_mask": mask,
        "moving": moving,
        "moving_model_mask": mask,
        "moving_tissue_mask": mask,
        "moving_damage_mask": torch.zeros_like(mask),
        "moving_visible_mask": mask,
        "fixed_to_moving": torch.zeros(batch_size, 2, height, width),
        "moving_to_fixed": torch.zeros(batch_size, 2, height, width),
        "similarity_h": torch.eye(3).expand(batch_size, -1, -1).clone(),
        "fixed_labels": torch.ones(batch_size, 1, height, width, dtype=torch.long),
        "moving_labels": torch.ones(batch_size, 1, height, width, dtype=torch.long),
        "local_velocity": torch.zeros(batch_size, 2, height, width),
    }


def tiny_dense_loss(registrar, batch):
    assert not any(name.startswith("wrong_candidate") for name in batch)
    loss = sum(parameter.square().mean() for parameter in registrar.parameters())
    return loss, {"total": float(loss.detach())}, {"teacher_batch": batch}


def tiny_render_pose(pose):
    value = (pose[:, :1] / 60.0 + pose[:, 1:2] + pose[:, 2:3]).view(-1, 1, 1, 1)
    image = value.expand(-1, 1, 16, 16)
    mask = torch.ones_like(image, dtype=torch.bool)
    labels = torch.ones_like(image, dtype=torch.long)
    return image, mask, labels


def objective(model, batch, **kwargs):
    dense_loss_fn = kwargs.pop("dense_loss_fn", tiny_dense_loss)
    return trainer.joint_objective(
        model,
        batch,
        render_pose=tiny_render_pose,
        refinement_steps=2,
        live_initializer_fraction=1.0,
        candidate_chunk_size=2,
        gradient_checkpointing=False,
        dense_loss_fn=dense_loss_fn,
        **kwargs,
    )


def test_warm_start_loads_pose_model_and_dense_ema_with_hash_receipt(tmp_path):
    source = TinyJointModel()
    target = TinyJointModel()
    with torch.no_grad():
        for parameter in source.pose_initializer.parameters():
            parameter.fill_(0.25)
        for parameter in source.registrar.parameters():
            parameter.fill_(-0.5)
    pose = tmp_path / "pose.pt"
    dense = tmp_path / "dense.pt"
    torch.save({"model": source.pose_initializer.state_dict()}, pose)
    torch.save({"ema": {"shadow": source.registrar.state_dict()}}, dense)

    receipt = trainer.warm_start_model(target, pose, dense)

    assert receipt["pose"]["state"] == "model"
    assert receipt["dense"]["state"] == "ema.shadow"
    assert receipt["pose"]["sha256"] == trainer.sha256_file(pose)
    assert receipt["dense"]["sha256"] == trainer.sha256_file(dense)
    assert all(
        torch.equal(source.pose_initializer.state_dict()[name], value)
        for name, value in target.pose_initializer.state_dict().items()
    )
    assert all(
        torch.equal(source.registrar.state_dict()[name], value)
        for name, value in target.registrar.state_dict().items()
    )


def test_joint_objective_never_exposes_wrong_plane_to_dense_supervision():
    model = TinyJointModel()
    batch = tiny_batch(candidates=3)
    calls = []

    def recording_dense_loss(registrar, teacher_batch):
        calls.append(teacher_batch)
        return tiny_dense_loss(registrar, teacher_batch)

    loss, terms, outputs = objective(
        model, batch, dense_loss_fn=recording_dense_loss
    )
    loss.backward()

    assert len(calls) == 1
    assert set(calls[0]) <= set(trainer._TRUE_DENSE_KEYS)
    assert outputs["candidates"]["refined_pose"].shape == (2, 5, 3)
    assert outputs["candidates"]["compatibility_logits"].shape == (2, 5)
    assert set(trainer.DEFAULT_LOSS_WEIGHTS) <= set(terms)
    assert model.review_head.weight.grad is not None
    metrics = trainer._batch_metrics(batch, outputs)
    assert metrics["end_to_end_region_correspondence"] == 1.0
    assert metrics["end_to_end_macro_region_dice"] == 1.0


def test_review_training_skips_constant_dense_loss_without_changing_reviewer_gradients():
    torch.manual_seed(731)
    skipped_model = TinyJointModel()
    reference_model = copy.deepcopy(skipped_model)
    trainer.apply_training_stage(skipped_model, "review")
    trainer.apply_training_stage(reference_model, "review")
    batch = tiny_batch(seed=732)
    calls = []

    def constant_dense_loss(registrar, teacher_batch):
        calls.append(teacher_batch)
        constant = teacher_batch["fixed"].new_tensor(4.25)
        return constant, {"constant": constant}, {"constant": constant}

    skipped_loss, skipped_terms, _ = objective(
        skipped_model, batch, dense_loss_fn=constant_dense_loss
    )
    reference_loss, _, _ = trainer.pose_review_objective(
        reference_model,
        batch,
        render_pose=tiny_render_pose,
        refinement_steps=2,
        live_initializer_fraction=1.0,
        candidate_chunk_size=2,
        gradient_checkpointing=False,
    )
    reference_loss = reference_loss + reference_loss.new_tensor(4.25)
    skipped_loss.backward()
    reference_loss.backward()

    assert calls == []
    assert skipped_terms["dense_skipped"] == 1.0
    assert "dense" not in skipped_terms
    for skipped, reference in zip(
        skipped_model.review_head.parameters(), reference_model.review_head.parameters()
    ):
        torch.testing.assert_close(skipped.grad, reference.grad, rtol=0.0, atol=0.0)

    skipped_model.eval()
    with torch.no_grad():
        _, evaluation_terms, _ = objective(
            skipped_model, batch, dense_loss_fn=constant_dense_loss
        )
    assert len(calls) == 1
    assert evaluation_terms["dense_skipped"] == 0.0
    assert evaluation_terms["dense"] == pytest.approx(4.25)


def test_training_can_skip_discarded_final_registration_without_changing_gradients():
    torch.manual_seed(733)
    full_model = TinyJointModel()
    skipped_model = copy.deepcopy(full_model)
    batch = tiny_batch(seed=734)

    full_loss, _, full_outputs = objective(full_model, batch)
    skipped_loss, _, skipped_outputs = objective(
        skipped_model,
        batch,
        compute_final_registration=False,
    )
    full_loss.backward()
    skipped_loss.backward()

    torch.testing.assert_close(full_loss, skipped_loss, rtol=0.0, atol=0.0)
    assert "final_registration" in full_outputs["recurrent"]
    assert "final_registration" not in skipped_outputs["recurrent"]
    for full, skipped in zip(full_model.parameters(), skipped_model.parameters()):
        torch.testing.assert_close(full.grad, skipped.grad, rtol=0.0, atol=0.0)


def test_empty_final_overlap_is_never_rewarded_as_perfect_dice():
    model = TinyJointModel()
    batch = tiny_batch(batch_size=1, candidates=1)
    _, _, outputs = objective(model, batch)
    final = outputs["recurrent"]["final_registration"]
    final["fixed_to_source_model_map"] = (
        final["fixed_to_source_model_map"].clone() + 1000.0
    )
    metrics = trainer._batch_metrics(batch, outputs)
    assert metrics["end_to_end_region_correspondence"] == 0.0
    assert metrics["end_to_end_macro_region_dice"] == 0.0
    assert metrics["end_to_end_valid_fraction"] == 0.0
    assert metrics["end_to_end_retained_coverage"] == 0.0
    assert metrics["coverage_failure_fraction"] == 1.0
    assert metrics["invalid_endpoint_fraction"] == 1.0


def test_tiny_high_agreement_patch_is_penalized_by_retained_coverage():
    model = TinyJointModel()
    batch = tiny_batch(batch_size=1, candidates=1)
    batch["moving_visible_mask"].zero_()
    batch["moving_visible_mask"][:, :, 8, 8] = True
    _, _, outputs = objective(model, batch)
    metrics = trainer._batch_metrics(batch, outputs)
    assert metrics["end_to_end_region_correspondence"] == 1.0
    assert metrics["end_to_end_macro_region_dice"] == 1.0
    assert metrics["end_to_end_retained_coverage"] < 0.01
    assert metrics["coverage_failure_fraction"] == 1.0


def test_outline_normalization_composes_exact_dense_maps_roundtrip():
    batch = tiny_batch(batch_size=1, candidates=1)
    y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    source = (((x - 7.0) / 6.0) ** 2 + ((y - 7.0) / 5.0) ** 2 <= 1)[None, None]
    target = (((x - 8.0) / 7.0) ** 2 + ((y - 8.0) / 6.0) ** 2 <= 1)[None, None]
    batch["moving_model_mask"] = source
    batch["moving_tissue_mask"] = source
    batch["moving_visible_mask"] = source
    batch["moving_damage_mask"] = torch.zeros_like(source)
    batch["fixed_mask"] = target
    batch["fixed_visible_mask"] = target
    batch["fixed_damage_mask"] = torch.zeros_like(target)
    identity = identity_pixel_map(1, 16, 16)
    batch["fixed_to_moving"] = identity.clone()
    batch["moving_to_fixed"] = identity.clone()
    batch["similarity_h"] = torch.eye(3)[None]

    normalized = trainer.normalize_synthetic_dense_contract(batch)
    roundtrip = compose_pixel_maps(
        normalized["fixed_to_moving"], normalized["moving_to_fixed"]
    )
    forward = normalized["fixed_to_moving"]
    in_bounds = target[:, 0] & (forward[:, 0] > 0.5) & (forward[:, 0] < 14.5)
    in_bounds &= (forward[:, 1] > 0.5) & (forward[:, 1] < 14.5)
    assert torch.max((roundtrip - identity).abs()[in_bounds[:, None].expand_as(roundtrip)]) < 2e-4
    assert torch.allclose(
        normalized["fixed_to_moving"],
        trainer.apply_homography_to_map(normalized["similarity_h"], identity),
        atol=1e-5,
    )


def test_each_wrong_pose_gets_its_own_outline_scale_and_center():
    batch = tiny_batch(batch_size=1, candidates=1)
    y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    source = (((x - 7.0) / 6.0) ** 2 + ((y - 7.0) / 5.0) ** 2 <= 1)[None, None]
    targets = torch.stack(
        (
            (((x - 6.0) / 6.0) ** 2 + ((y - 7.0) / 5.0) ** 2 <= 1),
            # Deliberately different scale and center, like a wrong atlas plane.
            (((x - 9.0) / 7.0) ** 2 + ((y - 9.0) / 6.0) ** 2 <= 1),
        )
    )[:, None]
    batch["moving_model_mask"] = source
    batch["_outline_source_mask"] = source
    batch["_outline_source_moving"] = batch["moving"]
    _, aligned, receipt = trainer.prepare_moving_for_fixed(batch, targets)
    assert not torch.allclose(
        receipt["source_to_aligned_h"][0], receipt["source_to_aligned_h"][1]
    )
    for observed, expected in zip(aligned[:, 0], targets[:, 0]):
        observed_y, observed_x = torch.where(observed)
        expected_y, expected_x = torch.where(expected)
        assert abs(float(observed_x.float().mean() - expected_x.float().mean())) < 0.6
        assert abs(float(observed_y.float().mean() - expected_y.float().mean())) < 0.6
        assert abs(int(observed_x.max() - observed_x.min()) - int(expected_x.max() - expected_x.min())) <= 1
        assert abs(int(observed_y.max() - observed_y.min()) - int(expected_y.max() - expected_y.min())) <= 1


def test_final_anatomy_metric_does_not_double_compose_source_canvas_map():
    model = TinyJointModel()
    batch = tiny_batch(batch_size=1, candidates=1)
    _, _, outputs = objective(model, batch)
    identity = identity_pixel_map(1, 16, 16)
    homography = torch.tensor(
        [[[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]]
    )
    batch["moving_labels"].zero_()
    batch["moving_visible_mask"].zero_()
    batch["_outline_source_labels"] = torch.ones_like(batch["moving_labels"])
    batch["_outline_source_visible_mask"] = torch.ones_like(
        batch["moving_visible_mask"]
    )
    final = outputs["recurrent"]["final_registration"]
    final["fixed_labels"] = torch.ones_like(final["fixed_labels"])
    final["fixed_to_source_model_map"] = identity
    final["map_domain_receipt"] = {
        "map_pose": outputs["recurrent"]["pose"],
        "source_to_aligned_h": homography,
        "source_shape": (16, 16),
        "map_space": "source-model-canvas",
    }
    metrics = trainer._batch_metrics(batch, outputs)
    assert metrics["end_to_end_region_correspondence"] == 1.0
    assert metrics["end_to_end_macro_region_dice"] == 1.0
    assert metrics["end_to_end_retained_coverage"] == 1.0


def test_recurrent_rollout_uses_live_initializer_and_binds_maps_to_final_pose():
    model = TinyJointModel()
    batch = tiny_batch(batch_size=1, candidates=1)
    initialization = model.initialize(batch["pose_image"])
    rollout = trainer.recurrent_training_rollout(
        model,
        batch,
        initialization,
        tiny_render_pose,
        refinement_steps=3,
        live_initializer_fraction=1.0,
        gradient_checkpointing=False,
    )
    assert rollout["pose_sequence"].shape == (1, 3, 3)
    expected_fixed, expected_mask, _ = tiny_render_pose(rollout["pose"].detach())
    assert torch.equal(
        rollout["final_registration"]["fixed_atlas"],
        trainer._two_channel(expected_fixed, expected_mask),
    )
    assert rollout["final_registration"]["map_space"] == "source-model-canvas"
    assert rollout["final_registration"]["map_domain_receipt"]["source_shape"] == (
        16,
        16,
    )
    trainer.deep_pose_loss(rollout["pose_sequence"], batch["true_pose"]).backward()
    assert model.pose_initializer.weight.grad is not None
    assert model.pose_initializer.weight.grad.abs().sum() > 0


def test_shared_refiner_learns_through_more_than_one_recurrent_step():
    class GainRefiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.tensor(0.0))

        def refine_once(self, fixed, moving, pose, features):
            return {
                "pose": pose * (1.0 - self.gain),
                "compatibility_logit": pose[:, 0] * 0.0,
            }

        def review_once(self, fixed, moving, pose, features):
            output = self.refine_once(fixed, moving, pose, features)
            return output["pose"], output["compatibility_logit"]

        def register_final_pose(self, fixed, moving, pose, features, receipt):
            return {"pose": pose, "fixed_atlas": fixed}

    model = GainRefiner()
    batch = tiny_batch(batch_size=1, candidates=1)
    batch["true_pose"].zero_()
    batch["initial_pose"].fill_(2.0)
    initialization = {"pose": batch["initial_pose"], "pose_features": torch.zeros(1, 3)}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        rollout = trainer.recurrent_training_rollout(
            model,
            batch,
            initialization,
            tiny_render_pose,
            refinement_steps=2,
            live_initializer_fraction=0.0,
            gradient_checkpointing=False,
        )
        loss = trainer.deep_pose_loss(rollout["pose_sequence"], batch["true_pose"])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_wrong_candidate_dense_validity_is_a_hard_failure():
    batch = tiny_batch()
    batch["wrong_candidate_dense_target_valid"][0, 0] = True
    with pytest.raises(ValueError, match="cannot carry dense-flow"):
        objective(TinyJointModel(), batch)


def test_deep_pose_loss_supervises_every_step():
    target = torch.zeros(1, 3)
    prediction = torch.tensor([[[60.0, 0.0, 0.0], [0.0, 0.9, 0.0]]], requires_grad=True)
    loss = trainer.deep_pose_loss(prediction, target)
    loss.backward()
    assert prediction.grad[0, 0, 0] != 0
    assert prediction.grad[0, 1, 1] != 0


def test_plane_anchor_penalizes_equal_normalized_error_by_whole_plane_displacement():
    target = torch.zeros(1, 3)
    ap_error = torch.tensor([[trainer.PHYSICAL_POSE_LOSS_SCALE[0], 0.0, 0.0]])
    lr_error = torch.tensor([[0.0, trainer.PHYSICAL_POSE_LOSS_SCALE[1], 0.0]])
    assert trainer.normalized_pose_loss(ap_error, target) == pytest.approx(
        trainer.normalized_pose_loss(lr_error, target)
    )
    assert trainer.quicknii_plane_anchor_loss(lr_error, target) > trainer.quicknii_plane_anchor_loss(
        ap_error, target
    )


def test_training_stages_freeze_and_unfreeze_declared_parts():
    model = TinyJointModel()
    review_count = trainer.apply_training_stage(model, "review")
    assert review_count == sum(parameter.numel() for parameter in model.review_head.parameters())
    assert not any(parameter.requires_grad for parameter in model.pose_initializer.parameters())
    geometry_count = trainer.apply_training_stage(model, "geometry")
    assert geometry_count > review_count
    joint_count = trainer.apply_training_stage(model, "joint")
    assert joint_count == sum(parameter.numel() for parameter in model.parameters())


def test_stagewise_artifact_sampling_is_deterministic_and_uses_declared_mix():
    first = np.random.default_rng(219)
    second = np.random.default_rng(219)
    expected = [
        trainer.sample_training_stratum(
            first, "joint", trainer.DEFAULT_STRATUM_PROBABILITIES
        )
        for _ in range(20)
    ]
    observed = [
        trainer.sample_training_stratum(
            second, "joint", trainer.DEFAULT_STRATUM_PROBABILITIES
        )
        for _ in range(20)
    ]
    assert observed == expected
    assert trainer.DEFAULT_STRATUM_PROBABILITIES["review"] == (0.30, 0.50, 0.20)
    assert trainer.DEFAULT_STRATUM_PROBABILITIES["geometry"] == (0.15, 0.50, 0.35)
    assert trainer.DEFAULT_STRATUM_PROBABILITIES["joint"] == (0.10, 0.45, 0.45)


def test_high_tilt_retention_is_deterministic_spans_domain_and_keeps_exact_dense():
    first_data = TinyHighTiltData()
    second_data = TinyHighTiltData()
    first = trainer.generate_high_tilt_retention_batch(
        first_data, 8, "train", 7182, "hard", 3
    )
    second = trainer.generate_high_tilt_retention_batch(
        second_data, 8, "train", 7182, "hard", 3
    )
    assert torch.equal(first["true_pose"], second["true_pose"])
    absolute_tilt = first["true_pose"][:, 1:].abs()
    assert float(absolute_tilt.max()) <= 35.0
    regimes = first["high_tilt_regime"]
    assert set(regimes) == {"lr_only", "dv_only", "both"}
    for values, regime in zip(absolute_tilt, regimes):
        if regime in {"lr_only", "both"}:
            assert float(values[0]) >= 15.0
        else:
            assert float(values[0]) <= 15.0
        if regime in {"dv_only", "both"}:
            assert float(values[1]) >= 15.0
        else:
            assert float(values[1]) <= 15.0
    assert first["true_dense_target_valid"].all()
    offsets = first_data.last_manifest["wrong_candidate_offset"]
    assert np.all(np.abs(offsets[:, 0, 0]) == 25.0)
    assert np.all(np.abs(offsets[:, 1, 1]) == 0.25)
    assert np.all(np.abs(offsets[:, 2, 2]) == 0.25)


class TinyData:
    contract = {"tiny": "v1"}

    def generate(self, count, split, seed, stratum, negatives_per_sample, **kwargs):
        return tiny_batch(seed, count, negatives_per_sample)

    render_pose = staticmethod(tiny_render_pose)


class TinyHighTiltData:
    def __init__(self):
        self.last_manifest = None

    render_pose = staticmethod(tiny_render_pose)

    def make_manifest(self, count, split, seed, stratum, negatives_per_sample):
        wrong = np.zeros((count, negatives_per_sample, 3), np.float32)
        wrong[:, 0, 0] = 25.0
        if negatives_per_sample > 1:
            wrong[:, 1, 1] = 0.25
        if negatives_per_sample > 2:
            wrong[:, 2, 2] = 0.25
        return {
            "true_pose": np.zeros((count, 3), np.float32),
            "initial_pose_offset": np.tile((25.0, 0.5, -0.5), (count, 1)).astype(np.float32),
            "wrong_candidate_offset": wrong,
            "registration__tilt_lr_deg": np.zeros(count, np.float32),
            "registration__tilt_dv_deg": np.zeros(count, np.float32),
            "registration__manifest_sha256": "old",
            "registration_manifest_sha256": "old",
            "joint_manifest_sha256": "old",
        }

    def batch(self, manifest, **kwargs):
        self.last_manifest = copy.deepcopy(manifest)
        count = len(manifest["true_pose"])
        candidates = manifest["wrong_candidate_offset"].shape[1]
        batch = tiny_batch(311, count, candidates)
        true_pose = torch.from_numpy(manifest["true_pose"].copy())
        batch.update(
            true_pose=true_pose,
            initial_pose=true_pose
            + torch.from_numpy(manifest["initial_pose_offset"].copy()),
            wrong_candidate_pose=true_pose[:, None]
            + torch.from_numpy(manifest["wrong_candidate_offset"].copy()),
        )
        identity = identity_pixel_map(count, 16, 16)
        batch["fixed_to_moving"] = identity
        batch["moving_to_fixed"] = identity
        return batch


def test_high_tilt_k6_initial_and_wrong_poses_stay_in_canonical_domain_many_seeds():
    for seed in range(64):
        batch = trainer.generate_high_tilt_retention_batch(
            TinyHighTiltData(), 8, "train", seed, "hard", 6
        )
        poses = torch.cat(
            (batch["initial_pose"][:, None], batch["wrong_candidate_pose"]), dim=1
        )
        assert torch.all((-4500.0 <= poses[..., 0]) & (poses[..., 0] <= 500.0))
        assert torch.all((-35.0 <= poses[..., 1:]) & (poses[..., 1:] <= 35.0))


def test_high_tilt_exact_pair_calls_dense_loss_without_wrong_plane_targets():
    data = TinyHighTiltData()
    batch = trainer.generate_high_tilt_retention_batch(
        data, 2, "train", 8123, "hard", 3
    )
    calls = []

    def recording_dense_loss(registrar, teacher_batch):
        calls.append(teacher_batch)
        assert not any(name.startswith("wrong_candidate") for name in teacher_batch)
        return tiny_dense_loss(registrar, teacher_batch)

    _, terms, _ = trainer.joint_objective(
        TinyJointModel(),
        batch,
        render_pose=tiny_render_pose,
        refinement_steps=2,
        live_initializer_fraction=1.0,
        candidate_chunk_size=1,
        gradient_checkpointing=False,
        prepare_moving=trainer.prepare_moving_for_fixed,
        dense_loss_fn=recording_dense_loss,
    )
    assert len(calls) == 1
    assert "dense" in terms


def test_high_tilt_validation_reports_dense_endpoint_by_regime_and_stratum():
    report = trainer.evaluate_high_tilt_retention(
        TinyJointModel(),
        TinyHighTiltData(),
        count_per_stratum=1,
        batch_size=1,
        seed=9123,
        negatives_per_sample=3,
        refinement_steps=1,
        candidate_chunk_size=1,
        dense_loss_fn=tiny_dense_loss,
    )
    assert set(report["by_regime"]) == {"lr_only", "dv_only", "both"}
    assert set(report["by_artifact_stratum"]) == {"clean", "mild", "hard"}
    for group in (report["by_regime"], report["by_artifact_stratum"]):
        assert all("dense" in summary for summary in group.values())
        assert all(
            "end_to_end_region_correspondence" in summary
            for summary in group.values()
        )


def tiny_registered_batch(seed=1, batch_size=1, candidates=1):
    batch = tiny_batch(seed, batch_size, candidates)
    for name in (
        "true_dense_target_valid",
        "wrong_candidate_dense_target_valid",
        "fixed_to_moving",
        "moving_to_fixed",
        "similarity_h",
        "moving_labels",
        "local_velocity",
        "moving_damage_mask",
    ):
        batch.pop(name)
    batch["source"] = "allen_registered_product5"
    return batch


class TinyRegisteredData:
    def __init__(self, split="train"):
        self.contract = {"registered": f"product5-{split}-v1"}
        self.calls = 0

    def generate(self, count, seed, negatives_per_sample):
        self.calls += 1
        return tiny_registered_batch(seed, count, negatives_per_sample)

    render_pose = staticmethod(tiny_render_pose)

    @staticmethod
    def moving_for_fixed(batch, target_mask):
        repeats = len(target_mask) // len(batch["moving"])
        return (
            batch["moving"].repeat_interleave(repeats, dim=0),
            batch["moving_model_mask"].repeat_interleave(repeats, dim=0),
        )


def tiny_registered_evaluation(model, data, **kwargs):
    value = float(sum(parameter.square().sum() for parameter in model.parameters()).detach())
    return {
        "selection_score": -value,
        "ap_mae_um": value,
        "lr_mae_deg": value,
        "dv_mae_deg": value,
        "initial_ap_mae_um": value,
        "initial_lr_mae_deg": value,
        "initial_dv_mae_deg": value,
        "ranking_accuracy": 1.0,
        "count": 1,
        "role": "development_consumed_product5_validation",
    }


def tiny_evaluation(model, data, **kwargs):
    value = float(sum(parameter.square().sum() for parameter in model.parameters()).detach())
    return {
        "selection_score": -value,
        "ap_mae_um": value,
        "lr_mae_deg": value,
        "dv_mae_deg": value,
        "ranking_accuracy": 1.0,
        "dense": value,
    }


def training_config(tmp_path, run_name):
    return {
        "workspace": str(tmp_path),
        "run_name": run_name,
        "device": "cpu",
        "seed": 17,
        "data_seed": 19,
        "total_views": 4,
        "batch_size": 1,
        "negatives_per_sample": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "scheduler_warmup_views": 1,
        "gradient_clip": 10.0,
        "ema_decay": 0.9,
        "amp": False,
        "amp_initial_scale": 1024.0,
        "validation_every_views": 4,
        "validation_count_per_stratum": 1,
        "validation_batch_size": 1,
        "validation_seed": 23,
        "progress_every_seconds": 0.0,
        "checkpoint_every_views": 2,
        "refinement_steps": 2,
        "candidate_chunk_size": 1,
        "gradient_checkpointing": False,
        "live_initializer_fraction_by_stage": {
            "review": 1.0,
            "geometry": 1.0,
            "joint": 1.0,
        },
        "registered_fraction_by_stage": {
            "review": 0.0,
            "geometry": 0.0,
            "joint": 0.0,
        },
        "high_tilt_fraction_by_stage": {
            "review": 0.0,
            "geometry": 0.0,
            "joint": 0.0,
        },
        "high_tilt_validation_count_per_stratum": 0,
        "stages": [{"name": "joint", "until_views": 4}],
        "resume": True,
    }


def test_amp_initial_scale_is_normalized_and_validated(tmp_path):
    config = training_config(tmp_path, "amp-scale")
    assert trainer._normalized_config(config)["amp_initial_scale"] == 1024.0

    config["amp_initial_scale"] = 0.0
    with pytest.raises(ValueError, match="AMP initial scale"):
        trainer._normalized_config(config)


def test_interrupted_resume_reproduces_cpu_state_and_rng_continuation(tmp_path):
    initial = TinyJointModel().state_dict()
    direct = TinyJointModel()
    direct.load_state_dict(copy.deepcopy(initial))
    resumed = TinyJointModel()
    resumed.load_state_dict(copy.deepcopy(initial))

    direct_config = training_config(tmp_path, "direct")
    direct_config["registered_fraction_by_stage"] = {
        "review": 0.5,
        "geometry": 0.5,
        "joint": 0.5,
    }
    direct_registered = TinyRegisteredData()
    trainer.train(
        direct_config,
        model=direct,
        data=TinyData(),
        registered_data=direct_registered,
        registered_validation_data=TinyRegisteredData("validation"),
        dense_loss_fn=tiny_dense_loss,
        evaluation_fn=tiny_evaluation,
        registered_evaluation_fn=tiny_registered_evaluation,
    )
    interrupted_config = training_config(tmp_path, "resumed")
    interrupted_config["registered_fraction_by_stage"] = dict(
        direct_config["registered_fraction_by_stage"]
    )
    interrupted_config["stop_after_views"] = 2
    first_registered = TinyRegisteredData()
    interrupted = trainer.train(
        interrupted_config,
        model=resumed,
        data=TinyData(),
        registered_data=first_registered,
        registered_validation_data=TinyRegisteredData("validation"),
        dense_loss_fn=tiny_dense_loss,
        evaluation_fn=tiny_evaluation,
        registered_evaluation_fn=tiny_registered_evaluation,
    )
    assert json.loads((interrupted.parent / "progress.json").read_text())["status"] == "interrupted"

    fresh = TinyJointModel()
    resumed_config = training_config(tmp_path, "resumed")
    resumed_config["registered_fraction_by_stage"] = dict(
        direct_config["registered_fraction_by_stage"]
    )
    second_registered = TinyRegisteredData()
    trainer.train(
        resumed_config,
        model=fresh,
        data=TinyData(),
        registered_data=second_registered,
        registered_validation_data=TinyRegisteredData("validation"),
        dense_loss_fn=tiny_dense_loss,
        evaluation_fn=tiny_evaluation,
        registered_evaluation_fn=tiny_registered_evaluation,
    )
    direct_state = trainer.load_checkpoint(tmp_path / "runs" / "direct" / "latest.pt")["model"]
    resumed_state = trainer.load_checkpoint(tmp_path / "runs" / "resumed" / "latest.pt")["model"]
    assert direct_state.keys() == resumed_state.keys()
    assert all(torch.equal(direct_state[name], resumed_state[name]) for name in direct_state)
    progress = json.loads((tmp_path / "runs" / "resumed" / "progress.json").read_text())
    assert progress["status"] == "complete"
    assert (tmp_path / "runs" / "resumed" / "training.log").is_file()
    assert (tmp_path / "runs" / "resumed" / "checkpoints" / "views-000000002.pt").is_file()
    assert direct_registered.calls > 0
    assert first_registered.calls + second_registered.calls == direct_registered.calls


def test_non_batch_aligned_interruption_is_rejected_before_truncating_manifest(tmp_path):
    config = training_config(tmp_path, "nonaligned")
    config.update(batch_size=2, stop_after_views=1)
    with pytest.raises(ValueError, match="complete training batch"):
        trainer.train(
            config,
            model=TinyJointModel(),
            data=TinyData(),
            dense_loss_fn=tiny_dense_loss,
            evaluation_fn=tiny_evaluation,
        )


def test_registered_objective_has_joint_gradients_and_no_dense_term():
    model = TinyJointModel()
    batch = tiny_registered_batch(batch_size=2, candidates=2)
    loss, terms, _ = trainer.registered_objective(
        model,
        batch,
        render_pose=tiny_render_pose,
        refinement_steps=2,
        live_initializer_fraction=1.0,
        candidate_chunk_size=1,
        gradient_checkpointing=False,
    )
    loss.backward()
    assert "dense" not in terms
    assert model.pose_initializer.weight.grad is not None
    assert model.review_head.weight.grad is not None


def test_early_stopping_is_driven_only_by_validation_scores(tmp_path):
    scores = iter((1.0, 0.9, 0.8))

    def declining_validation(model, data, **kwargs):
        score = next(scores)
        return {
            "selection_score": score,
            "ap_mae_um": 1.0,
            "lr_mae_deg": 0.1,
            "dv_mae_deg": 0.1,
            "ranking_accuracy": 1.0,
            "dense": 0.1,
        }

    config = training_config(tmp_path, "early")
    config.update(
        total_views=6,
        validation_every_views=1,
        early_stopping_patience_validations=2,
        stages=[{"name": "joint", "until_views": 6}],
    )
    best = trainer.train(
        config,
        model=TinyJointModel(),
        data=TinyData(),
        dense_loss_fn=tiny_dense_loss,
        evaluation_fn=declining_validation,
    )
    progress = json.loads((best.parent / "progress.json").read_text())
    latest = trainer.load_checkpoint(best.parent / "latest.pt")
    assert progress["status"] == "early_stopped"
    assert progress["completed_views"] == 3
    assert latest["validation_checkpoints_without_improvement"] == 2
    assert trainer.load_checkpoint(best)["completed_views"] == 1


def test_micro_overfit_hook_reduces_a_fixed_batch_objective():
    model = TinyJointModel()
    history = trainer.micro_overfit(
        model,
        tiny_batch(batch_size=1, candidates=1),
        render_pose=tiny_render_pose,
        steps=12,
        learning_rate=5e-3,
        dense_loss_fn=tiny_dense_loss,
        normalize_outline=False,
    )
    assert min(history[-3:]) < history[0]


def test_micro_overfit_supports_review_stage():
    model = TinyJointModel()
    history = trainer.micro_overfit(
        model,
        tiny_batch(batch_size=1, candidates=1),
        render_pose=tiny_render_pose,
        stage="review",
        steps=4,
        learning_rate=5e-3,
        dense_loss_fn=tiny_dense_loss,
        normalize_outline=False,
    )
    assert len(history) == 4
    assert all(parameter.requires_grad for parameter in model.review_head.parameters())
    assert not any(parameter.requires_grad for parameter in model.pose_initializer.parameters())
    assert not any(parameter.requires_grad for parameter in model.registrar.parameters())


def test_validation_score_uses_only_validation_report():
    report = {
        "ap_mae_um": 60.0,
        "lr_mae_deg": 0.9,
        "dv_mae_deg": 1.75,
        "ranking_accuracy": 0.75,
        "dense": 0.5,
        "end_to_end_region_correspondence": 1.0,
        "end_to_end_macro_region_dice": 1.0,
    }
    assert trainer.validation_score(report) == pytest.approx(-3.75)


def test_product5_regression_lowers_combined_checkpoint_selection_score():
    synthetic = {"selection_score": -2.0, "ap_mae_um": 10.0}
    good_real = {"selection_score": -1.0}
    bad_real = {"selection_score": -5.0}
    good = trainer.combine_validation_reports(synthetic, good_real, 0.5)
    bad = trainer.combine_validation_reports(synthetic, bad_real, 0.5)
    assert bad["selection_score"] < good["selection_score"]
    assert good["registered_product5"] is good_real


def test_pose_tail_report_declares_p95_and_catastrophic_failure_rate():
    errors = np.zeros((20, 3), np.float32)
    errors[-1] = (300.0, 6.0, 6.0)
    report = trainer._pose_error_tail_report(errors)
    assert report["pose_p95_supported"]
    assert report["catastrophic_pose_failure_rate"] == pytest.approx(0.05)
    assert report["ap_p95_um"] == pytest.approx(15.0)
    assert report["lr_p95_deg"] == pytest.approx(0.3)
    assert report["dv_p95_deg"] == pytest.approx(0.3)


def test_tail_and_worst_stratum_regression_cannot_win_on_a_better_mean():
    unreliable = {
        "ap_p95_um": 300.0,
        "lr_p95_deg": 6.0,
        "dv_p95_deg": 6.0,
        "catastrophic_pose_failure_rate": 0.10,
    }
    reliable = {
        "ap_p95_um": 60.0,
        "lr_p95_deg": 0.9,
        "dv_p95_deg": 1.75,
        "catastrophic_pose_failure_rate": 0.0,
    }
    unreliable_score, penalties = trainer.robust_validation_selection_score(
        -1.0, unreliable, [-4.0, -1.0, -1.0]
    )
    reliable_score, _ = trainer.robust_validation_selection_score(
        -1.5, reliable, [-1.6, -1.5, -1.4]
    )
    assert unreliable_score < reliable_score
    assert penalties["pose_p95"] > 0.0
    assert penalties["catastrophic_pose_failures"] > 0.0
    assert penalties["worst_group_gap"] > 0.0


def test_nonfinite_validation_score_is_rejected():
    report = {
        "ap_mae_um": math.nan,
        "lr_mae_deg": 0.1,
        "dv_mae_deg": 0.1,
        "ranking_accuracy": 1.0,
        "dense": 0.1,
        "end_to_end_region_correspondence": 1.0,
        "end_to_end_macro_region_dice": 1.0,
    }
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.validation_score(report)


def test_resume_contract_binds_joint_source_and_canvas_semantics():
    contract = trainer._generator_contract(
        TinyData(), TinyRegisteredData(), TinyRegisteredData("validation")
    )
    assert contract["format_version"] == trainer.FORMAT_VERSION
    assert set(contract["source_sha256"]) == {
        "trainer",
        "model",
        "synthetic_adapter",
        "registered_adapter_and_canvas",
        "atlas_pose_models",
        "dense_registration_model",
        "dense_loss_ema_and_checkpoint",
        "atlas_pose_preprocessing",
        "dense_registration_preprocessing",
    }
    assert all(len(value) == 64 for value in contract["source_sha256"].values())
    assert set(contract["preprocessing_contract"]) == {
        "atlas_pose_version",
        "atlas_pose_sha256",
        "dense_registration_version",
        "dense_mask_sha256",
    }


@pytest.mark.parametrize(
    "dependency",
    (
        "atlas_pose_models",
        "dense_registration_model",
        "dense_loss_ema_and_checkpoint",
        "atlas_pose_preprocessing",
        "dense_registration_preprocessing",
    ),
)
def test_resume_rejects_each_direct_semantic_dependency_hash_change(dependency):
    contract = trainer._generator_contract(
        TinyData(), TinyRegisteredData(), TinyRegisteredData("validation")
    )
    changed = copy.deepcopy(contract)
    changed["source_sha256"][dependency] = "0" * 64
    with pytest.raises(ValueError, match="resume generator contract differs"):
        trainer._verify_resume_generator_contract(changed, contract)


def test_stage_schedule_rejects_reverse_or_repeated_stages():
    config = {
        "total_views": 4,
        "stages": [
            {"name": "joint", "until_views": 2},
            {"name": "geometry", "until_views": 4},
        ],
    }
    with pytest.raises(ValueError, match="ordered"):
        trainer._stage_schedule(config)
