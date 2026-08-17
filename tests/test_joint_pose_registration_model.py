import pytest
import torch
from torch import nn

from training.dense_registration_model import DenseRegistrationModel, identity_pixel_map
from training.joint_pose_registration_model import (
    JointInitializerExport,
    JointPoseRegistrationModel,
    JointRefinerExport,
    PoseReviewHead,
    compose_aligned_maps_to_source_model,
    project_pose_to_domain,
)


class _TinyPoseInitializer(nn.Module):
    def __init__(self, feature_count: int = 8):
        super().__init__()
        self.encoder = nn.Conv2d(3, feature_count, 1)
        self.feature_head = nn.Linear(feature_count, feature_count)
        self.pose_head = nn.Linear(feature_count, 3)
        self.orientation_head = nn.Linear(feature_count, 1)

    def training_outputs(
        self,
        image: torch.Tensor,
        include_anatomy: bool = False,
    ) -> dict[str, torch.Tensor]:
        features = self.encoder(image).mean(dim=(-2, -1))
        pooled = self.feature_head(features)
        pose = self.pose_head(pooled)
        orientation = self.orientation_head(pooled).squeeze(1)
        return {
            "pose": pose,
            "image_frame_pose": pose,
            "orientation_inverted_logit": orientation,
            "pooled_features": pooled,
        }

    def forward_with_orientation(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.training_outputs(image)
        return outputs["pose"], outputs["orientation_inverted_logit"]


def _tiny_model() -> JointPoseRegistrationModel:
    return JointPoseRegistrationModel(
        pose_initializer=_TinyPoseInitializer(),
        registrar=DenseRegistrationModel(
            channels=(4, 8),
            correlation_radii=(1, 1),
            integration_steps=1,
        ),
        pose_feature_count=8,
        review_channels=(4, 8),
        review_hidden_features=16,
    )


def _alignment_receipt(
    pose: torch.Tensor,
    homography: torch.Tensor | None = None,
    source_shape: tuple[int, int] = (12, 14),
    **extra,
):
    if homography is None:
        homography = torch.eye(3, device=pose.device, dtype=pose.dtype).expand(
            pose.shape[0], -1, -1
        ).clone()
    return {
        "map_pose": pose.detach().clone(),
        "source_to_aligned_h": homography,
        "source_shape": source_shape,
        **extra,
    }


def test_joint_contract_shapes_and_zero_initialized_pose_parity():
    torch.manual_seed(31)
    model = _tiny_model().eval()
    pose_image = torch.rand(2, 3, 24, 28)
    fixed = torch.rand(2, 2, 16, 20)
    moving = torch.rand(2, 2, 16, 20)

    expected_pose, expected_orientation = model.pose_initializer.forward_with_orientation(
        pose_image
    )
    initialized = model.initialize(pose_image)
    assert torch.equal(initialized["pose"], expected_pose)
    assert torch.equal(initialized["orientation_inverted_logit"], expected_orientation)
    assert initialized["pose_features"].shape == (2, 8)
    assert initialized["pose_features"] is initialized["pooled_features"]

    refined = model.refine_once(
        fixed,
        moving,
        initialized["pose"],
        initialized["pose_features"],
    )
    assert torch.count_nonzero(refined["pose_delta"]) == 0
    assert torch.equal(refined["pose"], initialized["pose"])
    assert refined["compatibility_logit"].shape == (2,)
    assert refined["fixed_to_moving_map"].shape == (2, 2, 16, 20)
    assert refined["moving_to_fixed_map"].shape == (2, 2, 16, 20)
    assert refined["similarity_parameters"].shape == (2, 4)
    assert refined["local_velocity"].shape == (2, 2, 16, 20)
    assert refined["warped_moving_slice"].shape == moving.shape

    initializer_outputs = JointInitializerExport(model)(pose_image)
    refiner_outputs = JointRefinerExport(model)(
        fixed,
        moving,
        initialized["pose"],
        initialized["pose_features"],
    )
    assert len(initializer_outputs) == 3
    assert len(refiner_outputs) == 7
    assert torch.equal(initializer_outputs[0], expected_pose)
    assert torch.equal(refiner_outputs[0], initialized["pose"])


def test_refinement_preserves_the_composed_dense_registrar_outputs_exactly():
    torch.manual_seed(37)
    model = _tiny_model().eval()
    fixed = torch.rand(1, 2, 17, 21)
    moving = torch.rand(1, 2, 17, 21)
    pose = torch.tensor([[-1400.0, 2.0, -1.0]])
    pose_features = torch.rand(1, 8)
    expected = model.registrar.forward_with_details(fixed, moving)
    observed = model.refine_once(fixed, moving, pose, pose_features)
    for key in (
        "fixed_to_moving_map",
        "moving_to_fixed_map",
        "similarity_parameters",
        "local_velocity",
    ):
        assert torch.equal(observed[key], expected[key])
    assert len(observed["pyramid_velocities"]) == len(expected["pyramid_velocities"])
    assert all(
        torch.equal(observed_level, expected_level)
        for observed_level, expected_level in zip(
            observed["pyramid_velocities"], expected["pyramid_velocities"]
        )
    )


def test_recurrence_reuses_one_review_head_and_gradients_reach_all_three_branches():
    torch.manual_seed(41)
    model = _tiny_model()
    pose_image = torch.rand(2, 3, 24, 28)
    moving = torch.rand(2, 2, 16, 20)
    call_count = 0

    def count_calls(_module, _inputs, _outputs):
        nonlocal call_count
        call_count += 1

    handle = model.review_head.register_forward_hook(count_calls)
    def prepare_pair(pose):
        return (
            moving * 0.25 + pose[:, :1, None, None] / 5000.0,
            moving + pose[:, :1, None, None] / 5000.0,
            _alignment_receipt(pose, source_shape=moving.shape[-2:]),
        )

    rollout = model.rollout(pose_image, prepare_pair, refinement_steps=2)
    handle.remove()
    assert call_count == 3
    assert len([module for module in model.modules() if isinstance(module, PoseReviewHead)]) == 1
    assert rollout["compatibility_logits"].shape == (2, 2)
    assert rollout["final_compatibility_logit"].shape == (2,)

    loss = (
        rollout["compatibility_logits"].mean()
        + rollout["pose"].square().mean()
        + rollout["fixed_to_source_model_map"].square().mean()
    )
    loss.backward()
    assert model.pose_initializer.encoder.weight.grad is not None
    assert torch.count_nonzero(model.pose_initializer.encoder.weight.grad) > 0
    assert model.registrar.encoder.stem[0].weight.grad is not None
    assert torch.count_nonzero(model.registrar.encoder.stem[0].weight.grad) > 0
    assert model.review_head.encoder.blocks[0][0].weight.grad is not None
    assert torch.count_nonzero(model.review_head.encoder.blocks[0][0].weight.grad) > 0
    assert model.review_head.pose_delta_head.weight.grad is not None


class _MarkerRegistrar(nn.Module):
    def __init__(self):
        super().__init__()
        self.moving_markers = []

    def forward_with_details(
        self,
        fixed_atlas: torch.Tensor,
        moving_slice: torch.Tensor,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        batch, _, height, width = fixed_atlas.shape
        self.moving_markers.append(moving_slice[:, :1, :1, :1].detach().clone())
        marker = fixed_atlas[:, :1, :1, :1]
        pixel_map = marker.expand(batch, 2, height, width).clone()
        return {
            "fixed_to_moving_map": pixel_map,
            "moving_to_fixed_map": pixel_map,
            "similarity_parameters": fixed_atlas.new_zeros((batch, 4)),
            "local_velocity": fixed_atlas.new_zeros((batch, 2, height, width)),
            "pyramid_velocities": (),
        }


def test_rollout_rerenders_and_registers_again_at_the_final_updated_pose():
    model = JointPoseRegistrationModel(
        pose_initializer=_TinyPoseInitializer(),
        registrar=_MarkerRegistrar(),
        pose_feature_count=8,
        review_channels=(4,),
        review_hidden_features=8,
        maximum_pose_delta=(10.0, 1.0, 1.0),
    ).eval()
    model.review_head.pose_delta_head.bias.data.fill_(0.2)
    rendered_poses = []

    def prepare_pair(pose: torch.Tensor):
        rendered_poses.append(pose.detach().clone())
        marker = float(len(rendered_poses))
        moving_marker = pose[:, :1, None, None] / 10.0
        return (
            pose.new_full((pose.shape[0], 2, 12, 14), marker),
            moving_marker.expand(pose.shape[0], 2, 12, 14).clone(),
            _alignment_receipt(pose, call=len(rendered_poses)),
        )

    output = model.rollout(
        torch.rand(1, 3, 18, 20),
        prepare_pair,
        refinement_steps=2,
    )
    assert len(rendered_poses) == 3
    assert torch.equal(rendered_poses[-1], output["pose"])
    assert torch.equal(output["map_pose"], output["pose"])
    assert torch.all(output["fixed_atlas"] == 3.0)
    assert torch.all(output["fixed_to_source_model_map"] == 3.0)
    assert torch.all(output["fixed_to_aligned_moving_map"] == 3.0)
    torch.testing.assert_close(
        output["aligned_moving_to_fixed_map"], output["source_model_to_fixed_map"],
        rtol=0.0, atol=1e-5,
    )
    assert output["map_space"] == "source-model-canvas"
    assert output["aligned_map_space"] == "candidate-aligned-moving"
    assert output["map_domain_receipt"]["call"] == 3
    assert torch.equal(output["map_domain_receipt"]["map_pose"], output["pose"])
    assert torch.all(output["steps"][-1]["fixed_to_aligned_moving_map"] == 2.0)
    assert [step["map_domain_receipt"]["call"] for step in output["steps"]] == [1, 2]
    assert all(
        torch.equal(step["map_domain_receipt"]["map_pose"], step["map_pose"])
        for step in output["steps"]
    )
    assert not torch.equal(
        model.registrar.moving_markers[0], model.registrar.moving_markers[1]
    )
    assert not torch.equal(
        model.registrar.moving_markers[1], model.registrar.moving_markers[2]
    )
    assert torch.equal(
        output["moving_slice"][:, :1, :1, :1], model.registrar.moving_markers[-1]
    )
    assert not torch.equal(output["suggested_next_pose"], output["pose"])
    assert torch.allclose(
        output["suggested_next_pose"],
        output["pose"] + output["pose_delta"],
    )
    assert output["final_compatibility_logit"].shape == (1,)


def test_zero_step_rollout_still_scores_and_registers_the_initial_pose_once():
    model = JointPoseRegistrationModel(
        pose_initializer=_TinyPoseInitializer(),
        registrar=_MarkerRegistrar(),
        pose_feature_count=8,
        review_channels=(4,),
        review_hidden_features=8,
        maximum_pose_delta=(10.0, 1.0, 1.0),
    ).eval()
    model.review_head.pose_delta_head.bias.data.fill_(0.2)
    rendered_poses = []

    def prepare_pair(pose: torch.Tensor):
        rendered_poses.append(pose.detach().clone())
        fixed = pose.new_full((pose.shape[0], 2, 12, 14), 1.0)
        moving = pose.new_full((pose.shape[0], 2, 12, 14), 7.0)
        return fixed, moving, _alignment_receipt(pose, stage="final")

    output = model.rollout(
        torch.rand(2, 3, 18, 20),
        prepare_pair,
        refinement_steps=0,
    )
    assert len(rendered_poses) == 1
    assert output["steps"] == ()
    assert output["compatibility_logits"].shape == (2, 0)
    assert output["final_compatibility_logit"].shape == (2,)
    assert torch.equal(output["pose"], output["initial_pose"])
    assert torch.equal(output["map_pose"], output["pose"])
    assert torch.equal(rendered_poses[0], output["pose"])
    assert torch.all(output["fixed_to_source_model_map"] == 1.0)
    assert torch.all(output["moving_slice"] == 7.0)
    assert output["map_space"] == "source-model-canvas"
    assert output["map_domain_receipt"]["stage"] == "final"
    assert torch.equal(output["map_domain_receipt"]["map_pose"], output["pose"])
    assert not torch.equal(output["suggested_next_pose"], output["pose"])


def test_training_stage_boundaries_are_explicit():
    model = _tiny_model()
    model.set_training_stage("review")
    assert all(parameter.requires_grad for parameter in model.review_head.parameters())
    assert not any(parameter.requires_grad for parameter in model.pose_initializer.parameters())
    assert not any(parameter.requires_grad for parameter in model.registrar.parameters())

    model.set_training_stage("geometry")
    assert not any(parameter.requires_grad for parameter in model.pose_initializer.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.pose_initializer.pose_head.parameters())
    assert not any(parameter.requires_grad for parameter in model.registrar.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.registrar.velocity_heads.parameters())

    model.set_training_stage("joint")
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_aligned_maps_compose_into_raw_source_coordinates():
    height, width = 7, 9
    identity = identity_pixel_map(1, height, width)
    homography = torch.tensor(
        [[[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]]
    )
    fixed_to_source, source_to_fixed = compose_aligned_maps_to_source_model(
        identity, identity, homography
    )
    torch.testing.assert_close(
        fixed_to_source, identity - torch.tensor([2.0, 1.0])[None, :, None, None]
    )
    expected_source_to_fixed = identity + torch.tensor([2.0, 1.0])[None, :, None, None]
    expected_source_to_fixed[:, 0].clamp_(0, width - 1)
    expected_source_to_fixed[:, 1].clamp_(0, height - 1)
    torch.testing.assert_close(source_to_fixed, expected_source_to_fixed)


@torch.no_grad()
def test_rollout_rejects_missing_stale_or_invalid_alignment_receipts():
    model = _tiny_model().eval()
    image = torch.rand(1, 3, 18, 20)
    pair = torch.rand(1, 2, 12, 14)

    with pytest.raises(ValueError, match="must return"):
        model.rollout(image, lambda _pose: (pair, pair), refinement_steps=0)

    def stale(pose):
        return pair, pair, _alignment_receipt(
            pose + torch.tensor([[25.0, 0.0, 0.0]])
        )

    with pytest.raises(ValueError, match="stale"):
        model.rollout(image, stale, refinement_steps=0)

    def singular(pose):
        return pair, pair, _alignment_receipt(pose, torch.zeros(1, 3, 3))

    with pytest.raises(ValueError, match="invertible"):
        model.rollout(image, singular, refinement_steps=0)

    def wrong_shape(pose):
        return pair, pair, _alignment_receipt(pose, source_shape=(10, 10))

    with pytest.raises(ValueError, match="source_shape"):
        model.rollout(image, wrong_shape, refinement_steps=0)


def test_every_pose_proposal_is_projected_to_the_canonical_domain_and_export_matches():
    model = _tiny_model().eval()
    model.review_head.pose_delta_head.bias.data[:] = torch.tensor((10.0, 10.0, -10.0))
    fixed = torch.rand(1, 2, 12, 14)
    moving = torch.rand(1, 2, 12, 14)
    pose = torch.tensor([[490.0, 34.0, -34.0]])
    features = torch.rand(1, 8)
    refined = model.refine_once(fixed, moving, pose, features)
    assert torch.equal(refined["pose"], torch.tensor([[500.0, 35.0, -35.0]]))
    exported = JointRefinerExport(model)(fixed, moving, pose, features)
    assert torch.equal(exported[0], refined["pose"])
    assert torch.equal(
        project_pose_to_domain(torch.tensor([[900.0, 50.0, -50.0]])),
        torch.tensor([[500.0, 35.0, -35.0]]),
    )


def test_rollout_maps_and_receipts_are_bound_to_every_projected_pose():
    model = _tiny_model().eval()
    model.review_head.pose_delta_head.bias.data[:] = torch.tensor((10.0, 10.0, 10.0))
    image = torch.rand(1, 3, 18, 20)
    fixed = torch.rand(1, 2, 12, 14)
    moving = torch.rand(1, 2, 12, 14)
    seen = []

    def prepare(pose):
        seen.append(pose.detach().clone())
        return fixed, moving, _alignment_receipt(pose)

    result = model.rollout(image, prepare, refinement_steps=3)
    assert len(seen) == 4
    for pose in seen:
        assert bool((pose[:, 0] >= -4500.0).all() and (pose[:, 0] <= 500.0).all())
        assert bool((pose[:, 1:].abs() <= 35.0).all())
    for step in result["steps"]:
        assert step["map_space"] == "source-model-canvas"
        assert step["aligned_map_space"] == "candidate-aligned-moving"
        assert torch.equal(step["map_pose"], step["map_domain_receipt"]["map_pose"])
    assert result["map_space"] == "source-model-canvas"
    assert torch.equal(result["pose"], result["map_domain_receipt"]["map_pose"])
