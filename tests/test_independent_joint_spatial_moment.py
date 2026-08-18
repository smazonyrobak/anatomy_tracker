from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch

import training.run_independent_pose_identifiability as diagnostic
from training.independent_joint_model import IndependentJointModel
from training.independent_joint_variants import (
    IndependentJointSpatialMomentModel,
    SpatialMomentProbabilisticPoseHead,
    VariantInitializerExport,
)


ROOT = Path(__file__).parents[1]
BASE_CONFIG = ROOT / "training/configs/independent_pose_identifiability_300_r4322.json"
SPATIAL_CONFIG = (
    ROOT / "training/configs/independent_pose_identifiability_spatial_moment_300_r4322.json"
)


def _small_model():
    torch.manual_seed(17)
    return IndependentJointSpatialMomentModel(
        pyramid_channels=(8, 8, 8, 8),
        pose_context_features=24,
        pair_features=16,
        hidden_channels=16,
        integration_steps=3,
    )


def _source(batch: int):
    generator = torch.Generator().manual_seed(51 + batch)
    image = torch.rand(batch, 1, 32, 40, generator=generator)
    outline = torch.ones_like(image)
    outline[:, :, :3] = 0.0
    available = torch.arange(batch).remainder(2).float().view(batch, 1, 1, 1)
    return image, outline, available


def test_spatial_softmax_normalization_and_first_second_moment_geometry():
    logits = torch.full((1, 4, 3, 5), -30.0)
    logits[0, 0, 0, 4] = 30.0
    logits[0, 1, 0, 0] = logits[0, 1, 2, 4] = 30.0
    logits[0, 2, 0, 4] = logits[0, 2, 2, 0] = 30.0
    logits[0, 3] = 0.0
    weights, flat = SpatialMomentProbabilisticPoseHead.moments_from_logits(logits)
    moments = flat.reshape(1, 4, 5)

    assert torch.allclose(weights.sum(dim=(-2, -1)), torch.ones(1, 4))
    assert torch.allclose(moments[0, 0], torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(moments[0, 1], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]), atol=1e-6)
    assert torch.allclose(moments[0, 2], torch.tensor([0.0, 0.0, 1.0, 1.0, -1.0]), atol=1e-6)
    assert torch.allclose(
        moments[0, 3], torch.tensor([0.0, 0.0, 0.5, 2.0 / 3.0, 0.0]), atol=1e-6
    )


def test_spatial_initializer_gradients_are_finite_and_trainable_boundary_includes_maps():
    model = _small_model().train()
    parameters = diagnostic._pose_parameter_group(model)
    image, outline, available = _source(3)
    output = model.initialize(image, outline, available)
    loss = (
        output["ap_logits"][:, 0].sum()
        + output["lr_logits"][:, 1].sum()
        + output["dv_logits"][:, 2].sum()
        + output["continuous_residual"].square().sum()
    )
    loss.backward()

    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert {
        "pose_head.spatial_attention_logits.weight",
        "pose_head.spatial_attention_logits.bias",
    }.issubset(trainable)
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in parameters)
    attention_gradient = model.pose_head.spatial_attention_logits.weight.grad
    assert torch.count_nonzero(attention_gradient) > 0
    assert torch.count_nonzero(model.pose_head.context[1].weight.grad[:, -20:]) > 0


def test_only_initializer_differs_and_parameter_delta_is_below_two_percent():
    arguments = {
        "pyramid_channels": (8, 8, 8, 8),
        "pose_context_features": 24,
        "pair_features": 16,
        "hidden_channels": 16,
        "integration_steps": 3,
    }
    torch.manual_seed(23)
    base = IndependentJointModel(**arguments).eval()
    torch.manual_seed(23)
    spatial = IndependentJointSpatialMomentModel(**arguments).eval()
    base_state = base.state_dict()
    spatial_state = spatial.state_dict()
    shared_non_pose = [name for name in base_state if not name.startswith("pose_head.")]
    assert all(torch.equal(base_state[name], spatial_state[name]) for name in shared_non_pose)

    source = _source(2)
    with torch.no_grad():
        base_features = base.encode_source(*source)
        spatial_features = spatial.encode_source(*source)
    assert all(torch.equal(left, right) for left, right in zip(base_features, spatial_features))
    atlas = torch.flip(source[0], dims=(-1,))
    atlas_mask = torch.flip(source[1], dims=(-1,))
    atlas_available = torch.ones(2, 1, 1, 1)
    pose = torch.tensor([[-2100.0, -4.0, 6.0], [-900.0, 3.0, -5.0]])
    context = torch.linspace(-0.2, 0.2, 48).reshape(2, 24)
    base_state_tensor = base.initial_hidden_state(atlas)
    spatial_state_tensor = spatial.initial_hidden_state(atlas)
    with torch.no_grad():
        base_score = base.score_candidate_from_features(
            atlas, atlas_mask, atlas_available, pose, context,
            base_state_tensor, base_features,
        )
        spatial_score = spatial.score_candidate_from_features(
            atlas, atlas_mask, atlas_available, pose, context,
            spatial_state_tensor, spatial_features,
        )
        base_refined = base.refine_from_features(
            atlas, atlas_mask, atlas_available, pose, context,
            base_state_tensor, base_features,
        )
        spatial_refined = spatial.refine_from_features(
            atlas, atlas_mask, atlas_available, pose, context,
            spatial_state_tensor, spatial_features,
        )
    assert all(torch.equal(base_score[name], spatial_score[name]) for name in base_score)
    assert all(torch.equal(base_refined[name], spatial_refined[name]) for name in base_refined)
    assert type(base.pose_head).__name__ == "ProbabilisticPoseHead"
    assert base.learned_weight_dependencies == spatial.learned_weight_dependencies == ()
    assert base.initialization == spatial.initialization == "random"

    base_parameters = sum(value.numel() for value in IndependentJointModel().parameters())
    spatial_parameters = sum(
        value.numel() for value in IndependentJointSpatialMomentModel().parameters()
    )
    assert base_parameters == 1_369_070
    assert spatial_parameters == 1_373_338
    assert spatial_parameters < base_parameters * 1.02


def test_frozen_spatial_diagnostic_changes_only_model_identity_and_lineage():
    base = diagnostic.load_pose_identifiability_config(BASE_CONFIG)
    spatial = diagnostic.load_pose_identifiability_config(SPATIAL_CONFIG)
    for name in (
        "schema_version", "frozen", "purpose", "role", "product5_access",
        "calibration_access", "final_test_access", "learned_checkpoint_dependencies",
        "seed", "device", "paths", "data", "training", "evaluation", "gates",
    ):
        assert spatial[name] == base[name]
    assert diagnostic.latent_pose_table(spatial).tobytes() == diagnostic.latent_pose_table(base).tobytes()
    assert diagnostic.nuisance_transform_tables(spatial).keys() == diagnostic.nuisance_transform_tables(base).keys()
    assert spatial["model"]["kwargs"] == base["model"]["kwargs"]
    assert spatial["model"]["class"].endswith(".IndependentJointSpatialMomentModel")
    assert spatial["model"]["expected_parameter_count"] == 1_373_338
    assert spatial["name"] != base["name"]
    assert diagnostic._model_contract(spatial)["factory"] is IndependentJointSpatialMomentModel
    assert set(spatial["lineage"]["source_sha256"]) == set(
        base["lineage"]["source_sha256"]
    ) | {"training/independent_joint_variants.py"}
    with pytest.raises(ValueError, match="not allowlisted"):
        diagnostic._model_contract({"model": {"class": "builtins.eval"}})


def test_spatial_initializer_onnx_checker_and_cpu_runtime_dynamic_batch_parity():
    onnx = __import__("onnx")
    ort = __import__("onnxruntime")
    model = _small_model().eval()
    wrapper = VariantInitializerExport(model)
    traced_inputs = _source(2)
    output_names = [
        "pose", "pose_context", "ap_logits", "lr_logits", "dv_logits",
        "pose_cholesky", "source_feature_0", "source_feature_1",
        "source_feature_2", "source_feature_3",
    ]
    buffer = io.BytesIO()
    torch.onnx.export(
        wrapper,
        traced_inputs,
        buffer,
        input_names=["source_image", "source_mask", "mask_available"],
        output_names=output_names,
        dynamic_axes={
            **{name: {0: "source_batch"} for name in (
                "source_image", "source_mask", "mask_available",
            )},
            **{name: {0: "source_batch"} for name in output_names},
        },
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load_from_string(buffer.getvalue())
    onnx.checker.check_model(graph)
    assert "Softmax" in {node.op_type for node in graph.graph.node}

    dynamic_inputs = _source(3)
    session = ort.InferenceSession(
        buffer.getvalue(), providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        None,
        {
            "source_image": dynamic_inputs[0].numpy(),
            "source_mask": dynamic_inputs[1].numpy(),
            "mask_available": dynamic_inputs[2].numpy(),
        },
    )
    with torch.no_grad():
        expected = wrapper(*dynamic_inputs)
    assert actual[0].shape == (3, 3)
    assert all(
        torch.allclose(left, torch.from_numpy(right), atol=5e-4, rtol=1e-4)
        for left, right in zip(expected, actual)
    )
