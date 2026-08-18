from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest
import torch

from training.independent_joint_model import IndependentJointModel, identity_pixel_map
from training.independent_joint_variants import (
    DEFAULT_VARIANT_PARAMETERS,
    LEADER_PARAMETER_REFERENCE,
    FactorizedCNNControl,
    RecurrentAttentionVariant,
    VariantCachedRefinerExport,
    VariantCandidateScorerExport,
    VariantInitializerExport,
    install_resource_hooks,
    resource_snapshot,
)


def _factorized() -> FactorizedCNNControl:
    torch.manual_seed(101)
    return FactorizedCNNControl(
        pyramid_channels=(8, 8, 8, 8),
        pose_context_features=24,
        pair_features=16,
        hidden_channels=16,
        fusion_channels=16,
        integration_steps=3,
    )


def _attention() -> RecurrentAttentionVariant:
    torch.manual_seed(101)
    return RecurrentAttentionVariant(
        pyramid_channels=(8, 8, 8, 8),
        pose_context_features=24,
        pair_features=16,
        hidden_channels=16,
        attention_channels=4,
        integration_steps=3,
    )


def _inputs(batch: int = 2, candidates: int = 2, height: int = 32, width: int = 40):
    generator = torch.Generator().manual_seed(29)
    source = torch.rand(batch, 1, height, width, generator=generator)
    source_mask = torch.ones_like(source)
    source_mask[:, :, :2] = 0.0
    source_available = torch.tensor([1.0, 0.0])[:batch].view(batch, 1, 1, 1)
    source_index = torch.arange(batch).repeat_interleave(candidates)
    atlas = torch.flip(source, dims=(-1,)).repeat_interleave(candidates, dim=0)
    atlas_mask = torch.flip(source_mask, dims=(-1,)).repeat_interleave(
        candidates, dim=0
    )
    atlas_available = torch.ones(batch * candidates, 1, 1, 1)
    return (
        source,
        source_mask,
        source_available,
        atlas,
        atlas_mask,
        atlas_available,
        source_index,
    )


def _cached_inputs(model: IndependentJointModel):
    source, source_mask, source_available, atlas, atlas_mask, atlas_available, index = (
        _inputs()
    )
    initialization = model.initialize(source, source_mask, source_available)
    features = model.encode_source(source, source_mask, source_available)
    pose = initialization["pose"].index_select(0, index)
    state = model.initial_hidden_state(atlas)
    return (
        source,
        source_mask,
        source_available,
        atlas,
        atlas_mask,
        atlas_available,
        index,
        initialization,
        features,
        pose,
        state,
    )


@pytest.mark.parametrize("factory", [_factorized, _attention])
def test_variant_shapes_maps_and_cached_contract(factory):
    model = factory().eval()
    (
        _, _, _, atlas, atlas_mask, atlas_available, index,
        initialization, features, pose, state,
    ) = _cached_inputs(model)
    with torch.no_grad():
        scored = model.score_candidate_from_features(
            atlas, atlas_mask, atlas_available, pose,
            initialization["pose_context"], state, features[-2:], index,
        )
        refined = model.refine_from_features(
            atlas, atlas_mask, atlas_available, pose,
            initialization["pose_context"], state, features, index,
        )
    assert scored["pose"].shape == (4, 3)
    assert scored["compatibility_logit"].shape == (4,)
    assert scored["hidden_state"].shape == (4, 16, 2, 3)
    assert refined["similarity_parameters"].shape == (4, 5)
    assert refined["stationary_velocity"].shape == (4, 2, 32, 40)
    assert refined["affine_velocity_coefficients"].shape == (4, 2, 3)
    assert refined["validity_logits"].shape == (4, 1, 32, 40)
    assert refined["fixed_to_moving_map"].shape == (4, 2, 32, 40)
    identity = identity_pixel_map(
        4, 32, 40, device=atlas.device, dtype=atlas.dtype
    )
    assert torch.allclose(refined["fixed_to_moving_map"], identity, atol=5e-4)
    assert torch.allclose(refined["moving_to_fixed_map"], identity, atol=5e-4)
    assert all(torch.isfinite(value).all() for value in refined.values())


def test_factorized_control_is_stateless_while_attention_variant_is_recurrent():
    for model, recurrent in ((_factorized().eval(), False), (_attention().eval(), True)):
        assert model.uses_recurrent_state is recurrent
        assert model.comparison_refinement_steps == 3
        (
            _, _, _, atlas, atlas_mask, atlas_available, index,
            initialization, features, pose, state,
        ) = _cached_inputs(model)
        with torch.no_grad():
            zero = model.score_candidate_from_features(
                atlas, atlas_mask, atlas_available, pose,
                initialization["pose_context"], state, features[-2:], index,
            )
            nonzero = model.score_candidate_from_features(
                atlas, atlas_mask, atlas_available, pose,
                initialization["pose_context"], torch.ones_like(state) * 0.4,
                features[-2:], index,
            )
        changed = not torch.equal(zero["hidden_state"], nonzero["hidden_state"])
        assert changed is recurrent
        if not recurrent:
            assert torch.equal(
                zero["compatibility_logit"], nonzero["compatibility_logit"]
            )


@pytest.mark.parametrize(
    ("factory", "specific_parameters"),
    [
        (_factorized, ("factor_coarse.0.weight", "factor_fusion.0.weight")),
        (
            _attention,
            ("coarse_attention.query.weight", "recurrent.gates.weight"),
        ),
    ],
)
def test_gradients_reach_variant_fusion_and_shared_output_heads(
    factory, specific_parameters
):
    model = factory().train()
    (
        _, _, _, atlas, atlas_mask, atlas_available, index,
        initialization, features, pose, state,
    ) = _cached_inputs(model)
    outputs = model.refine_from_features(
        atlas, atlas_mask, atlas_available, pose,
        initialization["pose_context"], state, features, index,
    )
    spatial_weight = torch.linspace(-1.0, 1.0, 40)[None, None, None, :]
    loss = (
        initialization["ap_logits"].mean()
        + initialization["pose_cholesky"].mean()
        + outputs["compatibility_logit"].mean()
        + outputs["validity_logits"].mean()
        + outputs["pose_delta"].sum() * 1e-3
        + outputs["similarity_parameters"].sum()
        + (outputs["stationary_velocity"] * spatial_weight).sum()
        + outputs["affine_velocity_coefficients"].sum()
    )
    loss.backward()
    parameters = dict(model.named_parameters())
    required = specific_parameters + (
        "pyramid.slice_stem.0.weight",
        "pyramid.atlas_stem.0.weight",
        "pose_delta_head.weight",
        "similarity_head.weight",
        "decoder.velocity_head.weight",
        "decoder.validity_head.weight",
    )
    for name in required:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name


def test_random_lineage_parameter_matching_and_resource_hooks():
    source_path = Path(__file__).parents[1] / "training" / "independent_joint_variants.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    forbidden = {
        "training.atlas_pose_models_v7",
        "training.dense_registration_model",
        "training.joint_pose_registration_model",
    }
    assert imported.isdisjoint(forbidden)
    source = source_path.read_text(encoding="utf-8")
    assert "torch.load" not in source
    assert "load_state_dict" not in source

    defaults = (FactorizedCNNControl(), RecurrentAttentionVariant())
    for model in defaults:
        parameters = sum(parameter.numel() for parameter in model.parameters())
        assert parameters == DEFAULT_VARIANT_PARAMETERS[model.architecture_family]
        assert abs(parameters / LEADER_PARAMETER_REFERENCE - 1.0) <= 0.10
        assert model.learned_weight_dependencies == ()
        assert model.initialization == "random"

    small_arguments = {
        "pyramid_channels": (8, 8, 8, 8),
        "pose_context_features": 24,
        "pair_features": 16,
        "hidden_channels": 16,
    }
    for variant, variant_argument in (
        (FactorizedCNNControl, {"fusion_channels": 16}),
        (RecurrentAttentionVariant, {"attention_channels": 4}),
    ):
        torch.manual_seed(5)
        first = variant(**small_arguments, **variant_argument)
        torch.manual_seed(5)
        second = variant(**small_arguments, **variant_argument)
        torch.manual_seed(6)
        third = variant(**small_arguments, **variant_argument)
        assert all(
            torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())
        )
        assert any(
            not torch.equal(a, b) for a, b in zip(first.parameters(), third.parameters())
        )

    model = _attention().eval()
    statistics, handles = install_resource_hooks(model)
    with torch.no_grad():
        cached = _cached_inputs(model)
        model.score_candidate_from_features(
            cached[3], cached[4], cached[5], cached[9],
            cached[7]["pose_context"], cached[10], cached[8][-2:], cached[6],
        )
    for handle in handles:
        handle.remove()
    snapshot = resource_snapshot(model, statistics)
    assert snapshot["parameters"] == sum(p.numel() for p in model.parameters())
    assert snapshot["macs"] > 0
    assert snapshot["peak_vram_bytes"] == 0


def _export_and_run_cpu(model: IndependentJointModel):
    onnx = __import__("onnx")
    ort = __import__("onnxruntime")
    model.eval()
    cached = _cached_inputs(model)
    source, source_mask, source_available = cached[:3]
    atlas, atlas_mask, atlas_available, index = cached[3:7]
    initialization, features, pose, state = cached[7:]

    initializer_names = ["source_image", "source_mask", "mask_available"]
    initializer_outputs = [
        "pose", "pose_context", "ap_logits", "lr_logits", "dv_logits",
        "pose_cholesky", "source_feature_0", "source_feature_1",
        "source_feature_2", "source_feature_3",
    ]
    initializer_buffer = io.BytesIO()
    torch.onnx.export(
        VariantInitializerExport(model),
        (source, source_mask, source_available),
        initializer_buffer,
        input_names=initializer_names,
        output_names=initializer_outputs,
        dynamic_axes={
            name: {0: "source_batch"}
            for name in initializer_names + initializer_outputs
        },
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load_from_string(initializer_buffer.getvalue()))

    common_names = [
        "atlas_image", "atlas_mask", "atlas_mask_available", "current_pose",
        "pose_context", "hidden_state", "source_index",
    ]
    scorer_names = common_names + ["source_feature_2", "source_feature_3"]
    scorer_outputs = [
        "pose", "pose_delta", "compatibility_logit", "hidden_state_out"
    ]
    scorer_buffer = io.BytesIO()
    torch.onnx.export(
        VariantCandidateScorerExport(model),
        (
            atlas, atlas_mask, atlas_available, pose, initialization["pose_context"],
            state, index, *features[-2:],
        ),
        scorer_buffer,
        input_names=scorer_names,
        output_names=scorer_outputs,
        dynamic_axes={
            **{
                name: {0: "candidate_batch"}
                for name in common_names
                if name not in {"pose_context"}
            },
            "pose_context": {0: "source_batch"},
            "source_feature_2": {0: "source_batch"},
            "source_feature_3": {0: "source_batch"},
            **{name: {0: "candidate_batch"} for name in scorer_outputs},
        },
        opset_version=17,
        dynamo=False,
    )
    scorer_graph = onnx.load_from_string(scorer_buffer.getvalue())
    onnx.checker.check_model(scorer_graph)

    refiner_names = common_names + [
        "source_feature_0", "source_feature_1", "source_feature_2", "source_feature_3"
    ]
    refiner_outputs = [
        "pose", "pose_delta", "similarity_parameters", "stationary_velocity",
        "affine_velocity_coefficients", "fixed_to_moving_map",
        "moving_to_fixed_map", "compatibility_logit", "validity_logits",
        "hidden_state_out",
    ]
    refiner_buffer = io.BytesIO()
    torch.onnx.export(
        VariantCachedRefinerExport(model),
        (
            atlas, atlas_mask, atlas_available, pose, initialization["pose_context"],
            state, index, *features,
        ),
        refiner_buffer,
        input_names=refiner_names,
        output_names=refiner_outputs,
        dynamic_axes={
            **{
                name: {0: "candidate_batch"}
                for name in common_names
                if name not in {"pose_context"}
            },
            "pose_context": {0: "source_batch"},
            **{f"source_feature_{level}": {0: "source_batch"} for level in range(4)},
            **{name: {0: "candidate_batch"} for name in refiner_outputs},
        },
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load_from_string(refiner_buffer.getvalue()))

    initializer_session = ort.InferenceSession(
        initializer_buffer.getvalue(), providers=["CPUExecutionProvider"]
    )
    initialized = initializer_session.run(
        None,
        {
            "source_image": source.numpy(),
            "source_mask": source_mask.numpy(),
            "mask_available": source_available.numpy(),
        },
    )
    with torch.no_grad():
        initialized_expected = VariantInitializerExport(model)(
            source, source_mask, source_available
        )
    assert all(
        torch.allclose(expected, torch.from_numpy(actual), atol=5e-4, rtol=1e-4)
        for expected, actual in zip(initialized_expected, initialized)
    )
    common_inputs = {
        "atlas_image": atlas.numpy(),
        "atlas_mask": atlas_mask.numpy(),
        "atlas_mask_available": atlas_available.numpy(),
        "current_pose": torch.from_numpy(initialized[0]).index_select(0, index).numpy(),
        "pose_context": initialized[1],
        "hidden_state": state.numpy(),
        "source_index": index.numpy(),
    }
    scorer_inputs = {
        **common_inputs,
        "source_feature_2": initialized[8],
        "source_feature_3": initialized[9],
    }
    scorer_session = ort.InferenceSession(
        scorer_buffer.getvalue(), providers=["CPUExecutionProvider"]
    )
    scored = scorer_session.run(None, scorer_inputs)
    with torch.no_grad():
        scored_expected = VariantCandidateScorerExport(model)(
            *(torch.from_numpy(scorer_inputs[name]) for name in scorer_names)
        )
    assert scored[0].shape == (4, 3)
    assert scored[3].shape == (4, 16, 2, 3)
    assert all(
        torch.allclose(expected, torch.from_numpy(actual), atol=5e-4, rtol=1e-4)
        for expected, actual in zip(scored_expected, scored)
    )

    refiner_session = ort.InferenceSession(
        refiner_buffer.getvalue(), providers=["CPUExecutionProvider"]
    )
    refiner_inputs = {
        **common_inputs,
        **{
            f"source_feature_{level}": initialized[6 + level]
            for level in range(4)
        },
    }
    refined = refiner_session.run(None, refiner_inputs)
    with torch.no_grad():
        refined_expected = VariantCachedRefinerExport(model)(
            *(torch.from_numpy(refiner_inputs[name]) for name in refiner_names)
        )
    assert refined[0].shape == (4, 3)
    assert refined[5].shape == (4, 2, 32, 40)
    assert all(
        torch.allclose(expected, torch.from_numpy(actual), atol=5e-4, rtol=1e-4)
        for expected, actual in zip(refined_expected, refined)
    )

    dynamic_index = torch.arange(2).repeat_interleave(3)
    dynamic_atlas = torch.flip(source, dims=(-1,)).repeat_interleave(3, dim=0)
    dynamic_mask = torch.flip(source_mask, dims=(-1,)).repeat_interleave(3, dim=0)
    dynamic_common = {
        "atlas_image": dynamic_atlas.numpy(),
        "atlas_mask": dynamic_mask.numpy(),
        "atlas_mask_available": torch.ones(6, 1, 1, 1).numpy(),
        "current_pose": torch.from_numpy(initialized[0]).index_select(0, dynamic_index).numpy(),
        "pose_context": initialized[1],
        "hidden_state": model.initial_hidden_state(dynamic_atlas).numpy(),
        "source_index": dynamic_index.numpy(),
    }
    dynamic_scored = scorer_session.run(
        None,
        {
            **dynamic_common,
            "source_feature_2": initialized[8],
            "source_feature_3": initialized[9],
        },
    )
    assert dynamic_scored[0].shape == (6, 3)
    dynamic_refined = refiner_session.run(
        None,
        {
            **dynamic_common,
            **{
                f"source_feature_{level}": initialized[6 + level]
                for level in range(4)
            },
        },
    )
    assert dynamic_refined[5].shape == (6, 2, 32, 40)
    return scorer_buffer.getvalue(), scorer_inputs, scorer_graph


@pytest.mark.parametrize("factory", [_factorized, _attention])
def test_cached_onnx_wrappers_run_cpu_with_two_sources_and_two_candidates(factory):
    _, _, graph = _export_and_run_cpu(factory())
    operations = {node.op_type for node in graph.graph.node}
    if factory is _attention:
        assert "Softmax" in operations


def test_attention_candidate_directml_tiny_preflight():
    ort = __import__("onnxruntime")
    if "DmlExecutionProvider" not in ort.get_available_providers():
        pytest.skip("DirectML execution provider is not installed")
    model_bytes, inputs, _ = _export_and_run_cpu(_attention())
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        model_bytes, sess_options=options, providers=["DmlExecutionProvider"]
    )
    outputs = session.run(None, inputs)
    assert outputs[0].shape == (4, 3)
    assert all(torch.isfinite(torch.from_numpy(output)).all() for output in outputs)
