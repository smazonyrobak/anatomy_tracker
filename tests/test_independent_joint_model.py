from __future__ import annotations

import ast
import io
import math
from pathlib import Path
from unittest.mock import patch

import torch

from training.independent_joint_model import (
    AP_MAX_UM,
    AP_MIN_UM,
    TILT_MAX_DEG,
    TILT_MIN_DEG,
    IndependentCachedRefinerExport,
    IndependentCandidateScorerExport,
    IndependentInitializerExport,
    IndependentJointModel,
    IndependentRefinerExport,
    StructuralPyramid,
    affine_velocity_coefficients,
    identity_pixel_map,
    jacobian_determinant,
    project_affine_free_velocity,
    project_pose_to_domain,
    registration_maps,
    warp_tensor,
)


def _model() -> IndependentJointModel:
    torch.manual_seed(71)
    return IndependentJointModel(
        pyramid_channels=(8, 8, 8, 8),
        pose_context_features=24,
        pair_features=16,
        hidden_channels=16,
        integration_steps=3,
    )


def _inputs(batch: int = 2, height: int = 32, width: int = 40):
    generator = torch.Generator().manual_seed(19)
    image = torch.rand(batch, 1, height, width, generator=generator)
    outline = torch.ones_like(image)
    outline[:, :, :2] = 0.0
    available = torch.ones(batch, 1, 1, 1)
    atlas = torch.flip(image, dims=(-1,))
    atlas_outline = torch.flip(outline, dims=(-1,))
    atlas_available = torch.ones(batch, 1, 1, 1)
    return image, outline, available, atlas, atlas_outline, atlas_available


def test_source_has_random_self_contained_lineage_and_no_forbidden_imports():
    source_path = Path(__file__).parents[1] / "training" / "independent_joint_model.py"
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

    torch.manual_seed(5)
    first = IndependentJointModel(
        pyramid_channels=(8, 8, 8, 8), pose_context_features=24,
        pair_features=16, hidden_channels=16,
    )
    torch.manual_seed(5)
    second = IndependentJointModel(
        pyramid_channels=(8, 8, 8, 8), pose_context_features=24,
        pair_features=16, hidden_channels=16,
    )
    torch.manual_seed(6)
    third = IndependentJointModel(
        pyramid_channels=(8, 8, 8, 8), pose_context_features=24,
        pair_features=16, hidden_channels=16,
    )
    assert first.learned_weight_dependencies == ()
    assert first.initialization == "random"
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(first.parameters(), third.parameters()))


def test_explicit_mask_availability_supports_accurate_imperfect_and_absent_outlines():
    model = _model().eval()
    image, outline, available, _, _, _ = _inputs(batch=1)
    altered_background = image + (1.0 - outline) * 11.0
    absent_mask = torch.zeros_like(outline)
    absent = torch.zeros_like(available)
    imperfect = outline.clone()
    imperfect[:, :, 6:10, 9:14] = 0.0

    accurate_input = StructuralPyramid._input(image, outline, available)
    altered_accurate_input = StructuralPyramid._input(
        altered_background, outline, available
    )
    absent_input = StructuralPyramid._input(image, absent_mask, absent)
    altered_absent_input = StructuralPyramid._input(
        altered_background, absent_mask, absent
    )
    assert accurate_input.shape[1] == 3
    assert torch.equal(accurate_input, altered_accurate_input)
    assert not torch.equal(absent_input, altered_absent_input)
    assert torch.equal(absent_input[:, 0:1], image)
    assert torch.equal(absent_input[:, 1:2], absent_mask)
    assert torch.equal(absent_input[:, 2:3], absent.expand_as(image))

    with torch.no_grad():
        accurate = model.encode_source(image, outline, available)
        imperfect_features = model.encode_source(image, imperfect, available)
        absent_features = model.encode_source(image, absent_mask, absent)
        direct_pose = model(image, outline, available)
    assert direct_pose.shape == (1, 3)
    assert len(accurate) == len(imperfect_features) == len(absent_features) == 4
    assert all(
        torch.isfinite(feature).all()
        for group in (accurate, imperfect_features, absent_features)
        for feature in group
    )
    assert any(not torch.equal(a, b) for a, b in zip(accurate, imperfect_features))
    assert any(not torch.equal(a, b) for a, b in zip(accurate, absent_features))


def test_initializer_exposes_calibratable_pose_distribution_and_point_estimate():
    model = _model().eval()
    image, outline, available, _, _, _ = _inputs()
    with torch.no_grad():
        outputs = model.initialize(image, outline, available)

    assert outputs["pose"].shape == (2, 3)
    assert outputs["ap_logits"].shape == (2, 41)
    assert outputs["lr_logits"].shape == outputs["dv_logits"].shape == (2, 29)
    assert torch.allclose(outputs["ap_probability"].sum(1), torch.ones(2))
    assert torch.allclose(outputs["lr_probability"].sum(1), torch.ones(2))
    assert torch.allclose(outputs["dv_probability"].sum(1), torch.ones(2))
    assert outputs["pose_cholesky"].shape == (2, 3, 3)
    assert outputs["pose_covariance"].shape == (2, 3, 3)
    assert torch.all(torch.diagonal(outputs["pose_cholesky"], dim1=1, dim2=2) > 0)
    eigenvalues = torch.linalg.eigvalsh(outputs["pose_covariance"])
    assert torch.all(eigenvalues > 0)
    assert torch.allclose(
        outputs["pose_covariance"], outputs["pose_covariance"].transpose(1, 2)
    )
    assert torch.all(
        (AP_MIN_UM <= outputs["pose"][:, 0])
        & (outputs["pose"][:, 0] <= AP_MAX_UM)
    )
    assert torch.all(
        (TILT_MIN_DEG <= outputs["pose"][:, 1:])
        & (outputs["pose"][:, 1:] <= TILT_MAX_DEG)
    )

    exported = IndependentInitializerExport(model)(image, outline, available)
    assert len(exported) == 10
    assert torch.equal(exported[0], outputs["pose"])
    assert torch.equal(exported[5], outputs["pose_cholesky"])
    cached = model.encode_source(image, outline, available)
    assert all(torch.equal(a, b) for a, b in zip(exported[6:], cached))


def test_pose_domain_projection_is_exact_and_differentiable_inside_domain():
    pose = torch.tensor(
        [[-9000.0, -100.0, 100.0], [-1200.0, -2.0, 3.0], [4000.0, 80.0, -80.0]],
        requires_grad=True,
    )
    projected = project_pose_to_domain(pose)
    expected = torch.tensor(
        [
            [AP_MIN_UM, TILT_MIN_DEG, TILT_MAX_DEG],
            [-1200.0, -2.0, 3.0],
            [AP_MAX_UM, TILT_MAX_DEG, TILT_MIN_DEG],
        ]
    )
    assert torch.equal(projected, expected)
    projected[1].sum().backward()
    assert torch.equal(pose.grad[1], torch.ones(3))


def test_refiner_tensor_contract_and_zero_initialized_identity():
    model = _model().eval()
    image, outline, available, atlas, atlas_outline, atlas_available = _inputs()
    with torch.no_grad():
        initialization = model.initialize(image, outline, available)
        state = model.initial_hidden_state(atlas)
        outputs = model.refine_once(
            atlas, atlas_outline, atlas_available, image, outline, available,
            initialization["pose"], initialization["pose_context"], state,
        )

    assert outputs["pose"].shape == outputs["pose_delta"].shape == (2, 3)
    assert outputs["similarity_parameters"].shape == (2, 5)
    expected_similarity = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]).expand(2, -1)
    assert torch.equal(outputs["similarity_parameters"], expected_similarity)
    assert outputs["stationary_velocity"].shape == (2, 2, 32, 40)
    assert outputs["affine_velocity_coefficients"].shape == (2, 2, 3)
    assert outputs["fixed_to_moving_map"].shape == (2, 2, 32, 40)
    assert outputs["moving_to_fixed_map"].shape == (2, 2, 32, 40)
    assert outputs["compatibility_logit"].shape == (2,)
    assert outputs["validity_logits"].shape == (2, 1, 32, 40)
    assert outputs["validity_probability"].shape == (2, 1, 32, 40)
    assert outputs["hidden_state"].shape == (2, 16, 2, 3)
    assert torch.equal(outputs["pose_delta"], torch.zeros_like(outputs["pose_delta"]))
    assert torch.equal(
        outputs["stationary_velocity"], torch.zeros_like(outputs["stationary_velocity"])
    )
    assert torch.equal(
        outputs["affine_velocity_coefficients"],
        torch.zeros_like(outputs["affine_velocity_coefficients"]),
    )
    assert torch.equal(outputs["pose"], initialization["pose"])
    identity = identity_pixel_map(2, 32, 40, device=image.device, dtype=image.dtype)
    assert torch.allclose(outputs["fixed_to_moving_map"], identity, atol=5e-4, rtol=0.0)
    assert torch.allclose(outputs["moving_to_fixed_map"], identity, atol=5e-4, rtol=0.0)

    exported = IndependentRefinerExport(model)(
        atlas, atlas_outline, atlas_available, image, outline, available,
        initialization["pose"], initialization["pose_context"], state,
    )
    assert len(exported) == 10
    assert torch.equal(exported[0], outputs["pose"])
    assert torch.equal(exported[-1], outputs["hidden_state"])


def test_cached_candidate_score_matches_raw_and_skips_dense_registration():
    model = _model().eval()
    image, outline, available, atlas, atlas_outline, atlas_available = _inputs(batch=1)
    with torch.no_grad():
        initialization = model.initialize(image, outline, available)
        source_features = model.encode_source(image, outline, available)
        with patch.object(
            model.decoder, "forward", side_effect=AssertionError("dense decoder invoked")
        ), patch(
            "training.independent_joint_model.registration_maps",
            side_effect=AssertionError("map integration invoked"),
        ):
            raw = model.score_candidate(
                atlas, atlas_outline, atlas_available, image, outline, available,
                initialization["pose"], initialization["pose_context"],
            )
            cached = model.score_candidate_from_features(
                atlas, atlas_outline, atlas_available, initialization["pose"],
                initialization["pose_context"], None, source_features,
            )
    assert set(raw) == {"pose", "pose_delta", "compatibility_logit", "hidden_state"}
    assert all(torch.equal(raw[name], cached[name]) for name in raw)

    with torch.no_grad():
        batched = model.score_candidate_from_features(
            atlas.expand(3, -1, -1, -1),
            atlas_outline.expand(3, -1, -1, -1),
            atlas_available.expand(3, -1, -1, -1),
            initialization["pose"].expand(3, -1),
            initialization["pose_context"], None, source_features,
        )
    assert batched["compatibility_logit"].shape == (3,)
    assert batched["hidden_state"].shape[0] == 3


def test_cached_final_refiner_matches_raw_and_export_wrappers_share_one_checkpoint():
    model = _model().eval()
    image, outline, available, atlas, atlas_outline, atlas_available = _inputs(batch=1)
    with torch.no_grad():
        initialization = model.initialize(image, outline, available)
        source_features = model.encode_source(image, outline, available)
        state = model.initial_hidden_state(atlas)
        raw = model.refine_once(
            atlas, atlas_outline, atlas_available, image, outline, available,
            initialization["pose"], initialization["pose_context"], state,
        )
        cached = model.refine_from_features(
            atlas, atlas_outline, atlas_available, initialization["pose"],
            initialization["pose_context"], state, source_features,
        )
        scored = IndependentCandidateScorerExport(model)(
            atlas, atlas_outline, atlas_available, initialization["pose"],
            initialization["pose_context"], state, torch.zeros(1, dtype=torch.long),
            *source_features[-2:],
        )
        refined = IndependentCachedRefinerExport(model)(
            atlas, atlas_outline, atlas_available, initialization["pose"],
            initialization["pose_context"], state, torch.zeros(1, dtype=torch.long),
            *source_features,
        )
    assert all(torch.equal(raw[name], cached[name]) for name in raw)
    assert len(scored) == 4
    assert len(refined) == 10
    assert torch.equal(scored[2], raw["compatibility_logit"])
    assert torch.equal(refined[5], raw["fixed_to_moving_map"])


def test_source_index_maps_multiple_sources_to_their_candidate_lattices():
    model = _model().eval()
    image, outline, available, atlas, atlas_outline, atlas_available = _inputs(batch=2)
    candidates_per_source = 3
    source_index = torch.arange(2).repeat_interleave(candidates_per_source)
    with torch.no_grad():
        initialization = model.initialize(image, outline, available)
        source_features = model.encode_source(image, outline, available)
        batched = model.score_candidate_from_features(
            atlas.repeat_interleave(candidates_per_source, dim=0),
            atlas_outline.repeat_interleave(candidates_per_source, dim=0),
            atlas_available.repeat_interleave(candidates_per_source, dim=0),
            initialization["pose"].repeat_interleave(candidates_per_source, dim=0),
            initialization["pose_context"],
            None,
            source_features[-2:],
            source_index,
        )
        separate = []
        for source in range(2):
            separate.append(
                model.score_candidate_from_features(
                    atlas[source : source + 1].expand(candidates_per_source, -1, -1, -1),
                    atlas_outline[source : source + 1].expand(candidates_per_source, -1, -1, -1),
                    atlas_available[source : source + 1].expand(candidates_per_source, -1, -1, -1),
                    initialization["pose"][source : source + 1].expand(candidates_per_source, -1),
                    initialization["pose_context"][source : source + 1],
                    None,
                    tuple(feature[source : source + 1] for feature in source_features[-2:]),
                )
            )
    assert batched["pose"].shape == (6, 3)
    assert torch.allclose(
        batched["compatibility_logit"],
        torch.cat([output["compatibility_logit"] for output in separate]),
        atol=1e-6,
        rtol=0.0,
    )


def test_convgru_state_changes_the_next_recurrent_state():
    model = _model().eval()
    image, outline, available, atlas, atlas_outline, atlas_available = _inputs(batch=1)
    with torch.no_grad():
        initialization = model.initialize(image, outline, available)
        zero = model.initial_hidden_state(atlas)
        nonzero = torch.ones_like(zero) * 0.35
        from_zero = model.score_candidate(
            atlas, atlas_outline, atlas_available, image, outline, available,
            initialization["pose"], initialization["pose_context"], zero,
        )
        from_nonzero = model.score_candidate(
            atlas, atlas_outline, atlas_available, image, outline, available,
            initialization["pose"], initialization["pose_context"], nonzero,
        )
    assert not torch.allclose(from_zero["hidden_state"], from_nonzero["hidden_state"])
    assert not torch.allclose(from_zero["hidden_state"], zero)
    assert not torch.allclose(
        from_zero["compatibility_logit"], from_nonzero["compatibility_logit"]
    )


def test_similarity_maps_cover_rotation_translation_scale_and_are_inverse_consistent():
    height, width = 48, 64
    velocity = torch.zeros(1, 2, height, width)
    angle = 0.31
    scale = 1.18
    translation_x, translation_y = 2.2, -1.4
    similarity = torch.tensor(
        [[math.cos(angle), math.sin(angle), translation_x, translation_y, math.log(scale)]]
    )
    forward, inverse = registration_maps(similarity, velocity, integration_steps=4)
    identity = identity_pixel_map(1, height, width, device=velocity.device, dtype=velocity.dtype)
    x = identity[:, 0] - (width - 1.0) / 2.0
    y = identity[:, 1] - (height - 1.0) / 2.0
    expected = torch.stack(
        (
            scale * (math.cos(angle) * x - math.sin(angle) * y)
            + translation_x + (width - 1.0) / 2.0,
            scale * (math.sin(angle) * x + math.cos(angle) * y)
            + translation_y + (height - 1.0) / 2.0,
        ),
        dim=1,
    )
    assert torch.allclose(forward, expected, atol=2e-5, rtol=0.0)
    cycle = warp_tensor(inverse, forward)
    assert (
        cycle[:, :, 10:-10, 12:-12] - identity[:, :, 10:-10, 12:-12]
    ).abs().max() < 0.08
    assert torch.all(jacobian_determinant(forward) > 0.0)


def test_similarity_head_is_periodic_and_covers_full_training_scale_range():
    model = _model().eval()
    image, outline, available, atlas, atlas_outline, atlas_available = _inputs(batch=1)
    with torch.no_grad():
        initialization = model.initialize(image, outline, available)
        model.similarity_head.bias.copy_(torch.tensor([-1.0, 0.0, 20.0, -20.0, -20.0]))
        low = model.refine_once(
            atlas, atlas_outline, atlas_available, image, outline, available,
            initialization["pose"], initialization["pose_context"],
        )["similarity_parameters"]
        model.similarity_head.bias[4] = 20.0
        high = model.refine_once(
            atlas, atlas_outline, atlas_available, image, outline, available,
            initialization["pose"], initialization["pose_context"],
        )["similarity_parameters"]
    assert torch.allclose(low[:, :2], torch.tensor([[-1.0, 0.0]]), atol=1e-6)
    assert torch.allclose(low[:, 2:4], torch.tensor([[32.0, -32.0]]), atol=1e-5)
    assert torch.allclose(torch.exp(low[:, 4]), torch.tensor([0.4]), atol=1e-5)
    assert torch.allclose(torch.exp(high[:, 4]), torch.tensor([2.0]), atol=1e-5)


def test_nonuniform_velocity_exponentiates_to_diffeomorphic_inverse_maps():
    height, width = 40, 52
    x = torch.linspace(-1.0, 1.0, width)[None, None, None, :]
    y = torch.linspace(-1.0, 1.0, height)[None, None, :, None]
    velocity = torch.cat(
        (
            (0.35 * torch.sin(math.pi * x) * torch.cos(math.pi * y)).expand(
                1, 1, height, width
            ),
            (0.30 * torch.cos(math.pi * x) * torch.sin(math.pi * y)).expand(
                1, 1, height, width
            ),
        ),
        dim=1,
    )
    velocity, _ = project_affine_free_velocity(velocity)
    similarity = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    forward, inverse = registration_maps(similarity, velocity, integration_steps=6)
    identity = identity_pixel_map(
        1, height, width, device=velocity.device, dtype=velocity.dtype
    )
    cycle = warp_tensor(inverse, forward)
    assert (
        cycle[:, :, 4:-4, 4:-4] - identity[:, :, 4:-4, 4:-4]
    ).abs().max() < 0.003
    assert torch.all(jacobian_determinant(forward) > 0.0)


def test_affine_velocity_is_exposed_and_removed_without_erasing_local_deformation():
    height, width = 25, 31
    x = torch.linspace(-1.0, 1.0, width)[None, None, None, :]
    y = torch.linspace(-1.0, 1.0, height)[None, None, :, None]
    affine = torch.cat(
        (
            (2.0 + 3.0 * x - 0.5 * y).expand(1, 1, height, width),
            (-1.0 + 0.25 * x + 4.0 * y).expand(1, 1, height, width),
        ),
        dim=1,
    )
    residual, coefficients = project_affine_free_velocity(affine)
    expected = torch.tensor([[[2.0, 3.0, -0.5], [-1.0, 0.25, 4.0]]])
    assert torch.allclose(coefficients, expected, atol=2e-5, rtol=0.0)
    assert residual.abs().max() < 2e-5

    local = torch.cat(
        (
            (x.square() - x.square().mean()).expand(1, 1, height, width),
            (y.square() - y.square().mean()).expand(1, 1, height, width),
        ),
        dim=1,
    )
    retained, leak = project_affine_free_velocity(local)
    assert affine_velocity_coefficients(local).abs().max() < 2e-6
    assert leak.abs().max() < 2e-6
    assert torch.allclose(retained, local, atol=2e-6, rtol=0.0)


def test_losses_route_gradients_through_modalities_state_and_all_output_heads():
    model = _model().train()
    image, outline, available, atlas, atlas_outline, atlas_available = _inputs(batch=1)
    initialization = model.initialize(image, outline, available)
    outputs = model.refine_once(
        atlas, atlas_outline, atlas_available, image, outline, available,
        initialization["pose"], initialization["pose_context"],
        model.initial_hidden_state(atlas),
    )
    width = outputs["stationary_velocity"].shape[-1]
    spatial_weight = torch.linspace(-1.0, 1.0, width)[None, None, None, :]
    loss = (
        initialization["ap_logits"].mean()
        + initialization["lr_logits"].mean()
        + initialization["dv_logits"].mean()
        + initialization["pose_cholesky"].mean()
        + 1e-3 * outputs["pose_delta"].sum()
        + outputs["similarity_parameters"].sum()
        + (outputs["stationary_velocity"] * spatial_weight).sum()
        + outputs["affine_velocity_coefficients"].sum()
        + outputs["compatibility_logit"].mean()
        + outputs["validity_logits"].mean()
    )
    loss.backward()

    required = (
        "pyramid.slice_stem.0.weight",
        "pyramid.atlas_stem.0.weight",
        "pyramid.levels.0.transition.0.weight",
        "pose_head.ap_logits.weight",
        "pose_head.local_cholesky.weight",
        "pair_projection.0.weight",
        "recurrent.gates.weight",
        "pose_delta_head.weight",
        "similarity_head.weight",
        "compatibility_head.weight",
        "decoder.velocity_head.weight",
        "decoder.validity_head.weight",
    )
    parameters = dict(model.named_parameters())
    for name in required:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name


def test_onnx_cached_runtime_has_dynamic_batch_and_coarse_graph_omits_dense_path():
    onnx = __import__("onnx")
    ort = __import__("onnxruntime")
    model = _model().eval()
    image, outline, available, atlas, atlas_outline, atlas_available = _inputs(batch=1)
    with torch.no_grad():
        initialization = model.initialize(image, outline, available)
        source_features = model.encode_source(image, outline, available)
        state = model.initial_hidden_state(atlas)

    initializer = IndependentInitializerExport(model)
    initializer_inputs = (image, outline, available)
    initializer_input_names = ["source_image", "source_mask", "mask_available"]
    initializer_output_names = [
        "pose", "pose_context", "ap_logits", "lr_logits", "dv_logits",
        "pose_cholesky", "source_feature_0", "source_feature_1",
        "source_feature_2", "source_feature_3",
    ]
    initializer_buffer = io.BytesIO()
    torch.onnx.export(
        initializer,
        initializer_inputs,
        initializer_buffer,
        input_names=initializer_input_names,
        output_names=initializer_output_names,
        dynamic_axes={
            name: {0: "source_batch"}
            for name in initializer_input_names + initializer_output_names
        },
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load_from_string(initializer_buffer.getvalue()))

    cached_inputs = (
        atlas, atlas_outline, atlas_available, initialization["pose"],
        initialization["pose_context"], state, torch.zeros(1, dtype=torch.long),
        *source_features,
    )
    refiner_input_names = [
        "atlas_image", "atlas_mask", "atlas_mask_available", "current_pose",
        "pose_context", "hidden_state", "source_index", "source_feature_0",
        "source_feature_1", "source_feature_2", "source_feature_3",
    ]
    scorer_inputs = cached_inputs[:7] + cached_inputs[-2:]
    scorer_input_names = refiner_input_names[:7] + refiner_input_names[-2:]
    scorer_output_names = [
        "pose", "pose_delta", "compatibility_logit", "hidden_state_out"
    ]
    scorer_buffer = io.BytesIO()
    torch.onnx.export(
        IndependentCandidateScorerExport(model),
        scorer_inputs,
        scorer_buffer,
        input_names=scorer_input_names,
        output_names=scorer_output_names,
        dynamic_axes={
            **{name: {0: "candidate_or_source_batch"} for name in scorer_input_names},
            **{name: {0: "candidate_batch"} for name in scorer_output_names},
        },
        opset_version=17,
        dynamo=False,
    )
    scorer_graph = onnx.load_from_string(scorer_buffer.getvalue())
    onnx.checker.check_model(scorer_graph)
    assert "GridSample" not in {node.op_type for node in scorer_graph.graph.node}
    assert not any("decoder" in value.name for value in scorer_graph.graph.initializer)
    assert not any("similarity_head" in value.name for value in scorer_graph.graph.initializer)

    refiner_output_names = [
        "pose", "pose_delta", "similarity_parameters", "stationary_velocity",
        "affine_velocity_coefficients", "fixed_to_moving_map",
        "moving_to_fixed_map", "compatibility_logit", "validity_logits",
        "hidden_state_out",
    ]
    refiner_buffer = io.BytesIO()
    torch.onnx.export(
        IndependentCachedRefinerExport(model),
        cached_inputs,
        refiner_buffer,
        input_names=refiner_input_names,
        output_names=refiner_output_names,
        dynamic_axes={
            **{name: {0: "candidate_or_source_batch"} for name in refiner_input_names},
            **{name: {0: "candidate_batch"} for name in refiner_output_names},
        },
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load_from_string(refiner_buffer.getvalue()))

    initializer_session = ort.InferenceSession(
        initializer_buffer.getvalue(), providers=["CPUExecutionProvider"]
    )
    initializer_values = initializer_session.run(
        None,
        {
            "source_image": image.numpy(),
            "source_mask": outline.numpy(),
            "mask_available": available.numpy(),
        },
    )
    with torch.no_grad():
        initializer_expected = initializer(image, outline, available)
    assert all(
        torch.allclose(expected, torch.from_numpy(actual), atol=5e-4, rtol=1e-4)
        for expected, actual in zip(initializer_expected, initializer_values)
    )
    candidate_batch = 3
    scorer_session = ort.InferenceSession(
        scorer_buffer.getvalue(), providers=["CPUExecutionProvider"]
    )
    candidate_inputs = {
        "atlas_image": atlas.expand(candidate_batch, -1, -1, -1).contiguous().numpy(),
        "atlas_mask": atlas_outline.expand(candidate_batch, -1, -1, -1).contiguous().numpy(),
        "atlas_mask_available": atlas_available.expand(candidate_batch, -1, -1, -1).contiguous().numpy(),
        "current_pose": initialization["pose"].expand(candidate_batch, -1).contiguous().numpy(),
        "pose_context": initializer_values[1],
        "hidden_state": state.expand(candidate_batch, -1, -1, -1).contiguous().numpy(),
        "source_index": torch.zeros(candidate_batch, dtype=torch.long).numpy(),
        "source_feature_2": initializer_values[8],
        "source_feature_3": initializer_values[9],
    }
    scored = scorer_session.run(None, candidate_inputs)
    with torch.no_grad():
        scored_expected = IndependentCandidateScorerExport(model)(
            *(torch.from_numpy(candidate_inputs[name]) for name in scorer_input_names)
        )
    assert scored[0].shape == (candidate_batch, 3)
    assert scored[2].shape == (candidate_batch,)
    assert all(
        torch.allclose(expected, torch.from_numpy(actual), atol=5e-4, rtol=1e-4)
        for expected, actual in zip(scored_expected, scored)
    )

    refiner_session = ort.InferenceSession(
        refiner_buffer.getvalue(), providers=["CPUExecutionProvider"]
    )
    refiner_inputs = {
        **candidate_inputs,
        "source_feature_0": initializer_values[6],
        "source_feature_1": initializer_values[7],
    }
    refined = refiner_session.run(None, refiner_inputs)
    with torch.no_grad():
        refined_expected = IndependentCachedRefinerExport(model)(
            *(torch.from_numpy(refiner_inputs[name]) for name in refiner_input_names)
        )
    assert refined[0].shape == (candidate_batch, 3)
    assert refined[2].shape == (candidate_batch, 5)
    assert refined[5].shape == (candidate_batch, 2, 32, 40)
    assert all(
        torch.allclose(expected, torch.from_numpy(actual), atol=5e-4, rtol=1e-4)
        for expected, actual in zip(refined_expected, refined)
    )

    image_2, outline_2, available_2, atlas_2, atlas_outline_2, atlas_available_2 = _inputs(batch=2)
    initializer_values_2 = initializer_session.run(
        None,
        {
            "source_image": image_2.numpy(),
            "source_mask": outline_2.numpy(),
            "mask_available": available_2.numpy(),
        },
    )
    repeats = 3
    source_index_2 = torch.arange(2).repeat_interleave(repeats)
    candidate_inputs_2 = {
        "atlas_image": atlas_2.repeat_interleave(repeats, dim=0).numpy(),
        "atlas_mask": atlas_outline_2.repeat_interleave(repeats, dim=0).numpy(),
        "atlas_mask_available": atlas_available_2.repeat_interleave(repeats, dim=0).numpy(),
        "current_pose": torch.from_numpy(initializer_values_2[0]).repeat_interleave(repeats, dim=0).numpy(),
        "pose_context": initializer_values_2[1],
        "hidden_state": torch.zeros(6, 16, 2, 3).numpy(),
        "source_index": source_index_2.numpy(),
        "source_feature_2": initializer_values_2[8],
        "source_feature_3": initializer_values_2[9],
    }
    scored_2 = scorer_session.run(None, candidate_inputs_2)
    assert scored_2[0].shape == (6, 3)
    refined_2 = refiner_session.run(
        None,
        {
            **candidate_inputs_2,
            "source_feature_0": initializer_values_2[6],
            "source_feature_1": initializer_values_2[7],
        },
    )
    assert refined_2[5].shape == (6, 2, 32, 40)
