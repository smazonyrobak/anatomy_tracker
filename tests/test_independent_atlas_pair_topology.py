from __future__ import annotations

import hashlib
import inspect

import pytest
import torch
import torch.nn as nn

import training.independent_atlas_pair_topology as topology_module
import training.independent_atlas_pair_energy as energy_module
from training.independent_atlas_pair_energy import AtlasPairEnergyModel, parameter_count
from training.independent_atlas_pair_topology import (
    EXPECTED_PARAMETER_COUNT,
    INVERSE_PERMUTATION8,
    INVERSE_PERMUTATION16,
    PERMUTATION8,
    PERMUTATION16,
    STATISTICS_DIMENSION,
    AtlasPairEnergyNativeTopology,
    AtlasPairEnergyScrambledTopologyControl,
    atlas_pair_loss,
    lattice_permutation_audit,
    make_atlas_pair_topology_pair,
    off_center_depthwise_coefficients,
    permutation_is_bijective,
    permute_lattice,
    restore_lattice,
)


MODEL_CLASSES = (
    AtlasPairEnergyScrambledTopologyControl,
    AtlasPairEnergyNativeTopology,
)


def _inputs(candidates=3, batch=1):
    generator = torch.Generator().manual_seed(221)
    source = torch.rand(batch, 1, 160, 232, generator=generator)
    source_mask = torch.zeros_like(source, dtype=torch.bool)
    available = torch.zeros(batch, 1, 1, 1)
    atlas = torch.rand(batch, candidates, 1, 160, 232, generator=generator)
    atlas_mask = torch.zeros_like(atlas, dtype=torch.bool)
    atlas_mask[..., 8:-8, 12:-12] = True
    return source, source_mask, available, atlas, atlas_mask


def test_pair_is_parameter_state_and_architecture_exact_without_learned_dependencies():
    torch.manual_seed(1804322)
    control, treatment = make_atlas_pair_topology_pair()

    assert isinstance(control, AtlasPairEnergyModel)
    assert isinstance(treatment, AtlasPairEnergyModel)
    assert parameter_count(control) == parameter_count(treatment) == 284_058
    assert EXPECTED_PARAMETER_COUNT == 284_058
    assert list(control.state_dict()) == list(treatment.state_dict())
    assert all(
        torch.equal(control.state_dict()[name], treatment.state_dict()[name])
        for name in control.state_dict()
    )
    control_storages = {
        value.untyped_storage().data_ptr()
        for _, value in (*control.named_parameters(), *control.named_buffers())
    }
    treatment_storages = {
        value.untyped_storage().data_ptr()
        for _, value in (*treatment.named_parameters(), *treatment.named_buffers())
    }
    assert len(control_storages) == len(
        (*tuple(control.named_parameters()), *tuple(control.named_buffers()))
    )
    assert len(treatment_storages) == len(
        (*tuple(treatment.named_parameters()), *tuple(treatment.named_buffers()))
    )
    assert control_storages.isdisjoint(treatment_storages)
    for module in (topology_module, energy_module):
        source = inspect.getsource(module)
        assert all(
            token not in source
            for token in (
                "torch.load(",
                "torch.hub.load",
                "load_state_dict_from_url",
                "safetensors",
            )
        )
    assert off_center_depthwise_coefficients(control).count_nonzero() == 0
    assert off_center_depthwise_coefficients(treatment).count_nonzero() == 0

    centers = []
    for encoder in (control.topology8, control.topology16):
        assert encoder.input_projection.bias is None
        assert encoder.input_projection.in_channels == 82
        assert encoder.input_projection.out_channels == 32
        assert encoder.output_projection.bias is None
        assert encoder.output_projection.in_channels == 32
        assert encoder.output_projection.out_channels == 16
        assert [block.depthwise.dilation for block in encoder.blocks] == [
            (1, 1),
            (2, 2),
            (4, 4),
        ]
        for block in encoder.blocks:
            assert block.depthwise.bias is None
            assert block.depthwise.groups == 32
            assert block.pointwise.bias is None
            centers.append(block.depthwise.weight[:, 0, 1, 1])
    assert all(value.count_nonzero() == 32 for value in centers)
    assert all(not torch.equal(first, second) for first, second in zip(centers, centers[1:]))
    for head in (control.energy8, control.energy16):
        assert isinstance(head[0], nn.LayerNorm)
        assert head[0].normalized_shape == (STATISTICS_DIMENSION,) == (194,)
        assert (head[1].in_features, head[1].out_features) == (194, 48)
        assert (head[3].in_features, head[3].out_features) == (48, 1)


def test_update_zero_outputs_are_bit_exact_including_both_scales_on_cpu():
    torch.manual_seed(1804322)
    control, treatment = make_atlas_pair_topology_pair()
    control.eval()
    treatment.eval()
    inputs = _inputs(candidates=3)
    with torch.no_grad():
        null_output = control(*inputs, candidate_chunk_size=2)
        native_output = treatment(*inputs, candidate_chunk_size=2)
    assert set(null_output) == {"energy", "energy8", "energy16"}
    assert all(
        torch.equal(null_output[name], native_output[name]) for name in null_output
    )


@pytest.mark.parametrize(
    (
        "height",
        "width",
        "permutation",
        "inverse",
        "expected_sha256",
        "expected_inverse_sha256",
        "expected_fixed_points",
        "expected_retained_edges",
    ),
    [
        (
            20,
            29,
            PERMUTATION8,
            INVERSE_PERMUTATION8,
            "283089a1c7ff350cb32b15218cca8c8507a7ab6bf2d5cb739b7a93f52e078a9f",
            "100761938c9b19320414075ddfe1367848ab377fba7395e2caf6f60d9e0406b9",
            2,
            7,
        ),
        (
            10,
            15,
            PERMUTATION16,
            INVERSE_PERMUTATION16,
            "4cc5dd17a8328d0b66b4fb54cea2cc4ddd52f0d5f93e26ac9179128ec78a5617",
            "121716ff0e162330646b6b6c78d99bd5e5284db971d3ca3d7f7983c7e895d5a2",
            1,
            5,
        ),
    ],
)
def test_precommitted_permutations_are_bijective_and_preserve_exact_vectors(
    height,
    width,
    permutation,
    inverse,
    expected_sha256,
    expected_inverse_sha256,
    expected_fixed_points,
    expected_retained_edges,
):
    assert hashlib.sha256(permutation.numpy().tobytes()).hexdigest() == expected_sha256
    assert hashlib.sha256(inverse.numpy().tobytes()).hexdigest() == expected_inverse_sha256
    assert permutation_is_bijective(permutation, inverse)
    assert not torch.equal(permutation, torch.arange(height * width))
    positions = torch.arange(height * width)
    assert int((permutation == positions).sum()) == expected_fixed_points
    native_edges = {
        tuple(sorted((row * width + column, row * width + column + 1)))
        for row in range(height)
        for column in range(width - 1)
    } | {
        tuple(sorted((row * width + column, (row + 1) * width + column)))
        for row in range(height - 1)
        for column in range(width)
    }
    retained_edges = sum(
        tuple(sorted((int(permutation[first]), int(permutation[second]))))
        in native_edges
        for first, second in native_edges
    )
    assert retained_edges == expected_retained_edges
    lattice = torch.arange(2 * 82 * height * width, dtype=torch.int64).reshape(
        2, 82, height, width
    )
    scrambled = permute_lattice(lattice, permutation)
    recovered = restore_lattice(scrambled, inverse)
    assert torch.equal(scrambled.flatten(2), lattice.flatten(2)[:, :, permutation])
    assert torch.equal(recovered, lattice)
    assert lattice_permutation_audit(lattice, permutation, inverse) == {
        "bijection": True,
        "vector_multiset_exact": True,
        "recovery_exact": True,
    }


def test_lattice_audit_checks_bijection_multiset_and_recovery_independently():
    lattice = torch.arange(2 * 3 * 2 * 3).reshape(2, 3, 2, 3)
    permutation = torch.tensor([2, 5, 0, 4, 1, 3])
    inverse = permutation.argsort()
    wrong_inverse = inverse.roll(1)
    duplicated = permutation.clone()
    duplicated[0] = duplicated[1]
    assert lattice_permutation_audit(lattice, permutation, wrong_inverse) == {
        "bijection": False,
        "vector_multiset_exact": True,
        "recovery_exact": False,
    }
    assert lattice_permutation_audit(lattice, duplicated, inverse) == {
        "bijection": False,
        "vector_multiset_exact": False,
        "recovery_exact": False,
    }


def test_native_and_scrambled_arms_diverge_only_after_off_center_access_is_enabled():
    torch.manual_seed(1804322)
    control, treatment = make_atlas_pair_topology_pair()
    with torch.no_grad():
        for model in (control, treatment):
            for encoder in (model.topology8, model.topology16):
                encoder.blocks[0].depthwise.weight[:, :, 0, 1].fill_(0.4)
    assert all(
        torch.equal(control.state_dict()[name], treatment.state_dict()[name])
        for name in control.state_dict()
    )
    inputs = _inputs(candidates=2)
    with torch.no_grad():
        null_output = control(*inputs, candidate_chunk_size=2)
        native_output = treatment(*inputs, candidate_chunk_size=2)
    assert all(
        not torch.equal(null_output[name], native_output[name])
        for name in ("energy", "energy8", "energy16")
    )


def test_deterministic_ranking_loss_reaches_all_off_center_coefficients_in_both_arms():
    torch.manual_seed(1804322)
    control, treatment = make_atlas_pair_topology_pair()
    inputs = _inputs(candidates=3)
    candidate_pose = torch.tensor(
        [[[-250.0, 0.0, 0.0], [0.0, 2.5, 0.0], [250.0, 0.0, -2.5]]]
    )
    truth_pose = candidate_pose[:, 1].clone()
    target_index = torch.tensor([1])

    for model in (control, treatment):
        output = model(*inputs, candidate_chunk_size=2)
        loss = atlas_pair_loss(
            output, candidate_pose, truth_pose, target_index
        )["total"]
        loss.backward()
        gradients = off_center_depthwise_coefficients(model, gradients=True)
        assert torch.isfinite(loss)
        assert gradients.shape == (2 * 3 * 32 * 8,)
        assert torch.isfinite(gradients).all()
        assert gradients.count_nonzero() == gradients.numel()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_batch_source_reuse_chunking_and_per_candidate_order_are_invariant(model_class):
    torch.manual_seed(73)
    model = model_class().eval()
    with torch.no_grad():
        for encoder in (model.topology8, model.topology16):
            for block in encoder.blocks:
                block.depthwise.weight[:, :, 0, 1].normal_(0.0, 0.1)
    inputs = _inputs(candidates=5, batch=2)
    permutations = torch.tensor([[4, 0, 3, 1, 2], [1, 3, 0, 4, 2]])
    atlas_index = permutations[:, :, None, None, None].expand_as(inputs[3])
    mask_index = permutations[:, :, None, None, None].expand_as(inputs[4])
    permuted_inputs = (
        *inputs[:3],
        inputs[3].gather(1, atlas_index),
        inputs[4].gather(1, mask_index),
    )
    with torch.no_grad():
        partial_chunks = model(*inputs, candidate_chunk_size=2)
        all_at_once = model(*inputs, candidate_chunk_size=5)
        permuted = model(*permuted_inputs, candidate_chunk_size=3)
    inverse = permutations.argsort(1)
    assert all(
        torch.allclose(partial_chunks[name], all_at_once[name], atol=1e-6, rtol=1e-6)
        and torch.allclose(
            partial_chunks[name], permuted[name].gather(1, inverse), atol=1e-6, rtol=1e-6
        )
        for name in partial_chunks
    )
