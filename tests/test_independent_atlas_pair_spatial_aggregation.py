from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from training.independent_atlas_pair_energy import AtlasPairEnergyModel, parameter_count
from training.independent_atlas_pair_spatial_aggregation import (
    CORRELATION_CHANNELS,
    EXPECTED_PARAMETER_COUNT,
    STATISTICS_DIMENSION,
    AtlasPairEnergyGlobalAggregationControl,
    AtlasPairEnergyHaar2x2SpatialAggregation,
    _haar_2x2_ac_weights,
    _masked_global_haar_statistics,
    _symmetric_halves,
)


ROOT = Path(__file__).parents[1]
FROZEN_SHA256 = {
    "training/independent_atlas_pair_energy.py": (
        "6187cb051d048d1e5eec3137b9edc6ac09706cecffcf989507951708681589ec"
    ),
    "training/run_independent_atlas_pair_energy.py": (
        "21c73f88a48ca87ac0a44ff022993eea5dc2cfbb1f8c72237ec4b05fa4445b19"
    ),
    "training/configs/independent_oracle_atlas_pair_energy_1500.json": (
        "8e747c82ba8f477c0a71ae803cf214e052e14c4b824c1c9859c6e1b00ad061e2"
    ),
    "publication/atlas_pair_energy_diagnostic.yaml": (
        "0fd4cad60e8d6ad43f51d5df14263d0703adba5aaba0f83c929fbbff045031a9"
    ),
}
MODEL_CLASSES = (
    AtlasPairEnergyGlobalAggregationControl,
    AtlasPairEnergyHaar2x2SpatialAggregation,
)


def _inputs(batch=1, candidates=3, height=32, width=40, device="cpu"):
    generator = torch.Generator().manual_seed(911)
    source = torch.rand(batch, 1, height, width, generator=generator).to(device)
    source_mask = torch.zeros_like(source, dtype=torch.bool)
    available = torch.zeros(batch, 1, 1, 1, device=device)
    candidate = torch.rand(
        batch, candidates, 1, height, width, generator=generator
    ).to(device)
    candidate_mask = torch.zeros_like(candidate, dtype=torch.bool)
    candidate_mask[..., 3:-3, 4:-4] = True
    return source, source_mask, available, candidate, candidate_mask


def test_consumed_atlas_pair_files_remain_byte_exact():
    for relative, expected in FROZEN_SHA256.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_control_and_treatment_are_parameter_and_initial_state_exact():
    torch.manual_seed(1404322)
    control = AtlasPairEnergyGlobalAggregationControl()
    torch.manual_seed(1404322)
    treatment = AtlasPairEnergyHaar2x2SpatialAggregation()

    assert CORRELATION_CHANNELS == 81
    assert STATISTICS_DIMENSION == 405
    assert parameter_count(AtlasPairEnergyModel()) == 271_450
    assert parameter_count(control) == parameter_count(treatment) == EXPECTED_PARAMETER_COUNT
    assert EXPECTED_PARAMETER_COUNT == 271_780
    assert list(control.state_dict()) == list(treatment.state_dict())
    assert all(
        torch.equal(control.state_dict()[name], treatment.state_dict()[name])
        for name in control.state_dict()
    )
    for head in (control.energy8, control.energy16, treatment.energy8, treatment.energy16):
        assert isinstance(head[0], nn.LayerNorm)
        assert head[0].normalized_shape == (405,)
        assert (head[1].in_features, head[1].out_features) == (405, 25)
        assert (head[3].in_features, head[3].out_features) == (25, 1)


@pytest.mark.parametrize(
    ("length", "expected_first"),
    [
        (4, [1.0, 1.0, 0.0, 0.0]),
        (5, [1.0, 1.0, 0.5, 0.0, 0.0]),
    ],
)
def test_even_and_odd_symmetric_half_partitions(length, expected_first):
    first, second = _symmetric_halves(
        length, device=torch.device("cpu"), dtype=torch.float32
    )
    assert torch.equal(first, torch.tensor(expected_first))
    assert torch.equal(first + second, torch.ones(length))
    assert torch.equal(first, second.flip(0))


@pytest.mark.parametrize(("height", "width"), [(4, 6), (5, 7)])
def test_soft_quadrants_and_haar_ac_algebra(height, width):
    top, bottom = _symmetric_halves(
        height, device=torch.device("cpu"), dtype=torch.float64
    )
    left, right = _symmetric_halves(
        width, device=torch.device("cpu"), dtype=torch.float64
    )
    quadrants = torch.stack(
        (
            top[:, None] * left[None],
            top[:, None] * right[None],
            bottom[:, None] * left[None],
            bottom[:, None] * right[None],
        )
    )
    weights = _haar_2x2_ac_weights(
        height, width, device=torch.device("cpu"), dtype=torch.float64
    )
    expected = 0.5 * torch.stack(
        (
            quadrants[0] + quadrants[1] - quadrants[2] - quadrants[3],
            quadrants[0] + quadrants[2] - quadrants[1] - quadrants[3],
            quadrants[0] + quadrants[3] - quadrants[1] - quadrants[2],
        )
    )
    assert torch.equal(quadrants.sum(0), torch.ones(height, width))
    assert torch.equal(weights, expected)
    assert torch.equal(weights[0].flip(0), -weights[0])
    assert torch.equal(weights[1].flip(1), -weights[1])
    assert torch.equal(weights[2].flip(0), -weights[2])
    assert torch.equal(weights[2].flip(1), -weights[2])
    if height % 2:
        assert torch.equal(weights[[0, 2], height // 2], torch.zeros(2, width))
    if width % 2:
        assert torch.equal(weights[[1, 2], :, width // 2], torch.zeros(2, height))


def test_haar_coefficients_match_hand_computed_quadrant_contrasts():
    correlation = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    statistics = _masked_global_haar_statistics(
        correlation, mask, use_haar_coefficients=True
    )
    assert torch.equal(
        statistics,
        torch.tensor([[2.5, 4.0, -0.5, -0.25, 0.0]]),
    )


def test_statistics_flatten_in_mean_max_tb_lr_diagonal_channel_blocks():
    correlation = torch.tensor(
        [[[[112.0, 92.0], [96.0, 100.0]], [[230.0, 186.0], [190.0, 194.0]]]]
    )
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    statistics = _masked_global_haar_statistics(
        correlation, mask, use_haar_coefficients=True
    )
    assert torch.equal(
        statistics,
        torch.tensor([[100.0, 200.0, 112.0, 230.0, 1.0, 4.0, 2.0, 5.0, 3.0, 6.0]]),
    )


@pytest.mark.parametrize(("height", "width"), [(20, 29), (10, 15)])
def test_mixed_parity_haar_matches_independent_quadrant_reference(height, width):
    row = torch.arange(height, dtype=torch.float64)[:, None]
    column = torch.arange(width, dtype=torch.float64)[None]
    correlation = torch.stack(
        ((row + 1) * (column + 2), row.square() - 0.3 * column.square())
    )[None]
    mask2d = ((7 * row.long() + 3 * column.long()) % 11 < 8) & (
        2 * column < 2 * width - row
    )
    mask = mask2d[None, None]
    assert mask.any() and not mask.all() and not torch.equal(mask, mask.flip(-1))

    values = correlation[0]
    support = mask2d.sum()
    mean = (values * mask2d).sum((-2, -1)) / support
    centered = (values - mean[:, None, None]) * mask2d
    middle = width // 2
    top_left = centered[:, : height // 2, :middle].sum((-2, -1))
    top_middle = 0.5 * centered[:, : height // 2, middle].sum(-1)
    bottom_left = centered[:, height // 2 :, :middle].sum((-2, -1))
    bottom_middle = 0.5 * centered[:, height // 2 :, middle].sum(-1)
    quadrants = torch.stack(
        (
            top_left + top_middle,
            top_middle + centered[:, : height // 2, middle + 1 :].sum((-2, -1)),
            bottom_left + bottom_middle,
            bottom_middle
            + centered[:, height // 2 :, middle + 1 :].sum((-2, -1)),
        )
    )
    coefficients = 0.5 * torch.stack(
        (
            quadrants[0] + quadrants[1] - quadrants[2] - quadrants[3],
            quadrants[0] + quadrants[2] - quadrants[1] - quadrants[3],
            quadrants[0] + quadrants[3] - quadrants[1] - quadrants[2],
        )
    ) / support
    maximum = values.masked_fill(~mask2d, -torch.inf).amax((-2, -1))
    expected = torch.cat((mean, maximum, coefficients.flatten()))
    observed = _masked_global_haar_statistics(
        correlation, mask, use_haar_coefficients=True
    )[0]
    assert torch.allclose(observed, expected, atol=1e-10, rtol=1e-10)


def test_constant_correlations_have_zero_haar_ac_under_asymmetric_masks():
    constants = torch.tensor([0.25, -1.5, 3.0]).reshape(1, 3, 1, 1)
    correlation = constants.expand(1, 3, 5, 7).clone()
    mask = torch.zeros(1, 1, 5, 7, dtype=torch.bool)
    mask[..., :4, :3] = True
    mask[..., 1, 5:] = True
    mask[..., 4, 6] = True
    assert not torch.equal(mask, mask.flip(-1))
    control = _masked_global_haar_statistics(
        correlation, mask, use_haar_coefficients=False
    )
    treatment = _masked_global_haar_statistics(
        correlation, mask, use_haar_coefficients=True
    )
    assert torch.equal(control[:, :6], treatment[:, :6])
    assert torch.allclose(treatment[:, 6:], torch.zeros_like(treatment[:, 6:]), atol=1e-7)
    assert torch.allclose(control, treatment, atol=1e-7)


def test_moved_localized_signal_is_control_indistinguishable_and_haar_distinguishable():
    first = torch.zeros(1, 1, 5, 7)
    second = torch.zeros_like(first)
    first[..., 0, 0] = 2.0
    second[..., -1, -1] = 2.0
    mask = torch.ones(1, 1, 5, 7, dtype=torch.bool)
    control_first = _masked_global_haar_statistics(
        first, mask, use_haar_coefficients=False
    )
    control_second = _masked_global_haar_statistics(
        second, mask, use_haar_coefficients=False
    )
    treatment_first = _masked_global_haar_statistics(
        first, mask, use_haar_coefficients=True
    )
    treatment_second = _masked_global_haar_statistics(
        second, mask, use_haar_coefficients=True
    )
    assert torch.equal(control_first, control_second)
    assert torch.equal(treatment_first[:, :2], treatment_second[:, :2])
    assert not torch.allclose(treatment_first[:, 2:], treatment_second[:, 2:])


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_forward_chunking_and_candidate_order_preserve_standard_energy_abi(model_class):
    torch.manual_seed(47)
    model = model_class().eval()
    inputs = _inputs(batch=2, candidates=5, height=48, width=64)
    with torch.no_grad():
        first = model(*inputs, candidate_chunk_size=2)
        second = model(*inputs, candidate_chunk_size=5)
    assert set(first) == {"energy", "energy8", "energy16"}
    assert all(value.shape == (2, 5) for value in first.values())
    assert all(
        torch.allclose(first[name], second[name], atol=1e-6, rtol=1e-6)
        for name in first
    )

    order = torch.tensor([4, 1, 3, 0, 2])
    permuted_inputs = (*inputs[:3], inputs[3][:, order], inputs[4][:, order])
    with torch.no_grad():
        permuted = model(*permuted_inputs, candidate_chunk_size=3)
    inverse = order.argsort()
    assert all(
        torch.allclose(first[name], permuted[name][:, inverse], atol=1e-6, rtol=1e-6)
        for name in first
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_all_cpu_gradients_are_finite(model_class):
    torch.manual_seed(83)
    model = model_class().train()
    output = model(*_inputs(), candidate_chunk_size=2)
    loss = sum(value.square().mean() for value in output.values())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_cuda_amp_unscale_clip_and_step_gradients_are_finite(model_class):
    torch.manual_seed(101)
    torch.cuda.manual_seed_all(101)
    model = model_class().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=512.0)
    inputs = _inputs(device="cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        output = model(*inputs, candidate_chunk_size=2)
        loss = sum(value.square().mean() for value in output.values())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters()]
    assert torch.isfinite(loss)
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    assert torch.isfinite(norm)
    tracked = next(model.parameters())
    step_before = int(optimizer.state.get(tracked, {}).get("step", 0))
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    step_after = int(optimizer.state[tracked]["step"])
    assert step_after == step_before + 1
    assert scaler.get_scale() >= scale_before
