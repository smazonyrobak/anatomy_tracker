import math

import numpy as np
import pytest
import torch

from training.arbitrary_plane_catalogue_v3 import make_arbitrary_plane_catalogue_v3
from training.arbitrary_plane_coarse_proposal_v6 import AntipodalPlaneProposalV6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components,
    full_frame_state_to_components,
)


def _catalogue(batch=2):
    catalogue = make_arbitrary_plane_catalogue_v3(
        np.ones((7, 7, 7), dtype=bool),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        normal_count=4,
        offset_count=2,
        roll_count=2,
        raster_shape_h_w=(8, 8),
        raster_physical_span_y_x_um=(8.0, 8.0),
    )
    states = torch.from_numpy(catalogue["arrays"]["cell_states_float64"]).float()
    states = states[None].expand(batch, -1, -1).clone()
    cells = states.shape[1]
    return states, torch.full((batch, cells), -math.log(cells))


def _state(normal):
    normal = torch.as_tensor(normal, dtype=torch.float64)
    normal = normal / torch.linalg.vector_norm(normal)
    reference = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    u = torch.linalg.cross(reference, normal)
    u = u / torch.linalg.vector_norm(u)
    v = torch.linalg.cross(normal, u)
    frame = torch.stack((u, v, normal), dim=-1)
    center = 3.0 * normal
    return full_frame_state_from_components(center, frame, torch.eye(2, dtype=torch.float64))


def test_full_catalogue_mixture_and_component_probabilities_are_normalized():
    torch.manual_seed(3)
    states, log_mass = _catalogue()
    head = AntipodalPlaneProposalV6(4, proposal_channels=5, mixture_components=3)
    source = torch.randn(2, 4, 12, 12, requires_grad=True)
    result = head(
        source,
        states,
        log_mass,
        (3.5, 3.5, 3.5),
        expected_catalogue_cell_count=states.shape[1],
    )
    assert result["catalogue_complete"]
    assert not result["probabilities_calibrated"]
    assert torch.allclose(result["mixture_probability"].sum(1), torch.ones(2))
    assert torch.allclose(
        result["component_cell_probability"].sum(2), torch.ones(2, 3), atol=1e-6
    )
    assert torch.allclose(result["cell_probability"].sum(1), torch.ones(2), atol=1e-6)
    assert torch.allclose(
        result["cell_probability"],
        torch.einsum(
            "bl,blk->bk",
            result["mixture_probability"],
            result["component_cell_probability"],
        ),
        atol=1e-6,
    )
    assert torch.equal(
        result["raw_full_catalogue_cell_log_probability"],
        result["cell_log_probability"],
    )
    (-result["cell_log_probability"][:, 0].mean()).backward()
    assert torch.isfinite(source.grad).all() and torch.count_nonzero(source.grad) > 0
    gradient = head.normal_query.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_declared_complete_catalogue_is_required():
    states, log_mass = _catalogue(batch=1)
    head = AntipodalPlaneProposalV6(2)
    with pytest.raises(ValueError, match="complete catalogue"):
        head(
            torch.randn(1, 2, 8, 8),
            states,
            log_mass,
            (3.5, 3.5, 3.5),
            expected_catalogue_cell_count=states.shape[1] + 1,
        )


def test_catalogue_permutation_only_permutes_component_and_marginal_cells():
    torch.manual_seed(5)
    states, log_mass = _catalogue(batch=1)
    head = AntipodalPlaneProposalV6(3, proposal_channels=4, mixture_components=2)
    source = torch.randn(1, 3, 12, 12)
    first = head(
        source,
        states,
        log_mass,
        (3.5, 3.5, 3.5),
        expected_catalogue_cell_count=states.shape[1],
    )
    permutation = torch.randperm(states.shape[1])
    second = head(
        source,
        states[:, permutation],
        log_mass[:, permutation],
        (3.5, 3.5, 3.5),
        expected_catalogue_cell_count=states.shape[1],
    )
    assert torch.allclose(
        second["component_cell_log_probability"],
        first["component_cell_log_probability"][:, :, permutation],
        atol=1e-6,
    )
    assert torch.allclose(
        second["cell_log_probability"],
        first["cell_log_probability"][:, permutation],
        atol=1e-6,
    )
    assert torch.allclose(second["entropy"], first["entropy"], atol=1e-6)


def test_smooth_geometry_is_antipodal_invariant_and_continuous_across_old_seam():
    head = AntipodalPlaneProposalV6(1)
    state = _state((1.0, -1.0, 0.2))[None, None]
    center, frame, basis = full_frame_state_to_components(state)
    antipodal_frame = frame.clone()
    antipodal_frame[..., :, 0] *= -1.0
    antipodal_frame[..., :, 2] *= -1.0
    antipodal = full_frame_state_from_components(center, antipodal_frame, basis)
    original_geometry = head._geometry(state, (0.0, 0.0, 0.0))
    antipodal_geometry = head._geometry(antipodal, (0.0, 0.0, 0.0))
    for original, equivalent in zip(original_geometry, antipodal_geometry):
        assert torch.allclose(original, equivalent, atol=1e-12, rtol=0.0)

    epsilon = 1e-7
    left = _state((1.0 + epsilon, -1.0, 0.2))[None, None]
    right = _state((1.0, -1.0 - epsilon, 0.2))[None, None]
    left_geometry = head._geometry(left, (0.0, 0.0, 0.0))
    right_geometry = head._geometry(right, (0.0, 0.0, 0.0))
    for first, second in zip(left_geometry, right_geometry):
        assert torch.max(torch.abs(first - second)) < 5e-7


def test_fixed_spatial_bins_distinguish_rearrangements_with_equal_global_mean():
    states = torch.stack((_state((1.0, 0.0, 0.0)), _state((0.0, 1.0, 0.0))))[
        None
    ].float()
    log_mass = torch.full((1, 2), -math.log(2.0))
    head = AntipodalPlaneProposalV6(
        1, proposal_channels=1, mixture_components=1, spatial_bins_h_w=(2, 2)
    )
    with torch.no_grad():
        head.source_context[0].weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        head.source_context[0].bias.zero_()
        head.normal_query.weight.fill_(1.0)
        head.normal_query.bias.zero_()
        head.offset_query.weight.zero_()
        head.offset_query.bias.zero_()
        head.roll_query.weight.zero_()
        head.roll_query.bias.zero_()
        head.normal_embedding[0].weight.zero_()
        head.normal_embedding[0].weight[0, 0] = 1.0
        head.normal_embedding[0].bias.zero_()
        head.normal_embedding[2].weight.fill_(1.0)
        head.normal_embedding[2].bias.zero_()

    top_left = torch.zeros(1, 1, 4, 4)
    top_left[..., :2, :2] = 1.0
    bottom_right = torch.zeros_like(top_left)
    bottom_right[..., 2:, 2:] = 1.0
    first = head(
        top_left,
        states,
        log_mass,
        (0.0, 0.0, 0.0),
        expected_catalogue_cell_count=2,
    )
    second = head(
        bottom_right,
        states,
        log_mass,
        (0.0, 0.0, 0.0),
        expected_catalogue_cell_count=2,
    )
    assert top_left.mean() == bottom_right.mean()
    assert not torch.allclose(
        first["cell_log_probability"], second["cell_log_probability"]
    )
