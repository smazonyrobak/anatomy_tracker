import math

import numpy as np
import pytest
import torch

from training.arbitrary_plane_catalogue_v3 import make_arbitrary_plane_catalogue_v3
from training.arbitrary_plane_coarse_proposal_v5 import (
    AntipodalPlaneProposalV5,
    COARSE_PROPOSAL_GEOMETRY,
)
from training.arbitrary_plane_deformation_primitives import identity_pixel_map_yx
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components,
    full_frame_state_to_components,
)
from training.arbitrary_plane_inference_v3 import _model_executable_contract
from training.arbitrary_plane_joint_loss import arbitrary_plane_joint_loss
from training.arbitrary_plane_joint_model import ArbitraryPlaneJointModel


def _catalogue():
    return make_arbitrary_plane_catalogue_v3(
        np.ones((7, 7, 7), dtype=bool),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        normal_count=4,
        offset_count=2,
        roll_count=2,
        raster_shape_h_w=(8, 8),
        raster_physical_span_y_x_um=(8.0, 8.0),
    )


def _inputs(batch=1):
    catalogue = _catalogue()
    states = torch.from_numpy(
        catalogue["arrays"]["cell_states_float64"]
    ).float()[None].expand(batch, -1, -1).clone()
    cells = states.shape[1]
    affine = torch.from_numpy(
        catalogue["arrays"][
            "representation_to_canonical_raster_affine_float64"
        ]
    ).float()[None].expand(batch, -1, -1, -1, -1).clone()
    return catalogue, {
        "volume": torch.rand(2, 9, 9, 9),
        "image": torch.rand(batch, 1, 8, 8),
        "outline": torch.ones(batch, 1, 8, 8),
        "available": torch.tensor([row % 2 for row in range(batch)]).float(),
        "cell_id": torch.arange(cells),
        "states": states,
        "log_mass": torch.full((batch, cells), -math.log(cells)),
        "log_weight": torch.full((batch, cells, 2), -math.log(2.0)),
        "affine": affine,
    }


def _model(proposal_count=4):
    torch.manual_seed(101)
    return ArbitraryPlaneJointModel(
        atlas_channels=2,
        feature_channels=4,
        hidden_channels=6,
        correlation_radius=1,
        update_limits=(0.08, 0.08, 0.5, 0.08, 0.5, 0.5, 0.04, 0.04, 0.04),
        plane_tangent_scales=(0.08, 0.08, 0.5),
        max_velocity_fraction_yx=(0.05, 0.04),
        deformation_integration_steps=3,
        proposal_count=proposal_count,
        proposal_channels=5,
        proposal_mixture_components=3,
        proposal_offset_scale_um=10.0,
    )


def _forward(model, data, **kwargs):
    return model(
        data["image"],
        data["outline"],
        data["available"],
        data["volume"],
        data["cell_id"],
        data["states"],
        data["log_mass"],
        data["log_weight"],
        data["affine"],
        (8, 8),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (3.5, 3.5, 3.5),
        torch.tensor([-0.5, 0.0, 0.5]),
        torch.tensor([0.25, 0.5, 0.25]),
        expected_catalogue_cell_count=data["states"].shape[1],
        top_k=2,
        refinement_steps=1,
        pose_only_steps=2,
        retrieval_shape_h_w=(4, 4),
        catalogue_chunk_size=2,
        **kwargs,
    )


def test_proposal_is_normalized_multimodal_and_geometry_factorized():
    catalogue, data = _inputs(batch=2)
    head = AntipodalPlaneProposalV5(
        4, proposal_channels=5, mixture_components=3, offset_scale_um=10.0
    )
    source = torch.randn(2, 4, 2, 2, requires_grad=True)
    result = head(
        source,
        data["states"],
        data["log_mass"],
        catalogue["support_geometry"]["support_origin_ap_dv_ml_um"],
    )
    assert result["geometry_contract"] == COARSE_PROPOSAL_GEOMETRY
    assert result["mixture_log_probability"].shape == (2, 3)
    assert result["cell_log_probability"].shape == (2, 16)
    assert torch.allclose(
        result["cell_probability"].sum(dim=1), torch.ones(2), atol=1e-6
    )
    assert not result["probabilities_calibrated"]
    (-result["cell_log_probability"][:, 0].mean()).backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert torch.count_nonzero(source.grad) > 0


def test_proposal_geometry_is_invariant_to_antipodal_frame_reencoding():
    _, data = _inputs()
    head = AntipodalPlaneProposalV5(4, proposal_channels=5, mixture_components=3)
    state = data["states"][:, :1]
    center, frame, basis = full_frame_state_to_components(state)
    antipodal_frame = frame.clone()
    antipodal_frame[..., :, 0] *= -1.0
    antipodal_frame[..., :, 2] *= -1.0
    antipodal = full_frame_state_from_components(center, antipodal_frame, basis)
    first = head._geometry(state, (3.5, 3.5, 3.5))
    second = head._geometry(antipodal, (3.5, 3.5, 3.5))
    for observed, expected in zip(first, second):
        assert torch.allclose(observed, expected, atol=1e-6, rtol=0.0)


def test_joint_path_renders_only_top_m_then_top_k(monkeypatch):
    _, data = _inputs()
    model = _model(proposal_count=4).eval()
    calls = []
    original = model.pose_model.score_catalogue_chunk

    def record(*args, **kwargs):
        calls.append(args[3].shape[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(model.pose_model, "score_catalogue_chunk", record)
    output = _forward(model, data)
    pose = output["pose"]
    assert calls == [2, 2, 2]
    assert pose["proposal_topm_catalogue_index"].shape == (1, 4)
    assert pose["retrieval_topk_catalogue_index"].shape == (1, 2)
    assert torch.allclose(pose["retrieval_cell_probability"].sum(), torch.tensor(1.0))
    assert pose["retrieval_execution"] == (
        "amortized_antipodal_proposal_plus_exact_finite_top_m_rerank"
    )
    assert pose["retrieval_tail_scope"] == (
        "complete_amortized_proposal; exact_finite_likelihood_top_m_only"
    )
    assert not pose["retrieval_teacher_forced_mask"].any()


def test_truth_can_only_be_injected_during_training_and_proposal_learns():
    _, data = _inputs()
    model = _model(proposal_count=4).eval()
    with pytest.raises(ValueError, match="training-only"):
        _forward(model, data, training_truth_catalogue_index=torch.tensor([0]))

    model.train()
    output = _forward(
        model, data, training_truth_catalogue_index=torch.tensor([0])
    )
    pose = output["pose"]
    loss = -pose["retrieval_cell_log_probability"][:, 0].mean()
    loss.backward()
    gradient = model.pose_model.coarse_proposal.normal_query.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_joint_loss_directly_trains_exact_top_m_reranker():
    _, data = _inputs()
    model = _model(proposal_count=4).train()
    output = _forward(model, data)
    pose = output["pose"]
    truth_cell = pose["retrieval_topk_cell_id"][:, 0].detach()
    losses = arbitrary_plane_joint_loss(
        output,
        pose["final_cell_state"][:, 0].detach(),
        truth_cell,
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, 2, 8, 8),
        identity_pixel_map_yx(1, (8, 8)),
        torch.ones(1, 1, 8, 8),
        (3.5, 3.5, 3.5),
    )
    assert torch.isfinite(losses["exact_rerank_nll"])
    assert losses["exact_rerank_nll"] > 0.0
    losses["exact_rerank_nll"].backward()
    gradient = model.pose_model.candidate_log_likelihood.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_production_proposal_reduces_exact_render_scope_by_1536_fold():
    production_cells = 384 * 16 * 16
    proposal_count = 64
    representation_count = 2
    axial_sample_count = 9
    retrieval_pixels = 48 * 48
    exhaustive_atlas_sample_points = (
        production_cells
        * axial_sample_count
        * retrieval_pixels
    )
    proposed_atlas_sample_points = (
        proposal_count
        * axial_sample_count
        * retrieval_pixels
    )
    exhaustive_representation_sample_points = (
        production_cells * representation_count * retrieval_pixels
    )
    proposed_representation_sample_points = (
        proposal_count * representation_count * retrieval_pixels
    )
    exhaustive_represented_images = production_cells * representation_count
    proposed_represented_images = proposal_count * representation_count
    assert production_cells == 98304
    assert exhaustive_atlas_sample_points == 2_038_431_744
    assert proposed_atlas_sample_points == 1_327_104
    assert exhaustive_representation_sample_points == 452_984_832
    assert proposed_representation_sample_points == 294_912
    assert exhaustive_represented_images == 196_608
    assert proposed_represented_images == 128
    assert exhaustive_atlas_sample_points // proposed_atlas_sample_points == 1536
    assert (
        exhaustive_representation_sample_points
        // proposed_representation_sample_points
        == 1536
    )


def test_inference_contract_binds_proposal_geometry_and_uncalibrated_scope():
    contract = _model_executable_contract(_model(proposal_count=4))[
        "coarse_proposal"
    ]
    assert contract == {
        "proposal_count": 4,
        "proposal_channels": 5,
        "mixture_components": 3,
        "offset_scale_um": 10.0,
        "geometry_contract": list(COARSE_PROPOSAL_GEOMETRY),
        "probabilities_calibrated": False,
        "exact_render_scope": "top-M only",
    }
