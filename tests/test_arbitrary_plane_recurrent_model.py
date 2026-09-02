import ast
import inspect
import math

import pytest
import torch

import training.arbitrary_plane_joint_loss as joint_loss
import training.arbitrary_plane_recurrent_model as recurrent
from training.arbitrary_plane_full_frame_primitives import (
    FULL_FRAME_STATE_SIZE,
    FULL_FRAME_UPDATE_SIZE,
    compose_full_frame_state,
    full_frame_state_from_components,
    full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import positive_inplane_basis


def _state(center, frame=None, spans=(4.0, 4.0), shear=0.0, dtype=torch.float32):
    center = torch.tensor(center, dtype=dtype)
    frame = torch.eye(3, dtype=dtype) if frame is None else torch.as_tensor(frame, dtype=dtype)
    basis = positive_inplane_basis(
        torch.log(torch.tensor(spans, dtype=dtype)), torch.tensor(shear, dtype=dtype)
    )
    return full_frame_state_from_components(center, frame, basis)


def _fixture(batch=2, cells=3, representations=2, dtype=torch.float32, state_dtype=None):
    torch.manual_seed(43)
    state_dtype = dtype if state_dtype is None else state_dtype
    volume = torch.rand(2, 10, 10, 10, dtype=dtype)
    image = torch.rand(batch, 1, 8, 8, dtype=dtype)
    outline = torch.ones_like(image)
    available = torch.tensor([index % 2 for index in range(batch)], dtype=dtype)
    base_states = torch.stack(
        [
            torch.stack(
                [
                    _state(
                        (4.7 + 0.2 * cell, 5.0 + 0.1 * row, 5.1),
                        spans=(4.0, 3.5),
                        shear=0.03,
                        dtype=state_dtype,
                    )
                    for cell in range(cells)
                ]
            )
            for row in range(batch)
        ]
    )
    states = base_states
    log_mass = torch.log(
        torch.tensor([1.0 + cell for cell in range(cells)], dtype=dtype)
    )[None].expand(batch, -1)
    log_weight = torch.full(
        (batch, cells, representations), -math.log(representations), dtype=dtype
    )
    raster = torch.eye(2, 3, dtype=dtype).expand(batch, cells, representations, 2, 3).clone()
    return {
        "volume": volume,
        "image": image,
        "outline": outline,
        "available": available,
        "cell_id": torch.arange(cells),
        "states": states,
        "log_mass": log_mass,
        "log_weight": log_weight,
        "raster": raster,
    }


def _model(dtype=torch.float32, device=None):
    torch.manual_seed(47)
    return recurrent.ArbitraryPlaneRetrievalRefinementModel(
        atlas_channels=2,
        feature_channels=4,
        hidden_channels=6,
        correlation_radius=1,
        update_limits=(0.08, 0.08, 0.5, 0.08, 0.5, 0.5, 0.04, 0.04, 0.04),
        plane_tangent_scales=(0.08, 0.08, 0.5),
    ).to(device=device, dtype=dtype)


def _score(model, fixture, cell_slice=slice(None), axial_offsets=None, axial_weights=None):
    source = model.encode_histology(
        fixture["image"], fixture["outline"], fixture["available"]
    )
    return model.score_catalogue_chunk(
        source,
        fixture["volume"],
        fixture["cell_id"][cell_slice],
        fixture["states"][:, cell_slice],
        fixture["log_mass"][:, cell_slice],
        fixture["log_weight"][:, cell_slice],
        fixture["raster"][:, cell_slice],
        (8, 8),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (5.0, 5.0, 5.0),
        torch.tensor([-0.5, 0.0, 0.5], dtype=fixture["volume"].dtype, device=fixture["volume"].device)
        if axial_offsets is None
        else axial_offsets,
        torch.tensor([0.25, 0.5, 0.25], dtype=fixture["volume"].dtype, device=fixture["volume"].device)
        if axial_weights is None
        else axial_weights,
    )


def _forward(model, fixture, **kwargs):
    return model(
        fixture["image"],
        fixture["outline"],
        fixture["available"],
        fixture["volume"],
        fixture["cell_id"],
        fixture["states"],
        fixture["log_mass"],
        fixture["log_weight"],
        fixture["raster"],
        (8, 8),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (5.0, 5.0, 5.0),
        torch.tensor([-0.5, 0.0, 0.5], dtype=fixture["volume"].dtype, device=fixture["volume"].device),
        torch.tensor([0.25, 0.5, 0.25], dtype=fixture["volume"].dtype, device=fixture["volume"].device),
        expected_catalogue_cell_count=fixture["states"].shape[1],
        **kwargs,
    )


def _forward_streamed(model, fixture, **kwargs):
    return model.forward_streamed(
        fixture["image"],
        fixture["outline"],
        fixture["available"],
        fixture["volume"],
        fixture["cell_id"],
        fixture["states"],
        fixture["log_mass"],
        fixture["log_weight"],
        fixture["raster"],
        (8, 8),
        (8, 8),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (5.0, 5.0, 5.0),
        torch.tensor([-0.5, 0.0, 0.5], dtype=fixture["volume"].dtype),
        torch.tensor([0.25, 0.5, 0.25], dtype=fixture["volume"].dtype),
        expected_catalogue_cell_count=fixture["states"].shape[1],
        catalogue_chunk_size=2,
        **kwargs,
    )


def _to(fixture, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in fixture.items()}


def test_local_correlation_places_identical_features_at_center_displacement():
    torch.manual_seed(3)
    source = torch.randn(2, 5, 6, 7)
    correlation = recurrent.local_correlation(source, source, radius=1)
    assert correlation.shape == (2, 9, 6, 7)
    assert torch.allclose(correlation[:, 4], torch.ones(2, 6, 7), atol=1e-6)
    assert correlation[:, 4].mean() > correlation[:, [0, 1, 2, 3, 5, 6, 7, 8]].mean()


def test_local_correlation_zero_features_remain_finite_in_half_precision():
    zero = torch.zeros(2, 4, 5, 6, dtype=torch.float16)
    correlation = recurrent.local_correlation(zero, zero, radius=1)
    assert correlation.dtype == torch.float16
    assert torch.isfinite(correlation).all()
    assert torch.count_nonzero(correlation) == 0


def test_antipodal_plane_residual_has_exact_zero_offset_frame_and_tangent_semantics():
    state = _state((6.0, 4.0, 8.0), spans=(3.0, 2.0), dtype=torch.float64)
    zero = torch.zeros(FULL_FRAME_UPDATE_SIZE, dtype=torch.float64)
    assert torch.allclose(
        recurrent.compose_antipodal_plane_frame_residual(state, zero, (5.0, 5.0, 5.0)),
        state,
        atol=1e-12,
        rtol=0.0,
    )

    offset = zero.clone()
    offset[2] = 1.25
    moved = recurrent.compose_antipodal_plane_frame_residual(state, offset, (5.0, 5.0, 5.0))
    center, observed_frame, observed_basis = full_frame_state_to_components(moved)
    original_center, original_frame, original_basis = full_frame_state_to_components(state)
    assert torch.allclose(center, original_center + 1.25 * original_frame[:, 2], atol=1e-12)
    assert torch.allclose(observed_frame, original_frame, atol=1e-12)
    assert torch.allclose(observed_basis, original_basis, atol=1e-12)

    angle = 0.13
    for index, expected_normal in (
        (0, torch.tensor([math.sin(angle), 0.0, math.cos(angle)], dtype=torch.float64)),
        (1, torch.tensor([0.0, math.sin(angle), math.cos(angle)], dtype=torch.float64)),
    ):
        tangent = zero.clone()
        tangent[index] = angle
        _, tangent_frame, _ = full_frame_state_to_components(
            recurrent.compose_antipodal_plane_frame_residual(state, tangent, (5.0, 5.0, 5.0))
        )
        assert torch.allclose(tangent_frame[:, 2], expected_normal, atol=1e-12)

    finite_frame = zero.clone()
    finite_frame[3:] = torch.tensor([0.07, 0.4, -0.2, 0.03, -0.02, 0.01], dtype=torch.float64)
    local_update = torch.zeros_like(finite_frame)
    local_update[2] = finite_frame[3]
    local_update[3:5] = finite_frame[4:6]
    local_update[6:] = finite_frame[6:]
    assert torch.allclose(
        recurrent.compose_antipodal_plane_frame_residual(state, finite_frame, (5.0, 5.0, 5.0)),
        compose_full_frame_state(state, local_update),
        atol=1e-11,
        rtol=0.0,
    )


def test_optional_outline_channels_never_replace_or_mask_the_image():
    model = _model()
    fixture = _fixture(batch=1)
    captured = []
    hook = model.histology_stem[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    model.encode_histology(fixture["image"], torch.zeros_like(fixture["outline"]), torch.zeros(1))
    hook.remove()
    observed = captured[0]
    assert torch.equal(observed[:, :1], fixture["image"])
    assert torch.count_nonzero(observed[:, 1:]) == 0
    first = model.encode_histology(
        fixture["image"], torch.zeros_like(fixture["outline"]), torch.zeros(1)
    )
    second = model.encode_histology(
        fixture["image"], torch.rand_like(fixture["outline"]), torch.zeros(1)
    )
    assert torch.equal(first, second)


def test_chunk_scores_are_unnormalized_and_cell_mass_is_added_once_after_representation_marginalization():
    model = _model()
    torch.nn.init.zeros_(model.candidate_log_likelihood.weight)
    assert model.candidate_log_likelihood.bias is None
    assert model.recurrent_log_likelihood.bias is None
    fixture = _fixture(batch=1, representations=3)
    output = _score(model, fixture)
    assert torch.allclose(output["cell_log_unnormalized_mass"], fixture["log_mass"], atol=1e-7)
    assert not any("probability" in key or "posterior" in key or "tail" in key for key in output)


def test_representation_permutation_and_duplicate_split_weights_do_not_change_cell_mass_or_topk():
    model = _model().eval()
    one = _score(model, _fixture(batch=1, representations=1))
    duplicate = _fixture(batch=1, representations=2)
    two = _score(model, duplicate)
    assert torch.allclose(one["cell_log_unnormalized_mass"], two["cell_log_unnormalized_mass"], atol=2e-6)

    permutation = torch.tensor([1, 0])
    for key in ("log_weight", "raster"):
        duplicate[key] = duplicate[key][:, :, permutation]
    permuted = _score(model, duplicate)
    assert torch.allclose(two["cell_log_unnormalized_mass"], permuted["cell_log_unnormalized_mass"], atol=2e-6)
    first_top = model.normalize_complete_catalogue(two, 3, 2)["retrieval_topk_cell_id"]
    second_top = model.normalize_complete_catalogue(permuted, 3, 2)["retrieval_topk_cell_id"]
    assert torch.equal(first_top, second_top)


def test_explicit_raster_reparameterization_is_applied_before_scoring():
    tensor = torch.arange(48.0).reshape(1, 1, 2, 3, 2, 4)
    tensor[:, :, 1] = tensor[:, :, 0].flip(-2)
    affine = torch.eye(2, 3).reshape(1, 1, 1, 2, 3).expand(1, 1, 2, 2, 3).clone()
    affine[:, :, 1, 1, 1] = -1.0
    canonical = recurrent.canonicalize_representation_raster(tensor, affine)
    assert torch.equal(canonical[:, :, 0], canonical[:, :, 1])


def test_raster_nuisance_contract_rejects_nonflip_affines():
    fixture = _fixture(batch=1)
    fixture["raster"][0, 0, 0, 0, 2] = 0.1
    with pytest.raises(ValueError, match="exact identity/horizontal/vertical/both flips"):
        _score(_model(), fixture)


def test_antipodal_renderer_rejects_asymmetric_through_plane_psf():
    fixture = _fixture(batch=1)
    with pytest.raises(ValueError, match="requires symmetric offsets and weights"):
        _score(
            _model(),
            fixture,
            axial_offsets=torch.tensor([-0.5, 0.0, 0.6]),
            axial_weights=torch.tensor([0.25, 0.5, 0.25]),
        )
    with pytest.raises(ValueError, match="requires symmetric offsets and weights"):
        _score(
            _model(),
            fixture,
            axial_offsets=torch.tensor([-0.5, 0.0, 0.5]),
            axial_weights=torch.tensor([0.2, 0.5, 0.3]),
        )


def test_complete_catalogue_ties_use_cell_id_order_and_all_selected_tail_is_exact_zero():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        chunk = {
            "cell_id": torch.tensor([2, 0, 3, 1], device=device),
            "cell_log_unnormalized_mass": torch.zeros(2, 4, device=device),
        }
        tied = recurrent.ArbitraryPlaneRetrievalRefinementModel.normalize_complete_catalogue(
            chunk, 4, 3
        )
        assert torch.equal(
            tied["retrieval_topk_cell_id"],
            torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
        )
        complete = recurrent.ArbitraryPlaneRetrievalRefinementModel.normalize_complete_catalogue(
            chunk, 4, 4
        )
        assert torch.equal(
            complete["retrieval_omitted_probability"], torch.zeros(2, device=device)
        )
        assert torch.all(complete["retrieval_topk_retained_probability"] >= 0.0)
        assert torch.all(complete["retrieval_topk_retained_probability"] <= 1.0)


def test_complete_catalogue_verifier_is_chunk_invariant_and_rejects_missing_or_duplicate_cells():
    fixture = _fixture(batch=1, cells=4)
    model = _model().eval()
    whole = _score(model, fixture)
    chunks = (_score(model, fixture, slice(0, 2)), _score(model, fixture, slice(2, 4)))
    normalized_whole = model.normalize_complete_catalogue(whole, 4, 2)
    normalized_chunks = model.normalize_complete_catalogue(chunks, 4, 2)
    assert torch.allclose(
        normalized_whole["retrieval_cell_log_probability"],
        normalized_chunks["retrieval_cell_log_probability"],
        atol=2e-6,
    )
    with pytest.raises(ValueError, match="every expected unique cell ID"):
        model.normalize_complete_catalogue(chunks[:1], 4, 2)
    duplicate = dict(chunks[1])
    duplicate["cell_id"] = chunks[0]["cell_id"]
    with pytest.raises(ValueError, match="every expected unique cell ID"):
        model.normalize_complete_catalogue((chunks[0], duplicate), 4, 2)


def test_forward_preserves_cell_mass_tail_representation_axis_and_uncalibrated_scope():
    output = _forward(_model(), _fixture(), top_k=2, refinement_steps=2)
    assert output["retrieval_cell_probability"].shape == (2, 3)
    assert torch.allclose(output["retrieval_cell_probability"].sum(1), torch.ones(2), atol=1e-6)
    assert output["refined_cell_state_sequence"].shape == (2, 2, 3, 12)
    assert output["refinement_cell_update_sequence"].shape == (2, 2, 2, 9)
    assert output["final_canonical_render"].shape == (2, 2, 2, 2, 8, 8)
    assert output["refinement_cell_canonical_plane_covariance_sequence"].shape == (
        2,
        2,
        2,
        3,
        3,
    )
    assert output["final_cell_canonical_plane_covariance"].shape == (2, 2, 3, 3)
    assert torch.all(
        torch.linalg.eigvalsh(output["final_cell_canonical_plane_covariance"]) > 0.0
    )
    assert torch.allclose(output["conditional_within_topk_cell_probability"].sum(1), torch.ones(2), atol=1e-6)
    assert torch.allclose(
        output["retrieval_topk_retained_probability"] + output["retrieval_omitted_probability"],
        torch.ones(2),
        atol=1e-7,
    )
    assert output["catalogue_complete"] is True
    assert output["probabilities_calibrated"] is False
    assert output["retrieval_tail_scope"] == "complete_catalogue_at_retrieval"
    assert output["refinement_probability_scope"] == "conditional_within_retrieved_topk"
    initial_representation_probability = output[
        "representation_log_conditional_within_cell"
    ].exp()
    expected_initial_update = (
        initial_representation_probability[..., None]
        * output["initial_representation_canonical_residual"]
    ).sum(dim=2)
    assert torch.allclose(
        output["initial_cell_canonical_residual"], expected_initial_update, atol=1e-7
    )
    eigenvalues = torch.linalg.eigvalsh(output["initial_cell_canonical_plane_covariance"])
    assert torch.all(eigenvalues > 0.0)
    _, frame, basis = full_frame_state_to_components(output["final_cell_state"])
    assert torch.allclose(frame.transpose(-1, -2) @ frame, torch.eye(3).expand_as(frame), atol=2e-5)
    assert torch.all(torch.linalg.det(frame) > 0.0)
    assert torch.all(torch.linalg.diagonal(basis, dim1=-2, dim2=-1) > 0.0)


def test_streamed_chunks_match_monolithic_retrieval_and_topk_refinement():
    model = _model().eval()
    fixture = _fixture(batch=2, cells=5)
    whole = _forward(model, fixture, top_k=3, refinement_steps=2)
    streamed = _forward_streamed(model, fixture, top_k=3, refinement_steps=2)
    assert streamed["retrieval_execution"] == "checkpointed_low_resolution_chunks"
    assert streamed["catalogue_chunk_size"] == 2
    for key in (
        "retrieval_cell_log_probability",
        "retrieval_topk_log_probability",
        "topk_initial_cell_state",
        "topk_initial_cell_canonical_plane_covariance",
        "final_cell_state",
        "final_canonical_render",
    ):
        assert torch.allclose(streamed[key], whole[key], atol=3e-6, rtol=0.0), key
    assert torch.equal(
        streamed["retrieval_topk_cell_id"], whole["retrieval_topk_cell_id"]
    )


def test_streamed_checkpointed_training_reaches_coarse_and_refinement_parameters():
    model = _model().train()
    fixture = _fixture(batch=1, cells=5)
    output = _forward_streamed(model, fixture, top_k=2, refinement_steps=2)
    loss = (
        output["retrieval_cell_log_probability"].square().mean()
        + output["final_cell_state"].square().mean()
        + output["topk_initial_cell_canonical_plane_covariance"].square().mean()
        + output["final_cell_canonical_plane_covariance"].square().mean()
    )
    loss.backward()
    for parameter in (
        model.candidate_log_likelihood.weight,
        model.candidate_update.weight,
        model.candidate_plane_cholesky.weight,
        model.recurrent_cell.gates.weight,
        model.recurrent_update.weight,
        model.recurrent_plane_cholesky.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_training_only_truth_forcing_preserves_honest_complete_retrieval():
    model = _model().train()
    fixture = _fixture(batch=1, cells=5)
    honest = _forward_streamed(model, fixture, top_k=2, refinement_steps=1)
    omitted = next(
        cell
        for cell in range(5)
        if cell not in honest["retrieval_topk_cell_id"][0].tolist()
    )
    forced = _forward_streamed(
        model,
        fixture,
        top_k=2,
        refinement_steps=1,
        training_truth_catalogue_index=torch.tensor([omitted]),
    )

    assert torch.equal(
        forced["retrieval_cell_log_probability"],
        honest["retrieval_cell_log_probability"],
    )
    assert torch.equal(
        forced["honest_retrieval_topk_cell_id"], honest["retrieval_topk_cell_id"]
    )
    assert omitted not in forced["honest_retrieval_topk_cell_id"][0].tolist()
    assert omitted in forced["retrieval_topk_cell_id"][0].tolist()
    assert forced["retrieval_teacher_forced_mask"].tolist() == [True]
    assert torch.allclose(
        forced["retrieval_topk_retained_probability"]
        + forced["retrieval_omitted_probability"],
        torch.ones(1),
        atol=1e-7,
    )

    model.eval()
    with pytest.raises(ValueError, match="training-only"):
        _forward_streamed(
            model,
            fixture,
            top_k=2,
            refinement_steps=1,
            training_truth_catalogue_index=torch.tensor([omitted]),
        )


def test_cell_order_is_equivariant_and_normalization_returns_canonical_cell_id_order():
    fixture = _fixture(batch=1)
    model = _model().eval()
    first = _score(model, fixture)
    permutation = torch.tensor([2, 0, 1])
    for key in ("states", "log_mass", "log_weight", "raster"):
        fixture[key] = fixture[key][:, permutation]
    fixture["cell_id"] = fixture["cell_id"][permutation]
    second = _score(model, fixture)
    normalized_first = model.normalize_complete_catalogue(first, 3, 2)
    normalized_second = model.normalize_complete_catalogue(second, 3, 2)
    assert torch.equal(normalized_second["retrieval_cell_id"], torch.arange(3))
    assert torch.allclose(
        normalized_first["retrieval_cell_log_probability"],
        normalized_second["retrieval_cell_log_probability"],
        atol=2e-6,
    )


def test_forward_calls_renderer_for_all_representations_at_retrieval_each_update_and_final(monkeypatch):
    calls = []
    original = recurrent.render_finite_thickness_plane

    def spy(volume, state, *args, **kwargs):
        calls.append(state.detach().clone())
        return original(volume, state, *args, **kwargs)

    monkeypatch.setattr(recurrent, "render_finite_thickness_plane", spy)
    output = _forward(_model(), _fixture(batch=1), top_k=2, refinement_steps=3)
    assert len(calls) == 5
    assert calls[0].shape == (3, FULL_FRAME_STATE_SIZE)
    assert all(call.shape == (2, FULL_FRAME_STATE_SIZE) for call in calls[1:])
    assert torch.allclose(calls[-1], output["final_cell_state"].reshape(2, 12), atol=0.0, rtol=0.0)


def test_full_gradient_path_reaches_catalogue_state_mass_covariance_and_recurrence():
    fixture = _fixture(batch=1, dtype=torch.float64)
    fixture["volume"] = fixture["volume"].requires_grad_()
    fixture["image"] = fixture["image"].requires_grad_()
    fixture["states"] = fixture["states"].detach().requires_grad_()
    fixture["log_mass"] = fixture["log_mass"].detach().requires_grad_()
    model = _model(dtype=torch.float64)
    output = _forward(model, fixture, top_k=2, refinement_steps=2)
    loss = (
        output["cell_log_unnormalized_mass"].square().mean()
        + output["initial_cell_canonical_plane_covariance"].square().mean()
        + output["initial_cell_canonical_residual"].square().mean()
        + output["final_cell_state"].square().mean()
        + output["final_canonical_render"].square().mean()
        + output["final_representation_log_score"].square().mean()
        + output["final_cell_canonical_plane_covariance"].square().mean()
    )
    loss.backward()
    for value in (fixture["volume"], fixture["image"], fixture["states"], fixture["log_mass"]):
        assert torch.isfinite(value.grad).all() and torch.count_nonzero(value.grad) > 0
    for parameter in (
        model.candidate_log_likelihood.weight,
        model.candidate_update.weight,
        model.candidate_plane_cholesky.weight,
        model.recurrent_cell.gates.weight,
        model.recurrent_update.weight,
        model.recurrent_plane_cholesky.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_float64_catalogue_states_work_with_float32_model():
    fixture = _fixture(batch=1, state_dtype=torch.float64)
    output = _forward(_model(), fixture, top_k=1, refinement_steps=1)
    assert output["final_cell_state"].dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for autocast")
def test_cuda_autocast_keeps_geometry_in_catalogue_state_dtype():
    fixture = _to(_fixture(batch=1), "cuda")
    model = _model(device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        output = _forward(model, fixture, top_k=1, refinement_steps=1)
        loss = output["final_canonical_render"].square().mean()
    loss.backward()
    assert output["final_cell_state"].dtype == fixture["states"].dtype
    assert model.recurrent_update.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for autocast")
def test_cuda_autocast_keeps_physical_plane_covariance_and_nll_finite_above_fp16_range():
    fixture = _to(_fixture(batch=1), "cuda")
    model = _model(device="cuda").train()
    with torch.no_grad():
        model.plane_tangent_scales[2] = 600.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        reference_point_state = _forward(
            model, fixture, top_k=2, refinement_steps=1
        )["final_cell_state"]
    with torch.no_grad():
        for head in (
            model.candidate_plane_cholesky,
            model.recurrent_plane_cholesky,
        ):
            head.weight.zero_()
            head.bias.zero_()
            head.bias[:2] = -2.0

    with torch.autocast("cuda", dtype=torch.float16):
        output = _forward(model, fixture, top_k=2, refinement_steps=1)
    with torch.autocast("cuda", enabled=False):
        center, frame, basis = full_frame_state_to_components(
            output["topk_initial_cell_state"][:, 0]
        )
        truth = full_frame_state_from_components(
            center + 10_000.0 * frame[..., :, 2], frame, basis
        )
    with torch.autocast("cuda", dtype=torch.float16):
        initial_nll = joint_loss.gaussian_plane_mixture_nll(
            output["topk_initial_cell_state"],
            output["topk_initial_cell_canonical_plane_covariance"],
            output["retrieval_topk_log_probability"],
            truth,
            (5.0, 5.0, 5.0),
        )
        final_log_mass = (
            output["retrieval_topk_retained_probability"].clamp_min(1e-8).log()[:, None]
            + output["conditional_within_topk_cell_log_probability"]
        )
        final_nll = joint_loss.gaussian_plane_mixture_nll(
            output["final_cell_state"],
            output["final_cell_canonical_plane_covariance"],
            final_log_mass,
            truth,
            (5.0, 5.0, 5.0),
        )

    for covariance in (
        output["topk_initial_cell_canonical_plane_covariance"],
        output["final_cell_canonical_plane_covariance"],
    ):
        assert covariance.dtype == torch.float32
        assert torch.isfinite(covariance).all()
        assert torch.all(covariance[..., 2, 2] > torch.finfo(torch.float16).max)
    assert torch.isfinite(initial_nll)
    assert torch.isfinite(final_nll)
    assert torch.isfinite(output["final_cell_state"]).all()
    assert torch.equal(output["final_cell_state"], reference_point_state)

    (initial_nll + final_nll).backward()
    for head in (
        model.candidate_plane_cholesky,
        model.recurrent_plane_cholesky,
    ):
        assert head.bias.grad is not None
        assert torch.isfinite(head.bias.grad).all()
        assert head.bias.grad[2] != 0


def test_new_model_has_no_legacy_checkpoint_filesystem_or_deformation_path():
    tree = ast.parse(inspect.getsource(recurrent))
    imports = set()
    calls = set()

    def dotted_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = dotted_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        elif isinstance(node, ast.Call):
            calls.add(dotted_name(node.func))

    forbidden = (
        "pathlib",
        "pickle",
        "timm",
        "torchvision",
        "training.atlas_pose",
        "training.dense_registration",
        "training.independent_joint",
        "training.joint_pose_registration",
    )
    assert not any(
        imported == prefix or imported.startswith(prefix + ".")
        for imported in imports
        for prefix in forbidden
    )
    assert not ({"torch.load", "open"} & calls)
    model = _model()
    assert not any(
        token in name.lower()
        for name, _ in model.named_parameters()
        for token in ("velocity", "deformation", "svf", "legacy")
    )
    output = _forward(model, _fixture(batch=1), top_k=1, refinement_steps=1)
    assert not any(
        token in key.lower()
        for key in output
        for token in ("velocity", "deformation", "svf", "dense_map")
    )
