from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from training.independent_joint_model import (
    IndependentJointModel,
    identity_pixel_map,
    project_affine_free_velocity,
    registration_maps,
)
from training.train_independent_joint import (
    _validate_development_panel,
    _scheduled_learning_rate,
    curriculum_batch,
    independent_joint_forward,
    independent_joint_loss,
    initializer_pose_losses,
    normalized_full_cholesky_nll,
    raw_prediction_records,
    shuffle_candidates,
    train_independent_joint,
    training_lineage,
)


HASH = "a" * 64
STREAM_CONTRACTS = {
    "regular_synthetic": {"contract_sha256": "4" * 64},
    "high_tilt": {"contract_sha256": "4" * 64},
    "product5": {"contract_sha256": "4" * 64, "specimen_ids": [100]},
}


class TinyRenderer:
    device = torch.device("cpu")
    contract = {
        "contract_sha256": "1" * 64,
        "average_template_sha256": "2" * 64,
        "annotation_sha256": "3" * 64,
    }

    def __init__(self, height=24, width=32):
        self.height = height
        self.width = width
        self.calls = []

    def render_planes(self, ap_index, tilt_lr_deg, tilt_dv_deg):
        ap_um = (216.0 - ap_index) * 25.0
        pose = torch.stack((ap_um, tilt_lr_deg, tilt_dv_deg), dim=1)
        self.calls.append(pose.detach().clone())
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.height, device=pose.device),
            torch.linspace(-1.0, 1.0, self.width, device=pose.device),
            indexing="ij",
        )
        image = torch.sigmoid(
            1.2 * x[None]
            - 0.7 * y[None]
            + pose[:, 0, None, None] / 2500.0
            + pose[:, 1, None, None] * x[None] / 35.0
            + pose[:, 2, None, None] * y[None] / 35.0
        )[:, None]
        mask = torch.ones_like(image, dtype=torch.bool)
        mask[:, :, :1] = False
        image = image * mask
        labels = (
            1
            + (x[None, None] > 0).long()
            + 2 * (y[None, None] > 0).long()
        ).expand(len(pose), -1, -1, -1)
        labels = labels * mask
        return image, mask, labels


def _model(seed=19):
    torch.manual_seed(seed)
    return IndependentJointModel(
        pyramid_channels=(4, 4, 4, 4),
        pose_context_features=12,
        pair_features=8,
        hidden_channels=8,
        integration_steps=1,
    )


def _batch(renderer, batch_size=2, *, dense=True):
    true_pose = torch.tensor(
        [[-1200.0 - 100.0 * item, 3.0 + item, -2.0 - item] for item in range(batch_size)]
    )
    offsets = torch.tensor(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [-100.0, 0.0, 0.0],
         [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )
    candidate_pose = true_pose[:, None] + offsets[None]
    candidate_image, candidate_mask, candidate_labels = renderer.render_planes(
        216.0 - candidate_pose[..., 0].reshape(-1) / 25.0,
        candidate_pose[..., 1].reshape(-1),
        candidate_pose[..., 2].reshape(-1),
    )
    shape = (batch_size, 7, 1, renderer.height, renderer.width)
    candidate_image = candidate_image.reshape(shape)
    candidate_mask = candidate_mask.reshape(shape)
    candidate_labels = candidate_labels.reshape(shape)
    source_image = candidate_image[:, 0].clone()
    source_mask = candidate_mask[:, 0].clone()
    fixed_labels = candidate_labels[:, 0].clone()
    identity = identity_pixel_map(
        batch_size, renderer.height, renderer.width, device=torch.device("cpu"), dtype=torch.float32
    )
    x = torch.linspace(-1.0, 1.0, renderer.width)[None, None, None]
    y = torch.linspace(-1.0, 1.0, renderer.height)[None, None, :, None]
    velocity = torch.cat(
        (
            (0.5 * torch.sin(torch.pi * x) * torch.cos(torch.pi * y)).expand(batch_size, 1, -1, -1),
            (0.4 * torch.cos(torch.pi * x) * torch.sin(torch.pi * y)).expand(batch_size, 1, -1, -1),
        ),
        dim=1,
    )
    velocity, _ = project_affine_free_velocity(velocity)
    angle = torch.linspace(-0.15, 0.15, batch_size)
    similarity = torch.stack(
        (angle.cos(), angle.sin(), torch.ones(batch_size), -torch.ones(batch_size), torch.zeros(batch_size)),
        dim=1,
    )
    forward, inverse = registration_maps(similarity, velocity, integration_steps=2)
    batch = {
        "source_type": "synthetic_ccf" if dense else "allen_registered_product5",
        "data_contract_sha256": "4" * 64,
        "data_split": "train",
        "source_image": source_image,
        "source_mask": source_mask,
        "mask_available": torch.ones(batch_size, 1, 1, 1),
        "input_outline_mode": torch.arange(batch_size, dtype=torch.int8) % 3,
        "true_pose": true_pose,
        "candidate_pose": candidate_pose,
        "candidate_fixed_image": candidate_image,
        "candidate_fixed_mask": candidate_mask,
        "candidate_fixed_labels": candidate_labels,
        "candidate_in_training_domain": torch.ones(batch_size, 7, dtype=torch.bool),
        "candidate_dense_truth_valid": torch.zeros(batch_size, 7, dtype=torch.bool),
        "listwise_target_index": torch.zeros(batch_size, dtype=torch.long),
        "listwise_positive_mask": torch.nn.functional.one_hot(
            torch.zeros(batch_size, dtype=torch.long), 7
        ).bool(),
        "dense_truth_valid": torch.full((batch_size,), dense, dtype=torch.bool),
        "animal_id": torch.arange(100, 100 + batch_size),
        "specimen_id": torch.arange(100, 100 + batch_size),
        "experiment_id": torch.arange(200, 200 + batch_size),
        "section_image_id": torch.arange(300, 300 + batch_size),
    }
    batch["candidate_dense_truth_valid"][:, 0] = dense
    if dense:
        batch.update(
            {
                "sample_manifest_sha256": "sample-manifest",
                "truth_fixed_image": candidate_image[:, 0],
                "truth_fixed_mask": candidate_mask[:, 0],
                "truth_fixed_labels": fixed_labels,
                "truth_source_labels": fixed_labels.clone(),
                "truth_svf": velocity,
                "truth_fixed_to_source_map": forward,
                "truth_source_to_fixed_map": inverse,
                "truth_fixed_valid_mask": source_mask.clone(),
                "truth_source_valid_mask": source_mask.clone(),
                "truth_similarity_parameters": similarity,
            }
        )
    else:
        batch.update(
            {
                "product_id": torch.full((batch_size,), 5),
                "batch_manifest_sha256": "product5-batch",
                "record_provenance_sha256": [f"{item + 5}" * 64 for item in range(batch_size)],
                "source_relative_path": [f"animal/{item}.jpg" for item in range(batch_size)],
            }
        )
    renderer.calls.clear()
    return batch


def test_candidate_shuffle_is_deterministic_aligned_and_has_no_positive_index_leak():
    renderer = TinyRenderer()
    batch = _batch(renderer)
    first = shuffle_candidates(batch, seed=17, counter=4)
    second = shuffle_candidates(batch, seed=17, counter=4)
    assert torch.equal(first["candidate_permutation"], second["candidate_permutation"])
    assert torch.equal(first["candidate_pose"], second["candidate_pose"])
    for row in range(2):
        for destination in range(7):
            origin = first["candidate_permutation"][row, destination]
            assert torch.equal(
                first["candidate_pose"][row, destination], batch["candidate_pose"][row, origin]
            )
            assert torch.equal(
                first["candidate_fixed_image"][row, destination],
                batch["candidate_fixed_image"][row, origin],
            )
            assert torch.equal(
                first["candidate_fixed_labels"][row, destination],
                batch["candidate_fixed_labels"][row, origin],
            )
        target = first["listwise_target_index"][row]
        assert first["listwise_positive_mask"][row].sum() == 1
        assert first["listwise_positive_mask"][row, target]
        assert torch.equal(first["candidate_pose"][row, target], batch["true_pose"][row])
    destinations = [
        int(shuffle_candidates(batch, 17, counter)["listwise_target_index"][0])
        for counter in range(7)
    ]
    assert sorted(destinations) == list(range(7))


def test_initializer_has_categorical_sub_bin_and_proper_normalized_full_cholesky_nll():
    mean = torch.tensor([[0.2, -0.3, 0.4]])
    target = torch.tensor([[0.8, 0.1, -0.2]])
    scale = torch.tensor([2.0, 3.0, 4.0])
    normalized_l = torch.tensor([[[0.8, 0.0, 0.0], [0.2, 0.7, 0.0], [-0.1, 0.3, 0.9]]])
    physical_l = normalized_l * scale[None, :, None]
    actual = normalized_full_cholesky_nll(mean, target, physical_l, scale)
    expected = -torch.distributions.MultivariateNormal(
        mean / scale, scale_tril=normalized_l
    ).log_prob(target / scale).mean()
    assert torch.allclose(actual, expected, atol=1e-6)

    model = _model()
    renderer = TinyRenderer()
    batch = _batch(renderer)
    features = model.encode_source(
        batch["source_image"], batch["source_mask"], batch["mask_available"]
    )
    losses = initializer_pose_losses(model.pose_head(features), batch["true_pose"], model)
    assert set(losses) == {
        "initializer_categorical", "initializer_sub_bin", "initializer_gaussian_nll",
        "initializer_plane_anchor",
    }
    assert all(torch.isfinite(value) for value in losses.values())


def test_forward_caches_source_once_uses_explicit_indices_and_binds_final_pose():
    model = _model()
    with torch.no_grad():
        model.pose_delta_head.bias.fill_(0.1)
    renderer = TinyRenderer()
    batch = shuffle_candidates(_batch(renderer), 9, 2)
    source_indices = []
    original_score = model.score_candidate_from_features
    original_dense = model.refine_from_features

    def score(*args, **kwargs):
        source_indices.append(kwargs.get("source_index", args[7] if len(args) > 7 else None).detach().clone())
        return original_score(*args, **kwargs)

    dense_calls = []

    def dense(*args, **kwargs):
        dense_calls.append((args, kwargs))
        return original_dense(*args, **kwargs)

    with patch.object(model, "encode_source", wraps=model.encode_source) as encode, patch.object(
        model, "score_candidate_from_features", side_effect=score
    ), patch.object(model, "refine_from_features", side_effect=dense):
        output = independent_joint_forward(model, batch, renderer)

    assert encode.call_count == 1
    assert len(source_indices) == 5  # rank + T=3 + settled-pose receipt
    assert torch.equal(source_indices[0], torch.arange(2).repeat_interleave(7))
    assert all(torch.equal(index, torch.arange(2)) for index in source_indices[1:])
    assert len(renderer.calls) == 4
    assert torch.allclose(renderer.calls[-1], output["settled_pose"].detach(), atol=5e-4, rtol=0.0)
    assert torch.equal(output["final_render_pose"], output["settled_pose"])
    assert not torch.equal(output["final_receipt"]["pose"], output["final_render_pose"])
    assert len(dense_calls) == 1
    assert torch.equal(output["dense_binding_pose"], batch["true_pose"])
    assert torch.equal(dense_calls[0][0][0], batch["truth_fixed_image"])
    assert torch.equal(dense_calls[0][0][3], batch["true_pose"])


def test_symmetric_truth_centered_candidates_cannot_seed_or_leak_into_recurrence():
    model = _model().eval()
    renderer = TinyRenderer()
    batch = shuffle_candidates(_batch(renderer, dense=False), 3, 0)
    with torch.no_grad():
        first = independent_joint_forward(model, batch, renderer)
        changed = dict(batch)
        changed["candidate_pose"] = batch["candidate_pose"] + torch.tensor([800.0, 12.0, -9.0])
        changed["candidate_fixed_image"] = torch.rand_like(batch["candidate_fixed_image"])
        changed["candidate_fixed_mask"] = torch.ones_like(batch["candidate_fixed_mask"])
        second = independent_joint_forward(model, changed, renderer)
    assert not torch.allclose(first["ranking_logits"], second["ranking_logits"])
    assert torch.equal(first["settled_pose"], second["settled_pose"])
    assert torch.equal(first["final_render_pose"], first["recurrent"][-1]["pose"])


def test_product5_never_dense_decodes_or_receives_dense_pseudo_truth():
    model = _model()
    renderer = TinyRenderer()
    batch = shuffle_candidates(_batch(renderer, dense=False), 4, 1)
    with patch.object(
        model, "refine_from_features", side_effect=AssertionError("Product-5 dense decode")
    ):
        output = independent_joint_forward(model, batch, renderer)
    assert output["dense"] is None
    assert output["dense_sample_index"].numel() == 0
    total, components = independent_joint_loss(model, output, batch)
    assert torch.isfinite(total)
    assert not any(name.startswith("dense_") for name in components)
    assert not any(name.startswith("truth_") for name in batch)

    invalid = _batch(TinyRenderer())
    invalid["candidate_dense_truth_valid"][:, 0] = False
    invalid["candidate_dense_truth_valid"][:, 1] = True
    with pytest.raises(RuntimeError, match="shuffled positive"):
        independent_joint_forward(_model(), invalid, TinyRenderer())


def test_accuracy_scale_and_quicknii_anchors_penalize_near_but_not_exact_pose():
    model = _model().eval()
    renderer = TinyRenderer()
    batch = shuffle_candidates(_batch(renderer, dense=False), 21, 0)
    output = independent_joint_forward(model, batch, renderer)
    exact = dict(output)
    exact["recurrent"] = [{**step, "pose": batch["true_pose"]} for step in output["recurrent"]]
    exact["settled_pose"] = batch["true_pose"]
    exact["final_receipt"] = {
        **output["final_receipt"],
        "compatibility_logit": torch.full((len(batch["true_pose"]),), 3.0),
    }
    offset = batch["true_pose"].new_tensor((500.0, 5.0, 0.0))
    wrong = dict(exact)
    wrong["recurrent"] = [{**step, "pose": batch["true_pose"] + offset} for step in output["recurrent"]]
    wrong["settled_pose"] = batch["true_pose"] + offset
    _, exact_components = independent_joint_loss(model, exact, batch)
    _, wrong_components = independent_joint_loss(model, wrong, batch)
    assert exact_components["recurrent_pose"] == 0
    assert exact_components["recurrent_plane_anchor"] == 0
    assert wrong_components["recurrent_pose"] > 1.0
    assert wrong_components["recurrent_plane_anchor"] > 1.0
    assert wrong_components["final_compatibility"] > exact_components["final_compatibility"]


def test_loss_routes_exact_dense_truth_only_and_gradients_reach_every_head():
    model = _model()
    renderer = TinyRenderer()
    batch = shuffle_candidates(_batch(renderer), 5, 3)
    output = independent_joint_forward(model, batch, renderer)
    total, components = independent_joint_loss(model, output, batch)
    assert all(torch.isfinite(value) for value in components.values())
    assert {
        "dense_map_forward", "dense_map_inverse", "dense_similarity", "dense_svf",
        "dense_validity", "dense_smoothness", "dense_cycle", "dense_jacobian",
        "dense_topology", "dense_region_dice", "dense_region_boundary",
    }.issubset(components)
    total.backward()
    parameters = dict(model.named_parameters())
    for name in (
        "pose_head.ap_logits.weight",
        "pose_head.local_cholesky.weight",
        "compatibility_head.weight",
        "pose_delta_head.weight",
        "similarity_head.weight",
        "decoder.velocity_head.weight",
        "decoder.validity_head.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient), name

    mixed = _batch(TinyRenderer())
    mixed["dense_truth_valid"][1] = False
    mixed["candidate_dense_truth_valid"][1] = False
    first = independent_joint_forward(_model(), mixed, TinyRenderer())
    changed = dict(mixed)
    changed["truth_svf"] = mixed["truth_svf"].clone()
    changed["truth_svf"][1] = 10000.0
    second_model = _model()
    second = independent_joint_forward(second_model, changed, TinyRenderer())
    _, first_components = independent_joint_loss(_model(), first, mixed)
    _, second_components = independent_joint_loss(second_model, second, changed)
    for name in first_components:
        if name.startswith("dense_"):
            assert torch.allclose(first_components[name], second_components[name], atol=1e-6)


def test_curriculum_and_raw_predictions_preserve_per_animal_provenance():
    providers = {
        name: (lambda selected: lambda step, counter: {"selected": selected, "step": step, "counter": counter})(name)
        for name in ("regular_synthetic", "high_tilt", "product5")
    }
    names = [curriculum_batch(providers, ("regular_synthetic", "high_tilt", "product5"), 0, i)[0] for i in range(6)]
    assert names == ["regular_synthetic", "high_tilt", "product5"] * 2

    model = _model()
    renderer = TinyRenderer()
    batch = shuffle_candidates(_batch(renderer, dense=False), 7, 0)
    output = independent_joint_forward(model, batch, renderer)
    records = raw_prediction_records(output, batch)
    assert [record["animal_id"] for record in records] == [100, 101]
    assert [record["section_image_id"] for record in records] == [300, 301]
    assert [record["record_provenance_sha256"] for record in records] == ["5" * 64, "6" * 64]
    assert all(len(record["candidate_score_softmax_uncalibrated"]) == 7 for record in records)
    assert all(len(record["candidate_pose"]) == 7 for record in records)
    assert all(len(record["initializer_cholesky"]) == 3 for record in records)
    assert all(record["mask_available"] == 1.0 for record in records)
    assert all(record["probabilities_calibrated"] is False for record in records)


def test_training_rejects_unbound_or_mismatched_stream_contracts(tmp_path):
    renderer = TinyRenderer(height=16, width=24)
    batch = _batch(renderer, batch_size=1, dense=False)
    providers = {
        name: (lambda step, counter, payload=batch: dict(payload))
        for name in ("regular_synthetic", "high_tilt", "product5")
    }
    with pytest.raises(ValueError, match="explicit contract"):
        train_independent_joint(
            _model(), renderer, providers, {"contract_sha256": "4" * 64},
            tmp_path / "opaque", 1, amp=False, resume=False,
        )
    with pytest.raises(ValueError, match="train_animal_ids"):
        train_independent_joint(
            _model(), renderer, providers, STREAM_CONTRACTS,
            tmp_path / "omitted-animals", 1, amp=False, resume=False,
        )
    with pytest.raises(ValueError, match="train_animal_ids"):
        train_independent_joint(
            _model(), renderer, providers, STREAM_CONTRACTS,
            tmp_path / "wrong-animals", 1, amp=False, resume=False,
            train_animal_ids=[101],
        )
    providers["regular_synthetic"] = lambda step, counter: {
        **batch, "data_contract_sha256": "f" * 64
    }
    with pytest.raises(RuntimeError, match="frozen data contract"):
        train_independent_joint(
            _model(), renderer, providers, STREAM_CONTRACTS,
            tmp_path / "mismatch", 1, amp=False, resume=False,
            train_animal_ids=[100],
        )


def test_checkpoint_selection_rejects_nonfresh_overlapping_or_final_panels():
    panel = {
        "partition": "validation",
        "fresh_checkpoint_step": 4,
        "panel_manifest_sha256": "b" * 64,
        "animal_ids": [20, 21],
        "selection_metric": 0.4,
        "raw_predictions": [
            {
                "animal_id": animal,
                "record_provenance_sha256": f"{animal % 10}" * 64,
                "candidate_score_softmax_uncalibrated": [1.0],
                "initializer_covariance": [[1.0]],
            }
            for animal in (20, 21)
        ],
    }
    assert _validate_development_panel(panel, 4, [1, 2]) == (0.4, "b" * 64)
    with pytest.raises(RuntimeError):
        _validate_development_panel({**panel, "partition": "final_test"}, 4, [1, 2])
    with pytest.raises(RuntimeError):
        _validate_development_panel({**panel, "fresh_checkpoint_step": 3}, 4, [1, 2])
    with pytest.raises(RuntimeError):
        _validate_development_panel({**panel, "animal_ids": [2, 20]}, 4, [1, 2])


def test_warmup_cosine_schedule_reaches_base_then_decays():
    values = [_scheduled_learning_rate(step, 10, 1e-3, 2, 0.1) for step in range(11)]
    assert values[0] == pytest.approx(5e-4)
    assert values[1] == pytest.approx(1e-3)
    assert values[2] == pytest.approx(1e-3)
    assert values[-1] == pytest.approx(1e-4)
    assert all(left >= right for left, right in zip(values[2:], values[3:]))


def test_development_selection_runs_on_ema_and_records_selected_state(tmp_path):
    renderer = TinyRenderer(height=16, width=24)
    template = _batch(renderer, batch_size=1, dense=False)

    def provider(step, counter):
        return {
            name: value.clone() if torch.is_tensor(value) else list(value) if isinstance(value, list) else value
            for name, value in template.items()
        }

    observed = {}

    def evaluator(model, step):
        observed["parameter_name"], parameter = next(iter(model.named_parameters()))
        observed["parameter"] = parameter.detach().clone()
        return {
            "partition": "validation",
            "fresh_checkpoint_step": step,
            "panel_manifest_sha256": "c" * 64,
            "panel_contract_sha256": "d" * 64,
            "animal_ids": [999],
            "selection_metric": 0.25,
            "raw_predictions": [{
                "animal_id": 999,
                "record_provenance_sha256": "e" * 64,
                "candidate_score_softmax_uncalibrated": [1.0],
                "initializer_covariance": [[1.0]],
            }],
        }

    model = _model(seed=44)
    providers = {name: provider for name in ("regular_synthetic", "high_tilt", "product5")}
    result = train_independent_joint(
        model,
        renderer,
        providers,
        STREAM_CONTRACTS,
        tmp_path,
        1,
        seed=7,
        amp=False,
        ema_decay=0.5,
        checkpoint_interval=1,
        evaluate_every=1,
        development_evaluator=evaluator,
        development_panel_contract_sha256="d" * 64,
        train_animal_ids=[100],
        resume=False,
    )
    checkpoint = torch.load(result["best_checkpoint"], weights_only=False)
    parameter_name = observed["parameter_name"]
    assert torch.equal(observed["parameter"], checkpoint["ema"][parameter_name])
    assert not torch.equal(observed["parameter"], checkpoint["model"][parameter_name])
    assert checkpoint["checkpoint_selection_state"] == "ema"
    assert checkpoint["development_panel"]["raw_predictions"][0]["animal_id"] == 999


def test_atomic_resume_is_deterministic_and_checkpoint_lineage_is_random(tmp_path):
    renderer_a = TinyRenderer(height=16, width=24)
    template = _batch(renderer_a, batch_size=1, dense=False)

    def provider(step, counter):
        return {name: value.clone() if torch.is_tensor(value) else list(value) if isinstance(value, list) else value
                for name, value in template.items()}

    providers = {name: provider for name in ("regular_synthetic", "high_tilt", "product5")}
    contract = STREAM_CONTRACTS
    torch.manual_seed(88)
    uninterrupted = _model(seed=88)
    train_independent_joint(
        uninterrupted, renderer_a, providers, contract, tmp_path / "full", 2,
        seed=31, amp=False, resume=False, train_animal_ids=[100],
    )

    renderer_b = TinyRenderer(height=16, width=24)
    torch.manual_seed(88)
    split = _model(seed=88)
    train_independent_joint(
        split, renderer_b, providers, contract, tmp_path / "resume", 2,
        seed=31, amp=False, resume=False, max_steps_this_call=1,
        train_animal_ids=[100],
    )
    resumed = _model(seed=999)
    result = train_independent_joint(
        resumed, renderer_b, providers, contract, tmp_path / "resume", 2,
        seed=31, amp=False, resume=True, train_animal_ids=[100],
    )
    for expected, actual in zip(uninterrupted.state_dict().values(), resumed.state_dict().values()):
        assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-6)
    checkpoint = torch.load(result["latest_checkpoint"], weights_only=False)
    assert checkpoint["step"] == checkpoint["data_counter"] == 2
    assert checkpoint["learned_checkpoint_dependencies"] == []
    assert checkpoint["lineage"]["initialization"] == "random"
    assert checkpoint["lineage"]["learned_checkpoint_dependencies"] == []
    assert checkpoint["lineage"]["run_config"]["planned_steps"] == 2
    assert checkpoint["lineage"]["run_config"]["optimizer"] == "AdamW"
    assert checkpoint["lineage"]["run_config"]["recurrent_steps"] == 3
    assert checkpoint["lineage"]["model_constructor"]["parameter_count"] > 0
    assert not list((tmp_path / "resume").glob("*.tmp"))
    with pytest.raises(RuntimeError, match="lineage"):
        train_independent_joint(
            _model(seed=5), renderer_b, providers, contract, tmp_path / "resume", 2,
            seed=31, learning_rate=9e-4, amp=False, resume=True,
            train_animal_ids=[100],
        )


def test_trainer_has_no_legacy_model_or_checkpoint_dependency():
    source_path = Path(__file__).parents[1] / "training" / "train_independent_joint.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported.isdisjoint(
        {
            "training.atlas_pose_models_v7",
            "training.dense_registration_model",
            "training.joint_pose_registration_model",
        }
    )
    assert "learned_checkpoint_dependencies\": []" in source
    model = _model()
    lineage = training_lineage(
        model,
        {"contract_sha256": "4" * 64},
        TinyRenderer.contract,
    )
    assert lineage["learned_checkpoint_dependencies"] == []
    assert len(lineage["lineage_sha256"]) == 64
    same = training_lineage(
        _model(seed=19), {"contract_sha256": "4" * 64}, TinyRenderer.contract
    )
    different = training_lineage(
        _model(seed=20), {"contract_sha256": "4" * 64}, TinyRenderer.contract
    )
    assert lineage["initial_state_sha256"] == same["initial_state_sha256"]
    assert lineage["initial_state_sha256"] != different["initial_state_sha256"]


def test_lineage_distinguishes_leader_factorized_and_attention_sources_and_graphs():
    from training.independent_joint_variants import (
        FactorizedCNNControl,
        RecurrentAttentionVariant,
    )

    arguments = dict(
        pyramid_channels=(4, 4, 4, 4),
        pose_context_features=12,
        pair_features=8,
        hidden_channels=8,
        integration_steps=1,
    )
    models = (
        IndependentJointModel(**arguments),
        FactorizedCNNControl(**arguments, fusion_channels=8),
        RecurrentAttentionVariant(**arguments, attention_channels=4),
    )
    lineages = [
        training_lineage(
            model, {"contract_sha256": "4" * 64}, TinyRenderer.contract, {"screen": "unit"}
        )
        for model in models
    ]
    assert len({lineage["lineage_sha256"] for lineage in lineages}) == 3
    assert len({lineage["model_constructor"]["fully_qualified_class"] for lineage in lineages}) == 3
    assert all(lineage["model_constructor"]["parameter_count"] > 0 for lineage in lineages)
    assert all(len(lineage["model_constructor"]["architecture_source_sha256"]) == 64 for lineage in lineages)
