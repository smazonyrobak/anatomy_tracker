import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

import training.train_dense_registration as training
from training.dense_registration_model import DenseRegistrationModel, identity_pixel_map
from training.synthetic_registration import STRATA
from training.train_dense_registration import (
    _evaluate_records,
    _prepare_evaluation_records,
    _sample_metrics,
    build_parser,
    capture_rng_state,
    evaluate_model,
    evaluation_sample_seed,
    identity_training_batch,
    registration_loss,
    restore_rng_state,
    similarity_parameters_from_homography,
    summarize_metrics,
    training_batch_seed,
    training_config_from_args,
    write_registration_qa,
)


def identity_batch(height=24, width=28):
    fixed = torch.rand(1, 1, height, width)
    mask = torch.ones(1, 1, height, width, dtype=torch.bool)
    empty = torch.zeros_like(mask)
    labels = torch.ones(1, 1, height, width, dtype=torch.long)
    labels[:, :, :, width // 2 :] = 17
    identity = identity_pixel_map(1, height, width)
    return {
        "fixed": fixed,
        "moving": fixed.clone(),
        "fixed_mask": mask,
        "moving_tissue_mask": mask.clone(),
        "moving_damage_mask": empty.clone(),
        "moving_visible_mask": mask.clone(),
        "moving_model_mask": mask.clone(),
        "fixed_damage_mask": empty.clone(),
        "fixed_visible_mask": mask.clone(),
        "fixed_labels": labels,
        "moving_labels": labels.clone(),
        "fixed_to_moving": identity,
        "moving_to_fixed": identity,
        "local_velocity": torch.zeros_like(identity),
        "similarity_h": torch.eye(3)[None],
    }


def test_homography_target_recovers_model_similarity_convention():
    height, width = 41, 47
    angle, tx, ty, log_scale = 0.13, 2.5, -3.0, 0.08
    scale = np.exp(log_scale)
    matrix = torch.tensor(
        [[scale * np.cos(angle), -scale * np.sin(angle)],
         [scale * np.sin(angle), scale * np.cos(angle)]],
        dtype=torch.float32,
    )
    centre = torch.tensor(((width - 1) / 2, (height - 1) / 2))
    homography = torch.eye(3)[None]
    homography[0, :2, :2] = matrix
    homography[0, :2, 2] = centre + torch.tensor((tx, ty)) - matrix @ centre
    observed = similarity_parameters_from_homography(homography, (height, width))
    assert torch.allclose(
        observed, torch.tensor([[angle, tx, ty, log_scale]]), atol=1e-6
    )


def test_registration_objective_uses_two_channel_masks_and_backpropagates():
    model = DenseRegistrationModel(
        channels=(4, 8), correlation_radii=(1, 1), integration_steps=2
    )
    batch = identity_batch()
    batch["moving_model_mask"] = torch.zeros_like(batch["moving_model_mask"])
    observed_inputs = []
    handle = model.encoder.register_forward_pre_hook(
        lambda _module, inputs: observed_inputs.append(inputs[0].detach().clone())
    )
    loss, terms, details = registration_loss(model, batch)
    handle.remove()
    assert set(terms) == {
        "forward_flow", "inverse_flow", "similarity", "deep_flow", "regions",
        "structure", "smoothness", "inverse_cycle", "topology", "total",
    }
    assert torch.isfinite(loss)
    assert details["fixed_to_moving_map"].shape == (1, 2, 24, 28)
    assert torch.equal(observed_inputs[1][:, 1:], batch["moving_model_mask"].float())
    loss.backward()
    assert model.encoder.stem[0].weight.grad is not None


def test_v2_registration_uses_model_mask_and_weighted_visible_damage_targets():
    torch.manual_seed(17)
    model = DenseRegistrationModel(
        channels=(4, 8), correlation_radii=(1, 1), integration_steps=2
    )
    batch = identity_batch()
    height, width = batch["fixed"].shape[-2:]
    fixed_visible = torch.zeros_like(batch["fixed_mask"])
    fixed_visible[:, :, :, : width // 2] = True
    fixed_damage = batch["fixed_mask"] & ~fixed_visible
    moving_visible = torch.zeros_like(batch["moving_tissue_mask"])
    moving_visible[:, :, : height // 2, :] = True
    moving_damage = batch["moving_tissue_mask"] & ~moving_visible
    model_mask = torch.zeros_like(batch["moving_tissue_mask"])
    model_mask[:, :, 2:-2, 3:-3] = True
    batch.update(
        fixed_visible_mask=fixed_visible,
        fixed_damage_mask=fixed_damage,
        moving_visible_mask=moving_visible,
        moving_damage_mask=moving_damage,
        moving_model_mask=model_mask,
    )
    observed_inputs = []
    handle = model.encoder.register_forward_pre_hook(
        lambda _module, inputs: observed_inputs.append(inputs[0].detach().clone())
    )
    _, terms, details = registration_loss(model, batch, damage_flow_weight=0.1)
    handle.remove()

    forward_weights = fixed_visible.float() + 0.1 * fixed_damage.float()
    inverse_weights = moving_visible.float() + 0.1 * moving_damage.float()
    assert torch.equal(observed_inputs[1][:, 1:].bool(), model_mask)
    assert terms["forward_flow"] == pytest.approx(float(training._robust_endpoint_loss(
        details["fixed_to_moving_map"], batch["fixed_to_moving"], forward_weights
    ).detach()))
    assert terms["inverse_flow"] == pytest.approx(float(training._robust_endpoint_loss(
        details["moving_to_fixed_map"], batch["moving_to_fixed"], inverse_weights
    ).detach()))
    expected_deep = []
    for velocity in details["pyramid_velocities"]:
        target = training.resize_vector_field(
            batch["local_velocity"], velocity.shape[-2:]
        )
        mask = F.interpolate(
            forward_weights, size=velocity.shape[-2:], mode="nearest"
        )
        expected_deep.append(training._robust_endpoint_loss(velocity, target, mask))
    assert terms["deep_flow"] == pytest.approx(
        float(torch.stack(expected_deep).mean().detach())
    )

    expected_region = training.sampled_region_loss(
        batch["fixed_labels"],
        batch["moving_labels"],
        details["fixed_to_moving_map"],
        fixed_visible,
    )
    fixed_descriptor = training.modality_independent_descriptor(batch["fixed"])
    moving_descriptor = training.modality_independent_descriptor(batch["moving"])
    warped_descriptor = training.warp_tensor(
        moving_descriptor, details["fixed_to_moving_map"], padding_mode="border"
    )
    expected_structure = training._masked_mean(
        (fixed_descriptor - warped_descriptor).abs(), fixed_visible
    )
    identity = identity_pixel_map(1, height, width)
    expected_cycle = training._robust_endpoint_loss(
        training.compose_pixel_maps(
            details["fixed_to_moving_map"], details["moving_to_fixed_map"]
        ),
        identity,
        batch["fixed_mask"],
    )
    expected_topology = training._masked_mean(
        F.relu(0.05 - training.jacobian_determinant(
            details["fixed_to_moving_map"]
        )).square(),
        batch["fixed_mask"],
    )
    assert terms["regions"] == pytest.approx(float(expected_region.detach()))
    assert terms["structure"] == pytest.approx(float(expected_structure.detach()))
    assert terms["inverse_cycle"] == pytest.approx(float(expected_cycle.detach()))
    assert terms["topology"] == pytest.approx(float(expected_topology.detach()))


def test_registration_uses_only_the_production_v2_batch_contract():
    torch.manual_seed(19)
    model = DenseRegistrationModel(
        channels=(4, 8), correlation_radii=(1, 1), integration_steps=2
    )
    batch = identity_batch()
    loss, _, _ = registration_loss(model, batch)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("weight", (-0.01, 1.01, float("nan")))
def test_damage_flow_weight_must_be_a_finite_fraction(weight):
    model = DenseRegistrationModel(
        channels=(4, 8), correlation_radii=(1, 1), integration_steps=2
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        registration_loss(
            model,
            identity_batch(),
            damage_flow_weight=weight,
        )


def test_boundary_endpoint_default_is_numerically_identical_to_unweighted_flow():
    torch.manual_seed(4)
    model = DenseRegistrationModel(
        channels=(4, 8), correlation_radii=(1, 1), integration_steps=2
    )
    batch = identity_batch()
    _, implicit, _ = registration_loss(model, batch)
    _, explicit, details = registration_loss(
        model, batch, boundary_endpoint_weight=1.0
    )
    expected_forward = training._robust_endpoint_loss(
        details["fixed_to_moving_map"],
        batch["fixed_to_moving"],
        batch["fixed_visible_mask"],
    )
    expected_inverse = training._robust_endpoint_loss(
        details["moving_to_fixed_map"],
        batch["moving_to_fixed"],
        batch["moving_visible_mask"],
    )
    assert implicit == explicit
    assert explicit["forward_flow"] == float(expected_forward.detach())
    assert explicit["inverse_flow"] == float(expected_inverse.detach())


def test_boundary_endpoint_threefold_weighting_is_exact_symmetric_and_final_only():
    torch.manual_seed(5)
    model = DenseRegistrationModel(
        channels=(4, 8), correlation_radii=(1, 1), integration_steps=2
    )
    batch = identity_batch()
    batch["moving_labels"] = torch.ones_like(batch["moving_labels"])
    batch["moving_labels"][:, :, batch["moving_labels"].shape[-2] // 2 :, :] = 23
    batch["fixed_visible_mask"][:, :, 0] = False
    batch["moving_visible_mask"][:, :, :, 0] = False

    _, unweighted, _ = registration_loss(model, batch)
    _, weighted, details = registration_loss(
        model, batch, boundary_endpoint_weight=3.0
    )
    fixed_weights = batch["fixed_visible_mask"].float() * (
        1.0 + 2.0 * training.label_boundary(batch["fixed_labels"]).float()
    )
    moving_weights = batch["moving_visible_mask"].float() * (
        1.0 + 2.0 * training.label_boundary(batch["moving_labels"]).float()
    )
    expected_forward = training._robust_endpoint_loss(
        details["fixed_to_moving_map"], batch["fixed_to_moving"], fixed_weights
    )
    expected_inverse = training._robust_endpoint_loss(
        details["moving_to_fixed_map"], batch["moving_to_fixed"], moving_weights
    )
    assert set(torch.unique(fixed_weights).tolist()) == {0.0, 1.0, 3.0}
    assert set(torch.unique(moving_weights).tolist()) == {0.0, 1.0, 3.0}
    assert weighted["forward_flow"] == float(expected_forward.detach())
    assert weighted["inverse_flow"] == float(expected_inverse.detach())
    for name in set(weighted) - {"forward_flow", "inverse_flow", "total"}:
        assert weighted[name] == unweighted[name]


def test_boundary_endpoint_weight_rejects_values_below_one():
    model = DenseRegistrationModel(
        channels=(4, 8), correlation_radii=(1, 1), integration_steps=2
    )
    with pytest.raises(ValueError, match="at least 1.0"):
        registration_loss(model, identity_batch(), boundary_endpoint_weight=0.99)


def test_identity_warmup_replaces_all_geometry_targets():
    batch = identity_batch()
    batch["moving"] = torch.zeros_like(batch["moving"])
    warmed = identity_training_batch(batch)
    assert torch.equal(warmed["moving"], warmed["fixed"])
    assert torch.equal(warmed["moving_labels"], warmed["fixed_labels"])
    assert torch.equal(warmed["fixed_to_moving"], batch["moving_to_fixed"])
    assert torch.count_nonzero(warmed["local_velocity"]) == 0
    for name in (
        "moving_tissue_mask",
        "moving_visible_mask",
        "moving_model_mask",
        "fixed_visible_mask",
    ):
        assert torch.equal(warmed[name], batch["fixed_mask"])
    assert not warmed["moving_damage_mask"].any()
    assert not warmed["fixed_damage_mask"].any()


def test_identity_metrics_are_exact():
    batch = identity_batch()
    samples = _sample_metrics(
        batch, batch["fixed_to_moving"], batch["moving_to_fixed"]
    )
    summary = summarize_metrics(samples)
    assert summary["foreground_correspondence"] == 1.0
    assert summary["macro_region_dice"] == 1.0
    assert summary["endpoint_p95_px"] == 0.0
    assert summary["inverse_endpoint_p95_px"] == 0.0
    assert summary["damage_pixel_count"] == 0
    assert summary["damaged_sample_count"] == 0
    assert summary["sample_damage_endpoint_p95_q95_px"] is None
    assert summary["sample_endpoint_p95_q95_px"] == 0.0
    assert summary["inverse_cycle_p95_px"] == pytest.approx(0.0, abs=1e-5)
    assert summary["reverse_cycle_p95_px"] == pytest.approx(0.0, abs=1e-5)
    assert summary["fold_count"] == 0
    assert summary["inverse_fold_count"] == 0
    assert summary["jacobian_min"] == 1.0
    assert summary["inverse_jacobian_min"] == 1.0


def test_primary_region_metrics_sample_the_moving_labels_directly():
    batch = identity_batch()
    batch["moving_labels"] = torch.where(
        batch["moving_labels"] == 1,
        torch.tensor(17),
        torch.tensor(1),
    )
    summary = summarize_metrics(
        _sample_metrics(batch, batch["fixed_to_moving"], batch["moving_to_fixed"])
    )
    assert summary["foreground_correspondence"] == 0.0
    assert summary["analytic_foreground_correspondence"] == 1.0


def test_metrics_cover_damaged_tissue_and_report_damage_tails():
    batch = identity_batch(height=20, width=20)
    batch["fixed_visible_mask"][:, :, :, :10] = True
    batch["fixed_visible_mask"][:, :, :, 10:] = False
    batch["fixed_damage_mask"] = batch["fixed_mask"] & ~batch["fixed_visible_mask"]
    forward = batch["fixed_to_moving"].clone()
    forward[:, 0, :, 10:] = 0.0
    summary = summarize_metrics(
        _sample_metrics(batch, forward, batch["moving_to_fixed"])
    )
    assert summary["foreground_correspondence"] == 0.5
    assert summary["damage_endpoint_p95_px"] > 10.0
    assert summary["damage_pixel_count"] == 200
    assert summary["damaged_sample_count"] == 1
    assert summary["sample_damage_endpoint_p95_q95_px"] > 10.0
    assert summary["endpoint_p99_px"] >= summary["endpoint_p95_px"] > 0.0
    assert summary["sample_foreground_correspondence_q05"] == 0.5
    assert summary["sample_macro_region_dice_q05"] == summary["macro_region_dice"]


def test_internal_boundaries_exclude_background_edges_only():
    labels = torch.zeros(1, 1, 8, 8, dtype=torch.long)
    labels[:, :, 2:6, 2:6] = 3
    assert not training.internal_label_boundary(labels).any()
    labels[:, :, 2:6, 4:6] = 9
    boundary = training.internal_label_boundary(labels)
    assert boundary.any()
    assert not boundary[:, :, 1].any()
    assert not boundary[:, :, 6].any()


def test_broken_inverse_fails_even_when_forward_is_perfect():
    batch = identity_batch()
    inverse = batch["moving_to_fixed"].clone()
    inverse[:, 0] = inverse.shape[-1] - 1 - inverse[:, 0]
    summary = summarize_metrics(
        _sample_metrics(batch, batch["fixed_to_moving"], inverse)
    )
    assert summary["endpoint_p95_px"] == 0.0
    assert summary["foreground_correspondence"] == 1.0
    assert summary["inverse_endpoint_p95_px"] > 2.0
    assert summary["reverse_cycle_p95_px"] > 1.0
    assert summary["inverse_fold_count"] > 0


def test_sealed_split_is_not_reachable_through_development_evaluation():
    with pytest.raises(ValueError, match="only the validation split"):
        evaluate_model(None, None, split="sealed-test")


class IdentityModel(torch.nn.Module):
    def forward(self, fixed, moving):
        identity = identity_pixel_map(
            fixed.shape[0], fixed.shape[-2], fixed.shape[-1],
            device=fixed.device, dtype=fixed.dtype,
        )
        return identity, identity


def test_qa_disagreement_preserves_full_integer_allen_ids(tmp_path):
    batch = identity_batch()
    batch["fixed"].zero_()
    batch["moving"].zero_()
    batch["fixed_labels"].fill_(2**24 + 1)
    batch["moving_labels"].fill_(2**24 + 1)
    output = write_registration_qa(
        IdentityModel(), batch, tmp_path / "qa.png", maximum_items=1
    )
    image = np.asarray(Image.open(output))
    width = batch["fixed"].shape[-1]
    assert not image[24:, 3 * width : 4 * width].any()


class IdentityGenerator:
    def make_manifest(self, count, split, seed, stratum, **_kwargs):
        payload = f"{count}:{split}:{seed}:{stratum}".encode()
        return {
            "seed": seed,
            "stratum": stratum,
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        }

    def batch(self, _manifest):
        return identity_batch()


def test_evaluation_cohort_and_metrics_do_not_depend_on_batching_or_order():
    model = IdentityModel()
    generator = IdentityGenerator()
    first = evaluate_model(
        model, generator, samples_per_stratum=3, batch_size=1, seed=704
    )
    batched = evaluate_model(
        model, generator, samples_per_stratum=3, batch_size=5, seed=704
    )
    assert first["evaluation_samples"] == batched["evaluation_samples"]
    assert first["evaluation_manifest_sha256"] == batched["evaluation_manifest_sha256"]
    assert first["overall"] == batched["overall"]
    records = _prepare_evaluation_records(
        generator, "validation", 3, 704, tuple(STRATA)
    )
    ordered, _ = _evaluate_records(model, generator, records, 4, tuple(STRATA))
    reversed_order, _ = _evaluate_records(
        model, generator, list(reversed(records)), 4, tuple(STRATA)
    )
    assert ordered == reversed_order


def test_development_evaluation_cache_materializes_each_manifest_once():
    class CountingGenerator(IdentityGenerator):
        def __init__(self):
            self.calls = 0

        def batch(self, manifest):
            self.calls += 1
            return super().batch(manifest)

    generator = CountingGenerator()
    cache = {}
    first = evaluate_model(
        IdentityModel(), generator, samples_per_stratum=2, batch_size=2,
        seed=704, pair_cache=cache,
    )
    second = evaluate_model(
        IdentityModel(), generator, samples_per_stratum=2, batch_size=3,
        seed=704, pair_cache=cache,
    )
    assert generator.calls == 2 * len(STRATA)
    assert len(cache) == 2 * len(STRATA)
    assert first["overall"] == second["overall"]


def test_evaluation_reports_appearance_and_mask_offset_subgroups():
    class DescriptorGenerator(IdentityGenerator):
        def make_manifest(self, count, split, seed, stratum, **kwargs):
            manifest = super().make_manifest(count, split, seed, stratum, **kwargs)
            manifest["moving_appearance_mode"] = np.asarray([f"mode-{stratum}"])
            manifest["mask_offset_px"] = np.asarray([
                float(tuple(STRATA).index(stratum))
            ])
            return manifest

    generator = DescriptorGenerator()
    records = _prepare_evaluation_records(
        generator, "validation", 1, 704, tuple(STRATA)
    )
    metrics, oracle = _evaluate_records(
        IdentityModel(), generator, records, 2, tuple(STRATA)
    )
    assert set(metrics["appearance_subgroups"]) == {
        f"mode-{stratum}" for stratum in STRATA
    }
    assert set(metrics["mask_offset_subgroups"]) == {
        str(float(index)) for index, _ in enumerate(STRATA)
    }
    assert metrics["appearance_subgroups"] == oracle["appearance_subgroups"]
    assert metrics["mask_offset_subgroups"] == oracle["mask_offset_subgroups"]


def test_training_and_evaluation_seed_domains_are_deterministic_and_disjoint():
    training_seeds = {
        training_batch_seed(base, ordinal)
        for base in (4, 9)
        for ordinal in range(20)
    }
    evaluation_seeds = {
        evaluation_sample_seed(base, stratum, sample)
        for base in (4, 9)
        for stratum in range(3)
        for sample in range(20)
    }
    assert len(training_seeds) == 40
    assert len(evaluation_seeds) == 120
    assert training_seeds.isdisjoint(evaluation_seeds)
    assert training_batch_seed(4, 3) == training_batch_seed(4, 3)


def test_training_cli_exposes_only_the_selected_v2_winner():
    default_args = build_parser().parse_args(["train"])
    default_config = training_config_from_args(default_args)
    assert default_config["damage_flow_weight"] == 0.1
    assert default_config["boundary_endpoint_weight"] == 1.0
    assert default_config["early_stopping_patience_checkpoints"] == 0
    assert default_config["model"] == training.model_config(
        (16, 24, 32, 48),
        (4, 3, 2, 2),
    )
    args = build_parser().parse_args(
        [
            "train",
            "--channels", "8,16,24",
            "--correlation-radii", "3,2,1",
            "--damage-flow-weight", "0.25",
            "--boundary-endpoint-weight", "3",
            "--early-stopping-patience-checkpoints", "4",
        ]
    )
    config = training_config_from_args(args)
    assert config["boundary_endpoint_weight"] == 3.0
    assert config["damage_flow_weight"] == 0.25
    assert config["early_stopping_patience_checkpoints"] == 4
    assert config["model"] == training.model_config((8, 16, 24), (3, 2, 1))
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--boundary-endpoint-weight", "0.5"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--damage-flow-weight", "1.01"])
    with pytest.raises(ValueError, match="same number of stages"):
        training_config_from_args(build_parser().parse_args([
            "train", "--channels", "8,16", "--correlation-radii", "2"
        ]))
    with pytest.raises(ValueError, match="cannot be negative"):
        training_config_from_args(build_parser().parse_args([
            "train", "--early-stopping-patience-checkpoints", "-1"
        ]))


@pytest.mark.parametrize(
    "obsolete_option",
    (
        "--generator-profile",
        "--full-resolution-refiner",
        "--registration-estimator",
        "--self-similarity-features",
        "--recurrent-sequence-supervision",
    ),
)
def test_obsolete_architecture_and_generator_options_are_absent(obsolete_option):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", obsolete_option])


@pytest.mark.parametrize("command", ("freeze-candidate", "sealed-test", "export"))
def test_v1_release_commands_are_absent(command):
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


def test_checkpoint_reconstructs_the_selected_model(tmp_path):
    config = training.model_config((4, 8), (1, 1))
    model = training.build_model(config, "cpu")
    checkpoint = tmp_path / "winner.pt"
    torch.save(
        {"model_config": config, "model": model.state_dict()},
        checkpoint,
    )
    restored, payload = training.model_from_checkpoint(
        checkpoint, "cpu", use_ema=False
    )
    assert payload["model_config"] == config
    assert all(
        torch.equal(model.state_dict()[name], restored.state_dict()[name])
        for name in model.state_dict()
    )


def test_new_run_can_initialize_only_compatible_ema_weights(tmp_path):
    config = training.model_config((4, 8), (1, 1))
    source = training.build_model(config, "cpu")
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.125)
    checkpoint = tmp_path / "initializer.pt"
    torch.save(
        {
            "model_config": config,
            "ema": {"shadow": source.state_dict()},
        },
        checkpoint,
    )
    target = training.build_model(config, "cpu")
    training.load_ema_initialization(target, config, checkpoint)
    assert all(
        torch.equal(source.state_dict()[name], target.state_dict()[name])
        for name in source.state_dict()
    )
    incompatible = dict(config, channels=(4, 12))
    with pytest.raises(ValueError, match="architecture differs"):
        training.load_ema_initialization(target, incompatible, checkpoint)


def test_legacy_v13_ema_initializes_all_180_winner_tensors(tmp_path):
    config = training.model_config((16, 24, 32, 48), (4, 3, 2, 2))
    source = training.build_model(config, "cpu")
    legacy_config = {**config, **training.LEGACY_V13_MODEL_PARAMETERS}
    checkpoint = tmp_path / "v13.pt"
    torch.save(
        {"model_config": legacy_config, "ema": {"shadow": source.state_dict()}},
        checkpoint,
    )
    target = training.build_model(config, "cpu")
    training.load_ema_initialization(target, config, checkpoint)
    assert len(target.state_dict()) == 180
    assert all(
        torch.equal(target.state_dict()[name], source.state_dict()[name])
        for name in source.state_dict()
    )


def test_legacy_initialization_rejects_missing_changed_or_extra_config(tmp_path):
    config = training.model_config((4, 8), (1, 1))
    model = training.build_model(config, "cpu")
    valid = {**config, **training.LEGACY_V13_MODEL_PARAMETERS}
    invalid = []
    for name, value in training.LEGACY_V13_MODEL_PARAMETERS.items():
        changed = dict(valid)
        changed[name] = not value if isinstance(value, bool) else f"not-{value}"
        invalid.append(changed)
        missing = dict(valid)
        missing.pop(name)
        invalid.append(missing)
    extra = dict(valid, mystery_branch="enabled")
    invalid.append(extra)
    checkpoint = tmp_path / "invalid.pt"
    for candidate in invalid:
        torch.save(
            {"model_config": candidate, "ema": {"shadow": model.state_dict()}},
            checkpoint,
        )
        with pytest.raises(ValueError, match="legacy model config|model config keys"):
            training.load_ema_initialization(model, config, checkpoint)


@pytest.mark.parametrize(
    "altered_setting",
    ("model", "damage_flow_weight", "boundary_endpoint_weight"),
)
def test_resume_rejects_checkpoint_config_before_loading_state(
    tmp_path, monkeypatch, altered_setting
):
    class SentinelModel(torch.nn.Linear):
        state_loaded = False

        def load_state_dict(self, state_dict, strict=True, assign=False):
            self.state_loaded = True
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

    class ContractOnlyGenerator:
        contract = {}

    model = SentinelModel(1, 1)
    monkeypatch.setattr(training, "build_model", lambda *_args: model)
    monkeypatch.setattr(
        training,
        "SyntheticRegistrationGenerator",
        lambda *_args, **_kwargs: ContractOnlyGenerator(),
    )
    config = {
        "device": "cpu",
        "seed": 7,
        "workspace": str(tmp_path),
        "run_name": "resume-integrity",
        "atlas": "unused",
        "model": training.model_config((4, 8), (1, 1)),
        "boundary_endpoint_weight": 1.0,
        "damage_flow_weight": 0.1,
        "early_stopping_patience_checkpoints": 0,
        "ema_decay": 0.99,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "total_views": 1,
        "batch_size": 1,
        "scheduler_warmup_views": 1,
        "amp": False,
        "data_seed": 8,
        "resume": True,
    }
    normalized = json.loads(json.dumps(config))
    normalized.pop("resume")
    run_folder = tmp_path / "runs" / config["run_name"]
    run_folder.mkdir(parents=True)
    (run_folder / "config.json").write_text(json.dumps(normalized), encoding="utf-8")
    altered = json.loads(json.dumps(normalized))
    if altered_setting == "model":
        altered["model"]["channels"] = [4, 12]
    elif altered_setting == "damage_flow_weight":
        altered["damage_flow_weight"] = 0.25
    else:
        altered["boundary_endpoint_weight"] = 3.0
    torch.save({"config": altered}, run_folder / "latest.pt")

    with pytest.raises(ValueError, match="checkpoint config differs"):
        training.train(config)
    assert not model.state_loaded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint restore")
def test_rng_restore_accepts_a_checkpoint_loaded_onto_cuda():
    rng = np.random.default_rng(91)
    state = capture_rng_state(rng)
    state["torch_cpu"] = state["torch_cpu"].cuda()
    state["torch_cuda"] = [value.cuda() for value in state["torch_cuda"]]
    restore_rng_state(state, rng)


def test_rng_state_is_reproducible():
    rng = np.random.default_rng(91)
    state = capture_rng_state(rng)
    expected = (random_draw := rng.random(), torch.rand(1))
    restore_rng_state(state, rng)
    assert rng.random() == random_draw
    assert torch.equal(torch.rand(1), expected[1])


def test_training_early_stops_after_configured_validation_patience(
    tmp_path, monkeypatch
):
    class Generator:
        contract = {"test": "contract"}

        def generate(self, *_args, **_kwargs):
            return {}

    model = torch.nn.Linear(1, 1)
    scores = iter((1.0, 0.9, 0.8))
    metrics = {
        "overall": {
            "foreground_correspondence": 0.9,
            "macro_region_dice": 0.9,
            "boundary_f1_2px": 0.9,
            "endpoint_p95_px": 1.0,
            "fold_count": 0,
        }
    }
    monkeypatch.setattr(training, "SyntheticRegistrationGenerator", lambda *_a, **_k: Generator())
    monkeypatch.setattr(training, "build_model", lambda *_a, **_k: model)
    monkeypatch.setattr(
        training,
        "registration_loss",
        lambda current, *_a, **_k: (
            sum(parameter.square().sum() for parameter in current.parameters()),
            {"total": 1.0, "forward_flow": 0.5, "inverse_flow": 0.5, "regions": 0.1},
            {},
        ),
    )
    monkeypatch.setattr(training, "evaluate_model", lambda *_a, **_k: metrics)
    monkeypatch.setattr(training, "validation_score", lambda _metrics: next(scores))
    monkeypatch.setattr(training, "write_registration_qa", lambda *_a, **_k: None)
    checkpoint_writes = []
    save_checkpoint = training.save_checkpoint

    def recording_save_checkpoint(path, state):
        checkpoint_writes.append(Path(path).name)
        return save_checkpoint(path, state)

    monkeypatch.setattr(training, "save_checkpoint", recording_save_checkpoint)
    config = {
        "format_version": training.FORMAT_VERSION,
        "workspace": str(tmp_path),
        "atlas": "unused",
        "run_name": "early-stop",
        "device": "cpu",
        "seed": 7,
        "data_seed": 8,
        "model": training.model_config((4, 8), (1, 1)),
        "total_views": 8,
        "batch_size": 1,
        "identity_warmup_views": 0,
        "scheduler_warmup_views": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "gradient_clip": 2.0,
        "ema_decay": 0.99,
        "amp": False,
        "checkpoint_every_views": 1,
        "progress_every_seconds": 0.0,
        "validation_samples_per_stratum": 1,
        "validation_batch_size": 1,
        "validation_seed": 9,
        "early_stopping_patience_checkpoints": 2,
        "resume": True,
    }

    best = training.train(config)

    progress = json.loads((best.parent / "progress.json").read_text())
    latest = training.load_checkpoint(best.parent / "latest.pt")
    assert progress["status"] == "early_stopped"
    assert progress["completed_views"] == 3
    assert latest["validation_checkpoints_without_improvement"] == 2
    assert training.load_checkpoint(best)["completed_views"] == 1
    assert checkpoint_writes[:3] == [
        "views-000000001.pt",
        "best-validation.pt",
        "latest.pt",
    ]

    checkpoint_writes.clear()
    assert training.train(config) == best
    assert checkpoint_writes == []
