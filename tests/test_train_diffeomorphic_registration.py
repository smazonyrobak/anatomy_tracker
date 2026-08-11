import torch

from training.diffeomorphic_registration_model import (
    jacobian_determinant,
    pixel_identity_grid,
    remove_global_affine,
)
from training.train_diffeomorphic_registration import (
    RegistrationWithRejector,
    VALIDATION_STRATA,
    EMA,
    export_gate_failures,
    hierarchical_label_dice_loss,
    make_synthetic_pair,
    registration_objective,
)


def toy_plane(batch: int = 2, height: int = 32, width: int = 48):
    y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij")
    mask = ((x / 0.88).square() + (y / 0.78).square() < 1.0)[None, None].repeat(batch, 1, 1, 1)
    labels = torch.zeros(batch, 1, height, width, dtype=torch.long)
    labels[:, 0][mask[:, 0]] = 1
    labels[:, 0][((x + 0.35).square() + (y * 1.4).square() < 0.10)[None].repeat(batch, 1, 1)] = 2
    labels[:, 0][((x - 0.28).square() + (y + 0.10).square() < 0.08)[None].repeat(batch, 1, 1)] = 4
    labels[:, 0][((x * 1.8).square() + (y - 0.22).square() < 0.06)[None].repeat(batch, 1, 1)] = 8
    template = ((0.35 + 0.25 * torch.cos(8 * x) + 0.2 * torch.sin(7 * y))[None, None]).repeat(batch, 1, 1, 1)
    return template.clamp(0, 1), labels, mask.float()


def test_exact_deformation_target_is_positive_j_and_affine_free():
    template, labels, mask = toy_plane()
    pair = make_synthetic_pair(template, labels, mask, seed=12, stratum="smooth_deformation")
    affine_component = pair["target_velocity"] - remove_global_affine(pair["target_velocity"])

    assert pair["fixed"].shape == pair["moving"].shape == (2, 1, 32, 48)
    assert pair["target_atlas_to_affine"].shape == (2, 2, 32, 48)
    assert jacobian_determinant(pair["target_atlas_to_affine"]).min() > 0.0
    assert affine_component.abs().max() < 2e-5


def test_nuisance_damage_changes_observation_but_not_known_target_flow():
    template, labels, mask = toy_plane()
    smooth = make_synthetic_pair(template, labels, mask, seed=91, stratum="smooth_deformation")
    damaged = make_synthetic_pair(template, labels, mask, seed=91, stratum="nuisance_damage")

    assert torch.equal(smooth["target_velocity"], damaged["target_velocity"])
    assert torch.equal(smooth["target_atlas_to_affine"], damaged["target_atlas_to_affine"])
    assert not torch.equal(smooth["moving"], damaged["moving"])
    assert damaged["moving_mask"].sum() <= smooth["moving_mask"].sum()


def test_identity_extreme_modality_pair_has_exact_identity_target():
    template, labels, mask = toy_plane()
    pair = make_synthetic_pair(template, labels, mask, seed=7, stratum="identity_extreme")
    identity = pixel_identity_grid(2, 32, 48)

    assert torch.equal(pair["target_atlas_to_affine"], identity)
    assert torch.equal(pair["target_affine_to_atlas"], identity)
    assert (pair["fixed"] - pair["moving"]).abs().mean() > 0.05


def test_wrong_plane_is_labelled_for_rejection_and_has_no_warp_target():
    template, labels, mask = toy_plane()
    wrong_template = torch.flip(template, (-1,))
    wrong_labels = torch.flip(labels, (-1,))
    pair = make_synthetic_pair(
        template,
        labels,
        mask,
        seed=4,
        stratum="wrong_ap",
        wrong_template=wrong_template,
        wrong_labels=wrong_labels,
        wrong_mask=mask,
    )
    identity = pixel_identity_grid(2, 32, 48)

    assert pair["wrong_pair"].all()
    assert torch.equal(pair["target_atlas_to_affine"], identity)
    assert torch.equal(pair["target_velocity"], torch.zeros_like(identity))


def test_hierarchical_dice_rewards_the_known_deformation():
    template, labels, mask = toy_plane()
    pair = make_synthetic_pair(template, labels, mask, seed=27, stratum="smooth_deformation")
    identity = pixel_identity_grid(2, 32, 48)
    affine_loss = hierarchical_label_dice_loss(
        pair["fixed_labels"], pair["moving_labels"], identity, mask
    )
    known_loss = hierarchical_label_dice_loss(
        pair["fixed_labels"], pair["moving_labels"], pair["target_atlas_to_affine"], mask
    )

    assert known_loss < affine_loss


def test_combined_training_objective_has_model_gradients_and_ema_updates():
    template, labels, mask = toy_plane(batch=1)
    pair = make_synthetic_pair(template, labels, mask, seed=31, stratum="nuisance_damage")
    model = RegistrationWithRejector(base_channels=4)
    ema = EMA(model, decay=0.5)
    before = ema.model.rejector[-1].weight.clone()
    loss, terms = registration_objective(model, pair)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(terms) == {"mind", "flow", "dice", "inverse", "smooth", "topology", "rejection", "wrong_identity"}
    assert model.registration.velocity_head.weight.grad is not None
    assert model.rejector[-1].weight.grad is not None
    with torch.no_grad():
        model.rejector[-1].weight.add_(1.0)
    ema.update(model)
    assert not torch.equal(before, ema.model.rejector[-1].weight)


def passing_gate_metrics():
    return {"gates": {
        "folded_voxels": 0,
        "roundtrip_p95_px": 0.4,
        "residual_affine_max_px": 0.001,
        "landmark_tre_px": 0.8,
        "affine_landmark_tre_px": 2.5,
        "label_dice": 0.91,
        "affine_label_dice": 0.74,
        "identity_tre_p95_px": 0.3,
        "wrong_reject_rate": 0.98,
        "wrong_displacement_p95_px": 0.2,
        "valid_reject_rate": 0.02,
        "runtime_ms": 35.0,
        "runtime_limit_ms": 250.0,
    }}


def test_export_gates_cover_geometry_accuracy_rejection_and_runtime():
    assert VALIDATION_STRATA == (
        "identity_extreme", "smooth_deformation", "nuisance_damage", "wrong_ap"
    )
    metrics = passing_gate_metrics()
    assert export_gate_failures(metrics) == []
    metrics["gates"]["folded_voxels"] = 1
    metrics["gates"]["label_dice"] = 0.70
    metrics["gates"]["wrong_reject_rate"] = 0.50
    failures = export_gate_failures(metrics)
    assert any("fold" in failure for failure in failures)
    assert any("Dice" in failure for failure in failures)
    assert any("rejection" in failure for failure in failures)
