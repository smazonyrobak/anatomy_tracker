import pytest
import torch

from training.diffeomorphic_registration_model import (
    jacobian_determinant,
    pixel_identity_grid,
    remove_global_affine,
)
from training.train_diffeomorphic_registration import (
    LOCKED_SEED_BASE,
    LOCKED_STRATA,
    RegistrationWithRejector,
    SELECTION_SEED_BASE,
    SELECTION_STRATA,
    EMA,
    checkpoint_selection_key,
    export_gate_failures,
    export_candidate_model,
    hierarchical_label_dice_loss,
    make_synthetic_pair,
    registration_objective,
    OnnxRegistrationModel,
    onnx_parity_report,
    surface_affine_calibrate,
)
from training.real_histology_registration import native_registration_batch


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
    assert set(torch.unique(pair["moving_mask"]).tolist()) <= {0.0, 1.0}


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
        stratum="wrong_ap_near",
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


def test_label_free_objective_has_finite_loss_and_gradients():
    template, _, mask = toy_plane(batch=1)
    labels = torch.zeros_like(template, dtype=torch.long)
    pair = make_synthetic_pair(
        template, labels, mask, seed=29, stratum="smooth_deformation_label_free"
    )
    model = RegistrationWithRejector(base_channels=4)

    loss, terms = registration_objective(model, pair)
    loss.backward()

    assert torch.isfinite(loss)
    assert terms["dice"] == 0.0
    assert model.registration.velocity_head.weight.grad is not None
    assert torch.isfinite(model.registration.velocity_head.weight.grad).all()


def test_combined_training_objective_has_model_gradients_and_ema_updates():
    template, labels, mask = toy_plane(batch=1)
    pair = make_synthetic_pair(template, labels, mask, seed=31, stratum="nuisance_damage")
    model = RegistrationWithRejector(base_channels=4)
    ema = EMA(model, decay=0.5)
    before = ema.model.rejector[-1].weight.clone()
    loss, terms = registration_objective(model, pair)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(terms) == {
        "mind", "flow", "dice", "inverse", "smooth", "topology", "affine", "support",
        "rejection", "wrong_identity",
    }
    assert model.registration.velocity_head.weight.grad is not None
    assert model.rejector[-1].weight.grad is not None
    with torch.no_grad():
        model.rejector[-1].weight.add_(1.0)
    ema.update(model)
    assert not torch.equal(before, ema.model.rejector[-1].weight)


def test_native_positive_has_finite_unsupervised_objective_without_fake_targets():
    template, _, mask = toy_plane(batch=1)
    section = {
        "fixed": template[0, 0].numpy(),
        "moving": (template[0, 0].square() * mask[0, 0]).numpy(),
        "fixed_mask": mask[0, 0].numpy().astype(bool),
        "moving_mask": mask[0, 0].numpy().astype(bool),
    }
    batch = native_registration_batch([section], torch.device("cpu"))
    model = RegistrationWithRejector(base_channels=4)

    loss, terms = registration_objective(model, batch)
    loss.backward()

    assert batch["similarity_supervision"].all()
    assert not batch["dense_supervision"].any() and not batch["label_supervision"].any()
    assert "target_atlas_to_affine" not in batch and "fixed_labels" not in batch
    assert torch.isfinite(loss) and torch.isfinite(terms["mind"])
    assert terms["flow"] == 0.0 and terms["dice"] == 0.0
    assert model.registration.velocity_head.weight.grad is not None
    assert torch.isfinite(model.registration.velocity_head.weight.grad).all()


def test_native_wrong_plane_only_supervises_rejection_and_identity():
    template, _, mask = toy_plane(batch=1)
    target = {
        "fixed": template[0, 0].numpy(),
        "moving": template[0, 0].numpy(),
        "fixed_mask": mask[0, 0].numpy().astype(bool),
        "moving_mask": mask[0, 0].numpy().astype(bool),
    }
    wrong_mask = torch.zeros_like(mask)
    wrong_mask[:, :, 7:27, 14:36] = 1.0
    wrong = {
        "fixed": torch.flip(template[0, 0], (-1,)).numpy(),
        "fixed_mask": wrong_mask[0, 0].numpy().astype(bool),
    }
    batch = native_registration_batch(
        [target], torch.device("cpu"), wrong_sections=[wrong], wrong_kind="wrong_tilt"
    )
    model = RegistrationWithRejector(base_channels=4)
    loss, terms = registration_objective(model, batch)
    loss.backward()

    assert batch["wrong_pair"].all()
    assert not batch["similarity_supervision"].any()
    assert not batch["dense_supervision"].any()
    assert not batch["geometry_supervision"].any()
    assert terms["mind"] == terms["flow"] == terms["dice"] == terms["support"] == 0.0
    assert torch.isfinite(loss)
    assert model.rejector[-1].weight.grad is not None


def passing_gate_metrics():
    return {"gates": {
        "folded_voxels": 0,
        "minimum_jacobian": 0.8,
        "maximum_abs_log_jacobian_p99": 0.2,
        "maximum_abs_log_jacobian": 0.3,
        "roundtrip_p95_px": 0.4,
        "roundtrip_max_px": 0.8,
        "inverse_finite_fraction": 1.0,
        "map_finite_fraction": 1.0,
        "residual_affine_max_px": 0.001,
        "outside_tissue_displacement_max_px": 0.0,
        "displacement_p95_px": 4.0,
        "displacement_max_px": 6.0,
        "landmark_tre_px": 0.8,
        "affine_landmark_tre_px": 2.5,
        "landmark_tre_p95_px": 1.2,
        "affine_landmark_tre_p95_px": 3.5,
        "tre_improvement_px": 1.7,
        "tre_relative_improvement": 0.68,
        "tre_p95_improvement_px": 2.3,
        "tre_p95_relative_improvement": 0.66,
        "label_dice": 0.91,
        "affine_label_dice": 0.74,
        "label_dice_improvement": 0.17,
        "identity_tre_p95_px": 0.3,
        "wrong_reject_rate": 0.98,
        "wrong_displacement_p95_px": 0.2,
        "valid_reject_rate": 0.02,
        "retained_overlap": 0.8,
        "runtime_ms": 35.0,
        "runtime_limit_ms": 250.0,
    }}


def test_checkpoint_selection_prefers_gate_feasibility_then_normalized_violation():
    feasible = passing_gate_metrics()
    slightly_invalid = {"gates": dict(feasible["gates"], landmark_tre_px=1.1)}
    severely_invalid = {"gates": dict(feasible["gates"], landmark_tre_px=2.0)}

    assert checkpoint_selection_key(feasible, None, 100.0) < checkpoint_selection_key(
        slightly_invalid, None, 0.0
    )
    assert checkpoint_selection_key(slightly_invalid, None, 100.0) < checkpoint_selection_key(
        severely_invalid, None, 0.0
    )


def test_export_gates_cover_geometry_accuracy_rejection_and_runtime():
    assert SELECTION_STRATA == (
        "identity_extreme", "smooth_deformation", "nuisance_damage",
        "wrong_ap_near", "wrong_ap_far", "wrong_tilt",
    )
    assert LOCKED_STRATA == tuple(f"{name}_label_free" for name in SELECTION_STRATA)
    assert SELECTION_SEED_BASE != LOCKED_SEED_BASE
    metrics = passing_gate_metrics()
    assert export_gate_failures(metrics) == []
    metrics["gates"]["folded_voxels"] = 1
    metrics["gates"]["label_dice"] = 0.70
    metrics["gates"]["wrong_reject_rate"] = 0.50
    failures = export_gate_failures(metrics)
    assert any("fold" in failure for failure in failures)
    assert any("Dice" in failure for failure in failures)
    assert any("rejection" in failure for failure in failures)


def test_export_gates_require_absolute_accuracy_and_material_improvement():
    metrics = passing_gate_metrics()
    metrics["gates"].update({
        "landmark_tre_px": 1.1,
        "landmark_tre_p95_px": 2.1,
        "affine_landmark_tre_px": 1.2,
        "affine_landmark_tre_p95_px": 2.2,
        "tre_improvement_px": 0.1,
        "tre_relative_improvement": 0.08,
        "tre_p95_improvement_px": 0.1,
        "tre_p95_relative_improvement": 0.05,
        "label_dice": 0.84,
        "affine_label_dice": 0.82,
        "label_dice_improvement": 0.02,
    })
    failures = export_gate_failures(metrics)
    assert any("median TRE exceeds" in failure for failure in failures)
    assert any("p95 exceeds" in failure for failure in failures)
    assert any("improvement" in failure for failure in failures)
    assert any("Dice is below" in failure for failure in failures)


def test_prediction_cannot_hide_flow_error_by_leaving_the_target_domain():
    template, labels, mask = toy_plane(batch=1)
    pair = make_synthetic_pair(template, labels, mask, seed=101, stratum="smooth_deformation")

    class EscapingModel(torch.nn.Module):
        def forward(self, fixed, moving, fixed_mask, moving_mask):
            identity = pixel_identity_grid(1, fixed.shape[-2], fixed.shape[-1])
            velocity = torch.zeros_like(identity)
            return identity + 100.0, identity - 100.0, velocity, torch.zeros(1)

    _, terms = registration_objective(EscapingModel(), pair)
    assert terms["flow"] > 50.0


def test_label_free_locked_pair_does_not_leak_segmentation_palette():
    template, labels, mask = toy_plane(batch=1)
    changed_labels = torch.flip(labels, (-1,))
    original = make_synthetic_pair(template, labels, mask, seed=44, stratum="identity_extreme_label_free")
    changed = make_synthetic_pair(template, changed_labels, mask, seed=44, stratum="identity_extreme_label_free")

    assert torch.equal(original["fixed"], changed["fixed"])
    assert torch.equal(original["moving"], changed["moving"])


def test_wrong_plane_surface_calibration_removes_outline_centroid_and_scale_shortcut():
    template, labels, target_mask = toy_plane(batch=1)
    source_mask = torch.zeros_like(target_mask)
    source_mask[:, :, 9:25, 12:34] = 1.0
    calibrated_template, _, calibrated_mask = surface_affine_calibrate(
        template, labels, source_mask, target_mask
    )

    def moments(value):
        y, x = torch.where(value[0, 0] > 0.5)
        return torch.stack((x.float().mean(), y.float().mean(), x.float().std(), y.float().std()))

    assert calibrated_template.shape == template.shape
    assert torch.allclose(moments(calibrated_mask), moments(target_mask), atol=1.0)


def test_exported_onnx_matches_pytorch_maps_velocity_and_rejection(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = RegistrationWithRejector(base_channels=4).eval()
    destination = tmp_path / "candidate.onnx"
    exported, failures = export_candidate_model(model, passing_gate_metrics(), destination)
    assert exported and failures == []

    fixed = torch.rand(1, 1, 320, 464)
    moving = torch.rand_like(fixed)
    mask = torch.ones_like(fixed)
    batch = {"fixed": fixed, "moving": moving, "fixed_mask": mask, "moving_mask": mask}
    parity = onnx_parity_report(model, OnnxRegistrationModel(destination), batch)

    assert parity["passed"]
