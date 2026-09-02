import copy

import numpy as np
import pytest
import torch

import training.arbitrary_plane_deformation_gauge_v4 as gauge
from training.arbitrary_plane_deformation_primitives import (
    scaling_and_squaring_yx,
    support_weighted_affine_projection_yx,
)
from training.arbitrary_plane_synthetic_ops import (
    FIXED_SEVEN_DECODER_INTEGRATION,
    fixed_source_maps,
    integrate_stationary_velocity_fixed_decoder_steps,
    remove_uniform_canvas_affine_component_decoder_parity,
)


def _direct_target():
    rng = np.random.default_rng(20260902)
    velocity_xy = remove_uniform_canvas_affine_component_decoder_parity(
        rng.normal(0.0, 0.3, (2, 29, 37)).astype(np.float32)
    )
    _, source_to_fixed_xy, steps = (
        integrate_stationary_velocity_fixed_decoder_steps(velocity_xy)
    )
    target_yx = np.moveaxis((-velocity_xy)[::-1], 0, -1)
    pullback_yx = np.moveaxis(source_to_fixed_xy[::-1], 0, -1)
    artifact = gauge.certify_direct_deformation_target_v4(
        target_yx,
        pullback_yx,
        np.arange(9, dtype=np.float64).reshape(3, 3),
        np.ones((29, 37), dtype=bool),
    )
    return velocity_xy, source_to_fixed_xy, steps, artifact


def test_fixed_float32_integrator_and_projection_are_decoder_identical():
    velocity_xy, source_to_fixed_xy, steps, artifact = _direct_target()
    target = torch.from_numpy(
        np.ascontiguousarray((-velocity_xy)[::-1])
    ).unsqueeze(0)
    expected = scaling_and_squaring_yx(target, 7)[0].numpy()[::-1]
    assert steps == 7
    assert np.array_equal(source_to_fixed_xy, expected)

    residual, removed, post, _ = support_weighted_affine_projection_yx(
        target, torch.ones((1, 1, 29, 37), dtype=torch.float32)
    )
    arrays = artifact["arrays"]
    assert np.array_equal(
        arrays["certified_pullback_map_yx_px_float32"],
        arrays["source_to_fixed_pullback_map_yx_px_float32"],
    )
    assert np.array_equal(
        arrays["projected_stationary_velocity_yx_px_float32"],
        np.moveaxis(residual[0].numpy(), 0, -1),
    )
    assert np.array_equal(
        arrays["uniform_canvas_affine_coefficients_yx_float32"],
        removed[0].numpy(),
    )
    assert np.array_equal(
        arrays["postprojection_affine_coefficients_yx_float32"],
        post[0].numpy(),
    )
    assert artifact["diagnostics"]["valid_certification_error_max_px"] == 0.0
    assert artifact["diagnostics"]["parent_pose_adjustment_max_abs"] == 0.0


def test_identity_similarity_source_to_fixed_is_exact_decoder_inverse_map():
    rng = np.random.default_rng(29)
    velocity_xy = remove_uniform_canvas_affine_component_decoder_parity(
        rng.normal(0.0, 0.2, (2, 23, 31)).astype(np.float32)
    )
    maps = fixed_source_maps(
        velocity_xy,
        np.eye(2),
        angle_rad=0.0,
        scale=1.0,
        translation_xy=(0.0, 0.0),
        integration_contract=FIXED_SEVEN_DECODER_INTEGRATION,
    )
    assert maps["integration_steps"] == 7
    assert np.array_equal(
        maps["source_to_fixed_map"], maps["local_fixed_inverse_map"]
    )
    assert (
        maps["identity_similarity_inverse_composition_error_max_abs_px"]
        == 0.0
    )


def test_direct_target_replay_and_tamper_rejection():
    _, _, _, artifact = _direct_target()
    replay = gauge.replay_direct_deformation_target_v4(artifact)
    assert gauge.direct_deformation_target_receipt_v4(replay) == (
        gauge.direct_deformation_target_receipt_v4(artifact)
    )
    assert gauge.verify_direct_deformation_target_v4(artifact)

    changed = copy.deepcopy(artifact)
    changed["arrays"]["target_pullback_stationary_velocity_yx_px_float32"][
        0, 0, 0
    ] += np.float32(0.25)
    with pytest.raises(ValueError, match="receipt or source binding changed"):
        gauge.verify_direct_deformation_target_v4(changed)

    changed_runtime = copy.deepcopy(artifact)
    changed_runtime["runtime_versions"]["torch"] = "tampered"
    with pytest.raises(ValueError, match="receipt or source binding changed"):
        gauge.verify_direct_deformation_target_v4(changed_runtime)


@pytest.mark.parametrize(
    "valid",
    (
        np.zeros((9, 11), dtype=bool),
        np.full((9, 11), np.nan, dtype=np.float32),
    ),
)
def test_empty_or_nonfinite_valid_mask_is_rejected(valid):
    velocity = np.zeros((9, 11, 2), dtype=np.float32)
    y, x = np.indices((9, 11), dtype=np.float32)
    identity = np.stack((y, x), axis=-1)
    with pytest.raises(ValueError, match="validity mask"):
        gauge.certify_direct_deformation_target_v4(
            velocity, identity, np.eye(3), valid
        )
