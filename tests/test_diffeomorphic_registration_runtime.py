import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))

from diffeomorphic_registration_runtime import (
    DiffeomorphicRegistrationRejected,
    MODEL_SHAPE,
    run_diffeomorphic_registration,
)


class ValueInfo:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(self, *, rejection_logit=-10.0, fold=False):
        self.rejection_logit = rejection_logit
        self.fold = fold
        self.feeds = None

    def get_inputs(self):
        return [ValueInfo(name, ["batch", 1, *MODEL_SHAPE]) for name in (
            "fixed", "moving", "fixed_mask", "moving_mask"
        )]

    def get_outputs(self):
        return [
            ValueInfo("atlas_to_affine", ["batch", 2, *MODEL_SHAPE]),
            ValueInfo("affine_to_atlas", ["batch", 2, *MODEL_SHAPE]),
            ValueInfo("velocity", ["batch", 2, *MODEL_SHAPE]),
            ValueInfo("rejection_logit", ["batch"]),
        ]

    def get_providers(self):
        return ["FakeExecutionProvider"]

    def run(self, output_names, feeds):
        assert output_names == ["atlas_to_affine", "affine_to_atlas", "velocity", "rejection_logit"]
        self.feeds = feeds
        yy, xx = np.mgrid[: MODEL_SHAPE[0], : MODEL_SHAPE[1]].astype(np.float32)
        identity = np.stack((xx, yy))[None]
        forward = identity.copy()
        if self.fold:
            forward[:, 0] = forward[:, 0, :, ::-1]
        return forward, identity.copy(), np.zeros_like(identity), np.asarray([self.rejection_logit], np.float32)


def inputs(shape):
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    image = xx + 2.0 * yy
    mask = np.ones(shape, dtype=bool)
    return image, image.copy(), mask, mask.copy()


def test_centered_padding_unpads_identity_maps_and_coordinates_exactly():
    shape = (42, 70)
    session = FakeSession()
    warp, diagnostics = run_diffeomorphic_registration(*inputs(shape), session=session)
    top = (MODEL_SHAPE[0] - shape[0]) // 2
    left = (MODEL_SHAPE[1] - shape[1]) // 2

    assert diagnostics["model_offset_yx"] == (top, left)
    assert session.feeds["fixed"].shape == (1, 1, *MODEL_SHAPE)
    assert session.feeds["fixed_mask"][0, 0, top, left] == 1.0
    assert session.feeds["fixed_mask"][0, 0, top - 1, left] == 0.0
    assert np.array_equal(warp.atlas_to_affine_xy, warp.affine_to_atlas_xy)
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    assert np.array_equal(warp.atlas_to_affine_xy, np.stack((xx, yy), axis=-1))
    assert diagnostics["inverse_p95_px"] == pytest.approx(0.0)
    assert diagnostics["displacement_max_px"] == pytest.approx(0.0)


def test_center_crop_restores_native_coordinate_offsets():
    shape = (MODEL_SHAPE[0] + 20, MODEL_SHAPE[1] + 16)
    fixed, moving, fixed_mask, moving_mask = inputs(shape)
    fixed_mask[:] = moving_mask[:] = False
    fixed_mask[20:-20, 20:-20] = True
    moving_mask[:] = fixed_mask
    warp, diagnostics = run_diffeomorphic_registration(
        fixed, moving, fixed_mask, moving_mask, session=FakeSession()
    )

    assert diagnostics["source_offset_yx"] == (10, 8)
    assert diagnostics["model_offset_yx"] == (0, 0)
    points = np.asarray([[8.0, 10.0], [120.0, 90.0], [shape[1] - 9.0, shape[0] - 11.0]])
    assert np.allclose(warp.map_atlas_to_affine(points), points)


def test_model_rejection_is_explicit_and_returns_no_warp():
    with pytest.raises(DiffeomorphicRegistrationRejected) as error:
        run_diffeomorphic_registration(*inputs((40, 64)), session=FakeSession(rejection_logit=10.0))

    assert "model rejection probability" in str(error.value)
    assert error.value.diagnostics["rejection_probability"] > 0.99


def test_folded_map_is_rejected_before_caller_can_install_it():
    with pytest.raises(DiffeomorphicRegistrationRejected) as error:
        run_diffeomorphic_registration(*inputs(MODEL_SHAPE), session=FakeSession(fold=True))

    assert "Jacobian failures" in str(error.value)
    assert error.value.diagnostics["fold_count"] > 0


def test_onnx_names_and_shapes_are_enforced():
    session = FakeSession()
    session.get_inputs = lambda: [ValueInfo("wrong", ["batch", 1, *MODEL_SHAPE])]
    with pytest.raises(RuntimeError, match="training contract"):
        run_diffeomorphic_registration(*inputs((40, 64)), session=session)
