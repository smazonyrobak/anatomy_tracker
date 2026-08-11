import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))

from nonlinear_registration import (
    NonlinearWarp2D,
    compose_atlas_to_slice_points,
    compose_slice_to_atlas_points,
)


def smooth_inverse_warp(shape=(80, 96), amplitude=2.0):
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    phase = np.sin(2.0 * np.pi * yy / (shape[0] - 1.0))
    forward = np.stack((xx + amplitude * phase, yy), axis=-1)
    inverse = np.stack((xx - amplitude * phase, yy), axis=-1)
    return NonlinearWarp2D(forward, inverse)


def test_identity_preserves_image_points_and_jacobian():
    warp = NonlinearWarp2D.identity((32, 48))
    image = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
    points = np.asarray([[0.0, 0.0], [8.25, 11.5], [47.0, 31.0]])
    assert np.array_equal(warp.render_affine_image_in_atlas(image, cv2.INTER_NEAREST), image)
    assert np.allclose(warp.map_atlas_to_affine(points), points)
    assert np.allclose(warp.map_affine_to_atlas(points), points)
    assert np.allclose(warp.jacobian_determinant(), 1.0)
    assert np.nanmax(warp.inverse_consistency_error()) < 1e-6


def test_smooth_inverse_round_trip_and_positive_jacobian():
    warp = smooth_inverse_warp()
    points = np.asarray([[12.5, 18.0], [40.0, 40.0], [70.25, 62.5]])
    round_trip = warp.map_affine_to_atlas(warp.map_atlas_to_affine(points))
    assert np.max(np.linalg.norm(round_trip - points, axis=1)) < 0.03
    assert warp.jacobian_determinant().min() > 0.99


def test_affine_and_nonlinear_point_composition_are_mutual_inverses():
    warp = smooth_inverse_warp()
    affine = np.asarray([[1.1, 0.0, 4.0], [0.0, 1.1, 3.0], [0.0, 0.0, 1.0]])
    inverse = np.linalg.inv(affine)
    slice_points = np.asarray([[20.0, 15.0], [35.0, 30.0], [50.0, 45.0]])
    atlas_points = compose_slice_to_atlas_points(slice_points, affine, warp)
    recovered = compose_atlas_to_slice_points(atlas_points, inverse, warp)
    assert np.max(np.linalg.norm(recovered - slice_points, axis=1)) < 0.03


def test_fold_has_nonpositive_jacobian():
    warp = NonlinearWarp2D.identity((24, 32))
    folded = warp.atlas_to_affine_xy.copy()
    folded[..., 0] = folded[:, ::-1, 0]
    invalid = NonlinearWarp2D(folded, warp.affine_to_atlas_xy)
    assert invalid.jacobian_determinant().max() < 0.0
