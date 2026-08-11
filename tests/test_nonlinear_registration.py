import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))

from nonlinear_registration import NonlinearWarp2D, SliceAtlasTransform2D


def smooth_inverse_warp(shape=(80, 96), amplitude=2.0):
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    phase = np.sin(2.0 * np.pi * yy / (shape[0] - 1.0))
    forward = np.stack((xx + amplitude * phase, yy), axis=-1)
    inverse = np.stack((xx - amplitude * phase, yy), axis=-1)
    return NonlinearWarp2D(forward, inverse)


def test_identity_preserves_display_image_points_and_diagnostics():
    shape = (32, 48)
    transform = SliceAtlasTransform2D(np.eye(3), shape, shape, NonlinearWarp2D.identity(shape))
    image = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    points = np.asarray([[0.0, 0.0], [8.25, 11.5], [47.0, 31.0]])

    assert np.array_equal(transform.render_display_image_in_atlas(image, cv2.INTER_NEAREST), image)
    assert np.allclose(transform.map_display_to_atlas(points), points)
    assert np.allclose(transform.map_atlas_to_display(points), points)
    diagnostics = transform.check_invariants(maximum_inverse_p95_px=1e-6)
    assert diagnostics["coordinate_convention"].startswith("display_xy")
    assert diagnostics["minimum_jacobian"] == pytest.approx(1.0)
    assert diagnostics["fold_count"] == 0


def test_projective_homography_and_nonlinear_maps_are_mutual_inverses():
    shape = (80, 96)
    homography = np.asarray([[1.1, 0.02, 4.0], [-0.01, 1.08, 3.0], [0.0001, -0.0002, 1.0]])
    transform = SliceAtlasTransform2D(homography, shape, shape, smooth_inverse_warp(shape))
    display_points = np.asarray([[20.0, 15.0], [35.0, 30.0], [50.0, 45.0]])

    atlas_points = transform.map_display_to_atlas(display_points)
    recovered = transform.map_atlas_to_display(atlas_points)
    assert np.max(np.linalg.norm(recovered - display_points, axis=1)) < 0.04
    assert transform.diagnostics()["minimum_jacobian"] > 0.99


def test_single_pass_render_map_is_the_canonical_atlas_to_display_mapping():
    shape = (40, 56)
    homography = np.asarray([[0.95, 0.03, 2.0], [-0.02, 1.04, 1.0], [0.0, 0.0, 1.0]])
    transform = SliceAtlasTransform2D(homography, shape, shape, smooth_inverse_warp(shape, 0.8))
    render_map = transform.atlas_to_display_render_map()
    atlas_points = np.asarray([[8.0, 7.0], [27.0, 20.0], [45.0, 31.0]])
    sampled_map = np.asarray([render_map[int(y), int(x)] for x, y in atlas_points])

    assert render_map.shape == (*shape, 2)
    assert render_map.dtype == np.float32
    assert np.allclose(sampled_map, transform.map_atlas_to_display(atlas_points), atol=1e-5)


def test_raw_and_display_frames_cannot_be_silently_interchanged():
    transform = SliceAtlasTransform2D(np.eye(3), (24, 32), (24, 32))
    raw_image_with_different_shape = np.zeros((32, 24), dtype=np.uint8)
    with pytest.raises(ValueError, match="display coordinate frame"):
        transform.render_display_image_in_atlas(raw_image_with_different_shape)
    with pytest.raises(ValueError, match="N x 2"):
        transform.map_display_to_atlas(np.asarray([4.0, 5.0]))


def test_fold_is_reported_and_fails_invariants():
    shape = (24, 32)
    identity = NonlinearWarp2D.identity(shape)
    folded = identity.atlas_to_affine_xy.copy()
    folded[..., 0] = folded[:, ::-1, 0]
    transform = SliceAtlasTransform2D(
        np.eye(3), shape, shape, NonlinearWarp2D(folded, identity.affine_to_atlas_xy)
    )

    assert transform.diagnostics()["fold_count"] == (shape[0] * shape[1])
    with pytest.raises(ValueError, match="Jacobian"):
        transform.check_invariants()


@pytest.mark.parametrize("with_nonlinear", [False, True])
def test_npz_round_trip_is_exact_and_byte_deterministic(tmp_path, with_nonlinear):
    shape = (40, 56)
    homography = np.asarray([[1.02, -0.03, 5.0], [0.01, 0.98, 2.0], [0.0, 0.0, 1.0]]) * 7.0
    nonlinear = smooth_inverse_warp(shape, 0.7) if with_nonlinear else None
    transform = SliceAtlasTransform2D(homography, (44, 60), shape, nonlinear)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    transform.save_npz(first)
    transform.save_npz(second)
    restored = SliceAtlasTransform2D.load_npz(first)
    points = np.asarray([[10.0, 8.0], [21.5, 17.25], [35.0, 28.0]])

    assert first.read_bytes() == second.read_bytes()
    assert np.array_equal(restored.display_to_affine_atlas_h, transform.display_to_affine_atlas_h)
    assert restored.display_shape == transform.display_shape
    assert restored.atlas_shape == transform.atlas_shape
    assert np.allclose(restored.map_display_to_atlas(points), transform.map_display_to_atlas(points))
    assert (restored.nonlinear is None) == (transform.nonlinear is None)
