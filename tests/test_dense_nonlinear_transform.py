import numpy as np
import pytest

from source.nonlinear_registration import SliceAtlasTransform2D


def identity_map(shape):
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    return np.stack((xx, yy), axis=-1)


def test_dense_transform_uses_one_map_for_rendering_and_points(tmp_path):
    shape = (32, 48)
    atlas_to_affine = identity_map(shape)
    atlas_to_affine[..., 0] += 2.0
    atlas_to_affine[..., 1] += 1.0
    affine_to_atlas = identity_map(shape)
    affine_to_atlas[..., 0] -= 2.0
    affine_to_atlas[..., 1] -= 1.0
    transform = SliceAtlasTransform2D(
        np.eye(3),
        shape,
        shape,
        atlas_to_affine,
        affine_to_atlas,
        np.ones(shape, dtype=bool),
        '{"model_sha256":"test"}',
    )

    atlas_points = np.asarray([[10.0, 10.0], [20.0, 15.0]])
    display_points = transform.map_atlas_to_display(atlas_points)
    assert np.allclose(display_points, atlas_points + (2.0, 1.0))
    assert np.allclose(transform.map_display_to_atlas(display_points), atlas_points)

    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    rendered = transform.render_display_image_in_atlas(image)
    assert rendered[10, 10] == image[11, 12]

    path = tmp_path / "dense_transform.npz"
    transform.save_npz(path)
    restored = SliceAtlasTransform2D.load_npz(path)
    assert restored.nonlinear
    assert np.array_equal(restored.atlas_to_affine_xy, atlas_to_affine)
    assert np.array_equal(restored.affine_to_atlas_xy, affine_to_atlas)
    assert restored.registration_metadata_json == '{"model_sha256":"test"}'


def test_affine_transform_v4_round_trip(tmp_path):
    matrix = np.asarray([[1.2, 0.0, 3.0], [0.0, 0.9, -2.0], [0.0, 0.0, 1.0]])
    transform = SliceAtlasTransform2D(matrix, (20, 30), (24, 36))
    points = np.asarray([[4.0, 5.0], [12.0, 8.0]])
    assert np.allclose(transform.map_atlas_to_display(transform.map_display_to_atlas(points)), points)

    path = tmp_path / "affine_transform.npz"
    transform.save_npz(path)
    restored = SliceAtlasTransform2D.load_npz(path)
    assert not restored.nonlinear
    assert np.allclose(restored.display_to_affine_atlas_h, transform.display_to_affine_atlas_h)


def test_v3_experimental_dense_fields_remain_affine_only(tmp_path):
    shape = (16, 24)
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        format_version=np.asarray(3, dtype=np.uint16),
        coordinate_convention=np.asarray("display_xy->affine_atlas_xy->atlas_xy;pixel_centers"),
        display_shape=np.asarray(shape, dtype=np.int32),
        atlas_shape=np.asarray(shape, dtype=np.int32),
        display_to_affine_atlas_h=np.eye(3),
        nonlinear=np.asarray(True, dtype=np.uint8),
        atlas_to_affine_xy=np.zeros((*shape, 2), dtype=np.float32),
        affine_to_atlas_xy=np.zeros((*shape, 2), dtype=np.float32),
    )

    restored = SliceAtlasTransform2D.load_npz(path)

    assert not restored.nonlinear
    assert np.allclose(restored.map_display_to_atlas(np.asarray([[3.0, 4.0]])), [[3.0, 4.0]])


def test_folded_dense_transform_cannot_be_validated_or_saved(tmp_path):
    shape = (12, 16)
    folded = np.zeros((*shape, 2), dtype=np.float32)
    transform = SliceAtlasTransform2D(np.eye(3), shape, shape, folded, folded)
    with pytest.raises(ValueError, match="folded"):
        transform.check_invariants()
    with pytest.raises(ValueError, match="folded"):
        transform.save_npz(tmp_path / "invalid.npz")


def test_dense_point_mapping_rejects_locations_outside_the_valid_tissue_mask():
    shape = (20, 24)
    valid = np.zeros(shape, dtype=bool)
    valid[4:16, 5:19] = True
    transform = SliceAtlasTransform2D(
        np.eye(3), shape, shape, identity_map(shape), identity_map(shape), valid
    )

    atlas_points = transform.map_display_to_atlas(
        np.asarray([[10.0, 10.0], [2.0, 2.0]])
    )
    display_points = transform.map_atlas_to_display(
        np.asarray([[10.0, 10.0], [2.0, 2.0]])
    )

    assert np.allclose(atlas_points[0], [10.0, 10.0])
    assert np.allclose(display_points[0], [10.0, 10.0])
    assert np.isnan(atlas_points[1]).all()
    assert np.isnan(display_points[1]).all()


def test_near_singular_dense_transform_is_rejected_even_without_a_fold():
    shape = (12, 16)
    compressed = identity_map(shape) * 0.05
    transform = SliceAtlasTransform2D(
        np.eye(3), shape, shape, compressed, compressed
    )

    with pytest.raises(ValueError, match="near-singular"):
        transform.check_invariants()
