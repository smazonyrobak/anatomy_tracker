from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _sample_map(field_xy: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    height, width = field_xy.shape[:2]
    valid = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= width - 1.0)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= height - 1.0)
    )
    map_x = points[:, 0].reshape(-1, 1)
    map_y = points[:, 1].reshape(-1, 1)
    sampled = np.column_stack(
        [
            cv2.remap(
                field_xy[..., axis],
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).ravel()
            for axis in range(2)
        ]
    )
    sampled[~valid] = np.nan
    return sampled


@dataclass(frozen=True)
class NonlinearWarp2D:
    """Mutually inverse residual maps in the affine atlas canvas.

    ``atlas_to_affine_xy[y, x]`` is the affine-histology coordinate sampled to
    render atlas pixel ``(x, y)``. ``affine_to_atlas_xy`` maps points in the
    opposite direction.
    """

    atlas_to_affine_xy: np.ndarray
    affine_to_atlas_xy: np.ndarray

    def __post_init__(self) -> None:
        atlas_to_affine = np.ascontiguousarray(self.atlas_to_affine_xy, dtype=np.float32)
        affine_to_atlas = np.ascontiguousarray(self.affine_to_atlas_xy, dtype=np.float32)
        if atlas_to_affine.shape != affine_to_atlas.shape or atlas_to_affine.ndim != 3 or atlas_to_affine.shape[2] != 2:
            raise ValueError("Nonlinear maps must have matching H x W x 2 shapes")
        if not np.isfinite(atlas_to_affine).all() or not np.isfinite(affine_to_atlas).all():
            raise ValueError("Nonlinear maps contain non-finite coordinates")
        object.__setattr__(self, "atlas_to_affine_xy", atlas_to_affine)
        object.__setattr__(self, "affine_to_atlas_xy", affine_to_atlas)

    @property
    def shape(self) -> tuple[int, int]:
        return self.atlas_to_affine_xy.shape[:2]

    @classmethod
    def identity(cls, shape: tuple[int, int]) -> "NonlinearWarp2D":
        yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
        grid = np.stack((xx, yy), axis=-1)
        return cls(grid, grid.copy())

    def render_affine_image_in_atlas(self, image: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        height, width = self.shape
        return cv2.remap(
            image,
            self.atlas_to_affine_xy[..., 0],
            self.atlas_to_affine_xy[..., 1],
            interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def map_atlas_to_affine(self, points_xy: np.ndarray) -> np.ndarray:
        return _sample_map(self.atlas_to_affine_xy, points_xy)

    def map_affine_to_atlas(self, points_xy: np.ndarray) -> np.ndarray:
        return _sample_map(self.affine_to_atlas_xy, points_xy)

    def jacobian_determinant(self) -> np.ndarray:
        field = self.atlas_to_affine_xy
        dx_dy, dx_dx = np.gradient(field[..., 0])
        dy_dy, dy_dx = np.gradient(field[..., 1])
        return dx_dx * dy_dy - dx_dy * dy_dx

    def inverse_consistency_error(self) -> np.ndarray:
        height, width = self.shape
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        affine = self.map_atlas_to_affine(grid)
        atlas_round_trip = self.map_affine_to_atlas(affine)
        return np.linalg.norm(atlas_round_trip - grid, axis=1).reshape(height, width)


def compose_slice_to_atlas_points(
    display_points_xy: np.ndarray,
    slice_to_atlas_affine: np.ndarray,
    warp: NonlinearWarp2D | None,
) -> np.ndarray:
    points = np.asarray(display_points_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    affine_points = (np.asarray(slice_to_atlas_affine, dtype=np.float64) @ homogeneous.T).T[:, :2]
    return affine_points if warp is None else warp.map_affine_to_atlas(affine_points)


def compose_atlas_to_slice_points(
    atlas_points_xy: np.ndarray,
    atlas_to_slice_affine: np.ndarray,
    warp: NonlinearWarp2D | None,
) -> np.ndarray:
    points = np.asarray(atlas_points_xy, dtype=np.float64).reshape(-1, 2)
    affine_points = points if warp is None else warp.map_atlas_to_affine(points)
    homogeneous = np.column_stack((affine_points, np.ones(len(affine_points))))
    return (np.asarray(atlas_to_slice_affine, dtype=np.float64) @ homogeneous.T).T[:, :2]
