"""Canonical display-slice to 2-D atlas transform.

Raw histology coordinates are deliberately outside this module. The caller
first applies its raw-to-display rotation/flip matrix; every source point and
image accepted here is in the resulting display pixel coordinates ``(x, y)``.
Atlas and affine-atlas coordinates are likewise pixel-center ``(x, y)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np


FORMAT_VERSION = 1
COORDINATE_CONVENTION = "display_xy->affine_atlas_xy->atlas_xy;pixel_centers"


def _points(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must have shape N x 2 in (x, y) order")
    return points


def _apply_homography(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = _points(points_xy)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    mapped = (homography @ homogeneous.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return mapped[:, :2] / mapped[:, 2:3]


def _sample_map(field_xy: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = _points(points_xy).astype(np.float32)
    height, width = field_xy.shape[:2]
    valid = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= width - 1.0)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= height - 1.0)
    )
    sampled = np.column_stack(
        [
            cv2.remap(
                field_xy[..., axis],
                points[:, 0].reshape(-1, 1),
                points[:, 1].reshape(-1, 1),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).ravel()
            for axis in range(2)
        ]
    )
    sampled[~valid] = np.nan
    return sampled


def _identity_map(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    return np.stack((xx, yy), axis=-1)


@dataclass(frozen=True)
class NonlinearWarp2D:
    """Mutually inverse residual maps on the affine-atlas canvas.

    ``atlas_to_affine_xy[y, x]`` gives the affine-slice location sampled while
    rendering atlas pixel ``(x, y)``. ``affine_to_atlas_xy`` maps points in the
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
        atlas_to_affine.setflags(write=False)
        affine_to_atlas.setflags(write=False)
        object.__setattr__(self, "atlas_to_affine_xy", atlas_to_affine)
        object.__setattr__(self, "affine_to_atlas_xy", affine_to_atlas)

    @property
    def shape(self) -> tuple[int, int]:
        return self.atlas_to_affine_xy.shape[:2]

    @classmethod
    def identity(cls, shape: tuple[int, int]) -> "NonlinearWarp2D":
        grid = _identity_map(shape)
        return cls(grid, grid.copy())

    def map_atlas_to_affine(self, points_xy: np.ndarray) -> np.ndarray:
        return _sample_map(self.atlas_to_affine_xy, points_xy)

    def map_affine_to_atlas(self, points_xy: np.ndarray) -> np.ndarray:
        return _sample_map(self.affine_to_atlas_xy, points_xy)

    def jacobian_determinant(self) -> np.ndarray:
        dx_dy, dx_dx = np.gradient(self.atlas_to_affine_xy[..., 0])
        dy_dy, dy_dx = np.gradient(self.atlas_to_affine_xy[..., 1])
        return dx_dx * dy_dy - dx_dy * dy_dx

    def inverse_consistency_error(self) -> np.ndarray:
        height, width = self.shape
        grid = _identity_map(self.shape).reshape(-1, 2)
        round_trip = self.map_affine_to_atlas(self.map_atlas_to_affine(grid))
        return np.linalg.norm(round_trip - grid, axis=1).reshape(height, width)


@dataclass(frozen=True)
class SliceAtlasTransform2D:
    """One display-to-affine homography followed by an optional residual warp."""

    display_to_affine_atlas_h: np.ndarray
    display_shape: tuple[int, int]
    atlas_shape: tuple[int, int]
    nonlinear: NonlinearWarp2D | None = None

    coordinate_convention: ClassVar[str] = COORDINATE_CONVENTION

    def __post_init__(self) -> None:
        homography = np.ascontiguousarray(self.display_to_affine_atlas_h, dtype=np.float64)
        if homography.shape != (3, 3) or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) != 3:
            raise ValueError("Display-to-affine-atlas homography must be finite, invertible, and 3 x 3")
        pivot = homography.flat[int(np.argmax(np.abs(homography)))]
        homography = homography / pivot
        display_shape = tuple(int(value) for value in self.display_shape)
        atlas_shape = tuple(int(value) for value in self.atlas_shape)
        if len(display_shape) != 2 or len(atlas_shape) != 2 or min(*display_shape, *atlas_shape) < 2:
            raise ValueError("Display and atlas shapes must be positive (height, width) pairs")
        if self.nonlinear is not None and self.nonlinear.shape != atlas_shape:
            raise ValueError("Nonlinear map shape must equal the atlas canvas shape")
        homography.setflags(write=False)
        object.__setattr__(self, "display_to_affine_atlas_h", homography)
        object.__setattr__(self, "display_shape", display_shape)
        object.__setattr__(self, "atlas_shape", atlas_shape)

    @property
    def affine_atlas_to_display_h(self) -> np.ndarray:
        inverse = np.linalg.inv(self.display_to_affine_atlas_h)
        inverse.setflags(write=False)
        return inverse

    def map_display_to_atlas(self, display_points_xy: np.ndarray) -> np.ndarray:
        affine_points = _apply_homography(display_points_xy, self.display_to_affine_atlas_h)
        return affine_points if self.nonlinear is None else self.nonlinear.map_affine_to_atlas(affine_points)

    def map_atlas_to_display(self, atlas_points_xy: np.ndarray) -> np.ndarray:
        atlas_points = _points(atlas_points_xy)
        affine_points = atlas_points if self.nonlinear is None else self.nonlinear.map_atlas_to_affine(atlas_points)
        return _apply_homography(affine_points, self.affine_atlas_to_display_h)

    def atlas_to_display_render_map(self) -> np.ndarray:
        affine_map = _identity_map(self.atlas_shape) if self.nonlinear is None else self.nonlinear.atlas_to_affine_xy
        display_map = _apply_homography(affine_map.reshape(-1, 2), self.affine_atlas_to_display_h)
        return np.ascontiguousarray(display_map.reshape(*self.atlas_shape, 2), dtype=np.float32)

    def render_display_image_in_atlas(
        self,
        display_image: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        if display_image.shape[:2] != self.display_shape:
            raise ValueError("Image must already be in the declared display coordinate frame")
        render_map = self.atlas_to_display_render_map()
        return cv2.remap(
            display_image,
            render_map[..., 0],
            render_map[..., 1],
            interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def diagnostics(self) -> dict[str, float | int | bool | tuple[int, int] | str]:
        condition = float(np.linalg.cond(self.display_to_affine_atlas_h))
        render_finite_fraction = float(np.isfinite(self.atlas_to_display_render_map()).all(axis=2).mean())
        if self.nonlinear is None:
            minimum_jacobian = 1.0
            fold_count = 0
            inverse_finite_fraction = 1.0
            inverse_p95 = 0.0
            inverse_max = 0.0
        else:
            jacobian = self.nonlinear.jacobian_determinant()
            inverse_error = self.nonlinear.inverse_consistency_error()
            finite = np.isfinite(inverse_error)
            finite_error = inverse_error[finite]
            minimum_jacobian = float(jacobian.min())
            fold_count = int((jacobian <= 0.0).sum())
            inverse_finite_fraction = float(finite.mean())
            inverse_p95 = float(np.percentile(finite_error, 95)) if len(finite_error) else float("inf")
            inverse_max = float(finite_error.max()) if len(finite_error) else float("inf")
        return {
            "coordinate_convention": self.coordinate_convention,
            "display_shape": self.display_shape,
            "atlas_shape": self.atlas_shape,
            "nonlinear": self.nonlinear is not None,
            "homography_condition_number": condition,
            "render_map_finite_fraction": render_finite_fraction,
            "minimum_jacobian": minimum_jacobian,
            "fold_count": fold_count,
            "inverse_finite_fraction": inverse_finite_fraction,
            "inverse_p95_px": inverse_p95,
            "inverse_max_px": inverse_max,
        }

    def check_invariants(
        self,
        *,
        minimum_jacobian: float = 0.0,
        maximum_inverse_p95_px: float = 1.0,
        maximum_homography_condition: float = 1e8,
    ) -> dict[str, float | int | bool | tuple[int, int] | str]:
        diagnostics = self.diagnostics()
        failures = []
        if diagnostics["homography_condition_number"] > maximum_homography_condition:
            failures.append("ill-conditioned display-to-atlas homography")
        if diagnostics["render_map_finite_fraction"] < 1.0:
            failures.append("non-finite atlas rendering coordinates")
        if diagnostics["minimum_jacobian"] <= minimum_jacobian:
            failures.append("non-positive nonlinear Jacobian")
        if diagnostics["inverse_p95_px"] > maximum_inverse_p95_px:
            failures.append("nonlinear inverse-consistency p95 exceeds limit")
        if failures:
            raise ValueError("; ".join(failures))
        return diagnostics

    def save_npz(self, path: str | Path) -> None:
        empty_map = np.empty((0, 0, 2), dtype=np.float32)
        atlas_to_affine = empty_map if self.nonlinear is None else self.nonlinear.atlas_to_affine_xy
        affine_to_atlas = empty_map if self.nonlinear is None else self.nonlinear.affine_to_atlas_xy
        np.savez_compressed(
            Path(path),
            format_version=np.asarray(FORMAT_VERSION, dtype=np.uint16),
            coordinate_convention=np.asarray(self.coordinate_convention),
            display_shape=np.asarray(self.display_shape, dtype=np.int32),
            atlas_shape=np.asarray(self.atlas_shape, dtype=np.int32),
            display_to_affine_atlas_h=self.display_to_affine_atlas_h,
            nonlinear=np.asarray(self.nonlinear is not None, dtype=np.uint8),
            atlas_to_affine_xy=atlas_to_affine,
            affine_to_atlas_xy=affine_to_atlas,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "SliceAtlasTransform2D":
        with np.load(Path(path), allow_pickle=False) as values:
            if int(values["format_version"]) != FORMAT_VERSION:
                raise ValueError("Unsupported SliceAtlasTransform2D format version")
            if str(values["coordinate_convention"]) != COORDINATE_CONVENTION:
                raise ValueError("Serialized coordinate convention does not match this implementation")
            nonlinear = None
            if bool(values["nonlinear"]):
                nonlinear = NonlinearWarp2D(values["atlas_to_affine_xy"], values["affine_to_atlas_xy"])
            return cls(
                values["display_to_affine_atlas_h"],
                tuple(values["display_shape"].tolist()),
                tuple(values["atlas_shape"].tolist()),
                nonlinear,
            )
