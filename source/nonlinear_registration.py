"""Affine display-slice to atlas-plane coordinate transform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np


FORMAT_VERSION = 3
COORDINATE_CONVENTION = "display_xy->affine_atlas_xy;pixel_centers"


def _points(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must have shape N x 2 in (x, y) order")
    return points


def _apply_homography(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = _points(points_xy)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    mapped = (homography @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


@dataclass(frozen=True)
class SliceAtlasTransform2D:
    """One invertible display-to-atlas homography on an atlas-plane canvas."""

    display_to_affine_atlas_h: np.ndarray
    display_shape: tuple[int, int]
    atlas_shape: tuple[int, int]

    coordinate_convention: ClassVar[str] = COORDINATE_CONVENTION

    def __post_init__(self) -> None:
        homography = np.ascontiguousarray(self.display_to_affine_atlas_h, dtype=np.float64)
        if homography.shape != (3, 3) or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) != 3:
            raise ValueError("Display-to-atlas homography must be finite, invertible, and 3 x 3")
        pivot = homography.flat[int(np.argmax(np.abs(homography)))]
        homography = homography / pivot
        display_shape = tuple(int(value) for value in self.display_shape)
        atlas_shape = tuple(int(value) for value in self.atlas_shape)
        if len(display_shape) != 2 or len(atlas_shape) != 2 or min(*display_shape, *atlas_shape) < 2:
            raise ValueError("Display and atlas shapes must be positive (height, width) pairs")
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
        return _apply_homography(display_points_xy, self.display_to_affine_atlas_h)

    def map_atlas_to_display(self, atlas_points_xy: np.ndarray) -> np.ndarray:
        return _apply_homography(atlas_points_xy, self.affine_atlas_to_display_h)

    def atlas_to_display_render_map(self) -> np.ndarray:
        yy, xx = np.mgrid[: self.atlas_shape[0], : self.atlas_shape[1]].astype(np.float32)
        atlas_points = np.column_stack((xx.ravel(), yy.ravel()))
        display_points = self.map_atlas_to_display(atlas_points)
        return np.ascontiguousarray(display_points.reshape(*self.atlas_shape, 2), dtype=np.float32)

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

    def check_invariants(self, *, maximum_homography_condition: float = 1e8) -> dict:
        condition = float(np.linalg.cond(self.display_to_affine_atlas_h))
        finite_fraction = float(np.isfinite(self.atlas_to_display_render_map()).all(axis=2).mean())
        if condition > maximum_homography_condition:
            raise ValueError("ill-conditioned display-to-atlas homography")
        if finite_fraction < 1.0:
            raise ValueError("non-finite atlas rendering coordinates")
        return {
            "coordinate_convention": self.coordinate_convention,
            "display_shape": self.display_shape,
            "atlas_shape": self.atlas_shape,
            "homography_condition_number": condition,
            "render_map_finite_fraction": finite_fraction,
        }

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            format_version=np.asarray(FORMAT_VERSION, dtype=np.uint16),
            coordinate_convention=np.asarray(self.coordinate_convention),
            display_shape=np.asarray(self.display_shape, dtype=np.int32),
            atlas_shape=np.asarray(self.atlas_shape, dtype=np.int32),
            display_to_affine_atlas_h=self.display_to_affine_atlas_h,
            nonlinear=np.asarray(False, dtype=np.uint8),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "SliceAtlasTransform2D":
        with np.load(Path(path), allow_pickle=False) as values:
            if int(values["format_version"]) != FORMAT_VERSION:
                raise ValueError("Unsupported SliceAtlasTransform2D format version")
            if str(values["coordinate_convention"]) not in {
                COORDINATE_CONVENTION,
                "display_xy->affine_atlas_xy->atlas_xy;pixel_centers",
            }:
                raise ValueError("Serialized coordinate convention does not match this implementation")
            transform = cls(
                values["display_to_affine_atlas_h"],
                tuple(values["display_shape"].tolist()),
                tuple(values["atlas_shape"].tolist()),
            )
        transform.check_invariants()
        return transform
