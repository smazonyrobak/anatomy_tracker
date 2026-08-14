"""Canonical display-slice to atlas-plane coordinate transform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np


FORMAT_VERSION = 4
LEGACY_FORMAT_VERSION = 3
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
        tuple(
            cv2.remap(
                field_xy[..., axis],
                points[:, 0].reshape(-1, 1),
                points[:, 1].reshape(-1, 1),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).ravel()
            for axis in range(2)
        )
    )
    sampled[~valid] = np.nan
    return sampled


def _identity_map(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    return np.stack((xx, yy), axis=-1)


def _points_inside_mask(mask: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = _points(points_xy).astype(np.float32)
    height, width = mask.shape
    finite = np.isfinite(points).all(axis=1)
    safe = np.where(finite[:, None], points, 0.0)
    inside = (
        finite
        & (safe[:, 0] >= 0.0)
        & (safe[:, 0] <= width - 1.0)
        & (safe[:, 1] >= 0.0)
        & (safe[:, 1] <= height - 1.0)
    )
    sampled = cv2.remap(
        mask.astype(np.uint8),
        safe[:, 0].reshape(-1, 1),
        safe[:, 1].reshape(-1, 1),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).ravel().astype(bool)
    return inside & sampled


def _cycle_error(first_map: np.ndarray, second_map: np.ndarray) -> np.ndarray:
    height, width = first_map.shape[:2]
    valid = (
        (first_map[..., 0] >= 0.0)
        & (first_map[..., 0] <= width - 1.0)
        & (first_map[..., 1] >= 0.0)
        & (first_map[..., 1] <= height - 1.0)
    )
    round_trip = np.stack(
        tuple(
            cv2.remap(
                second_map[..., axis],
                first_map[..., 0],
                first_map[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            for axis in range(2)
        ),
        axis=-1,
    )
    error = np.linalg.norm(round_trip - _identity_map((height, width)), axis=2)
    error[~valid] = np.nan
    return error


def _jacobian_determinant(pixel_map: np.ndarray) -> np.ndarray:
    d_dx = pixel_map[:-1, 1:] - pixel_map[:-1, :-1]
    d_dy = pixel_map[1:, :-1] - pixel_map[:-1, :-1]
    return d_dx[..., 0] * d_dy[..., 1] - d_dx[..., 1] * d_dy[..., 0]


@dataclass(frozen=True)
class SliceAtlasTransform2D:
    """Display-to-atlas transform with an optional residual dense warp.

    ``atlas_to_affine_xy[y, x]`` is the affine-atlas pixel sampled to render
    atlas output pixel ``(x, y)``. ``affine_to_atlas_xy`` maps points in the
    opposite direction. Both maps use absolute ``(x, y)`` pixel coordinates.
    """

    display_to_affine_atlas_h: np.ndarray
    display_shape: tuple[int, int]
    atlas_shape: tuple[int, int]
    atlas_to_affine_xy: np.ndarray | None = None
    affine_to_atlas_xy: np.ndarray | None = None
    valid_atlas_mask: np.ndarray | None = None
    registration_metadata_json: str | None = None

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

        forward = self.atlas_to_affine_xy
        inverse = self.affine_to_atlas_xy
        if (forward is None) != (inverse is None):
            raise ValueError("Dense registration requires both forward and inverse maps")
        if forward is not None:
            forward = np.ascontiguousarray(forward, dtype=np.float32)
            inverse = np.ascontiguousarray(inverse, dtype=np.float32)
            expected = (*atlas_shape, 2)
            if forward.shape != expected or inverse.shape != expected:
                raise ValueError(f"Dense maps must both have shape {expected}")
            if not np.isfinite(forward).all() or not np.isfinite(inverse).all():
                raise ValueError("Dense maps must contain only finite coordinates")
            forward.setflags(write=False)
            inverse.setflags(write=False)

        valid = self.valid_atlas_mask
        if valid is not None:
            valid = np.ascontiguousarray(valid, dtype=bool)
            if valid.shape != atlas_shape:
                raise ValueError("Dense registration validity mask must match the atlas canvas")
            valid.setflags(write=False)

        homography.setflags(write=False)
        object.__setattr__(self, "display_to_affine_atlas_h", homography)
        object.__setattr__(self, "display_shape", display_shape)
        object.__setattr__(self, "atlas_shape", atlas_shape)
        object.__setattr__(self, "atlas_to_affine_xy", forward)
        object.__setattr__(self, "affine_to_atlas_xy", inverse)
        object.__setattr__(self, "valid_atlas_mask", valid)

    @property
    def nonlinear(self) -> bool:
        return self.atlas_to_affine_xy is not None

    @property
    def affine_atlas_to_display_h(self) -> np.ndarray:
        inverse = np.linalg.inv(self.display_to_affine_atlas_h)
        inverse.setflags(write=False)
        return inverse

    def map_display_to_atlas(self, display_points_xy: np.ndarray) -> np.ndarray:
        affine_points = _apply_homography(display_points_xy, self.display_to_affine_atlas_h)
        if self.affine_to_atlas_xy is None:
            return affine_points
        atlas_points = _sample_map(self.affine_to_atlas_xy, affine_points)
        if self.valid_atlas_mask is not None:
            atlas_points[~_points_inside_mask(self.valid_atlas_mask, atlas_points)] = np.nan
        return atlas_points

    def map_atlas_to_display(self, atlas_points_xy: np.ndarray) -> np.ndarray:
        atlas_points = _points(atlas_points_xy)
        valid = (
            _points_inside_mask(self.valid_atlas_mask, atlas_points)
            if self.valid_atlas_mask is not None
            else np.ones(len(atlas_points), dtype=bool)
        )
        affine_points = (
            atlas_points
            if self.atlas_to_affine_xy is None
            else _sample_map(self.atlas_to_affine_xy, atlas_points)
        )
        display_points = _apply_homography(affine_points, self.affine_atlas_to_display_h)
        display_points[~valid] = np.nan
        return display_points

    def atlas_to_display_render_map(self) -> np.ndarray:
        atlas_points = (
            _identity_map(self.atlas_shape)
            if self.atlas_to_affine_xy is None
            else self.atlas_to_affine_xy
        )
        display_points = _apply_homography(
            atlas_points.reshape(-1, 2),
            self.affine_atlas_to_display_h,
        )
        return np.ascontiguousarray(display_points.reshape(*self.atlas_shape, 2), dtype=np.float32)

    def render_display_image_in_atlas(
        self,
        display_image: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        if display_image.shape[:2] != self.display_shape:
            raise ValueError("Image must already be in the declared display coordinate frame")
        render_map = self.atlas_to_display_render_map()
        rendered = cv2.remap(
            display_image,
            render_map[..., 0],
            render_map[..., 1],
            interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if self.valid_atlas_mask is not None:
            rendered = rendered.copy()
            rendered[~self.valid_atlas_mask] = 0
        return rendered

    def check_invariants(
        self,
        *,
        maximum_homography_condition: float = 1e8,
        maximum_inverse_cycle_p95_px: float = 2.0,
        minimum_jacobian: float = 0.01,
    ) -> dict:
        condition = float(np.linalg.cond(self.display_to_affine_atlas_h))
        finite_fraction = float(np.isfinite(self.atlas_to_display_render_map()).all(axis=2).mean())
        if condition > maximum_homography_condition:
            raise ValueError("ill-conditioned display-to-affine-atlas homography")
        if finite_fraction < 1.0:
            raise ValueError("non-finite atlas rendering coordinates")
        diagnostics = {
            "coordinate_convention": self.coordinate_convention,
            "display_shape": self.display_shape,
            "atlas_shape": self.atlas_shape,
            "nonlinear": self.nonlinear,
            "homography_condition_number": condition,
            "render_map_finite_fraction": finite_fraction,
        }
        if self.nonlinear:
            forward_cycle = _cycle_error(self.atlas_to_affine_xy, self.affine_to_atlas_xy)
            inverse_cycle = _cycle_error(self.affine_to_atlas_xy, self.atlas_to_affine_xy)
            forward_jacobian = _jacobian_determinant(self.atlas_to_affine_xy)
            inverse_jacobian = _jacobian_determinant(self.affine_to_atlas_xy)
            if self.valid_atlas_mask is None:
                forward_valid = np.isfinite(forward_cycle)
                inverse_valid = np.isfinite(inverse_cycle)
                forward_jacobian_valid = np.ones_like(forward_jacobian, dtype=bool)
                inverse_jacobian_valid = np.ones_like(inverse_jacobian, dtype=bool)
            else:
                forward_valid = np.isfinite(forward_cycle) & self.valid_atlas_mask
                inverse_domain_valid = cv2.remap(
                    self.valid_atlas_mask.astype(np.uint8),
                    self.affine_to_atlas_xy[..., 0],
                    self.affine_to_atlas_xy[..., 1],
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(bool)
                inverse_valid = np.isfinite(inverse_cycle) & inverse_domain_valid
                forward_jacobian_valid = (
                    self.valid_atlas_mask[:-1, :-1]
                    | self.valid_atlas_mask[:-1, 1:]
                    | self.valid_atlas_mask[1:, :-1]
                    | self.valid_atlas_mask[1:, 1:]
                )
                inverse_jacobian_valid = (
                    inverse_domain_valid[:-1, :-1]
                    | inverse_domain_valid[:-1, 1:]
                    | inverse_domain_valid[1:, :-1]
                    | inverse_domain_valid[1:, 1:]
                )
            cycle = np.concatenate((forward_cycle[forward_valid], inverse_cycle[inverse_valid]))
            valid_jacobian = np.concatenate(
                (
                    forward_jacobian[forward_jacobian_valid],
                    inverse_jacobian[inverse_jacobian_valid],
                )
            )
            fold_count = int(np.count_nonzero(valid_jacobian <= 0.0))
            cycle_p95 = float(np.quantile(cycle, 0.95)) if cycle.size else float("inf")
            diagnostics.update(
                inverse_cycle_p95_px=cycle_p95,
                inverse_cycle_max_px=float(cycle.max()) if cycle.size else float("inf"),
                minimum_jacobian=float(valid_jacobian.min()) if valid_jacobian.size else float("-inf"),
                fold_count=fold_count,
            )
            if fold_count:
                raise ValueError("dense registration contains a folded coordinate map")
            if diagnostics["minimum_jacobian"] < minimum_jacobian:
                raise ValueError("dense registration contains a near-singular coordinate map")
            if cycle_p95 > maximum_inverse_cycle_p95_px:
                raise ValueError("dense registration forward/inverse cycle error is too large")
        return diagnostics

    def save_npz(self, path: str | Path) -> None:
        self.check_invariants()
        values = {
            "format_version": np.asarray(FORMAT_VERSION, dtype=np.uint16),
            "coordinate_convention": np.asarray(self.coordinate_convention),
            "display_shape": np.asarray(self.display_shape, dtype=np.int32),
            "atlas_shape": np.asarray(self.atlas_shape, dtype=np.int32),
            "display_to_affine_atlas_h": self.display_to_affine_atlas_h,
            "nonlinear": np.asarray(self.nonlinear, dtype=np.uint8),
        }
        if self.nonlinear:
            values["atlas_to_affine_xy"] = self.atlas_to_affine_xy
            values["affine_to_atlas_xy"] = self.affine_to_atlas_xy
        if self.valid_atlas_mask is not None:
            values["valid_atlas_mask"] = self.valid_atlas_mask.astype(np.uint8)
        if self.registration_metadata_json is not None:
            values["registration_metadata_json"] = np.asarray(self.registration_metadata_json)
        np.savez_compressed(Path(path), **values)

    @classmethod
    def load_npz(cls, path: str | Path) -> "SliceAtlasTransform2D":
        with np.load(Path(path), allow_pickle=False) as values:
            version = int(values["format_version"])
            if version not in {LEGACY_FORMAT_VERSION, FORMAT_VERSION}:
                raise ValueError("Unsupported SliceAtlasTransform2D format version")
            if str(values["coordinate_convention"]) not in {
                "display_xy->affine_atlas_xy;pixel_centers",
                COORDINATE_CONVENTION,
            }:
                raise ValueError("Serialized coordinate convention does not match this implementation")
            nonlinear = version == FORMAT_VERSION and bool(int(values["nonlinear"]))
            transform = cls(
                values["display_to_affine_atlas_h"],
                tuple(values["display_shape"].tolist()),
                tuple(values["atlas_shape"].tolist()),
                values["atlas_to_affine_xy"] if nonlinear else None,
                values["affine_to_atlas_xy"] if nonlinear else None,
                values["valid_atlas_mask"].astype(bool) if "valid_atlas_mask" in values.files else None,
                str(values["registration_metadata_json"]) if "registration_metadata_json" in values.files else None,
            )
        transform.check_invariants()
        return transform
