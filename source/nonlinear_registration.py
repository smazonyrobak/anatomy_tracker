"""Canonical display-slice to 2-D atlas transform.

Raw histology coordinates are deliberately outside this module. The caller
first applies its raw-to-display rotation/flip matrix; every source point and
image accepted here is in the resulting display pixel coordinates ``(x, y)``.
Atlas and affine-atlas coordinates are likewise pixel-center ``(x, y)``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np


FORMAT_VERSION = 2
COORDINATE_CONVENTION = "display_xy->affine_atlas_xy->atlas_xy;pixel_centers"
MODEL_CONTRACT_VERSION = 1
MODEL_SHAPE = (320, 464)
MODEL_PIXEL_SPACING_UM = 25.0
MODEL_SPATIAL_CONTRACT = "one_to_one_atlas_pixels_center_pad_or_crop_no_resize"
MODEL_INPUT_NAMES = ("fixed", "moving", "fixed_mask", "moving_mask")
MODEL_OUTPUT_NAMES = ("atlas_to_affine", "affine_to_atlas", "velocity", "rejection_logit")
MINIMUM_JACOBIAN = 0.20
MAXIMUM_ABS_LOG_JACOBIAN_P99 = 1.50
MAXIMUM_ABS_LOG_JACOBIAN = 1.61
MAXIMUM_INVERSE_P95_PX = 1.0
MAXIMUM_INVERSE_PX = 2.0
MAXIMUM_RESIDUAL_AFFINE_PX = 0.05
MAXIMUM_OUTSIDE_TISSUE_DISPLACEMENT_PX = 1e-3
MAXIMUM_DISPLACEMENT_P95_PX = 8.0
MAXIMUM_DISPLACEMENT_PX = 12.0
REJECTION_PROBABILITY_THRESHOLD = 0.5
RUNTIME_GATE_CONTRACT = {
    "minimum_jacobian": MINIMUM_JACOBIAN,
    "maximum_abs_log_jacobian_p99": MAXIMUM_ABS_LOG_JACOBIAN_P99,
    "maximum_abs_log_jacobian": MAXIMUM_ABS_LOG_JACOBIAN,
    "maximum_inverse_p95_px": MAXIMUM_INVERSE_P95_PX,
    "maximum_inverse_px": MAXIMUM_INVERSE_PX,
    "maximum_residual_affine_px": MAXIMUM_RESIDUAL_AFFINE_PX,
    "maximum_outside_tissue_displacement_px": MAXIMUM_OUTSIDE_TISSUE_DISPLACEMENT_PX,
    "maximum_displacement_p95_px": MAXIMUM_DISPLACEMENT_P95_PX,
    "maximum_displacement_px": MAXIMUM_DISPLACEMENT_PX,
    "rejection_probability_threshold": REJECTION_PROBABILITY_THRESHOLD,
}


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


def _hard_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    hard = np.asarray(mask) > 0.5
    if hard.shape != shape or not hard.any():
        raise ValueError("Trusted masks must be non-empty and match the nonlinear map canvas")
    return hard


def _cell_mask(mask: np.ndarray) -> np.ndarray:
    return mask[:-1, :-1] | mask[:-1, 1:] | mask[1:, :-1] | mask[1:, 1:]


def _cycle_error(first_map: np.ndarray, second_map: np.ndarray) -> np.ndarray:
    height, width = first_map.shape[:2]
    valid = (
        (first_map[..., 0] >= 0.0) & (first_map[..., 0] <= width - 1.0)
        & (first_map[..., 1] >= 0.0) & (first_map[..., 1] <= height - 1.0)
    )
    round_trip = np.stack(
        [
            cv2.remap(
                second_map[..., axis], first_map[..., 0], first_map[..., 1],
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            for axis in range(2)
        ],
        axis=-1,
    )
    error = np.linalg.norm(round_trip - _identity_map((height, width)), axis=2)
    error[~valid] = np.nan
    return error


def tissue_affine_max(pixel_map: np.ndarray, tissue_mask: np.ndarray) -> float:
    """Maximum magnitude of the best affine displacement fitted inside tissue."""
    height, width = tissue_mask.shape
    grid = _identity_map((height, width))
    x = grid[..., 0] * (2.0 / (width - 1)) - 1.0
    y = grid[..., 1] * (2.0 / (height - 1)) - 1.0
    basis = np.stack((np.ones_like(x), x, y), axis=-1)[tissue_mask]
    displacement = (pixel_map - grid)[tissue_mask]
    coefficients = np.linalg.lstsq(basis.astype(np.float64), displacement.astype(np.float64), rcond=None)[0]
    return float(np.abs(basis @ coefficients).max())


def nonlinear_acceptance_failures(diagnostics: dict[str, float | int]) -> list[str]:
    checks = (
        (diagnostics["fold_count"] == 0, "predicted map contains a fold"),
        (diagnostics["minimum_jacobian"] >= MINIMUM_JACOBIAN, "minimum Jacobian is below 0.20"),
        (
            diagnostics["maximum_abs_log_jacobian_p99"] <= MAXIMUM_ABS_LOG_JACOBIAN_P99,
            "trusted-tissue absolute log-Jacobian p99 exceeds 1.50",
        ),
        (
            diagnostics["maximum_abs_log_jacobian"] <= MAXIMUM_ABS_LOG_JACOBIAN,
            "trusted-tissue absolute log-Jacobian maximum exceeds 1.61",
        ),
        (diagnostics["inverse_finite_fraction"] == 1.0, "a forward/inverse cycle leaves the canvas"),
        (diagnostics["inverse_p95_px"] <= MAXIMUM_INVERSE_P95_PX, "round-trip p95 exceeds 1 px"),
        (diagnostics["inverse_max_px"] <= MAXIMUM_INVERSE_PX, "round-trip maximum exceeds 2 px"),
        (
            diagnostics["residual_affine_max_px"] <= MAXIMUM_RESIDUAL_AFFINE_PX,
            "residual global affine exceeds 0.05 px",
        ),
        (
            diagnostics["outside_tissue_displacement_max_px"] <= MAXIMUM_OUTSIDE_TISSUE_DISPLACEMENT_PX,
            "outside-tissue map is not identity",
        ),
        (
            diagnostics["displacement_p95_px"] <= MAXIMUM_DISPLACEMENT_P95_PX,
            "accepted-pair displacement p95 exceeds 8 px",
        ),
        (
            diagnostics["displacement_max_px"] <= MAXIMUM_DISPLACEMENT_PX,
            "accepted-pair displacement maximum exceeds 12 px",
        ),
    )
    return [message for passed, message in checks if not passed]


@dataclass(frozen=True)
class NonlinearWarpAttestation:
    atlas_mask: np.ndarray
    affine_mask: np.ndarray
    model_sha256: str
    manifest_sha256: str
    pixel_spacing_um: float = MODEL_PIXEL_SPACING_UM

    def __post_init__(self) -> None:
        atlas_mask = np.asarray(self.atlas_mask) > 0.5
        affine_mask = np.asarray(self.affine_mask) > 0.5
        if atlas_mask.shape != affine_mask.shape or not atlas_mask.any() or not affine_mask.any():
            raise ValueError("Attestation masks must be non-empty and share one canvas")
        for value in (self.model_sha256, self.manifest_sha256):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
                raise ValueError("Attestation hashes must be lowercase SHA-256 hex digests")
        if not np.isclose(float(self.pixel_spacing_um), MODEL_PIXEL_SPACING_UM):
            raise ValueError("Nonlinear warp attestation must use 25 um one-to-one atlas pixels")
        atlas_mask.setflags(write=False)
        affine_mask.setflags(write=False)
        object.__setattr__(self, "atlas_mask", atlas_mask)
        object.__setattr__(self, "affine_mask", affine_mask)
        object.__setattr__(self, "model_sha256", self.model_sha256.lower())
        object.__setattr__(self, "manifest_sha256", self.manifest_sha256.lower())
        object.__setattr__(self, "pixel_spacing_um", float(self.pixel_spacing_um))


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
        if min(atlas_to_affine.shape[:2]) < 2:
            raise ValueError("Nonlinear maps must have at least two pixels on each axis")
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
        d_dx = self.atlas_to_affine_xy[:-1, 1:] - self.atlas_to_affine_xy[:-1, :-1]
        d_dy = self.atlas_to_affine_xy[1:, :-1] - self.atlas_to_affine_xy[:-1, :-1]
        return d_dx[..., 0] * d_dy[..., 1] - d_dx[..., 1] * d_dy[..., 0]

    def inverse_consistency_error(self) -> np.ndarray:
        return _cycle_error(self.atlas_to_affine_xy, self.affine_to_atlas_xy)

    def diagnostics(
        self,
        atlas_mask: np.ndarray | None = None,
        affine_mask: np.ndarray | None = None,
    ) -> dict[str, float | int]:
        atlas_mask = np.ones(self.shape, bool) if atlas_mask is None else _hard_mask(atlas_mask, self.shape)
        affine_mask = np.ones(self.shape, bool) if affine_mask is None else _hard_mask(affine_mask, self.shape)
        forward_jacobian = self.jacobian_determinant()
        reverse = NonlinearWarp2D(self.affine_to_atlas_xy, self.atlas_to_affine_xy)
        inverse_jacobian = reverse.jacobian_determinant()
        forward_cycle = _cycle_error(self.atlas_to_affine_xy, self.affine_to_atlas_xy)[atlas_mask]
        inverse_cycle = _cycle_error(self.affine_to_atlas_xy, self.atlas_to_affine_xy)[affine_mask]
        forward_finite = np.isfinite(forward_cycle)
        inverse_finite = np.isfinite(inverse_cycle)
        cycle = np.concatenate((forward_cycle[forward_finite], inverse_cycle[inverse_finite]))
        trusted_jacobian = np.concatenate(
            (forward_jacobian[_cell_mask(atlas_mask)], inverse_jacobian[_cell_mask(affine_mask)])
        )
        positive_jacobian = trusted_jacobian[trusted_jacobian > 0.0]
        absolute_log_jacobian = np.abs(np.log(positive_jacobian))
        identity = _identity_map(self.shape)
        forward_displacement = np.linalg.norm(self.atlas_to_affine_xy - identity, axis=2)
        inverse_displacement = np.linalg.norm(self.affine_to_atlas_xy - identity, axis=2)
        trusted = atlas_mask | affine_mask
        outside_forward = forward_displacement[~trusted]
        outside_inverse = inverse_displacement[~trusted]
        outside = np.concatenate((outside_forward, outside_inverse))
        displacement = np.concatenate((forward_displacement[atlas_mask], inverse_displacement[affine_mask]))
        forward_affine = tissue_affine_max(self.atlas_to_affine_xy, atlas_mask)
        inverse_affine = tissue_affine_max(self.affine_to_atlas_xy, affine_mask)
        return {
            "map_finite_fraction": 1.0,
            "minimum_forward_jacobian": float(forward_jacobian.min()),
            "minimum_inverse_jacobian": float(inverse_jacobian.min()),
            "minimum_jacobian": float(min(forward_jacobian.min(), inverse_jacobian.min())),
            "maximum_abs_log_jacobian_p99": (
                float(np.percentile(absolute_log_jacobian, 99)) if len(absolute_log_jacobian) else float("inf")
            ),
            "maximum_abs_log_jacobian": (
                float(absolute_log_jacobian.max()) if len(absolute_log_jacobian) else float("inf")
            ),
            "fold_count": int((forward_jacobian <= 0.0).sum() + (inverse_jacobian <= 0.0).sum()),
            "inverse_finite_fraction": float(
                (forward_finite.sum() + inverse_finite.sum()) / (len(forward_cycle) + len(inverse_cycle))
            ),
            "inverse_p95_px": float(np.percentile(cycle, 95)) if len(cycle) else float("inf"),
            "inverse_max_px": float(cycle.max()) if len(cycle) else float("inf"),
            "forward_tissue_affine_max_px": forward_affine,
            "inverse_tissue_affine_max_px": inverse_affine,
            "residual_affine_max_px": max(forward_affine, inverse_affine),
            "outside_tissue_displacement_max_px": float(outside.max()) if len(outside) else 0.0,
            "displacement_p95_px": float(np.percentile(displacement, 95)),
            "displacement_max_px": float(displacement.max()),
        }

    def check_acceptance(self, atlas_mask: np.ndarray, affine_mask: np.ndarray) -> dict[str, float | int]:
        diagnostics = self.diagnostics(atlas_mask, affine_mask)
        failures = nonlinear_acceptance_failures(diagnostics)
        if failures:
            raise ValueError("; ".join(failures))
        return diagnostics


def _attestation_sha256(warp: NonlinearWarp2D, attestation: NonlinearWarpAttestation) -> str:
    digest = hashlib.sha256()
    for array in (
        warp.atlas_to_affine_xy,
        warp.affine_to_atlas_xy,
        attestation.atlas_mask.astype(np.uint8),
        attestation.affine_mask.astype(np.uint8),
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(attestation.model_sha256.encode("ascii"))
    digest.update(attestation.manifest_sha256.encode("ascii"))
    digest.update(np.asarray(attestation.pixel_spacing_um, dtype="<f8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SliceAtlasTransform2D:
    """One display-to-affine homography followed by an optional residual warp."""

    display_to_affine_atlas_h: np.ndarray
    display_shape: tuple[int, int]
    atlas_shape: tuple[int, int]
    nonlinear: NonlinearWarp2D | None = None
    nonlinear_attestation: NonlinearWarpAttestation | None = None

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
        if self.nonlinear is None and self.nonlinear_attestation is not None:
            raise ValueError("An affine-only transform cannot carry a nonlinear attestation")
        if self.nonlinear_attestation is not None and self.nonlinear_attestation.atlas_mask.shape != atlas_shape:
            raise ValueError("Nonlinear attestation masks must equal the atlas canvas shape")
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
        if self.nonlinear is None:
            return affine_points
        atlas_points = self.nonlinear.map_affine_to_atlas(affine_points)
        outside = ~np.isfinite(atlas_points).all(axis=1)
        atlas_points[outside] = affine_points[outside]
        return atlas_points

    def map_atlas_to_display(self, atlas_points_xy: np.ndarray) -> np.ndarray:
        atlas_points = _points(atlas_points_xy)
        if self.nonlinear is None:
            affine_points = atlas_points
        else:
            affine_points = self.nonlinear.map_atlas_to_affine(atlas_points)
            outside = ~np.isfinite(affine_points).all(axis=1)
            affine_points[outside] = atlas_points[outside]
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
            warp_diagnostics = {
                "map_finite_fraction": 1.0,
                "minimum_forward_jacobian": 1.0,
                "minimum_inverse_jacobian": 1.0,
                "minimum_jacobian": 1.0,
                "maximum_abs_log_jacobian_p99": 0.0,
                "maximum_abs_log_jacobian": 0.0,
                "fold_count": 0,
                "inverse_finite_fraction": 1.0,
                "inverse_p95_px": 0.0,
                "inverse_max_px": 0.0,
                "forward_tissue_affine_max_px": 0.0,
                "inverse_tissue_affine_max_px": 0.0,
                "residual_affine_max_px": 0.0,
                "outside_tissue_displacement_max_px": 0.0,
                "displacement_p95_px": 0.0,
                "displacement_max_px": 0.0,
            }
        else:
            warp_diagnostics = self.nonlinear.diagnostics(
                None if self.nonlinear_attestation is None else self.nonlinear_attestation.atlas_mask,
                None if self.nonlinear_attestation is None else self.nonlinear_attestation.affine_mask,
            )
        return {
            "coordinate_convention": self.coordinate_convention,
            "display_shape": self.display_shape,
            "atlas_shape": self.atlas_shape,
            "nonlinear": self.nonlinear is not None,
            "homography_condition_number": condition,
            "render_map_finite_fraction": render_finite_fraction,
            **warp_diagnostics,
        }

    def check_invariants(
        self,
        *,
        maximum_homography_condition: float = 1e8,
    ) -> dict[str, float | int | bool | tuple[int, int] | str]:
        diagnostics = self.diagnostics()
        failures = []
        if diagnostics["homography_condition_number"] > maximum_homography_condition:
            failures.append("ill-conditioned display-to-atlas homography")
        if diagnostics["render_map_finite_fraction"] < 1.0:
            failures.append("non-finite atlas rendering coordinates")
        if self.nonlinear is not None:
            failures.extend(nonlinear_acceptance_failures(diagnostics))
            if self.nonlinear_attestation is None:
                failures.append("nonlinear warp has no acceptance attestation")
        if failures:
            raise ValueError("; ".join(failures))
        return diagnostics

    def save_npz(self, path: str | Path) -> None:
        if self.nonlinear is not None:
            self.check_invariants()
            assert self.nonlinear_attestation is not None
        empty_map = np.empty((0, 0, 2), dtype=np.float32)
        empty_mask = np.empty((0, 0), dtype=np.uint8)
        atlas_to_affine = empty_map if self.nonlinear is None else self.nonlinear.atlas_to_affine_xy
        affine_to_atlas = empty_map if self.nonlinear is None else self.nonlinear.affine_to_atlas_xy
        attestation = self.nonlinear_attestation
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
            atlas_mask=empty_mask if attestation is None else attestation.atlas_mask.astype(np.uint8),
            affine_mask=empty_mask if attestation is None else attestation.affine_mask.astype(np.uint8),
            model_sha256=np.asarray("" if attestation is None else attestation.model_sha256),
            manifest_sha256=np.asarray("" if attestation is None else attestation.manifest_sha256),
            pixel_spacing_um=np.asarray(0.0 if attestation is None else attestation.pixel_spacing_um, dtype=np.float64),
            attestation_sha256=np.asarray(
                "" if attestation is None else _attestation_sha256(self.nonlinear, attestation)
            ),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "SliceAtlasTransform2D":
        with np.load(Path(path), allow_pickle=False) as values:
            if int(values["format_version"]) != FORMAT_VERSION:
                raise ValueError("Unsupported SliceAtlasTransform2D format version")
            if str(values["coordinate_convention"]) != COORDINATE_CONVENTION:
                raise ValueError("Serialized coordinate convention does not match this implementation")
            nonlinear = None
            attestation = None
            if bool(values["nonlinear"]):
                nonlinear = NonlinearWarp2D(values["atlas_to_affine_xy"], values["affine_to_atlas_xy"])
                attestation = NonlinearWarpAttestation(
                    values["atlas_mask"],
                    values["affine_mask"],
                    str(values["model_sha256"]),
                    str(values["manifest_sha256"]),
                    float(values["pixel_spacing_um"]),
                )
                if str(values["attestation_sha256"]) != _attestation_sha256(nonlinear, attestation):
                    raise ValueError("Serialized nonlinear acceptance attestation is corrupt")
            transform = cls(
                values["display_to_affine_atlas_h"],
                tuple(values["display_shape"].tolist()),
                tuple(values["atlas_shape"].tolist()),
                nonlinear,
                attestation,
            )
            transform.check_invariants()
            return transform
