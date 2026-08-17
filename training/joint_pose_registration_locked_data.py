"""Independent locked synthetic benchmark for joint pose and dense registration.

This module intentionally shares no generator code with the training pipeline.  It
reads the Allen volumes itself, owns a disjoint AP-block split, and uses a cubic
control-grid stationary velocity field plus an independent grayscale corruption
family. Locally locked qualification seeds are supplied by the evaluator; no
hidden-test seed is stored here. Local locking is reproducibility machinery, not
a cryptographic substitute for an externally administered sealed benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import nrrd
import numpy as np
import torch
import torch.nn.functional as F

from source.dense_registration_preprocessing import (
    MASK_CONTRACT_SHA256,
    PREPROCESSING_CONTRACT_V2,
    numpy_cosine_mask_feather,
)


VOXEL_UM = 25.0
BREGMA_AP_INDEX = 216.0
MODEL_SHAPE = (320, 464)
AP_RANGE_UM = (-4500.0, 500.0)
AP_BAND_WIDTH_UM = 500.0
AP_BAND_WIDTH_INDICES = int(AP_BAND_WIDTH_UM / VOXEL_UM)
AP_BAND_COUNT = int((AP_RANGE_UM[1] - AP_RANGE_UM[0]) / AP_BAND_WIDTH_UM)
AP_SPLIT_PATTERN = (
    *("guard",) * 4,
    *("development",) * 4,
    *("guard",) * 2,
    *("locked-validation",) * 4,
    *("guard",) * 2,
    *("sealed-test",) * 4,
)
AP_GUARD_MIN_INDEX_DISTANCE = 3
DENSE_V2_AP_BLOCK_WIDTH = 4
DENSE_V2_AP_SPLIT_PATTERN = (
    "train", "train", "guard", "validation", "guard",
    "train", "train", "guard", "sealed-test", "guard",
)
PUBLIC_SPLITS = ("development", "locked-validation")
ALL_SPLITS = (*PUBLIC_SPLITS, "sealed-test")
SEVERITIES = {
    "clean": dict(
        control=2.0, radial=0.5, anisotropic=0.5, shear=0.5,
        refiner_rotation=2.0, refiner_scale=(0.98, 1.02),
        pose_view_rotation=2.0, pose_view_scale=(0.98, 1.02), translation=2.0,
        gamma=0.0, gain=0.0, offset=(0.0, 0.0), background=0.0,
        bias=0.0, tile=0.0, blowout=0.0, noise=0.0, blur=0.0, damage=0.00,
        speck=0.0, scratch=0.0, bubble=0.0, edge_loss=0.0, blackout=0.0,
    ),
    "mild": dict(
        control=5.0, radial=2.0, anisotropic=2.0, shear=2.0,
        refiner_rotation=7.0, refiner_scale=(0.93, 1.07),
        pose_view_rotation=15.0, pose_view_scale=(0.88, 1.12), translation=5.0,
        gamma=0.28, gain=0.20, offset=(-0.07, 0.10), background=0.10,
        bias=0.09, tile=0.09, blowout=0.22, noise=0.03, blur=0.30, damage=0.18,
        speck=0.00012, scratch=0.22, bubble=0.15, edge_loss=0.16, blackout=0.04,
    ),
    "moderate": dict(
        control=9.0, radial=4.0, anisotropic=4.0, shear=4.0,
        refiner_rotation=12.0, refiner_scale=(0.90, 1.10),
        pose_view_rotation=60.0, pose_view_scale=(0.70, 1.30), translation=10.0,
        gamma=0.55, gain=0.38, offset=(-0.15, 0.22), background=0.22,
        bias=0.18, tile=0.20, blowout=0.48, noise=0.06, blur=0.60, damage=0.42,
        speck=0.00035, scratch=0.50, bubble=0.35, edge_loss=0.36, blackout=0.08,
    ),
    "severe": dict(
        control=14.0, radial=7.0, anisotropic=7.0, shear=7.0,
        refiner_rotation=15.0, refiner_scale=(0.86, 1.14),
        pose_view_rotation=180.0, pose_view_scale=(0.50, 1.50), translation=15.0,
        gamma=0.80, gain=0.50, offset=(-0.20, 0.32), background=0.30,
        bias=0.24, tile=0.28, blowout=0.70, noise=0.09, blur=0.80, damage=0.65,
        speck=0.00065, scratch=0.72, bubble=0.55, edge_loss=0.55, blackout=0.12,
    ),
}
NEGATIVE_AP_UM = (25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
NEGATIVE_TILT_DEG = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
GENERATOR_VERSION = 4
REFINER_PREPROCESSING_CONTRACT = "independent-candidate-pose-mask-affine-v3"
_SEALED_EVALUATOR_CAPABILITY = object()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_sha256() -> str:
    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(source).hexdigest()


def _canonical(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    return value


def _payload_sha256(payload: dict) -> str:
    encoded = json.dumps(
        _canonical(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _case_hashes(manifest: dict) -> np.ndarray:
    count = int(manifest["sample_count"])
    shared = {
        "version": manifest["version"],
        "contract_sha256": manifest["contract_sha256"],
        "split": manifest["split"],
        "seed": manifest["seed"],
        "severity": manifest["severity"],
    }
    hashes = []
    for item in range(count):
        case = dict(shared)
        for key, value in manifest.items():
            if key in {"manifest_sha256", "case_sha256"}:
                continue
            if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == count:
                case[key] = value[item]
        hashes.append(_payload_sha256(case))
    return np.asarray(hashes, dtype="<U64")


def _rng(seed: int, field: str) -> np.random.Generator:
    digest = hashlib.sha256(
        f"joint-locked-independent-v4:{int(seed)}:{field}".encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def _split_indices(split: str, *, allow_sealed: bool = False) -> np.ndarray:
    if split not in ALL_SPLITS:
        raise ValueError(f"split must be one of {ALL_SPLITS}")
    if split == "sealed-test" and not allow_sealed:
        raise PermissionError("locked test data require the local evaluator capability")
    first = int(round(BREGMA_AP_INDEX - AP_RANGE_UM[1] / VOXEL_UM))
    last = int(round(BREGMA_AP_INDEX - AP_RANGE_UM[0] / VOXEL_UM))
    indices = np.arange(first, last + 1, dtype=np.int32)
    relative = indices - first
    bands = np.minimum(relative // AP_BAND_WIDTH_INDICES, AP_BAND_COUNT - 1)
    within_band = relative - bands * AP_BAND_WIDTH_INDICES
    names = np.full(len(indices), "guard", dtype=object)
    patterned = within_band < len(AP_SPLIT_PATTERN)
    names[patterned] = np.asarray(AP_SPLIT_PATTERN, dtype=object)[within_band[patterned]]
    selected = indices[names == split]
    if split == "sealed-test" and np.intersect1d(
        selected, _dense_v2_train_validation_indices()
    ).size:
        raise RuntimeError(
            "locked test AP centers overlap the frozen dense-v2 development union"
        )
    return selected


def _ap_band_indices(indices: np.ndarray) -> np.ndarray:
    first = int(round(BREGMA_AP_INDEX - AP_RANGE_UM[1] / VOXEL_UM))
    return np.minimum(
        (np.asarray(indices, dtype=np.int32) - first) // AP_BAND_WIDTH_INDICES,
        AP_BAND_COUNT - 1,
    ).astype(np.int16)


def _dense_v2_train_validation_indices() -> np.ndarray:
    """Reproduce the frozen dense-v2 center exclusion without importing it."""
    first = int(round(BREGMA_AP_INDEX - AP_RANGE_UM[1] / VOXEL_UM))
    last = int(round(BREGMA_AP_INDEX - AP_RANGE_UM[0] / VOXEL_UM))
    indices = np.arange(first, last + 1, dtype=np.int32)
    blocks = (indices - first) // DENSE_V2_AP_BLOCK_WIDTH
    roles = np.asarray(DENSE_V2_AP_SPLIT_PATTERN, dtype=object)[
        blocks % len(DENSE_V2_AP_SPLIT_PATTERN)
    ]
    return indices[np.isin(roles, ("train", "validation"))]


def split_ap_indices(split: str) -> np.ndarray:
    """Return public split centers; the sealed block map is deliberately gated."""
    return _split_indices(split)


def _identity_grid(batch: int, height: int, width: int, device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((x, y), dim=0)[None].expand(batch, -1, -1, -1).clone()


def _normalized_grid(pixel_map: torch.Tensor) -> torch.Tensor:
    height, width = pixel_map.shape[-2:]
    return torch.stack(
        (
            pixel_map[:, 0] / max(width - 1, 1) * 2.0 - 1.0,
            pixel_map[:, 1] / max(height - 1, 1) * 2.0 - 1.0,
        ),
        dim=-1,
    )


def sample_at(image: torch.Tensor, pixel_map: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    return F.grid_sample(
        image, _normalized_grid(pixel_map), mode=mode,
        padding_mode="zeros", align_corners=True,
    )


def _sample_labels(labels: torch.Tensor, pixel_map: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour label sampling without losing large Allen IDs to float32."""
    sampled = torch.empty_like(labels)
    for item in range(labels.shape[0]):
        values, encoded = torch.unique(labels[item], sorted=True, return_inverse=True)
        encoded = encoded.reshape_as(labels[item]).float()[None]
        indices = sample_at(encoded, pixel_map[item : item + 1], "nearest").long()[0]
        sampled[item] = values[indices.clamp(0, len(values) - 1)]
    return sampled


def compose_pixel_maps(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return ``first(second(x))`` for absolute x/y pixel maps."""
    return sample_at(first, second)


def integrate_stationary_velocity(velocity: torch.Tensor, steps: int = 7) -> torch.Tensor:
    identity = _identity_grid(
        velocity.shape[0], velocity.shape[-2], velocity.shape[-1], velocity.device
    )
    displacement = velocity / float(2**steps)
    for _ in range(steps):
        displacement = displacement + sample_at(displacement, identity + displacement)
    return identity + displacement


def jacobian_determinant(pixel_map: torch.Tensor) -> torch.Tensor:
    dx = pixel_map[:, :, :-1, 1:] - pixel_map[:, :, :-1, :-1]
    dy = pixel_map[:, :, 1:, :-1] - pixel_map[:, :, :-1, :-1]
    return dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]


def _brain_orientation_affine(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    y, x = np.nonzero(mask)
    if len(x) < 64:
        raise ValueError("a candidate-pose mask affine requires at least 64 tissue pixels")
    centered_x = x.astype(np.float64) - x.mean()
    centered_y = y.astype(np.float64) - y.mean()
    angle = np.degrees(
        0.5 * np.arctan2(
            2.0 * np.mean(centered_x * centered_y),
                np.mean(np.square(centered_x)) - np.mean(np.square(centered_y)),
        )
    )
    center = ((mask.shape[1] - 1.0) / 2.0, (mask.shape[0] - 1.0) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    corners = np.asarray(
        (
            (0.0, 0.0, 1.0), (mask.shape[1] - 1.0, 0.0, 1.0),
            (0.0, mask.shape[0] - 1.0, 1.0),
            (mask.shape[1] - 1.0, mask.shape[0] - 1.0, 1.0),
        )
    )
    rotated = (matrix @ corners.T).T
    matrix[:, 2] -= rotated.min(axis=0)
    return np.vstack((matrix, (0.0, 0.0, 1.0)))


def candidate_pose_mask_affine(source_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    """Independently remove raw roll/scale before a candidate-plane refiner call."""
    source_y, source_x = np.nonzero(np.asarray(source_mask, dtype=bool))
    target_y, target_x = np.nonzero(np.asarray(target_mask, dtype=bool))
    if min(len(source_x), len(target_x)) < 64:
        raise ValueError("source and candidate atlas masks require at least 64 tissue pixels")
    orientation = _brain_orientation_affine(source_mask)
    source_points = (
        orientation @ np.column_stack((source_x, source_y, np.ones(len(source_x)))).T
    ).T[:, :2]
    source_span = np.ptp(source_points, axis=0)
    target_span = np.asarray((np.ptp(target_x), np.ptp(target_y)), dtype=np.float64)
    scale = float(np.median(target_span / np.maximum(source_span, 1.0)))
    source_center = (source_points.min(axis=0) + source_points.max(axis=0)) / 2.0
    target_center = np.asarray(
        ((target_x.min() + target_x.max()) / 2.0, (target_y.min() + target_y.max()) / 2.0)
    )
    affine = np.diag((scale, scale, 1.0))
    affine[:2, 2] = target_center - scale * source_center
    return affine @ orientation


def _apply_homography(pixel_map: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return (
        torch.einsum("bij,bjhw->bihw", matrix[:, :2, :2], pixel_map)
        + matrix[:, :2, 2, None, None]
    )


def _remove_control_grid_affine(control: torch.Tensor) -> torch.Tensor:
    """Keep the dense target local; rotation/scale/translation have separate truth."""
    _, _, rows, columns = control.shape
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, rows, device=control.device),
        torch.linspace(-1.0, 1.0, columns, device=control.device), indexing="ij",
    )
    design = torch.stack((torch.ones_like(x), x, y), dim=-1).reshape(-1, 3)
    projection = design @ torch.linalg.pinv(design)
    flat = control.permute(0, 2, 3, 1).reshape(control.shape[0], -1, 2)
    local = flat - torch.einsum("ij,bjk->bik", projection, flat)
    return local.reshape(control.shape[0], rows, columns, 2).permute(0, 3, 1, 2)


def _cubic_velocity(control: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        _remove_control_grid_affine(control), size=shape,
        mode="bicubic", align_corners=True,
    )


def _independent_local_velocity(
    radial: torch.Tensor,
    anisotropic: torch.Tensor,
    shear: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Compose independent smooth radial, anisotropic, and shear velocities."""
    height, width = shape
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=radial.device),
        torch.linspace(-1.0, 1.0, width, device=radial.device),
        indexing="ij",
    )
    x, y = x[None], y[None]

    radial_radius = radial[:, 2, None, None].clamp_min(0.05)
    radial_x = (x - radial[:, 0, None, None]) / radial_radius
    radial_y = (y - radial[:, 1, None, None]) / radial_radius
    radial_weight = torch.exp(-0.5 * (radial_x.square() + radial_y.square()))
    radial_velocity = torch.stack((radial_x, radial_y), dim=1)
    radial_velocity *= radial_weight[:, None] * radial[:, 3, None, None, None]

    anisotropic_radius = anisotropic[:, 2, None, None].clamp_min(0.05)
    dx = (x - anisotropic[:, 0, None, None]) / anisotropic_radius
    dy = (y - anisotropic[:, 1, None, None]) / anisotropic_radius
    cosine = torch.cos(anisotropic[:, 3, None, None])
    sine = torch.sin(anisotropic[:, 3, None, None])
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    anisotropic_weight = torch.exp(-0.5 * (local_x.square() + local_y.square()))
    displaced_x = anisotropic[:, 4, None, None] * local_x * anisotropic_weight
    displaced_y = anisotropic[:, 5, None, None] * local_y * anisotropic_weight
    anisotropic_velocity = torch.stack(
        (
            cosine * displaced_x - sine * displaced_y,
            sine * displaced_x + cosine * displaced_y,
        ),
        dim=1,
    )

    shear_radius = shear[:, 2, None, None].clamp_min(0.05)
    dx = (x - shear[:, 0, None, None]) / shear_radius
    dy = (y - shear[:, 1, None, None]) / shear_radius
    cosine = torch.cos(shear[:, 3, None, None])
    sine = torch.sin(shear[:, 3, None, None])
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    shear_weight = torch.exp(-0.5 * (local_x.square() + local_y.square()))
    displaced_x = shear[:, 4, None, None] * local_y * shear_weight
    shear_velocity = torch.stack(
        (cosine * displaced_x, sine * displaced_x), dim=1
    )
    return radial_velocity + anisotropic_velocity + shear_velocity


def _similarity_maps(
    rotation_deg: torch.Tensor,
    scale: torch.Tensor,
    translation_xy: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(rotation_deg)
    height, width = shape
    identity = _identity_grid(batch, height, width, rotation_deg.device)
    center = torch.tensor(
        ((width - 1) / 2.0, (height - 1) / 2.0),
        device=rotation_deg.device,
    )[None, :, None, None]
    angle = torch.deg2rad(rotation_deg)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    matrix = torch.stack((cosine, -sine, sine, cosine), dim=1).reshape(batch, 2, 2)
    matrix = matrix * scale[:, None, None]
    forward = (
        torch.einsum("bij,bjhw->bihw", matrix, identity - center)
        + center + translation_xy[:, :, None, None]
    )
    inverse_matrix = torch.linalg.inv(matrix)
    inverse = torch.einsum(
        "bij,bjhw->bihw",
        inverse_matrix,
        identity - center - translation_xy[:, :, None, None],
    ) + center
    return forward, inverse, matrix, inverse_matrix


def _blur(image: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    blurred = F.avg_pool2d(image, 5, stride=1, padding=2)
    return image * (1.0 - amount[:, None, None, None]) + blurred * amount[:, None, None, None]


class LockedJointSyntheticBenchmark:
    """Independent generator with a locally capability-gated locked-test path."""

    def __init__(
        self,
        atlas_folder: str | Path,
        device: str | torch.device = "cpu",
    ):
        self.atlas_folder = Path(atlas_folder)
        average_path = self.atlas_folder / "average_template_25.nrrd"
        annotation_path = self.atlas_folder / "annotation_25.nrrd"
        average = nrrd.read(str(average_path))[0]
        annotation = nrrd.read(str(annotation_path))[0]
        if average.ndim != 3 or average.shape != annotation.shape:
            raise ValueError("Allen average and annotation must share AP x DV x ML shape")
        if average.shape[1] > MODEL_SHAPE[0] or average.shape[2] > MODEL_SHAPE[1]:
            raise ValueError("Allen coronal planes exceed the independent benchmark canvas")
        self.device = torch.device(device)
        self.average = torch.from_numpy(
            average.astype(np.float32) / max(float(average.max()), 1.0)
        ).to(self.device)[None, None]
        self.annotation = torch.from_numpy(annotation.astype(np.int64)).to(self.device)
        self.volume_shape = tuple(int(value) for value in average.shape)
        self.pad_y = (MODEL_SHAPE[0] - average.shape[1]) // 2
        self.pad_x = (MODEL_SHAPE[1] - average.shape[2]) // 2
        self.__sealed_consumed = False
        split_contract = {
            "ap_range_um": AP_RANGE_UM,
            "ap_band_width_um": AP_BAND_WIDTH_UM,
            "ap_band_count": AP_BAND_COUNT,
            "pattern": AP_SPLIT_PATTERN,
            "bregma_ap_index": BREGMA_AP_INDEX,
            "excluded_dense_v2_center_contract": {
                "ap_block_width": DENSE_V2_AP_BLOCK_WIDTH,
                "pattern": DENSE_V2_AP_SPLIT_PATTERN,
                "excluded_roles": ("train", "validation"),
                "scope": (
                    "leakage audit for dense-v2/joint generator centers only; "
                    "AtlasPose V7 used the full AP domain, so this benchmark does "
                    "not claim unseen AP anatomy"
                ),
            },
        }
        excluded_centers = _dense_v2_train_validation_indices()
        local_test_centers = _split_indices("sealed-test", allow_sealed=True)
        overlap = np.intersect1d(excluded_centers, local_test_centers)
        if overlap.size:
            raise RuntimeError(
                "locked test AP centers overlap the frozen dense-v2 development union"
            )
        self.contract = {
            "generator_version": GENERATOR_VERSION,
            "implementation": "independent-cubic-plus-local-composition-svf",
            "refiner_preprocessing": REFINER_PREPROCESSING_CONTRACT,
            "dense_preprocessing_contract": PREPROCESSING_CONTRACT_V2,
            "dense_mask_contract_sha256": MASK_CONTRACT_SHA256,
            "voxel_um": VOXEL_UM,
            "model_shape": MODEL_SHAPE,
            "average_template_sha256": _sha256(average_path),
            "annotation_sha256": _sha256(annotation_path),
            "split_contract_sha256": _payload_sha256(split_contract),
            "dense_v2_train_validation_exclusion_sha256": _payload_sha256(
                {
                    "contract": split_contract[
                        "excluded_dense_v2_center_contract"
                    ],
                    "excluded_centers": excluded_centers,
                }
            ),
            "ap_center_exclusion_receipt": {
                "scope": split_contract["excluded_dense_v2_center_contract"]["scope"],
                "dense_v2_train_validation_centers": tuple(
                    int(value) for value in excluded_centers
                ),
                "locked_test_centers": tuple(int(value) for value in local_test_centers),
                "overlap_centers": tuple(int(value) for value in overlap),
            },
            "generator_and_evaluator_source_sha256": _source_sha256(),
        }
        self.contract["contract_sha256"] = _payload_sha256(self.contract)

    def render_planes(
        self,
        ap_index: torch.Tensor | np.ndarray,
        tilt_lr_deg: torch.Tensor | np.ndarray,
        tilt_dv_deg: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ap_index = torch.as_tensor(ap_index, device=self.device, dtype=torch.float32).reshape(-1)
        tilt_lr_deg = torch.as_tensor(
            tilt_lr_deg, device=self.device, dtype=torch.float32
        ).reshape(-1)
        tilt_dv_deg = torch.as_tensor(
            tilt_dv_deg, device=self.device, dtype=torch.float32
        ).reshape(-1)
        if not (len(ap_index) == len(tilt_lr_deg) == len(tilt_dv_deg)):
            raise ValueError("pose arrays must have equal length")
        batch = len(ap_index)
        native_height, native_width = self.volume_shape[1:]
        dv, ml = torch.meshgrid(
            torch.arange(native_height, device=self.device, dtype=torch.float32),
            torch.arange(native_width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        sampled_ap = ap_index[:, None, None]
        sampled_ap = sampled_ap + torch.tan(torch.deg2rad(tilt_lr_deg))[:, None, None] * (
            ml - (native_width - 1) / 2.0
        )
        sampled_ap = sampled_ap + torch.tan(torch.deg2rad(tilt_dv_deg))[:, None, None] * (
            dv - (native_height - 1) / 2.0
        )
        grid = torch.stack(
            (
                ml[None].expand(batch, -1, -1) / max(native_width - 1, 1) * 2.0 - 1.0,
                dv[None].expand(batch, -1, -1) / max(native_height - 1, 1) * 2.0 - 1.0,
                sampled_ap / max(self.volume_shape[0] - 1, 1) * 2.0 - 1.0,
            ), dim=-1,
        )[:, None]
        image = F.grid_sample(
            self.average.expand(batch, -1, -1, -1, -1), grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )[:, :, 0]
        nearest_ap = sampled_ap.round().long()
        valid = (nearest_ap >= 0) & (nearest_ap < self.volume_shape[0])
        labels = self.annotation[
            nearest_ap.clamp(0, self.volume_shape[0] - 1),
            dv.long()[None].expand(batch, -1, -1),
            ml.long()[None].expand(batch, -1, -1),
        ]
        labels = torch.where(valid, labels, 0)[:, None]
        mask = labels > 0
        normalized = torch.zeros_like(image)
        for item in range(batch):
            values = image[item, 0][mask[item, 0]]
            if values.numel():
                low, high = torch.quantile(values, torch.tensor((0.01, 0.99), device=self.device))
                normalized[item] = ((image[item] - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
        pad = (
            self.pad_x, MODEL_SHAPE[1] - native_width - self.pad_x,
            self.pad_y, MODEL_SHAPE[0] - native_height - self.pad_y,
        )
        return F.pad(normalized * mask, pad), F.pad(mask, pad), F.pad(labels, pad)

    def prepare_refiner_inputs(self, batch: dict, candidate_pose: torch.Tensor) -> dict:
        """Apply the runtime-equivalent smart-mask affine for each candidate plane."""
        pose = torch.as_tensor(candidate_pose, device=self.device, dtype=torch.float32)
        if pose.shape != batch["pose"].shape:
            raise ValueError("candidate poses must have shape [batch, 3]")
        fixed, fixed_mask, fixed_labels = self.render_planes(
            BREGMA_AP_INDEX - pose[:, 0] / VOXEL_UM, pose[:, 1], pose[:, 2]
        )
        aligned_image, aligned_raw, aligned_mask, matrices = [], [], [], []
        for image, source_mask, target_mask in zip(
            batch["moving_raw_uint8"], batch["moving_model_mask"], fixed_mask
        ):
            matrix = candidate_pose_mask_affine(
                source_mask[0].detach().cpu().numpy(),
                target_mask[0].detach().cpu().numpy(),
            )
            warped_raw = cv2.warpAffine(
                image[0].detach().cpu().numpy(), matrix[:2],
                (MODEL_SHAPE[1], MODEL_SHAPE[0]), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            warped_mask = cv2.warpAffine(
                source_mask[0].detach().cpu().numpy().astype(np.uint8), matrix[:2],
                (MODEL_SHAPE[1], MODEL_SHAPE[0]), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            ).astype(bool)
            aligned_raw.append(warped_raw)
            aligned_mask.append(warped_mask)
            aligned_image.append(
                warped_raw.astype(np.float32) / 255.0
                * numpy_cosine_mask_feather(warped_mask)
            )
            matrices.append(matrix)
        raw_to_aligned = torch.as_tensor(
            np.stack(matrices), device=self.device, dtype=torch.float32
        )
        return {
            "fixed": fixed,
            "fixed_mask": fixed_mask,
            "fixed_labels": fixed_labels,
            "aligned_moving": torch.from_numpy(np.stack(aligned_image)[:, None]).to(
                self.device, dtype=torch.float32
            ),
            "aligned_moving_raw_uint8": torch.from_numpy(
                np.stack(aligned_raw)[:, None]
            ).to(self.device, dtype=torch.uint8),
            "aligned_moving_mask": torch.from_numpy(np.stack(aligned_mask)[:, None]).to(
                self.device
            ),
            "raw_to_aligned": raw_to_aligned,
            "aligned_to_raw": torch.linalg.inv(raw_to_aligned),
            "map_pose": pose,
            "refiner_preprocessing": REFINER_PREPROCESSING_CONTRACT,
            "dense_preprocessing_contract": PREPROCESSING_CONTRACT_V2,
            "dense_mask_contract_sha256": MASK_CONTRACT_SHA256,
        }

    def compose_refiner_maps_to_source_model(
        self,
        fixed_to_aligned: torch.Tensor,
        aligned_to_fixed: torch.Tensor,
        raw_to_aligned: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose aligned maps onto the canonical pre-refiner model canvas.

        ``raw_to_aligned`` maps the 320 x 464 source model canvas used by this
        benchmark into the candidate-aligned registrar canvas. It does not map an
        arbitrary original photograph; the GUI owns that separate composition.
        """
        raw_to_aligned = torch.as_tensor(
            raw_to_aligned, device=fixed_to_aligned.device, dtype=fixed_to_aligned.dtype
        )
        aligned_to_raw = torch.linalg.inv(raw_to_aligned)
        fixed_to_source = _apply_homography(fixed_to_aligned, aligned_to_raw)
        source_grid = _identity_grid(
            fixed_to_aligned.shape[0], *fixed_to_aligned.shape[-2:], fixed_to_aligned.device
        )
        aligned_grid = _apply_homography(source_grid, raw_to_aligned)
        source_to_fixed = sample_at(aligned_to_fixed, aligned_grid)
        return fixed_to_source, source_to_fixed

    def make_manifest(
        self,
        count: int,
        split: str,
        seed: int,
        severity: str,
        negatives_per_sample: int = 6,
    ) -> dict:
        if split == "sealed-test":
            raise PermissionError(
                "locked manifests are available only inside local qualification"
            )
        return self._make_manifest(count, split, seed, severity, negatives_per_sample, False)

    def _make_manifest(
        self,
        count: int,
        split: str,
        seed: int,
        severity: str,
        negatives_per_sample: int,
        allow_sealed: bool,
    ) -> dict:
        if count < 1 or negatives_per_sample < 1:
            raise ValueError("count and negatives_per_sample must be positive")
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {tuple(SEVERITIES)}")
        pool = _split_indices(split, allow_sealed=allow_sealed)
        pool_bands = _ap_band_indices(pool)
        per_band = np.full(AP_BAND_COUNT, count // AP_BAND_COUNT, dtype=np.int32)
        remainder_order = _rng(seed, "ap-band-remainder").permutation(AP_BAND_COUNT)
        per_band[remainder_order[: count % AP_BAND_COUNT]] += 1
        selected_centers = []
        for band, band_count in enumerate(per_band):
            if not band_count:
                continue
            band_pool = pool[pool_bands == band]
            if not len(band_pool):
                raise ValueError(f"split {split} has no center in AP band {band}")
            selected_centers.append(
                _rng(seed, f"ap-band-{band}").choice(
                    band_pool, size=int(band_count), replace=band_count > len(band_pool)
                )
            )
        center = np.concatenate(selected_centers).astype(np.int32)
        center = center[_rng(seed, "ap-case-order").permutation(count)]
        ap_index = center.astype(np.float32) + _rng(seed, "ap-subvoxel").uniform(
            -0.45, 0.45, count
        ).astype(np.float32)
        first_index = BREGMA_AP_INDEX - AP_RANGE_UM[1] / VOXEL_UM
        last_index = BREGMA_AP_INDEX - AP_RANGE_UM[0] / VOXEL_UM
        ap_index = np.clip(ap_index, first_index, last_index)
        pose = np.column_stack(
            (
                (BREGMA_AP_INDEX - ap_index) * VOXEL_UM,
                _rng(seed, "tilt-lr").uniform(-35.0, 35.0, count),
                _rng(seed, "tilt-dv").uniform(-35.0, 35.0, count),
            )
        ).astype(np.float32)
        control = _rng(seed, "control-grid").normal(size=(count, 2, 5, 7)).astype(np.float32)
        norm = np.sqrt(np.mean(control**2, axis=(1, 2, 3), keepdims=True)).clip(1e-6)
        control *= (
            _rng(seed, "control-amplitude").uniform(
                0.25, SEVERITIES[severity]["control"], count
            ).astype(np.float32)[:, None, None, None] / norm
        )
        damage_probability = SEVERITIES[severity]["damage"]
        negative_pool = np.asarray(
            [(sign * value, 0.0, 0.0) for value in NEGATIVE_AP_UM for sign in (-1, 1)]
            + [(0.0, sign * value, 0.0) for value in NEGATIVE_TILT_DEG for sign in (-1, 1)]
            + [(0.0, 0.0, sign * value) for value in NEGATIVE_TILT_DEG for sign in (-1, 1)],
            dtype=np.float32,
        )
        chosen_offsets = []
        for item in range(count):
            candidates = negative_pool.copy()
            candidate_pose = pose[item, None] + candidates
            candidate_centers = np.rint(
                BREGMA_AP_INDEX - candidate_pose[:, 0] / VOXEL_UM
            ).astype(np.int32)
            valid = (
                (candidate_pose[:, 0] >= AP_RANGE_UM[0])
                & (candidate_pose[:, 0] <= AP_RANGE_UM[1])
                & (np.abs(candidate_pose[:, 1]) <= 35.0)
                & (np.abs(candidate_pose[:, 2]) <= 35.0)
                & np.isin(candidate_centers, pool)
            )
            candidates = candidates[valid]
            if len(candidates) < negatives_per_sample:
                raise ValueError("not enough domain-valid hard negatives for this pose")
            required = []
            required_specs = (
                ("adjacent-ap-negative", 0, VOXEL_UM),
                ("adjacent-lr-negative", 1, NEGATIVE_TILT_DEG[0]),
                ("adjacent-dv-negative", 2, NEGATIVE_TILT_DEG[0]),
            )
            for field, axis, magnitude in required_specs[:min(negatives_per_sample, 3)]:
                other_axes = [value for value in range(3) if value != axis]
                matches = np.flatnonzero(
                    (np.abs(candidates[:, axis]) == magnitude)
                    & (candidates[:, other_axes] == 0.0).all(axis=1)
                )
                if not len(matches):
                    raise ValueError(
                        f"no domain-valid adjacent {('AP', 'L-R', 'D-V')[axis]} "
                        "negative is available"
                    )
                required.append(int(_rng(seed + item, field).choice(matches)))
            remaining = np.setdiff1d(
                np.arange(len(candidates)), np.asarray(required), assume_unique=True
            )
            extra_count = negatives_per_sample - len(required)
            extra = (
                _rng(seed + item, "hard-negatives").choice(
                    remaining, extra_count, replace=False
                )
                if extra_count else np.empty(0, dtype=np.int64)
            )
            choice = np.concatenate((np.asarray(required, dtype=np.int64), extra))
            chosen_offsets.append(candidates[choice])

        polygon_rng = _rng(seed, "polygon")
        polygon_center = polygon_rng.uniform(-0.70, 0.70, (count, 2))
        polygon_angles = np.sort(
            polygon_rng.uniform(-math.pi, math.pi, (count, 6)), axis=1
        )
        polygon_radius = polygon_rng.uniform(0.10, 0.38, (count, 6))
        polygon_xy = polygon_center[:, None] + polygon_radius[:, :, None] * np.stack(
            (np.cos(polygon_angles), np.sin(polygon_angles)), axis=2
        )
        polygon_xy = np.clip(polygon_xy, -1.0, 1.0).astype(np.float32)
        manifest = {
            "version": GENERATOR_VERSION,
            "contract_sha256": self.contract["contract_sha256"],
            "split": split,
            "seed": int(seed),
            "severity": severity,
            "sample_count": int(count),
            "negatives_per_sample": int(negatives_per_sample),
            "ap_block_center": center.astype(np.int32),
            "ap_band_index": _ap_band_indices(center),
            "ap_band_counts": tuple(int(value) for value in per_band),
            "pose": pose,
            "control_velocity_xy": control,
            "radial_velocity": _rng(seed, "radial").uniform(
                (-0.70, -0.70, 0.15, -SEVERITIES[severity]["radial"]),
                (0.70, 0.70, 0.55, SEVERITIES[severity]["radial"]),
                (count, 4),
            ).astype(np.float32),
            "anisotropic_velocity": _rng(seed, "anisotropic").uniform(
                (-0.70, -0.70, 0.15, -math.pi, -SEVERITIES[severity]["anisotropic"],
                 -SEVERITIES[severity]["anisotropic"]),
                (0.70, 0.70, 0.55, math.pi, SEVERITIES[severity]["anisotropic"],
                 SEVERITIES[severity]["anisotropic"]),
                (count, 6),
            ).astype(np.float32),
            "shear_velocity": _rng(seed, "shear").uniform(
                (-0.70, -0.70, 0.15, -math.pi, -SEVERITIES[severity]["shear"]),
                (0.70, 0.70, 0.55, math.pi, SEVERITIES[severity]["shear"]),
                (count, 5),
            ).astype(np.float32),
            "refiner_rotation_deg": _rng(seed, "refiner-rotation").uniform(
                -SEVERITIES[severity]["refiner_rotation"],
                SEVERITIES[severity]["refiner_rotation"], count
            ).astype(np.float32),
            "refiner_scale": _rng(seed, "refiner-scale").uniform(
                *SEVERITIES[severity]["refiner_scale"], count
            ).astype(np.float32),
            "pose_view_rotation_deg": _rng(seed, "pose-view-rotation").uniform(
                -SEVERITIES[severity]["pose_view_rotation"],
                SEVERITIES[severity]["pose_view_rotation"], count
            ).astype(np.float32),
            "pose_view_scale": _rng(seed, "pose-view-scale").uniform(
                *SEVERITIES[severity]["pose_view_scale"], count
            ).astype(np.float32),
            "translation_xy": _rng(seed, "translation").uniform(
                -SEVERITIES[severity]["translation"], SEVERITIES[severity]["translation"],
                (count, 2),
            ).astype(np.float32),
            "gamma": np.exp(_rng(seed, "gamma").uniform(
                -SEVERITIES[severity]["gamma"], SEVERITIES[severity]["gamma"], count
            )).astype(np.float32),
            "gain": np.exp(_rng(seed, "gain").uniform(
                -SEVERITIES[severity]["gain"], SEVERITIES[severity]["gain"], count
            )).astype(np.float32),
            "offset": _rng(seed, "offset").uniform(
                *SEVERITIES[severity]["offset"], count
            ).astype(np.float32),
            "background": _rng(seed, "background").uniform(
                0.0, SEVERITIES[severity]["background"], count
            ).astype(np.float32),
            "bias": _rng(seed, "bias").normal(
                0.0, SEVERITIES[severity]["bias"], (count, 5)
            ).astype(np.float32),
            "tile_period_xy": _rng(seed, "tile-period").uniform(35.0, 100.0, (count, 2)).astype(np.float32),
            "tile_phase_xy": _rng(seed, "tile-phase").uniform(0.0, 1.0, (count, 2)).astype(np.float32),
            "tile_strength": _rng(seed, "tile-strength").uniform(
                0.0, SEVERITIES[severity]["tile"], count
            ).astype(np.float32),
            "blowout_xy": _rng(seed, "blowout-xy").uniform(-0.8, 0.8, (count, 2)).astype(np.float32),
            "blowout_radius": _rng(seed, "blowout-radius").uniform(0.08, 0.35, count).astype(np.float32),
            "blowout_strength": _rng(seed, "blowout-strength").uniform(
                0.0, SEVERITIES[severity]["blowout"], count
            ).astype(np.float32),
            "noise": _rng(seed, "noise-level").uniform(
                0.0, SEVERITIES[severity]["noise"], count
            ).astype(np.float32),
            "blur": _rng(seed, "blur").uniform(
                0.0, SEVERITIES[severity]["blur"], count
            ).astype(np.float32),
            "speck_density": _rng(seed, "speck-density").uniform(
                0.0, SEVERITIES[severity]["speck"], count
            ).astype(np.float32),
            "scratch_enabled": (
                _rng(seed, "scratch-enable").random(count) < SEVERITIES[severity]["scratch"]
            ).astype(np.bool_),
            "scratch": _rng(seed, "scratch").uniform(
                (-math.pi, -0.75, 0.002, 0.25),
                (math.pi, 0.75, 0.012, 0.90), (count, 4),
            ).astype(np.float32),
            "bubble_enabled": (
                _rng(seed, "bubble-enable").random(count) < SEVERITIES[severity]["bubble"]
            ).astype(np.bool_),
            "bubble": _rng(seed, "bubble").uniform(
                (-0.75, -0.75, 0.06, 0.003, 0.15),
                (0.75, 0.75, 0.35, 0.018, 0.70), (count, 5),
            ).astype(np.float32),
            "tear_enabled": (_rng(seed, "tear-enable").random(count) < damage_probability).astype(np.bool_),
            "tear": _rng(seed, "tear").uniform(
                (-0.7, -0.6, -0.25, 0.006), (0.7, 0.6, 0.25, 0.035), (count, 4)
            ).astype(np.float32),
            "missing_enabled": (_rng(seed, "missing-enable").random(count) < damage_probability).astype(np.bool_),
            "missing": _rng(seed, "missing").uniform(
                (-math.pi, 0.12, 0.10), (math.pi, 0.50, 0.42), (count, 3)
            ).astype(np.float32),
            "occlusion_enabled": (_rng(seed, "occlusion-enable").random(count) < damage_probability * 0.6).astype(np.bool_),
            "occlusion": _rng(seed, "occlusion").uniform(
                (-0.65, -0.65, 0.08, 0.08, -math.pi),
                (0.65, 0.65, 0.38, 0.38, math.pi), (count, 5),
            ).astype(np.float32),
            "polygon_enabled": (
                _rng(seed, "polygon-enable").random(count) < damage_probability * 0.6
            ).astype(np.bool_),
            "polygon_xy": polygon_xy,
            "edge_loss_enabled": (
                _rng(seed, "edge-loss-enable").random(count)
                < SEVERITIES[severity]["edge_loss"]
            ).astype(np.bool_),
            "edge_loss_side": _rng(seed, "edge-loss-side").integers(
                0, 4, count, dtype=np.int8
            ),
            "edge_loss": _rng(seed, "edge-loss").uniform(
                (-0.65, 0.12, 0.12, -0.20),
                (0.65, 0.75, 0.68, 0.20), (count, 4),
            ).astype(np.float32),
            "blackout_enabled": (
                _rng(seed, "blackout-enable").random(count)
                < SEVERITIES[severity]["blackout"]
            ).astype(np.bool_),
            "blackout": _rng(seed, "blackout").uniform(
                (-math.pi, -0.35, 0.025),
                (math.pi, 0.35, 0.16), (count, 3),
            ).astype(np.float32),
            "negative_pose_offset": np.stack(chosen_offsets).astype(np.float32),
        }
        pair_ids = []
        for item in range(count):
            digest = hashlib.sha256(
                f"joint-locked-reference-pair-v1:{split}:{severity}:{seed}:{item}".encode()
            )
            for name in (
                "pose", "control_velocity_xy", "radial_velocity",
                "anisotropic_velocity", "shear_velocity", "refiner_rotation_deg",
                "refiner_scale", "pose_view_rotation_deg", "pose_view_scale",
                "translation_xy",
            ):
                digest.update(np.ascontiguousarray(manifest[name][item]).tobytes())
            pair_ids.append(digest.hexdigest())
        manifest["pair_id"] = np.asarray(pair_ids, dtype="<U64")
        manifest["case_sha256"] = _case_hashes(manifest)
        manifest["manifest_sha256"] = _payload_sha256(manifest)
        return manifest

    def _validate_manifest(self, manifest: dict, *, allow_sealed: bool = False) -> None:
        expected = manifest.get("manifest_sha256")
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if expected != _payload_sha256(payload):
            raise ValueError("manifest hash does not match its contents")
        if manifest.get("contract_sha256") != self.contract["contract_sha256"]:
            raise ValueError("manifest belongs to a different independent benchmark contract")
        if not np.array_equal(np.asarray(manifest.get("case_sha256")), _case_hashes(manifest)):
            raise ValueError("independent case hashes do not match their exact parameters")
        split = str(manifest.get("split"))
        pool = _split_indices(split, allow_sealed=allow_sealed)
        count = int(manifest.get("sample_count", -1))
        centers = np.asarray(manifest.get("ap_block_center"), dtype=np.int32)
        pose = np.asarray(manifest.get("pose"), dtype=np.float32)
        if centers.shape != (count,) or pose.shape != (count, 3):
            raise ValueError("manifest sample arrays have inconsistent shapes")
        rendered_centers = BREGMA_AP_INDEX - pose[:, 0] / VOXEL_UM
        if (
            not np.isin(centers, pool).all()
            or np.any(np.abs(rendered_centers - centers) > 0.451)
        ):
            raise ValueError("manifest AP centers are outside their declared split blocks")
        bands = np.asarray(manifest.get("ap_band_index"), dtype=np.int16)
        band_counts = np.asarray(manifest.get("ap_band_counts"), dtype=np.int32)
        observed_counts = np.bincount(_ap_band_indices(centers), minlength=AP_BAND_COUNT)
        if (
            bands.shape != (count,)
            or not np.array_equal(bands, _ap_band_indices(centers))
            or band_counts.shape != (AP_BAND_COUNT,)
            or not np.array_equal(band_counts, observed_counts)
            or observed_counts.max() - observed_counts.min() > 1
        ):
            raise ValueError("manifest AP-band allocation is not balanced or hash-bound")
        if np.abs(pose[:, 1:]).max(initial=0.0) > 35.0:
            raise ValueError("manifest tilts exceed the benchmark pose domain")
        severity = SEVERITIES[str(manifest["severity"])]
        refiner_rotation = np.asarray(manifest["refiner_rotation_deg"])
        refiner_scale = np.asarray(manifest["refiner_scale"])
        pose_view_rotation = np.asarray(manifest["pose_view_rotation_deg"])
        pose_view_scale = np.asarray(manifest["pose_view_scale"])
        if np.abs(refiner_rotation).max(initial=0.0) > severity["refiner_rotation"] + 1e-5:
            raise ValueError("refiner rotation exceeds its post-outline-affine range")
        if (
            np.any(refiner_scale < severity["refiner_scale"][0] - 1e-5)
            or np.any(refiner_scale > severity["refiner_scale"][1] + 1e-5)
        ):
            raise ValueError("refiner scale exceeds its post-outline-affine range")
        if np.abs(pose_view_rotation).max(initial=0.0) > severity["pose_view_rotation"] + 1e-5:
            raise ValueError("pose-view rotation exceeds its raw-image nuisance range")
        if (
            np.any(pose_view_scale < severity["pose_view_scale"][0] - 1e-5)
            or np.any(pose_view_scale > severity["pose_view_scale"][1] + 1e-5)
        ):
            raise ValueError("pose-view scale exceeds its raw-image nuisance range")
        pair_ids = np.asarray(manifest.get("pair_id"))
        if pair_ids.shape != (count,) or not all(
            isinstance(value, str) and len(value) == 64 for value in pair_ids.tolist()
        ):
            raise ValueError("reference/challenge pair identities are missing or malformed")
        offsets = np.asarray(manifest["negative_pose_offset"], dtype=np.float32)
        negative_pose = pose[:, None] + offsets
        negative_centers = np.rint(
            BREGMA_AP_INDEX - negative_pose[:, :, 0] / VOXEL_UM
        ).astype(np.int32)
        if not np.isin(negative_centers, pool).all():
            raise ValueError("negative poses cross the declared AP split")
        adjacent = (
            (np.abs(offsets[:, :, 0]) == VOXEL_UM)
            & (offsets[:, :, 1] == 0.0)
            & (offsets[:, :, 2] == 0.0)
        )
        if not adjacent.any(axis=1).all():
            raise ValueError("every case requires a same-split adjacent AP negative")
        if offsets.shape[1] >= 2:
            adjacent_lr = (
                (offsets[:, :, 0] == 0.0)
                & (np.abs(offsets[:, :, 1]) == NEGATIVE_TILT_DEG[0])
                & (offsets[:, :, 2] == 0.0)
            )
            if not adjacent_lr.any(axis=1).all():
                raise ValueError("every case with K>=2 requires an adjacent L-R negative")
        if offsets.shape[1] >= 3:
            adjacent_dv = (
                (offsets[:, :, 0] == 0.0)
                & (offsets[:, :, 1] == 0.0)
                & (np.abs(offsets[:, :, 2]) == NEGATIVE_TILT_DEG[0])
            )
            if not adjacent_dv.any(axis=1).all():
                raise ValueError("every case with K>=3 requires an adjacent D-V negative")

    def batch(self, manifest: dict, *, qa: bool = False) -> dict:
        if manifest.get("split") == "sealed-test":
            raise PermissionError(
                "locked manifests can only be materialized by local qualification"
            )
        self._validate_manifest(manifest)
        return self._batch(manifest, qa=qa)

    def generate(
        self, count: int, split: str, seed: int, severity: str,
        negatives_per_sample: int = 6, *, qa: bool = False,
    ) -> dict:
        return self.batch(
            self.make_manifest(count, split, seed, severity, negatives_per_sample), qa=qa
        )

    def generate_sealed_once(
        self,
        count: int,
        seed: int,
        severity: str,
        negatives_per_sample: int = 6,
        *,
        _capability=None,
        qa: bool = False,
    ) -> dict:
        if _capability is not _SEALED_EVALUATOR_CAPABILITY:
            raise PermissionError(
                "locked test data require the local evaluator capability"
            )
        if self.__sealed_consumed:
            raise PermissionError("the sealed evaluator capability has already been consumed")
        self.__sealed_consumed = True
        manifest = self._make_manifest(
            count, "sealed-test", seed, severity, negatives_per_sample, True
        )
        self._validate_manifest(manifest, allow_sealed=True)
        return self._batch(manifest, qa=qa)

    def make_balanced_sealed_manifests_once(
        self,
        base_seed: int,
        count_per_stratum: int,
        negatives_per_sample: int,
        *,
        _capability=None,
    ) -> dict[str, dict]:
        """Consume one local capability and freeze all four stratum manifests."""
        if _capability is not _SEALED_EVALUATOR_CAPABILITY:
            raise PermissionError(
                "locked test data require the local evaluator capability"
            )
        if self.__sealed_consumed:
            raise PermissionError("the sealed evaluator capability has already been consumed")
        self.__sealed_consumed = True
        manifests = {
            severity: self._make_manifest(
                count_per_stratum,
                "sealed-test",
                int.from_bytes(
                    hashlib.sha256(
                        int(base_seed).to_bytes(8, "big") + severity.encode("ascii")
                    ).digest()[:8],
                    "big",
                ) & ((1 << 63) - 1),
                severity,
                negatives_per_sample,
                True,
            )
            for severity in SEVERITIES
        }
        for manifest in manifests.values():
            self._validate_manifest(manifest, allow_sealed=True)
        return manifests

    def _batch(self, manifest: dict, *, qa: bool) -> dict:
        pose = torch.as_tensor(manifest["pose"], device=self.device, dtype=torch.float32)
        fixed, fixed_mask, fixed_labels = self.render_planes(
            BREGMA_AP_INDEX - pose[:, 0] / VOXEL_UM, pose[:, 1], pose[:, 2]
        )
        control = torch.as_tensor(
            manifest["control_velocity_xy"], device=self.device, dtype=torch.float32
        )
        radial = torch.as_tensor(
            manifest["radial_velocity"], device=self.device, dtype=torch.float32
        )
        anisotropic = torch.as_tensor(
            manifest["anisotropic_velocity"], device=self.device, dtype=torch.float32
        )
        shear = torch.as_tensor(
            manifest["shear_velocity"], device=self.device, dtype=torch.float32
        )
        velocity = _cubic_velocity(control, MODEL_SHAPE) + _independent_local_velocity(
            radial, anisotropic, shear, MODEL_SHAPE
        )
        cell_mask = (
            fixed_mask[:, :, :-1, :-1] | fixed_mask[:, :, :-1, 1:]
            | fixed_mask[:, :, 1:, :-1] | fixed_mask[:, :, 1:, 1:]
        )[:, 0]
        for _ in range(24):
            local_forward = integrate_stationary_velocity(velocity)
            local_inverse = integrate_stationary_velocity(-velocity)
            forward_min = jacobian_determinant(local_forward).masked_fill(~cell_mask, torch.inf).amin((1, 2))
            inverse_min = jacobian_determinant(local_inverse).masked_fill(~cell_mask, torch.inf).amin((1, 2))
            failing = torch.minimum(forward_min, inverse_min) < 0.20
            if not bool(failing.any()):
                break
            velocity = velocity * torch.where(failing, 0.72, 1.0)[:, None, None, None]
        if bool(failing.any()):
            raise RuntimeError("independent deformation failed the positive-Jacobian contract")

        rotation = torch.as_tensor(manifest["refiner_rotation_deg"], device=self.device)
        scale = torch.as_tensor(manifest["refiner_scale"], device=self.device)
        translation = torch.as_tensor(manifest["translation_xy"], device=self.device)
        similarity_forward, similarity_inverse, similarity_matrix, _ = _similarity_maps(
            rotation, scale, translation, MODEL_SHAPE
        )
        center = torch.tensor(
            ((MODEL_SHAPE[1] - 1) / 2.0, (MODEL_SHAPE[0] - 1) / 2.0),
            device=self.device,
        )[None, :, None, None]
        fixed_to_moving = (
            torch.einsum("bij,bjhw->bihw", similarity_matrix, local_forward - center)
            + center + translation[:, :, None, None]
        )
        identity = _identity_grid(len(pose), *MODEL_SHAPE, self.device)
        moving_to_fixed = similarity_inverse + sample_at(
            local_inverse - identity, similarity_inverse
        )
        moving_clean = sample_at(fixed, moving_to_fixed)
        moving_labels = _sample_labels(fixed_labels, moving_to_fixed)
        moving_tissue = sample_at(fixed_mask.float(), moving_to_fixed, "nearest") > 0.5
        moving, artifact_masks = self._appearance(moving_clean, moving_tissue, manifest)
        damage = artifact_masks["damage"]
        moving_model = moving_tissue & ~(
            artifact_masks["missing"] | artifact_masks["edge_loss"]
        )
        moving_visible = moving_tissue & ~damage
        fixed_visible = fixed_mask & (sample_at(
            moving_visible.float(), fixed_to_moving, "nearest"
        ) > 0.5)
        reference_visible_labels = _sample_labels(moving_labels, fixed_to_moving)

        pose_view_rotation = torch.as_tensor(
            manifest["pose_view_rotation_deg"], device=self.device
        )
        pose_view_scale = torch.as_tensor(manifest["pose_view_scale"], device=self.device)
        pose_view_translation = torch.zeros(
            (len(pose), 2), device=self.device, dtype=torch.float32
        )
        pose_view_forward, pose_view_inverse, _, _ = _similarity_maps(
            pose_view_rotation, pose_view_scale, pose_view_translation, MODEL_SHAPE
        )
        moving_raw_uint8 = torch.round(moving * 255.0).to(torch.uint8)
        pose_view_raw_uint8 = torch.round(
            sample_at(moving_raw_uint8.float(), pose_view_inverse)
        ).clamp(0, 255).to(torch.uint8)
        pose_view = pose_view_raw_uint8.float() / 255.0
        pose_view_mask = sample_at(
            moving_model.float(), pose_view_inverse, "nearest"
        ) > 0.5
        reference_moving_raw_uint8 = torch.round(
            moving_clean.clamp(0, 1) * 255.0
        ).to(torch.uint8)
        reference_pose_view_raw_uint8 = torch.round(
            sample_at(reference_moving_raw_uint8.float(), pose_view_inverse)
        ).clamp(0, 255).to(torch.uint8)
        reference_pose_view_mask = sample_at(
            moving_tissue.float(), pose_view_inverse, "nearest"
        ) > 0.5

        offsets = torch.as_tensor(
            manifest["negative_pose_offset"], device=self.device, dtype=torch.float32
        )
        negative_pose = pose[:, None] + offsets
        flat = negative_pose.reshape(-1, 3)
        negative_fixed, negative_mask, negative_labels = self.render_planes(
            BREGMA_AP_INDEX - flat[:, 0] / VOXEL_UM, flat[:, 1], flat[:, 2]
        )
        batch, negatives = negative_pose.shape[:2]
        result = {
            "fixed": fixed.float(),
            "moving": moving.float(),
            "moving_raw_uint8": moving_raw_uint8,
            "pose_view": pose_view.float(),
            "pose_view_raw_uint8": pose_view_raw_uint8,
            "pose_view_mask": pose_view_mask,
            "reference_moving_raw_uint8": reference_moving_raw_uint8,
            "reference_moving_model_mask": moving_tissue,
            "reference_pose_view_raw_uint8": reference_pose_view_raw_uint8,
            "reference_pose_view_mask": reference_pose_view_mask,
            "refiner_to_pose_view_map": pose_view_forward.float(),
            "pose_view_to_refiner_map": pose_view_inverse.float(),
            "pose_view_dense_target_valid": torch.zeros(
                len(pose), dtype=torch.bool, device=self.device
            ),
            "fixed_mask": fixed_mask,
            "moving_tissue_mask": moving_tissue,
            "moving_model_mask": moving_model,
            "moving_damage_mask": damage,
            "moving_edge_loss_mask": artifact_masks["edge_loss"],
            "moving_blackout_mask": artifact_masks["blackout"],
            "moving_polygon_mask": artifact_masks["polygon"],
            "moving_optical_artifact_mask": artifact_masks["optical"],
            "moving_speck_mask": artifact_masks["speck"],
            "moving_scratch_mask": artifact_masks["scratch"],
            "moving_bubble_mask": artifact_masks["bubble"],
            "moving_visible_mask": moving_visible,
            "fixed_visible_mask": fixed_visible,
            "fixed_labels": fixed_labels.long(),
            "moving_labels": moving_labels.long(),
            "reference_visible_labels": reference_visible_labels.long(),
            "fixed_to_moving": fixed_to_moving.float(),
            "moving_to_fixed": moving_to_fixed.float(),
            "local_velocity": velocity.float(),
            "similarity_forward": similarity_forward.float(),
            "similarity_inverse": similarity_inverse.float(),
            "refiner_rotation_deg": rotation.float(),
            "refiner_scale": scale.float(),
            "pose_view_rotation_deg": pose_view_rotation.float(),
            "pose_view_scale": pose_view_scale.float(),
            "translation_xy": translation.float(),
            "pose": pose,
            "negative_pose": negative_pose,
            "negative_pose_offset": offsets,
            "negative_fixed": negative_fixed.reshape(batch, negatives, 1, *MODEL_SHAPE),
            "negative_fixed_mask": negative_mask.reshape(batch, negatives, 1, *MODEL_SHAPE),
            "negative_fixed_labels": negative_labels.reshape(batch, negatives, 1, *MODEL_SHAPE),
            "negative_dense_target_valid": torch.zeros(
                (batch, negatives), dtype=torch.bool, device=self.device
            ),
            "case_sha256": np.asarray(manifest["case_sha256"]).copy(),
            "pair_id": np.asarray(manifest["pair_id"]).copy(),
            "split": str(manifest["split"]),
            "severity": str(manifest["severity"]),
            "manifest_sha256": str(manifest["manifest_sha256"]),
            "contract": dict(self.contract),
        }
        if qa:
            result["moving_clean"] = moving_clean.float()
        return result

    def _appearance(
        self, clean: torch.Tensor, tissue: torch.Tensor, manifest: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, _, height, width = clean.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=self.device),
            torch.linspace(-1.0, 1.0, width, device=self.device), indexing="ij",
        )
        x, y = x[None, None], y[None, None]
        tensor = lambda name: torch.as_tensor(
            manifest[name], device=self.device, dtype=torch.float32
        )
        image = clean.clamp(0, 1).pow(tensor("gamma")[:, None, None, None])
        image = image * tensor("gain")[:, None, None, None] + tensor("offset")[:, None, None, None]
        bias = tensor("bias")[:, :, None, None]
        image *= torch.exp(
            bias[:, 0:1] * x + bias[:, 1:2] * y + bias[:, 2:3] * x * y
            + bias[:, 3:4] * x.square() + bias[:, 4:5] * y.square()
        )
        periods = tensor("tile_period_xy")[:, :, None, None]
        phase = tensor("tile_phase_xy")[:, :, None, None]
        tile_coordinate_x = (x + 1) * width / periods[:, 0:1] + phase[:, 0:1]
        tile_coordinate_y = (y + 1) * height / periods[:, 1:2] + phase[:, 1:2]
        tile_x = torch.remainder(tile_coordinate_x, 1.0) - 0.5
        tile_y = torch.remainder(tile_coordinate_y, 1.0) - 0.5
        strength = tensor("tile_strength")[:, None, None, None]
        image *= 1.0 - strength * (tile_x.square() + tile_y.square())
        image += strength * 0.18 * ((tile_x.abs() > 0.47) | (tile_y.abs() > 0.47))
        tile_variation = torch.sin(
            torch.floor(tile_coordinate_x) * 12.9898
            + torch.floor(tile_coordinate_y) * 78.233
            + phase[:, 0:1] * 37.719
        )
        image *= 1.0 + 0.30 * strength * tile_variation
        blowout_xy = tensor("blowout_xy")
        radius = tensor("blowout_radius")[:, None, None, None]
        blowout = torch.exp(
            -((x - blowout_xy[:, 0, None, None, None]).square()
              + (y - blowout_xy[:, 1, None, None, None]).square())
            / radius.square().clamp_min(1e-5)
        )
        image += blowout * tensor("blowout_strength")[:, None, None, None]
        noise_parts, speck_parts = [], []
        for case_hash in manifest["case_sha256"]:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(str(case_hash)[:16], 16) ^ 0x4C4F434B)
            noise_parts.append(
                torch.randn((1, 1, height, width), device=self.device, generator=generator)
            )
            speck_parts.append(
                torch.rand((1, 1, height, width), device=self.device, generator=generator)
            )
        noise = torch.cat(noise_parts)
        image += noise * tensor("noise")[:, None, None, None]
        image = _blur(image, tensor("blur"))
        background = (
            tensor("background")[:, None, None, None] + 0.2 * noise * tensor("noise")[:, None, None, None]
        ).clamp(0, 1)
        image = torch.where(tissue, image, background)

        specks = torch.cat(speck_parts) < tensor("speck_density")[:, None, None, None]
        specks = F.max_pool2d(specks.float(), 3, stride=1, padding=1) > 0.5
        scratch = tensor("scratch")
        scratch_signed = (
            torch.cos(scratch[:, 0, None, None, None]) * x
            + torch.sin(scratch[:, 0, None, None, None]) * y
            - scratch[:, 1, None, None, None]
        )
        scratch_mask = scratch_signed.abs() < scratch[:, 2, None, None, None]
        scratch_mask &= torch.as_tensor(
            manifest["scratch_enabled"], device=self.device
        )[:, None, None, None]
        bubble = tensor("bubble")
        bubble_distance = torch.sqrt(
            (x - bubble[:, 0, None, None, None]).square()
            + (y - bubble[:, 1, None, None, None]).square()
        )
        bubble_mask = (
            (bubble_distance - bubble[:, 2, None, None, None]).abs()
            < bubble[:, 3, None, None, None]
        )
        bubble_mask &= torch.as_tensor(
            manifest["bubble_enabled"], device=self.device
        )[:, None, None, None]
        optical = specks | scratch_mask | bubble_mask
        image = torch.maximum(image, specks.float())
        image += scratch_mask * scratch[:, 3, None, None, None]
        image += bubble_mask * bubble[:, 4, None, None, None]

        tear = tensor("tear")
        tear_signed = y - tear[:, 0, None, None, None] - tear[:, 1, None, None, None] * x
        tear_signed -= tear[:, 2, None, None, None] * x.square()
        tear_mask = tear_signed.abs() < tear[:, 3, None, None, None]
        tear_mask &= torch.as_tensor(
            manifest["tear_enabled"], device=self.device
        )[:, None, None, None]
        missing = tensor("missing")
        edge_x = 0.95 * torch.cos(missing[:, 0, None, None, None])
        edge_y = 0.95 * torch.sin(missing[:, 0, None, None, None])
        missing_mask = (
            ((x - edge_x) / missing[:, 1, None, None, None]).square()
            + ((y - edge_y) / missing[:, 2, None, None, None]).square()
        ) < 1.0
        missing_mask &= torch.as_tensor(
            manifest["missing_enabled"], device=self.device
        )[:, None, None, None]
        occult = tensor("occlusion")
        cosine = torch.cos(occult[:, 4, None, None, None])
        sine = torch.sin(occult[:, 4, None, None, None])
        dx, dy = x - occult[:, 0, None, None, None], y - occult[:, 1, None, None, None]
        rx, ry = cosine * dx - sine * dy, sine * dx + cosine * dy
        occlusion_mask = (
            (rx / occult[:, 2, None, None, None]).square()
            + (ry / occult[:, 3, None, None, None]).square()
        ) < 1.0
        occlusion_mask &= torch.as_tensor(
            manifest["occlusion_enabled"], device=self.device
        )[:, None, None, None]
        polygon_mask = torch.zeros_like(tissue)
        for item, (enabled, vertices) in enumerate(
            zip(manifest["polygon_enabled"], manifest["polygon_xy"])
        ):
            if not bool(enabled):
                continue
            points = np.asarray(vertices, dtype=np.float32).copy()
            points[:, 0] = (points[:, 0] + 1.0) * (width - 1) / 2.0
            points[:, 1] = (points[:, 1] + 1.0) * (height - 1) / 2.0
            raster = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(raster, [np.rint(points).astype(np.int32)], 1)
            polygon_mask[item, 0] = torch.from_numpy(raster).to(
                self.device, dtype=torch.bool
            )
        edge = tensor("edge_loss")
        side = torch.as_tensor(
            manifest["edge_loss_side"], device=self.device
        )[:, None, None, None]
        edge_depth = torch.where(
            side == 0, x + 1.0,
            torch.where(side == 1, 1.0 - x, torch.where(side == 2, y + 1.0, 1.0 - y)),
        )
        edge_along = torch.where(side < 2, y, x)
        fraction = edge_depth / edge[:, 2, None, None, None].clamp_min(1e-4)
        edge_center = edge[:, 0, None, None, None] + edge[:, 3, None, None, None] * fraction
        edge_half_width = edge[:, 1, None, None, None] * (1.0 - fraction)
        edge_loss_mask = (
            (edge_depth >= 0.0) & (fraction <= 1.0)
            & ((edge_along - edge_center).abs() <= edge_half_width)
        )
        edge_loss_mask &= torch.as_tensor(
            manifest["edge_loss_enabled"], device=self.device
        )[:, None, None, None]
        blackout = tensor("blackout")
        blackout_signed = (
            torch.cos(blackout[:, 0, None, None, None]) * x
            + torch.sin(blackout[:, 0, None, None, None]) * y
            - blackout[:, 1, None, None, None]
        )
        blackout_mask = blackout_signed.abs() < blackout[:, 2, None, None, None]
        blackout_mask &= torch.as_tensor(
            manifest["blackout_enabled"], device=self.device
        )[:, None, None, None]
        edge_loss_mask &= tissue
        blackout_mask &= tissue
        damage = (
            tear_mask | missing_mask | occlusion_mask | polygon_mask
            | edge_loss_mask | blackout_mask
        ) & tissue
        image = torch.where(damage, background, image)
        image = torch.where(blackout_mask, torch.zeros_like(image), image)
        return image.clamp(0, 1), {
            "damage": damage,
            "missing": missing_mask & tissue,
            "edge_loss": edge_loss_mask,
            "blackout": blackout_mask,
            "polygon": polygon_mask & tissue,
            "optical": optical,
            "speck": specks,
            "scratch": scratch_mask,
            "bubble": bubble_mask,
        }

    def evaluate_predictions(self, batch: dict, prediction: dict) -> dict[str, float]:
        """Evaluate final maps in their predicted-pose atlas coordinate domain."""
        return evaluate_predictions(self, batch, prediction)


def _macro_dice(truth: torch.Tensor, estimate: torch.Tensor, valid: torch.Tensor) -> float:
    values = []
    for item in range(truth.shape[0]):
        for region in torch.unique(truth[item, 0][valid[item, 0]]):
            if int(region) == 0:
                continue
            target = (truth[item, 0] == region) & valid[item, 0]
            predicted = (estimate[item, 0] == region) & valid[item, 0]
            values.append(2.0 * (target & predicted).sum() / (target.sum() + predicted.sum()))
    return float(torch.stack(values).mean()) if values else float("nan")


def _label_interior(labels: torch.Tensor) -> torch.Tensor:
    interior = torch.ones_like(labels, dtype=torch.bool)
    interior[:, :, 0] = interior[:, :, -1] = False
    interior[:, :, :, 0] = interior[:, :, :, -1] = False
    interior[:, :, 1:] &= labels[:, :, 1:] == labels[:, :, :-1]
    interior[:, :, :-1] &= labels[:, :, :-1] == labels[:, :, 1:]
    interior[:, :, :, 1:] &= labels[:, :, :, 1:] == labels[:, :, :, :-1]
    interior[:, :, :, :-1] &= labels[:, :, :, :-1] == labels[:, :, :, 1:]
    return interior


def _cycle_p95(
    forward: torch.Tensor, inverse: torch.Tensor, valid: torch.Tensor
) -> float:
    cycle = compose_pixel_maps(inverse, forward)
    identity = _identity_grid(forward.shape[0], *forward.shape[-2:], forward.device)
    error = torch.linalg.vector_norm(cycle - identity, dim=1)[valid[:, 0]]
    return float(torch.quantile(error, 0.95)) if error.numel() else float("nan")


def _negative_jacobian_fraction(pixel_map: torch.Tensor, mask: torch.Tensor) -> float:
    cells = (
        mask[:, :, :-1, :-1] | mask[:, :, :-1, 1:]
        | mask[:, :, 1:, :-1] | mask[:, :, 1:, 1:]
    )[:, 0]
    values = jacobian_determinant(pixel_map)[cells]
    return float((values <= 0).float().mean()) if values.numel() else float("nan")


def evaluate_predictions(
    benchmark: LockedJointSyntheticBenchmark, batch: dict, prediction: dict
) -> dict[str, float]:
    """Score maps on the canonical pre-refiner source model canvas."""
    pose = torch.as_tensor(prediction["pose"], device=batch["pose"].device)
    if "map_pose" not in prediction:
        raise ValueError("map_pose is required to bind final maps to their atlas plane")
    if prediction.get("map_space") != "source-model-canvas":
        raise ValueError("final maps must be composed onto the source model canvas")
    if tuple(prediction.get("source_shape", ())) != MODEL_SHAPE:
        raise ValueError("source_shape must declare the canonical source model canvas")
    if prediction.get("refiner_preprocessing") != REFINER_PREPROCESSING_CONTRACT:
        raise ValueError("the candidate-pose mask affine preprocessing receipt is required")
    forward = torch.as_tensor(
        prediction["fixed_to_source_model"], device=batch["fixed_to_moving"].device
    )
    inverse = torch.as_tensor(
        prediction["source_model_to_fixed"], device=batch["moving_to_fixed"].device
    )
    map_pose = torch.as_tensor(prediction["map_pose"], device=batch["pose"].device)
    if not torch.allclose(map_pose, pose, rtol=0.0, atol=1e-5):
        raise ValueError("final maps must declare the same predicted pose as the pose output")
    pose_error = (pose - batch["pose"]).abs()
    predicted_fixed, predicted_mask, predicted_labels = benchmark.render_planes(
        BREGMA_AP_INDEX - map_pose[:, 0] / VOXEL_UM, map_pose[:, 1], map_pose[:, 2]
    )
    del predicted_fixed
    predicted_visible = predicted_mask & (
        sample_at(batch["moving_visible_mask"].float(), forward, "nearest") > 0.5
    )
    recovered_labels = _sample_labels(batch["moving_labels"], forward)
    exact = (recovered_labels == predicted_labels)[predicted_visible]
    interior_visible = predicted_visible & _label_interior(predicted_labels)
    interior_exact = (recovered_labels == predicted_labels)[interior_visible]
    metrics = {
        "ap_mae_um": float(pose_error[:, 0].mean()),
        "lr_mae_deg": float(pose_error[:, 1].mean()),
        "dv_mae_deg": float(pose_error[:, 2].mean()),
        "end_to_end_visible_region_correspondence": (
            float(exact.float().mean()) if exact.numel() else float("nan")
        ),
        "end_to_end_macro_region_dice": _macro_dice(
            predicted_labels, recovered_labels, predicted_visible
        ),
        "end_to_end_interior_region_correspondence": (
            float(interior_exact.float().mean()) if interior_exact.numel() else float("nan")
        ),
        "end_to_end_interior_fraction": float(
            interior_visible.sum() / predicted_visible.sum().clamp_min(1)
        ),
        "end_to_end_visible_fraction": float(
            predicted_visible.sum() / predicted_mask.sum().clamp_min(1)
        ),
        "end_to_end_negative_jacobian_fraction": _negative_jacobian_fraction(
            forward, predicted_mask
        ),
        "end_to_end_forward_inverse_cycle_p95_px": _cycle_p95(
            forward, inverse, predicted_visible
        ),
    }

    exact_forward = prediction.get("exact_plane_fixed_to_source_model")
    exact_inverse = prediction.get("exact_plane_source_model_to_fixed")
    exact_pose = prediction.get("exact_plane_pose")
    exact_receipt = (exact_forward, exact_inverse, exact_pose)
    if any(value is not None for value in exact_receipt) and not all(
        value is not None for value in exact_receipt
    ):
        raise ValueError("exact-plane warp evaluation requires maps and their exact-plane pose")
    if all(value is None for value in exact_receipt) and torch.allclose(
        map_pose, batch["pose"], rtol=0.0, atol=1e-5
    ):
        exact_forward, exact_inverse, exact_pose = forward, inverse, map_pose
    if torch.allclose(map_pose, batch["pose"], rtol=0.0, atol=1e-5):
        ceiling = (
            batch["reference_visible_labels"] == batch["fixed_labels"]
        )[batch["fixed_visible_mask"]]
        metrics["true_pose_source_model_correspondence_ceiling"] = float(
            ceiling.float().mean()
        )
    if exact_forward is not None or exact_inverse is not None or exact_pose is not None:
        if exact_forward is None or exact_inverse is None or exact_pose is None:
            raise ValueError("exact-plane warp evaluation requires maps and their exact-plane pose")
        exact_pose = torch.as_tensor(exact_pose, device=batch["pose"].device)
        if not torch.allclose(exact_pose, batch["pose"], rtol=0.0, atol=1e-5):
            raise ValueError("exact-plane warp maps are not in the true-plane coordinate domain")
        exact_forward = torch.as_tensor(exact_forward, device=forward.device)
        exact_inverse = torch.as_tensor(exact_inverse, device=inverse.device)
        valid = batch["fixed_visible_mask"]
        endpoint = torch.linalg.vector_norm(
            exact_forward - batch["fixed_to_moving"], dim=1
        )[valid[:, 0]]
        recovered = _sample_labels(batch["moving_labels"], exact_forward)
        reference = batch["reference_visible_labels"]
        correspondence = (recovered == reference)[valid]
        metrics.update(
            warp_only_endpoint_mean_px=float(endpoint.mean()),
            warp_only_endpoint_p95_px=float(torch.quantile(endpoint, 0.95)),
            warp_only_visible_region_correspondence=float(correspondence.float().mean()),
            warp_only_macro_region_dice=_macro_dice(reference, recovered, valid),
            warp_only_negative_jacobian_fraction=_negative_jacobian_fraction(
                exact_forward, batch["fixed_mask"]
            ),
            warp_only_forward_inverse_cycle_p95_px=_cycle_p95(
                exact_forward, exact_inverse, valid
            ),
        )
    return metrics
