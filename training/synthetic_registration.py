"""Deterministic on-the-fly pairs for atlas-to-histology registration.

The fixed image is an Allen atlas plane. ``fixed_to_moving`` gives the moving
pixel occupied by each fixed point; ``moving_to_fixed`` is its numerical inverse
and is used to synthesize the moving image. Maps are absolute ``(x, y)`` pixel
coordinates with shape ``B, 2, H, W``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import nrrd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from source.dense_registration_preprocessing import (
    MASK_CONTRACT_SHA256,
    MODEL_SHAPE,
    NATIVE_SHAPE,
    PAD_X,
    PREPROCESSING_CONTRACT_V2,
    cosine_mask_feather,
)

VOXEL_UM = 25.0
BREGMA_AP_INDEX = 216.0
AP_MIN_UM = -4500.0
AP_MAX_UM = 500.0
GENERATOR_VERSION = 2
AP_BLOCK_WIDTH = 4
AP_SPLIT_PATTERN = (
    "train", "train", "guard", "validation", "guard",
    "train", "train", "guard", "sealed-test", "guard",
)
SPLITS = ("train", "validation", "sealed-test")
_FINAL_HOLDOUT_CAPABILITY = object()
COMPONENTS = 4
QUERY_SHA256 = "5347daf90e02ac1d1cfcbf9c8af86ff23a2fb32cd7e7a2ba2881951931286dbd"
V2_SEMANTICS_VERSION = 1
V2_APPEARANCE_CONTRACT = (
    "template-or-query-hierarchy-conditioned-grayscale;per-sample-independent-style;"
    "all-artifacts-then-uint8-quantization-then-shared-cosine-mask-feather;"
    "fixed-nonvisible-tissue-is-unobservable-damage"
)
PLANE_SAMPLING_CONTRACT = (
    "oblique_coronal_plane_after_upstream_pose_freeze;AP_split_is_by_plane_center;"
    "tilted_training_and_evaluation_cover_the_runtime_fixed-atlas_distribution"
)

STRATA = {
    "clean": {
        "rotation": 2.0, "translation": 2.0, "log_scale": math.log(1.02),
        "atlas_tilt": 2.0,
        "components": (1, 2), "amplitude": (1.0, 4.0), "damage": 0.05,
        "noise": (0.0, 0.015), "blur": (0.0, 0.18),
        "gamma_log": 0.08, "gain_log": 0.08, "offset": (-0.03, 0.03),
        "bias": 0.05, "background": 0.03, "invert": 0.0,
    },
    "mild": {
        "rotation": 7.0, "translation": 6.0, "log_scale": math.log(1.07),
        "atlas_tilt": 8.0,
        "components": (1, 3), "amplitude": (2.0, 10.0), "damage": 0.35,
        "noise": (0.005, 0.045), "blur": (0.0, 0.45),
        "gamma_log": 0.45, "gain_log": 0.35, "offset": (-0.10, 0.15),
        "bias": 0.25, "background": 0.15, "invert": 0.12,
    },
    "hard": {
        "rotation": 15.0, "translation": 12.0, "log_scale": math.log(1.14),
        "atlas_tilt": 15.0,
        "components": (2, 4), "amplitude": (4.0, 20.0), "damage": 0.75,
        "noise": (0.015, 0.085), "blur": (0.12, 0.75),
        "gamma_log": 0.80, "gain_log": 0.55, "offset": (-0.22, 0.32),
        "bias": 0.45, "background": 0.38, "invert": 0.35,
    },
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_sha256(path: Path) -> str:
    source = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
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


def _v2_rng(seed: int, field: str) -> np.random.Generator:
    digest = hashlib.sha256(f"dense-registration-v2:{int(seed)}:{field}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def _query_hierarchy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    records: dict[int, tuple[int, ...]] = {0: (0,)}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            structure_id = int(row["id"])
            hierarchy = tuple(
                int(value) for value in row["structure_id_path"].split("/") if value
            )
            records[structure_id] = hierarchy or (structure_id,)
    ids = np.asarray(sorted(records), dtype=np.int64)
    ancestors = np.asarray(
        [
            [records[int(structure_id)][min(depth, len(records[int(structure_id)]) - 1)]
             for structure_id in ids]
            for depth in range(4, 8)
        ],
        dtype=np.int64,
    )
    return ids, ancestors


def _label_tones(ancestor_ids: np.ndarray, seed: np.uint64) -> np.ndarray:
    with np.errstate(over="ignore"):
        values = ancestor_ids.astype(np.uint64) ^ np.uint64(seed)
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
    tones = 0.08 + 0.84 * (
        ((values >> np.uint64(40)) & np.uint64(0xFFFFFF)).astype(np.float32)
        / float(0xFFFFFF)
    )
    tones[ancestor_ids == 0] = 0.0
    return tones.astype(np.float32)


def split_ap_indices(split: str, *, _final_capability=None) -> np.ndarray:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if split == "sealed-test" and _final_capability is not _FINAL_HOLDOUT_CAPABILITY:
        raise PermissionError("final holdout is available only to the one-shot evaluator")
    first = int(round(BREGMA_AP_INDEX - AP_MAX_UM / VOXEL_UM))
    last = int(round(BREGMA_AP_INDEX - AP_MIN_UM / VOXEL_UM))
    indices = np.arange(first, last + 1, dtype=np.int32)
    blocks = (indices - first) // AP_BLOCK_WIDTH
    names = np.asarray(AP_SPLIT_PATTERN, dtype=object)[blocks % len(AP_SPLIT_PATTERN)]
    return indices[names == split]


def _identity_grid(batch: int, height: int, width: int, device, dtype=torch.float32) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def _sample(image: torch.Tensor, pixel_map: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    height, width = pixel_map.shape[-2:]
    grid = torch.stack(
        (
            pixel_map[:, 0] * (2.0 / (width - 1)) - 1.0,
            pixel_map[:, 1] * (2.0 / (height - 1)) - 1.0,
        ),
        dim=-1,
    )
    return F.grid_sample(
        image, grid, mode=mode, padding_mode="zeros", align_corners=True
    )


def _sample_labels(labels: torch.Tensor, pixel_map: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour sampling without casting full Allen IDs to float."""
    batch, _, height, width = labels.shape
    x = pixel_map[:, 0].round().long()
    y = pixel_map[:, 1].round().long()
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    linear = (y.clamp(0, height - 1) * width + x.clamp(0, width - 1)).flatten(1)
    sampled = labels[:, 0].flatten(1).gather(1, linear).reshape(batch, 1, height, width)
    return torch.where(valid[:, None], sampled, 0)


def _sample_field(field: torch.Tensor, pixel_map: torch.Tensor) -> torch.Tensor:
    height, width = pixel_map.shape[-2:]
    grid = torch.stack(
        (
            pixel_map[:, 0] * (2.0 / (width - 1)) - 1.0,
            pixel_map[:, 1] * (2.0 / (height - 1)) - 1.0,
        ),
        dim=-1,
    )
    return F.grid_sample(
        field, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def compose_pixel_maps(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return ``second(first(x))`` for absolute pixel maps."""
    return _sample_field(second, first)


def integrate_velocity(velocity: torch.Tensor, steps: int = 8) -> torch.Tensor:
    identity = _identity_grid(
        velocity.shape[0], velocity.shape[-2], velocity.shape[-1],
        velocity.device, torch.float32,
    )
    displacement = velocity.float() / float(2**steps)
    for _ in range(steps):
        displacement = displacement + _sample_field(displacement, identity + displacement)
    return identity + displacement


def jacobian_determinant(pixel_map: torch.Tensor) -> torch.Tensor:
    d_x = pixel_map[:, :, :-1, 1:] - pixel_map[:, :, :-1, :-1]
    d_y = pixel_map[:, :, 1:, :-1] - pixel_map[:, :, :-1, :-1]
    return d_x[:, 0] * d_y[:, 1] - d_x[:, 1] * d_y[:, 0]


def _remove_tissue_affine(
    velocity: torch.Tensor, support: torch.Tensor, tissue: torch.Tensor
) -> torch.Tensor:
    batch, _, height, width = velocity.shape
    grid = _identity_grid(batch, height, width, velocity.device)
    basis = torch.stack(
        (
            torch.ones_like(grid[:, 0]),
            grid[:, 0] * (2.0 / (width - 1)) - 1.0,
            grid[:, 1] * (2.0 / (height - 1)) - 1.0,
        ),
        dim=1,
    ).flatten(2)
    weights = (support * tissue.float()).flatten(2)
    gram = (basis * weights) @ basis.transpose(1, 2)
    regularizer = gram.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1.0) * 1e-6
    gram = gram + torch.eye(3, device=velocity.device)[None] * regularizer[:, None, None]
    moments = (velocity.flatten(2) * weights) @ basis.transpose(1, 2)
    coefficients = torch.linalg.solve(gram, moments.transpose(1, 2)).transpose(1, 2)
    affine = (coefficients @ basis).reshape_as(velocity)
    return (velocity - affine) * support


def _similarity_inverse_and_homography(
    rotation_deg: torch.Tensor,
    scale: torch.Tensor,
    translation_xy: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(rotation_deg)
    height, width = shape
    identity = _identity_grid(batch, height, width, rotation_deg.device)
    center = torch.tensor(((width - 1) / 2.0, (height - 1) / 2.0), device=rotation_deg.device)
    angle = torch.deg2rad(rotation_deg)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    matrix = scale[:, None, None] * torch.stack(
        (cosine, -sine, sine, cosine), dim=1
    ).reshape(-1, 2, 2)
    inverse_matrix = torch.linalg.inv(matrix)
    inverse_points = identity.flatten(2).transpose(1, 2) - center - translation_xy[:, None]
    inverse = (inverse_points @ inverse_matrix.transpose(1, 2) + center).transpose(1, 2)
    inverse = inverse.reshape(batch, 2, height, width)
    homography = torch.eye(3, device=rotation_deg.device)[None].repeat(batch, 1, 1)
    homography[:, :2, :2] = matrix
    homography[:, :2, 2] = center + translation_xy - torch.einsum("bij,j->bi", matrix, center)
    return inverse, homography


def make_registration_manifest(
    contract: dict,
    count: int,
    split: str,
    seed: int,
    stratum: str,
    *,
    _final_capability=None,
) -> dict[str, np.ndarray | str | int]:
    """Purely regenerate a committed v2 sample manifest without loading the atlas."""
    contract_payload = dict(contract)
    commitment = contract_payload.pop("contract_sha256", None)
    if (
        contract.get("profile") != "v2"
        or not commitment
        or _payload_sha256(contract_payload) != commitment
    ):
        raise ValueError("invalid v2 generator contract")
    if stratum not in STRATA:
        raise ValueError(f"stratum must be one of {tuple(STRATA)}")
    pool = split_ap_indices(split, _final_capability=_final_capability)
    rng = np.random.default_rng(seed)
    config = STRATA[stratum]
    ap_index = rng.choice(pool, count, replace=True).astype(np.float32)
    component_count = rng.integers(
        config["components"][0], config["components"][1] + 1, count, dtype=np.int16
    )
    scale_classes = rng.integers(0, 3, (count, COMPONENTS), dtype=np.int8)
    sigma_ranges = np.asarray(
        (((72, 150), (52, 112)), ((30, 82), (24, 68)), ((12, 42), (10, 36))),
        dtype=np.float32,
    )
    sigma = np.empty((count, COMPONENTS, 2), np.float32)
    for item in range(count):
        for component in range(COMPONENTS):
            ranges = sigma_ranges[scale_classes[item, component]]
            sigma[item, component, 0] = rng.uniform(*ranges[0])
            sigma[item, component, 1] = rng.uniform(*ranges[1])
    damage = config["damage"]
    manifest: dict[str, np.ndarray | str | int] = {
        "format_version": GENERATOR_VERSION,
        "contract_sha256": commitment,
        "split": split,
        "stratum": stratum,
        "seed": int(seed),
        "ap_index": ap_index,
        "ap_um": ((BREGMA_AP_INDEX - ap_index) * VOXEL_UM).astype(np.float32),
        "tilt_lr_deg": rng.uniform(-config["atlas_tilt"], config["atlas_tilt"], count).astype(np.float32),
        "tilt_dv_deg": rng.uniform(-config["atlas_tilt"], config["atlas_tilt"], count).astype(np.float32),
        "rotation_deg": rng.uniform(-config["rotation"], config["rotation"], count).astype(np.float32),
        "scale": np.exp(rng.uniform(-config["log_scale"], config["log_scale"], count)).astype(np.float32),
        "translation_xy": rng.uniform(-config["translation"], config["translation"], (count, 2)).astype(np.float32),
        "component_count": component_count,
        "component_kind": rng.integers(0, 3, (count, COMPONENTS), dtype=np.int8),
        "component_center_xy": np.stack(
            (rng.uniform(0.18, 0.82, (count, COMPONENTS)) * (MODEL_SHAPE[1] - 1),
             rng.uniform(0.14, 0.86, (count, COMPONENTS)) * (MODEL_SHAPE[0] - 1)),
            axis=-1,
        ).astype(np.float32),
        "component_sigma_xy": sigma,
        "component_amplitude": (
            rng.uniform(*config["amplitude"], (count, COMPONENTS))
            * rng.choice(np.asarray([-1.0, 1.0]), (count, COMPONENTS))
        ).astype(np.float32),
        "component_anisotropy": rng.uniform(-1.0, 1.0, (count, COMPONENTS)).astype(np.float32),
        "gamma": np.exp(rng.uniform(-config["gamma_log"], config["gamma_log"], count)).astype(np.float32),
        "gain": np.exp(rng.uniform(-config["gain_log"], config["gain_log"], count)).astype(np.float32),
        "offset": rng.uniform(*config["offset"], count).astype(np.float32),
        "invert": rng.random(count) < config["invert"],
        "bias": rng.uniform(-config["bias"], config["bias"], (count, 5)).astype(np.float32),
        "background": rng.uniform(0.0, config["background"], count).astype(np.float32),
        "noise": rng.uniform(*config["noise"], count).astype(np.float32),
        "blur": rng.uniform(*config["blur"], count).astype(np.float32),
        "tile_period_xy": rng.uniform(24.0, 96.0, (count, 2)).astype(np.float32),
        "tile_angle_deg": rng.uniform(-45.0, 45.0, count).astype(np.float32),
        "tile_strength": rng.uniform(0.0, (0.08, 0.20, 0.34)[tuple(STRATA).index(stratum)], count).astype(np.float32),
        "speck_density": rng.uniform(0.0, (0.0002, 0.0012, 0.0035)[tuple(STRATA).index(stratum)], count).astype(np.float32),
        "blowout_center_xy": rng.uniform((0.15, 0.12), (0.85, 0.88), (count, 2)).astype(np.float32),
        "blowout_radius": rng.uniform(0.025, 0.14, count).astype(np.float32),
        "blowout_strength": rng.uniform(0.0, (0.10, 0.45, 0.85)[tuple(STRATA).index(stratum)], count).astype(np.float32),
        "tear_enabled": rng.random(count) < damage * 0.70,
        "tear_parameters": rng.uniform((-0.65, -0.9, -0.28, 0.006), (0.65, 0.9, 0.28, 0.026), (count, 4)).astype(np.float32),
        "missing_enabled": rng.random(count) < damage * 0.60,
        "missing_parameters": rng.uniform((0.0, 0.12, 0.10), (2.0 * math.pi, 0.34, 0.32), (count, 3)).astype(np.float32),
        "occlusion_enabled": rng.random(count) < damage * 0.35,
        "occlusion_parameters": rng.uniform((0.18, 0.18, 0.08, 0.08, -math.pi), (0.82, 0.82, 0.32, 0.32, math.pi), (count, 5)).astype(np.float32),
    }
    if stratum == "clean":
        offsets, probabilities = (-1, 0, 1), (0.10, 0.80, 0.10)
    elif stratum == "mild":
        offsets, probabilities = (-2, -1, 0, 1, 2), (0.10, 0.15, 0.50, 0.15, 0.10)
    else:
        offsets = (-3, -2, -1, 0, 1, 2, 3)
        probabilities = (0.08, 0.12, 0.15, 0.30, 0.15, 0.12, 0.08)
    label_probability = 0.70 if split == "train" else 0.50
    manifest.update(
        moving_appearance_mode=(
            _v2_rng(seed, "moving_appearance_mode").random(count) < label_probability
        ).astype(np.uint8),
        label_style_seed=_v2_rng(seed, "label_style_seed").integers(
            0, np.iinfo(np.uint64).max, count, dtype=np.uint64, endpoint=True
        ),
        label_hierarchy_depth=_v2_rng(seed, "label_hierarchy_depth").integers(
            4, 8, count, dtype=np.int8
        ),
        label_blur_sigma_px=_v2_rng(seed, "label_blur_sigma_px").uniform(
            0.35, 1.5, count
        ).astype(np.float32),
        mask_offset_px=_v2_rng(seed, "mask_offset_px").choice(
            np.asarray(offsets, dtype=np.int8), count, p=probabilities
        ).astype(np.int8),
    )
    manifest["manifest_sha256"] = _payload_sha256(manifest)
    return manifest


class SyntheticRegistrationGenerator:
    def __init__(
        self,
        atlas_folder: str | Path,
        device: str | torch.device = "cuda",
    ):
        self.atlas_folder = Path(atlas_folder)
        average_path = self.atlas_folder / "average_template_25.nrrd"
        annotation_path = self.atlas_folder / "annotation_25.nrrd"
        average = nrrd.read(str(average_path))[0]
        annotation = nrrd.read(str(annotation_path))[0]
        if average.shape != annotation.shape or average.ndim != 3:
            raise ValueError("Allen average and annotation volumes must share AP x DV x ML shape")
        if average.shape[1] > MODEL_SHAPE[0] or average.shape[2] > MODEL_SHAPE[1]:
            raise ValueError("Allen coronal plane does not fit the 320 x 464 one-to-one canvas")
        self.device = torch.device(device)
        self.average = torch.from_numpy(average.astype(np.float32) / max(float(average.max()), 1.0)).to(self.device)[None, None]
        self.annotation = torch.from_numpy(annotation.astype(np.int32)).to(self.device)
        self.volume_shape = average.shape
        self.pad_y = (MODEL_SHAPE[0] - average.shape[1]) // 2
        self.pad_x = (MODEL_SHAPE[1] - average.shape[2]) // 2
        query_path = self.atlas_folder / "query.csv"
        query_sha256 = _sha256(query_path)
        if query_sha256 != QUERY_SHA256:
            raise ValueError("Allen query.csv differs from the v2 generator contract")
        query_ids, query_ancestors = _query_hierarchy(query_path)
        annotation_ids = np.unique(annotation).astype(np.int64)
        positions = np.searchsorted(query_ids, annotation_ids)
        matched = positions < len(query_ids)
        matched[matched] &= query_ids[positions[matched]] == annotation_ids[matched]
        if bool(((annotation_ids != 0) & ~matched).any()):
            raise ValueError("Every nonzero Allen annotation ID must match query.csv exactly")
        self.query_ids = torch.from_numpy(query_ids).to(self.device)
        self.query_ancestor_values = query_ancestors
        split_contract = {
            "block_width": AP_BLOCK_WIDTH,
            "pattern": AP_SPLIT_PATTERN,
            "ap_range_um": [AP_MIN_UM, AP_MAX_UM],
            "bregma_ap_index": BREGMA_AP_INDEX,
            "plane_sampling": PLANE_SAMPLING_CONTRACT,
        }
        self.contract = {
            "generator_version": GENERATOR_VERSION,
            "model_shape": MODEL_SHAPE,
            "pixel_spacing_um": VOXEL_UM,
            "plane_sampling": PLANE_SAMPLING_CONTRACT,
            "average_template_sha256": _sha256(average_path),
            "annotation_sha256": _sha256(annotation_path),
            "split_contract_sha256": _payload_sha256(split_contract),
            "profile": "v2",
            "semantics_version": V2_SEMANTICS_VERSION,
            "generator_source_sha256": _source_sha256(Path(__file__).resolve()),
            "preprocessing_contract": PREPROCESSING_CONTRACT_V2,
            "mask_contract_sha256": MASK_CONTRACT_SHA256,
            "appearance_contract_sha256": hashlib.sha256(
                V2_APPEARANCE_CONTRACT.encode()
            ).hexdigest(),
            "query_sha256": query_sha256,
        }
        self.contract["contract_sha256"] = _payload_sha256(self.contract)

    def make_manifest(
        self, count: int, split: str, seed: int, stratum: str, *, _final_capability=None
    ) -> dict[str, np.ndarray | str | int]:
        return make_registration_manifest(
            self.contract, count, split, seed, stratum,
            _final_capability=_final_capability,
        )

    @staticmethod
    def _tensor(manifest: dict, name: str, device: torch.device, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(manifest[name], device=device, dtype=dtype)

    def render_planes(
        self, ap_index: torch.Tensor, tilt_lr_deg: torch.Tensor, tilt_dv_deg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(ap_index)
        native_height, native_width = self.volume_shape[1:]
        dv, ml = torch.meshgrid(
            torch.arange(native_height, device=self.device, dtype=torch.float32),
            torch.arange(native_width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        ap = ap_index[:, None, None]
        ap = ap + torch.tan(torch.deg2rad(tilt_lr_deg))[:, None, None] * (ml - (native_width - 1) / 2.0)
        ap = ap + torch.tan(torch.deg2rad(tilt_dv_deg))[:, None, None] * (dv - (native_height - 1) / 2.0)
        grid = torch.stack(
            (
                ml[None].expand(batch, -1, -1) / (native_width - 1) * 2.0 - 1.0,
                dv[None].expand(batch, -1, -1) / (native_height - 1) * 2.0 - 1.0,
                ap / (self.volume_shape[0] - 1) * 2.0 - 1.0,
            ),
            dim=-1,
        )[:, None]
        image = F.grid_sample(
            self.average.expand(batch, -1, -1, -1, -1), grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )[:, :, 0]
        ap_nearest = ap.round().long()
        dv_index = dv.long()[None].expand(batch, -1, -1)
        ml_index = ml.long()[None].expand(batch, -1, -1)
        valid = (ap_nearest >= 0) & (ap_nearest < self.volume_shape[0])
        labels = self.annotation[
            ap_nearest.clamp(0, self.volume_shape[0] - 1), dv_index, ml_index
        ].long()
        labels = torch.where(valid, labels, 0)[:, None]
        mask = labels > 0
        normalized = torch.zeros_like(image)
        for item in range(batch):
            values = image[item, 0][mask[item, 0]]
            low, high = torch.quantile(values, torch.tensor((0.005, 0.995), device=self.device))
            normalized[item] = ((image[item] - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
        pad = (
            self.pad_x, MODEL_SHAPE[1] - native_width - self.pad_x,
            self.pad_y, MODEL_SHAPE[0] - native_height - self.pad_y,
        )
        return (
            F.pad(normalized * mask, pad),
            F.pad(mask, pad),
            F.pad(labels, pad),
        )

    def _velocity(self, manifest: dict, fixed_mask: torch.Tensor) -> torch.Tensor:
        batch = fixed_mask.shape[0]
        height, width = MODEL_SHAPE
        grid = _identity_grid(batch, height, width, self.device)
        x, y = grid[:, 0], grid[:, 1]
        velocity = torch.zeros_like(grid)
        counts = self._tensor(manifest, "component_count", self.device, torch.int64)
        kinds = self._tensor(manifest, "component_kind", self.device, torch.int64)
        centers = self._tensor(manifest, "component_center_xy", self.device)
        sigma = self._tensor(manifest, "component_sigma_xy", self.device)
        amplitude = self._tensor(manifest, "component_amplitude", self.device)
        anisotropy = self._tensor(manifest, "component_anisotropy", self.device)
        for component in range(COMPONENTS):
            enabled = (counts > component).float()[:, None, None]
            dx = (x - centers[:, component, 0, None, None]) / sigma[:, component, 0, None, None]
            dy = (y - centers[:, component, 1, None, None]) / sigma[:, component, 1, None, None]
            envelope = torch.exp(-0.5 * (dx.square() + dy.square()))
            strength = 2.3 * amplitude[:, component, None, None] * envelope * enabled
            radial = torch.stack((strength * dx, strength * dy), dim=1)
            stretch = torch.stack(
                (strength * dx, strength * dy * anisotropy[:, component, None, None]), dim=1
            )
            swirl = torch.stack((-strength * dy, strength * dx), dim=1) * 0.75
            kind = kinds[:, component, None, None, None]
            velocity += torch.where(kind == 0, radial, torch.where(kind == 1, stretch, swirl))
        tissue = fixed_mask.float()
        support = F.avg_pool2d(tissue, 25, stride=1, padding=12) * tissue
        velocity = _remove_tissue_affine(velocity, support, tissue)
        cell_mask = (
            fixed_mask[:, :, :-1, :-1] | fixed_mask[:, :, :-1, 1:]
            | fixed_mask[:, :, 1:, :-1] | fixed_mask[:, :, 1:, 1:]
        )[:, 0]
        minimum = torch.full((batch,), -torch.inf, device=self.device)
        for _ in range(24):
            forward = integrate_velocity(velocity)
            inverse = integrate_velocity(-velocity)
            minimum = torch.minimum(
                jacobian_determinant(forward).masked_fill(~cell_mask, torch.inf).amin((1, 2)),
                jacobian_determinant(inverse).masked_fill(~cell_mask, torch.inf).amin((1, 2)),
            )
            if bool((minimum >= 0.20).all()):
                break
            velocity = velocity * torch.where(minimum < 0.20, 0.72, 1.0)[:, None, None, None]
        if bool((minimum < 0.20).any()):
            raise RuntimeError("synthetic deformation could not satisfy the positive-Jacobian contract")
        return velocity

    def _label_conditioned_grayscale(
        self,
        clean: torch.Tensor,
        labels: torch.Tensor,
        tissue: torch.Tensor,
        manifest: dict,
    ) -> torch.Tensor:
        positions = torch.searchsorted(self.query_ids, labels)
        safe = positions.clamp_max(len(self.query_ids) - 1)
        matched = self.query_ids[safe] == labels
        if bool(((labels != 0) & ~matched).any()):
            raise ValueError("Every nonzero rendered annotation ID must match query.csv exactly")
        styled = torch.zeros_like(clean)
        for item in range(clean.shape[0]):
            depth = int(manifest["label_hierarchy_depth"][item])
            ancestor_ids = self.query_ancestor_values[depth - 4]
            style_seed = np.uint64(manifest["label_style_seed"][item])
            tones = _label_tones(ancestor_ids, style_seed)
            sigma_tones = _label_tones(
                ancestor_ids, style_seed ^ np.uint64(0xD1B54A32D192ED03)
            )
            tone_table = torch.from_numpy(tones).to(self.device)
            sigma_table = torch.from_numpy(
                0.015 + 0.085 * np.clip((sigma_tones - 0.08) / 0.84, 0.0, 1.0)
            ).to(self.device)
            regions = tone_table[safe[item]]
            generator = torch.Generator(device=self.device)
            generator.manual_seed(
                int(style_seed ^ np.uint64(0xA24BAED4963EE407))
            )
            regions = regions + torch.randn(
                regions.shape, device=self.device, generator=generator
            ) * sigma_table[safe[item]]
            sigma = float(manifest["label_blur_sigma_px"][item])
            radius = max(1, int(math.ceil(3.0 * sigma)))
            axis = torch.arange(-radius, radius + 1, device=self.device, dtype=torch.float32)
            kernel = torch.exp(-0.5 * (axis / sigma).square())
            kernel /= kernel.sum()
            regions = F.conv2d(
                F.pad(regions, (radius, radius, 0, 0), mode="replicate"),
                kernel.reshape(1, 1, 1, -1),
            )
            regions = F.conv2d(
                F.pad(regions, (0, 0, radius, radius), mode="replicate"),
                kernel.reshape(1, 1, -1, 1),
            )
            styled[item] = (0.28 * clean[item] + 0.72 * regions) * tissue[item]
        return styled.clamp(0.0, 1.0)

    def _damage_masks(
        self, tissue: torch.Tensor, manifest: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = tissue.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=self.device),
            torch.linspace(-1.0, 1.0, width, device=self.device), indexing="ij",
        )
        x, y = x[None, None], y[None, None]
        tear = self._tensor(manifest, "tear_parameters", self.device)
        signed = y - tear[:, 0, None, None, None] - tear[:, 1, None, None, None] * x - tear[:, 2, None, None, None] * x.square()
        tear_mask = signed.abs() < tear[:, 3, None, None, None]
        tear_mask &= self._tensor(manifest, "tear_enabled", self.device, torch.bool)[:, None, None, None]
        missing = self._tensor(manifest, "missing_parameters", self.device)
        edge_x = 0.90 * torch.cos(missing[:, 0, None, None, None])
        edge_y = 0.90 * torch.sin(missing[:, 0, None, None, None])
        missing_mask = ((x - edge_x) / missing[:, 1, None, None, None]).square() + ((y - edge_y) / missing[:, 2, None, None, None]).square() < 1.0
        missing_mask &= self._tensor(manifest, "missing_enabled", self.device, torch.bool)[:, None, None, None]
        occult = self._tensor(manifest, "occlusion_parameters", self.device)
        ox, oy = occult[:, 0, None, None, None] * 2 - 1, occult[:, 1, None, None, None] * 2 - 1
        cosine, sine = torch.cos(occult[:, 4, None, None, None]), torch.sin(occult[:, 4, None, None, None])
        rx, ry = cosine * (x - ox) - sine * (y - oy), sine * (x - ox) + cosine * (y - oy)
        occlusion_mask = (rx / occult[:, 2, None, None, None]).square() + (ry / occult[:, 3, None, None, None]).square() < 1.0
        occlusion_mask &= self._tensor(manifest, "occlusion_enabled", self.device, torch.bool)[:, None, None, None]
        return tear_mask & tissue, missing_mask & tissue, occlusion_mask & tissue

    @staticmethod
    def _native(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[..., : NATIVE_SHAPE[0], PAD_X : PAD_X + NATIVE_SHAPE[1]]

    @staticmethod
    def _padded_native(tensor: torch.Tensor) -> torch.Tensor:
        return F.pad(tensor, (PAD_X, MODEL_SHAPE[1] - NATIVE_SHAPE[1] - PAD_X))

    @staticmethod
    def _offset_mask(mask: torch.Tensor, offsets: np.ndarray) -> torch.Tensor:
        result = mask.clone()
        for item, offset in enumerate(np.asarray(offsets, dtype=np.int8)):
            value = result[item : item + 1]
            for _ in range(abs(int(offset))):
                if offset > 0:
                    value = F.max_pool2d(value.float(), 3, stride=1, padding=1) > 0.5
                elif offset < 0:
                    outside = F.pad((~value).float(), (1, 1, 1, 1), value=1.0)
                    value = ~(F.max_pool2d(outside, 3, stride=1) > 0.5)
            result[item : item + 1] = value
        return result

    def _appearance_v2(
        self, clean: torch.Tensor, tissue: torch.Tensor, manifest: dict
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        image, masks = self._appearance(clean, tissue, manifest)
        image = self._native(image)
        masks = {name: self._native(value) for name, value in masks.items()}
        brush = masks["brush"]
        model_mask = self._offset_mask(brush, manifest["mask_offset_px"])
        raw_uint8 = torch.round(image * 255.0).to(torch.uint8)
        alpha = cosine_mask_feather(
            model_mask,
            dilate=lambda value: F.max_pool2d(
                value.float(), 3, stride=1, padding=1
            ) > 0.5,
            zeros_like=lambda value: torch.zeros_like(value, dtype=torch.float32),
            where=torch.where,
        )
        return self._padded_native(raw_uint8.float() / 255.0 * alpha), self._padded_native(raw_uint8), {
            "damage": self._padded_native(masks["damage"]),
            "visible": self._padded_native(masks["visible"]),
            "brush": self._padded_native(brush),
            "model": self._padded_native(model_mask),
        }

    def _appearance(
        self, clean: torch.Tensor, tissue: torch.Tensor, manifest: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, _, height, width = clean.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=self.device),
            torch.linspace(-1.0, 1.0, width, device=self.device), indexing="ij",
        )
        x, y = x[None, None], y[None, None]
        gamma = self._tensor(manifest, "gamma", self.device)[:, None, None, None]
        gain = self._tensor(manifest, "gain", self.device)[:, None, None, None]
        offset = self._tensor(manifest, "offset", self.device)[:, None, None, None]
        image = clean.clamp(0.0, 1.0).pow(gamma) * gain + offset
        invert = self._tensor(manifest, "invert", self.device, torch.bool)[:, None, None, None]
        image = torch.where(invert, 1.0 - image, image)
        bias = self._tensor(manifest, "bias", self.device)[:, :, None, None]
        field = bias[:, 0:1] * x + bias[:, 1:2] * y + bias[:, 2:3] * x * y + bias[:, 3:4] * x.square() + bias[:, 4:5] * y.square()
        image = image * torch.exp(field)

        angle = torch.deg2rad(self._tensor(manifest, "tile_angle_deg", self.device))[:, None, None, None]
        rotated_x = torch.cos(angle) * x - torch.sin(angle) * y
        rotated_y = torch.sin(angle) * x + torch.cos(angle) * y
        periods = self._tensor(manifest, "tile_period_xy", self.device)[:, :, None, None]
        fraction_x = torch.remainder((rotated_x + 1.0) * width / periods[:, 0:1], 1.0) - 0.5
        fraction_y = torch.remainder((rotated_y + 1.0) * height / periods[:, 1:2], 1.0) - 0.5
        tile_strength = self._tensor(manifest, "tile_strength", self.device)[:, None, None, None]
        image *= 1.0 - tile_strength * (fraction_x.square() + fraction_y.square())
        image += tile_strength * 0.20 * ((fraction_x.abs() > 0.47) | (fraction_y.abs() > 0.47))

        centers = self._tensor(manifest, "blowout_center_xy", self.device)
        radius = self._tensor(manifest, "blowout_radius", self.device)[:, None, None, None]
        blowout = torch.exp(-((x - (centers[:, 0, None, None, None] * 2 - 1)) ** 2 + (y - (centers[:, 1, None, None, None] * 2 - 1)) ** 2) / radius.square().clamp_min(1e-5))
        image += blowout * self._tensor(manifest, "blowout_strength", self.device)[:, None, None, None]

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(manifest["seed"]) ^ 0x5A17C0DE)
        noise = torch.randn(image.shape, device=self.device, generator=generator)
        image += noise * self._tensor(manifest, "noise", self.device)[:, None, None, None]
        specks = torch.rand(image.shape, device=self.device, generator=generator) < self._tensor(
            manifest, "speck_density", self.device
        )[:, None, None, None]
        image = torch.maximum(image, F.max_pool2d(specks.float(), 3, stride=1, padding=1))
        blurred = F.avg_pool2d(image, 5, stride=1, padding=2)
        blur = self._tensor(manifest, "blur", self.device)[:, None, None, None]
        image = image * (1.0 - blur) + blurred * blur

        background = self._tensor(manifest, "background", self.device)[:, None, None, None]
        background_image = (background + noise * 0.35 * self._tensor(
            manifest, "noise", self.device
        )[:, None, None, None]).clamp(0.0, 1.0)
        image = torch.where(tissue, image, background_image)

        tear, missing, occlusion = self._damage_masks(tissue, manifest)
        damage = tear | missing | occlusion
        visible = tissue & ~damage
        image = torch.where(damage, background_image, image)
        return image.clamp(0.0, 1.0), {
            "damage": damage,
            "visible": visible,
            "brush": tissue & ~missing,
        }

    def batch(
        self, manifest: dict, *, qa: bool = False, _final_capability=None
    ) -> dict[str, torch.Tensor | str | dict]:
        if manifest.get("contract_sha256") != self.contract["contract_sha256"]:
            raise ValueError("manifest belongs to a different atlas or generator contract")
        expected_hash = manifest.get("manifest_sha256")
        unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if expected_hash != _payload_sha256(unhashed):
            raise ValueError("manifest hash does not match its parameters")
        split_pool = split_ap_indices(
            manifest.get("split"), _final_capability=_final_capability
        )
        ap_indices = np.asarray(manifest.get("ap_index"))
        if ap_indices.ndim != 1 or not np.isin(ap_indices, split_pool).all():
            raise ValueError("manifest AP positions are outside its declared split")
        ap = self._tensor(manifest, "ap_index", self.device)
        fixed, fixed_mask, fixed_labels = self.render_planes(
            ap,
            self._tensor(manifest, "tilt_lr_deg", self.device),
            self._tensor(manifest, "tilt_dv_deg", self.device),
        )
        velocity = self._velocity(manifest, fixed_mask)
        local_forward = integrate_velocity(velocity)
        local_inverse = integrate_velocity(-velocity)
        similarity_inverse, similarity_h = _similarity_inverse_and_homography(
            self._tensor(manifest, "rotation_deg", self.device),
            self._tensor(manifest, "scale", self.device),
            self._tensor(manifest, "translation_xy", self.device),
            MODEL_SHAPE,
        )
        fixed_to_moving = (
            torch.einsum("bij,bjhw->bihw", similarity_h[:, :2, :2], local_forward)
            + similarity_h[:, :2, 2, None, None]
        )
        identity = _identity_grid(
            fixed.shape[0], MODEL_SHAPE[0], MODEL_SHAPE[1], self.device
        )
        moving_to_fixed = similarity_inverse + _sample_field(
            local_inverse - identity, similarity_inverse
        )
        moving_clean = _sample(fixed, moving_to_fixed)
        moving_tissue_mask = _sample(fixed_mask.float(), moving_to_fixed, "nearest") > 0.5
        moving_labels = _sample_labels(fixed_labels, moving_to_fixed)
        moving_tissue_mask = self._padded_native(self._native(moving_tissue_mask))
        label_clean = self._label_conditioned_grayscale(
            moving_clean, moving_labels, moving_tissue_mask, manifest
        )
        label_mode = self._tensor(
            manifest, "moving_appearance_mode", self.device, torch.bool
        )[:, None, None, None]
        appearance_clean = torch.where(label_mode, label_clean, moving_clean)
        moving, moving_raw_uint8, masks = self._appearance_v2(
            appearance_clean, moving_tissue_mask, manifest
        )
        fixed_visible_mask = fixed_mask & (
            _sample(masks["visible"].float(), fixed_to_moving, "nearest") > 0.5
        )
        result = {
            "fixed": fixed.float(),
            "moving": moving.float(),
            "fixed_mask": fixed_mask,
            "moving_tissue_mask": moving_tissue_mask,
            "moving_damage_mask": masks["damage"],
            "moving_visible_mask": masks["visible"],
            "moving_model_mask": masks["model"],
            "fixed_damage_mask": fixed_mask & ~fixed_visible_mask,
            "fixed_visible_mask": fixed_visible_mask,
            "fixed_labels": fixed_labels.long(),
            "moving_labels": moving_labels.long(),
            "fixed_to_moving": fixed_to_moving.float(),
            "moving_to_fixed": moving_to_fixed.float(),
            "local_velocity": velocity.float(),
            "similarity_h": similarity_h.float(),
            "manifest_sha256": str(expected_hash),
            "contract": dict(self.contract),
        }
        if qa:
            result.update(
                moving_clean=moving_clean.float(),
                moving_raw_uint8=moving_raw_uint8,
                moving_appearance_clean=appearance_clean.float(),
                moving_brush_mask=masks["brush"],
            )
        return result

    def generate(
        self, count: int, split: str, seed: int, stratum: str, *, qa: bool = False,
        _final_capability=None,
    ) -> dict[str, torch.Tensor | str | dict]:
        return self.batch(
            self.make_manifest(
                count, split, seed, stratum,
                _final_capability=_final_capability,
            ),
            qa=qa, _final_capability=_final_capability,
        )


def save_qa_montage(
    pair: dict[str, torch.Tensor | str | dict], path: str | Path, max_items: int = 4
) -> Path:
    """Save fixed/deformed/damaged/recovered panels for visual generator review."""
    fixed = pair["fixed"]
    moving_clean = pair["moving_clean"]
    moving = pair["moving"]
    fixed_to_moving = pair["fixed_to_moving"]
    recovered = _sample(moving_clean, fixed_to_moving)
    titles = (
        "fixed atlas", "deformed template", "appearance base", "raw artifacts",
        "model input", "damage mask", "model mask", "recovered geometry",
    )
    tensors = (
        fixed,
        moving_clean,
        pair["moving_appearance_clean"],
        pair["moving_raw_uint8"].float() / 255.0,
        moving,
        pair["moving_damage_mask"].float(),
        pair["moving_model_mask"].float(),
        recovered,
    )
    count = min(int(fixed.shape[0]), int(max_items))
    height, width = fixed.shape[-2:]
    header = 22
    montage = Image.new("L", (len(titles) * width, count * (height + header)), 0)
    draw = ImageDraw.Draw(montage)
    for row in range(count):
        for column, (title, tensor) in enumerate(zip(titles, tensors)):
            array = (tensor[row, 0].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            panel = Image.fromarray(array, mode="L")
            left, top = column * width, row * (height + header)
            montage.paste(panel, (left, top + header))
            draw.text((left + 5, top + 4), title, fill=230)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    montage.save(destination)
    return destination
