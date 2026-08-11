from __future__ import annotations

import csv
from pathlib import Path

import cv2
import nrrd
import numpy as np
import torch
import torch.nn.functional as F


IMAGE_SIZE = 299
CANVAS_SIZE = 448
VOXEL_UM = 25.0
BREGMA_AP_INDEX = 216.0
AP_MIN_UM = -4500.0
AP_MAX_UM = 500.0
TARGET_CENTER = np.asarray([-2000.0, 0.0, 0.0], dtype=np.float32)
TARGET_SCALE = np.asarray([2500.0, 20.0, 20.0], dtype=np.float32)
COHORT_NAMES = np.asarray(["clean", "mild", "moderate", "severe"])
COARSE_ANATOMY_CLASSES = (
    "exterior",
    "cortex",
    "hippocampal",
    "nuclei",
    "thalamus",
    "hypothalamus",
    "midbrain_hindbrain",
    "cerebellum",
    "fiber_tracts_cavities",
)
GEOMETRY_MANIFEST_KEYS = (
    "ap_um",
    "ap_index",
    "tilt_lr_deg",
    "tilt_dv_deg",
    "rotation_deg",
    "orientation_residual_deg",
    "scale",
    "translation_xy",
    "cohort",
    "warp",
    "warp_amplitude",
    "warp_scale_xy",
    "warp_shear_xy",
    "warp_bulge_center",
    "warp_bulge_strength",
    "warp_seed",
    "occlusion_type",
    "damage_mode",
    "occlusion_seed",
)
APPEARANCE_MANIFEST_KEYS = (
    "flaw_mask",
    "anatomy_seed",
    "anatomy_mix",
    "anatomy_channel_mix",
    "anatomy_edge_strength",
    "contrast_gain",
    "contrast_gamma",
    "contrast_offset",
    "contrast_invert",
    "exposure_center",
    "exposure_width",
    "exposure_strength",
    "exposure_sign",
    "tile_period",
    "tile_period_xy",
    "tile_phase",
    "tile_angle_deg",
    "tile_seed",
    "tile_strength",
    "tile_power",
    "tile_seam_width",
    "tile_seam_strength",
    "bias_coefficients",
    "local_gamma_xy",
    "sensor_enabled",
    "sensor_seed",
    "sensor_noise",
    "speck_density",
    "speck_strength",
    "streak_angle_deg",
    "streak_offset",
    "streak_width",
    "blowout_center",
    "blowout_radius",
    "blowout_strength",
    "background_level",
    "background_texture",
    "background_seed",
)


def _stratified_uniform(rng: np.random.Generator, count: int, low: float, high: float) -> np.ndarray:
    values = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(values)
    return low + (high - low) * values


def _coarse_anatomy_volume(annotation: np.ndarray, structure_table: Path) -> np.ndarray:
    roots = ((1089, 2), (688, 1), (623, 3), (549, 4), (1097, 5), (313, 6), (1065, 6), (512, 7),
             (1009, 8), (73, 8), (1024, 8))
    structure_classes = {0: 0, 997: 3}
    with structure_table.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            structure_id = int(row["id"])
            path = row["structure_id_path"]
            structure_classes[structure_id] = next(
                (class_index for root, class_index in roots if f"/{root}/" in path),
                3,
            )
    ids = np.asarray(sorted(structure_classes), dtype=np.uint32)
    classes = np.asarray([structure_classes[int(structure_id)] for structure_id in ids], dtype=np.uint8)
    coarse = np.empty(annotation.shape, dtype=np.uint8)
    for start in range(0, annotation.shape[0], 32):
        slab = annotation[start : start + 32]
        coarse[start : start + 32] = classes[np.searchsorted(ids, slab)]
    return coarse


def make_manifest(count: int, split: str, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    ap_fraction = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(ap_fraction)
    ap_um = AP_MAX_UM + (AP_MIN_UM - AP_MAX_UM) * ap_fraction
    tilt = np.column_stack(
        (
            _stratified_uniform(rng, count, -35.0, 35.0),
            _stratified_uniform(rng, count, -35.0, 35.0),
        )
    )

    severity_draw = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(severity_draw)
    cohort = np.searchsorted(np.asarray([0.10, 0.55, 0.90]), severity_draw).astype(np.uint8)
    optical = cohort > 0
    flaw_count = cohort.clip(0, 3)
    flaw_priority = rng.random((count, 3))
    flaw_rank = np.argsort(np.argsort(flaw_priority, axis=1), axis=1)
    flaw_mask = (flaw_rank < flaw_count[:, None]) & optical[:, None]

    warp = rng.random(count) < np.asarray([0.0, 0.48, 0.81, 1.0])[cohort]
    occluded = rng.random(count) < np.asarray([0.0, 0.24, 0.59, 0.85])[cohort]
    strip = occluded & (rng.random(count) < 0.10)
    occlusion_type = np.zeros(count, dtype=np.uint8)
    occlusion_type[strip] = 1
    occlusion_type[occluded & ~strip] = 2

    severity_scale = np.asarray([0.0, 0.45, 0.72, 1.0], dtype=np.float32)[cohort]
    tile_period = rng.integers(28, 104, (count, 2), dtype=np.int32)
    tile_period = np.maximum(tile_period, 10)

    return {
        "ap_um": ap_um.astype(np.float32),
        "ap_index": (BREGMA_AP_INDEX - ap_um / VOXEL_UM).astype(np.float32),
        "tilt_lr_deg": tilt[:, 0].astype(np.float32),
        "tilt_dv_deg": tilt[:, 1].astype(np.float32),
        "rotation_deg": rng.uniform(-180.0, 180.0, count).astype(np.float32),
        "orientation_residual_deg": rng.uniform(-12.0, 12.0, count).astype(np.float32),
        "scale": rng.uniform(0.5, 1.5, count).astype(np.float32),
        "translation_xy": rng.uniform(-0.025, 0.025, (count, 2)).astype(np.float32),
        "cohort": cohort,
        "warp": warp,
        "warp_amplitude": (rng.triangular(3.0, 10.0, 22.0, count) * (0.6 + 0.55 * severity_scale)).astype(np.float32),
        "warp_scale_xy": (1.0 + rng.uniform(-0.16, 0.16, (count, 2)) * (0.35 + 0.65 * severity_scale[:, None])).astype(np.float32),
        "warp_shear_xy": rng.uniform(-0.035, 0.035, (count, 2)).astype(np.float32),
        "warp_bulge_center": rng.uniform(-0.65, 0.65, (count, 2)).astype(np.float32),
        "warp_bulge_strength": (rng.uniform(-0.055, 0.055, count) * severity_scale).astype(np.float32),
        "warp_seed": rng.integers(0, 2**31 - 1, count, dtype=np.int64),
        "flaw_mask": flaw_mask,
        "anatomy_seed": rng.integers(0, 2**31 - 1, count, dtype=np.int64),
        "anatomy_mix": rng.uniform(0.02, 0.72, count).astype(np.float32) * np.asarray([0.0, 0.35, 0.68, 1.0], dtype=np.float32)[cohort],
        "anatomy_channel_mix": rng.uniform(0.0, 1.0, count).astype(np.float32),
        "anatomy_edge_strength": rng.uniform(-0.28, 0.42, count).astype(np.float32) * np.asarray([0.0, 0.50, 0.75, 1.0], dtype=np.float32)[cohort],
        "contrast_gain": np.exp(rng.uniform(np.log(0.45), np.log(1.9), count)).astype(np.float32),
        "contrast_gamma": np.exp(rng.uniform(np.log(0.55), np.log(1.8), count)).astype(np.float32),
        "contrast_offset": rng.uniform(-0.18, 0.18, count).astype(np.float32),
        "contrast_invert": (rng.random(count) < 0.50),
        "exposure_center": rng.uniform(0.18, 0.82, count).astype(np.float32),
        "exposure_width": rng.uniform(0.08, 0.35, count).astype(np.float32),
        "exposure_strength": rng.uniform(0.35, 0.95, count).astype(np.float32),
        "exposure_sign": rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), count),
        "tile_period": tile_period[:, 0],
        "tile_period_xy": tile_period,
        "tile_phase": rng.uniform(0.0, 1.0, (count, 2)).astype(np.float32),
        "tile_angle_deg": rng.uniform(-38.0, 38.0, count).astype(np.float32),
        "tile_seed": rng.integers(0, 2**31 - 1, count, dtype=np.int64),
        "tile_strength": (rng.uniform(0.08, 0.52, count) * (0.35 + 0.65 * severity_scale)).astype(np.float32),
        "tile_power": rng.uniform(1.0, 3.5, count).astype(np.float32),
        "tile_seam_width": rng.uniform(0.008, 0.055, count).astype(np.float32),
        "tile_seam_strength": (rng.uniform(-0.15, 0.22, count) * (0.20 + 0.80 * severity_scale)).astype(np.float32),
        "bias_coefficients": (rng.normal(0.0, 0.32, (count, 6)) * severity_scale[:, None]).astype(np.float32),
        "local_gamma_xy": np.exp(rng.uniform(np.log(0.55), np.log(1.85), (count, 2))).astype(np.float32),
        "sensor_enabled": (optical & (rng.random(count) < (0.20 + 0.65 * severity_scale))),
        "sensor_seed": rng.integers(0, 2**31 - 1, count, dtype=np.int64),
        "sensor_noise": (rng.uniform(0.006, 0.075, count) * (0.4 + 0.6 * severity_scale)).astype(np.float32),
        "speck_density": rng.uniform(0.00008, 0.0018, count).astype(np.float32),
        "speck_strength": rng.uniform(0.35, 1.0, count).astype(np.float32),
        "streak_angle_deg": rng.uniform(-180.0, 180.0, count).astype(np.float32),
        "streak_offset": rng.uniform(-0.7, 0.7, count).astype(np.float32),
        "streak_width": rng.uniform(0.0015, 0.012, count).astype(np.float32),
        "blowout_center": rng.uniform(-0.75, 0.75, (count, 2)).astype(np.float32),
        "blowout_radius": rng.uniform(0.025, 0.18, count).astype(np.float32),
        "blowout_strength": rng.uniform(0.25, 1.0, count).astype(np.float32),
        "occlusion_type": occlusion_type,
        "damage_mode": rng.choice(np.asarray([0, 1, 2], dtype=np.uint8), count, p=[0.55, 0.28, 0.17]),
        "occlusion_seed": rng.integers(0, 2**31 - 1, count, dtype=np.int64),
        "background_level": rng.uniform(0.005, 0.22, count).astype(np.float32),
        "background_texture": rng.uniform(0.005, 0.08, count).astype(np.float32),
        "background_seed": rng.integers(0, 2**31 - 1, count, dtype=np.int64),
    }


def paired_appearance_manifest(base: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    count = len(base["ap_um"])
    donor = make_manifest(max(64, count), "train", seed)
    selector = np.random.default_rng(seed ^ 0x5EED5EED)
    donor_indices = np.empty(count, dtype=np.int64)
    for cohort in range(4):
        selected = np.flatnonzero(base["cohort"] == cohort)
        pool = np.flatnonzero(donor["cohort"] == cohort)
        donor_indices[selected] = selector.choice(pool, len(selected), replace=True)
    paired = {key: np.array(value, copy=True) for key, value in base.items()}
    for key in APPEARANCE_MANIFEST_KEYS:
        paired[key] = np.array(donor[key][donor_indices], copy=True)
    return paired


def save_manifest(path: Path, manifest: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **manifest)


def load_manifest(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {name: values[name] for name in values.files}


class SyntheticAtlas:
    def __init__(self, atlas_folder: Path, device: str = "cuda"):
        self.atlas_folder = Path(atlas_folder).resolve()
        average = nrrd.read(str(atlas_folder / "average_template_25.nrrd"))[0]
        annotation = nrrd.read(str(atlas_folder / "annotation_25.nrrd"))[0]
        coarse_anatomy = _coarse_anatomy_volume(annotation, atlas_folder / "query.csv")
        label_key = annotation.astype(np.uint32, copy=True)
        label_key ^= label_key >> np.uint32(16)
        label_key *= np.uint32(0x7FEB352D)
        label_key ^= label_key >> np.uint32(15)
        label_key *= np.uint32(0x846CA68B)
        label_key ^= label_key >> np.uint32(16)
        label_key = (label_key & np.uint32(0x00FFFFFF)).astype(np.float32) / float(0x00FFFFFF)
        label_key[annotation == 0] = 0.0
        self.device = torch.device(device)
        self.shape = average.shape
        self.volume = torch.from_numpy(average.astype(np.float32) / float(average.max())).to(self.device)[None, None]
        self.mask_volume = torch.from_numpy((annotation > 0).astype(np.float32)).to(self.device)[None, None]
        self.label_volume = torch.from_numpy(label_key).to(self.device)[None, None]
        self.coarse_volume = torch.from_numpy(coarse_anatomy.astype(np.float32)).to(self.device)[None, None]
        axis = torch.linspace(-1.0, 1.0, IMAGE_SIZE, device=self.device)
        self.yy, self.xx = torch.meshgrid(axis, axis, indexing="ij")
        canvas_axis = torch.linspace(-1.0, 1.0, CANVAS_SIZE, device=self.device)
        self.canvas_y, self.canvas_x = torch.meshgrid(canvas_axis, canvas_axis, indexing="ij")
        dv = torch.arange(self.shape[1], device=self.device, dtype=torch.float32)
        ml = torch.arange(self.shape[2], device=self.device, dtype=torch.float32)
        self.native_dv, self.native_ml = torch.meshgrid(dv, ml, indexing="ij")
        pixels = torch.arange(IMAGE_SIZE, device=self.device, dtype=torch.float32)
        self.pixel_y, self.pixel_x = torch.meshgrid(pixels, pixels, indexing="ij")

    def _tensor(self, manifest: dict[str, np.ndarray], name: str, indices: slice, dtype=torch.float32):
        return torch.as_tensor(manifest[name][indices], device=self.device, dtype=dtype)

    def _elastic_flow(
        self,
        seeds: torch.Tensor,
        amplitudes: torch.Tensor,
        enabled: torch.Tensor,
        scale_xy: torch.Tensor,
        shear_xy: torch.Tensor,
        bulge_center: torch.Tensor,
        bulge_strength: torch.Tensor,
    ) -> torch.Tensor:
        cell = torch.arange(72, device=self.device, dtype=torch.float32)[None]
        seed = seeds.float()[:, None]
        noise = torch.remainder(torch.sin(seed * 0.000173 + cell * 12.9898) * 43758.5453, 1.0) * 2.0 - 1.0
        flow = noise.reshape(-1, 2, 6, 6)
        flow = F.interpolate(flow, (IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=True)
        flow /= flow.flatten(2).std(dim=2, keepdim=True).clamp_min(1e-4)[..., None]
        flow *= amplitudes[:, None, None, None] * enabled[:, None, None, None]
        x = self.xx[None]
        y = self.yy[None]
        affine_x = (scale_xy[:, 0, None, None] - 1.0) * x + shear_xy[:, 0, None, None] * y
        affine_y = (scale_xy[:, 1, None, None] - 1.0) * y + shear_xy[:, 1, None, None] * x
        dx = x - bulge_center[:, 0, None, None]
        dy = y - bulge_center[:, 1, None, None]
        radial = torch.exp(-(dx.square() + dy.square()) / 0.28)
        flow[:, 0] += (affine_x + bulge_strength[:, None, None] * radial * dx) * (IMAGE_SIZE - 1.0) / 2.0 * enabled[:, None, None]
        flow[:, 1] += (affine_y + bulge_strength[:, None, None] * radial * dy) * (IMAGE_SIZE - 1.0) / 2.0 * enabled[:, None, None]
        for _ in range(6):
            ux_y, ux_x = torch.gradient(flow[:, 0], dim=(1, 2))
            uy_y, uy_x = torch.gradient(flow[:, 1], dim=(1, 2))
            determinant = (1.0 + ux_x) * (1.0 + uy_y) - ux_y * uy_x
            shrink = torch.where(determinant.amin(dim=(1, 2)) < 0.25, 0.72, 1.0)
            flow *= shrink[:, None, None, None]
        return flow

    def _render(
        self,
        manifest: dict[str, np.ndarray],
        indices: slice,
        return_anatomy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = len(manifest["ap_um"][indices])
        ap_center = self._tensor(manifest, "ap_index", indices)[:, None, None]
        tilt_lr = torch.deg2rad(self._tensor(manifest, "tilt_lr_deg", indices))[:, None, None]
        tilt_dv = torch.deg2rad(self._tensor(manifest, "tilt_dv_deg", indices))[:, None, None]
        ml = self.native_ml[None].expand(count, -1, -1)
        dv = self.native_dv[None].expand(count, -1, -1)
        ap = ap_center + torch.tan(tilt_lr) * (ml - (self.shape[2] - 1.0) / 2.0)
        ap += torch.tan(tilt_dv) * (dv - (self.shape[1] - 1.0) / 2.0)
        grid = torch.stack(
            (
                ml / (self.shape[2] - 1.0) * 2.0 - 1.0,
                dv / (self.shape[1] - 1.0) * 2.0 - 1.0,
                ap / (self.shape[0] - 1.0) * 2.0 - 1.0,
            ),
            dim=-1,
        )[:, None]
        image = F.grid_sample(
            self.volume.expand(count, -1, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, :, 0]
        mask = F.grid_sample(
            self.mask_volume.expand(count, -1, -1, -1, -1),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[:, :, 0] > 0.5
        label_key = F.grid_sample(
            self.label_volume.expand(count, -1, -1, -1, -1),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[:, :, 0]
        tissue_count = mask.flatten(2).sum(2).clamp_min(1.0)
        mean = (image * mask).flatten(2).sum(2) / tissue_count
        variance = (((image - mean[:, :, None, None]) ** 2) * mask).flatten(2).sum(2) / tissue_count
        template = ((image - mean[:, :, None, None]) / variance.sqrt().clamp_min(1e-4)[:, :, None, None] / 3.2 + 0.5).clamp(0.0, 1.0)
        phase = torch.remainder(self._tensor(manifest, "anatomy_seed", indices).float(), 104729.0)[:, None, None, None]
        frequency = 5.0 + torch.remainder(phase, 17.0)
        label_a = torch.remainder(torch.sin(label_key * frequency + phase * 0.00013) * 43758.5453, 1.0)
        label_b = torch.remainder(torch.sin(label_key * (frequency + 19.0) + phase * 0.00031) * 24634.6345, 1.0)
        channel_mix = self._tensor(manifest, "anatomy_channel_mix", indices)[:, None, None, None]
        anatomy = channel_mix * label_a + (1.0 - channel_mix) * label_b
        edge_x = F.pad(torch.abs(label_key[:, :, :, 1:] - label_key[:, :, :, :-1]), (0, 1, 0, 0))
        edge_y = F.pad(torch.abs(label_key[:, :, 1:, :] - label_key[:, :, :-1, :]), (0, 0, 0, 1))
        edges = F.max_pool2d(((edge_x + edge_y) > 0.015).float(), 3, stride=1, padding=1)
        anatomy_mix = self._tensor(manifest, "anatomy_mix", indices)[:, None, None, None]
        edge_strength = self._tensor(manifest, "anatomy_edge_strength", indices)[:, None, None, None]
        image = ((1.0 - anatomy_mix) * template + anatomy_mix * anatomy + edge_strength * edges).clamp(0.0, 1.0)
        if not return_anatomy:
            return image * mask, mask
        anatomy = F.grid_sample(
            self.coarse_volume.expand(count, -1, -1, -1, -1),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[:, :, 0]
        return image * mask, mask, anatomy

    def _crop_to_brain(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        anatomy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        height, width = mask.shape[-2:]
        rows = mask[:, 0].any(dim=2)
        columns = mask[:, 0].any(dim=1)
        y0 = rows.float().argmax(dim=1)
        y1 = height - rows.flip(1).float().argmax(dim=1) - 1
        x0 = columns.float().argmax(dim=1)
        x1 = width - columns.flip(1).float().argmax(dim=1) - 1
        center_x = (x0 + x1).float() / 2.0
        center_y = (y0 + y1).float() / 2.0
        half = torch.maximum((x1 - x0).float(), (y1 - y0).float()).clamp_min(16.0) * 0.57
        sample_x = center_x[:, None, None] + self.xx[None] * half[:, None, None]
        sample_y = center_y[:, None, None] + self.yy[None] * half[:, None, None]
        grid = torch.stack(
            (sample_x / (width - 1.0) * 2.0 - 1.0, sample_y / (height - 1.0) * 2.0 - 1.0),
            dim=-1,
        )
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        mask = F.grid_sample(mask.float(), grid, mode="nearest", padding_mode="zeros", align_corners=True) > 0.5
        if anatomy is not None:
            anatomy = F.grid_sample(anatomy, grid, mode="nearest", padding_mode="zeros", align_corners=True)
        return image, mask, anatomy

    def _warp(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        flow: torch.Tensor,
        anatomy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        grid = torch.stack((self.xx, self.yy), dim=-1)[None]
        grid = grid + flow.permute(0, 2, 3, 1) * (2.0 / (IMAGE_SIZE - 1.0))
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        mask = F.grid_sample(mask.float(), grid, mode="nearest", padding_mode="zeros", align_corners=True) > 0.5
        if anatomy is not None:
            anatomy = F.grid_sample(anatomy, grid, mode="nearest", padding_mode="zeros", align_corners=True)
        return image, mask, anatomy

    def _affine_canvas(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        rotation_deg: torch.Tensor,
        scale: torch.Tensor,
        translation: torch.Tensor,
        anatomy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if image.shape[-1] != CANVAS_SIZE:
            before = (CANVAS_SIZE - IMAGE_SIZE) // 2
            after = CANVAS_SIZE - IMAGE_SIZE - before
            image = F.pad(image, (before, after, before, after))
            mask = F.pad(mask, (before, after, before, after))
            if anatomy is not None:
                anatomy = F.pad(anatomy, (before, after, before, after))
        rotation = torch.deg2rad(rotation_deg)[:, None, None]
        x = self.canvas_x[None] - translation[:, 0, None, None]
        y = self.canvas_y[None] - translation[:, 1, None, None]
        cosine, sine = torch.cos(rotation), torch.sin(rotation)
        source_x = (cosine * x + sine * y) / scale[:, None, None]
        source_y = (-sine * x + cosine * y) / scale[:, None, None]
        grid = torch.stack((source_x, source_y), dim=-1)
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        mask = F.grid_sample(mask.float(), grid, mode="nearest", padding_mode="zeros", align_corners=True) > 0.5
        if anatomy is not None:
            anatomy = F.grid_sample(anatomy, grid, mode="nearest", padding_mode="zeros", align_corners=True)
        return image, mask, anatomy

    def _mask_orientation_deg(self, mask: torch.Tensor) -> torch.Tensor:
        weights = mask[:, 0].float()
        mass = weights.sum(dim=(1, 2)).clamp_min(1.0)
        center_x = (weights * self.canvas_x).sum(dim=(1, 2)) / mass
        center_y = (weights * self.canvas_y).sum(dim=(1, 2)) / mass
        x = self.canvas_x[None] - center_x[:, None, None]
        y = self.canvas_y[None] - center_y[:, None, None]
        xx = (weights * x.square()).sum(dim=(1, 2)) / mass
        yy = (weights * y.square()).sum(dim=(1, 2)) / mass
        xy = (weights * x * y).sum(dim=(1, 2)) / mass
        return torch.rad2deg(0.5 * torch.atan2(2.0 * xy, xx - yy))

    def _optical_flaws(self, image: torch.Tensor, mask: torch.Tensor, manifest: dict[str, np.ndarray], indices: slice) -> torch.Tensor:
        flaw = self._tensor(manifest, "flaw_mask", indices, torch.bool)
        tissue_count = mask.flatten(2).sum(2).clamp_min(1.0)
        mean = (image * mask).flatten(2).sum(2) / tissue_count

        selected = flaw[:, 0, None, None, None]
        gain = self._tensor(manifest, "contrast_gain", indices)[:, None, None, None]
        gamma = self._tensor(manifest, "contrast_gamma", indices)[:, None, None, None]
        offset = self._tensor(manifest, "contrast_offset", indices)[:, None, None, None]
        contrasted = ((image - mean[:, :, None, None]) * gain + mean[:, :, None, None] + offset).clamp(0.0, 1.0)
        contrasted = contrasted.clamp_min(1e-5).pow(gamma)
        inverted = self._tensor(manifest, "contrast_invert", indices, torch.bool)[:, None, None, None]
        contrasted = torch.where(inverted, 1.0 - contrasted, contrasted)
        image = torch.where(selected, contrasted, image)

        selected = flaw[:, 1, None, None, None]
        center = self._tensor(manifest, "exposure_center", indices)[:, None, None, None]
        width = self._tensor(manifest, "exposure_width", indices)[:, None, None, None]
        strength = self._tensor(manifest, "exposure_strength", indices)[:, None, None, None]
        sign = self._tensor(manifest, "exposure_sign", indices)[:, None, None, None]
        band = torch.sigmoid((image - center + width / 2.0) / 0.025) - torch.sigmoid((image - center - width / 2.0) / 0.025)
        exposed = image + band * strength * torch.where(sign > 0, 1.0 - image, -image)
        image = torch.where(selected, exposed.clamp(0.0, 1.0), image)

        periods = self._tensor(manifest, "tile_period_xy", indices)
        phase = self._tensor(manifest, "tile_phase", indices)
        angle = torch.deg2rad(self._tensor(manifest, "tile_angle_deg", indices))[:, None, None]
        centered_x = self.pixel_x[None] - (IMAGE_SIZE - 1.0) / 2.0
        centered_y = self.pixel_y[None] - (IMAGE_SIZE - 1.0) / 2.0
        rotated_x = torch.cos(angle) * centered_x + torch.sin(angle) * centered_y
        rotated_y = -torch.sin(angle) * centered_x + torch.cos(angle) * centered_y
        tile_x = rotated_x / periods[:, 0, None, None] + phase[:, 0, None, None]
        tile_y = rotated_y / periods[:, 1, None, None] + phase[:, 1, None, None]
        tx = torch.remainder(tile_x, 1.0)
        ty = torch.remainder(tile_y, 1.0)
        edge = torch.maximum(torch.abs(tx - 0.5), torch.abs(ty - 0.5)) * 2.0
        strength = self._tensor(manifest, "tile_strength", indices)[:, None, None]
        power = self._tensor(manifest, "tile_power", indices)[:, None, None]
        vignette = 1.0 - strength * edge.pow(power)
        tile_seed = torch.remainder(self._tensor(manifest, "tile_seed", indices).float(), 65521.0)[:, None, None]
        cell = torch.floor(tile_x) * 12.9898 + torch.floor(tile_y) * 78.233 + tile_seed * 0.017
        tile_gain = 1.0 + strength * (torch.remainder(torch.sin(cell) * 43758.5453, 1.0) - 0.5) * 1.25
        tile_offset = strength * (torch.remainder(torch.sin(cell + 17.17) * 24634.6345, 1.0) - 0.5) * 0.45
        tile_gamma = torch.exp(strength * (torch.remainder(torch.sin(cell + 51.73) * 19642.3491, 1.0) - 0.5) * 1.4)
        seam_width = self._tensor(manifest, "tile_seam_width", indices)[:, None, None]
        seam = (torch.minimum(tx, 1.0 - tx) < seam_width) | (torch.minimum(ty, 1.0 - ty) < seam_width)
        seam_strength = self._tensor(manifest, "tile_seam_strength", indices)[:, None, None]
        tiled = (image * vignette[:, None] * tile_gain[:, None] + tile_offset[:, None]).clamp(1e-5, 1.0)
        tiled = tiled.pow(tile_gamma[:, None])
        tiled = (tiled + seam[:, None] * seam_strength[:, None]).clamp(0.0, 1.0)
        image = torch.where(flaw[:, 2, None, None, None], tiled, image)

        coefficients = self._tensor(manifest, "bias_coefficients", indices)
        x = self.xx[None]
        y = self.yy[None]
        bias = (
            coefficients[:, 0, None, None] * x
            + coefficients[:, 1, None, None] * y
            + coefficients[:, 2, None, None] * x.square()
            + coefficients[:, 3, None, None] * y.square()
            + coefficients[:, 4, None, None] * x * y
            + coefficients[:, 5, None, None] * (x.square() + y.square())
        )
        local_gamma = self._tensor(manifest, "local_gamma_xy", indices)
        gamma_field = torch.exp(
            torch.log(local_gamma[:, 0, None, None]) * x
            + torch.log(local_gamma[:, 1, None, None]) * y
        ).clamp(0.35, 2.8)
        tone_mapped = (image * torch.exp(bias)[:, None]).clamp(1e-5, 1.0).pow(gamma_field[:, None])
        image = torch.where(flaw.any(1)[:, None, None, None], tone_mapped, image)
        return image

    def _sensor_artifacts(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        manifest: dict[str, np.ndarray],
        indices: slice,
    ) -> torch.Tensor:
        enabled = self._tensor(manifest, "sensor_enabled", indices, torch.bool)[:, None, None, None]
        seed = torch.remainder(self._tensor(manifest, "sensor_seed", indices).float(), 104729.0)[:, None, None]
        x = self.pixel_x[None]
        y = self.pixel_y[None]
        uniform_a = torch.remainder(torch.sin(x * 12.9898 + y * 78.233 + seed * 0.017) * 43758.5453, 1.0)
        uniform_b = torch.remainder(torch.sin(x * 39.3467 + y * 11.135 + seed * 0.031) * 24634.6345, 1.0)
        noise = (uniform_a + uniform_b - 1.0) * self._tensor(manifest, "sensor_noise", indices)[:, None, None]
        speck = uniform_a > (1.0 - self._tensor(manifest, "speck_density", indices)[:, None, None])
        speck = F.max_pool2d(speck.float()[:, None], 3, stride=1, padding=1)[:, 0]
        speck *= self._tensor(manifest, "speck_strength", indices)[:, None, None]

        angle = torch.deg2rad(self._tensor(manifest, "streak_angle_deg", indices))[:, None, None]
        line_coordinate = -torch.sin(angle) * self.xx[None] + torch.cos(angle) * self.yy[None]
        streak = torch.exp(
            -0.5
            * (
                (line_coordinate - self._tensor(manifest, "streak_offset", indices)[:, None, None])
                / self._tensor(manifest, "streak_width", indices)[:, None, None]
            ).square()
        )
        streak *= (torch.remainder(seed[:, 0, 0], 13.0) < 7.0)[:, None, None]

        center = self._tensor(manifest, "blowout_center", indices)
        radius = self._tensor(manifest, "blowout_radius", indices)[:, None, None]
        blowout = torch.exp(
            -0.5
            * (
                (self.xx[None] - center[:, 0, None, None]).square()
                + (self.yy[None] - center[:, 1, None, None]).square()
            )
            / radius.square()
        )
        blowout *= self._tensor(manifest, "blowout_strength", indices)[:, None, None]
        blowout *= (torch.remainder(seed[:, 0, 0], 17.0) < 11.0)[:, None, None]
        corrupted = image + (noise + speck + 0.70 * streak + blowout)[:, None] * mask
        return torch.where(enabled, corrupted.clamp(0.0, 1.0), image)

    def _damage_background(self, manifest: dict[str, np.ndarray], indices: slice) -> torch.Tensor:
        seed = torch.remainder(self._tensor(manifest, "background_seed", indices).float(), 65521.0)[:, None, None]
        noise = torch.remainder(
            torch.sin(self.pixel_x[None] * 19.193 + self.pixel_y[None] * 47.117 + seed * 0.019) * 43758.5453,
            1.0,
        ) - 0.5
        level = self._tensor(manifest, "background_level", indices)[:, None, None]
        texture = self._tensor(manifest, "background_texture", indices)[:, None, None]
        gradient = 0.12 * level * (self.xx[None] + self.yy[None])
        return (level + texture * noise + gradient).clamp(0.0, 0.35)[:, None]

    def _occlude(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        manifest: dict[str, np.ndarray],
        indices: slice,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        types = manifest["occlusion_type"][indices]
        if not np.any(types):
            return image, mask
        masks = np.ones((len(types), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        fold_light = np.zeros_like(masks)
        fold_dark = np.zeros_like(masks)
        brain_masks = mask[:, 0].detach().cpu().numpy()
        damage_modes = manifest["damage_mode"][indices]
        cohorts = manifest["cohort"][indices]
        for item, kind in enumerate(types):
            if not kind:
                continue
            rng = np.random.default_rng(int(manifest["occlusion_seed"][indices][item]))
            if kind == 1:
                angle = rng.uniform(0.0, np.pi)
                normal = np.asarray([np.cos(angle), np.sin(angle)])
                center = np.asarray([IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0]) + normal * rng.uniform(-0.20, 0.20) * IMAGE_SIZE
                tangent = np.asarray([-normal[1], normal[0]]) * IMAGE_SIZE
                half_width = rng.uniform(
                    np.asarray([0.0, 0.015, 0.025, 0.040])[cohorts[item]],
                    np.asarray([0.0, 0.045, 0.080, 0.130])[cohorts[item]],
                ) * IMAGE_SIZE
                polygon = np.asarray(
                    [center - tangent - normal * half_width, center + tangent - normal * half_width,
                     center + tangent + normal * half_width, center - tangent + normal * half_width],
                    dtype=np.int32,
                )
                occluded = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                cv2.fillPoly(occluded, [polygon], 1)
            elif damage_modes[item] == 0:
                contours, _ = cv2.findContours(brain_masks[item].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
                if rng.random() < 0.80:
                    center = contour[rng.integers(len(contour))].astype(np.float32)
                else:
                    y, x = np.nonzero(brain_masks[item])
                    choice = rng.integers(len(x))
                    center = np.asarray([x[choice], y[choice]], dtype=np.float32)
                radius = rng.uniform(
                    np.asarray([0.0, 0.035, 0.055, 0.080])[cohorts[item]],
                    np.asarray([0.0, 0.150, 0.250, 0.340])[cohorts[item]],
                ) * max(np.ptp(contour[:, 0]), np.ptp(contour[:, 1]))
                angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, rng.integers(5, 10)))
                radii = radius * rng.uniform(0.45, 1.0, len(angles))
                polygon = np.column_stack([center[0] + np.cos(angles) * radii, center[1] + np.sin(angles) * radii]).astype(np.int32)
                occluded = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                cv2.fillPoly(occluded, [polygon], 1)
                brain_area = max(1, int(brain_masks[item].sum()))
                center = polygon.astype(np.float32).mean(axis=0)
                maximum_fraction = np.asarray([0.0, 0.20, 0.35, 0.50])[cohorts[item]]
                while np.count_nonzero(occluded & brain_masks[item]) > maximum_fraction * brain_area:
                    polygon = np.rint(center + 0.80 * (polygon - center)).astype(np.int32)
                    occluded.fill(0)
                    cv2.fillPoly(occluded, [polygon], 1)
            else:
                contours, _ = cv2.findContours(brain_masks[item].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
                start = contour[rng.integers(len(contour))].astype(np.float32)
                center = np.asarray([IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0], dtype=np.float32)
                direction = center - start
                direction /= np.linalg.norm(direction)
                tangent = np.asarray([-direction[1], direction[0]])
                distance = rng.uniform(
                    np.asarray([0.0, 0.25, 0.30, 0.35])[cohorts[item]],
                    np.asarray([0.0, 0.55, 0.78, 1.10])[cohorts[item]],
                ) * IMAGE_SIZE
                steps = np.linspace(0.0, distance, rng.integers(6, 11))
                jitter = rng.normal(
                    0.0,
                    rng.uniform(
                        np.asarray([0.0, 1.5, 2.0, 3.0])[cohorts[item]],
                        np.asarray([0.0, 4.0, 7.0, 10.0])[cohorts[item]],
                    ),
                    len(steps),
                )
                points = np.rint(start + steps[:, None] * direction + jitter[:, None] * tangent).astype(np.int32)
                occluded = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                width = int(rng.integers(
                    np.asarray([1, 1, 2, 3])[cohorts[item]],
                    np.asarray([2, 4, 7, 11])[cohorts[item]],
                ))
                cv2.polylines(occluded, [points], False, 1, width, cv2.LINE_AA)
                if damage_modes[item] == 2:
                    shift = np.rint(tangent * max(2, width)).astype(np.int32)
                    cv2.polylines(fold_light[item], [points + shift], False, 1, max(1, width // 2), cv2.LINE_AA)
                    cv2.polylines(fold_dark[item], [points - shift], False, 1, max(1, width // 2), cv2.LINE_AA)
            masks[item][occluded > 0] = 0
        keep = torch.from_numpy(masks).to(self.device, dtype=image.dtype)[:, None]
        visible_mask = mask & (keep > 0.5)
        background = self._damage_background(manifest, indices)
        image = image * keep + background * (1.0 - keep)
        light = torch.from_numpy(fold_light).to(self.device, dtype=image.dtype)[:, None]
        dark = torch.from_numpy(fold_dark).to(self.device, dtype=image.dtype)[:, None]
        image = (image + 0.65 * light - 0.45 * dark).clamp(0.0, 1.0)
        return image, visible_mask

    def batch(
        self,
        manifest: dict[str, np.ndarray],
        start: int,
        count: int,
        return_anatomy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = slice(start, start + count)
        rendered = self._render(manifest, indices, return_anatomy)
        if return_anatomy:
            image, mask, anatomy = rendered
        else:
            image, mask = rendered
            anatomy = None
        image, mask, anatomy = self._crop_to_brain(image, mask, anatomy)
        flow = self._elastic_flow(
            self._tensor(manifest, "warp_seed", indices, torch.int64),
            self._tensor(manifest, "warp_amplitude", indices),
            self._tensor(manifest, "warp", indices),
            self._tensor(manifest, "warp_scale_xy", indices),
            self._tensor(manifest, "warp_shear_xy", indices),
            self._tensor(manifest, "warp_bulge_center", indices),
            self._tensor(manifest, "warp_bulge_strength", indices),
        )
        image, mask, anatomy = self._warp(image, mask, flow, anatomy)
        rotation = self._tensor(manifest, "rotation_deg", indices)
        image, mask, anatomy = self._affine_canvas(
            image,
            mask,
            rotation,
            self._tensor(manifest, "scale", indices),
            self._tensor(manifest, "translation_xy", indices),
            anatomy,
        )
        measured_orientation = self._mask_orientation_deg(mask)
        image, mask, anatomy = self._affine_canvas(
            image,
            mask,
            -measured_orientation + self._tensor(manifest, "orientation_residual_deg", indices),
            torch.ones_like(rotation),
            torch.zeros((count, 2), device=self.device),
            anatomy,
        )
        image, mask, anatomy = self._crop_to_brain(image, mask, anatomy)
        image = self._optical_flaws(image, mask, manifest, indices)
        image = self._sensor_artifacts(image, mask, manifest, indices)
        outline_mask = mask
        image, visible_mask = self._occlude(image, mask, manifest, indices)
        if anatomy is not None:
            anatomy = anatomy.round().long() * visible_mask.long()
        tissue_count = outline_mask.flatten(2).sum(2).clamp_min(1.0)
        mean = (image * outline_mask).flatten(2).sum(2) / tissue_count
        variance = (((image - mean[:, :, None, None]) ** 2) * outline_mask).flatten(2).sum(2) / tissue_count
        image = (
            (image - mean[:, :, None, None])
            / variance.sqrt().clamp_min(1e-4)[:, :, None, None]
            / 4.0
            + 0.5
        ).clamp(0.0, 1.0)
        image = (image * outline_mask).mul(255.0).round().div(255.0)
        image = image.repeat(1, 3, 1, 1)
        targets = torch.stack(
            (
                self._tensor(manifest, "ap_um", indices),
                self._tensor(manifest, "tilt_lr_deg", indices),
                self._tensor(manifest, "tilt_dv_deg", indices),
            ),
            dim=1,
        )
        normalized_targets = (targets - torch.as_tensor(TARGET_CENTER, device=self.device)) / torch.as_tensor(
            TARGET_SCALE, device=self.device
        )
        if return_anatomy:
            return image, normalized_targets, targets, anatomy[:, 0]
        return image, normalized_targets, targets
