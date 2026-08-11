from __future__ import annotations

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


def make_manifest(count: int, split: str, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    ap_fraction = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(ap_fraction)
    ap_um = AP_MAX_UM + (AP_MIN_UM - AP_MAX_UM) * ap_fraction

    tilt = np.clip(rng.normal(0.0, 5.0, (count, 2)), -15.0, 15.0)
    tail = rng.random((count, 2)) < 0.10
    tilt[tail] = rng.uniform(-25.0, 25.0, int(tail.sum()))

    optical = rng.random(count) < 0.90
    flaw_count = rng.integers(1, 4, count)
    flaw_priority = rng.random((count, 3))
    flaw_rank = np.argsort(np.argsort(flaw_priority, axis=1), axis=1)
    flaw_mask = (flaw_rank < flaw_count[:, None]) & optical[:, None]

    occlusion_draw = rng.random(count)
    occlusion_type = np.zeros(count, dtype=np.uint8)
    occlusion_type[occlusion_draw < 0.04] = 1
    occlusion_type[(occlusion_draw >= 0.04) & (occlusion_draw < 0.40)] = 2

    return {
        "ap_um": ap_um.astype(np.float32),
        "ap_index": (BREGMA_AP_INDEX - ap_um / VOXEL_UM).astype(np.float32),
        "tilt_lr_deg": tilt[:, 0].astype(np.float32),
        "tilt_dv_deg": tilt[:, 1].astype(np.float32),
        "rotation_deg": rng.uniform(-180.0, 180.0, count).astype(np.float32),
        "orientation_residual_deg": rng.uniform(-12.0, 12.0, count).astype(np.float32),
        "scale": rng.uniform(0.5, 1.5, count).astype(np.float32),
        "translation_xy": rng.uniform(-0.025, 0.025, (count, 2)).astype(np.float32),
        "warp": (rng.random(count) < 0.60),
        "warp_amplitude": rng.triangular(4.0, 12.0, 20.0, count).astype(np.float32),
        "warp_seed": rng.integers(0, 2**31 - 1, count, dtype=np.int64),
        "flaw_mask": flaw_mask,
        "contrast_gain": np.exp(rng.uniform(np.log(0.45), np.log(1.9), count)).astype(np.float32),
        "contrast_gamma": np.exp(rng.uniform(np.log(0.55), np.log(1.8), count)).astype(np.float32),
        "contrast_offset": rng.uniform(-0.18, 0.18, count).astype(np.float32),
        "contrast_invert": (rng.random(count) < 0.50),
        "exposure_center": rng.uniform(0.18, 0.82, count).astype(np.float32),
        "exposure_width": rng.uniform(0.08, 0.35, count).astype(np.float32),
        "exposure_strength": rng.uniform(0.35, 0.95, count).astype(np.float32),
        "exposure_sign": rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), count),
        "tile_period": rng.integers(24, 92, count, dtype=np.int32),
        "tile_phase": rng.uniform(0.0, 1.0, (count, 2)).astype(np.float32),
        "tile_strength": rng.uniform(0.12, 0.55, count).astype(np.float32),
        "tile_power": rng.uniform(1.0, 3.5, count).astype(np.float32),
        "occlusion_type": occlusion_type,
        "occlusion_seed": rng.integers(0, 2**31 - 1, count, dtype=np.int64),
    }


def save_manifest(path: Path, manifest: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **manifest)


def load_manifest(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {name: values[name] for name in values.files}


class SyntheticAtlas:
    def __init__(self, atlas_folder: Path, device: str = "cuda"):
        average = nrrd.read(str(atlas_folder / "average_template_25.nrrd"))[0]
        annotation = nrrd.read(str(atlas_folder / "annotation_25.nrrd"))[0]
        self.device = torch.device(device)
        self.shape = average.shape
        self.volume = torch.from_numpy(average.astype(np.float32) / float(average.max())).to(self.device)[None, None]
        self.mask_volume = torch.from_numpy((annotation > 0).astype(np.float32)).to(self.device)[None, None]
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

    def _elastic_flow(self, seeds: torch.Tensor, amplitudes: torch.Tensor, enabled: torch.Tensor) -> torch.Tensor:
        cell = torch.arange(72, device=self.device, dtype=torch.float32)[None]
        seed = seeds.float()[:, None]
        noise = torch.remainder(torch.sin(seed * 0.000173 + cell * 12.9898) * 43758.5453, 1.0) * 2.0 - 1.0
        flow = noise.reshape(-1, 2, 6, 6)
        flow = F.interpolate(flow, (IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=True)
        flow /= flow.flatten(2).std(dim=2, keepdim=True).clamp_min(1e-4)[..., None]
        flow *= amplitudes[:, None, None, None] * enabled[:, None, None, None]
        for _ in range(6):
            ux_y, ux_x = torch.gradient(flow[:, 0], dim=(1, 2))
            uy_y, uy_x = torch.gradient(flow[:, 1], dim=(1, 2))
            determinant = (1.0 + ux_x) * (1.0 + uy_y) - ux_y * uy_x
            shrink = torch.where(determinant.amin(dim=(1, 2)) < 0.25, 0.72, 1.0)
            flow *= shrink[:, None, None, None]
        return flow

    def _render(self, manifest: dict[str, np.ndarray], indices: slice) -> tuple[torch.Tensor, torch.Tensor]:
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
        return image, mask

    def _crop_to_brain(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        return image, mask

    def _warp(self, image: torch.Tensor, mask: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grid = torch.stack((self.xx, self.yy), dim=-1)[None]
        grid = grid + flow.permute(0, 2, 3, 1) * (2.0 / (IMAGE_SIZE - 1.0))
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        mask = F.grid_sample(mask.float(), grid, mode="nearest", padding_mode="zeros", align_corners=True) > 0.5
        return image, mask

    def _affine_canvas(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        rotation_deg: torch.Tensor,
        scale: torch.Tensor,
        translation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.shape[-1] != CANVAS_SIZE:
            before = (CANVAS_SIZE - IMAGE_SIZE) // 2
            after = CANVAS_SIZE - IMAGE_SIZE - before
            image = F.pad(image, (before, after, before, after))
            mask = F.pad(mask, (before, after, before, after))
        rotation = torch.deg2rad(rotation_deg)[:, None, None]
        x = self.canvas_x[None] - translation[:, 0, None, None]
        y = self.canvas_y[None] - translation[:, 1, None, None]
        cosine, sine = torch.cos(rotation), torch.sin(rotation)
        source_x = (cosine * x + sine * y) / scale[:, None, None]
        source_y = (-sine * x + cosine * y) / scale[:, None, None]
        grid = torch.stack((source_x, source_y), dim=-1)
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        mask = F.grid_sample(mask.float(), grid, mode="nearest", padding_mode="zeros", align_corners=True) > 0.5
        return image, mask

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

        period = self._tensor(manifest, "tile_period", indices)[:, None, None]
        phase = self._tensor(manifest, "tile_phase", indices)
        tx = torch.remainder(self.pixel_x[None] / period + phase[:, 0, None, None], 1.0)
        ty = torch.remainder(self.pixel_y[None] / period + phase[:, 1, None, None], 1.0)
        edge = torch.maximum(torch.abs(tx - 0.5), torch.abs(ty - 0.5)) * 2.0
        strength = self._tensor(manifest, "tile_strength", indices)[:, None, None]
        power = self._tensor(manifest, "tile_power", indices)[:, None, None]
        vignette = 1.0 - strength * edge.pow(power)
        checker = (
            (torch.floor(self.pixel_x[None] / period) + torch.floor(self.pixel_y[None] / period)) % 2.0
        ) * 2.0 - 1.0
        vignette *= 1.0 + 0.08 * checker
        tiled = (image * vignette[:, None]).clamp(0.0, 1.0)
        image = torch.where(flaw[:, 2, None, None, None], tiled, image)
        return image

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
        brain_masks = mask[:, 0].detach().cpu().numpy()
        for item, kind in enumerate(types):
            if not kind:
                continue
            rng = np.random.default_rng(int(manifest["occlusion_seed"][indices][item]))
            if kind == 1:
                angle = rng.uniform(0.0, np.pi)
                normal = np.asarray([np.cos(angle), np.sin(angle)])
                center = np.asarray([IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0]) + normal * rng.uniform(-0.20, 0.20) * IMAGE_SIZE
                tangent = np.asarray([-normal[1], normal[0]]) * IMAGE_SIZE
                half_width = rng.uniform(0.03, 0.13) * IMAGE_SIZE
                polygon = np.asarray(
                    [center - tangent - normal * half_width, center + tangent - normal * half_width,
                     center + tangent + normal * half_width, center - tangent + normal * half_width],
                    dtype=np.int32,
                )
            else:
                contours, _ = cv2.findContours(brain_masks[item].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
                if rng.random() < 0.80:
                    center = contour[rng.integers(len(contour))].astype(np.float32)
                else:
                    y, x = np.nonzero(brain_masks[item])
                    choice = rng.integers(len(x))
                    center = np.asarray([x[choice], y[choice]], dtype=np.float32)
                radius = rng.uniform(0.08, 0.34) * max(np.ptp(contour[:, 0]), np.ptp(contour[:, 1]))
                angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, rng.integers(5, 10)))
                radii = radius * rng.uniform(0.45, 1.0, len(angles))
                polygon = np.column_stack([center[0] + np.cos(angles) * radii, center[1] + np.sin(angles) * radii]).astype(np.int32)
            occluded = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            cv2.fillPoly(occluded, [polygon], 1)
            if kind == 2:
                brain_area = max(1, int(brain_masks[item].sum()))
                center = polygon.astype(np.float32).mean(axis=0)
                while np.count_nonzero(occluded & brain_masks[item]) > 0.50 * brain_area:
                    polygon = np.rint(center + 0.80 * (polygon - center)).astype(np.int32)
                    occluded.fill(0)
                    cv2.fillPoly(occluded, [polygon], 1)
            masks[item][occluded > 0] = 0
        keep = torch.from_numpy(masks).to(self.device, dtype=image.dtype)[:, None]
        visible_mask = mask & (keep > 0.5)
        return image * keep, visible_mask

    def batch(
        self,
        manifest: dict[str, np.ndarray],
        start: int,
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = slice(start, start + count)
        image, mask = self._render(manifest, indices)
        image, mask = self._crop_to_brain(image, mask)
        flow = self._elastic_flow(
            self._tensor(manifest, "warp_seed", indices, torch.int64),
            self._tensor(manifest, "warp_amplitude", indices),
            self._tensor(manifest, "warp", indices),
        )
        image, mask = self._warp(image, mask, flow)
        rotation = self._tensor(manifest, "rotation_deg", indices)
        image, mask = self._affine_canvas(
            image,
            mask,
            rotation,
            self._tensor(manifest, "scale", indices),
            self._tensor(manifest, "translation_xy", indices),
        )
        measured_orientation = self._mask_orientation_deg(mask)
        image, mask = self._affine_canvas(
            image,
            mask,
            -measured_orientation + self._tensor(manifest, "orientation_residual_deg", indices),
            torch.ones_like(rotation),
            torch.zeros((count, 2), device=self.device),
        )
        image, mask = self._crop_to_brain(image, mask)
        image = self._optical_flaws(image, mask, manifest, indices)
        image, _ = self._occlude(image, mask, manifest, indices)
        tissue_count = mask.flatten(2).sum(2).clamp_min(1.0)
        mean = (image * mask).flatten(2).sum(2) / tissue_count
        variance = (((image - mean[:, :, None, None]) ** 2) * mask).flatten(2).sum(2) / tissue_count
        image = (
            (image - mean[:, :, None, None])
            / variance.sqrt().clamp_min(1e-4)[:, :, None, None]
            / 4.0
            + 0.5
        ).clamp(0.0, 1.0)
        image *= mask
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
        return image, normalized_targets, targets
