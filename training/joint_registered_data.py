"""Deterministic Product-5 real-section batches for joint pose/review training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from source.atlas_pose_runtime import (
    as_gray,
    automatic_brain_mask,
    preprocess_atlas_pose_image,
)
from training.dense_registration_model import identity_pixel_map, warp_tensor
from source.dense_registration_preprocessing import FEATHER_RING_VALUES, MODEL_SHAPE, NATIVE_SHAPE, PAD_X
from training.atlas_pose_models_v7 import AP_MAX_UM, AP_MIN_UM, TILT_MAX_DEG, TILT_MIN_DEG
from training.joint_pose_registration_data import AP_OFFSET_LEVELS_UM, TILT_OFFSET_LEVELS_DEG
from training.registered_section_dataset import (
    RegisteredSectionDataset,
    registered_image_cache_key,
    registered_static_cache_contract,
    registered_static_cache_key,
)
from training.train_dense_registration import sha256_file


REGISTERED_JOINT_VERSION = 1
SUPERVISED_PRODUCT_IDS = (5,)


def mask_affine_homography(
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Batched Torch equivalent of the runtime outline PCA/bbox affine."""
    source_mask = source_mask.detach().bool()
    target_mask = target_mask.detach().bool()
    if source_mask.shape != target_mask.shape or source_mask.ndim != 4:
        raise ValueError("source and target masks must have matching B,1,H,W shapes")
    batch, _, height, width = source_mask.shape
    device = source_mask.device
    dtype = torch.float64
    source = source_mask[:, 0].reshape(batch, -1)
    target = target_mask[:, 0].reshape(batch, -1)
    source_count = source.sum(1)
    target_count = target.sum(1)
    if bool(((source_count < 64) | (target_count < 64)).any()):
        raise ValueError("a brain surface is missing from the slice or atlas plane")

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    weight = source.to(dtype)
    count = source_count.to(dtype)
    mean_x = (weight * x).sum(1) / count
    mean_y = (weight * y).sum(1) / count
    centered_x = x - mean_x[:, None]
    centered_y = y - mean_y[:, None]
    covariance_xy = (weight * centered_x * centered_y).sum(1) / count
    variance_x = (weight * centered_x.square()).sum(1) / count
    variance_y = (weight * centered_y.square()).sum(1) / count
    angle = 0.5 * torch.atan2(2.0 * covariance_xy, variance_x - variance_y)
    cosine, sine = angle.cos(), angle.sin()
    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    rotation_tx = (1.0 - cosine) * center_x - sine * center_y
    rotation_ty = sine * center_x + (1.0 - cosine) * center_y

    corners_x = source_mask.new_tensor(
        (0.0, width - 1.0, 0.0, width - 1.0), dtype=dtype
    )[None]
    corners_y = source_mask.new_tensor(
        (0.0, 0.0, height - 1.0, height - 1.0), dtype=dtype
    )[None]
    rotated_corner_x = (
        cosine[:, None] * corners_x
        + sine[:, None] * corners_y
        + rotation_tx[:, None]
    )
    rotated_corner_y = (
        -sine[:, None] * corners_x
        + cosine[:, None] * corners_y
        + rotation_ty[:, None]
    )
    rotation_tx = rotation_tx - rotated_corner_x.min(1).values
    rotation_ty = rotation_ty - rotated_corner_y.min(1).values

    rotated_x = cosine[:, None] * x + sine[:, None] * y + rotation_tx[:, None]
    rotated_y = -sine[:, None] * x + cosine[:, None] * y + rotation_ty[:, None]
    infinity = torch.tensor(torch.inf, device=device, dtype=dtype)
    source_min_x = torch.where(source, rotated_x, infinity).min(1).values
    source_max_x = torch.where(source, rotated_x, -infinity).max(1).values
    source_min_y = torch.where(source, rotated_y, infinity).min(1).values
    source_max_y = torch.where(source, rotated_y, -infinity).max(1).values
    target_min_x = torch.where(target, x, infinity).min(1).values
    target_max_x = torch.where(target, x, -infinity).max(1).values
    target_min_y = torch.where(target, y, infinity).min(1).values
    target_max_y = torch.where(target, y, -infinity).max(1).values
    scale_x = (target_max_x - target_min_x) / (source_max_x - source_min_x).clamp_min(1.0)
    scale_y = (target_max_y - target_min_y) / (source_max_y - source_min_y).clamp_min(1.0)
    scale = 0.5 * (scale_x + scale_y)
    source_center_x = 0.5 * (source_min_x + source_max_x)
    source_center_y = 0.5 * (source_min_y + source_max_y)
    target_center_x = 0.5 * (target_min_x + target_max_x)
    target_center_y = 0.5 * (target_min_y + target_max_y)

    homography = torch.zeros(batch, 3, 3, device=device, dtype=dtype)
    homography[:, 0, 0] = scale * cosine
    homography[:, 0, 1] = scale * sine
    homography[:, 1, 0] = -scale * sine
    homography[:, 1, 1] = scale * cosine
    homography[:, 0, 2] = (
        scale * rotation_tx + target_center_x - scale * source_center_x
    )
    homography[:, 1, 2] = (
        scale * rotation_ty + target_center_y - scale * source_center_y
    )
    homography[:, 2, 2] = 1.0
    return homography.to(dtype=torch.float32)


def affine_pixel_map(homography: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Return output-to-input pixels for warping through a source-to-output affine."""
    height, width = shape
    identity = identity_pixel_map(
        len(homography), height, width, device=homography.device, dtype=homography.dtype
    )
    homogeneous = torch.cat((identity, torch.ones_like(identity[:, :1])), dim=1)
    return torch.einsum("bij,bjhw->bihw", torch.linalg.inv(homography)[:, :2], homogeneous)


def apply_homography_to_map(
    homography: torch.Tensor,
    pixel_map: torch.Tensor,
) -> torch.Tensor:
    homogeneous = torch.cat((pixel_map, torch.ones_like(pixel_map[:, :1])), dim=1)
    return torch.einsum("bij,bjhw->bihw", homography[:, :2], homogeneous)


def mask_normalized_moving(
    moving: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    apply_cosine_feather: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized runtime-contract outline normalization for one or more targets."""
    if len(target_mask) % len(moving):
        raise ValueError("target masks are not grouped by source slice")
    repeats = len(target_mask) // len(moving)
    moving = moving.repeat_interleave(repeats, dim=0)
    source_mask = source_mask.repeat_interleave(repeats, dim=0).to(target_mask.device)
    moving = moving.to(target_mask.device)
    homography = mask_affine_homography(source_mask, target_mask)
    inverse_map = affine_pixel_map(homography, target_mask.shape[-2:])
    aligned_moving = warp_tensor(moving, inverse_map, padding_mode="zeros")
    aligned_mask = warp_tensor(
        source_mask.float(), inverse_map, mode="nearest", padding_mode="zeros"
    ) > 0.5
    if apply_cosine_feather:
        support = aligned_mask
        alpha = support.float()
        for value in FEATHER_RING_VALUES:
            expanded = torch.nn.functional.max_pool2d(
                support.float(), 3, stride=1, padding=1
            ) > 0.5
            alpha = torch.where(expanded & ~support, float(value), alpha)
            support = expanded
        aligned_moving = aligned_moving * alpha
    return aligned_moving, aligned_mask, homography, inverse_map


def _payload_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _seeded_rng(seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"joint-registered-v1:{int(seed)}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def canonical_registration_image(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place raw uint8/255 tissue on a native canvas; candidate affine feathers it."""
    gray = np.clip(np.rint(as_gray(image)), 0, 255).astype(np.uint8)
    mask = np.asarray(mask, dtype=bool)
    y, x = np.nonzero(mask)
    height, width = NATIVE_SHAPE
    scale = min(
        (width - 1) / max(float(x.max() - x.min()), 1.0),
        (height - 1) / max(float(y.max() - y.min()), 1.0),
    )
    source_center = np.asarray(
        ((x.min() + x.max()) / 2.0, (y.min() + y.max()) / 2.0),
        dtype=np.float32,
    )
    target_center = np.asarray(((width - 1) / 2.0, (height - 1) / 2.0), np.float32)
    matrix = np.asarray(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0]], dtype=np.float32
    )
    matrix[:, 2] = target_center - scale * source_center
    native = cv2.warpAffine(
        gray, matrix, (width, height), flags=cv2.INTER_LINEAR
    )
    native_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    canvas = np.zeros(MODEL_SHAPE, np.float32)
    canvas_mask = np.zeros(MODEL_SHAPE, bool)
    canvas[:, PAD_X : PAD_X + width] = native.astype(np.float32) / 255.0
    canvas_mask[:, PAD_X : PAD_X + width] = native_mask
    return (
        torch.from_numpy(canvas[None].astype(np.float32)),
        torch.from_numpy(canvas_mask[None]),
    )


def _candidate_poses(
    true_pose: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.empty_like(true_pose, dtype=np.float32)
    wrong = np.empty((len(true_pose), count, 3), dtype=np.float32)
    signed_tilt = np.concatenate((-TILT_OFFSET_LEVELS_DEG, TILT_OFFSET_LEVELS_DEG))
    for item, pose in enumerate(true_pose):
        signed_ap = np.asarray(
            [
                sign * level
                for level in AP_OFFSET_LEVELS_UM
                for sign in (-1.0, 1.0)
                if AP_MIN_UM <= pose[0] + sign * level <= AP_MAX_UM
            ],
            dtype=np.float32,
        )
        valid_lr = signed_tilt[
            (TILT_MIN_DEG <= pose[1] + signed_tilt)
            & (pose[1] + signed_tilt <= TILT_MAX_DEG)
        ]
        valid_dv = signed_tilt[
            (TILT_MIN_DEG <= pose[2] + signed_tilt)
            & (pose[2] + signed_tilt <= TILT_MAX_DEG)
        ]
        pool = (
            [(value, 0.0, 0.0) for value in signed_ap]
            + [(0.0, value, 0.0) for value in valid_lr]
            + [(0.0, 0.0, value) for value in valid_dv]
        )
        if count > len(pool):
            raise ValueError("too many registered hard negatives requested")
        mandatory = []
        if count >= 1:
            adjacent_ap = signed_ap[np.abs(signed_ap) == AP_OFFSET_LEVELS_UM[0]]
            mandatory.append((float(rng.choice(adjacent_ap)), 0.0, 0.0))
        if count >= 2:
            adjacent_lr = valid_lr[np.abs(valid_lr) == TILT_OFFSET_LEVELS_DEG[0]]
            mandatory.append((0.0, float(rng.choice(adjacent_lr)), 0.0))
        if count >= 3:
            adjacent_dv = valid_dv[np.abs(valid_dv) == TILT_OFFSET_LEVELS_DEG[0]]
            mandatory.append((0.0, 0.0, float(rng.choice(adjacent_dv))))
        remaining_pool = [offset for offset in pool if offset not in mandatory]
        remaining_count = count - len(mandatory)
        selected = (
            rng.choice(len(remaining_pool), remaining_count, replace=False)
            if remaining_count
            else ()
        )
        offsets = mandatory + [remaining_pool[int(index)] for index in selected]
        wrong[item] = pose + np.asarray(offsets, np.float32)
        initial[item] = pose + np.asarray(
            (
                rng.choice(signed_ap),
                rng.choice(valid_lr),
                rng.choice(valid_dv),
            ),
            dtype=np.float32,
        )
    return initial, wrong


class JointRegisteredData:
    """Balanced deterministic sampling from the canonical Product-5 train split."""

    def __init__(
        self,
        manifest_root: str | Path,
        atlas_folder: str | Path,
        joint_synthetic_data,
        *,
        split: str = "train",
        dataset: RegisteredSectionDataset | None = None,
    ):
        if split not in {"train", "validation"}:
            raise ValueError("registered joint data supports train or validation")
        self.split = split
        self.root = Path(manifest_root)
        self.atlas_folder = Path(atlas_folder)
        self.joint_synthetic_data = joint_synthetic_data
        self.device = joint_synthetic_data.generator.device
        self.dataset = dataset or RegisteredSectionDataset(
            self.root,
            self.atlas_folder,
            split=split,
            include_anatomy=False,
            allowed_product_ids=SUPERVISED_PRODUCT_IDS,
        )
        static_contract = registered_static_cache_contract(self.root, self.atlas_folder)
        static_key = registered_static_cache_key(self.root, self.atlas_folder)
        static_root = self.root / ".atlas_pose_cache" / static_key
        static_contract_path = static_root / "contract.json"
        expected_static_payload = {"cache_key": static_key, "contract": static_contract}
        self.static_mask_cache_folder = (
            static_root / "training_static"
            if static_contract_path.is_file()
            and json.loads(static_contract_path.read_text(encoding="utf-8"))
            == expected_static_payload
            else None
        )
        self.record_indices = [
            index
            for index, record in enumerate(self.dataset.records)
            if bool(
                record.get(
                    "in_training_ap_domain",
                    AP_MIN_UM <= float(record["ap_um"]) <= AP_MAX_UM,
                )
            )
        ]
        selected_records = [self.dataset.records[index] for index in self.record_indices]
        if not selected_records:
            raise RuntimeError("registered Product-5 train split has no in-domain sections")
        if any(record["split"] != split for record in selected_records):
            raise RuntimeError(f"registered joint {split} contains a different split")
        if any(
            not set(
                int(value)
                for value in self.dataset.datasets[int(record["experiment_id"])]["product_ids"]
            ).issubset(SUPERVISED_PRODUCT_IDS)
            for record in selected_records
        ):
            raise RuntimeError("registered joint training contains a non-Product-5 section")
        split_specimens: dict[str, set[int]] = {}
        for record in self.dataset.datasets.values():
            split_specimens.setdefault(record["split"], set()).add(int(record["specimen_id"]))
        if split_specimens.get("train", set()) & split_specimens.get("validation", set()):
            raise RuntimeError("registered train and validation specimens overlap")

        section_counts: dict[int, int] = {}
        self.specimen_positions: dict[int, list[int]] = {}
        for position, record in enumerate(selected_records):
            specimen = int(record["specimen_id"])
            section_counts[specimen] = section_counts.get(specimen, 0) + 1
            self.specimen_positions.setdefault(specimen, []).append(position)
        specimen_count = len(section_counts)
        self.sampling_weights = np.asarray(
            [
                1.0
                / (
                    specimen_count
                    * section_counts[int(record["specimen_id"])]
                )
                for record in selected_records
            ],
            dtype=np.float64,
        )
        self.sampling_weights /= self.sampling_weights.sum()
        record_ids = [int(record["section_image_id"]) for record in selected_records]
        contract = {
            "version": REGISTERED_JOINT_VERSION,
            "split": split,
            "product_ids": list(SUPERVISED_PRODUCT_IDS),
            "record_count": len(record_ids),
            "record_ids_sha256": _payload_sha256({"section_image_ids": record_ids}),
            "specimen_ids": sorted(section_counts),
            "registered_image_contract_sha256": registered_image_cache_key(self.root),
            "registered_static_mask_contract_sha256": static_key,
            "registrar_orientation_contract": (
                "Product-5 registered-reference orientation; arbitrary pose-view rotation "
                "is encoder-only and never rotates the registrar source"
            ),
            "quality_manifest_sha256": self.dataset.quality_manifest_sha256,
            "datasets_sha256": _optional_sha256(self.root / "datasets.jsonl"),
            "sections_sha256": _optional_sha256(self.root / "sections.jsonl"),
            "downloads_sha256": _optional_sha256(self.root / "downloads.jsonl"),
            "adapter_source_sha256": sha256_file(Path(__file__)),
        }
        contract["contract_sha256"] = _payload_sha256(contract)
        self.contract = contract

    def render_pose(self, pose: torch.Tensor):
        return self.joint_synthetic_data.render_pose(pose)

    def moving_for_fixed(
        self,
        batch: dict,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Apply the runtime outline-derived affine separately for every candidate."""
        moving, mask, homography, _ = mask_normalized_moving(
            batch.get("_outline_source_moving", batch["moving"]),
            batch.get("_outline_source_mask", batch["moving_model_mask"]),
            target_mask,
            apply_cosine_feather=True,
        )
        return moving, mask, {
            "source_to_aligned_h": homography,
        }

    def _raw_image_and_mask(
        self, index: int, item: dict | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if item is not None and "raw_image" in item and "raw_mask" in item:
            return np.asarray(item["raw_image"]), np.asarray(item["raw_mask"], dtype=bool)
        record = self.dataset.records[index]
        with Image.open(self.root / record["relative_path"]) as source:
            image = np.asarray(source).copy()
        cache_path = (
            self.static_mask_cache_folder
            / record["split"]
            / f"{int(record['section_image_id'])}.npz"
            if self.static_mask_cache_folder is not None
            else None
        )
        if cache_path is not None and cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                return image, cached["mask"].astype(bool)
        return image, np.asarray(automatic_brain_mask(image), dtype=bool)

    def _item_and_registration(self, index: int) -> tuple[dict, tuple[torch.Tensor, torch.Tensor]]:
        record = self.dataset.records[index]
        if "relative_path" not in record:
            item = self.dataset[index]
            image, mask = self._raw_image_and_mask(index, item)
            return item, canonical_registration_image(image, mask)
        image, mask = self._raw_image_and_mask(index)
        item = {
            "image": torch.from_numpy(preprocess_atlas_pose_image(image, mask)),
            "pose": torch.tensor(
                (
                    float(record["ap_um"]),
                    float(record["tilt_lr_deg"]),
                    float(record["tilt_dv_deg"]),
                ),
                dtype=torch.float32,
            ),
            "specimen_id": torch.tensor(int(record["specimen_id"])),
            "section_image_id": torch.tensor(int(record["section_image_id"])),
        }
        return item, canonical_registration_image(image, mask)

    def generate(self, count: int, seed: int, negatives_per_sample: int) -> dict:
        rng = _seeded_rng(seed)
        positions = rng.choice(
            len(self.record_indices), count, replace=True, p=self.sampling_weights
        )
        return self.batch_positions(positions, seed, negatives_per_sample)

    def fixed_validation_positions(self, count: int, seed: int) -> np.ndarray:
        if self.split != "validation":
            raise RuntimeError("fixed validation positions require the validation split")
        rng = _seeded_rng(seed)
        specimens = np.asarray(sorted(self.specimen_positions), dtype=np.int64)
        specimens = specimens[rng.permutation(len(specimens))]
        positions = []
        cycle = 0
        while len(positions) < count:
            for specimen in specimens:
                choices = self.specimen_positions[int(specimen)]
                positions.append(choices[(cycle + int(rng.integers(len(choices)))) % len(choices)])
                if len(positions) == count:
                    break
            cycle += 1
        return np.asarray(positions, dtype=np.int64)

    def batch_positions(
        self,
        positions,
        seed: int,
        negatives_per_sample: int,
    ) -> dict:
        positions = np.asarray(positions, dtype=np.int64)
        rng = _seeded_rng(seed)
        dataset_indices = [self.record_indices[int(index)] for index in positions]
        loaded = [self._item_and_registration(index) for index in dataset_indices]
        items = [item for item, _ in loaded]
        registration = [pair for _, pair in loaded]
        pose_image = torch.stack([item["image"] for item in items]).to(self.device)
        true_pose = torch.stack([item["pose"] for item in items]).to(self.device)
        moving = torch.stack([item[0] for item in registration]).to(self.device)
        moving_mask = torch.stack([item[1] for item in registration]).to(self.device)
        initial_np, wrong_np = _candidate_poses(
            true_pose.detach().cpu().numpy(), negatives_per_sample, rng
        )
        initial_pose = torch.from_numpy(initial_np).to(self.device)
        wrong_pose = torch.from_numpy(wrong_np).to(self.device)
        fixed, fixed_mask, fixed_labels = self.render_pose(true_pose)
        initial_fixed, initial_mask, initial_labels = self.render_pose(initial_pose)
        batch, candidates = wrong_pose.shape[:2]
        wrong_fixed, wrong_mask, wrong_labels = self.render_pose(
            wrong_pose.reshape(batch * candidates, 3)
        )
        shape = wrong_fixed.shape[1:]
        return {
            "source": "allen_registered_product5",
            "registered_contract_sha256": self.contract["contract_sha256"],
            "pose_image": pose_image.float(),
            "true_pose": true_pose.float(),
            "orientation_inverted_target": torch.zeros(
                len(positions), device=self.device, dtype=torch.bool
            ),
            "moving": moving.float(),
            "moving_model_mask": moving_mask,
            "moving_tissue_mask": moving_mask,
            "moving_visible_mask": moving_mask,
            "fixed": fixed,
            "fixed_mask": fixed_mask,
            "fixed_labels": fixed_labels,
            "initial_pose": initial_pose,
            "initial_fixed": initial_fixed,
            "initial_fixed_mask": initial_mask,
            "initial_fixed_labels": initial_labels,
            "wrong_candidate_pose": wrong_pose,
            "wrong_candidate_fixed": wrong_fixed.reshape(batch, candidates, *shape),
            "wrong_candidate_fixed_mask": wrong_mask.reshape(batch, candidates, *shape),
            "wrong_candidate_fixed_labels": wrong_labels.reshape(batch, candidates, *shape),
            "specimen_id": torch.stack([item["specimen_id"] for item in items]).to(
                self.device
            ),
            "section_image_id": torch.stack(
                [item["section_image_id"] for item in items]
            ).to(self.device),
        }
