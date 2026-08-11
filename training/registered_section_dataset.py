from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
from importlib.metadata import version
from functools import lru_cache
from pathlib import Path

import cv2
import nrrd
import numpy as np
import torch
import PIL
from PIL import Image
from torch.utils.data import Dataset

from source.atlas_pose_runtime import (
    ATLAS_POSE_PREPROCESSING_VERSION,
    AUTOMATIC_BRAIN_MASK_VERSION,
    POSE_IMAGE_SIZE,
    automatic_brain_mask,
    atlas_pose_preprocessing_contract_sha256,
    brain_orientation_affine,
    preprocess_atlas_pose_image,
)
from source.registered_image_quality import (
    REGISTERED_IMAGE_QUALITY_MANIFEST,
    REGISTERED_IMAGE_QUALITY_VERSION,
    load_registered_image_quality_manifest,
)


DOWNSAMPLE = 5
DOWNSAMPLE_FACTOR = 2**DOWNSAMPLE
VOXEL_UM = 25.0
SEALED_SPLIT = "sealed_deepslice_s2p"
COARSE_ANATOMY_CLASSES = (
    "exterior_background",
    "cortex",
    "hippocampal_formation",
    "cerebral_nuclei",
    "thalamus",
    "hypothalamus",
    "midbrain_hindbrain",
    "cerebellum",
    "fiber_tracts_ventricles_internal_cavities",
)
COARSE_ANATOMY_ROOT_IDS = {
    1: (688,),
    2: (1089,),
    3: (623,),
    4: (549,),
    5: (1097,),
    6: (313, 1065),
    7: (512,),
    8: (1009, 73),
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@lru_cache(maxsize=16)
def _versioned_file_sha256(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return _versioned_file_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def registered_image_cache_key(root: str | Path) -> str:
    root = Path(root)
    contract = {
        "sections_sha256": _file_sha256(root / "sections.jsonl"),
        "downloads_sha256": _file_sha256(root / "downloads.jsonl"),
        "preprocessing_version": ATLAS_POSE_PREPROCESSING_VERSION,
        "preprocessing_contract_sha256": atlas_pose_preprocessing_contract_sha256(),
        "automatic_brain_mask_version": AUTOMATIC_BRAIN_MASK_VERSION,
        "registered_image_quality_version": REGISTERED_IMAGE_QUALITY_VERSION,
        "registered_image_quality_manifest_sha256": _file_sha256(
            root / REGISTERED_IMAGE_QUALITY_MANIFEST
        ),
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _callable_contract_sha256(function) -> str | None:
    if function is None:
        return None
    source_path = Path(inspect.getsourcefile(function) or "")
    if not source_path.is_file():
        raise ValueError(f"Cannot hash augmentation source for {function.__qualname__}")
    payload = {
        "module": function.__module__,
        "name": function.__qualname__,
        "defaults": repr(function.__defaults__),
        "source_sha256": _file_sha256(source_path),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def registered_static_cache_key(
    root: str | Path,
    atlas_folder: str | Path,
    augmentation=None,
    seed: int = 0,
    views: int = 1,
) -> str:
    root = Path(root)
    atlas_folder = Path(atlas_folder)
    contract = {
        "image_contract_sha256": registered_image_cache_key(root),
        "datasets_sha256": _file_sha256(root / "datasets.jsonl"),
        "annotation_sha256": _file_sha256(atlas_folder / "annotation_25.nrrd"),
        "atlas_labels_sha256": _file_sha256(atlas_folder / "atlas_labels.pkl"),
        "registered_dataset_source_sha256": _file_sha256(Path(__file__)),
        "augmentation_contract_sha256": _callable_contract_sha256(augmentation),
        "dependency_versions": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
            "pynrrd": version("pynrrd"),
        },
        "seed": int(seed),
        "views": int(views),
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _coarse_lookup(annotation: np.ndarray, atlas_labels: dict) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(atlas_labels["index"], dtype=np.int64)
    parents = np.asarray(atlas_labels["parent"], dtype=np.int64)
    parent = {int(structure_id): int(parent_id) for structure_id, parent_id in zip(ids, parents)}
    roots = {
        root_id: class_id
        for class_id, root_ids in COARSE_ANATOMY_ROOT_IDS.items()
        for root_id in root_ids
    }
    annotation_ids = np.unique(annotation).astype(np.int64)
    classes = np.zeros(len(annotation_ids), dtype=np.uint8)
    for index, structure_id in enumerate(annotation_ids):
        ancestors = []
        current = int(structure_id)
        while current > 0 and current not in ancestors:
            ancestors.append(current)
            current = parent.get(current, -1)
        if 1089 in ancestors:
            classes[index] = 2
            continue
        for ancestor in ancestors:
            if ancestor in roots:
                classes[index] = roots[ancestor]
                break
    return annotation_ids, classes


def _map_coarse(annotation_ids: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(keys, annotation_ids)
    valid = positions < len(keys)
    valid[valid] &= keys[positions[valid]] == annotation_ids[valid]
    result = np.zeros(annotation_ids.shape, dtype=np.uint8)
    result[valid] = values[positions[valid]]
    return result


def project_annotation(
    annotation: np.ndarray,
    coarse_keys: np.ndarray,
    coarse_values: np.ndarray,
    dataset_record: dict,
    section_record: dict,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_shape
    source_x = (np.arange(width, dtype=np.float64) + 0.5) * DOWNSAMPLE_FACTOR - 0.5
    source_y = (np.arange(height, dtype=np.float64) + 0.5) * DOWNSAMPLE_FACTOR - 0.5
    source_x, source_y = np.meshgrid(source_x, source_y)
    tsv = np.asarray(section_record["alignment2d_tsv"], dtype=np.float64)
    volume_x = tsv[0] * source_x + tsv[1] * source_y + tsv[4]
    volume_y = tsv[2] * source_x + tsv[3] * source_y + tsv[5]
    volume_z = float(section_record["section_number"]) * float(dataset_record["section_thickness_um"])
    tvr = np.asarray(dataset_record["alignment3d_tvr"], dtype=np.float64)
    reference_p = tvr[0] * volume_x + tvr[1] * volume_y + tvr[2] * volume_z + tvr[9]
    reference_i = tvr[3] * volume_x + tvr[4] * volume_y + tvr[5] * volume_z + tvr[10]
    reference_r = tvr[6] * volume_x + tvr[7] * volume_y + tvr[8] * volume_z + tvr[11]
    ap = np.rint(reference_p / VOXEL_UM).astype(np.int32)
    dv = np.rint(reference_i / VOXEL_UM).astype(np.int32)
    ml = np.rint(reference_r / VOXEL_UM).astype(np.int32)
    valid = (
        (0 <= ap)
        & (ap < annotation.shape[0])
        & (0 <= dv)
        & (dv < annotation.shape[1])
        & (0 <= ml)
        & (ml < annotation.shape[2])
    )
    projected = np.zeros((height, width), dtype=annotation.dtype)
    projected[valid] = annotation[ap[valid], dv[valid], ml[valid]]

    contours, _ = cv2.findContours((projected > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brain_mask = np.zeros((height, width), dtype=np.uint8)
    if contours:
        cv2.drawContours(brain_mask, contours, -1, 1, -1)
    brain_mask = brain_mask.astype(bool)
    coarse = _map_coarse(projected.astype(np.int64), coarse_keys, coarse_values)
    coarse[brain_mask & (coarse == 0)] = 8
    coarse[~brain_mask] = 0
    return projected, brain_mask, coarse


def preprocess_anatomy_target(anatomy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    matrix, size = brain_orientation_affine(mask)
    oriented_mask = cv2.warpAffine(
        mask.astype(np.uint8), matrix[:2], size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    ).astype(bool)
    oriented_anatomy = cv2.warpAffine(
        anatomy.astype(np.uint8), matrix[:2], size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    )
    y, x = np.nonzero(oriented_mask)
    center_x = (float(x.min()) + float(x.max())) / 2.0
    center_y = (float(y.min()) + float(y.max())) / 2.0
    side = max(float(x.max() - x.min()), float(y.max() - y.min())) * 1.14
    axis = np.linspace(-0.5, 0.5, POSE_IMAGE_SIZE, dtype=np.float32)
    sample_x, sample_y = np.meshgrid(center_x + axis * side, center_y + axis * side)
    target = cv2.remap(
        oriented_anatomy, sample_x, sample_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    )
    target_mask = cv2.remap(
        oriented_mask.astype(np.uint8), sample_x, sample_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    ).astype(bool)
    target[~target_mask] = 0
    return target.astype(np.int64)


class RegisteredSectionDataset(Dataset):
    def __init__(
        self,
        manifest_root: str | Path,
        atlas_folder: str | Path,
        split: str | None = "train",
        augmentation=None,
        seed: int = 0,
        views: int = 1,
        include_sealed: bool = False,
        brain_masker=automatic_brain_mask,
        include_anatomy: bool = True,
        cache_images: bool = False,
        cache_static: bool = False,
    ):
        self.root = Path(manifest_root)
        self.atlas_folder = Path(atlas_folder)
        self.augmentation = augmentation
        self.seed = int(seed)
        self.views = int(views)
        self.brain_masker = brain_masker
        self.include_anatomy = bool(include_anatomy)
        self.cache_images = bool(cache_images)
        self.cache_static = bool(cache_static)
        if self.views not in (1, 2):
            raise ValueError("Registered-section training supports one or two image views")
        if self.cache_images and (
            self.include_anatomy
            or self.augmentation is not None
            or self.views != 1
            or self.brain_masker is not automatic_brain_mask
        ):
            raise ValueError("The immutable registered cache supports production image-only evaluation")
        if self.cache_static and (
            not self.include_anatomy
            or self.cache_images
            or self.brain_masker is not automatic_brain_mask
        ):
            raise ValueError("The immutable static cache supports production anatomy-enabled training")

        datasets = _read_jsonl(self.root / "datasets.jsonl")
        specimen_splits = {}
        for record in datasets:
            specimen_id = int(record["specimen_id"])
            previous = specimen_splits.setdefault(specimen_id, record["split"])
            if previous != record["split"]:
                raise ValueError(f"Specimen {specimen_id} appears in multiple splits")
        self.datasets = {int(record["experiment_id"]): record for record in datasets}

        quality_manifest, approved_section_ids, rejected_records = (
            load_registered_image_quality_manifest(self.root)
        )
        assessed_splits = set(quality_manifest["assessed_splits"])
        sections = _read_jsonl(self.root / "sections.jsonl")
        requested_records = [
            record
            for record in sections
            if (split is None or record["split"] == split)
            and (include_sealed or record["split"] != SEALED_SPLIT)
        ]
        for record in requested_records:
            dataset = self.datasets[int(record["experiment_id"])]
            if dataset["split"] != record["split"] or int(dataset["specimen_id"]) != int(record["specimen_id"]):
                raise ValueError("Section and experiment manifests disagree on specimen split")
        self.records = [
            record
            for record in requested_records
            if record["split"] not in assessed_splits
            or int(record["section_image_id"]) in approved_section_ids
        ]
        self.quality_manifest_sha256 = quality_manifest["manifest_sha256"]
        requested_ids = {int(record["section_image_id"]) for record in requested_records}
        self.quality_rejections = {
            section_id: rejected_records[section_id]
            for section_id in rejected_records
            if section_id in requested_ids
        }

        self.annotation = None
        self.coarse_keys = None
        self.coarse_values = None
        self.cache_folder = (
            self.root / ".atlas_pose_cache" / registered_image_cache_key(self.root)
            if self.cache_images
            else None
        )
        self.static_cache_folder = (
            self.root
            / ".atlas_pose_cache"
            / registered_static_cache_key(
                self.root,
                self.atlas_folder,
                augmentation=self.augmentation,
                seed=self.seed,
                views=self.views,
            )
            / "training_static"
            if self.cache_static
            else None
        )

    def __len__(self) -> int:
        return len(self.records)

    def _load_atlas(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.annotation is None:
            self.annotation = nrrd.read(str(self.atlas_folder / "annotation_25.nrrd"))[0]
            with (self.atlas_folder / "atlas_labels.pkl").open("rb") as stream:
                atlas_labels = pickle.load(stream)
            self.coarse_keys, self.coarse_values = _coarse_lookup(self.annotation, atlas_labels)
        return self.annotation, self.coarse_keys, self.coarse_values

    def _cache_path(self, record: dict) -> Path:
        return self.cache_folder / record["split"] / f"{int(record['section_image_id'])}.npy"

    def _static_cache_path(self, record: dict) -> Path:
        return self.static_cache_folder / record["split"] / f"{int(record['section_image_id'])}.npz"

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        dataset = self.datasets[int(record["experiment_id"])]
        cache_path = self._cache_path(record) if self.cache_images else None
        static_cache_path = self._static_cache_path(record) if self.cache_static else None
        latent = observed_mask = None
        cached_anatomy = None
        if static_cache_path is not None and static_cache_path.is_file():
            with np.load(static_cache_path, allow_pickle=False) as cached:
                gray = cached["image"]
                image = torch.from_numpy(
                    np.repeat(gray[None], 3, axis=0)
                    if self.views == 1
                    else np.repeat(gray[:, None], 3, axis=1)
                )
                cached_anatomy = torch.from_numpy(cached["anatomy"].astype(np.int64))
        elif cache_path is not None and cache_path.is_file():
            gray = np.load(cache_path, allow_pickle=False)
            image = torch.from_numpy(np.repeat(gray[None], 3, axis=0))
        else:
            with Image.open(self.root / record["relative_path"]) as source:
                latent = np.asarray(source).copy()
            observed_mask = np.asarray(self.brain_masker(latent), dtype=bool)
            images = []
            for view in range(self.views):
                styled = latent
                if self.augmentation is not None:
                    rng = np.random.default_rng(np.random.SeedSequence((self.seed, index, view)))
                    styled = np.asarray(self.augmentation(latent.copy(), rng))
                    if styled.shape[:2] != latent.shape[:2]:
                        raise ValueError("Registered-section augmentation must preserve image geometry")
                images.append(torch.from_numpy(preprocess_atlas_pose_image(styled, observed_mask)))
            image = images[0] if self.views == 1 else torch.stack(images)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npy")
                np.save(temporary, image[0].numpy(), allow_pickle=False)
                os.replace(temporary, cache_path)

        item = {
            "image": image,
            "pose": torch.tensor(
                [record["ap_um"], record["tilt_lr_deg"], record["tilt_dv_deg"]], dtype=torch.float32
            ),
            "specimen_id": torch.tensor(int(record["specimen_id"]), dtype=torch.int64),
            "experiment_id": torch.tensor(int(record["experiment_id"]), dtype=torch.int64),
            "section_image_id": torch.tensor(int(record["section_image_id"]), dtype=torch.int64),
            "in_training_ap_domain": torch.tensor(
                bool(record.get("in_training_ap_domain", -4500.0 <= float(record["ap_um"]) <= 500.0))
            ),
        }
        if self.include_anatomy:
            if cached_anatomy is None:
                if latent is None:
                    with Image.open(self.root / record["relative_path"]) as source:
                        latent = np.asarray(source).copy()
                    observed_mask = np.asarray(self.brain_masker(latent), dtype=bool)
                annotation, coarse_keys, coarse_values = self._load_atlas()
                _, _, anatomy = project_annotation(
                    annotation,
                    coarse_keys,
                    coarse_values,
                    dataset,
                    record,
                    latent.shape[:2],
                )
                cached_anatomy = torch.from_numpy(preprocess_anatomy_target(anatomy, observed_mask))
                if static_cache_path is not None:
                    static_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = static_cache_path.with_name(
                        f"{static_cache_path.stem}.{os.getpid()}.tmp.npz"
                    )
                    np.savez_compressed(
                        temporary,
                        image=(image[0] if self.views == 1 else image[:, 0]).numpy(),
                        anatomy=cached_anatomy.numpy().astype(np.uint8),
                    )
                    os.replace(temporary, static_cache_path)
            item["anatomy"] = cached_anatomy
        return item
