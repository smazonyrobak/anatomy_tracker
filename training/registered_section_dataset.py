from __future__ import annotations

import json
import pickle
from pathlib import Path

import cv2
import nrrd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from source.atlas_pose_runtime import POSE_IMAGE_SIZE, brain_orientation_affine, preprocess_atlas_pose_image


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
    ):
        self.root = Path(manifest_root)
        self.augmentation = augmentation
        self.seed = int(seed)
        self.views = int(views)
        if self.views not in (1, 2):
            raise ValueError("Registered-section training supports one or two image views")

        datasets = _read_jsonl(self.root / "datasets.jsonl")
        specimen_splits = {}
        for record in datasets:
            specimen_id = int(record["specimen_id"])
            previous = specimen_splits.setdefault(specimen_id, record["split"])
            if previous != record["split"]:
                raise ValueError(f"Specimen {specimen_id} appears in multiple splits")
        self.datasets = {int(record["experiment_id"]): record for record in datasets}

        sections = _read_jsonl(self.root / "sections.jsonl")
        self.records = [
            record
            for record in sections
            if (split is None or record["split"] == split)
            and (include_sealed or record["split"] != SEALED_SPLIT)
        ]
        for record in self.records:
            dataset = self.datasets[int(record["experiment_id"])]
            if dataset["split"] != record["split"] or int(dataset["specimen_id"]) != int(record["specimen_id"]):
                raise ValueError("Section and experiment manifests disagree on specimen split")

        atlas_folder = Path(atlas_folder)
        self.annotation = nrrd.read(str(atlas_folder / "annotation_25.nrrd"))[0]
        with (atlas_folder / "atlas_labels.pkl").open("rb") as stream:
            atlas_labels = pickle.load(stream)
        self.coarse_keys, self.coarse_values = _coarse_lookup(self.annotation, atlas_labels)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        dataset = self.datasets[int(record["experiment_id"])]
        with Image.open(self.root / record["relative_path"]) as source:
            latent = np.asarray(source).copy()
        _, mask, anatomy = project_annotation(
            self.annotation,
            self.coarse_keys,
            self.coarse_values,
            dataset,
            record,
            latent.shape[:2],
        )
        anatomy = torch.from_numpy(preprocess_anatomy_target(anatomy, mask))
        images = []
        for view in range(self.views):
            styled = latent
            if self.augmentation is not None:
                rng = np.random.default_rng(np.random.SeedSequence((self.seed, index, view)))
                styled = np.asarray(self.augmentation(latent.copy(), rng))
                if styled.shape[:2] != latent.shape[:2]:
                    raise ValueError("Registered-section augmentation must preserve image geometry")
            images.append(torch.from_numpy(preprocess_atlas_pose_image(styled, mask)))
        image = images[0] if self.views == 1 else torch.stack(images)
        return {
            "image": image,
            "pose": torch.tensor(
                [record["ap_um"], record["tilt_lr_deg"], record["tilt_dv_deg"]], dtype=torch.float32
            ),
            "anatomy": anatomy,
            "specimen_id": torch.tensor(int(record["specimen_id"]), dtype=torch.int64),
            "experiment_id": torch.tensor(int(record["experiment_id"]), dtype=torch.int64),
            "section_image_id": torch.tensor(int(record["section_image_id"]), dtype=torch.int64),
        }
