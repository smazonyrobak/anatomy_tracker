import json
import pickle
from pathlib import Path

import nrrd
import numpy as np
import pytest
import torch
from PIL import Image

from source.atlas_pose_runtime import preprocess_atlas_pose_image
from training.registered_section_dataset import (
    COARSE_ANATOMY_CLASSES,
    RegisteredSectionDataset,
    _coarse_lookup,
    project_annotation,
)


def write_jsonl(path: Path, records: list[dict]):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def fixture(tmp_path: Path):
    atlas = tmp_path / "atlas"
    atlas.mkdir()
    annotation = np.zeros((4, 10, 10), dtype=np.int32)
    plane = np.full((8, 8), 688, dtype=np.int32)
    plane[1, 1:7] = [1089, 623, 549, 1097, 313, 1065]
    plane[2, 1:4] = [512, 1009, 73]
    plane[3:5, 3:5] = 0
    annotation[1, 1:9, 1:9] = plane
    nrrd.write(str(atlas / "annotation_25.nrrd"), annotation)
    labels = {
        "index": np.asarray([997, 8, 567, 688, 1089, 623, 343, 1129, 549, 1097, 313, 1065, 512, 1009, 73]),
        "parent": np.asarray([-1, 997, 8, 567, 688, 567, 8, 343, 1129, 1129, 343, 343, 8, 997, 997]),
    }
    with (atlas / "atlas_labels.pkl").open("wb") as stream:
        pickle.dump(labels, stream)

    alignment3d = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    dataset = {
        "experiment_id": 1,
        "specimen_id": 101,
        "split": "train",
        "section_thickness_um": 25.0,
        "alignment3d_tvr": alignment3d,
    }
    sealed_dataset = {
        **dataset,
        "experiment_id": 2,
        "specimen_id": 202,
        "split": "sealed_deepslice_s2p",
    }
    offset = 25.0 - 15.5 * 25.0 / 32.0
    section = {
        "section_image_id": 11,
        "experiment_id": 1,
        "specimen_id": 101,
        "split": "train",
        "section_number": 1,
        "width": 256,
        "height": 256,
        "alignment2d_tsv": [25.0 / 32.0, 0.0, 0.0, 25.0 / 32.0, offset, offset],
        "ap_um": -250.0,
        "tilt_lr_deg": 2.5,
        "tilt_dv_deg": -1.5,
        "relative_path": "images/train/1/11.jpg",
    }
    sealed_section = {
        **section,
        "section_image_id": 22,
        "experiment_id": 2,
        "specimen_id": 202,
        "split": "sealed_deepslice_s2p",
        "relative_path": "images/sealed_deepslice_s2p/2/22.jpg",
    }
    write_jsonl(tmp_path / "datasets.jsonl", [dataset, sealed_dataset])
    write_jsonl(tmp_path / "sections.jsonl", [section, sealed_section])
    image = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
    for record in (section, sealed_section):
        path = tmp_path / record["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path, format="JPEG", quality=100, subsampling=0)
    return atlas, annotation, labels, dataset, section, image


def test_downsample_pixel_centers_project_exact_mask_and_nine_anatomy_classes(tmp_path):
    atlas, annotation, labels, dataset, section, _ = fixture(tmp_path)
    del atlas
    keys, values = _coarse_lookup(annotation, labels)
    projected, mask, coarse = project_annotation(annotation, keys, values, dataset, section, (8, 8))
    assert np.array_equal(projected, annotation[1, 1:9, 1:9])
    assert mask.all()
    assert coarse[0, 0] == 1
    assert coarse[1, 1:7].tolist() == [2, 3, 4, 5, 6, 6]
    assert coarse[2, 1:4].tolist() == [7, 8, 8]
    assert np.all(coarse[3:5, 3:5] == 8)
    assert len(COARSE_ANATOMY_CLASSES) == 9


def test_dataset_uses_production_preprocessing_and_returns_deterministic_independent_views(tmp_path):
    atlas, annotation, labels, dataset_record, section_record, _ = fixture(tmp_path)

    def style(image, rng):
        noise = rng.normal(0.0, 24.0, image.shape[:2])
        return np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    dataset = RegisteredSectionDataset(tmp_path, atlas, augmentation=style, seed=71, views=2)
    item = dataset[0]
    repeat = dataset[0]
    assert item["image"].shape == (2, 3, 299, 299)
    assert not torch.equal(item["image"][0], item["image"][1])
    assert torch.equal(item["image"], repeat["image"])
    assert item["anatomy"].shape == (299, 299)
    assert item["anatomy"].dtype == torch.int64
    assert set(torch.unique(item["anatomy"]).tolist()) <= set(range(9))
    assert torch.allclose(item["pose"], torch.tensor([-250.0, 2.5, -1.5]))
    assert (item["specimen_id"].item(), item["experiment_id"].item()) == (101, 1)

    single = RegisteredSectionDataset(tmp_path, atlas, views=1)[0]
    keys, values = _coarse_lookup(annotation, labels)
    _, mask, _ = project_annotation(annotation, keys, values, dataset_record, section_record, (8, 8))
    with Image.open(tmp_path / section_record["relative_path"]) as source:
        source_image = np.asarray(source).copy()
    expected = torch.from_numpy(preprocess_atlas_pose_image(source_image, mask))
    assert torch.equal(single["image"], expected)


def test_sealed_sections_are_excluded_unless_explicitly_requested(tmp_path):
    atlas, *_ = fixture(tmp_path)
    assert len(RegisteredSectionDataset(tmp_path, atlas, split=None)) == 1
    assert len(RegisteredSectionDataset(tmp_path, atlas, split=None, include_sealed=True)) == 2
    assert len(RegisteredSectionDataset(tmp_path, atlas, split="sealed_deepslice_s2p")) == 0
    assert len(
        RegisteredSectionDataset(
            tmp_path,
            atlas,
            split="sealed_deepslice_s2p",
            include_sealed=True,
        )
    ) == 1


def test_specimen_cannot_cross_recorded_splits(tmp_path):
    atlas, *_ = fixture(tmp_path)
    records = [
        {
            "experiment_id": 1,
            "specimen_id": 101,
            "split": "train",
            "section_thickness_um": 25.0,
            "alignment3d_tvr": [0.0] * 12,
        },
        {
            "experiment_id": 2,
            "specimen_id": 101,
            "split": "validation",
            "section_thickness_um": 25.0,
            "alignment3d_tvr": [0.0] * 12,
        },
    ]
    write_jsonl(tmp_path / "datasets.jsonl", records)
    with pytest.raises(ValueError, match="multiple splits"):
        RegisteredSectionDataset(tmp_path, atlas)


def test_geometry_changing_style_callback_is_rejected(tmp_path):
    atlas, *_ = fixture(tmp_path)
    dataset = RegisteredSectionDataset(
        tmp_path,
        atlas,
        augmentation=lambda image, rng: image[:-1],
        views=2,
    )
    with pytest.raises(ValueError, match="preserve image geometry"):
        dataset[0]
