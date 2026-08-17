import copy
import json

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from source.atlas_pose_runtime import brain_mask_affine
from training.joint_registered_data import (
    JointRegisteredData,
    canonical_registration_image,
    mask_affine_homography,
    mask_normalized_moving,
)
from source.dense_registration_preprocessing import numpy_cosine_mask_feather
from training.registered_section_dataset import (
    registered_static_cache_contract,
    registered_static_cache_key,
)


class FakeRegisteredDataset:
    quality_manifest_sha256 = "a" * 64

    def __init__(self, overlap=False, wrong_product=False, split="train"):
        self.root = None
        train_records = [
            {
                "split": "train",
                "experiment_id": 1,
                "specimen_id": 10,
                "section_image_id": 101,
                "ap_um": -1000.0,
            },
            {
                "split": "train",
                "experiment_id": 1,
                "specimen_id": 10,
                "section_image_id": 102,
                "ap_um": -1500.0,
            },
            {
                "split": "train",
                "experiment_id": 2,
                "specimen_id": 20,
                "section_image_id": 201,
                "ap_um": -2000.0,
            },
        ]
        validation_records = [
            {
                "split": "validation",
                "experiment_id": 3,
                "specimen_id": 10 if overlap else 30,
                "section_image_id": 301,
                "ap_um": -1250.0,
            },
            {
                "split": "validation",
                "experiment_id": 4,
                "specimen_id": 40,
                "section_image_id": 401,
                "ap_um": -1750.0,
            },
        ]
        self.records = train_records if split == "train" else validation_records
        self.datasets = {
            1: {
                "split": "train",
                "specimen_id": 10,
                "product_ids": [8 if wrong_product else 5],
            },
            2: {"split": "train", "specimen_id": 20, "product_ids": [5]},
            3: {
                "split": "validation",
                "specimen_id": 10 if overlap else 30,
                "product_ids": [5],
            },
            4: {"split": "validation", "specimen_id": 40, "product_ids": [5]},
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        y, x = np.ogrid[:48, :64]
        mask = ((x - 32) / 24) ** 2 + ((y - 24) / 17) ** 2 <= 1
        image = np.zeros((48, 64), np.uint8)
        image[mask] = 80 + 10 * index
        return {
            "image": torch.full((3, 299, 299), 0.1 * (index + 1)),
            "pose": torch.tensor((record["ap_um"], 1.0, -2.0)),
            "specimen_id": torch.tensor(record["specimen_id"]),
            "section_image_id": torch.tensor(record["section_image_id"]),
            "raw_image": image,
            "raw_mask": mask,
        }


class FakeGenerator:
    device = torch.device("cpu")


class FakeJointSyntheticData:
    generator = FakeGenerator()

    @staticmethod
    def render_pose(pose):
        batch = len(pose)
        value = pose[:, :1].view(batch, 1, 1, 1) / 4500.0
        image = value.expand(batch, 1, 320, 464)
        mask = torch.ones_like(image, dtype=torch.bool)
        labels = torch.ones_like(image, dtype=torch.long)
        return image, mask, labels


def make_data(tmp_path, dataset=None, split="train"):
    return JointRegisteredData(
        tmp_path,
        tmp_path,
        FakeJointSyntheticData(),
        split=split,
        dataset=dataset or FakeRegisteredDataset(split=split),
    )


def test_product5_contract_is_specimen_split_and_content_bound(tmp_path):
    data = make_data(tmp_path)
    assert data.contract["split"] == "train"
    assert data.contract["product_ids"] == [5]
    assert data.contract["specimen_ids"] == [10, 20]
    assert data.contract["record_count"] == 3
    assert len(data.contract["contract_sha256"]) == 64
    assert data.sampling_weights[0] == pytest.approx(0.25)
    assert data.sampling_weights[1] == pytest.approx(0.25)
    assert data.sampling_weights[2] == pytest.approx(0.50)


def test_registered_batches_are_deterministic_and_have_no_dense_targets(tmp_path):
    data = make_data(tmp_path)
    first = data.generate(2, 9182, 3)
    second = data.generate(2, 9182, 3)
    for name in (
        "pose_image",
        "true_pose",
        "moving",
        "moving_model_mask",
        "initial_pose",
        "wrong_candidate_pose",
        "section_image_id",
    ):
        assert torch.equal(first[name], second[name])
    assert first["moving"].shape == (2, 1, 320, 464)
    assert first["wrong_candidate_pose"].shape == (2, 3, 3)
    offsets = first["wrong_candidate_pose"] - first["true_pose"][:, None]
    assert torch.all(offsets[:, 0, 0].abs() == 25.0)
    assert torch.all(offsets[:, 0, 1:] == 0.0)
    assert torch.all(offsets[:, 1, 1].abs() == 0.25)
    assert torch.all(offsets[:, 1, (0, 2)] == 0.0)
    assert torch.all(offsets[:, 2, 2].abs() == 0.25)
    assert torch.all(offsets[:, 2, :2] == 0.0)
    assert not any("dense_target" in name for name in first)
    assert "fixed_to_moving" not in first
    assert "moving_labels" not in first


def test_registered_moving_brain_is_outline_affined_to_each_candidate(tmp_path):
    data = make_data(tmp_path)
    batch = data.generate(1, 9182, 3)
    y, x = torch.meshgrid(torch.arange(320), torch.arange(464), indexing="ij")
    target = ((((x - 232) / 170) ** 2 + ((y - 160) / 120) ** 2) <= 1)[
        None, None
    ]
    _, aligned_mask, receipt = data.moving_for_fixed(batch, target)
    assert receipt["source_to_aligned_h"].shape == (1, 3, 3)
    target_y, target_x = torch.where(target[0, 0])
    aligned_y, aligned_x = torch.where(aligned_mask[0, 0])
    assert abs(int(target_x.max() - target_x.min()) - int(aligned_x.max() - aligned_x.min())) <= 3
    assert abs(int(target_y.max() - target_y.min()) - int(aligned_y.max() - aligned_y.min())) <= 3


def test_torch_mask_affine_matches_runtime_brain_mask_affine():
    y, x = np.ogrid[:320, :464]
    source = (((x - 208) / 130) ** 2 + ((y - 145) / 92) ** 2 <= 1)
    source &= (x + 0.28 * y) > 105
    target = (((x - 244) / 162) ** 2 + ((y - 166) / 118) ** 2 <= 1)
    observed = mask_affine_homography(
        torch.from_numpy(source)[None, None], torch.from_numpy(target)[None, None]
    )[0].numpy()
    expected = brain_mask_affine(source, target)
    assert np.allclose(observed, expected, atol=2e-4)


def test_product5_candidate_canvas_matches_runtime_raw_uint8_feather_contract():
    y, x = np.ogrid[:120, :180]
    source_mask = ((x - 87) / 72) ** 2 + ((y - 58) / 45) ** 2 <= 1
    source_image = np.clip(35 + 1.1 * x + 0.35 * y, 0, 255).astype(np.uint8)
    yy, xx = np.ogrid[:320, :464]
    target_mask = (((xx - 241) / 174) ** 2 + ((yy - 161) / 122) ** 2 <= 1)

    moving, moving_mask = canonical_registration_image(source_image, source_mask)
    observed, observed_mask, _, _ = mask_normalized_moving(
        moving[None], moving_mask[None], torch.from_numpy(target_mask)[None, None],
        apply_cosine_feather=True,
    )
    direct_h = brain_mask_affine(source_mask, target_mask)
    direct_mask = cv2.warpAffine(
        source_mask.astype(np.uint8), direct_h[:2], (464, 320), flags=cv2.INTER_NEAREST
    ).astype(bool)
    expected = cv2.warpAffine(
        source_image, direct_h[:2], (464, 320), flags=cv2.INTER_LINEAR
    ).astype(np.float32) / 255.0
    expected *= numpy_cosine_mask_feather(direct_mask)
    observed_np = observed[0, 0].numpy()
    intersection = np.logical_and(observed_mask[0, 0].numpy(), direct_mask).sum()
    union = np.logical_or(observed_mask[0, 0].numpy(), direct_mask).sum()
    assert intersection / union > 0.985
    assert np.mean(np.abs(observed_np[target_mask] - expected[target_mask])) < 0.012


def test_validation_subset_is_fixed_specimen_disjoint_and_balanced(tmp_path):
    train = make_data(tmp_path, split="train")
    validation = make_data(tmp_path, split="validation")
    assert set(train.contract["specimen_ids"]).isdisjoint(
        validation.contract["specimen_ids"]
    )
    first = validation.fixed_validation_positions(4, 81)
    second = validation.fixed_validation_positions(4, 81)
    assert np.array_equal(first, second)
    batch = validation.batch_positions(first, 91, 3)
    assert set(batch["specimen_id"].tolist()) == {30, 40}
    assert "fixed_to_moving" not in batch


def test_exact_static_mask_cache_is_reused_without_resegmenting(tmp_path, monkeypatch):
    dataset = FakeRegisteredDataset()
    y, x = np.ogrid[:48, :64]
    mask = ((x - 32) / 24) ** 2 + ((y - 24) / 17) ** 2 <= 1
    image = np.zeros((48, 64), np.uint8)
    image[mask] = 120
    for record in dataset.records:
        record.update(
            relative_path=f"{record['section_image_id']}.png",
            tilt_lr_deg=1.0,
            tilt_dv_deg=-2.0,
        )
        Image.fromarray(image).save(tmp_path / record["relative_path"])
    key = registered_static_cache_key(tmp_path, tmp_path)
    cache_root = tmp_path / ".atlas_pose_cache" / key
    contract = registered_static_cache_contract(tmp_path, tmp_path)
    cache_root.mkdir(parents=True)
    (cache_root / "contract.json").write_text(
        json.dumps({"cache_key": key, "contract": contract}), encoding="utf-8"
    )
    for record in dataset.records:
        path = cache_root / "training_static" / "train" / f"{record['section_image_id']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mask=mask.astype(np.uint8), anatomy=np.zeros(mask.shape, np.uint8))
    monkeypatch.setattr(
        "training.joint_registered_data.automatic_brain_mask",
        lambda image: (_ for _ in ()).throw(AssertionError("mask was recomputed")),
    )
    data = make_data(tmp_path, dataset=dataset)
    assert data.static_mask_cache_folder == cache_root / "training_static"
    batch = data.generate(1, 8177, 3)
    assert batch["moving_model_mask"].any()


@pytest.mark.parametrize(
    ("dataset", "message"),
    (
        (FakeRegisteredDataset(overlap=True), "specimens overlap"),
        (FakeRegisteredDataset(wrong_product=True), "non-Product-5"),
    ),
)
def test_registered_adapter_rejects_split_leakage_and_wrong_product(
    tmp_path, dataset, message
):
    with pytest.raises(RuntimeError, match=message):
        make_data(tmp_path, copy.deepcopy(dataset))
