import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from training.independent_joint_data import (
    INDEPENDENT_DATA_VERSION,
    PRODUCT5_CANDIDATE_SCHEDULE,
    IndependentProduct5Data,
    IndependentSyntheticData,
    canonical_source_canvas,
    product5_candidate_offsets,
)
from training.independent_joint_model import IndependentJointModel
from training.registered_section_dataset import (
    registered_static_cache_contract,
    registered_static_cache_key,
)
from training.synthetic_registration import BREGMA_AP_INDEX, VOXEL_UM, split_ap_indices


class FakeRenderer:
    device = torch.device("cpu")
    contract = {
        "contract_sha256": "1" * 64,
        "generator_source_sha256": "2" * 64,
        "average_template_sha256": "3" * 64,
        "annotation_sha256": "4" * 64,
    }

    def render_planes(self, ap_index, tilt_lr_deg, tilt_dv_deg):
        batch = len(ap_index)
        pose_value = (
            ap_index[:, None, None, None] / 1000.0
            + tilt_lr_deg[:, None, None, None] / 100.0
            + tilt_dv_deg[:, None, None, None] / 1000.0
        )
        image = pose_value.expand(batch, 1, 320, 464).float()
        mask = torch.ones_like(image, dtype=torch.bool)
        labels = torch.round(ap_index)[:, None, None, None].long().expand_as(image)
        return image, mask, labels


class FakeSyntheticGenerator(FakeRenderer):
    def make_manifest(self, count, split, seed, stratum, *, _final_capability=None):
        del _final_capability
        pool = split_ap_indices(split)
        rng = np.random.default_rng(seed)
        ap_index = rng.choice(pool, count).astype(np.float32)
        return {
            "contract_sha256": self.contract["contract_sha256"],
            "manifest_sha256": f"manifest-{split}-{seed}-{stratum}-{count}",
            "split": split,
            "seed": seed,
            "stratum": stratum,
            "ap_index": ap_index,
            "ap_um": ((BREGMA_AP_INDEX - ap_index) * VOXEL_UM).astype(np.float32),
            "tilt_lr_deg": rng.uniform(-10, 10, count).astype(np.float32),
            "tilt_dv_deg": rng.uniform(-10, 10, count).astype(np.float32),
            "rotation_deg": np.zeros(count, np.float32),
            "scale": np.ones(count, np.float32),
            "translation_xy": np.zeros((count, 2), np.float32),
        }

    def batch(self, manifest, *, qa=False, _final_capability=None):
        del qa, _final_capability
        pose = torch.as_tensor(
            np.column_stack(
                (
                    manifest["ap_index"],
                    manifest["tilt_lr_deg"],
                    manifest["tilt_dv_deg"],
                )
            ),
            dtype=torch.float32,
        )
        fixed, mask, labels = self.render_planes(pose[:, 0], pose[:, 1], pose[:, 2])
        batch = len(pose)
        y, x = torch.meshgrid(torch.arange(320), torch.arange(464), indexing="ij")
        identity = torch.stack((x, y)).float()[None].expand(batch, -1, -1, -1)
        velocity = torch.zeros_like(identity)
        homography = torch.eye(3)[None].expand(batch, -1, -1).clone()
        return {
            "fixed": fixed,
            "moving": fixed + 0.1,
            "moving_raw_uint8": torch.full_like(fixed, 120, dtype=torch.uint8),
            "fixed_mask": mask,
            "moving_model_mask": mask,
            "moving_brush_mask": mask,
            "moving_tissue_mask": mask,
            "moving_damage_mask": torch.zeros_like(mask),
            "fixed_labels": labels,
            "moving_labels": labels + 1,
            "local_velocity": velocity,
            "fixed_to_moving": identity,
            "moving_to_fixed": identity,
            "similarity_h": homography,
            "fixed_visible_mask": mask,
            "moving_visible_mask": mask,
        }


class DamagedFakeSyntheticGenerator(FakeSyntheticGenerator):
    def batch(self, manifest, *, qa=False, _final_capability=None):
        pair = super().batch(
            manifest, qa=qa, _final_capability=_final_capability
        )
        region = (slice(None), slice(None), slice(120, 180), slice(200, 260))
        pair["moving_damage_mask"] = pair["moving_damage_mask"].clone()
        pair["moving_damage_mask"][region] = True
        pair["moving_visible_mask"] = pair["moving_visible_mask"].clone()
        pair["moving_visible_mask"][region] = False
        pair["fixed_visible_mask"] = pair["fixed_visible_mask"].clone()
        pair["fixed_visible_mask"][region] = False
        return pair


class FakeRegisteredDataset:
    quality_manifest_sha256 = "5" * 64
    brain_masker = staticmethod(lambda image: np.asarray(image) > 0)

    def __init__(self, split="train", overlap=False, wrong_product=False):
        validation_specimen = 10 if overlap else 30
        self.datasets = {
            1: {
                "split": "train",
                "specimen_id": 10,
                "product_ids": [8 if wrong_product else 5],
            },
            2: {"split": "train", "specimen_id": 20, "product_ids": [5]},
            3: {
                "split": "validation",
                "specimen_id": validation_specimen,
                "product_ids": [5],
            },
            4: {"split": "validation", "specimen_id": 40, "product_ids": [5]},
        }
        records = {
            "train": [
                self._record("train", 1, 10, 101, -1000.0),
                self._record("train", 1, 10, 102, -1500.0),
                self._record("train", 2, 20, 201, -2000.0),
            ],
            "validation": [
                self._record("validation", 3, validation_specimen, 301, -1250.0),
                self._record("validation", 4, 40, 401, -1750.0),
            ],
        }
        self.records = records[split]

    @staticmethod
    def _record(split, experiment, specimen, section, ap):
        return {
            "split": split,
            "experiment_id": experiment,
            "specimen_id": specimen,
            "section_image_id": section,
            "ap_um": ap,
            "tilt_lr_deg": 2.0,
            "tilt_dv_deg": -3.0,
        }

    def __getitem__(self, index):
        y, x = np.ogrid[:48, :64]
        mask = ((x - 32) / 24) ** 2 + ((y - 24) / 17) ** 2 <= 1
        image = np.zeros((48, 64), np.uint8)
        image[mask] = 80 + 10 * index
        return {"raw_image": image, "raw_mask": mask}


def _assert_nested_equal(first, second):
    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], dict):
            _assert_nested_equal(first[key], second[key])
        elif isinstance(first[key], np.ndarray):
            assert np.array_equal(first[key], second[key]), key
        else:
            assert first[key] == second[key], key


def test_synthetic_manifest_is_deterministic_and_exactly_split():
    data = IndependentSyntheticData(FakeSyntheticGenerator())
    first = data.make_manifest(16, "train", 9182, "hard", 6)
    repeated = data.make_manifest(16, "train", 9182, "hard", 6)
    changed = data.make_manifest(16, "train", 9183, "hard", 6)
    _assert_nested_equal(first, repeated)
    assert first["manifest_sha256"] != changed["manifest_sha256"]
    split = set(split_ap_indices("train").tolist())
    true_index = np.rint(BREGMA_AP_INDEX - first["true_pose"][:, 0] / VOXEL_UM).astype(int)
    wrong_pose = first["true_pose"][:, None] + first["wrong_candidate_offset"]
    wrong_index = np.rint(BREGMA_AP_INDEX - wrong_pose[:, :, 0] / VOXEL_UM).astype(int)
    assert set(true_index.tolist()) <= split
    assert set(wrong_index.ravel().tolist()) <= split


def test_synthetic_batch_preserves_exact_pose_svf_map_and_label_semantics():
    data = IndependentSyntheticData(FakeSyntheticGenerator())
    batch = data.generate(2, "validation", 761, "mild", 6)
    assert batch["source_image"].shape == batch["source_mask"].shape == (2, 1, 320, 464)
    assert batch["truth_svf"].shape == batch["truth_fixed_to_source_map"].shape == (
        2,
        2,
        320,
        464,
    )
    assert torch.count_nonzero(batch["truth_svf"]) == 0
    y, x = torch.meshgrid(torch.arange(320), torch.arange(464), indexing="ij")
    identity = torch.stack((x, y)).float()[None].expand(2, -1, -1, -1)
    expected_forward = torch.einsum(
        "bij,bjhw->bihw",
        batch["truth_source_view_h"][:, :2],
        torch.cat((identity, torch.ones_like(identity[:, :1])), dim=1),
    )
    assert torch.allclose(batch["truth_fixed_to_source_map"], expected_forward, atol=1e-5)
    inverse_h = torch.linalg.inv(batch["truth_source_view_h"])
    expected_inverse = torch.einsum(
        "bij,bjhw->bihw",
        inverse_h[:, :2],
        torch.cat((identity, torch.ones_like(identity[:, :1])), dim=1),
    )
    inside = (
        (expected_inverse[:, 0] >= 0.0)
        & (expected_inverse[:, 0] <= 463.0)
        & (expected_inverse[:, 1] >= 0.0)
        & (expected_inverse[:, 1] <= 319.0)
    )
    assert torch.allclose(
        batch["truth_source_to_fixed_map"].permute(0, 2, 3, 1)[inside],
        expected_inverse.permute(0, 2, 3, 1)[inside],
        atol=6e-5,
    )
    assert torch.allclose(
        batch["truth_similarity_h"], batch["truth_source_view_h"], atol=1e-6
    )
    assert batch["truth_similarity_parameters"].shape == (2, 5)
    assert batch["truth_generator_similarity_parameters"].shape == (2, 4)
    parameters = batch["truth_similarity_parameters"]
    matrix = parameters[:, 4].exp()[:, None, None] * torch.stack(
        (
            parameters[:, 0],
            -parameters[:, 1],
            parameters[:, 1],
            parameters[:, 0],
        ),
        dim=1,
    ).reshape(-1, 2, 2)
    center = torch.tensor(((464 - 1.0) / 2.0, (320 - 1.0) / 2.0))
    reconstructed = torch.eye(3)[None].repeat(2, 1, 1)
    reconstructed[:, :2, :2] = matrix
    reconstructed[:, :2, 2] = (
        center
        + parameters[:, 2:4]
        - torch.einsum("bij,j->bi", matrix, center)
    )
    assert torch.allclose(reconstructed, batch["truth_similarity_h"], atol=2e-5)
    assert torch.equal(batch["candidate_pose"][:, 0], batch["true_pose"])
    assert torch.equal(batch["candidate_fixed_labels"][:, 0], batch["truth_fixed_labels"])
    for item in range(2):
        observed = torch.unique(batch["truth_source_labels"][item])
        observed = observed[observed != 0]
        expected = batch["truth_fixed_labels"][item, 0, 0, 0] + 1
        assert observed.tolist() == [int(expected)]
    assert batch["dense_truth_valid"].all()
    assert batch["candidate_dense_truth_valid"][:, 0].all()
    assert not batch["candidate_dense_truth_valid"][:, 1:].any()
    assert batch["listwise_positive_mask"].sum(1).tolist() == [1, 1]
    assert batch["listwise_target_index"].tolist() == [0, 0]


def test_source_view_manifest_spans_raw_rotation_and_scale_and_is_hash_bound():
    data = IndependentSyntheticData(FakeSyntheticGenerator())
    manifest = data.make_manifest(2048, "train", 1201, "hard", 3)
    rotation = manifest["source_view_rotation_deg"]
    scale = manifest["source_view_scale"]
    assert -180.0 <= rotation.min() < -179.8
    assert 179.8 < rotation.max() <= 180.0
    assert 0.5 <= scale.min() < 0.501
    assert 1.499 < scale.max() <= 1.5
    changed = dict(manifest)
    changed["source_view_scale"] = scale.copy()
    changed["source_view_scale"][0] += 0.01
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        data.batch(changed)


def test_outline_curriculum_has_three_hash_bound_modes_and_preserves_truth_masks():
    data = IndependentSyntheticData(FakeSyntheticGenerator())
    manifest = data.make_manifest(3, "train", 1208, "hard", 3)
    assert set(manifest["outline_plan"]["mode"].tolist()) == {0, 1, 2}
    assert len(set(manifest["outline_plan"]["sample_receipt_sha256"])) == 3
    batch = data.batch(manifest)
    assert set(batch["input_outline_mode"].tolist()) == {0, 1, 2}
    assert batch["mask_available"].shape == (3, 1, 1, 1)
    assert torch.equal(
        batch["mask_available"].flatten().bool(), batch["input_outline_mode"] != 2
    )
    for item in range(3):
        if bool(batch["mask_available"][item, 0, 0, 0]):
            assert torch.count_nonzero(
                batch["source_image"][item][~batch["source_mask"][item]]
            ) == 0
            assert torch.isfinite(batch["input_outline_quality_iou"][item])
        else:
            assert not batch["source_mask"][item].any()
            assert batch["source_image"][item].any()
            assert torch.isnan(batch["input_outline_quality_iou"][item])
    assert "truth_source_tissue_mask" in batch
    assert "truth_source_damage_mask" in batch
    assert "truth_source_valid_mask" in batch
    assert not torch.equal(batch["source_mask"], batch["truth_source_valid_mask"])


def test_outline_curriculum_uses_frozen_35_35_30_largest_remainder_mix():
    data = IndependentSyntheticData(FakeSyntheticGenerator())
    plan = data.make_manifest(1000, "train", 994, "mild", 3)["outline_plan"]
    assert plan["mode_counts"].tolist() == [350, 350, 300]
    np.testing.assert_allclose(plan["mode_probabilities"], (0.35, 0.35, 0.30))
    assert np.bincount(plan["mode"], minlength=3).tolist() == [350, 350, 300]


def test_anatomical_damage_stays_separate_and_is_never_a_dense_valid_target():
    data = IndependentSyntheticData(DamagedFakeSyntheticGenerator())
    batch = data.generate(3, "train", 2014, "hard", 3)
    assert batch["truth_source_damage_mask"].any()
    assert not (
        batch["truth_source_damage_mask"] & batch["truth_source_valid_mask"]
    ).any()
    assert not batch["truth_fixed_valid_mask"][:, :, 120:180, 200:260].any()
    assert batch["dense_truth_valid"].all()


@pytest.mark.parametrize("split", ("train", "validation"))
def test_high_tilt_positive_stream_spans_modes_magnitudes_and_balanced_signs(split):
    data = IndependentSyntheticData(FakeSyntheticGenerator())
    manifest = data.make_manifest(
        90,
        split,
        881,
        "hard",
        3,
        pose_regime="high_tilt",
    )
    pose = manifest["true_pose"]
    modes = manifest["high_tilt_mode"]
    assert set(modes.tolist()) == {0, 1, 2}
    assert np.all(pose[modes == 0, 2] == 0.0)
    assert np.all(pose[modes == 1, 1] == 0.0)
    assert np.all(pose[modes == 2, 1:] != 0.0)
    for axis in (1, 2):
        active = pose[:, axis] != 0.0
        magnitude = np.abs(pose[active, axis])
        assert 15.0 <= magnitude.min() < 15.5
        assert 34.5 < magnitude.max() <= 35.0
        assert int((pose[active, axis] < 0).sum()) == int(
            (pose[active, axis] > 0).sum()
        )
    batch = data.batch(manifest)
    assert batch["pose_regime"] == "high_tilt"
    assert torch.equal(batch["high_tilt_mode"], torch.from_numpy(modes))


def test_product5_offsets_are_sign_balanced_and_cover_the_frozen_schedule():
    observed = []
    for step, expected in enumerate(PRODUCT5_CANDIDATE_SCHEDULE):
        offsets, level = product5_candidate_offsets(3, step)
        observed.append((level["name"], level["ap_um"], level["tilt_deg"]))
        assert offsets.shape == (3, 6, 3)
        np.testing.assert_array_equal(offsets[:, 0, 0], -offsets[:, 1, 0])
        np.testing.assert_array_equal(offsets[:, 2, 1], -offsets[:, 3, 1])
        np.testing.assert_array_equal(offsets[:, 4, 2], -offsets[:, 5, 2])
        assert np.all(np.count_nonzero(offsets, axis=2) == 1)
        assert level["name"] == expected[0]
    assert tuple(observed) == PRODUCT5_CANDIDATE_SCHEDULE
    assert product5_candidate_offsets(1, len(PRODUCT5_CANDIDATE_SCHEDULE))[1]["name"] == "nearest"


def test_product5_contract_preserves_animal_split_provenance_and_balance(tmp_path):
    train = IndependentProduct5Data(
        tmp_path, tmp_path, FakeRenderer(), dataset=FakeRegisteredDataset("train")
    )
    validation = IndependentProduct5Data(
        tmp_path,
        tmp_path,
        FakeRenderer(),
        split="validation",
        dataset=FakeRegisteredDataset("validation"),
    )
    assert train.contract["version"] == INDEPENDENT_DATA_VERSION
    assert train.contract["product_ids"] == [5]
    assert train.contract["record_count"] == 3
    provenance = train.provenance_manifest()
    assert [record["animal_id"] for record in provenance["record_identities"]] == [
        10,
        10,
        20,
    ]
    assert [record["section_image_id"] for record in provenance["record_identities"]] == [
        101,
        102,
        201,
    ]
    assert all(
        len(record["section_record_sha256"]) == 64
        and len(record["experiment_record_sha256"]) == 64
        for record in provenance["record_identities"]
    )
    assert len(provenance["manifest_sha256"]) == 64
    assert train.contract["specimen_ids"] == [10, 20]
    assert validation.contract["specimen_ids"] == [30, 40]
    assert set(train.contract["specimen_ids"]).isdisjoint(validation.contract["specimen_ids"])
    np.testing.assert_allclose(train.sampling_weights, (0.25, 0.25, 0.50))
    assert train.contract["quality_manifest_sha256"] == "5" * 64
    assert train.contract["learned_checkpoint_dependencies"] == []
    assert len(train.contract["contract_sha256"]) == 64


def test_product5_batch_has_singleton_labels_and_no_dense_correspondence_truth(tmp_path):
    data = IndependentProduct5Data(
        tmp_path, tmp_path, FakeRenderer(), dataset=FakeRegisteredDataset("train")
    )
    first = data.batch_positions([0, 2], 81, schedule_step=1)
    repeated = data.batch_positions([0, 2], 81, schedule_step=1)
    for key in (
        "source_image",
        "source_mask",
        "true_pose",
        "candidate_pose",
        "candidate_fixed_image",
        "candidate_fixed_mask",
        "candidate_fixed_labels",
        "listwise_target_index",
        "listwise_positive_mask",
    ):
        assert torch.equal(first[key], repeated[key]), key
    assert first["source_image"].shape == first["source_mask"].shape == (2, 1, 320, 464)
    assert first["candidate_pose"].shape == (2, 7, 3)
    assert first["candidate_level_name"] == "resolvable-boundary"
    assert first["listwise_positive_mask"].sum(1).tolist() == [1, 1]
    assert first["listwise_target_index"].tolist() == [0, 0]
    assert first["data_split"] == "train"
    assert first["product_id"].tolist() == [5, 5]
    assert first["animal_id"].tolist() == [10, 20]
    assert first["specimen_id"].tolist() == [10, 20]
    assert first["experiment_id"].tolist() == [1, 2]
    assert first["section_image_id"].tolist() == [101, 201]
    assert len(first["record_provenance_sha256"]) == 2
    assert all(len(value) == 64 for value in first["record_provenance_sha256"])
    assert len(first["batch_manifest_sha256"]) == 64
    assert first["batch_manifest_sha256"] == repeated["batch_manifest_sha256"]
    assert not first["dense_truth_valid"].any()
    assert not first["candidate_dense_truth_valid"].any()
    forbidden_truth = {
        "truth_svf",
        "truth_fixed_to_source_map",
        "truth_source_to_fixed_map",
        "truth_source_labels",
        "truth_fixed_labels",
    }
    assert forbidden_truth.isdisjoint(first)


def test_product5_outline_modes_keep_smart_brush_optional_and_receipted(tmp_path):
    data = IndependentProduct5Data(
        tmp_path, tmp_path, FakeRenderer(), dataset=FakeRegisteredDataset("train")
    )
    batch = data.batch_positions([0, 1, 2], 811, schedule_step=2)
    assert set(batch["input_outline_mode"].tolist()) == {0, 1, 2}
    assert batch["mask_available"].shape == (3, 1, 1, 1)
    assert torch.equal(
        batch["mask_available"].flatten().bool(), batch["input_outline_mode"] != 2
    )
    assert len(batch["input_outline_receipt_sha256"]) == 3
    assert all(len(value) == 64 for value in batch["input_outline_receipt_sha256"])
    for item in range(3):
        if bool(batch["mask_available"][item, 0, 0, 0]):
            assert not batch["source_image"][item][~batch["source_mask"][item]].any()
            assert torch.isfinite(batch["input_outline_quality_iou"][item])
        else:
            assert not batch["source_mask"][item].any()
            assert batch["source_image"][item].any()
            assert torch.isnan(batch["input_outline_quality_iou"][item])
    assert data.contract["benchmark_primary_mask_policy"] == (
        "no-user-mask-or-common-automatic-mask"
    )
    assert data.contract["benchmark_assisted_mask_policy"] == (
        "smart-brush-reported-separately"
    )

    model = IndependentJointModel(
        pyramid_channels=(8, 8, 8, 8),
        pose_context_features=24,
        pair_features=16,
        hidden_channels=16,
        integration_steps=3,
    ).eval()
    with torch.no_grad():
        outputs = model.initialize(
            batch["source_image"], batch["source_mask"], batch["mask_available"]
        )
    assert outputs["pose"].shape == (3, 3)
    assert torch.isfinite(outputs["pose"]).all()


def test_product5_reuses_hash_matched_static_outline_cache(tmp_path):
    dataset = FakeRegisteredDataset("train")
    y, x = np.ogrid[:48, :64]
    mask = ((x - 32) / 24) ** 2 + ((y - 24) / 17) ** 2 <= 1
    image = np.zeros((48, 64), np.uint8)
    image[mask] = 100
    for record in dataset.records:
        record["relative_path"] = f"{record['section_image_id']}.png"
        Image.fromarray(image).save(tmp_path / record["relative_path"])
    cache_key = registered_static_cache_key(tmp_path, tmp_path)
    cache_root = tmp_path / ".atlas_pose_cache" / cache_key
    cache_root.mkdir(parents=True)
    (cache_root / "contract.json").write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "contract": registered_static_cache_contract(tmp_path, tmp_path),
            }
        ),
        encoding="utf-8",
    )
    for record in dataset.records:
        path = cache_root / "training_static" / "train" / f"{record['section_image_id']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mask=mask.astype(np.uint8))
    dataset.brain_masker = lambda image: (_ for _ in ()).throw(
        AssertionError("static outline was not reused")
    )
    data = IndependentProduct5Data(
        tmp_path, tmp_path, FakeRenderer(), dataset=dataset
    )
    assert data.static_mask_cache_folder == cache_root / "training_static"
    batch = data.batch_positions([0, 1, 2], 909, schedule_step=0)
    assert batch["source_image"].shape == (3, 1, 320, 464)


def test_canonical_canvas_preserves_masked_grayscale_and_orientation():
    image = np.zeros((50, 80, 3), np.uint8)
    mask = np.zeros((50, 80), bool)
    mask[10:40, 20:70] = True
    image[mask] = (30, 60, 90)
    canvas, canvas_mask = canonical_source_canvas(image, mask)
    assert canvas.shape == canvas_mask.shape == (1, 320, 464)
    assert canvas.dtype == torch.float32 and canvas_mask.dtype == torch.bool
    assert canvas_mask.any()
    assert float(canvas[canvas_mask].median()) == pytest.approx(60.0 / 255.0, abs=2e-3)
    assert ((canvas > 0.0) & (canvas < 0.95 * 60.0 / 255.0) & canvas_mask).any()
    assert not canvas_mask[:, :, :4].any()
    assert not canvas_mask[:, :, -4:].any()


@pytest.mark.parametrize(
    ("dataset", "message"),
    (
        (FakeRegisteredDataset("train", overlap=True), "overlap"),
        (FakeRegisteredDataset("train", wrong_product=True), "another Allen product"),
    ),
)
def test_product5_adapter_rejects_specimen_leakage_and_other_products(
    tmp_path, dataset, message
):
    with pytest.raises(RuntimeError, match=message):
        IndependentProduct5Data(tmp_path, tmp_path, FakeRenderer(), dataset=dataset)


def test_data_module_has_no_legacy_model_import_or_checkpoint_loader():
    path = Path(__file__).resolve().parents[1] / "training" / "independent_joint_data.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    forbidden = {
        "training.atlas_pose_models",
        "training.atlas_pose_models_v7",
        "training.dense_registration_model",
        "training.joint_pose_registration_model",
        "source.atlas_pose_model",
    }
    assert imports.isdisjoint(forbidden)
    assert "torch.load" not in source
    synthetic = IndependentSyntheticData(FakeSyntheticGenerator())
    assert synthetic.contract["learned_checkpoint_dependencies"] == []
