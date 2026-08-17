import csv
import json

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from training.dense_registration_model import identity_pixel_map
from training.evaluate_joint_registration_premise import (
    _production_samples,
    evaluate_joint_registration_premise,
)


class IdentityRegistrar(nn.Module):
    def forward_with_details(self, fixed, moving):
        batch, _, height, width = fixed.shape
        identity = identity_pixel_map(batch, height, width, device=fixed.device)
        return {
            "fixed_to_moving_map": identity,
            "moving_to_fixed_map": identity,
            "similarity_parameters": fixed.new_zeros((batch, 4)),
            "local_velocity": fixed.new_zeros((batch, 2, height, width)),
            "pyramid_velocities": (),
        }


class FailingRegistrar(IdentityRegistrar):
    def forward_with_details(self, fixed, moving):
        if bool((fixed[:, 0, 0, 0] < 0.0).any()):
            raise RuntimeError("deliberate candidate failure")
        return super().forward_with_details(fixed, moving)


class RecordingRegistrar(IdentityRegistrar):
    def __init__(self):
        super().__init__()
        self.moving = []

    def forward_with_details(self, fixed, moving):
        self.moving.append(moving.detach().cpu())
        return super().forward_with_details(fixed, moving)


def _shift(tensor, pixels):
    return torch.roll(tensor, shifts=pixels, dims=-1)


def _sample(sample_id="fixture"):
    height = width = 32
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    mask = (((x - 15.5) / 11.0) ** 2 + ((y - 15.5) / 9.0) ** 2 < 1.0).float()
    image = (0.2 + 0.8 * ((x + 2 * y) % 11).float() / 10.0) * mask
    moving = torch.stack((image, mask))
    wrong_one = torch.stack((_shift(image, 5), mask))
    wrong_two = torch.stack((torch.flip(image, (-2,)), mask))
    return {
        "sample_id": sample_id,
        "artifact_stratum": "controlled",
        "moving_input": moving,
        "candidate_fixed": torch.stack((moving, wrong_one, wrong_two)),
        "candidate_pose": torch.tensor(
            ((-1500.0, 1.0, -2.0), (-1450.0, 1.0, -2.0), (-1500.0, 3.0, -2.0))
        ),
        "candidate_offset": torch.tensor(
            ((0.0, 0.0, 0.0), (50.0, 0.0, 0.0), (0.0, 2.0, 0.0))
        ),
        "candidate_kind": ["true", "initial", "hard_negative"],
        "candidate_id": ["true", "initial", "hard_negative_000"],
    }


def test_true_plane_ranks_first_and_writes_diagnostic_artifacts(tmp_path):
    result = evaluate_joint_registration_premise(
        IdentityRegistrar(), [_sample()], candidate_chunk_size=2, run_folder=tmp_path
    )

    assert result["metrics"]["top1_accuracy"] == 1.0
    assert result["metrics"]["mean_reciprocal_rank"] == 1.0
    assert result["sample_metrics"][0]["true_rank"] == 1
    assert all(row["dense_flow_target_used"] is False for row in result["candidate_rows"])
    assert all(
        row["evidence_score"] > result["candidate_rows"][0]["evidence_score"]
        for row in result["candidate_rows"][1:]
    )

    summary = json.loads((tmp_path / "premise-summary.json").read_text())
    with (tmp_path / "premise-candidates.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert "not probabilities" in summary["interpretation"]
    assert len(rows) == 3
    assert rows[1]["dense_flow_target_used"] == "false"


def test_candidate_chunk_size_does_not_change_scores_or_ranks():
    one = evaluate_joint_registration_premise(
        IdentityRegistrar(), [_sample()], candidate_chunk_size=1
    )
    three = evaluate_joint_registration_premise(
        IdentityRegistrar(), [_sample()], candidate_chunk_size=3
    )

    assert one["metrics"] == three["metrics"]
    assert [row["evidence_score"] for row in one["candidate_rows"]] == pytest.approx(
        [row["evidence_score"] for row in three["candidate_rows"]]
    )


def test_candidate_outline_scale_and_center_are_normalized_before_evidence():
    sample = _sample("outline-normalization")
    moving = sample["moving_input"]
    resized = F.interpolate(
        moving[None], size=(20, 20), mode="bilinear", align_corners=True
    )[0]
    resized[1] = (resized[1] > 0.5).float()
    shifted = torch.zeros_like(moving)
    shifted[:, 3:23, 10:30] = resized
    sample["candidate_fixed"] = torch.stack((moving, shifted))
    sample["candidate_pose"] = sample["candidate_pose"][:2]
    sample["candidate_offset"] = sample["candidate_offset"][:2]
    sample["candidate_kind"] = sample["candidate_kind"][:2]
    sample["candidate_id"] = sample["candidate_id"][:2]

    result = evaluate_joint_registration_premise(
        IdentityRegistrar(), [sample], candidate_chunk_size=2
    )
    rows = result["candidate_rows"]

    assert rows[0]["outline_dice"] > 0.99
    assert rows[1]["outline_dice"] > 0.95
    assert "candidate-specific" in result["score"]["candidate_preprocessing"]


def test_candidate_canvas_affine_is_followed_by_runtime_cosine_feather():
    sample = _sample("raw-then-feather")
    mask = sample["moving_input"][1].bool()
    sample["moving_input"][0][~mask] = 0.75
    registrar = RecordingRegistrar()

    result = evaluate_joint_registration_premise(
        registrar, [sample], candidate_chunk_size=3
    )
    observed = registrar.moving[0]

    assert sample["moving_input"][0, 0, 0] == 0.75
    assert torch.all(observed[:, 0, 0, 0] == 0.0)
    assert result["format_version"] == 3
    assert "raw uint8/255" in result["score"]["candidate_preprocessing"]


def test_production_samples_use_raw_uint8_not_the_old_preprocessed_moving():
    class Generator:
        device = torch.device("cpu")

        def __init__(self):
            self.qa = None

        def batch(self, _manifest, *, qa):
            self.qa = qa
            return {
                "moving": torch.full((1, 1, 8, 8), 0.99),
                "moving_raw_uint8": torch.full((1, 1, 8, 8), 51, dtype=torch.uint8),
                "moving_model_mask": torch.ones(1, 1, 8, 8, dtype=torch.bool),
            }

        def render_planes(self, ap, lr, dv):
            count = len(ap)
            return (
                torch.zeros(count, 1, 8, 8),
                torch.ones(count, 1, 8, 8, dtype=torch.bool),
                torch.zeros(count, 1, 8, 8, dtype=torch.long),
            )

    generator = Generator()
    manifest = {
        "joint_manifest_sha256": "fixture",
        "artifact_stratum": "clean",
        "true_pose": torch.tensor(((-1500.0, 0.0, 0.0),)).numpy(),
        "initial_pose_offset": torch.tensor(((25.0, 0.0, 0.0),)).numpy(),
        "wrong_candidate_offset": torch.tensor((((50.0, 0.0, 0.0),),)).numpy(),
        "registration__manifest_sha256": "base",
    }

    sample = next(_production_samples(manifest, generator))

    assert generator.qa is True
    assert torch.allclose(sample["moving_input"][:, :1], torch.full((1, 1, 8, 8), 0.2))


def test_candidate_failures_remain_in_sample_and_metric_denominators():
    good = _sample("good")
    bad = _sample("bad")
    bad["candidate_fixed"] = bad["candidate_fixed"].clone()
    bad["candidate_fixed"][0, 0, 0, 0] = -1.0

    result = evaluate_joint_registration_premise(
        FailingRegistrar(), [good, bad], candidate_chunk_size=1
    )

    assert result["metrics"]["sample_count"] == 2
    assert result["metrics"]["complete_sample_count"] == 1
    assert result["metrics"]["failed_candidate_count"] == 1
    assert result["metrics"]["top1_accuracy"] == 0.5
    assert result["metrics"]["mean_reciprocal_rank"] == 0.5
    assert result["sample_metrics"][1]["top1"] is False
    assert result["sample_metrics"][1]["reciprocal_rank"] == 0.0


def test_wrong_plane_dense_flow_targets_are_never_read():
    class AccessTrackingDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.accessed = []

        def __getitem__(self, key):
            self.accessed.append(key)
            if "dense_target" in key:
                raise AssertionError("wrong-plane dense target was accessed")
            return super().__getitem__(key)

    sample = AccessTrackingDict(_sample())
    sample["wrong_candidate_dense_target"] = object()
    sample["wrong_candidate_dense_target_valid"] = torch.zeros(2, dtype=torch.bool)

    result = evaluate_joint_registration_premise(
        IdentityRegistrar(), [sample], candidate_chunk_size=2
    )

    assert result["metrics"]["top1_accuracy"] == 1.0
    assert not any("dense_target" in key for key in sample.accessed)
