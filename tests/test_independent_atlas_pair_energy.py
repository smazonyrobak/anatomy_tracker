import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import training.run_independent_atlas_pair_energy as runner
from training.independent_atlas_pair_energy import (
    AtlasPairEnergyModel,
    atlas_pair_loss,
    parameter_count,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "training/configs/independent_oracle_atlas_pair_energy_1500.json"


def _inputs(batch=2, candidates=5, height=32, width=48):
    source = torch.rand(batch, 1, height, width)
    source_mask = torch.zeros_like(source, dtype=torch.bool)
    available = torch.zeros(batch, 1, 1, 1)
    atlas = torch.rand(batch, candidates, 1, height, width)
    atlas_mask = torch.ones_like(atlas, dtype=torch.bool)
    return source, source_mask, available, atlas, atlas_mask


def test_model_is_compact_random_pose_blind_and_candidate_order_equivariant():
    torch.manual_seed(4)
    model = AtlasPairEnergyModel().eval()
    assert parameter_count(model) == 271450 < 1_500_000
    assert model.source_stem is not model.atlas_stem
    assert not set(map(id, model.source_stem.parameters())) & set(
        map(id, model.atlas_stem.parameters())
    )
    assert "candidate_pose" not in inspect.signature(model.forward).parameters

    values = _inputs()
    output = model(*values, candidate_chunk_size=2)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = model(
        *values[:3], values[3][:, permutation], values[4][:, permutation],
        candidate_chunk_size=2,
    )
    inverse = permutation.argsort()
    for name in ("energy", "energy8", "energy16"):
        assert torch.allclose(output[name], permuted[name][:, inverse], atol=1e-6, rtol=1e-6)


def test_source_is_encoded_once_and_loss_reaches_both_stems(monkeypatch):
    model = AtlasPairEnergyModel().train()
    calls = 0
    original = model.encode_source

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "encode_source", counted)
    values = _inputs(candidates=5)
    output = model(*values, candidate_chunk_size=2)
    candidate_pose = torch.randn(2, 5, 3)
    truth = candidate_pose[:, 0].clone()
    target = torch.zeros(2, dtype=torch.long)
    losses = atlas_pair_loss(output, candidate_pose, truth, target)
    losses["total"].backward()
    assert calls == 1
    assert torch.isfinite(losses["total"])
    assert model.source_stem.level2[0].weight.grad is not None
    assert model.atlas_stem.level2[0].weight.grad is not None
    assert model.energy8[-1].weight.grad is not None
    assert model.energy16[-1].weight.grad is not None


def test_oracle_source_uses_pre_view_uint8_and_absent_masks():
    raw = torch.arange(2 * 1 * 20 * 30, dtype=torch.uint8).reshape(2, 1, 20, 30)
    source, mask, available = runner.oracle_source({"moving_raw_uint8": raw})
    expected = torch.nn.functional.interpolate(
        raw.float() / 255.0,
        runner.MODEL_INPUT_SHAPE,
        mode="bilinear",
        align_corners=False,
    )
    assert torch.equal(source, expected)
    assert source.shape == (2, 1, 160, 232)
    assert not bool(mask.any())
    assert not bool(available.any())
    with pytest.raises(RuntimeError, match="pre-view"):
        runner.oracle_source({"moving": raw.float()})


def test_candidate_table_has_exact_truth_target_unique_valid_and_no_clamping():
    truth = torch.tensor(
        [[0.0, 20.0, -20.0], [-2000.0, 0.0, 0.0]], dtype=torch.float32
    )
    poses, target, kinds = runner.candidate_pose_table(
        truth, 1004322, np.asarray([0, 7])
    )
    assert poses.shape == (2, 16, 3)
    assert target.shape == (2,)
    for row in range(2):
        assert torch.equal(poses[row, target[row]], truth[row])
        assert len(torch.unique(poses[row], dim=0)) == 16
        assert runner.AP_MIN_UM <= float(poses[row, :, 0].min())
        assert float(poses[row, :, 0].max()) <= runner.AP_MAX_UM
        assert bool((poses[row, :, 1:].abs() <= 35.0).all())
        assert kinds[row][target[row]] == "truth"
        offsets = poses[row][
            torch.as_tensor([name == "axis" for name in kinds[row]])
        ] - truth[row]
        expected = {
            (-125.0, 0.0, 0.0), (125.0, 0.0, 0.0),
            (-500.0, 0.0, 0.0), (500.0, 0.0, 0.0),
            (0.0, -2.5, 0.0), (0.0, 2.5, 0.0),
            (0.0, -10.0, 0.0), (0.0, 10.0, 0.0),
            (0.0, 0.0, -2.5), (0.0, 0.0, 2.5),
            (0.0, 0.0, -10.0), (0.0, 0.0, 10.0),
        }
        assert {tuple(value.tolist()) for value in offsets} == expected
        globals_ = poses[row][
            torch.as_tensor([name == "joint-global" for name in kinds[row]])
        ]
        assert bool(
            (((globals_ - truth[row]).abs() / torch.tensor((500.0, 10.0, 10.0))) >= 1)
            .all()
        )


_GENERATOR_PAYLOAD = {
    "average_template_sha256": "average",
    "annotation_sha256": "annotation",
    "query_sha256": "query",
}
_GENERATOR_CONTRACT = {
    **_GENERATOR_PAYLOAD,
    "contract_sha256": runner._payload_sha256(_GENERATOR_PAYLOAD),
}


class _ManifestFactory:
    contract = _GENERATOR_CONTRACT

    def make_manifest(self, count, split, seed, stratum):
        pool = np.resize(runner.split_ap_indices("train"), count).astype(np.float32)
        result = {
            "contract_sha256": self.contract["contract_sha256"],
            "seed": seed,
            "split": split,
            "stratum": stratum,
            "ap_index": pool,
            "ap_um": ((runner.BREGMA_AP_INDEX - pool) * runner.VOXEL_UM).astype(np.float32),
            "tilt_lr_deg": np.zeros(count, np.float32),
            "tilt_dv_deg": np.zeros(count, np.float32),
            "appearance": np.arange(count, dtype=np.int64),
        }
        result["manifest_sha256"] = runner._payload_sha256(result)
        return result


class _OracleFactory(_ManifestFactory):
    device = torch.device("cpu")

    def batch(self, manifest, qa=False):
        generator = torch.Generator().manual_seed(int(manifest["seed"]))
        raw = torch.randint(0, 256, (1, 1, 20, 30), generator=generator, dtype=torch.uint8)
        return {"moving_raw_uint8": raw}


def test_singleton_realizations_are_batch_order_invariant_and_distinct():
    generator = _OracleFactory()
    parent = generator.make_manifest(3, "train", 1004322, "clean")
    forward = runner.oracle_realizations(generator, parent, np.asarray([0, 1]))
    reverse = runner.oracle_realizations(generator, parent, np.asarray([1, 0]))
    separate = runner.oracle_realizations(generator, parent, np.asarray([0]))
    assert torch.equal(forward[0][0], reverse[0][1])
    assert torch.equal(forward[0][0], separate[0][0])
    assert forward[3][0] == separate[3][0]
    assert forward[3][0]["realization_seed"] != forward[3][1]["realization_seed"]
    assert forward[3][0]["synthetic_realization_id"] != forward[3][1]["synthetic_realization_id"]


def test_balanced_panel_is_six_by_eight_and_seed_committed():
    panel = runner.balanced_panel_manifest(_ManifestFactory(), 1104322)
    assert panel["seed"] == 1104322
    assert len(np.unique(panel["ap_um"])) == 6
    assert np.all(np.unique(panel["ap_um"], return_counts=True)[1] == 8)
    tilts = np.column_stack((panel["tilt_lr_deg"], panel["tilt_dv_deg"]))
    assert len(np.unique(tilts, axis=0)) == 8
    assert np.all(np.unique(tilts, axis=0, return_counts=True)[1] == 6)
    unhashed = {name: value for name, value in panel.items() if name != "manifest_sha256"}
    assert panel["manifest_sha256"] == runner._payload_sha256(unhashed)


def test_coarse_to_fine_search_is_bounded_and_reaches_fine_lattice(monkeypatch):
    target = torch.tensor([-1234.0, 6.0, -11.0])

    class Model:
        def encode_source(self, *unused):
            return ()

    class Renderer:
        device = torch.device("cpu")

    def score(unused_model, unused_renderer, unused_source, poses, unused_chunk):
        scale = poses.new_tensor((2500.0, 35.0, 35.0))
        energy = ((poses - target) / scale).square().sum(1)
        return {
            "energy": energy,
            "energy8": energy + 1,
            "energy16": energy + 2,
            "invalid_render": torch.zeros(len(poses), dtype=torch.bool),
        }

    monkeypatch.setattr(runner, "_score_pose_set", score)
    config = {
        "model": {"candidate_chunk_size": 8},
        "search": {"top_k": 3, "maximum_candidate_evaluations_per_slice": 468},
    }
    source = torch.zeros(1, 1, 16, 24)
    prediction, receipt = runner.coarse_to_fine_search(
        Model(), Renderer(), source, source.bool(), torch.zeros(1, 1, 1, 1), config
    )
    assert receipt["candidate_evaluations"] <= 468
    assert len(receipt["stages"]) == 4
    assert sum(len(value["candidate_pose"]) for value in receipt["stages"]) == receipt[
        "candidate_evaluations"
    ]
    assert receipt["nonfinite_count"] == 0
    assert receipt["invalid_render_count"] == 0
    assert abs(float(prediction[0] - target[0])) <= 78.125
    assert bool((prediction[1:] - target[1:]).abs().le(2.1875).all())


def test_qualification_manifest_requires_post_checkpoint_capability(tmp_path):
    config = {
        "data": {
            "qualification_seeds": [1204322, 1304322],
            "qualification_count_per_seed": 48,
        }
    }
    with pytest.raises(RuntimeError, match="final checkpoint"):
        runner.qualification_manifests(
            _ManifestFactory(), config, tmp_path, tmp_path / "missing.pt"
        )
    with pytest.raises(RuntimeError, match="before final freeze"):
        runner.balanced_panel_manifest(_ManifestFactory(), 1204322)


def test_freeze_capability_binds_current_source_config_and_checkpoint(tmp_path):
    config = {
        "contract_sha256": "config-contract",
        "config_file_sha256": "config-file",
        "training": {"max_updates": 1},
        "data": {
            "qualification_seeds": [1204322, 1304322],
            "qualification_count_per_seed": 48,
        },
    }
    checkpoint = tmp_path / "final_checkpoint.pt"
    lineage = {
        "generator_contract_sha256": _GENERATOR_CONTRACT["contract_sha256"],
        "generator_contract": _GENERATOR_CONTRACT,
        "atlas_sha256": _GENERATOR_PAYLOAD,
    }
    lineage["data_lineage_sha256"] = runner._canonical_sha256(lineage)
    history = [{"update": 1, "optimizer_step_applied": True}]
    torch.save(
        {
            "format": "independent-atlas-pair-energy-final-v1",
            "config_contract_sha256": "config-contract",
            "config_file_sha256": "config-file",
            "source_sha256": runner.source_hashes(),
            "learned_checkpoint_dependencies": [],
            "update": 1,
            "model": {"weight": torch.ones(1)},
            "data_lineage": lineage,
            "training_history": history,
            "training_history_sha256": runner._canonical_sha256(history),
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="before final freeze"):
        runner.verified_qualification_capability(tmp_path, config, checkpoint)
    runner.freeze_qualification(tmp_path, config, checkpoint)
    capability = runner.verified_qualification_capability(tmp_path, config, checkpoint)
    assert capability["generator_contract_sha256"] == _GENERATOR_CONTRACT["contract_sha256"]
    assert len(
        runner.qualification_manifests(
            _ManifestFactory(), config, tmp_path, checkpoint
        )
    ) == 2
    changed = torch.load(checkpoint, weights_only=False)
    changed["model"]["weight"].zero_()
    torch.save(changed, checkpoint)
    with pytest.raises(RuntimeError, match="no longer matches"):
        runner.verified_qualification_capability(tmp_path, config, checkpoint)


def test_frozen_config_is_exact_and_has_no_learned_or_protected_dependency():
    config = runner.inspect_config(CONFIG)
    assert config["learned_checkpoint_dependencies"] == []
    assert config["product5_access"] is False
    assert config["calibration_access"] is False
    assert config["final_test_access"] is False
    assert config["data"]["qualification_seeds"] == [1204322, 1304322]
