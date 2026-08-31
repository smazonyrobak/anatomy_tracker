import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import training.run_independent_atlas_pair_spatial_aggregation as runner


ROOT = Path(__file__).parents[1]
CONFIG = (
    ROOT
    / "training/configs/independent_oracle_atlas_pair_spatial_aggregation_pair_1500_r1404322.json"
)


class _TinyEnergy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))
        self.bias = torch.nn.Parameter(torch.tensor(-0.1))
        self.input_ids = []

    def forward(
        self,
        source_image,
        source_mask,
        mask_available,
        candidate_image,
        candidate_mask,
        *,
        candidate_chunk_size,
    ):
        self.input_ids.append(
            tuple(
                id(value)
                for value in (
                    source_image,
                    source_mask,
                    mask_available,
                    candidate_image,
                    candidate_mask,
                )
            )
        )
        source_term = source_image.mean((1, 2, 3), keepdim=False)[:, None]
        candidate_term = candidate_image.mean((2, 3, 4))
        energy8 = self.weight * (candidate_term - source_term).square() + self.bias
        energy16 = (self.weight + 0.2) * (candidate_term + source_term).square()
        return {
            "energy": 0.5 * (energy8 + energy16),
            "energy8": energy8,
            "energy16": energy16,
        }


def _tiny_config(max_updates=2):
    config = runner.inspect_config(CONFIG)
    config = copy.deepcopy(config)
    config["training"]["amp"] = False
    config["training"]["max_updates"] = max_updates
    config["training"]["resume_every_updates"] = 1
    config["model"]["candidate_chunk_size"] = 2
    return config


def _tiny_pair(config, device=torch.device("cpu")):
    torch.manual_seed(19)
    null = _TinyEnergy().to(device)
    treatment = _TinyEnergy().to(device)
    treatment.load_state_dict(null.state_dict(), strict=True)
    models = {"null": null, "treatment": treatment}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        for name, model in models.items()
    }
    amp = config["training"]["amp"] and device.type == "cuda"
    scalers = {
        name: torch.amp.GradScaler(
            device.type,
            enabled=amp,
            init_scale=config["training"]["amp_initial_scale"],
        )
        for name in models
    }
    state = null.state_dict()
    digest = runner._state_dict_sha256(state)
    initialization = {
        "seed": 19,
        "parameter_count_each": sum(value.numel() for value in null.parameters()),
        "state_schema": list(state),
        "state_schema_sha256": runner._canonical_sha256(list(state)),
        "null_state_sha256": digest,
        "treatment_state_sha256": digest,
        "full_initial_state_equal": True,
    }
    return models, optimizers, scalers, amp, initialization


def _tiny_batch(offset=0.0, device=torch.device("cpu")):
    source = (torch.arange(16, dtype=torch.float32).reshape(2, 1, 2, 4) / 16 + offset).to(device)
    candidates = (
        torch.arange(48, dtype=torch.float32).reshape(2, 3, 1, 2, 4) / 48
        + offset
    ).to(device)
    pose = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [125.0, 2.5, 0.0], [-125.0, 0.0, 2.5]],
            [[-500.0, 0.0, 0.0], [-375.0, 2.5, 0.0], [-625.0, 0.0, 2.5]],
        ],
        device=device,
    )
    target = torch.tensor([0, 0], device=device)
    return {
        "source_image": source,
        "source_mask": torch.zeros_like(source, dtype=torch.bool),
        "mask_available": torch.zeros(2, 1, 1, 1, device=device),
        "true_pose": pose[:, 0],
        "candidate_pose": pose,
        "candidate_image": candidates,
        "candidate_mask": torch.ones_like(candidates, dtype=torch.bool),
        "target_index": target,
    }


def _actual_model_batch(offset=0.0, device=torch.device("cuda")):
    source = (
        torch.arange(2 * 32 * 48, dtype=torch.float32).reshape(2, 1, 32, 48)
        / (2 * 32 * 48)
        + offset
    ).to(device)
    candidate = (
        torch.arange(2 * 3 * 32 * 48, dtype=torch.float32).reshape(
            2, 3, 1, 32, 48
        )
        / (2 * 3 * 32 * 48)
        + offset
    ).to(device)
    pose = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [125.0, 2.5, 0.0], [-125.0, 0.0, 2.5]],
            [[-500.0, 0.0, 0.0], [-375.0, 2.5, 0.0], [-625.0, 0.0, 2.5]],
        ],
        device=device,
    )
    return {
        "source_image": source,
        "source_mask": torch.zeros_like(source, dtype=torch.bool),
        "mask_available": torch.zeros(2, 1, 1, 1, device=device),
        "true_pose": pose[:, 0],
        "candidate_pose": pose,
        "candidate_image": candidate,
        "candidate_mask": torch.ones_like(candidate, dtype=torch.bool),
        "target_index": torch.tensor([0, 0], device=device),
    }


def _history_row(update, step):
    return {"update": update, **step}


def _statistics_fixture():
    payload = {
        "levels": {
            "8": {
                "first_162_exact": True,
                "null_last_243_exact_zero": True,
                "null_statistics_sha256": "a" * 64,
                "treatment_statistics_sha256": "b" * 64,
            },
            "16": {
                "first_162_exact": True,
                "null_last_243_exact_zero": True,
                "null_statistics_sha256": "c" * 64,
                "treatment_statistics_sha256": "d" * 64,
            },
        },
        "first_162_exact_across_modes": True,
        "null_contrasts_exact_zero": True,
    }
    payload["integrity_sha256"] = runner._canonical_sha256(payload)
    return payload


def test_frozen_config_and_family_self_hash_are_strict(tmp_path):
    config = runner.inspect_config(CONFIG)
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    commitment = raw.pop("contract_sha256")
    assert commitment == runner._canonical_sha256(raw) == config["family_self_sha256"]
    assert config["data"]["qualification_seeds"] == [1604322, 1704322]
    assert config["data"]["consumed_or_forbidden_qualification_seeds"] == [
        1204322,
        1304322,
        1504322,
    ]
    assert config["product5_access"] is False
    assert config["calibration_access"] is False
    assert config["final_test_access"] is False
    assert config["learned_checkpoint_dependencies"] == []
    assert runner.source_hashes()["training/run_independent_atlas_pair_energy.py"] == (
        "21c73f88a48ca87ac0a44ff022993eea5dc2cfbb1f8c72237ec4b05fa4445b19"
    )
    assert runner.source_hashes()["training/independent_atlas_pair_energy.py"] == (
        "6187cb051d048d1e5eec3137b9edc6ac09706cecffcf989507951708681589ec"
    )
    changed = copy.deepcopy(raw)
    changed["training"]["max_updates"] = 1501
    changed["contract_sha256"] = runner._canonical_sha256(changed)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen family"):
        runner.inspect_config(path)


def test_long_run_rechecks_live_source_and_config_bytes(monkeypatch):
    config = runner.inspect_config(CONFIG)
    changed_sources = dict(config["lineage"]["source_sha256"])
    changed_sources["training/run_independent_atlas_pair_spatial_aggregation.py"] = (
        "0" * 64
    )
    monkeypatch.setattr(runner, "source_hashes", lambda: changed_sources)
    with pytest.raises(RuntimeError, match="changed during"):
        runner._assert_frozen_files(config)

    monkeypatch.undo()
    original_sha256 = runner._sha256
    monkeypatch.setattr(
        runner,
        "_sha256",
        lambda path: "0" * 64
        if Path(path).resolve() == Path(config["config_path"]).resolve()
        else original_sha256(path),
    )
    with pytest.raises(RuntimeError, match="changed during"):
        runner._assert_frozen_files(config)


def test_pair_initialization_is_fully_bit_identical_with_same_schema():
    config = runner.inspect_config(CONFIG)
    models, receipt = runner.initialize_pair(config, torch.device("cpu"))
    assert receipt["parameter_count_each"] == 271780
    assert receipt["full_initial_state_equal"]
    assert receipt["null_state_sha256"] == receipt["treatment_state_sha256"]
    assert list(models["null"].state_dict()) == list(models["treatment"].state_dict())
    for name in models["null"].state_dict():
        assert torch.equal(
            models["null"].state_dict()[name], models["treatment"].state_dict()[name]
        )


def test_first_162_statistics_match_and_null_contrasts_are_exact_zero():
    config = runner.inspect_config(CONFIG)
    models, _ = runner.initialize_pair(config, torch.device("cpu"))
    source = torch.rand(1, 1, 32, 48)
    candidates = torch.rand(1, 2, 1, 32, 48)
    batch = {
        "source_image": source,
        "source_mask": torch.zeros_like(source, dtype=torch.bool),
        "mask_available": torch.zeros(1, 1, 1, 1),
        "candidate_image": candidates,
        "candidate_mask": torch.ones_like(candidates, dtype=torch.bool),
    }
    receipt = runner.initial_statistics_integrity(models, batch)
    assert receipt["first_162_exact_across_modes"]
    assert receipt["null_contrasts_exact_zero"]


def test_paired_update_uses_one_exact_shared_input_set_before_either_step():
    config = _tiny_config()
    models, optimizers, scalers, amp, _ = _tiny_pair(config)
    batch = _tiny_batch()
    before = runner._input_commitments(batch)
    step = runner.paired_optimizer_update(
        models, optimizers, scalers, batch, config, amp
    )
    assert models["null"].input_ids == models["treatment"].input_ids
    assert step["input_sha256"] == before == runner._input_commitments(batch)
    assert step["paired_input_identity"]
    assert step["paired_barrier_completed_before_steps"]
    assert step["optimizer_step_after"] == {"null": 1, "treatment": 1}


def test_nonfinite_treatment_cannot_partially_step_either_arm():
    class NonfiniteTreatment(_TinyEnergy):
        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            return {name: value * torch.nan for name, value in result.items()}

    config = _tiny_config()
    null = _TinyEnergy()
    treatment = NonfiniteTreatment()
    treatment.load_state_dict(null.state_dict(), strict=True)
    models = {"null": null, "treatment": treatment}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        for name, model in models.items()
    }
    scalers = {
        name: torch.amp.GradScaler("cpu", enabled=False, init_scale=512)
        for name in models
    }
    before = {
        name: runner._state_dict_sha256(model.state_dict())
        for name, model in models.items()
    }
    with pytest.raises(RuntimeError, match="nonfinite forward"):
        runner.paired_optimizer_update(
            models, optimizers, scalers, _tiny_batch(), config, False
        )
    assert {
        name: runner._state_dict_sha256(model.state_dict())
        for name, model in models.items()
    } == before
    assert all(runner._optimizer_step(optimizers[name], models[name]) == 0 for name in models)


def test_two_steps_direct_equal_one_step_plus_joint_atomic_resume(tmp_path):
    config = _tiny_config(max_updates=2)
    lineage = {"data_lineage_sha256": "lineage"}
    statistics = _statistics_fixture()

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    direct = _tiny_pair(config)
    direct_history = []
    for update, batch in enumerate((_tiny_batch(0.0), _tiny_batch(0.05)), 1):
        step = runner.paired_optimizer_update(*direct[:3], batch, config, direct[3])
        direct_history.append(_history_row(update, step))
    direct_models, direct_optimizers, direct_scalers = direct[:3]
    direct_model_hashes = {
        name: runner._state_dict_sha256(model.state_dict())
        for name, model in direct_models.items()
    }
    direct_optimizer_hashes = {
        name: runner._canonical_sha256(optimizer.state_dict())
        for name, optimizer in direct_optimizers.items()
    }
    direct_scalers_state = {
        name: scaler.state_dict() for name, scaler in direct_scalers.items()
    }
    direct_random = (random.random(), float(np.random.random()), float(torch.rand(())))

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    paused = _tiny_pair(config)
    first = runner.paired_optimizer_update(
        *paused[:3], _tiny_batch(0.0), config, paused[3]
    )
    paused_history = [_history_row(1, first)]
    resume_path = tmp_path / "joint_resume.pt"
    runner._atomic_torch(
        resume_path,
        runner._resume_payload(
            config,
            lineage,
            paused[0],
            paused[1],
            paused[2],
            torch.device("cpu"),
            1,
            paused_history,
            [],
            paused[4],
            statistics,
        ),
    )
    resumed = _tiny_pair(config)
    update, history, development, observed_statistics = runner._load_joint_resume(
        resume_path,
        config,
        lineage,
        resumed[0],
        resumed[1],
        resumed[2],
        torch.device("cpu"),
        resumed[4],
    )
    assert update == 1 and development == [] and observed_statistics == statistics
    second = runner.paired_optimizer_update(
        *resumed[:3], _tiny_batch(0.05), config, resumed[3]
    )
    history.append(_history_row(2, second))
    assert runner._history_is_valid(history, 2, config)
    assert {
        name: runner._state_dict_sha256(model.state_dict())
        for name, model in resumed[0].items()
    } == direct_model_hashes
    assert {
        name: runner._canonical_sha256(optimizer.state_dict())
        for name, optimizer in resumed[1].items()
    } == direct_optimizer_hashes
    assert {name: scaler.state_dict() for name, scaler in resumed[2].items()} == (
        direct_scalers_state
    )
    resumed_random = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert resumed_random == direct_random


def test_joint_freeze_blocks_fresh_qualification_before_both_final_states(tmp_path):
    config = _tiny_config()
    final_path = tmp_path / "joint_final.pt"
    resume_path = tmp_path / "joint_resume.pt"
    with pytest.raises(RuntimeError, match="paired final checkpoint"):
        runner.freeze_qualification(tmp_path, config, final_path, resume_path)
    torch.save(
        {
            "format": runner.FINAL_FORMAT,
            "model": {"null": {"weight": torch.ones(1)}},
            "joint_final_state_complete": False,
        },
        final_path,
    )
    with pytest.raises(RuntimeError, match="committed joint state"):
        runner.freeze_qualification(tmp_path, config, final_path, resume_path)
    with pytest.raises(RuntimeError, match="joint state"):
        runner.qualification_manifests(
            object(), config, tmp_path, final_path, resume_path
        )


def _valid_joint_final_fixture(tmp_path):
    config = _tiny_config(max_updates=2)
    config["training"]["amp"] = True
    history = []
    for update in (1, 2):
        history.append(
            {
                "update": update,
                "paired_input_identity": True,
                "finite_forward": {"null": True, "treatment": True},
                "finite_unscaled_gradients": {"null": True, "treatment": True},
                "paired_barrier_completed_before_steps": True,
                "optimizer_step_applied": {"null": True, "treatment": True},
                "optimizer_step_before": {
                    "null": update - 1,
                    "treatment": update - 1,
                },
                "optimizer_step_after": {"null": update, "treatment": update},
                "amp_enabled": True,
                "amp_scale_before": {"null": 512.0, "treatment": 512.0},
                "amp_scale_after": {"null": 512.0, "treatment": 512.0},
                "gradient_clipped": {"null": False, "treatment": False},
                "unscaled_gradient_norm": {"null": 1.0, "treatment": 1.0},
                "loss": {
                    arm: {
                        "total": 1.0,
                        "ranking": 1.0,
                        "auxiliary_ranking": 1.0,
                        "point": 0.0,
                    }
                    for arm in ("null", "treatment")
                },
            }
        )
    model = {
        "null": {"weight": torch.tensor([1.0])},
        "treatment": {"weight": torch.tensor([2.0])},
    }
    model_hashes = {
        name: runner._state_dict_sha256(state) for name, state in model.items()
    }
    initial_hash = runner._state_dict_sha256({"weight": torch.tensor([0.0])})
    initialization = {
        "state_schema": ["weight"],
        "null_state_sha256": initial_hash,
        "treatment_state_sha256": initial_hash,
        "full_initial_state_equal": True,
    }
    statistics = _statistics_fixture()
    lineage = {
        "family_self_sha256": config["family_self_sha256"],
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": config["lineage"]["source_sha256"],
        "learned_checkpoint_dependencies": [],
        "train_seed": config["data"]["train_seed"],
        "development_seed": config["data"]["development_seed"],
        "reserved_fresh_qualification_seeds": config["data"][
            "qualification_seeds"
        ],
        "consumed_or_forbidden_qualification_seeds": config["data"][
            "consumed_or_forbidden_qualification_seeds"
        ],
    }
    lineage["data_lineage_sha256"] = runner._canonical_sha256(lineage)
    optimizer = {
        arm: {
            "state": {0: {"step": torch.tensor(2.0)}},
            "param_groups": [{"params": [0]}],
        }
        for arm in ("null", "treatment")
    }
    scaler = {arm: {"scale": 512.0} for arm in ("null", "treatment")}
    rng_state = {
        "python": (1, 2, 3),
        "numpy": ("MT19937", np.asarray([1, 2], dtype=np.uint32), 0, 0, 0.0),
        "torch_cpu": torch.tensor([1, 2], dtype=torch.uint8),
        "torch_cuda": [torch.tensor([3, 4], dtype=torch.uint8)],
    }
    development = []
    resume_commitments = {
        "optimizer_state_sha256": runner._canonical_sha256(optimizer),
        "scaler_state_sha256": runner._canonical_sha256(scaler),
        "rng_state_sha256": runner._canonical_sha256(rng_state),
        "development_sha256": runner._canonical_sha256(development),
    }
    resume = {
        "format": runner.RESUME_FORMAT,
        "purpose": runner.PURPOSE,
        "family_self_sha256": config["family_self_sha256"],
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": runner.source_hashes(),
        "learned_checkpoint_dependencies": [],
        "data_lineage": lineage,
        "initialization": initialization,
        "initial_statistics_integrity": statistics,
        "update": 2,
        "training_history": history,
        "training_history_sha256": runner._canonical_sha256(history),
        "model": model,
        "model_state_sha256": model_hashes,
        "optimizer": optimizer,
        "scaler": scaler,
        "development": development,
        "rng_state": rng_state,
        "resume_state_commitments": resume_commitments,
    }
    resume_path = tmp_path / "valid_joint_resume.pt"
    runner._atomic_torch(resume_path, resume)
    integrity = runner._training_integrity(
        history, config, initialization, statistics, True
    )
    final = {
        "format": runner.FINAL_FORMAT,
        "purpose": runner.PURPOSE,
        "family_self_sha256": config["family_self_sha256"],
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": runner.source_hashes(),
        "learned_checkpoint_dependencies": [],
        "data_lineage": lineage,
        "initialization": initialization,
        "initial_statistics_integrity": statistics,
        "update": 2,
        "training_history": history,
        "training_history_sha256": runner._canonical_sha256(history),
        "development": development,
        "model": model,
        "model_state_sha256": model_hashes,
        "resume_file_sha256": runner._sha256(resume_path),
        "resume_state_commitments": resume_commitments,
        "training_integrity": integrity,
        "joint_final_state_complete": True,
    }
    final_path = tmp_path / "valid_joint_final.pt"
    runner._atomic_torch(final_path, final)
    return config, resume_path, final_path


def test_joint_final_recomputes_integrity_and_checks_actual_scaler_and_step(tmp_path):
    config, resume_path, final_path = _valid_joint_final_fixture(tmp_path)
    assert runner._verify_joint_final(config, final_path, resume_path)[
        "joint_final_state_complete"
    ]

    final = torch.load(final_path, weights_only=False)
    final["training_integrity"]["passed"] = False
    runner._atomic_torch(final_path, final)
    with pytest.raises(RuntimeError, match="training-integrity"):
        runner._verify_joint_final(config, final_path, resume_path)

    config, resume_path, final_path = _valid_joint_final_fixture(tmp_path / "scale")
    resume = torch.load(resume_path, weights_only=False)
    resume["scaler"]["treatment"]["scale"] = 256.0
    runner._atomic_torch(resume_path, resume)
    final = torch.load(final_path, weights_only=False)
    final["resume_file_sha256"] = runner._sha256(resume_path)
    runner._atomic_torch(final_path, final)
    with pytest.raises(RuntimeError, match="persisted resume"):
        runner._verify_joint_final(config, final_path, resume_path)

    config, resume_path, final_path = _valid_joint_final_fixture(tmp_path / "history")
    resume = torch.load(resume_path, weights_only=False)
    del resume["training_history"]
    runner._atomic_torch(resume_path, resume)
    final = torch.load(final_path, weights_only=False)
    final["resume_file_sha256"] = runner._sha256(resume_path)
    runner._atomic_torch(final_path, final)
    with pytest.raises(RuntimeError, match="persisted resume"):
        runner._verify_joint_final(config, final_path, resume_path)

    config, resume_path, final_path = _valid_joint_final_fixture(tmp_path / "step")
    resume = torch.load(resume_path, weights_only=False)
    resume["optimizer"]["null"]["state"][0]["step"] = torch.tensor(1.0)
    runner._atomic_torch(resume_path, resume)
    final = torch.load(final_path, weights_only=False)
    final["resume_file_sha256"] = runner._sha256(resume_path)
    runner._atomic_torch(final_path, final)
    with pytest.raises(RuntimeError, match="persisted resume"):
        runner._verify_joint_final(config, final_path, resume_path)


@pytest.mark.parametrize("seed", [1604322, 1704322])
def test_fresh_seed_panels_are_inaccessible_without_joint_capability(seed):
    with pytest.raises(RuntimeError, match="before joint freeze"):
        runner.balanced_panel_manifest(object(), seed)


@pytest.mark.parametrize("seed", [1204322, 1304322, 1504322])
def test_consumed_seed_panels_are_always_forbidden(seed):
    with pytest.raises(RuntimeError, match="consumed qualification seed"):
        runner.balanced_panel_manifest(object(), seed)


def _panel_result(null_correct=40, treatment_correct=48):
    raw = []
    free_raw = []
    null_flags = np.zeros(48, dtype=bool)
    null_flags[:null_correct] = True
    treatment_flags = np.zeros(48, dtype=bool)
    treatment_flags[:treatment_correct] = True
    for item in range(48):
        null_correct_row = bool(null_flags[item])
        treatment_correct_row = bool(treatment_flags[item])
        transition = (
            "null-wrong-treatment-correct"
            if not null_correct_row and treatment_correct_row
            else "null-correct-treatment-wrong"
            if null_correct_row and not treatment_correct_row
            else "both-correct"
            if null_correct_row
            else "both-wrong"
        )
        raw.append(
            {
                "pair_key": {"pair_key_sha256": f"{item:064x}"},
                "null": {
                    "top1_correct": null_correct_row,
                    "broken_atlas_binding_correct": item < 4,
                    "broken_source_pairing_correct": item < 5,
                },
                "treatment": {
                    "top1_correct": treatment_correct_row,
                    "broken_atlas_binding_correct": item < 3,
                    "broken_source_pairing_correct": item < 4,
                },
                "transition": transition,
                "net_correction": int(treatment_correct_row) - int(null_correct_row),
            }
        )
        source_key = {"pair_key_sha256": f"{item:064x}"}
        free_raw.append(
            {
                "source_key": source_key,
                "true_pose": [0.0, 0.0, 0.0],
                "null": {},
                "treatment": {},
            }
        )
    order = {
        "evaluated_sample_count": 48,
        "maximum_energy_difference": 0.0,
        "decoded_pose_maximum_difference": 0.0,
        "energies_allclose": True,
        "top1_unchanged": True,
    }
    search_good = {
        "mae": [200.0, 2.0, 2.0],
        "predicted_pose": [[0.0, 0.0, 0.0]] * 48,
        "absolute_pose_error": [[0.0, 0.0, 0.0]] * 48,
        "seconds_per_slice": [1.0] * 48,
        "search_receipts": [
            {"source_key": {"pair_key_sha256": f"{item:064x}"}}
            for item in range(48)
        ],
        "physical_improvement_over_constant_prior": 0.6,
        "ten_slice_projected_p95_seconds": 100.0,
        "nonfinite_count": 0,
        "invalid_render_count": 0,
    }
    return {
        "seed": 1604322,
        "sample_count": 48,
        "fixed_candidates": {
            "correct": {
                "null": {
                    "normal": null_correct,
                    "broken_atlas_binding": 4,
                    "broken_source_pairing": 5,
                },
                "treatment": {
                    "normal": treatment_correct,
                    "broken_atlas_binding": 3,
                    "broken_source_pairing": 4,
                },
            },
            "nonfinite_count": {"null": 0, "treatment": 0},
            "invalid_render_count": 0,
            "order_equivariance": {
                "null": copy.deepcopy(order),
                "treatment": copy.deepcopy(order),
            },
            "mcnemar": runner.exact_mcnemar(null_flags, treatment_flags),
            "paired_row_count": 48,
            "unique_pair_key_count": 48,
            "paired_input_identity": True,
            "raw": raw,
        },
        "free_search": {
            "paired_source_identity": True,
            "raw": free_raw,
            "null": {**search_good, "mae": [900.0, 8.0, 8.0]},
            "treatment": copy.deepcopy(search_good),
        },
    }


def test_mcnemar_pairing_and_interpretation_are_exact_and_control_aware():
    baseline = np.zeros(48, dtype=bool)
    baseline[:40] = True
    treatment = baseline.copy()
    treatment[40:48] = True
    assert runner.exact_mcnemar(baseline, treatment) == {
        "null_wrong_treatment_correct": 8,
        "null_correct_treatment_wrong": 0,
        "discordant_count": 8,
        "net_corrections": 8,
        "exact_two_sided_p": 0.0078125,
    }
    config = runner.inspect_config(CONFIG)
    result = _panel_result()
    status = runner.paired_panel_status(result, config, require_search=True)
    assert status["passed"]
    assert status["causal"]["passed"]
    assert status["interpretation_branch"] == (
        "causal-rescue-only-authorize-independent-confirmation"
    )

    failed_control = copy.deepcopy(result)
    failed_control["fixed_candidates"]["correct"]["null"][
        "broken_atlas_binding"
    ] = 13
    for item, row in enumerate(failed_control["fixed_candidates"]["raw"]):
        row["null"]["broken_atlas_binding_correct"] = item < 13
    status = runner.paired_panel_status(
        failed_control, config, require_search=True
    )
    assert not status["passed"]
    assert not status["causal"]["passed"]
    assert status["integrity"]["checks"]["fixed_statistics_recomputed"]
    assert not status["null_control_integrity_runtime"]["passed"]

    free_search_failure = copy.deepcopy(result)
    free_search_failure["free_search"]["treatment"]["mae"][0] = 251.0
    status = runner.paired_panel_status(
        free_search_failure, config, require_search=True
    )
    assert not status["passed"]
    assert status["causal"]["passed"]
    assert status["treatment_fixed_panel"]["passed"]
    assert status["interpretation_branch"] == (
        "local-fixed-haar-mechanism-supported-end-to-end-no-go"
    )

    fixed_integrity_failure = copy.deepcopy(free_search_failure)
    fixed_integrity_failure["fixed_candidates"]["order_equivariance"][
        "treatment"
    ]["top1_unchanged"] = False
    status = runner.paired_panel_status(
        fixed_integrity_failure, config, require_search=True
    )
    assert not status["treatment_fixed_panel"]["passed"]
    assert status["interpretation_branch"] != (
        "local-fixed-haar-mechanism-supported-end-to-end-no-go"
    )

    selectively_missing = copy.deepcopy(result)
    selectively_missing["free_search"]["treatment"]["search_receipts"].pop()
    status = runner.paired_panel_status(
        selectively_missing, config, require_search=True
    )
    assert not status["integrity"]["passed"]
    assert not status["causal"]["passed"]
    assert not status["passed"]


def _family_panel(seed, kind):
    status = {
        "passed": False,
        "integrity": {"passed": True},
        "arm_absolute": {
            "null": {"passed": False},
            "treatment": {"passed": False},
        },
        "causal": {"passed": False},
        "treatment_fixed_panel": {"passed": False},
        "interpretation_branch": "seed-fail-no-causal-rescue",
    }
    if kind == "rescue":
        status.update(
            passed=True,
            causal={"passed": True},
            treatment_fixed_panel={"passed": True},
            interpretation_branch="causal-rescue-only-authorize-independent-confirmation",
        )
        status["arm_absolute"]["treatment"]["passed"] = True
    elif kind == "integrity":
        status["integrity"]["passed"] = False
        status["interpretation_branch"] = "integrity-failure-invalid-stop"
    elif kind == "both-pass":
        status["arm_absolute"]["null"]["passed"] = True
        status["arm_absolute"]["treatment"]["passed"] = True
        status["treatment_fixed_panel"]["passed"] = True
        status["interpretation_branch"] = "both-pass-no-spatial-necessity-claim"
    elif kind == "local":
        status["causal"]["passed"] = True
        status["treatment_fixed_panel"]["passed"] = True
        status["interpretation_branch"] = (
            "local-fixed-haar-mechanism-supported-end-to-end-no-go"
        )
    elif kind == "null-pass":
        status["arm_absolute"]["null"]["passed"] = True
    elif kind != "both-fail":
        raise ValueError(kind)
    keys = [f"{seed}-{item}" for item in range(48)]
    return {
        "seed": seed,
        "fixed_candidates": {
            "paired_row_count": 48,
            "raw": [
                {"pair_key": {"pair_key_sha256": value}} for value in keys
            ],
        },
        "free_search": {
            "raw": [
                {"source_key": {"pair_key_sha256": value}} for value in keys
            ]
        },
        "status": status,
    }


@pytest.mark.parametrize(
    ("kinds", "expected_branch", "passed"),
    [
        (
            ("rescue", "rescue"),
            "causal-rescue-only-authorize-independent-confirmation",
            True,
        ),
        (("integrity", "rescue"), "integrity-failure-invalid-stop", False),
        (("both-pass", "both-pass"), "both-pass-no-spatial-necessity-claim", False),
        (("rescue", "both-fail"), "one-seed-pass-family-fail", False),
        (("both-fail", "both-fail"), "both-fail-insufficient-stop", False),
        (("null-pass", "both-fail"), "family-fail-no-causal-rescue", False),
        (
            ("local", "local"),
            "local-fixed-haar-mechanism-supported-end-to-end-no-go",
            False,
        ),
    ],
)
def test_family_interpretation_branches_are_two_seed_exact(
    kinds, expected_branch, passed
):
    panels = [
        _family_panel(seed, kind)
        for seed, kind in zip((1604322, 1704322), kinds)
    ]
    result = runner.family_status(panels, {"passed": True})
    assert result["interpretation_branch"] == expected_branch
    assert result["passed"] is passed
    assert result["paired_qualification_rows"] == 96
    assert result["paired_free_search_rows"] == 96
    assert result["unique_family_pair_keys"] == 96
    assert result["independent_confirmation_authorized"] is passed
    assert result["protected_data_access_authorized"] is False
    assert result["promotion_authorized"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_exact_cuda_amp512_path_unscales_both_before_stepping():
    device = torch.device("cuda")
    config = runner.inspect_config(CONFIG)
    models, optimizers, scalers, amp, _ = _tiny_pair(config, device)
    assert amp
    step = runner.paired_optimizer_update(
        models, optimizers, scalers, _tiny_batch(device=device), config, amp
    )
    assert step["amp_scale_before"] == {"null": 512.0, "treatment": 512.0}
    assert step["amp_scale_after"] == {"null": 512.0, "treatment": 512.0}
    assert step["optimizer_step_after"] == {"null": 1, "treatment": 1}
    assert step["paired_barrier_completed_before_steps"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp512_two_steps_direct_equal_one_plus_serialized_resume(tmp_path):
    device = torch.device("cuda")
    config = runner.inspect_config(CONFIG)
    config = copy.deepcopy(config)
    config["training"]["max_updates"] = 2
    config["training"]["resume_every_updates"] = 1
    config["model"]["candidate_chunk_size"] = 2
    lineage = {"data_lineage_sha256": "cuda-lineage"}
    statistics = _statistics_fixture()

    random.seed(29)
    np.random.seed(29)
    torch.manual_seed(29)
    torch.cuda.manual_seed_all(29)
    direct = _tiny_pair(config, device)
    direct_history = []
    for update, batch in enumerate(
        (_tiny_batch(0.0, device), _tiny_batch(0.05, device)), 1
    ):
        step = runner.paired_optimizer_update(
            *direct[:3], batch, config, direct[3]
        )
        direct_history.append(_history_row(update, step))
    direct_model_hashes = {
        name: runner._state_dict_sha256(model.state_dict())
        for name, model in direct[0].items()
    }
    direct_optimizer_hashes = {
        name: runner._canonical_sha256(optimizer.state_dict())
        for name, optimizer in direct[1].items()
    }
    direct_scalers = {name: scaler.state_dict() for name, scaler in direct[2].items()}
    direct_rng = (
        random.random(),
        float(np.random.random()),
        float(torch.rand((), device=device).cpu()),
    )

    random.seed(29)
    np.random.seed(29)
    torch.manual_seed(29)
    torch.cuda.manual_seed_all(29)
    paused = _tiny_pair(config, device)
    first = runner.paired_optimizer_update(
        *paused[:3], _tiny_batch(0.0, device), config, paused[3]
    )
    history = [_history_row(1, first)]
    resume_path = tmp_path / "cuda_joint_resume.pt"
    runner._atomic_torch(
        resume_path,
        runner._resume_payload(
            config,
            lineage,
            paused[0],
            paused[1],
            paused[2],
            device,
            1,
            history,
            [],
            paused[4],
            statistics,
        ),
    )
    resumed = _tiny_pair(config, device)
    update, history, _, _ = runner._load_joint_resume(
        resume_path,
        config,
        lineage,
        resumed[0],
        resumed[1],
        resumed[2],
        device,
        resumed[4],
    )
    assert update == 1
    second = runner.paired_optimizer_update(
        *resumed[:3], _tiny_batch(0.05, device), config, resumed[3]
    )
    history.append(_history_row(2, second))
    assert runner._canonical_sha256(history) == runner._canonical_sha256(
        direct_history
    )
    assert {
        name: runner._state_dict_sha256(model.state_dict())
        for name, model in resumed[0].items()
    } == direct_model_hashes
    assert {
        name: runner._canonical_sha256(optimizer.state_dict())
        for name, optimizer in resumed[1].items()
    } == direct_optimizer_hashes
    assert {name: scaler.state_dict() for name, scaler in resumed[2].items()} == (
        direct_scalers
    )
    assert all(value["scale"] == 512.0 for value in direct_scalers.values())
    resumed_rng = (
        random.random(),
        float(np.random.random()),
        float(torch.rand((), device=device).cpu()),
    )
    assert resumed_rng == direct_rng


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_actual_paired_models_cuda_amp512_resume_is_bit_exact(tmp_path):
    device = torch.device("cuda")
    config = copy.deepcopy(runner.inspect_config(CONFIG))
    config["training"]["max_updates"] = 2
    config["training"]["resume_every_updates"] = 1
    config["model"]["candidate_chunk_size"] = 2
    lineage = {"data_lineage_sha256": "actual-model-cuda-lineage"}
    batches = (
        _actual_model_batch(0.0, device),
        _actual_model_batch(0.03, device),
    )

    direct_models, direct_initialization = runner.initialize_pair(config, device)
    direct_optimizers, direct_scalers, direct_amp = runner._optimizers_and_scalers(
        direct_models, config, device
    )
    direct_statistics = runner.initial_statistics_integrity(
        direct_models, batches[0]
    )
    direct_history = []
    for update, batch in enumerate(batches, 1):
        step = runner.paired_optimizer_update(
            direct_models,
            direct_optimizers,
            direct_scalers,
            batch,
            config,
            direct_amp,
        )
        direct_history.append(_history_row(update, step))
    direct_model_hashes = {
        name: runner._state_dict_sha256(model.state_dict())
        for name, model in direct_models.items()
    }
    direct_optimizer_hashes = {
        name: runner._canonical_sha256(optimizer.state_dict())
        for name, optimizer in direct_optimizers.items()
    }
    direct_optimizer_steps = {
        name: {
            int(value["step"])
            for value in optimizer.state_dict()["state"].values()
            if "step" in value
        }
        for name, optimizer in direct_optimizers.items()
    }
    direct_scaler_states = {
        name: scaler.state_dict() for name, scaler in direct_scalers.items()
    }
    direct_rng = (
        random.random(),
        float(np.random.random()),
        float(torch.rand((), device=device).cpu()),
    )
    del direct_models, direct_optimizers, direct_scalers

    paused_models, paused_initialization = runner.initialize_pair(config, device)
    paused_optimizers, paused_scalers, paused_amp = runner._optimizers_and_scalers(
        paused_models, config, device
    )
    paused_statistics = runner.initial_statistics_integrity(
        paused_models, batches[0]
    )
    first = runner.paired_optimizer_update(
        paused_models,
        paused_optimizers,
        paused_scalers,
        batches[0],
        config,
        paused_amp,
    )
    history = [_history_row(1, first)]
    resume_path = tmp_path / "actual_pair_cuda_resume.pt"
    runner._atomic_torch(
        resume_path,
        runner._resume_payload(
            config,
            lineage,
            paused_models,
            paused_optimizers,
            paused_scalers,
            device,
            1,
            history,
            [],
            paused_initialization,
            paused_statistics,
        ),
    )
    del paused_models, paused_optimizers, paused_scalers

    resumed_models, resumed_initialization = runner.initialize_pair(config, device)
    resumed_optimizers, resumed_scalers, resumed_amp = runner._optimizers_and_scalers(
        resumed_models, config, device
    )
    update, history, development, resumed_statistics = runner._load_joint_resume(
        resume_path,
        config,
        lineage,
        resumed_models,
        resumed_optimizers,
        resumed_scalers,
        device,
        resumed_initialization,
    )
    assert update == 1 and development == []
    second = runner.paired_optimizer_update(
        resumed_models,
        resumed_optimizers,
        resumed_scalers,
        batches[1],
        config,
        resumed_amp,
    )
    history.append(_history_row(2, second))
    assert paused_initialization == direct_initialization == resumed_initialization
    assert paused_statistics == direct_statistics == resumed_statistics
    assert runner._canonical_sha256(history) == runner._canonical_sha256(
        direct_history
    )
    assert {
        name: runner._state_dict_sha256(model.state_dict())
        for name, model in resumed_models.items()
    } == direct_model_hashes
    assert {
        name: runner._canonical_sha256(optimizer.state_dict())
        for name, optimizer in resumed_optimizers.items()
    } == direct_optimizer_hashes
    assert {
        name: {
            int(value["step"])
            for value in optimizer.state_dict()["state"].values()
            if "step" in value
        }
        for name, optimizer in resumed_optimizers.items()
    } == direct_optimizer_steps == {"null": {2}, "treatment": {2}}
    assert {
        name: scaler.state_dict() for name, scaler in resumed_scalers.items()
    } == direct_scaler_states
    assert all(value["scale"] == 512.0 for value in direct_scaler_states.values())
    resumed_rng = (
        random.random(),
        float(np.random.random()),
        float(torch.rand((), device=device).cpu()),
    )
    assert resumed_rng == direct_rng
