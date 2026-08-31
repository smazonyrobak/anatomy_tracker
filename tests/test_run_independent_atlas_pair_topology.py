import copy
import errno
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import training.run_independent_atlas_pair_topology as runner


ROOT = Path(__file__).parents[1]
CONFIG = (
    ROOT
    / "training/configs/independent_oracle_atlas_pair_topology_pair_1500_r1804322.json"
)


@pytest.fixture(autouse=True)
def _deterministic_execution(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runner.configure_deterministic_execution()


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
        source = source_image.mean((1, 2, 3))[:, None]
        candidate = candidate_image.mean((2, 3, 4))
        energy8 = self.weight * (candidate - source).square() + self.bias
        energy16 = (self.weight + 0.2) * (candidate + source).square()
        return {
            "energy": 0.5 * (energy8 + energy16),
            "energy8": energy8,
            "energy16": energy16,
        }


def _tiny_config(max_updates=2):
    config = copy.deepcopy(runner.inspect_config(CONFIG))
    config["training"]["amp"] = False
    config["training"]["max_updates"] = max_updates
    config["training"]["resume_every_updates"] = 1
    config["model"]["candidate_chunk_size"] = 2
    config["integrity"]["enforce_topology_gradient_audit"] = False
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
    amp = bool(config["training"]["amp"] and device.type == "cuda")
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
    source = (
        torch.arange(16, dtype=torch.float32).reshape(2, 1, 2, 4) / 16 + offset
    ).to(device)
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
    return {
        "source_image": source,
        "source_mask": torch.zeros_like(source, dtype=torch.bool),
        "mask_available": torch.zeros(2, 1, 1, 1, device=device),
        "true_pose": pose[:, 0],
        "candidate_pose": pose,
        "candidate_image": candidates,
        "candidate_mask": torch.ones_like(candidates, dtype=torch.bool),
        "target_index": torch.tensor([0, 0], device=device),
    }


def _full_shape_batch(device=torch.device("cpu"), candidates=2):
    generator = torch.Generator(device="cpu").manual_seed(23)
    source = torch.rand(1, 1, 160, 232, generator=generator).to(device)
    candidate = torch.rand(
        1, candidates, 1, 160, 232, generator=generator
    ).to(device)
    pose = torch.zeros(1, candidates, 3, device=device)
    pose[0, :, 0] = torch.arange(candidates, device=device) * 125.0
    return {
        "source_image": source,
        "source_mask": torch.zeros_like(source, dtype=torch.bool),
        "mask_available": torch.zeros(1, 1, 1, 1, device=device),
        "true_pose": pose[:, 0],
        "candidate_pose": pose,
        "candidate_image": candidate,
        "candidate_mask": torch.ones_like(candidate, dtype=torch.bool),
        "target_index": torch.zeros(1, dtype=torch.long, device=device),
    }


def _history_row(update, step):
    return {"update": update, **step}


def test_frozen_config_self_hash_sources_and_seed_provenance_are_exact(tmp_path):
    config = runner.inspect_config(CONFIG)
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    commitment = raw.pop("contract_sha256")
    assert commitment == "1f9c551bb3ac5837ac41e2692a5cc58a159c8a95326820c0fab23264bb51b4dd"
    assert commitment == runner._canonical_sha256(raw) == config["family_self_sha256"]
    assert config["config_file_sha256"] == runner._sha256(CONFIG)
    assert config["lineage"]["source_sha256"] == runner.source_hashes()
    assert config["lineage"]["source_sha256"][
        "training/run_independent_atlas_pair_topology.py"
    ] == "b9d093ea7e003f912804e84f2b8f0d7677994b14fbcaf3c6b637368cc03e4451"
    assert set(config["lineage"]["source_sha256"]) == set(runner.SOURCE_FILES)
    assert all(
        len(value) == 64 for value in config["lineage"]["source_sha256"].values()
    )
    assert config["data"]["train_seed"] == 1904322
    assert config["data"]["development_seed"] == 2004322
    assert config["data"]["qualification_seeds"] == [2104322, 2204322]
    assert config["data"]["consumed_or_forbidden_prior_seeds"] == [
        1004322,
        1104322,
        1204322,
        1304322,
        1404322,
        1504322,
        1604322,
        1704322,
    ]
    assert config["learned_checkpoint_dependencies"] == []
    assert not config["product5_access"]
    assert not config["calibration_access"]
    assert not config["final_test_access"]

    changed = copy.deepcopy(raw)
    changed["training"]["max_updates"] = 1501
    changed["contract_sha256"] = runner._canonical_sha256(changed)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen family"):
        runner.inspect_config(path)


def test_live_source_and_config_bytes_are_rechecked(monkeypatch):
    config = runner.inspect_config(CONFIG)
    changed_sources = dict(config["lineage"]["source_sha256"])
    changed_sources["training/run_independent_atlas_pair_topology.py"] = "0" * 64
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


def test_deterministic_cuda_contract_flags_are_sealed(monkeypatch):
    config = runner.inspect_config(CONFIG)
    assert config["training"]["deterministic_execution"] == {
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
        "serialized_resume_must_be_bit_exact": True,
    }
    assert runner.configure_deterministic_execution(torch.device("cuda")) == {
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
        "cuda_execution": True,
    }
    with monkeypatch.context() as patch:
        patch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        with pytest.raises(RuntimeError, match="differs from the frozen value"):
            runner.configure_deterministic_execution()


def test_pair_initialization_and_topology_integrity_are_exact():
    config = runner.inspect_config(CONFIG)
    models, initialization = runner.initialize_pair(config, torch.device("cpu"))
    assert initialization["parameter_count_each"] == 284058
    assert initialization["full_initial_state_equal"]
    assert initialization["independent_state_storage"]
    assert initialization["within_arm_state_storage_unique"] == {
        "null": True,
        "treatment": True,
    }
    assert initialization["null_state_sha256"] == initialization[
        "treatment_state_sha256"
    ]

    receipt = runner.initial_topology_integrity(models, _full_shape_batch(), config)
    assert runner._topology_integrity_valid(receipt, config)
    assert receipt["complete_initial_state_exact"]
    assert receipt["complete_initial_outputs_exact"]
    assert receipt["permutations_bijective_fixed_and_hash_bound"]
    assert receipt["pixel_vector_multiset_and_inverse_recovery_exact"]
    assert receipt["initial_output_sha256"]["null"] == receipt[
        "initial_output_sha256"
    ]["treatment"]
    for level, pixels in (("8", 580), ("16", 150)):
        assert receipt["levels"][level]["channels"] == 82
        assert receipt["levels"][level]["pixel_vector_count"] == pixels
        assert receipt["levels"][level]["native_lattice_sha256"] != receipt[
            "levels"
        ][level]["scrambled_lattice_sha256"]
    for arm in ("null", "treatment"):
        audit = receipt["off_center_initial"][arm]
        assert audit["count"] == 1536
        assert audit["finite"] and audit["exact_zero"]
        assert len(audit["per_tensor"]) == 6
        assert all(
            item["count"] == 256
            and item["finite"]
            and item["nonzero_count"] == 0
            for item in audit["per_tensor"].values()
        )


def test_paired_barrier_uses_one_shared_input_set_and_blocks_partial_step():
    config = _tiny_config()
    models, optimizers, scalers, amp, _ = _tiny_pair(config)
    batch = _tiny_batch()
    before = runner._input_commitments(batch)
    step = runner.paired_optimizer_update(models, optimizers, scalers, batch, config, amp)
    assert models["null"].input_ids == models["treatment"].input_ids
    assert step["input_sha256"] == before == runner._input_commitments(batch)
    assert step["paired_input_identity"]
    assert step["paired_barrier_completed_before_steps"]
    assert step["optimizer_step_after"] == {"null": 1, "treatment": 1}

    class NonfiniteTreatment(_TinyEnergy):
        def forward(self, *args, **kwargs):
            return {
                name: value * torch.nan
                for name, value in super().forward(*args, **kwargs).items()
            }

    null = _TinyEnergy()
    treatment = NonfiniteTreatment()
    treatment.load_state_dict(null.state_dict(), strict=True)
    blocked_models = {"null": null, "treatment": treatment}
    blocked_optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        for name, model in blocked_models.items()
    }
    blocked_scalers = {
        name: torch.amp.GradScaler("cpu", enabled=False, init_scale=512)
        for name in blocked_models
    }
    state_before = {
        name: runner._state_dict_sha256(model.state_dict())
        for name, model in blocked_models.items()
    }
    with pytest.raises(RuntimeError, match="nonfinite forward"):
        runner.paired_optimizer_update(
            blocked_models,
            blocked_optimizers,
            blocked_scalers,
            _tiny_batch(),
            config,
            False,
        )
    assert state_before == {
        name: runner._state_dict_sha256(model.state_dict())
        for name, model in blocked_models.items()
    }
    assert all(
        runner._optimizer_step(blocked_optimizers[name], blocked_models[name]) == 0
        for name in blocked_models
    )


def test_cpu_two_steps_equal_one_step_plus_serialized_resume(tmp_path):
    config = _tiny_config()
    lineage = {"data_lineage_sha256": "cpu-lineage"}

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    direct = _tiny_pair(config)
    direct_history = []
    for update, batch in enumerate((_tiny_batch(), _tiny_batch(0.05)), 1):
        direct_history.append(
            _history_row(
                update,
                runner.paired_optimizer_update(*direct[:3], batch, config, direct[3]),
            )
        )
    direct_models = {
        name: runner._state_dict_sha256(model.state_dict())
        for name, model in direct[0].items()
    }
    direct_optimizers = {
        name: runner._canonical_sha256(optimizer.state_dict())
        for name, optimizer in direct[1].items()
    }
    direct_scalers = {name: scaler.state_dict() for name, scaler in direct[2].items()}
    direct_rng = (random.random(), float(np.random.random()), float(torch.rand(())))

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    paused = _tiny_pair(config)
    history = [
        _history_row(
            1,
            runner.paired_optimizer_update(*paused[:3], _tiny_batch(), config, paused[3]),
        )
    ]
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
            history,
            [],
            paused[4],
            {},
        ),
    )
    resumed = _tiny_pair(config)
    update, history, development, topology = runner._load_joint_resume(
        resume_path,
        config,
        lineage,
        resumed[0],
        resumed[1],
        resumed[2],
        torch.device("cpu"),
        resumed[4],
    )
    assert update == 1 and development == [] and topology == {}
    history.append(
        _history_row(
            2,
            runner.paired_optimizer_update(
                *resumed[:3], _tiny_batch(0.05), config, resumed[3]
            ),
        )
    )
    assert runner._canonical_sha256(history) == runner._canonical_sha256(direct_history)
    assert {
        name: runner._state_dict_sha256(model.state_dict())
        for name, model in resumed[0].items()
    } == direct_models
    assert {
        name: runner._canonical_sha256(optimizer.state_dict())
        for name, optimizer in resumed[1].items()
    } == direct_optimizers
    assert {name: scaler.state_dict() for name, scaler in resumed[2].items()} == direct_scalers
    resumed_rng = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert resumed_rng == direct_rng


def test_atomic_replace_retries_transient_errors_then_promotes_once(
    tmp_path, monkeypatch
):
    assert runner.ATOMIC_REPLACE_ATTEMPTS == 9
    assert len(runner.ATOMIC_REPLACE_DELAYS_SECONDS) == 8
    path = tmp_path / "receipt.json"
    path.write_bytes(b"old canonical\n")
    original_replace = runner.os.replace
    calls = []
    sleeps = []

    def replace(source, destination):
        calls.append((Path(source), Path(destination)))
        if len(calls) <= 3:
            raise PermissionError(errno.EACCES, "transient sharing violation")
        original_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", replace)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    payload = {"committed": True}
    runner._atomic_json(path, payload)
    assert len(calls) == 4
    assert sleeps == list(runner.ATOMIC_REPLACE_DELAYS_SECONDS[:3])
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("failure", ["nonretryable", "exhausted"])
def test_atomic_replace_failure_preserves_old_canonical_and_complete_temporary(
    tmp_path, monkeypatch, failure
):
    path = tmp_path / f"{failure}.json"
    old = b'{"old":true}\n'
    path.write_bytes(old)
    calls = []
    sleeps = []

    def replace(source, destination):
        calls.append((Path(source), Path(destination)))
        if failure == "nonretryable":
            raise FileNotFoundError(errno.ENOENT, "nonretryable")
        raise PermissionError(errno.EACCES, "persistent sharing violation")

    monkeypatch.setattr(runner.os, "replace", replace)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    error = FileNotFoundError if failure == "nonretryable" else PermissionError
    with pytest.raises(error):
        runner._atomic_json(path, {"new": True})
    assert path.read_bytes() == old
    temporary = path.with_suffix(".json.tmp")
    assert temporary.is_file()
    assert json.loads(temporary.read_text(encoding="utf-8")) == {"new": True}
    expected_attempts = 1 if failure == "nonretryable" else 9
    assert len(calls) == expected_attempts
    assert sleeps == (
        []
        if failure == "nonretryable"
        else list(runner.ATOMIC_REPLACE_DELAYS_SECONDS)
    )


def test_run_refuses_stale_atomic_temporary_before_data_or_model_work(
    tmp_path, monkeypatch
):
    config = runner.inspect_config(CONFIG)
    run_folder = tmp_path / config["name"]
    run_folder.mkdir()
    (run_folder / "joint_resume_state.pt.tmp").write_bytes(b"forensic bytes")
    monkeypatch.setenv(config["paths"]["run_root_env"], str(tmp_path))
    monkeypatch.setattr(runner, "_device", lambda unused: torch.device("cuda"))
    with pytest.raises(RuntimeError, match="stale atomic temporaries"):
        runner.run(CONFIG)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_production_updates_cover_all_coefficients_and_cuda_resume_is_bit_exact(
    tmp_path,
):
    device = torch.device("cuda")
    runner.configure_deterministic_execution(device)
    config = copy.deepcopy(runner.inspect_config(CONFIG))
    config["training"]["max_updates"] = 2
    config["training"]["resume_every_updates"] = 1
    generator = runner.SyntheticRegistrationGenerator(
        ROOT / config["paths"]["atlas_repo_relative"], device
    )
    manifest = runner.training_manifest(generator, config)
    cache = {}
    batches = tuple(
        runner._training_batch(update, generator, manifest, config, cache)
        for update in (0, 1)
    )
    assert [batch["sample_indices"].tolist() for batch in batches] == [
        runner.training_indices(
            update,
            config["data"]["train_count"],
            config["training"]["batch_size"],
            config["data"]["train_seed"],
        ).tolist()
        for update in (0, 1)
    ]

    def pair():
        models, initialization = runner.initialize_pair(config, device)
        optimizers, scalers, amp = runner._optimizers_and_scalers(
            models, config, device
        )
        initialization = runner.bind_optimizer_scaler_initialization(
            initialization, optimizers, scalers, config, amp
        )
        return models, optimizers, scalers, amp, initialization

    direct = pair()
    topology = runner.initial_topology_integrity(direct[0], batches[0], config)
    direct_history = []
    for update, batch in enumerate(batches, 1):
        direct_history.append(
            _history_row(
                update,
                runner.paired_optimizer_update(
                    direct[0], direct[1], direct[2], batch, config, direct[3]
                ),
            )
        )
    assert runner._early_topology_coverage_valid(direct_history, 2, config)
    for row in direct_history:
        for arm in ("null", "treatment"):
            gradient = row["off_center_gradient"][arm]
            assert gradient["count"] == 1536
            assert gradient["finite"] and gradient["positive_norm"]
            assert len(gradient["per_tensor"]) == 6
            assert all(
                value["count"] == 256
                and value["finite"]
                and value["positive_norm"]
                for value in gradient["per_tensor"].values()
            )
    for arm in ("null", "treatment"):
        covered = {
            index
            for row in direct_history
            for index in row["off_center_gradient"][arm]["nonzero_indices"]
        }
        assert covered == set(range(1536))
        weights = direct_history[1]["off_center_weight_after"][arm]
        assert weights["count"] == weights["nonzero_count"] == 1536
        assert weights["finite"]
        assert len(weights["per_tensor"]) == 6
    assert direct_history[1]["off_center_weight_after"]["null"]["sha256"] != (
        direct_history[1]["off_center_weight_after"]["treatment"]["sha256"]
    )
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
    del direct

    paused = pair()
    history = [
        _history_row(
            1,
            runner.paired_optimizer_update(
                paused[0], paused[1], paused[2], batches[0], config, paused[3]
            ),
        )
    ]
    resume_path = tmp_path / "actual_pair_cuda_resume.pt"
    lineage = {"data_lineage_sha256": "actual-model-cuda-lineage"}
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
            topology,
        ),
    )
    paused_initialization = paused[4]
    del paused

    resumed = pair()
    update, history, development, resumed_topology = runner._load_joint_resume(
        resume_path,
        config,
        lineage,
        resumed[0],
        resumed[1],
        resumed[2],
        device,
        resumed[4],
    )
    assert update == 1 and development == [] and resumed_topology == topology
    history.append(
        _history_row(
            2,
            runner.paired_optimizer_update(
                resumed[0], resumed[1], resumed[2], batches[1], config, resumed[3]
            ),
        )
    )
    assert paused_initialization == resumed[4]
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


def _valid_joint_final_fixture(tmp_path):
    config = _tiny_config()
    config["training"]["amp"] = True
    expected_determinism = {
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
        "cuda_execution": True,
    }
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
                "off_center_gradient": None,
                "off_center_weight_after": None,
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
        "determinism": expected_determinism,
    }
    permutation_sha256 = config["family"]["permutation_contract"][
        "permutation_contract_sha256"
    ]
    lineage = {
        "family_self_sha256": config["family_self_sha256"],
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": config["lineage"]["source_sha256"],
        "determinism": expected_determinism,
        "permutation_contract_sha256": permutation_sha256,
        "generator_contract_sha256": "generator",
        "atlas_sha256": {"average": "a", "annotation": "b", "query": "c"},
        "train_seed": config["data"]["train_seed"],
        "development_seed": config["data"]["development_seed"],
        "reserved_fresh_qualification_seeds": config["data"][
            "qualification_seeds"
        ],
        "consumed_or_forbidden_prior_seeds": config["data"][
            "consumed_or_forbidden_prior_seeds"
        ],
        "learned_checkpoint_dependencies": [],
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
    commitments = {
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
        "determinism": expected_determinism,
        "permutation_contract_sha256": permutation_sha256,
        "learned_checkpoint_dependencies": [],
        "data_lineage": lineage,
        "initialization": initialization,
        "initial_topology_integrity": {},
        "update": 2,
        "training_history": history,
        "training_history_sha256": runner._canonical_sha256(history),
        "model": model,
        "model_state_sha256": model_hashes,
        "optimizer": optimizer,
        "scaler": scaler,
        "development": development,
        "rng_state": rng_state,
        "resume_state_commitments": commitments,
    }
    resume_path = tmp_path / "joint_resume.pt"
    runner._atomic_torch(resume_path, resume)
    integrity = runner._training_integrity(history, config, initialization, {}, True)
    assert integrity["passed"]
    final = {
        "format": runner.FINAL_FORMAT,
        "purpose": runner.PURPOSE,
        "family_self_sha256": config["family_self_sha256"],
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": runner.source_hashes(),
        "determinism": expected_determinism,
        "permutation_contract_sha256": permutation_sha256,
        "learned_checkpoint_dependencies": [],
        "data_lineage": lineage,
        "initialization": initialization,
        "initial_topology_integrity": {},
        "update": 2,
        "training_history": history,
        "training_history_sha256": runner._canonical_sha256(history),
        "development": development,
        "model": model,
        "model_state_sha256": model_hashes,
        "resume_file_sha256": runner._sha256(resume_path),
        "resume_state_commitments": commitments,
        "training_integrity": integrity,
        "joint_final_state_complete": True,
    }
    final_path = tmp_path / "joint_final.pt"
    runner._atomic_torch(final_path, final)
    return config, resume_path, final_path, lineage


def test_joint_freeze_binds_both_states_and_provenance(tmp_path):
    config = _tiny_config()
    final_path = tmp_path / "missing_final.pt"
    resume_path = tmp_path / "missing_resume.pt"
    with pytest.raises(RuntimeError, match="paired final checkpoint"):
        runner.freeze_qualification(tmp_path, config, final_path, resume_path)

    config, resume_path, final_path, lineage = _valid_joint_final_fixture(
        tmp_path / "valid"
    )
    freeze_path = runner.freeze_qualification(
        final_path.parent, config, final_path, resume_path
    )
    capability = runner.verified_qualification_capability(
        final_path.parent, config, final_path, resume_path
    )
    final = torch.load(final_path, weights_only=False)
    assert capability["joint_final_file_sha256"] == runner._sha256(final_path)
    assert capability["both_model_state_sha256"] == final["model_state_sha256"]
    assert capability["data_lineage_sha256"] == lineage["data_lineage_sha256"]
    assert capability["source_sha256"] == runner.source_hashes()
    assert capability["qualification_seeds"] == (2104322, 2204322)
    assert capability["permutation_contract_sha256"] == config["family"][
        "permutation_contract"
    ]["permutation_contract_sha256"]

    tampered = json.loads(freeze_path.read_text(encoding="utf-8"))
    tampered["both_model_state_sha256"]["treatment"] = "0" * 64
    freeze_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no longer matches"):
        runner.verified_qualification_capability(
            final_path.parent, config, final_path, resume_path
        )


@pytest.mark.parametrize("seed", [2104322, 2204322])
def test_fresh_qualification_seeds_are_inaccessible_before_joint_freeze(seed):
    with pytest.raises(RuntimeError, match="before joint freeze"):
        runner.balanced_panel_manifest(object(), seed)


@pytest.mark.parametrize(
    "seed",
    [1004322, 1104322, 1204322, 1304322, 1404322, 1504322, 1604322, 1704322],
)
def test_all_consumed_prior_seeds_are_forbidden(seed):
    with pytest.raises(RuntimeError, match="consumed qualification seed"):
        runner.balanced_panel_manifest(object(), seed)


def _panel_result(null_correct=40, treatment_correct=48):
    null_flags = np.zeros(48, dtype=bool)
    treatment_flags = np.zeros(48, dtype=bool)
    null_flags[:null_correct] = True
    treatment_flags[:treatment_correct] = True
    raw = []
    free_raw = []
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
        key = {"pair_key_sha256": f"{item:064x}"}
        raw.append(
            {
                "pair_key": key,
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
        free_raw.append(
            {
                "source_key": key,
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
        "seed": 2104322,
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


def test_order_gate_uses_allclose_semantics_not_absolute_delta_alone():
    config = runner.inspect_config(CONFIG)
    result = _panel_result()
    result["fixed_candidates"]["order_equivariance"]["treatment"].update(
        maximum_energy_difference=1.5e-6,
        energies_allclose=True,
    )
    status = runner.paired_panel_status(result, config, require_search=True)
    assert 1.5e-6 > config["gates"]["order_energy_atol"]
    assert status["arm_absolute"]["treatment"]["checks"]["order_equivariance"]
    failed = copy.deepcopy(result)
    failed["fixed_candidates"]["order_equivariance"]["treatment"][
        "energies_allclose"
    ] = False
    assert not runner.paired_panel_status(
        failed, config, require_search=True
    )["arm_absolute"]["treatment"]["checks"]["order_equivariance"]


def test_qualification_raw_rows_and_search_receipts_close_exactly():
    config = runner.inspect_config(CONFIG)
    result = _panel_result()
    status = runner.paired_panel_status(result, config, require_search=True)
    assert status["passed"]
    assert status["integrity"]["passed"]
    assert status["causal"]["passed"]
    assert status["interpretation_branch"] == (
        "native-adjacency-evidence-authorize-independent-confirmation"
    )

    missing = copy.deepcopy(result)
    missing["free_search"]["treatment"]["search_receipts"].pop()
    status = runner.paired_panel_status(missing, config, require_search=True)
    assert not status["integrity"]["checks"]["no_selective_missing_free_search"]
    assert not status["causal"]["passed"]
    assert not status["passed"]

    mismatched = copy.deepcopy(result)
    mismatched["free_search"]["raw"][1]["source_key"][
        "pair_key_sha256"
    ] = "f" * 64
    status = runner.paired_panel_status(mismatched, config, require_search=True)
    assert not status["integrity"]["checks"]["no_selective_missing_free_search"]


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
            interpretation_branch="native-adjacency-evidence-authorize-independent-confirmation",
        )
        status["arm_absolute"]["treatment"]["passed"] = True
    elif kind == "integrity":
        status["integrity"]["passed"] = False
        status["interpretation_branch"] = "integrity-failure-invalid-stop"
    elif kind == "both-pass":
        status["arm_absolute"]["null"]["passed"] = True
        status["arm_absolute"]["treatment"]["passed"] = True
        status["treatment_fixed_panel"]["passed"] = True
        status["interpretation_branch"] = (
            "both-pass-native-adjacency-necessity-not-shown"
        )
    elif kind == "local":
        status["causal"]["passed"] = True
        status["treatment_fixed_panel"]["passed"] = True
        status["interpretation_branch"] = (
            "local-native-adjacency-supported-end-to-end-no-go"
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
            "raw": [{"pair_key": {"pair_key_sha256": value}} for value in keys],
        },
        "free_search": {
            "raw": [{"source_key": {"pair_key_sha256": value}} for value in keys]
        },
        "status": status,
    }


@pytest.mark.parametrize(
    ("kinds", "expected_branch", "passed"),
    [
        (
            ("rescue", "rescue"),
            "native-adjacency-evidence-authorize-independent-confirmation",
            True,
        ),
        (("integrity", "rescue"), "integrity-failure-invalid-stop", False),
        (
            ("both-pass", "both-pass"),
            "both-pass-native-adjacency-necessity-not-shown",
            False,
        ),
        (("rescue", "both-fail"), "one-seed-pass-family-fail", False),
        (
            ("both-fail", "both-fail"),
            "both-fail-change-feature-or-candidate-construction-before-recurrence",
            False,
        ),
        (("null-pass", "both-fail"), "family-fail-no-causal-rescue", False),
        (
            ("local", "local"),
            "local-native-adjacency-supported-end-to-end-no-go",
            False,
        ),
    ],
)
def test_family_interpretation_truth_table(kinds, expected_branch, passed):
    panels = [
        _family_panel(seed, kind)
        for seed, kind in zip((2104322, 2204322), kinds)
    ]
    result = runner.family_status(panels, {"passed": True})
    assert result["interpretation_branch"] == expected_branch
    assert result["passed"] is passed
    assert result["paired_qualification_rows"] == 96
    assert result["paired_free_search_rows"] == 96
    assert result["unique_family_pair_keys"] == 96
    assert result["independent_confirmation_authorized"] is passed
    assert not result["protected_data_access_authorized"]
    assert not result["promotion_authorized"]
