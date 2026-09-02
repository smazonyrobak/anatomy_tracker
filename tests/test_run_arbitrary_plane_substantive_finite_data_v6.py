import copy
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest

import training.run_arbitrary_plane_substantive_finite_data_v6 as launcher


def _fake_manifest(partition, config, receipt="a" * 64):
    return {
        "receipt_sha256": receipt,
        "status": "FROZEN",
        "row_count": partition["row_count"],
        "generation_lineage": {"split": partition["split"]},
        "finite_psf_run_contract": {
            "render_mode": "finite_boxcar",
            "axial_sample_count": 9,
        },
        "finite_psf_capability_sha256": config[
            "finite_psf_capability_sha256"
        ],
        "generator_binding": {
            "receipt_sha256": "b" * 64,
            "generation_lineage_sha256": "c" * 64,
        },
    }


def _install_fake_v6_replay(monkeypatch, partition, config, tamper=None):
    manifest = _fake_manifest(partition, config)
    calls = []

    def load_manifest(cache_directory, *, expected_manifest_receipt_sha256):
        assert expected_manifest_receipt_sha256 == manifest["receipt_sha256"]
        return manifest

    def load_rows(
        cache_directory, indices, *, expected_manifest_receipt_sha256
    ):
        assert indices is not None
        assert expected_manifest_receipt_sha256 == manifest["receipt_sha256"]
        calls.append(list(indices))
        rows = []
        for index in indices:
            animal = index // config["sections_per_animal"]
            lineage = {
                name: (
                    f"{partition['identity_prefix']}-{name}-{index:08d}"
                    if name == "section_id"
                    else f"{partition['identity_prefix']}-{name}-{animal:04d}"
                )
                for name in launcher.FIVE_IDS
            }
            lineage["split"] = partition["split"]
            rows.append(
                {
                    "training_row_id": (
                        f"{partition['identity_prefix']}-row-{index:08d}"
                    ),
                    "receipt_sha256": f"{index + 1:064x}",
                    "lineage": lineage,
                    "selected_mode": launcher.TRAINABLE_MODES[index % 3],
                    "finite_psf_contract": {
                        "render_mode": "finite_boxcar",
                        "axial_sample_count": 9,
                    },
                }
            )
        payload = {
            "schema_version": "anatomy-tracker.frozen-training-rows/v6",
            "training_data_manifest_receipt_sha256": manifest[
                "receipt_sha256"
            ],
            "cache_manifest_receipt_sha256": manifest["receipt_sha256"],
            "generator_binding_receipt_sha256": manifest[
                "generator_binding"
            ]["receipt_sha256"],
            "generation_lineage_sha256": manifest["generator_binding"][
                "generation_lineage_sha256"
            ],
            "row_indices": list(indices),
            "training_row_ids": [row["training_row_id"] for row in rows],
            "training_row_receipts_sha256": [
                row["receipt_sha256"] for row in rows
            ],
            "rows": rows,
        }
        payload["selection_receipt_sha256"] = (
            launcher.finite_row_binding_v6.frozen_row_selection_receipt_v6(
                payload
            )
        )
        if tamper == "selection_receipt":
            payload["selection_receipt_sha256"] = "0" * 64
        elif tamper == "cache_manifest":
            payload["cache_manifest_receipt_sha256"] = "0" * 64
            payload["selection_receipt_sha256"] = (
                launcher.finite_row_binding_v6.frozen_row_selection_receipt_v6(
                    payload
                )
            )
        elif tamper == "row_ids":
            payload["training_row_ids"][0] = "different-row"
            payload["selection_receipt_sha256"] = (
                launcher.finite_row_binding_v6.frozen_row_selection_receipt_v6(
                    payload
                )
            )
        return payload

    monkeypatch.setattr(
        launcher.finite_row_binding_v6,
        "load_frozen_row_cache_manifest_v6",
        load_manifest,
    )
    monkeypatch.setattr(
        launcher.finite_row_binding_v6,
        "load_frozen_training_rows_v6",
        load_rows,
    )
    return calls


def test_configuration_is_exact_i_only_and_copy_safe():
    config = launcher.substantive_finite_data_configuration_v6()
    train = config["partitions"]["training"]
    development = config["partitions"]["internal_development"]

    assert inspect.signature(
        launcher.substantive_finite_data_configuration_v6
    ).parameters == {}
    assert inspect.signature(
        launcher.generate_substantive_finite_data_v6
    ).parameters == {}
    assert config["receipt_sha256"] == launcher._hash_json(
        {key: value for key, value in config.items() if key != "receipt_sha256"}
    )
    assert config["output_shape_h_w"] == [96, 96]
    assert config["sections_per_animal"] == 16
    assert config["render_mode"] == "finite_boxcar"
    assert config["axial_sample_count"] == 9
    assert (train["pose_row_count"], train["joint_row_count"]) == (3072, 2048)
    assert (train["row_count"], train["animal_count"]) == (5120, 320)
    assert train["mode_counts"] == {
        "smart-brush-accurate": 1707,
        "smart-brush-imperfect": 1707,
        "smart-brush-absent": 1706,
    }
    assert (
        development["pose_row_count"],
        development["joint_row_count"],
    ) == (384, 256)
    assert (development["row_count"], development["animal_count"]) == (
        640,
        40,
    )
    assert development["mode_counts"] == {
        "smart-brush-accurate": 214,
        "smart-brush-imperfect": 213,
        "smart-brush-absent": 213,
    }
    assert len(
        {
            partition[name]
            for partition in config["partitions"].values()
            for name in ("pose_root_seed_uint64", "joint_root_seed_uint64")
        }
    ) == 4
    assert len(
        {partition["identity_prefix"] for partition in config["partitions"].values()}
    ) == 2
    assert len(
        {partition["generation_run_id"] for partition in config["partitions"].values()}
    ) == 2
    for path in (
        config["output_root"],
        config["temp_root"],
        config["summary_path"],
        config["atlas_sources"]["template"]["path"],
        config["atlas_sources"]["annotation"]["path"],
        train["cache_directory"],
        development["cache_directory"],
    ):
        assert os.path.splitdrive(path)[0].upper() == "I:"
    for name in (
        "prior_model_weight_dependencies",
        "prior_feature_dependencies",
        "prior_pseudolabel_dependencies",
    ):
        assert config[name] == []
    assert config["public_benchmark_accessed"] is False

    config["partitions"]["training"]["row_count"] = 1
    assert launcher.substantive_finite_data_configuration_v6()["partitions"][
        "training"
    ]["row_count"] == 5120


def test_make_partition_binding_forwards_the_exact_real_contract():
    config = launcher.substantive_finite_data_configuration_v6()
    partition = config["partitions"]["training"]
    prepared = {
        "prepared_context_sha256": launcher.PREPARED_CONTEXT_SHA256,
        "support_index": {
            "support_index_sha256": launcher.SUPPORT_INDEX_SHA256
        },
    }
    commit = "1" * 40

    binding = launcher._make_partition_binding(
        prepared, partition, config, commit
    )
    composite = binding["generation_config"]
    pose = composite["component_generation_configs"][
        "finite_identity_pose_curriculum"
    ]
    joint = composite["component_generation_configs"][
        "finite_nonidentity_joint_curriculum"
    ]

    assert composite["row_count"] == 5120
    assert composite["output_shape_h_w"] == [96, 96]
    assert composite["finite_psf_run_contract"]["axial_sample_count"] == 9
    assert (pose["row_count"], joint["row_count"]) == (3072, 2048)
    assert pose["identity_prefix"].endswith("-pose")
    assert joint["identity_prefix"].endswith("-joint")
    assert pose["split"] == joint["split"] == "train"
    assert joint["render_mode"] == "finite_boxcar"
    assert joint["nominal_cut_thickness_um"] is None
    for component in (pose, joint):
        assert component["sections_per_animal"] == 16
        assert component["finite_parent_generator_source_commit"] == commit
        assert component["finite_slab_generator_source_commit"] == commit
    assert binding["generation_lineage"]["source_commit"] == commit
    assert binding["generation_lineage"]["generation_run_id"] == partition[
        "generation_run_id"
    ]


@pytest.mark.parametrize("partition_name", ["training", "internal_development"])
def test_full_v6_replay_covers_every_row_and_exact_identity(partition_name, monkeypatch):
    config = launcher.substantive_finite_data_configuration_v6()
    partition = config["partitions"][partition_name]
    calls = _install_fake_v6_replay(monkeypatch, partition, config)

    record, identities = launcher._full_replay_partition(
        Path(partition["cache_directory"]),
        partition,
        config,
        "a" * 64,
    )

    assert [index for chunk in calls for index in chunk] == list(
        range(partition["row_count"])
    )
    assert all(0 < len(chunk) <= config["full_replay_chunk_size"] for chunk in calls)
    assert record["full_row_replay"] is True
    assert record["row_count"] == partition["row_count"]
    assert record["animal_count"] == partition["animal_count"]
    assert record["mode_counts"] == partition["mode_counts"]
    assert record["identity_counts"] == {
        "animal_id": partition["animal_count"],
        "specimen_id": partition["animal_count"],
        "experiment_id": partition["animal_count"],
        "synthetic_animal_id": partition["animal_count"],
        "section_id": partition["row_count"],
    }
    assert {name: len(values) for name, values in identities.items()} == record[
        "identity_counts"
    ]
    for name in (
        "selection_receipts_sha256",
        "ordered_training_row_ids_sha256",
        "ordered_training_row_receipts_sha256",
    ):
        assert len(record[name]) == 64


@pytest.mark.parametrize(
    "tamper", ["selection_receipt", "cache_manifest", "row_ids"]
)
def test_full_v6_replay_rejects_selection_cross_link_tamper(tamper, monkeypatch):
    config = launcher.substantive_finite_data_configuration_v6()
    partition = config["partitions"]["internal_development"]
    _install_fake_v6_replay(monkeypatch, partition, config, tamper=tamper)

    with pytest.raises(RuntimeError, match="different selection"):
        launcher._full_replay_partition(
            Path(partition["cache_directory"]),
            partition,
            config,
            "a" * 64,
        )


def test_generation_uses_resumable_cache_and_requires_clean_freeze_audit(monkeypatch):
    config = launcher.substantive_finite_data_configuration_v6()
    partition = config["partitions"]["internal_development"]
    prepared = object()
    commit = "2" * 40
    binding = {"receipt_sha256": "b" * 64}
    calls = []

    monkeypatch.setattr(
        launcher,
        "_make_partition_binding",
        lambda *arguments: binding,
    )

    def resume(cache_directory, prepared_context, supplied_binding, *, chunk_size):
        calls.append(
            (cache_directory, prepared_context, supplied_binding, chunk_size)
        )
        manifest = {
            "status": "FROZEN",
            "row_count": partition["row_count"],
            "generator_binding": binding,
            "generation_lineage": {"source_commit": commit},
            "receipt_sha256": "a" * 64,
            "prior_model_weight_dependencies": [],
            "prior_feature_dependencies": [],
            "prior_pseudolabel_dependencies": [],
        }
        audit = {
            "row_count": partition["row_count"],
            "manifest_receipt_sha256": "a" * 64,
            "generator_binding_receipt_sha256": "b" * 64,
            "all_rows_authenticated": True,
            "learned_dependencies": [],
            "temporary_file_count": 0,
            "render_mode": "finite_boxcar",
            "axial_sample_count": 9,
        }
        return manifest, audit

    monkeypatch.setattr(
        launcher.finite_composite_v4,
        "resume_finite_composite_cache_v4",
        resume,
    )
    expected = ({"full_row_replay": True}, {name: set() for name in launcher.FIVE_IDS})
    monkeypatch.setattr(launcher, "_full_replay_partition", lambda *args: expected)

    assert launcher._generate_and_replay_partition(
        prepared, partition, config, commit
    ) == expected
    assert calls == [
        (
            Path(partition["cache_directory"]).resolve(),
            prepared,
            binding,
            16,
        )
    ]

    def dirty_resume(*args, **kwargs):
        manifest, audit = resume(*args, **kwargs)
        audit["temporary_file_count"] = 1
        return manifest, audit

    monkeypatch.setattr(
        launcher.finite_composite_v4,
        "resume_finite_composite_cache_v4",
        dirty_resume,
    )
    with pytest.raises(RuntimeError, match="freeze audit failed"):
        launcher._generate_and_replay_partition(
            prepared, partition, config, commit
        )


def test_clean_repository_source_accepts_only_exact_committed_launcher(monkeypatch):
    current = Path(launcher.__file__).read_bytes()

    def clean_run(arguments, **kwargs):
        if arguments[1:3] == ["status", "--porcelain=v1"]:
            return SimpleNamespace(stdout="")
        if arguments[1:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="a" * 40 + "\n")
        if arguments[1] == "show":
            return SimpleNamespace(stdout=current)
        raise AssertionError(arguments)

    monkeypatch.setattr(launcher.subprocess, "run", clean_run)
    source = launcher._clean_repository_source()
    assert source["git_commit"] == "a" * 40
    assert source["launcher_sha256"] == launcher._file_sha256(
        Path(launcher.__file__)
    )

    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="?? untracked.py\n"),
    )
    with pytest.raises(RuntimeError, match="clean committed worktree"):
        launcher._clean_repository_source()


@pytest.mark.parametrize("overlap_name", launcher.FIVE_IDS)
def test_cross_partition_id_overlap_never_writes_success(overlap_name, monkeypatch):
    source = {
        "git_commit": "3" * 40,
        "launcher_relative_path": "training/launcher.py",
        "launcher_sha256": "4" * 64,
    }
    monkeypatch.setattr(launcher, "_clean_repository_source", lambda: source)
    monkeypatch.setattr(launcher, "_require_i_environment", lambda config: None)
    monkeypatch.setattr(launcher, "_prepare_finite_context", lambda config: object())

    def partition_result(prepared, partition, config, commit):
        label = partition["data_role"]
        identities = {
            name: ({"overlap"} if name == overlap_name else {f"{label}-{name}"})
            for name in launcher.FIVE_IDS
        }
        return {"data_role": label}, identities

    monkeypatch.setattr(launcher, "_generate_and_replay_partition", partition_result)
    monkeypatch.setattr(
        launcher,
        "_write_or_verify_summary",
        lambda *args: pytest.fail("an overlapping run must not write success"),
    )

    with pytest.raises(RuntimeError, match="exact IDs overlap"):
        launcher.generate_substantive_finite_data_v6()


def test_success_summary_is_receipted_and_rechecks_source(monkeypatch):
    source = {
        "git_commit": "5" * 40,
        "launcher_relative_path": "training/launcher.py",
        "launcher_sha256": "6" * 64,
    }
    source_calls = []

    def clean_source():
        source_calls.append(True)
        return source

    monkeypatch.setattr(launcher, "_clean_repository_source", clean_source)
    monkeypatch.setattr(launcher, "_require_i_environment", lambda config: None)
    monkeypatch.setattr(launcher, "_prepare_finite_context", lambda config: object())

    def partition_result(prepared, partition, config, commit):
        label = partition["data_role"]
        return (
            {"data_role": label, "full_row_replay": True},
            {
                name: {f"{label}-{name}"}
                for name in launcher.FIVE_IDS
            },
        )

    monkeypatch.setattr(launcher, "_generate_and_replay_partition", partition_result)
    written = []
    monkeypatch.setattr(
        launcher,
        "_write_or_verify_summary",
        lambda path, summary: written.append((path, summary)),
    )

    summary = launcher.generate_substantive_finite_data_v6()
    assert len(source_calls) == 2
    assert written == [(Path(summary["configuration"]["summary_path"]), summary)]
    assert summary["receipt_sha256"] == launcher._hash_json(
        {key: value for key, value in summary.items() if key != "receipt_sha256"}
    )
    assert summary["all_partition_ids_disjoint"] is True
    assert summary["all_rows_replayed_by_finite_row_binding_v6"] is True
    assert summary["public_benchmark_accessed"] is False
    assert summary["release_qualifying"] is False


def test_i_only_environment_and_atomic_idempotent_summary(monkeypatch):
    config = launcher.substantive_finite_data_configuration_v6()
    monkeypatch.setenv("TEMP", config["temp_root"])
    monkeypatch.setenv("TMP", config["temp_root"])
    launcher._require_i_environment(config)
    monkeypatch.setenv("TMP", r"C:\not-allowed")
    with pytest.raises(RuntimeError, match="set TMP exactly"):
        launcher._require_i_environment(config)

    root = Path("I:/AnatomyTracker/pytest_tmp") / (
        "substantive_finite_data_v6_" + uuid.uuid4().hex
    )
    path = root / "summary.json"
    try:
        summary = launcher._with_receipt({"schema_version": "test", "rows": 2})
        launcher._write_or_verify_summary(path, summary)
        first = path.read_bytes()
        launcher._write_or_verify_summary(path, summary)
        assert path.read_bytes() == first
        assert json.loads(first)["receipt_sha256"] == summary["receipt_sha256"]
        with pytest.raises(RuntimeError, match="summary differs"):
            launcher._write_or_verify_summary(
                path, launcher._with_receipt({"schema_version": "changed"})
            )
        with pytest.raises(ValueError, match="remain on I"):
            launcher._write_or_verify_summary(
                Path(r"C:\forbidden-summary.json"), summary
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_clean_subprocess_import_has_no_training_or_model_dependency():
    script = r'''
import importlib
import json
import sys

importlib.import_module("training.run_arbitrary_plane_substantive_finite_data_v6")
tokens = (
    "candidate_bank",
    "training_bank",
    "staged_training",
    "training_runner",
    "staged_trainer",
    "joint_model",
    "recurrent_model",
)
forbidden = sorted(
    name for name in sys.modules
    if name.startswith("training.") and any(token in name for token in tokens)
)
torch = sys.modules.get("torch")
print(json.dumps({
    "forbidden": forbidden,
    "cuda_initialized": bool(torch is not None and torch.cuda.is_initialized()),
}))
'''
    environment = os.environ.copy()
    environment["TEMP"] = r"I:\AnatomyTracker\pytest_tmp"
    environment["TMP"] = r"I:\AnatomyTracker\pytest_tmp"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=launcher.REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(result.stdout)
    assert audit == {"forbidden": [], "cuda_initialized": False}
