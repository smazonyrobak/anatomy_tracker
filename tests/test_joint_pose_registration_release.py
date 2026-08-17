import subprocess
import sys

import pytest
import torch

from training import joint_pose_registration_release as release


def checkpoint_payload(*, latest_score=-1.25, with_selector=True):
    payload = {
        "format_version": release.JOINT_CHECKPOINT_FORMAT_VERSION,
        "model": {"weight": torch.zeros(2)},
        "ema": {"shadow": {"weight": torch.ones(2)}},
        "completed_views": 120,
        "best_validation_score": -1.25,
        "latest_validation": {"selection_score": latest_score},
        "generator_contract": {"contract": "test"},
    }
    if with_selector:
        payload["release_selection"] = {
            "state": release.RELEASE_STATE,
            "criterion": release.RELEASE_CRITERION,
            "validation_score": -1.25,
            "completed_views": 120,
        }
    return payload


def test_best_validation_checkpoint_selects_ema_with_hash_receipt(tmp_path):
    path = tmp_path / "best-validation.pt"
    torch.save(checkpoint_payload(), path)
    state, receipt = release.load_joint_release_state(path)
    assert torch.equal(state["weight"], torch.ones(2))
    assert receipt["selected_state"] == "ema.shadow"
    assert receipt["best_validation_score"] == -1.25
    assert len(receipt["checkpoint_sha256"]) == 64
    assert len(receipt["release_loader_source_sha256"]) == 64
    assert receipt["release_contract_sha256"] == release.RELEASE_CONTRACT_SHA256


@pytest.mark.parametrize("name", ("latest.pt", "views-000000120.pt"))
def test_ordinary_resume_checkpoints_are_not_release_eligible(tmp_path, name):
    path = tmp_path / name
    torch.save(checkpoint_payload(with_selector=False), path)
    with pytest.raises(ValueError, match="not a best-validation release checkpoint"):
        release.load_joint_release_state(path)


def test_selector_bearing_nonbest_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "latest.pt"
    torch.save(checkpoint_payload(latest_score=-2.0), path)
    with pytest.raises(ValueError, match="not bound to this best validation state"):
        release.load_joint_release_state(path)


def test_normalized_source_hash_is_line_ending_invariant(tmp_path):
    windows = tmp_path / "windows.py"
    unix = tmp_path / "unix.py"
    windows.write_bytes(b"x = 1\r\ny = 2\r\n")
    unix.write_bytes(b"x = 1\ny = 2\n")
    assert release.normalized_source_sha256(windows) == release.normalized_source_sha256(
        unix
    )


def test_release_module_import_does_not_load_training_data_modules():
    code = """
import sys
import training.joint_pose_registration_release
for name in (
    'training.joint_pose_registration_data',
    'training.joint_registered_data',
    'training.synthetic_registration',
    'training.train_joint_pose_registration',
):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
