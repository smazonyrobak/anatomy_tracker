"""Dependency-light validation and loading of joint-model release checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


JOINT_CHECKPOINT_FORMAT_VERSION = 2
RELEASE_STATE = "ema.shadow"
RELEASE_CRITERION = "validation_selection_score"


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized_source_sha256(path: str | Path) -> str:
    source = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(source).hexdigest()


def payload_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


RELEASE_CONTRACT = {
    "checkpoint_format_version": JOINT_CHECKPOINT_FORMAT_VERSION,
    "selected_state": RELEASE_STATE,
    "selection_criterion": RELEASE_CRITERION,
    "best_state_binding": "selector score == payload best == latest validation score",
}
RELEASE_CONTRACT_SHA256 = payload_sha256(RELEASE_CONTRACT)


def load_joint_release_state(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict]:
    """Return only the validation-selected EMA state and its hash-bound receipt."""
    path = Path(checkpoint_path).resolve()
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != JOINT_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("joint checkpoint format differs from the release contract")
    selection = payload.get("release_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("state") != RELEASE_STATE
        or selection.get("criterion") != RELEASE_CRITERION
    ):
        raise ValueError("joint checkpoint is not a best-validation release checkpoint")
    latest_score = float(
        payload.get("latest_validation", {}).get("selection_score", math.nan)
    )
    best_score = float(payload.get("best_validation_score", math.nan))
    if (
        int(selection.get("completed_views", -1))
        != int(payload.get("completed_views", -2))
        or not math.isclose(
            float(selection.get("validation_score", math.nan)),
            best_score,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(latest_score, best_score, rel_tol=0.0, abs_tol=0.0)
    ):
        raise ValueError("joint release selector is not bound to this best validation state")
    state = payload.get("ema", {}).get("shadow")
    if not isinstance(state, dict):
        raise ValueError("joint checkpoint has no EMA shadow state")
    receipt = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": file_sha256(path),
        "format_version": payload["format_version"],
        "selected_state": RELEASE_STATE,
        "selection_criterion": RELEASE_CRITERION,
        "completed_views": int(payload["completed_views"]),
        "best_validation_score": best_score,
        "generator_contract_sha256": payload_sha256(
            payload.get("generator_contract", {})
        ),
        "release_contract_sha256": RELEASE_CONTRACT_SHA256,
        "release_loader_source_sha256": normalized_source_sha256(__file__),
    }
    return state, receipt
