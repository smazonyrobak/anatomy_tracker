"""Test whether frozen registration evidence ranks the correct atlas plane.

This is a Stage-1 diagnostic, not a confidence model and not a release gate.  The
    raw uint8 moving image is normalized to each candidate atlas mask with the same
    detached PCA/scale/centering affine used by training and runtime, then multiplied
    by the shared three-ring cosine feather.  Each candidate is registered independently
    and assigned the fixed evidence score::

    MIND residual
    + 0.35 * (1 - tissue-outline Dice)
    + 0.10 * normalized local-velocity magnitude
    + 1.00 * topology penalty
    + 0.05 * normalized similarity magnitude

Only images, tissue masks, candidate poses, and registrar predictions enter the
score.  In particular, no synthetic dense-flow target exists for a wrong plane
and no dense-flow target is read for any candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from training.dense_registration_model import (
    jacobian_determinant,
    modality_independent_descriptor,
    warp_tensor,
)
from training.joint_registered_data import mask_normalized_moving


FORMAT_VERSION = 3
SCORE_WEIGHTS = {
    "postwarp_mind_residual": 1.0,
    "outline_mismatch": 0.35,
    "normalized_velocity_magnitude": 0.10,
    "topology_penalty": 1.0,
    "normalized_similarity_magnitude": 0.05,
}
CSV_FIELDS = (
    "sample_index",
    "sample_id",
    "artifact_stratum",
    "candidate_index",
    "candidate_id",
    "candidate_kind",
    "is_true",
    "pose_ap_um",
    "pose_lr_deg",
    "pose_dv_deg",
    "offset_ap_um",
    "offset_lr_deg",
    "offset_dv_deg",
    "offset_axis",
    "offset_level",
    "status",
    "error",
    "postwarp_mind_residual",
    "outline_dice",
    "normalized_velocity_magnitude",
    "topology_penalty",
    "nonpositive_jacobian_fraction",
    "normalized_similarity_magnitude",
    "valid_overlap_fraction",
    "evidence_score",
    "dense_flow_target_used",
    "sample_complete",
    "true_rank",
    "sample_top1",
    "sample_reciprocal_rank",
)


def _masked_mean_per_item(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.to(values.dtype)
    return (values * mask).flatten(1).sum(1) / mask.expand_as(values).flatten(1).sum(1).clamp_min(1.0)


def _candidate_features(
    registrar: torch.nn.Module,
    fixed_candidates: torch.Tensor,
    moving_slice: torch.Tensor,
) -> list[dict[str, float]]:
    fixed_mask = fixed_candidates[:, 1:2].clamp(0.0, 1.0)
    aligned_image, aligned_mask, _, _ = mask_normalized_moving(
        moving_slice[:, :1],
        moving_slice[:, 1:2] > 0.5,
        fixed_mask > 0.5,
        apply_cosine_feather=True,
    )
    moving = torch.cat((aligned_image, aligned_mask.to(aligned_image.dtype)), dim=1)
    details = registrar.forward_with_details(fixed_candidates, moving)
    forward = details["fixed_to_moving_map"]
    warped = warp_tensor(moving, forward, padding_mode="zeros")

    warped_mask = warped[:, 1:2].clamp(0.0, 1.0)
    overlap = fixed_mask * warped_mask
    if bool((overlap.flatten(1).sum(1) < 1.0).any()):
        raise ValueError("candidate has no valid post-warp tissue overlap")

    fixed_descriptor = modality_independent_descriptor(fixed_candidates[:, :1])
    moving_descriptor = modality_independent_descriptor(warped[:, :1])
    mind = _masked_mean_per_item((fixed_descriptor - moving_descriptor).abs(), overlap)
    intersection = (fixed_mask * warped_mask).flatten(1).sum(1)
    outline_dice = (2.0 * intersection + 1e-6) / (
        fixed_mask.flatten(1).sum(1) + warped_mask.flatten(1).sum(1) + 1e-6
    )

    velocity = details["local_velocity"]
    height, width = fixed_candidates.shape[-2:]
    velocity_scale = max(0.15 * min(height, width), 1.0)
    velocity_magnitude = _masked_mean_per_item(
        torch.linalg.vector_norm(velocity, dim=1, keepdim=True), fixed_mask
    ) / velocity_scale

    determinant = jacobian_determinant(forward)
    topology_deficit = _masked_mean_per_item(
        torch.relu(0.05 - determinant), fixed_mask
    )
    nonpositive = _masked_mean_per_item((determinant <= 0.0).float(), fixed_mask)
    topology = topology_deficit + nonpositive

    similarity = details["similarity_parameters"]
    similarity_scale = similarity.new_tensor(
        (math.radians(15.0), width * 0.05, height * 0.05, math.log(1.1))
    )
    similarity_magnitude = torch.sqrt(
        ((similarity / similarity_scale).square()).mean(dim=1)
    )
    overlap_fraction = overlap.flatten(1).mean(1)

    score = (
        SCORE_WEIGHTS["postwarp_mind_residual"] * mind
        + SCORE_WEIGHTS["outline_mismatch"] * (1.0 - outline_dice)
        + SCORE_WEIGHTS["normalized_velocity_magnitude"] * velocity_magnitude
        + SCORE_WEIGHTS["topology_penalty"] * topology
        + SCORE_WEIGHTS["normalized_similarity_magnitude"] * similarity_magnitude
    )
    stacked = torch.stack(
        (
            mind,
            outline_dice,
            velocity_magnitude,
            topology,
            nonpositive,
            similarity_magnitude,
            overlap_fraction,
            score,
        ),
        dim=1,
    ).detach().cpu()
    if not bool(torch.isfinite(stacked).all()):
        raise ValueError("registrar produced non-finite premise evidence")
    names = (
        "postwarp_mind_residual",
        "outline_dice",
        "normalized_velocity_magnitude",
        "topology_penalty",
        "nonpositive_jacobian_fraction",
        "normalized_similarity_magnitude",
        "valid_overlap_fraction",
        "evidence_score",
    )
    return [dict(zip(names, map(float, values))) for values in stacked.tolist()]


def _offset_stratum(offset: Sequence[float]) -> tuple[str, str]:
    values = np.asarray(offset, dtype=np.float64)
    nonzero = np.flatnonzero(values != 0.0)
    if len(nonzero) == 0:
        return "none", "0"
    if len(nonzero) > 1:
        return "mixed", "mixed"
    axis = ("ap", "lr", "dv")[int(nonzero[0])]
    return axis, f"{abs(float(values[nonzero[0]])):g}"


def _base_row(sample: Mapping, sample_index: int, candidate_index: int) -> dict:
    pose = np.asarray(sample["candidate_pose"][candidate_index], dtype=np.float64)
    offset = np.asarray(sample["candidate_offset"][candidate_index], dtype=np.float64)
    kinds = list(sample["candidate_kind"])
    ids = list(sample.get("candidate_id", kinds))
    axis, level = _offset_stratum(offset)
    return {
        "sample_index": sample_index,
        "sample_id": str(sample.get("sample_id", sample_index)),
        "artifact_stratum": str(sample.get("artifact_stratum", "unspecified")),
        "candidate_index": candidate_index,
        "candidate_id": str(ids[candidate_index]),
        "candidate_kind": str(kinds[candidate_index]),
        "is_true": candidate_index == 0,
        "pose_ap_um": float(pose[0]),
        "pose_lr_deg": float(pose[1]),
        "pose_dv_deg": float(pose[2]),
        "offset_ap_um": float(offset[0]),
        "offset_lr_deg": float(offset[1]),
        "offset_dv_deg": float(offset[2]),
        "offset_axis": axis,
        "offset_level": level,
        "status": "pending",
        "error": "",
        "postwarp_mind_residual": None,
        "outline_dice": None,
        "normalized_velocity_magnitude": None,
        "topology_penalty": None,
        "nonpositive_jacobian_fraction": None,
        "normalized_similarity_magnitude": None,
        "valid_overlap_fraction": None,
        "evidence_score": None,
        "dense_flow_target_used": False,
    }


@torch.inference_mode()
def evaluate_joint_registration_premise(
    registrar: torch.nn.Module,
    samples: Iterable[Mapping],
    *,
    candidate_chunk_size: int = 2,
    device: str | torch.device = "cpu",
    provenance: Mapping | None = None,
    run_folder: str | Path | None = None,
) -> dict:
    """Rank the true plane among each sample's initial and hard-negative planes.

    A sample with any failed candidate remains in every aggregate denominator and
    receives top-1 = 0 and reciprocal rank = 0.  Ties are resolved against the
    true plane.  The first candidate must be the true plane.
    """
    if candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be positive")
    target = torch.device(device)
    registrar = registrar.to(target).eval()
    rows: list[dict] = []
    sample_metrics: list[dict] = []

    for sample_index, sample in enumerate(samples):
        fixed = torch.as_tensor(sample["candidate_fixed"], dtype=torch.float32)
        moving = torch.as_tensor(sample["moving_input"], dtype=torch.float32)
        if moving.ndim == 3:
            moving = moving.unsqueeze(0)
        count = len(fixed)
        if count < 2 or moving.shape[0] != 1:
            raise ValueError("each premise sample needs one moving image and at least two candidates")
        if len(sample["candidate_pose"]) != count or len(sample["candidate_offset"]) != count:
            raise ValueError("candidate tensors and pose metadata have different lengths")
        if len(sample["candidate_kind"]) != count or str(sample["candidate_kind"][0]) != "true":
            raise ValueError("candidate_kind must begin with the single true plane")

        sample_rows = [_base_row(sample, sample_index, index) for index in range(count)]
        for start in range(0, count, candidate_chunk_size):
            stop = min(start + candidate_chunk_size, count)
            try:
                features = _candidate_features(
                    registrar,
                    fixed[start:stop].to(target),
                    moving.to(target),
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                for row in sample_rows[start:stop]:
                    row.update(status="failed", error=message)
            else:
                for row, values in zip(sample_rows[start:stop], features):
                    row.update(status="ok", **values)

        complete = all(row["status"] == "ok" for row in sample_rows)
        if complete:
            true_score = float(sample_rows[0]["evidence_score"])
            rank = 1 + sum(
                float(row["evidence_score"]) <= true_score for row in sample_rows[1:]
            )
            reciprocal_rank = 1.0 / rank
            top1 = rank == 1
        else:
            rank = count
            reciprocal_rank = 0.0
            top1 = False
        for row in sample_rows:
            row.update(
                sample_complete=complete,
                true_rank=rank,
                sample_top1=top1,
                sample_reciprocal_rank=reciprocal_rank,
            )
        rows.extend(sample_rows)
        sample_metrics.append(
            {
                "sample_index": sample_index,
                "sample_id": sample_rows[0]["sample_id"],
                "candidate_count": count,
                "complete": complete,
                "true_rank": rank,
                "top1": top1,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    if not sample_metrics:
        raise ValueError("premise evaluation received no samples")
    margins: dict[tuple[str, str, str], dict[str, list[float] | int]] = defaultdict(
        lambda: {"total": 0, "values": []}
    )
    by_sample: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_sample[int(row["sample_index"])].append(row)
    for sample_rows in by_sample.values():
        true = sample_rows[0]
        for row in sample_rows[1:]:
            key = (row["candidate_kind"], row["offset_axis"], row["offset_level"])
            margins[key]["total"] += 1
            if true["status"] == row["status"] == "ok":
                margins[key]["values"].append(
                    float(row["evidence_score"]) - float(true["evidence_score"])
                )
    stratified = []
    for (kind, axis, level), values in sorted(margins.items()):
        observed = np.asarray(values["values"], dtype=np.float64)
        stratified.append(
            {
                "candidate_kind": kind,
                "offset_axis": axis,
                "offset_level": level,
                "candidate_count": int(values["total"]),
                "valid_margin_count": int(len(observed)),
                "mean_margin": float(observed.mean()) if len(observed) else None,
                "median_margin": float(np.median(observed)) if len(observed) else None,
                "true_better_rate": float((observed > 0.0).mean()) if len(observed) else None,
            }
        )

    sample_count = len(sample_metrics)
    failed_candidates = sum(row["status"] != "ok" for row in rows)
    result = {
        "format_version": FORMAT_VERSION,
        "status": "diagnostic_complete",
        "interpretation": (
            "Uncalibrated Stage-1 premise evidence only; scores and margins are not "
            "probabilities, confidence estimates, benchmark claims, or release gates."
        ),
        "dense_flow_supervision": "not read for true, initial, or hard-negative candidates",
        "score": {
            "lower_is_better": True,
            "candidate_preprocessing": (
                "raw uint8/255 followed by candidate-specific detached runtime "
                "outline PCA/orientation, isotropic span scaling, centering, and "
                "the shared three-ring cosine feather"
            ),
            "weights": dict(SCORE_WEIGHTS),
            "feature_definitions": {
                "postwarp_mind_residual": "mean absolute six-neighbour MIND residual over valid post-warp tissue overlap",
                "outline_dice": "soft Dice between candidate atlas tissue and warped moving tissue",
                "normalized_velocity_magnitude": "mean local displacement magnitude divided by 0.15 times the smaller image dimension",
                "topology_penalty": "mean ReLU(0.05-Jacobian) plus nonpositive-Jacobian fraction inside candidate tissue",
                "normalized_similarity_magnitude": "residual RMS angle/translation/log-scale effort after candidate-specific outline normalization, normalized by 15 deg, 5% width/height, and log(1.1)",
            },
        },
        "provenance": dict(provenance or {}),
        "metrics": {
            "sample_count": sample_count,
            "candidate_count": len(rows),
            "complete_sample_count": sum(item["complete"] for item in sample_metrics),
            "failed_candidate_count": failed_candidates,
            "candidate_failure_rate": failed_candidates / len(rows),
            "top1_accuracy": sum(item["top1"] for item in sample_metrics) / sample_count,
            "mean_reciprocal_rank": sum(item["reciprocal_rank"] for item in sample_metrics) / sample_count,
            "mean_true_rank": sum(item["true_rank"] for item in sample_metrics) / sample_count,
        },
        "offset_stratified_margins": stratified,
        "sample_metrics": sample_metrics,
        "candidate_rows": rows,
    }
    if run_folder is not None:
        write_evaluation_artifacts(result, run_folder)
    return result


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_evaluation_artifacts(result: Mapping, run_folder: str | Path) -> tuple[Path, Path]:
    folder = Path(run_folder)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "premise-candidates.csv"
    json_path = folder / "premise-summary.json"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    with csv_temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(row.get(key)) for key in CSV_FIELDS}
            for row in result["candidate_rows"]
        )
    os.replace(csv_temporary, csv_path)

    summary = dict(result)
    summary.pop("candidate_rows", None)
    summary["artifacts"] = {
        "per_candidate_csv": csv_path.name,
        "per_candidate_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    json_temporary = json_path.with_suffix(".json.tmp")
    json_temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(json_temporary, json_path)
    return json_path, csv_path


def _production_samples(manifest: Mapping, generator) -> Iterator[dict]:
    from training.joint_pose_registration_data import registration_manifest
    from training.synthetic_registration import BREGMA_AP_INDEX, VOXEL_UM

    base = registration_manifest(manifest)
    pair = generator.batch(base, qa=True)
    true_pose = np.asarray(manifest["true_pose"], dtype=np.float32)
    initial_offset = np.asarray(manifest["initial_pose_offset"], dtype=np.float32)
    hard_offsets = np.asarray(manifest["wrong_candidate_offset"], dtype=np.float32)
    for sample_index in range(len(true_pose)):
        offsets = np.concatenate(
            (
                np.zeros((1, 3), np.float32),
                initial_offset[sample_index : sample_index + 1],
                hard_offsets[sample_index],
            ),
            axis=0,
        )
        poses = true_pose[sample_index] + offsets
        rendered_image, rendered_mask, _ = generator.render_planes(
            torch.as_tensor(
                BREGMA_AP_INDEX - poses[:, 0] / VOXEL_UM,
                device=generator.device,
                dtype=torch.float32,
            ),
            torch.as_tensor(poses[:, 1], device=generator.device),
            torch.as_tensor(poses[:, 2], device=generator.device),
        )
        fixed_candidates = torch.cat((rendered_image, rendered_mask.float()), dim=1)
        moving_input = torch.cat(
            (
                pair["moving_raw_uint8"][sample_index : sample_index + 1].float() / 255.0,
                pair["moving_model_mask"][sample_index : sample_index + 1].float(),
            ),
            dim=1,
        )
        hard_count = len(hard_offsets[sample_index])
        yield {
            "sample_id": f"{manifest['joint_manifest_sha256']}:{sample_index}",
            "artifact_stratum": manifest["artifact_stratum"],
            "moving_input": moving_input,
            "candidate_fixed": fixed_candidates,
            "candidate_pose": poses,
            "candidate_offset": offsets,
            "candidate_kind": ["true", "initial", *(["hard_negative"] * hard_count)],
            "candidate_id": [
                "true",
                "initial",
                *(f"hard_negative_{index:03d}" for index in range(hard_count)),
            ],
        }


def run_production_evaluation(
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    joint_manifest_path: str | Path,
    atlas: str | Path,
    run_folder: str | Path,
    device: str = "cuda",
    candidate_chunk_size: int = 2,
) -> dict:
    """Load the frozen EMA/config and evaluate only a declared development manifest."""
    from training.dense_registration_release import checkpoint_identity, sha256_file
    from training.joint_pose_registration_data import load_joint_manifest
    from training.synthetic_registration import SyntheticRegistrationGenerator
    from training.train_dense_registration import model_from_checkpoint

    manifest = load_joint_manifest(joint_manifest_path)
    if str(manifest["split"]) not in {"train", "validation"}:
        raise ValueError("premise evaluation is development-only and cannot open a hidden/test split")
    identity, checkpoint = checkpoint_identity(checkpoint_path)
    if identity["checkpoint_file_sha256"] != expected_checkpoint_sha256.lower():
        raise ValueError("dense-registration checkpoint SHA-256 differs from the declared checkpoint")
    registrar, loaded = model_from_checkpoint(checkpoint_path, device, use_ema=True)
    if loaded["model_config"] != checkpoint["model_config"]:
        raise ValueError("loaded dense-registration model config differs from verified checkpoint")
    generator = SyntheticRegistrationGenerator(atlas, device)
    if checkpoint["generator_contract"] != generator.contract:
        raise ValueError("dense checkpoint and installed atlas generator contracts differ")
    provenance = {
        "checkpoint": identity,
        "joint_manifest_path": str(Path(joint_manifest_path).resolve()),
        "joint_manifest_file_sha256": sha256_file(joint_manifest_path),
        "joint_manifest_sha256": manifest["joint_manifest_sha256"],
        "split": manifest["split"],
        "seed": int(manifest["seed"]),
        "artifact_stratum": manifest["artifact_stratum"],
        "atlas": str(Path(atlas).resolve()),
        "generator_contract_sha256": generator.contract["contract_sha256"],
        "weights": "frozen EMA",
        "candidate_canvas_preprocessing": (
            "raw uint8/255; shared mask_normalized_moving runtime-outline affine; "
            "shared three-ring cosine feather"
        ),
        "hidden_benchmark_access": False,
    }
    return evaluate_joint_registration_premise(
        registrar,
        _production_samples(manifest, generator),
        candidate_chunk_size=candidate_chunk_size,
        device=device,
        provenance=provenance,
        run_folder=run_folder,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--joint-manifest", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate-chunk-size", type=int, default=2)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_production_evaluation(
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        joint_manifest_path=args.joint_manifest,
        atlas=args.atlas,
        run_folder=args.run_folder,
        device=args.device,
        candidate_chunk_size=args.candidate_chunk_size,
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
