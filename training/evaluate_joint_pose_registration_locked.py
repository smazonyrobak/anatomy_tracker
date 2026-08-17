"""Public and reproducible locally locked evaluation for the joint benchmark.

The locally locked runner is fail-closed within one workspace, but is not a
cryptographic one-shot or publication-grade hidden test. That requires an
external custodian or service with a signed secret cohort.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import secrets
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from source.atlas_pose_runtime import (
    atlas_pose_preprocessing_contract_sha256,
    preprocess_atlas_pose_image,
)
from training import joint_pose_registration_locked_data as locked
from training.joint_pose_registration_release import load_joint_release_state


EVALUATION_SCHEMA_VERSION = "joint-locked-evaluation-v1"
MAP_SPACE = "source-model-canvas"
DEFAULT_ERROR_THRESHOLD_UM = 150.0
FAILURE_POSE_ERROR = (5000.0, 70.0, 70.0)
FAILURE_DISTANCE_UM = 10000.0
FAILURE_PIXEL_DISTANCE = math.hypot(*locked.MODEL_SHAPE)
FAILURE_LOG_JACOBIAN_STD = 20.0
CCF_ML_DV_SHAPE = (456.0, 320.0)
SEALED_TOTAL_CASES = 8192
SEALED_CASES_PER_STRATUM = SEALED_TOTAL_CASES // len(locked.SEVERITIES)
SEALED_NEGATIVES_PER_CASE = 6
_LOCAL_LOCKED_RUN_CAPABILITY = object()
RISK_LATTICE_OFFSETS = (
    (0.0, 0.0, 0.0),
    (-25.0, 0.0, 0.0),
    (25.0, 0.0, 0.0),
    (0.0, -0.25, 0.0),
    (0.0, 0.25, 0.0),
    (0.0, 0.0, -0.25),
    (0.0, 0.0, 0.25),
)
RISK_SCORE_CONTRACT = {
    "version": "predicted-pose-local-compatibility-entropy-v1",
    "offsets_ap_um_lr_deg_dv_deg": RISK_LATTICE_OFFSETS,
    "boundary_rule": (
        "use symmetric +/- one-step neighbors when possible; otherwise use "
        "one-step and two-step inward distinct neighbors"
    ),
    "candidate_invocation": "independent one-candidate compatibility calls",
    "score": "one minus softmax probability assigned to the center candidate",
    "interpretation": "monotone ordering score only; not a calibrated probability",
}
RISK_SCORE_CONTRACT_SHA256 = locked._payload_sha256(RISK_SCORE_CONTRACT)
PAIRED_METRIC_DOMAIN = "challenged_visible_common_support"


def _risk_candidate_poses(center: torch.Tensor) -> torch.Tensor:
    center = torch.as_tensor(center).reshape(3)
    limits = ((locked.AP_RANGE_UM[0], locked.AP_RANGE_UM[1]), (-35.0, 35.0), (-35.0, 35.0))
    steps = (25.0, 0.25, 0.25)
    if any(not low <= float(value) <= high for value, (low, high) in zip(center, limits)):
        raise ValueError("predicted pose lies outside the canonical AtlasPose domain")
    candidates = [center.clone()]
    for axis, (step, (low, high)) in enumerate(zip(steps, limits)):
        value = float(center[axis])
        offsets = (
            (-step, step)
            if value - step >= low and value + step <= high
            else ((step, 2.0 * step) if value - step < low else (-step, -2.0 * step))
        )
        for offset in offsets:
            candidate = center.clone()
            candidate[axis] += offset
            candidates.append(candidate)
    result = torch.stack(candidates)
    if len(torch.unique(result, dim=0)) != len(result):
        raise RuntimeError("risk lattice contains duplicate boundary candidates")
    return result


def _compatibility_risk_score(logits: torch.Tensor) -> torch.Tensor:
    logits = torch.as_tensor(logits).reshape(-1)
    if len(logits) != len(RISK_LATTICE_OFFSETS) or not bool(torch.isfinite(logits).all()):
        raise ValueError("risk lattice requires seven finite compatibility logits")
    return 1.0 - torch.softmax(logits, dim=0)[0]


def _file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluator_source_sha256() -> str:
    return _file_sha256(__file__)


def evaluator_dependency_tree() -> dict:
    repository = Path(__file__).resolve().parents[1]
    sources = {
        "atlas_pose_runtime.py": repository / "source" / "atlas_pose_runtime.py",
        "dense_registration_preprocessing.py": (
            repository / "source" / "dense_registration_preprocessing.py"
        ),
        "joint_pose_registration_release.py": (
            repository / "training" / "joint_pose_registration_release.py"
        ),
    }
    return {
        "sources": {name: _file_sha256(path) for name, path in sources.items()},
        "packages": {
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "opencv": str(cv2.__version__),
            "nrrd": str(getattr(locked.nrrd, "__version__", "unknown")),
        },
    }


def evaluator_dependency_tree_sha256() -> str:
    return locked._payload_sha256(evaluator_dependency_tree())


def _json_ready(value):
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(_json_ready(payload), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _exclusive_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(_json_ready(payload), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _candidate_hash(case_sha256: str, poses: torch.Tensor, contract_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"joint-locked-candidate-order-v1\0")
    digest.update(case_sha256.encode("ascii"))
    digest.update(contract_sha256.encode("ascii"))
    digest.update(np.ascontiguousarray(poses.detach().cpu(), dtype="<f4").tobytes())
    return digest.hexdigest()


def build_predictor_payload(
    benchmark: locked.LockedJointSyntheticBenchmark,
    batch: dict,
    *,
    shuffle_secret: bytes | None = None,
    view: str = "challenged",
) -> tuple[dict, list[dict]]:
    """Expose deployable inputs and evaluator-owned shuffled ranking candidates only."""
    if view not in {"challenged", "reference"}:
        raise ValueError("view must be challenged or reference")
    shuffle_secret = secrets.token_bytes(32) if shuffle_secret is None else shuffle_secret
    count = len(batch["pose_view"])
    ordered_poses = []
    candidate_hashes = []
    true_indices = []
    lattice = np.asarray(
        sorted(
            np.ndindex(5, 5, 5),
            key=lambda point: (
                sum((value - 2) ** 2 for value in point),
                *(value - 2 for value in point),
            ),
        ),
        dtype=np.float32,
    ) - 2.0
    split_pool = locked._split_indices(
        str(batch["split"]), allow_sealed=str(batch["split"]) == "sealed-test"
    )
    for item in range(count):
        candidate_count = int(batch["negative_pose"].shape[1]) + 1
        if candidate_count > len(lattice):
            raise ValueError("locked ranking lattice does not contain enough candidates")
        units = lattice[:candidate_count]
        case_hash = str(batch["case_sha256"][item])
        seed = int.from_bytes(
            hashlib.sha256(
                shuffle_secret + case_hash.encode("ascii") + b":candidate-order-v2"
            ).digest()[:8],
            "little",
        )
        rng = np.random.default_rng(seed)
        truth = batch["pose"][item].detach().cpu().numpy()
        candidate_sets = truth[None, None] + (
            units[None] - units[:, None]
        ) * np.asarray((locked.VOXEL_UM, locked.NEGATIVE_TILT_DEG[0], locked.NEGATIVE_TILT_DEG[0]))
        centers = np.rint(
            locked.BREGMA_AP_INDEX - candidate_sets[:, :, 0] / locked.VOXEL_UM
        ).astype(np.int32)
        valid_pivots = np.flatnonzero(
            (candidate_sets[:, :, 0] >= locked.AP_RANGE_UM[0]).all(axis=1)
            & (candidate_sets[:, :, 0] <= locked.AP_RANGE_UM[1]).all(axis=1)
            & (np.abs(candidate_sets[:, :, 1:]) <= 35.0).all(axis=(1, 2))
            & np.isin(centers, split_pool).all(axis=1)
        )
        if not len(valid_pivots):
            raise ValueError("no domain-valid evaluator-private ranking lattice is available")
        pivot = int(rng.choice(valid_pivots))
        candidates = torch.as_tensor(
            candidate_sets[pivot], device=batch["pose"].device, dtype=batch["pose"].dtype
        )
        permutation = torch.as_tensor(rng.permutation(len(candidates)), device=candidates.device)
        poses = candidates[permutation]
        ordered_poses.append(poses)
        true_indices.append(int(torch.nonzero(permutation == pivot, as_tuple=False)[0]))
        candidate_hashes.append(
            _candidate_hash(
                case_hash, poses, batch["contract"]["contract_sha256"]
            )
        )
    candidate_poses = torch.stack(ordered_poses)
    flat = candidate_poses.reshape(-1, 3)
    candidate_fixed, candidate_masks, _ = benchmark.render_planes(
        locked.BREGMA_AP_INDEX - flat[:, 0] / locked.VOXEL_UM,
        flat[:, 1], flat[:, 2],
    )
    candidate_fixed = candidate_fixed.reshape(
        count, candidate_poses.shape[1], 1, *locked.MODEL_SHAPE
    )
    candidate_masks = candidate_masks.reshape_as(candidate_fixed)
    pose_raw_name = (
        "pose_view_raw_uint8" if view == "challenged" else "reference_pose_view_raw_uint8"
    )
    pose_mask_name = "pose_view_mask" if view == "challenged" else "reference_pose_view_mask"
    moving_raw_name = "moving_raw_uint8" if view == "challenged" else "reference_moving_raw_uint8"
    moving_mask_name = "moving_model_mask" if view == "challenged" else "reference_moving_model_mask"
    pose_images = torch.from_numpy(
        np.stack(
            [
                preprocess_atlas_pose_image(
                    image[0].detach().cpu().numpy(),
                    mask[0].detach().cpu().numpy(),
                )
                for image, mask in zip(
                    batch[pose_raw_name], batch[pose_mask_name]
                )
            ]
        )
    ).to(batch["pose_view"].device)
    payload = {
        "task": "end_to_end",
        "pose_image": pose_images,
        "pose_view_raw_uint8": batch[pose_raw_name],
        "pose_image_mask": batch[pose_mask_name],
        "moving_raw_uint8": batch[moving_raw_name],
        "moving_model_mask": batch[moving_mask_name],
        "benchmark_contract_sha256": batch["contract"]["contract_sha256"],
        "refiner_preprocessing": locked.REFINER_PREPROCESSING_CONTRACT,
        "dense_preprocessing_contract": locked.PREPROCESSING_CONTRACT_V2,
        "dense_mask_contract_sha256": locked.MASK_CONTRACT_SHA256,
        "pose_preprocessing_contract_sha256": atlas_pose_preprocessing_contract_sha256(),
        "evaluator_dependency_tree_sha256": evaluator_dependency_tree_sha256(),
    }
    expected = [
        {
            "candidate_poses": candidate_poses[item].clone(),
            "candidate_fixed": candidate_fixed[item].clone(),
            "candidate_fixed_mask": candidate_masks[item].clone(),
            "candidate_set_sha256": candidate_hashes[item],
            "true_index": true_indices[item],
        }
        for item in range(count)
    ]
    return payload, expected


def _run_predictor(
    predictor: Callable[[dict], list[dict]], payload: dict
) -> tuple[list[dict], dict]:
    tracemalloc.start()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            predictions = predictor(payload)
    finally:
        wall_time = time.perf_counter() - start
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return predictions, {
        "wall_time_seconds": wall_time,
        "python_peak_memory_bytes": peak_memory,
    }


def _single_case_payload(payload: dict, item: int) -> dict:
    count = len(payload["pose_image"])
    return {
        name: (value[item : item + 1] if isinstance(value, torch.Tensor) and len(value) == count else value)
        for name, value in payload.items()
    }


def _run_bound_predictor(
    benchmark: locked.LockedJointSyntheticBenchmark,
    predictor: Callable[[dict], object],
    payload: dict,
    expected_candidates: list[dict],
    *,
    exact_predictor: Callable[[dict], object] | None = None,
) -> tuple[list[dict], dict]:
    main_start = time.perf_counter()
    try:
        predictions, measured = _run_predictor(predictor, payload)
    except Exception as error:
        predictions = [
            {
                "status": "failed",
                "failure_reason": (
                    f"predictor failure: {type(error).__name__}: {error}"
                ),
                "provider": "failed-before-provider-receipt",
                "wall_time_seconds": 0.0,
                "peak_memory_bytes": 0,
            }
            for _ in expected_candidates
        ]
        measured = {
            "wall_time_seconds": time.perf_counter() - main_start,
            "python_peak_memory_bytes": 0,
        }
    if not isinstance(predictions, (list, tuple)):
        predictions = [predictions] * len(expected_candidates)
    else:
        predictions = list(predictions)
    if len(predictions) != len(expected_candidates):
        raise ValueError("predictor returned the wrong number of case receipts")
    predictions = [
        prediction
        if isinstance(prediction, dict)
        else {
            "status": "failed",
            "failure_reason": (
                "malformed end-to-end output: expected one dictionary per case, "
                f"received {type(prediction).__name__}"
            ),
            "provider": "malformed-output",
            "wall_time_seconds": 0.0,
            "peak_memory_bytes": 0,
        }
        for prediction in predictions
    ]
    exact_predictor = predictor if exact_predictor is None else exact_predictor
    private_tasks_start = time.perf_counter()
    case_payloads = [
        _single_case_payload(payload, item) for item in range(len(expected_candidates))
    ]
    for item, (prediction, expected) in enumerate(zip(predictions, expected_candidates)):
        if not isinstance(prediction, dict):
            continue
        reserved = {
            "candidate_poses", "candidate_set_sha256", "compatibility_logits",
            "risk_score", "risk_candidate_poses", "risk_compatibility_logits",
            "risk_score_contract_sha256",
            "exact_plane_pose", "exact_plane_fixed_to_source_model",
            "exact_plane_source_model_to_fixed", "exact_plane_map_domain_receipt",
        }
        if reserved.intersection(prediction):
            prediction["candidate_protocol_violation"] = True
            continue
        if prediction.get("status", "success") == "failed":
            prediction["risk_score"] = 1e9
            prediction["risk_score_contract_sha256"] = RISK_SCORE_CONTRACT_SHA256
            continue
        logits = []
        case_payload = case_payloads[item]
        for candidate_pose, candidate_fixed, candidate_mask in zip(
            expected["candidate_poses"],
            expected["candidate_fixed"],
            expected["candidate_fixed_mask"],
        ):
            trial = {
                **case_payload,
                "task": "compatibility",
                "candidate_pose": candidate_pose[None],
                "candidate_fixed": candidate_fixed[None],
                "candidate_fixed_mask": candidate_mask[None],
            }
            try:
                with torch.inference_mode():
                    score = predictor(trial)
                if isinstance(score, dict):
                    score = score["compatibility_logit"]
                score = torch.as_tensor(score).reshape(-1)
                if score.numel() != 1 or not bool(torch.isfinite(score).all()):
                    raise ValueError("compatibility scorer must return one finite logit")
                logits.append(score[0].to(candidate_pose.device))
            except Exception:
                logits.append(candidate_pose.new_tensor(float("nan")))
        prediction.update(
            candidate_poses=expected["candidate_poses"].clone(),
            candidate_set_sha256=expected["candidate_set_sha256"],
            compatibility_logits=torch.stack(logits),
        )
        try:
            center = torch.as_tensor(
                prediction["final_pose"],
                device=expected["candidate_poses"].device,
                dtype=expected["candidate_poses"].dtype,
            ).reshape(3)
            risk_poses = _risk_candidate_poses(center)
            risk_fixed, risk_masks, _ = benchmark.render_planes(
                locked.BREGMA_AP_INDEX - risk_poses[:, 0] / locked.VOXEL_UM,
                risk_poses[:, 1],
                risk_poses[:, 2],
            )
            risk_logits = []
            for candidate_pose, candidate_fixed, candidate_mask in zip(
                risk_poses, risk_fixed, risk_masks
            ):
                trial = {
                    **case_payload,
                    "task": "compatibility",
                    "candidate_pose": candidate_pose[None],
                    "candidate_fixed": candidate_fixed[None],
                    "candidate_fixed_mask": candidate_mask[None],
                }
                with torch.inference_mode():
                    score = predictor(trial)
                if isinstance(score, dict):
                    score = score["compatibility_logit"]
                score = torch.as_tensor(score).reshape(-1)
                if score.numel() != 1 or not bool(torch.isfinite(score).all()):
                    raise ValueError("compatibility scorer must return one finite logit")
                risk_logits.append(score[0].to(candidate_pose.device))
            risk_logits = torch.stack(risk_logits)
            risk_score = _compatibility_risk_score(risk_logits)
            prediction.update(
                risk_score=float(risk_score),
                risk_candidate_poses=risk_poses,
                risk_compatibility_logits=risk_logits,
                risk_score_contract_sha256=RISK_SCORE_CONTRACT_SHA256,
            )
        except Exception as error:
            prediction["risk_score"] = 1e9
            prediction["risk_score_contract_sha256"] = RISK_SCORE_CONTRACT_SHA256
            prediction["risk_score_failure_reason"] = (
                f"{type(error).__name__}: {error}"
            )
    # Truth is first disclosed only after every end-to-end, ranking, and risk call
    # for the complete payload has finished. A distinct frozen predictor instance
    # is supplied by locally locked qualification for this warp-only phase.
    for item, (prediction, expected) in enumerate(zip(predictions, expected_candidates)):
        if not isinstance(prediction, dict):
            continue
        case_payload = case_payloads[item]
        true_index = int(expected["true_index"])
        exact_trial = {
            **case_payload,
            "task": "exact_plane_registration",
            "candidate_pose": expected["candidate_poses"][true_index : true_index + 1],
            "candidate_fixed": expected["candidate_fixed"][true_index : true_index + 1],
            "candidate_fixed_mask": expected["candidate_fixed_mask"][
                true_index : true_index + 1
            ],
        }
        try:
            with torch.inference_mode():
                exact = exact_predictor(exact_trial)
            required = {
                "fixed_to_source_model",
                "source_model_to_fixed",
                "map_domain_receipt",
            }
            if not isinstance(exact, dict) or set(exact) != required:
                raise ValueError(
                    "exact-plane registration must return only the two source-model "
                    "maps and their domain receipt"
                )
            prediction.update(
                exact_plane_pose=exact_trial["candidate_pose"][0].clone(),
                exact_plane_fixed_to_source_model=exact[
                    "fixed_to_source_model"
                ],
                exact_plane_source_model_to_fixed=exact[
                    "source_model_to_fixed"
                ],
                exact_plane_map_domain_receipt=exact["map_domain_receipt"],
            )
        except Exception as error:
            prediction["exact_plane_failure_reason"] = (
                f"{type(error).__name__}: {error}"
            )
    measured["wall_time_seconds"] += time.perf_counter() - private_tasks_start
    return predictions, measured


def _manifest_slice(manifest: dict, start: int, stop: int) -> dict:
    """Materialize a bounded chunk without changing its frozen parent-case receipts."""
    count = int(manifest["sample_count"])
    result = {}
    for name, value in manifest.items():
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == count:
            result[name] = value[start:stop]
        else:
            result[name] = value
    result["sample_count"] = stop - start
    return result


def _as_tensor(value, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _pose_plane_errors(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    ml = (CCF_ML_DV_SHAPE[0] - 1.0) / 2.0
    dv = (CCF_ML_DV_SHAPE[1] - 1.0) / 2.0
    anchors = prediction.new_tensor(
        ((0.0, 0.0), (-ml, -dv), (ml, -dv), (-ml, dv), (ml, dv))
    )

    def ap_at(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        return pose[0] - locked.VOXEL_UM * (
            torch.tan(torch.deg2rad(pose[1])) * points[:, 0]
            + torch.tan(torch.deg2rad(pose[2])) * points[:, 1]
        )

    tre = (ap_at(anchors, prediction) - ap_at(anchors, truth)).abs()
    return {
        "plane_anchor_tre_um": float(tre[0]),
        "plane_corner_tre_mean_um": float(tre[1:].mean()),
        "plane_corner_tre_p95_um": float(torch.quantile(tre[1:], 0.95)),
        "plane_corner_tre_max_um": float(tre[1:].max()),
        "five_anchor_plane_distance_um": float(tre.mean()),
    }


def _label_interior(labels: torch.Tensor) -> torch.Tensor:
    interior = torch.ones_like(labels, dtype=torch.bool)
    interior[:, :, 0] = interior[:, :, -1] = False
    interior[:, :, :, 0] = interior[:, :, :, -1] = False
    interior[:, :, 1:] &= labels[:, :, 1:] == labels[:, :, :-1]
    interior[:, :, :-1] &= labels[:, :, :-1] == labels[:, :, 1:]
    interior[:, :, :, 1:] &= labels[:, :, :, 1:] == labels[:, :, :, :-1]
    interior[:, :, :, :-1] &= labels[:, :, :, :-1] == labels[:, :, :, 1:]
    return interior


def _region_dice(
    truth: torch.Tensor, estimate: torch.Tensor, valid: torch.Tensor
) -> list[float]:
    values = []
    for region in torch.unique(truth[valid]):
        if int(region) == 0:
            continue
        target = (truth == region) & valid
        predicted = (estimate == region) & valid
        denominator = target.sum() + predicted.sum()
        values.append(float(2.0 * (target & predicted).sum() / denominator.clamp_min(1)))
    return values


def _label_boundary(labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    boundary = torch.zeros_like(valid)
    horizontal = labels[..., 1:] != labels[..., :-1]
    vertical = labels[..., 1:, :] != labels[..., :-1, :]
    boundary[..., 1:] |= horizontal
    boundary[..., :-1] |= horizontal
    boundary[..., 1:, :] |= vertical
    boundary[..., :-1, :] |= vertical
    return boundary & valid & (labels != 0)


def _boundary_metrics(
    truth: torch.Tensor, estimate: torch.Tensor, valid: torch.Tensor, tolerance: int = 2
) -> dict[str, float]:
    truth_boundary = _label_boundary(truth, valid)
    estimate_boundary = _label_boundary(estimate, valid)
    if not bool(truth_boundary.any()) or not bool(estimate_boundary.any()):
        return {"boundary_f1_2px": float("nan"), "boundary_assd_px": float("nan"),
                "boundary_hd95_px": float("nan")}
    truth_dilated = F.max_pool2d(
        truth_boundary.float(), 2 * tolerance + 1, stride=1, padding=tolerance
    ) > 0.0
    estimate_dilated = F.max_pool2d(
        estimate_boundary.float(), 2 * tolerance + 1, stride=1, padding=tolerance
    ) > 0.0
    precision = (estimate_boundary & truth_dilated).sum() / estimate_boundary.sum()
    recall = (truth_boundary & estimate_dilated).sum() / truth_boundary.sum()
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)

    truth_np = truth_boundary[0, 0].detach().cpu().numpy()
    estimate_np = estimate_boundary[0, 0].detach().cpu().numpy()
    distance_to_truth = cv2.distanceTransform((~truth_np).astype(np.uint8), cv2.DIST_L2, 5)
    distance_to_estimate = cv2.distanceTransform(
        (~estimate_np).astype(np.uint8), cv2.DIST_L2, 5
    )
    distances = np.concatenate(
        (distance_to_truth[estimate_np], distance_to_estimate[truth_np])
    )
    return {
        "boundary_f1_2px": float(f1),
        "boundary_assd_px": float(distances.mean()),
        "boundary_hd95_px": float(np.quantile(distances, 0.95)),
    }


def _map_diagnostics(
    forward: torch.Tensor,
    inverse: torch.Tensor,
    fixed_valid: torch.Tensor,
    moving_valid: torch.Tensor,
) -> dict[str, float]:
    identity_fixed = locked._identity_grid(
        1, forward.shape[-2], forward.shape[-1], forward.device
    )
    identity_moving = locked._identity_grid(
        1, inverse.shape[-2], inverse.shape[-1], inverse.device
    )
    forward_cycle = locked.compose_pixel_maps(inverse, forward)
    inverse_cycle = locked.compose_pixel_maps(forward, inverse)
    forward_cycle_error = torch.linalg.vector_norm(
        forward_cycle - identity_fixed, dim=1
    )[fixed_valid[:, 0]]
    inverse_cycle_error = torch.linalg.vector_norm(
        inverse_cycle - identity_moving, dim=1
    )[moving_valid[:, 0]]

    def jacobian_metrics(pixel_map: torch.Tensor, mask: torch.Tensor, prefix: str):
        cells = (
            mask[:, :, :-1, :-1] | mask[:, :, :-1, 1:]
            | mask[:, :, 1:, :-1] | mask[:, :, 1:, 1:]
        )[:, 0]
        determinant = locked.jacobian_determinant(pixel_map)[cells]
        return {
            f"{prefix}_negative_jacobian_fraction": float(
                (determinant <= 0.0).float().mean()
            ),
            f"{prefix}_log_jacobian_std": float(
                torch.log(determinant.clamp_min(1e-6)).std(unbiased=False)
            ),
        }

    metrics = {
        "forward_inverse_cycle_mean_px": float(forward_cycle_error.mean()),
        "forward_inverse_cycle_p95_px": float(torch.quantile(forward_cycle_error, 0.95)),
        "inverse_forward_cycle_mean_px": float(inverse_cycle_error.mean()),
        "inverse_forward_cycle_p95_px": float(torch.quantile(inverse_cycle_error, 0.95)),
    }
    metrics.update(jacobian_metrics(forward, fixed_valid, "forward"))
    metrics.update(jacobian_metrics(inverse, moving_valid, "inverse"))
    return metrics


def _validate_map_receipt(
    record: dict,
    expected_pose: torch.Tensor,
    shape: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pose = _as_tensor(record["map_pose"], device).reshape(3)
    if not torch.allclose(pose, expected_pose, rtol=0.0, atol=1e-5):
        raise ValueError("stale map receipt: map_pose does not match its accepted pose")
    if record.get("map_space") != MAP_SPACE:
        raise ValueError("maps must be composed onto the source model canvas")
    if record.get("refiner_preprocessing") != locked.REFINER_PREPROCESSING_CONTRACT:
        raise ValueError("map receipt has the wrong refiner preprocessing contract")
    receipt = record.get("map_domain_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("an explicit map-domain receipt is required")
    if receipt.get("map_space") != MAP_SPACE:
        raise ValueError("map-domain receipt does not declare source-model-canvas space")
    if receipt.get("refiner_preprocessing") != locked.REFINER_PREPROCESSING_CONTRACT:
        raise ValueError("map-domain receipt has the wrong preprocessing contract")
    receipt_pose = _as_tensor(receipt["map_pose"], device).reshape(3)
    if not torch.allclose(receipt_pose, expected_pose, rtol=0.0, atol=1e-5):
        raise ValueError("stale map-domain receipt pose")
    if tuple(receipt.get("source_shape", ())) != tuple(shape):
        raise ValueError("map-domain receipt source_shape does not match the model canvas")
    forward = _as_tensor(record["fixed_to_source_model"], device).reshape(1, 2, *shape)
    inverse = _as_tensor(record["source_model_to_fixed"], device).reshape(1, 2, *shape)
    return forward, inverse


def _spatial_metrics(
    benchmark: locked.LockedJointSyntheticBenchmark,
    batch: dict,
    item: int,
    pose: torch.Tensor,
    map_record: dict,
) -> tuple[dict[str, float], list[float]]:
    device = batch["pose"].device
    forward, inverse = _validate_map_receipt(
        map_record, pose, locked.MODEL_SHAPE, device
    )
    _, predicted_mask, predicted_labels = benchmark.render_planes(
        locked.BREGMA_AP_INDEX - pose[:1, 0] / locked.VOXEL_UM,
        pose[:1, 1],
        pose[:1, 2],
    )
    moving_visible = batch["moving_visible_mask"][item : item + 1]
    moving_labels = batch["moving_labels"][item : item + 1]
    predicted_visible = predicted_mask & (
        locked.sample_at(moving_visible.float(), forward, "nearest") > 0.5
    )
    recovered_labels = locked._sample_labels(moving_labels, forward)
    exact = (recovered_labels == predicted_labels)[predicted_visible]
    interior_visible = predicted_visible & _label_interior(predicted_labels)
    interior_exact = (recovered_labels == predicted_labels)[interior_visible]
    dice = _region_dice(predicted_labels, recovered_labels, predicted_visible)
    bottom_count = max(1, math.ceil(0.30 * len(dice))) if dice else 0
    metrics = {
        "visible_region_correspondence": float(exact.float().mean()),
        "interior_region_correspondence": float(interior_exact.float().mean()),
        "visible_fraction": float(
            predicted_visible.sum() / predicted_mask.sum().clamp_min(1)
        ),
        "macro_region_dice": float(np.mean(dice)) if dice else float("nan"),
        "bottom_30_region_dice": (
            float(np.mean(sorted(dice)[:bottom_count])) if dice else float("nan")
        ),
    }
    metrics.update(_boundary_metrics(predicted_labels, recovered_labels, predicted_visible))
    metrics.update(
        _map_diagnostics(forward, inverse, predicted_visible, moving_visible)
    )
    return metrics, dice


def _exact_plane_metrics(
    benchmark: locked.LockedJointSyntheticBenchmark,
    batch: dict,
    item: int,
    prediction: dict,
    device: torch.device,
) -> dict[str, float]:
    names = (
        "exact_plane_pose",
        "exact_plane_fixed_to_source_model",
        "exact_plane_source_model_to_fixed",
        "exact_plane_map_domain_receipt",
    )
    if not all(prediction.get(name) is not None for name in names):
        raise ValueError("warp-only evaluation requires a complete exact-plane receipt")
    truth_pose = batch["pose"][item]
    exact_record = {
        "map_pose": prediction["exact_plane_pose"],
        "fixed_to_source_model": prediction["exact_plane_fixed_to_source_model"],
        "source_model_to_fixed": prediction["exact_plane_source_model_to_fixed"],
        "map_space": MAP_SPACE,
        "refiner_preprocessing": locked.REFINER_PREPROCESSING_CONTRACT,
        "map_domain_receipt": prediction["exact_plane_map_domain_receipt"],
    }
    forward, inverse = _validate_map_receipt(
        exact_record, truth_pose, locked.MODEL_SHAPE, device
    )
    expected_forward = batch["fixed_to_moving"][item : item + 1]
    expected_inverse = batch["moving_to_fixed"][item : item + 1]
    fixed_valid = batch["fixed_visible_mask"][item : item + 1]
    moving_valid = batch["moving_visible_mask"][item : item + 1]
    forward_error = torch.linalg.vector_norm(forward - expected_forward, dim=1)[
        fixed_valid[:, 0]
    ]
    inverse_error = torch.linalg.vector_norm(inverse - expected_inverse, dim=1)[
        moving_valid[:, 0]
    ]
    spatial, _ = _spatial_metrics(
        benchmark, batch, item, truth_pose[None], exact_record
    )
    metrics = {
        "warp_only_forward_endpoint_mean_px": float(forward_error.mean()),
        "warp_only_forward_endpoint_p95_px": float(torch.quantile(forward_error, 0.95)),
        "warp_only_inverse_endpoint_mean_px": float(inverse_error.mean()),
        "warp_only_inverse_endpoint_p95_px": float(torch.quantile(inverse_error, 0.95)),
    }
    metrics.update({f"warp_only_{name}": value for name, value in spatial.items()})
    return metrics


def _all_finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return not np.issubdtype(value.dtype, np.floating) or bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    comparison = positive[:, None] - negative[None, :]
    return float(((comparison > 0).sum() + 0.5 * (comparison == 0).sum()) / comparison.size)


def _binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if not labels.any():
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].mean())


def _aggregate(
    case_results: list[dict],
    error_threshold_um: float,
) -> dict:
    attempted = len(case_results)
    successful = [result for result in case_results if result["status"] == "success"]
    failures = attempted - len(successful)

    pose_penalties = dict(zip(("ap", "lr", "dv"), FAILURE_POSE_ERROR))
    metrics = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "attempted_count": attempted,
        "success_count": len(successful),
        "failure_count": failures,
        "failure_rate": failures / attempted,
        "nonfinite_output_count": sum(result.get("nonfinite", False) for result in case_results),
    }
    for axis in ("ap", "lr", "dv"):
        signed = np.asarray(
            [
                result["pose_signed_error"][axis]
                if result["status"] == "success"
                else pose_penalties[axis]
                for result in case_results
            ],
            dtype=np.float64,
        )
        absolute = np.abs(signed)
        unit = "um" if axis == "ap" else "deg"
        metrics[f"{axis}_mae_{unit}"] = float(absolute.mean())
        success_signed = [
            result["pose_signed_error"][axis] for result in successful
        ]
        metrics[f"{axis}_bias_{unit}"] = (
            float(np.mean(success_signed)) if success_signed else float("nan")
        )
        metrics[f"{axis}_p95_{unit}"] = float(np.quantile(absolute, 0.95))

    high_is_good = (
        "visible_region_correspondence", "interior_region_correspondence",
        "visible_fraction", "macro_region_dice", "bottom_30_region_dice",
        "boundary_f1_2px",
    )
    distance_penalty = {
        "boundary_assd_px": FAILURE_PIXEL_DISTANCE,
        "boundary_hd95_px": FAILURE_PIXEL_DISTANCE,
        "forward_inverse_cycle_mean_px": FAILURE_PIXEL_DISTANCE,
        "forward_inverse_cycle_p95_px": FAILURE_PIXEL_DISTANCE,
        "inverse_forward_cycle_mean_px": FAILURE_PIXEL_DISTANCE,
        "inverse_forward_cycle_p95_px": FAILURE_PIXEL_DISTANCE,
        "forward_negative_jacobian_fraction": 1.0,
        "inverse_negative_jacobian_fraction": 1.0,
        "forward_log_jacobian_std": FAILURE_LOG_JACOBIAN_STD,
        "inverse_log_jacobian_std": FAILURE_LOG_JACOBIAN_STD,
    }
    for name in high_is_good:
        metrics[f"end_to_end_{name}"] = float(
            sum(result["spatial"][name] for result in successful) / attempted
        )
    for name, penalty in distance_penalty.items():
        metrics[f"end_to_end_{name}"] = float(
            (
                sum(result["spatial"][name] for result in successful)
                + failures * penalty
            )
            / attempted
        )

    plane_aggregates = {
        "plane_anchor_tre_um": ("plane_anchor_tre_mean_um", "plane_anchor_tre_p95_um"),
        "plane_corner_tre_mean_um": (
            "plane_corner_tre_mean_um", "plane_corner_tre_case_p95_um"
        ),
        "plane_corner_tre_p95_um": (
            "plane_corner_within_case_p95_mean_um",
            "plane_corner_within_case_p95_case_p95_um",
        ),
        "plane_corner_tre_max_um": (
            "plane_corner_max_mean_um", "plane_corner_maximum_um"
        ),
        "five_anchor_plane_distance_um": (
            "five_anchor_plane_distance_mean_um",
            "five_anchor_plane_distance_p95_um",
        ),
    }
    for name, (mean_name, tail_name) in plane_aggregates.items():
        values = [
            result["plane"][name] if result["status"] == "success" else FAILURE_DISTANCE_UM
            for result in case_results
        ]
        metrics[mean_name] = float(np.mean(values))
        metrics[tail_name] = (
            float(np.max(values)) if tail_name.endswith("maximum_um")
            else float(np.quantile(values, 0.95))
        )

    pose_monotonic = [
        result["pose_monotonic"] if result["status"] == "success" else False
        for result in case_results
    ]
    correspondence_monotonic = [
        result["correspondence_monotonic"] if result["status"] == "success" else False
        for result in case_results
    ]
    metrics["pose_monotonic_case_rate"] = float(np.mean(pose_monotonic))
    metrics["correspondence_monotonic_case_rate"] = float(
        np.mean(correspondence_monotonic)
    )
    pose_transitions = [
        transition
        for result in successful
        for transition in result["pose_monotonic_transitions"]
    ]
    correspondence_transitions = [
        transition
        for result in successful
        for transition in result["correspondence_monotonic_transitions"]
    ]
    metrics["pose_monotonic_transition_rate"] = (
        float(np.mean(pose_transitions)) if pose_transitions else float("nan")
    )
    metrics["correspondence_monotonic_transition_rate"] = (
        float(np.mean(correspondence_transitions))
        if correspondence_transitions else float("nan")
    )
    metrics["true_plane_rank_1_rate"] = float(
        sum(result.get("true_plane_rank", math.inf) == 1 for result in successful)
        / attempted
    )
    metrics["true_plane_mrr"] = float(
        sum(1.0 / result["true_plane_rank"] for result in successful) / attempted
    )

    warp_high_is_good = (
        "warp_only_visible_region_correspondence",
        "warp_only_interior_region_correspondence",
        "warp_only_visible_fraction",
        "warp_only_macro_region_dice",
        "warp_only_bottom_30_region_dice",
        "warp_only_boundary_f1_2px",
    )
    warp_penalties = {
        "warp_only_forward_endpoint_mean_px": FAILURE_PIXEL_DISTANCE,
        "warp_only_forward_endpoint_p95_px": FAILURE_PIXEL_DISTANCE,
        "warp_only_inverse_endpoint_mean_px": FAILURE_PIXEL_DISTANCE,
        "warp_only_inverse_endpoint_p95_px": FAILURE_PIXEL_DISTANCE,
        "warp_only_boundary_assd_px": FAILURE_PIXEL_DISTANCE,
        "warp_only_boundary_hd95_px": FAILURE_PIXEL_DISTANCE,
        "warp_only_forward_inverse_cycle_mean_px": FAILURE_PIXEL_DISTANCE,
        "warp_only_forward_inverse_cycle_p95_px": FAILURE_PIXEL_DISTANCE,
        "warp_only_inverse_forward_cycle_mean_px": FAILURE_PIXEL_DISTANCE,
        "warp_only_inverse_forward_cycle_p95_px": FAILURE_PIXEL_DISTANCE,
        "warp_only_forward_negative_jacobian_fraction": 1.0,
        "warp_only_inverse_negative_jacobian_fraction": 1.0,
        "warp_only_forward_log_jacobian_std": FAILURE_LOG_JACOBIAN_STD,
        "warp_only_inverse_log_jacobian_std": FAILURE_LOG_JACOBIAN_STD,
    }
    warp_successful = [
        result for result in case_results
        if result.get("warp_only_status") == "success"
    ]
    warp_failures = attempted - len(warp_successful)
    metrics["warp_only_attempted_count"] = attempted
    metrics["warp_only_receipt_count"] = len(warp_successful)
    metrics["warp_only_failure_count"] = warp_failures
    metrics["any_mode_failure_count"] = sum(
        result["status"] != "success"
        or result.get("warp_only_status") != "success"
        for result in case_results
    )
    for name in warp_high_is_good:
        metrics[name] = float(
            sum(result["warp_only"][name] for result in warp_successful) / attempted
        )
    for name, penalty in warp_penalties.items():
        metrics[name] = float(
            (
                sum(result["warp_only"][name] for result in warp_successful)
                + warp_failures * penalty
            )
            / attempted
        )

    risks = np.asarray(
        [
            result["plane"]["five_anchor_plane_distance_um"]
            if result["status"] == "success" else FAILURE_DISTANCE_UM
            for result in case_results
        ]
    )
    risk_scores = np.asarray([result["risk_score"] for result in case_results])
    order = np.argsort(risk_scores, kind="stable")
    risk_curve = np.cumsum(risks[order]) / np.arange(1, attempted + 1)
    coverage = np.arange(1, attempted + 1) / attempted
    errors = risks > error_threshold_um
    metrics["risk_ranking"] = {
        "error_threshold_um": error_threshold_um,
        "coverage": coverage.tolist(),
        "risk_um": risk_curve.tolist(),
        "aurc_um": float(risk_curve.mean()),
        "error_detection_auroc": _binary_roc_auc(errors, risk_scores),
        "error_detection_auprc": _binary_average_precision(errors, risk_scores),
        "score_contract": (
            "compatibility-derived monotone risk score; ordering only, not a "
            "calibrated probability"
        ),
    }
    wall = np.asarray([result["wall_time_seconds"] for result in case_results])
    memory = np.asarray([result["peak_memory_bytes"] for result in case_results])
    metrics["attested_runtime"] = {
        "wall_time_total_seconds": float(wall.sum()),
        "wall_time_mean_seconds": float(wall.mean()),
        "wall_time_p95_seconds": float(np.quantile(wall, 0.95)),
        "peak_memory_bytes_maximum": int(memory.max()),
        "providers": sorted({result["provider"] for result in case_results}),
    }
    return metrics


def _paired_degradation(
    reference_results: list[dict],
    challenged_results: list[dict],
    strata: list[str],
) -> dict:
    if not (len(reference_results) == len(challenged_results) == len(strata)):
        raise ValueError("reference/challenge pair receipts are incomplete")

    def pose_risk(result):
        return (
            result["plane"]["five_anchor_plane_distance_um"]
            if result["status"] == "success" else FAILURE_DISTANCE_UM
        )

    def correspondence(result):
        return (
            result["spatial"]["visible_region_correspondence"]
            if result["status"] == "success" else 0.0
        )

    def summarize(indices):
        pose_delta = np.asarray(
            [
                pose_risk(challenged_results[item]) - pose_risk(reference_results[item])
                for item in indices
            ]
        )
        correspondence_drop = np.asarray(
            [
                correspondence(reference_results[item])
                - correspondence(challenged_results[item])
                for item in indices
            ]
        )
        return {
            "pair_count": len(indices),
            "challenged_minus_reference_pose_risk_um_mean": float(pose_delta.mean()),
            "reference_minus_challenged_correspondence_mean": float(
                correspondence_drop.mean()
            ),
            "challenged_pose_risk_not_better_rate": float(np.mean(pose_delta >= 0.0)),
            "challenged_correspondence_not_better_rate": float(
                np.mean(correspondence_drop >= 0.0)
            ),
        }

    return {
        "paired_metric_domain": PAIRED_METRIC_DOMAIN,
        "overall": summarize(range(len(strata))),
        "per_stratum": {
            name: summarize([item for item, label in enumerate(strata) if label == name])
            for name in locked.SEVERITIES
        },
    }


def _paired_support_records(batch: dict) -> list[dict[str, float]]:
    tissue = batch["moving_tissue_mask"].flatten(1).sum(1).clamp_min(1)
    visible = batch["moving_visible_mask"].flatten(1).sum(1)
    challenged = visible / tissue
    return [
        {
            "reference_full_tissue_pixels": int(full_pixels),
            "challenged_visible_pixels": int(visible_pixels),
            "reference_full_tissue_support": 1.0,
            "challenged_visible_support": float(value),
            "coverage_loss": float(1.0 - value),
        }
        for full_pixels, visible_pixels, value in zip(tissue, visible, challenged)
    ]


def _paired_support_summary(records: list[dict], strata: list[str]) -> dict:
    if len(records) != len(strata):
        raise ValueError("paired support receipts do not match the cohort")

    def summarize(indices):
        indices = list(indices)
        reference_pixels = sum(
            records[item]["reference_full_tissue_pixels"] for item in indices
        )
        challenged_pixels = sum(
            records[item]["challenged_visible_pixels"] for item in indices
        )
        summary = {
            "case_count": len(indices),
            "reference_full_tissue_pixels": int(reference_pixels),
            "challenged_visible_pixels": int(challenged_pixels),
            "aggregate_coverage_loss": float(
                1.0 - challenged_pixels / max(reference_pixels, 1)
            ),
        }
        summary.update(
            {
                name: float(np.mean([records[item][name] for item in indices]))
                for name in (
                "reference_full_tissue_support",
                "challenged_visible_support",
                "coverage_loss",
                )
            }
        )
        return summary

    return {
        "support_denominator": "reference_full_tissue_pixels",
        "overall": summarize(range(len(records))),
        "per_stratum": {
            name: summarize(
                [item for item, label in enumerate(strata) if label == name]
            )
            for name in locked.SEVERITIES
        },
    }


def _evaluate_case(
    benchmark: locked.LockedJointSyntheticBenchmark,
    batch: dict,
    item: int,
    prediction: dict,
    expected_candidates: dict,
) -> dict:
    if prediction.get("candidate_protocol_violation"):
        raise ValueError("predictor attempted to supply evaluator-owned candidates")
    provider = str(prediction["provider"])
    wall_time = float(prediction["wall_time_seconds"])
    peak_memory = int(prediction["peak_memory_bytes"])
    risk_score = float(prediction["risk_score"])
    if (
        not provider
        or not math.isfinite(wall_time)
        or wall_time < 0.0
        or peak_memory < 0
        or not math.isfinite(risk_score)
        or risk_score < 0.0
    ):
        raise ValueError("provider, timing, memory, and risk-score receipts are invalid")
    base = {
        "case_index": item,
        "provider": provider,
        "wall_time_seconds": wall_time,
        "peak_memory_bytes": peak_memory,
        "risk_score": risk_score,
    }
    if prediction.get("status", "success") == "failed":
        return {**base, "status": "failed", "failure_reason": str(prediction["failure_reason"])}
    if (
        risk_score > 1.0
        or prediction.get("risk_score_contract_sha256")
        != RISK_SCORE_CONTRACT_SHA256
        or prediction.get("risk_score_failure_reason") is not None
    ):
        raise ValueError("evaluator-owned local-neighborhood risk score is invalid")
    if not _all_finite(prediction):
        return {**base, "status": "failed", "failure_reason": "nonfinite output", "nonfinite": True}

    device = batch["pose"].device
    truth_pose = batch["pose"][item]
    initial_pose = _as_tensor(prediction["initial_pose"], device).reshape(3)
    recurrent = _as_tensor(prediction["recurrent_poses"], device).reshape(-1, 3)
    final_pose = _as_tensor(prediction["final_pose"], device).reshape(3)
    accepted_poses = torch.cat((initial_pose[None], recurrent), dim=0)
    if not torch.allclose(accepted_poses[-1], final_pose, rtol=0.0, atol=1e-5):
        raise ValueError("final pose does not match the last accepted recurrent pose")
    iteration_predictions = prediction["iteration_predictions"]
    if len(iteration_predictions) != len(accepted_poses):
        raise ValueError("one map receipt is required for the initial and every recurrent pose")

    iteration_spatial = []
    iteration_plane = []
    for pose, record in zip(accepted_poses, iteration_predictions):
        spatial, _ = _spatial_metrics(benchmark, batch, item, pose[None], record)
        iteration_spatial.append(spatial)
        iteration_plane.append(_pose_plane_errors(pose, truth_pose))
    final_record = {
        "map_pose": prediction["map_pose"],
        "fixed_to_source_model": prediction["fixed_to_source_model"],
        "source_model_to_fixed": prediction["source_model_to_fixed"],
        "map_space": prediction["map_space"],
        "refiner_preprocessing": prediction["refiner_preprocessing"],
        "map_domain_receipt": prediction["map_domain_receipt"],
    }
    final_spatial, final_dice = _spatial_metrics(
        benchmark, batch, item, final_pose[None], final_record
    )
    _validate_map_receipt(
        iteration_predictions[-1], final_pose, locked.MODEL_SHAPE, device
    )
    for name in ("fixed_to_source_model", "source_model_to_fixed"):
        if not torch.equal(
            _as_tensor(iteration_predictions[-1][name], device),
            _as_tensor(final_record[name], device),
        ):
            raise ValueError("final maps differ from the final recurrent map receipt")

    signed_error = final_pose - truth_pose
    plane = _pose_plane_errors(final_pose, truth_pose)
    pose_risk = [value["five_anchor_plane_distance_um"] for value in iteration_plane]
    correspondence = [value["visible_region_correspondence"] for value in iteration_spatial]
    if prediction.get("candidate_set_sha256") != expected_candidates["candidate_set_sha256"]:
        raise ValueError("candidate-set hash is missing or does not match the evaluator order")
    candidate_pose = _as_tensor(prediction["candidate_poses"], device).reshape(-1, 3)
    expected_pose = expected_candidates["candidate_poses"].to(device)
    if candidate_pose.shape != expected_pose.shape or not torch.equal(
        candidate_pose, expected_pose
    ):
        raise ValueError("candidate poses were omitted, changed, or reordered")
    candidate_logits = _as_tensor(prediction["compatibility_logits"], device).reshape(-1)
    if len(candidate_pose) != len(candidate_logits):
        raise ValueError("candidate poses and compatibility logits have different lengths")
    true_index = int(expected_candidates["true_index"])
    true_logit = candidate_logits[true_index]
    competitors = torch.arange(len(candidate_logits), device=device) != true_index
    true_rank = 1 + int((candidate_logits[competitors] >= true_logit).sum())
    result = {
        **base,
        "status": "success",
        "pose_signed_error": {
            "ap": float(signed_error[0]), "lr": float(signed_error[1]),
            "dv": float(signed_error[2]),
        },
        "plane": plane,
        "spatial": final_spatial,
        "region_dice": final_dice,
        "iteration_plane": iteration_plane,
        "iteration_spatial": iteration_spatial,
        "pose_monotonic_transitions": [
            following <= previous + 1e-6
            for previous, following in zip(pose_risk[:-1], pose_risk[1:])
        ],
        "correspondence_monotonic_transitions": [
            following + 1e-8 >= previous
            for previous, following in zip(correspondence[:-1], correspondence[1:])
        ],
        "pose_monotonic": all(
            following <= previous + 1e-6
            for previous, following in zip(pose_risk[:-1], pose_risk[1:])
        ),
        "correspondence_monotonic": all(
            following + 1e-8 >= previous
            for previous, following in zip(correspondence[:-1], correspondence[1:])
        ),
        "true_plane_rank": true_rank,
    }
    if not _all_finite(result):
        raise ValueError("derived evaluation metrics contain nonfinite values")
    return result


def _invalid_case_result(item: int, prediction, error: Exception) -> dict:
    risk_score = prediction.get("risk_score", 1e9) if isinstance(prediction, dict) else 1e9
    risk_score = float(risk_score) if isinstance(risk_score, (int, float)) else 1e9
    if not math.isfinite(risk_score) or risk_score < 0.0:
        risk_score = 1e9
    wall = prediction.get("wall_time_seconds", 0.0) if isinstance(prediction, dict) else 0.0
    memory = prediction.get("peak_memory_bytes", 0) if isinstance(prediction, dict) else 0
    return {
        "case_index": item,
        "status": "failed",
        "failure_reason": f"invalid model output: {type(error).__name__}: {error}",
        "nonfinite": not _all_finite(prediction),
        "risk_score": risk_score,
        "provider": str(prediction.get("provider", "invalid-output"))
        if isinstance(prediction, dict) else "invalid-output",
        "wall_time_seconds": max(0.0, float(wall)) if isinstance(wall, (int, float)) else 0.0,
        "peak_memory_bytes": max(0, int(memory)) if isinstance(memory, (int, float)) else 0,
    }


def _evaluate_cases(
    benchmark: locked.LockedJointSyntheticBenchmark,
    batch: dict,
    predictions: list[dict],
    expected_candidates: list[dict],
    *,
    case_index_offset: int = 0,
) -> list[dict]:
    if len(predictions) != len(batch["pose"]):
        raise ValueError("prediction count must equal the attempted case count")
    results = []
    for item, (prediction, candidates) in enumerate(zip(predictions, expected_candidates)):
        try:
            warp_only = _exact_plane_metrics(
                benchmark, batch, item, prediction, batch["pose"].device
            )
            warp_status = "success"
            warp_failure_reason = None
        except (KeyError, TypeError, ValueError, RuntimeError, IndexError) as error:
            warp_only = None
            warp_status = "failed"
            warp_failure_reason = f"{type(error).__name__}: {error}"
        try:
            result = _evaluate_case(benchmark, batch, item, prediction, candidates)
        except (KeyError, TypeError, ValueError, RuntimeError, IndexError) as error:
            result = _invalid_case_result(item, prediction, error)
        result["case_index"] = case_index_offset + item
        result["warp_only_status"] = warp_status
        if warp_only is not None:
            result["warp_only"] = warp_only
        else:
            result["warp_only_failure_reason"] = warp_failure_reason
        results.append(result)
    return results


def _evaluate_and_save(
    benchmark: locked.LockedJointSyntheticBenchmark,
    batch: dict,
    predictions: list[dict],
    output_dir: str | Path,
    *,
    error_threshold_um: float,
    allow_sealed: bool,
    expected_candidates: list[dict] | None = None,
    measured_runtime: dict | None = None,
    strata: list[str] | None = None,
) -> dict:
    count = len(batch["pose"])
    if len(predictions) != count:
        raise ValueError("prediction count must equal the attempted case count")
    split = str(batch["split"])
    if split == "sealed-test" and not allow_sealed:
        raise PermissionError("locked-test predictions require the local evaluator")
    if split not in locked.ALL_SPLITS:
        raise ValueError("batch has an unknown locked split")
    if expected_candidates is None:
        _, expected_candidates = build_predictor_payload(benchmark, batch)
    if len(expected_candidates) != count:
        raise ValueError("candidate contract count does not match attempted cases")
    case_results = _evaluate_cases(
        benchmark, batch, predictions, expected_candidates
    )
    aggregate = _aggregate(case_results, error_threshold_um)
    aggregate.update(
        split=split,
        severity=str(batch["severity"]),
        manifest_sha256=str(batch["manifest_sha256"]),
        benchmark_contract_sha256=str(batch["contract"]["contract_sha256"]),
        evaluator_source_sha256=evaluator_source_sha256(),
        risk_score_contract_sha256=RISK_SCORE_CONTRACT_SHA256,
        evaluator_dependency_tree=evaluator_dependency_tree(),
        evaluator_dependency_tree_sha256=evaluator_dependency_tree_sha256(),
    )
    if measured_runtime is not None:
        aggregate["measured_inference_runtime"] = dict(measured_runtime)
    if strata is not None:
        if len(strata) != count:
            raise ValueError("stratum labels do not match attempted cases")
        aggregate["per_stratum"] = {
            name: _aggregate(
                [result for result, label in zip(case_results, strata) if label == name],
                error_threshold_um,
            )
            for name in locked.SEVERITIES
        }
    destination = Path(output_dir)
    raw_path = destination / "raw_predictions.pt"
    metrics_path = destination / "aggregate_metrics.json"
    receipt_path = destination / "evaluation_receipt.json"
    _atomic_torch_save(raw_path, {"predictions": predictions, "case_results": case_results})
    aggregate["raw_predictions_sha256"] = _file_sha256(raw_path)
    _atomic_json(metrics_path, aggregate)
    receipt = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "split": split,
        "attempted_count": count,
        "manifest_sha256": str(batch["manifest_sha256"]),
        "raw_predictions_sha256": aggregate["raw_predictions_sha256"],
        "aggregate_metrics_sha256": _file_sha256(metrics_path),
        "evaluator_source_sha256": evaluator_source_sha256(),
        "risk_score_contract_sha256": RISK_SCORE_CONTRACT_SHA256,
        "evaluator_dependency_tree_sha256": evaluator_dependency_tree_sha256(),
    }
    _atomic_json(receipt_path, receipt)
    aggregate["evaluation_receipt_sha256"] = _file_sha256(receipt_path)
    return aggregate


def evaluate_public_split(
    benchmark: locked.LockedJointSyntheticBenchmark,
    batch: dict,
    predictor: Callable[[dict], list[dict]],
    output_dir: str | Path,
    *,
    error_threshold_um: float = DEFAULT_ERROR_THRESHOLD_UM,
) -> dict:
    if str(batch["split"]) not in locked.PUBLIC_SPLITS:
        raise PermissionError("the public evaluator accepts development or locked-validation only")
    if not callable(predictor):
        raise TypeError("public evaluation requires a predictor callable")
    payload, expected_candidates = build_predictor_payload(benchmark, batch)
    try:
        predictions, measured_runtime = _run_bound_predictor(
            benchmark, predictor, payload, expected_candidates
        )
    except Exception as error:
        predictions = [
            {
                "status": "failed",
                "failure_reason": f"predictor failure: {type(error).__name__}: {error}",
                "risk_score": 1e9,
                "provider": "failed-before-provider-receipt",
                "wall_time_seconds": 0.0,
                "peak_memory_bytes": 0,
            }
            for _ in expected_candidates
        ]
        measured_runtime = {"wall_time_seconds": 0.0, "python_peak_memory_bytes": 0}
    return _evaluate_and_save(
        benchmark,
        batch,
        predictions,
        output_dir,
        error_threshold_um=error_threshold_um,
        allow_sealed=False,
        expected_candidates=expected_candidates,
        measured_runtime=measured_runtime,
    )


def validate_frozen_locked_receipt(
    benchmark: locked.LockedJointSyntheticBenchmark, receipt: dict
) -> None:
    if receipt.get("frozen") is not True:
        raise ValueError(
            "locally locked evaluation requires an explicitly frozen candidate"
        )
    checkpoint_path = Path(receipt["checkpoint_path"])
    if checkpoint_path.name != "best-validation.pt":
        raise ValueError(
            "locally locked qualification requires the canonical best-validation.pt"
        )
    if receipt.get("checkpoint_sha256") != _file_sha256(checkpoint_path):
        raise ValueError("frozen checkpoint hash mismatch")
    _, release_state_receipt = load_joint_release_state(checkpoint_path, "cpu")
    if receipt.get("release_state_receipt") != release_state_receipt:
        raise ValueError("frozen candidate is not bound to the canonical EMA release state")
    bundle = receipt.get("inference_bundle")
    if not isinstance(bundle, dict):
        raise ValueError(
            "locally locked evaluation requires the complete frozen inference bundle"
        )
    files = bundle.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("frozen inference bundle has no executable files")
    roles = set()
    for file_receipt in files:
        if not isinstance(file_receipt, dict):
            raise ValueError("invalid inference bundle file receipt")
        role = str(file_receipt.get("role", ""))
        path = Path(file_receipt.get("path", ""))
        if not role or role in roles or file_receipt.get("sha256") != _file_sha256(path):
            raise ValueError("frozen inference bundle file hash/role mismatch")
        roles.add(role)
    if "checkpoint" not in roles or "metadata" not in roles:
        raise ValueError("frozen bundle must include checkpoint and metadata")
    if sum(value == "predictor_source" for value in roles) != 1:
        raise ValueError(
            "locally locked qualification requires exactly one hash-frozen "
            "predictor source adapter"
        )
    if bundle.get("preprocessing_contract") != locked.PREPROCESSING_CONTRACT_V2:
        raise ValueError("frozen bundle preprocessing contract mismatch")
    if bundle.get("mask_contract_sha256") != locked.MASK_CONTRACT_SHA256:
        raise ValueError("frozen bundle mask contract mismatch")
    if (
        bundle.get("pose_preprocessing_contract_sha256")
        != atlas_pose_preprocessing_contract_sha256()
    ):
        raise ValueError("frozen bundle pose preprocessing contract mismatch")
    if bundle.get("risk_score_contract_sha256") != RISK_SCORE_CONTRACT_SHA256:
        raise ValueError("frozen bundle risk-score lattice contract mismatch")
    if (
        bundle.get("evaluator_dependency_tree_sha256")
        != evaluator_dependency_tree_sha256()
    ):
        raise ValueError("frozen bundle evaluator dependency tree mismatch")
    if not isinstance(bundle.get("recurrence_count"), int) or bundle["recurrence_count"] < 0:
        raise ValueError("frozen bundle recurrence count is invalid")
    if not isinstance(bundle.get("configuration"), dict):
        raise ValueError("frozen bundle configuration is missing")
    providers = bundle.get("provider_policy")
    if not isinstance(providers, list) or not providers or not all(
        isinstance(value, str) and value for value in providers
    ):
        raise ValueError("frozen bundle provider policy is invalid")
    if receipt.get("inference_bundle_sha256") != locked._payload_sha256(bundle):
        raise ValueError("frozen inference bundle receipt hash mismatch")
    if bundle.get("release_state_receipt") != release_state_receipt:
        raise ValueError("frozen inference bundle does not identify the EMA release state")
    metadata_files = [value for value in files if value["role"] == "metadata"]
    if len(metadata_files) != 1:
        raise ValueError("frozen bundle requires exactly one metadata file")
    with Path(metadata_files[0]["path"]).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("release_state_receipt") != release_state_receipt:
        raise ValueError("frozen metadata does not identify the EMA release state")
    checkpoint_files = [value for value in files if value["role"] == "checkpoint"]
    if (
        len(checkpoint_files) != 1
        or Path(checkpoint_files[0]["path"]).resolve() != checkpoint_path.resolve()
        or checkpoint_files[0]["sha256"] != receipt["checkpoint_sha256"]
    ):
        raise ValueError("checkpoint is not bound into the frozen inference bundle")
    if receipt.get("evaluator_source_sha256") != evaluator_source_sha256():
        raise ValueError("frozen evaluator source hash mismatch")
    if (
        receipt.get("benchmark_contract_sha256")
        != benchmark.contract["contract_sha256"]
    ):
        raise ValueError("frozen locked-generator contract hash mismatch")
    if receipt.get("generator_source_sha256") != locked._source_sha256():
        raise ValueError("frozen locked-generator source hash mismatch")


def _load_frozen_predictor(frozen_receipt: dict) -> Callable[[dict], object]:
    bundle = frozen_receipt["inference_bundle"]
    source_receipts = [
        value for value in bundle["files"] if value["role"] == "predictor_source"
    ]
    if len(source_receipts) != 1:
        raise ValueError(
            "locally locked qualification requires one hash-frozen predictor source adapter"
        )
    source = source_receipts[0]
    if source["sha256"] != _file_sha256(source["path"]):
        raise ValueError("frozen predictor source changed before execution")
    module_name = f"joint_locked_predictor_{source['sha256']}"
    specification = importlib.util.spec_from_file_location(module_name, source["path"])
    if specification is None or specification.loader is None:
        raise ValueError("frozen predictor source cannot be imported")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    factory = getattr(module, "create_frozen_predictor", None)
    if not callable(factory):
        raise ValueError("frozen predictor source must define create_frozen_predictor")
    predictor = factory(bundle)
    if not callable(predictor):
        raise ValueError("frozen predictor factory did not return a callable")
    return predictor


def run_locally_locked_once(
    benchmark: locked.LockedJointSyntheticBenchmark,
    output_dir: str | Path,
    frozen_receipt: dict,
    *,
    error_threshold_um: float = DEFAULT_ERROR_THRESHOLD_UM,
    _capability=None,
) -> dict:
    """Run a reproducible local qualification, not a cryptographic sealed benchmark.

    Publication-grade secrecy requires an external custodian/service and a signed
    hidden cohort; local Python cannot defend against a hostile machine owner.
    """
    if _capability is not _LOCAL_LOCKED_RUN_CAPABILITY:
        raise PermissionError("locally locked evaluation requires its private capability")
    validate_frozen_locked_receipt(benchmark, frozen_receipt)
    predictor = _load_frozen_predictor(frozen_receipt)
    exact_predictor = _load_frozen_predictor(frozen_receipt)
    if exact_predictor is predictor:
        raise ValueError(
            "frozen predictor factory must create distinct truth-free and exact-plane instances"
        )
    destination = Path(output_dir)
    hidden_seed = secrets.randbits(63)
    commitment_material = (
        hidden_seed.to_bytes(8, "big")
        + bytes.fromhex(frozen_receipt["inference_bundle_sha256"])
        + bytes.fromhex(evaluator_source_sha256())
        + bytes.fromhex(locked._source_sha256())
        + bytes.fromhex(RISK_SCORE_CONTRACT_SHA256)
        + bytes.fromhex(evaluator_dependency_tree_sha256())
    )
    seed_commitment = hashlib.sha256(commitment_material).hexdigest()
    _exclusive_json(
        destination / "LOCALLY_LOCKED_CLAIM.json",
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "locally-locked-claimed",
            "security_scope": "reproducible local qualification; not cryptographically sealed",
            "case_count": SEALED_TOTAL_CASES,
            "inference_view_count": 2 * SEALED_TOTAL_CASES,
            "cases_per_stratum": SEALED_CASES_PER_STRATUM,
            "negative_candidates_per_case": SEALED_NEGATIVES_PER_CASE,
            "seed_commitment_sha256": seed_commitment,
            "checkpoint_sha256": frozen_receipt["checkpoint_sha256"],
            "release_state_receipt": frozen_receipt["release_state_receipt"],
            "inference_bundle_sha256": frozen_receipt["inference_bundle_sha256"],
            "evaluator_source_sha256": evaluator_source_sha256(),
            "generator_source_sha256": locked._source_sha256(),
            "benchmark_contract_sha256": benchmark.contract["contract_sha256"],
            "risk_score_contract_sha256": RISK_SCORE_CONTRACT_SHA256,
            "evaluator_dependency_tree": evaluator_dependency_tree(),
            "evaluator_dependency_tree_sha256": evaluator_dependency_tree_sha256(),
        },
    )
    manifests = benchmark.make_balanced_sealed_manifests_once(
        hidden_seed,
        SEALED_CASES_PER_STRATUM,
        SEALED_NEGATIVES_PER_CASE,
        _capability=locked._SEALED_EVALUATOR_CAPABILITY,
    )
    cohort_manifest_sha256 = locked._payload_sha256(
        {name: value["manifest_sha256"] for name, value in manifests.items()}
    )
    raw_directory = destination / "raw_prediction_chunks"
    case_results, reference_case_results, raw_chunks, strata = [], [], [], []
    paired_support_records = []
    measured_wall = 0.0
    measured_peak = 0
    global_index = 0
    for severity, manifest in manifests.items():
        for start in range(0, SEALED_CASES_PER_STRATUM, 16):
            stop = min(start + 16, SEALED_CASES_PER_STRATUM)
            batch = benchmark._batch(_manifest_slice(manifest, start, stop), qa=False)
            shuffle_secret = hashlib.sha256(
                hidden_seed.to_bytes(8, "big")
                + severity.encode("ascii")
                + start.to_bytes(4, "big")
            ).digest()
            payload, expected = build_predictor_payload(
                benchmark,
                batch,
                shuffle_secret=shuffle_secret,
                view="challenged",
            )
            reference_payload, reference_expected = build_predictor_payload(
                benchmark,
                batch,
                shuffle_secret=shuffle_secret,
                view="reference",
            )
            for challenged, reference in zip(expected, reference_expected):
                if (
                    challenged["candidate_set_sha256"]
                    != reference["candidate_set_sha256"]
                    or not torch.equal(
                        challenged["candidate_poses"], reference["candidate_poses"]
                    )
                    or challenged["true_index"] != reference["true_index"]
                ):
                    raise ValueError(
                        "reference/challenge candidates are not geometry-paired"
                    )
            try:
                predictions, measured = _run_bound_predictor(
                    benchmark,
                    predictor,
                    payload,
                    expected,
                    exact_predictor=exact_predictor,
                )
            except Exception as error:
                predictions = [
                    {
                        "status": "failed",
                        "failure_reason": f"predictor failure: {type(error).__name__}: {error}",
                        "risk_score": 1e9,
                        "provider": "failed-before-provider-receipt",
                        "wall_time_seconds": 0.0,
                        "peak_memory_bytes": 0,
                    }
                    for _ in range(stop - start)
                ]
                measured = {"wall_time_seconds": 0.0, "python_peak_memory_bytes": 0}
            try:
                reference_predictions, reference_measured = _run_bound_predictor(
                    benchmark,
                    predictor,
                    reference_payload,
                    reference_expected,
                    exact_predictor=exact_predictor,
                )
            except Exception as error:
                reference_predictions = [
                    {
                        "status": "failed",
                        "failure_reason": (
                            f"predictor failure: {type(error).__name__}: {error}"
                        ),
                        "risk_score": 1e9,
                        "provider": "failed-before-provider-receipt",
                        "wall_time_seconds": 0.0,
                        "peak_memory_bytes": 0,
                    }
                    for _ in range(stop - start)
                ]
                reference_measured = {
                    "wall_time_seconds": 0.0,
                    "python_peak_memory_bytes": 0,
                }
            results = _evaluate_cases(
                benchmark,
                batch,
                predictions,
                expected,
                case_index_offset=global_index,
            )
            reference_results = _evaluate_cases(
                benchmark,
                batch,
                reference_predictions,
                reference_expected,
                case_index_offset=global_index,
            )
            chunk_path = raw_directory / f"{severity}-{start:04d}-{stop:04d}.pt"
            _atomic_torch_save(
                chunk_path,
                {
                    "case_sha256": tuple(str(value) for value in batch["case_sha256"]),
                    "pair_id": tuple(str(value) for value in batch["pair_id"]),
                    "candidate_set_sha256": tuple(
                        value["candidate_set_sha256"] for value in expected
                    ),
                    "challenged_predictions": predictions,
                    "challenged_case_results": results,
                    "reference_predictions": reference_predictions,
                    "reference_case_results": reference_results,
                    "paired_metric_domain": PAIRED_METRIC_DOMAIN,
                    "paired_support": _paired_support_records(batch),
                },
            )
            raw_chunks.append(
                {
                    "path": str(chunk_path.relative_to(destination)),
                    "sha256": _file_sha256(chunk_path),
                    "severity": severity,
                    "start": start,
                    "stop": stop,
                }
            )
            case_results.extend(results)
            reference_case_results.extend(reference_results)
            paired_support_records.extend(_paired_support_records(batch))
            strata.extend([severity] * (stop - start))
            global_index += stop - start
            measured_wall += (
                measured["wall_time_seconds"]
                + reference_measured["wall_time_seconds"]
            )
            measured_peak = max(
                measured_peak,
                measured["python_peak_memory_bytes"],
                reference_measured["python_peak_memory_bytes"],
            )
    raw_manifest_path = destination / "raw_predictions_manifest.json"
    _atomic_json(
        raw_manifest_path,
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "cohort_manifest_sha256": cohort_manifest_sha256,
            "attempted_count": SEALED_TOTAL_CASES,
            "inference_view_count": 2 * SEALED_TOTAL_CASES,
            "pair_contract": (
                "same-case geometry; clean pre-appearance reference plus "
                "severity-challenged view"
            ),
            "paired_metric_domain": PAIRED_METRIC_DOMAIN,
            "chunks": raw_chunks,
        },
    )
    raw_predictions_sha256 = _file_sha256(raw_manifest_path)
    aggregate = _aggregate(case_results, error_threshold_um)
    aggregate.update(
        split="sealed-test",
        severity="balanced",
        per_stratum={
            severity: _aggregate(
                [result for result, label in zip(case_results, strata) if label == severity],
                error_threshold_um,
            )
            for severity in locked.SEVERITIES
        },
        paired_metric_domain=PAIRED_METRIC_DOMAIN,
        reference_common_support_view=_aggregate(
            reference_case_results,
            error_threshold_um,
        ),
        paired_degradation=_paired_degradation(
            reference_case_results, case_results, strata
        ),
        paired_support=_paired_support_summary(paired_support_records, strata),
        inference_view_count=2 * SEALED_TOTAL_CASES,
        cohort_manifest_sha256=cohort_manifest_sha256,
        benchmark_contract_sha256=benchmark.contract["contract_sha256"],
        release_state_receipt=frozen_receipt["release_state_receipt"],
        evaluator_source_sha256=evaluator_source_sha256(),
        risk_score_contract_sha256=RISK_SCORE_CONTRACT_SHA256,
        evaluator_dependency_tree=evaluator_dependency_tree(),
        evaluator_dependency_tree_sha256=evaluator_dependency_tree_sha256(),
        generator_source_sha256=locked._source_sha256(),
        raw_predictions_sha256=raw_predictions_sha256,
        measured_inference_runtime={
            "wall_time_seconds": measured_wall,
            "python_peak_memory_bytes_maximum": measured_peak,
        },
    )
    metrics_path = destination / "aggregate_metrics.json"
    _atomic_json(metrics_path, aggregate)
    evaluation_receipt = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "split": "sealed-test",
        "attempted_count": SEALED_TOTAL_CASES,
        "inference_view_count": 2 * SEALED_TOTAL_CASES,
        "cohort_manifest_sha256": cohort_manifest_sha256,
        "raw_predictions_sha256": raw_predictions_sha256,
        "aggregate_metrics_sha256": _file_sha256(metrics_path),
        "checkpoint_sha256": frozen_receipt["checkpoint_sha256"],
        "release_state_receipt": frozen_receipt["release_state_receipt"],
        "inference_bundle_sha256": frozen_receipt["inference_bundle_sha256"],
        "evaluator_source_sha256": evaluator_source_sha256(),
        "generator_source_sha256": locked._source_sha256(),
        "risk_score_contract_sha256": RISK_SCORE_CONTRACT_SHA256,
        "paired_metric_domain": PAIRED_METRIC_DOMAIN,
        "evaluator_dependency_tree_sha256": evaluator_dependency_tree_sha256(),
    }
    evaluation_receipt_path = destination / "evaluation_receipt.json"
    _atomic_json(evaluation_receipt_path, evaluation_receipt)
    aggregate["evaluation_receipt_sha256"] = _file_sha256(evaluation_receipt_path)
    consumption_receipt = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "consumed",
        "hidden_seed": hidden_seed,
        "seed_commitment_sha256": seed_commitment,
        "checkpoint_sha256": frozen_receipt["checkpoint_sha256"],
        "release_state_receipt": frozen_receipt["release_state_receipt"],
        "inference_bundle_sha256": frozen_receipt["inference_bundle_sha256"],
        "evaluator_source_sha256": evaluator_source_sha256(),
        "generator_source_sha256": locked._source_sha256(),
        "risk_score_contract_sha256": RISK_SCORE_CONTRACT_SHA256,
        "paired_metric_domain": PAIRED_METRIC_DOMAIN,
        "evaluator_dependency_tree": evaluator_dependency_tree(),
        "evaluator_dependency_tree_sha256": evaluator_dependency_tree_sha256(),
        "benchmark_contract_sha256": benchmark.contract["contract_sha256"],
        "cohort_manifest_sha256": cohort_manifest_sha256,
        "raw_predictions_sha256": raw_predictions_sha256,
        "evaluation_receipt_sha256": aggregate["evaluation_receipt_sha256"],
        "attempted_count": aggregate["attempted_count"],
        "inference_view_count": aggregate["inference_view_count"],
        "failure_count": aggregate["failure_count"],
    }
    consumption_receipt.update(
        status="locally-locked-consumed",
        security_scope=(
            "reproducible local qualification only; publication-grade sealed claims "
            "require an external custodian/service and signed hidden cohort"
        ),
    )
    _exclusive_json(
        destination / "LOCALLY_LOCKED_CONSUMPTION_RECEIPT.json", consumption_receipt
    )
    return {
        **aggregate,
        "locally_locked": True,
        "security_scope": consumption_receipt["security_scope"],
        "seed_commitment_sha256": seed_commitment,
    }
