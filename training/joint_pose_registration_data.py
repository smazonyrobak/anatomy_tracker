"""Deterministic joint pose-and-registration samples built on the frozen v2 generator."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from source.atlas_pose_runtime import (
    atlas_pose_preprocessing_contract_sha256,
    preprocess_atlas_pose_image,
)
from training.atlas_pose_models_v7 import TILT_MAX_DEG, TILT_MIN_DEG
from training.synthetic_registration import (
    BREGMA_AP_INDEX,
    VOXEL_UM,
    SyntheticRegistrationGenerator,
    _payload_sha256,
    split_ap_indices,
)


JOINT_MANIFEST_VERSION = 1
AP_OFFSET_LEVELS_UM = np.asarray((25.0, 50.0, 100.0, 250.0, 500.0, 1000.0), np.float32)
TILT_OFFSET_LEVELS_DEG = np.asarray((0.25, 0.5, 1.0, 2.0, 5.0, 10.0), np.float32)
STRATUM_INDEX = {"clean": 0, "mild": 1, "hard": 2}
_REGISTRATION_PREFIX = "registration__"


def _source_sha256() -> str:
    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(source).hexdigest()


def _rng(seed: int, stream: str) -> np.random.Generator:
    digest = hashlib.sha256(
        f"joint-pose-registration-v1:{int(seed)}:{stream}".encode()
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def _stratified_uniform(
    rng: np.random.Generator, count: int, low: float, high: float
) -> np.ndarray:
    values = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(values)
    return (low + (high - low) * values).astype(np.float32)


def _wrapped_rotation(rotation_deg: np.ndarray) -> np.ndarray:
    return ((np.asarray(rotation_deg, dtype=np.float32) + 180.0) % 360.0 - 180.0).astype(
        np.float32
    )


def _pack_registration_manifest(manifest: dict) -> dict:
    return {_REGISTRATION_PREFIX + key: value for key, value in manifest.items()}


def registration_manifest(manifest: dict) -> dict:
    return {
        key[len(_REGISTRATION_PREFIX) :]: value
        for key, value in manifest.items()
        if key.startswith(_REGISTRATION_PREFIX)
    }


def _pose_from_registration(manifest: dict) -> np.ndarray:
    return np.column_stack(
        (manifest["ap_um"], manifest["tilt_lr_deg"], manifest["tilt_dv_deg"])
    ).astype(np.float32)


def _ap_index(ap_um: float) -> int:
    return int(round(BREGMA_AP_INDEX - float(ap_um) / VOXEL_UM))


def _valid_ap_offsets(ap_um: float, split_pool: set[int]) -> list[float]:
    return [
        sign * float(level)
        for level in AP_OFFSET_LEVELS_UM
        for sign in (-1.0, 1.0)
        if _ap_index(ap_um + sign * float(level)) in split_pool
    ]


def _ensure_adjacent_ap_centers(
    manifest: dict,
    split: str,
    rng: np.random.Generator,
    *,
    _final_capability=None,
) -> dict:
    pool = np.asarray(
        split_ap_indices(split, _final_capability=_final_capability), dtype=np.int32
    )
    pool_set = set(int(value) for value in pool)
    eligible = np.asarray(
        [value for value in pool if int(value) - 1 in pool_set or int(value) + 1 in pool_set],
        dtype=np.int32,
    )
    ap_index = np.asarray(manifest["ap_index"], dtype=np.float32).copy()
    for sample, value in enumerate(ap_index.astype(np.int32)):
        if int(value) - 1 in pool_set or int(value) + 1 in pool_set:
            continue
        distance = np.abs(eligible - int(value))
        nearest = eligible[distance == distance.min()]
        ap_index[sample] = rng.choice(nearest)
    if np.array_equal(ap_index, manifest["ap_index"]):
        return manifest
    adjusted = dict(manifest)
    adjusted["ap_index"] = ap_index
    adjusted["ap_um"] = ((BREGMA_AP_INDEX - ap_index) * VOXEL_UM).astype(np.float32)
    adjusted["manifest_sha256"] = _payload_sha256(
        {key: value for key, value in adjusted.items() if key != "manifest_sha256"}
    )
    return adjusted


def _candidate_offsets(
    true_pose: np.ndarray,
    split: str,
    count: int,
    rng: np.random.Generator,
    *,
    _final_capability=None,
) -> tuple[np.ndarray, np.ndarray]:
    split_pool = set(
        int(value)
        for value in split_ap_indices(split, _final_capability=_final_capability)
    )
    initial = np.empty_like(true_pose, dtype=np.float32)
    candidates = np.zeros((len(true_pose), count, 3), dtype=np.float32)
    signed_tilt = [
        sign * float(level)
        for level in TILT_OFFSET_LEVELS_DEG
        for sign in (-1.0, 1.0)
    ]
    for sample, pose in enumerate(true_pose):
        ap_offsets = _valid_ap_offsets(float(pose[0]), split_pool)
        if not ap_offsets:
            raise RuntimeError("No AP perturbation remains inside the sample's declared split")
        valid_lr = [
            value for value in signed_tilt
            if TILT_MIN_DEG <= pose[1] + value <= TILT_MAX_DEG
        ]
        valid_dv = [
            value for value in signed_tilt
            if TILT_MIN_DEG <= pose[2] + value <= TILT_MAX_DEG
        ]
        initial[sample] = (
            rng.choice(ap_offsets),
            rng.choice(valid_lr),
            rng.choice(valid_dv),
        )
        adjacent = [value for value in ap_offsets if abs(value) == VOXEL_UM]
        if not adjacent:
            raise RuntimeError("Every sample requires a split-safe adjacent AP plane")
        ap_pool = [(value, 0.0, 0.0) for value in ap_offsets]
        lr_pool = [(0.0, value, 0.0) for value in valid_lr]
        dv_pool = [(0.0, 0.0, value) for value in valid_dv]
        required = [(float(rng.choice(adjacent)), 0.0, 0.0)]
        if count >= 2:
            adjacent_lr = [value for value in valid_lr if abs(value) == TILT_OFFSET_LEVELS_DEG[0]]
            required.append((0.0, float(rng.choice(adjacent_lr)), 0.0))
        if count >= 3:
            adjacent_dv = [value for value in valid_dv if abs(value) == TILT_OFFSET_LEVELS_DEG[0]]
            required.append((0.0, 0.0, float(rng.choice(adjacent_dv))))
        pool = [value for value in ap_pool + lr_pool + dv_pool if value not in required]
        if count - len(required) > len(pool):
            raise ValueError(
                f"negatives_per_sample={count} exceeds the distinct split-safe offsets"
            )
        selected = rng.choice(len(pool), size=count - len(required), replace=False)
        candidates[sample] = np.asarray(
            required + [pool[int(index)] for index in selected], np.float32
        )
    return initial, candidates


def _expanded_similarity(
    image: np.ndarray,
    mask: np.ndarray,
    rotation_deg: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    center = ((width - 1.0) / 2.0, (height - 1.0) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(rotation_deg), float(scale))
    corners = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (width - 1.0, 0.0, 1.0),
            (0.0, height - 1.0, 1.0),
            (width - 1.0, height - 1.0, 1.0),
        )
    )
    transformed = (matrix @ corners.T).T
    low = transformed.min(axis=0)
    high = transformed.max(axis=0)
    matrix[:, 2] -= low
    size = tuple(np.ceil(high - low + 1.0).astype(int))
    return (
        cv2.warpAffine(
            image,
            matrix,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ),
        cv2.warpAffine(
            mask.astype(np.uint8),
            matrix,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(bool),
    )


def make_joint_manifest(
    generator: SyntheticRegistrationGenerator,
    count: int,
    split: str,
    seed: int,
    stratum: str,
    negatives_per_sample: int = 6,
    *,
    _final_capability=None,
) -> dict:
    """Create one immutable batch manifest with true-plane and wrong-plane poses."""
    if count < 1:
        raise ValueError("count must be positive")
    if negatives_per_sample < 1:
        raise ValueError("negatives_per_sample must be positive")
    if stratum not in STRATUM_INDEX:
        raise ValueError(f"stratum must be one of {tuple(STRATUM_INDEX)}")
    base = generator.make_manifest(
        count,
        split,
        seed,
        stratum,
        _final_capability=_final_capability,
    )
    base = _ensure_adjacent_ap_centers(
        base,
        split,
        _rng(seed, "adjacent-ap-centers"),
        _final_capability=_final_capability,
    )
    true_pose = _pose_from_registration(base)
    initial_offset, wrong_offset = _candidate_offsets(
        true_pose,
        split,
        negatives_per_sample,
        _rng(seed, "pose-offsets"),
        _final_capability=_final_capability,
    )
    pose_view_rotation = _stratified_uniform(
        _rng(seed, "pose-view-rotation"), count, -180.0, 180.0
    )
    pose_view_scale = _stratified_uniform(
        _rng(seed, "pose-view-scale"), count, 0.5, 1.5
    )
    pose_view_total_rotation = _wrapped_rotation(
        np.asarray(base["rotation_deg"], dtype=np.float32) + pose_view_rotation
    )
    manifest = {
        "joint_manifest_version": JOINT_MANIFEST_VERSION,
        "joint_generator_source_sha256": _source_sha256(),
        "pose_preprocessing_contract_sha256": atlas_pose_preprocessing_contract_sha256(),
        "split": split,
        "seed": int(seed),
        "artifact_stratum": stratum,
        "sample_count": int(count),
        "negatives_per_sample": int(negatives_per_sample),
        "registration_manifest_sha256": base["manifest_sha256"],
        "registration_contract_sha256": generator.contract["contract_sha256"],
        "registration_generator_source_sha256": generator.contract["generator_source_sha256"],
        "average_template_sha256": generator.contract["average_template_sha256"],
        "annotation_sha256": generator.contract["annotation_sha256"],
        "hard_negative_ap_levels_um": AP_OFFSET_LEVELS_UM.copy(),
        "hard_negative_tilt_levels_deg": TILT_OFFSET_LEVELS_DEG.copy(),
        "true_pose": true_pose,
        "initial_pose_offset": initial_offset,
        "wrong_candidate_offset": wrong_offset,
        "pose_view_rotation_deg": pose_view_rotation,
        "pose_view_total_rotation_deg": pose_view_total_rotation,
        "pose_view_scale": pose_view_scale,
        "wrong_candidate_dense_target_valid": np.zeros(
            (count, negatives_per_sample), dtype=np.bool_
        ),
        **_pack_registration_manifest(base),
    }
    manifest["joint_manifest_sha256"] = _payload_sha256(manifest)
    return manifest


def _verified(manifest: dict) -> dict:
    expected = manifest.get("joint_manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "joint_manifest_sha256"}
    if expected != _payload_sha256(payload):
        raise ValueError("joint manifest hash does not match its parameters")
    if manifest.get("joint_manifest_version") != JOINT_MANIFEST_VERSION:
        raise ValueError("joint manifest version is unsupported")
    if manifest.get("joint_generator_source_sha256") != _source_sha256():
        raise ValueError("joint data source differs from the manifest provenance")
    if (
        manifest.get("pose_preprocessing_contract_sha256")
        != atlas_pose_preprocessing_contract_sha256()
    ):
        raise ValueError("AtlasPose preprocessing differs from the manifest provenance")
    base = registration_manifest(manifest)
    if manifest.get("registration_manifest_sha256") != base.get("manifest_sha256"):
        raise ValueError("embedded registration manifest provenance is inconsistent")
    if (
        manifest.get("split") != base.get("split")
        or manifest.get("seed") != base.get("seed")
        or manifest.get("artifact_stratum") != base.get("stratum")
    ):
        raise ValueError("joint and registration manifest identities differ")
    true_pose = np.asarray(manifest.get("true_pose"), dtype=np.float32)
    count = int(manifest.get("sample_count", -1))
    candidates = int(manifest.get("negatives_per_sample", -1))
    if true_pose.shape != (count, 3) or not np.array_equal(
        true_pose, _pose_from_registration(base)
    ):
        raise ValueError("joint true poses differ from the registration manifest")
    if np.asarray(manifest.get("initial_pose_offset")).shape != (count, 3):
        raise ValueError("joint initial-pose offsets have the wrong shape")
    if np.asarray(manifest.get("wrong_candidate_offset")).shape != (count, candidates, 3):
        raise ValueError("joint wrong-candidate offsets have the wrong shape")
    rotation = np.asarray(manifest.get("pose_view_rotation_deg"))
    total_rotation = np.asarray(manifest.get("pose_view_total_rotation_deg"))
    scale = np.asarray(manifest.get("pose_view_scale"))
    if rotation.shape != (count,) or np.any((rotation < -180.0) | (rotation > 180.0)):
        raise ValueError("raw pose-view rotations have the wrong shape or range")
    if scale.shape != (count,) or np.any((scale < 0.5) | (scale > 1.5)):
        raise ValueError("raw pose-view scales have the wrong shape or range")
    expected_total_rotation = _wrapped_rotation(base["rotation_deg"] + rotation)
    if total_rotation.shape != (count,) or not np.array_equal(
        total_rotation, expected_total_rotation
    ):
        raise ValueError("total pose-view rotations differ from their component rotations")
    if np.asarray(manifest.get("wrong_candidate_dense_target_valid")).any():
        raise ValueError("wrong atlas planes cannot carry dense-flow supervision")
    return base


def save_joint_manifest(manifest: dict, path: str | Path) -> Path:
    _verified(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **{key: np.asarray(value) for key, value in manifest.items()})
    return destination


def load_joint_manifest(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as archive:
        manifest = {
            key: (value.item() if value.ndim == 0 else np.array(value, copy=True))
            for key, value in archive.items()
        }
    _verified(manifest)
    return manifest


class JointSyntheticData:
    """Materialize a joint manifest without changing either frozen generator."""

    def __init__(self, generator: SyntheticRegistrationGenerator):
        self.generator = generator

    def make_manifest(
        self,
        count: int,
        split: str,
        seed: int,
        stratum: str,
        negatives_per_sample: int = 6,
        *,
        _final_capability=None,
    ) -> dict:
        return make_joint_manifest(
            self.generator,
            count,
            split,
            seed,
            stratum,
            negatives_per_sample,
            _final_capability=_final_capability,
        )

    def batch(self, manifest: dict, *, qa: bool = False, _final_capability=None) -> dict:
        base = _verified(manifest)
        pair = self.generator.batch(
            base,
            qa=qa,
            _final_capability=_final_capability,
        )
        moving = pair["moving"].detach().cpu().numpy()[:, 0]
        moving_mask = pair["moving_model_mask"].detach().cpu().numpy()[:, 0]
        pose_views = [
            _expanded_similarity(image, mask, rotation, scale)
            for image, mask, rotation, scale in zip(
                moving,
                moving_mask,
                manifest["pose_view_rotation_deg"],
                manifest["pose_view_scale"],
            )
        ]
        pose_image = torch.from_numpy(
            np.stack(
                [
                    preprocess_atlas_pose_image(image, mask)
                    for image, mask in pose_views
                ]
            )
        ).to(self.generator.device)

        true_pose = torch.as_tensor(
            manifest["true_pose"], device=self.generator.device, dtype=torch.float32
        )
        initial_offset = torch.as_tensor(
            manifest["initial_pose_offset"], device=self.generator.device, dtype=torch.float32
        )
        wrong_offset = torch.as_tensor(
            manifest["wrong_candidate_offset"], device=self.generator.device, dtype=torch.float32
        )
        initial_pose = true_pose + initial_offset
        wrong_pose = true_pose[:, None] + wrong_offset

        initial_fixed, initial_mask, initial_labels = self.render_pose(initial_pose)
        batch, candidates = wrong_pose.shape[:2]
        wrong_fixed, wrong_mask, wrong_labels = self.render_pose(
            wrong_pose.reshape(batch * candidates, 3)
        )
        shape = wrong_fixed.shape[1:]
        result = dict(pair)
        result.update(
            pose_image=pose_image.float(),
            true_pose=true_pose,
            initial_pose=initial_pose,
            initial_pose_offset=initial_offset,
            initial_fixed=initial_fixed,
            initial_fixed_mask=initial_mask,
            initial_fixed_labels=initial_labels,
            wrong_candidate_pose=wrong_pose,
            wrong_candidate_offset=wrong_offset,
            wrong_candidate_fixed=wrong_fixed.reshape(batch, candidates, *shape),
            wrong_candidate_fixed_mask=wrong_mask.reshape(batch, candidates, *shape),
            wrong_candidate_fixed_labels=wrong_labels.reshape(batch, candidates, *shape),
            wrong_candidate_dense_target_valid=torch.zeros(
                (batch, candidates), device=self.generator.device, dtype=torch.bool
            ),
            true_dense_target_valid=torch.ones(
                batch, device=self.generator.device, dtype=torch.bool
            ),
            orientation_inverted_target=torch.as_tensor(
                np.abs(manifest["pose_view_total_rotation_deg"]) > 90.0,
                device=self.generator.device,
                dtype=torch.bool,
            ),
            pose_view_rotation_deg=torch.as_tensor(
                manifest["pose_view_rotation_deg"],
                device=self.generator.device,
                dtype=torch.float32,
            ),
            pose_view_scale=torch.as_tensor(
                manifest["pose_view_scale"],
                device=self.generator.device,
                dtype=torch.float32,
            ),
            pose_view_total_rotation_deg=torch.as_tensor(
                manifest["pose_view_total_rotation_deg"],
                device=self.generator.device,
                dtype=torch.float32,
            ),
            artifact_stratum=str(manifest["artifact_stratum"]),
            artifact_stratum_index=torch.full(
                (batch,),
                STRATUM_INDEX[str(manifest["artifact_stratum"])],
                device=self.generator.device,
                dtype=torch.int64,
            ),
            joint_manifest_sha256=str(manifest["joint_manifest_sha256"]),
        )
        return result

    def render_pose(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render physical AP/L-R/D-V poses through the differentiable CCF image path."""
        ap_index = BREGMA_AP_INDEX - pose[:, 0] / VOXEL_UM
        return self.generator.render_planes(ap_index, pose[:, 1], pose[:, 2])

    def generate(
        self,
        count: int,
        split: str,
        seed: int,
        stratum: str,
        negatives_per_sample: int = 6,
        *,
        qa: bool = False,
        _final_capability=None,
    ) -> dict:
        manifest = self.make_manifest(
            count,
            split,
            seed,
            stratum,
            negatives_per_sample,
            _final_capability=_final_capability,
        )
        return self.batch(
            manifest,
            qa=qa,
            _final_capability=_final_capability,
        )
