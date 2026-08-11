"""Animal-disjoint real-histology evidence for residual registration.

Dense accuracy is measured only after applying a known synthetic
diffeomorphism to real Allen histology texture. Native atlas/histology pairs
have no dense ground truth and therefore contribute only acceptance,
topology, MIND, and surface-overlap non-degradation evidence.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from functools import lru_cache
from pathlib import Path

import cv2
import nrrd
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import map_coordinates

from source.atlas_pose_runtime import AUTOMATIC_BRAIN_MASK_VERSION, automatic_brain_mask
from source.registered_image_quality import (
    REGISTERED_IMAGE_QUALITY_MANIFEST,
    REGISTERED_IMAGE_QUALITY_VERSION,
    load_registered_image_quality_manifest,
)
from source.nonlinear_registration import (
    MODEL_PIXEL_SPACING_UM,
    MODEL_SHAPE,
    NonlinearWarp2D,
    nonlinear_acceptance_failures,
)
from training.acquire_allen_s2p import SPLIT_SALT
from training.diffeomorphic_registration_model import (
    hard_cell_mask,
    jacobian_determinant,
    mind_loss,
    pixel_identity_grid,
    preprocess_registration_tensor,
    sample_at_pixel_map,
)


REAL_HISTOLOGY_CONTRACT_VERSION = 2
ALLEN_IMAGE_DOWNSAMPLE = 5
REAL_HISTOLOGY_TRAIN_BANK_SEED = 2_000_039
REAL_HISTOLOGY_TRAIN_BANK_ANIMALS = 256
REAL_HISTOLOGY_SELECTION_SEED = 3_000_051
REAL_HISTOLOGY_LOCKED_SEED = 4_000_073
REAL_HISTOLOGY_MAX_ANIMALS = 20
REAL_HISTOLOGY_SECTIONS_PER_ANIMAL = 4
REAL_HISTOLOGY_MIN_ANIMALS = 20
REAL_HISTOLOGY_MIN_SECTIONS = 80
REAL_HISTOLOGY_BOOTSTRAP_REPLICATES = 5000
MAX_DENSE_EPE_MEDIAN_PX = 1.0
MAX_DENSE_EPE_P95_PX = 2.0
MAX_TRE_MEDIAN_PX = 1.0
MAX_TRE_P95_PX = 2.0
MAX_JACOBIAN_ERROR_P95 = 0.20
MIN_DENSE_IMPROVEMENT_PX = 0.50
MIN_DENSE_RELATIVE_IMPROVEMENT = 0.25
MIN_INTERIOR_IMPROVEMENT_PX = 0.15
MIN_VALID_ACCEPT_RATE = 0.95
MIN_NATIVE_ACCEPT_RATE = 0.90
MAX_NATIVE_MIND_INCREASE = 0.01
MAX_NATIVE_SURFACE_DICE_LOSS = 0.01
MIN_NATIVE_RETAINED_COVERAGE = 0.95
ALLOWED_SPLITS = {"train", "validation", "test"}
DENSE_STRATA = (
    "real_histology_interior_label_free",
    "smooth_deformation_label_free",
    "nuisance_damage_label_free",
)


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def torch_model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def callable_sha256(function) -> str:
    source_path = inspect.getsourcefile(function)
    payload = inspect.getsource(function).encode("utf-8")
    if source_path:
        payload += Path(source_path).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _hash_order(seed: int, kind: str, value: int) -> bytes:
    return hashlib.sha256(f"{seed}:{kind}:{int(value)}".encode()).digest()


def _source_contract(root: Path, atlas_folder: Path) -> dict:
    required = (
        root / "datasets.jsonl",
        root / "sections.jsonl",
        root / "downloads.jsonl",
        root / "provenance.json",
        root / REGISTERED_IMAGE_QUALITY_MANIFEST,
        atlas_folder / "average_template_25.nrrd",
        atlas_folder / "annotation_25.nrrd",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Real-histology evidence is incomplete: " + ", ".join(missing))
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    datasets_sha256 = file_sha256(root / "datasets.jsonl")
    sections_sha256 = file_sha256(root / "sections.jsonl")
    if provenance.get("datasets_sha256") != datasets_sha256:
        raise ValueError("Allen datasets.jsonl differs from acquisition provenance")
    if provenance.get("sections_sha256") != sections_sha256:
        raise ValueError("Allen sections.jsonl differs from acquisition provenance")
    if provenance.get("split") != {
        "unit": "specimen_id",
        "salt": SPLIT_SALT,
        "fractions": [0.9, 0.05, 0.05],
    }:
        raise ValueError("Allen specimen split contract differs from the acquisition contract")
    if provenance.get("download", {}).get("downsample") != ALLEN_IMAGE_DOWNSAMPLE:
        raise ValueError("Allen image downsample differs from the registered-coordinate contract")
    return {
        "contract_version": REAL_HISTOLOGY_CONTRACT_VERSION,
        "datasets_sha256": datasets_sha256,
        "sections_sha256": sections_sha256,
        "downloads_sha256": file_sha256(root / "downloads.jsonl"),
        "provenance_sha256": file_sha256(root / "provenance.json"),
        "average_template_sha256": file_sha256(atlas_folder / "average_template_25.nrrd"),
        "annotation_sha256": file_sha256(atlas_folder / "annotation_25.nrrd"),
        "split_unit": "specimen_id",
        "split_salt": SPLIT_SALT,
        "pixel_spacing_um": MODEL_PIXEL_SPACING_UM,
        "automatic_brain_mask_version": AUTOMATIC_BRAIN_MASK_VERSION,
        "registered_image_quality_version": REGISTERED_IMAGE_QUALITY_VERSION,
        "registered_image_quality_manifest_sha256": file_sha256(
            root / REGISTERED_IMAGE_QUALITY_MANIFEST
        ),
        "evaluator_sha256": file_sha256(__file__),
    }


class RegisteredHistologySource:
    """Immutable Allen records and native 25-um atlas/histology canvases."""

    def __init__(self, root: str | Path, atlas_folder: str | Path):
        self.root = Path(root)
        self.atlas_folder = Path(atlas_folder)
        self.contract = _source_contract(self.root, self.atlas_folder)
        quality_manifest, approved_section_ids, rejected_records = (
            load_registered_image_quality_manifest(self.root)
        )
        self.quality_manifest = quality_manifest
        self.rejected_records = rejected_records
        datasets = _read_jsonl(self.root / "datasets.jsonl")
        sections = _read_jsonl(self.root / "sections.jsonl")
        downloads = _read_jsonl(self.root / "downloads.jsonl")
        specimen_splits: dict[int, str] = {}
        for record in datasets:
            specimen_id = int(record["specimen_id"])
            previous = specimen_splits.setdefault(specimen_id, record["split"])
            if previous != record["split"]:
                raise ValueError(f"Specimen {specimen_id} crosses Allen splits")
        self.datasets = {int(record["experiment_id"]): record for record in datasets}
        if len(self.datasets) != len(datasets):
            raise ValueError("Allen datasets.jsonl contains duplicate experiment IDs")
        self.records: dict[str, list[dict]] = {split: [] for split in (*ALLOWED_SPLITS, "sealed_deepslice_s2p")}
        section_ids = set()
        for record in sections:
            section_id = int(record["section_image_id"])
            if section_id in section_ids:
                raise ValueError("Allen sections.jsonl contains duplicate section IDs")
            section_ids.add(section_id)
            dataset = self.datasets[int(record["experiment_id"])]
            if (
                record["split"] != dataset["split"]
                or int(record["specimen_id"]) != int(dataset["specimen_id"])
            ):
                raise ValueError("Allen section and dataset split manifests disagree")
            if record["split"] in ALLOWED_SPLITS and section_id not in approved_section_ids:
                continue
            self.records.setdefault(record["split"], []).append(record)
        self.downloads = {int(record["section_image_id"]): record["sha256"] for record in downloads}
        if len(self.downloads) != len(downloads):
            raise ValueError("Allen downloads.jsonl contains duplicate section IDs")
        self.by_section_id = {
            int(record["section_image_id"]): record
            for split in ALLOWED_SPLITS
            for record in self.records[split]
        }
        self.average = nrrd.read(str(self.atlas_folder / "average_template_25.nrrd"))[0].astype(np.float32)
        self.average /= max(float(self.average.max()), 1.0)
        self.brain = nrrd.read(str(self.atlas_folder / "annotation_25.nrrd"))[0] > 0
        if self.average.shape != self.brain.shape:
            raise ValueError("Allen average template and annotation volumes have different shapes")
        if self.average.shape[1] > MODEL_SHAPE[0] or self.average.shape[2] > MODEL_SHAPE[1]:
            raise ValueError("Allen native coronal plane does not fit the nonlinear model canvas")

    def evaluation_manifest(self, split: str, seed: int) -> dict:
        if split not in {"validation", "test"}:
            raise ValueError("Real-histology evaluation is restricted to validation or test animals")
        grouped: dict[int, list[dict]] = {}
        for record in self.records[split]:
            grouped.setdefault(int(record["specimen_id"]), []).append(record)
        specimen_ids = sorted(grouped, key=lambda value: _hash_order(seed, "specimen", value))[
            :REAL_HISTOLOGY_MAX_ANIMALS
        ]
        entries = []
        for specimen_id in specimen_ids:
            ordered = sorted(
                grouped[specimen_id],
                key=lambda record: (
                    float(record["ap_um"]),
                    _hash_order(seed, "section", int(record["section_image_id"])),
                ),
            )
            positions = np.unique(
                np.rint(np.linspace(0, len(ordered) - 1, min(REAL_HISTOLOGY_SECTIONS_PER_ANIMAL, len(ordered))))
                .astype(int)
            )
            selected = [ordered[position] for position in positions]
            for record in selected:
                section_id = int(record["section_image_id"])
                if section_id not in self.downloads:
                    raise FileNotFoundError(f"Section {section_id} is absent from downloads.jsonl")
                entries.append({
                    "specimen_id": specimen_id,
                    "experiment_id": int(record["experiment_id"]),
                    "section_image_id": section_id,
                    "image_sha256": self.downloads[section_id],
                    "ap_um": float(record["ap_um"]),
                    "synthetic_seeds": {
                        stratum: int.from_bytes(
                            _hash_order(seed, stratum, section_id)[:8], "big"
                        ) % 900_000_000
                        for stratum in DENSE_STRATA
                    },
                })
        payload = {
            "contract_version": REAL_HISTOLOGY_CONTRACT_VERSION,
            "benchmark_role": "checkpoint_selection" if split == "validation" else "locked_promotion_gate",
            "split": split,
            "seed": int(seed),
            "bootstrap_seed": int(seed) + 7919,
            "source": self.contract,
            "selection": {
                "maximum_animals": REAL_HISTOLOGY_MAX_ANIMALS,
                "sections_per_animal": REAL_HISTOLOGY_SECTIONS_PER_ANIMAL,
                "section_selection": "AP-stratified endpoints and equally spaced order statistics",
                "specimen_count": len(specimen_ids),
                "section_count": len(entries),
            },
            "entries": entries,
            "sealed_data_used": False,
        }
        payload["manifest_sha256"] = canonical_sha256(payload)
        return payload

    def training_bank_manifest(self) -> dict:
        grouped: dict[int, list[dict]] = {}
        for record in self.records["train"]:
            grouped.setdefault(int(record["specimen_id"]), []).append(record)
        specimen_ids = sorted(
            grouped,
            key=lambda value: _hash_order(REAL_HISTOLOGY_TRAIN_BANK_SEED, "specimen", value),
        )[:REAL_HISTOLOGY_TRAIN_BANK_ANIMALS]
        entries = []
        for specimen_id in specimen_ids:
            record = min(
                grouped[specimen_id],
                key=lambda value: _hash_order(
                    REAL_HISTOLOGY_TRAIN_BANK_SEED, "section", int(value["section_image_id"])
                ),
            )
            entries.append({
                "specimen_id": specimen_id,
                "experiment_id": int(record["experiment_id"]),
                "section_image_id": int(record["section_image_id"]),
                "image_sha256": self.downloads[int(record["section_image_id"])],
            })
        payload = {
            "contract_version": REAL_HISTOLOGY_CONTRACT_VERSION,
            "benchmark_role": "bounded_training_texture_bank",
            "split": "train",
            "seed": REAL_HISTOLOGY_TRAIN_BANK_SEED,
            "source": self.contract,
            "entries": entries,
            "specimen_count": len(entries),
            "sealed_data_used": False,
        }
        payload["manifest_sha256"] = canonical_sha256(payload)
        return payload

    def verify_manifest(self, manifest: dict) -> None:
        checksum = manifest.get("manifest_sha256")
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if checksum != canonical_sha256(payload):
            raise ValueError("Real-histology evaluation manifest checksum failed")
        if manifest.get("source") != self.contract:
            raise ValueError("Real-histology evaluation manifest does not match the current source")
        if manifest.get("split") not in {"validation", "test"} or manifest.get("sealed_data_used") is not False:
            raise ValueError("Sealed or non-evaluation Allen data cannot enter this gate")
        if manifest != self.evaluation_manifest(manifest["split"], int(manifest["seed"])):
            raise ValueError("Real-histology manifest is not the preregistered deterministic selection")
        for entry in manifest["entries"]:
            record = self.by_section_id.get(int(entry["section_image_id"]))
            if record is None or record["split"] != manifest["split"]:
                raise ValueError("Real-histology entry is outside its bound specimen split")
            if (
                int(record["specimen_id"]) != int(entry["specimen_id"])
                or int(record["experiment_id"]) != int(entry["experiment_id"])
                or self.downloads.get(int(entry["section_image_id"])) != entry["image_sha256"]
            ):
                raise ValueError("Real-histology entry differs from the immutable Allen manifests")

    @lru_cache(maxsize=384)
    def section(self, section_image_id: int) -> dict[str, np.ndarray | int]:
        if int(section_image_id) in self.rejected_records:
            raise ValueError(
                f"Allen image {section_image_id} was rejected by the registered-image quality contract"
            )
        record = self.by_section_id[int(section_image_id)]
        dataset = self.datasets[int(record["experiment_id"])]
        path = self.root / record["relative_path"]
        expected_sha256 = self.downloads.get(int(section_image_id))
        if expected_sha256 is None or file_sha256(path) != expected_sha256:
            raise ValueError(f"Allen image {section_image_id} fails downloads.jsonl integrity")
        with Image.open(path) as image:
            observed = np.asarray(image).copy()
        observed_mask = np.asarray(automatic_brain_mask(observed), dtype=bool)
        fixed, moving, fixed_mask, moving_mask = canonical_registered_pair(
            observed,
            observed_mask,
            self.average,
            self.brain,
            dataset,
            record,
        )
        return {
            "fixed": fixed,
            "moving": moving,
            "fixed_mask": fixed_mask,
            "moving_mask": moving_mask,
            "specimen_id": int(record["specimen_id"]),
            "experiment_id": int(record["experiment_id"]),
            "section_image_id": int(section_image_id),
        }


def downloaded_pixel_to_reference_index(dataset: dict, section: dict) -> np.ndarray:
    """Return the affine mapping downloaded ``(x,y,1)`` to Allen ``(AP,DV,ML)`` indices."""
    tsv = np.asarray(section["alignment2d_tsv"], dtype=np.float64)
    downsample_factor = float(2**ALLEN_IMAGE_DOWNSAMPLE)
    pixel_center_offset = (downsample_factor - 1.0) / 2.0
    image_to_volume = np.asarray(
        (
            (tsv[0] * downsample_factor, tsv[1] * downsample_factor,
             (tsv[0] + tsv[1]) * pixel_center_offset + tsv[4]),
            (tsv[2] * downsample_factor, tsv[3] * downsample_factor,
             (tsv[2] + tsv[3]) * pixel_center_offset + tsv[5]),
            (0.0, 0.0, float(section["section_number"]) * float(dataset["section_thickness_um"])),
        ),
        dtype=np.float64,
    )
    tvr = np.asarray(dataset["alignment3d_tvr"], dtype=np.float64)
    reference = tvr[:9].reshape(3, 3) @ image_to_volume
    reference[:, 2] += tvr[9:12]
    return reference / MODEL_PIXEL_SPACING_UM


def canonical_registered_pair(
    image: np.ndarray,
    observed_mask: np.ndarray,
    average: np.ndarray,
    atlas_mask: np.ndarray,
    dataset: dict,
    section: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample official Allen registration onto the one-to-one 25-um model canvas."""
    transform = downloaded_pixel_to_reference_index(dataset, section)
    ml_dv = transform[[2, 1]]
    inverse = np.linalg.inv(ml_dv[:, :2])
    atlas_top = (MODEL_SHAPE[0] - average.shape[1]) // 2
    atlas_left = (MODEL_SHAPE[1] - average.shape[2]) // 2
    yy, xx = np.mgrid[:MODEL_SHAPE[0], :MODEL_SHAPE[1]].astype(np.float64)
    atlas_ml = xx - atlas_left
    atlas_dv = yy - atlas_top
    source = inverse @ np.stack(
        (atlas_ml - ml_dv[0, 2], atlas_dv - ml_dv[1, 2]), axis=0
    ).reshape(2, -1)
    source_x = source[0].reshape(MODEL_SHAPE).astype(np.float32)
    source_y = source[1].reshape(MODEL_SHAPE).astype(np.float32)
    atlas_ap = (
        transform[0, 0] * source_x + transform[0, 1] * source_y + transform[0, 2]
    )
    atlas_ml = atlas_ml.astype(np.float32)
    atlas_dv = atlas_dv.astype(np.float32)
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = gray[..., :3].astype(np.float32).mean(axis=2)
    gray = gray.astype(np.float32)
    moving = cv2.remap(gray, source_x, source_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if moving.max() > 1.0:
        moving /= 255.0
    moving_mask = cv2.remap(
        observed_mask.astype(np.uint8), source_x, source_y,
        cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    coordinates = np.stack((atlas_ap, atlas_dv, atlas_ml))
    fixed = map_coordinates(average, coordinates, order=1, mode="constant", cval=0.0).astype(np.float32)
    fixed_mask = map_coordinates(
        atlas_mask.astype(np.uint8), coordinates, order=0, mode="constant", cval=0
    ).astype(bool)
    moving[~moving_mask] = 0.0
    fixed[~fixed_mask] = 0.0
    if not fixed_mask.any() or not moving_mask.any():
        raise ValueError("Registered section has no tissue on the native 25-um model canvas")
    return fixed, moving, fixed_mask, moving_mask


def _percentile(values: torch.Tensor, quantile: float) -> float:
    return float(torch.quantile(values.float(), quantile))


def _map_pair_sha256(forward: torch.Tensor, inverse: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in (forward, inverse):
        array = value.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        digest.update(array.tobytes())
    return digest.hexdigest()


def _dice(left: torch.Tensor, right: torch.Tensor) -> float:
    left, right = left > 0.5, right > 0.5
    return float(2.0 * (left & right).sum() / (left.sum() + right.sum()).clamp_min(1))


def _warp_diagnostics(
    forward: torch.Tensor,
    inverse: torch.Tensor,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
) -> tuple[dict, list[str]]:
    warp = NonlinearWarp2D(
        np.moveaxis(forward.detach().cpu().numpy(), 0, -1),
        np.moveaxis(inverse.detach().cpu().numpy(), 0, -1),
    )
    diagnostics = warp.diagnostics(
        fixed_mask.detach().cpu().numpy() > 0.5,
        moving_mask.detach().cpu().numpy() > 0.5,
    )
    return diagnostics, nonlinear_acceptance_failures(diagnostics)


def _dense_row(outputs: tuple[torch.Tensor, ...], pair: dict, item: int, identifiers: dict) -> dict:
    forward, inverse, _, rejection_logit = outputs
    predicted_maps = (forward[item], inverse[item])
    target_maps = (pair["target_atlas_to_affine"][item], pair["target_affine_to_atlas"][item])
    masks = (pair["atlas_supervision_mask"][item, 0] > 0.5,
             pair["affine_supervision_mask"][item, 0] > 0.5)
    errors, baselines = [], []
    for predicted, target, mask in zip(predicted_maps, target_maps, masks):
        identity = pixel_identity_grid(
            1, *target.shape[-2:], device=target.device, dtype=target.dtype
        )[0]
        errors.append(torch.linalg.vector_norm(predicted - target, dim=0)[mask])
        baselines.append(torch.linalg.vector_norm(identity - target, dim=0)[mask])
    epe = torch.cat(errors)
    baseline = torch.cat(baselines)
    lattice = torch.zeros_like(masks[0])
    lattice[8::16, 8::16] = True
    tre_mask = masks[0] & lattice
    if tre_mask.sum() < 16:
        tre_mask = masks[0]
    tre = torch.linalg.vector_norm(predicted_maps[0] - target_maps[0], dim=0)[tre_mask]
    target_tre = torch.linalg.vector_norm(
        pixel_identity_grid(1, *target_maps[0].shape[-2:], device=target_maps[0].device)[0]
        - target_maps[0], dim=0,
    )[tre_mask]
    jacobian_errors = []
    for predicted, target, mask in zip(predicted_maps, target_maps, masks):
        cell_mask = hard_cell_mask(mask[None, None])[0, 0]
        jacobian_errors.append(
            (jacobian_determinant(predicted[None])[0] - jacobian_determinant(target[None])[0]).abs()[cell_mask]
        )
    jacobian_error = torch.cat(jacobian_errors)
    diagnostics, failures = _warp_diagnostics(
        predicted_maps[0], predicted_maps[1],
        pair["fixed_mask"][item, 0], pair["moving_mask"][item, 0],
    )
    epe_median = _percentile(epe, 0.5)
    epe_p95 = _percentile(epe, 0.95)
    baseline_median = _percentile(baseline, 0.5)
    baseline_p95 = _percentile(baseline, 0.95)
    tre_median = _percentile(tre, 0.5)
    tre_p95 = _percentile(tre, 0.95)
    baseline_tre_median = _percentile(target_tre, 0.5)
    baseline_tre_p95 = _percentile(target_tre, 0.95)
    return {
        **identifiers,
        "epe_median_px": epe_median,
        "epe_p95_px": epe_p95,
        "baseline_epe_median_px": baseline_median,
        "baseline_epe_p95_px": baseline_p95,
        "epe_improvement_px": baseline_median - epe_median,
        "epe_relative_improvement": (baseline_median - epe_median) / max(baseline_median, 1e-6),
        "epe_p95_improvement_px": baseline_p95 - epe_p95,
        "epe_p95_relative_improvement": (baseline_p95 - epe_p95) / max(baseline_p95, 1e-6),
        "tre_median_px": tre_median,
        "tre_p95_px": tre_p95,
        "baseline_tre_median_px": baseline_tre_median,
        "baseline_tre_p95_px": baseline_tre_p95,
        "tre_improvement_px": baseline_tre_median - tre_median,
        "tre_p95_improvement_px": baseline_tre_p95 - tre_p95,
        "jacobian_error_p95": _percentile(jacobian_error, 0.95),
        "target_maps_sha256": _map_pair_sha256(target_maps[0], target_maps[1]),
        "accepted": float(torch.sigmoid(rejection_logit[item]) < 0.5),
        "geometry_passed": not failures,
        "geometry_failures": failures,
        "geometry": diagnostics,
    }


def _native_row(outputs: tuple[torch.Tensor, ...], batch: dict, item: int, identifiers: dict) -> dict:
    forward, inverse, _, rejection_logit = outputs
    fixed = batch["fixed"][item : item + 1]
    moving = batch["moving"][item : item + 1]
    fixed_mask = batch["fixed_mask"][item : item + 1]
    moving_mask = batch["moving_mask"][item : item + 1]
    identity = pixel_identity_grid(1, *fixed.shape[-2:], device=fixed.device, dtype=fixed.dtype)
    before_overlap = (fixed_mask > 0.5) & (moving_mask > 0.5)
    warped_mask = sample_at_pixel_map(moving_mask, forward[item : item + 1], padding_mode="zeros") > 0.5
    mind_before = float(mind_loss(fixed, moving, identity, before_overlap.float()))
    mind_after = float(mind_loss(fixed, moving, forward[item : item + 1], before_overlap.float()))
    surface_before = _dice(fixed_mask, moving_mask)
    surface_after = _dice(fixed_mask, warped_mask)
    retained_coverage = float((before_overlap & warped_mask).sum() / before_overlap.sum().clamp_min(1))
    diagnostics, failures = _warp_diagnostics(
        forward[item], inverse[item], fixed_mask[0, 0], moving_mask[0, 0]
    )
    return {
        **identifiers,
        "mind_before": mind_before,
        "mind_after": mind_after,
        "mind_delta": mind_after - mind_before,
        "surface_dice_before": surface_before,
        "surface_dice_after": surface_after,
        "surface_dice_delta": surface_after - surface_before,
        "retained_coverage": retained_coverage,
        "accepted": float(torch.sigmoid(rejection_logit[item]) < 0.5),
        "geometry_passed": not failures,
        "geometry_failures": failures,
        "geometry": diagnostics,
    }


def _animal_summary(
    rows: list[dict],
    metrics: tuple[str, ...],
    rate_metrics: tuple[str, ...] = (),
) -> dict[int, dict[str, float]]:
    specimen_ids = sorted({int(row["specimen_id"]) for row in rows})
    return {
        specimen_id: {
            metric: float(
                (np.mean if metric in rate_metrics else np.median)(
                    [row[metric] for row in rows if int(row["specimen_id"]) == specimen_id]
                )
            )
            for metric in metrics
        }
        for specimen_id in specimen_ids
    }


def _animal_bootstrap(
    animals: dict[int, dict[str, float]],
    metrics: tuple[str, ...],
    seed: int,
) -> dict[str, dict[str, float]]:
    values = np.asarray([[row[metric] for metric in metrics] for row in animals.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), (REAL_HISTOLOGY_BOOTSTRAP_REPLICATES, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        metric: {
            "estimate": float(values[:, column].mean()),
            "lower95": float(np.quantile(samples[:, column], 0.025)),
            "upper95": float(np.quantile(samples[:, column], 0.975)),
        }
        for column, metric in enumerate(metrics)
    }


def real_histology_gate_failures(gates: dict) -> list[str]:
    checks = (
        (gates["animal_count"] >= REAL_HISTOLOGY_MIN_ANIMALS, "fewer than 20 independent animals"),
        (gates["section_count"] >= REAL_HISTOLOGY_MIN_SECTIONS, "fewer than 80 registered sections"),
        (gates["dense_epe_median_px"] <= MAX_DENSE_EPE_MEDIAN_PX, "dense EPE median exceeds 1 px"),
        (gates["dense_epe_p95_px"] <= MAX_DENSE_EPE_P95_PX, "dense EPE p95 exceeds 2 px"),
        (gates["tre_median_px"] <= MAX_TRE_MEDIAN_PX, "sparse TRE median exceeds 1 px"),
        (gates["tre_p95_px"] <= MAX_TRE_P95_PX, "sparse TRE p95 exceeds 2 px"),
        (gates["jacobian_error_p95"] <= MAX_JACOBIAN_ERROR_P95, "Jacobian error p95 exceeds 0.20"),
        (gates["epe_improvement_px"] >= MIN_DENSE_IMPROVEMENT_PX, "dense EPE improvement is below 0.5 px"),
        (
            gates["interior_epe_improvement_px"] >= MIN_INTERIOR_IMPROVEMENT_PX,
            "interior-only dense EPE improvement is below 0.15 px",
        ),
        (
            gates["epe_relative_improvement"] >= MIN_DENSE_RELATIVE_IMPROVEMENT,
            "dense EPE relative improvement is below 25%",
        ),
        (gates["epe_p95_improvement_px"] >= MIN_DENSE_IMPROVEMENT_PX, "dense EPE p95 improvement is below 0.5 px"),
        (
            gates["interior_epe_p95_improvement_px"] >= MIN_INTERIOR_IMPROVEMENT_PX,
            "interior-only dense EPE p95 improvement is below 0.15 px",
        ),
        (
            gates["epe_p95_relative_improvement"] >= MIN_DENSE_RELATIVE_IMPROVEMENT,
            "dense EPE p95 relative improvement is below 25%",
        ),
        (gates["dense_accept_rate"] >= MIN_VALID_ACCEPT_RATE, "valid dense-pair acceptance is below 95%"),
        (gates["native_accept_rate"] >= MIN_NATIVE_ACCEPT_RATE, "native-pair acceptance is below 90%"),
        (gates["native_mind_delta"] <= MAX_NATIVE_MIND_INCREASE, "native MIND worsens by more than 0.01"),
        (
            gates["native_surface_dice_delta"] >= -MAX_NATIVE_SURFACE_DICE_LOSS,
            "native surface Dice drops by more than 0.01",
        ),
        (
            gates["native_retained_coverage"] >= MIN_NATIVE_RETAINED_COVERAGE,
            "native warped support retains less than 95% of the fixed evaluation support",
        ),
        (gates["geometry_passed"], "at least one real-histology prediction fails geometry gates"),
    )
    return [message for passed, message in checks if not passed]


@torch.inference_mode()
def evaluate_real_histology(
    model,
    source: RegisteredHistologySource,
    manifest: dict,
    pair_factory,
    device: torch.device,
    model_sha256: str,
    batch_size: int = 4,
) -> dict:
    source.verify_manifest(manifest)
    if len(model_sha256) != 64 or any(character not in "0123456789abcdef" for character in model_sha256):
        raise ValueError("Real-histology evidence requires the evaluated model SHA-256")
    dense_rows, native_rows = [], []
    entries = manifest["entries"]
    for start in range(0, len(entries), batch_size):
        chunk = entries[start : start + batch_size]
        sections = [source.section(int(entry["section_image_id"])) for entry in chunk]
        fixed = torch.stack([torch.from_numpy(section["fixed"]) for section in sections])[:, None].to(device)
        moving = torch.stack([torch.from_numpy(section["moving"]) for section in sections])[:, None].to(device)
        fixed_mask = torch.stack([torch.from_numpy(section["fixed_mask"]) for section in sections])[:, None].to(device).float()
        moving_mask = torch.stack([torch.from_numpy(section["moving_mask"]) for section in sections])[:, None].to(device).float()

        native_batch = {
            "fixed": preprocess_registration_tensor(fixed, fixed_mask),
            "moving": preprocess_registration_tensor(moving, moving_mask),
            "fixed_mask": fixed_mask,
            "moving_mask": moving_mask,
        }
        native_outputs = model.eval()(*(native_batch[name] for name in ("fixed", "moving", "fixed_mask", "moving_mask")))
        dense_pairs = []
        dense_identifiers = []
        for item, entry in enumerate(chunk):
            labels = torch.zeros_like(moving[item : item + 1], dtype=torch.long)
            for stratum in DENSE_STRATA:
                dense_pairs.append(pair_factory(
                    moving[item : item + 1], labels, moving_mask[item : item + 1],
                    seed=int(entry["synthetic_seeds"][stratum]), stratum=stratum,
                ))
                dense_identifiers.append({
                    "specimen_id": int(entry["specimen_id"]),
                    "experiment_id": int(entry["experiment_id"]),
                    "section_image_id": int(entry["section_image_id"]),
                    "stratum": stratum,
                })
        dense_batch = {
            key: torch.cat([pair[key] for pair in dense_pairs])
            for key in dense_pairs[0]
        }
        dense_outputs = model.eval()(*(dense_batch[name] for name in ("fixed", "moving", "fixed_mask", "moving_mask")))
        for item, identifiers in enumerate(dense_identifiers):
            dense_rows.append(_dense_row(dense_outputs, dense_batch, item, identifiers))
        for item, entry in enumerate(chunk):
            identifiers = {
                "specimen_id": int(entry["specimen_id"]),
                "experiment_id": int(entry["experiment_id"]),
                "section_image_id": int(entry["section_image_id"]),
            }
            native_rows.append(_native_row(native_outputs, native_batch, item, identifiers))

    dense_metrics = (
        "epe_median_px", "epe_p95_px", "epe_improvement_px", "epe_relative_improvement",
        "epe_p95_improvement_px", "epe_p95_relative_improvement",
        "tre_median_px", "tre_p95_px", "jacobian_error_p95", "accepted",
    )
    native_metrics = ("mind_delta", "surface_dice_delta", "retained_coverage", "accepted")
    dense_animals = {
        stratum: _animal_summary(
            [row for row in dense_rows if row["stratum"] == stratum],
            dense_metrics,
            rate_metrics=("accepted",),
        )
        for stratum in DENSE_STRATA
    }
    native_animals = _animal_summary(native_rows, native_metrics, rate_metrics=("accepted",))
    dense_bootstrap = {
        stratum: _animal_bootstrap(
            dense_animals[stratum], dense_metrics,
            int(manifest["bootstrap_seed"]) + index,
        )
        for index, stratum in enumerate(DENSE_STRATA)
    }
    native_bootstrap = _animal_bootstrap(
        native_animals, native_metrics, int(manifest["bootstrap_seed"]) + 1
    )
    interior = dense_bootstrap["real_histology_interior_label_free"]
    full_deformation = [
        dense_bootstrap[stratum]
        for stratum in DENSE_STRATA
        if stratum != "real_histology_interior_label_free"
    ]
    gates = {
        "animal_count": len(native_animals),
        "section_count": len(entries),
        "dense_epe_median_px": max(
            report["epe_median_px"]["upper95"] for report in dense_bootstrap.values()
        ),
        "dense_epe_p95_px": max(
            report["epe_p95_px"]["upper95"] for report in dense_bootstrap.values()
        ),
        "tre_median_px": max(
            report["tre_median_px"]["upper95"] for report in dense_bootstrap.values()
        ),
        "tre_p95_px": max(
            report["tre_p95_px"]["upper95"] for report in dense_bootstrap.values()
        ),
        "jacobian_error_p95": max(
            report["jacobian_error_p95"]["upper95"] for report in dense_bootstrap.values()
        ),
        "epe_improvement_px": min(
            report["epe_improvement_px"]["lower95"] for report in full_deformation
        ),
        "interior_epe_improvement_px": interior["epe_improvement_px"]["lower95"],
        "epe_relative_improvement": min(
            report["epe_relative_improvement"]["lower95"] for report in dense_bootstrap.values()
        ),
        "epe_p95_improvement_px": min(
            report["epe_p95_improvement_px"]["lower95"] for report in full_deformation
        ),
        "interior_epe_p95_improvement_px": interior["epe_p95_improvement_px"]["lower95"],
        "epe_p95_relative_improvement": min(
            report["epe_p95_relative_improvement"]["lower95"] for report in dense_bootstrap.values()
        ),
        "dense_accept_rate": min(
            report["accepted"]["lower95"] for report in dense_bootstrap.values()
        ),
        "native_accept_rate": native_bootstrap["accepted"]["lower95"],
        "native_mind_delta": native_bootstrap["mind_delta"]["upper95"],
        "native_surface_dice_delta": native_bootstrap["surface_dice_delta"]["lower95"],
        "native_retained_coverage": native_bootstrap["retained_coverage"]["lower95"],
        "geometry_passed": all(row["geometry_passed"] for row in (*dense_rows, *native_rows)),
    }
    failures = real_histology_gate_failures(gates)
    report = {
        "contract_version": REAL_HISTOLOGY_CONTRACT_VERSION,
        "benchmark_role": manifest["benchmark_role"],
        "split": manifest["split"],
        "evaluation_manifest_sha256": manifest["manifest_sha256"],
        "model_sha256": model_sha256,
        "source": manifest["source"],
        "evaluation_manifest": manifest,
        "pair_factory_sha256": callable_sha256(pair_factory),
        "dense_ground_truth": {
            "kind": "exact_synthetic_diffeomorphism_on_held_out_real_histology_texture",
            "claims": ["dense_epe", "sparse_tre", "jacobian_error"],
        },
        "native_pairs": {
            "kind": "official_allen_registered_atlas_histology",
            "claims": ["acceptance", "geometry", "mind_non_degradation", "surface_non_degradation"],
            "dense_ground_truth_available": False,
        },
        "gates": gates,
        "animal_bootstrap": {"dense_by_stratum": dense_bootstrap, "native": native_bootstrap},
        "failures": failures,
        "passed": not failures,
        "dense_rows": dense_rows,
        "native_rows": native_rows,
        "sealed_data_used": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report
