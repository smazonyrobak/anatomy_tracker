"""Load authenticated frozen joint rows once for conditional local learning."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = str(ROOT / "tmp")
os.environ["TMP"] = str(ROOT / "tmp")

import numpy as np
import torch

from training import arbitrary_plane_finite_row_binding_v6 as frozen

DATA = ROOT / "runs/arbitrary_plane_finite_v6_substantive_data_001"
PREPARED = ROOT / "runs/joint_v6_proposal_substantive_001"
OUTPUT = ROOT / "data/joint_v6_local_refinement_frozen_001"
MANIFESTS = {
    "training": "af778499dd77ac13f673ce0778972218519f421b97fa01c4d055e00d1dd16266",
    "internal_development": "22d6ae273f13974250d74a2f5b32446a49a1c2c3051d40ad45706300aa1159de",
}
PREPARED_SHA256 = {
    "training": "27a1084bdc22c98a10808f8059659199aad1b789745392e10c92f827c919d07d",
    "internal_development": "89e078582aaa365299d9823deb2a005308f360ab8909db7125a4534556d622c4",
}
ARRAYS = {
    "channels": "model_input_channels_float32",
    "target_ccf_coordinates_ap_dv_ml_um_float64": "target_ccf_coordinates_ap_dv_ml_um_float64",
    "truth_stationary_velocity_yx_px": "truth_section_pullback_stationary_velocity_yx_px_float64",
    "truth_pullback_map_yx_px": "truth_section_pullback_map_yx_px_float64",
    "truth_section_deformation_valid_mask": "truth_section_deformation_valid_mask",
    "target_valid_correspondence_mask": "target_valid_correspondence_mask",
    "target_correspondence_abstention_mask": "target_correspondence_abstention_mask",
    "target_correspondence_weight_float32": "target_correspondence_weight_float32",
    "source_tissue_ground_truth_mask": "source_tissue_ground_truth_mask",
}

OUTPUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
started = time.perf_counter()
repository = Path(__file__).resolve().parents[1]
source_names = (
    "training/prepare_joint_v6_local_refinement_pack.py",
    "training/arbitrary_plane_finite_row_binding_v6.py",
    "training/arbitrary_plane_training_data_v6.py",
    "training/arbitrary_plane_joint_loss_v6.py",
)
summary = {
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
    "source_sha256": {name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in source_names},
    "scope": "conditional teacher-forced local learning, not honest global evaluation",
    "grouping": "organizational synthetic groups from one Allen atlas, not biological subjects",
    "selection": "all frozen joint-component rows, including zero-weight rows",
    "excluded_pose_anchors": "3072 training / 384 development identity-G1 rows have dense supervision weight zero; no zero-deformation supervision is inferred",
    "dense_coordinates": "AP-DV-ML micrometres, observed reflected raster; invalid values remain governed by original masks",
    "deformation_coordinates": "y-x pixel pullback/velocity in observed reflected raster; float64 preserved",
    "reflection": "two bound states, identity/horizontal; stored 3x3 pixel affine is not the model's normalized 2x3 sampling affine",
    "psf": "original nine offsets and global normalized weights; no per-pixel renormalization",
    "omitted_array": "source_label_ground_truth_canvas_int64 is not needed by local geometric losses and remains in the source cache",
    "learned_dependencies": [],
    "partitions": {},
}

for partition, manifest_receipt in MANIFESTS.items():
    cache = DATA / f"{partition}_cache"
    manifest = frozen.load_frozen_row_cache_manifest_v6(cache, expected_manifest_receipt_sha256=manifest_receipt)
    selected = [record for record in manifest["rows"] if "-joint-" in record["lineage"]["section_id"]]
    prepared_path = PREPARED / f"{partition}_prepared.pt"
    with prepared_path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == PREPARED_SHA256[partition]
    prepared = torch.load(prepared_path, map_location="cpu", weights_only=False)
    prepared_index = {record["training_row_id"]: index for index, record in enumerate(prepared["records"])}
    indices = torch.tensor([prepared_index[record["training_row_id"]] for record in selected])
    pack = {
        "prepared_row_index": indices,
        "frozen_cache_row_index": torch.tensor([record["row_index"] for record in selected]),
        "truth_state": prepared["truth_state"][indices].clone(),
        "truth_catalogue_index": prepared["label"][indices].clone(),
        "pose_supervision_weight": prepared["weight"][indices].clone(),
        "dense_deformation_supervision_weight": torch.empty(len(selected)),
        "axial_offsets_um": torch.empty((len(selected), 9), dtype=torch.float64),
        "axial_weights": torch.empty((len(selected), 9), dtype=torch.float64),
        "reflection_representation_index": torch.empty(len(selected), dtype=torch.int64),
        "reflection_representation_affine_xy_float64": torch.empty((len(selected), 3, 3), dtype=torch.float64),
        "records": [],
        "finite_psf_contracts": [],
        "cache_directory": str(cache),
        "cache_manifest_receipt_sha256": manifest_receipt,
        "prepared_source_sha256": PREPARED_SHA256[partition],
    }
    for local_index, record in enumerate(selected):
        row = frozen._load_record(cache, record, manifest)
        previous = prepared["records"][indices[local_index].item()]
        assert previous["row_receipt_sha256"] == row["receipt_sha256"]
        support = row["upstream_reference"]["support_supervision_contract"]
        assert pack["pose_supervision_weight"][local_index].item() == support["point_pose_supervision_weight"]
        for key, source_key in ARRAYS.items():
            array = row["arrays"][source_key]
            tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1) if array.ndim == 3 else array[None]))
            if local_index == 0:
                pack[key] = torch.empty((len(selected), *tensor.shape), dtype=tensor.dtype)
            pack[key][local_index].copy_(tensor)
        pack["dense_deformation_supervision_weight"][local_index] = support["dense_deformation_supervision_weight"]
        pack["axial_offsets_um"][local_index] = torch.tensor(row["finite_psf_contract"]["axial_offsets_um"], dtype=torch.float64)
        pack["axial_weights"][local_index] = torch.tensor(row["finite_psf_contract"]["axial_weights"], dtype=torch.float64)
        pack["reflection_representation_index"][local_index] = row["reflection_representation_index"]
        pack["reflection_representation_affine_xy_float64"][local_index] = torch.tensor(row["reflection_representation_affine_xy_float64"], dtype=torch.float64)
        pack["records"].append({
            **previous, "frozen_cache_record": record,
            "source_observation_receipt_sha256": row["source_observation_receipt_sha256"],
            "synthetic_realization_id": row["synthetic_realization_id"],
            "support_supervision_contract": support,
            "deformation_pose_gauge_reference": row["deformation_pose_gauge_reference"],
            "canonical_effective_quicknii_ouv_float64": row["canonical_effective_quicknii_ouv_float64"],
            "observed_effective_quicknii_ouv_float64": row["observed_effective_quicknii_ouv_float64"],
        })
        pack["finite_psf_contracts"].append(row["finite_psf_contract"])
        if (local_index + 1) % 256 == 0:
            print(f"Authenticated {partition} {local_index + 1}/{len(selected)} rows in {time.perf_counter() - started:.1f}s", flush=True)
    pack["image"] = pack["channels"][:, :1]
    pack["outline"] = pack["channels"][:, 1:2]
    pack["outline_available"] = pack["channels"][:, 2].mean((-2, -1))
    pack["retrieval_supervision_weight"] = pack["pose_supervision_weight"]
    pack["deformation_weight"] = (
        pack["truth_section_deformation_valid_mask"] & pack["target_valid_correspondence_mask"]
        & ~pack["target_correspondence_abstention_mask"]
    ).float() * pack["target_correspondence_weight_float32"]
    path = OUTPUT / f"{partition}.pt"
    torch.save(pack, path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    summary["partitions"][partition] = {
        "path": str(path), "sha256": digest, "bytes": path.stat().st_size,
        "row_count": len(selected), "cache_manifest_receipt_sha256": manifest_receipt,
        "prepared_source_sha256": PREPARED_SHA256[partition],
        "pose_supervised_rows": int((pack["pose_supervision_weight"] > 0).sum()),
        "dense_supervised_rows": int((pack["dense_deformation_supervision_weight"] > 0).sum()),
        "reflection_counts": torch.bincount(pack["reflection_representation_index"], minlength=2).tolist(),
        "tensor_fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in pack.items() if isinstance(value, torch.Tensor)},
    }
    print(json.dumps({"partition": partition, **{key: value for key, value in summary["partitions"][partition].items() if key != "tensor_fields"}}), flush=True)
    del prepared, pack, manifest

summary["elapsed_seconds"] = time.perf_counter() - started
(OUTPUT / "pack.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Completed local-refinement pack in {summary['elapsed_seconds']:.1f}s", flush=True)
