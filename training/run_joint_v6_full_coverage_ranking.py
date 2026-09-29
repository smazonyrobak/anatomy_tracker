"""Fixed remaining-cell coverage, uniform-first then conditional snapshot-hard control."""
import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda")
sys.dont_write_bytecode = True
READY_AFTER_SOURCE_REVIEW = True
assert READY_AFTER_SOURCE_REVIEW, "Prepared only: root reviews, commits and schedules this controlled experiment"

import copy
import hashlib
import json
import random
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_catalogue_runtime_v6 import make_complete_catalogue_runtime_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components, full_frame_state_to_components, full_frame_state_to_physical_ouv, render_finite_thickness_plane
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6

RUN = ROOT / "runs/joint_v6_full_coverage_ranking_001"
PARENT_RUN = ROOT / "runs/joint_v6_canonical_rehearsal_001"
REAL_EXPANSION = ROOT / "data/allen_real_training_expansion_20260929"
REAL_EXPANSION_SUMMARY_SHA256 = "dfe59bde72f55ade32c8b2b6d144fb4ce0064d9fa27bd74c98dd16362def3334"
assert len(REAL_EXPANSION_SUMMARY_SHA256) == 64
BRIDGE = ROOT / "runs/joint_v6_canonical_bridge_001"
PREPARED = ROOT / "runs/joint_v6_proposal_substantive_001"
REPLAY = ROOT / "runs/joint_v6_proposal_curriculum_003"
REAL_TRAIN = ROOT / "data/allen_real_training_inputs_20260929"
REAL_DEV = ROOT / "runs/joint_v6_imagekey_retrieval_001_allen_raw"
# C10000 failed real normal retention; whole experimental lineage, not promotion.
PARENT_CHECKPOINT = PARENT_RUN / "joint_model_step_10000.pt"
PARENT_SHA256 = "888ad9cc493453ce769628fda1aebe70813c031f7809192076c08b000720d9af"
PARENT_AUDIT = ROOT / "runs/joint_v6_canonical_rehearsal_001_independent_audit/audit.json"
PARENT_AUDIT_SHA256 = "8af7c7e208326de15555ec3fc90290d5a8ab2579fd77a80f0dfb8ffebc14934b"
PARENT_COMPLETION = PARENT_RUN / "completed.json"
PARENT_COMPLETION_SHA256 = "be86f269a97fb99275710fd80a95e41d9a742925c8b4d16ae6371d67a8f1cdeb"
B8000_SHA256 = "c6aab521d86eee312f12327965b71c6da1d88f184b7e3427dd7ba2782c88c9f8"
PARENT_REAL_METRICS = BRIDGE / "B/allen_raw/summary.json"
PARENT_REAL_SHA256 = "6e6c21bd57dd1bf3b26a3cb7a93706de65dd1425ff2e4a29bc800196f4fb8025"
REFERENCE_SYNTHETIC = BRIDGE / "parent_synthetic_metrics.json"
REFERENCE_SYNTHETIC_SHA256 = "281fa539824103abed383e5a2c20401377527c53e2bfc2156cfec9d8fb211722"
REFERENCE_REAL = BRIDGE / "parent_real_metrics.json"
REFERENCE_REAL_SHA256 = "465bc4c932b4164f14023c1983dc9a1531f151be21e72aed000e0193e7e04e0b"
REFERENCE_A6000_SHA256 = "280836b65fb6db22930c8ee268eb4a880997c7858fb91a1e31ad6c7c94937eb2"
OPTIONAL_SOURCE_CHANGES = {
    "training/arbitrary_plane_recurrent_model.py",
    "training/arbitrary_plane_recurrent_model_v6.py",
    "training/arbitrary_plane_joint_model_v6.py",
}
REAL_TRAIN_SUMMARY_SHA256 = "4f8484bbd2ab5719971d9862f3f5ea87ea3d1c9de76468a936af67bcc65da9fd"
REAL_DEV_SUMMARY_SHA256 = "2394eae62ef6c64e04b41947e57ca94dd64a2c456e0bc04a58b8dc2997107b37"
STEPS, REAL_BATCH, SYNTHETIC_BATCH, GENERATED, N_REAL = 12576, 8, 8, 4, 1280
APPEARANCE_SEED, FROZEN_SEED, DONOR_SEED, CHART_SEED = 2026092931, 2026092932, 2026092933, 2026092934
NEGATIVE_SEED, CONDITIONAL_SEED, REFRESH_INTERVAL = 2026092935, 2026092936, 1024
SHAPE, DESCRIPTOR_DIM, KEY_CHUNK = (96, 96), 256, 16
TEMPERATURE, KEY_THICKNESS_UM, SUPPORT_MASS_THRESHOLD = .1, 50., 64.
TRAINABLE = ("pose_model.histology_stem.", "pose_model.atlas_stem.", "pose_model.shared_encoder.", "pose_model.image_key_descriptor.")
repository = Path(__file__).resolve().parents[1]
bindings = {}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


for path, expected in ((PARENT_CHECKPOINT, PARENT_SHA256), (PARENT_AUDIT, PARENT_AUDIT_SHA256), (PARENT_COMPLETION, PARENT_COMPLETION_SHA256), (PARENT_REAL_METRICS, PARENT_REAL_SHA256), (REFERENCE_SYNTHETIC, REFERENCE_SYNTHETIC_SHA256), (REFERENCE_REAL, REFERENCE_REAL_SHA256), (REAL_TRAIN / "summary.json", REAL_TRAIN_SUMMARY_SHA256), (REAL_DEV / "summary.json", REAL_DEV_SUMMARY_SHA256)):
    assert sha(path) == expected
    bindings[str(path)] = expected
audit = json.loads(PARENT_AUDIT.read_text())
assert audit["integrity_passed"] and not audit["recovery_gate_passed"]
assert audit["artifact_sha256"][str(PARENT_CHECKPOINT)] == PARENT_SHA256
completed_parent = json.loads(PARENT_COMPLETION.read_text())
assert completed_parent["source_unchanged"] and completed_parent["optimizer_steps_applied"] == 10000
parent = torch.load(PARENT_CHECKPOINT, weights_only=False, map_location="cpu", mmap=True)
previous = parent["experiment"]
assert json.loads(json.dumps(previous)) == completed_parent["experiment"]
assert parent["phase"] == "experimental_image_key_proposal_only"
assert parent["step"] == parent["optimizer_steps_applied"] == 10000
assert tuple(previous["trained_modules"]) == TRAINABLE and previous["parent_sha256"] == B8000_SHA256
assert not previous["model_kwargs"].get("signed_pose_evidence", False)
parent_real = json.loads(PARENT_REAL_METRICS.read_text())
reference_synthetic = json.loads(REFERENCE_SYNTHETIC.read_text())
reference_real = json.loads(REFERENCE_REAL.read_text())
assert parent_real["checkpoint_sha256"] == B8000_SHA256
assert reference_synthetic["checkpoint_sha256"] == reference_real["checkpoint_sha256"] == REFERENCE_A6000_SHA256
parent_real_macro, reference_real_macro = parent_real["donor_macro"], reference_real["donor_macro"]
START, FINAL = 10000, 10000 + STEPS
for directory, expected_hashes in ((PREPARED, previous["prepared_source_sha256"]), (REPLAY, previous["replay_sha256"])):
    for name, expected in expected_hashes.items():
        assert sha(directory / name) == expected
        bindings[str(directory / name)] = expected
real_summary = json.loads((REAL_TRAIN / "summary.json").read_text())
dev_summary = json.loads((REAL_DEV / "summary.json").read_text())
for directory, summary in ((REAL_TRAIN, real_summary), (REAL_DEV, dev_summary)):
    for name in ("raw_model_input.npy", "image_geometry.jsonl"):
        assert sha(directory / name) == summary["output_sha256"][name]
        bindings[str(directory / name)] = summary["output_sha256"][name]
assert sha(REAL_EXPANSION / "summary.json") == REAL_EXPANSION_SUMMARY_SHA256
bindings[str(REAL_EXPANSION / "summary.json")] = REAL_EXPANSION_SUMMARY_SHA256
expansion = json.loads((REAL_EXPANSION / "summary.json").read_text())
assert expansion["additional_section_count"] == 1024 and expansion["union_section_count"] == N_REAL and expansion["donor_count"] == 58 and expansion["failed_download_count"] == 0
for name in ("raw_model_input.npy", "image_geometry.jsonl", "union_training_index.jsonl"):
    assert sha(REAL_EXPANSION / name) == expansion["output_sha256"][name]
    bindings[str(REAL_EXPANSION / name)] = expansion["output_sha256"][name]
real_images = np.concatenate((np.load(REAL_TRAIN / "raw_model_input.npy", allow_pickle=False), np.load(REAL_EXPANSION / "raw_model_input.npy", allow_pickle=False)))
dev_images = np.load(REAL_DEV / "raw_model_input.npy", allow_pickle=False)
real_records = [json.loads(line) for line in (REAL_EXPANSION / "union_training_index.jsonl").read_text().splitlines()]
real_dev_records = [json.loads(line) for line in (REAL_DEV / "image_geometry.jsonl").read_text().splitlines()]
assert real_images.shape == (N_REAL, 1, 96, 96) and dev_images.shape == (64, 1, 96, 96)
assert all(row["split"] == "development_train" and row["union_row_index"] == i for i, row in enumerate(real_records))
assert len({row["section_id"] for row in real_records}) == N_REAL
for i, row in enumerate(real_records):
    assert Path(row["source_array_path"]) == (REAL_TRAIN if i < 256 else REAL_EXPANSION) / "raw_model_input.npy"
    assert row["source_array_row_index"] == (i if i < 256 else i - 256)
assert all(row["split"] == "development_validation" for row in real_dev_records)
donors = np.array([row["animal_id"] for row in real_records]); donor_ids = np.unique(donors)
assert len(donor_ids) == 58 and len({row["animal_id"] for row in real_dev_records}) == 6
for key in ("animal_id", "specimen_id", "experiment_id", "section_id"):
    assert not {row[key] for row in real_records} & {row[key] for row in real_dev_records}
catalogue = torch.load(PREPARED / "catalogue.pt", weights_only=False, map_location="cpu")
train = torch.load(PREPARED / "training_prepared.pt", weights_only=False, map_location="cpu", mmap=True)
dev = torch.load(PREPARED / "internal_development_prepared.pt", weights_only=False, map_location="cpu", mmap=True)
with np.load(REPLAY / "generated_schedule.npz", allow_pickle=False) as saved:
    cell_order = saved["cell_index"][:98304]
assert np.array_equal(np.sort(cell_order), np.arange(98304))
remaining = cell_order[48000:98304].copy()
assert len(remaining) == STEPS * GENERATED and not set(remaining) & set(cell_order[:48000])
rng = np.random.default_rng(APPEARANCE_SEED)
count = len(remaining)
schedule = {"cell_index": remaining, "sample_seed": APPEARANCE_SEED * 1_000_000 + np.arange(count, dtype=np.int64),
    "mode_index": rng.permutation(np.repeat(np.arange(3), count // 3)),
    "horizontal_flip": rng.integers(0, 2, count).astype(bool), "thickness_um": rng.uniform(25, 100, count).astype(np.float32),
    "gain": rng.uniform(.6, 1.4, count).astype(np.float32), "gamma": np.exp(rng.uniform(np.log(.6), np.log(1.6), count)).astype(np.float32),
    "invert_tissue": rng.integers(0, 2, count).astype(bool), "noise_std": rng.uniform(.005, .05, count).astype(np.float32),
    "background_mean": rng.uniform(0, .8, count).astype(np.float32), "background_slope_yx": rng.uniform(-.15, .15, (count, 2)).astype(np.float32),
    "mask_dilate": rng.integers(0, 2, count).astype(bool), "mask_radius_px": rng.integers(1, 4, count)}
for field in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    schedule[field] = np.array([f"fullcoverage001-training-atlas-organizational-{field}-{i:07d}" for i in range(count)])
schedule["atlas_anatomy_source_id"] = np.array("single_allen_atlas_binding_" + hashlib.sha256(json.dumps(previous["atlas_binding"], sort_keys=True).encode()).hexdigest())
schedule["biological_animal_id_is_null"] = np.array(True)
schedule["organizational_identifiers_are_biological"] = np.array(False)
frozen_rng = np.random.default_rng(FROZEN_SEED)
frozen_schedule = np.concatenate([frozen_rng.permutation(len(train["label"])) for _ in range((STEPS * 4 + len(train["label"]) - 1) // len(train["label"]))])[:STEPS * 4].reshape(STEPS, 4)
for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    assert not ({row[key] for row in train["records"]} | set(schedule[key])) & {row[key] for row in dev["records"]}
donor_rng = row_rng = np.random.default_rng(DONOR_SEED)
donor_schedule = np.concatenate([donor_rng.permutation(donor_ids) for _ in range((STEPS * REAL_BATCH + len(donor_ids) - 1) // len(donor_ids))])[:STEPS * REAL_BATCH].reshape(STEPS, REAL_BATCH)
real_row_schedule = np.empty_like(donor_schedule)
for donor in donor_ids:
    mask = donor_schedule == donor
    real_row_schedule[mask] = row_rng.choice(np.flatnonzero(donors == donor), size=int(mask.sum()), replace=True)
negative_rng = np.random.default_rng(NEGATIVE_SEED)
global_schedule = negative_rng.integers(0, 98304, (STEPS, 16), dtype=np.int64)
conditional_draw = np.random.default_rng(CONDITIONAL_SEED).random((STEPS, REAL_BATCH + SYNTHETIC_BATCH))
local_rank_schedule = negative_rng.integers(0, 8, (STEPS, SYNTHETIC_BATCH), dtype=np.int64)
chart_schedule = np.random.default_rng(CHART_SEED).permutation(np.repeat(np.array([0, 1], dtype=np.int64), STEPS * REAL_BATCH // 2)).reshape(STEPS, REAL_BATCH)
affines = torch.tensor([row["model_pixel_to_ap_dv_ml_um"] for row in real_records], dtype=torch.float64)
real_ouv = torch.stack((affines[:, :, 2], 96 * affines[:, :, 0], 96 * affines[:, :, 1]), dim=1)
real_states = full_frame_state_from_components(*physical_ouv_to_frame(real_ouv))
assert torch.allclose(full_frame_state_to_physical_ouv(real_states).reshape(-1, 3, 3), real_ouv, rtol=0, atol=1e-8)
real_physical_center, real_frame, _ = full_frame_state_to_components(real_states)
support_origin = torch.as_tensor(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"], dtype=torch.float64)
real_plane_normal = real_frame[:, :, 2]
prior_scatter = torch.stack([(real_plane_normal[donors == donor, :, None] * real_plane_normal[donors == donor, None, :]).mean(0) for donor in donor_ids]).mean(0)
prior_values, prior_vectors = torch.linalg.eigh(prior_scatter)
prior_normal = prior_vectors[:, -1]
prior_normal *= torch.where(prior_normal[0] < 0, -1., 1.)
canonical_center = support_origin + real_plane_normal * ((real_physical_center - support_origin) * real_plane_normal).sum(-1, keepdim=True)
canonical_states = full_frame_state_from_components(canonical_center, real_frame, torch.eye(2, dtype=torch.float64).expand(len(real_states), -1, -1) * 12000.)
canonical_c, canonical_frame, _ = full_frame_state_to_components(canonical_states)
assert torch.allclose(canonical_frame, real_frame, rtol=0, atol=1e-12) and torch.allclose(((canonical_c - support_origin) * canonical_frame[:, :, 2]).sum(-1), ((real_physical_center - support_origin) * real_plane_normal).sum(-1), rtol=0, atol=1e-8)
canonical_ouv = full_frame_state_to_physical_ouv(canonical_states).reshape(-1, 3, 3)
real_dev_affines = torch.tensor([row["model_pixel_to_ap_dv_ml_um"] for row in real_dev_records], dtype=torch.float64)
real_dev_ouv = torch.stack((real_dev_affines[:, :, 2], 96 * real_dev_affines[:, :, 0], 96 * real_dev_affines[:, :, 1]), dim=1)

RUN.mkdir(parents=True, exist_ok=False)
source_names = sorted(set(previous["source"]["file_sha256"]) | {Path(__file__).relative_to(repository).as_posix()})
source = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(), "file_sha256": {}}
for name in source_names:
    target = RUN / "source" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository / name, target)
    source["file_sha256"][name] = sha(target)
    if not name.startswith("training/run_") and name not in OPTIONAL_SOURCE_CHANGES:
        assert source["file_sha256"][name] == previous["source"]["file_sha256"][name], name
np.savez(RUN / "generated_schedule.npz", **schedule)
np.savez(RUN / "sampling_schedule.npz", donor_id=donor_schedule, real_row_index=real_row_schedule, frozen_row_index=frozen_schedule, global_cell_index=global_schedule, local_pool_rank=local_rank_schedule, canonical_chart_index=chart_schedule, conditional_uniform_fraction=conditional_draw)
np.savez(RUN / "weak_affine_anchors.npz", physical_ouv_ap_dv_ml_um=real_ouv.numpy(), full_frame_state=real_states.numpy(), nominal_section_thickness_um=[row["section_thickness_um"] for row in real_records], canonical_full_frame_state=canonical_states.numpy(), canonical_physical_ouv_ap_dv_ml_um=canonical_ouv.numpy())
for name, records in (("synthetic_training", train["records"]), ("synthetic_development", dev["records"]), ("real_training", real_records), ("real_development", real_dev_records)):
    (RUN / f"{name}_identities.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
for original_path, name in ((PREPARED / "catalogue.pt", "catalogue.pt"), (PARENT_AUDIT, "parent_independent_audit.json"), (PARENT_COMPLETION, "parent_completion.json"), (REFERENCE_SYNTHETIC, "reference_A6000_synthetic_metrics.json"), (REFERENCE_REAL, "reference_A6000_real_metrics.json"), (PARENT_REAL_METRICS, "parent_B8000_real_metrics.json"), (REAL_DEV / "raw_model_input.npy", "real_development_model_input.npy")):
    shutil.copyfile(original_path, RUN / name)
prior_receipt = {"rule": "equal-donor mean of within-donor normal outer products; leading eigenvector; all1280 TRAIN rows, no eligibility filter", "scatter": prior_scatter.tolist(), "eigenvalues": prior_values.tolist(), "normal_ap_dv_ml": prior_normal.tolist(), "donor_ids": donors.tolist(), "section_ids": [r["section_id"] for r in real_records], "geometry_sha256": {str(REAL_TRAIN / "image_geometry.jsonl"): bindings[str(REAL_TRAIN / "image_geometry.jsonl")], str(REAL_EXPANSION / "image_geometry.jsonl"): bindings[str(REAL_EXPANSION / "image_geometry.jsonl")]}, "scope": "readout-only antipodal scatter prior, no calibrated plane/offset or mean-angular-optimality claim"}
(RUN / "train_orientation_prior.json").write_text(json.dumps(prior_receipt, indent=2), encoding="utf-8")
schedule_sha256 = {name: sha(RUN / name) for name in ("generated_schedule.npz", "sampling_schedule.npz", "weak_affine_anchors.npz", "train_orientation_prior.json")}
config = {"source": source, "parent_kind": "canonical_rehearsal_C10000_experimental", "parent_checkpoint": str(PARENT_CHECKPOINT), "parent_sha256": PARENT_SHA256, "parent_audit_sha256": PARENT_AUDIT_SHA256, "parent_completion_sha256": PARENT_COMPLETION_SHA256, "reference_A6000_sha256": REFERENCE_A6000_SHA256, "parent_step": START, "additional_applied_updates": STEPS, "final_step": FINAL, "input_sha256": bindings, "schedule_sha256": schedule_sha256,
    "model_kwargs": previous["model_kwargs"], "trained_modules": TRAINABLE, "initialization": "strict whole experimental C10000 plus deep-copied AdamW and parent RNG restored; no LR change or encoder merge", "optimizer": "AdamW", "learning_rate": .00025, "weight_decay": 1e-4, "gradient_clip": 5., "precision": "FP32; AMP/TF32 disabled",
    "real_batch": REAL_BATCH, "synthetic_batch": SYNTHETIC_BATCH, "generated_per_batch": GENERATED, "appearance_seed": APPEARANCE_SEED, "frozen_row_seed": FROZEN_SEED, "donor_and_real_row_seed": DONOR_SEED, "conditional_negative_seed": CONDITIONAL_SEED, "negative_seed": NEGATIVE_SEED, "chart_seed": CHART_SEED, "real_sampling": "shuffled complete donor permutations then uniform row within donor with replacement", "synthetic_cell_permutation_slice": [48000, 98304], "fresh_appearance_draws": True, "synthetic_frozen_sampling": "complete shuffled TRAIN row cycles",
    "synthetic_loss": previous["synthetic_loss"], "synthetic_candidate_sampling": "8synthetic truths+8geometric local keys+16globaluniform+16query-specific wrong-plane keys, pooled/deduplicated; owner-only plane admissibility, unchanged full-frame loss masks; A uses uniform admissible, B snapshot-hard; zero-weight slots use shared uniform fallback", "real_loss": "eligible-weighted paired sampled NCE;8continuous selected-chart anchors plus same synthetic keys; equal raster prior, no cell prior; eligibility requires BOTH matched/canonical finite supports>=64; invalid real keys never negatives; own positive retained with zero weight when invalid", "total_loss": "(2/3)*eligible-weighted synthetic sampled NLL + (1/3)*eligible-weighted real paired NCE; zero real contribution for empty eligible batch",
    "synthetic_near_key_exclusion": "unchanged normal<=10deg AND finite four-corner RMS<=1000um, min identity/horizontal correspondence;95/96 finite edges", "real_near_key_exclusion": "physical plane normal<=10deg AND sign-aligned normal offset<=500um; antipodal; no support-invalid real negative; force own positive", "chart_schedule": "exactly50304 matched+50304 canonical choices from fresh frozen permutation, before common eligibility weighting", "canonical_chart": "preserve proper frame and roll; c0=S+n*dot(n,c-S),12mm diagonal basis,no shear; continuous plane,not nearest catalogue cell", "anchor_cache": "fixed atlas intensity/support images only; no cached learned features; two charts rendered once at50um9pointPSF", "real_reference_role": "weak upstream-affine pairing only; no nearest-cell onehot, dense correspondence, ribbon, deformation or uncertainty truth", "real_anchor_psf": "fixed50um9-point normalized boxcar-trapezoid; unverified engineering rendering assumption, not measured opticalPSF; nominal section_thickness provenance only", "real_pixels": "frozen arrays unchanged, no mask, extra photometry or outline; geometry/IDs never enter query encoder", "synthetic_outline_dropout": False,
    "temperature": TEMPERATURE, "gallery": previous["gallery"], "atlas_binding": previous["atlas_binding"], "catalogue_receipt_sha256": catalogue["receipt_sha256"], "prepared_source_directory": str(PREPARED), "prepared_source_sha256": previous["prepared_source_sha256"], "replay_directory": str(REPLAY), "replay_sha256": previous["replay_sha256"],
    "evaluation": "fresh full98304x2 endpoint gallery; all1280realTRAIN and5120frozenTRAIN streaming top128/metrics, unchanged640syntheticDEV+64realDEV raw scores; TRAIN scatter orientation prior; no intermediate DEV", "promotion_gate": {"real_donor_macro_normal_improvement_deg_min": 5., "real_top32_plane_capture_improvement_min": .10, "synthetic_normal_regression_deg_max": 2., "synthetic_top32_plane_capture_regression_max": .02, "real_retention_reference": "experimental B8000; angle<=+2deg,capture32>=-.02", "synthetic_retention_and_real_improvement_reference": "original A6000", "synthetic_subsets": "eligible overall and every populated eligible original mode"}, "scope": "preregistered uniform-first: B only if completedA fails fixed gate; bothstartwholeC, identical12576update data schedules; hard-minus-uniform is negative-selection policy including duplicate-count effects; no equal walltime claim, no native/calibration/publicbenchmark qualification", "probabilities_calibrated": False, "optional_source_changes": sorted(OPTIONAL_SOURCE_CHANGES), "optional_source_change_scope": "signed-pose option disabled; unchanged whole parent parameters and old default inference; source hashes captured explicitly"}
config["sequential_policy"] = "RunAuniform first; only if fixedgatefails runBsnapshot-hard fromsameC withidenticalbudget/data"
config["synthetic_anatomy"] = {"atlas_anatomy_source_id": str(schedule["atlas_anatomy_source_id"]), "independent_atlas_anatomy_sources": 1, "biological_animal_id": None, "organizational_ids": "engineering draw/group IDs only; disjointness is not independent biological-animal validation"}
config["mining"] = {"refresh_interval": REFRESH_INTERVAL, "training_refreshes_B": 13, "A_training_refreshes": 0, "maximum_snapshot_age_updates": 1023, "score": "cosine/0.1 plus representation prior, logsumexpR then cellmass once, for both querytypes; realNCE remainsno-cellmass", "features": "same frozen own-lineage encoder for miningquery and bank; cached values selectindices only; fresh currentencoder computes every losskey/query", "owner_admissibility": "reject <=10deg AND<=500umphysicalplane; otherqueryloss masks unchanged", "duplicates": "no refill; recordeffectivecandidatecount"}
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
started = time.perf_counter()
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"], device="cuda", dtype=torch.float32)
assert runtime.cell_count == 98304 and runtime.representation_count == 2
cell_states = torch.as_tensor(catalogue["arrays"]["cell_states_float64"])
cell_center, cell_frame, _ = full_frame_state_to_components(cell_states)
cell_normal = cell_frame[:, :, 2]
cell_ouv = full_frame_state_to_physical_ouv(cell_states).reshape(-1, 3, 3)
key_u, key_v = cell_ouv[:, 1] * (95 / 96), cell_ouv[:, 2] * (95 / 96)
key_center = cell_ouv[:, 0] + .5 * (key_u + key_v)
key_center, key_u, key_v, key_normal = [value.cuda().float() for value in (key_center, key_u, key_v, cell_normal)]
real_normal = real_plane_normal.cuda().float()
real_plane_offset = ((real_physical_center - support_origin) * real_plane_normal).sum(-1).cuda().float()
cell_plane_offset = ((cell_center - support_origin) * cell_normal).sum(-1).cuda().float()
render_states = cell_states.cuda().float()
train_center, train_frame, _ = full_frame_state_to_components(train["truth_state"].double())
train_normal = train_frame[:, :, 2]
train_offset = ((train_center - support_origin) * train_normal).sum(-1)
train_normal_gpu, train_offset_gpu = train_normal.cuda().float(), train_offset.cuda().float()
cell_log_mass = catalogue["tensors"]["cell_log_mass"][0].cuda().float()
representation_log_weight = catalogue["tensors"]["representation_log_weight"][0].cuda().float()
assert np.array_equal(np.asarray(catalogue["arrays"]["representation_to_canonical_raster_affine_float64"])[0], np.array([[[1., 0., 0.], [0., 1., 0.]], [[-1., 0., 0.], [0., 1., 0.]]]))
psf_weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device="cuda", dtype=torch.float32) / 16
psf_positions = torch.linspace(-.5, .5, 9, device="cuda")
yy, xx = torch.meshgrid(torch.linspace(-1, 1, 96, device="cuda"), torch.linspace(-1, 1, 96, device="cuda"), indexing="ij")
noise_generator = torch.Generator(device="cuda")
origin, spacing = [catalogue["support_geometry"][key] for key in ("origin_ap_dv_ml_um", "voxel_size_ap_dv_ml_um")]
support_origin = torch.as_tensor(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"], dtype=torch.float64)
real_channels = torch.cat((torch.from_numpy(real_images), torch.zeros(N_REAL, 2, 96, 96)), dim=1)
real_dev_channels = torch.cat((torch.from_numpy(dev_images), torch.zeros(64, 2, 96, 96)), dim=1)
with torch.no_grad():
    anchor_cache = torch.stack([torch.cat([render_finite_thickness_plane(atlas, part.cuda().float(), SHAPE, origin, spacing, psf_positions * KEY_THICKNESS_UM, psf_weights) for part in states.split(16)]) for states in (real_states, canonical_states)])
assert bool(torch.isfinite(anchor_cache).all())
anchor_support = anchor_cache[:, :, 1].clamp(0, 1).sum((-2, -1))
real_eligible = (anchor_support >= SUPPORT_MASS_THRESHOLD).all(0)
np.save(RUN / "real_anchor_image_cache.npy", anchor_cache.cpu().numpy())
np.savez(RUN / "real_anchor_eligibility.npz", matched_support_mass=anchor_support[0].cpu().numpy(), canonical_support_mass=anchor_support[1].cpu().numpy(), common_eligible=real_eligible.cpu().numpy())
config["anchor_cache_sha256"] = {name: sha(RUN / name) for name in ("real_anchor_image_cache.npy", "real_anchor_eligibility.npz")}
config["real_eligibility_counts"] = {"all": N_REAL, "eligible": int(real_eligible.sum()), "ineligible": int((~real_eligible).sum())}
(RUN / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
print(json.dumps({"anchor_cache_images": 2 * N_REAL, "common_real_eligibility": config["real_eligibility_counts"]}), flush=True)


def encode_keys(images):
    return model.pose_model.descriptor_from_features(model.pose_model._encode_atlas(images))


arm_results = {}
for arm in ("A", "B"):
    output = RUN / arm
    output.mkdir()
    model = ArbitraryPlaneJointModelV6(runtime, **previous["model_kwargs"]).cuda()
    model.load_state_dict(parent["model_state"], strict=True)
    assert all(torch.equal(value.cpu(), parent["model_state"][name]) for name, value in model.state_dict().items())
    assert not model.pose_model.signed_pose_evidence_enabled
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=.00025, weight_decay=1e-4)
    optimizer.load_state_dict(copy.deepcopy(parent["optimizer_state"]))
    loaded = optimizer.state_dict()
    assert loaded["param_groups"] == parent["optimizer_state"]["param_groups"] and all(group["lr"] == .00025 and group["weight_decay"] == 1e-4 for group in loaded["param_groups"])
    for key, state in parent["optimizer_state"]["state"].items():
        assert all(torch.equal(loaded["state"][key][name].cpu(), value) if torch.is_tensor(value) else loaded["state"][key][name] == value for name, value in state.items())
    del loaded
    random.setstate(parent["python_rng"]); np.random.set_state(parent["numpy_rng"])
    torch.set_rng_state(parent["torch_rng"]); torch.cuda.set_rng_state_all(parent["cuda_rng"])
    (output / "initialization.json").write_text(json.dumps({"arm": arm, "parent_sha256": PARENT_SHA256, "model_tensors_exact": True, "optimizer_state_exact": True, "learning_rate_unchanged": .00025, "betas_and_weight_decay_unchanged": True, "restored_parent_rng": True, "initial_gallery_rebuilt": False}), encoding="utf-8")
    applied = 0
    arm_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    eligible_generated_count = 0
    stage_positive_cells = torch.zeros(runtime.cell_count, device="cuda", dtype=torch.bool)
    with (output / "training_trace.jsonl").open("w", encoding="utf-8") as trace:
        for step in range(STEPS):
            if step % REFRESH_INTERVAL == 0:
                snapshot_path = output / f"joint_model_step_{START + applied:05d}.pt"
                torch.save({"experiment": config, "arm": arm, "phase": "experimental_image_key_proposal_only", "step": START + applied, "optimizer_steps_applied": START + applied, "additional_optimizer_steps_applied": applied, "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()}, "optimizer_state": optimizer.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "probabilities_calibrated": False}, snapshot_path)
                if arm == "B":
                    snapshot = copy.deepcopy(model, {id(runtime): runtime}).eval()
                    snapshot.requires_grad_(False)
                    snapshot_step, snapshot_sha256 = applied, sha(snapshot_path)
                    snapshot_bank = torch.empty((runtime.cell_count, 2, DESCRIPTOR_DIM), device="cuda", dtype=torch.float16)
                    with torch.no_grad():
                        for start in range(0, runtime.cell_count, 64):
                            images = render_finite_thickness_plane(atlas, render_states[start:start + 64], SHAPE, origin, spacing, psf_positions * KEY_THICKNESS_UM, psf_weights)
                            images = torch.stack((images, images.flip(-1)), dim=1).flatten(0, 1)
                            snapshot_bank[start:start + 64] = torch.cat([snapshot.pose_model.descriptor_from_features(snapshot.pose_model._encode_atlas(part)) for part in images.split(KEY_CHUNK)]).reshape(-1, 2, DESCRIPTOR_DIM).half()
                    assert bool(torch.isfinite(snapshot_bank).all())
                    print(json.dumps({"arm": arm, "snapshot_refresh_step": START + snapshot_step, "checkpoint_sha256": snapshot_sha256}), flush=True)
            model.train()
            positions = np.arange(step * GENERATED, (step + 1) * GENERATED)
            with torch.no_grad():
                generated_labels = torch.as_tensor(schedule["cell_index"][positions], device="cuda")
                thickness = torch.as_tensor(schedule["thickness_um"][positions], device="cuda")
                rendered = render_finite_thickness_plane(atlas, render_states[generated_labels], SHAPE, origin, spacing, thickness[:, None] * psf_positions[None], psf_weights)
                finite_support = rendered[:, 1].clamp(0, 1); tissue = finite_support > 0
                generated_inputs = torch.empty((GENERATED, 3, 96, 96), device="cuda"); visible_mass = torch.empty(GENERATED, device="cuda")
                for local, position in enumerate(positions):
                    noise_generator.manual_seed(int(schedule["sample_seed"][position]))
                    noise = torch.randn((96, 96), device="cuda", generator=noise_generator)
                    appearance = (rendered[local, 0] / finite_support[local].clamp_min(1e-6)).clamp(0, 1).pow(float(schedule["gamma"][position]))
                    if schedule["invert_tissue"][position]:
                        appearance = 1.0 - appearance
                    appearance = (appearance * float(schedule["gain"][position])).clamp(0, 1)
                    slope = schedule["background_slope_yx"][position]
                    background = float(schedule["background_mean"][position]) + float(slope[0]) * yy + float(slope[1]) * xx
                    image = (appearance * finite_support[local] + background * (1.0 - finite_support[local]) + float(schedule["noise_std"][position]) * noise).clamp(0, 1)
                    mode = int(schedule["mode_index"][position]); mask = tissue[local].clone()
                    if mode == 2:
                        radius = int(schedule["mask_radius_px"][position])
                        if schedule["mask_dilate"][position]:
                            mask = F.max_pool2d(mask[None, None].float(), 2 * radius + 1, 1, radius)[0, 0] > 0
                        else:
                            padded = F.pad(mask[None, None].float(), (radius,) * 4, value=0)
                            mask = -F.max_pool2d(-padded, 2 * radius + 1, 1)[0, 0] > 0
                    visible_mass[local] = (finite_support[local] * mask).sum() if mode else finite_support[local].sum()
                    outline = torch.zeros_like(mask)
                    if mode:
                        image = image * mask
                        eroded = mask.clone(); eroded[1:] &= mask[:-1]; eroded[:-1] &= mask[1:]
                        eroded[:, 1:] &= mask[:, :-1]; eroded[:, :-1] &= mask[:, 1:]
                        eroded[[0, -1], :] = False; eroded[:, [0, -1]] = False
                        outline = mask & ~eroded
                    generated_inputs[local] = torch.stack((image, outline.float(), torch.full_like(image, float(mode != 0))))
                    if schedule["horizontal_flip"][position]:
                        generated_inputs[local] = generated_inputs[local].flip(-1)
                support_mass = finite_support.sum((-2, -1))
                generated_weight = ((support_mass >= SUPPORT_MASS_THRESHOLD) & (visible_mass >= SUPPORT_MASS_THRESHOLD)).float()
                frozen_index, real_index = frozen_schedule[step], real_row_schedule[step]
                real_index_gpu = torch.as_tensor(real_index, device="cuda")
                synthetic_inputs = torch.cat((generated_inputs, train["channels"][frozen_index].cuda()))
                label = torch.cat((generated_labels, train["label"][frozen_index].cuda()))
                weight = torch.cat((generated_weight, train["weight"][frozen_index].cuda()))
                inputs = torch.cat((real_channels[real_index].cuda(), synthetic_inputs))
                distance2 = (key_center[label, None] - key_center[None]).square().sum(-1)
                same_u = (key_u[label, None] - key_u[None]).square().sum(-1)
                flip_u = (key_u[label, None] + key_u[None]).square().sum(-1)
                distance2 += .25 * (torch.minimum(same_u, flip_u) + (key_v[label, None] - key_v[None]).square().sum(-1))
                ignore = ((key_normal[label] @ key_normal.T).abs() >= np.cos(np.deg2rad(10.))) & (distance2 <= 1000.**2)
                ignore.scatter_(1, label[:, None], True)
                local_pool = distance2.masked_fill(ignore, torch.inf).topk(8, largest=False, sorted=True).indices
                local_ids = local_pool[torch.arange(SYNTHETIC_BATCH, device="cuda"), torch.as_tensor(local_rank_schedule[step], device="cuda")]
                real_weight = real_eligible[real_index_gpu].float()
                owner_normal = torch.cat((real_normal[real_index_gpu], key_normal[generated_labels], train_normal_gpu[frozen_index]))
                owner_offset = torch.cat((real_plane_offset[real_index_gpu], cell_plane_offset[generated_labels], train_offset_gpu[frozen_index]))
                owner_weight = torch.cat((real_weight, weight))
                owner_dot = owner_normal @ key_normal.T
                owner_admissible = ~((owner_dot.abs() >= np.cos(np.deg2rad(10.))) & ((cell_plane_offset[None] - torch.where(owner_dot < 0, -1., 1.) * owner_offset[:, None]).abs() <= 500.))
                owner_admissible[REAL_BATCH + torch.arange(SYNTHETIC_BATCH, device="cuda"), label] = False
                if arm == "A":
                    ranks = torch.floor(torch.as_tensor(conditional_draw[step], device="cuda") * owner_admissible.sum(-1)).long()
                    conditional_ids = (owner_admissible.cumsum(-1) > ranks[:, None]).to(torch.uint8).argmax(-1)
                else:
                    mining_query = snapshot.pose_model.descriptor_from_features(snapshot.pose_model.encode_histology(inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1))))
                    mining_score = torch.logsumexp(snapshot.pose_model.image_key_cosine_logits(mining_query, snapshot_bank, TEMPERATURE) + representation_log_weight[None], dim=-1) + cell_log_mass[None]
                    assert bool(torch.isfinite(mining_score).all())
                    conditional_ids = mining_score.masked_fill(~owner_admissible, -torch.inf).argmax(-1)
                    del mining_query, mining_score
                conditional_ids = torch.where(owner_weight > 0, conditional_ids, torch.as_tensor(global_schedule[step], device="cuda"))
                assert bool(owner_admissible[torch.arange(16, device="cuda"), conditional_ids][owner_weight > 0].all())
                pooled = torch.cat((label, torch.as_tensor(global_schedule[step], device="cuda"), local_ids, conditional_ids))
                candidate = torch.unique(pooled, sorted=True)
                target = torch.searchsorted(candidate, label)
                keep = ~ignore[:, candidate]; keep.scatter_(1, target[:, None], True)
                near_plane_far_frame = ~owner_admissible[REAL_BATCH:, candidate] & keep
                near_plane_far_frame.scatter_(1, target[:, None], False)
                near_plane_far_frame_counts = near_plane_far_frame.sum(-1).cpu().tolist()
                del owner_dot, owner_admissible, near_plane_far_frame
                chart_index = torch.as_tensor(chart_schedule[step], device="cuda")
                real_weight = real_eligible[real_index_gpu].float()
                real_key_images = anchor_cache[chart_index, real_index_gpu]
                real_support_mass = anchor_support[chart_index, real_index_gpu]
                synthetic_key_images = render_finite_thickness_plane(atlas, render_states[candidate], SHAPE, origin, spacing, psf_positions * KEY_THICKNESS_UM, psf_weights)
                key_images = torch.cat((real_key_images, synthetic_key_images))
                key_images = torch.stack((key_images, key_images.flip(-1)), dim=1).flatten(0, 1)
                normals = torch.cat((real_normal[real_index_gpu], key_normal[candidate]))
                plane_offsets = torch.cat((real_plane_offset[real_index_gpu], cell_plane_offset[candidate]))
                real_dot = real_normal[real_index_gpu] @ normals.T
                offset_difference = (plane_offsets[None] - torch.where(real_dot < 0, -1., 1.) * real_plane_offset[real_index_gpu, None]).abs()
                real_keep = ~((real_dot.abs() >= np.cos(np.deg2rad(10.))) & (offset_difference <= 500.))
                real_keep[:, :REAL_BATCH] &= real_eligible[real_index_gpu][None]
                real_keep[torch.arange(REAL_BATCH, device="cuda"), torch.arange(REAL_BATCH, device="cuda")] = True
                del distance2, same_u, flip_u, ignore, local_pool, offset_difference
            optimizer.zero_grad(set_to_none=True)
            query = model.pose_model.descriptor_from_features(model.pose_model.encode_histology(inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1))))
            keys = torch.cat([checkpoint(encode_keys, part, use_reentrant=False) for part in key_images.split(KEY_CHUNK)]).reshape(REAL_BATCH + len(candidate), 2, DESCRIPTOR_DIM)
            synthetic_scores = model.pose_model.image_key_cosine_logits(query[REAL_BATCH:], keys[REAL_BATCH:], TEMPERATURE)
            synthetic_scores = torch.logsumexp(synthetic_scores + representation_log_weight[candidate][None], dim=-1) + cell_log_mass[candidate][None]
            synthetic_nll = torch.logsumexp(synthetic_scores.masked_fill(~keep, -torch.inf), dim=-1) - synthetic_scores.gather(1, target[:, None])[:, 0]
            synthetic_loss = (synthetic_nll * weight).sum() / weight.sum().clamp_min(1.)
            real_scores = torch.logsumexp(model.pose_model.image_key_cosine_logits(query[:REAL_BATCH], keys, TEMPERATURE) - np.log(2.), dim=-1)
            real_nce = torch.logsumexp(real_scores.masked_fill(~real_keep, -torch.inf), dim=-1) - real_scores[:, :REAL_BATCH].diagonal()
            real_loss = (real_nce * real_weight).sum() / real_weight.sum().clamp_min(1.)
            chart_counts = [int(((chart_index == chart) & (real_weight > 0)).sum()) for chart in (0, 1)]
            chart_losses = [float(real_nce[(chart_index == chart) & (real_weight > 0)].detach().mean()) if chart_counts[chart] else None for chart in (0, 1)]
            loss = (2. / 3.) * synthetic_loss + (1. / 3.) * real_loss
            assert bool(torch.isfinite(loss))
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
            optimizer.step(); applied += 1
            eligible_generated_count += int(generated_weight.sum())
            stage_positive_cells[label[weight > 0]] = True
            record = {"arm": arm, "step": START + applied, "real_chart_index": chart_index.cpu().tolist(), "real_common_eligible": real_weight.cpu().tolist(), "real_supervision_mass": float(real_weight.sum()), "real_matched_nce": chart_losses[0], "real_canonical_nce": chart_losses[1], "real_matched_count": chart_counts[0], "real_canonical_count": chart_counts[1], "additional_optimizer_steps_applied": applied, "synthetic_nll": float(synthetic_loss.detach()), "real_paired_nce": float(real_loss.detach()), "loss": float(loss.detach()), "gradient_norm": float(gradient), "real_row_index": real_index.tolist(), "real_donor_id": donor_schedule[step].tolist(), "generated_indices": positions.tolist(), "original_permutation_indices": (positions + 48000).tolist(), "frozen_row_index": frozen_index.tolist(), "candidate_cell_index": candidate.cpu().tolist(), "candidate_cell_count": len(candidate), "pooled_duplicate_count": len(pooled) - len(candidate), "conditional_cell_index": conditional_ids.cpu().tolist(), "conditional_owner_weight": owner_weight.cpu().tolist(), "snapshot_step": START + snapshot_step if arm == "B" else None, "snapshot_age_updates": applied - 1 - snapshot_step if arm == "B" else None, "snapshot_checkpoint_sha256": snapshot_sha256 if arm == "B" else None, "synthetic_near_plane_far_frame_negative_counts": near_plane_far_frame_counts, "local_negative_cell_index": local_ids.cpu().tolist(), "synthetic_ignored_counts": (~keep).sum(-1).cpu().tolist(), "real_ignored_counts": (~real_keep).sum(-1).cpu().tolist(), "synthetic_supervision_mass": float(weight.sum()), "generated_point_pose_weight": generated_weight.cpu().tolist(), "finite_support_mass_px": support_mass.cpu().tolist(), "visible_support_mass_px": visible_mass.cpu().tolist(), "real_anchor_finite_support_mass_px": real_support_mass.cpu().tolist(), "fresh_unique_generated_cells_attempted": applied * GENERATED, "fresh_unique_generated_cells_eligible": eligible_generated_count, "stage_unique_eligible_positive_cells": int(stage_positive_cells.sum()), "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, "arm_elapsed_seconds": time.perf_counter() - arm_started, "elapsed_seconds": time.perf_counter() - started}
            trace.write(json.dumps(record) + "\n")
            if applied % 100 == 0:
                trace.flush(); print(json.dumps({key: record[key] for key in ("arm", "step", "loss", "synthetic_nll", "real_paired_nce", "real_matched_nce", "real_canonical_nce", "real_matched_count", "real_canonical_count", "gradient_norm", "fresh_unique_generated_cells_attempted", "fresh_unique_generated_cells_eligible", "stage_unique_eligible_positive_cells", "gpu_peak_allocated_mib", "gpu_peak_reserved_mib", "arm_elapsed_seconds", "elapsed_seconds")}), flush=True)
            del keys, query, loss, real_loss, synthetic_loss, real_nce, synthetic_nll, real_scores, synthetic_scores, inputs, synthetic_inputs, key_images, real_key_images, synthetic_key_images, generated_inputs, rendered
    assert applied == STEPS and all(int(state["step"]) == FINAL for state in optimizer.state_dict()["state"].values())
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    assert all(torch.equal(value.cpu(), parent["model_state"][name]) for name, value in model.state_dict().items() if not name.startswith(TRAINABLE))
    model.eval()
    if arm == "B":
        del snapshot, snapshot_bank
    checkpoint_path = output / f"joint_model_step_{FINAL:05d}.pt"
    torch.save({"experiment": config, "arm": arm, "phase": "experimental_image_key_proposal_only", "step": FINAL, "optimizer_steps_applied": FINAL, "additional_optimizer_steps_applied": applied, "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()}, "optimizer_state": optimizer.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "probabilities_calibrated": False}, checkpoint_path)
    checkpoint_sha256 = sha(checkpoint_path)
    bank = torch.empty((runtime.cell_count, 2, DESCRIPTOR_DIM), device="cuda", dtype=torch.float16)
    with torch.no_grad():
        for start in range(0, runtime.cell_count, 64):
            images = render_finite_thickness_plane(atlas, render_states[start:start + 64], SHAPE, origin, spacing, psf_positions * KEY_THICKNESS_UM, psf_weights)
            images = torch.stack((images, images.flip(-1)), dim=1).flatten(0, 1)
            bank[start:start + 64] = torch.cat([encode_keys(part) for part in images.split(KEY_CHUNK)]).reshape(-1, 2, DESCRIPTOR_DIM).half()
            if (start + 64) % 16384 == 0:
                print(json.dumps({"gallery_cells": start + 64}), flush=True)
    bank_path = output / f"gallery_descriptors_step_{FINAL:05d}.npy"
    np.save(bank_path, bank.cpu().numpy()); bank_sha256 = sha(bank_path)
    dev_center, dev_frame, _ = full_frame_state_to_components(dev["truth_state"].double())
    real_dev_center = torch.tensor([row["truth_center_ap_dv_ml_um"] for row in real_dev_records], dtype=torch.float64)
    real_dev_normal = torch.tensor([row["truth_normal_ap_dv_ml"] for row in real_dev_records], dtype=torch.float64)
    results = {}
    for dataset, channels, records, truth_center, truth_normal in (("allen_train", real_channels, real_records, real_physical_center, real_plane_normal), ("synthetic_train", train["channels"], train["records"], train_center, train_normal), ("synthetic", dev["channels"], dev["records"], dev_center, dev_frame[:, :, 2]), ("allen_raw", real_dev_channels, real_dev_records, real_dev_center, real_dev_normal)):
        directory = output / dataset; directory.mkdir()
        save_full = dataset in ("synthetic", "allen_raw")
        if save_full:
            raw = np.lib.format.open_memmap(directory / "raw_cell_log_probability.npy", mode="w+", dtype=np.float32, shape=(len(records), runtime.cell_count))
            components = np.lib.format.open_memmap(directory / "raw_component_log_score.npy", mode="w+", dtype=np.float32, shape=(len(records), runtime.cell_count, 2))
        prepared = train if dataset == "synthetic_train" else dev
        query_bank, tops, top_rep, top_lp, synthetic_rows = [], [], [], [], []
        normalization_error = 0.
        with torch.no_grad():
            for start in range(0, len(records), 16):
                inputs = channels[start:start + 16].cuda()
                query = model.pose_model.descriptor_from_features(model.pose_model.encode_histology(inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1))))
                representation_score = model.pose_model.image_key_cosine_logits(query, bank, TEMPERATURE) + representation_log_weight[None]
                log_probability = (torch.logsumexp(representation_score, dim=-1) + cell_log_mass[None]).log_softmax(dim=-1)
                assert bool(torch.isfinite(representation_score).all() and torch.isfinite(log_probability).all())
                lp_cpu = log_probability.cpu().numpy()
                normalization_error = max(normalization_error, float(np.abs(np.logaddexp.reduce(lp_cpu.astype(np.float64), axis=1)).max()))
                if save_full:
                    raw[start:start + len(inputs)] = lp_cpu
                    components[start:start + len(inputs)] = (representation_score + cell_log_mass[None, :, None]).cpu().numpy()
                if dataset.startswith("synthetic"):
                    labels_batch = prepared["label"][start:start + len(inputs)].numpy()
                    truth_lp = lp_cpu[np.arange(len(inputs)), labels_batch]
                    ranks = 1 + (lp_cpu > truth_lp[:, None]).sum(-1) + ((lp_cpu == truth_lp[:, None]) & (np.arange(runtime.cell_count)[None] < labels_batch[:, None])).sum(-1)
                    normal_lp = np.logaddexp.reduce(lp_cpu.reshape(len(inputs), 384, 256), axis=-1)
                    synthetic_rows.append(np.stack((-truth_lp, ranks, -normal_lp[np.arange(len(inputs)), labels_batch // 256], normal_lp.argmax(-1)), axis=1))
                top = torch.argsort(log_probability, dim=-1, descending=True, stable=True)[:, :128]
                query_bank.append(query.cpu().numpy()); tops.append(top.cpu()); top_lp.append(log_probability.gather(1, top).cpu().numpy())
                top_rep.append(representation_score[torch.arange(len(inputs), device="cuda")[:, None], top].softmax(-1).cpu().numpy())
        if save_full:
            raw.flush(); components.flush()
        assert normalization_error < 2e-5
        top = torch.cat(tops); normals = cell_normal[top]
        dot = (normals * truth_normal[:, None]).sum(-1)
        angles = torch.rad2deg(torch.atan2(torch.linalg.cross(normals, truth_normal[:, None]).norm(dim=-1), dot.abs()))
        offsets = (((cell_center[top] - support_origin) * normals).sum(-1) - torch.where(dot < 0, -1., 1.) * ((truth_center - support_origin) * truth_normal).sum(-1)[:, None]).abs()
        metrics = {"plane_angle_deg": angles[:, 0].numpy(), "normal_offset_error_um": offsets[:, 0].numpy(), **{f"physical_plane_capture_at_{k}": ((angles[:, :k] <= 10.) & (offsets[:, :k] <= 500.)).any(-1).double().numpy() for k in (32, 128)}}
        if dataset in ("allen_train", "allen_raw"):
            prior_angle = torch.rad2deg(torch.atan2(torch.linalg.cross(truth_normal, prior_normal[None]).norm(dim=-1), (truth_normal * prior_normal).sum(-1).abs())).numpy()
            metrics["train_orientation_prior_angle_deg"] = prior_angle
            metrics["model_minus_train_orientation_prior_angle_deg"] = metrics["plane_angle_deg"] - prior_angle
            reference_ouv = real_ouv if dataset == "allen_train" else real_dev_ouv
            reference_u, reference_v = reference_ouv[:, 1] * (95 / 96), reference_ouv[:, 2] * (95 / 96)
            reference_finite_center = reference_ouv[:, 0] + .5 * (reference_u + reference_v)
            predicted_ouv = cell_ouv[top[:, 0]]
            predicted_u, predicted_v = predicted_ouv[:, 1] * (95 / 96), predicted_ouv[:, 2] * (95 / 96)
            predicted_finite_center = predicted_ouv[:, 0] + .5 * (predicted_u + predicted_v)
            metrics["map_finite_frame_rms_um"] = ((predicted_finite_center - reference_finite_center).square().sum(-1) + .25 * (torch.minimum((predicted_u - reference_u).square().sum(-1), (predicted_u + reference_u).square().sum(-1)) + (predicted_v - reference_v).square().sum(-1))).sqrt().numpy()
        group = np.array([row["animal_id"] for row in records]); subsets = {"all": np.ones(len(records), dtype=bool)}
        row_arrays = {key: np.array([row[key] for row in records]) for key in ("animal_id", "specimen_id", "experiment_id", "section_id")}
        if dataset == "allen_train":
            common_eligible = real_eligible.cpu().numpy()
            subsets.update(eligible=common_eligible, ineligible=~common_eligible)
            row_arrays.update(common_anchor_eligible=common_eligible, matched_support_mass=anchor_support[0].cpu().numpy(), canonical_support_mass=anchor_support[1].cpu().numpy())
            with torch.no_grad():
                all_query = torch.from_numpy(np.concatenate(query_bank)).cuda()
                exact_keys = torch.stack([torch.cat([encode_keys(torch.stack((part, part.flip(-1)), dim=1).flatten(0, 1)) for part in cache.split(8)]).reshape(N_REAL, 2, DESCRIPTOR_DIM) for cache in anchor_cache])
                anchor_components = torch.stack([model.pose_model.image_key_cosine_logits(all_query, chart_keys, TEMPERATURE) for chart_keys in exact_keys])
                assert bool(torch.isfinite(anchor_components).all())
                anchor_scores = torch.logsumexp(anchor_components - np.log(2.), dim=-1).cpu().numpy()
            anchor_dot = real_plane_normal @ real_plane_normal.T
            reference_offsets = ((real_physical_center - support_origin) * real_plane_normal).sum(-1)
            anchor_near = ((anchor_dot.abs() >= np.cos(np.deg2rad(10.))) & ((reference_offsets[None] - torch.where(anchor_dot < 0, -1., 1.) * reference_offsets[:, None]).abs() <= 500.)).numpy()
            for chart, name in enumerate(("matched", "canonical")):
                scores = anchor_scores[chart]
                own = scores[np.arange(N_REAL), np.arange(N_REAL)]
                ranks = 1 + (scores > own[:, None]).sum(-1) + ((scores == own[:, None]) & (np.arange(N_REAL)[None] < np.arange(N_REAL)[:, None])).sum(-1)
                anchor_order = np.argsort(-scores, axis=-1, kind="stable")
                near_rank = 1 + np.argmax(np.take_along_axis(anchor_near, anchor_order, axis=1), axis=1)
                metrics[f"{name}_own_anchor_rank"] = ranks
                metrics[f"{name}_near_physical_anchor_rank"] = near_rank
                metrics.update({f"{name}_own_anchor_hit_at_{k}": (ranks <= k).astype(float) for k in (1, 8, 32)})
                metrics.update({f"{name}_near_physical_anchor_hit_at_{k}": (near_rank <= k).astype(float) for k in (1, 8, 32)})
            np.savez(directory / "exact_anchor_reconstruction.npz", query_descriptor=np.concatenate(query_bank), atlas_descriptor=exact_keys.cpu().numpy(), raw_component_logit=anchor_components.cpu().numpy(), uniform_representation_logit=anchor_scores, near_physical_mask=anchor_near)
            del all_query, exact_keys, anchor_components
        if dataset.startswith("synthetic"):
            labels = prepared["label"].numpy()
            values = np.concatenate(synthetic_rows)
            rank = values[:, 1]
            marginal_normal = cell_normal[values[:, 3].astype(np.int64) * 256]
            metrics.update({"nll": values[:, 0], "truth_rank": rank, "normal_marginal_nll": values[:, 2], "normal_marginal_map_angle_deg": torch.rad2deg(torch.atan2(torch.linalg.cross(marginal_normal, truth_normal).norm(dim=-1), (marginal_normal * truth_normal).sum(-1).abs())).numpy(), **{f"hit_at_{k}": (rank <= k).astype(float) for k in (1, 8, 32, 128)}})
            eligible = prepared["weight"].numpy() > 0; modes = np.array([row["selected_mode"] for row in records])
            subsets.update({"eligible": eligible, "censored": ~eligible, **{f"eligible_mode:{mode}": eligible & (modes == mode) for mode in np.unique(modes)}})
            row_arrays.update(label=labels, pose_supervision_weight=prepared["weight"].numpy(), selected_mode=modes)
        summaries = {}
        for name, selected in subsets.items():
            by_group = {str(identity): {"rows": int((selected & (group == identity)).sum()), **{key: float(values[selected & (group == identity)].mean()) for key, values in metrics.items()}} for identity in np.unique(group[selected])}
            summaries[name] = {"rows": int(selected.sum()), "groups": len(by_group), "by_group": by_group, "group_macro": {key: float(np.mean([row[key] for row in by_group.values()])) for key in metrics} if by_group else None}
        np.savez(directory / "rows.npz", **row_arrays, top128_cell_index=top.numpy(), top128_cell_log_probability=np.concatenate(top_lp), query_descriptor=np.concatenate(query_bank), top128_representation_probability=np.concatenate(top_rep), reference_center_ap_dv_ml_um=truth_center.numpy(), reference_normal_ap_dv_ml=truth_normal.numpy(), **metrics)
        if save_full:
            del raw, components
        saved_files = ("raw_cell_log_probability.npy", "raw_component_log_score.npy", "rows.npz") if save_full else ("rows.npz",)
        receipt = {"arm": arm, "step": FINAL, "full_raw_scores_saved": save_full, "streamed_train_reconstruction": "saved FP32 query descriptors, unchanged endpoint FP16 gallery and priors reconstruct all scores", "checkpoint_sha256": checkpoint_sha256, "gallery_sha256": bank_sha256, "subsets": summaries, "probabilities_calibrated": False, "raw_log_normalization_error": normalization_error, "output_sha256": {name: sha(directory / name) for name in saved_files}}
        if dataset == "allen_train":
            receipt["output_sha256"]["exact_anchor_reconstruction.npz"] = sha(directory / "exact_anchor_reconstruction.npz")
        if dataset in ("allen_train", "allen_raw"):
            receipt["donor_macro"] = summaries["all"]["group_macro"]
        (directory / "summary.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8"); results[dataset] = receipt
        print(json.dumps({"arm": arm, "dataset": dataset, "macro": summaries["eligible" if dataset.startswith("synthetic") else "all"]["group_macro"]}), flush=True)
    real_macro = results["allen_raw"]["donor_macro"]
    gates = {"real_B8000_normal_retention": real_macro["plane_angle_deg"] <= parent_real_macro["plane_angle_deg"] + 2., "real_B8000_top32_retention": real_macro["physical_plane_capture_at_32"] >= parent_real_macro["physical_plane_capture_at_32"] - .02, "real_A6000_normal_improvement_at_least_5deg": real_macro["plane_angle_deg"] <= reference_real_macro["plane_angle_deg"] - 5., "real_A6000_top32_plane_improvement_at_least_point10": real_macro["physical_plane_capture_at_32"] >= reference_real_macro["physical_plane_capture_at_32"] + .10}
    for name, candidate in results["synthetic"]["subsets"].items():
        if (name == "eligible" or name.startswith("eligible_mode:")) and candidate["rows"]:
            reference = reference_synthetic["subsets"][name]
            assert candidate["rows"] == reference["rows"] and candidate["groups"] == reference["groups"]
            b, a = candidate["group_macro"], reference["group_macro"]
            gates[f"synthetic_normal_retention:{name}"] = b["plane_angle_deg"] <= a["plane_angle_deg"] + 2.
            gates[f"synthetic_top32_retention:{name}"] = b["physical_plane_capture_at_32"] >= a["physical_plane_capture_at_32"] - .02

    for name, expected in source["file_sha256"].items():
        assert sha(repository / name) == expected
    completion = {"arm": arm, "experiment": config, "source_unchanged": True, "additional_optimizer_steps_applied": applied, "optimizer_steps_applied": FINAL, "frozen_nonretrieval_tensors_exact": True, "gates": gates, "recovery_gate_passed": bool(all(gates.values())), "results": results, "elapsed_seconds": time.perf_counter() - started, "probabilities_calibrated": False}
    (output / "completed.json").write_text(json.dumps(completion, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"arm": arm, "recovery_gate_passed": completion["recovery_gate_passed"], "gates": gates, "real_donor_macro": real_macro}), flush=True)
    arm_results[arm] = {"completion_sha256": sha(output / "completed.json"), "checkpoint_sha256": checkpoint_sha256, "gallery_sha256": bank_sha256, "gates": gates, "recovery_gate_passed": completion["recovery_gate_passed"], "real_donor_macro": real_macro}
    del model, optimizer, trainable, bank
    torch.cuda.empty_cache()
    if arm == "A" and completion["recovery_gate_passed"]:
        break

completion = {"experiment": config, "source_unchanged": True, "arm_results": arm_results, "policy": config["sequential_policy"], "negative_selection_comparison_available": "B" in arm_results, "elapsed_seconds": time.perf_counter() - started, "probabilities_calibrated": False}
(RUN / "completed.json").write_text(json.dumps(completion, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"arms_complete": list(arm_results), "arm_gates": {arm: row["recovery_gate_passed"] for arm, row in arm_results.items()}, "elapsed_seconds": completion["elapsed_seconds"]}), flush=True)
