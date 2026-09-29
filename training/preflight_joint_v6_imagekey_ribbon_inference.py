"""One original TRAIN observation; full-gallery -> same-model ribbon connectivity.

One frozen TRAIN-row connectivity check. No accuracy gate or weight merging.
"""

import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda")
sys.dont_write_bytecode = True
READY_AFTER_ROOT_REVIEW = True
assert READY_AFTER_ROOT_REVIEW, "Prepared only; root reviews, commits and schedules execution"

import hashlib
import json
import shutil
import subprocess
import time

import numpy as np
import torch

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_catalogue_runtime_v6 import make_complete_catalogue_runtime_v6
from training.arbitrary_plane_imagekey_ribbon_inference_v6 import imagekey_ribbon_inference_v6
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact

NATIVE = ROOT / "runs/joint_v6_ribbon_local_001"
COARSE = ROOT / "runs/joint_v6_imagekey_retrieval_001"
DATA = ROOT / "data/joint_v6_coherent_subject_cohort_sections_002"
OUTPUT = ROOT / "runs/joint_v6_imagekey_ribbon_inference_preflight_001"
NATIVE_AUDIT = ROOT / "runs/joint_v6_ribbon_local_001_independent_audit/audit.json"
COARSE_AUDIT = ROOT / "runs/joint_v6_imagekey_retrieval_001_independent_audit/audit.json"
CHECKPOINT = NATIVE / "joint_model_step_02000.pt"
PARENT = COARSE / "joint_model_step_04000.pt"
BANK = COARSE / "gallery_descriptors_step_04000.npy"
PINS = {
    NATIVE / "completed.json": "a2bee6dc1ffcaafcef3b4ad4a49223653f582f41ab071a3ca25928e7db7a83d7",
    NATIVE_AUDIT: "46b243156dc6e9388d7aa9dc2dac362f557b25a183c25226c0fefa5167b1f21d",
    CHECKPOINT: "966f236f98faa67cbc69c7b2236e12e84938d525496ed1a7644dbbc5e2745671",
    COARSE_AUDIT: "f9eb9c6845e048e5fc3ca840a4c5effcf48c1d6ed98afef9daa65a3983e4ca9b",
    PARENT: "d4d706e8d80e53a3638a70e79ce8661ff4af41f7b846143aa1ec68372bfb2ae5",
    BANK: "1568da8f1347641bba4ff93679ff39d5d6250693345db5d915322312bbe64666",
    COARSE / "catalogue.pt": "9b49d203cc73ce3a66e648bbe5228231eb5cc9c17d5db4669eefe0f08ae22c71",
    DATA / "completed.json": "ba51982a5b03b61d4bcf7f37f2c139dd6c1caf7ff66a6124cb678ab5ee9dd1f2",
    NATIVE / "schedule.pt": "e8aaa6eb1ff7e70097cbf1eeb1f6b9fc4d2ba4f01f26fc6d91b671c90c8dc0b9",
    NATIVE / "observation_identities.json": "1f3ea46ea294cf627b430a06cac7eea12cd32ffe762b6bcbf90b14319e0fb981",
}
for path, expected in PINS.items():
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
native_audit = json.loads(NATIVE_AUDIT.read_text())
coarse_audit = json.loads(COARSE_AUDIT.read_text())
done = json.loads((NATIVE / "completed.json").read_text())
assert native_audit["integrity_passed"] and native_audit["frozen_retrieval_state_tensors_exact"]
assert coarse_audit["integrity_passed"] and coarse_audit["advance_to_joint_integration_pilot"]
assert native_audit["artifact_sha256"][str(CHECKPOINT)] == PINS[CHECKPOINT]
assert coarse_audit["artifact_sha256"][str(BANK)] == PINS[BANK]
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
parent = torch.load(PARENT, map_location="cpu", weights_only=False, mmap=True)
config = checkpoint["experiment"]
assert json.loads(json.dumps(config)) == done["experiment"]
assert checkpoint["phase"] == "conditional_native_pose_ribbon" and checkpoint["step"] == 2000
assert checkpoint["global_frozen_tensors_exact"] and config["parent_sha256"] == PINS[PARENT]
assert parent["step"] == 4000 and not config["model_kwargs"].get("signed_pose_evidence", False)
encoder_prefixes = ("pose_model.histology_stem.", "pose_model.atlas_stem.", "pose_model.shared_encoder.",
                    "pose_model.spatial_residual_blocks.", "pose_model.image_key_descriptor.")
encoder_names = [name for name in parent["model_state"] if name.startswith(encoder_prefixes)]
assert encoder_names and all(name in config["frozen_state_tensor_names"] for name in encoder_names)
assert all(torch.equal(checkpoint["model_state"][name], parent["model_state"][name]) for name in encoder_names)
del parent

schedule = torch.load(NATIVE / "schedule.pt", map_location="cpu", weights_only=False)
row = int(schedule["training_observation_index"][0, 0])
identity = json.loads((NATIVE / "observation_identities.json").read_text())[row]
assert identity["split"] == "train"
cohort = json.loads((DATA / "completed.json").read_text())
record = cohort["sections"][identity["section_array_index"]]
assert record["lineage"]["section_id"] == identity["section_id"]
for name, expected in record["artifact_sha256"].items():
    with (DATA / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
section = _read_raw_artifact(DATA, record["artifacts"])
observation = next(item for item in section["observations"] if item["lineage"]["observation_id"] == identity["observation_id"])
assert observation["selected_mode"] == identity["selected_mode"]
channels = torch.from_numpy(observation["image_outline_availability_float32"])[None].float()
offsets = torch.from_numpy(section["axial_offsets_um_float64"])[None].float()
weights = torch.from_numpy(section["axial_weights_float64"])[None].float()
del section, schedule, observation

repository = Path(__file__).resolve().parents[1]
OUTPUT.mkdir(parents=True, exist_ok=False)
source_names = sorted((set(config["source"]["file_sha256"]) - {"training/run_joint_v6_ribbon_local.py"}) | {
    "training/preflight_joint_v6_imagekey_ribbon_inference.py",
    "training/arbitrary_plane_imagekey_ribbon_inference_v6.py",
})
source = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(), "file_sha256": {}}
for name in source_names:
    destination = OUTPUT / "source" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository / name, destination)
    source["file_sha256"][name] = hashlib.sha256(destination.read_bytes()).hexdigest()
torch.save({"observation_index": row, "identity": identity, "channels": channels,
            "axial_offsets_um": offsets, "axial_weights": weights,
            "section_artifact_sha256": record["artifact_sha256"]}, OUTPUT / "input.pt")
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
catalogue = torch.load(COARSE / "catalogue.pt", map_location="cpu", weights_only=False)
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=config["catalogue_receipt_sha256"], device="cuda", dtype=torch.float32)
model = ArbitraryPlaneJointModelV6(runtime, **config["model_kwargs"]).cuda()
model.load_state_dict(checkpoint["model_state"], strict=True)
model.eval()
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation
bank = torch.from_numpy(np.load(BANK)).cuda()
assert bank.shape == (98304, 2, 256)
origin, spacing = [catalogue["support_geometry"][key] for key in ("origin_ap_dv_ml_um", "voxel_size_ap_dv_ml_um")]
channels, offsets, weights = channels.cuda(), offsets.cuda(), weights.cuda()
results, elapsed = {}, {}
for chunk in (1, 4):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = imagekey_ribbon_inference_v6(
        model, channels[:, :1], channels[:, 1:2], channels[:, 2, 0, 0].bool(),
        bank, atlas, origin, spacing, offsets, weights,
        top_k=4, cell_chunk_size=chunk, refinement_steps=3, temperature=.1,
    )
    torch.cuda.synchronize()
    elapsed[str(chunk)] = time.perf_counter() - started
    results[chunk] = {key: value.cpu() if torch.is_tensor(value) else value for key, value in result.items()}
    torch.save(results[chunk], OUTPUT / f"prediction_chunk_{chunk}.pt")
    del result
assert all(torch.equal(model.state_dict()[name].cpu(), value) for name, value in checkpoint["model_state"].items())

a, b = results[1], results[4]
assert a["coarse_full_cell_log_probability"].shape == (1, 98304)
assert a["final_component_state"].shape == (1, 4, 2, 12)
assert a["final_observed_ccf_slab_ap_dv_ml_um"].shape == (1, 4, 2, 9, 96, 96, 3)
for key in ("top_cell_index", "selected_component_index", "selected_cell_index", "selected_representation_index", "horizontal_reflection"):
    assert torch.equal(a[key], b[key])
tolerances = {"coarse_full_cell_log_probability": 2e-6, "initial_component_log_mass": 2e-6,
              "final_canonical_residual_local_um": .02, "final_canonical_director_delta_local": 2e-5,
              "final_observed_centre_ccf_ap_dv_ml_um": .05, "final_observed_ccf_slab_ap_dv_ml_um": .05,
              "final_component_refinement_log_score": 2e-4, "final_component_log_score": 2e-4,
              "conditional_kept_component_log_probability": 2e-4}
maximum_differences = {}
for key, tolerance in tolerances.items():
    maximum_differences[key] = float((a[key] - b[key]).abs().max())
    assert torch.allclose(a[key], b[key], rtol=0., atol=tolerance)
state_tolerance = torch.tensor([.05] * 3 + [2e-5] * 9)
state_difference = (a["final_component_state"] - b["final_component_state"]).abs()
assert bool((state_difference <= state_tolerance).all())
normalization = {}
for chunk, result in results.items():
    assert all(torch.isfinite(value).all() for value in result.values() if torch.is_tensor(value) and value.is_floating_point())
    full = result["coarse_full_cell_log_probability"].double()
    selected = result["top_cell_index"]
    mask = torch.ones_like(full, dtype=torch.bool).scatter_(1, selected, False)
    kept, omitted = full.gather(1, selected).exp().sum(1), full.exp().masked_fill(~mask, 0.).sum(1)
    combined = result["initial_component_log_mass"].double() + result["final_component_refinement_log_score"].double()
    expected = combined - torch.logsumexp(combined.flatten(1), dim=1)[:, None, None]
    assert torch.allclose(expected, result["conditional_kept_component_log_probability"].double(), rtol=0., atol=3e-6)
    assert torch.allclose(combined, result["final_component_log_score"].double(), rtol=0., atol=3e-6)
    assert torch.allclose(torch.logsumexp(result["initial_component_log_mass"].double(), -1), full.gather(1, selected), rtol=0., atol=3e-6)
    assert torch.allclose(kept, result["coarse_retained_mass"].double(), rtol=0., atol=2e-6)
    assert torch.allclose(omitted, result["coarse_omitted_mass"].double(), rtol=0., atol=2e-6)
    assert abs(float(kept + omitted) - 1.) < 2e-6
    assert result["post_refinement_omitted_mass"] is None and not result["probabilities_calibrated"]
    normalization[str(chunk)] = {"full_mass": float(kept + omitted), "retained_coarse_mass": float(kept),
                                 "omitted_coarse_mass": float(omitted), "final_conditional_mass": float(expected.exp().sum())}
output_sha256 = {}
for name in ("input.pt", "prediction_chunk_1.pt", "prediction_chunk_4.pt"):
    with (OUTPUT / name).open("rb") as stream:
        output_sha256[name] = hashlib.file_digest(stream, "sha256").hexdigest()
for name, expected in source["file_sha256"].items():
    assert hashlib.sha256((repository / name).read_bytes()).hexdigest() == expected
summary = {"connectivity_passed": True, "source": source, "input_sha256": {str(path): value for path, value in PINS.items()},
           "observation_index": row, "identity": identity, "section_artifact_sha256": record["artifact_sha256"],
           "whole_native_checkpoint_only": True, "parent_loaded_for_equality_only": True,
           "encoder_tensor_names_exact_to_gallery_parent": encoder_names, "state_unchanged_after_inference": True,
           "full_catalogue_cells": 98304, "retained_cells": 4, "raster_representations": 2, "updates": 3,
           "chunk_sizes": [1, 4], "temperature": .1, "precision": "FP32, AMP and TF32 disabled",
           "maximum_absolute_chunk_differences": maximum_differences,
           "state_maximum_difference_by_coordinate": state_difference.amax((0, 1, 2)).tolist(),
           "absolute_tolerances": tolerances, "state_absolute_tolerance_by_coordinate": state_tolerance.tolist(),
           "normalization": normalization, "elapsed_seconds_not_speed_benchmark": elapsed,
           "output_sha256": output_sha256, "probabilities_calibrated": False,
           "scope": "one original scheduled TRAIN observation; numerical connectivity/chunk equivalence only, no low-error, global-capture, biological, training, calibration or release qualification; no teacher pose or truth masks passed; unknown post-refinement tail"}
(OUTPUT / "completed.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({key: summary[key] for key in ("connectivity_passed", "observation_index", "normalization", "elapsed_seconds_not_speed_benchmark")}), flush=True)
