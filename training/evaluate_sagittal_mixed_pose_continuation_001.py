"""Frozen direct-pose readout for the bounded mixed-pose continuation."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(root / "tmp")
os.environ["TORCH_HOME"] = str(root / "cache/torch")
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


run = root / "runs/sagittal_mixed_pose_continuation_001"
panel = root / "data/one_shot_fresh_synthetic_dev_panel_001"
coronal = root / "data/joint_v7_allen_fullcanvas_192_001"
sagittal = root / "data/allen_sagittal_ish_expansion_002_dev2_inputs_20261008"
catalogue_path = root / "data/allen_sagittal_ish_expansion_002_20261008/manifest.json"
availability_path = root / "data/allen_sagittal_ish_expansion_002_dev2_availability_20261008/manifest.json"
out = root / "runs/sagittal_mixed_pose_continuation_001_eval"
protocol = Path(__file__).resolve().parents[1] / "docs/publication/SAGITTAL_MIXED_POSE_CONTINUATION_001_PROTOCOL_20261008.md"
amendment = Path(__file__).resolve().parents[1] / "docs/publication/SAGITTAL_MIXED_POSE_DEV2_SOURCE_AMENDMENT_20261008.md"
steps, side = (0, 326, 653), 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

completed = json.loads((run / "completed.json").read_text())
config = json.loads((run / "config.json").read_text())
assert completed["updates"] == 653 and completed["unique_sagittal_train_sections"] == 653
assert completed["unique_coronal_train_donors"] == 653 and not completed["calibrated"]
assert sha(run / "config.json") == completed["config_sha256"]
assert config["parent_sha256"] == "70edfaa53fcd84a2d20be5a246948e17a7eeb0406e054cf65535e381f5895ab8"
assert config["protocol_sha256"] == sha(protocol)
assert sha(amendment) == "8701942ea162c7b0b35d38a4d165267c8a5a8e50f5bb2da788fd94739edc5e4b"
assert config["sagittal_frozen_catalogue_sha256"] == sha(catalogue_path)
for name, expected in config["source_sha256"].items():
    assert sha(Path(__file__).parent / name) == expected, name
for step in steps:
    assert sha(run / f"joint_step_{step:04d}.pt") == completed["checkpoint_sha256"][str(step)]

panel_done = json.loads((panel / "completed.json").read_text())
assert panel_done["eligible"] == 185 and panel_done["synthetic_subjects"] == 8
assert sha(panel / "records.jsonl") == panel_done["records_sha256"] == "7be8b6a95ed956c1223fa2dae8867d7747037ec15b8523a5e9b9969c154fc7fe"
synthetic = [json.loads(line) for line in (panel / "records.jsonl").read_text().splitlines()]
synthetic = [row for row in synthetic if row["eligible"]]
assert len(synthetic) == 185 and len({row["animal_id"] for row in synthetic}) == 8
for row in synthetic:
    assert sha(panel / row["file"]) == row["sha256"]

coronal_done = json.loads((coronal / "completed.json").read_text())
assert coronal_done["development_images"] == 64 and coronal_done["development_donors"] == 6
for name, expected in coronal_done["output_sha256"].items():
    assert sha(coronal / name) == expected, name
coronal_records = [row for row in map(json.loads, (coronal / "records.jsonl").open())
                   if row["training_split"] == "development"]
assert len(coronal_records) == 64 and len({row["animal_id"] for row in coronal_records}) == 6
coronal_images = np.load(coronal / "images.npy", mmap_mode="r")
with np.load(coronal / "geometry.npz", allow_pickle=False) as arrays:
    coronal_affines = arrays["model_pixel_to_ap_dv_ml_um"].copy()

sagittal_summary = json.loads((sagittal / "summary.json").read_text())
assert sagittal_summary["source_metadata_manifest_sha256"] == sha(catalogue_path) == "b4f77e00325273f937e4327b52d8b3f7db95458c130d4f88f682a8a08323f7bd"
assert sha(availability_path) == "8cec8d29c4d177073e58b8bc2871e849c1051620a4d00eb88d636017f402066d"
assert sagittal_summary["source_availability_manifest_sha256"] == sha(availability_path)
for name, expected in sagittal_summary["output_sha256"].items():
    assert sha(sagittal / name) == expected, name
sagittal_records = [json.loads(line) for line in (sagittal / "geometry.jsonl").read_text().splitlines()]
dev2_donors = {10422, 10405, 10355, 10347, 10430, 10248, 10443, 10354}
catalogue = json.loads(catalogue_path.read_text())
availability = json.loads(availability_path.read_text())
assert availability["source_manifest_sha256"] == sha(catalogue_path)
intended_dev2 = {(row["donor_id"], row["specimen_id"], row["experiment_id"], row["section_id"])
                 for row in catalogue["sections"] if row["donor_id"] in dev2_donors}
inventory_dev2 = {(row["donor_id"], row["specimen_id"], row["experiment_id"], row["section_id"])
                  for row in availability["sections"]}
assert len(intended_dev2) == len(inventory_dev2) == 159 and intended_dev2 == inventory_dev2
unavailable = [row for row in availability["sections"] if not row["available_decoded_jpeg"]]
assert [(row["donor_id"], row["section_id"]) for row in unavailable] == [(10430, 101345593)]
expected_dev2 = {(row["donor_id"], row["specimen_id"], row["experiment_id"], row["section_id"])
                 for row in availability["sections"] if row["available_decoded_jpeg"]}
actual_dev2 = {(row["donor_id"], row["specimen_id"], row["experiment_id"], row["section_id"])
               for row in sagittal_records}
assert len(sagittal_records) == len(expected_dev2) == len(actual_dev2) == 158
assert expected_dev2 == actual_dev2 and {row["donor_id"] for row in sagittal_records} == dev2_donors
assert [row["array_row_index"] for row in sagittal_records] == list(range(158))
sagittal_images = np.load(sagittal / "model_input.npy", mmap_mode="r")
assert sagittal_images.shape == (158, 1, side, side)

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.], [127.5, 127.5]], device="cuda")
five_uv = (pixels / side)[None].expand(2, -1, -1).clone()
five_uv[1, :, 0] = 255 / side - five_uv[1, :, 0]
five_uv -= .5
out.mkdir(parents=True, exist_ok=False)
rows = []

with torch.inference_mode(), (out / "rows.jsonl").open("w", encoding="utf-8") as stream:
    for step in steps:
        checkpoint = torch.load(run / f"joint_step_{step:04d}.pt", map_location="cpu", weights_only=True)
        assert checkpoint["step"] == step and not checkpoint["calibrated"]
        assert checkpoint["config"] == config
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        for record in synthetic:
            with np.load(panel / record["file"], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays["inputs"][None].copy()).cuda()
                target_surface = torch.from_numpy(arrays["target_centre_um"].copy()).cuda()
                valid = torch.from_numpy(arrays["valid_mask"].copy()).cuda().bool()
                offsets = torch.from_numpy(arrays["offsets_um"][None].copy()).cuda()
                weights = torch.from_numpy(arrays["weights"][None].copy()).cuda()
            prediction = model.predict(image)
            prior = (prediction["log_mass"][0, :, None] + torch.stack((
                F.logsigmoid(-prediction["reflection_logit"][0]),
                F.logsigmoid(prediction["reflection_logit"][0])), -1)).flatten()
            selected = int(prior.argmax())
            target = target_surface[valid]
            mapped_error = torch.empty(model.modes * 2, device="cuda")
            for first in range(0, model.modes * 2, 4):
                branch = torch.arange(first, first + 4, device="cuda")
                mapped = model.map(prediction, offsets, (branch // 2)[None], (branch % 2)[None],
                                   (side, side), atlas, weights)
                mapped_error[first:first + 4] = (mapped["centre_surface_ccf_ap_dv_ml_um"][0, :, valid]
                                                 - target[None]).norm(dim=-1).mean(-1)
            indices = valid.flatten().nonzero().flatten()
            chart = torch.stack((indices.remainder(side), indices.div(side, rounding_mode="floor")), -1).float() / side
            chart = chart[None].expand(2, -1, -1).clone()
            chart[1, :, 0] = 255 / side - chart[1, :, 0]
            centre, frame, basis = full_frame_state_to_components(prediction["state"])
            rigid = centre[0, :, None, None, :] + torch.einsum("mij,rpj->mrpi", frame[0, :, :, :2] @ basis[0], chart - .5)
            rigid_error = (rigid - target[None, None]).norm(dim=-1).mean(-1).flatten()
            row = {"set": "synthetic", "step": step,
                   **{key: record[key] for key in ("animal_id", "specimen_id", "experiment_id", "section_id", "sha256")},
                   "selected_branch": selected,
                   "selected_mapped_um": float(mapped_error[selected]),
                   "oracle_best_predicted_branch_mapped_um": float(mapped_error.min()),
                   "selected_rigid_um": float(rigid_error[selected]),
                   "oracle_best_predicted_branch_rigid_um": float(rigid_error.min()),
                   "all_branch_mapped_um": mapped_error.cpu().tolist(),
                   "all_branch_prior_log_mass": prior.cpu().tolist()}
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
        for domain, records in (("coronal", coronal_records), ("sagittal", sagittal_records)):
            for record in records:
                index = record["array_row_index"]
                if domain == "coronal":
                    native = np.concatenate((np.asarray(coronal_images[index], dtype=np.float32),
                                             np.zeros((4, 192, 192), dtype=np.float32)))[None]
                    image = F.interpolate(torch.from_numpy(native).cuda(), (side, side),
                                          mode="bilinear", align_corners=False)
                    affine = torch.as_tensor(coronal_affines[index], device="cuda", dtype=torch.float32)
                    reference = (affine[:, 2] + (192 / side) * pixels[:, :1] * affine[:, 0]
                                 + (192 / side) * pixels[:, 1:] * affine[:, 1])
                    identity = record["animal_id"]
                else:
                    native = np.concatenate((np.asarray(sagittal_images[index], dtype=np.float32),
                                             np.zeros((4, side, side), dtype=np.float32)))[None]
                    image = torch.from_numpy(native).cuda()
                    affine = torch.as_tensor(record["model_pixel_to_ccf_ref9_ap_dv_ml_um"],
                                             device="cuda", dtype=torch.float32)
                    reference = affine[:, 2] + pixels[:, :1] * affine[:, 0] + pixels[:, 1:] * affine[:, 1]
                    identity = record["donor_id"]
                prediction = model.predict(image)
                centre, frame, basis = full_frame_state_to_components(prediction["state"])
                fitted = centre[0, :, None, None, :] + torch.einsum("mij,rpj->mrpi", frame[0, :, :, :2] @ basis[0], five_uv)
                error = (fitted - reference).norm(dim=-1).mean(-1)
                normal = F.normalize(torch.linalg.cross(affine[:, 0], affine[:, 1]), dim=0)
                angle = torch.rad2deg((frame[0, :, :, 2] @ normal).abs().clamp(0, 1).acos())
                prior = (prediction["log_mass"][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction["reflection_logit"][0]),
                    F.logsigmoid(prediction["reflection_logit"][0])), -1)).flatten()
                selected = int(prior.argmax())
                row = {"set": domain, "step": step, "identity": identity,
                       **{key: record[key] for key in ("specimen_id", "experiment_id", "section_id")},
                       "selected_branch": selected,
                       "selected_five_point_um": float(error.flatten()[selected]),
                       "oracle_best_predicted_branch_five_point_um": float(error.min()),
                       "selected_normal_deg": float(angle[selected // 2]),
                       "oracle_best_predicted_mode_normal_deg": float(angle.min()),
                       "all_branch_five_point_um": error.flatten().cpu().tolist(),
                       "all_mode_normal_deg": angle.cpu().tolist(),
                       "all_branch_prior_log_mass": prior.cpu().tolist()}
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
        stream.flush()
        print(json.dumps({"checkpoint_batch": step, "synthetic_eligible": 185,
                          "coronal_weak": 64, "sagittal_weak_dev2_available": 158}), flush=True)

metrics = {"synthetic": ("selected_mapped_um", "oracle_best_predicted_branch_mapped_um",
                         "selected_rigid_um", "oracle_best_predicted_branch_rigid_um"),
           "coronal": ("selected_five_point_um", "oracle_best_predicted_branch_five_point_um",
                       "selected_normal_deg", "oracle_best_predicted_mode_normal_deg"),
           "sagittal": ("selected_five_point_um", "oracle_best_predicted_branch_five_point_um",
                        "selected_normal_deg", "oracle_best_predicted_mode_normal_deg")}
summary = {}
for step in steps:
    summary[str(step)] = {}
    for domain in metrics:
        group = [row for row in rows if row["set"] == domain and row["step"] == step]
        identity_key = "animal_id" if domain == "synthetic" else "identity"
        identities = sorted({row[identity_key] for row in group})
        by_identity = {str(identity): {"sections": sum(row[identity_key] == identity for row in group),
                                      **{name: float(np.mean([row[name] for row in group if row[identity_key] == identity]))
                                         for name in metrics[domain]}} for identity in identities}
        summary[str(step)][domain] = {"sections": len(group), "identities": len(identities),
                                      "by_identity": by_identity,
                                      "identity_equal": {name: float(np.mean([value[name] for value in by_identity.values()]))
                                                         for name in metrics[domain]}}

baseline = summary["0"]
gate = {}
for step in (326, 653):
    current = summary[str(step)]
    sagittal_angle_gain = (baseline["sagittal"]["identity_equal"]["selected_normal_deg"]
                           - current["sagittal"]["identity_equal"]["selected_normal_deg"])
    sagittal_five_gain = (baseline["sagittal"]["identity_equal"]["selected_five_point_um"]
                          - current["sagittal"]["identity_equal"]["selected_five_point_um"])
    coronal_worsening = (current["coronal"]["identity_equal"]["selected_five_point_um"]
                         - baseline["coronal"]["identity_equal"]["selected_five_point_um"])
    synthetic_worsening = (current["synthetic"]["identity_equal"]["selected_mapped_um"]
                           - baseline["synthetic"]["identity_equal"]["selected_mapped_um"])
    gate[str(step)] = {"sagittal_normal_improvement_deg": sagittal_angle_gain,
                       "sagittal_five_point_improvement_um": sagittal_five_gain,
                       "coronal_five_point_worsening_um": coronal_worsening,
                       "synthetic_mapped_worsening_um": synthetic_worsening,
                       "passes": bool(sagittal_angle_gain >= 20 and sagittal_five_gain >= 2000
                                      and coronal_worsening <= 200 and synthetic_worsening <= 250)}
qualified = [step for step in (326, 653) if gate[str(step)]["passes"]]
selected_step = (min(qualified, key=lambda step: (
    summary[str(step)]["sagittal"]["identity_equal"]["selected_normal_deg"],
    summary[str(step)]["sagittal"]["identity_equal"]["selected_five_point_um"], step))
                 if qualified else 0)
result = {"version": "sagittal-mixed-pose-continuation-001-eval", "summary": summary,
          "sagittal_weak_dev2_original_sections": 159,
          "sagittal_weak_dev2_source_available_sections": 158,
          "sagittal_weak_dev2_source_unavailable_section_ids": [101345593],
          "predeclared_gate": gate, "qualified_nonzero_steps": qualified,
          "selected_step": selected_step, "selected_step_rule": "lowest sagittal donor-equal selected normal among qualified, then five-point, then earliest; otherwise batch 0",
          "reference": "synthetic dense atlas truth; inherited weak Allen real affines, not independent expert labels",
          "scope": "internal mechanism gate only; not calibration, sealed holdout, public benchmark or GUI release",
          "input_sha256": {"training_completed": sha(run / "completed.json"),
                           "training_config": sha(run / "config.json"),
                           "checkpoints": completed["checkpoint_sha256"],
                           "synthetic_panel_completed": sha(panel / "completed.json"),
                           "synthetic_panel_records": panel_done["records_sha256"],
                           "coronal_completed": sha(coronal / "completed.json"),
                           "coronal_outputs": coronal_done["output_sha256"],
                           "sagittal_dev2_summary": sha(sagittal / "summary.json"),
                           "sagittal_dev2_outputs": sagittal_summary["output_sha256"],
                           "sagittal_frozen_catalogue": sha(catalogue_path),
                           "sagittal_dev2_source_availability": sha(availability_path),
                           "protocol": sha(protocol), "source_amendment": sha(amendment),
                           "script": sha(Path(__file__))},
          "rows_sha256": sha(out / "rows.jsonl"), "calibrated": False,
          "public_benchmark_used": False}
(out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"selected_step": selected_step, "gate": gate,
                  "rows_sha256": result["rows_sha256"]}, indent=2), flush=True)
