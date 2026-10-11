"""No-update direct-prior versus actual fitted-output audit on frozen 226 cases."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path("I:/AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(root / "tmp")
os.environ["TORCH_HOME"] = str(root / "cache/torch")
os.environ["XDG_CACHE_HOME"] = str(root / "cache")
os.environ["CUDA_CACHE_PATH"] = str(root / "cache/cuda")
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_inference import infer_one_shot
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.support_aware_framing_217 import sample_support_aware_framing_217


source = Path(__file__).resolve().parent
protocol = source.parent / "docs/publication/ONE_SHOT_FINAL_PATH_231_PROTOCOL_20261011.md"
parent_dir = root / "runs/pose_framing_221_framed_train_001"
trained_dir = root / "runs/pose_appearance_exposure_226_train_001"
readout_dir = root / "runs/pose_appearance_exposure_226_train_readout_001"
out = root / "runs/one_shot_final_path_231_002"
assert source.drive.upper() == out.drive.upper() == "I:" and not out.exists()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

checkpoints = {"221_parent": parent_dir / "joint_step_00400.pt",
               "226_step1200": trained_dir / "joint_step_01200.pt"}
parent_done = json.loads((parent_dir / "completed.json").read_text())
trained_done = json.loads((trained_dir / "completed.json").read_text())
readout_done = json.loads((readout_dir / "completed.json").read_text())
checkpoint_hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest()
                     for label, path in checkpoints.items()}
assert checkpoint_hashes["221_parent"] == parent_done["checkpoint_sha256"]["400"]
assert checkpoint_hashes["226_step1200"] == trained_done["checkpoint_sha256"]["1200"]
assert checkpoint_hashes["221_parent"] == trained_done["parent_checkpoint_sha256"]
cases_path = readout_dir / "cases.jsonl"
assert hashlib.sha256(cases_path.read_bytes()).hexdigest() == readout_done["output_sha256"]["cases.jsonl"]
case_rows = [json.loads(line) for line in cases_path.open()]
case_rows = [row for row in case_rows if row["panel"] == "v7_framed"]
assert len(case_rows) == 64 and len({row["physical_section_id"] for row in case_rows}) == 64

context = load_streaming_synthetic_v7_64(device="cuda")
cases = []
with torch.inference_mode():
    for row in case_rows:
        base, seed = int(row["base"]), int(row["seed"])
        source_sample = sample_panel_geometry_train_166(context, [base], seed, side=256)
        source_sample = sample_support_aware_framing_217(source_sample, seed)
        sample = sample_one_shot_slide_artifacts_v7(context, [base], seed, side=256,
                                                     source_sample=source_sample)
        image_hash = hashlib.sha256(sample["inputs"].cpu().numpy().tobytes()).hexdigest()
        assert bool(sample["eligible"][0])
        assert sample["provenance"][0]["physical_section_id"] == row["physical_section_id"]
        assert image_hash == row["input_sha256"]
        pixels = sample["valid_mask"][0].flatten().nonzero()[:, 0]
        pixels = pixels[torch.linspace(0, len(pixels) - 1, min(128, len(pixels)),
            device="cuda").long()]
        sites = torch.stack((pixels.remainder(256), pixels.div(256, rounding_mode="floor")), -1).float() / 256
        cases.append({"row": row, "image": sample["inputs"].cpu(),
                      "offsets": sample["offsets"].cpu(), "weights": sample["weights"].cpu(),
                      "sites": sites.cpu(), "pixels": pixels.cpu(),
                      "truth": sample["centre"][0].reshape(-1, 3)[pixels].cpu(),
                      "normal": full_frame_state_to_components(sample["state"])[1][0, :, 2].cpu()})

out.mkdir(parents=True)
config = {"schema": "anatomy-tracker.one-shot-final-path/231",
          "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "source_readout_cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
          "checkpoint_sha256": checkpoint_hashes,
          "case_count": len(cases), "trained": False, "public_benchmark_used": False,
          "reserved_dev_cal_final_used": False, "calibrated": False}
(out / "config.json").write_text(json.dumps(config, indent=2))

rows = []
with torch.inference_mode(), (out / "rows.jsonl").open("w") as stream:
    for label, path in checkpoints.items():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
            atlas_conditioning=True, fit_quality=True, vector_refinement=True,
            candidate_ranking=True, fitted_ranking=True).cuda().eval()
        attach_global_plane_matcher(model, enabled=True)
        model.load_state_dict(saved["model"], strict=True)
        del saved
        for case in cases:
            image = case["image"].cuda()
            offsets, weights = case["offsets"].cuda(), case["weights"].cuda()
            sites, truth, normal = case["sites"].cuda(), case["truth"].cuda(), case["normal"].cuda()
            predicted = model.predict(image)
            prior = (predicted["log_mass"][0, :, None] + torch.stack((
                F.logsigmoid(-predicted["reflection_logit"][0]),
                F.logsigmoid(predicted["reflection_logit"][0])), -1)).flatten()
            direct_branch = int(prior.argmax())
            states = predicted["state"][0].repeat_interleave(2, 0)
            reflections = torch.arange(2, device="cuda").repeat(model.modes)
            all_rigid = rigid_points_090(states, reflections, sites)
            all_error = (all_rigid - truth[None]).norm(dim=-1).mean(-1) / 1000
            direct_normal = full_frame_state_to_components(states[direct_branch])[1][:, 2]
            direct_angle = torch.rad2deg(torch.acos((direct_normal * normal).sum().abs().clamp_max(1)))
            direct_map = model.map(predicted, offsets, torch.tensor([[direct_branch // 2]], device="cuda"),
                                   torch.tensor([[direct_branch % 2]], device="cuda"),
                                   (256, 256), context["atlas"], weights, source_shape=(256, 256))
            direct_surface = direct_map["centre_surface_ccf_ap_dv_ml_um"][0, 0].reshape(-1, 3)[case["pixels"].cuda()]
            direct_local_mm = (direct_surface - truth).norm(dim=-1).mean() / 1000
            final = infer_one_shot(model, image, context["atlas"], offsets, weights)
            final_branch = 2 * final["source_selected_component"][0] + final["source_selected_component"][1]
            final_state = final["state"][0, 0].cuda()
            final_reflection = torch.tensor(final_branch % 2, device="cuda")
            final_rigid = rigid_points_090(final_state, final_reflection, sites)
            final_rigid_mm = (final_rigid - truth).norm(dim=-1).mean() / 1000
            final_normal = full_frame_state_to_components(final_state)[1][:, 2]
            final_angle = torch.rad2deg(torch.acos((final_normal * normal).sum().abs().clamp_max(1)))
            final_surface = final["surface"][0, 0].reshape(-1, 3)[case["pixels"]].cuda()
            final_local_mm = (final_surface - truth).norm(dim=-1).mean() / 1000
            record = {"checkpoint": label, "physical_section_id": case["row"]["physical_section_id"],
                      "input_sha256": case["row"]["input_sha256"],
                      "direct_branch": direct_branch, "final_branch": final_branch,
                      "direct_rigid_mm": float(all_error[direct_branch]),
                      "final_rigid_mm": float(final_rigid_mm),
                      "oracle160_direct_rigid_mm": float(all_error.min()),
                      "direct_normal_deg": float(direct_angle),
                      "final_normal_deg": float(final_angle),
                      "direct_local_mm": float(direct_local_mm),
                      "final_local_mm": float(final_local_mm)}
            rows.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        print(json.dumps({"checkpoint": label, "cases": len(cases)}), flush=True)
        del model
        torch.cuda.empty_cache()

summary = {}
for label in checkpoints:
    group = [row for row in rows if row["checkpoint"] == label]
    summary[label] = {name: {"mean": float(np.mean([row[name] for row in group])),
                             "median": float(np.median([row[name] for row in group]))}
                      for name in ("direct_rigid_mm", "final_rigid_mm", "oracle160_direct_rigid_mm",
                                   "direct_normal_deg", "final_normal_deg", "direct_local_mm", "final_local_mm")}
    summary[label]["paired_rigid_gain_median_mm"] = float(np.median([
        row["direct_rigid_mm"] - row["final_rigid_mm"] for row in group]))
    summary[label]["branch_changes"] = sum(row["direct_branch"] != row["final_branch"] for row in group)
summary["rescue_gate_226"] = bool(
    summary["226_step1200"]["direct_rigid_mm"]["median"] -
    summary["226_step1200"]["final_rigid_mm"]["median"] >= .30 and
    summary["226_step1200"]["final_normal_deg"]["median"] -
    summary["226_step1200"]["direct_normal_deg"]["median"] <= 2 and
    summary["226_step1200"]["final_local_mm"]["median"] <=
    summary["226_step1200"]["direct_local_mm"]["median"])
(out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / "completed.json").write_text(json.dumps({**config, "row_count": len(rows),
    "output_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                      for name in ("config.json", "rows.jsonl", "summary.json")}}, indent=2))
print(json.dumps({"event": "completed", "summary": summary}), flush=True)
