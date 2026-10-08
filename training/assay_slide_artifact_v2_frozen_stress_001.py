"""No-update TRAIN-only stress readout for independently sampled v2 slide artifacts."""

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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v2 import sample_one_shot_slide_artifacts_v2
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64


checkpoint_path = root / "runs/sagittal_mixed_pose_continuation_001_retry1/joint_step_0653.pt"
out = root / "runs/slide_artifact_v2_frozen_stress_001"
protocol = Path(__file__).resolve().parents[1] / "docs/publication/SLIDE_ARTIFACT_V2_FROZEN_STRESS_001_PROTOCOL_20261009.md"
seed, side, grid_side, beam, accepted_target = 2026100901, 256, 96, 8, 64
checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
assert checkpoint_sha == "94b9c86f3a9cc15cd5b061acc6b67df4e0690ccdf5f9637a03c9cddd79f73ed8"
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
context = load_streaming_synthetic_v7_64(device="cuda")
atlas = context["atlas"]
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
assert checkpoint["step"] == 653 and not checkpoint["calibrated"]
model.load_state_dict(checkpoint["model"], strict=True)
del checkpoint
rng = np.random.default_rng(seed)
out.mkdir(parents=True, exist_ok=False)
config = {"version": "slide-artifact-v2-frozen-stress-001", "seed": seed,
          "eligible_target": accepted_target, "grid_side": grid_side,
          "top_prior_branches": beam, "checkpoint_sha256": checkpoint_sha,
          "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
          "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ("assay_slide_artifact_v2_frozen_stress_001.py",
                                         "arbitrary_plane_one_shot_model.py",
                                         "arbitrary_plane_one_shot_slide_artifacts_v2.py",
                                         "arbitrary_plane_streaming_synthetic_v7.py",
                                         "arbitrary_plane_streaming_synthetic_v7_64.py")},
          "generator_provenance": context["provenance"], "trained": False,
          "public_benchmark_used": False}
(out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

rows = []
draw_seed = seed + 1
with torch.inference_mode(), (out / "attempts.jsonl").open("w", encoding="utf-8") as attempts, (out / "rows.jsonl").open("w", encoding="utf-8") as stream:
    while len(rows) < accepted_target:
        virtual = int(rng.integers(len(context["subjects"])))
        sample = sample_one_shot_slide_artifacts_v2(context, [virtual], draw_seed, side=side)
        meta = sample["provenance"][0]
        used = bool(sample["eligible"][0])
        attempts.write(json.dumps({"draw_seed": draw_seed, "virtual_index": virtual,
                                   "physical_section_id": meta["physical_section_id"], "used": used}) + "\n")
        draw_seed += 1
        if not used:
            continue
        image = sample["inputs"]
        offsets, weights = sample["offsets"], sample["weights"]
        prediction = model.predict(image)
        prior = (prediction["log_mass"][0, :, None] + torch.stack((
            F.logsigmoid(-prediction["reflection_logit"][0]),
            F.logsigmoid(prediction["reflection_logit"][0])), -1)).flatten()
        choice = prior.topk(beam).indices
        modes, reflected = (choice // 2)[None], (choice % 2)[None]
        direct = model.map(prediction, offsets, modes, reflected, (grid_side, grid_side),
                           atlas, weights, feature_side=grid_side, source_shape=(side, side))
        selected = {**prediction, "state": prediction["state"][:, choice // 2],
                    "log_mass": prediction["log_mass"][:, choice // 2],
                    "reflection_logit": prediction["reflection_logit"][:, choice // 2]}
        first = model.map(selected, offsets, torch.arange(beam, device="cuda")[None], reflected,
                          (64, 64), atlas, weights, return_refinement_feature=True,
                          feature_side=64, source_shape=(side, side))
        refined_state, delta, _ = model.refine(first["refinement_feature"], selected["state"])
        refined = {**selected, "state": refined_state}
        fitted = model.map(refined, offsets, torch.arange(beam, device="cuda")[None], reflected,
                           (grid_side, grid_side), atlas, weights, feature_side=grid_side,
                           source_shape=(side, side))
        score = model.score_fitted_candidates(image, refined, fitted, atlas, weights)[0] + delta[0]
        target = F.interpolate(sample["centre"].permute(0, 3, 1, 2), (grid_side, grid_side),
                               mode="bilinear", align_corners=False).permute(0, 2, 3, 1)[0]
        valid = F.interpolate(sample["valid_mask"][:, None].float(), (grid_side, grid_side),
                              mode="nearest")[0, 0].bool()
        direct_error = (direct["centre_surface_ccf_ap_dv_ml_um"][0, :, valid]
                        - target[valid][None]).norm(dim=-1).mean(-1)
        fitted_error = (fitted["centre_surface_ccf_ap_dv_ml_um"][0, :, valid]
                        - target[valid][None]).norm(dim=-1).mean(-1)
        true_normal = full_frame_state_to_components(sample["state"])[1][0, :, 2]
        direct_normal = full_frame_state_to_components(prediction["state"][:, choice // 2])[1][0, :, :, 2]
        fitted_normal = full_frame_state_to_components(refined_state)[1][0, :, :, 2]
        direct_angle = torch.rad2deg((direct_normal @ true_normal).abs().clamp(0, 1).acos())
        fitted_angle = torch.rad2deg((fitted_normal @ true_normal).abs().clamp(0, 1).acos())
        picked = int(score.argmax())
        normal = np.abs(np.asarray(meta["virtual_unit_normal"]))
        artifact = meta["one_shot_slide_artifacts_v2"]
        row = {"physical_section_id": meta["physical_section_id"],
               "virtual_subject_id": meta["virtual_subject_id"], "virtual_index": virtual,
               "mode": meta["mode"], "nearest_normal_axis": ("AP", "DV", "ML")[int(normal.argmax())],
               "max_abs_normal": float(normal.max()), "heavy_oblique": bool(normal.max() <= .85),
               "events": artifact["events"], "pixel_counts": artifact["pixel_counts"],
               "valid_fraction": float(sample["valid_mask"].float().mean()),
               "prior_branch": int(choice[0]), "fitted_branch": int(choice[picked]),
               "prior_selected_um": float(direct_error[0]), "fitted_selected_um": float(fitted_error[picked]),
               "best8_direct_um": float(direct_error.min()), "best8_fitted_um": float(fitted_error.min()),
               "prior_normal_deg": float(direct_angle[0]), "fitted_normal_deg": float(fitted_angle[picked]),
               "best8_direct_normal_deg": float(direct_angle.min()),
               "top8_prior_log_mass": prior[choice].cpu().tolist(),
               "top8_fitted_score": score.cpu().tolist(),
               "top8_direct_error_um": direct_error.cpu().tolist(),
               "top8_fitted_error_um": fitted_error.cpu().tolist()}
        rows.append(row)
        stream.write(json.dumps(row) + "\n")
        if len(rows) % 16 == 0:
            attempts.flush()
            stream.flush()
            print(json.dumps({"eligible": len(rows), "target": accepted_target,
                              "attempts": draw_seed - seed - 1}), flush=True)

edges = np.quantile([row["valid_fraction"] for row in rows], [.25, .5, .75]).tolist()
for row in rows:
    row["support_quartile"] = int(np.searchsorted(edges, row["valid_fraction"], side="right") + 1)
metrics = ("prior_selected_um", "fitted_selected_um", "best8_direct_um", "best8_fitted_um",
           "prior_normal_deg", "fitted_normal_deg", "best8_direct_normal_deg")
strata = {}
for field in ("mode", "nearest_normal_axis", "heavy_oblique", "support_quartile"):
    strata[field] = {str(value): {"n": len(group), **{name: float(np.mean([row[name] for row in group])) for name in metrics}}
                     for value in sorted({row[field] for row in rows})
                     if (group := [row for row in rows if row[field] == value])}
strata["event"] = {event: {str(present): {"n": len(group), **{name: float(np.mean([row[name] for row in group])) for name in metrics}}
                            for present in (False, True)
                            if (group := [row for row in rows if row["events"][event] == present])}
                   for event in ("fragment", "fold", "bubble", "tile_seam")}
summary = {"all": {name: float(np.mean([row[name] for row in rows])) for name in metrics},
           "branch_changes": sum(row["prior_branch"] != row["fitted_branch"] for row in rows),
           "support_quartile_edges": edges, "strata": strata,
           "scope": "TRAIN-only synthetic stress; intentionally uncalibrated artifact prevalence; no physical oblique validation"}
with (out / "rows.jsonl").open("w", encoding="utf-8") as stream:
    for row in rows:
        stream.write(json.dumps(row) + "\n")
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
receipt = {"eligible": len(rows), "attempts": draw_seed - seed - 1,
           "output_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                             for name in ("config.json", "attempts.jsonl", "rows.jsonl", "summary.json")},
           "trained": False, "public_benchmark_used": False}
(out / "completed.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"completed": receipt, "summary": summary["all"]}, indent=2), flush=True)
