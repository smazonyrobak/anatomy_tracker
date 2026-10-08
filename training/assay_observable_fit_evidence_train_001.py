"""Frozen TRAIN-only unwarped/warped HOG and deformation evidence on top-eight planes."""

import hashlib
import json
import math
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

from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v2 import sample_one_shot_slide_artifacts_v2
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64


checkpoint_path = root / "runs/sagittal_mixed_pose_continuation_001_retry1/joint_step_0653.pt"
out = root / "runs/observable_fit_evidence_train_assay_001"
protocol = Path(__file__).resolve().parents[1] / "docs/publication/OBSERVABLE_FIT_EVIDENCE_TRAIN_ASSAY_001_PROTOCOL_20261009.md"
seed, side, grid_side, beam, accepted_target = 2026100903, 256, 96, 8, 96
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
config = {"version": "observable-fit-evidence-train-assay-001", "seed": seed,
          "eligible_target": accepted_target, "grid_side": grid_side,
          "top_prior_branches": beam, "checkpoint_sha256": checkpoint_sha,
          "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
          "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ("assay_observable_fit_evidence_train_001.py",
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
                           atlas, weights, return_refinement_feature=True,
                           feature_side=grid_side, source_shape=(side, side))
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
        inherited = model.score_fitted_candidates(image, refined, fitted, atlas, weights)[0] + delta[0]
        masses = weights.expand(beam, -1)
        rendered = render_finite_thickness_coordinate_grid(
            atlas, fitted["coordinates"].flatten(0, 1), (0., 0., 0.), (25., 25., 25.), masses)
        rigid = direct["atlas_pair"][0, :, :1]
        rigid_support = direct["atlas_pair"][0, :, 1:2]
        warped_support = rendered[:, 1:2].clamp(0, 1)
        warped = rendered[:, :1] / warped_support.clamp_min(1e-4)
        atlas_images = torch.cat((rigid, warped), 0)
        supports = torch.cat((rigid_support, warped_support), 0)
        source = F.interpolate(image[:, :1], (grid_side, grid_side), mode="area").expand(2 * beam, -1, -1, -1)
        pictures = torch.cat((source, atlas_images), 0)
        gx = F.pad(pictures[..., 1:] - pictures[..., :-1], (0, 1, 0, 0))
        gy = F.pad(pictures[..., 1:, :] - pictures[..., :-1, :], (0, 0, 0, 1))
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
        angle = torch.remainder(torch.atan2(gy, gx), math.pi)
        centres = torch.arange(8, device="cuda", dtype=angle.dtype)[None, :, None, None] * (math.pi / 8)
        distance = torch.remainder(angle - centres + math.pi / 2, math.pi) - math.pi / 2
        bins = magnitude * (1 - distance.abs() / (math.pi / 8)).clamp_min(0)
        hist = F.avg_pool2d(bins, 8, 8)
        hist = hist / torch.sqrt(hist.square().sum(1, keepdim=True) + 1e-6)
        cell_distance = (hist[:2 * beam] - hist[2 * beam:]).square().sum(1).sqrt()
        cell_support = F.avg_pool2d(supports, 8, 8)[:, 0]
        hog = ((cell_distance * cell_support).sum((1, 2)) /
               cell_support.sum((1, 2)).clamp_min(1e-4)).reshape(2, beam)
        support = supports.mean((1, 2, 3)).reshape(2, beam)
        local = fitted["local_displacement_um"][0] / 1000
        warp_rms = local.square().sum(1).mean((1, 2)).sqrt()
        dx = F.pad((local[..., 1:] - local[..., :-1]).square().sum(1), (0, 1, 0, 0))
        dy = F.pad((local[..., 1:, :] - local[..., :-1, :]).square().sum(1), (0, 0, 0, 1))
        warp_roughness = (dx + dy).mean((1, 2)).sqrt()
        target = F.interpolate(sample["centre"].permute(0, 3, 1, 2), (grid_side, grid_side),
                               mode="bilinear", align_corners=False).permute(0, 2, 3, 1)[0]
        valid = F.interpolate(sample["valid_mask"][:, None].float(), (grid_side, grid_side),
                              mode="nearest")[0, 0].bool()
        direct_error = (direct["centre_surface_ccf_ap_dv_ml_um"][0, :, valid]
                        - target[valid][None]).norm(dim=-1).mean(-1)
        fitted_error = (fitted["centre_surface_ccf_ap_dv_ml_um"][0, :, valid]
                        - target[valid][None]).norm(dim=-1).mean(-1)
        normal = np.abs(np.asarray(meta["virtual_unit_normal"]))
        row = {"physical_section_id": meta["physical_section_id"],
               "virtual_subject_id": meta["virtual_subject_id"], "mode": meta["mode"],
               "heavy_oblique": bool(normal.max() <= .85),
               "valid_fraction": float(sample["valid_mask"].float().mean()),
               "events": meta["one_shot_slide_artifacts_v2"]["events"],
               "prior": prior[choice].cpu().tolist(), "inherited_fitted": inherited.cpu().tolist(),
               "rigid_hog": hog[0].cpu().tolist(), "warped_hog": hog[1].cpu().tolist(),
               "rigid_support": support[0].cpu().tolist(), "warped_support": support[1].cpu().tolist(),
               "warp_rms_mm": warp_rms.cpu().tolist(),
               "warp_roughness_mm": warp_roughness.cpu().tolist(),
               "direct_error_um": direct_error.cpu().tolist(),
               "fitted_error_um": fitted_error.cpu().tolist()}
        rows.append(row)
        stream.write(json.dumps(row) + "\n")
        if len(rows) % 16 == 0:
            attempts.flush()
            stream.flush()
            print(json.dumps({"eligible": len(rows), "target": accepted_target,
                              "attempts": draw_seed - seed - 1}), flush=True)

features = {"image_prior": ("prior", -1), "inherited_fitted": ("inherited_fitted", -1),
            "rigid_hog": ("rigid_hog", 1), "warped_hog": ("warped_hog", 1),
            "rigid_support": ("rigid_support", -1), "warped_support": ("warped_support", -1),
            "warp_rms_mm": ("warp_rms_mm", 1), "warp_roughness_mm": ("warp_roughness_mm", 1)}
strata = {}
for group_name, group in (("all", rows), ("raw", [row for row in rows if row["mode"] == "raw"]),
                          ("nonraw", [row for row in rows if row["mode"] != "raw"]),
                          ("heavy_oblique", [row for row in rows if row["heavy_oblique"]]),
                          ("less_oblique", [row for row in rows if not row["heavy_oblique"]])):
    report = {"n": len(group), "prior_selected_fitted_error_um": float(np.mean([row["fitted_error_um"][0] for row in group])),
              "oracle_fitted_error_um": float(np.mean([min(row["fitted_error_um"]) for row in group]))}
    for name, (field, direction) in features.items():
        right, total, selected = 0, 0, []
        for row in group:
            values = direction * np.asarray(row[field], dtype=np.float64)
            error = np.asarray(row["fitted_error_um"], dtype=np.float64)
            selected.append(error[int(values.argmin())])
            for i in range(beam):
                for j in range(i + 1, beam):
                    if abs(error[i] - error[j]) >= 250 and abs(values[i] - values[j]) > 1e-8:
                        right += (values[i] < values[j]) == (error[i] < error[j])
                        total += 1
        report[name] = {"pairwise_concordance": right / total if total else None,
                        "candidate_pairs": total, "selected_fitted_error_um": float(np.mean(selected))}
    strata[group_name] = report
summary = {"strata": strata, "scope": "TRAIN-only diagnostic of observable evidence, no weights updated",
           "go_no_go": {name: bool(strata["all"][name]["pairwise_concordance"] > .60
                                and strata["raw"][name]["pairwise_concordance"] > .50
                                and strata["heavy_oblique"][name]["pairwise_concordance"] > .50)
                        for name in ("rigid_hog", "warped_hog", "warp_rms_mm", "warp_roughness_mm")}}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
receipt = {"eligible": len(rows), "attempts": draw_seed - seed - 1,
           "output_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                             for name in ("config.json", "attempts.jsonl", "rows.jsonl", "summary.json")},
           "trained": False, "public_benchmark_used": False}
(out / "completed.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"completed": receipt, "main": strata["all"], "go_no_go": summary["go_no_go"]}, indent=2), flush=True)
