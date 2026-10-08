"""TRAIN-only, no-update fit-selector and synthetic pose-gauge assay."""

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
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64


checkpoint_path = root / "runs/sagittal_mixed_pose_continuation_001_retry1/joint_step_0653.pt"
out = root / "runs/frozen_fit_selector_artifact_gauge_001"
protocol = Path(__file__).resolve().parents[1] / "docs/publication/FROZEN_FIT_SELECTOR_AND_ARTIFACT_GAUGE_001_PROTOCOL_20261008.md"
seed, side, grid_side, beam, accepted_target = 2026100803, 256, 96, 8, 64
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == "94b9c86f3a9cc15cd5b061acc6b67df4e0690ccdf5f9637a03c9cddd79f73ed8"
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
config = {"version": "frozen-fit-selector-artifact-gauge-001", "seed": seed,
          "eligible_target": accepted_target, "grid_side": grid_side, "top_prior_branches": beam,
          "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
          "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
          "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ("assay_frozen_fit_selector_artifact_gauge_001.py",
                                         "arbitrary_plane_one_shot_model.py",
                                         "arbitrary_plane_one_shot_stream.py",
                                         "arbitrary_plane_streaming_synthetic_v7.py",
                                         "arbitrary_plane_streaming_synthetic_v7_64.py")},
          "generator_provenance": context["provenance"], "trained": False,
          "public_benchmark_used": False}
(out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.], [127.5, 127.5]], device="cuda") / side
rows = []
draw_seed = seed + 1
with torch.inference_mode(), (out / "attempts.jsonl").open("w", encoding="utf-8") as attempts, (out / "rows.jsonl").open("w", encoding="utf-8") as stream:
    while len(rows) < accepted_target:
        virtual = int(rng.integers(len(context["subjects"])))
        sample = sample_one_shot_stream(context, [virtual], draw_seed, side=side)
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
        source_state = torch.tensor(meta["one_shot"]["source_state"], device="cuda")[None]
        observed_state = torch.tensor(meta["one_shot"]["observed_affine_state"], device="cuda")[None]
        source_centre, source_frame, source_basis = full_frame_state_to_components(source_state)
        observed_centre, observed_frame, observed_basis = full_frame_state_to_components(observed_state)
        source_points = source_centre[:, None] + torch.einsum("bij,pj->bpi", source_frame[:, :, :2] @ source_basis, corners - .5)
        observed_points = observed_centre[:, None] + torch.einsum("bij,pj->bpi", observed_frame[:, :, :2] @ observed_basis, corners - .5)
        label_shift = (source_points - observed_points).norm(dim=-1).mean()
        label_angle = torch.rad2deg((source_frame[:, :, 2] * observed_frame[:, :, 2]).sum(-1).abs().clamp(0, 1).acos()).mean()
        normal = np.abs(np.asarray(meta["virtual_unit_normal"]))
        row = {"physical_section_id": meta["physical_section_id"], "virtual_subject_id": meta["virtual_subject_id"],
               "virtual_index": virtual, "mode": meta["mode"], "nearest_normal_axis": ("AP", "DV", "ML")[int(normal.argmax())],
               "events": meta["one_shot"]["events"], "valid_fraction": float(valid.float().mean()),
               "prior_branch": int(choice[0]), "fitted_branch": int(choice[int(score.argmax())]),
               "prior_selected_um": float(direct_error[0]), "fitted_selected_um": float(fitted_error[int(score.argmax())]),
               "best8_direct_um": float(direct_error.min()), "best8_fitted_um": float(fitted_error.min()),
               "source_to_observed_label_five_point_um": float(label_shift),
               "source_to_observed_label_normal_deg": float(label_angle),
               "top8_prior_log_mass": prior[choice].cpu().tolist(), "top8_fitted_score": score.cpu().tolist(),
               "top8_direct_error_um": direct_error.cpu().tolist(), "top8_fitted_error_um": fitted_error.cpu().tolist()}
        rows.append(row)
        stream.write(json.dumps(row) + "\n")
        if len(rows) % 16 == 0:
            attempts.flush()
            stream.flush()
            print(json.dumps({"eligible": len(rows), "target": accepted_target, "attempts": draw_seed - seed - 1}), flush=True)

names = ("prior_selected_um", "fitted_selected_um", "best8_direct_um", "best8_fitted_um",
         "source_to_observed_label_five_point_um", "source_to_observed_label_normal_deg")
summary = {"all": {name: float(np.mean([row[name] for row in rows])) for name in names},
           "appearance": {mode: {"n": sum(row["mode"] == mode for row in rows),
                                   **{name: float(np.mean([row[name] for row in rows if row["mode"] == mode])) for name in names}}
                                                     for mode in sorted({row["mode"] for row in rows})},
           "nearest_normal_axis": {axis: {"n": sum(row["nearest_normal_axis"] == axis for row in rows),
                                          **{name: float(np.mean([row[name] for row in rows if row["nearest_normal_axis"] == axis])) for name in names}}
                                   for axis in ("AP", "DV", "ML")},
           "event_label_shift": {name: {"present_n": sum(row["events"][name] for row in rows),
                                         "present_five_point_um": float(np.mean([row["source_to_observed_label_five_point_um"] for row in rows if row["events"][name]])),
                                         "absent_five_point_um": float(np.mean([row["source_to_observed_label_five_point_um"] for row in rows if not row["events"][name]]))}
                                 for name in ("tear", "missing", "fold", "bubble", "tile_seam")}}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
receipt = {"eligible": len(rows), "attempts": draw_seed - seed - 1,
           "output_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                             for name in ("config.json", "attempts.jsonl", "rows.jsonl", "summary.json")},
           "trained": False, "public_benchmark_used": False}
(out / "completed.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"completed": receipt, "summary": summary["all"]}, indent=2), flush=True)
