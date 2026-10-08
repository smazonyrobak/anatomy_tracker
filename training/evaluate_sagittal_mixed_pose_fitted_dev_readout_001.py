"""Frozen, read-only inherited atlas-fit versus image-prior DEV comparison."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(root / "tmp")
os.environ["TORCH_HOME"] = str(root / "cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(root / "cache/cuda")
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def points(state, reflection, pixels):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = (pixels / 256)[None, None].expand(*state.shape[:2], -1, -1).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / 256 - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum("...ij,...pj->...pi", frame[..., :, :2] @ basis, chart - .5)


run = root / "runs/sagittal_mixed_pose_continuation_001_retry1"
prior_eval = root / "runs/sagittal_mixed_pose_continuation_001_eval"
panel = root / "data/contextual_localization_102_fresh_dev_panel_001"
coronal = root / "data/joint_v7_allen_fullcanvas_192_001"
sagittal = root / "data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008"
availability = root / "data/allen_sagittal_ish_expansion_002_dev2_availability_20261008/manifest.json"
catalogue = root / "data/allen_sagittal_ish_expansion_002_20261008/manifest.json"
out = root / "runs/sagittal_mixed_pose_fitted_dev_readout_001"
repo = Path(__file__).resolve().parents[1]
protocol = repo / "docs/publication/SAGITTAL_MIXED_POSE_FITTED_DEV_READOUT_001_PROTOCOL_20261008.md"
checkpoint_path = run / "joint_step_0653.pt"
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

source_expected = {
    "arbitrary_plane_one_shot_model.py": "4a45d3b1b1f188eb93ca59e9480b8b2d6057999e8e8940f5fcdd1990539551ec",
    "arbitrary_plane_full_frame_primitives.py": "a998ddb3917cc860a22b74c98887a339b6c9e1024d5163ab43ad3324c5c01dd4",
    "arbitrary_plane_geometry.py": "432f2c58fff0c5df033072bee42ee49e6d5cf39391edd7891b15260318753030",
    "arbitrary_plane_allen_atlas_binding_v6.py": "b13901fe8e236387757cb44e9d44eaff98e47c4b461f51ee12506fa19d590e0b",
    "arbitrary_plane_joint_model_v7.py": "0d4f238c2320e00dc7316b285df5a7a591200f20589ac8c95ee6b2bde4cca4fa",
    "arbitrary_plane_recurrent_model.py": "90f961b1954236e9c0e39b33db46a5afc57d758859c34ff1615cf9c5083289f3",
    "arbitrary_plane_ribbon_v6.py": "d556936140462747f67e4881d3629d435218d9e22696ab99fa6856794abd3f1f",
}
for name, digest in source_expected.items():
    assert sha(repo / "training" / name) == digest, name
completed = json.loads((run / "completed.json").read_text())
assert completed["updates"] == 653 and not completed["calibrated"]
assert completed["checkpoint_sha256"]["653"] == sha(checkpoint_path) == "94b9c86f3a9cc15cd5b061acc6b67df4e0690ccdf5f9637a03c9cddd79f73ed8"
assert completed["config_sha256"] == sha(run / "config.json")
assert sha(prior_eval / "summary.json") == "f3a95a02578d506f5489e6a99b8f4e027715dc2f01ea0eddd72809fa18d50f8b"
assert json.loads((prior_eval / "summary.json").read_text())["selected_step"] == 653

panel_done = json.loads((panel / "completed.json").read_text())
assert (panel_done["physical_sections"], panel_done["eligible"], panel_done["synthetic_subjects"]) == (256, 241, 8)
assert sha(panel / "completed.json") == "9f4e0342ac3c29c56cc0f3c7c8d9974299dc51fc360be5a87784f4719b4de8e2"
assert sha(panel / "records.jsonl") == panel_done["records_sha256"] == "4a786fff6ca8cf23d48a3fe947e55170612a7ed814c62639b2abcb71e513a04c"
assert sha(panel / "protocol.json") == panel_done["protocol_sha256"]
assert sha(root / "data/contextual_localization_102_fresh_dev_plans_001/completed.json") == panel_done["plan_completed_sha256"]
panel_protocol = json.loads((panel / "protocol.json").read_text())
for name, digest in panel_protocol["source_sha256"].items():
    assert sha(panel / "source" / name) == digest, name
panel_records = [json.loads(line) for line in (panel / "records.jsonl").read_text().splitlines()]
assert len(panel_records) == 256 and sum(row["eligible"] for row in panel_records) == 241
assert len({row["animal_id"] for row in panel_records}) == 8
for record in panel_records:
    assert sha(panel / record["file"]) == record["sha256"]
synthetic = [row for row in panel_records if row["eligible"]]
support_edges = np.quantile([row["valid_pixels"] / 65536 for row in synthetic], [.25, .5, .75]).tolist()

coronal_done = json.loads((coronal / "completed.json").read_text())
assert coronal_done["development_images"] == 64 and coronal_done["development_donors"] == 6
for name, digest in coronal_done["output_sha256"].items():
    assert sha(coronal / name) == digest, name
coronal_records = [row for row in map(json.loads, (coronal / "records.jsonl").open()) if row["training_split"] == "development"]
assert len(coronal_records) == 64 and len({row["animal_id"] for row in coronal_records}) == 6
coronal_images = np.load(coronal / "images.npy", mmap_mode="r")
with np.load(coronal / "geometry.npz", allow_pickle=False) as arrays:
    coronal_affines = arrays["model_pixel_to_ap_dv_ml_um"].copy()
    coronal_thickness = arrays["thickness_um"].copy()

sagittal_summary = json.loads((sagittal / "summary.json").read_text())
for name, digest in sagittal_summary["output_sha256"].items():
    assert sha(sagittal / name) == digest, name
assert sha(availability) == sagittal_summary["source_availability_manifest_sha256"] == "8cec8d29c4d177073e58b8bc2871e849c1051620a4d00eb88d636017f402066d"
assert sha(catalogue) == sagittal_summary["source_metadata_manifest_sha256"] == "b4f77e00325273f937e4327b52d8b3f7db95458c130d4f88f682a8a08323f7bd"
assert sagittal_summary["original_section_count"] == 159 and sagittal_summary["source_available_section_count"] == 158
assert sagittal_summary["unavailable_source_section_ids_excluded"] == [101345593]
sagittal_records = [json.loads(line) for line in (sagittal / "geometry.jsonl").read_text().splitlines()]
assert len(sagittal_records) == 158 and len({row["donor_id"] for row in sagittal_records}) == 8
assert [row["array_row_index"] for row in sagittal_records] == list(range(158))
sagittal_images = np.load(sagittal / "model_input.npy", mmap_mode="r")
assert sagittal_images.shape == (158, 1, 256, 256)
assert sha(allen.TEMPLATE_PATH_V6) == allen.TEMPLATE_RAW_SHA256_V6
assert sha(allen.ANNOTATION_PATH_V6) == allen.ANNOTATION_RAW_SHA256_V6

atlas = torch.from_numpy(allen._decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
assert checkpoint["step"] == 653 and not checkpoint["calibrated"]
assert checkpoint["config"] == json.loads((run / "config.json").read_text())
model.load_state_dict(checkpoint["model"], strict=True)
del checkpoint
pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.], [127.5, 127.5]], device="cuda")
out.mkdir(parents=True, exist_ok=False)
rows = []

with torch.inference_mode(), (out / "rows.jsonl").open("w", encoding="utf-8") as stream:
    for domain, records in (("synthetic", synthetic), ("coronal", coronal_records), ("sagittal", sagittal_records)):
        for record in records:
            if domain == "synthetic":
                with np.load(panel / record["file"], allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays["inputs"][None].copy()).cuda()
                    target = torch.from_numpy(arrays["target_centre_um"].copy()).cuda()
                    valid = torch.from_numpy(arrays["valid_mask"].copy()).cuda().bool()
                    offsets = torch.from_numpy(arrays["offsets_um"][None].copy()).cuda()
                    weights = torch.from_numpy(arrays["weights"][None].copy()).cuda()
                assert int(valid.sum()) == record["valid_pixels"]
            else:
                i = record["array_row_index"]
                if domain == "coronal":
                    native = np.concatenate((np.asarray(coronal_images[i], dtype=np.float32),
                                             np.zeros((4, 192, 192), dtype=np.float32)))[None]
                    image = F.interpolate(torch.from_numpy(native).cuda(), (256, 256), mode="bilinear", align_corners=False)
                    affine = torch.as_tensor(coronal_affines[i], device="cuda", dtype=torch.float32)
                    reference = affine[:, 2] + (192 / 256) * pixels[:, :1] * affine[:, 0] + (192 / 256) * pixels[:, 1:] * affine[:, 1]
                    thickness = float(coronal_thickness[i])
                else:
                    native = np.concatenate((np.asarray(sagittal_images[i], dtype=np.float32),
                                             np.zeros((4, 256, 256), dtype=np.float32)))[None]
                    image = torch.from_numpy(native).cuda()
                    affine = torch.as_tensor(record["model_pixel_to_ccf_ref9_ap_dv_ml_um"], device="cuda", dtype=torch.float32)
                    reference = affine[:, 2] + pixels[:, :1] * affine[:, 0] + pixels[:, 1:] * affine[:, 1]
                    thickness = 50.0
                offsets = torch.linspace(-.5, .5, 9, device="cuda")[None] * thickness
                weights = torch.ones_like(offsets)
                weights[:, [0, -1]] = .5
                weights /= weights.sum(-1, keepdim=True)

            prediction = model.predict(image)
            prior = (prediction["log_mass"][0, :, None] + torch.stack((
                F.logsigmoid(-prediction["reflection_logit"][0]),
                F.logsigmoid(prediction["reflection_logit"][0])), -1)).flatten()
            choice = prior.topk(8).indices
            states = prediction["state"][:, choice // 2]
            reflected = (choice % 2)[None]
            selected = {**prediction, "state": states,
                        "log_mass": prediction["log_mass"][:, choice // 2],
                        "reflection_logit": prediction["reflection_logit"][:, choice // 2]}
            index = torch.arange(8, device="cuda")[None]
            first = model.map(selected, offsets, index, reflected, (64, 64), atlas, weights,
                              return_refinement_feature=True, feature_side=64, source_shape=(256, 256))
            refined_state, delta, _ = model.refine(first["refinement_feature"], states)
            refined = {**selected, "state": refined_state}
            fitted_map = model.map(refined, offsets, index, reflected, (96, 96), atlas, weights,
                                   feature_side=96, source_shape=(256, 256))
            fitted_score = model.score_fitted_candidates(image, refined, fitted_map, atlas, weights)[0] + delta[0]
            fitted_index = int(fitted_score.argmax())
            direct_branch = int(choice[0])
            fitted_branch = int(choice[fitted_index])
            del first, fitted_map

            if domain == "synthetic":
                direct_errors = torch.empty(8, device="cuda")
                for start in range(0, 8, 2):
                    sub = choice[start:start + 2]
                    mapped = model.map(prediction, offsets, sub[None] // 2, sub[None] % 2,
                                       (256, 256), atlas, weights)
                    direct_errors[start:start + 2] = (mapped["centre_surface_ccf_ap_dv_ml_um"][0, :, valid]
                                                      - target[valid][None]).norm(dim=-1).mean(-1)
                    del mapped
                local_indices = torch.tensor([[0, fitted_index]], device="cuda")
                fitted_final = model.map(refined, offsets, local_indices,
                                         reflected[:, local_indices[0]], (256, 256), atlas, weights)
                refined_top1_error = (fitted_final["centre_surface_ccf_ap_dv_ml_um"][0, 0, valid]
                                      - target[valid]).norm(dim=-1).mean()
                fitted_error = (fitted_final["centre_surface_ccf_ap_dv_ml_um"][0, 1, valid]
                                - target[valid]).norm(dim=-1).mean()
                normal = np.abs(np.asarray(record["plane_normal_ap_dv_ml"], dtype=np.float64))
                row = {"set": domain, "animal_id": record["animal_id"],
                       "specimen_id": record["specimen_id"], "experiment_id": record["experiment_id"],
                       "section_id": record["section_id"], "file_sha256": record["sha256"],
                       "appearance_mode": record["appearance_mode"],
                       "nearest_axis": ("AP", "DV", "ML")[int(normal.argmax())],
                       "valid_pixels": record["valid_pixels"],
                       "support_quartile": int(np.searchsorted(support_edges, record["valid_pixels"] / 65536, side="right") + 1),
                       "direct_mapped_um": float(direct_errors[0]),
                       "fitted_mapped_um": float(fitted_error),
                       "refined_top1_mapped_um": float(refined_top1_error),
                       "oracle_best8_direct_mapped_um": float(direct_errors.min()),
                       "all_top8_direct_mapped_um": direct_errors.cpu().tolist()}
                del fitted_final
            else:
                direct_points = points(states, reflected, pixels)
                fitted_points = points(refined_state, reflected, pixels)
                direct_errors = (direct_points - reference).norm(dim=-1).mean(-1)[0]
                fitted_errors = (fitted_points - reference).norm(dim=-1).mean(-1)[0]
                row = {"set": domain, "identity": record["animal_id"] if domain == "coronal" else record["donor_id"],
                       "specimen_id": record["specimen_id"], "experiment_id": record["experiment_id"],
                       "section_id": record["section_id"], "fit_thickness_um": thickness,
                       "direct_five_point_um": float(direct_errors[0]),
                       "fitted_five_point_um": float(fitted_errors[fitted_index]),
                       "refined_top1_five_point_um": float(fitted_errors[0]),
                       "oracle_best8_direct_five_point_um": float(direct_errors.min())}
            row.update({"direct_branch": direct_branch, "fitted_branch": fitted_branch,
                        "branch_changed": direct_branch != fitted_branch,
                        "top8_branches": choice.cpu().tolist(),
                        "top8_prior_log_mass": prior[choice].cpu().tolist(),
                        "top8_fitted_score": fitted_score.cpu().tolist()})
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
        stream.flush()
        print(json.dumps({"set": domain, "rows": len(records)}), flush=True)

summary = {}
for domain, direct_name, fitted_name, oracle_name in (
    ("synthetic", "direct_mapped_um", "fitted_mapped_um", "oracle_best8_direct_mapped_um"),
    ("coronal", "direct_five_point_um", "fitted_five_point_um", "oracle_best8_direct_five_point_um"),
    ("sagittal", "direct_five_point_um", "fitted_five_point_um", "oracle_best8_direct_five_point_um")):
    group = [row for row in rows if row["set"] == domain]
    identity_key = "animal_id" if domain == "synthetic" else "identity"
    by_identity = {}
    for identity in sorted({row[identity_key] for row in group}):
        subset = [row for row in group if row[identity_key] == identity]
        by_identity[str(identity)] = {"sections": len(subset),
                                     **{name: float(np.mean([row[name] for row in subset]))
                                        for name in (direct_name, fitted_name, oracle_name)}}
    summary[domain] = {"sections": len(group), "identities": len(by_identity),
                       "by_identity": by_identity,
                       "identity_equal_um": {name: float(np.mean([v[name] for v in by_identity.values()]))
                                             for name in (direct_name, fitted_name, oracle_name)},
                       "branch_changes": sum(row["branch_changed"] for row in group)}

synthetic_rows = [row for row in rows if row["set"] == "synthetic"]
strata = {}
for field in ("appearance_mode", "nearest_axis", "support_quartile"):
    strata[field] = {}
    for value in sorted({row[field] for row in synthetic_rows}):
        group = [row for row in synthetic_rows if row[field] == value]
        plans = sorted({row["animal_id"] for row in group})
        by_plan = {identity: {name: float(np.mean([row[name] for row in group if row["animal_id"] == identity]))
                              for name in ("direct_mapped_um", "fitted_mapped_um", "oracle_best8_direct_mapped_um")}
                   for identity in plans}
        strata[field][str(value)] = {"sections": len(group), "plans": len(plans),
                                     "plan_equal_um": {name: float(np.mean([v[name] for v in by_plan.values()]))
                                                       for name in ("direct_mapped_um", "fitted_mapped_um",
                                                                    "oracle_best8_direct_mapped_um")}}
summary["synthetic"]["strata"] = strata
summary["synthetic"]["support_quartile_edges"] = support_edges

synthetic_gain = (summary["synthetic"]["identity_equal_um"]["direct_mapped_um"]
                  - summary["synthetic"]["identity_equal_um"]["fitted_mapped_um"])
plans_improved = sum(v["fitted_mapped_um"] < v["direct_mapped_um"]
                     for v in summary["synthetic"]["by_identity"].values())
strata_worsening = {field: {value: metric["plan_equal_um"]["fitted_mapped_um"]
                                  - metric["plan_equal_um"]["direct_mapped_um"]
                            for value, metric in strata[field].items()}
                    for field in ("appearance_mode", "nearest_axis")}
real_worsening = {domain: summary[domain]["identity_equal_um"]["fitted_five_point_um"]
                         - summary[domain]["identity_equal_um"]["direct_five_point_um"]
                  for domain in ("coronal", "sagittal")}
gate = {"synthetic_plan_equal_gain_um": synthetic_gain,
        "synthetic_plans_improved": plans_improved,
        "strata_worsening_um": strata_worsening,
        "real_donor_equal_worsening_um": real_worsening,
        "passes": bool(synthetic_gain >= 200 and plans_improved >= 6
                       and all(worsening <= 100 for values in strata_worsening.values() for worsening in values.values())
                       and all(worsening <= 200 for worsening in real_worsening.values()))}
result = {"version": "sagittal-mixed-pose-fitted-dev-readout-001", "summary": summary,
          "predeclared_gate": gate,
          "synthetic_original_sections": 256, "synthetic_eligible_sections": 241,
          "sagittal_original_sections": 159, "sagittal_source_available_sections": 158,
          "sagittal_unavailable_section_ids": [101345593],
          "sagittal_assumed_fit_thickness_um": 50.0,
          "scope": "inherited mechanism readout; synthetic plans are not animals and real affines are weak labels; no holdout, calibration, public benchmark or GUI release",
          "input_sha256": {"checkpoint": sha(checkpoint_path), "run_completed": sha(run / "completed.json"),
                           "run_config": sha(run / "config.json"),
                           "prior_selection_summary": sha(prior_eval / "summary.json"),
                           "panel_completed": sha(panel / "completed.json"),
                           "panel_protocol": sha(panel / "protocol.json"),
                           "panel_records": sha(panel / "records.jsonl"),
                           "panel_plan_completed": sha(root / "data/contextual_localization_102_fresh_dev_plans_001/completed.json"),
                           "panel_npz": {row["file"]: row["sha256"] for row in panel_records},
                           "panel_source_snapshots": panel_protocol["source_sha256"],
                           "coronal_completed": sha(coronal / "completed.json"),
                           "coronal_outputs": coronal_done["output_sha256"],
                           "sagittal_summary": sha(sagittal / "summary.json"),
                           "sagittal_outputs": sagittal_summary["output_sha256"],
                           "sagittal_source_availability": sha(availability),
                           "sagittal_catalogue": sha(catalogue),
                           "atlas_template_raw": sha(allen.TEMPLATE_PATH_V6),
                           "atlas_annotation_raw": sha(allen.ANNOTATION_PATH_V6),
                           "model_sources": source_expected,
                           "protocol": sha(protocol), "evaluator": sha(Path(__file__))},
          "rows_sha256": sha(out / "rows.jsonl"), "calibrated": False,
          "public_benchmark_used": False}
(out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
(out / "completed.json").write_text(json.dumps({"rows": len(rows),
    "summary_sha256": sha(out / "summary.json"), "rows_sha256": sha(out / "rows.jsonl"),
    "checkpoint_sha256": sha(checkpoint_path), "protocol_sha256": sha(protocol),
    "evaluator_sha256": sha(Path(__file__)), "calibrated": False,
    "public_benchmark_used": False}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"event": "complete", "gate": gate, "rows_sha256": result["rows_sha256"]}, indent=2), flush=True)
