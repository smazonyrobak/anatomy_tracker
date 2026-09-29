"""Post-exit top128 geometry diagnosis; oracle selectors, never a frozen gate."""

import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "4"
sys.dont_write_bytecode = True

import hashlib
import json
import shutil

import numpy as np
import torch

RUN = ROOT / "runs/joint_v6_imagekey_retrieval_001"
AUDIT = ROOT / "runs/joint_v6_imagekey_retrieval_001_independent_audit/audit.json"
DEV = ROOT / "runs/joint_v6_proposal_substantive_001/internal_development_prepared.pt"
OUTPUT = ROOT / "runs/joint_v6_imagekey_frame_capture_posthoc_001"
AUDIT_SHA256 = "f9eb9c6845e048e5fc3ca840a4c5effcf48c1d6ed98afef9daa65a3983e4ca9b"
torch.set_num_threads(4)
assert hashlib.sha256(AUDIT.read_bytes()).hexdigest() == AUDIT_SHA256
audit = json.loads(AUDIT.read_text())
assert audit["integrity_passed"] and audit["advance_to_joint_integration_pilot"]
inputs = (RUN / "catalogue.pt", RUN / "development_rows_step_04000.npz", DEV)
input_hashes = {str(AUDIT): AUDIT_SHA256}
for path in inputs:
    with path.open("rb") as stream:
        input_hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    assert input_hashes[str(path)] == audit["artifact_sha256"][str(path)]
catalogue = torch.load(inputs[0], map_location="cpu", weights_only=False)
rows = np.load(inputs[1], allow_pickle=False)
dev = torch.load(DEV, map_location="cpu", weights_only=False, mmap=True)
top = rows["top128_cell_index"]
groups = np.array([record["animal_id"] for record in dev["records"]])
modes = np.array([record["selected_mode"] for record in dev["records"]])
eligible = dev["weight"].numpy() > 0
support_origin = np.asarray(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"], dtype=np.float64)
geometry = {}
for key, state in (("truth", dev["truth_state"].numpy()), ("cell", catalogue["arrays"]["cell_states_float64"])):
    state = np.asarray(state, dtype=np.float64)
    u = state[:, 3:6] / np.linalg.norm(state[:, 3:6], axis=-1, keepdims=True)
    v = state[:, 6:9] - (state[:, 6:9] * u).sum(-1, keepdims=True) * u
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    normal = np.cross(u, v)
    edge_u = u * np.exp(state[:, 9:10])
    edge_v = (state[:, 11:12] * u + v) * np.exp(state[:, 10:11])
    origin = state[:, :3] - .5 * (edge_u + edge_v)
    geometry[key] = {"centre": state[:, :3], "u": u, "v": v, "normal": normal,
                     "edge_u": edge_u, "edge_v": edge_v,
                     "corners": np.stack((origin, origin + edge_u, origin + edge_v, origin + edge_u + edge_v), axis=1),
                     "offset": ((state[:, :3] - support_origin) * normal).sum(-1)}
truth = {key: value[:, None] for key, value in geometry["truth"].items()}
candidate = {key: value[top] for key, value in geometry["cell"].items()}
dot = (candidate["normal"] * truth["normal"]).sum(-1)
angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(candidate["normal"], truth["normal"]), axis=-1), np.abs(dot)))
offset = np.abs(candidate["offset"] - np.where(dot < 0, -1., 1.) * truth["offset"])
centre_delta = candidate["centre"] - truth["centre"]
centre_mse = np.square(centre_delta).sum(-1)
edge_u_mse_plus = .25 * np.square(candidate["edge_u"] - truth["edge_u"]).sum(-1)
edge_u_mse_minus = .25 * np.square(-candidate["edge_u"] - truth["edge_u"]).sum(-1)
corner_sign = np.where(edge_u_mse_minus < edge_u_mse_plus, -1., 1.)
edge_u_mse = np.minimum(edge_u_mse_plus, edge_u_mse_minus)
edge_v_mse = .25 * np.square(candidate["edge_v"] - truth["edge_v"]).sum(-1)
corner = np.sqrt(centre_mse + edge_u_mse + edge_v_mse)
explicit_corner = np.sqrt(np.minimum(np.square(candidate["corners"] - truth["corners"]).sum(-1).mean(-1), np.square(candidate["corners"][..., [1, 0, 3, 2], :] - truth["corners"]).sum(-1).mean(-1)))
assert np.max(np.abs(corner - explicit_corner)) < 1e-8
near = (angle <= 10.) & (offset <= 500.)

# Minimal rotation transports the sign-aligned normal onto the truth normal;
# the remaining signed angle of u is roll, not an anatomical left/right flip.
normal_sign = np.where(dot < 0, -1., 1.)
aligned_n = candidate["normal"] * normal_sign[..., None]
aligned_u = candidate["u"] * normal_sign[..., None]
axis = np.cross(aligned_n, truth["normal"])
transported_u = aligned_u + np.cross(axis, aligned_u) + np.cross(axis, np.cross(axis, aligned_u)) / (1. + np.abs(dot))[..., None]
roll = np.degrees(np.arctan2((transported_u * truth["v"]).sum(-1), (transported_u * truth["u"]).sum(-1)))
normal_shift = (centre_delta * truth["normal"]).sum(-1)
inplane_shift = np.sqrt(np.maximum(centre_mse - normal_shift**2, 0.))
inplane_u_shift = (centre_delta * truth["u"]).sum(-1)
inplane_v_shift = (centre_delta * truth["v"]).sum(-1)
span_u = np.linalg.norm(candidate["edge_u"], axis=-1) / np.linalg.norm(truth["edge_u"], axis=-1)
span_v = np.linalg.norm(candidate["edge_v"], axis=-1) / np.linalg.norm(truth["edge_v"], axis=-1)
candidate_cos = (candidate["edge_u"] * candidate["edge_v"]).sum(-1) / (np.linalg.norm(candidate["edge_u"], axis=-1) * np.linalg.norm(candidate["edge_v"], axis=-1))
truth_cos = (truth["edge_u"] * truth["edge_v"]).sum(-1) / (np.linalg.norm(truth["edge_u"], axis=-1) * np.linalg.norm(truth["edge_v"], axis=-1))
shear_error = np.degrees(np.arcsin(np.clip(candidate_cos * corner_sign, -1, 1)) - np.arcsin(np.clip(truth_cos, -1, 1)))
raw = {"top128_cell_index": top, "eligible": eligible, "group": groups, "mode": modes,
       "candidate_angle_deg": angle, "candidate_offset_error_um": offset,
       "candidate_corner_rms_um": corner, "candidate_plane_captured": near,
       "candidate_roll_signed_deg": roll, "candidate_inplane_shift_um": inplane_shift,
       "candidate_inplane_u_shift_um": inplane_u_shift, "candidate_inplane_v_shift_um": inplane_v_shift,
       "candidate_normal_shift_um": normal_shift, "candidate_span_u_ratio": span_u,
       "candidate_span_v_ratio": span_v, "candidate_shear_angle_error_deg": shear_error,
       "candidate_corner_horizontal_sign": corner_sign,
       "candidate_centre_mse_um2": centre_mse, "candidate_edge_u_mse_um2": edge_u_mse,
       "candidate_edge_v_mse_um2": edge_v_mse}
per_row = {}
row_index = np.arange(len(top))
for k in (32, 128):
    minimum = corner[:, :k].min(-1)
    per_row[f"top{k}_minimum_corner_rms_um"] = minimum
    per_row[f"top{k}_plane_capture"] = near[:, :k].any(-1).astype(float)
    for threshold in (1000, 2000, 3000, 5000):
        per_row[f"top{k}_corner_capture_{threshold}um"] = (minimum <= threshold).astype(float)
        per_row[f"top{k}_joint_plane_corner_capture_{threshold}um"] = (near[:, :k] & (corner[:, :k] <= threshold)).any(-1).astype(float)
    selectors = {"best_corner": corner[:, :k].argmin(-1),
                 "best_plane_score": ((angle[:, :k] / 10.)**2 + (offset[:, :k] / 500.)**2).argmin(-1),
                 "near_plane_best_corner": np.where(near[:, :k], corner[:, :k], np.inf).argmin(-1)}
    for name, index in selectors.items():
        valid = near[:, :k].any(-1) if name == "near_plane_best_corner" else np.ones(len(top), dtype=bool)
        prefix = f"top{k}_{name}"
        raw[f"{prefix}_rank"] = np.where(valid, index + 1, -1)
        raw[f"{prefix}_cell_index"] = np.where(valid, top[row_index, index], -1)
        fields = {"rank": index + 1, "corner_rms_um": corner[row_index, index],
                  "angle_deg": angle[row_index, index], "offset_error_um": offset[row_index, index],
                  "roll_abs_deg": np.abs(roll[row_index, index]), "inplane_shift_um": inplane_shift[row_index, index],
                  "normal_shift_abs_um": np.abs(normal_shift[row_index, index]),
                  "span_u_ratio": span_u[row_index, index], "span_v_ratio": span_v[row_index, index],
                  "span_u_abs_pct_error": np.abs(span_u[row_index, index] - 1) * 100,
                  "span_v_abs_pct_error": np.abs(span_v[row_index, index] - 1) * 100,
                  "shear_abs_deg": np.abs(shear_error[row_index, index]),
                  "centre_mse_um2": centre_mse[row_index, index], "edge_u_mse_um2": edge_u_mse[row_index, index],
                  "edge_v_mse_um2": edge_v_mse[row_index, index],
                  "if_centre_corrected_corner_rms_um": np.sqrt(edge_u_mse[row_index, index] + edge_v_mse[row_index, index])}
        for field, value in fields.items():
            per_row[f"{prefix}_{field}"] = np.where(valid, value, np.nan)
scopes = {"all": np.ones(len(top), dtype=bool), "support_eligible": eligible, "censored": ~eligible,
          **{f"{mode}/support_eligible": eligible & (modes == mode) for mode in np.unique(modes)}}
summary = {}
for scope, mask in scopes.items():
    result = {"rows": int(mask.sum()), "groups": len(np.unique(groups[mask])), "metrics": {}}
    for name, value in per_row.items():
        finite = mask & np.isfinite(value)
        by_group = [float(value[finite & (groups == group)].mean()) for group in np.unique(groups[finite])]
        result["metrics"][name] = {"finite_rows": int(finite.sum()), "group_macro_mean": float(np.mean(by_group)) if by_group else None,
            "row_quantiles_05_25_50_75_90_95": np.quantile(value[finite], (.05, .25, .5, .75, .9, .95)).tolist() if finite.any() else None}
    summary[scope] = result
for k in (32, 128):
    for name, frozen_name in ((f"top{k}_plane_capture", f"physical_plane_capture_at_{k}"), (f"top{k}_corner_capture_1000um", f"frame_corner_capture_at_{k}")):
        assert abs(summary["support_eligible"]["metrics"][name]["group_macro_mean"] - audit["aggregates"]["support_eligible"]["group_macro"][frozen_name]) < 1e-12
OUTPUT.mkdir(parents=True, exist_ok=False)
shutil.copyfile(Path(__file__), OUTPUT / "diagnostic_source.py")
np.savez(OUTPUT / "rows.npz", **raw, **{f"metric_{key}": value for key, value in per_row.items()})
protocol = {"scope": "posthoc completed-top128 oracle geometry diagnosis, not a frozen gate, predictor or benchmark",
    "input_sha256": input_hashes, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "row_source": "all640 original internal-development rows;611 original positive-weight rows; unchanged organizational animal groups",
    "corner_convention": "exact independent-audit OUV edge-chart corners (s,t) in {0,1}; min identity/horizontal pixel correspondence, not anatomical ML symmetry; no change to frozen gate",
    "best_plane_score": "oracle argmin(angle/10deg)^2+(sign-aligned-offset/500um)^2 within saved topK",
    "near_plane_best_corner": "oracle minimum corner RMS among saved candidates satisfying angle<=10deg AND sign-aligned offset<=500um; missing rows excluded only from this conditional diagnostic",
    "roll": "sign-align normal by proper antipodal u,n flip, then minimal-rotation parallel transport of candidate u to truth normal; signed residual in truth u,v frame",
    "shift": "chart-centre difference resolved into truth in-plane and normal components; not identical to sign-aligned plane offset error when normals differ",
    "corner_mse_identity": "RMS^2=centre_difference^2 + edge_u_difference^2/4 + edge_v_difference^2/4; horizontal correspondence minimizes u term",
    "statistics": "rates and means macro-average organizational groups after eligibility/mode filtering; quantiles are explicitly pooled row quantiles; near-plane diagnostics condition on candidate existence",
    "no_model_or_gallery_reembedding": True, "rows_sha256": hashlib.sha256((OUTPUT / "rows.npz").read_bytes()).hexdigest()}
(OUTPUT / "summary.json").write_text(json.dumps({"protocol": protocol, "scopes": summary}, indent=2, allow_nan=False), encoding="utf-8")
for k in (32, 128):
    metrics = summary["support_eligible"]["metrics"]
    keys = [f"top{k}_minimum_corner_rms_um", f"top{k}_plane_capture"] + [f"top{k}_corner_capture_{t}um" for t in (1000, 2000, 3000, 5000)]
    keys += [f"top{k}_near_plane_best_corner_{field}" for field in ("corner_rms_um", "roll_abs_deg", "inplane_shift_um", "span_u_abs_pct_error", "span_v_abs_pct_error", "centre_mse_um2", "edge_u_mse_um2", "edge_v_mse_um2")]
    print(json.dumps({"topk": k, "eligible": {key: metrics[key] for key in keys}}), flush=True)
print(json.dumps({"output": str(OUTPUT), "source_sha256": protocol["source_sha256"], "rows_sha256": protocol["rows_sha256"]}), flush=True)
