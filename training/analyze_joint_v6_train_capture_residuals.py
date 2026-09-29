"""TRAIN-only catalogue-start residuals; saved GPU top128 order, no model inference."""
import os
import sys
from pathlib import Path
import hashlib
import json

ROOT = Path(r"I:\AnatomyTracker")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
sys.dont_write_bytecode = True
AUDIT_SHA256 = "c75c4ff817e7603eca31c8086080aafdb7f0953b3fec58c3f3f2f7842cf2afb3"
COMPLETION_SHA256 = "576b85c7bcdbeea9e361b15908e320fe0d838291d083c945d19ede2b1687625e"
READY_AFTER_ROOT_REVIEW = True
assert READY_AFTER_ROOT_REVIEW and len(AUDIT_SHA256) == len(COMPLETION_SHA256) == 64

import numpy as np
import torch

torch.set_num_threads(2)
RUN = ROOT / "runs/joint_v6_full_coverage_ranking_001"
AUDIT = ROOT / "runs/joint_v6_full_coverage_ranking_001_independent_audit/audit.json"
OUT = ROOT / "runs/joint_v6_train_capture_residuals_001"
PREPARED = ROOT / "runs/joint_v6_proposal_substantive_001/training_prepared.pt"
CACHE = ROOT / "runs/arbitrary_plane_finite_v6_substantive_data_001/training_cache"
MANIFEST_SHA256 = "97fd046d29d7623d55e7e9c0d403f50002e50b3f4e668757af450fa561e0b829"
SELECTIONS = ("top1_predicted_R", "first_near_cell_predicted_R", "best_frame_cell_and_R_truth_assisted")
COMPONENTS = ("normal_tangent_u_rad", "normal_tangent_v_rad", "plane_offset_um", "transported_roll_rad", "translation_u_um", "translation_v_um", "delta_log_basis_u", "delta_log_basis_v", "delta_shear")
LARGE = np.array([.15, .15, 600., .20, 600., 600., .12, .12, .12])
NECESSARY_THREE_STEP_LIMITS = np.array([3 * np.hypot(.18, .18), 1800., .36, .36])
F = 95 / 96
hashes = {}


def sha(path):
    path = Path(path)
    with path.open("rb") as stream:
        hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes[str(path)]


def state_ouv(state):
    state = np.asarray(state, dtype=np.float64)
    u = state[..., 3:6] / np.linalg.norm(state[..., 3:6], axis=-1, keepdims=True)
    v = state[..., 6:9] - np.sum(state[..., 6:9] * u, axis=-1, keepdims=True) * u
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    U = np.exp(state[..., 9, None]) * u
    V = np.exp(state[..., 10, None]) * (v + state[..., 11, None] * u)
    return np.stack((state[..., :3] - .5 * (U + V), U, V), axis=-2)


def reflected(ouv, horizontal):
    out = ouv.copy()
    out[..., 0, :] += np.asarray(horizontal)[..., None] * F * ouv[..., 1, :]
    out[..., 1, :] *= np.where(horizontal, -1., 1.)[..., None]
    return out


def parts(ouv):
    U, V = ouv[..., 1, :], ouv[..., 2, :]
    a = np.linalg.norm(U, axis=-1)
    u = U / a[..., None]
    p = np.sum(u * V, axis=-1)
    v = V - p[..., None] * u
    b = np.linalg.norm(v, axis=-1)
    v /= b[..., None]
    return ouv[..., 0, :] + .5 * (U + V), np.stack((u, v, np.cross(u, v)), axis=-1), a, b, p


def residual(start, target):
    # Exact inverse of archived compose_antipodal_plane_frame_residual, with R fixed.
    c0, R0, a0, b0, p0 = parts(start)
    ct, Rt, at, bt, pt = parts(target)
    n0, nt = R0[..., :, 2], Rt[..., :, 2]
    cross = np.cross(n0, nt)
    sine = np.linalg.norm(cross, axis=-1)
    cosine = np.clip(np.sum(n0 * nt, axis=-1), -1, 1)
    angle = np.arctan2(sine, cosine)
    ambiguous = (sine <= 1e-12) & (cosine < 0)
    axis = cross / np.maximum(sine[..., None], 1e-300)
    axis = np.where((sine <= 1e-12)[..., None], R0[..., :, 0], axis)
    tangent = np.cross(axis, n0) * angle[..., None]
    x, y, z = np.moveaxis(axis, -1, 0)
    zero = np.zeros_like(x)
    skew = np.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1).reshape(axis.shape[:-1] + (3, 3))
    Q = cosine[..., None, None] * np.eye(3) + (1 - cosine)[..., None, None] * axis[..., :, None] * axis[..., None, :] + sine[..., None, None] * skew
    d0 = np.sum((c0 - support_origin) * n0, axis=-1)
    dt = np.sum((ct - support_origin) * nt, axis=-1)
    post_c = support_origin + dt[..., None] * nt + np.einsum("...ij,...j->...i", Q, c0 - support_origin - d0[..., None] * n0)
    post_R = Q @ R0
    roll = np.arctan2(np.sum(Rt[..., :, 0] * post_R[..., :, 1], axis=-1), np.sum(Rt[..., :, 0] * post_R[..., :, 0], axis=-1))
    dc = ct - post_c
    result = np.stack((np.sum(tangent * R0[..., :, 0], axis=-1), np.sum(tangent * R0[..., :, 1], axis=-1), dt - d0, roll, np.sum(dc * post_R[..., :, 0], axis=-1), np.sum(dc * post_R[..., :, 1], axis=-1), np.log(at / a0), np.log(bt / b0), (pt * b0 / bt - p0) / a0), axis=-1)
    # Recompose O/U/V numerically, without importing any live model/geometry source.
    cr, sr = np.cos(roll)[..., None], np.sin(roll)[..., None]
    u = cr * post_R[..., :, 0] + sr * post_R[..., :, 1]
    v = -sr * post_R[..., :, 0] + cr * post_R[..., :, 1]
    a, b = a0 * np.exp(result[..., 6]), b0 * np.exp(result[..., 7])
    p = np.exp(result[..., 7]) * (a0 * result[..., 8] + p0)
    U, V = a[..., None] * u, p[..., None] * u + b[..., None] * v
    centre = post_c + result[..., 4, None] * post_R[..., :, 0] + result[..., 5, None] * post_R[..., :, 1]
    reconstructed = np.stack((centre - .5 * (U + V), U, V), axis=-2)
    return result, angle, ambiguous, float(np.max(np.abs(reconstructed - target)))


assert sha(AUDIT) == AUDIT_SHA256 and sha(RUN / "completed.json") == COMPLETION_SHA256
audit = json.loads(AUDIT.read_text())
assert audit["integrity_passed"] and all(audit["arm_audits"]["A"]["gates"].values())
completion = json.loads((RUN / "completed.json").read_text())
assert completion["arm_results"]["A"]["recovery_gate_passed"]
assert sha(PREPARED) == completion["experiment"]["prepared_source_sha256"]["training_prepared.pt"]
assert sha(RUN / "catalogue.pt") == completion["experiment"]["prepared_source_sha256"]["catalogue.pt"]
catalogue = torch.load(RUN / "catalogue.pt", map_location="cpu", weights_only=False)
cell_ouv = state_ouv(np.asarray(catalogue["arrays"]["cell_states_float64"]))
cell_centre = cell_ouv[:, 0] + .5 * F * (cell_ouv[:, 1] + cell_ouv[:, 2])
cell_normal = np.asarray(catalogue["arrays"]["cell_normal_ap_dv_ml_float64"])
support_origin = np.array(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"])
cell_offset = np.sum((cell_centre - support_origin) * cell_normal, axis=-1)
assert sha(CACHE / "manifest.json") == MANIFEST_SHA256
manifest = json.loads((CACHE / "manifest.json").read_text())
assert manifest["receipt_sha256"] == "af778499dd77ac13f673ce0778972218519f421b97fa01c4d055e00d1dd16266"
prepared = torch.load(PREPARED, map_location="cpu", weights_only=False, mmap=True)
assert all(record["split"] == "train" for record in prepared["records"])
assert all(record["row_receipt_sha256"] == frozen["training_row_receipt_sha256"] for record, frozen in zip(prepared["records"], manifest["rows"]))
assert len(prepared["records"]) == len(manifest["rows"]) == 5120
truth_reflection = np.array([row["reflection_state"] == "horizontal" for row in manifest["rows"]])
assert all(row["reflection_state"] in ("none", "horizontal") for row in manifest["rows"])
# The prepared state is canonical, not observed: generator stores reflection separately.
# Authenticate one recorded example of each convention; no image/NPZ regeneration.
reflection_examples = []
for flag in (False, True):
    record = manifest["rows"][int(np.flatnonzero(truth_reflection == flag)[0])]
    path = CACHE / record["metadata_relative_path"]
    assert sha(path) == record["metadata_file_sha256"]
    metadata = json.loads(path.read_text())
    canonical = np.array(metadata["canonical_effective_quicknii_ouv_float64"])
    observed = np.array(metadata["observed_effective_quicknii_ouv_float64"])
    assert np.allclose(reflected(canonical, np.array(flag)), observed, rtol=0, atol=1e-10)
    assert metadata["reflection_representation_index"] == int(flag)
    reflection_examples.append(str(path))
assert sha(RUN / "weak_affine_anchors.npz") == completion["experiment"]["schedule_sha256"]["weak_affine_anchors.npz"]
anchors = np.load(RUN / "weak_affine_anchors.npz", allow_pickle=False)
datasets = {"synthetic_train": (state_ouv(prepared["truth_state"].numpy()), truth_reflection), "allen_train": (anchors["physical_ouv_ap_dv_ml_um"], np.zeros(1280, dtype=bool))}
results = {}
recomposition_max = 0.
OUT.mkdir(parents=True, exist_ok=False)
(OUT / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
for dataset, (canonical_truth, source_reflection) in datasets.items():
    path = RUN / "A" / dataset / "rows.npz"
    assert sha(path) == audit["artifact_sha256"][str(path)]
    rows = np.load(path, allow_pickle=False)
    identities_path = RUN / ("synthetic_training_identities.json" if dataset == "synthetic_train" else "real_training_identities.json")
    assert sha(identities_path) == audit["artifact_sha256"][str(identities_path)]
    identities = json.loads(identities_path.read_text())
    assert all(row["split"] == ("train" if dataset == "synthetic_train" else "development_train") for row in identities)
    top = rows["top128_cell_index"]
    N = len(top)
    assert top.shape == (N, 128) and len(canonical_truth) == N
    truth = reflected(canonical_truth, source_reflection)
    cf = truth[:, 0] + .5 * F * (truth[:, 1] + truth[:, 2])
    _, Rt, _, _, _ = parts(truth)
    n = Rt[:, :, 2]
    normal_dot = np.sum(cell_normal[top] * n[:, None], axis=-1)
    plane_offset = np.sum((cf - support_origin) * n, axis=-1)
    offset_error = plane_offset[:, None] - np.where(normal_dot < 0, -1., 1.) * cell_offset[top]
    near = (np.abs(normal_dot) >= np.cos(np.deg2rad(10.))) & (np.abs(offset_error) <= 500.)
    no_near = ~near.any(axis=1)
    centre_delta = cf[:, None] - cell_centre[top]
    frame_squared = np.sum(centre_delta**2, axis=-1)[..., None] + .25 * F**2 * (np.sum((cell_ouv[top, 1, None, :] * np.array([1., -1.])[None, None, :, None] - truth[:, None, None, 1, :])**2, axis=-1) + np.sum((cell_ouv[top, 2] - truth[:, None, 2])**2, axis=-1)[..., None])
    best_flat = frame_squared.reshape(N, -1).argmin(axis=1)
    rank = np.stack((np.zeros(N, dtype=np.int64), near.argmax(axis=1), best_flat // 2), axis=1)
    representation = rows["top128_representation_probability"].argmax(axis=-1)[np.arange(N)[:, None], rank]
    representation[:, 2] = best_flat % 2
    available = np.ones((N, 3), dtype=bool); available[:, 1] = ~no_near
    index = top[np.arange(N)[:, None], rank]
    start = cell_ouv[index]
    observed_start = reflected(start, representation.astype(bool))
    target_canonical = reflected(np.broadcast_to(truth[:, None], start.shape), representation.astype(bool))
    correction, oriented_angle, ambiguous, error = residual(start, target_canonical)
    teacher, _, reverse_ambiguous, reverse_error = residual(target_canonical, start)
    recomposition_max = max(recomposition_max, error, reverse_error)
    assert recomposition_max < 1e-5
    actual_dc = cf[:, None] - cell_centre[index]
    normal_dc = np.sum(actual_dc * n[:, None], axis=-1)
    tangent_dc = np.einsum("nsc,nck->nsk", actual_dc, Rt[:, :, :2])
    delta_edges = observed_start[..., 1:, :] - truth[:, None, 1:, :]
    four_corner_rms = np.sqrt(frame_squared[np.arange(N)[:, None], rank, representation])
    dense_raster_rms = np.sqrt(np.sum(actual_dc**2, axis=-1) + (96**2 - 1) / (12 * 96**2) * np.sum(delta_edges**2, axis=(-1, -2)))
    selected_dot = normal_dot[np.arange(N)[:, None], rank]
    metrics = {"plane_angle_deg": np.rad2deg(np.arccos(np.clip(np.abs(selected_dot), 0, 1))), "signed_plane_offset_error_um": offset_error[np.arange(N)[:, None], rank], "finite_centre_normal_displacement_um": normal_dc, "finite_centre_tangent_u_displacement_um": tangent_dc[..., 0], "finite_centre_tangent_v_displacement_um": tangent_dc[..., 1], "finite_centre_tangent_distance_um": np.linalg.norm(tangent_dc, axis=-1), "oriented_branch_normal_angle_deg": np.rad2deg(oriented_angle), "finite_four_corner_rms_um": four_corner_rms, "dense_96x96_raster_rms_um": dense_raster_rms}
    metrics.update({f"correction:{key}": correction[..., i] for i, key in enumerate(COMPONENTS)})
    inverse_defined = available & ~(ambiguous | reverse_ambiguous)
    necessary_exceeded = np.stack((oriented_angle, np.abs(correction[..., 2]), np.abs(correction[..., 6]), np.abs(correction[..., 7])), axis=-1) > NECESSARY_THREE_STEP_LIMITS
    correction[~inverse_defined], teacher[~inverse_defined] = np.nan, np.nan
    for values in metrics.values(): values[~available] = np.nan
    eligible = rows["pose_supervision_weight"] > 0 if dataset == "synthetic_train" else rows["common_anchor_eligible"]
    mode = rows["selected_mode"] if dataset == "synthetic_train" else np.full(N, "raw")
    groups = rows["animal_id"]
    raw = {key: rows[key] for key in ("animal_id", "specimen_id", "experiment_id", "section_id")}
    raw["exact_inverse_defined"] = inverse_defined
    raw.update({key: rows[key] for key in (("pose_supervision_weight",) if dataset == "synthetic_train" else ("common_anchor_eligible", "matched_support_mass", "canonical_support_mass"))})
    raw.update(selection_available=available, selected_rank_zero_based=np.where(available, rank, -1), selected_cell_index=np.where(available, index, -1), selected_reflection_index=np.where(available, representation, -1), selected_representation_probability=np.where(available, rows["top128_representation_probability"][np.arange(N)[:, None], rank, representation], np.nan), best_reflection_for_selected_cell=np.where(available, frame_squared[np.arange(N)[:, None], rank].argmin(axis=-1), -1), correction_residual=correction, teacher_generation_direction_residual=teacher, teacher_LARGE_box_exceeded=np.abs(teacher) > LARGE, necessary_three_step_bound_exceeded=necessary_exceeded & available[..., None], antipodal_transport_ambiguous=ambiguous & available, source_record_horizontal_reflection=source_reflection, eligible=eligible, selected_mode=mode, no_near_cell_in_top128=no_near, plane_capture_at32=near[:, :32].any(axis=1), plane_capture_at128=~no_near, canonical_reference_ouv=canonical_truth, observed_reference_ouv=truth, **metrics)
    np.savez_compressed(OUT / f"{dataset}_residuals.npz", **raw)
    (OUT / f"{dataset}_identities.json").write_text(json.dumps(identities, indent=2), encoding="utf-8")
    scopes = {"all": np.ones(N, dtype=bool), "eligible": eligible, "censored": ~eligible}
    scopes.update({f"eligible_mode:{label}": eligible & (mode == label) for label in np.unique(mode)})
    summaries = {}
    for name, scope in scopes.items():
        summaries[name] = {"rows": int(scope.sum()), "groups": int(len(np.unique(groups[scope]))), "no_near_128_count": int((scope & no_near).sum()), "selections": {}}
        for j, label in enumerate(SELECTIONS):
            take = scope & available[:, j]
            teacher_take = take & inverse_defined[:, j]
            by_group = [take & (groups == group) for group in np.unique(groups[take])]
            statistics = {}
            for key, values in metrics.items():
                value = values[take, j]
                value = value[np.isfinite(value)]
                group_means = [np.abs(values[g & np.isfinite(values[:, j]), j]).mean() for g in by_group if np.any(g & np.isfinite(values[:, j]))]
                statistics[key] = {"finite_rows": len(value), "row_signed_q05_q50_q95": np.quantile(value, [.05, .5, .95]).tolist(), "row_abs_q50_q90_q95_max": np.quantile(np.abs(value), [.5, .9, .95, 1.]).tolist(), "group_macro_mean_abs": float(np.mean(group_means))} if len(value) else None
            summaries[name]["selections"][label] = {"rows": int(take.sum()), "metrics": statistics, "exact_inverse_defined_rows": int(teacher_take.sum()), "antipodal_transport_ambiguous_rows": int((take & ~inverse_defined[:, j]).sum()), "teacher_LARGE_box_exceedance_fraction_by_component": (np.abs(teacher[teacher_take, j]) > LARGE).mean(axis=0).tolist() if teacher_take.any() else None, "teacher_LARGE_any_component_exceedance_fraction": float((np.abs(teacher[teacher_take, j]) > LARGE).any(axis=1).mean()) if teacher_take.any() else None, "necessary_three_step_bound_exceedance_fraction": necessary_exceeded[take, j].mean(axis=0).tolist() if take.any() else None, "selected_R_disagrees_best_frame_R_fraction": float((representation[take, j] != raw["best_reflection_for_selected_cell"][take, j]).mean()) if take.any() else None}
    results[dataset] = summaries
    print(json.dumps({"dataset": dataset, "rows": N, "eligible": int(eligible.sum()), "no_near_128": int(no_near.sum()), "recomposition_max_abs_um": recomposition_max}), flush=True)
report = {"coarse_checkpoint_sha256": completion["arm_results"]["A"]["checkpoint_sha256"], "coarse_completion_sha256": COMPLETION_SHA256, "coarse_independent_audit_sha256": AUDIT_SHA256, "selections": SELECTIONS, "components": COMPONENTS, "teacher_large_component_box": LARGE.tolist(), "necessary_three_step_bounds": {"oriented_normal_geodesic_rad": NECESSARY_THREE_STEP_LIMITS[0], "signed_support_offset_um": 1800., "absolute_delta_log_basis_u": .36, "absolute_delta_log_basis_v": .36}, "selection_rule": "Saved GPU cell top128 order only, no CPU reranking. Selection0 top cell and its maximum-probability R; selection1 first cell within10deg antipodal plane and500um signed-aligned offset, retaining that cell's predictedR; selection2 minimum observed finite-four-corner RMS over all128x2 components, truth-assisted TRAIN capacity only. No-near selection1 remains unavailable.", "reflection_and_gauge": "Synthetic prepared truth_state is canonical_effective before the separately recorded row reflection. Apply source row reflection once to obtain observed target, then undo candidate R on that target to compute canonical correction. Reflection is x->95-x, O'=O+(95/96)U,U'=-U. No independent antipodal normal fold: it would silently alter correspondence/R. Oriented branch normal error is separate from physical antipodal plane error; raw synthetic reflection index is provenance, not a gauge-independent class label.", "finite_geometry": "Pixel centres0..95 with x/96,y/96. Report finite raster mean centre, four-corner RMS and all96x96-raster RMS separately. Plane signed offset relative to fixed atlas support origin is not finite-centre normal displacement. Basis centre used only for exact existing 9D composition inversion.", "source_convention_evidence": {"prepared_truth": "run_joint_v6_proposal_experiment.py: truth.append(_physical_state_from_quicknii_ouv_v6(row[canonical_effective_quicknii_ouv_float64],...))", "row_reflection": "arbitrary_plane_training_row_v3.py: canonical,observed,affine,representation_index=_reflection_geometry; canonical stored separately from observed", "authenticated_examples": reflection_examples}, "recomposition_max_abs_um": recomposition_max, "limits": "TRAIN residual-distribution analysis, not native model performance, calibrated uncertainty or biological validation. Real upstream affines are weak plane/chart references, not dense/ribbon truth. Synthetic rows reuse one atlas and organizational groups are not biological subjects. Three-step necessary normal/offset/log-diagonal bounds can rule out a fixed-component pose correction but cannot guarantee attainability; no cumulative translation/roll/shear reach claim. These are frame proxies, not observed dense-deformation errors. Exact antipodal transport makes the9D decomposition nonunique: such rows retain physical metrics but have NaN correction/teacher arrays and explicit undefined counts. Teacher-box tests use target->start generation direction, while correction arrays use start->target. Censoring does not remove rows; unavailable near selection is explicit. No DEV images/rows, benchmark, GPU, renderer, model import/inference or training.", "results": results, "input_sha256": dict(hashes)}
report["analysis_source_sha256"] = sha(Path(__file__))
report["output_sha256"] = {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}
(OUT / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
