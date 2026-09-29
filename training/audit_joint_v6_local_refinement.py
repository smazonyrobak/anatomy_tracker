"""CPU audit of frozen local001 endpoints; execute only after confirmed exit."""

import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TEMP"] = r"I:\AnatomyTracker\tmp"
os.environ["TMP"] = r"I:\AnatomyTracker\tmp"

import hashlib
import json

import numpy as np
import torch

torch.set_num_threads(4)
root = Path(r"I:\AnatomyTracker")
run = root / "runs/joint_v6_local_refinement_001"
pack_path = root / "data/joint_v6_local_refinement_frozen_001"
output = root / "runs/joint_v6_local_refinement_audit_001"
output.mkdir(exist_ok=False)
config = json.loads((run / "experiment.json").read_text())
completion = json.loads((run / "completed.json").read_text())
pack_manifest = json.loads((pack_path / "pack.json").read_text())
pack = torch.load(pack_path / "internal_development.pt", map_location="cpu", weights_only=False, mmap=True)
training_pack = torch.load(pack_path / "training.pt", map_location="cpu", weights_only=False, mmap=True)
schedule = torch.load(run / "schedule.pt", map_location="cpu", weights_only=False)
groups = np.array([row["animal_id"] for row in pack["records"]])
modes = np.array([row["selected_mode"] for row in pack["records"]])
truth_reflection = pack["reflection_representation_index"].numpy()
pose_weight = pack["pose_supervision_weight"].numpy()
dense_weight = pose_weight * pack["dense_deformation_supervision_weight"].numpy()
pixel_weight = pack["deformation_weight"].numpy().astype(np.float64)
mass = pixel_weight.sum((1, 2, 3))
pose_eligible = pose_weight > 0
dense_eligible = (dense_weight > 0) & (mass > 0)
truth_map = pack["truth_pullback_map_yx_px"].numpy()
truth_velocity = pack["truth_stationary_velocity_yx_px"].numpy()
truth_ccf = pack["target_ccf_coordinates_ap_dv_ml_um_float64"].numpy()
height, width = pixel_weight.shape[-2:]
identity = np.stack(np.meshgrid(np.arange(height), np.arange(width), indexing="ij"))[None].astype(np.float64)
valid_tissue = (
    pack["source_tissue_ground_truth_mask"].numpy()
    & pack["truth_section_deformation_valid_mask"].numpy()
    & pack["target_valid_correspondence_mask"].numpy()
    & ~pack["target_correspondence_abstention_mask"].numpy()
)
subsets = {
    "all": np.ones(len(groups), dtype=bool), "censored_pose": ~pose_eligible,
    **{mode: modes == mode for mode in np.unique(modes)},
    **{f"truth_reflection_{reflection}": truth_reflection == reflection for reflection in (0, 1)},
    **{f"{mode}/truth_reflection_{reflection}": (modes == mode) & (truth_reflection == reflection) for mode in np.unique(modes) for reflection in (0, 1)},
}
audit = {
    "scope": "Conditional truth-near local refinement with known synthetic PSF, not image-selected global capture or external benchmark",
    "grouping": "organizational single-atlas synthetic groups, not biological subjects",
    "metric_geometry": "Independent NumPy state-to-OUV, five frame corners/centre, antipodal normals; predicted reflection after observed-raster pullback; O+(x/W)U+(y/H)V",
    "baseline": "actual perturbed frame, identity pullback, identity reflection (uniform-prior argmax tie), not true reflection",
    "topology": "Jacobian of saved pullback BEFORE discrete reflection; horizontal representation reversal is not a deformation fold",
    "probabilities_calibrated": False, "endpoints": {},
}
initial_states, perturbations = {}, {}
for step in (0, 4000):
    raw = torch.load(run / f"local_development_raw_step_{step:05d}.pt", map_location="cpu", weights_only=False)
    saved = json.loads((run / f"local_development_metrics_step_{step:05d}.json").read_text())
    initial_states[step] = raw["initial_state"].numpy()
    perturbations[step] = raw["perturbation"].numpy()
    state_arrays = {"truth": pack["truth_state"].numpy(), "initial": initial_states[step], "final": raw["final_state"].numpy()}
    geometry = {}
    for name, state in state_arrays.items():
        state = state.astype(np.float64)
        u = state[:, 3:6] / np.linalg.norm(state[:, 3:6], axis=1, keepdims=True)
        v = state[:, 6:9] - (state[:, 6:9] * u).sum(1, keepdims=True) * u
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        edge_u = u * np.exp(state[:, 9:10])
        edge_v = (state[:, 11:12] * u + v) * np.exp(state[:, 10:11])
        origin = state[:, :3] - 0.5 * (edge_u + edge_v)
        normal = np.cross(edge_u, edge_v)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        landmarks = np.stack((origin, origin + edge_u, origin + edge_v, origin + edge_u + edge_v, origin + 0.5 * (edge_u + edge_v)), axis=1)
        geometry[name] = (origin, edge_u, edge_v, normal, landmarks)
    metrics = {}
    for name in ("initial", "final"):
        normal = geometry[name][3]
        truth_normal = geometry["truth"][3]
        metrics[f"{name}_landmark_error_um"] = np.linalg.norm(geometry[name][4] - geometry["truth"][4], axis=2).mean(1)
        metrics[f"{name}_plane_normal_error_deg"] = np.degrees(np.arctan2(np.linalg.norm(np.cross(normal, truth_normal), axis=1), np.abs((normal * truth_normal).sum(1))))
        for key in (f"{name}_landmark_error_um", f"{name}_plane_normal_error_deg"):
            metrics[key][~pose_eligible] = np.nan
    reflection_log = raw["reflection_log_probability"].numpy().astype(np.float64)
    reflection = reflection_log.argmax(1)
    pullback = raw["pullback_map_yx_px"].numpy().astype(np.float64)
    velocity = raw["stationary_velocity_yx_px"].numpy().astype(np.float64)
    metrics["reflection_correct"] = (reflection == truth_reflection).astype(np.float64)
    metrics["reflection_correct"][~pose_eligible] = np.nan
    metrics["svf_vector_rmse_px"] = np.sqrt((((velocity - truth_velocity) ** 2).sum(1, keepdims=True) * pixel_weight).sum((1, 2, 3)) / np.maximum(mass, 1))
    metrics["initial_pullback_endpoint_error_px"] = (np.linalg.norm(identity - truth_map, axis=1, keepdims=True) * pixel_weight).sum((1, 2, 3)) / np.maximum(mass, 1)
    metrics["pullback_endpoint_error_px"] = (np.linalg.norm(pullback - truth_map, axis=1, keepdims=True) * pixel_weight).sum((1, 2, 3)) / np.maximum(mass, 1)
    for name, coordinates in (("initial", np.broadcast_to(identity, pullback.shape)), ("final", pullback)):
        represented = coordinates.copy()
        if name == "final":
            represented[:, 1] = np.where(reflection[:, None, None] == 1, width - 1 - represented[:, 1], represented[:, 1])
        origin, edge_u, edge_v = geometry[name][:3]
        ccf = origin[:, :, None, None] + edge_u[:, :, None, None] * (represented[:, 1:2] / width) + edge_v[:, :, None, None] * (represented[:, :1] / height)
        error = (np.where(pixel_weight > 0, np.linalg.norm(ccf - truth_ccf, axis=1, keepdims=True), 0) * pixel_weight).sum((1, 2, 3)) / np.maximum(mass, 1)
        metrics["initial_joint_ccf_correspondence_error_um" if name == "initial" else "joint_ccf_correspondence_error_um"] = error
    for key in ("svf_vector_rmse_px", "initial_pullback_endpoint_error_px", "pullback_endpoint_error_px", "initial_joint_ccf_correspondence_error_um", "joint_ccf_correspondence_error_um"):
        metrics[key][~dense_eligible] = np.nan
    derivative_y, derivative_x = np.gradient(pullback, axis=(-2, -1), edge_order=1)
    determinant = (derivative_y[:, 0] * derivative_x[:, 1] - derivative_x[:, 0] * derivative_y[:, 1])[:, None]
    metrics["minimum_jacobian"] = determinant.reshape(len(groups), -1).min(1)
    aggregates = {}
    for subset, selected in subsets.items():
        macro, counts = {}, {}
        for key, values in metrics.items():
            eligible = selected & np.isfinite(values)
            group_values = [float(values[eligible & (groups == group)].mean()) for group in np.unique(groups[eligible])]
            macro[key] = float(np.mean(group_values)) if group_values else None
            counts[key] = int(eligible.sum())
        aggregates[subset] = {"row_count": int(selected.sum()), "group_macro": macro, "eligible_row_count": counts}
    by_group = {group: {key: float(values[(groups == group) & np.isfinite(values)].mean()) if np.any((groups == group) & np.isfinite(values)) else None for key, values in metrics.items()} for group in np.unique(groups)}
    differences = {key: float(np.nanmax(np.abs(values - raw["metrics"][key].numpy()))) for key, values in metrics.items()}
    topology = {}
    for name, mask in (("entire_canvas", np.ones_like(valid_tissue)), ("valid_tissue", valid_tissue), ("dense_supervised_pixels", (pixel_weight > 0) & dense_eligible[:, None, None, None])):
        topology[name] = {"pixel_count": int(mask.sum()), "nonpositive_count": int(((determinant <= 0) & mask).sum()), "minimum_jacobian": float(determinant[mask].min()) if mask.any() else None}
    record = {
        "row_indices_exact_0_to255": np.array_equal(raw["pack_row_index"].numpy(), np.arange(256)),
        "prepared_row_indices_match": torch.equal(raw["prepared_row_index"], pack["prepared_row_index"]),
        "perturbations_match_schedule": torch.equal(raw["perturbation"], schedule["development_perturbation"]),
        "row_eligibility_matches_pack": np.array_equal(raw["pose_supervision_weight"].numpy(), pose_weight) and np.array_equal(raw["dense_supervision_weight"].numpy(), dense_weight),
        "raw_finite": {key: bool(torch.isfinite(raw[key]).all()) for key in ("initial_state", "final_state", "reflection_log_probability", "stationary_velocity_yx_px", "pullback_map_yx_px")},
        "reflection_log_normalization_error": float(np.abs(np.logaddexp.reduce(reflection_log, axis=1)).max()),
        "aggregates": aggregates, "by_group": by_group, "topology": topology,
        "reflection_confusion_pose_eligible": [[int((pose_eligible & (truth_reflection == truth) & (reflection == predicted)).sum()) for predicted in (0, 1)] for truth in (0, 1)],
        "maximum_row_difference_from_saved_metrics": differences,
        "difference_from_saved_macro": {key: value - saved["animal_macro"][key] for key, value in aggregates["all"]["group_macro"].items()},
    }
    audit["endpoints"][str(step)] = record
    np.savez(output / f"recomputed_rows_step_{step:05d}.npz", pack_row_index=raw["pack_row_index"].numpy(), prediction_reflection=reflection, **metrics)
    print(json.dumps({"step": step, "macro": aggregates["all"]["group_macro"], "topology": topology}), flush=True)
    del raw, pullback, velocity, derivative_y, derivative_x, determinant

parent_path = Path(config["parent_checkpoint"])
parent = torch.load(parent_path, map_location="cpu", weights_only=False, mmap=True)
zero = torch.load(run / "joint_model_step_00000.pt", map_location="cpu", weights_only=False, mmap=True)
final = torch.load(run / "joint_model_step_04000.pt", map_location="cpu", weights_only=False, mmap=True)
prefixes = tuple(config["trainable_prefixes"])
audit["parent_and_freezing"] = {
    "zero_parameter_keys_equal_parent": zero["model_state"].keys() == parent["model_state"].keys(),
    "zero_tensor_difference_names": [key for key in parent["model_state"] if not torch.equal(parent["model_state"][key], zero["model_state"][key])],
    "frozen_final_difference_names": [key for key in parent["model_state"] if not key.startswith(prefixes) and not torch.equal(parent["model_state"][key], final["model_state"][key])],
    "trained_changed_tensor_names": [key for key in parent["model_state"] if key.startswith(prefixes) and not torch.equal(parent["model_state"][key], final["model_state"][key])],
    "final_model_finite": all(bool(torch.isfinite(value).all()) for value in final["model_state"].values()),
    "final_optimizer_step_values": sorted({int(value["step"]) for value in final["optimizer_state"]["state"].values()}),
}
trace = [json.loads(line) for line in (run / "training_trace.jsonl").read_text().splitlines()]
audit["training"] = {
    "completion": completion, "trace_steps_exact": [row["step"] for row in trace] == list(range(1, 4001)),
    "trace_row_schedule_matches": np.array_equal(np.array([row["row_index"] for row in trace]), schedule["training_row_index"].numpy()),
    "all_objectives_losses_gradients_finite": all(np.isfinite(row["objective"]) and np.isfinite(row["preclip_gradient_norm"]) and all(np.isfinite(value) for value in row["losses"].values()) for row in trace),
    "applied_steps": trace[-1]["optimizer_steps_applied"],
    "same_initial_states_and_perturbations_at_endpoints": np.array_equal(initial_states[0], initial_states[4000]) and np.array_equal(perturbations[0], perturbations[4000]),
}
audit["identity_checks"] = {
    "development_records_match": json.loads((run / "internal_development_identities.json").read_text()) == pack["records"],
    "training_records_match": json.loads((run / "training_identities.json").read_text()) == training_pack["records"],
    "overlap_counts": {key: len({row[key] for row in pack["records"]} & {row[key] for row in training_pack["records"]}) for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id")},
}
audit["artifact_sha256"] = {}
for path in (parent_path, pack_path / "pack.json", pack_path / "training.pt", pack_path / "internal_development.pt", *[run / name for name in ("experiment.json", "experiment_source.py", "schedule.pt", "training_trace.jsonl", "completed.json", "joint_model_step_00000.pt", "joint_model_step_04000.pt", "local_development_raw_step_00000.pt", "local_development_raw_step_04000.pt")]):
    with path.open("rb") as stream:
        audit["artifact_sha256"][str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
audit["receipts_match"] = {
    "parent": audit["artifact_sha256"][str(parent_path)] == config["parent_sha256"],
    "schedule": audit["artifact_sha256"][str(run / "schedule.pt")] == config["schedule_sha256"],
    "pack_manifest": pack_manifest == config["pack_manifest"],
    **{partition: audit["artifact_sha256"][str(pack_path / f"{partition}.pt")] == record["sha256"] for partition, record in pack_manifest["partitions"].items()},
}
audit["paired_continuation_criteria"] = {}
for subset, record in audit["endpoints"]["4000"]["aggregates"].items():
    values = record["group_macro"]
    if values["final_landmark_error_um"] is None or values["joint_ccf_correspondence_error_um"] is None:
        continue
    reduction = {label: 1 - values[final_key] / values[initial_key] for label, initial_key, final_key in (
        ("landmark", "initial_landmark_error_um", "final_landmark_error_um"),
        ("dense_ccf", "initial_joint_ccf_correspondence_error_um", "joint_ccf_correspondence_error_um"),
        ("pullback", "initial_pullback_endpoint_error_px", "pullback_endpoint_error_px"),
    )}
    audit["paired_continuation_criteria"][subset] = {
        "relative_error_reduction": reduction,
        "landmark_at_least20percent": reduction["landmark"] >= 0.20,
        "dense_ccf_at_least20percent": reduction["dense_ccf"] >= 0.20,
        "pullback_at_least10percent": reduction["pullback"] >= 0.10,
        "normal_not_worse": values["final_plane_normal_error_deg"] <= values["initial_plane_normal_error_deg"],
    }
audit["valid_tissue_no_folds"] = audit["endpoints"]["4000"]["topology"]["valid_tissue"]["nonpositive_count"] == 0
gate = audit["paired_continuation_criteria"]["all"]
audit["overall_numerical_continuation_gate"] = audit["valid_tissue_no_folds"] and all(value for key, value in gate.items() if key != "relative_error_reduction")
audit["scope_warning"] = "A numerical pass is only conditional truth-near local learning; brush-mode regressions still require investigation, not global capture qualification."
(output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps({"gate": gate, "valid_tissue_no_folds": audit["valid_tissue_no_folds"], "overall_numerical_continuation_gate": audit["overall_numerical_continuation_gate"], "parent_and_freezing": audit["parent_and_freezing"]}, indent=2), flush=True)
assert all(audit["receipts_match"].values()) and not any(audit["identity_checks"]["overlap_counts"].values())
assert audit["identity_checks"]["development_records_match"] and audit["identity_checks"]["training_records_match"]
assert not audit["parent_and_freezing"]["zero_tensor_difference_names"] and not audit["parent_and_freezing"]["frozen_final_difference_names"]
assert audit["parent_and_freezing"]["final_model_finite"] and audit["parent_and_freezing"]["zero_parameter_keys_equal_parent"]
assert audit["parent_and_freezing"]["final_optimizer_step_values"] == [audit["training"]["applied_steps"]]
assert audit["training"]["applied_steps"] == completion["optimizer_steps_applied"] and audit["training"]["trace_steps_exact"] and audit["training"]["trace_row_schedule_matches"]
assert audit["training"]["all_objectives_losses_gradients_finite"] and audit["training"]["same_initial_states_and_perturbations_at_endpoints"]
for record in audit["endpoints"].values():
    assert record["row_indices_exact_0_to255"] and record["prepared_row_indices_match"] and record["perturbations_match_schedule"] and record["row_eligibility_matches_pack"]
    assert all(record["raw_finite"].values()) and record["reflection_log_normalization_error"] < 2e-5
    assert max(record["maximum_row_difference_from_saved_metrics"].values()) < 0.05
