"""CPU audit of rehearsal003 versus coordinate002; run only after confirmed exit."""

import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TEMP"] = r"I:\AnatomyTracker\tmp"
os.environ["TMP"] = r"I:\AnatomyTracker\tmp"

import ast
import hashlib
import json

import numpy as np
import torch

torch.set_num_threads(4)
root = Path(r"I:\AnatomyTracker")
run = root / "runs/joint_v6_joint_rehearsal_003"
baseline = root / "runs/joint_v6_local_coordinate_control_002"
baseline_audit_path = root / "runs/joint_v6_local_coordinate_audit_002/audit.json"
pack_path = root / "data/joint_v6_local_refinement_frozen_001"
output = root / "runs/joint_v6_joint_rehearsal_audit_003"
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
    "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
    "zero_parameter_keys_equal_parent_plus_coordinate_input": set(zero["model_state"]) == set(parent["model_state"]) | {"pose_model.coordinate_evidence.weight"},
    "new_coordinate_input_initially_zero": not bool(torch.count_nonzero(zero["model_state"]["pose_model.coordinate_evidence.weight"])),
    "coordinate_input_weight_change_l2": float((final["model_state"]["pose_model.coordinate_evidence.weight"] - zero["model_state"]["pose_model.coordinate_evidence.weight"]).norm()),
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
    "all_replay_losses_finite": all(np.isfinite(row[key]) for row in trace for key in ("local_objective", "proposal_nll", "generated_nll", "frozen_nll")),
    "total_objective_matches_local_plus_replay": all(abs(row["objective"] - row["local_objective"] - config["proposal_rehearsal"]["loss_weight"] * row["proposal_nll"]) < 1e-6 for row in trace),
    "applied_counter_exact": [row["optimizer_steps_applied"] for row in trace] == list(range(1, 4001)),
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
baseline_schedule = torch.load(baseline / "schedule.pt", map_location="cpu", weights_only=False)
baseline_audit = json.loads(baseline_audit_path.read_text())
baseline_config = json.loads((baseline / "experiment.json").read_text())
comparison_keys = ("seed", "steps", "batch_size", "learning_rate", "amp", "evaluation_interval", "refinement_steps", "pose_only_steps", "parent_sha256", "pack_manifest", "catalogue_receipt_sha256", "small_noise_first_500_steps", "large_noise_after_500_steps", "model_kwargs")
local_source = []
for directory in (baseline, run):
    source_text = (directory / "experiment_source.py").read_text()
    local_source.append(ast.get_source_segment(source_text, next(node for node in ast.parse(source_text).body if isinstance(node, ast.FunctionDef) and node.name == "local_pass")))
baseline_zero = torch.load(baseline / "joint_model_step_00000.pt", map_location="cpu", weights_only=False, mmap=True)
audit["matched_coordinate002_comparison"] = {
    "same_numeric_schedule": schedule.keys() == baseline_schedule.keys() and all(torch.equal(value, baseline_schedule[key]) for key, value in schedule.items()),
    "same_experiment_fields": {key: config[key] == baseline_config[key] for key in comparison_keys},
    "same_local_pass_source": local_source[0] == local_source[1],
    "same_initial_model_keys_and_values": zero["model_state"].keys() == baseline_zero["model_state"].keys() and all(torch.equal(value, baseline_zero["model_state"][key]) for key, value in zero["model_state"].items()),
    "same_optimizer_settings_except_parameter_membership": [{key: value for key, value in group.items() if key != "params"} for group in zero["optimizer_state"]["param_groups"]] == [{key: value for key, value in group.items() if key != "params"} for group in baseline_zero["optimizer_state"]["param_groups"]],
    "intentional_unfreezing_exact": set(prefixes) == set(baseline_config["trainable_prefixes"]) | {"pose_model.histology_stem.", "pose_model.shared_encoder.", "pose_model.spatial_residual_blocks.", "pose_model.proposal_head_v6."},
    "final_group_macro_difference_003_minus_002": {
        subset: {key: value - baseline_audit["endpoints"]["4000"]["aggregates"][subset]["group_macro"][key]
                 for key, value in record["group_macro"].items() if value is not None}
        for subset, record in audit["endpoints"]["4000"]["aggregates"].items()
    },
    "baseline_audit_sha256": hashlib.sha256(baseline_audit_path.read_bytes()).hexdigest(),
}
del baseline_zero

# Independent full-catalogue readout; bounded 16-row float64 blocks, no inference.
prepared_path = Path(config["proposal_rehearsal"]["prepared_source_directory"])
proposal_dev = torch.load(prepared_path / "internal_development_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
proposal_train = torch.load(prepared_path / "training_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
catalogue = torch.load(run / "catalogue.pt", map_location="cpu", weights_only=False)
labels = proposal_dev["label"].numpy()
proposal_groups = np.array([row["animal_id"] for row in proposal_dev["records"]])
proposal_modes = np.array([row["selected_mode"] for row in proposal_dev["records"]])
proposal_eligible = proposal_dev["weight"].numpy() > 0
normal_count = catalogue["counts"]["normal_count"]
cells_per_normal = catalogue["counts"]["offset_count_per_normal"] * catalogue["counts"]["roll_count"]
cell_count = normal_count * cells_per_normal
proposal_geometry = {}
for name, states in (("truth", proposal_dev["truth_state"].numpy()), ("catalogue", catalogue["arrays"]["cell_states_float64"])):
    states = np.asarray(states, dtype=np.float64)
    u = states[:, 3:6] / np.linalg.norm(states[:, 3:6], axis=1, keepdims=True)
    v = states[:, 6:9] - (states[:, 6:9] * u).sum(1, keepdims=True) * u
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    normal = np.cross(u, v)
    proposal_geometry[name] = (states[:, :3], np.stack((u, v, normal), axis=-1))
truth_center, truth_frame = proposal_geometry["truth"]
cell_center, cell_frame = proposal_geometry["catalogue"]
normal_grid = cell_frame[::cells_per_normal, :, 2]
normal_angles = np.degrees(np.arccos(np.abs(truth_frame[:, :, 2] @ normal_grid.T).clip(0, 1)))
support_origin = np.array(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"], dtype=np.float64)
proposal_subsets = {
    "all": np.ones(len(labels), dtype=bool), "support_eligible": proposal_eligible,
    "censored_low_support": ~proposal_eligible,
    **{mode: proposal_modes == mode for mode in np.unique(proposal_modes)},
    **{f"{mode}/support_eligible": (proposal_modes == mode) & proposal_eligible for mode in np.unique(proposal_modes)},
    **{f"{mode}/censored_low_support": (proposal_modes == mode) & ~proposal_eligible for mode in np.unique(proposal_modes)},
}
audit["proposal"] = {
    "scope": "Same-run FP32 endpoints; equal organizational-group weighting within each positive-weight intersection; all/censored rows reported separately, not retention gates",
    "catalogue_receipt_matches": catalogue["receipt_sha256"] == config["catalogue_receipt_sha256"],
    "catalogue_normal_geometry_max_difference": float(np.abs(cell_frame[:, :, 2] - np.asarray(catalogue["arrays"]["cell_normal_ap_dv_ml_float64"])).max()),
    "normal_major_order_max_difference": float(np.abs(cell_frame[:, :, 2].reshape(normal_count, cells_per_normal, 3) - normal_grid[:, None]).max()),
    "endpoints": {},
}
assert len(labels) == 640 and cell_count == 98304
for step in (0, 4000):
    raw = np.load(run / f"proposal_development_log_probability_step_{step:05d}.npy", mmap_mode="r")
    saved = json.loads((run / f"proposal_development_metrics_step_{step:05d}.json").read_text())
    assert raw.shape == (len(labels), cell_count)
    prediction, ranks = np.empty(len(labels), dtype=np.int64), np.empty(len(labels), dtype=np.int64)
    normal_log = np.empty((len(labels), normal_count), dtype=np.float64)
    nll = np.empty(len(labels), dtype=np.float64)
    finite, norm_error = True, 0.0
    for start in range(0, len(labels), 16):
        block = np.asarray(raw[start:start + 16], dtype=np.float64)
        target_labels = labels[start:start + len(block)]
        target = block[np.arange(len(block)), target_labels]
        prediction[start:start + len(block)] = block.argmax(1)
        nll[start:start + len(block)] = -target
        finite &= bool(np.isfinite(block).all())
        norm_error = max(norm_error, float(np.abs(np.logaddexp.reduce(block, axis=1)).max()))
        normal_log[start:start + len(block)] = np.logaddexp.reduce(block.reshape(len(block), normal_count, cells_per_normal), axis=2)
        ranks[start:start + len(block)] = (block > target[:, None]).sum(1) + ((block == target[:, None]) & (np.arange(cell_count)[None] < target_labels[:, None])).sum(1) + 1
    frame, center = cell_frame[prediction], cell_center[prediction]
    dot = (frame[:, :, 2] * truth_frame[:, :, 2]).sum(1)
    sign = np.where(dot < 0, -1, 1)
    frame_cosine = ((frame * truth_frame).sum((1, 2)) - 1) / 2
    antipodal_cosine = ((frame * np.array([-1, 1, -1]) * truth_frame).sum((1, 2)) - 1) / 2
    joint_angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(frame[:, :, 2], truth_frame[:, :, 2]), axis=1), np.abs(dot)))
    marginal_angle = normal_angles[np.arange(len(labels)), normal_log.argmax(1)]
    metrics = {
        "nll": nll, "truth_rank": ranks, "plane_angle_deg": joint_angle,
        "antipodal_frame_angle_deg": np.degrees(np.arccos(np.maximum(frame_cosine, antipodal_cosine).clip(-1, 1))),
        "representation_sensitive_frame_angle_deg": np.degrees(np.arccos(frame_cosine.clip(-1, 1))),
        "normal_offset_error_um": np.abs(((center - support_origin) * frame[:, :, 2]).sum(1) - sign * ((truth_center - support_origin) * truth_frame[:, :, 2]).sum(1)),
        "normal_marginal_nll": -normal_log[np.arange(len(labels)), labels // cells_per_normal],
        "normal_marginal_map_angle_deg": marginal_angle,
        "joint_map_within10deg": (joint_angle < 10).astype(float),
        "normal_marginal_map_within10deg": (marginal_angle < 10).astype(float),
        "normal_mass_within10deg": (np.exp(normal_log) * (normal_angles < 10)).sum(1),
        **{f"hit_at_{k}": (ranks <= k).astype(float) for k in (1, 8, 32, 128)},
    }
    aggregates = {subset: {
        "row_count": int(selected.sum()), "group_count": len(np.unique(proposal_groups[selected])),
        "group_macro": {key: float(np.mean([values[selected & (proposal_groups == group)].mean() for group in np.unique(proposal_groups[selected])])) if selected.any() else None for key, values in metrics.items()},
    } for subset, selected in proposal_subsets.items()}
    with np.load(run / f"proposal_development_rows_step_{step:05d}.npz") as saved_rows:
        row_identity_match = np.array_equal(labels, saved_rows["label"]) and np.array_equal(prediction, saved_rows["prediction"]) and np.array_equal(proposal_dev["weight"].numpy(), saved_rows["pose_supervision_weight"])
        differences = {key: float(np.abs(values - saved_rows[key]).max()) for key, values in metrics.items() if key in saved_rows.files}
    audit["proposal"]["endpoints"][str(step)] = {
        "aggregates": aggregates, "raw_all_finite": finite, "raw_log_normalization_error": norm_error,
        "saved_row_identity_match": row_identity_match, "maximum_row_difference_from_saved": differences,
        "saved_macro_difference": {key: aggregates["all"]["group_macro"][key] - value for key, value in saved["animal_macro"].items()},
        "checkpoint_metrics_step_matches": (zero if step == 0 else final)["proposal_development"]["step"] == step,
    }
    np.savez(output / f"proposal_recomputed_rows_step_{step:05d}.npz", label=labels, prediction=prediction, pose_supervision_weight=proposal_dev["weight"].numpy(), **metrics)
    print(json.dumps({"proposal_step": step, "eligible_macro": aggregates["support_eligible"]["group_macro"], "normalization_error": norm_error}), flush=True)
    del raw, block

audit["global_retention_gate"] = {}
for subset in ("support_eligible", *[f"{mode}/support_eligible" for mode in np.unique(proposal_modes)]):
    initial_record = audit["proposal"]["endpoints"]["0"]["aggregates"][subset]
    if not initial_record["row_count"]:
        continue
    initial_values = initial_record["group_macro"]
    final_values = audit["proposal"]["endpoints"]["4000"]["aggregates"][subset]["group_macro"]
    nll_bound, angle_bound, recall_bound, ratio_bound = (.10, 1., .02, 1.05) if subset == "support_eligible" else (.20, 2., .04, 1.10)
    limits = {"nll": initial_values["nll"] + nll_bound, "plane_angle_deg": initial_values["plane_angle_deg"] + angle_bound,
              "hit_at_128": initial_values["hit_at_128"] - recall_bound,
              **{key: initial_values[key] * ratio_bound for key in ("antipodal_frame_angle_deg", "normal_offset_error_um")}}
    checks = {key: final_values[key] >= bound if key == "hit_at_128" else final_values[key] <= bound for key, bound in limits.items()}
    audit["global_retention_gate"][subset] = {
        "row_count": initial_record["row_count"], "initial": {key: initial_values[key] for key in limits},
        "final": {key: final_values[key] for key in limits}, "endpoint_bounds": limits,
        "checks": checks, "passed": all(checks.values()),
    }
audit["global_retention_passed"] = all(record["passed"] for record in audit["global_retention_gate"].values())

replay = config["proposal_rehearsal"]
for path in (baseline_audit_path, baseline / "experiment.json", baseline / "experiment_source.py", baseline / "schedule.pt", run / "catalogue.pt",
             *[prepared_path / name for name in replay["prepared_source_sha256"]],
             *[run / f"replay003_{name}" for name in replay["source_sha256"]],
             *[run / f"proposal_development_{kind}_step_{step:05d}.{suffix}" for step in (0, 4000) for kind, suffix in (("log_probability", "npy"), ("rows", "npz"), ("metrics", "json"))],
             run / "proposal_training_identities.json", run / "proposal_internal_development_identities.json"):
    with path.open("rb") as stream:
        audit["artifact_sha256"][str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
audit["receipts_match"].update({
    **{f"prepared:{name}": audit["artifact_sha256"][str(prepared_path / name)] == digest for name, digest in replay["prepared_source_sha256"].items()},
    **{f"replay:{name}": audit["artifact_sha256"][str(run / f"replay003_{name}")] == digest for name, digest in replay["source_sha256"].items()},
    "copied_catalogue": audit["artifact_sha256"][str(run / "catalogue.pt")] == replay["prepared_source_sha256"]["catalogue.pt"],
    "baseline_audit": audit["artifact_sha256"][str(baseline_audit_path)] == "7f14f9bb46de2cd6ce0bcd3572ea53b4137365d9375acd6bbc7b52ef338dbb01",
})
with np.load(run / "replay003_generated_schedule.npz", allow_pickle=False) as replay_schedule:
    replay_ids = {key: replay_schedule[key][:32000] for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id")}
    replay_cells = replay_schedule["cell_index"][:32000].copy()
frozen_replay = np.load(run / "replay003_frozen_training_row_indices.npy", allow_pickle=False)[:4000]
replay_weight = np.array([row["generated_pose_weight"] for row in trace])
expected_weight = ((np.array([row["finite_support_mass_px"] for row in trace]) >= 64) & (np.array([row["visible_support_mass_px"] for row in trace]) >= 64)).astype(float)
audit["replay_checks"] = {
    "generated_prefix_exact": np.array_equal(np.array([row["replayed_generated_indices"] for row in trace]), np.arange(32000).reshape(4000, 8)),
    "frozen_prefix_exact": np.array_equal(np.array([row["replayed_frozen_row_indices"] for row in trace]), frozen_replay),
    "support_weights_match_recorded_masses": np.array_equal(replay_weight, expected_weight),
    "supervision_counter_exact": np.array_equal(np.array([row["generated_supervised"] for row in trace]), np.cumsum(replay_weight.sum(1))),
    "censor_counter_exact": np.array_equal(np.array([row["generated_censored"] for row in trace]), np.cumsum((1 - replay_weight).sum(1))),
    "completion_counts_match": completion["replayed_generated_observations"] == 32000 and completion["new_observations"] == 0 and completion["unique_replayed_cells"] == len(np.unique(replay_cells)) and completion["unique_replayed_frozen_rows"] == len(np.unique(frozen_replay)),
    "checkpoint_receipts_match": all(checkpoint["schedule_sha256"] == config["schedule_sha256"] and checkpoint["replay_schedule_sha256"] == replay["source_sha256"] for checkpoint in (zero, final)),
    "replay_receipts_match_whole_parent": all(replay["source_sha256"][key] == value for key, value in parent["experiment"]["generated_schedule_sha256"].items()) and replay["prepared_source_sha256"] == parent["experiment"]["prepared_source_sha256"] and replay["source_sha256"]["experiment_source.py"] == parent["experiment"]["source"]["file_sha256"]["training/run_joint_v6_proposal_curriculum.py"],
    "loss_weight_one": replay["loss_weight"] == 1.0,
}
audit["proposal_identity_checks"] = {
    "development_records_match": json.loads((run / "proposal_internal_development_identities.json").read_text()) == proposal_dev["records"],
    "training_records_match": json.loads((run / "proposal_training_identities.json").read_text()) == proposal_train["records"],
    "overlap_counts": {key: len(({row[key] for row in proposal_train["records"] + training_pack["records"]} | set(replay_ids[key])) & {row[key] for row in proposal_dev["records"] + pack["records"]}) for key in replay_ids},
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
audit["combined_local_and_retention_gate"] = audit["overall_numerical_continuation_gate"] and audit["global_retention_passed"]
audit["local_mode_regressions"] = {
    mode: [final_key for initial_key, final_key in (("initial_landmark_error_um", "final_landmark_error_um"), ("initial_joint_ccf_correspondence_error_um", "joint_ccf_correspondence_error_um"), ("initial_pullback_endpoint_error_px", "pullback_endpoint_error_px"), ("initial_plane_normal_error_deg", "final_plane_normal_error_deg")) if audit["endpoints"]["4000"]["aggregates"][mode]["group_macro"][final_key] > audit["endpoints"]["4000"]["aggregates"][mode]["group_macro"][initial_key]]
    for mode in np.unique(modes)
}
audit["scope_warning"] = "Engineering tolerances, not statistical superiority. Both numerical gates passing still does not qualify global capture, calibration or shipping; unresolved local mode regressions preclude expansion."
(output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps({"gate": gate, "valid_tissue_no_folds": audit["valid_tissue_no_folds"], "overall_numerical_continuation_gate": audit["overall_numerical_continuation_gate"], "global_retention_gate": audit["global_retention_gate"], "combined_local_and_retention_gate": audit["combined_local_and_retention_gate"], "local_mode_regressions": audit["local_mode_regressions"], "parent_and_freezing": audit["parent_and_freezing"]}, indent=2), flush=True)
assert all(audit["receipts_match"].values()) and not any(audit["identity_checks"]["overlap_counts"].values())
assert audit["identity_checks"]["development_records_match"] and audit["identity_checks"]["training_records_match"]
assert not audit["parent_and_freezing"]["zero_tensor_difference_names"] and not audit["parent_and_freezing"]["frozen_final_difference_names"]
assert audit["parent_and_freezing"]["final_model_finite"] and audit["parent_and_freezing"]["zero_parameter_keys_equal_parent_plus_coordinate_input"]
assert audit["parent_and_freezing"]["new_coordinate_input_initially_zero"]
comparison = audit["matched_coordinate002_comparison"]
assert comparison["same_numeric_schedule"] and all(comparison["same_experiment_fields"].values())
assert comparison["same_local_pass_source"] and comparison["same_initial_model_keys_and_values"] and comparison["intentional_unfreezing_exact"]
assert comparison["same_optimizer_settings_except_parameter_membership"]
assert audit["parent_and_freezing"]["final_optimizer_step_values"] == [audit["training"]["applied_steps"]]
assert audit["training"]["applied_steps"] == completion["optimizer_steps_applied"] and audit["training"]["trace_steps_exact"] and audit["training"]["trace_row_schedule_matches"]
assert audit["training"]["all_objectives_losses_gradients_finite"] and audit["training"]["same_initial_states_and_perturbations_at_endpoints"]
assert audit["training"]["all_replay_losses_finite"] and audit["training"]["total_objective_matches_local_plus_replay"] and audit["training"]["applied_counter_exact"]
assert all(audit["replay_checks"].values()) and not any(audit["proposal_identity_checks"]["overlap_counts"].values())
assert audit["proposal_identity_checks"]["development_records_match"] and audit["proposal_identity_checks"]["training_records_match"]
assert audit["proposal"]["catalogue_receipt_matches"] and audit["proposal"]["catalogue_normal_geometry_max_difference"] < 1e-10 and audit["proposal"]["normal_major_order_max_difference"] < 1e-10
for record in audit["proposal"]["endpoints"].values():
    assert record["raw_all_finite"] and record["raw_log_normalization_error"] < 2e-5 and record["saved_row_identity_match"] and record["checkpoint_metrics_step_matches"]
    assert max(record["maximum_row_difference_from_saved"].values()) < 0.05 and max(abs(value) for value in record["saved_macro_difference"].values()) < 0.05
for record in audit["endpoints"].values():
    assert record["row_indices_exact_0_to255"] and record["prepared_row_indices_match"] and record["perturbations_match_schedule"] and record["row_eligibility_matches_pack"]
    assert all(record["raw_finite"].values()) and record["reflection_log_normalization_error"] < 2e-5
    assert max(record["maximum_row_difference_from_saved_metrics"].values()) < 0.05
