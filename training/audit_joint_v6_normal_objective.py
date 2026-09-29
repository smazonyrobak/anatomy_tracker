"""Run on CPU only AFTER root confirms 005 exited; compare frozen 20k endpoints."""

import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TEMP"] = r"I:\AnatomyTracker\tmp"
os.environ["TMP"] = r"I:\AnatomyTracker\tmp"

import hashlib
import json

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components

torch.set_num_threads(4)
root = Path(r"I:\AnatomyTracker\runs")
runs = {
    "curriculum_003": root / "joint_v6_proposal_curriculum_003",
    "normal_objective_005": root / "joint_v6_proposal_normal_objective_005",
}
output = root / "joint_v6_proposal_normal_objective_audit_003_005"
output.mkdir(exist_ok=False)
prepared = root / "joint_v6_proposal_substantive_001"
dev = torch.load(prepared / "internal_development_prepared.pt", map_location="cpu", weights_only=False)
catalogue = torch.load(prepared / "catalogue.pt", map_location="cpu", weights_only=False)
labels = dev["label"].numpy()
groups = np.array([row["animal_id"] for row in dev["records"]])
modes = np.array([row["selected_mode"] for row in dev["records"]])
weights = dev["weight"].numpy()
normal_count = catalogue["counts"]["normal_count"]
cells_per_normal = catalogue["counts"]["offset_count_per_normal"] * catalogue["counts"]["roll_count"]
normal_grid = catalogue["arrays"]["cell_normal_ap_dv_ml_float64"][::cells_per_normal]
truth_center, truth_frame, _ = full_frame_state_to_components(dev["truth_state"].double())
truth_normal = truth_frame[:, :, 2].numpy()
normal_angles = np.rad2deg(np.arccos(np.abs(truth_normal @ normal_grid.T).clip(0, 1)))
origin = np.asarray(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"])
truth_offset = ((truth_center.numpy() - origin) * truth_normal).sum(1)
cell_normals = catalogue["arrays"]["cell_normal_ap_dv_ml_float64"]
cell_offsets = catalogue["arrays"]["cell_signed_offset_um_float64"]
subsets = {
    "all": np.ones(len(labels), dtype=bool),
    "identifiable": weights > 0,
    "censored_low_support": weights == 0,
    **{mode: modes == mode for mode in np.unique(modes)},
    **{mode + "_identifiable": (modes == mode) & (weights > 0) for mode in np.unique(modes)},
}
id_fields = ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id")
configs, schedules, initial_states, identities, audited, hashes = {}, {}, {}, {}, {}, {}

for name, run in runs.items():
    config = configs[name] = json.loads((run / "experiment.json").read_text())
    completion = json.loads((run / "completed.json").read_text())
    trace = [json.loads(line) for line in (run / "training_trace.jsonl").read_text().splitlines()]
    with np.load(run / "generated_schedule.npz") as saved_schedule:
        schedules[name] = {key: saved_schedule[key] for key in saved_schedule.files}
    schedule = schedules[name]
    identities[name] = {partition: json.loads((run / f"{partition}_identities.json").read_text()) for partition in ("training", "internal_development")}
    initial_states[name] = torch.load(run / "joint_model_step_00000.pt", map_location="cpu", weights_only=False)
    final = torch.load(run / "joint_model_step_20000.pt", map_location="cpu", weights_only=False)
    saved = json.loads((run / "development_metrics_step_20000.json").read_text())
    raw = np.load(run / "development_log_probability_step_20000.npy", mmap_mode="r")
    prediction = np.empty(len(labels), dtype=np.int64)
    ranks = np.empty(len(labels), dtype=np.int64)
    target_log = np.empty(len(labels), dtype=np.float64)
    normal_log_probability = np.empty((len(labels), normal_count), dtype=np.float64)
    all_finite, normalization_error = True, 0.0
    for start in range(0, len(labels), 32):
        stop = min(start + 32, len(labels))
        block = np.asarray(raw[start:stop], dtype=np.float64)
        all_finite &= bool(np.isfinite(block).all())
        maximum = block.max(1)
        normalization_error = max(normalization_error, float(np.abs(maximum + np.log(np.exp(block - maximum[:, None]).sum(1))).max()))
        target = block[np.arange(len(block)), labels[start:stop]]
        target_log[start:stop] = target
        ranks[start:stop] = (block > target[:, None]).sum(1) + ((block == target[:, None]) & (np.arange(raw.shape[1])[None] < labels[start:stop, None])).sum(1) + 1
        prediction[start:stop] = block.argmax(1)
        normal_log_probability[start:stop] = np.logaddexp.reduce(block.reshape(len(block), normal_count, cells_per_normal), axis=2)
    normal_prediction = normal_log_probability.argmax(1)
    joint_angle = normal_angles[np.arange(len(labels)), prediction // cells_per_normal]
    marginal_angle = normal_angles[np.arange(len(labels)), normal_prediction]
    sign = np.where((cell_normals[prediction] * truth_normal).sum(1) < 0, -1, 1)
    metrics = {
        "nll": -target_log,
        "normal_marginal_nll": -normal_log_probability[np.arange(len(labels)), labels // cells_per_normal],
        "normal_marginal_map_angle_deg": marginal_angle,
        "plane_angle_deg": joint_angle,
        "normal_offset_error_um": np.abs(cell_offsets[prediction] - sign * truth_offset),
        "truth_rank": ranks,
        "joint_map_within_10deg": (joint_angle < 10).astype(float),
        "normal_marginal_map_within_10deg": (marginal_angle < 10).astype(float),
        "normal_probability_mass_within_10deg": (np.exp(normal_log_probability) * (normal_angles < 10)).sum(1),
        **{f"hit_at_{k}": (ranks <= k).astype(float) for k in (1, 8, 32, 128)},
    }
    aggregates = {subset: {
        "row_count": int(selected.sum()), "organizational_group_count": len(np.unique(groups[selected])),
        "synthetic_group_macro": {key: float(np.mean([values[selected & (groups == group)].mean() for group in np.unique(groups[selected])])) for key, values in metrics.items()} if selected.any() else None,
    } for subset, selected in subsets.items()}
    with np.load(run / "development_rows_step_20000.npz") as saved_rows:
        row_errors = {key: float(np.max(np.abs(saved_rows[key] - values))) for key, values in metrics.items() if key in saved_rows.files}
        identity_match = np.array_equal(saved_rows["label"], labels) and np.array_equal(saved_rows["prediction"], prediction) and np.array_equal(saved_rows["pose_supervision_weight"], weights)
    applied = np.array([row["optimizer_step_applied"] for row in trace])
    support = np.array([row["finite_slab_support_mass_px"] for row in trace]).ravel()
    visible = np.array([row["visible_support_mass_px"] for row in trace]).ravel()
    generated_weight = np.array([row["generated_point_pose_weight"] for row in trace]).ravel()
    changed = [key for key, value in final["model_state"].items() if not torch.equal(value, initial_states[name]["model_state"][key])]
    audited[name] = {
        "completion": completion, "step": 20000, "aggregates": aggregates,
        "raw_shape": list(raw.shape), "all_raw_log_probabilities_finite": all_finite,
        "maximum_absolute_log_normalization_error": normalization_error,
        "saved_labels_predictions_weights_match": bool(identity_match),
        "saved_per_row_maximum_absolute_difference": row_errors,
        "saved_macro_difference": {key: value - saved["animal_macro"][key] for key, value in aggregates["all"]["synthetic_group_macro"].items() if key in saved["animal_macro"]},
        "development_identities_unchanged": identities[name]["internal_development"] == dev["records"],
        "id_overlap_counts": {field: {
            "frozen_train_development": len({row[field] for row in identities[name]["training"]} & {row[field] for row in identities[name]["internal_development"]}),
            "generated_development": len(set(schedule[field]) & {row[field] for row in identities[name]["internal_development"]}),
            "generated_unique": len(np.unique(schedule[field])),
        } for field in id_fields},
        "trace_consecutive_1_to_20000": [row["step"] for row in trace] == list(range(1, 20001)),
        "attempted_steps": len(trace), "applied_steps": int(applied.sum()),
        "skipped_step_ids": [row["step"] for row in trace if not row["optimizer_step_applied"]],
        "all_training_losses_finite": all(np.isfinite(row[key]) for row in trace for key in ("weighted_nll", "generated_weighted_nll", "frozen_weighted_nll")),
        "last100_weighted_joint_nll": float(np.mean([row["weighted_nll"] for row in trace[-100:]])),
        "generated_trace_indices_exact": np.array_equal(np.array([row["generated_sample_indices"] for row in trace]).ravel(), np.arange(len(schedule["cell_index"]))),
        "generated_schedule_count": len(schedule["cell_index"]),
        "generated_distinct_cells": len(np.unique(schedule["cell_index"])),
        "first_cycle_complete": np.array_equal(np.sort(schedule["cell_index"][:98304]), np.arange(98304)),
        "support_censoring_matches": bool(np.array_equal(generated_weight, ((support >= 64) & (visible >= 64)).astype(float))),
        "generated_supervised": int(generated_weight.sum()), "generated_censored": int((generated_weight == 0).sum()),
        "all_final_model_tensors_finite": all(bool(torch.isfinite(value).all()) for value in final["model_state"].values()),
        "final_optimizer_step_values": sorted({int(value["step"]) for value in final["optimizer_state"]["state"].values()}),
        "changed_model_tensor_names": changed,
        "only_proposal_path_changed": all(key.startswith(("pose_model.histology_stem.", "pose_model.shared_encoder.", "pose_model.proposal_head_v6.")) for key in changed),
    }
    hashes[name] = {}
    for filename in ("experiment.json", "experiment_source.py", "source_diff.patch", "completed.json", "generated_schedule.npz", "frozen_training_row_indices.npy", "training_trace.jsonl", "training_identities.json", "internal_development_identities.json", "catalogue.pt", "joint_model_step_00000.pt", "joint_model_step_20000.pt", "development_log_probability_step_20000.npy", "development_rows_step_20000.npz", "development_metrics_step_20000.json"):
        with (run / filename).open("rb") as stream:
            hashes[name][filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    audited[name]["schedule_receipts_match"] = all(hashes[name][filename] == expected for filename, expected in config["generated_schedule_sha256"].items())
    audited[name]["source_archive_receipt_matches"] = hashes[name]["experiment_source.py"] == config["source"]["file_sha256"]["training/run_joint_v6_proposal_curriculum.py"]
    np.savez(output / f"{name}_rows.npz", label=labels, prediction=prediction, normal_prediction=normal_prediction, pose_supervision_weight=weights, **metrics)
    print(json.dumps({"run": name, "macro": aggregates["all"]["synthetic_group_macro"], "applied_steps": int(applied.sum())}), flush=True)
    del raw, final

before, after = (initial_states[key] for key in runs)
differences = []
stack = [(key, before[key], after[key]) for key in ("model_state", "optimizer_state", "scaler_state", "torch_rng", "cuda_rng", "numpy_rng", "python_rng")]
while stack:
    key, left, right = stack.pop()
    if isinstance(left, torch.Tensor):
        equal = isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        equal = isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    elif isinstance(left, dict):
        equal = isinstance(right, dict) and left.keys() == right.keys()
        if equal:
            stack.extend((f"{key}.{field}", value, right[field]) for field, value in left.items())
    elif isinstance(left, (list, tuple)):
        equal = isinstance(right, type(left)) and len(left) == len(right)
        if equal:
            stack.extend((f"{key}.{index}", value, right[index]) for index, value in enumerate(left))
    else:
        equal = left == right
    if not equal:
        differences.append(key)
left_schedule, right_schedule = schedules.values()
common_config = ("seed", "steps", "batch_size", "generated_per_batch", "learning_rate", "optimizer", "weight_decay", "evaluation_interval", "model_kwargs", "retrieval_shape_h_w", "prepared_source_directory", "prepared_source_sha256", "cache_manifests", "atlas_binding", "catalogue_receipt_sha256", "generator")
source_keys = configs["curriculum_003"]["source"]["file_sha256"].keys() & configs["normal_objective_005"]["source"]["file_sha256"].keys()
result = {
    "scope": "CPU audit of confirmed-exited frozen003/005 at20000 attempted steps; organizational single-atlas groups, not biological animals; no new inference or public benchmark",
    "metrics_scope": "absolute physical normal endpoints, not ML-reflection quotient; raw probabilities uncalibrated; exact-cell recall is not landmark accuracy",
    "results": audited,
    "initial_checkpoint_difference_paths": differences,
    "common_configuration_fields_equal": {key: configs["curriculum_003"][key] == configs["normal_objective_005"][key] for key in common_config},
    "numeric_schedule_fields_equal": {key: np.array_equal(left_schedule[key], right_schedule[key]) for key in left_schedule if key not in id_fields},
    "organizational_id_values_differ_only_run_namespace": {key: np.array_equal(np.char.replace(left_schedule[key], runs["curriculum_003"].name, "RUN"), np.char.replace(right_schedule[key], runs["normal_objective_005"].name, "RUN")) for key in id_fields},
    "frozen_training_schedule_equal": np.array_equal(*[np.load(run / "frozen_training_row_indices.npy") for run in runs.values()]),
    "training_development_identities_equal": identities["curriculum_003"] == identities["normal_objective_005"],
    "changed_common_source_receipts": {key: [configs[name]["source"]["file_sha256"][key] for name in runs] for key in sorted(source_keys) if configs["curriculum_003"]["source"]["file_sha256"][key] != configs["normal_objective_005"]["source"]["file_sha256"][key]},
    "objective_005": configs["normal_objective_005"]["objective"],
    "normal_objective_weight_005": configs["normal_objective_005"]["normal_marginal_nll_weight"],
    "artifact_sha256": hashes,
    "probabilities_calibrated": False,
}
(output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key not in ("results", "artifact_sha256")}, indent=2), flush=True)
assert not differences and all(result["common_configuration_fields_equal"].values())
assert all(result["numeric_schedule_fields_equal"].values()) and all(result["organizational_id_values_differ_only_run_namespace"].values())
assert result["frozen_training_schedule_equal"] and result["training_development_identities_equal"]
for record in audited.values():
    assert record["all_raw_log_probabilities_finite"] and record["maximum_absolute_log_normalization_error"] < 2e-5
    assert record["saved_labels_predictions_weights_match"] and record["development_identities_unchanged"]
    assert max(record["saved_per_row_maximum_absolute_difference"].values()) < 2e-5
    assert max(abs(value) for value in record["saved_macro_difference"].values()) < 2e-5
    assert record["trace_consecutive_1_to_20000"] and record["all_training_losses_finite"]
    assert record["applied_steps"] == record["completion"]["optimizer_steps_applied"] and record["final_optimizer_step_values"] == [record["applied_steps"]]
    assert record["generated_trace_indices_exact"] and record["first_cycle_complete"] and record["support_censoring_matches"]
    assert record["generated_supervised"] == record["completion"]["generated_supervised"] and record["generated_censored"] == record["completion"]["generated_censored"]
    assert record["all_final_model_tensors_finite"] and record["only_proposal_path_changed"]
    assert record["schedule_receipts_match"] and record["source_archive_receipt_matches"]
    assert all(value["frozen_train_development"] == value["generated_development"] == 0 for value in record["id_overlap_counts"].values())
