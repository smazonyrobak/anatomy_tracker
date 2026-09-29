"""Frozen005/007 CPU comparison at4000; execute only after confirmed007 exit."""

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
runs = {"base_005": root / "joint_v6_proposal_normal_objective_005", "depth_007": root / "joint_v6_spatial_depth_control_007"}
output = root / "joint_v6_spatial_depth_audit_005_007"
output.mkdir(exist_ok=True)
prepared = root / "joint_v6_proposal_substantive_001"
dev = torch.load(prepared / "internal_development_prepared.pt", map_location="cpu", weights_only=False)
catalogue = torch.load(prepared / "catalogue.pt", map_location="cpu", weights_only=False)
labels = dev["label"].numpy()
groups = np.array([row["animal_id"] for row in dev["records"]])
modes = np.array([row["selected_mode"] for row in dev["records"]])
components = np.array([group.rsplit("-animal-", 1)[0].rsplit("-", 1)[-1] for group in groups])
assert set(components) == {"pose", "joint"}
normal_count = catalogue["counts"]["normal_count"]
cells_per_normal = catalogue["counts"]["offset_count_per_normal"] * catalogue["counts"]["roll_count"]
normal_grid = catalogue["arrays"]["cell_normal_ap_dv_ml_float64"][::cells_per_normal]
_, truth_frame, _ = full_frame_state_to_components(dev["truth_state"].double())
angles = np.rad2deg(np.arccos(np.abs(truth_frame[:, :, 2].numpy() @ normal_grid.T).clip(0, 1)))
subsets = {"all": np.ones(len(labels), dtype=bool), "support_eligible": dev["weight"].numpy() > 0, "censored_low_support": dev["weight"].numpy() == 0, **{mode: modes == mode for mode in np.unique(modes)}}
subsets.update({component: components == component for component in np.unique(components)})
subsets.update({component + "/" + mode: (components == component) & (modes == mode) for component in np.unique(components) for mode in np.unique(modes)})
subsets.update({component + "/support_eligible": (components == component) & (dev["weight"].numpy() > 0) for component in np.unique(components)})
result = {
    "scope": "Frozen4000-step raw predictions only; CPU, no model inference; organizational synthetic groups, not biological animals; probabilities uncalibrated",
    "source_component": "pose/joint parsed from the authenticated fixed animal_id namespace suffix before '-animal-'",
    "occupancy_stratification": "Unavailable in prepared development payload: only channels, weight, truth_state, label, records are stored; no support/visible masses. No occupancy inferred from image thresholds or brush boundaries.",
    "support_scope": "positive point-pose weight is support eligibility, not demonstrated image identifiability or full-section coverage",
    "runs": {},
}
initial, configs = {}, {}
for name, run in runs.items():
    configs[name] = json.loads((run / "experiment.json").read_text())
    initial[name] = torch.load(run / "joint_model_step_00000.pt", map_location="cpu", weights_only=False)
    final = torch.load(run / "joint_model_step_04000.pt", map_location="cpu", weights_only=False)
    trace = [json.loads(line) for line in (run / "training_trace.jsonl").read_text().splitlines()]
    trace = [row for row in trace if row["step"] <= 4000]
    saved = json.loads((run / "development_metrics_step_04000.json").read_text())
    raw = np.load(run / "development_log_probability_step_04000.npy", mmap_mode="r")
    prediction = raw.argmax(1)
    normal_log = np.empty((len(labels), normal_count), dtype=np.float64)
    ranks = np.empty(len(labels), dtype=np.int64)
    finite, norm_error = True, 0.0
    for start in range(0, len(labels), 32):
        block = np.asarray(raw[start:start + 32], dtype=np.float64)
        target_labels = labels[start:start + len(block)]
        target = block[np.arange(len(block)), target_labels]
        finite &= bool(np.isfinite(block).all())
        maximum = block.max(1)
        norm_error = max(norm_error, float(np.abs(maximum + np.log(np.exp(block - maximum[:, None]).sum(1))).max()))
        normal_log[start:start + len(block)] = np.logaddexp.reduce(block.reshape(len(block), normal_count, cells_per_normal), axis=2)
        ranks[start:start + len(block)] = (block > target[:, None]).sum(1) + ((block == target[:, None]) & (np.arange(raw.shape[1])[None] < target_labels[:, None])).sum(1) + 1
    marginal_prediction = normal_log.argmax(1)
    joint_angle = angles[np.arange(len(labels)), prediction // cells_per_normal]
    marginal_angle = angles[np.arange(len(labels)), marginal_prediction]
    metrics = {
        "nll": -raw[np.arange(len(labels)), labels].astype(np.float64),
        "normal_marginal_nll": -normal_log[np.arange(len(labels)), labels // cells_per_normal],
        "plane_angle_deg": joint_angle, "normal_marginal_map_angle_deg": marginal_angle,
        "joint_map_within10deg": (joint_angle < 10).astype(float),
        "normal_marginal_map_within10deg": (marginal_angle < 10).astype(float),
        "normal_mass_within10deg": (np.exp(normal_log) * (angles < 10)).sum(1),
        **{f"hit_at_{k}": (ranks <= k).astype(float) for k in (32, 128)},
    }
    aggregates = {subset: {"row_count": int(selected.sum()), "group_count": len(np.unique(groups[selected])), "synthetic_group_macro": {key: float(np.mean([values[selected & (groups == group)].mean() for group in np.unique(groups[selected])])) for key, values in metrics.items()}} for subset, selected in subsets.items()}
    with np.load(run / "development_rows_step_04000.npz") as saved_rows:
        row_identity_match = np.array_equal(labels, saved_rows["label"]) and np.array_equal(prediction, saved_rows["prediction"]) and np.array_equal(dev["weight"].numpy(), saved_rows["pose_supervision_weight"])
    hashes = {}
    for filename in ("experiment.json", "experiment_source.py", "joint_model_step_00000.pt", "joint_model_step_04000.pt", "development_log_probability_step_04000.npy", "generated_schedule.npz", "frozen_training_row_indices.npy", "training_trace.jsonl", "internal_development_identities.json"):
        with (run / filename).open("rb") as stream:
            hashes[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {
        "aggregates": aggregates, "raw_all_finite": finite, "raw_log_normalization_error": norm_error,
        "saved_row_identity_match": bool(row_identity_match),
        "saved_macro_difference": {key: value - saved["animal_macro"][key] for key, value in aggregates["all"]["synthetic_group_macro"].items() if key in saved["animal_macro"]},
        "development_records_match": json.loads((run / "internal_development_identities.json").read_text()) == dev["records"],
        "trace_steps_exact": [row["step"] for row in trace] == list(range(1, 4001)),
        "applied_steps": sum(row["optimizer_step_applied"] for row in trace),
        "skipped_step_ids": [row["step"] for row in trace if not row["optimizer_step_applied"]],
        "all_trace_losses_finite": all(np.isfinite(row[key]) for row in trace for key in ("weighted_nll", "weighted_normal_marginal_nll", "weighted_objective")),
        "last100_joint_nll": float(np.mean([row["weighted_nll"] for row in trace[-100:]])),
        "optimizer_step_values": sorted({int(value["step"]) for value in final["optimizer_state"]["state"].values()}),
        "final_model_all_finite": all(bool(torch.isfinite(value).all()) for value in final["model_state"].values()),
        "changed_parameter_names": [key for key, value in final["model_state"].items() if not torch.equal(value, initial[name]["model_state"][key])],
        "artifact_sha256": hashes,
    }
    result["runs"][name] = record
    np.savez(output / f"{name}_rows.npz", label=labels, prediction=prediction, normal_prediction=marginal_prediction, **metrics)
    print(json.dumps({"run": name, "macro": aggregates["all"]["synthetic_group_macro"], "applied_steps": record["applied_steps"]}), flush=True)
    del raw, final

base, depth = initial["base_005"], initial["depth_007"]
base_parameters, depth_parameters = base["model_state"], depth["model_state"]
added = sorted(depth_parameters.keys() - base_parameters.keys())
result["initialization"] = {
    "base_parameter_difference_names": [key for key in base_parameters if key not in depth_parameters or not torch.equal(base_parameters[key], depth_parameters[key])],
    "added_parameter_names": added,
    "only_spatial_residual_parameters_added": bool(added) and all(key.startswith("pose_model.spatial_residual_blocks.") for key in added),
    "added_final_convolutions_zero": all(torch.count_nonzero(depth_parameters[key]).item() == 0 for key in added if key.split(".")[-2] == "5"),
    "cpu_rng_equal": torch.equal(base["torch_rng"], depth["torch_rng"]),
    "cuda_rng_equal": len(base["cuda_rng"]) == len(depth["cuda_rng"]) and all(torch.equal(left, right) for left, right in zip(base["cuda_rng"], depth["cuda_rng"])),
    "numpy_rng_equal": base["numpy_rng"][0] == depth["numpy_rng"][0] and np.array_equal(base["numpy_rng"][1], depth["numpy_rng"][1]) and base["numpy_rng"][2:] == depth["numpy_rng"][2:],
    "python_rng_equal": base["python_rng"] == depth["python_rng"], "scaler_equal": base["scaler_state"] == depth["scaler_state"],
    "both_optimizer_states_empty": not base["optimizer_state"]["state"] and not depth["optimizer_state"]["state"],
}
id_fields = ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id")
with np.load(runs["base_005"] / "generated_schedule.npz") as left, np.load(runs["depth_007"] / "generated_schedule.npz") as right:
    result["schedule_fields_equal"] = {key: bool(np.array_equal(left[key], right[key])) for key in left.files if key not in id_fields}
    result["generated_ids_differ_only_run_namespace"] = {key: bool(np.array_equal(np.char.replace(left[key], runs["base_005"].name, "RUN"), np.char.replace(right[key], runs["depth_007"].name, "RUN"))) for key in id_fields}
result["frozen_schedule_equal"] = bool(np.array_equal(*[np.load(run / "frozen_training_row_indices.npy") for run in runs.values()]))
result["common_configuration_equal"] = {key: configs["base_005"][key] == configs["depth_007"][key] for key in ("seed", "batch_size", "generated_per_batch", "learning_rate", "optimizer", "weight_decay", "retrieval_shape_h_w", "prepared_source_sha256", "catalogue_receipt_sha256", "normal_marginal_nll_weight")}
result["generator_acquisition_fields_equal"] = {key: value == configs["depth_007"]["generator"][key] for key, value in configs["base_005"]["generator"].items() if key not in ("new_rendered_observations_planned", "unique_cells_planned")}
result["planned_count_metadata"] = {name: {key: config["generator"][key] for key in ("new_rendered_observations_planned", "unique_cells_planned")} for name, config in configs.items()}
result["audit_implementation_correction"] = "Initial audit selectors incorrectly included all block5 tensors in the final-convolution zero check, and compared intended4k versus20k planned-render totals. Corrected to exact final module index and acquisition fields; no raw metrics or frozen artifacts changed."
result["base_model_configuration_equal"] = all(value == configs["depth_007"]["model_kwargs"].get(key) for key, value in configs["base_005"]["model_kwargs"].items())
result["depth_model_kwargs"] = configs["depth_007"]["model_kwargs"]
result["completion_007"] = json.loads((runs["depth_007"] / "completed.json").read_text())
(output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2), flush=True)
assert not result["initialization"]["base_parameter_difference_names"]
assert all(value for key, value in result["initialization"].items() if key not in ("base_parameter_difference_names", "added_parameter_names"))
assert all(result["schedule_fields_equal"].values()) and all(result["generated_ids_differ_only_run_namespace"].values()) and result["frozen_schedule_equal"]
assert all(result["common_configuration_equal"].values()) and all(result["generator_acquisition_fields_equal"].values()) and result["base_model_configuration_equal"]
for record in result["runs"].values():
    assert record["raw_all_finite"] and record["raw_log_normalization_error"] < 2e-5 and record["final_model_all_finite"]
    assert record["saved_row_identity_match"] and record["development_records_match"] and record["trace_steps_exact"] and record["all_trace_losses_finite"]
    assert max(abs(value) for value in record["saved_macro_difference"].values()) < 2e-5
    assert record["optimizer_step_values"] == [record["applied_steps"]]
