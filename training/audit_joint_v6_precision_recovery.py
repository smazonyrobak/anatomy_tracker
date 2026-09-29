"""CPU-only frozen 002/004 comparison; synthetic groups are not real animals."""

import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TEMP"] = r"I:\AnatomyTracker\tmp"
os.environ["TMP"] = r"I:\AnatomyTracker\tmp"
os.environ["MPLCONFIGDIR"] = r"I:\AnatomyTracker\cache\matplotlib"

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components

torch.set_num_threads(4)
root = Path(r"I:\AnatomyTracker\runs")
runs = {
    "capacity_002": root / "joint_v6_proposal_capacity_002",
    "recovery_004": root / "joint_v6_proposal_precision_recovery_004",
}
output = root / "joint_v6_proposal_precision_audit_002_004"
output.mkdir(exist_ok=False)
repository = Path(__file__).resolve().parents[1]
configs = {key: json.loads((run / "experiment.json").read_text()) for key, run in runs.items()}
traces = {key: [json.loads(line) for line in (run / "training_trace.jsonl").read_text().splitlines()] for key, run in runs.items()}
identities = {key: {part: json.loads((run / f"{part}_identities.json").read_text()) for part in ("training", "internal_development")} for key, run in runs.items()}
id_keys = ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id")
overlap = {key: {field: len({row[field] for row in ids["training"]} & {row[field] for row in ids["internal_development"]}) for field in id_keys} for key, ids in identities.items()}
cache = root / "arbitrary_plane_finite_v6_substantive_data_001"
cache_match = {}
for partition, ids in identities["recovery_004"].items():
    manifest = json.loads((cache / f"{partition}_cache" / "manifest.json").read_text())
    cache_match[partition] = (
        manifest["receipt_sha256"] == configs["recovery_004"]["cache_manifests"][partition]
        and len(manifest["rows"]) == len(ids)
        and all(all(row[field] == record["lineage"][field] for field in id_keys)
                and row["training_row_id"] == record["training_row_id"]
                and row["row_receipt_sha256"] == record["training_row_receipt_sha256"]
                for row, record in zip(ids, manifest["rows"]))
    )
schedule_equal = np.array_equal(*[np.load(run / "training_row_indices.npy") for run in runs.values()])
dev = torch.load(runs["recovery_004"] / "internal_development_prepared.pt", map_location="cpu", weights_only=False)
catalogue = torch.load(runs["recovery_004"] / "catalogue.pt", map_location="cpu", weights_only=False)
labels = dev["label"].numpy()
group = np.array([row["animal_id"] for row in dev["records"]])
modes = np.array([row["selected_mode"] for row in dev["records"]])
groups = np.unique(group)
truth_center, truth_frame, _ = full_frame_state_to_components(dev["truth_state"].double())
truth_normal = truth_frame[:, :, 2]
states = torch.as_tensor(catalogue["arrays"]["cell_states_float64"])
origin = torch.as_tensor(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"], dtype=torch.float64)
audited = {}
hashes = {}
for name, run in runs.items():
    audited[name] = {}
    artifact_names = ["experiment.json", "source_diff.patch", "catalogue.pt", "training_trace.jsonl", "training_row_indices.npy", "training_identities.json", "internal_development_identities.json", "training_prepared.pt", "internal_development_prepared.pt", "joint_model_step_03000.pt", "joint_model_step_10000.pt"]
    for step in range(3000, 10001, 1000):
        raw_name = f"development_log_probability_step_{step:05d}.npy"
        metric_name = f"development_metrics_step_{step:05d}.json"
        row_name = f"development_rows_step_{step:05d}.npz"
        artifact_names.extend((raw_name, metric_name, row_name))
        raw = np.load(run / raw_name, mmap_mode="r")
        saved = json.loads((run / metric_name).read_text())
        truth_log = raw[np.arange(len(labels)), labels].astype(np.float64)
        ranks = (raw > truth_log[:, None]).sum(1) + ((raw == truth_log[:, None]) & (np.arange(raw.shape[1])[None] < labels[:, None])).sum(1) + 1
        predicted = raw.argmax(1)
        center, frame, _ = full_frame_state_to_components(states[predicted])
        normal = frame[:, :, 2]
        dot = (normal * truth_normal).sum(-1)
        sign = torch.where(dot < 0, -1.0, 1.0)
        offset = (((center - origin) * normal).sum(-1) - sign * ((truth_center - origin) * truth_normal).sum(-1)).abs()
        cosine = ((frame * truth_frame).sum((-2, -1)) - 1.0) / 2.0
        antipodal_cosine = ((frame * torch.tensor([-1.0, 1.0, -1.0]) * truth_frame).sum((-2, -1)) - 1.0) / 2.0
        metrics = {
            "nll": -truth_log, "truth_rank": ranks,
            "plane_angle_deg": torch.rad2deg(torch.acos(dot.abs().clamp(0, 1))).numpy(),
            "normal_offset_error_um": offset.numpy(),
            "antipodal_frame_angle_deg": torch.rad2deg(torch.acos(torch.maximum(cosine, antipodal_cosine).clamp(-1, 1))).numpy(),
            "representation_sensitive_frame_angle_deg": torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).numpy(),
            **{f"hit_at_{k}": (ranks <= k).astype(float) for k in (1, 8, 32, 128)},
        }
        macro = {metric: float(np.mean([values[group == key].mean() for key in groups])) for metric, values in metrics.items()}
        subsets = {"identifiable": dev["weight"].numpy() > 0, "censored_low_support": dev["weight"].numpy() == 0, **{mode: modes == mode for mode in np.unique(modes)}}
        subset_macro = {subset: {metric: float(np.mean([values[selected & (group == key)].mean() for key in np.unique(group[selected])])) for metric, values in metrics.items()} for subset, selected in subsets.items()}
        normalization_error = 0.0
        finite = True
        for start in range(0, len(labels), 32):
            block = np.asarray(raw[start:start + 32], dtype=np.float64)
            finite &= bool(np.isfinite(block).all())
            maximum = block.max(1)
            normalization_error = max(normalization_error, float(np.abs(maximum + np.log(np.exp(block - maximum[:, None]).sum(1))).max()))
        with np.load(run / row_name) as saved_rows:
            row_errors = {metric: float(np.max(np.abs(saved_rows[metric] - values))) for metric, values in metrics.items()}
            labels_prediction_match = np.array_equal(saved_rows["label"], labels) and np.array_equal(saved_rows["prediction"], predicted) and np.array_equal(saved_rows["pose_supervision_weight"], dev["weight"].numpy())
        audited[name][step] = {
            "synthetic_group_macro": macro, "by_support_and_mode_synthetic_group_macro": subset_macro,
            "difference_from_saved_macro": {metric: macro[metric] - saved["animal_macro"][metric] for metric in macro},
            "saved_row_maximum_absolute_difference": row_errors,
            "labels_predictions_support_weights_match": bool(labels_prediction_match),
            "raw_shape": list(raw.shape), "all_log_probabilities_finite": finite,
            "maximum_absolute_log_normalization_error": normalization_error,
            "elapsed_seconds": saved["elapsed_seconds"],
        }
        print(name, step, macro["nll"], macro["plane_angle_deg"], macro["hit_at_128"], flush=True)
        del raw
    hashes[name] = {}
    for artifact in artifact_names:
        with (run / artifact).open("rb") as stream:
            hashes[name][artifact] = hashlib.file_digest(stream, "sha256").hexdigest()

parent = torch.load(runs["capacity_002"] / "joint_model_step_03000.pt", map_location="cpu", weights_only=False)
initial = torch.load(runs["recovery_004"] / "joint_model_step_03000.pt", map_location="cpu", weights_only=False)
final = torch.load(runs["recovery_004"] / "joint_model_step_10000.pt", map_location="cpu", weights_only=False)
capacity_final = torch.load(runs["capacity_002"] / "joint_model_step_10000.pt", map_location="cpu", weights_only=False)
continuation_fields = ("model_state", "optimizer_state", "scaler_state", "torch_rng", "cuda_rng", "numpy_rng", "python_rng")
differences = []
stack = [(key, parent[key], initial[key]) for key in continuation_fields]
while stack:
    key, before, after = stack.pop()
    if isinstance(before, torch.Tensor):
        equal = isinstance(after, torch.Tensor) and before.dtype == after.dtype and torch.equal(before, after)
    elif isinstance(before, np.ndarray):
        equal = isinstance(after, np.ndarray) and before.dtype == after.dtype and np.array_equal(before, after)
    elif isinstance(before, dict):
        equal = isinstance(after, dict) and before.keys() == after.keys()
        if equal:
            stack.extend((f"{key}.{field}", value, after[field]) for field, value in before.items())
    elif isinstance(before, (list, tuple)):
        equal = isinstance(after, type(before)) and len(before) == len(after)
        if equal:
            stack.extend((f"{key}.{index}", value, after[index]) for index, value in enumerate(before))
    else:
        equal = before == after
    if not equal:
        differences.append(key)
changed = [key for key, value in final["model_state"].items() if not torch.equal(value, initial["model_state"][key])]
allowed = ("pose_model.histology_stem.", "pose_model.shared_encoder.", "pose_model.proposal_head_v6.")
checkpoint_optimizer_steps = {name: sorted({int(state["step"]) for state in checkpoint["optimizer_state"]["state"].values()}) for name, checkpoint in (("parent_3000", parent), ("recovery_3000", initial), ("recovery_10000", final), ("capacity_10000", capacity_final))}
trace_summary = {name: {
    "attempted_steps": len(trace), "applied_steps": sum(row["optimizer_step_applied"] for row in trace),
    "skipped_step_ids": [row["step"] for row in trace if not row["optimizer_step_applied"]],
    "exact_consecutive_step_ids": [row["step"] for row in trace] == list(range(3001 if name == "recovery_004" else 1, 10001)),
    "all_losses_finite": bool(np.isfinite([row["weighted_nll"] for row in trace]).all()),
    "nonfinite_gradient_step_ids": [row["step"] for row in trace if row["gradient_norm"] is None or not np.isfinite(row["gradient_norm"])],
    "last100_training_weighted_nll": float(np.mean([row["weighted_nll"] for row in trace[-100:]])),
} for name, trace in traces.items()}
parent_applied = sum(row["optimizer_step_applied"] for row in traces["capacity_002"] if row["step"] <= 3000)
common_sources = configs["capacity_002"]["source"]["file_sha256"].keys() & configs["recovery_004"]["source"]["file_sha256"].keys()
audit = {
    "scope": "CPU-only audit of exited frozen002/004; synthetic organizational-group diagnostics, not biological animal inference",
    "runs": {key: str(run) for key, run in runs.items()}, "matched_step_recomputations": audited,
    "trace": trace_summary, "checkpoint_optimizer_steps": checkpoint_optimizer_steps,
    "cumulative_recovery_applied_steps": parent_applied + trace_summary["recovery_004"]["applied_steps"],
    "parent_start_state_exactly_equal_fields": list(continuation_fields), "parent_start_state_difference_paths": differences,
    "identical_complete_training_schedule": bool(schedule_equal),
    "continuation_indexing": "source iterates cumulative3001..10000 and indexes preserved schedule[step-1]",
    "identical_training_and_development_identities": identities["capacity_002"] == identities["recovery_004"],
    "identity_intersection_counts": overlap, "identities_match_frozen_cache": cache_match,
    "development_records_match": dev["records"] == identities["recovery_004"]["internal_development"],
    "development_rows": len(labels), "development_organizational_groups": len(groups),
    "configuration_equal_fields": {key: configs["capacity_002"][key] == configs["recovery_004"][key] for key in ("seed", "steps", "batch_size", "learning_rate", "optimizer", "weight_decay", "evaluation_interval", "model_kwargs", "retrieval_shape_h_w", "cache_manifests", "atlas_binding", "catalogue_receipt_sha256")},
    "changed_common_training_sources": {key: [configs["capacity_002"]["source"]["file_sha256"][key], configs["recovery_004"]["source"]["file_sha256"][key]] for key in sorted(common_sources) if configs["capacity_002"]["source"]["file_sha256"][key] != configs["recovery_004"]["source"]["file_sha256"][key]},
    "parent_sha256_matches_receipt": hashes["capacity_002"]["joint_model_step_03000.pt"] == configs["recovery_004"]["parent_sha256"],
    "checkpoint_experiment_matches": json.loads(json.dumps(initial["experiment"])) == configs["recovery_004"] and json.loads(json.dumps(final["experiment"])) == configs["recovery_004"],
    "checkpoint_models_all_finite": all(bool(torch.isfinite(value).all()) for checkpoint in (parent, initial, final, capacity_final) for value in checkpoint["model_state"].values()),
    "recovery_changed_tensor_names": changed, "only_proposal_path_changed": all(key.startswith(allowed) for key in changed),
    "recovery_parameter_delta_l2": float(torch.sqrt(sum((final["model_state"][key].double() - initial["model_state"][key].double()).square().sum() for key in changed))),
    "artifact_sha256": hashes,
    "gate_decision": "FAIL: precision-corrected continuation remains inadequate for ordinary joint refinement, deployment or public benchmarking",
    "probabilities_calibrated": False,
}
(output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

fig, axes = plt.subplots(2, 2, figsize=(10, 6.3))
for name, label, color in (("capacity_002", "Original half-precision head", "#b04c3b"), ("recovery_004", "FP32 centered-head continuation", "#226f9c")):
    steps = sorted(audited[name])
    for ax, metric, ylabel, factor in zip(axes.flat[:3], ("nll", "plane_angle_deg", "hit_at_128"), ("Held-out catalogue NLL (nats)", "MAP plane-normal error (degrees)", "Nearest-cell top-128 recall (%)"), (1, 1, 100)):
        ax.plot(steps, [audited[name][step]["synthetic_group_macro"][metric] * factor for step in steps], marker="o", ms=3, color=color, label=label)
        ax.set_ylabel(ylabel)
    trace = [row for row in traces[name] if row["step"] > 3000]
    axes[1, 1].plot([trace[start + 99]["step"] for start in range(0, len(trace), 100)], [np.mean([row["weighted_nll"] for row in trace[start:start + 100]]) for start in range(0, len(trace), 100)], color=color, label=label)
axes[1, 1].set_ylabel("Training weighted NLL / 100 updates")
for ax in axes.flat:
    ax.set_xlabel("Cumulative attempted optimization step")
    ax.axvline(3000, color="gray", linestyle=":", linewidth=1)
    ax.grid(alpha=0.2)
axes[0, 0].legend(fontsize=8)
fig.suptitle("Numerical recovery improves stability; coarse localization still fails")
fig.text(0.5, 0.015, "640 fixed held-out sections / 40 single-atlas organizational groups; not biological animal inference.\nStep3,000 is the same checkpoint evaluated with different head precision; subsequent training follows identical rows.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
figure = repository / "docs" / "publication" / "proposal_precision_recovery_002_004.png"
fig.savefig(figure, dpi=170)
plt.close(fig)
print(json.dumps({"trace": trace_summary, "optimizer_steps": checkpoint_optimizer_steps, "continuation_differences": differences, "changed_sources": audit["changed_common_training_sources"], "final_recovery_macro": audited["recovery_004"][10000]["synthetic_group_macro"], "audit": str(output / "audit.json"), "figure": str(figure)}, indent=2))
assert not differences and schedule_equal and audit["identical_training_and_development_identities"]
assert not any(value for overlaps in overlap.values() for value in overlaps.values()) and all(cache_match.values())
assert audit["parent_sha256_matches_receipt"] and audit["checkpoint_experiment_matches"] and all(audit["configuration_equal_fields"].values())
assert audit["checkpoint_models_all_finite"] and audit["only_proposal_path_changed"] and changed
assert checkpoint_optimizer_steps["recovery_10000"] == [audit["cumulative_recovery_applied_steps"]]
assert all(record["exact_consecutive_step_ids"] and record["all_losses_finite"] for record in trace_summary.values())
assert all(record["all_log_probabilities_finite"] and record["maximum_absolute_log_normalization_error"] < 2e-5 and record["labels_predictions_support_weights_match"] and max(abs(value) for value in record["difference_from_saved_macro"].values()) < 2e-6 for evaluations in audited.values() for record in evaluations.values())
