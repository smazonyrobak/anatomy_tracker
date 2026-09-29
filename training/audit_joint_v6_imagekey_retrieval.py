"""Run only AFTER image-key001 exits; CPU artifact audit, no model execution."""

import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
sys.dont_write_bytecode = True

import hashlib
import json
import time

import numpy as np
import torch

RUN = ROOT / "runs/joint_v6_imagekey_retrieval_001"
PREPARED = ROOT / "runs/joint_v6_proposal_substantive_001"
BASELINE = ROOT / "runs/joint_v6_imagekey_baseline_003_step4000/baseline.json"
OUTPUT = ROOT / "runs/joint_v6_imagekey_retrieval_001_independent_audit"
BASELINE_SHA = "5672303902e0b3d108507e40a883b848418c656c20e1bdc48754469009961eeb"
DRIVER_SHA = "caa126ea313b52dda21784a4493f0564f3e74ca6cdf46a5c3e2a645bac8321e6"
torch.set_num_threads(4)
started = time.perf_counter()
completed = json.loads((RUN / "completed.json").read_text())
config = json.loads((RUN / "experiment.json").read_text())
receipt = json.loads((RUN / "development_metrics_step_04000.json").read_text())
initial_receipt = json.loads((RUN / "development_metrics_step_00000.json").read_text())
baseline = json.loads(BASELINE.read_text())
checks, hashes = {}, {}
expected = {BASELINE: BASELINE_SHA, RUN / "baseline003_step4000.json": BASELINE_SHA,
            RUN / "experiment_source.py": DRIVER_SHA,
            RUN / "negative_schedule.npz": config["negative_schedule_sha256"],
            RUN / "joint_model_step_00000.pt": initial_receipt["checkpoint_sha256"],
            RUN / "gallery_descriptors_step_00000.npy": initial_receipt["gallery_sha256"],
            RUN / "joint_model_step_04000.pt": receipt["checkpoint_sha256"],
            RUN / "gallery_descriptors_step_04000.npy": receipt["gallery_sha256"]}
expected.update({PREPARED / name: digest for name, digest in config["prepared_source_sha256"].items()})
expected[RUN / "catalogue.pt"] = config["prepared_source_sha256"]["catalogue.pt"]
expected.update({RUN / f"replay003_{name}": digest for name, digest in config["replay_sha256"].items()})
expected.update({RUN / "source" / name: digest for name, digest in config["source"]["file_sha256"].items()})
paths = set(expected) | {Path(__file__), RUN / "completed.json", RUN / "experiment.json", RUN / "source_diff.patch",
    RUN / "development_metrics_step_04000.json", RUN / "development_rows_step_04000.npz",
    RUN / "development_metrics_step_00000.json",
    RUN / "development_log_probability_step_04000.npy", RUN / "training_trace.jsonl",
    RUN / "training_identities.json", RUN / "internal_development_identities.json", RUN / "joint_model_step_00000.pt"}
# This whole-run inventory is permitted only in the explicitly post-exit audit.
run_files = sorted(path for path in RUN.rglob("*") if path.is_file())
paths.update(run_files)
for path in sorted(paths):
    with path.open("rb") as stream:
        hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
for path, digest in expected.items():
    checks[f"sha256:{path}"] = hashes[str(path)] == digest
checks["baseline_fixed_budget"] = baseline["step"] == baseline["applied_updates_through_4000"] == 4000
checks["baseline_same_catalogue_and_rows"] = all(hashes[str(PREPARED / name)] == baseline["artifact_sha256"][str(PREPARED / name)] for name in ("catalogue.pt", "internal_development_prepared.pt"))
checks["pinned_baseline_and_driver_binding"] = config["baseline_report"]["sha256"] == BASELINE_SHA and config["source"]["file_sha256"]["training/run_joint_v6_imagekey_retrieval.py"] == DRIVER_SHA
checks["receipt_config_binding"] = receipt["source"] == config["source"] and receipt["gallery"] == config["gallery"] and receipt["temperature"] == config["temperature"] == .1
checks["endpoint_receipt_steps"] = initial_receipt["step"] == 0 and receipt["step"] == 4000 and initial_receipt["source"] == config["source"]
checks["catalogue_receipt_binding"] = receipt["catalogue_sha256"] == config["prepared_source_sha256"]["catalogue.pt"] and config["catalogue_receipt_sha256"] == baseline["catalogue_receipt_sha256"]
checks["planned_protocol"] = config["steps"] == 4000 and config["batch_size"] == 16 and config["generated_per_batch"] == 8 and config["seed"] == 2026092805 and config["negative_seed"] == 2026092910
checks["declared_not_calibrated"] = receipt["probabilities_calibrated"] is False
checks["fixed_gallery_psf"] = config["gallery"]["shape_h_w"] == [96, 96] and config["gallery"]["thickness_um"] == 50. and config["gallery"]["cell_count"] == 98304 and config["gallery"]["psf"] == "normalized boxcar-trapezoid9"
train = torch.load(PREPARED / "training_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
dev = torch.load(PREPARED / "internal_development_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
catalogue = torch.load(RUN / "catalogue.pt", map_location="cpu", weights_only=False)
rows = np.load(RUN / "development_rows_step_04000.npz", allow_pickle=False)
raw = np.load(RUN / "development_log_probability_step_04000.npy", mmap_mode="r", allow_pickle=False)
bank = np.load(RUN / "gallery_descriptors_step_04000.npy", mmap_mode="r", allow_pickle=False)
generated = np.load(RUN / "replay003_generated_schedule.npz", allow_pickle=False)
frozen = np.load(RUN / "replay003_frozen_training_row_indices.npy", allow_pickle=False)[:4000]
negative = np.load(RUN / "negative_schedule.npz", allow_pickle=False)
labels, weight = dev["label"].numpy(), dev["weight"].numpy()
groups = np.array([row["animal_id"] for row in dev["records"]])
modes = np.array([row["selected_mode"] for row in dev["records"]])
eligible = weight > 0
checks["exact640rows98304cells"] = raw.shape == (640, 98304) and raw.dtype == np.float32 and len(labels) == 640
checks["exact_gallery_and_query_shapes"] = bank.shape == (98304, 2, 256) and bank.dtype == np.float16 and rows["query_descriptor"].shape == (640, 256)
checks["original_labels_and_censor_weights"] = np.array_equal(rows["label"], labels) and np.array_equal(rows["pose_supervision_weight"], weight) and int(eligible.sum()) == 611
checks["train5120_development640"] = len(train["records"]) == 5120 and len(dev["records"]) == 640
checks["saved_identity_order"] = json.loads((RUN / "training_identities.json").read_text()) == train["records"] and json.loads((RUN / "internal_development_identities.json").read_text()) == dev["records"]
for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    checks[f"disjoint:{key}"] = not (({r[key] for r in train["records"]} | set(generated[key][:32000])) & {r[key] for r in dev["records"]})
negative_rng = np.random.default_rng(2026092910)
checks["negative_global_schedule"] = np.array_equal(negative["global_cell_index"], negative_rng.integers(0, 98304, (4000, 32), dtype=np.int64))
checks["negative_local_rank_schedule"] = np.array_equal(negative["local_pool_rank"], negative_rng.integers(0, 8, (4000, 16), dtype=np.int64))
trace = [json.loads(line) for line in (RUN / "training_trace.jsonl").read_text().splitlines()]
checks["trace_steps_and_applied4000"] = [row["step"] for row in trace] == list(range(1, 4001)) and [row["optimizer_steps_applied"] for row in trace] == list(range(1, 4001))
checks["completed_applied4000"] = completed["steps"] == completed["optimizer_steps_applied"] == 4000
checks["completed_replay_counts"] = completed["replayed_generated_observations"] == completed["replayed_frozen_presentations"] == 32000 and completed["new_query_observations"] == 0
schedule_ok, truth_retained, candidate_union_ok, censor_ok, finite_trace = True, True, True, True, True
supervised_count = 0
for i, record in enumerate(trace):
    positions = np.arange(i * 8, (i + 1) * 8)
    schedule_ok &= np.array_equal(record["generated_indices"], positions) and np.array_equal(record["frozen_row_indices"], frozen[i])
    target = np.concatenate((generated["cell_index"][positions], train["label"][frozen[i]].numpy()))
    candidate = np.asarray(record["candidate_ids"])
    truth_retained &= bool(np.isin(target, candidate).all())
    candidate_union_ok &= np.array_equal(candidate, np.unique(np.concatenate((target, negative["global_cell_index"][i], record["local_negative_ids"])))) and len(candidate) <= 64
    generated_weight = ((np.array(record["finite_support_mass_px"]) >= 64) & (np.array(record["visible_support_mass_px"]) >= 64)).astype(float)
    censor_ok &= np.array_equal(generated_weight, record["generated_point_pose_weight"])
    supervised_count += int(generated_weight.sum())
    finite_trace &= bool(np.isfinite([record["sampled_nll"], record["gradient_norm"], record["training_seconds"]]).all())
checks.update(exact_query_replay_schedule=bool(schedule_ok), all_training_own_truths_retained=bool(truth_retained),
              training_candidate_union=bool(candidate_union_ok), generated_censoring=bool(censor_ok), finite_training_trace=bool(finite_trace),
              generated_supervision_counts=completed["generated_supervised"] == supervised_count and completed["generated_censored"] == 32000 - supervised_count)
initial = torch.load(RUN / "joint_model_step_00000.pt", map_location="cpu", weights_only=False, mmap=True)
endpoint = torch.load(RUN / "joint_model_step_04000.pt", map_location="cpu", weights_only=False, mmap=True)
checks["checkpoint_config_binding"] = json.loads(json.dumps(endpoint["experiment"])) == config
checks["initial_checkpoint_config_binding"] = json.loads(json.dumps(initial["experiment"])) == config
checks["checkpoint_budget_and_scope"] = endpoint["step"] == endpoint["optimizer_steps_applied"] == 4000 and endpoint["phase"] == "experimental_image_key_proposal_only" and endpoint["probabilities_calibrated"] is False
checks["initial_checkpoint_zero"] = initial["step"] == initial["optimizer_steps_applied"] == 0
checks["whole_model_keys"] = set(initial["model_state"]) == set(endpoint["model_state"])
trainable_prefixes = tuple(config["trained_modules"])
checks["untrained_modules_unchanged"] = all(torch.equal(initial["model_state"][name], value) for name, value in endpoint["model_state"].items() if not name.startswith(trainable_prefixes))
checks["trained_modules_changed"] = all(any(name.startswith(prefix) and not torch.equal(initial["model_state"][name], value) for name, value in endpoint["model_state"].items()) for prefix in trainable_prefixes)
checks["checkpoint_finite"] = all(bool(torch.isfinite(value).all()) for value in endpoint["model_state"].values() if torch.is_floating_point(value))
checks["optimizer_budget"] = bool(endpoint["optimizer_state"]["state"]) and all(int(value["step"]) == 4000 for value in endpoint["optimizer_state"]["state"].values())
checks["rng_retained"] = all(key in endpoint for key in ("torch_rng", "cuda_rng", "numpy_rng", "python_rng"))
del initial, endpoint
support_origin = np.array(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"], dtype=np.float64)
geometry = {}
for name, state in (("truth", dev["truth_state"].numpy()), ("catalogue", catalogue["arrays"]["cell_states_float64"])):
    state = np.asarray(state, dtype=np.float64)
    u = state[:, 3:6] / np.linalg.norm(state[:, 3:6], axis=1, keepdims=True)
    v = state[:, 6:9] - (state[:, 6:9] * u).sum(1, keepdims=True) * u
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    edge_u, edge_v = u * np.exp(state[:, 9:10]), (state[:, 11:12] * u + v) * np.exp(state[:, 10:11])
    origin = state[:, :3] - .5 * (edge_u + edge_v)
    normal = np.cross(edge_u, edge_v)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    geometry[name] = (normal, ((state[:, :3] - support_origin) * normal).sum(1),
                      np.stack((origin, origin + edge_u, origin + edge_v, origin + edge_u + edge_v), axis=1), np.stack((u, v, normal), axis=-1))
truth_normal, truth_offset, truth_corners, truth_frame = geometry["truth"]
cell_normal, cell_offset, cell_corners, cell_frame = geometry["catalogue"]
metrics = {key: np.empty(640, dtype=np.float64) for key in ("nll", "truth_rank", "plane_angle_deg", "normal_offset_error_um", "frame_corner_rms_um", "antipodal_frame_angle_deg", "normal_marginal_nll", "normal_marginal_map_angle_deg", "hit_at_1", "hit_at_8", "hit_at_32", "hit_at_128", "physical_plane_capture_at_32", "physical_plane_capture_at_128", "frame_corner_capture_at_32", "frame_corner_capture_at_128")}
top_indices = np.empty((640, 128), dtype=np.int64)
query = rows["query_descriptor"].astype(np.float64)
bank = np.asarray(bank, dtype=np.float64)
checks["descriptors_finite_nonzero"] = bool(np.isfinite(query).all() and np.isfinite(bank).all() and (np.linalg.norm(query, axis=-1) > 0).all() and (np.linalg.norm(bank, axis=-1) > 0).all())
query /= np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1e-12)
bank /= np.maximum(np.linalg.norm(bank, axis=-1, keepdims=True), 1e-12)
log_cell = catalogue["tensors"]["cell_log_mass"][0].float().numpy().astype(np.float64)
log_rep = catalogue["tensors"]["representation_log_weight"][0].float().numpy().astype(np.float64)
raw_finite, normalization_error, descriptor_replay_error, representation_error = True, 0., 0., 0.
for start in range(0, 640, 8):
    stop = start + 8
    block = np.asarray(raw[start:stop], dtype=np.float64)
    raw_finite &= bool(np.isfinite(block).all())
    normalization_error = max(normalization_error, float(np.abs(np.logaddexp.reduce(block, axis=1)).max()))
    selected = np.argsort(-block, axis=1, kind="stable")[:, :128]
    top_indices[start:stop] = selected
    representation_score = np.einsum("bd,nrd->bnr", query[start:stop], bank, optimize=True) / .1 + log_rep[None]
    score = np.logaddexp.reduce(representation_score, axis=-1) + log_cell[None]
    reconstructed = score - np.logaddexp.reduce(score, axis=-1)[:, None]
    descriptor_replay_error = max(descriptor_replay_error, float(np.abs(reconstructed - block).max()))
    rep_probability = np.exp(representation_score - np.logaddexp.reduce(representation_score, axis=-1)[..., None])
    representation_error = max(representation_error, float(np.abs(rep_probability[np.arange(8), labels[start:stop]] - rows["truth_cell_representation_probability"][start:stop]).max()), float(np.abs(rep_probability[np.arange(8)[:, None], selected] - rows["top128_representation_probability"][start:stop]).max()))
    normals = cell_normal[selected]
    dot = (normals * truth_normal[start:stop, None]).sum(-1)
    angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(normals, truth_normal[start:stop, None]), axis=-1), np.abs(dot)))
    offset = np.abs(cell_offset[selected] - np.where(dot < 0, -1, 1) * truth_offset[start:stop, None])
    corners = cell_corners[selected]
    corner_rms = np.sqrt(np.minimum(((corners - truth_corners[start:stop, None])**2).sum(-1).mean(-1), ((corners[:, :, [1, 0, 3, 2]] - truth_corners[start:stop, None])**2).sum(-1).mean(-1)))
    truth_lp = block[np.arange(8), labels[start:stop]]
    rank = (block > truth_lp[:, None]).sum(-1) + ((block == truth_lp[:, None]) & (np.arange(98304)[None] < labels[start:stop, None])).sum(-1) + 1
    frame = cell_frame[selected[:, 0]]
    cosine = ((frame * truth_frame[start:stop]).sum((-2, -1)) - 1) / 2
    flip_cosine = ((frame * np.array([-1., 1., -1.]) * truth_frame[start:stop]).sum((-2, -1)) - 1) / 2
    normal_lp = np.logaddexp.reduce(block.reshape(8, 384, 256), axis=-1)
    marginal_normals = cell_normal[normal_lp.argmax(-1) * 256]
    metrics["nll"][start:stop], metrics["truth_rank"][start:stop] = -truth_lp, rank
    metrics["plane_angle_deg"][start:stop], metrics["normal_offset_error_um"][start:stop] = angle[:, 0], offset[:, 0]
    metrics["frame_corner_rms_um"][start:stop] = corner_rms[:, 0]
    metrics["antipodal_frame_angle_deg"][start:stop] = np.degrees(np.arccos(np.clip(np.maximum(cosine, flip_cosine), -1, 1)))
    metrics["normal_marginal_nll"][start:stop] = -normal_lp[np.arange(8), labels[start:stop] // 256]
    metrics["normal_marginal_map_angle_deg"][start:stop] = np.degrees(np.arctan2(np.linalg.norm(np.cross(marginal_normals, truth_normal[start:stop]), axis=-1), np.abs((marginal_normals * truth_normal[start:stop]).sum(-1))))
    for k in (1, 8, 32, 128):
        metrics[f"hit_at_{k}"][start:stop] = rank <= k
    for k in (32, 128):
        metrics[f"physical_plane_capture_at_{k}"][start:stop] = ((angle[:, :k] <= 10.) & (offset[:, :k] <= 500.)).any(-1)
        metrics[f"frame_corner_capture_at_{k}"][start:stop] = (corner_rms[:, :k] <= 1000.).any(-1)
    if stop % 80 == 0:
        print(json.dumps({"audited_rows": stop, "descriptor_log_probability_max_error": descriptor_replay_error}), flush=True)
checks["raw_finite_and_normalized"] = raw_finite and normalization_error <= 2e-5
checks["full_gallery_descriptor_reconstruction"] = descriptor_replay_error <= 5e-5
checks["representation_conditionals_reconstruction"] = representation_error <= 5e-5
checks["saved_top128_and_prediction"] = np.array_equal(rows["top128_cell_index"], top_indices) and np.array_equal(rows["prediction"], top_indices[:, 0])
row_metric_errors = {key: float(np.max(np.abs(value - rows[key]))) for key, value in metrics.items() if key in rows.files}
checks["saved_row_metrics_recomputed"] = all(error <= (1e-4 if "angle" in key or "offset" in key else 5e-5) for key, error in row_metric_errors.items())
subsets = {"all": np.ones(640, dtype=bool), "support_eligible": eligible, "censored_low_support": ~eligible,
           **{mode: modes == mode for mode in np.unique(modes)},
           **{f"{mode}/support_eligible": (modes == mode) & eligible for mode in np.unique(modes)},
           **{f"{mode}/censored_low_support": (modes == mode) & ~eligible for mode in np.unique(modes)}}
aggregates = {name: {"row_count": int(mask.sum()), "group_count": len(np.unique(groups[mask])),
    "group_macro": {key: float(np.mean([value[mask & (groups == group)].mean() for group in np.unique(groups[mask])])) if mask.any() else None for key, value in metrics.items()},
    "capture_row_counts": {key: int(value[mask].sum()) for key, value in metrics.items() if "capture_at_" in key or key.startswith("hit_at_")}}
    for name, mask in subsets.items()}
current, previous = aggregates["support_eligible"]["group_macro"], baseline["aggregates"]["support_eligible"]["group_macro"]
gate_values = {"eligible_normal_improvement_deg": previous["plane_angle_deg"] - current["plane_angle_deg"],
               "eligible_top32_plane_capture_improvement": current["physical_plane_capture_at_32"] - previous["physical_plane_capture_at_32"]}
gates = {"normal_improvement_at_least5deg": gate_values["eligible_normal_improvement_deg"] >= 5.,
         "top32_plane_capture_improvement_at_least005": gate_values["eligible_top32_plane_capture_improvement"] >= .05}
for mode in np.unique(modes):
    key = f"{mode}/support_eligible"
    now, before = aggregates[key]["group_macro"], baseline["aggregates"][key]["group_macro"]
    gate_values[mode] = {"normal_regression_deg": now["plane_angle_deg"] - before["plane_angle_deg"], "top32_capture_regression": before["physical_plane_capture_at_32"] - now["physical_plane_capture_at_32"]}
    gates[f"{mode}:normal_regression_at_most2deg"] = gate_values[mode]["normal_regression_deg"] <= 2.
    gates[f"{mode}:top32_capture_regression_at_most002"] = gate_values[mode]["top32_capture_regression"] <= .02
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / "audit_source.py").write_bytes(Path(__file__).read_bytes())
np.savez(OUTPUT / "recomputed_rows.npz", label=labels, weight=weight, group=groups, mode=modes, top128_cell_index=top_indices, **metrics)
result = {"integrity_passed": all(checks.values()), "performance_gates_passed": all(gates.values()),
          "advance_to_joint_integration_pilot": all(checks.values()) and all(gates.values()),
          "integrity_checks": checks, "failed_integrity_checks": [key for key, value in checks.items() if not value],
          "performance_gates": gates, "gate_values": gate_values, "aggregates": aggregates,
          "raw_log_normalization_max_error": normalization_error, "descriptor_full_gallery_log_probability_max_error": descriptor_replay_error,
          "representation_probability_max_error": representation_error, "saved_row_metric_max_errors": row_metric_errors,
          "artifact_sha256": hashes, "rows_sha256": hashlib.sha256((OUTPUT / "recomputed_rows.npz").read_bytes()).hexdigest(),
          "frozen_run_inventory_sha256": {path.relative_to(RUN).as_posix(): hashes[str(path)] for path in run_files},
          "elapsed_seconds": time.perf_counter() - started, "baseline_sha256": BASELINE_SHA,
          "scope": "CPU independent numerical artifact audit; synthetic organizational groups, not biological animals; raw uncalibrated full-gallery retrieval, not deployment/benchmark/trajectory qualification",
          "limits": "Descriptor-to-posterior reconstructed over every cell without exclusions. Saved source/checkpoint hashes bind descriptors; no model re-encoding, rendering, full mined-neighbour-pool replay or authenticity claim beyond these saved bindings."}
(OUTPUT / "audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps({key: result[key] for key in ("integrity_passed", "performance_gates_passed", "advance_to_joint_integration_pilot", "failed_integrity_checks", "gate_values", "elapsed_seconds")}), flush=True)
