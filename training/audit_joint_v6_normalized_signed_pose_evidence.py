"""Post-exit fixed-RMS signed-pose audit; independent NumPy geometry, matched control."""
import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
sys.dont_write_bytecode = True

import hashlib
import json
RUN_COMPLETION_SHA256 = "UNSET"
assert len(RUN_COMPLETION_SHA256) == 64, "Confirm exit and pin completed.json before any run access"

import numpy as np
import torch

RUN = ROOT / "runs/joint_v6_normalized_signed_pose_evidence_001"
COMPARATOR = ROOT / "runs/joint_v6_signed_pose_evidence_001"
COMPARATOR_COMPLETION_SHA256 = "7e814be7b2207e54544b67f6180a61808de17ea6bc530b41930092ced90828c1"
COMPARATOR_SCHEDULE_SHA256 = "e8aaa6eb1ff7e70097cbf1eeb1f6b9fc4d2ba4f01f26fc6d91b671c90c8dc0b9"
DATA = ROOT / "data/joint_v6_coherent_subject_cohort_sections_002"
OUTPUT = ROOT / "runs/joint_v6_normalized_signed_pose_evidence_001_independent_audit"
COMPARATOR_AUDIT_SHA256 = "7acdc3ef87f85691b296f67fc828428aa090b1afed88a5ce05853a6e81853ad5"
PREFLIGHT_SHA256 = "1d87df5a33bd2da6df570a84108b9e9a5b0fec888ef10f4c52578b8f7f4d68c4"
SCALE_RECEIPT = ROOT / "runs/joint_v6_signed_pose_activation_train_001/result.json"
SCALE_RECEIPT_SHA256 = "5762fdcda9d5eed305835b6e77c9f262710600ef97aa286fd1a1e167f091558b"
SIGNED_POSE_COST_RMS = (.08283151464815028, .0678978954706466, .07608602924318171)
PARENT_SHA256 = "d4d706e8d80e53a3638a70e79ce8661ff4af41f7b846143aa1ec68372bfb2ae5"
PARENT_AUDIT_SHA256 = "f9eb9c6845e048e5fc3ca840a4c5effcf48c1d6ed98afef9daa65a3983e4ca9b"
TRAINABLE = ("pose_model.refinement_pair_encoder.", "pose_model.recurrent_cell.", "pose_model.recurrent_update.", "pose_model.recurrent_log_likelihood.", "pose_model.coordinate_evidence.", "pose_model.signed_pose_evidence.", "ribbon_field_head.")
MODES = ("raw", "exact_black", "imperfect_brush")
SMALL = (.06, .06, 250., .08, 300., 300., .05, .05, .05)
LARGE = (.15, .15, 600., .20, 600., 600., .12, .12, .12)
torch.set_num_threads(2)
hashes = {}


def sha(path):
    path = Path(path)
    if str(path) not in hashes:
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes[str(path)]


def frame(state):
    state = np.asarray(state, dtype=np.float64)
    u = state[..., 3:6] / np.linalg.norm(state[..., 3:6], axis=-1, keepdims=True)
    v = state[..., 6:9] - (state[..., 6:9] * u).sum(-1, keepdims=True) * u
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    rotation = np.stack((u, v, np.cross(u, v)), axis=-1)
    a, b = np.exp(state[..., 9]), np.exp(state[..., 10])
    edge_u, edge_v = u * a[..., None], u * (state[..., 11] * b)[..., None] + v * b[..., None]
    origin = state[..., :3] - .5 * (edge_u + edge_v)
    y, x = np.mgrid[:96, :96] / 96.
    plane = origin[..., None, None, :] + x[..., None] * edge_u[..., None, None, :] + y[..., None] * edge_v[..., None, None, :]
    return plane, rotation


def distances(prediction, target, weight, normal):
    error = prediction - target
    distance = np.linalg.norm(error, axis=-1)
    mass = weight.sum()
    if mass <= 0:
        return {"mean_um": np.nan, "point_p95_um": np.nan, "normal_mean_um": np.nan, "tangential_mean_um": np.nan}
    normal_error = np.abs(error @ normal)
    tangent = np.sqrt(np.maximum(0., (error * error).sum(-1) - normal_error**2))
    order = np.argsort(distance.ravel(), kind="stable")
    index = np.searchsorted(weight.ravel()[order].cumsum(), .95 * mass, side="left")
    return {"mean_um": float(np.sum(weight * distance) / mass), "point_p95_um": float(distance.ravel()[order[min(index, len(order) - 1)]]), "normal_mean_um": float(np.sum(weight * normal_error) / mass), "tangential_mean_um": float(np.sum(weight * tangent) / mass)}


assert sha(RUN / "completed.json") == RUN_COMPLETION_SHA256
completion = json.loads((RUN / "completed.json").read_text())
config = json.loads((RUN / "experiment.json").read_text())
assert completion["experiment"] == config and completion["source_unchanged"] is True
assert sha(COMPARATOR / "completed.json") == COMPARATOR_COMPLETION_SHA256 == config["comparator"]["completion_sha256"]
assert sha(COMPARATOR / "schedule.pt") == COMPARATOR_SCHEDULE_SHA256 == config["comparator"]["schedule_sha256"]
assert sha(COMPARATOR / "source/training/run_joint_v6_signed_pose_evidence.py") == config["comparator"]["archived_driver_sha256"] == "2a45967aa9fe77fe11e51596da50a9ef95b575f844e9d869b983f4259a109da0"
comparator_completion = json.loads((COMPARATOR / "completed.json").read_text())
assert sha(RUN / "comparator_completed.json") == COMPARATOR_COMPLETION_SHA256
assert config["comparator"]["archived_source_sha256"] == comparator_completion["experiment"]["source"]["file_sha256"]
assert sha(RUN / "comparator_independent_audit.json") == config["comparator"]["independent_audit_sha256"] == COMPARATOR_AUDIT_SHA256
comparator_audit = json.loads((RUN / "comparator_independent_audit.json").read_text())
assert comparator_audit["integrity_passed"] and not comparator_audit["conditional_local_gate_passed"]
assert comparator_audit["artifact_sha256"][str(COMPARATOR / "completed.json")] == COMPARATOR_COMPLETION_SHA256
assert sha(RUN / "fixed_train_rms_receipt.json") == sha(SCALE_RECEIPT) == SCALE_RECEIPT_SHA256 == config["fixed_train_rms"]["receipt_sha256"]
scale_receipt = json.loads((RUN / "fixed_train_rms_receipt.json").read_text())
assert tuple(scale_receipt["metrics"]["raw_rms_per_axis_shared_sign"]) == tuple(config["fixed_train_rms"]["values"]) == SIGNED_POSE_COST_RMS
assert scale_receipt["train_schedule_index_zero_based"] == config["fixed_train_rms"]["train_schedule_index_zero_based"] == 250
assert scale_receipt["observation_indices"] == config["fixed_train_rms"]["training_observation_indices"] == [1464, 101, 19, 753]
assert all(row["split"] == "train" for row in scale_receipt["identities"])
assert scale_receipt["input_sha256"][str(COMPARATOR / "completed.json")] == COMPARATOR_COMPLETION_SHA256
assert config["fixed_train_rms"]["shared_by_plus_minus_sign"] and not config["fixed_train_rms"]["online_or_per_sample_adaptation"] and not config["fixed_train_rms"]["development_used"]
assert sha(RUN / "preflight.json") == config["preflight_sha256"] == PREFLIGHT_SHA256
preflight = json.loads((RUN / "preflight.json").read_text())
assert preflight["passed"] and preflight["optimizer_steps_applied"] == 0
assert all(preflight[key] for key in ("common_initialization_and_cpu_rng_exact", "all_zero_projection_outputs_exact", "all_parameters_unchanged"))
assert tuple(preflight["model_kwargs"]["signed_pose_cost_rms"]) == SIGNED_POSE_COST_RMS and preflight["comparison_model_kwargs"]["signed_pose_cost_rms"] is None
assert preflight["input_sha256"][str(SCALE_RECEIPT)] == SCALE_RECEIPT_SHA256
shared_source = set(preflight["source"]["file_sha256"]) & set(config["source"]["file_sha256"])
assert {"training/arbitrary_plane_joint_model_v6.py", "training/arbitrary_plane_recurrent_model_v6.py", "training/arbitrary_plane_recurrent_model.py", "training/arbitrary_plane_ribbon_v6.py", "training/arbitrary_plane_full_frame_primitives.py", "training/arbitrary_plane_joint_uncertainty.py"} <= shared_source
assert all(preflight["source"]["file_sha256"][name] == config["source"]["file_sha256"][name] for name in shared_source)
for key in ("seed", "steps", "batch_size", "precision", "optimizer", "learning_rate", "weight_decay", "clip_norm", "refinement_updates", "pose_prefix", "parent_sha256", "cohort_completion_sha256", "schedule_sha256", "trainable_prefixes", "small_noise_first_250", "large_noise_remaining_and_development", "loss", "director_target", "promotion_gate"):
    assert config[key] == comparator_completion["experiment"][key], key
assert config["promotion_gate"]["oracle_normal_reduction_min"] == .20 and config["promotion_gate"]["oracle_normal_nonregression_each_mode_and_subject"]
assert (config["seed"], config["steps"], config["batch_size"], config["refinement_updates"], config["pose_prefix"]) == (2026092913, 2000, 4, 3, 1)
assert config["learning_rate"] == 2e-4 and config["weight_decay"] == 1e-4 and config["clip_norm"] == 1.
assert config["precision"] == "FP32; AMP/TF32 disabled" and tuple(config["trainable_prefixes"]) == TRAINABLE
assert tuple(config["small_noise_first_250"]) == SMALL and tuple(config["large_noise_remaining_and_development"]) == LARGE
assert config["parent_sha256"] == PARENT_SHA256 and sha(config["parent_checkpoint"]) == PARENT_SHA256
assert sha(RUN / "parent_independent_audit.json") == PARENT_AUDIT_SHA256 == config["parent_independent_audit_sha256"]
assert json.loads((RUN / "parent_independent_audit.json").read_text())["advance_to_joint_integration_pilot"] is True
assert sha(DATA / "completed.json") == config["cohort_completion_sha256"] == sha(RUN / "parent_cohort_completed.json")
cohort = json.loads((DATA / "completed.json").read_text())
assert cohort["section_count"] == 640 and cohort["observation_count"] == 1920
for name, expected in config["source"]["file_sha256"].items():
    assert sha(RUN / "source" / name) == expected
assert sha(RUN / "schedule.pt") == config["schedule_sha256"]
assert sha(RUN / "schedule.pt") == COMPARATOR_SCHEDULE_SHA256
schedule = torch.load(RUN / "schedule.pt", map_location="cpu", weights_only=False)
identities = json.loads((RUN / "observation_identities.json").read_text())
targets, expected_rows = {}, []
for si, record in enumerate(cohort["sections"]):
    for name, expected in record["artifact_sha256"].items():
        assert sha(DATA / name) == expected
    for mode in MODES:
        expected_rows.append({**record["lineage"], "observation_id": f"{record['lineage']['section_id']}-{mode}", "selected_mode": mode, "section_array_index": si, "support_information_eligible": not record["by_mode"][mode]["support_censored"]})
    if record["lineage"]["split"] != "development":
        continue
    metadata = json.loads((DATA / record["artifacts"]["metadata"]).read_text())
    with np.load(DATA / record["artifacts"]["arrays"], allow_pickle=False) as arrays:
        ouv = arrays[metadata["canonical_anatomy_plane_fit"]["arrays"]["physical_ouv_ap_dv_ml_um_float64"]["__ndarray__"]].reshape(3, 3)
        normal = np.cross(ouv[1], ouv[2]); normal /= np.linalg.norm(normal)
        targets[si] = {
            "centre": arrays[metadata["target_centre_ccf_coordinates_ap_dv_ml_um_float64"]["__ndarray__"]],
            "slab": arrays[metadata["target_psf_ccf_coordinates_ap_dv_ml_um_float64"]["__ndarray__"]],
            "offsets": arrays[metadata["axial_offsets_um_float64"]["__ndarray__"]],
            "weights": arrays[metadata["axial_weights_float64"]["__ndarray__"]],
            "reflection": int(arrays[metadata["reflection_xy"]["__ndarray__"]][0]), "normal": normal,
            "visible": {item["selected_mode"]: arrays[item["visible_finite_support_float32"]["__ndarray__"]].astype(np.float64) for item in metadata["observations"]},
        }
assert len(identities) == len(expected_rows) == 1920 and len(targets) == 128
assert all(all(row[key] == expected[key] for key in expected) for row, expected in zip(identities, expected_rows))
split = np.array([row["split"] for row in identities])
subject = np.array([row["subject_id"] for row in identities])
eligible = np.array([row["support_information_eligible"] for row in identities])
dev_rows = np.flatnonzero(split == "development")
assert len(dev_rows) == 384 and np.array_equal(schedule["development_observation_index"].numpy(), dev_rows)
for key in ("subject_id", "animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    assert not {row[key] for row in identities if row["split"] == "train"} & {row[key] for row in identities if row["split"] == "development"}
train_index = schedule["training_observation_index"].numpy()
assert train_index.shape == (2000, 4) and np.all(split[train_index] == "train") and np.all(eligible[train_index])
train_subjects = np.unique(subject[split == "train"])
assert len(train_subjects) == 8 and np.array_equal(subject[train_index], train_subjects[schedule["training_subject_index"].numpy()])
perturbation = schedule["training_perturbation"].numpy()
assert perturbation.shape == (2000, 4, 9) and np.isfinite(perturbation).all()
assert np.all(np.abs(perturbation[:250]) <= np.asarray(SMALL, dtype=np.float32)) and np.all(np.abs(perturbation[250:]) <= np.asarray(LARGE, dtype=np.float32))
assert schedule["development_perturbation_by_section"].shape == (640, 9)
assert torch.isfinite(schedule["development_perturbation_by_section"]).all() and torch.all(schedule["development_perturbation_by_section"].abs() <= torch.tensor(LARGE))
trace = [json.loads(line) for line in (RUN / "training_trace.jsonl").read_text().splitlines()]
assert len(trace) == 2000 and all(row["step"] == row["optimizer_steps_applied"] == i + 1 for i, row in enumerate(trace))
assert all(row["observation_index"] == train_index[i].tolist() and row["subject_index"] == schedule["training_subject_index"][i].tolist() for i, row in enumerate(trace))
assert all(np.isfinite(row["loss"]) and np.isfinite(row["preclip_gradient_norm"]) for row in trace)
assert completion["optimizer_steps_applied"] == 2000
parent = torch.load(config["parent_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
assert parent["phase"] == "experimental_image_key_proposal_only" and parent["step"] == 4000
assert sha(RUN / "catalogue.pt") == parent["experiment"]["prepared_source_sha256"]["catalogue.pt"]
assert config["atlas_binding"] == parent["experiment"]["atlas_binding"]
expected_kwargs = {**parent["experiment"]["model_kwargs"], "pose_only_steps": 1, "coordinate_evidence_conditioning": True, "signed_pose_evidence": True, "signed_pose_cost_rms": SIGNED_POSE_COST_RMS, "ribbon_deformation": True, "joint_uncertainty_rank": None, "frame_centre_offset_conditioning": False}
assert config["model_kwargs"] == json.loads(json.dumps(expected_kwargs))
initial = torch.load(RUN / "joint_model_step_00000.pt", map_location="cpu", weights_only=False, mmap=True)
final = torch.load(RUN / "joint_model_step_02000.pt", map_location="cpu", weights_only=False, mmap=True)
for name in ("joint_model_step_00000.pt", "joint_model_step_02000.pt", "training_trace.jsonl", "observation_identities.json", "experiment.json", "development_metrics_step_00000.json", "development_metrics_step_02000.json"):
    sha(RUN / name)
assert initial["step"] == initial["optimizer_steps_applied"] == 0 and final["step"] == final["optimizer_steps_applied"] == 2000
assert initial["phase"] == final["phase"] == "conditional_native_pose_ribbon"
assert json.loads(json.dumps(initial["experiment"])) == config == json.loads(json.dumps(final["experiment"]))
assert set(initial["model_state"]) == set(final["model_state"])
new_keys = set(initial["model_state"]) - set(parent["model_state"])
assert new_keys == {"pose_model.coordinate_evidence.weight", "pose_model.signed_pose_evidence.weight", "ribbon_field_head.weight", "ribbon_field_head.bias"}
assert all(torch.equal(initial["model_state"][key], value) for key, value in parent["model_state"].items())
assert all(torch.count_nonzero(initial["model_state"][key]) == 0 for key in new_keys)
assert sha(COMPARATOR / "joint_model_step_00000.pt") == comparator_audit["artifact_sha256"][str(COMPARATOR / "joint_model_step_00000.pt")]
comparator_initial = torch.load(COMPARATOR / "joint_model_step_00000.pt", map_location="cpu", weights_only=False, mmap=True)
assert set(initial["model_state"]) == set(comparator_initial["model_state"])
assert all(torch.equal(value, comparator_initial["model_state"][key]) for key, value in initial["model_state"].items())
assert torch.equal(initial["torch_rng"], comparator_initial["torch_rng"])
assert len(initial["cuda_rng"]) == len(comparator_initial["cuda_rng"]) and all(torch.equal(a, b) for a, b in zip(initial["cuda_rng"], comparator_initial["cuda_rng"]))
assert initial["python_rng"] == comparator_initial["python_rng"]
assert all(np.array_equal(a, b) for a, b in zip(initial["numpy_rng"], comparator_initial["numpy_rng"]))
assert initial["optimizer_state"] == comparator_initial["optimizer_state"]
del comparator_initial
frozen_keys = [key for key in initial["model_state"] if not key.startswith(TRAINABLE)]
assert set(frozen_keys) == set(config["frozen_state_tensor_names"])
assert all(torch.equal(initial["model_state"][key], final["model_state"][key]) for key in frozen_keys)
assert all(torch.isfinite(value).all() for value in final["model_state"].values())
assert all(group["lr"] == 2e-4 and group["weight_decay"] == 1e-4 for group in final["optimizer_state"]["param_groups"])
assert all(int(value["step"]) == 2000 for value in final["optimizer_state"]["state"].values())
change = {prefix: float(sum((final["model_state"][key].double() - value.double()).square().sum().item() for key, value in initial["model_state"].items() if key.startswith(prefix)) ** .5) for prefix in TRAINABLE}
assert any(value > 0 for value in change.values())
del parent, initial, final

reports, geometric_states = {}, {}
saved_metric_max_difference_um, saved_p95_max_difference_um, reconstructed_coordinate_max_difference_um = 0., 0., 0.
saved_normal_max_difference_deg, saved_director_max_difference = 0., 0.
for endpoint_label, directory, step in (("0", RUN, 0), ("2000", RUN, 2000), ("unscaled_2000", COMPARATOR, 2000)):
    reported = json.loads((directory / f"development_metrics_step_{step:05d}.json").read_text())
    assert reported["step"] == reported["optimizer_steps_applied"] == step and reported["probabilities_calibrated"] is False
    if step == 2000:
        assert reported == (completion if directory == RUN else comparator_completion)["final_evaluation"]
    rows, seen = [], []
    for reference in reported["raw_prediction_files"]:
        raw_path = directory / reference["path"]
        assert sha(raw_path) == reference["sha256"]
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        out = raw["prediction"]
        assert np.array_equal(out["horizontal_reflection"].numpy(), [False, True])
        assert all(torch.isfinite(value).all() for key, value in out.items() if torch.is_tensor(value))
        logp = out["conditional_kept_component_log_probability"][:, 0].double().numpy()
        assert np.max(np.abs(np.exp(logp).sum(-1) - 1)) < 1e-5
        selected = logp.argmax(-1)
        assert np.array_equal(selected, raw["selected_reflection"].numpy())
        assert torch.count_nonzero(raw["initial_selected_reflection"]) == 0
        for local, oi in enumerate(raw["observation_index"].tolist()):
            identity = identities[oi]
            si, mode = identity["section_array_index"], identity["selected_mode"]
            truth = targets[si]
            assert si == int(raw["section_array_index"][local]) and truth["reflection"] == int(raw["oracle_reflection"][local])
            assert identity["support_information_eligible"] == bool(raw["support_information_eligible"][local])
            assert torch.equal(raw["perturbation"][local], schedule["development_perturbation_by_section"][si])
            initial_state = raw["initial_state"][local].numpy()
            if si in geometric_states:
                assert np.array_equal(geometric_states[si], initial_state)
            else:
                geometric_states[si] = initial_state.copy()
            base, base_frame = frame(initial_state)
            initial_slab = base[None] + truth["offsets"][:, None, None, None] * base_frame[:, 2]
            initial_centre = np.stack((base, base[:, ::-1]))
            initial_slab = np.stack((initial_slab, initial_slab[:, :, ::-1]))
            final_centre = out["final_observed_centre_ccf_ap_dv_ml_um"][local, 0].double().numpy()
            final_slab = out["final_observed_ccf_slab_ap_dv_ml_um"][local, 0].double().numpy()
            plane, rotation = frame(out["final_component_state"][local, 0].numpy())
            r = out["final_canonical_residual_local_um"][local, 0].double().numpy()
            d = out["final_canonical_director_delta_local"][local, 0].double().numpy()
            surface = plane + np.einsum("rij,rjhw->rhwi", rotation, r)
            director = rotation[:, None, None, :, 2] + np.einsum("rij,rjhw->rhwi", rotation, d)
            reconstructed = surface[:, None] + truth["offsets"][None, :, None, None, None] * director[:, None]
            surface[1] = surface[1, :, ::-1]
            reconstructed[1] = reconstructed[1, :, :, ::-1]
            director[1] = director[1, :, ::-1]
            reconstruction_error = max(float(np.abs(surface - final_centre).max()), float(np.abs(reconstructed - final_slab).max()))
            reconstructed_coordinate_max_difference_um = max(reconstructed_coordinate_max_difference_um, reconstruction_error)
            assert reconstruction_error < .02
            metrics = {}
            q, weights = truth["visible"][mode], truth["weights"]
            for label, centres, slabs in (("initial_selected", initial_centre[0], initial_slab[0]), ("initial_oracle", initial_centre[truth["reflection"]], initial_slab[truth["reflection"]]), ("selected", final_centre[selected[local]], final_slab[selected[local]]), ("oracle", final_centre[truth["reflection"]], final_slab[truth["reflection"]])):
                for name, pred, target, weight in (("centre", centres, truth["centre"], q), ("slab", slabs, truth["slab"], q[None] * weights[:, None, None])):
                    fullcanvas_key = f"{label}_{name}_fullcanvas_mean_um"
                    metrics[fullcanvas_key] = float(np.linalg.norm(pred - target, axis=-1).mean())
                    assert abs(metrics[fullcanvas_key] - float(raw["metrics"][fullcanvas_key][local])) < .02
                    for statistic, value in distances(pred, target, weight, truth["normal"]).items():
                        key = f"{label}_{name}_{statistic}"
                        metrics[key] = value
                        saved = float(raw["metrics"][key][local])
                        assert np.isnan(value) == np.isnan(saved)
                        if np.isfinite(value):
                            if statistic == "point_p95_um":
                                # Discrete weighted quantiles can cross a point at an FP32 CDF tie.
                                saved_p95_max_difference_um = max(saved_p95_max_difference_um, abs(value - saved))
                            else:
                                saved_metric_max_difference_um = max(saved_metric_max_difference_um, abs(value - saved))
                                assert abs(value - saved) < .02
            truth_director = np.einsum("s,shwc->hwc", weights * truth["offsets"], truth["slab"] - truth["centre"][None]) / np.sum(weights * truth["offsets"]**2)
            for readout, branch in (("selected", selected[local]), ("oracle", truth["reflection"])):
                key = f"{readout}_director_mean_error"
                metrics[key] = float((np.linalg.norm(director[branch] - truth_director, axis=-1) * q).sum() / q.sum()) if q.sum() > 0 else np.nan
                saved = float(raw["metrics"][key][local]); assert np.isnan(metrics[key]) == np.isnan(saved)
                if np.isfinite(saved):
                    saved_director_max_difference = max(saved_director_max_difference, abs(metrics[key] - saved)); assert abs(metrics[key] - saved) < 1e-4
            for readout, predicted_normal in (("selected", rotation[selected[local], :, 2]), ("oracle", rotation[truth["reflection"], :, 2]), ("initial", base_frame[:, 2])):
                key = f"{readout}_fitted_plane_normal_angle_deg"
                metrics[key] = float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(predicted_normal, truth["normal"])), abs(predicted_normal @ truth["normal"]))))
                difference = abs(metrics[key] - float(raw["metrics"][key][local])); saved_normal_max_difference_deg = max(saved_normal_max_difference_deg, difference); assert difference < .002
            metrics["reflection_correct"] = float(selected[local] == truth["reflection"])
            metrics["initial_reflection_correct"] = float(truth["reflection"] == 0)
            metrics["reflection_nll"] = float(-logp[local, truth["reflection"]])
            # Authenticated trainer measurements, explicitly not a duplicate topology replay.
            for key in ("all_components_iterations_minimum_relative_jacobian", "all_components_iterations_minimum_frame_determinant", "all_components_iterations_maximum_derivative_bound", "all_components_iterations_minimum_rescale"):
                metrics[key] = float(raw["metrics"][key][local])
            assert np.isfinite(list(metrics[key] for key in metrics if key.startswith("all_components"))).all()
            rows.append({"observation_index": oi, "subject_id": identity["subject_id"], "mode": mode, "eligible": identity["support_information_eligible"], **metrics})
            seen.append(oi)
        del raw
    assert seen == dev_rows.tolist()
    scopes = {"all": rows, "eligible": [row for row in rows if row["eligible"]], "censored": [row for row in rows if not row["eligible"]]}
    scopes.update({f"eligible_mode:{mode}": [row for row in rows if row["eligible"] and row["mode"] == mode] for mode in MODES})
    dev_subjects = sorted({row["subject_id"] for row in rows})
    assert len(dev_subjects) == 4
    scopes.update({f"eligible_subject:{subject}": [row for row in rows if row["eligible"] and row["subject_id"] == subject] for subject in dev_subjects})
    summaries = {}
    for name, scope in scopes.items():
        by_subject = {}
        for subject in dev_subjects:
            selection = [row for row in scope if row["subject_id"] == subject]
            by_subject[subject] = {key: float(np.mean([row[key] for row in selection if np.isfinite(row[key])])) if any(np.isfinite(row[key]) for row in selection) else None for key in metrics}
        summaries[name] = {"rows": len(scope), "by_subject": by_subject, "subject_macro": {key: float(np.mean([values[key] for values in by_subject.values() if values[key] is not None])) if any(values[key] is not None for values in by_subject.values()) else None for key in metrics}}
        assert summaries[name]["rows"] == reported["summaries"][name]["rows"]
        for key, value in summaries[name]["subject_macro"].items():
            saved = reported["summaries"][name]["subject_macro"][key]
            if key.endswith("point_p95_um"): continue
            assert (value is None) == (saved is None)
            if value is not None: assert abs(value - saved) < (.002 if key.endswith("angle_deg") else .02)
    gates = {}
    for readout in ("selected", "oracle"):
        for geometry in ("centre", "slab"):
            key, baseline = f"{readout}_{geometry}_mean_um", f"initial_{readout}_{geometry}_mean_um"
            macro = summaries["eligible"]["subject_macro"]
            gates[f"{readout}_{geometry}_overall_20pct"] = macro[key] is not None and macro[baseline] is not None and macro[key] <= .8 * macro[baseline]
            for name, summary in summaries.items():
                if name.startswith(("eligible_mode:", "eligible_subject:")):
                    values = summary["subject_macro"]
                    gates[f"{readout}_{geometry}_nonregression:{name}"] = values[key] is not None and values[baseline] is not None and values[key] <= values[baseline]
    macro = summaries["eligible"]["subject_macro"]
    gates["oracle_normal_overall_20pct"] = macro["oracle_fitted_plane_normal_angle_deg"] <= .8 * macro["initial_fitted_plane_normal_angle_deg"]
    for name, summary in summaries.items():
        if name.startswith(("eligible_mode:", "eligible_subject:")):
            values = summary["subject_macro"]
            gates[f"oracle_normal_nonregression:{name}"] = values["oracle_fitted_plane_normal_angle_deg"] <= values["initial_fitted_plane_normal_angle_deg"]
    for key, decision in gates.items():
        assert decision == reported["gate_conditions"][key], key
    topology = all(row["all_components_iterations_minimum_relative_jacobian"] > 0 and row["all_components_iterations_minimum_frame_determinant"] > 0 and row["all_components_iterations_maximum_derivative_bound"] <= .35001 for row in rows)
    assert topology == reported["gate_conditions"]["canonical_orientation_and_no_folds"]
    assert reported["gate_conditions"]["all_outputs_finite"] and reported["gate_conditions"]["frozen_global_tensors_exact"]
    assert bool(all(gates.values()) and topology) == reported["conditional_local_gate_passed"]
    reports[endpoint_label] = {"summaries": summaries, "independent_geometry_gates": gates, "authenticated_reported_topology_passed": topology, "conditional_local_gate_passed": bool(all(gates.values()) and topology), "rows": [{key: value if not isinstance(value, float) or np.isfinite(value) else None for key, value in row.items()} for row in rows]}

comparison = {}
for name, current in reports["2000"]["summaries"].items():
    old = reports["unscaled_2000"]["summaries"][name]
    comparison[name] = {"subject_macro": {}, "by_subject": {}}
    for key, value in current["subject_macro"].items():
        if key.startswith(("selected_", "oracle_")):
            reference = old["subject_macro"][key]
            comparison[name]["subject_macro"][key] = value - reference if value is not None and reference is not None else None
    for subject, values in current["by_subject"].items():
        comparison[name]["by_subject"][subject] = {key: value - old["by_subject"][subject][key] if value is not None and old["by_subject"][subject][key] is not None else None for key, value in values.items() if key.startswith(("selected_", "oracle_"))}
for key, value in completion["final_evaluation"]["difference_from_unscaled_signed_comparator"].items():
    assert abs(value - comparison["eligible"]["subject_macro"][key]) < (.002 if key.endswith("angle_deg") else .02)

report = {"scope": "one frozen conditional local run; no global-capture, benchmark, biological calibration or independent topology-replay claim", "integrity_passed": True, "optimizer_steps_applied": 2000, "training_presentations": 8000, "development_observations_each_endpoint": 384, "trainable_parameter_change_l2_by_prefix": change, "frozen_retrieval_state_tensors_exact": True, "saved_mean_metric_max_difference_um": saved_metric_max_difference_um, "saved_p95_max_difference_um_diagnostic_only": saved_p95_max_difference_um, "reconstructed_final_coordinate_max_difference_um": reconstructed_coordinate_max_difference_um, "arithmetic": "independent NumPy float64, original FP64 physical targets; FP32 mean/coordinate comparison tolerance .02um; weighted p95 difference diagnostic only because of discrete FP32 CDF boundary ties; no trainer metric imports", "topology_scope": "authenticated reported all-branch/all-iteration corner Jacobians and norm bounds; not independently replayed. Final grids independently recomposed from state/residual/director with spatial-only reflection.", "endpoints": reports, "conditional_local_gate_passed": reports["2000"]["conditional_local_gate_passed"], "artifact_sha256": {**hashes, str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
report.update(saved_normal_max_difference_deg=saved_normal_max_difference_deg, saved_director_max_difference=saved_director_max_difference, normalized_minus_unscaled_final=comparison, comparison_scope="same original4k fresh parameters/optimizer/RNG and scheduled/cohort384 observations; shared-sign fixed TRAIN RMS is the sole training intervention; independently reconstructed selected/oracle centre/slab mean,p95,normal,tangential,fullcanvas errors, director errors and fitted-plane normal errors for both endpoints; no rerender/encoder/gradient replay")
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / "audit_source.py").write_bytes(Path(__file__).read_bytes())
(OUTPUT / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"integrity_passed": True, "conditional_local_gate_passed": report["conditional_local_gate_passed"], "normalized_minus_unscaled_eligible": comparison["eligible"]["subject_macro"], "audit": str(OUTPUT / "audit.json")}), flush=True)
