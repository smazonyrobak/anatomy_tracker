"""One post-exit CPU audit of paired outline-dropout endpoints; no inference."""
import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
sys.dont_write_bytecode = True

import hashlib
import json
import numpy as np
import torch

RUN = ROOT / "runs/joint_v6_imagekey_outline_dropout_001"
OUTPUT = ROOT / "runs/joint_v6_imagekey_outline_dropout_001_independent_audit"
COMPLETION_SHA256 = "515d583cabd70ef54b7cde9a739165c44c5d407a2341947eea595e1abad8f0b1"
assert len(COMPLETION_SHA256) == 64
torch.set_num_threads(2)
hashes = {}


def sha(path):
    path = Path(path)
    if str(path) not in hashes:
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes[str(path)]


assert sha(RUN / "completed.json") == COMPLETION_SHA256
completion = json.loads((RUN / "completed.json").read_text())
config = json.loads((RUN / "experiment.json").read_text())
assert completion["experiment"] == config and completion["source_unchanged"] and completion["integrity_passed"]
assert config["parent_sha256"] == "d4d706e8d80e53a3638a70e79ce8661ff4af41f7b846143aa1ec68372bfb2ae5"
assert (config["parent_step"], config["additional_applied_updates_per_arm"], config["final_cumulative_step"]) == (4000, 2000, 6000)
assert (config["negative_seed"], config["dropout_seed"]) == (2026092914, 2026092915)
for path, expected in config["input_sha256"].items():
    assert sha(path) == expected
for name, expected in config["source"]["file_sha256"].items():
    assert sha(RUN / "source" / name) == expected
for name, expected in config["shared_schedule_sha256"].items():
    assert sha(RUN / name) == expected
parent = torch.load(config["parent_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
assert tuple(config["trained_modules"]) == tuple(parent["experiment"]["trained_modules"])
assert config["model_kwargs"] == parent["experiment"]["model_kwargs"]
prepared = Path(parent["experiment"]["prepared_source_directory"])
replay = Path(parent["experiment"]["replay_directory"])
train = torch.load(prepared / "training_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
dev = torch.load(prepared / "internal_development_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
catalogue = torch.load(RUN / "catalogue.pt", map_location="cpu", weights_only=False)
assert sha(RUN / "catalogue.pt") == parent["experiment"]["prepared_source_sha256"]["catalogue.pt"]
assert json.loads((RUN / "development_identities.json").read_text()) == dev["records"]
assert json.loads((RUN / "training_identities.json").read_text()) == train["records"]
with np.load(RUN / "generated_schedule.npz", allow_pickle=False) as saved, np.load(replay / "generated_schedule.npz", allow_pickle=False) as original:
    assert set(saved.files) == set(original.files) and all(np.array_equal(saved[key], original[key][32000:48000]) for key in saved.files)
    generated_mode = saved["mode_index"].reshape(2000, 8)
with np.load(RUN / "shared_schedule.npz", allow_pickle=False) as saved:
    shared = {key: saved[key] for key in saved.files}
frozen = np.load(replay / "frozen_training_row_indices.npy", allow_pickle=False)[4000:6000]
assert np.array_equal(shared["frozen_row_indices"], frozen)
rng = np.random.default_rng(2026092914)
assert np.array_equal(shared["global_cell_index"], rng.integers(0, 98304, (2000, 32), dtype=np.int64))
assert np.array_equal(shared["local_pool_rank"], rng.integers(0, 8, (2000, 16), dtype=np.int64))
draw = np.random.default_rng(2026092915).random((2000, 16)) < .5
available = np.concatenate((generated_mode != 0, train["channels"][frozen, 2, 0, 0].numpy().astype(bool)), axis=1)
assert np.array_equal(shared["dropout_draw"], draw) and np.array_equal(shared["original_outline_available"], available)
assert np.array_equal(shared["arm_B_outline_dropped"], draw & available)
traces = [[json.loads(line) for line in (RUN / arm / "training_trace.jsonl").read_text().splitlines()] for arm in ("A", "B")]
paired_keys = ("step", "additional_step", "optimizer_steps_applied", "candidate_ids", "local_negative_ids", "ignored_candidate_counts", "generated_indices", "frozen_row_indices", "original_mode", "original_outline_available", "finite_support_mass_px", "visible_support_mass_px", "generated_point_pose_weight")
assert all(len(trace) == 2000 for trace in traces)
for i, (a, b) in enumerate(zip(*traces)):
    assert all(a[key] == b[key] for key in paired_keys)
    assert a["step"] == a["optimizer_steps_applied"] == 4001 + i and a["additional_step"] == i + 1
    assert a["generated_indices"] == list(range(32000 + 8 * i, 32008 + 8 * i)) and a["frozen_row_indices"] == frozen[i].tolist()
    assert a["original_outline_available"] == available[i].tolist() and not any(a["outline_dropped"]) and b["outline_dropped"] == (draw[i] & available[i]).tolist()
    assert all(np.isfinite(row[key]) for row in (a, b) for key in ("sampled_nll", "gradient_norm"))
first_reference = None
for arm in ("A", "B"):
    sha(RUN / arm / "training_trace.jsonl")
    sha(RUN / arm / "initialization.json"); sha(RUN / arm / "first_batch.pt")
    initial = json.loads((RUN / arm / "initialization.json").read_text())
    assert initial["parent_sha256"] == config["parent_sha256"] and initial["model_tensors_exact"] and initial["optimizer_state_exact"] and initial["restored_parent_rng"] and not initial["initial_gallery_rebuilt"]
    first = torch.load(RUN / arm / "first_batch.pt", map_location="cpu", weights_only=False)
    assert np.array_equal(first["outline_dropped"].numpy(), draw[0] & available[0] if arm == "B" else np.zeros(16, dtype=bool))
    if first_reference is None:
        first_reference = {key: first[key] for key in ("input_channels_before", "labels", "weights")}
    else:
        assert all(torch.equal(first[key], value) for key, value in first_reference.items())
    expected = first["input_channels_before"].clone()
    expected[first["outline_dropped"], 1:] = 0
    assert torch.equal(first["input_channels_after"], expected)
    final = torch.load(RUN / arm / "joint_model_step_06000.pt", map_location="cpu", weights_only=False, mmap=True)
    assert final["step"] == final["optimizer_steps_applied"] == 6000 and final["additional_optimizer_steps_applied"] == 2000
    assert json.loads(json.dumps(final["experiment"])) == config and final["arm"] == arm
    assert set(final["model_state"]) == set(parent["model_state"])
    assert all(torch.isfinite(value).all() for value in final["model_state"].values())
    assert all(torch.equal(value, parent["model_state"][key]) for key, value in final["model_state"].items() if not key.startswith(tuple(config["trained_modules"])))
    assert all(int(value["step"]) == 6000 for value in final["optimizer_state"]["state"].values())
    assert final["optimizer_state"]["param_groups"] == parent["optimizer_state"]["param_groups"]
    del final, first
raw_summary = json.loads((RUN / "parent_allen_raw_summary.json").read_text())
assert sha(RUN / "parent_allen_raw_summary.json") == "2394eae62ef6c64e04b41947e57ca94dd64a2c456e0bc04a58b8dc2997107b37"
assert sha(RUN / "allen_raw_model_input.npy") == raw_summary["output_sha256"]["raw_model_input.npy"]
assert sha(RUN / "allen_raw_image_geometry.jsonl") == raw_summary["output_sha256"]["image_geometry.jsonl"]
raw_records = [json.loads(line) for line in (RUN / "allen_raw_image_geometry.jsonl").read_text().splitlines()]
states = np.asarray(catalogue["arrays"]["cell_states_float64"])
cell_normal = np.cross(states[:, 3:6], states[:, 6:9]); cell_normal /= np.linalg.norm(cell_normal, axis=-1, keepdims=True)
origin = np.asarray(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"])
cell_offset = ((states[:, :3] - origin) * cell_normal).sum(-1)
dev_state = dev["truth_state"].double().numpy()
dev_normal = np.cross(dev_state[:, 3:6], dev_state[:, 6:9]); dev_normal /= np.linalg.norm(dev_normal, axis=-1, keepdims=True)
results, maximum_score_error = {}, 0.
for arm in ("A", "B"):
    results[arm] = {}
    for dataset, records, centre, normal in (("synthetic", dev["records"], dev_state[:, :3], dev_normal), ("allen_raw", raw_records, np.array([row["truth_center_ap_dv_ml_um"] for row in raw_records]), np.array([row["truth_normal_ap_dv_ml"] for row in raw_records]))):
        directory = RUN / arm / dataset
        receipt = json.loads((directory / "summary.json").read_text())
        assert receipt == completion["results"][arm][dataset]
        assert sha(RUN / arm / "joint_model_step_06000.pt") == receipt["checkpoint_sha256"] and sha(RUN / arm / "gallery_descriptors_step_06000.npy") == receipt["gallery_sha256"]
        for name, expected in receipt["output_sha256"].items():
            assert sha(directory / name) == expected
        raw = np.load(directory / "raw_cell_log_probability.npy", mmap_mode="r", allow_pickle=False)
        components = np.load(directory / "raw_component_log_score.npy", mmap_mode="r", allow_pickle=False)
        assert raw.shape == (len(records), 98304) and components.shape == (len(records), 98304, 2)
        tops = []
        for start in range(0, len(records), 16):
            lp, joint = np.asarray(raw[start:start + 16], dtype=np.float64), np.asarray(components[start:start + 16], dtype=np.float64)
            assert np.isfinite(lp).all() and np.isfinite(joint).all() and np.max(np.abs(np.logaddexp.reduce(lp, axis=1))) < 2e-5
            score = np.logaddexp(joint[..., 0], joint[..., 1]); score -= np.logaddexp.reduce(score, axis=1)[:, None]
            difference = float(np.abs(score - lp).max()); maximum_score_error = max(maximum_score_error, difference)
            assert difference < 2e-5
            tops.append(np.argsort(-lp, axis=1, kind="stable")[:, :128])
        top = np.concatenate(tops)
        dot = (cell_normal[top] * normal[:, None]).sum(-1)
        angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(cell_normal[top], normal[:, None]), axis=-1), np.abs(dot)))
        offsets = np.abs(cell_offset[top] - np.where(dot < 0, -1, 1) * ((centre - origin) * normal).sum(-1)[:, None])
        metrics = {"plane_angle_deg": angles[:, 0], "normal_offset_error_um": offsets[:, 0], **{f"physical_plane_capture_at_{k}": ((angles[:, :k] <= 10.) & (offsets[:, :k] <= 500.)).any(-1).astype(float) for k in (32, 128)}}
        with np.load(directory / "rows.npz", allow_pickle=False) as saved:
            assert np.array_equal(top, saved["top128_cell_index"])
            assert all(np.array_equal(saved[key], np.array([row[key] for row in records])) for key in ("animal_id", "specimen_id", "experiment_id", "section_id"))
            assert all(np.max(np.abs(saved[key] - value)) < 1e-4 for key, value in metrics.items())
        group = np.array([row["animal_id"] for row in records])
        subsets = {"all": np.ones(len(records), dtype=bool)}
        if dataset == "synthetic":
            eligible = dev["weight"].numpy() > 0
            mode = np.array([row["selected_mode"] for row in records])
            subsets.update({"eligible": eligible, "censored": ~eligible, **{f"eligible_mode:{name}": eligible & (mode == name) for name in np.unique(mode)}})
        summaries = {}
        for name, mask in subsets.items():
            groups = np.unique(group[mask])
            by_group = {str(identity): {key: float(values[mask & (group == identity)].mean()) for key, values in metrics.items()} for identity in groups}
            macro = {key: float(np.mean([row[key] for row in by_group.values()])) for key in metrics} if len(groups) else None
            assert receipt["subsets"][name]["rows"] == int(mask.sum()) and receipt["subsets"][name]["groups"] == len(groups)
            assert macro is None or all(abs(value - receipt["subsets"][name]["group_macro"][key]) < 1e-4 for key, value in macro.items())
            summaries[name] = {"rows": int(mask.sum()), "groups": len(groups), "group_macro": macro, "by_group": by_group}
        results[arm][dataset] = summaries
        del raw, components
parent_summary = json.loads((RUN / "parent_synthetic_step4000.json").read_text())
assert sha(RUN / "parent_synthetic_step4000.json") == "abfb7b6bfca60500e51293bf592eaef84d92938dfddb9bc6f29e24673859ec58"
a_real, b_real = [results[arm]["allen_raw"]["all"]["group_macro"] for arm in ("A", "B")]
gates = {"B_real_normal_improvement_at_least_5deg": b_real["plane_angle_deg"] <= a_real["plane_angle_deg"] - 5., "B_real_top32_plane_improvement_at_least_point10": b_real["physical_plane_capture_at_32"] >= a_real["physical_plane_capture_at_32"] + .10}
for reference_name, reference in (("A6000", results["A"]["synthetic"]), ("parent4000", parent_summary["subsets"])):
    for name, candidate in results["B"]["synthetic"].items():
        if (name == "eligible" or name.startswith("eligible_mode:")) and candidate["rows"]:
            assert candidate["rows"] == reference[name]["rows"] and candidate["groups"] == reference[name]["groups"]
            b, a = candidate["group_macro"], reference[name]["group_macro"]
            gates[f"synthetic_normal_retention:{reference_name}:{name}"] = b["plane_angle_deg"] <= a["plane_angle_deg"] + 2.
            gates[f"synthetic_top32_retention:{reference_name}:{name}"] = b["physical_plane_capture_at_32"] >= a["physical_plane_capture_at_32"] - .02
assert gates == completion["gates"] and bool(all(gates.values())) == completion["outline_dropout_gate_passed"]
report = {"scope": "independent numeric endpoint/schedule audit; full component-to-cell normalization verified, no descriptor re-embedding or score reconstruction from descriptors. No replay of every nearest-negative pool: candidates/ignored counts are paired-trace and source authenticated. Initial optimizer/tensor copying is source/receipt authenticated, not independently observed in memory. No inference, topology, calibration or public benchmark claim", "integrity_passed": True, "additional_updates_per_arm": 2000, "maximum_component_to_cell_log_probability_error": maximum_score_error, "paired_schedule_and_actual_dropout_masks_match": True, "gates": gates, "outline_dropout_gate_passed": bool(all(gates.values())), "results": results, "artifact_sha256": {**hashes, str(Path(__file__)): sha(Path(__file__))}}
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / "audit_source.py").write_bytes(Path(__file__).read_bytes())
(OUTPUT / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"gates": gates, "outline_dropout_gate_passed": report["outline_dropout_gate_passed"]}), flush=True)
