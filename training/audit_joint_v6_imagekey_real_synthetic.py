"""Post-exit CPU audit of the fixed A6000 real+synthetic continuation."""
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

RUN = ROOT / "runs/joint_v6_imagekey_real_synthetic_001"
OUTPUT = ROOT / "runs/joint_v6_imagekey_real_synthetic_001_independent_audit"
COMPLETION_SHA256 = "a1cd0323d9456210e64b4e622ab3375524dc2f06810b6bdd43c3e0aa49c1be31"
PARENT_SHA256 = "280836b65fb6db22930c8ee268eb4a880997c7858fb91a1e31ad6c7c94937eb2"
PARENT_AUDIT_SHA256 = "2db902bdc6a2e502c911fa3ced308ed96d27219926356457efa0e063d53bd2e0"
PARENT_SYNTHETIC_SHA256 = "281fa539824103abed383e5a2c20401377527c53e2bfc2156cfec9d8fb211722"
PARENT_REAL_SHA256 = "465bc4c932b4164f14023c1983dc9a1531f151be21e72aed000e0193e7e04e0b"
TRAINABLE = ("pose_model.histology_stem.", "pose_model.atlas_stem.", "pose_model.shared_encoder.", "pose_model.image_key_descriptor.")
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
assert completion["experiment"] == config and completion["source_unchanged"]
assert config["parent_kind"] == "outline_dropout_A" and config["parent_sha256"] == PARENT_SHA256 and config["parent_audit_sha256"] == PARENT_AUDIT_SHA256
assert (config["parent_step"], config["additional_applied_updates"], config["final_step"], completion["additional_optimizer_steps_applied"], completion["optimizer_steps_applied"]) == (6000, 2000, 8000, 2000, 8000)
assert (config["real_batch"], config["synthetic_batch"], config["generated_per_batch"], config["donor_seed"], config["real_row_seed"], config["negative_seed"]) == (8, 8, 4, 2026092916, 2026092917, 2026092918)
assert tuple(config["trained_modules"]) == TRAINABLE and not config["synthetic_outline_dropout"] and not config["probabilities_calibrated"]
assert (config["learning_rate"], config["weight_decay"], config["gradient_clip"]) == (.001, 1e-4, 5.)
for path, expected in config["input_sha256"].items():
    assert sha(path) == expected
for name, expected in config["source"]["file_sha256"].items():
    assert sha(RUN / "source" / name) == expected
for name, expected in config["schedule_sha256"].items():
    assert sha(RUN / name) == expected
assert sha(config["parent_checkpoint"]) == PARENT_SHA256
assert sha(RUN / "parent_independent_audit.json") == PARENT_AUDIT_SHA256
parent_audit = json.loads((RUN / "parent_independent_audit.json").read_text())
assert parent_audit["integrity_passed"] and parent_audit["artifact_sha256"][config["parent_checkpoint"]] == PARENT_SHA256
assert sha(RUN / "parent_synthetic_metrics.json") == PARENT_SYNTHETIC_SHA256 and sha(RUN / "parent_real_metrics.json") == PARENT_REAL_SHA256
parent_synthetic = json.loads((RUN / "parent_synthetic_metrics.json").read_text())
parent_real = json.loads((RUN / "parent_real_metrics.json").read_text())
assert parent_synthetic["checkpoint_sha256"] == parent_real["checkpoint_sha256"] == PARENT_SHA256
prepared, replay = Path(config["prepared_source_directory"]), Path(config["replay_directory"])
train = torch.load(prepared / "training_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
dev = torch.load(prepared / "internal_development_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
catalogue = torch.load(RUN / "catalogue.pt", map_location="cpu", weights_only=False)
assert sha(RUN / "catalogue.pt") == config["prepared_source_sha256"]["catalogue.pt"]
assert json.loads((RUN / "synthetic_training_identities.json").read_text()) == train["records"]
assert json.loads((RUN / "synthetic_development_identities.json").read_text()) == dev["records"]
real_train_root, real_dev_root = ROOT / "data/allen_real_training_inputs_20260929", ROOT / "runs/joint_v6_imagekey_retrieval_001_allen_raw"
real_records = [json.loads(line) for line in (real_train_root / "image_geometry.jsonl").read_text().splitlines()]
real_dev_records = [json.loads(line) for line in (real_dev_root / "image_geometry.jsonl").read_text().splitlines()]
assert json.loads((RUN / "real_training_identities.json").read_text()) == real_records and json.loads((RUN / "real_development_identities.json").read_text()) == real_dev_records
assert sha(RUN / "real_training_model_input.npy") == sha(real_train_root / "raw_model_input.npy") == "b4c0bc908ca4b61a4000bbaa02b140e440c3b4da42069b6b793e435c64f38379"
assert sha(RUN / "real_development_model_input.npy") == sha(real_dev_root / "raw_model_input.npy") == "d2b893b6e10832b27ce300523c3fc54bb278bf69df9ce435919bd4d699b30d2e"
assert len(real_records) == 256 and len(real_dev_records) == 64 and len(dev["records"]) == 640
assert all(row["split"] == "development_train" and row["array_row_index"] == i for i, row in enumerate(real_records)) and all(row["split"] == "development_validation" for row in real_dev_records)
donors = np.array([row["animal_id"] for row in real_records]); donor_ids = np.unique(donors)
assert len(donor_ids) == 58 and len({row["animal_id"] for row in real_dev_records}) == 6
for key in ("animal_id", "specimen_id", "experiment_id", "section_id"):
    assert not {row[key] for row in real_records} & {row[key] for row in real_dev_records}
with np.load(RUN / "generated_schedule.npz", allow_pickle=False) as saved, np.load(replay / "generated_schedule.npz", allow_pickle=False) as original:
    assert set(saved.files) == set(original.files) and all(np.array_equal(saved[key], original[key][32000:40000]) for key in saved.files)
    generated_labels = saved["cell_index"].copy()
    for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
        assert not ({row[key] for row in train["records"]} | set(saved[key])) & {row[key] for row in dev["records"]}
with np.load(RUN / "sampling_schedule.npz", allow_pickle=False) as saved:
    schedule = {key: saved[key] for key in saved.files}
frozen = np.load(replay / "frozen_training_row_indices.npy", allow_pickle=False)[4000:6000, :4]
assert np.array_equal(schedule["frozen_row_index"], frozen)
donor_rng, row_rng = np.random.default_rng(2026092916), np.random.default_rng(2026092917)
expected_donors = np.concatenate([donor_rng.permutation(donor_ids) for _ in range((16000 + 57) // 58)])[:16000].reshape(2000, 8)
expected_rows = np.empty_like(expected_donors)
for donor in donor_ids:
    mask = expected_donors == donor
    expected_rows[mask] = row_rng.choice(np.flatnonzero(donors == donor), size=int(mask.sum()), replace=True)
assert np.array_equal(schedule["donor_id"], expected_donors) and np.array_equal(schedule["real_row_index"], expected_rows) and np.array_equal(donors[expected_rows], expected_donors)
donor_counts = {str(donor): int((expected_donors == donor).sum()) for donor in donor_ids}
assert set(donor_counts.values()) == {275, 276}
rng = np.random.default_rng(2026092918)
assert np.array_equal(schedule["global_cell_index"], rng.integers(0, 98304, (2000, 32), dtype=np.int64)) and np.array_equal(schedule["local_pool_rank"], rng.integers(0, 8, (2000, 8), dtype=np.int64))
affines = np.array([row["model_pixel_to_ap_dv_ml_um"] for row in real_records], dtype=np.float64)
ouv = np.stack((affines[:, :, 2], 96 * affines[:, :, 0], 96 * affines[:, :, 1]), axis=1)
with np.load(RUN / "weak_affine_anchors.npz", allow_pickle=False) as anchors:
    assert np.array_equal(anchors["physical_ouv_ap_dv_ml_um"], ouv) and np.array_equal(anchors["nominal_section_thickness_um"], [row["section_thickness_um"] for row in real_records])
    state = anchors["full_frame_state"]
    u = state[:, 3:6] / np.linalg.norm(state[:, 3:6], axis=-1, keepdims=True)
    v = state[:, 6:9] - (state[:, 6:9] * u).sum(-1, keepdims=True) * u; v /= np.linalg.norm(v, axis=-1, keepdims=True)
    edge_u = u * np.exp(state[:, 9:10]); edge_v = (u * state[:, 11:12] + v) * np.exp(state[:, 10:11])
    reconstructed = np.stack((state[:, :3] - .5 * (edge_u + edge_v), edge_u, edge_v), axis=1)
    anchor_error = float(np.abs(reconstructed - ouv).max()); assert anchor_error < 1e-8
trace = [json.loads(line) for line in (RUN / "training_trace.jsonl").read_text().splitlines()]
assert len(trace) == 2000
for i, row in enumerate(trace):
    assert row["step"] == 6001 + i and row["additional_optimizer_steps_applied"] == i + 1
    assert row["generated_indices"] == list(range(32000 + 4 * i, 32004 + 4 * i)) and row["frozen_row_index"] == frozen[i].tolist()
    assert row["real_row_index"] == expected_rows[i].tolist() and row["real_donor_id"] == expected_donors[i].tolist()
    labels = np.concatenate((generated_labels[4 * i:4 * i + 4], train["label"][frozen[i]].numpy()))
    assert row["candidate_cell_index"] == np.unique(np.concatenate((labels, schedule["global_cell_index"][i], row["local_negative_cell_index"]))).tolist()
    assert len(row["local_negative_cell_index"]) == 8 and np.isfinite([row[key] for key in ("loss", "synthetic_nll", "real_paired_nce", "gradient_norm", "synthetic_supervision_mass")]).all()
    assert np.isclose(row["loss"], .5 * (row["synthetic_nll"] + row["real_paired_nce"]), rtol=5e-7, atol=1e-7)
for name in ("training_trace.jsonl", "initialization.json", "experiment.json"):
    sha(RUN / name)
initial = json.loads((RUN / "initialization.json").read_text())
assert initial["parent_sha256"] == PARENT_SHA256 and initial["model_tensors_exact"] and initial["optimizer_state_exact"] and initial["restored_parent_rng"] and not initial["initial_gallery_rebuilt"]
parent = torch.load(config["parent_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
final = torch.load(RUN / "joint_model_step_08000.pt", map_location="cpu", weights_only=False, mmap=True)
assert parent["step"] == parent["optimizer_steps_applied"] == 6000 and parent["arm"] == "A"
assert final["step"] == final["optimizer_steps_applied"] == 8000 and final["additional_optimizer_steps_applied"] == 2000
assert json.loads(json.dumps(final["experiment"])) == config and config["model_kwargs"] == parent["experiment"]["model_kwargs"]
assert final["phase"] == parent["phase"] == "experimental_image_key_proposal_only" and set(final["model_state"]) == set(parent["model_state"])
assert all(torch.isfinite(value).all() for value in final["model_state"].values())
assert all(torch.equal(value, parent["model_state"][key]) for key, value in final["model_state"].items() if not key.startswith(TRAINABLE))
assert final["optimizer_state"]["param_groups"] == parent["optimizer_state"]["param_groups"] and all(int(value["step"]) == 8000 for value in final["optimizer_state"]["state"].values())
del final, parent
states = np.asarray(catalogue["arrays"]["cell_states_float64"])
cell_normal = np.cross(states[:, 3:6], states[:, 6:9]); cell_normal /= np.linalg.norm(cell_normal, axis=-1, keepdims=True)
origin = np.asarray(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"])
cell_offset = ((states[:, :3] - origin) * cell_normal).sum(-1)
dev_state = dev["truth_state"].double().numpy()
dev_normal = np.cross(dev_state[:, 3:6], dev_state[:, 6:9]); dev_normal /= np.linalg.norm(dev_normal, axis=-1, keepdims=True)
results, maximum_score_error = {}, 0.
for dataset, records, centre, normal in (("synthetic", dev["records"], dev_state[:, :3], dev_normal), ("allen_raw", real_dev_records, np.array([row["truth_center_ap_dv_ml_um"] for row in real_dev_records]), np.array([row["truth_normal_ap_dv_ml"] for row in real_dev_records]))):
    directory = RUN / dataset
    receipt = json.loads((directory / "summary.json").read_text()); assert receipt == completion["results"][dataset]
    assert sha(RUN / "joint_model_step_08000.pt") == receipt["checkpoint_sha256"] and sha(RUN / "gallery_descriptors_step_08000.npy") == receipt["gallery_sha256"]
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
        difference = float(np.abs(score - lp).max()); maximum_score_error = max(maximum_score_error, difference); assert difference < 2e-5
        tops.append(np.argsort(-lp, axis=1, kind="stable")[:, :128])
    top = np.concatenate(tops); dot = (cell_normal[top] * normal[:, None]).sum(-1)
    angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(cell_normal[top], normal[:, None]), axis=-1), np.abs(dot)))
    offsets = np.abs(cell_offset[top] - np.where(dot < 0, -1, 1) * ((centre - origin) * normal).sum(-1)[:, None])
    metrics = {"plane_angle_deg": angles[:, 0], "normal_offset_error_um": offsets[:, 0], **{f"physical_plane_capture_at_{k}": ((angles[:, :k] <= 10.) & (offsets[:, :k] <= 500.)).any(-1).astype(float) for k in (32, 128)}}
    with np.load(directory / "rows.npz", allow_pickle=False) as saved:
        assert np.array_equal(top, saved["top128_cell_index"])
        assert all(np.array_equal(saved[key], np.array([row[key] for row in records])) for key in ("animal_id", "specimen_id", "experiment_id", "section_id"))
        assert all(np.max(np.abs(saved[key] - value)) < 1e-4 for key, value in metrics.items())
    group = np.array([row["animal_id"] for row in records]); subsets = {"all": np.ones(len(records), dtype=bool)}
    if dataset == "synthetic":
        eligible = dev["weight"].numpy() > 0; mode = np.array([row["selected_mode"] for row in records])
        subsets.update({"eligible": eligible, "censored": ~eligible, **{f"eligible_mode:{name}": eligible & (mode == name) for name in np.unique(mode)}})
    summaries = {}
    for name, mask in subsets.items():
        groups = np.unique(group[mask]); by_group = {str(identity): {key: float(values[mask & (group == identity)].mean()) for key, values in metrics.items()} for identity in groups}
        macro = {key: float(np.mean([row[key] for row in by_group.values()])) for key in metrics} if len(groups) else None
        assert receipt["subsets"][name]["rows"] == int(mask.sum()) and receipt["subsets"][name]["groups"] == len(groups)
        assert macro is None or all(abs(value - receipt["subsets"][name]["group_macro"][key]) < 1e-4 for key, value in macro.items())
        summaries[name] = {"rows": int(mask.sum()), "groups": len(groups), "group_macro": macro, "by_group": by_group}
    results[dataset] = summaries; del raw, components
real_macro, baseline = results["allen_raw"]["all"]["group_macro"], parent_real["donor_macro"]
gates = {"real_normal_improvement_at_least_5deg": real_macro["plane_angle_deg"] <= baseline["plane_angle_deg"] - 5., "real_top32_plane_improvement_at_least_point10": real_macro["physical_plane_capture_at_32"] >= baseline["physical_plane_capture_at_32"] + .10}
for name, candidate in results["synthetic"].items():
    if (name == "eligible" or name.startswith("eligible_mode:")) and candidate["rows"]:
        reference = parent_synthetic["subsets"][name]
        assert candidate["rows"] == reference["rows"] and candidate["groups"] == reference["groups"]
        b, a = candidate["group_macro"], reference["group_macro"]
        gates[f"synthetic_normal_retention:{name}"] = b["plane_angle_deg"] <= a["plane_angle_deg"] + 2.
        gates[f"synthetic_top32_retention:{name}"] = b["physical_plane_capture_at_32"] >= a["physical_plane_capture_at_32"] - .02
assert gates == completion["gates"] and bool(all(gates.values())) == completion["weak_affine_adaptation_gate_passed"]
report = {"scope": "independent numeric endpoint/schedule/affine audit; no model inference or calibrated/benchmark claim", "non_replayed": "no descriptor re-embedding or score reconstruction from descriptors; no image/render replay, full nearest-negative pools, real exclusion masks, gradient/optimizer trajectory or independent memory observation of initial copying. Training semantics are archived-source/trace authenticated; candidate union and .5/.5 scalar loss arithmetic checked, not NCE recomputed from training embeddings", "integrity_passed": True, "additional_updates": 2000, "final_cumulative_updates": 8000, "real_presentations": 16000, "generated_presentations": 8000, "prepared_presentations": 8000, "real_presentations_by_donor": donor_counts, "maximum_affine_anchor_ouv_reconstruction_error_um": anchor_error, "maximum_component_to_cell_log_probability_error": maximum_score_error, "gates": gates, "weak_affine_adaptation_gate_passed": bool(all(gates.values())), "results": results, "artifact_sha256": {**hashes, str(Path(__file__)): sha(Path(__file__))}}
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / "audit_source.py").write_bytes(Path(__file__).read_bytes())
(OUTPUT / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"gates": gates, "weak_affine_adaptation_gate_passed": report["weak_affine_adaptation_gate_passed"]}), flush=True)
