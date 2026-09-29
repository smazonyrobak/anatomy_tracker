"""Post-exit CPU audit of saved rehearsal scores/geometry; no model/training replay."""
import os
import sys
from pathlib import Path
ROOT = Path("I:/AnatomyTracker")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "4"
sys.dont_write_bytecode = True
import hashlib
import json
import numpy as np
import torch

RUN = ROOT / "runs/joint_v6_canonical_rehearsal_001"
OUT = ROOT / "runs/joint_v6_canonical_rehearsal_001_independent_audit"
COMPLETION_SHA256 = "UNSET"
PARENT_SHA256 = "c6aab521d86eee312f12327965b71c6da1d88f184b7e3427dd7ba2782c88c9f8"
REFERENCE_SHA256 = "280836b65fb6db22930c8ee268eb4a880997c7858fb91a1e31ad6c7c94937eb2"
assert len(COMPLETION_SHA256) == 64, "Pin completion only after confirmed process exit"
torch.set_num_threads(4)
hashes = {}


def sha(path):
    path = Path(path)
    if str(path) not in hashes:
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes[str(path)]


def decode(state):
    s = np.asarray(state, dtype=np.float64)
    u = s[:, 3:6] / np.linalg.norm(s[:, 3:6], axis=-1, keepdims=True)
    v = s[:, 6:9] - (s[:, 6:9] * u).sum(-1, keepdims=True) * u
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    U, V = u * np.exp(s[:, 9:10]), (v + s[:, 11:12] * u) * np.exp(s[:, 10:11])
    return s[:, :3], np.cross(u, v), np.stack((s[:, :3] - .5 * (U + V), U, V), axis=1)


assert sha(RUN / "completed.json") == COMPLETION_SHA256
done = json.loads((RUN / "completed.json").read_text()); cfg = json.loads((RUN / "experiment.json").read_text())
assert done["experiment"] == cfg and done["source_unchanged"] and cfg["source"]["git_commit"].startswith("0e6dc0d")
assert (cfg["parent_step"], cfg["additional_applied_updates"], cfg["final_step"], done["additional_optimizer_steps_applied"], done["optimizer_steps_applied"]) == (8000, 2000, 10000, 2000, 10000)
assert cfg["parent_sha256"] == sha(cfg["parent_checkpoint"]) == PARENT_SHA256 and cfg["reference_A6000_sha256"] == REFERENCE_SHA256
assert not cfg["probabilities_calibrated"] and not cfg["model_kwargs"].get("signed_pose_evidence", False)
assert (cfg["real_batch"], cfg["synthetic_batch"], cfg["generated_per_batch"], cfg["learning_rate"], cfg["weight_decay"], cfg["gradient_clip"]) == (8, 8, 4, .00025, 1e-4, 5.)
assert tuple(cfg["trained_modules"]) == ("pose_model.histology_stem.", "pose_model.atlas_stem.", "pose_model.shared_encoder.", "pose_model.image_key_descriptor.")
assert (cfg["donor_seed"], cfg["real_row_seed"], cfg["negative_seed"], cfg["chart_seed"]) == (2026092921, 2026092922, 2026092923, 2026092924)
assert cfg["synthetic_generated_slice"] == [40000, 48000] and cfg["synthetic_frozen_steps_slice"] == [6000, 8000] and cfg["synthetic_frozen_columns"] == [0, 1, 2, 3]
for key, prefix in (("input_sha256", None), ("source", RUN / "source"), ("schedule_sha256", RUN), ("anchor_cache_sha256", RUN)):
    for name, expected in (cfg[key]["file_sha256"] if key == "source" else cfg[key]).items():
        assert sha(Path(name) if prefix is None else prefix / name) == expected, name
parent = torch.load(cfg["parent_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
assert parent["arm"] == "B" and parent["step"] == parent["optimizer_steps_applied"] == 8000
assert cfg["model_kwargs"] == json.loads(json.dumps(parent["experiment"]["model_kwargs"])) and tuple(parent["experiment"]["trained_modules"]) == tuple(cfg["trained_modules"])
assert sha(RUN / "parent_independent_audit.json") == cfg["parent_audit_sha256"] == "d53664d3999054fe37c292b9ba080420384dbd499d235562dc4fdbb088462392"
parent_audit = json.loads((RUN / "parent_independent_audit.json").read_text())
assert parent_audit["integrity_passed"] and not parent_audit["arm_adaptation_gate_passed"]["B"]
assert parent_audit["artifact_sha256"][cfg["parent_checkpoint"]] == PARENT_SHA256
assert sha(RUN / "parent_completion.json") == cfg["parent_completion_sha256"] == "81257a308b0d6e10c70a53de72cafa932fd546c0ec2a6cd13aa82f1edfac3d2f"
parent_done = json.loads((RUN / "parent_completion.json").read_text())
assert json.loads(json.dumps(parent["experiment"])) == parent_done["experiment"] and parent_done["source_unchanged"]
assert set(cfg["optional_source_changes"]) == {"training/arbitrary_plane_recurrent_model.py", "training/arbitrary_plane_recurrent_model_v6.py", "training/arbitrary_plane_joint_model_v6.py"}
for name, expected in parent["experiment"]["source"]["file_sha256"].items():
    if not name.startswith("training/run_") and name not in cfg["optional_source_changes"]:
        assert cfg["source"]["file_sha256"][name] == expected
references = {}
for name, expected, checkpoint in (("reference_A6000_synthetic_metrics.json", "281fa539824103abed383e5a2c20401377527c53e2bfc2156cfec9d8fb211722", REFERENCE_SHA256), ("reference_A6000_real_metrics.json", "465bc4c932b4164f14023c1983dc9a1531f151be21e72aed000e0193e7e04e0b", REFERENCE_SHA256), ("parent_B8000_real_metrics.json", "6e6c21bd57dd1bf3b26a3cb7a93706de65dd1425ff2e4a29bc800196f4fb8025", PARENT_SHA256)):
    assert sha(RUN / name) == expected
    references[name] = json.loads((RUN / name).read_text()); assert references[name]["checkpoint_sha256"] == checkpoint
prepared, replay = Path(cfg["prepared_source_directory"]), Path(cfg["replay_directory"])
train = torch.load(prepared / "training_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
dev = torch.load(prepared / "internal_development_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
catalogue = torch.load(RUN / "catalogue.pt", map_location="cpu", weights_only=False)
assert sha(RUN / "catalogue.pt") == cfg["prepared_source_sha256"]["catalogue.pt"]
assert json.loads((RUN / "synthetic_training_identities.json").read_text()) == train["records"] and json.loads((RUN / "synthetic_development_identities.json").read_text()) == dev["records"]
real = json.loads((RUN / "real_training_identities.json").read_text()); real_dev = json.loads((RUN / "real_development_identities.json").read_text())
for records, directory, name, split in ((real, ROOT / "data/allen_real_training_inputs_20260929", "real_training", "development_train"), (real_dev, ROOT / "runs/joint_v6_imagekey_retrieval_001_allen_raw", "real_development", "development_validation")):
    assert records == [json.loads(line) for line in (directory / "image_geometry.jsonl").read_text().splitlines()]
    assert sha(RUN / f"{name}_model_input.npy") == sha(directory / "raw_model_input.npy") and all(r["split"] == split for r in records)
assert (len(real), len(real_dev), len(dev["records"])) == (256, 64, 640)
assert (len({r["animal_id"] for r in real}), len({r["animal_id"] for r in real_dev})) == (58, 6)
for key in ("animal_id", "specimen_id", "experiment_id", "section_id"):
    assert not {r[key] for r in real} & {r[key] for r in real_dev}
schedule = dict(np.load(RUN / "sampling_schedule.npz")); generated = dict(np.load(RUN / "generated_schedule.npz"))
with np.load(replay / "generated_schedule.npz") as original:
    assert set(generated) == set(original.files) and all(np.array_equal(value, original[key][40000:48000]) for key, value in generated.items())
assert np.array_equal(schedule["frozen_row_index"], np.load(replay / "frozen_training_row_indices.npy")[6000:8000, :4])
for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    assert not ({r[key] for r in train["records"]} | set(generated[key])) & {r[key] for r in dev["records"]}
donor = np.array([r["animal_id"] for r in real]); unique = np.unique(donor)
rng, row_rng = np.random.default_rng(2026092921), np.random.default_rng(2026092922)
donor_schedule = np.concatenate([rng.permutation(unique) for _ in range(276)])[:16000].reshape(2000, 8)
rows = np.empty_like(donor_schedule)
for d in unique:
    mask = donor_schedule == d; rows[mask] = row_rng.choice(np.flatnonzero(donor == d), size=mask.sum(), replace=True)
assert np.array_equal(schedule["donor_id"], donor_schedule) and np.array_equal(schedule["real_row_index"], rows)
rng = np.random.default_rng(2026092923)
assert np.array_equal(schedule["global_cell_index"], rng.integers(0, 98304, (2000, 32), dtype=np.int64)) and np.array_equal(schedule["local_pool_rank"], rng.integers(0, 8, (2000, 8), dtype=np.int64))
chart = np.random.default_rng(2026092924).permutation(np.repeat(np.array([0, 1], dtype=np.int64), 8000)).reshape(2000, 8)
assert np.array_equal(chart, schedule["canonical_chart_index"]) and np.bincount(chart.ravel()).tolist() == [8000, 8000]
anchors = dict(np.load(RUN / "weak_affine_anchors.npz")); ac, an, auv = decode(anchors["full_frame_state"])
affine = np.array([r["model_pixel_to_ap_dv_ml_um"] for r in real]); expected_ouv = np.stack((affine[:, :, 2], 96 * affine[:, :, 0], 96 * affine[:, :, 1]), axis=1)
assert np.allclose(auv, expected_ouv, rtol=0, atol=1e-8) and np.array_equal(anchors["physical_ouv_ap_dv_ml_um"], expected_ouv)
origin = np.array(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]); cc, cn, cuv = decode(anchors["canonical_full_frame_state"])
assert np.allclose(cc, origin + an * ((ac - origin) * an).sum(-1, keepdims=True), rtol=0, atol=1e-8) and np.allclose(cn, an, rtol=0, atol=1e-12)
assert np.allclose(cuv, anchors["canonical_physical_ouv_ap_dv_ml_um"], rtol=0, atol=1e-8)
assert np.allclose(anchors["canonical_full_frame_state"][:, 3:9], anchors["full_frame_state"][:, 3:9], rtol=0, atol=1e-12)
assert np.allclose(anchors["canonical_full_frame_state"][:, 9:11], np.log(12000.), rtol=0, atol=1e-12) and np.all(anchors["canonical_full_frame_state"][:, 11] == 0)
cache = np.load(RUN / "real_anchor_image_cache.npy", mmap_mode="r"); eligibility = dict(np.load(RUN / "real_anchor_eligibility.npz"))
assert cache.shape == (2, 256, 2, 96, 96) and np.isfinite(cache).all()
support = np.clip(cache[:, :, 1], 0, 1).sum((-2, -1)); eligible = (support >= 64).all(0)
assert np.array_equal(eligible, eligibility["common_eligible"]) and np.allclose(support, np.stack((eligibility["matched_support_mass"], eligibility["canonical_support_mass"])), rtol=0, atol=1e-3)
endpoint_path = RUN / "joint_model_step_10000.pt"
endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False, mmap=True)
assert json.loads(json.dumps(endpoint["experiment"])) == cfg and endpoint["phase"] == parent["phase"] == "experimental_image_key_proposal_only"
assert (endpoint["step"], endpoint["optimizer_steps_applied"], endpoint["additional_optimizer_steps_applied"]) == (10000, 10000, 2000)
assert set(endpoint["optimizer_state"]["state"]) == set(parent["optimizer_state"]["state"]) and all(int(value["step"]) == 10000 for value in endpoint["optimizer_state"]["state"].values())
assert len(endpoint["optimizer_state"]["param_groups"]) == len(parent["optimizer_state"]["param_groups"])
for a, b in zip(endpoint["optimizer_state"]["param_groups"], parent["optimizer_state"]["param_groups"]):
    assert a["lr"] == .00025 and b["lr"] == .001 and {**a, "lr": .001} == b
assert set(endpoint["model_state"]) == set(parent["model_state"]) and all(torch.isfinite(value).all() for value in endpoint["model_state"].values())
frozen = [name for name in endpoint["model_state"] if not name.startswith(tuple(cfg["trained_modules"]))]
assert all(torch.equal(endpoint["model_state"][name], parent["model_state"][name]) for name in frozen) and done["frozen_nonretrieval_tensors_exact"]
changed = sum(not torch.equal(value, parent["model_state"][name]) for name, value in endpoint["model_state"].items() if name.startswith(tuple(cfg["trained_modules"])))
assert changed > 0
initialization = json.loads((RUN / "initialization.json").read_text())
assert initialization["parent_sha256"] == PARENT_SHA256 and initialization["model_tensors_exact"] and initialization["optimizer_state_exact_before_lr_change"] and initialization["restored_parent_rng"]
assert initialization["learning_rate_before"] == .001 and initialization["learning_rate_after"] == .00025 and initialization["betas_and_weight_decay_unchanged"] and not initialization["initial_gallery_rebuilt"]
trace = [json.loads(line) for line in (RUN / "training_trace.jsonl").read_text().splitlines()]; assert len(trace) == 2000
for i, row in enumerate(trace):
    selected, weights = chart[i], eligible[rows[i]].astype(float)
    assert row["step"] == 8001 + i and row["additional_optimizer_steps_applied"] == i + 1
    assert row["real_row_index"] == rows[i].tolist() and row["real_donor_id"] == donor_schedule[i].tolist() and row["real_chart_index"] == selected.tolist()
    assert row["generated_indices"] == list(range(40000 + 4 * i, 40004 + 4 * i)) and row["frozen_row_index"] == schedule["frozen_row_index"][i].tolist()
    finite, visible = np.asarray(row["finite_support_mass_px"]), np.asarray(row["visible_support_mass_px"])
    assert np.isfinite(finite).all() and np.isfinite(visible).all()
    generated_weight = ((finite >= 64) & (visible >= 64)).astype(float)
    assert row["generated_point_pose_weight"] == generated_weight.tolist() and np.isclose(row["synthetic_supervision_mass"], generated_weight.sum() + float(train["weight"][schedule["frozen_row_index"][i]].sum()))
    labels = np.concatenate((generated["cell_index"][4*i:4*i+4], train["label"][schedule["frozen_row_index"][i]].numpy()))
    assert len(row["local_negative_cell_index"]) == 8 and row["candidate_cell_index"] == np.unique(np.concatenate((labels, schedule["global_cell_index"][i], row["local_negative_cell_index"]))).tolist()
    assert row["real_common_eligible"] == weights.tolist() and row["real_supervision_mass"] == weights.sum()
    assert np.allclose(row["real_anchor_finite_support_mass_px"], support[selected, rows[i]], rtol=0, atol=1e-3)
    counts = [int(((selected == c) & (weights > 0)).sum()) for c in (0, 1)]
    assert [row["real_matched_count"], row["real_canonical_count"]] == counts
    chart_means = [row["real_matched_nce"], row["real_canonical_nce"]]
    assert all(value is None if count == 0 else value is not None and np.isfinite(value) for count, value in zip(counts, chart_means))
    assert np.isclose(sum(count * value for count, value in zip(counts, chart_means) if count) / max(sum(counts), 1), row["real_paired_nce"], rtol=1e-6, atol=1e-7)
    assert np.isfinite([row[k] for k in ("loss", "real_paired_nce", "synthetic_nll", "gradient_norm")]).all() and np.isclose(row["loss"], (2 / 3) * row["synthetic_nll"] + row["real_paired_nce"] / 3, rtol=5e-7, atol=1e-7)
checkpoint_sha256 = sha(endpoint_path); del endpoint
centres, normals, cell_ouv = decode(catalogue["arrays"]["cell_states_float64"]); dc, dn, _ = decode(dev["truth_state"])
dev_affine = np.array([r["model_pixel_to_ap_dv_ml_um"] for r in real_dev]); dev_ouv = np.stack((dev_affine[:, :, 2], 96 * dev_affine[:, :, 0], 96 * dev_affine[:, :, 1]), axis=1)
real_dc = dev_ouv[:, 0] + .5 * (dev_ouv[:, 1] + dev_ouv[:, 2]); real_dn = np.cross(dev_ouv[:, 1], dev_ouv[:, 2]); real_dn /= np.linalg.norm(real_dn, axis=-1, keepdims=True)
numeric, errors = {}, {}
for dataset, records, tc, tn, ouv in (("allen_train", real, ac, an, auv), ("synthetic", dev["records"], dc, dn, None), ("allen_raw", real_dev, real_dc, real_dn, dev_ouv)):
    directory = RUN / dataset; report = json.loads((directory / "summary.json").read_text()); saved = dict(np.load(directory / "rows.npz"))
    assert report == done["results"][dataset] and report["step"] == 10000 and report["checkpoint_sha256"] == checkpoint_sha256 and not report["probabilities_calibrated"]
    assert report["gallery_sha256"] == sha(RUN / "gallery_descriptors_step_10000.npy")
    for name, expected in report["output_sha256"].items(): assert sha(directory / name) == expected
    raw = np.load(directory / "raw_cell_log_probability.npy", mmap_mode="r"); component = np.load(directory / "raw_component_log_score.npy", mmap_mode="r")
    assert raw.shape == (len(records), 98304) and component.shape == (len(records), 98304, 2)
    tops, ranks, normal_lps = [], [], []; error = normalization_error = 0.
    for i in range(0, len(records), 16):
        block = np.asarray(raw[i:i+16]); assert np.isfinite(component[i:i+16]).all() and np.isfinite(block).all()
        logits = np.logaddexp.reduce(component[i:i+16].astype(np.float64), axis=-1); lp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
        error = max(error, float(np.abs(lp - block).max())); normalization_error = max(normalization_error, float(np.abs(np.logaddexp.reduce(block.astype(np.float64), axis=-1)).max()))
        tops.append(np.argsort(-block, axis=-1, kind="stable")[:, :128])
        if dataset == "synthetic":
            labels = dev["label"][i:i+len(block)].numpy(); truth = block[np.arange(len(block)), labels]
            ranks.append(1 + (block > truth[:, None]).sum(-1) + ((block == truth[:, None]) & (np.arange(98304)[None] < labels[:, None])).sum(-1))
            normal_lps.append(np.logaddexp.reduce(block.reshape(-1, 384, 256), axis=-1))
    assert error < 2e-5 and normalization_error < 2e-5
    errors[dataset] = {"component_to_cell_max_abs": error, "raw_log_normalization_max_abs": normalization_error}; top = np.concatenate(tops)
    top_component = component[np.arange(len(records))[:, None], top].astype(np.float64)
    assert np.allclose(np.exp(top_component - np.logaddexp.reduce(top_component, axis=-1, keepdims=True)), saved["top128_representation_probability"], rtol=0, atol=2e-5)
    assert np.array_equal(top, saved["top128_cell_index"]) and np.allclose(saved["reference_center_ap_dv_ml_um"], tc, atol=2e-3, rtol=0)
    assert np.allclose(np.abs((saved["reference_normal_ap_dv_ml"] * tn).sum(-1)), 1, rtol=0, atol=1e-6)
    dot = (normals[top] * tn[:, None]).sum(-1); angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(normals[top], tn[:, None]), axis=-1), np.abs(dot)))
    offset = np.abs(((centres[top] - origin) * normals[top]).sum(-1) - np.where(dot < 0, -1, 1) * ((tc - origin) * tn).sum(-1)[:, None])
    metrics = {"plane_angle_deg": angle[:, 0], "normal_offset_error_um": offset[:, 0], **{f"physical_plane_capture_at_{k}": ((angle[:, :k] <= 10) & (offset[:, :k] <= 500)).any(-1).astype(float) for k in (32, 128)}}
    subsets = {"all": np.ones(len(records), bool)}
    if ouv is not None:
        p = cell_ouv[top[:, 0]]; pu, pv = p[:, 1] * 95 / 96, p[:, 2] * 95 / 96; ru, rv = ouv[:, 1] * 95 / 96, ouv[:, 2] * 95 / 96
        metrics["map_finite_frame_rms_um"] = np.sqrt(((p[:, 0] + .5 * (pu + pv) - ouv[:, 0] - .5 * (ru + rv))**2).sum(-1) + .25 * (np.minimum(((pu - ru)**2).sum(-1), ((pu + ru)**2).sum(-1)) + ((pv - rv)**2).sum(-1)))
    if dataset == "synthetic":
        labels = dev["label"].numpy(); weight = dev["weight"].numpy(); modes = np.array([r["selected_mode"] for r in records]); truth = raw[np.arange(len(records)), labels]
        rank, nlp = np.concatenate(ranks), np.concatenate(normal_lps); mn = normals[nlp.argmax(-1) * 256]
        metrics.update(nll=-truth, truth_rank=rank, normal_marginal_nll=-nlp[np.arange(len(records)), labels // 256], normal_marginal_map_angle_deg=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(mn, tn), axis=-1), np.abs((mn * tn).sum(-1)))), **{f"hit_at_{k}": (rank <= k).astype(float) for k in (1, 8, 32, 128)})
        assert np.array_equal(saved["label"], labels) and np.array_equal(saved["pose_supervision_weight"], weight) and np.array_equal(saved["selected_mode"], modes)
        subsets.update(eligible=weight > 0, censored=weight <= 0, **{f"eligible_mode:{mode}": (weight > 0) & (modes == mode) for mode in np.unique(modes)})
    if dataset == "allen_train":
        subsets.update(eligible=eligible, ineligible=~eligible); assert np.array_equal(saved["common_anchor_eligible"], eligible)
        anchor = dict(np.load(directory / "exact_anchor_reconstruction.npz")); scores = anchor["uniform_representation_logit"]
        assert np.isfinite(anchor["raw_component_logit"]).all() and np.allclose(np.logaddexp.reduce(anchor["raw_component_logit"].astype(np.float64) - np.log(2.), axis=-1), scores, rtol=0, atol=2e-5)
        assert np.array_equal(anchor["query_descriptor"], saved["query_descriptor"])
        d = an @ an.T; offsets = ((ac - origin) * an).sum(-1); near = (np.abs(d) >= np.cos(np.deg2rad(10))) & (np.abs(offsets[None] - np.where(d < 0, -1, 1) * offsets[:, None]) <= 500)
        assert np.array_equal(anchor["near_physical_mask"], near)
        for chart_id, name in enumerate(("matched", "canonical")):
            a = scores[chart_id]; own = np.diag(a); rank = 1 + (a > own[:, None]).sum(-1) + ((a == own[:, None]) & (np.arange(256)[None] < np.arange(256)[:, None])).sum(-1)
            nrank = 1 + np.argmax(np.take_along_axis(near, np.argsort(-a, axis=-1, kind="stable"), axis=1), axis=1)
            metrics.update({f"{name}_own_anchor_rank": rank, f"{name}_near_physical_anchor_rank": nrank, **{f"{name}_own_anchor_hit_at_{k}": (rank <= k).astype(float) for k in (1, 8, 32)}, **{f"{name}_near_physical_anchor_hit_at_{k}": (nrank <= k).astype(float) for k in (1, 8, 32)}})
    for key, value in metrics.items(): assert np.allclose(value, saved[key], rtol=1e-6, atol=2e-3), (dataset, key)
    group = np.array([r["animal_id"] for r in records]); summary = {}
    for key in ("animal_id", "specimen_id", "experiment_id", "section_id"): assert np.array_equal(saved[key], np.array([r[key] for r in records]))
    for name, selected in subsets.items():
        groups = np.unique(group[selected]); by_group = {str(d): {key: float(value[selected & (group == d)].mean()) for key, value in metrics.items()} for d in groups}
        macro = {key: float(np.mean([row[key] for row in by_group.values()])) for key in metrics} if len(groups) else None
        expected = report["subsets"][name]; assert (expected["rows"], expected["groups"]) == (int(selected.sum()), len(groups))
        if macro:
            for key, value in macro.items(): assert np.isclose(value, expected["group_macro"][key], rtol=1e-6, atol=2e-3)
            for d, row in by_group.items():
                for key, value in row.items(): assert np.isclose(value, expected["by_group"][d][key], rtol=1e-6, atol=2e-3)
        summary[name] = {"rows": int(selected.sum()), "groups": len(groups), "group_macro": macro}
    if dataset != "synthetic":
        assert all(np.isclose(value, report["donor_macro"][key], rtol=1e-6, atol=2e-3) for key, value in summary["all"]["group_macro"].items())
    numeric[dataset] = summary
final_real = numeric["allen_raw"]["all"]["group_macro"]
b = references["parent_B8000_real_metrics.json"]["donor_macro"]; a = references["reference_A6000_real_metrics.json"]["donor_macro"]
gates = {"real_B8000_normal_retention": final_real["plane_angle_deg"] <= b["plane_angle_deg"] + 2., "real_B8000_top32_retention": final_real["physical_plane_capture_at_32"] >= b["physical_plane_capture_at_32"] - .02, "real_A6000_normal_improvement_at_least_5deg": final_real["plane_angle_deg"] <= a["plane_angle_deg"] - 5., "real_A6000_top32_plane_improvement_at_least_point10": final_real["physical_plane_capture_at_32"] >= a["physical_plane_capture_at_32"] + .10}
for name, result in numeric["synthetic"].items():
    if (name == "eligible" or name.startswith("eligible_mode:")) and result["rows"]:
        reference = references["reference_A6000_synthetic_metrics.json"]["subsets"][name]
        assert (result["rows"], result["groups"]) == (reference["rows"], reference["groups"])
        r, a = result["group_macro"], reference["group_macro"]
        gates[f"synthetic_normal_retention:{name}"] = r["plane_angle_deg"] <= a["plane_angle_deg"] + 2.
        gates[f"synthetic_top32_retention:{name}"] = r["physical_plane_capture_at_32"] >= a["physical_plane_capture_at_32"] - .02
assert gates == done["gates"] and all(gates.values()) == done["recovery_gate_passed"]
for name in ("experiment.json", "initialization.json", "training_trace.jsonl", "synthetic_training_identities.json", "synthetic_development_identities.json", "real_training_identities.json", "real_development_identities.json"):
    sha(RUN / name)
OUT.mkdir(parents=True, exist_ok=False)
result = {"integrity_passed": True, "gates": gates, "recovery_gate_passed": bool(all(gates.values())), "numeric_results": numeric, "component_errors": errors, "frozen_nonretrieval_tensor_count": len(frozen), "changed_retrieval_tensor_count": changed, "exact_additional_update_budget": 2000, "common_eligible_training_rows": int(eligible.sum()), "planned_chart_counts": [8000, 8000], "schedule_reconstruction_passed": True, "scope": "Independent post-exit saved-component normalization/ranks and physical geometry; original audited A6000/B8000 reference summaries retained at exact pinned bytes. Synthetic group macros are organizational, not biological-animal calibration.", "not_replayed": ["atlas rendering or source-image decoding", "encoder/gallery re-embedding or cosine score reconstruction", "optimizer/backpropagation trajectory and initial in-memory loading (archived source/init receipt only)", "full nearest-negative mining or per-step exclusion masks; sampled schedules, candidate union and recorded support/eligibility checked", "reference A6000/B8000 raw predictions are not recomputed again; use their pinned independently audited summaries"], "probabilities_calibrated": False, "artifact_sha256": hashes}
(OUT / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"integrity_passed": True, "recovery_gate_passed": result["recovery_gate_passed"], "audit": str(OUT / "audit.json")}), flush=True)
