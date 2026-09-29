"""Post-exit CPU saved-score/geometry audit; no rendering, model import or training replay."""
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

RUN = ROOT / "runs/joint_v6_full_coverage_ranking_001"
OUT = ROOT / "runs/joint_v6_full_coverage_ranking_001_independent_audit"
COMPLETION_SHA256 = "576b85c7bcdbeea9e361b15908e320fe0d838291d083c945d19ede2b1687625e"
PARENT_SHA256 = "888ad9cc493453ce769628fda1aebe70813c031f7809192076c08b000720d9af"
B8000_SHA256 = "c6aab521d86eee312f12327965b71c6da1d88f184b7e3427dd7ba2782c88c9f8"
REFERENCE_SHA256 = "280836b65fb6db22930c8ee268eb4a880997c7858fb91a1e31ad6c7c94937eb2"
assert len(COMPLETION_SHA256) == 64, "Pin only after the whole A/conditional-B process exits"
STEPS, START, FINAL, N_REAL, TOL = 12576, 10000, 22576, 1280, 2e-5
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
assert done["experiment"] == cfg and done["source_unchanged"] and cfg["source"]["git_commit"].startswith("2935584")
assert (cfg["parent_step"], cfg["additional_applied_updates"], cfg["final_step"]) == (START, STEPS, FINAL)
assert cfg["parent_sha256"] == sha(cfg["parent_checkpoint"]) == PARENT_SHA256 and cfg["reference_A6000_sha256"] == REFERENCE_SHA256
assert not cfg["probabilities_calibrated"] and not cfg["model_kwargs"].get("signed_pose_evidence", False)
assert (cfg["real_batch"], cfg["synthetic_batch"], cfg["generated_per_batch"], cfg["learning_rate"], cfg["weight_decay"], cfg["gradient_clip"]) == (8, 8, 4, .00025, 1e-4, 5.)
assert tuple(cfg["trained_modules"]) == ("pose_model.histology_stem.", "pose_model.atlas_stem.", "pose_model.shared_encoder.", "pose_model.image_key_descriptor.")
assert [cfg[k] for k in ("appearance_seed", "frozen_row_seed", "donor_and_real_row_seed", "chart_seed", "negative_seed", "conditional_negative_seed")] == list(range(2026092931, 2026092937))
assert cfg["synthetic_cell_permutation_slice"] == [48000, 98304] and cfg["fresh_appearance_draws"]
assert (cfg["mining"]["refresh_interval"], cfg["mining"]["training_refreshes_B"], cfg["mining"]["A_training_refreshes"]) == (1024, 13, 0)
assert list(done["arm_results"]) in (["A"], ["A", "B"]) and done["negative_selection_comparison_available"] == ("B" in done["arm_results"])
assert done["policy"] == cfg["sequential_policy"]
for key, prefix in (("input_sha256", None), ("source", RUN / "source"), ("schedule_sha256", RUN), ("anchor_cache_sha256", RUN)):
    for name, expected in (cfg[key]["file_sha256"] if key == "source" else cfg[key]).items():
        assert sha(Path(name) if prefix is None else prefix / name) == expected, name
parent = torch.load(cfg["parent_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
assert parent["step"] == parent["optimizer_steps_applied"] == START
assert cfg["model_kwargs"] == json.loads(json.dumps(parent["experiment"]["model_kwargs"])) and tuple(parent["experiment"]["trained_modules"]) == tuple(cfg["trained_modules"])
assert sha(RUN / "parent_independent_audit.json") == cfg["parent_audit_sha256"] == "8af7c7e208326de15555ec3fc90290d5a8ab2579fd77a80f0dfb8ffebc14934b"
parent_audit = json.loads((RUN / "parent_independent_audit.json").read_text())
assert parent_audit["integrity_passed"] and not parent_audit["recovery_gate_passed"] and parent_audit["artifact_sha256"][cfg["parent_checkpoint"]] == PARENT_SHA256
assert sha(RUN / "parent_completion.json") == cfg["parent_completion_sha256"] == "be86f269a97fb99275710fd80a95e41d9a742925c8b4d16ae6371d67a8f1cdeb"
assert json.loads((RUN / "parent_completion.json").read_text())["source_unchanged"]
for name, expected in parent["experiment"]["source"]["file_sha256"].items():
    if not name.startswith("training/run_") and name not in cfg["optional_source_changes"]:
        assert cfg["source"]["file_sha256"][name] == expected
references = {}
for name, expected, checkpoint in (("reference_A6000_synthetic_metrics.json", "281fa539824103abed383e5a2c20401377527c53e2bfc2156cfec9d8fb211722", REFERENCE_SHA256), ("reference_A6000_real_metrics.json", "465bc4c932b4164f14023c1983dc9a1531f151be21e72aed000e0193e7e04e0b", REFERENCE_SHA256), ("parent_B8000_real_metrics.json", "6e6c21bd57dd1bf3b26a3cb7a93706de65dd1425ff2e4a29bc800196f4fb8025", B8000_SHA256)):
    assert sha(RUN / name) == expected
    references[name] = json.loads((RUN / name).read_text()); assert references[name]["checkpoint_sha256"] == checkpoint
prepared, replay = Path(cfg["prepared_source_directory"]), Path(cfg["replay_directory"])
train = torch.load(prepared / "training_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
dev = torch.load(prepared / "internal_development_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
catalogue = torch.load(RUN / "catalogue.pt", map_location="cpu", weights_only=False)
assert sha(RUN / "catalogue.pt") == cfg["prepared_source_sha256"]["catalogue.pt"]
assert json.loads((RUN / "synthetic_training_identities.json").read_text()) == train["records"] and json.loads((RUN / "synthetic_development_identities.json").read_text()) == dev["records"]
real = json.loads((RUN / "real_training_identities.json").read_text()); real_dev = json.loads((RUN / "real_development_identities.json").read_text())
expansion = ROOT / "data/allen_real_training_expansion_20260929"; original = ROOT / "data/allen_real_training_inputs_20260929"
assert sha(expansion / "summary.json") == "dfe59bde72f55ade32c8b2b6d144fb4ce0064d9fa27bd74c98dd16362def3334"
assert real == [json.loads(line) for line in (expansion / "union_training_index.jsonl").read_text().splitlines()]
assert real_dev == [json.loads(line) for line in (ROOT / "runs/joint_v6_imagekey_retrieval_001_allen_raw/image_geometry.jsonl").read_text().splitlines()]
assert (len(real), len(real_dev), len(train["records"]), len(dev["records"])) == (1280, 64, 5120, 640)
assert (len({r["animal_id"] for r in real}), len({r["animal_id"] for r in real_dev}), len({r["section_id"] for r in real})) == (58, 6, N_REAL)
for i, row in enumerate(real):
    assert row["split"] == "development_train" and row["union_row_index"] == i
    assert Path(row["source_array_path"]) == (original if i < 256 else expansion) / "raw_model_input.npy" and row["source_array_row_index"] == (i if i < 256 else i - 256)
assert all(r["split"] == "development_validation" for r in real_dev)
for key in ("animal_id", "specimen_id", "experiment_id", "section_id"):
    assert not {r[key] for r in real} & {r[key] for r in real_dev}
schedule = dict(np.load(RUN / "sampling_schedule.npz")); generated = dict(np.load(RUN / "generated_schedule.npz"))
with np.load(replay / "generated_schedule.npz") as saved:
    order = saved["cell_index"][:98304]
    assert np.array_equal(np.sort(order), np.arange(98304)) and np.array_equal(generated["cell_index"], order[48000:])
    assert not np.intersect1d(generated["cell_index"], order[:48000]).size and not np.intersect1d(generated["sample_seed"], saved["sample_seed"]).size
assert len(generated["cell_index"]) == STEPS * 4 == 50304
rng = np.random.default_rng(2026092931); count = STEPS * 4
expected_generated = {"sample_seed": 2026092931 * 1_000_000 + np.arange(count, dtype=np.int64),
    "mode_index": rng.permutation(np.repeat(np.arange(3), count // 3)), "horizontal_flip": rng.integers(0, 2, count).astype(bool),
    "thickness_um": rng.uniform(25, 100, count).astype(np.float32), "gain": rng.uniform(.6, 1.4, count).astype(np.float32),
    "gamma": np.exp(rng.uniform(np.log(.6), np.log(1.6), count)).astype(np.float32), "invert_tissue": rng.integers(0, 2, count).astype(bool),
    "noise_std": rng.uniform(.005, .05, count).astype(np.float32), "background_mean": rng.uniform(0, .8, count).astype(np.float32),
    "background_slope_yx": rng.uniform(-.15, .15, (count, 2)).astype(np.float32), "mask_dilate": rng.integers(0, 2, count).astype(bool), "mask_radius_px": rng.integers(1, 4, count)}
assert all(np.array_equal(generated[k], value) for k, value in expected_generated.items())
assert np.bincount(generated["mode_index"]).tolist() == [16768] * 3
assert bool(generated["biological_animal_id_is_null"]) and not bool(generated["organizational_identifiers_are_biological"])
assert cfg["synthetic_anatomy"]["biological_animal_id"] is None and cfg["synthetic_anatomy"]["independent_atlas_anatomy_sources"] == 1
assert str(generated["atlas_anatomy_source_id"]) == cfg["synthetic_anatomy"]["atlas_anatomy_source_id"]
for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    assert not ({r[key] for r in train["records"]} | set(generated[key])) & {r[key] for r in dev["records"]}
rng = np.random.default_rng(2026092932)
frozen_rows = np.concatenate([rng.permutation(5120) for _ in range((count + 5119) // 5120)])[:count].reshape(STEPS, 4)
assert np.array_equal(schedule["frozen_row_index"], frozen_rows)
donor = np.array([r["animal_id"] for r in real]); unique = np.unique(donor); rng = np.random.default_rng(2026092933)
donor_schedule = np.concatenate([rng.permutation(unique) for _ in range((STEPS * 8 + 57) // 58)])[:STEPS * 8].reshape(STEPS, 8)
rows = np.empty_like(donor_schedule)
for d in unique:
    mask = donor_schedule == d; rows[mask] = rng.choice(np.flatnonzero(donor == d), size=mask.sum(), replace=True)
assert np.array_equal(schedule["donor_id"], donor_schedule) and np.array_equal(schedule["real_row_index"], rows)
chart = np.random.default_rng(2026092934).permutation(np.repeat(np.array([0, 1], dtype=np.int64), STEPS * 4)).reshape(STEPS, 8)
rng = np.random.default_rng(2026092935)
assert np.array_equal(schedule["global_cell_index"], rng.integers(0, 98304, (STEPS, 16), dtype=np.int64)) and np.array_equal(schedule["local_pool_rank"], rng.integers(0, 8, (STEPS, 8), dtype=np.int64))
assert np.array_equal(schedule["conditional_uniform_fraction"], np.random.default_rng(2026092936).random((STEPS, 16))) and np.array_equal(chart, schedule["canonical_chart_index"])
anchors = dict(np.load(RUN / "weak_affine_anchors.npz")); ac, an, auv = decode(anchors["full_frame_state"])
affine = np.array([r["model_pixel_to_ap_dv_ml_um"] for r in real]); expected_ouv = np.stack((affine[:, :, 2], 96 * affine[:, :, 0], 96 * affine[:, :, 1]), axis=1)
assert np.allclose(auv, expected_ouv, rtol=0, atol=1e-8) and np.array_equal(anchors["physical_ouv_ap_dv_ml_um"], expected_ouv)
origin = np.array(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]); cc, cn, cuv = decode(anchors["canonical_full_frame_state"])
assert np.allclose(cc, origin + an * ((ac - origin) * an).sum(-1, keepdims=True), rtol=0, atol=1e-8) and np.allclose(cn, an, rtol=0, atol=1e-12)
assert np.allclose(cuv, anchors["canonical_physical_ouv_ap_dv_ml_um"], rtol=0, atol=1e-8)
assert np.allclose(anchors["canonical_full_frame_state"][:, 3:9], anchors["full_frame_state"][:, 3:9], rtol=0, atol=1e-12)
assert np.allclose(anchors["canonical_full_frame_state"][:, 9:11], np.log(12000.), rtol=0, atol=1e-12) and np.all(anchors["canonical_full_frame_state"][:, 11] == 0)
cache = np.load(RUN / "real_anchor_image_cache.npy", mmap_mode="r"); eligibility = dict(np.load(RUN / "real_anchor_eligibility.npz"))
assert cache.shape == (2, N_REAL, 2, 96, 96) and np.isfinite(cache).all()
support = np.clip(cache[:, :, 1], 0, 1).sum((-2, -1)); eligible = (support >= 64).all(0)
assert np.array_equal(eligible, eligibility["common_eligible"]) and np.allclose(support, np.stack((eligibility["matched_support_mass"], eligibility["canonical_support_mass"])), rtol=0, atol=1e-3)
prior = json.loads((RUN / "train_orientation_prior.json").read_text())
scatter = np.mean([np.einsum("ni,nj->nij", an[donor == d], an[donor == d]).mean(0) for d in unique], axis=0)
eigenvalues, eigenvectors = np.linalg.eigh(scatter); prior_normal = eigenvectors[:, -1] * (-1 if eigenvectors[0, -1] < 0 else 1)
assert np.allclose(scatter, prior["scatter"], rtol=0, atol=1e-12) and np.allclose(eigenvalues, prior["eigenvalues"], rtol=0, atol=1e-12)
assert np.allclose(prior_normal, prior["normal_ap_dv_ml"], rtol=0, atol=1e-12) and prior["section_ids"] == [r["section_id"] for r in real] and prior["donor_ids"] == donor.tolist()
for path, expected in prior["geometry_sha256"].items(): assert sha(path) == expected
centres, normals, cell_ouv = decode(catalogue["arrays"]["cell_states_float64"]); tc_train, tn_train, _ = decode(train["truth_state"]); dc, dn, _ = decode(dev["truth_state"])
dev_affine = np.array([r["model_pixel_to_ap_dv_ml_um"] for r in real_dev]); dev_ouv = np.stack((dev_affine[:, :, 2], 96 * dev_affine[:, :, 0], 96 * dev_affine[:, :, 1]), axis=1)
real_dc = dev_ouv[:, 0] + .5 * (dev_ouv[:, 1] + dev_ouv[:, 2]); real_dn = np.cross(dev_ouv[:, 1], dev_ouv[:, 2]); real_dn /= np.linalg.norm(real_dn, axis=-1, keepdims=True)
cell_offsets = ((centres - origin) * normals).sum(-1); real_offsets = ((ac - origin) * an).sum(-1); train_offsets = ((tc_train - origin) * tn_train).sum(-1)
cell_mass = np.asarray(catalogue["tensors"]["cell_log_mass"][0], np.float32); rep_prior = np.asarray(catalogue["tensors"]["representation_log_weight"][0], np.float32)
arm_audits = {}
for arm in done["arm_results"]:
    directory = RUN / arm; arm_done = json.loads((directory / "completed.json").read_text())
    assert sha(directory / "completed.json") == done["arm_results"][arm]["completion_sha256"] and arm_done["experiment"] == cfg and arm_done["source_unchanged"]
    assert (arm_done["arm"], arm_done["additional_optimizer_steps_applied"], arm_done["optimizer_steps_applied"]) == (arm, STEPS, FINAL)
    initial = torch.load(directory / f"joint_model_step_{START:05d}.pt", map_location="cpu", weights_only=False, mmap=True)
    assert initial["arm"] == arm and initial["step"] == initial["optimizer_steps_applied"] == START and initial["additional_optimizer_steps_applied"] == 0
    assert json.loads(json.dumps(initial["experiment"])) == cfg and set(initial["model_state"]) == set(parent["model_state"])
    assert all(torch.equal(value, parent["model_state"][name]) for name, value in initial["model_state"].items())
    assert initial["optimizer_state"]["param_groups"] == parent["optimizer_state"]["param_groups"] and set(initial["optimizer_state"]["state"]) == set(parent["optimizer_state"]["state"])
    for key, state in parent["optimizer_state"]["state"].items():
        assert all(torch.equal(initial["optimizer_state"]["state"][key][name], value) if torch.is_tensor(value) else initial["optimizer_state"]["state"][key][name] == value for name, value in state.items())
    assert torch.equal(initial["torch_rng"], parent["torch_rng"]) and all(torch.equal(a, b) for a, b in zip(initial["cuda_rng"], parent["cuda_rng"]))
    assert initial["python_rng"] == parent["python_rng"] and initial["numpy_rng"][0] == parent["numpy_rng"][0] and np.array_equal(initial["numpy_rng"][1], parent["numpy_rng"][1]) and initial["numpy_rng"][2:] == parent["numpy_rng"][2:]
    del initial
    endpoint_path = directory / f"joint_model_step_{FINAL:05d}.pt"; endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False, mmap=True)
    assert json.loads(json.dumps(endpoint["experiment"])) == cfg and endpoint["arm"] == arm and endpoint["phase"] == parent["phase"] == "experimental_image_key_proposal_only"
    assert (endpoint["step"], endpoint["optimizer_steps_applied"], endpoint["additional_optimizer_steps_applied"]) == (FINAL, FINAL, STEPS)
    assert endpoint["optimizer_state"]["param_groups"] == parent["optimizer_state"]["param_groups"] and set(endpoint["optimizer_state"]["state"]) == set(parent["optimizer_state"]["state"])
    assert all(int(value["step"]) == FINAL for value in endpoint["optimizer_state"]["state"].values())
    assert set(endpoint["model_state"]) == set(parent["model_state"]) and all(torch.isfinite(value).all() for value in endpoint["model_state"].values())
    frozen = [name for name in endpoint["model_state"] if not name.startswith(tuple(cfg["trained_modules"]))]
    assert all(torch.equal(endpoint["model_state"][name], parent["model_state"][name]) for name in frozen) and arm_done["frozen_nonretrieval_tensors_exact"]
    changed = sum(not torch.equal(value, parent["model_state"][name]) for name, value in endpoint["model_state"].items() if name.startswith(tuple(cfg["trained_modules"])))
    assert changed > 0; checkpoint_sha256 = sha(endpoint_path); del endpoint
    assert checkpoint_sha256 == done["arm_results"][arm]["checkpoint_sha256"]
    trace = [json.loads(line) for line in (directory / "training_trace.jsonl").read_text().splitlines()]; assert len(trace) == STEPS
    fresh_eligible = 0; positive_cells = set()
    for i, row in enumerate(trace):
        selected, weights = chart[i], eligible[rows[i]].astype(float); position = np.arange(4 * i, 4 * i + 4); frozen_index = frozen_rows[i]
        assert row["arm"] == arm and row["step"] == START + i + 1 and row["additional_optimizer_steps_applied"] == i + 1
        assert row["real_row_index"] == rows[i].tolist() and row["real_donor_id"] == donor_schedule[i].tolist() and row["real_chart_index"] == selected.tolist()
        assert row["generated_indices"] == position.tolist() and row["original_permutation_indices"] == (position + 48000).tolist() and row["frozen_row_index"] == frozen_index.tolist()
        finite, visible = np.asarray(row["finite_support_mass_px"]), np.asarray(row["visible_support_mass_px"])
        assert np.isfinite(finite).all() and np.isfinite(visible).all()
        gw = ((finite >= 64) & (visible >= 64)).astype(float); sw = np.concatenate((gw, train["weight"][frozen_index].numpy())); labels = np.concatenate((generated["cell_index"][position], train["label"][frozen_index].numpy()))
        assert row["generated_point_pose_weight"] == gw.tolist() and np.isclose(row["synthetic_supervision_mass"], sw.sum())
        conditional = np.array(row["conditional_cell_index"]); owner_weight = np.concatenate((weights, sw))
        assert conditional.shape == (16,) and ((conditional >= 0) & (conditional < 98304)).all() and row["conditional_owner_weight"] == owner_weight.tolist()
        assert np.array_equal(conditional[owner_weight == 0], schedule["global_cell_index"][i][owner_weight == 0])
        owner_n = np.concatenate((an[rows[i]], normals[labels[:4]], tn_train[frozen_index])); owner_d = np.concatenate((real_offsets[rows[i]], cell_offsets[labels[:4]], train_offsets[frozen_index]))
        dot = (owner_n * normals[conditional]).sum(-1); offsets = np.abs(cell_offsets[conditional] - np.where(dot < 0, -1, 1) * owner_d)
        assert ((np.abs(dot) < np.cos(np.deg2rad(9.9998))) | (offsets > 499.99) | (owner_weight == 0)).all()
        pooled = np.concatenate((labels, schedule["global_cell_index"][i], row["local_negative_cell_index"], conditional)); candidates = np.unique(pooled)
        assert len(row["local_negative_cell_index"]) == 8 and row["candidate_cell_index"] == candidates.tolist() and row["candidate_cell_count"] == len(candidates) and row["pooled_duplicate_count"] == len(pooled) - len(candidates)
        assert row["real_common_eligible"] == weights.tolist() and row["real_supervision_mass"] == weights.sum() and np.allclose(row["real_anchor_finite_support_mass_px"], support[selected, rows[i]], rtol=0, atol=1e-3)
        counts = [int(((selected == c) & (weights > 0)).sum()) for c in (0, 1)]; means = [row["real_matched_nce"], row["real_canonical_nce"]]
        assert [row["real_matched_count"], row["real_canonical_count"]] == counts
        assert all(value is None if count == 0 else value is not None and np.isfinite(value) for count, value in zip(counts, means))
        assert np.isclose(sum(count * value for count, value in zip(counts, means) if count) / max(sum(counts), 1), row["real_paired_nce"], rtol=1e-6, atol=1e-7)
        assert np.isfinite([row[k] for k in ("loss", "real_paired_nce", "synthetic_nll", "gradient_norm")]).all() and np.isclose(row["loss"], (2 / 3) * row["synthetic_nll"] + row["real_paired_nce"] / 3, rtol=5e-7, atol=1e-7)
        fresh_eligible += int(gw.sum()); positive_cells.update(labels[sw > 0].tolist())
        assert (row["fresh_unique_generated_cells_attempted"], row["fresh_unique_generated_cells_eligible"], row["stage_unique_eligible_positive_cells"]) == (4 * (i + 1), fresh_eligible, len(positive_cells))
        if arm == "A":
            assert row["snapshot_step"] is row["snapshot_age_updates"] is row["snapshot_checkpoint_sha256"] is None
        else:
            snapshot_step = START + (i // 1024) * 1024
            assert row["snapshot_step"] == snapshot_step and row["snapshot_age_updates"] == i % 1024 and row["snapshot_checkpoint_sha256"] == sha(directory / f"joint_model_step_{snapshot_step:05d}.pt")
    assert {p.name for p in directory.glob("joint_model_step_*.pt")} == {f"joint_model_step_{START+i:05d}.pt" for i in range(0, STEPS, 1024)} | {f"joint_model_step_{FINAL:05d}.pt"}
    bank_path = directory / f"gallery_descriptors_step_{FINAL:05d}.npy"; bank_hash = sha(bank_path)
    assert bank_hash == done["arm_results"][arm]["gallery_sha256"]
    bank = np.load(bank_path).astype(np.float32); assert bank.shape == (98304, 2, 256) and np.isfinite(bank).all()
    bank /= np.maximum(np.linalg.norm(bank, axis=-1, keepdims=True), 1e-12)
    numeric, errors = {}, {}
    arm_directory = directory
    for dataset, records, tc, tn, ouv, pack in (("allen_train", real, ac, an, auv, None), ("synthetic_train", train["records"], tc_train, tn_train, None, train), ("synthetic", dev["records"], dc, dn, None, dev), ("allen_raw", real_dev, real_dc, real_dn, dev_ouv, None)):
        directory = arm_directory / dataset; report = json.loads((directory / "summary.json").read_text()); saved = dict(np.load(directory / "rows.npz"))
        full = dataset in ("synthetic", "allen_raw")
        assert report == arm_done["results"][dataset] and report["arm"] == arm and report["step"] == FINAL and report["checkpoint_sha256"] == checkpoint_sha256
        assert report["gallery_sha256"] == bank_hash and report["full_raw_scores_saved"] == full and not report["probabilities_calibrated"]
        for name, expected in report["output_sha256"].items(): assert sha(directory / name) == expected
        top = saved["top128_cell_index"]; assert top.shape == (len(records), 128) and ((top >= 0) & (top < 98304)).all()
        assert all(len(np.unique(row)) == 128 for row in top)
        if full:
            raw = np.load(directory / "raw_cell_log_probability.npy", mmap_mode="r"); component = np.load(directory / "raw_component_log_score.npy", mmap_mode="r")
            assert raw.shape == (len(records), 98304) and component.shape == (len(records), 98304, 2)
        synthetic_metrics = []; error = normalization_error = score_error = 0.; rank_ambiguities = normal_ambiguities = top_roundoff_rows = 0
        for i in range(0, len(records), 16):
            q = np.array(saved["query_descriptor"][i:i+16], np.float32); assert np.isfinite(q).all()
            q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
            reconstructed_component = (q @ bank.reshape(-1, 256).T).reshape(len(q), 98304, 2) / .1 + rep_prior[None] + cell_mass[None, :, None]
            assert np.isfinite(reconstructed_component).all()
            logits = np.logaddexp.reduce(reconstructed_component.astype(np.float64), axis=-1)
            reconstructed_lp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
            selected = top[i:i+len(q)]; selected_index = np.arange(len(q))[:, None]
            if full:
                block = np.asarray(raw[i:i+len(q)]); components = np.asarray(component[i:i+len(q)])
                assert np.isfinite(block).all() and np.isfinite(components).all()
                logit = np.logaddexp.reduce(components.astype(np.float64), axis=-1)
                expected_lp = logit - np.logaddexp.reduce(logit, axis=-1, keepdims=True)
                error = max(error, float(np.abs(expected_lp - block).max()))
                score_error = max(score_error, float(np.abs(reconstructed_component - components).max()), float(np.abs(reconstructed_lp - block).max()))
                assert np.array_equal(np.argsort(-block, axis=-1, kind="stable")[:, :128], selected)
                selected_component = components[selected_index, selected].astype(np.float64)
            else:
                block = reconstructed_lp
                ranked_score = np.sort(np.partition(block, -128, axis=-1)[:, -128:], axis=-1)[:, ::-1]
                selected_score = block[selected_index, selected]
                score_error = max(score_error, float(np.abs(selected_score - saved["top128_cell_log_probability"][i:i+len(q)]).max()))
                assert np.allclose(selected_score, saved["top128_cell_log_probability"][i:i+len(q)], rtol=0, atol=TOL)
                assert np.allclose(selected_score, ranked_score, rtol=0, atol=2*TOL)
                top_roundoff_rows += int(np.any(np.abs(selected_score - ranked_score) > 1e-10, axis=-1).sum())
                selected_component = reconstructed_component[selected_index, selected].astype(np.float64)
            normalization_error = max(normalization_error, float(np.abs(np.logaddexp.reduce(block.astype(np.float64), axis=-1)).max()))
            assert np.allclose(block[selected_index, selected], saved["top128_cell_log_probability"][i:i+len(q)], rtol=0, atol=TOL)
            assert np.allclose(np.exp(selected_component - np.logaddexp.reduce(selected_component, axis=-1, keepdims=True)), saved["top128_representation_probability"][i:i+len(q)], rtol=0, atol=TOL)
            if pack is not None:
                labels = pack["label"][i:i+len(q)].numpy(); truth = block[np.arange(len(q)), labels]
                nlp = np.logaddexp.reduce(block.reshape(-1, 384, 256), axis=-1)
                if full:
                    ranks = 1 + (block > truth[:, None]).sum(-1) + ((block == truth[:, None]) & (np.arange(98304)[None] < labels[:, None])).sum(-1)
                    mn = normals[nlp.argmax(-1) * 256]
                    normal_angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(mn, tn[i:i+len(q)]), axis=-1), np.abs((mn * tn[i:i+len(q)]).sum(-1))))
                else:
                    lower = 1 + (block > truth[:, None] + 2*TOL).sum(-1); upper = (block >= truth[:, None] - 2*TOL).sum(-1)
                    ranks = saved["truth_rank"][i:i+len(q)]; assert ((ranks >= lower) & (ranks <= upper)).all()
                    rank_ambiguities += int((lower != upper).sum())
                    all_normal = normals[::256]; dot = tn[i:i+len(q)] @ all_normal.T
                    angles = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(tn[i:i+len(q), None], all_normal[None]), axis=-1), np.abs(dot)))
                    allowed = nlp >= nlp.max(-1, keepdims=True) - 2*TOL
                    normal_angle = saved["normal_marginal_map_angle_deg"][i:i+len(q)]
                    assert np.all(np.min(np.where(allowed, np.abs(angles - normal_angle[:, None]), np.inf), axis=-1) < .002)
                    normal_ambiguities += int((allowed.sum(-1) > 1).sum())
                synthetic_metrics.append(np.stack((-truth, ranks, -nlp[np.arange(len(q)), labels // 256], normal_angle), axis=1))
        assert error < TOL and score_error < TOL and normalization_error < TOL
        errors[dataset] = {"full_raw_available": full, "component_to_cell_max_abs": error if full else None, "descriptor_reconstruction_max_abs": score_error, "descriptor_error_scope": "full component/cell scores" if full else "saved top128 cell scores", "log_normalization_max_abs": normalization_error, "train_top_rows_with_roundoff_order_difference": top_roundoff_rows, "train_truth_rank_roundoff_intervals": rank_ambiguities, "train_marginal_normal_roundoff_intervals": normal_ambiguities}
        assert np.allclose(saved["reference_center_ap_dv_ml_um"], tc, atol=.002, rtol=0) and np.allclose(np.abs((saved["reference_normal_ap_dv_ml"] * tn).sum(-1)), 1, rtol=0, atol=1e-6)
        dot = (normals[top] * tn[:, None]).sum(-1); angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(normals[top], tn[:, None]), axis=-1), np.abs(dot)))
        offset = np.abs(((centres[top] - origin) * normals[top]).sum(-1) - np.where(dot < 0, -1, 1) * ((tc - origin) * tn).sum(-1)[:, None])
        metrics = {"plane_angle_deg": angle[:, 0], "normal_offset_error_um": offset[:, 0], **{f"physical_plane_capture_at_{k}": ((angle[:, :k] <= 10) & (offset[:, :k] <= 500)).any(-1).astype(float) for k in (32, 128)}}
        subsets = {"all": np.ones(len(records), bool)}
        if ouv is not None:
            p = cell_ouv[top[:, 0]]; pu, pv = p[:, 1] * 95 / 96, p[:, 2] * 95 / 96; ru, rv = ouv[:, 1] * 95 / 96, ouv[:, 2] * 95 / 96
            metrics["map_finite_frame_rms_um"] = np.sqrt(((p[:, 0] + .5 * (pu + pv) - ouv[:, 0] - .5 * (ru + rv))**2).sum(-1) + .25 * (np.minimum(((pu - ru)**2).sum(-1), ((pu + ru)**2).sum(-1)) + ((pv - rv)**2).sum(-1)))
            pa = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(tn, prior_normal), axis=-1), np.abs(tn @ prior_normal)))
            metrics.update(train_orientation_prior_angle_deg=pa, model_minus_train_orientation_prior_angle_deg=angle[:, 0] - pa)
        if pack is not None:
            labels = pack["label"].numpy(); weight = pack["weight"].numpy(); modes = np.array([r["selected_mode"] for r in records]); values = np.concatenate(synthetic_metrics); ranks = values[:, 1]
            metrics.update(nll=values[:, 0], truth_rank=ranks, normal_marginal_nll=values[:, 2], normal_marginal_map_angle_deg=values[:, 3], **{f"hit_at_{k}": (ranks <= k).astype(float) for k in (1, 8, 32, 128)})
            assert np.array_equal(saved["label"], labels) and np.array_equal(saved["pose_supervision_weight"], weight) and np.array_equal(saved["selected_mode"], modes)
            subsets.update(eligible=weight > 0, censored=weight <= 0, **{f"eligible_mode:{mode}": (weight > 0) & (modes == mode) for mode in np.unique(modes)})
        if dataset == "allen_train":
            subsets.update(eligible=eligible, ineligible=~eligible); assert np.array_equal(saved["common_anchor_eligible"], eligible)
            assert np.allclose(saved["matched_support_mass"], support[0], atol=.001, rtol=0) and np.allclose(saved["canonical_support_mass"], support[1], atol=.001, rtol=0)
            anchor = dict(np.load(directory / "exact_anchor_reconstruction.npz")); scores = anchor["uniform_representation_logit"]
            assert np.isfinite(anchor["raw_component_logit"]).all() and np.allclose(np.logaddexp.reduce(anchor["raw_component_logit"].astype(np.float64) - np.log(2.), axis=-1), scores, rtol=0, atol=TOL)
            assert np.array_equal(anchor["query_descriptor"], saved["query_descriptor"])
            q = anchor["query_descriptor"].astype(np.float32); q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
            d = an @ an.T; offsets = ((ac - origin) * an).sum(-1); near = (np.abs(d) >= np.cos(np.deg2rad(10))) & (np.abs(offsets[None] - np.where(d < 0, -1, 1) * offsets[:, None]) <= 500)
            assert np.array_equal(anchor["near_physical_mask"], near)
            for chart_id, name in enumerate(("matched", "canonical")):
                keys = anchor["atlas_descriptor"][chart_id].astype(np.float32); keys /= np.maximum(np.linalg.norm(keys, axis=-1, keepdims=True), 1e-12)
                assert np.isfinite(keys).all() and np.allclose((q @ keys.reshape(-1, 256).T).reshape(N_REAL, N_REAL, 2) / .1, anchor["raw_component_logit"][chart_id], rtol=0, atol=TOL)
                a = scores[chart_id]; own = np.diag(a); rank = 1 + (a > own[:, None]).sum(-1) + ((a == own[:, None]) & (np.arange(N_REAL)[None] < np.arange(N_REAL)[:, None])).sum(-1)
                nrank = 1 + np.argmax(np.take_along_axis(near, np.argsort(-a, axis=-1, kind="stable"), axis=1), axis=1)
                metrics.update({f"{name}_own_anchor_rank": rank, f"{name}_near_physical_anchor_rank": nrank, **{f"{name}_own_anchor_hit_at_{k}": (rank <= k).astype(float) for k in (1, 8, 32)}, **{f"{name}_near_physical_anchor_hit_at_{k}": (nrank <= k).astype(float) for k in (1, 8, 32)}})
        for key, value in metrics.items(): assert np.allclose(value, saved[key], rtol=1e-6, atol=.002), (arm, dataset, key)
        group = np.array([r["animal_id"] for r in records]); summary = {}
        for key in ("animal_id", "specimen_id", "experiment_id", "section_id"): assert np.array_equal(saved[key], np.array([r[key] for r in records]))
        for name, selected in subsets.items():
            groups = np.unique(group[selected]); by_group = {str(d): {key: float(value[selected & (group == d)].mean()) for key, value in metrics.items()} for d in groups}
            macro = {key: float(np.mean([row[key] for row in by_group.values()])) for key in metrics} if len(groups) else None
            expected = report["subsets"][name]; assert (expected["rows"], expected["groups"]) == (int(selected.sum()), len(groups))
            if macro:
                for key, value in macro.items(): assert np.isclose(value, expected["group_macro"][key], rtol=1e-6, atol=.002)
                for d, row in by_group.items():
                    assert expected["by_group"][d]["rows"] == int((selected & (group.astype(str) == d)).sum())
                    for key, value in row.items(): assert np.isclose(value, expected["by_group"][d][key], rtol=1e-6, atol=.002)
            summary[name] = {"rows": int(selected.sum()), "groups": len(groups), "group_macro": macro}
        if ouv is not None:
            assert all(np.isclose(value, report["donor_macro"][key], rtol=1e-6, atol=.002) for key, value in summary["all"]["group_macro"].items())
        numeric[dataset] = summary
        if full: del raw, component
        print(json.dumps({"arm": arm, "audited_dataset": dataset, "rows": len(records)}), flush=True)
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
    assert gates == arm_done["gates"] == done["arm_results"][arm]["gates"] and all(gates.values()) == arm_done["recovery_gate_passed"] == done["arm_results"][arm]["recovery_gate_passed"]
    assert all(np.isclose(value, done["arm_results"][arm]["real_donor_macro"][key], atol=.002, rtol=1e-6) for key, value in final_real.items())
    arm_audits[arm] = {"gates": gates, "recovery_gate_passed": bool(all(gates.values())), "numeric_results": numeric, "numeric_errors": errors, "frozen_nonretrieval_tensors": len(frozen), "changed_retrieval_tensors": changed, "fresh_unique_cells_attempted": count, "fresh_unique_cells_eligible": fresh_eligible, "stage_unique_eligible_positive_cells": len(positive_cells)}
assert ("B" in arm_audits) == (not arm_audits["A"]["recovery_gate_passed"])
for path in sorted(RUN.rglob("*")):
    if path.is_file(): sha(path)
OUT.mkdir(parents=True, exist_ok=False)
result = {"integrity_passed": True, "arm_audits": arm_audits, "conditional_B_policy_verified": True, "same_parent_model_optimizer_rng_verified": True, "exact_updates_per_arm": STEPS, "fresh_cell_schedule_verified": True, "training_rows": {"real": N_REAL, "donors": 58, "frozen_synthetic": 5120}, "train_orientation_prior_verified": True, "scope": "Independent frozen raw DEV component/descriptor normalization, exact stable top128 and geometric gates; TRAIN descriptor reconstruction, bounded-roundoff ranks and original saved-top geometry/subset/group metrics; full post-exit file inventory. One-atlas organizational groups are not independent animals.", "not_replayed": ["atlas rendering, acquired-image decoding or encoder re-embedding", "optimizer/backpropagation trajectory between authenticated whole checkpoints", "full per-step conditional-uniform quantile or snapshot-hard argmax selection, local-neighbour pool ranks and exclusion-mask replay; exact random schedules/candidate unions/selected-owner admissibility/snapshot bindings are checked", "exact CPU/GPU agreement within2e-5 score roundoff: ambiguous TRAIN rank/normal intervals are reported, not claimed exact; raw DEV rankings/gates remain exact", "A6000/B8000 raw predictions are not recomputed again; their independently audited pinned summaries supply fixed references"], "probabilities_calibrated": False, "artifact_sha256": hashes}
(OUT / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"integrity_passed": True, "arm_gates": {arm: value["recovery_gate_passed"] for arm, value in arm_audits.items()}, "audit": str(OUT / "audit.json")}), flush=True)
