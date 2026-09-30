"""Fixed TRAIN-only coherent canonical adaptation; whole audited A22576 continuation."""
import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda")
sys.dont_write_bytecode = True
READY_AFTER_ROOT_REVIEW = True
assert READY_AFTER_ROOT_REVIEW, "Prepared only: root reviews, commits and schedules; no DEV or benchmark data"

import copy
import hashlib
import json
import random
import shutil
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_catalogue_runtime_v6 import make_complete_catalogue_runtime_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components, full_frame_state_to_components, full_frame_state_to_physical_ouv, render_finite_thickness_plane
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6

RUN = ROOT / "runs/joint_v6_coherent_canonical_adaptation_001"
PARENT = ROOT / "runs/joint_v6_full_coverage_ranking_001"
CACHE = ROOT / "runs/joint_v6_coherent_train_candidates_001"
PREPARED = ROOT / "runs/joint_v6_proposal_substantive_001"
PARENT_CHECKPOINT = PARENT / "A/joint_model_step_22576.pt"
PARENT_GALLERY = PARENT / "A/gallery_descriptors_step_22576.npy"
PARENT_AUDIT = ROOT / "runs/joint_v6_full_coverage_ranking_001_independent_audit/audit.json"
CACHE_AUDIT = ROOT / "runs/joint_v6_coherent_train_candidates_001_independent_audit/audit.json"
PINS = {PARENT_CHECKPOINT: "e22ff8e13a09b8c624c64518ca65a0ac9feda2c823dde30b6b8b5707995d52d8",
    PARENT_GALLERY: "5219117c897bf68fb6139b5bd482bfdd05f5c52e2b9a5a7296c9936d0f8d0b36",
    PARENT_AUDIT: "c75c4ff817e7603eca31c8086080aafdb7f0953b3fec58c3f3f2f7842cf2afb3",
    PARENT / "completed.json": "576b85c7bcdbeea9e361b15908e320fe0d838291d083c945d19ede2b1687625e",
    CACHE / "completed.json": "5f075e6ce34db1527c0c172757c6995a59fc8d7d42a833fbe25bf59942838ba2",
    CACHE_AUDIT: "995272fc036b7e1011bae56fa01678ef21622b79e8f6e1d28879e5f247e1af6f"}
STEPS, COHERENT_BATCH, REAL_BATCH, SYNTHETIC_BATCH, GENERATED = 1920, 8, 8, 8, 4
START, FINAL, COHERENT_SEED = 22576, 24496, 2026092937
SHAPE, DESCRIPTOR_DIM, KEY_CHUNK = (96, 96), 256, 16
TEMPERATURE, KEY_THICKNESS_UM, SUPPORT_MASS_THRESHOLD = .1, 50., 64.
TRAINABLE = ("pose_model.histology_stem.", "pose_model.atlas_stem.", "pose_model.shared_encoder.", "pose_model.image_key_descriptor.")
repository = Path(__file__).resolve().parents[1]
bindings = {}


def sha(path):
    with Path(path).open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    bindings[str(path)] = value
    return value


for path, expected in PINS.items(): assert sha(path) == expected
audit = json.loads(PARENT_AUDIT.read_text())
assert audit["integrity_passed"] and all(audit["arm_audits"]["A"]["gates"].values())
assert json.loads(CACHE_AUDIT.read_text())["integrity_pass"]
parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
previous = parent["experiment"]
assert parent["step"] == parent["optimizer_steps_applied"] == START and tuple(previous["trained_modules"]) == TRAINABLE
assert not previous["model_kwargs"].get("signed_pose_evidence", False)
cache_receipt = json.loads((CACHE / "completed.json").read_text())
for name in ("observation_identities.json", "section_provenance.json", "supervision_sidecar.npz"):
    assert sha(CACHE / name) == cache_receipt["output_sha256"][name]
coherent_records = json.loads((CACHE / "observation_identities.json").read_text())
sections = json.loads((CACHE / "section_provenance.json").read_text())
side = dict(np.load(CACHE / "supervision_sidecar.npz", allow_pickle=False))
assert len(coherent_records) == 1920 and len(sections) == 640 and {r["split"] for r in coherent_records} == {"train"}
assert len({r["subject_id"] for r in coherent_records}) == 8
coherent_inputs = np.empty((1920, 3, 96, 96), dtype=np.float32)
for si, section in enumerate(sections):
    directory, record = Path(section["source_directory"]), section["record"]
    assert record["lineage"]["split"] == "train"
    path = directory / record["artifacts"]["arrays"]
    assert sha(path) == record["artifact_sha256"][record["artifacts"]["arrays"]]
    with np.load(path, allow_pickle=False) as arrays:
        for j, mode in enumerate(("raw", "exact_black", "imperfect_brush")):
            i = 3 * si + j; row = coherent_records[i]
            assert row["section_index_in_cache"] == si and row["selected_mode"] == mode
            image = arrays[row["input_array_key"]]
            assert hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() == row["input_channels_sha256"]
            coherent_inputs[i] = image
    if (si + 1) % 128 == 0: print(json.dumps({"loaded_TRAIN_sections": si + 1}), flush=True)
coherent_channels = torch.from_numpy(coherent_inputs)
section_index = torch.arange(640).repeat_interleave(3)
original_coherent_eligible = torch.tensor([r["support_information_eligible"] for r in coherent_records])
for name in ("catalogue.pt", "training_prepared.pt"):
    assert sha(PREPARED / name) == previous["prepared_source_sha256"][name]
catalogue = torch.load(PREPARED / "catalogue.pt", map_location="cpu", weights_only=False)
train = torch.load(PREPARED / "training_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
assert len(train["records"]) == 5120 and {r["split"] for r in train["records"]} == {"train"}
for name in ("generated_schedule.npz", "sampling_schedule.npz", "weak_affine_anchors.npz", "real_training_identities.json", "real_anchor_image_cache.npy", "real_anchor_eligibility.npz"):
    assert sha(PARENT / name) == audit["artifact_sha256"][str(PARENT / name)]
with np.load(PARENT / "generated_schedule.npz", allow_pickle=False) as stored:
    schedule = {k: stored[k][:STEPS * GENERATED] if stored[k].ndim else stored[k] for k in stored.files}
with np.load(PARENT / "sampling_schedule.npz", allow_pickle=False) as stored:
    replay = {k: stored[k][:STEPS] for k in stored.files}
frozen_schedule, real_row_schedule = replay["frozen_row_index"], replay["real_row_index"]
global_schedule, local_rank_schedule = replay["global_cell_index"], replay["local_pool_rank"]
conditional_draw, chart_schedule, donor_schedule = replay["conditional_uniform_fraction"], replay["canonical_chart_index"], replay["donor_id"]
rng = np.random.default_rng(COHERENT_SEED)
coherent_schedule = np.concatenate([rng.permutation(1920) for _ in range(8)]).reshape(STEPS, COHERENT_BATCH)
real_records = json.loads((PARENT / "real_training_identities.json").read_text())
assert len(real_records) == 1280 and {r["split"] for r in real_records} == {"development_train"}
real_parts = []
for directory in ("allen_real_training_inputs_20260929", "allen_real_training_expansion_20260929"):
    path = ROOT / "data" / directory / "raw_model_input.npy"
    assert sha(path) == previous["input_sha256"][str(path)]
    real_parts.append(np.load(path, allow_pickle=False))
real_channels = torch.cat((torch.from_numpy(np.concatenate(real_parts)), torch.zeros(1280, 2, 96, 96)), dim=1)
del real_parts
weak = dict(np.load(PARENT / "weak_affine_anchors.npz", allow_pickle=False))
real_states = torch.from_numpy(weak["full_frame_state"])
real_physical_center, real_frame, _ = full_frame_state_to_components(real_states)
real_plane_normal = real_frame[:, :, 2]
coherent_ouv = torch.from_numpy(side["canonical_anatomy_fitted_ouv_ap_dv_ml_um"])
coherent_states = full_frame_state_from_components(*physical_ouv_to_frame(coherent_ouv))
coherent_center, coherent_frame, _ = full_frame_state_to_components(coherent_states)
coherent_normal = coherent_frame[:, :, 2]
support_origin = torch.as_tensor(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"], dtype=torch.float64)
canonical_center = support_origin + coherent_normal * ((coherent_center - support_origin) * coherent_normal).sum(-1, keepdim=True)
canonical_states = full_frame_state_from_components(canonical_center, coherent_frame, torch.eye(2, dtype=torch.float64).expand(640, -1, -1) * 12000.)
assert torch.allclose(full_frame_state_to_physical_ouv(coherent_states).reshape(-1, 3, 3), coherent_ouv, rtol=0, atol=1e-8)
RUN.mkdir(parents=True, exist_ok=False)
source_names = set(previous["source"]["file_sha256"]) | {"training/arbitrary_plane_joint_uncertainty.py", Path(__file__).relative_to(repository).as_posix(), "docs/publication/COHERENT_CANONICAL_ADAPTATION_001_PROTOCOL_20260929.md"}
source = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(), "file_sha256": {}}
for name in sorted(source_names):
    path = RUN / "source" / name; path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository / name, path); source["file_sha256"][name] = sha(path)
np.savez(RUN / "generated_replay_schedule.npz", **schedule)
np.savez(RUN / "sampling_schedule.npz", **replay, coherent_observation_index=coherent_schedule)
np.savez(RUN / "coherent_plane_targets.npz", canonical_anatomy_fitted_ouv=coherent_ouv.numpy(), canonical_key_state=canonical_states.numpy(), already_observed_fitted_ouv=side["already_observed_total_map_fitted_ouv_ap_dv_ml_um"], reflection_xy=side["reflection_xy"], target_axial_offsets_um=side["axial_offsets_um"], target_axial_weights=side["axial_weights"])
for name, records in (("coherent", coherent_records), ("synthetic", train["records"]), ("real", real_records)):
    (RUN / f"{name}_TRAIN_identities.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
shutil.copyfile(CACHE / "section_provenance.json", RUN / "section_provenance.json")
schedule_hashes = {name: sha(RUN / name) for name in ("generated_replay_schedule.npz", "sampling_schedule.npz", "coherent_plane_targets.npz", "coherent_TRAIN_identities.json", "synthetic_TRAIN_identities.json", "real_TRAIN_identities.json", "section_provenance.json")}
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
started = time.perf_counter()
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda(); del atlas_array, annotation
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"], device="cuda", dtype=torch.float32)
cell_states = torch.as_tensor(catalogue["arrays"]["cell_states_float64"])
cell_center, cell_frame, _ = full_frame_state_to_components(cell_states)
cell_normal = cell_frame[:, :, 2]
cell_ouv = full_frame_state_to_physical_ouv(cell_states).reshape(-1, 3, 3)
key_u, key_v = cell_ouv[:, 1] * (95 / 96), cell_ouv[:, 2] * (95 / 96)
key_center = cell_ouv[:, 0] + .5 * (key_u + key_v)
key_center, key_u, key_v, key_normal = [v.cuda().float() for v in (key_center, key_u, key_v, cell_normal)]
train_center, train_frame, _ = full_frame_state_to_components(train["truth_state"].double())
train_normal = train_frame[:, :, 2]
train_normal_gpu = train_normal.cuda().float()
train_offset_gpu = ((train_center - support_origin) * train_normal).sum(-1).cuda().float()
real_normal = real_plane_normal.cuda().float()
real_plane_offset = ((real_physical_center - support_origin) * real_plane_normal).sum(-1).cuda().float()
coherent_normal_gpu = coherent_normal.cuda().float()
coherent_offset_gpu = ((coherent_center - support_origin) * coherent_normal).sum(-1).cuda().float()
cell_plane_offset = ((cell_center - support_origin) * cell_normal).sum(-1).cuda().float()
render_states = cell_states.cuda().float()
cell_log_mass = catalogue["tensors"]["cell_log_mass"][0].cuda().float()
representation_log_weight = catalogue["tensors"]["representation_log_weight"][0].cuda().float()
psf_weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device="cuda", dtype=torch.float32) / 16
psf_positions = torch.linspace(-.5, .5, 9, device="cuda")
yy, xx = torch.meshgrid(torch.linspace(-1, 1, 96, device="cuda"), torch.linspace(-1, 1, 96, device="cuda"), indexing="ij")
noise_generator = torch.Generator(device="cuda")
origin, spacing = [catalogue["support_geometry"][k] for k in ("origin_ap_dv_ml_um", "voxel_size_ap_dv_ml_um")]
anchor_cache = torch.from_numpy(np.load(PARENT / "real_anchor_image_cache.npy", allow_pickle=False)).cuda()
anchor_support = anchor_cache[:, :, 1].clamp(0, 1).sum((-2, -1))
real_eligible = (anchor_support >= SUPPORT_MASS_THRESHOLD).all(0)
stored_real_eligibility = np.load(PARENT / "real_anchor_eligibility.npz", allow_pickle=False)
assert np.array_equal(real_eligible.cpu().numpy(), stored_real_eligibility["common_eligible"])
with torch.no_grad():
    coherent_anchor_cache = torch.cat([render_finite_thickness_plane(atlas, part.cuda().float(), SHAPE, origin, spacing, psf_positions * KEY_THICKNESS_UM, psf_weights) for part in canonical_states.split(16)])
coherent_key_support = coherent_anchor_cache[:, 1].clamp(0, 1).sum((-2, -1))
coherent_key_valid = coherent_key_support >= SUPPORT_MASS_THRESHOLD
coherent_weight_cpu = original_coherent_eligible & coherent_key_valid.cpu()[section_index]
np.save(RUN / "coherent_canonical_atlas_images.npy", coherent_anchor_cache.cpu().numpy())
np.savez(RUN / "coherent_eligibility.npz", original_query_eligible=original_coherent_eligible.numpy(), canonical_key_support=coherent_key_support.cpu().numpy(), canonical_key_valid=coherent_key_valid.cpu().numpy(), training_weight=coherent_weight_cpu.numpy())
config = {"source": source, "parent_checkpoint": str(PARENT_CHECKPOINT), "parent_sha256": PINS[PARENT_CHECKPOINT], "parent_gallery_sha256": PINS[PARENT_GALLERY], "parent_audit_sha256": PINS[PARENT_AUDIT], "model_kwargs": previous["model_kwargs"], "trained_modules": TRAINABLE, "input_sha256": dict(bindings), "schedule_sha256": schedule_hashes, "start": START, "additional_updates": STEPS, "final": FINAL, "batch": {"coherent": 8, "generated": 4, "frozen_synthetic": 4, "real": 8}, "coherent_seed": COHERENT_SEED, "coherent_passes": 8, "replay": "first1920 completedA updates, unchanged images/seeds/negative and chart schedules; no new coverage", "optimizer": "exact whole-parent AdamW and RNG; LR.00025,WD.0001,clip5,FP32 noAMP/TF32", "loss": ".5 coherent continuous NCE +1/3 unchanged synthetic sampled cellNLL +1/6 unchanged real continuous NCE", "coherent_positive": "canonical anatomy-fit plane and roll;support-centred12mm orthogonal chart;bothR;no nearest-cell label or dense teacher", "coherent_negative_exclusion": "per-query antipodal<=10deg AND sign-offset<=500um;invalid canonical keys excluded;own positive restored", "replay_denominators": "synthetic catalogue-only;real old real+catalogue;coherent new coherent+catalogue only", "key_psf": "normalized trapezoid9,50um;known mismatch to query25–100um;PSF never query input", "training_eligibility": "original finite/visible eligibility AND canonical-key support>=64;all observations retained", "endpoint_gate_eligibility": "original query eligibility, NOT new key-support intersection", "coherent_original_eligible": int(original_coherent_eligible.sum()), "coherent_trainable": int(coherent_weight_cpu.sum()), "canonical_unsupported_sections": int((~coherent_key_valid).sum()), "evaluation": "TRAINonly0/1920, all1920+5120+1280;fresh fullgallery, no DEV/benchmark", "probabilities_calibrated": False, "scope": "TRAIN-effectiveness only;eight synthetic subjects;continuous fit is not dense curved truth;no native or biological qualification"}
config["canonical_cache_sha256"] = {n: sha(RUN / n) for n in ("coherent_canonical_atlas_images.npy", "coherent_eligibility.npz")}
(RUN / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
model = ArbitraryPlaneJointModelV6(runtime, **previous["model_kwargs"]).cuda()
model.load_state_dict(parent["model_state"], strict=True)
assert all(torch.equal(v.cpu(), parent["model_state"][k]) for k, v in model.state_dict().items())
for name, p in model.named_parameters(): p.requires_grad_(name.startswith(TRAINABLE))
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=.00025, weight_decay=1e-4)
optimizer.load_state_dict(copy.deepcopy(parent["optimizer_state"]))
loaded = optimizer.state_dict()
assert loaded["param_groups"] == parent["optimizer_state"]["param_groups"]
assert all(g["lr"] == .00025 and g["weight_decay"] == 1e-4 for g in loaded["param_groups"])
for k, state in parent["optimizer_state"]["state"].items():
    assert all(torch.equal(loaded["state"][k][n].cpu(), v) if torch.is_tensor(v) else loaded["state"][k][n] == v for n, v in state.items())
del loaded


def encode_keys(images):
    return model.pose_model.descriptor_from_features(model.pose_model._encode_atlas(images))


def evaluate(applied):
    model.eval()
    output = RUN / f"step_{applied:04d}"; output.mkdir()
    cp = output / f"joint_model_step_{START + applied:05d}.pt"
    torch.save({"experiment": config, "phase": "experimental_image_key_proposal_only", "step": START + applied, "optimizer_steps_applied": START + applied, "additional_optimizer_steps_applied": applied, "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer_state": optimizer.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "probabilities_calibrated": False}, cp)
    cp_sha = sha(cp)
    bank = torch.empty((98304, 2, DESCRIPTOR_DIM), device="cuda", dtype=torch.float16)
    with torch.no_grad():
        for start in range(0, 98304, 64):
            images = render_finite_thickness_plane(atlas, render_states[start:start+64], SHAPE, origin, spacing, psf_positions * KEY_THICKNESS_UM, psf_weights)
            images = torch.stack((images, images.flip(-1)), dim=1).flatten(0, 1)
            bank[start:start+64] = torch.cat([encode_keys(part) for part in images.split(KEY_CHUNK)]).reshape(-1, 2, DESCRIPTOR_DIM).half()
        if applied == 0:
            assert np.array_equal(bank.cpu().numpy(), np.load(PARENT_GALLERY, allow_pickle=False))
        np.save(output / "gallery.npy", bank.cpu().numpy()); bank_sha = sha(output / "gallery.npy")
        anchor_keys = torch.cat([encode_keys(torch.stack((part, part.flip(-1)), 1).flatten(0, 1)) for part in coherent_anchor_cache.split(8)]).reshape(640, 2, DESCRIPTOR_DIM)
        results = {}
        datasets = (("coherent", coherent_channels, coherent_records, coherent_center[section_index], coherent_normal[section_index]), ("synthetic", train["channels"], train["records"], train_center, train_normal), ("real", real_channels, real_records, real_physical_center, real_plane_normal))
        for name, channels, records, centers, normals in datasets:
            descriptors, tops, lps, rlps, components, retained, z_values = [], [], [], [], [], [], []
            for start in range(0, len(records), 16):
                inputs = channels[start:start+16].cuda()
                query = model.pose_model.descriptor_from_features(model.pose_model.encode_histology(inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1))))
                rscore = model.pose_model.image_key_cosine_logits(query, bank, TEMPERATURE) + representation_log_weight[None]
                lp = (torch.logsumexp(rscore, -1) + cell_log_mass[None]).log_softmax(-1)
                assert bool(torch.isfinite(lp).all())
                top = torch.argsort(lp, descending=True, stable=True)[:, :128]
                top_lp = lp.gather(1, top)
                rlp = rscore[torch.arange(len(inputs), device="cuda")[:, None], top].log_softmax(-1)
                z = torch.logsumexp(lp.double(), -1)
                assert bool((z.abs() < 2e-5).all())
                descriptors.append(query.cpu().numpy()); tops.append(top.cpu())
                lps.append(top_lp.cpu().numpy()); rlps.append(rlp.cpu().numpy())
                components.append((top_lp[..., None] + rlp).cpu().numpy())
                retained.append(torch.exp(torch.logsumexp(top_lp.double(), -1)-z).cpu().numpy()); z_values.append(z.cpu().numpy())
            top = torch.cat(tops); candidates = cell_normal[top]
            dot = (candidates * normals[:, None]).sum(-1)
            angle = torch.rad2deg(torch.atan2(torch.linalg.cross(candidates, normals[:, None]).norm(dim=-1), dot.abs()))
            offset = (((cell_center[top]-support_origin)*candidates).sum(-1)-torch.where(dot < 0, -1., 1.)*((centers-support_origin)*normals).sum(-1)[:, None]).abs()
            metrics = {"plane_angle_deg": angle[:, 0].numpy(), "normal_offset_error_um": offset[:, 0].numpy(), **{f"capture{k}": ((angle[:, :k] <= 10) & (offset[:, :k] <= 500)).any(-1).double().numpy() for k in (32, 128)}, "retained_mass": np.concatenate(retained)}
            eligible = original_coherent_eligible.numpy() if name == "coherent" else train["weight"].numpy() > 0 if name == "synthetic" else np.ones(1280, dtype=bool)
            groups = np.array([r["subject_id"] if name == "coherent" else r["animal_id"] for r in records])
            scopes = {"all": np.ones(len(records), bool), "eligible": eligible, "censored": ~eligible}
            if name != "real":
                modes = np.array([r["selected_mode"] for r in records])
                scopes.update({f"eligible_mode:{mode}": eligible & (modes == mode) for mode in np.unique(modes)})
            if name == "coherent":
                corpus = np.array([r["corpus"] for r in records])
                scopes.update({f"eligible:{a}:{b}": eligible & (corpus == a) & (modes == b) for a in ("base", "acquisition") for b in ("raw", "exact_black", "imperfect_brush")})
                scopes["training_weight_eligible"] = coherent_weight_cpu.numpy()
                all_query = torch.from_numpy(np.concatenate(descriptors)).cuda()
                anchor_components = model.pose_model.image_key_cosine_logits(all_query, anchor_keys, TEMPERATURE)
                assert bool(torch.isfinite(anchor_components).all())
                score = torch.logsumexp(anchor_components - np.log(2.), -1).cpu().numpy()
                own = score[np.arange(1920), section_index.numpy()]
                rank = 1 + (score > own[:, None]).sum(-1) + ((score == own[:, None]) & (np.arange(640)[None] < section_index.numpy()[:, None])).sum(-1)
                ad = coherent_normal[section_index] @ coherent_normal.T
                aoff = ((coherent_center-support_origin)*coherent_normal).sum(-1)
                near = ((ad.abs() >= np.cos(np.deg2rad(10.))) & ((aoff[None]-torch.where(ad < 0, -1., 1.)*aoff[section_index, None]).abs() <= 500)).numpy()
                order = np.argsort(-score, kind="stable", axis=-1)
                near_rank = 1 + np.argmax(np.take_along_axis(near, order, axis=1), axis=1)
                metrics.update(own_canonical_rank=rank, near_canonical_rank=near_rank, own_canonical_hit1=(rank == 1).astype(float))
                np.savez_compressed(output / "coherent_anchor_reconstruction.npz", atlas_descriptor=anchor_keys.cpu().numpy(), raw_component_logit=anchor_components.cpu().numpy(), own_section_index=section_index.numpy(), near_plane_mask=near)
                del all_query, anchor_components
            summaries = {}
            for label, mask in scopes.items():
                per_group = {str(g): {"rows": int((mask & (groups == g)).sum()), **{k: float(v[mask & (groups == g)].mean()) for k, v in metrics.items()}} for g in np.unique(groups[mask])}
                summaries[label] = {"rows": int(mask.sum()), "groups": len(per_group), "by_group": per_group, "group_macro": {k: float(np.mean([r[k] for r in per_group.values()])) for k in metrics} if per_group else None}
            path = output / f"{name}_rows.npz"
            np.savez_compressed(path, query_descriptor=np.concatenate(descriptors), top128_cell_index=top.numpy(), top128_cell_log_probability=np.concatenate(lps), top128_conditional_representation_log_probability=np.concatenate(rlps), top128_representation_probability=np.exp(np.concatenate(rlps)), top128_full_component_log_probability=np.concatenate(components), omitted_mass=1-metrics["retained_mass"], full_cell_roundoff_log_normalizer=np.concatenate(z_values), original_eligible=eligible, row_index=np.arange(len(records)), reference_center=centers.numpy(), reference_normal=normals.numpy(), **metrics)
            results[name] = {"checkpoint_sha256": cp_sha, "gallery_sha256": bank_sha, "rows_sha256": sha(path), "subsets": summaries}
            print(json.dumps({"endpoint": applied, "dataset": name, "eligible": summaries["eligible"]["group_macro"]}), flush=True)
        (output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    return results


random.setstate(parent["python_rng"]); np.random.set_state(parent["numpy_rng"])
torch.set_rng_state(parent["torch_rng"]); torch.cuda.set_rng_state_all(parent["cuda_rng"])
initial = evaluate(0)
random.setstate(parent["python_rng"]); np.random.set_state(parent["numpy_rng"])
torch.set_rng_state(parent["torch_rng"]); torch.cuda.set_rng_state_all(parent["cuda_rng"])
(RUN / "initialization.json").write_text(json.dumps({"whole_parent_tensors_exact": True, "optimizer_state_exact": True, "parent_rng_restored_after_initial_evaluation": True, "fresh_initial_gallery_exact_parent": True, "optimizer_step_start": START}), encoding="utf-8")
applied = 0
with (RUN / "training_trace.jsonl").open("w", encoding="utf-8") as trace:
    for step in range(STEPS):
        model.train()
        positions = np.arange(step * GENERATED, (step + 1) * GENERATED)
        with torch.no_grad():
            generated_labels = torch.as_tensor(schedule["cell_index"][positions], device="cuda")
            thickness = torch.as_tensor(schedule["thickness_um"][positions], device="cuda")
            rendered = render_finite_thickness_plane(atlas, render_states[generated_labels], SHAPE, origin, spacing, thickness[:, None] * psf_positions[None], psf_weights)
            finite_support = rendered[:, 1].clamp(0, 1); tissue = finite_support > 0
            generated_inputs = torch.empty((GENERATED, 3, 96, 96), device="cuda"); visible_mass = torch.empty(GENERATED, device="cuda")
            for local, position in enumerate(positions):
                noise_generator.manual_seed(int(schedule["sample_seed"][position]))
                noise = torch.randn((96, 96), device="cuda", generator=noise_generator)
                appearance = (rendered[local, 0] / finite_support[local].clamp_min(1e-6)).clamp(0, 1).pow(float(schedule["gamma"][position]))
                if schedule["invert_tissue"][position]:
                    appearance = 1.0 - appearance
                appearance = (appearance * float(schedule["gain"][position])).clamp(0, 1)
                slope = schedule["background_slope_yx"][position]
                background = float(schedule["background_mean"][position]) + float(slope[0]) * yy + float(slope[1]) * xx
                image = (appearance * finite_support[local] + background * (1.0 - finite_support[local]) + float(schedule["noise_std"][position]) * noise).clamp(0, 1)
                mode = int(schedule["mode_index"][position]); mask = tissue[local].clone()
                if mode == 2:
                    radius = int(schedule["mask_radius_px"][position])
                    if schedule["mask_dilate"][position]:
                        mask = F.max_pool2d(mask[None, None].float(), 2 * radius + 1, 1, radius)[0, 0] > 0
                    else:
                        padded = F.pad(mask[None, None].float(), (radius,) * 4, value=0)
                        mask = -F.max_pool2d(-padded, 2 * radius + 1, 1)[0, 0] > 0
                visible_mass[local] = (finite_support[local] * mask).sum() if mode else finite_support[local].sum()
                outline = torch.zeros_like(mask)
                if mode:
                    image = image * mask
                    eroded = mask.clone(); eroded[1:] &= mask[:-1]; eroded[:-1] &= mask[1:]
                    eroded[:, 1:] &= mask[:, :-1]; eroded[:, :-1] &= mask[:, 1:]
                    eroded[[0, -1], :] = False; eroded[:, [0, -1]] = False
                    outline = mask & ~eroded
                generated_inputs[local] = torch.stack((image, outline.float(), torch.full_like(image, float(mode != 0))))
                if schedule["horizontal_flip"][position]:
                    generated_inputs[local] = generated_inputs[local].flip(-1)
            support_mass = finite_support.sum((-2, -1))
            generated_weight = ((support_mass >= SUPPORT_MASS_THRESHOLD) & (visible_mass >= SUPPORT_MASS_THRESHOLD)).float()
            frozen_index, real_index = frozen_schedule[step], real_row_schedule[step]
            real_index_gpu = torch.as_tensor(real_index, device="cuda")
            synthetic_inputs = torch.cat((generated_inputs, train["channels"][frozen_index].cuda()))
            label = torch.cat((generated_labels, train["label"][frozen_index].cuda()))
            weight = torch.cat((generated_weight, train["weight"][frozen_index].cuda()))
            coherent_index = coherent_schedule[step]
            coherent_sections = section_index[coherent_index].cuda()
            coherent_weight = coherent_weight_cpu[coherent_index].cuda().float()
            inputs = torch.cat((real_channels[real_index].cuda(), synthetic_inputs, coherent_channels[coherent_index].cuda()))
            distance2 = (key_center[label, None] - key_center[None]).square().sum(-1)
            same_u = (key_u[label, None] - key_u[None]).square().sum(-1)
            flip_u = (key_u[label, None] + key_u[None]).square().sum(-1)
            distance2 += .25 * (torch.minimum(same_u, flip_u) + (key_v[label, None] - key_v[None]).square().sum(-1))
            ignore = ((key_normal[label] @ key_normal.T).abs() >= np.cos(np.deg2rad(10.))) & (distance2 <= 1000.**2)
            ignore.scatter_(1, label[:, None], True)
            local_pool = distance2.masked_fill(ignore, torch.inf).topk(8, largest=False, sorted=True).indices
            local_ids = local_pool[torch.arange(SYNTHETIC_BATCH, device="cuda"), torch.as_tensor(local_rank_schedule[step], device="cuda")]
            real_weight = real_eligible[real_index_gpu].float()
            owner_normal = torch.cat((real_normal[real_index_gpu], key_normal[generated_labels], train_normal_gpu[frozen_index]))
            owner_offset = torch.cat((real_plane_offset[real_index_gpu], cell_plane_offset[generated_labels], train_offset_gpu[frozen_index]))
            owner_weight = torch.cat((real_weight, weight))
            owner_dot = owner_normal @ key_normal.T
            owner_admissible = ~((owner_dot.abs() >= np.cos(np.deg2rad(10.))) & ((cell_plane_offset[None] - torch.where(owner_dot < 0, -1., 1.) * owner_offset[:, None]).abs() <= 500.))
            owner_admissible[REAL_BATCH + torch.arange(SYNTHETIC_BATCH, device="cuda"), label] = False
            ranks = torch.floor(torch.as_tensor(conditional_draw[step], device="cuda") * owner_admissible.sum(-1)).long()
            conditional_ids = (owner_admissible.cumsum(-1) > ranks[:, None]).to(torch.uint8).argmax(-1)
            conditional_ids = torch.where(owner_weight > 0, conditional_ids, torch.as_tensor(global_schedule[step], device="cuda"))
            assert bool(owner_admissible[torch.arange(16, device="cuda"), conditional_ids][owner_weight > 0].all())
            pooled = torch.cat((label, torch.as_tensor(global_schedule[step], device="cuda"), local_ids, conditional_ids))
            candidate = torch.unique(pooled, sorted=True)
            target = torch.searchsorted(candidate, label)
            keep = ~ignore[:, candidate]; keep.scatter_(1, target[:, None], True)
            near_plane_far_frame = ~owner_admissible[REAL_BATCH:, candidate] & keep
            near_plane_far_frame.scatter_(1, target[:, None], False)
            near_plane_far_frame_counts = near_plane_far_frame.sum(-1).cpu().tolist()
            del owner_dot, owner_admissible, near_plane_far_frame
            chart_index = torch.as_tensor(chart_schedule[step], device="cuda")
            real_weight = real_eligible[real_index_gpu].float()
            real_key_images = anchor_cache[chart_index, real_index_gpu]
            real_support_mass = anchor_support[chart_index, real_index_gpu]
            synthetic_key_images = render_finite_thickness_plane(atlas, render_states[candidate], SHAPE, origin, spacing, psf_positions * KEY_THICKNESS_UM, psf_weights)
            coherent_key_images = coherent_anchor_cache[coherent_sections]
            key_images = torch.cat((real_key_images, synthetic_key_images, coherent_key_images))
            key_images = torch.stack((key_images, key_images.flip(-1)), dim=1).flatten(0, 1)
            normals = torch.cat((real_normal[real_index_gpu], key_normal[candidate]))
            plane_offsets = torch.cat((real_plane_offset[real_index_gpu], cell_plane_offset[candidate]))
            real_dot = real_normal[real_index_gpu] @ normals.T
            offset_difference = (plane_offsets[None] - torch.where(real_dot < 0, -1., 1.) * real_plane_offset[real_index_gpu, None]).abs()
            real_keep = ~((real_dot.abs() >= np.cos(np.deg2rad(10.))) & (offset_difference <= 500.))
            real_keep[:, :REAL_BATCH] &= real_eligible[real_index_gpu][None]
            real_keep[torch.arange(REAL_BATCH, device="cuda"), torch.arange(REAL_BATCH, device="cuda")] = True
            coherent_key_normals = torch.cat((coherent_normal_gpu[coherent_sections], key_normal[candidate]))
            coherent_key_offsets = torch.cat((coherent_offset_gpu[coherent_sections], cell_plane_offset[candidate]))
            coherent_dot = coherent_normal_gpu[coherent_sections] @ coherent_key_normals.T
            coherent_delta = (coherent_key_offsets[None] - torch.where(coherent_dot < 0, -1., 1.) * coherent_offset_gpu[coherent_sections, None]).abs()
            coherent_keep = ~((coherent_dot.abs() >= np.cos(np.deg2rad(10.))) & (coherent_delta <= 500.))
            coherent_keep[:, :COHERENT_BATCH] &= coherent_key_valid[coherent_sections][None]
            coherent_keep[torch.arange(COHERENT_BATCH, device="cuda"), torch.arange(COHERENT_BATCH, device="cuda")] = True
            del distance2, same_u, flip_u, ignore, local_pool, offset_difference, coherent_delta
        optimizer.zero_grad(set_to_none=True)
        query = model.pose_model.descriptor_from_features(model.pose_model.encode_histology(inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1))))
        keys = torch.cat([checkpoint(encode_keys, part, use_reentrant=False) for part in key_images.split(KEY_CHUNK)]).reshape(REAL_BATCH + len(candidate) + COHERENT_BATCH, 2, DESCRIPTOR_DIM)
        synthetic_scores = model.pose_model.image_key_cosine_logits(query[REAL_BATCH:REAL_BATCH + SYNTHETIC_BATCH], keys[REAL_BATCH:REAL_BATCH + len(candidate)], TEMPERATURE)
        synthetic_scores = torch.logsumexp(synthetic_scores + representation_log_weight[candidate][None], dim=-1) + cell_log_mass[candidate][None]
        synthetic_nll = torch.logsumexp(synthetic_scores.masked_fill(~keep, -torch.inf), dim=-1) - synthetic_scores.gather(1, target[:, None])[:, 0]
        synthetic_loss = (synthetic_nll * weight).sum() / weight.sum().clamp_min(1.)
        real_scores = torch.logsumexp(model.pose_model.image_key_cosine_logits(query[:REAL_BATCH], keys[:REAL_BATCH + len(candidate)], TEMPERATURE) - np.log(2.), dim=-1)
        real_nce = torch.logsumexp(real_scores.masked_fill(~real_keep, -torch.inf), dim=-1) - real_scores[:, :REAL_BATCH].diagonal()
        real_loss = (real_nce * real_weight).sum() / real_weight.sum().clamp_min(1.)
        chart_counts = [int(((chart_index == chart) & (real_weight > 0)).sum()) for chart in (0, 1)]
        chart_losses = [float(real_nce[(chart_index == chart) & (real_weight > 0)].detach().mean()) if chart_counts[chart] else None for chart in (0, 1)]
        coherent_keys = torch.cat((keys[REAL_BATCH + len(candidate):], keys[REAL_BATCH:REAL_BATCH + len(candidate)]))
        coherent_scores = torch.logsumexp(model.pose_model.image_key_cosine_logits(query[-COHERENT_BATCH:], coherent_keys, TEMPERATURE) - np.log(2.), dim=-1)
        coherent_nce = torch.logsumexp(coherent_scores.masked_fill(~coherent_keep, -torch.inf), dim=-1) - coherent_scores[:, :COHERENT_BATCH].diagonal()
        coherent_loss = (coherent_nce * coherent_weight).sum() / coherent_weight.sum().clamp_min(1.)
        loss = .5 * coherent_loss + (1. / 3.) * synthetic_loss + (1. / 6.) * real_loss
        assert bool(torch.isfinite(loss))
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        optimizer.step(); applied += 1
        record = {"step": START + applied, "additional_update": applied, "coherent_observation_index": coherent_index.tolist(), "coherent_training_weight": coherent_weight.cpu().tolist(), "coherent_key_support": coherent_key_support[coherent_sections].cpu().tolist(), "coherent_nce": float(coherent_loss.detach()), "coherent_ignored_counts": (~coherent_keep).sum(-1).cpu().tolist(), "synthetic_nll": float(synthetic_loss.detach()), "real_nce": float(real_loss.detach()), "loss": float(loss.detach()), "gradient_norm": float(gradient), "replayed_parent_update_zero_based": step, "generated_indices": positions.tolist(), "frozen_row_index": frozen_index.tolist(), "real_row_index": real_index.tolist(), "real_chart_index": chart_index.cpu().tolist(), "candidate_cell_index": candidate.cpu().tolist(), "conditional_cell_index": conditional_ids.cpu().tolist(), "local_negative_cell_index": local_ids.cpu().tolist(), "synthetic_supervision_weight": weight.cpu().tolist(), "real_supervision_weight": real_weight.cpu().tolist(), "generated_finite_support": support_mass.cpu().tolist(), "generated_visible_support": visible_mass.cpu().tolist(), "real_anchor_support": real_support_mass.cpu().tolist(), "synthetic_ignored_counts": (~keep).sum(-1).cpu().tolist(), "real_ignored_counts": (~real_keep).sum(-1).cpu().tolist(), "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20, "elapsed_seconds": time.perf_counter()-started}
        trace.write(json.dumps(record) + "\n")
        if applied % 100 == 0 or applied == STEPS:
            trace.flush(); print(json.dumps({k: record[k] for k in ("step", "additional_update", "coherent_nce", "synthetic_nll", "real_nce", "loss", "gradient_norm", "gpu_peak_allocated_mib", "elapsed_seconds")}), flush=True)
        if applied % 480 == 0:
            torch.save({"experiment": config, "phase": "experimental_image_key_proposal_only", "step": START + applied, "optimizer_steps_applied": START + applied, "additional_optimizer_steps_applied": applied, "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer_state": optimizer.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "probabilities_calibrated": False}, RUN / f"recovery_step_{START + applied:05d}.pt")
        del keys, query, loss, real_loss, synthetic_loss, coherent_loss, real_nce, synthetic_nll, coherent_nce, real_scores, synthetic_scores, coherent_scores, coherent_keys, inputs, synthetic_inputs, key_images, real_key_images, synthetic_key_images, coherent_key_images, generated_inputs, rendered

assert applied == STEPS and all(int(v["step"]) == FINAL for v in optimizer.state_dict()["state"].values())
assert all(torch.isfinite(v).all() for v in model.state_dict().values())
assert all(torch.equal(v.cpu(), parent["model_state"][k]) for k, v in model.state_dict().items() if not k.startswith(TRAINABLE))
final = evaluate(STEPS)
gates = {}
a, b = initial["coherent"]["subsets"]["eligible"]["group_macro"], final["coherent"]["subsets"]["eligible"]["group_macro"]
gates["coherent_normal_improves_5deg"] = b["plane_angle_deg"] <= a["plane_angle_deg"]-5.
gates["coherent_capture128_improves_point10"] = b["capture128"] >= a["capture128"]+.10
for label, row in final["coherent"]["subsets"].items():
    if label.startswith("eligible:") and row["rows"]:
        baseline = initial["coherent"]["subsets"][label]
        assert row["rows"] == baseline["rows"] and row["groups"] == baseline["groups"]
        gates["coherent_mode_retention:" + label] = row["group_macro"]["capture128"] >= baseline["group_macro"]["capture128"]-.02
for dataset in ("synthetic", "real"):
    for label, row in final[dataset]["subsets"].items():
        selected = label == "all" if dataset == "real" else label == "eligible" or label.startswith("eligible_mode:")
        if selected and row["rows"]:
            baseline = initial[dataset]["subsets"][label]
            assert row["rows"] == baseline["rows"] and row["groups"] == baseline["groups"]
            a, b = baseline["group_macro"], row["group_macro"]
            gates[f"{dataset}_normal_retention:{label}"] = b["plane_angle_deg"] <= a["plane_angle_deg"]+2.
            gates[f"{dataset}_capture32_retention:{label}"] = b["capture32"] >= a["capture32"]-.02
for name, expected in source["file_sha256"].items(): assert sha(repository / name) == expected
completion = {"experiment": config, "source_unchanged": True, "frozen_nonretrieval_tensors_exact": True, "optimizer_steps_applied": FINAL, "additional_optimizer_steps_applied": STEPS, "gates": gates, "TRAIN_effectiveness_gate_passed": bool(all(gates.values())), "endpoints": {"0": initial, str(STEPS): final}, "elapsed_seconds": time.perf_counter()-started, "probabilities_calibrated": False, "scope": "TRAIN-only engineering effectiveness;not biological validation,native refinement,calibration or a qualified deliverable", "output_sha256": {str(p.relative_to(RUN)): sha(p) for p in RUN.rglob("*") if p.is_file() and "source" not in p.relative_to(RUN).parts}}
(RUN / "completed.json").write_text(json.dumps(completion, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"TRAIN_effectiveness_gate_passed": completion["TRAIN_effectiveness_gate_passed"], "gates": gates, "elapsed_seconds": completion["elapsed_seconds"]}), flush=True)
