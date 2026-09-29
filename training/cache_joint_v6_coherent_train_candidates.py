"""Frozen own-lineage coherent TRAIN top128 candidates; no optimizer or refinement."""
import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda")
sys.dont_write_bytecode = True
READY_AFTER_ROOT_REVIEW = True
assert READY_AFTER_ROOT_REVIEW, "Commit and review this fixed TRAIN-only inference cache before launch"

import hashlib
import json
import shutil
import subprocess
import time
import numpy as np
import torch
from training.arbitrary_plane_catalogue_runtime_v6 import make_complete_catalogue_runtime_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_physical_ouv
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6

COARSE = ROOT / "runs/joint_v6_full_coverage_ranking_001"
CHECKPOINT = COARSE / "A/joint_model_step_22576.pt"
GALLERY = COARSE / "A/gallery_descriptors_step_22576.npy"
AUDIT = ROOT / "runs/joint_v6_full_coverage_ranking_001_independent_audit/audit.json"
OUT = ROOT / "runs/joint_v6_coherent_train_candidates_001"
CHECKPOINT_SHA = "e22ff8e13a09b8c624c64518ca65a0ac9feda2c823dde30b6b8b5707995d52d8"
GALLERY_SHA = "5219117c897bf68fb6139b5bd482bfdd05f5c52e2b9a5a7296c9936d0f8d0b36"
AUDIT_SHA = "c75c4ff817e7603eca31c8086080aafdb7f0953b3fec58c3f3f2f7842cf2afb3"
CORPORA = (
    ("base", ROOT / "data/joint_v6_coherent_subject_cohort_sections_002", "ba51982a5b03b61d4bcf7f37f2c139dd6c1caf7ff66a6124cb678ab5ee9dd1f2", 512),
    ("acquisition", ROOT / "data/joint_v6_coherent_acquisition_views_001", "c0fb0763bb520adf3f372385ad2b1c5c52df6887db855de8a7f3ed99123bf5a5", 128),
)
MODES = ("raw", "exact_black", "imperfect_brush")
BATCH, KEEP, TEMPERATURE = 16, 128, .1
repository = Path(__file__).resolve().parents[1]
hashes = {}


def sha(path):
    with Path(path).open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    hashes[str(path)] = value
    return value


assert sha(AUDIT) == AUDIT_SHA
audit = json.loads(AUDIT.read_text())
assert audit["integrity_passed"] and all(audit["arm_audits"]["A"]["gates"].values())
assert sha(CHECKPOINT) == CHECKPOINT_SHA and sha(GALLERY) == GALLERY_SHA
parent = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
assert parent["step"] == 22576 and parent["phase"] == "experimental_image_key_proposal_only"
assert sha(COARSE / "catalogue.pt") == parent["experiment"]["prepared_source_sha256"]["catalogue.pt"]
catalogue = torch.load(COARSE / "catalogue.pt", map_location="cpu", weights_only=False)
selected_sections = []
for corpus, directory, expected, count in CORPORA:
    assert sha(directory / "completed.json") == expected
    completed = json.loads((directory / "completed.json").read_text())
    # Only selected TRAIN paths below are opened: base also contains128 held-out sections.
    records = [row for row in completed["sections"] if row["lineage"]["split"] == "train"]
    assert len(records) == count
    selected_sections.extend((corpus, directory, row) for row in records)
subjects = {row["lineage"]["subject_id"] for _, _, row in selected_sections}
assert len(selected_sections) == 640 and len(subjects) == 8

OUT.mkdir(parents=True, exist_ok=False)
shutil.copyfile(AUDIT, OUT / "coarse_independent_audit.json")
source_names = set(parent["experiment"]["source"]["file_sha256"]) | {
    "training/cache_joint_v6_coherent_train_candidates.py",
    "training/arbitrary_plane_joint_uncertainty.py",
    "training/subject_deformed_slab_multiresolution_bundle_v2.py",
    "docs/publication/COHERENT_TRAIN_CANDIDATES_001_PROTOCOL_20260929.md",
}
source = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(), "file_sha256": {}}
for name in sorted(source_names):
    path = OUT / "source" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository / name, path)
    source["file_sha256"][name] = sha(path)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"], device="cuda", dtype=torch.float32)
model = ArbitraryPlaneJointModelV6(runtime, **parent["experiment"]["model_kwargs"]).cuda().eval()
model.load_state_dict(parent["model_state"], strict=True)
assert all(torch.equal(value.cpu(), parent["model_state"][name]) for name, value in model.state_dict().items())
bank = torch.from_numpy(np.load(GALLERY, allow_pickle=False)).cuda()
assert tuple(bank.shape) == (98304, 2, 256) and bank.dtype == torch.float16
cell_mass = catalogue["tensors"]["cell_log_mass"][0].cuda().float()
representation_prior = catalogue["tensors"]["representation_log_weight"][0].cuda().float()
cell_ouv = full_frame_state_to_physical_ouv(torch.as_tensor(catalogue["arrays"]["cell_states_float64"])).numpy().reshape(-1, 3, 3)
cell_normal = np.asarray(catalogue["arrays"]["cell_normal_ap_dv_ml_float64"])
support_origin = np.asarray(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"])
cell_offset = np.sum((cell_ouv[:, 0] - support_origin) * cell_normal, axis=-1)
config = {"source": source, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": CHECKPOINT_SHA, "gallery": str(GALLERY), "gallery_sha256": GALLERY_SHA, "coarse_audit_sha256": AUDIT_SHA, "catalogue": str(COARSE / "catalogue.pt"), "catalogue_sha256": hashes[str(COARSE / "catalogue.pt")], "model_kwargs": parent["experiment"]["model_kwargs"], "catalogue_receipt_sha256": catalogue["receipt_sha256"], "batch_size": BATCH, "keep": KEEP, "temperature": TEMPERATURE, "precision": "FP32 scores, no AMP/TF32; matching saved FP16 gallery", "query": "unchanged saved image/optional-outline/availability only", "gallery_psf": parent["experiment"]["gallery"], "probabilities_calibrated": False, "optimizer_steps": 0, "native_refinement_calls": 0, "gallery_rebuilds": 0}
(OUT / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
del parent
pending, identities, sections = [], [], []
canonical, observed, reflection, offsets, weights, observation_section = [], [], [], [], [], []
saved = {key: [] for key in ("top128_cell_index", "top128_cell_log_probability", "top128_conditional_representation_log_probability", "top128_conditional_representation_probability", "top128_full_catalogue_component_log_probability", "query_descriptor", "retained_coarse_mass", "omitted_coarse_mass", "full_cell_lp_roundoff_log_normalizer")}
started = time.perf_counter()


def flush():
    inputs = torch.from_numpy(np.stack(pending)).cuda()
    query = model.pose_model.descriptor_from_features(model.pose_model.encode_histology(inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1))))
    score_r = model.pose_model.image_key_cosine_logits(query, bank, TEMPERATURE) + representation_prior[None]
    lp = (torch.logsumexp(score_r, dim=-1) + cell_mass[None]).log_softmax(dim=-1)
    assert bool(torch.isfinite(lp).all() and torch.isfinite(score_r).all())
    top = torch.argsort(lp, dim=-1, descending=True, stable=True)[:, :KEEP]
    top_lp = lp.gather(1, top)
    selected_score_r = score_r[torch.arange(len(inputs), device="cuda")[:, None], top]
    rlp = selected_score_r.log_softmax(dim=-1)
    # Preserve established FP32 LP/ranking; remove only measured global roundoff for masses.
    z64 = torch.logsumexp(lp.double(), dim=-1)
    retained = torch.exp(torch.logsumexp(top_lp.double(), dim=-1) - z64)
    omitted = -torch.expm1(torch.logsumexp(top_lp.double(), dim=-1) - z64)
    assert bool((z64.abs() < 2e-5).all() and (retained >= 0).all() and (omitted >= 0).all())
    values = (top, top_lp, rlp, selected_score_r.softmax(dim=-1), top_lp[..., None] + rlp, query, retained, omitted, z64)
    for key, value in zip(saved, values): saved[key].append(value.cpu().numpy())
    pending.clear()


with torch.inference_mode():
    for si, (corpus, directory, record) in enumerate(selected_sections):
        for name, expected in record["artifact_sha256"].items():
            assert sha(directory / name) == expected
        metadata_path = directory / record["artifacts"]["metadata"]
        arrays_path = directory / record["artifacts"]["arrays"]
        metadata = json.loads(metadata_path.read_text())
        assert metadata["lineage"] == record["lineage"] and metadata["lineage"]["split"] == "train"
        assert [item["selected_mode"] for item in metadata["observations"]] == list(MODES)
        with np.load(arrays_path, allow_pickle=False) as arrays:
            canonical.append(arrays[metadata["canonical_anatomy_plane_fit"]["arrays"]["physical_ouv_ap_dv_ml_um_float64"]["__ndarray__"]].reshape(3, 3))
            observed.append(arrays[metadata["observed_total_map_plane_fit"]["arrays"]["physical_ouv_ap_dv_ml_um_float64"]["__ndarray__"]].reshape(3, 3))
            reflection.append(arrays[metadata["reflection_xy"]["__ndarray__"]])
            offsets.append(arrays[metadata["axial_offsets_um_float64"]["__ndarray__"]])
            weights.append(arrays[metadata["axial_weights_float64"]["__ndarray__"]])
            for item in metadata["observations"]:
                key = item["image_outline_availability_float32"]["__ndarray__"]
                image = arrays[key]
                assert image.shape == (3, 96, 96) and image.dtype == np.float32 and np.isfinite(image).all()
                if item["selected_mode"] == "raw": assert np.count_nonzero(image[1:]) == 0
                identities.append({**item["lineage"], "corpus": corpus, "selected_mode": item["selected_mode"], "section_index_in_cache": si, "arrays_path": str(arrays_path), "input_array_key": key, "input_channels_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(), **{name: item[name] for name in ("outline_available", "support_information_eligible", "support_censored", "finite_support_mass_px", "visible_support_mass_px", "central_brain_pixel_count", "visible_central_brain_pixel_count")}})
                observation_section.append(si)
                pending.append(image)
                if len(pending) == BATCH: flush()
        sections.append({"corpus": corpus, "source_directory": str(directory), "record": record, **{key: metadata[key] for key in ("subject_reference", "source_identifiers", "paired_appearance_parameters", "plane_sampling", "preparation_seed")}, "acquisition_chart": metadata.get("acquisition_chart")})
        if (si + 1) % 64 == 0: print(json.dumps({"sections": si + 1, "observations": len(identities), "encoded": sum(len(x) for x in saved["top128_cell_index"]), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    if pending: flush()
assert len(identities) == 1920 and len({row["observation_id"] for row in identities}) == 1920
assert {row["subject_id"] for row in identities} == subjects
saved = {key: np.concatenate(value) for key, value in saved.items()}
section_index = np.asarray(observation_section)
canonical, observed, reflection = np.stack(canonical), np.stack(observed), np.stack(reflection)
np.savez_compressed(OUT / "candidates.npz", observation_section_index=section_index, **saved)
np.savez_compressed(OUT / "supervision_sidecar.npz", canonical_anatomy_fitted_ouv_ap_dv_ml_um=canonical, already_observed_total_map_fitted_ouv_ap_dv_ml_um=observed, reflection_xy=reflection, axial_offsets_um=np.stack(offsets), axial_weights=np.stack(weights))
(OUT / "observation_identities.json").write_text(json.dumps(identities, indent=2), encoding="utf-8")
(OUT / "section_provenance.json").write_text(json.dumps(sections, indent=2), encoding="utf-8")

# Descriptive frame-fit proxies only; none of these targets affected retrieval or ranking.
top = saved["top128_cell_index"]
truth, observed_truth = canonical[section_index], observed[section_index]
normal = np.cross(truth[:, 1], truth[:, 2]); normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
dot = np.sum(cell_normal[top] * normal[:, None], axis=-1)
truth_offset = np.sum((truth[:, 0] - support_origin) * normal, axis=-1)
offset_error = np.abs(truth_offset[:, None] - np.where(dot < 0, -1., 1.) * cell_offset[top])
angle = np.rad2deg(np.arctan2(np.sqrt(np.maximum(0, 1 - dot**2)), np.abs(dot)))
near = (angle <= 10.) & (offset_error <= 500.)
finite_center = observed_truth[:, 0] + .5 * (95 / 96) * (observed_truth[:, 1] + observed_truth[:, 2])
cell_center = cell_ouv[:, 0] + .5 * (95 / 96) * (cell_ouv[:, 1] + cell_ouv[:, 2])
frame_squared = np.sum((cell_center[top] - finite_center[:, None])**2, axis=-1)[..., None] + .25 * (95 / 96)**2 * (np.sum((cell_ouv[top, 1, None, :] * np.array([1., -1.])[None, None, :, None] - observed_truth[:, None, None, 1, :])**2, axis=-1) + np.sum((cell_ouv[top, 2] - observed_truth[:, None, 2])**2, axis=-1)[..., None])
predicted_r = saved["top128_conditional_representation_probability"][:, 0].argmax(axis=-1)
best = frame_squared.reshape(1920, -1).argmin(axis=-1)
metrics = {"top1_plane_angle_deg": angle[:, 0], "top1_normal_offset_error_um": offset_error[:, 0], "physical_plane_capture_at32": near[:, :32].any(axis=1), "physical_plane_capture_at128": near.any(axis=1), "top1_predicted_R_finite_four_corner_rms_um": np.sqrt(frame_squared[np.arange(1920), 0, predicted_r]), "oracle_best_top128x2_finite_four_corner_rms_um": np.sqrt(frame_squared.reshape(1920, -1)[np.arange(1920), best]), "retained_coarse_mass": saved["retained_coarse_mass"], "omitted_coarse_mass": saved["omitted_coarse_mass"]}
np.savez_compressed(OUT / "descriptive_readouts.npz", predicted_R_of_top_cell=predicted_r, oracle_best_frame_rank_zero_based=best // 2, oracle_best_frame_R=best % 2, **metrics)
eligible = np.array([row["support_information_eligible"] for row in identities])
corpus = np.array([row["corpus"] for row in identities]); mode = np.array([row["selected_mode"] for row in identities]); subject = np.array([row["subject_id"] for row in identities])
scopes = {"all": np.ones(1920, dtype=bool)}
scopes.update({f"{a}:{b}:{c}": (corpus == a) & (mode == b) & (np.ones(1920, dtype=bool) if c == "all" else eligible if c == "eligible" else ~eligible) for a in ("base", "acquisition") for b in MODES for c in ("all", "eligible", "censored")})
summaries = {}
for label, mask in scopes.items():
    groups = [mask & (subject == name) for name in np.unique(subject[mask])]
    summaries[label] = {"observations": int(mask.sum()), "subjects": len(groups), "eligible": int((mask & eligible).sum()), "censored": int((mask & ~eligible).sum()), "subject_macro": {key: float(np.mean([value[g].mean() for g in groups])) if groups else None for key, value in metrics.items()}, "row_p50_p90_p95": {key: np.quantile(value[mask], [.5, .9, .95]).tolist() if mask.any() else None for key, value in metrics.items() if value.dtype != bool}}
for name, expected in source["file_sha256"].items():
    with (repository / name).open("rb") as stream: assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
summary = {"experiment": config, "section_count": 640, "observation_count": 1920, "synthetic_subject_count": 8, "input_sha256": dict(hashes), "source_unchanged": True, "summaries": summaries, "full_cell_lp_roundoff_max_abs_log_normalizer": float(np.abs(saved["full_cell_lp_roundoff_log_normalizer"]).max()), "posterior_semantics": "Top128 LP and componentLP come from all98304 cells, not beam normalization. ComponentLP=cellLP+conditionalRLP; R probability is conditional within each cell only. Retained/omitted mass uses FP64 summation and the saved full-cell FP32 roundoff normalizer; their sum is1. No probabilities are calibrated.", "geometry_scope": "Canonical anatomy fitted frame remains canonical/unreflected; observed total-map fit is already reflected/processed and is not flipped again. Readouts are frame proxies, not dense curved-target accuracy. Oracle best-frame rank/R are readout-only, never candidate selection.", "scope": "Same whole audited A22576 lineage, matching gallery, TRAIN1920 only; no target/query leakage, optimizer, native refinement, rerender, new biological subjects or model qualification. Cache is invalid after encoder/gallery contract changes; cached descriptors are reconstruction evidence, not feature supervision. Query PSF was not supplied; fixed50um gallery differs from each observation's true stored25–100um PSF.", "elapsed_seconds": time.perf_counter() - started, "output_sha256": {path.name: sha(path) for path in OUT.iterdir() if path.is_file()}}
(OUT / "completed.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"observations": 1920, "sections": 640, "subjects": 8, "elapsed_seconds": summary["elapsed_seconds"], "output": str(OUT)}), flush=True)
