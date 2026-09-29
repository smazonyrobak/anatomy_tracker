"""Train-only exact-anchor versus frozen-gallery closure, whole A6000 and failed8000."""
import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda")
sys.dont_write_bytecode = True

import hashlib
import json
import shutil
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_catalogue_runtime_v6 import make_complete_catalogue_runtime_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components, full_frame_state_to_physical_ouv, render_finite_thickness_plane
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6

FINAL = ROOT / "runs/joint_v6_imagekey_real_synthetic_001"
PARENT = ROOT / "runs/joint_v6_imagekey_outline_dropout_001/A"
OUT = ROOT / "runs/joint_v6_real_training_retrieval_001"
ENDPOINTS = (("A6000", PARENT, 6000,
    "280836b65fb6db22930c8ee268eb4a880997c7858fb91a1e31ad6c7c94937eb2",
    "2e459cc5075ede84e8e87471186b635f3ae7a3e882732e4d1192f82ced9983d4"),
    ("failed8000", FINAL, 8000,
    "3c689da1a615d637048f7890e0e1e1644e9976c3f9c8447991da097f2ca7186e",
    "6bcab0980f3c332f64d179f9e7a0c53a0cea76bc750ca4e191ad2fa4d1ce81c6"))
bindings = {
    FINAL / "experiment.json": "89f1b401d2f74f36d3ba2a723672e4f07fe9af6c0c3db2f13e8d9b844ab32057",
    FINAL / "completed.json": "a1cd0323d9456210e64b4e622ab3375524dc2f06810b6bdd43c3e0aa49c1be31",
    FINAL / "catalogue.pt": "9b49d203cc73ce3a66e648bbe5228231eb5cc9c17d5db4669eefe0f08ae22c71",
    FINAL / "weak_affine_anchors.npz": "ba886f8f1a1344f9921a76aec660cef70c6dce1dfe8c98c5af0c212b334803e7",
    FINAL / "real_training_model_input.npy": "b4c0bc908ca4b61a4000bbaa02b140e440c3b4da42069b6b793e435c64f38379",
    FINAL / "real_training_identities.json": "f18c3fe6434de01f84ca39d697e36db02cbc2986e58c9c5655d4e92afcf7ec0e",
    ROOT / "runs/joint_v6_imagekey_real_synthetic_001_independent_audit/audit.json": "579b3acdbb1cb7a62ea339508cbe4091637965d7f81efcc4333d02497bdee48b",
    ROOT / "runs/joint_v6_imagekey_outline_dropout_001_independent_audit/audit.json": "2db902bdc6a2e502c911fa3ced308ed96d27219926356457efa0e063d53bd2e0"}
for _, directory, step, checkpoint_hash, gallery_hash in ENDPOINTS:
    bindings[directory / f"joint_model_step_{step:05d}.pt"] = checkpoint_hash
    bindings[directory / f"gallery_descriptors_step_{step:05d}.npy"] = gallery_hash


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


for path, expected in bindings.items():
    assert sha(path) == expected, path
    if path.name == "audit.json":
        assert json.loads(path.read_text())["integrity_passed"]
config = json.loads((FINAL / "experiment.json").read_text())
repository = Path(__file__).resolve().parents[1]
for name, expected in config["source"]["file_sha256"].items():
    if not Path(name).name.startswith("run_"):
        assert sha(repository / name) == expected, name
        bindings[repository / name] = expected
records = json.loads((FINAL / "real_training_identities.json").read_text())
images = np.load(FINAL / "real_training_model_input.npy", allow_pickle=False)
donors = np.array([row["animal_id"] for row in records])
assert images.shape == (256, 1, 96, 96) and len(np.unique(donors)) == 58
assert all(row["split"] == "development_train" and row["array_row_index"] == i for i, row in enumerate(records))
assert not set(map(str, donors)) & {"14452", "15219", "15336", "15439", "15447", "15935"}
with np.load(FINAL / "weak_affine_anchors.npz", allow_pickle=False) as saved:
    anchor_ouv = saved["physical_ouv_ap_dv_ml_um"]
    anchor_states = torch.from_numpy(saved["full_frame_state"])
affines = np.array([row["model_pixel_to_ap_dv_ml_um"] for row in records])
assert np.array_equal(anchor_ouv, np.stack((affines[:, :, 2], 96 * affines[:, :, 0], 96 * affines[:, :, 1]), axis=1))
catalogue = torch.load(FINAL / "catalogue.pt", weights_only=False, map_location="cpu")
cell_states = torch.as_tensor(catalogue["arrays"]["cell_states_float64"])
cell_ouv = full_frame_state_to_physical_ouv(cell_states).numpy().reshape(-1, 3, 3)
anchor_centre, anchor_frame, _ = full_frame_state_to_components(anchor_states)
cell_centre, cell_frame, _ = full_frame_state_to_components(cell_states)
anchor_centre, cell_centre = anchor_centre.numpy(), cell_centre.numpy()
anchor_normal, cell_normal = anchor_frame[:, :, 2].numpy(), cell_frame[:, :, 2].numpy()
support_origin = np.array(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"])

# Finite corners are pixel centres 0 and 95; horizontal correspondence reverses U only.
all_ouv = np.concatenate((anchor_ouv, cell_ouv))
all_u, all_v = all_ouv[:, 1] * (95 / 96), all_ouv[:, 2] * (95 / 96)
all_centre = all_ouv[:, 0] + .5 * (all_u + all_v)
all_normal = np.concatenate((anchor_normal, cell_normal))
all_offset = ((np.concatenate((anchor_centre, cell_centre)) - support_origin) * all_normal).sum(-1)
near_anchor = np.empty((256, 256), bool)
near_cell = np.empty((256, len(cell_states)), bool)
plane_cell = np.empty_like(near_cell)
nearest = np.empty(256, np.int64)
geometry_rows = {key: np.empty(256) for key in ("nearest_frame_rms_um", "nearest_frame_normal_deg", "nearest_frame_offset_um", "plane_eligible_min_frame_rms_um")}
for i in range(256):
    dot = all_normal @ anchor_normal[i]
    angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(all_normal, anchor_normal[i]), axis=-1), np.abs(dot)))
    offset = np.abs(all_offset - np.where(dot < 0, -1., 1.) * ((anchor_centre[i] - support_origin) @ anchor_normal[i]))
    rms = np.sqrt(((all_centre - all_centre[i]) ** 2).sum(-1) + .25 * np.minimum(((all_u - all_u[i]) ** 2).sum(-1), ((all_u + all_u[i]) ** 2).sum(-1)) + .25 * ((all_v - all_v[i]) ** 2).sum(-1))
    equivalent = (angle <= 10.) & (rms <= 1000.)
    near_anchor[i], near_cell[i] = equivalent[:256], equivalent[256:]
    near_anchor[i, i] = True
    plane_cell[i] = (angle[256:] <= 10.) & (offset[256:] <= 500.)
    nearest[i] = j = int(rms[256:].argmin())
    geometry_rows["nearest_frame_rms_um"][i] = rms[256 + j]
    geometry_rows["nearest_frame_normal_deg"][i] = angle[256 + j]
    geometry_rows["nearest_frame_offset_um"][i] = offset[256 + j]
    geometry_rows["plane_eligible_min_frame_rms_um"][i] = rms[256:][plane_cell[i]].min() if plane_cell[i].any() else np.nan
geometry_rows.update(nearest_frame_cell_index=nearest, near_equivalent_anchor_count=near_anchor.sum(-1), near_equivalent_cell_count=near_cell.sum(-1), plane_eligible_cell_count=plane_cell.sum(-1))
del all_ouv, all_u, all_v, all_centre, all_normal, all_offset

OUT.mkdir(parents=True, exist_ok=False)
shutil.copyfile(__file__, OUT / "diagnostic_source.py")
shutil.copyfile(FINAL / "real_training_identities.json", OUT / "identities.json")
np.savez(OUT / "geometry.npz", anchor_ouv_ap_dv_ml_um=anchor_ouv, anchor_state=anchor_states.numpy(), reference_centre_ap_dv_ml_um=anchor_centre, reference_normal_ap_dv_ml=anchor_normal, **geometry_rows)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
started = time.perf_counter()
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation
origin, spacing = [catalogue["support_geometry"][key] for key in ("origin_ap_dv_ml_um", "voxel_size_ap_dv_ml_um")]
z = torch.linspace(-.5, .5, 9, device="cuda") * 50.
w = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device="cuda", dtype=torch.float32) / 16
with torch.no_grad():
    anchors = torch.cat([render_finite_thickness_plane(atlas, part.cuda().float(), (96, 96), origin, spacing, z, w).cpu() for part in anchor_states.split(16)])
del atlas
np.save(OUT / "anchor_rendered_channels.npy", anchors.numpy())
mass = anchors[:, 1].sum((-2, -1)).numpy()
strata = {"all": np.ones(256, bool), "zero_support": mass == 0, "positive_support_below64": (mass > 0) & (mass < 64), "support_at_least64": mass >= 64}
previews = np.concatenate((np.flatnonzero(mass < 64), np.flatnonzero(mass >= 64)[:12]))
sheet = Image.new("RGB", (550, 26 + 112 * len(previews)), "white")
draw = ImageDraw.Draw(sheet)
draw.text((4, 5), "row / donor / support                 query        atlas intensity       support", fill="black")
for k, i in enumerate(previews):
    y = 26 + 112 * k
    draw.text((4, y + 6), f"row {i} donor {donors[i]}\nsupport {mass[i]:.5g}", fill="black")
    for column, values in enumerate((images[i, 0], anchors[i, 0].numpy(), anchors[i, 1].numpy())):
        sheet.paste(Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255)).convert("RGB"), (220 + 108 * column, y))
sheet.save(OUT / "fixed_query_anchor_previews.png")
np.savez(OUT / "anchor_input_diagnostics.npz", support_mass=mass, query_nonzero_pixels=np.count_nonzero(images[:, 0], axis=(-2, -1)), anchor_intensity_nonzero_pixels=np.count_nonzero(anchors[:, 0].numpy(), axis=(-2, -1)), anchor_intensity_sum=anchors[:, 0].sum((-2, -1)).numpy(), preview_row_index=previews)
print(json.dumps({"rendered_anchors": 256, "zero_support_rows": np.flatnonzero(mass == 0).tolist(), "low_positive_support_rows": np.flatnonzero((mass > 0) & (mass < 64)).tolist()}), flush=True)

runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=config["catalogue_receipt_sha256"], device="cuda", dtype=torch.float32)
assert runtime.cell_count == 98304 and runtime.representation_count == 2
cell_mass = catalogue["tensors"]["cell_log_mass"][0].cuda().float()
representation_prior = catalogue["tensors"]["representation_log_weight"][0].cuda().float()
np.savez(OUT / "catalogue_priors.npz", cell_log_mass=cell_mass.cpu().numpy(), representation_log_weight=representation_prior.cpu().numpy())
summaries = {}
for label, directory, step, _, _ in ENDPOINTS:
    destination = OUT / label
    destination.mkdir()
    saved = torch.load(directory / f"joint_model_step_{step:05d}.pt", weights_only=False, map_location="cpu", mmap=True)
    assert saved["step"] == saved["optimizer_steps_applied"] == step
    assert saved["experiment"]["model_kwargs"] == config["model_kwargs"]
    model = ArbitraryPlaneJointModelV6(runtime, **config["model_kwargs"]).cuda().eval()
    model.load_state_dict(saved["model_state"], strict=True)
    del saved
    bank = torch.from_numpy(np.load(directory / f"gallery_descriptors_step_{step:05d}.npy", allow_pickle=False)).cuda()
    with torch.no_grad():
        query = torch.cat([model.pose_model.descriptor_from_features(model.pose_model.encode_histology(part.cuda(), torch.zeros_like(part).cuda(), torch.zeros(len(part), device="cuda"))) for part in torch.from_numpy(images).split(16)])
        keys = torch.cat([model.pose_model.descriptor_from_features(model.pose_model._encode_atlas(torch.stack((part, part.flip(-1)), dim=1).flatten(0, 1).cuda())).reshape(-1, 2, 256) for part in anchors.split(8)])
        anchor_components = model.pose_model.image_key_cosine_logits(query, keys, .1)
        anchor_scores = torch.logsumexp(anchor_components - np.log(2.), dim=-1).cpu().numpy()
        assert bool(torch.isfinite(anchor_components).all()) and np.isfinite(anchor_scores).all()
    np.savez(destination / "descriptor_reconstruction.npz", query_descriptor=query.cpu().numpy(), exact_anchor_descriptor=keys.cpu().numpy(), anchor_raw_component_logit=anchor_components.cpu().numpy(), anchor_uniform_representation_logit=anchor_scores)
    raw = np.lib.format.open_memmap(destination / "full_catalogue_log_probability.npy", mode="w+", dtype=np.float32, shape=(256, runtime.cell_count))
    metrics = {name: np.full(256, np.nan) for name in ("own_anchor_rank", "own_anchor_incompatible_only_rank", "near_anchor_best_rank", "own_anchor_margin_vs_incompatible", "own_anchor_logit", "near_anchor_best_logit", "nearest_frame_cell_rank", "best_plane_cell_rank", "best_near_frame_cell_rank", "best_plane_logit_margin_vs_incompatible", "own_anchor_minus_best_catalogue_appearance_logit", "nearest_frame_cell_appearance_logit", "plane_angle_deg", "normal_offset_error_um", "physical_plane_capture_at_1", "physical_plane_capture_at_32", "physical_plane_capture_at_128")}
    top128 = np.empty((256, 128), np.int64)
    for start in range(0, 256, 16):
        with torch.no_grad():
            component = model.pose_model.image_key_cosine_logits(query[start:start + 16], bank, .1)
            lp = (torch.logsumexp(component + representation_prior[None], dim=-1) + cell_mass[None]).log_softmax(-1)
            assert bool(torch.isfinite(lp).all())
            raw[start:start + 16] = lp.cpu().numpy()
            top128[start:start + 16] = torch.argsort(lp, descending=True, stable=True)[:, :128].cpu().numpy()
            appearance = torch.logsumexp(component - np.log(2.), dim=-1).cpu().numpy()
        for b, i in enumerate(range(start, start + 16)):
            a = anchor_scores[i]
            anchor_order = np.argsort(-a, kind="stable")
            outranks_own = (a > a[i]) | ((a == a[i]) & (np.arange(256) < i))
            metrics["own_anchor_rank"][i] = 1 + outranks_own.sum()
            metrics["own_anchor_incompatible_only_rank"][i] = 1 + (outranks_own & ~near_anchor[i]).sum()
            metrics["near_anchor_best_rank"][i] = 1 + np.flatnonzero(near_anchor[i, anchor_order])[0]
            metrics["own_anchor_margin_vs_incompatible"][i] = a[i] - a[~near_anchor[i]].max() if (~near_anchor[i]).any() else np.nan
            metrics["own_anchor_logit"][i] = a[i]
            metrics["near_anchor_best_logit"][i] = a[near_anchor[i]].max()
            cell_order = np.argsort(-raw[i], kind="stable")
            metrics["nearest_frame_cell_rank"][i] = 1 + np.flatnonzero(cell_order == nearest[i])[0]
            for mask, metric in ((plane_cell[i], "best_plane_cell_rank"), (near_cell[i], "best_near_frame_cell_rank")):
                if mask.any():
                    metrics[metric][i] = 1 + np.flatnonzero(mask[cell_order])[0]
            if plane_cell[i].any() and (~plane_cell[i]).any():
                metrics["best_plane_logit_margin_vs_incompatible"][i] = appearance[b, plane_cell[i]].max() - appearance[b, ~plane_cell[i]].max()
            metrics["own_anchor_minus_best_catalogue_appearance_logit"][i] = a[i] - appearance[b].max()
            metrics["nearest_frame_cell_appearance_logit"][i] = appearance[b, nearest[i]]
            j = top128[i, 0]
            dot = cell_normal[j] @ anchor_normal[i]
            metrics["plane_angle_deg"][i] = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(cell_normal[j], anchor_normal[i])), abs(dot)))
            metrics["normal_offset_error_um"][i] = abs((cell_centre[j] - support_origin) @ cell_normal[j] - (-1 if dot < 0 else 1) * ((anchor_centre[i] - support_origin) @ anchor_normal[i]))
            for k in (1, 32, 128):
                metrics[f"physical_plane_capture_at_{k}"][i] = plane_cell[i, top128[i, :k]].any()
        print(json.dumps({"endpoint": label, "queries_scored": start + 16}), flush=True)
    raw.flush()
    normalization_error = max(float(np.abs(np.logaddexp.reduce(np.asarray(raw[i:i + 16], dtype=np.float64), axis=-1)).max()) for i in range(0, 256, 16))
    assert normalization_error < 2e-5
    metrics.update({f"own_anchor_hit_at_{k}": (metrics["own_anchor_rank"] <= k).astype(float) for k in (1, 8, 32)})
    metrics.update({f"near_anchor_hit_at_{k}": (metrics["near_anchor_best_rank"] <= k).astype(float) for k in (1, 8, 32)})
    np.savez(destination / "rows.npz", animal_id=donors, row_index=np.arange(256), top128_cell_index=top128, support_mass=mass, **metrics)
    subsets = {}
    for name, selected in strata.items():
        by_donor = {}
        for donor in np.unique(donors[selected]):
            included = selected & (donors == donor)
            by_donor[str(donor)] = {"rows": int(included.sum()), "mean": {}, "finite_rows": {}}
            for key, values in {**geometry_rows, **metrics}.items():
                if key.endswith("index"):
                    continue
                finite = included & np.isfinite(values)
                by_donor[str(donor)]["mean"][key] = float(values[finite].mean()) if finite.any() else None
                by_donor[str(donor)]["finite_rows"][key] = int(finite.sum())
        macro = {}
        for key in next(iter(by_donor.values()))["mean"] if by_donor else ():
            values = [row["mean"][key] for row in by_donor.values() if row["mean"][key] is not None]
            macro[key] = float(np.mean(values)) if values else None
        subsets[name] = {"rows": int(selected.sum()), "donors": len(by_donor), "by_donor": by_donor, "donor_macro": macro}
    summaries[label] = {"step": step, "subsets": subsets, "maximum_log_normalization_error": normalization_error}
    (destination / "summary.json").write_text(json.dumps(summaries[label], indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"endpoint": label, "all_training_donor_macro": subsets["all"]["donor_macro"]}), flush=True)
    del raw, model, bank, query, keys, anchor_components
    torch.cuda.empty_cache()

summary = {"scope": "all256 existing training images/58donors only; weak upstream-affine reference, not biological accuracy, validation, calibration or promotion", "gallery_policy": "reuse two frozen98304x2 descriptor banks unchanged; render256 exact-affine anchors once, encode separately under each whole checkpoint", "query_policy": "exact saved red96x96/12mm pixels; zero outline and availability; no filtering, tuning or development rows", "anchor_psf": "recorded50um nine-point [1,2,2,2,2,2,2,2,1]/16, not a verified optical PSF", "near_equivalence": "antipodal normal<=10deg AND finite corner RMS<=1000um; horizontal correspondence only, pixel0..95", "plane_capture": "antipodal normal<=10deg AND sign-aligned normal offset<=500um; no exclusions in full-gallery posterior/ranks/capture", "scores": "native full-cell logposterior applies representation priors and cell mass once; cross-space appearance margins use uniform-R logmeanexp and no cell mass; exact-anchor rank has no cell mass", "tie_policy": "lower index first", "missing_candidates": "NaN in row arrays, null in summaries with finite-row counts; no fabricated positive", "preview_policy": "all support<64 plus first12 remaining rows, fixed0..1 grayscale, no content normalization", "probabilities_calibrated": False, "elapsed_seconds": time.perf_counter() - started, "input_sha256": {str(path): value for path, value in bindings.items()}, "results": summaries, "output_sha256": {str(path.relative_to(OUT)): sha(path) for path in OUT.rglob("*") if path.is_file()}}
(OUT / "completed.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"completed": str(OUT), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
