"""Sixteen training planes from one frozen 3D subject; CPU-only preparation.

Paired appearances share anatomy, appearance and noise. These coordinate targets
are deliberately NOT the legacy total-2D-SVF training-pack format.
"""

import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
sys.dont_write_bytecode = True

import hashlib
import json
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_coherent_subject_v6 import make_coherent_subject_section_v6
from training.arbitrary_plane_geometry import physical_um_to_allen_index_points
from training.arbitrary_plane_subject_deformation_v2 import subject_deformation_plan_receipt_v2
from training.arbitrary_plane_subject_sampling_v6 import sample_subject_planes
from training.arbitrary_plane_subject_section_v2 import sample_coordinate_rasters_v2
from training.arbitrary_plane_synthetic_generator_v2 import reduce_v2_slab_samples
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact, _write_raw_artifact

PLAN_ROOT = ROOT / "data/joint_v6_coherent_subject_pilot_001"
OUTPUT = ROOT / "data/joint_v6_coherent_subject_sections_001"
SEED, COUNT, SHAPE = 2026092908, 16, (96, 96)
SUPPORT_MASS_THRESHOLD = 64.0
MODES = ("raw", "exact_black", "imperfect_brush")
CHECK_SHA256 = "cce2057428b2caed178a6320cd9f79f9bcaab2003308dfcb6630b003e73644b9"
PLAN_RECEIPT_SHA256 = "496669aacf08f9a5cee3aedc4fcf3a600cfe9ced106fd1358c3d4bef2cf8c7bc"

with (PLAN_ROOT / "check_result.json").open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == CHECK_SHA256
pilot = json.loads((PLAN_ROOT / "check_result.json").read_text())
plan_files = ("subject_plan.metadata.json", "subject_plan.arrays.npz", "subject_plan_receipt.json", "context.json", "protocol.json")
for name in plan_files:
    with (PLAN_ROOT / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == pilot["file_sha256"][name], name
plan = _read_raw_artifact(PLAN_ROOT, pilot["plan_files"])
assert subject_deformation_plan_receipt_v2(plan)["receipt_sha256"] == PLAN_RECEIPT_SHA256 == plan["receipt_sha256"]
assert plan["provenance"]["split"] == "train" and plan["resolved_config"]["deformation_stratum"] == "standard"
assert plan["provenance"]["ccf_context_sha256"] == pilot["context_sha256"]
assert np.array_equal(plan["state"]["full_ccf_lower_um"], allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6)
assert np.array_equal(plan["state"]["full_ccf_upper_um"], np.asarray(allen.ATLAS_SHAPE_AP_DV_ML_V6) * allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6)
repository = Path(__file__).resolve().parents[1]
for name, expected in plan["source_sha256"].items():
    text = (repository / "training" / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    assert hashlib.sha256(text).hexdigest() == expected, name
OUTPUT.mkdir(parents=True, exist_ok=False)
for name in (*plan_files, "check_result.json"):
    shutil.copyfile(PLAN_ROOT / name, OUTPUT / f"parent_{name}")
source_names = (
    "prepare_joint_v6_coherent_subject_sections.py", "arbitrary_plane_coherent_subject_v6.py",
    "arbitrary_plane_subject_sampling_v6.py", "arbitrary_plane_subject_deformation_v2.py",
    "arbitrary_plane_subject_section_v2.py", "arbitrary_plane_full_frame_primitives.py",
    "arbitrary_plane_geometry.py", "arbitrary_plane_acquisition_v2.py",
    "arbitrary_plane_allen_atlas_binding_v6.py", "arbitrary_plane_synthetic_generator_v2.py",
    "subject_deformed_slab_multiresolution_bundle_v2.py",
)
source = {
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
    "file_sha256": {f"training/{name}": hashlib.sha256((repository / "training" / name).read_bytes()).hexdigest() for name in source_names},
}
(OUTPUT / "preparation_source.py").write_bytes(Path(__file__).read_bytes())
(OUTPUT / "source_diff.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=repository))
atlas_identity = {
    "atlas_id": "Allen CCFv3", "version": "2017 25um", "coordinate_axes": ["AP", "DV", "ML"],
    "template_path": str(allen.TEMPLATE_PATH_V6), "template_raw_sha256": allen.TEMPLATE_RAW_SHA256_V6,
    "annotation_path": str(allen.ANNOTATION_PATH_V6), "annotation_raw_sha256": allen.ANNOTATION_RAW_SHA256_V6,
    "render_volume_receipt": allen.ATLAS_FLOAT32_RECEIPT_V6,
    "annotation_decoded_receipt": allen.ANNOTATION_DECODED_RECEIPT_V6,
    "origin_um": allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6, "voxel_size_um": allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6,
    "intensity": "v6 pinned within-brain normalization with exact atlas exterior zero; distinct from pilot v2 scalar preprocessing",
}
protocol = {
    "source": source, "root_seed": SEED, "section_count": COUNT, "observations_per_section": 3,
    "subject_count": 1, "split": "train", "raster_shape_h_w": SHAPE,
    "plan_source_directory": str(PLAN_ROOT), "plan_check_sha256": CHECK_SHA256,
    "plan_file_sha256": {name: pilot["file_sha256"][name] for name in plan_files},
    "subject_plan_receipt_sha256": PLAN_RECEIPT_SHA256, "subject_plan_id": pilot["subject_plan_id"],
    "subject_realization_id": pilot["subject_realization_id"], "synthetic_animal_id": pilot["synthetic_animal_id"],
    "plan_context_sha256": pilot["context_sha256"], "atlas": atlas_identity,
    "authentication": "pinned completed accepted plan files and recomputed content receipt; no full sampler replay",
    "sampling": "independent uniform RP2 normal/roll and subject-box-intersecting finite-slab offsets; conservative full subject box, no tissue rejection",
    "physical_fov": "per-plane conservative subject-box projection spans; not fixed12mm, do not infer fixed125um pixel size",
    "rng": "independent NumPy PCG64 SeedSequence([root_seed,0,section]) geometry and [root_seed,1,section] appearance; Torch CPU noise seed=root_seed*1000000+section",
    "psf": "each section25-100um;9 positions linspace(-.5,.5)*thickness, integer masses[1,2,2,2,2,2,2,2,1] normalized once",
    "processing": "identity2D pullback in observed raster, then recorded horizontal reflection, then subject3D map; not zero total anatomical deformation",
    "modes": MODES, "pairing": "same physical section, reflection, gamma/gain/inversion, noise and synthetic background before optional brush masking",
    "annotations": "nearest ties-to-even at newly mapped physical PSF coordinates; lower-ID modal ties; zero outside atlas; weights include background label0",
    "support": "trilinear finite PSF support distinct from nearest central-brain mask and nearest PSF occupancy; all recomputed from this subject's coordinates",
    "censoring": "retain all16 draws and all48 mode observations; information eligibility requires finite and visible support mass>=64, not a finalized pose-target rule",
    "target_kind": "coherent_subject_3d_coordinate_targets_only",
    "legacy_total_2d_svf_pack_compatible": False, "catalogue_cell_truth": None,
    "label_limitation": "canonical full-canvas3D plane fit/residual saved, but no chosen catalogue label or total2D stationary velocity exists in this pack",
    "scope": "one generated anatomical subject, not an independent biological animal, held-out split, benchmark or model-accuracy result",
}
(OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
torch.set_num_threads(4)
started = time.perf_counter()
print("Decoding pinned v6 intensity/support atlas and annotation; CPU only", flush=True)
atlas_array, annotation_array = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array)
annotation = torch.from_numpy(annotation_array.astype(np.int64))
del annotation_array
identity_yx = np.stack(np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing="ij")).astype(np.float64)
yy, xx = torch.meshgrid(torch.linspace(-1, 1, SHAPE[0]), torch.linspace(-1, 1, SHAPE[1]), indexing="ij")
integer_masses = np.array([1, 2, 2, 2, 2, 2, 2, 2, 1], dtype=np.int64)
psf_weights = integer_masses.astype(np.float64) / integer_masses.sum()
records = []
for index in range(COUNT):
    tick = time.perf_counter()
    geometry_rng = np.random.default_rng(np.random.SeedSequence([SEED, 0, index]))
    appearance_rng = np.random.default_rng(np.random.SeedSequence([SEED, 1, index]))
    thickness = float(geometry_rng.uniform(25, 100))
    offsets = np.linspace(-0.5, 0.5, 9) * thickness
    reflection = (bool(geometry_rng.integers(0, 2)), False)
    draw = sample_subject_planes(plan, geometry_rng, 1, SHAPE, offsets)
    section_id = f"joint-v6-coherent-sections-001-section-{index:08d}"
    lineage = {**pilot["lineage"], "section_id": section_id}
    section = make_coherent_subject_section_v6(
        plan, draw["physical_ouv_ap_dv_ml_um"][0], identity_yx, reflection, offsets, psf_weights,
        lineage, atlas_identity, atlas, allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6,
        allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6,
    )
    rendered = section.pop("raw_rendered_channels")
    finite_support = rendered[1].clamp(0, 1)
    tissue = finite_support > 0
    allen_coordinates = physical_um_to_allen_index_points(
        torch.from_numpy(section["target_psf_ccf_coordinates_ap_dv_ml_um_float64"]).float(),
        allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6, allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6,
    )
    scalar_samples, label_samples = sample_coordinate_rasters_v2(atlas[0], annotation, allen_coordinates)
    slab = reduce_v2_slab_samples(scalar_samples.numpy(), label_samples.numpy(), integer_masses, section["centre_psf_index"])
    slab.pop("scalar")  # Appearance uses the adapter's continuous physical render.
    section["nearest_psf_annotation_int64"] = label_samples.numpy()
    section["annotation_supervision"] = slab
    section["finite_psf_trilinear_support_float32"] = finite_support.numpy()
    section["raw_rendered_channels_float32"] = rendered.numpy()
    section["plane_sampling"] = draw
    section["preparation_seed"] = {"root_seed": SEED, "geometry_seed_sequence": [SEED, 0, index], "appearance_seed_sequence": [SEED, 1, index]}
    section["target_kind"] = "coherent_subject_3d_coordinate_targets_only"
    section["legacy_total_2d_svf_pack_compatible"] = False
    section["catalogue_cell_truth"] = None
    parameters = {
        "gain": float(appearance_rng.uniform(.6, 1.4)),
        "gamma": float(np.exp(appearance_rng.uniform(np.log(.6), np.log(1.6)))),
        "invert_tissue": bool(appearance_rng.integers(0, 2)),
        "noise_std": float(appearance_rng.uniform(.005, .05)),
        "background_mean": float(appearance_rng.uniform(0, .8)),
        "background_slope_yx": appearance_rng.uniform(-.15, .15, 2).tolist(),
        "imperfect_mask_dilate": bool(appearance_rng.integers(0, 2)),
        "imperfect_mask_radius_px": int(appearance_rng.integers(1, 4)),
        "noise_seed": SEED * 1_000_000 + index, "noise_device": "CPU",
    }
    noise_generator = torch.Generator().manual_seed(parameters["noise_seed"])
    appearance_noise = torch.randn(SHAPE, generator=noise_generator)
    appearance = (rendered[0] / finite_support.clamp_min(1e-6)).clamp(0, 1).pow(parameters["gamma"])
    if parameters["invert_tissue"]:
        appearance = 1 - appearance
    appearance = (appearance * parameters["gain"]).clamp(0, 1)
    slope = parameters["background_slope_yx"]
    background = parameters["background_mean"] + slope[0] * yy + slope[1] * xx
    before_brush = (appearance * finite_support + background * (1 - finite_support) + parameters["noise_std"] * appearance_noise).clamp(0, 1)
    section["paired_appearance_parameters"] = parameters
    section["shared_pre_brush_image_float32"] = before_brush.numpy()
    section["observations"] = []
    for mode_index, mode in enumerate(MODES):
        mask = torch.ones(SHAPE, dtype=torch.bool) if mode_index == 0 else tissue.clone()
        if mode_index == 2:
            radius = parameters["imperfect_mask_radius_px"]
            if parameters["imperfect_mask_dilate"]:
                mask = F.max_pool2d(mask[None, None].float(), 2 * radius + 1, 1, radius)[0, 0] > 0
            else:
                padded = F.pad(mask[None, None].float(), (radius,) * 4, value=0)
                mask = -F.max_pool2d(-padded, 2 * radius + 1, 1)[0, 0] > 0
        outline = torch.zeros_like(mask)
        if mode_index:
            eroded = mask.clone()
            eroded[1:] &= mask[:-1]
            eroded[:-1] &= mask[1:]
            eroded[:, 1:] &= mask[:, :-1]
            eroded[:, :-1] &= mask[:, 1:]
            eroded[[0, -1], :] = False
            eroded[:, [0, -1]] = False
            outline = mask & ~eroded
        finite_mass, visible_mass = float(finite_support.sum()), float((finite_support * mask).sum())
        eligible = finite_mass >= SUPPORT_MASS_THRESHOLD and visible_mass >= SUPPORT_MASS_THRESHOLD
        section["observations"].append({
            "lineage": {**lineage, "observation_id": f"{section_id}-{mode}"}, "selected_mode": mode,
            "image_outline_availability_float32": torch.stack((before_brush * mask, outline.float(), torch.full_like(before_brush, float(mode_index != 0)))).numpy(),
            "brush_keep_mask": mask.numpy(), "outline_available": mode_index != 0,
            "visible_finite_support_float32": (finite_support * mask).numpy(),
            "visible_centre_label_correspondence_weight_float32": slab["slab_supervision_weight_or_abstention"]["dense_correspondence_weight"] * mask.numpy(),
            "finite_support_mass_px": finite_mass, "visible_support_mass_px": visible_mass,
            "central_brain_pixel_count": int(slab["centre_plane_support_mask"].sum()),
            "visible_central_brain_pixel_count": int((slab["centre_plane_support_mask"] & mask.numpy()).sum()),
            "support_information_eligible": eligible, "support_censored": not eligible,
        })
    paths = _write_raw_artifact(OUTPUT, f"section_{index:08d}", section)
    artifact_sha256 = {}
    for name in paths.values():
        with (OUTPUT / name).open("rb") as stream:
            artifact_sha256[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {
        "section_index": index, "lineage": lineage, "artifacts": paths, "artifact_sha256": artifact_sha256,
        "thickness_um": thickness, "horizontal_reflection": reflection[0],
        "canonical_fit_diagnostics": section["canonical_anatomy_plane_fit"]["diagnostics"],
        "observed_fit_diagnostics": section["observed_total_map_plane_fit"]["diagnostics"],
        "by_mode": {item["selected_mode"]: {key: item[key] for key in ("finite_support_mass_px", "visible_support_mass_px", "central_brain_pixel_count", "visible_central_brain_pixel_count", "support_censored")} for item in section["observations"]},
        "bytes": sum((OUTPUT / name).stat().st_size for name in paths.values()),
        "seconds": time.perf_counter() - tick,
    }
    records.append(record)
    print(json.dumps(record), flush=True)

summary = {
    "protocol": protocol, "sections": records, "section_count": len(records), "observation_count": len(records) * len(MODES),
    "subject_count": 1, "all_training": True, "rejected_sections": 0,
    "censored_by_mode": {mode: sum(record["by_mode"][mode]["support_censored"] for record in records) for mode in MODES},
    "section_artifact_bytes": sum(record["bytes"] for record in records), "elapsed_seconds": time.perf_counter() - started,
    "prepared_not_model_performance": True,
}
(OUTPUT / "completed.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({key: value for key, value in summary.items() if key not in ("protocol", "sections")}), flush=True)
