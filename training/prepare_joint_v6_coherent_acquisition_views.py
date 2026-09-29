"""128 new finite-FOV views of eight accepted TRAIN subjects;384 paired modes.

Borrow joint acquisition-chart tuples only from frozen TRAIN affine geometry.
No new subject, model, development draw, observed image crop or 2D-SVF target.
"""

import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda")
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "4"
sys.dont_write_bytecode = True
READY_AFTER_SOURCE_REVIEW = True
assert READY_AFTER_SOURCE_REVIEW, "Prepared only: root reviews, commits and schedules generation"

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
from training.arbitrary_plane_subject_deformation_v2 import subject_deformation_plan_receipt_v2, _ccf_to_subject_points_from_verified_plan_v2
from training.arbitrary_plane_subject_sampling_v6 import sample_subject_planes
from training.arbitrary_plane_subject_section_v2 import sample_coordinate_rasters_v2
from training.arbitrary_plane_subject_torch_v6 import map_accepted_subject_points_torch_v6
from training.arbitrary_plane_synthetic_generator_v2 import reduce_v2_slab_samples
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact, _write_raw_artifact

PLAN_ROOT = ROOT / "data/joint_v6_coherent_subject_plans_002"
OUTPUT = ROOT / "data/joint_v6_coherent_acquisition_views_001"
REAL_GEOMETRY = ROOT / "data/allen_real_training_inputs_20260929"
GEOMETRY_SHA256 = "9a718b2fb2791584e1d225123facbe1c02c96cfa701a9e6f3fc68f17a565d218"
GEOMETRY_SUMMARY_SHA256 = "4f8484bbd2ab5719971d9862f3f5ea87ea3d1c9de76468a936af67bcc65da9fd"
CATALOGUE = ROOT / "runs/joint_v6_proposal_substantive_001/catalogue.pt"
CATALOGUE_SHA256 = "9b49d203cc73ce3a66e648bbe5228231eb5cc9c17d5db4669eefe0f08ae22c71"
PLAN_COMPLETION_SHA256 = "83fa8fc0ab19f15c8b64babdfe1e98251a972c6759dade08e5513e70c76a7c89"
GPU_EQUIVALENCE_RECEIPT_SHA256 = "a5b71a0483c027499059160399ab742e914b3c9163145d0b4a895a3dc9646546"
GPU_EQUIVALENCE_RECEIPT = ROOT / "runs/joint_v6_subject_torch_gpu_check_001/completed.json"
CONTEXT_SHA256 = "c3bd31cc81af2788437cfd064f4cbf44d1d9f111919031c4c691552e796d94a8"
SEED, SHAPE, SUPPORT_MASS_THRESHOLD = 2026092925, (96, 96), 64.0
SPLITS = (("train", 8, 16),)
MODES = ("raw", "exact_black", "imperfect_brush")
repository = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
assert len(PLAN_COMPLETION_SHA256) == 64
completion_bytes = (PLAN_ROOT / "completed.json").read_bytes()
assert hashlib.sha256(completion_bytes).hexdigest() == PLAN_COMPLETION_SHA256
cohort = json.loads(completion_bytes)
assert cohort["context_sha256"] == CONTEXT_SHA256 and cohort["protocol"]["root_seed"] == 2026092911
assert len(cohort["subjects"]) == 12 and cohort["section_count"] == 0
assert hashlib.sha256(GPU_EQUIVALENCE_RECEIPT.read_bytes()).hexdigest() == GPU_EQUIVALENCE_RECEIPT_SHA256
gpu_equivalence = json.loads(GPU_EQUIVALENCE_RECEIPT.read_text())
geometry_bytes = (REAL_GEOMETRY / "image_geometry.jsonl").read_bytes()
assert hashlib.sha256(geometry_bytes).hexdigest() == GEOMETRY_SHA256
assert hashlib.sha256((REAL_GEOMETRY / "summary.json").read_bytes()).hexdigest() == GEOMETRY_SUMMARY_SHA256
geometry_summary = json.loads((REAL_GEOMETRY / "summary.json").read_text())
assert geometry_summary["output_sha256"]["image_geometry.jsonl"] == GEOMETRY_SHA256
with CATALOGUE.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == CATALOGUE_SHA256
catalogue = torch.load(CATALOGUE, map_location="cpu", weights_only=False)
support_origin = np.asarray(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"], dtype=np.float64)
assert np.array_equal(support_origin, [6600., 3862.5, 5700.])
geometry_lines = geometry_bytes.decode("utf-8").splitlines()
geometry_rows = [json.loads(line) for line in geometry_lines]
assert len(geometry_rows) == 256 and all(row["split"] == "development_train" and row["array_row_index"] == i for i, row in enumerate(geometry_rows))
donors = np.array([row["animal_id"] for row in geometry_rows]); donor_ids = np.unique(donors)
assert len(donor_ids) == 58
affines = np.array([row["model_pixel_to_ap_dv_ml_um"] for row in geometry_rows], dtype=np.float64)
O, U, V = affines[:, :, 2], 96 * affines[:, :, 0], 96 * affines[:, :, 1]
Lu = np.linalg.norm(U, axis=1); u = U / Lu[:, None]
V_along_u = (u * V).sum(-1); V_perpendicular = V - u * V_along_u[:, None]
Lv = np.linalg.norm(V_perpendicular, axis=1); v = V_perpendicular / Lv[:, None]
centres = O + .5 * (U + V)
nuisance = np.column_stack(((u * (centres - support_origin)).sum(-1), (v * (centres - support_origin)).sum(-1), Lu, Lv, V_along_u / Lv))
donor_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, 0])))
row_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, 1])))
donor_schedule = np.concatenate([donor_rng.permutation(donor_ids) for _ in range(3)])[:128]
nuisance_rows = np.empty(128, dtype=np.int64)
for donor in donor_ids:
    selected = donor_schedule == donor
    nuisance_rows[selected] = row_rng.choice(np.flatnonzero(donors == donor), size=int(selected.sum()), replace=True)
for split, count, plane_count in SPLITS:
    selected = [item for item in cohort["subjects"] if item["split"] == split]
    assert sorted(item["animal_index"] for item in selected) == list(range(count))
    assert all(item["planned_future_planes"] == 64 for item in selected)
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / "parent_plan_completed.json").write_bytes(completion_bytes)
(OUTPUT / "nuisance_source_geometry.jsonl").write_bytes(geometry_bytes)
shutil.copyfile(REAL_GEOMETRY / "summary.json", OUTPUT / "nuisance_source_summary.json")
shutil.copyfile(CATALOGUE, OUTPUT / "catalogue.pt")
np.savez(OUTPUT / "nuisance_schedule.npz", subject_index=np.repeat(np.arange(8), 16),
         section_index=np.tile(np.arange(16), 8), source_geometry_row_index=nuisance_rows,
         nuisance_source_donor_id=donor_schedule, selected_joint_nuisance=nuisance[nuisance_rows],
         all_source_joint_nuisance=nuisance, donor_seed_sequence=[SEED, 0], row_seed_sequence=[SEED, 1])
shutil.copyfile(GPU_EQUIVALENCE_RECEIPT, OUTPUT / "gpu_coordinate_equivalence_receipt.json")
for name in ("protocol.json", "context.json"):
    assert hashlib.sha256((PLAN_ROOT / name).read_bytes()).hexdigest() == cohort["artifact_sha256"][name]
    shutil.copyfile(PLAN_ROOT / name, OUTPUT / f"parent_plan_{name}")
source_names = (
    Path(__file__).name, "arbitrary_plane_coherent_subject_v6.py", "arbitrary_plane_subject_torch_v6.py",
    "arbitrary_plane_subject_sampling_v6.py", "arbitrary_plane_subject_deformation_v2.py",
    "arbitrary_plane_subject_section_v2.py", "arbitrary_plane_full_frame_primitives.py",
    "arbitrary_plane_geometry.py", "arbitrary_plane_acquisition_v2.py", "arbitrary_plane_allen_atlas_binding_v6.py",
    "arbitrary_plane_synthetic_generator_v2.py", "subject_deformed_slab_multiresolution_bundle_v2.py",
)
(OUTPUT / "source").mkdir()
source = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(), "file_sha256": {}}
for name in source_names:
    shutil.copyfile(repository / "training" / name, OUTPUT / "source" / name)
    source["file_sha256"][f"training/{name}"] = hashlib.sha256((OUTPUT / "source" / name).read_bytes()).hexdigest()
assert source["file_sha256"]["training/arbitrary_plane_subject_torch_v6.py"] == gpu_equivalence["mapper_source_sha256"]
atlas_identity = {
    "atlas_id": "Allen CCFv3", "version": "2017 25um", "coordinate_axes": ["AP", "DV", "ML"],
    "template_path": str(allen.TEMPLATE_PATH_V6), "template_raw_sha256": allen.TEMPLATE_RAW_SHA256_V6,
    "annotation_path": str(allen.ANNOTATION_PATH_V6), "annotation_raw_sha256": allen.ANNOTATION_RAW_SHA256_V6,
    "render_volume_receipt": allen.ATLAS_FLOAT32_RECEIPT_V6, "annotation_decoded_receipt": allen.ANNOTATION_DECODED_RECEIPT_V6,
    "origin_um": allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6, "voxel_size_um": allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6,
    "intensity": "v6 pinned within-brain normalization and exact atlas exterior zero; not the plan-context v2 scalar preprocessing",
}
mapping = {"evaluator": "map_accepted_subject_points_torch_v6", "source_sha256": source["file_sha256"]["training/arbitrary_plane_subject_torch_v6.py"], "device": "cuda", "dtype": "float64", "inverse": True, "batch_size": 8192, "gpu_equivalence_receipt_sha256": GPU_EQUIVALENCE_RECEIPT_SHA256, "rk4_steps": "exact accepted-plan resolved_config.flow.steps, recorded per subject/section", "coefficient_rule": "resident accepted coarse/fine coefficients once per subject; no reprojection, weighting or resampling"}
mapping.update({"gpu_name": torch.cuda.get_device_name(), "torch_version": torch.__version__,
                "support_origin_forward_evaluator": "_ccf_to_subject_points_from_verified_plan_v2",
                "support_origin_forward_device_dtype": "CPU NumPy float64; once per accepted TRAIN subject",
                "scale_flow_order": "forward positive RK4 then accepted diagonal scale about frozen centre; inverse diagonal scale then negative RK4",
                "exterior": "zero-extended cubic velocity, no physical-query clamp; inverse global scale remains outside velocity support"})
catalogue_binding = {"path": str(CATALOGUE), "sha256": CATALOGUE_SHA256, "receipt_sha256": catalogue["receipt_sha256"], "support_origin_ap_dv_ml_um": support_origin.tolist()}
nuisance_binding = {"geometry_path": str(REAL_GEOMETRY / "image_geometry.jsonl"), "geometry_sha256": GEOMETRY_SHA256, "summary_sha256": GEOMETRY_SUMMARY_SHA256,
                   "source_rows": 256, "source_train_donors": 58, "columns": ["tangent_shift_u_um", "tangent_shift_v_um", "Lu_um", "Lv_um", "dimensionless_shear"],
                   "half_voxel": "already present in model-pixel affine; no additional shift", "tuple_rule": "all five values from one source row; donor-balanced permutations then uniform row within donor",
                   "ancestry_role": "acquisition-nuisance source only, never synthetic biological parent",
                   "schedule_sha256": hashlib.sha256((OUTPUT / "nuisance_schedule.npz").read_bytes()).hexdigest()}
protocol = {
    "source": source, "root_seed": SEED, "subject_count": 8, "section_count": 128, "observation_count": 384,
    "splits": {split: {"subjects": count, "sections_per_subject": planes} for split, count, planes in SPLITS},
    "raster_shape_h_w": SHAPE, "plan_source_directory": str(PLAN_ROOT), "plan_completion_sha256": PLAN_COMPLETION_SHA256,
    "plan_context_sha256": CONTEXT_SHA256, "atlas": atlas_identity, "coordinate_mapping": mapping, "catalogue_binding": catalogue_binding, "acquisition_nuisance": nuisance_binding,
    "authentication": "completed parent inventory, exact subject files, content receipts and sampler source binding; no redundant plan replay",
    "rng": "NumPy PCG64: donor SeedSequence[root,0], row[root,1], section[root,2,animal_index,section_index,branch]; branch plane/thickness0,reflection1,appearance2,CPU-noise3. Every entropy vector and noise uint64 saved.",
    "sampling": "uniform RP2 normal/roll and conservative subject-box-intersecting finite-slab offset; no tissue rejection or pose retry",
    "physical_fov": "retain sampled RP2 plane/roll/slab offset; project forward-mapped support origin to plane; borrow the entire TRAIN tangent-centre/span/shear tuple in physical subject um with no extra global scale; x/96,y/96. Full-frame centre differs from actual pixel centroid by(U+V)/192.",
    "psf": "25-100um independently per section;9 linspace positions and integer masses[1,2,2,2,2,2,2,2,1] normalized globally",
    "processing": "identity observed2D pullback, finite horizontal raster reflection, subject3D map; not zero total anatomical deformation",
    "modes": MODES, "pairing": "three appearances share subject,section,reflection,gain/gamma/inversion,CPU noise and synthetic background before mask",
    "appearance_order": "post-PSF conditional-tissue intensity/rendered support -> gamma/inversion/gain -> restore support mass and synthetic background -> shared noise -> optional mask; not a calibrated acquisition model",
    "annotation": "existing CPU nearest ties-to-even labels at new FP32 Allen-index queries, lower-ID modal ties; background0 retained; trilinear finite support separate",
    "censoring": "retain every draw/mode; information eligibility finite and visible support mass>=64; not a final pose-supervision rule",
    "target_kind": "coherent_subject_3d_coordinate_targets_only", "legacy_total_2d_svf_pack_compatible": False, "catalogue_cell_truth": None,
    "scope": "128 sparse new physical views of eight existing one-atlas TRAIN synthetic subjects; no new independent animal, real appearance correction, model accuracy, calibration, training launch or benchmark claim",
    "split_policy": "only eight accepted TRAIN plans; no development plan/section/nuisance geometry or real pixels loaded",
    "outside_domain_counts": "target slab/centre queries outside physical atlas extent[origin,origin+shape*spacing), plus outside voxel-centre interpolation domain separately; partial-volume interpolation retained, targets never clipped",
    "atlas_padding": "global PSF normalization once; zero trilinear padding; nearest ties-to-even annotation outside array is0",
}
(OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
started = time.perf_counter()
print("Decoding pinned atlas on CPU; only accepted subject coordinate mapping uses CUDA FP64", flush=True)
atlas_array, annotation_array = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array)
annotation = torch.from_numpy(annotation_array.astype(np.int64))
del annotation_array
identity_yx = np.stack(np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing="ij")).astype(np.float64)
yy, xx = torch.meshgrid(torch.linspace(-1, 1, SHAPE[0]), torch.linspace(-1, 1, SHAPE[1]), indexing="ij")
integer_masses = np.array([1, 2, 2, 2, 2, 2, 2, 2, 1], dtype=np.int64)
psf_weights = integer_masses.astype(np.float64) / integer_masses.sum()
tensor_keys = ("accepted_coarse_coefficients_um", "coarse_origin_um", "coarse_spacing_um", "accepted_fine_coefficients_um", "fine_origin_um", "fine_spacing_um", "global_scale", "frozen_center_um")
records = []
for split, subject_count, plane_count in SPLITS:
    for subject in sorted((item for item in cohort["subjects"] if item["split"] == split), key=lambda item: item["animal_index"]):
        plan_directory = PLAN_ROOT / subject["directory"]
        for name, expected in subject["artifact_sha256"].items():
            with (plan_directory / name).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
        plan = _read_raw_artifact(plan_directory, subject["plan_files"])
        assert subject_deformation_plan_receipt_v2(plan)["receipt_sha256"] == plan["receipt_sha256"] == subject["subject_plan_receipt_sha256"]
        assert plan["provenance"]["split"] == split and plan["provenance"]["animal_id"] == subject["animal_id"]
        assert plan["synthetic_animal_id"] == subject["synthetic_animal_id"] and plan["provenance"]["ccf_context_sha256"] == CONTEXT_SHA256
        assert plan["resolved_config"]["deformation_stratum"] == "standard"
        assert np.array_equal(plan["state"]["full_ccf_lower_um"], allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6)
        assert np.array_equal(plan["state"]["full_ccf_upper_um"], np.asarray(allen.ATLAS_SHAPE_AP_DV_ML_V6) * allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6)
        for name, expected in plan["source_sha256"].items():
            text = (repository / "training" / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            assert hashlib.sha256(text).hexdigest() == expected
        parameters_gpu = [torch.tensor(plan["state"][key], dtype=torch.float64, device="cuda") for key in tensor_keys]
        flow_steps = int(plan["resolved_config"]["flow"]["steps"])
        subject_support_origin = _ccf_to_subject_points_from_verified_plan_v2(support_origin[None], plan)[0]
        mapper = lambda points: map_accepted_subject_points_torch_v6(torch.from_numpy(points).to(device="cuda", dtype=torch.float64), *parameters_gpu, inverse=True, steps=flow_steps, batch_size=8192).cpu().numpy()
        subject_output = OUTPUT / split / f"subject_{subject['animal_index']:08d}"
        subject_output.mkdir(parents=True)
        (subject_output / "parent_plan_reference.json").write_text(json.dumps(subject, indent=2), encoding="utf-8")
        (subject_output / "support_origin_mapping.json").write_text(json.dumps({"ccf_support_origin_ap_dv_ml_um": support_origin.tolist(), "mapped_subject_support_origin_ap_dv_ml_um": subject_support_origin.tolist(), "subject_plan_receipt_sha256": plan["receipt_sha256"], "global_scale": np.asarray(plan["state"]["global_scale"]).tolist(), "flow_steps": flow_steps, "evaluator": mapping["support_origin_forward_evaluator"]}, indent=2), encoding="utf-8")
        subject_records = []
        print(json.dumps({"event": "subject_sections_started", "split": split, "animal_index": subject["animal_index"], "animal_id": subject["animal_id"], "section_count": plane_count}), flush=True)
        for index in range(plane_count):
            tick = time.perf_counter()
            seed_prefix = [SEED, 2, subject["animal_index"], index]
            geometry_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([*seed_prefix, 0])))
            reflection_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([*seed_prefix, 1])))
            appearance_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([*seed_prefix, 2])))
            noise_seed = int(np.random.SeedSequence([*seed_prefix, 3]).generate_state(1, dtype=np.uint64)[0])
            thickness = float(geometry_rng.uniform(25, 100))
            offsets = np.linspace(-.5, .5, 9) * thickness
            reflection = (bool(reflection_rng.integers(0, 2)), False)
            draw = sample_subject_planes(plan, geometry_rng, 1, SHAPE, offsets)
            slot = subject["animal_index"] * plane_count + index
            source_row = int(nuisance_rows[slot]); source_geometry = geometry_rows[source_row]
            shift_u, shift_v, span_u, span_v, shear = nuisance[source_row]
            old_ouv = draw["physical_ouv_ap_dv_ml_um"][0]; normal = draw["normal_ap_dv_ml"][0]
            plane_u = old_ouv[1] / np.linalg.norm(old_ouv[1]); plane_v = old_ouv[2] / np.linalg.norm(old_ouv[2])
            projected_origin = subject_support_origin + normal * np.dot(normal, old_ouv[0] - subject_support_origin)
            new_centre = projected_origin + shift_u * plane_u + shift_v * plane_v
            new_u, new_v = span_u * plane_u, span_v * (plane_v + shear * plane_u)
            new_ouv = np.stack((new_centre - .5 * (new_u + new_v), new_u, new_v))
            pixel_centroid = new_centre - (new_u + new_v) / 192.
            nuisance_source = {"role": "acquisition_nuisance_not_synthetic_parent", "geometry_row_index": source_row,
                               "geometry_file_sha256": GEOMETRY_SHA256, "geometry_line_utf8_sha256": hashlib.sha256(geometry_lines[source_row].encode("utf-8")).hexdigest(),
                               "donor_id": source_geometry["animal_id"], "specimen_id": source_geometry["specimen_id"],
                               "experiment_id": source_geometry["experiment_id"], "section_id": source_geometry["section_id"], "split": source_geometry["split"]}
            acquisition_chart = {"nuisance_source": nuisance_source, "joint_nuisance_float64": nuisance[source_row].copy(),
                                 "source_model_pixel_to_ap_dv_ml_um_float64": affines[source_row].copy(),
                                 "support_origin_ccf_ap_dv_ml_um_float64": support_origin.copy(),
                                 "support_origin_subject_ap_dv_ml_um_float64": subject_support_origin.copy(),
                                 "projected_support_origin_on_sampled_plane_um_float64": projected_origin,
                                 "full_extent_frame_centre_subject_um_float64": new_centre,
                                 "actual_pixel_centroid_subject_um_float64": pixel_centroid,
                                 "subject_ouv_ap_dv_ml_um_float64": new_ouv,
                                 "sampled_normal_ap_dv_ml_float64": normal.copy(),
                                 "sampled_rolled_uv_ap_dv_ml_float64": np.stack((plane_u, plane_v)),
                                 "sampled_plane_preservation_error_um": float(abs(np.dot(normal, new_centre - old_ouv[0]))),
                                 "units_and_scale": "borrowed shifts/spans used directly in subject physical um; no additional G; shear dimensionless"}
            section_id = f"joint-v6-coherent-acquisition-001-{split}-subject-{subject['animal_index']:08d}-section-{index:08d}"
            lineage = {key: subject[key] for key in ("subject_id", "animal_id", "specimen_id", "experiment_id", "split", "synthetic_animal_id")}
            lineage["section_id"] = section_id
            section_sources = {**atlas_identity, "coordinate_mapping": {**mapping, "rk4_steps": flow_steps}, "catalogue_binding": catalogue_binding, "acquisition_nuisance_source": nuisance_source}
            section = make_coherent_subject_section_v6(plan, new_ouv, identity_yx, reflection, offsets, psf_weights, lineage, section_sources, atlas, allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6, allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6, subject_to_ccf_mapper=mapper)
            rendered = section.pop("raw_rendered_channels")
            finite_support = rendered[1].clamp(0, 1)
            tissue = finite_support > 0
            allen_coordinates = physical_um_to_allen_index_points(torch.from_numpy(section["target_psf_ccf_coordinates_ap_dv_ml_um_float64"]).float(), allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6, allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6)
            scalar_samples, label_samples = sample_coordinate_rasters_v2(atlas[0], annotation, allen_coordinates)
            slab = reduce_v2_slab_samples(scalar_samples.numpy(), label_samples.numpy(), integer_masses, section["centre_psf_index"])
            slab.pop("scalar")
            section["nearest_psf_annotation_int64"] = label_samples.numpy()
            section["annotation_supervision"] = slab
            section["finite_psf_trilinear_support_float32"] = finite_support.numpy()
            section["raw_rendered_channels_float32"] = rendered.numpy()
            section["plane_sampling"] = draw
            section["plane_sampling_role"] = "original full-box plane/roll/slab draw only; actual finite chart is subject_ouv_ap_dv_ml_um_float64"
            section["acquisition_chart"] = acquisition_chart
            ccf_grid = section["target_psf_ccf_coordinates_ap_dv_ml_um_float64"]
            atlas_lower = np.asarray(allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6, dtype=np.float64)
            voxel_size = np.asarray(allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6, dtype=np.float64)
            atlas_upper = atlas_lower + np.asarray(allen.ATLAS_SHAPE_AP_DV_ML_V6) * voxel_size
            exterior = ((ccf_grid < atlas_lower) | (ccf_grid >= atlas_upper)).any(-1)
            centre_exterior = exterior[section["centre_psf_index"]]
            interpolation_exterior = ((ccf_grid < atlas_lower + .5 * voxel_size) | (ccf_grid > atlas_upper - .5 * voxel_size)).any(-1)
            section["outside_domain_counts"] = {"psf_query_count": int(exterior.size), "outside_atlas_physical_extent_psf_query_count": int(exterior.sum()),
                                                "outside_atlas_physical_extent_centre_pixel_count": int(centre_exterior.sum()),
                                                "outside_atlas_voxel_centre_domain_psf_query_count": int(interpolation_exterior.sum()),
                                                "outside_atlas_physical_extent_psf_weighted_pixel_mass": float(np.einsum("s,shw->", psf_weights, exterior))}
            section["preparation_seed"] = {"root_seed": SEED, "donor_seed_sequence": [SEED, 0], "row_seed_sequence": [SEED, 1], "sampling_slot": slot, "source_geometry_row_index": source_row, "geometry_seed_sequence": [*seed_prefix, 0], "reflection_seed_sequence": [*seed_prefix, 1], "appearance_seed_sequence": [*seed_prefix, 2], "noise_seed_sequence": [*seed_prefix, 3], "noise_seed_uint64": noise_seed}
            section["target_kind"] = "coherent_subject_3d_coordinate_targets_only"
            section["legacy_total_2d_svf_pack_compatible"] = False
            section["catalogue_cell_truth"] = None
            parameters = {"gain": float(appearance_rng.uniform(.6, 1.4)), "gamma": float(np.exp(appearance_rng.uniform(np.log(.6), np.log(1.6)))), "invert_tissue": bool(appearance_rng.integers(0, 2)), "noise_std": float(appearance_rng.uniform(.005, .05)), "background_mean": float(appearance_rng.uniform(0, .8)), "background_slope_yx": appearance_rng.uniform(-.15, .15, 2).tolist(), "imperfect_mask_dilate": bool(appearance_rng.integers(0, 2)), "imperfect_mask_radius_px": int(appearance_rng.integers(1, 4)), "noise_seed": noise_seed, "noise_device": "CPU"}
            appearance_noise = torch.randn(SHAPE, generator=torch.Generator().manual_seed(noise_seed))
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
                section["observations"].append({"lineage": {**lineage, "observation_id": f"{section_id}-{mode}"}, "selected_mode": mode, "image_outline_availability_float32": torch.stack((before_brush * mask, outline.float(), torch.full_like(before_brush, float(mode_index != 0)))).numpy(), "brush_keep_mask": mask.numpy(), "outline_available": mode_index != 0, "visible_finite_support_float32": (finite_support * mask).numpy(), "visible_centre_label_correspondence_weight_float32": slab["slab_supervision_weight_or_abstention"]["dense_correspondence_weight"] * mask.numpy(), "finite_support_mass_px": finite_mass, "visible_support_mass_px": visible_mass, "central_brain_pixel_count": int(slab["centre_plane_support_mask"].sum()), "visible_central_brain_pixel_count": int((slab["centre_plane_support_mask"] & mask.numpy()).sum()), "support_information_eligible": eligible, "support_censored": not eligible})
            paths = _write_raw_artifact(subject_output, f"section_{index:08d}", section)
            paths = {key: (subject_output / name).relative_to(OUTPUT).as_posix() for key, name in paths.items()}
            artifact_sha256 = {}
            for name in paths.values():
                with (OUTPUT / name).open("rb") as stream:
                    artifact_sha256[name] = hashlib.file_digest(stream, "sha256").hexdigest()
            record = {"section_index": index, "lineage": lineage, "acquisition_nuisance_source": nuisance_source, "sampled_plane_preservation_error_um": acquisition_chart["sampled_plane_preservation_error_um"], "outside_domain_counts": section["outside_domain_counts"], "artifacts": paths, "artifact_sha256": artifact_sha256, "thickness_um": thickness, "horizontal_reflection": reflection[0], "subject_plan_receipt_sha256": plan["receipt_sha256"], "canonical_fit_diagnostics": section["canonical_anatomy_plane_fit"]["diagnostics"], "observed_fit_diagnostics": section["observed_total_map_plane_fit"]["diagnostics"], "by_mode": {item["selected_mode"]: {key: item[key] for key in ("finite_support_mass_px", "visible_support_mass_px", "central_brain_pixel_count", "visible_central_brain_pixel_count", "support_censored")} for item in section["observations"]}, "bytes": sum((OUTPUT / name).stat().st_size for name in paths.values()), "seconds": time.perf_counter() - tick}
            records.append(record)
            subject_records.append(record)
            print(json.dumps({"event": "section_frozen", "split": split, "animal_index": subject["animal_index"], "section_index": index, "completed_sections": len(records), "seconds": record["seconds"], "censored_by_mode": {mode: record["by_mode"][mode]["support_censored"] for mode in MODES}}), flush=True)
        (subject_output / "completed.json").write_text(json.dumps({"parent_plan": subject, "sections": subject_records, "section_count": len(subject_records), "observation_count": len(subject_records) * len(MODES)}, indent=2), encoding="utf-8")
        del mapper, parameters_gpu, plan
for name in source_names:
    assert hashlib.sha256((repository / "training" / name).read_bytes()).hexdigest() == source["file_sha256"][f"training/{name}"]
assert len(records) == 128 and all(row["lineage"]["split"] == "train" for row in records)
summary = {"protocol": protocol, "sections": records, "section_count": len(records), "observation_count": len(records) * len(MODES), "subject_count": 8, "rejected_sections": 0, "by_split": {split: {"section_count": sum(row["lineage"]["split"] == split for row in records), "censored_by_mode": {mode: sum(row["by_mode"][mode]["support_censored"] for row in records if row["lineage"]["split"] == split) for mode in MODES}} for split, _, _ in SPLITS}, "nuisance_donor_counts": {str(d): int((donor_schedule == d).sum()) for d in donor_ids}, "outside_atlas_physical_extent_psf_query_count": sum(row["outside_domain_counts"]["outside_atlas_physical_extent_psf_query_count"] for row in records), "outside_atlas_physical_extent_centre_pixel_count": sum(row["outside_domain_counts"]["outside_atlas_physical_extent_centre_pixel_count"] for row in records), "section_artifact_bytes": sum(row["bytes"] for row in records), "elapsed_seconds": time.perf_counter() - started, "prepared_not_model_performance": True}
(OUTPUT / "completed.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in summary.items() if key not in ("protocol", "sections")}), flush=True)
