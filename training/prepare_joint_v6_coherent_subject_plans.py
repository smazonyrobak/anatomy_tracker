"""Prepare eight training and four development synthetic 3D subjects; no sections."""

import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "4"
sys.dont_write_bytecode = True

import hashlib
import json
import shutil
import subprocess
import time

import numpy as np
import torch

from training import arbitrary_plane_acquisition_v2 as acquisition
from training import arbitrary_plane_subject_deformation_v2 as deformation
from training import subject_deformed_slab_multiresolution_bundle_v2 as bundle

OUTPUT = ROOT / "data/joint_v6_coherent_subject_plans_002"
SEED = 2026092911
CONTEXT_SHA256 = "c3bd31cc81af2788437cfd064f4cbf44d1d9f111919031c4c691552e796d94a8"
SPLITS = (("train", 8, 64), ("development", 4, 32))
repository = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / "source").mkdir()
source_names = sorted(set(acquisition._source_hashes()) | set(deformation._source_hashes()) | {Path(__file__).name, "subject_deformed_slab_multiresolution_bundle_v2.py"})
source_paths = [repository / "training" / name for name in source_names]
source_paths.append(repository / "publication/arbitrary_plane_acquisition_hardening_preflight.yaml")
source_hashes = {}
for path in source_paths:
    relative = path.relative_to(repository)
    archived = OUTPUT / "source" / relative
    archived.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, archived)
    source_hashes[relative.as_posix()] = hashlib.sha256(archived.read_bytes()).hexdigest()
roster = []
for split, count, future_planes in SPLITS:
    for index in range(count):
        prefix = f"joint-v6-coherent-cohort-002-{split}"
        roster.append({"split": split, "animal_index": index, "subject_id": f"{prefix}-subject-{index:08d}", "animal_id": f"{prefix}-animal-{index:08d}", "specimen_id": f"{prefix}-specimen-{index:08d}", "experiment_id": f"{prefix}-experiment-{index:08d}", "planned_future_planes": future_planes})
protocol = {
    "root_seed": SEED, "subject_roster": roster, "subject_count": 12,
    "source_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
    "source_file_sha256": source_hashes,
    "source_hash_kind": "exact archived file bytes; existing plan receipts separately use LF-normalized sampler source hashes",
    "expected_context_sha256": CONTEXT_SHA256,
    "plan_parameters": "existing sample_animal_subject_deformation_plan_v2 defaults; standard nonidentity stratum",
    "rng": "existing PCG64DXSM fields derived from root seed, split, animal index, scope, stage and field; no per-section plan resampling",
    "acceptance": "existing deterministic amplitude schedule and numerical gates; first accepted candidate; no subject redraw, identity substitution or relaxed gates",
    "authentication": "fresh accepted sampler results and content receipts, not an additional independent numerical replay",
    "scope": "12 independent RNG-generated synthetic subject maps from one pinned Allen atlas, not12 biological animals, population variability validation or a public benchmark",
    "split_policy": "all future sections, processing maps and appearances of one subject remain in its preassigned split; development subjects never become training rows",
    "future_sections": "not generated here;64 planes per training subject and32 per development subject are planned only after the separate GPU coordinate check",
    "no_model_dependencies": True, "section_count": 0,
}
(OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
started = time.perf_counter()
print("Preparing pinned Allen v2 context on CPU; no section rendering", flush=True)
context, atlas_inputs = bundle.load_pinned_allen_context(ROOT / "data/Allen Brain Atlas 25um")
assert context["v2_context_sha256"] == CONTEXT_SHA256
lower = np.zeros(3, dtype=np.float64)
upper = np.asarray(context["opaque_v1_context"]["scalar_tensor"].shape, dtype=np.float64) * 25.
context_record = {"v2_context_sha256": context["v2_context_sha256"], "receipt": acquisition._json_value(context["receipt"]), "atlas_inputs": atlas_inputs, "full_ccf_lower_ap_dv_ml_um": lower.tolist(), "full_ccf_upper_ap_dv_ml_um": upper.tolist(), "voxel_face_origin_ap_dv_ml_um": lower.tolist(), "voxel_size_ap_dv_ml_um": [25., 25., 25.]}
(OUTPUT / "context.json").write_text(json.dumps(context_record, indent=2), encoding="utf-8")
del context
records = []
for ordinal, identity in enumerate(roster):
    tick = time.perf_counter()
    print(json.dumps({"event": "subject_started", "ordinal": ordinal + 1, "total": len(roster), **identity}), flush=True)
    plan = deformation.sample_animal_subject_deformation_plan_v2(
        lower, upper, root_seed=SEED, split=identity["split"], animal_index=identity["animal_index"],
        animal_id=identity["animal_id"], ccf_context_sha256=CONTEXT_SHA256,
    )
    directory = OUTPUT / identity["split"] / f"subject_{identity['animal_index']:08d}"
    directory.mkdir(parents=True, exist_ok=False)
    files = bundle._write_raw_artifact(directory, "subject_plan", plan)
    receipt = deformation.subject_deformation_plan_receipt_v2(plan)
    assert receipt["receipt_sha256"] == plan["receipt_sha256"]
    (directory / "subject_plan_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    lineage = {**identity, "synthetic_animal_id": plan["synthetic_animal_id"], "subject_deformation_plan_id": plan["subject_deformation_plan_id"], "subject_deformation_realization_id": plan["subject_deformation_realization_id"], "subject_plan_receipt_sha256": plan["receipt_sha256"], "root_seed": SEED, "ccf_context_sha256": CONTEXT_SHA256}
    (directory / "lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    record = {
        **lineage, "directory": directory.relative_to(OUTPUT).as_posix(), "plan_files": files,
        "accepted_amplitude_um": plan["realization"]["accepted_amplitude_um"],
        "accepted_candidate_index": plan["realization"]["accepted_candidate_index"],
        "candidate_audits": acquisition._json_value(plan["realization"]["candidate_audits"]),
        "global_scale_ap_dv_ml": plan["state"]["global_scale"].tolist(),
        "generation_seconds": time.perf_counter() - tick,
        "artifact_sha256": {name: bundle._file_sha256(directory / name) for name in (*files.values(), "subject_plan_receipt.json", "lineage.json")},
    }
    records.append(record)
    (directory / "completed.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"event": "subject_frozen", "ordinal": ordinal + 1, "total": len(roster), **lineage, "accepted_amplitude_um": record["accepted_amplitude_um"], "accepted_candidate_index": record["accepted_candidate_index"], "generation_seconds": record["generation_seconds"]}), flush=True)
    del plan
for path in source_paths:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hashes[path.relative_to(repository).as_posix()]
completed = {"protocol": protocol, "context_sha256": CONTEXT_SHA256, "subjects": records, "section_count": 0, "total_seconds": time.perf_counter() - started, "artifact_sha256": {path.relative_to(OUTPUT).as_posix(): bundle._file_sha256(path) for path in sorted(OUTPUT.rglob("*")) if path.is_file()}}
(OUTPUT / "completed.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
print(json.dumps({"event": "cohort_frozen", "training_subjects": 8, "development_subjects": 4, "section_count": 0, "total_seconds": completed["total_seconds"]}), flush=True)
