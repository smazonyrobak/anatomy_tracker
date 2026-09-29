"""Post-exit FP64 target-capacity screen; no model, renderer or qualification.

Surface cap checks concern one affine-free representative, not the optimal
bounded prefield. Finite tanh logits and low-resolution decoder capacity are
not certified. Native derivative/gauge code is reused as an implementation
measurement, not an independent proof of that implementation.
"""

from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
DATA = ROOT / "data/joint_v6_coherent_subject_cohort_sections_002"
OUTPUT = ROOT / "runs/joint_v6_coherent_ribbon_target_capacity_002"
COMPLETION_SHA256 = "ba51982a5b03b61d4bcf7f37f2c139dd6c1caf7ff66a6124cb678ab5ee9dd1f2"
if len(COMPLETION_SHA256) != 64:
    raise RuntimeError("No cohort output has been read; supply the completion hash only after confirmed section-generator EXIT")

import os
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "4"
import hashlib
import json
import numpy as np
import torch
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_ribbon_v6 import project_surface_affine_out, ribbon_derivative_frobenius_bound
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact

completion_bytes = (DATA / "completed.json").read_bytes()
assert hashlib.sha256(completion_bytes).hexdigest() == COMPLETION_SHA256
completed = json.loads(completion_bytes)
assert completed["section_count"] == 640 and completed["observation_count"] == 1920
torch.set_num_threads(4)
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / "audit_source.py").write_bytes(Path(__file__).read_bytes())
section_rows, observation_rows = [], []
for index, record in enumerate(completed["sections"]):
    for name, expected in record["artifact_sha256"].items():
        with (DATA / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
    section = _read_raw_artifact(DATA, record["artifacts"])
    assert section["lineage"] == record["lineage"] and not section["reflection_xy"][1]
    centre = section["target_centre_ccf_coordinates_ap_dv_ml_um_float64"]
    slab = section["target_psf_ccf_coordinates_ap_dv_ml_um_float64"]
    height, width = centre.shape[:2]
    identity = np.stack(np.meshgrid(np.arange(height), np.arange(width), indexing="ij"))
    assert np.array_equal(section["section_processing_observed_pullback_yx_px_float64"], identity)
    reflected = bool(section["reflection_xy"][0])
    if reflected:
        centre, slab = centre[:, ::-1], slab[:, :, ::-1]  # raster W only; never CCF-component/director signs
    assert centre.dtype == slab.dtype == np.float64
    unflip_error = float(np.abs(centre - section["canonical_anatomy_centre_ccf_ap_dv_ml_um_float64"]).max())
    fit = section["canonical_anatomy_plane_fit"]["arrays"]
    ouv = np.asarray(fit["physical_ouv_ap_dv_ml_um_float64"]).reshape(3, 3)
    _, frame_tensor, basis = physical_ouv_to_frame(torch.from_numpy(ouv))
    frame, normal = frame_tensor.numpy(), frame_tensor.numpy()[:, 2]
    y, x = np.meshgrid(np.arange(height) / height, np.arange(width) / width, indexing="ij")
    plane = ouv[0] + x[..., None] * ouv[1] + y[..., None] * ouv[2]
    z, w = section["axial_offsets_um_float64"], section["axial_weights_float64"]
    assert abs(w.sum() - 1) < 1e-12 and np.any(z == 0) and np.sum(w * z * z) > 0
    director = np.einsum("s,shwc->hwc", w * z, slab - centre[None]) / np.sum(w * z * z)
    r = np.einsum("hwi,ij->jhw", centre - plane, frame)
    d = np.einsum("hwi,ij->jhw", director - normal, frame)
    rt, dt = torch.from_numpy(r.copy())[None], torch.from_numpy(d.copy())[None]
    gauge_free, affine = project_surface_affine_out(rt)
    bound = float(ribbon_derivative_frobenius_bound(gauge_free, dt, basis[None], torch.from_numpy(z)[None])[0])
    projected_slab = centre[None] + z[:, None, None, None] * director[None]
    projection_squared = np.square(projected_slab - slab).sum(-1)
    # One optimistic cap-closure construction, not the best reachable field.
    capped_r, _ = project_surface_affine_out(rt.clamp(-200., 200.))
    capped_d = dt.clamp(-.2, .2)
    capped_bound = float(ribbon_derivative_frobenius_bound(capped_r, capped_d, basis[None], torch.from_numpy(z)[None])[0])
    scale = min(1., .35 / max(capped_bound, .35))
    capped_centre = plane + np.einsum("ij,jhw->hwi", frame, capped_r[0].numpy() * scale)
    capped_director = normal + np.einsum("ij,jhw->hwi", frame, capped_d[0].numpy() * scale)
    capped_squared = np.square(capped_centre[None] + z[:, None, None, None] * capped_director[None] - slab).sum(-1)
    values = {
        "canonical_spatial_unflip_max_abs_um": unflip_error,
        "stored_plane_reconstruction_max_abs_um": float(np.abs(plane - fit["fitted_coordinate_raster_ap_dv_ml_um_float64"]).max()),
        "surface_absmax_um": float(np.abs(r).max()), "surface_components_at_or_beyond_200um_fraction": float((np.abs(r) >= 200).mean()),
        "director_delta_absmax": float(np.abs(d).max()), "director_components_at_or_beyond_point2_fraction": float((np.abs(d) >= .2).mean()),
        "surface_affine_gauge_coefficient_absmax_um": float(affine.abs().max()),
        "target_physical_derivative_bound": bound, "target_derivative_exceeds_point35": float(bound > .35),
        "normal_fit_residual_rms_um": float(np.sqrt(np.mean(r[2] ** 2))), "normal_fit_residual_max_um": float(np.abs(r[2]).max()),
        "slab_linear_projection_psf_rms_um": float(np.sqrt(np.mean(np.einsum("s,shw->hw", w, projection_squared)))),
        "slab_linear_projection_max_um": float(np.sqrt(projection_squared.max())),
        "cap_closure_derivative_bound_before_rescale": capped_bound, "cap_closure_joint_rescale": scale,
        "cap_closure_slab_psf_rms_um": float(np.sqrt(np.mean(np.einsum("s,shw->hw", w, capped_squared)))),
        "cap_closure_slab_max_um": float(np.sqrt(capped_squared.max())),
    }
    row = {**record["lineage"], "horizontal_reflection": reflected, "subject_plan_receipt_sha256": record["subject_plan_receipt_sha256"],
           "artifact_sha256": record["artifact_sha256"], "surface_local_component_absmax_um": np.abs(r).max((1, 2)).tolist(),
           "director_local_component_absmax": np.abs(d).max((1, 2)).tolist(),
           "stored_canonical_fit_diagnostics": section["canonical_anatomy_plane_fit"]["diagnostics"],
           "stored_observed_fit_diagnostics": section["observed_total_map_plane_fit"]["diagnostics"], **values}
    section_rows.append(row)
    for observation in section["observations"]:
        q = observation["visible_finite_support_float32"].astype(np.float64)
        if reflected:
            q = q[:, ::-1]
        mass = q.sum()
        observation_rows.append({**observation["lineage"], "selected_mode": observation["selected_mode"],
            "support_censored": observation["support_censored"], "visible_support_mass_px": float(mass), **values,
            "visible_normal_fit_residual_rms_um": float(np.sqrt((q * r[2] ** 2).sum() / mass)) if mass > 0 else None,
            "visible_slab_linear_projection_psf_rms_um": float(np.sqrt((q * np.einsum("s,shw->hw", w, projection_squared)).sum() / mass)) if mass > 0 else None,
            "visible_cap_closure_slab_psf_rms_um": float(np.sqrt((q * np.einsum("s,shw->hw", w, capped_squared)).sum() / mass)) if mass > 0 else None})
    if (index + 1) % 32 == 0:
        print(json.dumps({"sections_audited": index + 1, "retained_all_censored": True}), flush=True)

groups = {"all_sections": section_rows}
for split in ("train", "development"):
    groups[f"{split}/sections"] = [r for r in section_rows if r["split"] == split]
for subject in sorted({r["subject_id"] for r in section_rows}):
    groups[f"subject:{subject}/sections"] = [r for r in section_rows if r["subject_id"] == subject]
for subject in ("all", *sorted({r["subject_id"] for r in observation_rows})):
    for mode in ("raw", "exact_black", "imperfect_brush"):
        for support in ("all", "eligible", "censored"):
            groups[f"subject:{subject}/{mode}/{support}"] = [r for r in observation_rows if (subject == "all" or r["subject_id"] == subject) and r["selected_mode"] == mode and (support == "all" or r["support_censored"] == (support == "censored"))]
aggregates = {}
for name, rows in groups.items():
    metric_keys = [*values, *(k for k in observation_rows[0] if k.startswith("visible_") and k.endswith("rms_um"))]
    aggregates[name] = {"row_count": len(rows), "distinct_sections": len({r["section_id"] for r in rows}), "metrics": {}}
    for key in metric_keys:
        array = np.asarray([r[key] for r in rows if r.get(key) is not None], dtype=np.float64)
        aggregates[name]["metrics"][key] = {"count": len(array), "min_median_p95_max": np.quantile(array, [0, .5, .95, 1]).tolist(), "mean": float(array.mean())} if len(array) else None
for name, rows in (("sections.jsonl", section_rows), ("observations.jsonl", observation_rows)):
    (OUTPUT / name).write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
repository = Path(__file__).resolve().parents[1]
summary = {"scope": __doc__, "cohort": str(DATA), "completion_sha256": COMPLETION_SHA256,
    "section_count": len(section_rows), "observation_count": len(observation_rows), "rejected_sections": 0,
    "method": "FP64 original coordinates; spatial-only reflection undo; canonical OUV frame; r=R^T(C-P), D=sum(w*z*(slab-C))/sum(w*z*z), d=R^T(D-n)",
    "limits": "200um/.2 pre-projection tanh caps; shared physical derivative bound.35; cap-closure construction can attain boundary values only as infinite-logit limits; no minimum-error or learned-representation claim",
    "source_sha256": {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__).resolve(), repository / "training/arbitrary_plane_ribbon_v6.py", repository / "training/arbitrary_plane_geometry.py")},
    "aggregates": aggregates, "output_sha256": {name: hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() for name in ("sections.jsonl", "observations.jsonl")}}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
print(json.dumps({"output": str(OUTPUT), "sections": len(section_rows), "all_section_metrics": aggregates["all_sections"]}), flush=True)
