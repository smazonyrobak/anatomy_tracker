"""Frozen top-k candidate coverage diagnostic; no model or image is loaded."""

import hashlib
import json
from pathlib import Path

import numpy as np


root = Path(r"I:\AnatomyTracker")
evaluation = root / "runs/sagittal_mixed_pose_continuation_001_eval"
panel = root / "data/one_shot_fresh_synthetic_dev_panel_001"
out = root / "runs/sagittal_mixed_pose_candidate_analysis_001"
rows_bytes = (evaluation / "rows.jsonl").read_bytes()
assert hashlib.sha256(rows_bytes).hexdigest() == "9124dd5e8c4fcd4a70f436ef226baeefc899dd4fc5d6292f348eb0a371ecfa57"
panel_bytes = (panel / "records.jsonl").read_bytes()
assert hashlib.sha256(panel_bytes).hexdigest() == "7be8b6a95ed956c1223fa2dae8867d7747037ec15b8523a5e9b9969c154fc7fe"
rows = [json.loads(line) for line in rows_bytes.splitlines() if line]
synthetic_records = {row["section_id"]: row for row in map(json.loads, panel_bytes.splitlines()) if row["eligible"]}
assert len(synthetic_records) == 185
ks = (1, 2, 4, 8, 16, 32)
measures = {}
for domain, error_name in (("synthetic", "all_branch_mapped_um"),
                           ("coronal", "all_branch_five_point_um"),
                           ("sagittal", "all_branch_five_point_um")):
    chosen = [row for row in rows if row["step"] == 653 and row["set"] == domain]
    expected = {"synthetic": 185, "coronal": 64, "sagittal": 158}[domain]
    assert len(chosen) == expected
    per_identity = {}
    per_axis = {}
    for row in chosen:
        identity = row["animal_id"] if domain == "synthetic" else row["identity"]
        error = np.asarray(row[error_name], dtype=np.float64)
        prior = np.asarray(row["all_branch_prior_log_mass"], dtype=np.float64)
        assert error.shape == prior.shape == (32,)
        order = np.argsort(-prior, kind="stable")
        values = {f"top{k}_oracle_um": float(error[order[:k]].min()) for k in ks}
        values["selected_um"] = float(error[order[0]])
        values["oracle_regret_um"] = values["selected_um"] - values["top32_oracle_um"]
        per_identity.setdefault(str(identity), []).append(values)
        if domain == "synthetic":
            record = synthetic_records[row["section_id"]]
            normal = np.abs(np.asarray(record["plane_normal_ap_dv_ml"], dtype=np.float64))
            axis = ("AP", "DV", "ML")[int(normal.argmax())]
            per_axis.setdefault(axis, []).append(values)
    names = (*[f"top{k}_oracle_um" for k in ks], "selected_um", "oracle_regret_um")
    by_identity = {identity: {name: float(np.mean([value[name] for value in samples]))
                              for name in names} for identity, samples in per_identity.items()}
    measures[domain] = {"sections": expected, "identities": len(by_identity),
                        "identity_equal_mean_um": {name: float(np.mean([value[name] for value in by_identity.values()]))
                                                   for name in names}}
    if domain == "synthetic":
        measures[domain]["nearest_plane_axis"] = {
            axis: {"sections": len(samples), "selected_um": float(np.mean([value["selected_um"] for value in samples])),
                   "top8_oracle_um": float(np.mean([value["top8_oracle_um"] for value in samples])),
                   "top32_oracle_um": float(np.mean([value["top32_oracle_um"] for value in samples]))}
            for axis, samples in per_axis.items()}
summary = {"version": "sagittal-mixed-pose-candidate-analysis-001", "checkpoint_batch": 653,
           "scope": "frozen post hoc diagnostic of candidate coverage; oracle errors cannot select at inference",
           "metrics": measures,
           "source_sha256": {"evaluation_rows": hashlib.sha256(rows_bytes).hexdigest(),
                             "synthetic_panel_records": hashlib.sha256(panel_bytes).hexdigest(),
                             "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
out.mkdir(parents=True, exist_ok=False)
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"metrics": measures, "summary_sha256": hashlib.sha256((out / "summary.json").read_bytes()).hexdigest()},
                 indent=2), flush=True)
