"""Five-fold base-subject-disjoint, four-feature TRAIN-only candidate ranker."""

import hashlib
import json
from pathlib import Path

import numpy as np


root = Path(r"I:\AnatomyTracker")
source = root / "runs/observable_fit_evidence_train_assay_002"
out = root / "runs/observable_fit_ranker_crossfit_train_001"
protocol = Path(__file__).resolve().parents[1] / "docs/publication/OBSERVABLE_FIT_RANKER_CROSSFIT_001_PROTOCOL_20261009.md"
receipt = json.loads((source / "completed.json").read_text())
assert receipt["eligible"] == 96 and not receipt["trained"]
for name, digest in receipt["output_sha256"].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
rows = [json.loads(line) for line in (source / "rows.jsonl").read_text().splitlines()]
assert len(rows) == 96 and len({row["physical_section_id"] for row in rows}) == 96
base = [row["virtual_subject_id"].split("-virtual-affine-v7-")[0] for row in rows]
fold = np.array([int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little") % 5 for name in base])
assert all(len(set(np.array(base)[fold == i]) & set(np.array(base)[fold != i])) == 0 for i in range(5))
features = np.stack([np.stack((-np.asarray(row["prior"]),
                               -np.asarray(row["inherited_fitted"]),
                               np.asarray(row["rigid_hog"]),
                               np.asarray(row["warped_hog"])), -1) for row in rows])
target = np.stack([row["fitted_error_um"] for row in rows]) / 1000
predicted = np.empty((len(rows), 8))
coefficients = []
for i in range(5):
    train = fold != i
    assert train.any() and (~train).any()
    x = features[train] - features[train].mean(1, keepdims=True)
    scale = x.reshape(-1, 4).std(0).clip(1e-6)
    x = (x / scale).reshape(-1, 4)
    y = (target[train] - target[train].mean(1, keepdims=True)).reshape(-1)
    beta = np.linalg.solve(x.T @ x + 32 * np.eye(4), x.T @ y)
    held = features[~train] - features[~train].mean(1, keepdims=True)
    predicted[~train] = (held / scale) @ beta
    coefficients.append({"fold": i, "train_sections": int(train.sum()),
                         "readout_sections": int((~train).sum()),
                         "scale": scale.tolist(), "beta": beta.tolist()})

selected = predicted.argmin(1)
inherited = np.array([np.asarray(row["inherited_fitted"]).argmax() for row in rows])
choice = target[np.arange(len(rows)), selected]
control = target[np.arange(len(rows)), inherited]
oracle = target.min(1)
group_names = {"all": np.ones(len(rows), bool),
               "raw": np.array([row["mode"] == "raw" for row in rows]),
               "nonraw": np.array([row["mode"] != "raw" for row in rows]),
               "heavy_oblique": np.array([row["heavy_oblique"] for row in rows]),
               "less_oblique": np.array([not row["heavy_oblique"] for row in rows])}
summary = {name: {"sections": int(mask.sum()),
                  "crossfit_selected_mm": float(choice[mask].mean()),
                  "inherited_selected_mm": float(control[mask].mean()),
                  "best8_oracle_mm": float(oracle[mask].mean()),
                  "branch_changes": int((selected[mask] != inherited[mask]).sum())}
           for name, mask in group_names.items()}
bases = sorted(set(base))
base_equal = {name: float(np.mean([np.mean(values[np.array(base) == identity])
                                   for identity in bases]))
              for name, values in (("crossfit_selected_mm", choice),
                                   ("inherited_selected_mm", control),
                                   ("best8_oracle_mm", oracle))}
gain = base_equal["inherited_selected_mm"] - base_equal["crossfit_selected_mm"]
gate = bool(gain >= .2 and summary["raw"]["crossfit_selected_mm"] - summary["raw"]["inherited_selected_mm"] <= .2
            and summary["heavy_oblique"]["crossfit_selected_mm"] - summary["heavy_oblique"]["inherited_selected_mm"] <= .2)
out.mkdir(parents=True, exist_ok=False)
result = {"version": "observable-fit-ranker-crossfit-train-001",
          "features": ["negative_image_prior", "negative_inherited_fitted", "rigid_hog", "warped_hog"],
          "ridge": 32, "base_subjects": len(bases), "folds": coefficients,
          "case_equal": summary, "base_equal": base_equal,
          "predeclared_gate": {"base_equal_gain_mm": gain, "passes": gate},
          "source_sha256": {"rows": receipt["output_sha256"]["rows.jsonl"],
                            "completed": hashlib.sha256((source / "completed.json").read_bytes()).hexdigest(),
                            "protocol": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                            "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
          "scope": "TRAIN-only within-atlas exploratory cross-fit; not animal or deployment validation"}
(out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
with (out / "rows.jsonl").open("w", encoding="utf-8") as stream:
    for row, identity, i, predicted_row, picked, baseline in zip(rows, base, fold, predicted, selected, inherited):
        stream.write(json.dumps({"physical_section_id": row["physical_section_id"],
                                 "base_subject_id": identity, "fold": int(i),
                                 "crossfit_branch_index": int(picked), "inherited_branch_index": int(baseline),
                                 "crossfit_selected_mm": float(row["fitted_error_um"][picked] / 1000),
                                 "inherited_selected_mm": float(row["fitted_error_um"][baseline] / 1000),
                                 "score": predicted_row.tolist()}) + "\n")
(out / "completed.json").write_text(json.dumps({"sections": len(rows), "base_subjects": len(bases),
    "summary_sha256": hashlib.sha256((out / "summary.json").read_bytes()).hexdigest(),
    "rows_sha256": hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest(),
    "trained_main_model": False, "public_benchmark_used": False}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"base_equal": base_equal, "case_equal": summary, "predeclared_gate": result["predeclared_gate"]}, indent=2))
