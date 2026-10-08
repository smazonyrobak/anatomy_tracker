"""TRAIN-only full-canvas appearance census after the model's real-image preparation."""

import hashlib
import json
from pathlib import Path

import numpy as np


root = Path(r"I:\AnatomyTracker")
coronal = root / "data/joint_v7_allen_fullcanvas_192_001"
sagittal = root / "data/allen_sagittal_ish_expansion_002_train_inputs_20261008"
out = root / "runs/acquired_train_appearance_census_001"
coronal_records = [json.loads(line) for line in (coronal / "records.jsonl").read_text().splitlines()]
sagittal_records = [json.loads(line) for line in (sagittal / "geometry.jsonl").read_text().splitlines()]
coronal_train = [row for row in coronal_records if row["training_split"] == "train"]
assert len(coronal_train) == 1280 and len(sagittal_records) == 653
assert len({row["animal_id"] for row in coronal_train}) == 58
assert len({row["donor_id"] for row in sagittal_records}) == 40
assert all(row["split"] == "train" for row in sagittal_records)
assert len({row["section_id"] for row in sagittal_records}) == 653
arrays = (("coronal", np.load(coronal / "images.npy", mmap_mode="r"), coronal_train, "animal_id"),
          ("sagittal", np.load(sagittal / "model_input.npy", mmap_mode="r"), sagittal_records, "donor_id"))
summary = {}
for domain, array, records, identity in arrays:
    image = np.asarray(array[[row["array_row_index"] for row in records], 0], dtype=np.float32)
    values = {
        "exact_zero_fraction": (image == 0).mean((1, 2)),
        "near_zero_fraction_below_0_03": (image < .03).mean((1, 2)),
        "saturated_fraction_at_least_0_99": (image >= .99).mean((1, 2)),
        "q05": np.percentile(image, 5, axis=(1, 2)),
        "q50": np.percentile(image, 50, axis=(1, 2)),
        "q95": np.percentile(image, 95, axis=(1, 2)),
        "mean_absolute_adjacent_difference": (
            np.abs(np.diff(image, axis=1)).mean((1, 2)) +
            np.abs(np.diff(image, axis=2)).mean((1, 2))) / 2,
    }
    ids = sorted({row[identity] for row in records})
    donors = {str(donor): {name: float(np.mean(value[[row[identity] == donor for row in records]]))
                            for name, value in values.items()} for donor in ids}
    summary[domain] = {
        "images": len(records), "donors": len(ids), "shape_h_w": list(image.shape[-2:]),
        "mean_over_images": {name: float(value.mean()) for name, value in values.items()},
        "equal_donor_mean": {name: float(np.mean([donors[str(donor)][name] for donor in ids]))
                             for name in values},
        "donor_tenth_ninetieth_percentiles": {name: np.percentile(
            [donors[str(donor)][name] for donor in ids], [10, 90]).tolist() for name in values},
        "per_donor": donors,
    }
out.mkdir(parents=True, exist_ok=False)
source = {"coronal_records": hashlib.sha256((coronal / "records.jsonl").read_bytes()).hexdigest(),
          "coronal_images": hashlib.sha256((coronal / "images.npy").read_bytes()).hexdigest(),
          "sagittal_geometry": hashlib.sha256((sagittal / "geometry.jsonl").read_bytes()).hexdigest(),
          "sagittal_images": hashlib.sha256((sagittal / "model_input.npy").read_bytes()).hexdigest(),
          "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
result = {"version": "acquired-train-appearance-census-001", "scope": "TRAIN only; whole canvas, not segmented tissue or artifact annotation",
          "warning": "Coronal 192 and sagittal 256 images have different stains, fields and preparation; adjacent-pixel contrasts are not directly comparable.",
          "source_sha256": source, "summary": summary}
(out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "summary_sha256": hashlib.sha256((out / "summary.json").read_bytes()).hexdigest(),
                  "equal_donor_mean": {domain: data["equal_donor_mean"] for domain, data in summary.items()}}, indent=2), flush=True)
