"""Donor-equal appearance sample from the full reserved Allen TRAIN pool."""

import hashlib
import json
import os
from pathlib import Path

root = Path(r"I:\AnatomyTracker")
os.environ["MPLCONFIGDIR"] = str(root / "cache/matplotlib")
os.environ["TEMP"] = os.environ["TMP"] = str(root / "tmp")

import matplotlib.pyplot as plt
import numpy as np


source = root / "data/joint_v7_reserved_train_images_192_001"
out = root / "runs/reserved_real_train_appearance_census_001"
manifest_path = source / "completed.json"
manifest = json.loads(manifest_path.read_text())
assert manifest["training_donors"] == 1885 and manifest["training_images"] == 263754
rng = np.random.default_rng(2026100905)
donors = rng.choice(sorted(manifest["donor_receipts"]), 512, replace=False)
rows, pictures = [], []
for donor in donors:
    directory = source / f"donor_{donor}"
    assert hashlib.sha256((directory / "completed.json").read_bytes()).hexdigest() == manifest["donor_receipts"][donor]
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    image_array = np.load(directory / "images.npy", mmap_mode="r")
    index = int(rng.integers(len(records)))
    record = records[index]
    assert record["training_split"] == "train" and str(record["animal_id"]) == donor
    image = np.asarray(image_array[index, 0], dtype=np.float32)
    border = np.concatenate((image[0], image[-1], image[1:-1, 0], image[1:-1, -1]))
    rows.append({"donor_id": donor, "specimen_id": record["specimen_id"],
                 "experiment_id": record["experiment_id"], "section_id": record["section_id"],
                 "array_row_index": index, "donor_receipt_sha256": manifest["donor_receipts"][donor],
                 "exact_zero_fraction": float((image == 0).mean()),
                 "near_zero_fraction_below_0_03": float((image < .03).mean()),
                 "border_fraction_above_0_03": float((border > .03).mean()),
                 "saturated_fraction_at_least_0_99": float((image >= .99).mean()),
                 "q05": float(np.quantile(image, .05)),
                 "q50": float(np.quantile(image, .50)),
                 "q95": float(np.quantile(image, .95)),
                 "mean_absolute_adjacent_difference": float((np.abs(np.diff(image, axis=0)).mean()
                                                            + np.abs(np.diff(image, axis=1)).mean()) / 2)})
    if len(pictures) < 16:
        pictures.append((record, image.copy()))

out.mkdir(parents=True, exist_ok=False)
with (out / "rows.jsonl").open("w", encoding="utf-8") as stream:
    for row in rows:
        stream.write(json.dumps(row) + "\n")
names = ("exact_zero_fraction", "near_zero_fraction_below_0_03", "border_fraction_above_0_03",
         "saturated_fraction_at_least_0_99", "q05", "q50", "q95",
         "mean_absolute_adjacent_difference")
summary = {"version": "reserved-real-train-appearance-census-001", "seed": 2026100905,
           "scope": "512 donor-distinct, one-section-per-donor TRAIN sample from 1885 donors; whole canvas, not segmented tissue or artifact labels",
           "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
           "sampled_donors": len(rows), "metrics": {name: {
               "mean": float(np.mean([row[name] for row in rows])),
               "q10_q50_q90": np.quantile([row[name] for row in rows], [.1, .5, .9]).tolist()}
               for name in names},
           "rows_sha256": hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
for ax, (record, image) in zip(axes.flat, pictures):
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"donor {record['animal_id']} | section {record['section_id']}", fontsize=8)
    ax.axis("off")
fig.tight_layout()
fig.savefig(out / "montage.png", dpi=120)
(out / "completed.json").write_text(json.dumps({"sampled_donors": len(rows),
    "rows_sha256": summary["rows_sha256"],
    "summary_sha256": hashlib.sha256((out / "summary.json").read_bytes()).hexdigest(),
    "montage_sha256": hashlib.sha256((out / "montage.png").read_bytes()).hexdigest()}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "sampled_donors": len(rows),
                  "metrics": summary["metrics"]}, indent=2), flush=True)
