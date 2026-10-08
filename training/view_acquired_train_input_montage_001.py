"""Fixed-seed, donor-distinct view of acquired TRAIN inputs after actual preprocessing."""

import json
import os
from pathlib import Path

root = Path(r"I:\AnatomyTracker")
os.environ["MPLCONFIGDIR"] = str(root / "cache/matplotlib")
os.environ["TEMP"] = os.environ["TMP"] = str(root / "tmp")

import matplotlib.pyplot as plt
import numpy as np


coronal = root / "data/joint_v7_allen_fullcanvas_192_001"
sagittal = root / "data/allen_sagittal_ish_expansion_002_train_inputs_20261008"
out = root / "reports/acquired_train_input_montage_001"
coronal_records = [json.loads(line) for line in (coronal / "records.jsonl").read_text().splitlines()
                   if json.loads(line)["training_split"] == "train"]
sagittal_records = [json.loads(line) for line in (sagittal / "geometry.jsonl").read_text().splitlines()]
coronal_images = np.load(coronal / "images.npy", mmap_mode="r")
sagittal_images = np.load(sagittal / "model_input.npy", mmap_mode="r")
rng = np.random.default_rng(2026100902)
chosen = []
for domain, records, key, images in (("coronal", coronal_records, "animal_id", coronal_images),
                                     ("sagittal", sagittal_records, "donor_id", sagittal_images)):
    donors = rng.choice(sorted({row[key] for row in records}), 8, replace=False)
    for donor in donors:
        cohort = [row for row in records if row[key] == donor]
        record = cohort[int(rng.integers(len(cohort)))]
        chosen.append((domain, record, np.asarray(images[record["array_row_index"], 0], dtype=np.float32)))
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
for ax, (domain, record, image) in zip(axes.flat, chosen):
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{domain} | donor {record['animal_id'] if domain == 'coronal' else record['donor_id']}\nsection {record['section_id']}", fontsize=8)
    ax.axis("off")
fig.tight_layout()
out.mkdir(parents=True, exist_ok=False)
fig.savefig(out / "montage.png", dpi=120)
(out / "selected.json").write_text(json.dumps([{"domain": domain,
    "donor_id": record["animal_id"] if domain == "coronal" else record["donor_id"],
    "section_id": record["section_id"], "array_row_index": record["array_row_index"]}
    for domain, record, _ in chosen], indent=2) + "\n", encoding="utf-8")
print(out / "montage.png")
