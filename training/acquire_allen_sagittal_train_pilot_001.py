"""Eight physically sagittal Allen ISH sections for TRAIN-only appearance/weak-pose work."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


out = Path(r"I:\AnatomyTracker\data\allen_sagittal_train_pilot_001_20261008_run2")
dev_summary = Path(r"I:\AnatomyTracker\data\allen_real_development_20260928\cohort_summary.json")
experiment_ids = (69782969, 70813257)
chosen_indices = (1, 7, 12, 16)
base = "https://api.brain-map.org/api/v2"
out.mkdir(parents=True)
(out / "api").mkdir()
(out / "images").mkdir()

dev_bytes = dev_summary.read_bytes()
dev_donors = {row["animal_id"] for row in json.loads(dev_bytes)["by_animal"] if row["split"] == "development_validation"}
assert dev_donors == {14452, 15219, 15336, 15439, 15447, 15935}

transform_url = base + "/data/ReferenceToReferenceTransform/query.json?criteria=[from_reference_space_id$eq10],[to_reference_space_id$eq9]"
with urlopen(transform_url, timeout=60) as response:
    transform_bytes = response.read()
(out / "api" / "reference_space_10_to_9.json").write_bytes(transform_bytes)
transform = json.loads(transform_bytes)["msg"][0]
assert transform["from_reference_space_id"] == 10 and transform["to_reference_space_id"] == 9

experiments = []
sections = []
for experiment_id in experiment_ids:
    url = base + f"/data/SectionDataSet/{experiment_id}.json?include=products,genes,specimen(donor),alignment3d,section_images(alignment2d)"
    with urlopen(url, timeout=60) as response:
        raw = response.read()
    (out / "api" / f"experiment_{experiment_id}.json").write_bytes(raw)
    dataset = json.loads(raw)["msg"][0]
    donor_id = dataset["specimen"]["donor"]["id"]
    assert donor_id in {5331, 8884} and donor_id not in dev_donors
    assert dataset["failed"] is False and dataset["plane_of_section_id"] == 2 and dataset["reference_space_id"] == 10
    assert dataset["alignment3d"] and len(dataset["section_images"]) >= 17
    experiments.append({
        "experiment_id": experiment_id,
        "specimen_id": dataset["specimen"]["id"],
        "donor_id": donor_id,
        "donor_name": dataset["specimen"]["donor"]["name"],
        "gene": [gene["acronym"] for gene in dataset["genes"]],
        "product_ids": [product["id"] for product in dataset["products"]],
        "plane_of_section_id": 2,
        "reference_space_id": 10,
        "section_thickness_um": dataset["section_thickness"],
        "alignment3d": dataset["alignment3d"],
        "source_url": url,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    })
    for section in [sorted(dataset["section_images"], key=lambda row: row["section_number"])[i] for i in chosen_indices]:
        assert section["alignment2d"]
        image_url = base + f"/image_download/{section['id']}?downsample=4"
        with urlopen(image_url, timeout=60) as response:
            image_bytes = response.read()
        (out / "images" / f"{section['id']}.jpg").write_bytes(image_bytes)
        sections.append({
            "section_id": section["id"],
            "section_number": section["section_number"],
            "experiment_id": experiment_id,
            "specimen_id": dataset["specimen"]["id"],
            "donor_id": donor_id,
            "image_width_full_px": section["image_width"],
            "image_height_full_px": section["image_height"],
            "x_full_px": section["x"],
            "y_full_px": section["y"],
            "width_full_px": section["width"],
            "height_full_px": section["height"],
            "resolution_um_per_full_px": section["resolution"],
            "alignment2d": section["alignment2d"],
            "image_url": image_url,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_bytes": len(image_bytes),
        })

point_url = base + "/image_to_reference/69750516.json?x=7032&y=3648"
with urlopen(point_url, timeout=60) as response:
    point_bytes = response.read()
(out / "api" / "point_69750516_7032_3648.json").write_bytes(point_bytes)
point_api = json.loads(point_bytes)["msg"]["image_to_reference"]
section = next(row for row in sections if row["section_id"] == 69750516)
experiment = next(row for row in experiments if row["experiment_id"] == section["experiment_id"])
a2, a3 = section["alignment2d"], experiment["alignment3d"]
x, y = 7032, 3648
volume = [a2["tsv_00"] * x + a2["tsv_01"] * y + a2["tsv_04"],
          a2["tsv_02"] * x + a2["tsv_03"] * y + a2["tsv_05"],
          section["section_number"] * experiment["section_thickness_um"]]
point_local = [sum(a3[f"tvr_{3 * j + i:02d}"] * volume[i] for i in range(3)) + a3[f"tvr_{9 + j:02d}"] for j in range(3)]
point_error_um = max(abs(point_local[i] - point_api[axis]) for i, axis in enumerate("xyz"))
assert point_error_um < 1e-6

manifest = {
    "version": "allen-sagittal-train-pilot-001",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "role": "TRAIN only; physically acquired sagittal ISH appearance and weak automated affine pose, not expert ground truth or steep-oblique validation",
    "rights": "Allen Institute Terms of Use: noncommercial research use with citation; commercial redistribution requires permission. Raw images must not be committed to Git.",
    "terms_url": "https://alleninstitute.org/legal/terms-of-use",
    "citation_url": "https://alleninstitute.org/legal/citation-policy",
    "dev_summary_path": str(dev_summary),
    "dev_summary_sha256": hashlib.sha256(dev_bytes).hexdigest(),
    "dev_donor_ids": sorted(dev_donors),
    "donor_overlap_with_dev": [],
    "download_level": 4,
    "downloaded_pixel_to_full_pixel_center": "(pixel + 0.5) * 16 - 0.5",
    "reference_space_10_to_9": {"source_url": transform_url, "source_sha256": hashlib.sha256(transform_bytes).hexdigest(), "transform": transform},
    "point_check": {"source_url": point_url, "source_sha256": hashlib.sha256(point_bytes).hexdigest(), "section_id": 69750516, "full_resolution_xy": [x, y], "local_reference_space_10_xyz_um": point_local, "api_reference_space_10_xyz_um": [point_api[axis] for axis in "xyz"], "max_difference_um": point_error_um},
    "experiments": experiments,
    "sections": sections,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
receipt = {str(path.relative_to(out)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(out.rglob("*")) if path.is_file()}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "donor_ids": [row["donor_id"] for row in experiments], "section_count": len(sections), "point_error_um": point_error_um, "manifest_sha256": receipt["manifest.json"]["sha256"], "receipt_sha256": hashlib.sha256((out / "receipt.json").read_bytes()).hexdigest()}, indent=2))
