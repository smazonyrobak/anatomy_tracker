"""Download the frozen 240-section Allen sagittal TRAIN candidate cohort."""

import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from PIL import Image


source = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_train_candidates_001_20261008")
out = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_train_images_001_20261008")
source_bytes = (source / "manifest.json").read_bytes()
source_sha256 = hashlib.sha256(source_bytes).hexdigest()
assert source_sha256 == "80c16b11d96236269eb2b29fda6b212d8559797e279650a00cef25208646b0e1"
candidate = json.loads(source_bytes)
assert candidate["version"] == "allen-sagittal-ish-train-candidates-001"
assert len(candidate["experiments"]) == 16 and len(candidate["sections"]) == 240

dev_path = Path(candidate["dev_summary_path"])
dev_bytes = dev_path.read_bytes()
assert hashlib.sha256(dev_bytes).hexdigest() == candidate["dev_summary_sha256"]
dev_donors = {row["animal_id"] for row in json.loads(dev_bytes)["by_animal"] if row["split"] == "development_validation"}
assert dev_donors == set(candidate["excluded_dev_donor_ids"]) == {14452, 15219, 15336, 15439, 15447, 15935}
assert candidate["reserved_pvalb_donor_id"] == 7543

experiments = {row["experiment_id"]: row for row in candidate["experiments"]}
donors = {row["donor_id"] for row in candidate["experiments"]}
assert len(experiments) == len(donors) == 16 and donors.isdisjoint(dev_donors | {7543})
assert len({row["section_id"] for row in candidate["sections"]}) == 240
assert all(row["donor_age"] == "P56" and "ISH" in row["treatments"] and 1 in row["product_ids"]
           and row["plane_of_section_id"] == 2 and row["reference_space_id"] == 10 and row["alignment3d"]
           for row in candidate["experiments"])
assert all(row["experiment_id"] in experiments and row["donor_id"] == experiments[row["experiment_id"]]["donor_id"]
           and row["specimen_id"] == experiments[row["experiment_id"]]["specimen_id"]
           and row["image_status"] == "not_downloaded" and row["alignment2d"]
           and row["possible_image_url"] == f"https://api.brain-map.org/api/v2/image_download/{row['section_id']}?downsample=4"
           for row in candidate["sections"])

transform_bytes = (source / "api" / "reference_space_10_to_9.json").read_bytes()
assert hashlib.sha256(transform_bytes).hexdigest() == candidate["reference_space_10_to_9"]["source_sha256"]
assert json.loads(transform_bytes)["msg"][0] == candidate["reference_space_10_to_9"]["transform"]

out.mkdir(parents=True)
(out / "images").mkdir()
(out / "source_manifest.json").write_bytes(source_bytes)
(out / "reference_space_10_to_9.json").write_bytes(transform_bytes)

sections = []
for index, row in enumerate(candidate["sections"], 1):
    url = row["possible_image_url"]
    with urlopen(url, timeout=90) as response:
        image_bytes = response.read()
        response_url = response.geturl()
        content_type = response.headers.get("Content-Type")
    with Image.open(BytesIO(image_bytes)) as image:
        assert image.format == "JPEG"
        image.load()
        decoded_size = list(image.size)
        decoded_mode = image.mode
    image_path = out / "images" / f"{row['section_id']}.jpg"
    image_path.write_bytes(image_bytes)
    sections.append({**row, "image_status": "downloaded_and_decoded", "image_path": f"images/{row['section_id']}.jpg",
                     "image_url": url, "response_url": response_url, "content_type": content_type,
                     "image_sha256": hashlib.sha256(image_bytes).hexdigest(), "image_bytes": len(image_bytes),
                     "decoded_size_px": decoded_size, "decoded_mode": decoded_mode})
    if index % 20 == 0:
        print(f"downloaded and decoded {index}/240", flush=True)

manifest = {
    "version": "allen-sagittal-ish-train-images-001",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "role": "TRAIN only; raw physically sagittal ISH JPEGs with weak Allen automated affines, not expert ground truth or steep-oblique validation",
    "appearance": "Raw Allen brightfield JPEG bytes; no inversion, normalization, crop, or augmentation applied",
    "rights": candidate["rights"],
    "terms_url": candidate["terms_url"],
    "citation_url": candidate["citation_url"],
    "source_manifest_path": str(source / "manifest.json"),
    "source_manifest_sha256": source_sha256,
    "dev_summary_path": str(dev_path),
    "dev_summary_sha256": hashlib.sha256(dev_bytes).hexdigest(),
    "dev_donor_ids": sorted(dev_donors),
    "reserved_pvalb_donor_id": 7543,
    "donor_overlap_with_dev_or_reserved": [],
    "reference_space_10_to_9": candidate["reference_space_10_to_9"],
    "experiments": candidate["experiments"],
    "sections": sections,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
receipt = {str(path.relative_to(out)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
           for path in sorted(out.rglob("*")) if path.is_file()}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "donors": len(donors), "experiments": len(experiments), "sections": len(sections),
                  "image_bytes": sum(row["image_bytes"] for row in sections),
                  "manifest_sha256": receipt["manifest.json"]["sha256"],
                  "receipt_sha256": hashlib.sha256((out / "receipt.json").read_bytes()).hexdigest()}, indent=2), flush=True)
