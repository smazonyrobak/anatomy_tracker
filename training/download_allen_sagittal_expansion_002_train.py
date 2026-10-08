"""Download only the predeclared expanded sagittal TRAIN images."""

import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from PIL import Image


source = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_expansion_002_20261008")
out = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_expansion_002_train_images_20261008")
split_path = Path(r"I:\AnatomyTracker\agent_worktrees\joint_v6_integration\docs\publication\ALLEN_SAGITTAL_EXPANSION_002_SPLIT_20261008.md")
source_bytes = (source / "manifest.json").read_bytes()
assert hashlib.sha256(source_bytes).hexdigest() == "b4f77e00325273f937e4327b52d8b3f7db95458c130d4f88f682a8a08323f7bd"
candidate = json.loads(source_bytes)
assert candidate["version"] == "allen-sagittal-ish-expansion-002" and len(candidate["experiments"]) == 56 and len(candidate["sections"]) == 972
split_bytes = split_path.read_bytes()
assert hashlib.sha256(split_bytes).hexdigest() == "524684713f114c6ebccd5cb3e5f69d6ae85db4609d09d625b60e930c7a7c04b8"
weak_dev2 = {10422, 10405, 10355, 10347, 10430, 10248, 10443, 10354}
sealed_holdout = {10410, 10223, 10302, 10376, 10392, 10305, 10173, 10351}
assert weak_dev2.isdisjoint(sealed_holdout)
donor_counts = {experiment["donor_id"]: sum(section["donor_id"] == experiment["donor_id"] for section in candidate["sections"])
                for experiment in candidate["experiments"]}
ordered_multisection = sorted((donor for donor, count in donor_counts.items() if count >= 18),
                              key=lambda donor: hashlib.sha256(f"sagittal-expanded-split-20261008:{donor}".encode()).digest())
assert set(ordered_multisection[:8]) == weak_dev2 and set(ordered_multisection[8:16]) == sealed_holdout
assert sum(row["donor_id"] in weak_dev2 for row in candidate["sections"]) == 159
assert sum(row["donor_id"] in sealed_holdout for row in candidate["sections"]) == 160
experiments = [row for row in candidate["experiments"] if row["donor_id"] not in weak_dev2 | sealed_holdout]
train_rows = [row for row in candidate["sections"] if row["donor_id"] not in weak_dev2 | sealed_holdout]
train_donors = {row["donor_id"] for row in experiments}
assert len(experiments) == len(train_donors) == 40 and len(train_rows) == 653
assert len({row["section_id"] for row in train_rows}) == 653
assert {row["donor_id"] for row in train_rows} == train_donors
assert all(row["image_status"] == "not_downloaded" and row["alignment2d"] for row in train_rows)

out.mkdir(parents=True)
(out / "images").mkdir()
(out / "source_manifest.json").write_bytes(source_bytes)
(out / "frozen_split.md").write_bytes(split_bytes)
sections = []
for index, row in enumerate(train_rows, 1):
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
    if index % 50 == 0:
        print(f"downloaded and decoded {index}/653", flush=True)

manifest = {
    "version": "allen-sagittal-ish-expansion-002-train-images",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "role": "TRAIN only; excludes untouched weak DEV2 and sealed weak holdout; Allen automated affines are weak labels, not expert truth or steep-oblique validation",
    "appearance": "Raw Allen brightfield JPEG bytes; no inversion, normalization, crop, or augmentation applied",
    "rights": candidate["rights"],
    "terms_url": candidate["terms_url"],
    "citation_url": candidate["citation_url"],
    "source_manifest_path": str(source / "manifest.json"),
    "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "split_policy_commit": "d598bf7",
    "split_policy_path": str(split_path),
    "split_policy_sha256": hashlib.sha256(split_bytes).hexdigest(),
    "weak_dev2_donor_ids_not_downloaded": sorted(weak_dev2),
    "sealed_holdout_donor_ids_not_downloaded": sorted(sealed_holdout),
    "experiments": experiments,
    "sections": sections,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
receipt = {str(path.relative_to(out)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
           for path in sorted(out.rglob("*")) if path.is_file()}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "donors": len(train_donors), "experiments": len(experiments), "sections": len(sections),
                  "image_bytes": sum(row["image_bytes"] for row in sections),
                  "manifest_sha256": receipt["manifest.json"]["sha256"],
                  "receipt_sha256": hashlib.sha256((out / "receipt.json").read_bytes()).hexdigest()}, indent=2), flush=True)
