"""Acquire only the source-available frozen DEV2 SectionImages."""

import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from PIL import Image


source = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_expansion_002_20261008")
availability_dir = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_expansion_002_dev2_availability_20261008")
out = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_expansion_002_dev2_images_available_20261008")
source_bytes = (source / "manifest.json").read_bytes()
availability_bytes = (availability_dir / "manifest.json").read_bytes()
availability_receipt_bytes = (availability_dir / "receipt.json").read_bytes()
assert hashlib.sha256(source_bytes).hexdigest() == "b4f77e00325273f937e4327b52d8b3f7db95458c130d4f88f682a8a08323f7bd"
assert hashlib.sha256(availability_bytes).hexdigest() == "8cec8d29c4d177073e58b8bc2871e849c1051620a4d00eb88d636017f402066d"
assert hashlib.sha256(availability_receipt_bytes).hexdigest() == "975707c8ca6a2d60588c6b571052d2dea90c4fe016114b76cc1e38cb12fe38b2"
candidate = json.loads(source_bytes)
availability = json.loads(availability_bytes)
valid = [row for row in availability["sections"] if row["available_decoded_jpeg"]]
unavailable = [row for row in availability["sections"] if not row["available_decoded_jpeg"]]
assert len(valid) == 158 and len(unavailable) == 1 and unavailable[0]["section_id"] == 101345593
valid_ids = {row["section_id"] for row in valid}
valid_by_id = {row["section_id"]: row for row in valid}
dev_donors = set(availability["weak_dev2_donor_ids"])
sealed_holdout = set(availability["sealed_holdout_donor_ids_not_requested"])
experiments = [row for row in candidate["experiments"] if row["donor_id"] in dev_donors]
rows = [row for row in candidate["sections"] if row["section_id"] in valid_ids]
assert len(experiments) == 8 and len(rows) == len(valid_ids) == 158
assert {row["donor_id"] for row in rows} == dev_donors and not {row["donor_id"] for row in rows} & sealed_holdout
assert min(sum(row["donor_id"] == donor for row in rows) for donor in dev_donors) >= 10
assert all(row["possible_image_url"] == valid_by_id[row["section_id"]]["source_url"] for row in rows)

out.mkdir(parents=True)
(out / "images").mkdir()
(out / "source_manifest.json").write_bytes(source_bytes)
(out / "availability_manifest.json").write_bytes(availability_bytes)
(out / "availability_receipt.json").write_bytes(availability_receipt_bytes)
sections = []
for index, row in enumerate(rows, 1):
    url = row["possible_image_url"]
    with urlopen(url, timeout=90) as response:
        image_bytes = response.read()
        response_url = response.geturl()
        content_type = response.headers.get("Content-Type")
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    assert image_sha256 == valid_by_id[row["section_id"]]["body_sha256"]
    with Image.open(BytesIO(image_bytes)) as image:
        assert image.format == "JPEG"
        image.load()
        decoded_size = list(image.size)
        decoded_mode = image.mode
    image_path = out / "images" / f"{row['section_id']}.jpg"
    image_path.write_bytes(image_bytes)
    sections.append({**row, "image_status": "downloaded_and_decoded", "image_path": f"images/{row['section_id']}.jpg",
                     "image_url": url, "response_url": response_url, "content_type": content_type,
                     "image_sha256": image_sha256, "image_bytes": len(image_bytes),
                     "decoded_size_px": decoded_size, "decoded_mode": decoded_mode,
                     "availability_body_sha256": valid_by_id[row["section_id"]]["body_sha256"]})
    if index % 50 == 0:
        print(f"downloaded and decoded {index}/158", flush=True)

manifest = {
    "version": "allen-sagittal-ish-expansion-002-dev2-available-images",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "role": "frozen weak DEV2 source-available subset only; one pre-readout source-unavailable SectionImage excluded, no substitute, sealed holdout not requested",
    "appearance": "Raw Allen brightfield JPEG bytes; no inversion, normalization, crop, or augmentation applied",
    "rights": candidate["rights"], "terms_url": candidate["terms_url"], "citation_url": candidate["citation_url"],
    "source_metadata_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "availability_manifest_sha256": hashlib.sha256(availability_bytes).hexdigest(),
    "availability_receipt_sha256": hashlib.sha256(availability_receipt_bytes).hexdigest(),
    "weak_dev2_donor_ids": sorted(dev_donors),
    "sealed_holdout_donor_ids_not_downloaded": sorted(sealed_holdout),
    "unavailable_source_sections_not_downloaded": unavailable,
    "experiments": experiments,
    "sections": sections,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
receipt = {str(path.relative_to(out)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
           for path in sorted(out.rglob("*")) if path.is_file()}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "donors": len(experiments), "sections": len(sections),
                  "unavailable_source_section_ids": [row["section_id"] for row in unavailable],
                  "image_bytes": sum(row["image_bytes"] for row in sections),
                  "manifest_sha256": receipt["manifest.json"]["sha256"],
                  "receipt_sha256": hashlib.sha256((out / "receipt.json").read_bytes()).hexdigest()}, indent=2), flush=True)
