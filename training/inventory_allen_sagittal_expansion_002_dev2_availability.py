"""Audit the frozen DEV2 image URLs without retaining or viewing image pixels."""

import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from PIL import Image


source = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_expansion_002_20261008")
out = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_expansion_002_dev2_availability_20261008")
split_path = Path(r"I:\AnatomyTracker\agent_worktrees\joint_v6_integration\docs\publication\ALLEN_SAGITTAL_EXPANSION_002_SPLIT_20261008.md")
source_bytes = (source / "manifest.json").read_bytes()
split_bytes = split_path.read_bytes()
assert hashlib.sha256(source_bytes).hexdigest() == "b4f77e00325273f937e4327b52d8b3f7db95458c130d4f88f682a8a08323f7bd"
assert hashlib.sha256(split_bytes).hexdigest() == "524684713f114c6ebccd5cb3e5f69d6ae85db4609d09d625b60e930c7a7c04b8"
candidate = json.loads(source_bytes)
weak_dev2 = {10422, 10405, 10355, 10347, 10430, 10248, 10443, 10354}
sealed_holdout = {10410, 10223, 10302, 10376, 10392, 10305, 10173, 10351}
donor_counts = {experiment["donor_id"]: sum(section["donor_id"] == experiment["donor_id"] for section in candidate["sections"])
                for experiment in candidate["experiments"]}
ordered_multisection = sorted((donor for donor, count in donor_counts.items() if count >= 18),
                              key=lambda donor: hashlib.sha256(f"sagittal-expanded-split-20261008:{donor}".encode()).digest())
assert set(ordered_multisection[:8]) == weak_dev2 and set(ordered_multisection[8:16]) == sealed_holdout
rows = [row for row in candidate["sections"] if row["donor_id"] in weak_dev2]
assert len(rows) == len({row["section_id"] for row in rows}) == 159
assert {row["donor_id"] for row in rows} == weak_dev2 and not {row["donor_id"] for row in rows} & sealed_holdout

out.mkdir(parents=True)
(out / "source_manifest.json").write_bytes(source_bytes)
(out / "frozen_split.md").write_bytes(split_bytes)
availability = []
for index, row in enumerate(rows, 1):
    url = row["possible_image_url"]
    with urlopen(url, timeout=90) as response:
        body = response.read()
        status = response.status
        response_url = response.geturl()
        content_type = response.headers.get("Content-Type")
    jpeg_signature = body[:2] == b"\xff\xd8"
    decoded = False
    decoded_size = None
    decoded_mode = None
    reason = "not_jpeg_signature"
    if jpeg_signature:
        try:
            with Image.open(BytesIO(body)) as image:
                decoded = image.format == "JPEG"
                image.load()
                decoded_size = list(image.size)
                decoded_mode = image.mode
            reason = "decoded_jpeg" if decoded else "unexpected_decoded_format"
        except OSError as error:
            reason = f"jpeg_decode_error:{type(error).__name__}"
    availability.append({
        "section_id": row["section_id"], "donor_id": row["donor_id"],
        "specimen_id": row["specimen_id"], "experiment_id": row["experiment_id"],
        "section_number": row["section_number"], "source_url": url,
        "http_status": status, "response_url": response_url, "content_type": content_type,
        "body_bytes": len(body), "body_sha256": hashlib.sha256(body).hexdigest(),
        "body_prefix_hex": body[:32].hex(), "jpeg_signature": jpeg_signature,
        "available_decoded_jpeg": decoded, "reason": reason,
        "decoded_size_px": decoded_size, "decoded_mode": decoded_mode,
    })
    if index % 25 == 0:
        print(f"audited {index}/159", flush=True)

manifest = {
    "version": "allen-sagittal-ish-expansion-002-dev2-availability",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Frozen eight-donor weak DEV2 only; response/body signatures and decode status, no image bytes retained and no image preview; sealed weak holdout not requested",
    "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "split_policy_commit": "d598bf7",
    "split_policy_sha256": hashlib.sha256(split_bytes).hexdigest(),
    "weak_dev2_donor_ids": sorted(weak_dev2),
    "sealed_holdout_donor_ids_not_requested": sorted(sealed_holdout),
    "rights": candidate["rights"], "terms_url": candidate["terms_url"], "citation_url": candidate["citation_url"],
    "sections": availability,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
receipt = {str(path.relative_to(out)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
           for path in sorted(out.rglob("*")) if path.is_file()}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
valid = [row for row in availability if row["available_decoded_jpeg"]]
print(json.dumps({"output": str(out), "valid": len(valid), "unavailable": len(availability) - len(valid),
                  "valid_by_donor": {str(donor): sum(row["donor_id"] == donor for row in valid) for donor in sorted(weak_dev2)},
                  "unavailable_section_ids": [row["section_id"] for row in availability if not row["available_decoded_jpeg"]],
                  "manifest_sha256": receipt["manifest.json"]["sha256"],
                  "receipt_sha256": hashlib.sha256((out / "receipt.json").read_bytes()).hexdigest()}, indent=2), flush=True)
