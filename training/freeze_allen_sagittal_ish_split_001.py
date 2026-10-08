"""Freeze the predeclared donor split beside immutable sagittal acquisition files."""

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


repo = Path(__file__).resolve().parents[1]
root = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_train_images_001_20261008")
policy = "a2eb315:docs/publication/SAGITTAL_WEAK_COHORT_SPLIT_20261008.md"
policy_bytes = subprocess.check_output(["git", "show", policy], cwd=repo)
assert hashlib.sha256(policy_bytes).hexdigest() == "44bed4961269d7889481db79a752a6fcf1f9955324b962fa930bbe6e49318f3a"
manifest_bytes = (root / "manifest.json").read_bytes()
receipt_bytes = (root / "receipt.json").read_bytes()
acquisition = json.loads(manifest_bytes)
receipt = json.loads(receipt_bytes)
assert hashlib.sha256(manifest_bytes).hexdigest() == receipt["manifest.json"]["sha256"]
assert acquisition["version"] == "allen-sagittal-ish-train-images-001" and len(acquisition["sections"]) == 240
assert len(receipt) == 243 and all(f"images/{row['section_id']}.jpg" in receipt for row in acquisition["sections"])

counts = Counter(row["donor_id"] for row in acquisition["sections"])
eligible = [donor for donor, count in counts.items() if count >= 18 and donor not in {5331, 8884}]
weak_dev = set(sorted(eligible, key=lambda donor: hashlib.sha256(f"sagittal-weak-dev-20261008:{donor}".encode()).hexdigest())[:4])
assert weak_dev == {10230, 10275, 10186, 10409}
train = set(counts) - weak_dev
assert train == {5331, 8884, 10174, 10228, 10350, 10352, 10353, 10356, 10357, 10360, 10421, 10437}
assert sum(counts[donor] for donor in weak_dev) == 80 and sum(counts[donor] for donor in train) == 160
assert (train | weak_dev).isdisjoint(set(acquisition["dev_donor_ids"]) | {acquisition["reserved_pvalb_donor_id"]})

split = {
    "version": "allen-sagittal-ish-frozen-weak-split-001",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "role": "Normative frozen donor split for acquired physical sagittal sections; Allen affines are weak references, not expert ground truth or steep-oblique validation",
    "acquisition_manifest_train_only_label_note": "The immutable acquisition manifest used the pre-split TRAIN-candidate-pool label; this split manifest supersedes that role assignment without changing image bytes or identities.",
    "policy_commit": "a2eb315",
    "policy_path": policy,
    "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
    "acquisition_manifest_path": str(root / "manifest.json"),
    "acquisition_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "acquisition_receipt_path": str(root / "receipt.json"),
    "acquisition_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    "source_inventory_manifest_sha256": acquisition["source_manifest_sha256"],
    "reference_space_10_to_9": acquisition["reference_space_10_to_9"],
    "train_donor_ids": sorted(train),
    "weak_dev_donor_ids": sorted(weak_dev),
    "product5_dev_donor_ids": acquisition["dev_donor_ids"],
    "reserved_pvalb_donor_id": acquisition["reserved_pvalb_donor_id"],
    "counts": {"train": {"donors": len(train), "sections": 160}, "weak_dev": {"donors": len(weak_dev), "sections": 80}},
    "experiments": [{**row, "split": "weak_dev" if row["donor_id"] in weak_dev else "train"} for row in acquisition["experiments"]],
    "sections": [{**row, "split": "weak_dev" if row["donor_id"] in weak_dev else "train"} for row in acquisition["sections"]],
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
split_path = root / "split_manifest.json"
split_receipt_path = root / "split_receipt.json"
assert not split_path.exists() and not split_receipt_path.exists()
split_path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
split_receipt = {
    "split_manifest.json": {"bytes": split_path.stat().st_size, "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest()},
    "acquisition_manifest.json": {"bytes": len(manifest_bytes), "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
    "acquisition_receipt.json": {"bytes": len(receipt_bytes), "sha256": hashlib.sha256(receipt_bytes).hexdigest()},
    "frozen_split_policy.md": {"bytes": len(policy_bytes), "sha256": hashlib.sha256(policy_bytes).hexdigest()},
}
split_receipt_path.write_text(json.dumps(split_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"split_manifest_sha256": split_receipt["split_manifest.json"]["sha256"],
                  "split_receipt_sha256": hashlib.sha256(split_receipt_path.read_bytes()).hexdigest(),
                  "train_sections": 160, "weak_dev_sections": 80}, indent=2))
