"""Acquire a bounded donor-separated Allen appearance-development cohort on I:."""

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from training.allen_real_histology_metadata import acquire_metadata_snapshot, verify_metadata_snapshot
from training.acquire_allen_real_histology_images import acquire_image_snapshot

ROOT = Path(r"I:\AnatomyTracker\data\allen_real_development_20260928")
ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = r"I:\AnatomyTracker\tmp"
os.environ["TMP"] = r"I:\AnatomyTracker\tmp"
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=(500, 502, 503, 504), allowed_methods=("GET",))))
exclusion_audit = {
    "scope": "appearance training and internal development only; not untouched or final generalization evidence",
    "historical_source_review": ["docs/publication/STUDY_PROTOCOL.md", "docs/publication/BASELINE_LEDGER.md", "training/evaluate_real_ground_truth.py"],
    "historical_exposure": "Public DeepSlice brains and local1400-section comparison cohort were previously consumed; no joinable Allen Donor.id exclusion list was available in the I: project data/manifests inspected.",
    "public_benchmark_image_or_label_access": False,
    "final_test_role": False,
    "named_public_brains_not_used_as_training_sources": ["CamKII", "GLT1a", "PcP2", "Myelin", "Pitx3", "Calb1", "bAmyloid"],
    "partition_rule": "Existing Allen Donor.id hash split; inherited by all sections and experiments; donor round-robin image selection",
}
(ROOT / "exclusion_scope.json").write_text(json.dumps(exclusion_audit, indent=2), encoding="utf-8")
print("Acquiring metadata for64official Allen Product5experiments", flush=True)
if (ROOT / "metadata" / "receipt.json").exists():
    verify_metadata_snapshot(ROOT / "metadata")
    metadata = json.loads((ROOT / "metadata" / "manifest.json").read_text(encoding="utf-8"))
else:
    metadata = acquire_metadata_snapshot(ROOT / "metadata", 64, get=session.get)
print(json.dumps({"metadata_counts": metadata["counts"]}), flush=True)
images = acquire_image_snapshot(
    ROOT / "metadata", ROOT / "images",
    {"development_train": 256, "development_validation": 64}, get=session.get,
)
records = [json.loads(line) for line in (ROOT / "images" / "images.jsonl").read_text(encoding="utf-8").splitlines()]
by_animal = Counter((row["split"], row["animal_id"]) for row in records)
summary = {
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": exclusion_audit["scope"],
    "metadata_counts": metadata["counts"], "image_counts": images["counts"],
    "downloaded_image_bytes": sum(row["bytes"] for row in records),
    "by_animal": [{"split": split, "animal_id": animal, "section_count": count} for (split, animal), count in sorted(by_animal.items())],
    "metadata_receipt_file_sha256": hashlib.sha256((ROOT / "metadata" / "receipt.json").read_bytes()).hexdigest(),
    "images_receipt_file_sha256": hashlib.sha256((ROOT / "images" / "receipt.json").read_bytes()).hexdigest(),
    "cohort_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(ROOT / "cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary), flush=True)
