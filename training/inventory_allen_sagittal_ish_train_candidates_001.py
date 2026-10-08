"""Bounded, metadata-only Allen sagittal ISH TRAIN-candidate inventory."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


out = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_train_candidates_001_20261008")
dev_path = Path(r"I:\AnatomyTracker\data\allen_real_development_20260928\cohort_summary.json")
base = "https://api.brain-map.org/api/v2"
seed_ids = (69782969, 70813257)
target_experiments = 16
reserved_donor = 7543
out.mkdir(parents=True)
(out / "api").mkdir()

dev_bytes = dev_path.read_bytes()
dev_donors = {row["animal_id"] for row in json.loads(dev_bytes)["by_animal"] if row["split"] == "development_validation"}
assert dev_donors == {14452, 15219, 15336, 15439, 15447, 15935}

criteria = ("model::SectionDataSet,rma::criteria,[failed$eqfalse],[reference_space_id$eq10],"
            "[plane_of_section_id$eq2],products[id$eq1],rma::include,specimen(donor(age)),genes,"
            "rma::options[num_rows$eq96][start_row$eq0][order$eq'id']")
index_url = base + "/data/query.json?criteria=" + quote(criteria, safe="")
with urlopen(index_url, timeout=60) as response:
    index_bytes = response.read()
(out / "api" / "catalogue_page_0.json").write_bytes(index_bytes)
index = json.loads(index_bytes)
assert index["success"] and len(index["msg"]) == 96

transform_url = base + "/data/ReferenceToReferenceTransform/query.json?criteria=[from_reference_space_id$eq10],[to_reference_space_id$eq9]"
with urlopen(transform_url, timeout=60) as response:
    transform_bytes = response.read()
(out / "api" / "reference_space_10_to_9.json").write_bytes(transform_bytes)
transform = json.loads(transform_bytes)["msg"][0]
assert transform["from_reference_space_id"] == 10 and transform["to_reference_space_id"] == 9

ids = list(seed_ids) + [row["id"] for row in index["msg"] if row["id"] not in seed_ids]
experiments = []
sections = []
screened = []
donors = set()
genes = set()
for experiment_id in ids:
    if len(experiments) == target_experiments:
        break
    url = base + f"/data/SectionDataSet/{experiment_id}.json?include=products,genes,treatments,specimen(donor(age)),alignment3d,section_images(alignment2d)"
    with urlopen(url, timeout=60) as response:
        raw = response.read()
    (out / "api" / f"experiment_{experiment_id}.json").write_bytes(raw)
    dataset = json.loads(raw)["msg"][0]
    donor = dataset["specimen"]["donor"]
    donor_id = donor["id"]
    gene_symbols = [gene["acronym"] for gene in dataset["genes"]]
    treatments = [treatment["name"] for treatment in dataset["treatments"]]
    images = sorted(dataset["section_images"], key=lambda image: (image["section_number"], image["id"]))
    reason = None
    if donor_id in dev_donors or donor_id == reserved_donor:
        reason = "frozen_DEV_or_reserved_donor"
    elif donor_id in donors:
        reason = "duplicate_donor"
    elif any(gene in genes for gene in gene_symbols):
        reason = "duplicate_gene"
    elif dataset["failed"] or dataset["plane_of_section_id"] != 2 or dataset["reference_space_id"] != 10:
        reason = "not_successful_adult_sagittal_reference_space_10"
    elif 1 not in [product["id"] for product in dataset["products"]] or "ISH" not in treatments:
        reason = "not_Mouse_Brain_ISH"
    elif not dataset["alignment3d"] or not images or any(image["failed"] or not image["alignment2d"] for image in images):
        reason = "incomplete_section_or_volume_alignment"
    screened.append({"experiment_id": experiment_id, "donor_id": donor_id, "status": "selected" if reason is None else reason,
                     "source_url": url, "source_sha256": hashlib.sha256(raw).hexdigest()})
    if reason is not None:
        continue
    donors.add(donor_id)
    genes.update(gene_symbols)
    experiments.append({
        "experiment_id": experiment_id,
        "specimen_id": dataset["specimen"]["id"],
        "donor_id": donor_id,
        "donor_name": donor["name"],
        "donor_age": donor["age"]["name"],
        "gene_symbols": gene_symbols,
        "treatments": treatments,
        "product_ids": sorted(product["id"] for product in dataset["products"]),
        "plane_of_section_id": dataset["plane_of_section_id"],
        "reference_space_id": dataset["reference_space_id"],
        "section_thickness_um": dataset["section_thickness"],
        "alignment3d": dataset["alignment3d"],
        "source_url": url,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    })
    sections.extend({
        "section_id": image["id"],
        "section_number": image["section_number"],
        "experiment_id": experiment_id,
        "specimen_id": dataset["specimen"]["id"],
        "donor_id": donor_id,
        "image_width_full_px": image["image_width"],
        "image_height_full_px": image["image_height"],
        "resolution_um_per_full_px": image["resolution"],
        "alignment2d": image["alignment2d"],
        "possible_image_url": base + f"/image_download/{image['id']}?downsample=4",
        "image_status": "not_downloaded",
    } for image in images)

assert len(experiments) == target_experiments and len(donors) == target_experiments
assert donors.isdisjoint(dev_donors | {reserved_donor})
manifest = {
    "version": "allen-sagittal-ish-train-candidates-001",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "TRAIN candidates only; metadata-only physically sagittal ISH, weak Allen automated atlas affines, not expert ground truth or steep-oblique validation",
    "rights": "Allen Terms of Use allow noncommercial research use with citation; commercial redistribution requires permission. Raw images are not included or cleared for redistribution.",
    "terms_url": "https://alleninstitute.org/legal/terms-of-use",
    "citation_url": "https://alleninstitute.org/legal/citation-policy",
    "dev_summary_path": str(dev_path),
    "dev_summary_sha256": hashlib.sha256(dev_bytes).hexdigest(),
    "excluded_dev_donor_ids": sorted(dev_donors),
    "reserved_pvalb_donor_id": reserved_donor,
    "catalogue_page": {"source_url": index_url, "source_sha256": hashlib.sha256(index_bytes).hexdigest(), "matching_experiments_reported": index["total_rows"], "page_size": len(index["msg"])},
    "reference_space_10_to_9": {"source_url": transform_url, "source_sha256": hashlib.sha256(transform_bytes).hexdigest(), "transform": transform},
    "screened": screened,
    "experiments": experiments,
    "sections": sections,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
receipt = {str(path.relative_to(out)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(out.rglob("*")) if path.is_file()}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "donors": len(donors), "experiments": len(experiments), "sections": len(sections), "screened": len(screened),
                  "treatments": sorted({treatment for experiment in experiments for treatment in experiment["treatments"]}),
                  "product_id_sets": sorted({tuple(experiment["product_ids"]) for experiment in experiments}),
                  "manifest_sha256": receipt["manifest.json"]["sha256"],
                  "receipt_sha256": hashlib.sha256((out / "receipt.json").read_bytes()).hexdigest()}, indent=2))
