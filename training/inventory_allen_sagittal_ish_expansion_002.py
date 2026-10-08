"""Metadata-only Allen P56 sagittal ISH expansion; never downloads images."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


base = "https://api.brain-map.org/api/v2"
out = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_expansion_002_20261008")
old_path = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_train_candidates_001_20261008\manifest.json")
dev_path = Path(r"I:\AnatomyTracker\data\allen_real_development_20260928\cohort_summary.json")
old_bytes = old_path.read_bytes()
dev_bytes = dev_path.read_bytes()
old = json.loads(old_bytes)
dev_donors = {row["animal_id"] for row in json.loads(dev_bytes)["by_animal"] if row["split"] == "development_validation"}
excluded_donors = {experiment["donor_id"] for experiment in old["experiments"]} | dev_donors | {7543}
used_genes = {gene for experiment in old["experiments"] for gene in experiment["gene_symbols"]}
target_experiments = 56
max_pages = 4
page_size = 96
out.mkdir(parents=True)
(out / "api").mkdir()

experiments = []
sections = []
screened = []
catalogue_pages = []
used_donors = set()
for page in range(max_pages):
    start = page * page_size
    criteria = ("model::SectionDataSet,rma::criteria,[failed$eqfalse],[reference_space_id$eq10],"
                "[plane_of_section_id$eq2],products[id$eq1],rma::include,specimen(donor(age)),genes,"
                f"rma::options[num_rows$eq{page_size}][start_row$eq{start}][order$eq'id']")
    index_url = base + "/data/query.json?criteria=" + quote(criteria, safe="")
    with urlopen(index_url, timeout=60) as response:
        index_bytes = response.read()
    (out / "api" / f"catalogue_page_{page}.json").write_bytes(index_bytes)
    index = json.loads(index_bytes)
    assert index["success"] and len(index["msg"]) == page_size
    catalogue_pages.append({"source_url": index_url, "source_sha256": hashlib.sha256(index_bytes).hexdigest(),
                            "start_row": start, "page_size": len(index["msg"]), "matching_experiments_reported": index["total_rows"]})
    for row in index["msg"]:
        experiment_id = row["id"]
        donor_id = row["specimen"]["donor"]["id"]
        gene_symbols = [gene["acronym"] for gene in row["genes"]]
        age = row["specimen"]["donor"]["age"]["name"]
        reason = None
        if donor_id in excluded_donors:
            reason = "existing_cohort_DEV_or_reserved_donor"
        elif age != "P56":
            reason = "not_P56"
        elif donor_id in used_donors:
            reason = "duplicate_new_donor"
        elif not gene_symbols or any(gene in used_genes for gene in gene_symbols):
            reason = "duplicate_or_missing_gene"
        if reason is not None:
            screened.append({"experiment_id": experiment_id, "donor_id": donor_id, "gene_symbols": gene_symbols,
                             "age": age, "status": reason, "catalogue_page": page})
            continue
        url = base + f"/data/SectionDataSet/{experiment_id}.json?include=products,genes,treatments,specimen(donor(age)),alignment3d,section_images(alignment2d)"
        with urlopen(url, timeout=60) as response:
            raw = response.read()
        (out / "api" / f"experiment_{experiment_id}.json").write_bytes(raw)
        dataset = json.loads(raw)["msg"][0]
        images = sorted(dataset["section_images"], key=lambda image: (image["section_number"], image["id"]))
        treatments = [treatment["name"] for treatment in dataset["treatments"]]
        reason = None
        if dataset["failed"] or dataset["plane_of_section_id"] != 2 or dataset["reference_space_id"] != 10:
            reason = "not_successful_sagittal_reference_space_10"
        elif dataset["specimen"]["donor"]["id"] != donor_id or dataset["specimen"]["donor"]["age"]["name"] != "P56":
            reason = "donor_or_age_detail_mismatch"
        elif [gene["acronym"] for gene in dataset["genes"]] != gene_symbols:
            reason = "gene_detail_mismatch"
        elif 1 not in [product["id"] for product in dataset["products"]] or "ISH" not in treatments:
            reason = "not_Mouse_Brain_ISH"
        elif not dataset["alignment3d"] or not images or any(image["failed"] or not image["alignment2d"] for image in images):
            reason = "incomplete_section_or_volume_alignment"
        screened.append({"experiment_id": experiment_id, "donor_id": donor_id, "gene_symbols": gene_symbols,
                         "age": age, "status": "selected" if reason is None else reason,
                         "catalogue_page": page, "source_url": url, "source_sha256": hashlib.sha256(raw).hexdigest()})
        if reason is not None:
            continue
        used_donors.add(donor_id)
        used_genes.update(gene_symbols)
        experiments.append({
            "experiment_id": experiment_id,
            "specimen_id": dataset["specimen"]["id"],
            "donor_id": donor_id,
            "donor_name": dataset["specimen"]["donor"]["name"],
            "donor_age": "P56",
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
        if len(experiments) == target_experiments:
            break
    if len(experiments) == target_experiments:
        break

transform_url = base + "/data/ReferenceToReferenceTransform/query.json?criteria=[from_reference_space_id$eq10],[to_reference_space_id$eq9]"
with urlopen(transform_url, timeout=60) as response:
    transform_bytes = response.read()
(out / "api" / "reference_space_10_to_9.json").write_bytes(transform_bytes)
transform = json.loads(transform_bytes)["msg"][0]
assert transform["from_reference_space_id"] == 10 and transform["to_reference_space_id"] == 9
assert len(experiments) == target_experiments and len(used_donors) == target_experiments
manifest = {
    "version": "allen-sagittal-ish-expansion-002",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Metadata-only, physically sagittal P56 Allen ISH additional TRAIN candidates; automated weak atlas affines, not expert truth or oblique validation",
    "rights": "Allen Terms of Use allow noncommercial research use with citation; commercial redistribution requires permission. No raw images downloaded or cleared for redistribution.",
    "terms_url": "https://alleninstitute.org/legal/terms-of-use",
    "citation_url": "https://alleninstitute.org/legal/citation-policy",
    "existing_manifest_path": str(old_path),
    "existing_manifest_sha256": hashlib.sha256(old_bytes).hexdigest(),
    "dev_summary_path": str(dev_path),
    "dev_summary_sha256": hashlib.sha256(dev_bytes).hexdigest(),
    "excluded_donor_ids": sorted(excluded_donors),
    "excluded_existing_gene_symbols": sorted({gene for experiment in old["experiments"] for gene in experiment["gene_symbols"]}),
    "catalogue_pages": catalogue_pages,
    "reference_space_10_to_9": {"source_url": transform_url, "source_sha256": hashlib.sha256(transform_bytes).hexdigest(), "transform": transform},
    "screened": screened,
    "experiments": experiments,
    "sections": sections,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
receipt = {str(path.relative_to(out)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
           for path in sorted(out.rglob("*")) if path.is_file()}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "donors": len(used_donors), "experiments": len(experiments), "sections": len(sections),
                  "screened": len(screened), "catalogue_pages": len(catalogue_pages),
                  "treatments": sorted({treatment for experiment in experiments for treatment in experiment["treatments"]}),
                  "product_id_sets": sorted({tuple(experiment["product_ids"]) for experiment in experiments}),
                  "manifest_sha256": receipt["manifest.json"]["sha256"],
                  "receipt_sha256": hashlib.sha256((out / "receipt.json").read_bytes()).hexdigest()}, indent=2))
