import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from source.registered_image_quality import (
    build_registered_image_quality_manifest,
    load_registered_image_quality_manifest,
    registered_image_quality_metrics,
    registered_image_rejection_reason,
)


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_quality_rule_has_a_wide_margin_between_blank_and_low_contrast_tissue():
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    blank[::80, ::80] = 3
    tissue = np.full((240, 320, 3), 2, dtype=np.uint8)
    yy, xx = np.mgrid[:240, :320]
    inside = ((xx - 160.0) / 115.0) ** 2 + ((yy - 125.0) / 85.0) ** 2 < 1.0
    tissue[inside, 2] = np.clip(34.0 + 9.0 * np.sin(xx[inside] / 12.0), 0, 255)

    blank_metrics = registered_image_quality_metrics(blank)
    tissue_metrics = registered_image_quality_metrics(tissue)
    assert registered_image_rejection_reason(blank_metrics) == "uninformative_intensity"
    assert registered_image_rejection_reason(tissue_metrics) is None
    assert blank_metrics["robust_span"] < 4.0
    assert tissue_metrics["robust_span"] > 12.0


def test_manifest_excludes_blank_without_changing_record_identity_and_binds_source(tmp_path):
    records = [
        {
            "section_image_id": section_id,
            "split": split,
            "relative_path": f"images/{split}/{section_id}.jpg",
        }
        for section_id, split in ((11, "train"), (22, "sealed_deepslice_s2p"))
    ]
    informative = np.tile(np.arange(128, dtype=np.uint8), (96, 1))
    blank = np.zeros((96, 128), dtype=np.uint8)
    for record, image in zip(records, (informative, blank)):
        path = tmp_path / record["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path, format="JPEG", quality=100, subsampling=0)
    write_jsonl(tmp_path / "sections.jsonl", records)
    write_jsonl(
        tmp_path / "downloads.jsonl",
        [
            {
                "section_image_id": record["section_image_id"],
                "sha256": hashlib.sha256(
                    (tmp_path / record["relative_path"]).read_bytes()
                ).hexdigest(),
            }
            for record in records
        ],
    )
    (tmp_path / "provenance.json").write_text('{"source":"fixture"}', encoding="utf-8")

    built = build_registered_image_quality_manifest(tmp_path, workers=2)
    loaded, approved, rejected = load_registered_image_quality_manifest(tmp_path)
    assert loaded == built
    assert approved == {11}
    assert set(rejected) == {22}
    assert rejected[22]["reason"] == "uninformative_intensity"

    (tmp_path / "provenance.json").write_text('{"source":"changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the immutable source"):
        load_registered_image_quality_manifest(tmp_path)
