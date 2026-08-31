import hashlib
import json
from pathlib import Path

import numpy as np

from training.arbitrary_plane_rendered_generator import make_finite_arbitrary_plane_render
from training.arbitrary_plane_support import build_annotation_support_index
from training.arbitrary_plane_synthetic_generator import (
    ABSENT_OUTLINE,
    ACCURATE_OUTLINE,
    IMPERFECT_OUTLINE,
    make_arbitrary_plane_synthetic_realization,
    verify_arbitrary_plane_synthetic_realization,
)


shape = (17, 15, 13)
annotation = np.zeros(shape, dtype=np.uint16)
annotation[2:-2, 2:-2, 1:-2] = 7
annotation[6:-5, 5:-4, 4:-4] = 19
ap, dv, ml = np.indices(shape)
template = (100 + 3 * ap + 5 * dv + 7 * ml).astype(np.uint16)
support = build_annotation_support_index(
    annotation,
    atlas_id="fixture-ccf",
    atlas_version="fixture-v1",
    source_uri="file:///fixture/annotation.nrrd",
    source_sha256="3" * 64,
    source_entity_type="atlas-annotation",
    voxel_size_um=(11.0, 17.0, 29.0),
    origin_um=(-71.0, 23.0, 107.0),
    coordinate_axis_directions=("posterior", "inferior", "right"),
)

parent_arguments = {
    "split": "development",
    "seed": 2**63 + 101,
    "output_shape": (47, 53),
    "sample_index": 29,
    "margin_um": (13.0, 17.0),
    "scalar_source_uri": "file:///fixture/template.nrrd",
    "scalar_source_sha256": "4" * 64,
    "template_decoder": "pynrrd 1.1.3",
    "template_index_order": "F",
    "annotation_decoder": "pynrrd 1.1.3",
    "annotation_index_order": "F",
    "animal_id": "animal-7",
    "specimen_id": "specimen-7a",
    "experiment_id": "experiment-71",
}
parent = make_finite_arbitrary_plane_render(template, annotation, support, **parent_arguments)

records = []
for root_seed in range(100):
    artifact = make_arbitrary_plane_synthetic_realization(parent, support, root_seed=root_seed)
    verify_arbitrary_plane_synthetic_realization(artifact, support)
    records.append(
        {
            "root_seed": artifact["root_seed"],
            "resolved_config_sha256": artifact["generator"]["resolved_config_sha256"],
            "g1_accepted_attempt_index": artifact["g1"]["parameters"]["accepted_attempt_index"],
            "g2_accepted_attempt_index": artifact["g2"]["parameters"]["accepted_attempt_index"],
            "g3_accepted_attempt_index": artifact["g3"]["parameters"]["accepted_attempt_index"],
            "g2_source_family": artifact["g2"]["parameters"]["source_family"],
            "damage_event_types": [
                event["type"] for event in artifact["g3"]["parameters"]["events"]
            ],
            "outline_mode": artifact["outline"]["parameters"]["mode"],
            "deformation_realization_id": artifact["g1"]["deformation_realization_id"],
            "appearance_realization_id": artifact["g2"]["appearance_realization_id"],
            "damage_realization_id": artifact["g3"]["damage_realization_id"],
            "outline_realization_id": artifact["outline"]["outline_realization_id"],
            "paired_view_group_id": artifact["paired_view_group_id"],
            "synthetic_realization_id": artifact["synthetic_realization_id"],
            "synthetic_receipt_sha256": artifact["synthetic_receipt_sha256"],
        }
    )

outline_counts = {
    mode: sum(record["outline_mode"] == mode for record in records)
    for mode in (ACCURATE_OUTLINE, IMPERFECT_OUTLINE, ABSENT_OUTLINE)
}
appearance_family_counts = {
    family: sum(record["g2_source_family"] == family for record in records)
    for family in sorted({record["g2_source_family"] for record in records})
}
damage_type_counts = {
    kind: sum(kind in record["damage_event_types"] for record in records)
    for kind in sorted({kind for record in records for kind in record["damage_event_types"]})
}

paired = {}
for mode in (ACCURATE_OUTLINE, IMPERFECT_OUTLINE, ABSENT_OUTLINE):
    artifact = make_arbitrary_plane_synthetic_realization(
        parent, support, root_seed=2**63 + 77, outline_mode=mode
    )
    verify_arbitrary_plane_synthetic_realization(artifact, support)
    paired[mode] = {
        "resolved_config_sha256": artifact["generator"]["resolved_config_sha256"],
        "deformation_realization_id": artifact["g1"]["deformation_realization_id"],
        "appearance_realization_id": artifact["g2"]["appearance_realization_id"],
        "damage_realization_id": artifact["g3"]["damage_realization_id"],
        "outline_realization_id": artifact["outline"]["outline_realization_id"],
        "paired_view_group_id": artifact["paired_view_group_id"],
        "synthetic_realization_id": artifact["synthetic_realization_id"],
        "quality_iou": artifact["outline"]["parameters"]["quality_iou"],
        "black_exterior_exact": artifact["outline"]["parameters"]["black_exterior_exact"],
    }

orientation_cases = []
for name, parent_seed, synthetic_seed, dominant_axis in (
    ("near-AP", 826, 1826, 0),
    ("near-DV", 419, 1419, 1),
    ("near-ML", 85, 1085, 2),
    ("oblique", 2**63 + 101, 812, None),
):
    case_parent = make_finite_arbitrary_plane_render(
        template, annotation, support, **{**parent_arguments, "seed": parent_seed}
    )
    artifact = make_arbitrary_plane_synthetic_realization(
        case_parent, support, root_seed=synthetic_seed
    )
    verify_arbitrary_plane_synthetic_realization(artifact, support)
    orientation_cases.append(
        {
            "name": name,
            "parent_seed": f"0x{parent_seed:016x}",
            "synthetic_seed": f"0x{synthetic_seed:016x}",
            "dominant_axis": dominant_axis,
            "normal_rp2_ap_dv_ml": case_parent["geometry"]["normal_rp2_ap_dv_ml"],
            "brain_pixel_count": int(case_parent["raster"]["brain_mask"].sum()),
            "finite_plane_render_id": case_parent["finite_plane_render_id"],
            "synthetic_realization_id": artifact["synthetic_realization_id"],
        }
    )

canonical_records = json.dumps(
    records, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
).encode("utf-8")
result = {
    "schema_version": "anatomy-tracker.arbitrary-plane-synthetic-audit/v1",
    "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "root_seed_sequence": {
        "encoding": "canonical-lowercase-uint64-hex/v1",
        "start_inclusive": "0x0000000000000000",
        "stop_exclusive": "0x0000000000000064",
        "step": 1,
    },
    "completed_record_count": len(records),
    "raw_records_sha256": hashlib.sha256(canonical_records).hexdigest(),
    "maximum_accepted_attempt_index": {
        stage: max(record[f"{stage}_accepted_attempt_index"] for record in records)
        for stage in ("g1", "g2", "g3")
    },
    "outline_mode_counts": outline_counts,
    "appearance_family_counts": appearance_family_counts,
    "damage_type_presence_counts": damage_type_counts,
    "fixture": {
        "finite_plane_render_id": parent["finite_plane_render_id"],
        "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
        "support_index_sha256": parent["support_index_sha256"],
        "animal_id": parent["provenance"]["animal_id"],
        "specimen_id": parent["provenance"]["specimen_id"],
        "experiment_id": parent["provenance"]["experiment_id"],
        "parent_arguments": parent_arguments,
    },
    "model_independence": {
        key: records and artifact["generator"][key]
        for key in (
            "learned_checkpoint_dependencies",
            "previous_model_dependencies",
            "pretrained_feature_dependencies",
        )
    },
    "paired_outline_root_seed": "0x800000000000004d",
    "paired_outline_results": paired,
    "pinned_orientation_results": orientation_cases,
}
result["result_payload_sha256"] = hashlib.sha256(
    json.dumps(result, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
).hexdigest()
print(json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True))
