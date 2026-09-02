from pathlib import Path

import nrrd
import numpy as np
import pytest

import training.arbitrary_plane_joint_curriculum_v3 as joint_curriculum
import training.arbitrary_plane_training_row_v3 as training_row
from training.arbitrary_plane_rendered_generator import prepare_finite_render_context
from training.arbitrary_plane_support import build_annotation_support_index


ATLAS_ROOT = Path(r"I:\AnatomyTracker\data\Allen Brain Atlas 25um")
TEMPLATE_PATH = ATLAS_ROOT / "average_template_25.nrrd"
ANNOTATION_PATH = ATLAS_ROOT / "annotation_25.nrrd"


@pytest.mark.skipif(
    not TEMPLATE_PATH.is_file() or not ANNOTATION_PATH.is_file(),
    reason="pinned I:-drive Allen sources are unavailable",
)
def test_original_allen_sample_159_now_certifies_on_attempt_zero():
    template = nrrd.read(str(TEMPLATE_PATH), index_order="F")[0]
    annotation = nrrd.read(str(ANNOTATION_PATH), index_order="F")[0]
    support = build_annotation_support_index(
        annotation,
        atlas_id="Allen CCFv3",
        atlas_version="2017 25um",
        source_uri=str(ANNOTATION_PATH.resolve()),
        source_sha256=(
            "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42"
        ),
        source_entity_type="atlas-annotation",
        voxel_size_um=(25.0, 25.0, 25.0),
        origin_um=(0.0, 0.0, 0.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    prepared = prepare_finite_render_context(
        template,
        annotation,
        support,
        scalar_source_uri=str(TEMPLATE_PATH.resolve()),
        scalar_source_sha256=(
            "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b"
        ),
        scalar_source_entity_type="atlas-template",
        scalar_dtype="float32",
        template_decoder="pynrrd==1.1.3 nrrd.read(index_order=F)",
        template_index_order="F",
        annotation_decoder="pynrrd==1.1.3 nrrd.read(index_order=F)",
        annotation_index_order="F",
    )
    sample_index = 159
    animal_index = sample_index // 16
    prefix = "authv3-train-joint"
    row = joint_curriculum.make_joint_curriculum_training_row_v4(
        prepared,
        root_seed="0x2026090200000002",
        sample_index=sample_index,
        output_shape_h_w=(160, 160),
        selected_mode=training_row.TRAINABLE_MODES[
            sample_index % len(training_row.TRAINABLE_MODES)
        ],
        reflection_state=training_row.REFLECTION_STATES[
            (sample_index // len(training_row.TRAINABLE_MODES))
            % len(training_row.REFLECTION_STATES)
        ],
        amplitude_band="moderate",
        animal_id=f"{prefix}-animal-{animal_index:08d}",
        specimen_id=f"{prefix}-specimen-{animal_index:08d}",
        experiment_id=f"{prefix}-experiment-{animal_index:08d}",
        synthetic_animal_id=f"{prefix}-synthetic-animal-{animal_index:08d}",
        section_id=f"{prefix}-section-{sample_index:08d}",
        split="train",
        joint_attempt_number=0,
        joint_rejection_history=[],
    )
    assert row["numeric_rng_provenance"] == {
        "schema_version": joint_curriculum.JOINT_CURRICULUM_V4_SCHEMA,
        "root_seed_uint64": "0x2026090200000002",
        "sample_index": 159,
        "joint_attempt_number": 0,
        "derived_plane_sample_index": 1150045317203523431,
        "finite_render_seed_uint64": "0x47be5a245e2e1c57",
        "synthetic_seed_uint64": "0x4c4e418dff508711",
    }
    assert row["upstream_reference"]["joint_rejection_history"] == []
    certification = row["upstream_reference"][
        "direct_deformation_target_certification_summary"
    ]
    assert certification["diagnostics"]["valid_certification_error_max_px"] == 0.0
    assert (
        certification["diagnostics"][
            "uniform_canvas_affine_coefficient_max_abs"
        ]
        < 1e-6
    )
    assert certification["diagnostics"]["parent_pose_adjustment_max_abs"] == 0.0
    assert row["canonical_effective_quicknii_ouv_float64"] == row[
        "upstream_reference"
    ]["effective_quicknii_ouv_ml_ap_dv_before_gauge"]
    assert joint_curriculum.verify_joint_curriculum_training_row_v4(
        row, prepared
    )
