from pathlib import Path

import numpy as np
import pytest

import training.arbitrary_plane_semantic_oracle_panel_v2 as panel_v2
import training.arbitrary_plane_training_data_v3 as training_data_v3
import training.arbitrary_plane_training_row_v3 as training_row_v3
from training.run_arbitrary_plane_semantic_oracle_panel_v2 import (
    load_pinned_allen_context_v2,
)


ATLAS = Path("data/Allen Brain Atlas 25um")


def test_prepared_section_rejects_changed_parent_spec_but_allows_descendant_fields(
    monkeypatch,
):
    spec = {
        "support_root_seed": "0x535550504f525401",
        "split": "train",
        "split_index": 0,
        "animal_index": 2,
        "animal_id": "animal-002",
        "section_index": 3,
        "plane_stratum": "general_oblique",
        "specimen_id": "specimen-002",
        "experiment_id": "experiment-002",
        "section_root_seed": "0x53454354494f4e01",
        "section_id": "section-003",
        "window_root_seed": "0x57494e444f570001",
        "observation_root_seed": "0x4f42534552564501",
        "observation_index": 0,
        "modality": "brightfield-nissl-like",
        "realization_index": 0,
    }
    prepared = {"preparation_spec": training_data_v3._preparation_spec(spec)}
    monkeypatch.setattr(
        training_data_v3,
        "verify_prepared_training_section_v3",
        lambda prepared, context: True,
    )
    changed = {**spec, "section_index": spec["section_index"] + 1}
    with pytest.raises(
        ValueError, match="descendant generation spec changes its prepared section"
    ):
        training_data_v3.make_training_bundle_from_prepared_section_v3(
            prepared, {}, changed
        )
    descendant_only = {**spec, "observation_index": 9}
    assert training_data_v3._preparation_spec(descendant_only) == prepared[
        "preparation_spec"
    ]


@pytest.mark.skipif(not ATLAS.exists(), reason="pinned Allen atlas is not present")
def test_real_allen_general_oblique_chain_replays_to_one_model_ready_v3_row():
    context = load_pinned_allen_context_v2(ATLAS)
    panel = panel_v2.arbitrary_plane_semantic_oracle_development_panel_v2()
    animal = panel["animals"][0]
    case = next(
        item
        for item in panel["cases"]
        if item["animal_slot"] == 0 and item["plane_stratum"] == "general_oblique"
    )
    subject = training_data_v3.make_training_subject_v3(
        context,
        root_seed=animal["subject_deformation_root_seed_uint64"],
        split=case["split"],
        animal_index=case["animal_index"],
        animal_id=case["animal_id"],
    )
    spec = {
        **case,
        "support_root_seed": case["root_seeds_uint64"]["support_resolution"],
        "section_root_seed": case["root_seeds_uint64"]["section_processing"],
        "window_root_seed": "0x57494e444f570304",
        "observation_root_seed": case["root_seeds_uint64"]["observation"],
        "realization_index": 0,
    }
    bundle = training_data_v3.make_training_bundle_v3(context, subject, spec)
    assert bundle["schema_version"] == training_data_v3.TRAINING_BUNDLE_V3_SCHEMA
    assert bundle["legacy_chain_adapter_v3"]["v2_module_mutation"] is False
    assert bundle["legacy_chain_adapter_v3"]["parallel_row_generation_safe"] is True
    assert bundle["receipt_sha256"] == training_data_v3.acquisition._payload_sha256(
        training_data_v3.training_bundle_receipt_v3(bundle)
    )
    row = bundle["training_row"]
    training_row_v3.verify_arbitrary_plane_training_row_v3(
        row, bundle["observation"], spec["realization_index"]
    )
    arrays = row["arrays"]
    assert arrays["model_input_channels_float32"].shape == (192, 256, 3)
    assert arrays["truth_section_pullback_map_yx_px_float64"].shape == (192, 256, 2)
    assert arrays["truth_section_pullback_stationary_velocity_yx_px_float64"].shape == (
        192,
        256,
        2,
    )
    assert np.isfinite(arrays["model_input_channels_float32"]).all()
    assert row["lineage"]["animal_id"] == case["animal_id"]
    assert row["lineage"]["specimen_id"] == case["specimen_id"]
    assert row["lineage"]["experiment_id"] == case["experiment_id"]
    assert row["prior_model_dependencies"] == []
    assert row["prior_feature_dependencies"] == []
