import copy
import json

import numpy as np
import pytest

import training.arbitrary_plane_synthetic_generator as synthetic
from training.arbitrary_plane_rendered_generator import make_finite_arbitrary_plane_render
from training.arbitrary_plane_support import build_annotation_support_index
from training.arbitrary_plane_synthetic_generator import (
    ABSENT_OUTLINE,
    ACCURATE_OUTLINE,
    IMPERFECT_OUTLINE,
    derive_field_seed,
    make_arbitrary_plane_synthetic_realization,
    replay_arbitrary_plane_synthetic_realization,
    synthetic_realization_receipt,
    verify_arbitrary_plane_synthetic_realization,
)
from training.arbitrary_plane_synthetic_ops import nearest_sample_labels


def _assets(shape=(17, 15, 13)):
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
    return template, annotation, support


def _parent(
    *, seed=2**63 + 101, output_shape=(47, 53),
    animal_id="animal-7", specimen_id="specimen-7a", experiment_id="experiment-71",
):
    template, annotation, support = _assets()
    parent = make_finite_arbitrary_plane_render(
        template,
        annotation,
        support,
        "development",
        seed,
        output_shape,
        sample_index=29,
        margin_um=(13.0, 17.0),
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="4" * 64,
        template_decoder="pynrrd 1.1.3",
        template_index_order="F",
        annotation_decoder="pynrrd 1.1.3",
        annotation_index_order="F",
        animal_id=animal_id,
        specimen_id=specimen_id,
        experiment_id=experiment_id,
    )
    return parent, support


def _make(**kwargs):
    parent, support = _parent()
    artifact = make_arbitrary_plane_synthetic_realization(
        parent, support, root_seed=kwargs.pop("root_seed", 2**63 + 55), **kwargs
    )
    return artifact, parent, support


def test_complete_realization_replays_preserves_provenance_and_is_model_independent():
    artifact, parent, support = _make()
    replayed = replay_arbitrary_plane_synthetic_realization(artifact, support)

    verify_arbitrary_plane_synthetic_realization(artifact, support)
    assert artifact["synthetic_realization_id"] == replayed["synthetic_realization_id"]
    assert all(
        np.array_equal(value, replayed["arrays"][name])
        for name, value in artifact["arrays"].items()
    )
    assert artifact["provenance"] == parent["provenance"]
    assert [artifact["provenance"][key] for key in ("animal_id", "specimen_id", "experiment_id")] == [
        "animal-7", "specimen-7a", "experiment-71"
    ]
    dependency_keys = {
        key for key in artifact["generator"] if key.endswith("_dependencies")
    }
    assert dependency_keys == {
        "learned_checkpoint_dependencies", "previous_model_dependencies", "pretrained_feature_dependencies"
    }
    assert all(artifact["generator"][key] == [] for key in dependency_keys)
    assert int(artifact["root_seed"], 16) > 2**53
    json.dumps(synthetic_realization_receipt(artifact), allow_nan=False)


def test_map_pullback_physical_coordinates_and_exclusive_mask_algebra_are_exact():
    artifact, parent, _ = _make(root_seed=0)
    arrays = artifact["arrays"]
    source_labels = nearest_sample_labels(
        parent["raster"]["annotation"], arrays["source_to_fixed_map"]
    )

    assert np.array_equal(arrays["source_annotation"], source_labels)
    assert np.array_equal(arrays["source_clean_tissue_mask"], source_labels != 0)
    assert np.all(arrays["source_ccf_ap_dv_ml_um"][~arrays["source_map_domain_mask"]] == 0.0)
    assert np.allclose(
        arrays["fixed_to_source_map_uv_um"],
        arrays["fixed_to_source_map_uv_um_float64_from_effective_map"].astype(np.float32),
        atol=5e-5,
    )
    tissue = arrays["source_clean_tissue_mask"]
    missing, occlusion, appearance = (
        arrays["physical_loss_mask"], arrays["occlusion_mask"], arrays["appearance_artifact_mask"]
    )
    assert not np.any(missing & occlusion)
    assert not np.any(missing & appearance)
    assert not np.any(occlusion & appearance)
    assert np.array_equal(arrays["visible_mask"] | arrays["missing_mask"] | arrays["artifact_mask"], tissue)
    assert np.all(arrays["fixed_valid_correspondence_mask"] <= arrays["fixed_clean_tissue_mask"])
    assert np.all(arrays["fixed_valid_correspondence_mask"] <= arrays["fixed_map_domain_mask"])
    assert np.array_equal(
        arrays["source_valid_correspondence_mask"],
        arrays["source_map_domain_mask"]
        & arrays["source_clean_tissue_mask"]
        & ~arrays["observation_invalid_mask"],
    )
    assert np.array_equal(
        arrays["fixed_valid_correspondence_mask"],
        arrays["fixed_map_domain_mask"]
        & arrays["fixed_clean_tissue_mask"]
        & nearest_sample_labels(
            arrays["source_valid_correspondence_mask"].astype(np.uint8),
            arrays["fixed_to_source_map"],
        ).astype(bool),
    )
    accepted = artifact["g1"]["parameters"]["accepted_attempt"]
    assert accepted["topology_metrics"]["accepted"]
    assert accepted["integration_steps"] <= 12
    assert accepted["fov_gate_passed"]
    assert accepted["maximum_displacement_gate_passed"]
    assert accepted["postintegration_rms_target_gate_passed"]
    seeds = accepted["field_stream_seed_uint64"]
    assert seeds["fine-svf-field"] != seeds["coarse-svf-field"]
    assert accepted["achieved_postintegration_rms_displacement_um"] == pytest.approx(
        accepted["target_rms_displacement_um"], rel=0.02, abs=1e-6
    )


def test_outline_counterfactuals_share_latent_stages_but_have_distinct_final_identities():
    parent, support = _parent()
    artifacts = {
        mode: make_arbitrary_plane_synthetic_realization(
            parent, support, root_seed=2**63 + 77, outline_mode=mode
        )
        for mode in (ACCURATE_OUTLINE, IMPERFECT_OUTLINE, ABSENT_OUTLINE)
    }
    latent_keys = (
        ("g1", "deformation_realization_id"),
        ("g2", "appearance_realization_id"),
        ("g3", "damage_realization_id"),
    )

    assert len({value["paired_view_group_id"] for value in artifacts.values()}) == 1
    assert all(
        len({value[stage][key] for value in artifacts.values()}) == 1
        for stage, key in latent_keys
    )
    assert len({value["synthetic_realization_id"] for value in artifacts.values()}) == 3
    for mode in (ACCURATE_OUTLINE, IMPERFECT_OUTLINE):
        arrays = artifacts[mode]["arrays"]
        assert np.all(arrays["model_input_image"][~arrays["input_outline_mask"]] == 0.0)
        assert artifacts[mode]["outline"]["parameters"]["outline_available"]
    imperfect_iou = artifacts[IMPERFECT_OUTLINE]["outline"]["parameters"]["quality_iou"]
    assert 0.70 <= imperfect_iou <= 0.98
    absent = artifacts[ABSENT_OUTLINE]
    assert not absent["outline"]["parameters"]["outline_available"]
    assert not absent["arrays"]["input_outline_mask"].any()
    assert np.array_equal(absent["arrays"]["model_input_image"], absent["arrays"]["damaged_acquired_image"])


def test_stage_and_large_seed_domains_are_isolated_and_canonical():
    seed = 2**63 + 987654321
    values = {
        derive_field_seed(seed, split, sample, stage, field, attempt)
        for split, sample, stage, field, attempt in (
            ("train", 2**53 + 1, "g1", "svf-noise", 0),
            ("development", 2**53 + 1, "g1", "svf-noise", 0),
            ("train", 2**53 + 2, "g1", "svf-noise", 0),
            ("train", 2**53 + 1, "g2", "svf-noise", 0),
            ("train", 2**53 + 1, "g1", "gain", 0),
            ("train", 2**53 + 1, "g1", "svf-noise", 1),
        )
    }
    assert len(values) == 6
    assert derive_field_seed(seed, "train", 2**53 + 1, "g1", "svf-noise", 0) == derive_field_seed(
        seed, "train", 2**53 + 1, "g1", "svf-noise", 0
    )


def test_tampering_incomplete_state_and_coherent_local_rehash_are_rejected():
    artifact, _, support = _make()
    incomplete = copy.deepcopy(artifact)
    del incomplete["synthetic_realization_id"]
    with pytest.raises(ValueError, match="incomplete"):
        replay_arbitrary_plane_synthetic_realization(incomplete, support)

    tampered = copy.deepcopy(artifact)
    tampered["arrays"]["model_input_image"] = tampered["arrays"]["model_input_image"].copy()
    tampered["arrays"]["model_input_image"][0, 0] = np.float32(0.37)
    tampered["array_receipts"] = synthetic._array_receipts(tampered["arrays"])
    tampered["synthetic_artifacts_sha256"] = synthetic._payload_sha256(tampered["array_receipts"])
    tampered["synthetic_receipt_sha256"] = synthetic._payload_sha256(
        synthetic_realization_receipt(tampered)
    )
    with pytest.raises(ValueError):
        verify_arbitrary_plane_synthetic_realization(tampered, support)


@pytest.mark.parametrize(
    "kind,category",
    (
        ("boundary-bite-or-missing-cortex", "physical_loss"),
        ("internal-hole", "physical_loss"),
        ("tear-or-crack", "physical_loss"),
        ("blackout-or-occluding-polygon", "occlusion"),
        ("fold-like-bright-or-doubled-strip", "appearance_artifact"),
    ),
)
def test_all_five_damage_families_change_tissue_pixels(kind, category):
    parent, support = _parent()
    config = synthetic._resolved_config(
        parent, 919, None, "ordinary", ACCURATE_OUTLINE, None
    )
    tissue = parent["raster"]["brain_mask"]
    mask, actual_category, receipt = synthetic._damage_event(kind, tissue, config, 0, 0)
    assert actual_category == category
    assert mask.any()
    assert np.all(mask <= tissue)
    assert receipt["changed_pixel_count"] > 0


def test_near_cardinal_and_oblique_finite_parents_and_nondefault_shape_are_not_benchmarks():
    normals = []
    for parent_seed, output_shape, synthetic_seed in (
        (3033, (41, 57), 811),
        (2**63 + 101, (47, 53), 812),
    ):
        parent, support = _parent(seed=parent_seed, output_shape=output_shape)
        artifact = make_arbitrary_plane_synthetic_realization(
            parent, support, root_seed=synthetic_seed
        )
        verify_arbitrary_plane_synthetic_realization(artifact, support)
        assert artifact["arrays"]["model_input_image"].shape == output_shape
        assert not artifact["development_scope"]["benchmark"]
        normals.append(np.asarray(parent["geometry"]["normal_rp2_ap_dv_ml"]))
    assert np.max(np.abs(normals[0])) > 0.99
    assert np.max(np.abs(normals[1])) < 0.95


def test_identity_appearance_bypasses_artifacts_and_tiny_support_disables_damage():
    artifact, _, _ = _make(
        config_overrides={
            "g1": {"identity_probability": 1.0},
            "g2": {"identity_probability": 1.0, "artifact_probability": 1.0},
            "g3": {"disable_damage_below_pixels": 10**9, "event_count_probabilities": [0.0, 0.0, 1.0]},
        }
    )
    assert artifact["g1"]["parameters"]["accepted_attempt"]["identity_path"]
    assert artifact["g2"]["parameters"]["identity_bypass"]
    assert not artifact["arrays"]["g2_appearance_artifact_mask"].any()
    assert not artifact["g3"]["parameters"]["damage_eligible"]
    assert artifact["g3"]["parameters"]["event_count"] == 0


def test_default_outline_plan_is_deterministic_and_records_its_isolated_seed():
    parent, support = _parent()
    first = make_arbitrary_plane_synthetic_realization(parent, support, root_seed=12345)
    second = make_arbitrary_plane_synthetic_realization(parent, support, root_seed=12345)
    assert first["outline"]["parameters"]["outline_plan"] == second["outline"]["parameters"]["outline_plan"]
    assert first["synthetic_realization_id"] == second["synthetic_realization_id"]
    plan = first["outline"]["parameters"]["outline_plan"]
    assert plan["assignment"] == "isolated equal-probability outline-plan draw"
    assert plan["selected_mode"] in (ACCURATE_OUTLINE, IMPERFECT_OUTLINE, ABSENT_OUTLINE)
    assert plan["field_stream_seed_uint64"].startswith("0x")


def test_dependency_and_loaded_source_claim_tampering_are_rejected():
    artifact, _, support = _make()
    nonempty = copy.deepcopy(artifact)
    nonempty["generator"]["previous_model_dependencies"] = ["legacy-model.pt"]
    with pytest.raises(ValueError, match="model independent"):
        verify_arbitrary_plane_synthetic_realization(nonempty, support)
    extra = copy.deepcopy(artifact)
    extra["generator"]["learned_style_dependencies"] = []
    with pytest.raises(ValueError, match="exactly three"):
        verify_arbitrary_plane_synthetic_realization(extra, support)
    source_tamper = copy.deepcopy(artifact)
    source_tamper["generator"]["implementation"]["loaded_source_sha256"]["ops"] = "0" * 64
    with pytest.raises(ValueError, match="sources"):
        verify_arbitrary_plane_synthetic_realization(source_tamper, support)


def test_subject_identifier_types_and_null_are_preserved_without_conflation():
    parent, support = _parent(animal_id=np.int64(7), specimen_id=None, experiment_id="7")
    artifact = make_arbitrary_plane_synthetic_realization(parent, support, root_seed=987)
    assert artifact["provenance"]["animal_id"] == 7
    assert isinstance(artifact["provenance"]["animal_id"], int)
    assert artifact["provenance"]["specimen_id"] is None
    assert artifact["provenance"]["experiment_id"] == "7"
    assert artifact["provenance"] == parent["provenance"]


def test_parent_identity_and_resolved_config_tampering_are_rejected():
    artifact, _, support = _make()
    parent_id = copy.deepcopy(artifact)
    parent_id["finite_parent_identity"]["finite_plane_render_id"] = "0" * 64
    with pytest.raises(ValueError, match="parent identity"):
        verify_arbitrary_plane_synthetic_realization(parent_id, support)
    config = copy.deepcopy(artifact)
    config["generator"]["resolved_config"]["sample_index"] += 1
    with pytest.raises(ValueError, match="config hash"):
        verify_arbitrary_plane_synthetic_realization(config, support)
