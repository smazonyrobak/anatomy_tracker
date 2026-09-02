import copy
import json

import numpy as np
import pytest

import training.arbitrary_plane_synthetic_generator as synthetic
from training.arbitrary_plane_rendered_generator import make_finite_arbitrary_plane_render
from training.arbitrary_plane_psf_v4 import make_finite_psf_schedule_v4
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
from training.arbitrary_plane_synthetic_ops import (
    bilinear_sample_scalar,
    nearest_sample_labels,
)


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
    sample_index=29, minimum_brain_pixels=1,
    animal_id="animal-7", specimen_id="specimen-7a", experiment_id="experiment-71",
    generator_source_commit=None,
):
    template, annotation, support = _assets()
    parent = make_finite_arbitrary_plane_render(
        template,
        annotation,
        support,
        "development",
        seed,
        output_shape,
        sample_index=sample_index,
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
        minimum_brain_pixels=minimum_brain_pixels,
        generator_source_commit=generator_source_commit,
    )
    return parent, support


def _make(**kwargs):
    parent, support = _parent()
    artifact = make_arbitrary_plane_synthetic_realization(
        parent, support, root_seed=kwargs.pop("root_seed", 2**63 + 55), **kwargs
    )
    return artifact, parent, support


def _slab_observation_v4(parent, support, *, full_canvas=False):
    center = np.asarray(parent["raster"]["brain_mask"], dtype=bool)
    support_mask = center.copy()
    support_mask[1:] |= center[:-1]
    support_mask[:-1] |= center[1:]
    support_mask[:, 1:] |= center[:, :-1]
    support_mask[:, :-1] |= center[:, 1:]
    if full_canvas:
        support_mask[:] = True
    slab_only = support_mask & ~center
    assert slab_only.any()
    observed = np.asarray(parent["raster"]["scalar"], dtype=np.float32).copy()
    observed[slab_only] = np.float32(observed[center].mean() + 17.0)
    occupancy = np.where(center, 1.0, np.where(slab_only, 0.35, 0.0)).astype(
        np.float32
    )
    modal = np.asarray(parent["raster"]["annotation"], dtype=np.int64).copy()
    modal[slab_only] = 19
    purity = np.where(center, 1.0, np.where(slab_only, 0.7, 0.0)).astype(
        np.float32
    )
    dense_weight = center.astype(np.float32)
    arrays = {
        "observed_scalar_float32": observed,
        "slab_brain_occupancy_float32": occupancy,
        "slab_observable_support_mask": support_mask,
        "centre_label_psf_mass_float32": center.astype(np.float32),
        "slab_modal_annotation_int64": modal,
        "slab_modal_purity_float32": purity,
        "dense_correspondence_weight_float32": dense_weight,
        "dense_correspondence_abstention_mask": ~center,
    }
    selection_payload = {
        "selection_mode": "explicit-receipted-thickness",
        "seed_encoding": "canonical-lowercase-uint64-hex/v1",
        "thickness_seed_uint64": None,
        "draw_fraction": None,
        "distribution": "explicit caller-supplied thickness within declared capability",
        "production_thickness_range_um": [25.0, 100.0],
        "nominal_cut_thickness_um": 40.0,
    }
    selection = {
        **selection_payload,
        "thickness_selection_sha256": synthetic._payload_sha256_v4(selection_payload),
    }
    finite_psf = make_finite_psf_schedule_v4(
        "finite_boxcar",
        40.0,
        thickness_selection_sha256=selection["thickness_selection_sha256"],
    )
    block = {
        "schema": "anatomy-tracker.slab-observation/v4",
        "finite_plane_render_id": parent["finite_plane_render_id"],
        "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
        "plane_realization_id": parent["plane_realization_id"],
        "support_index_sha256": parent["support_index_sha256"],
        "provenance_sha256": parent["provenance_sha256"],
        "split": parent["split"],
        "sample_index": parent["sample_index"],
        "animal_id": parent["provenance"]["animal_id"],
        "specimen_id": parent["provenance"]["specimen_id"],
        "experiment_id": parent["provenance"]["experiment_id"],
        "thickness_selection": selection,
        "finite_psf": finite_psf,
        "centre_plane_targets_receipt_sha256": (
            synthetic._centre_plane_targets_receipt_sha256_v4(parent, support)
        ),
        **arrays,
        "array_receipts": {
            name: synthetic._slab_observation_array_receipt_v4(value)
            for name, value in sorted(arrays.items())
        },
    }
    block.update(
        {
            "combined_sha256": synthetic._payload_sha256_v4(
                {
                    "schema": "anatomy-tracker.slab-observation-arrays/v4",
                    "array_receipts": block["array_receipts"],
                }
            ),
            "centre_plane_brain_pixel_count": int(center.sum()),
            "slab_observable_pixel_count": int(support_mask.sum()),
            "slab_effective_brain_pixel_mass": float(
                occupancy.astype(np.float64).sum()
            ),
            "dense_abstention_pixel_count": int((~center).sum()),
            "dense_eligible_pixel_count": int(center.sum()),
            "dense_effective_supervision_mass": float(
                dense_weight.astype(np.float64).sum()
            ),
        }
    )
    block["slab_observation_id"] = synthetic._payload_sha256_v4(
        {
            "parent": parent["finite_plane_render_id"],
            "finite_psf": block["finite_psf"],
            "arrays": block["array_receipts"],
        }
    )
    block["receipt_sha256"] = synthetic._payload_sha256_v4(
        synthetic.slab_observation_v4_receipt(block)
    )
    return block


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


@pytest.mark.parametrize(
    "outline_mode", (ACCURATE_OUTLINE, IMPERFECT_OUTLINE, ABSENT_OUTLINE)
)
def test_one_pixel_marginal_plane_is_retained_and_replays_without_pose_redraw(
    outline_mode,
):
    parent, support = _parent(
        output_shape=(7, 7),
        sample_index=0,
        minimum_brain_pixels=64,
    )
    assert parent["acceptance_contract"]["brain_pixel_count"] == 1
    assert parent["acceptance_contract"]["pose_redrawn_for_raster_support"] is False
    assert len(parent["rejection_attempts"]) == 1
    artifact = make_arbitrary_plane_synthetic_realization(
        parent,
        support,
        root_seed=2**63 + 57,
        outline_mode=outline_mode,
    )
    replay = replay_arbitrary_plane_synthetic_realization(artifact, support)
    verify_arbitrary_plane_synthetic_realization(artifact, support)
    assert artifact["synthetic_receipt_sha256"] == replay[
        "synthetic_receipt_sha256"
    ]
    assert artifact["support_supervision"] == {
        "continuous_plane_sample_retained": True,
        "pose_redrawn_for_raster_support": False,
        "raster_brain_pixel_count": 1,
        "requested_identifiability_threshold_pixels": 64,
        "raster_support_meets_requested_identifiability_threshold": False,
        "marginal_support_generation_policy": (
            "identity deformation, damage/information/outline gates bypassed only as explicitly "
            "recorded; point and dense loss eligibility is decided by the curriculum row"
        ),
    }
    assert artifact["g1"]["parameters"][
        "marginal_raster_support_identity_bypass"
    ] is True
    assert artifact["g2"]["parameters"][
        "marginal_raster_support_information_bypass"
    ] is True
    assert artifact["g3"]["parameters"][
        "marginal_raster_support_visibility_bypass"
    ] is True
    assert artifact["outline"]["parameters"][
        "marginal_raster_support_outline_bypass"
    ] is True


def test_entrypoints_bind_finite_parent_source_commit_and_legacy_none_remains_compatible():
    source_commit, wrong_commit = "a" * 40, "b" * 40
    parent, support = _parent(generator_source_commit=source_commit)
    with pytest.raises(ValueError, match="source commit does not match"):
        make_arbitrary_plane_synthetic_realization(
            parent,
            support,
            root_seed=2**63 + 55,
            finite_parent_generator_source_commit=wrong_commit,
        )
    artifact = make_arbitrary_plane_synthetic_realization(
        parent,
        support,
        root_seed=2**63 + 55,
        finite_parent_generator_source_commit=source_commit,
    )
    replayed = replay_arbitrary_plane_synthetic_realization(
        artifact,
        support,
        finite_parent_generator_source_commit=source_commit,
    )
    verify_arbitrary_plane_synthetic_realization(
        artifact,
        support,
        finite_parent_generator_source_commit=source_commit,
    )
    assert replayed["synthetic_realization_id"] == artifact["synthetic_realization_id"]
    for operation in (
        replay_arbitrary_plane_synthetic_realization,
        verify_arbitrary_plane_synthetic_realization,
    ):
        with pytest.raises(ValueError, match="source commit does not match"):
            operation(
                artifact,
                support,
                finite_parent_generator_source_commit=wrong_commit,
            )

    legacy_artifact, _, legacy_support = _make()
    replay_arbitrary_plane_synthetic_realization(legacy_artifact, legacy_support)
    verify_arbitrary_plane_synthetic_realization(legacy_artifact, legacy_support)


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


@pytest.mark.parametrize(
    "parent_seed,synthetic_seed,dominant_axis",
    (
        (826, 1826, 0),
        (419, 1419, 1),
        (85, 1085, 2),
        (2**63 + 101, 812, None),
    ),
)
def test_pinned_near_cardinal_and_oblique_complete_realizations_are_not_benchmarks(
    parent_seed, synthetic_seed, dominant_axis
):
    output_shape = (47, 53)
    parent, support = _parent(seed=parent_seed, output_shape=output_shape)
    artifact = make_arbitrary_plane_synthetic_realization(
        parent, support, root_seed=synthetic_seed
    )
    verify_arbitrary_plane_synthetic_realization(artifact, support)
    normal = np.asarray(parent["geometry"]["normal_rp2_ap_dv_ml"])

    assert artifact["arrays"]["model_input_image"].shape == output_shape
    assert not artifact["development_scope"]["benchmark"]
    if dominant_axis is None:
        assert np.max(np.abs(normal)) < 0.95
    else:
        assert np.argmax(np.abs(normal)) == dominant_axis
        assert np.abs(normal[dominant_axis]) > 0.99


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


def test_legacy_v3_observation_arrays_remain_byte_exact_without_slab_keyword():
    artifact, _, _ = _make()
    expected = {
        "source_scalar_clean": "41064d7b375fd839cd1bcc112328fe97b60f570f61618273bca5279ee715696a",
        "source_annotation": "7cc7402be2d9a49a52f53f91a427513458115c29d5dff79de96ee82814f227d6",
        "source_clean_tissue_mask": "bb55faf00560c15a669d26d5cd92e25455d0475bd832a3135a76444bb835e5a6",
        "source_map_domain_mask": "923a9e38d54764d5de3fc2722d093626f72516f9046dffd7f01f78511b798266",
        "normalized_source_scalar": "36d7db5df19c7f18b0790d6889c0dbbb5ecf3ca7b377ffc8a76fd05678d8a1ff",
        "damaged_acquired_image": "2feed2b99b60ff54fa6cf08eb250d8b4ae8b237ee38859c042760fdb003f43c5",
        "model_input_image": "2feed2b99b60ff54fa6cf08eb250d8b4ae8b237ee38859c042760fdb003f43c5",
    }
    assert {
        name: artifact["array_receipts"][name]["array_sha256"]
        for name in expected
    } == expected
    assert "slab_observation_v4" not in artifact


def test_exact_producer_shaped_slab_block_interoperates_and_preserves_unicode_ids():
    parent, support = _parent(
        animal_id="mysz-Ł",
        specimen_id="skrawek-ósmy",
        experiment_id="doświadczenie-α",
    )
    slab = _slab_observation_v4(parent, support)
    expected_keys = {
        "schema",
        "slab_observation_id",
        "finite_plane_render_id",
        "finite_render_receipt_sha256",
        "plane_realization_id",
        "support_index_sha256",
        "provenance_sha256",
        "split",
        "sample_index",
        "animal_id",
        "specimen_id",
        "experiment_id",
        "thickness_selection",
        "finite_psf",
        "centre_plane_targets_receipt_sha256",
        *synthetic.SLAB_OBSERVATION_V4_ARRAY_NAMES,
        "array_receipts",
        "combined_sha256",
        "centre_plane_brain_pixel_count",
        "slab_observable_pixel_count",
        "slab_effective_brain_pixel_mass",
        "dense_abstention_pixel_count",
        "dense_eligible_pixel_count",
        "dense_effective_supervision_mass",
        "receipt_sha256",
    }
    assert set(slab) == expected_keys
    artifact = make_arbitrary_plane_synthetic_realization(
        parent,
        support,
        slab_observation_v4=slab,
        root_seed=444,
        outline_mode=ABSENT_OUTLINE,
        config_overrides={"g3": {"disable_damage_below_pixels": 10**9}},
    )

    verify_arbitrary_plane_synthetic_realization(artifact, support)
    assert artifact["slab_observation_v4_identity"]["receipt_sha256"] == slab[
        "receipt_sha256"
    ]
    assert artifact["provenance"]["animal_id"] == "mysz-Ł"
    assert artifact["generator"]["implementation"][
        "finite_psf_model_capability_receipt_sha256"
    ] == slab["finite_psf"]["finite_psf_capability_sha256"]


@pytest.mark.parametrize(
    "outline_mode", (ACCURATE_OUTLINE, IMPERFECT_OUTLINE, ABSENT_OUTLINE)
)
def test_slab_only_observation_is_input_support_but_center_zero_and_dense_abstained(
    outline_mode,
):
    parent, support = _parent()
    slab = _slab_observation_v4(parent, support)
    artifact = make_arbitrary_plane_synthetic_realization(
        parent,
        support,
        slab_observation_v4=slab,
        root_seed=2**63 + 55,
        outline_mode=outline_mode,
        config_overrides={
            "g1": {"identity_probability": 1.0},
            "g2": {"identity_probability": 1.0, "artifact_probability": 0.0},
            "g3": {
                "disable_damage_below_pixels": 10**9,
                "event_count_probabilities": [1.0, 0.0, 0.0],
            },
        },
    )
    verify_arbitrary_plane_synthetic_realization(artifact, support)
    arrays = artifact["arrays"]
    slab_only = arrays["source_slab_only_observation_mask"]

    assert slab_only.any()
    assert np.all(arrays["source_annotation"][slab_only] == 0)
    assert np.all(arrays["source_dense_correspondence_abstention_mask"][slab_only])
    assert not arrays["source_valid_correspondence_mask"][slab_only].any()
    assert np.all(arrays["source_dense_correspondence_weight_float32"][slab_only] == 0.0)
    assert np.all(slab_only <= arrays["observable_footprint_mask"])
    assert np.any(arrays["damaged_acquired_image"][slab_only] != arrays["acquired_background"][slab_only])
    assert artifact["g3"]["parameters"]["dual_mask_contract"] == {
        "damage_and_brush_support": "post-G1 finite-slab observable support",
        "center_annotation_and_ccf_targets": True,
        "slab_only_pixels_are_dense_abstained": True,
        "pose_loss_gated_by_pixel_mask": False,
        "automatic_segmentation_dependency": False,
    }
    if outline_mode == ABSENT_OUTLINE:
        assert np.array_equal(
            arrays["model_input_image"], arrays["damaged_acquired_image"]
        )
        assert not arrays["input_outline_mask"].any()
    else:
        assert np.all(
            arrays["model_input_image"][~arrays["input_outline_mask"]] == 0.0
        )
        assert np.any(arrays["input_outline_mask"] & slab_only)


def test_nonidentity_g1_uses_one_map_for_slab_observation_and_center_targets():
    parent, support = _parent()
    slab = _slab_observation_v4(parent, support)
    artifact = make_arbitrary_plane_synthetic_realization(
        parent,
        support,
        slab_observation_v4=slab,
        root_seed=0,
        outline_mode=ACCURATE_OUTLINE,
        config_overrides={
            "g1": {"identity_probability": 0.0},
            "g2": {"identity_probability": 1.0, "artifact_probability": 0.0},
            "g3": {"disable_damage_below_pixels": 10**9},
        },
    )
    arrays = artifact["arrays"]
    pullback = arrays["source_to_fixed_map"]

    assert not artifact["g1"]["parameters"]["accepted_attempt"]["identity_path"]
    assert np.array_equal(
        arrays["source_scalar_clean"],
        bilinear_sample_scalar(slab["observed_scalar_float32"], pullback),
    )
    assert np.array_equal(
        arrays["source_slab_brain_occupancy_float32"],
        bilinear_sample_scalar(slab["slab_brain_occupancy_float32"], pullback),
    )
    assert np.array_equal(
        arrays["source_annotation"],
        nearest_sample_labels(parent["raster"]["annotation"], pullback),
    )
    assert np.all(
        arrays["source_ccf_ap_dv_ml_um"][~arrays["source_map_domain_mask"]] == 0.0
    )
    verify_arbitrary_plane_synthetic_realization(artifact, support)


def test_slab_support_controls_g2_g3_and_point_pose_evidence_when_center_is_marginal():
    parent, support = _parent()
    slab = _slab_observation_v4(parent, support, full_canvas=True)
    center_count = int(np.asarray(parent["raster"]["brain_mask"]).sum())
    full_count = int(np.asarray(parent["raster"]["brain_mask"]).size)
    threshold = center_count + 1
    assert threshold < full_count
    artifact = make_arbitrary_plane_synthetic_realization(
        parent,
        support,
        slab_observation_v4=slab,
        root_seed=321,
        synthetic_stratum="low-information-stress",
        outline_mode=ABSENT_OUTLINE,
        config_overrides={
            "ordinary_minimum_clean_brain_pixels_floor": threshold,
            "ordinary_minimum_clean_brain_fraction": 0.0,
            "g2": {"identity_probability": 1.0, "artifact_probability": 0.0},
            "g3": {
                "disable_damage_below_pixels": threshold,
                "event_count_probabilities": [1.0, 0.0, 0.0],
            },
        },
    )

    assert artifact["g1"]["parameters"]["marginal_raster_support_identity_bypass"]
    assert not artifact["g2"]["parameters"]["marginal_raster_support_information_bypass"]
    assert not artifact["g3"]["parameters"]["marginal_raster_support_visibility_bypass"]
    assert artifact["g3"]["parameters"]["damage_eligible"]
    assert artifact["g2"]["parameters"]["normalization"]["tissue_pixel_count"] == full_count
    assert artifact["support_supervision"]["point_pose_evidence_pixel_count"] == full_count
    effective_mass = float(
        np.asarray(
            artifact["arrays"]["source_slab_brain_occupancy_float32"],
            dtype=np.float64,
        ).sum()
    )
    assert artifact["support_supervision"][
        "point_pose_evidence_effective_brain_pixel_mass"
    ] == pytest.approx(effective_mass)
    assert 0.0 < effective_mass < full_count
    assert artifact["support_supervision"][
        "point_pose_effective_mass_meets_requested_identifiability_threshold"
    ] is (effective_mass >= threshold)
    assert artifact["support_supervision"]["point_pose_supervision_evidence_metric"] == (
        "post-G1 sum(source_slab_brain_occupancy_float32)"
    )
    assert artifact["support_supervision"]["center_plane_target_pixel_count"] == center_count
    assert not artifact["support_supervision"]["point_pose_loss_gated_by_pixel_mask"]
    verify_arbitrary_plane_synthetic_realization(artifact, support)


def test_slab_outline_pairing_replay_receipts_and_model_independence_are_exact():
    parent, support = _parent()
    slab = _slab_observation_v4(parent, support)
    artifacts = {
        mode: make_arbitrary_plane_synthetic_realization(
            parent,
            support,
            slab_observation_v4=slab,
            root_seed=2**63 + 77,
            outline_mode=mode,
            config_overrides={"g3": {"disable_damage_below_pixels": 10**9}},
        )
        for mode in (ACCURATE_OUTLINE, IMPERFECT_OUTLINE, ABSENT_OUTLINE)
    }

    assert len({value["paired_view_group_id"] for value in artifacts.values()}) == 1
    assert len({value["g1"]["deformation_realization_id"] for value in artifacts.values()}) == 1
    assert len({value["g2"]["appearance_realization_id"] for value in artifacts.values()}) == 1
    assert len({value["g3"]["damage_realization_id"] for value in artifacts.values()}) == 1
    assert len({value["synthetic_realization_id"] for value in artifacts.values()}) == 3
    for artifact in artifacts.values():
        replay = replay_arbitrary_plane_synthetic_realization(artifact, support)
        verify_arbitrary_plane_synthetic_realization(artifact, support)
        assert replay["synthetic_receipt_sha256"] == artifact["synthetic_receipt_sha256"]
        assert artifact["g1"]["parameters"]["accepted_attempt"]["similarity"]["reflection"] is False
        assert artifact["slab_observation_v4"]["thickness_selection"] == slab[
            "thickness_selection"
        ]
        dependency_keys = {
            key for key in artifact["generator"] if key.endswith("_dependencies")
        }
        assert all(artifact["generator"][key] == [] for key in dependency_keys)


def test_slab_raw_array_parent_and_center_target_tampering_are_rejected():
    parent, support = _parent()
    slab = _slab_observation_v4(parent, support)
    artifact = make_arbitrary_plane_synthetic_realization(
        parent,
        support,
        slab_observation_v4=slab,
        root_seed=123,
        config_overrides={"g3": {"disable_damage_below_pixels": 10**9}},
    )

    raw = copy.deepcopy(artifact)
    raw["slab_observation_v4"]["observed_scalar_float32"][0, 0] += np.float32(1)
    with pytest.raises(ValueError, match="array receipts"):
        verify_arbitrary_plane_synthetic_realization(raw, support)
    parent_binding = copy.deepcopy(artifact)
    parent_binding["slab_observation_v4"]["finite_plane_render_id"] = "0" * 64
    with pytest.raises(ValueError, match="finite-parent binding"):
        verify_arbitrary_plane_synthetic_realization(parent_binding, support)
    center_receipt = copy.deepcopy(artifact)
    center_receipt["slab_observation_v4"][
        "centre_plane_targets_receipt_sha256"
    ] = "0" * 64
    center_receipt["slab_observation_v4"]["receipt_sha256"] = synthetic._payload_sha256_v4(
        synthetic.slab_observation_v4_receipt(center_receipt["slab_observation_v4"])
    )
    with pytest.raises(ValueError, match="center-target receipt"):
        verify_arbitrary_plane_synthetic_realization(center_receipt, support)
    psf = copy.deepcopy(slab)
    psf["finite_psf"]["axial_weights"][0] += 1e-6
    psf_payload = {
        key: value
        for key, value in psf["finite_psf"].items()
        if key != "finite_psf_sha256"
    }
    psf["finite_psf"]["finite_psf_sha256"] = synthetic._payload_sha256(
        psf_payload
    )
    psf["receipt_sha256"] = synthetic._payload_sha256_v4(
        synthetic.slab_observation_v4_receipt(psf)
    )
    with pytest.raises(ValueError, match="finite-PSF"):
        make_arbitrary_plane_synthetic_realization(
            parent, support, slab_observation_v4=psf, root_seed=123
        )


def test_authenticated_empty_slab_is_retained_and_nonfinite_input_is_rejected():
    parent, support = _parent()
    slab = _slab_observation_v4(parent, support)
    empty = copy.deepcopy(slab)
    empty["observed_scalar_float32"][:] = 0.0
    empty["slab_brain_occupancy_float32"][:] = 0.0
    empty["slab_observable_support_mask"][:] = False
    empty["centre_label_psf_mass_float32"][:] = 0.0
    empty["slab_modal_annotation_int64"][:] = 0
    empty["slab_modal_purity_float32"][:] = 0.0
    empty["dense_correspondence_weight_float32"][:] = 0.0
    empty["dense_correspondence_abstention_mask"][:] = True
    empty["array_receipts"] = {
        name: synthetic._slab_observation_array_receipt_v4(empty[name])
        for name in synthetic.SLAB_OBSERVATION_V4_ARRAY_NAMES
    }
    empty.update(
        {
            "combined_sha256": synthetic._payload_sha256_v4(
                {
                    "schema": "anatomy-tracker.slab-observation-arrays/v4",
                    "array_receipts": empty["array_receipts"],
                }
            ),
            "slab_observable_pixel_count": 0,
            "slab_effective_brain_pixel_mass": 0.0,
            "dense_abstention_pixel_count": int(
                empty["dense_correspondence_abstention_mask"].sum()
            ),
            "dense_eligible_pixel_count": 0,
            "dense_effective_supervision_mass": 0.0,
        }
    )
    empty["slab_observation_id"] = synthetic._payload_sha256_v4(
        {"empty_authenticated_ablation": empty["array_receipts"]}
    )
    empty["receipt_sha256"] = synthetic._payload_sha256_v4(
        synthetic.slab_observation_v4_receipt(empty)
    )
    retained = make_arbitrary_plane_synthetic_realization(
        parent,
        support,
        slab_observation_v4=empty,
        root_seed=1,
        outline_mode=ABSENT_OUTLINE,
    )
    assert retained["support_supervision"]["point_pose_evidence_pixel_count"] == 0
    assert retained["support_supervision"][
        "point_pose_evidence_effective_brain_pixel_mass"
    ] == 0.0
    assert not retained["support_supervision"][
        "point_pose_effective_mass_meets_requested_identifiability_threshold"
    ]
    assert retained["g2"]["parameters"]["marginal_raster_support_information_bypass"]
    assert retained["g3"]["parameters"]["marginal_raster_support_visibility_bypass"]
    assert not retained["arrays"]["source_valid_correspondence_mask"].any()
    assert retained["arrays"]["source_dense_correspondence_abstention_mask"].all()
    verify_arbitrary_plane_synthetic_realization(retained, support)
    nonfinite = copy.deepcopy(slab)
    nonfinite["observed_scalar_float32"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        make_arbitrary_plane_synthetic_realization(
            parent, support, slab_observation_v4=nonfinite, root_seed=1
        )
