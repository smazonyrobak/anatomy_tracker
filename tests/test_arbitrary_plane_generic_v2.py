import copy

import numpy as np
import pytest
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_subject_slab_v2 as subject_slab
import training.arbitrary_plane_synthetic_generator_v2 as slab
from training.arbitrary_plane_subject_deformation_v2 import (
    sample_animal_subject_deformation_plan_v2,
)
from training.arbitrary_plane_support import build_annotation_support_index


def _context():
    annotation = np.zeros((17, 15, 13), dtype=np.uint16)
    annotation[2:15, 3:13, 1:11] = 7
    annotation[7:11, 4:8, 7:11] = 19
    ap, dv, ml = np.indices(annotation.shape)
    scalar = (100 + 3 * ap + 5 * dv + 7 * ml).astype(np.float32)
    support = build_annotation_support_index(
        annotation,
        atlas_id="generic-fixture-ccf",
        atlas_version="fixture-v1",
        source_uri="file:///fixture/annotation.nrrd",
        source_sha256="3" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(11.0, 17.0, 29.0),
        origin_um=(-71.0, 23.0, 107.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    return acquisition.prepare_arbitrary_plane_acquisition_context_v2(
        scalar,
        annotation,
        support,
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="4" * 64,
        template_decoder="fixture decoder",
        annotation_decoder="fixture decoder",
    )


@pytest.fixture(scope="module")
def prepared():
    return _context()


def _centre(context, sample_index=31, stratum="reference", **lineage):
    return acquisition.make_v2_generic_global_reference_centre_render(
        context,
        "custom-train-fold",
        "0x47454e4552494301",
        sample_index,
        stratum,
        animal_id=lineage.get("animal_id", "animal-5"),
        animal_index=lineage.get("animal_index", 5),
        specimen_id="specimen-5a",
        experiment_id="experiment-5",
    )


def _generic_slab(
    context, *, animal_id="animal-5", animal_index=5, split="train"
):
    return slab.make_v2_generic_global_reference_slab_render(
        context,
        split,
        "0x47454e4552494302",
        41,
        "general_oblique",
        nominal_cut_thickness_um=55.0,
        animal_id=animal_id,
        animal_index=animal_index,
        specimen_id="specimen-5a",
        experiment_id="experiment-5",
    )


def _plan(context):
    support = acquisition._context_support(context)
    lower = np.asarray(support["origin_um"], dtype=np.float64)
    upper = lower + np.asarray(support["annotation_shape"], dtype=np.float64) * np.asarray(
        support["voxel_size_um"], dtype=np.float64
    )
    return sample_animal_subject_deformation_plan_v2(
        lower,
        upper,
        root_seed="0x47454e4552494303",
        split="train",
        animal_index=5,
        animal_id="animal-5",
        ccf_context_sha256=context["v2_context_sha256"],
        coarse_spacing_um=500.0,
        fine_spacing_um=250.0,
        coarse_padding_um=2000.0,
        fine_padding_um=1000.0,
        smoothing_sigma_knots=0.7,
        a0_um=5.0,
        max_local_displacement_um=100.0,
        speed_l2_bound_um_max=100.0,
        minimum_halo_um=0.0,
        integration_steps=2,
    )


def test_reference_and_named_stress_strata_are_deterministic_and_distinct(prepared):
    support = acquisition._context_support(prepared)
    samples = {
        stratum: acquisition.sample_v2_generic_plane_pose(
            support,
            "arbitrary-split-name",
            "0x47454e4552494301",
            index,
            stratum,
        )
        for index, stratum in enumerate(
            ("reference", "near_AP", "general_oblique", "edge_or_partial"), 11
        )
    }
    for index, (stratum, sample) in enumerate(samples.items(), 11):
        replay = acquisition.sample_v2_generic_plane_pose(
            support,
            "arbitrary-split-name",
            "0x47454e4552494301",
            index,
            stratum,
        )
        normal = np.asarray(sample["normal_rp2_ap_dv_ml"])
        intervals = np.asarray(sample["shifted_intervals"]["support_origin_interval_union_um"])
        assert sample == replay
        assert np.isclose(np.linalg.norm(normal), 1.0)
        assert np.any(
            (sample["signed_offset_um_about_support_origin"] >= intervals[:, 0])
            & (sample["signed_offset_um_about_support_origin"] <= intervals[:, 1])
        )
        assert sample["animal_label_rng_dependencies"] == []
        assert sample["rejection_attempts"] == []
    assert samples["reference"]["reference_measure"] is True
    assert "Haar-uniform RP2" in samples["reference"]["normal_sampling_measure"]
    assert abs(samples["reference"]["normal_rp2_ap_dv_ml"][1]) > 0.05
    assert abs(samples["reference"]["normal_rp2_ap_dv_ml"][2]) > 0.05
    assert abs(samples["near_AP"]["normal_rp2_ap_dv_ml"][0]) >= 0.9
    assert np.min(np.abs(samples["general_oblique"]["normal_rp2_ap_dv_ml"])) > 0.2
    edge_fraction = samples["edge_or_partial"]["offset_measure_fraction"]
    assert edge_fraction < 0.03 or edge_fraction > 0.97
    same_index_reference = acquisition.sample_v2_generic_plane_pose(
        support, "arbitrary-split-name", "0x47454e4552494301", 99, "reference"
    )
    same_index_edge = acquisition.sample_v2_generic_plane_pose(
        support, "arbitrary-split-name", "0x47454e4552494301", 99, "edge_or_partial"
    )
    assert (
        same_index_reference["field_stream_seed_uint64"][
            "isotropic-gaussian-normal"
        ]
        != same_index_edge["field_stream_seed_uint64"][
            "isotropic-gaussian-normal"
        ]
    )
    assert same_index_reference["field_stream_stage"] != same_index_edge[
        "field_stream_stage"
    ]


def test_reference_sampler_empirically_matches_haar_rp2_marginals(prepared):
    support = acquisition._context_support(prepared)
    normals = np.asarray(
        [
            acquisition.sample_v2_generic_plane_pose(
                support,
                "measure-audit",
                "0x47454e4552494301",
                index,
                "reference",
            )["normal_rp2_ap_dv_ml"]
            for index in range(512)
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        np.mean(normals * normals, axis=0), np.full(3, 1.0 / 3.0), atol=0.04, rtol=0.0
    )
    np.testing.assert_allclose(
        np.quantile(np.abs(normals).ravel(), [0.1, 0.5, 0.9]),
        [0.1, 0.5, 0.9],
        atol=0.04,
        rtol=0.0,
    )


def test_disconnected_interval_union_uses_total_length_measure():
    annotation = np.zeros((12, 3, 3), dtype=np.uint16)
    annotation[1, 1, 1] = 7
    annotation[7:10, 1, 1] = 7
    support = build_annotation_support_index(
        annotation,
        atlas_id="disconnected-fixture-ccf",
        atlas_version="fixture-v1",
        source_uri="file:///fixture/disconnected-annotation.nrrd",
        source_sha256="8" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(1.0, 1.0, 1.0),
        origin_um=(0.0, 0.0, 0.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    intervals = np.asarray(
        acquisition.shifted_component_interval_union(
            np.asarray([1.0, 0.0, 0.0]), support
        )["support_origin_interval_union_um"],
        dtype=np.float64,
    )
    np.testing.assert_allclose(intervals[:, 1] - intervals[:, 0], [1.0, 3.0])
    fractions = (np.arange(400, dtype=np.float64) + 0.5) / 400.0
    draws = [acquisition._offset_at_measure_fraction(intervals, value) for value in fractions]
    assert np.array_equal(np.bincount([index for _, index in draws]), [100, 300])
    assert all(intervals[index, 0] <= offset <= intervals[index, 1] for offset, index in draws)


def test_generic_centre_is_label_independent_fixed_canvas_and_exactly_replayable(prepared):
    first = _centre(prepared)
    renamed = _centre(prepared, animal_id="renamed", animal_index=99)
    replay = acquisition.replay_v2_generic_global_reference_centre_render(first, prepared)
    acquisition.verify_v2_generic_global_reference_centre_render(first, prepared)
    assert first["sampling"] == renamed["sampling"]
    assert first["geometry"]["global_reference_grid_id"] == renamed["geometry"][
        "global_reference_grid_id"
    ]
    assert first["v2_plane_realization_id"] != renamed["v2_plane_realization_id"]
    assert first["geometry"]["output_shape_h_w"] == [256, 256]
    assert first["geometry"]["raster_endpoint_semantics"]["pixel_mapping"] == (
        "P(x,y)=O+(x/W)U+(y/H)V"
    )
    height, width = first["geometry"]["output_shape_h_w"]
    origin, edge_u, edge_v = np.asarray(
        first["geometry"]["physical_ouv_ap_dv_ml_um"], dtype=np.float64
    ).reshape(3, 3)
    np.testing.assert_array_equal(
        first["geometry"]["raster_endpoint_semantics"]["first_sample_ap_dv_ml_um"],
        origin,
    )
    np.testing.assert_allclose(
        first["geometry"]["raster_endpoint_semantics"]["last_sample_ap_dv_ml_um"],
        origin + ((width - 1) / width) * edge_u + ((height - 1) / height) * edge_v,
        rtol=0.0,
        atol=1e-12,
    )
    assert first["generator"]["resolved_config"]["split"] == "custom-train-fold"
    assert "preflight" not in first["generator"]["resolved_config"]
    for key in (
        "learned_checkpoint_dependencies",
        "previous_model_dependencies",
        "pretrained_feature_dependencies",
        "learned_style_model_dependencies",
    ):
        assert first["generator"]["resolved_config"][key] == []
    assert "synthetic_realization_id" not in first
    assert acquisition.v2_generic_centre_render_receipt(first) == (
        acquisition.v2_generic_centre_render_receipt(replay)
    )
    for name in ("scalar", "annotation", "brain_mask"):
        assert np.array_equal(first["raster"][name], replay["raster"][name])


def test_generic_path_never_calls_the_smoke_preflight(prepared, monkeypatch):
    monkeypatch.setattr(
        acquisition,
        "_preflight_provenance",
        lambda: (_ for _ in ()).throw(AssertionError("smoke preflight read")),
    )
    acquisition.verify_v2_generic_global_reference_centre_render(
        _centre(prepared, sample_index=32), prepared
    )


@pytest.mark.parametrize(
    ("sample_index", "stratum"),
    ((51, "near_AP"), (52, "general_oblique"), (53, "edge_or_partial")),
)
def test_generic_named_stress_strata_render_brain_on_the_same_canvas(
    prepared, sample_index, stratum
):
    artifact = _centre(prepared, sample_index=sample_index, stratum=stratum)
    acquisition.verify_v2_generic_global_reference_centre_render(artifact, prepared)
    assert artifact["raster"]["brain_pixel_count"] > 0
    assert artifact["geometry"]["output_shape_h_w"] == [256, 256]


def test_generic_centre_rejects_extra_and_coherently_receipted_rng_tamper(prepared):
    artifact = _centre(prepared)
    extra = copy.deepcopy(artifact)
    extra["unexpected"] = 1
    with pytest.raises(ValueError, match="extra fields"):
        acquisition.verify_v2_generic_global_reference_centre_render(extra, prepared)

    changed = copy.deepcopy(artifact)
    config = changed["generator"]["resolved_config"]
    config["root_seed_uint64"] = "0x47454e4552494304"
    changed["generator"]["resolved_config_sha256"] = acquisition._payload_sha256(config)
    changed["v2_plane_realization_id"] = acquisition._payload_sha256(
        acquisition._generic_plane_identity_payload(changed)
    )
    changed["centre_plane_render_id"] = acquisition._payload_sha256(
        acquisition._generic_centre_render_identity_payload(changed)
    )
    changed["receipt_sha256"] = acquisition._payload_sha256(
        acquisition.v2_generic_centre_render_receipt(changed)
    )
    with pytest.raises(ValueError, match="replay"):
        acquisition.verify_v2_generic_global_reference_centre_render(changed, prepared)


@pytest.mark.parametrize(
    ("animal_id", "animal_index", "error"),
    (
        (None, 5, "animal lineage"),
        ("animal-5", None, "animal lineage"),
        ("animal-5", -1, "animal_index"),
    ),
)
def test_generic_training_makers_require_animal_lineage(
    prepared, animal_id, animal_index, error
):
    with pytest.raises(ValueError, match=error):
        acquisition.make_v2_generic_global_reference_centre_render(
            prepared,
            "train",
            "0x47454e4552494301",
            31,
            "reference",
            animal_id=animal_id,
            animal_index=animal_index,
        )
    with pytest.raises(ValueError, match=error):
        slab.make_v2_generic_global_reference_slab_render(
            prepared,
            "train",
            "0x47454e4552494302",
            41,
            "general_oblique",
            nominal_cut_thickness_um=25.0,
            animal_id=animal_id,
            animal_index=animal_index,
        )


def test_generic_replay_and_verifiers_reject_rereceipted_null_lineage(prepared):
    centre = copy.deepcopy(_centre(prepared))
    centre["provenance"]["animal_id"] = None
    centre["v2_plane_realization_id"] = acquisition._payload_sha256(
        acquisition._generic_plane_identity_payload(centre)
    )
    centre["centre_plane_render_id"] = acquisition._payload_sha256(
        acquisition._generic_centre_render_identity_payload(centre)
    )
    centre["receipt_sha256"] = acquisition._payload_sha256(
        acquisition.v2_generic_centre_render_receipt(centre)
    )
    with pytest.raises(ValueError, match="animal lineage"):
        acquisition.verify_v2_generic_global_reference_centre_render(centre, prepared)
    with pytest.raises(ValueError, match="animal lineage"):
        acquisition.replay_v2_generic_global_reference_centre_render(centre, prepared)

    finite = copy.deepcopy(_generic_slab(prepared))
    finite["provenance"]["animal_index"] = None
    finite["slab_render_id"] = acquisition._payload_sha256(
        slab._slab_render_identity_payload(finite)
    )
    finite["receipt_sha256"] = acquisition._payload_sha256(
        slab.v2_generic_slab_render_receipt(finite)
    )
    with pytest.raises(ValueError, match="animal lineage"):
        slab.verify_v2_generic_global_reference_slab_render(finite, prepared)
    with pytest.raises(ValueError, match="animal lineage"):
        slab.replay_v2_generic_global_reference_slab_render(finite, prepared)


def test_generic_finite_slab_replays_and_rejects_weak_schema_dispatch(prepared):
    artifact = _generic_slab(prepared)
    replay = slab.replay_v2_generic_global_reference_slab_render(artifact, prepared)
    slab.verify_v2_generic_global_reference_slab_render(artifact, prepared)
    assert artifact["schema_version"] == slab.V2_GENERIC_SLAB_SCHEMA
    assert artifact["slab_recipe"]["render_mode"] == "finite_boxcar"
    assert artifact["slab_recipe"]["nominal_cut_thickness_um"] == 55.0
    assert "preflight" not in artifact["generator"]["resolved_config"]
    assert "smoke_case_assignment" not in artifact
    assert "synthetic_realization_id" not in artifact
    for name, array in slab._slab_arrays(artifact["raster"]).items():
        assert np.array_equal(array, slab._slab_arrays(replay["raster"])[name])

    extra = copy.deepcopy(artifact)
    extra["unexpected"] = 1
    with pytest.raises(ValueError, match="extra fields"):
        slab.verify_v2_generic_global_reference_slab_render(extra, prepared)

    changed = copy.deepcopy(artifact)
    changed["slab_recipe"]["nominal_cut_thickness_um"] = 56.0
    changed["slab_recipe_id"] = acquisition._payload_sha256(changed["slab_recipe"])
    changed["slab_render_id"] = acquisition._payload_sha256(
        slab._slab_render_identity_payload(changed)
    )
    changed["receipt_sha256"] = acquisition._payload_sha256(
        slab.v2_generic_slab_render_receipt(changed)
    )
    with pytest.raises(ValueError, match="replay"):
        slab.verify_v2_generic_global_reference_slab_render(changed, prepared)

    weak = copy.deepcopy(artifact)
    weak["algorithm"] = slab.V2_SLAB_ALGORITHM
    with pytest.raises(ValueError, match="schema/algorithm pair"):
        subject_slab.make_subject_slab_render_v2(prepared, weak, subject_plan=None)


@pytest.fixture(scope="module")
def generic_precursor(prepared):
    return _generic_slab(prepared)


@pytest.fixture(scope="module")
def subject_plan(prepared):
    return _plan(prepared)


def test_generic_subject_pullback_binds_lineage_support_and_replay(
    prepared, generic_precursor, subject_plan
):
    artifact = subject_slab.make_subject_slab_render_v2(
        prepared, generic_precursor, subject_plan=subject_plan, batch_size=65537
    )
    subject_slab.verify_subject_slab_render_v2(
        artifact, prepared, generic_precursor, subject_plan=subject_plan
    )
    assert artifact["precursor_reference"]["precursor_contract"] == (
        "authenticated-generic-v2"
    )
    assert artifact["synthetic_animal_id"] == subject_plan["synthetic_animal_id"]
    assert artifact["support_acceptance"]["accepted"] is True
    assert artifact["support_acceptance"]["centre_plane_brain_pixel_count"] > 0
    assert artifact["support_acceptance"]["target_image_overlap_used"] is False
    assert artifact["support_acceptance"]["redraw_attempted"] is False

    for wrong_lineage in (
        _generic_slab(prepared, animal_id="animal-other"),
        _generic_slab(prepared, animal_index=6),
        _generic_slab(prepared, split="heldout-animal-fold"),
    ):
        with pytest.raises(ValueError, match="animal lineage"):
            subject_slab.make_subject_slab_render_v2(
                prepared, wrong_lineage, subject_plan=subject_plan
            )


def test_subject_pullback_hard_fails_zero_mapped_centre_support_without_redraw(
    prepared, generic_precursor, subject_plan, monkeypatch
):
    def zero_sampler(scalar, annotation, coordinates):
        shape = tuple(coordinates.shape[:-1])
        return torch.zeros(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.int64)

    monkeypatch.setattr(subject_slab, "sample_coordinate_rasters_v2", zero_sampler)
    with pytest.raises(ValueError, match="no brain support"):
        subject_slab.make_subject_slab_render_v2(
            prepared, generic_precursor, subject_plan=subject_plan, batch_size=65537
        )


def test_continuous_support_intersection_can_be_discretely_empty_and_subject_gate_rejects(
    prepared,
):
    centre = acquisition.make_v2_generic_global_reference_centre_render(
        prepared,
        "audit",
        "0x47454e4552494301",
        66,
        "reference",
        animal_id="audit-animal",
        animal_index=66,
    )
    assert centre["geometry"]["projection_origin_membership_certificate"][
        "intersects"
    ] is True
    assert centre["raster"]["brain_pixel_count"] == 0

    precursor = slab.make_v2_generic_global_reference_slab_render(
        prepared,
        "audit",
        "0x47454e4552494301",
        66,
        "reference",
        nominal_cut_thickness_um=25.0,
        animal_id="audit-animal",
        animal_index=66,
    )
    assert precursor["raster"]["centre_plane_brain_pixel_count"] == 0
    with pytest.raises(ValueError, match="no brain support"):
        subject_slab.make_subject_slab_render_v2(
            prepared, precursor, subject_plan=None
        )
