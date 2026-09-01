import copy

import numpy as np
import pytest
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_subject_slab_v2 as subject_slab
import training.arbitrary_plane_synthetic_generator_v2 as slab
from training.arbitrary_plane_geometry import physical_um_to_allen_index_points
from training.arbitrary_plane_subject_deformation_v2 import (
    sample_animal_subject_deformation_plan_v2,
    subject_to_ccf_points_v2,
)
from training.arbitrary_plane_subject_section_v2 import (
    sample_coordinate_rasters_v2,
    sample_nearest_annotation_coordinate_rasters_v2,
)
from training.arbitrary_plane_support import build_annotation_support_index


def _prepared_context(*, origin=(-71.0, 23.0, 107.0), scalar_delta=0.0):
    altered = tuple(origin) != (-71.0, 23.0, 107.0) or scalar_delta != 0.0
    annotation = np.zeros((17, 15, 13), dtype=np.uint16)
    annotation[2:15, 3:13, 1:11] = 7
    annotation[7:11, 4:8, 7:11] = 19
    ap, dv, ml = np.indices(annotation.shape)
    scalar = (100 + 3 * ap + 5 * dv + 7 * ml + scalar_delta).astype(np.float32)
    support = build_annotation_support_index(
        annotation,
        atlas_id="subject-slab-fixture-ccf",
        atlas_version="fixture-v2" if altered else "fixture-v1",
        source_uri="file:///fixture/annotation.nrrd",
        source_sha256=("6" if altered else "3") * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(11.0, 17.0, 29.0),
        origin_um=origin,
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    return acquisition.prepare_arbitrary_plane_acquisition_context_v2(
        scalar,
        annotation,
        support,
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256=("5" if altered else "4") * 64,
        template_decoder="fixture decoder",
        annotation_decoder="fixture decoder",
    )


def _precursor(context, sample_index=12):
    result = slab.make_v2_smoke_global_reference_slab_render(
        context,
        "development",
        "0x415154564f320001",
        sample_index,
        acquisition.V2_PLANE_STRATA[sample_index // 4],
        animal_id="subject-2",
        animal_index=2,
        specimen_id="specimen-fixture",
        experiment_id="experiment-fixture",
    )
    slab.verify_v2_smoke_global_reference_slab_render(result, context)
    return result


def _plan(context, *, animal_id="subject-2", animal_index=2, split="development"):
    support = acquisition._context_support(context)
    lower = np.asarray(support["origin_um"], dtype=np.float64)
    upper = lower + np.asarray(support["annotation_shape"], dtype=np.float64) * np.asarray(
        support["voxel_size_um"], dtype=np.float64
    )
    return sample_animal_subject_deformation_plan_v2(
        lower,
        upper,
        root_seed="0x53454354494f4e32",
        split=split,
        animal_index=animal_index,
        animal_id=animal_id,
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


def _raw_bytes(array):
    return np.ascontiguousarray(array).tobytes(order="C")


def _rereceipt_probe(probe):
    probe["subject_centre_support_probe_id"] = acquisition._payload_sha256(
        subject_slab._support_probe_identity_payload(probe)
    )
    probe["receipt_sha256"] = acquisition._payload_sha256(
        subject_slab.subject_centre_support_probe_receipt_v2(probe)
    )


def test_shared_nearest_annotation_sampler_matches_combined_sampler_exactly():
    annotation = torch.arange(1, 28, dtype=torch.int64).reshape(3, 3, 3)
    scalar = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
    coordinates = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.5, 1.0, 1.0], [1.5, 1.0, 1.0]],
                [[2.0, 2.0, 2.0], [-0.6, 1.0, 1.0], [2.6, 1.0, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    nearest = sample_nearest_annotation_coordinate_rasters_v2(
        annotation, coordinates
    )
    _, combined = sample_coordinate_rasters_v2(scalar, annotation, coordinates)
    assert torch.equal(nearest, combined)
    assert torch.equal(
        nearest,
        torch.tensor([[[1, 5, 23], [27, 0, 0]]], dtype=torch.int64),
    )


def test_subject_ablation_abstains_outside_the_centre_brain_mask():
    reduced = subject_slab._reduce_samples_like_precursor(
        np.zeros((1, 1, 2), dtype=np.float32),
        np.asarray([[[0, 7]]], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        0,
        "centre_plane_ablation",
    )
    assert np.array_equal(reduced["dense_correspondence_weight"], [[0.0, 1.0]])
    assert np.array_equal(
        reduced["dense_correspondence_abstention_mask"], [[True, False]]
    )


@pytest.fixture(scope="module")
def prepared():
    return _prepared_context()


@pytest.fixture(scope="module")
def precursor(prepared):
    result = _precursor(prepared)
    normal = np.asarray(result["geometry"]["normal_rp2_ap_dv_ml"])
    assert np.count_nonzero(np.abs(normal) > 0.1) == 3
    assert len(result["slab_recipe"]["optical_kernel_offsets_um"]) > 1
    return result


@pytest.fixture(scope="module")
def plan(prepared):
    return _plan(prepared)


@pytest.fixture(scope="module")
def identity_artifact(prepared, precursor):
    return subject_slab.make_subject_slab_render_v2(
        prepared, precursor, subject_plan=None
    )


@pytest.fixture(scope="module")
def identity_support_probe(prepared, precursor):
    return subject_slab.make_subject_centre_support_probe_v2(
        prepared, precursor, subject_plan=None
    )


@pytest.fixture(scope="module")
def deformed_artifact(prepared, precursor, plan):
    return subject_slab.make_subject_slab_render_v2(
        prepared, precursor, subject_plan=plan, batch_size=65537
    )


def test_support_probe_is_replayable_support_only_and_bound_to_full_render(
    prepared, precursor, identity_support_probe, identity_artifact
):
    probe = identity_support_probe
    centre_index = identity_artifact["coordinate_map"]["kernel"]["centre_index"]
    assert probe["support_acceptance"]["accepted"] is True
    assert probe["support_acceptance"]["centre_plane_brain_pixel_count"] > 0
    assert probe["decision_disclosure"] == {
        "decision_inputs": [
            "mapped centre-plane Allen-index coordinates",
            "authenticated atlas annotation",
        ],
        "scalar_samples_computed": False,
        "appearance_used": False,
        "target_image_overlap_used": False,
    }
    assert probe["lineage"] == {
        "split": "development",
        "plane_sample_index": 12,
        "animal_id": "subject-2",
        "animal_index": 2,
        "specimen_id": "specimen-fixture",
        "experiment_id": "experiment-fixture",
        "synthetic_animal_id": None,
    }
    assert probe["lineage_receipt_sha256"] == acquisition._payload_sha256(
        probe["lineage"]
    )
    assert probe["mapped_centre_coordinate_receipt"] == acquisition._array_receipt(
        identity_artifact["coordinate_map"]["arrays"][
            "mapped_allen_index_coordinates_float32"
        ][centre_index]
    )
    assert probe["centre_annotation_receipt"] == acquisition._array_receipt(
        identity_artifact["sample_arrays"]["annotation_samples_int64"][centre_index]
    )
    assert identity_artifact["support_probe_reference"] == {
        "subject_centre_support_probe_id": probe[
            "subject_centre_support_probe_id"
        ],
        "receipt_sha256": probe["receipt_sha256"],
    }
    subject_slab.verify_subject_centre_support_probe_v2(
        probe, prepared, precursor, subject_plan=None
    )
    assert probe == subject_slab.replay_subject_centre_support_probe_v2(
        probe, prepared, precursor, subject_plan=None
    )


def test_support_probe_maps_only_h_w_points_before_full_slab_maps_s_h_w(
    prepared, precursor, plan, monkeypatch
):
    mapped_point_counts = []
    full_precursor_verifications = []
    original_precursor_verifier = (
        subject_slab.verify_v2_smoke_global_reference_slab_render
    )

    def counted_identity_mapping(points, subject_plan, *, batch_size=None):
        values = np.asarray(points, dtype=np.float64)
        mapped_point_counts.append(int(np.prod(values.shape[:-1])))
        return np.array(values, copy=True, order="C")

    monkeypatch.setattr(
        subject_slab, "subject_to_ccf_points_v2", counted_identity_mapping
    )

    def counted_precursor_verifier(candidate, context):
        full_precursor_verifications.append(candidate["slab_render_id"])
        return original_precursor_verifier(candidate, context)

    monkeypatch.setattr(
        subject_slab,
        "verify_v2_smoke_global_reference_slab_render",
        counted_precursor_verifier,
    )
    height, width = precursor["raster"]["scalar"].shape
    slab_levels = len(precursor["slab_recipe"]["optical_kernel_offsets_um"])

    probe = subject_slab.make_subject_centre_support_probe_v2(
        prepared, precursor, subject_plan=plan
    )
    assert probe["support_acceptance"]["accepted"] is True
    assert mapped_point_counts == [height * width]
    assert full_precursor_verifications == []

    mapped_point_counts.clear()
    subject_slab.make_subject_slab_render_v2(
        prepared, precursor, subject_plan=plan
    )
    assert mapped_point_counts == [height * width, slab_levels * height * width]
    assert full_precursor_verifications == [precursor["slab_render_id"]]


def test_support_probe_rejects_structure_source_receipt_and_coherent_lineage_tamper(
    prepared, precursor, identity_support_probe
):
    changed = copy.deepcopy(identity_support_probe)
    changed["extra"] = 1
    with pytest.raises(ValueError, match="missing or extra"):
        subject_slab.verify_subject_centre_support_probe_v2(
            changed, prepared, precursor, subject_plan=None
        )

    changed = copy.deepcopy(identity_support_probe)
    changed["implementation_source_sha256"][
        "arbitrary_plane_subject_slab_v2.py"
    ] = "0" * 64
    _rereceipt_probe(changed)
    with pytest.raises(ValueError, match="source or live receipt"):
        subject_slab.verify_subject_centre_support_probe_v2(
            changed, prepared, precursor, subject_plan=None
        )

    changed = copy.deepcopy(identity_support_probe)
    changed["mapped_centre_coordinate_receipt"]["array_sha256"] = "0" * 64
    _rereceipt_probe(changed)
    with pytest.raises(ValueError, match="replay"):
        subject_slab.verify_subject_centre_support_probe_v2(
            changed, prepared, precursor, subject_plan=None
        )

    changed = copy.deepcopy(identity_support_probe)
    changed["lineage"]["specimen_id"] = "coherently-rereceipted-other-specimen"
    changed["lineage_receipt_sha256"] = acquisition._payload_sha256(
        changed["lineage"]
    )
    _rereceipt_probe(changed)
    with pytest.raises(ValueError, match="source or live receipt"):
        subject_slab.verify_subject_centre_support_probe_v2(
            changed, prepared, precursor, subject_plan=None
        )


def test_support_probe_authenticates_rejected_discrete_centre_without_rendering(
    prepared,
):
    empty_precursor = slab.make_v2_generic_global_reference_slab_render(
        prepared,
        "audit",
        "0x47454e4552494301",
        66,
        "reference",
        nominal_cut_thickness_um=25.0,
        animal_id="audit-animal",
        animal_index=66,
        specimen_id="audit-specimen",
        experiment_id="audit-experiment",
    )
    assert empty_precursor["geometry"]["projection_origin_membership_certificate"][
        "intersects"
    ] is True
    probe = subject_slab.make_subject_centre_support_probe_v2(
        prepared, empty_precursor, subject_plan=None
    )
    assert probe["support_acceptance"] == {
        "rule": "at least one mapped centre-plane annotation pixel is nonzero",
        "centre_plane_brain_pixel_count": 0,
        "accepted": False,
        "target_image_overlap_used": False,
        "redraw_attempted": False,
    }
    subject_slab.verify_subject_centre_support_probe_v2(
        probe, prepared, empty_precursor, subject_plan=None
    )
    with pytest.raises(ValueError, match="no brain support"):
        subject_slab.make_subject_slab_render_v2(
            prepared, empty_precursor, subject_plan=None
        )


def test_identity_reuses_exact_legacy_offset_grids_and_every_reduced_array(
    prepared, precursor, identity_artifact
):
    coordinate = identity_artifact["coordinate_map"]
    arrays = coordinate["arrays"]
    assert np.array_equal(
        arrays["subject_allen_index_coordinates_float32"],
        arrays["mapped_allen_index_coordinates_float32"],
    )
    assert np.array_equal(
        arrays["subject_physical_coordinates_ap_dv_ml_um_float64"],
        arrays["mapped_ccf_physical_coordinates_ap_dv_ml_um_float64"],
    )
    for index, offset_receipt in enumerate(precursor["offset_render_receipts"]):
        assert acquisition._array_receipt(
            arrays["subject_allen_index_coordinates_float32"][index]
        ) == offset_receipt["grid_array_receipts"][
            "coordinate_raster_allen_index_float32"
        ]

    context_parent = prepared["opaque_v1_context"]
    scalar, annotation = sample_coordinate_rasters_v2(
        context_parent["scalar_tensor"],
        context_parent["annotation_tensor"].to(torch.int64),
        torch.from_numpy(arrays["mapped_allen_index_coordinates_float32"]),
    )
    assert np.array_equal(
        scalar.numpy(), identity_artifact["sample_arrays"]["scalar_samples_float32"]
    )
    assert np.array_equal(
        annotation.numpy(),
        identity_artifact["sample_arrays"]["annotation_samples_int64"],
    )
    reduced = slab.reduce_v2_slab_samples(
        scalar.numpy(),
        annotation.numpy(),
        np.asarray(precursor["slab_recipe"]["optical_kernel_integer_masses"], dtype=np.int64),
        coordinate["kernel"]["centre_index"],
    )
    reduced_arrays = subject_slab._reduced_arrays(reduced)
    artifact_arrays = subject_slab._reduced_arrays(identity_artifact["raster"])
    precursor_arrays = subject_slab._reduced_arrays(precursor["raster"])
    for name in precursor_arrays:
        assert _raw_bytes(reduced_arrays[name]) == _raw_bytes(precursor_arrays[name])
        assert _raw_bytes(artifact_arrays[name]) == _raw_bytes(precursor_arrays[name])
    outside = ~artifact_arrays["centre_plane_support_mask"]
    assert outside.any()
    assert not artifact_arrays["dense_correspondence_weight"][outside].any()
    assert artifact_arrays["dense_correspondence_abstention_mask"][outside].all()
    assert identity_artifact["identity_reference_path"] is True
    assert identity_artifact["synthetic_animal_id"] is None
    assert identity_artifact["precursor_reference"]["precursor_contract"] == (
        "frozen-smoke-v2"
    )
    assert identity_artifact["support_acceptance"]["accepted"] is True
    subject_slab.verify_subject_slab_render_v2(
        identity_artifact, prepared, precursor, subject_plan=None
    )


def test_nonidentity_maps_every_oblique_slab_level_subject_to_ccf_directly(
    prepared, precursor, plan, deformed_artifact
):
    arrays = deformed_artifact["coordinate_map"]["arrays"]
    expected_physical = subject_to_ccf_points_v2(
        arrays["subject_physical_coordinates_ap_dv_ml_um_float64"],
        plan,
        batch_size=49157,
    )
    support = acquisition._context_support(prepared)
    expected_allen = (
        physical_um_to_allen_index_points(
            torch.from_numpy(expected_physical),
            tuple(support["origin_um"]),
            tuple(support["voxel_size_um"]),
        )
        .to(torch.float32)
        .numpy()
    )
    assert np.array_equal(
        arrays["mapped_ccf_physical_coordinates_ap_dv_ml_um_float64"],
        expected_physical,
    )
    assert np.array_equal(
        arrays["mapped_allen_index_coordinates_float32"], expected_allen
    )
    assert all(
        not np.array_equal(expected_physical[index], expected_physical[index + 1])
        for index in range(expected_physical.shape[0] - 1)
    )
    fit = deformed_artifact["coordinate_map"]["centre_plane_fit"]
    centre = deformed_artifact["coordinate_map"]["kernel"]["centre_index"]
    assert np.allclose(
        fit["arrays"]["fitted_coordinate_raster_ap_dv_ml_um_float64"]
        + fit["arrays"]["residual_coordinate_field_ap_dv_ml_um_float64"],
        expected_physical[centre],
        atol=2e-15,
        rtol=0.0,
    )
    assert fit["diagnostics"]["residual_rms_um"] > 0.0
    assert deformed_artifact["synthetic_animal_id"] == plan["synthetic_animal_id"]
    assert deformed_artifact["coordinate_map"]["synthetic_animal_id"] == plan[
        "synthetic_animal_id"
    ]
    subject_slab.verify_subject_slab_render_v2(
        deformed_artifact, prepared, precursor, subject_plan=plan
    )


def test_authoritative_context_plan_precursor_and_atlas_mismatches_are_rejected(
    prepared, precursor, plan, deformed_artifact
):
    wrong_plan = _plan(prepared, animal_id="subject-other-label")
    with pytest.raises(ValueError, match="animal lineage"):
        subject_slab.make_subject_slab_render_v2(
            prepared, precursor, subject_plan=wrong_plan
        )
    with pytest.raises(ValueError, match="binding|replay"):
        subject_slab.verify_subject_slab_render_v2(
            deformed_artifact, prepared, precursor, subject_plan=wrong_plan
        )

    wrong_index_plan = _plan(prepared, animal_index=3)
    with pytest.raises(ValueError, match="animal lineage"):
        subject_slab.make_subject_slab_render_v2(
            prepared, precursor, subject_plan=wrong_index_plan
        )

    wrong_split_plan = _plan(prepared, split="train")
    with pytest.raises(ValueError, match="animal lineage"):
        subject_slab.make_subject_slab_render_v2(
            prepared, precursor, subject_plan=wrong_split_plan
        )

    other_precursor = _precursor(prepared, sample_index=16)
    with pytest.raises(ValueError, match="binding|replay"):
        subject_slab.verify_subject_slab_render_v2(
            deformed_artifact, prepared, other_precursor, subject_plan=plan
        )

    altered_context = _prepared_context(origin=(-69.0, 23.0, 107.0), scalar_delta=1.0)
    altered_precursor = _precursor(altered_context)
    with pytest.raises(ValueError, match="binding|context|bounds|replay"):
        subject_slab.verify_subject_slab_render_v2(
            deformed_artifact,
            altered_context,
            altered_precursor,
            subject_plan=plan,
        )


def test_exact_nested_schemas_live_receipts_and_source_tamper_rejection(
    prepared, precursor, identity_artifact
):
    changed = copy.deepcopy(identity_artifact)
    changed["raster"]["slab_supervision_weight_or_abstention"]["extra"] = 1
    with pytest.raises(ValueError, match="extra fields"):
        subject_slab.verify_subject_slab_render_v2(
            changed, prepared, precursor, subject_plan=None
        )

    changed = copy.deepcopy(identity_artifact)
    changed["coordinate_map"]["atlas_domain"]["extra"] = 1
    with pytest.raises(ValueError, match="extra fields"):
        subject_slab.verify_subject_slab_render_v2(
            changed, prepared, precursor, subject_plan=None
        )

    changed = copy.deepcopy(identity_artifact)
    changed["sample_arrays"]["scalar_samples_float32"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="live array receipt"):
        subject_slab.verify_subject_slab_render_v2(
            changed, prepared, precursor, subject_plan=None
        )

    changed = copy.deepcopy(identity_artifact)
    changed["implementation_source_sha256"][
        "arbitrary_plane_subject_slab_v2.py"
    ] = "0" * 64
    changed["coordinate_map"]["implementation_source_sha256"][
        "arbitrary_plane_subject_slab_v2.py"
    ] = "0" * 64
    changed["coordinate_map"]["subject_coordinate_map_id"] = acquisition._payload_sha256(
        subject_slab._coordinate_identity_payload(changed["coordinate_map"])
    )
    changed["subject_coordinate_map_id"] = changed["coordinate_map"][
        "subject_coordinate_map_id"
    ]
    changed["subject_slab_render_id"] = acquisition._payload_sha256(
        subject_slab._render_identity_payload(changed)
    )
    changed["receipt_sha256"] = acquisition._payload_sha256(
        subject_slab.subject_slab_render_receipt_v2(changed)
    )
    with pytest.raises(ValueError, match="source"):
        subject_slab.verify_subject_slab_render_v2(
            changed, prepared, precursor, subject_plan=None
        )
