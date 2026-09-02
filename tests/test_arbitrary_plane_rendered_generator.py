import copy
import json

import numpy as np
import pytest

import training.arbitrary_plane_rendered_generator as rendered
from training.arbitrary_plane_rendered_generator import (
    BOUNDARY_STRESS_STRATUM,
    FINITE_RENDER_SCHEMA,
    REFERENCE_STRATUM,
    component_interval_union,
    effective_renderer_sampling_arrays,
    finite_plane_raster_geometry,
    finite_render_receipt,
    make_finite_arbitrary_plane_render,
    make_finite_arbitrary_plane_render_from_context,
    oriented_support_projection_bounds,
    physical_plane_frame,
    prepare_finite_render_context,
    render_finite_arbitrary_plane,
    replay_finite_arbitrary_plane_render,
    replay_finite_arbitrary_plane_render_from_context,
    sample_interval_union_offset,
    verify_finite_arbitrary_plane_render,
)
from training.arbitrary_plane_manifest import canonicalize_plane
from training.arbitrary_plane_support import build_annotation_support_index


def _volumes(annotation_dtype=np.uint16, spacing=(11.0, 17.0, 29.0)):
    annotation = np.zeros((17, 15, 13), dtype=annotation_dtype)
    annotation[2:15, 3:13, 1:11] = 7
    annotation[6:11, 6:10, 4:8] = 19
    ap, dv, ml = np.indices(annotation.shape)
    template = (100 + 3 * ap + 5 * dv + 7 * ml).astype(np.uint16)
    support = build_annotation_support_index(
        annotation,
        atlas_id="fixture-ccf",
        atlas_version="fixture-v1",
        source_uri="file:///fixture/annotation.nrrd",
        source_sha256="3" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=spacing,
        origin_um=(-71.0, 23.0, 107.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    return template, annotation, support


def _make(template=None, annotation=None, support=None, **kwargs):
    if template is None:
        template, annotation, support = _volumes()
    return make_finite_arbitrary_plane_render(
        template,
        annotation,
        support,
        kwargs.pop("split", "development"),
        kwargs.pop("seed", 2**63 + 101),
        kwargs.pop("output_shape", (47, 53)),
        sample_index=kwargs.pop("sample_index", 29),
        margin_um=kwargs.pop("margin_um", (13.0, 17.0)),
        scalar_source_uri=kwargs.pop("scalar_source_uri", "file:///fixture/template.nrrd"),
        scalar_source_sha256=kwargs.pop("scalar_source_sha256", "4" * 64),
        template_decoder=kwargs.pop("template_decoder", "pynrrd 1.1.3"),
        template_index_order=kwargs.pop("template_index_order", "F"),
        annotation_decoder=kwargs.pop("annotation_decoder", "pynrrd 1.1.3"),
        annotation_index_order=kwargs.pop("annotation_index_order", "F"),
        **kwargs,
    )


@pytest.mark.parametrize(
    "normal",
    (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, -2.0, 3.0],
        [1e-7, 1.0, -2e-7],
    ),
)
def test_cardinal_and_extreme_oblique_physical_frames_render_nonempty(normal):
    template, annotation, support = _volumes(spacing=(9.0, 21.0, 37.0))
    interval = component_interval_union(normal, support)[0]
    offset = float(interval.mean())
    geometry = finite_plane_raster_geometry(normal, offset, 1.137, support, (49, 55), (7.0, 11.0))
    raster = render_finite_arbitrary_plane(template.astype(np.float32), annotation, geometry)
    frame = np.asarray(geometry["frame_ap_dv_ml_physical"])

    assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(frame[:, 2], canonicalize_plane(normal, offset)[0])
    assert raster["brain_pixel_count"] > 0
    assert np.array_equal(raster["brain_mask"], raster["annotation"] != 0)
    assert geometry["sampling_contract"] == "quicknii-raster-index-x-over-W-y-over-H-v1"
    assert geometry["reflection_state"] == {
        "horizontal": False,
        "vertical": False,
        "status": "no raster reflection sampled in finite-geometry precursor v1",
    }


def test_official_quicknii_endpoints_cover_oriented_support_with_anisotropic_spacing():
    _, _, support = _volumes(spacing=(9.0, 21.0, 37.0))
    normal = np.asarray([0.0, 0.0, 1.0])
    offset = float(component_interval_union(normal, support)[0].mean())
    height, width = 31, 43
    geometry = finite_plane_raster_geometry(normal, offset, 0.0, support, (height, width), (13.0, 17.0))
    frame = np.asarray(geometry["frame_ap_dv_ml_physical"])
    u, v, n = frame.T
    physical_ouv = np.asarray(geometry["physical_ouv_ap_dv_ml_um"])
    origin, edge_u, edge_v = physical_ouv[:3], physical_ouv[3:6], physical_ouv[6:9]
    center = np.asarray(support["projection_origin_um"]) + offset * n
    last = origin + (width - 1) / width * edge_u + (height - 1) / height * edge_v
    u_bounds = np.asarray(geometry["sampled_endpoint_bounds_u_um"])
    v_bounds = np.asarray(geometry["sampled_endpoint_bounds_v_um"])
    spacing = np.asarray(support["voxel_size_um"])
    atlas_origin = np.asarray(support["origin_um"])
    index_ouv = np.asarray(geometry["allen_index_ouv_ap_dv_ml"])

    assert np.dot(origin - center, u) == pytest.approx(u_bounds[0], abs=1e-10)
    assert np.dot(last - center, u) == pytest.approx(u_bounds[1], abs=1e-10)
    assert np.dot(origin - center, v) == pytest.approx(v_bounds[0], abs=1e-10)
    assert np.dot(last - center, v) == pytest.approx(v_bounds[1], abs=1e-10)
    assert np.allclose(atlas_origin + (index_ouv[:3] + 0.5) * spacing, origin)
    assert np.allclose(index_ouv[3:6] * spacing, edge_u)
    assert np.allclose(index_ouv[6:9] * spacing, edge_v)
    assert geometry["raster_endpoint_semantics"]["u_edge_factor"] == pytest.approx(width / (width - 1))
    assert geometry["raster_endpoint_semantics"]["v_edge_factor"] == pytest.approx(height / (height - 1))
    assert geometry["array_receipts"]["effective_coordinate_raster_allen_index_float32"]["shape"] == [height, width, 3]
    assert geometry["array_receipts"]["effective_normalized_interpolation_grid_xyz_float32"]["shape"] == [height, width, 3]
    assert geometry["array_receipts"]["valid_atlas_label_sampling_mask"]["dtype"] == "|b1"
    required_u = np.asarray(geometry["required_endpoint_bounds_u_um"])
    required_v = np.asarray(geometry["required_endpoint_bounds_v_um"])
    assert u_bounds[0] <= required_u[0] < required_u[1] <= u_bounds[1]
    assert v_bounds[0] <= required_v[0] < required_v[1] <= v_bounds[1]
    assert geometry["reference_aspect_policy"]["pixel_pitch_u_um"] == pytest.approx(
        geometry["reference_aspect_policy"]["pixel_pitch_v_um"]
    )


def test_oriented_projection_does_not_mirror_an_rp2_folded_tangent():
    _, _, support = _volumes()
    canonical = oriented_support_projection_bounds([1.0, 0.0, 0.0], support)
    folded = oriented_support_projection_bounds([-1.0, 0.0, 0.0], support)
    normal = [0.0, 0.0, 1.0]
    frame = physical_plane_frame(normal, 0.0)
    offset = float(component_interval_union(normal, support)[0].mean())
    geometry = finite_plane_raster_geometry(normal, offset, 0.0, support, (31, 33))

    assert np.array_equal(folded, np.stack((-canonical[:, 1], -canonical[:, 0]), axis=-1))
    assert np.array_equal(frame[:, 0], [0.0, -1.0, 0.0])
    assert np.array_equal(
        np.asarray(geometry["component_projection_bounds_u_um"]),
        oriented_support_projection_bounds(frame[:, 0], support),
    )


def test_disjoint_unequal_and_overlapping_intervals_are_length_weighted_without_gap_samples():
    raw = np.asarray([[0.0, 1.0], [0.5, 2.0], [10.0, 14.0]])
    sampled = np.asarray([sample_interval_union_offset(raw, seed)[0] for seed in range(6000)])
    in_first = (sampled >= 0.0) & (sampled <= 2.0)
    in_second = (sampled >= 10.0) & (sampled <= 14.0)

    assert np.all(in_first | in_second)
    assert in_second.mean() == pytest.approx(4 / 6, abs=0.025)

    annotation = np.zeros((19, 5, 5), dtype=np.uint16)
    annotation[1:3, 1:4, 1:4] = 1
    annotation[13:18, 1:4, 1:4] = 2
    _, _, support = _volumes()
    support = build_annotation_support_index(
        annotation,
        atlas_id="fixture-ccf",
        atlas_version="fixture-v1",
        source_uri="file:///fixture/disconnected.nrrd",
        source_sha256="5" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(10.0, 10.0, 10.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    intervals = component_interval_union([1.0, 0.0, 0.0], support)
    offsets = np.asarray([sample_interval_union_offset(intervals, seed)[0] for seed in range(1000)])
    assert len(intervals) == 2
    assert not np.any((offsets > intervals[0, 1]) & (offsets < intervals[1, 0]))


def test_sampled_normal_is_equal_area_roll_is_uniform_and_split_sample_domains_differ():
    count = 6000
    normals = []
    rolls = []
    for sample_index in range(count):
        normal_seed = rendered._derived_seed(911, "train", sample_index, "normal", 0)
        roll_seed = rendered._derived_seed(911, "train", sample_index, "roll", 0)
        raw = np.random.Generator(np.random.PCG64(normal_seed)).normal(size=3)
        normals.append(canonicalize_plane(raw, 0.0)[0])
        rolls.append(np.random.Generator(np.random.PCG64(roll_seed)).uniform(0.0, 2.0 * np.pi))
    normals = np.asarray(normals)
    rolls = np.asarray(rolls)

    assert np.mean(normals**2, axis=0) == pytest.approx([1 / 3] * 3, abs=0.018)
    assert abs(np.mean(np.exp(1j * rolls))) < 0.025
    assert rendered._derived_seed(911, "train", 3, "normal", 0) != rendered._derived_seed(
        911, "development", 3, "normal", 0
    )
    assert rendered._derived_seed(911, "train", 3, "normal", 0) != rendered._derived_seed(
        911, "train", 4, "normal", 0
    )


@pytest.mark.parametrize("annotation_dtype", [np.uint16, np.uint32])
def test_unsigned_annotation_renders_losslessly_as_int64(annotation_dtype):
    template, annotation, support = _volumes(annotation_dtype)
    artifact = _make(template, annotation, support)

    assert artifact["raster"]["annotation"].dtype == np.int64
    assert artifact["provenance"]["annotation_decoded"]["dtype"] == annotation.dtype.str
    assert artifact["generator"]["resolved_config"]["annotation_decoded_dtype"] == annotation.dtype.str
    assert set(np.unique(artifact["raster"]["annotation"])) <= {0, 7, 19}


def test_render_is_replayable_hash_bound_json_receipted_and_model_independent():
    template, annotation, support = _volumes()
    artifact = _make(
        template,
        annotation,
        support,
        stratum=BOUNDARY_STRESS_STRATUM,
        boundary_stress_fraction=0.15,
        animal_id="animal-7",
        specimen_id="specimen-7a",
        experiment_id="experiment-71",
    )
    replayed = replay_finite_arbitrary_plane_render(artifact, template, annotation, support)

    verify_finite_arbitrary_plane_render(artifact, support)
    assert artifact["schema_version"] == FINITE_RENDER_SCHEMA
    assert artifact["finite_plane_render_id"] == replayed["finite_plane_render_id"]
    assert artifact["plane_realization_id"] == replayed["plane_realization_id"]
    assert all(np.array_equal(artifact["raster"][key], replayed["raster"][key]) for key in artifact["raster"])
    assert "synthetic_realization_id" not in artifact
    assert artifact["generator"]["learned_checkpoint_dependencies"] == []
    assert artifact["generator"]["previous_model_dependencies"] == []
    assert artifact["generator"]["pretrained_feature_dependencies"] == []
    assert artifact["provenance"]["scalar_source"]["decoded"]["index_order"] == "F"
    assert artifact["provenance"]["scalar_source"]["float_conversion"]["normalization"] == "none"
    assert artifact["provenance"]["scalar_source"]["float_conversion"]["dtype"] == "<f4"
    assert artifact["provenance"]["animal_id"] == "animal-7"
    json.dumps(finite_render_receipt(artifact), allow_nan=False)
    seeds = artifact["rejection_attempts"][artifact["accepted_attempt_index"]]["field_stream_seed_uint64"]
    assert all(isinstance(seed, str) and seed.startswith("0x") and len(seed) == 18 for seed in seeds.values())
    assert int(artifact["root_seed"], 16) > 2**53

    tampered = copy.deepcopy(artifact)
    tampered["raster"]["scalar"] = tampered["raster"]["scalar"].copy()
    tampered["raster"]["scalar"][0, 0] += 1.0
    with pytest.raises(ValueError, match="raster hashes"):
        verify_finite_arbitrary_plane_render(tampered, support)


def test_verifier_replays_geometry_from_authenticated_seed_not_rounded_pose():
    template, annotation, support = _volumes()
    artifact = _make(template, annotation, support, sample_index=22)
    geometry = artifact["geometry"]
    rounded_pose_replay = rendered._finite_plane_raster_geometry_trusted(
        np.asarray(geometry["normal_rp2_ap_dv_ml"], dtype=np.float64),
        float(geometry["signed_offset_um"]),
        float(geometry["roll_rad"]),
        support,
        tuple(geometry["output_shape_h_w"]),
        tuple(geometry["margin_u_v_um"]),
    )

    assert geometry != rounded_pose_replay
    assert geometry["signed_offset_um"] - rounded_pose_replay["signed_offset_um"] == np.spacing(
        np.float64(geometry["signed_offset_um"])
    )
    verify_finite_arbitrary_plane_render(artifact, support)

    changed = copy.deepcopy(artifact)
    changed["geometry"] = rounded_pose_replay
    accepted = changed["accepted_attempt_index"]
    changed["rejection_attempts"][accepted]["geometry_sha256"] = rounded_pose_replay[
        "geometry_sha256"
    ]
    changed["rejection_attempts_sha256"] = rendered._payload_sha256(
        changed["rejection_attempts"]
    )
    changed["finite_plane_geometry_sha256"] = rendered._payload_sha256(
        {
            "schema": "anatomy-tracker.finite-plane-geometry/v1",
            "plane_realization_id": changed["plane_realization_id"],
            "geometry_sha256": rounded_pose_replay["geometry_sha256"],
        }
    )
    effective_receipts = changed["rendered_artifacts_receipt"][
        "effective_sampling_array_receipts"
    ]
    for key in effective_receipts:
        effective_receipts[key] = rounded_pose_replay["array_receipts"][key]
    changed["rendered_artifacts_sha256"] = rendered._payload_sha256(
        changed["rendered_artifacts_receipt"]
    )
    changed["finite_plane_render_id"] = rendered._payload_sha256(
        rendered._finite_render_identity(changed)
    )
    changed["finite_render_receipt_sha256"] = rendered._payload_sha256(
        rendered._receipt_payload(changed)
    )

    with pytest.raises(ValueError, match="authenticated seed"):
        verify_finite_arbitrary_plane_render(changed, support)


def test_plane_and_finite_identifier_layers_have_distinct_dependencies():
    template, annotation, support = _volumes()
    first = _make(template, annotation, support, output_shape=(47, 53), margin_um=(5.0, 7.0))
    second = _make(
        template + 1,
        annotation,
        support,
        output_shape=(59, 61),
        margin_um=(11.0, 13.0),
        scalar_source_sha256="6" * 64,
    )
    other_sample = _make(template, annotation, support, sample_index=30)

    assert first["accepted_attempt_index"] == second["accepted_attempt_index"] == 0
    assert first["plane_realization_id"] == second["plane_realization_id"]
    assert first["finite_plane_geometry_sha256"] != second["finite_plane_geometry_sha256"]
    assert first["rendered_artifacts_sha256"] != second["rendered_artifacts_sha256"]
    assert first["finite_plane_render_id"] != second["finite_plane_render_id"]
    assert first["plane_realization_id"] != other_sample["plane_realization_id"]


def test_support_is_verified_once_per_make_and_attempt_cross_links_are_enforced(monkeypatch):
    template, annotation, support = _volumes()
    calls = 0
    original = rendered.verify_annotation_support_index

    def counted(index):
        nonlocal calls
        calls += 1
        return original(index)

    monkeypatch.setattr(rendered, "verify_annotation_support_index", counted)
    artifact = _make(template, annotation, support)
    assert calls == 1

    changed = copy.deepcopy(artifact)
    accepted = changed["accepted_attempt_index"]
    changed["rejection_attempts"][accepted]["brain_pixel_count"] += 1
    changed["rejection_attempts_sha256"] = rendered._payload_sha256(changed["rejection_attempts"])
    changed["finite_plane_render_id"] = rendered._payload_sha256(rendered._finite_render_identity(changed))
    changed["finite_render_receipt_sha256"] = rendered._payload_sha256(
        rendered._receipt_payload(changed)
    )
    with pytest.raises(ValueError, match="accepted attempt"):
        verify_finite_arbitrary_plane_render(changed, support)


def test_development_only_shape_finiteness_and_marginal_support_contracts():
    template, annotation, support = _volumes()
    with pytest.raises(ValueError, match="train or development"):
        _make(template, annotation, support, split="test")
    with pytest.raises(ValueError, match="H,W > 1"):
        _make(template, annotation, support, output_shape=(1, 31))
    bad = template.astype(np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="numeric 3-D"):
        _make(bad, annotation, support)
    marginal = _make(
        template,
        annotation,
        support,
        output_shape=(7, 7),
        minimum_brain_pixels=50,
    )
    contract = marginal["acceptance_contract"]
    assert marginal["accepted_attempt_index"] == 0
    assert len(marginal["rejection_attempts"]) == 1
    assert contract["pose_redrawn_for_raster_support"] is False
    assert contract["pose_draw_count"] == 1
    assert contract["raster_support_meets_requested_identifiability_threshold"] is False
    assert "unconditioned" in marginal["sampling_measure"]["conditioning"]
    verify_finite_arbitrary_plane_render(marginal, support)


def test_reference_and_boundary_stress_strata_are_named_and_stay_in_component_union():
    template, annotation, support = _volumes()
    reference = _make(template, annotation, support, stratum=REFERENCE_STRATUM)
    stress = _make(
        template,
        annotation,
        support,
        stratum=BOUNDARY_STRESS_STRATUM,
        boundary_stress_fraction=0.1,
    )
    for artifact in (reference, stress):
        attempt = artifact["rejection_attempts"][artifact["accepted_attempt_index"]]
        offset = attempt["signed_offset_um"]
        intervals = np.asarray(attempt["stratum_sampling_interval_union_um"])
        assert np.any((offset >= intervals[:, 0]) & (offset <= intervals[:, 1]))
    assert reference["stratum"] == "reference"
    assert stress["stratum"] == "boundary-stress"


def test_raster_support_threshold_cannot_condition_or_redraw_pose():
    template, annotation, support = _volumes()
    permissive = _make(
        template,
        annotation,
        support,
        output_shape=(7, 7),
        minimum_brain_pixels=1,
    )
    censored = _make(
        template,
        annotation,
        support,
        output_shape=(7, 7),
        minimum_brain_pixels=10_000,
    )
    assert permissive["geometry"]["geometry_sha256"] == censored["geometry"][
        "geometry_sha256"
    ]
    assert permissive["raster_hashes"] == censored["raster_hashes"]
    assert permissive["rejection_attempts"][0]["normal_rp2_ap_dv_ml"] == censored[
        "rejection_attempts"
    ][0]["normal_rp2_ap_dv_ml"]
    assert permissive["rejection_attempts"][0]["signed_offset_um"] == censored[
        "rejection_attempts"
    ][0]["signed_offset_um"]
    assert permissive["rejection_attempts"][0]["roll_rad"] == censored[
        "rejection_attempts"
    ][0]["roll_rad"]
    assert permissive["acceptance_contract"][
        "raster_support_meets_requested_identifiability_threshold"
    ] is True
    assert censored["acceptance_contract"][
        "raster_support_meets_requested_identifiability_threshold"
    ] is False


def test_prepared_context_hashes_atlas_once_then_supports_random_access_and_cached_replay(monkeypatch):
    template, annotation, support = _volumes()
    support_verifications = 0
    atlas_hashes = 0
    original_verify = rendered.verify_annotation_support_index
    original_hash = rendered._array_sha256

    def counted_verify(index):
        nonlocal support_verifications
        support_verifications += 1
        return original_verify(index)

    def counted_hash(array):
        nonlocal atlas_hashes
        if np.asarray(array).shape == template.shape:
            atlas_hashes += 1
        return original_hash(array)

    monkeypatch.setattr(rendered, "verify_annotation_support_index", counted_verify)
    monkeypatch.setattr(rendered, "_array_sha256", counted_hash)
    context = prepare_finite_render_context(
        template,
        annotation,
        support,
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="4" * 64,
        template_decoder="pynrrd 1.1.3",
        template_index_order="F",
        annotation_decoder="pynrrd 1.1.3",
        annotation_index_order="F",
    )
    prepared_counts = (support_verifications, atlas_hashes)
    first = make_finite_arbitrary_plane_render_from_context(
        context, "development", 991, (41, 45), sample_index=3, margin_um=5.0
    )
    second = make_finite_arbitrary_plane_render_from_context(
        context, "development", 991, (41, 45), sample_index=4, margin_um=5.0
    )
    replayed = replay_finite_arbitrary_plane_render_from_context(first, context)

    assert prepared_counts == (1, 3)
    assert (support_verifications, atlas_hashes) == prepared_counts
    assert first["plane_realization_id"] != second["plane_realization_id"]
    assert replayed["finite_plane_render_id"] == first["finite_plane_render_id"]


def test_prepared_context_owns_assets_and_rejects_context_mutation():
    template, annotation, support = _volumes()
    context = prepare_finite_render_context(
        template,
        annotation,
        support,
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="4" * 64,
    )
    first = make_finite_arbitrary_plane_render_from_context(
        context, "train", 119, (37, 39), sample_index=2
    )
    template[:] = 0
    annotation[:] = 0
    support["component_hull_indices"][0][0, 0] += 1
    repeated = make_finite_arbitrary_plane_render_from_context(
        context, "train", 119, (37, 39), sample_index=2
    )

    assert repeated["finite_plane_render_id"] == first["finite_plane_render_id"]
    assert np.array_equal(repeated["raster"]["annotation"], first["raster"]["annotation"])
    with pytest.raises(TypeError):
        context["scalar_tensor"] = context["scalar_tensor"].clone()
    with pytest.raises(ValueError):
        context["support_index"]["component_hull_indices"][0][0, 0] += 1

    context["scalar_tensor"].add_(1.0)
    with pytest.raises(ValueError, match="tensor identity or version"):
        make_finite_arbitrary_plane_render_from_context(
            context, "train", 119, (37, 39), sample_index=2
        )


def test_model_independence_mutations_are_rejected_even_with_rehashed_receipt():
    artifact = _make()
    _, _, support = _volumes()
    for field in (
        "learned_checkpoint_dependencies",
        "previous_model_dependencies",
        "pretrained_feature_dependencies",
    ):
        changed = copy.deepcopy(artifact)
        changed["generator"][field] = ["forbidden.ckpt"]
        changed["finite_render_receipt_sha256"] = rendered._payload_sha256(
            rendered._receipt_payload(changed)
        )
        with pytest.raises(ValueError, match="model-dependency"):
            verify_finite_arbitrary_plane_render(changed, support)

    changed = copy.deepcopy(artifact)
    changed["generator"]["initialization"] = "warm-start previous model"
    changed["finite_render_receipt_sha256"] = rendered._payload_sha256(
        rendered._receipt_payload(changed)
    )
    with pytest.raises(ValueError, match="random-only"):
        verify_finite_arbitrary_plane_render(changed, support)


def test_effective_float32_grid_controls_near_half_index_label_sampling():
    ideal_origin_ap = 1.49999999
    geometry = {
        "renderer_center_ap_dv_ml": [ideal_origin_ap + 0.1, 1.1, 1.0],
        "renderer_frame_ap_dv_ml": np.eye(3).tolist(),
        "renderer_inplane_basis": [[0.2, 0.0], [0.0, 0.2]],
        "output_shape_h_w": [2, 2],
    }
    annotation = np.zeros((4, 4, 4), dtype=np.uint32)
    annotation[1, 1, 1] = 11
    annotation[2, 1, 1] = 22
    arrays = effective_renderer_sampling_arrays(geometry, annotation.shape)
    raster = render_finite_arbitrary_plane(np.zeros_like(annotation, dtype=np.float32), annotation, geometry)

    assert np.rint(ideal_origin_ap) == 1
    assert arrays["coordinate_raster_allen_index_float32"][0, 0, 0] == np.float32(1.5)
    assert raster["annotation"][0, 0] == 22
    assert raster["annotation"].dtype == np.int64


def test_rendered_artifact_and_json_receipt_hashes_bind_effective_grid_and_outputs():
    artifact = _make()
    receipt = finite_render_receipt(artifact)
    expected_rendered = artifact["rendered_artifacts_receipt"]

    assert set(expected_rendered["effective_sampling_array_receipts"]) == {
        "effective_coordinate_raster_allen_index_float32",
        "effective_normalized_interpolation_grid_xyz_float32",
        "valid_atlas_label_sampling_mask",
    }
    assert artifact["rendered_artifacts_sha256"] == rendered._payload_sha256(expected_rendered)
    assert artifact["finite_render_receipt_sha256"] == rendered._payload_sha256(
        {key: value for key, value in receipt.items() if key != "finite_render_receipt_sha256"}
    )
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize(
    ("value_name", "receipt_name", "dtype"),
    (
        (
            "effective_allen_index_ouv_ap_dv_ml",
            "effective_allen_index_ouv_ap_dv_ml_float32",
            np.float32,
        ),
        (
            "effective_physical_ouv_ap_dv_ml_um",
            "effective_physical_ouv_ap_dv_ml_um_from_float32_state",
            np.float64,
        ),
        (
            "effective_quicknii_ouv_ml_ap_dv",
            "effective_quicknii_ouv_ml_ap_dv_float32",
            np.float32,
        ),
    ),
)
def test_coherently_rehashed_effective_ouv_tampering_is_rejected(value_name, receipt_name, dtype):
    artifact = _make()
    _, _, support = _volumes()
    changed = copy.deepcopy(artifact)
    changed["geometry"][value_name][0] += 1.0
    array = np.asarray(changed["geometry"][value_name], dtype=dtype)
    changed["geometry"]["array_receipts"][receipt_name] = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "array_sha256": rendered._array_sha256(array),
    }
    geometry_payload = {
        key: value for key, value in changed["geometry"].items() if key != "geometry_sha256"
    }
    changed["geometry"]["geometry_sha256"] = rendered._payload_sha256(geometry_payload)
    accepted = changed["accepted_attempt_index"]
    changed["rejection_attempts"][accepted]["geometry_sha256"] = changed["geometry"][
        "geometry_sha256"
    ]
    changed["rejection_attempts_sha256"] = rendered._payload_sha256(changed["rejection_attempts"])
    changed["finite_plane_geometry_sha256"] = rendered._payload_sha256(
        {
            "schema": "anatomy-tracker.finite-plane-geometry/v1",
            "plane_realization_id": changed["plane_realization_id"],
            "geometry_sha256": changed["geometry"]["geometry_sha256"],
        }
    )
    changed["finite_plane_render_id"] = rendered._payload_sha256(
        rendered._finite_render_identity(changed)
    )
    changed["finite_render_receipt_sha256"] = rendered._payload_sha256(
        rendered._receipt_payload(changed)
    )

    with pytest.raises(ValueError, match="geometry does not replay|effective O/U/V"):
        verify_finite_arbitrary_plane_render(changed, support)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("annotation_raw", "annotation atlas/source"),
        ("annotation_decoded", "decoded annotation"),
        ("scalar_raw", "prepared asset provenance"),
    ),
)
def test_coherently_rehashed_provenance_contradictions_are_rejected(mutation, message):
    artifact = _make()
    _, _, support = _volumes()
    changed = copy.deepcopy(artifact)
    config = changed["generator"]["resolved_config"]
    if mutation == "annotation_raw":
        changed["provenance"]["annotation_source"]["source_sha256"] = "9" * 64
    elif mutation == "annotation_decoded":
        changed["provenance"]["annotation_decoded"]["array_sha256"] = "9" * 64
        config["annotation_array_sha256"] = "9" * 64
    else:
        changed["provenance"]["scalar_source"]["source_sha256"] = "9" * 64
        config["scalar_source_sha256"] = "9" * 64
    changed["provenance_sha256"] = rendered._payload_sha256(changed["provenance"])
    changed["generator"]["resolved_config_sha256"] = rendered._payload_sha256(config)
    changed["finite_plane_render_id"] = rendered._payload_sha256(
        rendered._finite_render_identity(changed)
    )
    changed["finite_render_receipt_sha256"] = rendered._payload_sha256(
        rendered._receipt_payload(changed)
    )

    with pytest.raises(ValueError, match=message):
        verify_finite_arbitrary_plane_render(changed, support)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("algorithm", "generator algorithm"),
        ("source_path", "implementation metadata"),
        ("geometry_contract", "implementation metadata"),
    ),
)
def test_coherently_rehashed_loaded_implementation_contradictions_are_rejected(mutation, message):
    artifact = _make()
    _, _, support = _volumes()
    changed = copy.deepcopy(artifact)
    implementation = changed["generator"]["implementation"]
    if mutation == "algorithm":
        changed["generator_algorithm"] = "not-the-renderer"
        changed["generator"]["resolved_config"]["generator_algorithm"] = "not-the-renderer"
        changed["generator"]["resolved_config_sha256"] = rendered._payload_sha256(
            changed["generator"]["resolved_config"]
        )
    elif mutation == "source_path":
        implementation["source_path"] = "training/elsewhere.py"
    else:
        implementation["dependency_contract_versions"]["geometry"] = "bogus"
    if mutation != "algorithm":
        implementation["implementation_sha256"] = rendered._payload_sha256(
            {key: value for key, value in implementation.items() if key != "implementation_sha256"}
        )
    changed["finite_plane_render_id"] = rendered._payload_sha256(
        rendered._finite_render_identity(changed)
    )
    changed["finite_render_receipt_sha256"] = rendered._payload_sha256(
        rendered._receipt_payload(changed)
    )

    with pytest.raises(ValueError, match=message):
        verify_finite_arbitrary_plane_render(changed, support)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (("sampling_measure", "sampling-measure"), ("acceptance_predicate", "acceptance predicate")),
)
def test_coherently_rehashed_algorithm_claim_contradictions_are_rejected(mutation, message):
    artifact = _make()
    _, _, support = _volumes()
    changed = copy.deepcopy(artifact)
    if mutation == "sampling_measure":
        changed["sampling_measure"]["orientation"] = "biased axis sampling"
        changed["generator"]["resolved_config"]["sampling_measure"]["orientation"] = (
            "biased axis sampling"
        )
        changed["generator"]["resolved_config_sha256"] = rendered._payload_sha256(
            changed["generator"]["resolved_config"]
        )
        changed["finite_plane_render_id"] = rendered._payload_sha256(
            rendered._finite_render_identity(changed)
        )
    else:
        changed["acceptance_contract"]["predicate"] = "always accept"
    changed["finite_render_receipt_sha256"] = rendered._payload_sha256(
        rendered._receipt_payload(changed)
    )

    with pytest.raises(ValueError, match=message):
        verify_finite_arbitrary_plane_render(changed, support)
